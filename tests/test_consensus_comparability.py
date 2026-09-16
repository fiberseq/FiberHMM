import numpy as np
import pytest
from scipy.special import expit

from fiberhmm.inference.consensus.comparability import (inclusion_log_bf,prevalence_posterior,quadrature,
    compare_strands,q_from_error,wilson_interval,sample_conditional_configurations,window_probability)
from fiberhmm.inference.consensus.lattice import RegionFamilyLattice


def test_inclusion_odds_remove_prior_not_just_logit_posterior():
    prior=np.array([.01,.2,.8,.99]);bf=np.array([-3.,0.,2.,4.])
    posterior=expit(np.log(prior)-np.log1p(-prior)+bf)
    actual,available,saturated=inclusion_log_bf(posterior,prior)
    np.testing.assert_allclose(actual,bf,atol=1e-11)
    assert available.all() and not saturated.any()


def test_prior_only_marginals_are_neutral_and_impossible_is_unavailable():
    prior=np.array([0.,.001,.99,1.])
    bf,available,saturated=inclusion_log_bf(prior,prior)
    np.testing.assert_array_equal(bf,np.zeros(4))
    np.testing.assert_array_equal(available,[False,True,True,False])
    assert not saturated.any()


def test_numerical_saturation_is_flagged_not_hidden():
    bf,available,saturated=inclusion_log_bf([0.,1.],[.5,.5])
    assert available.all() and saturated.all() and np.isfinite(bf).all()


def fits(ct,ga,size=384):
    grid,w=quadrature(size)
    a=prevalence_posterior(np.asarray(ct),grid,w);b=prevalence_posterior(np.asarray(ga),grid,w)
    return a,b,a['grid']


def test_blind_strand_does_not_get_existence_from_a_continuous_prior():
    a,b,g=fits(np.zeros(1000),np.zeros(1000))
    assert a['mass'][0]==pytest.approx(.5)
    assert a['log_marginal_bf']==pytest.approx(0.)
    c=compare_strands(a,b,g)
    assert c['comparative_Q_model'] is None
    assert c['native_existence_Q_model']=={'CT':None,'GA':None}
    assert c['population_quantification_mask']=={'CT':False,'GA':False}


def observation_factors(pi,sensitivity,false_positive,n,seed):
    rng=np.random.default_rng(seed)
    hit=rng.random(n)<(pi*sensitivity+(1-pi)*false_positive)
    return np.where(hit,np.log(sensitivity/false_positive),np.log((1-sensitivity)/(1-false_positive))),hit.mean()


def test_same_latent_frequency_with_unequal_detection_rates_is_concordant():
    ct,rate_ct=observation_factors(.3,.95,.01,4000,6)
    ga,rate_ga=observation_factors(.3,.35,.01,4000,12)
    assert rate_ct>2*rate_ga
    a,b,g=fits(ct,ga)
    result=compare_strands(a,b,g,equivalence_margin=.10)
    assert abs(a['mean']-.3)<.04 and abs(b['mean']-.3)<.04
    assert result['comparative_Q_model']>10
    assert result['status']=='two_strand_supported'


def test_informative_disagreement_is_not_excused_by_different_sensitivity():
    ct,_=observation_factors(.6,.9,.01,2000,1)
    ga,_=observation_factors(.1,.5,.01,2000,2)
    a,b,g=fits(ct,ga)
    r=compare_strands(a,b,g)
    assert r['status']=='strand_model_discrepancy'
    assert r['comparative_Q_model']<.1


def test_one_blind_strand_preserves_native_supported_side_without_cross_bonus():
    ct,_=observation_factors(.4,.9,.01,2000,1)
    a,b,g=fits(ct,np.zeros(2000))
    r=compare_strands(a,b,g)
    assert r['native_existence_Q_model']['CT']>=20
    assert r['native_existence_Q_model']['GA'] is None
    assert r['status']=='CT_population_only'
    assert r['comparative_Q_model'] is None


def test_two_strands_of_absence_do_not_receive_shared_footprint_support():
    a,b,g=fits(np.full(400,-4.),np.full(400,-4.))
    r=compare_strands(a,b,g)
    assert all(r['population_quantification_mask'].values())
    assert r['probability_within_margin']>.95
    assert r['comparative_Q_model']<.1
    assert r['status']=='no_recurrent_support_on_either_strand'


def test_more_depth_can_improve_population_precision_without_changing_unit_factors():
    ct,_=observation_factors(.25,.35,.10,200,9)
    a,b,g=fits(ct,np.tile(ct,10))
    assert b['width95']<a['width95']


def test_q_thresholds_and_equivalence_margins_are_nested():
    ct,_=observation_factors(.3,.8,.02,700,1)
    ga,_=observation_factors(.33,.7,.01,700,2)
    a,b,g=fits(ct,ga)
    scores=[compare_strands(a,b,g,equivalence_margin=d)['comparative_Q_model'] for d in [.05,.1,.2]]
    assert scores==sorted(scores)
    assert q_from_error(.1)==pytest.approx(10.)
    assert q_from_error(.01)==pytest.approx(20.)


def test_comparison_is_symmetric_between_strands():
    ct,_=observation_factors(.3,.8,.01,800,12)
    ga,_=observation_factors(.4,.6,.01,900,3)
    a,b,g=fits(ct,ga)
    x=compare_strands(a,b,g);y=compare_strands(b,a,g)
    assert x['probability_joint_support']==pytest.approx(y['probability_joint_support'],abs=1e-12)


def toy_sample(present,observed=True,draws=2000):
    kernel=RegionFamilyLattice(np.arange(7),np.array([[1,3],[1,2],[5,6]]),0)
    eta=np.array([.5,-.2,.7]);weights=eta[kernel.gf]+kernel.logq
    values,selected,possible=sample_conditional_configurations(kernel.offsets,kernel.dest,kernel.edge_geo,
        kernel.ga,kernel.gb,kernel.gf,weights,np.ones(len(weights),bool),kernel.n_nodes,kernel.f,0,present,
        np.full(7,.9),np.full(7,.1),np.full(7,observed,dtype=bool),draws,18)
    return kernel,eta,values,selected,possible


def test_conditional_sampler_obeys_presence_competition_and_exact_prior():
    k,eta,values,selected,possible=toy_sample(True)
    assert possible and np.all(selected[:,0]>=0) and np.all(selected[:,1]<0)
    # Disjoint third family remains an exact independent prior choice.
    assert np.mean(selected[:,2]>=0)==pytest.approx(expit(eta[2]),abs=.035)
    for gs in selected[:30]:
        intervals=sorted((k.ga[g],k.gb[g]) for g in gs if g>=0)
        assert all(a[1]<b[0] for a,b in zip(intervals,intervals[1:]))


def test_conditional_exclusion_keeps_other_competing_states():
    k,eta,values,selected,possible=toy_sample(False)
    assert possible and np.all(selected[:,0]<0)
    assert np.mean(selected[:,1]>=0)==pytest.approx(expit(eta[1]),abs=.035)


def test_blind_simulated_read_has_no_information_despite_latent_protection():
    k,eta,values,selected,possible=toy_sample(True,observed=False,draws=20)
    assert possible and np.all(selected[:,0]>=0)
    np.testing.assert_array_equal(values,np.zeros_like(values))
    p=k.evaluate(values,eta)['family_inclusion']
    p0=k.evaluate(np.zeros_like(values),eta)['family_inclusion']
    bf,_,_=inclusion_log_bf(p,p0)
    np.testing.assert_array_equal(bf,np.zeros_like(bf))


def test_conditional_negative_simulation_respects_likelihood_ratio_bound():
    k,eta,values,selected,possible=toy_sample(False,draws=1500)
    posterior=k.evaluate(values,eta)['family_inclusion'][:,0]
    prior=k.evaluate(np.zeros((1,k.k)),eta)['family_inclusion'][0,0]
    bf,_,_=inclusion_log_bf(posterior,prior)
    assert np.mean(bf>=np.log(20))<.075


def test_mc_uncertainty_is_not_ignored_in_power_mask():
    lo,hi=wilson_interval(32,32)
    assert .8<lo<1 and hi==pytest.approx(1.)
    lo,_=wilson_interval(27,32)
    assert lo<.8
    assert wilson_interval(0,0)==(None,None)




def test_marginal_roundoff_tolerance_matches_full_configuration_kernel():
    bf,available,saturated=inclusion_log_bf([1+5e-8],[.5])
    assert available[0] and saturated[0] and np.isfinite(bf[0])


def test_continuous_window_matches_analytic_spike_uniform_probability():
    a,b,g=fits(np.zeros(100),np.zeros(100),size=768)
    d=.1;floor=.01
    # .25 both-zero + .5*d atom/slab + .25*(2*d-d*d) slab/slab.
    assert window_probability(a['mass'],b['mass'],g,d)==pytest.approx(.25+d-.25*d*d,abs=3e-7)
    assert window_probability(a['mass'],b['mass'],g,d,floor)==pytest.approx(.25*(2*(1-floor)*d-d*d),abs=2e-7)


def test_near_floor_comparison_is_stable_when_quadrature_resolution_doubles():
    ct,_=observation_factors(.035,.9,.01,2000,1)
    ga,_=observation_factors(.015,.65,.01,2000,2)
    results=[]
    for size in [768,1536]:
        a,b,g=fits(ct,ga,size=size)
        results.append(compare_strands(a,b,g)['comparative_Q_model'])
    assert results[0]==pytest.approx(results[1],abs=.01)
