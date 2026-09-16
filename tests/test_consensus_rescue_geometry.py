import itertools
import numpy as np
import pytest
from scipy.special import logsumexp

from fiberhmm.inference.consensus.rescue_geometry import (fit_geometry_mixture, projected_typicality,
    population_predictive, arbitrary_gap_predictive, rescue_decision)


def test_uninformative_source_keeps_geometry_prior():
    q=np.array([.15,.35,.5])
    fitted,info=fit_geometry_mixture(np.tile(q,(12,1)),q)
    np.testing.assert_allclose(fitted,q,atol=1e-12)
    assert info['converged']


def test_competitor_geometry_prior_is_removed_not_treated_as_data():
    q=np.array([.3,.7]);prior=np.array([[.99,.01],[.01,.99],[.8,.2]])
    fitted,_=fit_geometry_mixture(prior,q,conditional_prior=prior)
    np.testing.assert_allclose(fitted,q,atol=1e-12)


def test_joint_edge_distribution_retains_modes_not_chimeras():
    # Four joint geometries: the data support only two. Independent start/end
    # marginals would invent the crossed edge combinations; this model must not.
    q=np.ones(4)/4
    post=np.tile([[.499,.001,.001,.499]],(100,1))
    fitted,_=fit_geometry_mixture(post,q)
    assert fitted[0]>.49 and fitted[3]>.49
    assert fitted[1]<.01 and fitted[2]<.01
    assert np.all(fitted>0)


def test_no_source_members_return_explicit_untrained_prior():
    q=np.array([.2,.8]);fitted,info=fit_geometry_mixture(np.empty((0,2)),q)
    np.testing.assert_array_equal(fitted,q)
    assert info['units']==0


def test_missing_opportunity_merges_geometry_not_arbitrary_bp_penalty():
    positions=np.array([0,10,20,30])
    # These have different bp lengths, but all protect exactly {10,20}.
    coords=np.array([[1,21],[8,29],[9,23]])
    r=projected_typicality(positions,coords,[.7,.2,.1],[4,26])
    assert r['projection_prior_mass']==pytest.approx(1.)
    assert r['hpd_mass_before_candidate']==0


def test_observable_tiny_patch_is_not_broad_family_geometry():
    positions=np.arange(0,41,2)
    coords=np.array([[10,30],[12,28],[10,16]])
    r=projected_typicality(positions,coords,[.7,.25,.05],[10,16])
    assert r['hpd_mass_before_candidate']==pytest.approx(.95)
    assert r['projection_prior_mass']==pytest.approx(.05)


def test_typicality_ties_do_not_depend_on_coordinate_order():
    p=np.arange(10);g=np.array([[0,3],[3,6],[6,9]])
    for c in g:
        r=projected_typicality(p,g,np.ones(3),c)
        assert r['hpd_mass_before_candidate']==0


def test_population_prediction_preserves_lost_testability_mass():
    r=population_predictive(np.array([7.,1.]),np.array([3,0]),np.array([1.,0.]),np.array([.05,.95]))
    assert r['population_log_bf']==pytest.approx(7.)
    assert r['physical_prior_mass']==pytest.approx(.05)
    assert r['three_opportunity_prior_mass']==pytest.approx(.05)


def test_prediction_matches_explicit_integer_sum():
    q=np.array([.2,.4,.4]);frac=np.array([.5,1.,.25]);llr=np.array([2.,-1.,4.])
    r=population_predictive(llr,np.array([3,4,2]),frac,q)
    expected=np.log(np.sum(q*frac*np.exp(llr))/np.sum(q*frac))
    assert r['population_log_bf']==pytest.approx(expected)
    assert r['three_opportunity_prior_mass']==pytest.approx(.5)
    informative=np.array([True,True,False])
    matched=np.log(np.sum((q*frac*np.exp(llr))[informative])/np.sum((q*frac)[informative]))
    assert r['observable_log_bf']==pytest.approx(matched)


def test_shape_comparison_matches_opportunity_conditioning_but_keeps_mass_gate():
    r=population_predictive(np.array([4.,0.]),np.array([3,0]),np.ones(2),np.array([.5,.5]))
    assert r['observable_log_bf']==pytest.approx(4.)
    assert r['population_log_bf']<r['observable_log_bf']
    assert r['three_opportunity_prior_mass']==pytest.approx(.5)
    no_info=population_predictive(np.zeros(2),np.array([2,0]),np.ones(2),np.array([.5,.5]))
    assert no_info['observable_log_bf'] is None
    assert no_info['population_log_bf']==pytest.approx(0.)


def test_arbitrary_patch_uses_whole_common_domain_and_normalizes_search():
    positions=np.arange(0,8);steps=np.array([-2.,-2.,2.,2.,2.,-2.,-2.,-2.])
    domains=[(0,8)];ll=[]
    for a,b in itertools.combinations(range(9),2):
        if b-a>=3:ll.append(steps[a:b].sum())
    actual=arbitrary_gap_predictive(positions,steps,(0,8),domains)
    assert actual==pytest.approx(logsumexp(ll)-np.log(len(ll)))
    assert arbitrary_gap_predictive(positions,np.zeros(8),(0,8),domains)==pytest.approx(0.)


def valid_record():
    return dict(source_geometry_units=30,updated_family_inclusion_mass=.96,
        physical_prior_mass=1.,three_opportunity_prior_mass=.95,
        in_population_projection_support=True,hpd_mass_before_candidate=.2,
        population_log_bf=5.,observable_log_bf=5.,arbitrary_gap_log_bf=5.2)


def test_high_local_score_cannot_override_mismatched_family_geometry():
    r=valid_record();r.update(updated_family_inclusion_mass=.999,hpd_mass_before_candidate=.97)
    decision=rescue_decision(r)
    assert not decision['accepted']
    assert 'atypical_joint_geometry_on_recipient_lattice' in decision['failures']


def test_patch_only_evidence_does_not_count_as_whole_family():
    r=valid_record();r.update(population_log_bf=-2.,observable_log_bf=-2.,arbitrary_gap_log_bf=6.)
    d=rescue_decision(r)
    assert not d['accepted']
    assert 'no_native_whole_family_protection_evidence' in d['failures']
    assert 'arbitrary_gap_fits_better_than_population_family' in d['failures']


def test_zero_family_information_not_rescued_by_exposure_conditioning():
    r=valid_record();r.update(three_opportunity_prior_mass=.05,population_log_bf=7.)
    assert not rescue_decision(r)['accepted']


def test_tiers_are_nested_for_fixed_evidence():
    rng=np.random.default_rng(4)
    for _ in range(100):
        r=valid_record();r.update(hpd_mass_before_candidate=rng.random(),population_log_bf=rng.normal(4,3),
                                  observable_log_bf=rng.normal(4,3),
                                  arbitrary_gap_log_bf=rng.normal(5,3))
        keep=[rescue_decision(r,credible_mass=c,loss_odds=o)['accepted']
              for c,o in [(.8,3),(.9,3),(.95,3),(.95,10)]]
        assert keep==sorted(keep)


def test_confident_new_patch_and_matching_family_pass():
    assert rescue_decision(valid_record())['accepted']


def test_native_reference_can_occupy_its_own_call_but_rescue_cannot():
    from fiberhmm.inference.consensus.recall import free_domains
    REGION=(1000,2000)
    a=REGION[0]
    u=dict(_region=REGION,representative_raw_tf_intervals=[[a+100,a+120]],unit_id='test_native_exposure',strand='CT',positions=[a+100,a+105,a+110],
        hits=[0,0,0],p_accessible=[.9]*3,p_protected=[.1]*3,
        native_multi_interval_tf_intervals=[[a+100,a+120]],raw_nuc_intervals=[[a+150,a+190]],
        aligned_blocks=[[a,a+200]],msp_intervals=[[a,a+200]])
    native=free_domains(u,[],block_native=False)
    recall=free_domains(u,[],block_native=True)
    assert any(lo<=a+100 and hi>=a+120 for lo,hi in native)
    assert not any(lo<a+120 and hi>a+100 for lo,hi in recall)
    assert all(not(lo<a+190 and hi>a+150) for lo,hi in native+recall)
    assert u['native_multi_interval_tf_intervals']==[[a+100,a+120]]
