import copy
import itertools
import math

import numpy as np
import pytest

from fiberhmm.inference.consensus.native_resolution import compare_interval_states


def unit(positions, hits, pa=.8, pp=.05, **extra):
    return dict(positions=list(positions), hits=list(hits),
        p_accessible=np.broadcast_to(pa, (len(positions),)).tolist(),
        p_protected=np.broadcast_to(pp, (len(positions),)).tolist(), **extra)


def brute_force(u, intervals_a, intervals_b):
    """Independent full-observation distribution, including unchanged positions."""
    p0=[];p1=[]
    for p,pa,pp in zip(u['positions'],u['p_accessible'],u['p_protected']):
        p0.append(pp if any(a<=p<b for a,b in intervals_a) else pa)
        p1.append(pp if any(a<=p<b for a,b in intervals_b) else pa)
    patterns=list(itertools.product((0,1),repeat=len(p0)))
    prob0=np.asarray([math.prod(p if h else 1-p for h,p in zip(bits,p0)) for bits in patterns])
    prob1=np.asarray([math.prod(p if h else 1-p for h,p in zip(bits,p1)) for bits in patterns])
    lr=np.log(prob1/prob0)
    observed=patterns.index(tuple(u['hits']))
    result=dict(log_lr=float(lr[observed]),kl01=float(sum(p*math.log(p/q) for p,q in zip(prob0,prob1))),
        kl10=float(sum(q*math.log(q/p) for p,q in zip(prob0,prob1))),
        tv=float(.5*sum(abs(p-q) for p,q in zip(prob0,prob1))),tests={})
    for level in (95.,99.,99.9):
        alpha=(100-level)/100
        ta=np.asarray([sum(p for p,v in zip(prob0,lr) if v>=x-1e-12) for x in lr])
        tb=np.asarray([sum(p for p,v in zip(prob1,lr) if v<=x+1e-12) for x in lr])
        result['tests'][level]=(float(prob0[ta<=alpha].sum()),float(prob1[ta<=alpha].sum()),
                               float(prob1[tb<=alpha].sum()),float(prob0[tb<=alpha].sum()))
    return result


def test_exact_diagnostic_matches_independent_full_joint_enumeration():
    u=unit([1,4,7,10,13],[1,0,1,0,1],pa=[.6,.8,.7,.91,.42],pp=[.03,.09,.12,.02,.06])
    a=[[2,11]];b=[[2,5],[9,14]]
    expected=brute_force(u,a,b)
    actual=compare_interval_states(u,a,b,include_outcome_table=True)
    assert actual['log_lr_b_over_a']==pytest.approx(expected['log_lr'])
    assert actual['KL_a_vs_b_nats']==pytest.approx(expected['kl01'])
    assert actual['KL_b_vs_a_nats']==pytest.approx(expected['kl10'])
    assert actual['exact']['total_variation']==pytest.approx(expected['tv'])
    assert actual['exact']['outcomes_enumerated']==4  # two differing positions, not all five
    for row in actual['exact']['tests']:
        values=tuple(row[k] for k in ('rejection_rate_under_a','power_under_b','rejection_rate_under_b','power_under_a'))
        assert values==pytest.approx(expected['tests'][row['confidence_percent']])
    assert sum(r['probability_a'] for r in actual['exact']['outcome_table'])==pytest.approx(1)


def test_swapping_models_reverses_evidence_not_information():
    u=unit([10,12,14],[1,0,1],pa=[.8,.7,.9],pp=[.02,.07,.1])
    a=[[9,15]];b=[[9,11],[13,15]]
    x=compare_interval_states(u,a,b);y=compare_interval_states(u,b,a)
    assert x['log_lr_b_over_a']==pytest.approx(-y['log_lr_b_over_a'])
    assert x['KL_a_vs_b_nats']==pytest.approx(y['KL_b_vs_a_nats'])
    assert x['exact']['total_variation']==pytest.approx(y['exact']['total_variation'])
    assert x['exact']['observed_upper_tail_under_a']==pytest.approx(y['exact']['observed_lower_tail_under_b'])
    for px,py in zip(x['exact']['tests'],y['exact']['tests']):
        assert px['power_under_b']==pytest.approx(py['power_under_a'])
        assert px['rejection_rate_under_a']==pytest.approx(py['rejection_rate_under_b'])


def test_genomic_reflection_and_reordered_observations_are_invariant():
    u=unit([1,4,7,10,13],[1,0,1,0,1],pa=[.6,.8,.7,.91,.42],pp=[.03,.09,.12,.02,.06])
    a=[[2,11]];b=[[2,5],[9,14]];length=20
    reflected=unit([length-1-p for p in reversed(u['positions'])],list(reversed(u['hits'])),
        pa=list(reversed(u['p_accessible'])),pp=list(reversed(u['p_protected'])))
    x=compare_interval_states(u,a,b)
    y=compare_interval_states(reflected,[[length-r,length-l] for l,r in a],[[length-r,length-l] for l,r in b])
    for key in ('log_lr_b_over_a','KL_a_vs_b_nats','KL_b_vs_a_nats'):
        assert x[key]==pytest.approx(y[key])
    assert x['exact']['total_variation']==pytest.approx(y['exact']['total_variation'])


def test_missing_opportunity_is_not_an_observed_miss_or_shared_support():
    a=[[10,20]];b=[[10,14],[16,20]]
    no=compare_interval_states(unit([11,18],[0,0]),a,b)
    miss=compare_interval_states(unit([11,14,18],[0,0,0]),a,b)
    assert no['status']=='identical_observed_distributions'
    assert no['likelihood_ratio_b_over_a']==1
    assert no['KL_a_vs_b_nats']==0
    assert no['exact']['optimal_equal_prior_accuracy']==.5
    assert no['unobserved_differing_base_intervals']==[[14,16]]
    assert no['supports_shared_identity'] is False
    assert miss['informative_positions']==[14]
    assert miss['informative_misses']==1
    assert miss['log_lr_b_over_a']<0
    assert miss['unobserved_differing_base_intervals']==[[15,16]]


def test_empty_unit_has_no_evidence():
    result=compare_interval_states(unit([],[]),[[10,20]],[[10,15]])
    assert result['exact']['outcomes_enumerated']==1
    assert result['log_lr_b_over_a']==0
    assert result['information_bounds']['total_variation_upper']==0


def test_identical_emissions_are_distinct_from_a_missing_observation():
    result=compare_interval_states(unit([10],[1],pa=.4,pp=.4),[[10,11]],[])
    assert result['changed_state_observations']==1
    assert result['changed_state_equal_emission_observations']==1
    assert result['changed_state_equal_emission_positions']==[10]
    assert result['unobserved_differing_bases']==0
    assert result['identical_distribution_reason']=='changed_states_have_identical_native_emissions'


def test_adding_observed_opportunities_restores_discrimination():
    x=compare_interval_states(unit([11],[0],pa=.9,pp=.1),[[10,20]],[])
    y=compare_interval_states(unit([11,13,15],[0,0,0],pa=.9,pp=.1),[[10,20]],[])
    assert y['KL_a_vs_b_nats']==pytest.approx(3*x['KL_a_vs_b_nats'])
    assert y['exact']['optimal_equal_prior_accuracy']>x['exact']['optimal_equal_prior_accuracy']


def test_strong_internal_modifications_cannot_be_masked_by_outer_agreement():
    u=unit([10,12,14,15,16,17,18,20,22],[0,0,1,1,1,1,1,0,0],pa=.8,pp=.01)
    out=compare_interval_states(u,[[9,23]],[[9,14],[19,23]])
    assert out['informative_hits']==5
    assert out['unchanged_state_observations']==4
    assert out['log_lr_b_over_a']>20
    assert out['exact']['observed_upper_tail_under_a']==pytest.approx(1e-10)
    assert out['observed_log_lr_contributions']==pytest.approx([math.log(80)]*5)


def test_exact_budget_does_not_truncate_information_and_bounds_contain_truth():
    u=unit([10,11,12,13],[0,1,0,1],pa=[.3,.4,.5,.6],pp=[.29,.38,.49,.58])
    exact=compare_interval_states(u,[[10,14]],[],max_exact_outcomes=16)
    bounded=compare_interval_states(u,[[10,14]],[],max_exact_outcomes=8)
    assert bounded['status']=='information_bounds_only'
    assert bounded['informative_observations']==4
    assert bounded['exact_required_outcomes']==16
    assert bounded['log_lr_b_over_a']==exact['log_lr_b_over_a']
    assert bounded['KL_a_vs_b_nats']==exact['KL_a_vs_b_nats']
    assert bounded['exact']['total_variation'] is None
    lo=bounded['information_bounds']['total_variation_lower'];hi=bounded['information_bounds']['total_variation_upper']
    assert lo<=exact['exact']['total_variation']<=hi
    for power, bound in zip(exact['exact']['tests'],bounded['exact']['tests']):
        assert bound['power_under_b'] is None
        assert power['power_under_b']<=bound['power_under_b_upper_bound']


def test_single_hit_does_not_violate_nonrandomized_test_size():
    out=compare_interval_states(unit([10],[1],pa=.6,pp=.1),[[10,11]],[])
    for row in out['exact']['tests']:
        assert row['power_under_b']==0
        assert row['rejection_rate_under_a']<=row['alpha']


def test_nearly_identical_emissions_have_nonnegative_stable_kl():
    out=compare_interval_states(unit([10],[0],pa=.5000000001,pp=.5),[[10,11]],[])
    assert out['KL_a_vs_b_nats']==pytest.approx(2e-20,rel=1e-5,abs=1e-30)
    tv=out['exact']['total_variation']
    assert out['information_bounds']['total_variation_lower']<=tv<=out['information_bounds']['total_variation_upper']


def test_interval_unions_no_double_count_and_inputs_are_immutable():
    u=unit([9,10,11,12,13,14],[0,0,1,0,1,1],aligned_blocks=[[9,15]])
    old=copy.deepcopy(u)
    out=compare_interval_states(u,[[10,12],[11,14]],[[10,11],[11,12]])
    simple=compare_interval_states(u,[[10,14]],[[10,12]])
    assert out==simple
    assert u==old
    assert out['protected_intervals_a']==[[10,14]]


def test_alignment_missingness_is_reported_separately():
    out=compare_interval_states(unit([10,12],[0,1],aligned_blocks=[[10,13]]),[[9,15]],[])
    assert out['differing_intervals_outside_alignment']==[[9,10],[13,15]]
    assert out['aligned_differing_bases_without_recorded_opportunity']==[[11,12]]


@pytest.mark.parametrize('change',[
    dict(positions=[1.,2.]), dict(positions=[2,1]), dict(positions=[1,1]),
    dict(hits=[0,.9]), dict(hits=[0,2]), dict(p_accessible=[.8,float('nan')]),
    dict(p_protected=[.1,0.]), dict(p_accessible=.8), dict(aligned_blocks=[[4,6]])])
def test_invalid_or_incompatible_native_input_is_rejected(change):
    u=unit([1,2],[0,1]);u.update(change)
    with pytest.raises(ValueError):compare_interval_states(u,[[1,3]],[])


@pytest.mark.parametrize('budget',[0,-1,True,2.5])
def test_invalid_exact_budget_rejected(budget):
    with pytest.raises(ValueError):compare_interval_states(unit([1],[0]),[[1,2]],[],max_exact_outcomes=budget)


def test_explicit_index_limit_preserves_all_information_without_enumeration():
    result=compare_interval_states(unit(list(range(64)),[0]*64),[[0,64]],[],max_exact_outcomes=2**65)
    assert result['exact']['available'] is False
    assert result['exact']['reason']=='exact_uint64_outcome_index_limit'
    assert result['informative_observations']==64
    assert result['log_lr_b_over_a']==pytest.approx(64*math.log(.2/.95))
    assert result['exact_required_log2_outcomes']==64
    assert result['exact']['outcomes_enumerated']==0


@pytest.mark.parametrize('level',[0,100,True,float('nan'),float('inf')])
def test_invalid_confidence_levels_rejected(level):
    with pytest.raises(ValueError):
        compare_interval_states(unit([1],[0]),[[1,2]],[],confidence_levels=[level])


def test_deterministic_random_interval_patterns_match_full_joint_references():
    rng=np.random.default_rng(57019)
    for _ in range(30):
        positions=np.arange(3,22,3).tolist()
        u=unit(positions,rng.integers(0,2,len(positions)).tolist(),
            pa=rng.uniform(.02,.98,len(positions)),pp=rng.uniform(.02,.98,len(positions)))
        a=[[p,p+1] for p in positions if rng.random()<.6]
        b=[[p,p+1] for p in positions if rng.random()<.6]
        expected=brute_force(u,a,b)
        actual=compare_interval_states(u,a,b)
        assert actual['log_lr_b_over_a']==pytest.approx(expected['log_lr'])
        assert actual['KL_a_vs_b_nats']==pytest.approx(expected['kl01'],abs=1e-12)
        assert actual['KL_b_vs_a_nats']==pytest.approx(expected['kl10'],abs=1e-12)
        assert actual['exact']['total_variation']==pytest.approx(expected['tv'],abs=1e-12)
        bounds=actual['information_bounds']
        assert bounds['total_variation_lower']-1e-12<=expected['tv']<=bounds['total_variation_upper']+1e-12
        for row in actual['exact']['tests']:
            values=tuple(row[k] for k in ('rejection_rate_under_a','power_under_b','rejection_rate_under_b','power_under_a'))
            assert values==pytest.approx(expected['tests'][row['confidence_percent']],abs=1e-12)
