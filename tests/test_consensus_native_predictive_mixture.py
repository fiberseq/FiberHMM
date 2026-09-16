import copy
import itertools
import math

import numpy as np
import pytest

from fiberhmm.inference.consensus.native_predictive_mixture import (
    NativeMixtureComparison,compare_native_mixtures,interval_mixture)
from fiberhmm.inference.consensus.native_resolution import compare_interval_states


def unit(positions,hits,pa=.8,pp=.1):
    return dict(positions=list(positions),hits=list(hits),
        p_accessible=np.broadcast_to(pa,(len(positions),)).tolist(),
        p_protected=np.broadcast_to(pp,(len(positions),)).tolist())


def model(masks,weights):
    return dict(protected_masks=np.asarray(masks,bool),weights=weights)


def brute(u,m):
    patterns=list(itertools.product((0,1),repeat=len(u['positions'])))
    probabilities=[]
    for pattern in patterns:
        probabilities.append(math.fsum(w*math.prod(
            (pp if inside else pa) if hit else (1-(pp if inside else pa))
            for hit,inside,pa,pp in zip(pattern,mask,u['p_accessible'],u['p_protected']))
            for mask,w in zip(m['protected_masks'],m['weights'])))
    return patterns,np.asarray(probabilities)


def test_full_mixture_exact_reference_matches_independent_brute_force():
    u=unit([10,12,14,16],[0,1,0,1],pa=[.7,.8,.9,.6],pp=[.1,.04,.12,.05])
    a=model([[1,1,0,1],[0,1,1,1]],[.25,.75]);b=model([[1,1,1,1],[0,1,0,1]],[.5,.5])
    patterns,pa=brute(u,a);_,pb=brute(u,b)
    result=compare_native_mixtures(u,a,b,include_outcome_table=True)
    i=patterns.index(tuple(u['hits']))
    assert result['log_lr_b_over_a']==pytest.approx(math.log(pb[i]/pa[i]))
    assert result['observed_log_likelihood_a']==pytest.approx(math.log(pa[i]))
    assert result['observed_log_likelihood_b']==pytest.approx(math.log(pb[i]))
    assert result['exact']['total_variation']==pytest.approx(.5*np.abs(pa-pb).sum())
    assert result['exact']['outcomes_enumerated']==4  # common protected columns cancel


def test_same_mixture_is_exact_identity_even_above_outcome_budget():
    u=unit(list(range(20)),[0]*20)
    a=model([[0]*20,[1]*20],[.25,.75])
    result=compare_native_mixtures(u,a,copy.deepcopy(a),max_exact_outcomes=1,monte_carlo_samples=1000)
    assert result['log_lr_b_over_a']==0
    assert result['exact']['total_variation']==0
    assert result['exact']['optimal_equal_prior_accuracy']==.5
    assert result['algebraically_identical_induced_mixtures'] is True
    assert result['monte_carlo']['samples']==0


def test_different_mixing_same_induced_distribution_is_identity():
    u=unit([10,12],[1,0],pa=[.4,.8],pp=[.4,.1])
    a=model([[0,1],[1,1]],[.25,.75]);b=model([[0,1],[1,1]],[.75,.25])
    r=compare_native_mixtures(u,a,b)
    assert r['log_lr_b_over_a']==0
    assert r['exact']['total_variation']==0
    assert r['mixture_a']['unique_induced_masks']==1


def test_equal_marginals_do_not_erase_different_latent_correlations():
    u=unit([10,12],[0,0])
    a=model([[0,0],[1,1]],[.5,.5]);b=model([[0,1],[1,0]],[.5,.5])
    r=compare_native_mixtures(u,a,b)
    assert r['variable_informative_opportunities']==2
    assert r['exact']['total_variation']>0.4
    assert r['log_lr_b_over_a']<0


def test_single_state_matches_interval_resolution_including_unions():
    u=unit([1,4,7,10,13],[1,0,1,0,1],pa=[.6,.8,.7,.91,.42],pp=[.03,.09,.12,.02,.06])
    a=[[2,11]];b=[[2,5],[9,14]]
    reference=compare_interval_states(u,a,b)
    r=compare_native_mixtures(u,interval_mixture(u['positions'],[a],weights=[1.]),
                              interval_mixture(u['positions'],[b],weights=[1.]))
    assert r['log_lr_b_over_a']==pytest.approx(reference['log_lr_b_over_a'])
    for field in ('total_variation','optimal_equal_prior_accuracy'):
        assert r['exact'][field]==pytest.approx(reference['exact'][field])
    assert r['exact']['KL_a_vs_b_nats']==pytest.approx(reference['KL_a_vs_b_nats'])


def test_component_permutation_and_duplicate_mask_quotient_preserve_distribution():
    u=unit([10,12],[0,1])
    a=model([[0,0],[1,1]],[.25,.75]);aliases=model([[1,1],[0,0],[1,1]],[.25,.25,.5])
    b=model([[0,1],[1,0]],[.5,.5])
    first=compare_native_mixtures(u,a,b)
    second=compare_native_mixtures(u,aliases,b)
    assert first['log_lr_b_over_a']==pytest.approx(second['log_lr_b_over_a'])
    assert first['exact']['total_variation']==pytest.approx(second['exact']['total_variation'])
    assert second['mixture_a']['unique_induced_masks']==2
    permutation=model(aliases['protected_masks'][::-1],aliases['weights'][::-1])
    third=compare_native_mixtures(u,permutation,b)
    assert second['log_lr_b_over_a']==third['log_lr_b_over_a']


def test_model_swap_reverses_lr_and_preserves_tv():
    u=unit([10,12,14],[1,0,1])
    a=model([[0,0,1],[1,1,1]],[.5,.5]);b=model([[1,0,0],[0,1,0]],[.25,.75])
    x=compare_native_mixtures(u,a,b);y=compare_native_mixtures(u,b,a)
    assert x['log_lr_b_over_a']==pytest.approx(-y['log_lr_b_over_a'])
    assert x['exact']['total_variation']==pytest.approx(y['exact']['total_variation'])


def test_missing_is_not_an_observed_miss_and_reference_does_not_use_hits():
    empty=unit([],[]);a=interval_mixture([],[[[10,20]]],weights=[1.]);b=interval_mixture([],[[[20,30]]],weights=[1.])
    r=compare_native_mixtures(empty,a,b)
    assert r['log_lr_b_over_a']==0 and r['exact']['optimal_equal_prior_accuracy']==.5
    u=unit([10],[0]);a=model([[1]],[1.]);b=model([[0]],[1.])
    comparison=NativeMixtureComparison(u,a,b)
    assert comparison.observed([0])['log_lr_b_over_a']<0
    assert comparison.observed([1])['log_lr_b_over_a']>0
    opposite_hits=NativeMixtureComparison(unit([10],[1]),a,b)
    assert comparison.reference(monte_carlo_samples=1000,seed=4)==opposite_hits.reference(monte_carlo_samples=1000,seed=4)


def test_additional_actual_opportunity_can_resolve_latent_correlation():
    a=model([[0,0],[1,1]],[.5,.5]);b=model([[0,1],[1,0]],[.5,.5])
    one=compare_native_mixtures(unit([10],[0]),model(a['protected_masks'][:,:1],a['weights']),
                                model(b['protected_masks'][:,:1],b['weights']))
    both=compare_native_mixtures(unit([10,12],[0,0]),a,b)
    assert one['exact']['total_variation']==0
    assert both['exact']['total_variation']>.4


def test_seeded_bounded_mc_contains_exact_tv_and_keeps_all_components():
    u=unit([10,12,14],[0,0,1])
    a=model([[0,0,0],[1,1,1],[1,0,1]],[.25,.25,.5]);b=model([[1,1,0],[0,0,1]],[.75,.25])
    result=compare_native_mixtures(u,a,b,monte_carlo_samples=30000,seed=19,monte_carlo_confidence=.99)
    exact=result['exact']['total_variation'];mc=result['monte_carlo']
    assert mc['total_variation_interval'][0]<=exact<=mc['total_variation_interval'][1]
    assert mc['untruncated_half_width']==pytest.approx(math.sqrt(math.log(200)/(2*30000)))
    assert mc['samples']==30000
    repeat=compare_native_mixtures(u,a,b,monte_carlo_samples=30000,seed=19,monte_carlo_confidence=.99)
    assert result['monte_carlo']==repeat['monte_carlo']
    limited=compare_native_mixtures(u,a,b,max_exact_outcomes=4,monte_carlo_samples=30000,seed=19,monte_carlo_confidence=.99)
    assert limited['exact']['available'] is False
    assert limited['monte_carlo']==mc


def test_log_weight_tail_is_retained_in_observed_likelihood_not_culled():
    # A rare all-protected geometry dominates a sufficiently unlikely observed
    # all-miss pattern. Its prior weight alone must not be used for pruning.
    u=unit(list(range(20)),[0]*20,pa=.999999999,pp=.1)
    a=dict(protected_masks=np.array([[0]*20,[1]*20],bool),log_weights=np.array([0.,-100.]))
    b=model([[0]*20],[1.])
    r=compare_native_mixtures(u,a,b,max_exact_outcomes=1)
    expected=np.logaddexp(0.,-100.+20*math.log(.9/(1-.999999999)))
    assert r['observed_log_marginal_a_over_accessible']==pytest.approx(expected)
    assert r['log_lr_b_over_a'] < -300
    assert r['mixture_a']['positive_weight_geometries']==2
    assert r['exact']['available'] is False and r['monte_carlo']['available'] is False


def test_strong_internal_modifications_are_not_masked_by_common_outer_geometry():
    u=unit(list(range(10)),[0,0,1,1,1,1,1,1,0,0],pa=.8,pp=.01)
    a=model([[1]*10],[1.]);b=model([[1,1,0,0,0,0,0,0,1,1]],[1.])
    r=compare_native_mixtures(u,a,b)
    assert r['log_lr_b_over_a']==pytest.approx(6*math.log(80))
    assert r['variable_informative_opportunities']==6


def test_inputs_are_not_changed_and_preparation_copies_them():
    u=unit([10,12],[0,1]);a=model([[0,0],[1,1]],[.5,.5]);b=model([[1,0]],[1.])
    saved=copy.deepcopy(a)
    comparison=NativeMixtureComparison(u,a,b);before=comparison.observed()
    a['protected_masks'][:]=False;u['p_accessible'][0]=.2
    assert comparison.observed()==before
    assert saved['protected_masks'][1].all()


@pytest.mark.parametrize('bad',[
    dict(protected_masks=[[1]],weights=[.5]),dict(protected_masks=[[1]],weights=[-1.]),
    dict(protected_masks=[[1]],weights=[float('nan')]),dict(protected_masks=[[2]],weights=[1.]),
    dict(protected_masks=[[1.]],weights=[1.]),dict(protected_masks=[[1]],log_weights=[1.]),
    dict(protected_masks=[[1]],weights=[1.],log_weights=[0.]),
])
def test_invalid_mixtures_fail_closed(bad):
    with pytest.raises(ValueError):compare_native_mixtures(unit([10],[0]),bad,model([[0]],[1.]))
