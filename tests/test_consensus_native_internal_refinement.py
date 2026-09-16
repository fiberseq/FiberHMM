import itertools
import numpy as np
import pytest
from scipy.special import logsumexp

from fiberhmm.inference.consensus.native_internal_refinement import (
    internal_gap_scores, shared_gap_posterior, predictive_gap_log_ratio)


def obs(p, y, pa=.8, pp=.1):
    return dict(positions=np.array(p, int), hits=np.array(y, int),
                p_accessible=np.full(len(p), pa), p_protected=np.full(len(p), pp))


def brute(o, outer, logw, gaps, background=(), flank=1, existing=()):
    p = np.asarray(o['positions'])
    y, pa, pp = (np.asarray(o[k]) for k in ('hits', 'p_accessible', 'p_protected'))
    fixed = np.zeros(len(p), bool)
    for a, b in background: fixed |= (p >= a) & (p < b)
    values = np.where(y, np.log(pp/pa), np.log1p(-pp)-np.log1p(-pa))
    weights = np.asarray(logw)-logsumexp(logw)
    flat, fine = [], []
    for (a, b), w in zip(outer, weights):
        mask = (p >= a) & (p < b) & ~fixed
        for s, t in existing:
            if a+flank <= s and t+flank <= b: mask &= ~((p >= s) & (p < t))
        flat.append(w+values[mask].sum())
    for s, t in gaps:
        terms = []
        for (a, b), w in zip(outer, weights):
            mask = (p >= a) & (p < b) & ~fixed
            for u, v in existing:
                if a+flank <= u and v+flank <= b: mask &= ~((p >= u) & (p < v))
            if a+flank <= s and t+flank <= b: mask &= ~((p >= s) & (p < t))
            terms.append(w+values[mask].sum())
        fine.append(logsumexp(terms)-logsumexp(flat))
    return np.array(fine)


def test_exact_corner_sum_matches_every_small_integer_state_and_actual_outcome():
    outer = np.array(list(itertools.combinations(range(7), 2)))
    gaps = np.array(list(itertools.combinations(range(-1, 9), 2)))
    rng = np.random.default_rng(12)
    logw = rng.normal(size=len(outer))
    for bits in itertools.product((0, 1), repeat=4):
        o = obs([0, 2, 4, 6], bits)
        r = internal_gap_scores(o, outer, logw, gaps, protected_background=[(1, 3)])
        np.testing.assert_allclose(r['log_refined_vs_continuous'],
                                   brute(o, outer, logw, gaps, [(1, 3)]), atol=2e-13)


def test_unobserved_gap_is_exactly_neutral_not_an_imputed_miss():
    gaps = [[604, 606]]
    r = internal_gap_scores(obs([602, 603, 606], [0, 0, 0]), [[600, 609]], [0.], gaps)
    assert r['log_refined_vs_continuous'][0] == 0.
    assert r['gap_observed_opportunities'][0] == 0
    seen = internal_gap_scores(obs([602, 603, 604, 606], [0, 0, 0, 0]), [[600, 609]], [0.], gaps)
    assert seen['log_refined_vs_continuous'][0] < 0


def test_no_fitting_outer_geometry_stays_continuous_without_renormalization():
    r = internal_gap_scores(obs([4], [1]), [[0, 2], [4, 5]], np.log([.5, .5]), [[3, 5]])
    assert r['log_refined_vs_continuous'][0] == 0.
    assert np.isneginf(r['log_outer_prior_mass_admitting_gap'][0])


def test_tiny_outer_tail_does_not_become_full_support():
    o = obs([4], [1], pa=.8, pp=.01)
    g, w, h = [[0, 2], [0, 8]], np.log([1-1e-9, 1e-9]), [[3, 6]]
    r = internal_gap_scores(o, g, w, h)
    assert 0 < r['log_refined_vs_continuous'][0] < 1e-8
    assert r['log_outer_prior_mass_admitting_gap'][0] == pytest.approx(np.log(1e-9))
    assert r['log_refined_vs_continuous'][0] == pytest.approx(brute(o, g, w, h)[0], abs=1e-14)


def test_common_fixed_protection_is_not_erased_by_a_gap():
    o = obs([2, 3], [1, 1])
    r = internal_gap_scores(o, [[0, 6]], [0.], [[2, 4]], protected_background=[(1, 5)])
    assert r['log_refined_vs_continuous'][0] == 0.
    assert r['common_background_opportunities'] == 2


def test_duplicate_geometry_aliases_preserve_mass_not_evidence_count():
    o = obs([1, 3, 5], [0, 1, 0]); gaps = [[2, 4], [1, 3]]
    a = internal_gap_scores(o, [[0, 6], [1, 5]], np.log([.3, .7]), gaps)
    b = internal_gap_scores(o, [[0, 6], [0, 6], [1, 5]], np.log([.1, .2, .7]), gaps)
    np.testing.assert_allclose(a['log_refined_vs_continuous'], b['log_refined_vs_continuous'], atol=1e-14)


def test_gap_gain_is_not_full_pattern_adequacy_when_other_positions_contradict():
    o = obs(range(1, 20), np.ones(19), pa=.9, pp=.01)
    r = internal_gap_scores(o, [[0, 21]], [0.], [[9, 11]])
    assert r['log_refined_vs_continuous'][0] > 0
    assert r['log_refined_vs_common_background'][0] < -50


def test_hundreds_of_nats_and_tiny_tail_stay_finite_and_exact():
    o = obs(range(1, 90), np.zeros(89), pa=.9999, pp=.0001)
    outer, w, gaps = [[0, 91], [20, 70], [40, 41]], [0., -900., -1200.], [[1, 90], [30, 40], [0, 10]]
    r = internal_gap_scores(o, outer, w, gaps)
    assert np.all(np.isfinite(r['log_refined_vs_continuous']))
    np.testing.assert_allclose(r['log_refined_vs_continuous'], brute(o, outer, w, gaps), atol=1e-11)


def test_shared_posterior_and_predictive_score_are_normalized_and_group_checked():
    x = np.array([[1., -1., 0.], [2., 0., 0.]])
    r = shared_gap_posterior(x, evidence_groups=['a', 'b'])
    assert logsumexp(r['log_posterior']) == pytest.approx(0.)
    assert r['log_shared_gap_vs_continuous'] == pytest.approx(logsumexp(x.sum(0))-np.log(3))
    assert predictive_gap_log_ratio(np.zeros(3), r['log_posterior']) == pytest.approx(0.)
    with pytest.raises(ValueError, match='distinct evidence'):
        shared_gap_posterior(x, evidence_groups=['a', 'a'])


def test_second_gap_keeps_first_gap_and_entire_outer_law_exactly():
    outer = np.array(list(itertools.combinations(range(10), 2)))
    weights = np.linspace(-6., 0., len(outer))
    gaps, fixed = [[4, 5], [5, 7], [7, 8]], [[1, 3]]
    for bits in itertools.product((0, 1), repeat=5):
        o = obs([1, 2, 4, 6, 8], bits)
        r = internal_gap_scores(o, outer, weights, gaps, existing_gaps=fixed,
                                protected_background=[(5, 7)])
        np.testing.assert_allclose(r['log_refined_vs_current_pattern'],
            brute(o, outer, weights, gaps, [(5, 7)], existing=fixed), atol=1e-13)
        first = internal_gap_scores(o, outer, weights, fixed, protected_background=[(5, 7)])
        assert r['log_current_pattern_vs_continuous'] == pytest.approx(
            first['log_refined_vs_continuous'][0], abs=1e-13)
        np.testing.assert_allclose(r['log_refined_vs_continuous'],
            r['log_refined_vs_current_pattern']+r['log_current_pattern_vs_continuous'], atol=1e-13)


def test_two_separate_observed_features_do_not_have_to_compete_for_one_gap():
    o = obs([2, 5], [1, 1], pa=.9, pp=.01)
    first = internal_gap_scores(o, [[0, 8]], [0.], [[2, 3]])
    second = internal_gap_scores(o, [[0, 8]], [0.], [[5, 6]], existing_gaps=[[2, 3]])
    assert second['log_refined_vs_continuous'][0] == pytest.approx(2*first['log_refined_vs_continuous'][0])
    unseen = internal_gap_scores(obs([2], [1]), [[0, 8]], [0.], [[5, 6]], existing_gaps=[[2, 3]])
    assert unseen['log_refined_vs_current_pattern'][0] == 0.
    assert unseen['log_refined_vs_continuous'][0] > 0.


def test_cauchy_and_directional_two_gap_bounds_retain_entire_outer_law():
    o=obs([1, 2, 4, 6, 8], [0, 1, 1, 0, 1],pa=.93,pp=.03)
    outer=[[0,9],[2,8],[0,3],[7,9]]; lw=np.log([.3,.5,1e-100,.2])
    gaps=[[1,3],[5,8]]
    first=internal_gap_scores(o,outer,lw,gaps,moment_orders=[0.,1.,2.])
    pair=internal_gap_scores(o,outer,lw,[gaps[1]],existing_gaps=[gaps[0]])['log_refined_vs_continuous'][0]
    np.testing.assert_allclose(first['gap_log_moments_vs_current_pattern'][1],first['log_refined_vs_continuous'])
    assert pair <= .5*first['gap_log_moments_vs_current_pattern'][2].sum()+1e-12
    for i,j in ((0,1),(1,0)):
        assert pair <= first['log_refined_vs_continuous'][i]+first['gap_maximum_log_gain'][j]+1e-12


@pytest.mark.parametrize('fixed,candidate', [([[2, 4]], [[3, 5]]), ([[2, 4]], [[4, 5]]),
    ([[2, 4], [3, 6]], [[7, 8]])])
def test_overlapping_gap_evidence_is_never_counted_twice(fixed, candidate):
    with pytest.raises(ValueError, match='separated protected flanks'):
        internal_gap_scores(obs([2, 5], [1, 1]), [[0, 9]], [0.], candidate, existing_gaps=fixed)


@pytest.mark.parametrize('flank', [0, -1, 1.5])
def test_no_empty_or_unprotected_flank_configuration(flank):
    with pytest.raises(ValueError, match='flank'):
        internal_gap_scores(obs([2], [1]), [[0, 6]], [0.], [[2, 4]], minimum_flank_bp=flank)
