"""Exact posterior-event reclassification; no refit or altered state prior."""
import math

import numpy as np
import pytest
from scipy.special import logsumexp

from fiberhmm.inference.footprint_pair_edge_tolerance import (
    prior_far_probabilities, score_tolerance_details, score_tolerances,
)
from fiberhmm.inference.molecule_footprint_pair import score_pairs


def explicit(a, b, radius):
    geometries = [(0, 0)] + [(s, e) for s in range(len(a)) for e in range(s + 1, len(a) + 1)]
    g = len(geometries)
    la = np.asarray([sum(a[s:e]) for s, e in geometries])
    lb = np.asarray([sum(b[s:e]) for s, e in geometries])
    logprior = np.full((g, g), math.log(.5 / (g * (g - 1))))
    np.fill_diagonal(logprior, math.log(.5 / g))
    logposterior = la[:, None] + lb[None, :] + logprior
    far = np.ones((g, g), dtype=bool)
    far[0, 0] = False
    for i in range(1, g):
        for j in range(1, g):
            far[i, j] = sum(abs(x - y) for x, y in zip(geometries[i], geometries[j])) > radius
    return float(np.exp(logsumexp(logposterior[far]) - logsumexp(logposterior)))


@pytest.mark.parametrize("seed", range(8))
def test_all_signed_boundary_offsets_equal_explicit_original_prior_enumeration(seed):
    rng = np.random.default_rng(seed)
    for k in range(1, 7):
        values = rng.normal(0, 3, (2, k))
        observed = rng.random((2, k)) > .3
        observed[:, 0] = True
        union = observed.any(axis=0)
        a = np.where(observed[0, union], values[0, union], 0.)
        b = np.where(observed[1, union], values[1, union], 0.)
        values[~observed] = np.nan
        result = score_tolerances(values, observed, [0], [1], [0], [k], (0, 1, 2, 3, 4, 12), use_numba=False)
        for r, p in result.items():
            assert p[0] == pytest.approx(explicit(a, b, r), abs=3e-11)


def test_zero_radius_is_original_kernel_bit_for_bit_and_radii_are_monotone():
    rng = np.random.default_rng(765)
    values = rng.normal(0, 3, (30, 50))
    observed = rng.random(values.shape) > .2
    ua, ub = rng.integers(0, 30, (2, 75))
    left = rng.integers(0, 25, 75)
    right = left + rng.integers(1, 26, 75)
    original = score_pairs(values, observed, ua, ub, left, right)
    result = score_tolerances(values, observed, ua, ub, left, right, (0, 1, 2, 4, 8))
    np.testing.assert_array_equal(result[0], original["posterior_distinct"])
    assert np.all(np.diff(np.stack(list(result.values())), axis=0) <= 0)


def test_fast_and_reference_backends_match_with_missing_opportunities():
    rng = np.random.default_rng(233)
    values = rng.normal(0, 6, (5, 18))
    observed = rng.random(values.shape) > .4
    values[~observed] = np.nan
    args = (values, observed, [0, 1, 2, 0], [3, 4, 1, 4], [0, 2, 4, 5], [18, 16, 17, 15])
    fast = score_tolerances(*args)
    slow = score_tolerances(*args, use_numba=False)
    for radius in fast:
        np.testing.assert_allclose(fast[radius], slow[radius], atol=1e-11, rtol=1e-11)


def test_large_tolerance_retains_empty_vs_nonempty_posterior_mass():
    a, b = np.array([2., -1., 3.]), np.array([-1., 2., -2.])
    values = np.vstack((a, b))
    result = score_tolerances(values, np.ones_like(values, bool), [0], [1], [0], [3], (0, 4, 100_000))
    expected = explicit(a, b, 100_000)
    assert expected > 0
    assert result[4][0] == pytest.approx(expected)
    assert result[100_000][0] == pytest.approx(expected)


def test_prior_event_volume_changes_but_the_original_prior_is_not_renormalized():
    for k in range(1, 8):
        values = np.zeros((2, k))
        details = score_tolerance_details(values, np.ones_like(values, bool), [0], [1], [0], [k], (0, 1, 2, 100), use_numba=False)
        assert not details["pair_discrimination_available"][0]
        for r, posterior in details["posterior_far"].items():
            assert posterior[0] == pytest.approx(details["prior_far"][r][0], abs=2e-13)
            assert posterior[0] == pytest.approx(explicit(np.zeros(k), np.zeros(k), r), abs=2e-13)
        assert details["posterior_far"][0][0] == .5
        assert details["prior_far"][100][0] == pytest.approx(1 / (1 + k * (k + 1) / 2))


def test_one_uninformative_side_is_flagged_not_interpreted_as_observed_matching():
    values = np.array([[0., 0., 0., 0.], [-3., 4., 4., -3.]])
    details = score_tolerance_details(values, np.ones_like(values, bool), [0], [1], [0], [4])
    assert not details["informative_a"][0]
    assert details["informative_b"][0]
    assert not details["pair_discrimination_available"][0]
    assert details["posterior_far"][0][0] == .5
    for r, p in details["posterior_far"].items():
        assert p[0] == pytest.approx(explicit(values[0], values[1], r), abs=2e-12)


def test_zero_opportunity_domain_retains_only_an_explicit_legacy_sentinel():
    result = score_tolerance_details(np.empty((2, 0)), np.empty((2, 0), bool), [0], [1], [0], [0])
    assert result["empty_domain"][0]
    for r, value in result["posterior_far"].items():
        assert value[0] == .5
        assert np.isnan(result["prior_far"][r][0])


def test_observed_zero_lr_counts_but_jointly_missing_grid_positions_do_not():
    values = np.array([[2., 0., -1., 3.], [1., 0., 2., -2.]])
    observed = np.ones_like(values, bool)
    all_positions = score_tolerance_details(values, observed, [0], [1], [0], [4])
    observed[:, 1] = False
    missing = score_tolerance_details(values, observed, [0], [1], [0], [4])
    compressed = score_tolerance_details(values[:, [0, 2, 3]], np.ones((2, 3), bool), [0], [1], [0], [3])
    assert all_positions["base_scores"]["n_union"][0] == 4
    assert missing["base_scores"]["n_union"][0] == 3
    for r in missing["posterior_far"]:
        assert missing["posterior_far"][r] == pytest.approx(compressed["posterior_far"][r])


def test_swapping_and_reflecting_molecules_preserves_tolerance_event():
    rng = np.random.default_rng(63)
    values = rng.normal(0, 2, (2, 14))
    observed = rng.random(values.shape) > .2
    original = score_tolerances(values, observed, [0], [1], [0], [14])
    swap = score_tolerances(values, observed, [1], [0], [0], [14])
    reflect = score_tolerances(values[:, ::-1], observed[:, ::-1], [0], [1], [0], [14])
    for r in original:
        assert original[r] == pytest.approx(swap[r], abs=1e-11)
        assert original[r] == pytest.approx(reflect[r], abs=1e-11)


def test_terminal_edge_generosity_does_not_erase_a_long_eight_hit_extension():
    pa, pp = .8, .05
    hit = np.array([[0] * 24, [0] * 16 + [1] * 8], bool)
    values = np.where(hit, math.log(pp / pa), math.log((1 - pp) / (1 - pa)))
    result = score_tolerances(values, np.ones_like(hit), [0], [1], [0], [24])
    assert result[0][0] > .999
    assert result[4][0] > .95


@pytest.mark.parametrize("bad", [(), (-1,), (1.5,), (True,)])
def test_invalid_tolerances_fail(bad):
    with pytest.raises(ValueError):
        score_tolerances(np.zeros((2, 1)), np.ones((2, 1), bool), [0], [1], [0], [1], bad)


def test_empty_batch_is_valid():
    result = score_tolerances(np.zeros((2, 3)), np.ones((2, 3), bool), [], [], [], [])
    assert all(v.shape == (0,) for v in result.values())
