"""Exact bounded work for the original, fully simulated predictive tail."""
import numpy as np
import pytest
from numba import njit

from fiberhmm.inference.consensus.measurement_distribution import (
    _monotone_projection_ranges, _predictive_exceedances)


@njit(cache=True)
def exhaustive(pa, pp, starts, ends, penalty, cdf, threshold, replicates, seed):
    """Frozen pre-optimization implementation, including all RNG operations."""
    np.random.seed(seed)
    count = 0
    hit = np.empty(len(pa)); miss = np.empty(len(pa))
    for j in range(len(pa)):
        hit[j] = np.log(pp[j]/pa[j])
        miss[j] = np.log1p(-pp[j])-np.log1p(-pa[j])
    prefix = np.zeros(len(pa)+1)
    for _ in range(replicates):
        g = np.searchsorted(cdf, np.random.random())
        for j in range(len(pa)):
            p = pp[j] if starts[g] <= j < ends[g] else pa[j]
            modified = np.random.random() < p
            prefix[j+1] = prefix[j]+(hit[j] if modified else miss[j])
        best = -np.inf; explained = -np.inf
        for h in range(len(starts)):
            value = prefix[ends[h]]-prefix[starts[h]]
            best = max(best, value)
            explained = max(explained, value+penalty[h])
        count += best-explained >= threshold-1e-10
    return count


@pytest.mark.parametrize('kind', ['full', 'constrained', 'invisible', 'holes', 'shuffled'])
def test_exact_randomized_tail_counts(kind):
    rng = np.random.default_rng(904)
    for k in (1, 3, 9, 25):
        a, b = np.triu_indices(k+1, 0 if kind == 'invisible' else 1)
        if kind == 'constrained':
            keep = (a <= k//2) & (b >= k//2+1)
            a, b = a[keep], b[keep]
        elif kind == 'holes':
            keep = rng.random(len(a)) > .4; keep[0] = True
            a, b = a[keep], b[keep]
        elif kind == 'shuffled':
            order = rng.permutation(len(a)); a, b = a[order], b[order]
        pa = rng.uniform(.15, .99, k); pp = rng.uniform(.001, .14, k)
        penalty = -rng.exponential(7, len(a)); penalty[0] = 0.
        penalty[1::5] = -np.inf
        weights = rng.random(len(a)); cdf = np.cumsum(weights/weights.sum()); cdf[-1] = 1.
        for threshold in (0., 1e-10, .01, .5, 3., 15., 100.):
            args = (pa, pp, a, b, penalty, cdf, threshold, 255, 792)
            assert _predictive_exceedances(*args) == exhaustive(*args)


def test_recognizer_refuses_holes_and_nonmonotone_ranges():
    for a, b in (([0, 0], [1, 3]), ([0, 1], [3, 2]), ([1, 0], [2, 2])):
        assert not _monotone_projection_ranges(np.array(a), np.array(b))[0]
    a, b = np.triu_indices(8, 0)
    ok, left, lower, upper = _monotone_projection_ranges(a, b)
    assert ok
    np.testing.assert_array_equal(left, np.arange(8))
    np.testing.assert_array_equal(lower, np.arange(8))
    np.testing.assert_array_equal(upper, np.full(8, 7))


def test_tail_equality_and_near_threshold_rounding():
    # Enumerate the possible losses of a tiny lattice, then test directly
    # on and immediately around each decision boundary in the same RNG stream.
    pa = np.array([.9, .35, .7]); pp = np.array([.1, .01, .05])
    a, b = np.triu_indices(4, 0)
    penalty = -np.arange(len(a))*.137; penalty[4] = -np.inf
    cdf = np.cumsum(np.full(len(a), 1/len(a))); cdf[-1] = 1.
    for pattern in range(8):
        hits = (pattern >> np.arange(3)) & 1
        values = np.where(hits, np.log(pp/pa), np.log1p(-pp)-np.log1p(-pa))
        prefix = np.r_[0., np.cumsum(values)]
        scores = prefix[b]-prefix[a]
        loss = float(scores.max()-(scores+penalty).max())
        for delta in (-2e-10, -1e-10, 0., 1e-10, 2e-10):
            args = (pa, pp, a, b, penalty, cdf, loss+delta, 1023, 123)
            assert _predictive_exceedances(*args) == exhaustive(*args)


def test_positive_penalties_and_extreme_finite_emissions_do_not_invalidate_bound():
    pa = np.array([1e-200, 1-1e-15, .5]); pp = np.array([.2, 1e-250, .49])
    a, b = np.triu_indices(4, 1)
    penalty = np.array([100., -1000., 0., -np.inf, 1e-13, -1e-13])
    cdf = np.cumsum(np.full(len(a), 1/len(a))); cdf[-1] = 1.
    for threshold in (-100., 0., 1e-10, 500.):
        args = (pa, pp, a, b, penalty, cdf, threshold, 1023, 81)
        assert _predictive_exceedances(*args) == exhaustive(*args)


@pytest.mark.parametrize('invalid', ['empty', 'nan_penalty', 'nan_threshold', 'reversed', 'overflow'])
def test_invalid_experiments_fail_instead_of_becoming_positive_tail_counts(invalid):
    pa = np.array([.9]); pp = np.array([.01]); a = np.array([0]); b = np.array([1])
    penalty = np.array([0.]); cdf = np.array([1.]); threshold = 1.
    if invalid == 'empty':
        a = a[:0]; b = b[:0]; penalty = penalty[:0]; cdf = cdf[:0]
    elif invalid == 'nan_penalty': penalty[0] = np.nan
    elif invalid == 'nan_threshold': threshold = np.nan
    elif invalid == 'reversed': a[0], b[0] = 1, 0
    else: pa[0], pp[0] = 1e-320, .5
    with pytest.raises(ValueError):
        _predictive_exceedances(pa, pp, a, b, penalty, cdf, threshold, 15, 1)


def test_duplicate_pairs_infinite_thresholds_and_penalties_match_frozen_count():
    pa = np.array([.9, .4]); pp = np.array([.02, .01])
    a = np.array([0, 0, 0, 1]); b = np.array([1, 1, 2, 2])
    cdf = np.array([.1, .3, .6, 1.])
    for penalty in (np.array([0., -.7, -np.inf, -3.]), np.full(4, -np.inf)):
        for threshold in (np.inf, -np.inf, 1e-10, 1e300):
            args = (pa, pp, a, b, penalty, cdf, threshold, 255, 87)
            assert _predictive_exceedances(*args) == exhaustive(*args)
