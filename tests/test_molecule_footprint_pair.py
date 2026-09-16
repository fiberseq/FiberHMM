"""Exact, pair-only, full-union native-emission interval comparisons."""
import math

import numpy as np
import pytest
from scipy.special import expit, logsumexp

from fiberhmm.inference.molecule_footprint_pair import compare_observations, score_pairs


def brute_force(a, b):
    logs_a, logs_b = [0.], [0.]
    for left in range(len(a)):
        for right in range(left + 1, len(a) + 1):
            logs_a.append(sum(a[left:right]))
            logs_b.append(sum(b[left:right]))
    aa, bb = np.array(logs_a), np.array(logs_b)
    same = logsumexp(aa + bb) - math.log(len(aa))
    joint = aa[:, None] + bb[None, :]
    separate = logsumexp(joint[~np.eye(len(aa), dtype=bool)]) - math.log(len(aa) * (len(aa)-1))
    return separate - same


@pytest.mark.parametrize("seed", range(8))
def test_linear_sums_match_explicit_all_interval_and_off_diagonal_models(seed):
    rng = np.random.default_rng(seed)
    for size in range(1, 5):
        values = rng.normal(0, 3, (2, size))
        observed = rng.random((2, size)) > .3
        union = observed.any(axis=0)
        values[~observed] = np.nan
        result = score_pairs(values, observed, [0], [1], [0], [size], use_numba=False)
        if not union.any():
            assert result['posterior_distinct'][0] == .5
            continue
        a = np.where(observed[0, union], values[0, union], 0)
        b = np.where(observed[1, union], values[1, union], 0)
        expected = brute_force(a, b)
        assert result['log_bayes_factor'][0] == pytest.approx(expected, abs=2e-10)
        assert result['posterior_distinct'][0] == pytest.approx(expit(expected), abs=2e-12)


def test_numba_and_reference_backends_agree():
    rng = np.random.default_rng(341)
    values = rng.normal(0, 4, (9, 30))
    observed = rng.random(values.shape) > .2
    values[~observed] = np.nan
    ua, ub = rng.integers(0, 9, (2, 31))
    left = rng.integers(0, 20, 31)
    right = left + rng.integers(0, 11, 31)
    fast = score_pairs(values, observed, ua, ub, left, right)
    slow = score_pairs(values, observed, ua, ub, left, right, use_numba=False)
    for key in fast:
        assert fast[key] == pytest.approx(slow[key], abs=1e-10)


def comparison(hits_a, hits_b, pa=.8, pp=.05):
    positions = np.arange(len(hits_a))
    return compare_observations(positions, hits_a, pa, pp, positions, hits_b, pa, pp,
                                0, len(positions), use_numba=False)


def test_internal_observations_distinguish_calls_with_identical_supplied_edges():
    hits_a = np.zeros(24, dtype=int)
    hits_b = np.r_[np.zeros(12, dtype=int), np.ones(12, dtype=int)]
    same = comparison(hits_a, hits_a)
    distinct = comparison(hits_a, hits_b)
    assert same['posterior_distinct'] < .5
    assert distinct['posterior_distinct'] > .999


def test_one_difference_is_not_automatically_a_class_but_eight_can_be_resolved():
    a = np.zeros(24, dtype=int)
    one = a.copy()
    one[-1] = 1
    eight = a.copy()
    eight[-8:] = 1
    weak = comparison(a, one)
    strong = comparison(a, eight)
    assert weak['posterior_distinct'] < .99
    assert strong['posterior_distinct'] > .999
    assert strong['log_bayes_factor'] > weak['log_bayes_factor']


def test_actual_emission_reliability_changes_inference():
    a = np.zeros(24, dtype=int)
    b = np.r_[np.zeros(16, dtype=int), np.ones(8, dtype=int)]
    strong = comparison(a, b, .8, .05)
    weak = comparison(a, b, .25, .20)
    null = comparison(a, b, .25, .25)
    assert strong['posterior_distinct'] > .999
    assert weak['posterior_distinct'] < .9
    assert null['posterior_distinct'] == .5
    assert null['n_union'] == 24


def test_disjoint_lattices_use_union_contiguity_not_empty_intersection():
    pos_a, pos_b = np.arange(0, 48, 2), np.arange(1, 48, 2)
    a = np.zeros(24, dtype=int)
    b = np.r_[np.zeros(12, dtype=int), np.ones(12, dtype=int)]
    result = compare_observations(pos_a, a, .8, .05, pos_b, b, .8, .05, 0, 48, use_numba=False)
    assert result['n_shared'] == 0
    assert result['n_union'] == 48
    assert result['n_a'] == result['n_b'] == 24
    assert result['posterior_distinct'] > .99


def test_missing_positions_are_not_unmodified_observations():
    values = np.array([[2., 2., 2., 2.], [-3., -3., np.nan, np.nan]])
    observed = np.isfinite(values)
    missing = score_pairs(values, observed, [0], [1], [0], [4], use_numba=False)
    unmodified = score_pairs(np.nan_to_num(values, nan=2.), np.ones_like(observed), [0], [1], [0], [4], use_numba=False)
    assert missing['n_b'][0] == 2
    assert unmodified['n_b'][0] == 4
    assert missing['log_bayes_factor'][0] != pytest.approx(unmodified['log_bayes_factor'][0])
    # A real observation with pa==pp retains its union opportunity even though
    # it supplies zero LR; a jointly unobserved column does not add a state.
    values[:, 2] = 0.
    masked = observed.copy()
    masked[:, 2] = False
    with_zero = score_pairs(values, np.isfinite(values), [0], [1], [0], [4], use_numba=False)
    without = score_pairs(values, masked, [0], [1], [0], [4], use_numba=False)
    assert with_zero['n_union'][0] == without['n_union'][0] + 1


def test_molecule_swap_reflection_and_unrelated_dataset_depth_do_not_change_score():
    rng = np.random.default_rng(124)
    values = rng.normal(0, 2, (2, 12))
    observed = rng.random(values.shape) > .3
    original = score_pairs(values, observed, [0], [1], [0], [12], use_numba=False)
    swapped = score_pairs(values, observed, [1], [0], [0], [12], use_numba=False)
    reflected = score_pairs(values[:, ::-1], observed[:, ::-1], [0], [1], [0], [12], use_numba=False)
    augmented = score_pairs(np.r_[values, rng.normal(0, 20, (1000, 12))],
                            np.r_[observed, np.ones((1000, 12), dtype=bool)], [0], [1], [0], [12], use_numba=False)
    for candidate in (swapped, reflected, augmented):
        assert candidate['posterior_distinct'] == pytest.approx(original['posterior_distinct'], abs=1e-12)
        assert candidate['log_bayes_factor'] == pytest.approx(original['log_bayes_factor'], abs=1e-12)


def test_empty_and_uninformative_observations_have_no_evidence():
    result = compare_observations([], [], .8, .05, [], [], .8, .05, 0, 12, use_numba=False)
    assert result['status'] == 'uninformative_empty_domain'
    assert result['posterior_distinct'] == .5
    values = np.array([[0., 0., 0.], [100., -20., 4.]])
    uninformative = score_pairs(values, np.ones_like(values, dtype=bool), [0], [1], [0], [3], use_numba=False)
    assert uninformative['log_bayes_factor'][0] == 0
    assert uninformative['posterior_distinct'][0] == .5


def test_half_open_bounds_and_scalar_batch_agreement():
    positions = np.arange(10, 20)
    ha = positions % 3 == 0
    hb = positions % 4 == 0
    pa = np.linspace(.4, .9, 10)
    pp = np.linspace(.03, .15, 10)
    scalar = compare_observations(positions, ha, pa, pp, positions, hb, pa, pp, 12, 18, use_numba=False)
    logs = np.stack([np.where(h, np.log(pp/pa), np.log((1-pp)/(1-pa))) for h in (ha, hb)])
    batch = score_pairs(logs, np.ones_like(logs, dtype=bool), [0], [1], [2], [8], use_numba=False)
    assert scalar['union_positions'] == list(range(12, 18))
    for key, value in batch.items():
        assert scalar[key] == pytest.approx(value[0], abs=1e-12)


def test_extreme_finite_likelihoods_are_stable():
    values = np.array([[1000., 1000., -1000.], [1000., -1000., -1000.]])
    result = score_pairs(values, np.ones_like(values, dtype=bool), [0, 0], [1, 0], [0, 0], [3, 3], use_numba=False)
    assert result['posterior_distinct'][0] == 1
    assert 0 <= result['posterior_distinct'][1] <= 1
    assert not np.isnan(result['log_bayes_factor']).any()


def test_invalid_inputs_are_not_silently_repaired():
    with pytest.raises(ValueError, match='strictly increasing'):
        compare_observations([1, 1], [0, 0], .8, .1, [], [], .8, .1, 0, 3)
    with pytest.raises(ValueError, match='between zero and one'):
        compare_observations([1], [0], 1., .1, [], [], .8, .1, 0, 3)
    with pytest.raises(ValueError, match='binary'):
        compare_observations([1], [.5], .8, .1, [], [], .8, .1, 0, 3)
    with pytest.raises(ValueError, match='finite'):
        score_pairs([[np.nan]], [[True]], [0], [0], [0], [1])
    with pytest.raises(ValueError, match='integer'):
        score_pairs([[1.]], [[True]], [0.5], [0], [0], [1])
    with pytest.raises(ValueError, match='Domain indices'):
        score_pairs([[1.]], [[True]], [0], [0], [0], [2])


def test_under_declared_generative_prior_model_posterior_is_calibrated_exactly():
    # Enumerate every 2-position observation pair and both latent model priors.
    # This verifies Bayesian calibration of the DECLARED model, not a claim
    # about caller-selected windows or misspecified emissions in real data.
    size = 2
    geometries = [np.zeros(size, dtype=bool)]
    for start in range(size):
        for end in range(start + 1, size + 1):
            state = np.zeros(size, dtype=bool)
            state[start:end] = True
            geometries.append(state)
    g = len(geometries)
    pa, pp = np.array([.7, .85]), np.array([.06, .13])
    joint_probability = 0.
    posterior_average = 0.
    for pattern in range(1 << (2 * size)):
        hits = np.array([(pattern >> bit) & 1 for bit in range(2 * size)]).reshape(2, size)
        likelihood = np.empty((2, g))
        for unit in range(2):
            for geometry, state in enumerate(geometries):
                theta = np.where(state, pp, pa)
                likelihood[unit, geometry] = np.prod(np.where(hits[unit], theta, 1-theta))
        same = np.sum(likelihood[0] * likelihood[1]) / g
        distinct = (np.sum(likelihood[0]) * np.sum(likelihood[1]) - np.sum(likelihood[0]*likelihood[1])) / (g*(g-1))
        mass = (same + distinct) / 2
        called = compare_observations(np.arange(size), hits[0], pa, pp, np.arange(size), hits[1], pa, pp, 0, size, use_numba=False)
        assert called['posterior_distinct'] == pytest.approx(distinct / (same + distinct), abs=1e-12)
        joint_probability += mass
        posterior_average += mass * called['posterior_distinct']
    assert joint_probability == pytest.approx(1)
    assert posterior_average == pytest.approx(.5)
