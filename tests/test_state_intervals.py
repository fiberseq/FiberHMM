"""Brute-force gates for the additive fixed-interval hard-core kernel."""

import itertools
import math
import random

import pytest

from fiberhmm.inference.state_intervals import FixedInterval, infer_fixed_intervals


def _lse(values):
    values = list(values)
    if not values or max(values) == -math.inf:
        return -math.inf
    maximum = max(values)
    return maximum + math.log(math.fsum(math.exp(x - maximum) for x in values))


def _brute(intervals, evidence, activities):
    records = sorted(zip(intervals, evidence, activities), key=lambda row: row[0].state_id)
    intervals, evidence, activities = zip(*records) if records else ((), (), ())
    valid = []
    for bits in itertools.product((0, 1), repeat=len(intervals)):
        selected = [i for i, present in enumerate(bits) if present]
        if any(max(intervals[i].start, intervals[j].start) <
               min(intervals[i].end, intervals[j].end)
               for i, j in itertools.combinations(selected, 2)):
            continue
        prior = sum(activities[i] for i in selected)
        score = prior + sum(evidence[i] for i in selected)
        valid.append((bits, prior, score))
    zp, zo = _lse(row[1] for row in valid), _lse(row[2] for row in valid)
    posterior = [math.fsum(math.exp(score - zo) for bits, _, score in valid if bits[i])
                 for i in range(len(intervals))]
    prior = [math.fsum(math.exp(score - zp) for bits, score, _ in valid if bits[i])
             for i in range(len(intervals))]
    include = [_lse(score for bits, _, score in valid if bits[i]) for i in range(len(intervals))]
    exclude = [_lse(score for bits, _, score in valid if not bits[i]) for i in range(len(intervals))]
    best = min(valid, key=lambda row: (-row[2], sum(row[0]), tuple(
        intervals[i].state_id for i, present in enumerate(row[0]) if present)))
    map_ids = tuple(intervals[i].state_id for i, present in enumerate(best[0]) if present)
    return zo, zp, posterior, prior, include, exclude, map_ids, valid


@pytest.mark.parametrize("n", range(13))
def test_partitions_marginals_and_covariance_match_brute_force(n):
    rng = random.Random(7281 + n)
    for _ in range(8):
        intervals = [FixedInterval(f"s{i:02}", start := rng.randrange(-5, 25),
                                   start + rng.randrange(1, 12)) for i in range(n)]
        evidence = [rng.uniform(-6, 6) for _ in intervals]
        eta = [rng.uniform(-2, 2) for _ in intervals]
        result = infer_fixed_intervals(intervals, evidence, eta)
        zo, zp, post, prior, inc, exc, map_ids, valid = _brute(intervals, evidence, eta)
        assert result.log_z_observation == pytest.approx(zo, abs=1e-12)
        assert result.log_z_prior == pytest.approx(zp, abs=1e-12)
        assert result.log_marginal_likelihood_ratio == pytest.approx(zo - zp, abs=1e-12)
        assert result.posterior_marginals == pytest.approx(post, abs=1e-12)
        assert result.prior_marginals == pytest.approx(prior, abs=1e-12)
        assert result.log_include == pytest.approx(inc, abs=1e-12)
        assert result.log_exclude == pytest.approx(exc, abs=1e-12)
        assert result.map_state_ids == map_ids
        vector = [rng.uniform(-2, 2) for _ in intervals]
        for weight_column, normalizer, marginals, operation in (
            (2, zo, post, result.posterior_covariance_vector_product),
            (1, zp, prior, result.prior_covariance_vector_product),
        ):
            mean = sum(v * p for v, p in zip(vector, marginals))
            expected = [math.fsum(
                math.exp(row[weight_column] - normalizer) * row[0][i] *
                (sum(v * bit for v, bit in zip(vector, row[0])) - mean)
                for row in valid) for i in range(n)]
            assert operation(vector) == pytest.approx(expected, abs=1e-12)


def test_activity_gradient_and_hessian_match_finite_differences():
    intervals = [FixedInterval("a", 0, 10), FixedInterval("b", 4, 8),
                 FixedInterval("c", 10, 15), FixedInterval("d", 6, 13)]
    evidence, eta, vector = [2.1, -0.8, 1.3, 0.4], [-0.3, 0.9, -1.0, 0.2], [0.2, -0.7, 1.1, 0.5]
    result = infer_fixed_intervals(intervals, evidence, eta)
    h = 1e-5
    for i in range(len(eta)):
        plus, minus = eta.copy(), eta.copy()
        plus[i] += h
        minus[i] -= h
        derivative = (infer_fixed_intervals(intervals, evidence, plus).log_marginal_likelihood_ratio -
                      infer_fixed_intervals(intervals, evidence, minus).log_marginal_likelihood_ratio) / (2 * h)
        assert result.activity_gradient[i] == pytest.approx(derivative, abs=1e-9)
    plus = infer_fixed_intervals(intervals, evidence, [a + h * v for a, v in zip(eta, vector)])
    minus = infer_fixed_intervals(intervals, evidence, [a - h * v for a, v in zip(eta, vector)])
    expected = [(a - b) / (2 * h) for a, b in zip(plus.activity_gradient, minus.activity_gradient)]
    assert result.activity_hessian_vector_product(vector) == pytest.approx(expected, abs=1e-9)


def test_disjoint_and_touching_intervals_are_independent_bernoulli():
    intervals = [FixedInterval("a", 0, 5), FixedInterval("b", 5, 8), FixedInterval("c", 10, 20)]
    eta, evidence = [0.5, -0.7, 1.2], [1.1, -1.9, 0.4]
    result = infer_fixed_intervals(intervals, evidence, eta)
    logistic = lambda x: 1 / (1 + math.exp(-x))
    assert result.prior_marginals == pytest.approx([logistic(x) for x in eta], abs=1e-12)
    assert result.posterior_marginals == pytest.approx(
        [logistic(a + b) for a, b in zip(eta, evidence)], abs=1e-12)
    assert result.prior_covariance_vector_product([1, 0, 0]) == pytest.approx(
        [logistic(eta[0]) * (1 - logistic(eta[0])), 0, 0], abs=1e-12)


def test_overlapping_activity_is_not_logit_marginal():
    result = infer_fixed_intervals([FixedInterval("a", 0, 10), FixedInterval("b", 2, 5)], [0, 0])
    assert result.prior_marginals == pytest.approx([1 / 3, 1 / 3], abs=1e-12)
    assert result.log_z_prior == pytest.approx(math.log(3), abs=1e-12)
    # The impossible joint configuration has exactly zero second moment.
    covariance = result.prior_covariance_vector_product([0, 1])[0]
    assert covariance + result.prior_marginals[0] * result.prior_marginals[1] == pytest.approx(0, abs=1e-15)


def test_canonical_order_reflection_and_deterministic_ties():
    intervals = [FixedInterval("z", 0, 10), FixedInterval("a", 1, 9), FixedInterval("b", 10, 12)]
    evidence = [1, 1, 0]
    expected = infer_fixed_intervals(intervals, evidence)
    assert expected.map_state_ids == ("a",)
    for order in itertools.permutations(range(3)):
        result = infer_fixed_intervals([intervals[i] for i in order], [evidence[i] for i in order])
        assert result == expected
    reflected = infer_fixed_intervals(
        [FixedInterval(item.state_id, -item.end, -item.start) for item in intervals], evidence)
    for name in ("state_ids", "log_z_observation", "log_z_prior", "posterior_marginals",
                 "prior_marginals", "log_include", "log_exclude", "map_state_ids"):
        assert getattr(reflected, name) == getattr(expected, name)


def test_negative_infinite_evidence_preserves_prior_and_empty_configuration():
    intervals = [FixedInterval("a", 0, 10), FixedInterval("b", 2, 5)]
    result = infer_fixed_intervals(intervals, [-math.inf, -math.inf], [1, 2])
    assert result.log_z_observation == 0
    assert result.posterior_marginals == (0, 0)
    assert result.log_include == (-math.inf, -math.inf)
    assert result.log_exclude == (0, 0)
    assert all(p > 0 for p in result.prior_marginals)
    assert result.map_state_ids == ()
    assert result.posterior_covariance_vector_product([2, 3]) == (0, 0)


def test_extreme_scores_do_not_lose_excluded_partition():
    intervals = [FixedInterval("a", 0, 10), FixedInterval("b", 10, 20), FixedInterval("c", 30, 31)]
    result = infer_fixed_intervals(intervals, [1000, 900, -1000])
    assert result.log_z_observation == 1900
    assert result.log_exclude == (900, 1000, 1900)
    assert result.posterior_marginals == (1, 1, 0)
    assert result.log_include == (1900, 1900, 900)
    assert result.activity_hessian_vector_product([1, 1, 1]) == pytest.approx([-0.25] * 3)


def test_zero_evidence_is_normalized_not_positive_population_support():
    intervals = [FixedInterval("a", 0, 10), FixedInterval("b", 4, 8), FixedInterval("c", 20, 30)]
    result = infer_fixed_intervals(intervals, [0, 0, 0], [-2, 1, 0.4])
    assert result.log_marginal_likelihood_ratio == 0
    assert result.posterior_marginals == result.prior_marginals
    assert result.activity_gradient == (0, 0, 0)
    assert result.activity_hessian_vector_product([1, 2, 3]) == (0, 0, 0)


@pytest.mark.parametrize("state_id,start,end", [("", 0, 1), ("a", 1, 1), ("a", 2, 1),
                                                ("a", 0.1, 1), ("a", True, 2)])
def test_invalid_intervals_fail(state_id, start, end):
    with pytest.raises(ValueError):
        FixedInterval(state_id, start, end)


@pytest.mark.parametrize("evidence,activities", [([math.nan], None), ([math.inf], None),
                                                  ([0], [math.inf]), ([0], [-math.inf]),
                                                  ([], None), ([0], [])])
def test_invalid_scores_fail(evidence, activities):
    with pytest.raises(ValueError):
        infer_fixed_intervals([FixedInterval("a", 0, 1)], evidence, activities)


def test_duplicate_ids_and_bad_direction_fail():
    with pytest.raises(ValueError):
        infer_fixed_intervals([FixedInterval("a", 0, 1), FixedInterval("a", 2, 3)], [0, 0])
    result = infer_fixed_intervals([FixedInterval("a", 0, 1)], [0])
    with pytest.raises(ValueError):
        result.prior_covariance_vector_product([])
    with pytest.raises(ValueError):
        result.activity_hessian_vector_product([math.nan])


def test_common_observation_base_multiplier_cancels_before_kernel():
    # The API consumes likelihood RATIOS: multiplying every configuration's
    # likelihood and the shared accessible base by the same constant leaves
    # its inputs unchanged. Adding a constant to every state score is NOT the
    # same operation (it instead changes the state-count prior).
    intervals = [FixedInterval("a", 0, 2)]
    base, state, multiplier = 0.2, 0.7, 1e-120
    first = infer_fixed_intervals(intervals, [math.log(state) - math.log(base)])
    second = infer_fixed_intervals(intervals, [math.log(state * multiplier) - math.log(base * multiplier)])
    assert first.posterior_marginals == pytest.approx(second.posterior_marginals, abs=1e-12)
    assert first.log_marginal_likelihood_ratio == pytest.approx(second.log_marginal_likelihood_ratio, abs=1e-12)


def test_batch_implementation_matches_scalar_oracle():
    """The throughput path and scalar derivative oracle share one contract."""
    import numpy as np
    from fiberhmm.inference.hierarchical_state_model import FixedStateCatalog, StateInterval

    rng = random.Random(18032)
    for n in (0, 1, 4, 8, 12, 30):
        intervals = [FixedInterval(f"s{i:02}", start := rng.randrange(60),
                                   start + rng.randrange(1, 15)) for i in range(n)]
        rng.shuffle(intervals)
        eta = np.array([rng.uniform(-2, 2) for _ in intervals])
        evidence = np.array([[rng.uniform(-10, 10) for _ in intervals] for _ in range(7)])
        catalog = FixedStateCatalog([StateInterval(item.state_id, item.start, item.end) for item in intervals])
        batch, batch_prior = catalog.infer(evidence + eta), catalog.infer(eta)
        canonical_order = sorted(range(n), key=lambda i: intervals[i].state_id)
        for row in range(len(evidence)):
            scalar = infer_fixed_intervals(intervals, evidence[row], eta)
            assert batch.log_z[row] == pytest.approx(scalar.log_z_observation, abs=1e-12)
            assert batch_prior.log_z[0] == pytest.approx(scalar.log_z_prior, abs=1e-12)
            assert batch.marginals[row, canonical_order] == pytest.approx(scalar.posterior_marginals, abs=1e-12)
            assert batch_prior.marginals[0, canonical_order] == pytest.approx(scalar.prior_marginals, abs=1e-12)
            assert tuple(intervals[i].state_id for i in catalog.map_indices(evidence[row] + eta)) == scalar.map_state_ids
