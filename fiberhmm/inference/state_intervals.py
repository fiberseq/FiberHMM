"""Exact hard-core inference for additive, fixed, half-open intervals.

This is a mathematical kernel, not a footprint caller.  ``log_evidence[k]``
must be the additive contribution of state k to ONE shared observation-domain
log likelihood ratio.  Activities are log activities (eta), not marginal
prevalences.  The empty configuration is always present and has weight one.

Variable geometry, shared observations with nonadditive likelihoods, and
pairwise interactions must be integrated by a joint kernel BEFORE using this
fast path, and only when that integration provably factorizes.  In particular,
gamma=0 alone does not justify reducing variable geometry to scalar weights.

Partition/marginal work is O(sum(component_size**2)); explicit leave-one-state
partitions avoid catastrophic subtraction when inclusion probability is near
one.  Directional covariance products are O(n log n), without a dense matrix.
Disjoint conflict components are factored exactly.  No external dependency is
required.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass, field
import math
from numbers import Integral
from typing import Sequence


@dataclass(frozen=True)
class FixedInterval:
    """A uniquely identified positive-width interval [start, end)."""

    state_id: str
    start: int
    end: int

    def __post_init__(self) -> None:
        if not isinstance(self.state_id, str) or not self.state_id:
            raise ValueError("state_id must be a nonempty string")
        if any(isinstance(x, bool) or not isinstance(x, Integral)
               for x in (self.start, self.end)):
            raise ValueError("interval coordinates must be integers")
        if self.end <= self.start:
            raise ValueError("fixed intervals must have positive width")


@dataclass(frozen=True)
class FixedIntervalResult:
    """Exact quantities in canonical state-ID order.

    ``log_include`` and ``log_exclude`` are unnormalized observation-weighted
    configuration partitions, not Bayes factors or conditioned likelihoods.
    Their log-sum is ``log_z_observation`` for each state.  They include prior
    activities; the normalized observation likelihood ratio subtracts the
    prior partition.  ``activity_gradient`` is the derivative of that ratio.

    MAP ties prefer fewer selected states and then lexicographically sorted
    IDs.  This explicit rule is invariant to the caller's interval ordering.
    The covariance/Hessian methods take vectors aligned with ``state_ids``.
    """

    state_ids: tuple[str, ...]
    log_z_observation: float
    log_z_prior: float
    log_marginal_likelihood_ratio: float
    posterior_marginals: tuple[float, ...]
    prior_marginals: tuple[float, ...]
    log_include: tuple[float, ...]
    log_exclude: tuple[float, ...]
    map_state_ids: tuple[str, ...]
    activity_gradient: tuple[float, ...]
    _intervals: tuple[FixedInterval, ...] = field(repr=False)
    _observation_weights: tuple[float, ...] = field(repr=False)
    _activities: tuple[float, ...] = field(repr=False)

    def posterior_covariance_vector_product(
        self, vector: Sequence[float]
    ) -> tuple[float, ...]:
        """Return Cov_posterior(z) @ vector without constructing Cov."""
        return _covariance_product(
            self._intervals, self._observation_weights,
            self.posterior_marginals, vector,
        )

    def prior_covariance_vector_product(
        self, vector: Sequence[float]
    ) -> tuple[float, ...]:
        """Return Cov_prior(z) @ vector without constructing Cov."""
        return _covariance_product(
            self._intervals, self._activities, self.prior_marginals, vector,
        )

    def activity_hessian_vector_product(
        self, vector: Sequence[float]
    ) -> tuple[float, ...]:
        """Return [Cov_posterior(z) - Cov_prior(z)] @ vector."""
        post = self.posterior_covariance_vector_product(vector)
        prior = self.prior_covariance_vector_product(vector)
        return tuple(a - b for a, b in zip(post, prior))


def _logadd(a: float, b: float) -> float:
    if a == -math.inf:
        return b
    if b == -math.inf:
        return a
    hi, lo = (a, b) if a >= b else (b, a)
    return hi + math.log1p(math.exp(lo - hi))


def _probability(include: float, exclude: float) -> float:
    if include == -math.inf:
        return 0.0
    delta = include - exclude
    if delta >= 0:
        return 1.0 / (1.0 + math.exp(-delta))
    odds = math.exp(delta)
    return odds / (1.0 + odds)


def _values(values: Sequence[float], n: int, name: str,
            allow_negative_infinity: bool = False) -> tuple[float, ...]:
    result = tuple(float(x) for x in values)
    if len(result) != n:
        raise ValueError(f"{name} must have one value per interval")
    if any(not math.isfinite(x) and not
           (allow_negative_infinity and x == -math.inf) for x in result):
        raise ValueError(f"{name} must be finite" +
                         (" or negative infinity" if allow_negative_infinity else ""))
    return result


def _components(intervals: Sequence[FixedInterval]) -> list[tuple[int, ...]]:
    ordered = sorted(range(len(intervals)),
                     key=lambda i: (intervals[i].start, intervals[i].end,
                                    intervals[i].state_id))
    result: list[tuple[int, ...]] = []
    current: list[int] = []
    end = None
    for i in ordered:
        interval = intervals[i]
        if end is not None and interval.start >= end:
            result.append(tuple(sorted(current)))
            current = []
            end = None
        current.append(i)
        end = interval.end if end is None else max(end, interval.end)
    if current:
        result.append(tuple(sorted(current)))
    return result


class _Component:
    def __init__(self, intervals: Sequence[FixedInterval]) -> None:
        self.intervals = tuple(intervals)
        self.n = len(intervals)
        self.by_end = sorted(range(self.n),
                             key=lambda i: (intervals[i].end,
                                            intervals[i].start, intervals[i].state_id))
        self.by_start = sorted(range(self.n),
                               key=lambda i: (intervals[i].start,
                                              intervals[i].end, intervals[i].state_id))
        ends = [intervals[i].end for i in self.by_end]
        starts = [intervals[i].start for i in self.by_start]
        self.left = [bisect_right(ends, item.start) for item in intervals]
        self.right = [bisect_left(starts, item.end) for item in intervals]

    def prefix(self, weights: Sequence[float], omit: int = -1,
               direction: Sequence[float] | None = None
               ) -> tuple[list[float], list[float]]:
        logs = [0.0] * (self.n + 1)
        means = [0.0] * (self.n + 1)
        for pos, i in enumerate(self.by_end, 1):
            if i == omit:
                logs[pos], means[pos] = logs[pos - 1], means[pos - 1]
                continue
            left = self.left[i]
            skip, take = logs[pos - 1], weights[i] + logs[left]
            logs[pos] = _logadd(skip, take)
            if direction is not None:
                p = _probability(take, skip)
                means[pos] = ((1.0 - p) * means[pos - 1] +
                              p * (direction[i] + means[left]))
        return logs, means

    def suffix(self, weights: Sequence[float],
               direction: Sequence[float] | None = None
               ) -> tuple[list[float], list[float]]:
        logs = [0.0] * (self.n + 1)
        means = [0.0] * (self.n + 1)
        for pos in range(self.n - 1, -1, -1):
            i = self.by_start[pos]
            right = self.right[i]
            skip, take = logs[pos + 1], weights[i] + logs[right]
            logs[pos] = _logadd(skip, take)
            if direction is not None:
                p = _probability(take, skip)
                means[pos] = ((1.0 - p) * means[pos + 1] +
                              p * (direction[i] + means[right]))
        return logs, means

    def distribution(self, weights: Sequence[float]
                     ) -> tuple[float, list[float], list[float], list[float]]:
        prefix, _ = self.prefix(weights)
        suffix, _ = self.suffix(weights)
        include = [weights[i] + prefix[self.left[i]] + suffix[self.right[i]]
                   for i in range(self.n)]
        # Do not subtract Z_include from Z: both can round to the same number.
        exclude = [self.prefix(weights, omit=i)[0][-1] for i in range(self.n)]
        return prefix[-1], include, exclude, [
            _probability(a, b) for a, b in zip(include, exclude)
        ]

    def map_ids(self, weights: Sequence[float]) -> tuple[str, ...]:
        scores = [0.0] * (self.n + 1)
        selections: list[tuple[str, ...]] = [()] * (self.n + 1)
        for pos, i in enumerate(self.by_end, 1):
            previous = self.left[i]
            take_score = weights[i] + scores[previous]
            take = tuple(sorted(selections[previous] + (self.intervals[i].state_id,)))
            skip = selections[pos - 1]
            if take_score > scores[pos - 1] or (
                take_score == scores[pos - 1] and
                (len(take), take) < (len(skip), skip)
            ):
                scores[pos], selections[pos] = take_score, take
            else:
                scores[pos], selections[pos] = scores[pos - 1], skip
        return selections[-1]


def _covariance_product(intervals: Sequence[FixedInterval],
                        weights: Sequence[float], marginals: Sequence[float],
                        vector: Sequence[float]) -> tuple[float, ...]:
    vector = _values(vector, len(intervals), "vector")
    result = [0.0] * len(intervals)
    for indices in _components(intervals):
        component = _Component([intervals[i] for i in indices])
        local_weights = [weights[i] for i in indices]
        local_vector = [vector[i] for i in indices]
        _, prefix_mean = component.prefix(local_weights, direction=local_vector)
        _, suffix_mean = component.suffix(local_weights, direction=local_vector)
        for j, i in enumerate(indices):
            conditional_mean = (local_vector[j] + prefix_mean[component.left[j]] +
                                suffix_mean[component.right[j]])
            result[i] = marginals[i] * (conditional_mean - prefix_mean[-1])
    return tuple(result)


def infer_fixed_intervals(
    intervals: Sequence[FixedInterval], log_evidence: Sequence[float],
    activities: Sequence[float] | None = None,
) -> FixedIntervalResult:
    """Infer one observation under a normalized hard-core interval prior.

    Input vectors follow the input interval order; every returned vector follows
    sorted ``state_ids``.  ``activities=None`` is eta=0 (uniform valid
    configurations, NOT independent prevalence 0.5 for overlapping states).
    Negative-infinite evidence represents exactly zero state likelihood; it
    does not remove that state from the prior or catalog.  Positive infinity,
    NaN, nonfinite activities, duplicate IDs, and zero-width intervals fail.
    """
    intervals = tuple(intervals)
    if any(not isinstance(item, FixedInterval) for item in intervals):
        raise TypeError("intervals must contain FixedInterval objects")
    if len({item.state_id for item in intervals}) != len(intervals):
        raise ValueError("state_id values must be unique")
    evidence = _values(log_evidence, len(intervals), "log_evidence", True)
    eta = ((0.0,) * len(intervals) if activities is None else
           _values(activities, len(intervals), "activities"))
    order = sorted(range(len(intervals)), key=lambda i: intervals[i].state_id)
    intervals = tuple(intervals[i] for i in order)
    evidence = tuple(evidence[i] for i in order)
    eta = tuple(eta[i] for i in order)
    weights = tuple(a + b for a, b in zip(evidence, eta))
    if any(not math.isfinite(w) and e != -math.inf for w, e in zip(weights, evidence)):
        raise ValueError("log_evidence + activities overflowed")
    n = len(intervals)
    post, prior = [0.0] * n, [0.0] * n
    include, exclude = [0.0] * n, [0.0] * n
    observation_zs: list[float] = []
    prior_zs: list[float] = []
    component_records = []
    map_ids: list[str] = []
    for indices in _components(intervals):
        component = _Component([intervals[i] for i in indices])
        local_weights = [weights[i] for i in indices]
        observation_z, inc, exc, probabilities = component.distribution(local_weights)
        prior_z, _, _, prior_probabilities = component.distribution([eta[i] for i in indices])
        observation_zs.append(observation_z)
        prior_zs.append(prior_z)
        component_records.append((indices, observation_z, inc, exc))
        map_ids.extend(component.map_ids(local_weights))
        for j, i in enumerate(indices):
            post[i], prior[i] = probabilities[j], prior_probabilities[j]
    log_z_observation = math.fsum(observation_zs)
    log_z_prior = math.fsum(prior_zs)
    if not math.isfinite(log_z_observation) or not math.isfinite(log_z_prior):
        raise ValueError("configuration partition overflowed")
    for indices, component_z, inc, exc in component_records:
        outside = log_z_observation - component_z
        for j, i in enumerate(indices):
            include[i], exclude[i] = inc[j] + outside, exc[j] + outside
    return FixedIntervalResult(
        tuple(item.state_id for item in intervals), log_z_observation, log_z_prior,
        log_z_observation - log_z_prior, tuple(post), tuple(prior),
        tuple(include), tuple(exclude), tuple(sorted(map_ids)),
        tuple(a - b for a, b in zip(post, prior)), intervals, weights, eta,
    )
