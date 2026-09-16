"""Reclassify the original pair posterior using an explicit edge tolerance.

The original observation likelihoods and prior are unchanged: probability
1/2 on an exact shared state, uniform over G states, and probability 1/2 on
ordered unequal states, uniform over G(G-1) choices. States comprise one
accessible empty state and every nonempty contiguous pair-union projection.

For two nonempty states, boundary distance is |left_a-left_b|+|right_a-right_b|
in UNION OPPORTUNITY INDEX STEPS, not bp, actual hit differences, or model
confidence. A tolerance changes only which posterior state pairs count as
``far``. Empty/nonempty pairs always remain far. At radius zero this is the
original posterior_distinct exactly, including its uninformative sentinel.

For sums A=sum L_a, B=sum L_b, D=sum L_a L_b, and N_r=sum of products of
likelihoods over near state pairs, P(far_r|Y)=(AB-N_r)/(AB+(G-2)D).
N_r includes (empty,empty) once. It is not renormalized to create a new prior.

The event prior decreases with radius. Therefore 1-P(far_r) is NOT necessarily
evidence for matching, especially without information from both observations.
The details API exposes the event prior and information flags. These scores
are conditional model quantities, not empirically calibrated p/q values.
"""
from __future__ import annotations

from functools import lru_cache
import math
from numbers import Integral

import numpy as np

from .molecule_footprint_pair import score_pairs

try:
    from numba import njit
except ImportError:
    njit = None


MODEL_DESCRIPTION = (
    "posterior boundary-far event under the unchanged exact-same/distinct "
    "50:50 mixture prior; tolerance is total L1 boundary displacement in "
    "pair-union opportunity indices; not bp, hit differences, or calibrated q"
)


def _radii(tolerances):
    radii = tuple(tolerances)
    if not radii:
        raise ValueError("At least one edge tolerance is required")
    if any(isinstance(r, (bool, np.bool_)) or not isinstance(r, Integral) or r < 0 for r in radii):
        raise ValueError("Edge tolerances must be nonnegative integer opportunity-index steps")
    if any(r > np.iinfo(np.int64).max for r in radii):
        raise ValueError("Edge tolerance does not fit an integer index")
    return tuple(sorted({int(r) for r in radii}))


def _posterior_far(log_product, log_near, log_denominator):
    gap = log_product - log_near
    tolerance = 128 * np.finfo(np.float64).eps * (1 + abs(log_product) + abs(log_near))
    if gap < -tolerance:
        raise FloatingPointError("Near interval-pair sum exceeds the all-pair sum")
    if gap <= 0:
        return 0.0
    if gap <= math.log(2):
        log_fraction = math.log(-math.expm1(-gap))
    else:
        log_fraction = math.log1p(-math.exp(-gap))
    return math.exp(log_product + log_fraction - log_denominator)


def _near_batch_reference(values, observed, ua, ub, left, right, radii,
                          log_a, log_b, log_diagonal, original_probability):
    size = len(ua)
    result = np.empty((size, len(radii)), dtype=np.float64)
    informative = np.zeros((size, 2), dtype=np.bool_)
    maximum_domain = 0
    for pair in range(size):
        maximum_domain = max(maximum_domain, right[pair] - left[pair])
    prefix_a = np.zeros(maximum_domain + 1, dtype=np.float64)
    prefix_b = np.zeros(maximum_domain + 1, dtype=np.float64)
    # Radius larger than every possible nonempty-state distance needs no
    # offset enumeration. This also bounds temporary storage for huge radii.
    maximum_radius = min(radii[-1], max(0, 2 * maximum_domain - 2))
    shells = np.empty(maximum_radius + 1, dtype=np.float64)
    for pair in range(size):
        k = 0
        prefix_a[0] = prefix_b[0] = 0.0
        for column in range(left[pair], right[pair]):
            oa, ob = observed[ua[pair], column], observed[ub[pair], column]
            if not oa and not ob:
                continue
            va = values[ua[pair], column] if oa else 0.0
            vb = values[ub[pair], column] if ob else 0.0
            informative[pair, 0] = informative[pair, 0] or va != 0.0
            informative[pair, 1] = informative[pair, 1] or vb != 0.0
            prefix_a[k + 1] = prefix_a[k] + va
            prefix_b[k + 1] = prefix_b[k] + vb
            k += 1
        if k == 0:
            # The legacy API returns .5 for the undefined G=1 distinct model.
            # Preserve that sentinel, but details mark the domain unavailable.
            for radius_index in range(len(radii)):
                result[pair, radius_index] = .5
            continue
        g = 1 + k * (k + 1) // 2
        log_product = log_a[pair] + log_b[pair]
        log_denominator = log_product
        if g > 2:
            log_denominator = np.logaddexp(log_product, math.log(g - 2) + log_diagonal[pair])
        maximum_distance = 2 * k - 2
        limit = 0
        for radius in radii:
            if radius < maximum_distance:
                limit = max(limit, radius)
        # Once every pair of nonempty states is near, its sum factorizes.
        # Keep accessible/nonempty alternatives outside this near event.
        all_nonempty_near = -math.inf
        if radii[-1] >= maximum_distance:
            nonempty_a = -math.inf
            nonempty_b = -math.inf
            if log_a[pair] > 0:
                if log_a[pair] <= math.log(2):
                    nonempty_a = math.log(math.expm1(log_a[pair]))
                else:
                    nonempty_a = log_a[pair] + math.log1p(-math.exp(-log_a[pair]))
            if log_b[pair] > 0:
                if log_b[pair] <= math.log(2):
                    nonempty_b = math.log(math.expm1(log_b[pair]))
                else:
                    nonempty_b = log_b[pair] + math.log1p(-math.exp(-log_b[pair]))
            all_nonempty_near = np.logaddexp(0.0, nonempty_a + nonempty_b)
        for distance in range(limit + 1):
            shells[distance] = -math.inf
        # A state pair has exactly one signed (delta_left, delta_right).
        # At fixed offsets, valid starts form a growing prefix as e advances.
        for dl in range(-min(limit, k - 1), min(limit, k - 1) + 1):
            remaining = limit - abs(dl)
            for dr in range(-min(remaining, k - 1), min(remaining, k - 1) + 1):
                distance = abs(dl) + abs(dr)
                if distance == 0:
                    continue  # Reuse the exact original diagonal sum.
                first_start = max(0, -dl)
                first_end = max(1, 1 - dr)
                final_end = min(k, k - dr)
                correction = min(0, dr - dl)
                next_start = first_start
                start_sum = -math.inf
                offset_sum = -math.inf
                for end in range(first_end, final_end + 1):
                    last_start = end - 1 + correction
                    while next_start <= last_start:
                        start_sum = np.logaddexp(start_sum, -prefix_a[next_start] - prefix_b[next_start + dl])
                        next_start += 1
                    if start_sum != -math.inf:
                        offset_sum = np.logaddexp(offset_sum, start_sum + prefix_a[end] + prefix_b[end + dr])
                shells[distance] = np.logaddexp(shells[distance], offset_sum)
        cumulative_near = log_diagonal[pair]
        previous_radius = 0
        previous_probability = original_probability[pair]
        for radius_index in range(len(radii)):
            radius = radii[radius_index]
            for distance in range(previous_radius + 1, min(radius, limit) + 1):
                cumulative_near = np.logaddexp(cumulative_near, shells[distance])
            previous_radius = min(radius, limit)
            if radius == 0:
                probability = original_probability[pair]
            else:
                if radius >= maximum_distance:
                    cumulative_near = all_nonempty_near
                probability = _posterior_far(log_product, cumulative_near, log_denominator)
                tolerance = 512 * np.finfo(np.float64).eps * (1 + abs(log_product) + abs(cumulative_near))
                if probability > previous_probability + tolerance:
                    raise FloatingPointError("Boundary-far posterior increased with edge tolerance")
                # Only protect exact r=0/no-information sentinels from tiny
                # cancellation error. A material monotonicity failure raises.
                probability = min(previous_probability, probability)
            result[pair, radius_index] = probability
            previous_probability = probability
    return result, informative


# Numba needs the stable scalar helper registered before compiling its caller.
if njit is not None:
    _posterior_far = njit(cache=True)(_posterior_far)
    _near_batch_compiled = njit(cache=True)(_near_batch_reference)
else:
    _near_batch_compiled = None


@lru_cache(maxsize=16_384)
def _near_count(k, radius):
    """Count ordered near pairs without using any observed outcomes."""
    g = 1 + k * (k + 1) // 2
    if radius >= 2 * k - 2:
        return 1 + (g - 1) ** 2
    count = g
    for dl in range(-min(radius, k - 1), min(radius, k - 1) + 1):
        remaining = radius - abs(dl)
        for dr in range(-min(remaining, k - 1), min(remaining, k - 1) + 1):
            if dl == dr == 0:
                continue
            lower = max(0, -dl)
            correction = min(0, dr - dl)
            first = max(1, 1 - dr, lower - correction + 1)
            last = min(k, k - dr)
            n = max(0, last - first + 1)
            count += n * (first + last) // 2 + n * (correction - lower)
    return count


def prior_far_probabilities(n_union, tolerances=(0, 2, 4)):
    """Original-mixture event priors; K=0 entries are NaN (unavailable)."""
    counts = np.asarray(n_union)
    if counts.ndim != 1 or counts.dtype.kind not in "iu" or np.any(counts < 0):
        raise ValueError("Union opportunity counts must be a one-dimensional nonnegative integer array")
    answer = {}
    for radius in _radii(tolerances):
        values = np.full(len(counts), np.nan, dtype=float)
        for k in np.unique(counts):
            if not k:
                continue
            k = int(k)
            g = 1 + k * (k + 1) // 2
            near = _near_count(k, radius)
            values[counts == k] = (g * g - near) / (2 * g * (g - 1))
        answer[radius] = values
    return answer


def score_tolerance_details(values, observed, ua, ub, left, right,
                            tolerances=(0, 2, 4), *, use_numba=True):
    """Score posterior far events and expose their unchanged-model priors.

    Arguments use the original score_pairs matrix/index conventions. Jointly
    missing columns do not enter the union lattice; observed zero LRs do.
    ``pair_discrimination_available`` requires some nonzero LR in EACH read.
    If it is false, do not interpret posterior nearness as pair agreement.
    With one informative read, its geometry alone can update event probability
    through the fixed prior's neighborhood volume; no pair match was observed.
    """
    radii = _radii(tolerances)
    base = score_pairs(values, observed, ua, ub, left, right, use_numba=use_numba)
    backend = _near_batch_compiled if use_numba and _near_batch_compiled is not None else _near_batch_reference
    posterior, informative = backend(
        np.ascontiguousarray(values, dtype=float), np.ascontiguousarray(observed, dtype=bool),
        np.asarray(ua, dtype=np.int64), np.asarray(ub, dtype=np.int64),
        np.asarray(left, dtype=np.int64), np.asarray(right, dtype=np.int64),
        np.asarray(radii, dtype=np.int64), base["log_sum_a"], base["log_sum_b"],
        base["log_sum_same"], base["posterior_distinct"])
    answer = {r: posterior[:, i] for i, r in enumerate(radii)}
    if np.any(~np.isfinite(posterior)) or np.any((posterior < 0) | (posterior > 1)):
        raise FloatingPointError("Invalid boundary-far posterior")
    return {"posterior_far": answer, "prior_far": prior_far_probabilities(base["n_union"], radii),
            "base_scores": base, "informative_a": informative[:, 0], "informative_b": informative[:, 1],
            "pair_discrimination_available": np.all(informative, axis=1),
            "empty_domain": base["n_union"] == 0, "edge_tolerance_opportunities": radii,
            "model": MODEL_DESCRIPTION}


def score_tolerances(values, observed, ua, ub, left, right,
                     tolerances=(0, 2, 4), *, use_numba=True):
    """Return ``{edge_tolerance_opportunities: posterior_far_array}``.

    See score_tolerance_details for essential prior and no-information flags.
    This function never modifies observations, emissions, native calls, or the
    existing pair kernel. Raising the tolerance cannot raise P(far).
    """
    return score_tolerance_details(values, observed, ua, ub, left, right,
                                   tolerances, use_numba=use_numba)["posterior_far"]
