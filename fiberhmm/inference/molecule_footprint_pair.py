"""Compare TWO observations on their full union opportunity lattice.

This module does not pool molecules, fit population mixtures, assign a new
footprint, or change a supplied native interval. Within a fixed half-open
comparison domain, the latent state is either accessibility (the empty state)
or one contiguous protected interval on the pair's union opportunity lattice.
Every identifiable union-lattice interval has equal prior mass.

``same`` gives the two observations one shared latent state. ``distinct`` gives
them two different latent states, uniformly over ordered off-diagonal pairs.
The model odds are fixed at 1:1. Returned probabilities are CONDITIONAL MODEL
PROBABILITIES, not frequentist p/q values or guarantees of a pair false-split
rate. In particular, choosing the domain using these observations, uncertain
emission parameters, dependence, and multiple protected segments are not
accounted for by that probability. Calibration must address those separately.

All actual observations inside the full union domain contribute, including
internal modifications when the two native boundaries happen to agree. An
unobserved position contributes likelihood factor one, never a missing hit.
The emission likelihoods must have one common accessible base measure.

For relative interval likelihoods A_g and B_g, including the empty state,
S_A=sum A_g, S_B=sum B_g, S_AB=sum A_g B_g, and G is the state count. Then
BF(distinct:same) = (S_A S_B / S_AB - 1) / (G - 1).
All three sums are computed exactly in O(K), not by selecting a best boundary.
"""
from __future__ import annotations

import math
from numbers import Integral

import numpy as np

try:
    from numba import njit
except ImportError:  # A correct slower reference backend remains available.
    njit = None


MODEL_DESCRIPTION = (
    "conditional model probability: one common contiguous protected interval "
    "versus two distinct intervals; uniform union-projection interval prior; "
    "fixed 1:1 model odds; not a calibrated p/q value or false-split rate"
)


def _score_batch_reference(values, observed, units_a, units_b, left, right):
    """Array-only implementation, also compiled verbatim by optional Numba."""
    size = len(units_a)
    scores = np.empty((size, 5), dtype=np.float64)
    counts = np.empty((size, 5), dtype=np.int64)
    eps = np.finfo(np.float64).eps
    for pair in range(size):
        unit_a, unit_b = units_a[pair], units_b[pair]
        ending_a = -math.inf
        ending_b = -math.inf
        ending_same = -math.inf
        total_a = 0.0  # The single empty state has likelihood one.
        total_b = 0.0
        total_same = 0.0
        n_a = n_b = n_union = n_shared = 0
        informative_a = informative_b = False
        for column in range(left[pair], right[pair]):
            oa, ob = observed[unit_a, column], observed[unit_b, column]
            if not oa and not ob:
                continue  # No opportunity for either read: no extra state.
            n_a += int(oa)
            n_b += int(ob)
            n_union += 1
            n_shared += int(oa and ob)
            va = values[unit_a, column] if oa else 0.0
            vb = values[unit_b, column] if ob else 0.0
            informative_a = informative_a or va != 0.0
            informative_b = informative_b or vb != 0.0
            # Sum of intervals ending here = exp(v) * (1 + previous ending).
            ending_a = va + np.logaddexp(0.0, ending_a)
            ending_b = vb + np.logaddexp(0.0, ending_b)
            ending_same = va + vb + np.logaddexp(0.0, ending_same)
            total_a = np.logaddexp(total_a, ending_a)
            total_b = np.logaddexp(total_b, ending_b)
            total_same = np.logaddexp(total_same, ending_same)
        n_states = 1 + n_union * (n_union + 1) // 2
        if n_union == 0 or not informative_a or not informative_b:
            # If one observation has constant likelihood, the models are
            # exactly indistinguishable. Avoid roundoff from log(G)-log(G).
            log_bf = 0.0
            posterior = 0.5
        else:
            difference = total_a + total_b - total_same
            tolerance = 64.0 * eps * (1.0 + abs(total_a) + abs(total_b) + abs(total_same))
            if difference < -tolerance:
                raise FloatingPointError("Diagonal interval sum exceeds the product of all interval sums")
            # Nonnegative analytically. At extremely concentrated same-state
            # evidence the positive off-diagonal remainder can round to zero.
            difference = max(0.0, difference)
            if difference == 0.0:
                log_bf = -math.inf
            elif difference <= math.log(2.0):
                log_bf = math.log(math.expm1(difference)) - math.log(n_states - 1)
            else:
                log_bf = difference + math.log1p(-math.exp(-difference)) - math.log(n_states - 1)
            if log_bf >= 0.0:
                posterior = 1.0 / (1.0 + math.exp(-log_bf))
            else:
                odds = math.exp(log_bf)
                posterior = odds / (1.0 + odds)
        scores[pair, 0] = posterior
        scores[pair, 1] = log_bf
        scores[pair, 2] = total_a
        scores[pair, 3] = total_b
        scores[pair, 4] = total_same
        counts[pair, 0] = n_a
        counts[pair, 1] = n_b
        counts[pair, 2] = n_union
        counts[pair, 3] = n_shared
        counts[pair, 4] = n_states
    return scores, counts


_score_batch_compiled = njit(cache=True)(_score_batch_reference) if njit is not None else None


def _indices(values, name):
    array = np.asarray(values)
    if array.ndim != 1 or (array.size and array.dtype.kind not in "iu"):
        raise ValueError(f"{name} must be a one-dimensional integer array")
    return np.ascontiguousarray(array, dtype=np.int64)


def score_pairs(log_lr_matrix, observed_matrix, pair_units_a, pair_units_b,
                domain_left_indices, domain_right_indices, *, use_numba=True):
    """Score many observation pairs on one fixed genomic opportunity grid.

    ``log_lr_matrix`` and ``observed_matrix`` are units x grid positions.
    Each pair uses [left_index, right_index); unobserved columns in BOTH units
    are omitted from its union lattice. Observed zero logLR is still a real
    opportunity and affects the interval prior. Missing values are ignored,
    even when the corresponding numerical matrix cell contains NaN.

    Returns a dict of one-dimensional NumPy arrays. ``posterior_distinct`` is
    not a q value. K=0 is explicitly uninformative (0.5 posterior, G=1), rather
    than claiming evidence for a distinct model with no possible states.
    The fast backend is exact to floating-point arithmetic, not subsampling.
    """
    values = np.asarray(log_lr_matrix, dtype=np.float64)
    observed_input = np.asarray(observed_matrix)
    if values.ndim != 2 or observed_input.shape != values.shape:
        raise ValueError("Evidence and observation mask must be equal-shaped two-dimensional arrays")
    if observed_input.dtype.kind != "b" and np.any((observed_input != 0) & (observed_input != 1)):
        raise ValueError("Observation mask must contain only boolean/0/1 entries")
    observed = np.ascontiguousarray(observed_input, dtype=np.bool_)
    if np.any(~np.isfinite(values[observed])):
        raise ValueError("Observed log likelihood ratios must be finite")
    values = np.ascontiguousarray(values)
    ua = _indices(pair_units_a, "pair_units_a")
    ub = _indices(pair_units_b, "pair_units_b")
    left = _indices(domain_left_indices, "domain_left_indices")
    right = _indices(domain_right_indices, "domain_right_indices")
    if not (len(ua) == len(ub) == len(left) == len(right)):
        raise ValueError("Pair and domain arrays must have the same length")
    if np.any(ua < 0) or np.any(ub < 0) or np.any(ua >= len(values)) or np.any(ub >= len(values)):
        raise ValueError("Pair unit index outside the evidence matrix")
    if np.any(left < 0) or np.any(right < left) or np.any(right > values.shape[1]):
        raise ValueError("Domain indices must satisfy 0 <= left <= right <= grid size")
    backend = _score_batch_compiled if use_numba and _score_batch_compiled is not None else _score_batch_reference
    scores, counts = backend(values, observed, ua, ub, left, right)
    if np.any(~np.isfinite(scores[:, 0])) or np.any(np.isnan(scores[:, 1])):
        raise FloatingPointError("Nonfinite interval-model probability")
    names = ("posterior_distinct", "log_bayes_factor", "log_sum_a", "log_sum_b", "log_sum_same")
    result = {name: scores[:, column] for column, name in enumerate(names)}
    count_names = ("n_a", "n_b", "n_union", "n_shared", "n_interval_states")
    result.update({name: counts[:, column] for column, name in enumerate(count_names)})
    return result


def _observation(positions, hits, p_accessible, p_protected, start, end):
    positions = _indices(positions, "positions")
    hits = np.asarray(hits)
    if hits.shape != positions.shape or np.any((hits != 0) & (hits != 1)):
        raise ValueError("One binary modification observation is required per position")
    if len(positions) > 1 and np.any(positions[1:] <= positions[:-1]):
        raise ValueError("Observation positions must be strictly increasing and unique")
    try:
        pa = np.broadcast_to(np.asarray(p_accessible, dtype=float), positions.shape)
        pp = np.broadcast_to(np.asarray(p_protected, dtype=float), positions.shape)
    except ValueError as exc:
        raise ValueError("Emission probabilities must be scalar or one per observation") from exc
    if np.any(~np.isfinite(pa)) or np.any(~np.isfinite(pp)) or np.any(pa <= 0) or np.any(pa >= 1) or np.any(pp <= 0) or np.any(pp >= 1):
        raise ValueError("Emission probabilities must be finite and strictly between zero and one")
    # No forced assumption pp < pa: calibration inputs remain visible to the
    # caller, and an uninformative pp==pa correctly produces exactly zero LR.
    keep = (positions >= start) & (positions < end)
    hit = hits[keep].astype(bool)
    pa, pp = pa[keep], pp[keep]
    log_lr = np.where(hit, np.log(pp) - np.log(pa), np.log1p(-pp) - np.log1p(-pa))
    return positions[keep], log_lr


def compare_observations(pos_a, hits_a, pa_a, pp_a, pos_b, hits_b, pa_b, pp_b,
                         start, end, *, use_numba=True):
    """Emission-aware comparison of two individual native observations.

    Positions are integer genomic coordinates; start/end define the full
    half-open comparison domain. This API never uses the cohort or the called
    boundaries as evidence. They can define a domain in the caller, but that
    outcome-dependent selection must be disclosed when interpreting scores.
    """
    if not isinstance(start, Integral) or not isinstance(end, Integral) or end < start:
        raise ValueError("Integer half-open domain with end >= start required")
    positions_a, values_a = _observation(pos_a, hits_a, pa_a, pp_a, start, end)
    positions_b, values_b = _observation(pos_b, hits_b, pa_b, pp_b, start, end)
    union = np.union1d(positions_a, positions_b)
    values = np.zeros((2, len(union)), dtype=float)
    observed = np.zeros(values.shape, dtype=bool)
    ia, ib = np.searchsorted(union, positions_a), np.searchsorted(union, positions_b)
    values[0, ia], values[1, ib] = values_a, values_b
    observed[0, ia], observed[1, ib] = True, True
    batch = score_pairs(values, observed, [0], [1], [0], [len(union)], use_numba=use_numba)
    result = {key: value[0].item() for key, value in batch.items()}
    result["union_positions"] = union.tolist()
    result["model"] = MODEL_DESCRIPTION
    result["status"] = "uninformative_empty_domain" if not len(union) else "conditional_model_comparison"
    return result
