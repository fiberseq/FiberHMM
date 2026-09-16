"""Native shared-boundary compatibility, independent of assay and stage.

The statistic is a profile likelihood loss for one shared contiguous protected
interval versus separately fitted intervals, on one common observation domain.
It is NOT a Bayes factor, posterior, p-value or calibrated error probability.
No population frequencies or arbitrary counts of possible geometries enter.

Existing calls nominate the domain/overlap constraints, not likelihood bonuses.
Each observation retains its own emission probabilities. Missing data are
neutral. The minimum bp tolerance is a separately recorded operational floor:
it never bounds the native candidate search or modifies the native statistic.
This primitive alone does not establish protection, rescue, class identity,
population recurrence, or comparative testability.
"""
from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
import math

import numpy as np
from numba import njit


@dataclass(frozen=True)
class FootprintObservation:
    positions: np.ndarray
    hits: np.ndarray
    p_accessible: np.ndarray
    p_protected: np.ndarray
    interval: tuple[int, int]
    observation_id: str = ''

    def __post_init__(self):
        pos = np.asarray(self.positions)
        hits = np.asarray(self.hits)
        interval = np.asarray(self.interval)
        if pos.ndim != 1 or (pos.size and pos.dtype.kind not in 'iu') or np.any(np.diff(pos) <= 0):
            raise ValueError('Strictly increasing integer opportunity positions required')
        if hits.shape != pos.shape or np.any((hits != 0) & (hits != 1)):
            raise ValueError('One binary observation required per opportunity')
        if interval.shape != (2,) or interval.dtype.kind not in 'iu' or interval[0] >= interval[1]:
            raise ValueError('A positive-width integer footprint interval is required')
        for name in ('p_accessible', 'p_protected'):
            try:
                probabilities = np.broadcast_to(np.asarray(getattr(self, name), float), pos.shape)
            except ValueError as exc:
                raise ValueError('One probability per opportunity, or a scalar, required') from exc
            if np.any(~np.isfinite(probabilities)) or np.any((probabilities <= 0) | (probabilities >= 1)):
                raise ValueError('Native probabilities must be strictly between zero and one')
            object.__setattr__(self, name, probabilities.copy())
        object.__setattr__(self, 'positions', pos.astype(np.int64, copy=True))
        object.__setattr__(self, 'hits', hits.astype(np.int8, copy=True))
        object.__setattr__(self, 'interval', tuple(map(int, interval)))

    @property
    def steps(self):
        return np.where(self.hits,
                        np.log(self.p_protected / self.p_accessible),
                        np.log1p(-self.p_protected) - np.log1p(-self.p_accessible))


@njit(cache=True)
def _best_interval(values, anchor_start, anchor_end):
    """Exact maximum over nonempty intervals overlapping the anchor projection.

    Return opportunity-cut indices, not rounded coordinates. The common-state
    anchor uses max(starts)/min(ends), so even an empty shared cut range can be
    spanned without forcing an artificial tiny intersection core.
    """
    best = -np.inf
    best_a = best_b = -1
    pref = 0.
    minimum = np.inf
    minimum_index = -1
    for b in range(1, len(values) + 1):
        a = b - 1
        if a < anchor_end and pref < minimum:
            minimum = pref
            minimum_index = a
        pref += values[a]
        if b > anchor_start and minimum_index >= 0:
            score = pref - minimum
            if score > best:
                best, best_a, best_b = score, minimum_index, b
    return best, best_a, best_b


@njit(cache=True)
def _best_by_end(values, anchor_start, anchor_end):
    """Profile the start edge at every fixed end; same domain/anchor as above."""
    scores = np.full(len(values)+1, -np.inf)
    pref = 0.; minimum = np.inf
    for b in range(1, len(values)+1):
        a=b-1
        if a < anchor_end:
            minimum=min(minimum,pref)
        pref += values[a]
        if b > anchor_start:
            scores[b]=pref-minimum
    return scores


@njit(cache=True)
def _score_arrays(values, observed, unit_a, unit_b, left, right,
                  start_a, end_a, start_b, end_b):
    result = np.empty((len(unit_a), 11), dtype=np.float64)
    counts = np.zeros((len(unit_a), 6), dtype=np.int64)
    for pair in range(len(unit_a)):
        ia, ib = unit_a[pair], unit_b[pair]
        va = np.empty(right[pair] - left[pair])
        vb = np.empty_like(va)
        k = sa = ea = sb = eb = 0
        for j in range(left[pair], right[pair]):
            oa, ob = observed[ia, j], observed[ib, j]
            if not oa and not ob:
                continue
            sa += int(j < start_a[pair]); ea += int(j < end_a[pair])
            sb += int(j < start_b[pair]); eb += int(j < end_b[pair])
            va[k] = values[ia, j] if oa else 0.
            vb[k] = values[ib, j] if ob else 0.
            counts[pair, 0] += int(oa)
            counts[pair, 1] += int(ob)
            counts[pair, 2] += 1
            counts[pair, 3] += int(oa and ob)
            counts[pair, 4] += int(oa and va[k] != 0.)
            counts[pair, 5] += int(ob and vb[k] != 0.)
            k += 1
        if not k or sa == ea or sb == eb:
            result[pair, :] = np.nan
            continue
        ma, aa, ab = _best_interval(va[:k], sa, ea)
        mb, ba, bb = _best_interval(vb[:k], sb, eb)
        shared, ca, cb = _best_interval(va[:k] + vb[:k], max(sa, sb), min(ea, eb))
        loss = ma + mb - shared
        tolerance = 1e-10 * (1 + abs(ma) + abs(mb) + abs(shared))
        if loss < -tolerance:
            raise FloatingPointError('Shared optimum exceeds separate optima')
        result[pair, 0] = max(0., loss)
        result[pair, 1] = ma
        result[pair, 2] = mb
        result[pair, 3] = shared
        result[pair, 4] = aa
        result[pair, 5] = ab
        result[pair, 6] = ba
        result[pair, 7] = bb
        result[pair, 8] = ca * (k + 1) + cb
        shared_end = np.max(_best_by_end(va[:k],sa,ea)+_best_by_end(vb[:k],sb,eb))
        shared_start = np.max(_best_by_end(va[:k][::-1],k-ea,k-sa)+_best_by_end(vb[:k][::-1],k-eb,k-sb))
        result[pair, 9] = max(0.,ma+mb-shared_end)
        result[pair, 10] = max(0.,ma+mb-shared_start)
    return result, counts


def score_pairs(values, observed, unit_a, unit_b, domain_left, domain_right,
                start_a, end_a, start_b, end_b):
    """Batch API; all domains/anchors are cuts in one genomic opportunity grid.

    Candidate boundaries cover every identifiable nonempty pair-union interval.
    There is deliberately no bp-radius argument. Native profiles must not be
    truncated by a floor intended only to relax operational class distinctions.
    Returned geometry indices refer to the pair's observed UNION, not the
    cohort grid. Both-unobserved columns cannot change the statistic.
    """
    values = np.asarray(values, float)
    observed = np.asarray(observed)
    if values.ndim != 2 or observed.shape != values.shape or observed.dtype.kind != 'b':
        raise ValueError('Equal-shaped evidence and boolean observation matrices required')
    if np.any(~np.isfinite(values[observed])):
        raise ValueError('Finite native log likelihood ratios required at observed positions')
    indices = []
    for a in (unit_a, unit_b, domain_left, domain_right, start_a, end_a, start_b, end_b):
        v = np.asarray(a)
        if v.ndim != 1 or (v.size and v.dtype.kind not in 'iu'):
            raise ValueError('Integer index vectors required')
        indices.append(np.ascontiguousarray(v, dtype=np.int64))
    if len({len(v) for v in indices}) > 1:
        raise ValueError('One index of each kind required per pair')
    ua, ub, lo, hi, sa, ea, sb, eb = indices
    if np.any(ua < 0) or np.any(ub < 0) or np.any(ua >= len(values)) or np.any(ub >= len(values)):
        raise ValueError('Unit index outside evidence matrix')
    if (np.any(lo < 0) or np.any(hi < lo) or np.any(hi > values.shape[1])
            or np.any(sa < lo) or np.any(sb < lo) or np.any(ea > hi) or np.any(eb > hi)
            or np.any(ea < sa) or np.any(eb < sb)):
        raise ValueError('Anchors must lie inside the common comparison domain')
    rows, counts = _score_arrays(np.ascontiguousarray(values), np.ascontiguousarray(observed), *indices)
    result = {name: rows[:, i] for i, name in enumerate((
        'native_loss', 'best_a', 'best_b', 'best_shared', 'best_a_start',
        'best_a_end', 'best_b_start', 'best_b_end', 'shared_geometry_code',
        'left_edge_relaxed_loss', 'right_edge_relaxed_loss'))}
    result.update({name: counts[:, i] for i, name in enumerate((
        'n_a', 'n_b', 'n_union', 'n_shared', 'n_informative_a', 'n_informative_b'))})
    result['informative_both'] = np.isfinite(result['native_loss']) & (counts[:, 4] > 0) & (counts[:, 5] > 0)
    return result


def edge_floor_loss(native, left_relaxed, right_relaxed, left_discrepancy,
                    right_discrepancy, minimum_bp):
    """Apply the declared tolerance separately to each observed edge.

    When only one edge falls within the bp floor, the other edge still has to
    pass its full native profile comparison. This permits, for example, a 5-bp
    high-information left discrepancy plus a naturally unresolvable 40-bp right
    discrepancy at X=10. A BOTH-edges-within-X shortcut incorrectly rejects it.
    The floor relaxes correspondence, not native probabilities. Its score must
    never be presented as the original shared-state likelihood statistic.
    """
    out=np.asarray(native,float).copy()
    if minimum_bp > 0:
        left=np.asarray(left_discrepancy)<=minimum_bp
        right=np.asarray(right_discrepancy)<=minimum_bp
        out=np.where(left,np.minimum(out,left_relaxed),out)
        out=np.where(right,np.minimum(out,right_relaxed),out)
        out=np.where(left & right,0.,out)
    return out


def _adequacy(observation, start, end):
    keep = (observation.positions >= start) & (observation.positions < end)
    return {'opportunities': int(keep.sum()),
            'hits': int(observation.hits[keep].sum()),
            'protected_vs_accessible_log_lr': float(observation.steps[keep].sum())}


def compare(a, b, *, loss_odds=100., minimum_edge_tolerance_bp=0,
            core_contradiction_odds=100.):
    """Stage-neutral pair diagnostic with an explicit, non-statistical bp floor.

    Domain selection is caller-conditioned. Neither an uninformative match nor
    a floor override is positive evidence for a footprint or a shared family.
    The core check is a native LR adequacy firewall, not a calibrated test.
    """
    if not isinstance(a, FootprintObservation) or not isinstance(b, FootprintObservation):
        raise TypeError('Two native FootprintObservation objects required')
    if not math.isfinite(loss_odds) or loss_odds < 1 or not math.isfinite(core_contradiction_odds) or core_contradiction_odds < 1:
        raise ValueError('Finite likelihood-loss odds of at least one required')
    if isinstance(minimum_edge_tolerance_bp, bool) or not isinstance(minimum_edge_tolerance_bp, Integral) or minimum_edge_tolerance_bp < 0:
        raise ValueError('Nonnegative integer edge-discrepancy floor required')
    start = min(a.interval[0], b.interval[0]); end = max(a.interval[1], b.interval[1])
    overlap_start = max(a.interval[0], b.interval[0]); overlap_end = min(a.interval[1], b.interval[1])
    if overlap_start >= overlap_end:
        return dict(status='nonoverlapping', compatible=False, native_compatible=False,
                    edge_floor_compatible=False, native_loss=None, domain=[start, end])
    positions = np.union1d(a.positions[(a.positions >= start) & (a.positions < end)],
                           b.positions[(b.positions >= start) & (b.positions < end)])
    values = np.zeros((2, len(positions))); observed = np.zeros(values.shape, bool)
    for i, o in enumerate((a, b)):
        keep = (o.positions >= start) & (o.positions < end)
        cols = np.searchsorted(positions, o.positions[keep])
        values[i, cols] = o.steps[keep]; observed[i, cols] = True
    sa, ea = np.searchsorted(positions, a.interval)
    sb, eb = np.searchsorted(positions, b.interval)
    scored = score_pairs(values, observed, [0], [1], [0], [len(positions)], [sa], [ea], [sb], [eb])
    usable = bool(scored['informative_both'][0])
    loss = float(scored['native_loss'][0]) if usable else None
    core_a = _adequacy(a, overlap_start, overlap_end)
    core_b = _adequacy(b, overlap_start, overlap_end)
    contradicted = any(c['opportunities'] and c['protected_vs_accessible_log_lr'] < -math.log(core_contradiction_odds)
                       for c in (core_a, core_b))
    core_observed_both = all(c['opportunities'] > 0 for c in (core_a, core_b))
    edge_discrepancies = np.abs(np.asarray(a.interval) - np.asarray(b.interval)).tolist()
    adjusted=float(edge_floor_loss(scored['native_loss'],scored['left_edge_relaxed_loss'],
        scored['right_edge_relaxed_loss'],[edge_discrepancies[0]],[edge_discrepancies[1]],
        minimum_edge_tolerance_bp)[0]) if usable else None
    if not core_observed_both or contradicted:
        adjusted=loss
    floor = bool(usable and minimum_edge_tolerance_bp > 0 and adjusted <= math.log(loss_odds)+1e-10
                 and adjusted < loss-1e-10 and not contradicted)
    native = bool(usable and loss <= math.log(loss_odds) + 1e-10)
    # Contradicted central protection is not rehabilitated by small edge loss.
    # A provisional raw label can survive downstream; this is not shared support.
    compatible = (native or floor) and not contradicted
    status = ('core_contradicted' if contradicted else 'measurement_unavailable' if not usable
              else 'native_compatible' if native else 'edge_floor_compatible' if floor else 'distinguishable')
    return dict(status=status, compatible=bool(compatible), native_compatible=native,
                edge_floor_compatible=floor, native_loss=loss,
                floor_adjusted_loss=adjusted,
                loss_allowance=math.log(loss_odds), minimum_edge_tolerance_bp=int(minimum_edge_tolerance_bp),
                edge_discrepancies=edge_discrepancies, domain=[start, end],
                core_a=core_a, core_b=core_b, core_contradicted=bool(contradicted),
                n_a=int(scored['n_a'][0]), n_b=int(scored['n_b'][0]),
                n_union=int(scored['n_union'][0]), n_shared=int(scored['n_shared'][0]),
                best_a=float(scored['best_a'][0]) if usable else None,
                best_b=float(scored['best_b'][0]) if usable else None,
                best_shared=float(scored['best_shared'][0]) if usable else None,
                semantics='caller-conditioned profile likelihood loss; floor is operational; not posterior/p/q/FDR')
