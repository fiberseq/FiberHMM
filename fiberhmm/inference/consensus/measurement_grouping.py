"""Classify every existing footprint using native pairwise measurement loss.

This is an inspectable replacement-classification engine, not the old bounded
configuration decoder with a new score attached. It never deletes, splits,
moves, or invents an input footprint. Hierarchical classes are operational
complete-link groups, not joint posterior probabilities or biological IDs.
The production SR/rescue/XCR adapters must explicitly consume this contract;
old configuration-model caches are not interchangeable with these records.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import math

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage

from .measurement_compatibility import score_pairs, edge_floor_loss


def _calls(stratum):
    result = []
    seen = set()
    for m, u in enumerate(stratum['units']):
        if u['unit_id'] in seen:
            raise ValueError('Evidence units must be unique before grouping')
        seen.add(u['unit_id'])
        for ordinal, interval in enumerate(u['representative_raw_tf_intervals']):
            a, b = interval
            if not isinstance(a, (int, np.integer)) or not isinstance(b, (int, np.integer)) or a >= b:
                raise ValueError('Existing calls must be positive-width integer intervals')
            result.append(dict(unit_index=m, unit_id=u['unit_id'], ordinal=ordinal,
                               start=int(a), end=int(b), strand=u['strand'],
                               evidence_group_id=u.get('fold_group_id', u['unit_id'])))
    return sorted(result, key=lambda c: (c['start'], c['end'], c['unit_id'], c['ordinal']))


def _components(starts, ends):
    result = []
    lo = 0; hi = -1
    for j, (a, b) in enumerate(zip(starts, ends)):
        if j and a >= hi:
            result.append((lo, j)); lo = j
        hi = max(hi, b)
    if len(starts):
        result.append((lo, len(starts)))
    return result


def group_observations(stratum, *, loss_odds_levels=(10., 100., 1000.),
                       minimum_edge_tolerance_bp=0, core_contradiction_odds=100.,
                       maximum_matrix_bytes=2 * 1024**3, batch_size=200000,
                       progress=None, comparison_only=False):
    """All overlapping independent call pairs, followed by nested grouping.

    Source calls may be frozen SR-normalized observations. Their geometry is
    retained verbatim. The footprint-call-conditioned full pair-union domain
    is scored with each observation's native emissions. Thresholds are native
    likelihood-loss allowances, NOT measured false-split confidence levels.
    """
    if not loss_odds_levels or any(not math.isfinite(v) or v < 1 for v in loss_odds_levels):
        raise ValueError('Finite likelihood-loss odds of at least one required')
    if isinstance(minimum_edge_tolerance_bp, bool) or not isinstance(minimum_edge_tolerance_bp, (int, np.integer)) or minimum_edge_tolerance_bp < 0:
        raise ValueError('Nonnegative integer minimum edge tolerance required')
    if not math.isfinite(core_contradiction_odds) or core_contradiction_odds < 1:
        raise ValueError('Finite core contradiction odds required')
    if batch_size < 1 or maximum_matrix_bytes < 1:
        raise ValueError('Positive compute budgets required')
    progress = progress or (lambda message: None)
    calls = _calls(stratum); n = len(calls); units = stratum['units']
    if not calls:
        return dict(status='empty', calls=[], partitions={}, pairs={}, diagnostics={})
    starts = np.array([c['start'] for c in calls], dtype=np.int64)
    ends = np.array([c['end'] for c in calls], dtype=np.int64)
    unit_indices = np.array([c['unit_index'] for c in calls], dtype=np.int64)
    folds = np.array([u.get('fold_group_id', u['unit_id']) for u in units])
    components = _components(starts, ends)
    pair_counts = np.searchsorted(starts, ends, side='left') - np.arange(n) - 1
    pair_total = int(pair_counts.sum())
    grid = np.unique(np.concatenate([np.asarray(u['positions'], dtype=np.int64) for u in units]))
    grid = grid[(grid >= starts.min()) & (grid < ends.max())]
    largest_cells = 0 if comparison_only else max((b-a)*(b-a-1)//2 for a, b in components)
    # Includes two condensed arrays for SciPy's working copy, observation
    # prefixes/masks, retained pair arrays and a bounded scoring batch. Payload
    # objects and Python metadata are additional process RSS, not hidden reads.
    estimated = (len(units) * len(grid) * 22 + pair_total * 48
                 + largest_cells * 16 + min(batch_size, pair_total) * 160)
    if estimated > maximum_matrix_bytes:
        raise MemoryError(f'Exact grouping requires approximately {estimated} matrix bytes; '
                          f'budget {maximum_matrix_bytes}. No reads/classes were omitted.')
    progress(f'{n:,} existing calls; {pair_total:,} overlapping pairs before same-unit exclusion')
    first = np.repeat(np.arange(n, dtype=np.int64), pair_counts)
    second = np.empty(pair_total, dtype=np.int64)
    offset = 0
    for i, count in enumerate(pair_counts):
        second[offset:offset+count] = np.arange(i+1, i+1+count)
        offset += count
    independent = folds[unit_indices[first]] != folds[unit_indices[second]]
    excluded = int((~independent).sum())
    first, second = first[independent], second[independent]
    del independent
    values = np.zeros((len(units), len(grid)))
    observed = np.zeros(values.shape, bool)
    for m, u in enumerate(units):
        positions = np.asarray(u['positions'])
        hits = np.asarray(u['hits'])
        pa, pp = np.asarray(u['p_accessible'], float), np.asarray(u['p_protected'], float)
        if not (positions.shape == hits.shape == pa.shape == pp.shape) or np.any(np.diff(positions) <= 0):
            raise ValueError('Complete ordered native opportunities and probabilities required')
        if (np.any((hits != 0) & (hits != 1)) or np.any(~np.isfinite(pa)) or np.any(~np.isfinite(pp))
                or np.any((pa <= 0) | (pa >= 1) | (pp <= 0) | (pp >= 1))):
            raise ValueError('Valid binary observations and strictly interior probabilities required')
        keep = (positions >= starts.min()) & (positions < ends.max())
        columns = np.searchsorted(grid, positions[keep])
        h, a, p = hits[keep], pa[keep], pp[keep]
        values[m, columns] = np.where(h, np.log(p/a), np.log1p(-p)-np.log1p(-a))
        observed[m, columns] = True
    pref = np.c_[np.zeros(len(units)), np.cumsum(values, axis=1)]
    opref = np.c_[np.zeros(len(units), dtype=np.int32), np.cumsum(observed, axis=1, dtype=np.int32)]
    ca, cb = np.searchsorted(grid, starts), np.searchsorted(grid, ends)
    native_loss = np.empty(len(first)); floor = np.zeros(len(first), bool)
    adjusted_loss=np.empty(len(first))
    contradicted = np.zeros(len(first), bool); informative = np.zeros(len(first), bool)
    for lo in range(0, len(first), batch_size):
        hi = min(len(first), lo+batch_size); i, j = first[lo:hi], second[lo:hi]
        ua, ub = unit_indices[i], unit_indices[j]
        score = score_pairs(values, observed, ua, ub, np.minimum(ca[i], ca[j]),
                            np.maximum(cb[i], cb[j]), ca[i], cb[i], ca[j], cb[j])
        native_loss[lo:hi] = score['native_loss']
        informative[lo:hi] = score['informative_both']
        a, b = np.maximum(ca[i], ca[j]), np.minimum(cb[i], cb[j])
        core_a, core_b = pref[ua, b]-pref[ua, a], pref[ub, b]-pref[ub, a]
        op_a, op_b = opref[ua, b]-opref[ua, a], opref[ub, b]-opref[ub, a]
        bad = ((op_a > 0) & (core_a < -math.log(core_contradiction_odds))
               | (op_b > 0) & (core_b < -math.log(core_contradiction_odds)))
        contradicted[lo:hi] = bad
        adjusted=edge_floor_loss(score['native_loss'],score['left_edge_relaxed_loss'],
            score['right_edge_relaxed_loss'],np.abs(starts[i]-starts[j]),np.abs(ends[i]-ends[j]),
            minimum_edge_tolerance_bp)
        adjusted_loss[lo:hi]=np.where((op_a>0)&(op_b>0)&~bad,adjusted,score['native_loss'])
        floor[lo:hi]=adjusted_loss[lo:hi]<score['native_loss']-1e-10
        progress(f'Scored {hi:,}/{len(first):,} pairs using native models')
    del values, observed, pref, opref
    pair_ledger=dict(first=first,second=second,native_loss=native_loss,
        floor_adjusted_loss=adjusted_loss,edge_floor_compatible=floor,
        core_contradicted=contradicted,informative_both=informative)
    if comparison_only:
        return dict(status='complete',calls=calls,partitions={},pairs=pair_ledger,
            diagnostics=dict(source_units=len(units),source_calls=n,tested_pairs=len(first),
                minimum_edge_tolerance_bp=int(minimum_edge_tolerance_bp),
                core_contradiction_odds=core_contradiction_odds,
                same_evidence_group_pairs_excluded=excluded,
                estimated_matrix_bytes=estimated,comparison_only=True,
                statistic='native shared-state profile loss and separate per-edge tolerance score; not calibrated confidence'))
    effective = adjusted_loss.copy()
    unusable = ~informative | contradicted | ~np.isfinite(effective)
    sentinel = max(1e6, max(math.log(v) for v in loss_odds_levels) + 100)
    effective[unusable] = sentinel
    levels = sorted(set(float(v) for v in loss_odds_levels))
    labels = {v: np.zeros(n, dtype=np.int64) for v in levels}
    next_label = dict.fromkeys(levels, 0)
    for component, (lo, hi) in enumerate(components):
        size = hi-lo
        if size == 1:
            for v in levels:
                next_label[v] += 1; labels[v][lo] = next_label[v]
            continue
        p, q = np.searchsorted(first, [lo, hi])
        i, j = first[p:q]-lo, second[p:q]-lo
        if np.any(j >= size):
            raise AssertionError('A pair crossed a true nonoverlap component boundary')
        distance = np.full(size*(size-1)//2, sentinel)
        indices = size*i-i*(i+1)//2+j-i-1
        # Only resolves exact/effectively exact evidence ties, so a broad call
        # preferentially joins similarly shaped calls rather than whichever
        # read ID happens to occur first. Not a native likelihood reward.
        geom = np.maximum(np.abs(starts[first[p:q]]-starts[second[p:q]]),
                          np.abs(ends[first[p:q]]-ends[second[p:q]]))
        tie = 1e-10 * geom / max(1, int(ends[lo:hi].max()-starts[lo]))
        distance[indices] = effective[p:q] + tie
        progress(f'Grouping component {component+1}/{len(components)}: all {size:,} calls')
        tree = linkage(distance, method='complete', optimal_ordering=False)
        del distance
        for v in levels:
            local = fcluster(tree, math.log(v) + 2e-10, criterion='distance')
            labels[v][lo:hi] = local + next_label[v]
            next_label[v] += int(local.max())
        del tree
    partitions = {}
    previous = None
    for level in levels:
        lab = labels[level]
        same = lab[first] == lab[second]
        if np.any(unusable[same]) or np.any(effective[same] > math.log(level) + 2e-10):
            raise AssertionError('A class contains a contradicted or distinguished pair')
        sizes = np.bincount(lab)
        if int((sizes*(sizes-1)//2).sum()) != int(same.sum()):
            raise AssertionError('A class contains an untested/nonoverlapping/same-unit pair')
        if previous is not None:
            for old in np.unique(previous):
                if len(np.unique(lab[previous == old])) != 1:
                    raise AssertionError('Loss-threshold partitions are not nested')
        previous = lab
        catalog = []; lookup = {}; maximum_loss = np.zeros(int(lab.max())+1)
        np.maximum.at(maximum_loss, lab[first[same]], native_loss[same])
        for index in np.unique(lab):
            members = np.flatnonzero(lab == index)
            identity = '\n'.join(sorted(f'{calls[k]["unit_id"]}|{calls[k]["ordinal"]}' for k in members))
            fid = stratum['dataset_id'] + ':MC_' + hashlib.sha256(identity.encode()).hexdigest()[:16]
            spans = np.array([[calls[k]['start'], calls[k]['end']] for k in members])
            center = np.median(spans, axis=0)
            representative = spans[np.argmin(np.abs(spans-center).sum(axis=1))].tolist()
            catalog.append(dict(family=fid, representative_interval=representative,
                                source_calls=len(members), units_by_strand=dict(Counter(calls[k]['strand'] for k in members)),
                                maximum_within_native_loss=float(maximum_loss[index]),
                                provisional_singleton=len(members) == 1,
                                member_indices=members.tolist()))
            lookup[int(index)] = fid
        assignments = [{**c, 'interval': [c['start'], c['end']], 'family': lookup[int(lab[k])]}
                       for k, c in enumerate(calls)]
        partitions[str(level)] = dict(catalog=catalog, assignments=assignments,
            loss_odds=level, classes=len(catalog), singleton_classes=sum(c['provisional_singleton'] for c in catalog),
            within_class_pairs=int(same.sum()), within_class_floor_pairs=int((same & floor).sum()),
            all_source_calls_retained=len(assignments) == n, boundaries_changed=False)
    return dict(status='complete', calls=calls, partitions=partitions,
        pairs=pair_ledger,
        diagnostics=dict(source_units=len(units), source_calls=n, tested_pairs=len(first),
                         same_evidence_group_pairs_excluded=excluded,
                         components=[b-a for a, b in components], estimated_matrix_bytes=estimated,
                         minimum_edge_tolerance_bp=int(minimum_edge_tolerance_bp),
                         core_contradiction_odds=core_contradiction_odds,
                         probabilities_changed=False, call_count_cap=None,
                         read_count_cap=None, family_count_cap=None,
                         clustering='deterministic complete-link; geometry resolves native-loss ties only',
                         statistic='native profile likelihood loss, NOT Bayes factor/posterior/p/q/FDR'))
