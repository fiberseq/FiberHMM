"""Caller-conditioned classification against frozen population proposals.

This module does NOT estimate footprint existence, occupancy, a posterior, or
an FDR. The native pair ledger supplies empirical compatibility with independent
source observations. Endpoint distance nominates witnesses and chooses a
primary display label among compatible alternatives; it is not likelihood.
The strict complete-link diagnostic remains available in measurement_grouping.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import math

import numpy as np


def classify_catalog(calls, catalog, pairs, *, loss_odds_levels=(10., 100., 1000.),
                     witness_quantile=.5, region=None):
    """Keep every in-region input call exactly once, including unresolved calls.

    Witness homes are frozen before looking at thresholds. Each candidate family
    is tested against ALL overlapping independent source units available in the
    pair ledger; no P50-selected subset or class-frequency prior is used. One
    source unit contributes at most once per recipient/family. The user-selected
    quantile is a robust population compatibility rule, NOT confidence.

    If no recurrent catalog proposal is compatible, an explicitly provisional
    exact-span label preserves the input. Such labels do not establish recurrence
    or new footprint evidence. Canonical intervals are never painted over a
    source call or used as a second positive-LR detection gate.
    """
    if not 0 < witness_quantile <= 1 or not math.isfinite(witness_quantile):
        raise ValueError('Witness quantile must lie in (0, 1]')
    if not loss_odds_levels or any(not math.isfinite(v) or v < 1 for v in loss_odds_levels):
        raise ValueError('Finite likelihood-loss odds of at least one required')
    n, nf = len(calls), len(catalog)
    if not nf:
        raise ValueError('A frozen native proposal catalog is required')
    centers = np.asarray([[f['consensus_start'], f['consensus_end']] for f in catalog])
    if np.any(centers[:, 0] >= centers[:, 1]):
        raise ValueError('Nonpositive family interval')
    ids = [f['family'] for f in catalog]
    if len(set(ids)) != nf:
        raise ValueError('Duplicate family identities')
    spans = np.asarray([[c['start'], c['end']] for c in calls]).reshape(-1, 2)
    active = np.ones(n, bool) if region is None else ((spans[:, 0] < region[1]) & (spans[:, 1] > region[0]))
    homes = np.full(n, -1, np.int32)
    # Source provenance only. No assignment posterior enters witness nomination.
    # Ties use immutable family IDs, independent of input catalog order.
    for i in np.flatnonzero(active):
        eligible = np.flatnonzero((centers[:, 0] < spans[i, 1]) & (centers[:, 1] > spans[i, 0]))
        if len(eligible):
            homes[i] = min(eligible, key=lambda f: (float(np.square(centers[f]-spans[i]).sum()), ids[f]))

    first, second = np.asarray(pairs['first']), np.asarray(pairs['second'])
    if (np.any(first < 0) or np.any(second >= n) or np.any(first >= second)):
        raise ValueError('Invalid pair ledger indices')
    loss = np.asarray(pairs['native_loss'], float)
    informative = np.asarray(pairs['informative_both'], bool)
    veto = np.asarray(pairs['core_contradicted'], bool)
    floor = np.asarray(pairs['edge_floor_compatible'], bool)
    if any(v.shape != first.shape for v in (second, loss, informative, veto, floor)):
        raise ValueError('Pair ledger shapes differ')
    if np.any(~np.isfinite(loss[informative])) or np.any(loss[informative] < 0):
        raise ValueError('Informative pair records require nonnegative finite native losses')
    source_group = np.asarray([c.get('evidence_group_id', c['unit_id']) for c in calls])
    valid = informative & active[first] & active[second] & (source_group[first] != source_group[second])
    # A contradicted pair counts as a failed comparison, not as missing data.
    adjusted = np.asarray(pairs['floor_adjusted_loss'],float) if 'floor_adjusted_loss' in pairs else np.where(floor,0.,loss)
    effective = np.where(veto, np.inf, adjusted)
    aa, bb = first[valid], second[valid]
    recipient = np.r_[aa, bb]
    witness = np.r_[bb, aa]
    native = np.r_[loss[valid], loss[valid]]
    eff = np.r_[effective[valid], effective[valid]]
    via_floor = np.r_[floor[valid] & ~veto[valid], floor[valid] & ~veto[valid]]
    bad = np.r_[veto[valid], veto[valid]]
    has_home = homes[witness] >= 0
    recipient, witness, native, eff, via_floor, bad = [v[has_home] for v in
        (recipient, witness, native, eff, via_floor, bad)]
    _, group_indices = np.unique(source_group, return_inverse=True)
    key = recipient.astype(np.int64)*nf + homes[witness]
    geometry = np.square(spans[recipient]-spans[witness]).sum(1)
    # Choose the geometrically closest occurrence from an independent source
    # unit, if that unit has multiple calls nominated to this family. Do not
    # select its best likelihood or count multiple copies as recurrence.
    order = np.lexsort((witness, geometry, group_indices[witness], key))
    sorted_key, sorted_groups = key[order], group_indices[witness[order]]
    unique = np.r_[True, (sorted_key[1:] != sorted_key[:-1]) | (sorted_groups[1:] != sorted_groups[:-1])] if len(key) else np.empty(0, bool)
    order = order[unique]; key = key[order]
    recipient, witness, native, eff, via_floor, bad = [v[order] for v in
        (recipient, witness, native, eff, via_floor, bad)]
    boundaries = np.r_[0, np.flatnonzero(key[1:] != key[:-1])+1, len(key)]
    summaries = [[] for _ in calls]
    for lo, hi in zip(boundaries[:-1], boundaries[1:]):
        if lo == hi:
            continue
        i, f = divmod(int(key[lo]), nf)
        kth = min(hi-lo-1, max(0, int(math.ceil(witness_quantile*(hi-lo)))-1))
        value = float(np.partition(eff[lo:hi], kth)[kth])
        nvalue = float(np.partition(native[lo:hi], kth)[kth])
        summaries[i].append(dict(family=ids[f], family_index=f,
            effective_loss_quantile=value if math.isfinite(value) else None,
            native_loss_quantile=nvalue, eligible_source_units=int(hi-lo),
            core_contradicted_source_units=int(bad[lo:hi].sum()),
            floor_compatible_source_units=int(via_floor[lo:hi].sum()),
            geometry_distance_sq=int(np.square(centers[f]-spans[i]).sum())))

    partitions = {}
    for odds in sorted(set(map(float, loss_odds_levels))):
        assignments = []; members = {}; provisional = {}
        for i in np.flatnonzero(active):
            compatible = sorted([s for s in summaries[i] if s['effective_loss_quantile'] is not None
                and s['effective_loss_quantile'] <= math.log(odds)+1e-10],
                key=lambda s: (s['geometry_distance_sq'], s['effective_loss_quantile'], s['family']))
            if compatible:
                primary = compatible[0]; fid = primary['family']; status = 'compatible_catalog_label'
            else:
                primary = None
                ident = f'{spans[i,0]}|{spans[i,1]}'
                fid = ids[0].split(':')[0]+':unresolved_'+hashlib.sha256(ident.encode()).hexdigest()[:12]
                provisional[fid] = spans[i].tolist(); status = 'provisional_unresolved'
            assignments.append(dict(**calls[i], interval=spans[i].tolist(), family=fid,
                classification_status=status, primary_evidence=primary,
                compatible_alternatives=[s['family'] for s in compatible[1:]],
                source_geometry_home=ids[homes[i]] if homes[i] >= 0 else None))
            members.setdefault(fid, []).append(int(i))
        out_catalog = []
        by_id = {f['family']: f for f in catalog}
        for fid, mm in sorted(members.items()):
            is_provisional = fid in provisional
            interval = provisional[fid] if is_provisional else [by_id[fid]['consensus_start'], by_id[fid]['consensus_end']]
            out_catalog.append(dict(family=fid, representative_interval=interval,
                source_calls=len(mm), calls_by_strand=dict(Counter(calls[i]['strand'] for i in mm)),
                units_by_strand={s: len({calls[i]['unit_id'] for i in mm if calls[i]['strand']==s})
                                 for s in sorted({calls[i]['strand'] for i in mm})},
                member_indices=mm, provisional_unresolved=is_provisional,
                provisional_singleton=len(mm)==1, native_proposal_preserved=not is_provisional))
        partitions[str(odds)] = dict(catalog=out_catalog, assignments=assignments, loss_odds=odds,
            witness_quantile=float(witness_quantile),
            classes=len(out_catalog), unresolved_calls=sum(a['classification_status']=='provisional_unresolved' for a in assignments),
            ambiguous_calls=sum(bool(a['compatible_alternatives']) for a in assignments),
            all_source_calls_retained=len(assignments)==int(active.sum()), boundaries_changed=False)
    return dict(status='complete', calls=calls, partitions=partitions,
        call_family_evidence=summaries, source_homes=[ids[f] if f>=0 else None for f in homes],
        diagnostics=dict(source_calls=int(active.sum()), out_of_region_calls=int((~active).sum()),
            witness_quantile=float(witness_quantile), original_catalog_families=nf,
            class_frequency_prior=False, absolute_count_filter=False, hard_complete_link=False,
            calibrated_confidence=False, raw_call_geometry_used='Witness nomination and primary display tie policy only',
            probabilities_changed=False, boundaries_changed=False))
