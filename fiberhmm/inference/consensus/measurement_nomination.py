"""Residual CR proposal nomination without a bp box or a support-count cutoff.

This is a continuation of an initial catalog, not a claim of independent
genome-wide discovery. Unsupported EXISTING source calls can nominate a new
shape; they cannot create a new footprint. Every exact cohort-projection class
is retained. A new one-unit proposal is explicitly provisional: it cannot supply
independent evidence to classify its own source unit.
"""
from __future__ import annotations

import hashlib
import numpy as np


def augment_catalog(catalog, result, positions, *, dataset_id, region,
                    nomination_reference_percent=99.9):
    """Freeze proposals once, independently of subsequent display stringency.

    The predeclared most-generous predictive reference is used for residual
    nomination. Slider changes do NOT call this function again. Opportunity
    equivalence uses the COMPLETE outcome-free cohort lattice, including both
    observable strands when applicable. Raw/SR source aliases remain traceable.
    """
    positions = np.asarray(positions, np.int64)
    if positions.ndim != 1 or np.any(np.diff(positions) <= 0):
        raise ValueError('Strictly ordered complete-cohort lattice required')
    if not 0 < nomination_reference_percent < 100:
        raise ValueError('A finite predeclared nomination reference required')
    output = [dict(f) for f in catalog]
    existing = {tuple(np.searchsorted(positions, [f['consensus_start'], f['consensus_end']]))
                for f in catalog}
    groups = {}; unavailable = []; alpha = 1-nomination_reference_percent/100.
    for i, (call, evidence) in enumerate(zip(result['calls'], result['call_family_evidence'])):
        if call['start'] >= region[1] or call['end'] <= region[0]:
            continue
        if any(s['status'] == 'scored' and s.get('predictive_tail_interval', [0., 0.])[1] >= alpha for s in evidence):
            continue
        projection = tuple(np.searchsorted(positions, [call['start'], call['end']]))
        if projection[0] == projection[1]:
            unavailable.append(i); continue
        if projection not in existing:
            groups.setdefault(projection, []).append(i)
    additions = []
    for projection, members in sorted(groups.items()):
        spans = np.asarray([[result['calls'][i]['start'], result['calls'][i]['end']] for i in members])
        # Representative actual source span, never an intersection chimera.
        center = np.median(spans, axis=0)
        representative = min(map(tuple, spans), key=lambda v: (float(np.square(v-center).sum()), v))
        aliases = [dict(unit_id=result['calls'][i]['unit_id'], ordinal=result['calls'][i]['ordinal'],
                        interval=[result['calls'][i]['start'], result['calls'][i]['end']]) for i in members]
        units = {result['calls'][i].get('evidence_group_id', result['calls'][i]['unit_id']) for i in members}
        identity = f'{dataset_id}|{positions[projection[0]]}|{positions[projection[1]-1]}|residual-native-v1'
        proposal = dict(family=dataset_id+':RN_'+hashlib.sha256(identity.encode()).hexdigest()[:12],
            consensus_start=int(representative[0]), consensus_end=int(representative[1]),
            nomination_provenance='unexplained_existing_call_projection', source_aliases=aliases,
            nomination_source_units=len(units), nomination_source_calls=len(members),
            provisional_single_source=len(units) == 1, cohort_projection=list(map(int, projection)))
        output.append(proposal); additions.append(proposal)
    return output, dict(initial_proposals=len(catalog), added_proposals=len(additions), additions=additions,
        nomination_reference_percent=nomination_reference_percent,
        no_cohort_opportunity_call_indices=unavailable,
        nomination_support_floor=1, exact_cohort_projection_deduplication=True,
        raw_footprints_added=0, family_count_cap=None, display_threshold_independent=True)
