"""Nominate native-compatible LLR cohorts without creating singleton vetoes.

Only candidate geometry is formed here. Actual compatibility remains the pinned
native predictive classifier; shared-center sets are not family assignments.
"""
from collections import Counter
import hashlib
import numpy as np


def nominate(stratum, region, radius=2):
    from fiberhmm.inference.consensus.measurement_grouping import _calls
    from .reference.native_cell_consolidation import nominate_parents
    calls = _calls(stratum); units = stratum['units']; aliases = {}; unavailable = []
    for index, call in enumerate(calls):
        unit = units[call['unit_index']]
        positions = np.asarray(unit['positions'], np.int64)
        a, b = np.searchsorted(positions, [call['start'], call['end']])
        if a == b:
            unavailable.append(index); continue
        left = [region[0] if a == 0 else int(positions[a-1])+1, int(positions[a])]
        right = [int(positions[b-1])+1, region[1] if b == len(positions) else int(positions[b])]
        # Endpoint aliases are bounded by this read's MSP/nucleosome scaffold.
        for lo, hi in unit['eligible_intervals']:
            if lo <= call['start'] and call['end'] <= hi:
                left[0] = max(left[0], lo); right[1] = min(right[1], hi); break
        key = (call['start'], call['end'], *left, *right)
        aliases.setdefault(key, []).append(index)
    models = []; members = {}
    for key, indices in sorted(aliases.items()):
        identity = 'seed_' + hashlib.sha256(repr(key).encode()).hexdigest()[:16]
        models.append(dict(family=identity, reference_interval=list(key[:2]),
                           native_projection_cell=[list(key[2:4]), list(key[4:6])]))
        members[identity] = indices
    proposals = nominate_parents(models, radius)
    covered = {f for p in proposals for f in p['children']}
    groups = [p['children'] for p in proposals] + [[m['family']] for m in models if m['family'] not in covered]
    candidates = {}; provisional = []
    for children in groups:
        indices = sorted({i for child in children for i in members[child]})
        physical = {calls[i]['evidence_group_id'] for i in indices}
        if len(physical) < 2:
            provisional.extend(indices); continue
        spans = np.asarray([[calls[i]['start'], calls[i]['end']] for i in indices])
        median = np.median(spans, axis=0)
        representative = min(map(tuple, spans), key=lambda iv: (float(np.square(np.asarray(iv)-median).sum()), iv))
        # Deduplicate identical proposal coordinates, not overlapping populations.
        candidates.setdefault(representative, set()).update(indices)
    catalog = []
    for span, indices in sorted(candidates.items()):
        key = stratum['dataset_id']+'|'+repr(span)+'|common-native-cell-v1'
        catalog.append(dict(family=stratum['dataset_id']+':NC_'+hashlib.sha256(key.encode()).hexdigest()[:12],
                            consensus_start=int(span[0]), consensus_end=int(span[1]),
                            source_aliases=[dict(unit_id=calls[i]['unit_id'], ordinal=calls[i]['ordinal'],
                                                interval=[calls[i]['start'], calls[i]['end']]) for i in sorted(indices)],
                            nomination_source_units=len({calls[i]['evidence_group_id'] for i in indices}),
                            nomination_source_calls=len(indices),
                            nomination_provenance='common_native_edge_cell_with_actual_call_overlap'))
    return catalog, dict(proposals=len(catalog), raw_projection_groups=len(models),
        candidate_common_center_sets=len(proposals), nomination_radius_bp=radius,
        no_cohort_opportunity_call_indices=unavailable, provisional_single_source_indices=sorted(set(provisional)),
        family_count_cap=None, raw_footprints_added=0, final_memberships_set_by_native_predictive_MC=True,
        transitive_components_used_as_families=False, minimum_nomination_molecules=2)
