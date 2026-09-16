# Extracted reference kernels; see SOURCE_MANIFEST.json.
from collections import Counter, defaultdict
import hashlib
import numpy as np
from .bounded_parent_prototype import prepare_events, score_event
from .run_bounded_parent_panel import call_key

def overlaps(a, b):
    return max(a[0], b[0]) < min(a[1], b[1])

def reindex_fit(fit, old_region, new_region):
    """Transport exactly the same physical mass to another integer grid.

    Recipient context may expand to retain its whole called interval. No refit,
    clipping, extra physical tolerance, or recipient-dependent mass is introduced.
    """
    if fit.get('status') != 'fitted' or list(old_region) == list(new_region):
        return fit
    old_n = old_region[1] - old_region[0] + 1
    (li, ri) = np.triu_indices(old_n, 1)
    cc = np.asarray(fit['columns'], int)
    left = li[cc] + old_region[0] - new_region[0]
    right = ri[cc] + old_region[0] - new_region[0]
    n = new_region[1] - new_region[0] + 1
    if np.any(left < 0) or np.any(right >= n):
        raise ValueError('Recipient domain would truncate fitted parent support')
    indices = left * (2 * n - left - 1) // 2 + right - left - 1
    return dict(fit, columns=indices)

def extend_parent(case, result, region, maximum_bytes=4 * 1024 ** 3):
    """Score all additional actual-overlap events, with physical-group exclusion.

    Source scores remain unchanged. A second event on a training molecule uses
    that molecule's excluded-fold model, never the full model that saw it.
    """
    result = dict(result)
    records = [dict(r, comparison_scope='source_cohort', scoring_region=region) for r in result['records']]
    full = result['full_model']
    known = {call_key(r['call']) for r in records}
    group_folds = {}
    for r in records:
        group = r['call'].get('evidence_group_id', r['call']['unit_id'])
        fold = r['fold']
        if group in group_folds and group_folds[group] != fold:
            raise ValueError('Physical group has inconsistent source fold labels')
        group_folds[group] = fold
    if full.get('status') == 'fitted':
        for call in case['calls']:
            if call_key(call) in known or not overlaps([call['start'], call['end']], full['anchor']):
                continue
            group = call.get('evidence_group_id', call['unit_id'])
            fold = group_folds.get(group, 'external_group')
            model = result['fold_models'].get(str(fold), full)
            if group in model.get('training_groups', []):
                raise ValueError('Recipient physical molecule leaked into parent fit')
            expanded = [min(region[0], call['start']), max(region[1], call['end'])]
            if expanded[0] < case['source_extent'][0] or expanded[1] > case['source_extent'][1]:
                raise ValueError('Eligible event outside original source extent')
            data = prepare_events(case['units'], [call], expanded, maximum_bytes=maximum_bytes)
            seed = int.from_bytes(hashlib.sha256(f'7123|{group}'.encode()).digest()[:4], 'little')
            score = score_event(data, 0, reindex_fit(model, region, expanded), result['replicates'], seed)
            records.append(dict(call=call, evidence_group=group, fold=fold, comparison_scope='additional_actual_overlap', scoring_region=expanded, fit_warning=model.get('diagnostics', {}).get('converged') is False, recipient_group_excluded_verified=True, **score))
    result.update(records=records, source_events=result['events'], events=len(records), additional_overlap_events=sum((r['comparison_scope'] == 'additional_actual_overlap' for r in records)), compatible=sum((r.get('compatible') is True for r in records)), rejected=sum((r.get('compatible') is False for r in records)), unassessed=sum((r.get('compatible') is None for r in records)), statuses=dict(Counter((r['status'] for r in records))), actual_simulations=sum((r.get('simulations', 0) for r in records)), all_eligible_full_anchor_overlaps_evaluated=full.get('status') == 'fitted')
    return result
