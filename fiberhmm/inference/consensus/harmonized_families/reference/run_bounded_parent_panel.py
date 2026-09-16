# Extracted reference kernels; see SOURCE_MANIFEST.json.
from collections import Counter
import hashlib
from .bounded_parent_prototype import prepare_events, evaluate_parent, score_event

def call_key(c):
    return (c['unit_id'], c['start'], c['end'])

def evaluate_cohort(units, calls, region, radii, maximum_bytes, *, center_bounds=None):
    """Fit each physical molecule once; still score ALL selected LLR events.

    Rare molecules with two selected events are excluded from fitting, not removed
    from output. Each of their events is scored on its native lattice against the
    full fit trained entirely on other molecules. No likelihood/MC rule changes.
    """
    groups = Counter((c.get('evidence_group_id', c['unit_id']) for c in calls))
    unique = [c for c in calls if groups[c.get('evidence_group_id', c['unit_id'])] == 1]
    duplicate = [c for c in calls if groups[c.get('evidence_group_id', c['unit_id'])] > 1]
    if len(unique) < 2:
        raise ValueError('Fewer than two unambiguous physical fitting groups')
    data = prepare_events(units, unique, region, maximum_bytes=maximum_bytes)
    for radius in radii:
        result = evaluate_parent(data, radius=radius, folds=10, replicates=4095, seed=7123, center_bounds=center_bounds)
        by_key = {call_key(r['call']): r for r in result['records']}
        model = result['full_model']
        for call in duplicate:
            group = call.get('evidence_group_id', call['unit_id'])
            if group in model.get('training_groups', []):
                raise ValueError('Duplicate group leaked into fit')
            one = prepare_events(units, [call], region, maximum_bytes=maximum_bytes)
            seed = int.from_bytes(hashlib.sha256(f'7123|{group}'.encode()).digest()[:4], 'little')
            score = score_event(one, 0, model, 4095, seed)
            by_key[call_key(call)] = dict(call=call, evidence_group=group, fold='excluded_multi_event_group', excluded_from_parent_fit=True, fit_warning=model.get('diagnostics', {}).get('converged') is False, **score)
        result['records'] = [by_key[call_key(c)] for c in calls]
        result.update(events=len(calls), physical_groups=len(groups), fit_source_events=len(unique), multi_event_groups_excluded_from_fit=sum((n > 1 for n in groups.values())), multi_event_calls_still_scored=len(duplicate), compatible=sum((r.get('compatible') is True for r in result['records'])), rejected=sum((r.get('compatible') is False for r in result['records'])), unassessed=sum((r.get('compatible') is None for r in result['records'])), statuses=dict(Counter((r['status'] for r in result['records']))), actual_simulations=sum((r.get('simulations', 0) for r in result['records'])))
        yield (radius, result, data['estimated_bytes'])
