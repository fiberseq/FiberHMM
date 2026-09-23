# Extracted reference kernels; see SOURCE_MANIFEST.json.
from collections import defaultdict
from copy import deepcopy
import hashlib
import numpy as np
from .native_cell_consolidation import ledger_members, promotion

def coalesce_fit_alias_proposals(case, proposals, results, receipts, identity):
    groups = defaultdict(list)
    for receipt in receipts:
        if receipt['reused_from']:
            groups[identity(receipt['reused_from'])].append(receipt)
    models = {m['family']: m for m in case['models']}
    members = ledger_members(case)
    removed = set()
    additions = []
    new_receipts = []
    results = dict(results)
    for (model, group) in sorted(groups.items()):
        if len(group) < 2:
            continue
        children = sorted({f for r in group for f in r['proposal']['children']})
        overlap = [max((models[f]['reference_interval'][0] for f in children)), min((models[f]['reference_interval'][1] for f in children))]
        if overlap[0] >= overlap[1]:
            continue
        first = min(group, key=lambda r: (r['reused_from'], r['proposal']['id']))
        result = results[first['proposal']['id']]
        for r in group:
            other = results[r['proposal']['id']]
            if other is result:
                continue
            try:
                np.testing.assert_equal(other, result)
            except (AssertionError, ValueError):
                raise ValueError('One cached model identity has different fits or scores') from None
        prop = deepcopy(first['proposal'])
        ids = sorted((r['proposal']['id'] for r in group))
        token = hashlib.sha256('|'.join(ids).encode()).hexdigest()[:12]
        prop.update(id=f"P:I{token}:r{prop['radius']}", children=children, replaces_parents=sorted({p for r in group for p in r['proposal']['replaces_parents']}), actual_reference_overlap=overlap, nomination_geometry_bounds=None, nomination='exact_cached_model_identity_with_common_actual_overlap', exact_fit_alias_groups=[deepcopy(r['proposal']) for r in group])
        decision = promotion(case, prop, result, own_members=members)
        if not decision['accepted']:
            raise AssertionError('Identical witnesses lost constructive support')
        selected = first['reused_from']
        receipt = dict(proposal=deepcopy(prop), reused_from=selected, passing_models=[selected], all_constituents_support_union=False, checks=[dict(candidate=selected, accepted=True, reason=decision['reason'], compatible_source_groups=decision['compatible_source_groups'], children_without_support=[])], training_cohort_overlaps=[], status='identical_cached_model_aliases', original_score_model=model, original_alias_receipts=deepcopy(group))
        removed.update(ids)
        additions.append(prop)
        new_receipts.append(receipt)
        for fid in ids:
            results.pop(fid)
        results[prop['id']] = result
    return ([p for p in proposals if p['id'] not in removed] + additions, results, [r for r in receipts if r['proposal']['id'] not in removed] + new_receipts)
