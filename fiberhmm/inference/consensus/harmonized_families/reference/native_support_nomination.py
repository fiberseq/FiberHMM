# Extracted reference kernels; see SOURCE_MANIFEST.json.
from collections import defaultdict
import hashlib
from .native_cell_consolidation import maximal_sets, nominate_parents
from .run_bounded_parent_panel import call_key

def nominate_supported_unions(case, annotation, load_result, physical_support=False):
    if 'refinement_decisions' in annotation or annotation.get('settings', {}).get('refinement'):
        raise ValueError('One-shot refinement: use the original first-stage annotation')
    parents = {h['id']: h for h in annotation['hypotheses'] if h['display'] and h['kind'] == 'bounded_parent' and (not h['fit_warning']) and (h.get('status') != 'retained_informative_alternative')}
    models = {m['family']: m for m in case['models']}
    members = defaultdict(set)
    for row in case['ledger']:
        for f in row['compatible_families']:
            members[f].add((row['unit_id'], *row['interval']))
    members = {f: members[f] & {tuple(k) for k in m['source_call_keys']} for (f, m) in models.items()}
    refs = {fid: [max((models[f]['reference_interval'][0] for f in h['children'])), min((models[f]['reference_interval'][1] for f in h['children']))] for (fid, h) in parents.items()}
    if any((a >= b for (a, b) in refs.values())):
        raise ValueError('Parent lacks common actual original-child overlap')
    witnesses = defaultdict(set)
    for fid in sorted(parents):
        result = load_result(fid)
        fits = [result['full_model'], *result['fold_models'].values()]
        if not all((f.get('status') == 'fitted' and f.get('diagnostics', {}).get('converged') for f in fits)):
            continue
        passing = {call_key(s['call']) for s in result['records'] if s.get('compatible') is True and (not s.get('fit_warning'))}
        source_groups = {s['call'].get('evidence_group_id', s['call']['unit_id']) for s in result['records'] if call_key(s['call']) in passing and s['comparison_scope'] == 'source_cohort'}
        if len(source_groups) < 2:
            continue
        supported = {f for (f, keys) in members.items() if keys & passing}
        eligible = {p for (p, h) in parents.items() if set(h['children']) <= supported and max(refs[fid][0], refs[p][0]) < min(refs[fid][1], refs[p][1])}
        if fid not in eligible:
            continue
        if physical_support:
            proxies = [dict(family=p, reference_interval=refs[p], native_projection_cell=[[v - parents[p]['physical_radius'], v + parents[p]['physical_radius']] for v in parents[p]['reference_interval']]) for p in sorted(eligible)]
            groups = [frozenset(p['children']) for p in nominate_parents(proxies, 0)]
        else:
            groups = [frozenset((p for p in eligible if refs[p][0] <= x < refs[p][1])) for x in sorted({refs[p][0] for p in eligible})]
        for group in groups:
            if fid in group and len(group) >= 2:
                witnesses[group].add(fid)
    proposals = []
    for group in maximal_sets(witnesses):
        ids = sorted(group)
        children = sorted({f for p in ids for f in parents[p]['children']})
        reference = [max((models[f]['reference_interval'][0] for f in children)), min((models[f]['reference_interval'][1] for f in children))]
        if reference[0] >= reference[1]:
            raise AssertionError('Transitive-only overlap cannot nominate a union')
        token = hashlib.sha256('|'.join(ids).encode()).hexdigest()[:12]
        proposals.append(dict(id=f"P:R{token}:r{annotation['radius']}", children=children, replaces_parents=ids, center_bounds=None, actual_reference_overlap=reference, radius=annotation['radius'], every_pair_actual_overlap=True, nomination='actual_overlap_and_one_cached_native_support_witness', frozen_physical_support_required=physical_support, nomination_witnesses=sorted(witnesses[group]), original_children_flattened=True, recursive_nomination=False))
    return sorted(proposals, key=lambda p: (*p['actual_reference_overlap'], p['id']))
