# Extracted reference kernels; see SOURCE_MANIFEST.json.
from copy import deepcopy
from .reuse_consensus_fits import reuse_consensus
from .refine_consensus_parents import annotation_view
from .synthetic_state_benchmark import digest

def resolve_representatives(case, annotation, load_result, minimum_retention_groups=2):
    if annotation['settings'].get('representative_resolution'):
        raise ValueError('One-shot representative resolution: frozen phase already complete')
    if annotation['settings'].get('refinement') != 'cached_common_native_explanation':
        raise ValueError('Resolve selected cached representatives, not arbitrary model histories')
    frozen = digest(annotation)
    if [r['original'] for r in annotation['records']] != case['ledger']:
        raise ValueError('Representative input must retain original ledger order and records')
    hypotheses = {h['id']: h for h in annotation['hypotheses']}
    active = {f: h for (f, h) in hypotheses.items() if h['display'] and h['kind'] == 'bounded_parent'}

    def root(fid):
        seen = set()
        while hypotheses[fid].get('reused_from'):
            if fid in seen:
                raise ValueError('Cyclic fit provenance')
            seen.add(fid)
            fid = hypotheses[fid]['reused_from']
        return fid
    # Shallow view: only top-level keys and settings are edited here; the final
    # digest assertion still proves the input annotation is unchanged.
    view = annotation_view(annotation)
    previous_decisions = view.pop('refinement_decisions')
    view['settings'].pop('refinement')
    view['settings']['frozen_representative_input_digest'] = frozen
    result_cache = {}

    def load(fid):
        original = root(fid)
        if original not in result_cache:
            result_cache[original] = load_result(original)
        return result_cache[original]
    (answer, receipts) = reuse_consensus(case, view, load, nomination_mode='physical_witness', score_identity=root, minimum_retention_groups=minimum_retention_groups)
    new_ids = {r['proposal']['id'] for r in receipts if r['reused_from']}
    original_rows = {(r['unit_id'], *r['interval']): r for r in annotation['records']}
    if len(original_rows) != len(annotation['records']):
        raise ValueError('Duplicate original event keys')
    for row in answer['records']:
        previous = original_rows[(row['unit_id'], *row['interval'])]
        row['memberships_before_representative_resolution'] = deepcopy(previous['display_hypotheses'])
        row['status_before_representative_resolution'] = previous['status']
        row['residual_before_representative_resolution'] = deepcopy(previous['residual_assessment'])
        row['representative_resolution_evaluations'] = row['refinement_evaluations']
        for evidence in row['representative_resolution_evaluations']:
            evidence['selected_representative'] = evidence['native_score_source_model']
            evidence['native_score_source_model'] = root(evidence['native_score_source_model'])
        row['refinement_evaluations'] = deepcopy(previous['refinement_evaluations']) + row['representative_resolution_evaluations']
        row['initial_refinement_history'] = {k: deepcopy(v) for (k, v) in previous.items() if k.endswith('_before_refinement') or k == 'refinement_replacing_hypotheses'}
    for h in answer['hypotheses']:
        if h['id'] not in new_ids:
            continue
        selected = h['reused_from']
        original = root(selected)
        h.update(selected_representative=selected, reused_from=original, fit_source_children=deepcopy(hypotheses[original]['children']), representative_resolution=True, frozen_representative_supports={f: [[v - active[f]['physical_radius'], v + active[f]['physical_radius']] for v in active[f]['reference_interval']] for f in h['replaces_parents']})
    for receipt in receipts:
        if receipt['reused_from']:
            receipt['original_score_model'] = root(receipt['reused_from'])
    answer['initial_refinement_decisions'] = previous_decisions
    answer['representative_resolution_decisions'] = answer['refinement_decisions']
    answer['refinement_decisions'] = previous_decisions + answer['refinement_decisions']
    answer['summary'].update(representative_resolution_proposals=len(receipts), representative_resolution_unions=len(new_ids), active_before_representative_resolution=annotation['summary']['active_hypotheses'], assignments_lost_in_representative_resolution=sum((bool(p['display_hypotheses']) and (not r['display_hypotheses']) for (p, r) in zip(annotation['records'], answer['records']))), assignments_gained_in_representative_resolution=sum((not p['display_hypotheses'] and bool(r['display_hypotheses']) for (p, r) in zip(annotation['records'], answer['records']))))
    answer['settings'].update(representative_resolution='one_frozen_physical_domain_common_native_witness', initial_cached_nomination_mode=annotation['settings']['cached_nomination_mode'], no_recursive_representative_resolution=True, original_radius_and_native_scores_unchanged=True)
    assert digest(annotation) == frozen
    assert [r['original'] for r in answer['records']] == case['ledger']
    return (answer, receipts)
