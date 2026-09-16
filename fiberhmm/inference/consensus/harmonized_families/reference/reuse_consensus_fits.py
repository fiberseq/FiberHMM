# Extracted reference kernels; see SOURCE_MANIFEST.json.
from copy import deepcopy
from itertools import combinations
from .run_bounded_parent_panel import call_key
from .native_cell_consolidation import promotion
from .refine_consensus_parents import nominate_refinements, refine_annotation

def reuse_consensus(case, annotation, load_result, nomination_mode='fitted_boxes', score_identity=None, minimum_retention_groups=2):
    hypotheses = {h['id']: h for h in annotation['hypotheses']}
    if nomination_mode == 'fitted_boxes':
        proposals = nominate_refinements(case, annotation)
    elif nomination_mode == 'physical_support':
        proposals = nominate_refinements(case, annotation, box_kind='physical_support')
    elif nomination_mode in ('native_support', 'physical_witness'):
        from .native_support_nomination import nominate_supported_unions
        proposals = nominate_supported_unions(case, annotation, load_result, physical_support=nomination_mode == 'physical_witness')
    else:
        raise ValueError('Unknown cached consensus nomination mode')
    cache = {}
    results = {}
    receipts = []
    score_maps = {}
    for prop in proposals:
        prop['id'] = prop['id'].replace('P:R', 'P:C', 1)
        prop['nomination_geometry_bounds'] = prop.pop('center_bounds')
        prop['center_bounds'] = None
        if nomination_mode == 'fitted_boxes':
            prop['nomination'] = 'frozen_fitted_box_intersection_cached_native_support'
        candidates = []
        checks = []
        for fid in prop['replaces_parents']:
            if fid not in cache:
                cache[fid] = load_result(fid)
                result = cache[fid]
                scores = {call_key(s['call']): s for s in result['records']}
                if len(scores) != len(result['records']):
                    raise ValueError('Duplicate cached event scores')
                (left, right) = result['full_model']['anchor']
                expected = {call_key(c) for c in case['calls'] if c['start'] < right and c['end'] > left}
                if not expected <= scores.keys():
                    raise ValueError('Missing cached actual-overlap event scores; no implicit recomputation')
                if result['replicates'] != 4095:
                    raise ValueError('Expected the reference 4095-replicate policy')
                for (key, score) in scores.items():
                    fit = result['fold_models'].get(str(score['fold']), result['full_model'])
                    group = score['call'].get('evidence_group_id', key[0])
                    if group in fit.get('training_groups', []):
                        raise ValueError('Cached recipient molecule leaked into fit')
                score_maps[fid] = scores
            result = cache[fid]
            if not result.get('all_eligible_full_anchor_overlaps_evaluated'):
                raise ValueError('Cached result lacks complete direct overlap scoring')
            if result['full_model']['physical_radius'] != annotation['radius']:
                raise ValueError('Cannot reuse a different physical radius')
            decision = promotion(case, prop, result)
            checks.append(dict(candidate=fid, accepted=decision['accepted'], reason=decision['reason'], compatible_source_groups=decision['compatible_source_groups'], children_without_support=[f for (f, s) in decision['child_support'].items() if not s['compatible_source_groups']]))
            if decision['accepted']:
                candidates.append((-len(result['full_model']['training_groups']), fid))
        if candidates:
            (_, selected) = min(candidates)
            results[prop['id']] = cache[selected]
            prop.update(reused_from=selected, fit_source_children=hypotheses[selected]['children'])
        else:
            selected = None
        overlaps = [dict(parents=[a, b], shared_training_molecules=len(set(cache[a]['full_model']['training_groups']) & set(cache[b]['full_model']['training_groups']))) for (a, b) in combinations(prop['replaces_parents'], 2)]
        receipts.append(dict(proposal=deepcopy(prop), checks=checks, reused_from=selected, passing_models=[c['candidate'] for c in checks if c['accepted']], all_constituents_support_union=all((c['accepted'] for c in checks)), training_cohort_overlaps=overlaps, status='supported_cached_explanation' if selected else 'no_cached_model_supports_union'))
    if score_identity is not None:
        from .resolve_fit_aliases import coalesce_fit_alias_proposals
        (proposals, results, receipts) = coalesce_fit_alias_proposals(case, proposals, results, receipts, score_identity)
    answer = refine_annotation(case, annotation, proposals, results, minimum_retention_groups)
    receipt_by_id = {r['proposal']['id']: r for r in receipts}
    for decision in answer['refinement_decisions']:
        receipt = receipt_by_id[decision['proposal']['id']]
        if not decision['accepted']:
            decision['reason'] = 'no_cached_model_supports_union'
        decision['cached_checks'] = receipt['checks']
    for h in answer['hypotheses']:
        if h.get('status') == 'archived_by_joint_refinement':
            h['status'] = 'archived_redundant_cached_explanation'
        if h['id'] not in receipt_by_id:
            continue
        prop = receipt_by_id[h['id']]['proposal']
        h.update(status='shared_existing_native_explanation', reused_from=prop['reused_from'], fit_source_children=prop['fit_source_children'], nomination_geometry_bounds=prop['nomination_geometry_bounds'], joint_union_refit=False, original_fit_and_all_scores_unchanged=True)
        if prop.get('exact_fit_alias_groups'):
            h['exact_fit_alias_groups'] = prop['exact_fit_alias_groups']
    for row in answer['records']:
        key = (row['unit_id'], *row['interval'])
        for evidence in row['refinement_evaluations']:
            fid = receipt_by_id[evidence['hypothesis']]['reused_from']
            score = score_maps[fid][key]
            fold = str(score['fold'])
            origin = 'out_of_sample_molecule' if fold == 'external_group' else 'excluded_multi_event_source' if fold == 'excluded_multi_event_group' else 'molecule_excluded'
            evidence.update(native_score_source_model=fid, native_scoring_fold=score['fold'], evidence_origin=origin, original_native_score_unchanged=True)
    answer['settings'].update(refinement='cached_common_native_explanation', cached_nomination_mode=nomination_mode, physical_radius_paid_once_in_refit=False, physical_radius_unchanged_no_refit=True, extra_refits=0, extra_predictive_simulations=0, representative_selection='largest_original_training_cohort_then_id', candidate_nomination_is_not_a_distinct_population_test=True)
    if nomination_mode in ('native_support', 'physical_support', 'physical_witness'):
        answer['settings'].update(refinement_nomination_box_mass=None, refinement_nomination_extra_expansion_bp=None, fitted_edge_boxes_gate_consolidation=False)
    if nomination_mode in ('physical_support', 'physical_witness'):
        answer['settings'].update(refinement_nomination_extra_expansion_bp=0, existing_physical_support_intersection_gates_consolidation=True)
    answer['summary'].update(cached_consensus_checks=sum((len(r['checks']) for r in receipts)), cached_consensus_unions=sum((r['reused_from'] is not None for r in receipts)), extra_refits=0, extra_predictive_simulations=0)
    return (answer, receipts)
