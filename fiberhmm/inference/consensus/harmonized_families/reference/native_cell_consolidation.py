# Extracted reference kernels; see SOURCE_MANIFEST.json.
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import itertools
from .overlapping_family_update import overlaps
from .run_bounded_parent_panel import call_key

def maximal_sets(sets):
    """Keep every inclusion-maximal alternative, never choose one partition."""
    answer = []
    for group in sorted(set(map(frozenset, sets)), key=lambda s: (-len(s), tuple(sorted(s)))):
        if group and (not any((group <= other for other in answer))):
            answer.append(group)
    return answer

def center_box(model, radius):
    (left, right) = model['native_projection_cell']
    (a, b) = model['reference_interval']
    return [[left[0] - radius, min(left[1] + radius, b - 1)], [max(right[0] - radius, a + 1), right[1] + radius]]

def intersect_boxes(boxes):
    box = [[max((b[e][0] for b in boxes)), min((b[e][1] for b in boxes))] for e in range(2)]
    if any((a > b for (a, b) in box)) or box[0][0] >= box[1][1]:
        return None
    return box

def nominate_parents(models, radius):
    """Exact maximal shared-center sets over native-cell rectangles.

    All reference intervals in a group must share ACTUAL positive overlap.
    Within each spatial clique, sweep rectangle starts, including y=x+1 for
    positive-width centers. A common intersection certifies all members jointly;
    pairwise compatibility/connected components alone never nominate a parent.
    """
    if not isinstance(radius, int) or radius < 0:
        raise ValueError('Nonnegative integer radius required')
    by_id = {m['family']: m for m in models}
    if len(by_id) != len(models):
        raise ValueError('Duplicate native family IDs')
    if not models:
        return []
    spatial = maximal_sets([{m['family'] for m in models if m['reference_interval'][0] <= x < m['reference_interval'][1]} for x in sorted({m['reference_interval'][0] for m in models})])
    boxes = {f: center_box(m, radius) for (f, m) in by_id.items()}
    found = []
    for clique in spatial:
        for x in sorted({boxes[f][0][0] for f in clique}):
            at_x = [f for f in clique if boxes[f][0][0] <= x <= boxes[f][0][1]]
            for y in sorted({boxes[f][1][0] for f in at_x} | {x + 1}):
                if y <= x:
                    continue
                group = frozenset((f for f in at_x if boxes[f][1][0] <= y <= boxes[f][1][1]))
                if len(group) >= 2:
                    found.append(group)
    result = []
    for group in maximal_sets(found):
        ids = sorted(group)
        box = intersect_boxes([boxes[f] for f in ids])
        ref = [max((by_id[f]['reference_interval'][0] for f in ids)), min((by_id[f]['reference_interval'][1] for f in ids))]
        if box is None or ref[0] >= ref[1]:
            raise AssertionError('Non-shared parent nomination')
        key = hashlib.sha256('|'.join(ids).encode()).hexdigest()[:8]
        result.append(dict(id=f'P:G{key}:r{radius}', children=ids, center_bounds=box, actual_reference_overlap=ref, radius=radius, nomination='common_native_best_projection_cells_not_confidence_regions', every_pair_actual_overlap=all((overlaps(by_id[a]['reference_interval'], by_id[b]['reference_interval']) for (a, b) in itertools.combinations(ids, 2)))))
    return sorted(result, key=lambda g: (g['center_bounds'][0][0], g['center_bounds'][1][0], g['id']))

def ledger_members(case):
    """family -> {(unit, start, end)} of the case's original compatible calls."""
    members = defaultdict(set)
    for r in case['ledger']:
        for f in r['compatible_families']:
            members[f].add((r['unit_id'], *r['interval']))
    return members

def promotion(case, proposal, result, minimum_source_groups=2, own_members=None):
    """Constructive support for the shared proposal, never all-read agreement.

    This is an operational support gate, NOT statistical equality/recurrence
    certification. Each nominated child must have a source observation that
    supports both its original child fit and the refitted parent. All rejects
    remain in the ledger; they do not each instantiate another active family.
    ``own_members`` is ``ledger_members(case)``; callers scoring many proposals
    against one unchanged case pass it once instead of rebuilding it per call.
    """
    fits = [result['full_model'], *result['fold_models'].values()]
    converged = all((f.get('status') == 'fitted' and f.get('diagnostics', {}).get('converged') for f in fits))
    models = {m['family']: m for m in case['models']}
    if own_members is None:
        own_members = ledger_members(case)
    supported = {call_key(r['call']): r for r in result['records'] if r.get('compatible') is True and (not r.get('fit_warning'))}
    children = {}
    for f in proposal['children']:
        sources = {tuple(k) for k in models[f]['source_call_keys']}
        keys = sorted(sources & own_members[f] & supported.keys())
        groups = sorted({supported[k]['call'].get('evidence_group_id', k[0]) for k in keys})
        children[f] = dict(compatible_source_groups=groups, compatible_source_event_keys=keys, source_calls=len(sources), compatible_events=len(keys))
    source_groups = {r['call'].get('evidence_group_id', r['call']['unit_id']) for r in result['records'] if r.get('compatible') is True and (not r.get('fit_warning')) and (r['comparison_scope'] == 'source_cohort')}
    accepted = converged and len(source_groups) >= minimum_source_groups and all((c['compatible_source_groups'] for c in children.values()))
    return dict(accepted=bool(accepted), converged=bool(converged), compatible_source_groups=len(source_groups), child_support=children, minimum_source_groups=minimum_source_groups, all_historical_members_must_pass=False, statistical_equality_claim=False, reason='shared_cell_parent_supported' if accepted else 'numerical_or_fit_failure' if not converged else 'insufficient_parent_source_support' if len(source_groups) < minimum_source_groups else 'child_lacks_constructive_parent_support')

def annotate(case, proposals, results, radius):
    model_by_id = {m['family']: m for m in case['models']}
    child_members = ledger_members(case)
    replaced = defaultdict(list)
    parent_members = defaultdict(list)
    evaluations = defaultdict(list)
    hypotheses = []
    decisions = []
    for p in proposals:
        result = results.get(p['id'])
        if result is None:
            decisions.append(dict(proposal=p, accepted=False, reason='unassessed_fit'))
            continue
        decision = promotion(case, p, result, own_members=child_members)
        decisions.append(dict(proposal=p, **decision))
        fit = result['full_model']
        accepted = decision['accepted']
        if accepted:
            for child in p['children']:
                replaced[child].append(p['id'])
        for r in result['records']:
            k = call_key(r['call'])
            evaluations[k].append(dict(hypothesis=p['id'], compatible=r.get('compatible'), status=r['status'], fit_warning=r.get('fit_warning', False), comparison_scope=r['comparison_scope'], parent_accepted=accepted))
            if accepted and r.get('compatible') is True and (not r.get('fit_warning')):
                parent_members[k].append(p['id'])
        hypotheses.append(dict(id=p['id'], kind='bounded_parent', query=p['id'].split(':')[1], status='consolidated_shared_cell_hypothesis' if accepted else 'not_promoted', display=accepted, fit_warning=not decision['converged'], reference_interval=fit.get('anchor'), geometry=fit.get('geometry'), physical_radius=radius, compatible_events=result['compatible'], children=p['children'], displayed_instead_of_children=p['children'] if accepted else [], common_center_bounds=p['center_bounds'], actual_reference_overlap=p['actual_reference_overlap'], population_identity_established=False))
    for (f, m) in sorted(model_by_id.items()):
        hypotheses.append(dict(id=f, kind='native_child', reference_interval=m['reference_interval'], geometry=m.get('normalized_geometry'), status='archived_after_consolidation' if replaced[f] else 'retained_alternative' if child_members[f] else 'no_compatible_members', display=bool(child_members[f]) and (not replaced[f]), nested_under=replaced[f], fit_warning=m.get('fit_warning', False), native_projection_cell=m.get('native_projection_cell'), compatible_events=len(child_members[f]), population_identity_established=False))
    records = []
    for old in case['ledger']:
        key = (old['unit_id'], *old['interval'])
        parents = sorted(parent_members[key])
        children = list(old['compatible_families'])
        active = sorted(set(parents + [f for f in children if not replaced[f]]))
        ev = evaluations[key]
        if active:
            status = 'multi_compatible' if len(active) > 1 else 'compatible'
        elif not old['inference_eligible']:
            status = 'outside_inference_scaffold'
        elif any((replaced[f] for f in children)):
            status = 'residual_after_consolidation'
        elif not ev:
            status = 'no_parent_test'
        elif any((e['compatible'] is None or e['fit_warning'] or e['compatible'] is True for e in ev)):
            status = 'parent_unassessed_or_not_promoted'
        else:
            status = 'tested_parents_rejected'
        replacing = sorted({p for f in children for p in replaced[f]})
        assessed = {e['hypothesis']: e for e in ev if e['hypothesis'] in replacing}
        residual_reason = None
        if status == 'residual_after_consolidation':
            if any((p not in assessed for p in replacing)):
                residual_reason = 'replacing_parent_not_tested'
            elif any((e['compatible'] is None or e['fit_warning'] for e in assessed.values())):
                residual_reason = 'replacing_parent_unassessed'
            else:
                residual_reason = 'all_replacing_parents_rejected'
        # The ledger row is shared, not copied: it is frozen (consolidate_scope
        # re-digests the case at the end) and only ever read through 'original'.
        records.append(dict(original=old, call_id=old['call_id'], unit_id=old['unit_id'], interval=old['interval'], strand=old['strand'], compatible_children=children, compatible_parents=parents, display_hypotheses=active, status=status, parent_evaluations=ev, replacing_hypotheses=replacing, residual_assessment=residual_reason, original_llr_observation_retained=True))
    visible_parents = sum((h['kind'] == 'bounded_parent' and h['display'] for h in hypotheses))
    visible_children = sum((h['kind'] == 'native_child' and h['display'] for h in hypotheses))
    baseline = sum((bool(v) for v in child_members.values()))
    summary = dict(original_calls=len(records), original_child_links=sum((len(r['compatible_children']) for r in records)), preserved_child_links=sum((len(r['original']['compatible_families']) for r in records)), baseline_active_hypotheses=baseline, active_hypotheses=visible_parents + visible_children, net_hypotheses_removed=baseline - visible_parents - visible_children, supported_parent_hypotheses=visible_parents, visible_child_alternatives=visible_children, nested_child_hypotheses=sum((bool(v) for v in replaced.values())), proposed_parents=len(proposals), failed_parent_proposals=sum((not d['accepted'] for d in decisions)), calls_with_any_hypothesis=sum((bool(r['display_hypotheses']) for r in records)), calls_with_parent=sum((bool(r['compatible_parents']) for r in records)), calls_retained_only_by_children=sum((bool(r['display_hypotheses']) and (not r['compatible_parents']) for r in records)), multi_compatible_calls=sum((len(r['display_hypotheses']) > 1 for r in records)), residual_after_consolidation=sum((r['status'] == 'residual_after_consolidation' for r in records)), status_counts=dict(Counter((r['status'] for r in records))), residual_assessment_counts=dict(Counter((r['residual_assessment'] for r in records if r['residual_assessment']))))
    return dict(schema='native_cell_consolidation_v1', locus=case['locus'], channel=case['channel'], radius=radius, hypotheses=hypotheses, records=records, decisions=decisions, summary=summary, settings=dict(physical_parent_radius_bp=radius, additional_parent_matching_floor_bp=0, fixed_child_scoring_floor_bp=2, predictive_reference_percent=99.9, replicates=4095, all_historical_members_must_pass=False, native_cells_are_nomination_not_confidence_intervals=True, full_locus_nomination=True, distinct_population_count_certified=False, manual_browser_merge_implemented=False, failed_maximal_proposals_keep_children_no_exhaustive_subgroup_retry=True))
