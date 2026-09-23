# Extracted reference kernels; see SOURCE_MANIFEST.json.
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import math
from .native_cell_consolidation import ledger_members, nominate_parents, promotion
from .run_bounded_parent_panel import call_key
from .family_retirement import retention_checks

def nominate_refinements(case, annotation, box_kind='conditional_95'):
    if box_kind not in ('conditional_95', 'physical_support'):
        raise ValueError('Unknown nomination box kind')
    if 'refinement_decisions' in annotation or annotation.get('settings', {}).get('refinement'):
        raise ValueError('One-shot refinement: already-refined parents cannot nominate another pass')
    models = {m['family']: m for m in case['models']}
    parents = {h['id']: h for h in annotation['hypotheses'] if h['display'] and h['kind'] == 'bounded_parent' and (not h['fit_warning']) and (h.get('status') != 'retained_informative_alternative')}
    proxies = []
    for (fid, h) in parents.items():
        box = h['geometry']['credible_boxes']['0.95'] if box_kind == 'conditional_95' else {e: [a - h['physical_radius'], a + h['physical_radius']] for (e, a) in zip(('left', 'right'), h['reference_interval'])}
        cell = [[math.ceil(box[e][0]), math.floor(box[e][1])] for e in ('left', 'right')]
        if any((a > b for (a, b) in cell)):
            raise ValueError('Expected integer edge boxes')
        children = [models[f] for f in h['children']]
        reference = [max((m['reference_interval'][0] for m in children)), min((m['reference_interval'][1] for m in children))]
        if reference[0] >= reference[1]:
            raise ValueError('First-stage parent lacks actual common original overlap')
        proxies.append(dict(family=fid, reference_interval=reference, native_projection_cell=cell))
    proposals = []
    for nominated in nominate_parents(proxies, 0):
        replaced = nominated['children']
        children = sorted({f for p in replaced for f in parents[p]['children']})
        token = hashlib.sha256('|'.join(replaced).encode()).hexdigest()[:12]
        proposals.append(dict(id=f"P:R{token}:r{annotation['radius']}", children=children, replaces_parents=replaced, center_bounds=nominated['center_bounds'], actual_reference_overlap=nominated['actual_reference_overlap'], radius=annotation['radius'], every_pair_actual_overlap=True, nomination=f'frozen_{box_kind}_edge_box_intersection_no_expansion', original_children_flattened=True, recursive_nomination=False))
    return proposals

def support_diagnostics(case, old, prop, result):
    """Report all support changes without inventing a fraction/majority gate."""
    source = {m['family']: m for m in case['models']}
    parents = {h['id']: h for h in old['hypotheses']}
    original_members = defaultdict(set)
    native_members = defaultdict(set)
    for row in old['records']:
        key = (row['unit_id'], *row['interval'])
        for f in row['display_hypotheses']:
            original_members[f].add(key)
        for f in row['original']['compatible_families']:
            native_members[f].add(key)
    scores = {call_key(s['call']): s for s in result['records']}
    joint = {k for (k, s) in scores.items() if s.get('compatible') is True and (not s['fit_warning'])}
    fit = result['full_model']
    radius = prop['radius']
    support = [[a - radius, a + radius] for a in fit['anchor']] if fit.get('status') == 'fitted' else None

    def comparison(keys, baseline):
        n = len(keys)
        (before, after) = (len(keys & baseline), len(keys & joint))
        return dict(source_events=n, previously_compatible=before, union_compatible=after, previously_compatible_fraction=before / n if n else None, union_compatible_fraction=after / n if n else None, fraction_change=(after - before) / n if n else None, lost_previous_members=len((keys & baseline) - joint), gained_members=len((keys & joint) - baseline), unassessed=sum((k not in scores or scores[k].get('compatible') is None or scores[k]['fit_warning'] for k in keys)))
    child_info = {}
    for f in prop['children']:
        m = source[f]
        keys = {tuple(k) for k in m['source_call_keys']}
        cell = m['native_projection_cell']
        child_info[f] = dict(**comparison(keys, native_members[f]), native_modal_cell=cell, modal_cell_within_union_support=bool(support and all((lo <= a <= b <= hi for ((a, b), (lo, hi)) in zip(cell, support)))))
    parent_info = {}
    for fid in prop['replaces_parents']:
        h = parents[fid]
        keys = {tuple(k) for f in h['children'] for k in source[f]['source_call_keys']}
        box = h['geometry']['credible_boxes']['0.95']
        touches = {e: box[e][0] == a - h['physical_radius'] or box[e][1] == a + h['physical_radius'] for (e, a) in zip(('left', 'right'), h['reference_interval'])}
        parent_info[fid] = dict(**comparison(keys, original_members[fid]), all_prior_compatible_calls=len(original_members[fid]), all_prior_calls_compatible_with_union=len(original_members[fid] & joint), box_touches_support=touches, nomination_box_censored=any(touches.values()))
    return dict(children=child_info, constituent_parents=parent_info, union_physical_support=support, fractions_are_diagnostics_not_promotion_gates=True, nomination_censored=any((d['nomination_box_censored'] for d in parent_info.values())))

def refine_annotation(case, old, proposals, results, minimum_retention_groups=2):
    ann = deepcopy(old)
    before = {h['id']: h for h in old['hypotheses']}
    replacements = defaultdict(list)
    passing = defaultdict(list)
    evaluations = defaultdict(list)
    (decisions, additions) = ([], [])
    members = ledger_members(case)
    for prop in proposals:
        fid = prop['id']
        if fid in before:
            raise ValueError('Refinement hypothesis ID collision')
        result = results.get(fid)
        if result is None:
            decisions.append(dict(proposal=prop, accepted=False, reason='unassessed_fit'))
            continue
        decision = promotion(case, prop, result, own_members=members)
        decision['support_diagnostics'] = support_diagnostics(case, old, prop, result)
        decisions.append(dict(proposal=prop, **decision))
        accepted = decision['accepted']
        fit = result['full_model']
        for score in result['records']:
            key = call_key(score['call'])
            evaluations[key].append(dict(hypothesis=fid, compatible=score.get('compatible'), fit_warning=score.get('fit_warning', False), status=score['status'], parent_accepted=accepted, comparison_scope=score['comparison_scope']))
            if accepted and score.get('compatible') is True and (not score.get('fit_warning')):
                passing[key].append(fid)
        if accepted:
            for parent in prop['replaces_parents']:
                replacements[parent].append(fid)
        additions.append(dict(id=fid, kind='bounded_parent', display=accepted, status='consolidated_refitted_parent_union' if accepted else 'not_promoted', geometry=fit.get('geometry'), reference_interval=fit.get('anchor', prop['actual_reference_overlap']), physical_radius=old['radius'], children=prop['children'], displayed_instead_of_children=[], replaces_parents=prop['replaces_parents'], common_center_bounds=prop['center_bounds'], actual_reference_overlap=prop['actual_reference_overlap'], source_channels=sorted({ch for p in prop['replaces_parents'] for ch in before[p]['source_channels']}), compatible_events=result['compatible'], fit_warning=not decision['converged'], support_diagnostics=decision['support_diagnostics'], population_identity_established=False, nomination=prop['nomination']))
    retirement = retention_checks(case, old, replacements, evaluations, minimum_retention_groups)
    for h in ann['hypotheses']:
        check = retirement.get(h['id'])
        if check:
            h['retirement_check'] = check
        if check and check['retain']:
            h.update(display=True, status='retained_informative_alternative')
            replacements[h['id']] = []
        if replacements[h['id']]:
            h['status_before_refinement'] = h['status']
            h.update(display=False, status='archived_by_joint_refinement', replaced_by=replacements[h['id']])
    ann['hypotheses'].extend(additions)
    for row in ann['records']:
        key = (row['unit_id'], *row['interval'])
        old_ids = row['display_hypotheses']
        row['memberships_before_refinement'] = list(old_ids)
        row['compatible_parents_before_refinement'] = list(row.get('compatible_parents', []))
        row['status_before_refinement'] = row['status']
        row['residual_assessment_before_refinement'] = row['residual_assessment']
        row['refinement_evaluations'] = evaluations[key]
        row['refinement_replacing_hypotheses'] = sorted({f for p in old_ids for f in replacements[p]})
        ids = sorted({f for f in old_ids if not replacements[f]} | set(passing[key]))
        row['display_hypotheses'] = ids
        row['compatible_parents'] = sorted({f for f in row.get('compatible_parents', []) if not replacements[f]} | set(passing[key]))
        if ids:
            row['status'] = 'multi_compatible' if len(ids) > 1 else 'compatible'
            row['residual_assessment'] = None
        elif old_ids:
            row['status'] = 'residual_after_consolidation'
            replacing = row['refinement_replacing_hypotheses']
            scores = {e['hypothesis']: e for e in evaluations[key]}
            if any((f not in scores for f in replacing)):
                reason = 'replacing_parent_not_tested'
            elif any((scores[f]['compatible'] is None or scores[f]['fit_warning'] for f in replacing)):
                reason = 'replacing_parent_unassessed'
            else:
                reason = 'all_replacing_parents_rejected'
            row['residual_assessment'] = reason
    active = {h['id']: h for h in ann['hypotheses'] if h['display']}
    counts = Counter((f for row in ann['records'] for f in row['display_hypotheses']))
    for h in ann['hypotheses']:
        h['active_compatible_events_including_foreign'] = counts[h['id']]
    ann['pre_refinement_summary'] = deepcopy(old['summary'])
    ann['refinement_decisions'] = decisions
    rows = ann['records']
    ann['summary'].update(active_hypotheses=len(active), supported_parent_hypotheses=sum((h['kind'] == 'bounded_parent' for h in active.values())), visible_child_alternatives=sum((h['kind'] != 'bounded_parent' for h in active.values())), calls_with_any_hypothesis=sum((bool(r['display_hypotheses']) for r in rows)), calls_with_parent=sum((bool(r['compatible_parents']) for r in rows)), multi_compatible_calls=sum((len(r['display_hypotheses']) > 1 for r in rows)), calls_retained_only_by_children=sum((bool(r['display_hypotheses']) and all((active[f]['kind'] != 'bounded_parent' for f in r['display_hypotheses'])) for r in rows)), residual_after_consolidation=sum((r['status'] == 'residual_after_consolidation' for r in rows)), residual_assessment_counts=dict(Counter((r['residual_assessment'] for r in rows if r['residual_assessment']))), status_counts=dict(Counter((r['status'] for r in rows))), refinement_proposals=len(proposals), refinement_accepted=sum((d['accepted'] for d in decisions)), first_stage_parents_archived=sum((bool(v) for v in replacements.values())), cross_source_active_parents=sum((h['kind'] == 'bounded_parent' and len(h['source_channels']) > 1 for h in active.values())))
    promoted_diagnostics = [d['support_diagnostics'] for d in decisions if d['accepted']]
    ann['summary'].update(accepted_unions_with_censored_nomination_boxes=sum((d['nomination_censored'] for d in promoted_diagnostics)), accepted_child_comparisons_with_lower_compatible_fraction=sum((c['fraction_change'] is not None and c['fraction_change'] < 0 for d in promoted_diagnostics for c in d['children'].values())), accepted_parent_comparisons_with_lower_compatible_fraction=sum((c['fraction_change'] is not None and c['fraction_change'] < 0 for d in promoted_diagnostics for c in d['constituent_parents'].values())))
    ann['summary']['by_source'] = {}
    for channel in case['source_channels']:
        source_rows = [r for r in rows if r['source_channel'] == channel]
        ann['summary']['by_source'][channel] = dict(original_calls=len(source_rows), eligible_calls=sum((r['original']['inference_eligible'] for r in source_rows)), assigned=sum((bool(r['display_hypotheses']) for r in source_rows)), multi_compatible=sum((len(r['display_hypotheses']) > 1 for r in source_rows)), residual=sum((r['status'] == 'residual_after_consolidation' for r in source_rows)))
    ann['settings'].update(refinement='frozen_fitted_edge_boxes_original_event_joint_refit', refinement_nomination_box_mass=0.95, refinement_nomination_extra_expansion_bp=0, recursive_nomination=False, physical_radius_paid_once_in_refit=True)
    ann['settings']['minimum_retention_groups'] = minimum_retention_groups
    ann['retirement_checks'] = retirement
    ann['summary']['informative_alternatives_retained'] = sum((c['retain'] for c in retirement.values()))
    if 'baseline_active_hypotheses' in ann['summary']:
        ann['summary']['net_hypotheses_removed'] = ann['summary']['baseline_active_hypotheses'] - len(active)
    return ann
