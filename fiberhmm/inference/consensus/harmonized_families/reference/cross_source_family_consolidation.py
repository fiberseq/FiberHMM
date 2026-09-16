# Extracted reference kernels; see SOURCE_MANIFEST.json.
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
import numpy as np
from .synthetic_state_benchmark import read_json, digest
from .run_bounded_parent_panel import call_key
from .overlapping_family_update import overlaps
from .native_cell_consolidation import annotate
from fiberhmm.inference.consensus.native_cross import frozen_model_geometry, boundary_grid, transferred_call
from fiberhmm.inference.consensus.cross_preparation import NativeReadCache, TransferGeometryCache

def combine_cases(parts):
    """Scope hypothesis IDs only; never rewrite native call geometry/probabilities."""
    if len({tuple(p['source_extent']) for p in parts.values()}) != 1:
        raise ValueError('Matching source extents required')
    units = {}
    calls = []
    ledger = []
    models = []
    source_by_unit = {}
    model_source = {}
    physical = {}
    for (channel, case) in sorted(parts.items()):
        for (uid, u) in case['units'].items():
            if uid in units:
                raise ValueError('Duplicated cross-source unit ID')
            group = u['fold_group_id']
            if group in physical:
                raise ValueError('Shared physical group needs explicit joint-view fitting; not independent')
            physical[group] = channel
            units[uid] = u
            source_by_unit[uid] = channel
        calls.extend(case['calls'])
        family = lambda f: channel + '::' + f
        for m in case['models']:
            model = deepcopy(m)
            model['source_family'] = m['family']
            model['source_channel'] = channel
            model['family'] = family(m['family'])
            models.append(model)
            model_source[model['family']] = channel
        for old in case['ledger']:
            row = deepcopy(old)
            row['original_source_record'] = deepcopy(old)
            row['source_channel'] = channel
            row['call_id'] = channel + '::' + str(old['call_id'])
            row['compatible_families'] = [family(f) for f in old['compatible_families']]
            for field in ['compatible_model_fit_warnings', 'evaluated_model_fit_warnings']:
                if field in row:
                    row[field] = [family(f) for f in row[field]]
            ledger.append(row)
    if len({call_key(c) for c in calls}) != len(calls):
        raise ValueError('Duplicate event across sources')
    return dict(locus=next(iter(parts.values()))['locus'], channel='SR' if set(parts) == {'CT', 'GA'} else 'CRX', units=units, calls=calls, ledger=ledger, models=models, source_by_unit=source_by_unit, model_source=model_source, source_extent=next(iter(parts.values()))['source_extent'], source_channels=sorted(parts), source_digests={k: digest(v) for (k, v) in parts.items()})

def freeze_children(part):
    native = read_json(Path(part['native_cell_provenance']['path']))
    if digest(native) != part['native_cell_provenance']['digest']:
        raise ValueError('Native model changed since reference')
    native_models = {m['family']: m for m in native['family_models']}
    indexed = {c['unit_index']: part['units'][c['unit_id']] for c in native['calls']}
    for c in native['calls']:
        if indexed[c['unit_index']]['unit_id'] != c['unit_id']:
            raise ValueError('Source native index mismatch')
    return {m['family']: frozen_model_geometry(native_models[m['family']], native['calls'], indexed) for m in part['models']}

def transfer_child(frozen, unit, call, *, replicates=4095, read_cache=None, cache=None):
    """Transport source-cell density; retain whole recipient span and native masks.

    Use the reference child's 2 bp scoring floor, not the parent's physical r.
    Disable legacy fraction gates: any positive admissible mass can be scored,
    exactly as native CR does; report retained mass instead of thresholding it.
    """
    source = frozen['grid']
    lo = min(source['domain'][0], call['start'])
    hi = max(source['domain'][1], call['end'])
    grid = source if [lo, hi] == list(source['domain']) else boundary_grid(source['positions'], [lo, hi], model_domains=[source['domain']])
    score = transferred_call(frozen, grid, unit, call, floor_bp=2, replicates=replicates, minimum_retained_mass=np.nextafter(0.0, 1.0), minimum_call_attribution_mass=0.0, require_call_overlap=True, _read_cache=read_cache, _geometry_cache=cache if grid is source else None)
    compatible = score['status'] == 'scored' and score['predictive_tail_interval'][1] >= 0.001
    assessed = score['status'] in ('scored', 'core_contradicted')
    if not score['original_span_unchanged'] or not score['recipient_raw_span_within_comparison_domain']:
        raise AssertionError('Transferred call truncated')
    return dict(score, compatible=bool(compatible) if assessed else None)

def foreign_child_scores(case, parts, progress=None):
    """Every eligible foreign event × actually overlapping fitted child, once."""
    models = {m['family']: m for m in case['models']}
    records = []
    reads = NativeReadCache(64 * 1024 ** 2)
    for (channel, part) in sorted(parts.items()):
        source_groups = {u['fold_group_id'] for u in part['units'].values()}
        frozen = freeze_children(part)
        foreign = [c for c in case['calls'] if case['source_by_unit'][c['unit_id']] != channel]
        for (index, (fid, f)) in enumerate(sorted(frozen.items())):
            qualified = channel + '::' + fid
            model = models[qualified]
            cache = TransferGeometryCache(f, f['grid'], 2, 16 * 1024 ** 2)
            for call in foreign:
                if not overlaps([call['start'], call['end']], model['reference_interval']):
                    continue
                group = call.get('evidence_group_id', call['unit_id'])
                if group in source_groups:
                    raise ValueError('Foreign recipient molecule seen by source model')
                score = transfer_child(f, case['units'][call['unit_id']], call, read_cache=reads, cache=cache)
                records.append(dict(hypothesis=qualified, source_channel=channel, recipient_channel=case['source_by_unit'][call['unit_id']], call_key=list(call_key(call)), fit_warning=model['fit_warning'], source_group_exclusion_verified=True, **score))
            if progress and index % 25 == 0:
                progress(channel, index + 1, len(frozen), len(records))
    return records

def cross_annotation(case, proposals, results, radius, foreign):
    """Promote from original memberships, then add only direct foreign links.

    Frozen foreign links do not alter nominations, promotion, or restore archived
    children. Archived compatibility remains explicit provenance for every event.
    """
    ann = annotate(case, proposals, results, radius)
    hypotheses = {h['id']: h for h in ann['hypotheses']}
    scores = defaultdict(list)
    for score in foreign:
        scores[tuple(score['call_key'])].append(score)
    for row in ann['records']:
        evidence = scores[(row['unit_id'], *row['interval'])]
        passing = sorted({s['hypothesis'] for s in evidence if s['compatible'] is True and (not s['fit_warning'])})
        row['foreign_child_evaluations'] = [dict(hypothesis=s['hypothesis'], compatible=s['compatible'], status=s['status'], fit_warning=s['fit_warning']) for s in evidence]
        row['foreign_compatible_children'] = passing
        row['status_before_foreign_child_links'] = row['status']
        active = {f for f in passing if hypotheses[f]['display']}
        row['display_hypotheses'] = sorted(set(row['display_hypotheses']) | active)
        if row['display_hypotheses']:
            row['status'] = 'multi_compatible' if len(row['display_hypotheses']) > 1 else 'compatible'
            row['residual_assessment'] = None
        row['source_channel'] = case['source_by_unit'][row['unit_id']]
    by_hyp = Counter((f for r in ann['records'] for f in r['display_hypotheses']))
    for h in ann['hypotheses']:
        h['active_compatible_events_including_foreign'] = by_hyp[h['id']]
        h['source_channels'] = sorted({case['model_source'][c] for c in h.get('children', [h['id']])})
    by_source = {}
    for source in case['source_channels']:
        rows = [r for r in ann['records'] if r['source_channel'] == source]
        by_source[source] = dict(original_calls=len(rows), eligible_calls=sum((r['original']['inference_eligible'] for r in rows)), assigned=sum((bool(r['display_hypotheses']) for r in rows)), multi_compatible=sum((len(r['display_hypotheses']) > 1 for r in rows)), residual=sum((r['status'] == 'residual_after_consolidation' for r in rows)), foreign_child_rescued_assignment=sum((bool(r['display_hypotheses']) and r['status_before_foreign_child_links'] not in ('compatible', 'multi_compatible') for r in rows)))
    ann['summary'].update(calls_with_any_hypothesis=sum((bool(r['display_hypotheses']) for r in ann['records'])), multi_compatible_calls=sum((len(r['display_hypotheses']) > 1 for r in ann['records'])), residual_after_consolidation=sum((r['status'] == 'residual_after_consolidation' for r in ann['records'])), status_counts=dict(Counter((r['status'] for r in ann['records']))), by_source=by_source, foreign_child_comparisons=len(foreign), foreign_child_statuses=dict(Counter((s['status'] for s in foreign))), cross_source_active_parents=sum((h['display'] and h['kind'] == 'bounded_parent' and (len(h['source_channels']) > 1) for h in ann['hypotheses'])))
    ann['settings'].update(foreign_child_floor_bp=2, foreign_child_retained_mass_threshold=0, foreign_child_all_actual_overlaps_tested=True, source_memberships_unchanged=True, cross_links_do_not_change_parent_promotion=True, abundance_estimated=False)
    return ann
