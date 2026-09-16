# Extracted reference kernels; see SOURCE_MANIFEST.json.
from collections import Counter
from copy import deepcopy
from .synthetic_state_benchmark import read_json, digest
from .raw_family_segmented_region import prepare_stratum
from .windowless_llr_inventory import call_ledger

def prepare_input(source, region, strand=None):
    source = deepcopy(source)
    if strand is not None:
        source['units'] = [u for u in source['units'] if u['strand'] == strand]
    (units, _, _, stats) = prepare_stratum(source, region)
    (ledger, _, ledger_stats) = call_ledger(source, region, units)
    lookup = {u['unit_id']: i for (i, u) in enumerate(units)}
    original_by_id = {source['dataset_id'] + '::' + u['unit_id']: u for u in source['units']}
    assert len(units) == len(source['units']) and set(lookup) == set(original_by_id)
    for u in units:
        original = original_by_id[u['unit_id']]
        u['representative_raw_tf_intervals'] = []
        u['raw_nuc_intervals'] = original['raw_nuc_intervals']
        u['source_unit_id'] = original['unit_id']
    retained = []
    for c in ledger:
        c['strand'] = units[lookup[c['unit_id']]]['strand']
        c['rerun_touches_outer_boundary'] = bool(source.get('model_manifest', {}).get('call_layer') == 'matched_explicit_native_llr_sweep' and (c['interval'][0] == region[0] or c['interval'][1] == region[1]))
        if c['scaffold_status'] == 'fully_in_msp_scaffold' and (not c['outer_region_censored']) and (not c['rerun_touches_outer_boundary']):
            units[lookup[c['unit_id']]]['representative_raw_tf_intervals'].append(c['interval'])
            c['inference_eligible'] = True
        else:
            c['inference_eligible'] = False
        retained.append(c)
    dataset = source['dataset_id'] + ('::' + strand if strand else '::pooled_SR')
    return (dict(dataset_id=dataset, chemistry=source['chemistry'], units=units), retained, dict(preparation=stats, ledger=ledger_stats, source_sha256=digest(source['units']), inference_eligible_calls=sum((c['inference_eligible'] for c in retained)), call_layer=source.get('model_manifest', {}).get('call_layer', 'original_saved_raw_tf_intervals')))

def summarize(native, ledger, reference, catalog=None):
    partition = native['predictive_partitions'][str(float(reference))]
    by_call = {(c['unit_id'], tuple(c['interval'])): c for c in partition['assignments']}
    assert len(by_call) == len(partition['assignments'])
    assert set(by_call) == {(c['unit_id'], tuple(c['interval'])) for c in ledger if c['inference_eligible']}
    models = {m['family']: m for m in native['family_models']}
    evidence = {(c['unit_id'], (c['start'], c['end'])): row for (c, row) in zip(native['calls'], native['call_family_evidence'])}
    geometries = {c['family']: [c['consensus_start'], c['consensus_end']] for c in catalog} if catalog else {m['family']: m['reference_interval'] for m in native['family_models'] if 'reference_interval' in m}
    fit_warnings = [dict(family=m['family'], fold=fold, diagnostic=d) for m in native['family_models'] for (fold, d) in m.get('fit_diagnostics', {}).items() if not d['converged']]
    warned = {w['family'] for w in fit_warnings}
    records = []
    for original in ledger:
        c = dict(original)
        assignment = by_call.get((c['unit_id'], tuple(c['interval'])))
        if not c['inference_eligible']:
            c.update(assignment_status='not_evaluated', compatible_families=[], primary_display_family=None)
        else:
            if assignment is None:
                raise AssertionError('Eligible original call missing from native output')
            assigned = assignment['classification_status'] == 'compatible_catalog_label'
            compatible = [assignment['family']] + assignment['compatible_alternatives'] if assigned else []
            row = evidence[c['unit_id'], tuple(c['interval'])]
            assert len(compatible) == len(set(compatible))
            assert set(compatible) == {s['family'] for s in row if s['status'] == 'scored' and s.get('predictive_tail_interval', [0.0, 0.0])[1] >= 1 - reference / 100.0}
            for score in row:
                (a, b) = geometries[score['family']]
                assert max(a, c['interval'][0]) < min(b, c['interval'][1])
            overlap_count = sum((max(a, c['interval'][0]) < min(b, c['interval'][1]) for (a, b) in geometries.values()))
            status_counts = Counter((s['status'] for s in row))
            reason = 'no_overlapping_seed' if not overlap_count else 'no_fitted_candidate_evidence' if not row else 'unresolved_with_fit_warning' if any((s['family'] in warned for s in row)) else 'candidates_partly_unscored' if any((s['status'] not in ('scored', 'core_contradicted') for s in row)) else 'all_tested_candidates_rejected_or_core_contradicted'
            c.update(assignment_status=('ambiguous' if len(compatible) > 1 else 'assigned') if assigned else 'unresolved', compatible_families=compatible, primary_display_family=assignment['family'] if assigned else None, primary_evidence=assignment['primary_evidence'], raw_classification_status=assignment['classification_status'], overlapping_seed_count=overlap_count, evaluated_candidate_count=len(row), candidate_status_counts=dict(status_counts), unresolved_reason=None if assigned else reason, evaluated_model_fit_warnings=sorted({s['family'] for s in row if s['family'] in warned}), compatible_model_fit_warnings=[f for f in compatible if any((not d['converged'] for d in models[f].get('fit_diagnostics', {}).values()))])
        records.append(c)
    diagnostics = dict(all_calls_preserved=len(records) == len(ledger), calls=len(records), assignment_counts=dict(Counter((c['assignment_status'] for c in records))), nominated_models=len(native['family_models']), fitted_models=sum((m['status'] == 'fitted' for m in native['family_models'])), unconverged_model_fits=sum((not d['converged'] for m in native['family_models'] for d in m.get('fit_diagnostics', {}).values())), actual_simulations=sum((s.get('simulations', 0) for row in native['call_family_evidence'] for s in row)), simulation_requests=sum((s.get('simulations', 0) > 0 for row in native['call_family_evidence'] for s in row)), exact_zero_loss_shortcuts=sum((s.get('predictive_tail_interval') == [1.0, 1.0] and s.get('simulations') == 0 for row in native['call_family_evidence'] for s in row)), predictive_reference_percent=reference, fit_warnings=fit_warnings, all_model_fits_converged=not fit_warnings, unresolved_reasons=dict(Counter((c.get('unresolved_reason') for c in records if c['assignment_status'] == 'unresolved'))), assigned_family_hypotheses=len({f for c in records for f in c['compatible_families']}), statistical_decision_unchanged=True, biological_family_count_certified=False)
    return (records, diagnostics)
