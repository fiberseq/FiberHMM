import copy
import json

import numpy as np
import pytest

from fiberhmm.inference.consensus.artifacts import digest, json_default
from fiberhmm.inference.consensus.measurement_distribution import fit_native_distribution
from fiberhmm.inference.consensus.measurement_family import _native_values
from fiberhmm.inference.consensus.measurement_grouping import _calls
from fiberhmm.inference.consensus.native_family_consolidation import (
    consolidate_native_families, restore_consolidated_geometry)


def fixture():
    p = np.arange(0, 61, 3)
    units = []
    for i in range(6):
        a, b = (12, 36) if i < 3 else (13, 37)
        units.append(dict(unit_id=f'u{i}', strand='CT' if i % 2 else 'GA',
            positions=p.tolist(), hits=((p < 12) | (p >= 36)).astype(int).tolist(),
            p_accessible=np.full(len(p), .85).tolist(), p_protected=np.full(len(p), .02).tolist(),
            representative_raw_tf_intervals=[[a, b]], raw_nuc_intervals=[]))
    data = dict(dataset_id='d', chemistry='ddda', units=units)
    calls = _calls(data)
    evidence = []
    for call in calls:
        fid = 'd:A' if int(call['unit_id'][1:]) < 3 else 'd:B'
        evidence.append([dict(family=fid, status='scored', predictive_tail_interval=[.4, .6],
                             geometry_distance_sq=0, floor_adjusted_loss=0.)])
    models = [dict(family=f, status='fitted', family_model='latent_distribution',
                   reference_interval=[12, 36] if f == 'd:A' else [13, 37], domain=[0, 61],
                   fold_models={'full': dict(parameters=[0, 0, 0, 0, 0])}) for f in ('d:A', 'd:B')]
    return data, dict(calls=calls, call_family_evidence=evidence, family_models=models)


def run(data, result, **kwargs):
    return consolidate_native_families(data, result, ['d:B', 'd:A'],
        parent_family_id='d:parent', region=(0, 61), **kwargs)


def test_genuine_parent_fit_preserves_native_inputs_and_has_no_oof_assignments():
    data, result = fixture(); before = digest([data, result])
    out = run(data, result)
    assert digest([data, result]) == before
    assert out['model']['source_units'] == 6
    assert out['model']['child_family_ids'] == ['d:A', 'd:B']
    assert set(out['model']['fold_models']) == {'full'}
    assert out['model']['parameters_leave_evidence_group_out'] is False
    assert out['provenance']['parent_per_molecule_assignments_published'] is False
    assert out['provenance']['old_models_or_scores_averaged'] is False
    assert out['model']['reference_interval'] == [12.5, 36.5]
    assert len(out['source_calls']) == 6
    assert out['model']['fit_diagnostics']['full']['converged']
    json.dumps(out, default=json_default, allow_nan=False)


def test_fit_equals_direct_pooled_native_fitter_on_same_domain_and_grid():
    data, result = fixture(); out = run(data, result)
    grid = out['grid']; a, b = grid['starts'], grid['ends']
    by_id = {u['unit_id']: u for u in data['units']}; profiles = []; masks = []
    for row in out['source_calls']:
        call = result['calls'][row['source_call_index']]
        values, _ = _native_values(by_id[call['unit_id']], grid['positions'], call)
        prefix = np.r_[0., np.cumsum(values)]
        ca, cb = np.searchsorted(grid['positions'], [call['start'], call['end']])
        profiles.append(prefix[b] - prefix[a]); masks.append((a < cb) & (b > ca))
    fit = fit_native_distribution(profiles, masks, grid['coordinates'], grid['areas'],
                                  reference=out['model']['reference_interval'])
    assert fit['parameters'] == pytest.approx(out['model']['fold_models']['full']['parameters'])
    assert fit['objective'] == pytest.approx(out['model']['fit_diagnostics']['full']['objective'])


def test_duplicate_group_uses_one_call_with_deterministic_geometry_not_score():
    data, result = fixture()
    for u in data['units'][:2]: u['fold_group_id'] = 'same_physical_group'
    for call in result['calls']:
        if call['unit_id'] in ('u0', 'u1'): call['evidence_group_id'] = 'same_physical_group'
    # Changing selected-call evidence strength cannot weight duplicate choice.
    for call, scores in zip(result['calls'], result['call_family_evidence']):
        if call['unit_id'] == 'u1': scores[0]['predictive_tail_interval'] = [.99, 1.]
    out = run(data, result)
    assert out['model']['source_units'] == 5
    assert [r['unit_id'] for r in out['excluded_duplicate_calls']] == ['u1']
    assert [r['unit_id'] for r in out['source_calls'] if r['evidence_group_id'] == 'same_physical_group'] == ['u0']
    assert out['provenance']['primary_union_calls'] == 6
    assert out['provenance']['primary_union_evidence_groups'] == 5


def test_unit_child_and_call_order_do_not_change_actual_fit():
    data, result = fixture(); first = run(data, result)
    data['units'].reverse()
    result['family_models'].reverse()
    result['calls'].reverse(); result['call_family_evidence'].reverse()
    second = run(data, result)
    assert second['model']['fold_models']['full']['parameters'] == pytest.approx(first['model']['fold_models']['full']['parameters'])
    assert second['provenance']['projection_grid_sha256'] == first['provenance']['projection_grid_sha256']
    assert second['provenance']['likelihood_profiles_sha256'] == first['provenance']['likelihood_profiles_sha256']
    assert [r['source_call_id'] for r in second['source_calls']] == [r['source_call_id'] for r in first['source_calls']]


def test_child_parameter_changes_cannot_replace_observation_refit():
    data, result = fixture(); first = run(data, result)
    for m in result['family_models']:
        m['fold_models']['full']['parameters'] = [999., -100., 6., 5., -6.]
    changed_parameters = run(data, result)
    assert changed_parameters['model']['fold_models']['full']['parameters'] == pytest.approx(first['model']['fold_models']['full']['parameters'])
    # Same original geometry; actual observation changes still change the fit.
    data['units'][0]['hits'][6] = 1
    changed_data = run(data, result)
    assert changed_data['provenance']['likelihood_profiles_sha256'] != first['provenance']['likelihood_profiles_sha256']
    assert changed_data['model']['fit_diagnostics']['full']['objective'] != pytest.approx(first['model']['fit_diagnostics']['full']['objective'])


def test_json_roundtrip_preserves_exact_frozen_grid_and_rejects_tampering():
    data, result = fixture(); out = run(data, result)
    portable = json.loads(json.dumps(out, default=json_default))
    frozen = restore_consolidated_geometry(portable)
    for key in out['grid']: assert np.array_equal(frozen['grid'][key], out['grid'][key])
    portable['grid']['areas'][0] += 1
    with pytest.raises(ValueError, match='digest'): restore_consolidated_geometry(portable)


def test_neighbors_condition_likelihood_and_whole_geometry_without_deleting_calls():
    data, result = fixture()
    data['units'][0]['representative_raw_tf_intervals'].append([42, 58])
    calls = _calls(data); old = {(c['unit_id'], c['ordinal']): e for c, e in zip(result['calls'], result['call_family_evidence'])}
    result['calls'] = calls
    result['call_family_evidence'] = [old.get((c['unit_id'], c['ordinal']), []) for c in calls]
    before = digest([data, result]); out = run(data, result)
    row = next(r for r in out['source_calls'] if r['unit_id'] == 'u0')
    assert row['frozen_neighbor_limits'] == [0, 42]
    assert row['observed_opportunities'] == 15  # 45..57 and 42 conditioned; own core is retained.
    assert digest([data, result]) == before


def test_no_information_is_reported_not_replaced_by_another_same_group_call():
    data, result = fixture()
    data['units'][0]['p_protected'] = list(data['units'][0]['p_accessible'])
    out = run(data, result)
    assert out['model']['source_units'] == 5
    assert out['excluded_source_calls'][0]['reason'] == 'no_native_information'


def test_bad_axes_ids_empty_cohort_and_budget_fail_without_truncation():
    data, result = fixture()
    with pytest.raises(MemoryError, match='no sources/projections removed'):
        run(data, result, maximum_matrix_bytes=1)
    bad = copy.deepcopy(result); bad['calls'][0]['start'] += 1
    with pytest.raises(ValueError, match='identity/span'): run(data, bad)
    bad = copy.deepcopy(result); bad['call_family_evidence'] = [[] for _ in bad['calls']]
    with pytest.raises(ValueError, match='No primary'): run(data, bad)
    with pytest.raises(ValueError, match='overwrite'):
        consolidate_native_families(data, result, ['d:A', 'd:B'], parent_family_id='d:A', region=(0, 61))
    with pytest.raises(ValueError, match='distinct'):
        consolidate_native_families(data, result, ['d:A', 'd:A'], parent_family_id='d:C', region=(0, 61))


def test_nonconvergence_is_preserved_explicitly(monkeypatch):
    from fiberhmm.inference.consensus import native_family_consolidation as module
    real = module.fit_native_distribution
    def nonconverged(*args, **kwargs):
        result = real(*args, **kwargs)
        result.update(converged=False, message='test numerical limit')
        return result
    monkeypatch.setattr(module, 'fit_native_distribution', nonconverged)
    data, result = fixture(); out = run(data, result)
    assert out['model']['fit_diagnostics']['full']['converged'] is False
    assert out['model']['fit_diagnostics']['full']['message'] == 'test numerical limit'
