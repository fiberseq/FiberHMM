"""Native CR labels must never be replaced by an auxiliary detector's calls."""
from copy import deepcopy

import numpy as np
import pytest

from fiberhmm.inference.consensus.artifacts import digest
from fiberhmm.inference.consensus.measurement_family import classify_family_profiles
from fiberhmm.inference.consensus.native_auxiliary import (
    _source_detections, grouped_source_tables, run_native_auxiliary,
)
from fiberhmm.inference.consensus.native_auxiliary_shape import NativeAuxiliaryShape, grouped_fold
from fiberhmm.inference.consensus.parameters import parse_options


def unit(uid, strand='CT', span=(120, 144), *, called=True, group=None):
    p = np.arange(100, 201, 2)
    a, b = span
    return dict(unit_id=uid, read_name=uid, fold_group_id=group or uid, strand=strand,
        positions=p.tolist(), hits=((p < a) | (p >= b)).astype(int).tolist(),
        p_accessible=np.full(len(p), .85).tolist(), p_protected=np.full(len(p), .03).tolist(),
        contexts=[0]*len(p), representative_raw_tf_intervals=[list(span)] if called else [],
        native_multi_interval_tf_intervals=[list(span)] if called else [],
        raw_nuc_intervals=[], msp_intervals=[[100, 202]], aligned_blocks=[[100, 202]],
        _region=[100, 202], reference_start=100, reference_end=202)


@pytest.fixture(scope='module')
def native_fixture():
    units = [unit(f'{s}{i}', s, (120-i%2, 144+i%2)) for s in ('CT', 'GA') for i in range(16)]
    units += [unit('recall_good', called=False), unit('saturated', called=False),
              unit('no_information', called=False)]
    units[-2]['hits'] = [1]*len(units[-2]['positions'])
    for key in ('positions', 'hits', 'p_accessible', 'p_protected', 'contexts'):
        units[-1][key] = []
    stratum = dict(dataset_id='d', stratum_id='d', chemistry='ddda', units=units)
    catalog = [dict(family='d:F1', family_index=0, consensus_start=120, consensus_end=144)]
    native = classify_family_profiles(stratum, catalog, region=(100, 202),
        family_model='latent_distribution', minimum_edge_tolerance_bp=0,
        max_fit_iterations=100, predictive_replicates=31, scoring_folds=2)
    return stratum, catalog, native


def options(**changes):
    params = dict(input=dict(correct_native=False), sr=dict(enabled=False),
        cr=dict(ambiguity_bp=2, predictive_replicates=31, family_fit_iterations=100),
        rescue=dict(enabled=True, null_replicates=1, proposal_edge_radius_bp=2),
        comparability=dict(enabled=False), split=dict(enabled=False))
    for section, values in changes.items():
        params[section].update(values)
    return parse_options(params)


def test_grouped_source_tables_count_each_physical_group_once_and_restrict_donor():
    data = dict(strands=np.array(['CT', 'GA', 'GA', 'GA']),
        fold_group_ids=['g', 'g', 'g', 'h'], unit_ids=['ct', 'ga0', 'ga1', 'ga2'])
    membership = np.array([[1.], [0.], [1.], [1.]])
    eligible = np.ones_like(membership, bool)
    m, e = grouped_source_tables(membership, eligible, data, 'CT', 'opposite_strand')
    assert m[:, 0].tolist() == [0, 0, 1, 1]
    assert e[:, 0].tolist() == [False, False, True, True]
    assert grouped_fold('g') == grouped_fold(data['fold_group_ids'][2])


def test_native_shape_refit_excludes_recipient_group_and_wrong_donor(native_fixture):
    s, _, native = native_fixture
    validator = NativeAuxiliaryShape(s, native, mode='opposite_strand',
                                    predictive_replicates=31, minimum_source_units=2)
    u = s['units'][-3]
    record = validator.validate(u, (120, 144), 'd:F1')
    assert record['accepted'], record
    assert record['extra_boundary_tolerance_bp'] == 0
    assert u['fold_group_id'] not in record['training_evidence_groups']
    assert all(name.startswith('GA') for name in record['training_unit_ids'])
    assert all(grouped_fold(g) != grouped_fold(u['fold_group_id']) for g in record['training_evidence_groups'])
    assert not record['native_catalog_refit'] and not record['source_calls_added']
    assert not record['native_shape_evidence']['minimum_edge_floor_applied']


def test_auxiliary_native_shape_veto_survives_high_native_class_compatibility(native_fixture):
    s, _, native = native_fixture
    validator = NativeAuxiliaryShape(s, native, mode='opposite_strand',
                                    predictive_replicates=31, minimum_source_units=2)
    record = validator.validate(s['units'][-2], (120, 144), 'd:F1')
    assert not record['accepted']
    assert record['status'] == 'core_contradicted'
    assert record['native_shape_evidence']['mean_core_native_log_lr'] < -np.log(100)


def test_zero_recipient_information_is_not_native_shape_support(native_fixture):
    s, _, native = native_fixture
    validator = NativeAuxiliaryShape(s, native, mode='opposite_strand',
                                    predictive_replicates=31, minimum_source_units=2)
    record = validator.validate(s['units'][-1], (120, 144), 'd:F1')
    assert not record['accepted'] and record['status'] == 'no_recipient_information'


def test_no_same_group_self_training_even_with_many_source_records(native_fixture):
    s, _, native = deepcopy(native_fixture)
    # Group transport deliberately makes every source a mate of this recipient.
    # The original catalog can still exist, but no independent fit may use it.
    target = s['units'][-3]
    for c in native['calls']:
        c['evidence_group_id'] = target['fold_group_id']
    validator = NativeAuxiliaryShape(s, native, mode='opposite_strand', predictive_replicates=31)
    record = validator.validate(target, (120, 144), 'd:F1')
    assert not record['accepted']
    assert record['status'] == 'insufficient_independent_native_shape_sources'
    assert record['training_evidence_groups'] == []


def test_shape_budget_errors_never_sample_sources(native_fixture):
    s, _, native = native_fixture
    validator = NativeAuxiliaryShape(s, native, maximum_matrix_bytes=1)
    with pytest.raises(MemoryError, match='no source sampling'):
        validator.validate(s['units'][-3], (120, 144), 'd:F1')


def test_frozen_unit_indices_cannot_silently_refer_to_different_molecules(native_fixture):
    s, _, native = deepcopy(native_fixture)
    s['units'].reverse()
    with pytest.raises(ValueError, match='evidence-unit identities'):
        NativeAuxiliaryShape(s, native)


def test_low_draw_zero_tail_is_unresolved_not_automatic_shape_compatibility(native_fixture, monkeypatch):
    from fiberhmm.inference.consensus import native_auxiliary_shape
    monkeypatch.setattr(native_auxiliary_shape, 'transferred_call', lambda *a, **kw: dict(
        status='scored', native_loss=15., predictive_tail_interval=[0., .11], predictive_tail=1/32.))
    s, _, native = native_fixture
    validator = NativeAuxiliaryShape(s, native, mode='opposite_strand', predictive_replicates=31)
    result = validator.validate(s['units'][-3], (120, 144), 'd:F1')
    assert not result['accepted']
    assert result['status'] == 'native_shape_reference_underresolved'


def test_classification_without_positive_native_protection_cannot_seed_recall(native_fixture):
    s, cat, native = deepcopy(native_fixture)
    # Classification records remain frozen; the source-detection firewall must
    # still reject an observed saturated core rather than trust a color.
    s['units'][0]['hits'] = [1]*len(s['units'][0]['positions'])
    membership, _, ledger = _source_detections(s, cat, native, 100.)
    assert membership[0, 0] == 0
    assert any(r['unit_id'] == s['units'][0]['unit_id'] and not r['direct_native_detection'] for r in ledger)


def test_strict_rescue_adds_supported_missing_call_without_old_cr_reassignment(native_fixture, tmp_path, monkeypatch):
    from fiberhmm.inference.consensus import stages, workflow
    def forbidden(*_, **__):
        raise AssertionError('Legacy CR must not replace native classification')
    monkeypatch.setattr(stages, 'assign_cr', forbidden)
    monkeypatch.setattr(workflow, 'assign_cr', forbidden)
    monkeypatch.setattr(stages, 'discover_cr', forbidden)
    s, cat, native = native_fixture
    frozen = digest([s, cat, native])
    out = run_native_auxiliary(s, cat, native, options(), tmp_path)['rescue']
    assert out['status'] == 'complete'
    called = {r['unit_id']: r['calls'] for r in out['records']}
    assert called['recall_good'], out
    assert not called['saturated'] and not called['no_information']
    assert all(not called[u['unit_id']] for u in s['units'][:-3])
    assert out['cr_records'] == []  # In particular, no old lattice re-labeling.
    assert out['native_cr_records_unchanged']
    assert not out['recalled_calls_train_catalog'] and not out['recalled_calls_supply_cross_support']
    assert all(c['native_family_validation']['accepted'] for r in out['records'] for c in r['calls'])
    assert digest([s, cat, native]) == frozen


def test_opposite_strand_not_applicable_does_not_block_pooled_assay(native_fixture, tmp_path, monkeypatch):
    from fiberhmm.inference.consensus import native_auxiliary
    def unnecessary(*_):
        raise AssertionError('Inapplicable stage must not construct a lattice')
    monkeypatch.setattr(native_auxiliary, '_prepare_lattice', unnecessary)
    s, cat, native = deepcopy(native_fixture)
    s['chemistry'] = 'hia5-pacbio'
    for u in s['units']:
        u['strand'] = 'pooled'
    out = run_native_auxiliary(s, cat, native, options(), tmp_path)['rescue']
    assert out['status'] == 'not_applicable'
    assert out['records'] == []


@pytest.mark.parametrize('limited_recall', [False, True])
def test_nucleosome_split_requires_independent_native_family_piece(tmp_path, monkeypatch, limited_recall):
    if limited_recall:
        from fiberhmm.inference.consensus import native_auxiliary
        def limited(*_):
            raise MemoryError('Explicit auxiliary recall budget test')
        monkeypatch.setattr(native_auxiliary, '_prepare_lattice', limited)
    sources = [unit(f's{i}', 'pooled', (110, 136)) for i in range(20)]
    target = unit('split_target', 'pooled', (110, 174), called=False)
    target['raw_nuc_intervals'] = [[110, 174]]
    target['hits'] = [int(p < 110 or p >= 174 or 136 <= p < 150) for p in target['positions']]
    intact = deepcopy(target); intact['unit_id'] = intact['fold_group_id'] = 'intact'
    intact['hits'] = [int(p < 110 or p >= 174) for p in intact['positions']]
    s = dict(dataset_id='d', stratum_id='d', chemistry='hia5-pacbio', units=sources+[target, intact])
    cat = [dict(family='d:F1', family_index=0, consensus_start=110, consensus_end=136)]
    native = classify_family_profiles(s, cat, region=(100, 202), family_model='latent_distribution',
        max_fit_iterations=100, predictive_replicates=31, scoring_folds=2)
    before = digest([s, cat, native])
    stages = run_native_auxiliary(s, cat, native, options(rescue=dict(enabled=limited_recall, source_mode='pooled'),
        split=dict(enabled=True, ambiguity_bp=2, require_nuc_model=False, null_replicates=1)), tmp_path)
    out = stages['split']
    if limited_recall:
        assert stages['rescue']['status'] == 'resource_limited'
    assert out['accepted_spans'] == 1, out
    row = out['records'][0]
    assert row['unit_id'] == 'split_target'
    assert row['original_nuc'] == [110, 174]
    assert any(v['accepted'] for v in row['independent_native_piece_validations'])
    assert out['outer_boundaries_unchanged'] and not out['raw_nucleosomes_modified']
    assert digest([s, cat, native]) == before
