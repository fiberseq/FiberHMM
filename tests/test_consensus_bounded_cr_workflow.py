"""The bounded allowance must reach native CR and its worker/provenance paths."""
import copy

import numpy as np
import pytest

from fiberhmm.inference.consensus.measurement_family import classify_family_profiles
from fiberhmm.inference.consensus.native_catalog_update import _validate_options
from fiberhmm.inference.consensus.native_workflow import _fit_kwargs, _empty_native_result
from fiberhmm.inference.consensus.parameters import parse_options


def fixture():
    positions = np.arange(0, 65, 3)
    units = []
    for i, (start, end) in enumerate([(12, 48), (11, 49), (12, 47), (12, 27), (11, 28), (12, 26)]):
        units.append(dict(unit_id=f'u{i}', strand='CT', positions=positions.tolist(),
            hits=((positions < start) | (positions >= end)).astype(int).tolist(),
            p_accessible=np.full(len(positions), .85).tolist(),
            p_protected=np.full(len(positions), .02).tolist(),
            representative_raw_tf_intervals=[[start, end]], raw_nuc_intervals=[]))
    return dict(dataset_id='d', chemistry='ddda', units=units), [
        dict(family='d:broad', consensus_start=12, consensus_end=48),
        dict(family='d:small', consensus_start=12, consensus_end=27)]


def run(floor, mode='bounded', cores=1):
    source, catalog = fixture()
    return classify_family_profiles(source, catalog, region=(0, 65), family_model='latent_distribution',
        minimum_edge_tolerance_bp=floor, edge_tolerance_mode=mode,
        predictive_replicates=63, scoring_folds=3, cores=cores)


def test_bounded_zero_reproduces_legacy_scores_and_all_fits():
    old, new = run(0, 'legacy_profile'), run(0)
    assert old['family_models'] == new['family_models']
    assert old['calls'] == new['calls']
    for before, after in zip(old['call_family_evidence'], new['call_family_evidence']):
        for a, b in zip(before, after):
            assert all(b[k] == v for k, v in a.items())


def test_larger_bounded_allowance_never_increases_raw_mismatch():
    results = [run(floor) for floor in (0, 5, 10, 15)]
    assert all(r['family_models'] == results[0]['family_models'] for r in results)
    assert all(r['calls'] == results[0]['calls'] for r in results)
    for a, b in zip(results, results[1:]):
        for before, after in zip(a['call_family_evidence'], b['call_family_evidence']):
            for x, y in zip(before, after):
                if 'floor_adjusted_loss' in x:
                    assert y['floor_adjusted_loss'] <= x['floor_adjusted_loss'] + 1e-10
                    assert y['native_loss'] == x['native_loss']
                    assert y['edge_tolerance_changes_generative_mass'] is False
                    assert y['edge_tolerance_semantics'] == 'bounded_joint_endpoint_cell_profile_v1'


def test_worker_mode_matches_serial_exactly():
    assert run(5, cores=2) == run(5, cores=1)


def test_workflow_option_and_empty_binding_carry_mode():
    options = parse_options({'cr': {'edge_tolerance_mode': 'bounded'}})
    source, _ = fixture()
    kwargs = _fit_kwargs(source, options['cr'], options['compute'])
    assert kwargs['edge_tolerance_mode'] == 'bounded'
    empty = _empty_native_result(source, (0, 65), kwargs)
    assert _validate_options(kwargs, empty['diagnostics'])['edge_tolerance_mode'] == 'bounded'
    wrong = copy.deepcopy(empty['diagnostics']); wrong['edge_tolerance_mode'] = 'legacy_profile'
    with pytest.raises(ValueError, match='edge_tolerance_mode'):
        _validate_options(kwargs, wrong)
    legacy = dict(kwargs); legacy.pop('edge_tolerance_mode')
    with pytest.raises(ValueError, match='edge_tolerance_mode'):
        _validate_options(legacy, empty['diagnostics'])


def test_mode_cannot_silently_fall_back_or_reuse_other_mode():
    source, catalog = fixture()
    with pytest.raises(ValueError, match='edge_tolerance_mode'):
        classify_family_profiles(source, catalog, region=(0, 65), edge_tolerance_mode='unknown')
    with pytest.raises(ValueError, match='latent_distribution'):
        classify_family_profiles(source, catalog, region=(0, 65), edge_tolerance_mode='bounded')
    frozen = run(5, 'legacy_profile')
    with pytest.raises(ValueError, match='edge tolerance'):
        classify_family_profiles(source, catalog, region=(0, 65), family_model='latent_distribution',
            edge_tolerance_mode='bounded', minimum_edge_tolerance_bp=5, _frozen_result=frozen)
