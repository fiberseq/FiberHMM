import copy

import pytest

from test_consensus_measurement_family import fixture
from fiberhmm.inference.consensus.measurement_family import classify_family_profiles


def run(source, catalog, **kwargs):
    return classify_family_profiles(source, catalog, region=(0, 81), family_model='latent_distribution',
        training_evidence_groups={'u0', 'u1', 'u3', 'u4'}, predictive_replicates=31,
        scoring_folds=2, edge_tolerance_mode='bounded', **kwargs)


def test_confirmation_outcomes_and_calls_cannot_change_training_models():
    source, catalog = fixture()
    original = copy.deepcopy(source)
    a = run(source, catalog)
    for index in (2, 5):
        unit = source['units'][index]
        unit['hits'] = [1-h for h in unit['hits']]
        unit['representative_raw_tf_intervals'] = [[1, 79]]
    b = run(source, catalog)
    for x, y in zip(a['family_models'], b['family_models']):
        # The sorted complete call axis changes when confirmation spans change;
        # resolve source indices to physical identities before comparison.
        sources = lambda result, model: [result['calls'][i]['unit_id'] for i in model['source_call_indices']]
        assert sources(a, x) == sources(b, y)
        assert {k: v for k, v in x.items() if k != 'source_call_indices'} == {
            k: v for k, v in y.items() if k != 'source_call_indices'}
    assert a['call_family_evidence'] != b['call_family_evidence']
    for m in a['family_models']:
        for groups in m['training_evidence_groups'].values():
            assert set(groups) <= {'u0', 'u1', 'u3', 'u4'}
    # Training inputs are unchanged as well.
    for index in (0, 1, 3, 4):
        assert source['units'][index] == original['units'][index]


def test_training_contract_reaches_parallel_workers():
    source, catalog = fixture()
    assert run(source, catalog, cores=1) == run(source, catalog, cores=2)
    assert run(source, catalog, cores=1, max_fit_iterations=1, retry_fit_iterations=500) == run(
        source, catalog, cores=2, max_fit_iterations=1, retry_fit_iterations=500)


def test_failed_fits_are_retried_and_recorded():
    source, catalog = fixture()
    result = run(source, catalog, max_fit_iterations=1, retry_fit_iterations=500)
    assert any(m['fit_retry_diagnostics'] for m in result['family_models'])
    assert all(d['initial_iterations'] <= 1 for m in result['family_models']
               for d in m['fit_retry_diagnostics'].values())
    assert all(d['retry_objective'] <= d['initial_objective'] + 1e-7
               for m in result['family_models'] for d in m['fit_retry_diagnostics'].values() if d['accepted'])


def test_invalid_training_and_retry_requests_fail():
    source, catalog = fixture()
    for groups in ([], {'absent'}, 'u0'):
        with pytest.raises(ValueError, match='training'):
            classify_family_profiles(source, catalog, region=(0, 81), training_evidence_groups=groups)
    with pytest.raises(ValueError, match='Retry'):
        run(source, catalog, retry_fit_iterations=50)
