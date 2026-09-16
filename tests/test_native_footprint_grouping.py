"""Operational native interval comparisons, without assignment or edge edits."""
import math

import numpy as np
import pytest
from scipy.optimize import minimize
from scipy.special import logsumexp

from fiberhmm.inference.native_footprint_grouping import _fit_mixtures, compare_pairs, compare_group


@pytest.mark.parametrize('seed', range(5))
def test_tiny_mixture_matches_independent_constrained_likelihood_maximum(seed):
    rng = np.random.default_rng(seed + 12)
    logs = np.c_[np.zeros(150), rng.normal(-1, 2, (150, 2))]
    fitted = _fit_mixtures(logs[None, :, :], max_iterations=256, tolerance=1e-8)
    def objective(w):
        scaled = np.exp(logs - logs.max(axis=1)[:, None])
        z = scaled @ w
        return -np.sum(np.log(z) + logs.max(axis=1)), -np.sum(scaled / z[:, None], axis=0)
    reference = minimize(objective, np.full(3, 1/3), jac=True, method='SLSQP',
                         bounds=[(0, 1)] * 3, constraints={'type':'eq','fun':lambda w:w.sum()-1,'jac':lambda w:np.ones(3)},
                         options={'maxiter':1000,'ftol':1e-11})
    assert reference.success
    assert fitted['converged'][0]
    assert fitted['log_likelihood'][0] == pytest.approx(-reference.fun, abs=2e-5)
    assert fitted['weights'][0] == pytest.approx(reference.x, abs=2e-5)
    assert fitted['gap_per_unit'][0] <= 1e-8


def planted(n=600):
    labels = np.arange(n) % 5
    evidence = np.full((n, 2), -10.)
    evidence[labels == 0, 0] = 15
    evidence[labels == 1, 1] = 15
    return evidence, (np.arange(n) // 5) % 2


def test_identical_observation_columns_are_operationally_compatible():
    evidence, folds = planted()
    evidence[:, 1] = evidence[:, 0]
    result = compare_pairs(evidence, [[0, 1]], folds)[0]
    assert result['converged'] and result['compatible']
    assert not result['distinguishable']
    assert result['gain'] == pytest.approx(0, abs=1e-5)
    assert all([1,2] in fold['h2_identical_training_component_pairs'] for fold in result['folds'])


def test_two_native_patterns_are_distinguished_by_held_out_evidence():
    evidence, folds = planted()
    result = compare_pairs(evidence, [[0, 1]], folds)[0]
    assert result['converged'] and result['distinguishable']
    assert result['gain'] > result['threshold']
    assert result['all_single_baselines_beaten']
    for fold in result['folds']:
        assert fold['h2_weights'] == pytest.approx([.6, .2, .2], abs=1e-3)
        assert fold['train_log_likelihood_h2'] >= fold['train_log_likelihood_h1'] - 1e-5


def test_outcome_free_rows_add_no_fit_or_score_evidence():
    evidence, folds = planted()
    before = compare_pairs(evidence, [[0, 1]], folds)[0]
    after = compare_pairs(np.r_[evidence, np.zeros((200, 2))], [[0, 1]], np.r_[folds, np.arange(200)%2])[0]
    assert after['gain'] == pytest.approx(before['gain'], abs=1e-9)
    assert after['se'] == pytest.approx(before['se'], abs=1e-9)
    assert after['informative_units'] == before['informative_units']
    assert after['folds'][0]['h2_weights'] == pytest.approx(before['folds'][0]['h2_weights'], abs=1e-10)


def test_h1_geometry_selection_uses_training_fold_only():
    evidence = np.full((200, 2), -4.)
    folds = np.arange(200)%2
    evidence[folds==0,0] = 4
    evidence[folds==1,1] = 4
    result = compare_pairs(evidence, [[0,1]], folds)[0]
    for fold in result['folds']:
        assert fold['h1_geometry'] == 1 - fold['held_fold']


def test_training_winner_oscillation_cannot_manufacture_resolution_split():
    # Each training fold slightly favors the opposite fixed geometry from the
    # held fold. A soft mixture beats that switching, deliberately poor H1 by
    # 54 nats, but loses to BOTH fixed cross-predictive H1s by 74 nats.
    units = 4000
    folds = np.arange(units) // (units // 2)
    difference = np.where(folds == 0, .032, -.032) + np.where(np.arange(units) % 2, .2, -.2)
    evidence = np.c_[10 + difference, 10 - difference]
    pair = compare_pairs(evidence, [[0,1]], folds, max_iterations=1000)[0]
    assert pair['converged']
    assert pair['training_selected_gain'] > pair['threshold']
    assert pair['training_selected_gain'] == pytest.approx(54.06088091, abs=1e-5)
    assert all(check['gain'] < -70 for check in pair['candidate_baseline_checks'])
    assert not pair['all_single_baselines_beaten']
    assert pair['compatible'] and not pair['distinguishable']
    group = compare_group(evidence, folds, max_iterations=1000)
    assert group['compatible'] and not group['distinguishable']
    for left, right in zip(group['candidate_baseline_checks'], pair['candidate_baseline_checks']):
        assert left['geometry_index'] == right['geometry_index']
        assert left['beaten'] == right['beaten']
        for key in ('gain', 'se', 'threshold'):
            assert left[key] == pytest.approx(right[key], abs=1e-7)


def test_nonconvergence_never_becomes_compatibility():
    rng = np.random.default_rng(9)
    evidence = rng.normal(0, 2, (200, 2))
    result = compare_pairs(evidence, [[0,1]], np.arange(200)%2, max_iterations=1, tolerance=1e-16)[0]
    assert not result['converged']
    assert not result['compatible']
    assert not result['distinguishable']
    assert result['status'] == 'unresolved_numerics'


def test_pair_and_group_comparisons_agree_for_two_members():
    evidence, folds = planted()
    a = compare_pairs(evidence, [[0,1]], folds)[0]
    b = compare_group(evidence, folds)
    assert b['gain'] == pytest.approx(a['gain'], abs=1e-8)
    assert b['se'] == pytest.approx(a['se'], abs=1e-8)
    assert b['distinguishable'] == a['distinguishable']


def test_rare_alternative_is_not_removed_by_one_percent_weight_floor():
    evidence = np.full((4000,2), -200.)
    evidence[:800,0] = 200
    evidence[800:820,1] = 200
    result = compare_pairs(evidence, [[0,1]], np.arange(4000)%2)[0]
    assert result['converged'] and result['distinguishable']
    assert all(1e-8 < fold['h2_weights'][2] < .01 for fold in result['folds'])


def test_all_missing_is_zero_gain_and_no_native_calls_are_generated():
    result = compare_group(np.zeros((20,3)), np.arange(20)%2)
    assert result['converged'] and result['compatible']
    assert result['gain'] == result['se'] == result['informative_units'] == 0
    assert not any(key in result for key in ('start','end','assignments','rescued'))


def test_invalid_inputs_rejected():
    with pytest.raises(ValueError):
        compare_pairs(np.zeros((4,2)), [[0,1]], [0,0,0,0])
    with pytest.raises(ValueError):
        compare_pairs(np.zeros((4,2)), [[0,0]], [0,1,0,1])
    with pytest.raises(ValueError):
        compare_group(np.zeros((4,1)), [0,1,0,1])
