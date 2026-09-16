"""Generic simplex likelihood fitting; no forced family-count interpretation."""
import math

import numpy as np
import pytest
from scipy.optimize import minimize

from fiberhmm.inference.fixed_likelihood_mixture import fit_fixed_likelihood_mixture, posterior_mixture_responsibilities


def test_known_categorical_mixture_weights_and_full_gradient_certificate():
    labels = np.r_[np.zeros(60, dtype=int), np.ones(30, dtype=int), np.full(10, 2, dtype=int)]
    logs = np.full((len(labels), 3), -np.inf)
    logs[np.arange(len(labels)), labels] = 0
    fit = fit_fixed_likelihood_mixture(logs, tolerance=1e-10)
    assert fit['converged']
    assert fit['weights'] == pytest.approx([.6, .3, .1], abs=1e-10)
    likelihood = np.exp(logs)
    gradient = likelihood.T @ (1 / (likelihood @ fit['weights']))
    assert fit['frank_wolfe_gap'] == pytest.approx(max(0, gradient.max() - len(labels)), abs=1e-10)
    assert fit['hard_pruned_columns'] == 0


@pytest.mark.parametrize('seed', range(3))
def test_small_objective_matches_independent_constrained_optimizer(seed):
    rng = np.random.default_rng(seed + 453)
    logs = rng.normal(0, 2, (130, 5))
    likelihood = np.exp(logs - logs.max(axis=1)[:, None])
    def objective(w):
        density = likelihood @ w
        return -np.log(density).sum(), -(likelihood.T @ (1 / density))
    reference = minimize(objective, np.ones(5)/5, jac=True, method='SLSQP', bounds=[(0, 1)]*5,
                         constraints={'type':'eq','fun':lambda w:w.sum()-1,'jac':lambda w:np.ones(5)},
                         options={'maxiter':1000, 'ftol':1e-11})
    fit = fit_fixed_likelihood_mixture(logs, tolerance=1e-9)
    assert reference.success and fit['converged']
    assert fit['objective'] == pytest.approx(-reference.fun + logs.max(axis=1).sum(), abs=2e-7)
    assert fit['weights'] == pytest.approx(reference.x, abs=2e-6)
    for start in fit['starts']:
        assert np.all(np.diff(start['objective_history']) >= -1e-8)


def test_identical_columns_expose_nonunique_weights_not_unique_families():
    rng = np.random.default_rng(503)
    values = rng.normal(size=90)
    logs = np.column_stack([values, values, values])
    fit = fit_fixed_likelihood_mixture(logs, n_starts=3)
    assert fit['converged'] and fit['n_informative_rows'] == 0
    assert fit['duplicate_likelihood_column_groups'] == [[0, 1, 2]]
    assert fit['max_start_prediction_difference'] == 0
    assert fit['max_start_weight_l1_difference'] > .05
    assert fit['objective'] == pytest.approx(values.sum())
    post = posterior_mixture_responsibilities(logs, [.8, .1, .1])
    assert np.all(post['prior_only_rows'])
    assert post['learned_posterior'] == pytest.approx(np.tile([.8, .1, .1], (90, 1)))
    assert post['uniform_posterior'] == pytest.approx(np.full((90, 3), 1/3))


def test_duplicate_likelihood_columns_preserve_total_probability_mass():
    logs = np.r_[np.tile([0., -12.], (80, 1)), np.tile([-12., 0.], (20, 1))]
    original = fit_fixed_likelihood_mixture(logs, tolerance=1e-9)
    duplicate = fit_fixed_likelihood_mixture(logs[:, [0, 0, 1]], tolerance=1e-9)
    assert duplicate['converged']
    assert duplicate['weights'][:2].sum() == pytest.approx(original['weights'][0], abs=1e-7)
    assert duplicate['log_predictive'] == pytest.approx(original['log_predictive'], abs=1e-7)
    assert duplicate['duplicate_likelihood_column_groups'] == [[0, 1]]


def test_empty_and_uninformative_rows_do_not_manufacture_support():
    empty = fit_fixed_likelihood_mixture(np.empty((0, 4)))
    assert empty['objective'] == 0 and empty['n_informative_rows'] == 0
    logs = np.r_[np.tile([4., -5.], (50, 1)), np.tile([-5., 4.], (25, 1))]
    original = fit_fixed_likelihood_mixture(logs)
    augmented = fit_fixed_likelihood_mixture(np.r_[logs, np.full((500, 2), 3.)])
    assert augmented['weights'] == pytest.approx(original['weights'], abs=1e-9)
    assert augmented['objective'] == pytest.approx(original['objective'] + 1500)
    assert augmented['n_informative_rows'] == len(logs)
    assert augmented['frank_wolfe_gap'] == pytest.approx(original['frank_wolfe_gap'], abs=1e-9)


def test_row_base_shifts_and_row_column_permutations_preserve_fit():
    rng = np.random.default_rng(803)
    logs = rng.normal(0, 3, (100, 4))
    shifts = rng.normal(0, 100, 100)
    rows, columns = rng.permutation(100), rng.permutation(4)
    original = fit_fixed_likelihood_mixture(logs, tolerance=1e-9)
    changed = fit_fixed_likelihood_mixture((logs + shifts[:, None])[rows][:, columns], tolerance=1e-9)
    assert original['converged'] and changed['converged']
    assert changed['weights'] == pytest.approx(original['weights'][columns], abs=2e-7)
    assert changed['log_predictive'] == pytest.approx((original['log_predictive']+shifts)[rows], abs=2e-6)


def test_extreme_scores_and_zero_start_columns_remain_reachable():
    logs = np.r_[np.tile([1000., -1000.], (20, 1)), np.tile([-1000., 1000.], (80, 1))]
    fit = fit_fixed_likelihood_mixture(logs, start_weights=[1., 0.], tolerance=1e-9, n_starts=1)
    assert fit['converged']
    assert fit['weights'] == pytest.approx([.2, .8], abs=1e-8)
    post = posterior_mixture_responsibilities(logs, fit['weights'])
    assert np.isfinite(post['log_predictive']).all()
    assert post['log_predictive'] == pytest.approx(fit['log_predictive'])
    assert post['learned_posterior'].sum(axis=1) == pytest.approx(np.ones(100))


def test_budget_stop_and_independent_unit_contract_are_explicit():
    rng = np.random.default_rng(902)
    logs = rng.normal(0, 2, (120, 15))
    fit = fit_fixed_likelihood_mixture(logs, max_iter=1, tolerance=1e-14, n_starts=1)
    assert not fit['converged']
    assert fit['frank_wolfe_gap'] > 1e-12
    with pytest.raises(ValueError, match='unique'):
        fit_fixed_likelihood_mixture(logs, group_ids=['duplicate']*120)
    with pytest.raises(ValueError, match='budget'):
        fit_fixed_likelihood_mixture(logs, max_matrix_cells=10)
    with pytest.raises(ValueError, match='possible'):
        fit_fixed_likelihood_mixture([[-np.inf, -np.inf]])


def interval_likelihood(hits, intervals, pa=.9, pp=.1):
    positions = np.arange(hits.shape[1])
    output = []
    for start, end in intervals:
        probability = np.where((positions >= start) & (positions < end), pp, pa)
        output.append(np.where(hits, np.log(probability), np.log1p(-probability)).sum(axis=1))
    return np.column_stack(output)


@pytest.mark.parametrize('kind', ['single', 'two', 'continuum'])
def test_single_interval_population_examples_use_each_molecule_once(kind):
    rng = np.random.default_rng(1208)
    intervals = [(0, 0)] + [(8, end) for end in range(24, 37)]
    n = 1000
    if kind == 'single':
        ends = np.full(n, 26)
    elif kind == 'two':
        ends = np.where(np.arange(n)%2, 34, 26)
    else:
        ends = np.resize(np.arange(26, 35), n)
    protected = (np.arange(40)[None, :] >= 8) & (np.arange(40)[None, :] < ends[:, None])
    hits = rng.random((n, 40)) < np.where(protected, .1, .9)
    logs = interval_likelihood(hits, intervals)
    fit = fit_fixed_likelihood_mixture(logs, group_ids=np.arange(n), tolerance=1e-7)
    assert fit['converged']
    assert fit['n_rows'] == n
    weights = fit['weights']
    by_end = {end: weights[index] for index, (start, end) in enumerate(intervals) if start == 8}
    if kind == 'single':
        assert sum(weight for end, weight in by_end.items() if abs(end-26) <= 1) > .97
    elif kind == 'two':
        assert .45 < sum(weight for end, weight in by_end.items() if abs(end-26) <= 1) < .55
        assert .45 < sum(weight for end, weight in by_end.items() if abs(end-34) <= 1) < .55
    else:
        assert sum(weight for end, weight in by_end.items() if 26 <= end <= 34) > .95
        assert max(by_end.values()) < .25
    # A density fit is deliberately NOT translated into a forced family count.
    assert 'family_labels' not in fit
