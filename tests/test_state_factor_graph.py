"""Literal enumeration checks for the exact bounded factor-graph kernel."""
import importlib.util
import itertools
from pathlib import Path
import sys

import numpy as np
import pytest


_path = Path(__file__).resolve().parents[1] / "fiberhmm/inference/state_factor_graph.py"
_spec = importlib.util.spec_from_file_location("tested_state_factor_graph", _path)
fg = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = fg
_spec.loader.exec_module(fg)


def _enumerate(n, factors):
    batch = max((np.asarray(f.log_values).shape[0] if np.asarray(f.log_values).ndim == 2 else 1 for f in factors), default=1)
    assignments = np.asarray(list(itertools.product((0, 1), repeat=n)), dtype=int).reshape(1 << n, n)
    logw = np.zeros((batch, 1 << n))
    for factor in factors:
        values = np.atleast_2d(factor.log_values)
        index = sum((assignments[:, var] << bit for bit, var in enumerate(factor.scope)), start=np.zeros(1 << n, dtype=int))
        logw += values[:, index]
    maximum = np.max(logw, axis=1)
    logz = maximum + np.log(np.exp(logw - maximum[:, None]).sum(axis=1))
    posterior = np.exp(logw - logz[:, None])
    return logz, posterior @ assignments, maximum


@pytest.mark.parametrize("n", [1, 2, 3, 5, 7])
def test_random_factors_match_literal_enumeration(n):
    rng = np.random.default_rng(71 + n)
    factors = [fg.BinaryFactor((i,), rng.normal(size=(3, 2))) for i in range(n)]
    for _ in range(n + 2):
        scope = tuple(rng.choice(n, size=min(3, n), replace=False).tolist())
        factors.append(fg.BinaryFactor(scope, rng.normal(size=(3, 1 << len(scope)))))
    factors.append(fg.BinaryFactor((), np.array([0.7])))
    graph = fg.BinaryFactorGraph(n, factors)
    expected_z, expected_m, expected_map = _enumerate(n, factors)
    posterior = graph.infer()
    np.testing.assert_allclose(posterior.log_z, expected_z, atol=2e-12)
    np.testing.assert_allclose(posterior.marginals, expected_m, atol=2e-12)
    result = graph.map_assignments()
    np.testing.assert_allclose(result.log_weights, expected_map, atol=2e-12)
    for row in range(3):
        actual = 0.0
        for factor in factors:
            values = np.atleast_2d(factor.log_values)
            index = sum(int(result.assignments[row, variable]) << bit for bit, variable in enumerate(factor.scope))
            actual += values[min(row, len(values) - 1), index]
        assert actual == pytest.approx(expected_map[row])


def test_hard_conflict_and_missing_evidence():
    factors = [fg.BinaryFactor((0,), np.array([[0., 2.], [0., 0.]])), fg.BinaryFactor((1,), np.array([0., 1.])), fg.BinaryFactor((0, 1), np.array([0., 0., 0., -np.inf]))]
    graph = fg.BinaryFactorGraph(3, factors)
    expected_z, expected_m, _ = _enumerate(3, factors)
    result = graph.infer()
    np.testing.assert_allclose(result.log_z, expected_z)
    np.testing.assert_allclose(result.marginals, expected_m)
    np.testing.assert_allclose(result.marginals[:, 2], 0.5)
    assert not np.any(np.all(graph.map_assignments().assignments[:, :2], axis=1))


def test_disconnected_scaling_does_not_enumerate_all_variables():
    factors = [fg.BinaryFactor((i,), np.array([0., 0.3])) for i in range(150)]
    graph = fg.BinaryFactorGraph(150, factors, max_frontier_cells=2, max_total_cells=2000)
    result = graph.infer()
    np.testing.assert_allclose(result.log_z, [150 * np.logaddexp(0, 0.3)])
    np.testing.assert_allclose(result.marginals, 1 / (1 + np.exp(-0.3)))
    assert result.diagnostics["maximum_frontier_variables"] == 1


def test_value_replacement_and_logz_only():
    graph = fg.BinaryFactorGraph(2, [fg.BinaryFactor((1, 0), np.zeros(4))])
    replacement = [fg.BinaryFactor((1, 0), np.array([[0., 1., 2., 3.], [0., -1., 0.5, 2.]]))]
    expected_z, expected_m, _ = _enumerate(2, replacement)
    np.testing.assert_allclose(graph.infer(replacement).marginals, expected_m)
    result = graph.infer(replacement, with_marginals=False)
    np.testing.assert_allclose(result.log_z, expected_z)
    assert result.marginals.shape == (2, 0)
    with pytest.raises(ValueError, match="topology"):
        graph.infer([fg.BinaryFactor((0, 1), np.zeros(4))])


def test_factor_marginals_are_partition_derivatives():
    factors = [fg.BinaryFactor((0, 1), np.array([0.2, 0.3, -0.5, 1.])), fg.BinaryFactor((1, 2), np.array([0., -0.4, 0.8, 0.1])), fg.BinaryFactor((), np.array([0.7]))]
    graph = fg.BinaryFactorGraph(3, factors)
    result = graph.infer(with_factor_marginals=True)
    epsilon = 1e-5
    for factor_index, factor in enumerate(factors):
        np.testing.assert_allclose(result.factor_marginals[factor_index].sum(axis=1), 1.)
        for column in range(len(factor.log_values)):
            plus, minus = factor.log_values.copy(), factor.log_values.copy()
            plus[column] += epsilon
            minus[column] -= epsilon
            left, right = list(factors), list(factors)
            left[factor_index] = fg.BinaryFactor(factor.scope, plus)
            right[factor_index] = fg.BinaryFactor(factor.scope, minus)
            difference = (graph.infer(left, with_marginals=False).log_z - graph.infer(right, with_marginals=False).log_z) / (2 * epsilon)
            np.testing.assert_allclose(difference, result.factor_marginals[factor_index][:, column], atol=1e-9)


def test_impossible_rows_fail_explicitly():
    graph = fg.BinaryFactorGraph(1, [fg.BinaryFactor((0,), np.array([[-np.inf, -np.inf], [0., 0.]]))])
    with pytest.raises(fg.InfeasibleFactorGraphError, match="rows \\[0\\]"):
        graph.infer()
    with pytest.raises(fg.InfeasibleFactorGraphError):
        graph.map_assignments()


def test_budget_failure_is_explicit_not_truncation():
    with pytest.raises(fg.FactorGraphBudgetError, match="frontier"):
        fg.BinaryFactorGraph(5, [fg.BinaryFactor(tuple(range(5)), np.zeros(32))], max_frontier_cells=16)
    with pytest.raises(fg.FactorGraphBudgetError, match="Batch"):
        fg.BinaryFactorGraph(2, [fg.BinaryFactor((0, 1), np.zeros((5, 4)))], max_frontier_cells=16)
    with pytest.raises(fg.FactorGraphBudgetError, match="retained"):
        fg.BinaryFactorGraph(5, [], max_total_cells=10)


def test_empty_graph_and_zero_ties():
    graph = fg.BinaryFactorGraph(0, [fg.BinaryFactor((), np.array([[2.], [-1.]]))])
    result = graph.infer()
    np.testing.assert_array_equal(result.log_z, [2., -1.])
    assert result.marginals.shape == (2, 0)
    graph = fg.BinaryFactorGraph(4, [])
    np.testing.assert_allclose(graph.infer().log_z, [4 * np.log(2)])
    assert not graph.map_assignments().assignments.any()


@pytest.mark.parametrize("factor", [fg.BinaryFactor((0,), np.array([0., np.nan])), fg.BinaryFactor((0,), np.array([0., np.inf])), fg.BinaryFactor((0, 0), np.zeros(4)), fg.BinaryFactor((1,), np.zeros(2)), fg.BinaryFactor((0,), np.zeros(3))])
def test_invalid_inputs_fail(factor):
    with pytest.raises(ValueError):
        fg.BinaryFactorGraph(1, [factor])
