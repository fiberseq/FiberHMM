"""Exact weighted-independent-set DAG checks, independent of assay semantics."""
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest


def _load(name, filename):
    path = Path(__file__).resolve().parents[1] / "fiberhmm/inference" / filename
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


cg = _load("tested_state_conflict_graph", "state_conflict_graph.py")


def _enumerate(adjacency, weights):
    n = len(adjacency)
    assignments = []
    for mask in range(1 << n):
        if any(mask & (1 << i) and adjacency[i] & mask for i in range(n)):
            continue
        assignments.append([(mask >> i) & 1 for i in range(n)])
    assignments = np.asarray(assignments, dtype=float).reshape(-1, n)
    # Avoid 0 * -inf in the literal reference.
    log_weights = np.zeros((len(weights), len(assignments)))
    for j in range(n):
        log_weights += np.where(assignments[None, :, j] == 1, weights[:, j, None], 0.)
    maximum = np.max(log_weights, axis=1)
    logz = maximum + np.log(np.exp(log_weights - maximum[:, None]).sum(axis=1))
    return logz, np.exp(log_weights - logz[:, None]) @ assignments, maximum


@pytest.mark.parametrize("n", [1, 2, 4, 7, 10])
def test_random_graphs_match_literal_enumeration(n):
    rng = np.random.default_rng(51 + n)
    for probability in (0., 0.2, 0.6, 1.):
        edges = [(i, j) for i in range(n) for j in range(i + 1, n) if rng.random() < probability]
        graph = cg.ConflictGraphCatalog.from_edges(n, edges)
        weights = rng.normal(size=(4, n)) * 2
        weights[0, 0] = -np.inf
        expected_z, expected_m, expected_max = _enumerate(graph.adjacency, weights)
        actual = graph.infer(weights)
        np.testing.assert_allclose(actual.log_z, expected_z, atol=3e-12)
        np.testing.assert_allclose(actual.marginals, expected_m, atol=3e-12)
        result = graph.map_assignments(weights)
        np.testing.assert_allclose(result.log_weights, expected_max, atol=3e-12)
        for row, selected in enumerate(result.assignments):
            mask = sum(int(value) << i for i, value in enumerate(selected))
            assert all(not selected[i] or not graph.adjacency[i] & mask for i in range(n))
            assert float(weights[row, selected].sum()) == pytest.approx(expected_max[row])


def test_clique_1000_has_linear_dag_not_binary_truth_table():
    n = 1000
    full = (1 << n) - 1
    graph = cg.ConflictGraphCatalog([full ^ (1 << i) for i in range(n)], max_dag_nodes=n + 1, max_batch_cells=20_000)
    result = graph.infer(np.zeros((3, n)))
    assert graph.dag_nodes == n + 1
    np.testing.assert_allclose(result.log_z, np.log(n + 1), atol=5e-12)
    np.testing.assert_allclose(result.marginals, 1 / (n + 1), atol=5e-12)
    assert graph.map_indices(np.zeros(n)) == ()


def test_interval_graph_agrees_with_interval_dp():
    interval = _load("conflict_test_interval_reference", "hierarchical_state_model.py")
    rng = np.random.default_rng(81)
    starts = np.sort(rng.integers(0, 200, size=45))
    ends = starts + rng.integers(1, 25, size=45)
    states = [interval.StateInterval(str(i), int(a), int(b)) for i, (a, b) in enumerate(zip(starts, ends))]
    edges = [(i, j) for i in range(len(states)) for j in range(i + 1, len(states)) if min(ends[i], ends[j]) > max(starts[i], starts[j])]
    graph = cg.ConflictGraphCatalog.from_edges(len(states), edges)
    weights = rng.normal(size=(7, len(states)))
    left, right = graph.infer(weights), interval.FixedStateCatalog(states).infer(weights)
    np.testing.assert_allclose(left.log_z, right.log_z, atol=5e-12)
    np.testing.assert_allclose(left.marginals, right.marginals, atol=5e-12)


def test_reverse_ad_marginals_equal_partition_derivatives():
    graph = cg.ConflictGraphCatalog.from_edges(7, [(0, 2), (1, 2), (1, 4), (3, 4), (3, 5), (5, 6)])
    weights = np.random.default_rng(25).normal(size=(3, 7))
    result = graph.infer(weights)
    epsilon = 1e-5
    for j in range(7):
        plus, minus = weights.copy(), weights.copy()
        plus[:, j] += epsilon
        minus[:, j] -= epsilon
        derivative = (graph.infer(plus).log_z - graph.infer(minus).log_z) / (2 * epsilon)
        np.testing.assert_allclose(derivative, result.marginals[:, j], atol=1e-9)


def test_empty_disconnected_and_negative_infinite_nodes():
    empty = cg.ConflictGraphCatalog([])
    np.testing.assert_array_equal(empty.infer(np.empty((2, 0))).log_z, [0., 0.])
    assert empty.map_indices([]) == ()
    graph = cg.ConflictGraphCatalog([0] * 1000, max_dag_nodes=1001)
    weights = np.zeros((2, 1000))
    weights[0] = -np.inf
    result = graph.infer(weights)
    np.testing.assert_allclose(result.log_z, [0., 1000 * np.log(2)], atol=2e-11)
    np.testing.assert_allclose(result.marginals[0], 0.)
    np.testing.assert_allclose(result.marginals[1], 0.5)


def test_dag_and_batch_budget_failure_are_explicit():
    with pytest.raises(cg.ConflictGraphBudgetError, match="DAG nodes"):
        cg.ConflictGraphCatalog([0] * 10, max_dag_nodes=5)
    graph = cg.ConflictGraphCatalog([0] * 10, max_batch_cells=100)
    with pytest.raises(cg.ConflictGraphBudgetError, match="scalar cells"):
        graph.infer(np.zeros((3, 10)))


def test_bad_graph_and_weights_fail():
    for adjacency in ([1], [-1], [2], [2, 0]):
        with pytest.raises(ValueError):
            cg.ConflictGraphCatalog(adjacency)
    graph = cg.ConflictGraphCatalog([0])
    for weights in ([np.inf], [np.nan], [], np.zeros((0, 1)), np.zeros((2, 2))):
        with pytest.raises(ValueError):
            graph.infer(weights)
    with pytest.raises(ValueError):
        cg.ConflictGraphCatalog.from_edges(2, [(0, 0)])
