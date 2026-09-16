"""Exact weighted independent-set inference by memoized deletion/contraction.

Nodes may represent concrete states or feasible mixed-block superstates. The
caller supplies node weights and the complete conflict graph; this module
neither constructs those scientific objects nor approximates them. The empty
set is always present with weight one.

Compilation chooses the highest-index remaining vertex, so callers should
order nodes by genomic end for an interval-like graph. Memoized remaining-set
subproblems form a directed acyclic graph. Reverse differentiation through that
DAG gives every inclusion marginal in one pass. Explicit compile/batch budgets
raise an exception rather than truncating the state universe.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


class ConflictGraphBudgetError(ValueError):
    """The exact DAG or its batch tables exceed the declared budget."""


@dataclass
class ConflictGraphPosterior:
    log_z: np.ndarray
    marginals: np.ndarray
    diagnostics: dict


@dataclass
class ConflictGraphMAP:
    assignments: np.ndarray
    log_weights: np.ndarray
    diagnostics: dict


class ConflictGraphCatalog:
    """Compile an exact arbitrary conflict graph in caller-specified node order.

    ``adjacency[i]`` is a Python-integer bitmask containing other nodes that
    conflict with node ``i``; masks must be symmetric and omit self edges.
    ``max_batch_cells`` counts two DAG tables and two units-by-node tables,
    conservatively covering reverse differentiation and returned marginals.
    A compiled DAG can be reused for any signed finite/negative-infinite batch
    of node log weights. No minimum support or weight threshold is applied.
    """

    def __init__(
        self, adjacency: Sequence[int], *, max_dag_nodes: int = 1_000_000,
        max_batch_cells: int = 50_000_000,
    ):
        adjacency = tuple(adjacency)
        if any(not isinstance(value, (int, np.integer)) for value in adjacency):
            raise ValueError("Adjacency masks must be integers")
        self.adjacency = tuple(int(value) for value in adjacency)
        self.n = len(self.adjacency)
        if not isinstance(max_dag_nodes, int) or max_dag_nodes < 1:
            raise ValueError("max_dag_nodes must be a positive integer")
        if not isinstance(max_batch_cells, int) or max_batch_cells < 1:
            raise ValueError("max_batch_cells must be a positive integer")
        self.max_dag_nodes = max_dag_nodes
        self.max_batch_cells = max_batch_cells
        allowed = (1 << self.n) - 1
        for i, mask in enumerate(self.adjacency):
            if mask < 0 or mask & ~allowed or mask & (1 << i):
                raise ValueError("Adjacency must contain only other in-range nodes")
            remaining = mask
            while remaining:
                bit = remaining & -remaining
                j = bit.bit_length() - 1
                if not self.adjacency[j] & (1 << i):
                    raise ValueError("Conflict adjacency must be symmetric")
                remaining ^= bit
        # Iterative postorder avoids Python recursion limits even for a clique
        # with thousands of vertices. Children always precede parents.
        index = {0: 0}
        discovered = {0}
        variables, skipped, included = [-1], [0], [0]
        stack = [(allowed, False)] if allowed else []
        maximum_stack = len(stack)
        while stack:
            state, expanded = stack.pop()
            if state in index:
                continue
            variable = state.bit_length() - 1
            without = state ^ (1 << variable)
            compatible = without & ~self.adjacency[variable]
            if not expanded:
                if state not in discovered:
                    discovered.add(state)
                    if len(discovered) > max_dag_nodes:
                        raise ConflictGraphBudgetError(f"Exact deletion/contraction requires more than {max_dag_nodes} distinct remaining-set DAG nodes; no states were truncated")
                stack.append((state, True))
                if compatible not in index:
                    stack.append((compatible, False))
                if without not in index and without != compatible:
                    stack.append((without, False))
                maximum_stack = max(maximum_stack, len(stack))
                continue
            if without not in index or compatible not in index:
                raise RuntimeError("Internal deletion/contraction postorder invariant failed")
            index[state] = len(variables)
            variables.append(variable)
            skipped.append(index[without])
            included.append(index[compatible])
        self.variables = np.asarray(variables, dtype=np.int64)
        self.skipped = np.asarray(skipped, dtype=np.int64)
        self.included = np.asarray(included, dtype=np.int64)
        self.root = index[allowed]
        self.dag_nodes = len(variables)
        self.maximum_compile_stack = maximum_stack
        self.edge_count = sum(mask.bit_count() for mask in self.adjacency) // 2
        self._check_batch_budget(1)

    @classmethod
    def from_edges(cls, n_nodes: int, edges: Iterable[tuple[int, int]], **kwargs):
        if not isinstance(n_nodes, int) or n_nodes < 0:
            raise ValueError("n_nodes must be a nonnegative integer")
        adjacency = [0] * n_nodes
        for left, right in edges:
            if not isinstance(left, (int, np.integer)) or not isinstance(right, (int, np.integer)) or left < 0 or right < 0 or left >= n_nodes or right >= n_nodes or left == right:
                raise ValueError("Edges must connect two distinct in-range nodes")
            left, right = int(left), int(right)
            adjacency[left] |= 1 << right
            adjacency[right] |= 1 << left
        return cls(adjacency, **kwargs)

    def _check_batch_budget(self, batch: int) -> None:
        cells = batch * (2 * self.dag_nodes + 2 * self.n)
        if cells > self.max_batch_cells:
            raise ConflictGraphBudgetError(f"Exact batch requires {cells} scalar cells for batch={batch}, DAG={self.dag_nodes}, nodes={self.n}; budget={self.max_batch_cells}; no states were truncated")

    def _weights(self, log_weights):
        weights = np.asarray(log_weights, dtype=float)
        was_vector = weights.ndim == 1
        if was_vector:
            weights = weights[None, :]
        if weights.ndim != 2 or weights.shape[1] != self.n or weights.shape[0] < 1:
            raise ValueError("Expected a nonempty batch of units-by-nodes weights, or one node vector")
        if np.any(np.isnan(weights)) or np.any(np.isposinf(weights)):
            raise ValueError("Log weights must be finite or negative infinity")
        self._check_batch_budget(len(weights))
        return weights, was_vector

    def _diagnostics(self, batch: int) -> dict:
        return {"algorithm": "exact_memoized_deletion_contraction", "nodes": self.n, "conflict_edges": self.edge_count, "dag_nodes_including_empty": self.dag_nodes, "elimination_order": "highest_remaining_node_index; caller_genomic_end_order", "maximum_compile_stack": self.maximum_compile_stack, "batch_scalar_cell_bound": batch * (2 * self.dag_nodes + 2 * self.n), "max_dag_nodes": self.max_dag_nodes, "max_batch_cells": self.max_batch_cells, "truncated": False}

    def infer(self, log_weights) -> ConflictGraphPosterior:
        weights, _ = self._weights(log_weights)
        batch = len(weights)
        log_values = np.zeros((self.dag_nodes, batch))
        for node in range(1, self.dag_nodes):
            variable = self.variables[node]
            log_values[node] = np.logaddexp(log_values[self.skipped[node]], weights[:, variable] + log_values[self.included[node]])
        if not np.all(np.isfinite(log_values[self.root])):
            raise FloatingPointError("Finite node weights overflowed the float64 partition")
        adjoint = np.zeros_like(log_values)
        adjoint[self.root] = 1.0
        marginals = np.zeros((batch, self.n))
        for node in range(self.dag_nodes - 1, 0, -1):
            variable, skipped, included = self.variables[node], self.skipped[node], self.included[node]
            probability_skip = np.exp(log_values[skipped] - log_values[node])
            probability_include = np.exp(weights[:, variable] + log_values[included] - log_values[node])
            skip_mass = adjoint[node] * probability_skip
            include_mass = adjoint[node] * probability_include
            marginals[:, variable] += include_mass
            adjoint[skipped] += skip_mass
            adjoint[included] += include_mass
        # Do not let a returned root-row view pin the complete DAG workspace.
        np.clip(marginals, 0.0, 1.0, out=marginals)
        return ConflictGraphPosterior(log_values[self.root].copy(), marginals, self._diagnostics(batch))

    def map_assignments(self, log_weights) -> ConflictGraphMAP:
        weights, _ = self._weights(log_weights)
        batch = len(weights)
        best = np.zeros((self.dag_nodes, batch))
        choose_include = np.zeros((self.dag_nodes, batch), dtype=bool)
        for node in range(1, self.dag_nodes):
            skipped = best[self.skipped[node]]
            included = weights[:, self.variables[node]] + best[self.included[node]]
            choose_include[node] = included > skipped  # Exact ties omit node.
            best[node] = np.maximum(skipped, included)
        if not np.all(np.isfinite(best[self.root])):
            raise FloatingPointError("Finite node weights overflowed the float64 MAP score")
        assignments = np.zeros((batch, self.n), dtype=bool)
        for row in range(batch):
            node = self.root
            while node:
                if choose_include[node, row]:
                    assignments[row, self.variables[node]] = True
                    node = self.included[node]
                else:
                    node = self.skipped[node]
        return ConflictGraphMAP(assignments, best[self.root].copy(), self._diagnostics(batch))

    def map_indices(self, log_weights):
        was_vector = np.asarray(log_weights).ndim == 1
        result = self.map_assignments(log_weights)
        indices = tuple(tuple(np.flatnonzero(row).tolist()) for row in result.assignments)
        return indices[0] if was_vector else indices
