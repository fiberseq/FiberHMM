"""Exact bounded binary-factor inference for the experimental native state model.

Factors may encode arbitrary subset likelihoods, including joint-geometry
normalizers. This module neither invents nor approximates those factors. It
computes their exact partition, variable marginals, and MAP by deterministic
min-fill bucket elimination. Explicit scalar-cell budgets fail closed before
allocating a large elimination frontier; nothing is silently truncated.

Factor columns use little-endian local assignments: bit j selects variable
``scope[j]``. A factor has shape ``(batch, 2**len(scope))`` or an unbatched
``(2**len(scope),)`` vector. Batch-one factors broadcast. Missing observations
must be represented by zero log factors, not fabricated outcomes.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np


class FactorGraphBudgetError(ValueError):
    """Exact inference exceeds a declared frontier or retained-cell budget."""


class InfeasibleFactorGraphError(ValueError):
    """At least one batch row has no finite-weight assignment."""


@dataclass(frozen=True)
class BinaryFactor:
    scope: tuple[int, ...]
    log_values: np.ndarray


@dataclass
class FactorGraphPosterior:
    log_z: np.ndarray
    marginals: np.ndarray
    diagnostics: dict
    factor_marginals: tuple[np.ndarray, ...] | None = None


@dataclass
class FactorGraphMAP:
    assignments: np.ndarray
    log_weights: np.ndarray
    diagnostics: dict


def _projection(source: tuple[int, ...], target: tuple[int, ...]) -> np.ndarray:
    """Project source assignment integers into target's local bit order."""
    masks = np.arange(1 << len(source), dtype=np.int64)
    result = np.zeros_like(masks)
    for bit, variable in enumerate(target):
        result |= ((masks >> source.index(variable)) & 1) << bit
    return result


def _min_fill_order(n_variables: int, scopes: Sequence[tuple[int, ...]]) -> tuple[int, ...]:
    neighbors = [set() for _ in range(n_variables)]
    for scope in scopes:
        for variable in scope:
            neighbors[variable].update(set(scope) - {variable})
    remaining = set(range(n_variables))
    order = []
    while remaining:
        def priority(variable):
            adjacent = sorted(neighbors[variable] & remaining)
            missing = sum(right not in neighbors[left] for i, left in enumerate(adjacent) for right in adjacent[i + 1:])
            return missing, len(adjacent), variable
        variable = min(remaining, key=priority)
        adjacent = sorted(neighbors[variable] & remaining)
        for left in adjacent:
            neighbors[left].update(set(adjacent) - {left})
            neighbors[left].discard(variable)
        remaining.remove(variable)
        order.append(variable)
    return tuple(order)


class BinaryFactorGraph:
    """Compile one factor topology; replace factor values without recompiling.

    ``max_frontier_cells`` bounds one batch-expanded elimination table.
    ``max_total_cells`` conservatively counts input factors, optional original-
    factor posterior tables, and all stored joint/posterior/message tables.
    Each scalar is float64; projection indexing
    metadata is unbatched. A marginal call uses exact backward clique messages,
    not a separate inference rerun for every variable.
    """

    def __init__(
        self, n_variables: int, factors: Sequence[BinaryFactor], *,
        max_frontier_cells: int = 2_000_000,
        max_total_cells: int = 50_000_000,
    ):
        if not isinstance(n_variables, int) or n_variables < 0:
            raise ValueError("n_variables must be a nonnegative integer")
        if max_frontier_cells < 1 or max_total_cells < 1:
            raise ValueError("Cell budgets must be positive")
        self.n_variables = n_variables
        self.factors = tuple(factors)
        self.scopes = tuple(tuple(factor.scope) for factor in self.factors)
        self.max_frontier_cells = int(max_frontier_cells)
        self.max_total_cells = int(max_total_cells)
        for scope in self.scopes:
            if len(scope) != len(set(scope)) or any(not isinstance(v, (int, np.integer)) or v < 0 or v >= n_variables for v in scope):
                raise ValueError("Factor scopes must contain distinct in-range variable indices")
        self.order = _min_fill_order(n_variables, self.scopes)
        rank = {variable: index for index, variable in enumerate(self.order)}
        buckets = [[] for _ in self.order]
        all_scopes = list(self.scopes)
        self.constant_indices = []
        self.factor_buckets = []
        for index, scope in enumerate(self.scopes):
            if scope:
                bucket = min(rank[v] for v in scope)
                buckets[bucket].append(index)
                self.factor_buckets.append(bucket)
            else:
                self.constant_indices.append(index)
                self.factor_buckets.append(None)
        self.plan = []
        unbatched_total = 2 * sum(1 << len(scope) for scope in self.scopes)
        max_width = 0
        # Check width BEFORE constructing any exponentially sized projections.
        for step, variable in enumerate(self.order):
            inputs = tuple(buckets[step])
            remainder = tuple(sorted({v for index in inputs for v in all_scopes[index]} - {variable}))
            scope = (variable, *remainder)
            width = len(scope)
            frontier = 1 << width
            if frontier > self.max_frontier_cells:
                raise FactorGraphBudgetError(f"Variable {variable} requires {frontier} frontier cells before batching; budget={self.max_frontier_cells}")
            unbatched_total += 2 * frontier + (1 << len(remainder))
            if unbatched_total > self.max_total_cells:
                raise FactorGraphBudgetError(f"Compiled graph requires at least {unbatched_total} retained cells before batching; budget={self.max_total_cells}")
            max_width = max(max_width, width)
            output_index = len(all_scopes)
            all_scopes.append(remainder)
            parent = min((rank[v] for v in remainder), default=None)
            if parent is not None:
                buckets[parent].append(output_index)
            self.plan.append({"variable": variable, "scope": scope, "remainder": remainder, "inputs": inputs, "output_index": output_index, "parent": parent})
        self.all_scopes = tuple(all_scopes)
        self.unbatched_total_cells = unbatched_total
        self.maximum_frontier_width = max_width
        for item in self.plan:
            item["input_projections"] = tuple(_projection(item["scope"], self.all_scopes[index]) for index in item["inputs"])
            if item["parent"] is not None:
                parent_scope = self.plan[item["parent"]]["scope"]
                projected = _projection(parent_scope, item["remainder"])
                item["parent_group_order"] = np.argsort(projected, kind="stable")
        self.factor_group_orders = tuple(
            None if bucket is None else np.argsort(_projection(self.plan[bucket]["scope"], scope), kind="stable")
            for bucket, scope in zip(self.factor_buckets, self.scopes)
        )
        self._values(None)  # Validate initial shapes/batches before first use.

    def _values(self, factors: Sequence[BinaryFactor] | None):
        factors = self.factors if factors is None else tuple(factors)
        if tuple(tuple(factor.scope) for factor in factors) != self.scopes:
            raise ValueError("Replacement factor topology differs from the compiled graph")
        arrays = []
        batches = set()
        for factor, scope in zip(factors, self.scopes):
            value = np.asarray(factor.log_values, dtype=float)
            if value.ndim == 1:
                value = value[None, :]
            if value.ndim != 2 or value.shape[1] != 1 << len(scope) or value.shape[0] < 1:
                raise ValueError("Factor values must be (batch, 2**len(scope)) or an unbatched vector")
            if np.any(np.isnan(value)) or np.any(np.isposinf(value)):
                raise ValueError("Log factors must be finite or negative infinity")
            batches.add(len(value))
            arrays.append(value)
        batch = max(batches, default=1)
        if any(size not in {1, batch} for size in batches):
            raise ValueError("Factor batch dimensions must match or equal one")
        frontier = batch * (1 << self.maximum_frontier_width) if self.plan else batch
        retained = batch * self.unbatched_total_cells
        if frontier > self.max_frontier_cells:
            raise FactorGraphBudgetError(f"Batch {batch} requires {frontier} frontier cells; budget={self.max_frontier_cells}")
        if retained > self.max_total_cells:
            raise FactorGraphBudgetError(f"Batch {batch} requires {retained} retained scalar cells; budget={self.max_total_cells}")
        return arrays, batch

    def _diagnostics(self, batch: int) -> dict:
        return {"algorithm": "exact_min_fill_bucket_elimination", "elimination_order": list(self.order), "maximum_frontier_variables": self.maximum_frontier_width, "maximum_frontier_cells": batch * (1 << self.maximum_frontier_width) if self.plan else batch, "retained_scalar_cell_bound": batch * self.unbatched_total_cells, "truncated": False}

    def _forward(self, factors, maximum: bool):
        values, batch = self._values(factors)
        values.extend([None] * self.n_variables)
        joint_tables, choices = [], []
        total = np.zeros(batch)
        for index in self.constant_indices:
            total += values[index][:, 0]
        for item in self.plan:
            joint = np.zeros((batch, 1 << len(item["scope"])))
            for index, projection in zip(item["inputs"], item["input_projections"]):
                joint += values[index][:, projection]
            zero, one = joint[:, 0::2], joint[:, 1::2]
            message = np.maximum(zero, one) if maximum else np.logaddexp(zero, one)
            values[item["output_index"]] = message
            if item["parent"] is None:
                total += message[:, 0]
            joint_tables.append(joint)
            if maximum:
                choices.append(one > zero)  # Deterministic ties choose zero.
        bad = np.flatnonzero(~np.isfinite(total))
        if len(bad):
            raise InfeasibleFactorGraphError(f"No finite-weight complete assignment in batch rows {bad.tolist()}")
        return total, values, joint_tables, choices, batch

    def infer(self, factors: Sequence[BinaryFactor] | None = None, *, with_marginals: bool = True, with_factor_marginals: bool = False) -> FactorGraphPosterior:
        total, values, joints, _choices, batch = self._forward(factors, maximum=False)
        marginals = np.empty((batch, self.n_variables)) if with_marginals else np.empty((batch, 0))
        factor_marginals = None
        if with_marginals or with_factor_marginals:
            beliefs = [None] * self.n_variables
            for step in range(self.n_variables - 1, -1, -1):
                item = self.plan[step]
                if item["parent"] is None:
                    remainder_probability = np.ones((batch, 1))
                else:
                    parent_belief = beliefs[item["parent"]]
                    ordered = parent_belief[:, item["parent_group_order"]]
                    remainder_probability = ordered.reshape(batch, 1 << len(item["remainder"]), -1).sum(axis=-1)
                denominator = np.repeat(values[item["output_index"]], 2, axis=1)
                # Impossible separator assignments have zero conditional mass.
                conditional_log = np.full_like(joints[step], -math.inf)
                np.subtract(joints[step], denominator, out=conditional_log, where=np.isfinite(denominator))
                belief = np.exp(conditional_log) * np.repeat(remainder_probability, 2, axis=1)
                beliefs[step] = belief
                if with_marginals:
                    marginals[:, item["variable"]] = belief[:, 1::2].sum(axis=1)
            if with_factor_marginals:
                factor_marginals = tuple(
                    np.ones((batch, 1)) if bucket is None else beliefs[bucket][:, order].reshape(batch, 1 << len(scope), -1).sum(axis=-1)
                    for bucket, scope, order in zip(self.factor_buckets, self.scopes, self.factor_group_orders)
                )
        return FactorGraphPosterior(total, marginals, self._diagnostics(batch), factor_marginals)

    def map_assignments(self, factors: Sequence[BinaryFactor] | None = None) -> FactorGraphMAP:
        total, _values, _joints, choices, batch = self._forward(factors, maximum=True)
        assignments = np.zeros((batch, self.n_variables), dtype=bool)
        rows = np.arange(batch)
        for step in range(self.n_variables - 1, -1, -1):
            item = self.plan[step]
            indices = np.zeros(batch, dtype=np.int64)
            for bit, variable in enumerate(item["remainder"]):
                indices |= assignments[:, variable].astype(np.int64) << bit
            assignments[:, item["variable"]] = choices[step][rows, indices]
        return FactorGraphMAP(assignments, total, self._diagnostics(batch))
