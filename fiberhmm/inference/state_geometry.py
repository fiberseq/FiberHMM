"""Exact reference for the experimental v5 joint family/geometry model.

This is not the legacy CR scorer. A configuration is a subset of families,
each with ONE concrete geometry. Geometry weights are normalized conditional
on physical compatibility *within that family configuration*. Integrating
each family's geometry independently and then scheduling its MAP interval is
not equivalent. No display veto or discovery count enters this likelihood.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import math
from typing import Mapping, Sequence

import numpy as np
from scipy.special import logsumexp


@dataclass(frozen=True)
class StateGeometry:
    geometry_id: str
    start: int
    end: int
    log_likelihood_ratio: float
    weight: float = 1.0

    def __post_init__(self):
        if not self.geometry_id or self.end <= self.start:
            raise ValueError("Geometry needs an ID and a nonempty half-open interval")
        if not math.isfinite(self.weight) or self.weight <= 0:
            raise ValueError("Geometry weights must be finite and positive")
        if math.isnan(self.log_likelihood_ratio) or self.log_likelihood_ratio == math.inf:
            raise ValueError("Evidence must be finite or negative infinity")


@dataclass(frozen=True)
class GeometryFamily:
    state_id: str
    geometries: tuple[StateGeometry, ...]

    def __post_init__(self):
        if not self.state_id or not self.geometries:
            raise ValueError("Family needs an ID and at least one geometry")
        if len({g.geometry_id for g in self.geometries}) != len(self.geometries):
            raise ValueError("Duplicate geometry ID in a family")


@dataclass(frozen=True)
class JointGeometryResult:
    state_ids: tuple[str, ...]
    configurations: tuple[tuple[str, ...], ...]
    configuration_log_ratios: tuple[float, ...]
    configuration_posteriors: tuple[float, ...]
    configuration_priors: tuple[float, ...]
    log_z_observation: float
    log_z_prior: float
    posterior_marginals: tuple[float, ...]
    prior_marginals: tuple[float, ...]
    map_state_ids: tuple[str, ...]
    evaluated_geometry_choices: int

    @property
    def log_marginal_likelihood_ratio(self):
        return self.log_z_observation - self.log_z_prior

    def group_probability(self, state_ids: Sequence[str], *, prior=False) -> float:
        """Probability of ANY member, without inventing a group interval."""
        members = frozenset(state_ids)
        if not members.issubset(self.state_ids):
            raise ValueError("Unknown group member")
        values = self.configuration_priors if prior else self.configuration_posteriors
        return math.fsum(p for c, p in zip(self.configurations, values) if members.intersection(c))

    def decision_configuration(self, veto_state_ids: Sequence[str] = ()) -> tuple[str, ...]:
        """Constrained display decision, NOT a new MAP posterior or likelihood."""
        veto = frozenset(veto_state_ids)
        if not veto.issubset(self.state_ids):
            raise ValueError("Unknown veto member")
        possible = [(p, c) for c, p in zip(self.configurations, self.configuration_posteriors)
                    if not veto.intersection(c)]
        best = max(p for p, _ in possible)
        return min(c for p, c in possible if p == best)


def infer_joint_geometry(
    families: Sequence[GeometryFamily],
    activities: Mapping[str, float] | None = None,
    *,
    max_geometry_choices: int = 250_000,
) -> JointGeometryResult:
    """Enumerate the joint reference; fail explicitly rather than truncate mass.

    Geometry LRs must be additive for each valid concrete tiling on one shared
    observation domain/base measure. Families may overlap as alternatives;
    selected geometries may not overlap. Activities define a normalized prior
    over feasible FAMILY configurations, not over individual geometries.
    """
    ordered = sorted(families, key=lambda f: f.state_id)
    ids = tuple(f.state_id for f in ordered)
    if len(set(ids)) != len(ids):
        raise ValueError("Family IDs must be unique")
    if max_geometry_choices < 1:
        raise ValueError("Reference budget must be positive")
    acts = dict(activities or {})
    if not set(acts).issubset(ids) or any(not math.isfinite(x) for x in acts.values()):
        raise ValueError("Activities must have known IDs and finite values")
    budget = math.prod(1 + len(f.geometries) for f in ordered)
    if budget > max_geometry_choices:
        raise ValueError(f"Joint reference budget exceeded: {budget} > {max_geometry_choices}; no truncation")
    choices = [[None, *sorted(f.geometries, key=lambda g: g.geometry_id)] for f in ordered]
    numerators, denominators = {}, {}
    for combination in product(*choices):
        intervals = sorted((g.start, g.end) for g in combination if g is not None)
        if any(a[1] > b[0] for a, b in zip(intervals, intervals[1:])):
            continue
        config = tuple(f.state_id for f, g in zip(ordered, combination) if g is not None)
        log_weight = math.fsum(math.log(g.weight) for g in combination if g is not None)
        evidence = math.fsum(g.log_likelihood_ratio for g in combination if g is not None)
        numerators.setdefault(config, []).append(log_weight + evidence)
        denominators.setdefault(config, []).append(log_weight)
    configs = tuple(sorted(numerators))
    ratios = np.array([float(logsumexp(numerators[c]) - logsumexp(denominators[c])) for c in configs])
    prior_logs = np.array([math.fsum(acts.get(k, 0.0) for k in c) for c in configs])
    zo, zp = float(logsumexp(prior_logs + ratios)), float(logsumexp(prior_logs))
    post, prior = np.exp(prior_logs + ratios - zo), np.exp(prior_logs - zp)
    marginals = tuple(float(sum(p for c, p in zip(configs, post) if k in c)) for k in ids)
    prior_marginals = tuple(float(sum(p for c, p in zip(configs, prior) if k in c)) for k in ids)
    maximum = float(np.max(prior_logs + ratios))
    map_ids = min(c for c, score in zip(configs, prior_logs + ratios) if score == maximum)
    return JointGeometryResult(ids, configs, tuple(ratios), tuple(post), tuple(prior), zo, zp,
                               marginals, prior_marginals, map_ids, budget)
