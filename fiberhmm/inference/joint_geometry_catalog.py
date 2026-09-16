"""Exact compiled family/geometry inference with explicit resource budgets.

Each feasible FAMILY configuration has prior weight exp(sum(eta)). Within it,
geometry products are normalized over physically compatible concrete tilings.
The required geometry denominator is configuration-specific, not a family
constant in general. Evidence must be additive on one shared observation base.

Two exact representations are used, never a truncated or MAP-geometry proxy:
* If every family-support pair is all-overlapping or all-disjoint, independent
  family geometry integration followed by interval DP is provably exact.
* Otherwise all compatible tilings of that conflict component are enumerated.
  Exceeding any declared budget raises ExactGeometryBudgetError.

The large factorized case exposes configuration probabilities on demand instead
of materializing an exponentially large table. It remains an exact distribution.
"""
from __future__ import annotations

from bisect import bisect_left

from dataclasses import dataclass, field
import heapq
import math
from numbers import Integral
from typing import Sequence

import numpy as np
from scipy.optimize import minimize
from scipy.sparse import csr_matrix
from scipy.special import logsumexp

from .hierarchical_state_model import FixedStateCatalog, StateInterval


class ExactGeometryBudgetError(ValueError):
    """An exact computation exceeded its declared budget; no mass was dropped."""


@dataclass(frozen=True)
class GeometryInterval:
    geometry_id: str
    start: int
    end: int

    def __post_init__(self):
        if not isinstance(self.geometry_id, str) or not self.geometry_id:
            raise ValueError("Geometry ID must be a nonempty string")
        if any(isinstance(x, bool) or not isinstance(x, Integral) for x in (self.start, self.end)):
            raise ValueError("Geometry coordinates must be integers")
        if self.end <= self.start:
            raise ValueError("Geometry must be a positive-width half-open interval")


@dataclass(frozen=True)
class FamilyGeometry:
    family_id: str
    geometry_indices: tuple[int, ...]
    weights: tuple[float, ...] | None = None


@dataclass
class ComponentPosterior:
    family_ids: tuple[str, ...]
    mode: str
    configurations: tuple[tuple[str, ...], ...] | None
    configuration_posteriors: np.ndarray | None
    configuration_priors: np.ndarray | None
    log_z_observation: np.ndarray
    log_z_prior: float
    equal_prior_log_z: np.ndarray
    _log_ratios: np.ndarray = field(repr=False)

    @property
    def configuration_log_ratios(self):
        """Explicit configuration ratios for enumerated components, else None."""
        return None if self.configurations is None else self._log_ratios

    @property
    def family_log_ratios(self):
        """Additive family ratios only where factorization was proved."""
        return self._log_ratios if self.configurations is None else None


@dataclass
class JointGeometryBatchResult:
    family_ids: tuple[str, ...]
    geometry_ids: tuple[str, ...]
    log_z_observation: np.ndarray
    log_z_prior: float
    log_marginal_likelihood_ratio: np.ndarray
    family_marginals: np.ndarray
    geometry_marginals: np.ndarray
    prior_family_marginals: np.ndarray
    prior_geometry_marginals: np.ndarray
    equal_prior_family_marginals: np.ndarray | None
    equal_prior_geometry_marginals: np.ndarray | None
    components: tuple[ComponentPosterior, ...]
    map_family_ids: tuple[tuple[str, ...], ...] | None
    map_geometry_ids: tuple[tuple[str, ...], ...] | None
    map_geometry_indices: tuple[tuple[int, ...], ...] | None
    activity_gradient: np.ndarray
    _catalog: object = field(repr=False)
    _activities: np.ndarray = field(repr=False)

    def configuration_probability(self, family_ids: Sequence[str], row: int = 0,
                                  *, prior: bool = False, equal_prior: bool = False) -> float:
        """Exact probability of this complete family configuration.

        ``prior`` and ``equal_prior`` are mutually exclusive. Equal-prior means
        observed evidence with eta=0, not prior-only prediction.
        """
        if prior and equal_prior:
            raise ValueError("Choose prior-only or equal-prior observation, not both")
        selected = frozenset(family_ids)
        if len(selected) != len(family_ids) or not selected.issubset(self.family_ids):
            raise ValueError("Unknown or duplicate family ID")
        if not 0 <= row < len(self.log_z_observation):
            raise IndexError("Unknown observation row")
        logp = 0.0
        for component, output in zip(self._catalog._components, self.components):
            local = tuple(i for i, global_i in enumerate(component.families)
                          if self.family_ids[global_i] in selected)
            if component.factorized:
                chosen = [component.representatives[i] for i in local]
                ordered = sorted(chosen, key=lambda x: (x.start, x.end))
                if any(a.end > b.start for a, b in zip(ordered, ordered[1:])):
                    return 0.0
                evidence = 0.0 if prior else math.fsum(output._log_ratios[row, i] for i in local)
                activity = 0.0 if equal_prior else math.fsum(self._activities[component.families[i]] for i in local)
            else:
                key = tuple(sorted(local))
                k = component.configuration_index.get(key)
                if k is None:
                    return 0.0
                evidence = 0.0 if prior else output._log_ratios[row, k]
                activity = 0.0 if equal_prior else math.fsum(self._activities[component.families[i]] for i in local)
            normalizer = output.log_z_prior if prior else (
                output.equal_prior_log_z[row] if equal_prior else output.log_z_observation[row])
            logp += activity + evidence - normalizer
        return float(math.exp(min(0.0, logp)))


def _overlap(a, b):
    return a.start < b.end and b.start < a.end


class _CompiledComponent:
    def __init__(self, owner, families, factorized):
        self.owner = owner
        self.families = tuple(sorted(families, key=lambda i: owner.family_ids[i]))
        self.geometries = tuple(sorted((g for f in self.families for g in owner.families[f].geometry_indices),
                                       key=lambda i: owner.geometry_ids[i]))
        self.local_geometry = {g: i for i, g in enumerate(self.geometries)}
        self.factorized = factorized
        self.representatives = tuple(owner.geometries[owner.families[f].geometry_indices[0]] for f in self.families)
        self.schedule = FixedStateCatalog([StateInterval(owner.family_ids[f], g.start, g.end)
                                          for f, g in zip(self.families, self.representatives)])
        if factorized:
            self.configurations = None
            self.compile_steps = 0
            self.n_tilings = 0
            return
        if len(self.families) > owner.max_enumerated_families:
            raise ExactGeometryBudgetError(
                f"Mixed component has {len(self.families)} families, exceeding "
                f"max_enumerated_families={owner.max_enumerated_families}; no approximation")
        choices = {}
        self.compile_steps = 0
        count = 0
        ordered = sorted(self.geometries, key=lambda g: (owner.geometries[g].start,
                         owner.geometries[g].end, owner.geometry_ids[g]))
        starts = [owner.geometries[g].start for g in ordered]
        next_positions = [bisect_left(starts, owner.geometries[g].end, i + 1)
                          for i, g in enumerate(ordered)]
        geometry_family = {g: i for i, f in enumerate(self.families)
                           for g in owner.families[f].geometry_indices}
        def visit(position, family_mask, selected_families, selected_geometries, log_weight):
            nonlocal count
            self.compile_steps += 1
            if self.compile_steps > owner.max_compile_steps_per_component:
                raise ExactGeometryBudgetError("Exact compatible-tiling compile-step budget exceeded; no truncation")
            count += 1
            if count > owner.max_tilings_per_component:
                raise ExactGeometryBudgetError("Exact compatible-tiling count budget exceeded; no truncation")
            key = tuple(sorted(selected_families))
            if key not in choices and len(choices) >= owner.max_configurations_per_component:
                raise ExactGeometryBudgetError("Exact family-configuration budget exceeded; no truncation")
            choices.setdefault(key, []).append((tuple(sorted(selected_geometries,
                                key=lambda g: owner.geometry_ids[g])), log_weight))
            # Every half-open compatible tiling has one unique start-ordered
            # traversal. Jump past all geometries overlapping the last choice;
            # a bit mask also forbids reusing a family with disjoint alternatives.
            for i in range(position, len(ordered)):
                self.compile_steps += 1
                if self.compile_steps > owner.max_compile_steps_per_component:
                    raise ExactGeometryBudgetError("Exact geometry-choice budget exceeded; no truncation")
                g = ordered[i]
                f = geometry_family[g]
                if family_mask & (1 << f):
                    continue
                visit(next_positions[i], family_mask | (1 << f), (*selected_families, f), (*selected_geometries, g),
                      log_weight + owner.log_geometry_weights[g])
        visit(0, 0, (), (), 0.0)
        self.configurations = tuple(sorted(choices, key=lambda c: tuple(owner.family_ids[self.families[i]] for i in c)))
        self.configuration_index = {c: i for i, c in enumerate(self.configurations)}
        self.config_offsets = []
        tilings, log_weights, tile_configs, denominators = [], [], [], []
        cr, cc = [], []
        for k, config in enumerate(self.configurations):
            self.config_offsets.append(len(tilings))
            block = sorted(choices[config], key=lambda x: tuple(sorted(owner.geometry_ids[g] for g in x[0])))
            denominators.append(float(logsumexp([w for _, w in block])))
            for geometry_tuple, logw in block:
                tilings.append(geometry_tuple)
                log_weights.append(logw)
                tile_configs.append(k)
            cr.extend([k] * len(config)); cc.extend(config)
        self.tilings = tuple(tilings)
        self.n_tilings = len(tilings)
        self.log_weights = np.asarray(log_weights)
        self.tile_configs = np.asarray(tile_configs, dtype=np.int64)
        self.denominators = np.asarray(denominators)
        self.config_offsets = np.asarray(self.config_offsets, dtype=np.int64)
        self.family_incidence = csr_matrix((np.ones(len(cr)), (cr, cc)),
                                          shape=(len(self.configurations), len(self.families)))
        tr, tc = [], []
        for k, geometry_tuple in enumerate(tilings):
            tr.extend([k] * len(geometry_tuple))
            tc.extend(self.local_geometry[g] for g in geometry_tuple)
        self.geometry_incidence = csr_matrix((np.ones(len(tr)), (tr, tc)),
                                            shape=(len(tilings), len(self.geometries)))

    def scores(self, evidence):
        if self.factorized:
            return np.column_stack([logsumexp(evidence[:, self.owner.families[f].geometry_indices] +
                                             self.owner.log_geometry_weights[list(self.owner.families[f].geometry_indices)], axis=1)
                                    for f in self.families]), None
        tile_scores = np.asarray(self.geometry_incidence @ evidence[:, self.geometries].T).T
        tile_scores += self.log_weights
        ratios = np.logaddexp.reduceat(tile_scores, self.config_offsets, axis=1) - self.denominators
        return ratios, tile_scores

    def family_distribution(self, ratios, activities):
        if self.factorized:
            r = self.schedule.infer(ratios + activities)
            return r.log_z, r.marginals, None
        prior_logs = np.asarray(self.family_incidence @ activities).reshape(-1)
        logs = ratios + prior_logs
        z = logsumexp(logs, axis=1)
        probabilities = np.exp(logs - z[:, None])
        marginals = np.asarray(self.family_incidence.T @ probabilities.T).T
        return z, marginals, probabilities

    def geometry_distribution(self, evidence, ratios, tile_scores, activities, log_z, family_marginals):
        if self.factorized:
            result = np.zeros((len(evidence), len(self.geometries)))
            for local_f, global_f in enumerate(self.families):
                indices = list(self.owner.families[global_f].geometry_indices)
                finite = np.isfinite(ratios[:, local_f])
                conditional = np.zeros((len(evidence), len(indices)))
                conditional[finite] = np.exp(evidence[np.ix_(finite, indices)] +
                                             self.owner.log_geometry_weights[indices] - ratios[finite, local_f, None])
                result[:, [self.local_geometry[g] for g in indices]] = conditional * family_marginals[:, local_f, None]
            return result
        prior_logs = np.asarray(self.family_incidence @ activities).reshape(-1)
        log_joint = tile_scores - self.denominators[self.tile_configs] + prior_logs[self.tile_configs]
        tile_posteriors = np.exp(log_joint - log_z[:, None])
        return np.asarray(self.geometry_incidence.T @ tile_posteriors.T).T

    def best(self, evidence_row, ratios_row, tile_scores_row, activities):
        if self.factorized:
            family_local = self.schedule.map_indices(ratios_row + activities)
            geometry_global = []
            for local in family_local:
                f = self.families[local]
                indices = sorted(self.owner.families[f].geometry_indices, key=lambda g: self.owner.geometry_ids[g])
                scores = evidence_row[indices] + self.owner.log_geometry_weights[indices]
                geometry_global.append(indices[int(np.argmax(scores))])
            return tuple(self.families[f] for f in family_local), tuple(geometry_global)
        scores = ratios_row + np.asarray(self.family_incidence @ activities).reshape(-1)
        # A MAP is a score optimum. Prefer fewer families, then canonical IDs,
        # matching the fixed-interval implementation on exact score ties.
        maximum = float(np.max(scores))
        config = min(np.flatnonzero(scores == maximum),
                     key=lambda k: (len(self.configurations[k]),
                                    tuple(self.owner.family_ids[self.families[f]]
                                          for f in self.configurations[k])))
        begin = self.config_offsets[config]
        finish = self.config_offsets[config + 1] if config + 1 < len(self.config_offsets) else self.n_tilings
        tile = int(begin + np.argmax(tile_scores_row[begin:finish]))
        return tuple(self.families[f] for f in self.configurations[config]), self.tilings[tile]


class JointGeometryCatalog:
    """Compiled exact distribution; input columns retain their original order.

    Family geometry indices must form a disjoint complete partition. Positive
    geometry weights are normalized within each family before compilation (this
    changes neither reference likelihood nor configuration-specific correction).
    """
    def __init__(self, geometries: Sequence[GeometryInterval], families: Sequence[FamilyGeometry], *,
                 max_tilings_per_component=250_000, max_configurations_per_component=100_000,
                 max_compile_steps_per_component=1_000_000, max_total_tilings=1_000_000,
                 max_enumerated_families=24, max_working_cells=4_000_000,
                 max_evidence_cells=25_000_000, max_output_cells=50_000_000):
        self.geometries, self.families = tuple(geometries), tuple(families)
        self.geometry_ids = tuple(g.geometry_id for g in self.geometries)
        self.family_ids = tuple(f.family_id for f in self.families)
        for name, value in (("max_tilings_per_component", max_tilings_per_component),
                            ("max_configurations_per_component", max_configurations_per_component),
                            ("max_compile_steps_per_component", max_compile_steps_per_component),
                            ("max_total_tilings", max_total_tilings), ("max_enumerated_families", max_enumerated_families),
                            ("max_working_cells", max_working_cells), ("max_evidence_cells", max_evidence_cells),
                            ("max_output_cells", max_output_cells)):
            if not isinstance(value, Integral) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
            setattr(self, name, int(value))
        if len(set(self.geometry_ids)) != len(self.geometry_ids) or len(set(self.family_ids)) != len(self.family_ids):
            raise ValueError("Geometry and family IDs must each be unique")
        self.log_geometry_weights = np.empty(len(self.geometries))
        geometry_family = np.full(len(self.geometries), -1, dtype=int)
        for i, family in enumerate(self.families):
            if not isinstance(family.family_id, str) or not family.family_id or not family.geometry_indices:
                raise ValueError("Each family needs an ID and at least one geometry")
            indices = tuple(family.geometry_indices)
            if any(not isinstance(g, Integral) or isinstance(g, bool) or not 0 <= g < len(self.geometries) for g in indices):
                raise ValueError("Invalid family geometry index")
            if len(set(indices)) != len(indices) or any(geometry_family[g] != -1 for g in indices):
                raise ValueError("Families must be a disjoint geometry partition")
            weights = np.ones(len(indices)) if family.weights is None else np.asarray(family.weights, dtype=float)
            if weights.shape != (len(indices),) or not np.all(np.isfinite(weights)) or np.any(weights <= 0):
                raise ValueError("Family weights must be positive, finite, and aligned")
            self.log_geometry_weights[list(indices)] = np.log(weights) - logsumexp(np.log(weights))
            geometry_family[list(indices)] = i
        if np.any(geometry_family < 0):
            raise ValueError("Every geometry must belong to exactly one family")
        parent = list(range(len(self.families)))
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]; x = parent[x]
            return x
        neighbors = [set() for _ in self.families]
        active, counts = [], {}
        for g in sorted(range(len(self.geometries)), key=lambda j: (self.geometries[j].start, self.geometries[j].end, self.geometry_ids[j])):
            geometry = self.geometries[g]
            while active and active[0][0] <= geometry.start:
                _, previous = heapq.heappop(active)
                counts[previous] -= 1
                if counts[previous] == 0:
                    del counts[previous]
            f = int(geometry_family[g])
            for previous in counts:
                if previous != f:
                    neighbors[f].add(previous); neighbors[previous].add(f)
                    parent[find(previous)] = find(f)
            heapq.heappush(active, (geometry.end, f)); counts[f] = counts.get(f, 0) + 1
        groups = {}
        for f in range(len(self.families)):
            groups.setdefault(find(f), []).append(f)
        minimum_end = [min(self.geometries[g].end for g in f.geometry_indices) for f in self.families]
        maximum_start = [max(self.geometries[g].start for g in f.geometry_indices) for f in self.families]
        self._components = []
        total_tilings = 0
        for group in sorted(groups.values(), key=lambda fs: min(self.family_ids[f] for f in fs)):
            # A nonedge is all-disjoint by the exact sweep. An edge is
            # all-overlap iff EVERY start in each family precedes EVERY end in
            # the other. Any remaining edge is genuinely mixed compatibility.
            factorized = all(maximum_start[f] < minimum_end[h] and maximum_start[h] < minimum_end[f]
                             for f in group for h in neighbors[f])
            component = _CompiledComponent(self, group, factorized)
            total_tilings += component.n_tilings
            if total_tilings > max_total_tilings:
                raise ExactGeometryBudgetError("Total exact tiling budget exceeded; no truncation")
            self._components.append(component)
        self.compile_summary = [{"family_ids": [self.family_ids[f] for f in c.families],
                                 "mode": "proven_factorized_interval_dp" if c.factorized else "exact_compatible_tilings",
                                 "families": len(c.families), "geometries": len(c.geometries),
                                 "tilings": c.n_tilings, "compile_steps": c.compile_steps,
                                 "configurations": None if c.factorized else len(c.configurations)} for c in self._components]

    def _evidence(self, values):
        evidence = np.asarray(values, dtype=float)
        if evidence.ndim == 1:
            evidence = evidence[None, :]
        if evidence.ndim != 2 or evidence.shape[1] != len(self.geometries):
            raise ValueError("Expected units-by-geometries log likelihood ratios")
        if np.any(np.isnan(evidence)) or np.any(np.isposinf(evidence)):
            raise ValueError("Evidence must be finite or negative infinity")
        if evidence.size > self.max_evidence_cells:
            raise ExactGeometryBudgetError("Evidence matrix budget exceeded")
        return evidence

    def _activities(self, values):
        result = np.zeros(len(self.families)) if values is None else np.asarray(values, dtype=float)
        if result.shape != (len(self.families),) or not np.all(np.isfinite(result)):
            raise ValueError("Activities must be finite, one per family")
        return result

    def _batch_size(self, component, requested):
        if not isinstance(requested, Integral) or requested < 1:
            raise ValueError("batch_size must be a positive integer")
        columns = max(len(component.geometries), len(component.families), component.n_tilings, 1)
        if columns > self.max_working_cells:
            raise ExactGeometryBudgetError("One exact observation exceeds working-matrix budget")
        return min(requested, self.max_working_cells // columns)

    def infer(self, log_evidence, activities=None, *, batch_size=128, include_equal_prior=True,
              include_configurations=True, include_map=True):
        evidence, eta = self._evidence(log_evidence), self._activities(activities)
        n, ng, nf = len(evidence), len(self.geometries), len(self.families)
        config_columns = sum(len(c.families) if c.factorized else len(c.configurations) for c in self._components)
        output_cells = n * ((ng + nf) * (2 if include_equal_prior else 1) + config_columns * (2 if include_configurations else 1))
        if output_cells > self.max_output_cells:
            raise ExactGeometryBudgetError("Exact inference output budget exceeded; request smaller batches/results")
        fm, gm = np.zeros((n, nf)), np.zeros((n, ng))
        pf, pg = np.zeros(nf), np.zeros(ng)
        ef, eg = (np.zeros_like(fm), np.zeros_like(gm)) if include_equal_prior else (None, None)
        global_z, prior_z = np.zeros(n), 0.0
        outputs = []
        map_families, map_geometries = [[] for _ in range(n)], [[] for _ in range(n)]
        for component in self._components:
            f, g = list(component.families), list(component.geometries)
            acts = eta[f]
            # Prior-only geometry probabilities must also use compatibility
            # normalization; zero observation LRs do not mean uniform tilings.
            zero = np.zeros((1, ng))
            prior_ratios, prior_tiles = component.scores(zero)
            pz, p_marg, p_configs = component.family_distribution(prior_ratios, acts)
            pf[f] = p_marg[0]
            pg[g] = component.geometry_distribution(zero, prior_ratios, prior_tiles, acts, pz, p_marg)[0]
            prior_z += float(pz[0])
            width = len(f) if component.factorized else len(component.configurations)
            ratios_all = np.empty((n, width))
            config_post = np.empty((n, width)) if include_configurations and not component.factorized else None
            z_all, equal_z = np.empty(n), np.empty(n)
            for begin in range(0, n, self._batch_size(component, batch_size)):
                finish = min(n, begin + self._batch_size(component, batch_size))
                batch = evidence[begin:finish]
                ratios, tiles = component.scores(batch)
                z, marginals, cp = component.family_distribution(ratios, acts)
                fm[np.ix_(range(begin, finish), f)] = marginals
                gm[np.ix_(range(begin, finish), g)] = component.geometry_distribution(batch, ratios, tiles, acts, z, marginals)
                ez, em, _ = component.family_distribution(ratios, np.zeros(len(f)))
                if include_equal_prior:
                    ef[np.ix_(range(begin, finish), f)] = em
                    eg[np.ix_(range(begin, finish), g)] = component.geometry_distribution(batch, ratios, tiles, np.zeros(len(f)), ez, em)
                ratios_all[begin:finish], z_all[begin:finish], equal_z[begin:finish] = ratios, z, ez
                if config_post is not None:
                    config_post[begin:finish] = cp
                if include_map:
                    for local_row, row in enumerate(range(begin, finish)):
                        chosen_f, chosen_g = component.best(batch[local_row], ratios[local_row],
                                                          None if tiles is None else tiles[local_row], acts)
                        map_families[row].extend(chosen_f); map_geometries[row].extend(chosen_g)
            global_z += z_all
            configurations = None if component.factorized else tuple(tuple(self.family_ids[f[i]] for i in c) for c in component.configurations)
            outputs.append(ComponentPosterior(tuple(self.family_ids[i] for i in f),
                           "proven_factorized_interval_dp" if component.factorized else "exact_compatible_tilings",
                           configurations, config_post, None if p_configs is None else p_configs[0],
                           z_all, float(pz[0]), equal_z, ratios_all))
        return JointGeometryBatchResult(self.family_ids, self.geometry_ids, global_z, prior_z,
            global_z - prior_z, fm, gm, pf, pg, ef, eg, tuple(outputs),
            tuple(tuple(sorted(self.family_ids[f] for f in row)) for row in map_families) if include_map else None,
            tuple(tuple(sorted(self.geometry_ids[g] for g in row)) for row in map_geometries) if include_map else None,
            tuple(tuple(sorted(row, key=lambda g: self.geometry_ids[g])) for row in map_geometries) if include_map else None,
            fm - pf, self, eta.copy())

    def fit_activities(self, log_evidence, *, regularization=0.5, prior_center=0.0,
                       max_iterations=300, batch_size=128):
        """Fit training-cohort activities, including -N log Z_prior.

        The explicit Gaussian center is identical for all families by default
        (eta=0). It never depends on geometry multiplicity or competitor count.
        This is a conditional model fit, not a calibrated existence/FDR test.
        Geometry likelihoods are cached once; optimization is family-level.
        """
        evidence = self._evidence(log_evidence)
        if not len(evidence):
            raise ValueError("Training needs at least one observation")
        if not math.isfinite(regularization) or regularization <= 0:
            raise ValueError("regularization must be finite and positive")
        center = np.broadcast_to(np.asarray(prior_center, dtype=float), (len(self.families),)).copy()
        if not np.all(np.isfinite(center)):
            raise ValueError("prior_center must be finite")
        if not isinstance(max_iterations, Integral) or max_iterations < 1:
            raise ValueError("max_iterations must be positive")
        cached = []
        required = len(evidence) * sum(len(c.families) if c.factorized else len(c.configurations) for c in self._components)
        if required > self.max_output_cells:
            raise ExactGeometryBudgetError("Training likelihood cache budget exceeded")
        for component in self._components:
            blocks = []
            step = self._batch_size(component, batch_size)
            for begin in range(0, len(evidence), step):
                blocks.append(component.scores(evidence[begin:begin + step])[0])
            cached.append(np.concatenate(blocks))
        def evaluate(eta):
            ll = 0.0
            gradient = np.zeros(len(self.families))
            priors = np.zeros(len(self.families))
            for component, ratios in zip(self._components, cached):
                indices = list(component.families)
                acts = eta[indices]
                zp, prior, _ = component.family_distribution(np.zeros((1, ratios.shape[1])), acts)
                observation_sum = 0.0
                posterior_sum = np.zeros(len(indices))
                step = self._batch_size(component, batch_size)
                for begin in range(0, len(ratios), step):
                    z, post, _ = component.family_distribution(ratios[begin:begin + step], acts)
                    observation_sum += float(z.sum())
                    posterior_sum += post.sum(axis=0)
                ll += observation_sum - len(evidence) * float(zp[0])
                gradient[indices] = posterior_sum - len(evidence) * prior[0]
                priors[indices] = prior[0]
            delta = eta - center
            return -ll + .5 * regularization * float(delta @ delta), -gradient + regularization * delta, ll, priors
        if not len(self.families):
            return {"activities": [], "prior_marginals": [], "converged": True, "iterations": 0,
                    "log_likelihood_ratio": 0.0, "regularization": regularization, "activity_prior_center": []}
        result = minimize(lambda eta: evaluate(eta)[:2], center, jac=True, method="L-BFGS-B",
                          options={"maxiter": int(max_iterations), "ftol": 1e-11, "gtol": 1e-7, "maxls": 30})
        _, _, ll, priors = evaluate(result.x)
        return {"activities": result.x.tolist(), "prior_marginals": priors.tolist(),
                "converged": bool(result.success), "message": str(result.message), "iterations": int(result.nit),
                "log_likelihood_ratio": ll, "regularization": regularization,
                "activity_prior_center": center.tolist(), "objective": float(result.fun),
                "max_abs_gradient": float(np.max(np.abs(result.jac))), "compile_summary": self.compile_summary}
