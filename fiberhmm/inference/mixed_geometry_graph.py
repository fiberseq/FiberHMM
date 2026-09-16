"""Exact family/geometry inference via mixed blocks and a conflict-graph DAG.

Geometry compatibility has three pair types: never overlap, always overlap,
or mixed across geometry alternatives. Its normalization factors over connected
components of the MIXED pair graph. Within each block, enumerate only feasible
concrete tilings and family subsets. Nonempty subsets with identical external
always-conflict neighborhoods are compressed into one superstate. Their SUM
weights give partition functions, while separate MAX weights give family MAP.
The implicit empty configuration remains weight one. Conditional probabilities
recover original family/geometry marginals. No hypotheses or priors change.

The posterior uses sum(family eta) plus the exact geometry-integrated native
likelihood ratio of each selected subset. Both observation and prior graph
partitions are mandatory. Geometry decisions maximize conditional native
likelihood times fixed geometry weights; family activities never rank geometry
alternatives within a selected subset. Resource overruns fail explicitly.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import heapq
import json
import math
from numbers import Integral

import numpy as np
from scipy.optimize import minimize
from scipy.sparse import csr_matrix

from .joint_geometry_catalog import (
    ExactGeometryBudgetError, FamilyGeometry, GeometryInterval,
    JointGeometryCatalog, _CompiledComponent,
)
from .state_conflict_graph import ConflictGraphCatalog


def _checked_probabilities(values, name, *, tolerance=1e-10):
    """Repair floating-point roundoff only; reject invalid inference results."""
    array = np.asarray(values)
    if not np.all(np.isfinite(array)) or np.any(array < -tolerance) or np.any(array > 1 + tolerance):
        raise ValueError(f"Invalid {name} probability beyond numerical tolerance {tolerance:g}")
    changed = int(np.count_nonzero((array < 0) | (array > 1)))
    np.clip(array, 0, 1, out=array)
    return changed


@dataclass(frozen=True)
class GeometrySubsetNode:
    node_id: str
    block_index: int
    configuration_index: int
    family_indices: tuple[int, ...]
    start: int
    end: int
    configuration_indices: tuple[int, ...]
    external_family_mask: int


@dataclass
class MixedGeometryResult:
    family_ids: tuple[str, ...]
    geometry_ids: tuple[str, ...]
    family_marginals: np.ndarray
    geometry_marginals: np.ndarray
    prior_family_marginals: np.ndarray
    prior_geometry_marginals: np.ndarray
    equal_prior_family_marginals: np.ndarray | None
    equal_prior_geometry_marginals: np.ndarray | None
    log_z_observation: np.ndarray
    log_z_prior: float
    equal_prior_log_z_observation: np.ndarray
    log_marginal_likelihood_ratio: np.ndarray
    node_marginals: np.ndarray
    prior_node_marginals: np.ndarray
    node_log_likelihood_ratios: np.ndarray
    map_family_ids: tuple[tuple[str, ...], ...] | None
    map_geometry_ids: tuple[tuple[str, ...], ...] | None
    map_geometry_indices: tuple[tuple[int, ...], ...] | None
    map_node_indices: tuple[tuple[int, ...], ...] | None
    activity_gradient: np.ndarray
    diagnostics: dict
    _catalog: object = field(repr=False)
    _activities: np.ndarray = field(repr=False)
    _evidence: np.ndarray = field(repr=False)

    def configuration_probability(self, family_ids, row=0, *, prior=False, equal_prior=False):
        """Exact full family-configuration probability; no dense 2**F table."""
        if prior and equal_prior:
            raise ValueError("prior and equal_prior are mutually exclusive")
        requested = frozenset(family_ids)
        if len(requested) != len(family_ids) or not requested.issubset(self.family_ids):
            raise ValueError("Unknown or duplicate family ID")
        if not 0 <= row < len(self.log_z_observation):
            raise IndexError("Unknown observation row")
        selected = []
        native = 0.0
        for block in self._catalog.blocks:
            key = tuple(i for i, f in enumerate(block.families) if self.family_ids[f] in requested)
            if key:
                config = block.component.configuration_index.get(key)
                if config is None:
                    return 0.0
                node = block.node_by_configuration[config]
                selected.append(node)
                if not prior:
                    ratios, _ = block.native_scores(self._evidence[row:row + 1])
                    native += float(ratios[0, config])
        selected_mask = sum(1 << i for i in selected)
        if any(self._catalog.adjacency[i] & selected_mask for i in selected):
            return 0.0
        activity = 0.0 if equal_prior else math.fsum(
            self._activities[i] for i, family_id in enumerate(self.family_ids) if family_id in requested)
        normalizer = self.log_z_prior if prior else (
            self.equal_prior_log_z_observation[row] if equal_prior else self.log_z_observation[row])
        return float(math.exp(min(0.0, activity + native - normalizer)))


class _MixedBlock:
    def __init__(self, owner, families, index, options):
        self.families = tuple(sorted(families, key=lambda f: owner.family_ids[f]))
        self.geometries = tuple(sorted((g for f in self.families for g in owner.families[f].geometry_indices),
                                       key=lambda g: owner.geometry_ids[g]))
        local = {g: i for i, g in enumerate(self.geometries)}
        local_families = [FamilyGeometry(owner.family_ids[f],
                          tuple(local[g] for g in owner.families[f].geometry_indices), owner.families[f].weights)
                          for f in self.families]
        self.catalog = JointGeometryCatalog([owner.geometries[g] for g in self.geometries], local_families,
            max_enumerated_families=max(1, len(families)), **options)
        # A mixed connected block has one any-overlap component. A singleton
        # block is also explicitly enumerated so its nonempty subset is a node.
        if len(self.catalog._components) != 1:
            raise AssertionError("Mixed block unexpectedly split during exact compilation")
        component = self.catalog._components[0]
        self.component = (_CompiledComponent(self.catalog, tuple(range(len(local_families))), False)
                          if component.factorized else component)
        self.index = index
        self.node_by_configuration = {}
        self.node_by_family_subset = {}
        self.nodes = ()

    def install_groups(self, owner):
        """Precompute dense indices only for actually feasible configurations."""
        self.nodes = tuple(i for i, node in enumerate(owner.nodes) if node.block_index == self.index)
        self.group_configs = tuple(owner.nodes[i].configuration_indices for i in self.nodes)
        self.ordered_configs = np.asarray([c for group in self.group_configs for c in group], dtype=np.int64)
        self.group_offsets = np.cumsum([0] + [len(group) for group in self.group_configs[:-1]], dtype=np.int64)
        self.config_group = np.zeros(len(self.component.configurations), dtype=np.int64)
        for group, configs in enumerate(self.group_configs):
            self.config_group[list(configs)] = group
        self.nonempty = self.ordered_configs

    def config_scores(self, ratios, activities):
        return ratios + np.asarray(self.component.family_incidence @ activities[list(self.families)]).reshape(-1)

    def group_scores(self, config_scores, *, maximum=False):
        values = config_scores[:, self.ordered_configs]
        reduction = np.maximum if maximum else np.logaddexp
        return reduction.reduceat(values, self.group_offsets, axis=1)

    def configuration_marginals(self, config_scores, group_scores, node_marginals):
        mass = np.zeros_like(config_scores)
        indices = self.nonempty
        with np.errstate(invalid="ignore"):
            conditional = np.exp(config_scores[:, indices] - group_scores[:, self.config_group[indices]])
        conditional[~np.isfinite(conditional)] = 0.0
        mass[:, indices] = conditional * node_marginals[:, np.asarray(self.nodes)[self.config_group[indices]]]
        # Empty has no family/geometry contribution, but retain its probability
        # for the exact full configuration interpretation.
        mass[:, self.component.configuration_index[()]] = np.clip(1 - node_marginals[:, self.nodes].sum(axis=1), 0, 1)
        return mass

    def family_marginals(self, config_mass):
        return np.asarray(self.component.family_incidence.T @ config_mass.T).T

    def native_scores(self, global_evidence):
        return self.component.scores(global_evidence[:, self.geometries])

    def geometry_marginals(self, tile_scores, config_mass):
        component = self.component
        numerator = np.logaddexp.reduceat(tile_scores, component.config_offsets, axis=1)
        with np.errstate(invalid="ignore"):
            conditional = np.exp(tile_scores - numerator[:, component.tile_configs])
        conditional[~np.isfinite(conditional)] = 0.0
        weighted = conditional * config_mass[:, component.tile_configs]
        # Component geometry order is canonical ID order; its local catalog
        # geometries use the same canonical order as this global block mapping.
        result = np.asarray(component.geometry_incidence.T @ weighted.T).T
        return result

    def best_configuration(self, node, config_scores_row):
        indices = node.configuration_indices
        maximum = max(config_scores_row[i] for i in indices)
        return min((i for i in indices if config_scores_row[i] == maximum),
                   key=lambda i: (len(self.component.configurations[i]), self.component.configurations[i]))

    def best_geometry(self, configuration, tile_scores_row):
        component = self.component
        k = configuration
        lo = component.config_offsets[k]
        hi = component.config_offsets[k + 1] if k + 1 < len(component.config_offsets) else component.n_tilings
        tile = int(lo + np.argmax(tile_scores_row[lo:hi]))
        return tuple(self.geometries[g] for g in component.tilings[tile])


class MixedGeometryGraphCatalog:
    """Exact scalable adapter; array columns retain input family/geometry order."""
    def __init__(self, geometries, families, *, max_tilings_per_block=250_000,
                 max_configurations_per_block=100_000, max_compile_steps_per_block=1_000_000,
                 max_total_tilings=1_000_000, max_superstates=100_000,
                 max_dag_nodes=1_000_000, max_batch_cells=50_000_000,
                 max_working_cells=4_000_000, max_evidence_cells=25_000_000,
                 max_output_cells=50_000_000):
        self.geometries, self.families = tuple(geometries), tuple(families)
        self.geometry_ids = tuple(g.geometry_id for g in self.geometries)
        self.family_ids = tuple(f.family_id for f in self.families)
        for name, value in (("max_tilings_per_block", max_tilings_per_block),
                            ("max_configurations_per_block", max_configurations_per_block),
                            ("max_compile_steps_per_block", max_compile_steps_per_block),
                            ("max_total_tilings", max_total_tilings), ("max_superstates", max_superstates),
                            ("max_dag_nodes", max_dag_nodes), ("max_batch_cells", max_batch_cells),
                            ("max_working_cells", max_working_cells), ("max_evidence_cells", max_evidence_cells),
                            ("max_output_cells", max_output_cells)):
            if not isinstance(value, Integral) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
            setattr(self, name, int(value))
        if len(set(self.geometry_ids)) != len(self.geometry_ids) or len(set(self.family_ids)) != len(self.family_ids):
            raise ValueError("Geometry and family IDs must each be unique")
        membership = np.full(len(self.geometries), -1, dtype=int)
        for f, family in enumerate(self.families):
            if not isinstance(family.family_id, str) or not family.family_id or not family.geometry_indices:
                raise ValueError("Each family needs an ID and nonempty geometry support")
            for g in family.geometry_indices:
                if not isinstance(g, Integral) or isinstance(g, bool) or not 0 <= g < len(self.geometries):
                    raise ValueError("Invalid geometry index")
                if membership[g] >= 0:
                    raise ValueError("Families must be a disjoint geometry partition")
                membership[g] = f
        if np.any(membership < 0):
            raise ValueError("Every geometry must belong to one family")
        # First identify exact any-overlap family pairs with an interval sweep.
        any_edges = set()
        active, counts = [], {}
        for g in sorted(range(len(geometries)), key=lambda g: (geometries[g].start, geometries[g].end, self.geometry_ids[g])):
            geometry = geometries[g]
            while active and active[0][0] <= geometry.start:
                _, old = heapq.heappop(active)
                counts[old] -= 1
                if counts[old] == 0:
                    del counts[old]
            f = int(membership[g])
            any_edges.update((min(f, other), max(f, other)) for other in counts if f != other)
            heapq.heappush(active, (geometry.end, f)); counts[f] = counts.get(f, 0) + 1
        minimum_end = [min(geometries[g].end for g in f.geometry_indices) for f in families]
        maximum_start = [max(geometries[g].start for g in f.geometry_indices) for f in families]
        parent = list(range(len(families)))
        def find(f):
            while parent[f] != f:
                parent[f] = parent[parent[f]]; f = parent[f]
            return f
        always, mixed = set(), set()
        for a, b in any_edges:
            if maximum_start[a] < minimum_end[b] and maximum_start[b] < minimum_end[a]:
                always.add((a, b))
            else:
                mixed.add((a, b)); parent[find(b)] = find(a)
        grouped = {}
        for f in range(len(families)):
            grouped.setdefault(find(f), []).append(f)
        self.always_conflicting_family_pairs = tuple(sorted(always))
        self.mixed_family_pairs = tuple(sorted(mixed))
        family_block_root = [find(f) for f in range(len(families))]
        external_neighbors = [0] * len(families)
        for a, b in always:
            if family_block_root[a] != family_block_root[b]:
                external_neighbors[a] |= 1 << b
                external_neighbors[b] |= 1 << a
        self.blocks, proposed_nodes = [], []
        tiling_total = 0
        block_options = dict(max_tilings_per_component=max_tilings_per_block,
                             max_configurations_per_component=max_configurations_per_block,
                             max_compile_steps_per_component=max_compile_steps_per_block,
                             max_total_tilings=max_total_tilings,
                             max_working_cells=max_working_cells,
                             max_evidence_cells=max_evidence_cells, max_output_cells=max_output_cells)
        for block_index, group in enumerate(sorted(grouped.values(), key=lambda fs: min(self.family_ids[f] for f in fs))):
            block = _MixedBlock(self, group, block_index, block_options)
            self.blocks.append(block)
            tiling_total += block.component.n_tilings
            if tiling_total > max_total_tilings:
                raise ExactGeometryBudgetError("Mixed-block total tiling budget exceeded; no truncation")
            signatures = {}
            for config_index, config in enumerate(block.component.configurations):
                if not config:
                    continue
                selected_families = tuple(block.families[block.component.families[f]] for f in config)
                signature = 0
                for f in selected_families:
                    signature |= external_neighbors[f]
                signatures.setdefault(signature, []).append(config_index)
            for signature, configurations in sorted(signatures.items()):
                representative = configurations[0]
                selected_families = tuple(block.families[block.component.families[f]]
                                          for f in block.component.configurations[representative])
                all_families = {block.families[block.component.families[f]]
                                for c in configurations for f in block.component.configurations[c]}
                geometry_support = [g for f in all_families for g in self.families[f].geometry_indices]
                identity = json.dumps([sorted(self.family_ids[f] for f in block.families),
                                       sorted(self.family_ids[f] for f in range(len(families)) if signature & (1 << f))],
                                      separators=(",", ":"))
                proposed_nodes.append(GeometrySubsetNode("external_" + hashlib.sha256(identity.encode()).hexdigest()[:24],
                    block_index, representative, selected_families,
                    min(self.geometries[g].start for g in geometry_support),
                    max(self.geometries[g].end for g in geometry_support), tuple(configurations), signature))
                if len(proposed_nodes) > max_superstates:
                    raise ExactGeometryBudgetError("Mixed-block superstate budget exceeded; no truncation")
        # The graph kernel eliminates from the last node. Genomic-end order
        # keeps local interactions local without changing the exact model.
        self.nodes = tuple(sorted(proposed_nodes, key=lambda n: (n.end, n.start, n.node_id)))
        self.adjacency = [0] * len(self.nodes)
        family_nodes = [[] for _ in self.families]
        block_nodes = [[] for _ in self.blocks]
        for node_index, node in enumerate(self.nodes):
            block = self.blocks[node.block_index]
            for configuration in node.configuration_indices:
                block.node_by_configuration[configuration] = node_index
            block_nodes[node.block_index].append(node_index)
            for f in node.family_indices:
                family_nodes[f].append(node_index)
        for nodes in block_nodes:
            mask = sum(1 << n for n in nodes)
            for n in nodes:
                self.adjacency[n] |= mask ^ (1 << n)
        family_block = {f: b.index for b in self.blocks for f in b.families}
        family_node_masks = [sum(1 << n for n in ns) for ns in family_nodes]
        for a, b in always:
            if family_block[a] != family_block[b]:
                for x in family_nodes[a]:
                    self.adjacency[x] |= family_node_masks[b]
                for y in family_nodes[b]:
                    self.adjacency[y] |= family_node_masks[a]
        for block in self.blocks:
            block.install_groups(self)
        self.graph = ConflictGraphCatalog(self.adjacency, max_dag_nodes=max_dag_nodes, max_batch_cells=max_batch_cells)
        self.total_tilings = tiling_total
        self.compile_summary = {"families": len(families), "geometries": len(geometries),
            "mixed_blocks": len(self.blocks), "mixed_pairs": len(mixed), "always_conflicting_pairs": len(always),
            "superstates": len(self.nodes), "tilings": tiling_total,
            "blocks": [{"families": len(b.families), "geometries": len(b.geometries),
                        "feasible_subsets_including_empty": len(b.component.configurations),
                        "nonempty_external_signature_groups": len(b.nodes),
                        "compatible_tilings_including_empty": b.component.n_tilings,
                        "compile_steps": b.component.compile_steps} for b in self.blocks]}
        self.diagnostics = self.compile_summary

    def _evidence(self, values):
        evidence = np.asarray(values, dtype=float)
        if evidence.ndim == 1:
            evidence = evidence[None, :]
        if evidence.ndim != 2 or evidence.shape[1] != len(self.geometries):
            raise ValueError("Expected units-by-geometries native likelihood ratios")
        if np.any(np.isnan(evidence)) or np.any(np.isposinf(evidence)):
            raise ValueError("Evidence must be finite or negative infinity")
        if evidence.size > self.max_evidence_cells:
            raise ExactGeometryBudgetError("Evidence matrix budget exceeded")
        return evidence

    def _activities(self, values):
        eta = np.zeros(len(self.families)) if values is None else np.asarray(values, dtype=float)
        if eta.shape != (len(self.families),) or not np.all(np.isfinite(eta)):
            raise ValueError("Expected one finite activity per family")
        return eta

    def _step(self, batch_size):
        if not isinstance(batch_size, Integral) or isinstance(batch_size, bool) or batch_size < 1:
            raise ValueError("batch_size must be positive")
        columns = max(1, self.total_tilings + len(self.nodes))
        if columns > self.max_working_cells:
            raise ExactGeometryBudgetError("One exact observation exceeds working-matrix budget")
        graph_columns = 2 * self.graph.dag_nodes + 2 * self.graph.n
        graph_rows = self.graph.max_batch_cells // max(1, graph_columns)
        if graph_rows < 1:
            raise ExactGeometryBudgetError("One exact observation exceeds graph-DAG batch budget")
        return min(batch_size, self.max_working_cells // columns, graph_rows)

    def native_configuration_scores(self, log_evidence, *, batch_size=128):
        """Equal-activity SUM weights of external groups, not family evidence.

        These compressed weights cannot be reused by adding one activity per
        group: each constituent family configuration has its own activity sum.
        Fitting below caches uncompressed per-block ratios explicitly instead.
        """
        evidence = self._evidence(log_evidence)
        if len(evidence) * len(self.nodes) > self.max_output_cells:
            raise ExactGeometryBudgetError("Native subset evidence cache budget exceeded")
        result = np.zeros((len(evidence), len(self.nodes)))
        step = self._step(batch_size)
        for begin in range(0, len(evidence), step):
            for block in self.blocks:
                ratios, _ = block.native_scores(evidence[begin:begin + step])
                result[begin:begin + step, block.nodes] = block.group_scores(ratios)
        return result

    def _prior(self, eta):
        weights = np.zeros((1, len(self.nodes)))
        scores = []
        for block in self.blocks:
            values = block.config_scores(np.zeros((1, len(block.component.configurations))), eta)
            scores.append(values)
            weights[:, block.nodes] = block.group_scores(values)
        result = self.graph.infer(weights)
        family = np.zeros(len(self.families))
        masses = []
        for block, values in zip(self.blocks, scores):
            mass = block.configuration_marginals(values, weights[:, block.nodes], result.marginals)
            masses.append(mass)
            family[list(block.families)] = block.family_marginals(mass)[0]
        return result, family, masses

    def infer(self, log_evidence, activities=None, *, batch_size=128, include_equal_prior=True, include_map=True):
        evidence, eta = self._evidence(log_evidence), self._activities(activities)
        equal_activities = bool(np.all(eta == 0))
        n, nf, ng, nn = len(evidence), len(self.families), len(self.geometries), len(self.nodes)
        cells = n * ((nf + ng) * (2 if include_equal_prior else 1) + nn * 2)
        if cells > self.max_output_cells:
            raise ExactGeometryBudgetError("Inference output budget exceeded")
        prior, prior_family, prior_configurations = self._prior(eta)
        prior_nodes = prior.marginals[0]
        prior_geometry = np.zeros(ng)
        for block, masses in zip(self.blocks, prior_configurations):
            _, tiles = block.native_scores(np.zeros((1, ng)))
            prior_geometry[list(block.geometries)] = block.geometry_marginals(tiles, masses)[0]
        family, geometry = np.zeros((n, nf)), np.zeros((n, ng))
        equal_family, equal_geometry = (np.zeros_like(family), np.zeros_like(geometry)) if include_equal_prior else (None, None)
        node_marginals, node_ratios = np.zeros((n, nn)), np.zeros((n, nn))
        log_z, equal_z = np.zeros(n), np.zeros(n)
        chosen_nodes, chosen_geometry, chosen_family = [], [], []
        step = self._step(batch_size)
        for begin in range(0, n, step):
            finish = min(n, begin + step)
            batch = evidence[begin:finish]
            weights, native_weights, map_weights = (np.zeros((len(batch), nn)) for _ in range(3))
            tile_arrays, config_arrays, native_arrays = [], [], []
            for block in self.blocks:
                block_ratios, tiles = block.native_scores(batch)
                tile_arrays.append(tiles)
                configs = block.config_scores(block_ratios, eta)
                config_arrays.append(configs)
                native_arrays.append(block_ratios)
                weights[:, block.nodes] = block.group_scores(configs)
                native_weights[:, block.nodes] = block.group_scores(block_ratios)
                if include_map:
                    map_weights[:, block.nodes] = block.group_scores(configs, maximum=True)
            posterior = self.graph.infer(weights)
            equal = posterior if equal_activities else self.graph.infer(native_weights)
            log_z[begin:finish], equal_z[begin:finish] = posterior.log_z, equal.log_z
            node_marginals[begin:finish], node_ratios[begin:finish] = posterior.marginals, native_weights
            for block, tiles, configs, native in zip(self.blocks, tile_arrays, config_arrays, native_arrays):
                mass = block.configuration_marginals(configs, weights[:, block.nodes], posterior.marginals)
                family[begin:finish, block.families] = block.family_marginals(mass)
                geometry[np.ix_(range(begin, finish), block.geometries)] = block.geometry_marginals(tiles, mass)
                if include_equal_prior:
                    if equal_activities:
                        equal_family[begin:finish, block.families] = family[begin:finish, block.families]
                        equal_geometry[np.ix_(range(begin, finish), block.geometries)] = geometry[np.ix_(range(begin, finish), block.geometries)]
                    else:
                        equal_mass = block.configuration_marginals(native, native_weights[:, block.nodes], equal.marginals)
                        equal_family[begin:finish, block.families] = block.family_marginals(equal_mass)
                        equal_geometry[np.ix_(range(begin, finish), block.geometries)] = block.geometry_marginals(tiles, equal_mass)
            if include_map:
                # MAP of summed group mass is NOT the MAP family configuration.
                # A distinct max-product graph retains the original objective.
                for row, selection in enumerate(self.graph.map_indices(map_weights)):
                    geometry_indices = []
                    family_indices = []
                    for index in selection:
                        node = self.nodes[index]
                        block = self.blocks[node.block_index]
                        configuration = block.best_configuration(node, config_arrays[node.block_index][row])
                        geometry_indices.extend(block.best_geometry(configuration, tile_arrays[node.block_index][row]))
                        family_indices.extend(block.families[f] for f in block.component.configurations[configuration])
                    chosen_nodes.append(tuple(selection))
                    chosen_geometry.append(tuple(sorted(geometry_indices, key=lambda g: self.geometry_ids[g])))
                    chosen_family.append(tuple(sorted(self.family_ids[f] for f in family_indices)))
        roundoff_clipped = {}
        for name, values in (("family_marginals", family), ("geometry_marginals", geometry),
                             ("prior_family_marginals", prior_family), ("prior_geometry_marginals", prior_geometry),
                             ("equal_prior_family_marginals", equal_family), ("equal_prior_geometry_marginals", equal_geometry),
                             ("node_marginals", node_marginals), ("prior_node_marginals", prior_nodes)):
            if values is not None:
                changed = _checked_probabilities(values, name)
                if changed:
                    roundoff_clipped[name] = changed
        return MixedGeometryResult(self.family_ids, self.geometry_ids, family, geometry,
            prior_family, prior_geometry, equal_family, equal_geometry, log_z, float(prior.log_z[0]),
            equal_z, log_z - prior.log_z[0], node_marginals, prior_nodes, node_ratios,
            tuple(chosen_family) if include_map else None,
            tuple(tuple(self.geometry_ids[g] for g in selected) for selected in chosen_geometry) if include_map else None,
            tuple(chosen_geometry) if include_map else None, tuple(chosen_nodes) if include_map else None,
            family - prior_family, {"compile": self.compile_summary, "graph": prior.diagnostics,
                "node_weights": "sum over exact configurations for posterior; separate maximum for MAP",
                "probability_roundoff_tolerance": 1e-10, "probability_roundoff_clipped": roundoff_clipped},
            self, eta.copy(), evidence)

    def fit_activities(self, log_evidence, *, regularization=.5, prior_center=0.0,
                       max_iterations=300, batch_size=128):
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
        cache_columns = sum(len(b.component.configurations) for b in self.blocks)
        if len(evidence) * cache_columns > self.max_output_cells:
            raise ExactGeometryBudgetError("Exact activity-fit native configuration cache budget exceeded; no approximation")
        ratios = [np.empty((len(evidence), len(b.component.configurations))) for b in self.blocks]
        step = self._step(batch_size)
        for begin in range(0, len(evidence), step):
            for block, cached in zip(self.blocks, ratios):
                values, _ = block.native_scores(evidence[begin:begin + step])
                cached[begin:begin + step] = values
        def evaluate(eta):
            prior, family_prior, _ = self._prior(eta)
            ll, family_sum = -len(evidence) * float(prior.log_z[0]), np.zeros(len(self.families))
            for begin in range(0, len(evidence), step):
                finish = min(len(evidence), begin + step)
                weights = np.zeros((finish - begin, len(self.nodes)))
                config_arrays = []
                for block, cached in zip(self.blocks, ratios):
                    values = block.config_scores(cached[begin:finish], eta)
                    config_arrays.append(values)
                    weights[:, block.nodes] = block.group_scores(values)
                posterior = self.graph.infer(weights)
                ll += float(posterior.log_z.sum())
                for block, values in zip(self.blocks, config_arrays):
                    mass = block.configuration_marginals(values, weights[:, block.nodes], posterior.marginals)
                    family_sum[list(block.families)] += block.family_marginals(mass).sum(axis=0)
            gradient = family_sum - len(evidence) * family_prior
            delta = eta - center
            return -ll + .5 * regularization * float(delta @ delta), -gradient + regularization * delta, ll, family_prior
        if not len(self.families):
            return {"activities": [], "prior_marginals": [], "converged": True, "iterations": 0,
                    "log_likelihood_ratio": 0.0, "regularization": regularization, "activity_prior_center": []}
        fit = minimize(lambda eta: evaluate(eta)[:2], center, jac=True, method="L-BFGS-B",
                       options={"maxiter": int(max_iterations), "ftol": 1e-11, "gtol": 1e-7, "maxls": 30})
        _, _, ll, prior_family = evaluate(fit.x)
        return {"activities": fit.x.tolist(), "prior_marginals": prior_family.tolist(),
                "converged": bool(fit.success), "message": str(fit.message), "iterations": int(fit.nit),
                "log_likelihood_ratio": ll, "regularization": regularization,
                "activity_prior_center": center.tolist(), "max_abs_gradient": float(np.max(np.abs(fit.jac))),
                "objective": float(fit.fun), "compile_summary": self.compile_summary}


# Explicit long name above distinguishes this adapter from its block reference;
# the shorter alias is convenient for analysis code without changing semantics.
MixedGeometryCatalog = MixedGeometryGraphCatalog
