"""Experimental soft grouping of an immutable individual-pair evidence graph.

For each TESTED pair, w_ij = logit(separation_confidence) - logBF_ij.
The exact operational objective is sum(w_ij for pairs in the same class).
Positive edges favor grouping, negative edges favor separation, and untested
pairs have no objective term. A class must have one nonempty common genomic
intersection, but it need not be a clique of statistically compatible pairs.

Optimization is bounded deterministic multistart coordinate ascent. It is
NOT guaranteed to find the global optimum. Pair observations are dependent,
so neither this composite graph score nor its membership margins are joint
likelihoods or calibrated posterior probabilities. Independently optimized
confidence levels need not form nested partitions.

This module never recalculates a pair score, invents an interval, pools native
emissions, filters an observation, or edits the frozen pair ledger.
"""
from __future__ import annotations

import hashlib
import math
from numbers import Integral

import numpy as np

try:
    from numba import njit
except ImportError:
    njit = None


def _optimize_reference(starts, ends, offsets, neighbors, weights, edge_a, edge_b,
                        edge_weights, order, max_passes, tolerance):
    count = len(starts)
    labels = np.arange(count, dtype=np.int64)
    head = np.arange(count, dtype=np.int64)
    next_node = np.full(count, -1, dtype=np.int64)
    previous_node = np.full(count, -1, dtype=np.int64)
    lower, upper = starts.copy(), ends.copy()
    size = np.ones(count, dtype=np.int64)
    free_classes = np.empty(count, dtype=np.int64)
    n_free = 0
    totals = np.zeros(count, dtype=np.float64)
    stamps = np.zeros(count, dtype=np.int64)
    touched = np.empty(count, dtype=np.int64)
    stamp = 0
    moves_by_pass = np.zeros(max_passes, dtype=np.int64)
    objectives = np.zeros(max_passes + 1, dtype=np.float64)
    completed = 0
    converged = False
    for step in range(max_passes):
        moves = 0
        for node in order:
            stamp += 1
            n_touched = 0
            current = labels[node]
            for position in range(offsets[node], offsets[node + 1]):
                group = labels[neighbors[position]]
                if stamps[group] != stamp:
                    stamps[group] = stamp
                    totals[group] = 0.0
                    touched[n_touched] = group
                    n_touched += 1
                totals[group] += weights[position]
            current_score = totals[current] if stamps[current] == stamp else 0.0
            best_score, best_group = current_score, current
            # A new singleton has objective contribution zero. Unknown edges
            # therefore cannot manufacture evidence for joining another class.
            if current_score < -tolerance:
                best_score, best_group = 0.0, -1
            for candidate in range(n_touched):
                group = touched[candidate]
                if group == current:
                    continue
                if max(lower[group], starts[node]) >= min(upper[group], ends[node]):
                    continue
                score = totals[group]
                if score > best_score + tolerance:
                    best_score, best_group = score, group
                elif (best_group >= 0 and best_group != current and
                      abs(score - best_score) <= tolerance and group < best_group):
                    # Stable tie among improving alternatives; never move on
                    # a tie with the current assignment or an empty singleton.
                    best_group = group
            if best_group == current:
                continue
            before, after = previous_node[node], next_node[node]
            if before >= 0:
                next_node[before] = after
            else:
                head[current] = after
            if after >= 0:
                previous_node[after] = before
            size[current] -= 1
            if size[current] == 0:
                free_classes[n_free] = current
                n_free += 1
            elif starts[node] == lower[current] or ends[node] == upper[current]:
                member = head[current]
                lower[current], upper[current] = starts[member], ends[member]
                member = next_node[member]
                while member >= 0:
                    lower[current] = max(lower[current], starts[member])
                    upper[current] = min(upper[current], ends[member])
                    member = next_node[member]
            if best_group == -1:
                n_free -= 1
                best_group = free_classes[n_free]
                head[best_group] = -1
                lower[best_group], upper[best_group] = starts[node], ends[node]
            else:
                lower[best_group] = max(lower[best_group], starts[node])
                upper[best_group] = min(upper[best_group], ends[node])
            previous_node[node] = -1
            next_node[node] = head[best_group]
            if head[best_group] >= 0:
                previous_node[head[best_group]] = node
            head[best_group] = node
            size[best_group] += 1
            labels[node] = best_group
            moves += 1
        objective = 0.0
        for edge in range(len(edge_a)):
            if labels[edge_a[edge]] == labels[edge_b[edge]]:
                objective += edge_weights[edge]
        moves_by_pass[step] = moves
        objectives[step + 1] = objective
        completed = step + 1
        if moves == 0:
            converged = True
            break
    return labels, moves_by_pass[:completed], objectives[:completed + 1], converged


def _diagnostics_reference(starts, ends, offsets, neighbors, weights, labels):
    count = len(starts)
    lower = np.full(count, np.iinfo(np.int64).min, dtype=np.int64)
    upper = np.full(count, np.iinfo(np.int64).max, dtype=np.int64)
    for node in range(count):
        group = labels[node]
        lower[group] = max(lower[group], starts[node])
        upper[group] = min(upper[group], ends[node])
    totals = np.zeros(count, dtype=np.float64)
    edge_counts = np.zeros(count, dtype=np.int64)
    stamps = np.zeros(count, dtype=np.int64)
    touched = np.empty(count, dtype=np.int64)
    floating = np.zeros((count, 5), dtype=np.float64)
    integers = np.zeros((count, 5), dtype=np.int64)
    for node in range(count):
        stamp = node + 1
        n_touched = 0
        current = labels[node]
        for position in range(offsets[node], offsets[node + 1]):
            group, weight = labels[neighbors[position]], weights[position]
            if stamps[group] != stamp:
                stamps[group] = stamp
                totals[group] = 0.0
                edge_counts[group] = 0
                touched[n_touched] = group
                n_touched += 1
            totals[group] += weight
            edge_counts[group] += 1
            if group == current:
                integers[node, 0] += 1
                if weight < 0:
                    integers[node, 1] += 1
                    floating[node, 4] -= weight
            elif weight > 0:
                integers[node, 2] += 1
        current_score = totals[current] if stamps[current] == stamp else 0.0
        best_alternative, best_group, best_edges = 0.0, -1, 0
        for candidate in range(n_touched):
            group = touched[candidate]
            if group == current or max(lower[group], starts[node]) >= min(upper[group], ends[node]):
                continue
            score = totals[group]
            if score > best_alternative or (score == best_alternative and best_group >= 0 and group < best_group):
                best_alternative, best_group, best_edges = score, group, edge_counts[group]
        floating[node, 0] = current_score
        floating[node, 1] = best_alternative
        floating[node, 2] = current_score - best_alternative
        floating[node, 3] = best_alternative - current_score
        integers[node, 3] = best_group + 1  # Zero denotes the singleton alternative.
        integers[node, 4] = best_edges
    return floating, integers


_optimize = njit(cache=True)(_optimize_reference) if njit is not None else _optimize_reference
_node_diagnostics = njit(cache=True)(_diagnostics_reference) if njit is not None else _diagnostics_reference


def _integer_array(values, name):
    array = np.asarray(values)
    if array.ndim != 1 or (array.size and array.dtype.kind not in "iu"):
        raise ValueError(f"{name} must be a one-dimensional integer array")
    return np.ascontiguousarray(array, dtype=np.int64)


def cluster_signed_pairs(starts, ends, node_a, node_b, log_bf, confidence, *,
                         observation_ids=None, max_passes=30, n_starts=3,
                         tolerance=1e-10):
    """Return (positive labels, JSON-serializable optimization diagnostics).

    All arrays of node diagnostics and labels follow the INPUT node order.
    Supplying unique stable observation IDs makes results invariant to node
    reindexing and pair ordering, up to floating-point arithmetic. Without
    supplied IDs, positional IDs provide determinism but not reindex invariance.
    Exactly one record per unordered tested pair is required. Unknown pairs do
    not receive synthetic weights, and an input node is never dropped.
    """
    starts, ends = _integer_array(starts, "starts"), _integer_array(ends, "ends")
    first, second = _integer_array(node_a, "node_a"), _integer_array(node_b, "node_b")
    evidence = np.asarray(log_bf, dtype=np.float64)
    count = len(starts)
    if ends.shape != starts.shape or np.any(ends <= starts):
        raise ValueError("Each node requires a nonempty half-open interval")
    if evidence.ndim != 1 or not (len(first) == len(second) == len(evidence)):
        raise ValueError("Pair endpoints and logBF arrays must have equal lengths")
    if np.any(~np.isfinite(evidence)):
        raise ValueError("Saved pair logBFs must be finite; no silent clipping is permitted")
    if np.any(first < 0) or np.any(second < 0) or np.any(first >= count) or np.any(second >= count) or np.any(first == second):
        raise ValueError("Invalid or self-paired observation index")
    if not math.isfinite(confidence) or not .5 < confidence < 1:
        raise ValueError("Separation confidence must be strictly between 0.5 and 1")
    if not isinstance(max_passes, Integral) or max_passes < 1 or not isinstance(n_starts, Integral) or n_starts < 1:
        raise ValueError("Positive integer pass and multistart budgets required")
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("Finite positive improvement tolerance required")
    ids = np.asarray([str(i) for i in range(count)] if observation_ids is None else observation_ids, dtype=str)
    if ids.shape != (count,) or len(np.unique(ids)) != count:
        raise ValueError("One unique stable observation ID is required per node")
    canonical = np.lexsort((ids, ends, starts))
    inverse = np.empty(count, dtype=np.int64)
    inverse[canonical] = np.arange(count)
    cs, ce, ci = starts[canonical], ends[canonical], ids[canonical]
    a = np.minimum(inverse[first], inverse[second])
    b = np.maximum(inverse[first], inverse[second])
    pair_order = np.lexsort((b, a))
    a, b, evidence = a[pair_order], b[pair_order], evidence[pair_order]
    if len(a) > 1 and np.any((a[1:] == a[:-1]) & (b[1:] == b[:-1])):
        raise ValueError("Duplicate tested pairs would double-count evidence")
    threshold = math.log(confidence) - math.log1p(-confidence)
    edge_weights = threshold - evidence
    source, target = np.r_[a, b], np.r_[b, a]
    directed_order = np.lexsort((target, source))
    source, neighbors = source[directed_order], target[directed_order]
    weights = np.r_[edge_weights, edge_weights][directed_order]
    offsets = np.r_[0, np.cumsum(np.bincount(source, minlength=count))].astype(np.int64)
    starts_diagnostics = []
    best_labels, best_objective, selected_start = np.arange(count), -math.inf, 0
    for attempt in range(n_starts):
        if attempt == 0:
            order = np.arange(count, dtype=np.int64)
            order_name = "canonical_coordinate_and_observation_id"
        else:
            hashes = np.asarray([int.from_bytes(hashlib.sha256(f"soft_pair_graph:{attempt}:{identity}".encode()).digest()[:8], "big") for identity in ci], dtype=np.uint64)
            order = np.lexsort((np.arange(count), hashes)).astype(np.int64)
            order_name = f"stable_observation_id_sha256_start_{attempt}"
        labels, moves, objectives, converged = _optimize(cs, ce, offsets, neighbors, weights,
                                                       a, b, edge_weights, order, max_passes, tolerance)
        allowance = 1e-10 * np.maximum(1., np.abs(objectives[:-1]))
        if np.any(np.diff(objectives) < -allowance):
            raise FloatingPointError("Coordinate updates decreased the declared graph objective")
        objective = float(objectives[-1])
        starts_diagnostics.append({"start": attempt, "order": order_name, "objective": objective,
                                   "passes": len(moves), "moves_by_pass": moves.tolist(),
                                   "objective_by_pass": objectives.tolist(), "converged": bool(converged),
                                   "classes": int(len(np.unique(labels)))})
        if objective > best_objective + tolerance:
            best_labels, best_objective, selected_start = labels.copy(), objective, attempt
    # Canonical positive labels depend on class membership, not arbitrary
    # mutable internal class IDs from coordinate moves.
    remap = np.full(count, -1, dtype=np.int64)
    n_classes = 0
    for index in range(count):
        old = best_labels[index]
        if remap[old] < 0:
            remap[old] = n_classes
            n_classes += 1
    best_labels = remap[best_labels]
    floating, integers = _node_diagnostics(cs, ce, offsets, neighbors, weights, best_labels)
    labels = np.empty(count, dtype=np.int64)
    labels[canonical] = best_labels + 1
    float_input, int_input = np.empty_like(floating), np.empty_like(integers)
    float_input[canonical], int_input[canonical] = floating, integers
    within = best_labels[a] == best_labels[b]
    objective_check = float(np.sum(edge_weights[within]))
    if not np.isclose(objective_check, best_objective, rtol=1e-10, atol=1e-8):
        raise FloatingPointError("Final objective differs from independently summed immutable edges")
    anchors = []
    for group in range(n_classes):
        members = best_labels == group
        left, right = int(cs[members].max()), int(ce[members].min())
        if left >= right:
            raise AssertionError("A class lost its common genomic intersection")
        anchors.append({"label": group + 1, "common_start": left, "common_end": right,
                        "observations": int(members.sum())})
    diagnostics = {
        "method": "experimental_deterministic_multistart_single_node_coordinate_ascent",
        "objective_definition": "sum within-class tested edges of logit(confidence)-saved_logBF",
        "interpretation": "operational dependent-pair composite graph score, NOT likelihood or calibrated posterior",
        "global_optimum_guaranteed": False, "confidence_partitions_guaranteed_nested": False,
        "unknown_edges": "zero objective contribution; never invented similarity or repulsion",
        "spatial_constraint": "every class has a nonempty common genomic intersection",
        "pair_scores_recomputed": False, "observations_retained": count, "tested_pairs": len(a),
        "confidence": float(confidence), "edge_threshold_log_odds": threshold,
        "max_passes": int(max_passes), "n_starts": int(n_starts), "tolerance": float(tolerance),
        "selected_start": selected_start, "objective": objective_check,
        "selected_start_converged": starts_diagnostics[selected_start]["converged"],
        "starts": starts_diagnostics, "classes": n_classes, "class_anchors": anchors,
        "within_tested_pairs": int(within.sum()),
        "within_negative_residual_pairs": int(np.sum(within & (edge_weights < 0))),
        "within_negative_residual_weight": float(-np.sum(edge_weights[within & (edge_weights < 0)])),
        "across_positive_residual_pairs": int(np.sum(~within & (edge_weights > 0))),
        "across_positive_residual_weight": float(np.sum(edge_weights[~within & (edge_weights > 0)])),
        "node_diagnostics_order": "input node order; alternative label0 means singleton",
        "node_current_internal_score": float_input[:, 0].tolist(),
        "node_best_alternative_score": float_input[:, 1].tolist(),
        "node_alternative_margin": float_input[:, 2].tolist(),
        "node_best_alternative_gain": float_input[:, 3].tolist(),
        "node_within_negative_weight": float_input[:, 4].tolist(),
        "node_within_tested_count": int_input[:, 0].tolist(),
        "node_within_negative_count": int_input[:, 1].tolist(),
        "node_across_positive_count": int_input[:, 2].tolist(),
        "node_best_alternative_label": int_input[:, 3].tolist(),
        "node_best_alternative_tested_count": int_input[:, 4].tolist(),
    }
    return labels, diagnostics
