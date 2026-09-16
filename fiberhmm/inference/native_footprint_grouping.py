"""Cross-predictive resolution comparisons of existing native intervals.

No new interval, assignment, recall, or boundary is produced here. On a shared
accessible observation baseline H1 is one training-selected geometry with
fitted occupancy; H2 is a mutually exclusive mixture over the supplied native
geometries and accessibility. The two-fold held-out score difference is an
exploratory operational distinguishability measure, NOT a calibrated p/q value
or proof that two protected states are biologically identical.

Tiny mixture fits maximize a concave likelihood in simplex weights. Batched
EM with monotone SQUAREM acceleration stops only after a concavity-based upper
bound on the remaining likelihood gap is below tolerance. Nonconvergence is
explicit and never treated as compatibility. All-zero evidence rows contribute
exactly zero and are omitted from numerical fitting and score-SE denominators.
"""
from __future__ import annotations

import math
from numbers import Integral

import numpy as np


def _validate(log_lr, fold_ids):
    values = np.asarray(log_lr, dtype=np.float64)
    folds = np.asarray(fold_ids)
    if values.ndim != 2 or folds.shape != (values.shape[0],):
        raise ValueError("Expected units-by-geometries evidence and one fold ID per unit")
    if not len(values) or not values.shape[1]:
        raise ValueError("At least one unit and geometry are required")
    if np.any(np.isnan(values)) or np.any(np.isposinf(values)):
        raise ValueError("Native log likelihood ratios must be finite or negative infinity")
    unique = np.unique(folds)
    if len(unique) != 2:
        raise ValueError("Exactly two nonempty deterministic folds are required")
    return values, folds, unique


def _fit_mixtures(log_likelihoods, *, max_iterations=128, tolerance=1e-7):
    """Fit B independent simplex mixtures; input shape B x units x components.

    Returned optimality gap is per informative unit. For concave f(w),
    max_j gradient_j(f)-w.gradient(f) bounds f(w*)-f(w). EM's component update
    factors are the mean gradients and w.gradient=1, giving max(factor)-1.
    Acceleration is accepted only if its likelihood is not below two EM steps.
    """
    logs = np.asarray(log_likelihoods, dtype=float)
    if logs.ndim != 3 or logs.shape[2] < 2 or logs.shape[1] < 1:
        raise ValueError("Mixture likelihoods need batch, units, and >=2 components")
    if np.any(np.isnan(logs)) or np.any(np.isposinf(logs)):
        raise ValueError("Invalid mixture likelihood")
    if not isinstance(max_iterations, Integral) or max_iterations < 1 or tolerance <= 0 or not math.isfinite(tolerance):
        raise ValueError("Positive iteration limit and finite tolerance required")
    batch, units, components = logs.shape
    offset = np.max(logs, axis=2)
    if np.any(~np.isfinite(offset)):
        raise ValueError("Every unit must have a possible component (accessible baseline)")
    likelihood = np.exp(logs - offset[:, :, None])
    informative = np.any(logs != logs[:, :, :1], axis=2)
    counts = informative.sum(axis=1)
    scale = np.maximum(counts, 1)
    weights = np.full((batch, components), 1 / components)
    converged = counts == 0
    iterations = np.zeros(batch, dtype=int)
    gaps = np.zeros(batch)
    objective = np.zeros(batch)

    def evaluate(w):
        denominator = np.sum(likelihood * w[:, None, :], axis=2)
        denominator = np.maximum(denominator, np.finfo(float).tiny)
        ratios = (likelihood / denominator[:, :, None]) * informative[:, :, None]
        factors = ratios.sum(axis=1) / scale[:, None]
        factors[counts == 0] = 1
        score = np.sum(np.where(informative, np.log(denominator) + offset, 0), axis=1)
        gap = np.maximum(0., factors.max(axis=1) - 1.)
        return score, factors, gap

    def em(w):
        score, factors, gap = evaluate(w)
        new = w * factors
        new = np.maximum(new, 1e-15)
        new /= new.sum(axis=1)[:, None]
        return new, score, gap

    for iteration in range(1, max_iterations + 1):
        first, _, _ = em(weights)
        second, _, _ = em(first)
        direction = first - weights
        curvature = second - 2 * first + weights
        numerator = np.sum(direction * direction, axis=1)
        denominator = np.sum(curvature * curvature, axis=1)
        alpha = -np.sqrt(numerator / np.maximum(denominator, np.finfo(float).tiny))
        alpha = np.clip(alpha, -1000., -1.)
        proposal = weights - 2 * alpha[:, None] * direction + alpha[:, None] ** 2 * curvature
        proposal = np.maximum(proposal, 1e-15)
        proposal /= proposal.sum(axis=1)[:, None]
        accelerated, _, _ = em(proposal)
        fast_score, _, _ = evaluate(accelerated)
        slow_score, _, _ = evaluate(second)
        accepted = fast_score >= slow_score - 1e-10
        update = np.where(accepted[:, None], accelerated, second)
        weights[~converged] = update[~converged]
        objective, _, gaps = evaluate(weights)
        newly = ~converged & (gaps <= tolerance)
        iterations[newly] = iteration
        converged |= newly
        if np.all(converged):
            break
    iterations[~converged] = max_iterations
    return {"weights": weights, "log_likelihood": objective, "converged": converged,
            "iterations": iterations, "gap_per_unit": gaps, "informative_units": counts}


def _single_fits(values, folds, fold_values, needed, batch_size, max_iterations, tolerance):
    output = []
    for held in fold_values:
        train = values[folds != held]
        record = {"weights": np.zeros((values.shape[1], 2)), "log_likelihood": np.zeros(values.shape[1]),
                  "converged": np.zeros(values.shape[1], dtype=bool),
                  "gap_per_unit": np.zeros(values.shape[1]), "iterations": np.zeros(values.shape[1], dtype=int)}
        for begin in range(0, len(needed), batch_size):
            columns = needed[begin:begin + batch_size]
            protected = train[:, columns].T
            fitted = _fit_mixtures(np.stack([np.zeros_like(protected), protected], axis=2),
                                  max_iterations=max_iterations, tolerance=tolerance)
            for key in record:
                record[key][columns] = fitted[key]
        output.append(record)
    return output


def _predict(geometry_logs, weights):
    """B x units x G native logs; one B x (G+1) simplex weight vector."""
    maximum = np.maximum(0., np.max(geometry_logs, axis=2))
    accessible = weights[:, 0, None] * np.exp(-maximum)
    protected = np.sum(np.exp(geometry_logs - maximum[:, :, None]) * weights[:, None, 1:], axis=2)
    return np.log(np.maximum(accessible + protected, np.finfo(float).tiny)) + maximum


def _score_check(differences, minimum_gain, se_multiplier):
    gain = float(differences.sum())
    se = float(math.sqrt(len(differences) * differences.var(ddof=1))) if len(differences) > 1 else 0.
    threshold = max(minimum_gain, se_multiplier * se)
    return {"gain": gain, "se": se, "threshold": threshold, "beaten": bool(gain > threshold)}


def _records(indices, deltas, information, fold_records, minimum_gain, se_multiplier,
             minimum_component_weight, candidate_deltas):
    output = []
    for row, members in enumerate(indices):
        active = information[row]
        differences = deltas[row, active]
        selected_check = _score_check(differences, minimum_gain, se_multiplier)
        gain, se, threshold = (selected_check[k] for k in ("gain", "se", "threshold"))
        candidate_checks = [{"geometry_index": int(member),
                             **_score_check(candidate_deltas[row, active, j], minimum_gain, se_multiplier)}
                            for j, member in enumerate(members)]
        all_fixed_beaten = all(check["beaten"] for check in candidate_checks)
        records = [fold[row] for fold in fold_records]
        converged = all(f["converged"] for f in records)
        uses_multiple = all(sum(w > minimum_component_weight for w in f["h2_weights"][1:]) >= 2 for f in records)
        distinguished = converged and uses_multiple and selected_check["beaten"] and all_fixed_beaten
        output.append({"pair": list(map(int, members)) if len(members) == 2 else None,
            "geometry_indices": list(map(int, members)), "distinguishable": bool(distinguished),
            "compatible": bool(converged and not distinguished), "converged": bool(converged),
            "gain": gain, "se": se, "threshold": threshold, "informative_units": int(active.sum()),
            "training_selected_gain": gain, "candidate_baseline_checks": candidate_checks,
            "all_single_baselines_beaten": bool(all_fixed_beaten),
            "both_alternatives_used_across_folds": bool(uses_multiple), "folds": records,
            "max_optimality_gap_per_unit": max(max(f["optimality_gap_per_unit"], f["h1_max_optimality_gap_per_unit"]) for f in records),
            "status": "unresolved_numerics" if not converged else ("distinguished" if distinguished else "not_distinguished"),
            "interpretation": "exploratory two-fold predictive resolution comparison; not calibrated p/q or biological identity"})
    return output


def compare_pairs(log_lr, pairs, fold_ids, *, minimum_gain=math.log(100.), se_multiplier=2.,
                  minimum_component_weight=1e-8, batch_size=256, max_iterations=128,
                  tolerance=1e-7, progress_callback=None):
    """Compare all supplied pairs without downsampling; cache every H1 fit.

    Caller supplies geometrically plausible overlapping pairs. Every unit is
    evaluated using only mixture weights and H1 geometry selected on the other
    fold. A split must also beat EACH fixed candidate's cross-predictive H1,
    preventing fold-dependent H1 winner switches from manufacturing a split.
    This extra comparison guard does not refit any model using held-out data.
    Uncomputed/nonconverged pairs cannot be treated as compatible.
    """
    values, folds, fold_values = _validate(log_lr, fold_ids)
    pairs = np.asarray(pairs, dtype=np.int64)
    if pairs.size == 0:
        return []
    if pairs.ndim != 2 or pairs.shape[1] != 2 or np.any(pairs < 0) or np.any(pairs >= values.shape[1]) or np.any(pairs[:, 0] == pairs[:, 1]):
        raise ValueError("Pairs must contain two distinct in-range geometry indices")
    if not isinstance(batch_size, Integral) or batch_size < 1:
        raise ValueError("batch_size must be positive")
    singles = _single_fits(values, folds, fold_values, np.unique(pairs), batch_size, max_iterations, tolerance)
    results = []
    for begin in range(0, len(pairs), batch_size):
        current = pairs[begin:begin + batch_size]
        deltas = np.zeros((len(current), len(values)))
        candidate_deltas = np.zeros((len(current), len(values), 2))
        information = np.any(values[:, current] != 0, axis=2).T
        fold_records = []
        for index, held in enumerate(fold_values):
            train_mask, held_mask = folds != held, folds == held
            train = values[train_mask][:, current].transpose(1, 0, 2)
            fitted = _fit_mixtures(np.concatenate([np.zeros((*train.shape[:2], 1)), train], axis=2),
                                  max_iterations=max_iterations, tolerance=tolerance)
            baseline = singles[index]
            winner = np.argmax(baseline["log_likelihood"][current], axis=1)
            chosen = current[np.arange(len(current)), winner]
            test = values[held_mask][:, current].transpose(1, 0, 2)
            prediction_h2 = _predict(test, fitted["weights"])
            prediction_h1 = _predict(values[held_mask][:, chosen].T[:, :, None], baseline["weights"][chosen])
            fold_delta = prediction_h2 - prediction_h1
            deltas[:, held_mask] = fold_delta
            for candidate in range(2):
                fixed_h1 = _predict(test[:, :, candidate:candidate + 1], baseline["weights"][current[:, candidate]])
                candidate_deltas[:, held_mask, candidate] = prediction_h2 - fixed_h1
            identical = (np.all(train[:, :, 0] == 0, axis=1),
                         np.all(train[:, :, 1] == 0, axis=1),
                         np.all(train[:, :, 0] == train[:, :, 1], axis=1))
            records = []
            for row, geometry in enumerate(chosen):
                records.append({"held_fold": held.item() if hasattr(held, "item") else held,
                    "gain": float(fold_delta[row].sum()), "h1_geometry": int(geometry),
                    "h1_weights": baseline["weights"][geometry].tolist(), "h2_weights": fitted["weights"][row].tolist(),
                    "train_log_likelihood_h1": float(baseline["log_likelihood"][geometry]),
                    "train_log_likelihood_h2": float(fitted["log_likelihood"][row]),
                    "converged": bool(fitted["converged"][row] and np.all(baseline["converged"][current[row]])),
                    "h1_all_converged": bool(np.all(baseline["converged"][current[row]])),
                    "h1_max_optimality_gap_per_unit": float(np.max(baseline["gap_per_unit"][current[row]])),
                    "candidate_h1_weights": baseline["weights"][current[row]].tolist(),
                    "h2_identical_training_component_pairs": [list(pair) for pair, flag in
                        zip(((0,1),(0,2),(1,2)), identical) if flag[row]],
                    "optimality_gap_per_unit": float(fitted["gap_per_unit"][row]),
                    "iterations": int(fitted["iterations"][row])})
            fold_records.append(records)
        results.extend(_records(current, deltas, information, fold_records, minimum_gain, se_multiplier,
                                minimum_component_weight, candidate_deltas))
        if progress_callback:
            progress_callback({"pairs_completed": min(begin + batch_size, len(pairs)), "pairs_total": len(pairs),
                               "converged": sum(r["converged"] for r in results)})
    return results


def compare_group(log_lr, fold_ids, *, minimum_gain=math.log(100.), se_multiplier=2.,
                  minimum_component_weight=1e-8, max_iterations=256, tolerance=1e-7):
    """Validate a proposed operational class against its whole member mixture."""
    values, folds, fold_values = _validate(log_lr, fold_ids)
    if values.shape[1] < 2:
        raise ValueError("A group comparison requires at least two geometries")
    members = np.arange(values.shape[1])
    singles = _single_fits(values, folds, fold_values, members, 256, max_iterations, tolerance)
    deltas = np.zeros((1, len(values)))
    candidate_deltas = np.zeros((1, len(values), len(members)))
    fold_records = []
    for index, held in enumerate(fold_values):
        train = values[folds != held][None, :, :]
        fitted = _fit_mixtures(np.concatenate([np.zeros((*train.shape[:2], 1)), train], axis=2),
                              max_iterations=max_iterations, tolerance=tolerance)
        baseline = singles[index]
        chosen = int(np.argmax(baseline["log_likelihood"]))
        test = values[folds == held][None, :, :]
        prediction_h2 = _predict(test, fitted["weights"])[0]
        change = prediction_h2 - _predict(test[:, :, chosen:chosen + 1], baseline["weights"][chosen:chosen + 1])[0]
        deltas[0, folds == held] = change
        for member in members:
            prediction_h1 = _predict(test[:, :, member:member + 1], baseline["weights"][member:member + 1])[0]
            candidate_deltas[0, folds == held, member] = prediction_h2 - prediction_h1
        component_classes = {}
        for component, column in enumerate(np.r_[np.zeros((1, train.shape[1])), train[0].T]):
            column = column.copy()
            column[column == 0] = 0.  # Canonicalize signed zero for exact equality.
            component_classes.setdefault(column.tobytes(), []).append(component)
        identical_groups = [group for group in component_classes.values() if len(group) > 1]
        fold_records.append([{"held_fold": held.item() if hasattr(held, "item") else held,
            "gain": float(change.sum()), "h1_geometry": chosen, "h1_weights": baseline["weights"][chosen].tolist(),
            "h2_weights": fitted["weights"][0].tolist(), "train_log_likelihood_h1": float(baseline["log_likelihood"][chosen]),
            "train_log_likelihood_h2": float(fitted["log_likelihood"][0]),
            "converged": bool(fitted["converged"][0] and np.all(baseline["converged"])),
            "h1_all_converged": bool(np.all(baseline["converged"])),
            "h1_max_optimality_gap_per_unit": float(np.max(baseline["gap_per_unit"])),
            "candidate_h1_weights": baseline["weights"].tolist(),
            "h2_identical_training_component_groups": identical_groups,
            "optimality_gap_per_unit": float(fitted["gap_per_unit"][0]), "iterations": int(fitted["iterations"][0])}])
    return _records([members], deltas, np.any(values != 0, axis=1)[None, :], fold_records,
                    minimum_gain, se_multiplier, minimum_component_weight, candidate_deltas)[0]
