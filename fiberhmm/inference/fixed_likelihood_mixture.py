"""Fixed-grid nonparametric maximum likelihood for independent observations.

Maximize sum_i log(sum_g w_g L_ig) over the simplex. Every input row is one
independent evidence unit on a common row-wise observation/base measure.
Weights are estimated parameters, NOT a fixed prior or a marginal Bayes factor.
The objective is concave, but mixing weights need not be identifiable or unique.

Monotone accelerated EM is augmented by full-gradient vertex line searches.
Every column remains in the gradient search; no hard likelihood/weight pruning
is performed. A zero current weight therefore does not permanently exclude a
grid state. The reported Frank--Wolfe gap max_g gradient_g - N is an upper
bound on the remaining log-likelihood improvement. Convergence uses this gap
per informative row, rather than an optimizer success flag or weight changes.

This generic kernel does not assert that a grid of single intervals is an
adequate model for simultaneous protected intervals. It does not create family
labels, determine biological family count, emit q values, or claim calibrated
plug-in membership probabilities.
"""
from __future__ import annotations

import hashlib
import math
from numbers import Integral

import numpy as np
from scipy.special import logsumexp


def _validated(log_likelihood, group_ids=None):
    logs = np.asarray(log_likelihood, dtype=np.float64)
    if logs.ndim != 2 or logs.shape[1] < 1:
        raise ValueError("A rows-by-grid-state matrix with at least one state is required")
    if np.any(np.isnan(logs)) or np.any(np.isposinf(logs)):
        raise ValueError("Log likelihoods must be finite or negative infinity")
    if len(logs) and np.any(np.all(np.isneginf(logs), axis=1)):
        raise ValueError("Every observation must be possible under at least one grid state")
    if group_ids is not None:
        ids = np.asarray(group_ids)
        if ids.shape != (len(logs),) or len(np.unique(ids)) != len(logs):
            raise ValueError("group_ids must be unique: one independent observation row per evidence unit")
    return logs


def _weights(values, columns):
    if values is None:
        return np.full(columns, 1. / columns)
    weights = np.asarray(values, dtype=np.float64)
    if weights.shape != (columns,) or np.any(~np.isfinite(weights)) or np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("Initial weights must be finite, nonnegative, and have positive sum")
    weights = weights / weights.sum()
    # This changes only an optimizer start, not the parameter space or model.
    # It prevents an initial row density of zero; later zero weights are legal
    # and can be revived by the full-gradient vertex direction.
    weights = np.maximum(weights, 1e-12 / columns)
    return weights / weights.sum()


def _duplicate_columns(logs):
    hashes = {}
    for column in range(logs.shape[1]):
        key = hashlib.sha256(np.ascontiguousarray(logs[:, column]).tobytes()).digest()
        hashes.setdefault(key, []).append(column)
    groups = []
    for members in hashes.values():
        if len(members) > 1:
            first = logs[:, members[0]]
            if not all(np.array_equal(first, logs[:, member]) for member in members[1:]):
                raise RuntimeError("Unexpected likelihood-column digest collision")
            groups.append(members)
    return groups


def _vertex_line_search(likelihood, density, gradient, weights):
    """Exact concave one-dimensional search toward the full-gradient maximum."""
    entering = int(np.argmax(gradient))
    direction = likelihood[:, entering] - density
    slope_zero = float(gradient[entering] - len(density))
    if slope_zero <= 0:
        return weights, 0.
    endpoint = likelihood[:, entering]
    if np.all(endpoint > 0) and float(np.sum(direction / endpoint)) >= 0:
        result = np.zeros_like(weights)
        result[entering] = 1.
        return result, 1.
    low, high = 0., 1.
    alpha = .5
    for _ in range(70):
        candidate = density + alpha * direction
        if np.any(candidate <= 0):
            high = alpha
            alpha = (low + high) / 2
            continue
        ratio = direction / candidate
        slope = float(ratio.sum())
        if abs(slope) <= 1e-11 * max(1, len(density)):
            break
        if slope > 0:
            low = alpha
        else:
            high = alpha
        if high - low <= 1e-14:
            alpha = (low + high) / 2
            break
        curvature = float(np.dot(ratio, ratio))
        proposal = alpha + slope / curvature if curvature > 0 else (low + high) / 2
        alpha = proposal if low < proposal < high else (low + high) / 2
    result = (1 - alpha) * weights
    result[entering] += alpha
    return result, alpha


def _fit_one(likelihood, initial, max_iter, tolerance):
    rows = len(likelihood)
    weights = initial.copy()
    if rows == 0:
        return {"weights": weights, "relative_objective": 0., "gap": 0., "iterations": 0,
                "converged": True, "objective_history": [0.], "vertex_steps": 0}

    def evaluate(w):
        density = likelihood @ w
        if np.any(density <= 0) or np.any(~np.isfinite(density)):
            raise FloatingPointError("Mixture reached a nonpositive or nonfinite observation density")
        gradient = likelihood.T @ (1. / density)
        objective = float(np.log(density).sum())
        raw_gap = float(gradient.max() - rows)
        if raw_gap < -1e-10 * rows:
            raise FloatingPointError("Invalid mixture gradient normalization")
        return objective, density, gradient, max(0., raw_gap)

    def em(w, gradient):
        updated = w * (gradient / rows)
        total = updated.sum()
        if not np.isfinite(total) or total <= 0:
            raise FloatingPointError("Invalid EM simplex update")
        return updated / total

    objective, density, gradient, gap = evaluate(weights)
    history = [objective]
    vertex_steps = 0
    iteration = 0
    for iteration in range(1, max_iter + 1):
        if gap / rows <= tolerance:
            iteration -= 1
            break
        previous_objective = objective
        first = em(weights, gradient)
        _, _, gradient_first, _ = evaluate(first)
        second = em(first, gradient_first)
        slow = evaluate(second)
        residual = first - weights
        curvature = second - 2 * first + weights
        denominator = float(np.dot(curvature, curvature))
        alpha = -math.sqrt(float(np.dot(residual, residual)) / max(denominator, np.finfo(float).tiny))
        alpha = min(-1., max(-1000., alpha))
        proposed = weights - 2 * alpha * residual + alpha * alpha * curvature
        proposed = np.maximum(proposed, 0.)
        if proposed.sum() <= 0:
            proposed = second.copy()
        else:
            proposed /= proposed.sum()
        try:
            _, _, gradient_proposed, _ = evaluate(proposed)
            accelerated = em(proposed, gradient_proposed)
            fast = evaluate(accelerated)
        except FloatingPointError:
            fast = (-math.inf, None, None, math.inf)
            accelerated = second
        if fast[0] >= slow[0]:
            weights, (objective, density, gradient, gap) = accelerated, fast
        else:
            weights, (objective, density, gradient, gap) = second, slow
        if gap / rows > tolerance:
            vertex, step = _vertex_line_search(likelihood, density, gradient, weights)
            if step > 0:
                candidate = evaluate(vertex)
                if candidate[0] >= objective:
                    weights, (objective, density, gradient, gap) = vertex, candidate
                    vertex_steps += 1
        allowance = 1e-10 * max(1., abs(previous_objective))
        if objective < previous_objective - allowance:
            raise FloatingPointError("Accelerated mixture update decreased the fixed likelihood")
        history.append(objective)
    return {"weights": weights, "relative_objective": objective, "gap": gap,
            "iterations": iteration, "converged": bool(gap / rows <= tolerance),
            "objective_history": history, "vertex_steps": vertex_steps}


def fit_fixed_likelihood_mixture(log_likelihood, *, group_ids=None, max_iter=1000,
                                 tolerance=1e-7, start_weights=None, n_starts=3,
                                 random_seed=0, max_matrix_cells=100_000_000):
    """Fit simplex weights with a full-column optimality certificate.

    Stopping criterion: Frank--Wolfe gap / number of informative rows <=
    tolerance. Both the full gap (in log-likelihood units) and this scaled gap
    are reported. Empty/constant rows carry no weight-fitting information and
    are never counted as additional effective observations. Their original
    row likelihood constants are preserved in predictive scores/objective.

    Duplicate evidence-unit IDs are rejected, not silently pooled or weighted.
    Exactly identical likelihood columns are reported but never deleted.
    Every result must be checked for convergence; a budget stop is not success.
    """
    logs = _validated(log_likelihood, group_ids)
    if not isinstance(max_iter, Integral) or max_iter < 1 or not isinstance(n_starts, Integral) or n_starts < 1:
        raise ValueError("Positive integer optimization and multistart budgets required")
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("Finite positive tolerance required")
    if logs.size > max_matrix_cells:
        raise ValueError("Exact likelihood-matrix budget exceeded; no rows or columns were pruned")
    rows, columns = logs.shape
    informative = np.any(logs != logs[:, :1], axis=1)
    offsets = np.max(logs, axis=1) if rows else np.empty(0)
    relative = np.exp(logs[informative] - offsets[informative, None])
    relative = np.ascontiguousarray(relative)
    n_informative = int(informative.sum())
    initial = _weights(start_weights, columns)
    rng = np.random.default_rng(random_seed)
    fits = []
    for attempt in range(n_starts):
        weights = initial if attempt == 0 else _weights(rng.lognormal(0., 1., columns), columns)
        fits.append(_fit_one(relative, weights, max_iter, tolerance))
    best_index = max(range(n_starts), key=lambda index: fits[index]["relative_objective"])
    best = fits[best_index]
    offset_sum = float(offsets.sum())
    predictions = []
    for fit in fits:
        predictive = offsets.copy()
        if n_informative:
            predictive[informative] += np.log(relative @ fit["weights"])
        predictions.append(predictive)
    predictive = predictions[best_index]
    starts = [{"start": index, "objective": fit["relative_objective"] + offset_sum,
               "converged": fit["converged"], "iterations": fit["iterations"],
               "frank_wolfe_gap": fit["gap"],
               "gap_per_informative_row": fit["gap"] / max(1, n_informative),
               "vertex_steps": fit["vertex_steps"],
               "objective_history": [value + offset_sum for value in fit["objective_history"]]}
              for index, fit in enumerate(fits)]
    duplicate_groups = _duplicate_columns(logs)
    max_prediction_difference = max((float(np.max(np.abs(value - predictive))) if rows else 0.) for value in predictions)
    max_weight_difference = max(float(np.sum(np.abs(fit["weights"] - best["weights"]))) for fit in fits)
    return {"weights": best["weights"], "log_predictive": predictive,
            "objective": float(predictive.sum()), "converged": best["converged"],
            "frank_wolfe_gap": best["gap"], "gap_per_informative_row": best["gap"] / max(1, n_informative),
            "tolerance": float(tolerance), "stopping_rule": "full Frank-Wolfe gap per informative row <= tolerance",
            "n_rows": rows, "n_informative_rows": n_informative, "n_grid_columns": columns,
            "n_uninformative_rows": rows - n_informative, "selected_start": best_index, "starts": starts,
            "duplicate_likelihood_column_groups": duplicate_groups,
            "max_start_prediction_difference": max_prediction_difference,
            "max_start_weight_l1_difference": max_weight_difference,
            "max_start_weight_difference": max_weight_difference,
            "prior_only_rows": ~informative, "hard_pruned_columns": 0,
            "interpretation": "fixed-grid maximum likelihood; estimated possibly nonunique weights, not a Bayes factor or calibrated posterior",
            "column_underflow_entries": int(np.sum((relative == 0) & np.isfinite(logs[informative]))) }


def posterior_mixture_responsibilities(log_likelihood, weights, *, max_matrix_cells=100_000_000):
    """Return separate learned-prior and likelihood-only/uniform posteriors.

    An uninformative row's learned posterior equals the fitted prior, NOT a
    newly supported individual assignment. Parameter uncertainty is not
    integrated here: these are explicitly plug-in empirical-Bayes quantities.
    """
    logs = _validated(log_likelihood)
    if logs.size > max_matrix_cells:
        raise ValueError("Posterior matrix budget exceeded; request fewer rows explicitly")
    prior = np.asarray(weights, dtype=np.float64)
    if prior.shape != (logs.shape[1],) or np.any(~np.isfinite(prior)) or np.any(prior < 0) or prior.sum() <= 0:
        raise ValueError("One nonnegative finite prior weight per state is required")
    prior = prior / prior.sum()
    log_prior = np.full(len(prior), -np.inf)
    positive = prior > 0
    log_prior[positive] = np.log(prior[positive])
    combined = logs + log_prior[None, :]
    predictive = logsumexp(combined, axis=1)
    if np.any(~np.isfinite(predictive)):
        raise ValueError("Supplied weights make an observation impossible")
    learned = np.exp(combined - predictive[:, None])
    uniform = np.exp(logs - logsumexp(logs, axis=1)[:, None])
    return {"learned_posterior": learned, "uniform_posterior": uniform,
            "prior_weights": prior, "log_predictive": predictive,
            "prior_only_rows": ~np.any(logs != logs[:, :1], axis=1),
            "interpretation": "plug-in empirical-Bayes versus likelihood-only posterior; no fitted-weight uncertainty integrated"}
