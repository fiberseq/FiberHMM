"""Experimental fixed-geometry hierarchical state kernel, not production CR.

The observation domain/base measure is shared across all states. This batch
fast path requires ONE fixed half-open interval per state and additive signed
likelihood ratios. Uncertain interacting geometry belongs in state_geometry.
Activities are fitted with the hard-core prior normalizer, never equated with
logit marginal prevalence. Folds/nomination are the adapter's responsibility.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np
from scipy.optimize import minimize


@dataclass(frozen=True)
class StateInterval:
    state_id: str
    start: int
    end: int


@dataclass
class BatchPosterior:
    log_z: np.ndarray
    marginals: np.ndarray


class FixedStateCatalog:
    """Compiled interval schedule, retaining caller-specified column order."""
    def __init__(self, states: Sequence[StateInterval]):
        self.states = tuple(states)
        self.n = len(states)
        if len({s.state_id for s in states}) != self.n:
            raise ValueError("Duplicate state IDs")
        if any(not s.state_id or s.end <= s.start for s in states):
            raise ValueError("Invalid state interval")
        self.ends_order = np.array(sorted(range(self.n), key=lambda i: (states[i].end, states[i].start, states[i].state_id)), dtype=int)
        self.starts_order = np.array(sorted(range(self.n), key=lambda i: (states[i].start, states[i].end, states[i].state_id)), dtype=int)
        self.starts = np.array([s.start for s in states], dtype=np.int64)
        self.ends = np.array([s.end for s in states], dtype=np.int64)
        self.left = np.searchsorted(self.ends[self.ends_order], self.starts, side="right")
        self.right = np.searchsorted(self.starts[self.starts_order], self.ends, side="left")

    def infer(self, log_weights) -> BatchPosterior:
        weights = np.asarray(log_weights, dtype=float)
        if weights.ndim == 1:
            weights = weights[None, :]
        if weights.ndim != 2 or weights.shape[1] != self.n:
            raise ValueError("Expected units-by-states log weights")
        if np.any(np.isnan(weights)) or np.any(np.isposinf(weights)):
            raise ValueError("Weights must be finite or negative infinity")
        m = len(weights)
        forward = np.zeros((self.n + 1, m), dtype=float)
        backward = np.zeros_like(forward)
        for k, j in enumerate(self.ends_order):
            forward[k + 1] = np.logaddexp(forward[k], weights[:, j] + forward[self.left[j]])
        for k in range(self.n - 1, -1, -1):
            j = self.starts_order[k]
            backward[k] = np.logaddexp(backward[k + 1], weights[:, j] + backward[self.right[j]])
        log_z = forward[-1]
        log_include = weights.T + forward[self.left] + backward[self.right]
        marginals = np.exp(log_include - log_z).T
        return BatchPosterior(log_z, np.minimum(marginals, 1.0))

    def map_indices(self, log_weights, forbidden=()) -> tuple[int, ...]:
        weights = np.asarray(log_weights, dtype=float)
        if weights.shape != (self.n,) or np.any(np.isnan(weights)) or np.any(np.isposinf(weights)):
            raise ValueError("Invalid MAP weights")
        blocked = set(forbidden)
        if any(i < 0 or i >= self.n for i in blocked):
            raise ValueError("Unknown forbidden index")
        values = np.zeros(self.n + 1)
        configs = [()] * (self.n + 1)
        for k, j in enumerate(self.ends_order):
            included = -math.inf if j in blocked else weights[j] + values[self.left[j]]
            proposal = tuple(sorted((*configs[self.left[j]], int(j)), key=lambda i: self.states[i].state_id))
            skipped = values[k]
            proposal_key = (len(proposal), tuple(self.states[i].state_id for i in proposal))
            skipped_key = (len(configs[k]), tuple(self.states[i].state_id for i in configs[k]))
            if included > skipped or (included == skipped and proposal_key < skipped_key):
                values[k + 1], configs[k + 1] = included, proposal
            else:
                values[k + 1], configs[k + 1] = skipped, configs[k]
        return configs[-1]

    def fit_activities(self, log_evidence, *, regularization=0.5, max_iterations=100) -> dict:
        """Normalized observed likelihood with weak Gaussian activity shrinkage.

        This is an empirical training fit, not a calibrated population-existence
        test. Candidate count, regularization and convergence remain explicit.
        No clipping/deleting of negative molecule evidence is performed.
        """
        evidence = np.asarray(log_evidence, dtype=float)
        if evidence.ndim != 2 or evidence.shape[1] != self.n or not len(evidence):
            raise ValueError("Need nonempty training units and matching states")
        if not np.all(np.isfinite(evidence)):
            raise ValueError("Training evidence must be finite")
        if not math.isfinite(regularization) or regularization <= 0:
            raise ValueError("Positive regularization is required")
        # Number of direct competitors, not the whole amplicon's candidate count.
        competitors = np.array([np.count_nonzero((self.starts < s.end) & (self.ends > s.start)) for s in self.states])
        center = -np.log(np.maximum(competitors, 1)) - math.log(4.0)
        if self.n == 0:
            return {"activities": [], "prior_marginals": [], "converged": True, "iterations": 0,
                    "log_likelihood_ratio": 0.0, "regularization": regularization}
        def objective(eta):
            obs, prior = self.infer(evidence + eta), self.infer(eta)
            ll = float(np.sum(obs.log_z) - len(evidence)*prior.log_z[0])
            delta = eta-center
            value = -ll + .5*regularization*float(delta @ delta)
            gradient = -obs.marginals.sum(axis=0) + len(evidence)*prior.marginals[0] + regularization*delta
            return value, gradient
        result = minimize(objective, center, jac=True, method="L-BFGS-B",
                          options={"maxiter": max_iterations, "ftol": 1e-10, "gtol": 1e-6, "maxls": 30})
        prior, obs = self.infer(result.x), self.infer(evidence + result.x)
        return {"activities": result.x.tolist(), "prior_marginals": prior.marginals[0].tolist(),
                "converged": bool(result.success), "message": str(result.message), "iterations": int(result.nit),
                "log_likelihood_ratio": float(np.sum(obs.log_z) - len(evidence)*prior.log_z[0]),
                "regularization": regularization, "activity_prior_center": center.tolist(),
                "objective": float(result.fun), "max_abs_gradient": float(np.max(np.abs(result.jac)))}


def interval_evidence(positions, hits, p_accessible, p_protected, starts, ends) -> dict:
    """Exact Bernoulli likelihood on a unit's native callable lattice.

    Outside each candidate interval the common accessible base measure cancels.
    Absence of callable opportunities is zero evidence, not an accessible call.
    All per-opportunity probabilities must already reflect the frozen forward
    model; no local-flank efficiency is estimated here.
    """
    pos = np.asarray(positions, dtype=np.int64)
    y = np.asarray(hits)
    pa, pp = np.asarray(p_accessible, dtype=float), np.asarray(p_protected, dtype=float)
    lo, hi = np.asarray(starts), np.asarray(ends)
    if any(x.shape != pos.shape for x in (y, pa, pp)) or pos.ndim != 1 or np.any(np.diff(pos) <= 0):
        raise ValueError("Opportunity arrays must align and positions be unique/increasing")
    if np.any((y != 0) & (y != 1)) or np.any(~np.isfinite(pa)) or np.any(~np.isfinite(pp)):
        raise ValueError("Expected binary observations and finite probabilities")
    if np.any((pa <= 0) | (pa >= 1) | (pp <= 0) | (pp >= 1)):
        raise ValueError("Forward probabilities must be strictly between zero and one")
    if lo.shape != hi.shape or np.any(hi <= lo):
        raise ValueError("Invalid candidate intervals")
    hit_lr = np.log(pp/pa)
    miss_lr = np.log((1-pp)/(1-pa))
    steps = np.where(y, hit_lr, miss_lr)
    max_steps = np.maximum(hit_lr, miss_lr)
    kl = pp*hit_lr + (1-pp)*miss_lr
    informative = np.abs(pa-pp) > 1e-12
    left, right = np.searchsorted(pos, lo), np.searchsorted(pos, hi)
    def sums(values):
        prefix = np.r_[0., np.cumsum(values)]
        return prefix[right]-prefix[left]
    return {"log_likelihood_ratio": sums(steps), "opportunities": right-left,
            "informative_opportunities": sums(informative).astype(int), "hits": sums(y).astype(int),
            "maximum_attainable_log_ratio": sums(max_steps), "expected_protected_information": sums(kl)}


def posterior_quality(probability):
    """Posterior error Phred scale. Not an empirical FDR q-value."""
    p = np.asarray(probability, dtype=float)
    if np.any(~np.isfinite(p)) or np.any((p < 0) | (p > 1)):
        raise ValueError("Invalid posterior probability")
    return -10*np.log10(np.maximum(1-p, 1e-12))
