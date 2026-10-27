"""Conditional logistic regression (exact conditional likelihood per stratum) with a one-step score bootstrap.

Each stratum (a pair of hits, an anchor, a matched set) contributes the probability of its observed set of cases
given the number of cases, so stratum effects (locus prevalence) drop out:

    L_s(beta) = exp(sum_{i in cases} x_i beta) / e_k(exp(x_1 beta), ..., exp(x_n beta))

with e_k the elementary symmetric polynomial of degree k (the number of cases). e_k, its gradient and Hessian come
from one dynamic-programming pass over the stratum's rows (Gail, Lubin and Rubinstein 1981), vectorised over strata.
A stratum with more cases than controls is fitted through its complement (controls as cases, x -> -x), so the pass
needs degree min(k, n - k).

Intervals use a one-step score bootstrap: beta* = beta + H^-1 sum_i w_i u_i, with the per-row score contributions
u_i = (x_i - xbar_s)(y_i - pi_i) (pi_i = the conditional probability that row i is a case; xbar_s the stratum mean,
which leaves every stratum's total unchanged because sum_s (y_i - pi_i) = 0) and bootstrap weights w from
:class:`fiberhmm.assoc.resample.BlockBootstrap`. When the resampling splits strata (whole molecules), each row's
contribution is scaled by sqrt(n_s / (n_s - 1)), the within-stratum correction of the pair-stratified score.
"""
from __future__ import annotations

from typing import Optional

import numpy as np


def _pack(X, y, strata, beta):
    """Informative strata padded into (S, n_max) arrays, with complements applied."""
    order = np.argsort(strata, kind='stable')
    s = strata[order]
    starts = np.flatnonzero(np.r_[True, s[1:] != s[:-1]])
    ends = np.r_[starts[1:], len(s)]
    n = ends - starts
    k = np.add.reduceat(y[order], starts) if len(starts) else np.zeros(0, np.int64)
    keep = (k > 0) & (k < n)
    starts, ends, n, k = starts[keep], ends[keep], n[keep], k[keep]
    flip = k > n - k
    S = len(starts)
    n_max = int(n.max()) if S else 0
    p = X.shape[1]
    Xp = np.zeros((S, n_max, p))
    yp = np.zeros((S, n_max))
    rows = np.full((S, n_max), -1, dtype=np.int64)
    pos = np.arange(n_max)
    for t in range(S):           # one copy per stratum (cheap); the DP below is vectorised
        idx = order[starts[t]:ends[t]]
        sign = -1.0 if flip[t] else 1.0
        Xp[t, :n[t]] = sign * X[idx]
        yp[t, :n[t]] = (1 - y[idx]) if flip[t] else y[idx]
        rows[t, :n[t]] = idx
    kk = np.where(flip, n - k, k)
    valid = pos[None, :] < n[:, None]
    return Xp, yp, rows, valid, kk.astype(np.int64), flip


def _dp(Xp, valid, kk, beta, need_q=True):
    """log-likelihood pieces per stratum: (log e_k, mean of sum_S x, covariance of sum_S x)."""
    S, n_max, p = Xp.shape
    eta = Xp @ beta
    eta = np.where(valid, eta, -np.inf)
    m = np.max(eta, axis=1, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    r = np.where(valid, np.exp(eta - m), 0.0)
    K = int(kk.max()) if S else 0
    e = np.zeros((S, K + 1))
    e[:, 0] = 1.0
    s = np.zeros((S, K + 1, p))
    Q = np.zeros((S, K + 1, p, p)) if need_q else None
    logsc = np.zeros(S)
    above = np.arange(K + 1)[None, :] > kk[:, None]
    for j in range(n_max):
        rj = r[:, j]
        if not np.any(rj):
            continue
        xj = Xp[:, j, :]
        ep, sp = e[:, :-1], s[:, :-1]
        if need_q:
            Qp = Q[:, :-1]
            xs = xj[:, None, :, None] * sp[:, :, None, :]
            Q[:, 1:] = Q[:, 1:] + rj[:, None, None, None] * (
                Qp + xs + np.swapaxes(xs, 2, 3) + (xj[:, :, None] * xj[:, None, :])[:, None] * ep[:, :, None, None])
        s[:, 1:] = s[:, 1:] + rj[:, None, None] * (sp + xj[:, None, :] * ep[:, :, None])
        e[:, 1:] = e[:, 1:] + rj[:, None] * ep
        # coefficients above a stratum's own degree are never needed: zero them, then rescale every stratum's
        # polynomials by its largest needed coefficient (ratios s/e and Q/e are unchanged); scales kept in log
        e[above] = 0.0
        s[above] = 0.0
        if need_q:
            Q[above] = 0.0
        sc = np.max(e, axis=1)
        sc = np.where(sc > 0, sc, 1.0)
        e /= sc[:, None]
        s /= sc[:, None, None]
        if need_q:
            Q /= sc[:, None, None, None]
        logsc += np.log(sc)
    ar = np.arange(S)
    ek = e[ar, kk]
    mean = s[ar, kk] / ek[:, None]
    loge = np.log(ek) + logsc + kk * m[:, 0]
    cov = (Q[ar, kk] / ek[:, None, None] - mean[:, :, None] * mean[:, None, :]) if need_q else None
    return loge, mean, cov, r


def clogit(X, y, strata, *, max_iter: int = 50, tol: float = 1e-8, l2: float = 1e-6,
           beta0: Optional[np.ndarray] = None) -> dict:
    """Fit by Newton-Raphson. ``X`` (n, p) without intercept, ``y`` 0/1, ``strata`` labels. Returns beta, the
    model-based SE, the log-likelihood, H (observed information), the per-row conditional case probabilities
    ``pi`` and score contributions ``u`` (n, p), and the number of informative strata."""
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    y = np.asarray(y, dtype=np.int64)
    strata = np.unique(np.asarray(strata), return_inverse=True)[1]
    p = X.shape[1]
    Xp, yp, rows, valid, kk, flip = _pack(X, y, strata, None)
    beta = np.zeros(p) if beta0 is None else np.asarray(beta0, float).copy()
    if not len(kk):
        return {'beta': beta * np.nan, 'se': beta * np.nan, 'n_strata': 0, 'converged': False,
                'loglik': float('nan')}
    obs = np.einsum('snp,sn->sp', Xp, yp)
    ll_old = -np.inf
    converged = False
    for it in range(max_iter):
        loge, mean, cov, _ = _dp(Xp, valid, kk, beta)
        ll = float(np.sum(obs @ beta - loge)) - 0.5 * l2 * float(beta @ beta)
        g = (obs - mean).sum(0) - l2 * beta
        H = cov.sum(0) + l2 * np.eye(p)
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(H, g, rcond=None)[0]
        # damped step when the likelihood falls
        t = 1.0
        while t > 1e-4:
            nb = beta + t * step
            l2_, _, _, _ = _dp(Xp, valid, kk, nb, need_q=False)
            lln = float(np.sum(obs @ nb - l2_)) - 0.5 * l2 * float(nb @ nb)
            if lln >= ll - 1e-10:
                break
            t /= 2
        beta = beta + t * step
        if np.max(np.abs(t * step)) < tol:
            converged = True
            break
        ll_old = ll
    loge, mean, cov, r = _dp(Xp, valid, kk, beta)
    H = cov.sum(0) + l2 * np.eye(p)
    try:
        Hinv = np.linalg.inv(H)
        se = np.sqrt(np.maximum(np.diag(Hinv), 0))
    except np.linalg.LinAlgError:
        Hinv = np.full((p, p), np.nan)
        se = np.full(p, np.nan)
    pi = _case_probabilities(r, valid, kk)
    # per-row score contributions in the original orientation; x centred within its stratum (stratum totals are
    # unchanged because the residuals y - pi sum to zero in every stratum)
    n = len(y)
    pi_row = np.full(n, np.nan)
    u = np.zeros((n, p))
    u_split = np.zeros((n, p))
    for t_ in range(len(kk)):
        idx = rows[t_][valid[t_]]
        pr = pi[t_][valid[t_]]
        if flip[t_]:
            pr = 1 - pr
        pi_row[idx] = pr
        xc = X[idx] - X[idx].mean(0)
        u[idx] = xc * (y[idx] - pr)[:, None]
        ns = len(idx)
        u_split[idx] = u[idx] * np.sqrt(ns / (ns - 1))
    ll = float(np.sum(obs @ beta - loge))
    return {'beta': beta, 'se': se, 'loglik': ll, 'H': H, 'Hinv': Hinv, 'pi': pi_row, 'u': u, 'u_split': u_split,
            'n_strata': int(len(kk)), 'n_rows': int(valid.sum()), 'converged': converged, 'iterations': it + 1}


def _case_probabilities(r, valid, kk):
    """pi_i = r_i e_{k-1}(r without i) / e_k(r), by forward and backward elementary polynomials (each row rescaled,
    scales kept in log, so large balanced strata do not overflow)."""
    S, n_max = r.shape
    K = int(kk.max()) if S else 0
    F = np.zeros((S, n_max + 1, K + 1))
    lF = np.zeros((S, n_max + 1))
    F[:, 0, 0] = 1.0
    above = np.arange(K + 1)[None, :] > kk[:, None]
    for j in range(n_max):
        row = F[:, j].copy()
        row[:, 1:] += r[:, j, None] * F[:, j, :-1]
        row[above] = 0.0
        sc = row.max(1)
        sc = np.where(sc > 0, sc, 1.0)
        F[:, j + 1] = row / sc[:, None]
        lF[:, j + 1] = lF[:, j] + np.log(sc)
    B = np.zeros((S, n_max + 2, K + 1))
    lB = np.zeros((S, n_max + 2))
    B[:, n_max + 1, 0] = 1.0
    B[:, n_max, 0] = 1.0
    for j in range(n_max - 1, -1, -1):
        row = B[:, j + 2].copy()
        row[:, 1:] += r[:, j, None] * B[:, j + 2, :-1]
        row[above] = 0.0
        sc = row.max(1)
        sc = np.where(sc > 0, sc, 1.0)
        B[:, j + 1] = row / sc[:, None]
        lB[:, j + 1] = lB[:, j + 2] + np.log(sc)
    ar = np.arange(S)
    log_ek = np.log(F[ar, n_max, kk]) + lF[:, n_max]
    pi = np.zeros((S, n_max))
    for j in range(n_max):
        # e_{k-1} without row j = sum_l F[j][l] B[j+2][k-1-l]
        tot = np.zeros(S)
        for l in range(K):
            idx = kk - 1 - l
            ok = idx >= 0
            tot[ok] += F[ok, j, l] * B[ok, j + 2, idx[ok]]
        with np.errstate(divide='ignore'):
            lp = np.log(np.where(r[:, j] > 0, r[:, j], 1.0)) + np.log(np.where(tot > 0, tot, 1.0)) \
                + lF[:, j] + lB[:, j + 2] - log_ek
        pi[:, j] = np.where(valid[:, j] & (tot > 0) & (r[:, j] > 0), np.exp(lp), 0.0)
    return pi


def score_bootstrap(fit: dict, weights_iter, index: int = 0, split_strata: bool = False) -> np.ndarray:
    """One-step bootstrap draws of beta[index]: beta + (H^-1 sum_i w_i u_i)[index] for every weight row.
    ``split_strata``: the weights resample below the stratum (whole molecules, items), so use the corrected
    contributions."""
    u = np.nan_to_num(fit['u_split'] if split_strata else fit['u'])
    hrow = fit['Hinv'][index]
    out = []
    for W in weights_iter:
        g = W @ u              # (draws, p)
        out.append(fit['beta'][index] + g @ hrow)
    return np.concatenate(out) if out else np.zeros(0)
