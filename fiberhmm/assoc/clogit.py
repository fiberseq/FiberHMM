"""Conditional logistic regression (exact conditional likelihood per stratum) with a one-step score bootstrap.

Each stratum (a pair of hits, an anchor, a matched set) contributes the probability of its observed set of cases
given the number of cases, so stratum effects (locus prevalence) drop out:

    L_s(beta) = exp(sum_{i in cases} x_i beta) / e_k(exp(x_1 beta), ..., exp(x_n beta))

with e_k the elementary symmetric polynomial of degree k (the number of cases). e_k, its gradient and Hessian come
from one dynamic-programming pass over the stratum's rows (Gail, Lubin and Rubinstein 1981), vectorised over strata.
A stratum with more cases than controls is fitted through its complement (controls as cases, x -> -x), so the pass
needs degree min(k, n - k).

Each stratum's e_k, gradient and Hessian come from one of two exact routes (``fiberhmm.assoc._clogit_kernels``):
the DP above, one stratum at a time, or, for strata with many cases (pairs of hits with thousands of molecules),
the inverse DFT of the tilted Poisson-binomial characteristic function, which needs O(n sqrt(n) p) work instead of
O(n k p^2). Both give the same numbers to floating-point rounding (tests/test_assoc.py).

Intervals use a one-step score bootstrap: beta* = beta + H^-1 sum_i w_i u_i, with the per-row score contributions
u_i = (x_i - xbar_s)(y_i - pi_i) (pi_i = the conditional probability that row i is a case; xbar_s the stratum mean,
which leaves every stratum's total unchanged because sum_s (y_i - pi_i) = 0) and bootstrap weights w from
:class:`fiberhmm.assoc.resample.BlockBootstrap`. When the resampling splits strata (whole molecules), each row's
contribution is scaled by n_s / (n_s - 1): the residual against the stratum's other rows (leave-one-out), so a
molecule's own outcome does not shrink its residual.
"""
from __future__ import annotations

import contextlib
import threading
from typing import Optional

import numpy as np

_BLAS_LOCK = threading.Lock()
_BLAS_DEPTH = 0
_BLAS_LIMIT = None


@contextlib.contextmanager
def single_thread_blas():
    """One BLAS thread while any fit runs (the per-stratum products are small: threaded BLAS spends far longer waking
    threads). threadpoolctl's limit is process-wide and restores the value it saw on entry, so overlapping fits in
    several threads would restore in the wrong order and leave the process at one thread: the limit is reference
    counted instead. The first fit to enter applies it, the last to leave restores what the process had."""
    global _BLAS_DEPTH, _BLAS_LIMIT
    with _BLAS_LOCK:
        if _BLAS_DEPTH == 0:
            try:
                from threadpoolctl import threadpool_limits
                _BLAS_LIMIT = threadpool_limits(limits=1, user_api='blas')
            except Exception:   # noqa: BLE001 - optional at run time
                _BLAS_LIMIT = None
        _BLAS_DEPTH += 1
    try:
        yield
    finally:
        with _BLAS_LOCK:
            _BLAS_DEPTH -= 1
            if _BLAS_DEPTH == 0 and _BLAS_LIMIT is not None:
                try:
                    _BLAS_LIMIT.restore_original_limits()
                finally:
                    _BLAS_LIMIT = None


def _pack(X, y, strata):
    """Informative strata as contiguous row blocks (``offs``), with complements applied: rows (original indices),
    oriented X and y, the degree k (= min(cases, controls)) and whether the stratum was flipped."""
    order = np.argsort(strata, kind='stable')
    s = strata[order]
    starts = np.flatnonzero(np.r_[True, s[1:] != s[:-1]]) if len(s) else np.zeros(0, np.int64)
    ends = np.r_[starts[1:], len(s)].astype(np.int64)
    n = ends - starts
    k = np.add.reduceat(y[order], starts) if len(starts) else np.zeros(0, np.int64)
    keep = (k > 0) & (k < n)
    n, k = n[keep], k[keep]
    flip = k > n - k
    rows = np.concatenate([order[a:b] for a, b in zip(starts[keep], ends[keep])]) if keep.any() \
        else np.zeros(0, np.int64)
    offs = np.r_[0, np.cumsum(n)].astype(np.int64)
    fr = np.repeat(flip, n)
    Xf = np.ascontiguousarray(np.where(fr[:, None], -X[rows], X[rows]))
    yf = np.where(fr, 1 - y[rows], y[rows]).astype(float)
    kk = np.where(flip, n - k, k).astype(np.int64)
    return Xf, yf, rows.astype(np.int64), offs, kk, flip


# DFT cut: frequencies whose terms are all below exp(-(T0 + T1 log N)) are left out (module _clogit_kernels)
_DFT_T0, _DFT_T1 = 40.0, 3.0
# strata checked for a cancel between blocks of about this many rows
_CANCEL_ROWS = 50_000


#: threads for the DFT strata (FIBERHMM_ASSOC_THREADS; default min(4, CPUs))
_POOLS = {}


def _dft_workers(n_strata: int) -> int:
    import os
    try:
        w = int(os.environ.get('FIBERHMM_ASSOC_THREADS', '0'))
    except ValueError:
        w = 0
    if w <= 0:
        w = min(4, os.cpu_count() or 1)
    return max(1, min(w, n_strata))


def _pool(workers: int):
    from concurrent.futures import ThreadPoolExecutor
    pool = _POOLS.get(workers)
    if pool is None:
        pool = _POOLS[workers] = ThreadPoolExecutor(workers, thread_name_prefix='fiberhmm-clogit')
    return pool


def _moments(Xf, offs, kk, beta, need_q=True, need_pi=False):
    """Per stratum: log e_k, the conditional mean of sum_cases x and (need_q) its covariance; (need_pi) the case
    probabilities of every row. Each stratum takes the DFT route when its band is narrow enough, else the DP."""
    from . import _clogit_kernels as K
    from .control import cancelled
    S = len(kk)
    p = Xf.shape[1]
    eta = Xf @ beta
    a = np.zeros(S)
    M = np.zeros(S, np.int64)
    K.tilt(eta, offs, kk, _DFT_T0, _DFT_T1, a, M)
    loge = np.zeros(S)
    mean = np.zeros((S, p))
    cov = np.zeros((S, p, p)) if need_q else np.zeros((1, 1, 1))
    pi = np.zeros(len(eta)) if need_pi else None
    small = np.flatnonzero(M < 0)
    # DP strata in blocks of about _CANCEL_ROWS rows, with a cancel check between blocks
    if len(small):
        sizes = (offs[small + 1] - offs[small]) * (kk[small] + 1)
        blocks = np.cumsum(sizes) // max(1, _CANCEL_ROWS * 50)
        for b in np.unique(blocks):
            if cancelled():
                from .control import AssociationCancelled
                raise AssociationCancelled('association cancelled')
            which = small[blocks == b].astype(np.int64)
            K.dp_strata(Xf, eta, offs, kk, which, need_q, loge, mean, cov)
            if need_pi:
                K.dp_case_probabilities(eta, offs, kk, which, pi)
    big = np.flatnonzero(M >= 0)
    want = need_q or need_pi

    def one(t):
        a0, a1 = int(offs[t]), int(offs[t + 1])
        Mt = int(M[t])
        shape = (a1 - a0, Mt + 1) if want else (0, 0)
        Qr, Qi = np.empty(shape), np.empty(shape)
        le, pr, wv, c = K.dft_stratum(eta, a0, a1, int(kk[t]), float(a[t]), Mt, want, Qr, Qi)
        loge[t] = le
        if not want:
            return
        x = Xf[a0:a1]
        xbar = x.mean(0)
        xc = x - xbar
        mc = xc.T @ pr                                    # mean of sum_cases (x - xbar)
        mean[t] = mc + kk[t] * xbar
        if need_pi:
            pi[a0:a1] = pr
        if need_q:
            # g_m of the centred rows, (p, M + 1); Re sum_m c_m g_m g_m' in real arithmetic
            Gr, Gi = xc.T @ Qr, xc.T @ Qi
            cr, ci = c.real, c.imag
            M2 = (Gr * cr) @ Gr.T - (Gi * cr) @ Gi.T - (Gr * ci) @ Gi.T - (Gi * ci) @ Gr.T
            M2 += (xc * wv[:, None]).T @ xc
            cov[t] = M2 - np.outer(mc, mc)

    # strata are independent (each writes its own slots): a few threads (the kernels release the GIL), with a
    # cancel check between blocks of about _CANCEL_ROWS rows
    rows = np.diff(offs)[big]
    blocks = np.cumsum(rows) // _CANCEL_ROWS
    workers = _dft_workers(len(big))
    pool = _pool(workers) if workers > 1 else None
    for b in np.unique(blocks):
        if cancelled():
            from .control import AssociationCancelled
            raise AssociationCancelled('association cancelled')
        part = big[blocks == b]
        if pool is None:
            for t in part:
                one(t)
        else:
            list(pool.map(one, part))
    return loge, mean, (cov if need_q else None), pi


def clogit(X, y, strata, *, max_iter: int = 50, tol: float = 1e-8, l2: float = 1e-6,
           beta0: Optional[np.ndarray] = None) -> dict:
    """Fit by Newton-Raphson. ``X`` (n, p) without intercept, ``y`` 0/1, ``strata`` labels. Returns beta, the
    model-based SE, the log-likelihood, H (observed information), the per-row conditional case probabilities
    ``pi`` and score contributions ``u`` (n, p), and the number of informative strata."""
    with single_thread_blas():
        return _clogit(X, y, strata, max_iter=max_iter, tol=tol, l2=l2, beta0=beta0)


def _clogit(X, y, strata, *, max_iter, tol, l2, beta0):
    from .control import checkpoint
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    y = np.asarray(y, dtype=np.int64)
    strata = np.unique(np.asarray(strata), return_inverse=True)[1]
    p = X.shape[1]
    Xf, yf, rows, offs, kk, flip = _pack(X, y, strata)
    beta = np.zeros(p) if beta0 is None else np.asarray(beta0, float).copy()
    if not len(kk):
        return {'beta': beta * np.nan, 'se': beta * np.nan, 'n_strata': 0, 'converged': False,
                'loglik': float('nan')}
    sid = np.repeat(np.arange(len(kk)), np.diff(offs))
    obs = np.zeros((len(kk), p))
    np.add.at(obs, sid, Xf * yf[:, None])
    ll_old = -np.inf
    converged = False
    for it in range(max_iter):
        checkpoint('clogit', it, max_iter)
        loge, mean, cov, _ = _moments(Xf, offs, kk, beta)
        ll = float(np.sum(obs @ beta - loge)) - 0.5 * l2 * float(beta @ beta)
        g = (obs - mean).sum(0) - l2 * beta
        H = cov.sum(0) + l2 * np.eye(p)
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(H, g, rcond=None)[0]
        # damped step when the likelihood falls; a step whose likelihood is not finite (e_k under- or overflowing
        # on a nearly separated stratum) is a failed step, never an infinitely good one
        if not (np.isfinite(ll) and np.all(np.isfinite(step))):
            break
        t = 1.0
        accepted = False
        while t > 1e-4:
            nb = beta + t * step
            l2_, _, _, _ = _moments(Xf, offs, kk, nb, need_q=False)
            lln = float(np.sum(obs @ nb - l2_)) - 0.5 * l2 * float(nb @ nb)
            if np.isfinite(lln) and lln >= ll - 1e-10:
                accepted = True
                break
            t /= 2
        if not accepted:
            # no improvement along the step: a tiny step as before, but only to a point with a finite likelihood
            l2_, _, _, _ = _moments(Xf, offs, kk, beta + t * step, need_q=False)
            if not np.all(np.isfinite(l2_)):
                break                   # keep the last finite beta
        beta = beta + t * step
        if np.max(np.abs(t * step)) < tol:
            converged = True
            break
        ll_old = ll
    checkpoint('clogit', max_iter, max_iter)
    loge, mean, cov, pi = _moments(Xf, offs, kk, beta, need_pi=True)
    H = cov.sum(0) + l2 * np.eye(p)
    try:
        Hinv = np.linalg.inv(H)
        se = np.sqrt(np.maximum(np.diag(Hinv), 0))
    except np.linalg.LinAlgError:
        Hinv = np.full((p, p), np.nan)
        se = np.full(p, np.nan)
    # per-row score contributions in the original orientation; x centred within its stratum (stratum totals are
    # unchanged because the residuals y - pi sum to zero in every stratum)
    n = len(y)
    fr = np.repeat(flip, np.diff(offs))
    pr = np.where(fr, 1 - pi, pi)
    pi_row = np.full(n, np.nan)
    pi_row[rows] = pr
    ns = np.diff(offs).astype(float)
    xr = X[rows]
    xsum = np.zeros((len(kk), p))
    np.add.at(xsum, sid, xr)
    xc = xr - (xsum / ns[:, None])[sid]
    u = np.zeros((n, p))
    u_split = np.zeros((n, p))
    u[rows] = xc * (y[rows] - pr)[:, None]
    u_split[rows] = u[rows] * (ns / (ns - 1))[sid][:, None]          # leave-molecule-out residual scale
    ll = float(np.sum(obs @ beta - loge))
    return {'beta': beta, 'se': se, 'loglik': ll, 'H': H, 'Hinv': Hinv, 'pi': pi_row, 'u': u, 'u_split': u_split,
            'n_strata': int(len(kk)), 'n_rows': int(len(rows)), 'converged': converged, 'iterations': it + 1}


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
