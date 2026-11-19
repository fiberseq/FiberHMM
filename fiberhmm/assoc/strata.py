"""Strata for association tests (HUB 7.3, 7.4, 7.8).

- per-molecule **mark density** bins (6 quantile bins by default; NUC 2.1: sparse molecules look closed and
  unbound);
- per-molecule **openness** bins of about ``per_bin`` molecules within each channel (``coaccess.openness_strata``);
- the **channel** (a DAF strand) is always a stratum;
- unknown values (NaN) always get their own bin.

Labels are small integers; :func:`combine` joins several label arrays into one.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

DENSITY_BINS = 6
PER_BIN = 50


def quantile_bins(values, n_bins: int, edges: Optional[np.ndarray] = None) -> np.ndarray:
    """Quantile bin (0..n_bins-1) of every value; NaN -> n_bins. ``edges`` (inner edges) may be given, so bins
    computed on one population apply to another. A value equal to an edge goes to the lower bin, so a discrete
    covariate whose values sit on the edges keeps its levels apart."""
    v = np.asarray(values, dtype=float)
    out = np.full(len(v), n_bins, dtype=np.int64)
    ok = np.isfinite(v)
    if not ok.any() or n_bins <= 1:
        out[ok] = 0
        return out
    if edges is None:
        edges = quantile_edges(v[ok], n_bins)
    out[ok] = np.searchsorted(edges, v[ok], side='left')
    return out


def quantile_edges(values, n_bins: int) -> np.ndarray:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if not len(v) or n_bins <= 1:
        return np.zeros(0)
    return np.unique(np.quantile(v, np.linspace(0, 1, n_bins + 1)[1:-1]))


def density_bins(density, n_bins: int = DENSITY_BINS, edges=None) -> np.ndarray:
    return quantile_bins(density, n_bins, edges)


def openness_bins(openness, channel=None, per_bin: int = PER_BIN, min_bins: int = 5) -> np.ndarray:
    """Openness quantile bins of about ``per_bin`` molecules (at least ``min_bins``) within each channel, as
    ``coaccess.openness_strata`` (NaN openness gets its own bin per channel)."""
    op = np.asarray(openness, dtype=float)
    ch = np.zeros(len(op), dtype=np.int64) if channel is None else np.unique(np.asarray(channel), return_inverse=True)[1]
    out = np.zeros(len(op), dtype=np.int64)
    base = 0
    for c in np.unique(ch):
        m = ch == c
        ok = m & np.isfinite(op)
        nb = max(min_bins, int(ok.sum()) // per_bin)
        lab = np.full(int(m.sum()), nb, dtype=np.int64)
        sub = op[m]
        fin = np.isfinite(sub)
        if fin.any():
            qb = np.quantile(sub[fin], np.linspace(0, 1, nb + 1)[1:-1])
            lab[fin] = np.searchsorted(qb, sub[fin], side='right')
        out[m] = base + lab
        base += nb + 1
    return out


def combine(*labels) -> np.ndarray:
    """One integer label per row for the tuple of label arrays (0..K-1, in sorted tuple order)."""
    labels = [np.asarray(x) for x in labels if x is not None]
    if not labels:
        raise ValueError('no labels')
    n = len(labels[0])
    if not n:
        return np.zeros(0, dtype=np.int64)
    cols = [np.unique(x, return_inverse=True)[1].astype(np.int64) for x in labels]
    key = np.zeros(n, dtype=np.int64)
    for c in cols:
        key = key * (int(c.max()) + 1) + c
    return np.unique(key, return_inverse=True)[1].astype(np.int64)


def bin_mask(spans_lo, spans_hi, bins: Sequence[tuple]) -> np.ndarray:
    """Coverage mask (bit b set when the molecule measures bin b entirely) from each molecule's measured extent
    ``[lo, hi)`` in the analysis axis (for example signed distance from an anchor) and the bins ``[(a, b), ...]``."""
    lo = np.asarray(spans_lo, dtype=float)
    hi = np.asarray(spans_hi, dtype=float)
    out = np.zeros(len(lo), dtype=np.int64)
    for k, (a, b) in enumerate(bins):
        out |= (((lo <= a) & (hi >= b)).astype(np.int64) << k)
    return out


def logistic_fit(X, y, l2: float = 1e-6, max_iter: int = 50):
    """Logistic regression by IRLS (intercept added). Returns coefficients [intercept, ...]."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    Xd = np.column_stack([np.ones(len(y)), X]) if X.size else np.ones((len(y), 1))
    beta = np.zeros(Xd.shape[1])
    for _ in range(max_iter):
        eta = np.clip(Xd @ beta, -30, 30)
        p = 1 / (1 + np.exp(-eta))
        w = p * (1 - p) + 1e-12
        H = (Xd * w[:, None]).T @ Xd + l2 * np.eye(Xd.shape[1])
        g = Xd.T @ (y - p) - l2 * beta
        step = np.linalg.solve(H, g)
        beta = beta + step
        if np.max(np.abs(step)) < 1e-8:
            break
    return beta


def covariate_matrix(columns) -> np.ndarray:
    """Columns of per-molecule covariates (None skipped); NaN imputed by the column median, then standardised."""
    cols = []
    for c in columns:
        if c is None:
            continue
        v = np.asarray(c, dtype=float).copy()
        ok = np.isfinite(v)
        v[~ok] = np.median(v[ok]) if ok.any() else 0.0
        sd = v.std()
        cols.append((v - v.mean()) / sd if sd > 0 else v * 0)
    return np.column_stack(cols) if cols else np.zeros((0, 0))


def adaptive_bins(index, n_per_bin: int, max_bins: int = 10) -> np.ndarray:
    """Equal-count bins of a 1-D index with about ``n_per_bin`` rows each (1..max_bins bins)."""
    v = np.asarray(index, dtype=float)
    k = int(max(1, min(max_bins, len(v) // max(1, n_per_bin))))
    if k == 1:
        return np.zeros(len(v), dtype=np.int64)
    r = np.argsort(np.argsort(v, kind='stable'), kind='stable')
    return (r * k // len(v)).astype(np.int64)
