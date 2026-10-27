"""Recurrence across molecules: a depth-aware enrichment test (NUC 13.1).

Patches of a per-molecule kind (regular, disordered, undersized, missing)
are windows where more molecules carry the kind than chance placement of
each molecule's own calls would give.

**Null (shift within read).** Each molecule's measured bases (aligned, not
low-information) are taken in reference order as one compressed sequence; its
kind indicator over them is shifted circularly by a uniform offset. Each
molecule keeps its own amount and pattern of the kind; the alignment of
calls across molecules is destroyed.

**Statistic.** Windows sit on a fixed genome grid. A molecule enters a window
when at least half of the window's bases are among its measured bases; those
bases are contiguous in its compressed sequence (m of them). Its score is
v = floor(K x c / m), c the kind-covered bases among them (K = 10 levels).
Over all L circular shifts the window's c takes known values, so v has an
exact null distribution per molecule; the window statistic S = sum v has
the exact null distribution of their convolution, and p = P(S >= s). The
observed statistic and the null use the same discretisation, so p is exact
for the discretised statistic.

**FDR.** Benjamini-Hochberg over a contig's windows (overlapping windows are
positively dependent, under which BH holds). Significant windows are merged
into patches.
"""
from __future__ import annotations

from bisect import bisect_left
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

K_LEVELS = 10


class MeasuredTrack:
    """One molecule's measured bases (sorted disjoint reference intervals) and
    a kind indicator over them in compressed coordinates."""

    __slots__ = ('starts', 'ends', 'offsets', 'length', 'x', '_pmf')

    def __init__(self, measured: Sequence[Tuple[int, int]]):
        iv = sorted((int(a), int(b)) for a, b in measured if b > a)
        merged: List[List[int]] = []
        for a, b in iv:
            if merged and a <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], b)
            else:
                merged.append([a, b])
        self.starts = [a for a, _ in merged]
        self.ends = [b for _, b in merged]
        lens = [b - a for a, b in merged]
        self.offsets = [0]
        for n in lens:
            self.offsets.append(self.offsets[-1] + n)
        self.length = self.offsets[-1]
        self.x: Dict[str, np.ndarray] = {}
        self._pmf: Dict[Tuple[str, int], np.ndarray] = {}

    def index(self, pos: int) -> int:
        """Compressed index of the first measured base at or after ``pos``."""
        k = bisect_left(self.ends, pos + 1)          # first interval with end > pos
        if k >= len(self.starts):
            return self.length
        return self.offsets[k] + max(0, pos - self.starts[k])

    def set_kind(self, kind: str, intervals: Sequence[Tuple[int, int]]):
        x = np.zeros(self.length, dtype=np.int32)
        for a, b in intervals:
            i, j = self.index(int(a)), self.index(int(b))
            if j > i:
                x[i:j] = 1
        self.x[kind] = x

    def window(self, w0: int, w1: int) -> Tuple[int, int]:
        return self.index(w0), self.index(w1)

    def pmf(self, kind: str, m: int, K: int = K_LEVELS) -> np.ndarray:
        """Null distribution of v = floor(K c / m) over every circular shift."""
        key = (kind, m)
        got = self._pmf.get(key)
        if got is not None:
            return got
        x = self.x[kind]
        L = self.length
        if x.sum() == 0:
            out = np.zeros(K + 1)
            out[0] = 1.0
        elif x.sum() == L:
            out = np.zeros(K + 1)
            out[K] = 1.0
        else:
            reps = -(-m // L) + 1
            xx = np.tile(x, reps)[:L + m]
            cs = np.concatenate([[0], np.cumsum(xx)])
            sums = cs[m:m + L] - cs[:L]
            v = np.floor(K * sums / m + 1e-9).astype(np.int64)
            out = np.bincount(np.clip(v, 0, K), minlength=K + 1) / float(L)
        self._pmf[key] = out
        return out


def window_test(tracks: Sequence[MeasuredTrack], kind: str, w0: int, w1: int,
                K: int = K_LEVELS, min_molecules: int = 3) -> Optional[dict]:
    """Exact shift-null test of one window: observed and expected mean
    fraction, molecule count and p (None with fewer than ``min_molecules``)."""
    W = w1 - w0
    obs = 0
    pmfs = []
    for t in tracks:
        if kind not in t.x:
            continue
        i, j = t.window(w0, w1)
        m = j - i
        if 2 * m < W or m <= 0:
            continue
        c = int(t.x[kind][i:j].sum())
        obs += int(np.floor(K * c / m + 1e-9))
        pmfs.append(t.pmf(kind, m, K))
    n = len(pmfs)
    if n < min_molecules:
        return None
    exp = float(sum(float(np.dot(np.arange(K + 1), p)) for p in pmfs))
    if obs == 0:
        p_val = 1.0                    # P(S >= 0) = 1
    else:
        dist = np.ones(1)
        for p in pmfs:
            dist = np.convolve(dist, p)
        p_val = float(min(1.0, dist[obs:].sum()))
    return {'n': n, 'obs': obs / (K * n), 'exp': exp / (K * n), 'p': p_val}


def bh(pvals: Sequence[float], q: float) -> np.ndarray:
    """Benjamini-Hochberg rejections at FDR ``q`` (NaN never rejected)."""
    p = np.asarray(pvals, dtype=float)
    ok = np.isfinite(p)
    out = np.zeros(len(p), dtype=bool)
    m = int(ok.sum())
    if m == 0:
        return out
    idx = np.flatnonzero(ok)
    order = idx[np.argsort(p[idx], kind='stable')]
    thresh = q * np.arange(1, m + 1) / m
    passed = np.flatnonzero(p[order] <= thresh)
    if len(passed):
        out[order[:passed[-1] + 1]] = True
    return out


def bh_threshold(small_pvals: Sequence[float], m: int, q: float) -> Optional[float]:
    """The Benjamini-Hochberg rejection threshold at FDR ``q`` over ``m`` tests, from the p-values <= ``q``
    only (no larger p-value can be rejected, and they all rank after these): the largest p_(k) with
    p_(k) <= k q / m, or None when nothing is rejected. Rejecting every p <= the threshold gives exactly
    :func:`bh`'s rejections over the ``m`` p-values."""
    p = np.sort(np.asarray([x for x in small_pvals if np.isfinite(x) and x <= q], dtype=float), kind='stable')
    if m <= 0 or not len(p):
        return None
    passed = np.flatnonzero(p <= q * np.arange(1, len(p) + 1) / m)
    return float(p[passed[-1]]) if len(passed) else None


def grid_windows(z0: int, z1: int, step: int) -> List[int]:
    """Window starts on the fixed genome grid that fall in [z0, z1)."""
    return list(range((z0 + step - 1) // step * step, z1, step))


def merge_windows(starts: Sequence[int], W: int) -> List[Tuple[int, int]]:
    """Union of [s, s + W) for the given starts."""
    out: List[List[int]] = []
    for s in sorted(starts):
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], s + W)
        else:
            out.append([s, s + W])
    return [(a, b) for a, b in out]
