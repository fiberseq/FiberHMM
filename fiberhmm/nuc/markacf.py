"""Caller-free spacing check: autocorrelation of mark positions (NUC 2.2).

No footprint calls are used. Marks are centred on informative sites (each
informative site gets the molecule's mean mark rate subtracted, so base
composition does not create structure), the raw autocovariances are summed
over molecules, normalised by lag 0, and detrended with a running mean. The
peak lag in ``lag_range`` is the repeat length seen directly in the marks.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np
from scipy.fft import irfft, next_fast_len, rfft

from fiberhmm.nuc.records import NucMolecule


@dataclass
class MarkACF:
    acf: np.ndarray
    detrended: np.ndarray
    peak_lag: int
    n_molecules: int

    def as_dict(self):
        return {'peak_lag': int(self.peak_lag), 'n_molecules': int(self.n_molecules)}


def mark_acf(molecules: Iterable[NucMolecule], max_lag: int = 600,
             min_marks_per_bp: float = 0.015, max_molecules: int = 1500,
             lag_range: Tuple[int, int] = (150, 260),
             detrend_bp: int = 121) -> MarkACF:
    """Pooled mark autocorrelation over the first ``max_molecules`` molecules
    with at least ``min_marks_per_bp`` marks per bp (dense molecules carry
    the signal)."""
    H = np.zeros(max_lag + 1)
    n = 0
    for mol in molecules:
        L = mol.length
        x = mol.hits.astype(float)
        if x.sum() < min_marks_per_bp * L:
            continue
        inf = mol.informative.astype(float)
        if inf.sum() == 0:
            continue
        xc = x - inf * (x.sum() / inf.sum())
        m = next_fast_len(L + max_lag + 1, real=True)
        f = rfft(xc, m)
        H += irfft(f * np.conj(f), m)[:max_lag + 1]
        n += 1
        if n >= max_molecules:
            break
    if n == 0 or H[0] == 0:
        z = np.zeros(max_lag + 1)
        return MarkACF(z, z, -1, 0)
    H = H / H[0]
    base = np.convolve(H, np.ones(detrend_bp) / detrend_bp, 'same')
    d = H - base
    lo, hi = lag_range
    lag = int(np.argmax(d[lo:hi + 1]) + lo)
    return MarkACF(H, d, lag, n)
