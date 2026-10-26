"""Per-molecule spacing and regularity (NUC 3.4).

**Eligible dyads**, one rule for every statistic: centres of coarse-mono
footprints when coarse calls exist; otherwise centres of footprints of
``mono_frac`` x NRL_pooled (0.45-1.25 x); edge-truncated footprints are never
eligible. Before NRL_pooled is known (a fresh dataset) the bootstrap range
``bootstrap_mono_bp`` (90-230 bp) gives a first P_dataset, the range is
redefined from it and the statistics are computed once more
(:func:`bootstrap_nrl_pooled`).

**Gaps** are between eligible dyads whose footprints are adjacent in the
molecule's nucleosome list (no other nucleosome call between them) and that
are not separated by an MSP longer than ``long_msp_bp``. Multi(m) footprints
add virtual dyads for segmentation only; a gap that spans one is excluded
here by the adjacency rule.

**Regularity** S = rho(tau*) - rho(tau*/2) on the dyad indicator (1 within
+/-``acf_halfwidth`` bp of an eligible dyad, mean-centred over the window),
with tau* the ACF maximum over ``acf_lag``; biased FFT autocorrelation,
normalised by lag 0. **Spectral SNR** is the periodogram power at 1/tau*
over the median power across ``snr_band_bp``. **CV_g** = SD(g)/mean(g).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

from fiberhmm.nuc.density import edge_truncated
from fiberhmm.nuc.params import DEFAULT_PARAMS, NucParams
from fiberhmm.nuc.records import NucMolecule


def mono_range(nrl_pooled: Optional[float], params: NucParams = DEFAULT_PARAMS) -> Tuple[float, float]:
    if nrl_pooled is None or not np.isfinite(nrl_pooled):
        return float(params.bootstrap_mono_bp[0]), float(params.bootstrap_mono_bp[1])
    return params.mono_frac[0] * nrl_pooled, params.mono_frac[1] * nrl_pooled


@dataclass
class DyadSet:
    """Eligible dyads of one molecule (molecular bp) and their gaps."""
    centres: np.ndarray            # eligible dyad positions, sorted
    index: np.ndarray              # index of each eligible footprint in mol.nucs
    gaps: np.ndarray               # valid adjacent gaps
    gap_left: np.ndarray           # left dyad position of each valid gap
    eligible_mask: np.ndarray      # per mol.nucs footprint


def eligible_dyads(mol: NucMolecule, nrl_pooled: Optional[float] = None,
                   params: NucParams = DEFAULT_PARAMS,
                   coarse_codes: Optional[np.ndarray] = None,
                   window: Optional[Tuple[int, int]] = None) -> DyadSet:
    """Eligible dyads, optionally restricted to footprints centred in
    molecular ``window``.

    ``coarse_codes`` (per nucleosome call; 2 = mono) replaces the size rule
    when coarse calls are available.
    """
    nucs = mol.nucs
    n = len(nucs)
    if n == 0:
        e = np.zeros(0)
        return DyadSet(e, np.zeros(0, dtype=np.int64), e, e, np.zeros(0, dtype=bool))
    lengths = nucs[:, 1] - nucs[:, 0]
    centres = (nucs[:, 0] + nucs[:, 1]) / 2.0
    trunc = edge_truncated(mol, nucs)
    if coarse_codes is not None:
        mono = np.asarray(coarse_codes) == 2
    else:
        lo, hi = mono_range(nrl_pooled, params)
        mono = (lengths >= lo) & (lengths <= hi)
    elig = mono & ~trunc
    if window is not None:
        elig &= (centres >= window[0]) & (centres < window[1])
    idx = np.flatnonzero(elig)
    c = centres[idx]
    # adjacency: consecutive eligible footprints that are also consecutive calls
    if len(idx) >= 2:
        adj = np.diff(idx) == 1
        g = np.diff(c)
        left = c[:-1]
        if len(mol.msps) and adj.any():
            long_msp = mol.msps[(mol.msps[:, 1] - mol.msps[:, 0]) > params.long_msp_bp]
            if len(long_msp):
                a = nucs[idx[:-1], 1]
                b = nucs[idx[1:], 0]
                for ms, me in long_msp:
                    adj &= ~((ms >= a - 1) & (me <= b + 1))
        gaps = g[adj]
        gap_left = left[adj]
    else:
        gaps = np.zeros(0)
        gap_left = np.zeros(0)
    return DyadSet(c, idx, gaps, gap_left, elig)


def _biased_acf(x: np.ndarray, max_lag: int) -> np.ndarray:
    """Biased autocorrelation for lags 0..max_lag, normalised by lag 0
    (zero padding to a fast length >= n + max_lag avoids wrap-around)."""
    from scipy.fft import irfft, next_fast_len, rfft
    n = len(x)
    m = next_fast_len(n + max_lag + 1, real=True)
    f = rfft(x, m)
    ac = irfft(f * np.conj(f), m)[:max_lag + 1]
    if len(ac) < max_lag + 1:
        ac = np.concatenate([ac, np.zeros(max_lag + 1 - len(ac))])
    return ac / ac[0] if ac[0] > 0 else np.zeros(max_lag + 1)


def dyad_indicator(centres: np.ndarray, start: int, end: int, halfwidth: int) -> np.ndarray:
    """1 within +/-``halfwidth`` bp (half-open) of each dyad, over [start, end)."""
    n = max(int(end - start), 0)
    if n == 0 or len(centres) == 0:
        return np.zeros(n)
    c = np.round(np.asarray(centres, dtype=float)).astype(np.int64) - start
    a = np.clip(c - halfwidth, 0, n)
    b = np.clip(c + halfwidth, 0, n)
    ok = b > a
    d = np.zeros(n + 1, dtype=np.int64)
    np.add.at(d, a[ok], 1)
    np.add.at(d, b[ok], -1)
    return (np.cumsum(d[:-1]) > 0).astype(float)


@dataclass
class SpacingMetrics:
    nrl: float = float('nan')
    regularity: float = float('nan')
    tau: float = float('nan')
    spectral_snr: float = float('nan')
    gap_cv: float = float('nan')
    n_dyads: int = 0
    n_gaps: int = 0
    too_short_nrl: bool = True
    too_short_regularity: bool = True
    too_short_snr: bool = True
    acf: Optional[np.ndarray] = field(default=None, repr=False)

    def as_dict(self) -> dict:
        out = {k: v for k, v in self.__dict__.items() if k != 'acf'}
        for k, v in out.items():
            if isinstance(v, (np.floating, np.integer)):
                out[k] = v.item()
        return out


def spacing_metrics(dyads: DyadSet, span: Tuple[int, int],
                    params: NucParams = DEFAULT_PARAMS,
                    keep_acf: bool = False) -> SpacingMetrics:
    """NRL, S, tau*, SNR and CV_g over molecular ``span`` from ``dyads``."""
    out = SpacingMetrics(n_dyads=int(len(dyads.centres)), n_gaps=int(len(dyads.gaps)))
    g = dyads.gaps
    if len(g) >= params.min_gaps_nrl:
        out.nrl = float(np.median(g))
        out.too_short_nrl = False
    if len(g) >= params.min_gaps_regularity:
        mu = float(np.mean(g))
        out.gap_cv = float(np.std(g) / mu) if mu > 0 else float('nan')
    start, end = int(span[0]), int(span[1])
    lag_lo, lag_hi = params.acf_lag
    need_snr = (end - start) >= params.snr_min_bp and len(dyads.centres) >= params.snr_min_dyads
    need_acf = len(g) >= params.min_gaps_regularity or need_snr or keep_acf
    if need_acf and end - start > lag_hi * 2 and len(dyads.centres) >= 2:
        x = dyad_indicator(dyads.centres, start, end, params.acf_halfwidth)
        x = x - x.mean()
        ac = _biased_acf(x, max(lag_hi, params.snr_band_bp[1]))
        tau = int(np.argmax(ac[lag_lo:lag_hi + 1]) + lag_lo)
        if keep_acf:
            out.acf = ac
        if len(g) >= params.min_gaps_regularity:
            out.tau = float(tau)
            out.regularity = float(ac[tau] - ac[int(round(tau / 2))])
            out.too_short_regularity = False
        if need_snr:
            nfft = 1 << int(np.ceil(np.log2(max(len(x) * 4, 1024))))
            power = np.abs(np.fft.rfft(x, nfft)) ** 2
            freqs = np.fft.rfftfreq(nfft)
            lo_f, hi_f = 1.0 / params.snr_band_bp[1], 1.0 / params.snr_band_bp[0]
            band = (freqs >= lo_f) & (freqs <= hi_f)
            med = float(np.median(power[band])) if band.any() else 0.0
            t = tau if np.isfinite(out.tau) else tau
            k = int(np.argmin(np.abs(freqs - 1.0 / t)))
            if med > 0:
                out.spectral_snr = float(power[k] / med)
                out.too_short_snr = False
    return out


def molecule_spacing(mol: NucMolecule, nrl_pooled: Optional[float] = None,
                     params: NucParams = DEFAULT_PARAMS,
                     coarse_codes: Optional[np.ndarray] = None,
                     window: Optional[Tuple[int, int]] = None,
                     keep_acf: bool = False) -> Tuple[SpacingMetrics, DyadSet]:
    """Spacing metrics of a whole molecule, or of a molecular window."""
    d = eligible_dyads(mol, nrl_pooled, params, coarse_codes, window)
    span = window if window is not None else (0, mol.length)
    return spacing_metrics(d, span, params, keep_acf), d


def dataset_period(nrls: Sequence[float]) -> float:
    """P_dataset: median of per-molecule NRLs (finite values only)."""
    a = np.asarray([x for x in nrls if np.isfinite(x)], dtype=float)
    return float(np.median(a)) if len(a) else float('nan')


def bootstrap_nrl_pooled(molecules: Sequence[NucMolecule],
                         params: NucParams = DEFAULT_PARAMS) -> Tuple[float, float]:
    """Two-step NRL_pooled: bootstrap mono range -> P0 -> 0.45-1.25 x P0 -> P1.

    Returns ``(P0, P1)``.
    """
    p0 = dataset_period([molecule_spacing(m, None, params)[0].nrl for m in molecules])
    p1 = dataset_period([molecule_spacing(m, p0, params)[0].nrl for m in molecules])
    return p0, p1


def mean_dyad_acf(molecules: Sequence[NucMolecule], nrl_pooled: Optional[float],
                  params: NucParams = DEFAULT_PARAMS, max_lag: int = 400,
                  min_gaps: int = 0) -> Tuple[np.ndarray, int]:
    """Mean dyad autocorrelogram over molecules (each normalised by lag 0)."""
    acc = np.zeros(max_lag + 1)
    n = 0
    for m in molecules:
        d = eligible_dyads(m, nrl_pooled, params)
        if len(d.gaps) < min_gaps or len(d.centres) < 2:
            continue
        x = dyad_indicator(d.centres, 0, m.length, params.acf_halfwidth)
        x = x - x.mean()
        acc += _biased_acf(x, max_lag)
        n += 1
    return (acc / n if n else acc), n


def all_centre_regularity(mol: NucMolecule, params: NucParams = DEFAULT_PARAMS) -> float:
    """The demo script's regularity: ACF over every footprint centre (no
    eligibility rule). Kept for the NUC 2.2 comparison only."""
    if len(mol.nucs) < 3:
        return float('nan')
    c = (mol.nucs[:, 0] + mol.nucs[:, 1]) // 2
    x = dyad_indicator(c, 0, mol.length, params.acf_halfwidth)
    x = x - x.mean()
    ac = _biased_acf(x, 400)
    lag = int(np.argmax(ac[150:260]) + 150)
    return float(ac[lag] - ac[int(lag * 0.5)])
