"""Footprint sizes: the density-matched reference curve, z, coarse calls and
(absolute calibration only) named classes (NUC 3.2).

Observation model: L = T + o_L + o_R + eps. The overshoot o shrinks as the
accessible marking rate lambda rises, so footprint length depends on mark
density as much as on what protects the DNA.

**Reference curve.** mu_mono(lambda) is the mode of mono-sized footprints
(``mono_frac`` x NRL_pooled) as a function of lambda_i: modes in quantile
bins of log(lambda), made monotone non-increasing by isotonic regression and
joined piecewise-linearly over log(lambda) (constant outside the knots).
sigma_mono(lambda) is the robust spread (1.4826 x MAD) in the same bins.
Both describe observed lengths only; they make no claim about T.

**z** = (L - mu_mono(lambda_i)) / sigma_mono(lambda_i).

**Coarse call.** Classes are regions of true length: short (z < -2), mono
(-2 <= z <= 2), multi(m) for L within 2 sigma_m of (m - 1) NRL + mu_mono
(sigma_m^2 = sigma_mono^2 + (m - 1) sigma_gap^2), multi(3+) above the
multi(3) lower bound, and long for the rest above mono. The posterior of each
class is the mass of N(L, sigma_e^2) inside its region, sigma_e^2 =
2 sigma_loc(lambda)^2 being the thinning-derived edge noise of the observed
length (sigma_loc = 0 without thinning: the posterior is then 1 for the
region L falls in). A footprint is called when the top posterior is at least
``coarse_min_posterior`` and min(n_L, n_R) >= ``min_edge_sites``; otherwise
it is can't-tell. Edge-truncated footprints get their own code.

**Named classes** need an absolute control (NUC 4.2): locations T_k + b(lambda)
with b from the control, never fitted jointly with the weights.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
from scipy.stats import norm

from fiberhmm.nuc.params import DEFAULT_PARAMS, NucParams

# Coarse codes (MA nucsz Q1)
CANT_TELL, SHORT, MONO, LONG, MULTI2, MULTI3, EDGE_TRUNCATED = range(7)
COARSE_NAMES = {CANT_TELL: 'cant_tell', SHORT: 'short', MONO: 'mono', LONG: 'long',
                MULTI2: 'multi2', MULTI3: 'multi3+', EDGE_TRUNCATED: 'edge_truncated'}
# Named codes (MA nucsz Q2)
NAMED_NONE, NAMED_SMALL, NAMED_HEXA, NAMED_OCT, NAMED_CHROMATO = range(5)
NAMED_CODES = {'small': NAMED_SMALL, 'hexasome': NAMED_HEXA, 'octasome': NAMED_OCT,
               'chromatosome': NAMED_CHROMATO}


def kde_mode(values: np.ndarray, lo: float, hi: float, bw: float = 3.0) -> float:
    """Mode of ``values`` within [lo, hi] by a Gaussian KDE on a 1 bp grid."""
    v = np.asarray(values, dtype=float)
    v = v[(v >= lo) & (v <= hi)]
    if len(v) < 5:
        return float('nan')
    grid = np.arange(np.floor(lo), np.ceil(hi) + 1)
    h = np.bincount(np.clip(np.round(v - grid[0]).astype(int), 0, len(grid) - 1),
                    minlength=len(grid)).astype(float)
    half = int(np.ceil(4 * bw))
    k = np.exp(-0.5 * (np.arange(-half, half + 1) / bw) ** 2)
    s = np.convolve(h, k, mode='same')
    return float(grid[int(np.argmax(s))])


def _isotonic_decreasing(y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Pool-adjacent-violators for a non-increasing fit."""
    y = list(-np.asarray(y, dtype=float))
    w = list(np.asarray(w, dtype=float))
    blocks = [[yi, wi, 1] for yi, wi in zip(y, w)]
    out: List[list] = []
    for b in blocks:
        out.append(b)
        while len(out) > 1 and out[-2][0] > out[-1][0]:
            y2, w2, n2 = out.pop()
            y1, w1, n1 = out.pop()
            out.append([(y1 * w1 + y2 * w2) / (w1 + w2), w1 + w2, n1 + n2])
    res = []
    for yb, _, nb in out:
        res.extend([-yb] * nb)
    return np.asarray(res)


@dataclass
class ReferenceCurve:
    """mu_mono and sigma_mono over log(lambda), as knots."""
    log_lambda: List[float]
    mu: List[float]
    sigma: List[float]
    n: List[int] = field(default_factory=list)
    raw_mode: List[float] = field(default_factory=list)

    def mu_at(self, lam) -> np.ndarray:
        x = np.log(np.clip(np.asarray(lam, dtype=float), 1e-6, 1.0))
        return np.interp(x, self.log_lambda, self.mu)

    def sigma_at(self, lam) -> np.ndarray:
        x = np.log(np.clip(np.asarray(lam, dtype=float), 1e-6, 1.0))
        return np.interp(x, self.log_lambda, self.sigma)

    def z(self, lengths, lam) -> np.ndarray:
        return (np.asarray(lengths, dtype=float) - self.mu_at(lam)) / np.maximum(self.sigma_at(lam), 1.0)

    def to_dict(self):
        return {'log_lambda': [float(x) for x in self.log_lambda],
                'mu': [float(x) for x in self.mu], 'sigma': [float(x) for x in self.sigma],
                'n': [int(x) for x in self.n], 'raw_mode': [float(x) for x in self.raw_mode]}

    @classmethod
    def from_dict(cls, d):
        return cls(list(d['log_lambda']), list(d['mu']), list(d['sigma']),
                   list(d.get('n', [])), list(d.get('raw_mode', [])))


def fit_reference_curve(lengths: np.ndarray, lambdas: np.ndarray, nrl_pooled: float,
                        params: NucParams = DEFAULT_PARAMS) -> ReferenceCurve:
    """Fit mu_mono / sigma_mono from footprint lengths and their lambda_i."""
    L = np.asarray(lengths, dtype=float)
    lam = np.asarray(lambdas, dtype=float)
    lo, hi = params.mono_frac[0] * nrl_pooled, params.mono_frac[1] * nrl_pooled
    ok = (L >= lo) & (L <= hi) & np.isfinite(lam) & (lam > 0)
    L, lam = L[ok], lam[ok]
    if len(L) < params.curve_min_per_bin:
        mode = kde_mode(L, lo, hi) if len(L) >= 5 else (lo + hi) / 2
        mad = float(np.median(np.abs(L - np.median(L)))) * 1.4826 if len(L) else 20.0
        return ReferenceCurve([np.log(0.5)], [mode], [max(mad, 5.0)], [int(len(L))], [mode])
    x = np.log(lam)
    nb = int(min(params.curve_knots, max(1, len(L) // params.curve_min_per_bin)))
    edges = np.quantile(x, np.linspace(0, 1, nb + 1))
    edges[-1] += 1e-9
    centres, modes, sig, ns = [], [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        sel = (x >= a) & (x < b)
        if sel.sum() < 20:
            continue
        v = L[sel]
        m = kde_mode(v, lo, hi)
        centres.append(float(np.median(x[sel])))
        modes.append(m)
        sig.append(max(float(np.median(np.abs(v - np.median(v)))) * 1.4826, 5.0))
        ns.append(int(sel.sum()))
    mono = _isotonic_decreasing(np.asarray(modes), np.asarray(ns, dtype=float))
    return ReferenceCurve(centres, [float(v) for v in mono], sig, ns, modes)


@dataclass
class CoarseCall:
    code: np.ndarray        # per footprint
    posterior: np.ndarray   # top posterior
    z: np.ndarray
    posteriors: Optional[np.ndarray] = None  # (n, 5) short, mono, long, multi2, multi3


def coarse_regions(mu: np.ndarray, sigma: np.ndarray, nrl: np.ndarray,
                   sigma_gap: float, params: NucParams = DEFAULT_PARAMS):
    """Region bounds per footprint: (short_hi, mono_hi, m2_lo, m2_hi, m3_lo)."""
    s2 = np.sqrt(sigma ** 2 + sigma_gap ** 2)
    s3 = np.sqrt(sigma ** 2 + 2 * sigma_gap ** 2)
    short_hi = mu + params.z_short * sigma
    mono_hi = mu + params.z_long * sigma
    m2 = mu + nrl
    m3 = mu + 2 * nrl
    m2_lo = np.maximum(m2 - 2 * s2, mono_hi)
    m2_hi = m2 + 2 * s2
    m3_lo = np.maximum(m3 - 2 * s3, m2_hi)
    return short_hi, mono_hi, m2_lo, m2_hi, m3_lo


def coarse_posteriors(lengths, mu, sigma, nrl, sigma_gap, sigma_e,
                      params: NucParams = DEFAULT_PARAMS) -> np.ndarray:
    """Posterior mass of N(L, sigma_e^2) in each coarse region.

    Columns: short, mono, long, multi2, multi3+. With sigma_e = 0 the row is
    one-hot on the region containing L.
    """
    L = np.asarray(lengths, dtype=float)
    short_hi, mono_hi, m2_lo, m2_hi, m3_lo = coarse_regions(
        np.asarray(mu, float), np.asarray(sigma, float), np.asarray(nrl, float), sigma_gap, params)
    se = np.maximum(np.asarray(sigma_e, dtype=float), 1e-6)

    def cdf(b):
        return norm.cdf((b - L) / se)
    c_short = cdf(short_hi)
    c_mono = cdf(mono_hi)
    c_m2lo = cdf(m2_lo)
    c_m2hi = cdf(m2_hi)
    c_m3lo = cdf(m3_lo)
    p_short = c_short
    p_mono = c_mono - c_short
    p_long = (c_m2lo - c_mono) + (c_m3lo - c_m2hi)
    p_m2 = c_m2hi - c_m2lo
    p_m3 = 1.0 - c_m3lo
    P = np.stack([p_short, p_mono, p_long, p_m2, p_m3], axis=1)
    return np.clip(P, 0.0, 1.0)


_COL_CODE = np.array([SHORT, MONO, LONG, MULTI2, MULTI3])


def call_coarse(lengths, lambdas, curve: ReferenceCurve, nrl, sigma_gap: float,
                sigma_loc_fn, n_left, n_right, truncated,
                params: NucParams = DEFAULT_PARAMS,
                lambda_supported=None) -> CoarseCall:
    """Coarse calls for footprints of one molecule.

    ``sigma_loc_fn(lambda)`` gives the per-edge noise from thinning (zeros
    without calibration). ``lambda_supported(lambda)`` returns False outside
    the calibration's supported range: those footprints are can't-tell.
    """
    L = np.asarray(lengths, dtype=float)
    lam = np.asarray(lambdas, dtype=float)
    mu = curve.mu_at(lam)
    sig = np.maximum(curve.sigma_at(lam), 1.0)
    z = (L - mu) / sig
    nrl_arr = np.broadcast_to(np.asarray(nrl, dtype=float), L.shape)
    se = np.sqrt(2.0) * np.asarray(sigma_loc_fn(lam), dtype=float)
    P = coarse_posteriors(L, mu, sig, nrl_arr, sigma_gap, se, params)
    top = P.argmax(axis=1) if len(L) else np.zeros(0, dtype=int)
    post = P[np.arange(len(L)), top] if len(L) else np.zeros(0)
    code = _COL_CODE[top] if len(L) else np.zeros(0, dtype=int)
    weak = (post < params.coarse_min_posterior) | (np.minimum(n_left, n_right) < params.min_edge_sites)
    if lambda_supported is not None and len(L):
        weak |= ~np.asarray(lambda_supported(lam), dtype=bool)
    code = np.where(weak, CANT_TELL, code)
    code = np.where(np.asarray(truncated, dtype=bool), EDGE_TRUNCATED, code)
    return CoarseCall(code.astype(np.int64), post, z, P)


@dataclass
class AbsoluteClasses:
    """Named-class model from an absolute control (NUC 4.2)."""
    T: Dict[str, float]                 # class -> true protected length
    b_log_lambda: List[float]           # b(lambda) knots
    b: List[float]
    sigma_b: List[float]
    sigma_struct: Dict[str, float]
    weights: Dict[str, float] = field(default_factory=dict)

    def locations(self, lam):
        x = np.log(np.clip(np.asarray(lam, float), 1e-6, 1.0))
        b = np.interp(x, self.b_log_lambda, self.b)
        sb = np.interp(x, self.b_log_lambda, self.sigma_b)
        names = list(self.T)
        mu = np.stack([self.T[k] + b for k in names], axis=-1)
        sd = np.stack([np.sqrt(self.sigma_struct.get(k, 5.0) ** 2 + sb ** 2) for k in names], axis=-1)
        return names, mu, sd

    def posteriors(self, lengths, lam):
        names, mu, sd = self.locations(lam)
        L = np.asarray(lengths, float)[..., None]
        w = np.asarray([self.weights.get(k, 1.0 / len(names)) for k in names])
        like = w * norm.pdf(L, mu, sd)
        tot = like.sum(axis=-1, keepdims=True)
        return names, np.where(tot > 0, like / np.maximum(tot, 1e-300), 0.0)

    def to_dict(self):
        return {'T': dict(self.T), 'b_log_lambda': list(self.b_log_lambda), 'b': list(self.b),
                'sigma_b': list(self.sigma_b), 'sigma_struct': dict(self.sigma_struct),
                'weights': dict(self.weights)}

    @classmethod
    def from_dict(cls, d):
        return cls(dict(d['T']), list(d['b_log_lambda']), list(d['b']), list(d['sigma_b']),
                   dict(d['sigma_struct']), dict(d.get('weights', {})))


def call_named(lengths, lambdas, model: Optional[AbsoluteClasses], n_left, n_right,
               params: NucParams = DEFAULT_PARAMS):
    """(named code, posterior) per footprint; code 0 without an absolute model
    or when the call is can't-tell (max posterior < named_min_posterior or
    min(n_L, n_R) < min_edge_sites)."""
    L = np.asarray(lengths, float)
    if model is None or len(L) == 0:
        return np.zeros(len(L), dtype=np.int64), np.zeros(len(L))
    names, P = model.posteriors(L, lambdas)
    top = P.argmax(axis=1)
    post = P[np.arange(len(L)), top]
    codes = np.asarray([NAMED_CODES.get(names[t], NAMED_NONE) for t in top], dtype=np.int64)
    weak = (post < params.named_min_posterior) | (np.minimum(n_left, n_right) < params.min_edge_sites)
    return np.where(weak, NAMED_NONE, codes), post
