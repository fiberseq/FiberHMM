"""Phasing and positional dispersion (NUC 3.5).

Two frames:

``anchor``
    distances from a fixed site (a reference position projected into each
    molecule's molecular coordinates through its alignment).
``molecule``
    distances from the molecule's own barrier at the anchor: the edges of its
    MSP covering anchor +/- ``barrier_slop`` (at least ``barrier_min_msp``
    long), else of its nucleosome footprint covering the anchor.

Distances are measured along the molecule (molecular bp), before any
projection. With an orientation, x = o * s_read * (d - a), where s_read is -1
for reverse reads (molecular coordinates run against the reference); the
downstream side is x > 0 and the upstream side is reported separately.
Unoriented anchors fold to |d - a|.

**Bins span whole periods**: B_k = [x0 + kP, x0 + (k + K)P) plus a summary of
``phase_summary_periods`` periods, so a uniform dyad distribution has zero
expected resultant. Bins fixed in bp get the uniform expectation subtracted
instead (:func:`uniform_expectation`).

**Null**: no Rayleigh test (dyads within a molecule are dependent). Each
molecule gets one uniform phase shift, shared by its appearances at several
anchors; ``phase_n_null`` draws give
E_null[R] and the exceedance p. **Period sensitivity** reports the maximum R
over P +/- ``phase_period_window`` bp.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from fiberhmm.nuc.params import DEFAULT_PARAMS, NucParams
from fiberhmm.nuc.records import NucMolecule
from fiberhmm.nuc.spacing import eligible_dyads


def uniform_expectation(lo: float, hi: float, P: float) -> complex:
    """E[exp(2 pi i x / P)] for x uniform on [lo, hi)."""
    w = hi - lo
    if w <= 0:
        return 0j
    k = 2j * np.pi / P
    return complex((np.exp(k * hi) - np.exp(k * lo)) / (k * w))


def period_bins(P: float, params: NucParams = DEFAULT_PARAMS,
                n_bins: Optional[int] = None) -> List[Tuple[float, float]]:
    """Sliding bins of ``phase_bin_periods`` periods stepped by one period,
    covering the summary span, then the summary bin itself (last)."""
    x0 = params.phase_x0
    K = params.phase_bin_periods
    S = params.phase_summary_periods
    if n_bins is None:
        n_bins = max(S - K + 1, 1)
    bins = [(x0 + k * P, x0 + (k + K) * P) for k in range(n_bins)]
    bins.append((x0, x0 + S * P))
    return bins


def barrier_at(mol: NucMolecule, a_mol: int, params: NucParams = DEFAULT_PARAMS) -> Optional[Tuple[int, int, str]]:
    """(start, end, kind) of the molecule's barrier at molecular ``a_mol``."""
    lo, hi = a_mol - params.barrier_slop, a_mol + params.barrier_slop
    if len(mol.msps):
        m = mol.msps
        ok = (m[:, 0] <= lo) & (m[:, 1] >= hi) & ((m[:, 1] - m[:, 0]) >= params.barrier_min_msp)
        if ok.any():
            i = int(np.flatnonzero(ok)[0])
            return int(m[i, 0]), int(m[i, 1]), 'msp'
    if len(mol.nucs):
        n = mol.nucs
        ok = (n[:, 0] <= a_mol) & (n[:, 1] > a_mol)
        if ok.any():
            i = int(np.flatnonzero(ok)[0])
            return int(n[i, 0]), int(n[i, 1]), 'footprint'
    return None


def anchor_offsets(mol: NucMolecule, dyads: np.ndarray, anchor_ref: int,
                   orientation: int = 0, frame: str = 'anchor',
                   params: NucParams = DEFAULT_PARAMS) -> Tuple[Optional[np.ndarray], str]:
    """Signed (oriented) or folded distances of ``dyads`` from the anchor.

    Returns ``(x, reason)``; ``x`` is None when the anchor is absent from the
    molecule or (molecule frame) there is no barrier.
    """
    a = int(mol.ref_to_mol([anchor_ref])[0])
    if a < 0:
        return None, 'anchor_absent'
    s_read = -1 if mol.is_reverse else 1
    d = np.asarray(dyads, dtype=float)
    if frame == 'anchor':
        rel = d - a
    elif frame == 'molecule':
        b = barrier_at(mol, a, params)
        if b is None:
            return None, 'no_barrier'
        bs, be, _ = b
        rel = np.where(d >= be, d - be, np.where(d < bs, d - bs, np.nan))
        rel = rel[np.isfinite(rel)]
    else:
        raise ValueError(f"unknown frame {frame!r}")
    if orientation:
        return orientation * s_read * rel, 'ok'
    return np.abs(rel), 'ok'


@dataclass
class BinResult:
    lo: float
    hi: float
    R: float
    phi: float
    n_dyads: int
    n_molecules: int
    null_mean: float
    null_p: float
    R_minus_null: float
    R_max_period: float
    P_max: float

    def as_dict(self):
        return {k: (float(v) if isinstance(v, (float, np.floating)) else v) for k, v in self.__dict__.items()}


@dataclass
class PhasingResult:
    frame: str
    P: float
    side: str
    bins: List[BinResult]
    oscillation: Dict[str, float] = field(default_factory=dict)
    per_molecule: List[dict] = field(default_factory=list)
    excluded: Dict[str, int] = field(default_factory=dict)
    dispersion_bp: float = float('nan')

    @property
    def summary(self) -> BinResult:
        return self.bins[-1]

    def as_dict(self):
        return {'frame': self.frame, 'P': self.P, 'side': self.side,
                'bins': [b.as_dict() for b in self.bins], 'oscillation': dict(self.oscillation),
                'excluded': dict(self.excluded), 'dispersion_bp': self.dispersion_bp,
                'n_molecules': len(self.per_molecule)}


def _resultant(x_list: Sequence[np.ndarray], lo: float, hi: float, P: float, correct: bool = False):
    V = np.zeros(len(x_list), dtype=complex)
    N = np.zeros(len(x_list))
    eu = uniform_expectation(lo, hi, P) if correct else 0j
    for i, x in enumerate(x_list):
        sel = x[(x >= lo) & (x < hi)]
        if len(sel):
            V[i] = np.exp(2j * np.pi * sel / P).sum() - eu * len(sel)
            N[i] = len(sel)
    return V, N


def resultant_with_null(x_list: Sequence[np.ndarray], lo: float, hi: float, P: float,
                        n_null: int, rng: np.random.Generator, correct: bool = False,
                        molecule_ids: Optional[Sequence[int]] = None):
    """Pooled resultant, phase and molecule-level phase-randomisation null.

    ``molecule_ids`` groups appearances of one physical molecule (at several
    anchors): they share one random shift per draw, and the reported molecule
    count is of distinct molecules."""
    V, N = _resultant(x_list, lo, hi, P, correct)
    tot = N.sum()
    if tot == 0:
        return float('nan'), float('nan'), 0, 0, float('nan'), float('nan')
    S = V.sum()
    R = abs(S) / tot
    phi = float(np.angle(S))
    ids = np.arange(len(V)) if molecule_ids is None else np.asarray(molecule_ids)
    use = N > 0
    uniq, inv = np.unique(ids[use], return_inverse=True)
    Vm = np.zeros(len(uniq), dtype=complex)
    np.add.at(Vm, inv, V[use])                      # one vector per molecule
    shifts = rng.uniform(0, 2 * np.pi, size=(n_null, len(uniq)))
    Rn = np.abs((Vm[None, :] * np.exp(1j * shifts)).sum(axis=1)) / tot
    p = (1 + int(np.sum(Rn >= R))) / (n_null + 1)
    return R, phi, int(tot), int(len(uniq)), float(Rn.mean()), float(p)


def _osc_fit(x_all: np.ndarray, lo: float, hi: float, P: float) -> Dict[str, float]:
    from scipy.optimize import curve_fit
    if len(x_all) < 50:
        return {}
    edges = np.arange(lo, hi + 10, 10.0)
    h, _ = np.histogram(x_all, bins=edges)
    xc = (edges[:-1] + edges[1:]) / 2

    def f(x, D0, A, ell, x1):
        return D0 + A * np.exp(-(x - lo) / ell) * np.cos(2 * np.pi * (x - x1) / P)
    D0 = max(h.mean(), 1e-9)
    best = None
    for x1g in np.linspace(lo, lo + P, 6, endpoint=False):
        try:
            popt, _ = curve_fit(f, xc, h, p0=[D0, D0 * 0.3, 500.0, x1g],
                                bounds=([0, 0, 20, lo - P], [np.inf, np.inf, 1e5, lo + 2 * P]),
                                maxfev=4000)
            res = float(np.sum((f(xc, *popt) - h) ** 2))
            if best is None or res < best[0]:
                best = (res, popt)
        except Exception:
            continue
    if best is None:
        return {}
    D0, A, ell, x1 = best[1]
    return {'A_over_D0': float(A / D0) if D0 > 0 else float('nan'), 'ell': float(ell),
            'x1': float((x1 - lo) % P + lo), 'D0': float(D0)}


def phasing(molecules: Sequence[NucMolecule], anchors: Sequence[Tuple[str, int, int]],
            frame: str = 'anchor', P: Optional[float] = None, *,
            nrl_pooled: Optional[float] = None, params: NucParams = DEFAULT_PARAMS,
            dyads: Optional[Dict[int, np.ndarray]] = None, side: str = 'both',
            bins: Optional[List[Tuple[float, float]]] = None, correct_uniform: bool = False,
            seed: int = 0, require_span: bool = True) -> PhasingResult:
    """Pooled phasing over molecule appearances at ``anchors``.

    ``anchors`` are ``(chrom, ref_pos, orientation)``. A molecule appears at
    an anchor when it is on that contig and (``require_span``) its alignment
    covers the anchor +/- the summary span. ``side`` is ``'both'`` for
    unoriented (folded) analysis, or ``'down'``/``'up'`` for oriented anchors.
    ``dyads`` optionally maps ``id(molecule)`` to precomputed eligible dyads.
    """
    if P is None:
        P = nrl_pooled if nrl_pooled else 189.0
    rng = np.random.default_rng(seed)
    if bins is None:
        bins = period_bins(P, params)
    span = params.phase_x0 + params.phase_summary_periods * P
    x_list: List[np.ndarray] = []
    who: List[Tuple[int, int]] = []
    excluded: Dict[str, int] = {}
    by_chrom: Dict[str, List[int]] = {}
    for i, m in enumerate(molecules):
        by_chrom.setdefault(m.chrom, []).append(i)
    for ai, (chrom, pos, orient) in enumerate(anchors):
        o = int(orient) if side != 'both' else 0
        for i in by_chrom.get(chrom, []):
            m = molecules[i]
            if require_span and not (m.ref_start <= pos - span and m.ref_end >= pos + span):
                continue
            if not (m.ref_start <= pos < m.ref_end):
                continue
            d = dyads.get(id(m)) if dyads is not None else None
            if d is None:
                d = eligible_dyads(m, nrl_pooled, params).centres
            x, why = anchor_offsets(m, d, pos, o if o else 0, frame, params)
            if x is None:
                excluded[why] = excluded.get(why, 0) + 1
                continue
            if o and side == 'up':
                x = -x
            if o:
                x = x[x > 0]
            x_list.append(x)
            who.append((i, ai))
    results = []
    mol_ids = [i for i, _ai in who]
    for lo, hi in bins:
        R, phi, nd, nm, null, p = resultant_with_null(x_list, lo, hi, P, params.phase_n_null, rng,
                                                      correct_uniform, molecule_ids=mol_ids)
        best_R, best_P = R, P
        for Pp in np.arange(P - params.phase_period_window, P + params.phase_period_window + 1, 1.0):
            k = (hi - lo) / P
            lo2 = lo if correct_uniform else params.phase_x0 + (lo - params.phase_x0) / P * Pp
            hi2 = lo2 + k * Pp if not correct_uniform else hi
            V, N = _resultant(x_list, lo2, hi2, Pp, correct_uniform)
            if N.sum() > 0:
                r = abs(V.sum()) / N.sum()
                if r > best_R:
                    best_R, best_P = r, float(Pp)
        results.append(BinResult(lo, hi, R, phi, nd, nm, null, p,
                                 R - null if np.isfinite(R) else float('nan'), best_R, best_P))
    summ = bins[-1]
    x_all = np.concatenate([x[(x >= summ[0]) & (x < summ[1])] for x in x_list]) if x_list else np.zeros(0)
    osc = _osc_fit(x_all, summ[0], summ[1], P)
    per_mol = []
    phi_W = results[-1].phi
    for (i, ai), x in zip(who, x_list):
        sel = x[(x >= summ[0]) & (x < summ[1])]
        n = len(sel)
        if n == 0:
            per_mol.append({'molecule': i, 'anchor': ai, 'n': 0, 'r': float('nan'),
                            'theta': float('nan'), 'reliable': False, 'f_bp': float('nan')})
            continue
        v = np.exp(2j * np.pi * sel / P).sum()
        r = abs(v) / n
        th = float(np.angle(v))
        reliable = bool(r >= params.theta_min_r and n >= params.theta_min_dyads)
        f = abs(np.angle(np.exp(1j * (th - phi_W)))) * P / (2 * np.pi) if reliable else float('nan')
        per_mol.append({'molecule': i, 'anchor': ai, 'n': n, 'r': float(r), 'theta': th,
                        'reliable': reliable, 'f_bp': float(f)})
    Rs = results[-1].R
    disp = float(P / (2 * np.pi) * np.sqrt(-2 * np.log(Rs))) if np.isfinite(Rs) and Rs > 0 else float('nan')
    return PhasingResult(frame, float(P), side, results, osc, per_mol, excluded, disp)
