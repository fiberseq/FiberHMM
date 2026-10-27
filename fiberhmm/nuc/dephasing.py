"""Fuzzy / dephased nucleosomes at anchors (NUC 13.1, kind 4).

Phasing at an anchor needs two things: molecules that are periodic near the
anchor, and molecules that share a phase. This module separates them.

For one anchor, each molecule spanning the anchor's summary span (``phase_x0``
to ``phase_x0 + phase_summary_periods`` periods on each side of an unoriented
anchor, or downstream when oriented) contributes per side s a vector
V_m,s = sum exp(2 pi i x / P) over its observed mono dyads at distance x in
that span (whole periods, so a uniform dyad distribution has zero
expectation). The sides are kept apart (folding them into one vector cancels
them when the array sits a quarter period off the anchor):

- ``R`` = sum_s |sum_m V_m,s| / N (N dyads), with the large-sample
  molecule-level phase null E[R] = sum_s sqrt(pi/4 sum_m |V_m,s|^2) / N, and
  ``excess`` = R - null;
- ``coherence`` = sum_s |sum_m V_m,s| / sum_{m,s} |V_m,s|: 1 when every
  molecule has the same phase, about 1/sqrt(n) when phases are random;
- ``rbar`` = mean over molecules of sum_s |V_m,s| / n_m: within-molecule
  periodicity, whatever the phase;
- ``dispersion_bp`` = P / (2 pi) sqrt(-2 ln coherence): the positional
  spread the coherence implies.

A locus is fuzzy (dephased) when its molecules keep their periodicity but
lose their shared phase: lower coherence than matched background anchors at
similar rbar. Matched backgrounds are anchors of the same kind of locus
without the property under test (the caller supplies them: control peaks for
enhancers, say), compared within strata of molecule density (lambda_acc)
and depth. :func:`dephasing_test` permutes test and background labels within
strata; :func:`matched_percentile` places one anchor among matched
backgrounds (window mode).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from fiberhmm.nuc.params import DEFAULT_PARAMS, NucParams
from fiberhmm.nuc.phasing import anchor_offsets
from fiberhmm.nuc.sizes import MONO


@dataclass
class AnchorPhase:
    chrom: str
    pos: int
    orientation: int
    frame: str
    P: float
    n_molecules: int
    n_dyads: int
    R: float
    null: float
    excess: float
    coherence: float
    rbar: float
    dispersion_bp: float
    lambda_acc: float

    def as_dict(self):
        return {k: (float(v) if isinstance(v, (float, np.floating)) else v) for k, v in asdict(self).items()}


def _mono_dyads(res) -> Optional[np.ndarray]:
    fp = res.footprints if res is not None else None
    if fp is None:
        return None
    sel = (fp['coarse'] == MONO) & ~fp['is_tf'] & ~fp['truncated']
    return np.sort((fp['start'][sel] + fp['end'][sel]) / 2.0)


def anchor_phase(molecules, anchor: Tuple[str, int, int], cal, params: NucParams = DEFAULT_PARAMS,
                 results: Optional[Dict[int, object]] = None, frame: str = 'anchor',
                 P: Optional[float] = None) -> AnchorPhase:
    """Phase statistics of the molecules at one anchor ``(chrom, pos, orientation)``."""
    from fiberhmm.nuc.molecule import molecule_outputs
    chrom, pos, orient = anchor
    P = float(P or (cal.nrl_pooled if cal is not None else 189.0))
    x0 = params.phase_x0
    hi = x0 + params.phase_summary_periods * P
    Vr, Vl, ns, lams = [], [], [], []
    for m in molecules:
        if m.chrom != chrom or not (m.ref_start <= pos - hi and m.ref_end >= pos + hi):
            continue
        res = results.get(id(m)) if results is not None else None
        if res is None:
            res = molecule_outputs(m, params, cal)
            if results is not None:
                results[id(m)] = res
        if res.flags.get('low_information'):
            continue
        d = _mono_dyads(res)
        if d is None or not len(d):
            continue
        # signed offsets (read strand applied), then folded unless oriented
        x, _why = anchor_offsets(m, d, pos, int(orient) or 1, frame, params)
        if x is None:
            continue
        down = x[(x >= x0) & (x < hi)]
        up = -x[(-x >= x0) & (-x < hi)] if not orient else np.zeros(0)
        if not (len(down) + len(up)):
            continue
        Vr.append(np.exp(2j * np.pi * down / P).sum())
        Vl.append(np.exp(2j * np.pi * up / P).sum())
        ns.append(len(down) + len(up))
        lams.append(float(res.scalars.get('lambda_acc', float('nan'))))
    nan = float('nan')
    if not ns:
        return AnchorPhase(chrom, int(pos), int(orient), frame, P, 0, 0, nan, nan, nan, nan, nan, nan, nan)
    Vr, Vl = np.asarray(Vr), np.asarray(Vl)
    ns = np.asarray(ns, float)
    N = float(ns.sum())
    S = float(abs(Vr.sum()) + abs(Vl.sum()))
    mags = np.abs(Vr) + np.abs(Vl)
    tot = float(mags.sum())
    R = S / N
    null = float((np.sqrt(np.pi / 4.0 * np.sum(np.abs(Vr) ** 2)) +
                  np.sqrt(np.pi / 4.0 * np.sum(np.abs(Vl) ** 2))) / N)
    coh = min(1.0, S / tot) if tot > 0 else nan            # rounding can push S / tot past 1
    disp = float(P / (2 * np.pi) * np.sqrt(-2 * np.log(coh))) if 0 < coh < 1 else (0.0 if coh == 1 else nan)
    return AnchorPhase(chrom, int(pos), int(orient), frame, P, len(ns), int(N), R, null, R - null, coh,
                       float(np.mean(mags / ns)), disp, float(np.nanmedian(lams)))


def _strata(values: np.ndarray, n: int) -> np.ndarray:
    ok = np.isfinite(values)
    out = np.zeros(len(values), dtype=np.int64)
    out[~ok] = n                                   # unknowns always get their own stratum
    if ok.sum() < n or n <= 1:
        return out
    edges = np.quantile(values[ok], np.linspace(0, 1, n + 1)[1:-1])
    out[ok] = np.searchsorted(edges, values[ok], side='right')
    return out


def dephasing_test(test: Sequence[AnchorPhase], background: Sequence[AnchorPhase], stat: str = 'coherence',
                   n_strata: int = 3, n_perm: int = 2000, seed: int = 0, min_molecules: int = 5) -> dict:
    """Stratified permutation test: is ``stat`` lower at the test anchors than
    at matched background anchors?

    Anchors with fewer than ``min_molecules`` molecules are dropped. Strata
    are the tertiles (``n_strata``) of molecule density and of depth over the
    pooled anchors; labels are permuted within strata. The effect is the
    stratum-weighted mean difference test - background (negative = dephased);
    ``p_dephased`` is one-sided for lower values, ``p_more_phased`` for higher."""
    rows = [(a, 1) for a in test if a.n_molecules >= min_molecules and np.isfinite(getattr(a, stat))] + \
           [(a, 0) for a in background if a.n_molecules >= min_molecules and np.isfinite(getattr(a, stat))]
    if not rows:
        return {'n_test': 0, 'n_background': 0, 'effect': None, 'p_dephased': None, 'p_more_phased': None}
    y = np.array([getattr(a, stat) for a, _ in rows], float)
    lab = np.array([t for _, t in rows], dtype=bool)
    lam = np.array([a.lambda_acc for a, _ in rows], float)
    dep = np.array([a.n_molecules for a, _ in rows], float)
    st = _strata(lam, n_strata) * (n_strata + 1) + _strata(dep, n_strata)
    groups = [np.flatnonzero(st == s) for s in np.unique(st)]
    groups = [g for g in groups if lab[g].any() and (~lab[g]).any()]
    if not groups:
        return {'n_test': int(lab.sum()), 'n_background': int((~lab).sum()), 'effect': None,
                'p_dephased': None, 'p_more_phased': None, 'note': 'no stratum holds both labels'}

    def effect(labels):
        num = den = 0.0
        for g in groups:
            t, b = labels[g], ~labels[g]
            w = float(t.sum())
            num += w * (y[g][t].mean() - y[g][b].mean())
            den += w
        return num / den
    obs = effect(lab)
    rng = np.random.default_rng(seed)
    perm = np.empty(n_perm)
    for i in range(n_perm):
        lp = lab.copy()
        for g in groups:
            lp[g] = rng.permutation(lab[g])
        perm[i] = effect(lp)
    tol = 1e-12
    return {'stat': stat, 'n_test': int(sum(lab[g].sum() for g in groups)),
            'n_background': int(sum((~lab[g]).sum() for g in groups)), 'strata': len(groups),
            'effect': float(obs),
            'p_dephased': float((1 + np.sum(perm <= obs + tol)) / (n_perm + 1)),
            'p_more_phased': float((1 + np.sum(perm >= obs - tol)) / (n_perm + 1))}


def matched_percentile(anchor: AnchorPhase, background: Sequence[AnchorPhase], stat: str = 'coherence',
                       n_match: int = 20, min_molecules: int = 5) -> dict:
    """One anchor among its matched backgrounds: the ``n_match`` background
    anchors nearest in (log lambda_acc, log depth); ``p_dephased`` = (1 + #
    matched with stat <= the anchor's) / (n + 1)."""
    v = getattr(anchor, stat)
    bg = [b for b in background if b.n_molecules >= min_molecules and np.isfinite(getattr(b, stat))]
    if anchor.n_molecules < min_molecules or not np.isfinite(v) or not bg:
        return {'stat': stat, 'value': None if not np.isfinite(v) else float(v), 'n_matched': 0,
                'p_dephased': None}
    la = np.log(max(anchor.lambda_acc, 1e-4)) if np.isfinite(anchor.lambda_acc) else 0.0
    d = np.array([np.hypot((np.log(max(b.lambda_acc, 1e-4)) if np.isfinite(b.lambda_acc) else 0.0) - la,
                           np.log(b.n_molecules) - np.log(anchor.n_molecules)) for b in bg])
    pick = [bg[i] for i in np.argsort(d, kind='stable')[:n_match]]
    vals = np.array([getattr(b, stat) for b in pick], float)
    tol = 1e-9 * max(1.0, abs(v))                  # numerical ties count against the anchor
    return {'stat': stat, 'value': float(v), 'n_matched': len(pick),
            'background_median': float(np.median(vals)),
            'p_dephased': float((1 + np.sum(vals <= v + tol)) / (len(vals) + 1))}
