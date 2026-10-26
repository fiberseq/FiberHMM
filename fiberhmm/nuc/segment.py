"""Factorial segmentation of molecules into array regions (NUC 3.6).

**Streams** (molecular coordinates):

- spacing: eligible dyads (observed coarse-mono centres) plus virtual dyads
  from multi(m) calls (m equally spaced dyads, weighted by the call's
  posterior). Steps are the gaps between consecutive dyads.
- composition: every footprint (nucleosome and TF calls) is attached to the
  step whose interval contains its centre. A run of short/TF footprints with
  no dyad within ``seg_composition_flank`` bp forms its own composition-only
  step (no gap emission), so sub-rich stretches are callable where no mono
  dyad exists.

Molecules are pre-split at MSPs longer than ``long_msp_bp``; no state covers
them.

**States**: order {R_q for q in ``seg_q_grid`` at two tightnesses (tight:
``seg_sigma_tight``, the design's 8 bp; loose: the dataset's fitted jitter),
I, X} x composition {canonical, sub-rich}; X has no composition (47 base
states; 25 with ``seg_sigma_tight=None``). Tight and loose are both regular. Minimum
durations by expanded chains: ``seg_min_steps_regular`` for R_q,
``seg_min_steps_sub`` for sub-rich, ``seg_min_steps_irregular`` for I; a path
must end on a chain's last sub-state.

**Emissions per step** (each normalised, so LLRs compare):

1. gap, on the dyad grid: R_q sum_k w_k N(g; k q, k sigma_R^2 + 2 sigma_loc^2)
   for k = 1, 2, 3 (a can't-tell, short or long call between two mono calls
   hides a dyad but the grid still fits); I the same grid over the broad
   irregular density of the calibration's gap model (``seg_irregular =
   'empirical'`` restores the dataset's empirical gap density); X uniform on
   [0, 2000). sigma_R is the gap model's regular jitter minus the edge noise
   at the dataset's typical density. Gaps touching a virtual dyad carry no gap
   evidence (continuity only: virtual dyads are equally spaced by
   construction).
2. composition: multinomial over {mono, short, long, multi, TF} (can't-tell
   and edge-truncated are missing) with the composition's frequencies, times
   a Poisson on the step's total count scaled by its length. X uses the
   dataset mixture.
3. information: a step is flagged low-information when its local lambda is
   below the calibration floor (``seg_cant_tell_low_info`` also flags steps
   holding a can't-tell footprint); P(flag | X) = 0.8, P(flag | other) = 0.05.

**Transitions**: R_q -> R_q' only for |q - q'| <= ``seg_dq_max`` (1/20 spread
over the neighbours); other switches (order or composition) cost
``seg_switch_prob`` each.

**Decoding**: forward-backward marginals and the Viterbi path, whose runs of
one (order, composition) become segments (``seg_decode='posterior'`` labels
each step by its highest order marginal, summed over repeat and tightness:
more sensitive on sparse molecules, more false calls); a regular run's q-hat is the ML q over its
gaps. Order with LLR < ``seg_merge_llr`` nats against the best alternative
single order becomes irregular; a sub-rich flag with LLR below it is dropped;
regular and sub-rich segments need ``seg_min_dyads`` dyads and
``seg_min_bp`` bp. Molecules shorter than ``seg_min_molecule_bp`` are too
short.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from fiberhmm.nuc.params import DEFAULT_PARAMS, NucParams
from fiberhmm.nuc.sizes import (CANT_TELL, EDGE_TRUNCATED, LONG, MONO, MULTI2, MULTI3,
                                SHORT)

try:
    from numba import njit
except ImportError:  # pragma: no cover
    def njit(*a, **k):
        def w(f):
            return f
        return w if not (a and callable(a[0])) else a[0]

CATS = ('mono', 'short', 'long', 'multi', 'tf')
_X_UNIFORM = 1.0 / 2000.0
REGULAR, IRREGULAR, LOWINFO = 1, 2, 3
STATE_NAMES = {REGULAR: 'regular', IRREGULAR: 'irregular', LOWINFO: 'low_information'}


def _cat_of(code: int, is_tf: bool) -> int:
    if is_tf:
        return 4
    if code == MONO:
        return 0
    if code == SHORT:
        return 1
    if code == LONG:
        return 2
    if code in (MULTI2, MULTI3):
        return 3
    return -1


# ---------------------------------------------------------------------------
# State space
# ---------------------------------------------------------------------------

@dataclass
class StateSpace:
    base_order: np.ndarray     # per base state: 0..nq-1 = R_q index, nq = I, nq+1 = X
    base_comp: np.ndarray      # 0 canonical, 1 sub-rich (X: 0)
    exp_base: np.ndarray       # per expanded state: base index
    exp_first: np.ndarray      # bool: first sub-state
    exp_last: np.ndarray       # bool: last sub-state
    pred_ptr: np.ndarray       # CSR over expanded states: predecessors
    pred_idx: np.ndarray
    pred_logp: np.ndarray
    log_init: np.ndarray
    nq: int                    # number of regular columns (q x tightness)
    q_grid: np.ndarray         # repeat of each regular column
    tight: np.ndarray          # tight-regular flag of each regular column


_SPACE_CACHE: Dict[tuple, StateSpace] = {}


def state_space(params: NucParams = DEFAULT_PARAMS) -> StateSpace:
    n_tight = 2 if params.seg_sigma_tight is not None else 1
    key = (tuple(params.seg_q_grid), params.seg_min_steps_regular, params.seg_min_steps_sub,
           params.seg_min_steps_irregular, params.seg_switch_prob, params.seg_dq_max, n_tight)
    if key in _SPACE_CACHE:
        return _SPACE_CACHE[key]
    q1 = np.asarray(params.seg_q_grid, dtype=float)
    q = np.tile(q1, n_tight)                 # regular columns: loose q..., then tight q...
    tight = np.repeat(np.arange(n_tight) == 1, len(q1))
    nq = len(q)
    base_order, base_comp, dur = [], [], []
    for c in (0, 1):
        for i in range(nq):
            base_order.append(i)
            base_comp.append(c)
            dur.append(max(params.seg_min_steps_regular, params.seg_min_steps_sub if c else 1))
    for c in (0, 1):
        base_order.append(nq)
        base_comp.append(c)
        dur.append(max(params.seg_min_steps_irregular, params.seg_min_steps_sub if c else 1))
    base_order.append(nq + 1)
    base_comp.append(0)
    dur.append(1)
    nb = len(base_order)
    first_of = []
    last_of = []
    exp_base = []
    for b in range(nb):
        first_of.append(len(exp_base))
        for k in range(dur[b]):
            exp_base.append(b)
        last_of.append(len(exp_base) - 1)
    ns = len(exp_base)
    exp_base = np.asarray(exp_base)
    exp_first = np.zeros(ns, dtype=bool)
    exp_last = np.zeros(ns, dtype=bool)
    exp_first[first_of] = True
    exp_last[last_of] = True
    sw = params.seg_switch_prob
    # switch probabilities between base states (from a last sub-state)
    P = np.zeros((nb, nb))
    for a in range(nb):
        oa, ca = base_order[a], base_comp[a]
        for b in range(nb):
            if a == b:
                continue
            ob, cb = base_order[b], base_comp[b]
            if oa < nq and ob < nq:                      # R_q -> R_q'
                if ca == cb and abs(q[oa] - q[ob]) <= params.seg_dq_max:
                    n_nb = sum(1 for j in range(nq) if j != oa and abs(q[j] - q[oa]) <= params.seg_dq_max)
                    # (neighbouring repeats, or the other tightness at the same repeat)
                    P[a, b] = (1.0 / 20.0) / max(n_nb, 1)
                elif oa == ob and ca != cb:
                    P[a, b] = sw
            elif oa == nq + 1:                               # X -> anything
                P[a, b] = sw / (nb - 1)
            elif ob == nq + 1:                               # -> X
                P[a, b] = sw
            elif oa == ob and ca != cb:                      # I comp switch
                P[a, b] = sw
            elif ca == cb and (oa < nq) != (ob < nq):        # R <-> I, same comp
                P[a, b] = sw / (nq if ob < nq else 1)
    stay = 1.0 - P.sum(axis=1)
    if (stay <= 0).any():
        raise ValueError('segmentation switch probabilities exceed 1')
    preds: List[List[Tuple[int, float]]] = [[] for _ in range(ns)]
    for s in range(ns):
        b = exp_base[s]
        if not exp_last[s]:
            preds[s + 1].append((s, 0.0))
        else:
            preds[s].append((s, float(np.log(stay[b]))))
            for b2 in range(nb):
                if P[b, b2] > 0:
                    preds[first_of[b2]].append((s, float(np.log(P[b, b2]))))
    ptr = np.zeros(ns + 1, dtype=np.int64)
    idx, lp = [], []
    for s in range(ns):
        for p, l in preds[s]:
            idx.append(p)
            lp.append(l)
        ptr[s + 1] = len(idx)
    log_init = np.full(ns, -np.inf)
    log_init[exp_first] = -np.log(nb)
    sp = StateSpace(np.asarray(base_order), np.asarray(base_comp), exp_base, exp_first, exp_last,
                    ptr, np.asarray(idx, dtype=np.int64), np.asarray(lp), log_init, nq, q, tight)
    _SPACE_CACHE[key] = sp
    return sp


@njit(cache=True)
def _logsumexp2(a, b):
    if a == -np.inf:
        return b
    if b == -np.inf:
        return a
    m = a if a > b else b
    return m + np.log(np.exp(a - m) + np.exp(b - m))


@njit(cache=True)
def _forward_backward(E, log_init, ptr, idx, lp, end_mask):
    T, S = E.shape
    F = np.full((T, S), -np.inf)
    for s in range(S):
        F[0, s] = log_init[s] + E[0, s]
    for t in range(1, T):
        for s in range(S):
            acc = -np.inf
            for k in range(ptr[s], ptr[s + 1]):
                acc = _logsumexp2(acc, F[t - 1, idx[k]] + lp[k])
            F[t, s] = acc + E[t, s]
    B = np.full((T, S), -np.inf)
    for s in range(S):
        if end_mask[s]:
            B[T - 1, s] = 0.0
    for t in range(T - 2, -1, -1):
        for s2 in range(S):
            v = B[t + 1, s2] + E[t + 1, s2]
            if v == -np.inf:
                continue
            for k in range(ptr[s2], ptr[s2 + 1]):
                s = idx[k]
                B[t, s] = _logsumexp2(B[t, s], v + lp[k])
    logZ = -np.inf
    for s in range(S):
        logZ = _logsumexp2(logZ, F[T - 1, s] + B[T - 1, s])
    return F, B, logZ


@njit(cache=True)
def _viterbi(E, log_init, ptr, idx, lp, end_mask):
    T, S = E.shape
    V = np.full((T, S), -np.inf)
    BP = np.zeros((T, S), dtype=np.int64)
    for s in range(S):
        V[0, s] = log_init[s] + E[0, s]
    for t in range(1, T):
        for s in range(S):
            best = -np.inf
            arg = 0
            for k in range(ptr[s], ptr[s + 1]):
                v = V[t - 1, idx[k]] + lp[k]
                if v > best:
                    best = v
                    arg = idx[k]
            V[t, s] = best + E[t, s]
            BP[t, s] = arg
    best = -np.inf
    last = 0
    for s in range(S):
        if end_mask[s] and V[T - 1, s] > best:
            best = V[T - 1, s]
            last = s
    path = np.zeros(T, dtype=np.int64)
    path[T - 1] = last
    for t in range(T - 1, 0, -1):
        path[t - 1] = BP[t, path[t]]
    return path, best


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------

@dataclass
class Step:
    start: int
    end: int
    gap: float            # NaN for composition-only steps
    weight: float         # 1 for observed dyad pairs, posterior for virtual
    counts: np.ndarray    # per CATS
    low_info: bool
    observed_left: bool   # left dyad observed (counts toward n_dyads)
    observed_right: bool = False


def _pieces(length: int, msps: np.ndarray, long_msp: int) -> List[Tuple[int, int]]:
    cuts = [(int(s), int(e)) for s, e in msps if e - s > long_msp]
    out = []
    pos = 0
    for s, e in cuts:
        if s > pos:
            out.append((pos, s))
        pos = max(pos, e)
    if pos < length:
        out.append((pos, length))
    return out


def build_steps(fp_start, fp_end, fp_code, fp_post, fp_is_tf, fp_lambda, fp_trunc,
                piece: Tuple[int, int], nrl: float, mu_fn, lam_floor: Optional[float],
                params: NucParams = DEFAULT_PARAMS) -> List[Step]:
    """Steps of one pre-split piece from per-footprint calls."""
    a, b = piece
    centres = (fp_start + fp_end) / 2.0
    inside = (centres >= a) & (centres < b)
    dy_pos, dy_w, dy_obs = [], [], []
    for i in np.flatnonzero(inside):
        if fp_is_tf[i] or fp_trunc[i]:
            continue
        if fp_code[i] == MONO:
            dy_pos.append(centres[i])
            dy_w.append(1.0)
            dy_obs.append(True)
        elif fp_code[i] in (MULTI2, MULTI3):
            L = fp_end[i] - fp_start[i]
            m = 2 if fp_code[i] == MULTI2 else max(3, int(round((L - mu_fn(fp_lambda[i])) / nrl)) + 1)
            for k in range(m):
                dy_pos.append(fp_start[i] + (k + 0.5) * L / m)
                dy_w.append(float(fp_post[i]))
                dy_obs.append(False)
    order = np.argsort(dy_pos, kind='stable')
    dy_pos = np.asarray(dy_pos, dtype=float)[order]
    dy_w = np.asarray(dy_w, dtype=float)[order]
    dy_obs = np.asarray(dy_obs, dtype=bool)[order]
    cats = np.array([_cat_of(int(c), bool(t)) for c, t in zip(fp_code, fp_is_tf)], dtype=np.int64)
    cant = (fp_code == CANT_TELL) & ~fp_is_tf
    flank = params.seg_composition_flank
    # composition-only runs: short/TF footprints far from every dyad
    comp_idx = [i for i in np.flatnonzero(inside) if cats[i] in (1, 4)]
    far = []
    for i in comp_idx:
        c = centres[i]
        if len(dy_pos) == 0 or np.min(np.abs(dy_pos - c)) > flank:
            far.append(i)
    far_set = set(far)
    runs: List[List[int]] = []
    for i in sorted(far, key=lambda j: centres[j]):
        if runs and centres[i] - centres[runs[-1][-1]] <= flank:
            runs[-1].append(i)
        else:
            runs.append([i])
    steps: List[Step] = []
    assigned = np.zeros(len(fp_start), dtype=bool)
    for i in far_set:
        assigned[i] = True
    for j in range(len(dy_pos) - 1):
        s, e = dy_pos[j], dy_pos[j + 1]
        sel = inside & ~assigned & (centres >= s) & (centres < e)
        if j == 0:
            sel |= inside & ~assigned & (centres < s)
        if j == len(dy_pos) - 2:
            sel |= inside & ~assigned & (centres >= e)
        assigned |= sel
        cnt = np.zeros(len(CATS))
        for i in np.flatnonzero(sel):
            if cats[i] >= 0:
                cnt[cats[i]] += 1
        lam_sel = fp_lambda[sel & ~fp_is_tf]
        low = bool(params.seg_cant_tell_low_info and cant[sel].any())
        if lam_floor is not None and len(lam_sel) and np.nanmedian(lam_sel) < lam_floor:
            low = True
        steps.append(Step(int(round(s)), int(round(e)), float(e - s), float(min(dy_w[j], dy_w[j + 1])),
                          cnt, low, bool(dy_obs[j]), bool(dy_obs[j + 1])))
    for run in runs:
        # one step per footprint, spanning to the next one's start, so a long
        # dyad-free stretch has enough steps for the minimum durations
        run = sorted(run, key=lambda j: fp_start[j])
        for n_, i in enumerate(run):
            cnt = np.zeros(len(CATS))
            cnt[cats[i]] += 1
            s = int(fp_start[i])
            e = int(fp_start[run[n_ + 1]]) if n_ + 1 < len(run) else int(fp_end[i])
            steps.append(Step(s, max(e, int(fp_end[i]) if n_ + 1 == len(run) else e), float('nan'), 1.0,
                              cnt, False, False))
    steps.sort(key=lambda st: (st.start, st.end))
    return _partition(steps)


def _partition(steps: List[Step]) -> List[Step]:
    """Make steps disjoint: a dyad-gap step that spans composition-only steps
    is cut around them into gap-free pieces (its counts stay on the first
    piece), so decoded segments never overlap."""
    comp = [st for st in steps if not np.isfinite(st.gap)]
    if not comp:
        return steps
    out: List[Step] = []
    for st in steps:
        if not np.isfinite(st.gap):
            out.append(st)
            continue
        inner = sorted((c for c in comp if c.start < st.end and c.end > st.start), key=lambda c: c.start)
        if not inner:
            out.append(st)
            continue
        pieces = []
        pos = st.start
        for c in inner:
            if c.start > pos:
                pieces.append((pos, c.start))
            pos = max(pos, c.end)
        if pos < st.end:
            pieces.append((pos, st.end))
        for n_, (a, b) in enumerate(pieces):
            cnt = st.counts if n_ == 0 else np.zeros(len(CATS))
            out.append(Step(int(a), int(b), float('nan'), st.weight, cnt, st.low_info, False, False))
    out.sort(key=lambda x: (x.start, x.end))
    return out


# ---------------------------------------------------------------------------
# Emissions
# ---------------------------------------------------------------------------

def _gap_density_fn(cal):
    gd = cal.gap_density if cal is not None else {}
    if gd:
        edges = np.asarray(gd['edges'], float)
        dens = np.asarray(gd['density'], float)

        def f(g):
            k = np.clip(np.searchsorted(edges, g, side='right') - 1, 0, len(dens) - 1)
            return np.where((g >= edges[0]) & (g < edges[-1]), dens[k], 1e-8)
        return f
    return lambda g: np.where((g > 0) & (g < 300), 1.0 / 300.0, 1e-8)


def default_composition() -> dict:
    canon = np.array([0.70, 0.08, 0.10, 0.07, 0.05])
    sub = np.array([0.35, 0.30, 0.05, 0.05, 0.25])
    return {'cats': list(CATS), 'canonical': canon.tolist(), 'sub_rich': sub.tolist(),
            'rate_canonical': 1.0 / 190.0, 'rate_sub_rich': 1.6 / 190.0, 'weight_sub_rich': 0.05}


def _comp_loglik(steps: List[Step], comp: dict) -> np.ndarray:
    """(T, 3): canonical, sub-rich, dataset mixture (for X)."""
    from scipy.special import gammaln
    pc = np.clip(np.asarray(comp['canonical'], float), 1e-4, None)
    ps = np.clip(np.asarray(comp['sub_rich'], float), 1e-4, None)
    pc /= pc.sum()
    ps /= ps.sum()
    w = float(comp.get('weight_sub_rich', 0.05))
    pm = (1 - w) * pc + w * ps
    rc, rs = float(comp['rate_canonical']), float(comp['rate_sub_rich'])
    rm = (1 - w) * rc + w * rs
    out = np.zeros((len(steps), 3))
    for t, st in enumerate(steps):
        n = st.counts
        tot = n.sum()
        span = max(st.end - st.start, 1)
        for k, (p, r) in enumerate(((pc, rc), (ps, rs), (pm, rm))):
            lam = r * span
            out[t, k] = float((n * np.log(p)).sum()) + tot * np.log(lam) - lam - gammaln(tot + 1)
    return out


def emissions(steps: List[Step], space: StateSpace, cal, params: NucParams = DEFAULT_PARAMS,
              sigma_loc: float = 0.0) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Expanded-state log emissions (T, S) and the per-factor parts."""
    T = len(steps)
    nq = space.nq
    q = space.q_grid
    comp = cal.composition if (cal is not None and cal.composition) else default_composition()
    gap = np.array([st.gap for st in steps])
    has_gap = np.isfinite(gap)
    g = np.where(has_gap, gap, 0.0)
    gm = resolve_gap_model(cal, params)
    w = np.asarray(gm['grid_weights'], float)
    # the fitted regular SD is observed-level (it holds the edge noise at the
    # dataset's typical density); use its structural part plus this molecule's
    # own edge noise
    s_typ = float(gm.get('sigma_loc_typical') or 0.0) if params.seg_sigma_r is None else 0.0
    s_struct2 = max(gm['sigma_regular'] ** 2 - 2.0 * s_typ ** 2, 16.0)
    s2_col = np.where(space.tight, (params.seg_sigma_tight or 0.0) ** 2, s_struct2)[None, :]
    dens_R = np.zeros((T, len(q)))
    for k, wk in enumerate(w, start=1):
        var = k * s2_col + 2.0 * sigma_loc ** 2
        dens_R += wk * np.exp(-0.5 * (g[:, None] - k * q[None, :]) ** 2 / var) / np.sqrt(2 * np.pi * var)
    if params.seg_irregular == 'empirical':
        dens_I = _gap_density_fn(cal)(g)
    else:
        P = float(cal.nrl_pooled) if cal is not None else 189.0
        dens_I = np.full(T, 0.05 / 1450.0)
        for k, wk in enumerate(w, start=1):
            var = k * gm['sigma_irregular'] ** 2
            dens_I += 0.95 * wk * np.exp(-0.5 * (g - k * P) ** 2 / var) / np.sqrt(2 * np.pi * var)
    dens_X = np.full(T, _X_UNIFORM)
    # Steps touching a virtual dyad (from a multi(m) call) carry no gap
    # evidence: virtual dyads are equally spaced by construction, so they give
    # array continuity only and never regularity.
    observed = np.array([st.observed_left and st.observed_right for st in steps], dtype=bool)
    informative = has_gap & observed
    gR = np.where(informative[:, None], np.log(np.maximum(dens_R, 1e-300)), 0.0)
    gI = np.where(informative, np.log(np.maximum(dens_I, 1e-300)), 0.0)
    gX = np.where(informative, np.log(dens_X), 0.0)
    cl = _comp_loglik(steps, comp)
    low = np.array([st.low_info for st in steps])
    iX = np.where(low, np.log(0.8), np.log(0.2))
    iO = np.where(low, np.log(0.05), np.log(0.95))
    nb = len(space.base_order)
    Eb = np.zeros((T, nb))
    for b in range(nb):
        o, c = space.base_order[b], space.base_comp[b]
        if o < nq:
            Eb[:, b] = gR[:, o] + cl[:, c] + iO
        elif o == nq:
            Eb[:, b] = gI + cl[:, c] + iO
        else:
            Eb[:, b] = gX + cl[:, 2] + iX
    parts = {'gR': gR, 'gI': gI, 'gX': gX, 'comp': cl, 'iX': iX, 'iO': iO}
    return Eb[:, space.exp_base], parts


DEFAULT_GAP_MODEL = {'sigma_regular': 17.0, 'sigma_irregular': 30.0, 'grid_weights': [0.90, 0.07, 0.03],
                     'w_regular': None, 'nrl_sd': None, 'n_gaps': 0, 'sigma_loc_typical': 0.0}


def fit_gap_model(molecules, nrl_pooled: float, params: NucParams = DEFAULT_PARAMS) -> dict:
    """Dyad-gap model of a dataset (calibration).

    Deviations of adjacent mono-dyad gaps from each molecule's own median
    gap are fitted by two zero-mean Gaussians: the narrow one is the jitter
    of regular arrays, the broad one (widened by the spread of molecule
    NRLs) the irregular gap density. Grid weights are the shares of
    consecutive mono-dyad spacings near 1, 2 and 3 repeats (a can't-tell,
    short or long call between two mono calls hides a dyad).
    """
    from fiberhmm.nuc.spacing import eligible_dyads
    dev, meds, kcount = [], [], np.zeros(3)
    for m in molecules:
        d = eligible_dyads(m, nrl_pooled, params)
        g = d.gaps
        if len(g) < params.min_gaps_regularity:
            continue
        core = g[(g > 0.6 * nrl_pooled) & (g < 1.4 * nrl_pooled)]
        if len(core) < params.min_gaps_regularity:
            continue
        med = float(np.median(core))
        meds.append(med)
        dev.extend((core - med).tolist())
        sp = np.diff(d.centres)
        k = np.round(sp / med)
        ok = (k >= 1) & (k <= 3) & (np.abs(sp - k * med) <= 0.2 * med)
        for kk in (1, 2, 3):
            kcount[kk - 1] += int((ok & (k == kk)).sum())
    if len(dev) < 200:
        return dict(DEFAULT_GAP_MODEL)
    x = np.asarray(dev, float)
    s1, s2, wn = 10.0, 40.0, 0.5
    for _ in range(300):
        p1 = wn * np.exp(-0.5 * (x / s1) ** 2) / s1
        p2 = (1 - wn) * np.exp(-0.5 * (x / s2) ** 2) / s2
        r = p1 / np.maximum(p1 + p2, 1e-300)
        wn_new = float(r.mean())
        s1_new = float(np.sqrt((r * x * x).sum() / max(r.sum(), 1e-9)))
        s2_new = float(np.sqrt(((1 - r) * x * x).sum() / max((1 - r).sum(), 1e-9)))
        done = abs(s1_new - s1) < 1e-4 and abs(s2_new - s2) < 1e-4
        s1, s2, wn = max(s1_new, 3.0), max(s2_new, 3.0), wn_new
        if done:
            break
    nrl_sd = float(1.4826 * np.median(np.abs(np.asarray(meds) - np.median(meds))))
    gw = (kcount + 1.0) / (kcount.sum() + 3.0)
    return {'sigma_regular': float(min(s1, s2)), 'sigma_irregular': float(np.sqrt(max(s1, s2) ** 2 + nrl_sd ** 2)),
            'grid_weights': [float(v) for v in gw], 'w_regular': float(wn if s1 <= s2 else 1 - wn),
            'nrl_sd': nrl_sd, 'n_gaps': int(len(x)), 'sigma_loc_typical': 0.0}


def resolve_gap_model(cal, params: NucParams = DEFAULT_PARAMS) -> dict:
    """The gap model in use: the calibration's, with parameter overrides."""
    gm = dict(DEFAULT_GAP_MODEL)
    if cal is not None and getattr(cal, 'gap_model', None):
        gm.update({k: v for k, v in cal.gap_model.items() if v is not None})
    if params.seg_sigma_r is not None:
        gm['sigma_regular'] = float(params.seg_sigma_r)
    if params.seg_grid_weights is not None:
        gm['grid_weights'] = [float(v) for v in params.seg_grid_weights]
    return gm


# ---------------------------------------------------------------------------
# Decoding into segments
# ---------------------------------------------------------------------------

@dataclass
class Segment:
    start: int
    end: int
    state: int                 # REGULAR / IRREGULAR / LOWINFO
    sub_rich: bool
    posterior: float
    llr: float
    q_hat: Optional[float]
    n_dyads: int
    n_steps: int
    boundary: Tuple[float, float] = (float('nan'), float('nan'))   # change prob at left/right
    boundary_ci: Tuple[int, int, int, int] = (-1, -1, -1, -1)
    regularity: float = float('nan')

    @property
    def code(self) -> int:
        return self.state + (8 if self.sub_rich else 0)

    def as_dict(self):
        d = dict(self.__dict__)
        d['state_name'] = STATE_NAMES[self.state]
        d['code'] = self.code
        return d


def _class_of(space: StateSpace, base: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    o = space.base_order[base]
    order = np.where(o < space.nq, REGULAR, np.where(o == space.nq, IRREGULAR, LOWINFO))
    return order, space.base_comp[base]


def segment_steps(steps: List[Step], cal, params: NucParams = DEFAULT_PARAMS,
                  sigma_loc: float = 0.0) -> List[Segment]:
    """Segments of one piece from its steps."""
    T = len(steps)
    if T == 0:
        return []
    space = state_space(params)
    E, parts = emissions(steps, space, cal, params, sigma_loc)
    end_mask = space.exp_last.copy()
    F, B, logZ = _forward_backward(E, space.log_init, space.pred_ptr, space.pred_idx,
                                   space.pred_logp, end_mask)
    if not np.isfinite(logZ):
        return []
    post = np.exp(F + B - logZ)                       # (T, S)
    exp_order, exp_comp = _class_of(space, space.exp_base)
    p_order = np.stack([post[:, exp_order == k].sum(axis=1) for k in (REGULAR, IRREGULAR, LOWINFO)], axis=1)
    p_sub = post[:, exp_comp == 1].sum(axis=1)
    if params.seg_decode == 'viterbi':
        path, _ = _viterbi(E, space.log_init, space.pred_ptr, space.pred_idx, space.pred_logp, end_mask)
        order, comp = _class_of(space, space.exp_base[path])
    else:
        # posterior decoding: order class with the highest marginal (summed over
        # repeat and tightness), composition by its marginal; the run rules
        # below then apply the minimum sizes and the LLR test
        order = np.array([REGULAR, IRREGULAR, LOWINFO])[np.argmax(p_order, axis=1)]
        comp = (p_sub > 0.5).astype(np.int64)
    change = _change_probs(F, B, E, logZ, space, exp_order, exp_comp)

    runs = []
    t0 = 0
    for t in range(1, T + 1):
        if t == T or order[t] != order[t0] or comp[t] != comp[t0]:
            runs.append((t0, t))
            t0 = t
    gR, gI, gX, cl = parts['gR'], parts['gI'], parts['gX'], parts['comp']
    iX, iO = parts['iX'], parts['iO']

    def run_stats(a, b, o, c):
        # the same factors the decoder used: R and I with the run's composition,
        # X with the dataset mixture
        sl = slice(a, b)
        cc = float(cl[sl, c].sum())
        ll_R = gR[sl].sum(axis=0) + iO[sl].sum() + cc
        ll = {REGULAR: float(ll_R.max()), IRREGULAR: float(gI[sl].sum() + iO[sl].sum() + cc),
              LOWINFO: float(gX[sl].sum() + iX[sl].sum() + cl[sl, 2].sum())}
        llr_order = ll[o] - max(v for k, v in ll.items() if k != o)
        llr_comp = float(cl[sl, c].sum() - cl[sl, 1 - c].sum())
        gap_steps = [st for st in steps[a:b] if np.isfinite(st.gap)]
        n_dy = sum(st.observed_left for st in gap_steps) + (gap_steps[-1].observed_right if gap_steps else 0)
        return ll_R, llr_order, llr_comp, int(n_dy), steps[a].start, steps[b - 1].end

    # 1) relabel each Viterbi run by the LLR and minimum-size rules
    labelled = []
    for (a, b) in runs:
        o, c = int(order[a]), int(comp[a])
        _, llr_o, llr_c, n_dy, start, end = run_stats(a, b, o, c)
        span = end - start
        if o == REGULAR and (llr_o < params.seg_merge_llr or n_dy < params.seg_min_dyads
                             or span < params.seg_min_bp):
            o = IRREGULAR
        elif o == LOWINFO and llr_o < params.seg_merge_llr:
            o = IRREGULAR
        sub = c == 1 and not (llr_c < params.seg_merge_llr or span < params.seg_min_bp
                              or (b - a) < params.seg_min_steps_sub)
        labelled.append([a, b, o, int(sub)])
    # 2) merge neighbours that now share a label
    merged = []
    for r in labelled:
        if merged and merged[-1][2] == r[2] and merged[-1][3] == r[3]:
            merged[-1][1] = r[1]
        else:
            merged.append(r)
    # 3) final statistics on the merged runs
    segs: List[Segment] = []
    for a, b, o, sub in merged:
        ll_R, llr_o, _, n_dy, start, end = run_stats(a, b, o, sub)
        sl = slice(a, b)
        col = {REGULAR: 0, IRREGULAR: 1, LOWINFO: 2}[o]
        pst = float(p_order[sl, col].mean())
        if sub:
            pst = float(min(pst, p_sub[sl].mean()))
        q_hat = float(space.q_grid[int(np.argmax(ll_R))]) if o == REGULAR else None
        seg = Segment(start, end, o, bool(sub), pst, float(llr_o), q_hat, n_dy, b - a)
        pl = float(change[a - 1]) if a > 0 else float('nan')
        pr = float(change[b - 1]) if b - 1 < len(change) else float('nan')
        seg.boundary = (pl, pr)
        seg.boundary_ci = (_ci(steps, change, a - 1, left=True), _ci(steps, change, a - 1, left=False),
                           _ci(steps, change, b - 1, left=True), _ci(steps, change, b - 1, left=False))
        segs.append(seg)
    return segs


def _change_probs(F, B, E, logZ, space, exp_order, exp_comp) -> np.ndarray:
    """P(class(t) != class(t+1)) from pairwise marginals."""
    cls = (exp_order * 2 + exp_comp).astype(np.int64)
    return _change_probs_nb(F, B, E, logZ, space.pred_ptr, space.pred_idx, space.pred_logp, cls)


@njit(cache=True)
def _change_probs_nb(F, B, E, logZ, ptr, idx, lp, cls):
    T, S = E.shape
    out = np.zeros(max(T - 1, 0))
    for t in range(T - 1):
        tot_same = 0.0
        for s2 in range(S):
            v = B[t + 1, s2] + E[t + 1, s2] - logZ
            if v == -np.inf:
                continue
            for k in range(ptr[s2], ptr[s2 + 1]):
                s = idx[k]
                if cls[s] == cls[s2]:
                    tot_same += np.exp(F[t, s] + lp[k] + v)
        out[t] = max(0.0, 1.0 - tot_same)
    return out


def _ci(steps: List[Step], change: np.ndarray, t: int, left: bool) -> int:
    """Molecular bp bound of the boundary interval: gaps around boundary t
    with change probability >= 0.1."""
    if t < 0 or t >= len(change):
        return -1
    if left:
        k = t
        while k - 1 >= 0 and change[k - 1] >= 0.1:
            k -= 1
        return int(steps[k].end)
    k = t
    while k + 1 < len(change) and change[k + 1] >= 0.1:
        k += 1
    return int(steps[k + 1].start)


def segment_molecule(length: int, msps: np.ndarray, fp: Dict[str, np.ndarray], cal,
                     params: NucParams = DEFAULT_PARAMS) -> Tuple[List[Segment], bool]:
    """Segments of a molecule from its per-footprint calls.

    ``fp`` holds arrays ``start, end, coarse, posterior, is_tf, lambda,
    truncated``. Returns ``(segments, too_short)``.
    """
    if length < params.seg_min_molecule_bp:
        return [], True
    nrl = float(cal.nrl_pooled) if cal is not None else 189.0
    mu_fn = (lambda lam: float(cal.curve.mu_at(lam))) if (cal is not None and cal.curve) else (lambda lam: 147.0)
    lam_floor = cal.lambda_floor if cal is not None else None
    sigma_loc = 0.0
    if cal is not None and cal.sigma_loc and len(fp['lambda']):
        lam_med = float(np.nanmedian(fp['lambda'][~fp['is_tf']])) if (~fp['is_tf']).any() else 0.3
        sigma_loc = float(cal.sigma_loc_at(lam_med))
    segs: List[Segment] = []
    centres = (fp['start'] + fp['end']) / 2.0
    for piece in _pieces(length, msps, params.long_msp_bp):
        steps = build_steps(fp['start'], fp['end'], fp['coarse'], fp['posterior'], fp['is_tf'],
                            fp['lambda'], fp['truncated'], piece, nrl, mu_fn, lam_floor, params)
        got = segment_steps(steps, cal, params, sigma_loc)
        if not got:
            # calls but nothing measurable (can't-tell, truncated): low information
            inside = (centres >= piece[0]) & (centres < piece[1]) & ~fp['is_tf']
            if inside.any():
                got = [lowinfo_segment(int(fp['start'][inside].min()), int(fp['end'][inside].max()))]
        segs.extend(got)
    return segs, False


def lowinfo_segment(start: int, end: int) -> Segment:
    """A low-information segment with no decoded evidence."""
    return Segment(start, end, LOWINFO, False, 1.0, 0.0, None, 0, 0)


def fit_composition(molecules, cal, params: NucParams = DEFAULT_PARAMS, max_iter: int = 30) -> dict:
    """EM for canonical and sub-rich composition over the molecules' steps.

    Two-component mixture of (multinomial x Poisson-rate) per step. Starts
    from the pooled frequencies (canonical) and a short/TF-enriched copy
    (sub-rich); the sub-rich component is kept enriched in short + TF.
    """
    from scipy.special import gammaln
    from fiberhmm.nuc.molecule import footprint_calls
    counts, spans = [], []
    for m in molecules:
        fp = footprint_calls(m, params, cal)
        if fp is None:
            continue
        for piece in _pieces(m.length, m.msps, params.long_msp_bp):
            for st in build_steps(fp['start'], fp['end'], fp['coarse'], fp['posterior'], fp['is_tf'],
                                  fp['lambda'], fp['truncated'], piece, cal.nrl_pooled,
                                  lambda lam: float(cal.curve.mu_at(lam)), None, params):
                counts.append(st.counts)
                spans.append(max(st.end - st.start, 1))
    d = default_composition()
    if len(counts) < 100:
        return d
    N = np.asarray(counts)
    Sp = np.asarray(spans, float)
    pooled = N.sum(axis=0) + 1.0
    pc = pooled / pooled.sum()
    ps = pc * np.array([0.5, 3.0, 0.5, 0.5, 3.0])
    ps /= ps.sum()
    rc = N.sum() / Sp.sum()
    rs = 1.6 * rc
    w = 0.05
    tot = N.sum(axis=1)
    for _ in range(max_iter):
        def ll(p, r):
            lam = r * Sp
            return (N * np.log(p)).sum(axis=1) + tot * np.log(lam) - lam - gammaln(tot + 1)
        a = np.log(1 - w) + ll(pc, rc)
        b = np.log(w) + ll(ps, rs)
        m = np.maximum(a, b)
        r_sub = np.exp(b - m) / (np.exp(a - m) + np.exp(b - m))
        w_new = float(np.clip(r_sub.mean(), 1e-3, 0.5))
        pc_new = ((1 - r_sub)[:, None] * N).sum(axis=0) + 1.0
        ps_new = (r_sub[:, None] * N).sum(axis=0) + 1.0
        pc_new /= pc_new.sum()
        ps_new /= ps_new.sum()
        rc = float(((1 - r_sub) * tot).sum() / max(((1 - r_sub) * Sp).sum(), 1.0))
        rs = float((r_sub * tot).sum() / max((r_sub * Sp).sum(), 1.0))
        if ps_new[1] + ps_new[4] <= pc_new[1] + pc_new[4]:
            break   # keep the enriched start: the components would swap meaning
        converged = abs(w_new - w) < 1e-4
        pc, ps, w = pc_new, ps_new, w_new
        if converged:
            break
    return {'cats': list(CATS), 'canonical': pc.tolist(), 'sub_rich': ps.tolist(),
            'rate_canonical': float(rc), 'rate_sub_rich': float(max(rs, rc)), 'weight_sub_rich': float(w),
            'n_steps': int(len(N))}
