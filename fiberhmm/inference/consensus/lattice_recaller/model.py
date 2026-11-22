"""Local lattice model, EM prevalence, learned internal spots, support and resolution.

Port of the 25 Sep 2026 prototype (lattice_local2/3/4.py). For one overlap group of classes and one channel, each
molecule gets a log-likelihood (relative to all-accessible) for every hypothesis: each class, broader protection
(one interval covering every class edge box), other shape (any other single protected interval) and accessible.
Inside a hypothesis's protected interval sites are scored at the protected rate; an accessible linker (nearest site
plus linker_bp) is required beyond each edge ('both') or beyond at least one ('either'); sites further out are of
unknown state (accessible with probability f, the channel's accessible fraction). Class buckets weight each
configuration by the edge positions (bp) it covers inside the class boxes ('bp'), or average over configurations
('configurations'); other buckets average over configurations.
"""
from __future__ import annotations

import hashlib
import math

import numpy as np

EM_WEIGHT_FLOOR = 1e-12
LABEL_SEMANTICS = 'likelihood_ratio_v2'


def _hash2(uid):
    return int(hashlib.sha1(uid.encode()).hexdigest(), 16) % 2


def em(M, iters=1000):
    """Mixture weights and posteriors for a molecules x hypotheses log-likelihood matrix."""
    k = M.shape[1]; w = np.full(k, 1/k)
    for _ in range(iters):
        A = M + np.log(np.maximum(w, EM_WEIGHT_FLOOR)); A -= A.max(1, keepdims=True); P = np.exp(A); P /= P.sum(1, keepdims=True); nw = P.mean(0)
        if np.abs(nw - w).max() < 1e-8:
            w = nw; break
        w = nw
    A = M + np.log(np.maximum(w, EM_WEIGHT_FLOOR)); A -= A.max(1, keepdims=True); P = np.exp(A); P /= P.sum(1, keepdims=True)
    return w, P


def posteriors(M, w):
    """Per-molecule posteriors of a molecules x hypotheses log-likelihood matrix under fixed mixture weights w (the
    E-step of em(), with its weight floor). Rows sum to 1."""
    A = M + np.log(np.maximum(w, EM_WEIGHT_FLOOR)); A -= A.max(1, keepdims=True); P = np.exp(A); P /= P.sum(1, keepdims=True)
    return P


def fit_channel_heldout(units, gs, f, opt, fit_uids):
    """fit_channel on the molecules named in fit_uids only (cross-fitting, recaller.fold_weights='heldout'), then every
    molecule's posterior under that fit: the fitted geometry (edge contraction), learned spots and EM weights are fixed
    and applied to all units. Prevalence (and its tiers), support gain and resolution are those of the fitted molecules (n); n_scored
    counts every molecule that received a posterior."""
    fit_units = [u for u in units if u['uid'] in fit_uids]
    res = fit_channel(fit_units, gs, f, opt) if fit_units else None
    if res is None:
        return None
    gs2 = res['gs']; prof = [dict(s) for s in res['spots']]; w = res['w']
    sc = Scorer(units, gs2, f, opt)
    M, keep = sc.rows(prof)
    if not len(M):
        return None
    P = posteriors(M, w); log_bfs = log_bayes_factors(M, w)
    # Prevalence tiers stay those of the fitted molecules (res['tiers']), like the EM weights.
    return dict(res, P=P, M=M, log_bf=log_bfs, units=keep, n_scored=len(M),
                calls=[dict(classes=[molecule_edges(it, c, gs2[c], prof[c]) for c in range(len(gs2))], broader=it['broader'],
                            tiers=molecule_tiers(u, gs2, prof))
                       for it, u in zip(sc.items, sc.keep)])


def loglik(M, w):
    X = M + np.log(np.maximum(w, EM_WEIGHT_FLOOR)); m = X.max(1)
    return float(np.sum(m + np.log(np.exp(X - m[:, None]).sum(1))))


def _logsumexp(v):
    m = v.max(); return float(m + math.log(np.exp(v - m).sum()))


def log_bayes_factors(M, w):
    """Class likelihood / fitted mixture of all other hypotheses, in log space.

    Use the same weight floor as EM, normalizing the alternatives by their own
    sum. Subtracting clipped posterior/prior odds loses evidence when either
    rounds to zero or one. Prefix/suffix sums avoid that subtraction, including
    when one hypothesis dominates, in O(molecules * hypotheses) work.
    """
    logw = np.log(np.maximum(w, EM_WEIGHT_FLOOR))
    if M.ndim != 2 or M.shape[1] < 2 or logw.shape != (M.shape[1],):
        raise ValueError('Bayes factors require a matrix and at least two hypothesis weights')
    X = M + logw
    def excluding_each(X):
        left = np.logaddexp.accumulate(X, axis=-1)
        right = np.logaddexp.accumulate(X[..., ::-1], axis=-1)[..., ::-1]
        out = np.empty_like(X)
        out[..., 0], out[..., -1] = right[..., 1], left[..., -2]
        out[..., 1:-1] = np.logaddexp(left[..., :-2], right[..., 2:])
        return out
    return M - (excluding_each(X) - excluding_each(logw))


def _prepare(u, gs, f, opt):
    """Everything about one molecule that does not depend on learned spots, or None if it does not span the group.

    Per class: the (i, j) configurations it scores (protected sites i..j-1), their spot-free scores s0 and their log
    weights (bp weighting over the class boxes, or uniform), so that a spot profile only adds a cumulative-sum term."""
    L0 = min(g['L'][0] for g in gs); R1 = max(g['R'][1] for g in gs); lo, hi = L0 - opt.flank_bp, R1 + opt.flank_bp
    pos = u['pos']
    if not (pos.min() < lo and pos.max() >= hi):
        return None
    sel = (pos >= lo) & (pos < hi); n = int(sel.sum())
    if n < 2:
        return None
    p = pos[sel]; dP = u['d'][sel]; h = u['hit'][sel]; pa = u['pa'][sel]
    dU = np.log(f + (1 - f)*np.exp(dP))
    SP = np.concatenate([[0.], np.cumsum(dP)]); SU = np.concatenate([[0.], np.cumsum(dU)]); tot = SU[-1]
    i = np.arange(n)[:, None]; j = np.arange(1, n + 1)[None, :]; valid = j > i
    pm = np.r_[lo - 1, p]; l0 = np.where(i == 0, lo, pm[i] + 1); l1 = p[i]
    r0 = p[j - 1] + 1; r1 = np.where(j < n, np.r_[p, hi][np.minimum(j, n)], hi)
    a = np.minimum(np.searchsorted(p, p - opt.linker_bp, side='left'), np.maximum(np.arange(n) - 1, 0))
    gl = np.where(np.arange(n) >= 1, SU[np.arange(n)] - SU[a], 0.)
    b = np.searchsorted(p, p + opt.linker_bp, side='left'); jj = np.arange(1, n + 1); b = np.minimum(np.maximum(b, np.minimum(jj + 1, n)), n)
    gr = np.where(jj < n, SU[b] - SU[jj], 0.)
    if opt.linker == 'both':
        base = tot - gl[:, None] - gr[None, :]
    else:
        both = tot - gl[:, None] - gr[None, :]; left = tot - gl[:, None] + 0*gr[None, :]; right = tot + 0*gl[:, None] - gr[None, :]
        base = np.logaddexp(np.logaddexp(both + math.log(.5), left + math.log(.25)), right + math.log(.25))
    s0 = base + (SP[j] - SP[i]) - (SU[j] - SU[i])

    def lme(s, mask):
        if not mask.any():
            return -1e3
        v = s[mask]; m = v.max(); return float(m + math.log(np.mean(np.exp(v - m))))

    anycls = np.zeros_like(valid); cms = []
    for g in gs:
        cm = valid & (l0 <= g['L'][1]) & (l1 >= g['L'][0]) & (r0 <= g['R'][1]) & (r1 >= g['R'][0]); cms.append(cm); anycls |= cm
    br = np.zeros_like(valid)
    for g in gs:
        br |= valid & (l1 <= g['L'][1]) & (r0 >= g['R'][0])
    br &= ~anycls
    full = lambda x: np.broadcast_to(x, valid.shape)
    other = valid & ~anycls & ~br
    cls = []
    for c, (g, cm) in enumerate(zip(gs, cms)):
        Lw, Rw = g['L'][1] - g['L'][0] + 1, g['R'][1] - g['R'][0] + 1
        if opt.class_weighting == 'bp':
            wl = np.clip(np.minimum(l1, g['L'][1]) - np.maximum(l0, g['L'][0]) + 1, 0, None)
            wr = np.clip(np.minimum(r1, g['R'][1]) - np.maximum(r0, g['R'][0]) + 1, 0, None)
            W = np.broadcast_to(wl*wr, cm.shape).astype(float)
            use = cm & (W > 0)
            logw = np.log(W[use]) - math.log(Lw*Rw)
        else:
            use = cm
            logw = np.full(int(use.sum()), -math.log(max(int(use.sum()), 1)))
        ii, jc = np.nonzero(use)
        # Stored once per molecule, class and configuration for the whole fit: int32 site indices and the summed
        # spot-free score (s0 + log weight, the sum _class_vector always formed first; same values).
        cls.append((ii.astype(np.int32), (jc + 1).astype(np.int32), s0[use] + logw))
    # Each configuration's edge ranges (left edge in [l0, l1], right edge in [r0, r1]) depend only on its first
    # protected site i (l0, l1) and its end j (r0, r1): kept as per-site vectors, not per configuration and class.
    bounds = (full(l0)[:, 0].astype(float), full(l1)[:, 0].astype(float),
              full(r0)[0, :].astype(float), full(r1)[0, :].astype(float))
    # The broader-protection configuration that best explains the molecule (its edge ranges), if any.
    broader = None
    if br.any():
        k = np.argmax(np.where(br, s0, -np.inf)); bi, bj = np.unravel_index(k, br.shape)
        e = [float(full(x)[bi, bj]) for x in (l0, l1, r0, r1)]
        broader = ((e[0] + e[1])/2, (e[2] + e[3])/2, (round(e[0]), round(e[1])), (round(e[2]), round(e[3])))
    # The best other-shape configuration (edge ranges), for the loosest prevalence tier.
    oshape = None
    if other.any():
        k = np.argmax(np.where(other, s0, -np.inf)); oi, oj = np.unravel_index(k, other.shape)
        e = [float(full(x)[oi, oj]) for x in (l0, l1, r0, r1)]
        oshape = ((e[0] + e[1])/2, (e[2] + e[3])/2, (round(e[0]), round(e[1])), (round(e[2]), round(e[3])))
    c0 = min(g['span'][0] for g in gs); c1 = max(g['span'][1] for g in gs); ins = (p >= c0) & (p < c1)
    return dict(p=p, dP=dP, h=h, pa=pa, cls=cls, bounds=bounds, broader=broader, oshape=oshape,
                fixed=[lme(s0, br), lme(s0, other), float(dU[~ins].sum())])


def _class_vector(item, c, pc=None):
    """Log weight + log-likelihood of each configuration of class c (with the learned spots pc), or None."""
    ii, jc, v0 = item['cls'][c]
    if not len(v0):
        return None
    v = v0
    if pc:
        ppc = np.array([pc.get(int(x), np.nan) for x in item['p']]); use = ~np.isnan(ppc)
        if use.any():
            p, h, pa = ppc[use], item['h'][use], item['pa'][use]
            delta = np.zeros(len(item['p'])); delta[use] = np.where(h, np.log(p) - np.log(pa), np.log1p(-p) - np.log1p(-pa)) - item['dP'][use]
            D = np.concatenate([[0.], np.cumsum(delta)]); v = v + D[jc] - D[ii]
    return v


def _class_score(item, c, pc=None):
    v = _class_vector(item, c, pc)
    return -1e3 if v is None else _logsumexp(v)


def molecule_edges(item, c, g, pc=None):
    """The molecule's own call for class c: the best-supported configuration's edge ranges, intersected with the class
    edge boxes, each taken at its midpoint. None if the molecule has no configuration of the class."""
    v = _class_vector(item, c, pc)
    if v is None:
        return None
    k = int(np.argmax(v)); i, j = int(item['cls'][c][0][k]), int(item['cls'][c][1][k]) - 1
    el0, el1, er0, er1 = item['bounds']; l0, l1, r0, r1 = float(el0[i]), float(el1[i]), float(er0[j]), float(er1[j])
    a0, a1 = max(l0, g['L'][0]), min(l1, g['L'][1]); b0, b1 = max(r0, g['R'][0]), min(r1, g['R'][1])
    if a0 > a1: a0, a1 = l0, l1
    if b0 > b1: b0, b1 = r0, r1
    left, right = (a0 + a1)/2, (b0 + b1)/2
    # Midpoint edges, and each edge's range (how far it can move before a mark says otherwise).
    return (round(left), round(max(right, left + 1)), (round(a0), round(a1)), (round(b0), round(b1)))


class Scorer:
    """Per-molecule configuration scores for one group and channel, computed once; spot profiles are applied on top."""

    def __init__(self, units, gs, f, opt):
        self.items, self.keep = [], []
        for u in units:
            it = _prepare(u, gs, f, opt)
            if it is not None:
                self.items.append(it); self.keep.append(u)
        self.k = len(gs)
        self.base = np.array([[_class_score(it, c) for c in range(self.k)] + it['fixed'] for it in self.items], float).reshape(-1, self.k + 3)

    def subset(self, units):
        """Reuse immutable configurations in a fold, retaining its original molecule order."""
        indices = {id(u): i for i, u in enumerate(self.keep)}
        rows = [indices[id(u)] for u in units if id(u) in indices]
        view = object.__new__(Scorer)
        view.k = self.k
        view.items = [self.items[i] for i in rows]
        view.keep = [self.keep[i] for i in rows]
        view.base = self.base[rows]
        return view

    def rows(self, prof=None):
        M = self.base.copy()
        for c in range(self.k):
            pc = prof[c] if prof else None
            if pc:
                keys = np.fromiter(pc.keys(), float)
                for r, it in enumerate(self.items):
                    if np.isin(it['p'], keys).any():
                        M[r, c] = _class_score(it, c, pc)
        return M, self.keep


def joint(u, gs, f, opt, prof=None):
    """[class_1..k, broader, other, accessible] log-likelihoods for one molecule, or None if it does not span the group."""
    it = _prepare(u, gs, f, opt)
    if it is None:
        return None
    return [_class_score(it, c, prof[c] if prof else None) for c in range(len(gs))] + it['fixed']


def rows_of(units, gs, f, opt, prof):
    return Scorer(units, gs, f, opt).rows(prof)


def learn(units, gs, f, opt, allowed, scorer=None):
    """EM alternating with learned protected-state mark rates at the allowed interior positions of each class."""
    sc = scorer or Scorer(units, gs, f, opt)
    prof = [dict() for _ in gs]
    if any(allowed):
        arr = [np.fromiter(a, float) if a else None for a in allowed]
        for _ in range(opt.spot_iterations):
            M, keep = sc.rows(prof)
            if not len(M):
                break
            _, P = em(M); new = []
            for c in range(len(gs)):
                num, den, pp0 = {}, {}, {}
                if arr[c] is not None:
                    for u, pc in zip(keep, P[:, c]):
                        m = np.isin(u['pos'], arr[c])
                        for x, hh, q in zip(u['pos'][m], u['hit'][m], u['pp'][m]):
                            x = int(x); num[x] = num.get(x, 0.) + pc*hh; den[x] = den.get(x, 0.) + pc; pp0.setdefault(x, []).append(q)
                new.append({x: float(min(opt.spot_cap, max(np.median(pp0[x]), (num[x] + opt.spot_pseudo_units*np.median(pp0[x]))/(den[x] + opt.spot_pseudo_units))))
                            for x in den})
            prof = new
    M, keep = sc.rows(prof)
    if not len(M):
        return None, None, keep, prof
    w, P = em(M)
    return w, P, keep, prof


def interior_positions(units, g, edge_bp):
    inside = (lambda x: g['L'][0] <= x < g['R'][1]) if g.get('contracted') else (lambda x: True)
    return {int(x) for u in units for x in u['pos'] if g['span'][0] + edge_bp <= x < g['span'][1] - edge_bp and inside(x)}


def propose_edges(keep, P, gs, opt):
    """Per class, the edge boxes pulled inward past sites its members almost always mark (Codex's rule, 26 Sep 2026).

    Scanning from each end of the class span inward, a run of sites whose posterior-weighted member mark rate exceeds
    edge_contraction_rate (sites with >= edge_minimum_members members) is moved outside the box, up to the first site
    members leave unmarked. Only sites inside the existing edge box qualify, so a class is never consumed.
    Returns [(contracted class or None, [moves])]."""
    out = []
    for c, g in enumerate(gs):
        stats = {}
        for u, pc in zip(keep, P[:, c]):
            m = (u['pos'] >= g['span'][0]) & (u['pos'] < g['span'][1])
            for x, h in zip(u['pos'][m], u['hit'][m]):
                a = stats.setdefault(int(x), [0., 0.]); a[0] += pc*h; a[1] += pc
        v = sorted((x, k/n) for x, (k, n) in stats.items() if n >= opt.edge_minimum_members)
        L, R, moves = list(g['L']), list(g['R']), []
        for side, seq in (('left', v), ('right', v[::-1])):
            run = []
            for x, r in seq:
                if r > opt.edge_contraction_rate:
                    run.append(x)
                else:
                    break
            if not run or len(run) == len(seq):
                continue
            prot = seq[len(run)][0]
            if side == 'left':
                x = max(run)
                if x > g['L'][1] or prot >= g['R'][0]:
                    continue
                new = [max(L[0], x + 1), min(L[1], prot)]
                if new[0] > new[1]:
                    continue
                moves.append(('left', L, new)); L = new
            else:
                x = min(run)
                if x < g['R'][0] or prot <= g['L'][1]:
                    continue
                new = [max(R[0], prot + 1), min(R[1], x)]
                if new[0] > new[1]:
                    continue
                moves.append(('right', R, new)); R = new
        out.append((dict(g, L=L, R=R, contracted=True) if moves and L[1] < R[0] else None, moves))
    return out


def _aligned_rows(units, sc_a, gs_b, f, opt):
    """Rows for two geometries of the same group over the molecules both score (contracted boxes can admit more)."""
    Ma, ka = sc_a.rows(); Mb, kb = rows_of(units, gs_b, f, opt, None)
    ids = {u['uid'] for u in ka} & {u['uid'] for u in kb}
    ia = [i for i, u in enumerate(ka) if u['uid'] in ids]; ib = [i for i, u in enumerate(kb) if u['uid'] in ids]
    return Ma[ia], Mb[ib]


def contract_edges(units, gs, f, opt):
    """Per-channel edge contraction, kept for a class only if proposed on both halves of the molecules and the held-out
    likelihood of the whole group rises by >= edge_gain_nats (summed over both halves). In each fold the contracted
    boxes are proposed from that fold's training molecules and scored on its held-out molecules; after acceptance the
    final boxes are the proposal from all molecules. Returns (gs, records)."""
    records = [''] * len(gs)
    none = [set() for _ in gs]
    sc = Scorer(units, gs, f, opt)
    _, P, keep, _ = learn(units, gs, f, opt, none, scorer=sc)
    if P is None:
        return gs, records
    full = propose_edges(keep, P, gs, opt)
    cands = [c for c, (g2, _) in enumerate(full) if g2 is not None]
    if not cands:
        return gs, records
    fold = np.array([_hash2(u['uid']) for u in units]); gain = {c: 0. for c in cands}; both = {c: True for c in cands}
    for k in (0, 1):
        tr = [u for u, z in zip(units, fold) if z != k]; te = [u for u, z in zip(units, fold) if z == k]
        sc_tr, sc_te = sc.subset(tr), sc.subset(te)
        _, Ptr, ktr, _ = learn(tr, gs, f, opt, none, scorer=sc_tr)
        if Ptr is None or len(ktr) < 10:
            return gs, records
        prop = propose_edges(ktr, Ptr, gs, opt)
        for c in cands:
            # The fold's boundaries come from its training molecules only; the held-out half never chooses them.
            both[c] &= prop[c][0] is not None
            if prop[c][0] is None:
                continue
            alt = [prop[c][0] if i == c else g for i, g in enumerate(gs)]
            M0, M1 = _aligned_rows(tr, sc_tr, alt, f, opt); T0, T1 = _aligned_rows(te, sc_te, alt, f, opt)
            if len(M0) < 10 or not len(T0):
                both[c] = False; continue
            w0, _ = em(M0); w1, _ = em(M1)
            gain[c] += loglik(T1, w1) - loglik(T0, w0)
    out = list(gs)
    for c in cands:
        desc = ';'.join(f'{s}:{o[0]}-{o[1]}>{n[0]}-{n[1]}' for s, o, n in full[c][1])
        if both[c] and gain[c] >= opt.edge_gain_nats:
            out[c] = full[c][0]; records[c] = f'{desc} (+{gain[c]:.1f} nats)'
        else:
            records[c] = f'rejected {desc} ({gain[c]:+.1f} nats{"" if both[c] else ", not proposed in both halves"})'
    return out, records


def select_spots(units, gs, f, opt, scorer=None):
    """Interior positions whose learned rate raises held-out likelihood (both folds, summed) by >= spot_gain_nats."""
    fold = np.array([_hash2(u['uid']) for u in units])
    allowed = [interior_positions(units, g, opt.spot_edge_bp) for g in gs]
    gain = [dict() for _ in gs]; seen = [dict() for _ in gs]
    sc = scorer or Scorer(units, gs, f, opt)
    for k in (0, 1):
        tr = [u for u, z in zip(units, fold) if z != k]; te = [u for u, z in zip(units, fold) if z == k]
        sc_tr, sc_te = sc.subset(tr), sc.subset(te)
        _, _, _, prof = learn(tr, gs, f, opt, allowed, scorer=sc_tr)
        M0, _ = sc_tr.rows(); T0, _ = sc_te.rows()
        if len(M0) < 10 or not len(T0):
            return [set() for _ in gs]
        w0, _ = em(M0); base = loglik(T0, w0)
        want = np.fromiter({x for pc in prof for x in pc}, float); pps = {}
        for u in tr:
            m = np.isin(u['pos'], want)
            for xx, q in zip(u['pos'][m], u['pp'][m]):
                pps.setdefault(int(xx), []).append(q)
        for c in range(len(gs)):
            for x, v in prof[c].items():
                med = np.median(pps[x])
                if v <= med + 0.02:
                    continue
                one = [dict() for _ in gs]; one[c] = {x: v}
                Mq, _ = sc_tr.rows(one); wq, _ = em(Mq); Tq, _ = sc_te.rows(one)
                gain[c][x] = gain[c].get(x, 0.) + loglik(Tq, wq) - base; seen[c][x] = seen[c].get(x, 0) + 1
    return [{x for x, gv in gain[c].items() if gv >= opt.spot_gain_nats and seen[c][x] == 2} for c in range(len(gs))]




CORE_EDGE_TOLERANCE_BP = 5   # a run edge within this far inside the core still lines up with the class edge
CORE_MIN_NATS = 3.           # a clean core must carry at least this much evidence of protection (~20:1)


def core_run(u, g, spots=()):
    """The molecule's unmodified run over class g's core ([L1, R0], or the span's central half when the edge boxes leave
    under 8 bp between them), ignoring
    marks at learned spot positions: (clean, left, right). clean = the core has sites and none is marked (spots
    excepted) and its unmarked sites carry >= CORE_MIN_NATS of evidence for protection (a sparse lattice can leave a
    core unmarked by chance); left/right = the run's edges (nearest marks outside, else the molecule's lattice ends)."""
    s0, s1 = g['span']
    c0, c1 = (g['L'][1], g['R'][0]) if g['R'][0] - g['L'][1] >= 8 else (s0 + (s1 - s0)/4, s1 - (s1 - s0)/4)
    pos, hit = u['pos'], u['hit']
    keep = ~np.isin(pos, list(spots)) if spots else np.ones(len(pos), bool)
    incore = (pos >= c0) & (pos < c1)
    if not incore.any():
        return False, None, None
    evidence = float(np.sum(np.log1p(-u['pp'][incore & keep]) - np.log1p(-u['pa'][incore & keep])))
    clean = not bool((hit & incore & keep).any()) and evidence >= CORE_MIN_NATS
    marks = pos[hit & keep]
    lo = marks[marks < c0]; hi = marks[marks >= c1]
    return clean, (int(lo.max()) + 1 if len(lo) else int(pos.min())), (int(hi.min()) if len(hi) else int(pos.max()) + 1)


def tier_match(u, g, spots=()):
    """A non-member molecule's looser-tier call for class g from its own lattice: ('edge', call) when the core is
    clean and the protected run has an edge lined up with the class edge (between the edge box and
    CORE_EDGE_TOLERANCE_BP inside the core) on at least one side; ('loose', call) when the core is clean and covered
    by any protection; else None. call = (left, right, (left range), (right range))."""
    clean, a, b = core_run(u, g, spots)
    if not clean:
        return None
    t = CORE_EDGE_TOLERANCE_BP
    left = g['L'][0] <= a <= g['L'][1] + t; right = g['R'][0] - t <= b <= g['R'][1]
    s0, s1 = round(g['span'][0]), round(g['span'][1])
    if left or right:
        return 'edge', (a if left else s0, b if right else s1, (a, a) if left else tuple(g['L']), (b, b) if right else tuple(g['R']))
    return 'loose', (max(a, s0), min(b, s1), tuple(g['L']), tuple(g['R']))


def prevalence_tiers(units, P, w, gs, prof, bf_threshold, log_bfs=None):
    """Three prevalences per class, conservative to loose: core (the class itself: both edges fit); edge (plus
    non-member molecules with a clean core whose protected run lines up with a class edge); loose (plus non-member
    molecules with a clean core under any protection).

    Non-members are molecules whose per-molecule label (label() at bf_threshold) is not 'member', so the molecules
    added here are exactly those molecules.tsv.gz labels tier 'edge' / 'loose'. Each tier is a coherent union of
    molecule-level contributions: core is the mean class posterior (the EM weight), and a qualifying molecule adds
    only its remaining 1 - P mass (its posterior P is already in core), i.e.
    tier = mean_i [P_i + (1 - P_i) * 1{molecule i is in the tier}], never above 1."""
    n = max(len(units), 1); out = []
    for c, g in enumerate(gs):
        spots = set((prof[c] or {}).keys()) if prof else set()
        e = l = 0.
        for i, (u, pc) in enumerate(zip(units, P[:, c])):
            if label(pc, w[c], bf_threshold, log_bf=None if log_bfs is None else log_bfs[i, c])[0] == 'member':
                continue
            m = tier_match(u, g, spots)
            if m:
                rest = 1. - float(pc)
                e += rest if m[0] == 'edge' else 0.; l += rest
        core = float(w[c]); out.append(dict(core=core, edge=core + e/n, loose=core + l/n))
    return out


def molecule_tiers(u, gs, prof):
    """Per class, the molecule's looser-tier call (tier, call) or None (see tier_match)."""
    return [tier_match(u, g, set((prof[c] or {}).keys()) if prof else set()) for c, g in enumerate(gs)]


def expected_evidence(units, g, margin=30):
    """Median over spanning molecules of the expected log-likelihood ratio (accessible vs protected) of the class span."""
    a0, b0 = g['span']; vals = []
    for u in units:
        if u['pos'].min() > a0 - margin or u['pos'].max() < b0 + margin:
            continue
        m = (u['pos'] >= a0) & (u['pos'] < b0); pa, pp = u['pa'][m], u['pp'][m]
        vals.append(float(np.sum(pa*np.log(pa/pp) + (1 - pa)*np.log((1 - pa)/(1 - pp)))))
    return float(np.median(vals)) if vals else 0.


def opportunity_diagnostics(units, g, opt):
    """Substrate summaries on the actual scored denominator, without reading hits.

    KL uses supplied accessible/protected rates at every span site, including
    spot positions without their learned rates. Accessible rates may already
    include calibration from channel-level marks. These summaries describe
    interior opportunity, not edge identifiability or empirical calibration.
    Existing resolution_nats uses a different (30-bp spanning) denominator.
    """
    values = {k: [] for k in ('median_span_opportunities', 'median_left_flank_opportunities',
                             'median_right_flank_opportunities', 'scored_span_accessible_nats',
                             'scored_span_protected_nats')}
    for u in units:
        p = u['pos']; span = (p >= g['span'][0]) & (p < g['span'][1])
        values['median_span_opportunities'].append(int(span.sum()))
        values['median_left_flank_opportunities'].append(int(((p >= g['L'][0] - opt.flank_bp) & (p < g['L'][0])).sum()))
        values['median_right_flank_opportunities'].append(int(((p >= g['R'][1]) & (p < g['R'][1] + opt.flank_bp)).sum()))
        pa, pp = u['pa'][span], u['pp'][span]
        for key, a, b in (('scored_span_accessible_nats', pa, pp), ('scored_span_protected_nats', pp, pa)):
            values[key].append(float(np.sum(a*np.log(a/b) + (1-a)*np.log((1-a)/(1-b)))))
    return {k: float(np.median(v)) if v else None for k, v in values.items()}


def wilson_lo(k, n, z=1.96):
    if n <= 0:
        return 0.
    p = k/n; d = 1 + z*z/n; c = p + z*z/(2*n); r = z*math.sqrt(p*(1 - p)/n + z*z/(4*n*n)); return (c - r)/d


def fit_channel(units, gs, f, opt):
    """Everything for one overlap group and channel: EM weights, posteriors, edge contraction, spots, support gains,
    resolution. Returns the channel's class geometry (gs, contracted where kept) and the contraction records."""
    info = [expected_evidence(units, g) for g in gs]
    edges = [''] * len(gs)
    if opt.edge_contraction:
        gs, edges = contract_edges(units, gs, f, opt)
    sc = Scorer(units, gs, f, opt)
    if opt.learned_spots and any(v >= opt.spot_minimum_evidence_nats for v in info):
        acc = select_spots(units, gs, f, opt, scorer=sc)
        acc = [a if info[c] >= opt.spot_minimum_evidence_nats else set() for c, a in enumerate(acc)]
    else:
        acc = [set() for _ in gs]
    w, P, keep, prof = learn(units, gs, f, opt, acc, scorer=sc)
    capped = [{x for x in acc[c] if prof[c].get(x, 0) >= opt.spot_cap - 1e-9} for c in range(len(gs))]
    if any(capped):
        acc = [acc[c] - capped[c] for c in range(len(gs))]; w, P, keep, prof = learn(units, gs, f, opt, acc, scorer=sc)
    if w is None:
        return None
    M, keep2 = sc.rows(prof)
    fold = np.array([_hash2(u['uid']) for u in keep2]); gains = []
    for c in range(len(gs)):
        gain = 0.
        for kf in (0, 1):
            tr, te = M[fold != kf], M[fold == kf]
            if len(tr) < 10 or len(te) < 1:
                gain = float('nan'); break
            wf, _ = em(tr); wd, _ = em(np.delete(tr, c, axis=1))
            gain += loglik(te, wf) - loglik(np.delete(te, c, axis=1), wd)
        gains.append(gain)
    log_bfs = log_bayes_factors(M, w)
    return dict(w=w, P=P, M=M, log_bf=log_bfs, units=keep, spots=[{int(x): prof[c][x] for x in sorted(acc[c])} for c in range(len(gs))],
                support_gain=gains, resolution=info, n=len(M), gs=gs, edges=edges, tiers=prevalence_tiers(sc.keep, P, w, gs, prof, opt.bf_threshold, log_bfs),
                # Per molecule (aligned with P / units): its own edges for each class, and its best broader stretch.
                calls=[dict(classes=[molecule_edges(it, c, gs[c], prof[c]) for c in range(len(gs))], broader=it['broader'],
                            tiers=molecule_tiers(u, gs, prof))
                       for it, u in zip(sc.items, sc.keep)])


def label(posterior, weight, bf, *, log_bf=None):
    """Three-way evidence label; production callers supply the stable likelihood ratio.

    The posterior-odds fallback preserves the legacy helper API, but cannot
    recover evidence lost to rounding at mixture boundaries.
    """
    if log_bf is None:
        post = min(max(float(posterior), 1e-12), 1 - 1e-12); prior = math.log(max(weight, 1e-12)/max(1 - weight, 1e-12))
        log_bf = math.log(post/(1 - post)) - prior
    lbf = float(log_bf); t = math.log(bf)
    return ('member' if lbf >= t else 'non_member' if lbf <= -t else 'abstain'), lbf
