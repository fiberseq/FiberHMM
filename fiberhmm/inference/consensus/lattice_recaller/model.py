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


def _hash2(uid):
    return int(hashlib.sha1(uid.encode()).hexdigest(), 16) % 2


def em(M, iters=1000):
    """Mixture weights and posteriors for a molecules x hypotheses log-likelihood matrix."""
    k = M.shape[1]; w = np.full(k, 1/k)
    for _ in range(iters):
        A = M + np.log(np.maximum(w, 1e-12)); A -= A.max(1, keepdims=True); P = np.exp(A); P /= P.sum(1, keepdims=True); nw = P.mean(0)
        if np.abs(nw - w).max() < 1e-8:
            w = nw; break
        w = nw
    A = M + np.log(np.maximum(w, 1e-12)); A -= A.max(1, keepdims=True); P = np.exp(A); P /= P.sum(1, keepdims=True)
    return w, P


def loglik(M, w):
    X = M + np.log(np.maximum(w, 1e-12)); m = X.max(1)
    return float(np.sum(m + np.log(np.exp(X - m[:, None]).sum(1))))


def _logsumexp(v):
    m = v.max(); return float(m + math.log(np.exp(v - m).sum()))


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
    br &= ~anycls; other = valid & ~anycls & ~br
    cls = []
    for g, cm in zip(gs, cms):
        if opt.class_weighting == 'bp':
            wl = np.clip(np.minimum(l1, g['L'][1]) - np.maximum(l0, g['L'][0]) + 1, 0, None)
            wr = np.clip(np.minimum(r1, g['R'][1]) - np.maximum(r0, g['R'][0]) + 1, 0, None)
            W = np.broadcast_to(wl*wr, cm.shape); use = cm & (W > 0)
            norm = (g['L'][1] - g['L'][0] + 1)*(g['R'][1] - g['R'][0] + 1)
            logw = np.log(W[use].astype(float)) - math.log(norm)
        else:
            use = cm; logw = np.full(int(cm.sum()), -math.log(max(int(cm.sum()), 1)))
        ii, jc = np.nonzero(use)
        cls.append((ii, jc + 1, s0[use], logw))
    c0 = min(g['span'][0] for g in gs); c1 = max(g['span'][1] for g in gs); ins = (p >= c0) & (p < c1)
    return dict(p=p, dP=dP, h=h, pa=pa, cls=cls, fixed=[lme(s0, br), lme(s0, other), float(dU[~ins].sum())])


def _class_score(item, c, pc=None):
    ii, jc, s0c, logw = item['cls'][c]
    if not len(s0c):
        return -1e3
    v = s0c + logw
    if pc:
        ppc = np.array([pc.get(int(x), np.nan) for x in item['p']]); use = ~np.isnan(ppc)
        if use.any():
            p, h, pa = ppc[use], item['h'][use], item['pa'][use]
            delta = np.zeros(len(item['p'])); delta[use] = np.where(h, np.log(p) - np.log(pa), np.log1p(-p) - np.log1p(-pa)) - item['dP'][use]
            D = np.concatenate([[0.], np.cumsum(delta)]); v = v + D[jc] - D[ii]
    return _logsumexp(v)


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
    likelihood of the whole group rises by >= edge_gain_nats (summed over both halves). Returns (gs, records)."""
    records = [''] * len(gs)
    none = [set() for _ in gs]
    _, P, keep, _ = learn(units, gs, f, opt, none)
    if P is None:
        return gs, records
    full = propose_edges(keep, P, gs, opt)
    cands = [c for c, (g2, _) in enumerate(full) if g2 is not None]
    if not cands:
        return gs, records
    fold = np.array([_hash2(u['uid']) for u in units]); gain = {c: 0. for c in cands}; both = {c: True for c in cands}
    for k in (0, 1):
        tr = [u for u, z in zip(units, fold) if z != k]; te = [u for u, z in zip(units, fold) if z == k]
        sc_tr, sc_te = Scorer(tr, gs, f, opt), Scorer(te, gs, f, opt)
        _, Ptr, ktr, _ = learn(tr, gs, f, opt, none, scorer=sc_tr)
        if Ptr is None or len(ktr) < 10:
            return gs, records
        prop = propose_edges(ktr, Ptr, gs, opt)
        for c in cands:
            both[c] &= prop[c][0] is not None
            alt = [full[c][0] if i == c else g for i, g in enumerate(gs)]
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


def select_spots(units, gs, f, opt):
    """Interior positions whose learned rate raises held-out likelihood (both folds, summed) by >= spot_gain_nats."""
    fold = np.array([_hash2(u['uid']) for u in units])
    allowed = [interior_positions(units, g, opt.spot_edge_bp) for g in gs]
    gain = [dict() for _ in gs]; seen = [dict() for _ in gs]
    for k in (0, 1):
        tr = [u for u, z in zip(units, fold) if z != k]; te = [u for u, z in zip(units, fold) if z == k]
        sc_tr, sc_te = Scorer(tr, gs, f, opt), Scorer(te, gs, f, opt)
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


def expected_evidence(units, g, margin=30):
    """Median over spanning molecules of the expected log-likelihood ratio (accessible vs protected) of the class span."""
    a0, b0 = g['span']; vals = []
    for u in units:
        if u['pos'].min() > a0 - margin or u['pos'].max() < b0 + margin:
            continue
        m = (u['pos'] >= a0) & (u['pos'] < b0); pa, pp = u['pa'][m], u['pp'][m]
        vals.append(float(np.sum(pa*np.log(pa/pp) + (1 - pa)*np.log((1 - pa)/(1 - pp)))))
    return float(np.median(vals)) if vals else 0.


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
    if opt.learned_spots and any(v >= opt.spot_minimum_evidence_nats for v in info):
        acc = select_spots(units, gs, f, opt)
        acc = [a if info[c] >= opt.spot_minimum_evidence_nats else set() for c, a in enumerate(acc)]
    else:
        acc = [set() for _ in gs]
    sc = Scorer(units, gs, f, opt)
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
    return dict(w=w, P=P, units=keep, spots=[{int(x): prof[c][x] for x in sorted(acc[c])} for c in range(len(gs))],
                support_gain=gains, resolution=info, n=len(M), gs=gs, edges=edges)


def label(posterior, weight, bf):
    """Three-way per-molecule label: posterior odds over prior odds against the rest of the mixture."""
    post = min(max(float(posterior), 1e-12), 1 - 1e-12); prior = math.log(max(weight, 1e-12)/max(1 - weight, 1e-12))
    lbf = math.log(post/(1 - post)) - prior; t = math.log(bf)
    return ('member' if lbf >= t else 'non_member' if lbf <= -t else 'abstain'), lbf
