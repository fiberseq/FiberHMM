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


def joint(u, gs, f, opt, prof=None):
    """[class_1..k, broader, other, accessible] log-likelihoods for one molecule, or None if it does not span the group."""
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

    def lbp(s, g, cm):
        if not cm.any():
            return -1e3
        wl = np.clip(np.minimum(l1, g['L'][1]) - np.maximum(l0, g['L'][0]) + 1, 0, None)
        wr = np.clip(np.minimum(r1, g['R'][1]) - np.maximum(r0, g['R'][0]) + 1, 0, None)
        W = (wl*wr)[cm].astype(float); v = s[cm]; ok = W > 0
        if not ok.any():
            return -1e3
        m = v[ok].max(); norm = (g['L'][1] - g['L'][0] + 1)*(g['R'][1] - g['R'][0] + 1)
        return float(m + math.log(np.sum(W[ok]*np.exp(v[ok] - m))/norm))

    score = lbp if opt.class_weighting == 'bp' else (lambda s, g, cm: lme(s, cm))
    cls = []
    for c, cm in enumerate(cms):
        s = s0
        if prof and prof[c]:
            ppc = np.array([prof[c].get(int(x), np.nan) for x in p]); use = ~np.isnan(ppc)
            if use.any():
                ppc = np.where(use, ppc, 0.5); dc = dP.copy()
                dc[use] = np.where(h[use], np.log(ppc[use]) - np.log(pa[use]), np.log1p(-ppc[use]) - np.log1p(-pa[use]))
                SPc = np.concatenate([[0.], np.cumsum(dc)]); s = base + (SPc[j] - SPc[i]) - (SU[j] - SU[i])
        cls.append(score(s, gs[c], cm))
    c0 = min(g['span'][0] for g in gs); c1 = max(g['span'][1] for g in gs); ins = (p >= c0) & (p < c1)
    return cls + [lme(s0, br), lme(s0, other), float(dU[~ins].sum())]


def rows_of(units, gs, f, opt, prof):
    R, keep = [], []
    for u in units:
        r = joint(u, gs, f, opt, prof)
        if r is not None:
            R.append(r); keep.append(u)
    return np.asarray(R, float).reshape(-1, len(gs) + 3), keep


def learn(units, gs, f, opt, allowed):
    """EM alternating with learned protected-state mark rates at the allowed interior positions of each class."""
    prof = [dict() for _ in gs]
    if any(allowed):
        for _ in range(opt.spot_iterations):
            M, keep = rows_of(units, gs, f, opt, prof)
            if not len(M):
                break
            _, P = em(M); new = []
            for c in range(len(gs)):
                num, den, pp0 = {}, {}, {}
                for u, pc in zip(keep, P[:, c]):
                    for x, hh, q in zip(u['pos'], u['hit'], u['pp']):
                        x = int(x)
                        if x in allowed[c]:
                            num[x] = num.get(x, 0.) + pc*hh; den[x] = den.get(x, 0.) + pc; pp0.setdefault(x, []).append(q)
                new.append({x: float(min(opt.spot_cap, max(np.median(pp0[x]), (num[x] + opt.spot_pseudo_units*np.median(pp0[x]))/(den[x] + opt.spot_pseudo_units))))
                            for x in den})
            prof = new
    M, keep = rows_of(units, gs, f, opt, prof)
    if not len(M):
        return None, None, keep, prof
    w, P = em(M)
    return w, P, keep, prof


def interior_positions(units, g, edge_bp):
    return {int(x) for u in units for x in u['pos'] if g['span'][0] + edge_bp <= x < g['span'][1] - edge_bp}


def select_spots(units, gs, f, opt):
    """Interior positions whose learned rate raises held-out likelihood (both folds, summed) by >= spot_gain_nats."""
    fold = np.array([_hash2(u['uid']) for u in units])
    allowed = [interior_positions(units, g, opt.spot_edge_bp) for g in gs]
    gain = [dict() for _ in gs]; seen = [dict() for _ in gs]
    for k in (0, 1):
        tr = [u for u, z in zip(units, fold) if z != k]; te = [u for u, z in zip(units, fold) if z == k]
        _, _, _, prof = learn(tr, gs, f, opt, allowed)
        M0, _ = rows_of(tr, gs, f, opt, None); T0, _ = rows_of(te, gs, f, opt, None)
        if len(M0) < 10 or not len(T0):
            return [set() for _ in gs]
        w0, _ = em(M0); base = loglik(T0, w0)
        for c in range(len(gs)):
            for x, v in prof[c].items():
                med = np.median([q for u in tr for xx, q in zip(u['pos'], u['pp']) if int(xx) == x])
                if v <= med + 0.02:
                    continue
                one = [dict() for _ in gs]; one[c] = {x: v}
                Mq, _ = rows_of(tr, gs, f, opt, one); wq, _ = em(Mq); Tq, _ = rows_of(te, gs, f, opt, one)
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
    """Everything for one overlap group and channel: EM weights, posteriors, spots, support gains, resolution."""
    info = [expected_evidence(units, g) for g in gs]
    if opt.learned_spots:
        acc = select_spots(units, gs, f, opt)
        acc = [a if info[c] >= opt.spot_minimum_evidence_nats else set() for c, a in enumerate(acc)]
    else:
        acc = [set() for _ in gs]
    w, P, keep, prof = learn(units, gs, f, opt, acc)
    capped = [{x for x in acc[c] if prof[c].get(x, 0) >= opt.spot_cap - 1e-9} for c in range(len(gs))]
    if any(capped):
        acc = [acc[c] - capped[c] for c in range(len(gs))]; w, P, keep, prof = learn(units, gs, f, opt, acc)
    if w is None:
        return None
    M, keep2 = rows_of(units, gs, f, opt, prof)
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
                support_gain=gains, resolution=info, n=len(M))


def label(posterior, weight, bf):
    """Three-way per-molecule label: posterior odds over prior odds against the rest of the mixture."""
    post = min(max(float(posterior), 1e-12), 1 - 1e-12); prior = math.log(max(weight, 1e-12)/max(1 - weight, 1e-12))
    lbf = math.log(post/(1 - post)) - prior; t = math.log(bf)
    return ('member' if lbf >= t else 'non_member' if lbf <= -t else 'abstain'), lbf
