"""Class discovery: k-means on censored call edges, prediction-strength k, exact held-out identity merging.

Port of the 25 Sep 2026 prototype (paper/analysis/cr_class_quality_20260925: kclass.py, hybrid.py, lattice_fix.py,
lattice_ps09.py). Channels are labels, never features: calls from every channel are clustered together.
"""
from __future__ import annotations

import hashlib

import numpy as np
from sklearn.cluster import KMeans

MIN_MOLECULES = 10     # kmax = max(2, min(kmax_cap, calls // (2 * MIN_MOLECULES)))
SD_FLOOR = 1.5
CLIP = 50.


def _hash(text, modulo=None, bits=8):
    value = int(hashlib.sha1(text.encode()).hexdigest()[:bits], 16)
    return value if modulo is None else value % modulo


def split_half(uid, salt):
    return _hash(salt + uid) & 1


def censor(u, a, b, censor_bp):
    """Lattice-censored edges of call [a, b): left edge after the previous event, right edge up to the next one."""
    ev = u['pos'][u['hit']]
    prev = ev[ev < a]; nxt = ev[ev >= b]
    l0 = max(prev.max() + 1 if len(prev) else a - censor_bp, a - censor_bp)
    r1 = min(nxt.min() if len(nxt) else b + censor_bp, b + censor_bp)
    return l0, a, b, r1


def features(units, censor_bp):
    X, meta = [], []
    for i, u in enumerate(units):
        for a, b, _llr in u['calls']:
            l0, l1, r0, r1 = censor(u, a, b, censor_bp)
            X.append(((l0 + l1)/2, (r0 + r1)/2)); meta.append((i, a, b, l0, l1, r0, r1))
    return np.asarray(X, float).reshape(-1, 2), meta


def _fit(X, k, seed):
    km = KMeans(k, n_init=10, random_state=seed).fit(X)
    return km, km.predict


def prediction_strength(X, groups, k, seed, splits):
    """Worst-cluster prediction strength averaged over molecule split-halves, and each split's per-cluster values.

    In split b the molecules are hashed into halves A and T (molecule-level, so a molecule's calls stay together);
    k-means is fitted on each, and test cluster j's strength is the fraction of its call pairs that A's fit also puts
    together. Returns (mean over splits of the worst cluster, splits), where splits holds one (test_rows, test_labels,
    strengths) triple per split: the indices into X of T's calls, their T-cluster labels and the k strengths. The
    triples let cluster_stability average a final cluster's strength over every split."""
    worst, detail = [], []
    for b in range(splits):
        h = np.array([split_half(g, f's{seed}b{b}') for g in groups]); A, T = X[h == 0], X[h == 1]
        if len(A) < k or len(T) < k:
            return 0.0, []
        _, pa = _fit(A, k, seed); _, pt = _fit(T, k, seed)
        lt = pt(T); la = pa(T); ps = []
        for j in range(k):
            idx = np.where(lt == j)[0]
            if len(idx) < 2:
                ps.append(0.0); continue
            same = la[idx][:, None] == la[idx][None, :]; n = len(idx)
            ps.append((same.sum() - n)/(n*(n - 1)))
        worst.append(min(ps))
        detail.append((np.flatnonzero(h == 1), lt, np.asarray(ps, float)))
    return float(np.mean(worst)), detail


def cluster_stability(members, splits):
    """A final cluster's prediction strength, averaged over every split-half.

    members: indices into X of the cluster's calls; splits: prediction_strength's per-split triples. In each split the
    cluster's calls that fall in the test half are matched to the test-half clusters holding them, and the split's
    value is those clusters' strengths weighted by the share of the cluster's calls each holds (one test cluster
    holding them all: its strength). Splits where none of the cluster's calls are in the test half carry no evidence
    and are skipped; a cluster with no held-out calls in any split scores 0. The value is independent of call order,
    of how each split's k-means numbers its clusters and of the order of the splits.

    (Up to 3.0.0 a cluster took the values of the single rounded test-half centroid nearest its own; centroids rarely
    coincide across splits, so in practice one split decided it.)"""
    members = np.asarray(members, dtype=np.int64); vals = []
    for rows, labels, ps in splits:
        held = labels[np.isin(rows, members)]
        if len(held):
            vals.append(float(np.bincount(held, minlength=len(ps)) @ ps)/len(held))
    return float(np.mean(vals)) if vals else 0.0


def nominate(units, opt):
    """Candidates at the chosen k: the largest k <= kmax with prediction strength >= stringency, each with its
    split-averaged stability (cluster_stability; 1.0 when k = 1)."""
    X, meta = features(units, opt.censor_bp)
    if len(X) < 2:
        return [], 0, []
    groups = [units[m[0]]['uid'] for m in meta]
    kmax = max(2, min(opt.kmax, len(X)//(2*MIN_MOLECULES)))
    scored = {k: prediction_strength(X, groups, k, opt.seed, opt.prediction_splits) for k in range(1, kmax + 1)}
    choice = [(k, scored[k][0]) for k in range(1, kmax + 1)]
    ok = [k for k, ps in choice if ps >= opt.stringency]; k = max(ok) if ok else 1
    _, pred = _fit(X, k, opt.seed); lab = pred(X)
    cands = []
    for j in range(k):
        idx = np.where(lab == j)[0]
        if len(idx) < opt.minimum_candidate_calls:
            continue
        cands.append(dict(id=f'c{j}', members=[f'c{j}'], calls=[meta[i] for i in idx], X=X[idx],
                          stability=cluster_stability(idx, scored[k][1]) if k > 1 else 1.0))
    return cands, k, choice


def boxes(cand, quantiles=(10, 90)):
    """Pooled edge boxes (quantiles of the censored edge ranges) and the class span (median expected edges)."""
    M = np.array([m[3:] for m in cand['calls']], float); X = cand['X']; q0, q1 = quantiles
    return dict(L=[int(np.percentile(M[:, 0], q0)), int(np.percentile(M[:, 1], q1))],
                R=[int(np.percentile(M[:, 2], q0)), int(np.percentile(M[:, 3], q1))],
                span=(float(np.median(X[:, 0])), float(np.median(X[:, 1]))))


def merge(a, b):
    return dict(id=a['id'] + '+' + b['id'], members=a['members'] + b['members'], calls=a['calls'] + b['calls'],
                X=np.vstack([a['X'], b['X']]), stability=max(a['stability'], b['stability']))


def overlaps(a, b):
    sa, sb = boxes(a)['span'], boxes(b)['span']
    return min(sa[1], sb[1]) - max(sa[0], sb[0]) > 0


# ---------------------------------------------------------------- exact-likelihood identity test
def _edge_grid(l0, l1, r0, r1, min_w, max_w):
    L = np.arange(l0, l1); R = np.arange(r0, r1)
    W = R[None, :] - L[:, None]; valid = ((W >= min_w) & (W <= max_w)).ravel()
    ll, rr = np.meshgrid(L, R, indexing='ij')
    return L, R, valid, ll.ravel()[valid], rr.ravel()[valid]


def _llr_matrix(mols, L, R, valid):
    rows = []
    for m in mols:
        ia = np.searchsorted(m['positions'], L); ib = np.searchsorted(m['positions'], R)
        rows.append((m['prefix'][ib][None, :] - m['prefix'][ia][:, None]).ravel()[valid])
    if not rows:
        return np.zeros((0, int(valid.sum())))
    return np.exp(np.clip(np.array(rows), -CLIP, CLIP))


def _floor_cov(cov):
    vals, vecs = np.linalg.eigh((cov + cov.T)/2)
    return (vecs*np.maximum(vals, SD_FLOOR**2)) @ vecs.T


def _gauss_weights(ll, rr, mu, cov):
    x = np.stack([ll - mu[0], rr - mu[1]], 1)
    q = np.einsum('ij,jk,ik->i', x, np.linalg.inv(cov), x)
    w = np.exp(-.5*(q - q.min())); return w/w.sum()


def _fit_mix(chans, ll, rr, mus, iters=150, tol=1e-6):
    k = len(mus); mu = [np.array(m, float) for m in mus]; cov = [np.diag([9., 9.]) for _ in mus]
    W = [np.r_[.5, np.full(k, .3/k), .2] for _ in chans]; prev = -np.inf
    for _ in range(iters):
        G = [_gauss_weights(ll, rr, m, c) for m, c in zip(mu, cov)]; rc = [np.zeros(len(ll)) for _ in range(k)]; total = 0.
        for g, (E, o) in enumerate(chans):
            S = np.stack([E @ Gk for Gk in G], 1); w = W[g]
            den = w[0] + S @ w[1:1 + k] + w[-1]*o; total += np.log(den).sum()
            for j in range(k):
                rc[j] += G[j]*(E.T @ (w[1 + j]/den))
            nw = np.r_[(w[0]/den).mean(), (S*w[1:1 + k]/den[:, None]).mean(0), (w[-1]*o/den).mean()]
            nw = np.clip(nw, 1e-6, 1); W[g] = nw/nw.sum()
        for j in range(k):
            m = rc[j].sum()
            if m > 0:
                mu[j] = np.array([(rc[j]*ll).sum(), (rc[j]*rr).sum()])/m
                d = np.stack([ll - mu[j][0], rr - mu[j][1]], 1); cov[j] = _floor_cov((d*rc[j][:, None]).T @ d/m)
        if abs(total - prev) < tol*max(1., abs(total)):
            break
        prev = total
    return dict(mu=mu, cov=cov, W=W)


def _loglik_mix(chans, F, ll, rr):
    G = [_gauss_weights(ll, rr, m, c) for m, c in zip(F['mu'], F['cov'])]; tot = 0.
    for (E, o), w in zip(chans, F['W']):
        if len(E) == 0:
            continue
        S = np.stack([E @ Gk for Gk in G], 1); tot += np.log(w[0] + S @ w[1:1 + len(G)] + w[-1]*o).sum()
    return float(tot)


def identity_gain(a, b, units, opt):
    """Held-out log-likelihood gain (nats) of two edge geometries over one, pooled over channels and folds."""
    ga, gb = boxes(a), boxes(b); pad, wide, maxw, folds = opt.identity_pad_bp, opt.identity_wide_bp, opt.identity_max_width_bp, opt.identity_folds
    l0 = min(ga['L'][0], gb['L'][0]) - pad; l1 = max(ga['L'][1], gb['L'][1]) + pad
    r0 = min(ga['R'][0], gb['R'][0]) - pad; r1 = max(ga['R'][1], gb['R'][1]) + pad
    L, R, valid, ll, rr = _edge_grid(l0, l1 + 1, r0, r1 + 1, 4, maxw)
    Lo, Ro, vo, _, _ = _edge_grid(l0 - wide, r1 + 1, l0, r1 + wide + 1, 4, maxw)
    per_ch = []
    for ch in sorted({u['ch'] for u in units}):
        mols = [dict(positions=u['pos'], prefix=np.concatenate([[0.], np.cumsum(u['d'])]), uid=u['uid']) for u in units
                if u['ch'] == ch and len(u['pos']) and u['pos'].min() <= l0 and u['pos'].max() >= r1]
        if len(mols) < 10:
            continue
        E = _llr_matrix(mols, L, R, valid); O = _llr_matrix(mols, Lo, Ro, vo).mean(1)
        per_ch.append((E, O, np.array([_hash(m['uid'], folds) for m in mols])))
    if not per_ch:
        return 0.0, 0
    mu_a = (np.median(a['X'][:, 0]), np.median(a['X'][:, 1])); mu_b = (np.median(b['X'][:, 0]), np.median(b['X'][:, 1]))
    mu_ab = tuple(np.median(np.vstack([a['X'], b['X']]), 0)); gain = 0.
    for f in range(folds):
        tr = [(E[fo != f], O[fo != f]) for E, O, fo in per_ch]; te = [(E[fo == f], O[fo == f]) for E, O, fo in per_ch]
        F2 = _fit_mix(tr, ll, rr, [mu_a, mu_b]); F1 = _fit_mix(tr, ll, rr, [mu_ab])
        gain += _loglik_mix(te, F2, ll, rr) - _loglik_mix(te, F1, ll, rr)
    return float(gain), int(sum(len(E) for E, _, _ in per_ch))


def agglomerate(cands, units, opt, cache=None):
    """Merge overlapping candidates, smallest held-out gain first, while the gain is below identity_nats."""
    cur = list(cands); cache = {} if cache is None else cache; log = []
    while True:
        best = None
        for i in range(len(cur)):
            for j in range(i + 1, len(cur)):
                if not overlaps(cur[i], cur[j]):
                    continue
                key = frozenset([cur[i]['id'], cur[j]['id']])
                if key not in cache:
                    cache[key] = identity_gain(cur[i], cur[j], units, opt)
                g = cache[key][0]
                if g < opt.identity_nats and (best is None or g < best[0]):
                    best = (g, i, j)
        if best is None:
            return cur, log
        g, i, j = best; log.append((cur[i]['id'], cur[j]['id'], round(g, 2)))
        m = merge(cur[i], cur[j]); cur = [c for t, c in enumerate(cur) if t not in (i, j)] + [m]


def discover_tile(units, opt):
    """Final classes of one tile: stable merged candidates with their pooled geometry (sorted by left edge)."""
    cands, k, choice = nominate(units, opt)
    final, log = agglomerate(cands, units, opt)
    out = []
    for c in final:
        if c['stability'] < opt.stringency:
            continue
        g = boxes(c, (opt.edge_quantile_low, opt.edge_quantile_high)); core = boxes(c, (opt.core_quantile_low, opt.core_quantile_high))
        out.append(dict(g, core_bp=core['R'][0] - core['L'][1], calls=len(c['calls']), stability=c['stability'], candidate=c['id'],
                        molecules=len({units[m[0]]['uid'] for m in c['calls']})))
    out.sort(key=lambda g: g['span'][0])
    return out, dict(k=k, prediction_strength=[[kk, round(ps, 4)] for kk, ps in choice], merges=log)


def core_width(g):
    """Protected core between the core-rule boxes (falls back to the scoring boxes)."""
    return g.get('core_bp', g['R'][0] - g['L'][1])


def same_class(g, o, centre_bp=3.0):
    """One class geometry found twice: span centres within centre_bp and width ratio 0.7-1.43."""
    w = max(1., g['span'][1] - g['span'][0])
    return abs(sum(g['span'])/2 - sum(o['span'])/2) <= centre_bp and 0.7 <= w/max(1., o['span'][1] - o['span'][0]) <= 1.43


def dedupe(classes, centre_bp=3.0):
    """Classes found in overlapping tiles: same class (same_class) keep the better supported."""
    out = []
    for g in sorted(classes, key=lambda g: -g['calls']):
        if any(same_class(g, o, centre_bp) for o in out):
            continue
        out.append(g)
    return sorted(out, key=lambda g: g['span'][0])


def overlap_groups(classes):
    """Connected components of classes whose spans overlap (nested or partial)."""
    comp = []
    for i, g in enumerate(classes):
        placed = [c for c in comp if any(min(g['span'][1], classes[j]['span'][1]) > max(g['span'][0], classes[j]['span'][0]) for j in c)]
        if not placed:
            comp.append([i])
        else:
            base = placed[0]; base.append(i)
            for c in placed[1:]:
                base.extend(c); comp.remove(c)
    return comp
