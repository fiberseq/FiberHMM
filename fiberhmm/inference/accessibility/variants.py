"""NFR variant discovery and read-level prevalence (EXPERIMENTAL preview).

The lattice recaller's discovery recipe, applied to nucleosome-bounded gaps instead of footprint calls:

1. k-means on the (left, right) edge midpoints of every gap; k is the largest k <= kmax whose split-half prediction
   strength (molecule halves by unit_id hash; ``lattice_recaller.discovery.prediction_strength``) reaches
   ``stringency``.
2. identity merge (the recaller's ``agglomerate`` pattern): overlapping candidates are merged, smallest gain first,
   while the 3-fold held-out (folds by unit_id hash) log-likelihood gain of two edge Gaussians over one is below
   ``identity_nats``.
3. support: a variant stays only if dropping it costs >= ``support_nats`` held-out nats, it has >= ``min_reads``
   expected member reads, and its split-averaged prediction-strength stability (the recaller's
   ``cluster_stability``) is >= ``stringency``; otherwise the weakest
   failing candidate is dropped and the test repeats.

Identity and support use the same fitting rule as the final estimator: geometry is the robust (MinCovDet) centre and
covariance of the candidate's own (training) members, held fixed; only weights are fitted. A dropped candidate's gaps
therefore fall to "other shape" instead of stretching a neighbour.

Prevalence: read-level EM over configurations (closed, or a left-to-right tuple of labels, one per gap, each a
variant or "other"); a variant appears at most once per read and variants keep their positional order. Prevalence of a
variant = summed weight of the configurations containing it. It is reported as a range, strict (reads with membership
>= 0.9) to EM. Bootstrap intervals are conditional on the fixed catalogue and geometry.

Everything is ordered by unit_id; read order in the BAM cannot change the answer. ``robust(n)`` reruns discovery
under n hashed read orders.
"""
from __future__ import annotations

import itertools

import numpy as np
from sklearn.cluster import KMeans

from ..consensus.lattice_recaller.discovery import _hash, cluster_stability, prediction_strength

SD_FLOOR = 5.0
TIMER_DEPTHS = ((500, 'actuated'), (300, 'merged'), (175, 'pioneered'))


# ------------------------------------------------------------------ edge-mixture likelihood
def _gauss_logpdf_vec(X, VE, mu, cov):
    """log N(x; mu, cov + diag(ve)) per row; the censoring range adds its variance (w^2/12)."""
    C = cov[None, :, :] + VE[:, :, None]*np.eye(2)[None]
    det = C[:, 0, 0]*C[:, 1, 1] - C[:, 0, 1]*C[:, 1, 0]
    d = X - mu
    inv00 = C[:, 1, 1]/det; inv11 = C[:, 0, 0]/det; inv01 = -C[:, 0, 1]/det
    q = d[:, 0]**2*inv00 + 2*d[:, 0]*d[:, 1]*inv01 + d[:, 1]**2*inv11
    return -0.5*(q + np.log(det) + 2*np.log(2*np.pi))


def _floor(cov):
    vals, vecs = np.linalg.eigh((cov + cov.T)/2)
    return (vecs*np.maximum(vals, SD_FLOOR**2)) @ vecs.T


class EdgeSpace:
    """Gap features and the "other shape" density (uniform over gaps that overlap the region)."""

    def __init__(self, gaps, region, min_gap_bp):
        self.X = np.array([[g['L'], g['R']] for g in gaps], float).reshape(-1, 2)
        self.VE = np.array([[g['wL']**2/12, g['wR']**2/12] for g in gaps], float).reshape(-1, 2)
        lo = min(self.X[:, 0].min() if len(gaps) else region[0], region[0]) - 50
        hi = max(self.X[:, 1].max() if len(gaps) else region[1], region[1]) + 50
        r0, r1 = region
        area = 0
        for L in range(int(lo), int(r1)):
            rmin = max(L + min_gap_bp, r0 + 1)
            area += max(0, int(hi) - rmin)
        self.log_other = -np.log(max(area, 1))


def fit_mixture(X, VE, log_other, mus, covs=None, iters=200, tol=1e-7, fixed=False, w_init=None):
    """Gap-level EM: Gaussians (floored covariance + censoring variance) + uniform "other".

    fixed=True: geometry fixed, weights only; 'mu': centres fixed, covariances refitted; False: both refitted."""
    k = len(mus)
    mu = [np.asarray(m, float) for m in mus]
    cov = [np.asarray(c, float) for c in covs] if covs is not None else [np.diag([400., 400.]) for _ in mus]
    w = np.asarray(w_init, float) if w_init is not None else np.r_[np.full(k, .9/k), .1]
    prev = -np.inf
    total = 0.
    r = np.zeros((len(X), k + 1))
    for _ in range(iters):
        logp = np.column_stack([_gauss_logpdf_vec(X, VE, mu[j], cov[j]) for j in range(k)] + [np.full(len(X), log_other)])
        a = logp + np.log(np.maximum(w, 1e-300)); m = a.max(1, keepdims=True)
        ll = m[:, 0] + np.log(np.exp(a - m).sum(1)); r = np.exp(a - ll[:, None])
        total = ll.sum()
        w = np.clip(r.mean(0), 1e-8, 1); w /= w.sum()
        if fixed is not True:
            for j in range(k):
                s = r[:, j].sum()
                if s > 1e-6:
                    if fixed is False:
                        mu[j] = (r[:, j, None]*X).sum(0)/s
                    d = X - mu[j]; cov[j] = _floor((d*r[:, j, None]).T @ d/s)
        if abs(total - prev) < tol*max(1., abs(total)):
            break
        prev = total
    return dict(mu=mu, cov=cov, w=w, ll=float(total), resp=r)


def loglik(X, VE, log_other, F):
    k = len(F['mu'])
    logp = np.column_stack([_gauss_logpdf_vec(X, VE, F['mu'][j], F['cov'][j]) for j in range(k)] + [np.full(len(X), log_other)])
    a = logp + np.log(np.maximum(F['w'], 1e-300)); m = a.max(1, keepdims=True)
    return float((m[:, 0] + np.log(np.exp(a - m).sum(1))).sum())


def robust_geometry(Xc, seed):
    """Members' robust centre and covariance (MinCovDet, deterministic): k-means cells include neighbours' tails."""
    from sklearn.covariance import MinCovDet
    if len(Xc) < 10:
        return np.median(Xc, 0), _floor(np.cov(Xc.T) if len(Xc) > 2 else np.diag([400., 400.]))
    m = MinCovDet(support_fraction=0.75, random_state=seed).fit(Xc)
    return m.location_, _floor(m.covariance_)


def heldout_gain_groups(X, VE, log_other, folds_of, groups_full, groups_reduced, nfold, seed=1):
    """Held-out log-likelihood gain of the full over the reduced model, fitted with the final estimator's rule:
    per training fold, each group's geometry is the MinCovDet centre/covariance of its training members, fixed;
    only weights are fitted. ``groups_*`` are lists of index arrays into X."""
    g = 0.
    for f in range(nfold):
        tr, te = folds_of != f, folds_of == f
        if te.sum() == 0 or tr.sum() < 5:
            continue
        lls = []
        for groups in (groups_full, groups_reduced):
            geo = []
            for idx in groups:
                t = idx[tr[idx]]
                if len(t) >= 3:
                    geo.append(robust_geometry(X[t], seed))
            if not geo:
                lls.append(te.sum()*log_other); continue
            F = fit_mixture(X[tr], VE[tr], log_other, [m for m, _ in geo], covs=[c for _, c in geo], fixed=True)
            lls.append(loglik(X[te], VE[te], log_other, F))
        g += lls[0] - lls[1]
    return g


# ------------------------------------------------------------------ discovery
def canonical(reads, salt=''):
    """Gaps in canonical order: by unit_id (or by hash(salt + unit_id) for reordered runs), then left edge."""
    key = (lambda r: r['uid']) if not salt else (lambda r: _hash(salt + r['uid']))
    gaps, gid = [], []
    for r in sorted([r for r in reads if r['callable']], key=key):
        for g in r['gaps']:
            gaps.append(g); gid.append(r['uid'])
    return gaps, gid


def discover(reads, region, opt, salt='', progress=None):
    """Supported variants (list of dicts with mu, cov, stability, gain, exp_reads), diagnostics and the EdgeSpace."""
    gaps, gid = canonical(reads, salt)
    es = EdgeSpace(gaps, region, opt.min_gap_bp)
    X, VE = es.X, es.VE
    diag = dict(gaps=len(gaps), callable_reads=sum(r['callable'] for r in reads), reads=len(reads),
                k=0, ps_curve=[], candidates=[], merges=[], dropped=[])
    if len(X) < 2*opt.min_reads:
        return [], diag, es
    kmax = min(opt.kmax, max(2, len(X)//(2*int(opt.min_reads))))     # == the prototype's rule for kmax >= 2
    curve, splits, strength = [], {}, {}
    for k in range(1, kmax + 1):
        strength[k], splits[k] = prediction_strength(X, gid, k, opt.seed, opt.splits)
        curve.append((k, round(strength[k], 4)))     # rounded for the diagnostics only
        if progress:
            progress(k, kmax)
    # k is chosen on full precision, as the recaller does (a rounded score could pass a threshold it misses).
    ok = [k for k in strength if strength[k] >= opt.stringency]
    k = max(ok) if ok else 1
    km = KMeans(k, n_init=10, random_state=opt.seed).fit(X); lab = km.labels_
    cands = []
    for j in range(k):
        idx = np.where(lab == j)[0]
        if len({gid[i] for i in idx}) < opt.min_reads:
            continue
        # The recaller's split-averaged stability (cluster_stability over the chosen k's split-halves; 1.0 at k = 1).
        cands.append(dict(id=f'k{j}', idx=idx, stability=cluster_stability(idx, splits[k]) if k > 1 else 1.0))
    diag.update(k=k, ps_curve=curve, candidates=[dict(id=c['id'], L=float(np.median(X[c['idx'], 0])), R=float(np.median(X[c['idx'], 1])),
                                                      gaps=int(len(c['idx'])), stability=round(c['stability'], 3)) for c in cands])
    folds = np.array([_hash('fold' + u, opt.folds) for u in gid])

    def mu_of(c):
        return np.median(X[c['idx']], 0)

    def overl(a, b):
        ma, mb = mu_of(a), mu_of(b)
        return min(ma[1], mb[1]) - max(ma[0], mb[0]) > 0

    merges = []
    while True:
        best = None
        for i, j in itertools.combinations(range(len(cands)), 2):
            if not overl(cands[i], cands[j]):
                continue
            sel = np.r_[cands[i]['idx'], cands[j]['idx']]
            ni = len(cands[i]['idx']); loc = np.arange(len(sel))
            g = heldout_gain_groups(X[sel], VE[sel], es.log_other, folds[sel], [loc[:ni], loc[ni:]], [loc], opt.folds, opt.seed)
            if g < opt.identity_nats and (best is None or g < best[0]):
                best = (g, i, j)
        if best is None:
            break
        g, i, j = best
        merges.append((cands[i]['id'], cands[j]['id'], round(float(g), 1)))
        m = dict(id=cands[i]['id'] + '+' + cands[j]['id'], idx=np.r_[cands[i]['idx'], cands[j]['idx']],
                 stability=max(cands[i]['stability'], cands[j]['stability']))
        cands = [c for t, c in enumerate(cands) if t not in (i, j)] + [m]
    diag['merges'] = merges
    dropped = []
    tests = []
    while cands:
        geo_s = [robust_geometry(X[c['idx']], opt.seed) for c in cands]
        F = fit_mixture(X, VE, es.log_other, [m for m, _ in geo_s], covs=[c_ for _, c_ in geo_s], fixed=True)
        tests = []
        for t, c in enumerate(cands):
            groups = [cc['idx'] for cc in cands]
            gain = heldout_gain_groups(X, VE, es.log_other, folds, groups, [gi for s_, gi in enumerate(groups) if s_ != t], opt.folds, opt.seed)
            per_read = {}
            for i, u in enumerate(gid):
                per_read[u] = max(per_read.get(u, 0.), F['resp'][i, t])
            exp_reads = float(sum(per_read.values()))
            reasons = []
            if gain < opt.support_nats:
                reasons.append('support')
            if exp_reads < opt.min_reads:
                reasons.append('reads')
            if c['stability'] < opt.stringency:
                reasons.append('stability')
            tests.append(dict(gain=float(gain), exp_reads=exp_reads, stability=c['stability'], ok=not reasons, reasons=reasons))
        bad = [t for t in range(len(cands)) if not tests[t]['ok']]
        if not bad:
            break
        worst = min(bad, key=lambda t: tests[t]['gain'])
        idx = cands[worst]['idx']
        dropped.append(dict(id=cands[worst]['id'], L=float(np.median(X[idx, 0])), R=float(np.median(X[idx, 1])),
                            **{k_: (round(v, 2) if isinstance(v, float) else v) for k_, v in tests[worst].items()}))
        cands = [c for t, c in enumerate(cands) if t != worst]
    diag['dropped'] = dropped
    if not cands:
        return [], diag, es
    geo = [robust_geometry(X[c['idx']], opt.seed) for c in cands]
    F = fit_mixture(X, VE, es.log_other, [g[0] for g in geo], covs=[g[1] for g in geo], fixed=True)
    variants = []
    for t, c in enumerate(cands):
        variants.append(dict(cand=c['id'], mu=F['mu'][t], cov=F['cov'][t], stability=c['stability'],
                             gain=tests[t]['gain'], exp_reads=tests[t]['exp_reads']))
    variants.sort(key=lambda v: (v['mu'][0], v['mu'][1]))
    return variants, diag, es


# ------------------------------------------------------------------ labels
def describe(variants):
    for v in variants:
        L, R = v['mu']
        v['L'], v['R'], v['width'] = float(L), float(R), float(R - L)
        sd = np.sqrt(np.diag(v['cov']))
        v['Lsd'], v['Rsd'] = float(sd[0]), float(sd[1])
        v['depth'] = next((name for w, name in TIMER_DEPTHS if v['width'] >= w), 'sub-pioneered')
    return variants


def relation(v, ref, shift_bp=25):
    """Edge-wise relation to the reference: each edge is same (< shift_bp), out (wider) or in (narrower).
    Overlap <= half the smaller width -> 'alternative' (Timer's alternative opening register)."""
    if v is ref:
        return 'full'
    ov = min(v['R'], ref['R']) - max(v['L'], ref['L'])
    if ov <= 0.5*min(v['width'], ref['width']):
        return 'alternative-' + ('left' if v['L'] + v['R'] < ref['L'] + ref['R'] else 'right')
    dl = ref['L'] - v['L']; dr = v['R'] - ref['R']
    el = 'same' if abs(dl) < shift_bp else ('out' if dl > 0 else 'in')
    er = 'same' if abs(dr) < shift_bp else ('out' if dr > 0 else 'in')
    return {('same', 'same'): 'full-like', ('out', 'same'): 'extended-left', ('same', 'out'): 'extended-right',
            ('in', 'same'): 'trimmed-left', ('same', 'in'): 'trimmed-right', ('in', 'in'): 'core',
            ('out', 'out'): 'extended', ('out', 'in'): 'shifted-left', ('in', 'out'): 'shifted-right'}[(el, er)]


# ------------------------------------------------------------------ read-level EM over configurations
def _em(LL, iters=2000, w0=None):
    w = np.full(LL.shape[1], 1/LL.shape[1]) if w0 is None else np.asarray(w0, float).copy()
    p = None
    for _ in range(iters):
        a = LL + np.log(np.maximum(w, 1e-300)); m = a.max(1, keepdims=True)
        p = np.exp(a - m); p /= p.sum(1, keepdims=True)
        nw = p.mean(0)
        if np.abs(nw - w).max() < 1e-9:
            w = nw; break
        w = nw
    return w, p


def config_loglik(reads, variants, es):
    """(callable reads ordered by unit_id, configurations, read x configuration log-likelihood): the input of the
    configuration EM (``quantify``; ``analysis.group_prevalence`` reuses it for every read group)."""
    k = len(variants)
    callable_reads = sorted([r for r in reads if r['callable']], key=lambda r: r['uid'])
    rows = []
    for r in callable_reads:
        if not r['gaps']:
            rows.append(np.zeros((0, k + 1))); continue
        X = np.array([[g['L'], g['R']] for g in r['gaps']])
        VE = np.array([[g['wL']**2/12, g['wR']**2/12] for g in r['gaps']])
        lp = np.column_stack([_gauss_logpdf_vec(X, VE, v['mu'], v['cov']) for v in variants] + [np.full(len(X), es.log_other)])
        rows.append(lp)
    order = np.argsort([v['mu'][0] for v in variants]); rank = {int(j): r for r, j in enumerate(order)}

    def allowed(c):
        vs_ = [j for j in c if j < k]
        return len(set(vs_)) == len(vs_) and all(rank[a] < rank[b] for a, b in zip(vs_, vs_[1:]))
    configs = sorted({()} | {c for lp in rows for c in itertools.product(range(k + 1), repeat=len(lp)) if allowed(c)},
                     key=lambda c: (len(c), c))
    cidx = {c: i for i, c in enumerate(configs)}
    LL = np.full((len(rows), len(configs)), -np.inf)
    for i, lp in enumerate(rows):
        for c in itertools.product(range(k + 1), repeat=len(lp)):
            if not allowed(c):
                continue
            LL[i, cidx[c]] = sum(lp[g, c[g]] for g in range(len(c))) if c else 0.
    return callable_reads, configs, LL


def quantify(reads, variants, es, opt):
    """Configuration EM, prevalence (strict to EM), conditional bootstrap intervals, per-read membership, labels."""
    k = len(variants); OTHER = k
    callable_reads, configs, LL = config_loglik(reads, variants, es)
    cidx = {c: i for i, c in enumerate(configs)}
    rows = LL
    if len(rows):
        w, P = _em(LL)
    else:
        w, P = np.ones(len(configs))/len(configs), np.zeros((0, len(configs)))
    contains = np.array([[j in c for c in configs] for j in range(k + 1)], float)
    prev = contains @ w
    rng = np.random.default_rng(opt.seed)
    boots = []
    if len(rows):
        for _ in range(opt.bootstrap):
            s = rng.integers(0, len(LL), len(LL)); wb, _ = _em(LL[s], 500); boots.append(contains @ wb)
    boots = np.array(boots)
    ci = dict(lo=np.percentile(boots, 2.5, 0), hi=np.percentile(boots, 97.5, 0)) if len(boots) else None
    Pv = P @ contains.T if len(rows) else np.zeros((0, k + 1))
    describe(variants)
    for j, v in enumerate(variants):
        v['prevalence'] = float(prev[j])
        v['ci'] = (float(ci['lo'][j]), float(ci['hi'][j])) if ci else None
        v['only'] = float(w[cidx[(j,)]]) if (j,) in cidx else 0.
        v['strict'] = float(np.mean(Pv[:, j] >= 0.9)) if len(Pv) else 0.
        v['name'] = f'V{j + 1}'
    ref = max(variants, key=lambda v: v['prevalence']) if variants else None
    jr = variants.index(ref) if variants else None
    for j, v in enumerate(variants):
        v['relation'] = relation(v, ref, opt.shift_bp)
        both = sum(w[i] for i, c in enumerate(configs) if j in c and jr in c)
        v['with_ref'] = float(both/max(prev[j], 1e-12)) if j != jr else 1.0
        if v['relation'].startswith('alternative') and v['with_ref'] > 0.5:
            v['relation'] = v['relation'].replace('alternative', 'satellite')
    for v in variants:
        def cov_frac(o):
            return max(0., min(v['R'], o['R']) - max(v['L'], o['L']))/max(o['width'], 1)
        inside = [o for o in variants if o is not v and o['width'] < v['width'] and cov_frac(o) >= 0.7]
        pairs = [(a, b) for a, b in itertools.combinations(inside, 2) if min(a['R'], b['R']) - max(a['L'], b['L']) <= 0]
        if pairs:
            a, b = pairs[0]
            v['relation'] = f"merged {a['name']}+{b['name']}"
    names = [v['name'] for v in variants] + ['other']
    labels = ['closed' if not c else '+'.join(names[x] for x in c) for c in configs]
    return dict(reads=callable_reads, variants=variants, configs=configs, labels=labels, weights=w, P=P, Pv=Pv,
                ci=ci, closed=float(w[cidx[()]]), other_prev=float(prev[OTHER]),
                other_ci=(float(ci['lo'][OTHER]), float(ci['hi'][OTHER])) if ci else None,
                other_strict=float(np.mean(Pv[:, OTHER] >= 0.9)) if len(Pv) else 0.)


def robust(reads, region, variants, opt, n=5, centre_bp=15):
    """Fraction of n reordered discovery runs that recover each variant (centre within centre_bp, width ratio 0.7-1.43)."""
    hits = np.zeros(len(variants))
    for s in range(n):
        vs, _, _ = discover(reads, region, opt, salt=f'order{s}')
        describe(vs)
        for j, v in enumerate(variants):
            if any(abs((o['L'] + o['R']) - (v['L'] + v['R']))/2 <= centre_bp and 0.7 <= o['width']/max(v['width'], 1) <= 1.43 for o in vs):
                hits[j] += 1
    return hits/max(n, 1)


# ------------------------------------------------------------------ Timer depth states (alternative mode)
def depth_states(reads, thresholds=(175, 300, 500)):
    """Timer's nested width states per callable read: the widest gap overlapping the region >= each threshold."""
    names = {175: 'pioneered', 300: 'merged', 500: 'actuated'}
    callable_reads = sorted([r for r in reads if r['callable']], key=lambda r: r['uid'])
    widest = np.array([r.get('widest', max((g['g1'] - g['g0'] for g in r['gaps']), default=0)) for r in callable_reads], float)
    states = []
    for t in thresholds:
        states.append(dict(name=f'>={t}', depth=names.get(t, f'>={t}'), threshold=t, open=(widest >= t).astype(int)))
    return callable_reads, widest, states
