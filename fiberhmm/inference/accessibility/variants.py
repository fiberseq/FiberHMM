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
BOOT_CHUNK = 20         # bootstrap replicates per progress report
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


def robust_geometry(Xc, seed, cache=None):
    """Members' robust centre and covariance (MinCovDet, deterministic): k-means cells include neighbours' tails.

    ``cache`` (optional dict): memo of earlier results. MinCovDet is a deterministic function of the members' values,
    their order and the seed, so the key is exactly those (the bytes of ``Xc``); a hit returns copies of the same
    arrays the fit returned. Discovery fits the same members many times (every held-out test re-fits every group)."""
    if cache is None or isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        return _robust_geometry(Xc, seed)        # only an integer seed makes the fit a function of its input
    Xc = np.asarray(Xc)
    key = (Xc.shape, Xc.dtype.str, int(seed), np.ascontiguousarray(Xc).tobytes())
    hit = cache.get(key)
    if hit is None:
        hit = cache[key] = _robust_geometry(Xc, seed)
    return hit[0].copy(), hit[1].copy()


def _robust_geometry(Xc, seed):
    from sklearn.covariance import MinCovDet
    if len(Xc) < 10:
        return np.median(Xc, 0), _floor(np.cov(Xc.T) if len(Xc) > 2 else np.diag([400., 400.]))
    m = MinCovDet(support_fraction=0.75, random_state=seed).fit(Xc)
    return m.location_, _floor(m.covariance_)


def heldout_gain_groups(X, VE, log_other, folds_of, groups_full, groups_reduced, nfold, seed=1, cache=None):
    """Held-out log-likelihood gain of the full over the reduced model, fitted with the final estimator's rule:
    per training fold, each group's geometry is the MinCovDet centre/covariance of its training members, fixed;
    only weights are fitted. ``groups_*`` are lists of index arrays into X. ``cache``: ``robust_geometry``'s memo."""
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
                    geo.append(robust_geometry(X[t], seed, cache))
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


def discover(reads, region, opt, salt='', progress=None, stage=None):
    """Supported variants (list of dicts with mu, cov, stability, gain, exp_reads), diagnostics and the EdgeSpace.

    progress(k, kmax): after each prediction-strength k; stage(name): as the identity merge and the support test start."""
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
    # MinCovDet memo for this discovery: identity merges and support tests re-fit the same members in every fold
    geo_cache = {}

    def mu_of(c):
        return np.median(X[c['idx']], 0)

    def overl(a, b):
        ma, mb = mu_of(a), mu_of(b)
        return min(ma[1], mb[1]) - max(ma[0], mb[0]) > 0

    if stage:
        stage('identity')
    merges = []
    while True:
        best = None
        for i, j in itertools.combinations(range(len(cands)), 2):
            if not overl(cands[i], cands[j]):
                continue
            sel = np.r_[cands[i]['idx'], cands[j]['idx']]
            ni = len(cands[i]['idx']); loc = np.arange(len(sel))
            g = heldout_gain_groups(X[sel], VE[sel], es.log_other, folds[sel], [loc[:ni], loc[ni:]], [loc], opt.folds, opt.seed,
                                    geo_cache)
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
    if stage:
        stage('support')
    dropped = []
    tests = []
    while cands:
        geo_s = [robust_geometry(X[c['idx']], opt.seed, geo_cache) for c in cands]
        F = fit_mixture(X, VE, es.log_other, [m for m, _ in geo_s], covs=[c_ for _, c_ in geo_s], fixed=True)
        tests = []
        for t, c in enumerate(cands):
            groups = [cc['idx'] for cc in cands]
            gain = heldout_gain_groups(X, VE, es.log_other, folds, groups, [gi for s_, gi in enumerate(groups) if s_ != t], opt.folds, opt.seed,
                                       geo_cache)
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
    geo = [robust_geometry(X[c['idx']], opt.seed, geo_cache) for c in cands]
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
def _em_dense(LL, iters=2000, w0=None):
    """The configuration EM on the full read x configuration matrix (the reference ``_em`` reproduces exactly)."""
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


def _span_column_sums_reference(Pu, inv, c0, c1, ncol):
    """Column sums of Pu[inv] in row order, skipping each row's exact zeros outside its span (x + 0.0 == x)."""
    acc = np.zeros(ncol)
    for i in range(len(inv)):
        r = inv[i]
        for j in range(c0[r], c1[r]):
            acc[j] += Pu[r, j]
    return acc


try:
    from numba import njit as _njit
    _span_column_sums = _njit(cache=True, nogil=True)(_span_column_sums_reference)
except ImportError:      # without numba the read average is taken over the rebuilt full matrix
    _span_column_sums = None


class _EMRows:
    """The distinct rows of a read x configuration log-likelihood matrix, each restricted to its span of possible
    configurations, for computing ``_em_dense`` bit for bit at a fraction of the cost.

    A read's possible configurations are those with one label per gap, so every row is -inf outside one contiguous
    span of columns (configurations are ordered by length). Outside the span the dense EM computes exp(-inf) = 0
    exactly; inside it this computes the same elementwise values. Rows are kept at full width (exact zeros outside
    the span) for the row sums, so each row sum adds the same values in the same order as ``_em_dense``. Identical
    rows (closed reads, bootstrap duplicates) are computed once. Rows with NaN or +inf, or with no possible
    configuration, use the dense formula over the whole row.

    The read average: numpy reduces a C-ordered matrix over its rows (the slow axis) by adding row after row, so
    each column's sum is the row-ordered sum of its non-zero entries (adding an exact +0.0 changes nothing). With
    Numba that sum is taken over the spans directly, without rebuilding the read x configuration matrix; the first
    average of every EM is checked bit for bit against numpy's and any difference switches to numpy's."""

    def __init__(self, LL):
        A = np.ascontiguousarray(LL)
        self.n, self.c = A.shape
        _, first, inv = np.unique(A.view(np.dtype((np.void, A.dtype.itemsize*self.c))).ravel(),
                                  return_index=True, return_inverse=True)
        U = A[first]
        live = U > -np.inf                                     # False for -inf and NaN
        dense = np.isnan(U).any(1) | np.isposinf(U).any(1) | ~live.any(1)
        c0 = np.where(dense, 0, live.argmax(1))
        c1 = np.where(dense, self.c, self.c - live[:, ::-1].argmax(1))
        order = np.lexsort((np.arange(len(U)), c1, c0))       # one contiguous block of distinct rows per span
        pos = np.empty(len(order), np.intp); pos[order] = np.arange(len(order))
        self.U, self.inv = U[order], pos[np.asarray(inv).ravel()]
        self.c0, self.c1 = c0[order].astype(np.int64), c1[order].astype(np.int64)
        starts = np.flatnonzero(np.r_[True, (self.c0[1:] != self.c0[:-1]) | (self.c1[1:] != self.c1[:-1])])
        self.blocks = [(int(r0), int(r1), int(self.c0[r0]), int(self.c1[r0]),
                        np.ascontiguousarray(self.U[r0:r1, self.c0[r0]:self.c1[r0]]))
                       for r0, r1 in zip(starts, np.r_[starts[1:], len(U)])]
        # the span sums need numpy's row-after-row reduction: a matrix with one column reduces as a vector
        self.span_sums = _span_column_sums is not None and self.c > 1
        self.checked = False

    @classmethod
    def of(cls, LL):
        LL = np.asarray(LL)
        # a matrix in another memory order makes the dense EM's reductions add in another order: keep it dense
        if LL.ndim != 2 or LL.dtype != np.float64 or 0 in LL.shape or not LL.flags.c_contiguous:
            return None
        return cls(LL)

    def posteriors(self, logw):
        """(``_em_dense``'s p for log-weights ``logw``, one row per distinct row; whether zeros are exact outside spans)."""
        if not np.isfinite(logw).all():
            # a non-finite log-weight reaches every column of the dense formula: compute it as the dense EM does
            a = self.U + logw; m = a.max(1, keepdims=True)
            p = np.exp(a - m); p /= p.sum(1, keepdims=True)
            return p, False
        Pu = np.zeros((len(self.U), self.c))
        for r0, r1, c0, c1, Ub in self.blocks:
            a = Ub + logw[c0:c1]; m = a.max(1, keepdims=True)
            Pu[r0:r1, c0:c1] = np.exp(a - m)
        s = Pu.sum(1, keepdims=True)
        for r0, r1, c0, c1, _ in self.blocks:
            Pu[r0:r1, c0:c1] /= s[r0:r1]                    # outside the span 0/s = 0 (s >= 1: the row max gives exp(0))
        return Pu, True

    def full(self, Pu):
        return Pu[self.inv]

    def mean(self, Pu, spans_exact):
        """``full(Pu).mean(0)``, bit for bit."""
        if not (self.span_sums and spans_exact):
            return self.full(Pu).mean(0)
        nw = _span_column_sums(Pu, self.inv, self.c0, self.c1, self.c)/self.n
        if not self.checked:
            self.checked = True
            ref = self.full(Pu).mean(0)
            if ref.tobytes() != nw.tobytes():
                self.span_sums = False
                return ref
        return nw


def _em(LL, iters=2000, w0=None):
    """The configuration EM: weights and per-read posteriors. Bit-identical to ``_em_dense`` (see ``_EMRows``)."""
    rows = _EMRows.of(LL)
    if rows is None:
        return _em_dense(LL, iters, w0)
    w = np.full(LL.shape[1], 1/LL.shape[1]) if w0 is None else np.asarray(w0, float).copy()
    Pu = None
    for _ in range(iters):
        Pu, exact = rows.posteriors(np.log(np.maximum(w, 1e-300)))
        nw = rows.mean(Pu, exact)
        if np.abs(nw - w).max() < 1e-9:
            w = nw; break
        w = nw
    return w, (None if Pu is None else rows.full(Pu))


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


def quantify(reads, variants, es, opt, progress=None):
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
        for b in range(opt.bootstrap):
            if progress and b % BOOT_CHUNK == 0:
                progress(b, opt.bootstrap)
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


def robust(reads, region, variants, opt, n=5, centre_bp=15, progress=None):
    """Fraction of n reordered discovery runs that recover each variant (centre within centre_bp, width ratio 0.7-1.43)."""
    hits = np.zeros(len(variants))
    for s in range(n):
        if progress:
            progress(s, n)
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
