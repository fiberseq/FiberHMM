"""Element co-accessibility statistics (EXPERIMENTAL preview).

Elements are agnostic: an NFR variant, a whole NFR, a Timer depth state, or a footprint class. Each element has a
per-read binary state over the reads that span it (``state``: unit_id -> 0/1) and, for footprint classes, the read's
channel (a DAF strand is one channel; classes are supported per channel).

Pair test (reads spanning both elements; Timer's "shared" rule: a read where one gap covers both NFR elements'
centres is excluded; physically overlapping NFR x NFR and class x class pairs are not tested):

* 2x2 table, Haldane log2 OR with a Woolf 95% CI, Fisher exact p (pooled; what Timer Fig. S10 reports);
* strata = channel x quantile bins of ~``per_bin`` reads of per-read openness (non-nucleosome fraction over the
  window, *excluding the two tested elements*); with fewer than 10 background bins left the pair is stratified by
  channel only and flagged;
* inference: the exact conditional test of independence given the strata (sum of hypergeometrics; Fisher's
  two-sided rule; equals Fisher's exact p with one stratum);
* effect: Mantel-Haenszel OR over the same strata with the Robins-Breslow-Greenland CI (openness-adjusted);
* BH across all tested pairs; classes: co-accessible / anti (q <= 0.1 and the MH CI excludes 0),
  independence-compatible (MH 95% CI within 0.5-2x and including 1), unresolved.

Combinations: reads spanning all chosen elements, pattern counts against independence (product of margins) and a
curveball null (Strona et al. 2014) that fixes every read's number of open elements and every element's openness.
The null does not preserve pairwise associations: enrichment is beyond margins and openness, not beyond pairs.
"""
from __future__ import annotations

import itertools
from collections import Counter

import numpy as np
from scipy.stats import fisher_exact, hypergeom

Z95 = 1.959964
PER_BIN = 50


def openness_strata(op, ch, per_bin=PER_BIN):
    """Strata = channel x openness quantile bins of ~per_bin reads (at least 5 bins)."""
    out = np.empty(len(op), object)
    for c in np.unique(ch):
        m = ch == c
        nb = max(5, int(m.sum())//per_bin)
        qb = np.quantile(op[m], np.linspace(0, 1, nb + 1)[1:-1])
        out[m] = [f'{c}|{b}' for b in np.searchsorted(qb, op[m], side='right')]
    return out.astype(str)


def log2_or(t):
    a, b, c, d = (x + 0.5 for x in t)
    lor = np.log(a*d/(b*c)); se = np.sqrt(1/a + 1/b + 1/c + 1/d)
    return lor/np.log(2), (lor - Z95*se)/np.log(2), (lor + Z95*se)/np.log(2)


def table(x, y):
    return (int(((x == 1) & (y == 1)).sum()), int(((x == 1) & (y == 0)).sum()),
            int(((x == 0) & (y == 1)).sum()), int(((x == 0) & (y == 0)).sum()))


def mantel_haenszel(x, y, strata):
    """MH log2 OR and RGB 95% CI over strata (strata with a single level of x or y contribute nothing)."""
    R = S = PR = PS_QR = QS = 0.
    for s in np.unique(strata):
        m = strata == s; n = m.sum()
        if n < 2:
            continue
        a, b, c, d = table(x[m], y[m])
        r, s_ = a*d/n, b*c/n
        p, q = (a + d)/n, (b + c)/n
        R += r; S += s_; PR += p*r; PS_QR += p*s_ + q*r; QS += q*s_
    if R == 0 or S == 0:
        return np.nan, np.nan, np.nan
    lor = np.log(R/S)
    se = np.sqrt(PR/(2*R**2) + PS_QR/(2*R*S) + QS/(2*S**2))
    return lor/np.log(2), (lor - Z95*se)/np.log(2), (lor + Z95*se)/np.log(2)


def stratified_perm(x, y, strata, n_perm, rng):
    """Null pooled log2 ORs with y permuted within strata (display only: the null centre of the pooled OR)."""
    Y = np.repeat(y[None, :], n_perm, 0)
    for s in np.unique(strata):
        idx = np.where(strata == s)[0]
        if len(idx) > 1:
            perm = np.argsort(rng.random((n_perm, len(idx))), 1)
            Y[:, idx] = y[idx][perm]
    a = (Y[:, x == 1] == 1).sum(1) + .5; b = (Y[:, x == 1] == 0).sum(1) + .5
    c = (Y[:, x == 0] == 1).sum(1) + .5; d = (Y[:, x == 0] == 0).sum(1) + .5
    return np.log2(a*d/(b*c))


def exact_stratified(x, y, strata):
    """Exact conditional test given strata (all margins fixed within each stratum): the 11 count is a sum of
    independent hypergeometrics; two-sided p = probability of outcomes no more likely than the observed one.
    Returns (p, expected 11, observed 11)."""
    dist = np.array([1.]); lo_tot = 0; t_obs = 0; e = 0.
    for s in np.unique(strata):
        m = strata == s; n = int(m.sum()); a = int(x[m].sum()); b = int(y[m].sum())
        t_obs += int((x[m] & y[m]).sum())
        lo, hi = max(0, a + b - n), min(a, b)
        pmf = hypergeom.pmf(np.arange(lo, hi + 1), n, b, a)
        dist = np.convolve(dist, pmf); lo_tot += lo; e += a*b/max(n, 1)
    k = t_obs - lo_tot
    p = float(dist[dist <= dist[k]*(1 + 1e-7)].sum())
    return min(1., p), e, t_obs


def bh(p):
    p = np.asarray(p, float); n = len(p)
    if n == 0:
        return np.zeros(0)
    o = np.argsort(p); q = np.empty(n)
    q[o] = np.minimum.accumulate((p[o]*n/np.arange(1, n + 1))[::-1])[::-1]
    return np.minimum(q, 1)


def classify(lo, hi, q, q_max=0.1):
    if q <= q_max and lo > 0:
        return 'co-accessible'
    if q <= q_max and hi < 0:
        return 'anti'
    if -1 <= lo <= 0 <= hi <= 1:
        return 'independence-compatible'
    return 'unresolved'


def nested(a, b):
    """'inside' = the class lies within the NFR element; 'edge' = partial overlap; '' = disjoint or same kind."""
    if a['kind'] == b['kind'] or min(a['end'], b['end']) - max(a['start'], b['start']) <= 0:
        return ''
    tf, nf = (a, b) if a['kind'] == 'tf' else (b, a)
    return 'inside' if tf['start'] >= nf['start'] and tf['end'] <= nf['end'] else 'edge'


def overlapping(a, b):
    return min(a['end'], b['end']) - max(a['start'], b['start']) > 0


def element_open_excluding(c, ex):
    """Openness excluding the bins inside the given intervals (an element cannot define its own stratum)."""
    keep = np.ones(len(c['x']), bool)
    for a, b in ex:
        keep &= ~((c['x'] >= a) & (c['x'] < b))
    return float(1 - c['closed'][keep].mean()) if keep.sum() >= 10 else np.nan


def _shared(A, B, gaps, uids):
    ca = (A['start'] + A['end'])/2; cb = (B['start'] + B['end'])/2
    keep, shared = [], 0
    for u in uids:
        if any(g0 <= min(ca, cb) and g1 >= max(ca, cb) for g0, g1 in gaps.get(u, [])):
            shared += 1
        else:
            keep.append(u)
    return keep, shared


def pair_table(els, gaps, cov, *, scope='all', clusters=None, n_perm=500, seed=7, min_reads=50, min_marginal=10,
               per_bin=PER_BIN, q_max=0.1, pad_bp=0, progress=None):
    """All testable pairs. ``scope``: 'all', or 'nfr' (pairs with at least one NFR element).
    ``clusters``: None, or callable(uids, exclude_intervals) -> {uid: label} (the within-cluster check)."""
    rng = np.random.default_rng(seed); rows = []; skipped = []
    combos = list(itertools.combinations(els, 2))
    for done, (A, B) in enumerate(combos):
        if progress and done % 10 == 0:
            progress(done, len(combos))
        if scope == 'nfr' and A['kind'] == 'tf' and B['kind'] == 'tf':
            continue
        if A['kind'] == B['kind'] and overlapping(A, B):
            skipped.append(dict(a=A['id'], b=B['id'], reason='overlap'))
            continue
        uids = sorted(set(A['state']) & set(B['state']) & set(cov))
        if A['kind'] == 'tf' and B['kind'] == 'tf':
            uids = [u for u in uids if A['channel'][u] == B['channel'][u]]
        shared = 0
        if A['kind'] == 'nfr' and B['kind'] == 'nfr':
            uids, shared = _shared(A, B, gaps, uids)
        if len(uids) < min_reads:
            skipped.append(dict(a=A['id'], b=B['id'], reason=f'spanning reads {len(uids)} < {min_reads}'))
            continue
        # per-read openness without the two tested elements; a read with < 10 background bins has none and is left
        # out of this pair (never: the whole pair loses its openness adjustment)
        ex = [(A['start'] - pad_bp, A['end'] + pad_bp), (B['start'] - pad_bp, B['end'] + pad_bp)]
        op_all = {u: element_open_excluding(cov[u], ex) for u in uids}
        no_background = sum(1 for u in uids if np.isnan(op_all[u]))
        uids = [u for u in uids if not np.isnan(op_all[u])]
        if len(uids) < min_reads:
            skipped.append(dict(a=A['id'], b=B['id'], reason=f'spanning reads with background {len(uids)} < {min_reads}'))
            continue
        x = np.array([A['state'][u] for u in uids]); y = np.array([B['state'][u] for u in uids])
        if min(x.sum(), (1 - x).sum(), y.sum(), (1 - y).sum()) < min_marginal:
            skipped.append(dict(a=A['id'], b=B['id'], reason=f'a marginal state has < {min_marginal} reads'))
            continue
        t = table(x, y)
        op = np.array([op_all[u] for u in uids])
        ch = np.array([cov[u]['ch'] for u in uids])
        strata = openness_strata(op, ch, per_bin); adjust = 'openness x channel'
        l2, lo, hi = log2_or(t)
        _, fp = fisher_exact([[t[0], t[1]], [t[2], t[3]]])
        null = stratified_perm(x, y, strata, n_perm, rng)
        pp, e11, _o11 = exact_stratified(x, y, strata)
        mh = mantel_haenszel(x, y, strata)
        row = dict(a=A['id'], b=B['id'], kind_a=A['kind'], kind_b=B['kind'], n=len(uids), shared=shared, table=list(t),
                   log2or=float(l2), lo=float(lo), hi=float(hi), fisher_p=float(fp), null_median=float(np.median(null)),
                   null_lo=float(np.percentile(null, 2.5)), null_hi=float(np.percentile(null, 97.5)), p_exact=float(pp),
                   mh=float(mh[0]), mh_lo=float(mh[1]), mh_hi=float(mh[2]), adjust=adjust, exp11=float(e11), obs11=int(_o11),
                   no_background=no_background,
                   nested=nested(A, B), dist=float(abs((A['start'] + A['end'])/2 - (B['start'] + B['end'])/2)))
        if clusters is not None:
            lab = clusters(uids, ex) if callable(clusters) else clusters
            cl = np.array([lab.get(u, -1) for u in uids]); m = cl >= 0
            row['cluster'] = (float(mantel_haenszel(x[m], y[m], np.array([f'{c}|{v}' for c, v in zip(ch[m], cl[m])]))[0])
                              if m.sum() > 50 else float('nan'))
            row['cluster_n'] = int(m.sum())
        rows.append(row)
    q = bh([r['p_exact'] for r in rows])
    for r, qq in zip(rows, q):
        r['q'] = float(qq); r['class'] = classify(r['mh_lo'], r['mh_hi'], qq, q_max)
        r['separation'] = bool(np.isnan(r['mh']))
        if r['separation'] and qq <= q_max:
            # complete separation within strata: the MH effect is undefined but the exact test is valid
            r['class'] = 'co-accessible' if r['obs11'] > r['exp11'] else 'anti'
    return rows, skipped


# ------------------------------------------------------------------ combinations
def curveball(M, n_samples, rng, burn=None):
    """Strona et al. 2014 curveball: binary matrices with the same row and column sums."""
    rows = [set(np.where(r)[0]) for r in M]; n = len(rows); k = M.shape[1]
    burn = burn or 5*n
    out = []

    def step():
        i, j = rng.choice(n, 2, replace=False)
        a, b = rows[i], rows[j]
        ua, ub = a - b, b - a
        if not ua or not ub:
            return
        pool = list(ua | ub); rng.shuffle(pool)   # small-int set order is deterministic
        na = len(ua); A = set(pool[:na]); B = set(pool[na:])
        common = a & b
        rows[i] = common | A; rows[j] = common | B
    for _ in range(burn):
        step()
    thin = max(1, n//2)
    for _ in range(n_samples):
        for _ in range(thin):
            step()
        X = np.zeros((n, k), np.int8)
        for i, r in enumerate(rows):
            X[i, list(r)] = 1
        out.append(X)
    return out


def combinations(els, gaps=None, n_samples=300, seed=11, q_max=0.1, min_reads=30):
    for A, B in itertools.combinations(els, 2):
        if A['kind'] == B['kind'] and overlapping(A, B):
            raise ValueError(f"{A['id']} and {B['id']} overlap: one opening cannot be two elements")
    uids = sorted(set.intersection(*[set(e['state']) for e in els]))
    n_spanning = len(uids); n_shared = 0
    if gaps is not None:
        cen = [(e['start'] + e['end'])/2 for e in els if e['kind'] == 'nfr']

        def shared(u):
            return any(sum(g0 <= c <= g1 for c in cen) >= 2 for g0, g1 in gaps.get(u, []))
        kept = [u for u in uids if not shared(u)]
        n_shared = len(uids) - len(kept); uids = kept
    M = np.array([[e['state'][u] for e in els] for u in uids], np.int8).reshape(-1, len(els))
    base = dict(elements=[e['id'] for e in els], labels=[e['label'] for e in els], n=len(M), n_spanning=n_spanning,
                n_shared_excluded=n_shared)
    if len(M) < min_reads:
        return dict(base, status='too_few_reads', patterns=[], margins=[], row_sums=[])
    p = M.mean(0); keys = [''.join(map(str, r)) for r in M]
    pats = sorted(set(keys) | {''.join(t) for t in itertools.product('01', repeat=len(els))})
    obs = Counter(keys)
    exp = {s: len(M)*np.prod([p[j] if c == '1' else 1 - p[j] for j, c in enumerate(s)]) for s in pats}
    rng = np.random.default_rng(seed)
    null = {s: [] for s in pats}
    for X in curveball(M, n_samples, rng):
        c = Counter(''.join(map(str, r)) for r in X)
        for s in pats:
            null[s].append(c.get(s, 0))
    out = []
    for s in pats:
        nv = np.array(null[s]); med = np.median(nv)
        pz = (1 + np.sum(np.abs(nv - med) >= abs(obs.get(s, 0) - med)))/(1 + len(nv))
        out.append(dict(pattern=s, obs=int(obs.get(s, 0)), exp_indep=float(exp[s]), null_med=float(med),
                        null_lo=float(np.percentile(nv, 2.5)), null_hi=float(np.percentile(nv, 97.5)), p=float(pz),
                        pinned=bool(np.percentile(nv, 2.5) == np.percentile(nv, 97.5))))
    q = bh([o['p'] for o in out])
    for o, qq in zip(out, q):
        o['q'] = float(qq); o['significant'] = bool(qq <= q_max)
    return dict(base, status='ok', patterns=sorted(out, key=lambda o: (-o['obs'], o['pattern'])), margins=p.tolist(),
                row_sums=np.bincount(M.sum(1), minlength=len(els) + 1).tolist())
