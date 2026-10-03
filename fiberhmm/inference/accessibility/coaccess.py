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
import math
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


def _stratified_perm_reference(x, y, strata, n_perm, rng):
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


def stratified_perm(x, y, strata, n_perm, rng):
    """Null pooled log2 ORs with y permuted within strata (display only: the null centre of the pooled OR).

    The same random draws and permutations as ``_stratified_perm_reference``, counted without building the permuted
    matrix: a permutation within a stratum keeps its count of 1s, so with binary x and y the four cells of every null
    table follow from the 1s that land where x is 1 (exact integer counts)."""
    x, y = np.asarray(x), np.asarray(y)
    if (x.shape != y.shape or x.ndim != 1 or x.dtype.kind not in 'iub' or y.dtype.kind not in 'iub'
            or not np.isin(x, (0, 1)).all() or not np.isin(y, (0, 1)).all()):
        return _stratified_perm_reference(x, y, strata, n_perm, rng)
    x, y = x.astype(np.int64), y.astype(np.int64)
    a1 = np.zeros(n_perm, np.int64)
    for s in np.unique(strata):
        idx = np.where(strata == s)[0]
        if len(idx) > 1:
            perm = np.argsort(rng.random((n_perm, len(idx))), 1)
            a1 += y[idx][perm][:, x[idx] == 1].sum(1)
        else:
            a1 += int(((x[idx] == 1) & (y[idx] == 1)).sum())
    n1, n0, t1 = int((x == 1).sum()), int((x == 0).sum()), int((y == 1).sum())
    a = a1 + .5; b = (n1 - a1) + .5
    c = (t1 - a1) + .5; d = (n0 - (t1 - a1)) + .5
    return np.log2(a*d/(b*c))


def exact_stratified(x, y, strata):
    """Exact conditional test given strata (all margins fixed within each stratum): the 11 count is a sum of
    independent hypergeometrics; two-sided p = probability of outcomes no more likely than the observed one.
    Returns (p, expected 11, observed 11)."""
    dist = np.array([1.]); lo_tot = 0; t_obs = 0; e = 0.
    # every stratum's pmf in one call (an elementwise ufunc: the same values as one call per stratum), then the same
    # convolutions in stratum order
    ks, ns, bs, as_, spans = [], [], [], [], []
    for s in np.unique(strata):
        m = strata == s; n = int(m.sum()); a = int(x[m].sum()); b = int(y[m].sum())
        t_obs += int((x[m] & y[m]).sum())
        lo, hi = max(0, a + b - n), min(a, b)
        k = np.arange(lo, hi + 1)
        ks.append(k); ns.append(np.full(len(k), n)); bs.append(np.full(len(k), b)); as_.append(np.full(len(k), a))
        spans.append(len(k)); lo_tot += lo; e += a*b/max(n, 1)
    if spans:
        pmfs = np.split(hypergeom.pmf(np.concatenate(ks), np.concatenate(ns), np.concatenate(bs), np.concatenate(as_)),
                        np.cumsum(spans)[:-1])
        for pmf in pmfs:
            dist = np.convolve(dist, pmf)
    k = t_obs - lo_tot
    p = float(dist[dist <= dist[k]*(1 + 1e-7)].sum())
    return min(1., p), e, t_obs


def _exact_stratified_reference(x, y, strata):
    """``exact_stratified`` with one pmf call per stratum (the reference it reproduces)."""
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


class _OpennessIndex:
    """Per-read openness without given intervals, for many reads at once, equal bit for bit to
    ``element_open_excluding``. A read qualifies when its grid is integer with a constant positive step (the
    ``arange`` grids of ``gaps.read_covariates``); the bins an interval [a, b) masks are then found with exact integer
    arithmetic (integer x >= a iff x >= ceil(a); x < b iff x < ceil(b)), closed bins are counted from a prefix sum and
    the openness is 1 - closed/kept, the same float operations as the boolean mean. Other reads, and intervals with a
    non-finite (or beyond 2**53) end, fall back to ``element_open_excluding``."""

    def __init__(self, cov):
        self.cov = cov
        self.row, starts, steps, lens, offs, prefix = {}, [], [], [], [], []
        off = 0
        for u, c in cov.items():
            x, closed = np.asarray(c['x']), np.asarray(c['closed'])
            if (x.ndim != 1 or x.dtype.kind not in 'iu' or closed.shape != x.shape or closed.dtype != bool or not len(x)
                    or abs(int(x[0])) >= 2**53 or abs(int(x[-1])) >= 2**53):
                continue
            step = int(x[1] - x[0]) if len(x) > 1 else 1
            if step <= 0 or (len(x) > 1 and not np.array_equal(np.diff(x), np.full(len(x) - 1, step))):
                continue
            self.row[u] = len(starts)
            starts.append(int(x[0])); steps.append(step); lens.append(len(x)); offs.append(off)
            prefix.append(np.r_[0, np.cumsum(closed, dtype=np.int64)]); off += len(x) + 1
        self.start, self.step, self.len, self.off = (np.array(v, np.int64) for v in (starts, steps, lens, offs))
        self.prefix = np.concatenate(prefix) if prefix else np.zeros(0, np.int64)

    def openness(self, uids, ex):
        """{uid: element_open_excluding(cov[uid], ex)} for ``uids``."""
        if len(ex) != 2 or not all(math.isfinite(v) and abs(v) < 2**53 for iv in ex for v in iv):
            return {u: element_open_excluding(self.cov[u], ex) for u in uids}
        fast = [u for u in uids if u in self.row]
        r = np.array([self.row[u] for u in fast], np.int64)
        S, St, N, O = self.start[r], self.step[r], self.len[r], self.off[r]

        def first(v):                                   # first grid index with x >= v (x < v before it)
            return np.clip(-((S - math.ceil(v))//St), 0, N)
        (l1, h1), (l2, h2) = [(first(a), first(b)) for a, b in ex]
        h1 = np.maximum(h1, l1); h2 = np.maximum(h2, l2)
        lo, hi = np.maximum(l1, l2), np.minimum(h1, h2)
        both = hi > lo
        P = self.prefix

        def closed(l, h):
            return P[O + h] - P[O + l]
        masked = (h1 - l1) + (h2 - l2) - np.where(both, hi - lo, 0)
        masked_closed = closed(l1, h1) + closed(l2, h2) - np.where(both, closed(np.minimum(lo, hi), hi), 0)
        kept = N - masked
        kept_closed = (P[O + N] - P[O]) - masked_closed
        ok = kept >= 10
        val = np.full(len(fast), np.nan)
        val[ok] = 1 - kept_closed[ok].astype(np.float64)/kept[ok].astype(np.float64)
        out = dict(zip(fast, val.tolist()))
        return {u: out[u] if u in out else element_open_excluding(self.cov[u], ex) for u in uids}


class _GapIndex:
    """Every read's gaps as flat arrays, for the "shared" rule over many reads at once (``_shared``)."""

    def __init__(self, gaps):
        self.gaps = gaps
        own, g0, g1 = [], [], []
        self.uids = list(gaps)
        for i, u in enumerate(self.uids):
            for a, b in gaps[u]:
                own.append(i); g0.append(a); g1.append(b)
        self.own, self.g0, self.g1 = np.array(own, np.int64), np.array(g0), np.array(g1)
        self.exact = all(isinstance(v, (int, np.integer)) and abs(int(v)) < 2**53 for v in g0 + g1)

    def covering(self, lo, hi):
        """The uids with a gap g0 <= lo and g1 >= hi."""
        hit = np.zeros(len(self.uids), bool)
        if len(self.own):
            hit[self.own[(self.g0 <= lo) & (self.g1 >= hi)]] = True
        return {u for u, h in zip(self.uids, hit.tolist()) if h}


def _shared(A, B, gaps, uids, index=None):
    """(kept, shared) unit lists: a read where one gap covers both elements' centres is "shared" (Timer)."""
    ca = (A['start'] + A['end'])/2; cb = (B['start'] + B['end'])/2
    if index is not None and index.gaps is gaps and index.exact and math.isfinite(ca) and math.isfinite(cb):
        # integer gap ends compare exactly with float centres in numpy as in Python (|values| < 2**53)
        cover = index.covering(min(ca, cb), max(ca, cb))
        return [u for u in uids if u not in cover], [u for u in uids if u in cover]
    keep, shared = [], []
    for u in uids:
        if any(g0 <= min(ca, cb) and g1 >= max(ca, cb) for g0, g1 in gaps.get(u, [])):
            shared.append(u)
        else:
            keep.append(u)
    return keep, shared


def pair_eligibility(A, B, gaps, cov, pad_bp=0, universe=None, _index=None):
    """The reads a pair test uses, and why every other read is left out. ``pair_table`` and the read split of the
    analysis views (``analysis.pair_split``) both call this, so a split's four groups are the test's 2x2 table.

    Returns uids (eligible, sorted), op ({uid: openness without the two elements}), n_spanning (reads left after the
    spanning, coverage, channel and shared rules: the count the spanning-read minimum applies to), and excluded:
    {reason: [uids]} with reasons not_spanning_a / not_spanning_b (no state for that element: censored, abstained or
    off-channel), no_coverage (< 200 bp of nucleosome-bounded coverage in the window), channel (class x class on
    different channels), shared (one opening covers both centres) and no_background (< 10 openness bins left once
    the two elements are masked). ``universe`` (optional): the reads to account for; without it only reads with a
    state for both elements are."""
    excluded = {}
    sa, sb = A['state'], B['state']
    if universe is not None:
        for u in sorted(universe):
            if u not in sa:
                excluded.setdefault('not_spanning_a', []).append(u)
            elif u not in sb:
                excluded.setdefault('not_spanning_b', []).append(u)
    both = sorted(set(sa) & set(sb))
    uids = [u for u in both if u in cov]
    if len(uids) < len(both):
        excluded['no_coverage'] = [u for u in both if u not in cov]
    if A['kind'] == 'tf' and B['kind'] == 'tf':
        off = [u for u in uids if A['channel'][u] != B['channel'][u]]
        if off:
            excluded['channel'] = off
        uids = [u for u in uids if A['channel'][u] == B['channel'][u]]
    if A['kind'] == 'nfr' and B['kind'] == 'nfr':
        uids, shared = _shared(A, B, gaps, uids, None if _index is None else _index[1])
        if shared:
            excluded['shared'] = shared
    n_spanning = len(uids)
    # per-read openness without the two tested elements; a read with < 10 background bins has none and is left out of
    # this pair (never: the whole pair loses its openness adjustment)
    ex = [(A['start'] - pad_bp, A['end'] + pad_bp), (B['start'] - pad_bp, B['end'] + pad_bp)]
    if _index is not None and _index[0].cov is cov:
        op = _index[0].openness(uids, ex)
    else:
        op = {u: element_open_excluding(cov[u], ex) for u in uids}
    nb = [u for u in uids if np.isnan(op[u])]
    if nb:
        excluded['no_background'] = nb
    uids = [u for u in uids if not np.isnan(op[u])]
    return dict(uids=uids, op={u: op[u] for u in uids}, n_spanning=n_spanning, excluded=excluded, ex=ex)


def pair_table(els, gaps, cov, *, scope='all', clusters=None, n_perm=500, seed=7, min_reads=50, min_marginal=10,
               per_bin=PER_BIN, q_max=0.1, pad_bp=0, progress=None):
    """All testable pairs. ``scope``: 'all', or 'nfr' (pairs with at least one NFR element).
    ``clusters``: None, or callable(uids, exclude_intervals) -> {uid: label} (the within-cluster check)."""
    rng = np.random.default_rng(seed); rows = []; skipped = []
    combos = list(itertools.combinations(els, 2))
    index = (_OpennessIndex(cov), _GapIndex(gaps))     # built once: every pair reads the same reads
    for done, (A, B) in enumerate(combos):
        if progress and done % 10 == 0:
            progress(done, len(combos))
        if scope == 'nfr' and A['kind'] == 'tf' and B['kind'] == 'tf':
            continue
        if A['kind'] == B['kind'] and overlapping(A, B):
            skipped.append(dict(a=A['id'], b=B['id'], reason='overlap'))
            continue
        el = pair_eligibility(A, B, gaps, cov, pad_bp, _index=index)
        shared = len(el['excluded'].get('shared', ()))
        if el['n_spanning'] < min_reads:
            skipped.append(dict(a=A['id'], b=B['id'], reason=f"spanning reads {el['n_spanning']} < {min_reads}"))
            continue
        uids, op_all, ex = el['uids'], el['op'], el['ex']
        no_background = len(el['excluded'].get('no_background', ()))
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


def combo_eligibility(els, gaps=None, universe=None):
    """Reads a combination table uses: spanning every element, minus "shared" reads (one gap covers two NFR elements'
    centres). Shared by ``combinations`` and ``analysis.combo_split``. excluded: {not_spanning, shared: [uids]}."""
    uids = sorted(set.intersection(*[set(e['state']) for e in els]))
    excluded = {}
    if universe is not None:
        miss = sorted(set(universe) - set(uids))
        if miss:
            excluded['not_spanning'] = miss
    n_spanning = len(uids)
    if gaps is not None:
        cen = [(e['start'] + e['end'])/2 for e in els if e['kind'] == 'nfr']

        def shared(u):
            return any(sum(g0 <= c <= g1 for c in cen) >= 2 for g0, g1 in gaps.get(u, []))
        sh = [u for u in uids if shared(u)]
        if sh:
            excluded['shared'] = sh
        uids = [u for u in uids if not shared(u)]
    return dict(uids=uids, n_spanning=n_spanning, excluded=excluded)


def combinations(els, gaps=None, n_samples=300, seed=11, q_max=0.1, min_reads=30):
    for A, B in itertools.combinations(els, 2):
        if A['kind'] == B['kind'] and overlapping(A, B):
            raise ValueError(f"{A['id']} and {B['id']} overlap: one opening cannot be two elements")
    el = combo_eligibility(els, gaps)
    uids, n_spanning = el['uids'], el['n_spanning']
    n_shared = len(el['excluded'].get('shared', ()))
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
