"""Per-read nucleosome-bounded gaps, NFR profiles and per-read openness (EXPERIMENTAL preview).

NFR on a molecule = the gap between consecutive nucleosome calls of at least ``NUC_MIN`` bp (Timer preprint,
Catalano et al. 2026, Methods "Reconstruction of centered NFRs and opening width"). Factor-sized protections
inside the gap do not split it; they are kept as annotations only.
"""
from __future__ import annotations

import numpy as np

NUC_MIN = 90       # nucleosome calls shorter than this are not nucleosomes for NFR purposes
EDGE_CAP = 30      # an edge is uncertain between the nucleosome boundary and the first mark, capped at this
PROFILE_STEP = 10


def units_of(payload):
    """All evidence units of a FiberHMM consensus payload, ordered by unit_id (never by BAM order)."""
    out = []
    for stratum in payload['strata']:
        for u in stratum['units']:
            out.append(dict(u, dataset=stratum['dataset_id'], chemistry=stratum.get('chemistry')))
    return sorted(out, key=lambda u: u['unit_id'])


def channel(u):
    return f"{u['dataset']}::{u['strand']}"


def nucleosomes(u, nuc_min=NUC_MIN):
    return sorted((int(a), int(b)) for a, b in u.get('raw_nuc_intervals') or () if b - a >= nuc_min)


def read_gaps(u, region, min_gap_bp, nuc_min=NUC_MIN):
    """(callable, gaps) for one unit; a gap is a nucleosome-bounded accessible stretch overlapping ``region``.

    callable: the read has a nucleosome starting before the region and one ending after it, so both edges of any
    gap that overlaps the region are observed. Edge ranges run from the nucleosome boundary to the first (last)
    modification inside the gap, capped at EDGE_CAP; the features are the range midpoints."""
    r0, r1 = region
    nuc = nucleosomes(u, nuc_min)
    if not nuc or nuc[0][0] >= r0 or nuc[-1][1] <= r1:
        return False, []
    pos = np.asarray(u.get('positions') or (), dtype=np.int64)
    hit = np.asarray(u.get('hits') or (), dtype=np.int64) > 0
    hp = pos[hit] if len(pos) else pos
    out = []
    for (_a0, b0), (a1, _b1) in zip(nuc, nuc[1:]):
        g0, g1 = b0, a1
        if g1 - g0 < min_gap_bp or g1 <= r0 or g0 >= r1:
            continue
        inside = hp[(hp >= g0) & (hp < g1)]
        f = int(inside.min()) if len(inside) else g0 + EDGE_CAP
        l_ = int(inside.max()) + 1 if len(inside) else g1 - EDGE_CAP
        lr = (g0, min(f, g0 + EDGE_CAP))
        rr = (max(l_, g1 - EDGE_CAP), g1)
        tfs = [(int(a), int(b)) for a, b in u.get('raw_tf_intervals') or () if a >= g0 and b <= g1]
        out.append(dict(g0=int(g0), g1=int(g1), L=(lr[0] + lr[1])/2, R=(rr[0] + rr[1])/2,
                        wL=lr[1] - lr[0], wR=rr[1] - rr[0], lr=lr, rr=rr, hits=int(len(inside)), tfs=tfs))
    return True, out


def collect(units, region, min_gap_bp, max_gaps):
    """Per-read gaps in ``region``. Reads with more than ``max_gaps`` gaps keep the first ones and are flagged."""
    reads = []
    for u in units:
        ok, gaps = read_gaps(u, region, min_gap_bp)
        overflow = max(0, len(gaps) - max_gaps)
        widest = max((g['g1'] - g['g0'] for g in gaps), default=0)    # before truncation (depth mode)
        if overflow:
            gaps = gaps[:max_gaps]
        reads.append(dict(uid=u['unit_id'], ch=channel(u), callable=ok, gaps=gaps, overflow=overflow, widest=widest,
                          read_name=u.get('read_name'), dataset=u['dataset'], strand=u.get('strand'),
                          members=sorted({str(m.get('read_name')) for m in u.get('source_members') or () if m.get('read_name')}),
                          span=(u.get('reference_start'), u.get('reference_end'))))
    return reads


def profile_counts(units, lo, hi, step=PROFILE_STEP, long_gaps=None):
    """Per position of a ``step`` grid over [lo, hi): callable reads (inside the read's nucleosome-bounded span) and
    the reads inside a gap >= 100 / >= 175 bp, as counts. ``long_gaps`` (a list): filled with the (start, end) of
    every >= 175-bp gap that overlaps the window."""
    x = np.arange(lo, hi, step)
    cov = np.zeros(len(x)); g100 = np.zeros(len(x)); g175 = np.zeros(len(x))
    for u in units:
        nuc = nucleosomes(u)
        if not nuc:
            continue
        s, e = nuc[0][0], nuc[-1][1]
        cov += (x >= s) & (x < e)
        for (_a0, b0), (a1, _b1) in zip(nuc, nuc[1:]):
            if a1 > b0 and a1 > lo and b0 < hi:
                m = (x >= b0) & (x < a1)
                if a1 - b0 >= 100:
                    g100 += m
                if a1 - b0 >= 175:
                    g175 += m
                    if long_gaps is not None:
                        long_gaps.append((b0, a1))
    return x, cov, g100, g175


def nfr_profile(units, lo, hi, step=PROFILE_STEP):
    """Per-position callable coverage and the fraction of callable reads inside a gap >= 100 / >= 175 bp."""
    x, cov, g100, g175 = profile_counts(units, lo, hi, step)
    return x, cov, g100/np.maximum(cov, 1), g175/np.maximum(cov, 1)


# ---------------------------------------------------------------- NFR region detection
MIN_CALLABLE = 20          # callable reads a position needs to be tested
MIN_WIDTH = 60             # shortest NFR region kept (bp)
BACKGROUND_QUANTILE = .2   # background: this quantile of the >= 175-bp share over the window's callable positions
BACKGROUND_FLOOR = .005    # ... never below this (a window without any long gap still has a nonzero background)
BACKGROUND_MAX = .15       # above this (the fixed rule's NFR level) the window has too little closed chromatin for a background
BACKGROUND_MIN_POSITIONS = 20   # callable positions needed to estimate it (200 bp on the 10-bp grid)
OPENING_SHARE = .5         # neighbouring peaks are one NFR when this share of the long gaps reach both


def _runs(on):
    """[i, j) index runs where ``on`` holds."""
    out, i, n = [], 0, len(on)
    while i < n:
        if on[i]:
            j = i
            while j < n and on[j]:
                j += 1
            out.append((i, j)); i = j
        else:
            i += 1
    return out


def split_by_openings(x, f, long_gaps, i, j, share=OPENING_SHARE, min_len=1):
    """Split the run [i, j) into NFR regions by the openings that make them. Walking the run's local maxima of ``f``
    left to right, a maximum joins the current region when at least ``share`` of the >= 175-bp gaps at the lower of
    it and the region's top also reach the higher one (the same openings); otherwise it starts a new region, and
    the cut is at the lowest point between them. At the run's two ends, the stretch beyond the outermost maximum
    whose gaps mostly (``share``) do not reach its region's top is a shoulder with no peak of its own (a rare opening
    beside a common NFR, raised by the common one's wider openings) and is split off too. A rare NFR next to a
    common one is made of other openings, so it becomes its own region; a common NFR's wider openings and noise
    bumps stay part of it. Pieces shorter than ``min_len`` positions are dropped (a shoulder that short stays
    attached). Returns [(i, j), ...] in order."""
    if j - i < min_len:
        return []
    if share <= 0 or not len(long_gaps) or j - i < 2:
        return [(i, j)]
    seg = np.asarray(f[i:j], float)
    n = len(seg)
    peaks = [m for m in range(n) if (m == 0 or seg[m] > seg[m - 1]) and (m == n - 1 or seg[m] >= seg[m + 1])]
    g0, g1 = long_gaps[:, 0], long_gaps[:, 1]
    keep = (g1 > x[i]) & (g0 <= x[j - 1])            # the gaps that touch this run
    g0, g1 = g0[keep], g1[keep]

    def reaches(m, top):
        """Whether at least ``share`` of the long gaps at position m also cover position top."""
        at = (g0 <= x[i + m]) & (g1 > x[i + m])
        if not at.any():
            return True
        return (at & (g0 <= x[i + top]) & (g1 > x[i + top])).sum() >= share*at.sum()

    def same_openings(m1, m2):
        return reaches(m1, m2) if seg[m1] <= seg[m2] else reaches(m2, m1)

    groups = []                                       # [first peak, last peak, top peak]
    for m in peaks:
        if groups and same_openings(m, groups[-1][2]):
            g = groups[-1]
            g[1] = m
            if seg[m] > seg[g[2]]:
                g[2] = m
        else:
            groups.append([m, m, m])
    cuts = [0]
    for g, h in zip(groups, groups[1:]):
        lo_, hi_ = g[1], h[0]
        cuts.append(lo_ + int(np.argmin(seg[lo_:hi_ + 1])))
    cuts.append(n)
    # shoulders at the run's ends: walk out from the outermost maximum while the gaps still reach the region's top
    first, last = groups[0], groups[-1]
    a = first[0]
    while a > 0 and reaches(a - 1, first[2]):
        a -= 1
    if a >= min_len and first[0] - a + 1 >= 1:
        cuts.insert(1, a)
    b = last[1]
    while b + 1 < n and reaches(b + 1, last[2]):
        b += 1
    if n - (b + 1) >= min_len:
        cuts.insert(len(cuts) - 1, b + 1)
    cuts = sorted(set(cuts))
    return [(i + a, i + b) for a, b in zip(cuts, cuts[1:]) if b - a >= min_len]


def _region(x, f, cov, k, i, j, hi, step, background=None, qv=None):
    p = i + int(np.argmax(f[i:j]))
    return dict(start=int(x[i]), end=min(int(x[j - 1]) + step, int(hi)), peak=round(float(f[p]), 4), peak_at=int(x[p]),
                open_reads=int(k[p]), peak_callable=int(cov[p]),
                background=None if background is None else round(float(background), 4),
                fold=None if background is None else round(float(f[p]/background), 2),
                q=None if qv is None else float(f'{qv[p]:.3g}'))


def detect_nfrs(units, lo, hi, threshold=.15, min_callable=MIN_CALLABLE, min_width=MIN_WIDTH, step=PROFILE_STEP):
    """NFR regions in [lo, hi), the fixed rule: runs where more than ``threshold`` of callable reads are inside a
    >= 175-bp gap."""
    x, cov, _g100, k = profile_counts(units, lo, hi, step)
    f175 = k/np.maximum(cov, 1)
    on = (f175 > threshold) & (cov >= min_callable)
    return [_region(x, f175, cov, k, i, j, hi, step) for i, j in _runs(on) if min(int(x[j - 1]) + step, int(hi)) - int(x[i]) >= min_width]


def _bh(p):
    """Benjamini-Hochberg q-values."""
    p = np.asarray(p, float)
    n = len(p)
    if not n:
        return p
    order = np.argsort(p, kind='stable')
    q = p[order]*n/np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(q, 1)
    return out


def background_share(f175, callable_mask):
    """(background, raw quantile, reason): the window's background share of reads inside a >= 175-bp gap, as a low
    quantile over callable positions floored at BACKGROUND_FLOOR; ``reason`` is '' when it is usable, else why not."""
    vals = f175[callable_mask]
    if len(vals) < BACKGROUND_MIN_POSITIONS:
        return None, None, (f'only {len(vals)} positions have {MIN_CALLABLE} or more callable reads '
                            f'({BACKGROUND_MIN_POSITIONS} needed to estimate the background)')
    raw = float(np.quantile(vals, BACKGROUND_QUANTILE))
    if raw > BACKGROUND_MAX:
        return None, raw, (f'most of this window is open: even its quietest {BACKGROUND_QUANTILE:.0%} of positions have '
                           f'{raw:.0%} of reads inside a >= 175-bp gap, so there is too little closed chromatin to estimate a background')
    return max(raw, BACKGROUND_FLOOR), raw, ''


def detect_nfrs_enriched(units, lo, hi, *, min_fraction=.05, q=.01, min_fold=2., min_open_reads=10,
                         min_callable=MIN_CALLABLE, min_width=MIN_WIDTH, opening_share=OPENING_SHARE, step=PROFILE_STEP):
    """NFR regions in [lo, hi), the depth-aware enrichment rule. Returns (regions, detection).

    Per position: k reads inside a >= 175-bp gap of n callable reads, share f = k/n. The background b is a low
    quantile of f over the window's callable positions (BACKGROUND_QUANTILE, floored at BACKGROUND_FLOOR), so a
    window mostly of strong NFRs still compares against its quietest chromatin. A position is in an NFR when
    f >= min_fraction, k >= min_open_reads, n >= min_callable, f/b >= min_fold, and the one-sided binomial test of k
    against n*b passes Benjamini-Hochberg at ``q`` over the window's callable positions. Runs are split by the
    openings that make them (``split_by_openings``) and kept when at least ``min_width`` bp. When the background cannot be estimated (too few
    callable positions, or a window that is mostly open) the fold and enrichment tests are off, only min_fraction,
    min_open_reads and min_callable apply, and ``detection['note']`` says so."""
    from scipy.stats import binom
    long_gaps = []
    x, cov, _g100, k = profile_counts(units, lo, hi, step, long_gaps)
    long_gaps = np.asarray(long_gaps, dtype=np.int64).reshape(-1, 2)
    f175 = k/np.maximum(cov, 1)
    callable_ = cov >= min_callable
    b, raw, reason = background_share(f175, callable_)
    on = callable_ & (f175 >= min_fraction) & (k >= min_open_reads)
    qv = np.ones(len(x))
    note = ''
    if b is not None:
        p = binom.sf(k[callable_] - 1, cov[callable_].astype(int), b)
        qv[callable_] = _bh(p)
        on &= (f175 >= min_fold*b) & (qv <= q)
    else:
        note = (f'Background not estimated: {reason}. NFRs here use only the share threshold '
                f'(>= {min_fraction:.0%} of callable reads, >= {min_open_reads} reads), without the enrichment test; '
                'widen the view to include closed chromatin.')
    min_len = max(1, -(-min_width//step))
    regions = []
    for i, j in _runs(on):
        for a, c in split_by_openings(x, f175, long_gaps, i, j, opening_share, min_len):
            if min(int(x[c - 1]) + step, int(hi)) - int(x[a]) >= min_width:
                regions.append(_region(x, f175, cov, k, a, c, hi, step, b, qv if b is not None else None))
    detection = dict(method='enrichment', background=None if b is None else round(float(b), 4),
                     background_raw=None if raw is None else round(raw, 4), background_quantile=BACKGROUND_QUANTILE,
                     background_floor=BACKGROUND_FLOOR, background_max=BACKGROUND_MAX, enrichment_test=b is not None,
                     callable_positions=int(callable_.sum()), positions=len(x), step=step,
                     min_fraction=min_fraction, q=q, min_fold=min_fold, min_open_reads=int(min_open_reads),
                     min_callable=min_callable, min_width=min_width, opening_share=opening_share, note=note)
    return regions, detection


def read_covariates(units, lo, hi, step=PROFILE_STEP, min_bp=200):
    """Per unit: non-nucleosome fraction of its nucleosome-bounded coverage inside [lo, hi), on a 10-bp grid."""
    cov = {}
    for u in units:
        if u['unit_id'] in cov:
            continue
        nuc = nucleosomes(u)
        if not nuc:
            continue
        s, e = max(lo, nuc[0][0]), min(hi, nuc[-1][1])
        if e - s < min_bp:
            continue
        x = np.arange(s, e, step); m = np.zeros(len(x), bool)
        for a, b in nuc:
            m |= (x >= a) & (x < b)
        cov[u['unit_id']] = dict(open=float(1 - m.mean()), ch=channel(u), read=u.get('read_name'), x=x, closed=m, span=(int(s), int(e)))
    return cov


def covariate_from_nucs(nucs, span, ch, step=PROFILE_STEP):
    """``read_covariates``' entry for one unit rebuilt from its stored nucleosome calls and span (the analysis views
    recompute per-read openness from a saved run; the stored calls must cover the span)."""
    s, e = span
    x = np.arange(s, e, step); m = np.zeros(len(x), bool)
    for a, b in nucs:
        m |= (x >= a) & (x < b)
    return dict(open=float(1 - m.mean()) if len(m) else float('nan'), ch=ch, x=x, closed=m, span=(int(s), int(e)))


def access_matrix(units, lo, hi, step=PROFILE_STEP):
    """Exploratory accessibility features: 10-bp bins, 1 = not inside a >= 90-bp nucleosome call."""
    x = np.arange(lo, hi, step)
    M = np.zeros((len(units), len(x)), np.float32)
    for i, u in enumerate(units):
        m = np.ones(len(x), bool)
        for a, b in nucleosomes(u):
            m &= ~((x >= a) & (x < b))
        M[i] = m
    return x, M
