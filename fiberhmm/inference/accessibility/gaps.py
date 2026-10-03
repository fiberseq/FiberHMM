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


def nfr_profile(units, lo, hi, step=PROFILE_STEP):
    """Per-position callable coverage and the fraction of callable reads inside a gap >= 100 / >= 175 bp."""
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
    return x, cov, g100/np.maximum(cov, 1), g175/np.maximum(cov, 1)


def detect_nfrs(units, lo, hi, threshold=.15, min_callable=20, min_width=60, step=PROFILE_STEP):
    """NFR regions in [lo, hi): runs where more than ``threshold`` of callable reads are inside a >= 175-bp gap."""
    x, cov, _f100, f175 = nfr_profile(units, lo, hi, step)
    on = (f175 > threshold) & (cov >= min_callable)
    runs, i = [], 0
    while i < len(x):
        if on[i]:
            j = i
            while j < len(x) and on[j]:
                j += 1
            a, b = int(x[i]), min(int(x[j - 1]) + step, int(hi))
            if b - a >= min_width:
                runs.append(dict(start=a, end=b, peak=round(float(f175[i:j].max()), 3)))
            i = j
        else:
            i += 1
    return runs


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
