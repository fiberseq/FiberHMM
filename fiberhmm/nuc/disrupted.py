"""Regular-rule overlay and disrupted nucleosomes per molecule (NUC 13.1).

**Model-free regular rule.** On the molecule's eligible mono dyads (at least
8), spacings g, the molecule's NRL (median spacing within 0.7-1.3 x
NRL_pooled, else NRL_pooled), k = max(1, round(g / NRL)): a spacing fits when
|g - k NRL| <= ``seg_rule_tol_bp`` and k <= ``seg_rule_max_k``. Runs of at
least ``seg_rule_min_spacings`` fitting spacings are regular from their first
to their last dyad, outside low-information segments; they are OR-ed with the
decoder's regular segments (:func:`overlay_rule`).

**Undersized runs** replace the composition sub-rich flag. A nucleosome-scale
footprint (a nucleosome call, or a TF call of at least ``undersized_min_bp``:
recall turns footprints under 85 bp into TF calls, and the many shorter TF
calls sit in linkers) is undersized at density-matched z <=
``undersized_z``. Such footprints with enough edge sites, not truncated and
inside the supported density range score
log(p1 / p0) when undersized and log((1 - p1) / (1 - p0)) otherwise, where
p0(lambda) is the dataset's undersized rate at that density (calibration) and
p1 = max(``undersized_p1``, 2 p0); other footprints score 0. Runs are the
maximal-scoring segments (Ruzzo-Tompa) per piece (molecules split at MSPs
longer than ``long_msp_bp``) with score >= ``undersized_min_llr`` and at
least ``undersized_min_count`` undersized footprints. Scores are
log-likelihood ratios, so the chance tail of the maximal segment score is
about exp(-S) whatever p0 is: false runs do not follow density.

**Missing nucleosome.** Inside a regular segment (the array whose dyad grid
predicts the nucleosome; the grid spans the two-repeat gap a missing core
leaves), two consecutive nucleosome calls, both confident (not can't-tell or
truncated; no nucleosome call of any kind between them), whose open stretch
is ``missing_len_bp`` long and whose nearest dyads are 2q apart within
``missing_grid_tol_bp`` (q: the segment's repeat), and whose marks where
the grid puts the missing nucleosome (the predicted dyad +/- 60 bp, inside
the open stretch) favour accessible over protected by
``missing_min_llr`` nats (accessible rate: the flanks' local lambda;
protected rate: the dataset's in-core mark rate at that density,
calibration). Longer openings are the NFR module's.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from fiberhmm.nuc.params import DEFAULT_PARAMS, NucParams
from fiberhmm.nuc.segment import LOWINFO, REGULAR, Segment, _pieces
from fiberhmm.nuc.sizes import CANT_TELL, EDGE_TRUNCATED, MONO

UNDERSIZED, MISSING = 16, 17
EVENT_NAMES = {UNDERSIZED: 'undersized', MISSING: 'missing'}
DEFAULT_UNDERSIZED_RATE = 0.05
DEFAULT_PROTECTED_RATE = 0.01


@dataclass
class Event:
    """A per-molecule disruption event (molecular coordinates)."""
    start: int
    end: int
    kind: int            # UNDERSIZED / MISSING
    score: float         # LLR (nats)
    count: int           # undersized footprints in the run; marks in the open stretch

    def as_dict(self):
        return {'start': self.start, 'end': self.end, 'kind': EVENT_NAMES[self.kind],
                'score': self.score, 'count': self.count}


# ---------------------------------------------------------------------------
# regular rule
# ---------------------------------------------------------------------------

def _nuc_codes(mol, fp) -> np.ndarray:
    codes = np.full(len(mol.nucs), CANT_TELL)
    sel = ~fp['is_tf']
    codes[fp['src_index'][sel]] = fp['coarse'][sel]
    return codes


def rule_runs(mol, fp, nrl_pooled: float, params: NucParams = DEFAULT_PARAMS):
    """Model-free regular runs: ``(dyad centres, [(start, end, q_hat, n_dyads), ...])``."""
    from fiberhmm.nuc.spacing import eligible_dyads
    if fp is None:
        return np.zeros(0), []
    d = eligible_dyads(mol, nrl_pooled, params, coarse_codes=_nuc_codes(mol, fp))
    cen = d.centres
    if len(cen) < params.seg_rule_min_spacings + 2:
        return cen, []
    g = np.diff(cen)
    near = g[(g > 0.7 * nrl_pooled) & (g < 1.3 * nrl_pooled)]
    nrl = float(np.median(near)) if len(near) else float(nrl_pooled)
    k = np.maximum(1, np.round(g / nrl))
    ok = (np.abs(g - k * nrl) <= params.seg_rule_tol_bp) & (k <= params.seg_rule_max_k)
    out, i = [], 0
    while i < len(ok):
        j = i
        while j + 1 < len(ok) and ok[j + 1] == ok[i]:
            j += 1
        if ok[i] and j - i + 1 >= params.seg_rule_min_spacings:
            q = float(np.mean(g[i:j + 1] / k[i:j + 1]))
            out.append((int(cen[i]), int(cen[j + 1]), round(q, 1), j - i + 2))
        i = j + 1
    return cen, out


def _subtract(iv, cut):
    out = []
    for a, b in iv:
        pieces = [(a, b)]
        for c0, c1 in cut:
            nxt = []
            for p0, p1 in pieces:
                if c1 <= p0 or c0 >= p1:
                    nxt.append((p0, p1))
                    continue
                if c0 > p0:
                    nxt.append((p0, c0))
                if c1 < p1:
                    nxt.append((c1, p1))
            pieces = nxt
        out += pieces
    return out


def _union(iv):
    iv = sorted((int(a), int(b)) for a, b in iv if b > a)
    out: List[List[int]] = []
    for a, b in iv:
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def _track_mean(track, a: int, b: int) -> float:
    """bp-weighted mean step P(regular) over molecular [a, b) (NaN without steps)."""
    w = num = 0.0
    for s, e, p in track:
        ov = np.clip(np.minimum(e, b) - np.maximum(s, a), 0, None)
        w += float(ov.sum())
        num += float((ov * p).sum())
    return num / w if w > 0 else float('nan')


def overlay_rule(segs: List[Segment], runs, dyads: np.ndarray, track=()) -> List[Segment]:
    """OR the rule's runs (minus low-information segments) into the segments.

    The result still partitions the covered stretches. Segments the rule does
    not touch are returned unchanged. A changed or new regular segment gets
    the bp-weighted posterior of its parts (decoder segments: theirs; rule
    parts: the mean step P(regular)), the bp-weighted repeat of its parts
    (decoder q-hat, rule mean k-normalised spacing), its eligible dyad count
    and ``source`` 'rule' or 'hmm+rule'."""
    if not runs:
        return segs
    low = [(s.start, s.end) for s in segs if s.state == LOWINFO]
    rule = _union(_subtract([(a, b) for a, b, _q, _n in runs], low))
    if not rule:
        return segs
    rule_q = [(a, b, q) for a, b, q, _n in runs]
    pts = sorted({p for s in segs for p in (s.start, s.end)} | {p for iv in rule for p in iv})
    ordered = sorted(segs, key=lambda s: (s.start, s.end))
    pieces = []
    for x0, x1 in zip(pts[:-1], pts[1:]):
        cover = next((s for s in ordered if s.start <= x0 and s.end >= x1), None)
        in_rule = any(a <= x0 and b >= x1 for a, b in rule)
        if in_rule:
            pieces.append([x0, x1, REGULAR, cover, True])
        elif cover is not None:
            pieces.append([x0, x1, cover.state, cover, False])
    groups: List[List[list]] = []
    for p in pieces:
        if groups and groups[-1][-1][1] == p[0] and groups[-1][-1][2] == p[2]:
            groups[-1].append(p)
        else:
            groups.append([p])
    out: List[Segment] = []
    for g in groups:
        start, end, state = g[0][0], g[-1][1], g[0][2]
        covers = {id(p[3]) for p in g}
        only = g[0][3]
        if (not any(p[4] for p in g)) and len(covers) == 1 and only is not None \
                and only.start == start and only.end == end:
            out.append(only)                     # untouched
            continue
        w = post = qw = qsum = 0.0
        llrs = []
        src = set()
        for x0, x1, st, cov, by_rule in g:
            n = x1 - x0
            if st == REGULAR and cov is not None and cov.state == REGULAR:
                p, q = cov.posterior, cov.q_hat
                llrs.append(cov.llr)
                src.add('hmm')
                if by_rule:
                    src.add('rule')
            elif st == REGULAR:
                p = _track_mean(track, x0, x1)
                q = next((qq for a, b, qq in rule_q if a <= x0 and b >= x1), None)
                src.add('rule')
            else:
                p, q = cov.posterior, None
            if np.isfinite(p):
                post += p * n
                w += n
            if q is not None:
                qsum += q * n
                qw += n
        n_dy = int(((dyads >= start) & (dyads <= end)).sum())
        inside = {id(p[3]): p[3].n_steps for p in g
                  if p[3] is not None and p[3].start >= start and p[3].end <= end}
        left = next((s for s in ordered if s.start == start), None)
        right = next((s for s in ordered if s.end == end), None)
        seg = Segment(int(start), int(end), state, False, post / w if w else float('nan'),
                      max(llrs) if llrs else float('nan'),
                      round(qsum / qw, 1) if (state == REGULAR and qw) else None, n_dy,
                      int(sum(inside.values())),
                      source=('hmm+rule' if src == {'hmm', 'rule'} else ('rule' if 'rule' in src else 'hmm'))
                      if state == REGULAR else (only.source if only is not None else ''))
        seg.boundary = (left.boundary[0] if left is not None else float('nan'),
                        right.boundary[1] if right is not None else float('nan'))
        seg.boundary_ci = ((left.boundary_ci[0], left.boundary_ci[1]) if left is not None else (-1, -1)) + \
                          ((right.boundary_ci[2], right.boundary_ci[3]) if right is not None else (-1, -1))
        out.append(seg)
    return out


# ---------------------------------------------------------------------------
# calibration: density-matched background rates
# ---------------------------------------------------------------------------

def _binned_rate(lam: np.ndarray, hit: np.ndarray, n_bins: int = 8, min_per_bin: int = 200,
                 weights: Optional[np.ndarray] = None) -> Dict[str, list]:
    """Rate of ``hit`` per quantile bin of log lambda (inner edges, rates, counts)."""
    lam = np.asarray(lam, float)
    hit = np.asarray(hit, float)
    w = np.ones(len(lam)) if weights is None else np.asarray(weights, float)
    if len(lam) == 0:
        return {}
    x = np.log(np.clip(lam, 1e-6, 1.0))
    nb = int(max(1, min(n_bins, w.sum() // min_per_bin)))
    inner = np.quantile(x, np.linspace(0, 1, nb + 1)[1:-1]) if nb > 1 else np.zeros(0)
    inner = np.unique(inner)
    k = np.searchsorted(inner, x, side='right')
    rates, ns = [], []
    for b in range(len(inner) + 1):
        s = k == b
        tot = float(w[s].sum())
        rates.append(float((float((hit[s] * w[s]).sum()) + 0.5) / (tot + 1.0)))
        ns.append(tot)
    return {'log_lambda_edges': [float(v) for v in inner], 'rate': rates, 'n': ns}


def _lookup(table: Dict[str, list], lam, default: float) -> np.ndarray:
    lam = np.asarray(lam, float)
    if not table or not table.get('rate'):
        return np.full(lam.shape, default)
    x = np.log(np.clip(lam, 1e-6, 1.0))
    k = np.searchsorted(np.asarray(table['log_lambda_edges'], float), x, side='right')
    return np.asarray(table['rate'], float)[np.clip(k, 0, len(table['rate']) - 1)]


def undersized_eligible(fp, cal, params: NucParams = DEFAULT_PARAMS) -> np.ndarray:
    return (~fp['truncated'] & (np.minimum(fp['n_left'], fp['n_right']) >= params.min_edge_sites)
            & (fp['length'] >= params.undersized_min_bp)
            & np.asarray(cal.is_supported(fp['lambda']), dtype=bool))


def fit_disruption(molecules, cal, params: NucParams = DEFAULT_PARAMS) -> dict:
    """Density-matched background rates for the disruption calls.

    ``undersized``: share of eligible footprints (nucleosome calls and TF
    calls of at least ``undersized_min_bp``) with z <= ``undersized_z``, per
    lambda_i bin. ``protected``: mark rate
    at informative sites inside confident mono nucleosome calls (15 bp in
    from each edge), per lambda_i bin."""
    from fiberhmm.nuc.molecule import footprint_calls
    lu, hu, lp, kp, npr = [], [], [], [], []
    for m in molecules:
        fp = footprint_calls(m, params, cal)
        if fp is None:
            continue
        ok = undersized_eligible(fp, cal, params)
        lu.append(fp['lambda'][ok])
        hu.append(fp['z'][ok] <= params.undersized_z)
        mono = ~fp['is_tf'] & (fp['coarse'] == MONO) & ~fp['truncated']
        for i in np.flatnonzero(mono):
            a, b = int(fp['start'][i]) + 15, int(fp['end'][i]) - 15
            if b <= a:
                continue
            inf = m.informative[a:b]
            n = int(inf.sum())
            if n == 0:
                continue
            lp.append(float(fp['lambda'][i]))
            kp.append(int((m.hits[a:b] & inf).sum()))
            npr.append(n)
    out = {'undersized_z': float(params.undersized_z), 'undersized_min_bp': int(params.undersized_min_bp)}
    if lu:
        out['undersized'] = _binned_rate(np.concatenate(lu), np.concatenate(hu).astype(float))
    if lp:
        lp_, kp_, np_ = np.asarray(lp), np.asarray(kp, float), np.asarray(npr, float)
        out['protected'] = _binned_rate(lp_, kp_ / np_, weights=np_)
    return out


def undersized_p0(cal, lam) -> np.ndarray:
    tab = (getattr(cal, 'disruption', None) or {}).get('undersized', {})
    return np.clip(_lookup(tab, lam, DEFAULT_UNDERSIZED_RATE), 0.002, 0.45)


def protected_rate(cal, lam) -> np.ndarray:
    tab = (getattr(cal, 'disruption', None) or {}).get('protected', {})
    return np.clip(_lookup(tab, lam, DEFAULT_PROTECTED_RATE), 1e-4, 0.2)


# ---------------------------------------------------------------------------
# undersized runs
# ---------------------------------------------------------------------------

def maximal_segments(scores: Sequence[float]) -> List[Tuple[int, int, float]]:
    """All maximal-scoring segments (Ruzzo and Tompa 1999): ``(i, j, score)``
    with half-open index ranges."""
    segs: List[List[float]] = []          # [start, end, L, R]
    cum = 0.0
    for i, x in enumerate(scores):
        if x <= 0:
            cum += x
            continue
        new = [i, i + 1, cum, cum + x]
        cum += x
        while True:
            j = len(segs) - 1
            while j >= 0 and segs[j][2] >= new[2]:
                j -= 1
            if j < 0 or segs[j][3] >= new[3]:
                segs.append(new)
                break
            new = [segs[j][0], new[1], segs[j][2], new[3]]
            del segs[j:]
    return [(int(a), int(b), float(R - L)) for a, b, L, R in segs]


def undersized_runs(mol, fp, cal, params: NucParams = DEFAULT_PARAMS) -> List[Event]:
    if fp is None or len(fp['start']) == 0:
        return []
    ok = undersized_eligible(fp, cal, params)
    under = ok & (fp['z'] <= params.undersized_z)
    p0 = undersized_p0(cal, fp['lambda'])
    p1 = np.minimum(np.maximum(params.undersized_p1, 2.0 * p0), 0.95)
    score = np.where(under, np.log(p1 / p0), np.log((1 - p1) / (1 - p0)))
    score = np.where(ok, score, 0.0)
    centres = (fp['start'] + fp['end']) / 2.0
    out: List[Event] = []
    for a, b in _pieces(mol.length, mol.msps, params.long_msp_bp):
        idx = np.flatnonzero((centres >= a) & (centres < b))
        if len(idx) == 0:
            continue
        idx = idx[np.argsort(fp['start'][idx], kind='stable')]
        for i, j, sc in maximal_segments(score[idx].tolist()):
            sel = idx[i:j]
            n_u = int(under[sel].sum())
            if sc >= params.undersized_min_llr and n_u >= params.undersized_min_count:
                out.append(Event(int(fp['start'][sel].min()), int(fp['end'][sel].max()), UNDERSIZED,
                                 float(sc), n_u))
    return out


# ---------------------------------------------------------------------------
# missing nucleosomes
# ---------------------------------------------------------------------------

def missing_nucleosomes(mol, fp, segs: List[Segment], dyads: np.ndarray, nrl_mol: float, cal,
                        params: NucParams = DEFAULT_PARAMS) -> List[Event]:
    if fp is None:
        return []
    nuc = np.flatnonzero(~fp['is_tf'])
    if len(nuc) < 2:
        return []
    nuc = nuc[np.argsort(fp['start'][nuc], kind='stable')]
    lo, hi = params.missing_len_bp
    tol = params.missing_grid_tol_bp
    regs = [s for s in segs if s.state == REGULAR]
    out: List[Event] = []
    for a, b in zip(nuc[:-1], nuc[1:]):
        ca, cb = int(fp['coarse'][a]), int(fp['coarse'][b])
        if ca in (CANT_TELL, EDGE_TRUNCATED) or cb in (CANT_TELL, EDGE_TRUNCATED):
            continue
        if fp['truncated'][a] or fp['truncated'][b]:
            continue
        s0, s1 = int(fp['end'][a]), int(fp['start'][b])
        if not (lo <= s1 - s0 <= hi):
            continue
        lam = float(np.nanmean([fp['lambda'][a], fp['lambda'][b]]))
        mu = float(cal.curve.mu_at(lam)) if cal.curve is not None else 147.0
        ha = (fp['end'][a] - fp['start'][a]) / 2.0 if ca == MONO else mu / 2.0
        hb = (fp['end'][b] - fp['start'][b]) / 2.0 if cb == MONO else mu / 2.0
        left, right = s0 - ha, s1 + hb
        enclosing = next((s for s in regs if s.start <= s0 and s.end >= s1), None)
        if enclosing is None:
            continue
        q = float(enclosing.q_hat) if enclosing.q_hat else (float(nrl_mol) if np.isfinite(nrl_mol)
                                                              else float(cal.nrl_pooled))
        if abs((right - left) - 2.0 * q) > tol:
            continue
        # evidence where the grid puts the missing nucleosome: the predicted
        # dyad +/- 60 bp, inside the open stretch
        c = 0.5 * (left + right)
        x0, x1 = int(max(s0 + 10, c - 60)), int(min(s1 - 10, c + 60))
        if x1 <= x0:
            continue
        inf = mol.informative[x0:x1]
        n = int(inf.sum())
        if n == 0:
            continue
        k = int((mol.hits[x0:x1] & inf).sum())
        lam_c = min(max(lam, 1e-3), 0.95)
        pp = float(protected_rate(cal, lam))
        if lam_c <= pp:
            continue
        llr = k * np.log(lam_c / pp) + (n - k) * np.log((1 - lam_c) / (1 - pp))
        if llr >= params.missing_min_llr:
            out.append(Event(s0, s1, MISSING, float(llr), k))
    return out
