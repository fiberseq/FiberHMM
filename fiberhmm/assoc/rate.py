"""M3: within-molecule rate of point features B inside versus outside interval features A (HUB 7.5; NUC 3.9).

Each molecule is reduced to a **compressed coordinate**: its admissible positions in order (informative and not
low-information; for nucleosome analyses, informative sites in the uncensored interior). A's intervals and B's
points are given in that coordinate (:func:`compress`).

- **Exposures and counts.** E_in = admissible length inside A, E_out = outside; O_in, O_out = B points inside and
  outside.
- **Model.** log E[O_k] = log E_k + alpha_m + beta 1[k = in]. Conditioning on O = O_in + O_out removes alpha_m:
  O_in | O ~ Binomial(O, pi_m), logit pi_m = log(E_in / E_out) + beta. beta (the log rate ratio) is fitted by
  conditional maximum likelihood. Its score equation is sum_m O_in = sum_m O expit(log(E_in/E_out) + beta), so
  beta-hat is a monotone function of S = sum_m O_in.
- **Null (circular shift).** On each molecule's compressed circle A is rotated by a uniform offset (exposure is
  preserved exactly; A never lands on inadmissible positions). An interval crossing the junction is cut there;
  offsets that would cut more than 10% of the crossing interval's length are redrawn, so the junction adds no
  artificial edges. 1,000 joint shifts (one offset per molecule each); the test compares S with its shift
  distribution (equivalently beta-hat).
- **Interval**: a block bootstrap over molecules grouped in genomic blocks (``block_bp``, by molecule midpoint;
  molecules are never cut); pigeonhole fallback when one block holds more than 5% of the molecules.
- **Groups**: beta per group; the difference beta_g - beta_ref (the beta x g interaction) with a joint block
  bootstrap (Wald). Unphased molecules are left out of haplotype contrasts (D14).
- **Edge association** (:func:`edge_association`): the mean distance from each A boundary to the nearest B on the
  same molecule, against circular shifts of B. Boundaries at read ends or mask junctions are not boundaries.

Molecules with E_in = 0, E_out = 0 or no B are excluded and counted.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from fiberhmm.assoc.groups import group_contrast, group_labels
from fiberhmm.assoc.resample import BlockBootstrap, perm_p_two_from_tails, perm_p

CUT_MAX = 0.10


@dataclass
class RateData:
    """Per molecule: compressed ``length``, A ``intervals`` ((n, 2) arrays in the compressed coordinate), B
    ``points`` (arrays), the genomic ``chrom``/``midpoint`` (for blocks), and optional ``labels`` (group) and mask
    ``junctions`` (compressed positions where measured stretches join; boundaries there are not edges)."""

    length: np.ndarray
    intervals: List[np.ndarray]
    points: List[np.ndarray]
    chrom: Optional[np.ndarray] = None
    midpoint: Optional[np.ndarray] = None
    molecule_ids: Optional[np.ndarray] = None
    labels: Dict[str, np.ndarray] = field(default_factory=dict)
    junctions: Optional[List[np.ndarray]] = None

    def __post_init__(self):
        self.length = np.asarray(self.length, np.int64)
        n = len(self.length)
        self.intervals = [np.asarray(iv, np.int64).reshape(-1, 2) for iv in self.intervals]
        self.points = [np.asarray(p, np.int64).ravel() for p in self.points]
        if len(self.intervals) != n or len(self.points) != n:
            raise ValueError('one interval array and one point array per molecule')
        if self.chrom is None:
            self.chrom = np.array(['.'] * n, dtype=object)
        if self.midpoint is None:
            self.midpoint = np.zeros(n, np.int64)
        self.labels = {k: np.asarray(v, dtype=object) for k, v in (self.labels or {}).items()}

    @property
    def n(self) -> int:
        return len(self.length)

    def subset(self, m) -> 'RateData':
        idx = np.flatnonzero(m) if np.asarray(m).dtype == bool else np.asarray(m)
        return RateData(self.length[idx], [self.intervals[i] for i in idx], [self.points[i] for i in idx],
                        self.chrom[idx], self.midpoint[idx],
                        None if self.molecule_ids is None else self.molecule_ids[idx],
                        {k: v[idx] for k, v in self.labels.items()},
                        None if self.junctions is None else [self.junctions[i] for i in idx])


def compress(admissible: Sequence[tuple], intervals: Sequence[tuple], points: Sequence[float]):
    """Map one molecule to its compressed coordinate. ``admissible``: disjoint (start, end) stretches in any
    coordinate (bp or informative-site index); ``intervals`` (A) are clipped to them; ``points`` (B) outside them
    are dropped. Returns (length, A intervals, B points, junctions)."""
    from fiberhmm.nuc.enrichment import MeasuredTrack
    tr = MeasuredTrack(admissible)
    L = tr.length
    iv = []
    for a, b in intervals:
        i, j = tr.index(int(a)), tr.index(int(b))
        if j > i:
            iv.append((i, j))
    pts = []
    starts, ends = np.asarray(tr.starts), np.asarray(tr.ends)
    for p in points:
        p = int(p)
        k = np.searchsorted(ends, p, 'right')
        if k < len(starts) and starts[k] <= p < ends[k]:
            pts.append(tr.index(p))
    junctions = np.asarray(tr.offsets[1:-1], np.int64)
    return int(L), np.array(iv, np.int64).reshape(-1, 2), np.array(pts, np.int64), junctions


def _merge(iv: np.ndarray) -> np.ndarray:
    if not len(iv):
        return iv
    iv = iv[np.argsort(iv[:, 0], kind='stable')]
    out = [list(iv[0])]
    for a, b in iv[1:]:
        if a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return np.array(out, np.int64)


def _indicator(L, iv):
    ind = np.zeros(L, bool)
    for a, b in iv:
        ind[max(0, a):min(L, b)] = True
    return ind


def _allowed_junction_points(L, iv, cut_max=CUT_MAX):
    """Positions u (in A's frame) that may sit at the junction: not strictly inside an A interval farther than
    ``cut_max`` of its length from both of its ends."""
    ok = np.ones(L, bool)
    for a, b in iv:
        w = b - a
        lim = cut_max * w
        u = np.arange(a + 1, b)
        bad = u[np.minimum(u - a, b - u) > lim]
        ok[bad] = False
    return np.flatnonzero(ok)


def _solve_beta(O, off, S, w=None, iters=60):
    """beta solving sum w O expit(off + beta) = S (vectorised over the leading dimension of S / w)."""
    O = np.asarray(O, float)
    off = np.asarray(off, float)
    S = np.atleast_1d(np.asarray(S, float))
    W = np.ones((len(S), len(O))) if w is None else np.atleast_2d(w)
    tot = (W * O).sum(1)
    beta = np.zeros(len(S))
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        lo = S <= 0
        hi = S >= tot
        for _ in range(iters):
            p = 1 / (1 + np.exp(-(off[None, :] + beta[:, None])))
            g = (W * O * p).sum(1) - S
            h = (W * O * p * (1 - p)).sum(1)
            step = np.where(h > 0, g / np.maximum(h, 1e-300), 0)
            beta = np.clip(beta - np.clip(step, -5, 5), -30, 30)
            if np.all(np.abs(step) < 1e-10):
                break
    beta[lo] = -np.inf
    beta[hi] = np.inf
    return beta


def _prepare(d: RateData):
    """Per-molecule arrays for the used molecules, and exclusion counts."""
    used, O, Oin, Ein, Eout, ex = [], [], [], [], [], {'no_b': 0, 'no_inside': 0, 'no_outside': 0}
    merged = []
    for m in range(d.n):
        L = int(d.length[m])
        iv = _merge(np.clip(d.intervals[m], 0, L))
        iv = iv[iv[:, 1] > iv[:, 0]] if len(iv) else iv
        pts = d.points[m]
        pts = pts[(pts >= 0) & (pts < L)]
        e_in = int(np.sum(iv[:, 1] - iv[:, 0])) if len(iv) else 0
        if not len(pts):
            ex['no_b'] += 1
            continue
        if e_in == 0:
            ex['no_inside'] += 1
            continue
        if e_in >= L:
            ex['no_outside'] += 1
            continue
        ind = _indicator(L, iv)
        used.append(m)
        merged.append((L, iv, pts, ind))
        O.append(len(pts))
        Oin.append(int(ind[pts].sum()))
        Ein.append(e_in)
        Eout.append(L - e_in)
    return np.array(used, np.int64), merged, np.array(O, float), np.array(Oin, float), \
        np.array(Ein, float), np.array(Eout, float), ex


def _shift_null(merged, n_shift, rng, cut_max):
    """Null S per shift (accumulated over molecules) and per-molecule null O_in means."""
    S = np.zeros(n_shift)
    for L, iv, pts, ind in merged:
        allowed = _allowed_junction_points(L, iv, cut_max)
        u = allowed[rng.integers(0, len(allowed), n_shift)]
        delta = (L - u) % L
        S += ind[(pts[None, :] - delta[:, None]) % L].sum(1)
    return S


def rate_ratio(d: RateData, *, n_shift: int = 1000, n_boot: int = 1000, block_bp: int = 50_000,
               cut_max: float = CUT_MAX, group: Optional[str] = None, group_kind: str = 'label',
               seed: int = 0) -> dict:
    """M3 conditional-binomial rate ratio of B inside versus outside A (module docstring)."""
    rng = np.random.default_rng(seed)
    out = {'mode': 'M3', 'settings': dict(n_shift=n_shift, n_boot=n_boot, block_bp=block_bp, cut_max=cut_max,
                                          group=group, group_kind=group_kind, seed=seed)}
    glab = None
    scope = np.ones(d.n, bool)
    if group is not None:
        glab, ginfo = group_labels(d.labels[group], group_kind)
        out['group_info'] = ginfo
        if group_kind == 'haplotype':
            scope = np.array([g is not None for g in glab])
    if glab is not None and group_kind == 'haplotype':
        un = d.subset(~scope)
        u_used, _m, uO, uOin, uEin, uEout, uex = _prepare(un)
        desc = {'n_molecules': int(len(u_used)), 'excluded': uex,
                'note': 'descriptive; excluded from haplotype contrasts (D14)'}
        if len(u_used):
            ub = float(_solve_beta(uO, np.log(uEin / uEout), [float(uOin.sum())])[0])
            desc.update(log_rate_ratio=ub, rate_ratio=float(np.exp(ub)))
        out['unphased'] = desc
    dd = d.subset(scope)
    gl = glab[scope] if glab is not None else None
    used, merged, O, Oin, Ein, Eout, ex = _prepare(dd)
    out['excluded'] = ex
    out['n_molecules'] = int(len(used))
    if not len(used):
        out['status'] = 'no informative molecules'
        return out
    off = np.log(Ein / Eout)
    S = float(Oin.sum())
    beta = float(_solve_beta(O, off, [S])[0])
    out.update({'O_in': S, 'O_out': float(O.sum() - S), 'E_in': float(Ein.sum()), 'E_out': float(Eout.sum()),
                'log_rate_ratio': beta, 'rate_ratio': float(np.exp(beta)),
                'crude_rate_ratio': float((S / Ein.sum()) / ((O.sum() - S) / Eout.sum())) if O.sum() > S else
                float('inf')})
    if n_shift:
        Snull = _shift_null(merged, n_shift, rng, cut_max)
        out['p_shift'] = perm_p_two_from_tails(S, Snull)
        out['p_depleted'] = perm_p(S, Snull, 'less')
        out['p_enriched'] = perm_p(S, Snull, 'greater')
        out['null_O_in_mean'] = float(Snull.mean())
        out['null_log_rate_ratio_mean'] = float(np.nanmean(_solve_beta(O, off, Snull)[np.isfinite(Snull)]))
        out['test'] = f'circular shifts of A on the compressed coordinate ({n_shift} joint shifts)'
    groups = ginfo['groups'] if glab is not None else []
    gu = gl[used] if gl is not None else None
    if groups:
        out['groups'] = {}
        for g in groups:
            m = gu == g
            if not m.any():
                out['groups'][g] = {'n_molecules': 0}
                continue
            Sg = float(Oin[m].sum())
            bg = float(_solve_beta(O[m], off[m], [Sg])[0])
            row = {'n_molecules': int(m.sum()), 'log_rate_ratio': bg, 'rate_ratio': float(np.exp(bg))}
            if n_shift:
                sub = [merged[i] for i in np.flatnonzero(m)]
                Sn = _shift_null(sub, n_shift, rng, cut_max)
                row['p_shift'] = perm_p_two_from_tails(Sg, Sn)
            out['groups'][g] = row
    if n_boot:
        chrom = dd.chrom[used]
        blk = np.array([f'{c}:{int(x) // block_bp}' for c, x in zip(chrom, dd.midpoint[used])], dtype=object)
        if dd.molecule_ids is not None:
            # records of one physical molecule (a primary and its supplementaries) share the first one's block
            first: Dict[str, str] = {}
            for k, m in enumerate(dd.molecule_ids[used]):
                first.setdefault(m, blk[k])
            blk = np.array([first[m] for m in dd.molecule_ids[used]], dtype=object)
        bidx = np.unique(blk, return_inverse=True)[1]
        mol_items = (np.unique(dd.molecule_ids[used], return_inverse=True)[1] if dd.molecule_ids is not None
                     else np.arange(len(used)))
        bb = BlockBootstrap(np.arange(int(bidx.max()) + 1), bidx, mol_items, seed=seed + 1)
        out['bootstrap'] = bb.describe()

        def fn(W):
            res = {'all': _solve_beta(O, off, W @ Oin, W)}
            for g in groups:
                m = gu == g
                res[g] = (_solve_beta(O[m], off[m], W[:, m] @ Oin[m], W[:, m]) if m.any()
                          else np.full(W.shape[0], np.nan))
            return res
        boot = bb.collect(fn, n_boot, chunk=max(1, min(100, int(2e7 // len(used)))))
        b_all = boot.summary('all', beta)
        out['log_rate_ratio_ci'] = b_all
        out['rate_ratio_lo'] = float(np.exp(b_all['lo']))
        out['rate_ratio_hi'] = float(np.exp(b_all['hi']))
        for g in groups:
            if out['groups'][g].get('n_molecules'):
                bs = boot.summary(g, out['groups'][g]['log_rate_ratio'])
                out['groups'][g]['rate_ratio_lo'] = float(np.exp(bs['lo']))
                out['groups'][g]['rate_ratio_hi'] = float(np.exp(bs['hi']))
        ok = [g for g in groups if out['groups'][g].get('n_molecules')]
        if len(ok) >= 2:
            out['interaction'] = group_contrast({g: out['groups'][g]['log_rate_ratio'] for g in ok}, boot)
            out['interaction']['scale'] = 'log rate ratio (beta x group)'
            out['interaction']['test'] = 'difference of group betas, joint block bootstrap (Wald)'
    return out


def edge_association(d: RateData, *, n_shift: int = 1000, seed: int = 0, boundaries: Optional[List] = None) -> dict:
    """Mean distance (compressed coordinate) from each A boundary to the nearest B on the same molecule, against
    circular shifts of B (one offset per molecule per shift). ``boundaries`` (per molecule) overrides A's interval
    ends (for example the ends of runs of regular dyad gaps). Boundaries at 0, at the molecule's end or at a mask
    junction are excluded; molecules without B or without boundaries are excluded."""
    rng = np.random.default_rng(seed)
    obs_sum = 0.0
    n_bd = 0
    null_sum = np.zeros(n_shift)
    used = 0
    for m in range(d.n):
        L = int(d.length[m])
        pts = np.sort(d.points[m][(d.points[m] >= 0) & (d.points[m] < L)])
        if boundaries is not None:
            bd = np.asarray(boundaries[m], np.int64)
        else:
            iv = _merge(np.clip(d.intervals[m], 0, L))
            bd = iv.ravel() if len(iv) else np.zeros(0, np.int64)
        junc = set(d.junctions[m].tolist()) if d.junctions is not None else set()
        bd = np.array([b for b in np.unique(bd) if 0 < b < L and int(b) not in junc], np.int64)
        if not len(pts) or not len(bd):
            continue
        used += 1

        def nearest(b, p):
            i = np.searchsorted(p, b)
            dl = np.where(i > 0, b - p[np.maximum(i - 1, 0)], np.inf)
            dr = np.where(i < len(p), p[np.minimum(i, len(p) - 1)] - b, np.inf)
            return np.minimum(dl, dr)
        obs_sum += float(nearest(bd, pts).sum())
        n_bd += len(bd)
        delta = rng.integers(0, L, n_shift)
        shifted = np.sort((pts[None, :] + delta[:, None]) % L, axis=1)
        # one flat sorted array: row r is offset by r * 4L, so neighbours never come from another row
        off = (np.arange(n_shift) * 4 * L)[:, None]
        flat = (shifted + off).ravel()
        q = (bd[None, :] + off).ravel()
        dist = nearest(q, flat).reshape(n_shift, len(bd))
        null_sum += dist.sum(1)
    if not n_bd:
        return {'n_molecules': 0, 'n_boundaries': 0}
    obs = obs_sum / n_bd
    null = null_sum / n_bd
    return {'n_molecules': used, 'n_boundaries': int(n_bd), 'mean_distance': float(obs),
            'null_mean_distance': float(null.mean()), 'p_closer': perm_p(obs, null, 'less'),
            'p_two_sided': perm_p_two_from_tails(obs, null),
            'test': f'circular shifts of B on the compressed coordinate ({n_shift})'}
