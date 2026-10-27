"""Nucleosome analyses on the association layer (NUC 3.8, 3.9; feature hub M2 and M3).

**State conditioning** (:func:`condition_stats`, NUC 3.8). At anchors (any hit set), molecules are open (an MSP of
at least ``barrier_min_msp`` (80) bp covers the anchor +/- ``barrier_slop`` (30) bp) or closed (a nucleosome call
covers the anchor), or a caller-supplied state rule. The response is the phasing of the flanking nucleosomes:

- **frames**: the anchor frame (distances from the anchor) and the molecule frame (distances from the molecule's own
  barrier at the anchor: its MSP, else its footprint);
- **bins** span whole periods (``fiberhmm.nuc.phasing.period_bins``: sliding 2-period bins and the 5-period
  summary), so a uniform dyad distribution has zero expected resultant; the two sides of an unoriented anchor are
  kept apart;
- **metrics** are sums over molecules, refitted under every permutation: the resultant R_B
  (:func:`resultant_metric`), R_B minus its molecule-level phase-null expectation, and the oscillation amplitude
  A/D0 by a linear fit with fixed period and decay (:func:`oscillation_metric`);
- **strata**: lambda_acc measured outside anchor +/- 2 kb (so the state cannot drive it), 12 quantile bins; the
  two states are balanced within locus x stratum x bin (``fiberhmm.assoc.condition``);
- **backgrounds**: matched random positions (same contig, coverage decile, site-density decile, at least 2 x flank
  from every anchor and each other), with the same state rule; open molecules at random positions supply the
  matched random MSP edges of the molecule frame. Delta_site = Delta_anchor - Delta_random is tested by permuting
  which member of each matched set plays the anchor;
- **selection**: anchors discovered from the same molecules (pooled-MSP anchors, :func:`pooled_msp_anchors`) use
  held-out discovery (:func:`held_out`): discover on one hash half of the molecules, test on the other, and
  cross-fit both directions (the combined statistic is the mean of the two; its null pairs the two independent
  permutation nulls draw by draw).

**Reverse direction** (:func:`reverse_association`, NUC 3.8 item 7): Y = the anchor state, X = a flank state of the
same molecule. A CMH common OR over anchor x lambda strata, and a conditional logistic regression with anchors as
strata (Y ~ X + log lambda_acc + coverage pattern); both with locus-group bootstrap intervals.

**Association with other features** (:func:`association_stats`, NUC 3.9): segments of one kind (X: regular arrays)
against point features (Y: TF footprint centres) per molecule, on the **informative coordinate** (informative sites
inside the uncensored interior [c, L - c), low-information segments removed): the conditional binomial rate ratio
with circular shifts (``fiberhmm.assoc.rate``) and the edge association.
"""
from __future__ import annotations

import hashlib
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from fiberhmm.assoc import strata as ST
from fiberhmm.assoc.condition import Appearances, LocusTable, SumMetric, condition

RES_COLUMNS = ('re_down', 'im_down', 're_up', 'im_up', 'abs2_down', 'abs2_up', 'n')


# ------------------------------------------------------------------ metrics
def resultant_metric(null_adjusted: bool = False) -> SumMetric:
    """Pooled phasing resultant R = (|sum V_down| + |sum V_up|) / N over the summed per-molecule vectors; with
    ``null_adjusted``, minus the molecule-level phase-null expectation (sqrt(pi/4 sum |V_down|^2) + sqrt(pi/4 sum
    |V_up|^2)) / N (weighted sums use the squared balance weight for the |V|^2 terms)."""
    def fn(s):
        R = (np.hypot(s[..., 0], s[..., 1]) + np.hypot(s[..., 2], s[..., 3])) / s[..., 6]
        if null_adjusted:
            R = R - (np.sqrt(np.pi / 4 * s[..., 4]) + np.sqrt(np.pi / 4 * s[..., 5])) / s[..., 6]
        return R
    return SumMetric('R_minus_null' if null_adjusted else 'R', RES_COLUMNS, (1, 1, 1, 1, 2, 2, 1), fn)


def oscillation_metric(lo: float, hi: float, P: float, ell: float = 500.0, step: float = 10.0) -> SumMetric:
    """Oscillation amplitude A/D0 of the pooled dyad density over [lo, hi): D(x) = D0 + exp(-(x - lo) / ell) (a
    cos(2 pi x / P) + b sin(2 pi x / P)), fitted by least squares with P and ell fixed (a linear fit, so it is
    refitted exactly under every permutation). Statistics: dyad counts per ``step`` bp bin (folded distances)."""
    edges = np.arange(lo, hi + step / 2, step)
    xc = (edges[:-1] + edges[1:]) / 2
    dec = np.exp(-(xc - lo) / ell)
    D = np.column_stack([np.ones_like(xc), dec * np.cos(2 * np.pi * xc / P), dec * np.sin(2 * np.pi * xc / P)])
    proj = np.linalg.pinv(D)                       # (3, n_bins)

    def fn(s):
        coef = np.einsum('kb,...b->...k', proj, s)
        return np.hypot(coef[..., 1], coef[..., 2]) / coef[..., 0]
    m = SumMetric('A_over_D0', tuple(f'h{i}' for i in range(len(xc))), tuple([1] * len(xc)), fn)
    m.edges = edges
    return m


# ------------------------------------------------------------------ per-molecule pieces
def nuc_state(mol, a_mol: int, params=None) -> str:
    """'1' open (an MSP >= barrier_min_msp covers a +/- barrier_slop), '0' closed (a nucleosome call covers a),
    '.' otherwise (NUC 2.3 rules)."""
    from fiberhmm.nuc.params import DEFAULT_PARAMS
    params = params or DEFAULT_PARAMS
    lo, hi = a_mol - params.barrier_slop, a_mol + params.barrier_slop
    m = mol.msps
    if len(m) and np.any((m[:, 0] <= lo) & (m[:, 1] >= hi) & ((m[:, 1] - m[:, 0]) >= params.barrier_min_msp)):
        return '1'
    n = mol.nucs
    if len(n) and np.any((n[:, 0] <= a_mol) & (n[:, 1] > a_mol)):
        return '0'
    return '.'


def molecule_dyads(mol, nrl_pooled: float, params=None) -> np.ndarray:
    """Eligible mono-sized dyads (``fiberhmm.nuc.spacing.eligible_dyads``, size rule 0.45-1.25 x NRL, no
    edge-truncated calls)."""
    from fiberhmm.nuc.params import DEFAULT_PARAMS
    from fiberhmm.nuc.spacing import eligible_dyads
    return eligible_dyads(mol, nrl_pooled, params or DEFAULT_PARAMS).centres


def molecule_key(mol) -> str:
    """Physical molecule key of a decoded record: PacBio movie/zmw, else the QNAME (scan contracts section 4)."""
    parts = (mol.name or '').split('/')
    if len(parts) >= 2 and parts[1].isdigit():
        return f'pacbio/{parts[0]}/{parts[1]}'
    return f'read/{mol.name}'


class _ChromIndex:
    def __init__(self, molecules):
        self.by = {}
        for i, m in enumerate(molecules):
            self.by.setdefault(m.chrom, []).append(i)
        self.starts = {}
        for c, idx in self.by.items():
            idx = np.array(sorted(idx, key=lambda i: molecules[i].ref_start))
            self.by[c] = idx
            self.starts[c] = np.array([molecules[i].ref_start for i in idx])
            self.ends_max = None
        self.molecules = molecules

    def spanning(self, chrom, lo, hi):
        idx = self.by.get(chrom)
        if idx is None:
            return []
        k = np.searchsorted(self.starts[chrom], lo, 'right')
        return [i for i in idx[:k] if self.molecules[i].ref_end >= hi]


def site_covariates(molecules, positions: Sequence[Tuple[str, int]], span: int, window: int = 1000,
                    max_molecules: int = 10, index: Optional[_ChromIndex] = None) -> Dict[str, np.ndarray]:
    """Per position: coverage (molecules spanning pos +/- span) and site density (informative-site fraction over
    pos +/- window in up to ``max_molecules`` of them; A/T for Hia5, the encoder's targets otherwise)."""
    ix = index or _ChromIndex(molecules)
    cov = np.zeros(len(positions))
    dens = np.full(len(positions), np.nan)
    for k, (c, p) in enumerate(positions):
        sp = ix.spanning(c, p - span, p + span)
        cov[k] = len(sp)
        vals = []
        for i in sp[:max_molecules]:
            m = molecules[i]
            a = m.ref_to_mol([p - window, p + window])
            if a[0] < 0 or a[1] < 0:
                continue
            s, e = sorted((int(a[0]), int(a[1])))
            hits, inf = m.counts(s, e)
            vals.append(inf / max(e - s, 1))
        if vals:
            dens[k] = float(np.mean(vals))
    return {'coverage': cov, 'site_density': dens}


def anchor_appearances(molecules, loci: LocusTable, *, frame: str = 'anchor', P: float = 189.0, params=None,
                       prior=None, state_fn: Optional[Callable] = None, nrl_pooled: Optional[float] = None,
                       flank_exclude: int = 2000, density_bins: int = 12, labels: Optional[Dict[str, Callable]] = None,
                       dyads: Optional[Dict[int, np.ndarray]] = None, oscillation: Optional[SumMetric] = None,
                       index: Optional[_ChromIndex] = None, min_sites: int = 10):
    """M2 appearances of molecules at loci for the phasing metrics (module docstring).

    Each physical molecule (its longest record) reaching the first bin on both sides of a locus, with a state,
    gives one row: resultant statistics per period bin, a coverage mask (bin b is measured when the molecule
    extends past the bin on both sides of the anchor, or of its barrier in the molecule frame, with at least
    ``min_sites`` informative sites on each side; unmeasured bins carry no statistics), dyad counts over the
    summary span with ``oscillation``, lambda_acc outside the anchor +/- ``flank_exclude``, and labels
    (``labels``: name -> f(molecule)). Returns (Appearances, bins, info)."""
    from fiberhmm.nuc.density import FLAT_PRIOR, lambda_acc
    from fiberhmm.nuc.params import DEFAULT_PARAMS
    from fiberhmm.nuc.phasing import anchor_offsets, barrier_at, period_bins
    params = params or DEFAULT_PARAMS
    prior = prior or FLAT_PRIOR
    nrl = nrl_pooled or P
    bins = period_bins(P, params)
    span = params.phase_x0 + params.phase_summary_periods * P
    ix = index or _ChromIndex(molecules)
    key_index: Dict[str, int] = {}
    rows_l, rows_m, rows_s, rows_d, stats, osc, lab = [], [], [], [], [], [], {k: [] for k in (labels or {})}
    info = {'no_state': 0, 'anchor_absent': 0, 'no_barrier': 0, 'appearances': 0}
    dy_cache = dyads if dyads is not None else {}
    order = sorted(range(len(molecules)), key=lambda i: (-molecules[i].length, molecules[i].name or ''))
    rank = {i: r for r, i in enumerate(order)}
    masks = []
    for l in range(loci.n):
        c, p = loci.chrom[l], int(loci.pos[l])
        # records whose alignment covers the anchor; flank eligibility is decided in molecular coordinates
        # (an insertion can make the molecular flank longer than the reference span)
        cand = []
        for i in ix.spanning(c, p, p + 1):
            m = molecules[i]
            a = int(m.ref_to_mol([p])[0])
            if a < 0:
                info['anchor_absent'] += 1
                continue
            st = state_fn(m, a, l) if state_fn is not None else nuc_state(m, a, params)
            if st not in ('1', '0'):
                info['no_state'] += 1
                continue
            cand.append((rank[i], i, a, st))
        # one appearance per physical molecule: records that disagree on the state make it '.', otherwise the
        # longest record is used
        by_key: Dict[str, list] = {}
        for item in sorted(cand):
            by_key.setdefault(molecule_key(molecules[item[1]]), []).append(item)
        for k in sorted(by_key, key=lambda k_: by_key[k_][0][0]):
            recs = by_key[k]
            if len(recs) > 1:
                info['duplicate_records'] = info.get('duplicate_records', 0) + len(recs) - 1
                if len({r_[3] for r_ in recs}) > 1:
                    info['conflicting_records'] = info.get('conflicting_records', 0) + 1
                    continue
            _r, i, a, st = recs[0]
            m = molecules[i]
            if frame == 'molecule':
                bar = barrier_at(m, a, params)
                if bar is None:
                    info['no_barrier'] = info.get('no_barrier', 0) + 1
                    continue
                s0, e0 = bar[0], bar[1]
            else:
                s0, e0 = a, a
            mask = _measured_bins(m, s0, e0, bins, min_sites)
            if not mask:
                info['no_bin_measured'] = info.get('no_bin_measured', 0) + 1
                continue
            d = dy_cache.get(id(m))
            if d is None:
                d = dy_cache[id(m)] = molecule_dyads(m, nrl, params)
            x, why = anchor_offsets(m, d, p, 1, frame, params)
            if x is None:
                info[why] = info.get(why, 0) + 1
                continue
            xr, xl = x[x > 0], -x[x < 0]
            row = np.zeros((len(bins), len(RES_COLUMNS)))
            for b, (lo, hi) in enumerate(bins):
                if not (mask >> b) & 1:
                    continue
                vr = np.exp(2j * np.pi * xr[(xr >= lo) & (xr < hi)] / P)
                vl = np.exp(2j * np.pi * xl[(xl >= lo) & (xl < hi)] / P)
                Vr, Vl = vr.sum(), vl.sum()
                row[b] = (Vr.real, Vr.imag, Vl.real, Vl.imag, abs(Vr) ** 2, abs(Vl) ** 2, len(vr) + len(vl))
            if oscillation is not None:
                h = np.histogram(np.abs(x), bins=oscillation.edges)[0].astype(float)
                osc.append(h if (mask >> (len(bins) - 1)) & 1 else np.zeros_like(h))
            mid = key_index.setdefault(k, len(key_index))
            lam = lambda_acc(m, prior, params, exclude=(a - flank_exclude, a + flank_exclude))
            rows_l.append(l); rows_m.append(mid); rows_s.append(int(st)); rows_d.append(lam); stats.append(row)
            masks.append(mask)
            for name, f in (labels or {}).items():
                lab[name].append(f(m))
            info['appearances'] += 1
    n = len(rows_l)
    S = np.array(stats).reshape(n, len(bins), len(RES_COLUMNS)) if n else np.zeros((0, len(bins), len(RES_COLUMNS)))
    dens = np.array(rows_d, float)
    app = Appearances(np.array(rows_l, np.int64), np.array(rows_m, np.int64), np.array(rows_s, np.int64),
                      ST.density_bins(dens, density_bins) if n else np.zeros(0, np.int64),
                      np.array(masks, np.int64), S,
                      {k: np.array(v, dtype=object) for k, v in lab.items()}, density=dens)
    info['molecules'] = len(key_index)
    if oscillation is not None:
        app.osc = np.array(osc).reshape(n, -1) if n else np.zeros((0, len(oscillation.columns)))
    return app, bins, info


def _measured_bins(mol, s0: int, e0: int, bins, min_sites: int) -> int:
    """Coverage mask over ``bins`` for a molecule whose origin occupies molecular [s0, e0] (the anchor, or the
    barrier's edges): bin (lo, hi) is measured when the molecule extends hi bp past both edges and holds at least
    ``min_sites`` informative sites in each side's [lo, hi) window."""
    mask = 0
    for b, (lo, hi) in enumerate(bins):
        lo_i, hi_i = int(np.floor(lo)), int(np.ceil(hi))
        if s0 - hi_i < 0 or e0 + hi_i > mol.length:
            continue
        _h1, n_right = mol.counts(e0 + lo_i, e0 + hi_i)
        _h2, n_left = mol.counts(s0 - hi_i, s0 - lo_i)
        if n_right >= min_sites and n_left >= min_sites:
            mask |= 1 << b
    return mask


def _with_osc(app: Appearances) -> Appearances:
    summary = (app.mask >> (app.n_bins - 1)) & 1
    return Appearances(app.locus, app.molecule, app.state, app.stratum, summary.astype(np.int64),
                       app.osc[:, None, :], app.labels, app.density, app.channel)


# ------------------------------------------------------------------ backgrounds and anchors
def random_backgrounds(molecules, anchors: Sequence[Tuple[str, int]], *, k: int = 5, flank: int = 1500,
                       span: Optional[int] = None, step: int = 500, seed: int = 0, min_coverage: int = 5,
                       index: Optional[_ChromIndex] = None) -> Tuple[List[Tuple[str, int]], dict]:
    """Matched random background positions for transferable state rules: candidates every ``step`` bp inside the
    molecules' extent with at least ``min_coverage`` spanning molecules, matched on coverage and site density
    deciles, at least 2 x flank from every anchor and each other. Returns (candidates, matched-set record)."""
    from fiberhmm.assoc.background import matched_backgrounds
    ix = index or _ChromIndex(molecules)
    span = span or flank
    cands = []
    rng = np.random.default_rng(seed)
    for c, idx in ix.by.items():
        lo = min(molecules[i].ref_start for i in idx) + span
        hi = max(molecules[i].ref_end for i in idx) - span
        if hi <= lo:
            continue
        pts = np.arange(lo, hi, step) + rng.integers(0, step, len(range(lo, hi, step)))
        cands.extend((c, int(q)) for q in pts if q < hi)
    ccov = site_covariates(molecules, cands, span, index=ix)
    keep = ccov['coverage'] >= min_coverage
    cands = [cands[j] for j in np.flatnonzero(keep)]
    ccov = {k_: v[keep] for k_, v in ccov.items()}
    acov = site_covariates(molecules, anchors, span, index=ix)
    for d in (acov, ccov):
        d['site_density'] = np.where(np.isfinite(d['site_density']), d['site_density'],
                                     np.nanmedian(np.r_[acov['site_density'], ccov['site_density']]))
    mb = matched_backgrounds(anchors, cands, anchor_covariates=acov, candidate_covariates=ccov, k=k,
                             min_sep=2 * flank, seed=seed)
    return cands, mb


def pooled_msp_anchors(molecules, *, bin_bp: int = 10, min_frac: float = 0.25, min_cov: int = 10,
                       min_msp: int = 100, sep: int = 3000) -> List[Tuple[str, int]]:
    """Pooled-MSP anchors (NUC 2.3): bins where at least ``min_frac`` of covering molecules have an MSP of at least
    ``min_msp`` bp, local maxima at least ``sep`` apart, coverage at least ``min_cov``."""
    out = []
    by = {}
    for m in molecules:
        by.setdefault(m.chrom, []).append(m)
    for c, ms in sorted(by.items()):
        lo = min(m.ref_start for m in ms)
        hi = max(m.ref_end for m in ms)
        nb = (hi - lo) // bin_bp + 1
        cov = np.zeros(nb)
        acc = np.zeros(nb)
        for m in ms:
            cov[(m.ref_start - lo) // bin_bp:(m.ref_end - lo) // bin_bp] += 1
            for s, e in m.msps[(m.msps[:, 1] - m.msps[:, 0]) >= min_msp] if len(m.msps) else []:
                iv = m.mol_interval_to_ref(int(s), int(e))
                if iv is None:
                    continue
                acc[(iv[0] - lo) // bin_bp:(iv[1] - lo) // bin_bp + 1] += 1
        frac = np.where(cov >= min_cov, acc / np.maximum(cov, 1), 0)
        taken = np.zeros(nb, bool)
        w = sep // bin_bp
        for i in np.argsort(-frac, kind='stable'):
            if frac[i] < min_frac:
                break
            if taken[max(0, i - w):i + w].any():
                continue
            taken[i] = True
            out.append((c, int(lo + i * bin_bp + bin_bp // 2)))
    return sorted(out)


def hash_half(mol, salt: str = 'fiberhmm.assoc.held_out') -> int:
    """0 or 1 by a stable hash of the physical molecule key."""
    h = hashlib.sha256(f'{salt}\n{molecule_key(mol)}'.encode()).digest()
    return h[0] & 1


# ------------------------------------------------------------------ the analyses
def condition_stats(molecules, anchors: Sequence[Tuple[str, int]], *, cal=None, params=None, P: Optional[float] = None,
                    frames: Sequence[str] = ('anchor', 'molecule'), backgrounds='random', k: int = 5,
                    state_fn: Optional[Callable] = None, n_perm: int = 1000, n_role: int = 1000, n_boot: int = 1000,
                    group: Optional[str] = None, group_kind: str = 'label',
                    labels: Optional[Dict[str, Callable]] = None, metrics: Sequence[str] = ('R_minus_null', 'R'),
                    oscillation: bool = False, per_anchor: bool = True, seed: int = 0,
                    return_null: bool = False, prefetched_backgrounds=None) -> dict:
    """NUC 3.8 state conditioning at ``anchors`` ((chrom, pos) list) over decoded molecules (module docstring).

    ``backgrounds``: 'random' (matched random positions), None (anchors only), or a list of (chrom, pos)
    candidates. Returns per frame and metric the ``fiberhmm.assoc.condition`` result, plus the background record."""
    from fiberhmm.nuc.params import DEFAULT_PARAMS
    params = params or DEFAULT_PARAMS
    P = float(P or (cal.nrl_pooled if cal is not None else 189.0))
    prior = cal.prior if cal is not None else None
    ix = _ChromIndex(molecules)
    span = int(params.phase_x0 + params.phase_summary_periods * P)
    bg_info = None
    if prefetched_backgrounds is not None:
        cands, mb = prefetched_backgrounds
    elif backgrounds == 'random':
        cands, mb = random_backgrounds(molecules, anchors, k=k, flank=span, span=span, seed=seed, index=ix)
    elif backgrounds is None:
        cands, mb = [], {'sets': [[i, []] for i in range(len(anchors))], 'shortfall': len(anchors)}
    else:
        cands = list(backgrounds)
        cov_a = site_covariates(molecules, anchors, span, index=ix)
        cov_c = site_covariates(molecules, cands, span, index=ix)
        from fiberhmm.assoc.background import matched_backgrounds
        mb = matched_backgrounds(anchors, cands, anchor_covariates={'coverage': cov_a['coverage']},
                                 candidate_covariates={'coverage': cov_c['coverage']}, k=k, min_sep=2 * span,
                                 seed=seed)
    bg_info = {kk: v for kk, v in mb.items() if kk != 'sets'}
    loci = LocusTable.from_sets(list(anchors), cands, mb['sets'])
    out = {'P': P, 'span_bp': span, 'n_anchors': len(anchors), 'backgrounds': bg_info, 'frames': {}}
    dyads: Dict[int, np.ndarray] = {}
    for frame in frames:
        osc = oscillation_metric(params.phase_x0, params.phase_x0 + params.phase_summary_periods * P, P) \
            if oscillation else None
        app, bins, info = anchor_appearances(molecules, loci, frame=frame, P=P, params=params, prior=prior,
                                             state_fn=state_fn, nrl_pooled=P, labels=labels, dyads=dyads,
                                             oscillation=osc, index=ix)
        fr = {'appearances': info, 'bins': [list(b) for b in bins], 'metrics': {}}
        for name in metrics:
            met = resultant_metric(null_adjusted=(name == 'R_minus_null'))
            fr['metrics'][name] = condition(app, loci, met, bins=[f'{lo:.0f}-{hi:.0f}' for lo, hi in bins],
                                            n_perm=n_perm, n_role=n_role, n_boot=n_boot, group=group,
                                            group_kind=group_kind, per_anchor=per_anchor, seed=seed,
                                            return_null=return_null)
        if osc is not None and len(app.locus):
            fr['metrics']['A_over_D0'] = condition(_with_osc(app), loci, osc, bins=['summary'], n_perm=n_perm,
                                                   n_role=n_role, n_boot=n_boot, group=group, group_kind=group_kind,
                                                   per_anchor=False, seed=seed, return_null=return_null)
        out['frames'][frame] = fr
    return out


def held_out(molecules, discover: Callable, analyse: Callable, *, salt: str = 'fiberhmm.assoc.held_out',
             seed: int = 0) -> dict:
    """Held-out discovery with cross-fitting. ``discover(molecules) -> anchors``; ``analyse(molecules, anchors,
    seed) -> condition_stats result`` (run with ``return_null=True`` and the given seed, so the two directions draw
    independent permutations). Molecules are split by a stable hash of the physical molecule; each half's anchors
    are tested on the other half. ``cross_fit`` holds, per frame, metric and bin, the mean of the two directions'
    statistics and p-values from the pairwise-averaged (independent) permutation nulls."""
    from fiberhmm.assoc.resample import perm_p_two_from_tails
    halves = [[m for m in molecules if hash_half(m, salt) == h] for h in (0, 1)]
    res = []
    for d, t in ((0, 1), (1, 0)):
        anchors = discover(halves[d])
        r = analyse(halves[t], anchors, seed + 1_000_003 * (d + 1))
        r['discovered_on'] = f'half {d}'
        r['tested_on'] = f'half {t}'
        res.append(r)
    cross = {}
    for frame in res[0]['frames']:
        for name in res[0]['frames'][frame]['metrics']:
            rows = []
            b0 = res[0]['frames'][frame]['metrics'][name]['bins']
            b1 = res[1]['frames'][frame]['metrics'][name]['bins']
            for x, y in zip(b0, b1):
                row = {'bin': x['bin']}
                for stat, nk, pk in (('delta_anchor', 'null_within', 'p_within_anchor'),
                                     ('delta_site', 'null_site', 'p_site'),
                                     ('site_state1', 'null_site_state1', 'p_site_state1'),
                                     ('site_state0', 'null_site_state0', 'p_site_state0')):
                    if stat in x and stat in y:
                        row[stat] = (x[stat] + y[stat]) / 2
                        if nk in x and nk in y:
                            n = min(len(x[nk]), len(y[nk]))
                            row[pk] = perm_p_two_from_tails(row[stat], (x[nk][:n] + y[nk][:n]) / 2)
                for ci in ('delta_site_ci', 'delta_anchor_ci', 'delta_random_ci', 'site_state1_ci', 'site_state0_ci'):
                    if ci in x and ci in y:
                        row[ci.replace('_ci', '_se_mean')] = 0.5 * float(np.hypot(x[ci]['se'], y[ci]['se']))
                if 'delta_random' in x and 'delta_random' in y:
                    row['delta_random'] = (x['delta_random'] + y['delta_random']) / 2
                rows.append(row)
            cross[f'{frame}/{name}'] = rows
    for r in res:
        for fr in r['frames'].values():
            for met in fr['metrics'].values():
                for b in met['bins']:
                    for key in ('null_within', 'null_site', 'null_site_state1', 'null_site_state0'):
                        b.pop(key, None)
    return {'directions': res, 'cross_fit': cross,
            'note': 'anchors discovered on one hash half of the molecules and tested on the other; cross-fit = mean '
                    'of both directions'}


def reverse_association(locus, molecule, y, x, log_lambda, pattern=None, *, n_boot: int = 1000,
                        lambda_bins: int = 6, seed: int = 0) -> dict:
    """NUC 3.8 item 7: Y (anchor state) against X (a flank state), per (locus, molecule) row. CMH common OR over
    locus x lambda strata and a conditional logistic regression with loci as strata (Y ~ X + log lambda + coverage
    pattern dummies); both with locus-group bootstrap intervals (two-way variance when groups chain)."""
    from fiberhmm.assoc.clogit import clogit, score_bootstrap
    from fiberhmm.assoc.cooccur import _mh_counts
    from fiberhmm.assoc.resample import BlockBootstrap, locus_groups, wald_p
    locus = np.asarray(locus, np.int64)
    molecule = np.asarray(molecule, np.int64)
    y = np.asarray(y, np.int64)
    x = np.asarray(x, np.int64)
    ll = np.asarray(log_lambda, float)
    lb = ST.quantile_bins(ll, lambda_bins)
    cell = ST.combine(locus, lb)
    nc = int(cell.max()) + 1 if len(cell) else 0
    cat = (1 - x) * 2 + (1 - y)
    cnt = np.bincount(cell * 4 + cat, minlength=4 * nc).reshape(nc, 4).T
    lor, lo, hi = _mh_counts(*cnt)
    cols = [x.astype(float), np.where(np.isfinite(ll), ll, np.nanmedian(ll))]
    if pattern is not None:
        pu, pi = np.unique(np.asarray(pattern), return_inverse=True)
        cols += [(pi == k_).astype(float) for k_ in range(1, len(pu))]
    X = np.column_stack(cols)
    fit = clogit(X, y, locus)
    out = {'n_rows': int(len(y)), 'n_loci': int(len(np.unique(locus))),
           'cmh': {'log_or': lor, 'or': float(np.exp(lor)) if np.isfinite(lor) else float('nan'),
                   'rbg_lo': float(np.exp(lo)) if np.isfinite(lo) else float('nan'),
                   'rbg_hi': float(np.exp(hi)) if np.isfinite(hi) else float('nan')},
           'clogit': {'log_or': float(fit['beta'][0]), 'or': float(np.exp(fit['beta'][0])),
                      'se_model': float(fit['se'][0]), 'n_strata': int(fit['n_strata'])}}
    if n_boot and len(y):
        g = locus_groups(locus, molecule, int(locus.max()) + 1)
        bb = BlockBootstrap(g, locus, molecule, seed=seed)
        cat_m = cell * 4 + cat

        def fn(W):
            res = {'clogit': score_bootstrap(fit, [W], split_strata=bb.mode != 'groups')}
            c4 = np.stack([np.bincount(cat_m, weights=w, minlength=4 * nc).reshape(nc, 4).T for w in W])
            a, b, c, d = c4[:, 0], c4[:, 1], c4[:, 2], c4[:, 3]
            nn = a + b + c + d
            with np.errstate(divide='ignore', invalid='ignore'):
                R = np.where(nn > 0, a * d / nn, 0).sum(1)
                S = np.where(nn > 0, b * c / nn, 0).sum(1)
                res['cmh'] = np.where((R > 0) & (S > 0), np.log(R / S), np.nan)
            return res
        boot = bb.collect(fn, n_boot)
        for name in ('cmh', 'clogit'):
            s = boot.summary(name, out[name]['log_or'])
            from scipy.stats import t as tdist
            z = out[name]['log_or'] / s['se'] if (np.isfinite(s['se']) and s['se'] > 0) else float('nan')
            df = s.get('df', float('inf'))
            p_ = (float(2 * tdist.sf(abs(z), df)) if np.isfinite(df) else wald_p(out[name]['log_or'], s['se'])) \
                if np.isfinite(z) else float('nan')
            out[name].update(se_boot=s['se'], lo=float(np.exp(s['lo'])), hi=float(np.exp(s['hi'])), p=p_,
                             df=df, interval=s.get('interval'))
        out['bootstrap'] = boot.info
    return out


def association_stats(molecules, *, cal, params=None, x_state: str = 'regular', y: str = 'tf',
                      censor_bp: int = 500, n_shift: int = 1000, n_boot: int = 1000, group: Optional[str] = None,
                      group_kind: str = 'label', labels: Optional[Dict[str, Callable]] = None, min_tf_bp: int = 0,
                      results: Optional[Dict[int, object]] = None, seed: int = 0, edges: bool = True) -> dict:
    """NUC 3.9: X segments (``x_state``: 'regular', 'disordered', 'undetermined') against Y points (``y='tf'``: TF
    call centres; or a callable mol -> molecular positions) on each molecule's informative coordinate. Molecules
    below the low-information floor are left out. Returns the ``fiberhmm.assoc.rate`` result (and the edge test)."""
    from fiberhmm.assoc.rate import RateData, compress, edge_association, rate_ratio
    from fiberhmm.nuc.molecule import molecule_outputs
    from fiberhmm.nuc.params import DEFAULT_PARAMS
    from fiberhmm.nuc.segment import DISORDERED, LOWINFO, REGULAR
    params = params or DEFAULT_PARAMS
    try:
        from fiberhmm.nuc.segment import UNDETERMINED
    except ImportError:            # pragma: no cover
        UNDETERMINED = 4
    code = {'regular': REGULAR, 'disordered': DISORDERED, 'undetermined': UNDETERMINED}[x_state]
    L, IV, PT, JN, CH, MID, MOL = [], [], [], [], [], [], []
    lab = {k: [] for k in (labels or {})}
    skipped = {'low_information': 0, 'too_short': 0, 'no_informative': 0, 'duplicate_records': 0}
    best: Dict[str, object] = {}
    for m in molecules:                       # one record per physical molecule: the longest (ties by name)
        k = molecule_key(m)
        if k in best:
            skipped['duplicate_records'] += 1
            if (m.length, m.name or '') <= (best[k].length, best[k].name or ''):
                continue
        best[k] = m
    for m in sorted(best.values(), key=lambda m_: (m_.chrom or '', m_.ref_start, m_.name or '')):
        res = results.get(id(m)) if results is not None else None
        if res is None:
            res = molecule_outputs(m, params, cal)
            if results is not None:
                results[id(m)] = res
        if res.flags.get('low_information'):
            skipped['low_information'] += 1
            continue
        if m.length <= 2 * censor_bp:
            skipped['too_short'] += 1
            continue
        cum = np.concatenate([[0], np.cumsum(m.informative, dtype=np.int64)])
        lo_i, hi_i = int(cum[censor_bp]), int(cum[m.length - censor_bp])
        adm = [(lo_i, hi_i)]
        low = [(int(cum[s.start]), int(cum[min(s.end, m.length)])) for s in res.segments if s.state == LOWINFO]
        from fiberhmm.assoc.scanio import _subtract
        adm = _subtract(adm, low)
        if not adm:
            skipped['no_informative'] += 1
            continue
        a_iv = [(int(cum[s.start]), int(cum[min(s.end, m.length)])) for s in res.segments if s.state == code]
        if y == 'tf':
            tf = m.tfs[(m.tfs[:, 1] - m.tfs[:, 0]) >= min_tf_bp] if len(m.tfs) else m.tfs
            pts_mol = ((tf[:, 0] + tf[:, 1]) // 2) if len(tf) else np.zeros(0, np.int64)
        else:
            pts_mol = np.asarray(y(m), np.int64)
        # only points inside the uncensored interior and outside low-information segments (molecular bp), then
        # mapped to the informative coordinate (a non-informative base maps to the next informative site)
        keep = (pts_mol >= censor_bp) & (pts_mol < m.length - censor_bp)
        for sg in res.segments:
            if sg.state == LOWINFO:
                keep &= ~((pts_mol >= sg.start) & (pts_mol < sg.end))
        pts = cum[pts_mol[keep]]
        n, iv, pp, junc = compress(adm, a_iv, pts)
        if n <= 0:
            skipped['no_informative'] += 1
            continue
        L.append(n); IV.append(iv); PT.append(pp); JN.append(junc)
        CH.append(m.chrom); MID.append((m.ref_start + m.ref_end) // 2); MOL.append(molecule_key(m))
        for name, f in (labels or {}).items():
            lab[name].append(f(m))
    d = RateData(np.array(L, np.int64), IV, PT, np.array(CH, dtype=object), np.array(MID, np.int64),
                 np.array(MOL, dtype=object), {k: np.array(v, dtype=object) for k, v in lab.items()}, JN)
    out = rate_ratio(d, n_shift=n_shift, n_boot=n_boot, group=group, group_kind=group_kind, seed=seed)
    out['coordinate'] = 'informative sites inside [censor_bp, L - censor_bp), low-information segments removed'
    out['x_state'] = x_state
    out['y'] = y if isinstance(y, str) else 'custom'
    out['skipped'] = skipped
    if edges:
        out['edges'] = edge_association(d, n_shift=n_shift, seed=seed + 7)
    return out
