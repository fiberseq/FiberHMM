"""M1: per-molecule co-occurrence of two feature types' states (HUB 7.3).

For hits a in A and b in B with ``|anchor_a - anchor_b| <= max_distance`` (5 kb) on one contig:

1. **Molecules** with a state (1 or 0) for both hits. A '.' on either side leaves the molecule out of that pair;
   the exclusions are counted. Overlapping pairs of the same kind are not tested; nested pairs of different kinds
   are tested and flagged.
2. **Per pair** (the design's strata): channel x openness bins of about 50 molecules (``coaccess.openness_strata``),
   plus a mark-density bin (6 quantile bins) when A or B is a footprint or nucleosome type. Effect: the
   Mantel-Haenszel OR with the Robins-Breslow-Greenland CI; test: the exact conditional test given the strata
   (``coaccess.exact_stratified``); BH across pairs; directional readouts P(b = 1 | a = 1) and P(b = 1 | a = 0).
3. **Pooled** over pairs (``test='adjusted'``, the default):
   - test: a pair-stratified score, U = sum over molecules and pairs of (x - x_hat)(y - y_hat), where x_hat and
     y_hat are within-pair means plus pooled covariate adjustments (quantile-bin indicators and linear terms of
     log mark density and openness, and the channel). With no covariates it is the MH score over pairs. Its
     variance is clustered by physical molecule: a molecule's contributions are summed over every pair it spans
     before squaring, each pair's scaled by sqrt(n / (n - 1)). With fewer than ``small_n`` (200) molecules p comes
     from a sign-flip of the molecules' contributions (independent Rademacher signs);
   - effect: the conditional logistic OR of B's state on A's state with the same covariates, pairs as strata
     (``fiberhmm.assoc.clogit``), with a locus-group bootstrap interval (one-step score bootstrap). When one locus
     group holds more than 5% of the pairs the bootstrap resamples whole molecules: the pair strata condition out
     locus effects, so molecules carry the remaining dependence (the two-way and pigeonhole fallbacks were
     conservative here: 95% intervals covered 97-100% under the null).
   ``test='strata'`` instead pools the design's pair x stratum cells (MH score and common MH OR). At about 30x the
   design's strata leave most cells with one molecule, so they lose most of the power; the adjusted score is
   calibrated on the same confounded nulls (``fiberhmm.assoc.simulate.simulate_m1``). The pair x stratum MH OR
   is always reported as a descriptive readout.
4. **Groups** (dataset, haplotype, cluster): the pooled analysis per group and the interaction test: the
   difference of the groups' log ORs with a joint locus-group bootstrap covariance (Wald). Pairs are split by
   group, so every stratum holds one group. Unphased molecules are left out of haplotype contrasts and reported as
   their own group (D14).
5. **Three or more types**: :func:`combination_patterns` (pattern counts against the curveball null,
   ``coaccess.combinations``).

All inputs are plain arrays (no BAM access). Molecules are physical molecules (``mol:v1``): fold records first.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from fiberhmm.assoc import coaccess as C
from fiberhmm.assoc import strata as ST
from fiberhmm.assoc.clogit import clogit, score_bootstrap
from fiberhmm.assoc.groups import group_contrast, group_labels
from fiberhmm.assoc.resample import BlockBootstrap, bh, locus_groups

Z95 = 1.959964
LN2 = float(np.log(2.0))


@dataclass
class MoleculeTable:
    """Per physical molecule covariates. ``ids`` are molecule identities; every other array has one value per
    molecule (``None`` when not available)."""

    ids: np.ndarray
    channel: Optional[np.ndarray] = None
    openness: Optional[np.ndarray] = None
    density: Optional[np.ndarray] = None
    labels: Dict[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self):
        self.ids = np.asarray(self.ids, dtype=object)
        n = len(self.ids)
        for name in ('channel', 'openness', 'density'):
            v = getattr(self, name)
            if v is not None:
                v = np.asarray(v, dtype=object if name == 'channel' else float)
                if len(v) != n:
                    raise ValueError(f'{name}: {len(v)} values for {n} molecules')
                setattr(self, name, v)
        self.labels = {k: np.asarray(v, dtype=object) for k, v in (self.labels or {}).items()}
        for k, v in self.labels.items():
            if len(v) != n:
                raise ValueError(f'label {k}: {len(v)} values for {n} molecules')

    @property
    def n(self) -> int:
        return len(self.ids)

    def index(self) -> Dict[str, int]:
        return {m: i for i, m in enumerate(self.ids)}


def fold_states(hit, molecule, state):
    """One state per (hit, molecule) by the HUB 3.4 fold rule: 1 when every record is 1, 0 when every record is 0,
    otherwise '.' (dropped). Returns (hit, molecule, state) arrays sorted by hit then molecule."""
    hit = np.asarray(hit, dtype=np.int64)
    molecule = np.asarray(molecule, dtype=np.int64)
    state = np.asarray(state, dtype=np.int64)
    if not len(hit):
        return hit, molecule, state
    order = np.lexsort((molecule, hit))
    h, m, s = hit[order], molecule[order], state[order]
    key_change = np.r_[True, (h[1:] != h[:-1]) | (m[1:] != m[:-1])]
    starts = np.flatnonzero(key_change)
    ones = np.add.reduceat(s, starts)
    cnt = np.diff(np.r_[starts, len(s)])
    keep = (ones == 0) | (ones == cnt)
    return h[starts][keep], m[starts][keep], (ones[keep] == cnt[keep]).astype(np.int64)


@dataclass
class FeatureStates:
    """One feature type's per-molecule states at its hits (HUB 5.1 granularity 'state').

    Per hit: ``hit_ids``, ``chrom``, ``start``, ``end``, ``anchor``. Per state row: ``hit`` (index into the hits),
    ``molecule`` (index into a :class:`MoleculeTable`) and ``state`` (1 or 0; '.' rows are simply absent).
    ``kind`` is the overlap rule's kind (overlapping pairs of the same kind are not tested; default the type id).
    ``footprint_like``: a footprint or nucleosome type, which adds the mark-density stratum."""

    type_id: str
    hit_ids: np.ndarray
    chrom: np.ndarray
    start: np.ndarray
    end: np.ndarray
    anchor: np.ndarray
    hit: np.ndarray
    molecule: np.ndarray
    state: np.ndarray
    kind: str = ''
    footprint_like: bool = False

    def __post_init__(self):
        self.hit_ids = np.asarray(self.hit_ids, dtype=object)
        self.chrom = np.asarray(self.chrom, dtype=object)
        self.start = np.asarray(self.start, dtype=np.int64)
        self.end = np.asarray(self.end, dtype=np.int64)
        self.anchor = np.asarray(self.anchor, dtype=np.int64)
        self.hit, self.molecule, self.state = fold_states(self.hit, self.molecule, self.state)
        if not self.kind:
            self.kind = self.type_id
        n = len(self.hit_ids)
        self._starts = np.searchsorted(self.hit, np.arange(n), 'left')
        self._ends = np.searchsorted(self.hit, np.arange(n), 'right')

    @property
    def n_hits(self) -> int:
        return len(self.hit_ids)

    def states_of(self, i: int):
        """(molecules, states) of hit ``i``, sorted by molecule."""
        a, b = self._starts[i], self._ends[i]
        return self.molecule[a:b], self.state[a:b]


def find_pairs(a: FeatureStates, b: FeatureStates, max_distance: int = 5000):
    """Candidate pairs (i, j, nested, skipped reason or ''). When ``a`` and ``b`` are the same set each unordered
    pair appears once."""
    same = a is b
    out = []
    for chrom in sorted(set(a.chrom.tolist()) & set(b.chrom.tolist())):
        ia = np.flatnonzero(a.chrom == chrom)
        jb = np.flatnonzero(b.chrom == chrom)
        ob = jb[np.argsort(b.anchor[jb], kind='stable')]
        bx = b.anchor[ob]
        for i in ia[np.argsort(a.anchor[ia], kind='stable')]:
            lo = np.searchsorted(bx, a.anchor[i] - max_distance, 'left')
            hi = np.searchsorted(bx, a.anchor[i] + max_distance, 'right')
            for j in ob[lo:hi]:
                if same and j <= i:
                    continue
                A = dict(kind=a.kind, start=int(a.start[i]), end=int(a.end[i]))
                B = dict(kind=b.kind, start=int(b.start[j]), end=int(b.end[j]))
                if A['kind'] == B['kind'] and C.overlapping(A, B):
                    out.append((int(i), int(j), '', 'overlap'))
                    continue
                nest = _nesting(A, B) if A['kind'] != B['kind'] else ''
                out.append((int(i), int(j), nest, ''))
    return out


def _nesting(A, B) -> str:
    """'inside' when one interval lies within the other, 'edge' for a partial overlap, '' when disjoint."""
    if min(A['end'], B['end']) - max(A['start'], B['start']) <= 0:
        return ''
    if (A['start'] >= B['start'] and A['end'] <= B['end']) or (B['start'] >= A['start'] and B['end'] <= A['end']):
        return 'inside'
    return 'edge'


def _mh_counts(a, b, c, d):
    """Common MH log OR and RBG 95% CI from per-cell counts (natural log)."""
    a, b, c, d = (np.asarray(v, float) for v in (a, b, c, d))
    n = a + b + c + d
    ok = n > 1
    a, b, c, d, n = a[ok], b[ok], c[ok], d[ok], n[ok]
    R = a * d / n
    S = b * c / n
    P = (a + d) / n
    Q = (b + c) / n
    sR, sS = R.sum(), S.sum()
    if sR <= 0 or sS <= 0:
        return float('nan'), float('nan'), float('nan')
    lor = float(np.log(sR / sS))
    var = (np.sum(P * R) / (2 * sR ** 2) + np.sum(P * S + Q * R) / (2 * sR * sS) + np.sum(Q * S) / (2 * sS ** 2))
    se = float(np.sqrt(var))
    return lor, lor - Z95 * se, lor + Z95 * se




class _Items:
    """Pooled items (one molecule in one pair): pair, molecule, states, the design cell (pair x stratum) and the
    adjusted cell (pair, split by group)."""

    def __init__(self, pair, mol, x, y, cell, unit):
        self.pair = np.asarray(pair, np.int64)
        self.mol = np.asarray(mol, np.int64)
        self.x = np.asarray(x, np.int64)
        self.y = np.asarray(y, np.int64)
        self.cell = np.asarray(cell, np.int64)
        self.unit = np.asarray(unit, np.int64)

    def subset(self, m):
        return _Items(self.pair[m], self.mol[m], self.x[m], self.y[m], self.cell[m], self.unit[m])

    def __len__(self):
        return len(self.pair)


def _cell_counts(it: _Items, n_cells: int):
    """(a, b, c, d) per cell: a = x1 y1, b = x1 y0, c = x0 y1, d = x0 y0."""
    cat = (1 - it.x) * 2 + (1 - it.y)        # 0: a, 1: b, 2: c, 3: d
    cnt = np.bincount(it.cell * 4 + cat, minlength=4 * n_cells).astype(float)
    return cnt.reshape(n_cells, 4).T


def _clustered(u, w_cell, cells, mols, T, small_n, n_flip, rng, label):
    from scipy.stats import norm
    mol_u, mol_idx = np.unique(mols, return_inverse=True)
    Um = np.bincount(mol_idx, weights=u * w_cell[cells], minlength=len(mol_u))
    V = float(np.sum(Um ** 2))
    z = T / np.sqrt(V) if V > 0 else float('nan')
    out = {'n_molecules': int(len(mol_u)), 'U': float(T), 'var_clustered': V, 'z': float(z),
           'p_normal': float(2 * norm.sf(abs(z))) if np.isfinite(z) else float('nan')}
    if len(mol_u) < small_n and V > 0:
        eps = rng.integers(0, 2, size=(n_flip, len(mol_u)), dtype=np.int8) * 2 - 1
        Ts = eps @ Um
        tol = 1e-12 * max(1.0, abs(T))
        out['p_signflip'] = float((1 + np.sum(np.abs(Ts) >= abs(T) - tol)) / (n_flip + 1))
        out['p'] = out['p_signflip']
        out['test'] = f'{label}, molecule-clustered, sign-flip ({n_flip} flips; < {small_n} molecules)'
    else:
        out['p'] = out['p_normal']
        out['test'] = f'{label}, molecule-clustered variance, normal approximation'
    return out


def _mh_score(it: _Items, n_cells: int, small_n, n_flip, rng) -> dict:
    """MH score over the design cells: U = sum (O - E), molecule-clustered variance."""
    if not len(it):
        return {'n_molecules': 0, 'U': 0.0, 'p': float('nan')}
    n = np.bincount(it.cell, minlength=n_cells).astype(float)
    sx = np.bincount(it.cell, weights=it.x, minlength=n_cells)
    sy = np.bincount(it.cell, weights=it.y, minlength=n_cells)
    xbar = np.divide(sx, n, out=np.zeros_like(sx), where=n > 0)
    ybar = np.divide(sy, n, out=np.zeros_like(sy), where=n > 0)
    u = np.where(n[it.cell] >= 2, (it.x - xbar[it.cell]) * (it.y - ybar[it.cell]), 0.0)
    w = np.sqrt(np.divide(n, n - 1, out=np.zeros_like(n), where=n > 1))
    out = _clustered(u, w, it.cell, it.mol, float(u.sum()), small_n, n_flip, rng, 'MH score over pair x stratum')
    out['var_independent'] = float(np.sum(np.divide(sx * (n - sx) * sy * (n - sy), n ** 2 * (n - 1),
                                                    out=np.zeros_like(n), where=n > 1)))
    return out


def _demean(v, unit, n_units):
    v = np.asarray(v, float)
    if v.ndim == 1:
        s = np.bincount(unit, weights=v, minlength=n_units)
        c = np.bincount(unit, minlength=n_units)
        return v - (s / np.maximum(c, 1))[unit]
    return np.column_stack([_demean(v[:, k], unit, n_units) for k in range(v.shape[1])])


def covariate_basis(molecules: 'MoleculeTable', basis: str = 'bins', n_bins: int = 8) -> np.ndarray:
    """Per-molecule covariates for the adjusted test: for log mark density and openness, quantile-bin indicators
    (``n_bins``) and a standardised linear term (``basis='bins'``), or linear and quadratic terms (``'poly'``);
    plus channel indicators. ``'none'``: no covariates (the MH score over pairs)."""
    cols = []
    if basis != 'none':
        for v in (np.log(np.maximum(molecules.density, 1e-6)) if molecules.density is not None else None,
                  molecules.openness):
            if v is None:
                continue
            v = np.asarray(v, float).copy()
            ok = np.isfinite(v)
            if not ok.any():
                continue
            v[~ok] = np.median(v[ok])
            z = (v - v.mean()) / (v.std() or 1.0)
            cols.append(z)
            if basis == 'poly':
                cols.append(z ** 2)
            else:
                b = ST.quantile_bins(v, n_bins)
                for k in range(1, int(b.max()) + 1):
                    cols.append((b == k).astype(float))
            if not ok.all():
                cols.append((~ok).astype(float))
    if molecules.channel is not None:
        u, chi = np.unique(molecules.channel.astype(str), return_inverse=True)
        for k in range(1, len(u)):
            cols.append((chi == k).astype(float))
    return np.column_stack(cols) if cols else np.zeros((molecules.n, 0))


def _adjusted(it: _Items, Zm: np.ndarray, small_n, n_flip, rng) -> dict:
    """The adjusted pair-stratified score (module docstring, item 3)."""
    if not len(it):
        return {'n_molecules': 0, 'U': 0.0, 'p': float('nan')}
    unit = np.unique(it.unit, return_inverse=True)[1]
    nu = int(unit.max()) + 1
    n = np.bincount(unit, minlength=nu).astype(float)
    use = n[unit] >= 2
    xd = _demean(it.x, unit, nu)
    yd = _demean(it.y, unit, nu)
    if Zm.shape[1]:
        Zd = _demean(Zm[it.mol], unit, nu)
        keep = Zd[use].std(0) > 1e-12
        Zd = Zd[:, keep]
        if Zd.shape[1]:
            gx = np.linalg.lstsq(Zd[use], xd[use], rcond=None)[0]
            gy = np.linalg.lstsq(Zd[use], yd[use], rcond=None)[0]
            xd = xd - Zd @ gx
            yd = yd - Zd @ gy
    u = np.where(use, xd * yd, 0.0)
    w = np.sqrt(np.divide(n, n - 1, out=np.zeros_like(n), where=n > 1))
    out = _clustered(u, w, unit, it.mol, float(u.sum()), small_n, n_flip, rng,
                     'pair-stratified covariate-adjusted score')
    out['n_covariates'] = int(Zm.shape[1])
    return out


def _clogit_effect(it: _Items, Zm: np.ndarray) -> dict:
    X = np.column_stack([it.x.astype(float), Zm[it.mol]]) if Zm.shape[1] else it.x[:, None].astype(float)
    # drop covariate columns that never vary within a stratum
    unit = np.unique(it.unit, return_inverse=True)[1]
    if X.shape[1] > 1:
        Xd = _demean(X[:, 1:], unit, int(unit.max()) + 1)
        keep = np.r_[True, Xd.std(0) > 1e-12]
        X = X[:, keep]
    fit = clogit(X, it.y, it.unit)
    return fit


def _pooled_summary(it: _Items, n_cells: int, Zm, test, *, small_n, n_flip, rng, effect=True):
    a, b, c, d = _cell_counts(it, n_cells)
    lor, lo, hi = _mh_counts(a, b, c, d)
    out = {'n_pairs': int(len(np.unique(it.pair))) if len(it) else 0, 'n_items': int(len(it)),
           'mh_strata': {'n_cells': int(len(np.unique(it.cell))) if len(it) else 0,
                         'or_mh': float(np.exp(lor)) if np.isfinite(lor) else float('nan'),
                         'rbg_lo': float(np.exp(lo)) if np.isfinite(lo) else float('nan'),
                         'rbg_hi': float(np.exp(hi)) if np.isfinite(hi) else float('nan'),
                         'note': 'pair x stratum cells; the RBG interval assumes independent tables'},
           'p_b1_given_a1': float(a.sum() / (a.sum() + b.sum())) if a.sum() + b.sum() > 0 else float('nan'),
           'p_b1_given_a0': float(c.sum() / (c.sum() + d.sum())) if c.sum() + d.sum() > 0 else float('nan')}
    mh = _mh_score(it, n_cells, small_n, n_flip, rng)
    out['mh_strata'].update({k: mh[k] for k in ('U', 'z', 'p', 'test') if k in mh})
    if test == 'strata':
        out.update(mh)
        out['log_or'] = lor
        out['or'] = out['mh_strata']['or_mh']
        out['effect'] = 'common MH OR over pair x stratum cells'
        return out, None
    out.update(_adjusted(it, Zm, small_n, n_flip, rng))
    fit = _clogit_effect(it, Zm) if (len(it) and effect) else None
    if fit is not None and fit.get('n_strata'):
        out['log_or'] = float(fit['beta'][0])
        out['or'] = float(np.exp(fit['beta'][0]))
        out['log_or_se_model'] = float(fit['se'][0])
        out['clogit_strata'] = int(fit['n_strata'])
    else:
        out['log_or'] = out['or'] = float('nan')
    out['effect'] = 'conditional logistic OR (pairs as strata, covariate-adjusted)'
    return out, fit


def cooccurrence(a: FeatureStates, b: FeatureStates, molecules: MoleculeTable, *, max_distance: int = 5000,
                 group: Optional[str] = None, group_kind: str = 'label', test: str = 'adjusted',
                 basis: str = 'bins', min_molecules: int = 10, openness_per_bin: int = ST.PER_BIN,
                 density_bins: int = ST.DENSITY_BINS, add_density: Optional[bool] = None, n_boot: int = 1000,
                 n_flip: int = 10000, small_n: int = 200, seed: int = 0, per_pair: bool = True,
                 max_pair_rows: Optional[int] = None, effect: bool = True) -> dict:
    """M1 co-occurrence of A's and B's states on the same molecules (module docstring).

    ``group``: a ``molecules.labels`` key for group contrasts (``group_kind='haplotype'`` applies D14).
    ``test``: 'adjusted' (default) or 'strata'. ``basis``: covariates of the adjusted test ('bins', 'poly',
    'none'). ``add_density``: add the mark-density stratum to the per-pair strata (default: when A or B is
    footprint-like and densities exist). Per-pair rows are reported for pairs with at least ``min_molecules``
    molecules; every pair with two or more molecules enters the pooled analysis. ``effect=False`` skips the
    conditional logistic effect and its interval (the test only)."""
    if test not in ('adjusted', 'strata'):
        raise ValueError(f"test must be 'adjusted' or 'strata', not {test!r}")
    rng = np.random.default_rng(seed)
    if add_density is None:
        add_density = bool((a.footprint_like or b.footprint_like) and molecules.density is not None)
    pairs = find_pairs(a, b, max_distance)
    skipped = [dict(a=str(a.hit_ids[i]), b=str(b.hit_ids[j]), reason=why) for i, j, _n, why in pairs if why]
    pairs = [(i, j, nest) for i, j, nest, why in pairs if not why]

    glab, ginfo = (None, None)
    gcode = None
    if group is not None:
        if group not in molecules.labels:
            raise ValueError(f'no molecule label {group!r} (have {sorted(molecules.labels)})')
        glab, ginfo = group_labels(molecules.labels[group], group_kind)
        gcode = np.array([-1 if g is None else ginfo['groups'].index(g) for g in glab], dtype=np.int64)
    Zm = covariate_basis(molecules, basis) if test == 'adjusted' else np.zeros((molecules.n, 0))

    pair_data = []
    for pid, (i, j, nest) in enumerate(pairs):
        ma, sa = a.states_of(i)
        mb, sb = b.states_of(j)
        common, ia, ib = np.intersect1d(ma, mb, assume_unique=True, return_indices=True)
        pair_data.append((pid, i, j, nest, common, sa[ia], sb[ib], len(ma) - len(common), len(mb) - len(common)))

    def strata_for(mols):
        ch = molecules.channel[mols] if molecules.channel is not None else None
        if molecules.openness is not None:
            return ST.openness_bins(molecules.openness[mols], ch, openness_per_bin)
        return np.zeros(len(mols), np.int64) if ch is None else np.unique(ch.astype(str), return_inverse=True)[1]

    def build(scope_mask):
        """Items and per-pair rows for the molecules selected by ``scope_mask`` (bool per molecule). With a group,
        strata and units are split by group."""
        used = np.zeros(molecules.n, bool)
        for row in pair_data:
            used[row[4][scope_mask[row[4]]]] = True
        dens_edges = (ST.quantile_edges(molecules.density[used], density_bins)
                      if add_density and used.any() else None)
        rows, P, M, X, Y, CELL, UNIT = [], [], [], [], [], [], []
        n_cells = n_units = 0
        for pid, i, j, nest, mols, x, y, oa, ob in pair_data:
            keep = scope_mask[mols]
            mols, x, y = mols[keep], x[keep], y[keep]
            if len(mols) < 2:
                continue
            lab = strata_for(mols)
            if add_density:
                lab = ST.combine(lab, ST.density_bins(molecules.density[mols], density_bins, dens_edges))
            ulab = np.zeros(len(mols), np.int64)
            if gcode is not None:
                lab = ST.combine(lab, gcode[mols])
                ulab = np.unique(gcode[mols], return_inverse=True)[1]
            lab = np.unique(lab, return_inverse=True)[1]
            P.append(np.full(len(mols), pid)); M.append(mols); X.append(x); Y.append(y)
            CELL.append(n_cells + lab); UNIT.append(n_units + ulab)
            n_cells += int(lab.max()) + 1
            n_units += int(ulab.max()) + 1
            if per_pair and len(mols) >= min_molecules:
                rows.append(_pair_row(a, b, i, j, nest, mols, x, y, lab, oa, ob))
        if P:
            it = _Items(*(np.concatenate(v) for v in (P, M, X, Y, CELL, UNIT)))
        else:
            it = _Items([], [], [], [], [], [])
        if rows:
            for r, qq in zip(rows, bh([r['p_exact'] for r in rows])):
                r['q'] = float(qq)
        return it, n_cells, rows

    all_mask = np.ones(molecules.n, bool)
    if glab is not None and group_kind == 'haplotype':
        all_mask = np.array([g is not None for g in glab])
    it_all, nc_all, rows = build(all_mask)
    pooled, fit_all = _pooled_summary(it_all, nc_all, Zm, test, small_n=small_n, n_flip=n_flip, rng=rng,
                                      effect=effect)
    out = {'mode': 'M1', 'a': a.type_id, 'b': b.type_id, 'max_distance': int(max_distance),
           'n_pairs_found': len(pairs) + len(skipped), 'n_pairs_tested': len(pairs), 'skipped': skipped,
           'pooled': pooled,
           'settings': {'test': test, 'basis': basis if test == 'adjusted' else None, 'min_molecules': min_molecules,
                        'openness_per_bin': openness_per_bin, 'density_stratum': bool(add_density),
                        'density_bins': density_bins, 'n_boot': n_boot, 'n_flip': n_flip, 'small_n': small_n,
                        'seed': seed, 'group': group, 'group_kind': group_kind}}
    if per_pair:
        rows.sort(key=lambda r: (r['q'], r['p_exact']))
        out['pairs'] = rows if max_pair_rows is None else rows[:max_pair_rows]
        out['n_pair_rows'] = len(rows)

    group_names, git, gfits = [], None, {}
    if glab is not None:
        group_names = ginfo['groups']
        git = np.array([glab[m] for m in it_all.mol], dtype=object) if len(it_all) else np.zeros(0, object)
        out['groups'] = {}
        for g in group_names:
            out['groups'][g], gfits[g] = _pooled_summary(it_all.subset(git == g), nc_all, Zm, test,
                                                         small_n=small_n, n_flip=n_flip, rng=rng, effect=effect)
        out['group_info'] = ginfo
        if group_kind == 'haplotype' and ginfo.get('unphased'):
            it_u, nc_u, _ = build(np.array([g is None for g in glab]))
            desc, _f = _pooled_summary(it_u, nc_u, Zm, test, small_n=small_n, n_flip=n_flip, rng=rng)
            out['unphased'] = {k: desc.get(k) for k in ('n_molecules', 'n_pairs', 'or', 'p_b1_given_a1',
                                                        'p_b1_given_a0')}
            out['unphased']['note'] = 'descriptive; excluded from haplotype contrasts (D14)'

    if n_boot and len(it_all) and (effect or test == 'strata'):
        boot = _bootstrap(it_all, nc_all, fit_all, test, n_boot, seed + 1, git, group_names, gfits)
        pooled['bootstrap'] = dict(boot.summary('all', pooled['log_or']), **boot.info)
        pooled['or_lo'] = float(np.exp(pooled['bootstrap']['lo']))
        pooled['or_hi'] = float(np.exp(pooled['bootstrap']['hi']))
        for g in group_names:
            bs = boot.summary(g, out['groups'][g]['log_or'])
            out['groups'][g]['or_lo'] = float(np.exp(bs['lo']))
            out['groups'][g]['or_hi'] = float(np.exp(bs['hi']))
        if len(group_names) >= 2:
            est = {g: out['groups'][g]['log_or'] for g in group_names}
            out['interaction'] = group_contrast(est, boot)
            out['interaction']['scale'] = 'log odds ratio'
            out['interaction']['test'] = 'difference of group log ORs, joint locus-group bootstrap covariance (Wald)'
    return out


def _bootstrap(it, n_cells, fit_all, test, n_boot, seed, git, group_names, gfits):
    """Locus-group bootstrap of the pooled log OR (overall and per group), one weight draw shared by every group.
    Adjusted test: one-step score bootstrap of the conditional logistic fit; strata test: the common MH OR
    recomputed from weighted cell counts. Returns a :class:`fiberhmm.assoc.resample.BootDraws`."""
    n_units = int(it.pair.max()) + 1
    ug = locus_groups(it.pair, it.mol, n_units)
    # pair strata condition out locus effects, so chained groups fall back to whole molecules
    bb = BlockBootstrap(ug, it.pair, it.mol, seed=seed, fallback='molecules')
    names = ['all'] + list(group_names)
    masks = {'all': np.ones(len(it), bool)}
    for g in group_names:
        masks[g] = git == g
    if test == 'adjusted':
        fits = dict(gfits, all=fit_all)
        split = bb.mode != 'groups'

        def fn(W):
            out = {}
            for k in names:
                f = fits.get(k)
                if f is None or not f.get('n_strata'):
                    out[k] = np.full(W.shape[0], np.nan)
                else:
                    sub = masks[k]
                    out[k] = score_bootstrap(f, [W[:, sub] if k != 'all' else W], split_strata=split)
            return out
    else:
        from scipy.sparse import csr_matrix
        cat = (1 - it.x) * 2 + (1 - it.y)
        mats = {}
        for k in names:
            idx = np.flatnonzero(masks[k])
            mats[k] = csr_matrix((np.ones(len(idx)), (idx, it.cell[idx] * 4 + cat[idx])),
                                 shape=(len(it), 4 * n_cells))

        def fn(W):
            out = {}
            for k in names:
                cnt = (mats[k].T @ W.T).T.reshape(W.shape[0], n_cells, 4)
                a, b, c, d = cnt[..., 0], cnt[..., 1], cnt[..., 2], cnt[..., 3]
                nn = a + b + c + d
                with np.errstate(divide='ignore', invalid='ignore'):
                    R = np.where(nn > 0, a * d / nn, 0).sum(1)
                    S = np.where(nn > 0, b * c / nn, 0).sum(1)
                    out[k] = np.where((R > 0) & (S > 0), np.log(R / S), np.nan)
            return out
    return bb.collect(fn, n_boot)


def _pair_row(a, b, i, j, nest, mols, x, y, lab, only_a, only_b):
    t = C.table(x, y)
    if min(x.sum(), (1 - x).sum(), y.sum(), (1 - y).sum()) == 0:
        p, e11, o11 = 1.0, float('nan'), int(t[0])
        lor = lo = hi = float('nan')
    else:
        p, e11, o11 = C.exact_stratified(x, y, lab)
        l2, lo2, hi2 = C.mantel_haenszel(x, y, lab)
        lor, lo, hi = l2 * LN2, lo2 * LN2, hi2 * LN2
    n1 = t[0] + t[1]
    n0 = t[2] + t[3]
    return {'a': str(a.hit_ids[i]), 'b': str(b.hit_ids[j]), 'chrom': str(a.chrom[i]),
            'a_anchor': int(a.anchor[i]), 'b_anchor': int(b.anchor[j]),
            'distance': int(abs(int(a.anchor[i]) - int(b.anchor[j]))), 'nested': nest, 'n': int(len(mols)),
            'excluded_no_b_state': int(only_a), 'excluded_no_a_state': int(only_b), 'table': [int(v) for v in t],
            'strata': int(len(np.unique(lab))), 'or_mh': float(np.exp(lor)) if np.isfinite(lor) else float('nan'),
            'or_lo': float(np.exp(lo)) if np.isfinite(lo) else float('nan'),
            'or_hi': float(np.exp(hi)) if np.isfinite(hi) else float('nan'),
            'p_exact': float(p), 'expected_11': float(e11), 'observed_11': int(o11),
            'p_b1_given_a1': float(t[0] / n1) if n1 else float('nan'),
            'p_b1_given_a0': float(t[2] / n0) if n0 else float('nan')}


def combination_patterns(features: Sequence[tuple], n_samples: int = 300, seed: int = 11, min_reads: int = 30) -> dict:
    """Pattern counts of three or more elements on the molecules spanning all of them, against the curveball null
    (``coaccess.combinations``). ``features``: ``(FeatureStates, hit_index)`` tuples."""
    els = []
    for fs, i in features:
        mols, st = fs.states_of(i)
        els.append({'id': str(fs.hit_ids[i]), 'label': str(fs.hit_ids[i]), 'kind': fs.kind,
                    'start': int(fs.start[i]), 'end': int(fs.end[i]),
                    'state': {int(m): int(s) for m, s in zip(mols, st)}})
    return C.combinations(els, None, n_samples=n_samples, seed=seed, min_reads=min_reads)
