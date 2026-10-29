"""M2: state-conditioned response at anchors (HUB 7.4; NUC 3.8).

Given a feature type A with a per-molecule state at its anchors (``1``/``0``), does a response metric M of the
molecules differ by state? Delta = M(S) - M(not S), in flank bins.

**Data.** One row per (locus, molecule) appearance (:class:`Appearances`): the locus (an anchor or a matched
background), the physical molecule, the state, a density stratum, a coverage mask (bit b = the molecule measures
bin b) and per-bin sufficient statistics. A metric (:class:`SumMetric`) is a function of the summed statistics, so
pooled-shape metrics (the phasing resultant R_B, an oscillation amplitude) are refitted exactly under every
permutation and bootstrap draw by re-summing.

**Balance.** Within each cell (locus x density stratum x bin; ``mask_rule='exact'``: x the whole coverage mask)
the two states are weighted to the same total, min(n1, n0), so M(S) and M(not S) compare molecules of the same
density and coverage at the same loci. A cell with one state carries no weight. This keeps the density strata
intact (NUC 2.3: closed molecules had 9x lower density).

**Tests.**

- Delta_anchor (state matters at the anchors): labels permuted within cells, 1,000 permutations.
- Delta_site = Delta_anchor - Delta_random: each anchor and its matched backgrounds form a matched set; under "anchors
  behave like their matched backgrounds" the set's loci are exchangeable, so the test **permutes which locus in
  each set plays the anchor** (1,000 draws of one role per set). Every locus keeps its own molecules and
  self-seeding. The same draws test the per-state site contrasts M_anchor(S) - M_random(S) (``site_state1``)
  and M_anchor(not S) - M_random(not S) (``site_state0``).
- Intervals: a bootstrap over locus groups (sets that share molecules are joined; two-way fallback).
- Per anchor: Delta_a when each state has at least ``min_per_state`` (8) molecules in cells holding both states;
  within-anchor permutation (fixed 1,000, which also gives the SE), BH across anchors, and empirical-Bayes
  shrinkage for ranking. The aggregate is the primary result.
- Groups: Delta per group by permutation within group x cell; the difference of the groups' Deltas gets a joint
  locus-group bootstrap interval (Wald), with two-way clustered variance when locus groups chain. The
  subtraction term resamples whole (matched set, molecule) intersections. The few-clusters correction counts
  only units and molecules that contribute to the contrast; unmatched or ineligible sets supply no degrees
  of freedom. Two-group contrasts use group-specific CR1 factors and a Welch-Satterthwaite reference capped
  by the joint cluster degrees of freedom; each group needs at least two contributing clusters.
  Unphased molecules are left out of haplotype contrasts (D14).
"""
from __future__ import annotations

from .control import checkpoint, cooperative

from dataclasses import dataclass, field, replace
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from fiberhmm.assoc import strata as ST
from fiberhmm.assoc.groups import group_contrast, group_labels
from fiberhmm.assoc.resample import BlockBootstrap, BootDraws, bh, eb_shrink, locus_groups, perm_p_two_from_tails


@dataclass
class SumMetric:
    """A response metric computed from summed per-appearance statistics. ``powers`` gives, per statistic column, the
    power of the balance weight it is summed with (1 for ordinary sums; 2 for squared terms such as |V|^2 in a
    null expectation). ``fn`` maps sums (..., k) to the metric (...)."""

    name: str
    columns: tuple
    powers: tuple
    fn: Callable[[np.ndarray], np.ndarray]

    @property
    def k(self) -> int:
        return len(self.columns)

    def __call__(self, sums):
        with np.errstate(divide='ignore', invalid='ignore'):
            return self.fn(np.asarray(sums, float))


def mean_metric(name: str = 'mean') -> SumMetric:
    """Mean of a per-molecule value: statistics (value, 1)."""
    return SumMetric(name, ('value', 'n'), (1, 1), lambda s: s[..., 0] / s[..., 1])


def rate_metric(name: str = 'rate') -> SumMetric:
    """A pooled rate: statistics (events, exposure)."""
    return SumMetric(name, ('events', 'exposure'), (1, 1), lambda s: s[..., 0] / s[..., 1])


@dataclass
class Appearances:
    """One row per (locus, molecule) appearance. ``stats`` is (n, n_bins, k)."""

    locus: np.ndarray
    molecule: np.ndarray
    state: np.ndarray
    stratum: np.ndarray
    mask: np.ndarray
    stats: np.ndarray
    labels: Dict[str, np.ndarray] = field(default_factory=dict)
    density: Optional[np.ndarray] = None
    channel: Optional[np.ndarray] = None

    def __post_init__(self):
        self.locus = np.asarray(self.locus, np.int64)
        if self.density is not None:
            self.density = np.asarray(self.density, float)
        if self.channel is not None:
            self.channel = np.unique(np.asarray(self.channel).astype(str), return_inverse=True)[1].astype(np.int64)
        self.molecule = np.asarray(self.molecule, np.int64)
        self.state = np.asarray(self.state, np.int64)
        self.stratum = np.asarray(self.stratum, np.int64)
        self.mask = np.asarray(self.mask, np.int64)
        self.stats = np.asarray(self.stats, float)
        if self.stats.ndim == 2:
            self.stats = self.stats[:, None, :]
        n = len(self.locus)
        for name in ('molecule', 'state', 'stratum', 'mask'):
            if len(getattr(self, name)) != n:
                raise ValueError(f'{name}: length differs from locus')
        if self.stats.shape[0] != n:
            raise ValueError('stats: first dimension differs from locus')
        self.labels = {k: np.asarray(v, dtype=object) for k, v in (self.labels or {}).items()}

    @property
    def n_bins(self) -> int:
        return self.stats.shape[1]

    def subset(self, m) -> 'Appearances':
        return Appearances(self.locus[m], self.molecule[m], self.state[m], self.stratum[m], self.mask[m],
                           self.stats[m], {k: v[m] for k, v in self.labels.items()},
                           None if self.density is None else self.density[m],
                           None if self.channel is None else self.channel[m])


@dataclass
class LocusTable:
    """Loci: anchors and matched backgrounds. ``set_id``: the matched set (the anchor's index in its set table);
    -1 for loci in no set. ``is_anchor``: True for anchors."""

    ids: np.ndarray
    chrom: np.ndarray
    pos: np.ndarray
    is_anchor: np.ndarray
    set_id: np.ndarray

    def __post_init__(self):
        self.ids = np.asarray(self.ids, dtype=object)
        self.chrom = np.asarray(self.chrom, dtype=object)
        self.pos = np.asarray(self.pos, np.int64)
        self.is_anchor = np.asarray(self.is_anchor, bool)
        self.set_id = np.asarray(self.set_id, np.int64)

    @property
    def n(self) -> int:
        return len(self.ids)

    @classmethod
    def from_sets(cls, anchors: Sequence[tuple], backgrounds: Sequence[tuple], sets) -> 'LocusTable':
        """``anchors``/``backgrounds``: (chrom, pos) lists; ``sets``: ``matched_backgrounds(...)['sets']``. Loci are the
        anchors (indices 0..A-1) then the used backgrounds."""
        ids, chrom, pos, is_a, sid = [], [], [], [], []
        for i, (c, p) in enumerate(anchors):
            ids.append(f'anchor:{i}'); chrom.append(c); pos.append(p); is_a.append(True); sid.append(-1)
        for s, (ai, bl) in enumerate(sets):
            sid[ai] = s
            for j in bl:
                c, p = backgrounds[j]
                ids.append(f'background:{j}'); chrom.append(c); pos.append(p); is_a.append(False); sid.append(s)
        return cls(np.array(ids, dtype=object), np.array(chrom, dtype=object), np.array(pos), np.array(is_a),
                   np.array(sid))


class _Bin:
    """One bin's eligible appearances, cells, balance weights and per-locus sums."""

    def __init__(self, app: Appearances, loci: LocusTable, metric: SumMetric, b: int, mask_rule: str,
                 balance: bool, min_locus_per_state: int):
        elig = ((app.mask >> b) & 1) == 1
        self.idx = np.flatnonzero(elig)
        self.metric = metric
        loc = app.locus[self.idx]
        st = app.state[self.idx]
        key = [loc] + ([app.channel[self.idx]] if app.channel is not None else []) + [app.stratum[self.idx]] \
            + ([app.mask[self.idx]] if mask_rule == 'exact' else [])
        cell = ST.combine(*key) if len(self.idx) else np.zeros(0, np.int64)
        nc = int(cell.max()) + 1 if len(cell) else 0
        n1 = np.bincount(cell, weights=st, minlength=nc)
        n0 = np.bincount(cell, weights=1 - st, minlength=nc)
        if balance:
            m = np.minimum(n1, n0)
            w1 = np.divide(m, n1, out=np.zeros(nc), where=n1 > 0)
            w0 = np.divide(m, n0, out=np.zeros(nc), where=n0 > 0)
        else:
            w1 = (n1 > 0).astype(float)
            w0 = (n0 > 0).astype(float)
        P = np.asarray(metric.powers, float)
        X = app.stats[self.idx, b, :]
        self.X, self.P, self.nc, self.balance = X, P, nc, balance
        self.X1 = X * (w1[cell][:, None] ** P)
        self.X0 = X * (w0[cell][:, None] ** P)
        self.state = st
        self.cell = cell
        self.locus = loc
        self.molecule = app.molecule[self.idx]
        # per-locus balanced molecules per state (molecules in cells holding both states)
        both = (n1 > 0) & (n0 > 0)
        self.n1_bal = np.bincount(loc, weights=st * both[cell], minlength=loci.n)
        self.n0_bal = np.bincount(loc, weights=(1 - st) * both[cell], minlength=loci.n)
        self.locus_ok = (self.n1_bal >= min_locus_per_state) & (self.n0_bal >= min_locus_per_state)
        k = metric.k
        self.S1 = np.zeros((loci.n, k))
        self.S0 = np.zeros((loci.n, k))
        for j in range(k):
            self.S1[:, j] = np.bincount(loc, weights=self.X1[:, j] * st, minlength=loci.n)
            self.S0[:, j] = np.bincount(loc, weights=self.X0[:, j] * (1 - st), minlength=loci.n)

    def delta(self, S1, S0):
        return self.metric(S1) - self.metric(S0)


def _perm_labels(cell, state, n_perm, rng, *, one_at=None):
    """(n_perm, n) boolean label matrices: state permuted within cells."""
    n = len(cell)
    if n == 0:
        return np.zeros((n_perm, 0), bool)
    if one_at is None:
        order0 = np.argsort(cell, kind='stable')
        cs = cell[order0]
        starts = np.flatnonzero(np.r_[True, cs[1:] != cs[:-1]])
        ones = np.add.reduceat(state[order0], starts)
        pos = np.arange(n)
        gstart = np.repeat(starts, np.diff(np.r_[starts, n]))
        gones = np.repeat(ones, np.diff(np.r_[starts, n]))
        one_at = (pos - gstart) < gones
    keys = cell[None, :] + rng.random((n_perm, n))
    order = np.argsort(keys, axis=1, kind='stable')
    L = np.zeros((n_perm, n), bool)
    np.put_along_axis(L, order, np.broadcast_to(one_at, (n_perm, n)), axis=1)
    return L


def _sets(loci: LocusTable, ok: np.ndarray):
    """Matched sets with an eligible anchor and at least one eligible background: (n_sets, max_members) locus
    indices (anchor first, -1 padded) and member counts."""
    sets: Dict[int, list] = {}
    for l in np.flatnonzero(loci.set_id >= 0):
        sets.setdefault(int(loci.set_id[l]), []).append(int(l))
    rows = []
    for s in sorted(sets):
        mem = sets[s]
        anc = [l for l in mem if loci.is_anchor[l]]
        if len(anc) != 1 or not ok[anc[0]]:
            continue
        bg = [l for l in mem if not loci.is_anchor[l] and ok[l]]
        if bg:
            rows.append([anc[0]] + bg)
    if not rows:
        return np.zeros((0, 1), np.int64), np.zeros(0, np.int64)
    width = max(len(r) for r in rows)
    M = np.full((len(rows), width), -1, np.int64)
    for i, r in enumerate(rows):
        M[i, :len(r)] = r
    return M, np.array([len(r) for r in rows], np.int64)


def _bin_result(bd: _Bin, loci: LocusTable, *, n_perm, n_role, per_anchor, min_per_state, rng, chunk=100,
                return_null=False):
    out = {}
    anc = loci.is_anchor & bd.locus_ok
    S1A, S0A = bd.S1[anc].sum(0), bd.S0[anc].sum(0)
    d_anchor = float(bd.delta(S1A, S0A))
    out['delta_anchor'] = d_anchor
    out['M_anchor_state1'] = float(bd.metric(S1A))
    out['M_anchor_state0'] = float(bd.metric(S0A))
    out['n_anchors'] = int(anc.sum())
    out['n_anchor_molecules'] = int(len(np.unique(bd.molecule[anc[bd.locus]]))) if len(bd.locus) else 0
    # within-anchor permutation (anchor loci only)
    ai = np.flatnonzero(anc[bd.locus])
    null = []
    pa = None
    if n_perm and len(ai):
        chunk = max(1, min(chunk, 2_000_000 // len(ai)))
        cellA, stA = bd.cell[ai], bd.state[ai]
        X1A, X0A = bd.X1[ai], bd.X0[ai]
        locA = bd.locus[ai]
        # Cell membership is fixed across batches; derive the layout once.
        order0 = np.argsort(cellA, kind='stable')
        starts = np.flatnonzero(np.r_[True, np.diff(cellA[order0]) != 0])
        sizes = np.diff(np.r_[starts, len(cellA)])
        ones = np.add.reduceat(stA[order0], starts)
        one_at = np.arange(len(cellA)) - np.repeat(starts, sizes) < np.repeat(ones, sizes)
        want = np.flatnonzero(anc & (bd.n1_bal >= min_per_state) & (bd.n0_bal >= min_per_state)) if per_anchor \
            else np.zeros(0, np.int64)
        if len(want):
            from scipy.sparse import csr_matrix
            col = {int(l): c for c, l in enumerate(want)}
            sel = np.array([col.get(int(l), -1) for l in locA])
            rows_ = np.flatnonzero(sel >= 0)
            G = csr_matrix((np.ones(len(rows_)), (rows_, sel[rows_])), shape=(len(ai), len(want)))
            obs_a = bd.delta(bd.S1[want], bd.S0[want])
            acc = {'sum': np.zeros(len(want)), 'sq': np.zeros(len(want)), 'ge': np.zeros(len(want)),
                   'le': np.zeros(len(want)), 'n': np.zeros(len(want))}
        done = 0
        while done < n_perm:
            checkpoint("permutations", done, n_perm)
            c = min(chunk, n_perm - done)
            L = _perm_labels(cellA, stA, c, rng, one_at=one_at).astype(float)
            S1 = L @ X1A
            S0 = (1 - L) @ X0A
            null.extend(bd.delta(S1, S0).tolist())
            if len(want):
                S1l = np.stack([(L * X1A[:, j]) @ G for j in range(X1A.shape[1])], axis=-1)
                S0l = np.stack([((1 - L) * X0A[:, j]) @ G for j in range(X0A.shape[1])], axis=-1)
                d = np.asarray(bd.delta(S1l, S0l))
                fin = np.isfinite(d)
                tol = 1e-12 * np.maximum(1.0, np.abs(obs_a))
                acc['sum'] += np.where(fin, d, 0).sum(0)
                acc['sq'] += np.where(fin, d ** 2, 0).sum(0)
                acc['ge'] += (fin & (d >= obs_a - tol)).sum(0)
                acc['le'] += (fin & (d <= obs_a + tol)).sum(0)
                acc['n'] += fin.sum(0)
            done += c
        out['p_within_anchor'] = perm_p_two_from_tails(d_anchor, null)
        out['null_mean_within'] = float(np.nanmean(null)) if null else float('nan')
        if return_null:
            out['null_within'] = np.asarray(null, float)
        if len(want):
            nn = np.maximum(acc['n'], 1)
            mean = acc['sum'] / nn
            sd = np.sqrt(np.maximum(acc['sq'] / nn - mean ** 2, 0) * nn / np.maximum(nn - 1, 1))
            p = np.minimum(1.0, 2 * np.minimum((1 + acc['ge']) / (nn + 1), (1 + acc['le']) / (nn + 1)))
            pa = dict(locus=want, delta=obs_a, se=sd, p=p)
    # Delta_site: anchor-role permutation within matched sets
    M, nmem = _sets(loci, bd.locus_ok)
    out['n_sets'] = int(len(nmem))
    if len(nmem):
        k = bd.metric.k
        T1 = np.where(M[..., None] >= 0, bd.S1[np.maximum(M, 0)], 0).sum(1)       # (n_sets, k)
        T0 = np.where(M[..., None] >= 0, bd.S0[np.maximum(M, 0)], 0).sum(1)
        tot1, tot0 = T1.sum(0), T0.sum(0)

        def site(roles):            # roles (r, n_sets) -> (Delta_anchor_sets, Delta_random, Delta_site, site S, site not S)
            loc = np.take_along_axis(np.broadcast_to(M, (roles.shape[0],) + M.shape), roles[..., None], 2)[..., 0]
            A1 = bd.S1[loc].sum(1)
            A0 = bd.S0[loc].sum(1)
            da = bd.delta(A1, A0)
            dr = bd.delta(tot1 - A1, tot0 - A0)
            s1 = bd.metric(A1) - bd.metric(tot1 - A1)
            s0 = bd.metric(A0) - bd.metric(tot0 - A0)
            return da, dr, da - dr, s1, s0
        da, dr, ds, s1, s0 = (float(v[0]) for v in site(np.zeros((1, len(nmem)), np.int64)))
        out['site_state1'] = s1
        out['site_state0'] = s0
        out['delta_anchor_sets'] = da
        out['delta_random'] = dr
        out['delta_site'] = ds
        out['M_random_state1'] = float(bd.metric(tot1 - bd.S1[M[:, 0]].sum(0)))
        out['M_random_state0'] = float(bd.metric(tot0 - bd.S0[M[:, 0]].sum(0)))
        out['n_backgrounds'] = int(nmem.sum() - len(nmem))
        if n_role:
            nulls, n1s, n0s = [], [], []
            done = 0
            while done < n_role:
                checkpoint("roles", done, n_role)
                c = min(chunk, n_role - done)
                roles = np.floor(rng.random((c, len(nmem))) * nmem[None, :]).astype(np.int64)
                r_ = site(roles)
                nulls.append(r_[2]); n1s.append(r_[3]); n0s.append(r_[4])
                done += c
            nulls = np.concatenate(nulls)
            out['p_site_state1'] = perm_p_two_from_tails(s1, np.concatenate(n1s))
            out['p_site_state0'] = perm_p_two_from_tails(s0, np.concatenate(n0s))
            if return_null:
                out['null_site_state1'] = np.concatenate(n1s)
                out['null_site_state0'] = np.concatenate(n0s)
            out['p_site'] = perm_p_two_from_tails(ds, nulls)
            out['null_mean_site'] = float(np.nanmean(nulls))
            if return_null:
                out['null_site'] = nulls
    return out, pa, M, nmem


@cooperative
def condition(app: Appearances, loci: LocusTable, metric: SumMetric, *, bins: Optional[Sequence] = None,
              mask_rule: str = 'bin', balance: bool = True, n_perm: int = 1000, n_role: int = 1000,
              n_boot: int = 1000, per_anchor: bool = True, min_per_state: int = 8, min_locus_per_state: int = 1,
              group: Optional[str] = None, group_kind: str = 'label', seed: int = 0,
              max_anchor_rows: Optional[int] = 5000, return_null: bool = False) -> dict:
    """M2 state-conditioned response (module docstring). ``bins``: labels of the ``app.stats`` bins (default
    0..n_bins-1). ``group``: an ``app.labels`` key for group contrasts. ``return_null``: keep the permutation
    draws (``null_within``, ``null_site``) in each bin, for cross-fitting."""
    if mask_rule not in ('bin', 'exact'):
        raise ValueError("mask_rule must be 'bin' or 'exact'")
    nb = app.n_bins
    labels = list(bins) if bins is not None else list(range(nb))
    if len(labels) != nb:
        raise ValueError('bins: one label per statistics bin')
    rng = np.random.default_rng(seed)
    out = {'mode': 'M2', 'metric': metric.name, 'bins': [], 'settings': dict(
        mask_rule=mask_rule, balance=balance, n_perm=n_perm, n_role=n_role, n_boot=n_boot,
        min_per_state=min_per_state, min_locus_per_state=min_locus_per_state, seed=seed, group=group,
        group_kind=group_kind)}
    glab = None
    scope = np.ones(len(app.locus), bool)
    if group is not None:
        glab, ginfo = group_labels(app.labels[group], group_kind)
        out['group_info'] = ginfo
        if group_kind == 'haplotype':
            scope = np.array([g is not None for g in glab])
    main = app.subset(scope)
    gmain = glab[scope] if glab is not None else None
    # resampling units: matched sets (or the locus when in no set); locus groups over shared molecules
    unit_of_locus = np.where(loci.set_id >= 0, loci.set_id, loci.set_id.max(initial=-1) + 1 + np.arange(loci.n))
    unit_of_locus = np.unique(unit_of_locus, return_inverse=True)[1]
    unit = unit_of_locus[main.locus]
    bb = None
    if n_boot and len(main.locus):
        ug = locus_groups(unit, main.molecule, int(unit_of_locus.max()) + 1)
        bb = BlockBootstrap(ug, unit, main.molecule, seed=seed + 1)
        out['bootstrap'] = bb.describe()
    anchor_rows = []
    group_names = ginfo['groups'] if glab is not None else []
    for b in range(nb):
        bd = _Bin(main, loci, metric, b, mask_rule, balance, min_locus_per_state)
        res, pa, M, nmem = _bin_result(bd, loci, n_perm=n_perm, n_role=n_role, per_anchor=per_anchor,
                                       min_per_state=min_per_state, rng=rng, return_null=return_null)
        res['bin'] = labels[b]
        gres, gbd = {}, {}
        for g in group_names:
            sub = main.subset(gmain == g)
            gbd[g] = _Bin(sub, loci, metric, b, mask_rule, balance, min_locus_per_state)
            gres[g], _pa, _m, _n = _bin_result(gbd[g], loci, n_perm=n_perm, n_role=n_role, per_anchor=False,
                                               min_per_state=min_per_state, rng=rng)
        if bb is not None:
            boot = _boot_bin(bd, loci, bb, n_boot, gmain, group_names, gbd,
                             return_contrast=len(group_names) >= 2)
            if len(group_names) >= 2:
                boot, contrast_boot = boot
            for key in ('delta_anchor', 'delta_random', 'delta_site', 'site_state1', 'site_state0'):
                if key in res and f'all|{key}' in boot.names:
                    res[key + '_ci'] = boot.summary(f'all|{key}', res[key])
            for g in group_names:
                for key in ('delta_anchor', 'delta_site'):
                    if key in gres[g] and f'{g}|{key}' in boot.names:
                        gres[g][key + '_ci'] = boot.summary(f'{g}|{key}', gres[g][key])
            if len(group_names) >= 2:
                key = 'delta_site' if all('delta_site' in gres[g] for g in group_names) else 'delta_anchor'
                if all(f'{g}|{key}' in boot.names for g in group_names):
                    est = {g: gres[g][key] for g in group_names}
                    contrast_boot = _contrast_bootstrap(contrast_boot, bb, loci, gmain, gbd, key)
                    res['interaction'] = group_contrast(est, contrast_boot,
                                                        {g: f'{g}|{key}' for g in group_names},
                                                        cluster_counts=contrast_boot.info['group_clusters'])
                    res['interaction']['statistic'] = key
                    res['interaction']['test'] = 'difference of group Deltas, joint locus-group bootstrap (Wald)'
                    res['interaction']['bootstrap_support'] = contrast_boot.info
        if gres:
            res['groups'] = gres
        out['bins'].append(res)
        if pa is not None:
            for li, d, se, p in zip(pa['locus'], pa['delta'], pa['se'], pa['p']):
                anchor_rows.append({'bin': labels[b], 'locus': str(loci.ids[li]), 'chrom': str(loci.chrom[li]),
                                    'pos': int(loci.pos[li]), 'delta': float(d), 'se_perm': float(se),
                                    'p': float(p), 'n_state1': int(bd.n1_bal[li]), 'n_state0': int(bd.n0_bal[li])})
    if glab is not None and group_kind == 'haplotype' and ginfo.get('unphased'):
        un = app.subset(np.array([g is None for g in glab]))
        desc = []
        for b in range(nb):
            bd = _Bin(un, loci, metric, b, mask_rule, balance, min_locus_per_state)
            r, _pa, _m, _n = _bin_result(bd, loci, n_perm=0, n_role=0, per_anchor=False,
                                         min_per_state=min_per_state, rng=rng)
            desc.append({k: r.get(k) for k in ('delta_anchor', 'delta_site', 'n_anchors', 'n_anchor_molecules')})
        out['unphased'] = {'bins': desc, 'note': 'descriptive; excluded from haplotype contrasts (D14)'}
    if anchor_rows:
        for b in labels:
            rows = [r for r in anchor_rows if r['bin'] == b]
            q = bh([r['p'] for r in rows])
            sh = eb_shrink([r['delta'] for r in rows], [r['se_perm'] for r in rows])
            for r, qq, s in zip(rows, q, sh['shrunk']):
                r['q'] = float(qq)
                r['delta_shrunk'] = float(s)
        anchor_rows.sort(key=lambda r: (str(r['bin']), r['q'], r['p']))
        out['n_anchor_rows'] = len(anchor_rows)
        out['per_anchor'] = anchor_rows if max_anchor_rows is None else anchor_rows[:max_anchor_rows]
    return out


def _contrast_bootstrap(boot: BootDraws, bb: BlockBootstrap, loci: LocusTable, gmain, gbd, key):
    """Keep joint draws, but count only the contrast's contributing resampling clusters.

    Group-specific stratification can leave many loci with no balanced observations,
    especially after excluding unphased molecules. Those loci have zero bootstrap
    contribution and cannot increase the reference degrees of freedom. This changes
    only contrast inference; the pooled and per-group intervals retain their draws.
    """
    active = np.zeros(len(bb.item_unit), bool)
    group_clusters = {}
    for g, b_ in gbd.items():
        if key == 'delta_site':
            members, _ = _sets(loci, b_.locus_ok)
            used = np.zeros(loci.n, bool)
            used[members[members >= 0]] = True
        else:
            used = loci.is_anchor & b_.locus_ok
        take = used[b_.locus]
        if b_.balance:
            n1 = np.bincount(b_.cell, weights=b_.state, minlength=b_.nc)
            n0 = np.bincount(b_.cell, weights=1 - b_.state, minlength=b_.nc)
            take &= ((n1 > 0) & (n0 > 0))[b_.cell]
        cols = np.flatnonzero(gmain == g)
        rows = cols[b_.idx[take]]
        active[rows] = True
        group_units = np.unique(bb.item_unit[rows])
        group_molecules = np.unique(bb.item_mol[rows])
        if boot.mode == 'groups':
            group_clusters[g] = len(np.unique(bb.unit_group[group_units]))
        elif boot.mode == 'molecules':
            group_clusters[g] = len(group_molecules)
        else:
            group_clusters[g] = min(len(group_units), len(group_molecules))
    units = np.unique(bb.item_unit[active])
    molecules = np.unique(bb.item_mol[active])
    groups = np.unique(bb.unit_group[units])
    if boot.mode == 'groups':
        clusters = len(groups)
    elif boot.mode == 'molecules':
        clusters = len(molecules)
    else:
        clusters = min(len(units), len(molecules))
    intersections = len(np.unique(ST.combine(bb.item_unit[active], bb.item_mol[active])))
    info = dict(units=len(units), molecules=len(molecules), intersections=intersections,
                locus_groups=len(groups), clusters=clusters, group_clusters=group_clusters)
    return replace(boot, info=info)


def _boot_bin(bd: _Bin, loci: LocusTable, bb: BlockBootstrap, n_boot, gmain, group_names, gbd,
              *, return_contrast=False):
    """Bootstrap of Delta_anchor, Delta_random and Delta_site for one bin (overall and per group), from shared
    weight draws over the analysed appearances. Each target uses its own eligible loci and sets, as its point
    estimate does. Statistic names are '<target>|<statistic>'."""
    def roles(b_: _Bin):
        Mt, _n = _sets(loci, b_.locus_ok)
        sa = np.zeros(loci.n, bool)
        sb = np.zeros(loci.n, bool)
        if len(Mt):
            sa[Mt[:, 0]] = True
            mem = Mt[:, 1:]
            sb[mem[mem >= 0]] = True
        return loci.is_anchor & b_.locus_ok, sa, sb, len(Mt) > 0
    plan = {'all': (bd, None, roles(bd))}
    for g in group_names:
        plan[g] = (gbd[g], np.flatnonzero(gmain == g), roles(gbd[g]))

    from scipy.sparse import csr_matrix
    onehot = {}
    for key, (b_, _cols, _r) in plan.items():
        n = len(b_.idx)
        rows = np.arange(n)
        onehot[key] = (csr_matrix((b_.state.astype(float), (rows, b_.cell)), shape=(n, max(b_.nc, 1))),
                       csr_matrix(((1 - b_.state).astype(float), (rows, b_.cell)), shape=(n, max(b_.nc, 1))))

    def rebalance(key, b_: _Bin, W):
        """Per-draw balance weights: each cell's resampled state counts give min(n1, n0) / n_state."""
        if not b_.balance:
            n1 = (onehot[key][0].T @ W.T).T
            n0 = (onehot[key][1].T @ W.T).T
            return (n1 > 0).astype(float), (n0 > 0).astype(float)
        n1 = (onehot[key][0].T @ W.T).T
        n0 = (onehot[key][1].T @ W.T).T
        m = np.minimum(n1, n0)
        w1 = np.divide(m, n1, out=np.zeros_like(m), where=n1 > 0)
        w0 = np.divide(m, n0, out=np.zeros_like(m), where=n0 > 0)
        return w1, w0

    def sums(b_: _Bin, W, sel, w1, w0):
        st = b_.state.astype(float)
        S1 = np.zeros((W.shape[0], b_.X.shape[1]))
        S0 = np.zeros_like(S1)
        for pw in np.unique(b_.P):
            cols = np.flatnonzero(b_.P == pw)
            c1 = W * (w1[:, b_.cell] ** pw) * (st * sel)[None, :]
            c0 = W * (w0[:, b_.cell] ** pw) * ((1 - st) * sel)[None, :]
            S1[:, cols] = c1 @ b_.X[:, cols]
            S0[:, cols] = c0 @ b_.X[:, cols]
        return S1, S0

    def fn(Wall):
        out = {}
        for key, (b_, cols, (anc, sa, sb, has_sets)) in plan.items():
            W = Wall[:, b_.idx] if cols is None else Wall[:, cols][:, b_.idx]
            loc = b_.locus
            w1, w0 = rebalance(key, b_, W)
            out[f'{key}|delta_anchor'] = np.asarray(b_.delta(*sums(b_, W, anc[loc].astype(float), w1, w0)))
            if has_sets:
                a1, a0 = sums(b_, W, sa[loc].astype(float), w1, w0)
                r1, r0 = sums(b_, W, sb[loc].astype(float), w1, w0)
                da = b_.delta(a1, a0)
                dr = b_.delta(r1, r0)
                out[f'{key}|delta_random'] = np.asarray(dr)
                out[f'{key}|delta_site'] = np.asarray(da - dr)
                out[f'{key}|site_state1'] = np.asarray(b_.metric(a1) - b_.metric(r1))
                out[f'{key}|site_state0'] = np.asarray(b_.metric(a0) - b_.metric(r0))
        return out
    boot = bb.collect(fn, n_boot)
    if not return_contrast:
        return boot
    if bb.mode != 'twoway':
        return boot, boot

    # The intersection of the two clustering axes is (matched set, molecule),
    # which can contain several appearances. Resampling those appearances
    # independently would split a physical molecule within its matched set.
    # Keep the original interval draws and RNG stream; only the contrast gets
    # this intersection component in place of the appearance component.
    from copy import deepcopy
    rng = deepcopy(bb.rng)
    intersection = ST.combine(bb.item_unit, bb.item_mol)
    n = int(intersection.max()) + 1
    chunk = max(1, min(64, int(2e7 // max(len(intersection), 1))))
    acc = {}
    for start in range(0, n_boot, chunk):
        checkpoint('bootstrap', start, n_boot)
        count = min(chunk, n_boot - start)
        weights = rng.multinomial(n, np.full(n, 1.0 / n), size=count)[:, intersection].astype(float)
        for name, values in fn(weights).items():
            acc.setdefault(name, []).extend(np.asarray(values, float).tolist())
    items = {name: np.asarray(values) for name, values in acc.items()}
    return boot, replace(boot, parts=dict(boot.parts, items=items))
