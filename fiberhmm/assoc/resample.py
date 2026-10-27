"""Dependence between molecules and loci: locus groups and block resampling (HUB 7.2).

Molecules are shared across pairs and anchors, and pairs share hits. The rules:

- **Locus groups.** Units of analysis (pairs, anchors, matched sets) that share any physical molecule are joined,
  transitively. A group is never cut, so no molecule is split across resampling units.
- **Intervals** come from a bootstrap over locus groups. When one group holds more than ``max_group_share``
  (5%) of all units (long molecules chain loci together), the variance is the two-way clustered variance of
  Cameron, Gelbach and Miller: V = V_units + V_molecules - V_items, each term from its own bootstrap (whole units,
  whole molecules, single items), with normal intervals. The two-way "pigeonhole" bootstrap (units and molecules
  resampled independently, an item weighted by the product of the counts; ``fallback='pigeonhole'``) is
  conservative: on simulated chained data its 95% intervals covered 100% and its group contrasts rejected at
  0-0.5% at a nominal 5%, so it is not the default.

Everything here works on plain integer arrays: ``item_unit`` and ``item_mol`` give, for every item (one molecule
at one unit), its unit and its molecule.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterator, Optional, Sequence, Tuple

import numpy as np

MAX_GROUP_SHARE = 0.05


def locus_groups(item_unit, item_mol, n_units: Optional[int] = None, links: Optional[Sequence] = None) -> np.ndarray:
    """Locus-group label (0..G-1, numbered by first unit) of every unit.

    Units are joined when they share a molecule; ``links`` (pairs of unit indices) adds further joins (for example a
    matched set's members)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    item_unit = np.asarray(item_unit, dtype=np.int64)
    item_mol = np.asarray(item_mol, dtype=np.int64)
    if n_units is None:
        n_units = int(item_unit.max()) + 1 if len(item_unit) else 0
    if n_units == 0:
        return np.zeros(0, dtype=np.int64)
    n_mol = int(item_mol.max()) + 1 if len(item_mol) else 0
    rows = [item_unit]
    cols = [n_units + item_mol]
    if links is not None and len(links):
        lk = np.asarray(links, dtype=np.int64).reshape(-1, 2)
        rows.append(lk[:, 0])
        cols.append(lk[:, 1])
    r = np.concatenate(rows)
    c = np.concatenate(cols)
    n = n_units + n_mol
    g = coo_matrix((np.ones(len(r), dtype=np.int8), (r, c)), shape=(n, n))
    _, lab = connected_components(g, directed=False)
    lab = lab[:n_units]
    # renumber by first appearance so labels do not depend on scipy's numbering
    _, first = np.unique(lab, return_index=True)
    order = np.argsort(first, kind='stable')
    remap = np.empty(len(order), dtype=np.int64)
    remap[order] = np.arange(len(order))
    return remap[np.searchsorted(np.unique(lab), lab)]


@dataclass
class BlockBootstrap:
    """Bootstrap over locus groups, with the two-way fallback for chained groups.

    ``unit_group``: locus group of every unit (from :func:`locus_groups`). ``item_unit``/``item_mol``: the items.
    ``mode`` is ``'groups'`` unless one group holds more than ``max_group_share`` of the units; then it is
    ``fallback``: ``'twoway'`` (default), ``'pigeonhole'``, or ``'molecules'`` (whole physical molecules, for
    statistics whose units are conditioned out, such as M1's pair-stratified fits)."""

    unit_group: np.ndarray
    item_unit: np.ndarray
    item_mol: np.ndarray
    max_group_share: float = MAX_GROUP_SHARE
    seed: int = 0
    fallback: str = 'twoway'

    def __post_init__(self):
        if self.fallback not in ('twoway', 'pigeonhole', 'molecules'):
            raise ValueError("fallback must be 'twoway', 'pigeonhole' or 'molecules'")
        self.unit_group = np.asarray(self.unit_group, dtype=np.int64)
        self.item_unit = np.asarray(self.item_unit, dtype=np.int64)
        self.item_mol = np.asarray(self.item_mol, dtype=np.int64)
        self.n_units = len(self.unit_group)
        self.n_groups = int(self.unit_group.max()) + 1 if self.n_units else 0
        _, mol_idx = np.unique(self.item_mol, return_inverse=True)
        self._mol_idx = mol_idx.astype(np.int64)
        self.n_mols = int(mol_idx.max()) + 1 if len(mol_idx) else 0
        sizes = np.bincount(self.unit_group, minlength=self.n_groups) if self.n_units else np.zeros(0)
        self.largest_share = float(sizes.max() / self.n_units) if self.n_units else 0.0
        self.mode = self.fallback if self.largest_share > self.max_group_share else 'groups'
        self.rng = np.random.default_rng(self.seed)

    @property
    def n_clusters(self) -> int:
        """Independent resampling units behind the variance (for the few-clusters correction)."""
        if self.mode == 'groups':
            return int(self.n_groups)
        if self.mode == 'molecules':
            return int(self.n_mols)
        return int(min(self.n_units, self.n_mols))

    def describe(self) -> dict:
        return {'mode': self.mode, 'units': int(self.n_units), 'locus_groups': int(self.n_groups),
                'molecules': int(self.n_mols), 'largest_group_share': self.largest_share,
                'max_group_share': self.max_group_share, 'clusters': self.n_clusters}

    def _counts(self, n_draw: int, n: int) -> np.ndarray:
        if n == 0:
            return np.zeros((n_draw, 0))
        return self.rng.multinomial(n, np.full(n, 1.0 / n), size=n_draw).astype(np.float64)

    def item_weights(self, n_draw: int, kind: Optional[str] = None) -> np.ndarray:
        """(n_draw, n_items) item weights. ``kind``: 'groups' (whole locus groups), 'units', 'molecules', 'items'
        or 'pigeonhole'; default the mode's own ('groups', 'molecules' or 'pigeonhole'; twoway has no single
        kind)."""
        kind = kind or self.mode
        if kind == 'groups':
            return self._counts(n_draw, self.n_groups)[:, self.unit_group[self.item_unit]]
        if kind == 'units':
            return self._counts(n_draw, self.n_units)[:, self.item_unit]
        if kind == 'molecules':
            return self._counts(n_draw, self.n_mols)[:, self._mol_idx]
        if kind == 'items':
            return self._counts(n_draw, len(self.item_unit))
        if kind == 'pigeonhole':
            wu = self._counts(n_draw, self.n_units)
            wm = self._counts(n_draw, self.n_mols)
            return wu[:, self.item_unit] * wm[:, self._mol_idx]
        raise ValueError(f'{kind}: twoway mode draws units, molecules and items separately (use collect)')

    def iter_item_weights(self, n_boot: int, chunk: int = 64, kind: Optional[str] = None) -> Iterator[np.ndarray]:
        done = 0
        while done < n_boot:
            k = min(chunk, n_boot - done)
            yield self.item_weights(k, kind)
            done += k

    def collect(self, fn, n_boot: int, chunk: Optional[int] = None) -> 'BootDraws':
        """Bootstrap a set of statistics. ``fn(W)`` maps item weights (draws, n_items) to {name: (draws,) array}.
        In twoway mode ``n_boot`` draws are made for each of units, molecules and items."""
        n_items = max(len(self.item_unit), 1)
        chunk = chunk or max(1, min(64, int(2e7 // n_items)))

        def run(kind):
            acc: Dict[str, list] = {}
            for W in self.iter_item_weights(n_boot, chunk, kind):
                for k, v in fn(W).items():
                    acc.setdefault(k, []).extend(np.asarray(v, float).tolist())
            return {k: np.asarray(v) for k, v in acc.items()}
        if self.mode == 'twoway':
            return BootDraws('twoway', parts={k: run(k) for k in ('units', 'molecules', 'items')},
                             info=self.describe())
        return BootDraws(self.mode, draws=run(None), info=self.describe())


def _nancov(cols: Sequence[np.ndarray]) -> np.ndarray:
    X = np.column_stack(cols)
    ok = np.all(np.isfinite(X), axis=1)
    if ok.sum() < 3:
        return np.full((X.shape[1], X.shape[1]), np.nan)
    return np.atleast_2d(np.cov(X[ok], rowvar=False))


@dataclass
class BootDraws:
    """Bootstrap results: ``draws`` {name: array} (groups / pigeonhole) or, in twoway mode, ``parts`` {'units' |
    'molecules' | 'items': {name: array}}."""

    mode: str
    draws: Optional[Dict[str, np.ndarray]] = None
    parts: Optional[Dict[str, Dict[str, np.ndarray]]] = None
    info: Optional[dict] = None

    @property
    def names(self):
        return list((self.draws if self.draws is not None else self.parts['units']).keys())

    def cov(self, names: Sequence[str], part: Optional[str] = None) -> np.ndarray:
        """Covariance of the named statistics. ``part`` (twoway mode): one draw set only ('units', 'molecules' or
        'items') instead of the two-way combination."""
        if self.mode != 'twoway':
            return _nancov([self.draws[n] for n in names])
        if part is not None:
            return _nancov([self.parts[part][n] for n in names])
        c = (_nancov([self.parts['units'][n] for n in names]) + _nancov([self.parts['molecules'][n] for n in names])
             - _nancov([self.parts['items'][n] for n in names]))
        if not np.all(np.isfinite(c)):
            return c
        # positive semi-definite: clip negative eigenvalues, then give any variance clipped to zero the larger
        # one-way variance on the diagonal (adding a non-negative diagonal keeps the matrix PSD)
        c = (c + c.T) / 2
        w, v = np.linalg.eigh(c)
        if w.min() < 0:
            c = (v * np.maximum(w, 0)) @ v.T
            one = np.maximum(np.diag(_nancov([self.parts['units'][n] for n in names])),
                             np.diag(_nancov([self.parts['molecules'][n] for n in names])))
            d = np.diag(c)
            add = np.where(d <= 1e-15 * max(1.0, float(np.max(np.abs(one)))), np.maximum(one, 0), 0.0)
            c = c + np.diag(add)
        return c

    @property
    def clusters(self) -> int:
        return int((self.info or {}).get('clusters') or 0)

    def correction(self, part: Optional[str] = None) -> Tuple[float, float]:
        """(variance factor G / (G - 1), degrees of freedom G - 1) for G resampling clusters (CR1 with t(G - 1));
        (1, inf) when G is unknown. ``part='molecules'`` uses the molecule count."""
        G = self.clusters if part != 'molecules' else int((self.info or {}).get('molecules') or 0)
        if G >= 2:
            return G / (G - 1), G - 1.0
        if G == 1:
            return float('nan'), float('nan')        # one cluster: no cluster-based interval
        return 1.0, float('inf')

    def summary(self, name: str, est: float, level: float = 0.95) -> dict:
        """SE (bootstrap, or two-way) with the few-clusters correction, and a t interval with G - 1 degrees of
        freedom; groups mode also reports the percentile interval (``lo_pct``, ``hi_pct``)."""
        from scipy.stats import t as tdist
        fac, df = self.correction()
        var = self.cov([name])[0, 0]
        se = float(np.sqrt(max(var, 0) * fac)) if (np.isfinite(var) and np.isfinite(fac)) else float('nan')
        q = float(tdist.ppf(0.5 + level / 2, df)) if np.isfinite(df) else (1.959964 if np.isfinite(fac) else
                                                                          float('nan'))
        out = {'estimate': float(est), 'se': se, 'lo': float(est - q * se), 'hi': float(est + q * se), 'df': df,
               'interval': f't({df:g}), {"two-way clustered" if self.mode == "twoway" else self.mode} variance'}
        if self.mode != 'twoway':
            d = self.draws[name]
            out['n_boot'] = int(len(d))
            out['lo_pct'], out['hi_pct'] = percentile_ci(d, level)
        else:
            out['n_boot'] = int(len(self.parts['units'][name]))
        return out

    def draws_of(self, name: str) -> Optional[np.ndarray]:
        return None if self.draws is None else self.draws[name]


def percentile_ci(draws, level: float = 0.95):
    d = np.asarray(draws, dtype=float)
    d = d[np.isfinite(d)]
    if len(d) < 2:
        return (float('nan'), float('nan'))
    a = (1 - level) / 2
    return (float(np.quantile(d, a)), float(np.quantile(d, 1 - a)))


def boot_summary(est: float, draws, level: float = 0.95) -> dict:
    d = np.asarray(draws, dtype=float)
    fin = d[np.isfinite(d)]
    lo, hi = percentile_ci(fin, level)
    return {'estimate': float(est), 'se': float(np.std(fin, ddof=1)) if len(fin) > 1 else float('nan'),
            'lo': lo, 'hi': hi, 'n_boot': int(len(d)), 'n_finite': int(len(fin))}


def perm_p(obs: float, null, tail: str = 'two') -> float:
    """Permutation p-value (1 + #{as or more extreme}) / (N + 1). ``tail``: 'two' (|null| >= |obs|), 'greater',
    'less'. Numerical ties count as exceedances."""
    null = np.asarray(null, dtype=float)
    null = null[np.isfinite(null)]
    if not np.isfinite(obs) or not len(null):
        return float('nan')
    tol = 1e-12 * max(1.0, abs(obs))
    if tail == 'two':
        k = int(np.sum(np.abs(null) >= abs(obs) - tol))
    elif tail == 'greater':
        k = int(np.sum(null >= obs - tol))
    elif tail == 'less':
        k = int(np.sum(null <= obs + tol))
    else:
        raise ValueError(tail)
    return (1 + k) / (len(null) + 1)


def perm_p_two_from_tails(obs: float, null) -> float:
    """Two-sided p = min(1, 2 min(P(null >= obs), P(null <= obs))) with the +1 correction (for nulls that are not
    centred at zero)."""
    return min(1.0, 2 * min(perm_p(obs, null, 'greater'), perm_p(obs, null, 'less')))


def boot_p_two(draws) -> float:
    """Two-sided bootstrap p for 'estimate = 0': twice the smaller tail share of draws on either side of 0 (+1
    correction)."""
    d = np.asarray(draws, dtype=float)
    d = d[np.isfinite(d)]
    if not len(d):
        return float('nan')
    lo = (1 + np.sum(d <= 0)) / (len(d) + 1)
    hi = (1 + np.sum(d >= 0)) / (len(d) + 1)
    return float(min(1.0, 2 * min(lo, hi)))


def wald_p(est: float, se: float) -> float:
    from scipy.stats import norm
    if not (np.isfinite(est) and np.isfinite(se)) or se <= 0:
        return float('nan')
    return float(2 * norm.sf(abs(est / se)))


def random_effects(estimates, ses) -> dict:
    """DerSimonian-Laird random-effects combination (replicate libraries): estimate, SE, tau^2, Q, I^2, z, p."""
    from scipy.stats import chi2, norm
    y = np.asarray(estimates, dtype=float)
    s = np.asarray(ses, dtype=float)
    ok = np.isfinite(y) & np.isfinite(s) & (s > 0)
    y, s = y[ok], s[ok]
    k = len(y)
    if k == 0:
        return {'k': 0, 'estimate': float('nan'), 'se': float('nan'), 'tau2': float('nan'), 'p': float('nan')}
    w = 1 / s ** 2
    fixed = float(np.sum(w * y) / np.sum(w))
    Q = float(np.sum(w * (y - fixed) ** 2))
    c = float(np.sum(w) - np.sum(w ** 2) / np.sum(w))
    tau2 = max(0.0, (Q - (k - 1)) / c) if k > 1 and c > 0 else 0.0
    ws = 1 / (s ** 2 + tau2)
    est = float(np.sum(ws * y) / np.sum(ws))
    se = float(math.sqrt(1 / np.sum(ws)))
    z = est / se
    return {'k': int(k), 'estimate': est, 'se': se, 'lo': est - 1.959964 * se, 'hi': est + 1.959964 * se,
            'tau2': float(tau2), 'Q': Q, 'Q_p': float(chi2.sf(Q, k - 1)) if k > 1 else float('nan'),
            'I2': float(max(0.0, (Q - (k - 1)) / Q)) if Q > 0 and k > 1 else 0.0,
            'z': float(z), 'p': float(2 * norm.sf(abs(z))), 'model': 'DerSimonian-Laird'}


def eb_shrink(estimates, ses) -> dict:
    """Normal-normal empirical-Bayes shrinkage of per-anchor effects (method-of-moments tau^2): shrunk estimates
    and the prior mean and variance. Used for ranking only."""
    y = np.asarray(estimates, dtype=float)
    s = np.asarray(ses, dtype=float)
    ok = np.isfinite(y) & np.isfinite(s) & (s > 0)
    out = np.full(len(y), np.nan)
    if ok.sum() == 0:
        return {'shrunk': out, 'mu': float('nan'), 'tau2': float('nan')}
    re = random_effects(y[ok], s[ok])
    mu, tau2 = re['estimate'], re['tau2']
    b = tau2 / (tau2 + s[ok] ** 2) if tau2 > 0 else np.zeros(ok.sum())
    out[ok] = mu + b * (y[ok] - mu)
    return {'shrunk': out, 'mu': float(mu), 'tau2': float(tau2)}


def bh(p):
    """Benjamini-Hochberg q-values (NaN p-values stay NaN)."""
    p = np.asarray(p, dtype=float)
    q = np.full(len(p), np.nan)
    ok = np.isfinite(p)
    if ok.any():
        from fiberhmm.assoc.coaccess import bh as _bh
        q[ok] = _bh(p[ok])
    return q
