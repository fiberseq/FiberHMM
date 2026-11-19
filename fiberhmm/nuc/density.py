"""Mark-density quantities (NUC 3.1).

Three quantities, kept distinct:

``rho_read``
    marks / informative sites over the whole read.
``lambda_acc``
    marks / informative sites inside the molecule's MSPs, empirical-Bayes
    shrunk towards the dataset (beta-binomial). The caller places MSP
    boundaries on marks, so the ``msp_boundary_sites`` informative sites at
    each MSP end are left out: they are marks by construction and would
    inflate the rate (a 3 bp linker between two snapped footprints reads as
    lambda = 1 otherwise).
``lambda_i``
    the same rate restricted to MSP sequence within ``lambda_local_flank`` bp
    of footprint i; falls back to lambda_acc below ``lambda_local_min_sites``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

from fiberhmm.nuc.params import DEFAULT_PARAMS, NucParams
from fiberhmm.nuc.records import NucMolecule


@dataclass(frozen=True)
class BetaPrior:
    """Beta(alpha, beta) prior on a molecule's accessible marking rate."""
    alpha: float
    beta: float

    def shrink(self, k, n):
        k = np.asarray(k, dtype=float)
        n = np.asarray(n, dtype=float)
        return (k + self.alpha) / (n + self.alpha + self.beta)

    def to_dict(self):
        return {'alpha': float(self.alpha), 'beta': float(self.beta)}

    @classmethod
    def from_dict(cls, d):
        return cls(float(d['alpha']), float(d['beta']))


#: weak prior used when no dataset prior is available yet.
FLAT_PRIOR = BetaPrior(0.5, 0.5)


def rho_read(mol: NucMolecule) -> float:
    n = mol.n_informative
    return float(mol.n_hits) / n if n else float('nan')


def _interior_index_ranges(mol: NucMolecule, intervals: np.ndarray, boundary_sites: int):
    """Informative-site index ranges ``[lo, hi)`` (into ``flatnonzero(informative)``)
    of each interval's interior, ``boundary_sites`` sites dropped at each end."""
    inf_pos = np.flatnonzero(mol.informative)
    if len(intervals) == 0:
        z = np.zeros(0, dtype=np.int64)
        return inf_pos, z, z
    i0 = np.searchsorted(inf_pos, intervals[:, 0]) + boundary_sites
    i1 = np.searchsorted(inf_pos, intervals[:, 1]) - boundary_sites
    ok = i1 > i0
    return inf_pos, i0[ok], i1[ok]


def _interior_sites(mol: NucMolecule, intervals: np.ndarray, boundary_sites: int,
                    exclude: Optional[Tuple[int, int]] = None) -> Tuple[int, int]:
    """(marks, informative) inside ``intervals`` minus ``boundary_sites``
    informative sites at each end of each interval, optionally leaving out
    molecular ``exclude``."""
    inf_pos, lo, hi = _interior_index_ranges(mol, intervals, boundary_sites)
    if len(lo) == 0:
        return 0, 0
    ch = np.concatenate([[0], np.cumsum(mol.hits[inf_pos], dtype=np.int64)])
    if exclude is not None:
        e0 = np.searchsorted(inf_pos, exclude[0])
        e1 = np.searchsorted(inf_pos, exclude[1])
        lo_a, hi_a = lo, np.minimum(hi, e0)
        lo_b, hi_b = np.maximum(lo, e1), hi
        lo = np.concatenate([lo_a, lo_b])
        hi = np.concatenate([hi_a, hi_b])
        ok = hi > lo
        lo, hi = lo[ok], hi[ok]
    n = int((hi - lo).sum())
    k = int((ch[hi] - ch[lo]).sum())
    return k, n


def accessible_counts(mol: NucMolecule, params: NucParams = DEFAULT_PARAMS,
                      exclude: Optional[Tuple[int, int]] = None) -> Tuple[int, int]:
    """(marks, informative sites) inside the molecule's MSP interiors.

    ``exclude`` is a molecular ``[start, end)`` left out (anchor +/- 2 kb when
    conditioning, so the anchor's state cannot drive lambda_acc).
    """
    return _interior_sites(mol, mol.msps, params.msp_boundary_sites, exclude=exclude)


def lambda_acc(mol: NucMolecule, prior: BetaPrior = FLAT_PRIOR,
               params: NucParams = DEFAULT_PARAMS,
               exclude: Optional[Tuple[int, int]] = None) -> float:
    k, n = accessible_counts(mol, params, exclude)
    return float(prior.shrink(k, n))


def fit_beta_prior(ks: Sequence[int], ns: Sequence[int], min_sites: int = 50) -> BetaPrior:
    """Method-of-moments beta-binomial prior over molecules.

    Uses molecules with at least ``min_sites`` informative sites; the
    between-molecule variance is the observed variance of k/n minus the mean
    binomial variance. Falls back to a weak prior at the pooled mean when the
    excess variance is not positive.
    """
    k = np.asarray(ks, dtype=float)
    n = np.asarray(ns, dtype=float)
    ok = n >= min_sites
    if ok.sum() < 5:
        p = (k.sum() + 0.5) / (n.sum() + 1.0) if n.sum() > 0 else 0.5
        return BetaPrior(max(p * 2, 1e-3), max((1 - p) * 2, 1e-3))
    k, n = k[ok], n[ok]
    p = k / n
    m = float(np.average(p, weights=n))
    v_obs = float(np.average((p - m) ** 2, weights=n))
    v_bin = float(np.mean(m * (1 - m) / n))
    v = v_obs - v_bin
    if v <= 0 or m <= 0 or m >= 1:
        conc = 1000.0
    else:
        conc = max(m * (1 - m) / v - 1.0, 1.0)
    return BetaPrior(max(m * conc, 1e-3), max((1 - m) * conc, 1e-3))


def footprint_lambdas(mol: NucMolecule, intervals: np.ndarray, lam_mol: float,
                      params: NucParams = DEFAULT_PARAMS,
                      prior: BetaPrior = FLAT_PRIOR) -> np.ndarray:
    """lambda_i for each footprint: MSP interiors within the flank window,
    shrunk towards the molecule's lambda_acc; lambda_acc below the site floor."""
    out = np.full(len(intervals), lam_mol, dtype=float)
    if len(intervals) == 0 or len(mol.msps) == 0:
        return out
    flank = params.lambda_local_flank
    inf_pos, lo_i, hi_i = _interior_index_ranges(mol, mol.msps, params.msp_boundary_sites)
    if len(lo_i) == 0:
        return out
    diff = np.zeros(len(inf_pos) + 1, dtype=np.int64)
    np.add.at(diff, lo_i, 1)
    np.add.at(diff, hi_i, -1)
    keep = np.cumsum(diff[:-1]) > 0
    sites = inf_pos[keep]
    if len(sites) == 0:
        return out
    cum_hit = np.concatenate([[0], np.cumsum(mol.hits[sites])])
    lo = np.searchsorted(sites, intervals[:, 0] - flank)
    hi = np.searchsorted(sites, intervals[:, 1] + flank)
    n = hi - lo
    k = cum_hit[hi] - cum_hit[lo]
    ok = n >= params.lambda_local_min_sites
    # local rate, lightly shrunk towards the molecule's rate (10 pseudo-sites)
    out[ok] = (k[ok] + 10.0 * lam_mol) / (n[ok] + 10.0)
    return out


def edge_informativeness(mol: NucMolecule, intervals: np.ndarray,
                         params: NucParams = DEFAULT_PARAMS) -> Tuple[np.ndarray, np.ndarray]:
    """Informative sites within ``edge_window`` bp outside each edge."""
    w = params.edge_window
    if len(intervals) == 0:
        z = np.zeros(0, dtype=np.int64)
        return z, z
    _, nl = mol.counts(intervals[:, 0] - w, intervals[:, 0])
    _, nr = mol.counts(intervals[:, 1], intervals[:, 1] + w)
    return np.asarray(nl), np.asarray(nr)


def edge_truncated(mol: NucMolecule, intervals: np.ndarray) -> np.ndarray:
    """A footprint is edge-truncated when one of its boundaries is a read end
    or a no-call block rather than a mark: no mark lies between that edge and
    the read end, or a no-call block sits between the edge and the nearest
    mark outside it."""
    if len(intervals) == 0:
        return np.zeros(0, dtype=bool)
    hit_pos = np.flatnonzero(mol.hits)
    s, e = intervals[:, 0], intervals[:, 1]
    li = np.searchsorted(hit_pos, s) - 1          # last mark before s
    ri = np.searchsorted(hit_pos, e)              # first mark at/after e
    left_mark = np.where(li >= 0, hit_pos[np.clip(li, 0, max(len(hit_pos) - 1, 0))] if len(hit_pos) else -1, -1)
    right_mark = np.where(ri < len(hit_pos), hit_pos[np.clip(ri, 0, max(len(hit_pos) - 1, 0))] if len(hit_pos) else mol.length, mol.length)
    trunc = (left_mark < 0) | (right_mark >= mol.length)
    if len(mol.no_call):
        for a, b in mol.no_call:
            # a block between the left mark and the footprint, or between the
            # footprint and the right mark
            trunc |= (b > left_mark) & (a < s + 1)
            trunc |= (a < right_mark) & (b > e - 1)
    return trunc
