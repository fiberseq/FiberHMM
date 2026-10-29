"""Matched-random backgrounds for M2 (HUB 7.4, NUC 3.8).

Each anchor gets ``k`` (5) background loci matched on:

- the same chromosome;
- the same decile of every matching covariate (coverage; site density: A/T for Hia5, C or G context for DddA and
  DddB; for non-transferable types also prevalence);
- at least ``min_sep`` (2 x flank) from every anchor and from every other chosen background.

For **transferable** state rules (whole-NFR open/closed, nucleosome-covered, peak-open) the candidates are random
positions in the territory, evaluated with the same rule. For **non-transferable** types (a specific TF class, an
SV, an NFR variant) the candidates are other hits of the same type.

An anchor and its backgrounds form a **matched set**; the test of Delta_site permutes which member plays the anchor
(``fiberhmm.assoc.condition``).
"""
from __future__ import annotations

from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from fiberhmm.assoc import strata as ST


def matched_backgrounds(anchors: Sequence[Tuple[str, int]], candidates: Sequence[Tuple[str, int]], *,
                        anchor_covariates: Optional[Mapping[str, Sequence[float]]] = None,
                        candidate_covariates: Optional[Mapping[str, Sequence[float]]] = None, k: int = 5,
                        n_bins: int = 10, min_sep: int = 3000, seed: int = 0) -> dict:
    """Matched sets: ``{'sets': [[anchor_index, [candidate indices]], ...], 'shortfall': n, ...}``.

    Covariate deciles are computed over anchors and candidates pooled. Candidates within ``min_sep`` of any anchor,
    or of an already chosen background, are not used. An anchor that finds fewer than ``k`` matches keeps what it
    found and is counted in ``shortfall``; nothing is relaxed."""
    rng = np.random.default_rng(seed)
    a_chrom = np.array([c for c, _ in anchors], dtype=object)
    a_pos = np.array([int(p) for _, p in anchors], dtype=np.int64)
    c_chrom = np.array([c for c, _ in candidates], dtype=object)
    c_pos = np.array([int(p) for _, p in candidates], dtype=np.int64)
    names = sorted(anchor_covariates or {})
    if set(names) != set(candidate_covariates or {}):
        raise ValueError('anchors and candidates need the same covariates')
    a_bins, c_bins = [], []
    for name in names:
        av = np.asarray(anchor_covariates[name], float)
        cv = np.asarray(candidate_covariates[name], float)
        edges = ST.quantile_edges(np.r_[av, cv], n_bins)
        a_bins.append(ST.quantile_bins(av, n_bins, edges))
        c_bins.append(ST.quantile_bins(cv, n_bins, edges))
    a_key = [tuple([a_chrom[i]] + [int(b[i]) for b in a_bins]) for i in range(len(a_pos))]
    c_key = [tuple([c_chrom[j]] + [int(b[j]) for b in c_bins]) for j in range(len(c_pos))]
    # candidates far enough from every anchor
    ok = np.ones(len(c_pos), bool)
    for chrom in set(c_chrom.tolist()):
        m = c_chrom == chrom
        ax = np.sort(a_pos[a_chrom == chrom])
        if not len(ax):
            continue
        cp = c_pos[m]
        i = np.searchsorted(ax, cp)
        d = np.full(len(cp), np.inf)
        has_r = i < len(ax)
        d[has_r] = np.minimum(d[has_r], ax[i[has_r]] - cp[has_r])
        has_l = i > 0
        d[has_l] = np.minimum(d[has_l], cp[has_l] - ax[i[has_l] - 1])
        ok[np.flatnonzero(m)] = d >= min_sep
    pools: Dict[tuple, list] = {}
    for j in np.flatnonzero(ok):
        pools.setdefault(c_key[j], []).append(int(j))
    for key in pools:
        rng.shuffle(pools[key])
    from collections import deque
    pools = {key: deque(pool) for key, pool in pools.items()}
    chosen_pos: Dict[tuple, list] = {}
    used = np.zeros(len(c_pos), bool)
    sets = []
    shortfall = 0
    for i in rng.permutation(len(a_pos)):
        got = []
        pool = pools.get(a_key[i])
        while pool and len(got) < k:
            j = pool.popleft()
            # Rejected candidates can never become eligible later: chosen
            # backgrounds only accumulate. Each shuffled pool is consumed once.
            cell = int(c_pos[j]) // max(1, min_sep)
            nearby = (p for b in (cell - 1, cell, cell + 1)
                      for p in chosen_pos.get((c_chrom[j], b), ()))
            if any(abs(p - int(c_pos[j])) < min_sep for p in nearby):
                continue
            used[j] = True
            chosen_pos.setdefault((c_chrom[j], cell), []).append(int(c_pos[j]))
            got.append(j)
        if len(got) < k:
            shortfall += 1
        sets.append([int(i), got])
    sets.sort(key=lambda s: s[0])
    return {'sets': sets, 'shortfall': int(shortfall), 'k': int(k), 'n_bins': int(n_bins), 'min_sep': int(min_sep),
            'matched_on': ['chrom'] + names, 'n_candidates': int(len(c_pos)),
            'n_candidates_far_enough': int(ok.sum())}


def candidate_positions(territory: Sequence[Tuple[str, int, int]], step: int = 500, margin: int = 0,
                        seed: Optional[int] = None) -> list:
    """Candidate background positions on a grid of ``step`` bp inside the territory intervals, at least ``margin``
    from each interval's ends (with ``seed``, each grid point is jittered uniformly within its step)."""
    rng = np.random.default_rng(seed) if seed is not None else None
    out = []
    for chrom, start, end in territory:
        lo, hi = int(start) + margin, int(end) - margin
        if hi <= lo:
            continue
        pts = np.arange(lo, hi, step)
        if rng is not None:
            pts = np.minimum(pts + rng.integers(0, step, len(pts)), hi - 1)
        out.extend((chrom, int(p)) for p in pts)
    return out
