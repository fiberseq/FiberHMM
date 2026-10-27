"""Synthetic data with realistic confounding, for calibration and power checks of M1-M3 (HUB 7.10).

The generators plant no effect unless asked. They reproduce the confounders the hub must control:

- **mark density** per molecule drives states (sparse molecules look closed and unbound; NUC 2.1) and
  responses;
- **openness** per molecule drives states;
- **shared long molecules**: each molecule spans many hits, and carries latent factors that act on all of its
  states of one feature type (within-molecule correlation across pairs);
- **locus effects**: hits differ in prevalence; anchors sit in high-coverage loci;
- **clustered hits**.

Every generator returns plain arrays in the form the corresponding mode takes. ``fiberhmm.assoc.overlap`` has
its own generator for M4.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from fiberhmm.assoc.cooccur import FeatureStates, MoleculeTable


def _expit(x):
    return 1.0 / (1.0 + np.exp(-x))


def _molecules(rng, chrom_len, coverage, mean_len, n_datasets=1):
    """Molecules tiling a chromosome at ``coverage``: start, end, log density, openness, channel, dataset."""
    n = int(coverage * chrom_len / mean_len)
    length = np.clip(rng.lognormal(np.log(mean_len), 0.35, n), 2000, 4 * mean_len).astype(np.int64)
    start = rng.integers(-mean_len // 2, chrom_len, n)
    end = start + length
    dataset = rng.integers(0, n_datasets, n)
    logd = rng.normal(np.log(0.08), 0.6, n) + 0.3 * dataset          # density differs between datasets too
    # openness correlates with density (dense molecules read as more open)
    openness = _expit(-1.0 + 0.8 * (logd - np.log(0.08)) + rng.normal(0, 0.7, n))
    channel = rng.integers(0, 2, n)
    return dict(start=start, end=end, logd=logd, openness=openness, channel=channel, dataset=dataset)


def simulate_m1(seed: int = 0, *, chrom_len: int = 400_000, n_a: int = 60, n_b: int = 80, coverage: float = 30,
                mean_len: int = 15_000, confound: float = 1.0, latent: float = 0.8, effect: float = 0.0,
                effect_fraction: float = 1.0, n_datasets: int = 1, effect_by_dataset: Optional[Dict[int, float]] = None,
                unphased_fraction: float = 0.0, callable_p: float = 0.9) -> dict:
    """Two feature types (A: NFR-like, B: footprint-like) on one contig with molecule-level confounding.

    P(state = 1) = expit(locus + confound x (density and openness terms) + channel + latent), with an A-specific and
    a B-specific latent factor per molecule (so a molecule's A states are correlated across hits, and its B states
    too, but A and B are independent given the measured covariates). ``effect`` adds ``effect x (s_a - 0.5)`` to
    B's logit for pairs within 5 kb (a fraction ``effect_fraction`` of B hits), i.e. a true co-occurrence.
    ``effect_by_dataset`` overrides the effect per dataset (an interaction). Haplotype labels 1/2, with
    ``unphased_fraction`` of molecules unphased (label 0)."""
    rng = np.random.default_rng(seed)
    mol = _molecules(rng, chrom_len, coverage, mean_len, n_datasets)
    nm = len(mol['start'])
    za = rng.normal(0, 1, nm) * latent
    zb = rng.normal(0, 1, nm) * latent
    dens_term = mol['logd'] - np.log(0.08)
    open_term = mol['openness'] - 0.3
    ch_term = np.where(mol['channel'] == 1, 0.3, -0.3)

    def place(n, width_lo, width_hi):
        anchors = np.sort(rng.integers(10_000, chrom_len - 10_000, n))
        w = rng.integers(width_lo, width_hi, n)
        return anchors - w // 2, anchors - w // 2 + w, anchors

    sa, ea, xa = place(n_a, 150, 400)
    sb, eb, xb = place(n_b, 10, 40)
    alpha_a = rng.normal(0, 1, n_a)
    alpha_b = rng.normal(-0.5, 1, n_b)
    rows_a, rows_b = [], []
    state_a = {}
    for i in range(n_a):
        span = np.flatnonzero((mol['start'] <= sa[i] - 200) & (mol['end'] >= ea[i] + 200))
        call = span[rng.random(len(span)) < callable_p * _expit(2 + dens_term[span])]
        lp = alpha_a[i] + confound * (1.2 * dens_term[call] + 3.0 * open_term[call]) + ch_term[call] + za[call]
        s = (rng.random(len(call)) < _expit(lp)).astype(np.int64)
        for m, v in zip(call, s):
            rows_a.append((i, m, v))
            state_a[(i, m)] = v
    eff_b = rng.random(n_b) < effect_fraction
    for j in range(n_b):
        span = np.flatnonzero((mol['start'] <= sb[j] - 200) & (mol['end'] >= eb[j] + 200))
        call = span[rng.random(len(span)) < callable_p * _expit(2 + dens_term[span])]
        lp = alpha_b[j] + confound * (1.5 * dens_term[call] + 1.5 * open_term[call]) - ch_term[call] + zb[call]
        if (effect or effect_by_dataset) and eff_b[j]:
            near = np.flatnonzero(np.abs(xa - xb[j]) <= 5000)
            if len(near):
                i = near[np.argmin(np.abs(xa[near] - xb[j]))]
                sav = np.array([state_a.get((i, m), -1) for m in call])
                eff = np.full(len(call), effect, float)
                if effect_by_dataset:
                    eff = np.array([effect_by_dataset.get(int(d), effect) for d in mol['dataset'][call]])
                lp = lp + np.where(sav >= 0, eff * (sav - 0.5), 0.0)
        s = (rng.random(len(call)) < _expit(lp)).astype(np.int64)
        rows_b.extend((j, m, v) for m, v in zip(call, s))
    ra = np.array(rows_a, dtype=np.int64).reshape(-1, 3)
    rb = np.array(rows_b, dtype=np.int64).reshape(-1, 3)
    hap = rng.integers(1, 3, nm)
    hap[rng.random(nm) < unphased_fraction] = 0
    ids = np.array([f'mol{k:06d}' for k in range(nm)], dtype=object)
    mt = MoleculeTable(ids, channel=mol['channel'].astype(str), openness=mol['openness'],
                       density=np.exp(mol['logd']),
                       labels={'dataset': np.array([f'd{d}' for d in mol['dataset']], dtype=object),
                               'haplotype': hap.astype(object)})
    A = FeatureStates('nfr.region', np.array([f'a{i}' for i in range(n_a)], dtype=object), np.array(['chrS'] * n_a),
                      sa, ea, xa, ra[:, 0], ra[:, 1], ra[:, 2])
    B = FeatureStates('fp.class', np.array([f'b{j}' for j in range(n_b)], dtype=object), np.array(['chrS'] * n_b),
                      sb, eb, xb, rb[:, 0], rb[:, 1], rb[:, 2], footprint_like=True)
    return {'a': A, 'b': B, 'molecules': mt}


def simulate_m2(seed: int = 0, *, n_anchors: int = 60, k: int = 5, chrom_len: int = 3_000_000, coverage: float = 25,
                mean_len: int = 15_000, confound: float = 1.0, self_seed: float = 0.3, site_effect: float = 0.0,
                group_effect: Optional[Dict[int, float]] = None, n_datasets: int = 1, flank: int = 1500,
                n_bins: int = 3, unphased_fraction: float = 0.0, density_bins: int = 12) -> dict:
    """M2 data: anchors in high-coverage loci, matched backgrounds (coverage and site-density deciles), molecules
    whose density drives both the state (open/closed) and the response, self-seeding (the state shifts the response
    by ``self_seed`` at anchors and backgrounds alike), a per-molecule random effect shared across the loci a
    molecule spans, and partial coverage masks. ``site_effect`` adds an anchor-only state effect (a true
    Delta_site); ``group_effect`` sets it per dataset. Returns ``{'app', 'loci', 'metric', 'bins', 'sets'}``."""
    from fiberhmm.assoc.background import candidate_positions, matched_backgrounds
    from fiberhmm.assoc.condition import Appearances, LocusTable, mean_metric
    from fiberhmm.assoc import strata as ST
    rng = np.random.default_rng(seed)
    # coverage varies along the contig in 100 kb blocks; anchors prefer high-coverage blocks
    nblk = chrom_len // 100_000
    blk_cov = rng.lognormal(0, 0.35, nblk)
    blk_site = rng.normal(0, 1, nblk)
    mol = _molecules(rng, chrom_len, coverage, mean_len, n_datasets)
    # thin molecules by block coverage factor (keep with prob proportional to the block's factor)
    mid = np.clip((mol['start'] + mol['end']) // 2 // 100_000, 0, nblk - 1)
    keep = rng.random(len(mid)) < blk_cov[mid] / blk_cov.max()
    for key in mol:
        mol[key] = mol[key][keep]
    nm = len(mol['start'])
    u = rng.normal(0, 0.5, nm)
    w = blk_cov / blk_cov.sum()
    a_blk = rng.choice(nblk, size=n_anchors, p=w ** 2 / np.sum(w ** 2))
    a_pos = np.sort(a_blk * 100_000 + rng.integers(5_000, 95_000, n_anchors))
    cand = candidate_positions([('chrS', 5_000, chrom_len - 5_000)], step=700, seed=seed)
    c_pos = np.array([p for _c, p in cand])

    def covars(pos):
        b = np.clip(pos // 100_000, 0, nblk - 1)
        return {'coverage': blk_cov[b] + rng.normal(0, 0.02, len(pos)), 'site_density': blk_site[b]}
    mb = matched_backgrounds([('chrS', int(p)) for p in a_pos], cand, anchor_covariates=covars(a_pos),
                             candidate_covariates=covars(c_pos), k=k, min_sep=2 * flank, seed=seed)
    loci = LocusTable.from_sets([('chrS', int(p)) for p in a_pos], cand, mb['sets'])
    edges = np.linspace(150, flank, n_bins + 1)
    bins = [(float(edges[i]), float(edges[i + 1])) for i in range(n_bins)]
    alpha = rng.normal(0, 0.8, loci.n)
    mu = rng.normal(0, 0.5, (loci.n, n_bins))
    dens = mol['logd'] - np.log(0.08)
    rows = []
    for l in range(loci.n):
        p = loci.pos[l]
        span = np.flatnonzero((mol['start'] <= p - 200) & (mol['end'] >= p + 200))
        if not len(span):
            continue
        st = (rng.random(len(span)) < _expit(alpha[l] + confound * 1.5 * dens[span])).astype(np.int64)
        lo = np.minimum(p - mol['start'][span], mol['end'][span] - p)       # measured flank (both sides)
        mask = ST.bin_mask(np.zeros(len(span)), lo, bins)
        eff = self_seed + (site_effect if loci.is_anchor[l] else 0.0)
        if group_effect and loci.is_anchor[l]:
            eff = self_seed + np.array([group_effect.get(int(d), site_effect) for d in mol['dataset'][span]])
        y = (mu[l][None, :] + confound * 0.8 * dens[span][:, None] + u[span][:, None]
             + np.asarray(eff)[..., None] * st[:, None] * 1.0 + rng.normal(0, 1.0, (len(span), n_bins)))
        for t, m in enumerate(span):
            rows.append((l, m, st[t], mask[t], y[t]))
    loc = np.array([r[0] for r in rows]); mm = np.array([r[1] for r in rows])
    stv = np.array([r[2] for r in rows]); msk = np.array([r[3] for r in rows])
    Y = np.array([r[4] for r in rows])
    stats = np.stack([Y, np.ones_like(Y)], axis=-1)
    measured = ((msk[:, None] >> np.arange(n_bins)[None, :]) & 1).astype(float)
    stats = stats * measured[..., None]
    strat = ST.density_bins(np.exp(mol['logd'][mm]), density_bins)
    hap = rng.integers(1, 3, nm).astype(object)
    hap[rng.random(nm) < unphased_fraction] = 0
    app = Appearances(loc, mm, stv, strat, msk, stats,
                      labels={'dataset': np.array([f'd{d}' for d in mol['dataset'][mm]], dtype=object),
                              'haplotype': hap[mm]}, density=np.exp(mol['logd'][mm]))
    return {'app': app, 'loci': loci, 'metric': mean_metric(), 'bins': bins, 'sets': mb}


def simulate_m3(seed: int = 0, *, n_molecules: int = 400, mean_len: int = 15_000, beta: float = 0.0,
                beta_by_dataset: Optional[Dict[int, float]] = None, n_datasets: int = 1, confound: float = 1.0,
                clustered: bool = True, unphased_fraction: float = 0.0, chrom_len: int = 2_000_000) -> dict:
    """M3 data in compressed coordinates. Per molecule: a density that raises both the fraction of the molecule in
    A segments (regular arrays) and the B rate (footprints), masked stretches (junctions), A segments of 0.8-5 kb
    placed independently of B, and B points either Poisson or clustered (2-6 points within 300 bp). ``beta``
    multiplies B's rate inside A by exp(beta) (a planted effect); ``beta_by_dataset`` per dataset."""
    from fiberhmm.assoc.rate import RateData
    rng = np.random.default_rng(seed)
    L = np.clip(rng.lognormal(np.log(mean_len), 0.4, n_molecules), 3000, 60_000).astype(np.int64)
    logd = rng.normal(0, 0.6, n_molecules)
    dataset = rng.integers(0, n_datasets, n_molecules)
    mid = rng.integers(0, chrom_len, n_molecules)
    intervals, points, junctions = [], [], []
    for m in range(n_molecules):
        frac = _expit(-1.0 + confound * 0.8 * logd[m])
        iv, x = [], int(rng.integers(0, 2000))
        while x < L[m]:
            w = int(rng.integers(800, 5000))
            if rng.random() < frac:
                iv.append((x, min(L[m], x + w)))
            x += w + int(rng.integers(100, 1500))
        iv = np.array(iv, np.int64).reshape(-1, 2)
        ind = np.zeros(L[m], bool)
        for a, b in iv:
            ind[a:b] = True
        rate = 1.0 / 700 * np.exp(confound * 0.6 * logd[m])
        b_eff = beta if not beta_by_dataset else beta_by_dataset.get(int(dataset[m]), beta)
        if clustered:
            n_c = rng.poisson(rate * L[m] / 4)
            centres = rng.integers(0, L[m], n_c)
            pts = np.concatenate([np.clip(c + rng.integers(-150, 150, rng.integers(2, 7)), 0, L[m] - 1)
                                  for c in centres]) if n_c else np.zeros(0, np.int64)
        else:
            pts = rng.integers(0, L[m], rng.poisson(rate * L[m]))
        if b_eff and len(pts):
            # thin outside / keep inside to give a rate ratio of exp(beta) at the same expected total
            r = np.exp(b_eff)
            keep_p = np.where(ind[pts], min(1.0, r), min(1.0, 1 / r))
            pts = pts[rng.random(len(pts)) < keep_p]
        points.append(np.unique(pts))
        intervals.append(iv)
        junctions.append(np.sort(rng.integers(1, L[m] - 1, rng.poisson(2))))
    hap = rng.integers(1, 3, n_molecules).astype(object)
    hap[rng.random(n_molecules) < unphased_fraction] = 0
    return RateData(L, intervals, points, np.array(['chrS'] * n_molecules, dtype=object), mid,
                    np.array([f'mol{m:05d}' for m in range(n_molecules)], dtype=object),
                    labels={'dataset': np.array([f'd{d}' for d in dataset], dtype=object), 'haplotype': hap},
                    junctions=junctions)
