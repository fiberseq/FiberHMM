#!/usr/bin/env python3
"""Controlled assay-sensitivity checks for fixed consensus geometry.

This is a conditional simulation of one known class, not an end-to-end
calibration of discovery, GLMM p-values, or biological TF occupancy. The
all-accessible background uses f=1 and independent Bernoulli sites, deliberately
favorable assumptions. Run from a checkout with PYTHONPATH=. .
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
from scipy.stats import fisher_exact

from fiberhmm.inference.consensus.lattice_recaller import model as model
from fiberhmm.inference.consensus.parameters import RecallerOptions

GEOMETRY = dict(L=[80, 80], R=[112, 112], span=[80, 112])
SCENARIOS = {
    'equal_dense': ({}, {}),
    'equal_lower_efficiency_calibrated': ({}, dict(pa=.45)),
    'equal_lower_efficiency_miscalibrated': ({}, dict(pa=.45, scored_pa=.85)),
    'equal_sparse_but_resolved': ({}, dict(interior_step=16)),
    'equal_uninformative_opposite_strand': ({}, dict(interior_step=64)),
    'equal_unmasked_edge_variant': ({}, dict(variant=80)),
    'equal_shared_variant_mask': (dict(mask=80), dict(variant=80, mask=80)),
    'true_difference': (dict(occupancy=.15), dict(occupancy=.55)),
    'absent_on_one_haplotype': (dict(occupancy=0.), dict(occupancy=.35)),
    'equal_broader_background_lower_efficiency': (dict(broader=.3), dict(broader=.3, pa=.45)),
    'equal_heterogeneous_efficiency_unmodeled': ({}, dict(heterogeneity=3.)),
    'equal_heterogeneous_efficiency_modeled': ({}, dict(heterogeneity=3., per_molecule_rates=True)),
}


def simulate(seed, n, *, occupancy=.35, pa=.85, scored_pa=None,
             interior_step=4, variant=None, mask=None, broader=0.,
             heterogeneity=0., per_molecule_rates=False):
    rng = np.random.default_rng(seed)
    positions = np.arange(0, 200, 4)
    inside = (positions >= 80) & (positions < 112)
    positions = positions[~inside | ((positions - 80) % interior_step == 0)]
    state = rng.random(n)
    bound = state < occupancy
    broad = (state >= occupancy) & (state < occupancy + broader)
    rates = rng.beta(pa*heterogeneity, (1-pa)*heterogeneity, n) if heterogeneity else np.full(n, pa)
    rates = np.clip(rates, .020001, .999999)
    protected = ((bound[:, None] & ((positions >= 80) & (positions < 112))) |
                 (broad[:, None] & ((positions >= 40) & (positions < 152))))
    hit = rng.random((n, len(positions))) < np.where(
        protected, .02, rates[:, None])
    if variant is not None:
        hit[:, positions == variant] = True
    keep = positions != mask if mask is not None else np.ones(len(positions), bool)
    positions, hit = positions[keep], hit[:, keep]
    pp = np.full(len(positions), .02)
    units = []
    for i, h in enumerate(hit):
        pa_scored = np.full(len(positions), rates[i] if per_molecule_rates else pa if scored_pa is None else scored_pa)
        units.append(dict(uid=f'm{i}', pos=positions, hit=h, pa=pa_scored, pp=pp,
                          d=np.where(h, np.log(pp/pa_scored), np.log1p(-pp)-np.log1p(-pa_scored))))
    return units, bound


def analyze(units, bound, unknown_accessible_fraction):
    matrix, keep = model.Scorer(units, [GEOMETRY], unknown_accessible_fraction, RecallerOptions(learned_spots=False)).rows()
    weights, posterior = model.em(matrix)
    lbf = model.log_bayes_factors(matrix, weights)[:, 0]
    confident = (lbf >= np.log(3.)) & (posterior[:, 0] >= .5)
    return dict(n=len(keep), truth=float(bound.mean()), fitted=float(weights[0]),
                confident=int(confident.sum()), bf_positive=int((lbf >= np.log(3.)).sum()),
                resolution=float(model.expected_evidence(units, GEOMETRY)),
                sensitivity=float(confident[bound].mean()) if bound.any() else None,
                false_positive_fraction=float(confident[~bound].mean()) if (~bound).any() else None)


def one(task):
    scenario, rep, n, seed, f = task
    results = [analyze(*simulate(seed + rep*2 + hap, n, **spec), f)
               for hap, spec in enumerate(SCENARIOS[scenario])]
    a, b = results
    p = fisher_exact([[a['confident'], a['n']-a['confident']],
                      [b['confident'], b['n']-b['confident']]]).pvalue
    return dict(scenario=scenario, replicate=rep, h1=a, h2=b, naive_count_p=float(p))


def summarize(rows):
    out = {}
    for scenario in SCENARIOS:
        group = [r for r in rows if r['scenario'] == scenario]
        summary = {'replicates': len(group),
                   'fraction_naive_count_p_below_0_05': float(np.mean([r['naive_count_p'] < .05 for r in group]))}
        for name in ('truth', 'fitted', 'resolution'):
            for hap in ('h1', 'h2'):
                summary[f'{hap}_{name}_mean'] = float(np.mean([r[hap][name] for r in group]))
        for hap in ('h1', 'h2'):
            summary[f'{hap}_confident_fraction_mean'] = float(np.mean([r[hap]['confident']/r[hap]['n'] for r in group]))
        out[scenario] = summary
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replicates', type=int, default=40)
    parser.add_argument('--molecules', type=int, default=200)
    parser.add_argument('--seed', type=int, default=20261111)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--unknown-accessible-fraction', type=float, default=1.,
                        help='Scoring f; 1 is the favorable baseline, .95 checks the production upper clamp')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.replicates < 1 or args.molecules < 1 or args.workers < 1:
        parser.error('replicates, molecules and workers must be positive')
    if not 0 < args.unknown_accessible_fraction <= 1:
        parser.error('unknown-accessible-fraction must be in (0, 1]')
    tasks = [(name, rep, args.molecules, args.seed, args.unknown_accessible_fraction) for name in SCENARIOS for rep in range(args.replicates)]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(one, tasks))
    from fiberhmm.inference.consensus import parameters
    result = dict(seed=args.seed, molecules_per_haplotype=args.molecules,
                  unknown_accessible_fraction=args.unknown_accessible_fraction,
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  implementation_sha256={Path(p).name: hashlib.sha256(Path(p).read_bytes()).hexdigest()
                                         for p in (model.__file__, parameters.__file__)},
                  label_semantics=model.LABEL_SEMANTICS,
                  commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                  assumptions=__doc__, summary=summarize(rows), replicates=rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result['summary'], indent=2))


if __name__ == '__main__':
    main()
