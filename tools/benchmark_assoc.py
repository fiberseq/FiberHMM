#!/usr/bin/env python3
"""Reproducible association scaling benchmark; writes only a requested JSON file.

Use one process per mode/size. Analysis time excludes vectorized synthetic-data
construction. M1/M2 have 30 observations per locus; M2 has one matched background
per anchor. M3 uses SIZE molecules with 30 events and 1.5 kb admissible exposure.
M4 has SIZE hits in each set across a sparse territory (20 kb per anchor).
"""
import argparse
import hashlib
import json
import resource
import time

import numpy as np


def data(mode, n, seed, covariates=False):
    rng = np.random.default_rng(seed)
    chrom = np.full(n, 'chrS')
    pos = np.arange(n, dtype=np.int64) * 20000 + 5000
    if mode == 'M1':
        from fiberhmm.assoc import FeatureStates, MoleculeTable
        h = np.repeat(np.arange(n), 30)
        m = np.arange(n*30)
        def feature(prefix, offset):
            return FeatureStates(prefix, np.arange(n).astype(str), chrom, pos+offset, pos+offset+100,
                                 pos+offset+50, h, m, rng.integers(0, 2, len(m)))
        mt = MoleculeTable(m.astype(str),
                           density=np.exp(rng.normal(-3, 0.5, len(m))) if covariates else None,
                           openness=rng.random(len(m)) if covariates else None,
                           channel=rng.integers(0, 2, len(m)) if covariates else None)
        return feature('a', 0), feature('b', 500), mt
    if mode == 'M2':
        from fiberhmm.assoc.condition import Appearances, LocusTable, mean_metric
        lp = np.ravel(np.column_stack([pos, pos+5000]))
        loci = LocusTable(np.arange(2*n).astype(str), np.repeat(chrom, 2), lp,
                          np.tile([True, False], n), np.repeat(np.arange(n), 2))
        loc = np.repeat(np.arange(2*n), 30)
        # The anchor/background share their 30 physical molecules.
        mol = (loc//2)*30 + np.tile(np.arange(30), 2*n)
        st = rng.integers(0, 2, len(loc))
        y = rng.normal(size=len(loc)) + 0.2*st
        app = Appearances(loc, mol, st, np.zeros(len(loc), int), np.ones(len(loc), int),
                          np.stack([y, np.ones(len(y))], -1))
        return app, loci, mean_metric()
    if mode == 'M3':
        from fiberhmm.assoc.rate import RateData
        return RateData(np.full(n, 1500), [np.array([[200, 800]]) for _ in range(n)],
                        [np.sort(rng.integers(0, 1500, 30)) for _ in range(n)], chrom, pos,
                        np.arange(n).astype(str))
    from fiberhmm.assoc.overlap import HitSet
    return HitSet(chrom, pos, pos+100), HitSet(chrom, pos+5000, pos+5100), [('chrS', 0, n*20000)]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=['M1','M2','M3','M4'], required=True)
    p.add_argument('--size', type=int, required=True)
    p.add_argument('--draws', type=int, default=1000)
    p.add_argument('--covariates', action='store_true', help='M1: include density, openness and chemical channel')
    p.add_argument('--seed', type=int, default=71)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    from fiberhmm.inference.consensus.execution import single_threaded_blas
    with single_threaded_blas():
        sample = data(args.mode, args.size, args.seed, args.covariates)
        start = time.perf_counter()
        if args.mode == 'M1':
            from fiberhmm.assoc import cooccurrence
            result = cooccurrence(*sample, n_boot=args.draws, seed=args.seed)
        elif args.mode == 'M2':
            from fiberhmm.assoc.condition import condition
            result = condition(*sample, n_perm=args.draws, n_role=args.draws, n_boot=args.draws, seed=args.seed)
        elif args.mode == 'M3':
            from fiberhmm.assoc import rate_ratio
            result = rate_ratio(sample, n_shift=args.draws, n_boot=args.draws, seed=args.seed)
        else:
            from fiberhmm.assoc import overlap_enrichment
            result = overlap_enrichment(*sample, n_relocations=args.draws, seed=args.seed)
        elapsed = time.perf_counter()-start
    from fiberhmm.assoc.cli import _json_default
    encoded = json.dumps(result, sort_keys=True, default=_json_default).encode()
    report = dict(mode=args.mode, size=args.size, draws=args.draws, seed=args.seed,
                  covariates=args.covariates, seconds=elapsed,
                  peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss *
                  (1 if __import__('sys').platform == 'darwin' else 1024),
                  result_sha256=hashlib.sha256(encoded).hexdigest())
    from pathlib import Path
    Path(args.output).write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
