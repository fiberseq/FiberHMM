#!/usr/bin/env python3
"""fiberhmm-crossstrand -- one-command DAF cross-strand consensus + re-call.

Runs the full cross-strand pipeline on a footprint-called DAF-seq BAM:

    1. fiberhmm-pair   -- pair CT/GA reads of the same molecule by nucleosome
                          dyad-pattern cross-correlation (reciprocal-best +
                          null-calibrated margin gate);
    2. fiberhmm-merge  -- build one both-strand consensus read per pair (union
                          span, deam+/deam- regime in MA);
    3. --recall (ON)   -- re-call footprints on each consensus with BOTH the
                          nucleosome recaller and the TF/Pol II LLR recaller,
                          using both strands jointly (C and G both informative
                          in the both-strand core).

Output is a coordinate-sorted, indexed BAM: one consensus read per confident
pair plus every unmerged read passed through. The intermediate paired BAM is
written to a temp path and removed.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

from fiberhmm.cli.merge import run_merge
from fiberhmm.cli.pair import run_pair
from fiberhmm.crossstrand.pairing import PairParams


def run_pipeline(in_bam, out_bam, params: PairParams, recall=True, enzyme='ddda',
                 pairs_only=False, prob_threshold=0, pairs_tsv=None, io_threads=4):
    t0 = time.time()
    tmp_paired = out_bam + '.paired.tmp.bam'
    print("=== [1/2] fiberhmm-pair ===", file=sys.stderr)
    run_pair(in_bam, tmp_paired, params, prob_threshold=prob_threshold,
             pairs_tsv=pairs_tsv, io_threads=io_threads)
    print(f"=== [2/2] fiberhmm-merge {'--recall' if recall else ''} ===", file=sys.stderr)
    try:
        run_merge(tmp_paired, out_bam, prob_threshold=prob_threshold,
                  pairs_only=pairs_only, io_threads=io_threads,
                  recall=recall, enzyme=enzyme)
    finally:
        for p in (tmp_paired, tmp_paired + '.bai'):
            try:
                os.remove(p)
            except OSError:
                pass
    print(f"=== cross-strand pipeline done [{time.time()-t0:.0f}s] -> {out_bam} ===",
          file=sys.stderr)


def main():
    p = argparse.ArgumentParser(
        prog='fiberhmm-crossstrand',
        description='One-command DAF cross-strand consensus + both-strand '
                    're-call (pair -> merge -> recall).',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Full pipeline, both recallers on the both-strand consensus (default)
    fiberhmm-crossstrand -i calls.bam -o consensus.bam

    # Consensus only (skip the footprint re-call), and only consensus reads
    fiberhmm-crossstrand -i calls.bam -o consensus.bam --no-recall --pairs-only
        """,
    )
    p.add_argument('-i', '--input', required=True, help='Footprint-called DAF BAM (coord-sorted + indexed)')
    p.add_argument('-o', '--output', required=True, help='Output consensus BAM (sorted + indexed)')
    p.add_argument('--no-recall', action='store_true', help='Skip re-calling footprints on consensus reads')
    p.add_argument('--enzyme', default='ddda', help='Model preset for re-call (default ddda)')
    p.add_argument('--pairs-only', action='store_true', help='Emit only consensus reads (drop unmerged passthrough)')
    p.add_argument('--pairs-tsv', default=None, help='Write resolved pairs to this TSV')
    # pairing gate (calibrated defaults)
    p.add_argument('--min-score', type=float, default=0.25, help='Min cross-correlation floor (default 0.25)')
    p.add_argument('--min-margin', type=float, default=0.05, help='Min best-minus-competitor margin (default 0.05)')
    p.add_argument('--null-floor', type=float, default=0.24, help='Wrong-pair baseline / virtual competitor (default 0.24)')
    p.add_argument('--min-overlap', type=int, default=1500, help='Min genomic overlap bp (default 1500)')
    p.add_argument('--min-nucs', type=int, default=4, help='Min nucleosome dyads in the overlap, each read (default 4)')
    p.add_argument('--sigma', type=float, default=30.0, help='Gaussian dyad width bp (default 30)')
    p.add_argument('--grid', type=int, default=10, help='Signal resolution bp (default 10)')
    p.add_argument('--max-lag', type=int, default=60, help='+/- register-shift searched bp (default 60)')
    p.add_argument('-p', '--prob-threshold', type=int, default=0, help='Min ML prob for MM/ML dU calls (default 0)')
    p.add_argument('--io-threads', type=int, default=4, help='htslib compression threads (default 4)')
    args = p.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: input not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    params = PairParams(
        grid_bp=args.grid, sigma_bp=args.sigma, max_lag_bp=args.max_lag,
        min_overlap_bp=args.min_overlap, min_nucs=args.min_nucs,
        min_score=args.min_score, min_margin=args.min_margin, null_floor=args.null_floor,
    )
    run_pipeline(args.input, args.output, params, recall=not args.no_recall,
                 enzyme=args.enzyme, pairs_only=args.pairs_only,
                 prob_threshold=args.prob_threshold, pairs_tsv=args.pairs_tsv,
                 io_threads=args.io_threads)


if __name__ == '__main__':
    main()
