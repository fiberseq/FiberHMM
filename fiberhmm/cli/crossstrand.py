#!/usr/bin/env python3
"""DEPRECATED compatibility wrapper: use ``fiberhmm-pair`` (pair -> merge -> recall).

Runs the full cross-strand pipeline on a footprint-called DAF-seq BAM:

    1. fiberhmm-pair   -- pair CT/GA reads from direct sequence support plus
                          the high-confidence sequence-free model;
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

from fiberhmm.models import DEFAULT_PROB_THRESHOLD
from fiberhmm.cli.merge import run_merge
from fiberhmm.cli.duplex import run_pairing
from fiberhmm.crossstrand.pairing import PairParams
from fiberhmm.crossstrand.duplex import DuplexParams


def run_pipeline(in_bam, out_bam, params: PairParams, recall=True, enzyme='ddda',
                 pairs_only=False, prob_threshold=0, pairs_tsv=None, io_threads=4,
                 reference_path=None, phase_nrl=0,
                 nuc_recall_policy='conservative',
                 derived_tf_max_edge_ambiguity=12):
    if enzyme.lower() != 'ddda':
        raise ValueError(
            "cross-strand consensus is specific to double-strand DddA DAF-seq; "
            f"unsupported enzyme preset: {enzyme!r}"
        )
    if not reference_path:
        raise ValueError(
            "the unified default pairer requires a reference FASTA for "
            "non-CpG DddA opportunity enumeration"
        )
    t0 = time.time()
    tmp_paired = out_bam + '.paired.tmp.bam'
    print("=== [1/2] fiberhmm-pair ===", file=sys.stderr)
    run_pairing(
        in_bam, tmp_paired, reference_path,
        params=DuplexParams(
            min_overlap_bp=params.min_overlap_bp,
            min_nucs=params.min_nucs,
        ),
        sequence_params=params,
        prob_threshold=prob_threshold,
        pairs_tsv=pairs_tsv,
        io_threads=io_threads,
        create_index=False,
        pairing_mode='hybrid',
    )
    print(f"=== [2/2] fiberhmm-merge {'--recall' if recall else ''} ===", file=sys.stderr)
    try:
        run_merge(tmp_paired, out_bam, prob_threshold=prob_threshold,
                  pairs_only=pairs_only, io_threads=io_threads,
                  recall=recall, enzyme=enzyme, phase_nrl=phase_nrl,
                  nuc_recall_policy=nuc_recall_policy,
                  derived_tf_max_edge_ambiguity=(
                      derived_tf_max_edge_ambiguity))
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
        prog='python -m fiberhmm.cli.crossstrand',
        description='One-command DAF cross-strand consensus + both-strand '
                    're-call (pair -> merge -> recall).',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Full pipeline, both recallers on the both-strand consensus (default)
    # Deprecated: use fiberhmm-pair (same pipeline, supported command).
    python -m fiberhmm.cli.crossstrand -i calls.bam -o consensus.bam -r hg38.fa

    # Consensus only (skip the footprint re-call), and only consensus reads
    python -m fiberhmm.cli.crossstrand -i calls.bam -o consensus.bam -r hg38.fa --no-recall --pairs-only
        """,
    )
    p.add_argument('-i', '--input', required=True, help='Footprint-called DAF BAM (coord-sorted + indexed)')
    p.add_argument('-o', '--output', required=True, help='Output consensus BAM (sorted + indexed)')
    p.add_argument('-r', '--reference', default=None,
                   help='Reference FASTA required by the unified default pairer')
    p.add_argument('--no-recall', action='store_true', help='Skip re-calling footprints on consensus reads')
    p.add_argument('--enzyme', default='ddda', choices=['ddda'],
                   help='Cross-strand mode is specific to DddA DAF-seq')
    from fiberhmm.core.bam_reader import add_daf_run_mask_arguments, apply_daf_run_mask_arguments
    add_daf_run_mask_arguments(p)
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
    p.add_argument('--min-sequence-bases', type=int, default=500,
                   help='Min shared reference-A/T bases for a sequence edge (default 500)')
    p.add_argument('--max-sequence-mismatch-rate', type=float, default=0.002,
                   help='Hard veto above this sequence difference rate (default 0.002)')
    p.add_argument('--min-component-discordance-rate', type=float, default=0.02,
                   help='Min rejected-edge difference rate to constrain a 2x2 (default 0.02)')
    p.add_argument('--max-sequence-pair-rate', type=float, default=0.01,
                   help='Max difference rate on a sequence-selected pair (default 0.01)')
    p.add_argument('--min-sequence-margin', type=float, default=0.001,
                   help='Min sequence preference/assignment margin (default 0.001)')
    p.add_argument('-p', '--prob-threshold', type=int,
                   default=DEFAULT_PROB_THRESHOLD,
                   help='Min ML probability for MM/ML-native dU calls (0-255; '
                        f'default {DEFAULT_PROB_THRESHOLD}). R/Y- and MD-encoded '
                        'input is binary and ignores it.')
    p.add_argument('--io-threads', type=int, default=4, help='htslib compression threads (default 4)')
    p.add_argument('--phase-nrl', type=int, default=0,
                   help='Periodicity prior for consensus recall: nucleosome repeat length '
                        'in bp, or 0 for off (default 0). DddA radial recall ignores it.')
    p.add_argument('--nuc-recall-policy', choices=['conservative', 'topology'],
                   default='conservative',
                   help='Nucleosome geometry policy for consensus recall')
    p.add_argument(
        '--ddda-derived-tf-max-edge-gap', type=int, default=12, metavar='BP',
        help='Require TF calls exposed solely by DddA radial nucleosome '
             'refinement to have a deamination hit within BP on both sides '
             '(default 12; -1 disables).',
    )
    args = p.parse_args()
    print('fiberhmm-crossstrand is deprecated: use `fiberhmm-pair` (pair -> merge -> recall).',
          file=sys.stderr)
    try:
        apply_daf_run_mask_arguments(args, args.enzyme)
    except ValueError as exc:
        p.error(str(exc))

    if args.ddda_derived_tf_max_edge_gap < -1:
        p.error("--ddda-derived-tf-max-edge-gap must be -1 or >= 0")

    if not os.path.exists(args.input):
        print(f"Error: input not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    params = PairParams(
        grid_bp=args.grid, sigma_bp=args.sigma, max_lag_bp=args.max_lag,
        min_overlap_bp=args.min_overlap, min_nucs=args.min_nucs,
        min_score=args.min_score, min_margin=args.min_margin, null_floor=args.null_floor,
        min_sequence_bases=args.min_sequence_bases,
        max_sequence_mismatch_rate=args.max_sequence_mismatch_rate,
        min_component_discordance_rate=args.min_component_discordance_rate,
        min_sequence_margin=args.min_sequence_margin,
        max_sequence_pair_rate=args.max_sequence_pair_rate,
    )
    run_pipeline(args.input, args.output, params, recall=not args.no_recall,
                 enzyme=args.enzyme, pairs_only=args.pairs_only,
                 prob_threshold=args.prob_threshold, pairs_tsv=args.pairs_tsv,
                 io_threads=args.io_threads, reference_path=args.reference,
                 phase_nrl=args.phase_nrl,
                 nuc_recall_policy=args.nuc_recall_policy,
                 derived_tf_max_edge_ambiguity=(
                     None if args.ddda_derived_tf_max_edge_gap < 0
                     else args.ddda_derived_tf_max_edge_gap))


if __name__ == '__main__':
    main()
