#!/usr/bin/env python3
"""
FiberHMM apply_model CLI entry point.
Applies trained HMM to call chromatin footprints from fiber-seq BAM files.
"""

import os
import sys
import glob
import shutil
import tempfile
import argparse
import numpy as np
import pandas as pd

from fiberhmm.core.model_io import load_model_with_metadata
from fiberhmm.inference.engine import detect_mode_from_bam
from fiberhmm.inference.parallel import process_bam_for_footprints
from fiberhmm.inference.stats import FootprintStats, collect_stats_from_bam
from fiberhmm.cli.common import (
    add_mode_args, add_filter_args, add_edge_trim_args,
    add_parallel_args, add_stats_args, add_version_args,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Apply FiberHMM model to call chromatin footprints from fiber-seq BAM',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Output:
  Tagged BAM file with footprint annotations (ns/nl and as/al tags).
  Use extract_tags.py to convert to BED12/bigBed for visualization.

Examples:
  # Basic (BAM output)
  fiberhmm-apply -i data.bam -m model.json -o output/

  # With QC stats
  fiberhmm-apply -i data.bam -m model.json -o output/ --stats

  # Multi-core processing
  fiberhmm-apply -i data.bam -m model.json -o output/ -c 8

  # Extract to bigBed for browser visualization
  fiberhmm-extract-tags -i output/data_footprints.bam --footprint --bigbed
'''
    )

    add_version_args(parser)

    # Required
    parser.add_argument('-i', '--input', required=True,
                        help='Input BAM file with modification calls (must be indexed)')
    parser.add_argument('-m', '--model', required=True,
                        help='Path to trained HMM model (.json, .npz, or .pickle)')
    parser.add_argument('-o', '--outdir', required=True,
                        help='Output directory')

    # Mode
    add_mode_args(parser, default=None)

    # Context size (auto-detected from model by default)
    parser.add_argument('-k', '--context-size', type=int, default=None,
                        help='Context size (auto-detected from model if not specified)')

    # Parallelization
    add_parallel_args(parser, default_cores=1, default_region_size=10_000_000)

    # Filtering
    add_filter_args(parser, min_mapq=20, prob_threshold=128, min_read_length=1000)
    parser.add_argument('-t', '--train-reads', default=None,
                        help='TSV file of read IDs used in training (to exclude)')
    parser.add_argument('-l', '--min-footprints', type=int, default=0,
                        help='Minimum footprints required per read')
    parser.add_argument('--primary', action='store_true',
                        help='Only process primary alignments (skip secondary/supplementary)')

    # Processing
    add_edge_trim_args(parser, default=10)
    parser.add_argument('-r', '--circular', action='store_true',
                        help='Enable circular mode (tiles reads 3x)')

    # Output format flags
    parser.add_argument('--scores', action='store_true',
                        help='Compute per-footprint confidence scores (slower but more informative)')
    parser.add_argument('--scores-db', action='store_true',
                        help='Output SQLite database with detailed per-footprint scores')
    parser.add_argument('--msp-min-size', type=int, default=None,
                        help='Minimum size for MSP regions (default: 60)')

    # QC and statistics
    add_stats_args(parser)
    parser.add_argument('--stats-sample', type=int, default=10000,
                        help='Number of reads to sample for statistics (default: 10000)')
    parser.add_argument('--stats-seed', type=int, default=42,
                        help='Random seed for sampling (default: 42)')

    # Posteriors and debug
    parser.add_argument('--output-posteriors', type=str, default=None,
                        help='Export HMM posteriors to file (H5 or TSV)')
    parser.add_argument('--debug-timing', action='store_true',
                        help='Show per-read timing breakdown')

    # Testing
    parser.add_argument('--max-reads', type=int, default=None,
                        help=argparse.SUPPRESS)

    return parser.parse_args()


def main():
    args = parse_args()

    # Determine number of cores
    if args.cores == 0:
        import multiprocessing
        n_cores = multiprocessing.cpu_count()
        print(f"Auto-detected {n_cores} CPU cores")
    else:
        n_cores = args.cores

    # Create output directory
    os.makedirs(args.outdir, exist_ok=True)

    # Load model with metadata
    print(f"Loading model from {args.model}")
    model, model_context_size, model_mode = load_model_with_metadata(args.model)
    print(f"Model loaded successfully")
    print(f"  Start probs: {model.startprob_}")
    print(f"  Transition matrix:\n{model.transmat_}")

    # Show optimization status
    from fiberhmm.core.hmm import HAS_NUMBA
    if HAS_NUMBA:
        print(f"  Numba JIT: enabled (fast)")
    else:
        print(f"  Numba JIT: disabled (pip install numba for ~10x speedup)")

    # Determine context size (command line overrides model)
    if args.context_size is not None:
        context_size = args.context_size
        if context_size != model_context_size:
            print(f"  WARNING: Overriding model context size {model_context_size} with {context_size}")
    else:
        context_size = model_context_size
    print(f"  Context size: k={context_size} ({2*context_size + 1}-mer)")

    # Auto-detect mode from BAM MM tags if not specified
    detected_mode = detect_mode_from_bam(args.input)
    print(f"  Auto-detected mode from BAM: {detected_mode}")

    # Determine final mode (priority: command line > auto-detect > model)
    if args.mode is not None:
        mode = args.mode
        if mode != model_mode:
            print(f"  NOTE: Command line mode '{mode}' overrides model mode '{model_mode}'")
        if mode != detected_mode and detected_mode != 'unknown':
            print(f"  WARNING: Command line mode '{mode}' differs from auto-detected '{detected_mode}'")
    elif detected_mode != 'unknown' and detected_mode != model_mode:
        mode = detected_mode
        print(f"  WARNING: Model mode is '{model_mode}' but BAM appears to be '{detected_mode}'")
        print(f"           Using auto-detected mode '{mode}'. Use --mode to override.")
    else:
        mode = model_mode

    print(f"  Final mode: {mode}")
    args.mode = mode

    # Determine MSP minimum size (default 60bp for all modes)
    msp_min_size = args.msp_min_size if args.msp_min_size is not None else 60

    # Load training read IDs to exclude
    train_rids = set()
    if args.train_reads:
        train_df = pd.read_csv(args.train_reads, sep='\t')
        train_rids = set(train_df['rid'].tolist())
        print(f"Excluding {len(train_rids)} training reads")

    # Get dataset name
    dataset = os.path.basename(args.input).replace('.bam', '')

    # Determine if we need scores
    with_scores = args.scores or args.scores_db

    # Setup scores database path if needed
    db_path = None
    if args.scores_db:
        db_path = os.path.join(args.outdir, f"{dataset}_scores.db")

    # Print settings
    mode_descs = {
        'pacbio-fiber': 'PacBio fiber-seq (A-centered)',
        'nanopore-fiber': 'Nanopore fiber-seq (A-centered)',
        'daf': 'DAF-seq deamination (C/G-centered)'
    }
    mode_desc = mode_descs.get(mode, mode)

    print(f"\nProcessing: {args.input}")
    print(f"  Mode: {mode} ({mode_desc})")
    print(f"  Context: k={context_size} ({2*context_size + 1}-mer)")
    print(f"  Output: {args.outdir}")
    print(f"  Cores: {n_cores}")
    print(f"  Edge trim: {args.edge_trim} bp")
    print(f"  Min MAPQ: {args.min_mapq}")
    print(f"  Mod prob threshold: {args.prob_threshold}/255")
    if args.circular:
        print(f"  Circular mode: enabled")
    if with_scores:
        print(f"  Confidence scores: enabled")
    if args.scores_db:
        print(f"  Scores database: {db_path}")
    if mode == 'daf':
        print(f"  Strand detection: automatic (C=+, G=-)")
    elif mode == 'nanopore-fiber':
        print(f"  Strand detection: none (A-centered only)")
    print(f"  MSP min size: {msp_min_size} bp")
    if args.stats:
        print(f"  Stats: enabled")
    print()

    # Parse chromosomes
    chroms_set = set(args.chroms) if args.chroms else None
    if chroms_set:
        print(f"Processing only chromosomes: {', '.join(sorted(chroms_set))}")
    if args.skip_scaffolds:
        print("Skipping scaffold/contig chromosomes")

    # Region-parallel is always used when cores > 1
    use_region_parallel = n_cores > 1

    # === MAIN PROCESSING ===
    output_bam = os.path.join(args.outdir, f"{dataset}_footprints.bam")
    if args.max_reads:
        print(f"Processing BAM (limited to {args.max_reads:,} reads)...")
    else:
        print("Processing BAM...")

    total_reads, reads_with_footprints = process_bam_for_footprints(
        input_bam=args.input,
        output_bam=output_bam,
        model_or_path=args.model,
        train_rids=train_rids,
        edge_trim=args.edge_trim,
        circular=args.circular,
        mode=mode,
        context_size=context_size,
        msp_min_size=msp_min_size,
        min_mapq=args.min_mapq,
        prob_threshold=args.prob_threshold,
        min_read_length=args.min_read_length,
        with_scores=with_scores,
        n_cores=n_cores,
        max_reads=args.max_reads,
        debug_timing=args.debug_timing,
        region_parallel=use_region_parallel,
        region_size=args.region_size,
        skip_scaffolds=args.skip_scaffolds,
        chroms=chroms_set,
        primary_only=args.primary,
        output_posteriors=args.output_posteriors,
    )
    print(f"\nProcessed {total_reads:,} reads -> {reads_with_footprints:,} with footprints")
    print(f"BAM: {output_bam}")
    print(f"BAM index: {output_bam}.bai")

    # Generate stats if requested
    if args.stats:
        print("\nGenerating statistics...")
        stats_prefix = os.path.join(args.outdir, f"{dataset}_footprints")
        stats = collect_stats_from_bam(output_bam,
                                       n_samples=args.stats_sample,
                                       seed=args.stats_seed,
                                       with_scores=with_scores)
        stats.write_summary(f"{stats_prefix}_stats.txt")
        stats.plot_distributions(stats_prefix)
        print(f"Stats: {stats_prefix}_stats.txt, {stats_prefix}_stats.pdf")

    # Print scores database info
    if db_path and os.path.exists(db_path):
        import sqlite3
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM reads")
        n_reads = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM footprints")
        n_footprints = cursor.fetchone()[0]
        conn.close()
        print(f"Scores DB: {db_path}")
        print(f"  {n_reads:,} reads, {n_footprints:,} footprints")

    print("\nDone!")
    print(f"\nTo extract BED12/bigBed for browser visualization:")
    print(f"  fiberhmm-extract-tags -i {output_bam}")


if __name__ == '__main__':
    main()
