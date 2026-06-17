#!/usr/bin/env python3
"""
FiberHMM apply_model CLI entry point.
Applies trained HMM to call chromatin footprints from fiber-seq BAM files.
"""

import argparse
import os
import sys

import pandas as pd

from fiberhmm.cli.common import (
    add_edge_trim_args,
    add_filter_args,
    add_mode_args,
    add_parallel_args,
    add_stats_args,
    add_version_args,
    mode_description,
    resolve_core_count,
)
from fiberhmm.cli.model_selection import resolve_model_path as _resolve_cli_model_path
from fiberhmm.core.model_io import load_model_with_metadata
from fiberhmm.inference.parallel import process_bam_for_footprints
from fiberhmm.inference.stats import collect_stats_from_bam
from fiberhmm.io.bam_index import bam_index_exists


def parse_args():
    parser = argparse.ArgumentParser(
        description='Apply FiberHMM model to call chromatin footprints from fiber-seq BAM',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Output:
  Tagged BAM file with footprint annotations (ns/nl and as/al tags).
  Use extract_tags.py to convert to BED12/bigBed for visualization.

Examples:
  # Hia5 PacBio -- bundled model, no -m needed
  fiberhmm-apply -i data.bam --enzyme hia5 --seq pacbio -o output/

  # DddB Nanopore DAF-seq
  fiberhmm-apply -i data.bam --enzyme dddb -o output/

  # DddA amplicons (two-pass: nuc pass then TF recaller)
  fiberhmm-apply -i data.bam --enzyme ddda -o tmp/
  fiberhmm-recall-tfs -i tmp/data_footprints.bam -o recalled.bam --enzyme ddda

  # Override with a custom model
  fiberhmm-apply -i data.bam -m custom.json -o output/ -c 8

  # Extract to bigBed for browser visualization
  fiberhmm-extract-tags -i output/data_footprints.bam --footprint --bigbed
'''
    )

    add_version_args(parser)

    # Required
    parser.add_argument('-i', '--input', required=True,
                        help='Input BAM file with modification calls (must be indexed)')
    parser.add_argument('-m', '--model', default=None,
                        help='Path to trained HMM model (.json, .npz, or .pickle). '
                             'If omitted, the bundled model for --enzyme/--seq is used.')
    parser.add_argument('-o', '--outdir', required=True,
                        help='Output directory, or "-" to write BAM to stdout (for piping)')

    # Enzyme / platform (bundled model selection)
    from fiberhmm.models import SUPPORTED_ENZYMES as _ENZYMES
    parser.add_argument('--enzyme', choices=_ENZYMES, default=None,
                        help='Auto-select bundled model: hia5, dddb, or ddda. '
                             'Use with --seq pacbio|nanopore for Hia5.')
    parser.add_argument('--seq', choices=['pacbio', 'nanopore'], default=None,
                        help='Sequencing platform. Required for hia5 '
                             '(pacbio or nanopore); ignored for dddb/ddda.')

    # Mode
    add_mode_args(parser, default=None)

    # Context size (auto-detected from model by default)
    parser.add_argument('-k', '--context-size', type=int, default=None,
                        help='Context size (auto-detected from model if not specified)')

    # Parallelization
    add_parallel_args(parser, default_cores=1, default_region_size=10_000_000)

    # Filtering
    add_filter_args(parser, min_mapq=0, prob_threshold=128, min_read_length=1000)
    parser.add_argument('-t', '--train-reads', default=None,
                        help='TSV file of read IDs used in training (to exclude)')
    parser.add_argument('-l', '--min-footprints', type=int, default=0,
                        help='Minimum footprints required per read')
    parser.add_argument('--primary', action='store_true',
                        help='Only process primary alignments (skip secondary/supplementary)')
    parser.add_argument('--process-unmapped', action='store_true',
                        help='Process unmapped reads that have sequences and modification tags. '
                             'Enabled automatically in streaming mode when no BAM index exists.')

    # Processing
    add_edge_trim_args(parser, default=10)
    parser.add_argument('-r', '--circular', action='store_true',
                        help='Enable circular mode (tiles reads 3x)')

    # Output format flags
    parser.add_argument('--scores', action='store_true',
                        help='Compute per-footprint confidence scores (slower but more informative)')
    parser.add_argument('--scores-db', action='store_true',
                        help='Output SQLite database with detailed per-footprint scores')
    parser.add_argument('--msp-min-size', type=int, default=0,
                        help='Minimum size for MSP regions in bp. Default 0 '
                             '(emit every accessible run; matches fibertools, '
                             'which does not impose an MSP size filter at this '
                             'stage). Pass a positive value to filter.')
    parser.add_argument('--nuc-min-size', type=int, default=85,
                        help='Minimum footprint size (bp) to count as nucleosome-sized '
                             'for MSP boundary detection. Only footprints >= this size '
                             'split MSPs; smaller footprints are absorbed (default: 85)')
    parser.add_argument('--no-msps', action='store_true',
                        help='Do not write MSP tags (as/al/aq) to output BAM. '
                             'Useful for Fiber-seq where MSPs are computed differently by fibertools')

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


def _resolve_model_path(args):
    """Resolve the apply model path from --model or bundled --enzyme/--seq."""
    return _resolve_cli_model_path(
        args,
        tool='apply',
        bundled_message="Using bundled model: {model_path}",
    )


def _use_streaming_pipeline(input_bam: str, n_cores: int) -> bool:
    return input_bam == '-' or n_cores > 1


def _has_bam_index(input_bam: str) -> bool:
    return bam_index_exists(input_bam)


def _resolve_process_unmapped(args, use_streaming: bool) -> bool:
    process_unmapped = args.process_unmapped
    if use_streaming and not process_unmapped and args.input != '-':
        if not _has_bam_index(args.input):
            process_unmapped = True
            print("Enabling unmapped read processing (no BAM index)")
    return process_unmapped


def _resolve_apply_cores(requested_cores: int) -> int:
    n_cores = resolve_core_count(requested_cores)
    if int(requested_cores) == 0:
        print(f"Auto-detected {n_cores} CPU cores")
    return n_cores


def _dataset_name(input_bam: str) -> str:
    if input_bam == '-':
        return 'stdin'
    return os.path.basename(input_bam).replace('.bam', '')


def _resolve_output_bam(args, dataset: str, stdout_mode: bool) -> str:
    if stdout_mode:
        return '-'
    return os.path.join(args.outdir, f"{dataset}_footprints.bam")


def _resolve_scores_db_path(args, dataset: str):
    if not args.scores_db:
        return None
    return os.path.join(args.outdir, f"{dataset}_scores.db")


def _ddda_notice_needed(model_path: str, enzyme: str = None) -> bool:
    model_basename = os.path.basename(model_path).lower()
    return 'ddda' in model_basename or enzyme == 'ddda'


def _print_ddda_two_pass_notice(model_path: str, enzyme: str = None) -> None:
    if not _ddda_notice_needed(model_path, enzyme):
        return
    print(
        "\n"
        "------------------------------------------------------------------------\n"
        "  NOTE: DddA model detected.\n"
        "  This model calls NUCLEOSOMES only. To recover TF / Pol II\n"
        "  footprints, run the beta 2nd-pass recaller after this step:\n"
        "\n"
        "    fiberhmm-recall-tfs -i <output.bam> -o <recalled.bam> --enzyme ddda\n"
        "\n"
        "  fiberhmm-recall-tfs is a beta feature, first in fiberhmm 2.6.0.\n"
        "------------------------------------------------------------------------\n",
        file=sys.stderr,
    )


def _resolve_context_size(args, model_context_size: int) -> int:
    if args.context_size is not None:
        context_size = args.context_size
        if context_size != model_context_size:
            print(f"  WARNING: Overriding model context size {model_context_size} with {context_size}")
    else:
        context_size = model_context_size
    print(f"  Context size: k={context_size} ({2*context_size + 1}-mer)")
    return context_size


def _resolve_mode(args, model_mode: str) -> str:
    if args.mode is not None:
        mode = args.mode
        if mode != model_mode:
            print(f"  NOTE: Command line mode '{mode}' overrides model mode '{model_mode}'")
    elif model_mode and model_mode != 'unknown':
        mode = model_mode
    else:
        mode = 'pacbio-fiber'
        print(f"  No mode in model metadata, defaulting to '{mode}'")

    print(f"  Mode: {mode}")
    return mode


def _print_processing_settings(
    args,
    mode: str,
    context_size: int,
    n_cores: int,
    msp_min_size: int,
    with_scores: bool,
    db_path: str = None,
) -> None:
    mode_desc = mode_description(mode)

    print(f"\nProcessing: {args.input}")
    print(f"  Mode: {mode} ({mode_desc})")
    print(f"  Context: k={context_size} ({2*context_size + 1}-mer)")
    print(f"  Output: {args.outdir}")
    print(f"  Cores: {n_cores}")
    print(f"  Edge trim: {args.edge_trim} bp")
    print(f"  Min MAPQ: {args.min_mapq}")
    print(f"  Mod prob threshold: {args.prob_threshold}/255")
    if args.circular:
        print("  Circular mode: enabled")
    if with_scores:
        print("  Confidence scores: enabled")
    if args.scores_db:
        print(f"  Scores database: {db_path}")
    if mode == 'daf':
        print("  Strand detection: automatic (C=+, G=-)")
    elif mode == 'nanopore-fiber':
        print("  Strand detection: none (A-centered only)")
    if args.no_msps:
        print("  MSP output: disabled (--no-msps)")
    else:
        print(f"  MSP min size: {msp_min_size} bp")
    if args.stats:
        print("  Stats: enabled")
    print()


def _load_training_read_ids(train_reads):
    if not train_reads:
        return set()

    train_df = pd.read_csv(train_reads, sep='\t')
    train_rids = set(train_df['rid'].tolist())
    print(f"Excluding {len(train_rids)} training reads")
    return train_rids


def _resolve_chroms_set(chroms):
    return set(chroms) if chroms else None


def _print_region_filter_settings(args, chroms_set) -> None:
    if chroms_set:
        print(f"Processing only chromosomes: {', '.join(sorted(chroms_set))}")
    if args.skip_scaffolds:
        print("Skipping scaffold/contig chromosomes")


def main():
    args = parse_args()

    # Handle stdout output mode — redirect all prints to stderr
    # so they don't corrupt the BAM stream
    stdout_mode = (args.outdir == '-')
    if stdout_mode:
        sys.stdout = sys.stderr

    n_cores = _resolve_apply_cores(args.cores)

    # Create output directory (unless writing to stdout)
    if not stdout_mode:
        os.makedirs(args.outdir, exist_ok=True)

    model_path = _resolve_model_path(args)

    # Load model with metadata
    print(f"Loading model from {model_path}")
    model, model_context_size, model_mode = load_model_with_metadata(model_path)
    print("Model loaded successfully")
    print(f"  Start probs: {model.startprob_}")
    print(f"  Transition matrix:\n{model.transmat_}")

    # Surface the DddA two-pass workflow whenever a DddA model is detected.
    # ddda_nuc.json deliberately does NOT emit sub-nucleosomal TF calls;
    # users unaware of fiberhmm-recall-tfs will think their data just has
    # no TFs. Print a prominent notice (stderr so BAM streams stay clean).
    _print_ddda_two_pass_notice(model_path, getattr(args, 'enzyme', None))

    # Show optimization status
    from fiberhmm.core.hmm import HAS_NUMBA
    if HAS_NUMBA:
        print("  Numba JIT: enabled (fast)")
    else:
        print("  Numba JIT: disabled (pip install numba for ~10x speedup)")

    # Determine context size and mode (command line overrides model metadata)
    context_size = _resolve_context_size(args, model_context_size)
    mode = _resolve_mode(args, model_mode)
    args.mode = mode

    # Determine MSP minimum size (default 60bp for all modes)
    msp_min_size = args.msp_min_size if args.msp_min_size is not None else 0

    # Load training read IDs to exclude
    train_rids = _load_training_read_ids(args.train_reads)

    dataset = _dataset_name(args.input)

    # Determine if we need scores
    with_scores = args.scores or args.scores_db

    db_path = _resolve_scores_db_path(args, dataset)

    _print_processing_settings(
        args, mode, context_size, n_cores, msp_min_size, with_scores, db_path,
    )

    # Parse chromosomes
    chroms_set = _resolve_chroms_set(args.chroms)
    _print_region_filter_settings(args, chroms_set)

    # Mode selection:
    #   n_cores > 1 or stdin → streaming pipeline
    #   n_cores == 1 → single-threaded chunk mode
    use_streaming = _use_streaming_pipeline(args.input, n_cores)
    if args.input == '-':
        print("Reading from stdin, using streaming pipeline mode")

    # Auto-detect process_unmapped: enable when streaming without an index
    process_unmapped = _resolve_process_unmapped(args, use_streaming)

    # === MAIN PROCESSING ===
    output_bam = _resolve_output_bam(args, dataset, stdout_mode)
    if args.max_reads:
        print(f"Processing BAM (limited to {args.max_reads:,} reads)...")
    else:
        print("Processing BAM...")

    total_reads, reads_with_footprints = process_bam_for_footprints(
        input_bam=args.input,
        output_bam=output_bam,
        model_or_path=model_path,
        train_rids=train_rids,
        edge_trim=args.edge_trim,
        circular=args.circular,
        mode=mode,
        context_size=context_size,
        msp_min_size=msp_min_size,
        nuc_min_size=args.nuc_min_size,
        min_mapq=args.min_mapq,
        prob_threshold=args.prob_threshold,
        min_read_length=args.min_read_length,
        with_scores=with_scores,
        n_cores=n_cores,
        max_reads=args.max_reads,
        debug_timing=args.debug_timing,
        region_parallel=False,
        region_size=args.region_size,
        skip_scaffolds=args.skip_scaffolds,
        chroms=chroms_set,
        primary_only=args.primary,
        output_posteriors=args.output_posteriors,
        write_msps=not args.no_msps,
        io_threads=args.io_threads,
        streaming_pipeline=use_streaming,
        chunk_size=args.chunk_size,
        process_unmapped=process_unmapped,
    )
    print(f"\nProcessed {total_reads:,} reads -> {reads_with_footprints:,} with footprints",
          file=sys.stderr if stdout_mode else sys.stdout)
    if not stdout_mode:
        print(f"BAM: {output_bam}")
        print(f"BAM index: {output_bam}.bai")

    # Generate stats if requested (not available for stdout mode)
    if args.stats and not stdout_mode:
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

    if stdout_mode:
        print("\nDone!", file=sys.stderr)
    else:
        print("\nDone!")
        print("\nTo extract BED12/bigBed for browser visualization:")
        print(f"  fiberhmm-extract-tags -i {output_bam}")


if __name__ == '__main__':
    main()
