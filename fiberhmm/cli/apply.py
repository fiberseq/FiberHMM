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
    add_force_seq_arg,
    add_legacy_mode_override,
    add_parallel_args,
    add_stats_args,
    add_version_args,
    refuse_model_enzyme_assay_conflict,
    require_model_files,
    resolve_observation_mode,
    resolve_platform_argument,
)
from fiberhmm.core.model_io import (
    ModelContextError,
    load_model_with_metadata,
    validate_context_size,
)
from fiberhmm.inference.parallel import process_bam_for_footprints
from fiberhmm.inference.stats import collect_stats_from_bam


def parse_args():
    parser = argparse.ArgumentParser(
        description='Apply FiberHMM model to call chromatin footprints from fiber-seq BAM',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Output:
  Tagged BAM file with footprint annotations (ns/nl and as/al tags).
  Use fiberhmm-extract to convert to BED12/bigBed for visualization.
  fiberhmm-call (apply + nucleosome/TF recall in one pass) is the
  recommended entry point; fiberhmm-apply runs the HMM stage alone.

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
  fiberhmm-extract -i output/data_footprints.bam --nucleosome --msp
'''
    )

    add_version_args(parser)

    # Required
    parser.add_argument('-i', '--input', required=True,
                        help='Input BAM file with modification calls, or "-" for stdin '
                             '(unaligned and unindexed BAMs are streamed)')
    parser.add_argument('-m', '--model', default=None,
                        help='Path to trained HMM model (.json, .npz, or .pickle). '
                             'If omitted, the bundled model for --enzyme/--seq is used.')
    parser.add_argument('-o', '--outdir', required=True,
                        help='Output directory, or "-" to write BAM to stdout (for piping)')

    # Enzyme / platform (bundled model selection)
    from fiberhmm.models import SUPPORTED_ENZYMES as _ENZYMES
    parser.add_argument('--enzyme', choices=_ENZYMES, default=None,
                        help='Auto-select a supported bundled chemistry model. Use '
                             '--seq pacbio|nanopore for Hia5.')
    parser.add_argument('--seq', choices=['pacbio', 'nanopore'], default=None,
                        help='Hia5 sequencing platform. When omitted it is '
                             'detected from the input (MM specs: PacBio T-a vs '
                             'Nanopore A+a only; header records); conflicting '
                             'evidence stops the run, and a given --seq that the '
                             'reads contradict is refused (see --force-seq). '
                             'Ignored for dddb/ddda.')
    add_force_seq_arg(parser)

    # Backward-compatible escape hatch; normal workflows infer this.
    add_legacy_mode_override(parser)

    # Context size (auto-detected from model by default)
    parser.add_argument('-k', '--context-size', type=int, default=None,
                        help='Context size (auto-detected from model if not specified)')

    # Parallelization. --region-size/--skip-scaffolds/--chroms come from the
    # shared factory but have no apply implementation (apply streams the whole
    # BAM); main() rejects them instead of silently ignoring them. Use
    # fiberhmm-call --region-parallel for region selection.
    add_parallel_args(parser, default_cores=1, default_region_size=10_000_000)
    for action in parser._actions:
        if action.dest in ('region_size', 'skip_scaffolds', 'chroms'):
            action.help = argparse.SUPPRESS

    # Filtering
    add_filter_args(parser, min_mapq=0, prob_threshold=None, min_read_length=1000)
    parser.add_argument('-t', '--train-reads', default=None,
                        help='TSV file of read IDs used in training (to exclude)')
    # Parsed-but-never-implemented options, kept only so old command lines get
    # a clear error instead of an argparse "unrecognized argument".
    parser.add_argument('-l', '--min-footprints', type=int, default=0,
                        help=argparse.SUPPRESS)
    from fiberhmm.inference.read_filters import add_alignment_args
    add_alignment_args(parser)
    parser.add_argument('--process-unmapped', action=argparse.BooleanOptionalAction,
                        default=None,
                        help='Process unmapped reads that have sequences and modification tags. '
                             'Default: automatic -- on for stdin, unindexed and unaligned '
                             '(uBAM) input. A run that skips >90%% of records as unmapped '
                             'fails unless --no-process-unmapped is given.')

    # Processing
    add_edge_trim_args(parser, default=10)
    parser.add_argument('-r', '--circular', action='store_true',
                        help='Enable circular mode (tiles reads 3x)')

    # Output format flags
    parser.add_argument('--scores', action='store_true',
                        help='Compute per-footprint confidence scores (slower but more informative)')
    parser.add_argument('--scores-db', action='store_true',
                        help=argparse.SUPPRESS)
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
    from fiberhmm.core.bam_reader import add_daf_run_mask_arguments
    add_daf_run_mask_arguments(parser)

    return parser.parse_args()


def _reject_unimplemented_options(args):
    """Exit on options apply accepted but never implemented."""
    problems = []
    if args.chroms:
        problems.append("--chroms")
    if args.skip_scaffolds:
        problems.append("--skip-scaffolds")
    if args.region_size != 10_000_000:
        problems.append("--region-size")
    if problems:
        print(
            f"error: {', '.join(problems)} had no effect in fiberhmm-apply "
            "(it always streams the whole BAM) and are no longer accepted. "
            "Use fiberhmm-call --region-parallel for region selection, or "
            "subset the BAM with samtools view first.",
            file=sys.stderr,
        )
        sys.exit(2)
    if args.scores_db:
        print("error: --scores-db was never implemented (no database was "
              "written); use --scores and fiberhmm-extract.", file=sys.stderr)
        sys.exit(2)
    if args.min_footprints:
        print("error: -l/--min-footprints was never implemented (reads were "
              "never filtered by footprint count); filter downstream instead.",
              file=sys.stderr)
        sys.exit(2)


def _input_index_state(path):
    """(indexed, aligned) for a file input; (False, False) if unreadable."""
    import pysam

    try:
        with pysam.AlignmentFile(path, 'rb', check_sq=False) as bam:
            return bool(bam.has_index()), bool(bam.references)
    except (OSError, ValueError):
        return False, False


def _resolve_apply_prob_threshold(args):
    """Explicit --prob-threshold, else the chemistry's default threshold."""
    from fiberhmm.cli.common import resolve_input_prob_threshold

    return resolve_input_prob_threshold(
        args.prob_threshold, args.enzyme, args.seq, args.input)


def _apply_pg_record(args, mode, context_size, chemistry, daf_run_mask):
    """@PG record plus chemistry declaration for the output header.

    The same provenance ``fiberhmm-call`` writes (via the shared
    :func:`fiberhmm.cli.provenance.output_header_with_provenance`), so apply
    output can be re-called or recalled without restating ``--enzyme``.
    ``chemistry`` is the input-reconciled declaration; the defaults it implies
    are already resolved, so a stdin header may not change the enzyme late.
    """
    import fiberhmm
    from fiberhmm.cli.provenance import DEFAULTS_RESOLVED_KEY

    min_run, policy = daf_run_mask if mode == 'daf' else (0, None)
    return {
        'PN': 'fiberhmm-apply',
        'VN': getattr(fiberhmm, '__version__', 'unknown'),
        'CL': ' '.join(sys.argv),
        'chemistry': chemistry,
        DEFAULTS_RESOLVED_KEY: True,
        # Keep the literal `coord=molecular` token: downstream tools detect
        # the molecular frame of ns/nl/as/al from it.
        'DS': (f"FiberHMM apply; coord=molecular (ns/nl/as/al in molecular "
               f"original-fiber coordinates); mode={mode} "
               f"enzyme={args.enzyme or 'custom'} k={context_size} "
               f"prob_threshold={args.prob_threshold} "
               f"primary_only={'on' if args.alignments == 'primary' else 'off'} alignments={args.alignments} "
               f"daf_run_mask={f'>={min_run}/{policy}' if min_run else 'off'}"),
    }


def main():
    args = parse_args()
    from fiberhmm.cli.provenance import ChemistryConflictError
    from fiberhmm.inference.read_filters import MostlyUnmappedError
    from fiberhmm.inference.worker_results import WorkerFailureError

    try:
        _main(args)
    except ChemistryConflictError as exc:
        # apply has no --replace-chemistry: its output keeps the input's
        # declaration, so point at the tool that can re-declare it.
        message = str(exc).replace(
            ", or --replace-chemistry to re-declare the output deliberately.",
            ". fiberhmm-apply keeps the input's declaration; use "
            "fiberhmm-call --replace-chemistry to re-declare it deliberately.",
        ).replace(
            ", or --replace-chemistry to declare the run as custom.",
            " (fiberhmm-apply cannot re-declare the input's chemistry).",
        )
        print(f"error: {message}", file=sys.stderr)
        sys.exit(2)
    except (WorkerFailureError, MostlyUnmappedError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)


def _main(args):
    _reject_unimplemented_options(args)
    using_bundled_model = args.model is None

    # Handle stdout output mode — redirect all prints to stderr
    # so they don't corrupt the BAM stream
    stdout_mode = (args.outdir == '-')
    if stdout_mode:
        sys.stdout = sys.stderr

    # Determine number of cores
    if args.cores == 0:
        import multiprocessing
        n_cores = multiprocessing.cpu_count()
        print(f"Auto-detected {n_cores} CPU cores")
    else:
        n_cores = args.cores

    # Create output directory (unless writing to stdout)
    if not stdout_mode:
        os.makedirs(args.outdir, exist_ok=True)

    # A missing --seq is inferred from the input's own evidence (and refused
    # on conflicting evidence) before the bundled model is chosen.
    resolve_platform_argument(args, args.input, tool='fiberhmm-apply')
    require_model_files('fiberhmm-apply', ('-m/--model', args.model))

    # Resolve model path: explicit -m wins; else use bundled model for --enzyme
    model_path = args.model
    if model_path is None:
        if args.enzyme is None:
            print(
                "error: one of --model or --enzyme must be provided.\n"
                "  Use --enzyme hia5/dddb/ddda to pick a bundled model, or\n"
                "  use --model /path/to/model.json for a custom model.",
                file=sys.stderr,
            )
            sys.exit(1)
        from fiberhmm.models import get_model_path as _get_bundled
        try:
            model_path = _get_bundled(args.enzyme, tool='apply', seq=args.seq)
        except (KeyError, FileNotFoundError) as e:
            print(f"error: {e}", file=sys.stderr)
            sys.exit(1)
        print(f"Using bundled model: {model_path}")

    # Load model with metadata
    print(f"Loading model from {model_path}")
    model, model_context_size, model_mode = load_model_with_metadata(model_path)
    print("Model loaded successfully")
    print(f"  Start probs: {model.startprob_}")
    print(f"  Transition matrix:\n{model.transmat_}")

    # Show optimization status
    from fiberhmm.core.hmm import HAS_NUMBA
    if HAS_NUMBA:
        print("  Numba JIT: enabled (fast)")
    else:
        print("  Numba JIT: disabled (pip install numba for ~10x speedup)")

    # Determine context size (command line overrides model)
    if args.context_size is not None:
        context_size = args.context_size
        if context_size != model_context_size:
            print(f"  WARNING: Overriding model context size {model_context_size} with {context_size}")
    else:
        context_size = model_context_size
    try:
        validate_context_size(model, context_size, label=f"model {model_path}")
    except ModelContextError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(2)
    print(f"  Context size: k={context_size} ({2*context_size + 1}-mer)")

    # Bundled workflows infer mode from enzyme/platform; custom models use
    # metadata. The hidden legacy --mode remains the highest-priority override.
    from fiberhmm.models import get_metadata_mode_aliases, get_observation_mode
    inferred_mode = (
        get_observation_mode(
            args.enzyme, args.seq, warn_missing_seq=False
        )
        if using_bundled_model else None
    )
    metadata_mode_aliases = (
        get_metadata_mode_aliases(
            args.enzyme, args.seq, warn_missing_seq=False
        )
        if using_bundled_model else ()
    )
    try:
        mode = resolve_observation_mode(
            model_mode,
            inferred_mode=inferred_mode,
            explicit_mode=args.mode,
            source_label=(
                f"bundled {args.enzyme} model"
                if using_bundled_model else "custom model"
            ),
            metadata_mode_aliases=metadata_mode_aliases,
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(2)
    if not using_bundled_model:
        refuse_model_enzyme_assay_conflict(
            model_mode, args.enzyme, args.seq, args.model,
            tool='fiberhmm-apply', explicit_mode=args.mode)

    print(f"  Mode: {mode}")
    args.mode = mode

    # Chemistry: reconcile with the input's declaration BEFORE any
    # enzyme-dependent default (DAF run mask, ML threshold) is chosen, as
    # fiberhmm-call does. A custom -m without --enzyme inherits the input's
    # supported enzyme/platform when the observation mode matches, so it gets
    # exactly the defaults of --enzyme <inherited> with the given table; a
    # conflicting declaration is refused (the output keeps the input header's
    # declaration, which would then misdescribe these calls). A stdin header
    # cannot be read in advance: no inheritance there.
    from fiberhmm.cli.provenance import resolve_effective_chemistry
    input_header = None
    if args.input != '-':
        import pysam
        with pysam.AlignmentFile(args.input, 'rb', check_sq=False) as _bam:
            input_header = _bam.header
    run_chemistry = resolve_effective_chemistry(
        args, mode, input_header, model_path, None, tool='fiberhmm-apply',
    )

    from fiberhmm.core.bam_reader import apply_daf_run_mask_arguments
    try:
        daf_run_mask = apply_daf_run_mask_arguments(args, args.enzyme)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(2)

    # Surface the DddA two-pass workflow whenever a DddA model is detected.
    # ddda_nuc.json deliberately does NOT emit sub-nucleosomal TF calls;
    # users unaware of fiberhmm-recall-tfs will think their data just has
    # no TFs. Print a prominent notice (stderr so BAM streams stay clean).
    _model_basename = os.path.basename(model_path).lower()
    if 'ddda' in _model_basename or getattr(args, 'enzyme', None) == 'ddda':
        import sys as _sys
        print(
            "\n"
            "------------------------------------------------------------------------\n"
            "  NOTE: DddA model detected.\n"
            "  This model calls NUCLEOSOMES only. To recover TF / Pol II\n"
            "  footprints, run the 2nd-pass recaller after this step (or use\n"
            "  fiberhmm-call --enzyme ddda, which runs both passes):\n"
            "\n"
            "    fiberhmm-recall-tfs -i <output.bam> -o <recalled.bam> --enzyme ddda\n"
            "------------------------------------------------------------------------\n",
            file=_sys.stderr,
        )

    # ML threshold: explicit value, else the chemistry preset (Hia5 Nanopore
    # 248, otherwise 128). A custom -m on an input declaring an unsupported
    # chemistry still takes that declaration's threshold.
    args.prob_threshold = _resolve_apply_prob_threshold(args)
    pg_record = _apply_pg_record(args, mode, context_size, run_chemistry,
                                 daf_run_mask)

    # Determine MSP minimum size (default 60bp for all modes)
    msp_min_size = args.msp_min_size if args.msp_min_size is not None else 0

    # Load training read IDs to exclude
    train_rids = set()
    if args.train_reads:
        train_df = pd.read_csv(args.train_reads, sep='\t')
        train_rids = set(train_df['rid'].tolist())
        print(f"Excluding {len(train_rids)} training reads")

    # Get dataset name
    if args.input == '-':
        dataset = 'stdin'
    else:
        dataset = os.path.basename(args.input).replace('.bam', '')

    # Determine if we need scores
    with_scores = args.scores

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
        print("  Circular mode: enabled")
    if with_scores:
        print("  Confidence scores: enabled")
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

    # Unmapped reads are called automatically for stdin, unindexed and
    # unaligned (uBAM) input; an explicit --[no-]process-unmapped wins.
    process_unmapped = args.process_unmapped
    if process_unmapped is None:
        if args.input == '-':
            process_unmapped = True
            reason = "stdin input"
        else:
            indexed, aligned = _input_index_state(args.input)
            process_unmapped = not (indexed and aligned)
            reason = ("unaligned input (no @SQ reference sequences)"
                      if not aligned else "no BAM index")
        if process_unmapped:
            print(f"Enabling unmapped read processing ({reason})")

    # Mode selection:
    #   --streaming, stdin, n_cores > 1 or unmapped processing → streaming
    #   pipeline (the only path that calls unmapped reads)
    #   otherwise (n_cores == 1) → single-threaded chunk mode
    use_streaming = bool(
        args.streaming or args.input == '-' or n_cores > 1 or process_unmapped
    )
    if args.input == '-':
        print("Reading from stdin, using streaming pipeline mode")

    # === MAIN PROCESSING ===
    if stdout_mode:
        output_bam = '-'
    else:
        output_bam = os.path.join(args.outdir, f"{dataset}_footprints.bam")
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
        primary_only=args.alignments,
        output_posteriors=args.output_posteriors,
        write_msps=not args.no_msps,
        io_threads=args.io_threads,
        streaming_pipeline=use_streaming,
        chunk_size=args.chunk_size,
        process_unmapped=process_unmapped,
        # A run that skipped nearly everything as unmapped is an error unless
        # the user asked for pass-through explicitly.
        fail_on_mostly_unmapped=args.process_unmapped is not False,
        pg_record=pg_record,
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

    if stdout_mode:
        print("\nDone!", file=sys.stderr)
    else:
        print("\nDone!")
        print("\nTo extract BED12/bigBed for browser visualization:")
        print(f"  fiberhmm-extract -i {output_bam}")


if __name__ == '__main__':
    main()
