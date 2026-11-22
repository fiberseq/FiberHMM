#!/usr/bin/env python3
"""
fiberhmm-apply -- deprecated alias of ``fiberhmm-call --hmm-only``.

FiberHMM 3.3 folds the HMM-only step into ``fiberhmm-call --hmm-only``; this
command maps its options onto that call and runs it, so existing scripts
keep working for one more release. It is removed in the release after 3.3.

Option mapping (everything else is passed through unchanged):

* ``-o DIR``         -> ``-o DIR/<input name>_footprints.bam`` (``-o -`` stays stdout)
* ``--no-msps``      -> ``--keep-msps``
* ``-t/--train-reads`` -> ``--train-reads``
* ``--scores``       -> ``--scores`` (the HMM posterior-mean ``nq`` of nucleosomes)
* ``--stats``        -> fiberhmm-call's QC (``qc/`` beside the output);
  ``--stats-sample``/``--stats-seed`` -> ``--qc-sample-reads``/``--qc-seed``.
  Without ``--stats`` the alias passes ``--no-qc`` (apply wrote no report).
* ``--harmonize-legacy-tags`` -> ``--legacy-tags`` (whose ns/nl are the MA
  nucleosomes: the harmonized view); warned
* ``--output-posteriors`` / ``--debug-timing`` -> ignored with a warning
  (use ``fiberhmm-posteriors``; per-read timing was never printed)
* ``--streaming``    -> ignored (fiberhmm-call always streams without
  ``--region-parallel``)
"""

import argparse
import os
import sys

from fiberhmm import defaults as _D
from fiberhmm.cli.common import (
    add_edge_trim_args,
    add_filter_args,
    add_force_seq_arg,
    add_legacy_mode_override,
    add_legacy_tags_args,
    add_parallel_args,
    add_stats_args,
    add_version_args,
)

DEPRECATION_NOTICE = (
    "fiberhmm-apply is deprecated and will be removed in the release after "
    "3.3: running fiberhmm-call --hmm-only (same calls; see "
    "docs/workflows/calling.md).")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='Deprecated alias of fiberhmm-call --hmm-only (removed in the '
                    'release after 3.3): HMM footprint calls only, no recall.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
fiberhmm-apply runs fiberhmm-call --hmm-only with the options mapped (the
command it runs is printed and recorded in the output @PG). Output: an MA/AQ
tagged BAM with every HMM call -- footprints >= --nuc-min-size bp as nuc,
smaller footprints as tf, the HMM MSPs as msp -- ready for
fiberhmm-recall-nucs / fiberhmm-recall-tfs.

Use instead:
  fiberhmm-call -i data.bam -o out/data_footprints.bam --enzyme hia5 --seq pacbio --hmm-only
  fiberhmm-call -i data.bam -o calls.bam --enzyme hia5 --seq pacbio   # HMM + recall
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
    add_filter_args(parser, min_mapq=_D.MIN_MAPQ, prob_threshold=None,
                    min_read_length=_D.MIN_READ_LENGTH)
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
    add_edge_trim_args(parser, default=_D.EDGE_TRIM)
    parser.add_argument('-r', '--circular', action='store_true',
                        help='Enable circular mode (tiles reads 3x)')

    # Output format flags
    parser.add_argument('--scores', action='store_true',
                        help='fiberhmm-call --scores: the HMM posterior-mean nq '
                             'byte of each nucleosome (nuc.Q). No MSP score.')
    parser.add_argument('--scores-db', action='store_true',
                        help=argparse.SUPPRESS)
    parser.add_argument('--msp-min-size', type=int, default=_D.MSP_MIN_SIZE,
                        help='Minimum size for MSP regions in bp. Default 0 '
                             '(emit every accessible run; matches fibertools, '
                             'which does not impose an MSP size filter at this '
                             'stage). Pass a positive value to filter.')
    parser.add_argument('--nuc-min-size', type=int, default=_D.NUC_MIN_SIZE,
                        help='Minimum footprint size (bp) to count as nucleosome-sized '
                             'for MSP boundary detection. Only footprints >= this size '
                             'split MSPs; smaller footprints are absorbed (default: 85)')
    parser.add_argument('--no-msps', action='store_true',
                        help='fiberhmm-call --keep-msps: keep the input\'s own MSPs '
                             '(and fibertools FIRE) instead of writing the HMM\'s.')

    # QC and statistics
    add_stats_args(parser)
    parser.add_argument('--stats-sample', type=int, default=None,
                        help='fiberhmm-call --qc-sample-reads (with --stats).')
    parser.add_argument('--stats-seed', type=int, default=None,
                        help='fiberhmm-call --qc-seed (with --stats).')

    # Posteriors and debug
    parser.add_argument('--output-posteriors', type=str, default=None,
                        help='Ignored with a warning: export posteriors with '
                             'fiberhmm-posteriors.')
    parser.add_argument('--debug-timing', action='store_true',
                        help=argparse.SUPPRESS)

    # Testing
    parser.add_argument('--max-reads', type=int, default=None,
                        help=argparse.SUPPRESS)
    from fiberhmm.core.bam_reader import add_daf_run_mask_arguments
    add_daf_run_mask_arguments(parser)
    parser.add_argument('--daf-mask-unaligned', action=argparse.BooleanOptionalAction,
                        default=True,
                        help='DAF only: treat insertion and soft-clip bases as no '
                             'evidence and leave unaligned stretches of >= 50 bp '
                             'uncalled (default on, as in fiberhmm-call).')
    parser.add_argument('--use-m5c', action=argparse.BooleanOptionalAction,
                        default=None,
                        help='DddA CpG-aware calling, as in fiberhmm-call: CpG '
                             'observations are hidden from the HMM except inside '
                             'confident unmethylated islands (MA ddda_ucg from '
                             'fiberhmm-tag-m5c) the input carries. Default: on '
                             'for --enzyme ddda, off otherwise.')
    parser.add_argument('--cpg-mask-policy',
                        choices=('unmethylated-only', 'methylated-only'),
                        default=None,
                        help='With CpG-aware calling: keep CpGs only inside '
                             'ddda_ucg islands (default unmethylated-only), or '
                             'mask only ddda_mcg spans (methylated-only). DddA '
                             'only; ignored with a warning otherwise.')
    add_legacy_tags_args(parser)
    parser.add_argument('--harmonize-legacy-tags', action='store_true',
                        help='Deprecated: --legacy-tags now writes the harmonized '
                             'view (ns/nl = the MA nucleosomes, as/al = MSPs); '
                             'mapped to --legacy-tags with a warning.')

    return parser.parse_args(argv)


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


def _dataset_name(path):
    """apply's output stem: the input file name without ``.bam``."""
    if path == '-':
        return 'stdin'
    return os.path.basename(path).replace('.bam', '')


def call_argv(args):
    """``(argv, warnings)``: the ``fiberhmm-call --hmm-only`` arguments an
    apply command line means, and the notes for options it drops or maps."""
    warnings = []
    if args.outdir == '-':
        output = '-'
    else:
        output = os.path.join(args.outdir, f"{_dataset_name(args.input)}_footprints.bam")
    argv = ['-i', args.input, '-o', output, '--hmm-only']

    def opt(flag, value):
        if value is not None:
            argv.extend([flag, str(value)])

    def flag(name, on):
        if on:
            argv.append(name)

    opt('-m', args.model)
    opt('--enzyme', args.enzyme)
    opt('--seq', args.seq)
    flag('--force-seq', getattr(args, 'force_seq', False))
    opt('--mode', args.mode)
    opt('-k', args.context_size)
    # apply's own defaults (1 core, 4 I/O threads) are kept.
    opt('--cores', args.cores)
    opt('--io-threads', args.io_threads)
    opt('--chunk-size', args.chunk_size)
    opt('--min-mapq', args.min_mapq)
    opt('--prob-threshold', args.prob_threshold)
    opt('--min-read-length', args.min_read_length)
    opt('--alignments', args.alignments)
    opt('--train-reads', args.train_reads)
    if args.process_unmapped is not None:
        argv.append('--process-unmapped' if args.process_unmapped
                    else '--no-process-unmapped')
    opt('--edge-trim', args.edge_trim)
    flag('--circular', args.circular)
    flag('--scores', args.scores)
    opt('--msp-min-size', args.msp_min_size)
    opt('--nuc-min-size', args.nuc_min_size)
    flag('--keep-msps', args.no_msps)
    if args.stats:
        opt('--qc-sample-reads', args.stats_sample)
        opt('--qc-seed', args.stats_seed)
        warnings.append("--stats: fiberhmm-call writes its QC report (qc/ beside "
                        "the output BAM) instead of the apply stats files")
    else:
        argv.append('--no-qc')
        if args.stats_sample is not None or args.stats_seed is not None:
            warnings.append("--stats-sample/--stats-seed have no effect without --stats")
    if args.max_reads:
        opt('--max-reads', args.max_reads)
    if getattr(args, 'daf_mask_runs', None) is not None:
        opt('--daf-mask-runs', args.daf_mask_runs)
    if getattr(args, 'daf_run_policy', 'keep-one') != 'keep-one':
        opt('--daf-run-policy', args.daf_run_policy)
    if not args.daf_mask_unaligned:
        argv.append('--no-daf-mask-unaligned')
    if args.use_m5c is not None:
        argv.append('--use-m5c' if args.use_m5c else '--no-use-m5c')
    opt('--cpg-mask-policy', args.cpg_mask_policy)
    flag('--legacy-tags', args.legacy_tags or args.harmonize_legacy_tags)
    flag('--no-legacy-tags', getattr(args, 'no_legacy_tags', False))
    if args.harmonize_legacy_tags:
        warnings.append("--harmonize-legacy-tags is gone: --legacy-tags writes "
                        "the harmonized view (ns/nl = MA nuc, as/al = MA msp); "
                        "recall reads the MA nuc + tf of an hmm-only call")
    if args.output_posteriors:
        warnings.append(f"--output-posteriors {args.output_posteriors} ignored: "
                        "export posteriors with fiberhmm-posteriors")
    if args.debug_timing:
        warnings.append("--debug-timing ignored (it never printed a breakdown)")
    if args.streaming:
        warnings.append("--streaming ignored (fiberhmm-call streams unless "
                        "--region-parallel is given)")
    return argv, warnings


def main(argv=None):
    args = parse_args(argv)
    print(f"warning: {DEPRECATION_NOTICE}", file=sys.stderr)
    _reject_unimplemented_options(args)
    call_args, notes = call_argv(args)
    for note in notes:
        print(f"warning: fiberhmm-apply: {note}", file=sys.stderr)
    print("fiberhmm-apply -> fiberhmm-call " + ' '.join(call_args), file=sys.stderr)
    # fiberhmm-call records ' '.join(sys.argv) as the @PG CL: the translated
    # command, so the header can be re-run with fiberhmm-call itself.
    sys.argv = ['fiberhmm-call', *call_args]
    from fiberhmm.cli.call import main as call_main
    call_main(chemistry_error_hint=_chemistry_error_hint)


def _chemistry_error_hint(message):
    """The alias has no --replace-chemistry: point at fiberhmm-call."""
    return message.replace(
        ", or --replace-chemistry to re-declare the output deliberately.",
        ". fiberhmm-apply keeps the input's declaration; use "
        "fiberhmm-call --hmm-only --replace-chemistry to re-declare it deliberately.",
    ).replace(
        ", or --replace-chemistry to declare the run as custom.",
        " (fiberhmm-apply cannot re-declare the input's chemistry; "
        "fiberhmm-call --hmm-only --replace-chemistry can).",
    )


if __name__ == '__main__':
    main()
