#!/usr/bin/env python3
"""fiberhmm-call — fused apply + recall-tfs in a single Python process.

Eliminates the streaming-pipeline pipe serialization between `fiberhmm-apply`
and `fiberhmm-recall-tfs` by running the 2-state HMM (nucleosome/MSP calls)
and the LLR Kadane scan (TF calls) in the same worker per read.  Uses the
same slim-IPC main→worker payload as `fiberhmm-apply`.

Output BAM has BOTH:
  - Legacy ns/nl/as/al tags (post-unification: short nucs overlapping TF
    calls are demoted to the tf track)
  - MA/AQ spec tags (Molecular-annotation with tf.QQQ quality scoring)

Examples:
  # Hia5 PacBio, fused apply+recall (bundled model)
  fiberhmm-call -i aligned.bam -o out.bam --enzyme hia5 --seq pacbio -c 8

  # DddB DAF-seq with custom recall threshold
  fiberhmm-call -i aligned.bam -o out.bam --enzyme dddb --min-llr 6.0 -c 8

  # Stream to stdout for pipe to ft fire or samtools sort
  fiberhmm-call -i in.bam -o - --enzyme hia5 --seq pacbio | ft fire - -
"""
import argparse
import sys

from fiberhmm.cli.common import resolve_chroms_set
from fiberhmm.cli.recall_config import resolve_recall_defaults, should_write_legacy_tags
from fiberhmm.core.model_io import load_model_with_metadata
from fiberhmm.inference.parallel import (
    _process_bam_region_parallel_fused,
    _process_bam_streaming_pipeline_fused,
)
from fiberhmm.models import SUPPORTED_ENZYMES
from fiberhmm.models import get_model_path as _get_bundled_model


def _add_call_io_args(p) -> None:
    p.add_argument('-i', '--input', required=True,
                   help='Input BAM. Use "-" for stdin (streaming mode).')
    p.add_argument('-o', '--output', required=True,
                   help='Output BAM path or "-" for stdout (unsorted).')
    p.add_argument('-m', '--model', default=None,
                   help='Apply HMM model JSON. If omitted, bundled model for '
                        '--enzyme/--seq is used.')
    p.add_argument('--recall-model', default=None,
                   help='Separate model for TF LLR tables. Default: reuse apply model.')
    p.add_argument('--enzyme', choices=sorted(SUPPORTED_ENZYMES), default=None,
                   help='Enzyme preset (hia5/dddb/ddda).')
    p.add_argument('--seq', choices=['pacbio', 'nanopore'], default=None,
                   help='Platform (required for hia5).')
    p.add_argument('--reference', default=None,
                   help='Reference FASTA for --mode daf on raw BAMs that lack '
                        'both R/Y IUPAC encoding and MD tags. When present, acts '
                        'as a fallback source for deamination-site detection. '
                        'Ignored for BAMs that already have R/Y in the sequence '
                        'or have MD tags.')


def _add_call_apply_args(p) -> None:
    p.add_argument('--mode', default=None,
                   help='Observation mode override. Default: from model.')
    p.add_argument('-k', '--context-size', type=int, default=None,
                   help='Context size override. Default: from model.')
    p.add_argument('--edge-trim', type=int, default=10,
                   help='Bases to mask at edges (default 10)')
    p.add_argument('--min-mapq', type=int, default=0,
                   help='Min mapping quality (default 0)')
    p.add_argument('--prob-threshold', type=int, default=128,
                   help='Min modification probability 0-255 (default 128)')
    p.add_argument('--min-read-length', type=int, default=1000,
                   help='Min aligned read length (default 1000 — matches fiberhmm-apply)')
    p.add_argument('--msp-min-size', type=int, default=0,
                   help='Min MSP size (default 0)')
    p.add_argument('--nuc-min-size', type=int, default=85,
                   help='Min footprint size to count as nucleosome (default 85)')
    p.add_argument('--with-scores', action='store_true',
                   help='Compute confidence scores (nq/aq tags).')
    p.add_argument('-r', '--circular', action='store_true',
                   help='Enable circular molecule mode (3x tile internally, '
                        'emit wrapped MA/AQ/AN annotations).')
    p.add_argument('--process-unmapped', action='store_true',
                   help='Process unmapped reads (default: pass through).')
    p.add_argument('--primary', action='store_true',
                   help='Skip secondary/supplementary alignments.')


def _add_call_recall_args(p) -> None:
    p.add_argument('--min-llr', type=float, default=None,
                   help='Min LLR for TF call (default: enzyme preset).')
    p.add_argument('--min-opps', type=int, default=3,
                   help='Min informative target positions per TF call (default 3).')
    p.add_argument('--unify-threshold', type=int, default=90,
                   help='v2 nucs with nl < this may be demoted to tf+ (default 90).')
    p.add_argument('--emission-uplift', type=float, default=None,
                   help='Emission power transform. Default: enzyme preset.')
    p.add_argument('--no-legacy-tags', action='store_true',
                   help='Skip ns/nl/as/al, emit only MA/AQ.')
    p.add_argument('--downstream-compat', action='store_true',
                   help='Skip MA/AQ; write TF calls into legacy ns/nl track.')


def _add_call_nuc_recall_args(p) -> None:
    p.add_argument('--recall-nucs', action=argparse.BooleanOptionalAction, default=None,
                   help='Split over-merged nucleosomes + refine conservative edges '
                        '(emits nuc.QQQ), promote nucleosome-sized TF leaks to nuc, '
                        'and run the Pass-2 phase prior. ON by default (except DddA). '
                        'Use --no-recall-nucs for baseline HMM nucleosomes (nuc.Q).')
    p.add_argument('--split-min-llr', type=float, default=4.0,
                   help='Min accessible-run LLR to split a nucleosome (default 4.0).')
    p.add_argument('--split-min-opps', type=int, default=3,
                   help='Min informative positions in a nucleosome-splitting cut '
                        '(default 3).')
    p.add_argument('--phase-nrl', default='auto',
                   help='Pass-2 periodicity prior (with --recall-nucs): '
                        '"auto" (default; estimate the nucleosome repeat length from '
                        'this sample after Pass 1, clamped to ~150-215 bp anchored at '
                        '185), "off", or a fixed bp value (e.g. 185). Long footprints '
                        'are split at phase-predicted linkers using a lowered threshold '
                        'gated on >=1 local deamination event (never splits a '
                        'signal-desert).')


def _add_call_chimera_args(p) -> None:
    p.add_argument('--keep-chimeras', action='store_true',
                   help='DAF only: keep strand-swap chimeric reads (C->T in one '
                        'segment + G->A in another). Default: filter them out and '
                        'report the count.')
    p.add_argument('--chimera-min-seg', type=int, default=5,
                   help='DAF chimera: min same-strand deamination events per '
                        'segment to call a swap (default 5).')
    p.add_argument('--chimera-purity', type=float, default=0.8,
                   help='DAF chimera: min same-strand purity per segment '
                        '(default 0.8).')


def _add_call_parallel_args(p) -> None:
    p.add_argument('-c', '--cores', type=int, default=4,
                   help='Worker processes (default 4).')
    p.add_argument('--chunk-size', type=int, default=500,
                   help='Reads per worker chunk (default 500; streaming mode only).')
    p.add_argument('--io-threads', type=int, default=8,
                   help='htslib I/O threads per stage (default 8).')
    p.add_argument('--max-reads', type=int, default=0,
                   help='0 = no limit (default; streaming mode only)')


def _add_call_region_args(p) -> None:
    p.add_argument('--region-parallel', action='store_true',
                   help='Process genomic regions in parallel (one worker per region). '
                        'Scales linearly with --cores up to chromosome count. '
                        'Requires coordinate-sorted + indexed input BAM. '
                        'Recommended for full-genome runs; use streaming for stdin/unaligned.')
    p.add_argument('--region-size', type=int, default=10_000_000,
                   help='Region size in bp for --region-parallel (default 10 Mb).')
    p.add_argument('--skip-scaffolds', action='store_true',
                   help='Skip scaffold/contig chromosomes in region-parallel mode.')
    p.add_argument('--chroms', nargs='+', default=None,
                   help='Only process these chromosomes (region-parallel mode).')


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_call_io_args(p)
    _add_call_apply_args(p)
    _add_call_recall_args(p)
    _add_call_nuc_recall_args(p)
    _add_call_chimera_args(p)
    _add_call_parallel_args(p)
    _add_call_region_args(p)
    return p.parse_args()


def _resolve_apply_model(args):
    if args.model:
        return args.model
    if args.enzyme is None:
        print("error: one of --model or --enzyme required.", file=sys.stderr)
        sys.exit(1)
    return _get_bundled_model(args.enzyme, tool='apply', seq=args.seq)


def _resolve_recall_model(args):
    if args.recall_model:
        return args.recall_model
    # For recall, bundled model may differ; fallback to apply model if not available.
    if args.enzyme:
        try:
            return _get_bundled_model(args.enzyme, tool='recall', seq=args.seq)
        except (KeyError, FileNotFoundError):
            pass
    return None  # reuse apply model


def _call_mode_or_default(args_mode, model_mode) -> str:
    return args_mode or model_mode or 'pacbio-fiber'


def _call_context_size_or_default(args_context_size, model_k) -> int:
    return args_context_size or int(model_k or 3)


def _resolve_call_mode_context(args, model_k, model_mode):
    return (
        _call_mode_or_default(args.mode, model_mode),
        _call_context_size_or_default(args.context_size, model_k),
    )


_resolve_call_chroms = resolve_chroms_set


def _check_region_parallel_file_io(args) -> None:
    if args.input != '-' and args.output != '-':
        return
    print("error: --region-parallel requires file I/O "
          "(input must be indexed BAM, not stdin; output cannot be stdout).",
          file=sys.stderr)
    sys.exit(1)


def _normalize_phase_nrl_option(value) -> str:
    return str(value).strip().lower()


def _is_phase_nrl_off(raw: str) -> bool:
    return raw in ('off', '', '0', 'none')


def _parse_fixed_phase_nrl(raw: str):
    try:
        return max(0, int(raw))
    except ValueError:
        return None


def _phase_nrl_estimate_message(result: dict) -> str:
    ci = result['ci']
    ci_str = f" CI[{ci[0]:.0f}-{ci[1]:.0f}]" if ci else ""
    return (
        f"  phase NRL: {result['nrl']} bp ({result['source']}, "
        f"{result['n_pairs']:,} pairs from {result['n_reads']:,} reads{ci_str})"
    )


def _invalid_phase_nrl_message(value) -> str:
    return f"  WARNING: invalid --phase-nrl {value!r}; using off."


def _resolve_phase_nrl(args, apply_model_path, recall_model_path, mode, k,
                       recall_nucs) -> int:
    """Resolve --phase-nrl (off / auto / fixed bp) to an int (0 = off)."""
    raw = _normalize_phase_nrl_option(args.phase_nrl)
    if _is_phase_nrl_off(raw):
        return 0
    if not recall_nucs:
        # phase rides on the nuc recaller; silently off when recall is off.
        return 0
    if raw != 'auto':
        fixed_nrl = _parse_fixed_phase_nrl(raw)
        if fixed_nrl is None:
            print(_invalid_phase_nrl_message(args.phase_nrl), file=sys.stderr)
            return 0
        return fixed_nrl
    # auto-estimate
    if args.input == '-':
        print("  NOTE: --phase-nrl auto needs a file input to sample; "
              "falling back to anchor 185 bp.", file=sys.stderr)
        return 185
    from fiberhmm.inference.nrl_estimate import estimate_phase_nrl
    res = estimate_phase_nrl(
        args.input, apply_model_path, recall_model_path,
        mode=mode, context_size=k,
        split_min_llr=args.split_min_llr, split_min_opps=args.split_min_opps,
        nuc_min_size=args.nuc_min_size, msp_min_size=args.msp_min_size,
        prob_threshold=args.prob_threshold, edge_trim=args.edge_trim,
    )
    print(_phase_nrl_estimate_message(res), file=sys.stderr)
    return int(res['nrl'])


def _resolve_recall_nucs(args) -> bool:
    """Resolve the nucleosome recaller default and DddA-specific warnings."""
    if args.recall_nucs is None:
        recall_nucs = (args.enzyme != 'ddda')
        if args.enzyme == 'ddda':
            print("  NOTE: nucleosome recaller is OFF by default for DddA "
                  "(use --recall-nucs to force it on).", file=sys.stderr)
    else:
        recall_nucs = bool(args.recall_nucs)
        if recall_nucs and args.enzyme == 'ddda':
            print("  WARNING: --recall-nucs on DddA uses the uplifted TF model for "
                  "splitting (aggressive) — verify results.", file=sys.stderr)
    return recall_nucs


def _chimera_filter_state(mode: str, keep_chimeras: bool) -> str:
    if mode != 'daf':
        return 'n/a'
    return 'off' if keep_chimeras else 'on'


def _call_pg_description(mode, recall_nucs, phase_nrl, keep_chimeras) -> str:
    chimera_state = _chimera_filter_state(mode, keep_chimeras)
    return (
        f"FiberHMM fused apply+recall; coord=molecular "
        f"(ns/nl/as/al/MA in molecular original-fiber coordinates); "
        f"mode={mode} recall_nucs={recall_nucs} phase_nrl={phase_nrl} "
        f"chimera_filter={chimera_state}"
    )


def _build_pg_record(mode, recall_nucs, phase_nrl, keep_chimeras, argv=None):
    """Build the @PG provenance record for fused call output BAMs."""
    import fiberhmm as _fh

    argv = sys.argv if argv is None else argv
    return {
        'PN': 'fiberhmm-call',
        'VN': getattr(_fh, '__version__', 'unknown'),
        'CL': ' '.join(argv),
        # The `coord=molecular` token is a stable, version-independent contract
        # for downstream consumers (e.g. FiberBrowser) to detect that ns/nl/as/al
        # and MA are in molecular (original-fiber) frame -- keep the exact token.
        'DS': _call_pg_description(mode, recall_nucs, phase_nrl, keep_chimeras),
    }


def _call_mode_label(region_parallel: bool) -> str:
    return 'region-parallel' if region_parallel else 'streaming'


def _call_recall_model_label(recall_model_path) -> str:
    return recall_model_path or '(reuse apply model)'


def _call_enzyme_label(enzyme) -> str:
    return enzyme or 'custom'


def _call_circular_label(circular: bool) -> str:
    return ' circular=on' if circular else ''


def _call_banner_text(apply_model_path, recall_model_path, mode, k, enzyme,
                      min_llr, min_opps, unify_threshold, uplift,
                      cores, io_threads, circular, region_parallel):
    mode_label = _call_mode_label(region_parallel)
    recall_model_label = _call_recall_model_label(recall_model_path)
    enzyme_label = _call_enzyme_label(enzyme)
    circular_label = _call_circular_label(circular)
    return (
        "\n=========================================================================\n"
        f"  fiberhmm-call [BETA] — fused apply + recall-tfs ({mode_label})\n"
        f"  apply model:  {apply_model_path}\n"
        f"  recall model: {recall_model_label}\n"
        f"  mode={mode} k={k} enzyme={enzyme_label}\n"
        f"  min_llr={min_llr} min_opps={min_opps} "
        f"unify_threshold={unify_threshold} uplift={uplift}\n"
        f"  cores={cores} io-threads={io_threads}{circular_label}\n"
        "=========================================================================\n"
    )


def _new_daf_sniff_result() -> dict:
    return {
        'has_ry': False,
        'has_md': False,
        'md_bad': 0,
        'md_total': 0,
        'checked': 0,
    }


def _sniff_daf_input_sources(input_bam: str, n_sniff: int = 10):
    import pysam

    from fiberhmm.daf.encoder import md_matches_cigar

    result = _new_daf_sniff_result()
    try:
        with pysam.AlignmentFile(input_bam, 'rb', check_sq=False) as bam:
            for read in bam:
                if read.is_unmapped or read.is_secondary or read.is_supplementary:
                    continue
                seq = read.query_sequence
                if seq and ('R' in seq or 'Y' in seq):
                    result['has_ry'] = True
                if read.has_tag('MD'):
                    result['has_md'] = True
                    result['md_total'] += 1
                    if not md_matches_cigar(read):
                        result['md_bad'] += 1
                result['checked'] += 1
                if result['checked'] >= n_sniff:
                    break
    except (ValueError, OSError):
        # Let downstream error handling report a clear message about the BAM.
        return None
    return result


def _daf_sources_available(sniff: dict, has_ref: bool) -> bool:
    return bool(sniff['has_ry'] or sniff['has_md'] or has_ref)


def _should_warn_stale_daf_md(sniff: dict, has_ref: bool) -> bool:
    return bool(sniff['md_bad'] and not sniff['has_ry'] and not has_ref)


def _stale_daf_md_warning_message(sniff: dict) -> str:
    return (
        f"  NOTE: {sniff['md_bad']}/{sniff['md_total']} "
        f"of the sniffed reads with MD tags\n"
        f"  have MD/CIGAR length mismatches (typical of consensus BAMs\n"
        f"  where MD is stale after CIGAR was recomputed). Those reads\n"
        f"  will be skipped in DAF mode. The reads themselves are fine;\n"
        f"  only the MD annotation is stale. To recover calls on those\n"
        f"  reads, regenerate MD with:\n"
        f"    samtools calmd -b aligned.bam ref.fa > fixed.bam\n"
        f"  or pass --reference ref.fa to fiberhmm-call."
    )


def _missing_daf_source_message(input_bam: str, checked_reads: int) -> str:
    return (
        "error: --mode daf needs deamination calls, and none of the supported\n"
        f"  sources were found in the first {checked_reads} "
        f"mapped reads of {input_bam}:\n"
        "    - R/Y IUPAC codes in the stored query sequence\n"
        "      (produced by fiberhmm-daf-encode), or\n"
        "    - MD tags on aligned reads\n"
        "      (set by 'minimap2 --MD' or 'samtools calmd'), or\n"
        "    - a reference FASTA via --reference ref.fa\n"
        "\n"
        "  One of these is required for fiberhmm-call to locate C->T / G->A\n"
        "  deamination sites. Without it every read would be silently skipped.\n"
    )


def _check_daf_inputs(input_bam: str, reference: str = None,
                       n_sniff: int = 10) -> None:
    """Sniff the first ~N mapped reads of a DAF-mode input BAM and confirm
    at least one deamination source is available: R/Y IUPAC in the stored
    sequence, an MD tag, or a user-supplied ``--reference`` FASTA.

    Exits with an actionable error if none is available -- otherwise
    every read would be silently skipped and the run would produce an
    empty output BAM.
    """
    # Reference FASTA is sufficient by itself (we can always reconstruct
    # mismatches from it regardless of MD tag presence).
    has_ref = reference is not None

    sniff = _sniff_daf_input_sources(input_bam, n_sniff)
    if sniff is None:
        return

    if _daf_sources_available(sniff, has_ref):
        # Warn about stale MD only when we'll actually be relying on it
        # (no R/Y fast path available for these reads) AND some were bad.
        if _should_warn_stale_daf_md(sniff, has_ref):
            print(_stale_daf_md_warning_message(sniff), file=sys.stderr)
        return

    print(
        _missing_daf_source_message(input_bam, sniff['checked']),
        file=sys.stderr,
    )
    sys.exit(2)


def _should_check_daf_inputs(mode: str, input_path: str) -> bool:
    return mode == 'daf' and input_path != '-'


def _call_fused_common_kwargs(
    args,
    recall_model_path,
    mode: str,
    context_size: int,
    min_llr: float,
    emission_uplift: float,
    also_write_legacy: bool,
    recall_nucs: bool,
    phase_nrl: int,
    pg_record,
) -> dict:
    return {
        'input_bam': args.input,
        'output_bam': args.output,
        'recall_model_path': recall_model_path,
        'train_rids': set(),
        'edge_trim': args.edge_trim,
        'circular': args.circular,
        'mode': mode,
        'context_size': context_size,
        'msp_min_size': args.msp_min_size,
        'nuc_min_size': args.nuc_min_size,
        'min_mapq': args.min_mapq,
        'prob_threshold': args.prob_threshold,
        'min_read_length': args.min_read_length,
        'with_scores': args.with_scores,
        'min_llr': min_llr,
        'min_opps': args.min_opps,
        'unify_threshold': args.unify_threshold,
        'emission_uplift': emission_uplift,
        'also_write_legacy': also_write_legacy,
        'downstream_compat': args.downstream_compat,
        'n_cores': args.cores,
        'io_threads': args.io_threads,
        'primary_only': args.primary,
        'ref_fasta_path': args.reference,
        'recall_nucs': recall_nucs,
        'split_min_llr': args.split_min_llr,
        'split_min_opps': args.split_min_opps,
        'filter_chimeras': not args.keep_chimeras,
        'chimera_min_seg': args.chimera_min_seg,
        'chimera_purity': args.chimera_purity,
        'phase_nrl': phase_nrl,
        'pg_record': pg_record,
    }


def main():
    args = parse_args()
    stdout_mode = (args.output == '-')

    if stdout_mode:
        sys.stdout = sys.stderr  # informational prints → stderr, BAM → real stdout

    apply_model_path = _resolve_apply_model(args)
    recall_model_path = _resolve_recall_model(args)

    # Resolve mode/k from model metadata
    _, model_k, model_mode = load_model_with_metadata(apply_model_path)
    mode, k = _resolve_call_mode_context(args, model_k, model_mode)

    min_llr, uplift = resolve_recall_defaults(args)

    recall_nucs = _resolve_recall_nucs(args)

    # Fast-fail sniff for --mode daf BEFORE any BAM scanning (e.g. --phase-nrl
    # auto estimation): the DAF path needs R/Y in the stored sequence, MD tags,
    # or --reference. If none are available every read is silently skipped, so
    # error out in under a second with an actionable message.
    if _should_check_daf_inputs(mode, args.input):
        _check_daf_inputs(args.input, args.reference)

    # Resolve the Pass-2 phase prior: off / auto-estimate / fixed bp.
    phase_nrl = _resolve_phase_nrl(args, apply_model_path, recall_model_path, mode, k,
                                   recall_nucs)

    pg_record = _build_pg_record(
        mode, recall_nucs, phase_nrl, args.keep_chimeras, sys.argv,
    )

    print(
        _call_banner_text(
            apply_model_path=apply_model_path,
            recall_model_path=recall_model_path,
            mode=mode,
            k=k,
            enzyme=args.enzyme,
            min_llr=min_llr,
            min_opps=args.min_opps,
            unify_threshold=args.unify_threshold,
            uplift=uplift,
            cores=args.cores,
            io_threads=args.io_threads,
            circular=args.circular,
            region_parallel=args.region_parallel,
        ),
        file=sys.stderr,
    )

    also_write_legacy = should_write_legacy_tags(args)
    common_kwargs = _call_fused_common_kwargs(
        args,
        recall_model_path,
        mode,
        k,
        min_llr,
        uplift,
        also_write_legacy,
        recall_nucs,
        phase_nrl,
        pg_record,
    )

    if args.region_parallel:
        _check_region_parallel_file_io(args)
        chroms_set = _resolve_call_chroms(args.chroms)
        n_reads, n_fp = _process_bam_region_parallel_fused(
            apply_model_path=apply_model_path,
            region_size=args.region_size,
            skip_scaffolds=args.skip_scaffolds,
            chroms=chroms_set,
            **common_kwargs,
        )
    else:
        n_reads, n_fp = _process_bam_streaming_pipeline_fused(
            model_path=apply_model_path,
            max_reads=args.max_reads,
            chunk_size=args.chunk_size,
            process_unmapped=args.process_unmapped,
            **common_kwargs,
        )

    if not stdout_mode and not args.region_parallel:
        # region-parallel already indexes.  Streaming mode needs an index pass.
        import pysam
        try:
            pysam.index(args.output)
        except pysam.SamtoolsError:
            pass


if __name__ == '__main__':
    main()
