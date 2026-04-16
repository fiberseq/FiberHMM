#!/usr/bin/env python3
"""fiberhmm-call — fused apply + recall-tfs in a single Python process.

Eliminates the streaming-pipeline pipe serialization between `fiberhmm-apply`
and `fiberhmm-recall-tfs` by running the 2-state HMM (nucleosome/MSP calls)
and the LLR Kadane scan (TF calls) in the same worker per read.  Uses the
same slim-IPC main→worker payload as `fiberhmm-apply`.

Output BAM has BOTH:
  - Legacy ns/nl/as/al tags (post-unification: short nucs overlapping TF
    calls are demoted to the tf+ track)
  - MA/AQ spec tags (Molecular-annotation with tf+QQQ quality scoring)

Examples:
  # Hia5 PacBio, fused apply+recall (bundled model)
  fiberhmm-call -i aligned.bam -o out.bam --enzyme hia5 --seq pacbio -c 8

  # DddB DAF-seq with custom recall threshold
  fiberhmm-call -i aligned.bam -o out.bam --enzyme dddb --min-llr 6.0 -c 8

  # Stream to stdout for pipe to ft fire or samtools sort
  fiberhmm-call -i in.bam -o - --enzyme hia5 --seq pacbio | ft fire - -
"""
import argparse
import os
import sys

from fiberhmm.core.model_io import load_model_with_metadata
from fiberhmm.models import SUPPORTED_ENZYMES, get_model_path as _get_bundled_model
from fiberhmm.inference.parallel import (
    _process_bam_streaming_pipeline_fused,
    _process_bam_region_parallel_fused,
)
from fiberhmm.inference.tf_recaller import ENZYME_PRESETS


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # --- I/O ---
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

    # --- Apply params ---
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
    p.add_argument('--process-unmapped', action='store_true',
                   help='Process unmapped reads (default: pass through).')
    p.add_argument('--primary', action='store_true',
                   help='Skip secondary/supplementary alignments.')

    # --- Recall params ---
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

    # --- Parallelism ---
    p.add_argument('-c', '--cores', type=int, default=4,
                   help='Worker processes (default 4).')
    p.add_argument('--chunk-size', type=int, default=500,
                   help='Reads per worker chunk (default 500; streaming mode only).')
    p.add_argument('--io-threads', type=int, default=8,
                   help='htslib I/O threads per stage (default 8).')
    p.add_argument('--max-reads', type=int, default=0,
                   help='0 = no limit (default; streaming mode only)')

    # --- Region-parallel mode ---
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


def main():
    args = parse_args()
    stdout_mode = (args.output == '-')

    if stdout_mode:
        sys.stdout = sys.stderr  # informational prints → stderr, BAM → real stdout

    apply_model_path = _resolve_apply_model(args)
    recall_model_path = _resolve_recall_model(args)

    # Resolve mode/k from model metadata
    _, model_k, model_mode = load_model_with_metadata(apply_model_path)
    mode = args.mode or model_mode or 'pacbio-fiber'
    k = args.context_size or int(model_k or 3)

    # Enzyme preset for recall params
    preset = ENZYME_PRESETS.get(args.enzyme, {}) if args.enzyme else {}
    min_llr = args.min_llr if args.min_llr is not None else preset.get('min_llr', 5.0)
    uplift = args.emission_uplift if args.emission_uplift is not None \
             else preset.get('emission_uplift', 1.0)

    mode_label = 'region-parallel' if args.region_parallel else 'streaming'
    print(
        "\n=========================================================================\n"
        f"  fiberhmm-call [BETA] — fused apply + recall-tfs ({mode_label})\n"
        f"  apply model:  {apply_model_path}\n"
        f"  recall model: {recall_model_path or '(reuse apply model)'}\n"
        f"  mode={mode} k={k} enzyme={args.enzyme or 'custom'}\n"
        f"  min_llr={min_llr} min_opps={args.min_opps} "
        f"unify_threshold={args.unify_threshold} uplift={uplift}\n"
        f"  cores={args.cores} io-threads={args.io_threads}\n"
        "=========================================================================\n",
        file=sys.stderr,
    )

    also_write_legacy = True if args.downstream_compat else (not args.no_legacy_tags)

    if args.region_parallel:
        if args.input == '-' or args.output == '-':
            print("error: --region-parallel requires file I/O "
                  "(input must be indexed BAM, not stdin; output cannot be stdout).",
                  file=sys.stderr)
            sys.exit(1)
        chroms_set = set(args.chroms) if args.chroms else None
        n_reads, n_fp = _process_bam_region_parallel_fused(
            input_bam=args.input,
            output_bam=args.output,
            apply_model_path=apply_model_path,
            recall_model_path=recall_model_path,
            train_rids=set(),
            edge_trim=args.edge_trim,
            circular=False,
            mode=mode,
            context_size=k,
            msp_min_size=args.msp_min_size,
            nuc_min_size=args.nuc_min_size,
            min_mapq=args.min_mapq,
            prob_threshold=args.prob_threshold,
            min_read_length=args.min_read_length,
            with_scores=args.with_scores,
            min_llr=min_llr,
            min_opps=args.min_opps,
            unify_threshold=args.unify_threshold,
            emission_uplift=uplift,
            also_write_legacy=also_write_legacy,
            downstream_compat=args.downstream_compat,
            n_cores=args.cores,
            region_size=args.region_size,
            skip_scaffolds=args.skip_scaffolds,
            chroms=chroms_set,
            io_threads=args.io_threads,
            primary_only=args.primary,
        )
    else:
        n_reads, n_fp = _process_bam_streaming_pipeline_fused(
            input_bam=args.input,
            output_bam=args.output,
            model_path=apply_model_path,
            recall_model_path=recall_model_path,
            train_rids=set(),
            edge_trim=args.edge_trim,
            circular=False,
            mode=mode,
            context_size=k,
            msp_min_size=args.msp_min_size,
            nuc_min_size=args.nuc_min_size,
            min_mapq=args.min_mapq,
            prob_threshold=args.prob_threshold,
            min_read_length=args.min_read_length,
            with_scores=args.with_scores,
            min_llr=min_llr,
            min_opps=args.min_opps,
            unify_threshold=args.unify_threshold,
            emission_uplift=uplift,
            also_write_legacy=also_write_legacy,
            downstream_compat=args.downstream_compat,
            max_reads=args.max_reads,
            n_cores=args.cores,
            chunk_size=args.chunk_size,
            io_threads=args.io_threads,
            process_unmapped=args.process_unmapped,
            primary_only=args.primary,
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
