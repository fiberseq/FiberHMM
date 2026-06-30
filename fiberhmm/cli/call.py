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

from fiberhmm.core.model_io import load_model_with_metadata
from fiberhmm.inference.parallel import (
    _process_bam_region_parallel_fused,
    _process_bam_streaming_pipeline_fused,
)
from fiberhmm.inference.tf_recaller import ENZYME_PRESETS
from fiberhmm.models import SUPPORTED_ENZYMES
from fiberhmm.models import get_model_path as _get_bundled_model


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
    p.add_argument('--reference', default=None,
                   help='Reference FASTA for --mode daf on raw BAMs that lack '
                        'both R/Y IUPAC encoding and MD tags. When present, acts '
                        'as a fallback source for deamination-site detection. '
                        'Ignored for BAMs that already have R/Y in the sequence '
                        'or have MD tags.')

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
    p.add_argument('-r', '--circular', action='store_true',
                   help='Enable circular molecule mode (3x tile internally, '
                        'emit wrapped MA/AQ/AN annotations).')
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

    # --- Nucleosome recall params ---
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

    # --- DAF chimera filter (mode=daf only) ---
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

    # --- PCR dedup (DAF / ddda|dddb only) ---
    p.add_argument('--dedup', action='store_true',
                   help='DAF (ddda/dddb) only: remove PCR duplicates by '
                        'deamination-pattern fingerprint (see fiberhmm-dedup) '
                        'BEFORE footprinting, so the HMM/recaller only process '
                        'unique molecules. Amplicon/UMI-less DAF libraries can be '
                        'heavily PCR-duplicated and coordinate dedup does not '
                        'apply. Requires a file input (not stdin). Ignored for '
                        'fiber-seq (hia5).')
    p.add_argument('--dedup-min-jaccard', type=float, default=0.95,
                   help='Deamination-set Jaccard threshold for --dedup '
                        '(default 0.95).')
    p.add_argument('--dedup-flag-only', action='store_true',
                   help='With --dedup: mark duplicates (0x400 + di/ds tags) '
                        'instead of collapsing to one read per molecule.')

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


def _resolve_nuc_profile_path(args, recall_nucs: bool):
    """DddA uses a radial deamination-profile match-filter for the nucleosome
    split; other enzymes use the accessible-cut Kadane split (no profile)."""
    if recall_nucs and args.enzyme == 'ddda':
        from fiberhmm.models import _bundled_model_path
        return _bundled_model_path('ddda_nuc_profile.json')
    return None


def _resolve_phase_nrl(args, apply_model_path, recall_model_path, mode, k,
                       recall_nucs, input_bam) -> int:
    """Resolve --phase-nrl (off / auto / fixed bp) to an int (0 = off).

    ``input_bam`` is the BAM to sample for auto-estimation -- the deduped
    file when --dedup ran, so the NRL isn't biased by PCR duplicates.
    """
    raw = str(args.phase_nrl).strip().lower()
    if raw in ('off', '', '0', 'none'):
        return 0
    if not recall_nucs:
        # phase rides on the nuc recaller; silently off when recall is off.
        return 0
    if raw != 'auto':
        try:
            return max(0, int(raw))
        except ValueError:
            print(f"  WARNING: invalid --phase-nrl {args.phase_nrl!r}; using off.",
                  file=sys.stderr)
            return 0
    # auto-estimate
    if input_bam == '-':
        print("  NOTE: --phase-nrl auto needs a file input to sample; "
              "falling back to anchor 185 bp.", file=sys.stderr)
        return 185
    from fiberhmm.inference.nrl_estimate import estimate_phase_nrl
    res = estimate_phase_nrl(
        input_bam, apply_model_path, recall_model_path,
        mode=mode, context_size=k,
        split_min_llr=args.split_min_llr, split_min_opps=args.split_min_opps,
        nuc_min_size=args.nuc_min_size, msp_min_size=args.msp_min_size,
        prob_threshold=args.prob_threshold, edge_trim=args.edge_trim,
    )
    ci = res['ci']
    ci_str = f" CI[{ci[0]:.0f}-{ci[1]:.0f}]" if ci else ""
    print(f"  phase NRL: {res['nrl']} bp ({res['source']}, "
          f"{res['n_pairs']:,} pairs from {res['n_reads']:,} reads{ci_str})",
          file=sys.stderr)
    return int(res['nrl'])


def _check_daf_inputs(input_bam: str, reference: str = None,
                       n_sniff: int = 10) -> None:
    """Sniff the first ~N mapped reads of a DAF-mode input BAM and confirm
    at least one deamination source is available: R/Y IUPAC in the stored
    sequence, an MD tag, or a user-supplied ``--reference`` FASTA.

    Exits with an actionable error if none is available -- otherwise
    every read would be silently skipped and the run would produce an
    empty output BAM.
    """
    import pysam

    # Reference FASTA is sufficient by itself (we can always reconstruct
    # mismatches from it regardless of MD tag presence).
    has_ref = reference is not None

    from fiberhmm.daf.encoder import md_matches_cigar

    has_ry = False
    has_md = False
    md_bad = 0
    md_total = 0
    checked = 0
    try:
        with pysam.AlignmentFile(input_bam, 'rb', check_sq=False) as bam:
            for read in bam:
                if read.is_unmapped or read.is_secondary or read.is_supplementary:
                    continue
                seq = read.query_sequence
                if seq and ('R' in seq or 'Y' in seq):
                    has_ry = True
                if read.has_tag('MD'):
                    has_md = True
                    md_total += 1
                    if not md_matches_cigar(read):
                        md_bad += 1
                checked += 1
                if checked >= n_sniff:
                    break
    except (ValueError, OSError):
        # Let downstream error handling report a clear message about the BAM.
        return

    if has_ry or has_md or has_ref:
        # Warn about stale MD only when we'll actually be relying on it
        # (no R/Y fast path available for these reads) AND some were bad.
        if md_bad and not has_ry and not has_ref:
            print(
                f"  NOTE: {md_bad}/{md_total} of the sniffed reads with MD tags\n"
                f"  have MD/CIGAR length mismatches (typical of consensus BAMs\n"
                f"  where MD is stale after CIGAR was recomputed). Those reads\n"
                f"  will be skipped in DAF mode. The reads themselves are fine;\n"
                f"  only the MD annotation is stale. To recover calls on those\n"
                f"  reads, regenerate MD with:\n"
                f"    samtools calmd -b aligned.bam ref.fa > fixed.bam\n"
                f"  or pass --reference ref.fa to fiberhmm-call.",
                file=sys.stderr,
            )
        return

    print(
        "error: --mode daf needs deamination calls, and none of the supported\n"
        f"  sources were found in the first {checked} mapped reads of {input_bam}:\n"
        "    - R/Y IUPAC codes in the stored query sequence\n"
        "      (produced by fiberhmm-daf-encode), or\n"
        "    - MD tags on aligned reads\n"
        "      (set by 'minimap2 --MD' or 'samtools calmd'), or\n"
        "    - a reference FASTA via --reference ref.fa\n"
        "\n"
        "  One of these is required for fiberhmm-call to locate C->T / G->A\n"
        "  deamination sites. Without it every read would be silently skipped.\n",
        file=sys.stderr,
    )
    sys.exit(2)


def _dedup_input_first(input_bam, output_bam, min_jaccard, flag_only, io_threads,
                       region_parallel):
    """PCR-dedup the input BAM BEFORE footprinting (DAF/deaminase only).

    Deduping first means the HMM/recaller only processes unique molecules
    (collapse mode removes ~the duplicate fraction of reads) and NRL/phase
    estimation isn't biased by duplicate reads. Returns the path to a
    temporary deduped BAM to footprint instead of the original, or None if
    nothing was deduplicated (caller keeps the original input).
    """
    import os
    import tempfile

    import pysam

    from fiberhmm.cli.dedup import run_dedup
    print("\n  --dedup: collapsing PCR duplicates by deamination fingerprint "
          f"(Jaccard >= {min_jaccard}, {'flag' if flag_only else 'collapse'}) "
          "BEFORE footprinting...", file=sys.stderr)
    outdir = os.path.dirname(os.path.abspath(output_bam)) if output_bam != '-' else None
    fd, tmp = tempfile.mkstemp(prefix='fiberhmm_dedup_', suffix='.bam', dir=outdir)
    os.close(fd)
    stats = run_dedup(input_bam, tmp, min_jaccard=min_jaccard,
                      collapse=not flag_only, io_threads=io_threads)
    if stats is None:
        if os.path.exists(tmp):
            os.remove(tmp)
        return None
    if region_parallel:
        # region-parallel needs a coordinate index on its input; dedup
        # preserves the input's sort order so this is valid.
        try:
            pysam.index(tmp)
        except pysam.SamtoolsError:
            pass
    return tmp


def main():
    args = parse_args()
    stdout_mode = (args.output == '-')

    if args.dedup and args.input == '-':
        print("error: --dedup requires a file input (it two-passes the BAM to "
              "fingerprint reads); cannot dedup a stdin stream.", file=sys.stderr)
        sys.exit(1)

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

    # Nucleosome recaller: ON by default for all enzymes. DddA uses a dedicated
    # radial deamination-profile match-filter (the accessible-cut Kadane split
    # shatters DddA nucleosomes, since DddA deaminates *inside* them); other
    # enzymes use the Kadane split. Explicit --recall-nucs / --no-recall-nucs wins.
    if args.recall_nucs is None:
        recall_nucs = True
        if args.enzyme == 'ddda':
            print("  NOTE: DddA nucleosome recall uses the radial-template split "
                  "(bundled ddda_nuc_profile.json).", file=sys.stderr)
    else:
        recall_nucs = bool(args.recall_nucs)

    # Fast-fail sniff for --mode daf BEFORE any BAM scanning (e.g. --phase-nrl
    # auto estimation): the DAF path needs R/Y in the stored sequence, MD tags,
    # or --reference. If none are available every read is silently skipped, so
    # error out in under a second with an actionable message.
    if mode == 'daf' and args.input != '-':
        _check_daf_inputs(args.input, args.reference)

    # PCR dedup runs FIRST (DAF only): collapse duplicates before footprinting so
    # the HMM/recaller only process unique molecules and NRL/phase estimation
    # isn't biased by duplicate reads. working_input feeds everything downstream.
    working_input = args.input
    dedup_tmp = None
    if args.dedup:
        if mode != 'daf':
            print(f"  NOTE: --dedup applies to DAF-seq (ddda/dddb) deamination "
                  f"data; mode is {mode!r} (fiber-seq has no deamination) -- "
                  f"skipping dedup.", file=sys.stderr)
        else:
            dedup_tmp = _dedup_input_first(
                args.input, args.output, args.dedup_min_jaccard,
                args.dedup_flag_only, args.io_threads, args.region_parallel)
            if dedup_tmp is not None:
                working_input = dedup_tmp

    # Resolve the Pass-2 phase prior: off / auto-estimate / fixed bp.
    phase_nrl = _resolve_phase_nrl(args, apply_model_path, recall_model_path, mode, k,
                                   recall_nucs, working_input)

    # DddA uses a radial deamination-profile match-filter for the nucleosome
    # split; other enzymes use the accessible-cut Kadane split (no profile).
    nuc_profile_path = _resolve_nuc_profile_path(args, recall_nucs)

    # @PG provenance for the output BAM header. The molecular-frame note is the
    # important bit: it tells downstream tools how to read ns/nl/as/al/MA.
    import fiberhmm as _fh
    chimera_state = ('n/a' if mode != 'daf'
                     else ('off' if args.keep_chimeras else 'on'))
    dedup_state = ('off' if not args.dedup or dedup_tmp is None
                   else f"j{args.dedup_min_jaccard}"
                        f"{'/flag' if args.dedup_flag_only else '/collapse'}")
    pg_record = {
        'PN': 'fiberhmm-call',
        'VN': getattr(_fh, '__version__', 'unknown'),
        'CL': ' '.join(sys.argv),
        # The `coord=molecular` token is a stable, version-independent contract
        # for downstream consumers (e.g. FiberBrowser) to detect that ns/nl/as/al
        # and MA are in molecular (original-fiber) frame -- keep the exact token.
        'DS': (f"FiberHMM fused apply+recall; coord=molecular "
               f"(ns/nl/as/al/MA in molecular original-fiber coordinates); "
               f"mode={mode} recall_nucs={recall_nucs} phase_nrl={phase_nrl} "
               f"chimera_filter={chimera_state} dedup={dedup_state}"),
    }

    mode_label = 'region-parallel' if args.region_parallel else 'streaming'
    print(
        "\n=========================================================================\n"
        f"  fiberhmm-call [BETA] — fused apply + recall-tfs ({mode_label})\n"
        f"  apply model:  {apply_model_path}\n"
        f"  recall model: {recall_model_path or '(reuse apply model)'}\n"
        f"  mode={mode} k={k} enzyme={args.enzyme or 'custom'}\n"
        f"  min_llr={min_llr} min_opps={args.min_opps} "
        f"unify_threshold={args.unify_threshold} uplift={uplift}\n"
        f"  cores={args.cores} io-threads={args.io_threads}"
        f"{' circular=on' if args.circular else ''}\n"
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
            input_bam=working_input,
            output_bam=args.output,
            apply_model_path=apply_model_path,
            recall_model_path=recall_model_path,
            train_rids=set(),
            edge_trim=args.edge_trim,
            circular=args.circular,
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
            ref_fasta_path=args.reference,
            recall_nucs=recall_nucs,
            split_min_llr=args.split_min_llr,
            split_min_opps=args.split_min_opps,
            filter_chimeras=not args.keep_chimeras,
            chimera_min_seg=args.chimera_min_seg,
            chimera_purity=args.chimera_purity,
            phase_nrl=phase_nrl,
            nuc_profile_path=nuc_profile_path,
            pg_record=pg_record,
        )
    else:
        n_reads, n_fp = _process_bam_streaming_pipeline_fused(
            input_bam=working_input,
            output_bam=args.output,
            model_path=apply_model_path,
            recall_model_path=recall_model_path,
            train_rids=set(),
            edge_trim=args.edge_trim,
            circular=args.circular,
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
            ref_fasta_path=args.reference,
            recall_nucs=recall_nucs,
            split_min_llr=args.split_min_llr,
            split_min_opps=args.split_min_opps,
            filter_chimeras=not args.keep_chimeras,
            chimera_min_seg=args.chimera_min_seg,
            chimera_purity=args.chimera_purity,
            phase_nrl=phase_nrl,
            nuc_profile_path=nuc_profile_path,
            pg_record=pg_record,
        )

    if not stdout_mode and not args.region_parallel:
        # region-parallel already indexes.  Streaming mode needs an index pass.
        import pysam
        try:
            pysam.index(args.output)
        except pysam.SamtoolsError:
            pass

    # Clean up the pre-footprinting dedup temp (+ its index), if any.
    if dedup_tmp is not None:
        import os
        for _p in (dedup_tmp, dedup_tmp + '.bai'):
            try:
                os.remove(_p)
            except OSError:
                pass


if __name__ == '__main__':
    main()
