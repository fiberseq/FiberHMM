#!/usr/bin/env python3
"""fiberhmm-pipeline — reads + reference -> called BAM ready for FiberBrowser.

One command from a sequencing run to footprints: aligns the reads with
minimap2 (DAF-seq on Nanopore: -ax map-ont --MD -Y), keeps primary alignments
at MAPQ >= 20, joins reads that run through the origin of a circular plasmid,
trims unaligned read arms, calls footprints with fiberhmm-call (duplicate
marking, chimera filter and SNP screen as in fiberhmm-call) and runs
fiberhmm-qc. BAMs already aligned to the reference (e.g. Fiber-seq aligned
with pbmm2) are called as they are.

Examples:
  # A Plasmidsaurus DAF-seq run on a plasmid map
  fiberhmm-pipeline reads.fastq.gz --reference construct.dna --enzyme dddb -o out/

  # Amplicons on a genome (a directory of FASTQs is one sample)
  fiberhmm-pipeline fastq_pass/ --reference dm6.fa --enzyme dddb -o out/ -c 4

  # A Fiber-seq BAM already aligned to hg38: call only
  fiberhmm-pipeline sample.aligned.bam --reference hg38.fa --enzyme hia5 -o out/

OUTDIR gets <sample>.aligned.bam, <sample>.fiberhmm.bam (+ .bai), qc/, the
reference FASTA (and a copy of the plasmid map), optional tracks/, and
outputs.json (schema fiberhmm.pipeline.outputs.v1) saying what to open in
FiberBrowser. Re-running the same command skips completed steps whose outputs
still have their recorded SHA-256; a changed input, reference, setting or
--call-args file is refused before OUTDIR changes (--redo STEP replaces it).
One run owns OUTDIR at a time. SIGTERM or Ctrl-C stops cleanly and the next
run continues.
"""
from __future__ import annotations

import argparse
import shlex
import sys

from fiberhmm.cli.common import add_version_args


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="fiberhmm-pipeline",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("reads", nargs="+",
                   help="Read files: FASTQ (.fastq/.fq, optionally .gz), unaligned "
                        "BAM, a BAM aligned to --reference, or a directory of them. "
                        "All reads given form one sample.")
    p.add_argument("--reference", required=True,
                   help="Reference FASTA, or a plasmid map (.dna, .gb/.gbk/.genbank, "
                        ".embl) converted to a FASTA whose contig is named as "
                        "FiberBrowser names the map.")
    p.add_argument("--enzyme", required=True, choices=["ddda", "dddb", "hia5"],
                   help="Chemistry: ddda / dddb (DAF-seq) or hia5 (Fiber-seq).")
    p.add_argument("-o", "--outdir", required=True, help="Output directory.")
    p.add_argument("--sample", default=None,
                   help="Sample name for output files and the read group "
                        "(default: the first input's name without extensions). "
                        "One plain file name: no path separators, spaces or "
                        "leading '.'/'-'.")
    p.add_argument("-c", "--cores", type=int, default=4,
                   help="minimap2 threads and fiberhmm-call worker processes "
                        "(default 4).")
    p.add_argument("--seq", choices=["nanopore", "pacbio"], default=None,
                   help="Sequencing platform. Default: detected from the reads. "
                        "ddda/dddb: a BAM's FIBERHMM-CHEMISTRY declaration or "
                        "@RG PL/@PG records; reads with no record are Nanopore "
                        "when aligned here, and keep fiberhmm-call's default when "
                        "called as given. hia5: also the MM tags of the first "
                        "reads (T-a = PacBio, A+a only = Nanopore), an error when "
                        "nothing settles it. Sets the minimap2 preset (map-ont / "
                        "map-hifi), the read group's PL and fiberhmm-call's --seq.")
    p.add_argument("--force-chemistry", action="store_true",
                   help="Run although the reads contradict --enzyme/--seq (a "
                        "FIBERHMM-CHEMISTRY declaration naming another enzyme or "
                        "platform; m6A-tagged reads without deaminations for "
                        "ddda/dddb; deaminated or untagged reads for hia5). "
                        "Default: refuse.")
    p.add_argument("--topology", choices=["auto", "circular", "linear"], default="auto",
                   help="Reference topology. auto (default): a plasmid map's own "
                        "topology, FASTA contigs linear. circular: every contig is "
                        "circular (a plasmid FASTA).")
    p.add_argument("--region", action="append", default=[], metavar="CHR:START-END",
                   help="Keep only reads overlapping this region (1-based, "
                        "inclusive; repeatable). The first region is the one "
                        "outputs.json asks FiberBrowser to open.")

    align = p.add_argument_group("alignment")
    align.add_argument("--min-mapq", type=int, default=20,
                       help="Keep primary alignments with at least this MAPQ "
                            "(default 20); also passed to fiberhmm-call and "
                            "fiberhmm-qc.")
    align.add_argument("--keep-soft-clips", action="store_true",
                       help="Keep unaligned read arms as soft clips. Default: hard-clip "
                            "them for ddda/dddb (concatemer and chimera arms), keep "
                            "them for hia5.")
    align.add_argument("--no-origin-merge", dest="origin_merge", action="store_false",
                       help="Do not join the two pieces of reads that run through the "
                            "origin of a circular reference (keep the primary piece).")
    align.add_argument("--aligner", choices=["auto", "minimap2", "mappy"], default="auto",
                       help="The minimap2 program on PATH or the mappy module "
                            "(default: auto, program first).")

    call = p.add_argument_group("calling (passed to fiberhmm-call; defaults are "
                                "fiberhmm-call's)")
    call.add_argument("--min-read-length", type=int, default=None,
                      help="Minimum aligned read length to call (default 1000).")
    call.add_argument("--dedup", choices=["auto", "on", "off"], default="auto",
                      help="DAF PCR-duplicate detection (default auto: on for file "
                           "input).")
    call.add_argument("--dedup-mode", choices=["flag", "collapse"], default="flag",
                      help="flag (default): mark duplicates 0x400 and keep them; "
                           "collapse: keep one read per duplicate cluster.")
    call.add_argument("--snp-screen", choices=["auto", "on", "off"], default="auto",
                      help="DAF recurrent-SNP screen and mask (default auto: after a "
                           "depth preflight).")
    call.add_argument("--snp-mask", default=None, metavar="BED",
                      help="DAF: your own BED of SNP sites to exclude.")
    call.add_argument("--chimera-filter", action=argparse.BooleanOptionalAction,
                      default=True,
                      help="DAF: skip strand-swap chimeric reads (default on).")
    call.add_argument("--primary", action=argparse.BooleanOptionalAction, default=True,
                      help="Call primary alignments only (default on).")
    call.add_argument("--prob-threshold", type=int, default=None,
                      help="ML threshold override, 0-255 (default: chemistry preset).")
    call.add_argument("--use-m5c", action=argparse.BooleanOptionalAction, default=None,
                      help="DddA CpG-aware recall (default: on for ddda).")
    call.add_argument("--cpg-mask-policy", choices=["unmethylated-only", "methylated-only"],
                      default=None,
                      help="DddA CpG mask policy (default unmethylated-only).")
    call.add_argument("--call-args", default=None,
                      help='Other fiberhmm-call options, quoted as one string '
                           '(e.g. --call-args "--with-scores --min-llr 6").')
    call.add_argument("--call-mode", choices=["auto", "streaming", "resumable"],
                      default="auto",
                      help="auto (default): fiberhmm-call's resumable region-parallel "
                           "mode for genome-scale data (>=20,000 reads over at least "
                           "--cores regions), streaming for targeted runs (one "
                           "amplicon or plasmid), which it calls faster.")
    call.add_argument("--no-qc", dest="qc", action="store_false",
                      help="Skip fiberhmm-qc.")

    run = p.add_argument_group("outputs and running")
    run.add_argument("--tracks", action="store_true",
                     help="Also extract nucleosome/MSP/TF/deamination (or m6A) "
                          "tracks into OUTDIR/tracks (bigBed; BED without "
                          "bedToBigBed).")
    run.add_argument("--redo", choices=["all", "align", "call", "qc", "tracks"],
                     default=None,
                     help="Redo this step and the later ones although complete (also "
                          "needed to change the inputs or settings of an existing "
                          "OUTDIR; discards an interrupted call's resumable state).")
    run.add_argument("--progress-json", default=None, metavar="FILE",
                     help="Append JSON-lines progress events to FILE ('-' for "
                          "stdout).")
    run.add_argument("-v", "--verbose", action="store_true",
                     help="Echo the output of fiberhmm-call, -qc and -extract.")
    run.add_argument("-q", "--quiet", action="store_true",
                     help="No progress messages on stderr.")
    add_version_args(p)
    return p


def parse_args(argv=None):
    return build_parser().parse_args(argv)


def config_from_args(args):
    from fiberhmm.pipeline.runner import PipelineConfig
    return PipelineConfig(
        reads=list(args.reads),
        reference=args.reference,
        enzyme=args.enzyme,
        outdir=args.outdir,
        sample=args.sample,
        cores=max(1, args.cores),
        seq=args.seq,
        topology=args.topology,
        regions=list(args.region or []),
        min_mapq=args.min_mapq,
        min_read_length=args.min_read_length,
        hard_clip=False if args.keep_soft_clips else None,
        origin_merge=args.origin_merge,
        dedup=args.dedup,
        dedup_mode=args.dedup_mode,
        snp_screen=args.snp_screen,
        snp_mask=args.snp_mask,
        chimera_filter=args.chimera_filter,
        primary=args.primary,
        prob_threshold=args.prob_threshold,
        use_m5c=args.use_m5c,
        cpg_mask_policy=args.cpg_mask_policy,
        qc=args.qc,
        call_args=shlex.split(args.call_args or ""),
        call_mode=args.call_mode,
        tracks=args.tracks,
        aligner=args.aligner,
        force_chemistry=args.force_chemistry,
        redo=args.redo,
        verbose=args.verbose,
        quiet=args.quiet,
    )


def main(argv=None) -> int:
    args = parse_args(argv)
    from fiberhmm.pipeline.aligner import AlignerNotFound
    from fiberhmm.pipeline.progress import ProgressReporter
    from fiberhmm.pipeline.runner import (
        Pipeline,
        PipelineCancelled,
        PipelineError,
        install_cancel_handlers,
    )

    install_cancel_handlers()
    progress = ProgressReporter(args.progress_json)
    pipeline = Pipeline(config_from_args(args), progress)
    try:
        outputs = pipeline.run()
    except AlignerNotFound as exc:
        print(f"\nerror: {exc}", file=sys.stderr)
        return 2
    except PipelineCancelled as exc:
        print(f"\ncancelled. {exc.hint}", file=sys.stderr)
        return exc.exit_code
    except PipelineError as exc:
        print(f"\nerror: {exc}", file=sys.stderr)
        if exc.hint:
            print(exc.hint, file=sys.stderr)
        return 1
    finally:
        progress.close()
        pipeline.close()
    if not args.quiet:
        stats = outputs.get("stats", {})
        qc = (outputs.get("qc") or {}).get("verdicts", {})
        print("\nOutputs:", file=sys.stderr)
        for key in ("called_bam", "aligned_bam", "reference_fasta", "plasmid_map",
                    "qc_report"):
            if outputs.get(key):
                print(f"  {key.replace('_', ' '):16s} {outputs[key]}", file=sys.stderr)
        if qc:
            print(f"  QC               {qc.get('overall')}", file=sys.stderr)
        if stats.get("align", {}).get("message"):
            print(f"  reads            {stats['align']['message']}", file=sys.stderr)
        region = outputs["open"].get("region")
        print("\nOpen in FiberBrowser:\n  " + outputs["fiberbrowser_command"]
              + (f"\n  (region {region})" if region else ""), file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
