#!/usr/bin/env python3
"""Call CpG methylation domains from DAF-seq FiberHMM BAMs."""
from __future__ import annotations

import argparse
import os
import sys
from contextlib import ExitStack

import numpy as np
import pysam

from fiberhmm.cli.common import add_version_args
from fiberhmm.daf.m5c import (
    BETA_METH,
    BETA_UNMETH,
    DDDA_FIVE_PRIME_FACTORS,
    F_METH,
    M5CDomain,
    U_UNMETH,
    annotate_bam_from_domains,
    annotate_bam_per_read,
    call_domains,
    call_domains_from_emissions,
    collect_bam_observations,
    estimate_five_prime_factors,
    make_windows,
    window_log_likelihood,
    write_bed,
)


def _same_file(left: str, right: str) -> bool:
    if os.path.abspath(left) == os.path.abspath(right):
        return True
    return os.path.exists(left) and os.path.exists(right) and os.path.samefile(left, right)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="fiberhmm-call-m5c",
        description=("Opt-in genome-wide DddA DAF-seq caller: infer CpG "
                     "methylation domains from within-molecule CpG versus "
                     "non-CpG deamination contrast."),
    )
    add_version_args(parser)
    parser.add_argument("-i", "--input", nargs="+", required=True,
                        help="One or more coordinate-sorted, indexed DAF FiberHMM BAMs")
    parser.add_argument("-r", "--reference", required=True, help="Indexed reference FASTA")
    parser.add_argument("-o", "--output", required=True, help="Output BED6, or - for stdout")
    parser.add_argument("--region", required=True, help="contig[:start-end], 1-based display syntax")
    parser.add_argument(
        "--enzyme", required=True, choices=("ddda",),
        help="Required chemistry assertion. Only DddA DAF-seq is supported.",
    )
    parser.add_argument("--window", type=int, default=1000)
    parser.add_argument("--chunk-bp", type=int, default=5_000_000,
                        help="Bound observation collection in this many bp; "
                             "the HMM still runs once across the complete region")
    parser.add_argument("--min-other", type=int, default=10)
    parser.add_argument("--min-cpg", type=int, default=10)
    parser.add_argument("--posterior", type=float, default=0.99)
    parser.add_argument("--max-gap", type=int, default=1000)
    parser.add_argument("--five-prime-factors", default=None,
                        help="Comma-separated A,C,G,T factors; default calibrated DddA values")
    parser.add_argument("--estimate-factors", action="store_true",
                        help="Estimate 5' factors from the complete supplied region "
                             "(explicit high-memory audit option)")
    parser.add_argument("--tag-bam", default=None,
                        help="Optionally add domains as ddda_mcg MA spans to this BAM")
    parser.add_argument("--tag-output", default=None,
                        help="Output BAM for --tag-bam (required with --tag-bam)")
    parser.add_argument("--tag-mode", choices=("read", "locus"), default="read",
                        help="read: equal-odds per-read HMM (default); "
                             "locus: copy aggregate domains")
    parser.add_argument("--read-run-bp", type=float, default=5000.0,
                        help="Expected per-read HMM state length (default: 5000)")
    parser.add_argument("--read-posterior", type=float, default=0.99,
                        help="Per-CpG posterior required for ddda_mcg spans (default: 0.99)")
    parser.add_argument("--read-baseline-radius", type=int, default=250,
                        help="Centered non-CpG baseline radius (default: 250)")
    parser.add_argument("--read-min-run-cpg", type=int, default=2,
                        help="Consecutive supported CpGs per span (default: 2)")
    parser.add_argument("--read-max-cpg-gap", type=float, default=None,
                        help="Split read spans across longer unsupported gaps; "
                             "default: --read-run-bp")
    parser.add_argument("--tag-input-frame", choices=("auto", "molecular", "query"),
                        default="auto",
                        help="Frame of tag-BAM legacy ns/nl when MA is absent")
    parser.add_argument("--io-threads", type=int, default=4)
    return parser.parse_args(argv)


def _parse_region(text, fasta):
    if ":" not in text:
        return text, 0, fasta.get_reference_length(text)
    chrom, coords = text.rsplit(":", 1)
    length = fasta.get_reference_length(chrom)
    left, right = coords.replace(",", "").split("-", 1)
    start = max(0, int(left) - 1)
    end = min(int(right), length)
    if end <= start:
        raise ValueError("region end must exceed start")
    return chrom, start, end


def main(argv=None):
    args = parse_args(argv)
    from fiberhmm.cli.tag_m5c import _preflight_input
    for input_path in args.input:
        _preflight_input(input_path)
    if (args.window <= 0 or args.chunk_bp <= 0 or
            args.min_other < 1 or args.min_cpg < 1):
        raise SystemExit(
            "--window, --chunk-bp, --min-other and --min-cpg must be positive"
        )
    if args.chunk_bp % args.window:
        raise SystemExit("--chunk-bp must be a multiple of --window")
    if not 0.5 < args.posterior < 1.0:
        raise SystemExit("--posterior must be between 0.5 and 1")
    if args.max_gap < 0 or args.io_threads < 1:
        raise SystemExit("--max-gap must be non-negative and --io-threads positive")
    if args.read_run_bp <= 0 or args.read_baseline_radius <= 0:
        raise SystemExit("--read-run-bp and --read-baseline-radius must be positive")
    if not 0.5 < args.read_posterior < 1.0:
        raise SystemExit("--read-posterior must be between 0.5 and 1")
    if args.read_min_run_cpg < 1:
        raise SystemExit("--read-min-run-cpg must be positive")
    if args.read_max_cpg_gap is not None and args.read_max_cpg_gap <= 0:
        raise SystemExit("--read-max-cpg-gap must be positive")
    if bool(args.tag_bam) != bool(args.tag_output):
        raise SystemExit("--tag-bam and --tag-output must be supplied together")
    if args.output != "-" and any(_same_file(path, args.output) for path in args.input):
        raise SystemExit("output BED path must differ from every input BAM")
    if args.tag_bam and _same_file(args.tag_bam, args.tag_output):
        raise SystemExit("--tag-bam and --tag-output paths must differ")
    if args.tag_output and any(_same_file(path, args.tag_output) for path in args.input):
        raise SystemExit("--tag-output path must differ from every input BAM")
    if args.tag_output and args.output != "-" and _same_file(args.output, args.tag_output):
        raise SystemExit("--output BED and --tag-output BAM paths must differ")
    if args.estimate_factors and args.five_prime_factors:
        raise SystemExit("use either --estimate-factors or --five-prime-factors, not both")
    if args.five_prime_factors:
        factors = np.asarray([
            float(value) for value in args.five_prime_factors.split(",")
        ])
        if (factors.shape != (4,) or not np.all(np.isfinite(factors)) or
                np.any(factors <= 0)):
            raise SystemExit("--five-prime-factors requires four positive A,C,G,T values")
        factors /= factors.mean()
    elif args.estimate_factors:
        factors = None
    else:
        factors = DDDA_FIVE_PRIME_FACTORS / DDDA_FIVE_PRIME_FACTORS.mean()
    with pysam.FastaFile(args.reference) as fasta:
        chrom, start, end = _parse_region(args.region, fasta)
        if factors is None:
            # Estimation is an explicit two-pass/high-memory audit option. Keep
            # every observation once to estimate a single region-wide factor
            # vector; the production default and custom-factor paths below are
            # chunk bounded.
            observations = []
            offset = 0
            with ExitStack() as stack:
                bams = [
                    stack.enter_context(pysam.AlignmentFile(path, "rb"))
                    for path in args.input
                ]
                for bam in bams:
                    part, offset = collect_bam_observations(
                        bam, fasta, chrom, start, end, molecule_offset=offset,
                    )
                    observations.extend(part)
            if observations:
                factors = estimate_five_prime_factors(observations)
            windows = make_windows(
                observations, start, end, args.window, args.min_other,
                five_prime_factors=factors,
            ) if observations else []
            total_observations = len(observations)
            domains, _posterior = call_domains(
                windows, chrom, args.window, posterior_threshold=args.posterior,
                min_cpg=args.min_cpg, max_gap=args.max_gap,
            ) if windows else ([], np.empty((0, 2)))
            observed_windows = sum(w.n_cpg >= args.min_cpg for w in windows)
            total_windows = len(windows)
        else:
            starts_parts = []
            n_cpg_parts = []
            emission_parts = []
            total_observations = 0
            with ExitStack() as stack:
                bams = [
                    stack.enter_context(pysam.AlignmentFile(path, "rb"))
                    for path in args.input
                ]
                for chunk_start in range(start, end, args.chunk_bp):
                    chunk_end = min(end, chunk_start + args.chunk_bp)
                    observations = []
                    offset = 0
                    for bam in bams:
                        part, offset = collect_bam_observations(
                            bam, fasta, chrom, chunk_start, chunk_end,
                            molecule_offset=offset,
                        )
                        observations.extend(part)
                    total_observations += len(observations)
                    windows = make_windows(
                        observations, chunk_start, chunk_end, args.window,
                        args.min_other, five_prime_factors=factors,
                    )
                    starts_parts.append(np.asarray(
                        [window.start for window in windows], dtype=np.int64,
                    ))
                    n_cpg_parts.append(np.asarray(
                        [window.n_cpg for window in windows], dtype=np.int32,
                    ))
                    emission_parts.append(np.asarray([
                        [window_log_likelihood(window, BETA_UNMETH),
                         window_log_likelihood(window, BETA_METH)]
                        for window in windows
                    ]))
                    print(
                        f"m5C chunk {chrom}:{chunk_start + 1:,}-{chunk_end:,}: "
                        f"{len(observations):,} observations",
                        file=sys.stderr,
                    )
            starts = np.concatenate(starts_parts)
            n_cpg = np.concatenate(n_cpg_parts)
            emissions = np.concatenate(emission_parts)
            domains, _posterior = call_domains_from_emissions(
                emissions, starts, n_cpg, chrom, window_size=args.window,
                posterior_threshold=args.posterior, min_cpg=args.min_cpg,
                max_gap=args.max_gap,
            )
            observed_windows = int((n_cpg >= args.min_cpg).sum())
            total_windows = len(starts)
    if not total_observations:
        raise SystemExit(
            "no eligible DAF cytosine observations in the requested region; "
            "check coverage, Y/R encoding, and FiberHMM nuc/MSP tags"
        )
    domains = [
        M5CDomain(domain.chrom, max(start, domain.start), min(end, domain.end),
                  domain.methylated, domain.posterior)
        for domain in domains if domain.start < end and domain.end > start
    ]
    handle = sys.stdout if args.output == "-" else open(args.output, "w")
    try:
        write_bed(domains, handle)
    finally:
        if handle is not sys.stdout:
            handle.close()
    print(f"m5C: {total_observations:,} cytosine observations; "
          f"{observed_windows:,}/{total_windows:,} informative windows; "
          f"{len(domains):,} domains; U={U_UNMETH:.3f}, F={F_METH:.3f}; "
          f"5' A,C,G,T={np.round(factors, 3)}", file=sys.stderr)
    if args.tag_bam:
        import fiberhmm
        header_record = {
            "PN": "fiberhmm-call-m5c",
            "VN": getattr(fiberhmm, "__version__", "unknown"),
            "CL": " ".join(sys.argv),
            "DS": (f"DddA-inferred mCG tag projection; tag_mode={args.tag_mode}; "
                   "ddda_mcg_frame=molecular"),
        }
        if args.tag_mode == "locus":
            total, tagged = annotate_bam_from_domains(
                args.tag_bam, args.tag_output, domains, threads=args.io_threads,
                header_record=header_record,
            )
            print(f"m5C locus mode: tagged {tagged:,}/{total:,} reads", file=sys.stderr)
        else:
            # Keep the molecule caller on the same calibrated, custom, or
            # explicitly estimated context correction as the aggregate call.
            _preflight_input(args.tag_bam)
            stats = annotate_bam_per_read(
                args.tag_bam, args.tag_output, args.reference, factors,
                expected_run_bp=args.read_run_bp,
                posterior_threshold=args.read_posterior,
                baseline_radius=args.read_baseline_radius,
                min_other=args.min_other,
                min_call_cpg=args.read_min_run_cpg,
                max_call_gap_bp=args.read_max_cpg_gap,
                input_molecular_frame={
                    "auto": None, "molecular": True, "query": False,
                }[args.tag_input_frame],
                threads=args.io_threads,
                header_record=header_record,
            )
            print("m5C read mode: " + ", ".join(
                f"{key}={value:,}" for key, value in stats.items()
            ), file=sys.stderr)
        print(f"m5C: wrote {args.tag_output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    main()
