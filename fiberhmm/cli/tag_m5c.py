#!/usr/bin/env python3
"""Add molecule-specific DddA-inferred mCG spans with a per-read HMM."""
from __future__ import annotations

import argparse
import os
import re
import sys

import numpy as np

from fiberhmm.cli.common import add_version_args
from fiberhmm.daf.m5c import (
    DDDA_FIVE_PRIME_FACTORS,
    annotate_bam_per_read,
    estimate_bam_five_prime_factors,
    ma_group_feature,
)


def _declared_enzymes(header) -> set[str]:
    """Find enzyme names explicitly recorded in BAM @PG/@CO provenance."""
    data = header.to_dict() if hasattr(header, "to_dict") else dict(header)
    text = "\n".join([
        *(str(value) for value in data.get("CO", [])),
        *(" ".join(str(value) for value in record.values())
          for record in data.get("PG", [])),
    ]).lower()
    return {
        enzyme for enzyme in ("ddda", "dddb", "hia5", "ecogii", "sssi")
        if re.search(rf"(?<![a-z0-9]){enzyme}(?![a-z0-9])", text)
    }


def _preflight_input(path: str, max_primary_reads: int = 5000) -> None:
    """Require DAF evidence/structure and reject known non-DddA provenance."""
    if path == "-":
        raise SystemExit(
            "fiberhmm-tag-m5c requires a seekable R/Y-encoded BAM, not stdin"
        )
    import pysam

    seen = 0
    has_iupac = False
    has_structure = False
    with pysam.AlignmentFile(path, "rb", check_sq=False) as bam:
        declared = _declared_enzymes(bam.header)
        if declared and declared != {"ddda"}:
            labels = ", ".join(sorted(declared))
            raise SystemExit(
                "DddA mCG calling rejected this BAM: its header declares "
                f"incompatible or mixed enzyme provenance ({labels}). "
                "Only DddA DAF-seq is supported."
            )
        for read in bam:
            if read.is_unmapped or read.is_secondary or read.is_supplementary:
                continue
            seen += 1
            sequence = read.query_sequence or ""
            has_iupac |= "Y" in sequence or "R" in sequence
            has_structure |= (
                (read.has_tag("MA") and any(
                    ma_group_feature(group) in {"nuc", "msp", "tf"}
                    for group in str(read.get_tag("MA")).split(";")[1:]
                ))
                or (read.has_tag("ns") and read.has_tag("nl"))
                or (read.has_tag("as") and read.has_tag("al"))
            )
            if has_iupac and has_structure:
                return
            if seen >= max_primary_reads:
                break
    if not seen:
        raise SystemExit("input BAM contains no mapped primary reads")
    if not has_iupac:
        raise SystemExit(
            "input BAM has no Y/R-encoded DAF sequence in the first "
            f"{seen:,} primary reads; run fiberhmm-daf-encode before apply/tagging"
        )
    raise SystemExit(
        "input BAM has no FiberHMM nuc/MSP structure in the first "
        f"{seen:,} primary reads; run fiberhmm-apply before fiberhmm-tag-m5c"
    )


def _same_file(left: str, right: str) -> bool:
    if os.path.abspath(left) == os.path.abspath(right):
        return True
    return os.path.exists(left) and os.path.exists(right) and os.path.samefile(left, right)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="fiberhmm-tag-m5c",
        description=("Opt-in genome-wide DddA DAF-seq caller: infer "
                     "molecule-specific mCG state along ordered CpGs and add "
                     "confident runs as ddda_mcg MA spans."),
    )
    add_version_args(parser)
    parser.add_argument("-i", "--input", required=True,
                        help="R/Y-encoded, FiberHMM-tagged coordinate BAM")
    parser.add_argument("-o", "--output", required=True, help="Output BAM")
    parser.add_argument("-r", "--reference", required=True, help="Reference FASTA")
    parser.add_argument(
        "--enzyme", required=True, choices=("ddda",),
        help="Required chemistry assertion. Only DddA DAF-seq is supported.",
    )
    parser.add_argument("--run-bp", type=float, default=5000.0)
    parser.add_argument("--posterior", type=float, default=0.99)
    parser.add_argument("--baseline-radius", type=int, default=250)
    parser.add_argument("--min-other", type=int, default=10)
    parser.add_argument("--min-run-cpg", type=int, default=2)
    parser.add_argument("--max-cpg-gap", type=float, default=None,
                        help="Split output spans across longer evidence-free gaps; "
                             "default: --run-bp")
    parser.add_argument("--five-prime-factors", default=None,
                        help="Comma-separated A,C,G,T factors; default calibrated DddA values")
    parser.add_argument("--estimate-factors", action="store_true",
                        help="Estimate 5' factors from this BAM instead of using calibrated values")
    parser.add_argument("--factor-sample-reads", type=int, default=5000)
    parser.add_argument("--input-frame", choices=("auto", "molecular", "query"),
                        default="auto",
                        help="Frame of legacy ns/nl tags when MA is absent; "
                             "auto uses the FiberHMM header marker")
    parser.add_argument("--io-threads", type=int, default=4)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.run_bp <= 0 or args.baseline_radius <= 0:
        raise SystemExit("--run-bp and --baseline-radius must be positive")
    if not 0.5 < args.posterior < 1.0:
        raise SystemExit("--posterior must be between 0.5 and 1")
    if args.min_other < 1 or args.min_run_cpg < 1:
        raise SystemExit("--min-other and --min-run-cpg must be positive")
    if args.max_cpg_gap is not None and args.max_cpg_gap <= 0:
        raise SystemExit("--max-cpg-gap must be positive")
    if args.factor_sample_reads < 1 or args.io_threads < 1:
        raise SystemExit("--factor-sample-reads and --io-threads must be positive")
    if _same_file(args.input, args.output):
        raise SystemExit("input and output BAM paths must differ")
    input_molecular_frame = {
        "auto": None, "molecular": True, "query": False,
    }[args.input_frame]
    _preflight_input(args.input)
    if args.estimate_factors and args.five_prime_factors:
        raise SystemExit("use either --estimate-factors or --five-prime-factors, not both")
    if args.estimate_factors:
        factors = estimate_bam_five_prime_factors(
            args.input, args.reference, args.factor_sample_reads, args.io_threads,
            input_molecular_frame=input_molecular_frame,
        )
    elif args.five_prime_factors:
        factors = np.asarray([float(value) for value in args.five_prime_factors.split(",")])
        if (factors.shape != (4,) or not np.all(np.isfinite(factors)) or
                np.any(factors <= 0)):
            raise SystemExit("--five-prime-factors requires four positive A,C,G,T values")
        factors /= factors.mean()
    else:
        factors = DDDA_FIVE_PRIME_FACTORS / DDDA_FIVE_PRIME_FACTORS.mean()
    print(f"[tag_m5c] 5' factors A,C,G,T={np.round(factors, 3)}", file=sys.stderr)
    import fiberhmm
    header_record = {
        "PN": "fiberhmm-tag-m5c",
        "VN": getattr(fiberhmm, "__version__", "unknown"),
        "CL": " ".join(sys.argv),
        "DS": ("DddA molecule-specific mCG HMM; ddda_mcg_frame=molecular; "
               f"run_bp={args.run_bp} posterior={args.posterior} "
               f"baseline_radius={args.baseline_radius} "
               f"min_other={args.min_other} min_run_cpg={args.min_run_cpg} "
               "five_prime_factors=" + ",".join(f"{value:.6g}" for value in factors)),
    }
    stats = annotate_bam_per_read(
        args.input, args.output, args.reference, factors,
        expected_run_bp=args.run_bp, posterior_threshold=args.posterior,
        baseline_radius=args.baseline_radius, min_other=args.min_other,
        min_call_cpg=args.min_run_cpg, max_call_gap_bp=args.max_cpg_gap,
        input_molecular_frame=input_molecular_frame,
        threads=args.io_threads,
        header_record=header_record,
    )
    print("[tag_m5c] " + ", ".join(
        f"{key}={value:,}" for key, value in stats.items()
    ), file=sys.stderr)
    return 0


if __name__ == "__main__":
    main()
