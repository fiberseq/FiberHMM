#!/usr/bin/env python3
"""fiberhmm-qc — bounded signal, periodicity, and footprint QC.

The command samples a coordinate-indexed BAM through deterministic random
genomic windows. Unindexed BAMs use a capped reservoir over a bounded prefix.
It never performs a full-file rescan and never performs deduplication.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

from fiberhmm.qc.core import (
    DEFAULT_SAMPLE_READS,
    DEFAULT_SEED,
    load_references,
    run_multi_qc,
    run_qc,
)


def parse_args(argv=None):
    profiles = sorted(load_references()["profiles"])
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        nargs="+",
        action="append",
        metavar="BAM",
        help="One or more FiberHMM-compatible BAM/CRAM files; -i may be repeated",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default=None,
        help="QC output directory (default: qc/ beside the input BAMs)",
    )
    parser.add_argument(
        "--output-prefix",
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--mode", choices=["auto", "daf", "pacbio-fiber", "nanopore-fiber"], default="auto", help="Observation mode (default: infer from @PG/tags)")
    parser.add_argument("--enzyme", choices=["auto", "hia5", "ddda", "dddb"], default="auto", help="Enzyme (default: infer from @PG)")
    parser.add_argument("--reference-profile", choices=["auto", "none", *profiles], default="auto", help="Empirical QC reference (default: assay-aware auto selection)")
    parser.add_argument("--reference", default=None, help="Indexed FASTA fallback for raw DAF BAMs lacking MD/R/Y")
    parser.add_argument("--sample-reads", type=int, default=DEFAULT_SAMPLE_READS, help=f"Target bounded sample size (default {DEFAULT_SAMPLE_READS:,})")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help=f"Deterministic sampler seed (default {DEFAULT_SEED})")
    parser.add_argument("--min-mapq", type=int, default=20, help="Minimum mapping quality (default 20)")
    parser.add_argument("--prob-threshold", type=int, default=None, help="Minimum MM/ML probability, 0-255 (default: 248 for Hia5 Nanopore, the threshold its QC reference is calibrated at; 125 otherwise)")
    parser.add_argument("--min-opportunities", type=int, default=200, help="Minimum target sites per read for rate QC (default 200)")
    parser.add_argument("--snp-mask", default=None, help="Applied DAF SNP-mask BED to summarize (single input only)")
    parser.add_argument("--snp-report", default=None, help="fiberhmm-daf-snps JSON to plot (single input only)")
    parser.add_argument("--fail-on-qc", action="store_true", help="Exit 2 when the final status is FAIL")
    from fiberhmm.cli.common import add_version_args
    add_version_args(parser)
    return parser.parse_args(argv)


def _input_paths(groups) -> list[str]:
    return [path for group in groups for path in group]


def resolve_output_dir(input_paths: list[str], output_dir: Optional[str]) -> Path:
    """Resolve the predictable default and reject ambiguous multi-directory input."""
    if output_dir:
        return Path(output_dir)
    parents = {Path(path).resolve().parent for path in input_paths}
    if len(parents) != 1:
        raise ValueError(
            "inputs from different directories require -o/--output-dir"
        )
    return next(iter(parents)) / "qc"


def _stem(path: str) -> str:
    name = Path(path).name
    for suffix in (".bam", ".cram"):
        if name.lower().endswith(suffix):
            return name[: -len(suffix)]
    return name


def main(argv=None):
    args = parse_args(argv)
    input_paths = _input_paths(args.input)
    try:
        output_dir = resolve_output_dir(input_paths, args.output_dir)
    except ValueError as exc:
        print(f"fiberhmm-qc: error: {exc}", file=sys.stderr)
        return 2
    if args.output_prefix and len(input_paths) != 1:
        print(
            "fiberhmm-qc: error: legacy --output-prefix is valid only for one input; "
            "use -o/--output-dir for multiple inputs",
            file=sys.stderr,
        )
        return 2
    if (args.snp_mask or args.snp_report) and len(input_paths) != 1:
        print(
            "fiberhmm-qc: error: --snp-mask/--snp-report currently require one input",
            file=sys.stderr,
        )
        return 2

    common = {
        "mode": args.mode,
        "enzyme": args.enzyme,
        "reference_profile": args.reference_profile,
        "reference_fasta": args.reference,
        "sample_reads": args.sample_reads,
        "seed": args.seed,
        "min_mapq": args.min_mapq,
        "prob_threshold": args.prob_threshold,
        "min_opportunities": args.min_opportunities,
    }
    if len(input_paths) == 1:
        prefix = args.output_prefix or str(output_dir / _stem(input_paths[0]))
        result = run_qc(
            input_path=input_paths[0],
            output_prefix=prefix,
            snp_mask_path=args.snp_mask,
            snp_report_path=args.snp_report,
            **common,
        )
    else:
        result = run_multi_qc(
            input_paths=input_paths,
            output_dir=str(output_dir),
            **common,
        )
    if args.fail_on_qc and result["overall"]["status"] == "FAIL":
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
