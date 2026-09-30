#!/usr/bin/env python3
"""Call recurrent opposite-conversion SNPs in DAF-seq BAMs."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from fiberhmm.daf.snps import (
    DEFAULT_SNP_MIN_ALT_FIBERS,
    DEFAULT_SNP_MIN_DEPTH,
    DEFAULT_SNP_MIN_FRACTION,
    call_opposite_conversion_snps,
    write_snp_outputs,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-i", "--input", required=True, help="Input aligned BAM with MD tags")
    parser.add_argument("-o", "--output-prefix", default=None, help="Output prefix (default: <BAM stem>.daf_snps)")
    parser.add_argument(
        "--min-fraction",
        type=float,
        default=DEFAULT_SNP_MIN_FRACTION,
        help=(
            "Minimum mismatch fraction in each direction "
            f"(validated default {DEFAULT_SNP_MIN_FRACTION:g})"
        ),
    )
    parser.add_argument(
        "--min-depth",
        type=int,
        default=DEFAULT_SNP_MIN_DEPTH,
        help=(
            "Minimum depth in each conversion-direction class "
            f"(validated default {DEFAULT_SNP_MIN_DEPTH}; all thresholds "
            "must pass in both classes)"
        ),
    )
    parser.add_argument(
        "--min-alt-fibers",
        type=int,
        default=DEFAULT_SNP_MIN_ALT_FIBERS,
        help=(
            "Minimum mismatch-supporting fibers in each direction "
            f"(validated default {DEFAULT_SNP_MIN_ALT_FIBERS})"
        ),
    )
    parser.add_argument("--min-dominant-events", type=int, default=5, help="Minimum dominant conversions per classifiable fiber (default 5)")
    parser.add_argument("--min-dominant-purity", type=float, default=0.80, help="Minimum dominant-direction purity (default 0.80)")
    parser.add_argument("--min-mapq", type=int, default=20, help="Minimum mapping quality (default 20)")
    parser.add_argument("--min-amplicon-reads", type=int, default=20, help="Minimum aligned reads required for an amplicon consensus (default 20)")
    parser.add_argument("--reference", default=None, help="Indexed FASTA fallback for BAMs lacking MD")
    from fiberhmm.cli.common import add_version_args
    add_version_args(parser)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    input_path = Path(args.input)
    prefix = args.output_prefix or str(input_path.with_suffix("")) + ".daf_snps"
    payload = call_opposite_conversion_snps(
        args.input,
        min_fraction=args.min_fraction,
        min_depth=args.min_depth,
        min_alt_fibers=args.min_alt_fibers,
        min_dominant_events=args.min_dominant_events,
        min_dominant_purity=args.min_dominant_purity,
        min_mapq=args.min_mapq,
        reference_fasta=args.reference,
        min_amplicon_reads=args.min_amplicon_reads,
    )
    payload = write_snp_outputs(payload, prefix)
    print(
        f"FiberHMM DAF SNP mask: {payload['n_called_snps']:,} sites\n"
        f"  BED mask: {payload['outputs']['bed']}\n"
        f"  VCF:      {payload['outputs']['vcf']}\n"
        f"  amplicons: {payload['outputs']['amplicons_tsv']}\n"
        f"  report:   {payload['outputs']['json']}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
