#!/usr/bin/env python3
"""CLI entry point for fiberhmm-daf-encode.

Reads a plain aligned BAM, identifies C->T / G->A deamination mismatches,
and encodes them as IUPAC Y/R with an st:Z tag for fiberhmm-apply --mode daf.
"""

import sys
import argparse

from fiberhmm.cli.common import add_version_args
from fiberhmm.daf.encoder import process_bam_daf_encode


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="fiberhmm-daf-encode",
        description=(
            "Call deamination mismatches from aligned DAF-seq BAM and encode "
            "as IUPAC R/Y for fiberhmm-apply --mode daf."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  # Basic usage
  fiberhmm-daf-encode -i aligned.bam -o encoded.bam

  # Pipe directly to footprint calling
  fiberhmm-daf-encode -i aligned.bam -o - | \\
      fiberhmm-apply --mode daf --streaming -i - -o output/

  # Full pipeline from alignment
  minimap2 --MD -a ref.fa reads.fq | samtools view -b | \\
      fiberhmm-daf-encode -i - -o - | \\
      fiberhmm-apply --mode daf --streaming -i - -o output/

  # Use reference FASTA when MD tags are missing
  fiberhmm-daf-encode -i aligned.bam -o encoded.bam --reference ref.fa
""",
    )

    add_version_args(parser)

    parser.add_argument(
        "-i", "--input", required=True,
        help='Input BAM file, or "-" for stdin',
    )
    parser.add_argument(
        "-o", "--output", required=True,
        help='Output BAM path, or "-" for stdout',
    )
    parser.add_argument(
        "--reference", default=None,
        help="Reference FASTA (fallback if MD tag is missing)",
    )
    parser.add_argument(
        "-q", "--min-mapq", type=int, default=20,
        help="Minimum mapping quality (default: 20)",
    )
    parser.add_argument(
        "--min-read-length", type=int, default=1000,
        help="Minimum aligned read length in bp (default: 1000)",
    )
    parser.add_argument(
        "--io-threads", type=int, default=4,
        help="htslib I/O threads for BAM compression/decompression (default: 4)",
    )
    parser.add_argument(
        "--strand", choices=["CT", "GA", "auto"], default="auto",
        help='Force conversion strand: CT (+ strand), GA (- strand), '
             'or auto (per-read consensus, default: auto)',
    )

    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    force_strand = None if args.strand == "auto" else args.strand

    process_bam_daf_encode(
        input_bam=args.input,
        output_bam=args.output,
        reference=args.reference,
        min_mapq=args.min_mapq,
        min_read_length=args.min_read_length,
        io_threads=args.io_threads,
        force_strand=force_strand,
    )


if __name__ == "__main__":
    main()
