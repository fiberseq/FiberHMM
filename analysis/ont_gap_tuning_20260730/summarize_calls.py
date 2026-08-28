#!/usr/bin/env python3
"""Summarize FiberHMM nucleosome/MSP geometry on the matched tuning subsets.

The BAM tags are in molecular read coordinates, but interval lengths and
coverage fractions are invariant to strand, so no reference-coordinate
projection is needed here.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import pysam


REGIONS = (
    ("chr2L", 3_000_000, 3_050_000, "train", "chr2L_03"),
    ("chr2L", 7_000_000, 7_050_000, "holdout", "chr2L_07"),
    ("chr2R", 8_000_000, 8_050_000, "train", "chr2R_08"),
    ("chr2R", 12_000_000, 12_050_000, "holdout", "chr2R_12"),
    ("chr3L", 14_000_000, 14_050_000, "train", "chr3L_14"),
    ("chr3L", 19_000_000, 19_050_000, "holdout", "chr3L_19"),
    ("chr3R", 23_000_000, 23_050_000, "train", "chr3R_23"),
    ("chr3R", 29_000_000, 29_050_000, "holdout", "chr3R_29"),
)


def parse_input(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--input must be LABEL=PATH")
    label, path = value.split("=", 1)
    if not label:
        raise argparse.ArgumentTypeError("input label cannot be empty")
    return label, Path(path)


def clipped_intervals(starts, lengths, read_length: int, floor: int = 1):
    out = []
    for start_raw, length_raw in zip(starts, lengths):
        start = max(0, int(start_raw))
        end = min(read_length, start + int(length_raw))
        if end - start >= floor:
            out.append((start, end))
    return out


def union_bp(intervals: Iterable[tuple[int, int]]) -> int:
    ordered = sorted(intervals)
    if not ordered:
        return 0
    total = 0
    start, end = ordered[0]
    for next_start, next_end in ordered[1:]:
        if next_start <= end:
            end = max(end, next_end)
        else:
            total += end - start
            start, end = next_start, next_end
    return total + end - start


def assign_region(read) -> tuple[str, str]:
    chrom = read.reference_name
    read_start = int(read.reference_start)
    read_end = int(read.reference_end or read_start)
    best = None
    best_overlap = 0
    for region_chrom, start, end, split, name in REGIONS:
        if region_chrom != chrom:
            continue
        overlap = max(0, min(read_end, end) - max(read_start, start))
        if overlap > best_overlap:
            best = (split, name)
            best_overlap = overlap
    return best or ("unassigned", "unassigned")


@dataclass
class Accumulator:
    reads: int = 0
    query_bp: int = 0
    nuc_bp: int = 0
    msp_bp: int = 0
    annotated_bp: int = 0
    nuc_count: int = 0
    msp_count: int = 0
    nuc_lengths: list[int] = field(default_factory=list)
    msp_lengths: list[int] = field(default_factory=list)
    per_read_nuc_fraction: list[float] = field(default_factory=list)
    per_read_msp_fraction: list[float] = field(default_factory=list)

    def add(
        self,
        read_length: int,
        nuc_intervals: list[tuple[int, int]],
        msp_intervals: list[tuple[int, int]],
    ) -> tuple[float, float]:
        nuc_bp = union_bp(nuc_intervals)
        msp_bp = union_bp(msp_intervals)
        annotated_bp = union_bp(nuc_intervals + msp_intervals)
        nuc_fraction = nuc_bp / read_length
        msp_fraction = msp_bp / read_length

        self.reads += 1
        self.query_bp += read_length
        self.nuc_bp += nuc_bp
        self.msp_bp += msp_bp
        self.annotated_bp += annotated_bp
        self.nuc_count += len(nuc_intervals)
        self.msp_count += len(msp_intervals)
        self.nuc_lengths.extend(end - start for start, end in nuc_intervals)
        self.msp_lengths.extend(end - start for start, end in msp_intervals)
        self.per_read_nuc_fraction.append(nuc_fraction)
        self.per_read_msp_fraction.append(msp_fraction)
        return nuc_fraction, msp_fraction


def quantile(values: list[float] | list[int], q: float) -> float:
    if not values:
        return float("nan")
    return float(np.quantile(np.asarray(values), q))


def summarize(acc: Accumulator) -> dict[str, float | int]:
    qbp = max(1, acc.query_bp)
    return {
        "reads": acc.reads,
        "query_bp": acc.query_bp,
        "nuc_fraction": acc.nuc_bp / qbp,
        "msp_fraction": acc.msp_bp / qbp,
        "unannotated_fraction": 1.0 - (acc.annotated_bp / qbp),
        "nuc_per_kb": 1000.0 * acc.nuc_count / qbp,
        "msp_per_kb": 1000.0 * acc.msp_count / qbp,
        "per_read_nuc_fraction_median": quantile(acc.per_read_nuc_fraction, 0.5),
        "per_read_msp_fraction_median": quantile(acc.per_read_msp_fraction, 0.5),
        "nuc_length_p25": quantile(acc.nuc_lengths, 0.25),
        "nuc_length_median": quantile(acc.nuc_lengths, 0.5),
        "nuc_length_p75": quantile(acc.nuc_lengths, 0.75),
        "nuc_length_p90": quantile(acc.nuc_lengths, 0.9),
        "msp_length_p25": quantile(acc.msp_lengths, 0.25),
        "msp_length_median": quantile(acc.msp_lengths, 0.5),
        "msp_length_p75": quantile(acc.msp_lengths, 0.75),
        "msp_length_p90": quantile(acc.msp_lengths, 0.9),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        type=parse_input,
        metavar="LABEL=PATH",
    )
    parser.add_argument("--msp-floor", type=int, default=60)
    parser.add_argument("--min-read-length", type=int, default=1000)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--per-read", type=Path, required=True)
    args = parser.parse_args()

    if args.msp_floor < 1 or args.min_read_length < 1:
        parser.error("size floors must be positive")

    summary_rows = []
    per_read_rows = []
    for label, bam_path in args.input:
        if not bam_path.is_file():
            raise FileNotFoundError(bam_path)
        groups: dict[str, Accumulator] = defaultdict(Accumulator)
        with pysam.AlignmentFile(bam_path, "rb") as bam:
            for read in bam.fetch(until_eof=True):
                read_length = int(read.query_length or 0)
                if read_length < args.min_read_length:
                    continue
                if not (read.has_tag("ns") or read.has_tag("as")):
                    continue

                ns = read.get_tag("ns") if read.has_tag("ns") else ()
                nl = read.get_tag("nl") if read.has_tag("nl") else ()
                starts = read.get_tag("as") if read.has_tag("as") else ()
                lengths = read.get_tag("al") if read.has_tag("al") else ()
                nuc_intervals = clipped_intervals(ns, nl, read_length)
                msp_intervals = clipped_intervals(
                    starts, lengths, read_length, floor=args.msp_floor
                )
                split, region = assign_region(read)

                for group in ("overall", split, region):
                    groups[group].add(read_length, nuc_intervals, msp_intervals)

                nuc_bp = union_bp(nuc_intervals)
                msp_bp = union_bp(msp_intervals)
                per_read_rows.append(
                    {
                        "label": label,
                        "split": split,
                        "region": region,
                        "read_id": read.query_name,
                        "read_length": read_length,
                        "nuc_fraction": nuc_bp / read_length,
                        "msp_fraction": msp_bp / read_length,
                        "nuc_count": len(nuc_intervals),
                        "msp_count": len(msp_intervals),
                    }
                )

        ordered_groups = ["overall", "train", "holdout"] + [
            region[-1] for region in REGIONS
        ]
        for group in ordered_groups:
            if group not in groups:
                continue
            row = {"label": label, "group": group, **summarize(groups[group])}
            summary_rows.append(row)

    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.per_read.parent.mkdir(parents=True, exist_ok=True)
    with args.summary.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(summary_rows[0]), delimiter="\t"
        )
        writer.writeheader()
        writer.writerows(summary_rows)
    with args.per_read.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(per_read_rows[0]), delimiter="\t"
        )
        writer.writeheader()
        writer.writerows(per_read_rows)


if __name__ == "__main__":
    main()
