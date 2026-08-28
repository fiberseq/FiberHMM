#!/usr/bin/env python3
"""Measure ML>=threshold m6A contrast in supplied FiberHMM-tagged BAMs."""

from __future__ import annotations

import argparse
import bisect
import csv
from pathlib import Path

import pysam

from fiberhmm.inference.engine import _extract_fiber_read_from_pysam
from fiberhmm.io.ma_tags import flip_intervals_to_seq


VALID_MODES = {"nanopore-fiber", "pacbio-fiber"}


def parse_input(value: str) -> tuple[str, str, Path]:
    if "=" not in value or "|" not in value.split("=", 1)[0]:
        raise argparse.ArgumentTypeError(
            "--input must be LABEL|MODE=PATH"
        )
    label_mode, path = value.split("=", 1)
    label, mode = label_mode.rsplit("|", 1)
    if mode not in VALID_MODES:
        raise argparse.ArgumentTypeError(
            f"mode must be one of {sorted(VALID_MODES)}"
        )
    return label, mode, Path(path)


def interval_counts(
    sequence: str,
    positions: list[int],
    starts,
    lengths,
    read,
    target_bases: tuple[str, ...],
) -> tuple[int, int]:
    starts, lengths = flip_intervals_to_seq(starts, lengths, read)
    hit_count = 0
    opportunity_count = 0
    for start_raw, length_raw in zip(starts, lengths):
        start = max(0, int(start_raw))
        end = min(len(sequence), start + int(length_raw))
        if end <= start:
            continue
        hit_count += (
            bisect.bisect_left(positions, end)
            - bisect.bisect_left(positions, start)
        )
        subsequence = sequence[start:end]
        opportunity_count += sum(subsequence.count(base) for base in target_bases)
    return hit_count, opportunity_count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        action="append",
        type=parse_input,
        required=True,
        metavar="LABEL|MODE=PATH",
    )
    parser.add_argument("--threshold", type=int, default=248)
    parser.add_argument("--min-read-length", type=int, default=1000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 0 <= args.threshold <= 255:
        parser.error("--threshold must be in [0,255]")

    rows = []
    for label, mode, bam_path in args.input:
        if not bam_path.is_file():
            raise FileNotFoundError(bam_path)
        counts = {
            "all": [0, 0],
            "nuc": [0, 0],
            "msp": [0, 0],
        }
        event_count = 0
        isolated = {20: 0, 40: 0, 60: 0}
        read_count = 0
        with pysam.AlignmentFile(bam_path, "rb") as bam:
            for read in bam.fetch(until_eof=True):
                read_length = int(read.query_length or 0)
                if read_length < args.min_read_length:
                    continue
                if not (read.has_tag("ns") or read.has_tag("as")):
                    continue
                fiber_read = _extract_fiber_read_from_pysam(
                    read, mode, args.threshold
                )
                if fiber_read is None:
                    continue

                sequence = fiber_read["query_sequence"].upper()
                positions = sorted(fiber_read["m6a_query_positions"])
                if mode == "nanopore-fiber":
                    target_bases = ("T",) if read.is_reverse else ("A",)
                else:
                    target_bases = ("A", "T")

                read_count += 1
                counts["all"][0] += len(positions)
                counts["all"][1] += sum(
                    sequence.count(base) for base in target_bases
                )
                event_count += len(positions)
                for index, position in enumerate(positions):
                    left = (
                        position - positions[index - 1]
                        if index
                        else 10**9
                    )
                    right = (
                        positions[index + 1] - position
                        if index + 1 < len(positions)
                        else 10**9
                    )
                    nearest = min(left, right)
                    for distance in isolated:
                        isolated[distance] += nearest > distance

                for kind, start_tag, length_tag in (
                    ("nuc", "ns", "nl"),
                    ("msp", "as", "al"),
                ):
                    starts = read.get_tag(start_tag) if read.has_tag(start_tag) else ()
                    lengths = (
                        read.get_tag(length_tag)
                        if read.has_tag(length_tag)
                        else ()
                    )
                    hits, opportunities = interval_counts(
                        sequence,
                        positions,
                        starts,
                        lengths,
                        read,
                        target_bases,
                    )
                    counts[kind][0] += hits
                    counts[kind][1] += opportunities

        nuc_rate = counts["nuc"][0] / max(1, counts["nuc"][1])
        msp_rate = counts["msp"][0] / max(1, counts["msp"][1])
        rows.append(
            {
                "label": label,
                "mode": mode,
                "threshold": args.threshold,
                "reads": read_count,
                "all_hits": counts["all"][0],
                "all_opportunities": counts["all"][1],
                "all_hit_rate": counts["all"][0] / max(1, counts["all"][1]),
                "nuc_hit_rate": nuc_rate,
                "msp_hit_rate": msp_rate,
                "msp_to_nuc_rate_ratio": msp_rate / max(1e-12, nuc_rate),
                "isolated_gt20_fraction": isolated[20] / max(1, event_count),
                "isolated_gt40_fraction": isolated[40] / max(1, event_count),
                "isolated_gt60_fraction": isolated[60] / max(1, event_count),
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
