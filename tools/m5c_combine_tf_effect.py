#!/usr/bin/env python3
"""Combine whole-genome ``m5c_tf_effect.py`` metric shards."""
from __future__ import annotations

import argparse
import math
from collections import defaultdict
from pathlib import Path


def read_metrics(path: Path):
    values = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        key, value = line.split("\t", 1)
        try:
            number = float(value)
        except ValueError:
            continue
        if math.isfinite(number):
            values[key] = number
    return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    totals = defaultdict(float)
    score_shards = 0
    for path in args.inputs:
        metrics = read_metrics(path)
        score_shards += int("matched_llr_changed" in metrics)
        for key, value in metrics.items():
            # Region-only truth means are not additive and whole-genome shards
            # intentionally leave them NaN/zero.
            if key.endswith("_truth_mean") or key.endswith("_truth_gt_0.5"):
                continue
            totals[key] += value

    def ratio(numerator, denominator):
        return totals[numerator] / totals[denominator] if totals[denominator] else math.nan

    derived = {
        "input_shards": len(args.inputs),
        "score_metric_shards": score_shards,
        "tagged_read_fraction": ratio("tagged_reads", "reads"),
        "tf_net_change": totals["tf_on"] - totals["tf_off"],
        "tf_net_fraction": ratio("tf_on", "tf_off") - 1.0,
        "overlap_net_change": (
            totals["tf_on_overlapping_m5c"] - totals["tf_off_overlapping_m5c"]
        ),
        "overlap_net_fraction": (
            ratio("tf_on_overlapping_m5c", "tf_off_overlapping_m5c") - 1.0
        ),
        "removed_overlap_fraction": ratio(
            "removed_overlapping_m5c", "removed",
        ),
        "mean_changed_matched_llr_delta": ratio(
            "matched_llr_delta_sum", "matched_llr_changed",
        ),
        "matched_tq_change_fraction": ratio(
            "matched_tq_changed", "matched_llr_changed",
        ),
    }
    rows = list(totals.items()) + list(derived.items())
    text = "".join(f"{key}\t{value}\n" for key, value in rows)
    print(text, end="")
    if args.output:
        args.output.write_text(text)


if __name__ == "__main__":
    main()
