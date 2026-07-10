#!/usr/bin/env python3
"""Combine split ``m5c_genome.py`` runs and summarize their validation."""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import numpy as np


PRIMARY = re.compile(
    r"^chr(?P<number>[1-9]|1[0-9]|2[0-2]|X|Y)_(?P<hap>MATERNAL|PATERNAL)$"
)


def contig_key(contig: str):
    match = PRIMARY.match(contig)
    if not match:
        return (99, contig, 99)
    label = match.group("number")
    number = int(label) if label.isdigit() else {"X": 23, "Y": 24}[label]
    haplotype = {"MATERNAL": 0, "PATERNAL": 1}[match.group("hap")]
    return number, haplotype, contig


def read_metrics(path: Path):
    with path.open() as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def number(row, key, default=np.nan):
    value = row.get(key, "")
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", action="append", required=True,
                        help="Split-run directory; repeat for every shard")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    inputs = [Path(value) for value in args.input_dir]
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    beds = {}
    metrics = {}
    for directory in inputs:
        for path in directory.glob("*.m5c.bed"):
            if path.name == "genome.m5c.bed":
                continue
            contig = path.name.removesuffix(".m5c.bed")
            if contig in beds:
                raise SystemExit(f"duplicate BED for {contig}: {beds[contig]} and {path}")
            beds[contig] = path
        metrics_path = directory / "metrics.tsv"
        if metrics_path.exists():
            for row in read_metrics(metrics_path):
                contig = row["chrom"]
                if contig in metrics:
                    raise SystemExit(f"duplicate metrics for {contig}")
                metrics[contig] = row
    if not beds:
        raise SystemExit("no per-contig .m5c.bed files found")
    missing = sorted(set(beds) - set(metrics), key=contig_key)
    if missing:
        raise SystemExit("missing completed metrics for: " + ", ".join(missing))

    contigs = sorted(beds, key=contig_key)
    coverage = {"m5c_methylated": 0, "m5c_unmethylated": 0}
    interval_count = {key: 0 for key in coverage}
    with (output / "genome.m5c.bed").open("w") as sink:
        for contig in contigs:
            with beds[contig].open() as source:
                for line in source:
                    if not line.strip():
                        continue
                    fields = line.rstrip().split("\t")
                    feature = fields[3]
                    if feature in coverage:
                        coverage[feature] += int(fields[2]) - int(fields[1])
                        interval_count[feature] += 1
                    sink.write(line)

    keys = sorted({key for row in metrics.values() for key in row})
    with (output / "metrics.tsv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, delimiter="\t",
                                lineterminator="\n")
        writer.writeheader()
        for contig in contigs:
            writer.writerow(metrics[contig])

    rows = [metrics[contig] for contig in contigs]
    truth_weight = np.asarray([number(row, "truth_windows", 0) for row in rows])
    rho = np.asarray([number(row, "rho") for row in rows])
    auc = np.asarray([number(row, "auc_extreme") for row in rows])
    valid_rho = np.isfinite(rho) & (truth_weight > 0)
    valid_auc = np.isfinite(auc) & (truth_weight > 0)
    truth_records = sum(number(row, "truth_records", 0) for row in rows)
    truth_matched = sum(number(row, "truth_matched", 0) for row in rows)
    genome_length = sum(number(row, "length", 0) for row in rows)
    methylated_windows = np.asarray([
        number(row, "methylated_windows", 0) for row in rows
    ])
    unmethylated_windows = np.asarray([
        number(row, "unmethylated_windows", 0) for row in rows
    ])

    def weighted_metric(key, weights):
        values = np.asarray([number(row, key) for row in rows])
        valid = np.isfinite(values) & (weights > 0)
        return float(np.average(values[valid], weights=weights[valid])) \
            if valid.any() else np.nan

    summary = {
        "contigs": len(contigs),
        "genome_bp": int(genome_length),
        "windows": int(sum(number(row, "windows", 0) for row in rows)),
        "informative_windows": int(sum(number(row, "informative", 0) for row in rows)),
        "domains": int(sum(number(row, "domains", 0) for row in rows)),
        "methylated_intervals": interval_count["m5c_methylated"],
        "methylated_bp": coverage["m5c_methylated"],
        "methylated_fraction": coverage["m5c_methylated"] / genome_length,
        "unmethylated_intervals": interval_count["m5c_unmethylated"],
        "unmethylated_bp": coverage["m5c_unmethylated"],
        "unmethylated_fraction": coverage["m5c_unmethylated"] / genome_length,
        "truth_records": int(truth_records),
        "truth_matched": int(truth_matched),
        "truth_match_fraction": truth_matched / truth_records,
        "truth_windows": int(truth_weight.sum()),
        "methylated_truth_windows": int(methylated_windows.sum()),
        "methylated_truth_mean": weighted_metric(
            "methylated_truth_mean", methylated_windows,
        ),
        "methylated_precision": weighted_metric(
            "methylated_precision", methylated_windows,
        ),
        "unmethylated_truth_windows": int(unmethylated_windows.sum()),
        "unmethylated_truth_mean": weighted_metric(
            "unmethylated_truth_mean", unmethylated_windows,
        ),
        "unmethylated_precision": weighted_metric(
            "unmethylated_precision", unmethylated_windows,
        ),
        "window_weighted_contig_rho": float(np.average(
            rho[valid_rho], weights=truth_weight[valid_rho],
        )),
        "median_contig_rho": float(np.median(rho[valid_rho])),
        "min_contig_rho": float(np.min(rho[valid_rho])),
        "max_contig_rho": float(np.max(rho[valid_rho])),
        "window_weighted_contig_auc": float(np.average(
            auc[valid_auc], weights=truth_weight[valid_auc],
        )),
        "median_contig_auc": float(np.median(auc[valid_auc])),
        "min_contig_auc": float(np.min(auc[valid_auc])),
        "max_contig_auc": float(np.max(auc[valid_auc])),
    }
    with (output / "summary.tsv").open("w") as handle:
        for key, value in summary.items():
            handle.write(f"{key}\t{value}\n")
    for key, value in summary.items():
        print(f"{key}\t{value}")


if __name__ == "__main__":
    main()
