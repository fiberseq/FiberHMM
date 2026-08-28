#!/usr/bin/env python3
"""Empirically calibrate bidirectional DAF SNP thresholds by fiber thinning.

The production caller is first run with permissive discovery settings to
obtain per-site evidence.  Each direction-specific fiber pool is then thinned
with independent Bernoulli sampling.  This is the exact marginal distribution
of uniform molecule downsampling at a site, while avoiding hundreds of large
temporary BAMs.  It preserves the two direction classes and samples mismatch
and reference-supporting fibers separately.

This script deliberately uses precise validation labels:

* ``truth`` is a prespecified, unambiguous full-depth evidence class, not an
  external genotype truth set.
* ``background`` is a prespecified high-depth/low-mismatch evidence class.
* ``discordant`` means called after thinning but absent from the full-depth
  truth class.  It is not called a false positive without orthogonal genotype
  data.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import shlex
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np

from fiberhmm.daf.snps import call_opposite_conversion_snps


DEFAULT_FRACTIONS = (1.0, 0.75, 0.50, 0.35, 0.25, 0.18, 0.125, 0.09, 0.0625, 0.04, 0.025)
DEFAULT_POLICIES = (
    ("three_fiber", 3, 3, 0.20),
    ("five_fiber", 5, 5, 0.20),
    ("depth8_five_fiber", 8, 5, 0.20),
    ("depth10_five_fiber", 10, 5, 0.20),
    ("depth15_five_fiber", 15, 5, 0.20),
    ("depth20_five_fiber", 20, 5, 0.20),
)


@dataclass(frozen=True)
class Dataset:
    label: str
    cohort: str
    path: Path


@dataclass(frozen=True)
class Policy:
    label: str
    min_depth: int
    min_alt: int
    min_fraction: float


def _sha256(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_bytes)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _parse_dataset(value: str) -> Dataset:
    fields = value.split("=", 2)
    if len(fields) != 3 or not all(fields):
        raise argparse.ArgumentTypeError(
            "dataset must be LABEL=COHORT=/path/to/input.bam"
        )
    label, cohort, raw_path = fields
    path = Path(raw_path).expanduser().resolve()
    if not path.exists():
        raise argparse.ArgumentTypeError(f"dataset does not exist: {path}")
    return Dataset(label=label, cohort=cohort, path=path)


def _parse_policy(value: str) -> Policy:
    fields = value.split(":")
    if len(fields) != 4:
        raise argparse.ArgumentTypeError(
            "policy must be LABEL:MIN_DEPTH:MIN_ALT:MIN_FRACTION"
        )
    label, depth, alternate, fraction = fields
    try:
        policy = Policy(label, int(depth), int(alternate), float(fraction))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    if policy.min_depth < 1 or policy.min_alt < 1:
        raise argparse.ArgumentTypeError("depth and alternate count must be positive")
    if not 0 < policy.min_fraction <= 1:
        raise argparse.ArgumentTypeError("fraction must be in (0, 1]")
    return policy


def _write_tsv(path: Path, rows: Iterable[dict], fields: list[str]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _load_evidence(dataset: Dataset, cache_dir: Path, profile_sites: int) -> dict:
    cache_path = cache_dir / f"{dataset.label}.full_evidence.json"
    expected_parameters = {
        "min_fraction": 1e-9,
        "min_depth": 1,
        "min_alt_fibers": 3,
        "min_dominant_events": 5,
        "min_dominant_purity": 0.80,
        "min_mapq": 20,
        "site_profile_max_sites": profile_sites,
        "site_profile_min_depth_per_direction": 1,
        "min_amplicon_reads": 20,
    }
    if cache_path.exists():
        cached = json.loads(cache_path.read_text())
        cached_parameters = cached.get("parameters", {})
        if (
            cached.get("input") == str(dataset.path)
            and all(cached_parameters.get(key) == value for key, value in expected_parameters.items())
        ):
            print(f"[{dataset.label}] using cached evidence: {cache_path}", file=sys.stderr)
            return cached
        print(
            f"[{dataset.label}] ignoring stale/incompatible cache: {cache_path}",
            file=sys.stderr,
        )
    print(f"[{dataset.label}] profiling full-depth evidence", file=sys.stderr)
    payload = call_opposite_conversion_snps(
        str(dataset.path),
        min_fraction=1e-9,
        min_depth=1,
        min_alt_fibers=3,
        min_dominant_events=5,
        min_dominant_purity=0.80,
        min_mapq=20,
        max_profile_sites=profile_sites,
        profile_min_depth=1,
        min_amplicon_reads=20,
    )
    _atomic_json(cache_path, payload)
    return payload


def _site_rows(
    dataset: Dataset,
    payload: dict,
    truth_min_depth: int,
    truth_min_alt: int,
    truth_min_fraction: float,
    background_max_fraction: float,
) -> list[dict]:
    rows = []
    for site in payload.get("site_distribution", []):
        expected_depth = int(site.get("expected_direction_depth") or 0)
        opposite_depth = int(site.get("opposite_direction_depth") or 0)
        expected_alt = int(site.get("expected_direction_mismatch_fibers") or 0)
        opposite_alt = int(site.get("opposite_direction_mismatch_fibers") or 0)
        expected_fraction = expected_alt / expected_depth if expected_depth else 0.0
        opposite_fraction = opposite_alt / opposite_depth if opposite_depth else 0.0
        adequate_depth = min(expected_depth, opposite_depth) >= truth_min_depth
        truth = (
            adequate_depth
            and min(expected_alt, opposite_alt) >= truth_min_alt
            and min(expected_fraction, opposite_fraction) >= truth_min_fraction
        )
        background = (
            adequate_depth
            and max(expected_fraction, opposite_fraction) <= background_max_fraction
        )
        evidence_class = "truth" if truth else "background" if background else "intermediate"
        rows.append(
            {
                "dataset": dataset.label,
                "cohort": dataset.cohort,
                "chrom": site["chrom"],
                "position_0based": int(site["position_0based"]),
                "reference": site["reference"],
                "alternate": site["alternate"],
                "expected_depth": expected_depth,
                "expected_alt": expected_alt,
                "expected_fraction": expected_fraction,
                "opposite_depth": opposite_depth,
                "opposite_alt": opposite_alt,
                "opposite_fraction": opposite_fraction,
                "min_bidirectional_depth": min(expected_depth, opposite_depth),
                "evidence_class": evidence_class,
            }
        )
    return rows


def _called(
    expected_depth: np.ndarray,
    expected_alt: np.ndarray,
    opposite_depth: np.ndarray,
    opposite_alt: np.ndarray,
    policy: Policy,
) -> np.ndarray:
    expected_fraction = np.divide(
        expected_alt,
        expected_depth,
        out=np.zeros_like(expected_alt, dtype=float),
        where=expected_depth > 0,
    )
    opposite_fraction = np.divide(
        opposite_alt,
        opposite_depth,
        out=np.zeros_like(opposite_alt, dtype=float),
        where=opposite_depth > 0,
    )
    return (
        (expected_depth >= policy.min_depth)
        & (opposite_depth >= policy.min_depth)
        & (expected_alt >= policy.min_alt)
        & (opposite_alt >= policy.min_alt)
        & (expected_fraction >= policy.min_fraction)
        & (opposite_fraction >= policy.min_fraction)
    )


def _simulate_dataset(
    dataset: Dataset,
    rows: list[dict],
    policies: list[Policy],
    fractions: list[float],
    replicates: int,
    seed: int,
) -> tuple[list[dict], list[dict]]:
    expected_depth = np.asarray([row["expected_depth"] for row in rows], dtype=int)
    expected_alt = np.asarray([row["expected_alt"] for row in rows], dtype=int)
    opposite_depth = np.asarray([row["opposite_depth"] for row in rows], dtype=int)
    opposite_alt = np.asarray([row["opposite_alt"] for row in rows], dtype=int)
    truth = np.asarray([row["evidence_class"] == "truth" for row in rows], dtype=bool)
    background = np.asarray([row["evidence_class"] == "background" for row in rows], dtype=bool)
    intermediate = ~(truth | background)
    rng = np.random.default_rng(seed)
    metrics = []
    retention = defaultdict(lambda: [0, 0])
    truth_indices = np.flatnonzero(truth)
    for fraction in fractions:
        for replicate in range(replicates):
            # Independent thinning of mismatch and reference-supporting fibers
            # is distributionally identical to uniform Bernoulli read thinning
            # for the marginal evidence at each site.
            e_alt = rng.binomial(expected_alt, fraction)
            e_ref = rng.binomial(expected_depth - expected_alt, fraction)
            o_alt = rng.binomial(opposite_alt, fraction)
            o_ref = rng.binomial(opposite_depth - opposite_alt, fraction)
            e_depth = e_alt + e_ref
            o_depth = o_alt + o_ref
            truth_min_depths = np.minimum(e_depth[truth], o_depth[truth])
            median_truth_depth = (
                float(np.median(truth_min_depths)) if len(truth_min_depths) else None
            )
            for policy in policies:
                calls = _called(e_depth, e_alt, o_depth, o_alt, policy)
                truth_calls = int(np.count_nonzero(calls & truth))
                background_calls = int(np.count_nonzero(calls & background))
                intermediate_calls = int(np.count_nonzero(calls & intermediate))
                n_truth = int(np.count_nonzero(truth))
                n_background = int(np.count_nonzero(background))
                metrics.append(
                    {
                        "dataset": dataset.label,
                        "cohort": dataset.cohort,
                        "fraction": fraction,
                        "replicate": replicate,
                        "policy": policy.label,
                        "min_depth": policy.min_depth,
                        "min_alt": policy.min_alt,
                        "min_fraction": policy.min_fraction,
                        "median_truth_min_bidirectional_depth": median_truth_depth,
                        "n_truth_sites": n_truth,
                        "truth_calls": truth_calls,
                        "truth_recall": truth_calls / n_truth if n_truth else None,
                        "n_background_sites": n_background,
                        "background_calls": background_calls,
                        "background_call_rate": (
                            background_calls / n_background if n_background else None
                        ),
                        "intermediate_calls": intermediate_calls,
                        "discordant_calls": background_calls + intermediate_calls,
                        "total_calls": int(np.count_nonzero(calls)),
                    }
                )
                for site_index in truth_indices:
                    key = (policy.label, fraction, int(site_index))
                    retention[key][0] += int(calls[site_index])
                    retention[key][1] += 1
    retention_rows = []
    for (policy_label, fraction, site_index), (retained, total) in retention.items():
        site = rows[site_index]
        retention_rows.append(
            {
                "dataset": dataset.label,
                "cohort": dataset.cohort,
                "chrom": site["chrom"],
                "position_0based": site["position_0based"],
                "reference": site["reference"],
                "alternate": site["alternate"],
                "full_min_bidirectional_depth": site["min_bidirectional_depth"],
                "fraction": fraction,
                "policy": policy_label,
                "retained_replicates": retained,
                "replicates": total,
                "retention_probability": retained / total,
            }
        )
    return metrics, retention_rows


def _quantile(values: list[float], q: float) -> float | None:
    clean = np.asarray([value for value in values if value is not None], dtype=float)
    return float(np.quantile(clean, q)) if len(clean) else None


def _summarize(metrics: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in metrics:
        groups[(row["policy"], row["fraction"])].append(row)
    output = []
    for (policy, fraction), rows in sorted(groups.items()):
        recalls = [row["truth_recall"] for row in rows if row["truth_recall"] is not None]
        background_calls = [row["background_calls"] for row in rows]
        discordant_calls = [row["discordant_calls"] for row in rows]
        output.append(
            {
                "policy": policy,
                "fraction": fraction,
                "n_dataset_replicates": len(rows),
                "mean_median_truth_min_bidirectional_depth": np.mean(
                    [row["median_truth_min_bidirectional_depth"] for row in rows]
                ),
                "mean_truth_recall": np.mean(recalls) if recalls else None,
                "truth_recall_q10": _quantile(recalls, 0.10),
                "truth_recall_q90": _quantile(recalls, 0.90),
                "mean_background_calls": np.mean(background_calls),
                "background_calls_q90": _quantile(background_calls, 0.90),
                "mean_discordant_calls": np.mean(discordant_calls),
                "discordant_calls_q90": _quantile(discordant_calls, 0.90),
            }
        )
    return output


def _summarize_overall(metrics: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in metrics:
        groups[row["policy"]].append(row)
    output = []
    for policy, rows in sorted(groups.items()):
        background_calls = sum(row["background_calls"] for row in rows)
        background_trials = sum(row["n_background_sites"] for row in rows)
        truth_calls = sum(row["truth_calls"] for row in rows)
        truth_trials = sum(row["n_truth_sites"] for row in rows)
        full_depth = [row for row in rows if row["fraction"] == 1.0]
        output.append(
            {
                "policy": policy,
                "truth_calls_all_downsampling": truth_calls,
                "truth_site_trials_all_downsampling": truth_trials,
                "truth_recall_all_downsampling": truth_calls / truth_trials,
                "full_depth_truth_recall": float(
                    np.mean([row["truth_recall"] for row in full_depth])
                ),
                "background_calls_all_downsampling": background_calls,
                "background_site_trials_all_downsampling": background_trials,
                "background_calls_per_million_site_trials": (
                    1e6 * background_calls / background_trials
                ),
            }
        )
    return output


def _binned_retention(
    rows: list[dict],
    policy: str,
    cohort: str | None = None,
) -> list[tuple[float, float, float, float, int]]:
    edges = np.asarray((0, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 30, 50, 100, 200, np.inf))
    selected = [
        row for row in rows
        if row["policy"] == policy and (cohort is None or row["cohort"] == cohort)
    ]
    bins = defaultdict(list)
    for row in selected:
        expected_depth = row["fraction"] * row["full_min_bidirectional_depth"]
        index = int(np.searchsorted(edges, expected_depth, side="right") - 1)
        index = min(max(index, 0), len(edges) - 2)
        bins[index].append((expected_depth, row["retention_probability"]))
    output = []
    for index in sorted(bins):
        group = bins[index]
        x = float(np.median([item[0] for item in group]))
        retention = [item[1] for item in group]
        output.append(
            (
                x,
                float(np.mean(retention)),
                _quantile(retention, 0.10),
                _quantile(retention, 0.90),
                len(group),
            )
        )
    return output


def _retention_summary_rows(
    retention: list[dict],
    policies: list[Policy],
) -> list[dict]:
    cohorts: list[str | None] = [None, *sorted({row["cohort"] for row in retention})]
    output = []
    for policy in policies:
        for cohort in cohorts:
            for x, mean, q10, q90, count in _binned_retention(
                retention, policy.label, cohort=cohort
            ):
                output.append(
                    {
                        "policy": policy.label,
                        "cohort": cohort or "all cohorts",
                        "median_expected_min_bidirectional_depth": x,
                        "mean_truth_site_retention": mean,
                        "truth_site_retention_q10": q10,
                        "truth_site_retention_q90": q90,
                        "n_site_fraction_observations": count,
                    }
                )
    return output


def _make_figure(
    sites: list[dict],
    metrics: list[dict],
    summary: list[dict],
    overall_summary: list[dict],
    retention: list[dict],
    policies: list[Policy],
    output_prefix: Path,
    highlight: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure, axes = plt.subplots(2, 2, figsize=(7.2, 6.6), constrained_layout=True)
    colors = {
        policy.label: plt.cm.viridis(index / max(1, len(policies) - 1))
        for index, policy in enumerate(policies)
    }

    axis = axes[0, 0]
    rng = np.random.default_rng(20260824)
    background = [row for row in sites if row["evidence_class"] == "background"]
    if len(background) > 12000:
        background = [background[index] for index in rng.choice(len(background), 12000, replace=False)]
    intermediate = [row for row in sites if row["evidence_class"] == "intermediate"]
    truth = [row for row in sites if row["evidence_class"] == "truth"]
    for group, color, size, alpha, label in (
        (background, "#bdbdbd", 5, 0.25, "high-depth background"),
        (intermediate, "#fdae61", 8, 0.28, "intermediate evidence"),
        (truth, "#c51b7d", 25, 0.90, "full-depth truth class"),
    ):
        axis.scatter(
            [100 * row["expected_fraction"] for row in group],
            [100 * row["opposite_fraction"] for row in group],
            s=size,
            alpha=alpha,
            color=color,
            linewidths=0,
            label=label,
        )
    axis.plot([0, 100], [0, 100], ":", color="#777777", linewidth=0.8)
    axis.set(
        xlim=(-2, 102),
        ylim=(-2, 102),
        xlabel="expected-direction mismatch (%)",
        ylabel="opposite-direction mismatch (%)",
        title="A  Prespecified full-depth evidence classes",
    )
    axis.legend(frameon=False, loc="upper left")

    axis = axes[0, 1]
    for policy in policies:
        rows = _binned_retention(retention, policy.label)
        x = np.asarray([row[0] for row in rows])
        y = np.asarray([row[1] for row in rows])
        low = np.asarray([row[2] for row in rows])
        high = np.asarray([row[3] for row in rows])
        linewidth = 2.1 if policy.label == highlight else 1.0
        alpha = 1.0 if policy.label == highlight else 0.72
        axis.plot(x, y, marker="o", markersize=2.8, linewidth=linewidth, alpha=alpha,
                  color=colors[policy.label], label=policy.label.replace("_", " "))
        if policy.label == highlight:
            axis.fill_between(x, low, high, color=colors[policy.label], alpha=0.14, linewidth=0)
    axis.axhline(0.95, linestyle=":", color="#777777", linewidth=0.8)
    axis.set(
        xscale="log",
        xlim=(2, None),
        ylim=(-0.03, 1.03),
        xlabel="expected minimum bidirectional depth",
        ylabel="full-depth truth-class recall",
        title="B  Call stability during fiber downsampling",
    )
    axis.legend(frameon=False, ncol=2, loc="lower right")

    axis = axes[1, 0]
    overall_lookup = {row["policy"]: row for row in overall_summary}
    x_positions = np.arange(len(policies))
    heights = [
        overall_lookup[policy.label]["background_calls_per_million_site_trials"]
        for policy in policies
    ]
    bars = axis.bar(
        x_positions,
        heights,
        color=[colors[policy.label] for policy in policies],
        width=0.72,
    )
    for bar, policy in zip(bars, policies):
        row = overall_lookup[policy.label]
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            max(bar.get_height(), 0) + max(0.015, max(heights) * 0.04),
            f"{row['background_calls_all_downsampling']:,} / "
            f"{row['background_site_trials_all_downsampling']:,}",
            ha="center",
            va="bottom",
            fontsize=6.5,
            rotation=90,
        )
    axis.set(
        ylim=(0, max(0.12, max(heights) * 1.35)),
        ylabel="calls per million background site trials",
        title="C  Calls in the prespecified background class",
    )
    axis.set_xticks(
        x_positions,
        [policy.label.replace("_", "\n") for policy in policies],
        rotation=0,
        fontsize=6.5,
    )

    axis = axes[1, 1]
    cohorts = sorted({row["cohort"] for row in metrics})
    cohort_colors = {cohort: plt.cm.Set2(index) for index, cohort in enumerate(cohorts)}
    highlight_rows = [row for row in metrics if row["policy"] == highlight]
    for cohort in cohorts:
        points = _binned_retention(retention, highlight, cohort=cohort)
        if points:
            x_values = np.asarray([point[0] for point in points])
            y_values = np.asarray([point[1] for point in points])
            axis.plot(x_values, y_values, marker="o", linewidth=1.5,
                      color=cohort_colors[cohort], label=cohort)
            axis.fill_between(
                x_values,
                [point[2] for point in points],
                [point[3] for point in points],
                color=cohort_colors[cohort],
                alpha=0.12,
                linewidth=0,
            )
    axis.axhline(0.95, linestyle=":", color="#777777", linewidth=0.8)
    axis.set(
        xscale="log",
        xlim=(2, None),
        ylim=(-0.03, 1.03),
        xlabel="expected minimum bidirectional depth",
        ylabel=f"truth-class recall ({highlight.replace('_', ' ')})",
        title="D  Stability across independent cohorts",
    )
    axis.legend(frameon=False)

    figure.suptitle(
        "Bidirectional DAF SNP threshold calibration by marginal fiber downsampling",
        fontsize=11,
    )
    for suffix, dpi in ((".pdf", 300), (".svg", 300), (".png", 300)):
        figure.savefig(str(output_prefix) + suffix, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def _arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        action="append",
        type=_parse_dataset,
        required=True,
        help="Repeatable LABEL=COHORT=/path/to/input.bam specification.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--profile-sites", type=int, default=10000)
    parser.add_argument("--fraction", action="append", type=float)
    parser.add_argument("--policy", action="append", type=_parse_policy)
    parser.add_argument("--highlight-policy", default="five_fiber")
    parser.add_argument("--truth-min-depth", type=int, default=15)
    parser.add_argument("--truth-min-alt", type=int, default=15)
    parser.add_argument("--truth-min-fraction", type=float, default=0.80)
    parser.add_argument("--background-max-fraction", type=float, default=0.05)
    parser.add_argument("--release-id", default="provisional-working-tree-2026-08-24")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = _arguments(argv)
    if args.replicates < 1:
        raise SystemExit("--replicates must be positive")
    fractions = sorted(set(args.fraction or DEFAULT_FRACTIONS), reverse=True)
    if any(not 0 < fraction <= 1 for fraction in fractions):
        raise SystemExit("all --fraction values must be in (0, 1]")
    policies = args.policy or [Policy(*values) for values in DEFAULT_POLICIES]
    if args.highlight_policy not in {policy.label for policy in policies}:
        raise SystemExit("--highlight-policy must name one of the configured policies")
    labels = [dataset.label for dataset in args.dataset]
    if len(labels) != len(set(labels)):
        raise SystemExit("dataset labels must be unique")

    output_dir = args.output_dir.resolve()
    cache_dir = args.cache_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    all_sites = []
    all_metrics = []
    all_retention = []
    inputs = []
    caller_source = Path(call_opposite_conversion_snps.__code__.co_filename).resolve()
    for dataset_index, dataset in enumerate(args.dataset):
        payload = _load_evidence(dataset, cache_dir, args.profile_sites)
        rows = _site_rows(
            dataset,
            payload,
            args.truth_min_depth,
            args.truth_min_alt,
            args.truth_min_fraction,
            args.background_max_fraction,
        )
        class_counts = defaultdict(int)
        for row in rows:
            class_counts[row["evidence_class"]] += 1
        print(
            f"[{dataset.label}] {class_counts['truth']} truth, "
            f"{class_counts['background']} background, "
            f"{class_counts['intermediate']} intermediate sites",
            file=sys.stderr,
        )
        metrics, retention = _simulate_dataset(
            dataset,
            rows,
            policies,
            fractions,
            args.replicates,
            args.seed + 1000003 * dataset_index,
        )
        all_sites.extend(rows)
        all_metrics.extend(metrics)
        all_retention.extend(retention)
        inputs.append(
            {
                "label": dataset.label,
                "cohort": dataset.cohort,
                "path_at_run": str(dataset.path),
                "size_bytes": dataset.path.stat().st_size,
                "sha256": _sha256(dataset.path),
                "profile_accounting": payload.get("accounting", {}),
                "evidence_class_counts": dict(class_counts),
            }
        )

    summary = _summarize(all_metrics)
    overall_summary = _summarize_overall(all_metrics)
    retention_summary = _retention_summary_rows(all_retention, policies)
    site_fields = list(all_sites[0]) if all_sites else []
    metric_fields = list(all_metrics[0]) if all_metrics else []
    retention_fields = list(all_retention[0]) if all_retention else []
    summary_fields = list(summary[0]) if summary else []
    overall_summary_fields = list(overall_summary[0]) if overall_summary else []
    retention_summary_fields = list(retention_summary[0]) if retention_summary else []
    _write_tsv(output_dir / "full_depth_site_evidence.tsv", all_sites, site_fields)
    _write_tsv(output_dir / "downsampling_replicates.tsv", all_metrics, metric_fields)
    _write_tsv(output_dir / "truth_site_retention.tsv", all_retention, retention_fields)
    _write_tsv(output_dir / "policy_summary.tsv", summary, summary_fields)
    _write_tsv(
        output_dir / "policy_overall.tsv",
        overall_summary,
        overall_summary_fields,
    )
    _write_tsv(
        output_dir / "retention_by_effective_depth.tsv",
        retention_summary,
        retention_summary_fields,
    )
    figure_prefix = output_dir / "daf_snp_downsampling_validation"
    _make_figure(
        all_sites,
        all_metrics,
        summary,
        overall_summary,
        all_retention,
        policies,
        figure_prefix,
        args.highlight_policy,
    )

    manifest = {
        "schema_version": 1,
        "status": "provisional",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "release_id": args.release_id,
        "command": shlex.join(sys.argv),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "caller_source": {
            "path_at_run": str(caller_source),
            "sha256": _sha256(caller_source),
        },
        "inputs": inputs,
        "resampling": {
            "unit": "direction-classified fiber observation at each site",
            "method": "independent Bernoulli thinning of mismatch and reference-supporting fibers",
            "interpretation": "exact marginal distribution of uniform fiber downsampling per site",
            "seed": args.seed,
            "replicates": args.replicates,
            "fractions": fractions,
        },
        "evidence_classes": {
            "truth": {
                "minimum_depth_in_each_direction": args.truth_min_depth,
                "minimum_mismatch_fibers_in_each_direction": args.truth_min_alt,
                "minimum_mismatch_fraction_in_each_direction": args.truth_min_fraction,
                "scope": "prespecified unambiguous full-depth evidence class; not orthogonal genotype truth",
            },
            "background": {
                "minimum_depth_in_each_direction": args.truth_min_depth,
                "maximum_mismatch_fraction_in_either_direction": args.background_max_fraction,
                "scope": "prespecified high-depth low-mismatch class; not asserted reference-genotype truth",
            },
            "discordant_call": (
                "called after thinning but absent from the full-depth truth class; "
                "not labeled a false positive without orthogonal genotyping"
            ),
        },
        "policies": [policy.__dict__ for policy in policies],
        "highlight_policy": args.highlight_policy,
        "outputs": {
            "full_depth_site_evidence": "full_depth_site_evidence.tsv",
            "replicate_metrics": "downsampling_replicates.tsv",
            "truth_site_retention": "truth_site_retention.tsv",
            "policy_summary": "policy_summary.tsv",
            "policy_overall": "policy_overall.tsv",
            "retention_by_effective_depth": "retention_by_effective_depth.tsv",
            "figure_pdf": "daf_snp_downsampling_validation.pdf",
            "figure_svg": "daf_snp_downsampling_validation.svg",
            "figure_png": "daf_snp_downsampling_validation.png",
        },
    }
    _atomic_json(output_dir / "run_manifest.json", manifest)
    print(f"Wrote validation package to {output_dir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
