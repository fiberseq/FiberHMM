#!/usr/bin/env python3
"""Summarize four-state DddA mCG evidence on cross-strand consensus reads."""
from __future__ import annotations

import argparse
import json
from collections import Counter

import numpy as np
import pysam
from scipy.stats import spearmanr

from fiberhmm.core.bam_reader import cigar_to_query_ref
from fiberhmm.daf.m5c import (
    DDDA_FIVE_PRIME_FACTORS,
    PAIRED_STATE_NAMES,
    call_paired_read_m5c,
    collect_paired_read_observations,
    deamination_probability,
    distance_forward_backward,
    distance_forward_backward_states,
)


def _region(text: str, bam) -> tuple[str, int, int]:
    if ":" not in text:
        return text, 0, bam.get_reference_length(text)
    chrom, coordinates = text.rsplit(":", 1)
    left, right = coordinates.replace(",", "").split("-", 1)
    start = max(0, int(left) - 1)
    end = min(int(right), bam.get_reference_length(chrom))
    if end <= start:
        raise ValueError("region end must exceed start")
    return chrom, start, end


def _runs(mask: np.ndarray, positions: np.ndarray, max_gap: float,
          min_cpg: int) -> list[tuple[int, int]]:
    runs = []
    i = 0
    while i < len(mask):
        if not mask[i]:
            i += 1
            continue
        j = i
        while (j + 1 < len(mask) and mask[j + 1] and
               positions[j + 1] - positions[j] <= max_gap):
            j += 1
        if j - i + 1 >= min_cpg:
            runs.append((i, j))
        i = j + 1
    return runs


def _simulate_symmetric_result(result, symmetric_methylated, rng,
                               expected_run_bp):
    """Simulate both channels under a shared UU/MM state at every CpG."""
    n = len(result.reference_pos)
    c_u = np.zeros(n, dtype=float)
    c_m = np.zeros(n, dtype=float)
    g_u = np.zeros(n, dtype=float)
    g_m = np.zeros(n, dtype=float)
    for observed, baseline, target_u, target_m in (
        (result.c_observed, result.c_baseline, c_u, c_m),
        (result.g_observed, result.g_baseline, g_u, g_m),
    ):
        if not np.any(observed):
            continue
        beta = symmetric_methylated[observed].astype(float)
        p_simulated = deamination_probability(beta, baseline[observed])
        deaminated = rng.random(len(p_simulated)) < p_simulated
        p_u = np.clip(
            deamination_probability(0.0, baseline[observed]), 1e-12, 1 - 1e-12,
        )
        p_m = np.clip(
            deamination_probability(1.0, baseline[observed]), 1e-12, 1 - 1e-12,
        )
        target_u[observed] = np.where(
            deaminated, np.log(p_u), np.log1p(-p_u),
        )
        target_m[observed] = np.where(
            deaminated, np.log(p_m), np.log1p(-p_m),
        )
    emission = np.column_stack([
        c_u + g_u, c_u + g_m, c_m + g_u, c_m + g_m,
    ])
    return distance_forward_backward_states(
        emission, result.reference_pos, expected_run_bp,
    )


def _canonical_base_map(read) -> dict[int, str]:
    """Reference-position bases with DAF Y/R restored to canonical C/G."""
    sequence = (read.query_sequence or "").upper()
    q_to_r = cigar_to_query_ref(read)
    restore = {"Y": "C", "R": "G"}
    result = {}
    for query_pos in range(min(len(sequence), len(q_to_r))):
        reference_pos = int(q_to_r[query_pos])
        if reference_pos < 0:
            continue
        base = restore.get(sequence[query_pos], sequence[query_pos])
        if base in "ACGT":
            result[reference_pos] = base
    return result


def _source_pair_qc(source_bam, chrom, start, end, pair_records, thresholds):
    if not source_bam or not pair_records:
        return {}
    wanted = {name for record in pair_records for name in record[1]}
    reads = {}
    with pysam.AlignmentFile(source_bam, "rb") as source:
        for read in source.fetch(chrom, start, end):
            if (read.query_name not in wanted or read.is_unmapped or
                    read.is_secondary or read.is_supplementary):
                continue
            previous = reads.get(read.query_name)
            if previous is None or read.mapping_quality > previous.mapping_quality:
                reads[read.query_name] = read
    groups = {"all": []}
    groups.update({str(threshold): [] for threshold in thresholds})
    strict_candidates = []
    for merged_name, (left_name, right_name), candidate_thresholds, pair_score in pair_records:
        if left_name not in reads or right_name not in reads:
            continue
        left = _canonical_base_map(reads[left_name])
        right = _canonical_base_map(reads[right_name])
        common = left.keys() & right.keys()
        if not common:
            continue
        mismatches = sum(left[pos] != right[pos] for pos in common)
        row = (len(common), mismatches)
        groups["all"].append(row)
        for threshold in candidate_thresholds:
            groups[str(threshold)].append(row)
        if "0.99" in candidate_thresholds:
            strict_candidates.append({
                "read": merged_name,
                "sources": [left_name, right_name],
                "pair_score": pair_score,
                "comparable_bases": len(common),
                "mismatches": mismatches,
                "mismatch_rate": mismatches / len(common),
            })

    summary = {}
    for group, rows in groups.items():
        comparable = np.asarray([row[0] for row in rows], dtype=float)
        mismatches = np.asarray([row[1] for row in rows], dtype=float)
        rates = mismatches / np.maximum(comparable, 1)
        summary[group] = {
            "pairs": len(rows),
            "comparable_bases": int(comparable.sum()) if len(rows) else 0,
            "mismatches": int(mismatches.sum()) if len(rows) else 0,
            "weighted_mismatch_rate": (
                float(mismatches.sum() / comparable.sum())
                if comparable.sum() else None
            ),
            "pair_mismatch_rate_quantiles": (
                {str(q): float(np.quantile(rates, q))
                 for q in (0.5, 0.9, 0.95, 1.0)}
                if len(rates) else {}
            ),
        }
    summary["source_reads_found"] = len(reads)
    summary["source_reads_requested"] = len(wanted)
    summary["strict_candidates"] = sorted(
        strict_candidates, key=lambda row: row["mismatch_rate"], reverse=True,
    )
    return summary


def summarize(args) -> dict:
    factors = DDDA_FIVE_PRIME_FACTORS / DDDA_FIVE_PRIME_FACTORS.mean()
    thresholds = sorted(set(float(value) for value in args.thresholds.split(",")))
    if any(not 0.5 < value < 1.0 for value in thresholds):
        raise ValueError("every threshold must be between 0.5 and 1")
    counters = Counter()
    state_argmax = Counter()
    state_probability = []
    c_methylated = []
    g_methylated = []
    max_hemi_per_read = []
    pair_scores = []
    paired_results = []
    pair_records = []
    candidate_pair_scores = {str(value): [] for value in thresholds}
    pair_score_cutoffs = (0.3, 0.4, 0.5, 0.6, 0.7)
    score_strata = {
        str(value): {str(cutoff): Counter() for cutoff in pair_score_cutoffs}
        for value in thresholds
    }
    strict_candidate_intervals = []
    strict_threshold = max(thresholds)
    threshold_stats = {
        str(value): Counter() for value in thresholds
    }

    with pysam.FastaFile(args.reference) as fasta:
        with pysam.AlignmentFile(args.input, "rb", threads=args.io_threads) as bam:
            chrom, start, end = _region(args.region, bam)
            for read in bam.fetch(chrom, start, end):
                counters["reads"] += 1
                if (read.is_unmapped or read.is_secondary or
                        read.is_supplementary or not read.has_tag("MA")):
                    continue
                ma = str(read.get_tag("MA"))
                if ";deam+:" not in ma or ";deam-:" not in ma:
                    continue
                counters["cross_strand_reads"] += 1
                c_observations, g_observations = collect_paired_read_observations(
                    read, fasta,
                )
                if not c_observations or not g_observations:
                    continue
                counters["eligible_cross_strand_reads"] += 1
                result = call_paired_read_m5c(
                    c_observations, g_observations, factors,
                    expected_run_bp=args.run_bp,
                    posterior_threshold=min(thresholds),
                    baseline_radius=args.baseline_radius,
                    min_other=args.min_other,
                    min_call_cpg=args.min_run_cpg,
                    max_call_gap_bp=args.max_cpg_gap,
                )
                both = result.c_observed & result.g_observed
                if not np.any(both):
                    continue
                counters["scored_cross_strand_reads"] += 1
                counters["both_cpgs"] += int(np.count_nonzero(both))
                post = result.state_posterior[both]
                positions = result.reference_pos[both]
                state_probability.append(post)
                c_methylated.append(result.c_methylated_posterior[both])
                g_methylated.append(result.g_methylated_posterior[both])
                for index in np.argmax(post, axis=1):
                    state_argmax[PAIRED_STATE_NAMES[int(index)]] += 1
                hemi_max = np.max(post[:, [1, 2]], axis=1)
                max_hemi_per_read.append(float(hemi_max.max()))
                pair_score = (float(read.get_tag("mc")) / 1000.0
                              if read.has_tag("mc") else np.nan)
                pair_scores.append(pair_score)
                symmetric_posterior = distance_forward_backward(
                    result.log_emission[:, [0, 3]], result.reference_pos,
                    args.run_bp,
                )[:, 1]
                paired_results.append((
                    result, symmetric_posterior >= 0.5,
                ))
                for threshold in thresholds:
                    counts = threshold_stats[str(threshold)]
                    read_run_cpgs = 0
                    read_runs = 0
                    for state_index, state in ((1, "UM"), (2, "MU")):
                        selected = post[:, state_index] >= threshold
                        runs = _runs(
                            selected, positions, args.max_cpg_gap,
                            args.min_run_cpg,
                        )
                        counts[f"{state}_cpgs"] += int(np.count_nonzero(selected))
                        counts[f"{state}_runs"] += len(runs)
                        counts[f"{state}_run_cpgs"] += sum(
                            right - left + 1 for left, right in runs
                        )
                        counts[f"{state}_reads"] += int(bool(runs))
                        if threshold == strict_threshold:
                            for left, right in runs:
                                strict_candidate_intervals.append({
                                    "read": read.query_name,
                                    "state": state,
                                    "chrom": chrom,
                                    "start": int(positions[left]),
                                    "end": int(positions[right]) + 2,
                                    "n_cpg": right - left + 1,
                                    "mean_posterior": float(
                                        post[left:right + 1, state_index].mean()
                                    ),
                                    "min_posterior": float(
                                        post[left:right + 1, state_index].min()
                                    ),
                                    "pair_score": pair_score,
                                })
                        read_runs += len(runs)
                        read_run_cpgs += sum(
                            right - left + 1 for left, right in runs
                        )
                    if read_runs and np.isfinite(pair_score):
                        candidate_pair_scores[str(threshold)].append(pair_score)
                    if np.isfinite(pair_score):
                        for cutoff in pair_score_cutoffs:
                            if pair_score >= cutoff:
                                stratum = score_strata[str(threshold)][str(cutoff)]
                                stratum["reads"] += 1
                                stratum["both_cpgs"] += int(np.count_nonzero(both))
                                stratum["hemi_runs"] += read_runs
                                stratum["hemi_run_cpgs"] += read_run_cpgs
                if read.has_tag("cs"):
                    sources = str(read.get_tag("cs")).split(";", 1)
                    if len(sources) == 2:
                        candidate_thresholds = {
                            threshold
                            for threshold, values in threshold_stats.items()
                            # This cannot use cumulative counters; recompute the
                            # current read's state-run presence directly.
                            if any(
                                _runs(
                                    post[:, state_index] >= float(threshold),
                                    positions, args.max_cpg_gap,
                                    args.min_run_cpg,
                                )
                                for state_index in (1, 2)
                            )
                        }
                        pair_records.append((
                            read.query_name, tuple(sources),
                            candidate_thresholds, pair_score,
                        ))

    if state_probability:
        state_probability_array = np.concatenate(state_probability)
        c_methylated_array = np.concatenate(c_methylated)
        g_methylated_array = np.concatenate(g_methylated)
    else:
        state_probability_array = np.empty((0, 4))
        c_methylated_array = np.array([], dtype=float)
        g_methylated_array = np.array([], dtype=float)
    max_hemi_array = np.asarray(max_hemi_per_read, dtype=float)
    pair_score_array = np.asarray(pair_scores, dtype=float)
    finite_pair = np.isfinite(pair_score_array)
    if len(c_methylated_array) > 1:
        strand_rho = float(spearmanr(c_methylated_array, g_methylated_array).statistic)
    else:
        strand_rho = np.nan
    if finite_pair.sum() > 2 and np.ptp(pair_score_array[finite_pair]) > 0:
        pair_hemi_rho = float(spearmanr(
            pair_score_array[finite_pair], max_hemi_array[finite_pair],
        ).statistic)
    else:
        pair_hemi_rho = np.nan
    rng = np.random.default_rng(args.seed)
    null_replicates = {str(value): [] for value in thresholds}
    for _ in range(args.null_replicates):
        replicate = {str(value): Counter() for value in thresholds}
        for result, symmetric_methylated in paired_results:
            posterior = _simulate_symmetric_result(
                result, symmetric_methylated, rng, args.run_bp,
            )
            both = result.c_observed & result.g_observed
            positions = result.reference_pos[both]
            posterior = posterior[both]
            for threshold in thresholds:
                values = replicate[str(threshold)]
                for state_index in (1, 2):
                    runs = _runs(
                        posterior[:, state_index] >= threshold,
                        positions, args.max_cpg_gap, args.min_run_cpg,
                    )
                    values["runs"] += len(runs)
                    values["run_cpgs"] += sum(
                        right - left + 1 for left, right in runs
                    )
        for threshold in thresholds:
            null_replicates[str(threshold)].append(replicate[str(threshold)])

    null_summary = {}
    for threshold, replicates in null_replicates.items():
        null_summary[threshold] = {}
        for metric in ("runs", "run_cpgs"):
            values = np.asarray([row[metric] for row in replicates], dtype=float)
            null_summary[threshold][metric] = {
                "mean": float(values.mean()) if len(values) else None,
                "q95": float(np.quantile(values, 0.95)) if len(values) else None,
                "max": int(values.max()) if len(values) else None,
            }
    return {
        "input": args.input,
        "region": args.region,
        "parameters": {
            "run_bp": args.run_bp,
            "baseline_radius": args.baseline_radius,
            "min_other": args.min_other,
            "min_run_cpg": args.min_run_cpg,
            "max_cpg_gap": args.max_cpg_gap,
        },
        "counts": dict(counters),
        "argmax_state_cpgs": dict(state_argmax),
        "mean_state_posterior": {
            name: (float(state_probability_array[:, index].mean())
                   if len(state_probability_array) else None)
            for index, name in enumerate(PAIRED_STATE_NAMES)
        },
        "strand_methylation_spearman": strand_rho,
        "pair_score_vs_max_hemi_spearman": pair_hemi_rho,
        "pair_score_quantiles": (
            {str(q): float(np.quantile(pair_score_array[finite_pair], q))
             for q in (0.1, 0.25, 0.5, 0.75, 0.9)}
            if finite_pair.any() else {}
        ),
        "candidate_pair_score_quantiles": {
            threshold: (
                {str(q): float(np.quantile(values, q))
                 for q in (0.1, 0.25, 0.5, 0.75, 0.9)}
                if values else {}
            )
            for threshold, values in candidate_pair_scores.items()
        },
        "pair_score_strata": {
            threshold: {
                cutoff: {
                    **dict(values),
                    "hemi_run_cpg_fraction": (
                        values["hemi_run_cpgs"] / values["both_cpgs"]
                        if values["both_cpgs"] else None
                    ),
                }
                for cutoff, values in cutoffs.items()
            }
            for threshold, cutoffs in score_strata.items()
        },
        "max_hemi_posterior_per_read_quantiles": (
            {str(q): float(np.quantile(max_hemi_array, q))
             for q in (0.5, 0.9, 0.95, 0.99, 1.0)}
            if len(max_hemi_array) else {}
        ),
        "thresholds": {
            threshold: dict(values)
            for threshold, values in threshold_stats.items()
        },
        "strict_candidate_threshold": strict_threshold,
        "strict_candidate_intervals": strict_candidate_intervals,
        "symmetric_generative_null": {
            "replicates": args.null_replicates,
            "seed": args.seed,
            "thresholds": null_summary,
        },
        "source_pair_sequence_qc": _source_pair_qc(
            args.source_bam, chrom, start, end, pair_records, thresholds,
        ),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-r", "--reference", required=True)
    parser.add_argument("--region", required=True)
    parser.add_argument("--thresholds", default="0.8,0.9,0.95,0.99")
    parser.add_argument("--run-bp", type=float, default=5000.0)
    parser.add_argument("--baseline-radius", type=int, default=250)
    parser.add_argument("--min-other", type=int, default=10)
    parser.add_argument("--min-run-cpg", type=int, default=2)
    parser.add_argument("--max-cpg-gap", type=float, default=5000.0)
    parser.add_argument("--io-threads", type=int, default=4)
    parser.add_argument("--null-replicates", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7123)
    parser.add_argument("--source-bam", default=None,
                        help="Original pre-merge BAM for source-pair sequence QC")
    args = parser.parse_args(argv)
    print(json.dumps(summarize(args), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
