#!/usr/bin/env python3
"""Stress-test the equal-odds per-read CpG HMM against lifted HG002 truth."""
from __future__ import annotations

import argparse
import glob
import importlib.util
from collections import defaultdict

import numpy as np
import pysam
from scipy.stats import rankdata, spearmanr

from fiberhmm.daf.m5c import (
    DDDA_FIVE_PRIME_FACTORS,
    collect_bam_observations,
    distance_forward_backward,
    estimate_five_prime_factors,
    score_read_observations,
)

def load_truth(path, fasta, chrom, start, end):
    spec = importlib.util.spec_from_file_location("m5c_truth_external", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.load_truth(fasta, chrom, start, end, verbose=True)


def auc(labels, scores):
    labels = np.asarray(labels, dtype=bool)
    if not labels.any() or labels.all():
        return np.nan
    ranks = rankdata(scores)
    n1, n0 = labels.sum(), (~labels).sum()
    return float((ranks[labels].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def confident_runs(posterior, threshold, min_run, methylated=True):
    selected = posterior >= threshold if methylated else posterior <= 1.0 - threshold
    keep = np.zeros(len(selected), dtype=bool)
    i = 0
    while i < len(selected):
        if not selected[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(selected) and selected[j + 1]:
            j += 1
        if j - i + 1 >= min_run:
            keep[i:j + 1] = True
        i = j + 1
    return keep


def prepare(args, baseline_radius):
    observations = []
    offset = 0
    with pysam.FastaFile(args.reference) as fasta:
        truth = load_truth(args.truth_module, fasta, args.chrom, args.start, args.end)
        for path in sorted(glob.glob(args.bam_glob)):
            with pysam.AlignmentFile(path, "rb") as bam:
                part, offset = collect_bam_observations(
                    bam, fasta, args.chrom, args.start, args.end,
                    molecule_offset=offset,
                )
                observations.extend(part)
    factors = (estimate_five_prime_factors(observations)
               if args.estimate_factors else
               DDDA_FIVE_PRIME_FACTORS / DDDA_FIVE_PRIME_FACTORS.mean())
    by_molecule = defaultdict(list)
    for obs in observations:
        by_molecule[obs.molecule].append(obs)
    sequences = []
    for molecule, records in by_molecule.items():
        ref, query, baseline, deam, emission = score_read_observations(
            records, factors, baseline_radius=baseline_radius,
            min_other=args.min_other,
        )
        if not len(ref):
            continue
        beta = np.asarray([truth.get(int(pos), np.nan) for pos in ref])
        valid = np.isfinite(beta)
        if valid.any():
            sequences.append((molecule, ref[valid], emission[valid], beta[valid],
                              baseline[valid], deam[valid]))
    print(f"prepared radius={baseline_radius}: observations={len(observations):,} "
          f"molecules={offset:,} scored_reads={len(sequences):,} "
          f"truth_observations={sum(len(s[1]) for s in sequences):,} "
          f"5prime={np.round(factors, 3)}")
    return sequences


def evaluate(sequences, run_bp, threshold, min_run):
    post_all, truth_all, baseline_all = [], [], []
    positive_truth, negative_truth = [], []
    site_post = defaultdict(list)
    site_truth = {}
    reads_called = set()
    total_reads = len(sequences)
    transition_tp = transition_fn = transition_fp = 0
    for molecule, positions, emission, truth, baseline, _deam in sequences:
        posterior = distance_forward_backward(emission, positions, run_bp)[:, 1]
        positive = confident_runs(posterior, threshold, min_run, True)
        negative = confident_runs(posterior, threshold, min_run, False)
        if positive.any():
            reads_called.add(molecule)
            positive_truth.extend(truth[positive])
        if negative.any():
            negative_truth.extend(truth[negative])
        post_all.extend(posterior)
        truth_all.extend(truth)
        baseline_all.extend(baseline)
        for pos, post, beta in zip(positions, posterior, truth):
            site_post[int(pos)].append(float(post))
            site_truth[int(pos)] = float(beta)
        # Stress boundary behavior only where adjacent truth sites are decisive.
        truth_state = np.where(truth >= 0.7, 1, np.where(truth <= 0.3, 0, -1))
        called_state = np.where(posterior >= threshold, 1,
                                np.where(posterior <= 1 - threshold, 0, -1))
        for i in range(1, len(truth)):
            true_switch = truth_state[i - 1] >= 0 and truth_state[i] >= 0 and \
                truth_state[i - 1] != truth_state[i]
            call_switch = called_state[i - 1] >= 0 and called_state[i] >= 0 and \
                called_state[i - 1] != called_state[i]
            transition_tp += int(true_switch and call_switch)
            transition_fn += int(true_switch and not call_switch)
            transition_fp += int(call_switch and not true_switch)
    posterior = np.asarray(post_all)
    truth = np.asarray(truth_all)
    baseline = np.asarray(baseline_all)
    positive_truth = np.asarray(positive_truth)
    negative_truth = np.asarray(negative_truth)
    extreme = (truth <= 0.1) | (truth >= 0.9)
    positions = sorted(site_post)
    mean_site_post = np.asarray([np.mean(site_post[p]) for p in positions])
    mean_site_truth = np.asarray([site_truth[p] for p in positions])
    return {
        "rho_obs": float(spearmanr(posterior, truth).statistic),
        "rho_site": float(spearmanr(mean_site_post, mean_site_truth).statistic),
        "auc_extreme": auc(truth[extreme] >= 0.9, posterior[extreme]),
        "positive_n": len(positive_truth),
        "positive_truth": float(np.mean(positive_truth)) if len(positive_truth) else np.nan,
        "positive_precision": float(np.mean(positive_truth > 0.5)) if len(positive_truth) else np.nan,
        "negative_n": len(negative_truth),
        "negative_truth": float(np.mean(negative_truth)) if len(negative_truth) else np.nan,
        "read_coverage": len(reads_called) / max(total_reads, 1),
        "cpg_coverage": len(positive_truth) / max(len(truth), 1),
        "boundary_recall": transition_tp / max(transition_tp + transition_fn, 1),
        "boundary_precision": transition_tp / max(transition_tp + transition_fp, 1),
        "n_obs": len(truth),
        "low_b_rho": float(spearmanr(posterior[baseline < 0.4], truth[baseline < 0.4]).statistic),
        "high_b_rho": float(spearmanr(posterior[baseline >= 0.4], truth[baseline >= 0.4]).statistic),
    }


def print_result(radius, run_bp, threshold, min_run, result):
    print(f"{radius:6d} {run_bp:7.0f} {threshold:5.3f} {min_run:3d} "
          f"{result['rho_obs']:+.3f} {result['rho_site']:+.3f} "
          f"{result['auc_extreme']:.3f} {result['positive_n']:7d} "
          f"{result['positive_truth']:.3f} {result['positive_precision']:.3f} "
          f"{result['negative_n']:7d} {result['negative_truth']:.3f} "
          f"{result['read_coverage']:.3f} {result['cpg_coverage']:.3f} "
          f"{result['boundary_recall']:.3f} {result['boundary_precision']:.3f} "
          f"{result['low_b_rho']:+.3f} {result['high_b_rho']:+.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bam-glob", required=True,
                        help="Glob for input DddA FiberHMM BAMs")
    parser.add_argument("--reference", required=True,
                        help="Indexed diploid reference FASTA")
    parser.add_argument("--truth-module", required=True,
                        help="Python module providing load_truth()")
    parser.add_argument("--chrom", default="chr1_MATERNAL")
    parser.add_argument("--start", type=int, default=20_000_000)
    parser.add_argument("--end", type=int, default=23_000_000)
    parser.add_argument("--min-other", type=int, default=10)
    parser.add_argument("--baseline-radii", default="250,500,1000")
    parser.add_argument("--run-bp", default="250,500,1000,2000,5000,10000")
    parser.add_argument("--thresholds", default="0.8,0.9,0.95,0.975,0.99")
    parser.add_argument("--min-runs", default="1,2,3")
    parser.add_argument("--estimate-factors", action="store_true",
                        help="Re-estimate 5' factors in this region instead of "
                             "using the production calibrated constants")
    args = parser.parse_args()
    radii = [int(v) for v in args.baseline_radii.split(",")]
    run_lengths = [float(v) for v in args.run_bp.split(",")]
    thresholds = [float(v) for v in args.thresholds.split(",")]
    min_runs = [int(v) for v in args.min_runs.split(",")]
    print("radius run_bp post min rho_obs rho_site auc_ext pos_n pos_truth pos_prec "
          "neg_n neg_truth read_cov cpg_cov boundary_rec boundary_prec low_b high_b")
    for radius in radii:
        sequences = prepare(args, radius)
        for run_bp in run_lengths:
            for threshold in thresholds:
                for min_run in min_runs:
                    print_result(radius, run_bp, threshold, min_run,
                                 evaluate(sequences, run_bp, threshold, min_run))


if __name__ == "__main__":
    main()
