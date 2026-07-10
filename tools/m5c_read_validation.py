#!/usr/bin/env python3
"""Evaluate equal-odds per-read DAF m5C evidence against lifted CpG truth."""
from __future__ import annotations

import argparse
import glob
import importlib.util

import numpy as np
import pysam
from scipy.stats import rankdata, spearmanr

from fiberhmm.daf.m5c import (
    F_METH,
    U_UNMETH,
    collect_bam_observations,
    estimate_five_prime_factors,
)


def load_truth_module(path):
    spec = importlib.util.spec_from_file_location("m5c_truth_external", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def auc(labels, scores):
    labels = np.asarray(labels, dtype=bool)
    n1, n0 = labels.sum(), (~labels).sum()
    if not n1 or not n0:
        return np.nan
    ranks = rankdata(scores)
    return float((ranks[labels].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def score_observations(observations, truth, start, end, baseline_window=1000,
                       min_other=10):
    """Return molecule, position, LLR and truth beta for usable CpGs."""
    factors = estimate_five_prime_factors(observations)
    by_window = {}
    for obs in observations:
        by_window.setdefault((obs.reference_pos - start) // baseline_window, []).append(obs)
    rows = []
    for records in by_window.values():
        baseline = {}
        for obs in records:
            if not obs.is_cpg:
                pair = baseline.setdefault(obs.molecule, [0, 0])
                pair[0] += int(obs.deaminated)
                pair[1] += 1
        baseline = {m: h / n for m, (h, n) in baseline.items() if n >= min_other}
        for obs in records:
            beta = truth.get(obs.reference_pos)
            if not obs.is_cpg or obs.molecule not in baseline or beta is None:
                continue
            b = np.clip(baseline[obs.molecule] * factors[obs.five_prime_base], 0.01, 0.97)
            pu = 1.0 - (1.0 - b) ** U_UNMETH
            pm = 1.0 - (1.0 - b) ** F_METH
            llr = np.log(pm / pu) if obs.deaminated else np.log((1.0 - pm) / (1.0 - pu))
            rows.append((obs.molecule, obs.reference_pos, llr, beta, b))
    return np.asarray(rows, dtype=float)


def summarize(rows, block_size, min_cpg):
    groups = {}
    for molecule, pos, llr, beta, baseline in rows:
        key = (int(molecule), int(pos) // block_size)
        value = groups.setdefault(key, [0.0, 0.0, 0.0, 0, 0])
        value[0] += llr
        value[1] += beta
        value[2] += baseline
        value[3] += 1
        value[4] += int(beta > 0.5)
    values = np.asarray([
        (llr, beta_sum / n, baseline_sum / n, n, methylated_sites)
        for llr, beta_sum, baseline_sum, n, methylated_sites in groups.values()
        if n >= min_cpg
    ])
    if not len(values):
        return None
    llr, truth, baseline, n, methylated_sites = values.T
    extreme = (truth <= 0.1) | (truth >= 0.9)
    confident_m = llr >= np.log(19)
    confident_u = llr <= -np.log(19)
    return {
        "n": len(values),
        "median_cpg": float(np.median(n)),
        "rho_sum": float(spearmanr(llr, truth).statistic),
        "rho_per_cpg": float(spearmanr(llr / n, truth).statistic),
        "auc_extreme": auc(truth[extreme] >= 0.9, llr[extreme]),
        "extreme_n": int(extreme.sum()),
        "accuracy_equal_prior": float(np.mean((llr > 0) == (truth > 0.5))),
        "confident_fraction": float(np.mean(np.abs(llr) >= np.log(19))),
        "confident_accuracy": float(np.mean(
            (llr[np.abs(llr) >= np.log(19)] > 0) ==
            (truth[np.abs(llr) >= np.log(19)] > 0.5)
        )) if np.any(np.abs(llr) >= np.log(19)) else np.nan,
        "meth_blocks": int(confident_m.sum()),
        "meth_truth_mean": float(np.average(
            truth[confident_m], weights=n[confident_m],
        )) if confident_m.any() else np.nan,
        "meth_site_precision": float(
            methylated_sites[confident_m].sum() / n[confident_m].sum()
        ) if confident_m.any() else np.nan,
        "unmeth_blocks": int(confident_u.sum()),
        "unmeth_truth_mean": float(np.average(
            truth[confident_u], weights=n[confident_u],
        )) if confident_u.any() else np.nan,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bam-glob", required=True,
                        help="Glob for input DddA FiberHMM BAMs")
    parser.add_argument("--reference", required=True,
                        help="Indexed diploid reference FASTA")
    parser.add_argument("--truth-module", required=True,
                        help="Python module providing load_truth()")
    parser.add_argument("--chrom", default="chr1_MATERNAL")
    parser.add_argument("--start", type=int, default=30_000_000)
    parser.add_argument("--end", type=int, default=33_000_000)
    args = parser.parse_args()

    bams = sorted(glob.glob(args.bam_glob))
    observations = []
    offset = 0
    with pysam.FastaFile(args.reference) as fasta:
        truth_module = load_truth_module(args.truth_module)
        truth = truth_module.load_truth(
            fasta, args.chrom, args.start, args.end, verbose=True,
        )
        for path in bams:
            with pysam.AlignmentFile(path, "rb") as bam:
                part, offset = collect_bam_observations(
                    bam, fasta, args.chrom, args.start, args.end,
                    molecule_offset=offset,
                )
                observations.extend(part)
    rows = score_observations(observations, truth, args.start, args.end)
    print(f"observations={len(observations):,} molecules={offset:,} "
          f"truth-matched CpG observations={len(rows):,}")
    print("\nblock_kb min_cpg blocks median_cpg rho_sum rho_per_cpg "
          "auc_extreme equal_prior_acc confident_frac confident_acc "
          "meth_n meth_truth meth_site_prec unmeth_n unmeth_truth")
    for block in (1_000, 2_000, 5_000, 10_000, 20_000, 50_000):
        for min_cpg in (1, 3, 5, 10):
            result = summarize(rows, block, min_cpg)
            if result is None:
                continue
            print(f"{block//1000:8d} {min_cpg:7d} {result['n']:6d} "
                  f"{result['median_cpg']:10.1f} {result['rho_sum']:+.3f} "
                  f"{result['rho_per_cpg']:+.3f} {result['auc_extreme']:.3f} "
                  f"{result['accuracy_equal_prior']:.3f} "
                  f"{result['confident_fraction']:.3f} {result['confident_accuracy']:.3f}")
            print(f"{'':8s} {'':7s} {'':6s} {'':10s} {'':6s} {'':11s} "
                  f"{'':11s} {'':15s} {'':14s} {'':13s} "
                  f"{result['meth_blocks']:6d} {result['meth_truth_mean']:.3f} "
                  f"{result['meth_site_precision']:.3f} {result['unmeth_blocks']:8d} "
                  f"{result['unmeth_truth_mean']:.3f}")


if __name__ == "__main__":
    main()
