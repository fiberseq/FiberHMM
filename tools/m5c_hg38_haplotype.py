#!/usr/bin/env python3
"""Explore read-level DAF m5C variance and phasing on hg38-mapped scDAF."""
from __future__ import annotations

import argparse
import glob
import os
from collections import defaultdict

import numpy as np
import pysam
from scipy.stats import rankdata

from fiberhmm.daf.m5c import (
    call_read_m5c,
    collect_read_observations,
)

DEFAULT_FACTORS = np.array([0.865, 1.045, 0.944, 1.146])


class CachedReference:
    def __init__(self, chrom, sequence):
        self.chrom = chrom
        self.sequence = sequence

    def fetch(self, chrom, start, end):
        if chrom != self.chrom:
            raise KeyError(chrom)
        return self.sequence[max(0, start):min(len(self.sequence), end)]


def auc(labels, scores):
    labels = np.asarray(labels, dtype=bool)
    if not labels.any() or labels.all():
        return np.nan
    ranks = rankdata(scores)
    n1, n0 = labels.sum(), (~labels).sum()
    return float((ranks[labels].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def haplotype_labels(path, chrom, min_mapping_quality=1):
    """Return unambiguous primary diploid-reference haplotype assignments.

    MAPQ-zero alignments are appropriate evidence for aggregate methylation,
    but they are not a ground-truth haplotype label: in homozygous sequence the
    primary MAT/PAT choice is arbitrary.  Phasing validation therefore keeps
    only mapping-quality-supported primary alignments and drops any qname with
    conflicting primary assignments.
    """
    labels = {}
    conflicts = set()
    with pysam.AlignmentFile(path, "rb") as bam:
        for suffix, haplotype in (("MATERNAL", 1), ("PATERNAL", 2)):
            contig = f"{chrom}_{suffix}"
            if contig not in bam.references:
                continue
            for read in bam.fetch(contig):
                if (read.is_unmapped or read.is_secondary or
                        read.is_supplementary or
                        read.mapping_quality < min_mapping_quality):
                    continue
                previous = labels.get(read.query_name)
                if previous is not None and previous[0] != haplotype:
                    conflicts.add(read.query_name)
                elif previous is None or read.mapping_quality > previous[1]:
                    labels[read.query_name] = (haplotype, read.mapping_quality)
    for query_name in conflicts:
        labels.pop(query_name, None)
    return labels, len(conflicts)


def collect(args):
    factors = DEFAULT_FACTORS / DEFAULT_FACTORS.mean()
    with pysam.FastaFile(args.reference) as fasta:
        reference = CachedReference(args.chrom, fasta.fetch(args.chrom).upper())
    hg38_paths = sorted(glob.glob(os.path.join(args.hg38_dir, "*.fiberhmm.bam")))
    read_rows = []
    cpg_rows = []
    read_id = 0
    for sample_index, hg38_path in enumerate(hg38_paths):
        sample = os.path.basename(hg38_path).split(".")[0]
        hg002_path = os.path.join(args.hg002_dir, f"{sample}.fiberhmm.bam")
        labels, label_conflicts = haplotype_labels(
            hg002_path, args.chrom, args.min_haplotype_mapq,
        )
        counters = defaultdict(int)
        with pysam.AlignmentFile(hg38_path, "rb", threads=args.io_threads) as bam:
            for read in bam.fetch(args.chrom):
                counters["reads"] += 1
                if (read.is_unmapped or read.is_secondary or
                        read.is_supplementary or
                        read.mapping_quality < args.min_hg38_mapq):
                    continue
                label = labels.get(read.query_name)
                if label is None:
                    continue
                haplotype, haplotype_mapq = label
                counters["labelled"] += 1
                observations = collect_read_observations(read, reference)
                if not observations:
                    continue
                result = call_read_m5c(
                    observations, factors, expected_run_bp=args.run_bp,
                    posterior_threshold=args.posterior,
                    baseline_radius=args.baseline_radius,
                    min_other=args.min_other, min_call_cpg=args.min_run_cpg,
                )
                if not len(result.reference_pos):
                    continue
                counters["scored"] += 1
                called = np.zeros(len(result.reference_pos), dtype=bool)
                for call in result.calls:
                    called |= ((result.query_pos >= call.start) &
                               (result.query_pos < call.end))
                n_cpg = len(result.reference_pos)
                read_rows.append((
                    sample_index, haplotype, haplotype_mapq, read.reference_start,
                    read.reference_end, n_cpg,
                    float(result.log_likelihood_ratio.sum()),
                    float(result.log_likelihood_ratio.mean()),
                    float(result.methylated_posterior.mean()),
                    int(called.sum()), len(result.calls),
                    float(result.baseline.mean()),
                ))
                for pos, llr, post, baseline, is_called in zip(
                    result.reference_pos, result.log_likelihood_ratio,
                    result.methylated_posterior, result.baseline, called,
                ):
                    cpg_rows.append((sample_index, haplotype, haplotype_mapq,
                                     read_id, int(pos),
                                     float(llr), float(post), float(baseline),
                                     int(is_called)))
                read_id += 1
        print(sample, dict(counters), "hap_labels", len(labels),
              "conflicts", label_conflicts, flush=True)
    read_dtype = np.dtype([
        ("sample", "i2"), ("hap", "i1"), ("hap_mapq", "i2"),
        ("start", "i4"), ("end", "i4"),
        ("n_cpg", "i2"), ("llr", "f4"), ("llr_per_cpg", "f4"),
        ("mean_posterior", "f4"), ("called_cpg", "i2"), ("n_calls", "i2"),
        ("baseline", "f4"),
    ])
    cpg_dtype = np.dtype([
        ("sample", "i2"), ("hap", "i1"), ("hap_mapq", "i2"),
        ("read_id", "i4"), ("pos", "i4"),
        ("llr", "f4"), ("posterior", "f4"), ("baseline", "f4"),
        ("called", "i1"),
    ])
    reads = np.asarray(read_rows, dtype=read_dtype)
    cpgs = np.asarray(cpg_rows, dtype=cpg_dtype)
    np.savez_compressed(args.output, reads=reads, cpgs=cpgs,
                        samples=np.asarray([os.path.basename(p).split(".")[0]
                                            for p in hg38_paths]))
    return reads, cpgs


def grouped_haplotype_effects(reads, window_size, min_reads=3):
    groups = defaultdict(lambda: [[], []])
    for row in reads:
        if row["n_cpg"] < 3:
            continue
        window = int(((int(row["start"]) + int(row["end"])) // 2) // window_size)
        groups[(int(row["sample"]), window)][int(row["hap"]) - 1].append(
            float(row["llr_per_cpg"]))
    effects = []
    for (sample, window), (hap1, hap2) in groups.items():
        if len(hap1) < min_reads or len(hap2) < min_reads:
            continue
        a, b = np.asarray(hap1), np.asarray(hap2)
        pooled = np.sqrt(((len(a) - 1) * a.var(ddof=1) +
                          (len(b) - 1) * b.var(ddof=1)) /
                         max(len(a) + len(b) - 2, 1))
        effect = (b.mean() - a.mean()) / max(pooled, 1e-6)
        effects.append((sample, window, len(a), len(b), a.mean(), b.mean(), effect))
    dtype = np.dtype([
        ("sample", "i2"), ("window", "i4"), ("n1", "i2"), ("n2", "i2"),
        ("mean1", "f4"), ("mean2", "f4"), ("effect", "f4"),
    ])
    return np.asarray(effects, dtype=dtype)


def leave_one_sample_out(reads, effects, window_size, min_train_samples=4,
                         min_abs_effect=0.5):
    results = []
    samples = np.unique(reads["sample"])
    for test_sample in samples:
        train = effects[effects["sample"] != test_sample]
        by_window = defaultdict(list)
        for row in train:
            by_window[int(row["window"])].append(float(row["effect"]))
        direction = {}
        for window, values in by_window.items():
            values = np.asarray(values)
            if len(values) >= min_train_samples and abs(values.mean()) >= min_abs_effect:
                direction[window] = np.sign(values.mean())
        test = reads[(reads["sample"] == test_sample) & (reads["n_cpg"] >= 3)]
        scores, labels = [], []
        for row in test:
            window = int(((int(row["start"]) + int(row["end"])) // 2) // window_size)
            if window in direction:
                scores.append(float(row["llr_per_cpg"]) * direction[window])
                labels.append(int(row["hap"]) == 2)
        results.append((int(test_sample), len(direction), len(scores), auc(labels, scores)))
    return results


def variance_component(reads, effects, window_size, permutations=100):
    groups = defaultdict(list)
    for row in reads:
        if row["n_cpg"] < 3:
            continue
        window = int(((int(row["start"]) + int(row["end"])) // 2) // window_size)
        groups[(int(row["sample"]), window)].append(
            (float(row["llr_per_cpg"]), int(row["hap"])))
    def explained(rows, permute, rng):
        score = np.asarray([x[0] for x in rows])
        hap = np.asarray([x[1] for x in rows])
        if permute:
            hap = rng.permutation(hap)
        total = np.var(score)
        if total <= 1e-9 or len(np.unique(hap)) < 2:
            return np.nan
        fitted = np.where(hap == 1, score[hap == 1].mean(), score[hap == 2].mean())
        return float(np.var(fitted) / total)
    rng = np.random.default_rng(19)
    observed = []
    null = []
    for rows in groups.values():
        h = np.asarray([x[1] for x in rows])
        if len(rows) < 8 or min(np.sum(h == 1), np.sum(h == 2)) < 3:
            continue
        observed.append(explained(rows, False, rng))
        null.extend(explained(rows, True, rng) for _ in range(permutations))
    return np.asarray(observed), np.asarray(null)


def _training_site_effects(cpgs, test_sample, min_per_haplotype=5,
                           min_t=2.0, min_difference=0.2):
    train = cpgs[cpgs["sample"] != test_sample]
    key = train["pos"].astype(np.int64) * 2 + (train["hap"] - 1)
    unique, inverse = np.unique(key, return_inverse=True)
    count = np.bincount(inverse)
    total = np.bincount(inverse, weights=train["llr"])
    square = np.bincount(inverse, weights=train["llr"] ** 2)
    position, haplotype = unique // 2, unique % 2
    left = np.flatnonzero(
        (position[:-1] == position[1:]) &
        (haplotype[:-1] == 0) & (haplotype[1:] == 1)
    )
    right = left + 1
    enough = ((count[left] >= min_per_haplotype) &
              (count[right] >= min_per_haplotype))
    left, right = left[enough], right[enough]
    mean1, mean2 = total[left] / count[left], total[right] / count[right]
    var1 = np.maximum(square[left] / count[left] - mean1 ** 2, 1e-6)
    var2 = np.maximum(square[right] / count[right] - mean2 ** 2, 1e-6)
    difference = mean2 - mean1
    t_value = difference / np.sqrt(var1 / count[left] + var2 / count[right])
    selected = ((np.abs(t_value) >= min_t) &
                (np.abs(difference) >= min_difference))
    return (position[left][selected], np.sign(difference[selected]),
            ((mean1 + mean2) / 2)[selected])


def site_controlled_cross_validation(cpgs, min_sites_per_read=2,
                                     randomizations=100):
    """Discover ASM at exact CpGs in other samples, classify held-out reads."""
    rng = np.random.default_rng(55)
    samples = np.unique(cpgs["sample"])
    observed = []
    null = [[] for _ in range(randomizations)]
    selected_counts, read_counts = [], []
    for test_sample in samples:
        positions, direction, midpoint = _training_site_effects(cpgs, test_sample)
        selected_counts.append(len(positions))
        test = cpgs[cpgs["sample"] == test_sample]
        index = np.searchsorted(positions, test["pos"])
        keep = ((index < len(positions)) &
                (positions[np.minimum(index, max(len(positions) - 1, 0))] == test["pos"])) \
            if len(positions) else np.zeros(len(test), dtype=bool)
        test, index = test[keep], index[keep]
        read_id = test["read_id"]
        unique_read, inverse = np.unique(read_id, return_inverse=True)
        count = np.bincount(inverse)
        haplotype = np.zeros(len(unique_read), dtype=np.int8)
        haplotype[inverse] = test["hap"]
        evaluable = count >= min_sites_per_read
        read_counts.append(int(evaluable.sum()))

        def classify(sign):
            oriented = (test["llr"] - midpoint[index]) * sign[index]
            score = np.bincount(inverse, weights=oriented)
            return auc(haplotype[evaluable] == 2, score[evaluable])

        observed.append(classify(direction))
        for permutation in range(randomizations):
            random_direction = rng.choice(np.array([-1.0, 1.0]), len(direction))
            null[permutation].append(classify(random_direction))
    observed_mean = float(np.nanmean(observed))
    null_mean = np.asarray([np.nanmean(values) for values in null])
    return {
        "sample_auc": observed,
        "mean_auc": observed_mean,
        "selected_sites": selected_counts,
        "evaluable_reads": read_counts,
        "null_mean": float(np.nanmean(null_mean)),
        "null_sd": float(np.nanstd(null_mean, ddof=1)),
        "empirical_p": float((1 + np.sum(null_mean >= observed_mean)) /
                             (1 + len(null_mean))),
    }


def analyze(reads, cpgs, randomizations=1000):
    print(f"\nreads={len(reads):,} CpGs={len(cpgs):,} "
          f"tagged_reads={np.mean(reads['called_cpg'] > 0):.3f} "
          f"tagged_CpGs={np.mean(cpgs['called'] > 0):.3f}")
    site_cv = site_controlled_cross_validation(
        cpgs, randomizations=randomizations,
    )
    print("\nexact-site leave-one-sample-out:")
    print(f"  mean AUC={site_cv['mean_auc']:.3f}; random-direction "
          f"{site_cv['null_mean']:.3f}+/-{site_cv['null_sd']:.3f}; "
          f"empirical p={site_cv['empirical_p']:.4f}")
    print(f"  sample AUC={np.round(site_cv['sample_auc'], 3)}")
    print(f"  selected sites={site_cv['selected_sites']}")
    print(f"  evaluable reads={site_cv['evaluable_reads']}")
    for window_size in (10_000, 50_000, 100_000):
        effects = grouped_haplotype_effects(reads, window_size)
        print(f"\nwindow={window_size:,} groups={len(effects):,} "
              f"median_abs_effect={np.median(np.abs(effects['effect'])):.3f}")
        by_window = defaultdict(list)
        for row in effects:
            by_window[int(row["window"])].append(float(row["effect"]))
        ranked = []
        for window, values in by_window.items():
            values = np.asarray(values)
            if len(values) >= 4:
                ranked.append((abs(values.mean()), values.mean(), len(values),
                               np.mean(np.sign(values) == np.sign(values.mean())), window))
        for _, mean, n, consistency, window in sorted(ranked, reverse=True)[:10]:
            print(f"  candidate {window*window_size:,}-{(window+1)*window_size:,} "
                  f"effect={mean:+.2f} samples={n} sign_consistency={consistency:.2f}")
        cv = leave_one_sample_out(reads, effects, window_size)
        valid_auc = [x[3] for x in cv if np.isfinite(x[3])]
        print("  leave-one-sample-out:", cv)
        print(f"  mean held-out AUC={np.mean(valid_auc):.3f}" if valid_auc else "  no evaluable AUC")
        observed, null = variance_component(reads, effects, window_size)
        print(f"  haplotype variance fraction observed={np.nanmean(observed):.4f} "
              f"permuted={np.nanmean(null):.4f} groups={len(observed)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hg38-dir", required=True,
                        help="Directory of hg38-mapped per-sample BAMs")
    parser.add_argument("--hg002-dir", required=True,
                        help="Directory of matched diploid-reference BAMs")
    parser.add_argument("--reference", required=True,
                        help="Indexed hg38 reference FASTA")
    parser.add_argument("--chrom", default="chr1")
    parser.add_argument("--output", default="m5c_hg38_chr1.npz")
    parser.add_argument("--load", action="store_true")
    parser.add_argument("--run-bp", type=float, default=5000)
    parser.add_argument("--posterior", type=float, default=0.99)
    parser.add_argument("--baseline-radius", type=int, default=250)
    parser.add_argument("--min-other", type=int, default=10)
    parser.add_argument("--min-run-cpg", type=int, default=2)
    parser.add_argument("--min-haplotype-mapq", type=int, default=1,
                        help="Minimum MAPQ on the HG002 diploid-reference "
                             "alignment used as the external haplotype label")
    parser.add_argument("--min-hg38-mapq", type=int, default=1,
                        help="Minimum MAPQ on the hg38 alignment being scored")
    parser.add_argument("--io-threads", type=int, default=2)
    parser.add_argument("--randomizations", type=int, default=1000,
                        help="Random site-direction sets for the exact-site null")
    args = parser.parse_args()
    if args.load:
        data = np.load(args.output, allow_pickle=False)
        reads, cpgs = data["reads"], data["cpgs"]
    else:
        reads, cpgs = collect(args)
    analyze(reads, cpgs, args.randomizations)


if __name__ == "__main__":
    main()
