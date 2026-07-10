#!/usr/bin/env python3
"""Filter hg38 m5C phasing sites with read-derived haplotype context consensus.

Exact-site methylation can classify haplotypes either because methylation is
allele-specific or because a nearby germline variant changes DddA sequence
preference/mappability.  This control reconstructs a local reference-oriented
query consensus for both haplotypes and reruns cross-validation only at sites
whose two consensus strings are identical.
"""
from __future__ import annotations

import argparse
import os
from collections import Counter, defaultdict

import numpy as np
import pysam

from fiberhmm.core.bam_reader import cigar_to_query_ref
from m5c_hg38_haplotype import (
    _training_site_effects,
    haplotype_labels,
    site_controlled_cross_validation,
)


RESTORE = str.maketrans({"Y": "C", "R": "G", "y": "C", "r": "G"})


def candidate_positions(cpgs):
    positions = set()
    for sample in np.unique(cpgs["sample"]):
        selected, _direction, _midpoint = _training_site_effects(
            cpgs, int(sample),
        )
        positions.update(int(value) for value in selected)
    return np.asarray(sorted(positions), dtype=np.int64)


def collect_contexts(args, samples, positions):
    counts = defaultdict(Counter)
    for sample in samples:
        hg38_path = os.path.join(args.hg38_dir, f"{sample}.fiberhmm.bam")
        hg002_path = os.path.join(args.hg002_dir, f"{sample}.fiberhmm.bam")
        labels, conflicts = haplotype_labels(
            hg002_path, args.chrom, args.min_haplotype_mapq,
        )
        reads = used = kmers = 0
        with pysam.AlignmentFile(hg38_path, "rb", threads=args.io_threads) as bam:
            for read in bam.fetch(args.chrom):
                reads += 1
                if (read.is_unmapped or read.is_secondary or
                        read.is_supplementary or
                        read.mapping_quality < args.min_hg38_mapq or
                        read.query_sequence is None):
                    continue
                label = labels.get(read.query_name)
                if label is None:
                    continue
                haplotype = label[0]
                left = np.searchsorted(
                    positions, read.reference_start + args.flank, side="left",
                )
                right = np.searchsorted(
                    positions, read.reference_end - args.flank, side="right",
                )
                if right <= left:
                    continue
                q_to_ref = cigar_to_query_ref(read)
                aligned = q_to_ref >= 0
                ref = q_to_ref[aligned]
                query = np.flatnonzero(aligned)
                if not len(ref):
                    continue
                sequence = read.query_sequence.translate(RESTORE).upper()
                used += 1
                for position in positions[left:right]:
                    target = np.arange(
                        position - args.flank, position + args.flank + 1,
                        dtype=np.int64,
                    )
                    index = np.searchsorted(ref, target)
                    if np.any(index >= len(ref)):
                        continue
                    if not np.array_equal(ref[index], target):
                        continue
                    kmer = "".join(sequence[int(value)] for value in query[index])
                    if set(kmer) <= set("ACGT"):
                        counts[(int(position), haplotype)][kmer] += 1
                        kmers += 1
        print(sample, f"reads={reads:,}", f"used={used:,}",
              f"kmers={kmers:,}", f"labels={len(labels):,}",
              f"conflicts={conflicts}", flush=True)
    return counts


def consensus(counter):
    if not counter:
        return "", 0, 0.0
    value, count = counter.most_common(1)[0]
    total = sum(counter.values())
    return value, total, count / total


def safe_contexts(positions, counts, min_count, min_fraction):
    rows = []
    safe = []
    for position in positions:
        one, n1, f1 = consensus(counts[(int(position), 1)])
        two, n2, f2 = consensus(counts[(int(position), 2)])
        keep = (one == two and bool(one) and n1 >= min_count and
                n2 >= min_count and f1 >= min_fraction and f2 >= min_fraction)
        rows.append((int(position), one, n1, f1, two, n2, f2, int(keep)))
        if keep:
            safe.append(int(position))
    return np.asarray(safe, dtype=np.int64), rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="NPZ from m5c_hg38_haplotype.py")
    parser.add_argument("--hg38-dir", required=True,
                        help="Directory of hg38-mapped per-sample BAMs")
    parser.add_argument("--hg002-dir", required=True,
                        help="Directory of matched diploid-reference BAMs")
    parser.add_argument("--chrom", default="chr22")
    parser.add_argument("--flank", type=int, default=3)
    parser.add_argument("--min-count", type=int, default=5)
    parser.add_argument("--min-consensus-fraction", type=float, default=0.8)
    parser.add_argument("--min-haplotype-mapq", type=int, default=1)
    parser.add_argument("--min-hg38-mapq", type=int, default=1)
    parser.add_argument("--io-threads", type=int, default=2)
    parser.add_argument("--output-tsv", default="")
    parser.add_argument("--randomizations", type=int, default=1000)
    args = parser.parse_args()

    data = np.load(args.input, allow_pickle=False)
    cpgs, samples = data["cpgs"], data["samples"]
    positions = candidate_positions(cpgs)
    print(f"candidate positions={len(positions):,}")
    counts = collect_contexts(args, samples, positions)
    safe, rows = safe_contexts(
        positions, counts, args.min_count, args.min_consensus_fraction,
    )
    print(f"identical high-confidence contexts={len(safe):,}/{len(positions):,}")
    if args.output_tsv:
        with open(args.output_tsv, "w") as handle:
            handle.write("position\thap1_context\thap1_n\thap1_fraction\t"
                         "hap2_context\thap2_n\thap2_fraction\tkeep\n")
            for row in rows:
                handle.write("\t".join(str(value) for value in row) + "\n")
    keep = np.isin(cpgs["pos"], safe)
    result = site_controlled_cross_validation(
        cpgs[keep], randomizations=args.randomizations,
    )
    print(f"context-matched observations={int(keep.sum()):,}")
    print(f"mean AUC={result['mean_auc']:.4f}; random-direction "
          f"{result['null_mean']:.4f}+/-{result['null_sd']:.4f}; "
          f"empirical p={result['empirical_p']:.4f}")
    print("sample AUC", np.round(result["sample_auc"], 3))
    print("selected sites", result["selected_sites"])
    print("evaluable reads", result["evaluable_reads"])


if __name__ == "__main__":
    main()
