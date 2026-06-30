#!/usr/bin/env python3
"""fiberhmm-dedup -- PCR-duplicate detection for DAF-seq via deamination fingerprints.

DAF-seq libraries are typically amplicons: every read piles up on the same
locus with primer-fixed ends, so coordinate-based dedup (Picard / samtools
markdup) has no positional signal to work with and would collapse distinct
molecules together. There are no UMIs. The molecular fingerprint is instead
the per-read **deamination pattern** -- the set of reference positions
converted by the deaminase (R/Y, MM/ML dU, or MD mismatch). PCR copies of one
original molecule share that pattern, but rarely *exactly*: sequencing error
and missed/over-called deaminations typically perturb a handful of the ~hundreds
of calls, so exact-match dedup misses most duplicates.

This tool fingerprints each read's deamination set, clusters reads whose sets
match within a Jaccard threshold (MinHash + LSH, near-linear), and by default
collapses each cluster to one representative read. Pass ``--flag-only`` to
instead keep every read and just set the SAM 0x400 duplicate flag + cluster
tags. Reads with too few deamination calls to fingerprint reliably
(``--min-deam``) are passed through untouched.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from collections import Counter, defaultdict
from typing import List, Optional

import numpy as np
import pysam

from fiberhmm.cli.extract_tags import _build_query_to_ref, _deam_positions_list

_PRIME = (1 << 61) - 1
# Cluster-tag names (lowercase = locally-defined per SAM spec).
_TAG_CLUSTER = 'di'   # duplicate-cluster id (0-based)
_TAG_CLUSTER_SIZE = 'ds'  # number of reads in the cluster


class _UnionFind:
    def __init__(self, n: int):
        self.p = list(range(n))

    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb


def _minhash(pos_sets: List[Optional[frozenset]], k: int, seed: int) -> np.ndarray:
    """K-wide MinHash signature per read; rows of all-INT64_MAX for un-fingerprintable reads."""
    rng = np.random.default_rng(seed)
    a = rng.integers(1, _PRIME, size=k, dtype=np.int64)
    b = rng.integers(0, _PRIME, size=k, dtype=np.int64)
    sig = np.full((len(pos_sets), k), np.iinfo(np.int64).max, dtype=np.int64)
    for i, p in enumerate(pos_sets):
        if not p:
            continue
        arr = np.fromiter(p, dtype=np.int64, count=len(p))
        sig[i] = ((np.multiply.outer(arr, a) + b) % _PRIME).min(axis=0)
    return sig


def _jaccard(a: frozenset, b: frozenset) -> float:
    inter = len(a & b)
    if inter == 0:
        return 0.0
    return inter / (len(a) + len(b) - inter)


def cluster_reads(pos_sets, group_keys, min_jaccard, k, bands, seed):
    """Cluster fingerprintable reads (pos_sets[i] truthy) within a shared
    group_key by deamination-set Jaccard >= min_jaccard. Returns a labels
    array (cluster id per read; -1 for un-fingerprintable reads).

    LSH banding surfaces candidate pairs; each candidate is verified by exact
    Jaccard, so precision comes from verification and only recall depends on
    the band/row geometry. Within each band bucket we union members to the
    bucket representative (linear in members), and union-find transitivity
    stitches clusters across bands.
    """
    n = len(pos_sets)
    rows = k // bands
    if rows < 1:
        raise ValueError(f"--num-hashes ({k}) must be >= --bands ({bands})")
    sig = _minhash(pos_sets, k, seed)
    uf = _UnionFind(n)
    for band in range(bands):
        cols = sig[:, band * rows:(band + 1) * rows]
        buckets = defaultdict(list)
        for i in range(n):
            if not pos_sets[i]:
                continue
            buckets[(group_keys[i],) + tuple(cols[i].tolist())].append(i)
        for ids in buckets.values():
            if len(ids) < 2:
                continue
            rep = ids[0]
            rep_set = pos_sets[rep]
            for j in ids[1:]:
                if _jaccard(rep_set, pos_sets[j]) >= min_jaccard:
                    uf.union(rep, j)

    labels = np.full(n, -1, dtype=np.int64)
    remap = {}
    for i in range(n):
        if not pos_sets[i]:
            continue
        root = uf.find(i)
        if root not in remap:
            remap[root] = len(remap)
        labels[i] = remap[root]
    return labels


def _read_quality(read) -> tuple:
    """Representative-selection key: prefer the most-informative, highest-MAPQ,
    longest-aligned, primary read. Higher = better (kept)."""
    nm = read.get_tag('NM') if read.has_tag('NM') else 0
    return (
        0 if (read.is_secondary or read.is_supplementary) else 1,
        read.mapping_quality,
        read.reference_length or 0,
        -nm,
    )


def run_dedup(in_bam, out_bam, min_jaccard=0.95, min_deam=10, ignore_strand=False,
              k=32, bands=8, seed=7, collapse=True, prob_threshold=0,
              stats_tsv=None, io_threads=4):
    t0 = time.time()
    # ---- Pass 1: fingerprint every record (order = until_eof iteration) ----
    pos_sets: List[Optional[frozenset]] = []
    group_keys: List[Optional[tuple]] = []
    quals: List[tuple] = []
    n_total = n_fingerprintable = n_lowdeam = n_unmapped = 0

    with pysam.AlignmentFile(in_bam, "rb", check_sq=False) as bam:
        for read in bam.fetch(until_eof=True):
            n_total += 1
            clusterable = (
                not read.is_unmapped and not read.is_secondary
                and not read.is_supplementary and read.query_sequence is not None
            )
            if not clusterable:
                pos_sets.append(None)
                group_keys.append(None)
                quals.append(())
                if read.is_unmapped:
                    n_unmapped += 1
                continue
            pairs = _build_query_to_ref(read)
            calls = _deam_positions_list(read, pairs, prob_threshold)
            if len(calls) < min_deam:
                pos_sets.append(None)
                group_keys.append(None)
                quals.append(())
                n_lowdeam += 1
                continue
            pos_sets.append(frozenset(p for p, _ in calls))
            strand_key = '' if ignore_strand else ('-' if read.is_reverse else '+')
            group_keys.append((read.reference_id, strand_key))
            quals.append(_read_quality(read))
            n_fingerprintable += 1

    if n_fingerprintable == 0:
        print("No fingerprintable reads (no deamination calls found). Nothing to do.\n"
              "  Is this a DAF-seq BAM with R/Y, MM/ML dU, or MD-encoded deaminations?",
              file=sys.stderr)
        return None

    print(f"Pass 1: {n_total:,} records  "
          f"({n_fingerprintable:,} fingerprintable, {n_lowdeam:,} below --min-deam, "
          f"{n_unmapped:,} unmapped) [{time.time()-t0:.0f}s]", file=sys.stderr)

    # ---- Cluster ----
    labels = cluster_reads(pos_sets, group_keys, min_jaccard, k, bands, seed)
    cluster_sizes = Counter(int(c) for c in labels if c >= 0)
    n_clusters = len(cluster_sizes)
    n_dups = n_fingerprintable - n_clusters
    dup_pct = 100.0 * n_dups / n_fingerprintable

    # Pick one representative (best quality) per cluster.
    best_in_cluster = {}
    for idx, c in enumerate(labels):
        if c < 0:
            continue
        c = int(c)
        if c not in best_in_cluster or quals[idx] > quals[best_in_cluster[c]]:
            best_in_cluster[c] = idx
    representatives = set(best_in_cluster.values())

    print(f"Clustering (Jaccard >= {min_jaccard}): {n_fingerprintable:,} reads -> "
          f"{n_clusters:,} molecules | {n_dups:,} duplicates ({dup_pct:.1f}%) | "
          f"mean {n_fingerprintable / n_clusters:.2f} copies/molecule "
          f"[{time.time()-t0:.0f}s]", file=sys.stderr)
    size_hist = Counter(cluster_sizes.values())
    singletons = size_hist.get(1, 0)
    print(f"  molecules with 1 copy (singletons): {singletons:,} "
          f"({100.0*singletons/n_clusters:.1f}% of molecules); "
          f"largest cluster: {max(cluster_sizes.values())} reads", file=sys.stderr)

    if stats_tsv:
        with open(stats_tsv, 'w') as fh:
            fh.write("cluster_id\tn_reads\n")
            for c, sz in sorted(cluster_sizes.items()):
                fh.write(f"{c}\t{sz}\n")

    # ---- Pass 2: write output (same iteration order = same indexing) ----
    n_written = n_flagged = 0
    with pysam.AlignmentFile(in_bam, "rb", check_sq=False) as bam:
        out = pysam.AlignmentFile(out_bam, "wb", template=bam, threads=io_threads)
        for idx, read in enumerate(bam.fetch(until_eof=True)):
            c = int(labels[idx])
            if c >= 0:
                sz = cluster_sizes[c]
                if sz > 1:
                    read.set_tag(_TAG_CLUSTER, c, value_type='i')
                    read.set_tag(_TAG_CLUSTER_SIZE, sz, value_type='i')
                is_rep = idx in representatives
                if not is_rep:
                    read.is_duplicate = True
                    n_flagged += 1
                    if collapse:
                        continue
            out.write(read)
            n_written += 1
        out.close()

    mode = "collapsed" if collapse else "flagged"
    print(f"Pass 2: wrote {n_written:,} reads ({n_flagged:,} duplicates {mode}) "
          f"-> {out_bam} [{time.time()-t0:.0f}s]", file=sys.stderr)
    return {
        'n_total': n_total, 'n_fingerprintable': n_fingerprintable,
        'n_clusters': n_clusters, 'n_duplicates': n_dups,
        'duplication_pct': dup_pct, 'n_written': n_written,
    }


def main():
    parser = argparse.ArgumentParser(
        prog='fiberhmm-dedup',
        description='PCR-duplicate detection for DAF-seq via deamination-pattern '
                    'fingerprints (for amplicon / UMI-less libraries where '
                    'coordinate dedup does not apply).',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Default: collapse to one representative read per molecule
    fiberhmm-dedup -i sample.bam -o sample.dedup.bam

    # Keep all reads, just mark duplicates (0x400 + di/ds cluster tags)
    fiberhmm-dedup -i sample.bam -o sample.markdup.bam --flag-only

    # Looser matching for error-rich reads + per-cluster stats
    fiberhmm-dedup -i sample.bam -o out.bam --min-jaccard 0.90 --stats-tsv clusters.tsv
        """,
    )
    parser.add_argument('-i', '--input', required=True, help='Input DAF-seq BAM (indexed not required)')
    parser.add_argument('-o', '--output', required=True, help='Output BAM')
    parser.add_argument('--min-jaccard', type=float, default=0.95,
                        help='Min deamination-set Jaccard to call two reads the same '
                             'molecule (default 0.95; the bimodal gap sits ~0.90-0.95). '
                             'Lower = more aggressive collapsing.')
    parser.add_argument('--flag-only', action='store_true',
                        help='Keep all reads and only mark duplicates (set the 0x400 '
                             'duplicate flag + di/ds tags on non-representatives). '
                             'Default: collapse each cluster to one representative read.')
    parser.add_argument('--min-deam', type=int, default=10,
                        help='Reads with fewer than this many deamination calls are '
                             'not fingerprintable; passed through untouched (default 10).')
    parser.add_argument('--ignore-strand', action='store_true',
                        help='Cluster across strands. Default: only reads on the same '
                             'strand (deamination is strand-specific) can be duplicates.')
    parser.add_argument('-p', '--prob-threshold', type=int, default=0,
                        help='Min ML probability for MM/ML-native dU calls (0-255, '
                             'default 0 = accept all). Ignored for R/Y and MD sources.')
    parser.add_argument('--num-hashes', type=int, default=32,
                        help='MinHash signature width (default 32).')
    parser.add_argument('--bands', type=int, default=8,
                        help='LSH bands; rows = num-hashes / bands (default 8 -> rows 4). '
                             'More bands = higher recall, more candidate pairs.')
    parser.add_argument('--seed', type=int, default=7, help='MinHash RNG seed (default 7).')
    parser.add_argument('--stats-tsv', default=None,
                        help='Write a cluster_id<TAB>n_reads table to this path.')
    parser.add_argument('--io-threads', type=int, default=4,
                        help='htslib BAM compression threads for output (default 4).')
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: input not found: {args.input}", file=sys.stderr)
        sys.exit(1)
    if not 0.0 < args.min_jaccard <= 1.0:
        print("Error: --min-jaccard must be in (0, 1].", file=sys.stderr)
        sys.exit(1)

    run_dedup(
        in_bam=args.input, out_bam=args.output,
        min_jaccard=args.min_jaccard, min_deam=args.min_deam,
        ignore_strand=args.ignore_strand, k=args.num_hashes, bands=args.bands,
        seed=args.seed, collapse=not args.flag_only, prob_threshold=args.prob_threshold,
        stats_tsv=args.stats_tsv, io_threads=args.io_threads,
    )


if __name__ == '__main__':
    main()
