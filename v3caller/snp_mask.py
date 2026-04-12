#!/usr/bin/env python3
"""Detect probable SNPs from DAF-seq / Hia5 amplicon data.

On heavily-enriched amplicon data (NAPA, ENH30, UBA1, etc.), any
reference position where ALL or nearly ALL reads show a "deamination"
is almost certainly a SNP — the sample's genotype differs from the
reference at that position. These positions inflate hit counts and
confuse the nucleosome/TF caller.

This script does a single pileup pass over the BAM, computes the
per-position hit fraction (hits / total reads covering that position),
and outputs a BED file of positions exceeding a threshold (default
95%). The caller can then use `--snp-mask snps.bed` to exclude these
positions from the opp/hit arrays.

Works with both DAF (C→T / G→A / IUPAC Y/R) and Hia5 (m6A via
MM/ML) data. For DAF, a "hit" at a position means the query shows
T at ref-C (or A at ref-G, or Y/R). For Hia5, the hit is from the
MM/ML m6A call.

Usage:
    python snp_mask.py --in-bam amplicon.bam --out-bed snps.bed \
        --enzyme daf --min-fraction 0.95 --min-coverage 10

    python caller_v8.py --in-bam amplicon.bam --out-bam called.bam \
        --enzyme daf --snp-mask snps.bed
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict

import numpy as np
import pysam


def pileup_daf(bam_path: str, max_reads: int = 0,
                min_mapq: int = 20) -> dict:
    """Pileup DAF hits per reference position.

    Returns dict[chrom][(pos, ref_base)] → [n_opp, n_hit].
    Handles IUPAC Y/R and raw C→T/G→A encoding.
    """
    counts = defaultdict(lambda: [0, 0])  # (chrom, pos) → [opps, hits]
    bam = pysam.AlignmentFile(bam_path, 'rb', check_sq=False)
    n = 0
    for read in bam.fetch(until_eof=True):
        if read.is_unmapped or read.is_secondary or read.is_supplementary:
            continue
        if read.mapping_quality < min_mapq:
            continue
        n += 1
        if max_reads and n > max_reads:
            break
        q = read.query_sequence
        if q is None:
            continue
        chrom = read.reference_name
        try:
            pairs = read.get_aligned_pairs(with_seq=True, matches_only=True)
        except ValueError:
            continue
        for qp, rp, rb in pairs:
            if rb is None or qp is None:
                continue
            rbu = rb.upper()
            qb = q[qp].upper()
            if rbu == 'C':
                counts[(chrom, rp)][0] += 1
                if qb in ('T', 'Y'):
                    counts[(chrom, rp)][1] += 1
            elif rbu == 'G':
                counts[(chrom, rp)][0] += 1
                if qb in ('A', 'R'):
                    counts[(chrom, rp)][1] += 1
    bam.close()
    return counts, n


def pileup_hia5(bam_path: str, ml_threshold: int = 128,
                 max_reads: int = 0, min_mapq: int = 20) -> dict:
    """Pileup Hia5 m6A hits per reference position.

    Maps query-space m6A calls back to reference positions via
    aligned pairs, then counts hits per ref position.
    """
    counts = defaultdict(lambda: [0, 0])
    bam = pysam.AlignmentFile(bam_path, 'rb', check_sq=False)
    n = 0
    for read in bam.fetch(until_eof=True):
        if read.is_unmapped or read.is_secondary or read.is_supplementary:
            continue
        if read.mapping_quality < min_mapq:
            continue
        n += 1
        if max_reads and n > max_reads:
            break
        q = read.query_sequence
        if q is None:
            continue
        qlen = len(q)
        chrom = read.reference_name

        # Build qpos → rpos map
        qpos_to_rpos = {}
        for qp, rp in read.get_aligned_pairs(matches_only=True):
            if qp is not None and rp is not None:
                qpos_to_rpos[qp] = rp

        # Parse m6A from MM/ML
        hit_qpos = set()
        mm_str = ''
        try:
            mm_str = read.get_tag('MM') if read.has_tag('MM') else ''
        except KeyError:
            pass
        ml = None
        try:
            ml = list(read.get_tag('ML')) if read.has_tag('ML') else None
        except KeyError:
            pass
        if mm_str and ml is not None:
            ml_idx = 0
            for section in mm_str.rstrip(';').split(';'):
                if not section:
                    continue
                parts = section.split(',')
                header = parts[0]
                if len(header) < 3:
                    continue
                base = header[0]
                code = header[2]
                try:
                    skips = [int(x) for x in parts[1:] if x]
                except ValueError:
                    continue
                if code != 'a':
                    ml_idx += len(skips)
                    continue
                target = base.upper()
                qp_cursor = 0
                for skip in skips:
                    if ml_idx >= len(ml):
                        break
                    score = ml[ml_idx]
                    ml_idx += 1
                    needed = skip + 1
                    while qp_cursor < qlen and needed > 0:
                        if q[qp_cursor].upper() == target:
                            needed -= 1
                            if needed == 0:
                                break
                        qp_cursor += 1
                    if qp_cursor >= qlen:
                        break
                    if score >= ml_threshold:
                        hit_qpos.add(qp_cursor)
                    qp_cursor += 1

        # Map to ref positions: every A/T aligned position is an opp,
        # m6A positions are hits
        q_upper = q.upper()
        for qp, rp in qpos_to_rpos.items():
            if q_upper[qp] in ('A', 'T'):
                counts[(chrom, rp)][0] += 1
                if qp in hit_qpos:
                    counts[(chrom, rp)][1] += 1
    bam.close()
    return counts, n


def detect_snps(counts: dict, min_fraction: float = 0.95,
                 min_coverage: int = 10) -> list:
    """From pileup counts, find positions exceeding the hit fraction.

    Returns list of (chrom, pos, fraction, coverage) sorted by
    position.
    """
    snps = []
    for (chrom, pos), (opps, hits) in counts.items():
        if opps < min_coverage:
            continue
        frac = hits / opps
        if frac >= min_fraction:
            snps.append((chrom, pos, frac, opps))
    snps.sort()
    return snps


def write_bed(snps: list, out_path: str):
    """Write SNP positions as a BED file (0-based, 1-bp intervals)."""
    with open(out_path, 'w') as f:
        f.write('#chrom\tstart\tend\tfraction\tcoverage\n')
        for chrom, pos, frac, cov in snps:
            f.write(f'{chrom}\t{pos}\t{pos + 1}\t{frac:.4f}\t{cov}\n')


def load_snp_mask(bed_path: str) -> set:
    """Load a SNP mask BED file into a set of (chrom, pos) tuples.

    Used by the caller to exclude SNP positions from opp/hit arrays.
    """
    mask = set()
    with open(bed_path) as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.strip().split('\t')
            chrom = parts[0]
            start = int(parts[1])
            end = int(parts[2])
            for pos in range(start, end):
                mask.add((chrom, pos))
    return mask


def main():
    ap = argparse.ArgumentParser(
        description='Detect probable SNPs from amplicon DAF-seq / Hia5 data')
    ap.add_argument('--in-bam', required=True)
    ap.add_argument('--out-bed', required=True,
                    help='output BED file of SNP positions')
    ap.add_argument('--enzyme', required=True, choices=['daf', 'hia5'])
    ap.add_argument('--min-fraction', type=float, default=0.95,
                    help='minimum hit fraction to call a SNP (default 0.95)')
    ap.add_argument('--min-coverage', type=int, default=10,
                    help='minimum read coverage at position (default 10)')
    ap.add_argument('--max-reads', type=int, default=0,
                    help='cap on reads to process (0 = all)')
    ap.add_argument('--ml-threshold', type=int, default=128,
                    help='Hia5 only: m6A ML threshold')
    args = ap.parse_args()

    print(f'Piling up {args.in_bam} ({args.enzyme})...')
    if args.enzyme == 'daf':
        counts, n_reads = pileup_daf(args.in_bam, max_reads=args.max_reads)
    else:
        counts, n_reads = pileup_hia5(args.in_bam, ml_threshold=args.ml_threshold,
                                        max_reads=args.max_reads)

    print(f'  {n_reads} reads processed, {len(counts)} positions piled up')

    snps = detect_snps(counts, min_fraction=args.min_fraction,
                        min_coverage=args.min_coverage)
    print(f'  {len(snps)} positions flagged as probable SNPs '
          f'(hit fraction ≥ {args.min_fraction}, coverage ≥ {args.min_coverage})')

    if snps:
        fracs = [s[2] for s in snps]
        covs = [s[3] for s in snps]
        print(f'  fraction: mean={np.mean(fracs):.3f} '
              f'min={min(fracs):.3f} max={max(fracs):.3f}')
        print(f'  coverage: mean={np.mean(covs):.0f} '
              f'min={min(covs)} max={max(covs)}')

    write_bed(snps, args.out_bed)
    print(f'  Wrote {args.out_bed}')


if __name__ == '__main__':
    main()
