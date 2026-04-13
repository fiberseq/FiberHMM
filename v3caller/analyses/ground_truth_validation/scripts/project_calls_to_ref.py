#!/usr/bin/env python3
"""Project v2 / v3 calls from query coords → reference coords.

Output: a single BED6+ TSV with all calls from all input BAMs,
each classified by category (v2_only/v3_only/shared/etc.) and
annotated with call type (nuc/tf/fp_v2), size, and originating
read.

This is the feed for enrichment_at_peaks.py, which uses these
projected positions to look up ChIP-nexus / PRO-seq signal.
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict

import pysam

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   '..', '..', 'caller_comparison', 'scripts'))
from parse_ma_calls import parse_ma


def iou(a_s, a_e, b_s, b_e):
    inter = max(0, min(a_e, b_e) - max(a_s, b_s))
    union = max(a_e, b_e) - min(a_s, b_s)
    return inter / union if union > 0 else 0.0


def query_to_ref_center(read, q_center):
    """Find the reference coordinate for a given query position.
    Uses the aligned-pairs map, returns None if unaligned."""
    try:
        pairs = read.get_aligned_pairs(matches_only=True)
    except ValueError:
        return None
    best = None
    best_dist = 10**9
    for qp, rp in pairs:
        d = abs(qp - q_center)
        if d < best_dist:
            best_dist = d
            best = rp
            if d == 0:
                break
    if best is None or best_dist > 5:
        return None
    return best


def classify_v3_tf(v3_tf, v2s, match_iou=0.5, weak_iou=0.25):
    """Classify a v3 TF as shared/weak_shared/new relative to v2 fp."""
    s3, l3 = v3_tf
    e3 = s3 + l3
    best_u = 0.0
    best_v2 = None
    for s2, l2 in v2s:
        e2 = s2 + l2
        u = iou(s3, e3, s2, e2)
        if u > best_u:
            best_u = u
            best_v2 = (s2, l2)
    if best_u >= match_iou:
        return 'shared', best_u, best_v2
    if best_u >= weak_iou:
        return 'weak', best_u, best_v2
    return 'new', best_u, None


def classify_v2(v2_fp, v3_tfs, match_iou=0.5, weak_iou=0.25):
    """Classify v2 fp as shared-tf/unmatched relative to v3 tfs."""
    s2, l2 = v2_fp
    e2 = s2 + l2
    best_u = 0.0
    best_v3 = None
    for s3, l3 in v3_tfs:
        e3 = s3 + l3
        u = iou(s2, e2, s3, e3)
        if u > best_u:
            best_u = u
            best_v3 = (s3, l3)
    if best_u >= match_iou:
        return 'shared', best_u, best_v3
    if best_u >= weak_iou:
        return 'weak', best_u, best_v3
    return 'v2_only', best_u, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in-bam', action='append', required=True)
    ap.add_argument('--out-bed', required=True)
    ap.add_argument('--max-reads', type=int, default=0)
    ap.add_argument('--tf-size-min', type=int, default=20)
    ap.add_argument('--tf-size-max', type=int, default=90)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_bed) or '.', exist_ok=True)

    n_out = 0
    with open(args.out_bed, 'w') as out:
        out.write('#chrom\tstart\tend\tcall_id\tsize\tstrand\t'
                   'category\tcall_type\tread\tiou_partner\n')
        n_reads = 0
        for bam_path in args.in_bam:
            print(f'[{bam_path}]', flush=True)
            bam = pysam.AlignmentFile(bam_path, 'rb', check_sq=False)
            for r in bam.fetch(until_eof=True):
                if r.is_unmapped or r.is_secondary or r.is_supplementary:
                    continue
                n_reads += 1
                if args.max_reads and n_reads > args.max_reads:
                    break
                if not r.has_tag('MA'):
                    continue
                ma = parse_ma(r.get_tag('MA'))

                # v2 footprints — consider all sizes (v2 didn't split)
                v2s = ma['fp_v2']
                # v3 TF-sized footprints from both nuc+ and tf+. Focus on
                # TF-size range (the reviewer's interest: 20-90 bp).
                v3_tf_all = [(s, l) for s, l in ma['tf']
                              if args.tf_size_min <= l <= args.tf_size_max]
                # Also include any nuc+ entries within TF size range
                # (shouldn't happen since min_footprint=80, but guard)
                v3_tf_all += [(s, l) for s, l in ma['nuc']
                               if args.tf_size_min <= l <= args.tf_size_max]

                chrom = r.reference_name

                # Project and emit v3 TFs
                for s, l in v3_tf_all:
                    q_center = s + l // 2
                    ref_center = query_to_ref_center(r, q_center)
                    if ref_center is None:
                        continue
                    cat, u, partner = classify_v3_tf((s, l), v2s)
                    # Map "new" → "v3_only" for downstream analysis
                    if cat == 'new':
                        cat = 'v3_only'
                    # Bed 3 half-open, with strand '+'
                    out.write(f'{chrom}\t{ref_center - l//2}\t{ref_center + l//2}\t'
                               f'tf_{n_out}\t{l}\t+\t'
                               f'{cat}\tv3_tf\t{r.query_name}\t{u:.3f}\n')
                    n_out += 1

                # Project and emit v2 fp that are TF-sized
                v3_tf_for_v2 = [(s, l) for s, l in ma['tf']]
                for s, l in v2s:
                    if not (args.tf_size_min <= l <= args.tf_size_max):
                        continue
                    q_center = s + l // 2
                    ref_center = query_to_ref_center(r, q_center)
                    if ref_center is None:
                        continue
                    cat, u, partner = classify_v2((s, l), v3_tf_for_v2)
                    out.write(f'{chrom}\t{ref_center - l//2}\t{ref_center + l//2}\t'
                               f'v2_{n_out}\t{l}\t+\t'
                               f'{cat}\tv2_tf\t{r.query_name}\t{u:.3f}\n')
                    n_out += 1
            bam.close()
    print(f'\nWrote {n_out:,} projected calls across {n_reads:,} reads')
    print(f'Output: {args.out_bed}')


if __name__ == '__main__':
    main()
