#!/usr/bin/env python3
"""v2 vs v3 TF concordance as a function of v3 tq threshold.

Reports:
  - Per-threshold counts of v2 and v3 TF calls (size 20-89 bp)
  - Fraction of v2 calls matched (IoU≥0.5) by a surviving v3 TF
  - Fraction of v3 calls matched by a v2 fp_v2 entry
  - IoU distribution for matched pairs
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pysam

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   '..', '..', 'caller_comparison', 'scripts'))
from parse_ma_calls import parse_ma, split_aq


def iou(a_s, a_e, b_s, b_e):
    inter = max(0, min(a_e, b_e) - max(a_s, b_s))
    union = max(a_e, b_e) - min(a_s, b_s)
    return inter / union if union > 0 else 0.0


def extract_tfs_with_tq(ma_str, aq_array):
    """Returns list of (s, l, tq) for tf+QQQ entries (first Q = tq)."""
    parsed = parse_ma(ma_str)
    out = []
    aq_list = list(aq_array) if aq_array is not None else []
    idx = 0
    for name, strand, qspec, intervals in parsed['raw']:
        n_q = len(qspec)
        for s, l in intervals:
            vals = aq_list[idx:idx + n_q]
            idx += n_q
            if name == 'tf' and n_q >= 1:
                out.append((s, l, int(vals[0])))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in-bam', action='append', required=True)
    ap.add_argument('--thresholds', type=int, nargs='+',
                    default=[0, 20, 40, 60, 80, 100, 150, 200])
    ap.add_argument('--tf-size-min', type=int, default=20)
    ap.add_argument('--tf-size-max', type=int, default=89)
    ap.add_argument('--match-iou', type=float, default=0.5)
    ap.add_argument('--tss-bed', default=None,
                    help='if provided, restrict analysis to reads '
                         'whose alignment spans a TSS in this BED')
    ap.add_argument('--region', default='gene_body',
                    choices=['gene_body', 'pause', 'post_tes', 'all'],
                    help='position restriction: gene_body = TSS to '
                         'TSS+gene-body-bp; pause = TSS+10..+50; '
                         'post_tes = TSS+gene-body-bp..+gene-body-bp+2kb; '
                         'all = no restriction (default: gene_body)')
    ap.add_argument('--gene-body-bp', type=int, default=3000,
                    help='gene-body length (default 3000)')
    args = ap.parse_args()

    # Load TSS BED if provided
    tss_table = {}
    if args.tss_bed:
        with open(args.tss_bed) as f:
            for line in f:
                if line.startswith('#') or not line.strip(): continue
                parts = line.rstrip('\n').split('\t')
                chrom, start, end, name, _, strand = parts[:6]
                if strand == '+': tss, tes = int(start), int(end)
                else: tss, tes = int(end) - 1, int(start)
                tss_table.setdefault(chrom, []).append((tss, strand, name))
        print(f'Loaded {sum(len(v) for v in tss_table.values())} TSSs')

    def ref_to_query(read, ref_pos):
        try:
            pairs = read.get_aligned_pairs(matches_only=True)
        except ValueError:
            return None
        for qp, rp in pairs:
            if rp == ref_pos: return qp
        return None

    def in_region(center_q, tss_q, strand):
        if args.region == 'all': return True
        if args.region == 'pause':
            if strand == '+':
                return 10 <= center_q - tss_q <= 50
            else:
                return 10 <= tss_q - center_q <= 50
        if args.region == 'gene_body':
            if strand == '+':
                return 0 <= center_q - tss_q <= args.gene_body_bp
            else:
                return 0 <= tss_q - center_q <= args.gene_body_bp
        if args.region == 'post_tes':
            if strand == '+':
                return (args.gene_body_bp <= center_q - tss_q
                        <= args.gene_body_bp + 2000)
            else:
                return (args.gene_body_bp <= tss_q - center_q
                        <= args.gene_body_bp + 2000)
        return True

    # Accumulate across reads/bams
    # For each threshold, collect (n_v3, n_v2, n_v2_matched, n_v3_matched)
    totals = {t: {'v3': 0, 'v2': 0, 'v3_matched': 0, 'v2_matched': 0}
              for t in args.thresholds}

    n_reads = 0
    for bam_path in args.in_bam:
        print(f'[{bam_path}]', flush=True)
        bam = pysam.AlignmentFile(bam_path, 'rb', check_sq=False)
        for r in bam.fetch(until_eof=True):
            if r.is_unmapped or r.is_secondary or r.is_supplementary:
                continue
            n_reads += 1
            if not r.has_tag('MA'): continue
            ma_str = r.get_tag('MA')
            aq = r.get_tag('AQ') if r.has_tag('AQ') else []
            parsed = parse_ma(ma_str)
            v2s = [(s, l) for s, l in parsed['fp_v2']
                   if args.tf_size_min <= l <= args.tf_size_max]
            tfs_with_q = [(s, l, tq) for s, l, tq in
                          extract_tfs_with_tq(ma_str, aq)
                          if args.tf_size_min <= l <= args.tf_size_max]

            # If TSS-restricted, filter footprints to Pol II region
            if tss_table:
                chrom = r.reference_name
                if chrom not in tss_table: continue
                tss_qs = []
                for tss_ref, strand, _ in tss_table[chrom]:
                    if r.reference_start <= tss_ref <= r.reference_end:
                        tq = ref_to_query(r, tss_ref)
                        if tq is not None:
                            tss_qs.append((tq, strand))
                if not tss_qs: continue

                def any_tss_match(s, l):
                    center = s + l // 2
                    for tss_q, strand in tss_qs:
                        if in_region(center, tss_q, strand):
                            return True
                    return False

                v2s = [(s, l) for s, l in v2s if any_tss_match(s, l)]
                tfs_with_q = [(s, l, tq) for s, l, tq in tfs_with_q
                              if any_tss_match(s, l)]

            for th in args.thresholds:
                v3_surviving = [(s, l) for s, l, tq in tfs_with_q if tq >= th]
                totals[th]['v3'] += len(v3_surviving)
                totals[th]['v2'] += len(v2s)
                # For each v2, does it match any surviving v3?
                for s2, l2 in v2s:
                    best = max((iou(s2, s2+l2, s3, s3+l3)
                                for s3, l3 in v3_surviving), default=0)
                    if best >= args.match_iou:
                        totals[th]['v2_matched'] += 1
                # For each v3 surviving, does it match any v2?
                for s3, l3 in v3_surviving:
                    best = max((iou(s3, s3+l3, s2, s2+l2)
                                for s2, l2 in v2s), default=0)
                    if best >= args.match_iou:
                        totals[th]['v3_matched'] += 1
        bam.close()

    print(f'\nProcessed {n_reads:,} reads\n')
    print(f'{"tq_min":>6s}  {"v3_calls":>10s}  {"v2_calls":>10s}  '
          f'{"v3→v2":>8s}  {"v2→v3":>8s}  {"v3_match_%":>10s}  {"v2_match_%":>10s}')
    for th in args.thresholds:
        r = totals[th]
        v3_match = 100 * r['v3_matched'] / max(1, r['v3'])
        v2_match = 100 * r['v2_matched'] / max(1, r['v2'])
        print(f'{th:>6d}  {r["v3"]:>10,}  {r["v2"]:>10,}  '
              f'{r["v3_matched"]:>8,}  {r["v2_matched"]:>8,}  '
              f'{v3_match:>9.1f}%  {v2_match:>9.1f}%')


if __name__ == '__main__':
    main()
