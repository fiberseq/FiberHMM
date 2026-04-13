#!/usr/bin/env python3
"""For each v2 fp_v2 call in the Pol II size range (35-65 bp),
ask: does v3 have ANY atom (nuc or tf, any tq) at that position?

Answer categories per v2 call:
  - 'v3_present_high_tq' : v3 has an overlapping call (IoU > 0.1)
    AND its tq ≥ 80 (high-confidence v3 agreement)
  - 'v3_present_mid_tq'  : overlapping v3 call, 40 ≤ tq < 80
  - 'v3_present_low_tq'  : overlapping v3 call, tq < 40
  - 'v3_present_nuc'     : overlapping v3 NUC (no tq)
  - 'absent'             : no v3 atom overlaps at any IoU

If most v2 calls fall in 'present' categories at any tq, we can
recover concordance via tq-calibration. If many fall in 'absent',
v3's Pass-1 is structurally missing those positions.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pysam

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   '..', '..', 'caller_comparison', 'scripts'))
from parse_ma_calls import parse_ma


def iou(a_s, a_e, b_s, b_e):
    inter = max(0, min(a_e, b_e) - max(a_s, b_s))
    union = max(a_e, b_e) - min(a_s, b_s)
    return inter / union if union > 0 else 0.0


def extract_v3_with_tq(ma_str, aq_array):
    """Returns list of (s, l, tq, kind) where kind='tf' or 'nuc'.
    Nucs get tq=None."""
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
                out.append((s, l, int(vals[0]), 'tf'))
            elif name == 'nuc':
                out.append((s, l, None, 'nuc'))
    return out


def classify_v2_call(v2_call, v3_list, iou_thresh=0.1):
    """Given a v2 call and list of v3 (s, l, tq, kind), return
    (category, best_iou, best_tq, best_kind)."""
    s2, l2 = v2_call
    best_iou = 0.0
    best_tq = None
    best_kind = None
    for s3, l3, tq, kind in v3_list:
        u = iou(s2, s2 + l2, s3, s3 + l3)
        if u > best_iou:
            best_iou = u
            best_tq = tq
            best_kind = kind
    if best_iou < iou_thresh:
        return 'absent', best_iou, best_tq, best_kind
    if best_kind == 'nuc':
        return 'v3_present_nuc', best_iou, best_tq, best_kind
    # TF with tq
    if best_tq is None:
        return 'v3_present_low_tq', best_iou, best_tq, best_kind
    if best_tq >= 80:
        return 'v3_present_high_tq', best_iou, best_tq, best_kind
    if best_tq >= 40:
        return 'v3_present_mid_tq', best_iou, best_tq, best_kind
    return 'v3_present_low_tq', best_iou, best_tq, best_kind


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in-bam', action='append', required=True)
    ap.add_argument('--size-min', type=int, default=35)
    ap.add_argument('--size-max', type=int, default=65)
    ap.add_argument('--iou-thresh', type=float, default=0.1,
                    help='min IoU to consider overlap (default 0.1 — '
                         'very loose, asks only "is there any atom '
                         'there at all?")')
    args = ap.parse_args()

    from collections import Counter
    counts = Counter()
    iou_dist = []
    tq_of_matches = []
    kind_of_matches = Counter()

    n_reads = 0
    n_v2_total = 0
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
                   if args.size_min <= l <= args.size_max]
            v3 = extract_v3_with_tq(ma_str, aq)

            for v2 in v2s:
                n_v2_total += 1
                cat, u, tq, kind = classify_v2_call(v2, v3,
                                                       iou_thresh=args.iou_thresh)
                counts[cat] += 1
                if u > 0: iou_dist.append(u)
                if cat.startswith('v3_present') and tq is not None:
                    tq_of_matches.append(tq)
                if cat.startswith('v3_present'):
                    kind_of_matches[kind] += 1
        bam.close()

    print(f'\nProcessed {n_reads:,} reads, {n_v2_total:,} v2 calls '
          f'in size range [{args.size_min}, {args.size_max}] bp')
    print(f'\n=== Classification of v2 calls (IoU threshold {args.iou_thresh}) ===')
    for cat in ['v3_present_high_tq', 'v3_present_mid_tq',
                 'v3_present_low_tq', 'v3_present_nuc', 'absent']:
        n = counts.get(cat, 0)
        pct = 100 * n / max(1, n_v2_total)
        print(f'  {cat:<22s}: {n:>7,}  ({pct:5.1f}%)')

    n_present = sum(counts[c] for c in counts if c.startswith('v3_present'))
    print(f'\n  ANY v3 atom present:        {n_present:>7,}  '
          f'({100*n_present/max(1,n_v2_total):5.1f}%)')
    print(f'  v3 absent:                   {counts["absent"]:>7,}  '
          f'({100*counts["absent"]/max(1,n_v2_total):5.1f}%)')

    # How often is the best v3 match a nuc vs TF?
    print(f'\n=== Best v3 match kind (among present) ===')
    for kind, n in kind_of_matches.items():
        pct = 100 * n / max(1, n_present)
        print(f'  {kind}: {n:,} ({pct:.1f}%)')

    if tq_of_matches:
        tqa = np.array(tq_of_matches)
        print(f'\n=== tq distribution of best-matching v3 TF ===')
        print(f'  n with tq: {len(tqa):,}')
        print(f'  p10={np.percentile(tqa,10):.0f}  p25={np.percentile(tqa,25):.0f}  '
              f'p50={np.percentile(tqa,50):.0f}  p75={np.percentile(tqa,75):.0f}  '
              f'p90={np.percentile(tqa,90):.0f}')
        for th in [0, 20, 40, 60, 80, 100, 150]:
            n_above = (tqa >= th).sum()
            print(f'  ≥{th}: {n_above:,} ({100*n_above/len(tqa):.1f}%)')


if __name__ == '__main__':
    main()
