#!/usr/bin/env python3
"""Empirical characterization of merge-candidate gaps.

For each read in the (NOMERGE) v3 BAM, scan adjacent footprint pairs
and compute:
  - gap_len (bp between end of prev fp and start of next fp)
  - gap_hits (observed deamination hits in gap)
  - gap_opps (C/G opportunity count in gap)
  - merged_len (what the fused atom's length would be = next_end - prev_start)
  - prev_len, next_len (input atom sizes)

Plot:
  1. Gap length histogram (how short are inter-atom gaps?)
  2. Hit count conditional on short gap (≤20 bp): what's typical?
  3. Merged-call size distribution: for every adjacent pair, what would
     the fused length be?
  4. 2D: merged_len × gap_hits — which merges would be allowed under
     various thresholds?
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import Counter

import numpy as np
import pysam
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from parse_ma_calls import parse_ma


def deamination_qpos(read, enzyme='daf'):
    """Return sorted list of query positions with hits."""
    q = read.query_sequence
    if q is None: return []
    hits = []
    try:
        pairs = read.get_aligned_pairs(with_seq=True, matches_only=True)
    except ValueError:
        return []
    for qp, rp, rb in pairs:
        if rb is None or qp is None:
            continue
        rbu = rb.upper()
        qb = q[qp].upper()
        if enzyme == 'daf':
            if rbu == 'C' and qb in ('T', 'Y'):
                hits.append(qp)
            elif rbu == 'G' and qb in ('A', 'R'):
                hits.append(qp)
    return sorted(hits)


def opp_qpos(read, enzyme='daf'):
    """Return sorted list of query positions where a C or G is on ref."""
    try:
        pairs = read.get_aligned_pairs(with_seq=True, matches_only=True)
    except ValueError:
        return []
    opps = []
    for qp, rp, rb in pairs:
        if rb is None or qp is None: continue
        rbu = rb.upper()
        if enzyme == 'daf' and rbu in ('C', 'G'):
            opps.append(qp)
    return sorted(opps)


def count_in_range(sorted_list, lo, hi):
    from bisect import bisect_left, bisect_right
    return bisect_right(sorted_list, hi) - bisect_left(sorted_list, lo)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in-bam', action='append', required=True)
    ap.add_argument('--label', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--max-reads', type=int, default=0)
    ap.add_argument('--enzyme', default='daf', choices=['daf', 'hia5'])
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    gap_lens = []
    gap_hits = []
    gap_opps_list = []
    merged_lens = []
    prev_lens = []
    next_lens = []

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

            # Combined v3 footprints (nuc + tf), sorted by start
            combined = sorted(ma['nuc'] + ma['tf'])
            if len(combined) < 2:
                continue

            # Use pre-computed hits/opps for this read only (cached)
            hits_cache = None
            opps_cache = None

            for i in range(len(combined) - 1):
                s1, l1 = combined[i]
                s2, l2 = combined[i + 1]
                e1 = s1 + l1
                gap_len = s2 - e1
                if gap_len < 0:
                    continue  # overlap, skip
                if gap_len > 100:
                    continue  # focus on plausible merge candidates
                merged_len = (s2 + l2) - s1

                if hits_cache is None:
                    hits_cache = deamination_qpos(r, enzyme=args.enzyme)
                    opps_cache = opp_qpos(r, enzyme=args.enzyme)

                gh = count_in_range(hits_cache, e1, s2 - 1)
                go = count_in_range(opps_cache, e1, s2 - 1)

                gap_lens.append(gap_len)
                gap_hits.append(gh)
                gap_opps_list.append(go)
                merged_lens.append(merged_len)
                prev_lens.append(l1)
                next_lens.append(l2)
        bam.close()

    gap_lens = np.array(gap_lens)
    gap_hits = np.array(gap_hits)
    gap_opps_list = np.array(gap_opps_list)
    merged_lens = np.array(merged_lens)
    prev_lens = np.array(prev_lens)
    next_lens = np.array(next_lens)

    print(f'\nAnalyzed {len(gap_lens):,} adjacent footprint pairs across '
          f'{n_reads:,} reads')

    # Summary
    print(f'\n=== Gap length distribution ===')
    for lo, hi in [(0, 1), (1, 4), (4, 8), (8, 15), (15, 30), (30, 60), (60, 100)]:
        mask = (gap_lens >= lo) & (gap_lens < hi)
        n = int(mask.sum())
        if n == 0: continue
        median_hits = int(np.median(gap_hits[mask]))
        mean_hits = float(np.mean(gap_hits[mask]))
        median_merged = int(np.median(merged_lens[mask]))
        print(f'  gap_len [{lo}-{hi}):  n={n:>9,}  '
              f'hits median={median_hits}, mean={mean_hits:.2f}  '
              f'merged_len median={median_merged} bp')

    # Figure 1: gap_len + hit_count histograms
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    ax.hist(gap_lens, bins=np.arange(0, 101, 2), color='#1e3a8a', alpha=0.7)
    ax.set_xlabel('gap length (bp)')
    ax.set_ylabel('count')
    ax.set_title(f'{args.label}: inter-atom gap length\n(n={len(gap_lens):,} pairs)')
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    short_mask = gap_lens <= 20
    if short_mask.sum() > 0:
        bins_h = np.arange(0, max(gap_hits[short_mask].max() + 1, 10) + 1)
        ax.hist(gap_hits[short_mask], bins=bins_h,
                color='#dc2626', alpha=0.7)
        ax.set_xlabel('hits in gap')
        ax.set_ylabel('count')
        ax.set_title(f'Hits in SHORT gaps (≤20 bp)\n'
                     f'n={short_mask.sum():,}  '
                     f'{(gap_hits[short_mask] == 0).mean() * 100:.1f}% zero-hit  '
                     f'median={int(np.median(gap_hits[short_mask]))}')
        ax.grid(alpha=0.3)

    ax = axes[1, 0]
    ax.hist(merged_lens, bins=np.arange(0, 501, 10),
            color='#16a34a', alpha=0.7)
    ax.axvline(147, color='red', linestyle='--', label='mono-nuc (147)')
    ax.axvline(300, color='orange', linestyle='--', label='di-nuc (300)')
    ax.set_xlabel('would-be merged length (bp)')
    ax.set_ylabel('count')
    ax.set_title(f'Fused length if we merged every adjacent pair\n'
                 f'(n={len(merged_lens):,})')
    ax.legend(); ax.grid(alpha=0.3)
    ax.set_xlim(0, 500)

    # 2D hist: merged_len × gap_hits for short gaps
    ax = axes[1, 1]
    sel = (gap_lens <= 20)
    if sel.sum() > 0:
        h = ax.hist2d(merged_lens[sel], gap_hits[sel],
                      bins=[np.arange(0, 501, 10), np.arange(0, 11, 1)],
                      cmap='viridis',
                      norm=matplotlib.colors.LogNorm())
        plt.colorbar(h[3], ax=ax, label='count (log)')
        ax.axvline(147, color='red', linestyle='--')
        ax.axvline(300, color='orange', linestyle='--')
        ax.set_xlabel('merged length (bp)')
        ax.set_ylabel('hits in gap')
        ax.set_title(f'Short gaps: merged_len vs hit count\n'
                     f'(n={int(sel.sum()):,}) — red=147, orange=300')

    fig.tight_layout()
    png = os.path.join(args.out_dir, f'{args.label}_merge_candidates.png')
    fig.savefig(png, dpi=130, bbox_inches='tight'); plt.close(fig)
    print(f'\nWrote {png}')

    # Summary JSON
    import json
    summary = {
        'label': args.label,
        'n_pairs': int(len(gap_lens)),
        'n_reads': int(n_reads),
        'short_gaps_le_8bp': {
            'count': int((gap_lens <= 8).sum()),
            'median_hits': float(np.median(gap_hits[gap_lens <= 8])) if (gap_lens <= 8).sum() else 0,
            'pct_0_hits': float((gap_hits[gap_lens <= 8] == 0).mean() * 100) if (gap_lens <= 8).sum() else 0,
            'pct_1_hit': float((gap_hits[gap_lens <= 8] == 1).mean() * 100) if (gap_lens <= 8).sum() else 0,
            'pct_2plus_hits': float((gap_hits[gap_lens <= 8] >= 2).mean() * 100) if (gap_lens <= 8).sum() else 0,
            'median_merged_len': int(np.median(merged_lens[gap_lens <= 8])) if (gap_lens <= 8).sum() else 0,
            'pct_merged_in_nuc_range_120_180': float(
                ((merged_lens[gap_lens <= 8] >= 120) &
                 (merged_lens[gap_lens <= 8] <= 180)).mean() * 100)
                if (gap_lens <= 8).sum() else 0,
        },
    }
    jsn = os.path.join(args.out_dir, f'{args.label}_merge_candidates.json')
    with open(jsn, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'Wrote {jsn}')


if __name__ == '__main__':
    main()
