#!/usr/bin/env python3
"""Diagnostic plot: show the smoothed signal the trough detector uses,
with every detected trough marked and numbered.

This is the companion to measure_penetration.py — it doesn't compute
penetration, it just visualizes the trough-detection step so we can
vet which troughs are real nucleosomes and which are artifacts.

For each amplicon:
  - smoothed hit density (the signal fed to find_peaks)
  - every detected trough marked with its index and position
  - candidate regions labeled

Usage (same --in-bam list as measure_penetration.py):
  python plot_detector_diagnostic.py --in-bam napa.bam --label NAPA \
      --out-prefix figures/detector_diag \
      --enzyme daf --min-distance 160 --prominence 0.15 \
      --min-width 80 --max-width 200 --smooth-window 15
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict

import numpy as np
import pysam
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Import the same detection functions used by measure_penetration
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from measure_penetration import (
    pileup_hit_density, build_density_array, smooth, find_troughs,
)


def pick_best_chrom_and_window(opp_counts):
    """Pick chrom with the most total opps and trim to the amplicon window."""
    if not opp_counts:
        return None, 0, None, None
    total_opp = {c: sum(d.values()) for c, d in opp_counts.items()}
    best_chrom = max(total_opp, key=total_opp.get)
    positions = sorted(opp_counts[best_chrom].keys())
    span_start, span_end = positions[0], positions[-1] + 1
    return best_chrom, span_start, span_end, None


def auto_amplicon_window(opp_arr, pad=100):
    """Trim the array to the high-coverage amplicon window (≥20% of max)."""
    L = len(opp_arr)
    if L < 200:
        return 0, L
    opp_sm = np.convolve(opp_arr.astype(float),
                          np.ones(100) / 100, mode='same')
    peak_idx = int(np.argmax(opp_sm))
    peak_cov = opp_sm[peak_idx]
    thresh = 0.2 * peak_cov
    lo = peak_idx
    while lo > 0 and opp_sm[lo - 1] >= thresh:
        lo -= 1
    hi = peak_idx
    while hi < L - 1 and opp_sm[hi + 1] >= thresh:
        hi += 1
    return max(0, lo - pad), min(L, hi + pad)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in-bam', action='append', required=True)
    ap.add_argument('--label', action='append', default=None)
    ap.add_argument('--out-prefix', required=True)
    ap.add_argument('--enzyme', default='daf', choices=['daf', 'hia5'])
    ap.add_argument('--smooth-window', type=int, default=15)
    ap.add_argument('--min-distance', type=int, default=160)
    ap.add_argument('--prominence', type=float, default=0.15)
    ap.add_argument('--min-width', type=int, default=80)
    ap.add_argument('--max-width', type=int, default=200)
    ap.add_argument('--recenter-radius', type=int, default=30)
    ap.add_argument('--max-reads', type=int, default=5000)
    args = ap.parse_args()

    labels = args.label or [os.path.basename(b).replace('.bam', '')
                              for b in args.in_bam]
    if len(labels) != len(args.in_bam):
        labels = [os.path.basename(b).replace('.bam', '') for b in args.in_bam]

    datasets = []
    for bam, lbl in zip(args.in_bam, labels):
        if not os.path.exists(bam):
            continue
        print(f'[{lbl}] piling {bam}...')
        opp_counts, hit_counts, n_reads = pileup_hit_density(
            bam, enzyme=args.enzyme, max_reads=args.max_reads)

        best_chrom, span_start, span_end, _ = pick_best_chrom_and_window(
            opp_counts)
        if best_chrom is None:
            continue
        opp_arr, hit_arr = build_density_array(
            opp_counts, hit_counts, best_chrom, span_start, span_end)

        # Trim to amplicon window
        lo, hi = auto_amplicon_window(opp_arr)
        opp_arr = opp_arr[lo:hi]
        hit_arr = hit_arr[lo:hi]
        amp_start = span_start + lo

        density = smooth(hit_arr.astype(float), args.smooth_window)
        rate = np.zeros_like(opp_arr, dtype=float)
        mask = opp_arr > 0
        rate[mask] = hit_arr[mask] / opp_arr[mask]
        rate_sm = smooth(rate, args.smooth_window)

        troughs = find_troughs(
            density, min_distance=args.min_distance,
            prominence_frac=args.prominence,
            min_width=args.min_width, max_width=args.max_width,
        )
        # Snap to argmin within ±recenter_radius (same logic as
        # aggregate_profile)
        snapped = []
        for t in troughs:
            rlo = max(0, t - args.recenter_radius)
            rhi = min(len(density), t + args.recenter_radius + 1)
            snapped.append(rlo + int(np.argmin(density[rlo:rhi])))
        snapped = np.array(snapped, dtype=int)

        datasets.append((lbl, best_chrom, amp_start, opp_arr, hit_arr,
                          density, rate_sm, troughs, snapped, n_reads))
        print(f'  → {best_chrom}:{amp_start:,}-{amp_start+len(opp_arr):,} '
              f'({len(opp_arr)} bp), {len(troughs)} troughs')

    if not datasets:
        print('no data'); return

    n_ds = len(datasets)
    fig, axes = plt.subplots(n_ds, 1, figsize=(16, 3.2 * n_ds), sharex=False)
    if n_ds == 1:
        axes = [axes]

    for i, (lbl, chrom, start, opp_arr, hit_arr, density, rate_sm,
              troughs, snapped, n_reads) in enumerate(datasets):
        xs = np.arange(len(density)) + start
        ax = axes[i]

        # Smoothed hit density on left y-axis
        ax.plot(xs, density, color='#1e3a8a', linewidth=1.0,
                label='smoothed hit count')
        ax.fill_between(xs, 0, density, color='#1e3a8a', alpha=0.2)
        ax.set_ylabel('hit count (smoothed)', color='#1e3a8a', fontsize=9)
        ax.tick_params(axis='y', labelcolor='#1e3a8a')

        # Smoothed rate on right y-axis
        ax2 = ax.twinx()
        ax2.plot(xs, rate_sm, color='#dc2626', linewidth=0.7, alpha=0.7,
                  label='smoothed rate (hit/opp)')
        ax2.set_ylabel('hit / opp', color='#dc2626', fontsize=9)
        ax2.tick_params(axis='y', labelcolor='#dc2626')
        ax2.set_ylim(0, max(0.05, rate_sm.max() * 1.1))

        # Mark each snapped trough
        for j, t in enumerate(snapped):
            ax.axvline(xs[t], color='#16a34a', linestyle='--',
                        alpha=0.5, linewidth=0.8)
            ax.annotate(f'{j}\n{xs[t]:,}', xy=(xs[t], density[t]),
                         xytext=(xs[t], density[t] * 0.92),
                         ha='center', fontsize=7, color='#15803d')

        ax.set_title(f'{lbl}  [{chrom}:{start:,}-{start+len(density):,}]  '
                     f'({n_reads} reads, {len(snapped)} troughs detected)',
                     fontsize=10)
        ax.grid(alpha=0.3)
        if i == n_ds - 1:
            ax.set_xlabel('ref position (bp)')

    fig.tight_layout()
    png = args.out_prefix + '.png'
    fig.savefig(png, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'Wrote {png}')

    # Also save a TSV of (label, chrom, position) for every trough
    tsv = args.out_prefix + '_troughs.tsv'
    with open(tsv, 'w') as f:
        f.write('label\tchrom\tposition\ttrough_index\thit_density\n')
        for lbl, chrom, start, _, _, density, _, _, snapped, _ in datasets:
            for j, t in enumerate(snapped):
                f.write(f'{lbl}\t{chrom}\t{start + t}\t{j}\t{density[t]:.2f}\n')
    print(f'Wrote {tsv}')


if __name__ == '__main__':
    main()
