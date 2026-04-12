#!/usr/bin/env python3
"""Aggregate bulk hit density at HAND-CURATED anchor positions.

Companion to measure_penetration.py. Instead of automated trough
detection (which fires on both nucleosomes and TF footprints and
dilutes the signal), this script takes explicit (chrom, position)
anchors and computes the average profile around only those points.

The aggregator works in both HIT-COUNT and RATE (hit/opp) space.
Rate space is what you want for cross-position comparison because
it's invariant to per-position opp count variability.

Usage:
  python aggregate_at_anchors.py \
      --in-bam my.bam \
      --label my_amplicon \
      --anchor chrX:12345 --anchor chrX:12500 \
      --out-prefix figures/my_anchors \
      --profile-window 150 \
      --enzyme daf
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
import pysam
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from measure_penetration import (
    pileup_hit_density, build_density_array, smooth,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in-bam', required=True)
    ap.add_argument('--label', required=True)
    ap.add_argument('--anchor', action='append', required=True,
                    help='anchor position as chrom:pos (repeatable)')
    ap.add_argument('--out-prefix', required=True)
    ap.add_argument('--enzyme', default='daf', choices=['daf', 'hia5'])
    ap.add_argument('--smooth-window', type=int, default=15)
    ap.add_argument('--profile-window', type=int, default=150)
    ap.add_argument('--recenter-radius', type=int, default=20,
                    help='snap each anchor to argmin of smoothed signal '
                         'within ±radius (bp); 0 disables')
    ap.add_argument('--max-reads', type=int, default=0)
    ap.add_argument('--fp-rate', type=float, default=0.0,
                    help='flat per-context FP rate to subtract from '
                         'the rate-space profile. 0.008 = 0.8% (approx '
                         'DddA PacBio C→T FP).')
    args = ap.parse_args()

    # Parse anchors
    anchors = []
    for a in args.anchor:
        try:
            chrom, pos = a.split(':')
            anchors.append((chrom, int(pos)))
        except ValueError:
            print(f'bad anchor: {a}'); return
    print(f'[{args.label}] {len(anchors)} anchors: {anchors}')

    print(f'Piling up {args.in_bam}...')
    opp_counts, hit_counts, n_reads = pileup_hit_density(
        args.in_bam, enzyme=args.enzyme, max_reads=args.max_reads)
    print(f'  {n_reads} reads')

    W = args.profile_window
    count_profiles = []
    rate_profiles = []
    opp_profiles = []
    snapped_anchors = []

    for chrom, pos in anchors:
        if chrom not in opp_counts:
            print(f'  {chrom}:{pos} not in data'); continue
        # Build a dense array for ±(W + recenter) bp around the anchor
        R = args.recenter_radius
        lo = pos - W - R
        hi = pos + W + R + 1
        opp_arr, hit_arr = build_density_array(
            opp_counts, hit_counts, chrom, lo, hi)
        if opp_arr.sum() == 0:
            print(f'  {chrom}:{pos} no data in window'); continue

        density_sm = smooth(hit_arr.astype(float), args.smooth_window)
        rate = np.zeros_like(opp_arr, dtype=float)
        mask = opp_arr > 0
        rate[mask] = hit_arr[mask] / opp_arr[mask]
        rate_sm = smooth(rate, args.smooth_window)

        # Snap to argmin of smoothed density near the nominal anchor
        if R > 0:
            center = W + R  # index of anchor in the array
            rlo = max(0, center - R)
            rhi = min(len(density_sm), center + R + 1)
            snap_idx = rlo + int(np.argmin(density_sm[rlo:rhi]))
        else:
            snap_idx = W + R

        # Extract ±W around the snapped anchor
        left = snap_idx - W
        right = snap_idx + W + 1
        if left < 0 or right > len(density_sm):
            print(f'  {chrom}:{pos} window out of range'); continue
        count_profiles.append(density_sm[left:right])
        rate_profiles.append(rate_sm[left:right])
        opp_profiles.append(opp_arr[left:right])
        snapped_pos = lo + snap_idx
        snapped_anchors.append((chrom, snapped_pos, pos))
        print(f'  {chrom}:{pos} → snapped to {snapped_pos} '
              f'(Δ {snapped_pos - pos:+d} bp)')

    if not count_profiles:
        print('no profiles'); return

    count_arr = np.array(count_profiles)
    rate_arr = np.array(rate_profiles)
    opp_arr_all = np.array(opp_profiles)

    # Equal-weight average (each anchor is one nucleosome)
    count_mean = count_arr.mean(axis=0)
    opp_mean = opp_arr_all.mean(axis=0)

    # COVERAGE-WEIGHTED rate: Σhits / Σopps in a rolling window.
    # This is the correct way to aggregate rate across sparse-opp
    # positions: smoothing a per-position rate that's 0 at no-opp
    # positions artificially depresses the linker shoulders. Instead,
    # we pool hits and opps across all anchors (already done per
    # position — hit_arr/opp_arr are already aligned, just need the
    # raw un-smoothed hit/opp), then do a rolling sum and divide.
    #
    # Rebuild un-smoothed hit+opp profiles per anchor (same window as
    # rate_profiles). We already captured opp_profiles; need raw hit.
    raw_hit_profiles = []
    for chrom, pos in anchors:
        if chrom not in opp_counts:
            continue
        lo = pos - W - args.recenter_radius
        hi = pos + W + args.recenter_radius + 1
        opp_win, hit_win = build_density_array(
            opp_counts, hit_counts, chrom, lo, hi)
        if opp_win.sum() == 0:
            continue
        density_sm = smooth(hit_win.astype(float), args.smooth_window)
        R = args.recenter_radius
        center_i = W + R
        if R > 0:
            rlo = max(0, center_i - R)
            rhi = min(len(density_sm), center_i + R + 1)
            snap_i = rlo + int(np.argmin(density_sm[rlo:rhi]))
        else:
            snap_i = center_i
        left = snap_i - W
        right = snap_i + W + 1
        if left < 0 or right > len(density_sm):
            continue
        raw_hit_profiles.append(hit_win[left:right])

    hit_pool = np.array(raw_hit_profiles).sum(axis=0).astype(float)
    opp_pool = opp_arr_all.sum(axis=0).astype(float)
    # Rolling sum with window = smooth_window
    kernel = np.ones(args.smooth_window)
    hit_roll = np.convolve(hit_pool, kernel, mode='same')
    opp_roll = np.convolve(opp_pool, kernel, mode='same')
    rate_mean = np.divide(hit_roll, opp_roll,
                           out=np.zeros_like(hit_roll),
                           where=opp_roll > 0)

    # FP-subtract in rate space
    rate_mean_fp = np.maximum(0.0, rate_mean - args.fp_rate)

    d_arr = np.arange(-W, W + 1)

    # Penetration in rate space (with and without FP subtraction)
    center_idx = W
    def min_in(prof, width=10):
        return float(prof[center_idx - width:center_idx + width + 1].min())
    dyad_r_raw = min_in(rate_mean)
    dyad_r_fp = min_in(rate_mean_fp)
    linker_r_raw = float(rate_mean.max())
    linker_r_fp = float(rate_mean_fp.max())
    pen_raw = dyad_r_raw / linker_r_raw if linker_r_raw > 0 else float('nan')
    pen_fp = dyad_r_fp / linker_r_fp if linker_r_fp > 0 else float('nan')

    # Penetration in count space (for reference)
    dyad_c = min_in(count_mean)
    linker_c = float(count_mean.max())
    pen_c = dyad_c / linker_c if linker_c > 0 else float('nan')

    print(f'\n=== {args.label}:   {len(count_profiles)} anchors ===')
    print(f'  COUNT space:  dyad={dyad_c:.1f}  linker={linker_c:.1f}  pen={pen_c:.3f}')
    print(f'  RATE  raw :   dyad={dyad_r_raw:.4f}  linker={linker_r_raw:.4f}  pen={pen_raw:.3f}')
    if args.fp_rate > 0:
        print(f'  RATE  -FP :   dyad={dyad_r_fp:.4f}  linker={linker_r_fp:.4f}  pen={pen_fp:.3f}  (FP={args.fp_rate:.4f})')

    # Plot: count (per anchor + mean), rate (per anchor + mean + FP-subtracted)
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.5))

    ax = axes[0]
    for p in count_profiles:
        ax.plot(d_arr, p, color='#94a3b8', alpha=0.5, linewidth=0.8)
    ax.plot(d_arr, count_mean, color='#1e3a8a', linewidth=2.0,
            label=f'mean (n={len(count_profiles)})')
    ax.axvline(0, color='#dc2626', linestyle='--', alpha=0.5, label='dyad')
    ax.axvline(-73, color='#94a3b8', linestyle=':', alpha=0.4)
    ax.axvline(73, color='#94a3b8', linestyle=':', alpha=0.4)
    ax.set_title(f'{args.label}  HIT COUNT space\npen (raw) = {pen_c:.3f}',
                  fontsize=10)
    ax.set_xlabel('distance from dyad (bp)')
    ax.set_ylabel('hit density (smoothed count)')
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    ax = axes[1]
    for p in rate_profiles:
        ax.plot(d_arr, p, color='#94a3b8', alpha=0.5, linewidth=0.8)
    ax.plot(d_arr, rate_mean, color='#16a34a', linewidth=2.0,
            label=f'mean (n={len(rate_profiles)})')
    ax.axvline(0, color='#dc2626', linestyle='--', alpha=0.5)
    ax.axvline(-73, color='#94a3b8', linestyle=':', alpha=0.4)
    ax.axvline(73, color='#94a3b8', linestyle=':', alpha=0.4)
    if args.fp_rate > 0:
        ax.axhline(args.fp_rate, color='#f59e0b', linestyle=':', alpha=0.6,
                   label=f'FP rate ({args.fp_rate:.3f})')
    ax.set_title(f'{args.label}  RATE space (hit/opp)\npen (raw) = {pen_raw:.3f}',
                  fontsize=10)
    ax.set_xlabel('distance from dyad (bp)')
    ax.set_ylabel('hit / opp')
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    ax = axes[2]
    ax.plot(d_arr, rate_mean_fp, color='#7c3aed', linewidth=2.0,
            label=f'FP-subtracted')
    ax.plot(d_arr, rate_mean, color='#16a34a', linewidth=1.0,
            alpha=0.5, linestyle='--', label='raw rate')
    ax.axvline(0, color='#dc2626', linestyle='--', alpha=0.5)
    ax.axvline(-73, color='#94a3b8', linestyle=':', alpha=0.4)
    ax.axvline(73, color='#94a3b8', linestyle=':', alpha=0.4)
    ax.set_title(f'{args.label}  RATE − FP\npen = {pen_fp:.3f}  '
                  f'(FP={args.fp_rate:.3f})', fontsize=10)
    ax.set_xlabel('distance from dyad (bp)')
    ax.set_ylabel('hit / opp (FP-subtracted)')
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    fig.tight_layout()
    png = args.out_prefix + '.png'
    fig.savefig(png, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'\nWrote {png}')

    # TSV + JSON
    tsv = args.out_prefix + '_profile.tsv'
    with open(tsv, 'w') as f:
        f.write('offset\tcount_mean\trate_mean\trate_fp_subtracted\topp_mean\n')
        for d, c, r, rf, o in zip(d_arr, count_mean, rate_mean,
                                       rate_mean_fp, opp_mean):
            f.write(f'{d}\t{c:.4f}\t{r:.6f}\t{rf:.6f}\t{o:.2f}\n')
    print(f'Wrote {tsv}')

    summary = {
        'label': args.label,
        'n_reads': n_reads,
        'n_anchors': len(count_profiles),
        'anchors_snapped': [(c, sp, op) for c, sp, op in snapped_anchors],
        'count_space': {'dyad': dyad_c, 'linker': linker_c,
                         'penetration': pen_c},
        'rate_space': {'dyad': dyad_r_raw, 'linker': linker_r_raw,
                        'penetration': pen_raw},
        'rate_space_fp_subtracted': {'dyad': dyad_r_fp, 'linker': linker_r_fp,
                                        'penetration': pen_fp,
                                        'fp_rate': args.fp_rate},
        'params': {
            'smooth_window': args.smooth_window,
            'profile_window': W,
            'recenter_radius': args.recenter_radius,
            'fp_rate': args.fp_rate,
        },
    }
    jsn = args.out_prefix + '_summary.json'
    with open(jsn, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'Wrote {jsn}')


if __name__ == '__main__':
    main()
