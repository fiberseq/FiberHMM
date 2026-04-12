#!/usr/bin/env python3
"""Measure nucleosome penetration rate from bulk hit density.

De novo penetration measurement from our own caller is biased (we
select for low-penetration nucleosomes). Instead, use the BULK hit
density across many reads — the pattern resembles MNase-seq:
troughs = nucleosome centers (high occupancy), peaks = linkers
(accessible). This is unbiased by per-read calling decisions.

Procedure:
  1. Pileup hit density per reference position (all reads combined)
  2. Smooth with a sliding window
  3. Detect local minima (trough positions = putative dyads)
  4. For each trough, extract the hit profile at ±100 bp
  5. Average profiles across all troughs → canonical nucleosome signature
  6. Compute penetration_fraction = hit_rate_at_dyad / hit_rate_at_linker

Output:
  - penetration.tsv: per-offset hit rate (position relative to dyad)
  - penetration.png: plot of the average profile
  - penetration_summary.json: summary stats including inferred
    penetration_fraction

Usage:
  python measure_penetration.py --in-bam scdaf.bam \
      --out-prefix scdaf_penetration \
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


def pileup_hit_density(bam_path, enzyme='daf', max_reads=0,
                         min_mapq=20):
    """Per-reference-position hit count and opp count.

    Returns:
        dict[chrom] → (n_opp_arr, n_hit_arr) indexed by ref position
        (sparse, using defaultdict of arrays)
    """
    # Accumulate per (chrom, pos)
    opp_counts = defaultdict(lambda: defaultdict(int))
    hit_counts = defaultdict(lambda: defaultdict(int))

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
            if enzyme == 'daf':
                if rbu == 'C':
                    opp_counts[chrom][rp] += 1
                    if qb in ('T', 'Y'):
                        hit_counts[chrom][rp] += 1
                elif rbu == 'G':
                    opp_counts[chrom][rp] += 1
                    if qb in ('A', 'R'):
                        hit_counts[chrom][rp] += 1
            # Hia5 path: MM/ML — not implemented in this simple pileup.
            # For m6A data, use a different aggregation that parses
            # MM/ML tags per read.
    bam.close()
    return opp_counts, hit_counts, n


def build_density_array(opp_counts, hit_counts, chrom, start, end):
    """Convert sparse dicts to a dense array for one genomic region."""
    L = end - start
    opp = np.zeros(L, dtype=np.int32)
    hit = np.zeros(L, dtype=np.int32)
    for pos, count in opp_counts[chrom].items():
        if start <= pos < end:
            opp[pos - start] = count
    for pos, count in hit_counts[chrom].items():
        if start <= pos < end:
            hit[pos - start] = count
    return opp, hit


def smooth(x, window=10):
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode='same')


def find_troughs(density, min_distance=160, prominence_frac=0.1,
                 min_width=80, max_width=200):
    """Find local minima in the density array that could be dyads.

    min_distance: minimum spacing between troughs (≈ nucleosome
        repeat length). For fly ~180 bp, human ~190 bp.
    prominence_frac: trough depth relative to local max, as a
        fraction of the local range. 0.1 = trough must dip ≥10%
        below neighboring peaks to count.
    min_width / max_width: require the trough to be at least
        min_width bp wide (measured at rel_height=0.5 of prominence)
        and at most max_width bp. Nucleosomes are ~120 bp wide;
        TF footprints ~20-30 bp; so min_width=80 excludes TFs while
        accepting partially-merged nucs.
    """
    from scipy.signal import find_peaks
    inverted = -density
    # Use the built-in width filter in find_peaks
    peaks, props = find_peaks(
        inverted, distance=min_distance,
        width=(min_width, max_width), rel_height=0.5,
    )
    if len(peaks) == 0:
        return np.array([], dtype=int)
    # Additional filter: prominence relative to local range
    half = min_distance // 2
    kept = []
    for p in peaks:
        lo = max(0, p - half)
        hi = min(len(density), p + half + 1)
        local_max = density[lo:hi].max()
        local_min = density[p]
        if local_max <= 0:
            continue
        rel_depth = (local_max - local_min) / local_max
        if rel_depth >= prominence_frac:
            kept.append(p)
    return np.array(kept, dtype=int)


def aggregate_profile(density, trough_positions, window=120,
                       recenter_radius=30):
    """Average the density profile at ±window around each trough.

    recenter_radius: before extracting the profile, snap each trough
        center to the argmin of the smoothed signal within ±radius.
        This removes the ±10-20 bp jitter in peak detection that
        otherwise blurs the averaged trough bottom. Set to 0 to
        disable.
    """
    w = window
    profiles = []
    recentered = []
    for t in trough_positions:
        # Snap to argmin within ±recenter_radius
        if recenter_radius > 0:
            rlo = max(0, t - recenter_radius)
            rhi = min(len(density), t + recenter_radius + 1)
            t_snap = rlo + int(np.argmin(density[rlo:rhi]))
        else:
            t_snap = t
        lo = t_snap - w
        hi = t_snap + w + 1
        if lo < 0 or hi > len(density):
            continue
        profiles.append(density[lo:hi])
        recentered.append(t_snap)
    if not profiles:
        return None, 0
    profiles = np.array(profiles)
    return profiles.mean(axis=0), len(profiles)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--in-bam', required=True, action='append',
                    help='input BAM(s) — pass multiple times to '
                         'aggregate across datasets')
    ap.add_argument('--label', action='append', default=None,
                    help='short label per BAM (same order as --in-bam); '
                         'used in the plot legend')
    ap.add_argument('--out-prefix', required=True,
                    help='output file prefix (will write .tsv, .png, .json)')
    ap.add_argument('--enzyme', default='daf', choices=['daf', 'hia5'])
    ap.add_argument('--chrom', default=None,
                    help='restrict to single chromosome (optional)')
    ap.add_argument('--region-start', type=int, default=None,
                    help='with --chrom, restrict to this start position')
    ap.add_argument('--region-end', type=int, default=None,
                    help='with --chrom, restrict to this end position')
    ap.add_argument('--smooth-window', type=int, default=10,
                    help='bulk density smoothing window (bp)')
    ap.add_argument('--min-distance', type=int, default=160,
                    help='minimum spacing between troughs (bp, ~NRL)')
    ap.add_argument('--prominence', type=float, default=0.15,
                    help='minimum relative depth of trough (0-1)')
    ap.add_argument('--min-width', type=int, default=80,
                    help='minimum trough width at half prominence (bp); '
                         '80 excludes TF footprints (~20-30 bp) and keeps '
                         'nucleosome-scale troughs (~120 bp)')
    ap.add_argument('--max-width', type=int, default=200,
                    help='maximum trough width at half prominence (bp); '
                         'excludes over-wide features that span '
                         'multiple nucs')
    ap.add_argument('--profile-window', type=int, default=120,
                    help='±bp window around each trough')
    ap.add_argument('--recenter-radius', type=int, default=30,
                    help='snap each detected trough to argmin of '
                         'smoothed signal within ±radius (bp); '
                         'removes detection jitter that blurs the '
                         'averaged trough bottom. 0 to disable.')
    ap.add_argument('--max-reads', type=int, default=0,
                    help='cap on reads to pileup (0 = all)')
    args = ap.parse_args()

    labels = args.label or [os.path.basename(b).replace('.bam', '')
                              for b in args.in_bam]
    if len(labels) != len(args.in_bam):
        labels = [os.path.basename(b).replace('.bam', '') for b in args.in_bam]

    # Accumulate per-BAM profiles + combined
    all_profiles = []
    total_troughs = 0
    per_dataset = []  # list of (label, profile, n_used)
    for bam_path, label in zip(args.in_bam, labels):
        if not os.path.exists(bam_path):
            print(f'Skipping {label}: not found')
            continue
        print(f'\n[{label}] Piling up {bam_path}...')
        opp_counts, hit_counts, n_reads = pileup_hit_density(
            bam_path, enzyme=args.enzyme, max_reads=args.max_reads)
        total_pos = sum(len(c) for c in opp_counts.values())
        print(f'  {n_reads} reads, {total_pos} positions with ≥1 opp')

        # All chromosomes with enough coverage
        regions = []
        if args.chrom:
            if args.chrom in opp_counts:
                positions = sorted(opp_counts[args.chrom].keys())
                start = args.region_start if args.region_start is not None else positions[0]
                end = args.region_end if args.region_end is not None else positions[-1] + 1
                regions = [(args.chrom, start, end)]
        else:
            # Pick only the chromosome(s) that carry the bulk of
            # the reads — otherwise whole-genome BAMs bring in
            # scattered low-coverage positions that dominate the
            # search and give spurious troughs.
            total_opp = {c: sum(d.values()) for c, d in opp_counts.items()}
            if not total_opp:
                continue
            top_opp = max(total_opp.values())
            for chrom, d in opp_counts.items():
                if total_opp[chrom] < 0.1 * top_opp:
                    continue  # chrom has <10% of peak coverage
                positions = sorted(d.keys())
                regions.append((chrom, positions[0], positions[-1] + 1))

        ds_profiles = []
        ds_troughs = 0
        for chrom, start, end in regions:
            opp_arr, hit_arr = build_density_array(opp_counts, hit_counts,
                                                      chrom, start, end)
            density = smooth(hit_arr.astype(float), args.smooth_window)
            if density.sum() == 0:
                continue
            troughs = find_troughs(density, min_distance=args.min_distance,
                                      prominence_frac=args.prominence,
                                      min_width=args.min_width,
                                      max_width=args.max_width)
            if len(troughs) == 0:
                continue
            profile, n_used = aggregate_profile(
                density, troughs, window=args.profile_window,
                recenter_radius=args.recenter_radius)
            if profile is None:
                continue
            ds_profiles.append((profile, n_used))
            ds_troughs += n_used
            all_profiles.append((profile, n_used, f'{label}:{chrom}'))
            total_troughs += n_used
        if ds_profiles:
            # Combined profile for this dataset
            ds_weight = sum(n for _, n in ds_profiles)
            ds_mean = np.zeros_like(ds_profiles[0][0])
            for p, n in ds_profiles:
                ds_mean += p * n
            ds_mean /= ds_weight
            per_dataset.append((label, ds_mean, ds_troughs))
            print(f'  → {ds_troughs} troughs used, '
                  f'per-dataset trough/kb: {ds_troughs/max(1,total_pos)*1000:.2f}')

    if not all_profiles:
        print('No profiles found'); return

    # Weighted average across regions by trough count
    d_arr = np.arange(-args.profile_window, args.profile_window + 1)
    total_weight = sum(n for _, n, _ in all_profiles)
    mean_profile = np.zeros_like(all_profiles[0][0])
    for p, n, _ in all_profiles:
        mean_profile += p * n
    mean_profile /= total_weight

    # Compute penetration:
    #   - dyad rate = MIN of the averaged profile within ±20 bp of
    #     center (after recentering, the true trough bottom is at 0,
    #     but allow ±20 for residual jitter).
    #   - linker rate = MAX of the averaged profile (typically at
    #     |d| ≈ 90-100 bp, the linker flanks)
    center = args.profile_window
    dyad_rate = float(mean_profile[center - 20:center + 21].min())
    linker_rate = float(np.max(mean_profile))
    if linker_rate > 0:
        penetration = dyad_rate / linker_rate
    else:
        penetration = float('nan')

    print(f'\n=== Penetration estimate ===')
    print(f'  Troughs analyzed: {total_troughs}')
    print(f'  Dyad rate (±5 bp): {dyad_rate:.2f}')
    print(f'  Linker rate (max): {linker_rate:.2f}')
    print(f'  Penetration (dyad / linker): {penetration:.3f}')
    print(f'  → penetration_fraction ≈ {penetration:.2f}')

    # Write outputs
    tsv = args.out_prefix + '_profile.tsv'
    with open(tsv, 'w') as f:
        f.write('offset\thit_density\n')
        for d, v in zip(d_arr, mean_profile):
            f.write(f'{d}\t{v:.4f}\n')
    print(f'\nWrote {tsv}')

    # Plot: per-dataset (normalized so max=1) + combined average
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    cmap = plt.cm.tab10
    ax1.set_title('Per-dataset profiles (normalized to max)', fontsize=11)
    for i, (label, prof, n) in enumerate(per_dataset):
        if prof.max() > 0:
            norm = prof / prof.max()
            ax1.plot(d_arr, norm, linewidth=1.2, alpha=0.7,
                      color=cmap(i % 10),
                      label=f'{label} (n={n})')
    ax1.axvline(0, color='#dc2626', linestyle='--', alpha=0.6)
    ax1.axvline(-73, color='#94a3b8', linestyle=':', alpha=0.5)
    ax1.axvline(73, color='#94a3b8', linestyle=':', alpha=0.5)
    ax1.set_xlabel('distance from dyad (bp)')
    ax1.set_ylabel('hit density / max')
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=7, loc='lower right')

    # Combined profile (weighted across all datasets)
    ax2.plot(d_arr, mean_profile, color='#1e3a8a', linewidth=2.2)
    ax2.axvline(0, color='#dc2626', linestyle='--', alpha=0.6, label='dyad')
    ax2.axvline(-73, color='#94a3b8', linestyle=':', alpha=0.5,
                 label='canonical edge (±73)')
    ax2.axvline(73, color='#94a3b8', linestyle=':', alpha=0.5)
    ax2.axhline(dyad_rate, color='#f59e0b', linestyle='--', alpha=0.5,
                 label=f'dyad rate ({dyad_rate:.2f})')
    ax2.axhline(linker_rate, color='#16a34a', linestyle='--', alpha=0.5,
                 label=f'linker rate ({linker_rate:.2f})')
    ax2.set_xlabel('distance from dyad (bp)')
    ax2.set_ylabel('mean hit density (weighted)')
    ax2.set_title(f'Combined  ({total_troughs} dyads across {len(per_dataset)} '
                   f'datasets)\npenetration = {penetration:.3f}', fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)

    png = args.out_prefix + '.png'
    fig.tight_layout()
    fig.savefig(png, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'Wrote {png}')

    # Summary JSON
    summary = {
        'n_reads': n_reads,
        'n_troughs': total_troughs,
        'dyad_rate': float(dyad_rate),
        'linker_rate': float(linker_rate),
        'penetration_fraction': float(penetration),
        'params': {
            'smooth_window': args.smooth_window,
            'min_distance': args.min_distance,
            'prominence': args.prominence,
            'min_width': args.min_width,
            'max_width': args.max_width,
            'profile_window': args.profile_window,
        },
    }
    jsn = args.out_prefix + '_summary.json'
    with open(jsn, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'Wrote {jsn}')


if __name__ == '__main__':
    main()
