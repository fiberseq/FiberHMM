#!/usr/bin/env python3
"""V-plot for tf_recaller output.

Same layout as v3caller/analyses/ground_truth_validation/scripts/vplot.py
but reads:
  - v2 track from legacy ns/nl tags (no MA parsing needed)
  - recaller track from tn/tl/ts tags (new LLR TF calls)

For each anchor in the BED and each overlapping read, the full projected
width of every footprint is added to counts[size][rel_pos + W].
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict

import numpy as np
import pysam
from scipy.ndimage import gaussian_filter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_bed(bed_path):
    entries = []
    with open(bed_path) as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.rstrip('\n').split('\t')
            chrom = parts[0]
            start = int(parts[1])
            end = int(parts[2])
            name = parts[3] if len(parts) > 3 else f'{chrom}:{start}'
            category = parts[4] if len(parts) > 4 else 'all'
            strand = parts[5] if len(parts) > 5 else '+'
            center = (start + end) // 2
            entries.append((chrom, center, name, category, strand))
    return entries


def build_qr_map(read):
    try:
        pairs = read.get_aligned_pairs(matches_only=True)
    except ValueError:
        return {}
    return {qp: rp for qp, rp in pairs}


def project_footprint(qr_map, q_start, q_length):
    if not qr_map:
        return None
    def lookup(qp):
        for d in range(6):
            if qp + d in qr_map: return qr_map[qp + d]
            if qp - d in qr_map: return qr_map[qp - d]
        return None
    rs = lookup(q_start)
    re = lookup(q_start + q_length)
    if rs is None or re is None:
        return None
    if re < rs:
        rs, re = re, rs
    return rs, re


def accumulate(counts, rel_start, rel_end, size, W, max_size):
    if size < 1 or size > max_size:
        return
    lo = max(-W, int(rel_start))
    hi = min(W, int(rel_end))
    if hi < lo:
        return
    counts[size, lo + W:hi + W + 1] += 1


def get_arr(read, tag):
    try:
        return list(read.get_tag(tag))
    except KeyError:
        return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in-bam', required=True,
                    help='Sorted/indexed BAM from tf_recaller (has v2 ns/nl + new tn/tl/ts)')
    ap.add_argument('--anchors-bed', required=True)
    ap.add_argument('--out-prefix', required=True)
    ap.add_argument('--window', type=int, default=2000)
    ap.add_argument('--max-size', type=int, default=400)
    ap.add_argument('--tf-size-max', type=int, default=89)
    ap.add_argument('--smooth-sigma', type=float, default=3.0)
    ap.add_argument('--smooth-size-sigma', type=float, default=1.5)
    ap.add_argument('--min-ts', type=int, default=0,
                    help='Minimum ts (0-255) to include a recaller TF')
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out_prefix) or '.', exist_ok=True)

    anchors = load_bed(args.anchors_bed)
    print(f'Loaded {len(anchors)} anchors')
    if not anchors:
        return

    W = args.window
    MS = args.max_size
    counts = {
        'v2_tf':  np.zeros((MS + 1, 2 * W + 1), dtype=np.int32),
        'v2_nuc': np.zeros((MS + 1, 2 * W + 1), dtype=np.int32),
        'rc_tf':  np.zeros((MS + 1, 2 * W + 1), dtype=np.int32),
        'rc_nuc': np.zeros((MS + 1, 2 * W + 1), dtype=np.int32),
    }

    bam = pysam.AlignmentFile(args.in_bam, 'rb', check_sq=False)
    n_reads = 0
    n_tagged = 0
    for chrom, center, name, category, strand in anchors:
        sign = 1 if strand == '+' else -1
        lo = max(0, center - W)
        hi = center + W + 1
        try:
            iterator = bam.fetch(chrom, lo, hi)
        except ValueError:
            continue
        for r in iterator:
            if r.is_unmapped or r.is_secondary or r.is_supplementary:
                continue
            n_reads += 1
            ns = get_arr(r, 'ns')
            nl = get_arr(r, 'nl')
            if not ns:
                continue
            n_tagged += 1
            qr_map = build_qr_map(r)

            # v2 track -- from ns/nl. Split by size into TF/nuc sub-tracks.
            for s, l in zip(ns, nl):
                pr = project_footprint(qr_map, int(s), int(l))
                if pr is None:
                    continue
                rs, re = pr
                rel_s = sign * (rs - center)
                rel_e = sign * (re - center)
                if sign < 0:
                    rel_s, rel_e = rel_e, rel_s
                track = 'v2_tf' if l <= args.tf_size_max else 'v2_nuc'
                accumulate(counts[track], rel_s, rel_e, int(l), W, MS)

            # recaller track -- from tn/tl/ts
            tn = get_arr(r, 'tn')
            tl = get_arr(r, 'tl')
            ts = get_arr(r, 'ts') if r.has_tag('ts') else [255] * len(tn)
            # Also carry v2 nucleosomes (nl >= 90) into rc_nuc so the
            # bottom-right panel shows recaller+v2 nucs for context.
            for s, l in zip(ns, nl):
                if l >= 90:
                    pr = project_footprint(qr_map, int(s), int(l))
                    if pr is None:
                        continue
                    rs, re = pr
                    rel_s = sign * (rs - center)
                    rel_e = sign * (re - center)
                    if sign < 0:
                        rel_s, rel_e = rel_e, rel_s
                    accumulate(counts['rc_nuc'], rel_s, rel_e, int(l), W, MS)
            for s, l, ts_i in zip(tn, tl, ts):
                if ts_i < args.min_ts:
                    continue
                pr = project_footprint(qr_map, int(s), int(l))
                if pr is None:
                    continue
                rs, re = pr
                rel_s = sign * (rs - center)
                rel_e = sign * (re - center)
                if sign < 0:
                    rel_s, rel_e = rel_e, rel_s
                track = 'rc_tf' if l <= args.tf_size_max else 'rc_nuc'
                accumulate(counts[track], rel_s, rel_e, int(l), W, MS)
    bam.close()
    print(f'Processed {n_reads:,} reads ({n_tagged:,} carry v2 tags)')

    # Smooth
    smoothed = {}
    for k, c in counts.items():
        smoothed[k] = gaussian_filter(
            c.astype(float),
            sigma=(args.smooth_size_sigma, args.smooth_sigma)
        )

    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1])

    def plot_vplot(ax, mat, size_lo, size_hi, title):
        sub = mat[size_lo:size_hi + 1]
        if sub.sum() == 0:
            ax.text(0.5, 0.5, 'no data', ha='center', va='center',
                    transform=ax.transAxes)
            ax.set_title(title, fontsize=10)
            return
        vmax = np.percentile(sub[sub > 0], 99) if (sub > 0).sum() else 1
        im = ax.imshow(sub, aspect='auto', origin='lower',
                       extent=[-W, W, size_lo, size_hi],
                       cmap='viridis', vmin=0, vmax=max(vmax, 1))
        ax.axvline(0, color='red', linestyle='--', linewidth=0.5, alpha=0.7)
        ax.set_xlabel('position rel. TSS (bp)')
        ax.set_ylabel('footprint size (bp)')
        ax.set_title(title, fontsize=10)
        plt.colorbar(im, ax=ax, label='smoothed count')

    ax = fig.add_subplot(gs[0, 0])
    plot_vplot(ax, smoothed['v2_tf'], 20, args.tf_size_max,
               f'v2 ns/nl (TF-size 20-{args.tf_size_max})')
    ax = fig.add_subplot(gs[0, 1])
    plot_vplot(ax, smoothed['rc_tf'], 20, args.tf_size_max,
               f'recaller tn/tl (TF-size 20-{args.tf_size_max})')
    ax = fig.add_subplot(gs[0, 2])
    xs = np.arange(-W, W + 1)
    v2_density = smoothed['v2_tf'][20:args.tf_size_max + 1].sum(axis=0)
    rc_density = smoothed['rc_tf'][20:args.tf_size_max + 1].sum(axis=0)
    ax.plot(xs, v2_density, color='#dc2626', label='v2 ns/nl density', linewidth=1.2)
    ax.plot(xs, rc_density, color='#1e3a8a', label='recaller density', linewidth=1.2)
    ax.axvline(0, color='red', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.set_xlabel('position rel. TSS (bp)')
    ax.set_ylabel('summed TF-size count (smoothed)')
    ax.set_title('TF-size call density')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[1, 0])
    plot_vplot(ax, smoothed['v2_nuc'], 90, 350, 'v2 ns/nl (nuc-size 90-350)')
    ax = fig.add_subplot(gs[1, 1])
    plot_vplot(ax, smoothed['rc_nuc'], 90, 350, 'recaller (nuc-size 90-350)')
    ax = fig.add_subplot(gs[1, 2])
    ax.axis('off')

    fig.suptitle(
        f'V-plot at {len(anchors)} TSSs -- +/-{W} bp, sigma={args.smooth_sigma}/{args.smooth_size_sigma} bp',
        fontsize=11
    )
    fig.tight_layout()
    png = args.out_prefix + '_vplot.png'
    fig.savefig(png, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'Wrote {png}')

    npz = args.out_prefix + '_vplot_counts.npz'
    np.savez_compressed(
        npz, v2_tf=counts['v2_tf'], v2_nuc=counts['v2_nuc'],
        rc_tf=counts['rc_tf'], rc_nuc=counts['rc_nuc'],
        window=W, max_size=MS
    )
    print(f'Wrote {npz}')


if __name__ == '__main__':
    main()
