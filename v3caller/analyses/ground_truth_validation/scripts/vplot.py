#!/usr/bin/env python3
"""V-plot of footprint sizes vs position around enhancers.

For each anchor (enhancer center) in the BED and each input BAM read
that overlaps that anchor's ±window:

  For each footprint (v2 fp_v2, v3 nuc, v3 tf) on that read:
    Project start/end to reference coords.
    Let rel_start = ref_start - anchor_center,
        rel_end   = ref_end   - anchor_center.
    For each rel_pos in [rel_start, rel_end]:
        counts[size][rel_pos + W] += 1

i.e., the ENTIRE WIDTH of each footprint is added, not just the
center. A 5 bp footprint at rel=0 adds to counts[5][rel=-2..+2].

Gaussian-smooth the final 2D count matrix and plot as heatmap.

Panels:
  - v3 TF-size (20-89 bp)
  - v3 nuc-size (90-400 bp)
  - v2 TF-size (fp_v2 entries <90 bp)
  - v2 nuc-size (fp_v2 entries ≥90 bp)
  - ChIP-nexus metaprofile (optional, overlaid)

Splitting by size class prevents the nucleosome mass from
saturating the TF signal.
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

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   '..', '..', 'caller_comparison', 'scripts'))
from parse_ma_calls import parse_ma


def load_bed(bed_path):
    """Parse enhancer/anchor BED. Each row → (chrom, center, name, category)."""
    entries = []
    with open(bed_path) as f:
        for line in f:
            if line.startswith('#') or not line.strip(): continue
            parts = line.rstrip('\n').split('\t')
            chrom = parts[0]
            start = int(parts[1])
            end = int(parts[2])
            name = parts[3] if len(parts) > 3 else f'{chrom}:{start}'
            category = parts[4] if len(parts) > 4 else 'all'
            center = (start + end) // 2
            entries.append((chrom, center, name, category, start, end))
    return entries


def build_qr_map(read):
    """Build dict qp → rp from aligned pairs, once per read."""
    try:
        pairs = read.get_aligned_pairs(matches_only=True)
    except ValueError:
        return {}
    return {qp: rp for qp, rp in pairs}


def project_footprint(qr_map, q_start, q_length):
    """Project footprint via cached qp→rp dict.
    Returns (ref_start, ref_end) or None. Tolerates a few bp of
    indel slop around target qp."""
    if not qr_map: return None
    def lookup(qp):
        for d in range(6):
            if qp + d in qr_map: return qr_map[qp + d]
            if qp - d in qr_map: return qr_map[qp - d]
        return None
    rs = lookup(q_start)
    re = lookup(q_start + q_length)
    if rs is None or re is None: return None
    if re < rs: rs, re = re, rs
    return rs, re


def accumulate_vplot(counts, rel_start, rel_end, size, W, max_size):
    """Add this footprint's width to counts[size][rel_start..rel_end]."""
    if size < 1 or size > max_size: return
    # Clip to window
    lo = max(-W, int(rel_start))
    hi = min(W, int(rel_end))
    if hi < lo: return
    counts[size, lo + W:hi + W + 1] += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in-bam', action='append', required=True)
    ap.add_argument('--anchors-bed', required=True,
                    help='enhancer BED (chrom, start, end, name, category)')
    ap.add_argument('--out-prefix', required=True)
    ap.add_argument('--window', type=int, default=2000,
                    help='±bp around anchor center')
    ap.add_argument('--max-size', type=int, default=400)
    ap.add_argument('--tf-size-max', type=int, default=89)
    ap.add_argument('--smooth-sigma', type=float, default=2.0,
                    help='Gaussian smoothing sigma in bp (position-axis)')
    ap.add_argument('--smooth-size-sigma', type=float, default=1.5,
                    help='Gaussian smoothing sigma in bp (size-axis)')
    ap.add_argument('--max-reads', type=int, default=0)
    ap.add_argument('--category', default=None,
                    help='If set, only use anchors with this category')
    ap.add_argument('--chipnexus', action='append', default=[],
                    help='optional ChIP-nexus bigwig to overlay as '
                         'metaprofile (repeatable)')
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out_prefix) or '.', exist_ok=True)

    anchors = load_bed(args.anchors_bed)
    if args.category:
        anchors = [a for a in anchors if a[3] == args.category]
    print(f'Loaded {len(anchors)} anchors')
    if not anchors: return

    W = args.window
    MS = args.max_size
    # counts[track][size, 2W+1]
    counts = {
        'v3_tf':  np.zeros((MS + 1, 2 * W + 1), dtype=np.int32),
        'v3_nuc': np.zeros((MS + 1, 2 * W + 1), dtype=np.int32),
        'v2_tf':  np.zeros((MS + 1, 2 * W + 1), dtype=np.int32),
        'v2_nuc': np.zeros((MS + 1, 2 * W + 1), dtype=np.int32),
    }

    # Index anchors by chrom for quick fetch
    chrom_anchors = defaultdict(list)
    for a in anchors:
        chrom_anchors[a[0]].append(a)

    n_reads = 0
    n_overlap = 0
    for bam_path in args.in_bam:
        print(f'[{bam_path}]', flush=True)
        bam = pysam.AlignmentFile(bam_path, 'rb', check_sq=False)
        for chrom, chrom_a_list in chrom_anchors.items():
            for anchor in chrom_a_list:
                _, center, _, _, _, _ = anchor
                lo = center - W
                hi = center + W + 1
                try:
                    iterator = bam.fetch(chrom, max(0, lo), hi)
                except ValueError:
                    continue
                for r in iterator:
                    if r.is_unmapped or r.is_secondary or r.is_supplementary:
                        continue
                    n_reads += 1
                    if args.max_reads and n_reads > args.max_reads:
                        break
                    if not r.has_tag('MA'):
                        continue
                    n_overlap += 1
                    ma = parse_ma(r.get_tag('MA'))
                    qr_map = build_qr_map(r)   # cache once per read

                    for s, l in ma['nuc']:
                        pr = project_footprint(qr_map, s, l)
                        if pr is None: continue
                        rs, re = pr
                        track = 'v3_tf' if l <= args.tf_size_max else 'v3_nuc'
                        accumulate_vplot(counts[track],
                                          rs - center, re - center, l, W, MS)
                    for s, l in ma['tf']:
                        pr = project_footprint(qr_map, s, l)
                        if pr is None: continue
                        rs, re = pr
                        track = 'v3_tf' if l <= args.tf_size_max else 'v3_nuc'
                        accumulate_vplot(counts[track],
                                          rs - center, re - center, l, W, MS)
                    for s, l in ma['fp_v2']:
                        pr = project_footprint(qr_map, s, l)
                        if pr is None: continue
                        rs, re = pr
                        track = 'v2_tf' if l <= args.tf_size_max else 'v2_nuc'
                        accumulate_vplot(counts[track],
                                          rs - center, re - center, l, W, MS)
        bam.close()

    print(f'Processed {n_reads:,} reads, {n_overlap:,} with MA tags')

    # Smooth
    smoothed = {}
    for name, c in counts.items():
        smoothed[name] = gaussian_filter(
            c.astype(float),
            sigma=(args.smooth_size_sigma, args.smooth_sigma))

    # ChIP-nexus metaprofile
    nexus_profiles = {}
    if args.chipnexus:
        import pyBigWig
        xs = np.arange(-W, W + 1)
        for bw_path in args.chipnexus:
            name = os.path.basename(bw_path).replace('.bw', '')
            bw = pyBigWig.open(bw_path)
            profile = np.zeros(2 * W + 1)
            n_used = 0
            for chrom, center, *_ in anchors:
                try:
                    vals = bw.values(chrom, center - W, center + W + 1,
                                      numpy=True)
                except (RuntimeError, ValueError):
                    continue
                if vals is None or np.all(np.isnan(vals)):
                    continue
                vals = np.nan_to_num(vals, nan=0.0)
                if len(vals) == 2 * W + 1:
                    profile += vals
                    n_used += 1
            if n_used:
                profile /= n_used
            nexus_profiles[name] = profile
            bw.close()

    # Plot: 2x3 grid
    # [v2 TF vplot] [v3 TF vplot] [ChIP nexus metaprofile]
    # [v2 nuc vplot] [v3 nuc vplot] [caller TF density]
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1])

    def plot_vplot(ax, mat, size_lo, size_hi, title):
        # Slice to size range
        sub = mat[size_lo:size_hi + 1]
        if sub.sum() == 0:
            ax.text(0.5, 0.5, 'no data',
                     ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title, fontsize=10)
            return
        vmax = np.percentile(sub[sub > 0], 99) if (sub > 0).sum() else 1
        im = ax.imshow(sub, aspect='auto', origin='lower',
                       extent=[-W, W, size_lo, size_hi],
                       cmap='viridis', vmin=0, vmax=max(vmax, 1))
        ax.axvline(0, color='red', linestyle='--', linewidth=0.5, alpha=0.7)
        ax.set_xlabel('position rel. enhancer center (bp)')
        ax.set_ylabel('footprint size (bp)')
        ax.set_title(title, fontsize=10)
        plt.colorbar(im, ax=ax, label='smoothed count')

    ax = fig.add_subplot(gs[0, 0])
    plot_vplot(ax, smoothed['v2_tf'], 20, args.tf_size_max,
                f'v2 fp_v2+ (TF-size 20-{args.tf_size_max})')
    ax = fig.add_subplot(gs[0, 1])
    plot_vplot(ax, smoothed['v3_tf'], 20, args.tf_size_max,
                f'v3 nuc+tf (TF-size 20-{args.tf_size_max})')

    # ChIP-nexus metaprofile
    ax = fig.add_subplot(gs[0, 2])
    if nexus_profiles:
        xs = np.arange(-W, W + 1)
        for name, prof in nexus_profiles.items():
            ax.plot(xs, prof, label=name, linewidth=1.2)
        ax.axvline(0, color='red', linestyle='--', linewidth=0.5, alpha=0.7)
        ax.set_xlabel('position rel. enhancer center (bp)')
        ax.set_ylabel('ChIP-nexus mean signal')
        ax.set_title(f'ChIP-nexus metaprofile ({len(anchors)} anchors)')
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
    else:
        ax.axis('off')

    ax = fig.add_subplot(gs[1, 0])
    plot_vplot(ax, smoothed['v2_nuc'], 90, 350,
                'v2 fp_v2+ (nuc-size 90-350)')
    ax = fig.add_subplot(gs[1, 1])
    plot_vplot(ax, smoothed['v3_nuc'], 90, 350,
                'v3 nuc+tf (nuc-size 90-350)')

    # TF-size density (summed over size axis) — 1D comparison
    ax = fig.add_subplot(gs[1, 2])
    xs = np.arange(-W, W + 1)
    v2_tf_density = smoothed['v2_tf'][20:args.tf_size_max + 1].sum(axis=0)
    v3_tf_density = smoothed['v3_tf'][20:args.tf_size_max + 1].sum(axis=0)
    ax.plot(xs, v2_tf_density, color='#dc2626', label='v2 TF density',
            linewidth=1.2)
    ax.plot(xs, v3_tf_density, color='#1e3a8a', label='v3 TF density',
            linewidth=1.2)
    ax.axvline(0, color='red', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.set_xlabel('position rel. enhancer center (bp)')
    ax.set_ylabel('summed TF-size count (smoothed)')
    ax.set_title('TF-size call density (v2 vs v3)')
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    cat_suffix = f' ({args.category})' if args.category else ''
    fig.suptitle(f'V-plot at {len(anchors)} enhancers{cat_suffix} — '
                 f'±{W} bp, sigma={args.smooth_sigma}/{args.smooth_size_sigma} bp',
                 fontsize=11)
    fig.tight_layout()
    png = args.out_prefix + '_vplot.png'
    fig.savefig(png, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'Wrote {png}')

    # Save raw count matrices (compressed)
    npz = args.out_prefix + '_vplot_counts.npz'
    np.savez_compressed(npz,
                        v2_tf=counts['v2_tf'], v2_nuc=counts['v2_nuc'],
                        v3_tf=counts['v3_tf'], v3_nuc=counts['v3_nuc'],
                        window=W, max_size=MS)
    print(f'Wrote {npz}')


if __name__ == '__main__':
    main()
