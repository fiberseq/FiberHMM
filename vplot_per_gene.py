#!/usr/bin/env python3
"""Per-gene V-plot with asymmetric window + bigwig overlays.

Strand-aware: +x axis always points downstream (gene body direction).
For a gene on the - strand, upstream coordinates are mirrored so the
plot reads left-to-right as upstream -> TSS -> downstream.

Outputs one figure per BED entry, with:
  - v2 TF-size v-plot  (ns/nl with nl <= tf_size_max)
  - recaller TF v-plot (tn/tl with ts >= min_ts)
  - Stacked bigwig overlays (one per --bw flag)
  - v2 nuc v-plot and recaller "nuc+tf" v-plot below
"""
from __future__ import annotations

import argparse
import os
from collections import defaultdict

import numpy as np
import pysam
from scipy.ndimage import gaussian_filter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_bed(bed_path, only_name=None):
    entries = []
    with open(bed_path) as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.rstrip('\n').split('\t')
            chrom = parts[0]
            start = int(parts[1]); end = int(parts[2])
            name = parts[3] if len(parts) > 3 else f'{chrom}:{start}'
            strand = parts[5] if len(parts) > 5 else '+'
            center = (start + end) // 2
            if only_name and name != only_name:
                continue
            entries.append((chrom, center, name, strand))
    return entries


def build_qr_map(read):
    try:
        return {qp: rp for qp, rp in read.get_aligned_pairs(matches_only=True)}
    except ValueError:
        return {}


def project_footprint(qr_map, q_start, q_length):
    if not qr_map:
        return None
    def lookup(qp):
        for d in range(6):
            if qp + d in qr_map: return qr_map[qp + d]
            if qp - d in qr_map: return qr_map[qp - d]
        return None
    rs = lookup(q_start); re = lookup(q_start + q_length)
    if rs is None or re is None:
        return None
    if re < rs: rs, re = re, rs
    return rs, re


def accumulate(counts, rel_start, rel_end, size, lo_pos, hi_pos, max_size):
    if size < 1 or size > max_size: return
    lo = max(lo_pos, int(rel_start))
    hi = min(hi_pos - 1, int(rel_end))
    if hi < lo: return
    counts[size, lo - lo_pos:hi - lo_pos + 1] += 1


def get_arr(read, tag):
    try: return list(read.get_tag(tag))
    except KeyError: return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in-bam', required=True)
    ap.add_argument('--anchors-bed', required=True)
    ap.add_argument('--gene', default=None,
                    help='Only plot this gene (by BED name col 4). Omit to plot all.')
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--upstream', type=int, default=8000)
    ap.add_argument('--downstream', type=int, default=2000)
    ap.add_argument('--max-size', type=int, default=400)
    ap.add_argument('--tf-size-max', type=int, default=89)
    ap.add_argument('--smooth-sigma', type=float, default=20.0,
                    help='Gaussian sigma on position axis (bp). 20 bp is good for 8kb views.')
    ap.add_argument('--smooth-size-sigma', type=float, default=1.5)
    ap.add_argument('--min-ts', type=int, default=75,
                    help='Only recaller TFs with ts >= this (default 75 = LLR>=15 nats)')
    ap.add_argument('--bw', action='append', default=[],
                    help='Bigwig overlay, form NAME:PATH. Repeatable.')
    ap.add_argument('--enhancer', action='append', default=[],
                    help='Enhancer annotation, form NAME:GENE:REL_START:REL_END. '
                         'REL coords are in gene frame (+x = downstream). Repeatable.')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    anchors = load_bed(args.anchors_bed, only_name=args.gene)
    print(f'{len(anchors)} anchors to plot')

    # Parse enhancer annotations -> {gene: [(name, rel_s, rel_e), ...]}
    enh_by_gene = defaultdict(list)
    for spec in args.enhancer:
        parts = spec.split(':')
        if len(parts) != 4:
            print(f'Bad --enhancer spec (need NAME:GENE:S:E): {spec}'); continue
        ename, egene, es, ee = parts[0], parts[1], int(parts[2]), int(parts[3])
        enh_by_gene[egene].append((ename, es, ee))

    # Parse bigwig list
    bw_specs = []
    if args.bw:
        import pyBigWig
        for spec in args.bw:
            name, path = spec.split(':', 1)
            bw_specs.append((name, path, pyBigWig.open(path)))

    for chrom, center, name, strand in anchors:
        sign = 1 if strand == '+' else -1
        # Window in reference coordinates
        if strand == '+':
            ref_lo = center - args.upstream
            ref_hi = center + args.downstream
        else:
            ref_lo = center - args.downstream
            ref_hi = center + args.upstream

        # Relative coordinate range in "gene frame" (+x = downstream)
        rel_lo = -args.upstream
        rel_hi = args.downstream + 1
        WLEN = rel_hi - rel_lo  # total width

        counts = {
            'v2_tf':  np.zeros((args.max_size + 1, WLEN), dtype=np.int32),
            'v2_nuc': np.zeros((args.max_size + 1, WLEN), dtype=np.int32),
            'rc_tf':  np.zeros((args.max_size + 1, WLEN), dtype=np.int32),
            'rc_nuc': np.zeros((args.max_size + 1, WLEN), dtype=np.int32),
        }

        bam = pysam.AlignmentFile(args.in_bam, 'rb', check_sq=False)
        try:
            itr = bam.fetch(chrom, max(0, ref_lo), ref_hi)
        except ValueError:
            print(f'Skip {name}: fetch failed'); continue
        n_reads = 0
        for r in itr:
            if r.is_unmapped or r.is_secondary or r.is_supplementary: continue
            ns = get_arr(r, 'ns'); nl = get_arr(r, 'nl')
            if not ns: continue
            n_reads += 1
            qr_map = build_qr_map(r)

            for s, l in zip(ns, nl):
                pr = project_footprint(qr_map, int(s), int(l))
                if pr is None: continue
                rs, re = pr
                rel_s = sign * (rs - center)
                rel_e = sign * (re - center)
                if sign < 0: rel_s, rel_e = rel_e, rel_s
                track = 'v2_tf' if l <= args.tf_size_max else 'v2_nuc'
                accumulate(counts[track], rel_s, rel_e, int(l),
                           rel_lo, rel_hi, args.max_size)
                if l >= 90:
                    accumulate(counts['rc_nuc'], rel_s, rel_e, int(l),
                               rel_lo, rel_hi, args.max_size)

            tn = get_arr(r, 'tn'); tl = get_arr(r, 'tl')
            ts = get_arr(r, 'ts') if r.has_tag('ts') else [255] * len(tn)
            for s, l, ts_i in zip(tn, tl, ts):
                if ts_i < args.min_ts: continue
                pr = project_footprint(qr_map, int(s), int(l))
                if pr is None: continue
                rs, re = pr
                rel_s = sign * (rs - center)
                rel_e = sign * (re - center)
                if sign < 0: rel_s, rel_e = rel_e, rel_s
                track = 'rc_tf' if l <= args.tf_size_max else 'rc_nuc'
                accumulate(counts[track], rel_s, rel_e, int(l),
                           rel_lo, rel_hi, args.max_size)
        bam.close()

        # Gather bigwig profiles (strand-flipped to gene frame)
        bw_profiles = []
        for bw_name, _, bw in bw_specs:
            try:
                vals = bw.values(chrom, max(0, ref_lo), ref_hi, numpy=True)
            except (RuntimeError, ValueError):
                vals = None
            if vals is None:
                prof = np.zeros(WLEN)
            else:
                vals = np.nan_to_num(vals, nan=0.0)
                # Pad/trim to WLEN
                if len(vals) < WLEN:
                    vals = np.concatenate([vals, np.zeros(WLEN - len(vals))])
                else:
                    vals = vals[:WLEN]
                if strand == '-':
                    vals = vals[::-1]
                prof = vals
            bw_profiles.append((bw_name, prof))

        # Smooth
        sm = {}
        for k, c in counts.items():
            sm[k] = gaussian_filter(c.astype(float),
                                    sigma=(args.smooth_size_sigma, args.smooth_sigma))

        # Layout: header info + 2 v-plots stacked + bigwig stack + nuc v-plots
        n_bw = max(1, len(bw_profiles))
        fig = plt.figure(figsize=(14, 5 + 1.4 * n_bw))
        gs = fig.add_gridspec(4 + n_bw, 1, height_ratios=[2, 2] + [0.8] * n_bw + [2, 2],
                              hspace=0.15)

        xs = np.arange(rel_lo, rel_hi)

        def plot_vplot(ax, mat, size_lo, size_hi, title):
            sub = mat[size_lo:size_hi + 1]
            if sub.sum() == 0:
                ax.text(0.5, 0.5, 'no data', ha='center', va='center',
                        transform=ax.transAxes)
                ax.set_title(title, fontsize=10); return
            vmax = np.percentile(sub[sub > 0], 99) if (sub > 0).sum() else 1
            ax.imshow(sub, aspect='auto', origin='lower',
                      extent=[rel_lo, rel_hi - 1, size_lo, size_hi],
                      cmap='viridis', vmin=0, vmax=max(vmax, 1))
            ax.axvline(0, color='red', linestyle='--', linewidth=0.6, alpha=0.8)
            ax.set_ylabel('size (bp)', fontsize=9)
            ax.set_title(title, fontsize=10)
            ax.set_xlim(rel_lo, rel_hi - 1)

        # Helper to draw enhancer annotations on any axis
        enh_list = enh_by_gene.get(name, [])
        def draw_enhancers(ax, ymin=None, ymax=None, label_y_frac=0.92):
            for ename, es, ee in enh_list:
                ax.axvspan(es, ee, color='orange', alpha=0.18, linewidth=0)
                ax.axvline(es, color='orange', linestyle=':', linewidth=0.5, alpha=0.5)
                ax.axvline(ee, color='orange', linestyle=':', linewidth=0.5, alpha=0.5)
                if label_y_frac is not None:
                    mid = (es + ee) / 2
                    y0, y1 = ax.get_ylim()
                    ax.text(mid, y0 + (y1 - y0) * label_y_frac, ename,
                            ha='center', va='top', fontsize=8, color='darkorange',
                            fontweight='bold')

        ax_v2 = fig.add_subplot(gs[0, 0])
        plot_vplot(ax_v2, sm['v2_tf'], 20, args.tf_size_max,
                   f'{name}  ({strand} strand)  --  v2 ns/nl TF-size 20-{args.tf_size_max}  ({n_reads} reads)')
        draw_enhancers(ax_v2)
        ax_rc = fig.add_subplot(gs[1, 0], sharex=ax_v2)
        plot_vplot(ax_rc, sm['rc_tf'], 20, args.tf_size_max,
                   f'recaller tn/tl (ts>={args.min_ts}) TF-size 20-{args.tf_size_max}')
        draw_enhancers(ax_rc)

        # Bigwig overlays
        for i, (bw_name, prof) in enumerate(bw_profiles):
            ax = fig.add_subplot(gs[2 + i, 0], sharex=ax_v2)
            smooth_prof = gaussian_filter(prof, sigma=args.smooth_sigma)
            ax.fill_between(xs, 0, smooth_prof, alpha=0.6, color=f'C{i}')
            ax.plot(xs, smooth_prof, color=f'C{i}', linewidth=0.8)
            ax.axvline(0, color='red', linestyle='--', linewidth=0.6, alpha=0.8)
            ax.set_ylabel(bw_name, fontsize=9)
            ax.grid(alpha=0.3, axis='x')
            ax.set_xlim(rel_lo, rel_hi - 1)
            draw_enhancers(ax, label_y_frac=None)

        ax_v2n = fig.add_subplot(gs[2 + n_bw, 0], sharex=ax_v2)
        plot_vplot(ax_v2n, sm['v2_nuc'], 90, 250, 'v2 nuc-size 90-250')
        ax_rcn = fig.add_subplot(gs[3 + n_bw, 0], sharex=ax_v2)
        plot_vplot(ax_rcn, sm['rc_nuc'], 90, 250, 'recaller nuc-size 90-250')
        ax_rcn.set_xlabel('position rel. TSS (bp, +x = downstream)', fontsize=10)

        fig.suptitle(f'{name}: {chrom}:{ref_lo}-{ref_hi}  '
                     f'(-{args.upstream} / +{args.downstream})  strand={strand}',
                     fontsize=11, y=0.995)
        out = os.path.join(args.out_dir, f'pergene_{name}.png')
        fig.savefig(out, dpi=120, bbox_inches='tight')
        plt.close(fig)
        print(f'Wrote {out}  ({n_reads} reads)')

    for _, _, bw in bw_specs:
        bw.close()


if __name__ == '__main__':
    main()
