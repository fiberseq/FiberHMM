#!/usr/bin/env python3
"""Single-read snapshots of paused Pol II calls.

For reads classified by pol2_states.py as:
  - v2 only paused  (paused_v2>0, paused_v3=0)
  - v3 only paused  (paused_v2=0, paused_v3>0)
  - Both paused     (paused_v2>0, paused_v3>0)

render a zoomed view around the TSS showing:
  - deamination hits (tick marks) at each C→T / G→A position
  - v2 footprint track (fp_v2+)
  - v3 nuc track (nuc+)
  - v3 TF track (tf+)
  - TSS line and paused-Pol-II window (TSS+10 to +50, strand-flipped)

Usage:
  python snapshot_paused.py \
      --pol2-tsv data/dddb_spacetime_NOMERGE_pol2_states.tsv \
      --in-bam  iter17_calls/1.5-2.called.bam [--in-bam ...] \
      --out-dir figures/snapshots/ \
      --label dddb \
      --per-category 10
"""

from __future__ import annotations

import argparse
import gzip
import os
import sys
from collections import defaultdict

import numpy as np
import pysam
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from parse_ma_calls import parse_ma


PAUSED_POS = (10, 50)  # TSS+10 to TSS+50 on + strand (flipped for -)
PAUSED_SIZE = (35, 65)
ZOOM_FLANK = 500  # query-bp around TSS to show


def open_pol2(path):
    if path.endswith('.gz'):
        return gzip.open(path, 'rt')
    return open(path, 'r')


def load_pol2_rows(tsv_path):
    rows = []
    with open_pol2(tsv_path) as f:
        header = f.readline().rstrip('\n').split('\t')
        for line in f:
            vals = line.rstrip('\n').split('\t')
            if len(vals) < len(header):
                continue
            row = dict(zip(header, vals))
            rows.append(row)
    return rows


def load_bam_indices(bam_paths):
    """Build read_name → (bam_path, idx) index across all BAMs."""
    idx = {}
    for bp in bam_paths:
        print(f'Indexing {bp}...', flush=True)
        bam = pysam.AlignmentFile(bp, 'rb', check_sq=False)
        for r in bam.fetch(until_eof=True):
            if r.is_unmapped or r.is_secondary or r.is_supplementary:
                continue
            idx.setdefault(r.query_name, bp)
        bam.close()
    print(f'  indexed {len(idx)} unique reads across {len(bam_paths)} BAMs')
    return idx


def get_read_data(bam_path, read_name):
    """Fetch one read by name from a BAM. Linear scan — caller should
    use this only for a small number of reads."""
    bam = pysam.AlignmentFile(bam_path, 'rb', check_sq=False)
    for r in bam.fetch(until_eof=True):
        if r.is_unmapped or r.is_secondary or r.is_supplementary:
            continue
        if r.query_name == read_name:
            bam.close()
            return r
    bam.close()
    return None


def ref_to_query_pos(read, ref_pos):
    try:
        pairs = read.get_aligned_pairs(matches_only=True)
    except ValueError:
        return None
    for qp, rp in pairs:
        if rp == ref_pos:
            return qp
    return None


def deamination_positions(read, enzyme='daf'):
    """Return list of query positions where a deamination hit is
    called. For DAF reads with IUPAC encoding, looks for Y / R. For
    raw C→T encoding, checks query bases against reference."""
    q = read.query_sequence
    if q is None: return []
    hit_qpos = []
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
                hit_qpos.append(qp)
            elif rbu == 'G' and qb in ('A', 'R'):
                hit_qpos.append(qp)
    return hit_qpos


def draw_snapshot(ax, read, tss_ref, strand, gene_name, category, enzyme='daf'):
    """Render one read snapshot on the given axes."""
    tss_q = ref_to_query_pos(read, tss_ref)
    if tss_q is None:
        ax.text(0.5, 0.5, f'{read.query_name}\nno TSS in alignment',
                ha='center', va='center', transform=ax.transAxes, fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
        return

    win_lo, win_hi = tss_q - ZOOM_FLANK, tss_q + ZOOM_FLANK

    # Parse MA tag
    if not read.has_tag('MA'):
        ax.text(0.5, 0.5, f'{read.query_name}\nno MA tag',
                ha='center', va='center', transform=ax.transAxes, fontsize=8)
        return
    ma = parse_ma(read.get_tag('MA'))

    # Get hits
    hits = deamination_positions(read, enzyme=enzyme)

    # Tracks (y-position)
    Y_HITS = 0.0
    Y_V2 = 1.0
    Y_V3_TF = 2.0
    Y_V3_NUC = 3.0

    # Plot hit ticks
    for h in hits:
        if win_lo <= h <= win_hi:
            ax.vlines(h - tss_q, Y_HITS - 0.25, Y_HITS + 0.25,
                       color='#475569', linewidth=0.6, alpha=0.7)

    # v2 footprints
    for s, l in ma['fp_v2']:
        fp_lo, fp_hi = s - tss_q, (s + l) - tss_q
        if fp_hi < -ZOOM_FLANK or fp_lo > ZOOM_FLANK: continue
        is_nuc_size = l >= 90
        color = '#dc2626' if is_nuc_size else '#f59e0b'
        ax.add_patch(Rectangle((fp_lo, Y_V2 - 0.3), fp_hi - fp_lo, 0.6,
                                   facecolor=color, alpha=0.6,
                                   edgecolor='#7f1d1d', linewidth=0.4))

    # v3 TFs
    for s, l in ma['tf']:
        fp_lo, fp_hi = s - tss_q, (s + l) - tss_q
        if fp_hi < -ZOOM_FLANK or fp_lo > ZOOM_FLANK: continue
        ax.add_patch(Rectangle((fp_lo, Y_V3_TF - 0.3), fp_hi - fp_lo, 0.6,
                                   facecolor='#16a34a', alpha=0.7,
                                   edgecolor='#14532d', linewidth=0.4))

    # v3 nucs
    for s, l in ma['nuc']:
        fp_lo, fp_hi = s - tss_q, (s + l) - tss_q
        if fp_hi < -ZOOM_FLANK or fp_lo > ZOOM_FLANK: continue
        ax.add_patch(Rectangle((fp_lo, Y_V3_NUC - 0.3), fp_hi - fp_lo, 0.6,
                                   facecolor='#1e3a8a', alpha=0.7,
                                   edgecolor='#0f172a', linewidth=0.4))

    # TSS line
    ax.axvline(0, color='red', linestyle='--', linewidth=0.8, alpha=0.6)
    # Paused window (strand-flipped)
    if strand == '+':
        p_lo, p_hi = PAUSED_POS
    else:
        p_lo, p_hi = -PAUSED_POS[1], -PAUSED_POS[0]
    ax.axvspan(p_lo, p_hi, alpha=0.12, color='#f59e0b',
                ymin=0.0, ymax=1.0)

    ax.set_xlim(-ZOOM_FLANK, ZOOM_FLANK)
    ax.set_ylim(-0.5, 3.7)
    ax.set_yticks([Y_HITS, Y_V2, Y_V3_TF, Y_V3_NUC])
    ax.set_yticklabels(['hits', 'v2 fp', 'v3 TF', 'v3 nuc'], fontsize=7)
    ax.set_title(f'{read.query_name}  {gene_name} ({strand})  [{category}]',
                  fontsize=7)
    ax.tick_params(axis='x', labelsize=6)
    ax.grid(alpha=0.2, axis='x')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pol2-tsv', required=True,
                    help='pol2_states.tsv(.gz) with paused_v2/paused_v3 cols')
    ap.add_argument('--in-bam', action='append', required=True,
                    help='source BAM(s) (repeatable)')
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--label', required=True)
    ap.add_argument('--per-category', type=int, default=10)
    ap.add_argument('--enzyme', default='daf', choices=['daf', 'hia5'])
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    rows = load_pol2_rows(args.pol2_tsv)
    print(f'Loaded {len(rows)} pol2 rows')

    # Classify
    v2_only = []
    v3_only = []
    both = []
    for r in rows:
        pv2 = int(r['paused_v2'])
        pv3 = int(r['paused_v3'])
        if pv2 > 0 and pv3 == 0: v2_only.append(r)
        elif pv2 == 0 and pv3 > 0: v3_only.append(r)
        elif pv2 > 0 and pv3 > 0: both.append(r)

    print(f'Paused reads: v2_only={len(v2_only)}  v3_only={len(v3_only)}  '
          f'both={len(both)}')

    # Build a read_name → bam_path index so we know where to look
    bam_index = load_bam_indices(args.in_bam)

    # For each category, pick `per_category` unique reads
    import random
    random.seed(42)
    for category, rows_c in [
            ('v2_only', v2_only),
            ('v3_only', v3_only),
            ('both', both)]:
        if not rows_c:
            print(f'[{category}] no reads, skipping')
            continue
        sample = random.sample(rows_c, min(args.per_category, len(rows_c)))
        n_snap = len(sample)
        ncols = 2
        nrows = (n_snap + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols,
                                    figsize=(14, 1.8 * nrows),
                                    squeeze=False)
        for i, r in enumerate(sample):
            ax = axes[i // ncols, i % ncols]
            bam_path = bam_index.get(r['read'])
            if bam_path is None:
                ax.text(0.5, 0.5, f"{r['read']}\nnot in any BAM",
                         ha='center', va='center', transform=ax.transAxes)
                continue
            read_obj = get_read_data(bam_path, r['read'])
            if read_obj is None:
                ax.text(0.5, 0.5, f"{r['read']}\nnot found",
                         ha='center', va='center', transform=ax.transAxes)
                continue
            draw_snapshot(ax, read_obj, int(r['tss_ref']), r['strand'],
                            r['gene'], category, enzyme=args.enzyme)

        # Hide unused axes
        for i in range(len(sample), nrows * ncols):
            axes[i // ncols, i % ncols].axis('off')

        fig.suptitle(
            f'{args.label}: paused Pol II snapshots — {category} '
            f'(n={len(rows_c)} total, showing {n_snap})\n'
            'bottom=hits ticks; v2 fp (red/orange by size); '
            'v3 TF (green); v3 nuc (blue); dashed red line = TSS; '
            'yellow band = paused window (TSS+10..+50)',
            fontsize=10)
        fig.tight_layout()
        png = os.path.join(args.out_dir,
                            f'{args.label}_paused_{category}.png')
        fig.savefig(png, dpi=110, bbox_inches='tight')
        plt.close(fig)
        print(f'Wrote {png}')


if __name__ == '__main__':
    main()
