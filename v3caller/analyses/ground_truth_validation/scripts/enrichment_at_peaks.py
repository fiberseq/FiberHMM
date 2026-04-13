#!/usr/bin/env python3
"""Measure ChIP / PRO-seq signal at projected v2/v3 call positions.

For each bigwig, at each call's ±flank window, compute mean signal.
Compare distributions across categories (v3_only, v2_only, shared).

If v3 is correctly exposing TFs that v2 folded into nucs:
  → v3_only TFs will show HIGHER ChIP signal at pioneer factors
    (zld, gaf, twi, dl, bcd, cad) than v2_only TFs.
If v3 is just overcalling noise:
  → v3_only will show LOWER or baseline signal.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pyBigWig
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def mean_signal_at(bw, chrom, start, end):
    """Mean signal in [start, end). Returns np.nan if off-chromosome
    or no values."""
    try:
        v = bw.stats(chrom, start, end, type='mean')
    except (RuntimeError, ValueError):
        return float('nan')
    if v is None or len(v) == 0 or v[0] is None:
        return float('nan')
    return float(v[0])


def load_calls(bed_path):
    """Read projected-calls BED. Returns list of dicts."""
    calls = []
    with open(bed_path) as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.rstrip('\n').split('\t')
            if len(parts) < 10:
                continue
            calls.append({
                'chrom': parts[0],
                'start': int(parts[1]),
                'end': int(parts[2]),
                'call_id': parts[3],
                'size': int(parts[4]),
                'strand': parts[5],
                'category': parts[6],
                'call_type': parts[7],
                'read': parts[8],
                'iou': float(parts[9]),
            })
    return calls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--calls-bed', required=True)
    ap.add_argument('--bigwig', action='append', required=True,
                    help='Path to bigwig file (repeatable); name inferred '
                         'from basename')
    ap.add_argument('--out-prefix', required=True)
    ap.add_argument('--flank', type=int, default=100,
                    help='±bp around call center to average signal')
    ap.add_argument('--max-calls', type=int, default=0,
                    help='cap number of calls processed (0 = all)')
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_prefix) or '.', exist_ok=True)

    calls = load_calls(args.calls_bed)
    if args.max_calls:
        calls = calls[:args.max_calls]
    print(f'Loaded {len(calls):,} calls')
    cat_counts = {}
    for c in calls:
        key = (c['call_type'], c['category'])
        cat_counts[key] = cat_counts.get(key, 0) + 1
    for k, v in sorted(cat_counts.items()):
        print(f'  {k}: {v:,}')

    # Open all bigwigs
    bigwigs = {}
    for bw_path in args.bigwig:
        name = os.path.basename(bw_path).replace('.bw', '')
        bigwigs[name] = pyBigWig.open(bw_path)

    # Compute signal per call per bigwig
    # Structure: signal[bw_name][(call_type, category)] = list of mean signals
    signal = {bw: {} for bw in bigwigs}
    for i, c in enumerate(calls):
        if i % 50000 == 0 and i > 0:
            print(f'  processed {i:,} / {len(calls):,}', flush=True)
        center = (c['start'] + c['end']) // 2
        lo = center - args.flank
        hi = center + args.flank + 1
        for bw_name, bw in bigwigs.items():
            s = mean_signal_at(bw, c['chrom'], lo, hi)
            key = (c['call_type'], c['category'])
            signal[bw_name].setdefault(key, []).append(s)

    for bw in bigwigs.values():
        bw.close()

    # Summary TSV
    tsv_path = args.out_prefix + '_signal_summary.tsv'
    with open(tsv_path, 'w') as f:
        f.write('bigwig\tcall_type\tcategory\tn_calls\tmean_signal\t'
                 'median_signal\tq25\tq75\tn_nan\n')
        for bw_name, per_cat in signal.items():
            for (ct, cat), vals in sorted(per_cat.items()):
                arr = np.array(vals)
                nan_mask = np.isnan(arr)
                nn = arr[~nan_mask]
                if len(nn) == 0:
                    f.write(f'{bw_name}\t{ct}\t{cat}\t{len(arr)}\tnan\tnan\tnan\tnan\t{nan_mask.sum()}\n')
                    continue
                f.write(f'{bw_name}\t{ct}\t{cat}\t{len(arr)}\t'
                         f'{nn.mean():.4f}\t{np.median(nn):.4f}\t'
                         f'{np.percentile(nn,25):.4f}\t{np.percentile(nn,75):.4f}\t'
                         f'{nan_mask.sum()}\n')
    print(f'\nWrote {tsv_path}')

    # Figure: box/strip for each bigwig showing v3_only / v2_only / shared
    n_bw = len(bigwigs)
    ncols = 3
    nrows = (n_bw + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows),
                                squeeze=False)
    category_order = ['v3_only', 'shared', 'weak', 'v2_only']
    colors = {'v3_only': '#1e3a8a', 'shared': '#16a34a',
               'weak': '#94a3b8', 'v2_only': '#dc2626'}

    for idx, (bw_name, per_cat) in enumerate(sorted(signal.items())):
        ax = axes[idx // ncols, idx % ncols]
        # Group v3_tf per category, v2_tf per category
        data = []
        labels = []
        bar_colors = []
        for ct in ('v3_tf', 'v2_tf'):
            for cat in category_order:
                k = (ct, cat)
                if k not in per_cat:
                    continue
                vals = np.array(per_cat[k])
                vals = vals[~np.isnan(vals)]
                if len(vals) < 10:
                    continue
                data.append(vals)
                labels.append(f'{ct}\n{cat}\n(n={len(vals)})')
                bar_colors.append(colors.get(cat, '#9ca3af'))

        if not data:
            ax.text(0.5, 0.5, 'no data', ha='center', va='center',
                     transform=ax.transAxes)
            ax.set_title(bw_name, fontsize=10)
            continue

        # Boxplot w/ log y
        bp = ax.boxplot(data, showfliers=False, patch_artist=True, widths=0.6)
        for patch, color in zip(bp['boxes'], bar_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.5)
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels, fontsize=7, rotation=45, ha='right')
        ax.set_yscale('symlog', linthresh=0.1)
        ax.set_ylabel(f'mean signal ±{args.flank} bp')
        ax.set_title(bw_name, fontsize=10)
        ax.grid(alpha=0.3, axis='y')

    for j in range(len(bigwigs), nrows * ncols):
        axes[j // ncols, j % ncols].axis('off')

    fig.suptitle('v2/v3 call enrichment at reference peaks (box: Q1-Q3, whiskers: 1.5*IQR)',
                  fontsize=11)
    fig.tight_layout()
    png = args.out_prefix + '_enrichment.png'
    fig.savefig(png, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'Wrote {png}')


if __name__ == '__main__':
    main()
