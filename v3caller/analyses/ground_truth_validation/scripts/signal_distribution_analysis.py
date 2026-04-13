#!/usr/bin/env python3
"""Deeper look at ChIP signal distribution per call category.

Key ask: among v3_only calls (which are 20-40x more numerous than
v2_only), what fraction are at "real" binding positions?

Definition of "real": signal > P95 of random positions on the same
chromosome (sampled 10k random positions matched to amplicon
regions).

Reports:
  - Per-category histograms of signal values (log-scale)
  - "Hit rate": fraction of calls above the random-P95 threshold
  - Absolute count of "hits" per category (v3_only × hit_rate)
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pyBigWig
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_calls(bed_path):
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
                'center': (int(parts[1]) + int(parts[2])) // 2,
                'size': int(parts[4]),
                'category': parts[6],
                'call_type': parts[7],
            })
    return calls


def sample_random_positions(calls, n=10000, seed=42):
    """Sample random positions within the same chroms/ranges as the
    caller output. Keeps call-size distribution matched."""
    np.random.seed(seed)
    # Group calls by chrom to get ranges
    per_chrom = {}
    for c in calls:
        per_chrom.setdefault(c['chrom'], []).append(c['center'])
    random_pos = []
    chroms = list(per_chrom.keys())
    for _ in range(n):
        chrom = chroms[np.random.randint(0, len(chroms))]
        lo = min(per_chrom[chrom])
        hi = max(per_chrom[chrom])
        random_pos.append((chrom, np.random.randint(lo, hi + 1)))
    return random_pos


def mean_signal_at(bw, chrom, center, flank):
    try:
        v = bw.stats(chrom, center - flank, center + flank + 1, type='mean')
    except (RuntimeError, ValueError):
        return float('nan')
    if v is None or len(v) == 0 or v[0] is None:
        return float('nan')
    return float(v[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--calls-bed', required=True)
    ap.add_argument('--bigwig', action='append', required=True)
    ap.add_argument('--out-prefix', required=True)
    ap.add_argument('--flank', type=int, default=50)
    ap.add_argument('--n-random', type=int, default=10000)
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out_prefix) or '.', exist_ok=True)

    calls = load_calls(args.calls_bed)
    print(f'Loaded {len(calls):,} calls')

    random_pos = sample_random_positions(calls, n=args.n_random)

    fig_rows = len(args.bigwig)
    fig, axes = plt.subplots(fig_rows, 2, figsize=(14, 3.5 * fig_rows),
                                squeeze=False)

    summary_rows = []
    for i, bw_path in enumerate(args.bigwig):
        name = os.path.basename(bw_path).replace('.bw', '')
        print(f'\n=== {name} ===', flush=True)
        bw = pyBigWig.open(bw_path)

        # Signal at random positions → threshold
        rand_sigs = []
        for chrom, pos in random_pos:
            s = mean_signal_at(bw, chrom, pos, args.flank)
            if not np.isnan(s):
                rand_sigs.append(s)
        rand_sigs = np.array(rand_sigs)
        thresh = np.percentile(rand_sigs, 95)
        print(f'  random P50/P90/P95/P99: {np.percentile(rand_sigs, 50):.3f} / '
              f'{np.percentile(rand_sigs, 90):.3f} / {thresh:.3f} / '
              f'{np.percentile(rand_sigs, 99):.3f}')

        # Signal per category
        per_cat = {}
        for c in calls:
            s = mean_signal_at(bw, c['chrom'], c['center'], args.flank)
            if np.isnan(s): continue
            key = (c['call_type'], c['category'])
            per_cat.setdefault(key, []).append(s)

        # Hit rate above P95-random
        print(f'  hit rate (>P95 random) per category:')
        for (ct, cat), vals in sorted(per_cat.items()):
            arr = np.array(vals)
            hit_rate = (arr > thresh).mean()
            n_hits = int((arr > thresh).sum())
            fold = hit_rate / 0.05 if 0.05 > 0 else float('nan')
            print(f'    {ct} / {cat:<10s}: n={len(arr):>6,}  '
                  f'hits={n_hits:>5,}  hit_rate={hit_rate*100:.1f}%  '
                  f'({fold:.1f}× random)')
            summary_rows.append({
                'bigwig': name, 'call_type': ct, 'category': cat,
                'n_calls': int(len(arr)), 'n_hits_above_rand_p95': n_hits,
                'hit_rate_pct': round(hit_rate * 100, 2),
                'fold_over_random': round(fold, 2),
                'rand_p95_threshold': round(thresh, 4),
            })

        # Left: histogram of signal per category
        ax = axes[i, 0]
        bins = np.linspace(-1, 3, 80)
        for (ct, cat), vals in sorted(per_cat.items()):
            if cat == 'weak': continue
            if ct == 'v2_tf' and cat not in ('v2_only', 'shared'): continue
            if ct == 'v3_tf' and cat not in ('v3_only', 'shared'): continue
            vals_arr = np.array(vals)
            label = f'{ct}/{cat} (n={len(vals_arr):,})'
            ax.hist(vals_arr, bins=bins, alpha=0.5, density=True,
                     label=label)
        ax.axvline(thresh, color='k', linestyle='--', label='random P95')
        ax.set_xlabel(f'{name} mean signal (±{args.flank} bp)')
        ax.set_ylabel('density')
        ax.set_title(f'{name}: signal distribution per category')
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

        # Right: bar chart of hit count & hit rate
        ax = axes[i, 1]
        cats_order = [('v2_tf', 'v2_only'), ('v2_tf', 'shared'),
                       ('v3_tf', 'shared'), ('v3_tf', 'v3_only')]
        labels_p = []
        n_hits_p = []
        rates_p = []
        for key in cats_order:
            if key not in per_cat: continue
            arr = np.array(per_cat[key])
            n_hits_p.append(int((arr > thresh).sum()))
            rates_p.append(100 * (arr > thresh).mean())
            labels_p.append(f'{key[0]}\n{key[1]}')
        x = np.arange(len(labels_p))
        ax.bar(x, n_hits_p, color=['#dc2626', '#16a34a', '#16a34a', '#1e3a8a'])
        ax.set_xticks(x); ax.set_xticklabels(labels_p, fontsize=8)
        ax.set_ylabel(f'# calls > random-P95')
        ax2 = ax.twinx()
        ax2.plot(x, rates_p, 'ko-')
        ax2.set_ylabel('hit rate (%)')
        ax.set_title(f'{name}: hits above random-P95 threshold')
        for i_, (c, n, r) in enumerate(zip(x, n_hits_p, rates_p)):
            ax.text(c, n + max(n_hits_p)*0.02, f'{n:,}\n{r:.0f}%',
                     ha='center', fontsize=7)

        bw.close()

    fig.tight_layout()
    png = args.out_prefix + '_distribution.png'
    fig.savefig(png, dpi=130, bbox_inches='tight'); plt.close(fig)
    print(f'\nWrote {png}')

    tsv = args.out_prefix + '_hitrate_summary.tsv'
    with open(tsv, 'w') as f:
        f.write('bigwig\tcall_type\tcategory\tn_calls\tn_hits_above_rand_p95\t'
                 'hit_rate_pct\tfold_over_random\trand_p95_threshold\n')
        for row in summary_rows:
            f.write('\t'.join(str(row[k]) for k in
                                 ['bigwig', 'call_type', 'category',
                                  'n_calls', 'n_hits_above_rand_p95',
                                  'hit_rate_pct', 'fold_over_random',
                                  'rand_p95_threshold']) + '\n')
    print(f'Wrote {tsv}')


if __name__ == '__main__':
    main()
