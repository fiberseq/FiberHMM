#!/usr/bin/env python3
"""WT vs Dl- functional validation.

Dorsal (Dl) is the maternal NF-kB ortholog that activates ventral
fate specifiers. In dl- embryos, its direct targets (sna, twi) should
NOT be transcriptionally engaged, while non-targets (eve, ftz —
pair-rule, driven by different pathways) should be unchanged.

Loads pol2_states TSVs from WT (spacetime 2-3hr) and Dl- (2.5-3.5hr)
and computes per-gene rates of:
  - any elongating Pol II call
  - any paused Pol II call
  - hyperburst state (<50% nuc coverage in gene body)
  - accessible promoter
  - PIC

Compares WT vs Dl- per gene. Expectation:
  sna:  WT > Dl-  (Dl target)
  eve:  WT ≈ Dl-  (non-target)
  ftz:  WT ≈ Dl-  (non-target)

If v3's state calls reflect real biology, we should see the above
pattern. If not, v3 may be calling non-specific signals.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def open_maybe_gz(path):
    return gzip.open(path, 'rt') if path.endswith('.gz') else open(path, 'r')


def load_pol2_tsv(path, caller='v3'):
    """caller: 'v2' or 'v3' selects which columns to read."""
    rows = []
    with open_maybe_gz(path) as f:
        header = f.readline().rstrip('\n').split('\t')
        for line in f:
            vals = line.rstrip('\n').split('\t')
            if len(vals) < len(header): continue
            d = dict(zip(header, vals))
            rows.append({
                'gene': d['gene'],
                'paused': int(d[f'paused_{caller}']),
                'elong': int(d[f'elong_{caller}']),
                'pic': int(d[f'pic_{caller}']),
                'accessible': int(d[f'accessible_{caller}']),
                'hyperburst': int(d[f'hyperburst_{caller}']),
            })
    return rows


def summarize_per_gene(rows):
    """gene → {state: fraction of reads with >0}"""
    per_gene = {}
    by_gene = {}
    for r in rows:
        by_gene.setdefault(r['gene'], []).append(r)
    for gene, grows in by_gene.items():
        n = len(grows)
        per_gene[gene] = {
            'n_reads': n,
            'paused_frac': sum(1 for r in grows if r['paused'] > 0) / n,
            'elong_frac': sum(1 for r in grows if r['elong'] > 0) / n,
            'pic_frac': sum(1 for r in grows if r['pic'] > 0) / n,
            'accessible_frac': sum(1 for r in grows if r['accessible'] > 0) / n,
            'hyperburst_frac': sum(1 for r in grows if r['hyperburst'] > 0) / n,
        }
    return per_gene


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--wt-tsv', required=True,
                    help='pol2_states.tsv(.gz) for WT reads')
    ap.add_argument('--dl-tsv', required=True,
                    help='pol2_states.tsv(.gz) for Dl- reads')
    ap.add_argument('--out-prefix', required=True)
    ap.add_argument('--wt-label', default='WT')
    ap.add_argument('--dl-label', default='Dl-')
    ap.add_argument('--caller', default='v3', choices=['v2', 'v3'])
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out_prefix) or '.', exist_ok=True)

    caller = getattr(args, 'caller', 'v3')
    wt_rows = load_pol2_tsv(args.wt_tsv, caller=caller)
    dl_rows = load_pol2_tsv(args.dl_tsv, caller=caller)
    print(f'{args.wt_label}: {len(wt_rows)} pol2 rows')
    print(f'{args.dl_label}: {len(dl_rows)} pol2 rows')

    wt_by_gene = summarize_per_gene(wt_rows)
    dl_by_gene = summarize_per_gene(dl_rows)

    states = ['accessible', 'paused', 'pic', 'elong', 'hyperburst']
    genes = sorted(set(wt_by_gene) | set(dl_by_gene))

    # Expected Dl dependence per gene
    dl_target = {'sna': True, 'twi': True,
                  'eve': False, 'ftz': False, 'hb': False,
                  'zen': False, 'chrb': False, 'tll': False}

    # --- Print table ---
    print('\n=== Per-gene state fractions ===')
    print(f'{"gene":<6s} {"state":<12s} {"WT%":>7s} {"Dl-%":>7s} '
          f'{"delta":>7s} {"expected":>10s}')
    summary = {'wt': wt_by_gene, 'dl': dl_by_gene, 'delta': {}}
    for gene in genes:
        summary['delta'][gene] = {}
        for s in states:
            wt_frac = wt_by_gene.get(gene, {}).get(s + '_frac', 0) * 100
            dl_frac = dl_by_gene.get(gene, {}).get(s + '_frac', 0) * 100
            delta = dl_frac - wt_frac
            expected = 'drop' if dl_target.get(gene, False) else 'unchg'
            print(f'{gene:<6s} {s:<12s} {wt_frac:>6.1f}% {dl_frac:>6.1f}% '
                  f'{delta:+6.1f}% {expected:>10s}')
            summary['delta'][gene][s] = {'wt_pct': wt_frac,
                                           'dl_pct': dl_frac,
                                           'delta_pct': delta}

    # --- Figure ---
    fig, axes = plt.subplots(1, len(states), figsize=(4 * len(states), 5),
                                squeeze=False)
    for i, s in enumerate(states):
        ax = axes[0, i]
        wt_v = [wt_by_gene.get(g, {}).get(s + '_frac', 0) * 100 for g in genes]
        dl_v = [dl_by_gene.get(g, {}).get(s + '_frac', 0) * 100 for g in genes]
        x = np.arange(len(genes))
        w = 0.35
        ax.bar(x - w/2, wt_v, w, color='#1e3a8a', label=args.wt_label, alpha=0.85)
        ax.bar(x + w/2, dl_v, w, color='#dc2626', label=args.dl_label, alpha=0.85)
        # Mark target genes
        for j, g in enumerate(genes):
            if dl_target.get(g, False):
                ax.text(j, max(max(wt_v), max(dl_v)) * 1.05, '★',
                        ha='center', fontsize=12, color='#f59e0b')
        ax.set_xticks(x); ax.set_xticklabels(genes, fontsize=9)
        ax.set_ylabel(f'% reads with {s}' if i == 0 else '')
        ax.set_title(s)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, axis='y')
    fig.suptitle(f'{args.wt_label} vs {args.dl_label} per-gene state fractions\n'
                  '★ = Dl target (expect drop); no marker = non-target',
                  fontsize=11)
    fig.tight_layout()
    png = args.out_prefix + '_dl_minus_validation.png'
    fig.savefig(png, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'\nWrote {png}')

    jsn = args.out_prefix + '_dl_minus_summary.json'
    with open(jsn, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'Wrote {jsn}')


if __name__ == '__main__':
    main()
