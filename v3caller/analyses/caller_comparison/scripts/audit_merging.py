#!/usr/bin/env python3
"""Audit how often the merge step fires on DddB.

Uses the legacy `mq` tag (min Poisson-interval across any atom-gaps
absorbed into a nuc). mq == 255 → nuc was a single Pass-1 atom, no
merge. mq < 255 → nuc was made from ≥2 atoms fused by the merge step.

Also classifies by nuc length:
  - ≤180 bp: plausibly a single mononucleosome (fly NRL ≈ 180)
  - 181-300: moderate overmerge (dinuc fusion possible)
  - 301-500: likely trinucleosome or TF-bleeding
  - >500: clearly pathological mega-call

Usage:
  python audit_merging.py \
      --in-bam bam1.bam [--in-bam bam2.bam ...] \
      --label dddb --out-dir figures/
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pysam
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in-bam', action='append', required=True)
    ap.add_argument('--label', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--max-reads', type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    nuc_lens = []
    mqs = []
    merged_flag = []  # 1 if mq < 255
    read_count = 0

    for bam_path in args.in_bam:
        print(f'[{bam_path}]', flush=True)
        bam = pysam.AlignmentFile(bam_path, 'rb', check_sq=False)
        for r in bam.fetch(until_eof=True):
            if r.is_unmapped or r.is_secondary or r.is_supplementary:
                continue
            read_count += 1
            if args.max_reads and read_count > args.max_reads:
                break
            if not r.has_tag('nl'):
                continue
            nl = r.get_tag('nl')
            mq = r.get_tag('mq') if r.has_tag('mq') else [255] * len(nl)
            for l, m in zip(nl, mq):
                nuc_lens.append(int(l))
                mqs.append(int(m))
                merged_flag.append(1 if int(m) < 255 else 0)
        bam.close()

    nuc_lens = np.array(nuc_lens)
    mqs = np.array(mqs)
    merged = np.array(merged_flag)

    if len(nuc_lens) == 0:
        print('no nucs found'); return

    # Summary
    n = len(nuc_lens)
    n_merged = int(merged.sum())
    pct_merged = 100 * n_merged / n
    print(f'\n=== Merge audit: {args.label} ===')
    print(f'Total nucs: {n:,}')
    print(f'  mq=255 (pure single atom): {n - n_merged:,} ({100 - pct_merged:.1f}%)')
    print(f'  mq<255 (merged from ≥2 atoms): {n_merged:,} ({pct_merged:.1f}%)')
    print(f'\nSize buckets:')
    buckets = [
        ('≤180 (mono-nuc)', nuc_lens <= 180),
        ('181-300 (di-nuc)', (nuc_lens > 180) & (nuc_lens <= 300)),
        ('301-500 (tri-nuc)', (nuc_lens > 300) & (nuc_lens <= 500)),
        ('>500 (mega)', nuc_lens > 500),
    ]
    for label, mask in buckets:
        n_b = int(mask.sum())
        n_bm = int((mask & (merged == 1)).sum())
        print(f'  {label:<22s} {n_b:>10,}  ({100*n_b/n:4.1f}%)  '
              f'of which merged: {n_bm:>10,} ({100*n_bm/max(1,n_b):4.1f}%)')

    # Figure: size histogram split by merged/single
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    bins = np.arange(80, 601, 10)
    single = nuc_lens[merged == 0]
    fused = nuc_lens[merged == 1]
    ax1.hist(single, bins=bins, color='#1e3a8a', alpha=0.7,
              label=f'single atom (n={len(single):,})')
    ax1.hist(fused, bins=bins, color='#dc2626', alpha=0.7,
              label=f'merged ≥2 atoms (n={len(fused):,})',
              bottom=np.histogram(single, bins=bins)[0])
    ax1.axvline(180, color='k', linestyle='--', alpha=0.4,
                 label='fly NRL (180 bp)')
    ax1.set_xlabel('nuc length (bp)')
    ax1.set_ylabel('count')
    ax1.set_title(f'{args.label}: nuc length by merge status')
    ax1.legend(); ax1.grid(alpha=0.3)

    # mq distribution
    bin_m = np.arange(0, 257, 10)
    ax2.hist(mqs, bins=bin_m, color='#16a34a')
    ax2.axvline(255, color='k', linestyle='--',
                 label='no merge (mq=255)')
    ax2.set_xlabel('merge quality (mq): 0-255')
    ax2.set_ylabel('count')
    ax2.set_title(f'{args.label}: mq distribution\n'
                   'mq=255 = no internal gap; lower = stronger merge required')
    ax2.legend(); ax2.grid(alpha=0.3)

    fig.tight_layout()
    png = os.path.join(args.out_dir, f'{args.label}_merge_audit.png')
    fig.savefig(png, dpi=130, bbox_inches='tight'); plt.close(fig)
    print(f'\nWrote {png}')

    summary = {
        'label': args.label,
        'n_nucs': int(n),
        'n_merged': n_merged,
        'pct_merged': round(pct_merged, 2),
        'size_buckets': {
            label: {'count': int(mask.sum()),
                     'pct': round(100 * int(mask.sum()) / n, 2),
                     'pct_merged': round(100 * int((mask & (merged == 1)).sum())
                                          / max(1, int(mask.sum())), 2)}
            for label, mask in buckets
        },
    }
    jsn = os.path.join(args.out_dir, f'{args.label}_merge_audit.json')
    with open(jsn, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'Wrote {jsn}')


if __name__ == '__main__':
    main()
