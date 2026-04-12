#!/usr/bin/env python3
"""Sweep the flank-rate filter threshold and plot penetration.

If the penetration estimate is real biology, it should asymptote as
we require stricter linker-open evidence (because we're zeroing in
on reads that definitely have a nuc at the dyad). If it keeps
dropping with threshold, that tells us some of the signal was
positioning dilution, not intrinsic penetration.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import conditional_penetration as cp


def process_with_threshold(bam_path, chrom, dyad, threshold,
                              enzyme='daf', max_reads=20000):
    # Temporarily override
    old = cp.FLANK_MIN_RATE
    cp.FLANK_MIN_RATE = threshold
    try:
        res = cp.process_anchor(bam_path, chrom, dyad, enzyme=enzyme,
                                 max_reads=max_reads)
    finally:
        cp.FLANK_MIN_RATE = old
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in-bam', required=True)
    ap.add_argument('--label', required=True)
    ap.add_argument('--anchor', action='append', required=True)
    ap.add_argument('--out-prefix', required=True)
    ap.add_argument('--enzyme', default='daf')
    ap.add_argument('--max-reads', type=int, default=20000)
    ap.add_argument('--thresholds', type=float, nargs='+',
                    default=[0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35])
    args = ap.parse_args()

    anchors = []
    for a in args.anchor:
        c, p = a.split(':')
        anchors.append((c, int(p)))

    results = {}  # threshold -> list of per-anchor dicts
    for thresh in args.thresholds:
        print(f'\n=== threshold {thresh} ===')
        per_anchor = []
        for chrom, dyad in anchors:
            r = process_with_threshold(args.in_bam, chrom, dyad, thresh,
                                         enzyme=args.enzyme,
                                         max_reads=args.max_reads)
            if r is None:
                continue
            print(f'  {chrom}:{dyad}  wrapped={r["wrapped_reads"]:5d}  '
                  f'core={r["core_rate"]:.4f}  flank={r["flank_rate"]:.4f}  '
                  f'pen={r["penetration"]:.3f}')
            per_anchor.append(r)
        # Pool
        tch = sum(r['core_hits'] for r in per_anchor)
        tco = sum(r['core_opps'] for r in per_anchor)
        tfh = sum(r['flank_hits'] for r in per_anchor)
        tfo = sum(r['flank_opps'] for r in per_anchor)
        tw = sum(r['wrapped_reads'] for r in per_anchor)
        pc = tch / tco if tco > 0 else float('nan')
        pf = tfh / tfo if tfo > 0 else float('nan')
        pp = pc / pf if pf > 0 else float('nan')
        print(f'  POOLED n_wrapped={tw}  core={pc:.4f}  flank={pf:.4f}  pen={pp:.3f}')
        results[thresh] = {
            'per_anchor': per_anchor,
            'pooled_core': pc, 'pooled_flank': pf, 'pooled_pen': pp,
            'n_wrapped': tw,
        }

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    threshs = sorted(results.keys())
    pooled_pens = [results[t]['pooled_pen'] for t in threshs]
    n_wraps = [results[t]['n_wrapped'] for t in threshs]

    ax1.plot(threshs, pooled_pens, 'o-', color='#1e3a8a',
              linewidth=2, markersize=9, label='pooled')
    for i, (c, d) in enumerate(anchors):
        pens = [results[t]['per_anchor'][i]['penetration']
                for t in threshs if i < len(results[t]['per_anchor'])]
        ax1.plot(threshs[:len(pens)], pens, '.--', alpha=0.5,
                  label=f'{c}:{d:,}')
    ax1.axhline(0.15, color='#16a34a', linestyle=':', alpha=0.6,
                 label='biophys prior (0.15)')
    ax1.axhline(0.10, color='#f59e0b', linestyle=':', alpha=0.5)
    ax1.set_xlabel('flank hit-rate threshold')
    ax1.set_ylabel('penetration (core / flank)')
    ax1.set_title(f'{args.label}: penetration vs flank-open strictness',
                   fontsize=10)
    ax1.legend(fontsize=8); ax1.grid(alpha=0.3)
    ax1.set_ylim(0, 1.0)

    ax2.plot(threshs, n_wraps, 'o-', color='#dc2626',
              linewidth=2, markersize=9)
    ax2.set_xlabel('flank hit-rate threshold')
    ax2.set_ylabel('wrapped reads (pooled)')
    ax2.set_title('filter yield', fontsize=10)
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    png = args.out_prefix + '.png'
    fig.savefig(png, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'\nWrote {png}')

    # JSON
    sweep_summary = {
        'label': args.label,
        'anchors': [f'{c}:{p}' for c, p in anchors],
        'thresholds': [float(t) for t in threshs],
        'pooled_penetration': [float(results[t]['pooled_pen'])
                                   for t in threshs],
        'n_wrapped': [int(results[t]['n_wrapped']) for t in threshs],
    }
    jsn = args.out_prefix + '_summary.json'
    with open(jsn, 'w') as f:
        json.dump(sweep_summary, f, indent=2)
    print(f'Wrote {jsn}')


if __name__ == '__main__':
    main()
