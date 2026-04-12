#!/usr/bin/env python3
"""Plot raw bulk-metaprofile hit density across the full amplicon.

No trough detection, no windowing around dyads — just the raw signal
so we can eyeball what the population-level pattern actually looks
like on each amplicon. This is the diagnostic version of
`measure_penetration.py`: if these metaprofiles don't resemble
MNase-seq signal, the dyad-finding step is not the right approach.

For each --in-bam:
  - pileup per-position hit + opp
  - compute smoothed rate (hits/opp)
  - plot raw hit density and rate across the amplicon

Usage:
  python plot_raw_metaprofiles.py \
      --in-bam napa.bam --label NAPA \
      --in-bam uba1.bam --label UBA1 \
      --out-prefix figures/raw_metaprofiles \
      --enzyme daf
"""

from __future__ import annotations

import argparse
import os
from collections import defaultdict

import numpy as np
import pysam
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def pileup(bam_path, enzyme='daf', max_reads=0, min_mapq=20):
    """Per-ref-position hit + opp across all reads, for the highest-coverage chrom."""
    opp = defaultdict(lambda: defaultdict(int))
    hit = defaultdict(lambda: defaultdict(int))
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
                    opp[chrom][rp] += 1
                    if qb in ('T', 'Y'):
                        hit[chrom][rp] += 1
                elif rbu == 'G':
                    opp[chrom][rp] += 1
                    if qb in ('A', 'R'):
                        hit[chrom][rp] += 1
    bam.close()

    # Pick the chromosome with the highest total opp count (the amplicon
    # is where the depth stacks, not where reads are merely scattered)
    if not opp:
        return None, None, None, None, 0
    best_chrom = max(opp.keys(),
                     key=lambda c: sum(opp[c].values()))
    positions = sorted(opp[best_chrom].keys())
    span_start, span_end = positions[0], positions[-1] + 1
    L = span_end - span_start
    opp_arr = np.zeros(L, dtype=np.int32)
    hit_arr = np.zeros(L, dtype=np.int32)
    for p, c in opp[best_chrom].items():
        opp_arr[p - span_start] = c
    for p, c in hit[best_chrom].items():
        hit_arr[p - span_start] = c

    # Auto-detect the amplicon window: find max-opp position, then
    # extend left/right while smoothed coverage stays ≥ 20% of max.
    opp_smoothed = np.convolve(opp_arr.astype(float),
                                np.ones(100) / 100, mode='same')
    peak_idx = int(np.argmax(opp_smoothed))
    peak_cov = opp_smoothed[peak_idx]
    thresh = 0.2 * peak_cov
    lo = peak_idx
    while lo > 0 and opp_smoothed[lo - 1] >= thresh:
        lo -= 1
    hi = peak_idx
    while hi < L - 1 and opp_smoothed[hi + 1] >= thresh:
        hi += 1
    # Pad 100 bp on each side so we see the flank too
    lo = max(0, lo - 100)
    hi = min(L, hi + 100)
    opp_arr = opp_arr[lo:hi]
    hit_arr = hit_arr[lo:hi]
    amp_start = span_start + lo
    return best_chrom, amp_start, opp_arr, hit_arr, n


def smooth(x, w=10):
    k = np.ones(w) / w
    return np.convolve(x, k, mode='same')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in-bam', action='append', required=True)
    ap.add_argument('--label', action='append', default=None)
    ap.add_argument('--out-prefix', required=True)
    ap.add_argument('--enzyme', default='daf', choices=['daf', 'hia5'])
    ap.add_argument('--smooth-window', type=int, default=10)
    ap.add_argument('--max-reads', type=int, default=0)
    args = ap.parse_args()

    labels = args.label or [os.path.basename(b).replace('.bam', '') for b in args.in_bam]
    if len(labels) != len(args.in_bam):
        labels = [os.path.basename(b).replace('.bam', '') for b in args.in_bam]

    datasets = []
    for bam, lbl in zip(args.in_bam, labels):
        if not os.path.exists(bam):
            print(f'[{lbl}] not found: {bam}')
            continue
        print(f'[{lbl}] piling {bam}...')
        chrom, start, opp_arr, hit_arr, n = pileup(
            bam, enzyme=args.enzyme, max_reads=args.max_reads)
        if chrom is None:
            print(f'  → no data')
            continue
        print(f'  → {n} reads; {chrom}:{start}-{start+len(opp_arr)} '
              f'({len(opp_arr)} bp); max opp={opp_arr.max()}, max hit={hit_arr.max()}')
        datasets.append((lbl, chrom, start, opp_arr, hit_arr, n))

    if not datasets:
        print('no data'); return

    # Plot: two panels per dataset — raw hit density, and rate (hit/opp)
    n_ds = len(datasets)
    fig, axes = plt.subplots(n_ds, 2, figsize=(14, 2.8 * n_ds), sharex=False)
    if n_ds == 1:
        axes = axes.reshape(1, 2)

    for i, (lbl, chrom, start, opp_arr, hit_arr, n_reads) in enumerate(datasets):
        # Already trimmed to the amplicon window in pileup()
        opp = opp_arr
        hit = hit_arr
        lo, hi = 0, len(opp_arr)
        xs = np.arange(lo, hi) + start
        hit_sm = smooth(hit.astype(float), args.smooth_window)
        # Rate: hits per opp, only where opp > 0
        rate = np.zeros_like(opp, dtype=float)
        mask = opp > 0
        rate[mask] = hit[mask] / opp[mask]
        rate_sm = smooth(rate, args.smooth_window)

        ax_h, ax_r = axes[i, 0], axes[i, 1]

        ax_h.plot(xs, hit_sm, color='#1e3a8a', linewidth=0.9)
        ax_h.fill_between(xs, 0, hit_sm, color='#1e3a8a', alpha=0.25)
        ax_h.set_title(f'{lbl}  [{chrom}:{start+lo:,}-{start+hi:,}]  '
                       f'({n_reads} reads, max opp={opp.max()})', fontsize=10)
        ax_h.set_ylabel('hit density\n(smoothed count)', fontsize=9)
        ax_h.grid(alpha=0.3)

        ax_r.plot(xs, rate_sm, color='#dc2626', linewidth=0.9)
        ax_r.fill_between(xs, 0, rate_sm, color='#dc2626', alpha=0.25)
        ax_r.set_title(f'{lbl}  deamination rate (hits/opp)', fontsize=10)
        ax_r.set_ylabel('hit / opp', fontsize=9)
        ax_r.grid(alpha=0.3)
        ax_r.set_ylim(0, max(0.05, rate_sm.max() * 1.1))

        if i == n_ds - 1:
            ax_h.set_xlabel('ref position (bp)')
            ax_r.set_xlabel('ref position (bp)')

    fig.tight_layout()
    png = args.out_prefix + '.png'
    fig.savefig(png, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'Wrote {png}')


if __name__ == '__main__':
    main()
