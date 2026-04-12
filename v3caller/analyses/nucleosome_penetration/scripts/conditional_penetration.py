#!/usr/bin/env python3
"""Conditional penetration: measure dyad rate ONLY on reads whose own
hit pattern shows the flanking linker is accessible.

Rationale: bulk metaprofile pileup is confounded because at any given
candidate dyad, some reads carry a nucleosome there and others don't
(just linker). Averaging across both inflates the "dyad rate".

Fix: for each anchor, filter reads to those with ≥N hits in BOTH the
left flank (dyad − 150 bp to dyad − 80 bp) and the right flank
(dyad + 80 bp to dyad + 150 bp). This means the linker on both sides
is accessible, so if a nuc is positioned anywhere in the amplicon
it's plausibly at this dyad. We do NOT condition on core rate
(would be circular).

Then compute:
  penetration = (Σ hits in core [-40, +40] across filtered reads) /
                (Σ opps in core)
               / (Σ hits in flanks / Σ opps in flanks)

i.e., the core hit-rate normalized to the linker hit-rate, on the
subset of reads with open linker flanks.

Usage:
  python conditional_penetration.py \
      --in-bam in.bam \
      --label NAME \
      --anchor chr5:34762095 --anchor chr5:34762367 \
      --out-prefix figures/conditional \
      --enzyme daf
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


CORE_HALF = 40       # dyad core: ±40 bp → 80 bp (tighter than canonical 147)
FLANK_LO = 80        # flank starts at ±80 bp
FLANK_HI = 150       # flank ends at ±150 bp
# Require each flank's hit RATE (hits/opps) to exceed this, which is
# a meaningful "linker is open" threshold. 0.15 = 15% deamination,
# comfortably above the 1-2% basal rate but achievable in truly
# accessible linker (expected ~20-30% for PacBio DddA on chromatin).
FLANK_MIN_RATE = 0.15


def count_hits_opps_in_window(qp_by_rp, ref_base_at, q, lo, hi,
                                  enzyme='daf'):
    """qp_by_rp: {ref_pos: query_pos}. ref_base_at: {ref_pos: base}.
    q: read query sequence. Returns (hits, opps) in [lo, hi]."""
    hits = 0
    opps = 0
    for rp in range(lo, hi):
        rb = ref_base_at.get(rp)
        if rb is None:
            continue
        if enzyme == 'daf':
            if rb == 'C':
                opps += 1
                qp = qp_by_rp.get(rp)
                if qp is not None and q[qp] in ('T', 'Y'):
                    hits += 1
            elif rb == 'G':
                opps += 1
                qp = qp_by_rp.get(rp)
                if qp is not None and q[qp] in ('A', 'R'):
                    hits += 1
    return hits, opps


def build_ref_map(bam_path, chrom, lo, hi, max_scan=200):
    """Scan up to max_scan reads in the window and build a
    {ref_pos: ref_base_upper} lookup using with_seq=True. Once we
    have a base for every position we need, stop."""
    bam = pysam.AlignmentFile(bam_path, 'rb', check_sq=False)
    ref_base_at = {}
    needed = set(range(lo, hi))
    n = 0
    for read in bam.fetch(chrom, lo, hi):
        if n >= max_scan:
            break
        if read.is_unmapped or read.is_secondary or read.is_supplementary:
            continue
        n += 1
        try:
            pairs = read.get_aligned_pairs(with_seq=True, matches_only=True)
        except ValueError:
            continue
        for qp, rp, rb in pairs:
            if rb is None or rp not in needed:
                continue
            ref_base_at[rp] = rb.upper()
            needed.discard(rp)
            if not needed:
                break
        if not needed:
            break
    bam.close()
    return ref_base_at


def process_anchor(bam_path, chrom, dyad, enzyme='daf', min_mapq=20,
                    max_reads=0):
    """For one anchor, iterate reads covering it and count
    (hits, opps) in core + flanks. Filter on flanks, accumulate."""
    core_lo, core_hi = dyad - CORE_HALF, dyad + CORE_HALF + 1
    lflank_lo, lflank_hi = dyad - FLANK_HI, dyad - FLANK_LO
    rflank_lo, rflank_hi = dyad + FLANK_LO + 1, dyad + FLANK_HI + 1
    win_lo, win_hi = lflank_lo, rflank_hi

    # Cache reference bases for the window
    ref_base_at = build_ref_map(bam_path, chrom, win_lo, win_hi)

    bam = pysam.AlignmentFile(bam_path, 'rb', check_sq=False)
    tot_reads = 0
    covered_reads = 0
    wrapped_reads = 0
    core_hits = 0
    core_opps = 0
    flank_hits = 0
    flank_opps = 0

    try:
        iterator = bam.fetch(chrom, win_lo, win_hi)
    except ValueError:
        print(f'  {chrom}:{dyad} not in bam'); bam.close(); return None

    for read in iterator:
        tot_reads += 1
        if max_reads and tot_reads > max_reads:
            break
        if read.is_unmapped or read.is_secondary or read.is_supplementary:
            continue
        if read.mapping_quality < min_mapq:
            continue
        if read.reference_start > lflank_lo or read.reference_end < rflank_hi:
            continue
        covered_reads += 1

        q = read.query_sequence
        if q is None:
            continue
        # FAST path — no with_seq, just aligned pairs
        try:
            pairs = read.get_aligned_pairs(matches_only=True)
        except ValueError:
            continue
        qp_by_rp = {rp: qp for qp, rp in pairs}

        lfh, lfo = count_hits_opps_in_window(qp_by_rp, ref_base_at, q,
                                                lflank_lo, lflank_hi, enzyme)
        rfh, rfo = count_hits_opps_in_window(qp_by_rp, ref_base_at, q,
                                                rflank_lo, rflank_hi, enzyme)
        if lfo < 5 or rfo < 5:
            continue  # flanks need ≥5 C/G opportunities to be informative
        if (lfh / lfo) < FLANK_MIN_RATE or (rfh / rfo) < FLANK_MIN_RATE:
            continue
        wrapped_reads += 1

        ch, co = count_hits_opps_in_window(qp_by_rp, ref_base_at, q,
                                              core_lo, core_hi, enzyme)
        core_hits += ch
        core_opps += co
        flank_hits += (lfh + rfh)
        flank_opps += (lfo + rfo)

    bam.close()

    core_rate = core_hits / core_opps if core_opps > 0 else float('nan')
    flank_rate = flank_hits / flank_opps if flank_opps > 0 else float('nan')
    penetration = core_rate / flank_rate if flank_rate > 0 else float('nan')

    return {
        'chrom': chrom, 'dyad': dyad,
        'total_reads': tot_reads, 'covered_reads': covered_reads,
        'wrapped_reads': wrapped_reads,
        'core_hits': core_hits, 'core_opps': core_opps,
        'flank_hits': flank_hits, 'flank_opps': flank_opps,
        'core_rate': core_rate, 'flank_rate': flank_rate,
        'penetration': penetration,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in-bam', required=True)
    ap.add_argument('--label', required=True)
    ap.add_argument('--anchor', action='append', required=True,
                    help='anchor position as chrom:pos (repeatable)')
    ap.add_argument('--out-prefix', required=True)
    ap.add_argument('--enzyme', default='daf', choices=['daf', 'hia5'])
    ap.add_argument('--max-reads', type=int, default=0)
    args = ap.parse_args()

    anchors = []
    for a in args.anchor:
        chrom, pos = a.split(':')
        anchors.append((chrom, int(pos)))
    print(f'[{args.label}] {len(anchors)} anchors')

    results = []
    for chrom, dyad in anchors:
        print(f'\n--- {chrom}:{dyad} ---')
        res = process_anchor(args.in_bam, chrom, dyad, enzyme=args.enzyme,
                              max_reads=args.max_reads)
        if res is None:
            continue
        print(f'  covered reads:  {res["covered_reads"]}')
        print(f'  wrapped (flanks open):  {res["wrapped_reads"]}')
        print(f'  core:   {res["core_hits"]}/{res["core_opps"]} = {res["core_rate"]:.4f}')
        print(f'  flanks: {res["flank_hits"]}/{res["flank_opps"]} = {res["flank_rate"]:.4f}')
        print(f'  PENETRATION (core/flank):  {res["penetration"]:.3f}')
        results.append(res)

    # Pool across anchors
    tot_core_h = sum(r['core_hits'] for r in results)
    tot_core_o = sum(r['core_opps'] for r in results)
    tot_flank_h = sum(r['flank_hits'] for r in results)
    tot_flank_o = sum(r['flank_opps'] for r in results)
    tot_wrapped = sum(r['wrapped_reads'] for r in results)
    pooled_core = tot_core_h / tot_core_o if tot_core_o > 0 else float('nan')
    pooled_flank = tot_flank_h / tot_flank_o if tot_flank_o > 0 else float('nan')
    pooled_pen = pooled_core / pooled_flank if pooled_flank > 0 else float('nan')

    print(f'\n=== Pooled across {len(results)} anchors ===')
    print(f'  wrapped reads (total):  {tot_wrapped}')
    print(f'  core:   {tot_core_h}/{tot_core_o} = {pooled_core:.4f}')
    print(f'  flanks: {tot_flank_h}/{tot_flank_o} = {pooled_flank:.4f}')
    print(f'  POOLED PENETRATION:  {pooled_pen:.3f}')

    # Plot: per-anchor penetration + wrapped-read counts
    fig, ax = plt.subplots(1, 1, figsize=(8, 4.5))
    xs = np.arange(len(results))
    labels = [f'{r["chrom"]}:\n{r["dyad"]:,}' for r in results]
    pens = [r['penetration'] for r in results]
    ns = [r['wrapped_reads'] for r in results]

    bars = ax.bar(xs, pens, color='#1e3a8a', alpha=0.7)
    for i, (b, n) in enumerate(zip(bars, ns)):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01,
                f'n={n}', ha='center', fontsize=9)
    ax.axhline(pooled_pen, color='#dc2626', linestyle='--',
                label=f'pooled = {pooled_pen:.3f}')
    ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel('penetration (core rate / flank rate)')
    ax.set_title(f'{args.label}:  conditional penetration on flank-open reads\n'
                  f'filter: each flank [{FLANK_LO},{FLANK_HI}] bp '
                  f'hit rate ≥ {FLANK_MIN_RATE}; core = ±{CORE_HALF} bp',
                  fontsize=10)
    ax.set_ylim(0, max(0.5, max(pens) * 1.2 if pens else 0.5))
    ax.legend()
    ax.grid(alpha=0.3, axis='y')

    fig.tight_layout()
    png = args.out_prefix + '.png'
    fig.savefig(png, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'\nWrote {png}')

    # JSON
    summary = {
        'label': args.label,
        'n_anchors': len(results),
        'per_anchor': results,
        'pooled': {
            'core_rate': pooled_core,
            'flank_rate': pooled_flank,
            'penetration': pooled_pen,
            'wrapped_reads': tot_wrapped,
        },
        'params': {
            'core_half': CORE_HALF,
            'flank_lo': FLANK_LO,
            'flank_hi': FLANK_HI,
            'flank_min_rate': FLANK_MIN_RATE,
        },
    }
    jsn = args.out_prefix + '_summary.json'
    with open(jsn, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'Wrote {jsn}')


if __name__ == '__main__':
    main()
