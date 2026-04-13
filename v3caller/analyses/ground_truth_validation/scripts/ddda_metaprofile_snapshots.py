#!/usr/bin/env python3
"""Metaprofile + single-read snapshots for DddA HMM output.

For a given BAM (HMM-called with ns/nl tags):
  - Auto-detect the amplicon window (high-coverage chrom region)
  - Compute per-position: bulk hit density, nuc-call density,
    MSP-call density across all reads
  - Plot: 3-row metaprofile (hits / nuc density / MSP density)
    + 10 single-read snapshots zoomed to the amplicon window

Usage:
  python ddda_metaprofile_snapshots.py \
      --in-bam some_ddda_hmm.bam --label napa \
      --out-prefix figures/napa_ddda \
      --enzyme daf --n-snapshots 10
"""
from __future__ import annotations

import argparse, os
from collections import defaultdict
import numpy as np
import pysam
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def pileup_and_calls(bam_path, enzyme='daf', max_reads=0, min_mapq=20):
    """Scan BAM; accumulate per-ref-position:
      - opp count (C/G for DAF, A/T for Hia5)
      - hit count (C→T / G→A / IUPAC Y/R)
      - nuc presence (any read has a ns/nl nuc covering this ref pos)
      - MSP presence (any read has as/al covering this ref pos)
    """
    opp_by_chrom = defaultdict(lambda: defaultdict(int))
    hit_by_chrom = defaultdict(lambda: defaultdict(int))
    nuc_by_chrom = defaultdict(lambda: defaultdict(int))
    msp_by_chrom = defaultdict(lambda: defaultdict(int))

    bam = pysam.AlignmentFile(bam_path, 'rb', check_sq=False,
                                ignore_truncation=True)
    n = 0
    all_reads_meta = []  # for snapshots
    for r in bam.fetch(until_eof=True):
        if r.is_unmapped or r.is_secondary or r.is_supplementary:
            continue
        if r.mapping_quality < min_mapq:
            continue
        n += 1
        if max_reads and n > max_reads:
            break
        q = r.query_sequence
        if q is None: continue
        chrom = r.reference_name
        try:
            pairs = r.get_aligned_pairs(with_seq=True, matches_only=True)
        except ValueError:
            continue
        qp_to_rp = {}
        for qp, rp, rb in pairs:
            if rb is None: continue
            qp_to_rp[qp] = rp
            rbu = rb.upper()
            qb = q[qp].upper()
            if enzyme == 'daf':
                if rbu == 'C':
                    opp_by_chrom[chrom][rp] += 1
                    if qb in ('T', 'Y'):
                        hit_by_chrom[chrom][rp] += 1
                elif rbu == 'G':
                    opp_by_chrom[chrom][rp] += 1
                    if qb in ('A', 'R'):
                        hit_by_chrom[chrom][rp] += 1

        # Nuc calls: ns/nl in query coords → project to ref
        if r.has_tag('ns') and r.has_tag('nl'):
            ns = list(r.get_tag('ns'))
            nl = list(r.get_tag('nl'))
            for s, l in zip(ns, nl):
                # project start + end
                rp_s = qp_to_rp.get(int(s)) or qp_to_rp.get(int(s) + 1)
                rp_e = qp_to_rp.get(int(s + l - 1)) or qp_to_rp.get(int(s + l))
                if rp_s is None or rp_e is None: continue
                lo, hi = sorted((rp_s, rp_e))
                for rp in range(lo, hi + 1):
                    nuc_by_chrom[chrom][rp] += 1
        # MSP: as/al
        if r.has_tag('as') and r.has_tag('al'):
            a_s = list(r.get_tag('as'))
            a_l = list(r.get_tag('al'))
            for s, l in zip(a_s, a_l):
                rp_s = qp_to_rp.get(int(s)) or qp_to_rp.get(int(s) + 1)
                rp_e = qp_to_rp.get(int(s + l - 1)) or qp_to_rp.get(int(s + l))
                if rp_s is None or rp_e is None: continue
                lo, hi = sorted((rp_s, rp_e))
                for rp in range(lo, hi + 1):
                    msp_by_chrom[chrom][rp] += 1

        # Save read for snapshot selection
        all_reads_meta.append({
            'name': r.query_name, 'chrom': chrom,
            'ref_start': r.reference_start, 'ref_end': r.reference_end,
        })

    bam.close()
    return opp_by_chrom, hit_by_chrom, nuc_by_chrom, msp_by_chrom, n, all_reads_meta


def find_amplicon_window(opp_by_chrom):
    """Pick chrom with most total opp, then find the contiguous
    high-coverage region (≥20% of peak smoothed coverage)."""
    best_chrom = max(opp_by_chrom, key=lambda c: sum(opp_by_chrom[c].values()))
    positions = sorted(opp_by_chrom[best_chrom].keys())
    lo, hi = positions[0], positions[-1] + 1
    L = hi - lo
    opp_arr = np.zeros(L, dtype=np.int32)
    for p, c in opp_by_chrom[best_chrom].items():
        opp_arr[p - lo] = c
    opp_sm = np.convolve(opp_arr.astype(float),
                          np.ones(100) / 100, mode='same')
    peak = opp_sm.max()
    thresh = 0.2 * peak
    peak_idx = int(np.argmax(opp_sm))
    a = peak_idx
    while a > 0 and opp_sm[a - 1] >= thresh: a -= 1
    b = peak_idx
    while b < L - 1 and opp_sm[b + 1] >= thresh: b += 1
    return best_chrom, lo + max(0, a - 50), lo + min(L, b + 50)


def plot_metaprofile(ax_h, ax_n, ax_m, chrom, start, end,
                       opp_by, hit_by, nuc_by, msp_by, smooth=10):
    L = end - start
    opp_arr = np.zeros(L); hit_arr = np.zeros(L)
    nuc_arr = np.zeros(L); msp_arr = np.zeros(L)
    for p, c in opp_by[chrom].items():
        if start <= p < end: opp_arr[p - start] = c
    for p, c in hit_by[chrom].items():
        if start <= p < end: hit_arr[p - start] = c
    for p, c in nuc_by[chrom].items():
        if start <= p < end: nuc_arr[p - start] = c
    for p, c in msp_by[chrom].items():
        if start <= p < end: msp_arr[p - start] = c

    k = np.ones(smooth) / smooth
    hit_sm = np.convolve(hit_arr, k, mode='same')
    nuc_sm = np.convolve(nuc_arr, k, mode='same')
    msp_sm = np.convolve(msp_arr, k, mode='same')
    xs = np.arange(start, end)

    ax_h.plot(xs, hit_sm, color='#475569', linewidth=0.8)
    ax_h.fill_between(xs, 0, hit_sm, color='#475569', alpha=0.3)
    ax_h.set_ylabel('hit\ndensity', fontsize=9)
    ax_h.grid(alpha=0.3); ax_h.tick_params(labelbottom=False)

    ax_n.plot(xs, nuc_sm, color='#1e3a8a', linewidth=0.8)
    ax_n.fill_between(xs, 0, nuc_sm, color='#1e3a8a', alpha=0.35)
    ax_n.set_ylabel('nuc\ncalls/pos', fontsize=9)
    ax_n.grid(alpha=0.3); ax_n.tick_params(labelbottom=False)

    ax_m.plot(xs, msp_sm, color='#16a34a', linewidth=0.8)
    ax_m.fill_between(xs, 0, msp_sm, color='#16a34a', alpha=0.35)
    ax_m.set_ylabel('MSP\ncalls/pos', fontsize=9)
    ax_m.grid(alpha=0.3)
    ax_m.set_xlabel(f'{chrom} ref position (bp)')


def plot_snapshot(ax, read, chrom, win_start, win_end, enzyme='daf'):
    q = read.query_sequence
    if q is None: return
    try:
        pairs = read.get_aligned_pairs(with_seq=True, matches_only=True)
    except ValueError:
        return
    qp_to_rp = {}
    hit_ref = []
    for qp, rp, rb in pairs:
        if rb is None: continue
        qp_to_rp[qp] = rp
        if not (win_start <= rp < win_end): continue
        rbu = rb.upper()
        qb = q[qp].upper()
        if enzyme == 'daf':
            if (rbu == 'C' and qb in ('T', 'Y')) or \
               (rbu == 'G' and qb in ('A', 'R')):
                hit_ref.append(rp)

    # Tracks
    Y_HITS = 0.0; Y_MSP = 1.0; Y_NUC = 2.0
    for rp in hit_ref:
        ax.vlines(rp, Y_HITS - 0.2, Y_HITS + 0.2,
                   color='#475569', linewidth=0.4, alpha=0.7)

    if read.has_tag('ns') and read.has_tag('nl'):
        ns = list(read.get_tag('ns')); nl = list(read.get_tag('nl'))
        for s, l in zip(ns, nl):
            rp_s = qp_to_rp.get(int(s)) or qp_to_rp.get(int(s) + 1)
            rp_e = qp_to_rp.get(int(s + l - 1)) or qp_to_rp.get(int(s + l))
            if rp_s is None or rp_e is None: continue
            lo, hi = sorted((rp_s, rp_e))
            if hi < win_start or lo > win_end: continue
            color = '#1e3a8a' if l >= 90 else '#f59e0b'
            ax.add_patch(Rectangle((lo, Y_NUC - 0.3), hi - lo, 0.6,
                                       facecolor=color, alpha=0.6,
                                       edgecolor='#0f172a', linewidth=0.3))

    if read.has_tag('as') and read.has_tag('al'):
        a_s = list(read.get_tag('as')); a_l = list(read.get_tag('al'))
        for s, l in zip(a_s, a_l):
            rp_s = qp_to_rp.get(int(s)) or qp_to_rp.get(int(s) + 1)
            rp_e = qp_to_rp.get(int(s + l - 1)) or qp_to_rp.get(int(s + l))
            if rp_s is None or rp_e is None: continue
            lo, hi = sorted((rp_s, rp_e))
            if hi < win_start or lo > win_end: continue
            ax.add_patch(Rectangle((lo, Y_MSP - 0.25), hi - lo, 0.5,
                                       facecolor='#16a34a', alpha=0.55,
                                       edgecolor='#14532d', linewidth=0.3))

    ax.set_xlim(win_start, win_end)
    ax.set_ylim(-0.5, 2.7)
    ax.set_yticks([Y_HITS, Y_MSP, Y_NUC])
    ax.set_yticklabels(['hits', 'MSP', 'nuc'], fontsize=7)
    ax.set_title(read.query_name, fontsize=7)
    ax.tick_params(axis='x', labelsize=6)
    ax.grid(alpha=0.2, axis='x')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in-bam', required=True)
    ap.add_argument('--label', required=True)
    ap.add_argument('--out-prefix', required=True)
    ap.add_argument('--enzyme', default='daf', choices=['daf', 'hia5'])
    ap.add_argument('--n-snapshots', type=int, default=10)
    ap.add_argument('--max-reads', type=int, default=5000)
    ap.add_argument('--smooth', type=int, default=15)
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out_prefix) or '.', exist_ok=True)

    print(f'Scanning {args.in_bam}...', flush=True)
    opp_by, hit_by, nuc_by, msp_by, n_used, all_reads_meta = \
        pileup_and_calls(args.in_bam, enzyme=args.enzyme,
                         max_reads=args.max_reads)
    print(f'  {n_used} reads used')

    chrom, a_start, a_end = find_amplicon_window(opp_by)
    print(f'  amplicon window: {chrom}:{a_start:,}-{a_end:,} '
          f'({a_end - a_start} bp)')

    # --- Metaprofile ---
    fig, (ax_h, ax_n, ax_m) = plt.subplots(3, 1, figsize=(14, 6),
                                               sharex=True)
    plot_metaprofile(ax_h, ax_n, ax_m, chrom, a_start, a_end,
                       opp_by, hit_by, nuc_by, msp_by, smooth=args.smooth)
    fig.suptitle(f'{args.label} — {chrom}:{a_start:,}-{a_end:,} '
                  f'({n_used} reads)', fontsize=11)
    fig.tight_layout()
    png = args.out_prefix + '_metaprofile.png'
    fig.savefig(png, dpi=130, bbox_inches='tight'); plt.close(fig)
    print(f'Wrote {png}')

    # --- Snapshots ---
    # Pick reads that span most of the amplicon
    good = [r for r in all_reads_meta
            if r['chrom'] == chrom and
               r['ref_start'] <= a_start + 200 and
               r['ref_end'] >= a_end - 200]
    import random; random.seed(42)
    if len(good) > args.n_snapshots:
        good = random.sample(good, args.n_snapshots)

    # Re-open bam, fetch each read
    bam = pysam.AlignmentFile(args.in_bam, 'rb', check_sq=False,
                                ignore_truncation=True)
    read_objs = {}
    for r in bam.fetch(until_eof=True):
        if r.query_name in {g['name'] for g in good} and not r.is_secondary \
                and not r.is_supplementary and not r.is_unmapped:
            read_objs[r.query_name] = r
    bam.close()

    ncols = 1
    nrows = len(good)
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 1.6 * nrows),
                                squeeze=False)
    for i, g in enumerate(good):
        ax = axes[i, 0]
        ro = read_objs.get(g['name'])
        if ro is None: continue
        plot_snapshot(ax, ro, chrom, a_start, a_end, enzyme=args.enzyme)
    fig.suptitle(f'{args.label} single-read snapshots (zoom: amplicon window)',
                  fontsize=10)
    fig.tight_layout()
    png = args.out_prefix + '_snapshots.png'
    fig.savefig(png, dpi=120, bbox_inches='tight'); plt.close(fig)
    print(f'Wrote {png}')


if __name__ == '__main__':
    main()
