"""Pair-correlation stratified by MSP length.

Simpler replacement for the 2D decay heatmap: for each accessible
region (MSP in query coords, from iter-16 as/al tags), bin by its
length, and compute pair-correlation of hits WITHIN each MSP length
bin. Overlay the resulting curves.

Expected pattern:
  - Very short MSPs (≤30 bp): pair-correlation is dominated by the
    edge-effect of the short interval itself, not the helical
    pitch. Amplitude at lag 10 may look huge but it's mostly an
    artifact of the short baseline.
  - Short MSPs (30-80 bp, "linkers"): strongest clean 10 bp peak.
    Both sides pinned rotationally by flanking nucs.
  - Medium MSPs (80-160 bp): weaker 10 bp peak. Middle of the
    region is ~40-80 bp from either nuc.
  - Large MSPs (160-320 bp): weak-to-absent peak.
  - Very large MSPs (≥320 bp): essentially flat — the "NFR core" is
    far from any constraining nuc.

The slope of amplitude vs MSP length gives us the rotational
persistence length in chromatin-unwrapped DNA.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pysam
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))


MAX_LAG = 100
MAX_READS = 4000
MIN_ALIGN = 2000

# MSP length bins (bp) — non-uniform, denser at short end
MSP_BINS = [
    (20, 40,   'MSP 20-40',    '#d97706'),
    (40, 80,   'MSP 40-80',    '#f59e0b'),
    (80, 160,  'MSP 80-160',   '#eab308'),
    (160, 320, 'MSP 160-320',  '#dc2626'),
    (320, 800, 'MSP 320-800',  '#7c2d12'),
]


def daf_hit_opp_query(read):
    try:
        pairs = read.get_aligned_pairs(with_seq=True, matches_only=True)
    except ValueError:
        return None
    seq = read.query_sequence
    if seq is None:
        return None
    c_opp, y_hit, t_hit = [], [], []
    g_opp, r_hit, a_hit = [], [], []
    for qpos, rpos, rb in pairs:
        if rb is None or qpos is None:
            continue
        qb = seq[qpos].upper()
        rbu = rb.upper()
        if rbu == 'C':
            c_opp.append(qpos)
            if qb == 'Y':
                y_hit.append(qpos)
            elif qb == 'T':
                t_hit.append(qpos)
        elif rbu == 'G':
            g_opp.append(qpos)
            if qb == 'R':
                r_hit.append(qpos)
            elif qb == 'A':
                a_hit.append(qpos)
    counts = {
        'CtoY': (c_opp, y_hit), 'CtoT': (c_opp, t_hit),
        'GtoR': (g_opp, r_hit), 'GtoA': (g_opp, a_hit),
    }
    best = max(counts, key=lambda k: len(counts[k][1]))
    opps, hits = counts[best]
    if len(hits) == 0:
        return None
    return np.asarray(opps, dtype=np.int64), np.asarray(hits, dtype=np.int64)


def pair_hist_within(positions, lo, hi, max_lag):
    """Pair histogram of positions restricted to the interval [lo, hi).
    Only pairs where BOTH members fall in the interval.
    """
    # Filter
    mask = (positions >= lo) & (positions < hi)
    pos = positions[mask]
    if pos.size < 2:
        return np.zeros(max_lag + 1, dtype=np.int64)
    out = np.zeros(max_lag + 1, dtype=np.int64)
    for i in range(pos.size - 1):
        j = np.searchsorted(pos, pos[i] + max_lag, side='right')
        if j <= i + 1:
            continue
        ds = pos[i + 1:j] - pos[i]
        if ds.size:
            np.add.at(out, ds, 1)
    return out


def process_bam(bam_path, max_reads=MAX_READS):
    num = {i: np.zeros(MAX_LAG + 1, dtype=np.int64) for i in range(len(MSP_BINS))}
    den = {i: np.zeros(MAX_LAG + 1, dtype=np.int64) for i in range(len(MSP_BINS))}
    total_bp = {i: 0 for i in range(len(MSP_BINS))}
    total_msps = {i: 0 for i in range(len(MSP_BINS))}
    n_reads = 0
    bam = pysam.AlignmentFile(bam_path, 'rb', check_sq=False)
    for read in bam.fetch(until_eof=True):
        if read.is_unmapped or read.is_secondary or read.is_supplementary:
            continue
        if (read.query_alignment_length or 0) < MIN_ALIGN:
            continue
        rp = daf_hit_opp_query(read)
        if rp is None:
            continue
        opps, hits = rp
        try:
            as_ = list(read.get_tag('as'))
            al = list(read.get_tag('al'))
        except KeyError:
            continue
        for s, l in zip(as_, al):
            # Which bin does this MSP fall into?
            bin_i = None
            for i, (lo, hi, _, _) in enumerate(MSP_BINS):
                if lo <= l < hi:
                    bin_i = i
                    break
            if bin_i is None:
                continue
            num[bin_i] += pair_hist_within(hits, s, s + l, MAX_LAG)
            den[bin_i] += pair_hist_within(opps, s, s + l, MAX_LAG)
            total_bp[bin_i] += l
            total_msps[bin_i] += 1
        n_reads += 1
        if n_reads >= max_reads:
            break
    bam.close()
    return n_reads, num, den, total_bp, total_msps


def main():
    bam_path = os.path.join(HERE, 'output', 'bam',
                              'scDAF_PS00758__v8_gapcdf.bam')
    if not os.path.exists(bam_path):
        print(f'missing {bam_path}')
        return

    print(f'reading {bam_path}')
    n_reads, num, den, total_bp, total_msps = process_bam(bam_path)
    print(f'processed {n_reads} reads\n')

    # Compute enrichment per bin. Baseline = mean pair-correlation
    # at the LONGEST lags available for that bin.
    # For short MSPs (e.g. 20-40 bp), long lags are impossible so we
    # use the longest lag each can hold as a de facto baseline. This
    # means a "20-40 bp" bin has peak at 10 normalized by values at
    # lags ~30-38, which is within the same short region. That is
    # deliberately "dampened" — if the bin is too short to see the
    # decay baseline, the peak/trough metric is still meaningful
    # (difference is local to the region) even if the absolute
    # enrichment is skewed.
    norms = {}
    for i, (lo, hi, label, color) in enumerate(MSP_BINS):
        if total_msps[i] == 0:
            continue
        # Baseline window: lags 60-min(MAX_LAG, hi-2)
        baseline_lo = 60
        baseline_hi = min(MAX_LAG, hi - 2)
        if baseline_hi - baseline_lo < 5:
            # Short bin — fall back to its own tail
            baseline_lo = max(15, hi - 25)
            baseline_hi = max(baseline_lo + 5, hi - 2)
        d = den[i]
        pr = np.where(d > 0, num[i] / np.maximum(d, 1), 0.0)
        if baseline_hi <= baseline_lo or baseline_hi >= len(pr):
            continue
        baseline_slice = pr[baseline_lo:baseline_hi + 1]
        baseline_slice = baseline_slice[baseline_slice > 0]
        if baseline_slice.size < 3:
            continue
        baseline = float(np.mean(baseline_slice))
        if baseline <= 0:
            continue
        norms[i] = pr / baseline
        peak10 = norms[i][10]
        trough15 = np.mean(norms[i][15:18])
        print(f'  {label}  ({total_msps[i]} MSPs, {total_bp[i]} bp)  '
              f'baseline [lag {baseline_lo}-{baseline_hi}] '
              f'peak@10={peak10:.3f}  trough@15-17={trough15:.3f}  '
              f'amp={peak10 - trough15:+.3f}')

    # ---- plot ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    ax = axes[0]
    for i, (lo, hi, label, color) in enumerate(MSP_BINS):
        if i not in norms:
            continue
        y = norms[i]
        # Only show lags up to this bin's max (longer lags are impossible)
        last_lag = min(60, hi - 1)
        n = total_msps[i]
        ax.plot(np.arange(len(y))[3:last_lag],
                 y[3:last_lag],
                 color=color, linewidth=1.8,
                 label=f'{label}  (n={n})')
    ax.axhline(1.0, color='k', linewidth=0.6)
    for d in (10.4, 20.8, 31.2, 41.6, 52.0):
        ax.axvline(d, color='#888', linestyle=':', linewidth=0.6)
    ax.set_xlabel('pairwise distance (bp)')
    ax.set_ylabel('enrichment (pair_rate / baseline)')
    ax.set_title('Pair-correlation stratified by MSP length', fontsize=11)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc='best')
    ax.set_xlim(3, 60)

    # Panel B: amplitude (peak@10 minus trough@15-17) vs MSP length
    ax = axes[1]
    mid_lens = []
    amps = []
    for i, (lo, hi, label, color) in enumerate(MSP_BINS):
        if i not in norms:
            continue
        y = norms[i]
        last_lag = min(60, hi - 1)
        if last_lag < 17:
            continue
        peak = y[10]
        trough = np.mean(y[15:min(18, last_lag)])
        mid = 0.5 * (lo + hi)
        mid_lens.append(mid)
        amps.append(peak - trough)
        ax.plot([mid], [peak - trough], 'o', color=color, markersize=14)
        ax.annotate(label.replace('MSP ', ''), (mid, peak - trough),
                     xytext=(5, 5), textcoords='offset points',
                     fontsize=8, color=color)

    ax.plot(mid_lens, amps, '--', color='#888', linewidth=1)
    ax.axhline(0, color='k', linewidth=0.6)
    ax.set_xlabel('MSP length (bp, bin midpoint)')
    ax.set_ylabel('peak@10 − trough@15-17 (amplitude)')
    ax.set_title('Amplitude vs MSP length — decay curve',
                  fontsize=11)
    ax.grid(alpha=0.3)
    ax.set_xscale('log')

    fig.suptitle(
        f'scDAF PS00758 — periodicity stratified by MSP length  '
        f'(n={n_reads} reads)',
        fontsize=12, y=1.02,
    )
    out = os.path.join(HERE, 'output', 'periodicity_by_msp_size.png')
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'\nwrote {out}')


if __name__ == '__main__':
    main()
