"""Measure the rotational-phase decay as a function of distance
from the nearest flanking nucleosome, accounting for the
bidirectional contribution from both sides.

Model:
  A(d_left, d_right) = A0 * [exp(-d_left / tau) + exp(-d_right / tau)]

A hit at position p gets d_left = distance to nearest upstream nuc
boundary (edge of a called nuc), d_right = distance to nearest
downstream nuc boundary. Both are non-negative. Inside a nuc body
they're both 0 (but we skip the nuc body because it's hit-free).

For each pair (i, j) of hits in the same accessible region
(d_left > 0 for both), we tag the pair by the midpoint's
(d_left_mid, d_right_mid) and accumulate pair-correlation
histograms per cell of a coarse (d_left, d_right) grid. Then we
fit a cosine amplitude in each cell, and plot:

  1. A heatmap of amplitude vs (d_left, d_right)
  2. Marginal decay vs d_left alone (averaged over d_right)
  3. Cross-section along the diagonal (d_left == d_right) — this is
     the "short linker" regime where both sides contribute
  4. Cross-section along the edge (d_left small, d_right large) —
     this is the "NFR edge" regime where only one side contributes
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


MAX_LAG = 60
MAX_READS = 4000
MIN_ALIGN = 2000

# Distance bins (bp) to the nearest nuc edge — non-uniform so we
# get more resolution near the edge where the decay matters.
DIST_EDGES = np.array([0, 10, 20, 30, 40, 60, 80, 120, 200, 400])
N_BINS = len(DIST_EDGES) - 1


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


def per_position_nuc_distances(qlen, nucs):
    """For each query position 0..qlen-1, return d_left[pos] and
    d_right[pos] = distances to the nearest upstream / downstream
    nuc boundary. Inside a nuc body BOTH are 0 (sentinel meaning
    "in nuc, skip"). In accessible regions, d_left/d_right > 0.

    nucs is a list of (start, end) query intervals.
    """
    d_left = np.full(qlen, -1, dtype=np.int32)
    d_right = np.full(qlen, -1, dtype=np.int32)

    # Sorted nuc list
    nucs = sorted(nucs)

    # In-nuc mask
    in_nuc = np.zeros(qlen, dtype=bool)
    for s, e in nucs:
        in_nuc[max(0, s):min(qlen, e)] = True

    # Edges are nuc boundaries. For each position NOT in a nuc,
    # find the distance to the nearest boundary on each side.
    # Boundaries are the `e` of any nuc (downstream boundary) for
    # d_left, and the `s` of any nuc (upstream boundary) for d_right.
    #
    # More precisely: for a position p, d_left = p - (last nuc end <= p),
    # d_right = (next nuc start > p) - p.
    nuc_ends = np.array([e for _, e in nucs], dtype=np.int64)
    nuc_starts = np.array([s for s, _ in nucs], dtype=np.int64)

    for p in range(qlen):
        if in_nuc[p]:
            continue
        # Nearest preceding nuc end
        idx = np.searchsorted(nuc_ends, p, side='right')
        if idx > 0:
            d_left[p] = int(p - nuc_ends[idx - 1])
        else:
            d_left[p] = 10_000  # unbounded (read edge)
        # Nearest following nuc start
        idx = np.searchsorted(nuc_starts, p, side='left')
        if idx < len(nuc_starts):
            d_right[p] = int(nuc_starts[idx] - p)
        else:
            d_right[p] = 10_000
    return d_left, d_right


def bin_idx(dist):
    if dist < 0:
        return -1
    idx = np.searchsorted(DIST_EDGES, dist, side='right') - 1
    if idx < 0:
        return 0
    if idx >= N_BINS:
        return N_BINS - 1
    return int(idx)


def accumulate(positions, d_left, d_right, num_or_den, max_lag):
    """Accumulate per-(d_left_bin, d_right_bin, lag) histogram.

    Tag each pair by the midpoint's (d_left, d_right) bin, then
    increment num_or_den[lb, rb, lag] += 1.
    """
    for i in range(positions.size - 1):
        j = np.searchsorted(positions, positions[i] + max_lag,
                             side='right')
        if j <= i + 1:
            continue
        pi = int(positions[i])
        for k in range(i + 1, int(j)):
            pk = int(positions[k])
            lag = pk - pi
            if lag <= 0 or lag > max_lag:
                continue
            mid = (pi + pk) // 2
            if mid < 0 or mid >= d_left.shape[0]:
                continue
            dl = int(d_left[mid])
            dr = int(d_right[mid])
            if dl < 0 or dr < 0:
                continue
            lb = bin_idx(dl)
            rb = bin_idx(dr)
            num_or_den[lb, rb, lag] += 1


def process_bam(bam_path, max_reads=MAX_READS):
    num = np.zeros((N_BINS, N_BINS, MAX_LAG + 1), dtype=np.int64)
    den = np.zeros((N_BINS, N_BINS, MAX_LAG + 1), dtype=np.int64)
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
        if opps.size < 20 or hits.size < 3:
            continue
        try:
            ns = list(read.get_tag('ns'))
            nl = list(read.get_tag('nl'))
        except KeyError:
            continue
        if not ns:
            continue
        nucs = [(s, s + l) for s, l in zip(ns, nl)]
        qlen = read.query_length or 0
        if qlen == 0:
            continue
        d_left, d_right = per_position_nuc_distances(qlen, nucs)

        accumulate(opps, d_left, d_right, den, MAX_LAG)
        accumulate(hits, d_left, d_right, num, MAX_LAG)
        n_reads += 1
        if n_reads >= max_reads:
            break
    bam.close()
    print(f'processed {n_reads} reads')
    return n_reads, num, den


def robust_amplitude(enr, den_counts, min_count=200):
    """Robust amplitude = average enrichment at helical-pitch peaks
    (lags 9, 10, 11, 20, 21, 31, 32) MINUS average at troughs
    (lags 5, 6, 15, 16, 25, 26).

    Require each sampled lag to have >= min_count opp-pairs in its
    denominator so we're not dividing by noise.
    """
    peak_lags = (9, 10, 11, 20, 21, 31, 32)
    trough_lags = (5, 6, 15, 16, 25, 26)
    peak_vals = []
    for L in peak_lags:
        if L < len(enr) and den_counts[L] >= min_count:
            peak_vals.append(enr[L])
    trough_vals = []
    for L in trough_lags:
        if L < len(enr) and den_counts[L] >= min_count:
            trough_vals.append(enr[L])
    if len(peak_vals) < 3 or len(trough_vals) < 3:
        return np.nan
    return float(np.mean(peak_vals) - np.mean(trough_vals))


def main():
    bam_path = os.path.join(HERE, 'output', 'bam',
                              'scDAF_PS00758__v8_gapcdf.bam')
    if not os.path.exists(bam_path):
        print(f'missing {bam_path}')
        return
    n_reads, num, den = process_bam(bam_path)

    # Per-cell normalized enrichment and amplitude
    A_grid = np.full((N_BINS, N_BINS), np.nan, dtype=np.float64)
    pair_counts = np.zeros((N_BINS, N_BINS), dtype=np.int64)
    for lb in range(N_BINS):
        for rb in range(N_BINS):
            d = den[lb, rb]
            if d[4:40].sum() < 1000:
                continue
            pr = np.where(d > 0, num[lb, rb] / np.maximum(d, 1), 0.0)
            baseline = np.mean(pr[40:MAX_LAG + 1])
            if baseline <= 0:
                continue
            enr = pr / baseline
            A_grid[lb, rb] = robust_amplitude(enr, d, min_count=200)
            pair_counts[lb, rb] = d[4:40].sum()

    # Print grid
    print('\nAmplitude A(d_left_bin, d_right_bin)')
    print(f'Bin edges: {DIST_EDGES.tolist()}')
    print(f'{"":>10s}  ' + '  '.join(f'r{i}' for i in range(N_BINS)))
    labels = [f'{DIST_EDGES[i]}-{DIST_EDGES[i+1]}' for i in range(N_BINS)]
    for lb in range(N_BINS):
        row = '  '.join(
            f'{A_grid[lb, rb]:+.3f}' if not np.isnan(A_grid[lb, rb])
            else '  nan '
            for rb in range(N_BINS)
        )
        print(f'{labels[lb]:>10s}  {row}')

    print('\nPair counts (lag 4-40) per cell:')
    print(f'{"":>10s}  ' + '  '.join(f'r{i}' for i in range(N_BINS)))
    for lb in range(N_BINS):
        row = '  '.join(f'{pair_counts[lb, rb]:6d}' for rb in range(N_BINS))
        print(f'{labels[lb]:>10s}  {row}')

    # ---- plot ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5),
                               gridspec_kw={'width_ratios': [1, 1.3]})

    # Panel A: heatmap
    ax = axes[0]
    im = ax.imshow(A_grid, origin='lower', aspect='auto',
                    cmap='viridis', vmin=0, vmax=0.12)
    ax.set_xticks(range(N_BINS))
    ax.set_yticks(range(N_BINS))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel('d_right: distance to nearest downstream nuc (bp)')
    ax.set_ylabel('d_left: distance to nearest upstream nuc (bp)')
    ax.set_title('Cosine amplitude A per (d_left, d_right) cell',
                  fontsize=10)
    cbar = plt.colorbar(im, ax=ax, fraction=0.045)
    cbar.set_label('A (cosine amplitude)', fontsize=8)
    # Annotate cells with their A value
    for lb in range(N_BINS):
        for rb in range(N_BINS):
            v = A_grid[lb, rb]
            if not np.isnan(v):
                ax.text(rb, lb, f'{v:.2f}', ha='center', va='center',
                        fontsize=6,
                        color='white' if v < 0.06 else 'black')

    # Panel B: decay curves
    ax = axes[1]
    # Short linker slice: diagonal d_left == d_right
    diag_d = []
    diag_A = []
    for b in range(N_BINS):
        if not np.isnan(A_grid[b, b]):
            mid = 0.5 * (DIST_EDGES[b] + DIST_EDGES[b + 1])
            diag_d.append(mid)
            diag_A.append(A_grid[b, b])
    ax.plot(diag_d, diag_A, 'o-', color='#d97706', linewidth=2,
             markersize=8, label='Diagonal: d_left ≈ d_right (both sides)')

    # One-sided slice: d_left small, d_right large
    for lb in (0, 1):  # d_left in first bin (0-10 or 10-20)
        d_arr = []
        A_arr = []
        for rb in range(N_BINS):
            v = A_grid[lb, rb]
            if not np.isnan(v):
                mid = 0.5 * (DIST_EDGES[rb] + DIST_EDGES[rb + 1])
                d_arr.append(mid)
                A_arr.append(v)
        color = '#2563eb' if lb == 0 else '#60a5fa'
        label = f'd_left={labels[lb]}: vary d_right (one side only)'
        ax.plot(d_arr, A_arr, 's--', color=color, linewidth=1.5,
                 markersize=6, label=label)

    # Expected two-sided sum model with tau=50
    tau = 50.0
    A0 = 0.06
    x = np.linspace(5, 400, 200)
    ax.plot(x, A0 * 2 * np.exp(-x / tau),
             color='#888', linestyle=':', linewidth=1.2,
             label=f'Model: 2·A0·exp(-d/{tau:.0f}) (diag)')
    ax.plot(x, A0 * (1 + np.exp(-x / tau)),
             color='#444', linestyle=':', linewidth=1.2,
             label=f'Model: A0·(1 + exp(-d/{tau:.0f})) (one-sided)')

    ax.set_xlabel('distance to nearest nuc (bp)')
    ax.set_ylabel('cosine amplitude A')
    ax.set_xscale('log')
    ax.set_xlim(5, 500)
    ax.set_title('Amplitude decay: bidirectional vs one-sided',
                  fontsize=10)
    ax.grid(alpha=0.3, which='both')
    ax.legend(fontsize=7, loc='best')
    ax.axhline(0, color='k', linewidth=0.5)

    fig.suptitle(
        f'scDAF PS00758 — rotational amplitude vs distance to nuc '
        f'(bidirectional)  n={n_reads}',
        fontsize=11, y=1.02,
    )
    out = os.path.join(HERE, 'output', 'periodicity_decay.png')
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'\nwrote {out}')


if __name__ == '__main__':
    main()
