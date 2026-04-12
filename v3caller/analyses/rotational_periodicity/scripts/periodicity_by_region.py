"""Region-stratified pair-correlation periodicity analysis on scDAF.

Prediction (from the 2026-04-11 naked-DddB result):
  10 bp periodicity is chromatin-phased, not enzyme-intrinsic.
  DddB is blocked by histone contacts on the wrapped face, so
  periodicity only appears in DNA where rotation is constrained.

  BUT: nucleosome CORES in scDAF are essentially hit-free by
  construction (that's how we call them). The pair-correlation
  signal in bulk data must come from elsewhere. Candidates:

    - short linkers (≤80 bp): DNA rotationally pinned by the
      flanking nucs → expected to show oscillation
    - nuc edge / breathing zone (first/last 20 bp of each nuc
      call): has sparse hits, wrapping is partial → some signal
    - large NFRs / MSPs (≥200 bp): DNA is not wrapped → expected
      flat, like naked DddB
    - cross-MSP pairs: one hit in an MSP, next hit in the next
      MSP, distance = ~linker + ~nuc + ~linker. Tests whether
      the nuc pins the rotational register of its flanking
      linker DNA (transferable phase).
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


MAX_LAG = 80
MAX_READS = 4000
MIN_ALIGN = 2000


def daf_hit_opp_query(read):
    """(opp, hit) arrays in query coords via majority-vote encoding."""
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


def pair_hist(positions, max_lag):
    if positions.size < 2:
        return np.zeros(max_lag + 1, dtype=np.int64)
    out = np.zeros(max_lag + 1, dtype=np.int64)
    for i in range(positions.size - 1):
        j = np.searchsorted(positions, positions[i] + max_lag, side='right')
        if j <= i + 1:
            continue
        ds = positions[i + 1:j] - positions[i]
        if ds.size:
            np.add.at(out, ds, 1)
    return out


def build_masks_from_read(read):
    """Return dict of region-name -> list of (start, end) query intervals.

    Uses the iter-16 ns/nl + as/al tags in query coords.
    """
    try:
        ns = list(read.get_tag('ns'))
        nl = list(read.get_tag('nl'))
    except KeyError:
        ns, nl = [], []
    try:
        as_ = list(read.get_tag('as'))
        al = list(read.get_tag('al'))
    except KeyError:
        as_, al = [], []

    # Short linker-scale MSPs (likely rotationally pinned by flanking nucs)
    short_linker = [(s, s + l) for s, l in zip(as_, al) if l <= 80]
    # Mid linker + small MSP
    mid_linker = [(s, s + l) for s, l in zip(as_, al) if 80 < l < 200]
    # Large NFR-scale MSPs ≥200 bp, full interval
    large_nfr = [(s, s + l) for s, l in zip(as_, al) if l >= 200]
    # Inner 50% of large NFRs: trim 25% off each side. Tests whether
    # the residual periodicity in large NFRs lives at the edges
    # (adjacent to flanking nucs) or in the free middle.
    nfr_inner = []
    for s, l in zip(as_, al):
        if l >= 200:
            trim = int(l * 0.25)
            nfr_inner.append((s + trim, s + l - trim))

    # Nucleosome edge/breathing zone: first 25 bp + last 25 bp of each
    # called nuc. These are the wrapped regions that still carry some
    # hits (breathing in/out). Cores themselves (beyond 25 bp margin)
    # are essentially hit-free so don't contribute.
    nuc_edge = []
    for s, l in zip(ns, nl):
        ie = s + l
        if l > 60:
            nuc_edge.append((s, s + 25))
            nuc_edge.append((ie - 25, ie))
        else:
            # Short nuc — the whole thing is edge
            nuc_edge.append((s, ie))

    return {
        'short_linker': short_linker,
        'mid_linker': mid_linker,
        'large_nfr': large_nfr,
        'nfr_inner': nfr_inner,
        'nuc_edge': nuc_edge,
    }


def filter_positions(positions, intervals):
    """Sorted positions ∩ sorted non-overlapping intervals."""
    if not intervals or positions.size == 0:
        return np.empty(0, dtype=np.int64)
    intervals = sorted(intervals)
    out = []
    j = 0
    for p in positions:
        while j < len(intervals) and intervals[j][1] <= p:
            j += 1
        if j >= len(intervals):
            break
        if intervals[j][0] <= p < intervals[j][1]:
            out.append(p)
    return np.asarray(out, dtype=np.int64)


def pair_hist_cross_msps(positions, intervals, max_lag):
    """Count pairs where the two positions sit in DIFFERENT intervals.

    Intervals are sorted non-overlapping. For each position, find
    its interval index. A pair is "cross" if the indices differ.
    """
    if positions.size < 2 or len(intervals) < 2:
        return np.zeros(max_lag + 1, dtype=np.int64)
    intervals = sorted(intervals)
    out = np.zeros(max_lag + 1, dtype=np.int64)
    # Map each position to its interval index (or -1 if none)
    idx = np.full(positions.size, -1, dtype=np.int64)
    j = 0
    for k, p in enumerate(positions):
        while j < len(intervals) and intervals[j][1] <= p:
            j += 1
        if j < len(intervals) and intervals[j][0] <= p < intervals[j][1]:
            idx[k] = j
    valid = idx >= 0
    positions = positions[valid]
    idx = idx[valid]
    for i in range(positions.size - 1):
        jsw = np.searchsorted(positions, positions[i] + max_lag, side='right')
        if jsw <= i + 1:
            continue
        for k in range(i + 1, jsw):
            if idx[k] != idx[i]:
                d = int(positions[k] - positions[i])
                if 0 < d <= max_lag:
                    out[d] += 1
    return out


def process_bam(bam_path, max_reads=MAX_READS):
    regions = ['short_linker', 'mid_linker', 'large_nfr', 'nfr_inner', 'nuc_edge']
    num = {r: np.zeros(MAX_LAG + 1, dtype=np.int64) for r in regions}
    den = {r: np.zeros(MAX_LAG + 1, dtype=np.int64) for r in regions}
    total_bp = {r: 0 for r in regions}

    # Cross-MSP: num_hits_cross and num_opp_cross (using large + short)
    num_cross = np.zeros(MAX_LAG + 1, dtype=np.int64)
    den_cross = np.zeros(MAX_LAG + 1, dtype=np.int64)

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
        masks = build_masks_from_read(read)

        # Per-region intra-interval pair-correlation
        for region in regions:
            ivs = masks[region]
            if not ivs:
                continue
            reg_opps = filter_positions(opps, ivs)
            reg_hits = filter_positions(hits, ivs)
            if reg_opps.size >= 5:
                num[region] += pair_hist(reg_hits, MAX_LAG)
                den[region] += pair_hist(reg_opps, MAX_LAG)
                total_bp[region] += sum(e - s for s, e in ivs)

        # Cross-MSP pair-correlation: pairs of hits in DIFFERENT
        # short linkers (bridging a nucleosome). Tests whether the
        # flanking nuc transfers a rotational register to the linker
        # DNA on its other side.
        all_linkers = sorted(masks['short_linker'] + masks['mid_linker'])
        if len(all_linkers) >= 2:
            num_cross += pair_hist_cross_msps(hits, all_linkers, MAX_LAG)
            den_cross += pair_hist_cross_msps(opps, all_linkers, MAX_LAG)

        n_reads += 1
        if n_reads >= max_reads:
            break
    bam.close()

    pair_rate = {}
    for r in regions:
        pair_rate[r] = np.where(den[r] > 0, num[r] / np.maximum(den[r], 1),
                                   0.0)
    pair_rate_cross = np.where(den_cross > 0,
                                 num_cross / np.maximum(den_cross, 1), 0.0)
    return n_reads, pair_rate, pair_rate_cross, total_bp


def fit_amplitude(enrichment, d_lo=4, d_hi=50):
    """Fit A, phi for enrichment(d) = intercept + slope*d + A*cos(2π*d/10.4 + phi)."""
    from scipy.optimize import curve_fit
    d = np.arange(d_lo, d_hi)
    y = enrichment[d_lo:d_hi]
    if np.all(y == 0):
        return 0.0, 0.0, 1.0
    try:
        def model(d, A, phi, slope, intercept):
            return intercept + slope * d + A * np.cos(2 * np.pi * d / 10.4 + phi)
        popt, _ = curve_fit(model, d, y, p0=[0.1, 0.0, 0.0, 1.0],
                             maxfev=10000)
        A, phi, slope, intercept = popt
        pt = (intercept + abs(A)) / max(0.01, (intercept - abs(A)))
        return A, phi, pt
    except Exception:
        return 0.0, 0.0, 1.0


def main():
    bam_path = os.path.join(HERE, 'output', 'bam',
                              'scDAF_PS00758__v8_gapcdf.bam')
    if not os.path.exists(bam_path):
        print(f'missing bam: {bam_path}')
        return
    print(f'reading {bam_path}')
    n_reads, pair_rate, pair_rate_cross, total_bp = process_bam(bam_path)
    print(f'processed {n_reads} reads')
    for region, bp in total_bp.items():
        print(f'  {region:14s} total {bp} bp')

    # Normalize each to its own long-range baseline (lags 60-80)
    norms = {}
    for region in ['short_linker', 'mid_linker', 'large_nfr', 'nfr_inner',
                    'nuc_edge']:
        pr = pair_rate[region]
        baseline = np.mean(pr[60:MAX_LAG + 1])
        if baseline <= 0:
            baseline = np.mean(pr[pr > 0]) or 1.0
        norms[region] = pr / baseline
    # Cross-MSP baseline
    baseline_cross = np.mean(pair_rate_cross[60:MAX_LAG + 1])
    if baseline_cross <= 0:
        baseline_cross = np.mean(pair_rate_cross[pair_rate_cross > 0]) or 1.0
    norms['cross_msp'] = pair_rate_cross / baseline_cross

    print('\nFit A cos(2π*d/10.4 + φ) on lags 4-50:')
    fits = {}
    for region, y in norms.items():
        A, phi, pt = fit_amplitude(y)
        fits[region] = (A, phi, pt)
        print(f'  {region:14s}  A={A:+.3f}  phi={phi:+.2f}  '
              f'peak/trough={pt:.3f}')

    print('\nRaw enrichment, lags 4-22:')
    print(f'  {"":14s} ' + ' '.join(f'{l:>5d}' for l in range(4, 23)))
    for region, y in norms.items():
        print(f'  {region:14s} ' +
              ' '.join(f'{y[l]:5.2f}' for l in range(4, 23)))

    # ---- plot ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5),
                               gridspec_kw={'width_ratios': [1, 1.3]})
    colors = {
        'short_linker': '#d97706',
        'mid_linker':   '#f59e0b',
        'large_nfr':    '#dc2626',
        'nfr_inner':    '#7c2d12',
        'nuc_edge':     '#1e3a8a',
        'cross_msp':    '#7c3aed',
    }
    labels_pretty = {
        'short_linker': 'Short linker (MSP ≤80 bp)',
        'mid_linker':   'Mid linker (80-200 bp)',
        'large_nfr':    'Large NFR full (MSP ≥200 bp)',
        'nfr_inner':    'Large NFR inner 50% (25% trimmed)',
        'nuc_edge':     'Nuc edge (first/last 25 bp)',
        'cross_msp':    'Cross-MSP pairs (bridging nuc)',
    }

    # Four clean tracks showing the amplitude gradient with distance
    # to the nearest nucleosome:
    #   short_linker (both ends pinned) → full amplitude
    #   mid_linker                      → partial amplitude
    #   large_nfr full                  → weaker, but edges still carry signal
    #   nfr_inner (25% trimmed)         → noise floor (truly free DNA)
    # nuc_edge / cross_msp are omitted (too sparse).
    plot_regions = ['short_linker', 'mid_linker', 'large_nfr', 'nfr_inner']

    ax = axes[0]
    for region in plot_regions:
        y = norms[region]
        A = fits[region][0]
        ax.plot(np.arange(len(y))[3:60], y[3:60],
                 color=colors[region], linewidth=1.7,
                 label=f'{labels_pretty[region]}  A={A:+.3f}')
    ax.axhline(1.0, color='k', linewidth=0.6)
    for d in (10.4, 20.8, 31.2, 41.6, 52.0):
        ax.axvline(d, color='#888', linestyle=':', linewidth=0.6)
    ax.set_xlabel('pairwise distance (bp)')
    ax.set_ylabel('enrichment (pair_rate / baseline)')
    ax.set_title('Raw enrichment', fontsize=11)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc='best')

    ax = axes[1]
    d_arr = np.arange(3, 60)
    for region in plot_regions:
        y = norms[region][3:60]
        coeff = np.polyfit(d_arr, y, 1)
        y_det = y - (coeff[0] * d_arr + coeff[1])
        ax.plot(d_arr, y_det, color=colors[region], linewidth=1.7,
                 label=labels_pretty[region])
    ax.axhline(0.0, color='k', linewidth=0.6)
    for d in (10.4, 20.8, 31.2, 41.6, 52.0):
        ax.axvline(d, color='#888', linestyle=':', linewidth=0.6)
    ax.set_xlabel('pairwise distance (bp)')
    ax.set_ylabel('detrended (residual oscillation)')
    ax.set_title('Detrended — isolated oscillation', fontsize=11)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc='best')

    fig.suptitle(
        f'scDAF PS00758 periodicity by chromatin region  '
        f'(n={n_reads} reads; nuc cores omitted — hit-free by construction)',
        fontsize=12, y=1.02,
    )
    out = os.path.join(HERE, 'output', 'periodicity_by_region.png')
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'\nwrote {out}')


if __name__ == '__main__':
    main()
