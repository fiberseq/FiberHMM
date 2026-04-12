"""Control: does the oscillating decay profile we see when anchoring
at the nuc edge show up when we anchor FAR from any nuc?

Procedure:
  For each hit in the read, compute its minimum distance to the
  nearest nuc edge (d_nuc). Assign each hit to one of four anchor
  bins by d_nuc:

    near:     hits 0-20 bp from nearest nuc edge
    mid:      hits 20-60 bp
    far:      hits 60-120 bp
    deep:     hits ≥ 120 bp (middle of a long NFR)

  For each anchor hit, walk ±W bp and accumulate num[bin][d] += 1
  if the position is a hit, den[bin][d] += 1 if it's an opp.
  (We skip d=0 — the anchor hit — so we're measuring conditional
  P(hit at anchor+d | hit at anchor) at signed distance d.)

  Normalize each bin by its own far-positive baseline, plot the
  four curves overlaid. If the oscillation amplitude decays with
  d_nuc, the effect is nuc-driven. If the oscillation looks
  identical across all bins, it's a universal pair-correlation
  property of DAF hits on this DNA and has nothing to do with
  nucleosomes specifically.
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


MAX_DIST = 80
MAX_READS = 40000
MIN_ALIGN = 2000

ANCHOR_BINS = [
    (0,    15,  'Anchor 0-15 bp from nuc',    '#b91c1c'),
    (15,   30,  'Anchor 15-30 bp from nuc',   '#d97706'),
    (30,   60,  'Anchor 30-60 bp from nuc',   '#f59e0b'),
    (60,   100, 'Anchor 60-100 bp from nuc',  '#2563eb'),
    (100,  200, 'Anchor 100-200 bp from nuc', '#1e3a8a'),
    (200, 1000, 'Anchor ≥200 bp from nuc',    '#065f46'),
]


def opp_hit_arrays_query(read):
    try:
        pairs = read.get_aligned_pairs(with_seq=True, matches_only=True)
    except ValueError:
        return None
    seq = read.query_sequence
    if seq is None:
        return None
    qlen = read.query_length or 0
    if qlen == 0:
        return None
    c_opp = np.zeros(qlen, dtype=np.int8)
    g_opp = np.zeros(qlen, dtype=np.int8)
    y_hit = np.zeros(qlen, dtype=np.int8)
    t_hit = np.zeros(qlen, dtype=np.int8)
    r_hit = np.zeros(qlen, dtype=np.int8)
    a_hit = np.zeros(qlen, dtype=np.int8)
    ny = nt = nr = na = 0
    for qpos, rpos, rb in pairs:
        if rb is None or qpos is None:
            continue
        qb = seq[qpos].upper()
        rbu = rb.upper()
        if rbu == 'C':
            c_opp[qpos] = 1
            if qb == 'Y':
                y_hit[qpos] = 1; ny += 1
            elif qb == 'T':
                t_hit[qpos] = 1; nt += 1
        elif rbu == 'G':
            g_opp[qpos] = 1
            if qb == 'R':
                r_hit[qpos] = 1; nr += 1
            elif qb == 'A':
                a_hit[qpos] = 1; na += 1
    counts = {'CtoY': ny, 'CtoT': nt, 'GtoR': nr, 'GtoA': na}
    best = max(counts, key=counts.get)
    if counts[best] == 0:
        return None
    if best == 'CtoY': return c_opp, y_hit
    if best == 'CtoT': return c_opp, t_hit
    if best == 'GtoR': return g_opp, r_hit
    return g_opp, a_hit


def nuc_dist_and_direction(qlen, nucs):
    """For every query position, compute:
      d_nuc[pos]     = distance to the nearest nuc edge (≥0; -1 if inside nuc)
      nuc_dir[pos]   = +1 if nearest nuc edge is to the RIGHT of pos,
                        -1 if it's to the LEFT, 0 if inside nuc.

    This lets callers ORIENT the walk: walking in direction +nuc_dir
    goes TOWARD the nearest nuc, walking in -nuc_dir goes AWAY.
    """
    if not nucs:
        return (np.full(qlen, 10_000, dtype=np.int32),
                np.zeros(qlen, dtype=np.int8))
    edges = np.array(sorted(
        [s for s, _ in nucs] + [e for _, e in nucs]
    ), dtype=np.int64)
    in_nuc = np.zeros(qlen, dtype=bool)
    for s, e in nucs:
        in_nuc[max(0, s):min(qlen, e)] = True
    d_nuc = np.full(qlen, 10_000, dtype=np.int32)
    nuc_dir = np.zeros(qlen, dtype=np.int8)
    positions = np.arange(qlen, dtype=np.int64)
    idx = np.searchsorted(edges, positions)
    idx_left = np.clip(idx - 1, 0, len(edges) - 1)
    idx_right = np.clip(idx, 0, len(edges) - 1)
    dist_left = np.abs(positions - edges[idx_left])
    dist_right = np.abs(edges[idx_right] - positions)
    left_closer = dist_left <= dist_right
    d_nuc[:] = np.where(left_closer, dist_left, dist_right)
    # Direction: if nearest edge is to the LEFT (dist_left is smaller),
    # then nuc_dir = -1 (nuc is in the negative direction).
    # If nearest edge is to the RIGHT, nuc_dir = +1.
    nuc_dir[:] = np.where(left_closer, -1, +1)
    d_nuc[in_nuc] = -1
    nuc_dir[in_nuc] = 0
    return d_nuc, nuc_dir


def bin_anchor(dist):
    for i, (lo, hi, _, _) in enumerate(ANCHOR_BINS):
        if lo <= dist < hi:
            return i
    return None


def process_bam(bam_path, max_reads=MAX_READS):
    width = 2 * MAX_DIST + 1
    offset = MAX_DIST
    num = [np.zeros(width, dtype=np.int64) for _ in ANCHOR_BINS]
    den = [np.zeros(width, dtype=np.int64) for _ in ANCHOR_BINS]
    n_anchors = [0] * len(ANCHOR_BINS)
    n_reads = 0
    bam = pysam.AlignmentFile(bam_path, 'rb', check_sq=False)
    for read in bam.fetch(until_eof=True):
        if read.is_unmapped or read.is_secondary or read.is_supplementary:
            continue
        if (read.query_alignment_length or 0) < MIN_ALIGN:
            continue
        rp = opp_hit_arrays_query(read)
        if rp is None:
            continue
        opp_arr, hit_arr = rp
        qlen = opp_arr.shape[0]
        try:
            ns = list(read.get_tag('ns'))
            nl = list(read.get_tag('nl'))
        except KeyError:
            continue
        if not ns:
            continue
        nucs = [(s, s + l) for s, l in zip(ns, nl)]
        d_nuc, nuc_dir = nuc_dist_and_direction(qlen, nucs)

        # VECTORIZED anchor walk.
        # For each hit NOT inside a nuc, walk ±MAX_DIST in query coords
        # and orient relative to the nearest nuc:
        #   oriented_d > 0 = away from nearest nuc (into accessible DNA)
        #   oriented_d < 0 = toward nearest nuc (toward nuc body)
        hit_positions = np.where(hit_arr > 0)[0]

        # Filter to valid anchors (outside nucs, with known direction)
        anchor_dists = d_nuc[hit_positions]
        anchor_dirs = nuc_dir[hit_positions]
        valid_anchor = (anchor_dists >= 0) & (anchor_dirs != 0)
        anchors = hit_positions[valid_anchor]
        a_dists = anchor_dists[valid_anchor]
        a_dirs = anchor_dirs[valid_anchor]

        if anchors.size == 0:
            n_reads += 1
            if n_reads >= max_reads:
                break
            continue

        # Assign anchor bins
        a_bins = np.full(anchors.size, -1, dtype=np.int32)
        for bi, (lo, hi, _, _) in enumerate(ANCHOR_BINS):
            mask = (a_dists >= lo) & (a_dists < hi)
            a_bins[mask] = bi

        valid_bin = a_bins >= 0
        anchors = anchors[valid_bin]
        a_dirs = a_dirs[valid_bin]
        a_bins = a_bins[valid_bin]

        if anchors.size == 0:
            n_reads += 1
            if n_reads >= max_reads:
                break
            continue

        # Steps: -MAX_DIST .. -1, 1 .. MAX_DIST (exclude 0)
        steps = np.concatenate([np.arange(-MAX_DIST, 0),
                                  np.arange(1, MAX_DIST + 1)])

        # Broadcast: target positions = anchors[:, None] + steps[None, :]
        targets = anchors[:, None] + steps[None, :]  # (N, 2*MAX_DIST)

        # Clip to [0, qlen) and check validity
        in_bounds = (targets >= 0) & (targets < qlen)
        targets_clipped = np.clip(targets, 0, qlen - 1)

        # Look up arrays at target positions
        t_opp = opp_arr[targets_clipped]  # (N, S)
        t_hit = hit_arr[targets_clipped]
        t_dnuc = d_nuc[targets_clipped]

        # Valid = in bounds AND not inside nuc AND is an opp position
        valid = in_bounds & (t_dnuc >= 0) & (t_opp > 0)

        # Oriented distance: oriented_d = -direction * query_step
        # a_dirs is (N,), steps is (S,)
        oriented_d = -a_dirs[:, None] * steps[None, :]  # (N, S)
        hist_idx = oriented_d + offset  # map to histogram index

        # Filter to valid histogram indices
        valid &= (hist_idx >= 0) & (hist_idx < width)

        # Accumulate into per-bin histograms
        for bi in range(len(ANCHOR_BINS)):
            bin_mask = (a_bins == bi)
            if not bin_mask.any():
                continue
            n_anchors[bi] += int(bin_mask.sum())
            # Sub-select rows for this bin
            v = valid[bin_mask]
            h = hist_idx[bin_mask]
            is_hit = t_hit[bin_mask]
            # Flatten valid entries
            flat_idx = h[v]
            flat_hit = is_hit[v]
            np.add.at(den[bi], flat_idx, 1)
            np.add.at(num[bi], flat_idx[flat_hit > 0], 1)
        n_reads += 1
        if n_reads >= max_reads:
            break
    bam.close()
    print(f'reads={n_reads}  anchors/bin:')
    for i, (lo, hi, label, _) in enumerate(ANCHOR_BINS):
        print(f'  {label}  {n_anchors[i]}')
    return n_reads, num, den, offset, width


def fit_decay(d_signed, enr, d_lo=5, d_hi=60):
    from scipy.optimize import curve_fit
    mask = (d_signed >= d_lo) & (d_signed <= d_hi)
    d = d_signed[mask].astype(float)
    y = enr[mask]
    if len(d) < 5:
        return (np.nan, np.nan, np.nan)
    try:
        def model(d, A, phi, tau):
            return 1.0 + A * np.cos(2 * np.pi * d / 10.4 + phi) * np.exp(-d / tau)
        popt, _ = curve_fit(model, d, y, p0=[0.2, 0.0, 30.0],
                             bounds=([-2, -np.pi, 3], [2, np.pi, 500]),
                             maxfev=20000)
        return tuple(popt)
    except Exception:
        return (np.nan, np.nan, np.nan)


def main():
    global MAX_READS
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--bam', default=None,
                    help='iter-16 BAM to analyze. Default: pick the '
                         'first available scDAF BAM')
    ap.add_argument('--label', default='scDAF',
                    help='Short label used in the output figure name')
    ap.add_argument('--max-reads', type=int, default=MAX_READS)
    args = ap.parse_args()

    if args.bam is None:
        # Default: scDAF full or bench-subsample
        candidates = [
            '/tmp/scdaf_v8_gapcdf_full_iter16.bam',
            os.path.join(HERE, 'output', 'bam',
                          'scDAF_PS00758__v8_gapcdf.bam'),
        ]
        bam_path = next((p for p in candidates if os.path.exists(p)), None)
    else:
        bam_path = args.bam
    if bam_path is None or not os.path.exists(bam_path):
        print(f'missing bam: {bam_path}')
        return
    MAX_READS = args.max_reads
    print(f'reading {bam_path}')
    n_reads, num, den, offset, width = process_bam(bam_path)
    d_signed = np.arange(width) - offset

    print('\nDecay fits per anchor bin:')
    norms = []
    for i, (lo, hi, label, color) in enumerate(ANCHOR_BINS):
        d = den[i]
        if d.sum() < 500:
            norms.append(None)
            continue
        rate = np.where(d > 0, num[i] / np.maximum(d, 1), 0.0)
        # Baseline: mean hit rate at the FAR POSITIVE tail only
        # (d = +60 to +80, which is deep in accessible DNA, AWAY
        # from the nearest nuc). Do NOT use the negative tail — that
        # direction heads into the nuc body where rate → 0, which
        # would drag the baseline down and inflate everything.
        bl_slice = rate[offset + 60:offset + 80 + 1]
        bl_slice = bl_slice[bl_slice > 0]
        if bl_slice.size < 3:
            norms.append(None)
            continue
        baseline = float(np.mean(bl_slice))
        if baseline <= 0:
            norms.append(None)
            continue
        norm = rate / baseline
        norms.append(norm)
        A, phi, tau = fit_decay(d_signed, norm, d_lo=5, d_hi=60)
        # Also fit on the negative side (symmetric)
        A_neg, phi_neg, tau_neg = fit_decay(-d_signed, norm, d_lo=5, d_hi=60)
        print(f'  {label}')
        print(f'      pos arm: A={A:+.3f}  phi={phi:+.2f}  tau={tau:.1f} bp')
        print(f'      neg arm: A={A_neg:+.3f}  phi={phi_neg:+.2f}  tau={tau_neg:.1f} bp')

    # ---- plot ----
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    for i, (lo, hi, label, color) in enumerate(ANCHOR_BINS):
        if norms[i] is None:
            continue
        y = norms[i]
        ax.plot(d_signed, y, color=color, linewidth=1.5, label=label)
    ax.axhline(1.0, color='k', linewidth=0.7)
    ax.axvline(0, color='#dc2626', linewidth=0.8,
                label='anchor hit position')
    for d in (10.4, 20.8, 31.2, 41.6, 52.0, 62.4, 72.8):
        ax.axvline(d, color='#888', linestyle=':', linewidth=0.4)
        ax.axvline(-d, color='#888', linestyle=':', linewidth=0.4)
    ax.set_xlabel('oriented distance from anchor (bp)\n'
                   '← toward nearest nuc    |    away from nearest nuc →')
    ax.set_ylabel('hit rate / baseline')
    ax.set_title(f'{args.label} — oriented anchor profiles by '
                  f'distance from nearest nuc  (n={n_reads})',
                  fontsize=11)
    ax.set_xlim(-MAX_DIST, MAX_DIST)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc='best')

    out = os.path.join(HERE, 'output',
                        f'periodicity_anchor_control_{args.label}.png')
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'\nwrote {out}')


if __name__ == '__main__':
    main()
