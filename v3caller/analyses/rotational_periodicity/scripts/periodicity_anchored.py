"""Nuc-edge-anchored hit rate — clean rotational decay analysis.

Anchor = the FIRST HIT in the accessible region past a nucleosome's
downstream boundary (and the first hit before the upstream boundary,
mirrored). Anchoring on a hit rather than the caller's called edge
eliminates the d=0 selection spike, because the caller's edge is
placed where hits start (so d=0 is always "a hit by construction").

For each anchor we walk in both directions:

    d < 0: toward the nuc body (negative "inside-nuc" distance)
    d = 0: the anchor hit itself
    d > 0: further into the accessible region

At each offset d we record, per opportunity at that query position:
    den[d] += 1
    num[d] += 1 if it's a hit

hit_rate(d) = num[d] / den[d] shows the anchored rotational profile.

Two stratifications, each plotted in its own panel:

  (1) by MSP SIZE of the accessible region the anchor leads into:
        linker ≤60, mid 60-120, 120-250, NFR ≥250.
      The user's primary question: "does a long NFR recover to
      baseline faster than a short linker, and how far past a nuc
      edge does the rotational phasing extend?"

  (2) by NUC CONFIDENCE (mq) of the flanking nuc:
        high-mq (≥192) vs low-mq (<192) nucs.
      The user's secondary question: "is the harsh near-zero
      footprint at d<0 only the high-confidence cores?" If yes,
      then low-mq nucs should show internal deaminations (higher
      rate at d<0) even though they're labeled as nucs by the
      caller.

mq is parsed from the MA tag (iter-16 QQQQ layout: nq, mq, lq, rq).
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

from ma_tags import parse_ma_tag, parse_aq_array  # noqa: E402


DIST_INTO_NUC = 60
DIST_INTO_MSP = 150
MAX_READS = 4000
MIN_ALIGN = 2000

# Stratum 1: MSP length the anchor leads into
MSP_STRATA = [
    (20,  60,  'MSP 20-60 (short linker)', '#d97706'),
    (60,  120, 'MSP 60-120 (mid linker)',  '#f59e0b'),
    (120, 250, 'MSP 120-250',               '#dc2626'),
    (250, 1500,'MSP ≥250 (NFR)',            '#7c2d12'),
]

# Stratum 2: mq band of the flanking nuc
MQ_STRATA = [
    (192, 256, 'High-mq nuc (mq≥192)', '#065f46'),
    (64,  192, 'Mid-mq nuc (64≤mq<192)', '#f59e0b'),
    (0,   64,  'Low-mq nuc (mq<64)',   '#b91c1c'),
]


# ---- position extraction ----

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
    if best == 'CtoY':
        return c_opp, y_hit
    if best == 'CtoT':
        return c_opp, t_hit
    if best == 'GtoR':
        return g_opp, r_hit
    return g_opp, a_hit


def get_nucs_and_mqs(read):
    """Return list of (start, end, mq) per nucleosome.

    mq is parsed from the MA tag if present, else defaults to 255.
    """
    try:
        ns = list(read.get_tag('ns'))
        nl = list(read.get_tag('nl'))
    except KeyError:
        return []
    mqs = [255] * len(ns)
    if read.has_tag('MA') and read.has_tag('AQ'):
        try:
            parsed = parse_ma_tag(read.get_tag('MA'))
            aq = list(read.get_tag('AQ'))
            qspecs = [rt[2] for rt in parsed['raw_types']]
            npers = [len(rt[3]) for rt in parsed['raw_types']]
            per_ann = parse_aq_array(aq, qspecs, npers)
            idx = 0
            for rt in parsed['raw_types']:
                count = len(rt[3])
                if rt[0] == 'nuc' and len(rt[2]) >= 2:
                    for k in range(count):
                        if k < len(mqs):
                            mqs[k] = per_ann[idx + k][1]
                idx += count
        except Exception:
            pass
    return [(s, s + l, mq) for s, l, mq in zip(ns, nl, mqs)]


def get_msps(read):
    try:
        as_ = list(read.get_tag('as'))
        al = list(read.get_tag('al'))
    except KeyError:
        return []
    return sorted([(s, s + l) for s, l in zip(as_, al)])


def find_msp_after(nuc_end, msps):
    """MSP that starts AT or AFTER nuc_end. Return (start, end, length)
    or None."""
    for s, e in msps:
        if s >= nuc_end:
            return s, e, e - s
        if e > nuc_end:
            return nuc_end, e, e - nuc_end
    return None


def find_msp_before(nuc_start, msps):
    """MSP that ends AT or BEFORE nuc_start. Return (start, end, length)
    or None."""
    result = None
    for s, e in msps:
        if e <= nuc_start:
            result = (s, e, e - s)
        elif s < nuc_start <= e:
            result = (s, nuc_start, nuc_start - s)
            break
        else:
            break
    return result


def bin_msp(msp_len):
    for i, (lo, hi, _, _) in enumerate(MSP_STRATA):
        if lo <= msp_len < hi:
            return i
    return None


def bin_mq(mq):
    for i, (lo, hi, _, _) in enumerate(MQ_STRATA):
        if lo <= mq < hi:
            return i
    return None


# ---- accumulate histograms ----

def process_bam(bam_path, max_reads=MAX_READS):
    width = DIST_INTO_NUC + DIST_INTO_MSP + 1
    offset = DIST_INTO_NUC  # physical index for d=0

    num_msp = [np.zeros(width, dtype=np.int64) for _ in MSP_STRATA]
    den_msp = [np.zeros(width, dtype=np.int64) for _ in MSP_STRATA]
    num_mq  = [np.zeros(width, dtype=np.int64) for _ in MQ_STRATA]
    den_mq  = [np.zeros(width, dtype=np.int64) for _ in MQ_STRATA]
    num_all = np.zeros(width, dtype=np.int64)
    den_all = np.zeros(width, dtype=np.int64)

    n_reads = 0
    n_anchors = 0

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

        nucs = get_nucs_and_mqs(read)  # [(ns, ne, mq)]
        if not nucs:
            continue
        msps = get_msps(read)

        for nuc_s, nuc_e, nuc_mq in nucs:
            mq_bin = bin_mq(nuc_mq)
            nuc_l = nuc_e - nuc_s

            # ---- downstream anchor ----
            msp_after = find_msp_after(nuc_e, msps)
            if msp_after is not None:
                msp_s, msp_e, msp_len = msp_after
                msp_bin = bin_msp(msp_len)
                if msp_bin is not None and msp_len >= 10:
                    # Find the first HIT inside [nuc_e, msp_e)
                    anchor = None
                    for p in range(nuc_e, min(msp_e, qlen)):
                        if hit_arr[p]:
                            anchor = p
                            break
                    if anchor is not None:
                        # Walk back (d<0) into the nuc body
                        # and forward (d>0) into the MSP
                        d_min = -min(DIST_INTO_NUC, anchor - nuc_s + 20)
                        # Don't walk into previous nuc
                        prev_nuc_e = 0
                        for _, pe, _ in nucs:
                            if pe <= nuc_s and pe > prev_nuc_e:
                                prev_nuc_e = pe
                        d_min = max(d_min, prev_nuc_e - anchor)
                        d_max = min(DIST_INTO_MSP, msp_e - anchor - 1)
                        for d in range(d_min, d_max + 1):
                            pos = anchor + d
                            if pos < 0 or pos >= qlen:
                                continue
                            if opp_arr[pos]:
                                idx = d + offset
                                if 0 <= idx < width:
                                    den_all[idx] += 1
                                    den_msp[msp_bin][idx] += 1
                                    if mq_bin is not None:
                                        den_mq[mq_bin][idx] += 1
                                    if hit_arr[pos]:
                                        num_all[idx] += 1
                                        num_msp[msp_bin][idx] += 1
                                        if mq_bin is not None:
                                            num_mq[mq_bin][idx] += 1
                        n_anchors += 1

            # ---- upstream anchor (mirrored) ----
            msp_before = find_msp_before(nuc_s, msps)
            if msp_before is not None:
                msp_s, msp_e, msp_len = msp_before
                msp_bin = bin_msp(msp_len)
                if msp_bin is not None and msp_len >= 10:
                    # Find the last HIT inside [msp_s, nuc_s)
                    anchor = None
                    for p in range(min(nuc_s, qlen) - 1, msp_s - 1, -1):
                        if p < 0: break
                        if hit_arr[p]:
                            anchor = p
                            break
                    if anchor is not None:
                        # MIRROR: positive d goes UPSTREAM away from the nuc,
                        # negative d goes DOWNSTREAM into the nuc body
                        d_max_mirror = min(DIST_INTO_MSP, anchor - msp_s)
                        d_min_mirror = -min(DIST_INTO_NUC, nuc_e - anchor)
                        # Don't walk past the far end of the nuc
                        # into the next MSP
                        for d in range(d_min_mirror, d_max_mirror + 1):
                            # pos is mirrored: d>0 → upstream
                            pos = anchor - d if d > 0 else anchor - d
                            # same formula; just note semantics
                            if pos < 0 or pos >= qlen:
                                continue
                            if opp_arr[pos]:
                                idx = d + offset
                                if 0 <= idx < width:
                                    den_all[idx] += 1
                                    den_msp[msp_bin][idx] += 1
                                    if mq_bin is not None:
                                        den_mq[mq_bin][idx] += 1
                                    if hit_arr[pos]:
                                        num_all[idx] += 1
                                        num_msp[msp_bin][idx] += 1
                                        if mq_bin is not None:
                                            num_mq[mq_bin][idx] += 1
                        n_anchors += 1

        n_reads += 1
        if n_reads >= max_reads:
            break
    bam.close()
    print(f'reads={n_reads}  anchors={n_anchors}')
    return (n_reads, num_msp, den_msp, num_mq, den_mq,
            num_all, den_all, offset, width)


# ---- decay fit ----

def fit_decay(d_signed, enr, d_lo=5, d_hi=80):
    """Fit y = 1 + A*cos(2π*d/10.4 + φ)*exp(-d/τ) on the positive arm."""
    from scipy.optimize import curve_fit
    mask = (d_signed >= d_lo) & (d_signed <= d_hi)
    d = d_signed[mask].astype(float)
    y = enr[mask]
    if len(d) < 5 or np.all(y == 0):
        return (np.nan, np.nan, np.nan)
    try:
        def model(d, A, phi, tau):
            return 1.0 + A * np.cos(2 * np.pi * d / 10.4 + phi) * np.exp(-d / tau)
        popt, _ = curve_fit(model, d, y, p0=[0.2, 0.0, 50.0],
                             bounds=([-2, -np.pi, 5], [2, np.pi, 500]),
                             maxfev=20000)
        return tuple(popt)
    except Exception:
        return (np.nan, np.nan, np.nan)


def normalize(num, den, width, offset, fallback_baseline=None):
    """Baseline = mean hit rate at the far-positive tail (d = 80-150).
    For short regions that never populate those lags, fall back to
    the supplied `fallback_baseline` (e.g. the all-anchors baseline).
    """
    rate = np.where(den > 0, num / np.maximum(den, 1), 0.0)
    bl_lo = offset + 80
    bl_hi = offset + 150
    if bl_hi > width:
        bl_hi = width
    bl_slice = rate[bl_lo:bl_hi]
    bl_slice = bl_slice[bl_slice > 0]
    if bl_slice.size >= 3:
        baseline = float(np.mean(bl_slice))
        if baseline > 0:
            return rate / baseline, baseline
    if fallback_baseline is not None and fallback_baseline > 0:
        return rate / fallback_baseline, fallback_baseline
    return None, None


def main():
    bam_path = os.path.join(HERE, 'output', 'bam',
                              'scDAF_PS00758__v8_gapcdf.bam')
    if not os.path.exists(bam_path):
        print(f'missing {bam_path}')
        return
    print(f'reading {bam_path}')
    (n_reads, num_msp, den_msp, num_mq, den_mq,
     num_all, den_all, offset, width) = process_bam(bam_path)

    d_signed = np.arange(width) - offset

    norm_all, baseline_all = normalize(num_all, den_all, width, offset)
    norms_msp = []
    for i in range(len(MSP_STRATA)):
        n, _ = normalize(num_msp[i], den_msp[i], width, offset,
                          fallback_baseline=baseline_all)
        norms_msp.append(n)
    norms_mq = []
    for i in range(len(MQ_STRATA)):
        n, _ = normalize(num_mq[i], den_mq[i], width, offset,
                          fallback_baseline=baseline_all)
        norms_mq.append(n)

    # Decay fits on the overall profile
    if norm_all is not None:
        A, phi, tau = fit_decay(d_signed, norm_all)
        print(f'\nOverall decay fit: A={A:+.3f}  phi={phi:+.2f}  tau={tau:.1f} bp')
    print('\nPer-stratum decay fits (MSP size):')
    for i, (lo, hi, label, _) in enumerate(MSP_STRATA):
        if norms_msp[i] is None:
            continue
        A, phi, tau = fit_decay(d_signed, norms_msp[i])
        max_d = min(80, hi - 10)
        A2, phi2, tau2 = fit_decay(d_signed, norms_msp[i], d_hi=max_d)
        print(f'  {label}  A={A2:+.3f}  phi={phi2:+.2f}  '
              f'tau={tau2:.1f} bp')
    print('\nPer-stratum decay fits (nuc mq):')
    for i, (lo, hi, label, _) in enumerate(MQ_STRATA):
        if norms_mq[i] is None:
            continue
        A, phi, tau = fit_decay(d_signed, norms_mq[i])
        print(f'  {label}  A={A:+.3f}  phi={phi:+.2f}  tau={tau:.1f} bp')

    # ---- plot ----
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel (0,0): overall profile with fit
    ax = axes[0, 0]
    if norm_all is not None:
        ax.plot(d_signed, norm_all, color='#1e3a8a', linewidth=1.5,
                 label=f'all anchors (n={n_reads} reads)')
        A, phi, tau = fit_decay(d_signed, norm_all)
        if not np.isnan(A):
            d_fit = np.linspace(5, 120, 300)
            y_fit = 1 + A * np.cos(2 * np.pi * d_fit / 10.4 + phi) * np.exp(-d_fit / tau)
            ax.plot(d_fit, y_fit, '--', color='#dc2626', linewidth=1.2,
                     label=f'fit: A={A:.2f}, τ={tau:.0f} bp')
    # Grey-out the d < 0 selection-bias zone (first-hit anchor makes
    # d just below 0 artifactually low — if there were a hit there,
    # it would have been used as the anchor instead)
    ax.axvspan(-DIST_INTO_NUC, 0, color='#94a3b8', alpha=0.15,
                label='d<0: selection-biased\n(see panel D for\n internal rate)')
    ax.axhline(1.0, color='k', linewidth=0.6)
    ax.axvline(0, color='#dc2626', linewidth=1.1)
    for d in (10.4, 20.8, 31.2, 41.6, 52.0, 62.4):
        ax.axvline(d, color='#888', linestyle=':', linewidth=0.4)
        ax.axvline(-d, color='#888', linestyle=':', linewidth=0.4)
    ax.set_xlabel('signed distance from first-hit anchor (bp)')
    ax.set_ylabel('hit rate / baseline')
    ax.set_title('A. All anchors — overall decay profile', fontsize=11)
    ax.set_xlim(-DIST_INTO_NUC, DIST_INTO_MSP)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc='best')

    # Panel (0,1): stratified by MSP size
    ax = axes[0, 1]
    for i, (lo, hi, label, color) in enumerate(MSP_STRATA):
        if norms_msp[i] is None:
            continue
        y = norms_msp[i]
        max_positive = min(DIST_INTO_MSP, hi - 10)
        plot_from = 0
        plot_to = offset + max_positive
        ax.plot(d_signed[plot_from:plot_to], y[plot_from:plot_to],
                 color=color, linewidth=1.5, label=label)
    ax.axhline(1.0, color='k', linewidth=0.6)
    ax.axvline(0, color='#dc2626', linewidth=1.1)
    for d in (10.4, 20.8, 31.2, 41.6, 52.0, 62.4):
        ax.axvline(d, color='#888', linestyle=':', linewidth=0.4)
        ax.axvline(-d, color='#888', linestyle=':', linewidth=0.4)
    ax.set_xlabel('signed distance from first-hit anchor (bp)')
    ax.set_ylabel('hit rate / baseline')
    ax.set_title('B. Stratified by MSP length', fontsize=11)
    ax.set_xlim(-DIST_INTO_NUC, DIST_INTO_MSP)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc='best')

    # Panel (1,0): stratified by nuc mq
    ax = axes[1, 0]
    for i, (lo, hi, label, color) in enumerate(MQ_STRATA):
        if norms_mq[i] is None:
            continue
        y = norms_mq[i]
        ax.plot(d_signed, y, color=color, linewidth=1.5, label=label)
    ax.axhline(1.0, color='k', linewidth=0.6)
    ax.axvline(0, color='#dc2626', linewidth=1.1)
    for d in (10.4, 20.8, 31.2, 41.6, 52.0, 62.4):
        ax.axvline(d, color='#888', linestyle=':', linewidth=0.4)
        ax.axvline(-d, color='#888', linestyle=':', linewidth=0.4)
    ax.set_xlabel('signed distance from first-hit anchor (bp)')
    ax.set_ylabel('hit rate / baseline')
    ax.set_title('C. Stratified by nuc confidence (mq)', fontsize=11)
    ax.set_xlim(-DIST_INTO_NUC, DIST_INTO_MSP)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc='best')

    # Panel (1,1): absolute hit rate inside nuc body (not baseline-normalized)
    # — lets us see if low-mq nucs actually have internal deaminations
    ax = axes[1, 1]
    for i, (lo, hi, label, color) in enumerate(MQ_STRATA):
        if den_mq[i].sum() == 0:
            continue
        # absolute rate (not normalized)
        rate = np.where(den_mq[i] > 0,
                         num_mq[i] / np.maximum(den_mq[i], 1), 0.0)
        ax.plot(d_signed, rate, color=color, linewidth=1.5,
                 label=label)
    ax.axhline(0, color='k', linewidth=0.6)
    ax.axvline(0, color='#dc2626', linewidth=1.1)
    ax.set_xlabel('signed distance from first-hit anchor (bp)')
    ax.set_ylabel('ABSOLUTE hit rate (num / den)')
    ax.set_title('D. Absolute hit rate by mq (internal deaminations)',
                  fontsize=11)
    ax.set_xlim(-DIST_INTO_NUC, DIST_INTO_MSP)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc='best')

    fig.suptitle(
        f'scDAF PS00758 — first-hit-anchored rotational profile  (n={n_reads})',
        fontsize=12, y=1.00,
    )
    out = os.path.join(HERE, 'output', 'periodicity_anchored.png')
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'\nwrote {out}')


if __name__ == '__main__':
    main()
