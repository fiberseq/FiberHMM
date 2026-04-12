"""Pair-correlation analysis of enzyme periodicity across DAF-seq and Hia5.

Reproduces the 10.4 bp periodicity observation from
/Users/tt7739/Dropbox/Fiber-NET-seq/Spatial_DAF-seq/analysis/deamination_rates/periodicity.py
and extends it to DddA datasets and Hia5 naked-DNA data, to ask:

  - Does DddB hit one face of the helix? (~10.4 bp periodicity)
  - Does DddA show the same periodicity?
  - Does Hia5 (a totally different enzyme, m6A not C->T) show it?
  - On naked DNA (no chromatin), does the periodicity persist?
      → If YES: the 10.4 bp signal is intrinsic to the enzyme's
        binding geometry, NOT nucleosome-mediated.
      → If NO:  the signal is nucleosome phasing on chromatin.

For each read we extract the REFERENCE positions of every
opportunity (C on DAF forward, G on DAF reverse, A/T on Hia5) and
mark which got "hit" (converted for DAF, m6A>threshold for Hia5).
The pair_rate(d) = P(both hit | d bp apart) / long-range baseline
gives an enrichment curve that makes short-range periodicity
visible on top of any overall hit density difference.

Output: bench/output/periodicity_compare.png — three-panel figure
showing 0-60 bp zoom (enzyme-face signal), full 0-400 bp (nucleosome
signals), and an FFT periodogram.
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

from enzyme_extractors import Hia5Extractor  # noqa: E402


MAX_LAG = 400
MAX_READS = 2000
MIN_ALIGN = 1000
MIN_HIT_RATE = 0.005


# ---- position extractors ------------------------------------------

def daf_positions(read):
    """Reference-coord (opp_positions, hit_positions) for DAF-seq.

    Handles three sequence encodings:
      1. Raw C→T (DddB spacetime): C in ref + T in read → hit
      2. Raw G→A reverse: G in ref + A in read → hit (reverse strand)
      3. IUPAC-encoded (NAPA, ENH30, etc.): Y in read at C-in-ref, or
         R in read at G-in-ref — per-read majority vote picks the
         active encoding.
    """
    try:
        pairs = read.get_aligned_pairs(with_seq=True, matches_only=True)
    except ValueError:
        return None
    seq = read.query_sequence
    if seq is None:
        return None
    # Scan once, collect both C-opps and G-opps + all hit types, then
    # pick whichever encoding got the most hits.
    c_opps, y_hits, t_hits = [], [], []
    g_opps, r_hits, a_hits = [], [], []
    for qpos, rpos, rb in pairs:
        if rb is None or qpos is None:
            continue
        qb = seq[qpos].upper()
        rbu = rb.upper()
        if rbu == 'C':
            c_opps.append(rpos)
            if qb == 'Y':
                y_hits.append(rpos)
            elif qb == 'T':
                t_hits.append(rpos)
        elif rbu == 'G':
            g_opps.append(rpos)
            if qb == 'R':
                r_hits.append(rpos)
            elif qb == 'A':
                a_hits.append(rpos)

    # Which encoding is active on this read?
    counts = {
        'CtoY': (len(c_opps), len(y_hits)),
        'CtoT': (len(c_opps), len(t_hits)),
        'GtoR': (len(g_opps), len(r_hits)),
        'GtoA': (len(g_opps), len(a_hits)),
    }
    best_name = max(counts, key=lambda k: counts[k][1])
    n_opp, n_hit = counts[best_name]
    if n_hit == 0:
        return None
    if best_name == 'CtoY':
        return (np.asarray(c_opps, dtype=np.int64),
                np.asarray(y_hits, dtype=np.int64))
    if best_name == 'CtoT':
        return (np.asarray(c_opps, dtype=np.int64),
                np.asarray(t_hits, dtype=np.int64))
    if best_name == 'GtoR':
        return (np.asarray(g_opps, dtype=np.int64),
                np.asarray(r_hits, dtype=np.int64))
    return (np.asarray(g_opps, dtype=np.int64),
            np.asarray(a_hits, dtype=np.int64))


def hia5_positions_query(read, ml_threshold=128):
    """Query-coord (opp_positions, hit_positions) for Hia5 m6A.

    Parses MM/ML directly without requiring alignment — works on
    UNMAPPED reads (e.g. naked-DNA control) where we can't go through
    reference coords. Opportunities are query A's and T's; hits are
    positions with ML >= threshold for the 'a' (m6A) modification.
    """
    if not read.has_tag('MM') or not read.has_tag('ML'):
        return None
    mm_str = read.get_tag('MM')
    ml = list(read.get_tag('ML'))
    q = read.query_sequence
    if not q or not mm_str:
        return None

    # Opportunities: every A or T in the query
    opps = np.asarray([i for i, c in enumerate(q.upper()) if c in ('A', 'T')],
                        dtype=np.int64)

    # Parse MM sections
    hits = []
    ml_idx = 0
    for section in mm_str.rstrip(';').split(';'):
        if not section:
            continue
        parts = section.split(',')
        header = parts[0]
        if len(header) < 3:
            continue
        try:
            skips = [int(x) for x in parts[1:] if x]
        except ValueError:
            continue
        base = header[0]
        # strand = header[1]
        code = header[2]
        if code != 'a':
            ml_idx += len(skips)
            continue
        target_base = base.upper()
        qp_cursor = 0
        for skip in skips:
            if ml_idx >= len(ml):
                break
            score = ml[ml_idx]
            ml_idx += 1
            needed = skip + 1
            while qp_cursor < len(q) and needed > 0:
                if q[qp_cursor].upper() == target_base:
                    needed -= 1
                    if needed == 0:
                        break
                qp_cursor += 1
            if qp_cursor >= len(q):
                break
            if score >= ml_threshold:
                hits.append(qp_cursor)
            qp_cursor += 1
    return opps, np.asarray(sorted(set(hits)), dtype=np.int64)


# ---- pair histogram -----------------------------------------------

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


def process_sample(label, bams, enzyme, max_reads=MAX_READS):
    if enzyme == 'daf':
        pos_fn = daf_positions
        allow_unmapped = False
    elif enzyme == 'hia5':
        pos_fn = hia5_positions_query
        allow_unmapped = True   # query-coord Hia5 works on unmapped
    else:
        raise ValueError(enzyme)

    num = np.zeros(MAX_LAG + 1, dtype=np.int64)
    den = np.zeros(MAX_LAG + 1, dtype=np.int64)
    n_reads = 0
    for bam_path in bams:
        if not os.path.exists(bam_path):
            continue
        with pysam.AlignmentFile(bam_path, 'rb', check_sq=False) as bam:
            for read in bam.fetch(until_eof=True):
                if n_reads >= max_reads:
                    break
                if read.is_secondary or read.is_supplementary:
                    continue
                if read.is_unmapped and not allow_unmapped:
                    continue
                # Length filter — use query_length for unmapped
                qlen = (read.query_alignment_length or 0) if not read.is_unmapped else (read.query_length or 0)
                if qlen < MIN_ALIGN:
                    continue
                rp = pos_fn(read)
                if rp is None:
                    continue
                opps, hits = rp
                if opps.size < 20 or hits.size < 3:
                    continue
                if hits.size / opps.size < MIN_HIT_RATE:
                    continue
                den += pair_hist(opps, MAX_LAG)
                num += pair_hist(hits, MAX_LAG)
                n_reads += 1
        if n_reads >= max_reads:
            break
    pair_rate = np.where(den > 0, num / np.maximum(den, 1), 0.0)
    print(f'  {label}: {n_reads} reads', flush=True)
    return n_reads, pair_rate


# ---- samples ------------------------------------------------------

def sample_config():
    """Return list of (label, enzyme, bam_list, color)."""
    samples = []

    # DddB — chromatinized spacetime pooled (reproducing original obs)
    dddb_dir = ('/Users/tt7739/Dropbox/Fiber-NET-seq/Drosophila_phase2/'
                 'Datasets/DAF-seq/spacetime/combined_bam')
    dddb_windows = ['1-1.5', '1.5-2', '2-2.5', '2.5-3', '3-3.5', '3.5-4', '4-4.5']
    dddb_bams = [os.path.join(dddb_dir, f'{w}.sorted.bam') for w in dddb_windows]
    dddb_bams = [p for p in dddb_bams if os.path.exists(p)]
    if dddb_bams:
        samples.append(('DddB spacetime (chromatin)', 'daf', dddb_bams, '#4c78a8'))

    # Naked DddB — the key calibration set for iter-17 rotational
    # correction. Under-deaminated so split by percentile; 10pct is
    # the most heavily deaminated decile, 20pct the top quintile.
    # No chromatin → any oscillation is enzyme-intrinsic.
    naked_dddb_dir = os.path.join(os.path.dirname(HERE), 'data', 'dddb')
    naked_dddb_10 = os.path.join(naked_dddb_dir, 'naked_DNA_10pct.bam')
    naked_dddb_20 = os.path.join(naked_dddb_dir, 'naked_DNA_20pct.bam')
    if os.path.exists(naked_dddb_10):
        samples.append(('DddB naked (top 10%)', 'daf', [naked_dddb_10], '#1e3a8a'))
    if os.path.exists(naked_dddb_20):
        samples.append(('DddB naked (top 20%)', 'daf', [naked_dddb_20], '#3b82f6'))

    # DddA — a few representative datasets
    ddda_root = os.path.join(os.path.dirname(HERE), 'data', 'ddda')
    # NAPA (amplicon)
    napa = ('/Users/tt7739/Dropbox/Fiber-NET-seq/DAF-seq/Data/bam/for eitan/'
             'NAPA_PS00626_haplotype_corrected.bam')
    if os.path.exists(napa):
        samples.append(('DddA NAPA PS00626', 'daf', [napa], '#e45756'))
    # scDAF (single-cell whole-genome)
    scdaf = ('/Users/tt7739/Dropbox/Fiber-NET-seq/DAF-seq/Data/bam/'
              'PS00758_consensus_GA_HG38_corrected.haplotagged.bam')
    if os.path.exists(scdaf):
        samples.append(('DddA scDAF PS00758', 'daf', [scdaf], '#f28e2b'))
    # ENH30 (amplicon)
    enh = os.path.join(ddda_root, 'PCR1_ENH30_DMSO16.mapped.consensus.decorated.bam')
    if os.path.exists(enh):
        samples.append(('DddA ENH30 amplicon', 'daf', [enh], '#59a14f'))

    # Hia5 naked DNA (no chromatin, no phasing possible)
    hia5_naked = os.path.join(os.path.dirname(HERE), 'data', 'naked-dna_1.m6a.bam')
    if os.path.exists(hia5_naked):
        samples.append(('Hia5 naked DNA', 'hia5', [hia5_naked], '#b475cf'))

    # Hia5 chromatinized (Drosophila embryo 2-4hr, fly nuc/TF
    # landscape). The critical control: if chromatinized Hia5 also
    # shows a 10 bp peak but naked Hia5 does NOT, then bulk-sample
    # nucleosome phasing is sufficient to produce the periodicity on
    # its own — which would mean the DAF 10 bp signal is likely
    # chromatin-driven rather than enzyme-intrinsic.
    hia5_chrom = os.path.join(os.path.dirname(HERE), 'data',
                                'test_hia5_2-4hr_sna_eve_ftz.bam')
    if os.path.exists(hia5_chrom):
        samples.append(('Hia5 chromatinized (fly 2-4hr)', 'hia5',
                         [hia5_chrom], '#6b21a8'))

    return samples


def main():
    samples = sample_config()
    print(f'samples: {[s[0] for s in samples]}')
    results = {}
    for label, enzyme, bams, color in samples:
        print(f'\nprocessing {label} ({enzyme})...')
        n_reads, pair_rate = process_sample(label, bams, enzyme)
        if n_reads == 0:
            continue
        results[label] = (n_reads, pair_rate, color)

    if not results:
        print('no data')
        return

    # Normalize each to long-range baseline
    norms = {}
    for label, (n_reads, pair_rate, color) in results.items():
        pr = pair_rate.copy()
        baseline = np.mean(pr[300:MAX_LAG + 1])
        if baseline <= 0:
            baseline = np.mean(pr[pr > 0]) or 1.0
        norms[label] = (n_reads, pr / baseline, color)

    def smooth(x, k=3):
        if k <= 1:
            return x
        kernel = np.ones(k) / k
        return np.convolve(x, kernel, mode='same')

    fig = plt.figure(figsize=(16, 5.5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1.2, 1.2])
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])

    # Panel A: 0-60 bp zoom (one-face enzyme access)
    for label, (n_reads, y, color) in norms.items():
        ys = smooth(y, 1)
        ax1.plot(np.arange(len(ys))[2:60], ys[2:60],
                  color=color, linewidth=1.5,
                  label=f'{label}  n={n_reads}')
    ax1.set_title('A. 0-60 bp — enzyme-face access (~10.4 bp)', fontsize=10)
    ax1.set_xlabel('pairwise distance (bp)')
    ax1.set_ylabel('enrichment (pair_rate / baseline)')
    for d in (10.4, 20.8, 31.2, 41.6, 52.0):
        ax1.axvline(d, color='#888', linestyle=':', linewidth=0.7)
    ax1.axhline(1.0, color='k', linewidth=0.6)
    ax1.grid(alpha=0.25)
    ax1.legend(fontsize=7, loc='best', frameon=True)

    # Panel B: 0-400 bp (nucleosome footprint + NRL)
    for label, (n_reads, y, color) in norms.items():
        ys = smooth(y, 3)
        ax2.plot(np.arange(len(ys))[2:], ys[2:],
                  color=color, linewidth=1.2,
                  label=f'{label}')
    ax2.axvspan(80, 140, color='#d6e9f5', alpha=0.5, label='80-140 bp (nuc protection)')
    ax2.axvspan(170, 200, color='#fceca5', alpha=0.5, label='175-200 bp (NRL)')
    ax2.axhline(1.0, color='k', linewidth=0.6)
    ax2.set_xlim(0, MAX_LAG)
    ax2.set_xlabel('pairwise distance (bp)')
    ax2.set_ylabel('enrichment')
    ax2.set_title('B. Full range — nucleosome footprint and repeat length',
                    fontsize=10)
    ax2.grid(alpha=0.25)
    ax2.legend(fontsize=7, loc='best', frameon=True)

    # Panel C: FFT of detrended pair_rate
    for label, (n_reads, pair_rate, color) in [
        (lbl, (nr, results[lbl][1], results[lbl][2]))
        for lbl, (nr, _, _) in results.items()
    ]:
        y = pair_rate[2:MAX_LAG + 1].copy()
        x = np.arange(len(y))
        if len(x) < 4 or np.all(y == 0):
            continue
        coeff = np.polyfit(x, y, 1)
        y_det = y - (coeff[0] * x + coeff[1])
        n_fft = 2048
        Y = np.abs(np.fft.rfft(y_det, n=n_fft))
        freqs = np.fft.rfftfreq(n_fft, d=1.0)
        periods = np.where(freqs > 0, 1.0 / np.maximum(freqs, 1e-9), np.inf)
        mask = (periods >= 4) & (periods <= 250)
        ax3.plot(periods[mask], Y[mask],
                  color=color, linewidth=1.3,
                  label=f'{label}')
    ax3.set_xscale('log')
    ax3.set_xlim(4, 250)
    ax3.set_xlabel('period (bp)')
    ax3.set_ylabel('|FFT|')
    ax3.set_title('C. Periodogram — 10 bp = helical, 180 bp = NRL',
                    fontsize=10)
    for p in (10.4, 180, 195):
        ax3.axvline(p, color='#888', linestyle=':', linewidth=0.7)
    ax3.grid(alpha=0.25, which='both')
    ax3.legend(fontsize=7, loc='best', frameon=True)

    fig.suptitle(
        'Pair-correlation periodicity: DddB vs DddA vs Hia5 naked DNA',
        fontsize=12, y=1.02,
    )
    out = os.path.join(HERE, 'output', 'periodicity_compare.png')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'\nwrote {out}')


if __name__ == '__main__':
    main()
