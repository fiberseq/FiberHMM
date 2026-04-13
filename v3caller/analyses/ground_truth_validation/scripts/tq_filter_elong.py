#!/usr/bin/env python3
"""Sweep TF-quality (tq) threshold and measure sna elong fraction
in WT vs Dl- v3 output.

Hypothesis: v3 emits too many low-tq TF footprints that aren't
transcription-specific. Filtering to high-tq TFs should:
  - Drop v3 elong rate in WT (fewer spurious calls)
  - Drop v3 elong rate MORE in Dl- (true Pol II was scarce)
  - Result: Dl- drop should scale up as threshold increases

Uses the MA + AQ tags directly to recover tq per v3 TF call.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pysam

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   '..', '..', 'caller_comparison', 'scripts'))
from parse_ma_calls import parse_ma


# Elongating criterion: 35-65 bp TF-sized footprint in gene body
ELONG_MIN, ELONG_MAX = 35, 65
PAUSED_LO, PAUSED_HI = 10, 50  # TSS offset window (strand-flipped)


def ref_to_query_pos(read, ref_pos):
    try:
        pairs = read.get_aligned_pairs(matches_only=True)
    except ValueError:
        return None
    for qp, rp in pairs:
        if rp == ref_pos: return qp
    return None


def extract_tfs_with_tq(ma_str, aq_array):
    """Parse MA string + AQ array, return list of (s, l, tq) for
    v3 tf+QQQ entries."""
    parsed = parse_ma(ma_str)
    tfs_with_q = []
    aq_list = list(aq_array) if aq_array is not None else []
    idx = 0
    for name, strand, qspec, intervals in parsed['raw']:
        n_q = len(qspec)
        for s, l in intervals:
            if name == 'tf' and n_q >= 1:
                tq = aq_list[idx] if idx < len(aq_list) else 0
                tfs_with_q.append((s, l, tq))
            idx += n_q * 1  # each interval consumes n_q values
            # Actually idx should advance per-annotation; recompute
    # Simpler re-parse: rebuild in MA order
    out = []
    idx = 0
    for name, strand, qspec, intervals in parsed['raw']:
        n_q = len(qspec)
        for s, l in intervals:
            vals = aq_list[idx:idx + n_q]
            idx += n_q
            if name == 'tf' and n_q >= 1:
                out.append((s, l, vals[0]))
    return out


def count_elong(tfs, tss_q, strand, gene_body_span=2000, tq_min=0):
    """Count TF entries that are 35-65 bp, in gene body, and have
    tq >= tq_min."""
    if strand == '+':
        body_lo, body_hi = tss_q, tss_q + gene_body_span
    else:
        body_lo, body_hi = tss_q - gene_body_span, tss_q
    n = 0
    for s, l, tq in tfs:
        if not (ELONG_MIN <= l <= ELONG_MAX): continue
        if tq < tq_min: continue
        center = s + l // 2
        if not (body_lo <= center <= body_hi): continue
        # Exclude paused window
        if strand == '+':
            if PAUSED_LO <= (center - tss_q) <= PAUSED_HI: continue
        else:
            if PAUSED_LO <= (tss_q - center) <= PAUSED_HI: continue
        n += 1
    return n


def process_bam(bam_paths, tss_table, tq_thresholds):
    """Returns: {(gene, threshold): [elong_counts_per_read]}"""
    per_cond = {}
    for bam_path in bam_paths:
        bam = pysam.AlignmentFile(bam_path, 'rb', check_sq=False)
        for r in bam:
            if r.is_unmapped or r.is_secondary or r.is_supplementary:
                continue
            if not r.has_tag('MA'): continue
            chrom = r.reference_name
            if chrom not in tss_table: continue
            # Find TSSs within read span
            for tss_ref, tes_ref, strand, name in tss_table[chrom]:
                if not (r.reference_start <= tss_ref <= r.reference_end):
                    continue
                tss_q = ref_to_query_pos(r, tss_ref)
                if tss_q is None: continue
                ma_str = r.get_tag('MA')
                aq = r.get_tag('AQ') if r.has_tag('AQ') else []
                tfs = extract_tfs_with_tq(ma_str, aq)
                for th in tq_thresholds:
                    n = count_elong(tfs, tss_q, strand, tq_min=th)
                    per_cond.setdefault((name, th), []).append(n)
        bam.close()
    return per_cond


def load_tss_bed(path):
    d = {}
    with open(path) as f:
        for line in f:
            if line.startswith('#') or not line.strip(): continue
            parts = line.rstrip('\n').split('\t')
            chrom, start, end, name, _, strand = parts[:6]
            if strand == '+': tss, tes = int(start), int(end)
            else: tss, tes = int(end) - 1, int(start)
            d.setdefault(chrom, []).append((tss, tes, strand, name))
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--wt-bam', action='append', required=True)
    ap.add_argument('--dl-bam', action='append', required=True)
    ap.add_argument('--tss-bed', required=True)
    ap.add_argument('--gene', default='sna')
    ap.add_argument('--thresholds', type=int, nargs='+',
                    default=[0, 20, 40, 60, 80, 100, 120, 150])
    args = ap.parse_args()

    tss_table = load_tss_bed(args.tss_bed)
    print(f'WT BAMs: {args.wt_bam}')
    print(f'Dl- BAMs: {args.dl_bam}')
    print(f'TF quality thresholds: {args.thresholds}')

    print('\nProcessing WT...')
    wt_counts = process_bam(args.wt_bam, tss_table, args.thresholds)
    print('Processing Dl-...')
    dl_counts = process_bam(args.dl_bam, tss_table, args.thresholds)

    gene = args.gene
    print(f'\n=== sna elong rate vs tq threshold ===')
    print(f'{"tq_min":>6s}  {"WT_%pos":>7s}  {"Dl-_%pos":>7s}  '
          f'{"WT_mean":>8s}  {"Dl-_mean":>8s}  {"drop%":>7s}  '
          f'{"n_WT":>5s}  {"n_Dl-":>5s}')
    for th in args.thresholds:
        wt_arr = np.array(wt_counts.get((gene, th), []))
        dl_arr = np.array(dl_counts.get((gene, th), []))
        if len(wt_arr) == 0 or len(dl_arr) == 0:
            print(f'{th:>6d}  (no data)')
            continue
        wt_pos = (wt_arr > 0).mean() * 100
        dl_pos = (dl_arr > 0).mean() * 100
        wt_mean = wt_arr.mean()
        dl_mean = dl_arr.mean()
        drop = (wt_pos - dl_pos) / max(wt_pos, 0.01) * 100
        print(f'{th:>6d}  {wt_pos:>6.1f}%  {dl_pos:>6.1f}%  '
              f'{wt_mean:>8.2f}  {dl_mean:>8.2f}  {drop:>6.1f}%  '
              f'{len(wt_arr):>5d}  {len(dl_arr):>5d}')


if __name__ == '__main__':
    main()
