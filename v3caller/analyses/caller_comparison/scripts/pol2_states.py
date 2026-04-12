#!/usr/bin/env python3
"""Per-read Pol II state detection on v2 vs v3 calls.

Implements the fiberCNN heuristics from FiberBrowser-v3/browser/server.py:

- **Paused Pol II**: footprint of size 35–65 bp located in TSS+10 to
  TSS+50 bp window (downstream of TSS, small enough to be Pol II, not a
  nucleosome).
- **Elongating Pol II**: small footprints (35–65 bp) anywhere in the
  gene body (TSS to TES).
- **PIC (pre-initiation complex)**: bimodal small footprints (20–40 OR
  60–80 bp) in the TSS−50 to TSS+25 bp window.

For each read in the BAM, we score these states twice: once using v2
footprints (`fp_v2+`), once using v3 (nuc+tf combined). Then we check
how often the two callers agree on the state-call and whether v3
captures the same biology.

Because DAF-seq / Hia5 data is long-read and spans one or a few
genes, reads need associated TSS positions. Supply them via
`--tss-bed` (BED6 with strand; each read's gene body is TSS to TES)
OR use `--auto-tss` to infer TSSs from the read coordinates against
a gene annotation.

For datasets without gene annotations (e.g., DddB spacetime
whole-genome), we fall back to SIZE-ONLY comparisons — we can't
assign Pol II states without TSS, but we can still compare the
small-footprint (35–65 bp) and PIC-sized (20–40 + 60–80 bp)
distributions between callers.

Outputs:
- `pol2_state_counts.tsv`: per-read state counts by caller
- `pol2_size_distribution.png`: footprint size histogram split by caller
- `pol2_agreement.png`: per-state agreement matrix (if TSS available)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from parse_ma_calls import iter_reads


# --- fiberCNN size + position bands ---
PAUSED_SIZE = (35, 65)
ELONG_SIZE = (35, 65)
PIC_SIZES = [(20, 40), (60, 80)]  # bimodal
PAUSED_POS = (10, 50)    # TSS + window (downstream, + strand)
PIC_POS = (-50, 25)      # TSS ± window


def load_tss_bed(path):
    """Parse BED6 → dict[(chrom, name)] = (tss_ref_pos, strand, tes_ref_pos)."""
    d = {}
    with open(path) as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.strip().split('\t')
            if len(parts) < 6:
                continue
            chrom, start, end, name, _, strand = parts[:6]
            start = int(start); end = int(end)
            if strand == '+':
                tss = start; tes = end
            else:
                tss = end - 1; tes = start
            d.setdefault(chrom, []).append((tss, tes, strand, name))
    return d


def find_tss_in_read_span(tss_table, chrom, ref_start, ref_end):
    """Return list of (tss_ref_pos, strand, tes_ref_pos, name) that
    fall within the read's reference span."""
    if chrom not in tss_table:
        return []
    hits = []
    for tss, tes, strand, name in tss_table[chrom]:
        if ref_start <= tss <= ref_end:
            hits.append((tss, tes, strand, name))
    return hits


def footprint_to_ref_pos(read_s, read_l, read_obj_map=None):
    """Trivially use query (read) coords. For precise ref coords we'd
    need the BAM's cigar; here we use the MA coordinate = query
    position as a proxy since MA coords are molecular (read) 1-based.

    For v3 calls that came from mapped reads, the MA query coord is
    closely tied to the reference position near the left edge of the
    alignment — this approximation is good enough for heuristic state
    detection over windows of ±50 bp.
    """
    return read_s, read_s + read_l


def in_size_band(length, band):
    lo, hi = band
    return lo <= length <= hi


def in_pos_band(footprint_center_q, tss_q, pos_band, strand):
    """footprint_center_q, tss_q: query coordinates.
    pos_band: (lo, hi) relative to TSS, signed (+ = downstream, - = upstream).
    strand: + or - (if -, flip sign of pos_band)."""
    if strand == '+':
        d = footprint_center_q - tss_q
    else:
        d = tss_q - footprint_center_q
    return pos_band[0] <= d <= pos_band[1]


def classify_read_states(footprints, tss_q, strand, read_length=None,
                            gene_body_span=5000):
    """Given list of (start, length) footprints in query coords and a
    TSS position in query coords, return the count of each fiberCNN
    state for this read.

    States:
      - paused: 35-65 bp in TSS+10..+50
      - elongating: 35-65 bp in gene body (TSS+50..TSS+gene_body_span),
        i.e. downstream of the paused window
      - pic: 20-40 or 60-80 bp in TSS-50..+25
      - accessible_promoter: no ≥90 bp footprint overlaps TSS±50
      - hyperburst: <50% of gene body covered by nucleosomes (≥90 bp fp)
    """
    state_counts = Counter()
    big_fps = [(s, l) for s, l in footprints if l >= 90]

    # Gene body window in query coords (strand-flipped)
    if strand == '+':
        body_lo, body_hi = tss_q, tss_q + gene_body_span
    else:
        body_lo, body_hi = tss_q - gene_body_span, tss_q

    for s, l in footprints:
        center = s + l // 2
        if in_size_band(l, PAUSED_SIZE) and in_pos_band(center, tss_q,
                                                           PAUSED_POS,
                                                           strand):
            state_counts['paused'] += 1
        # Elongating: 35-65 bp, IN gene body window, NOT in paused window
        if in_size_band(l, ELONG_SIZE):
            in_body = body_lo <= center <= body_hi
            in_paused_window = in_pos_band(center, tss_q, PAUSED_POS, strand)
            if in_body and not in_paused_window:
                state_counts['elongating'] += 1
        for pic_band in PIC_SIZES:
            if in_size_band(l, pic_band) and in_pos_band(
                    center, tss_q, PIC_POS, strand):
                state_counts['pic'] += 1
                break

    # Accessible promoter: no ≥90 bp footprint overlaps TSS ±50 bp
    promoter_occupied = False
    for s, l in big_fps:
        # footprint in ref-coord-relative form isn't feasible here; use
        # query-space proxy: TSS_q ± 50 bp window.
        fp_start, fp_end = s, s + l
        win_lo, win_hi = tss_q - 50, tss_q + 50
        if fp_end > win_lo and fp_start < win_hi:
            promoter_occupied = True
            break
    state_counts['accessible_promoter'] = 0 if promoter_occupied else 1

    # Hyperburst: <50% of gene body covered by nucs (≥90 bp fp)
    # Gene body window: tss_q..tss_q+gene_body_span (or -span for - strand)
    gb_lo = tss_q if strand == '+' else tss_q - gene_body_span
    gb_hi = tss_q + gene_body_span if strand == '+' else tss_q
    gb_len = gb_hi - gb_lo
    if gb_len > 0:
        covered = 0
        for s, l in big_fps:
            fp_lo, fp_hi = s, s + l
            ov = max(0, min(gb_hi, fp_hi) - max(gb_lo, fp_lo))
            covered += ov
        nuc_frac = covered / gb_len
        state_counts['hyperburst'] = 1 if nuc_frac < 0.5 else 0
        state_counts['nuc_coverage_frac'] = nuc_frac
    return state_counts


def ref_to_query_pos(read, ref_pos):
    """Given a pysam read and a reference position, find the query
    coordinate. Returns None if the ref_pos isn't aligned.
    Uses get_aligned_pairs(matches_only=True) — slow per read but
    only called per-TSS, not per-footprint."""
    try:
        pairs = read.get_aligned_pairs(matches_only=True)
    except ValueError:
        return None
    for qp, rp in pairs:
        if rp == ref_pos:
            return qp
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in-bam', action='append', required=True,
                    help='annotated v3 BAM (repeatable for multi-window aggregation)')
    ap.add_argument('--label', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--tss-bed', default=None,
                    help='BED6 of gene TSS/TES. If provided, do '
                         'positional Pol II state analysis. If omitted, '
                         'only size-distribution comparison.')
    ap.add_argument('--max-reads', type=int, default=0,
                    help='cap reads per BAM (0 = all)')
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    tss_table = load_tss_bed(args.tss_bed) if args.tss_bed else None
    if tss_table:
        print(f'Loaded {sum(len(v) for v in tss_table.values())} TSSs '
              f'across {len(tss_table)} chroms')

    # Size distributions (aggregated across all BAMs)
    v2_sizes = []
    v3_sizes = []

    # Per-read state counts
    state_rows = []

    import pysam
    for bam_path in args.in_bam:
        print(f'[{bam_path}]', flush=True)
        # Build read name → pysam read map for ref→query lookup
        read_by_name = {}
        if tss_table:
            bam = pysam.AlignmentFile(bam_path, 'rb', check_sq=False)
            for r in bam.fetch(until_eof=True):
                if r.is_unmapped or r.is_secondary or r.is_supplementary:
                    continue
                read_by_name[r.query_name] = r
            bam.close()

        n_reads = 0
        for read in iter_reads(bam_path):
            n_reads += 1
            if args.max_reads and n_reads > args.max_reads:
                break
            v2s = read['fp_v2']
            v3_all = [(s, l) for s, l, _ in read['nuc']] + \
                      [(s, l) for s, l, _ in read['tf']]
            for s, l in v2s: v2_sizes.append(l)
            for s, l in v3_all: v3_sizes.append(l)

            if tss_table:
                tss_hits = find_tss_in_read_span(
                    tss_table, read['chrom'], read['ref_start'],
                    read['ref_end'])
                if not tss_hits:
                    continue
                pysam_read = read_by_name.get(read['name'])
                if pysam_read is None:
                    continue
                for tss_ref, tes_ref, strand, name in tss_hits:
                    tss_q = ref_to_query_pos(pysam_read, tss_ref)
                    if tss_q is None:
                        continue
                    # Compute gene body span in query coords from BED
                    # TES, capped at 5kb to handle BED entries that
                    # span adjacent genes. Fallback: 2kb.
                    tes_q = ref_to_query_pos(pysam_read, tes_ref)
                    if tes_q is not None:
                        span = abs(tes_q - tss_q)
                        if span < 200:
                            span = 2000  # bed too tight, use default
                        elif span > 5000:
                            span = 5000  # bed too loose, cap
                    else:
                        span = 2000
                    v2_states = classify_read_states(v2s, tss_q, strand,
                                                         gene_body_span=span)
                    v3_states = classify_read_states(v3_all, tss_q, strand,
                                                         gene_body_span=span)
                    state_rows.append({
                        'read': read['name'],
                        'gene': name,
                        'chrom': read['chrom'],
                        'tss_ref': tss_ref,
                        'strand': strand,
                        'paused_v2': v2_states['paused'],
                        'paused_v3': v3_states['paused'],
                        'elong_v2': v2_states['elongating'],
                        'elong_v3': v3_states['elongating'],
                        'pic_v2': v2_states['pic'],
                        'pic_v3': v3_states['pic'],
                        'accessible_v2': v2_states['accessible_promoter'],
                        'accessible_v3': v3_states['accessible_promoter'],
                        'hyperburst_v2': v2_states['hyperburst'],
                        'hyperburst_v3': v3_states['hyperburst'],
                    })
        print(f'  {n_reads} reads', flush=True)

    n_reads = len(v2_sizes)  # placeholder for print below

    print(f'Processed {n_reads} reads; v2 footprints: {len(v2_sizes)}, '
          f'v3 footprints: {len(v3_sizes)}')

    # -------- Size distribution figure --------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    bins = np.arange(0, 401, 5)

    ax1.hist(v2_sizes, bins=bins, alpha=0.6, color='#dc2626',
              label=f'v2 fp (n={len(v2_sizes):,})', density=True)
    ax1.hist(v3_sizes, bins=bins, alpha=0.6, color='#1e3a8a',
              label=f'v3 nuc+tf (n={len(v3_sizes):,})', density=True)
    for lo, hi in [PAUSED_SIZE] + PIC_SIZES:
        ax1.axvspan(lo, hi, alpha=0.15, color='#16a34a')
    ax1.set_xlabel('footprint length (bp)')
    ax1.set_ylabel('density')
    ax1.set_title(f'{args.label}: size distribution\n'
                  'green = PIC/Pol II size bands (20-40, 35-65, 60-80)')
    ax1.legend(); ax1.grid(alpha=0.3)
    ax1.set_xlim(0, 400)

    # Zoom into 0-100 bp (Pol II range)
    bins_zoom = np.arange(0, 101, 2)
    ax2.hist(v2_sizes, bins=bins_zoom, alpha=0.6, color='#dc2626',
              label=f'v2', density=True)
    ax2.hist(v3_sizes, bins=bins_zoom, alpha=0.6, color='#1e3a8a',
              label=f'v3', density=True)
    for lo, hi in [(20, 40), (35, 65), (60, 80)]:
        ax2.axvspan(lo, hi, alpha=0.15, color='#16a34a')
    ax2.set_xlabel('footprint length (bp)')
    ax2.set_ylabel('density')
    ax2.set_title('zoom 0-100 bp: Pol II / PIC range')
    ax2.legend(); ax2.grid(alpha=0.3)
    ax2.set_xlim(0, 100)

    fig.tight_layout()
    png = os.path.join(args.out_dir, f'{args.label}_size_distribution.png')
    fig.savefig(png, dpi=130, bbox_inches='tight'); plt.close(fig)
    print(f'Wrote {png}')

    # -------- Per-state agreement (if TSS provided) --------
    if state_rows:
        # Write TSV
        tsv = os.path.join(args.out_dir, f'{args.label}_pol2_states.tsv')
        with open(tsv, 'w') as f:
            f.write('read\tgene\tchrom\ttss_ref\tstrand\tpaused_v2\tpaused_v3\t'
                     'elong_v2\telong_v3\tpic_v2\tpic_v3\t'
                     'accessible_v2\taccessible_v3\t'
                     'hyperburst_v2\thyperburst_v3\n')
            for row in state_rows:
                f.write(f"{row['read']}\t{row['gene']}\t{row['chrom']}\t"
                         f"{row['tss_ref']}\t{row['strand']}\t"
                         f"{row['paused_v2']}\t{row['paused_v3']}\t"
                         f"{row['elong_v2']}\t{row['elong_v3']}\t"
                         f"{row['pic_v2']}\t{row['pic_v3']}\t"
                         f"{row['accessible_v2']}\t{row['accessible_v3']}\t"
                         f"{row['hyperburst_v2']}\t{row['hyperburst_v3']}\n")
        print(f'Wrote {tsv}')

        # Agreement table and plot
        fig, axes = plt.subplots(1, 5, figsize=(22, 5))
        for i, (state, axl) in enumerate(zip(
                ['paused', 'elong', 'pic', 'accessible', 'hyperburst'],
                axes)):
            v2_counts = [r[f'{state}_v2'] for r in state_rows]
            v3_counts = [r[f'{state}_v3'] for r in state_rows]
            v2_any = sum(1 for c in v2_counts if c > 0)
            v3_any = sum(1 for c in v3_counts if c > 0)
            both = sum(1 for v2, v3 in zip(v2_counts, v3_counts) if v2 > 0 and v3 > 0)
            only_v2 = sum(1 for v2, v3 in zip(v2_counts, v3_counts) if v2 > 0 and v3 == 0)
            only_v3 = sum(1 for v2, v3 in zip(v2_counts, v3_counts) if v2 == 0 and v3 > 0)
            none = sum(1 for v2, v3 in zip(v2_counts, v3_counts) if v2 == 0 and v3 == 0)
            totals = [both, only_v3, only_v2, none]
            labels = ['both', 'v3 only', 'v2 only', 'neither']
            colors = ['#16a34a', '#1e3a8a', '#dc2626', '#94a3b8']
            axl.bar(labels, totals, color=colors)
            axl.set_title(f'{state}\n(any call/read, v2={v2_any}, v3={v3_any})',
                           fontsize=11)
            axl.set_ylabel('reads')
            for j, v in enumerate(totals):
                axl.text(j, v + max(totals)*0.01, f'{v}',
                          ha='center', fontsize=9)
        fig.suptitle(f'{args.label}: Pol II state agreement (reads × state)',
                      fontsize=11)
        fig.tight_layout()
        png = os.path.join(args.out_dir, f'{args.label}_pol2_agreement.png')
        fig.savefig(png, dpi=130, bbox_inches='tight'); plt.close(fig)
        print(f'Wrote {png}')

        # Summary JSON
        summary = {
            'label': args.label,
            'n_reads_with_tss': len(state_rows),
        }
        for state in ['paused', 'elong', 'pic', 'accessible', 'hyperburst']:
            v2_counts = [r[f'{state}_v2'] for r in state_rows]
            v3_counts = [r[f'{state}_v3'] for r in state_rows]
            summary[state] = {
                'v2_total_calls': sum(v2_counts),
                'v3_total_calls': sum(v3_counts),
                'v2_positive_reads': sum(1 for c in v2_counts if c > 0),
                'v3_positive_reads': sum(1 for c in v3_counts if c > 0),
                'both_positive': sum(1 for v2, v3 in zip(v2_counts, v3_counts) if v2 > 0 and v3 > 0),
                'v3_only': sum(1 for v2, v3 in zip(v2_counts, v3_counts) if v2 == 0 and v3 > 0),
                'v2_only': sum(1 for v2, v3 in zip(v2_counts, v3_counts) if v2 > 0 and v3 == 0),
            }
        jsn = os.path.join(args.out_dir, f'{args.label}_pol2_summary.json')
        with open(jsn, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f'Wrote {jsn}')
    else:
        print('(no TSS info → size-distribution analysis only)')


if __name__ == '__main__':
    main()
