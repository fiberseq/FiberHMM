"""scDAF TSS meta-profile — aggregated nucleosome occupancy centered
on TSS positions.

Uses:
- `DAF-C/Reference/hg38.gtf` (NCBI RefSeq GCF_000001405.40) for TSS
  extraction. Contigs are NC_000001.11 style; we map to chrN.
- `bam/scDAF_PS00758__v8_gapcdf.bam` (or __v8.bam, __protected_runs+merge.bam)
  for nucleosome calls (legacy ns/nl tags in query coords).

For each TSS, we find reads overlapping ±3 kb, convert each
nucleosome call from query coords back to reference coords via
aligned pairs, and increment a TSS-relative occupancy histogram.

Output: `output/figure_scdaf_tss_meta.png` — one track per caller
(v7 stock, v8, v8 gap_cdf) showing the canonical -1/+1 nucleosome
pattern around active TSSes.
"""

from __future__ import annotations

import os
import sys
from collections import defaultdict

import numpy as np
import pysam
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))


# NCBI RefSeq hg38 → UCSC chr mapping (primary chromosomes only)
NC_TO_CHR = {
    'NC_000001.11': 'chr1',  'NC_000002.12': 'chr2',
    'NC_000003.12': 'chr3',  'NC_000004.12': 'chr4',
    'NC_000005.10': 'chr5',  'NC_000006.12': 'chr6',
    'NC_000007.14': 'chr7',  'NC_000008.11': 'chr8',
    'NC_000009.12': 'chr9',  'NC_000010.11': 'chr10',
    'NC_000011.10': 'chr11', 'NC_000012.12': 'chr12',
    'NC_000013.11': 'chr13', 'NC_000014.9':  'chr14',
    'NC_000015.10': 'chr15', 'NC_000016.10': 'chr16',
    'NC_000017.11': 'chr17', 'NC_000018.10': 'chr18',
    'NC_000019.10': 'chr19', 'NC_000020.11': 'chr20',
    'NC_000021.9':  'chr21', 'NC_000022.11': 'chr22',
    'NC_000023.11': 'chrX',  'NC_000024.10': 'chrY',
}


def parse_tss_from_gtf(gtf_path, max_tss=20000):
    """Yield (chrom, tss_pos, strand) for each transcript.

    TSS = first position on + strand, last position on - strand.
    We deduplicate per gene_id, keeping the first encountered
    transcript. Returns a list sorted by chrom + position.
    """
    tss_by_gene = {}
    n_lines = 0
    with open(gtf_path) as fh:
        for line in fh:
            if not line or line.startswith('#'):
                continue
            n_lines += 1
            if len(tss_by_gene) >= max_tss:
                break
            fields = line.rstrip('\n').split('\t')
            if len(fields) < 9:
                continue
            feature = fields[2]
            if feature != 'transcript':
                continue
            chrom_nc = fields[0]
            chrom = NC_TO_CHR.get(chrom_nc)
            if chrom is None:
                continue
            start = int(fields[3])  # 1-based inclusive
            end = int(fields[4])    # 1-based inclusive
            strand = fields[6]
            attrs = fields[8]
            gene_id = None
            for kv in attrs.split(';'):
                kv = kv.strip()
                if kv.startswith('gene_id'):
                    gene_id = kv.split('"')[1] if '"' in kv else None
                    break
            if gene_id is None:
                continue
            if gene_id in tss_by_gene:
                continue
            if strand == '+':
                tss = start  # 1-based
            elif strand == '-':
                tss = end
            else:
                continue
            # Convert to 0-based for downstream BAM logic
            tss_by_gene[gene_id] = (chrom, tss - 1, strand)
    out = list(tss_by_gene.values())
    out.sort()
    return out


def aggregate_tss_occupancy(bam_path, tss_list, window=3000,
                              min_reads_per_tss=3):
    """For each TSS, accumulate a ±window occupancy and coverage
    profile using the read's ns/nl tags (query coords) mapped back
    to reference coords via the linear query->ref approximation
    (valid for PacBio CCS reads with minimal indels).

    Only TSSes whose window has >= min_reads_per_tss overlapping
    reads contribute to the aggregate (drops low-coverage noise).

    Returns (occ, cov, n_tss_used) where occ and cov are length
    2*window + 1 arrays, centered on TSS, and n_tss_used is how many
    TSSes actually contributed.
    """
    width = 2 * window + 1
    occ = np.zeros(width, dtype=np.int64)
    cov = np.zeros(width, dtype=np.int64)

    # Build chrom -> list of (tss, strand) for fast lookup
    tss_by_chrom = defaultdict(list)
    for chrom, tss, strand in tss_list:
        tss_by_chrom[chrom].append((tss, strand))
    for chrom in tss_by_chrom:
        tss_by_chrom[chrom].sort()

    bam = pysam.AlignmentFile(bam_path, 'rb', check_sq=False)
    contigs_avail = set(b['SN'] for b in bam.header.get('SQ', []))
    usable_chroms = sorted(set(tss_by_chrom) & contigs_avail)
    n_total_tss = sum(len(tss_by_chrom[c]) for c in usable_chroms)
    print(f'  querying {len(usable_chroms)} chromosomes with '
          f'{n_total_tss} TSSes')

    n_tss_used = 0
    # Progress tracking
    n_tss_done = 0
    for chrom in usable_chroms:
        for tss_pos, strand in tss_by_chrom[chrom]:
            n_tss_done += 1
            if n_tss_done % 2000 == 0:
                print(f'    {n_tss_done}/{n_total_tss} TSSes processed, '
                      f'{n_tss_used} used', flush=True)
            win_lo = max(0, tss_pos - window)
            win_hi = tss_pos + window + 1
            try:
                reads = [r for r in bam.fetch(chrom, win_lo, win_hi)
                          if not r.is_unmapped
                          and not r.is_secondary
                          and not r.is_supplementary]
            except ValueError:
                continue
            if len(reads) < min_reads_per_tss:
                continue
            n_tss_used += 1
            for r in reads:
                rs = r.reference_start
                re = r.reference_end
                lo_ref = max(win_lo, rs)
                hi_ref = min(win_hi, re)
                if hi_ref <= lo_ref:
                    continue
                if strand == '+':
                    hist_lo = lo_ref - tss_pos + window
                    hist_hi = hi_ref - tss_pos + window
                else:
                    hist_lo = tss_pos - hi_ref + 1 + window
                    hist_hi = tss_pos - lo_ref + 1 + window
                hist_lo = max(0, hist_lo)
                hist_hi = min(width, hist_hi)
                if hist_hi > hist_lo:
                    cov[hist_lo:hist_hi] += 1
                try:
                    ns = list(r.get_tag('ns'))
                    nl = list(r.get_tag('nl'))
                except KeyError:
                    continue
                if not ns:
                    continue
                qas = r.query_alignment_start or 0
                for qs, l in zip(ns, nl):
                    rs_bp = rs + (qs - qas)
                    re_bp = rs + (qs + l - qas)
                    a = max(win_lo, rs_bp)
                    b = min(win_hi, re_bp)
                    if b <= a:
                        continue
                    if strand == '+':
                        h_lo = a - tss_pos + window
                        h_hi = b - tss_pos + window
                    else:
                        h_lo = tss_pos - b + 1 + window
                        h_hi = tss_pos - a + 1 + window
                    h_lo = max(0, h_lo)
                    h_hi = min(width, h_hi)
                    if h_hi > h_lo:
                        occ[h_lo:h_hi] += 1
    bam.close()
    return occ, cov, n_tss_used


def main():
    gtf_path = '/Users/tt7739/Dropbox/Fiber-NET-seq/DAF-C/Reference/hg38.gtf'
    bam_dir = os.path.join(HERE, 'output', 'bam')
    out_dir = os.path.join(HERE, 'output')

    # Try the full-scale scDAF BAM if I produced one, else the
    # subsampled bench output.
    # Prefer full-scale BAMs if present (much better statistics)
    candidates_v7 = [
        os.path.join(bam_dir, 'scDAF_PS00758__protected_runs+merge.bam'),
    ]
    candidates_v8 = [
        '/tmp/scdaf_v8_full.bam',
        os.path.join(bam_dir, 'scDAF_PS00758__v8.bam'),
    ]
    candidates_v8g = [
        '/tmp/scdaf_v8_gapcdf_full_iter16.bam',
        '/tmp/scdaf_v8_gapcdf_80k_iter16.bam',
        '/tmp/scdaf_v8_gapcdf_full.bam',
        os.path.join(bam_dir, 'scDAF_PS00758__v8_gapcdf.bam'),
    ]

    print('Parsing TSS positions from hg38 GTF...')
    tss_list = parse_tss_from_gtf(gtf_path, max_tss=20000)
    print(f'  got {len(tss_list)} TSSes across '
          f'{len(set(c for c,_,_ in tss_list))} chromosomes')

    results = {}
    for label, candidates in [
        ('v7 stock', candidates_v7),
        ('v8 default', candidates_v8),
        ('v8 gap_cdf', candidates_v8g),
    ]:
        bam_path = next((p for p in candidates if os.path.exists(p)), None)
        if bam_path is None:
            print(f'  skip {label}: no BAM')
            continue
        # pysam needs the index for .fetch() by region
        idx = bam_path + '.bai'
        if not os.path.exists(idx):
            print(f'  indexing {os.path.basename(bam_path)}...')
            pysam.index(bam_path)
        print(f'\n{label}: {os.path.basename(bam_path)}')
        occ, cov, n_used = aggregate_tss_occupancy(
            bam_path, tss_list, window=3000, min_reads_per_tss=3)
        with np.errstate(invalid='ignore', divide='ignore'):
            profile = np.where(cov > 0, occ / np.maximum(cov, 1), 0)
        # Smooth with a 21-bp rolling average to reduce jitter
        kernel = np.ones(21) / 21
        smoothed = np.convolve(profile, kernel, mode='same')
        results[label] = (smoothed, cov.max(), n_used)

    if not results:
        print('no results')
        return

    fig, axes = plt.subplots(len(results), 1, figsize=(13, 2.4 * len(results)),
                              sharex=True)
    if len(results) == 1:
        axes = [axes]
    x = np.arange(-3000, 3001)
    colors = {'v7 stock': '#dc2626', 'v8 default': '#be185d',
               'v8 gap_cdf': '#7c2d12'}
    for ax, (label, (profile, cov_max, n_used)) in zip(axes, results.items()):
        ax.fill_between(x, 0, profile, color=colors.get(label, '#888'),
                          alpha=0.85)
        ax.plot(x, profile, color='black', lw=0.5, alpha=0.6)
        ax.axvline(0, color='black', ls='--', lw=0.7,
                    label='TSS')
        # -1/+1 nucleosome expected positions (canonical: TSS+30 and TSS-160)
        ax.axvspan(30, 180, color='#fbbf24', alpha=0.12, label='+1 nuc')
        ax.axvspan(-200, -50, color='#22c55e', alpha=0.12, label='-1 nuc')
        ax.set_ylim(0, 1.0)
        ax.set_ylabel(label, fontsize=10, rotation=0, ha='right',
                       va='center', labelpad=55)
        ax.grid(alpha=0.2)
        ax.text(0.99, 0.95,
                f'{n_used} TSSes used (≥3 reads), peak cov ~{cov_max}',
                transform=ax.transAxes, ha='right', va='top', fontsize=8,
                color='#444')
    axes[0].legend(fontsize=8, loc='upper left')
    axes[-1].set_xlabel('Distance from TSS (bp)', fontsize=10)
    fig.suptitle('scDAF PS00758 — meta-occupancy around hg38 TSSes '
                  f'(max {len(tss_list)} genes; 21-bp smoothed)', fontsize=12)
    fig.tight_layout()
    out = os.path.join(out_dir, 'figure_scdaf_tss_meta.png')
    fig.savefig(out, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'\nWrote {out}')


if __name__ == '__main__':
    main()
