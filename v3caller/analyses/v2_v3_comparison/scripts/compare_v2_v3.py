#!/usr/bin/env python3
"""Compute v2-vs-v3 comparison stats from called BAMs.

v2 stores everything in ns/nl tag (nucs + TFs). Entries with nl < 90
are TF-scale, nl ≥ 90 are nucleosomes.
v3 stores nucs in ns/nl (all ≥ 90 bp by caller_v8 construction) and
TFs separately in tn/tl.

For each BAM we compute:
  - reads processed (tag-bearing)
  - nucs ≥90 / read (mean + median)
  - median nuc length
  - mean nuc length
  - % of nucs ≥300 bp (overmerge indicator)
  - % of nucs ≥500 bp
  - TFs / read (mean)

Usage:
  python compare_v2_v3.py --label LABEL --version {v2,v3} \\
      --in-bam BAM [--in-bam BAM ...]
"""

from __future__ import annotations

import argparse
import json
import sys
from statistics import median, mean

import pysam


def stats_from_bam(bam_path, version='v3'):
    bam = pysam.AlignmentFile(bam_path, 'rb', check_sq=False)
    n_reads = 0
    n_tagged = 0
    nuc_lens_per_read = []
    tf_counts = []
    for read in bam:
        if read.is_unmapped or read.is_secondary or read.is_supplementary:
            continue
        n_reads += 1
        tags = dict(read.tags)
        if 'nl' not in tags:
            continue
        n_tagged += 1
        ns = tags.get('ns', [])
        nl = tags.get('nl', [])
        if not nl:
            nuc_lens_per_read.append([])
        else:
            if version == 'v2':
                # v2: split by 90 bp cutoff
                nucs = [l for l in nl if l >= 90]
                tfs = [l for l in nl if l < 90]
            else:
                nucs = list(nl)  # v3 has separate tf tag
                tfs_v3 = tags.get('tl', [])
                tfs = list(tfs_v3) if tfs_v3 else []
            nuc_lens_per_read.append(nucs)
            tf_counts.append(len(tfs))
    bam.close()
    return n_reads, n_tagged, nuc_lens_per_read, tf_counts


def aggregate(nuc_lens_per_read, tf_counts):
    all_nuc_lens = [l for lens in nuc_lens_per_read for l in lens]
    nucs_per_read = [len(lens) for lens in nuc_lens_per_read]
    if not all_nuc_lens:
        return None
    return {
        'n_reads_tagged': len(nuc_lens_per_read),
        'total_nucs': len(all_nuc_lens),
        'nucs_per_read_mean': round(mean(nucs_per_read), 2),
        'nucs_per_read_median': int(median(nucs_per_read)),
        'nuc_len_mean': round(mean(all_nuc_lens), 1),
        'nuc_len_median': int(median(all_nuc_lens)),
        'pct_ge_300bp': round(100 * sum(1 for l in all_nuc_lens if l >= 300)
                               / len(all_nuc_lens), 1),
        'pct_ge_500bp': round(100 * sum(1 for l in all_nuc_lens if l >= 500)
                               / len(all_nuc_lens), 1),
        'tfs_per_read_mean': round(mean(tf_counts), 1) if tf_counts else 0.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--label', required=True)
    ap.add_argument('--version', choices=['v2', 'v3'], required=True)
    ap.add_argument('--in-bam', action='append', required=True)
    ap.add_argument('--out-json', default=None)
    args = ap.parse_args()

    per_bam = []
    combined_nucs = []
    combined_tfs = []
    for b in args.in_bam:
        print(f'[{b}]', file=sys.stderr)
        n_reads, n_tagged, nuc_lens, tf_counts = stats_from_bam(
            b, version=args.version)
        stats = aggregate(nuc_lens, tf_counts)
        if stats is None:
            print(f'  (no nucs)', file=sys.stderr); continue
        stats['n_reads'] = n_reads
        stats['bam'] = b
        per_bam.append(stats)
        combined_nucs.extend(nuc_lens)
        combined_tfs.extend(tf_counts)
        print(f'  {stats}', file=sys.stderr)

    combined = aggregate(combined_nucs, combined_tfs)
    combined['n_reads'] = sum(s['n_reads'] for s in per_bam)

    out = {
        'label': args.label,
        'version': args.version,
        'per_bam': per_bam,
        'combined': combined,
    }
    print(json.dumps(out, indent=2))

    if args.out_json:
        with open(args.out_json, 'w') as f:
            json.dump(out, f, indent=2)


if __name__ == '__main__':
    main()
