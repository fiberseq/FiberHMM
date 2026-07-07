#!/usr/bin/env python3
"""fiberhmm-pair -- cross-strand read pairing for DAF-seq via footprint pattern.

DddA deaminates both strands of a duplex; the two strands are sequenced as
separate reads of opposite flavor (CT = C->T, GA = G->A) that overlap in the
genome but sample different bases, so they cannot be matched by deamination
pattern. This tool matches them by their **nucleosome footprint pattern** --
a physical property of the duplex shared by both strands -- scoring candidate
pairs by the cross-correlation of their MA ``nuc`` dyad-density signals and
resolving each locus by reciprocal-best-match with a margin gate (the local
2x2 assignment; ambiguous loci are left unresolved rather than force-paired).

Output is non-destructive: every input read is written through unchanged except
for added local tags on reads that received a confident mate --

    mp:Z  mate partner query_name
    mc:i  pair cross-correlation x1000 (0..1000)
    mg:i  best-minus-second margin  x1000
    mt:A  status: 'P' paired, 'U' unresolved (had candidates, failed gate),
          '.' no overlapping opposite-strand candidate

and an optional ``--pairs-tsv`` table of resolved pairs. Pairing is done within
each chromosome, so a coordinate-sorted + indexed BAM is required.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from collections import Counter

import pysam

from fiberhmm.crossstrand.pairing import (
    PairParams, STATUS_PAIRED, STATUS_NONE, assign_pairs, build_feature,
)

_TAG_PARTNER = 'mp'
_TAG_CORR = 'mc'
_TAG_MARGIN = 'mg'
_TAG_STATUS = 'mt'


def run_pair(in_bam, out_bam, params: PairParams, prob_threshold=0,
             pairs_tsv=None, io_threads=4):
    t0 = time.time()
    bam = pysam.AlignmentFile(in_bam, 'rb')
    if bam.header.get('HD', {}).get('SO') != 'coordinate':
        print("Warning: input is not marked coordinate-sorted; pairing is "
              "per-chromosome and assumes sorted input.", file=sys.stderr)

    # Pass 1: per-chromosome, build features and resolve pairs. Store results
    # keyed by (query_name, flavor) so pass 2 can tag each read; the flavor
    # disambiguates the rare case of a name shared across strands.
    resolved = {}   # (name, flavor) -> (partner_name, corr, margin)
    status_of = {}  # (name, flavor) -> status char
    n_reads = n_feat = n_paired = 0
    per_chrom_paired = Counter()

    for chrom in bam.references:
        feats = []
        idx = 0
        try:
            it = bam.fetch(chrom)
        except ValueError:
            continue
        for read in it:
            n_reads += 1
            f = build_feature(read, idx, params, prob_threshold)
            if f is not None:
                feats.append(f)
                idx += 1
        if not feats:
            continue
        n_feat += len(feats)
        res = assign_pairs(feats, params)
        by_index = {f.index: f for f in feats}
        for i, st in res.status.items():
            f = by_index[i]
            key = (f.name, f.flavor)
            status_of[key] = st
            if st == STATUS_PAIRED:
                mate = by_index[res.partner[i]]
                resolved[key] = (mate.name, res.score[i], res.margin[i])
                per_chrom_paired[chrom] += 1
        n_paired += per_chrom_paired[chrom]
        print(f"  {chrom}: {len(feats):,} featurized reads -> "
              f"{per_chrom_paired[chrom]:,} paired reads [{time.time()-t0:.0f}s]",
              file=sys.stderr)
    bam.close()

    n_pairs = n_paired // 2
    print(f"Pass 1: {n_reads:,} reads ({n_feat:,} featurizable) -> "
          f"{n_pairs:,} cross-strand pairs covering {n_paired:,} reads "
          f"({100.0*n_paired/max(n_feat,1):.1f}% of featurizable) "
          f"[{time.time()-t0:.0f}s]", file=sys.stderr)

    if pairs_tsv:
        written = set()
        with open(pairs_tsv, 'w') as fh:
            fh.write("ct_read\tga_read\tcorr\tmargin\n")
            for (name, flavor), (mate, corr, marg) in resolved.items():
                pair_key = tuple(sorted((name, mate)))
                if pair_key in written:
                    continue
                written.add(pair_key)
                # order columns CT then GA
                ct, ga = (name, mate) if flavor == 1 else (mate, name)
                fh.write(f"{ct}\t{ga}\t{corr:.4f}\t{marg:.4f}\n")

    # Pass 2: write output with tags. Recompute flavor per read (cheap) so we
    # can look up the right (name, flavor) key.
    from fiberhmm.crossstrand.pairing import read_flavor
    bam = pysam.AlignmentFile(in_bam, 'rb')
    out = pysam.AlignmentFile(out_bam, 'wb', template=bam, threads=io_threads)
    n_written = 0
    for read in bam.fetch(until_eof=True):
        if not (read.is_unmapped or read.is_secondary or read.is_supplementary):
            fl = read_flavor(read, prob_threshold)
            if fl is not None:
                key = (read.query_name, fl)
                st = status_of.get(key)
                if st is not None:
                    read.set_tag(_TAG_STATUS, st, value_type='A')
                    if st == STATUS_PAIRED:
                        mate, corr, marg = resolved[key]
                        read.set_tag(_TAG_PARTNER, mate, value_type='Z')
                        read.set_tag(_TAG_CORR, int(round(1000 * corr)), value_type='i')
                        read.set_tag(_TAG_MARGIN, int(round(1000 * marg)), value_type='i')
        out.write(read)
        n_written += 1
    out.close()
    bam.close()
    print(f"Pass 2: wrote {n_written:,} reads -> {out_bam} [{time.time()-t0:.0f}s]",
          file=sys.stderr)
    return {'n_reads': n_reads, 'n_featurizable': n_feat, 'n_pairs': n_pairs}


def main():
    p = argparse.ArgumentParser(
        prog='fiberhmm-pair',
        description='Cross-strand read pairing for DAF-seq by nucleosome '
                    'footprint-pattern cross-correlation (reciprocal-best + '
                    'margin gate). Non-destructive: tags reads with their mate.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    fiberhmm-pair -i calls.bam -o calls.paired.bam --pairs-tsv pairs.tsv
    fiberhmm-pair -i calls.bam -o calls.paired.bam --min-score 0.55 --min-margin 0.1
        """,
    )
    p.add_argument('-i', '--input', required=True, help='Footprint-called DAF BAM (coordinate-sorted + indexed)')
    p.add_argument('-o', '--output', required=True, help='Output BAM (all reads, mate tags added)')
    p.add_argument('--pairs-tsv', default=None, help='Write resolved pairs to this TSV')
    p.add_argument('--min-score', type=float, default=0.5, help='Min cross-correlation to accept a pair (default 0.5)')
    p.add_argument('--min-margin', type=float, default=0.05, help='Min best-minus-second margin, both reads (default 0.05)')
    p.add_argument('--min-overlap', type=int, default=1500, help='Min genomic overlap bp (default 1500)')
    p.add_argument('--min-nucs', type=int, default=4, help='Min nucleosome dyads within the overlap, each read (default 4)')
    p.add_argument('--sigma', type=float, default=30.0, help='Gaussian dyad width bp (default 30)')
    p.add_argument('--grid', type=int, default=10, help='Signal resolution bp (default 10)')
    p.add_argument('--max-lag', type=int, default=60, help='+/- register-shift searched bp (default 60)')
    p.add_argument('-p', '--prob-threshold', type=int, default=0, help='Min ML prob for MM/ML dU calls (default 0)')
    p.add_argument('--io-threads', type=int, default=4, help='htslib compression threads for output (default 4)')
    args = p.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: input not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    params = PairParams(
        grid_bp=args.grid, sigma_bp=args.sigma, max_lag_bp=args.max_lag,
        min_overlap_bp=args.min_overlap, min_nucs=args.min_nucs,
        min_score=args.min_score, min_margin=args.min_margin,
    )
    run_pair(args.input, args.output, params, prob_threshold=args.prob_threshold,
             pairs_tsv=args.pairs_tsv, io_threads=args.io_threads)


if __name__ == '__main__':
    main()
