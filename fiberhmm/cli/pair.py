#!/usr/bin/env python3
"""fiberhmm-pair -- sequence-first cross-strand pairing for DAF-seq.

DddA deaminates both strands of a duplex; the two strands are sequenced as
separate reads of opposite flavor (CT = C->T, GA = G->A) that overlap in the
genome but sample different bases, so they cannot be matched by deamination
pattern. With ``--reference``, this tool first matches local CT/GA components
using bases at reference A/T positions, which DddA cannot alter. Assignment is
strictly reciprocal except at complete local 2x2 loci, where a grossly
discordant edge can resolve the opposite diagonal. Sequence-ambiguous reads
then fall back to nucleosome footprints.

Output is non-destructive: every input read is written through unchanged except
for added local tags on reads that received a confident mate --

    mp:Z  mate partner query_name
    mc:i  pair cross-correlation x1000 (0..1000)
    mg:i  best-minus-second margin  x1000
    mt:A  status: 'P' paired, 'U' unresolved (had candidates, failed gate),
          '.' no overlapping opposite-strand candidate
    pm:A  pairing method: 'S' sequence assignment, 'F' footprint fallback
    sb:i  shared deamination-safe sequence bases
    sd:i  sequence differences
    sr:i  sequence difference rate x1,000,000
    sg:i  sequence assignment margin x1,000,000 (sequence pairs only)
    pa:A  sequence assignment kind: 'R' reciprocal, 'C' constrained 2x2

and an optional ``--pairs-tsv`` table of resolved pairs. Pairing is done within
each chromosome, so a coordinate-sorted + indexed BAM is required.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from collections import Counter
from itertools import chain

import numpy as np
import pysam

from fiberhmm.crossstrand.pairing import (
    PairParams, STATUS_PAIRED, assign_pairs, build_feature,
)

_TAG_PARTNER = 'mp'
_TAG_CORR = 'mc'
_TAG_MARGIN = 'mg'
_TAG_STATUS = 'mt'
_TAG_METHOD = 'pm'
_TAG_SEQ_BASES = 'sb'
_TAG_SEQ_DIFF = 'sd'
_TAG_SEQ_RATE = 'sr'
_TAG_SEQ_MARGIN = 'sg'
_TAG_ASSIGNMENT = 'pa'


def run_pair(in_bam, out_bam, params: PairParams, prob_threshold=0,
             pairs_tsv=None, io_threads=4, reference_path=None):
    t0 = time.time()
    if reference_path is None:
        print("Warning: no reference FASTA supplied; sequence-first pairing is "
              "disabled and only the legacy footprint fallback will run.",
              file=sys.stderr)
    bam = pysam.AlignmentFile(in_bam, 'rb')
    if bam.header.get('HD', {}).get('SO') != 'coordinate':
        print("Warning: input is not marked coordinate-sorted; pairing is "
              "per-chromosome and assumes sorted input.", file=sys.stderr)

    # Pass 1: per-chromosome, build features and resolve pairs. Store results
    # keyed by (query_name, flavor) so pass 2 can tag each read; the flavor
    # disambiguates the rare case of a name shared across strands.
    resolved = {}
    status_of = {}  # (name, flavor) -> status char
    n_reads = n_feat = n_paired = 0
    per_chrom_paired = Counter()
    per_method = Counter()
    fasta = pysam.FastaFile(reference_path) if reference_path else None

    for chrom in bam.references:
        feats = []
        idx = 0
        try:
            it = bam.fetch(chrom)
        except ValueError:
            continue
        first = next(it, None)
        if first is None:
            continue
        reference = None
        if fasta is not None and chrom in fasta.references:
            reference = np.frombuffer(
                fasta.fetch(chrom).upper().encode(), dtype=np.uint8,
            )
        elif fasta is not None:
            print(f"  {chrom}: absent from reference; using footprint-only "
                  "fallback", file=sys.stderr)
        for read in chain((first,), it):
            n_reads += 1
            f = build_feature(read, idx, params, prob_threshold, reference)
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
                resolved[key] = (
                    mate.name, res.score.get(i), res.margin[i], res.method[i],
                    res.sequence.get(i), res.sequence_margin.get(i),
                    res.sequence_kind.get(i),
                )
                per_method[res.method[i]] += 1
                per_chrom_paired[chrom] += 1
        n_paired += per_chrom_paired[chrom]
        print(f"  {chrom}: {len(feats):,} featurized reads -> "
              f"{per_chrom_paired[chrom]:,} paired reads [{time.time()-t0:.0f}s]",
              file=sys.stderr)
    bam.close()
    if fasta is not None:
        fasta.close()

    n_pairs = n_paired // 2
    print(f"Pass 1: {n_reads:,} reads ({n_feat:,} featurizable) -> "
          f"{n_pairs:,} cross-strand pairs covering {n_paired:,} reads "
          f"({100.0*n_paired/max(n_feat,1):.1f}% of featurizable) "
          f"[sequence {per_method['S']//2:,}; footprint {per_method['F']//2:,}] "
          f"[{time.time()-t0:.0f}s]", file=sys.stderr)

    if pairs_tsv:
        written = set()
        with open(pairs_tsv, 'w') as fh:
            fh.write("ct_read\tga_read\tmethod\tassignment\tcorr\tmargin\tseq_bases\t"
                     "seq_differences\tseq_rate\tseq_margin\n")
            for (name, flavor), row in resolved.items():
                mate, corr, marg, method, seq, seq_marg, assignment = row
                pair_key = tuple(sorted((name, mate)))
                if pair_key in written:
                    continue
                written.add(pair_key)
                # order columns CT then GA
                ct, ga = (name, mate) if flavor == 1 else (mate, name)
                corr_text = '' if corr is None else f'{corr:.4f}'
                if seq is None:
                    seq_fields = ('', '', '', '')
                else:
                    seq_fields = (
                        str(seq.bases), str(seq.mismatches), f'{seq.rate:.6f}',
                        '' if seq_marg is None else f'{seq_marg:.6f}',
                    )
                fh.write(f"{ct}\t{ga}\t{method}\t{assignment or ''}\t"
                         f"{corr_text}\t{marg:.6f}\t"
                         + '\t'.join(seq_fields) + '\n')

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
                        mate, corr, marg, method, seq, seq_marg, assignment = resolved[key]
                        read.set_tag(_TAG_PARTNER, mate, value_type='Z')
                        if corr is not None:
                            read.set_tag(_TAG_CORR, int(round(1000 * corr)), value_type='i')
                        read.set_tag(_TAG_MARGIN, int(round(1000 * marg)), value_type='i')
                        read.set_tag(_TAG_METHOD, method, value_type='A')
                        if seq is not None:
                            read.set_tag(_TAG_SEQ_BASES, seq.bases, value_type='i')
                            read.set_tag(_TAG_SEQ_DIFF, seq.mismatches, value_type='i')
                            read.set_tag(
                                _TAG_SEQ_RATE, int(round(1_000_000 * seq.rate)),
                                value_type='i',
                            )
                        if seq_marg is not None:
                            read.set_tag(
                                _TAG_SEQ_MARGIN,
                                int(round(1_000_000 * seq_marg)), value_type='i',
                            )
                        if assignment is not None:
                            read.set_tag(_TAG_ASSIGNMENT, assignment, value_type='A')
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
        description='Sequence-first cross-strand read pairing for DAF-seq, '
                    'with nucleosome-footprint fallback.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    fiberhmm-pair -i calls.bam -o calls.paired.bam -r hg38.fa --pairs-tsv pairs.tsv
    fiberhmm-pair -i calls.bam -o calls.paired.bam -r hg38.fa --min-sequence-margin 0.002
        """,
    )
    p.add_argument('-i', '--input', required=True, help='Footprint-called DAF BAM (coordinate-sorted + indexed)')
    p.add_argument('-o', '--output', required=True, help='Output BAM (all reads, mate tags added)')
    p.add_argument('-r', '--reference', default=None,
                   help='Reference FASTA enabling sequence-first assignment; '
                        'omit for footprint-only legacy behavior')
    p.add_argument('--pairs-tsv', default=None, help='Write resolved pairs to this TSV')
    p.add_argument('--min-score', type=float, default=0.25, help='Min cross-correlation floor to accept any pair (default 0.25)')
    p.add_argument('--min-margin', type=float, default=0.05, help='Min best-minus-competitor margin, both reads (default 0.05)')
    p.add_argument('--null-floor', type=float, default=0.24, help='Wrong-pair correlation baseline; virtual competitor for lone (1+1) pairs (default 0.24, ~data null p90)')
    p.add_argument('--min-overlap', type=int, default=1500, help='Min genomic overlap bp (default 1500)')
    p.add_argument('--min-nucs', type=int, default=4, help='Min nucleosome dyads within the overlap, each read (default 4)')
    p.add_argument('--sigma', type=float, default=30.0, help='Gaussian dyad width bp (default 30)')
    p.add_argument('--grid', type=int, default=10, help='Signal resolution bp (default 10)')
    p.add_argument('--max-lag', type=int, default=60, help='+/- register-shift searched bp (default 60)')
    p.add_argument('--min-sequence-bases', type=int, default=500,
                   help='Min shared reference-A/T bases for a sequence edge (default 500)')
    p.add_argument('--max-sequence-mismatch-rate', type=float, default=0.002,
                   help='Hard veto above this sequence difference rate (default 0.002)')
    p.add_argument('--min-component-discordance-rate', type=float, default=0.02,
                   help='Min rejected-edge difference rate to constrain a 2x2 (default 0.02)')
    p.add_argument('--max-sequence-pair-rate', type=float, default=0.01,
                   help='Max difference rate on a sequence-selected pair (default 0.01)')
    p.add_argument('--min-sequence-margin', type=float, default=0.001,
                   help='Min sequence preference/assignment margin (default 0.001)')
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
        null_floor=args.null_floor,
        min_sequence_bases=args.min_sequence_bases,
        max_sequence_mismatch_rate=args.max_sequence_mismatch_rate,
        min_component_discordance_rate=args.min_component_discordance_rate,
        min_sequence_margin=args.min_sequence_margin,
        max_sequence_pair_rate=args.max_sequence_pair_rate,
    )
    run_pair(args.input, args.output, params, prob_threshold=args.prob_threshold,
             pairs_tsv=args.pairs_tsv, io_threads=args.io_threads,
             reference_path=args.reference)


if __name__ == '__main__':
    main()
