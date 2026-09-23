#!/usr/bin/env python3
"""fiberhmm-pair -- unified cross-strand pairing for DAF-seq.

DddA deaminates both strands of a duplex; the two strands are sequenced as
separate reads of opposite flavor (CT = C->T, GA = G->A) that overlap in the
genome but sample different bases, so they cannot be matched by deamination
pattern. The default workflow combines direct A/T sequence-supported
assignments with high-confidence assignments from a frozen sequence-free model
of the nucleosome lattice, aligned geometry, and non-CpG DddA protection.
Sequence-supported assignments take precedence on conflicts. ``--sequence-only``
disables the sequence-free route.

Output is non-destructive: every input read is written through unchanged except
for added local tags on reads that received a confident mate --

    mp:Z  mate partner query_name
    mc:i  optional nucleosome cross-correlation x1000 for sequence pairs
    dm:i  sequence-free model decision score x1000
    mg:i  sequence-free reciprocal margin x1000
    mv:Z  frozen sequence-free model identifier
    mt:A  status: 'P' paired, 'U' unresolved (had candidates, failed gate),
          '.' no overlapping opposite-strand candidate
    pm:A  pairing method: 'S' sequence-supported, 'D' sequence-free model
    sb:i  shared deamination-safe sequence bases
    sd:i  sequence differences
    sr:i  sequence difference rate x1,000,000
    sg:i  sequence assignment margin x1,000,000 (sequence pairs only)
    pa:A  sequence assignment kind: 'R' reciprocal, 'C' constrained 2x2

and an optional ``--pairs-tsv`` table of resolved pairs. Pairing is done within
each chromosome, so a coordinate-sorted + indexed BAM is required.

The module-level ``run_pair`` function below preserves the pre-unification
sequence/footprint implementation for API compatibility and validation replay;
the public command uses ``run_pairing`` from :mod:`fiberhmm.cli.duplex`.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from collections import Counter
from dataclasses import replace
from itertools import chain

import numpy as np
import pysam

from fiberhmm.crossstrand.pairing import (
    PairParams, STATUS_PAIRED, SequenceScore, assign_pairs, build_feature,
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


def _has_usable_sequence_score(score: SequenceScore | None) -> bool:
    """True only when sequence evidence can be serialized meaningfully."""
    return (score is not None and score.bases > 0 and
            bool(np.isfinite(score.rate)))


def run_pair(in_bam, out_bam, params: PairParams, prob_threshold=0,
             pairs_tsv=None, io_threads=4, reference_path=None,
             paired_only=False):
    t0 = time.time()
    if reference_path is None:
        print("Info: no reference FASTA supplied; sequence-first pairing will "
              "use per-read MD+CIGAR evidence where available and otherwise "
              "fall back to footprints.",
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
    seen_feature_names = {}
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
                previous = seen_feature_names.get(f.name)
                if previous is not None:
                    raise ValueError(
                        "paired-duplex BAM requires unique primary query names; "
                        f"{f.name!r} identifies both {previous} and "
                        f"{chrom}:{f.ref_start}-{f.ref_end}/{f.flavor_name}"
                    )
                seen_feature_names[f.name] = (
                    f"{chrom}:{f.ref_start}-{f.ref_end}/{f.flavor_name}"
                )
                feats.append(f)
                idx += 1
        if not feats:
            continue
        n_feat += len(feats)
        chrom_params = params
        if params.single_cell_haplotype and not (
            chrom.endswith('_MATERNAL') or chrom.endswith('_PATERNAL')
        ):
            chrom_params = replace(params, single_cell_haplotype=False)
        res = assign_pairs(feats, chrom_params)
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
          f"[haplotype {per_method['H']//2:,}; sequence {per_method['S']//2:,}; "
          f"footprint {per_method['F']//2:,}] "
          f"[{time.time()-t0:.0f}s]", file=sys.stderr)

    if pairs_tsv:
        written = set()
        with open(pairs_tsv, 'w') as fh:
            fh.write("ct_read\tga_read\tmethod\tassignment\tcorr\tfootprint_margin\tseq_bases\t"
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
                if not _has_usable_sequence_score(seq):
                    seq_fields = ('', '', '', '')
                else:
                    seq_fields = (
                        str(seq.bases), str(seq.mismatches), f'{seq.rate:.6f}',
                        '' if seq_marg is None else f'{seq_marg:.6f}',
                    )
                footprint_margin = f'{marg:.6f}' if method == 'F' else ''
                fh.write(f"{ct}\t{ga}\t{method}\t{assignment or ''}\t"
                         f"{corr_text}\t{footprint_margin}\t"
                         + '\t'.join(seq_fields) + '\n')

    # Pass 2: write output with tags. Recompute flavor per read (cheap) so we
    # can look up the right (name, flavor) key.
    from fiberhmm.crossstrand.pairing import read_flavor
    bam = pysam.AlignmentFile(in_bam, 'rb')
    out = pysam.AlignmentFile(out_bam, 'wb', template=bam, threads=io_threads)
    n_written = 0
    for read in bam.fetch(until_eof=True):
        write_record = not paired_only
        if not (read.is_unmapped or read.is_secondary or read.is_supplementary):
            fl = read_flavor(read, prob_threshold)
            if fl is not None:
                key = (read.query_name, fl)
                st = status_of.get(key)
                if st is not None:
                    read.set_tag(_TAG_STATUS, st, value_type='A')
                    if st == STATUS_PAIRED:
                        write_record = True
                        mate, corr, marg, method, seq, seq_marg, assignment = resolved[key]
                        read.set_tag(_TAG_PARTNER, mate, value_type='Z')
                        if corr is not None:
                            read.set_tag(_TAG_CORR, int(round(1000 * corr)), value_type='i')
                        if method == 'F':
                            read.set_tag(
                                _TAG_MARGIN, int(round(1000 * marg)), value_type='i',
                            )
                        read.set_tag(_TAG_METHOD, method, value_type='A')
                        if _has_usable_sequence_score(seq):
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
        if write_record:
            out.write(read)
            n_written += 1
    out.close()
    bam.close()
    print(f"Pass 2: wrote {n_written:,} reads -> {out_bam} [{time.time()-t0:.0f}s]",
          file=sys.stderr)
    return {'n_reads': n_reads, 'n_featurizable': n_feat, 'n_pairs': n_pairs}


def main():
    from fiberhmm.cli.duplex import run_pairing
    from fiberhmm.cli.merge import run_merge
    from fiberhmm.crossstrand.duplex import DuplexParams

    p = argparse.ArgumentParser(
        prog='fiberhmm-pair',
        description=(
            'Pair DddA CT/GA reads using sequence-supported assignments plus '
            'the high-confidence sequence-free duplex model.'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Default: sequence-supported + high-confidence sequence-free pairs
    fiberhmm-pair -i calls.bam -o calls.paired.bam -r hg38.fa --pairs-tsv pairs.tsv

    # Require direct A/T sequence support; FASTA is optional when MD is present
    fiberhmm-pair -i calls.bam -o calls.sequence-paired.bam --sequence-only

    # Pair, merge each duplex, and jointly re-call footprints in one command
    fiberhmm-pair -i calls.bam -o calls.duplex.bam -r hg38.fa --merge --recall
        """,
    )
    p.add_argument('-i', '--input', required=True,
                   help='Coordinate-sorted, indexed FiberHMM-called DddA BAM')
    p.add_argument('-o', '--output', required=True,
                   help='Output paired-source BAM, or consensus BAM with --merge')
    p.add_argument('-r', '--reference', default=None,
                   help='Matching indexed FASTA. Required by the default '
                        'sequence-free score; optional with --sequence-only '
                        'when MD+CIGAR is available')
    p.add_argument('--sequence-only', action='store_true',
                   help='Accept only direct A/T sequence-supported pairs; '
                        'disable the sequence-free model')
    p.add_argument('--pairs-tsv', default=None,
                   help='Write selected-pair evidence to this TSV')
    p.add_argument('--receipt-json', default=None,
                   help='Write a machine-readable pairing receipt')
    p.add_argument(
        '--pairs-only', '--paired-only', dest='pairs_only', action='store_true',
        help='Without --merge, write only paired source records. With --merge, '
             'write only consensus records.',
    )
    p.add_argument('--merge', action='store_true',
                   help='Merge each accepted pair into one both-strand consensus record')
    p.add_argument('--recall', action='store_true',
                   help='After merging, jointly re-call nucleosome and TF footprints '
                        '(implies --merge)')
    p.add_argument('--model', default=None,
                   help='Override the bundled frozen sequence-free model JSON')
    p.add_argument('--call-layer', choices=['auto', 'input-ma', 'rotational-recall'],
                   default='auto',
                   help='Nucleosome calibration for the sequence-free model (default auto)')
    p.add_argument('--min-margin', type=float, default=1.0,
                   help='Minimum two-sided sequence-free model margin (default 1.0)')
    p.add_argument('--null-floor', type=float, default=0.0,
                   help='Virtual null model score for a lone candidate (default 0.0)')
    p.add_argument('--min-overlap', type=int, default=1500, help='Min genomic overlap bp (default 1500)')
    p.add_argument('--min-nucs', type=int, default=4, help='Min nucleosome dyads within the overlap, each read (default 4)')
    p.add_argument('--min-sequence-bases', type=int, default=500,
                   help='Min shared reference-A/T bases for a sequence edge (default 500)')
    p.add_argument('--min-component-discordance-rate', type=float, default=0.02,
                   help='Min rejected-edge difference rate to constrain a 2x2 (default 0.02)')
    p.add_argument('--max-sequence-pair-rate', type=float, default=0.01,
                   help='Max difference rate on a sequence-selected pair (default 0.01)')
    p.add_argument('--min-sequence-margin', type=float, default=0.002,
                   help='Min sequence preference/assignment margin (default 0.002)')
    p.add_argument('-p', '--prob-threshold', type=int, default=0, help='Min ML prob for MM/ML dU calls (default 0)')
    p.add_argument('--max-component', type=int, default=10000,
                   help='Safety ceiling for a complete overlap component (default 10000)')
    p.add_argument('--io-threads', type=int, default=4, help='htslib compression threads for output (default 4)')
    p.add_argument('--no-index', action='store_true',
                   help='Do not index a paired-source output')
    p.add_argument('--phase-nrl', type=int, default=196,
                   help='Nucleosome repeat length for consensus recall (default 196)')
    p.add_argument('--nuc-recall-policy', choices=['conservative', 'topology'],
                   default='conservative', help='Nucleosome policy for consensus recall')
    p.add_argument('--ddda-derived-tf-max-edge-gap', type=int, default=12,
                   metavar='BP', help='Edge-evidence requirement for TF calls '
                   'exposed only by DddA nucleosome refinement (default 12; -1 disables)')
    from fiberhmm.core.bam_reader import add_daf_run_mask_arguments, apply_daf_run_mask_arguments
    add_daf_run_mask_arguments(p)
    args = p.parse_args()
    try:
        apply_daf_run_mask_arguments(args, 'ddda')
    except ValueError as exc:
        p.error(str(exc))

    if not os.path.isfile(args.input):
        p.error(f'input not found: {args.input}')
    if args.reference is not None and not os.path.isfile(args.reference):
        p.error(f'reference not found: {args.reference}')
    if not args.sequence_only and not args.reference:
        p.error('--reference is required for default pairing; use --sequence-only '
                'to require sequence-supported pairs only')
    if os.path.abspath(args.input) == os.path.abspath(args.output):
        p.error('input and output paths must differ')
    if args.ddda_derived_tf_max_edge_gap < -1:
        p.error('--ddda-derived-tf-max-edge-gap must be -1 or >= 0')

    duplex_params = DuplexParams(
        min_margin=args.min_margin, null_floor=args.null_floor,
        min_overlap_bp=args.min_overlap, min_nucs=args.min_nucs,
    )
    sequence_params = PairParams(
        min_overlap_bp=args.min_overlap, min_nucs=args.min_nucs,
        min_sequence_bases=args.min_sequence_bases,
        min_component_discordance_rate=args.min_component_discordance_rate,
        min_sequence_margin=args.min_sequence_margin,
        max_sequence_pair_rate=args.max_sequence_pair_rate,
    )
    merge = args.merge or args.recall
    paired_output = args.output if not merge else args.output + '.paired.tmp.bam'
    try:
        receipt = run_pairing(
            args.input, paired_output, args.reference,
            params=duplex_params, sequence_params=sequence_params,
            model_path=args.model, prob_threshold=args.prob_threshold,
            pairs_tsv=args.pairs_tsv, receipt_json=args.receipt_json,
            paired_only=(args.pairs_only and not merge),
            io_threads=args.io_threads, max_component=args.max_component,
            call_layer=args.call_layer,
            create_index=(not merge and not args.no_index),
            pairing_mode='sequence-only' if args.sequence_only else 'hybrid',
        )
        if merge:
            run_merge(
                paired_output, args.output,
                prob_threshold=args.prob_threshold,
                pairs_only=args.pairs_only,
                io_threads=args.io_threads,
                recall=args.recall,
                enzyme='ddda',
                phase_nrl=args.phase_nrl,
                nuc_recall_policy=args.nuc_recall_policy,
                derived_tf_max_edge_ambiguity=(
                    None if args.ddda_derived_tf_max_edge_gap < 0
                    else args.ddda_derived_tf_max_edge_gap
                ),
            )
    except (OSError, ValueError, RuntimeError) as error:
        print(f'fiberhmm-pair: error: {error}', file=sys.stderr)
        raise SystemExit(2) from error
    finally:
        if merge:
            for path in (paired_output, paired_output + '.bai'):
                try:
                    os.remove(path)
                except OSError:
                    pass

    counts = receipt['counts']
    print(
        f"fiberhmm-pair: {counts.get('pairs', 0):,} pairs "
        f"[sequence {counts.get('sequence_pairs', 0):,}; "
        f"sequence-free {counts.get('sequence_free_pairs', 0):,}] "
        f"in {receipt['seconds']:.1f}s",
        file=sys.stderr,
    )


if __name__ == '__main__':
    main()
