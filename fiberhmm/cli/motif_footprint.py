"""fiberhmm-footprint: motif-anchored footprint scoring from FiberHMM calls.

The FiberHMM analogue of fibertools `ft footprint`: for each motif in a BED
and each molecule spanning it, report whether the molecule is footprinted.
Uses the calls already in the BAM (no re-decoding):
  eligible     an MSP from the BAM's FiberHMM accessibility calls spans the motif
  footprinted  one FiberHMM TF/sub-nucleosomal call (LLR >= --min-llr, width <=
               --max-width) whose bounds (outer edges by default) contain the motif

Not to be confused with fiberhmm-footprint-model (population model of recurrent
footprint geometries) or fiberhmm-consensus (posterior-based footprint classes,
the more sophisticated alternative)."""
from __future__ import annotations

import argparse
import json
import sys

from fiberhmm.inference.motif_footprint import (
    DEFAULT_MAX_WIDTH,
    MAX_LAYERS_WITHOUT_CONFIRMATION,
    BigBedMotifs,
    check_layer_count,
    MOLECULE_COLUMNS,
    TABLE_COLUMNS,
    FootprintOptions,
    load_motif_bed,
    molecule_row,
    motif_row,
    iter_footprint,
    write_annotated_bam,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog='fiberhmm-footprint', description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--bam', required=True, help='Indexed BAM with FiberHMM calls (MA tf/nuc/msp layers)')
    p.add_argument('--bed', required=True,
                   help='Motifs: BED (BED6: chrom, start, end, name, score, strand; 0-based half-open; .gz ok) or '
                        'bigBed (detected by its magic number; e.g. JASPAR genome-wide TFBS tracks; queried only '
                        'where the BAM has reads, or in --region). Each distinct name (bigBed: TF-name field when '
                        'present) is one motif set and one MA layer pair (fp_NAME, fpneg_NAME)')
    p.add_argument('--motif-name', metavar='NAME[,NAME...]',
                   help='Keep motifs whose name, TF name or matrix ID matches (case-insensitive; MA0139 matches '
                        'MA0139.2)')
    p.add_argument('--min-score', type=float, help='Keep motifs with score column >= this')
    p.add_argument('--max-pvalue', type=float,
                   help='bigBed with a p-value field only: keep motifs with p <= this (JASPAR UCSC tracks instead '
                        'encode p in the score: score = -log10(p) x 100)')
    p.add_argument('--region', action='append', metavar='CHROM[:START-END]',
                   help='bigBed only: query just these regions (repeatable; 0-based half-open)')
    p.add_argument('--allow-many-layers', action='store_true',
                   help=f'Allow more than {MAX_LAYERS_WITHOUT_CONFIRMATION} motif sets (MA layer pairs) in one run')
    p.add_argument('-o', '--out', default='-',
                   help='Per-motif table, ft footprint style (default: stdout)')
    p.add_argument('--per-molecule', metavar='TSV',
                   help='Also write one row per molecule x motif (status, code, call bounds, LLR), in genomic order of the motifs')
    p.add_argument('--out-bam', metavar='BAM',
                   help='Also write a copy of --bam with per-motif MA layers fp_NAME (footprinted) and fpneg_NAME '
                        '(eligible, not footprinted). Existing layers are kept; layers of the same name are replaced')
    p.add_argument('--layer-name', metavar='NAME',
                   help='Use one layer name for every BED row instead of the BED name column '
                        '(sanitized to [A-Za-z0-9_]; written as fp_NAME / fpneg_NAME)')
    p.add_argument('--min-llr', type=float, default=5.0,
                   help='Minimum LLR (nats) of a footprint call; tq >= 10 x this (default 5)')
    p.add_argument('--edges', choices=['outer', 'conservative'], default='outer',
                   help='Call bounds that must contain the motif: outer = conservative bounds extended through the '
                        'edge ambiguity in the el/er bytes, up to the terminating informative position (default); '
                        'conservative = the call as stored')
    p.add_argument('--outer-bound', choices=['exclusive', 'inclusive'], default='exclusive',
                   help='With --edges outer: whether the outer bound stops just before the terminating informative '
                        'position (exclusive, default) or includes it (inclusive, one more base per side)')
    p.add_argument('--max-width', type=int, default=DEFAULT_MAX_WIDTH,
                   help=f'Maximum conservative reference width (bp) of the footprint call; 0 = no cap '
                        f'(default {DEFAULT_MAX_WIDTH}: label-free 95th percentile of motif-overlapping footprint '
                        'call widths)')
    p.add_argument('--require-unmodified', action='store_true',
                   help='Also require no m6A call (ML >= --ml) inside the motif (MM/ML m6A only)')
    p.add_argument('--ml', type=int, default=125, help='ML threshold for --require-unmodified (default 125, as ft)')
    p.add_argument('--msp-bam', metavar='BAM',
                   help='Advanced: take MSPs from this BAM instead (matched by read name), e.g. the BAM ft footprint '
                        'read, for like-for-like comparisons')
    p.add_argument('-F', '--filter', type=lambda v: int(v, 0), default=0x904,
                   help='Skip reads with any of these flag bits (default 0x904: unmapped, secondary, supplementary)')
    p.add_argument('--min-mapq', type=int, default=0, help='Skip reads below this MAPQ (default 0)')
    p.add_argument('--cores', type=int, default=1, help='Worker processes over motif clusters (default 1)')
    from fiberhmm.cli.common import add_version_args
    add_version_args(p)
    return p


def _open(path):
    return sys.stdout if path == '-' else open(path, 'w')


def main(argv=None):
    p = build_parser()
    try:
        return _main(p, argv)
    except ValueError as error:
        p.exit(2, f'{p.prog}: error: {error}\n')


def _main(p, argv):
    args = p.parse_args(argv)
    if args.cores < 1:
        p.error('--cores must be >= 1')
    try:
        options = FootprintOptions(min_llr=args.min_llr, edges=args.edges, outer_bound=args.outer_bound,
                                   max_width=args.max_width, require_unmodified=args.require_unmodified,
                                   min_ml=args.ml, flag_filter=args.filter, min_mapq=args.min_mapq)
    except ValueError as error:
        p.error(str(error))
    from fiberhmm.io.bigbed_motifs import MotifFilter, is_bigbed
    names = [n for n in (args.motif_name or '').split(',') if n.strip()] or None
    filt = MotifFilter(names=names, min_score=args.min_score, max_pvalue=args.max_pvalue) \
        if (names or args.min_score is not None or args.max_pvalue is not None) else None
    bigbed = is_bigbed(args.bed)
    if not bigbed and (args.region or args.max_pvalue is not None):
        p.error('--region and --max-pvalue apply to bigBed motif input')
    if bigbed:
        motifs = BigBedMotifs(args.bed, bam_path=args.bam, regions=args.region, motif_filter=filt,
                              layer_name=args.layer_name)
    else:
        motifs = load_motif_bed(args.bed, args.layer_name, filt)
        check_layer_count({m.label for m in motifs}, args.allow_many_layers, args.layer_name)
    # Stream: each cluster's observations become table rows and per-molecule
    # rows (written at once, in genomic order), then are dropped. A BED table
    # keeps BED order (rows held until the end); a bigBed table is written as
    # it goes, in genomic order.
    out = _open(args.out)
    table = {}
    labels = set()
    count = 0
    mol = open(args.per_molecule, 'w') if args.per_molecule else None
    try:
        out.write('\t'.join(TABLE_COLUMNS) + '\n')
        if mol:
            mol.write('\t'.join(MOLECULE_COLUMNS) + '\n')
        for m, obs in iter_footprint(args.bam, motifs, options, cores=args.cores, msp_bam_path=args.msp_bam):
            count += 1
            if m.label not in labels:
                labels.add(m.label)
                check_layer_count(labels, args.allow_many_layers, args.layer_name)
            row = '\t'.join(motif_row(m, obs))
            if bigbed:
                out.write(row + '\n')
            else:
                table[m.index] = row
            if mol:
                mol.writelines('\t'.join(molecule_row(m, o)) + '\n' for o in obs)
        if not bigbed:
            out.writelines(table[m.index] + '\n' for m in motifs)
    finally:
        if mol:
            mol.close()
        if out is not sys.stdout:
            out.close()
    summary = dict(status='complete', motifs=count, motif_sets=len(labels), parameters=options.__dict__)
    if args.out_bam:
        summary['out_bam'] = write_annotated_bam(args.bam, args.out_bam, motifs, None, options, args.bed,
                                                 args.msp_bam, argv=['fiberhmm-footprint', *(argv or sys.argv[1:])],
                                                 labels=labels)
    print(json.dumps(summary), file=sys.stderr)


if __name__ == '__main__':
    main()
