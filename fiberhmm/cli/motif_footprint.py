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
    MOLECULE_COLUMNS,
    TABLE_COLUMNS,
    FootprintOptions,
    load_motif_bed,
    molecule_row,
    motif_row,
    run_footprint,
    write_annotated_bam,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog='fiberhmm-footprint', description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--bam', required=True, help='Indexed BAM with FiberHMM calls (MA tf/nuc/msp layers)')
    p.add_argument('--bed', required=True,
                   help='Motif BED (BED6: chrom, start, end, name, score, strand; 0-based half-open; .gz ok). '
                        'Each distinct name is one motif set and one MA layer pair (fp_NAME, fpneg_NAME)')
    p.add_argument('-o', '--out', default='-',
                   help='Per-motif table, ft footprint style (default: stdout)')
    p.add_argument('--per-molecule', metavar='TSV',
                   help='Also write one row per molecule x motif (status, code, call bounds, LLR)')
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
                        f'(default {DEFAULT_MAX_WIDTH}, PROVISIONAL)')
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
    args = p.parse_args(argv)
    if args.cores < 1:
        p.error('--cores must be >= 1')
    try:
        options = FootprintOptions(min_llr=args.min_llr, edges=args.edges, outer_bound=args.outer_bound,
                                   max_width=args.max_width, require_unmodified=args.require_unmodified,
                                   min_ml=args.ml, flag_filter=args.filter, min_mapq=args.min_mapq)
    except ValueError as error:
        p.error(str(error))
    motifs = load_motif_bed(args.bed, args.layer_name)
    by_motif = run_footprint(args.bam, motifs, options, cores=args.cores, msp_bam_path=args.msp_bam,
                             want_spans=bool(args.out_bam))
    out = _open(args.out)
    try:
        out.write('\t'.join(TABLE_COLUMNS) + '\n')
        for m in motifs:
            out.write('\t'.join(motif_row(m, by_motif.get(m.index, ()))) + '\n')
    finally:
        if out is not sys.stdout:
            out.close()
    if args.per_molecule:
        with open(args.per_molecule, 'w') as fh:
            fh.write('\t'.join(MOLECULE_COLUMNS) + '\n')
            for m in motifs:
                for o in by_motif.get(m.index, ()):
                    fh.write('\t'.join(molecule_row(m, o)) + '\n')
    summary = dict(status='complete', motifs=len(motifs), parameters=options.__dict__)
    if args.out_bam:
        summary['out_bam'] = write_annotated_bam(args.bam, args.out_bam, motifs, by_motif, options, args.bed,
                                                 args.msp_bam, argv=['fiberhmm-footprint', *(argv or sys.argv[1:])])
    print(json.dumps(summary), file=sys.stderr)


if __name__ == '__main__':
    main()
