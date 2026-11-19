"""fiberhmm-footprint calibrate: learn the footprint width window from ChIP-seq labels.

Grid-searches [--min-width, --max-width] jointly to maximise the per-motif AUROC
of the footprinted fraction against ChIP labels (mean of per-cell AUROCs), with
held-out evaluation, and reports:
  - the optimal window and a ready-to-paste command line,
  - a robust range: the bounds of every window within --near (0.005) AUROC of the optimum,
  - held-out AUROC: leave-one-cell-type-out with several --bam/--chip pairs,
    otherwise odd versus even chromosomes (both directions),
  - the full 2D AUROC surface (TSV) and, with matplotlib, a heatmap,
  - a settings JSON that `fiberhmm-footprint --settings` loads, so a window learned
    in one cell type can be applied to another.

A learned window is factor-specific; the held-out AUROC is the honest number (the
in-sample optimum overstated AUROC by 0.006-0.034 in the benchmark).

Example (two cell types):
  fiberhmm-footprint calibrate --bed ctcf_motifs.bed \\
      --bam h1_a.bam,h1_b.bam --chip h1_ctcf_peaks.bed --cell H1 \\
      --bam k562.bam --chip k562_ctcf_peaks.bed --cell K562 \\
      --require-unmodified -o ctcf_window
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict

import numpy as np


def build_parser():
    p = argparse.ArgumentParser(prog='fiberhmm-footprint calibrate', description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--bed', required=True, help='Motif BED (BED6), the same motifs for every cell')
    p.add_argument('--bam', action='append', required=True, metavar='BAM[,BAM...]',
                   help='BAM(s) of one cell type, comma-separated; repeat per cell type, each with its own --chip')
    p.add_argument('--chip', action='append', metavar='PEAKS',
                   help='ChIP-seq peak BED for the matching --bam (a motif overlapping a peak is positive)')
    p.add_argument('--label-column', type=int, metavar='N',
                   help='Instead of --chip: 1-based column of the motif BED holding 1/0 labels (same for every cell)')
    p.add_argument('--cell', action='append', metavar='NAME', help='Name of the matching --bam cell type (optional)')
    p.add_argument('--wmin-range', default='0:80:5', metavar='START:STOP:STEP',
                   help='Grid of minimum widths, bp (default 0:80:5, as the benchmark)')
    p.add_argument('--wmax-range', default='10:150:5', metavar='START:STOP:STEP',
                   help='Grid of maximum widths, bp (default 10:150:5)')
    p.add_argument('--holdout', choices=['auto', 'cell', 'chromosome'], default='auto',
                   help='Held-out design: leave-one-cell-type-out (needs >= 2 cells) or odd/even chromosomes; '
                        'auto = cell when several cells are given (default)')
    p.add_argument('--near', type=float, default=0.005,
                   help='Robust range: windows within this AUROC of the optimum (default 0.005)')
    p.add_argument('-o', '--out', required=True, metavar='PREFIX',
                   help='Writes PREFIX.surface.tsv, PREFIX.heldout.tsv, PREFIX.settings.json and, with matplotlib, '
                        'PREFIX.heatmap.png')
    p.add_argument('--no-plot', action='store_true', help='Skip the heatmap')
    rule = p.add_argument_group('footprint rule (as fiberhmm-footprint; saved in the settings JSON)')
    rule.add_argument('--min-llr', type=float, default=5.0)
    rule.add_argument('--edges', choices=['outer', 'conservative'], default='outer')
    rule.add_argument('--outer-bound', choices=['exclusive', 'inclusive'], default='exclusive')
    rule.add_argument('--require-unmodified', action='store_true',
                      help='No m6A (ML >= --ml) in the motif; the benchmark windows were learned with it')
    rule.add_argument('--ml', type=int, default=125)
    rule.add_argument('-F', '--filter', type=lambda v: int(v, 0), default=0x904)
    rule.add_argument('--min-mapq', type=int, default=0)
    p.add_argument('--cores', type=int, default=1, help='Worker processes (default 1)')
    return p


def collect(cell_name, bams, motifs, labels, options, cores):
    """One cell: qualifying widths per eligible molecule, per motif, pooled over its BAMs."""
    from fiberhmm.inference.footprint_calibration import CellData
    from fiberhmm.inference.motif_footprint import iter_footprint
    per_motif = defaultdict(list)
    for bam in bams:
        for m, obs in iter_footprint(bam, motifs, options, cores=cores):
            per_motif[m.index].extend(o.widths for o in obs if o.code in (1, 3))
    return CellData(cell_name, [m.chrom for m in motifs], labels, [per_motif.get(m.index, []) for m in motifs])


def main(argv=None):
    p = build_parser()
    args = p.parse_args(argv)
    from fiberhmm.inference import footprint_calibration as C
    from fiberhmm.inference.motif_footprint import FootprintOptions, load_motif_bed
    try:
        if args.chip and len(args.chip) != len(args.bam):
            p.error('give one --chip per --bam group')
        if not args.chip and args.label_column is None:
            p.error('give ChIP labels: --chip per --bam, or --label-column')
        if args.cell and len(args.cell) != len(args.bam):
            p.error('give one --cell per --bam group, or none')
        names = args.cell or [f'cell{i + 1}' for i in range(len(args.bam))]
        if len(set(names)) != len(names):
            p.error('--cell names must be distinct')
        wmins, wmaxs = C.parse_range(args.wmin_range), C.parse_range(args.wmax_range)
        options = FootprintOptions(min_llr=args.min_llr, edges=args.edges, outer_bound=args.outer_bound,
                                   max_width=0, min_width=0, require_unmodified=args.require_unmodified,
                                   min_ml=args.ml, flag_filter=args.filter, min_mapq=args.min_mapq)
        motifs = load_motif_bed(args.bed, layer_name='calibration')
        cells = []
        for i, (name, group) in enumerate(zip(names, args.bam)):
            labels = C.label_motifs(motifs, args.chip[i] if args.chip else None, args.label_column, args.bed)
            print(f'{name}: {int(labels.sum())} ChIP+ / {len(labels)} motifs; scoring molecules', file=sys.stderr)
            cells.append(collect(name, [b for b in group.split(',') if b], motifs, labels, options, args.cores))
        design = args.holdout if args.holdout != 'auto' else ('cell' if len(cells) > 1 else 'chromosome')
        cache = {}
        surf = C.surface(cells, wmins, wmaxs, cache=cache)
        wmin, wmax, best = C.optimum(surf.mean, wmins, wmaxs)
        robust = C.robust_range(surf.mean, wmins, wmaxs, args.near)
        rows = C.held_out(cells, wmins, wmaxs, design, cache)
    except ValueError as error:
        p.exit(2, f'{p.prog}: error: {error}\n')
    ho = [r['heldout_auroc'] for r in rows if not np.isnan(r['heldout_auroc'])]
    label_free = None
    if 0 in wmins and 74 in wmaxs:
        label_free = float(surf.mean[wmins.index(0), wmaxs.index(74)])
    calibration = dict(
        method='mean per-cell AUROC of the per-motif footprinted fraction vs ChIP labels; grid search over '
               '[min_width, max_width] (conservative call width, inclusive)',
        cells=[dict(name=c.name, motifs=len(c.labels), chip_positive=int(c.labels.sum())) for c in cells],
        motif_bed=args.bed, grid=dict(wmin=args.wmin_range, wmax=args.wmax_range),
        optimum=dict(min_width=wmin, max_width=wmax, insample_auroc=best),
        robust_range=robust, label_free_0_74_auroc=label_free,
        held_out=dict(design=design, mean_auroc=float(np.mean(ho)) if ho else None, folds=rows),
        note='Factor-specific window; the held-out AUROC is the honest estimate.')
    payload = C.settings_payload(wmin, wmax, options, calibration)
    with open(f'{args.out}.settings.json', 'w') as fh:
        json.dump(payload, fh, indent=2)
    C.write_surface_tsv(f'{args.out}.surface.tsv', surf)
    import csv
    with open(f'{args.out}.heldout.tsv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), delimiter='\t')
        w.writeheader()
        w.writerows(rows)
    plotted = False if args.no_plot else C.write_heatmap(f'{args.out}.heatmap.png', surf, (wmin, wmax), rows, args.near)
    print(f'optimal window: [{wmin}, {wmax}] bp (in-sample mean AUROC {best:.3f}'
          + (f'; label-free 0-74 bp: {label_free:.3f}' if label_free is not None else '') + ')')
    print(f'robust range (within {args.near} AUROC, {robust["windows"]} windows): '
          f'min width {robust["wmin"][0]}-{robust["wmin"][1]}, max width {robust["wmax"][0]}-{robust["wmax"][1]}')
    held = ', '.join(f'{r["held_out"]} [{r["wmin"]}, {r["wmax"]}] {r["heldout_auroc"]:.3f}' for r in rows)
    print(f'held-out ({design}): mean AUROC {np.mean(ho):.3f}; {held}' if ho else f'held-out ({design}): not scorable')
    flags = ' --require-unmodified' if args.require_unmodified else ''
    print(f'use: fiberhmm-footprint --min-width {wmin} --max-width {wmax}{flags}')
    print(f'  or: fiberhmm-footprint --settings {args.out}.settings.json')
    print(f'wrote {args.out}.surface.tsv, {args.out}.heldout.tsv, {args.out}.settings.json'
          + (f', {args.out}.heatmap.png' if plotted else ' (no heatmap: install matplotlib, fiberhmm[plots])'
             if not args.no_plot else ''), file=sys.stderr)
    return 0


if __name__ == '__main__':
    main()
