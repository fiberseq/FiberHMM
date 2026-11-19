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


class _GroupAction(argparse.Action):
    """--bam and --calls-bb both start a cell-type group; keep them in command-line order."""

    def __call__(self, parser, namespace, values, option_string=None):
        groups = list(getattr(namespace, self.dest, None) or [])
        groups.append(('bam' if option_string == '--bam' else 'bb', values))
        setattr(namespace, self.dest, groups)


PER_GROUP = ('msp_bam', 'tf_bb', 'msp_bb', 'nuc_bb', 'm6a_bb')


def build_parser():
    p = argparse.ArgumentParser(prog='fiberhmm-footprint calibrate', description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--bed', required=True, help='Motif BED (BED6), the same motifs for every cell')
    src = p.add_argument_group(
        'call inputs, one group per cell type (as fiberhmm-footprint)',
        'Each --bam or --calls-bb starts a cell-type group; --chip, --cell and the per-group source options '
        'below pair with the groups in order (one value per group; "-" = none for that group). Within a group, '
        'several BAMs are comma-separated, and so are their --msp-bam files.')
    src.add_argument('--bam', action=_GroupAction, dest='groups', metavar='BAM[,BAM...]',
                     help='BAM(s) of one cell type, comma-separated; repeat per cell type, each with its own --chip')
    src.add_argument('--calls-bb', action=_GroupAction, dest='groups', metavar='PREFIX',
                     help='Instead of --bam: FiberHMM call bigBeds of one cell type (PREFIX_tf.bb, PREFIX_msp.bb, ...)')
    src.add_argument('--msp-bam', action='append', metavar='BAM[,BAM...]|-',
                     help='Per --bam group: take MSPs from these BAMs (matched by read name), as fiberhmm-footprint '
                          '--msp-bam')
    for flag, what in (('--tf-bb', 'TF call'), ('--msp-bb', 'MSP'), ('--nuc-bb', 'nucleosome'), ('--m6a-bb', 'm6A')):
        src.add_argument(flag, action='append', metavar='BB|-', help=f'Per --calls-bb group: explicit {what} bigBed')
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


def collect(cell_name, sources, motifs, labels, options, cores):
    """One cell: qualifying widths per eligible molecule, per motif, pooled over its call sources.

    ``sources``: ``[(source, msp_bam)]`` as returned by the main command's ``resolve_call_source``."""
    from fiberhmm.inference.footprint_calibration import CellData
    from fiberhmm.inference.motif_footprint import iter_footprint
    per_motif = defaultdict(list)
    for source, msp_bam in sources:
        for m, obs in iter_footprint(source, motifs, options, cores=cores, msp_bam_path=msp_bam):
            per_motif[m.index].extend(o.widths for o in obs if o.code in (1, 3))
    return CellData(cell_name, [m.chrom for m in motifs], labels, [per_motif.get(m.index, []) for m in motifs])


def group_sources(kind, value, extra, options):
    """Call sources of one cell-type group through the main command's resolve_call_source."""
    from fiberhmm.cli.motif_footprint import resolve_call_source
    if kind == 'bam':
        bams = [b for b in value.split(',') if b]
        msp = [m for m in (extra.get('msp_bam') or '').split(',') if m] if extra.get('msp_bam') else []
        if msp and len(msp) != len(bams):
            raise ValueError(f'--msp-bam {extra["msp_bam"]}: give one MSP BAM per BAM of the group ({value})')
        for key in ('tf_bb', 'msp_bb', 'nuc_bb', 'm6a_bb'):
            if extra.get(key):
                raise ValueError(f'--{key.replace("_", "-")} applies to --calls-bb groups')
        return [resolve_call_source(options, bam=b, msp_bam=msp[i] if msp else None)[:2] for i, b in enumerate(bams)]
    if extra.get('msp_bam'):
        raise ValueError('--msp-bam applies to --bam groups (a --calls-bb group takes MSPs from its msp bigBed)')
    return [resolve_call_source(options, calls_bb=value, tf_bb=extra.get('tf_bb'), msp_bb=extra.get('msp_bb'),
                                nuc_bb=extra.get('nuc_bb'), m6a_bb=extra.get('m6a_bb'))[:2]]


def main(argv=None):
    p = build_parser()
    args = p.parse_args(argv)
    from fiberhmm.inference import footprint_calibration as C
    from fiberhmm.inference.motif_footprint import FootprintOptions, load_motif_bed
    groups = args.groups or []
    if not groups:
        p.error('give at least one call-input group: --bam or --calls-bb')
    try:
        if args.chip and len(args.chip) != len(groups):
            p.error('give one --chip per --bam/--calls-bb group')
        if not args.chip and args.label_column is None:
            p.error('give ChIP labels: --chip per group, or --label-column')
        if args.cell and len(args.cell) != len(groups):
            p.error('give one --cell per group, or none')
        extras = [dict() for _ in groups]
        for key in PER_GROUP:
            values = getattr(args, key) or []
            if values and len(values) != len(groups):
                p.error(f'--{key.replace("_", "-")}: give one value per group ("-" for none), {len(groups)} in all')
            for e, v in zip(extras, values):
                e[key] = None if v in ('-', '') else v
        names = args.cell or [f'cell{i + 1}' for i in range(len(groups))]
        if len(set(names)) != len(names):
            p.error('--cell names must be distinct')
        wmins, wmaxs = C.parse_range(args.wmin_range), C.parse_range(args.wmax_range)
        options = FootprintOptions(min_llr=args.min_llr, edges=args.edges, outer_bound=args.outer_bound,
                                   max_width=0, min_width=0, require_unmodified=args.require_unmodified,
                                   min_ml=args.ml, flag_filter=args.filter, min_mapq=args.min_mapq)
        motifs = load_motif_bed(args.bed, layer_name='calibration')
        cells = []
        for i, (name, (kind, value)) in enumerate(zip(names, groups)):
            labels = C.label_motifs(motifs, args.chip[i] if args.chip else None, args.label_column, args.bed)
            print(f'{name}: {int(labels.sum())} ChIP+ / {len(labels)} motifs; scoring molecules', file=sys.stderr)
            sources = group_sources(kind, value, extras[i], options)
            cells.append(collect(name, sources, motifs, labels, options, args.cores))
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
        cells=[dict(name=c.name, motifs=len(c.labels), chip_positive=int(c.labels.sum()),
                    calls={'bam' if kind == 'bam' else 'calls_bb': value,
                           **{k: v for k, v in extra.items() if v}})
               for c, (kind, value), extra in zip(cells, groups, extras)],
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
