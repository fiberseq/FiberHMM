"""fiberhmm-nfr (EXPERIMENTAL preview): NFR variants, per-read membership and element co-accessibility for a region.

Loads regional evidence with FiberHMM's consensus loader (the same evidence units, unit_ids, duplicate
collapse and DAF channels as fiberhmm-consensus, so results join footprint classes by unit_id), discovers NFR
variants in each NFR region (given with --nfr, or detected from the gap >= 175-bp profile), and tests element
co-accessibility. Writes variants.tsv, configurations.tsv, molecules.tsv.gz, coaccess.tsv, combos.tsv, result.json
and manifest.json to --output. Deterministic: the same inputs and parameters give byte-identical files.

This is an experimental preview; outputs, parameters and formats may change without notice.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

LOADER_GROUPS = ('input', 'compute', 'families', 'cr', 'recaller')


def main(argv=None):
    from fiberhmm.cli.common import run_reporting_input_errors
    from fiberhmm.inference.consensus.cli import report_chemistry_errors
    return run_reporting_input_errors('fiberhmm-nfr', lambda: report_chemistry_errors(lambda: _main(argv), 'fiberhmm-nfr'))


def _span(parser, value, flag):
    a, dash, b = value.replace(',', '').partition('-')
    try:
        a, b = int(a), int(b)
    except ValueError:
        parser.error(f'{flag} {value!r}: expected START-END (0-based, half-open)')
    if not dash or a < 0 or b <= a:
        parser.error(f'{flag} {value!r}: expected START-END with 0 <= START < END')
    return a, b


def _parser():
    p = argparse.ArgumentParser(prog='fiberhmm-nfr', description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group()
    src.add_argument('--bam', action='append', help='Input BAM (repeat for separate datasets); chemistry from BAM @CO metadata')
    src.add_argument('--datasets', help='JSON list of {dataset_id, paths: [BAMs], chemistry?}')
    src.add_argument('--evidence', help="Saved evidence.json.gz (e.g. a fiberhmm-consensus run's), instead of loading BAMs")
    windows = p.add_mutually_exclusive_group()
    windows.add_argument('--region', help='CHROM:START-END analysis window (0-based, half-open); use this or --bed with BAMs')
    windows.add_argument('--bed', help='BED3 windows for independent NFR runs, written in BED order to window_XXXXXX directories')
    p.add_argument('--window-jobs', type=int, default=0, metavar='N',
                   help='Concurrent BED windows (0 = up to --cores; 1 = sequential). Workers are reused across windows')
    p.add_argument('--nfr', action='append', metavar='START-END',
                   help='NFR region inside the window (repeat for several). Default: detected from the share of callable '
                        'reads inside a >= 175-bp gap (see --detect-method)')
    p.add_argument('--chemistry', choices=['ddda', 'dddb', 'hia5-pacbio', 'hia5-nanopore'],
                   help='Explicit chemistry for --bam when the BAM header does not declare one')
    p.add_argument('--parameters', help='JSON: loader groups as for fiberhmm-consensus (input, compute, ...) and an "nfr" group '
                                        'with any engine parameter (see --schema)')
    p.add_argument('--stringency', type=float, help='Prediction strength needed for k (default 0.9; 0.85 admits finer variants)')
    p.add_argument('--mode', choices=['variants', 'depth'], help='variants (discovered, default) or depth (Timer 175/300/500-bp width states)')
    p.add_argument('--elements', choices=['variants', 'nfrs'], help='Co-accessibility elements: each variant (default) or whole NFRs')
    p.add_argument('--classes', action='append', metavar='DIR',
                   help='A fiberhmm-consensus (lattice recaller) output directory: its supported footprint classes become elements '
                        '(variant x class pairs; unit_ids must come from the same datasets in the same order). Repeatable; '
                        'class ids get the prefix R1:, R2: ... when more than one is given')
    p.add_argument('--no-internal-footprint-labels', action='store_true',
                   help='Do not label a split whose internal protection matches a supported class as "full NFR with internal footprint"')
    p.add_argument('--pairs', choices=['nfr', 'all'], help='Pairs tested: with at least one NFR element (default) or also class x class')
    p.add_argument('--within-clusters', type=int, metavar='K', help='Also report the effect within K masked k-means clusters (default off)')
    p.add_argument('--combo', help='Comma-separated element ids (3-8) for combination patterns (default: automatic)')
    p.add_argument('--robust', type=int, metavar='N', help='Rerun discovery under N hashed read orders (default 0)')
    p.add_argument('--bootstrap', type=int, metavar='N', help='Bootstrap replicates for prevalence intervals (default 200)')
    p.add_argument('--detect-method', choices=['raw', 'enrichment', 'fixed'],
                   help='NFR detection: raw (default; share >= --min-fraction), enrichment (also gates on fold and q), '
                        'or fixed (share above --detect-threshold)')
    p.add_argument('--min-fraction', type=float, help='Raw/enrichment: minimum share of callable reads inside a >= 175-bp gap '
                                                      '(default raw 0.10; enrichment 0.05)')
    p.add_argument('--clip-gaps-to-region', action=argparse.BooleanOptionalAction, default=None,
                   help='Assign each gap to the NFR region with largest overlap (ties: leftmost; default off)')
    p.add_argument('--min-fold', type=float, help='Enrichment detection: share over the window\'s background at least this (default 2)')
    p.add_argument('--detect-q', type=float, help='Enrichment detection: Benjamini-Hochberg q of the binomial test (default 0.01)')
    p.add_argument('--min-open-reads', type=int, help='Raw/enrichment detection: reads inside a >= 175-bp gap at least this (default 10)')
    p.add_argument('--detect-threshold', type=float, help='Fixed detection: share of callable reads inside a >= 175-bp gap above this '
                                                          '(default 0.15; given alone, it selects the fixed rule)')
    p.add_argument('--cores', type=int, help='Worker budget (compute.cores; default 4); BED runs reuse workers across independent windows')
    from ..depth_cap import add_arguments
    add_arguments(p)
    p.add_argument('--schema', action='store_true', help='Print the engine parameters with defaults as JSON and exit')
    p.add_argument('--json-progress', action='store_true', help='Structured progress on stderr (JSON lines)')
    p.add_argument('--output', help='New or empty output directory')
    from fiberhmm.cli.common import add_version_args
    add_version_args(p)
    return p


def _main(argv=None):
    import dataclasses
    from fiberhmm.inference.consensus.artifacts import read_json
    from fiberhmm.inference.consensus.cli import Progress, check_chemistry, _check_region_contigs, _parse_regions
    from fiberhmm.inference.consensus.parameters import parse_options
    from .workflow import NFROptions, run_accessibility, load_recaller_classes
    p = _parser()
    args = p.parse_args(argv)
    if args.schema:
        print(json.dumps(NFROptions().as_dict(), indent=2, sort_keys=True))
        return
    if not args.output or not any((args.bam, args.datasets, args.evidence)):
        p.error('Supply --bam/--datasets (with --region) or --evidence, and --output')
    out = Path(args.output).resolve()
    if out.exists() and any(out.iterdir()):
        p.error('Output directory must be empty; existing results are never overwritten')
    if args.window_jobs < 0:
        p.error('--window-jobs must be >= 0')
    if args.evidence and (args.region or args.bed):
        p.error('Saved evidence already fixes the window; omit --region/--bed')
    if not args.evidence and not (args.region or args.bed):
        p.error('--bam/--datasets need --region CHROM:START-END or --bed')
    if args.bed and (args.nfr or args.classes):
        p.error('--bed detects NFRs independently per window; --nfr and --classes require a single --region')
    values = read_json(args.parameters) if args.parameters else {}
    unknown = sorted(set(values) - set(LOADER_GROUPS) - {'nfr'})
    if unknown:
        p.error(f'--parameters: unknown group(s) {", ".join(unknown)}; use loader groups ({", ".join(LOADER_GROUPS)}) and "nfr"')
    nfr = dict(values.pop('nfr', {}) or {})
    for name, value in (('stringency', args.stringency), ('mode', args.mode), ('elements', args.elements), ('pairs', args.pairs),
                        ('within_clusters', args.within_clusters), ('robust', args.robust), ('bootstrap', args.bootstrap),
                        ('detect_threshold', args.detect_threshold), ('detect_method', args.detect_method),
                        ('min_fraction', args.min_fraction), ('min_fold', args.min_fold), ('detect_q', args.detect_q),
                        ('min_open_reads', args.min_open_reads), ('clip_gaps_to_region', args.clip_gaps_to_region)):
        if value is not None:
            nfr[name] = value
    from ..depth_cap import NAMES
    for name in NAMES:
        if getattr(args, name) is not None:
            nfr[name] = getattr(args, name)
    if args.no_internal_footprint_labels:
        nfr['internal_footprint_labels'] = False
    if args.combo:
        nfr['combo_elements'] = [x.strip() for x in args.combo.split(',') if x.strip()]
    if args.nfr:
        nfr['nfr_regions'] = [_span(p, v, '--nfr') for v in args.nfr]
    try:
        opt = NFROptions.from_params(nfr)
    except (TypeError, ValueError) as error:
        p.error(str(error))
    if args.cores is not None:
        values.setdefault('compute', {})['cores'] = args.cores
    values.setdefault('cr', {}).setdefault('engine', 'lattice_recaller')
    try:
        options = parse_options(values)
    except ValueError as error:
        p.error(str(error))
    classes = []
    for i, d in enumerate(args.classes or ()):
        if not (Path(d)/'classes.tsv').is_file() or not (Path(d)/'molecules.tsv.gz').is_file():
            p.error(f'--classes {d}: not a fiberhmm-consensus output directory (classes.tsv and molecules.tsv.gz)')
        classes += load_recaller_classes(d, opt.class_min_prevalence, prefix=f'R{i + 1}:' if len(args.classes) > 1 else '')
    progress = Progress(args.json_progress)
    datasets = None
    if args.evidence:
        payload = read_json(args.evidence)
    else:
        from fiberhmm.inference.consensus.bam import load_bam_payload
        datasets = read_json(args.datasets) if args.datasets else [
            dict(dataset_id=f'dataset_{i + 1}', paths=[str(Path(b).resolve())], chemistry=args.chemistry) for i, b in enumerate(args.bam)]
        check_chemistry(p, datasets)
        if args.bed:
            if opt.nfr_regions:
                p.error('--bed requires automatic NFR detection; omit nfr.nfr_regions from --parameters')
            return _run_bed(args, p, out, datasets, values, opt, options, progress)
        window, = _parse_regions(p, [args.region])
        _check_region_contigs(p, [window], datasets)
        if window['end'] - window['start'] > options['compute'].maximum_region_bp:
            p.error(f"--region spans {window['end'] - window['start']:,} bp, more than the "
                    f"{options['compute'].maximum_region_bp:,} bp analysis limit (compute.maximum_region_bp)")
        from fiberhmm.inference.consensus.execution import single_threaded_blas
        with single_threaded_blas():
            caps = [opt.max_quantification_molecules]
            if opt.mode != 'depth' or not opt.nfr_regions:
                caps.append(opt.max_discovery_molecules)
            sampling = dict(cap=max(caps) if all(caps) else 0, discovery_cap=opt.max_discovery_molecules,
                            seed=opt.sample_seed, mode='nfr',
                            windows=opt.nfr_regions or None, nfr_options=opt)
            payload = load_bam_payload(datasets, {k: window[k] for k in ('chrom', 'start', 'end')}, options, progress,
                                       loader_sampling=sampling)
    region = payload['region']
    for a, b in opt.nfr_regions:
        if a < region['start'] or b > region['end']:
            p.error(f'--nfr {a}-{b} is outside the analysis window {region["chrom"]}:{region["start"]}-{region["end"]}')
    if not any(s['units'] for s in payload['strata']):
        p.exit(2, 'fiberhmm-nfr: error: no molecules in the window (check chromosome, coordinates, MAPQ and chemistry)\n')
    inputs = dict(datasets=datasets, evidence=str(Path(args.evidence).resolve()) if args.evidence else None,
                  loader_parameters={k: (dataclasses.asdict(v) if dataclasses.is_dataclass(v) else v) for k, v in options.items()
                                     if k == 'input'},
                  classes=[str(Path(d).resolve()) for d in args.classes or ()])
    out.mkdir(parents=True, exist_ok=True)
    from fiberhmm.inference.consensus.execution import single_threaded_blas
    with single_threaded_blas():      # threshold-sensitive clustering: one numerical thread configuration
        result = run_accessibility(payload, opt, out, progress, classes=classes or None, inputs=inputs)
    summary = [dict(id=n['id'], start=n['start'], end=n['end'], status=n['status'], callable=n['callable'], reads=n['reads'],
                    peak=n.get('profile_peak'), variants=[v['name'] for v in n['variants']]) for n in result['nfrs']]
    det = result.get('detection') or {}
    print(json.dumps(dict(status='complete', experimental=True, output=str(out), nfrs=summary, pairs=len(result['pairs']),
                          detection={k: det.get(k) for k in ('method', 'background', 'enrichment_test', 'enrichment_gates', 'note') if k in det},
                          warnings=result['warnings'])))


def _bed_task(spec):
    """One independent site; the process pool keeps imports and JIT kernels warm."""
    import dataclasses
    from fiberhmm.inference.consensus.bam import load_bam_payload
    from fiberhmm.inference.consensus.parameters import parse_options
    from fiberhmm.inference.consensus.execution import single_threaded_blas
    from .workflow import NFROptions, run_accessibility
    options = parse_options(spec['values'])
    opt = NFROptions.from_params(spec['nfr'])
    caps = [opt.max_quantification_molecules, opt.max_discovery_molecules]
    sampling = dict(cap=max(caps) if all(caps) else 0, discovery_cap=opt.max_discovery_molecules,
                    seed=opt.sample_seed, mode='nfr', windows=None, nfr_options=opt)
    with single_threaded_blas():
        payload = load_bam_payload(spec['datasets'], spec['region'], options, loader_sampling=sampling)
        if not any(s['units'] for s in payload['strata']):
            raise ValueError(f"{spec['name']}: no molecules in the window (check chromosome, coordinates, MAPQ and chemistry)")
        inputs = dict(datasets=spec['datasets'], evidence=None,
                      loader_parameters={'input': dataclasses.asdict(options['input'])}, classes=[])
        result = run_accessibility(payload, opt, spec['output'], inputs=inputs)
    return dict(name=spec['name'], region=spec['region'], output=spec['output'],
                nfrs=len(result['nfrs']), pairs=len(result['pairs']), warnings=result['warnings'])


def _run_bed(args, parser, out, datasets, values, opt, options, progress):
    from copy import deepcopy
    from fiberhmm.inference.consensus.regions import load_bed
    from fiberhmm.inference.consensus.cli import _check_window_sizes, _check_region_contigs
    from fiberhmm.inference.consensus.execution import shared_worker_pool
    from fiberhmm.inference.consensus.artifacts import write_json
    windows = load_bed(args.bed)
    _check_window_sizes(parser, windows, options['compute'].maximum_region_bp, 'bed')
    _check_region_contigs(parser, windows, datasets)
    jobs = min(len(windows), options['compute'].cores, args.window_jobs or options['compute'].cores)
    values = deepcopy(values)
    values.setdefault('compute', {})['cores'] = 1
    specs = [dict(name=w['name'], region={k: w[k] for k in ('chrom', 'start', 'end')},
                  datasets=datasets, values=values, nfr=opt.as_dict(), output=str(out/f'window_{i+1:06d}'))
             for i, w in enumerate(windows)]
    out.mkdir(parents=True, exist_ok=True)
    with shared_worker_pool(jobs) as pool:
        results = map(_bed_task, specs) if pool is None else pool.get().map(_bed_task, specs)
        summary = []
        for i, row in enumerate(results):
            summary.append(row)
            progress.report('windows', f"{row['name']} complete", completed=i+1, total=len(specs))
    write_json(out/'regions.json', summary)
    print(json.dumps(dict(status='complete', experimental=True, output=str(out), regions=summary)))


if __name__ == '__main__':
    main()
