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


def parse_parameters(parser, value, flag='--parameters'):
    """--parameters as a JSON file path or an inline JSON object (text starting with '{')."""
    from fiberhmm.inference.consensus.artifacts import read_json
    text = str(value).strip()
    if text.startswith('{'):
        try:
            values = json.loads(text)
        except ValueError as error:
            parser.error(f"{flag}: inline JSON does not parse ({error}). Quote it for the shell, which otherwise strips the "
                         f"double quotes: {flag} '{{\"nfr\":{{\"min_gap_bp\":150}}}}'")
    elif Path(text).is_file():
        try:
            values = read_json(text)
        except (OSError, ValueError) as error:
            parser.error(f'{flag} {text}: not readable JSON ({error})')
    else:
        parser.error(f"{flag} {value!r}: neither a JSON file nor an inline JSON object (inline JSON starts with '{{')")
    if not isinstance(values, dict):
        parser.error(f'{flag}: expected a JSON object of parameter groups, got {type(values).__name__}')
    return values


def _parser():
    from fiberhmm import defaults as D
    p = argparse.ArgumentParser(prog='fiberhmm-nfr', description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group()
    src.add_argument('--bam', action='append', help='Input BAM (repeat for separate datasets); chemistry from BAM @CO metadata')
    src.add_argument('--datasets', help='JSON list of {dataset_id, paths: [BAMs], chemistry?}')
    src.add_argument('--evidence', help="Saved evidence.json.gz (e.g. a fiberhmm-consensus run's), instead of loading BAMs")
    windows = p.add_mutually_exclusive_group()
    windows.add_argument('--region', help='CHROM:START-END analysis window (0-based, half-open); use this or --bed with BAMs')
    windows.add_argument('--bed', '--regions-bed', dest='bed',
                         help='BED3/BED4 windows for independent NFR runs (one BAM pass per window, windows spread over '
                              '--cores), written in BED order to window_XXXXXX directories; names, status and a combined '
                              'variants table in regions.json/regions.tsv/variants.tsv. Add --continue to finish an '
                              'interrupted run in the same --output')
    p.add_argument('--window-jobs', type=int, default=D.NFR_WINDOW_JOBS, metavar='N',
                   help='Concurrent BED windows, one core each (0 = up to --cores; 1 = sequential, in this process). Workers '
                        'are reused across windows')
    p.add_argument('--continue', dest='continue_run', action='store_true',
                   help='--bed: finish an interrupted run in the same --output, skipping windows already complete '
                        '(refused when the BAMs, BED or parameters differ)')
    p.add_argument('--nfr', action='append', metavar='START-END',
                   help='NFR region inside the window (repeat for several). Default: detected from the share of callable '
                        'reads inside a >= 175-bp gap (see --detect-method)')
    p.add_argument('--chemistry', choices=['ddda', 'dddb', 'hia5-pacbio', 'hia5-nanopore'],
                   help='Explicit chemistry for --bam when the BAM header does not declare one')
    p.add_argument('--parameters', metavar='FILE_OR_JSON',
                   help='A JSON file, or an inline JSON object (quote it for the shell, e.g. \'{"nfr":{"min_gap_bp":150}}\'): '
                        'loader groups as for fiberhmm-consensus (input, compute, ...) and an "nfr" group with any engine '
                        'parameter (see --schema)')
    p.add_argument('--skip', metavar='STAGES',
                   help=f'Comma-separated optional stages to leave out: {", ".join(D.NFR_STAGES)} (pair tests and combination '
                        'patterns). Variants, configurations and molecules.tsv.gz are unchanged; skipped stages write no '
                        'coaccess.tsv/combos.tsv (default: none skipped)')
    p.add_argument('--discover-fold-spec', metavar='SPEC',
                   help='Cross-fitted discovery: detect NFRs and discover variants on reads whose read-name hash fold is '
                        'listed, assign membership to every read, and add fold/discovery columns to molecules.tsv.gz. '
                        'SPEC: "hash=md5,mod=10,discover=0-4" (fold = int(md5(read_name), 16) %% 10; several folds '
                        '"discover=0+2+4"). Default off')
    p.add_argument('--fold-weights', choices=['all', 'heldout'],
                   help='With --discover-fold-spec: fit the configuration EM weights on all reads (default) or on the '
                        'held-out folds only; memberships of all reads use those weights')
    p.add_argument('--edge-tolerance-bp', type=int, metavar='BP',
                   help=f'Edge consistency (variants.tsv edge_within): both edges within BP of the variant\'s edges '
                        f'(default {D.NFR_EDGE_TOLERANCE_BP})')
    p.add_argument('--fast-landscape', action='store_true',
                   help=f'Preset for landscape runs: --skip {",".join(D.NFR_FAST_LANDSCAPE_SKIP)} (variants, configurations and '
                        'per-read membership only)')
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
    from contextlib import ExitStack
    with ExitStack() as stack:      # the run's worker pool (load and analysis share it) closes with the run
        return _run(argv, stack)


def _run(argv, stack):
    import dataclasses
    from fiberhmm import defaults as D
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
    if args.continue_run and not args.bed:
        p.error('--continue applies to --bed runs')
    if out.exists() and any(out.iterdir()) and not args.continue_run:
        p.error('Output directory must be empty; existing results are never overwritten'
                + (' (add --continue to finish an interrupted --bed run)' if args.bed else ''))
    if args.window_jobs < 0:
        p.error('--window-jobs must be >= 0')
    if args.evidence and (args.region or args.bed):
        p.error('Saved evidence already fixes the window; omit --region/--bed')
    if not args.evidence and not (args.region or args.bed):
        p.error('--bam/--datasets need --region CHROM:START-END or --bed')
    if args.bed and (args.nfr or args.classes):
        p.error('--bed detects NFRs independently per window; --nfr and --classes require a single --region')
    values = parse_parameters(p, args.parameters) if args.parameters is not None else {}
    unknown = sorted(set(values) - set(LOADER_GROUPS) - {'nfr'})
    if unknown:
        p.error(f'--parameters: unknown group(s) {", ".join(unknown)}; use loader groups ({", ".join(LOADER_GROUPS)}) and "nfr"')
    nfr = dict(values.pop('nfr', {}) or {})
    for name, value in (('stringency', args.stringency), ('mode', args.mode), ('elements', args.elements), ('pairs', args.pairs),
                        ('within_clusters', args.within_clusters), ('robust', args.robust), ('bootstrap', args.bootstrap),
                        ('detect_threshold', args.detect_threshold), ('detect_method', args.detect_method),
                        ('min_fraction', args.min_fraction), ('min_fold', args.min_fold), ('detect_q', args.detect_q),
                        ('min_open_reads', args.min_open_reads), ('clip_gaps_to_region', args.clip_gaps_to_region),
                        ('discover_fold_spec', args.discover_fold_spec), ('fold_weights', args.fold_weights),
                        ('edge_tolerance_bp', args.edge_tolerance_bp)):
        if value is not None:
            nfr[name] = value
    from ..depth_cap import NAMES
    for name in NAMES:
        if getattr(args, name) is not None:
            nfr[name] = getattr(args, name)
    skip = nfr.get('skip') or ()
    skip = set(skip.split(',') if isinstance(skip, str) else skip)
    if args.skip is not None:
        skip |= {x.strip() for x in args.skip.split(',') if x.strip()}
    if args.fast_landscape:
        skip |= set(D.NFR_FAST_LANDSCAPE_SKIP)
    if skip:
        nfr['skip'] = sorted(skip)
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
        from fiberhmm.inference.consensus.execution import single_threaded_blas, shared_worker_pool
        stack.enter_context(shared_worker_pool(options['compute'].cores))
        with single_threaded_blas():
            caps = [opt.max_quantification_molecules]
            if opt.mode != 'depth' or not opt.nfr_regions:
                caps.append(opt.max_discovery_molecules)
            sampling = dict(cap=max(caps) if all(caps) else 0, discovery_cap=opt.max_discovery_molecules,
                            seed=opt.sample_seed, mode='nfr',
                            windows=opt.nfr_regions or None, nfr_options=opt)
            payload = load_payload(datasets, {k: window[k] for k in ('chrom', 'start', 'end')}, values, options, progress,
                                   sampling, jobs=options['compute'].cores)
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
        result = run_accessibility(payload, opt, out, progress, classes=classes or None, inputs=inputs,
                                   jobs=options['compute'].cores)
    summary = [dict(id=n['id'], start=n['start'], end=n['end'], status=n['status'], callable=n['callable'], reads=n['reads'],
                    peak=n.get('profile_peak'), variants=[v['name'] for v in n['variants']]) for n in result['nfrs']]
    det = result.get('detection') or {}
    print(json.dumps(dict(status='complete', experimental=True, output=str(out), nfrs=summary, pairs=len(result['pairs']),
                          detection={k: det.get(k) for k in ('method', 'background', 'enrichment_test', 'enrichment_gates', 'note') if k in det},
                          warnings=result['warnings'])))


def _load_dataset_task(args):
    """One dataset of a run loaded in a worker process (unit_ids keep the dataset's ordinal in the run)."""
    ordinal, item, region, values, sampling = args
    from fiberhmm.inference.consensus.bam import load_bam_payload
    from fiberhmm.inference.consensus.parameters import parse_options
    from fiberhmm.inference.consensus.execution import single_threaded_blas
    from .gaps import compact_unit
    with single_threaded_blas():
        return load_bam_payload([item], region, parse_options(values), loader_sampling=sampling, unit_hook=compact_unit,
                                dataset_ordinals={item['dataset_id']: ordinal})


def load_payload(datasets, region, values, options, progress=None, sampling=None, jobs=1):
    """The NFR loader: ``load_bam_payload`` with the compact unit hook. With jobs > 1, several datasets and no molecule
    cap (a cap samples the datasets jointly), each dataset is loaded in its own worker process; the payload is the
    one a single sequential load builds (strata and input files in dataset order, the same unit_ids)."""
    from fiberhmm.inference.consensus.bam import load_bam_payload
    from .gaps import compact_unit
    ids = [d['dataset_id'] for d in datasets]
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate dataset ID: ' + next(i for i in ids if ids.count(i) > 1))
    jobs = min(int(jobs or 1), len(datasets))
    if jobs <= 1 or (sampling and sampling.get('cap')):
        return load_bam_payload(datasets, region, options, progress, loader_sampling=sampling, unit_hook=compact_unit)
    from .workflow import worker_pool
    if progress is not None:
        progress.report('loading', f'loading {len(datasets)} datasets in {jobs} worker processes')
    with worker_pool(jobs) as executor:
        parts = list(executor.map(_load_dataset_task, [(i, item, region, values, sampling) for i, item in enumerate(datasets)]))
    for item, part in zip(datasets, parts):
        if progress is not None:
            progress.report('loading', f"{item['dataset_id']}: {sum(len(s['units']) for s in part['strata'])} molecules loaded",
                            dataset_id=item['dataset_id'])
    return dict(parts[0], strata=[s for part in parts for s in part['strata']],
                input_files=[f for part in parts for f in part['input_files']])


def _run_bed(args, parser, out, datasets, values, opt, options, progress):
    from fiberhmm.inference.consensus.regions import load_bed
    from fiberhmm.inference.consensus.cli import _check_window_sizes, _check_region_contigs
    from .batch import ContinueRefused, run_windows
    try:
        windows = load_bed(args.bed)
    except (OSError, ValueError) as error:
        parser.error(f'--bed {args.bed}: {error}')
    _check_window_sizes(parser, windows, options['compute'].maximum_region_bp, 'bed')
    _check_region_contigs(parser, windows, datasets)
    try:
        summary = run_windows(out, windows, datasets, values, opt.as_dict(), cores=options['compute'].cores,
                              window_jobs=args.window_jobs, continue_run=args.continue_run, progress=progress)
    except ContinueRefused as error:
        parser.error(str(error))
    counts = {}
    for row in summary:
        counts[row['status']] = counts.get(row['status'], 0) + 1
    failed = counts.get('failed', 0)
    print(json.dumps(dict(status='complete' if not failed else 'incomplete', experimental=True, output=str(out),
                          windows=counts, regions=[{k: r.get(k) for k in ('name', 'status', 'nfrs', 'variants', 'seconds', 'error')}
                                                   for r in summary])))
    if failed:
        parser.exit(1, f'fiberhmm-nfr: {failed} window(s) failed (see regions.tsv); fix the cause and rerun the same command '
                       'with --continue to redo only those\n')


if __name__ == '__main__':
    main()
