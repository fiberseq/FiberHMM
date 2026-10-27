"""fiberhmm-scan: pooled feature scans (footprint classes, NFRs, accessibility peaks) over fixed units.

Scans FiberHMM-called BAMs genome-wide (half-overlapping 50 kb units, each owning
its central 25 kb), over --region windows or over BED units, and writes one
versioned scan directory: manifest.json, bgzip + tabix tables of pooled hits and
values per module, and per-molecule results as MA tags in annotated BAMs
(genome scans; BED-scoped scans write them as tables). Units are resumable
(--continue), and a rerun gives byte-identical outputs.

Association statistics from a scan directory: `fiberhmm-scan assoc SCAN_DIR --mode ...` (see
`fiberhmm-scan assoc --help`).

Format reference: docs/reference/scan-format.md.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

DEFAULT_MODULES = 'fp,nfr,peaks'


def _parser():
    p = argparse.ArgumentParser(prog='fiberhmm-scan', description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group()
    src.add_argument('--bam', action='append', help='Input BAM (repeat for separate datasets); FiberHMM-called '
                                                    '(MA tags from fiberhmm-call/apply). Chemistry from @CO metadata')
    src.add_argument('--datasets', help='JSON list of {dataset_id, paths: [BAMs], chemistry?}')
    where = p.add_mutually_exclusive_group()
    where.add_argument('--genome', action='store_true', help='Scan whole contigs on the fixed grid (see --contigs '
                                                             'and --zones)')
    where.add_argument('--region', action='append', metavar='CHROM:START-END',
                       help='Scan a window (0-based, half-open; repeatable). A window longer than --unit-bp is '
                            'grid-planned restricted to it')
    where.add_argument('--bed', help='BED units (overlapping or touching intervals are merged; longer than '
                                     '--unit-bp: grid-planned inside)')
    p.add_argument('--contigs', help='With --genome: comma-separated contigs (default: every contig in the header)')
    p.add_argument('--zones', action='append', metavar='CHROM:START-END',
                   help='Scan only the genome-grid units whose owned zones meet this window (0-based, half-open; '
                        'repeatable; implies --genome). Units keep their genome-grid spans (not clipped to the '
                        'window), so their results are the genome scan\'s (the Python API\'s run_units(..., '
                        'extend=True) reuses them across plans; --continue needs the same plan). '
                        'FiberBrowser\'s view scans print this form')
    p.add_argument('--modules', default=DEFAULT_MODULES,
                   help=f'Comma-separated modules (default {DEFAULT_MODULES}; nfr is an EXPERIMENTAL preview)')
    p.add_argument('--parameters', help='JSON {"fp": {...}, "nfr": {...}, "peaks": {...}, "read_policy": {...}}; '
                                        'see --schema')
    p.add_argument('--chemistry', choices=['ddda', 'dddb', 'hia5-pacbio', 'hia5-nanopore'],
                   help='Chemistry for --bam inputs whose header does not declare one')
    p.add_argument('--unit-bp', type=int, default=50_000, help='Grid unit span (default 50000; at most the engines\' '
                                                               '50 kb budget)')
    p.add_argument('--stride-bp', type=int, default=25_000, help='Grid stride = owned zone (default 25000)')
    p.add_argument('--grid-offset', type=int, default=0, help='Shift of the grid\'s zone boundaries (default 0)')
    from fiberhmm.inference.depth_cap import add_arguments
    add_arguments(p)
    p.add_argument('--cores', type=int, default=4, help='Worker budget (default 4)')
    p.add_argument('--window-jobs', type=int, default=0, metavar='N',
                   help='Units computed at the same time (default 0 = automatic: up to --cores); each gets an '
                        'equal share of --cores')
    p.add_argument('--continue', dest='continue_run', action='store_true',
                   help='Finish an interrupted scan in its output directory: finished units are kept, the rest are '
                        'run, and the tables, annotated BAMs and manifest are rebuilt. Refused if inputs, plan, read '
                        'policy or module settings differ')
    p.add_argument('--json-progress', action='store_true', help='Structured progress on stderr (JSON lines)')
    ann = p.add_mutually_exclusive_group()
    ann.add_argument('--annotate', dest='annotate', action='store_true', default=None,
                     help='Write per-molecule results as MA tags in annotated BAMs (default for --genome and long '
                          '--region scans)')
    ann.add_argument('--no-annotate', dest='annotate', action='store_false',
                     help='Write per-molecule results as tables (states, marks, molecules) instead (default for --bed '
                          'and short --region scans)')
    p.add_argument('-o', '--output', help='New or empty scan directory (with --continue: the interrupted scan\'s)')
    p.add_argument('--schema', action='store_true', help='Print every module\'s parameters with defaults as JSON')
    p.add_argument('--inspect', metavar='SCAN_DIR', help='Print a scan directory\'s summary and validate its tables, '
                                                         'indexes and annotated BAMs (exit 1 on problems)')
    from fiberhmm.cli.common import add_version_args
    add_version_args(p)
    return p


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    if args[:1] == ['assoc']:
        # `fiberhmm-scan assoc SCAN_DIR ...`: association statistics from a scan directory (fiberhmm.assoc)
        from fiberhmm.assoc.cli import main as assoc_main
        return assoc_main(args[1:])
    from fiberhmm.cli.common import run_reporting_input_errors
    return run_reporting_input_errors('fiberhmm-scan', lambda: _main(argv))


def _parse_region(parser, value, flag='--region'):
    chrom, sep, span = value.rpartition(':')
    a, dash, b = span.replace(',', '').partition('-')
    try:
        a, b = int(a), int(b)
    except ValueError:
        parser.error(f'{flag} {value!r}: expected CHROM:START-END (0-based, half-open)')
    if not (sep and chrom and dash) or a < 0 or b <= a:
        parser.error(f'{flag} {value!r}: expected CHROM:START-END with 0 <= START < END')
    return chrom, a, b


# Left out of the command line recorded in annotated BAM headers: execution-only controls and the output
# directory, so a rerun, a continued run or a differently parallelised run writes the same bytes.
EXECUTION_FLAGS = {'--continue': 0, '--json-progress': 0, '--cores': 1, '--window-jobs': 1, '-o': 1, '--output': 1}


def _recorded_command(argv, bam_rank=None):
    """The command line recorded in annotated BAM headers, without execution-only flags (a continued or
    differently parallelised run writes the same bytes). With ``bam_rank`` (``--bam`` value -> canonical
    ordinal), every ``--bam`` comes first, in canonical order, so listing the BAMs in another order (or in
    other places on the line) records the same command."""
    out, skip, bams = [], 0, []
    pending_bam = False
    for arg in argv:
        if skip:
            skip -= 1
            continue
        flag = arg.split('=', 1)[0]
        if flag in EXECUTION_FLAGS:
            skip = 0 if '=' in arg else EXECUTION_FLAGS[flag]
            continue
        if pending_bam:
            bams.append(arg)
            pending_bam = False
        elif arg == '--bam':
            pending_bam = True
        elif arg.startswith('--bam='):
            bams.append(arg[len('--bam='):])
        else:
            out.append(arg)
    if bam_rank:
        bams = sorted(bams, key=lambda v: (bam_rank.get(v, len(bam_rank)), v))
    return ['fiberhmm-scan'] + [x for v in bams for x in ('--bam', v)] + out


def _schema(modules):
    from fiberhmm.scan.modules import load_module
    from fiberhmm.scan.reads import DEFAULT_READ_POLICY
    out = {'read_policy': DEFAULT_READ_POLICY}
    for name in modules:
        mod = load_module(name)
        out[name] = dict(version=mod.version, experimental=mod.experimental, reconcile=mod.reconcile,
                         parameters=mod.parameter_schema(),
                         feature_types=[t.to_json() for t in mod.feature_types()],
                         ma_groups=[g.to_json() for g in mod.ma_groups()])
    return out


def _main(argv=None):
    p = _parser()
    args = p.parse_args(argv)
    modules = [m.strip() for m in args.modules.split(',') if m.strip()]
    if args.schema:
        print(json.dumps(_schema(modules), indent=2, sort_keys=True, default=str))
        return 0
    if args.inspect:
        from fiberhmm.scan.reader import ScanDir
        scan = ScanDir(args.inspect)
        problems = scan.validate()
        print(json.dumps(dict(summary=scan.summary(), problems=problems), indent=2, sort_keys=True))
        return 1 if problems else 0
    if not args.output or not (args.bam or args.datasets):
        p.error('give --bam or --datasets, a territory (--genome, --region or --bed) and -o/--output')
    if args.zones and (args.region or args.bed or args.contigs):
        p.error('--zones selects genome-grid units: use it alone or with --genome, not with --region, --bed or '
                '--contigs')
    if not (args.genome or args.region or args.bed or args.zones):
        p.error('choose the territory: --genome, --zones, --region or --bed')
    if args.contigs and not args.genome:
        p.error('--contigs applies to --genome')
    if args.window_jobs < 0 or args.cores < 1:
        p.error('--window-jobs must be >= 0 and --cores >= 1')
    out = Path(args.output).expanduser().resolve()
    if not args.continue_run and out.exists() and any(e.name != '.scan.lock' for e in out.iterdir()):
        p.error('Output directory must be empty; existing results are never overwritten (to finish an '
                'interrupted scan there, add --continue)')
    params = {}
    if args.parameters:
        params = json.loads(Path(args.parameters).read_text())
        if not isinstance(params, dict):
            p.error('--parameters must hold a JSON object of module groups')
    if args.datasets:
        datasets = json.loads(Path(args.datasets).read_text())
    else:
        # labelled by canonical order (dataset_<ordinal + 1>, ordinal = rank by content id), not by --bam
        # position, so a scan with the BAMs listed in another order has the same keys and rows
        datasets = [dict(paths=[str(Path(b).expanduser().resolve())], chemistry=args.chemistry) for b in args.bam]
    read_policy = dict(params.pop('read_policy', {}) or {})
    from fiberhmm.inference.depth_cap import NAMES
    caps = dict(read_policy.get('caps', {}))
    for name in NAMES:
        if getattr(args, name) is not None:
            caps[name] = getattr(args, name)
    if caps:
        read_policy['caps'] = caps
    if any(m in ('fp', 'nfr') for m in modules):
        from fiberhmm.inference.consensus.cli import check_chemistry
        check_chemistry(p, [dict(d, dataset_id=d.get('dataset_id') or Path(d['paths'][0]).name) for d in datasets])
    from fiberhmm.scan.runner import ScanRefused, resolve_datasets
    from fiberhmm.scan.units import PlanError, plan
    try:
        _resolved, reference = resolve_datasets(datasets)
        bam_rank = {}
        if args.bam:
            ordinal = {p: d.ordinal for d in _resolved for p in d.paths}
            bam_rank = {b: ordinal.get(str(Path(b).expanduser().resolve()), len(args.bam)) for b in args.bam}
        units = plan(reference, genome=args.genome or bool(args.zones),
                     contigs=[c.strip() for c in args.contigs.split(',')] if args.contigs else None,
                     regions=[_parse_region(p, r) for r in args.region] if args.region else None, bed=args.bed,
                     unit_bp=args.unit_bp, stride_bp=args.stride_bp, grid_offset=args.grid_offset,
                     zones=[_parse_region(p, z, '--zones') for z in args.zones] if args.zones else None)
    except (PlanError, ScanRefused) as error:
        p.exit(2, f'fiberhmm-scan: error: {error}\n')
    from fiberhmm.inference.consensus.cli import Progress
    from fiberhmm.scan.runner import run_units
    progress = Progress(args.json_progress)
    try:
        from fiberhmm.inference.consensus.execution import single_threaded_blas
        with single_threaded_blas():
            summary = run_units(datasets, units, modules=modules, params=params, read_policy=read_policy, out=out,
                                annotate=args.annotate, progress=progress, cores=args.cores,
                                window_jobs=args.window_jobs, continue_run=args.continue_run,
                                command=_recorded_command(argv if argv is not None else sys.argv[1:], bam_rank))
    except ScanRefused as error:
        p.exit(2, f'fiberhmm-scan: error: {error}\n')
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == '__main__':
    sys.exit(main())
