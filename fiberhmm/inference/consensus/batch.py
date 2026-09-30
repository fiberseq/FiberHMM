"""Restartable multi-window consensus runs (``fiberhmm-consensus --bam ... --bed/--region``).

Every BED window (or --region) is an independent work unit. A unit writes its outputs into its own directory
(``window_XXXXXX/``; the output directory itself when there is only one window) and, last, an atomic completion
marker ``unit_complete.json`` carrying a digest of the unit's inputs, parameters and code. ``--continue`` skips units
whose marker matches and whose recorded files still have their size and SHA-256, redoes missing, partial, altered or
mismatched units, and refuses when the run-level inputs (BAM and index content digests), parameters or the effective
DAF run mask (command line or FIBERHMM_DAF_RUN_MASK) differ from the run manifest ``consensus_run.json``. A run owns
its output directory exclusively (``flock`` on ``.consensus_run.lock``) while it runs; see
:mod:`fiberhmm.io.run_state` for the digest-reuse rule and the lock. Aggregate outputs (``regions.json``, the top-level
``report.html`` and the family-tagged BAMs) are always rebuilt from the completed units' saved artifacts, so they are
the same whether units ran serially, in parallel, or across an interrupted and continued run.

This is distinct from ``--resume DIR``, which starts a *new* run from a finished run's saved evidence.
"""
from __future__ import annotations
from contextlib import ExitStack, redirect_stderr, redirect_stdout
from copy import deepcopy
from pathlib import Path
import hashlib
import json
import shutil
import sys
import time
from fiberhmm.io.run_state import DigestMemo, DirectoryBusy, DirectoryLock, sha256_file
from .artifacts import digest, read_json, write_json

RUN_MANIFEST = 'consensus_run.json'
UNIT_MARKER = 'unit_complete.json'
LOG_DIR = 'logs'
RUN_LOCK = '.consensus_run.lock'
RUN_SCHEMA = 'fiberhmm.consensus.run.v1'
UNIT_SCHEMA = 'fiberhmm.consensus.unit.v1'
# Top-level outputs rebuilt from the units at the end of every run.
AGGREGATES = ('bams', 'regions.json', 'report.html')
# Run-level entries that never belong to a unit (kept when a single-window unit is redone in place).
PROTECTED = (RUN_MANIFEST, LOG_DIR, RUN_LOCK)
# Execution-only controls: they never change a result, so --continue may change them.
EXECUTION_ONLY = (('compute', 'cores'),)

_CODE_DIGEST = None


def code_digest():
    """FiberHMM version plus the SHA-256 of every consensus source file: a unit computed by other code is redone."""
    global _CODE_DIGEST
    if _CODE_DIGEST is None:
        from fiberhmm import __version__
        root = Path(__file__).resolve().parent
        h = hashlib.sha256(__version__.encode())
        for path in sorted(root.rglob('*.py')):
            h.update(str(path.relative_to(root)).encode()); h.update(path.read_bytes())
        _CODE_DIGEST = h.hexdigest()
    return _CODE_DIGEST


def file_identity(path, memo=None):
    """Path, size, mtime and content SHA-256 of an input file and of its index.

    Different content is a different input even when size and mtime were preserved. The mtime stays part of the
    identity because the aggregate BAM export checks the evidence's recorded source mtimes. ``memo``
    (:class:`~fiberhmm.io.run_state.DigestMemo`) only avoids rehashing files whose metadata shows them unchanged."""
    memo = memo if memo is not None else DigestMemo()
    p = Path(path).expanduser().resolve()
    def content(q):
        try:
            s = q.stat()
            return dict(size=s.st_size, mtime_ns=s.st_mtime_ns, sha256=memo.sha256(q))
        except OSError:
            return None
    index = next(({'path': str(q), **content(q)} for q in (Path(str(p)+'.csi'), Path(str(p)+'.bai'), p.with_suffix('.bai'))
                  if q.is_file() and content(q)), None)
    return dict(path=str(p), file=content(p), index=index)


def _scientific(values):
    values = deepcopy(values)
    for group, name in EXECUTION_ONLY:
        values.get(group, {}).pop(name, None)
    return values


def run_record(datasets, windows, values, options, memo=None):
    """The run-level contract --continue must match (execution-only controls excluded).

    Dataset paths are normalised exactly as the loader reads them (a single path may be given as a string)."""
    from .bam import _flatten_paths
    memo = memo if memo is not None else DigestMemo()
    return dict(datasets=[dict(dataset_id=d['dataset_id'], chemistry=d.get('chemistry'),
                               paths=[file_identity(p, memo) for p in _flatten_paths(d.get('paths') or [])])
                          for d in datasets],
                windows=[dict(w) for w in windows], parameters=_scientific(values), options=dict(options))


def _differences(previous, current, prefix=''):
    if isinstance(previous, dict) and isinstance(current, dict):
        out = []
        for key in sorted(set(previous) | set(current)):
            out += _differences(previous.get(key), current.get(key), f'{prefix}.{key}' if prefix else str(key))
        return out
    if isinstance(previous, list) and isinstance(current, list) and len(previous) == len(current):
        out = []
        for i, (a, b) in enumerate(zip(previous, current)):
            out += _differences(a, b, f'{prefix}.{i}' if prefix else str(i))
        return out
    return [] if previous == current else [prefix or '(root)']


class ContinueRefused(ValueError):
    """--continue with inputs or parameters that differ from the original run."""


def open_run(out, record, *, continue_run, execution, memo=None):
    """Write (fresh) or validate (--continue) the run manifest; returns the run digest."""
    from fiberhmm import __version__
    path = out/RUN_MANIFEST
    run_digest = digest(record)
    if not continue_run and any(entry.name != RUN_LOCK for entry in out.iterdir()):
        raise ContinueRefused(f'{out} is not empty; existing results are never overwritten (to finish an '
                              'interrupted multi-window run there, add --continue)')
    if continue_run:
        if not path.is_file():
            raise ContinueRefused(f'{out} has no {RUN_MANIFEST}; --continue needs the output directory of a '
                                  'multi-window BAM run (start a new run without --continue)')
        previous = read_json(path)
        if previous.get('schema') != RUN_SCHEMA:
            raise ContinueRefused(f'{path} is not a {RUN_SCHEMA} manifest')
        changed = _differences({k: previous.get(k) for k in record}, json.loads(json.dumps(record)))
        if changed:
            shown = ', '.join(changed[:8])+(f' (+{len(changed)-8} more)' if len(changed) > 8 else '')
            raise ContinueRefused('--continue refused: inputs or parameters differ from the original run in '
                                  f'{out} ({shown}). Rerun with the original BAMs, windows and parameters, or write '
                                  'a new run to a new --output directory. Only --cores, --window-jobs and '
                                  '--json-progress may change between attempts')
        attempts = previous.get('attempts', []) + [dict(fiberhmm_version=__version__, execution=execution)]
    else:
        attempts = [dict(fiberhmm_version=__version__, execution=execution)]
    write_json(path, dict(schema=RUN_SCHEMA, status='running', run_digest=run_digest, **record, attempts=attempts,
                          digests=(memo or DigestMemo()).to_json()))
    return run_digest


def previous_digests(out):
    """The digest memo a run manifest in ``out`` remembers (empty when there is none)."""
    try:
        return DigestMemo(read_json(out/RUN_MANIFEST).get('digests'))
    except (OSError, ValueError, AttributeError):
        return DigestMemo()


def unit_folder(out, index, count):
    return out if count == 1 else out/f'window_{index+1:06d}'


def _unit_files(folder, single):
    files = {}
    for path in sorted(folder.rglob('*')):
        rel = path.relative_to(folder)
        if not path.is_file() or rel.name == UNIT_MARKER: continue
        if single and (rel.parts[0] in PROTECTED or rel.parts[0] in AGGREGATES and rel.parts[0] != 'report.html'): continue
        files[rel.as_posix()] = dict(size=path.stat().st_size, sha256=sha256_file(path))
    return files


def completed_unit(folder, unit_digest):
    """The unit's marker if it is complete for exactly this digest and every recorded file still has its size and
    SHA-256, else None (a marker from before content digests were recorded is not trusted)."""
    marker = folder/UNIT_MARKER
    try:
        value = read_json(marker)
    except (OSError, ValueError):
        return None
    if value.get('schema') != UNIT_SCHEMA or value.get('digest') != unit_digest:
        return None
    for rel, recorded in value.get('files', {}).items():
        path = folder/rel
        if (not isinstance(recorded, dict) or not path.is_file() or path.stat().st_size != recorded.get('size')
                or sha256_file(path) != recorded.get('sha256')):
            return None
    return value


def _clear_unit(folder, single):
    """Remove a partial or stale unit (in place for a single window, keeping the run manifest and logs)."""
    if not folder.exists(): return
    if not single:
        shutil.rmtree(folder); return
    for entry in folder.iterdir():
        if entry.name in PROTECTED: continue
        shutil.rmtree(entry) if entry.is_dir() and not entry.is_symlink() else entry.unlink()


class UnitLog:
    """Progress sink for one unit: appends to its log and optionally forwards to the parent's reporter."""

    def __init__(self, handle, forward=None):
        self.handle = handle; self.forward = forward

    def __call__(self, stage, message): self.report(stage, message)

    def report(self, stage, message, **work):
        if message is None: return
        done, total = work.get('completed'), work.get('total')
        count = f' [{done}/{total}]' if done is not None and total else ''
        print(f'{time.strftime("%H:%M:%S")} {stage}{count}: {message}', file=self.handle, flush=True)
        if self.forward is not None:
            self.forward.report(stage, message, **work)


def run_unit(spec, *, load, run, forward=None, isolate=False):
    """Load, analyse and mark one window. ``isolate`` (a worker process) sends all output to the unit's log."""
    from .parameters import parse_options
    from .execution import single_threaded_blas
    folder = Path(spec['folder']); single = spec['single']; log = Path(spec['log'])
    _clear_unit(folder, single); folder.mkdir(parents=True, exist_ok=True); log.parent.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        handle = stack.enter_context(log.open('a'))
        print(f'=== {time.strftime("%Y-%m-%d %H:%M:%S")} window {spec["name"]} ({spec["index"]+1}/{spec["count"]}) '
              f'-> {folder}', file=handle, flush=True)
        if isolate:
            stack.enter_context(redirect_stdout(handle)); stack.enter_context(redirect_stderr(handle))
            if spec.get('daf_mask') is not None:
                from fiberhmm.core.bam_reader import configure_daf_run_mask
                configure_daf_run_mask(*spec['daf_mask'])
            stack.enter_context(single_threaded_blas())
        progress = UnitLog(handle, forward)
        values = spec['values']; w = spec['window']
        payload = load(spec['datasets'], {k: w[k] for k in ('chrom', 'start', 'end')}, parse_options(values), progress)
        result = run(payload, values, folder, progress=progress)
        manifest = result['manifest']
        marker = dict(schema=UNIT_SCHEMA, digest=spec['digest'], index=spec['index'], name=spec['name'], window=w,
                      status=manifest.get('status'), seconds=manifest.get('seconds'),
                      has_input_files=bool(payload.get('input_files')), files=_unit_files(folder, single))
        del result, payload
        write_json(folder/UNIT_MARKER, marker)       # atomic, and last: the unit is complete only now
        print(f'=== complete in {marker["seconds"] or 0:.1f}s', file=handle, flush=True)
    return marker


def _eta(seconds):
    if seconds is None: return ''
    seconds = int(round(seconds)); h, rest = divmod(seconds, 3600); m, s = divmod(rest, 60)
    return f' ETA {h}:{m:02d}:{s:02d}' if h else f' ETA {m}:{s:02d}'


def dispatch(specs, completed, *, jobs, serial_task, parallel_task, progress):
    """Run the pending units (in-process when jobs == 1) with concise progress: units done/total and an ETA
    from this attempt's throughput. The first failure stops the run and kills the other units' workers."""
    total = len(specs); reused = len(completed); pending = [s for s in specs if s['index'] not in completed]
    started = time.monotonic(); fresh = 0
    progress.report('windows', f'{reused} complete from an earlier attempt; {len(pending)} to run '
                    f'({jobs} at a time)' if reused else f'{len(pending)} to run ({jobs} at a time)',
                    completed=reused, total=total, reused=reused)

    def finished(marker):
        nonlocal fresh
        completed[marker['index']] = marker; fresh += 1
        done = len(completed); elapsed = time.monotonic()-started
        eta = elapsed/fresh*(total-done) if done < total else 0.
        progress.report('windows', f"{marker['name']} complete{_eta(eta) if done < total else ''}", completed=done, total=total, reused=reused,
                        eta_seconds=round(eta, 1))

    if jobs <= 1:
        for spec in pending:
            progress.report('windows', f"{spec['name']} running (log {spec['log']})", completed=len(completed), total=total,
                            reused=reused)
            finished(serial_task(spec))
        return completed
    from concurrent.futures import as_completed
    from .execution import shared_worker_pool
    with shared_worker_pool(jobs) as pool:
        executor = pool.get()
        futures = {executor.submit(parallel_task, spec): spec for spec in pending}
        for future in as_completed(futures):
            spec = futures[future]
            try:
                marker = future.result()
            except BaseException:
                print(f"windows: {spec['name']} failed; see {spec['log']}. Completed windows are kept: fix the cause "
                      'and rerun the same command with --continue', file=sys.stderr, flush=True)
                raise
            finished(marker)
    return completed


def plan_jobs(pending, cores, window_jobs):
    """Concurrent windows and each window's worker budget, within ``cores`` processes in total."""
    if pending <= 1: return 1, cores
    jobs = min(pending, cores, window_jobs) if window_jobs else min(pending, cores)
    jobs = max(1, jobs)
    return jobs, max(1, cores//jobs)


def rebuild_aggregates(out, specs, completed, *, no_bam, recaller_layer, grouping, scope, progress):
    """regions.json, report.html and BAM export from the units' saved artifacts, in BED order."""
    count = len(specs)
    summary = [dict(name=s['name'], output=s['folder'], status=completed[s['index']]['status'],
                    seconds=completed[s['index']]['seconds']) for s in specs]
    bam_outputs = []
    if count > 1:
        for name in AGGREGATES:
            path = out/name
            if path.is_dir(): shutil.rmtree(path)
            elif path.exists(): path.unlink()
    elif (out/'bams').exists():
        shutil.rmtree(out/'bams')
    if not no_bam:
        from .bam_export import ExportPlan, export_bams
        plan = ExportPlan(recaller_layer=recaller_layer)
        for s in specs:
            if not completed[s['index']].get('has_input_files'): continue
            folder = Path(s['folder'])
            plan.add(read_json(folder/'result.json.gz'), read_json(folder/'evidence.json.gz'))
        if plan.analyses:
            bam_outputs = export_bams(plan, out/'bams', grouping=grouping, scope=scope, progress=progress)
    if count > 1:
        write_json(out/'regions.json', summary)
        import html
        (out/'report.html').write_text('<!doctype html><meta charset="utf-8"><h1>Consensus windows</h1>'+''.join(
            '<p><a href="'+str(Path(r['output']).relative_to(out))+'/report.html">'+html.escape(r['name'])+'</a></p>' for r in summary))
    return summary, bam_outputs


def finish_run(out):
    path = out/RUN_MANIFEST
    value = read_json(path); value['status'] = 'complete'
    write_json(path, value)


def run_windows(out, windows, datasets, values, *, continue_run, cores, window_jobs, daf_mask, options,
                serial_task, parallel_task, progress, export):
    """The whole multi-window run: manifest, unit scheduling, and aggregate rebuild. Returns (summary, bam_outputs)."""
    lock = DirectoryLock(out/RUN_LOCK, 'fiberhmm-consensus output directory', remove=True)
    try:
        lock.acquire()
    except DirectoryBusy as error:
        raise ContinueRefused(str(error)) from None
    try:
        return _run_windows_locked(out, windows, datasets, values, continue_run=continue_run, cores=cores,
                                   window_jobs=window_jobs, daf_mask=daf_mask, options=options,
                                   serial_task=serial_task, parallel_task=parallel_task, progress=progress,
                                   export=export)
    finally:
        lock.release()


def _run_windows_locked(out, windows, datasets, values, *, continue_run, cores, window_jobs, daf_mask, options,
                        serial_task, parallel_task, progress, export):
    count = len(windows)
    memo = previous_digests(out) if continue_run else DigestMemo()
    record = run_record(datasets, windows, values, dict(options, daf_mask=list(daf_mask) if daf_mask else None), memo)
    run_digest = open_run(out, record, continue_run=continue_run,
                          execution=dict(cores=cores, window_jobs=window_jobs), memo=memo)
    code = code_digest(); specs = []
    for i, w in enumerate(windows):
        folder = unit_folder(out, i, count)
        specs.append(dict(index=i, count=count, name=w['name'], window=w, folder=str(folder), single=count == 1,
                          log=str(out/LOG_DIR/f'window_{i+1:06d}.log'), datasets=datasets, values=values,
                          daf_mask=daf_mask, digest=digest([run_digest, code, i, w])))
    completed = {}
    for s in specs:
        marker = completed_unit(Path(s['folder']), s['digest'])
        if marker is not None: completed[s['index']] = marker
    pending = count-len(completed)
    jobs, inner = plan_jobs(pending, cores, window_jobs)
    if jobs > 1:
        # Each concurrent window gets its own share of the worker budget (execution only: results do not change).
        for s in specs:
            s['values'] = deepcopy(values); s['values'].setdefault('compute', {})['cores'] = inner
    dispatch(specs, completed, jobs=jobs, serial_task=serial_task, parallel_task=parallel_task, progress=progress)
    summary, bam_outputs = rebuild_aggregates(out, specs, completed, progress=progress, **export)
    finish_run(out)
    return summary, bam_outputs
