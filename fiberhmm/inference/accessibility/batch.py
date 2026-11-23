"""Many-window fiberhmm-nfr runs (``--bed`` / ``--regions-bed``), restartable (EXPERIMENTAL preview).

Every BED window is an independent unit, analysed exactly as ``--region`` would analyse it and written to
``window_XXXXXX/`` (BED order). Windows run in worker processes, up to ``--window-jobs`` (default: ``--cores``) at a
time, each on one core; a worker keeps its imports and compiled kernels across windows. Each window opens the BAMs
through their index and decodes only the reads overlapping it: windows do not share decoded reads, so a read that
spans two overlapping windows is decoded in both (opening an indexed BAM costs milliseconds; decoding and native
replay are the per-window cost).

A finished window writes ``window_complete.json`` last (atomic), with a digest of the run's inputs, parameters and
window and the size and SHA-256 of each output. ``--continue`` skips windows whose marker matches and whose files are
intact and reruns the rest; it refuses when the BAMs (path, size, mtime), windows or parameters differ from the run
record ``nfr_run.json``. A window that fails (or has no molecules) is reported with its status and error; the others
still run. ``regions.json``/``regions.tsv`` (one row per window) and the combined ``variants.tsv`` and
``configurations.tsv`` (window name and region prepended) are rebuilt from the completed windows at the end of every
attempt.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import shutil
import sys
import time
from copy import deepcopy
from pathlib import Path

RUN_RECORD = 'nfr_run.json'
MARKER = 'window_complete.json'
LOCK = '.nfr_run.lock'
RUN_SCHEMA = 'fiberhmm.nfr.run.v1'
MARKER_SCHEMA = 'fiberhmm.nfr.window.v1'
SUMMARY_FIELDS = ['index', 'name', 'chrom', 'start', 'end', 'status', 'nfrs', 'variants', 'molecules', 'seconds', 'folder',
                  'error']


def _sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for block in iter(lambda: fh.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), default=str).encode()).hexdigest()


def _write_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, sort_keys=True, indent=1) + '\n')
    tmp.replace(path)


_CODE = None


def code_digest():
    """FiberHMM version plus the SHA-256 of the accessibility and consensus (loader) sources: a window computed by
    other code is redone by --continue, even when the version string did not change."""
    global _CODE
    if _CODE is None:
        from fiberhmm import __version__
        h = hashlib.sha256(__version__.encode())
        pkg = Path(__file__).resolve().parents[1]
        for sub in ('accessibility', 'consensus'):
            for path in sorted((pkg/sub).rglob('*.py')):
                h.update(str(path.relative_to(pkg)).encode()); h.update(path.read_bytes())
        _CODE = h.hexdigest()
    return _CODE


def effective_daf_run_mask():
    """The explicit DAF run mask in force for this run (inherited FIBERHMM_DAF_RUN_MASK / FIBERHMM_DAF_RUN_POLICY, or
    one configured in this process) as [min_run_length, policy], else None (each dataset's chemistry default). Window
    workers inherit it from the environment."""
    from fiberhmm.core.bam_reader import daf_run_mask_explicit, daf_run_mask_min_length, daf_run_mask_policy
    return [daf_run_mask_min_length(), daf_run_mask_policy()] if daf_run_mask_explicit() else None


def run_record(datasets, windows, values, nfr):
    """What --continue must match. BAMs are identified by path, size and mtime (no content hash: these BAMs are often
    tens of GB); the DAF SNP mask (input.daf_snp_mask) by its content SHA-256 as well, and an explicit DAF run mask by
    its effective settings (both recorded only when given, so runs without them keep their record and digests);
    --cores, --window-jobs and --json-progress are execution-only."""
    from fiberhmm import __version__
    from fiberhmm.inference.consensus.batch import file_identity
    from fiberhmm.inference.consensus.bam import _flatten_paths
    values = deepcopy(values)
    values.get('compute', {}).pop('cores', None)
    files = []
    for d in datasets:
        for p in _flatten_paths(d.get('paths') or []):       # a single path may be given as a string, as the loader reads it
            st = Path(p).stat()
            files.append(dict(dataset_id=d['dataset_id'], path=str(Path(p).resolve()), size=st.st_size, mtime_ns=st.st_mtime_ns))
    record = dict(datasets=[dict(dataset_id=d['dataset_id'], chemistry=d.get('chemistry'),
                                 paths=_flatten_paths(d.get('paths') or [])) for d in datasets],
                  files=files, windows=[{k: w[k] for k in ('chrom', 'start', 'end', 'name')} for w in windows],
                  loader_parameters=values, nfr=nfr, fiberhmm=__version__)
    mask = (values.get('input') or {}).get('daf_snp_mask')
    if mask:
        record['daf_snp_mask'] = file_identity(mask)
    run_mask = effective_daf_run_mask()
    if run_mask is not None:
        record['daf_run_mask'] = run_mask
    return record


class ContinueRefused(ValueError):
    pass


def open_run(out, record, continue_run):
    path = out/RUN_RECORD
    if not continue_run:
        if any(e.name != LOCK for e in out.iterdir()):
            raise ContinueRefused(f'{out} is not empty; existing results are never overwritten (to finish an interrupted '
                                  'BED run there, add --continue)')
    else:
        if not path.is_file():
            raise ContinueRefused(f'{out} has no {RUN_RECORD}; --continue needs the output directory of a fiberhmm-nfr '
                                  'BED run')
        previous = json.loads(path.read_text())
        keys = set(record) | (set(previous) - {'schema'})      # an entry present in only one attempt differs too
        changed = sorted(k for k in keys if json.loads(json.dumps(record.get(k), default=str)) != previous.get(k))
        if previous.get('schema') != RUN_SCHEMA or changed:
            raise ContinueRefused('--continue refused: ' + (', '.join(changed) or 'schema') + ' differ from the original run '
                                  f'in {out}. Use the original BAMs, BED and parameters, or a new --output directory '
                                  '(only --cores, --window-jobs and --json-progress may change)')
    _write_json(path, dict(record, schema=RUN_SCHEMA))
    return _digest(record)


def completed(folder, digest):
    """The window's marker when it is complete for this digest and its files are intact, else None."""
    try:
        marker = json.loads((Path(folder)/MARKER).read_text())
    except (OSError, ValueError):
        return None
    if marker.get('schema') != MARKER_SCHEMA or marker.get('digest') != digest:
        return None
    for name, rec in (marker.get('files') or {}).items():
        p = Path(folder)/name
        if not p.is_file() or p.stat().st_size != rec.get('size') or _sha256(p) != rec.get('sha256'):
            return None
    return marker


def window_task(spec):
    """Load, analyse and mark one window (worker process or in-process). Never raises for a window's own failure:
    the marker-less result carries status 'failed' and the error, so the other windows still run."""
    import dataclasses
    import traceback
    from fiberhmm.inference.consensus.parameters import parse_options
    from fiberhmm.inference.consensus.execution import single_threaded_blas
    from .cli import load_payload, loader_sampling
    from .workflow import NFROptions, run_accessibility
    folder = Path(spec['folder'])
    started = time.monotonic()
    base = dict(index=spec['index'], name=spec['name'], folder=str(folder), **spec['region'])
    try:
        if folder.exists():
            shutil.rmtree(folder)          # a partial or stale attempt
        options = parse_options(spec['values'])
        opt = NFROptions.from_params(spec['nfr'])
        sampling = loader_sampling(opt)       # BED windows always detect NFRs (no nfr_regions): windows=None
        with single_threaded_blas():
            payload = load_payload(spec['datasets'], spec['region'], spec['values'], options, sampling=sampling)
            molecules = sum(len(s['units']) for s in payload['strata'])
            if not molecules:
                folder.mkdir(parents=True, exist_ok=True)
                row = dict(base, status='no_molecules', nfrs=0, variants=0, molecules=0, pairs=0, warnings=[],
                           seconds=round(time.monotonic() - started, 1),
                           error='no molecules in the window (check chromosome, coordinates, MAPQ and chemistry)')
            else:
                inputs = dict(datasets=spec['datasets'], evidence=None,
                              loader_parameters={'input': dataclasses.asdict(options['input'])}, classes=[])
                result = run_accessibility(payload, opt, folder, inputs=inputs)
                del payload
                row = dict(base, status='complete', nfrs=len(result['nfrs']),
                           variants=sum(len(n['variants']) for n in result['nfrs']), molecules=molecules,
                           pairs=len(result['pairs']), warnings=result['warnings'],
                           seconds=round(time.monotonic() - started, 1), error='')
        files = {p.name: dict(size=p.stat().st_size, sha256=_sha256(p)) for p in sorted(folder.iterdir())
                 if p.is_file() and p.name != MARKER}
        _write_json(folder/MARKER, dict(row, schema=MARKER_SCHEMA, digest=spec['digest'], files=files))
        return row
    except Exception as error:   # noqa: BLE001 - reported per window; --continue retries it
        return dict(base, status='failed', nfrs=None, variants=None, molecules=None, pairs=None, warnings=[],
                    seconds=round(time.monotonic() - started, 1), error=f'{type(error).__name__}: {error}',
                    traceback=traceback.format_exc(limit=8))


def _tsv(rows, fields):
    buf = io.StringIO()
    w = csv.writer(buf, delimiter='\t', lineterminator='\n')
    w.writerow(fields)
    for r in rows:
        w.writerow(['' if r.get(f) is None else r.get(f) for f in fields])
    return buf.getvalue()


def _combine(out, rows, name):
    """window name and region columns prepended to each completed window's ``name`` table."""
    header, body = None, []
    for r in rows:
        p = Path(r['folder'])/name
        if r['status'] != 'complete' or not p.is_file():
            continue
        lines = p.read_text().splitlines()
        if not lines:
            continue
        if header is None:
            header = 'window\twindow_chrom\twindow_start\twindow_end\t' + lines[0]
        prefix = f"{r['name']}\t{r['chrom']}\t{r['start']}\t{r['end']}\t"
        body.extend(prefix + line for line in lines[1:])
    if header is not None:
        (out/name).write_text('\n'.join([header] + body) + '\n')


def run_windows(out, windows, datasets, values, nfr, *, cores, window_jobs, continue_run, progress):
    """The whole BED run. Returns the summary rows (BED order)."""
    from fiberhmm.io.run_state import DirectoryBusy, DirectoryLock
    out.mkdir(parents=True, exist_ok=True)
    lock = DirectoryLock(out/LOCK, 'fiberhmm-nfr output directory', remove=True)
    try:
        lock.acquire()
    except DirectoryBusy as error:
        raise ContinueRefused(str(error)) from None
    try:
        return _run_locked(out, windows, datasets, values, nfr, cores=cores, window_jobs=window_jobs,
                           continue_run=continue_run, progress=progress)
    finally:
        lock.release()


def _run_locked(out, windows, datasets, values, nfr, *, cores, window_jobs, continue_run, progress):
    record = run_record(datasets, windows, values, nfr)
    run_digest = open_run(out, record, continue_run)
    inner = deepcopy(values); inner.setdefault('compute', {})['cores'] = 1
    specs = [dict(index=i, name=w['name'], region={k: w[k] for k in ('chrom', 'start', 'end')}, datasets=datasets,
                  values=inner, nfr=nfr, folder=str(out/f'window_{i + 1:06d}'),
                  digest=_digest([run_digest, code_digest(), i, {k: w[k] for k in ('chrom', 'start', 'end', 'name')}]))
             for i, w in enumerate(windows)]
    rows = {}
    for s in specs:
        marker = completed(s['folder'], s['digest'])
        if marker is not None:
            rows[s['index']] = {k: marker.get(k) for k in (*SUMMARY_FIELDS, 'pairs', 'warnings')}
    pending = [s for s in specs if s['index'] not in rows]
    total, reused = len(specs), len(rows)
    jobs = max(1, min(len(pending) or 1, cores, window_jobs or cores))
    progress.report('windows', (f'{reused} complete from an earlier attempt; ' if reused else '') +
                    f'{len(pending)} to run ({jobs} at a time)', completed=reused, total=total)
    started, fresh = time.monotonic(), 0

    def finished(row):
        nonlocal fresh
        rows[row['index']] = row; fresh += 1
        done = len(rows); elapsed = time.monotonic() - started
        eta = elapsed/fresh*(total - done)
        h, rest = divmod(int(round(eta)), 3600); m, s_ = divmod(rest, 60)
        tail = f' ETA {h}:{m:02d}:{s_:02d}' if done < total else ''
        progress.report('windows', f"{row['name']} {row['status']} ({row['seconds']} s){tail}", completed=done, total=total,
                        status=row['status'], eta_seconds=round(eta, 1))
        if row['status'] == 'failed':
            print(f"windows: {row['name']} failed: {row['error']}", file=sys.stderr, flush=True)

    if jobs <= 1:
        for s in pending:
            finished(window_task(s))
    else:
        from concurrent.futures import as_completed
        from .workflow import worker_pool
        with worker_pool(jobs) as executor:
            futures = {executor.submit(window_task, s): s for s in pending}
            for future in as_completed(futures):
                try:
                    row = future.result()
                except BaseException as error:   # noqa: BLE001 - a worker died (e.g. out of memory): record, go on
                    if isinstance(error, KeyboardInterrupt):
                        raise
                    s = futures[future]
                    row = dict(index=s['index'], name=s['name'], folder=s['folder'], **s['region'], status='failed',
                               nfrs=None, variants=None, molecules=None, pairs=None, warnings=[], seconds=None,
                               error=f'worker process failed ({type(error).__name__}: {error}); out of memory? '
                                     'lower --window-jobs and rerun with --continue')
                finished(row)
    summary = [rows[s['index']] for s in specs]
    _write_json(out/'regions.json', summary)
    (out/'regions.tsv').write_text(_tsv(summary, SUMMARY_FIELDS))
    for name in ('variants.tsv', 'configurations.tsv'):
        _combine(out, summary, name)
    return summary
