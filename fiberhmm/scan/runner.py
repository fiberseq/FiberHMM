"""``run_units``: the one entry point that computes a scan directory.

The CLI and FiberBrowser's live scan both call :func:`run_units`. It resolves
datasets (content ids, canonical ordinals), resolves module settings, plans
nothing itself (units come from :func:`fiberhmm.scan.units.plan`), runs every
unit (resumable: a unit's per-module result is reused when its key matches),
reconciles seams, and writes ``manifest.json``, the tables and, in annotated
mode, one annotated BAM per input file. Contract:
``docs/reference/scan-format.md``.
"""
from __future__ import annotations

import copy
import gzip
import json
import os
import shutil
import sys
import time
from contextlib import ExitStack, redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from . import identity as I
from .modules import ORDER, load_module
from .reads import overlapping_records, record_ref, resolve_read_policy, unit_survey
from .seams import reconcile_anchor, territory_runs
from .tables import COMMON, TableWriter
from .types import (SCHEMA, Dataset, FullMolecule, ScanInputs, ScanUnit, molecule_result_from_json,
                    molecule_result_to_json, unit_result_from_json, unit_result_to_json)
from .units import default_annotate

LOCK = '.scan.lock'
RUN_JSON = 'run.json'
WORK = 'work'
UNIT_MARKER_SCHEMA = 'fiberhmm.scan.unit.v1'


class ScanRefused(ValueError):
    """A run that cannot proceed as asked (inputs differ from a continued scan, a non-empty output, ...)."""


# ------------------------------------------------------------------ small IO helpers
def _write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name('.' + path.name + '.tmp')
    with open(tmp, 'wb') as handle:
        handle.write(data)
    os.replace(tmp, path)


def _write_json(path: Path, value: Any, *, indent: Optional[int] = None) -> None:
    text = json.dumps(value, sort_keys=True, indent=indent, separators=(',', ': ') if indent else (',', ':'),
                      default=I._json_default)
    data = (text + '\n').encode()
    if path.suffix == '.gz':
        import io
        buf = io.BytesIO()
        with gzip.GzipFile(fileobj=buf, mode='wb', mtime=0, filename='', compresslevel=1) as handle:
            handle.write(data)
        data = buf.getvalue()
    _write_bytes(path, data)


def _read_json(path: Path) -> Any:
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt') as handle:
        return json.load(handle)


def _sha(path: Path) -> str:
    from .tables import sha256_file
    return sha256_file(path)


class _NullProgress:
    def report(self, stage, message, **work):
        pass


class _LogProgress:
    """A unit's progress sink: its log file, plus an optional forward."""

    def __init__(self, handle, forward=None):
        self.handle, self.forward = handle, forward

    def __call__(self, stage, message):
        self.report(stage, message)

    def report(self, stage, message, **work):
        if message is None:
            return
        print(f'{stage}: {message}', file=self.handle, flush=True)
        if self.forward is not None:
            try:
                self.forward.report(stage, message, **work)
            except TypeError:
                self.forward.report(stage, message)


# ------------------------------------------------------------------ datasets and reference
def _bam_reference(path) -> List[Tuple[str, int]]:
    import pysam
    with pysam.AlignmentFile(str(path), 'rb', check_sq=False) as bam:
        return [(name, int(length)) for name, length in zip(bam.references, bam.lengths)]


def _calls_program(path) -> dict:
    """The last footprint-writing @PG of a BAM (FiberHMM apply/call/recall, fibertools nucleosome commands)."""
    import pysam
    from .peaks_reads import footprint_writer_id
    with pysam.AlignmentFile(str(path), 'rb', check_sq=False) as bam:
        ident = footprint_writer_id(bam.header)
        for pg in bam.header.to_dict().get('PG', []):
            if ident and str(pg.get('ID', '')) == ident:
                return dict(program=str(pg.get('PN') or pg.get('ID') or ''), version=str(pg.get('VN', '')), id=ident)
    return {}


def _declares_molecular(path) -> bool:
    import pysam
    from fiberhmm.io.annotation_frame import ma_annotation_frame, resolve_disabled_legacy_frame
    with pysam.AlignmentFile(str(path), 'rb', check_sq=False) as bam:
        return (ma_annotation_frame(bam.header) == 'molecular'
                or resolve_disabled_legacy_frame(bam.header) is not None)


def _carry_record(read) -> bool:
    """Put a record of an input without a molecular-coordinate declaration into molecular frame (in place).
    False when its calls cannot be carried. A reverse hard-clipped record whose legacy calls cover the whole
    molecule (fiberhmm-lift's scope rule) is mirrored against the molecule's length, not the stored SEQ's."""
    from fiberhmm.io.frame_carry import carry_to_molecular
    original = {t: read.get_tag(t) for t in ('ns', 'nl', 'nq', 'as', 'al', 'aq') if read.has_tag(t)}
    try:
        carry_to_molecular(read)
        return True
    except (ValueError, OverflowError):
        pass
    if not read.is_reverse or not original or read.has_tag('MA') or read.has_tag('Ma'):
        return False
    from fiberhmm.io.annotation_frame import MOLECULAR, legacy_frame_of
    from fiberhmm.io.graph_lift import annotation_scope
    from fiberhmm.io.ma_tags import flip_interval_frame
    import array
    for t, v in original.items():
        read.set_tag(t, v)
    ops = read.cigartuples or []
    stored = int(read.query_length or len(read.query_sequence or ''))
    hard = sum(n for op, n in ops[:1] + ops[-1:] if op == 5) if len(ops) > 1 else sum(n for op, n in ops if op == 5)
    total = stored + hard
    if annotation_scope(read, stored, total) != 'whole':
        return False
    if legacy_frame_of(read.header).for_read(read) == MOLECULAR:
        return True
    for start_tag, size_tag, score_tag in (('ns', 'nl', 'nq'), ('as', 'al', 'aq')):
        if not (read.has_tag(start_tag) and read.has_tag(size_tag)):
            continue
        scores = list(read.get_tag(score_tag)) if read.has_tag(score_tag) else None
        rows = []
        for k, (a, n) in enumerate(zip(read.get_tag(start_tag), read.get_tag(size_tag))):
            a, n = flip_interval_frame(int(a), int(n), total)
            rows.append((a, n, scores[k] if scores and k < len(scores) else None))
        rows.sort(key=lambda r: r[0])
        read.set_tag(start_tag, array.array('I', [r[0] for r in rows]))
        read.set_tag(size_tag, array.array('I', [r[1] for r in rows]))
        if scores is not None and len(scores) == len(rows):
            read.set_tag(score_tag, array.array('B', [int(r[2]) for r in rows]))
    return True


def resolve_datasets(datasets: Sequence[Mapping[str, Any]]):
    """Datasets in canonical order and the shared reference (contig -> length, header order).

    Every input must carry the same reference (same contig names and lengths). A dataset given without a
    ``dataset_id`` is labelled by its canonical ordinal, ``dataset_<ordinal + 1>``, so the same files give the
    same labels (and the same unit keys) whatever order they are listed in."""
    if not datasets:
        raise ScanRefused('no datasets')
    items = []
    reference = None
    for i, d in enumerate(datasets):
        from .signal import validate_spec
        validate_spec(d)
        paths = d.get('paths') or ([d['path']] if d.get('path') else [])
        if isinstance(paths, str):
            paths = [paths]
        if not paths:
            raise ScanRefused(f'dataset {d.get("dataset_id") or i + 1}: no BAM paths')
        paths = [str(Path(p).expanduser().resolve()) for p in paths]
        for p in paths:
            if not Path(p).is_file():
                raise ScanRefused(f'BAM not found: {p}')
            ref = _bam_reference(p)
            if reference is None:
                reference = ref
            elif ref != reference:
                raise ScanRefused(f'{p}: its reference (@SQ names and lengths) differs from the first input; '
                                  'a scan needs one reference')
        items.append(dict(dataset_id=str(d['dataset_id']) if d.get('dataset_id') else None, paths=paths,
                          chemistry=d.get('chemistry'), content_ids=list(d.get('content_ids') or [I.content_id(p) for p in paths])))
    if any(len(x['content_ids']) != len(x['paths']) for x in items):
        raise ScanRefused('content_ids must have one entry per input file')
    ordinals = I.canonical_order([x['content_ids'] for x in items])
    for x, o in zip(items, ordinals):
        if x['dataset_id'] is None:
            x['dataset_id'] = f'dataset_{o + 1}'
    ids = [x['dataset_id'] for x in items]
    if len(set(ids)) != len(ids):
        raise ScanRefused('dataset ids must be unique (an unlabelled dataset is dataset_<canonical ordinal + 1>)')
    out = [Dataset(dataset_id=x['dataset_id'], ordinal=o, chemistry=x['chemistry'], paths=tuple(x['paths']),
                   content_ids=tuple(x['content_ids'])) for x, o in zip(items, ordinals)]
    out.sort(key=lambda d: d.ordinal)
    return tuple(out), dict(reference)


# ------------------------------------------------------------------ modules
def resolve_modules(requested: Sequence[str], params: Mapping[str, Any], caps: Mapping[str, Any],
                    registry=None) -> List[Tuple[str, Any, dict]]:
    """(name, module, settings) for the requested modules and their dependencies, dependencies first."""
    instances: Dict[str, Any] = {}
    settings: Dict[str, dict] = {}
    pending = list(dict.fromkeys(requested))
    while pending:
        name = pending.pop(0)
        if name in instances:
            continue
        mod = load_module(name, registry)
        p = dict((params or {}).get(name) or {})
        p['caps'] = dict(caps)
        instances[name] = mod
        settings[name] = mod.resolve(p)
        for dep in sorted(mod.needs(settings[name])):
            if dep not in instances:
                pending.append(dep)
    unknown = set((params or {})) - set(instances) - {'read_policy'}
    if unknown:
        raise ScanRefused(f'--parameters names module(s) not in this scan: {", ".join(sorted(unknown))}')
    rank = {n: (ORDER.index(n) if n in ORDER else len(ORDER)) for n in instances}
    ordered: List[str] = []

    def visit(n, stack=()):
        if n in ordered:
            return
        if n in stack:
            raise ScanRefused(f'module dependency cycle at {n}')
        for dep in sorted(instances[n].needs(settings[n]), key=lambda d: (rank[d], d)):
            visit(dep, stack + (n,))
        ordered.append(n)

    for n in sorted(instances, key=lambda n: (rank[n], n)):
        visit(n)
    return [(n, instances[n], settings[n]) for n in ordered]


# ------------------------------------------------------------------ the manifest key
def _units_block(units: Sequence[ScanUnit], info: Mapping[str, Any]) -> dict:
    block = dict(info)
    if block.get('bed'):
        block['bed_sha256'] = I.digest(open(block['bed'], 'rb').read().decode('utf-8', 'replace'))
        block['bed'] = Path(block['bed']).name
    else:
        block.setdefault('bed_sha256', None)
    block['count'] = len(units)
    return block


def _module_block(name, mod, settings, calibration_sha) -> dict:
    decls = mod.feature_types()
    return dict(version=mod.version, experimental=bool(mod.experimental), reconcile=mod.reconcile,
                max_support_bp=int(mod.max_support_bp), settings=settings,
                settings_hash=I.settings_hash(settings), load_policy=mod.load_policy(settings),
                feature_types=[d.to_json() for d in decls],
                columns=dict(hits=mod.hit_columns(), states=mod.state_columns(), molecules=mod.scalar_names()),
                ma_groups=[g.to_json() for g in mod.ma_groups()], native_tables=list(mod.native_tables),
                calibration=None if calibration_sha is None else dict(path=f'{name}/extra/calibration.json',
                                                                      digest=calibration_sha),
                needs=sorted(mod.needs(settings)))


def scan_id_of(manifest: Mapping[str, Any]) -> str:
    units = {k: v for k, v in manifest['units'].items() if k != 'count'}
    key = [manifest['schema'], manifest['reference']['contigs_sha256'],
           [[d['ordinal'], d['dataset_id'], d['chemistry'], [f['content_id'] for f in d['files']]]
            for d in manifest['datasets']],
           manifest['read_policy'], units,
           {n: [m['version'], m['settings_hash'], (m.get('calibration') or {}).get('digest')]
            for n, m in sorted(manifest['modules'].items())}]
    return I.digest(key, 16)


def base_key(contigs_sha: str, resolved, policy: Mapping[str, Any]) -> str:
    """What every unit result depends on besides its module and its unit: the reference, the datasets
    (ordinal, label, chemistry, content ids) and the read policy. Not the plan and not other modules, so a
    scan can be extended with new units or re-run for one module and keep every other result."""
    return I.digest([contigs_sha, [[d.ordinal, d.dataset_id, d.chemistry, list(d.content_ids)] for d in resolved],
                     policy])


def _survey_key(base, unit: ScanUnit) -> str:
    """The unit survey's key: the unit's own geometry, not its place in the plan (index, neighbours), so a
    unit planned again in another plan (an extended browser scan, a --zones subset) keeps its survey."""
    return I.digest([base, unit.unit_id, unit.chrom, unit.start, unit.end, unit.zone_start, unit.zone_end,
                     unit.mode])


def _unit_key(base, module, version, settings_hash, cal_sha, unit: ScanUnit, upstream_keys) -> str:
    return I.digest([base, module, version, settings_hash, cal_sha, unit.unit_id, unit.start, unit.end,
                     unit.zone_start, unit.zone_end, upstream_keys])


# ------------------------------------------------------------------ unit work
def _unit_dir(out: Path, module: str, unit: ScanUnit) -> Path:
    return out / WORK / module / I.unit_dir_name(unit.unit_id)


def _marker_valid(folder: Path, key: str) -> Optional[dict]:
    try:
        marker = _read_json(folder / 'marker.json')
    except (OSError, ValueError):
        return None
    if marker.get('schema') != UNIT_MARKER_SCHEMA or marker.get('key') != key:
        return None
    for rel, rec in (marker.get('files') or {}).items():
        p = folder / rel
        if not p.is_file() or p.stat().st_size != rec.get('size') or _sha(p) != rec.get('sha256'):
            return None
    return marker


def _publish_marker(folder: Path, key: str, extra: dict) -> dict:
    files = {}
    for p in sorted(folder.rglob('*')):
        # engine/ holds the wrapped engine's own run directory (kept for inspection, never read back)
        if p.is_file() and p.name != 'marker.json' and p.relative_to(folder).parts[0] != 'engine':
            files[p.relative_to(folder).as_posix()] = dict(size=p.stat().st_size, sha256=_sha(p))
    marker = dict(schema=UNIT_MARKER_SCHEMA, key=key, files=files, **extra)
    _write_json(folder / 'marker.json', marker)
    return marker


def _save_tracks(folder: Path, tracks: Mapping[str, Any]) -> None:
    import numpy as np
    buf_path = folder / 'tracks.npz'
    tmp = folder / '.tracks.npz.tmp'
    with open(tmp, 'wb') as handle:
        np.savez_compressed(handle, **{k: np.asarray(v) for k, v in sorted(tracks.items())})
    os.replace(tmp, buf_path)


def _load_tracks(folder: Path) -> Optional[dict]:
    import numpy as np
    p = folder / 'tracks.npz'
    if not p.is_file():
        return None
    with np.load(p) as data:
        return {k: data[k] for k in data.files}


_TASK_CONTEXT = {}


def _task_spec(spec):
    """Load one immutable scan context per worker, instead of shipping the plan
    and all module keys with each unit (quadratic in a genome-wide unit count)."""
    path = spec.get('context_path')
    if path is None:
        return spec, tuple(ScanUnit.from_json(u) for u in spec['plan'])
    if _TASK_CONTEXT.get('path') != path:
        common = _read_json(Path(path))
        _TASK_CONTEXT.clear()
        _TASK_CONTEXT.update(path=path, common=common,
                             plan=tuple(ScanUnit.from_json(u) for u in common['plan']))
    return dict(_TASK_CONTEXT['common'], **{k: v for k, v in spec.items() if k != 'context_path'}), _TASK_CONTEXT['plan']


def run_unit_task(spec: dict) -> dict:
    """Run every module of one unit (in this process or a worker); returns the unit's summary."""
    spec, plan_units = _task_spec(spec)
    out = Path(spec['out'])
    unit = ScanUnit.from_json(spec['unit'])
    datasets = tuple(Dataset(d['dataset_id'], d['ordinal'], d['chemistry'], tuple(d['paths']),
                             tuple(d['content_ids'])) for d in spec['datasets'])
    inputs = ScanInputs(datasets=datasets, read_policy=spec['read_policy'], reference=spec['reference'],
                        plan=plan_units, cores=int(spec['cores']))
    log_path = out / WORK / 'logs' / (I.unit_dir_name(unit.unit_id) + '.log')
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        handle = stack.enter_context(open(log_path, 'a'))
        if spec.get('isolate'):
            stack.enter_context(redirect_stdout(handle))
            stack.enter_context(redirect_stderr(handle))
            try:
                from fiberhmm.inference.consensus.execution import single_threaded_blas
                stack.enter_context(single_threaded_blas())
            except ImportError:  # pragma: no cover
                pass
        progress = _LogProgress(handle)
        # Unit survey: emptiness and callable territory (orchestrator-owned, independent of modules).
        survey_dir = out / WORK / '_unit' / I.unit_dir_name(unit.unit_id)
        survey_key = _survey_key(spec['scan_id_core'], unit)
        marker = _marker_valid(survey_dir, survey_key)
        if marker is None:
            if survey_dir.exists():
                shutil.rmtree(survey_dir)
            survey_dir.mkdir(parents=True)
            survey = unit_survey(datasets, unit, inputs.read_policy) if datasets else dict(records=-1, callable={})
            _write_json(survey_dir / 'survey.json', survey)
            marker = _publish_marker(survey_dir, survey_key, {})
        survey = _read_json(survey_dir / 'survey.json')
        empty = survey['records'] == 0
        if empty:
            # An empty unit (no record in its span) runs no module; the survey marker records it.
            return dict(unit=unit.unit_id, index=unit.index, records=0, status='empty', modules={})
        registry = spec.get('registry')
        upstream_results: Dict[str, Any] = {}
        molecular_input: Dict[str, bool] = {}
        module_status = {}
        module_seconds = {}
        for m in spec['modules']:
            name = m['name']
            folder = _unit_dir(out, name, unit)
            key = m['unit_keys'][unit.unit_id]
            if _marker_valid(folder, key) is not None:
                module_status[name] = 'reused'
                continue
            if folder.exists():
                shutil.rmtree(folder)
            folder.mkdir(parents=True)
            module_started = time.monotonic()
            mod = load_module(name, registry)
            settings = m['settings']
            calibration = m.get('calibration')
            progress.report(name, f'{unit.unit_id}: running')
            upstream = {}
            for dep in m['needs']:
                if dep not in upstream_results:
                    upstream_results[dep] = unit_result_from_json(
                        _read_json(_unit_dir(out, dep, unit) / 'result.json.gz'))
                upstream[dep] = upstream_results[dep]
            scratch = folder / 'engine'
            scratch.mkdir()
            result = mod.run_unit(unit, inputs, settings, calibration, upstream, scratch, progress)
            molecules = []
            skipped = []
            if mod.wants_molecules:
                # Every record overlapping the unit: molecule-wide outputs do not depend on the unit, and
                # finalize keeps each record's result from the unit that owns it under the current plan
                # (so a scan extended with new units never reports a record twice).
                for ds, i, read, sha, occ in overlapping_records(datasets, unit, inputs.read_policy):
                    rec = record_ref(read, ds, i, sha, occ, unit.chrom)
                    if ds.paths[i] not in molecular_input:
                        molecular_input[ds.paths[i]] = _declares_molecular(ds.paths[i])
                    if getattr(mod, 'molecule_frame', 'molecular') == 'molecular' and not molecular_input[ds.paths[i]] \
                            and not _carry_record(read):
                        # FullMolecule.read carries molecular-frame annotations (the contract); a record whose
                        # calls cannot be carried there is counted (manifest status.records_not_carried)
                        skipped.append(rec)
                        continue
                    mr = mod.molecule_outputs(FullMolecule(rec, read, ds), settings, calibration)
                    if mr is not None:
                        molecules.append(mr)
            if scratch.exists() and not m.get('keep_engine', True):
                shutil.rmtree(scratch)
            if result.tracks is not None:
                _save_tracks(folder, result.tracks)
            _write_json(folder / 'result.json.gz', unit_result_to_json(result))
            _write_json(folder / 'molecules.json.gz', [molecule_result_to_json(x) for x in molecules])
            if skipped:
                _write_json(folder / 'skipped.json.gz', [r.to_json() for r in skipped])
            _publish_marker(folder, key, dict(status=result.status, unit=unit.unit_id, module=name))
            module_status[name] = result.status
            module_seconds[name] = time.monotonic() - module_started
        return dict(unit=unit.unit_id, index=unit.index, records=survey['records'], status='empty' if empty else 'done',
                    modules=module_status, module_seconds=module_seconds)


# ------------------------------------------------------------------ dispatch
def _plan_jobs(pending: int, cores: int, window_jobs: int) -> Tuple[int, int]:
    if pending <= 1:
        return 1, max(1, cores)
    jobs = min(pending, cores, window_jobs) if window_jobs else min(pending, cores)
    jobs = max(1, jobs)
    return jobs, max(1, cores // jobs)


def _dispatch(specs: List[dict], jobs: int, progress, cancel) -> List[dict]:
    total = len(specs)
    done: List[dict] = []
    started = time.monotonic()

    def finished(summary):
        done.append(summary)
        elapsed = time.monotonic() - started
        eta = elapsed / len(done) * (total - len(done)) if len(done) < total else 0.0
        progress.report('units', f"{summary['unit']} {summary['status']}", completed=len(done), total=total,
                        eta_seconds=round(eta, 1))

    if jobs <= 1:
        for spec in specs:
            if cancel is not None and cancel.is_set():
                raise KeyboardInterrupt('scan cancelled')
            finished(run_unit_task(spec))
        return done
    from concurrent.futures import as_completed
    from fiberhmm.inference.consensus.execution import shared_worker_pool
    with shared_worker_pool(jobs) as pool:
        executor = pool.get()
        futures = {executor.submit(run_unit_task, dict(spec, isolate=True)): spec for spec in specs}
        for future in as_completed(futures):
            spec = futures[future]
            if cancel is not None and cancel.is_set():
                raise KeyboardInterrupt('scan cancelled')
            try:
                summary = future.result()
            except BaseException:
                uid = spec['unit']['unit_id']
                print(f'units: {uid} failed; see {Path(spec["out"]) / WORK / "logs"}. Finished units are kept: '
                      'fix the cause and rerun the same command with --continue', file=sys.stderr, flush=True)
                raise
            finished(summary)
    return done


# ------------------------------------------------------------------ the entry point
def run_units(datasets: Sequence[Mapping[str, Any]], units, *, modules: Sequence[str],
              params: Optional[Mapping[str, Any]] = None, read_policy: Optional[Mapping[str, Any]] = None,
              out, annotate: Optional[bool] = None, progress=None, cancel=None, cores: int = 1,
              window_jobs: int = 0, continue_run: bool = False, units_info: Optional[Mapping[str, Any]] = None,
              registry: Optional[Mapping[str, Any]] = None, command: Optional[Sequence[str]] = None,
              extend: bool = False, evidence=None) -> dict:
    """Compute (or extend, with ``continue_run``) a scan directory. Returns a summary dict.

    ``units``: the plan, as returned by :func:`fiberhmm.scan.units.plan` (``(units, info)``) or a
    sequence of :class:`ScanUnit` with ``units_info``. ``datasets``: ``[{dataset_id, paths, chemistry?}]``.
    ``evidence`` optionally supplies records as an iterable or zero-argument
    callback; datasets provide header/headers, and multi-input records are
    (dataset_id, input_index, AlignedSegment) tuples. See CONTRACTS.md §14.
    ``continue_run`` finishes an interrupted scan with the same arguments; ``extend`` (with
    ``continue_run``) also accepts another plan or other module settings for the same inputs and read
    policy (a browser scan appending units): every unit result whose key still matches is reused."""
    progress = progress or _NullProgress()
    if isinstance(units, tuple) and len(units) == 2 and isinstance(units[1], dict):
        units, units_info = units
    units = tuple(units)
    units_info = dict(units_info or {})
    if not units:
        raise ScanRefused('the plan has no units')
    if len({u.unit_id for u in units}) != len(units):
        raise ScanRefused('the plan has duplicate unit ids')
    out = Path(out).expanduser().resolve()
    raw_policy = read_policy or (params or {}).get('read_policy') or {}
    policy = resolve_read_policy(raw_policy)
    # Modules see only the caps the caller gave (as fiberhmm-consensus/-nfr see only their cap flags), so an
    # explicit 0 or seed 1 overrides a parameter file and an absent cap leaves the module's own value.
    explicit_caps = dict(raw_policy.get('caps') or {})
    out.mkdir(parents=True, exist_ok=True)
    from fiberhmm.io.run_state import DirectoryBusy, DirectoryLock
    lock = DirectoryLock(out / LOCK, 'fiberhmm-scan output directory', remove=True)
    try:
        lock.acquire()
    except DirectoryBusy as error:
        raise ScanRefused(str(error)) from None
    try:
        if evidence is not None:
            if not continue_run and any(p.name != LOCK for p in out.iterdir()):
                raise ScanRefused('evidence output directory is not empty; use continue_run or a new directory')
            from .evidence import materialize_evidence
            datasets = materialize_evidence(datasets, evidence, out / WORK / '_evidence', cancel=cancel)
        signal_input = any(d.get('input_type') == 'signal-bam' for d in datasets)
        if signal_input:
            if not continue_run and any(p.name not in (LOCK,) for p in out.iterdir()) and evidence is None:
                raise ScanRefused('signal output directory is not empty; use continue_run or a new directory')
            from .signal import materialize
            datasets = materialize(datasets, out / WORK / '_signal',
                                   cancel=cancel.is_set if cancel is not None else None)
        resolved, reference = resolve_datasets(datasets)
        contigs = list(reference.items())
        for u in units:
            if u.chrom not in reference:
                raise ScanRefused(f'unit {u.unit_id}: contig not in the BAM header')
        mods = resolve_modules(modules, params or {}, explicit_caps, registry)
        annotate = default_annotate(units_info) if annotate is None else bool(annotate)
        return _run_locked(out, resolved, reference, contigs, units, units_info, mods, policy, annotate,
                           progress, cancel, cores, window_jobs, continue_run, registry, command, extend,
                           prepared_evidence=evidence is not None or signal_input)
    finally:
        lock.release()


def _manifest_core(resolved, contigs, units, units_info, mods, policy, cal_shas) -> dict:
    datasets = []
    for d in resolved:
        files = [dict(name=Path(p).name, path=p, input_index=i, content_id=cid, size=os.stat(p).st_size)
                 for i, (p, cid) in enumerate(zip(d.paths, d.content_ids))]
        for f in files:
            f['fiberhmm_calls'] = _calls_program(f['path'])
            from .signal import input_metadata
            signal = input_metadata(f['path'])
            if signal:
                f['signal_input'] = signal[-1]
        datasets.append(dict(dataset_id=d.dataset_id, ordinal=d.ordinal, chemistry=d.chemistry, files=files))
    from fiberhmm import __version__
    manifest = dict(schema=SCHEMA, fiberhmm=dict(version=__version__),
                    reference=dict(contigs=[[c, n] for c, n in contigs], contigs_sha256=I.contigs_sha256(contigs)),
                    frame=_frame_of(resolved), datasets=datasets, read_policy=policy,
                    units=_units_block(units, units_info),
                    modules={n: _module_block(n, m, s, cal_shas.get(n)) for n, m, s in mods})
    manifest['scan_id'] = scan_id_of(manifest)
    return manifest


def _frame_of(resolved) -> dict:
    import pysam
    lifted = []
    for d in resolved:
        for p in d.paths:
            with pysam.AlignmentFile(p, 'rb', check_sq=False) as bam:
                pgs = [pg for pg in bam.header.to_dict().get('PG', []) if str(pg.get('PN', pg.get('ID', ''))).startswith('fiberhmm-lift')]
            lifted.append(bool(pgs))
    if lifted and all(lifted):
        return dict(kind='lifted', lift_key=None, target=None)
    return dict(kind='linear')


def _check_continue(out: Path, manifest: dict, continue_run: bool, base: str, extend: bool = False,
                    prepared_evidence: bool = False) -> None:
    path = out / 'manifest.json'
    entries = [p for p in out.iterdir() if p.name != LOCK]
    if prepared_evidence and not continue_run:
        entries = [p for p in entries if p.name != WORK]
    if not continue_run:
        if entries:
            raise ScanRefused(f'{out} is not empty; existing results are never overwritten (to finish an interrupted '
                              'scan there, add --continue)')
        return
    if not entries:
        return
    pending = out / WORK / 'scan_key.json'
    previous = None
    if pending.is_file():
        previous = _read_json(pending)
    if previous is None:
        raise ScanRefused(f'--continue: {out} holds no scan (no manifest.json); give the output directory of an '
                          'interrupted fiberhmm-scan run')
    if extend:
        if previous.get('base') != base:
            raise ScanRefused(f'extending the scan in {out} refused: the reference, the datasets (files, labels, '
                              'chemistry) or the read policy differ')
        return
    if previous.get('scan_id') != manifest['scan_id'] or previous.get('base') != base:
        diffs = [k for k in ('reference', 'datasets', 'read_policy', 'units', 'modules')
                 if I.canonical_json(_key_view(previous, k)) != I.canonical_json(_key_view(manifest, k))]
        if previous.get('base') != base and not diffs:
            diffs = ['dataset labels']
        raise ScanRefused('--continue refused: inputs, plan, read policy or module settings differ from the scan in '
                          f'{out} ({", ".join(diffs) or "scan key"}). Rerun with the original arguments, or write a '
                          'new scan to a new output directory. Only --cores, --window-jobs and --json-progress may '
                          'change between attempts')


def _key_view(manifest, key):
    value = manifest.get(key)
    if key == 'units' and isinstance(value, dict):
        return {k: v for k, v in value.items() if k != 'count'}
    if key == 'datasets' and isinstance(value, list):
        return [[d.get('ordinal'), d.get('chemistry'), [f.get('content_id') for f in d.get('files', [])]] for d in value]
    if key == 'modules' and isinstance(value, dict):
        return {n: [m.get('version'), m.get('settings_hash')] for n, m in value.items()}
    if key == 'reference' and isinstance(value, dict):
        return value.get('contigs_sha256')
    return value


def _run_locked(out, resolved, reference, contigs, units, units_info, mods, policy, annotate, progress, cancel,
                cores, window_jobs, continue_run, registry, command, extend=False, prepared_evidence=False) -> dict:
    started = time.monotonic()
    # Calibration (dataset-level, before any unit): nuc only in v1.
    cal_shas: Dict[str, Optional[str]] = {}
    calibrations: Dict[str, Optional[dict]] = {}
    calibration_seconds = {}
    inputs = ScanInputs(datasets=resolved, read_policy=policy, reference=reference, plan=units, cores=cores)
    pre = _manifest_core(resolved, contigs, units, units_info, mods, policy, {})
    base = base_key(pre['reference']['contigs_sha256'], resolved, policy)
    _check_continue(out, pre, continue_run, base, extend, prepared_evidence)
    (out / WORK).mkdir(parents=True, exist_ok=True)
    _write_json(out / WORK / 'scan_key.json', dict(scan_id=pre['scan_id'], base=base, reference=pre['reference'],
                                                  datasets=pre['datasets'], read_policy=pre['read_policy'],
                                                  units=pre['units'], modules=pre['modules']))
    plan_digest = I.digest([u.to_json() for u in units])
    for name, mod, settings in mods:
        calibration_started = time.monotonic()
        # Calibration runs once per scan; a cached record is reused only for the same inputs, module
        # settings and plan.
        cal_dir = out / WORK / name / '_calibration'
        cal_key = I.digest([base, name, mod.version, I.settings_hash(settings),
                            plan_digest if mod.calibration_uses_plan(settings) else None])
        cal = None
        cached = cal_dir / 'calibration.json'
        if continue_run and _marker_valid(cal_dir, cal_key) is not None and cached.is_file():
            cal = _read_json(cached)
        else:
            if cal_dir.exists():
                shutil.rmtree(cal_dir)
            cal_dir.mkdir(parents=True)
            cal = mod.calibrate(inputs, units, settings, cal_dir, progress)
            _write_json(cached, cal)
            _publish_marker(cal_dir, cal_key, {})
        calibrations[name] = cal
        cal_path = out / name / 'extra' / 'calibration.json'
        if cal is not None:
            _write_json(cal_path, cal, indent=1)
            cal_shas[name] = _sha(cal_path)
        else:
            if cal_path.is_file():
                cal_path.unlink()
            cal_shas[name] = None
        calibration_seconds[name] = time.monotonic() - calibration_started
    manifest = _manifest_core(resolved, contigs, units, units_info, mods, policy, cal_shas)
    scan_id = manifest['scan_id']
    # Unit keys per module (upstream keys chained).
    unit_keys: Dict[str, Dict[str, str]] = {}
    for name, mod, settings in mods:
        sh = I.settings_hash(settings)
        keys = {}
        for u in units:
            up = [[dep, unit_keys[dep][u.unit_id]] for dep in sorted(mod.needs(settings))]
            keys[u.unit_id] = _unit_key(base, name, mod.version, sh, cal_shas.get(name), u, up)
        unit_keys[name] = keys
    module_specs = [dict(name=n, settings=s, needs=sorted(m.needs(s)), calibration=calibrations.get(n),
                         unit_keys=unit_keys[n]) for n, m, s in mods]
    core = base
    pending, reused = [], 0
    for u in units:
        survey_dir = out / WORK / '_unit' / I.unit_dir_name(u.unit_id)
        surveyed = _marker_valid(survey_dir, _survey_key(core, u)) is not None
        empty = surveyed and _read_json(survey_dir / 'survey.json')['records'] == 0
        all_ok = surveyed and (empty or all(_marker_valid(_unit_dir(out, n, u), unit_keys[n][u.unit_id]) is not None
                                            for n, _m, _s in mods))
        if all_ok:
            reused += 1
        else:
            pending.append(u)
    jobs, inner = _plan_jobs(len(pending), cores, window_jobs)
    plan_json = [u.to_json() for u in units]
    ds_json = [d.to_json() for d in resolved]
    common = dict(out=str(out), datasets=ds_json, plan=plan_json, read_policy=policy,
                  reference=reference, cores=inner, modules=module_specs, scan_id_core=core)
    context_path = out / WORK / ('dispatch_' + I.digest(common, 16) + '.json.gz')
    if pending and not context_path.exists():
        _write_json(context_path, common)
    specs = [dict(context_path=str(context_path), unit=u.to_json(), registry=registry) for u in pending]
    progress.report('units', f'{reused} complete from an earlier attempt; {len(pending)} to run ({jobs} at a time)'
                    if reused else f'{len(pending)} to run ({jobs} at a time)', completed=reused, total=len(units))
    completed = _dispatch(specs, jobs, progress, cancel)
    from .finalize import finalize
    # No module writes per-molecule MA groups (a peaks-only scan): nothing to annotate.
    annotate = bool(annotate) and any(m.ma_groups() for _n, m, _s in mods)
    if annotate:
        foreign = [p for d in resolved for p in d.paths if not _declares_molecular(p)]
        if foreign:
            # Inputs without either a declaration or fibertools calling provenance need frame conversion
            # for their other annotations; keep those scans in table mode.
            progress.report('scan', f'{len(foreign)} input(s) do not declare molecular coordinates in their header '
                                    '(coord=molecular or fibertools calling provenance): per-molecule results of every '
                                    'input are written as tables, not annotated BAMs')
            annotate = False
    finalize_started = time.monotonic()
    summary = finalize(out, manifest, resolved, reference, units, mods, unit_keys, annotate, progress,
                       command=command, calibrations=calibrations)
    run = dict(attempts=_attempts(out, cores, window_jobs, time.monotonic() - started, len(pending), reused))
    run['attempts'][-1]['calibration_seconds'] = calibration_seconds
    run['attempts'][-1]['module_unit_seconds'] = {n: sum(x.get('module_seconds', {}).get(n, 0.) for x in completed)
                                               for n, _m, _s in mods}
    run['attempts'][-1]['finalize_seconds'] = time.monotonic() - finalize_started
    _write_json(out / RUN_JSON, run, indent=1)
    return summary


def _attempts(out: Path, cores, window_jobs, seconds, ran, reused) -> list:
    path = out / RUN_JSON
    previous = []
    if path.is_file():
        try:
            previous = _read_json(path).get('attempts', [])
        except (OSError, ValueError):
            previous = []
    from fiberhmm import __version__
    return previous + [dict(fiberhmm=__version__, cores=cores, window_jobs=window_jobs, seconds=round(seconds, 1),
                            units_run=ran, units_reused=reused, finished=time.strftime('%Y-%m-%dT%H:%M:%S'))]
