"""Resumable region-parallel runs: work directory, run identity, progress lines, signals.

``fiberhmm-call --region-parallel`` keeps every finished region's temporary BAM in a work directory (by default
``.<output name>.fiberhmm-work`` beside the output) together with

* ``manifest.json`` -- the run identity: input BAM identity (path, size and content SHA-256 of the BAM and its
  index, header SHA-256), every effective parameter (resolved chemistry and defaults, the @PG record without its
  command line), model/profile/mask/reference content digests, the region plan and the FiberHMM version; and
* ``region_NNNNNN.done.json`` -- one marker per finished region, written atomically after the worker returned,
  holding that region's counts and the size and SHA-256 of its BAM.

``--resume`` reuses every region whose marker and BAM digest validate, reruns missing, partial or altered regions,
then merges and publishes atomically as usual. A work directory whose identity differs is refused, never mixed.
Identities compare file *content*; metadata only decides whether a remembered digest can be reused (the rule is in
:mod:`fiberhmm.io.run_state`). The directory is owned exclusively (``flock`` on ``.lock`` inside it) from the
moment it is inspected until the run has published and cleaned up, so a second invocation on the same work
directory is refused while the first is alive. It is removed after a successful publish unless ``--keep-work-dir``.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import threading
import time
import uuid

from fiberhmm.io.run_state import DigestMemo, DirectoryBusy, DirectoryLock, sha256_file as _sha256

WORK_SCHEMA = 'fiberhmm.call.work.v1'
PROGRESS_SCHEMA = 'fiberhmm.progress.v1'
MANIFEST = 'manifest.json'
LOCK = '.lock'


class ResumeRefused(ValueError):
    """The work directory belongs to a different input or parameter set (or --resume was not given)."""


class RunInterrupted(KeyboardInterrupt):
    """SIGTERM/SIGHUP delivered to the main thread, raised like Ctrl-C so every cleanup path runs."""

    def __init__(self, signum):
        super().__init__(f'signal {signum}')
        self.signum = signum


@contextmanager
def interrupt_on_terminate():
    """In the main thread, turn SIGTERM/SIGHUP into :class:`RunInterrupted` (a KeyboardInterrupt).

    Without this a plain ``kill`` ends the interpreter without running any ``finally``: a temporary output or a
    half-written work-directory marker could remain. Ignored signals stay ignored (``nohup``)."""
    import signal
    import threading
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    previous = {}

    def handler(signum, frame):
        raise RunInterrupted(signum)
    for sig in (signal.SIGTERM, getattr(signal, 'SIGHUP', None)):
        if sig is None or signal.getsignal(sig) in (signal.SIG_IGN, None):
            continue
        previous[sig] = signal.signal(sig, handler)
    try:
        yield
    finally:
        for sig, old in previous.items():
            signal.signal(sig, old)


def worker_initializer(initializer, *args):
    """Pool initializer: workers die on SIGTERM (default action) instead of inheriting the parent's handler."""
    import signal
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    if initializer is not None:
        initializer(*args)


def abort_executor(executor):
    """Stop a ProcessPoolExecutor now: cancel queued regions and kill running workers.

    ``shutdown(wait=True)`` would wait for every queued region. Killed workers leave only marker-less partial
    region BAMs, which a resume reruns."""
    processes = list((getattr(executor, '_processes', None) or {}).values())
    try:
        executor.shutdown(wait=False, cancel_futures=True)
    except Exception:
        pass
    for process in processes:
        try:
            process.kill()
        except Exception:
            pass
    for process in processes:
        try:
            process.join(5)
        except Exception:
            pass


def default_work_dir(output_bam):
    output = Path(output_bam).expanduser().absolute()
    return output.parent/f'.{output.name}.fiberhmm-work'


def sha256_file(path, chunk=1 << 20):
    return _sha256(path, chunk)


def input_identity(bam_path, memo=None):
    """Path, size, content SHA-256 and header SHA-256 of an input BAM, and the content identity of its index.

    ``memo`` (a :class:`~fiberhmm.io.run_state.DigestMemo`) avoids rehashing files whose metadata shows they are
    unchanged; the identity itself holds only content."""
    import pysam
    memo = memo if memo is not None else DigestMemo()
    path = str(Path(bam_path).expanduser().resolve())
    identity = dict(path=path, size=os.path.getsize(path), sha256=memo.sha256(path))
    with pysam.AlignmentFile(path, 'rb', check_sq=False) as bam:
        identity['header_sha256'] = hashlib.sha256(str(bam.header).encode()).hexdigest()
    stem = path[:-4] if path.endswith('.bam') else path
    for candidate in (path+'.csi', path+'.bai', stem+'.csi', stem+'.bai'):
        if os.path.exists(candidate):
            identity['index'] = dict(path=candidate, size=os.path.getsize(candidate), sha256=memo.sha256(candidate))
            break
    return identity


def file_digest(path, memo=None):
    """SHA-256 of a model, profile or mask file; ``None`` passes through."""
    if not path:
        return None
    memo = memo if memo is not None else DigestMemo()
    return dict(path=str(Path(path).expanduser().resolve()), sha256=memo.sha256(path))


def reference_identity(path, memo=None):
    """Reference FASTA: path, size, content SHA-256 and the .fai digest."""
    if not path:
        return None
    memo = memo if memo is not None else DigestMemo()
    identity = dict(path=str(Path(path).expanduser().resolve()), size=os.path.getsize(path),
                    sha256=memo.sha256(path))
    for fai in (str(path)+'.fai',):
        if os.path.exists(fai):
            identity['fai_sha256'] = memo.sha256(fai)
    return identity


def jsonable(value):
    """Canonical JSON form: sets sorted, tuples as lists."""
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (set, frozenset)):
        return sorted(jsonable(v) for v in value)
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def differences(previous, current, prefix=''):
    if isinstance(previous, dict) and isinstance(current, dict):
        out = []
        for key in sorted(set(previous) | set(current)):
            out += differences(previous.get(key), current.get(key), f'{prefix}.{key}' if prefix else str(key))
        return out
    return [] if previous == current else [prefix or '(root)']


def _atomic_json(path, value):
    path = Path(path)
    # pid + thread + random: two writers (even threads of one process) never share a temporary.
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.{threading.get_ident()}.{uuid.uuid4().hex[:8]}.tmp')
    try:
        with open(temporary, 'w') as handle:
            json.dump(value, handle, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.remove(temporary)
        except OSError:
            pass
        raise


class WorkDir:
    """A region-parallel run's resumable scratch directory, owned exclusively while open.

    ``identity`` is the run identity, or a callable ``identity(memo)`` that builds it with a
    :class:`~fiberhmm.io.run_state.DigestMemo` restored from the directory's manifest (so unchanged inputs are not
    rehashed on every resume). Call :meth:`release` (or :meth:`remove`) when the run ends."""

    def __init__(self, path, identity, pg_record=None, *, resume=False, log=None):
        self.path = Path(path).expanduser().absolute()
        self._identity = identity
        self.identity = None if callable(identity) else jsonable(identity)
        self.pg_record = pg_record
        self.resume = bool(resume)
        self.log = log or (lambda message: print(message, file=sys.stderr, flush=True))
        self.reused = 0
        self.lock = None
        self.memo = DigestMemo()

    def open(self):
        self.path.mkdir(parents=True, exist_ok=True)
        try:
            self.lock = DirectoryLock(self.path/LOCK, 'fiberhmm-call work directory', log=self.log).acquire()
        except DirectoryBusy as exc:
            raise ResumeRefused(str(exc)) from None
        try:
            return self._open_locked()
        except BaseException:
            self.release()
            raise

    def _open_locked(self):
        manifest = self.path/MANIFEST
        occupied = any(entry.name != LOCK for entry in self.path.iterdir())
        if occupied and not self.resume:
            raise ResumeRefused(
                f'work directory {self.path} from an earlier interrupted run exists. Rerun the same command with '
                '--resume to reuse its finished regions, or delete that directory to start over')
        previous = None
        if occupied:
            try:
                previous = json.loads(manifest.read_text())
            except (OSError, ValueError):
                raise ResumeRefused(f'{self.path} has no readable {MANIFEST}; it is not a fiberhmm-call work '
                                    'directory (delete it, or choose another --work-dir)') from None
            if previous.get('schema') != WORK_SCHEMA:
                raise ResumeRefused(f'{manifest} is not a {WORK_SCHEMA} manifest; delete {self.path} to start over')
            self.memo = DigestMemo(previous.get('digests'))
        if callable(self._identity):
            self.identity = jsonable(self._identity(self.memo))
        if previous is not None:
            changed = differences(previous.get('identity'), self.identity)
            if changed:
                shown = ', '.join(changed[:8])+(f' (+{len(changed)-8} more)' if len(changed) > 8 else '')
                raise ResumeRefused(
                    f'--resume refused: the input or parameters differ from the interrupted run in {self.path} '
                    f'({shown}). Rerun with the original input and options, or delete {self.path} to start over')
            # The published header records the original command line, so reused and new regions agree.
            if previous.get('pg_record') is not None:
                self.pg_record = previous['pg_record']
            previous['digests'] = self.memo.to_json()
            _atomic_json(manifest, previous)
            return self
        if self.resume:
            self.log(f'  NOTE: --resume: no earlier work directory at {self.path}; starting a new run')
        _atomic_json(manifest, dict(schema=WORK_SCHEMA, created=time.strftime('%Y-%m-%dT%H:%M:%S'),
                                    identity=self.identity, pg_record=self.pg_record,
                                    digests=self.memo.to_json()))
        return self

    def marker(self, index):
        return self.path/f'region_{index:06d}.done.json'

    def completed(self, work_items):
        """{index: RegionBamResult} for regions whose marker and BAM (size and SHA-256) validate."""
        from fiberhmm.inference.region_types import RegionBamResult
        done = {}
        for index, item in enumerate(work_items):
            try:
                value = json.loads(self.marker(index).read_text())
            except (OSError, ValueError):
                continue
            bam = self.path/value.get('bam', '')
            if (value.get('region') != jsonable(item.region) or value.get('passthrough') != bool(item.passthrough)
                    or Path(item.temp_bam_path).name != value.get('bam') or not bam.is_file()
                    or bam.stat().st_size != value.get('bam_size') or not value.get('bam_sha256')
                    or sha256_file(bam) != value['bam_sha256']):
                continue
            done[index] = RegionBamResult(
                temp_bam_path=str(bam), total_reads=int(value['total_reads']),
                reads_with_footprints=int(value['reads_with_footprints']), written=int(value['written']),
                temp_tsv_path=None, skip_reasons=dict(value.get('skip_reasons') or {}),
                metrics=dict(value.get('metrics') or {}), failure_messages=tuple(value.get('failure_messages') or ()))
        self.reused = len(done)
        return done

    def mark_done(self, index, item, result):
        bam = Path(result.temp_bam_path)
        exists = bam.exists()
        _atomic_json(self.marker(index), dict(
            region=jsonable(item.region), passthrough=bool(item.passthrough), bam=bam.name,
            bam_size=bam.stat().st_size if exists else None,
            bam_sha256=sha256_file(bam) if exists else None,
            total_reads=int(result.total_reads), reads_with_footprints=int(result.reads_with_footprints),
            written=int(result.written), skip_reasons=jsonable(dict(result.skip_reasons)),
            metrics=jsonable(dict(result.metrics)), failure_messages=list(result.failure_messages)))

    def release(self):
        """Give up ownership (the directory and its contents stay)."""
        if self.lock is not None:
            self.lock.release()
            self.lock = None

    def remove(self):
        """Delete the directory, then give up ownership."""
        if self.path.exists():
            for entry in self.path.iterdir():
                if entry.name == LOCK:
                    continue
                if entry.is_dir() and not entry.is_symlink():
                    shutil.rmtree(entry, ignore_errors=True)
                else:
                    try:
                        entry.unlink()
                    except OSError:
                        pass
        if self.lock is not None:
            self.lock.remove = True
        self.release()
        try:
            self.path.rmdir()
        except OSError:
            pass


class ProgressJSON:
    """Machine-readable progress: one JSON object per line on stderr (``-``) or appended to a file.

    Every line has ``schema`` (fiberhmm.progress.v1), ``tool``, ``event`` and ``time`` (Unix seconds); see the
    calling workflow docs for the event fields."""

    def __init__(self, target, tool):
        self.tool = tool
        self.target = target
        self.handle = None if target in (None, '-') else open(target, 'a', buffering=1)

    def __call__(self, event, **fields):
        line = json.dumps(dict(schema=PROGRESS_SCHEMA, tool=self.tool, event=event, time=round(time.time(), 3),
                               **jsonable(fields)), sort_keys=False)
        handle = self.handle or sys.stderr
        print(line, file=handle, flush=True)

    def close(self):
        if self.handle is not None:
            self.handle.close()
            self.handle = None
