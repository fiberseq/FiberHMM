"""Content identity and exclusive ownership for resumable runs.

Shared by ``fiberhmm-call --resume`` (region work directories),
``fiberhmm-consensus --continue`` (run manifests) and ``fiberhmm-pipeline``
(step markers).

Content identity
----------------
A resumable run decides whether earlier work can be reused by comparing the
*content* of its inputs and kept artifacts, never their metadata: an identity
is ``{"path", "size", "sha256"}`` (:func:`content_identity`). File metadata is
only used to avoid rehashing unchanged files (:class:`DigestMemo`):

    A remembered SHA-256 is reused only when the file's device, inode, size,
    modification time *and status-change time* (``st_ctime_ns``) all equal the
    values recorded when it was hashed; otherwise the file is hashed again.

``st_ctime_ns`` is what makes this safe. The kernel sets it on every write,
truncation, rename into place and metadata change, including the ``utime``
call that restores a modification time, and unprivileged tools cannot set it
back. So a file whose content changed while its size, mtime and inode were
preserved is always rehashed, and a file whose metadata alone changed (a
``chmod``, a sync client's extended attributes, a copy) is rehashed and then
accepted, because its digest still matches. SHA-256 runs at roughly 2-3 GB/s,
so a first run pays about one extra read of its inputs and later runs pay
nothing for unchanged files. (This relies on POSIX ``st_ctime`` semantics --
macOS, Linux, WSL; on native Windows it is the creation time.)

Ownership
---------
:class:`DirectoryLock` takes an exclusive ``flock`` on a lock file before a
directory's state is inspected and holds it until the run has published and
cleaned up, so two processes can never own one work directory. The kernel
releases the lock when its owner exits, however it exits, so there are no
stale locks to clean up; the lock file only records the owner (pid, host,
start time) for the refusal message. Where the filesystem cannot lock
(``ENOLCK``/``EOPNOTSUPP``, e.g. some network mounts) the run proceeds with a
warning, as before.
"""
from __future__ import annotations

import errno
import hashlib
import json
import os
import socket
import sys
import time
from typing import Optional

try:  # POSIX only
    import fcntl
except ImportError:  # pragma: no cover - Windows
    fcntl = None


def sha256_file(path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(chunk), b''):
            digest.update(block)
    return digest.hexdigest()


def stat_key(path) -> list:
    """``[dev, inode, size, mtime_ns, ctime_ns]``: when all are unchanged, so is the content."""
    s = os.stat(path)
    return [s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns]


class DigestMemo:
    """SHA-256 of files, remembered per path under the rule in the module docstring.

    ``entries`` is JSON-serialisable (``{real path: {"stat": [...], "sha256": ...}}``)
    so a run can store it beside its state and pass it back on the next run.
    """

    def __init__(self, entries: Optional[dict] = None):
        self.entries: dict = {}
        for path, entry in (entries or {}).items():
            if (isinstance(entry, dict) and isinstance(entry.get('stat'), list)
                    and isinstance(entry.get('sha256'), str)):
                self.entries[str(path)] = {'stat': list(entry['stat']), 'sha256': entry['sha256']}

    def sha256(self, path) -> str:
        real = os.path.realpath(path)
        before = stat_key(real)
        entry = self.entries.get(real)
        if entry is not None and entry['stat'] == before:
            return entry['sha256']
        digest = sha256_file(real)
        if stat_key(real) == before:  # not remembered if it changed while being read
            self.entries[real] = {'stat': before, 'sha256': digest}
        else:
            self.entries.pop(real, None)
        return digest

    def to_json(self) -> dict:
        return {path: dict(entry) for path, entry in sorted(self.entries.items())}


def content_identity(path, memo: Optional[DigestMemo] = None) -> dict:
    """``{"path", "size", "sha256"}`` of a file (``memo`` avoids rehashing unchanged files)."""
    memo = memo if memo is not None else DigestMemo()
    absolute = os.path.abspath(path)
    return {'path': absolute, 'size': os.path.getsize(absolute), 'sha256': memo.sha256(absolute)}


def load_memo_file(path) -> DigestMemo:
    try:
        with open(path, encoding='utf-8') as handle:
            return DigestMemo(json.load(handle))
    except (OSError, ValueError, AttributeError):
        return DigestMemo()


def save_memo_file(path, memo: DigestMemo) -> None:
    temporary = f'{path}.{os.getpid()}.{id(memo):x}.tmp'
    try:
        with open(temporary, 'w', encoding='utf-8') as handle:
            json.dump(memo.to_json(), handle)
        os.replace(temporary, path)
    except OSError:
        try:
            os.remove(temporary)
        except OSError:
            pass


class DirectoryBusy(RuntimeError):
    """Another live process owns the directory."""


_UNLOCKABLE = {getattr(errno, name) for name in ('ENOLCK', 'EOPNOTSUPP', 'ENOTSUP', 'ENOSYS')
               if hasattr(errno, name)}


class DirectoryLock:
    """Exclusive ownership of a directory through ``flock`` on ``lock_path``.

    ``acquire()`` raises :class:`DirectoryBusy` naming the owner when another
    process (or another open of the lock in this process) holds it. ``remove``
    deletes the lock file on release (for directories that are removed or must
    look empty afterwards); the inode check in ``acquire`` makes that safe
    against a process that opened the old file just before it was deleted.
    """

    def __init__(self, lock_path, what: str = 'directory', *, remove: bool = False, log=None):
        self.path = os.path.abspath(lock_path)
        self.what = what
        self.remove = remove
        self.fd: Optional[int] = None
        self.log = log or (lambda message: print(message, file=sys.stderr, flush=True))

    def acquire(self) -> 'DirectoryLock':
        if self.fd is not None:
            return self
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        for _ in range(20):
            fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o644)
            if fcntl is None:
                break
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                if exc.errno in _UNLOCKABLE:
                    self.log(f'  WARNING: cannot lock {self.path} ({exc.strerror}); '
                             f'make sure no other process uses this {self.what}.')
                    break
                owner = self._owner(fd)
                os.close(fd)
                raise DirectoryBusy(
                    f'{self.what} {os.path.dirname(self.path)} is in use by another running '
                    f'process{owner}. Wait for it to finish (or stop it) and run again; two '
                    'runs must never share it.') from None
            try:
                same = os.fstat(fd).st_ino == os.stat(self.path).st_ino
            except FileNotFoundError:
                same = False
            if same:
                break
            os.close(fd)  # the previous owner removed this lock file: take the new one
        else:  # pragma: no cover - pathological churn
            raise DirectoryBusy(f'could not lock {self.path}')
        self.fd = fd
        info = json.dumps({'pid': os.getpid(), 'host': socket.gethostname(),
                           'started': time.strftime('%Y-%m-%dT%H:%M:%S')}).encode()
        try:
            os.ftruncate(fd, 0)
            os.pwrite(fd, info, 0)
        except OSError:
            pass
        return self

    @staticmethod
    def _owner(fd) -> str:
        try:
            info = json.loads(os.pread(fd, 4096, 0).decode() or '{}')
        except (OSError, ValueError):
            return ''
        if not isinstance(info, dict) or 'pid' not in info:
            return ''
        return f" (pid {info.get('pid')} on {info.get('host', '?')}, started {info.get('started', '?')})"

    def release(self) -> None:
        if self.fd is None:
            return
        fd, self.fd = self.fd, None
        if self.remove:
            try:
                os.remove(self.path)
            except OSError:
                pass
        try:
            if fcntl is not None:
                fcntl.flock(fd, fcntl.LOCK_UN)
        except OSError:
            pass
        os.close(fd)

    def __enter__(self):
        return self.acquire()

    def __exit__(self, *exc):
        self.release()
        return False
