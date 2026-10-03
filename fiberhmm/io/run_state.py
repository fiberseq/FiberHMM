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
    values recorded when it was hashed, *and* both timestamps were safely older
    than the moment hashing began (:data:`RACY_MARGIN_NS`, git's "racily
    clean" rule); otherwise the file is hashed again.

``st_ctime_ns`` is what makes a remembered digest safe against tools that
preserve metadata. The kernel sets it on every write, truncation, rename into
place and metadata change, including the ``utime`` call that restores a
modification time, and unprivileged tools cannot set it back. So a file whose
content changed while its size, mtime and inode were preserved is rehashed,
and a file whose metadata alone changed (a ``chmod``, a sync client's extended
attributes, a copy) is rehashed and then accepted, because its digest still
matches.

Timestamps are only as fine as the filesystem's clock, though: Linux stamps
files from a coarse clock that advances once per scheduler tick (1-10 ms), FAT
keeps 2-second modification times, and network shares and sync folders can be
coarser still. A rewrite that lands in the same tick as the write before it
leaves every stat field unchanged (observed on WSL2 ext4 for a rewrite made
immediately after hashing). Hence the racily-clean rule: a digest is
remembered only for a file whose mtime and ctime were at least
:data:`RACY_MARGIN_NS` older than the wall-clock time at which hashing began.
Any later write then carries a newer timestamp, so it changes the stat key. A
file hashed while it was still fresh is simply hashed again on the next
lookup, and remembered then. A timestamp in the future (clock skew, a
future-dated ``utime``) is never safely old, and a remembered digest is also
returned only while the file's timestamps are safely older than the current
time, so a file is rehashed after the clock was set back past it. A file
that changes while it is being hashed is read again (up to
:data:`STABLE_READ_ATTEMPTS` times) and never remembered. SHA-256 runs at roughly 2-3 GB/s, so a first run pays about one
extra read of its inputs and later runs pay nothing for unchanged files.
(This relies on POSIX ``st_ctime`` semantics -- macOS, Linux, WSL; on native
Windows it is the creation time, and the rule rests on the mtime alone. On a
network share whose server clock lags this machine's by more than the margin,
the rule is weakened by that lag.)

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
    """``[dev, inode, size, mtime_ns, ctime_ns]``: when all are unchanged, so is the content
    (provided the digest is not racily clean, see :func:`digest_is_trusted`)."""
    s = os.stat(path)
    return [s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns]


#: How much older than the start of hashing a file's mtime and ctime must be
#: for its digest to be remembered. Covers coarse kernel clocks (a few ms),
#: FAT's 2-second mtime granularity and small clock differences, with room to
#: spare.
RACY_MARGIN_NS = 3_000_000_000

_now_ns = time.time_ns  # wall clock (file timestamps are wall-clock times); tests patch it


def digest_is_trusted(key, hashed_at_ns) -> bool:
    """Whether a digest taken at ``hashed_at_ns`` (wall-clock ns, read before the file was
    opened) of a file whose :func:`stat_key` was ``key`` may be reused while the key is
    unchanged: only when both the mtime and the ctime were at least :data:`RACY_MARGIN_NS`
    older than ``hashed_at_ns``. Otherwise a later write in the same timestamp tick could
    leave the key unchanged ("racily clean")."""
    try:
        mtime_ns, ctime_ns = int(key[3]), int(key[4])
        hashed_at_ns = int(hashed_at_ns)
    except (TypeError, ValueError, IndexError):
        return False
    return max(mtime_ns, ctime_ns) + RACY_MARGIN_NS <= hashed_at_ns


def remembered_digest_is_valid(key, hashed_at_ns, now_ns) -> bool:
    """Whether a remembered digest whose recorded :func:`stat_key` equals the file's current
    ``key`` may be returned: it was not racily clean when taken, and the file's timestamps are
    not within the margin of (or after) the current time either -- after the clock was set
    back, a new write could otherwise reproduce the recorded timestamps."""
    return digest_is_trusted(key, hashed_at_ns) and digest_is_trusted(key, now_ns)


#: Reads of a file that changed while it was being hashed before its digest is returned as is.
STABLE_READ_ATTEMPTS = 3


class DigestMemo:
    """SHA-256 of files, remembered per path under the rule in the module docstring.

    ``entries`` is JSON-serialisable (``{real path: {"stat": [...], "sha256": ...,
    "hashed_at_ns": ...}}``) so a run can store it beside its state and pass it back on the
    next run. Entries without ``hashed_at_ns`` (written before the racily-clean rule) are
    not trusted: those files are hashed once more.
    """

    def __init__(self, entries: Optional[dict] = None):
        self.entries: dict = {}
        for path, entry in (entries or {}).items():
            if (isinstance(entry, dict) and isinstance(entry.get('stat'), list)
                    and isinstance(entry.get('sha256'), str)
                    and isinstance(entry.get('hashed_at_ns'), int)
                    and not isinstance(entry.get('hashed_at_ns'), bool)):
                self.entries[str(path)] = {'stat': list(entry['stat']), 'sha256': entry['sha256'],
                                           'hashed_at_ns': entry['hashed_at_ns']}

    def sha256(self, path) -> str:
        real = os.path.realpath(path)
        hashed_at = _now_ns()  # before the file is looked at: a write after this is detectable
        before = stat_key(real)
        entry = self.entries.get(real)
        if (entry is not None and entry['stat'] == before
                and remembered_digest_is_valid(before, entry.get('hashed_at_ns'), hashed_at)):
            return entry['sha256']
        for attempt in range(STABLE_READ_ATTEMPTS):
            if attempt:  # it changed while being read: read it again
                hashed_at = _now_ns()
                before = stat_key(real)
            digest = sha256_file(real)
            stable = stat_key(real) == before
            if stable:
                break
        # Remembered only if the file did not change while it was read and was not racily clean.
        if stable and digest_is_trusted(before, hashed_at):
            self.entries[real] = {'stat': before, 'sha256': digest, 'hashed_at_ns': hashed_at}
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
