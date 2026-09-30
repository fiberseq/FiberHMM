"""Identity of the running FiberHMM code and of the tables it reads.

Producers record these in the ``FIBERHMM-CHEMISTRY`` declaration so a BAM says
exactly which code and emission tables made its calls, and
:mod:`fiberhmm.advisories` can later tell whether it needs re-running.

The commit comes from, in order:

1. a git checkout the package is imported from (``.git`` next to the
   ``fiberhmm`` package directory), with ``+dirty`` when files under
   ``fiberhmm/`` differ from that commit (changed or new, non-ignored files);
2. ``fiberhmm/_build_info.py``, written by ``setup.py`` when a wheel is built
   from a git checkout;
3. nothing (sdists, copied or frozen source trees): the commit is omitted,
   never guessed.
"""
from __future__ import annotations

import functools
import hashlib
import os
import subprocess
from pathlib import Path

_PACKAGE_DIR = Path(__file__).resolve().parent
_GIT_TIMEOUT_SECONDS = 5


def fiberhmm_version() -> str:
    from fiberhmm import __version__

    return str(__version__)


def _git(root: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), *args],
            capture_output=True, text=True, timeout=_GIT_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout if result.returncode == 0 else None


def git_commit_of_checkout(root: Path) -> str | None:
    """``<sha>`` or ``<sha>+dirty`` for the git checkout at ``root``, else None.

    Only ``root/.git`` (a directory, or a worktree's ``gitdir:`` file) counts,
    so an installed package inside some unrelated repository is never
    attributed to that repository's commit. Dirtiness is judged on
    ``fiberhmm/`` only (bounded, and the only files that decide results):
    changed tracked files and new, non-ignored files both count.
    """
    if not (root / ".git").exists():
        return None
    head = _git(root, "rev-parse", "HEAD")
    commit = (head or "").strip().lower()
    if len(commit) != 40 or any(c not in "0123456789abcdef" for c in commit):
        return None
    status = _git(root, "status", "--porcelain", "--untracked-files=normal", "--", "fiberhmm")
    if status is None:
        return commit
    return commit + ("+dirty" if status.strip() else "")


@functools.lru_cache(maxsize=1)
def fiberhmm_commit() -> str | None:
    """Commit of the running code (see module docstring), or None."""
    commit = git_commit_of_checkout(_PACKAGE_DIR.parent)
    if commit:
        return commit
    try:
        from fiberhmm._build_info import COMMIT  # type: ignore[import-not-found]
    except ImportError:
        return None
    commit = str(COMMIT or "").strip().lower()
    return commit or None


_SHA_CACHE: dict[tuple[str, int, int], str] = {}


def file_sha256(path) -> str | None:
    """sha256 of a file's bytes (cached by path, size and mtime), or None."""
    if not path:
        return None
    try:
        resolved = os.path.realpath(os.fspath(path))
        stat = os.stat(resolved)
    except (OSError, TypeError):
        return None
    key = (resolved, stat.st_size, stat.st_mtime_ns)
    cached = _SHA_CACHE.get(key)
    if cached is not None:
        return cached
    digest = hashlib.sha256()
    try:
        with open(resolved, "rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
    except OSError:
        return None
    _SHA_CACHE[key] = digest.hexdigest()
    return _SHA_CACHE[key]
