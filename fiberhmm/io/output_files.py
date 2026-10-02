"""Temporary files for outputs that are published by renaming.

``tempfile.mkstemp`` creates its file with mode 0600 whatever the umask, and
``os.replace`` keeps that mode, so an output published that way is readable
by its owner only: lab members sharing a results directory cannot open it.
:func:`mkstemp_shared` is a drop-in replacement that creates the file the way
``open()`` does (requested mode 0666, reduced by the process umask), so a
published output gets the same permissions as any other file the user writes.
The umask is applied by the kernel; it is never read or changed here (that
would be process-global and racy in threaded code).
"""
from __future__ import annotations

import errno
import os
import secrets

_FLAGS = os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_BINARY", 0)
_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_ATTEMPTS = 10_000


def mkstemp_shared(suffix: str = "", prefix: str = "tmp", dir=None) -> tuple[int, str]:
    """Like :func:`tempfile.mkstemp` (same arguments and ``(fd, path)``
    result), but the file is created with mode ``0666 & ~umask``.

    The name is unique and the file is created exclusively (``O_EXCL``), so a
    concurrent writer or an existing file is never reused.
    """
    directory = os.path.abspath(dir if dir is not None else os.getcwd())
    for _attempt in range(_ATTEMPTS):
        path = os.path.join(directory, f"{prefix}{secrets.token_hex(6)}{suffix}")
        try:
            descriptor = os.open(path, _FLAGS | _NOFOLLOW, 0o666)
        except FileExistsError:
            continue
        except PermissionError:
            # Windows reports a name held by a directory as EACCES.
            if os.name == "nt" and os.path.isdir(directory) and os.access(directory, os.W_OK):
                continue
            raise
        return descriptor, path
    raise FileExistsError(errno.EEXIST, "no usable temporary file name found", directory)
