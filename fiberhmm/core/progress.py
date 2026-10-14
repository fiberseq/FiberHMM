"""Progress lines that rewrite in place on a terminal and stay readable in logs.

On a TTY a progress update starts with ``\\r`` and no newline, so the next
update overwrites it. Redirected to a file or pipe the same update is written
as its own newline-terminated line, so logs do not run updates together.
"""
from __future__ import annotations

import sys


def _is_tty(stream) -> bool:
    try:
        return bool(stream.isatty())
    except (AttributeError, ValueError, OSError):
        return False


def progress(text: str, file=None, final: bool = False) -> None:
    """Write one progress update (``final=True`` ends the in-place line)."""
    stream = sys.stdout if file is None else file
    if _is_tty(stream):
        print("\r" + text, end="\n" if final else "", file=stream)
    else:
        print(text, file=stream)
    try:
        stream.flush()
    except (AttributeError, ValueError, OSError):
        pass


def progress_end(file=None) -> None:
    """Terminate an in-place progress line (nothing to do off a terminal)."""
    stream = sys.stdout if file is None else file
    if _is_tty(stream):
        print(file=stream)
