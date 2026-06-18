"""Small filesystem status predicates."""

from __future__ import annotations

import os
from typing import Union

PathInput = Union[str, os.PathLike]


def path_is_regular_file(path: PathInput) -> bool:
    """Return True only for regular filesystem files."""
    try:
        return os.path.isfile(path)
    except OSError:
        return False


def path_is_nonempty_file(path: PathInput) -> bool:
    """Return True only for regular files with at least one byte."""
    try:
        return path_is_regular_file(path) and os.path.getsize(path) > 0
    except OSError:
        return False
