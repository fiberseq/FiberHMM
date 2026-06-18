"""Small filesystem status predicates."""

from __future__ import annotations

import os
from typing import Union

PathInput = Union[str, os.PathLike]


def path_is_nonempty_file(path: PathInput) -> bool:
    """Return True only for regular files with at least one byte."""
    try:
        return os.path.isfile(path) and os.path.getsize(path) > 0
    except OSError:
        return False
