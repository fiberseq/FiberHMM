"""Small shared helpers for BAM tag mutation."""

from __future__ import annotations

from typing import Sequence


def _clear_tag(read, tag: str) -> None:
    if not read.has_tag(tag):
        return
    try:
        read.set_tag(tag, None)
    except Exception:
        pass


def clear_tags(read, tags: Sequence[str]) -> None:
    """Remove BAM tags when present, tolerating pysam-compatible failures."""
    for tag in tags:
        _clear_tag(read, tag)
