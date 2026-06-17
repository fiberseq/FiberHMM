"""Small shared helpers for BAM tag mutation."""

from __future__ import annotations

from typing import Sequence


def clear_tags(read, tags: Sequence[str]) -> None:
    """Remove BAM tags when present, tolerating pysam-compatible failures."""
    for tag in tags:
        if read.has_tag(tag):
            try:
                read.set_tag(tag, None)
            except Exception:
                pass
