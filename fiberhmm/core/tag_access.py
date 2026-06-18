"""Shared helpers for BAM auxiliary tag access."""

from __future__ import annotations

import numpy as np


def get_preferred_tag(read, primary: str, fallback: str, default=None,
                      errors=(KeyError,)):
    """Return ``primary`` tag if present, otherwise ``fallback`` tag."""
    try:
        if read.has_tag(primary):
            return read.get_tag(primary)
        if read.has_tag(fallback):
            return read.get_tag(fallback)
    except errors:
        return default
    return default


def compact_ml_value(value):
    """Convert ML-like byte arrays to bytes without materializing Python ints."""
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value)
    try:
        return np.asarray(value, dtype=np.uint8).reshape(-1).tobytes()
    except (TypeError, ValueError):
        return value
