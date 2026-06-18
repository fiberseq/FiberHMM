"""Shared multiprocessing context selection for inference worker pools."""

import multiprocessing
import os
import sys
from typing import Optional


_VALID_MP_CONTEXTS = ('spawn', 'fork', 'forkserver')


# Multiprocessing start method:
#
#   spawn (default):  workers re-import everything from scratch. Safe -- avoids
#                     pysam/htslib segfaults that fire reliably under Python
#                     >=3.14 and occasionally on earlier versions, and avoids
#                     fiberhmm-run pipe-EOF deadlocks (forked workers inherit
#                     the stdout pipe FD and prevent EOF reaching downstream
#                     stages).
#   fork:             workers inherit the parent's already-imported modules,
#                     JIT-compiled numba functions, loaded model state, and
#                     htslib decompression buffers -- significantly faster
#                     ramp-up and steady-state on long jobs (10x observed on
#                     360 GB Hia5 PacBio). But triggers segfaults on some
#                     datasets/Python versions and is incompatible with
#                     streaming-pipeline EOF.
#
# Auto-default: spawn on Python >=3.14 (fork unsafe), fork on Python <3.14
# Linux (fast, fork-safe enough for most data), fork on macOS (spawn cost
# is high, fork is the macOS default). Override via env var
# FIBERHMM_MP_CONTEXT=spawn|fork. Set =spawn explicitly if you hit a worker
# segfault during processing.
def _normalize_mp_context_override(value: str) -> Optional[str]:
    override = value.strip().lower()
    return override if override in _VALID_MP_CONTEXTS else None


def _default_mp_context_name(version_info=None) -> str:
    version_info = sys.version_info if version_info is None else version_info
    if version_info >= (3, 14):
        return 'spawn'
    # Python <3.14: fork is much faster and segfaults are rare.
    return 'fork'


def _select_mp_context() -> 'multiprocessing.context.BaseContext':
    override = _normalize_mp_context_override(
        os.environ.get('FIBERHMM_MP_CONTEXT', ''),
    )
    context_name = override or _default_mp_context_name()
    return multiprocessing.get_context(context_name)


_MP_CONTEXT = _select_mp_context()
