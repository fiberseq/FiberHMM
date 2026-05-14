"""Shared multiprocessing context selection for inference worker pools."""

import multiprocessing
import os
import sys


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
def _select_mp_context() -> 'multiprocessing.context.BaseContext':
    override = os.environ.get('FIBERHMM_MP_CONTEXT', '').strip().lower()
    if override in ('spawn', 'fork', 'forkserver'):
        return multiprocessing.get_context(override)
    if sys.version_info >= (3, 14):
        return multiprocessing.get_context('spawn')
    if sys.platform == 'darwin':
        return multiprocessing.get_context('fork')
    # Linux, Python <3.14: fork is much faster and segfaults are rare.
    return multiprocessing.get_context('fork')


_MP_CONTEXT = _select_mp_context()
