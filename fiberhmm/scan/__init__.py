"""``fiberhmm-scan``: pooled feature scans over fixed units, written as one versioned scan directory.

    from fiberhmm.scan import plan, run_units, open_scan
    units = plan(reference, genome=True, contigs=['chr3'])
    run_units(datasets, units, modules=['fp', 'nfr'], out='scan_dir')
    scan = open_scan('scan_dir')
    scan.hits('fp', 'chr3', 0, 1_000_000)
    state_at(read, 'nfr.region', 'chr3', 1_000, 1_400)      # '1' open, '0' closed, '.' not callable

Format reference: ``docs/reference/scan-format.md``.
"""
from __future__ import annotations

from .types import SCHEMA


def plan(*args, **kwargs):
    from .units import plan as _plan
    return _plan(*args, **kwargs)


def run_units(*args, **kwargs):
    from .runner import run_units as _run
    return _run(*args, **kwargs)


def open_scan(path):
    from .reader import ScanDir
    return ScanDir(path)


def state_at(*args, **kwargs):
    """The state of a transferable feature type (``nfr.region``) on one record; see :mod:`fiberhmm.scan.states`."""
    from .states import state_at as _state_at
    return _state_at(*args, **kwargs)


__all__ = ['SCHEMA', 'plan', 'run_units', 'open_scan', 'state_at']
