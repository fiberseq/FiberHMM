"""``fiberhmm-scan``: pooled feature scans over fixed units, written as one versioned scan directory.

    from fiberhmm.scan import plan, run_units, open_scan
    units = plan(reference, genome=True, contigs=['chr3'])
    run_units(datasets, units, modules=['fp', 'nfr'], out='scan_dir')
    scan = open_scan('scan_dir')
    scan.hits('fp', 'chr3', 0, 1_000_000)

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


__all__ = ['SCHEMA', 'plan', 'run_units', 'open_scan']
