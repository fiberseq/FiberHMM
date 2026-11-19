"""Lazy assembly of built-in nuc stitch tracks from immutable unit archives.

Only requested tracks or slices are expanded. Geometry is still decided by
territory_runs; other modules retain the ordinary ndarray protocol.
"""
from __future__ import annotations

import bisect
from functools import lru_cache

import numpy as np


@lru_cache(maxsize=64)
def _read_track(path, name, signature):
    # Unit archives are immutable while finalizing. The signature separates
    # rewrites in later continued scans; the cache retains at most 64 arrays.
    with np.load(path) as archive:
        return archive[name]


class _RunTrack:
    def __init__(self, name, start, end, pieces):
        self.name, self.start = name, start
        self.shape, self.dtype = (end-start,), np.dtype('float64')
        self.pieces = [(a, b, path, (path.stat().st_ino, path.stat().st_size, path.stat().st_mtime_ns))
                       for a, b, path in pieces]
        self.ends = [b-start for a, b, path in pieces]

    def __len__(self):
        return self.shape[0]

    def __array__(self, dtype=None, copy=None):
        return np.asarray(self[:], dtype=dtype)

    def __getitem__(self, selection):
        if not isinstance(selection, slice) or selection.step not in (None, 1):
            return self[:][selection]
        a, b, _step = selection.indices(len(self))
        b = max(a, b)
        out = np.zeros(b-a, dtype=self.dtype)
        i = bisect.bisect_right(self.ends, a)
        while i < len(self.pieces):
            start, end, path, signature = self.pieces[i]
            start, end = start-self.start, end-self.start
            if start >= b:
                break
            lo, hi = max(a, start), min(b, end)
            if hi > lo:
                source = _read_track(path, self.name, signature)
                out[lo-a:hi-a] = source[lo-start:hi-start]
            i += 1
        return out


def stitch_inputs(out, name, mod, settings, units, status, names=None, results=None):
    from .finalize import _read_json, _unit_dir
    from .types import unit_result_from_json
    from .seams import territory_runs
    zones, sources = [], []
    for unit in units:
        state = status[unit.unit_id]
        data = None
        if state == 'done':
            result = results.get(unit.unit_id) if results is not None else None
            if result is None:
                result = unit_result_from_json(_read_json(_unit_dir(out, name, unit) / 'result.json.gz'))
            path = _unit_dir(out, name, unit) / 'tracks.npz'
            if result.status == 'ok' and path.is_file():
                with np.load(path) as archive:
                    keys = set(archive.files) if names is None else set(archive.files) & set(names)
                sources.append((unit, path, keys))
                data = {}  # only geometry; no dense unit arrays retained
        zones.append((unit, data, 'ok' if data is not None else 'empty' if state in ('done', 'empty') else 'missing'))
    runs = territory_runs(zones, mod.stitch_support_bp(settings))
    territory = [(a, b, left, right) for a, b, _tracks, left, right in runs]
    keys = sorted({key for _unit, _path, keys in sources for key in keys})
    tracks = {}
    for key in keys:
        entries = []
        for a, b, _left, _right in territory:
            pieces = [(u.zone_start, u.zone_end, path) for u, path, keys in sources
                      if key in keys and u.zone_start < b and u.zone_end > a]
            entries.append(_RunTrack(key, a, b, pieces) if pieces else None)
        tracks[key] = entries
    return territory, tracks
