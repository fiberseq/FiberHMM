"""Bounded, invocation-local reuse of immutable cross-family preparation.

No observations, geometries or candidate pairs are omitted on eviction. These
caches do not persist across a run, dataset edit, model refit or parameter edit.
Random experiments are deliberately NOT cached by an incomplete lattice key.
"""
from collections import OrderedDict

import numpy as np


class ByteLRU:
    def __init__(self, maximum_bytes):
        self.maximum_bytes = max(0, int(maximum_bytes))
        self.entries = OrderedDict()
        self.bytes = 0
        self.hits = self.misses = self.evictions = 0

    def get(self, key):
        item = self.entries.get(key)
        if item is None:
            self.misses += 1
            return None
        self.entries.move_to_end(key)
        self.hits += 1
        return item[0]

    def put(self, key, value, size):
        size = int(size)
        if size > self.maximum_bytes:
            return value
        old = self.entries.pop(key, None)
        if old is not None:
            self.bytes -= old[1]
        while self.entries and self.bytes+size > self.maximum_bytes:
            _, (_, removed) = self.entries.popitem(last=False)
            self.bytes -= removed
            self.evictions += 1
        self.entries[key] = (value, size)
        self.bytes += size
        return value


class NativeReadCache(ByteLRU):
    def _entry(self, unit):
        from .measurement_family import _native_observation_arrays
        item = self.get(id(unit))
        if item is not None:
            # Retaining the input identity also prevents Python ID reuse.
            if item[0] is not unit:
                raise ValueError('Native read cache identity mismatch')
            return item
        # Own the numerical snapshot. Do not make arrays owned by the caller
        # read-only as a side effect, or allow mutation of a returned cache view.
        arrays = tuple(a.copy() for a in _native_observation_arrays(unit))
        spans = list(unit['representative_raw_tf_intervals'])+list(unit.get('raw_nuc_intervals', []))
        starts = np.sort(np.asarray([x for x, _ in spans], np.int64))
        ends = np.sort(np.asarray([y for _, y in spans], np.int64))
        occupied_prefix = np.r_[0, np.cumsum(arrays[2])]
        for a in (*arrays, starts, ends, occupied_prefix):
            a.setflags(write=False)
        entry = (unit, arrays, starts, ends, occupied_prefix)
        self.put(id(unit), entry, sum(a.nbytes for a in arrays)
                 + starts.nbytes+ends.nbytes+occupied_prefix.nbytes+512)
        return entry

    def native(self, unit):
        return self._entry(unit)[1]

    def limits(self, unit, call, domain):
        _, _, starts, ends, _ = self._entry(unit)
        before = np.searchsorted(ends, call['start'], side='right')-1
        after = np.searchsorted(starts, call['end'], side='left')
        return (max(domain[0], int(ends[before])) if before >= 0 else domain[0],
                min(domain[1], int(starts[after])) if after < len(starts) else domain[1])

    def visible_count(self, unit, call, first, last):
        _, native, _, _, occupied = self._entry(unit)
        p = native[0]
        a = int(np.searchsorted(p, first))
        b = int(np.searchsorted(p, last, side='right'))
        ca = max(a, min(b, int(np.searchsorted(p, call['start']))))
        cb = max(a, min(b, int(np.searchsorted(p, call['end']))))
        return int(b-a-(occupied[b]-occupied[a])+(occupied[cb]-occupied[ca]))


class TransferGeometryCache(ByteLRU):
    def __init__(self, frozen, grid, floor_bp, maximum_bytes):
        super().__init__(maximum_bytes)
        self.frozen, self.grid, self.floor_bp = frozen, grid, floor_bp

    def validate(self, frozen, grid, floor_bp):
        if frozen is not self.frozen or grid is not self.grid or floor_bp != self.floor_bp:
            raise ValueError('Transfer cache cannot be reused across models, grids or edge allowances')
