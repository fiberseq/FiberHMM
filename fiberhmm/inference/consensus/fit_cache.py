"""Persistent, content-addressed reuse of native boundary-distribution fits.

A fit is a deterministic function of its complete numerical inputs and of the
fitting code. The key is a digest of the source likelihood matrix, admission
mask, boundary cells, reference, iteration budget, optional initialization and
the exact excluded-fold row set, namespaced by the fitter source digest and the
numerical library versions. Only the optimizer output (five parameters and the
scalar diagnostics) is stored; the density, mass, center and covariance are
recomputed by the identical finalization code, so a cache hit is bit-identical
to a fresh fit. Different inputs never collide; a changed fitter or library
changes the namespace. Writes are atomic; a missing or unreadable entry is a
miss, never an error. This reuses work; it changes no model.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np

_NAMESPACE = None


def _namespace():
    """Fitter source and numerical-library identity; cached per process."""
    global _NAMESPACE
    if _NAMESPACE is None:
        import numpy, scipy
        parts = [Path(__file__).with_name('measurement_distribution.py').read_bytes(),
                 numpy.__version__.encode(), scipy.__version__.encode()]
        try:
            from threadpoolctl import threadpool_info
            blas = sorted((p.get('internal_api', ''), str(p.get('version', ''))) for p in threadpool_info()
                          if p.get('user_api') == 'blas')
            parts.append(json.dumps(blas).encode())
        except Exception:  # pragma: no cover - optional dependency
            parts.append(b'blas-unknown')
        digest = hashlib.sha256()
        for part in parts:
            digest.update(part); digest.update(b'\x1f')
        _NAMESPACE = digest.hexdigest()[:24]
    return _NAMESPACE


def _blob(array):
    a = np.ascontiguousarray(array)
    return a.dtype.str.encode() + b'|' + str(a.shape).encode() + b'|' + a.tobytes()


class NativeFitCache:
    def __init__(self, directory):
        self.root = Path(directory)/('native_fit_v1_'+_namespace())
        self.root.mkdir(parents=True, exist_ok=True)
        self.hits = self.misses = self.stores = 0

    def family_key(self, likelihood, allowed, coordinates, areas, reference, max_iterations, initial, backend='cpu'):
        digest = hashlib.sha256()
        if backend != 'cpu':
            # The dense reference and the separable objective are distinct
            # fitters: their results must never be exchanged through the cache.
            digest.update(('objective_backend=' + str(backend)).encode()); digest.update(b'\x1f')
        for value in (likelihood, allowed, coordinates, areas, np.asarray(reference)):
            digest.update(_blob(value)); digest.update(b'\x1f')
        digest.update(str(int(max_iterations)).encode()); digest.update(b'\x1f')
        digest.update(b'none' if initial is None else _blob(np.asarray(initial, float)))
        return digest.hexdigest()

    @staticmethod
    def job_key(family_key, rows):
        digest = hashlib.sha256(family_key.encode())
        digest.update(b'|full' if rows is None else b'|' + _blob(np.asarray(rows, np.int64)))
        return digest.hexdigest()

    def _path(self, key):
        return self.root/key[:2]/(key+'.json')

    def get(self, key):
        path = self._path(key)
        try:
            with open(path) as handle:
                entry = json.load(handle)
            if entry.get('schema') != 'fiberhmm.native_fit_cache.v1' or entry.get('key') != key:
                raise ValueError('foreign cache entry')
            self.hits += 1
            return entry
        except (OSError, ValueError, KeyError):
            self.misses += 1
            return None

    def put(self, key, parameters, objective, converged, iterations, message, source_units):
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        entry = dict(schema='fiberhmm.native_fit_cache.v1', key=key,
                     parameters=[float(v) for v in np.asarray(parameters, float)],
                     objective=float(objective), converged=bool(converged), iterations=int(iterations),
                     message=str(message), source_units=int(source_units))
        fd, temporary = tempfile.mkstemp(prefix='.'+path.name+'.', dir=path.parent)
        try:
            with os.fdopen(fd, 'w') as handle:
                json.dump(entry, handle, allow_nan=False)
            os.replace(temporary, path)
            self.stores += 1
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def statistics(self):
        return dict(directory=str(self.root), hits=self.hits, misses=self.misses, stores=self.stores)
