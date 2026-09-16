"""Bounded execution of independent native fits; no model changes.

Workers are spawned (never forked after OpenMP initialization), single-threaded,
and reused for an entire workflow. Large read-only family inputs are mapped
once per task instead of pickled for every excluded fold. Results are returned
in the original full/fold order, regardless of completion order.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from concurrent.futures import FIRST_COMPLETED, wait
from pathlib import Path
import tempfile

import numpy as np


_active_pool = ContextVar('consensus_native_fit_pool', default=None)
_default_cache_dir = ContextVar('consensus_native_fit_cache_dir', default='')
_default_fit_backend = ContextVar('consensus_native_fit_backend', default='cpu')
_default_fit_smoothing = ContextVar('consensus_native_fit_smoothing', default=4.)


def set_default_fit_backend(backend):
    _default_fit_backend.set(str(backend or 'cpu'))


def set_default_fit_smoothing(pseudo_units):
    """Pseudo-units of the nonparametric geometry's smoothing toward its Gaussian seed."""
    _default_fit_smoothing.set(float(4. if pseudo_units is None else pseudo_units))


def set_default_fit_cache(directory):
    """Process-wide default for standalone pools (for example family workers)."""
    _default_cache_dir.set(str(directory or ''))
_CHILD_THREADS = dict(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
                     MKL_NUM_THREADS='1', NUMBA_NUM_THREADS='1',
                     VECLIB_MAXIMUM_THREADS='1', NUMEXPR_NUM_THREADS='1',
                     BLIS_NUM_THREADS='1')


def _fit_task(directory, rows, reference, max_iterations, objective_backend='cpu', smoothing_pseudo_units=4.):
    """Import-safe worker entry point; initialization does not use a full fit."""
    from .execution import task_thread_budget
    from .measurement_distribution import fit_native_distribution
    task_thread_budget(1)
    root = Path(directory)
    def checkpoint(_message):
        if (root/'cancel').exists():
            raise RuntimeError('Native fit batch cancelled')
    checkpoint(None)
    arrays = [np.load(root/(name+'.npy'), mmap_mode='r', allow_pickle=False)
              for name in ('likelihood', 'allowed', 'coordinates', 'areas')]
    ll, allowed, coordinates, areas = arrays
    if rows is not None:
        ll, allowed = ll[rows], allowed[rows]
    return fit_native_distribution(ll, allowed, coordinates, areas,
        reference=reference, max_iterations=max_iterations, progress=checkpoint, objective_backend=objective_backend,
        smoothing_pseudo_units=smoothing_pseudo_units)


def _parallel_capacity(shape, jobs, cores, maximum_bytes, resident_bytes=0):
    """Conservative matrix/scratch budget, not a source/projection count cap.

    Reserve the caller's materialized matrices plus 96 bytes per source/cell
    per child for input copies, scaled matrices and stable-fallback temporaries.
    A further 96 MiB per child covers numerical imports/small work arrays. If
    parallel scratch will not fit, use the unchanged serial implementation.
    This is a matrix-working-set budget, not a whole-server RSS guarantee.
    """
    cells = int(shape[0])*int(shape[1])
    per_child = max(1, cells*96)
    available = maximum_bytes-resident_bytes-cores*96*1024**2
    return max(1, min(cores, jobs, max(0, available)//per_child))


class NativeFitPool:
    def __init__(self, cores, maximum_bytes, *, isolate_blas=False, backend='cpu',
                 accelerator_bytes=256*1024**2, cache_dir=None, smoothing_pseudo_units=None):
        if isinstance(cores, bool) or not isinstance(cores, int) or cores < 1:
            raise ValueError('Positive integer native worker budget required')
        if maximum_bytes < 1:
            raise ValueError('Positive native worker memory budget required')
        self.cores = cores
        self.maximum_bytes = maximum_bytes
        self.cache = None
        directory = _default_cache_dir.get() if cache_dir is None else cache_dir
        if directory:
            from .fit_cache import NativeFitCache
            self.cache = NativeFitCache(directory)
        self.executor = None
        self.executor_workers = 0
        from .accelerated_predictive import resolve_device
        # 'separable' is a CPU objective, not a device: same optimizer and data,
        # a different (declared, versioned) summation of the same terms.
        self.objective_backend = backend if backend in ('separable', 'nonparametric') else 'cpu'
        resolved, reason = resolve_device('cpu' if backend in ('separable', 'nonparametric') else backend)
        if resolved == 'mps':
            resolved, reason = 'cpu', 'MPS has no float64 native-fit objective; predictive scoring is supported separately'
        if backend == 'auto':
            resolved, reason = 'cpu', 'CUDA fitting remains validation-only after a real-locus optimizer divergence'
        self.backend = resolved
        self.accelerator_bytes = accelerator_bytes
        self.execution = dict(requested_backend=backend, resolved_backend=resolved, fallback_reason=reason,
                              objective_backend=self.objective_backend,
                              cuda_fit_calls=0, cpu_budget_fallbacks=0,
                              cuda_validation_only=resolved == 'cuda',
                              authoritative_models='cpu_reference', cuda_distribution_disagreements=0)
        self.smoothing_pseudo_units = float(_default_fit_smoothing.get() if smoothing_pseudo_units is None
                                            else smoothing_pseudo_units)
        if self.objective_backend == 'nonparametric':
            self.execution['smoothing_pseudo_units'] = self.smoothing_pseudo_units
        # Never change process-global BLAS settings in a shared browser. If
        # its existing BLAS pool is multithreaded, even a memory-limited serial
        # fit must run in a single-threaded child. Standalone unscoped calls
        # retain their historical execution behavior.
        self.isolate_serial = False
        if isolate_blas:
            from threadpoolctl import threadpool_info
            self.isolate_serial = any(p.get('user_api') == 'blas' and p['num_threads'] > 1
                                      for p in threadpool_info())

    def close(self):
        if self.executor is not None:
            if not getattr(self, '_borrowed', False):
                self.executor.shutdown(wait=True)
            self.executor = None
            self._borrowed = False
            self.executor_workers = 0

    def fit(self, likelihood, allowed, coordinates, areas, *, row_sets,
            reference, max_iterations, progress=None, resident_bytes=0):
        """Fit full and excluded-fold models with identical initialization.

        Cancellation is checked by the parent every 0.2 s, and by children at
        each optimizer iteration. No progress callback crosses process bounds.
        On cancellation/error, drain this batch before removing its mappings.
        """
        from .measurement_distribution import fit_native_distribution
        jobs = list(row_sets.items())
        cached = {}
        if self.cache is not None and self.backend != 'cuda' and self.objective_backend != 'nonparametric':
            from .measurement_distribution import finalize_native_distribution
            family_key = self.cache.family_key(likelihood, allowed, coordinates, areas, reference, max_iterations, None, backend=self.objective_backend)
            for key, rows in jobs:
                entry = self.cache.get(self.cache.job_key(family_key, rows))
                if entry is not None:
                    cached[key] = finalize_native_distribution(entry['parameters'], coordinates, areas, reference,
                        objective=entry['objective'], converged=entry['converged'], iterations=entry['iterations'],
                        message=entry['message'], source_units=entry['source_units'])
            jobs = [(key, rows) for key, rows in jobs if key not in cached]
            if not jobs:
                return {key: cached[key] for key in row_sets}
            self.execution['fit_cache'] = self.cache.statistics()
        shared = sum(a.nbytes for a in (likelihood, allowed, coordinates, areas))
        capacity = _parallel_capacity(likelihood.shape, len(jobs), self.cores,
                                     self.maximum_bytes, resident_bytes+shared)
        def note(message):
            if progress is not None:
                progress(message)
        if self.backend == 'cuda':
            # Real-locus validation found a nearly flat optimum: a tiny change
            # in reduction order selected a very different geometry mass with
            # almost identical likelihood. Until that is resolved, CUDA fit
            # selection is explicitly a validation run, NOT an acceleration.
            # Always return the ordinary CPU fits, even if the GPU loss agrees.
            reference_pool = NativeFitPool(self.cores, self.maximum_bytes,
                                           isolate_blas=self.isolate_serial)
            try:
                result = reference_pool.fit(likelihood, allowed, coordinates, areas,
                    row_sets=row_sets, reference=reference, max_iterations=max_iterations,
                    progress=progress, resident_bytes=resident_bytes)
            finally:
                reference_pool.close()
            for key, rows in jobs:
                note(f'validating CUDA full/held-out model {key}; CPU remains authoritative')
                ll, mask = (likelihood, allowed) if rows is None else (likelihood[rows], allowed[rows])
                try:
                    alternative = fit_native_distribution(ll, mask, coordinates, areas,
                        reference=reference, max_iterations=max_iterations, progress=note,
                        objective_backend='cuda', accelerator_bytes=self.accelerator_bytes)
                    self.execution['cuda_fit_calls'] += 1
                    difference = float(np.abs(np.exp(result[key]['log_mass'])-
                                              np.exp(alternative['log_mass'])).sum()/2)
                    self.execution['maximum_geometry_mass_total_variation'] = max(difference,
                        self.execution.get('maximum_geometry_mass_total_variation', 0.))
                    self.execution['cuda_distribution_disagreements'] += int(difference > 1e-6)
                except MemoryError:
                    self.execution['cpu_budget_fallbacks'] += 1
                except RuntimeError as exc:
                    self.execution['cuda_validation_error'] = str(exc)
            return result
        if capacity == 1 and not self.isolate_serial:
            result = {}
            for key, rows in jobs:
                name = 'full model' if key == 'full' else f'held-out model {key+1}/{len(row_sets)-1}'
                note(f'fitting {name}')
                ll, mask = (likelihood, allowed) if rows is None else (likelihood[rows], allowed[rows])
                result[key] = fit_native_distribution(ll, mask, coordinates, areas,
                    reference=reference, max_iterations=max_iterations,
                    progress=lambda msg: note(f'{name}: {msg}'), objective_backend=self.objective_backend,
                    smoothing_pseudo_units=self.smoothing_pseudo_units)
            return self._store(result, cached, row_sets, likelihood, allowed, coordinates, areas, reference, max_iterations)
        if self.executor is None or capacity > self.executor_workers:
            # Do not eagerly spawn the requested CPU count when the matrix
            # budget admits only one or two fits. Grow only between batches,
            # after all previous tasks/mappings have been drained.
            self.close()
            from .execution import _shared_pool
            shared = _shared_pool.get()
            if shared is not None and shared.cores >= capacity:
                # Borrow the run's pool: one spawn per run for every stage.
                self.executor = shared.get(); self._borrowed = True
            else:
                from joblib.externals.loky import ProcessPoolExecutor
                from joblib.externals.loky.backend.context import get_context
                self.executor = ProcessPoolExecutor(max_workers=capacity, timeout=None,
                    context=get_context('loky'), env=_CHILD_THREADS)
            self.executor_workers = capacity
        note(f'fitting {len(jobs)} independent full/held-out models; {capacity} concurrent workers')
        with tempfile.TemporaryDirectory(prefix='fiberhmm-native-fits-') as directory:
            root = Path(directory)
            for name, value in zip(('likelihood', 'allowed', 'coordinates', 'areas'),
                                   (likelihood, allowed, coordinates, areas)):
                np.save(root/(name+'.npy'), value, allow_pickle=False)
            pending = {}; result = {}; remaining = iter(jobs)
            def submit():
                item = next(remaining, None)
                if item is not None:
                    key, rows = item
                    pending[self.executor.submit(_fit_task, directory, rows, reference, max_iterations,
                                                  self.objective_backend, self.smoothing_pseudo_units)] = key
            try:
                for _ in range(capacity):
                    submit()
                while pending:
                    done, _ = wait(pending, timeout=.2, return_when=FIRST_COMPLETED)
                    for future in done:
                        key = pending.pop(future)
                        result[key] = future.result()
                        submit()
                    note(f'independent full/held-out models: {len(result)}/{len(jobs)} completed; '
                         f'{len(pending)} running or queued (budget {capacity})')
            except BaseException:
                (root/'cancel').touch()
                for future in pending:
                    future.cancel()
                wait(pending)
                raise
        return self._store(result, cached, row_sets, likelihood, allowed, coordinates, areas, reference, max_iterations)

    def _store(self, result, cached, row_sets, likelihood, allowed, coordinates, areas, reference, max_iterations):
        """Persist fresh fits (parameters and diagnostics only) and merge cached ones in original order."""
        if self.cache is not None and self.backend != 'cuda' and self.objective_backend != 'nonparametric' and result:
            family_key = self.cache.family_key(likelihood, allowed, coordinates, areas, reference, max_iterations, None, backend=self.objective_backend)
            for key, fit in result.items():
                self.cache.put(self.cache.job_key(family_key, row_sets[key]), fit['parameters'], fit['objective'],
                               fit['converged'], fit['iterations'], fit['message'], fit['source_units'])
            self.execution['fit_cache'] = self.cache.statistics()
        merged = dict(result); merged.update(cached)
        return {key: merged[key] for key in row_sets}


@contextmanager
def native_fit_pool(cores, maximum_bytes, *, backend='cpu', accelerator_bytes=256*1024**2, cache_dir='',
                    smoothing_pseudo_units=4.):
    """One owner per workflow; unrelated browser jobs never share mutable state."""
    pool = NativeFitPool(cores, maximum_bytes, isolate_blas=True, backend=backend,
                         accelerator_bytes=accelerator_bytes, cache_dir=cache_dir or '',
                         smoothing_pseudo_units=smoothing_pseudo_units)
    token = _active_pool.set(pool)
    default = _default_cache_dir.set(str(cache_dir or ''))
    backend_token = _default_fit_backend.set(pool.objective_backend)
    smoothing_token = _default_fit_smoothing.set(pool.smoothing_pseudo_units)
    try:
        yield pool
    finally:
        _default_fit_smoothing.reset(smoothing_token)
        _default_fit_backend.reset(backend_token)
        _default_cache_dir.reset(default)
        _active_pool.reset(token)
        pool.close()


def fit_native_models(likelihood, allowed, coordinates, areas, **kwargs):
    pool = _active_pool.get()
    if pool is None:
        pool = NativeFitPool(1, 1, backend=_default_fit_backend.get())  # serial; follows the run's declared backend
    return pool.fit(likelihood, allowed, coordinates, areas, **kwargs)


def fit_execution_status():
    pool = _active_pool.get()
    return dict(pool.execution) if pool is not None else dict(resolved_backend='cpu')
