"""Ordered, bounded parallel scoring on immutable native observations.

Only the independent recipient work is concurrent. Each simulation invocation
seeds its own thread-local Numba RNG; draw count/order inside an invocation is
unchanged. Completion order never controls seeds, output order, or assignment.
"""
from concurrent.futures import ThreadPoolExecutor, FIRST_COMPLETED, wait
from contextlib import contextmanager
from contextvars import ContextVar
from threading import Event


_active_pool = ContextVar('consensus_native_scoring_pool', default=None)


def _score_chunk(function, chunk, cancelled):
    result = []
    for item in chunk:
        if cancelled.is_set():
            raise RuntimeError('Native scoring batch cancelled')
        result.append(function(item))
    return result


class NativeScoringPool:
    def __init__(self, cores, *, backend='cpu', accelerator_bytes=256*1024**2, tilt=0.):
        if isinstance(cores, bool) or not isinstance(cores, int) or cores < 1:
            raise ValueError('Positive integer scoring worker budget required')
        self.cores = cores
        self.executor = None
        from .accelerated_predictive import resolve_device
        # 'vectorized' is a CPU kernel choice, not a device.
        self.kernel = 'vectorized' if str(backend).startswith('vectorized') else 'reference'
        self.tilt = tilt
        self.backend, reason = resolve_device('cpu' if self.kernel == 'vectorized' else backend)
        self.execution = dict(requested_backend=backend, resolved_backend=self.backend,
                              fallback_reason=reason,
                              rng='philox_counter_based' if self.kernel == 'vectorized' else 'reference_cpu_numba',
                              predictive_kernel=self.kernel, predictive_tilt=str(tilt),
                              unchanged_draw_counts=True)
        self.accelerator = None
        if self.backend != 'cpu':
            from .accelerated_predictive import PredictiveAccelerator
            self.accelerator = PredictiveAccelerator(self.backend, accelerator_bytes)
            self.execution['statistics'] = self.accelerator.stats

    def close(self):
        if self.executor is not None:
            self.executor.shutdown(wait=True)
            self.executor = None

    def map(self, function, items, *, maximum_bytes, bytes_per_item, progress=None, finish=None):
        values = list(items)
        capacity = max(1, min(self.cores, maximum_bytes//max(1, bytes_per_item)))
        def note(done):
            if progress is not None:
                progress(done, len(values))
        if self.accelerator is not None and finish is not None:
            # GPU work has one owner; do not contend for a shared context from
            # independent CPU workers or multiply device scratch by core count.
            result = []
            for i, value in enumerate(values):
                if i % 16 == 0:
                    note(i)
                prepared = function(value)
                result.append(self.accelerator.finish(prepared)
                    if '_native_predictive_request' in prepared else finish(prepared))
            note(len(values))
            return result
        if self.cores == 1 or capacity == 1 or len(values) < 16:
            result = []
            for i, value in enumerate(values):
                if i % 32 == 0:
                    note(i)
                prepared = function(value)
                result.append(finish(prepared) if finish is not None else prepared)
            note(len(values))
            return result
        if self.executor is None:
            self.executor = ThreadPoolExecutor(max_workers=self.cores,
                                               thread_name_prefix='native-score')
        if finish is not None:
            # Python/NumPy preparation stays on one thread. Only independent
            # compiled simulations run concurrently. Whole-call threading was
            # measured slower due to competing Python preparation work.
            return self._pipeline(function, finish, values, max(1, capacity-1), note)
        cancelled = Event()
        # Each pending chunk has at most eight results, not a copied observation
        # matrix. At most capacity chunks are submitted at once.
        chunks = [(i, values[i:i+8]) for i in range(0, len(values), 8)]
        remaining = iter(chunks); pending = {}; result = [None]*len(values)
        completed = 0
        def submit():
            chunk = next(remaining, None)
            if chunk is not None:
                start, items = chunk
                pending[self.executor.submit(_score_chunk, function, items, cancelled)] = start
        try:
            note(0)
            for _ in range(capacity):
                submit()
            while pending:
                done, _ = wait(pending, timeout=.2, return_when=FIRST_COMPLETED)
                for future in done:
                    start = pending.pop(future)
                    scored = future.result()
                    result[start:start+len(scored)] = scored
                    completed += len(scored)
                    submit()
                note(completed)
        except BaseException:
            cancelled.set()
            for future in pending:
                future.cancel()
            wait(pending)
            raise
        return result

    def _pipeline(self, prepare, finish, values, capacity, note):
        pending = {}; result = [None]*len(values); completed = 0
        try:
            note(0)
            for i, value in enumerate(values):
                # Keep one CPU for preparation; at most capacity simulations
                # are live. No all-recipient matrix/request list is retained.
                while len(pending) >= capacity:
                    done, _ = wait(pending, timeout=.2, return_when=FIRST_COMPLETED)
                    for future in done:
                        index = pending.pop(future)
                        result[index] = future.result()
                        completed += 1
                    note(completed)
                prepared = prepare(value)
                if '_native_predictive_request' not in prepared:
                    result[i] = finish(prepared)
                    completed += 1
                else:
                    pending[self.executor.submit(finish, prepared)] = i
                if i % 32 == 0:
                    note(completed)
            while pending:
                done, _ = wait(pending, timeout=.2, return_when=FIRST_COMPLETED)
                for future in done:
                    index = pending.pop(future)
                    result[index] = future.result()
                    completed += 1
                note(completed)
            note(completed)
        except BaseException:
            for future in pending:
                future.cancel()
            wait(pending)
            raise
        return result


@contextmanager
def native_scoring_pool(cores, *, backend='cpu', accelerator_bytes=256*1024**2, tilt=0.):
    pool = NativeScoringPool(cores, backend=backend, accelerator_bytes=accelerator_bytes, tilt=tilt)
    token = _active_pool.set(pool)
    from .measurement_distribution import _predictive_kernel
    kernel_token = _predictive_kernel.set((pool.kernel, pool.tilt))
    try:
        yield pool
    finally:
        _predictive_kernel.reset(kernel_token)
        _active_pool.reset(token)
        pool.close()


def score_native_recipients(function, items, **kwargs):
    pool = _active_pool.get()
    return (pool if pool is not None else NativeScoringPool(1)).map(function, items, **kwargs)


def defer_native_simulations():
    pool = _active_pool.get()
    return pool is not None and (pool.cores > 1 or pool.backend != 'cpu')


def scoring_execution_status():
    pool = _active_pool.get()
    return dict(pool.execution) if pool is not None else dict(resolved_backend='cpu')
