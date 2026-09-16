"""Scoped numerical execution controls; never scientific/model parameters."""
from contextlib import contextmanager
from contextvars import ContextVar
import os
import warnings


@contextmanager
def numerical_thread_budget(cores):
    """Honor the worker's requested Numba budget, restoring it even on failure.

    Numba's mask is thread-local. Set it inside the actual workflow worker,
    including the native engine (which bypasses the legacy workflow setup).
    Do not alter process-global BLAS settings in a concurrent browser server.
    """
    from numba import config, get_num_threads, set_num_threads
    previous = get_num_threads()
    set_num_threads(min(cores, config.NUMBA_NUM_THREADS))
    try:
        yield
    finally:
        set_num_threads(previous)


_BLAS_ENVIRONMENT = ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'OMP_NUM_THREADS')


def numerical_environment():
    """Observed BLAS/Numba thread state, recorded so an unpinned run is visible.

    The frozen native reference was produced with single-threaded BLAS. A
    multi-threaded pool in the calling process oversubscribes the kernel's own
    single-threaded workers (measured thousands of times slower dot products on
    a 32-thread host) and changes last-ulp results, so digests stop reproducing.
    This only observes and records; it changes no setting.
    """
    from numba import get_num_threads
    try:
        from threadpoolctl import threadpool_info
        pools = [{k: p.get(k) for k in ('user_api', 'internal_api', 'threading_layer', 'num_threads', 'version')}
                 for p in threadpool_info()]
        observed = True
    except Exception:
        pools, observed = [], False
    blas = [p for p in pools if p.get('user_api') == 'blas']
    multithreaded = observed and any((p.get('num_threads') or 1) > 1 for p in blas)
    return dict(environment={k: os.environ.get(k) for k in _BLAS_ENVIRONMENT},
                thread_pools=pools, thread_pools_observed=observed,
                numba_threads=int(get_num_threads()),
                blas_multithreaded=bool(multithreaded),
                blas_single_threaded=bool(observed and not multithreaded),
                reference_requires_single_threaded_blas=True)


BLAS_WARNING = ('BLAS is multi-threaded in this process; the frozen native reference requires '
                'single-threaded BLAS. Set OPENBLAS_NUM_THREADS=1 (MKL_NUM_THREADS / VECLIB_MAXIMUM_THREADS '
                'for other builds) before the process imports numpy. Results may differ in the last ulp '
                'and run far slower.')


def warn_if_blas_multithreaded(environment):
    """Loud, not silent: a run whose BLAS is not pinned may not reproduce."""
    if environment.get('blas_multithreaded'):
        threads = [p.get('num_threads') for p in environment.get('thread_pools', []) if p.get('user_api') == 'blas']
        warnings.warn(f'{BLAS_WARNING} Observed BLAS pool sizes: {threads}.', RuntimeWarning, stacklevel=3)
        return True
    return False


@contextmanager
def single_threaded_blas():
    """Pin BLAS to one thread for a one-shot process such as the CLI.

    A concurrent server must set the environment at launch instead (see
    numerical_thread_budget); this is process-wide while active. Without
    threadpoolctl it is a no-op and the manifest still records what ran.
    """
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        yield
        return
    with threadpool_limits(limits=1, user_api='blas'):
        yield


# One spawned worker pool per workflow run. Every stage that runs in processes
# borrows it instead of spawning its own, so the package is imported once per
# worker per run rather than once per stage. Importing this package costs about
# one second from a local disk and over a minute from a synced network folder;
# with five to ten stage pools per run the difference decides whether process
# parallelism is a gain or a loss. Values are unaffected: tasks are identical,
# only the process lifecycle changes.
WORKER_ENV = dict(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                  VECLIB_MAXIMUM_THREADS='1', NUMEXPR_NUM_THREADS='1', BLIS_NUM_THREADS='1',
                  # Upper bound for numba.set_num_threads inside a task; every
                  # task sets its own count explicitly (1 unless it asks for more).
                  NUMBA_NUM_THREADS='4')
WORKER_NUMBA_THREAD_LIMIT = 4

_shared_pool = ContextVar('consensus_shared_worker_pool', default=None)


class SharedWorkerPool:
    def __init__(self, cores):
        self.cores = int(cores)
        self.executor = None

    def get(self):
        if self.executor is None:
            from joblib.externals.loky import ProcessPoolExecutor
            from joblib.externals.loky.backend.context import get_context
            self.executor = ProcessPoolExecutor(max_workers=self.cores, timeout=None,
                                                context=get_context('loky'), env=WORKER_ENV)
        return self.executor

    def close(self, kill=False):
        executor, self.executor = self.executor, None
        if executor is not None:
            executor.shutdown(wait=True, kill_workers=kill)


@contextmanager
def shared_worker_pool(cores):
    """Own the run's worker pool; spawned lazily on first use, closed on exit."""
    if int(cores) <= 1:
        yield None
        return
    pool = SharedWorkerPool(cores)
    token = _shared_pool.set(pool)
    try:
        yield pool
    finally:
        _shared_pool.reset(token)
        pool.close()


def stage_executor(workers):
    """Executor for a stage wanting up to ``workers`` concurrent tasks.

    Returns (executor, release) where ``release(failed)`` must be called when
    the stage is done. Inside a run the shared pool is returned and stays alive;
    a failed stage kills its workers so no half-finished task can leak into the
    next stage. Outside a run a private pool of ``workers`` processes is created
    and shut down by ``release``.
    """
    pool = _shared_pool.get()
    if pool is not None:
        executor = pool.get()
        def release(failed=False):
            if failed:
                pool.close(kill=True)
        return executor, release
    from joblib.externals.loky import ProcessPoolExecutor
    from joblib.externals.loky.backend.context import get_context
    executor = ProcessPoolExecutor(max_workers=max(1, int(workers)), timeout=None,
                                   context=get_context('loky'), env=WORKER_ENV)
    def release(failed=False):
        executor.shutdown(wait=True, kill_workers=failed)
    return executor, release


def task_thread_budget(threads=1):
    """Called first thing inside every worker task: an explicit numba budget."""
    from numba import config, set_num_threads
    set_num_threads(max(1, min(int(threads), config.NUMBA_NUM_THREADS)))


_WORKER_STATES = []


def register_worker_state(state):
    """Stage modules register their per-worker state dict at import."""
    if not any(s is state for s in _WORKER_STATES):
        _WORKER_STATES.append(state)
    return state


def load_worker_state(state, path, loader):
    """Per-worker lazy payload: load ``path`` once; a new payload evicts every
    stage's previous one, so a worker holds one stage payload at a time."""
    if state.get('__path__') != path:
        for other in _WORKER_STATES:
            other.clear()
        state.clear()
        state.update(loader(path))
        state['__path__'] = path
    return state
