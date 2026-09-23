"""Independent-fit concurrency must preserve fits, fold order and cancellation."""
import os
from pathlib import Path

import numpy as np
import pytest

from fiberhmm.inference.consensus.fit_execution import (
    NativeFitPool, _parallel_capacity, fit_native_models, native_fit_pool)


def worker_probe():
    import os
    from threadpoolctl import threadpool_info
    from numba import get_num_threads
    return dict(numba=get_num_threads(),blas=[v['num_threads'] for v in threadpool_info()],
                environment={key:os.environ.get(key) for key in
                    ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS')})


def inputs():
    rng = np.random.default_rng(434)
    xy = np.array([(a, b) for a in range(0, 8, 2) for b in range(9, 19, 2)], float)
    ll = rng.normal(0, 3, (16, len(xy)))
    allowed = rng.random(ll.shape) > .2
    allowed[:, 0] = True
    return (ll, allowed, xy, np.ones(len(xy)))


def options():
    return dict(row_sets={'full': None, 0: list(range(8, 16)), 1: list(range(8))},
                reference=np.array([4, 12]), max_iterations=30)


def assert_fits_identical(left, right):
    assert list(left) == list(right)
    for key in left:
        assert left[key].keys() == right[key].keys()
        for field, value in left[key].items():
            if isinstance(value, np.ndarray):
                np.testing.assert_array_equal(value, right[key][field])
            else:
                assert value == right[key][field], (key, field)


def test_parallel_fit_keeps_reference_and_fold_initializations():
    arrays, kw = inputs(), options()
    expected = fit_native_models(*arrays, **kw)
    env = {key: os.environ.get(key) for key in ('OPENBLAS_NUM_THREADS', 'NUMBA_NUM_THREADS')}
    events = []
    with native_fit_pool(2, 1024**3) as pool:
        actual = fit_native_models(*arrays, **kw, progress=events.append)
        executor = pool.executor
        assert executor is not None
        probe=executor.submit(worker_probe).result()
        assert probe['numba']==1 and all(n==1 for n in probe['blas'])
        assert set(probe['environment'].values())=={'1'}
        again = fit_native_models(*arrays, **kw)
        assert pool.executor is executor
    assert_fits_identical(expected, actual)
    assert_fits_identical(expected, again)
    assert env == {key: os.environ.get(key) for key in env}
    assert any('concurrent workers' in msg for msg in events)
    assert any('3/3 completed' in msg for msg in events)
    assert pool.executor is None


def test_memory_budget_reduces_parallelism_not_training_rows(monkeypatch):
    import threadpoolctl
    # Single-threaded BLAS, so the serial path runs in-process (see the next test).
    monkeypatch.setattr(threadpoolctl,'threadpool_info',lambda:[dict(user_api='blas',num_threads=1)])
    assert _parallel_capacity((100, 1000), 11, 4, 2*1024**3) == 4
    assert _parallel_capacity((1295, 20301), 11, 4, 2*1024**3) == 1
    arrays, kw = inputs(), options()
    expected = fit_native_models(*arrays, **kw)
    with native_fit_pool(4, 1) as pool:
        actual = fit_native_models(*arrays, **kw)
        assert pool.executor is None
    assert_fits_identical(expected, actual)


def test_serial_fallback_cannot_use_unbounded_browser_blas(monkeypatch):
    import threadpoolctl
    arrays,kw=inputs(),options()
    expected=fit_native_models(*arrays,**kw)
    monkeypatch.setattr(threadpoolctl,'threadpool_info',lambda:[dict(user_api='blas',num_threads=32)])
    with native_fit_pool(128,1) as pool:
        actual=fit_native_models(*arrays,**kw)
        assert pool.isolate_serial and pool.executor is not None
        assert pool.executor_workers==1
        probe=pool.executor.submit(worker_probe).result()
        assert all(n==1 for n in probe['blas'])
    assert_fits_identical(expected,actual)


def test_cancel_parallel_batch_drains_and_cleans_mappings(monkeypatch, tmp_path):
    import tempfile
    monkeypatch.setattr(tempfile, 'tempdir', str(tmp_path))
    def cancel(message):
        if 'completed;' in message:
            raise InterruptedError('requested cancellation')
    with native_fit_pool(2, 1024**3):
        with pytest.raises(InterruptedError, match='requested cancellation'):
            fit_native_models(*inputs(), **options(), progress=cancel)
    assert not list(tmp_path.glob('fiberhmm-native-fits-*'))


def test_native_classifier_parallel_is_identical():
    from test_consensus_measurement_family import fixture
    from fiberhmm.inference.consensus.measurement_family import classify_family_profiles
    s, catalog = fixture()
    kw = dict(region=(0,81), family_model='latent_distribution',
              scoring_folds=2, predictive_replicates=31, max_fit_iterations=25)
    expected = classify_family_profiles(s, catalog, **kw)
    with native_fit_pool(2, 1024**3):
        actual = classify_family_profiles(s, catalog, **kw)
    assert expected == actual


@pytest.mark.parametrize('cores', [0, True, 1.5])
def test_invalid_cpu_budget(cores):
    with pytest.raises(ValueError):
        NativeFitPool(cores, 1024**3)
