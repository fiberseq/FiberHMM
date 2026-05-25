"""HMM core prediction benchmarks."""

import time

import numpy as np
import pytest

from fiberhmm.core.model_io import load_model

pytestmark = pytest.mark.benchmark


def test_viterbi_predict_throughput(bench_model_path):
    """Measure warm Viterbi prediction throughput on fixed-length observations."""
    model = load_model(bench_model_path)
    model.freeze_log_probs()
    n_symbols = model.emissionprob_.shape[1]
    obs = ((np.arange(5000, dtype=np.int64) * 37) % n_symbols).astype(np.int64)

    path = model.predict(obs)
    assert len(path) == len(obs)

    start = time.perf_counter()
    for _ in range(200):
        path = model.predict(obs)
    elapsed = time.perf_counter() - start

    print(
        "\n  Viterbi predict:"
        f" 200 reads x {len(obs)} bp in {elapsed:.4f}s"
        f" = {200 / elapsed:.0f} reads/s"
    )
    assert len(path) == len(obs)


def test_viterbi_log_freeze_speedup(bench_model_path):
    """Compare repeated prediction with dynamic vs frozen log arrays."""
    model = load_model(bench_model_path)
    n_symbols = model.emissionprob_.shape[1]
    obs = ((np.arange(5000, dtype=np.int64) * 37) % n_symbols).astype(np.int64)

    expected = model.predict(obs)

    dynamic_start = time.perf_counter()
    for _ in range(200):
        dynamic_path = model.predict(obs)
    dynamic_time = time.perf_counter() - dynamic_start

    model.freeze_log_probs()
    frozen_start = time.perf_counter()
    for _ in range(200):
        frozen_path = model.predict(obs)
    frozen_time = time.perf_counter() - frozen_start

    np.testing.assert_array_equal(dynamic_path, expected)
    np.testing.assert_array_equal(frozen_path, expected)
    print(
        "\n  Viterbi log cache:"
        f" dynamic={dynamic_time:.4f}s"
        f" frozen={frozen_time:.4f}s"
        f" speedup={dynamic_time / frozen_time:.2f}x"
    )
    assert frozen_time < dynamic_time
