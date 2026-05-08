"""HMM core prediction benchmarks."""

import time

import numpy as np
import pytest

from fiberhmm.core.model_io import load_model

pytestmark = pytest.mark.benchmark


def test_viterbi_predict_throughput(bench_model_path):
    """Measure warm Viterbi prediction throughput on fixed-length observations."""
    model = load_model(bench_model_path)
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
