"""
Benchmark-specific fixtures: larger synthetic BAMs and shared config.
"""
import os
import tempfile
import pytest

# Import from parent conftest
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from conftest import make_synthetic_bam


def pytest_configure(config):
    config.addinivalue_line("markers", "benchmark: performance benchmark test")


@pytest.fixture(scope="session")
def _bench_dir():
    """Session-scoped temp directory for benchmark BAMs."""
    with tempfile.TemporaryDirectory(prefix="fiberhmm_bench_") as d:
        yield d


@pytest.fixture(scope="session")
def bench_model_path():
    """Path to the real hia5 PacBio model."""
    path = os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'hia5_pacbio.json')
    path = os.path.abspath(path)
    assert os.path.exists(path), f"Model not found: {path}"
    return path


@pytest.fixture(scope="session")
def bench_bam_medium(_bench_dir):
    """5000 reads for throughput benchmarks."""
    path = os.path.join(_bench_dir, "bench_medium.bam")
    return make_synthetic_bam(path, n_reads=5000, n_chroms=5, read_length=5000)


@pytest.fixture(scope="session")
def bench_bam_large(_bench_dir):
    """20000 reads for scaling benchmarks."""
    path = os.path.join(_bench_dir, "bench_large.bam")
    return make_synthetic_bam(path, n_reads=20000, n_chroms=10, read_length=5000)
