"""
Shared pytest fixtures for FiberHMM tests.
"""
import pytest
import numpy as np
import tempfile
import os


@pytest.fixture
def simple_emission_probs():
    """
    Simple 2-state, 4-symbol emission probability matrix.
    State 0: Low methylation (footprint)
    State 1: High methylation (accessible)
    """
    return np.array([
        [0.1, 0.2, 0.3, 0.4],  # State 0: prefers higher symbols (unmethylated)
        [0.4, 0.3, 0.2, 0.1],  # State 1: prefers lower symbols (methylated)
    ])


@pytest.fixture
def hexamer_emission_probs():
    """
    Realistic emission probability matrix for k=3 context.
    4^3 * 2 = 128 symbols (64 methylated + 64 unmethylated contexts)
    """
    np.random.seed(42)
    n_contexts = 64  # 4^3
    n_symbols = n_contexts * 2  # methylated + unmethylated

    # State 0 (footprint): low methylation probability
    state0_meth = np.random.uniform(0.05, 0.15, n_contexts)
    state0_unmeth = 1 - state0_meth

    # State 1 (accessible): high methylation probability
    state1_meth = np.random.uniform(0.6, 0.9, n_contexts)
    state1_unmeth = 1 - state1_meth

    emit = np.zeros((2, n_symbols))
    emit[0, :n_contexts] = state0_meth
    emit[0, n_contexts:] = state0_unmeth
    emit[1, :n_contexts] = state1_meth
    emit[1, n_contexts:] = state1_unmeth

    # Normalize
    emit /= emit.sum(axis=1, keepdims=True)

    return emit


@pytest.fixture
def simple_observations():
    """Simple observation sequence that should show clear state transitions."""
    # Pattern: low symbols (state 1) -> high symbols (state 0) -> low symbols (state 1)
    return np.array([0, 0, 0, 1, 1, 1, 3, 3, 3, 3, 3, 3, 1, 1, 0, 0, 0], dtype=np.int32)


@pytest.fixture
def trained_model(simple_emission_probs, simple_observations):
    """A trained FiberHMM model for testing."""
    from fiberhmm.core.hmm import FiberHMM

    model = FiberHMM(n_states=2)
    model.emissionprob_ = simple_emission_probs
    model.startprob_ = np.array([0.5, 0.5])
    model.transmat_ = np.array([[0.9, 0.1], [0.1, 0.9]])

    # Train briefly
    X = simple_observations.reshape(-1, 1)
    model.fit(X, verbose=False)

    return model


@pytest.fixture
def k3_model(hexamer_emission_probs):
    """FiberHMM model with k=3 context (128 emission symbols)."""
    from fiberhmm.core.hmm import FiberHMM

    model = FiberHMM(n_states=2)
    model.emissionprob_ = hexamer_emission_probs
    model.startprob_ = np.array([0.5, 0.5])
    model.transmat_ = np.array([[0.95, 0.05], [0.05, 0.95]])
    return model


@pytest.fixture
def test_model_path():
    """Path to a real JSON model for integration tests (if available)."""
    path = os.path.join(os.path.dirname(__file__), '..', 'models', 'dm6_dddb.json')
    if os.path.exists(path):
        return path
    return None


@pytest.fixture
def temp_dir():
    """Temporary directory for file operations."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def mock_bam_data():
    """Mock BAM read data for testing bam_reader functions."""
    return {
        'read_id': 'test_read_001',
        'chrom': 'chr1',
        'start': 1000,
        'end': 2000,
        'sequence': 'ACGTACGTACGT' * 83,  # ~1000 bp
        'methylation_positions': [10, 25, 50, 75, 100, 150, 200],
        'methylation_probs': [200, 180, 220, 190, 240, 170, 210],  # 0-255 scale
    }
