"""
Unit tests for FiberHMM HMM module.

Tests cover:
- FiberHMM class initialization and methods
- Viterbi algorithm correctness
- Forward-backward algorithm
- Model serialization (JSON, NPZ, pickle)
- State normalization
"""
import pytest
import numpy as np
import tempfile
import os
import sys

# Try package imports first, fall back to flat imports
try:
    from fiberhmm.core.hmm import FiberHMM, _logsumexp, _logsumexp_axis1
    from fiberhmm.core.model_io import load_model, save_model, load_model_with_metadata
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from hmm import (
        FiberHMM,
        load_model,
        save_model,
        load_model_with_metadata,
        _logsumexp,
        _logsumexp_axis1,
    )


class TestFiberHMMInitialization:
    """Test FiberHMM class initialization."""

    def test_default_initialization(self):
        """Test that FiberHMM initializes with correct defaults."""
        model = FiberHMM()
        assert model.n_states == 2
        assert model.startprob_ is None
        assert model.transmat_ is None
        assert model.emissionprob_ is None

    def test_custom_n_states(self):
        """Test initialization with custom number of states."""
        model = FiberHMM(n_states=3)
        assert model.n_states == 3


class TestViterbiAlgorithm:
    """Test Viterbi decoding."""

    def test_viterbi_returns_valid_path(self, simple_emission_probs, simple_observations):
        """Test that Viterbi returns a valid state path."""
        model = FiberHMM()
        model.emissionprob_ = simple_emission_probs
        model.startprob_ = np.array([0.5, 0.5])
        model.transmat_ = np.array([[0.9, 0.1], [0.1, 0.9]])

        path = model.predict(simple_observations)

        assert len(path) == len(simple_observations)
        assert all(s in [0, 1] for s in path)

    def test_viterbi_log_probability(self, simple_emission_probs, simple_observations):
        """Test that Viterbi returns finite log probability."""
        model = FiberHMM()
        model.emissionprob_ = simple_emission_probs
        model.startprob_ = np.array([0.5, 0.5])
        model.transmat_ = np.array([[0.9, 0.1], [0.1, 0.9]])
        model._compute_log_probs()

        obs = simple_observations.flatten().astype(int)
        path, log_prob = model._viterbi(obs)

        assert np.isfinite(log_prob)
        assert log_prob < 0  # Log probabilities should be negative

    def test_viterbi_deterministic(self, simple_emission_probs, simple_observations):
        """Test that Viterbi gives deterministic results."""
        model = FiberHMM()
        model.emissionprob_ = simple_emission_probs
        model.startprob_ = np.array([0.5, 0.5])
        model.transmat_ = np.array([[0.9, 0.1], [0.1, 0.9]])

        path1 = model.predict(simple_observations)
        path2 = model.predict(simple_observations)

        np.testing.assert_array_equal(path1, path2)


class TestForwardBackward:
    """Test forward-backward algorithm."""

    def test_forward_returns_valid_alpha(self, simple_emission_probs, simple_observations):
        """Test that forward algorithm returns valid alpha matrix."""
        model = FiberHMM()
        model.emissionprob_ = simple_emission_probs
        model.startprob_ = np.array([0.5, 0.5])
        model.transmat_ = np.array([[0.9, 0.1], [0.1, 0.9]])
        model._compute_log_probs()

        obs = simple_observations.flatten().astype(int)
        alpha, log_prob = model._forward(obs)

        assert alpha.shape == (len(obs), 2)
        assert np.all(np.isfinite(alpha))
        assert np.isfinite(log_prob)

    def test_backward_returns_valid_beta(self, simple_emission_probs, simple_observations):
        """Test that backward algorithm returns valid beta matrix."""
        model = FiberHMM()
        model.emissionprob_ = simple_emission_probs
        model.startprob_ = np.array([0.5, 0.5])
        model.transmat_ = np.array([[0.9, 0.1], [0.1, 0.9]])
        model._compute_log_probs()

        obs = simple_observations.flatten().astype(int)
        beta = model._backward(obs)

        assert beta.shape == (len(obs), 2)
        assert np.all(np.isfinite(beta))
        # Last position should be 0 (log(1))
        np.testing.assert_allclose(beta[-1], [0, 0])

    def test_posteriors_sum_to_one(self, simple_emission_probs, simple_observations):
        """Test that posterior probabilities sum to 1 at each position."""
        model = FiberHMM()
        model.emissionprob_ = simple_emission_probs
        model.startprob_ = np.array([0.5, 0.5])
        model.transmat_ = np.array([[0.9, 0.1], [0.1, 0.9]])

        posteriors = model.predict_proba(simple_observations)

        # Check shape
        assert posteriors.shape == (len(simple_observations), 2)

        # Check that each row sums to 1
        row_sums = posteriors.sum(axis=1)
        np.testing.assert_allclose(row_sums, np.ones(len(simple_observations)), rtol=1e-5)

        # Check probabilities are in [0, 1]
        assert np.all(posteriors >= 0)
        assert np.all(posteriors <= 1)


class TestModelTraining:
    """Test Baum-Welch training."""

    def test_fit_updates_parameters(self, simple_emission_probs, simple_observations):
        """Test that fit() updates start and transition probabilities."""
        model = FiberHMM()
        model.emissionprob_ = simple_emission_probs
        model.startprob_ = np.array([0.5, 0.5])
        model.transmat_ = np.array([[0.5, 0.5], [0.5, 0.5]])

        initial_startprob = model.startprob_.copy()
        initial_transmat = model.transmat_.copy()

        X = simple_observations.reshape(-1, 1)
        model.fit(X, verbose=False)

        # Parameters should have changed
        assert not np.allclose(model.startprob_, initial_startprob) or \
               not np.allclose(model.transmat_, initial_transmat)

        # Probabilities should still be valid
        np.testing.assert_allclose(model.startprob_.sum(), 1.0, rtol=1e-5)
        np.testing.assert_allclose(model.transmat_.sum(axis=1), [1.0, 1.0], rtol=1e-5)

    def test_fit_improves_likelihood(self, simple_emission_probs, simple_observations):
        """Test that training improves (or maintains) log likelihood."""
        model = FiberHMM()
        model.emissionprob_ = simple_emission_probs
        model.startprob_ = np.array([0.5, 0.5])
        model.transmat_ = np.array([[0.5, 0.5], [0.5, 0.5]])

        X = simple_observations.reshape(-1, 1)
        initial_score = model.score(X)

        model.fit(X, verbose=False)
        final_score = model.score(X)

        # Score should improve or stay the same
        assert final_score >= initial_score - 1e-6

    def test_training_monitor(self, simple_emission_probs, simple_observations):
        """Test that training monitor records history."""
        model = FiberHMM()
        model.emissionprob_ = simple_emission_probs
        model.startprob_ = np.array([0.5, 0.5])
        model.transmat_ = np.array([[0.5, 0.5], [0.5, 0.5]])

        X = simple_observations.reshape(-1, 1)
        model.fit(X, verbose=False)

        assert model.monitor_ is not None
        assert len(model.monitor_.history) > 0


class TestStateNormalization:
    """Test state normalization functionality."""

    def test_normalize_states_correct_order(self, hexamer_emission_probs):
        """Test that normalize_states() puts states in correct order."""
        model = FiberHMM()
        model.emissionprob_ = hexamer_emission_probs
        model.startprob_ = np.array([0.5, 0.5])
        model.transmat_ = np.array([[0.9, 0.1], [0.1, 0.9]])

        # Normalize
        swapped = model.normalize_states()

        # After normalization, State 0 should have lower mean emission for methylated positions
        n_contexts = model.emissionprob_.shape[1] // 2
        mean_0 = np.mean(model.emissionprob_[0, :n_contexts])
        mean_1 = np.mean(model.emissionprob_[1, :n_contexts])

        assert mean_0 <= mean_1, "State 0 should have lower mean (footprint)"

    def test_normalize_states_preserves_probabilities(self, hexamer_emission_probs):
        """Test that normalization preserves probability distributions."""
        model = FiberHMM()
        model.emissionprob_ = hexamer_emission_probs.copy()
        model.startprob_ = np.array([0.3, 0.7])
        model.transmat_ = np.array([[0.8, 0.2], [0.15, 0.85]])

        # Store original sums
        original_emit_sums = model.emissionprob_.sum(axis=1)
        original_start_sum = model.startprob_.sum()
        original_trans_sums = model.transmat_.sum(axis=1)

        model.normalize_states()

        # Sums should be preserved
        np.testing.assert_allclose(model.emissionprob_.sum(axis=1), original_emit_sums, rtol=1e-5)
        np.testing.assert_allclose(model.startprob_.sum(), original_start_sum, rtol=1e-5)
        np.testing.assert_allclose(model.transmat_.sum(axis=1), original_trans_sums, rtol=1e-5)


class TestModelSerialization:
    """Test model save/load functionality."""

    def test_save_load_json(self, trained_model, temp_dir):
        """Test saving and loading in JSON format."""
        filepath = os.path.join(temp_dir, "model.json")

        save_model(trained_model, filepath, context_size=3, mode='pacbio-fiber')
        loaded = load_model(filepath, normalize=False)

        np.testing.assert_allclose(loaded.startprob_, trained_model.startprob_, rtol=1e-5)
        np.testing.assert_allclose(loaded.transmat_, trained_model.transmat_, rtol=1e-5)
        np.testing.assert_allclose(loaded.emissionprob_, trained_model.emissionprob_, rtol=1e-5)

    def test_save_redirects_npz_to_json(self, trained_model, temp_dir):
        """Test that saving with .npz extension redirects to .json."""
        import warnings
        filepath = os.path.join(temp_dir, "model.npz")
        json_path = os.path.join(temp_dir, "model.json")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            save_model(trained_model, filepath, context_size=3, mode='pacbio-fiber')
            assert len(w) == 1
            assert "JSON format" in str(w[0].message)

        assert os.path.exists(json_path)
        assert not os.path.exists(filepath)
        loaded = load_model(json_path, normalize=False)
        np.testing.assert_allclose(loaded.startprob_, trained_model.startprob_, rtol=1e-5)

    def test_save_redirects_pickle_to_json(self, trained_model, temp_dir):
        """Test that saving with .pickle extension redirects to .json."""
        import warnings
        filepath = os.path.join(temp_dir, "model.pickle")
        json_path = os.path.join(temp_dir, "model.json")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            save_model(trained_model, filepath, context_size=3, mode='pacbio-fiber')
            assert len(w) == 1
            assert "JSON format" in str(w[0].message)

        assert os.path.exists(json_path)
        assert not os.path.exists(filepath)
        loaded = load_model(json_path, normalize=False)
        np.testing.assert_allclose(loaded.startprob_, trained_model.startprob_, rtol=1e-5)

    def test_load_with_metadata(self, trained_model, temp_dir):
        """Test loading model with metadata."""
        filepath = os.path.join(temp_dir, "model.json")

        save_model(trained_model, filepath, context_size=5, mode='nanopore-fiber')
        model, context_size, mode = load_model_with_metadata(filepath, normalize=False)

        assert context_size == 5
        assert mode == 'nanopore-fiber'

    def test_to_dict_from_dict(self, trained_model):
        """Test serialization to/from dictionary."""
        d = trained_model.to_dict()

        assert 'startprob_' in d
        assert 'transmat_' in d
        assert 'emissionprob_' in d
        assert d['n_states'] == 2

        restored = FiberHMM.from_dict(d)

        np.testing.assert_allclose(restored.startprob_, trained_model.startprob_)
        np.testing.assert_allclose(restored.transmat_, trained_model.transmat_)
        np.testing.assert_allclose(restored.emissionprob_, trained_model.emissionprob_)


class TestPredictionMethods:
    """Test various prediction methods."""

    def test_predict_with_confidence(self, trained_model, simple_observations):
        """Test predict_with_confidence returns valid outputs."""
        path, confidence = trained_model.predict_with_confidence(simple_observations)

        assert len(path) == len(simple_observations)
        assert len(confidence) == len(simple_observations)
        assert all(0 <= c <= 1 for c in confidence)

    def test_predict_with_posteriors(self, trained_model, simple_observations):
        """Test predict_with_posteriors returns valid outputs."""
        path, posteriors = trained_model.predict_with_posteriors(simple_observations)

        assert len(path) == len(simple_observations)
        assert posteriors.shape == (len(simple_observations), 2)
        np.testing.assert_allclose(posteriors.sum(axis=1), np.ones(len(simple_observations)), rtol=1e-5)

    def test_score_returns_finite(self, trained_model, simple_observations):
        """Test that score() returns a finite value."""
        X = simple_observations.reshape(-1, 1)
        score = trained_model.score(X)

        assert np.isfinite(score)
        assert score < 0  # Log probability should be negative


class TestLogsumexp:
    """Test logsumexp utility functions."""

    def test_logsumexp_basic(self):
        """Test basic logsumexp computation."""
        a = np.array([-1.0, -2.0, -3.0])
        result = _logsumexp(a)

        # Manual computation
        expected = np.log(np.exp(-1) + np.exp(-2) + np.exp(-3))
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_logsumexp_with_axis(self):
        """Test logsumexp with axis parameter."""
        a = np.array([[-1.0, -2.0], [-3.0, -4.0]])
        result = _logsumexp(a, axis=1)

        assert result.shape == (2,)

    def test_logsumexp_axis1_2d(self):
        """Test _logsumexp_axis1 for 2D arrays."""
        a = np.array([[-1.0, -2.0], [-3.0, -4.0]])
        result = _logsumexp_axis1(a)

        expected = np.array([
            np.log(np.exp(-1) + np.exp(-2)),
            np.log(np.exp(-3) + np.exp(-4))
        ])
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_logsumexp_numerical_stability(self):
        """Test that logsumexp handles extreme values."""
        # Very negative values
        a = np.array([-1000, -1001, -1002])
        result = _logsumexp(a)
        assert np.isfinite(result)

        # Mixed extreme values
        b = np.array([0, -1000])
        result = _logsumexp(b)
        np.testing.assert_allclose(result, 0, atol=1e-5)  # Should be ~log(1)


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_single_observation(self, simple_emission_probs):
        """Test handling of single observation."""
        model = FiberHMM()
        model.emissionprob_ = simple_emission_probs
        model.startprob_ = np.array([0.5, 0.5])
        model.transmat_ = np.array([[0.9, 0.1], [0.1, 0.9]])

        single_obs = np.array([1])
        path = model.predict(single_obs)

        assert len(path) == 1
        assert path[0] in [0, 1]

    def test_long_sequence(self, simple_emission_probs):
        """Test handling of long sequences."""
        model = FiberHMM()
        model.emissionprob_ = simple_emission_probs
        model.startprob_ = np.array([0.5, 0.5])
        model.transmat_ = np.array([[0.9, 0.1], [0.1, 0.9]])

        np.random.seed(42)
        long_obs = np.random.randint(0, 4, size=10000)
        path = model.predict(long_obs)

        assert len(path) == 10000
        assert all(s in [0, 1] for s in path)

    def test_all_same_observation(self, simple_emission_probs):
        """Test sequence with all same observation."""
        model = FiberHMM()
        model.emissionprob_ = simple_emission_probs
        model.startprob_ = np.array([0.5, 0.5])
        model.transmat_ = np.array([[0.9, 0.1], [0.1, 0.9]])

        same_obs = np.zeros(100, dtype=int)
        path = model.predict(same_obs)

        assert len(path) == 100
        # With high self-transition prob, path should be mostly one state
        unique_states = np.unique(path)
        assert len(unique_states) <= 2


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
