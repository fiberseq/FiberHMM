"""
Unit tests for FiberHMM HMM module.

Tests cover:
- FiberHMM class initialization and methods
- Viterbi algorithm correctness
- Forward-backward algorithm
- Model serialization (JSON, NPZ, pickle)
- State normalization
"""
import os
import sys
import types

import numpy as np
import pytest

# Try package imports first, fall back to flat imports
try:
    from fiberhmm.core import hmm as hmm_module
    from fiberhmm.core.hmm import (
        FiberHMM,
        _baum_welch_estep_python,
        _baum_welch_estep_sequence,
        _fit_training_iteration,
        _hmmlearn_uses_categorical,
        _hmmlearn_version_tuple,
        _initialized_training_model,
        _logsumexp,
        _logsumexp_axis1,
        _methylated_emission_means,
        _normalized_start_counts,
        _normalized_transition_counts,
        _random_training_parameters,
        _rust_model_payload,
        _swap_hmm_state_order,
        _training_data_for_iteration,
        _training_has_converged,
        _training_observation_matrix,
        _training_progress_postfix,
        _try_create_hmmlearn,
        _updated_best_training_model,
    )
    from fiberhmm.core.model_io import load_model, load_model_with_metadata, save_model
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    import hmm as hmm_module
    from hmm import (
        FiberHMM,
        _baum_welch_estep_python,
        _baum_welch_estep_sequence,
        _fit_training_iteration,
        _hmmlearn_uses_categorical,
        _hmmlearn_version_tuple,
        _initialized_training_model,
        _logsumexp,
        _logsumexp_axis1,
        _methylated_emission_means,
        _normalized_start_counts,
        _normalized_transition_counts,
        _random_training_parameters,
        _rust_model_payload,
        _swap_hmm_state_order,
        _training_data_for_iteration,
        _training_has_converged,
        _training_observation_matrix,
        _training_progress_postfix,
        _try_create_hmmlearn,
        _updated_best_training_model,
        load_model,
        load_model_with_metadata,
        save_model,
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

    def test_training_count_normalization_helpers(self):
        startprob = _normalized_start_counts(np.array([0.0, 2.0]))
        transmat = _normalized_transition_counts(np.array([
            [0.0, 0.0],
            [1.0, 3.0],
        ]))

        assert startprob[0] > 0
        np.testing.assert_allclose(startprob.sum(), 1.0)
        np.testing.assert_allclose(transmat.sum(axis=1), [1.0, 1.0])
        assert np.all(transmat > 0)

    def test_baum_welch_estep_python_returns_expected_counts(
        self,
        simple_emission_probs,
        simple_observations,
    ):
        model = FiberHMM()
        model.emissionprob_ = simple_emission_probs
        model.startprob_ = np.array([0.5, 0.5])
        model.transmat_ = np.array([[0.8, 0.2], [0.2, 0.8]])
        model._compute_log_probs()

        obs = simple_observations.flatten().astype(int)
        start_counts, trans_counts, log_prob = _baum_welch_estep_python(
            model,
            obs,
        )

        assert start_counts.shape == (2,)
        assert trans_counts.shape == (2, 2)
        np.testing.assert_allclose(start_counts.sum(), 1.0)
        np.testing.assert_allclose(trans_counts.sum(), len(obs) - 1)
        np.testing.assert_allclose(log_prob, model.score(obs))

    def test_baum_welch_estep_sequence_uses_python_fallback(
        self,
        monkeypatch,
        simple_emission_probs,
        simple_observations,
    ):
        model = FiberHMM()
        model.emissionprob_ = simple_emission_probs
        model.startprob_ = np.array([0.5, 0.5])
        model.transmat_ = np.array([[0.8, 0.2], [0.2, 0.8]])
        model._compute_log_probs()
        obs = simple_observations.flatten().astype(int)

        monkeypatch.setattr(hmm_module, "HAS_NUMBA", False)
        fallback = _baum_welch_estep_sequence(model, obs)
        direct = _baum_welch_estep_python(model, obs)

        for fallback_value, direct_value in zip(fallback, direct):
            np.testing.assert_allclose(fallback_value, direct_value)

    def test_methylated_emission_means_use_first_half_only(self):
        means = _methylated_emission_means(np.array([
            [0.1, 0.3, 0.9, 0.9],
            [0.6, 0.8, 0.0, 0.0],
        ]))

        assert means == pytest.approx((0.2, 0.7))

    def test_hmmlearn_version_selection(self):
        assert _hmmlearn_version_tuple("0.2.8") == (0, 2)
        assert _hmmlearn_version_tuple("0.3.0") == (0, 3)
        assert _hmmlearn_version_tuple("1.0.0") == (1, 0)
        assert not _hmmlearn_uses_categorical((0, 2))
        assert _hmmlearn_uses_categorical((0, 3))
        assert _hmmlearn_uses_categorical((1, 0))

    def test_try_create_hmmlearn_selects_model_for_version(self, monkeypatch):
        class FakeCategoricalHMM:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        class FakeMultinomialHMM:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        def install_fake_hmmlearn(version):
            hmmlearn_module = types.ModuleType("hmmlearn")
            hmmlearn_module.__version__ = version
            hmm_module = types.ModuleType("hmmlearn.hmm")
            hmm_module.CategoricalHMM = FakeCategoricalHMM
            hmm_module.MultinomialHMM = FakeMultinomialHMM
            monkeypatch.setitem(sys.modules, "hmmlearn", hmmlearn_module)
            monkeypatch.setitem(sys.modules, "hmmlearn.hmm", hmm_module)

        emission_probs = np.array([[0.8, 0.2], [0.2, 0.8]])

        install_fake_hmmlearn("0.3.0")
        categorical = _try_create_hmmlearn(emission_probs)
        assert isinstance(categorical, FakeCategoricalHMM)
        assert categorical.kwargs == {
            "n_components": 2,
            "init_params": "",
            "params": "st",
            "n_iter": 1000,
        }
        np.testing.assert_array_equal(categorical.emissionprob_, emission_probs)

        install_fake_hmmlearn("0.2.8")
        multinomial = _try_create_hmmlearn(emission_probs)
        assert isinstance(multinomial, FakeMultinomialHMM)
        np.testing.assert_array_equal(multinomial.emissionprob_, emission_probs)

    def test_random_training_parameters_are_seeded_and_normalized(self):
        start_a, trans_a = _random_training_parameters(3)
        start_b, trans_b = _random_training_parameters(3)

        np.testing.assert_array_equal(start_a, start_b)
        np.testing.assert_array_equal(trans_a, trans_b)
        np.testing.assert_allclose(start_a.sum(), 1.0)
        np.testing.assert_allclose(trans_a.sum(axis=1), [1.0, 1.0])

    def test_training_data_for_iteration_cycles_dict_inputs(self):
        arrays = {
            0: np.array([0, 1]),
            1: np.array([2, 3]),
        }

        np.testing.assert_array_equal(_training_data_for_iteration(arrays, 0), [0, 1])
        np.testing.assert_array_equal(_training_data_for_iteration(arrays, 3), [2, 3])

        direct = np.array([4, 5])
        assert _training_data_for_iteration(direct, 99) is direct

    def test_training_model_initialization_and_observation_matrix(self):
        emission_probs = np.array([[0.8, 0.2], [0.2, 0.8]])

        model = _initialized_training_model(emission_probs, use_legacy=False, seed=4)
        expected_start, expected_trans = _random_training_parameters(4)

        assert isinstance(model, FiberHMM)
        np.testing.assert_array_equal(model.emissionprob_, emission_probs)
        np.testing.assert_allclose(model.startprob_, expected_start)
        np.testing.assert_allclose(model.transmat_, expected_trans)
        np.testing.assert_array_equal(
            _training_observation_matrix(np.array([1, 2, 3])),
            np.array([[1], [2], [3]]),
        )

    def test_updated_best_training_model_replaces_only_on_improvement(self):
        old_model = object()
        new_model = object()

        assert _updated_best_training_model(
            old_model, -10.0, new_model, -9.0,
        ) == (new_model, -9.0)
        assert _updated_best_training_model(
            old_model, -10.0, new_model, -11.0,
        ) == (old_model, -10.0)

    def test_fit_training_iteration_shapes_data_and_labels_fit(self):
        class FakeModel:
            def __init__(self):
                self.calls = []

            def fit(self, training, **kwargs):
                self.calls.append((training, kwargs))

        model = FakeModel()
        training = _fit_training_iteration(model, np.array([1, 2, 3]), 4)

        np.testing.assert_array_equal(training, np.array([[1], [2], [3]]))
        assert len(model.calls) == 1
        call_training, kwargs = model.calls[0]
        np.testing.assert_array_equal(call_training, training)
        assert kwargs == {
            "lengths": [3],
            "verbose": True,
            "desc": "Init 5 EM",
        }

    def test_training_progress_and_convergence_helpers(self):
        assert _training_progress_postfix(-1234.0, 0.0123) == {
            "logprob": "-1.23e+03",
            "delta": "1.23e-02",
        }
        assert not _training_has_converged(0, 0.0, 1e-4)
        assert not _training_has_converged(2, 1e-3, 1e-4)
        assert _training_has_converged(2, 1e-5, 1e-4)

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

    def test_swap_hmm_state_order_updates_probabilities_and_clears_logs(self):
        model = FiberHMM()
        model.emissionprob_ = np.array([[0.1, 0.2], [0.8, 0.9]])
        model.startprob_ = np.array([0.3, 0.7])
        model.transmat_ = np.array([[0.8, 0.2], [0.1, 0.9]])
        model._log_startprob = np.array([1.0])
        model._log_transmat = np.array([2.0])
        model._log_emissionprob = np.array([3.0])
        model._log_probs_frozen = True

        _swap_hmm_state_order(model)

        np.testing.assert_array_equal(model.emissionprob_, [[0.8, 0.9], [0.1, 0.2]])
        np.testing.assert_array_equal(model.startprob_, [0.7, 0.3])
        np.testing.assert_array_equal(model.transmat_, [[0.9, 0.1], [0.2, 0.8]])
        assert model._log_startprob is None
        assert model._log_transmat is None
        assert model._log_emissionprob is None
        assert model._log_probs_frozen is False

    def test_normalize_states_correct_order(self, hexamer_emission_probs):
        """Test that normalize_states() puts states in correct order."""
        model = FiberHMM()
        model.emissionprob_ = hexamer_emission_probs
        model.startprob_ = np.array([0.5, 0.5])
        model.transmat_ = np.array([[0.9, 0.1], [0.1, 0.9]])

        # Normalize
        model.normalize_states()

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

    def test_rust_model_payload_transposes_emissions(self):
        model = FiberHMM()
        model.startprob_ = np.array([0.25, 0.75])
        model.transmat_ = np.array([[0.8, 0.2], [0.1, 0.9]])
        model.emissionprob_ = np.array([
            [0.1, 0.2, 0.3],
            [0.9, 0.8, 0.7],
        ])

        assert _rust_model_payload(model) == {
            "startprob": [0.25, 0.75],
            "transmat": [[0.8, 0.2], [0.1, 0.9]],
            "emissionprob": [[0.1, 0.9], [0.2, 0.8], [0.3, 0.7]],
        }


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

    def test_posterior_matrix_matches_predict_proba(self, trained_model, simple_observations):
        """Private posterior helper stays aligned with public predict_proba."""
        obs = simple_observations.ravel()

        trained_model._compute_log_probs()

        np.testing.assert_allclose(
            trained_model._posterior_matrix(obs),
            trained_model.predict_proba(obs),
        )

    def test_viterbi_path_matches_predict(self, trained_model, simple_observations):
        """Private path helper stays aligned with public predict."""
        obs = simple_observations.ravel()

        trained_model._compute_log_probs()

        np.testing.assert_array_equal(
            trained_model._viterbi_path(obs),
            trained_model.predict(obs),
        )

    def test_score_returns_finite(self, trained_model, simple_observations):
        """Test that score() returns a finite value."""
        X = simple_observations.reshape(-1, 1)
        score = trained_model.score(X)

        assert np.isfinite(score)
        assert score < 0  # Log probability should be negative

    def test_predict_recomputes_logs_after_in_place_mutation(self):
        """Default predict() keeps existing public in-place mutation semantics."""
        model = FiberHMM()
        model.startprob_ = np.array([0.5, 0.5])
        model.transmat_ = np.array([[0.5, 0.5], [0.5, 0.5]])
        model.emissionprob_ = np.array([[0.9, 0.1], [0.1, 0.9]])
        obs = np.zeros(4, dtype=np.int64)

        before = model.predict(obs)
        model.emissionprob_[:] = np.array([[0.1, 0.9], [0.9, 0.1]])
        after = model.predict(obs)

        np.testing.assert_array_equal(before, np.zeros(4, dtype=np.int8))
        np.testing.assert_array_equal(after, np.ones(4, dtype=np.int8))

    def test_freeze_log_probs_is_explicit_read_only_cache(self):
        """Frozen logs stay stable until explicitly unfrozen."""
        model = FiberHMM()
        model.startprob_ = np.array([0.5, 0.5])
        model.transmat_ = np.array([[0.5, 0.5], [0.5, 0.5]])
        model.emissionprob_ = np.array([[0.9, 0.1], [0.1, 0.9]])
        obs = np.zeros(4, dtype=np.int64)

        model.freeze_log_probs()
        before = model.predict(obs)
        model.emissionprob_[:] = np.array([[0.1, 0.9], [0.9, 0.1]])
        frozen = model.predict(obs)
        model.unfreeze_log_probs()
        after = model.predict(obs)

        np.testing.assert_array_equal(before, np.zeros(4, dtype=np.int8))
        np.testing.assert_array_equal(frozen, before)
        np.testing.assert_array_equal(after, np.ones(4, dtype=np.int8))


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
