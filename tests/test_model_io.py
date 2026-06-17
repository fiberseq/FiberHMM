"""
Tests for fiberhmm.core.model_io module.
"""
import json
import os
import pickle
import warnings

import numpy as np
import pytest

from fiberhmm.core.hmm import FiberHMM
from fiberhmm.core.model_io import (
    _json_save_path,
    _json_save_redirect_warning,
    _model_file_format,
    _model_json_record,
    _normalize_model_if_requested,
    _pickle_payload_has_metadata,
    _prepare_loaded_model,
    load_model,
    load_model_for_inference,
    load_model_with_metadata,
    save_model,
)


@pytest.fixture
def sample_model():
    """Create a minimal FiberHMM model for testing."""
    model = FiberHMM(n_states=2)
    model.startprob_ = np.array([0.4, 0.6])
    model.transmat_ = np.array([[0.9, 0.1], [0.1, 0.9]])
    # k=3 context: 4^6 = 4096 symbols per state
    np.random.seed(42)
    emit = np.random.dirichlet(np.ones(128), size=2)
    model.emissionprob_ = emit
    return model


class TestLoadSaveRoundTrip:
    def test_model_file_format_classifies_supported_suffixes(self):
        assert _model_file_format("model.npz") == "npz"
        assert _model_file_format("model.json") == "json"
        assert _model_file_format("model.pkl") == "pickle"
        assert _model_file_format("model.legacy") == "pickle"

    def test_normalize_model_if_requested_calls_normalize_conditionally(self):
        class Model:
            def __init__(self):
                self.calls = 0

            def normalize_states(self):
                self.calls += 1

        model = Model()

        assert _normalize_model_if_requested(model, False) is model
        assert model.calls == 0
        assert _normalize_model_if_requested(model, True) is model
        assert model.calls == 1

    def test_prepare_loaded_model_normalizes_then_unfreezes(self):
        calls = []

        class Model:
            def normalize_states(self):
                calls.append("normalize")

            def unfreeze_log_probs(self):
                calls.append("unfreeze")

        model = Model()

        assert _prepare_loaded_model(model, True) is model
        assert calls == ["normalize", "unfreeze"]

    def test_json_round_trip(self, sample_model, tmp_path):
        filepath = str(tmp_path / "model.json")
        save_model(sample_model, filepath, context_size=3, mode='pacbio-fiber')

        loaded = load_model(filepath, normalize=False)
        np.testing.assert_allclose(loaded.startprob_, sample_model.startprob_, rtol=1e-5)
        np.testing.assert_allclose(loaded.transmat_, sample_model.transmat_, rtol=1e-5)
        np.testing.assert_allclose(loaded.emissionprob_, sample_model.emissionprob_, rtol=1e-5)

    def test_metadata_preserved(self, sample_model, tmp_path):
        filepath = str(tmp_path / "model.json")
        save_model(sample_model, filepath, context_size=5, mode='daf')

        model, ctx, mode = load_model_with_metadata(filepath, normalize=False)
        assert ctx == 5
        assert mode == 'daf'

    def test_npz_metadata_preserved(self, sample_model, tmp_path):
        filepath = str(tmp_path / "legacy.npz")
        np.savez(
            filepath,
            n_states=sample_model.n_states,
            startprob=sample_model.startprob_,
            transmat=sample_model.transmat_,
            emissionprob=sample_model.emissionprob_,
            context_size=5,
            mode='m6a',
        )

        model, ctx, mode = load_model_with_metadata(filepath, normalize=False)

        assert ctx == 5
        assert mode == 'pacbio-fiber'
        np.testing.assert_allclose(model.startprob_, sample_model.startprob_, rtol=1e-5)
        np.testing.assert_allclose(model.transmat_, sample_model.transmat_, rtol=1e-5)
        np.testing.assert_allclose(model.emissionprob_, sample_model.emissionprob_, rtol=1e-5)

    def test_load_model_for_inference_freezes_log_cache(self, sample_model, tmp_path):
        filepath = str(tmp_path / "model.json")
        save_model(sample_model, filepath, context_size=3, mode='pacbio-fiber')

        loaded = load_model_for_inference(filepath, normalize=False)

        assert loaded._log_probs_frozen

    def test_json_contains_expected_keys(self, sample_model, tmp_path):
        filepath = str(tmp_path / "model.json")
        save_model(sample_model, filepath, context_size=3, mode='pacbio-fiber')

        with open(filepath) as f:
            data = json.load(f)

        assert data['model_type'] == 'FiberHMM'
        assert data['version'] == '2.0'
        assert data['n_states'] == 2
        assert 'startprob' in data
        assert 'transmat' in data
        assert 'emissionprob' in data
        assert data['context_size'] == 3
        assert data['mode'] == 'pacbio-fiber'

    def test_model_json_record_uses_plain_serializable_values(self, sample_model):
        record = _model_json_record(sample_model, context_size=5, mode="daf")

        assert record["model_type"] == "FiberHMM"
        assert record["version"] == "2.0"
        assert record["n_states"] == 2
        assert record["context_size"] == 5
        assert record["mode"] == "daf"
        assert record["startprob"] == sample_model.startprob_.tolist()
        assert record["transmat"] == sample_model.transmat_.tolist()
        assert record["emissionprob"] == sample_model.emissionprob_.tolist()

    def test_legacy_pickle_load_unfreezes_log_cache(self, tmp_path):
        """Loaded public models should recompute logs unless workers re-freeze them."""
        model = FiberHMM(n_states=2)
        model.startprob_ = np.array([0.5, 0.5])
        model.transmat_ = np.array([[0.5, 0.5], [0.5, 0.5]])
        model.emissionprob_ = np.array([[0.9, 0.1], [0.1, 0.9]])
        model.freeze_log_probs()

        filepath = str(tmp_path / "legacy.pkl")
        with open(filepath, "wb") as f:
            pickle.dump(model, f)

        loaded = load_model(filepath, normalize=False)
        obs = np.zeros(4, dtype=np.int64)
        before = loaded.predict(obs)
        loaded.emissionprob_[:] = np.array([[0.1, 0.9], [0.9, 0.1]])
        after = loaded.predict(obs)

        assert not loaded._log_probs_frozen
        np.testing.assert_array_equal(before, np.zeros(4, dtype=np.int8))
        np.testing.assert_array_equal(after, np.ones(4, dtype=np.int8))

    def test_legacy_metadata_pickle_load_unfreezes_log_cache(self, tmp_path):
        model = FiberHMM(n_states=2)
        model.startprob_ = np.array([0.5, 0.5])
        model.transmat_ = np.array([[0.5, 0.5], [0.5, 0.5]])
        model.emissionprob_ = np.array([[0.9, 0.1], [0.1, 0.9]])
        model.freeze_log_probs()

        filepath = str(tmp_path / "legacy_with_metadata.pkl")
        with open(filepath, "wb") as f:
            pickle.dump({"model": model, "context_size": 5, "mode": "nanopore"}, f)

        loaded, context_size, mode = load_model_with_metadata(filepath, normalize=False)

        assert context_size == 5
        assert mode == "nanopore-fiber"
        assert not loaded._log_probs_frozen

    def test_metadata_pickle_accepts_native_dict_model(self, sample_model, tmp_path):
        filepath = str(tmp_path / "native_dict_with_metadata.pkl")
        with open(filepath, "wb") as f:
            pickle.dump(
                {"model": sample_model.to_dict(), "context_size": 5, "mode": "m6a"},
                f,
            )

        loaded, context_size, mode = load_model_with_metadata(filepath, normalize=False)

        assert context_size == 5
        assert mode == "pacbio-fiber"
        np.testing.assert_allclose(loaded.startprob_, sample_model.startprob_)
        np.testing.assert_allclose(loaded.transmat_, sample_model.transmat_)
        np.testing.assert_allclose(loaded.emissionprob_, sample_model.emissionprob_)

    def test_pickle_payload_has_metadata_identifies_wrapped_model_payloads(self):
        assert _pickle_payload_has_metadata({"model": object()})
        assert not _pickle_payload_has_metadata({"model_type": "FiberHMM_native"})
        assert not _pickle_payload_has_metadata(object())


class TestModeAliases:
    def test_nanopore_alias(self, sample_model, tmp_path):
        filepath = str(tmp_path / "model.json")
        # Manually write a JSON with old mode name
        data = {
            'model_type': 'FiberHMM',
            'version': '2.0',
            'n_states': 2,
            'startprob': sample_model.startprob_.tolist(),
            'transmat': sample_model.transmat_.tolist(),
            'emissionprob': sample_model.emissionprob_.tolist(),
            'context_size': 3,
            'mode': 'nanopore',
        }
        with open(filepath, 'w') as f:
            json.dump(data, f)

        _, _, mode = load_model_with_metadata(filepath, normalize=False)
        assert mode == 'nanopore-fiber'

    def test_pacbio_alias(self, sample_model, tmp_path):
        filepath = str(tmp_path / "model.json")
        data = {
            'model_type': 'FiberHMM',
            'version': '2.0',
            'n_states': 2,
            'startprob': sample_model.startprob_.tolist(),
            'transmat': sample_model.transmat_.tolist(),
            'emissionprob': sample_model.emissionprob_.tolist(),
            'context_size': 3,
            'mode': 'pacbio',
        }
        with open(filepath, 'w') as f:
            json.dump(data, f)

        _, _, mode = load_model_with_metadata(filepath, normalize=False)
        assert mode == 'pacbio-fiber'

    def test_m6a_alias(self, sample_model, tmp_path):
        filepath = str(tmp_path / "model.json")
        data = {
            'model_type': 'FiberHMM',
            'version': '2.0',
            'n_states': 2,
            'startprob': sample_model.startprob_.tolist(),
            'transmat': sample_model.transmat_.tolist(),
            'emissionprob': sample_model.emissionprob_.tolist(),
            'context_size': 3,
            'mode': 'm6a',
        }
        with open(filepath, 'w') as f:
            json.dump(data, f)

        _, _, mode = load_model_with_metadata(filepath, normalize=False)
        assert mode == 'pacbio-fiber'

    def test_stranded_alias(self, sample_model, tmp_path):
        filepath = str(tmp_path / "model.json")
        data = {
            'model_type': 'FiberHMM',
            'version': '2.0',
            'n_states': 2,
            'startprob': sample_model.startprob_.tolist(),
            'transmat': sample_model.transmat_.tolist(),
            'emissionprob': sample_model.emissionprob_.tolist(),
            'context_size': 3,
            'mode': 'stranded',
        }
        with open(filepath, 'w') as f:
            json.dump(data, f)

        _, _, mode = load_model_with_metadata(filepath, normalize=False)
        assert mode == 'nanopore-fiber'


class TestSaveRedirect:
    def test_json_save_path_helper_redirects_legacy_extensions(self):
        assert _json_save_path("/tmp/model.json") == (
            "/tmp/model.json",
            "/tmp/model.json",
        )
        assert _json_save_path("/tmp/model.npz") == (
            "/tmp/model.json",
            "/tmp/model.npz",
        )

    def test_json_save_redirect_warning_names_both_paths(self):
        message = _json_save_redirect_warning("/tmp/model.json", "/tmp/model.npz")

        assert "Only JSON format is supported for saving" in message
        assert "/tmp/model.json" in message
        assert "/tmp/model.npz" in message

    def test_npz_redirects_to_json(self, sample_model, tmp_path):
        npz_path = str(tmp_path / "model.npz")
        json_path = str(tmp_path / "model.json")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            save_model(sample_model, npz_path, context_size=3, mode='pacbio-fiber')
            assert len(w) == 1
            assert "JSON format" in str(w[0].message)

        assert os.path.exists(json_path)
        assert not os.path.exists(npz_path)

    def test_pickle_redirects_to_json(self, sample_model, tmp_path):
        pkl_path = str(tmp_path / "model.pickle")
        json_path = str(tmp_path / "model.json")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            save_model(sample_model, pkl_path, context_size=3, mode='pacbio-fiber')
            assert len(w) == 1
            assert "JSON format" in str(w[0].message)

        assert os.path.exists(json_path)
        assert not os.path.exists(pkl_path)

    def test_redirected_model_loadable(self, sample_model, tmp_path):
        npz_path = str(tmp_path / "model.npz")
        json_path = str(tmp_path / "model.json")

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            save_model(sample_model, npz_path, context_size=3, mode='pacbio-fiber')

        loaded = load_model(json_path, normalize=False)
        np.testing.assert_allclose(loaded.startprob_, sample_model.startprob_, rtol=1e-5)
