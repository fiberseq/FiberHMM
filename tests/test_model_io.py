"""
Tests for fiberhmm.core.model_io module.
"""
import pytest
import numpy as np
import json
import os
import warnings
import tempfile

from fiberhmm.core.hmm import FiberHMM
from fiberhmm.core.model_io import load_model, save_model, load_model_with_metadata


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
