"""Tests for shared worker warmup helpers."""

import numpy as np

from fiberhmm.inference import worker_warmup


class RecordingModel:
    def __init__(self):
        self.predict_lengths = []
        self.predict_proba_lengths = []

    def predict(self, obs):
        self.predict_lengths.append(len(obs))
        return np.zeros(len(obs), dtype=np.int8)

    def predict_proba(self, obs):
        self.predict_proba_lengths.append(len(obs))
        return np.zeros((len(obs), 2), dtype=np.float32)


def test_posterior_warmup_obs_uses_int32_and_min_length():
    obs = worker_warmup._posterior_warmup_obs(0)

    assert obs.dtype == np.int32
    assert obs.tolist() == [0]
    assert worker_warmup._posterior_warmup_obs(3).tolist() == [0, 0, 0]


def test_tf_warmup_obs_uses_expected_dummy_window():
    obs = worker_warmup._tf_warmup_obs()

    assert obs.dtype == np.int32
    assert obs.shape == (16,)
    assert obs.sum() == 0


def test_warm_up_model_posteriors_runs_predict_and_proba(monkeypatch):
    monkeypatch.setattr("fiberhmm.core.hmm.HAS_NUMBA", True)

    model = RecordingModel()
    worker_warmup.warm_up_model_posteriors(model, length=7)

    assert model.predict_lengths == [7]
    assert model.predict_proba_lengths == [7]


def test_warm_up_model_posteriors_skips_without_numba(monkeypatch):
    monkeypatch.setattr("fiberhmm.core.hmm.HAS_NUMBA", False)

    model = RecordingModel()
    worker_warmup.warm_up_model_posteriors(model)

    assert model.predict_lengths == []
    assert model.predict_proba_lengths == []
