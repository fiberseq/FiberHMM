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


def test_warm_up_model_predict_runs_dummy_prediction(monkeypatch):
    monkeypatch.setattr("fiberhmm.core.hmm.HAS_NUMBA", True)

    model = RecordingModel()
    worker_warmup.warm_up_model_predict(model)

    assert model.predict_lengths == [4]
    assert model.predict_proba_lengths == []


def test_warm_up_model_predict_skips_without_numba(monkeypatch):
    monkeypatch.setattr("fiberhmm.core.hmm.HAS_NUMBA", False)

    model = RecordingModel()
    worker_warmup.warm_up_model_predict(model)

    assert model.predict_lengths == []


def test_warm_up_model_predict_ignores_warmup_failures(monkeypatch):
    monkeypatch.setattr("fiberhmm.core.hmm.HAS_NUMBA", True)

    class FailingModel:
        def predict(self, obs):
            raise RuntimeError("predict failed")

    worker_warmup.warm_up_model_predict(FailingModel())


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


def test_warm_up_tf_recaller_runs_dummy_scan(monkeypatch):
    monkeypatch.setattr("fiberhmm.core.hmm.HAS_NUMBA", True)
    calls = []

    def fake_call(obs, lo, hi, llr_hit, llr_miss, **kwargs):
        calls.append((obs, lo, hi, llr_hit, llr_miss, kwargs))
        return []

    monkeypatch.setattr(
        "fiberhmm.inference.tf_recaller.call_tfs_in_interval",
        fake_call,
    )
    llr_hit = np.array([1.0])
    llr_miss = np.array([-1.0])

    worker_warmup.warm_up_tf_recaller(llr_hit, llr_miss)

    assert len(calls) == 1
    obs, lo, hi, got_hit, got_miss, kwargs = calls[0]
    assert obs.shape == (16,)
    assert lo == 0
    assert hi == 16
    assert got_hit is llr_hit
    assert got_miss is llr_miss
    assert kwargs == {"min_llr": 4.0, "min_opps": 3}


def test_warm_up_tf_recaller_skips_without_numba(monkeypatch):
    monkeypatch.setattr("fiberhmm.core.hmm.HAS_NUMBA", False)
    calls = []
    monkeypatch.setattr(
        "fiberhmm.inference.tf_recaller.call_tfs_in_interval",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    worker_warmup.warm_up_tf_recaller(np.array([1.0]), np.array([-1.0]))

    assert calls == []


def test_warm_up_tf_recaller_ignores_warmup_failures(monkeypatch):
    monkeypatch.setattr("fiberhmm.core.hmm.HAS_NUMBA", True)

    def fail_call(*args, **kwargs):
        raise RuntimeError("tf warmup failed")

    monkeypatch.setattr(
        "fiberhmm.inference.tf_recaller.call_tfs_in_interval",
        fail_call,
    )

    worker_warmup.warm_up_tf_recaller(np.array([1.0]), np.array([-1.0]))
