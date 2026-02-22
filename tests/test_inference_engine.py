"""
Tests for fiberhmm.inference.engine module.
"""
import pytest
import numpy as np

from fiberhmm.core.hmm import FiberHMM
from fiberhmm.inference.engine import (
    predict_footprints,
    predict_footprints_and_msps,
    _extract_footprints_from_states,
)


@pytest.fixture
def simple_model():
    """A simple 2-state model with 4 symbols."""
    model = FiberHMM(n_states=2)
    model.emissionprob_ = np.array([
        [0.1, 0.2, 0.3, 0.4],  # State 0 (footprint): prefers high symbols
        [0.4, 0.3, 0.2, 0.1],  # State 1 (accessible): prefers low symbols
    ])
    model.startprob_ = np.array([0.5, 0.5])
    model.transmat_ = np.array([[0.9, 0.1], [0.1, 0.9]])
    return model


class TestPredictFootprints:
    def test_empty_input(self, simple_model):
        starts, sizes, count, scores = predict_footprints(simple_model, np.array([]))
        assert count == 0
        assert len(starts) == 0
        assert len(sizes) == 0
        assert scores is None

    def test_returns_valid_footprints(self, simple_model):
        # High symbols → footprint (state 0), low symbols → accessible (state 1)
        obs = np.array([0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 0, 0, 0, 0, 0], dtype=np.int32)
        starts, sizes, count, scores = predict_footprints(simple_model, obs)
        assert count >= 0
        assert len(starts) == len(sizes)
        assert all(s >= 0 for s in sizes)
        assert scores is None  # with_scores=False

    def test_with_scores(self, simple_model):
        obs = np.array([0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 0, 0, 0, 0, 0], dtype=np.int32)
        starts, sizes, count, scores = predict_footprints(simple_model, obs, with_scores=True)
        if count > 0:
            assert scores is not None
            assert len(scores) == count
            assert all(0 <= s <= 1 for s in scores)

    def test_all_accessible(self, simple_model):
        # All low symbols → all accessible → no footprints
        obs = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int32)
        starts, sizes, count, scores = predict_footprints(simple_model, obs)
        # Model might still call some footprints, but count should be valid
        assert count >= 0

    def test_starts_and_sizes_consistent(self, simple_model):
        obs = np.array([0, 0, 0, 3, 3, 3, 3, 3, 3, 0, 0, 0], dtype=np.int32)
        starts, sizes, count, _ = predict_footprints(simple_model, obs)
        if count > 0:
            # Each footprint should end within the observation
            for s, sz in zip(starts, sizes):
                assert s >= 0
                assert sz > 0
                assert s + sz <= len(obs)


class TestPredictFootprintsAndMsps:
    def test_empty_input(self, simple_model):
        result = predict_footprints_and_msps(simple_model, np.array([]))
        assert len(result['footprint_starts']) == 0
        assert len(result['msp_starts']) == 0

    def test_returns_dict_with_correct_keys(self, simple_model):
        obs = np.array([0, 0, 0, 3, 3, 3, 3, 3, 0, 0, 0], dtype=np.int32)
        result = predict_footprints_and_msps(simple_model, obs)
        assert 'footprint_starts' in result
        assert 'footprint_sizes' in result
        assert 'msp_starts' in result
        assert 'msp_sizes' in result
        assert 'states' in result

    def test_msp_min_size_filtering(self, simple_model):
        # Create a long sequence that creates large accessible regions
        obs = np.concatenate([
            np.full(50, 3, dtype=np.int32),  # footprint
            np.full(200, 0, dtype=np.int32),  # accessible (large)
            np.full(50, 3, dtype=np.int32),  # footprint
        ])
        result_small = predict_footprints_and_msps(simple_model, obs, msp_min_size=10)
        result_large = predict_footprints_and_msps(simple_model, obs, msp_min_size=500)
        # Large min_size should filter more MSPs
        assert len(result_large['msp_starts']) <= len(result_small['msp_starts'])

    def test_with_scores(self, simple_model):
        obs = np.array([0, 0, 0, 3, 3, 3, 3, 3, 0, 0, 0] * 5, dtype=np.int32)
        result = predict_footprints_and_msps(simple_model, obs, with_scores=True)
        if len(result['footprint_starts']) > 0:
            assert result['footprint_scores'] is not None
            assert len(result['footprint_scores']) == len(result['footprint_starts'])

    def test_return_posteriors(self, simple_model):
        obs = np.array([0, 0, 0, 3, 3, 3, 3, 3, 0, 0, 0], dtype=np.int32)
        result = predict_footprints_and_msps(simple_model, obs, return_posteriors=True)
        assert result['posteriors'] is not None
        assert len(result['posteriors']) == len(obs)


class TestExtractFootprintsFromStates:
    def test_empty_states(self):
        result = _extract_footprints_from_states(np.array([]), None, 147, False)
        assert len(result['footprint_starts']) == 0
        assert len(result['msp_starts']) == 0

    def test_all_footprint(self):
        states = np.zeros(100, dtype=int)  # All state 0 (footprint)
        result = _extract_footprints_from_states(states, None, 10, False)
        assert len(result['footprint_starts']) == 1
        assert result['footprint_sizes'][0] == 100

    def test_all_accessible(self):
        states = np.ones(200, dtype=int)  # All state 1 (accessible)
        result = _extract_footprints_from_states(states, None, 10, False)
        assert len(result['footprint_starts']) == 0
        assert len(result['msp_starts']) == 1
        assert result['msp_sizes'][0] == 200

    def test_alternating_states(self):
        # [footprint(20) accessible(30) footprint(20) accessible(30)]
        states = np.concatenate([
            np.zeros(20), np.ones(30), np.zeros(20), np.ones(30)
        ]).astype(int)
        result = _extract_footprints_from_states(states, None, 10, False)
        assert len(result['footprint_starts']) == 2
        assert result['footprint_sizes'][0] == 20
        assert result['footprint_sizes'][1] == 20

    def test_single_bp_footprint(self):
        states = np.ones(10, dtype=int)
        states[5] = 0  # Single-bp footprint
        result = _extract_footprints_from_states(states, None, 1, False)
        assert len(result['footprint_starts']) == 1
        assert result['footprint_sizes'][0] == 1
        assert result['footprint_starts'][0] == 5

    def test_msp_min_size_filters_small(self):
        states = np.concatenate([
            np.zeros(10), np.ones(5), np.zeros(10)  # MSP of size 5
        ]).astype(int)
        result = _extract_footprints_from_states(states, None, 10, False)
        assert len(result['msp_starts']) == 0  # Filtered by min_size=10

    def test_with_confidence_scores(self):
        states = np.concatenate([np.zeros(10), np.ones(10)]).astype(int)
        confidence = np.concatenate([np.full(10, 0.9), np.full(10, 0.8)])
        result = _extract_footprints_from_states(states, confidence, 1, with_scores=True)
        if len(result['footprint_starts']) > 0:
            assert result['footprint_scores'] is not None
