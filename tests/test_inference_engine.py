"""
Tests for fiberhmm.inference.engine module.
"""
import array

import numpy as np
import pytest

from fiberhmm.core.hmm import FiberHMM
import fiberhmm.inference.engine as engine
from fiberhmm.inference.engine import (
    _extract_fiber_read_from_pysam,
    _extract_footprints_from_states,
    _footprint_runs,
    detect_mode_from_bam,
    predict_footprints,
    predict_footprints_and_msps,
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


def test_single_read_result_from_prediction_includes_optional_fields():
    fp_result = {
        "footprint_starts": np.array([1]),
        "footprint_sizes": np.array([10]),
        "footprint_scores": np.array([0.8]),
        "msp_starts": np.array([20]),
        "msp_sizes": np.array([5]),
        "msp_scores": np.array([0.6]),
        "circular": True,
        "circular_read_length": 100,
        "circular_ns": [(1, 10)],
        "circular_as": [(20, 5)],
        "circular_ns_scores": [0.8],
        "circular_as_scores": [0.6],
        "tiled_ns": [101],
        "tiled_nl": [10],
        "tiled_as": [120],
        "tiled_al": [5],
        "posteriors": np.array([0.1, 0.9]),
    }
    encoded = np.array([1, 2, 3])

    result = engine._single_read_result_from_prediction(
        fp_result, strand="+", encoded=encoded,
        return_posteriors=True, include_encoded=True,
    )

    np.testing.assert_array_equal(result["ns"], [1])
    np.testing.assert_array_equal(result["nl"], [10])
    assert result["circular"] is True
    assert result["circular_read_length"] == 100
    assert result["strand"] == "+"
    np.testing.assert_array_equal(result["posteriors"], [0.1, 0.9])
    np.testing.assert_array_equal(result["encoded"], [1, 2, 3])


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

    def test_predict_state_outputs_uses_plain_predict_when_possible(self):
        class FakeModel:
            def predict(self, encoded):
                return np.array([1, 0], dtype=np.int8)

            def predict_with_posteriors(self, encoded):
                raise AssertionError("posterior path should not run")

        states, confidence, posteriors = engine._predict_state_outputs(
            FakeModel(), np.array([2, 3]), with_scores=False, return_posteriors=False,
        )

        np.testing.assert_array_equal(states, [1, 0])
        assert confidence is None
        assert posteriors is None

    def test_predict_state_outputs_derives_confidence_from_posteriors(self):
        class FakeModel:
            def predict_with_posteriors(self, encoded):
                return (
                    np.array([0, 1], dtype=np.int8),
                    np.array([[0.8, 0.2], [0.1, 0.9]], dtype=np.float32),
                )

        states, confidence, posteriors = engine._predict_state_outputs(
            FakeModel(), np.array([2, 3]), with_scores=True, return_posteriors=False,
        )

        np.testing.assert_array_equal(states, [0, 1])
        np.testing.assert_allclose(confidence, [0.8, 0.9])
        np.testing.assert_allclose(posteriors, [[0.8, 0.2], [0.1, 0.9]])

    def test_footprint_posterior_track_extracts_first_column_as_float16(self):
        posteriors = np.array([
            [0.1, 0.9],
            [0.2, 0.8],
            [0.3, 0.7],
        ], dtype=np.float32)

        track = engine._footprint_posterior_track(posteriors, 1, 3)

        assert track.dtype == np.float16
        np.testing.assert_allclose(track, [0.2, 0.3], atol=1e-3)

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
    @pytest.mark.parametrize(
        "states",
        [
            np.array([], dtype=np.int32),
            np.ones(5, dtype=np.int32),
            np.zeros(5, dtype=np.int32),
            np.array([1, 0, 0, 1, 0, 1], dtype=np.int32),
            np.array([0, 1, 0, 0, 1, 0], dtype=np.int64),
        ],
    )
    def test_footprint_runs_matches_padded_diff_oracle(self, states):
        states_padded = np.concatenate([[1], states, [1]])
        diff = np.diff(states_padded)
        expected_starts = np.where(diff == -1)[0]
        expected_ends = np.where(diff == 1)[0]

        starts, ends = _footprint_runs(states)

        np.testing.assert_array_equal(starts, expected_starts)
        np.testing.assert_array_equal(ends, expected_ends)

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
        # nuc_min_size=1 so 10bp footprints count as MSP boundaries
        result = _extract_footprints_from_states(states, None, 10, False, nuc_min_size=1)
        assert len(result['msp_starts']) == 0  # Filtered by min_size=10

    def test_with_confidence_scores(self):
        states = np.concatenate([np.zeros(10), np.ones(10)]).astype(int)
        confidence = np.concatenate([np.full(10, 0.9), np.full(10, 0.8)])
        result = _extract_footprints_from_states(states, confidence, 1, with_scores=True)
        if len(result['footprint_starts']) > 0:
            assert result['footprint_scores'] is not None


class TestModeDetection:
    def test_mode_detection_helpers_count_mm_specs(self):
        counts = engine._new_mode_detection_counts()

        engine._record_mm_mode_specs(counts, "T-a,0;A+a,0;C+m,0;Z+z,0;malformed;")

        assert counts["t_minus_a"] == 1
        assert counts["a_plus_a"] == 1
        assert counts["c_plus_m"] == 1
        assert counts["other"] == 1

    def test_mode_detection_helper_uses_iupac_when_no_mm_tags(self):
        counts = engine._new_mode_detection_counts()
        counts["iupac"] = 1
        counts["st"] = 1

        assert engine._mode_from_detection_counts(counts) == "daf"

    def test_detect_mode_uses_valid_mm_specs_and_ignores_malformed(self, monkeypatch):
        class FakeRead:
            is_unmapped = False
            query_sequence = "A" * 20

            def __init__(self, mm_tag):
                self._mm_tag = mm_tag

            def has_tag(self, tag):
                return tag in {"MM", "Mm"}

            def get_tag(self, tag):
                if tag in {"MM", "Mm"}:
                    return self._mm_tag
                raise KeyError(tag)

        class FakeBam:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def fetch(self, until_eof=False):
                return iter([FakeRead("malformed;T-a,0;")])

        monkeypatch.setattr(
            engine.pysam,
            "AlignmentFile",
            lambda *args, **kwargs: FakeBam(),
        )

        assert detect_mode_from_bam("fake.bam", n_sample=1) == "daf"


class TestFiberReadExtraction:
    class FakeRead:
        query_name = "read1"
        is_reverse = False

        def __init__(self, sequence, tags=None):
            self.query_sequence = sequence
            self._tags = tags or {}

        def has_tag(self, tag):
            return tag in self._tags

        def get_tag(self, tag):
            if tag not in self._tags:
                raise KeyError(tag)
            return self._tags[tag]

    def test_extract_fiber_read_prefers_daf_iupac_branch(self):
        read = self.FakeRead("ACYR", {"st": "CT"})

        fiber_read = _extract_fiber_read_from_pysam(
            read,
            mode="daf",
            prob_threshold=125,
        )

        assert fiber_read == {
            "read_id": "read1",
            "query_sequence": "ACTA",
            "m6a_query_positions": {2, 3},
            "query_length": 4,
            "_daf_strand": "+",
        }

    def test_make_apply_payload_compacts_ml_tag(self):
        read = self.FakeRead(
            "AAAA",
            {
                "MM": "A+a,0;",
                "ML": array.array("B", [200]),
                "st": "CT",
            },
        )

        payload = engine.make_apply_payload(read)

        assert payload["query_name"] == "read1"
        assert payload["query_sequence"] == "AAAA"
        assert payload["is_reverse"] is False
        assert payload["tags"]["MM"] == "A+a,0;"
        assert payload["tags"]["ML"] == bytes([200])
        assert payload["tags"]["st"] == "CT"

    def test_extract_fiber_read_mm_ml_branch_preserves_reverse_flag(self):
        read = self.FakeRead(
            "AAAA",
            {
                "MM": "T+a,0;",
                "ML": bytes([200]),
            },
        )
        read.is_reverse = True

        fiber_read = _extract_fiber_read_from_pysam(
            read,
            mode="pacbio-fiber",
            prob_threshold=125,
        )

        assert fiber_read["read_id"] == "read1"
        assert fiber_read["query_sequence"] == "AAAA"
        assert fiber_read["query_length"] == 4
        assert fiber_read["is_reverse"] is True
        assert fiber_read["m6a_query_positions"] == {3}
