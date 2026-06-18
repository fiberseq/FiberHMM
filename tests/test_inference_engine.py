"""
Tests for fiberhmm.inference.engine module.
"""
import array

import numpy as np
import pytest

import fiberhmm.inference.engine as engine
from fiberhmm.core.hmm import FiberHMM
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


def test_base_single_read_fields_maps_linear_prediction_fields():
    fields = engine._base_single_read_fields({
        "footprint_starts": np.array([1]),
        "footprint_sizes": np.array([10]),
        "footprint_scores": np.array([0.8]),
        "msp_starts": np.array([20]),
        "msp_sizes": np.array([5]),
    })

    np.testing.assert_array_equal(fields["ns"], [1])
    np.testing.assert_array_equal(fields["nl"], [10])
    np.testing.assert_array_equal(fields["ns_scores"], [0.8])
    np.testing.assert_array_equal(fields["as"], [20])
    np.testing.assert_array_equal(fields["al"], [5])
    assert fields["as_scores"] is None


def test_circular_single_read_fields_maps_circular_prediction_fields():
    fields = engine._circular_single_read_fields({
        "circular_read_length": 100,
        "circular_ns": [(1, 10)],
        "circular_as": [(20, 5)],
        "circular_ns_scores": [0.8],
        "circular_as_scores": [0.6],
        "tiled_ns": [101],
        "tiled_nl": [10],
        "tiled_as": [120],
        "tiled_al": [5],
    })

    assert fields == {
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
    }


def test_encoding_inputs_for_read_tiles_only_circular_reads():
    inputs = engine._encoding_inputs_for_read(
        "ACGT",
        {1, 3},
        circular=False,
    )
    assert inputs.sequence == "ACGT"
    assert inputs.mod_positions == {1, 3}
    assert inputs.circular_read_length is None

    inputs = engine._encoding_inputs_for_read(
        "ACGT",
        {1, 3},
        circular=True,
    )
    assert inputs.sequence == "ACGTACGTACGT"
    assert inputs.mod_positions == {1, 3, 5, 7, 9, 11}
    assert inputs.circular_read_length == 4


def test_processing_strand_for_read_prefers_daf_tag_and_defaults(monkeypatch):
    assert engine._processing_strand_for_read(
        {"_daf_strand": "-"},
        "ACGT",
        {1},
        mode="daf",
    ) == "-"

    monkeypatch.setattr(engine, "detect_daf_strand", lambda seq, positions: "+")
    assert engine._processing_strand_for_read(
        {},
        "ACGT",
        {1},
        mode="daf",
    ) == "+"
    assert engine._processing_strand_for_read(
        {},
        "ACGT",
        {1},
        mode="pacbio-fiber",
    ) == "."


def test_should_skip_empty_prediction_respects_optional_outputs():
    empty = {
        "footprint_starts": np.array([]),
        "msp_starts": np.array([]),
    }
    nonempty = {
        "footprint_starts": np.array([1]),
        "msp_starts": np.array([]),
    }

    assert engine._should_skip_empty_prediction(
        empty, return_posteriors=False, include_encoded=False,
    )
    assert not engine._should_skip_empty_prediction(
        empty, return_posteriors=True, include_encoded=False,
    )
    assert not engine._should_skip_empty_prediction(
        empty, return_posteriors=False, include_encoded=True,
    )
    assert not engine._should_skip_empty_prediction(
        nonempty, return_posteriors=False, include_encoded=False,
    )


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

        prediction = engine._predict_state_outputs(
            FakeModel(), np.array([2, 3]), with_scores=False, return_posteriors=False,
        )

        np.testing.assert_array_equal(prediction.states, [1, 0])
        assert prediction.confidence is None
        assert prediction.posteriors is None

    def test_predict_state_outputs_derives_confidence_from_posteriors(self):
        class FakeModel:
            def predict_with_posteriors(self, encoded):
                return (
                    np.array([0, 1], dtype=np.int8),
                    np.array([[0.8, 0.2], [0.1, 0.9]], dtype=np.float32),
                )

        prediction = engine._predict_state_outputs(
            FakeModel(), np.array([2, 3]), with_scores=True, return_posteriors=False,
        )

        np.testing.assert_array_equal(prediction.states, [0, 1])
        np.testing.assert_allclose(prediction.confidence, [0.8, 0.9])
        np.testing.assert_allclose(
            prediction.posteriors,
            [[0.8, 0.2], [0.1, 0.9]],
        )

    def test_footprint_posterior_track_extracts_first_column_as_float16(self):
        posteriors = np.array([
            [0.1, 0.9],
            [0.2, 0.8],
            [0.3, 0.7],
        ], dtype=np.float32)

        track = engine._footprint_posterior_track(posteriors, 1, 3)

        assert track.dtype == np.float16
        np.testing.assert_allclose(track, [0.2, 0.3], atol=1e-3)

    def test_linear_prediction_result_adds_states_to_interval_result(self):
        states = np.array([1, 0, 0, 1, 1], dtype=np.int8)
        confidence = np.array([0.7, 0.8, 0.9, 0.6, 0.5], dtype=np.float32)

        result = engine._linear_prediction_result(
            engine._PredictionOutputs(states, confidence, None),
            msp_min_size=1,
            with_scores=True,
            nuc_min_size=1,
        )
        expected = engine._extract_footprints_from_states(
            states,
            confidence,
            msp_min_size=1,
            with_scores=True,
            nuc_min_size=1,
        )

        assert result["states"] is states
        for key, expected_value in expected.items():
            if isinstance(expected_value, np.ndarray):
                np.testing.assert_array_equal(result[key], expected_value)
            else:
                assert result[key] == expected_value

    def test_center_copy_states_returns_middle_tile_as_int8(self):
        states = np.arange(12, dtype=np.int64)
        center = engine._center_copy_states(states, read_length=4)

        assert center.dtype == np.int8
        np.testing.assert_array_equal(center, [4, 5, 6, 7])

    def test_nuc_boundaries_from_footprint_runs_filters_small_runs(self):
        boundaries = engine._nuc_boundaries_from_footprint_runs(
            np.array([0, 20, 100], dtype=np.int64),
            np.array([10, 120, 130], dtype=np.int64),
            nuc_min_size=85,
        )

        np.testing.assert_array_equal(boundaries.starts, [20])
        np.testing.assert_array_equal(boundaries.ends, [120])

    def test_msp_intervals_from_nuc_boundaries_names_starts_and_sizes(self):
        intervals = engine._msp_intervals_from_nuc_boundaries(
            np.array([10, 30], dtype=np.int64),
            np.array([20, 40], dtype=np.int64),
            read_length=50,
            msp_min_size=10,
        )

        np.testing.assert_array_equal(intervals.starts, [0, 20, 40])
        np.testing.assert_array_equal(intervals.sizes, [10, 10, 10])

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

    def test_mode_detection_read_indicator_helpers_update_counts(self):
        class FakeRead:
            query_sequence = "ACYR"

            def __init__(self, tags):
                self.tags = tags

            def has_tag(self, tag):
                return tag in self.tags

            def get_tag(self, tag):
                if tag not in self.tags:
                    raise KeyError(tag)
                return self.tags[tag]

        counts = engine._new_mode_detection_counts()
        read = FakeRead({"st": "CT", "MM": "T-a,0;"})

        engine._record_iupac_mode_indicators(counts, read)
        engine._record_mm_mode_indicators(counts, read)

        assert counts["iupac"] == 1
        assert counts["st"] == 1
        assert counts["reads_with_mm"] == 1
        assert counts["t_minus_a"] == 1

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

    def test_apply_payload_tags_uses_named_tag_set(self):
        read = self.FakeRead(
            "AAAA",
            {
                "MM": "A+a,0;",
                "Ml": array.array("B", [201]),
                "st": "CT",
                "RG": "ignored",
            },
        )

        assert engine._apply_payload_tags(read) == {
            "MM": "A+a,0;",
            "Ml": bytes([201]),
            "st": "CT",
        }
        assert set(engine._APPLY_PAYLOAD_TAGS) == {"MM", "Mm", "ML", "Ml", "st"}

    def test_apply_payload_tag_value_compacts_ml_only(self):
        read = self.FakeRead(
            "AAAA",
            {
                "MM": "A+a,0;",
                "ML": array.array("B", [200]),
            },
        )

        assert engine._apply_payload_tag_value(read, "MM") == "A+a,0;"
        assert engine._apply_payload_tag_value(read, "ML") == bytes([200])

    def test_apply_payload_daf_md_result_skips_iupac_and_delegates_raw_reads(
        self, monkeypatch,
    ):
        calls = []
        read = self.FakeRead("AAAA")
        ref_fasta = object()

        def fake_get_daf_positions(got_read, ref_fasta=None):
            calls.append((got_read, ref_fasta))
            return ([1], [], "CT")

        import fiberhmm.daf.encoder as daf_encoder

        monkeypatch.setattr(daf_encoder, "get_daf_positions", fake_get_daf_positions)

        assert engine._apply_payload_daf_md_result(read, "AYAA", ref_fasta) is None
        assert engine._apply_payload_daf_md_result(read, "AAAA", ref_fasta) == (
            [1],
            [],
            "CT",
        )
        assert calls == [(read, ref_fasta)]

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

    def test_extract_mm_ml_fiber_read_skips_empty_numpy_ml(self, monkeypatch):
        read = self.FakeRead(
            "AAAA",
            {
                "MM": "A+a,0;",
                "ML": np.asarray([], dtype=np.uint8),
            },
        )

        monkeypatch.setattr(
            engine,
            "parse_mm_tag_query_positions",
            lambda *args, **kwargs: pytest.fail("parser should not run"),
        )

        assert engine._extract_mm_ml_fiber_read(
            read,
            "AAAA",
            mode="pacbio-fiber",
            prob_threshold=125,
        ) is None
