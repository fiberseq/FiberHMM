from __future__ import annotations

import numpy as np

from fiberhmm.inference.circular import (
    _center_copy_bounds,
    _legacy_interval_score,
    _linear_segments_overlap,
    _tiled_mod_positions,
    circular_intervals_overlap,
    project_center_nuc_calls,
    project_center_runs,
    project_center_scores,
    project_center_tf_calls,
    split_intervals_for_legacy,
    tile_sequence_and_mods,
)
from fiberhmm.inference.engine import (
    _extract_footprints_from_states_circular,
    _project_circular_result_track,
)
from fiberhmm.inference.nuc_recaller import NucCall
from fiberhmm.inference.tf_recaller import TFCall
from fiberhmm.io.ma_tags import (
    format_an_tag,
    interval_wraps,
    parse_an_tag,
    split_circular_interval,
)


def test_split_circular_interval_keeps_ma_valid_pieces():
    assert split_circular_interval(20, 10, 100) == [(20, 10)]
    assert split_circular_interval(90, 20, 100) == [(0, 10), (90, 10)]
    assert split_circular_interval(90, 100, 100) == [(0, 100)]
    assert split_circular_interval(0, 0, 100) == []


def test_interval_wraps_detects_origin_crossing():
    assert not interval_wraps(10, 20, 100)
    assert interval_wraps(90, 20, 100)
    assert not interval_wraps(0, 100, 100)


def test_an_tag_roundtrip_uses_dot_for_empty_names():
    an = format_an_tag(["fhw_tf_0", "", None])
    assert an == "fhw_tf_0,.,."
    assert parse_an_tag(an) == ["fhw_tf_0", "", ""]


def test_tiled_mod_positions_filters_invalid_positions():
    assert _tiled_mod_positions(2, 10) == (2, 12, 22)
    assert _tiled_mod_positions(-1, 10) == ()
    assert _tiled_mod_positions(10, 10) == ()


def test_tile_sequence_and_mods_replicates_positions_across_three_copies():
    seq, mods = tile_sequence_and_mods("ACGT", {1, 3, 99})
    assert seq == "ACGTACGTACGT"
    assert mods == {1, 3, 5, 7, 9, 11}


def test_center_copy_bounds_selects_middle_tile():
    assert _center_copy_bounds(100) == (100, 200)


def test_project_center_runs_projects_wrapped_middle_copy():
    starts = np.asarray([20, 95, 195, 220])
    ends = np.asarray([40, 110, 215, 240])

    assert project_center_runs(starts, ends, 100) == [(95, 20)]


def test_project_center_scores_matches_projected_runs():
    starts = np.asarray([20, 95, 195])
    ends = np.asarray([40, 110, 215])
    scores = np.asarray([0.1, 0.2, 0.9], dtype=np.float32)

    projected = project_center_scores(starts, ends, scores, 100)

    assert np.allclose(projected, [0.9])


def test_project_center_tf_calls_preserves_call_metrics():
    calls = [
        TFCall(start=95, length=20, llr=1.0, n_opps=3, left_ambiguity=4, right_ambiguity=5),
        TFCall(start=195, length=20, llr=7.0, n_opps=6, left_ambiguity=1, right_ambiguity=2),
    ]

    projected = project_center_tf_calls(calls, 100)

    assert projected == [
        TFCall(start=95, length=20, llr=7.0, n_opps=6, left_ambiguity=1, right_ambiguity=2)
    ]


def test_project_center_nuc_calls_preserves_quality_bytes():
    calls = [
        NucCall(start=95, length=20, nq=1, el=2, er=3),
        NucCall(start=195, length=20, nq=7, el=8, er=9),
    ]

    projected = project_center_nuc_calls(calls, 100)

    assert projected == [NucCall(start=95, length=20, nq=7, el=8, er=9)]


def test_legacy_interval_score_defaults_missing_scores_to_none():
    assert _legacy_interval_score([0.25], 0) == 0.25
    assert _legacy_interval_score([0.25], 1) is None
    assert _legacy_interval_score(None, 0) is None


def test_split_intervals_for_legacy_duplicates_scores_for_wrapped_features():
    starts, lengths, scores = split_intervals_for_legacy(
        [(90, 20), (30, 5)],
        100,
        [0.8, 0.2],
    )

    assert starts.tolist() == [0, 30, 90]
    assert lengths.tolist() == [10, 5, 10]
    assert np.allclose(scores, [0.8, 0.2, 0.8])


def test_circular_intervals_overlap_across_origin():
    assert circular_intervals_overlap((90, 20), (5, 5), 100)
    assert circular_intervals_overlap((90, 20), (95, 2), 100)
    assert not circular_intervals_overlap((90, 5), (10, 5), 100)


def test_linear_segments_overlap_uses_half_open_lengths():
    assert not _linear_segments_overlap((10, 10), (20, 5))
    assert _linear_segments_overlap((10, 11), (20, 5))


def test_project_circular_result_track_returns_legacy_and_circular_views():
    track = _project_circular_result_track(
        starts=np.asarray([195], dtype=np.int32),
        sizes=np.asarray([20], dtype=np.int32),
        scores=np.asarray([0.9], dtype=np.float32),
        read_length=100,
    )

    assert track["circular_intervals"] == [(95, 20)]
    assert track["legacy_starts"].tolist() == [0, 95]
    assert track["legacy_sizes"].tolist() == [15, 5]
    assert np.allclose(track["legacy_scores"], [0.9, 0.9])
    assert track["tiled_starts"].tolist() == [195]
    assert track["tiled_sizes"].tolist() == [20]


def test_extract_footprints_from_states_circular_projects_middle_copy():
    states = np.ones(300, dtype=np.int8)
    states[195:215] = 0
    confidence = np.full(300, 0.5, dtype=np.float32)
    confidence[195:215] = 0.9

    result = _extract_footprints_from_states_circular(
        states,
        confidence,
        read_length=100,
        msp_min_size=0,
        with_scores=True,
        nuc_min_size=10,
    )

    assert result["circular_ns"] == [(95, 20)]
    assert result["footprint_starts"].tolist() == [0, 95]
    assert result["footprint_sizes"].tolist() == [15, 5]
    assert np.allclose(result["footprint_scores"], [0.9, 0.9])
