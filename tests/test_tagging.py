"""Tests for shared inference tagging helpers."""

from __future__ import annotations

import array as pyarray

import numpy as np

from fiberhmm.inference.tag_utils import _clear_tag, clear_tags
from fiberhmm.inference.tagging import (
    STALE_SPEC_TAGS,
    _clear_stale_spec_tags,
    _flip_legacy_intervals_to_molecular,
    _fused_recall_tag_intervals,
    _legacy_apply_interval_groups,
    _legacy_interval_group,
    _linear_intervals_overlap,
    _nuc_overlaps_any_circular_interval,
    _nuc_overlaps_any_linear_interval,
    _result_intervals,
    _should_keep_nuc_interval,
    _tf_circular_intervals,
    _tf_linear_intervals,
    scores_to_u8,
    set_legacy_apply_tags,
    unify_circular_nucs_with_tf_calls,
    unify_nucs_with_tf_calls,
)
from fiberhmm.inference.tf_recaller import TFCall


class RecordingRead:
    def __init__(self):
        self.tags = {}

    def set_tag(self, tag, value, value_type=None):
        if value is None:
            self.tags.pop(tag, None)
        else:
            self.tags[tag] = value

    def has_tag(self, tag):
        return tag in self.tags


def test_scores_to_u8_clips_and_returns_python_ints():
    values = scores_to_u8(np.asarray([-1.0, 0.0, 0.5, 1.0, 2.0]))

    assert values == [0, 0, 127, 255, 255]
    assert all(type(value) is int for value in values)


def test_clear_tag_removes_present_tag_and_ignores_missing_tag():
    read = RecordingRead()
    read.tags = {"MA": "tag"}

    _clear_tag(read, "AQ")
    assert read.tags == {"MA": "tag"}

    _clear_tag(read, "MA")
    assert read.tags == {}


def test_clear_tags_tolerates_set_tag_failures():
    class FailingRead(RecordingRead):
        def set_tag(self, tag, value, value_type=None):
            raise RuntimeError("clear failed")

    read = FailingRead()
    read.tags = {"MA": "tag", "AQ": [1]}

    clear_tags(read, ("MA", "AQ"))

    assert read.tags == {"MA": "tag", "AQ": [1]}


def test_flip_legacy_intervals_to_molecular_sorts_and_reorders_scores():
    group = _flip_legacy_intervals_to_molecular(
        [10, 80],
        [5, 10],
        [0.25, 0.75],
        read_length=100,
    )

    assert group.starts == [10, 85]
    assert group.lengths == [10, 5]
    assert group.scores == [0.75, 0.25]


def test_set_legacy_apply_tags_writes_unsigned_bam_arrays():
    read = RecordingRead()
    result = {
        "ns": np.asarray([10, 50], dtype=np.int32),
        "nl": np.asarray([30, 40], dtype=np.int32),
        "ns_scores": np.asarray([0.0, 1.0]),
        "as": np.asarray([100], dtype=np.int64),
        "al": np.asarray([200], dtype=np.int64),
        "as_scores": np.asarray([0.5]),
    }

    set_legacy_apply_tags(read, result, with_scores=True, write_msps=True)

    for tag in ("ns", "nl", "as", "al"):
        assert isinstance(read.tags[tag], pyarray.array)
        assert read.tags[tag].typecode == "I"
    assert read.tags["ns"].tolist() == [10, 50]
    assert read.tags["nl"].tolist() == [30, 40]
    assert read.tags["as"].tolist() == [100]
    assert read.tags["al"].tolist() == [200]
    assert read.tags["nq"] == [0, 255]
    assert read.tags["aq"] == [127]


def test_set_legacy_apply_tags_clears_stale_scores_when_scores_not_written():
    read = RecordingRead()
    read.tags["nq"] = [1]
    read.tags["aq"] = [2]
    result = {
        "ns": np.asarray([10], dtype=np.int32),
        "nl": np.asarray([30], dtype=np.int32),
        "ns_scores": None,
        "as": np.asarray([100], dtype=np.int32),
        "al": np.asarray([200], dtype=np.int32),
        "as_scores": None,
    }

    set_legacy_apply_tags(read, result, with_scores=False, write_msps=True)

    assert "nq" not in read.tags
    assert "aq" not in read.tags


def test_set_legacy_apply_tags_strips_stale_ma_an_aq_from_prior_recall():
    read = RecordingRead()
    read.tags["MA"] = "100;nuc+Q:1-10;tf+QQQ:30-15"
    read.tags["AN"] = "fh_nuc_0,fh_tf_0"
    read.tags["AQ"] = pyarray.array("B", [200, 180, 20, 30])
    result = {
        "ns": np.asarray([10], dtype=np.int32),
        "nl": np.asarray([30], dtype=np.int32),
        "ns_scores": None,
        "as": np.asarray([100], dtype=np.int32),
        "al": np.asarray([200], dtype=np.int32),
        "as_scores": None,
    }

    set_legacy_apply_tags(read, result, with_scores=False, write_msps=True)

    assert "MA" not in read.tags
    assert "AN" not in read.tags
    assert "AQ" not in read.tags
    assert read.tags["ns"].tolist() == [10]
    assert read.tags["nl"].tolist() == [30]


def test_clear_stale_spec_tags_removes_policy_tag_set():
    read = RecordingRead()
    for tag in STALE_SPEC_TAGS:
        read.tags[tag] = "stale"
    read.tags["ns"] = [10]

    _clear_stale_spec_tags(read)

    for tag in STALE_SPEC_TAGS:
        assert tag not in read.tags
    assert read.tags["ns"] == [10]


def test_set_legacy_apply_tags_strips_stale_ma_an_even_when_no_new_calls():
    read = RecordingRead()
    read.tags["MA"] = "100;nuc+Q:1-10"
    read.tags["AN"] = "fh_nuc_0"
    result = {
        "ns": np.asarray([], dtype=np.int32),
        "nl": np.asarray([], dtype=np.int32),
        "ns_scores": None,
        "as": np.asarray([], dtype=np.int32),
        "al": np.asarray([], dtype=np.int32),
        "as_scores": None,
    }

    set_legacy_apply_tags(read, result, with_scores=False, write_msps=True)

    assert "MA" not in read.tags
    assert "AN" not in read.tags


def test_legacy_apply_interval_groups_prepares_nucs_msps_and_scores():
    read = RecordingRead()
    result = {
        "ns": np.asarray([10], dtype=np.int32),
        "nl": np.asarray([30], dtype=np.int32),
        "ns_scores": np.asarray([0.25]),
        "as": np.asarray([100], dtype=np.int32),
        "al": np.asarray([200], dtype=np.int32),
        "as_scores": np.asarray([0.5]),
    }

    groups = _legacy_apply_interval_groups(result, read, with_scores=True)
    groups_without_scores = _legacy_apply_interval_groups(
        result,
        read,
        with_scores=False,
    )

    nucs = groups.nucs
    msps = groups.msps
    nucs_without_scores = groups_without_scores.nucs
    msps_without_scores = groups_without_scores.msps

    assert nucs.starts == [10]
    assert nucs.lengths == [30]
    assert nucs.scores == [0.25]
    assert msps.starts == [100]
    assert msps.lengths == [200]
    assert msps.scores == [0.5]
    assert nucs_without_scores.starts == [10]
    assert nucs_without_scores.lengths == [30]
    assert nucs_without_scores.scores is None
    assert msps_without_scores.starts == [100]
    assert msps_without_scores.lengths == [200]
    assert msps_without_scores.scores is None


def test_legacy_interval_group_resolves_keyed_arrays_and_scores():
    read = RecordingRead()
    result = {
        "starts": np.asarray([10], dtype=np.int32),
        "lengths": np.asarray([30], dtype=np.int32),
        "scores": np.asarray([0.25]),
    }

    group = _legacy_interval_group(
        result,
        "starts",
        "lengths",
        "scores",
        read,
        with_scores=True,
    )
    group_without_scores = _legacy_interval_group(
        result,
        "starts",
        "lengths",
        "scores",
        read,
        with_scores=False,
    )

    assert group.starts == [10]
    assert group.lengths == [30]
    assert group.scores == [0.25]
    assert group_without_scores.starts == [10]
    assert group_without_scores.lengths == [30]
    assert group_without_scores.scores is None


def test_fused_recall_tag_intervals_prefer_circular_intervals_when_present():
    result = {
        "ns": np.asarray([10], dtype=np.int32),
        "nl": np.asarray([5], dtype=np.int32),
        "as": np.asarray([20], dtype=np.int32),
        "al": np.asarray([6], dtype=np.int32),
    }

    intervals = _fused_recall_tag_intervals(result)

    assert intervals.kept_nucs == [(10, 5)]
    assert intervals.msps == [(20, 6)]

    result["circular_ns"] = [(90, 20)]
    result["circular_as"] = [(80, 10)]

    intervals = _fused_recall_tag_intervals(result)

    assert intervals.kept_nucs == [(90, 20)]
    assert intervals.msps == [(80, 10)]


def test_result_intervals_prefers_circular_key_before_arrays():
    result = {
        "starts": np.asarray([10], dtype=np.int32),
        "lengths": np.asarray([5], dtype=np.int32),
    }

    assert _result_intervals(result, "circular", "starts", "lengths") == [(10, 5)]

    result["circular"] = [(90, 20)]

    assert _result_intervals(result, "circular", "starts", "lengths") == [(90, 20)]


def test_linear_tf_interval_helpers_use_half_open_overlap():
    calls = [
        TFCall(start=20, length=10, llr=5.0, n_opps=3,
               left_ambiguity=0, right_ambiguity=0)
    ]

    assert _tf_linear_intervals(calls) == [(20, 30)]
    assert _linear_intervals_overlap((10, 20), (20, 30)) is False
    assert _linear_intervals_overlap((19, 20), (20, 30)) is False
    assert _linear_intervals_overlap((19, 21), (20, 30)) is True


def test_circular_tf_interval_helpers_use_wrapping_overlap():
    calls = [
        TFCall(start=90, length=20, llr=5.0, n_opps=3,
               left_ambiguity=0, right_ambiguity=0)
    ]

    assert _tf_circular_intervals(calls) == [(90, 20)]
    assert _nuc_overlaps_any_circular_interval((5, 5), [(90, 20)], 100)
    assert _nuc_overlaps_any_circular_interval((80, 10), [(90, 20)], 100) is False


def test_nuc_overlaps_any_linear_interval_uses_start_length_interval():
    tf_intervals = [(20, 30), (50, 60)]

    assert not _nuc_overlaps_any_linear_interval((10, 10), tf_intervals)
    assert _nuc_overlaps_any_linear_interval((19, 2), tf_intervals)


def test_should_keep_nuc_interval_applies_length_and_overlap_policy():
    def overlaps_tf(interval):
        return interval == (20, 10)

    assert not _should_keep_nuc_interval((0, 0), 90, overlaps_tf)
    assert not _should_keep_nuc_interval((20, 10), 90, overlaps_tf)
    assert _should_keep_nuc_interval((20, 100), 90, overlaps_tf)
    assert _should_keep_nuc_interval((40, 10), 90, overlaps_tf)


def test_unify_nucs_with_tf_calls_drops_short_overlaps_and_carries_scores():
    tf_calls = [
        TFCall(start=20, length=10, llr=5.0, n_opps=3,
               left_ambiguity=0, right_ambiguity=0)
    ]

    kept, scores = unify_nucs_with_tf_calls(
        ns=[0, 15, 100],
        nl=[10, 20, 120],
        tf_calls=tf_calls,
        unify_threshold=90,
        ns_scores=np.asarray([0.25, 0.5]),
    )

    assert kept == [(0, 10), (100, 120)]
    assert scores == [63, 0]


def test_unify_circular_nucs_with_tf_calls_handles_origin_overlap():
    tf_calls = [
        TFCall(start=5, length=10, llr=5.0, n_opps=3,
               left_ambiguity=0, right_ambiguity=0)
    ]

    kept, scores = unify_circular_nucs_with_tf_calls(
        nucs=[(90, 20), (40, 120)],
        tf_calls=tf_calls,
        unify_threshold=90,
        read_length=100,
        ns_scores=np.asarray([0.5, 1.0]),
    )

    assert kept == [(40, 120)]
    assert scores == [255]
