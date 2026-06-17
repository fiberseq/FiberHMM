"""Tests for shared inference tagging helpers."""

from __future__ import annotations

import array as pyarray

import numpy as np

from fiberhmm.inference.tagging import (
    _fused_recall_tag_intervals,
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


def test_fused_recall_tag_intervals_prefer_circular_intervals_when_present():
    result = {
        "ns": np.asarray([10], dtype=np.int32),
        "nl": np.asarray([5], dtype=np.int32),
        "as": np.asarray([20], dtype=np.int32),
        "al": np.asarray([6], dtype=np.int32),
    }

    assert _fused_recall_tag_intervals(result) == ([(10, 5)], [(20, 6)])

    result["circular_ns"] = [(90, 20)]
    result["circular_as"] = [(80, 10)]

    assert _fused_recall_tag_intervals(result) == ([(90, 20)], [(80, 10)])


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
