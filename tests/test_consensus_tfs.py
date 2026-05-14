"""Tests for `fiberhmm-consensus-tfs` helpers."""

from __future__ import annotations

from fiberhmm.cli.consensus_tfs import _iter_tf_calls
from fiberhmm.io.ma_tags import format_aq_array, format_ma_tag


class _FakeConsensusRead:
    def __init__(self, ma, aq, ref_map, strand="CT"):
        self._tags = {"MA": ma, "AQ": aq, "st": strand}
        self._ref_map = ref_map
        self.reference_positions_calls = 0

    def has_tag(self, tag):
        return tag in self._tags

    def get_tag(self, tag):
        if tag not in self._tags:
            raise KeyError(tag)
        return self._tags[tag]

    def get_reference_positions(self, full_length=False):
        assert full_length
        self.reference_positions_calls += 1
        return self._ref_map


def test_iter_tf_calls_reuses_ref_map_for_multiple_tf_annotations():
    ma = format_ma_tag(
        read_length=50,
        nuc_intervals=[],
        msp_intervals=[],
        tf_intervals=[(2, 3), (10, 2)],
    )
    aq = format_aq_array(
        nq_values=[],
        tf_q_values=[60, 40],
        tf_lq_values=[250, 240],
        tf_rq_values=[230, 220],
    )
    read = _FakeConsensusRead(ma, aq, list(range(50)))

    assert list(_iter_tf_calls(read, min_tq=0)) == [
        (2, 5, "CT", 60),
        (10, 12, "CT", 40),
    ]
    assert read.reference_positions_calls == 1


def test_iter_tf_calls_skips_ref_map_when_quality_filter_drops_all_tfs():
    ma = format_ma_tag(
        read_length=50,
        nuc_intervals=[],
        msp_intervals=[],
        tf_intervals=[(2, 3), (10, 2)],
    )
    aq = format_aq_array(
        nq_values=[],
        tf_q_values=[60, 40],
        tf_lq_values=[250, 240],
        tf_rq_values=[230, 220],
    )
    read = _FakeConsensusRead(ma, aq, list(range(50)))

    assert list(_iter_tf_calls(read, min_tq=100)) == []
    assert read.reference_positions_calls == 0


def test_iter_tf_calls_skips_annotations_inside_insertions():
    ma = format_ma_tag(
        read_length=10,
        nuc_intervals=[],
        msp_intervals=[],
        tf_intervals=[(2, 3)],
    )
    aq = format_aq_array(
        nq_values=[],
        tf_q_values=[60],
        tf_lq_values=[250],
        tf_rq_values=[230],
    )
    read = _FakeConsensusRead(ma, aq, [0, 1, None, None, None, 5, 6, 7, 8, 9])

    assert list(_iter_tf_calls(read, min_tq=0)) == []
    assert read.reference_positions_calls == 1


def test_iter_tf_calls_ignores_malformed_ma_tags():
    read = _FakeConsensusRead("not-a-valid-ma-tag", [], list(range(10)))

    assert list(_iter_tf_calls(read, min_tq=0)) == []
    assert read.reference_positions_calls == 0
