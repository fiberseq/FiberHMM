"""Tests for DAF-seq strand-swap chimera detection."""
from __future__ import annotations

from fiberhmm.daf.encoder import (
    _daf_chimera_segment_counts,
    _has_min_daf_chimera_events,
    _has_min_daf_chimera_strand_counts,
    is_daf_chimera,
)


def test_chimera_strand_count_gate_requires_both_strands():
    assert not _has_min_daf_chimera_strand_counts(5, 4, 5)
    assert _has_min_daf_chimera_strand_counts(5, 5, 5)


def test_chimera_event_count_gate_requires_two_segments():
    assert not _has_min_daf_chimera_events(9, 5)
    assert _has_min_daf_chimera_events(10, 5)


def test_chimera_segment_counts_split_left_and_right_strands():
    # total CT=7, GA=5; left has 4 CT among 6 total events.
    assert _daf_chimera_segment_counts(4, 6, 7, 5) == (6, 2, 3, 3)


def test_clean_ct_read_not_chimera():
    ct = list(range(10, 110, 10))      # 10 CT events
    ga = []                             # no opposite-strand signal
    assert not is_daf_chimera(ct, ga)


def test_clean_read_with_scattered_errors_not_chimera():
    # 8 CT events with a few scattered GA "errors" interspersed -> no pure
    # opposite-strand segment, so not a swap.
    ct = [10, 20, 30, 40, 50, 60, 70, 80]
    ga = [15, 35, 55, 75, 95]
    assert not is_daf_chimera(ct, ga)


def test_ct_then_ga_swap_is_chimera():
    ct = [10, 20, 30, 40, 50]          # pure CT segment (early)
    ga = [200, 210, 220, 230, 240]     # pure GA segment (late)
    assert is_daf_chimera(ct, ga)


def test_ga_then_ct_swap_is_chimera():
    ga = [10, 20, 30, 40, 50]
    ct = [200, 210, 220, 230, 240]
    assert is_daf_chimera(ct, ga)


def test_too_few_events_not_chimera():
    # below min_seg_events on the minority side -> cannot confidently call
    assert not is_daf_chimera([10, 20, 30], [200, 210, 220])


def test_min_seg_threshold_tunable():
    ct = [10, 20, 30, 40]
    ga = [200, 210, 220, 230]
    assert not is_daf_chimera(ct, ga, min_seg_events=5)   # 4 < 5
    assert is_daf_chimera(ct, ga, min_seg_events=4)       # exactly 4
