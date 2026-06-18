"""Tests for shared query-to-reference interval mapping helpers."""

import numpy as np

from fiberhmm.inference.reference_mapping import (
    _first_ref_in_query_interval,
    _last_ref_in_query_interval,
    _query_interval_bounds,
    _ref_positions_to_half_open_span,
    _scored_interval_record,
    query_interval_to_ref_block,
    query_interval_to_ref_span,
    query_to_ref_lookup,
    scored_interval_blocks,
    scored_interval_spans,
)


def test_query_to_ref_lookup_accepts_arrays_and_dicts():
    q2r = np.array([100, -1, 102], dtype=np.int64)

    assert query_to_ref_lookup(q2r, 0) == 100
    assert query_to_ref_lookup(q2r, 1) is None
    assert query_to_ref_lookup(q2r, 10) is None
    assert query_to_ref_lookup(q2r, -1) is None
    assert query_to_ref_lookup({0: 200, 2: 202}, 2) == 202
    assert query_to_ref_lookup({0: 200, 2: 202}, 1) is None
    assert query_to_ref_lookup({-1: 999, 0: 200}, -1) is None


def test_ref_positions_to_half_open_span_orders_positions():
    assert _ref_positions_to_half_open_span(100, 103) == (100, 104)
    assert _ref_positions_to_half_open_span(103, 100) == (100, 104)


def test_query_interval_bounds_normalizes_numeric_inputs():
    assert _query_interval_bounds("2", "3") == (2, 5)


def test_scored_interval_record_attaches_score_or_default():
    assert _scored_interval_record((10, 12), [7], 0) == (10, 12, 7)
    assert _scored_interval_record((10, 12), [7], 1) == (10, 12, 0)
    assert _scored_interval_record((10, 12), None, 0) == (10, 12, 0)


def test_exact_block_mapping_requires_aligned_endpoints():
    q2r = np.array([100, -1, 102, 103, -1, 105], dtype=np.int64)

    assert query_interval_to_ref_block(0, 4, q2r) == (100, 104)
    assert query_interval_to_ref_block(1, 3, q2r) is None
    assert query_interval_to_ref_block(2, 3, q2r) is None
    assert query_interval_to_ref_block(1, 0, q2r) is None
    assert query_interval_to_ref_block(2, -1, q2r) is None


def test_span_mapping_scans_inward_past_unaligned_edges():
    q2r = np.array([100, -1, 102, 103, -1, 105], dtype=np.int64)

    assert query_interval_to_ref_span(1, 3, q2r) == (102, 104)
    assert query_interval_to_ref_span(2, 3, q2r) == (102, 104)
    assert query_interval_to_ref_span(1, 1, q2r) is None
    assert query_interval_to_ref_span(1, 0, q2r) is None
    assert query_interval_to_ref_span(2, -1, q2r) is None


def test_interval_endpoint_scanners_skip_unaligned_positions():
    q2r = np.array([-1, 101, 102, -1, 104, -1], dtype=np.int64)

    assert _first_ref_in_query_interval(0, 4, q2r) == 101
    assert _last_ref_in_query_interval(0, 4, q2r) == 102
    assert _first_ref_in_query_interval(3, 1, q2r) is None
    assert _last_ref_in_query_interval(5, 1, q2r) is None


def test_scored_interval_helpers_keep_scores_aligned_after_sorting():
    q2r = np.array([50, 51, 52, 100, -1, 102], dtype=np.int64)

    exact = scored_interval_blocks([3, 0], [2, 2], [7, 9], q2r)
    span = scored_interval_spans([3, 0], [2, 2], [7, 9], q2r)

    assert exact == [(50, 52, 9)]
    assert span == [(50, 52, 9), (100, 101, 7)]


def test_scored_interval_helpers_default_missing_scores_to_zero():
    q2r = np.array([10, 11, 12, 13], dtype=np.int64)

    assert scored_interval_blocks([0, 2], [1, 1], None, q2r) == [
        (10, 11, 0),
        (12, 13, 0),
    ]
    assert scored_interval_spans([0, 2], [1, 1], [5], q2r) == [
        (10, 11, 5),
        (12, 13, 0),
    ]
