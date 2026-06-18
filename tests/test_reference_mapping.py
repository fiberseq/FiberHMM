"""Tests for shared query-to-reference interval mapping helpers."""

import numpy as np

from fiberhmm.inference.reference_mapping import (
    _first_ref_in_query_interval,
    _last_ref_in_query_interval,
    _query_interval_bounds,
    _QueryIntervalMappingRequest,
    _ref_positions_to_half_open_span,
    _scored_interval_record,
    _scored_interval_record_from_request,
    _scored_intervals,
    _scored_intervals_from_request,
    _ScoredIntervalRecord,
    _ScoredIntervalRecordRequest,
    _ScoredIntervalsRequest,
    query_interval_to_ref_block,
    query_interval_to_ref_block_from_request,
    query_interval_to_ref_span,
    query_interval_to_ref_span_from_request,
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
    forward_span = _ref_positions_to_half_open_span(100, 103)
    reverse_span = _ref_positions_to_half_open_span(103, 100)

    assert forward_span.start == 100
    assert forward_span.end == 104
    assert reverse_span.start == 100
    assert reverse_span.end == 104


def test_query_interval_bounds_normalizes_numeric_inputs():
    bounds = _query_interval_bounds("2", "3")

    assert bounds.start == 2
    assert bounds.end == 5


def test_scored_interval_record_attaches_score_or_default():
    request = _ScoredIntervalRecordRequest(
        block=(10, 12),
        scores=[7],
        index=0,
    )

    assert _scored_interval_record_from_request(request) == _ScoredIntervalRecord(
        10, 12, 7,
    )
    assert _scored_interval_record((10, 12), [7], 0) == _ScoredIntervalRecord(
        10, 12, 7,
    )
    assert _scored_interval_record((10, 12), [7], 1) == _ScoredIntervalRecord(
        10, 12, 0,
    )
    assert _scored_interval_record((10, 12), None, 0) == _ScoredIntervalRecord(
        10, 12, 0,
    )


def test_exact_block_mapping_requires_aligned_endpoints():
    q2r = np.array([100, -1, 102, 103, -1, 105], dtype=np.int64)
    request = _QueryIntervalMappingRequest(
        qstart=0,
        length=4,
        query_to_ref=q2r,
    )

    assert query_interval_to_ref_block_from_request(request) == (100, 104)
    assert query_interval_to_ref_block(0, 4, q2r) == (100, 104)
    assert query_interval_to_ref_block(1, 3, q2r) is None
    assert query_interval_to_ref_block(2, 3, q2r) is None
    assert query_interval_to_ref_block(1, 0, q2r) is None
    assert query_interval_to_ref_block(2, -1, q2r) is None


def test_span_mapping_scans_inward_past_unaligned_edges():
    q2r = np.array([100, -1, 102, 103, -1, 105], dtype=np.int64)
    request = _QueryIntervalMappingRequest(
        qstart=1,
        length=3,
        query_to_ref=q2r,
    )

    assert query_interval_to_ref_span_from_request(request) == (102, 104)
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
    request = _ScoredIntervalsRequest(
        starts=[3, 0],
        lengths=[2, 2],
        scores=[7, 9],
        query_to_ref=q2r,
        mapper=query_interval_to_ref_span,
    )

    exact = scored_interval_blocks([3, 0], [2, 2], [7, 9], q2r)
    requested_span = _scored_intervals_from_request(request)
    adapted_span = _scored_intervals(
        [3, 0],
        [2, 2],
        [7, 9],
        q2r,
        query_interval_to_ref_span,
    )
    span = scored_interval_spans([3, 0], [2, 2], [7, 9], q2r)

    assert exact == [(50, 52, 9)]
    assert requested_span == [(50, 52, 9), (100, 101, 7)]
    assert adapted_span == [(50, 52, 9), (100, 101, 7)]
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
