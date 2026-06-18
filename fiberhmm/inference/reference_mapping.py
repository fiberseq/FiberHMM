"""Query-to-reference coordinate helpers shared by extraction paths."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

from fiberhmm.core.bam_reader import cigar_to_query_ref

_IntervalMapper = Callable[[object, object, object], Optional[Tuple[int, int]]]


@dataclass(frozen=True)
class _QueryIntervalBounds:
    start: int
    end: int


@dataclass(frozen=True)
class _RefSpan:
    start: int
    end: int

    def as_tuple(self) -> Tuple[int, int]:
        return self.start, self.end


@dataclass(frozen=True)
class _ScoredIntervalRecord:
    start: int
    end: int
    score: int


@dataclass(frozen=True)
class _QueryIntervalMappingRequest:
    qstart: object
    length: object
    query_to_ref: object


@dataclass(frozen=True)
class _ScoredIntervalRecordRequest:
    block: Tuple[int, int]
    scores: object
    index: int


@dataclass(frozen=True)
class _ScoredIntervalsRequest:
    starts: object
    lengths: object
    scores: object
    query_to_ref: object
    mapper: _IntervalMapper


def build_query_to_ref(read):
    """Build the fast query-position to reference-position lookup for a read."""
    return cigar_to_query_ref(read)


def query_to_ref_lookup(query_to_ref, qpos: int) -> Optional[int]:
    """Bounds-checked lookup returning ``None`` for unmapped query positions."""
    qpos = int(qpos)
    if qpos < 0:
        return None
    if hasattr(query_to_ref, 'get'):
        ref_pos = query_to_ref.get(qpos)
        return int(ref_pos) if ref_pos is not None else None

    if 0 <= qpos < len(query_to_ref):
        ref_pos = int(query_to_ref[qpos])
        return ref_pos if ref_pos >= 0 else None
    return None


def _ref_positions_to_half_open_span(ref_start: int, ref_end: int) -> _RefSpan:
    ref_start, ref_end = min(ref_start, ref_end), max(ref_start, ref_end) + 1
    return _RefSpan(start=ref_start, end=ref_end)


def _query_interval_bounds(qstart, length) -> _QueryIntervalBounds:
    qstart = int(qstart)
    qend = qstart + int(length)
    return _QueryIntervalBounds(start=qstart, end=qend)


def query_interval_to_ref_block_from_request(
    request: _QueryIntervalMappingRequest,
) -> Optional[Tuple[int, int]]:
    """Map a query interval to a reference block using exact aligned endpoints."""
    bounds = _query_interval_bounds(request.qstart, request.length)
    if bounds.end <= bounds.start:
        return None
    ref_start = query_to_ref_lookup(request.query_to_ref, bounds.start)
    ref_end = query_to_ref_lookup(request.query_to_ref, bounds.end - 1)
    if ref_start is None or ref_end is None:
        return None
    return _ref_positions_to_half_open_span(ref_start, ref_end).as_tuple()


def query_interval_to_ref_block(qstart, length, query_to_ref) -> Optional[Tuple[int, int]]:
    return query_interval_to_ref_block_from_request(
        _QueryIntervalMappingRequest(
            qstart=qstart,
            length=length,
            query_to_ref=query_to_ref,
        )
    )


def _ref_in_query_interval(qstart, length, query_to_ref, *, reverse: bool) -> Optional[int]:
    bounds = _query_interval_bounds(qstart, length)
    query_positions = (
        range(bounds.end - 1, bounds.start - 1, -1)
        if reverse else range(bounds.start, bounds.end)
    )
    for qpos in query_positions:
        ref_pos = query_to_ref_lookup(query_to_ref, qpos)
        if ref_pos is not None:
            return ref_pos
    return None


def _first_ref_in_query_interval(qstart, length, query_to_ref) -> Optional[int]:
    return _ref_in_query_interval(qstart, length, query_to_ref, reverse=False)


def _last_ref_in_query_interval(qstart, length, query_to_ref) -> Optional[int]:
    return _ref_in_query_interval(qstart, length, query_to_ref, reverse=True)


def query_interval_to_ref_span_from_request(
    request: _QueryIntervalMappingRequest,
) -> Optional[Tuple[int, int]]:
    """Map a query interval to its reference span, scanning past unaligned ends."""
    bounds = _query_interval_bounds(request.qstart, request.length)
    if bounds.end <= bounds.start:
        return None
    length = bounds.end - bounds.start
    ref_start = _first_ref_in_query_interval(
        bounds.start,
        length,
        request.query_to_ref,
    )
    ref_end = _last_ref_in_query_interval(
        bounds.start,
        length,
        request.query_to_ref,
    )

    if ref_start is None or ref_end is None:
        return None
    span = _ref_positions_to_half_open_span(ref_start, ref_end)
    if span.end <= span.start:
        return None
    return span.as_tuple()


def query_interval_to_ref_span(qstart, length, query_to_ref) -> Optional[Tuple[int, int]]:
    return query_interval_to_ref_span_from_request(
        _QueryIntervalMappingRequest(
            qstart=qstart,
            length=length,
            query_to_ref=query_to_ref,
        )
    )


def _interval_score(scores, index: int) -> int:
    return int(scores[index]) if scores is not None and index < len(scores) else 0


def _scored_interval_record_from_request(
    request: _ScoredIntervalRecordRequest,
) -> _ScoredIntervalRecord:
    return _ScoredIntervalRecord(
        request.block[0],
        request.block[1],
        _interval_score(request.scores, request.index),
    )


def _scored_interval_record(
    block: Tuple[int, int],
    scores,
    index: int,
) -> _ScoredIntervalRecord:
    return _scored_interval_record_from_request(
        _ScoredIntervalRecordRequest(
            block=block,
            scores=scores,
            index=index,
        )
    )


def _scored_intervals_from_request(
    request: _ScoredIntervalsRequest,
) -> List[Tuple[int, int, int]]:
    records = []
    for i, (qstart, length) in enumerate(zip(request.starts, request.lengths)):
        block = request.mapper(qstart, length, request.query_to_ref)
        if block is None:
            continue
        records.append(
            _scored_interval_record_from_request(
                _ScoredIntervalRecordRequest(
                    block=block,
                    scores=request.scores,
                    index=i,
                )
            )
        )
    records.sort(key=lambda record: record.start)
    return [(record.start, record.end, record.score) for record in records]


def _scored_intervals(starts, lengths, scores, query_to_ref, mapper) -> List[Tuple[int, int, int]]:
    return _scored_intervals_from_request(
        _ScoredIntervalsRequest(
            starts=starts,
            lengths=lengths,
            scores=scores,
            query_to_ref=query_to_ref,
            mapper=mapper,
        )
    )


def scored_interval_blocks(starts, lengths, scores, query_to_ref) -> List[Tuple[int, int, int]]:
    """Map intervals with optional scores using exact endpoint mapping."""
    return _scored_intervals_from_request(
        _ScoredIntervalsRequest(
            starts=starts,
            lengths=lengths,
            scores=scores,
            query_to_ref=query_to_ref,
            mapper=query_interval_to_ref_block,
        )
    )


def scored_interval_spans(starts, lengths, scores, query_to_ref) -> List[Tuple[int, int, int]]:
    """Map intervals with optional scores, scanning inward past unaligned ends."""
    return _scored_intervals_from_request(
        _ScoredIntervalsRequest(
            starts=starts,
            lengths=lengths,
            scores=scores,
            query_to_ref=query_to_ref,
            mapper=query_interval_to_ref_span,
        )
    )
