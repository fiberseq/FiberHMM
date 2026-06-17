"""Query-to-reference coordinate helpers shared by extraction paths."""

from __future__ import annotations

from typing import List, Optional, Tuple

from fiberhmm.core.bam_reader import cigar_to_query_ref


def build_query_to_ref(read):
    """Build the fast query-position to reference-position lookup for a read."""
    return cigar_to_query_ref(read)


def query_to_ref_lookup(query_to_ref, qpos: int) -> Optional[int]:
    """Bounds-checked lookup returning ``None`` for unmapped query positions."""
    qpos = int(qpos)
    if hasattr(query_to_ref, 'get'):
        ref_pos = query_to_ref.get(qpos)
        return int(ref_pos) if ref_pos is not None else None

    if 0 <= qpos < len(query_to_ref):
        ref_pos = int(query_to_ref[qpos])
        return ref_pos if ref_pos >= 0 else None
    return None


def query_interval_to_ref_block(qstart, length, query_to_ref) -> Optional[Tuple[int, int]]:
    """Map a query interval to a reference block using exact aligned endpoints."""
    qstart = int(qstart)
    qend = qstart + int(length)
    ref_start = query_to_ref_lookup(query_to_ref, qstart)
    ref_end = query_to_ref_lookup(query_to_ref, qend - 1)
    if ref_start is None or ref_end is None:
        return None
    ref_start, ref_end = min(ref_start, ref_end), max(ref_start, ref_end) + 1
    return ref_start, ref_end


def _first_ref_in_query_interval(qstart, length, query_to_ref) -> Optional[int]:
    qstart = int(qstart)
    length = int(length)
    for offset in range(length):
        ref_pos = query_to_ref_lookup(query_to_ref, qstart + offset)
        if ref_pos is not None:
            return ref_pos
    return None


def _last_ref_in_query_interval(qstart, length, query_to_ref) -> Optional[int]:
    qstart = int(qstart)
    length = int(length)
    qend = qstart + length
    for offset in range(length):
        ref_pos = query_to_ref_lookup(query_to_ref, qend - 1 - offset)
        if ref_pos is not None:
            return ref_pos
    return None


def query_interval_to_ref_span(qstart, length, query_to_ref) -> Optional[Tuple[int, int]]:
    """Map a query interval to its reference span, scanning past unaligned ends."""
    qstart = int(qstart)
    length = int(length)
    ref_start = _first_ref_in_query_interval(qstart, length, query_to_ref)
    ref_end = _last_ref_in_query_interval(qstart, length, query_to_ref)

    if ref_start is None or ref_end is None:
        return None
    ref_start, ref_end = min(ref_start, ref_end), max(ref_start, ref_end) + 1
    if ref_end <= ref_start:
        return None
    return ref_start, ref_end


def _interval_score(scores, index: int) -> int:
    return int(scores[index]) if scores is not None and index < len(scores) else 0


def _scored_intervals(starts, lengths, scores, query_to_ref, mapper) -> List[Tuple[int, int, int]]:
    blocks = []
    for i, (qstart, length) in enumerate(zip(starts, lengths)):
        block = mapper(qstart, length, query_to_ref)
        if block is None:
            continue
        blocks.append((block[0], block[1], _interval_score(scores, i)))
    blocks.sort(key=lambda x: x[0])
    return blocks


def scored_interval_blocks(starts, lengths, scores, query_to_ref) -> List[Tuple[int, int, int]]:
    """Map intervals with optional scores using exact endpoint mapping."""
    return _scored_intervals(
        starts, lengths, scores, query_to_ref, query_interval_to_ref_block,
    )


def scored_interval_spans(starts, lengths, scores, query_to_ref) -> List[Tuple[int, int, int]]:
    """Map intervals with optional scores, scanning inward past unaligned ends."""
    return _scored_intervals(
        starts, lengths, scores, query_to_ref, query_interval_to_ref_span,
    )
