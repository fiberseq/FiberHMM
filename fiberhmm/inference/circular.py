"""Circular-read projection helpers for FiberHMM inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np

from fiberhmm.io.ma_tags import split_circular_interval

Interval = Tuple[int, int]


@dataclass(frozen=True)
class _LegacyIntervalPiece:
    start: int
    length: int
    score: Optional[float]


def _tiled_mod_positions(pos, read_length: int) -> tuple[int, ...]:
    p = int(pos)
    if 0 <= p < read_length:
        return p, p + read_length, p + 2 * read_length
    return ()


def tile_sequence_and_mods(sequence: str, mod_positions: Iterable[int]) -> tuple[str, set[int]]:
    """Return 3x tiled sequence and modification positions."""
    read_length = len(sequence)
    tiled_mods: set[int] = set()
    for pos in mod_positions:
        tiled_mods.update(_tiled_mod_positions(pos, read_length))
    return sequence * 3, tiled_mods


def _center_copy_bounds(read_length: int) -> tuple[int, int]:
    n = int(read_length)
    return n, 2 * n


def _project_center_interval(s_raw, e_raw, read_length: int) -> Optional[Interval]:
    n = int(read_length)
    s = int(s_raw)
    e = int(e_raw)
    if n <= 0 or e <= s:
        return None

    center_start, center_end = _center_copy_bounds(n)
    if center_start <= s < center_end:
        qstart = s - center_start
        length = min(e - s, n)
    elif s < center_start and e > center_end:
        qstart = 0
        length = n
    else:
        return None

    interval = (qstart % n, max(0, min(int(length), n)))
    return interval if interval[1] > 0 else None


def _project_unique_center_intervals(starts, ends, read_length: int) -> list[tuple[int, Interval]]:
    n = int(read_length)
    if n <= 0:
        return []

    out: list[tuple[int, Interval]] = []
    seen: set[Interval] = set()
    for idx, (s_raw, e_raw) in enumerate(zip(starts, ends)):
        interval = _project_center_interval(s_raw, e_raw, n)
        if interval is None or interval in seen:
            continue
        out.append((idx, interval))
        seen.add(interval)
    return out


def _project_unique_center_calls(calls: Sequence, read_length: int, build_call) -> list:
    n = int(read_length)
    if n <= 0:
        return []

    out = []
    seen: set[Interval] = set()
    for call in calls:
        s = int(call.start)
        e = s + int(call.length)
        interval = _project_center_interval(s, e, n)
        if interval is None or interval in seen:
            continue
        seen.add(interval)
        out.append(build_call(call, interval))
    return out


def _projected_tf_call(call, interval: Interval):
    return type(call)(
        start=interval[0],
        length=interval[1],
        llr=call.llr,
        n_opps=call.n_opps,
        left_ambiguity=call.left_ambiguity,
        right_ambiguity=call.right_ambiguity,
    )


def _projected_nuc_call(call, interval: Interval):
    return type(call)(
        start=interval[0],
        length=interval[1],
        nq=call.nq,
        el=call.el,
        er=call.er,
    )


def project_center_runs(
    starts: Sequence[int],
    ends: Sequence[int],
    read_length: int,
) -> list[Interval]:
    """Project runs from a 3x tiled read back to circular molecule coords.

    The middle copy is ``[read_length, 2 * read_length)``. Runs whose start is
    in that middle copy are the canonical copy of ordinary and wrapping
    features. A run that fully covers the middle copy is projected as one
    whole-molecule interval.
    """
    return [
        interval
        for _, interval in _project_unique_center_intervals(starts, ends, read_length)
    ]


def project_center_scores(
    starts: Sequence[int],
    ends: Sequence[int],
    scores: Optional[Sequence[float]],
    read_length: int,
) -> Optional[np.ndarray]:
    """Project per-run scores with the same selection as project_center_runs."""
    if scores is None:
        return None
    out = [
        float(scores[idx])
        for idx, _ in _project_unique_center_intervals(starts, ends, read_length)
        if idx < len(scores)
    ]
    return np.asarray(out, dtype=np.float32)


def project_center_tf_calls(tf_calls: Sequence, read_length: int) -> list:
    """Project tiled TFCall objects back to circular molecule coordinates."""
    return _project_unique_center_calls(
        tf_calls,
        read_length,
        _projected_tf_call,
    )


def project_center_nuc_calls(nuc_calls: Sequence, read_length: int) -> list:
    """Project tiled NucCall objects back to circular molecule coordinates.

    Mirrors ``project_center_tf_calls`` but preserves the nuc+QQQ quality bytes
    (nq/el/er). Uses ``type(call)(...)`` so this module needn't import NucCall.
    """
    return _project_unique_center_calls(
        nuc_calls,
        read_length,
        _projected_nuc_call,
    )


def _legacy_interval_score(score_values, index: int):
    if score_values is not None and index < len(score_values):
        return score_values[index]
    return None


def _legacy_interval_pieces(
    intervals: Sequence[Interval],
    read_length: int,
    score_values: Optional[Sequence[float]],
) -> list[_LegacyIntervalPiece]:
    pieces: list[_LegacyIntervalPiece] = []
    for idx, (start, length) in enumerate(intervals):
        score = _legacy_interval_score(score_values, idx)
        for piece_start, piece_len in split_circular_interval(start, length, read_length):
            pieces.append(_LegacyIntervalPiece(piece_start, piece_len, score))
    pieces.sort(key=lambda piece: piece.start)
    return pieces


def split_intervals_for_legacy(
    intervals: Sequence[Interval],
    read_length: int,
    scores: Optional[Sequence[float]] = None,
) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Split circular intervals into legacy-compatible linear start/length arrays."""
    score_values = list(scores) if scores is not None else None
    pieces = _legacy_interval_pieces(intervals, read_length, score_values)
    starts = np.asarray([piece.start for piece in pieces], dtype=np.int32)
    lengths = np.asarray([piece.length for piece in pieces], dtype=np.int32)
    if score_values is None:
        return starts, lengths, None
    score_array = np.asarray(
        [0.0 if piece.score is None else piece.score for piece in pieces],
        dtype=np.float32,
    )
    return starts, lengths, score_array


def interval_segments(interval: Interval, read_length: int) -> list[Interval]:
    """Return non-wrapping segments for a circular interval."""
    return split_circular_interval(interval[0], interval[1], read_length)


def _linear_segments_overlap(a: Interval, b: Interval) -> bool:
    a_start, a_len = a
    b_start, b_len = b
    return a_start < b_start + b_len and b_start < a_start + a_len


def circular_intervals_overlap(a: Interval, b: Interval, read_length: int) -> bool:
    """Return whether two circular intervals overlap."""
    for a_segment in interval_segments(a, read_length):
        for b_segment in interval_segments(b, read_length):
            if _linear_segments_overlap(a_segment, b_segment):
                return True
    return False
