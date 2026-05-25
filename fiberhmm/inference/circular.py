"""Circular-read projection helpers for FiberHMM inference."""

from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple

import numpy as np

from fiberhmm.io.ma_tags import split_circular_interval

Interval = Tuple[int, int]


def tile_sequence_and_mods(sequence: str, mod_positions: Iterable[int]) -> tuple[str, set[int]]:
    """Return 3x tiled sequence and modification positions."""
    read_length = len(sequence)
    tiled_mods: set[int] = set()
    for pos in mod_positions:
        p = int(pos)
        if 0 <= p < read_length:
            tiled_mods.add(p)
            tiled_mods.add(p + read_length)
            tiled_mods.add(p + 2 * read_length)
    return sequence * 3, tiled_mods


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
    n = int(read_length)
    if n <= 0:
        return []

    center_start = n
    center_end = 2 * n
    out: list[Interval] = []
    seen: set[Interval] = set()

    for s_raw, e_raw in zip(starts, ends):
        s = int(s_raw)
        e = int(e_raw)
        if e <= s:
            continue

        if center_start <= s < center_end:
            qstart = s - center_start
            length = min(e - s, n)
        elif s < center_start and e > center_end:
            qstart = 0
            length = n
        else:
            continue

        interval = (qstart % n, max(0, min(int(length), n)))
        if interval[1] > 0 and interval not in seen:
            out.append(interval)
            seen.add(interval)

    return out


def project_center_scores(
    starts: Sequence[int],
    ends: Sequence[int],
    scores: Optional[Sequence[float]],
    read_length: int,
) -> Optional[np.ndarray]:
    """Project per-run scores with the same selection as project_center_runs."""
    if scores is None:
        return None
    n = int(read_length)
    if n <= 0:
        return np.asarray([], dtype=np.float32)

    center_start = n
    center_end = 2 * n
    out: list[float] = []
    seen: set[Interval] = set()
    for s_raw, e_raw, score in zip(starts, ends, scores):
        s = int(s_raw)
        e = int(e_raw)
        if e <= s:
            continue
        if center_start <= s < center_end:
            interval = ((s - center_start) % n, max(0, min(e - s, n)))
        elif s < center_start and e > center_end:
            interval = (0, n)
        else:
            continue
        if interval[1] > 0 and interval not in seen:
            out.append(float(score))
            seen.add(interval)
    return np.asarray(out, dtype=np.float32)


def project_center_tf_calls(tf_calls: Sequence, read_length: int) -> list:
    """Project tiled TFCall objects back to circular molecule coordinates."""
    n = int(read_length)
    if n <= 0:
        return []

    center_start = n
    center_end = 2 * n
    out = []
    seen: set[Interval] = set()
    for call in tf_calls:
        s = int(call.start)
        e = s + int(call.length)
        if e <= s:
            continue
        if center_start <= s < center_end:
            interval = ((s - center_start) % n, max(0, min(e - s, n)))
        elif s < center_start and e > center_end:
            interval = (0, n)
        else:
            continue
        if interval[1] <= 0 or interval in seen:
            continue
        seen.add(interval)
        out.append(type(call)(
            start=interval[0],
            length=interval[1],
            llr=call.llr,
            n_opps=call.n_opps,
            left_ambiguity=call.left_ambiguity,
            right_ambiguity=call.right_ambiguity,
        ))
    return out


def split_intervals_for_legacy(
    intervals: Sequence[Interval],
    read_length: int,
    scores: Optional[Sequence[float]] = None,
) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Split circular intervals into legacy-compatible linear start/length arrays."""
    pieces: list[tuple[int, int, Optional[float]]] = []
    score_values = list(scores) if scores is not None else None

    for idx, (start, length) in enumerate(intervals):
        score = score_values[idx] if score_values is not None and idx < len(score_values) else None
        for piece_start, piece_len in split_circular_interval(start, length, read_length):
            pieces.append((piece_start, piece_len, score))

    pieces.sort(key=lambda item: item[0])
    starts = np.asarray([p[0] for p in pieces], dtype=np.int32)
    lengths = np.asarray([p[1] for p in pieces], dtype=np.int32)
    if score_values is None:
        return starts, lengths, None
    return starts, lengths, np.asarray([0.0 if p[2] is None else p[2] for p in pieces],
                                      dtype=np.float32)


def interval_segments(interval: Interval, read_length: int) -> list[Interval]:
    """Return non-wrapping segments for a circular interval."""
    return split_circular_interval(interval[0], interval[1], read_length)


def circular_intervals_overlap(a: Interval, b: Interval, read_length: int) -> bool:
    """Return whether two circular intervals overlap."""
    for a_start, a_len in interval_segments(a, read_length):
        a_end = a_start + a_len
        for b_start, b_len in interval_segments(b, read_length):
            b_end = b_start + b_len
            if a_start < b_end and b_start < a_end:
                return True
    return False
