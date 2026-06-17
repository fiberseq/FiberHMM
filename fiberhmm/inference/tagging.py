"""Shared annotation/tag writing helpers for inference pipelines."""

from __future__ import annotations

import array as pyarray
from typing import Callable, Iterable, Optional, Sequence, Tuple

import numpy as np

from fiberhmm.inference.circular import circular_intervals_overlap
from fiberhmm.inference.tag_utils import clear_tags
from fiberhmm.inference.tf_recaller import TFCall, write_ma_tags
from fiberhmm.io.ma_tags import _read_length_of, flip_interval_frame

Interval = Tuple[int, int]


def _flip_legacy_intervals_to_molecular(
    starts: Sequence[int],
    lengths: Sequence[int],
    scores: Optional[Sequence[float]],
    read_length: int,
):
    recs = sorted(
        (flip_interval_frame(s, length, read_length),
         (scores[i] if scores is not None else None))
        for i, (s, length) in enumerate(zip(starts, lengths))
    )
    new_s = [r[0][0] for r in recs]
    new_l = [r[0][1] for r in recs]
    new_sc = [r[1] for r in recs] if scores is not None else None
    return new_s, new_l, new_sc


def _to_molecular_legacy(starts, lengths, scores, read, with_scores):
    """Convert parallel (start, length[, score]) legacy arrays to molecular
    frame for a reverse-mapped read (no-op for forward reads). Returns lists;
    scores are reordered to stay aligned with the re-sorted intervals."""
    s_list = _array_to_ints(starts)
    l_list = _array_to_ints(lengths)
    sc = list(scores) if (with_scores and scores is not None) else None
    if not s_list or not getattr(read, 'is_reverse', False):
        return s_list, l_list, sc
    read_length = _read_length_of(read)
    if not read_length:
        return s_list, l_list, sc
    return _flip_legacy_intervals_to_molecular(s_list, l_list, sc, read_length)


def _array_to_ints(values) -> list[int]:
    if values is None:
        return []
    if hasattr(values, "tolist"):
        values = values.tolist()
    return [int(v) for v in values]


def intervals_from_arrays(starts, lengths) -> list[Interval]:
    """Convert parallel start/length arrays to positive-length intervals."""
    return [
        (int(s), int(length))
        for s, length in zip(_array_to_ints(starts), _array_to_ints(lengths))
        if int(length) > 0
    ]


def scores_to_u8(scores: Optional[Sequence[float]]) -> Optional[list[int]]:
    """Convert posterior confidence floats in [0, 1] to BAM-quality bytes."""
    if scores is None:
        return None
    arr = np.asarray(scores, dtype=np.float64)
    if arr.size == 0:
        return []
    return np.clip(arr * 255, 0, 255).astype(np.uint8).tolist()


def _u32_bam_array(values) -> pyarray.array:
    """Build a BAM B:I-compatible array without materializing Python ints."""
    arr = pyarray.array("I")
    values_u32 = np.asarray(values, dtype=np.uint32)
    if values_u32.size == 0:
        return arr
    values_u32 = np.ascontiguousarray(values_u32)
    if arr.itemsize == values_u32.dtype.itemsize:
        arr.frombytes(values_u32.tobytes())
    else:
        arr.extend(int(v) for v in values_u32)
    return arr


def _write_legacy_interval_tags(
    read,
    start_tag: str,
    length_tag: str,
    score_tag: str,
    starts: Sequence[int],
    lengths: Sequence[int],
    scores: Optional[Sequence[float]],
    with_scores: bool,
) -> None:
    if len(starts) == 0:
        return
    read.set_tag(start_tag, _u32_bam_array(starts))
    read.set_tag(length_tag, _u32_bam_array(lengths))
    if with_scores and scores is not None:
        read.set_tag(score_tag, scores_to_u8(scores))
    else:
        clear_tags(read, (score_tag,))


def _append_kept_interval(
    kept: list[Interval],
    kept_scores: Optional[list[int]],
    interval: Interval,
    score_values: Optional[list[int]],
    index: int,
) -> None:
    kept.append(interval)
    if kept_scores is not None:
        kept_scores.append(
            score_values[index]
            if score_values is not None and index < len(score_values)
            else 0
        )


def _should_keep_nuc_interval(
    interval: Interval,
    unify_threshold: int,
    overlaps_tf: Callable[[Interval], bool],
) -> bool:
    if interval[1] <= 0:
        return False
    return interval[1] >= unify_threshold or not overlaps_tf(interval)


def _filter_nucs_with_tf_overlap(
    nucs: Iterable[Interval],
    unify_threshold: int,
    ns_scores: Optional[Sequence[float]],
    overlaps_tf: Callable[[Interval], bool],
) -> tuple[list[Interval], Optional[list[int]]]:
    score_values = scores_to_u8(ns_scores)
    kept: list[Interval] = []
    kept_scores: Optional[list[int]] = [] if score_values is not None else None

    for idx, (s_raw, length_raw) in enumerate(nucs):
        interval = (int(s_raw), int(length_raw))
        if _should_keep_nuc_interval(interval, unify_threshold, overlaps_tf):
            _append_kept_interval(kept, kept_scores, interval, score_values, idx)

    return kept, kept_scores


def _tf_linear_intervals(tf_calls: Sequence[TFCall]) -> list[Interval]:
    return [(c.start, c.start + c.length) for c in tf_calls]


def _linear_intervals_overlap(left: Interval, right: Interval) -> bool:
    left_start, left_end = left
    right_start, right_end = right
    return left_start < right_end and right_start < left_end


def set_legacy_apply_tags(read, result: dict, with_scores: bool, write_msps: bool = True) -> None:
    """Write legacy apply tags (`ns/nl/as/al`, optional `nq/aq`) in place.

    This intentionally preserves the historical apply behavior: a non-empty
    result writes only the tag groups that have calls, and skipped/empty reads
    are passed through by callers.
    """
    # Apply recomputes the read structure, so any pre-existing MA/AN/AQ from a
    # prior fiberhmm-call/recall pass now refers to stale ns/nl coordinates.
    # Strip them so the BAM never carries an inconsistent annotation view.
    clear_tags(read, ("MA", "AN", "AQ"))

    # FiberHMM works in SEQ (query_sequence) coordinates; ns/nl/as/al must be
    # written in molecular (original-fiber) frame for fibertools. Flip + re-sort
    # for reverse-mapped reads (forward reads are unchanged).
    ns_s, ns_l, ns_sc = _to_molecular_legacy(
        result["ns"], result["nl"], result.get("ns_scores"),
        read, with_scores)
    as_s, as_l, as_sc = _to_molecular_legacy(
        result["as"], result["al"], result.get("as_scores"),
        read, with_scores)

    _write_legacy_interval_tags(read, "ns", "nl", "nq", ns_s, ns_l, ns_sc, with_scores)
    if write_msps:
        _write_legacy_interval_tags(read, "as", "al", "aq", as_s, as_l, as_sc, with_scores)


def unify_nucs_with_tf_calls(
    ns: Sequence[int],
    nl: Sequence[int],
    tf_calls: Sequence[TFCall],
    unify_threshold: int,
    ns_scores: Optional[Sequence[float]] = None,
) -> tuple[list[Interval], Optional[list[int]]]:
    """Drop short nucleosome calls that overlap TF calls.

    Returns kept nucleosome intervals and optional score bytes aligned to those
    kept intervals. Scores are carried through by position rather than
    re-derived from a later tag state, which keeps fused call output consistent
    with the HMM scoring pass.
    """
    tf_intervals = _tf_linear_intervals(tf_calls)

    def overlaps_tf(interval: Interval) -> bool:
        s, length = interval
        return any(
            _linear_intervals_overlap((s, s + length), tf_interval)
            for tf_interval in tf_intervals
        )

    return _filter_nucs_with_tf_overlap(
        zip(ns, nl),
        unify_threshold,
        ns_scores,
        overlaps_tf,
    )


def unify_circular_nucs_with_tf_calls(
    nucs: Sequence[Interval],
    tf_calls: Sequence[TFCall],
    unify_threshold: int,
    read_length: int,
    ns_scores: Optional[Sequence[float]] = None,
) -> tuple[list[Interval], Optional[list[int]]]:
    """Drop short circular nucleosome calls that overlap circular TF calls."""
    tf_intervals = [(c.start, c.length) for c in tf_calls]

    def overlaps_tf(interval: Interval) -> bool:
        return any(
            circular_intervals_overlap(interval, tf_interval, read_length)
            for tf_interval in tf_intervals
        )

    return _filter_nucs_with_tf_overlap(
        nucs,
        unify_threshold,
        ns_scores,
        overlaps_tf,
    )


def split_intervals(intervals: Sequence[Interval]) -> tuple[np.ndarray, np.ndarray]:
    """Return int32 start and length arrays for an interval list."""
    if not intervals:
        return np.asarray([], dtype=np.int32), np.asarray([], dtype=np.int32)
    starts = np.asarray([s for s, _ in intervals], dtype=np.int32)
    lengths = np.asarray([length for _, length in intervals], dtype=np.int32)
    return starts, lengths


def _fused_recall_tag_intervals(result: dict) -> tuple[list[Interval], list[Interval]]:
    kept_nucs = result.get("circular_ns") or intervals_from_arrays(
        result["ns"], result["nl"],
    )
    msps = result.get("circular_as") or intervals_from_arrays(
        result["as"], result["al"],
    )
    return kept_nucs, msps


def write_fused_recall_tags(
    read,
    read_length: int,
    result: dict,
    also_write_legacy: bool,
    downstream_compat: bool,
) -> None:
    """Apply a fused apply+TF-recall result to a pysam read."""
    kept_nucs, msps = _fused_recall_tag_intervals(result)
    write_ma_tags(
        read,
        read_length=read_length,
        tf_calls=result["tf_calls"],
        kept_nucs=kept_nucs,
        msps=msps,
        nq_for_kept_nucs=result.get("nq_for_kept_nucs"),
        also_write_legacy=also_write_legacy,
        downstream_compat=downstream_compat,
        nuc_el_for_kept=result.get("nuc_el_for_kept"),
        nuc_er_for_kept=result.get("nuc_er_for_kept"),
    )
