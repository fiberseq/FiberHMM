"""Shared annotation/tag writing helpers for inference pipelines."""

from __future__ import annotations

import array as pyarray
from typing import Optional, Sequence, Tuple

import numpy as np

from fiberhmm.inference.circular import circular_intervals_overlap
from fiberhmm.inference.tf_recaller import TFCall, write_ma_tags

Interval = Tuple[int, int]


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


def set_legacy_apply_tags(read, result: dict, with_scores: bool, write_msps: bool = True) -> None:
    """Write legacy apply tags (`ns/nl/as/al`, optional `nq/aq`) in place.

    This intentionally preserves the historical apply behavior: a non-empty
    result writes only the tag groups that have calls, and skipped/empty reads
    are passed through by callers.
    """
    # Apply recomputes the read structure, so any pre-existing MA/AN/AQ from a
    # prior fiberhmm-call/recall pass now refers to stale ns/nl coordinates.
    # Strip them so the BAM never carries an inconsistent annotation view.
    for stale_tag in ("MA", "AN", "AQ"):
        if read.has_tag(stale_tag):
            try:
                read.set_tag(stale_tag, None)
            except Exception:
                pass

    if len(result["ns"]) > 0:
        read.set_tag("ns", _u32_bam_array(result["ns"]))
        read.set_tag("nl", _u32_bam_array(result["nl"]))
        if with_scores and result.get("ns_scores") is not None:
            read.set_tag("nq", scores_to_u8(result["ns_scores"]))
        elif read.has_tag("nq"):
            try:
                read.set_tag("nq", None)
            except Exception:
                pass

    if write_msps and len(result["as"]) > 0:
        read.set_tag("as", _u32_bam_array(result["as"]))
        read.set_tag("al", _u32_bam_array(result["al"]))
        if with_scores and result.get("as_scores") is not None:
            read.set_tag("aq", scores_to_u8(result["as_scores"]))
        elif read.has_tag("aq"):
            try:
                read.set_tag("aq", None)
            except Exception:
                pass


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
    tf_intervals = [(c.start, c.start + c.length) for c in tf_calls]
    score_values = scores_to_u8(ns_scores)
    kept: list[Interval] = []
    kept_scores: Optional[list[int]] = [] if score_values is not None else None

    for idx, (s_raw, length_raw) in enumerate(zip(ns, nl)):
        s = int(s_raw)
        length = int(length_raw)
        if length <= 0:
            continue
        keep = length >= unify_threshold
        if not keep:
            nuc_end = s + length
            keep = not any(ts < nuc_end and te > s for ts, te in tf_intervals)
        if keep:
            kept.append((s, length))
            if kept_scores is not None:
                kept_scores.append(score_values[idx] if idx < len(score_values) else 0)

    return kept, kept_scores


def unify_circular_nucs_with_tf_calls(
    nucs: Sequence[Interval],
    tf_calls: Sequence[TFCall],
    unify_threshold: int,
    read_length: int,
    ns_scores: Optional[Sequence[float]] = None,
) -> tuple[list[Interval], Optional[list[int]]]:
    """Drop short circular nucleosome calls that overlap circular TF calls."""
    tf_intervals = [(c.start, c.length) for c in tf_calls]
    score_values = scores_to_u8(ns_scores)
    kept: list[Interval] = []
    kept_scores: Optional[list[int]] = [] if score_values is not None else None

    for idx, (s_raw, length_raw) in enumerate(nucs):
        interval = (int(s_raw), int(length_raw))
        if interval[1] <= 0:
            continue
        keep = interval[1] >= unify_threshold
        if not keep:
            keep = not any(
                circular_intervals_overlap(interval, tf_interval, read_length)
                for tf_interval in tf_intervals
            )
        if keep:
            kept.append(interval)
            if kept_scores is not None:
                kept_scores.append(score_values[idx] if idx < len(score_values) else 0)
    return kept, kept_scores


def split_intervals(intervals: Sequence[Interval]) -> tuple[np.ndarray, np.ndarray]:
    """Return int32 start and length arrays for an interval list."""
    if not intervals:
        return np.asarray([], dtype=np.int32), np.asarray([], dtype=np.int32)
    starts = np.asarray([s for s, _ in intervals], dtype=np.int32)
    lengths = np.asarray([length for _, length in intervals], dtype=np.int32)
    return starts, lengths


def write_fused_recall_tags(
    read,
    read_length: int,
    result: dict,
    also_write_legacy: bool,
    downstream_compat: bool,
) -> None:
    """Apply a fused apply+TF-recall result to a pysam read."""
    kept_nucs = result.get("circular_ns") or intervals_from_arrays(result["ns"], result["nl"])
    msps = result.get("circular_as") or intervals_from_arrays(result["as"], result["al"])
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
