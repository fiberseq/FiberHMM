"""Shared annotation/tag writing helpers for inference pipelines."""

from __future__ import annotations

import array as pyarray
from typing import Optional, Sequence, Tuple

import numpy as np

from fiberhmm.inference.circular import circular_intervals_overlap
from fiberhmm.inference.tf_recaller import TFCall, write_ma_tags
from fiberhmm.io.ma_tags import _read_length_of, flip_interval_frame

Interval = Tuple[int, int]


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
    recs = sorted(
        (flip_interval_frame(s, length, read_length),
         (sc[i] if sc is not None else None))
        for i, (s, length) in enumerate(zip(s_list, l_list))
    )
    new_s = [r[0][0] for r in recs]
    new_l = [r[0][1] for r in recs]
    new_sc = [r[1] for r in recs] if sc is not None else None
    return new_s, new_l, new_sc


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


def _footprints_outside_msps(result: dict, write_msps: bool) -> list:
    """Indexes of ``result``'s HMM footprints that are nucleosomes in the MA
    view: those overlapping no MSP (all of them when MSPs are not written)."""
    msps = ([(int(s), int(n)) for s, n in zip(result["as"], result["al"]) if int(n) > 0]
            if write_msps else [])
    return [i for i, (s, n) in enumerate(zip(result["ns"], result["nl"]))
            if int(n) > 0 and not any(int(s) < ms + ml and ms < int(s) + int(n)
                                      for ms, ml in msps)]


def _harmonized_result(result: dict, write_msps: bool) -> dict:
    """``result`` with only the nucleosomes of the MA view (and their scores)."""
    keep = _footprints_outside_msps(result, write_msps)
    out = dict(result)
    out["ns"] = np.asarray(result["ns"])[keep] if len(keep) else np.asarray([], dtype=np.int64)
    out["nl"] = np.asarray(result["nl"])[keep] if len(keep) else np.asarray([], dtype=np.int64)
    scores = result.get("ns_scores")
    if scores is not None and len(scores) == len(result["ns"]):
        out["ns_scores"] = np.asarray(scores)[keep]
    return out


def set_legacy_apply_tags(read, result: dict, with_scores: bool, write_msps: bool = True,
                          harmonize: bool = False) -> None:
    """Write legacy apply tags (`ns/nl/as/al`, optional `nq/aq`) in place.

    This intentionally preserves the compatibility apply behavior: a non-empty
    result writes only the tag groups that have calls, and skipped/empty reads
    are passed through by callers.

    Two views, on purpose: legacy ``ns/nl`` lists every HMM footprint (TF-sized
    ones inside MSPs included; that is what ``fiberhmm-recall-tfs`` reads), and
    ``as/al`` the MSPs. MA is written only when the read keeps DddA CpG-island
    groups (``ddda_ucg``/``ddda_mcg``), and then follows the MA spec: the
    nucleosomes are the footprints outside MSPs, so nuc and msp tile the read
    and ``ns/nl`` may list more footprints than MA ``nuc``.

    ``harmonize`` (``fiberhmm-apply --harmonize-legacy-tags``) writes ``ns/nl``
    from the MA view instead (footprints outside MSPs), so legacy tags and MA
    agree as in fiberhmm-call output; ``as/al`` are unchanged. Recall needs the
    raw footprints, so harmonized output is not recall input.
    """
    # Apply recomputes the read structure, so the footprint groups of any
    # pre-existing MA/AN/AQ from a prior call/recall pass are stale. They are
    # dropped; DddA CpG-island groups (ddda_ucg/ddda_mcg from tag-m5c) are
    # read-intrinsic and kept, so a later recall applies the same CpG policy.
    from fiberhmm.inference.read_filters import strip_stale_call_tags

    strip_stale_call_tags(read, legacy_tags=(), keep_m5c_groups=True)
    keeps_m5c = read.has_tag("MA")
    if harmonize:
        result = _harmonized_result(result, write_msps)

    # FiberHMM works in SEQ (query_sequence) coordinates; ns/nl/as/al must be
    # written in molecular (original-fiber) frame for fibertools. Flip + re-sort
    # for reverse-mapped reads (forward reads are unchanged).
    ns_s, ns_l, ns_sc = _to_molecular_legacy(
        result["ns"], result["nl"], result.get("ns_scores"),
        read, with_scores)
    as_s, as_l, as_sc = _to_molecular_legacy(
        result["as"], result["al"], result.get("as_scores"),
        read, with_scores)

    if len(ns_s) > 0:
        read.set_tag("ns", _u32_bam_array(ns_s))
        read.set_tag("nl", _u32_bam_array(ns_l))
        if with_scores and ns_sc is not None:
            read.set_tag("nq", scores_to_u8(ns_sc))
        elif read.has_tag("nq"):
            try:
                read.set_tag("nq", None)
            except Exception:
                pass

    if write_msps and len(as_s) > 0:
        read.set_tag("as", _u32_bam_array(as_s))
        read.set_tag("al", _u32_bam_array(as_l))
        if with_scores and as_sc is not None:
            read.set_tag("aq", scores_to_u8(as_sc))
        elif read.has_tag("aq"):
            try:
                read.set_tag("aq", None)
            except Exception:
                pass

    if keeps_m5c:
        # The kept CpG-island groups make MA present, and MA is authoritative
        # for readers, so it also carries this run's nucleosomes and MSPs.
        from fiberhmm.inference.tf_recaller import write_ma_tags

        footprints = [(int(s), int(n)) for s, n in zip(result["ns"], result["nl"])]
        msps = ([(int(s), int(n)) for s, n in zip(result["as"], result["al"]) if int(n) > 0]
                if write_msps else [])
        scores = scores_to_u8(result.get("ns_scores")) if with_scores else None
        if scores is not None and len(scores) != len(footprints):
            scores = None
        # Legacy ns also lists sub-nucleosomal footprints inside MSPs; in MA
        # (where nuc and msp tile the read) only footprints outside MSPs are
        # nucleosomes. TF-sized footprints are left to recall, as in call.
        keep = _footprints_outside_msps(result, write_msps)
        nucs = [footprints[i] for i in keep]
        nq = [scores[i] for i in keep] if scores is not None else None
        write_ma_tags(read, len(read.query_sequence or "") or int(read.query_length or 0),
                      [], nucs, msps, nq_for_kept_nucs=nq, also_write_legacy=False)


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
    if "ddda_mcg_spans" in result:
        # The fused caller generated these in SEQ/query coordinates. Append
        # after nuc/msp/tf writing so the shared helper can preserve AQ/AN
        # alignment while replacing any stale prior ddda_mcg group.
        from fiberhmm.daf.m5c import add_m5c_ma_tag
        add_m5c_ma_tag(read, result.get("ddda_mcg_spans") or [])
