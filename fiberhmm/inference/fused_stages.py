"""Stage helpers for fused HMM apply plus TF recall inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from fiberhmm.inference.circular import (
    project_center_nuc_calls,
    project_center_runs,
    project_center_tf_calls,
    split_intervals_for_legacy,
)
from fiberhmm.inference.engine import _process_single_read
from fiberhmm.inference.nuc_recaller import (
    assemble_circular_nuc_msp_tiling,
    assemble_nuc_msp_tiling,
    drop_short_nucs_overlapping_promoted,
    promote_large_tf_calls,
    recall_nucs_in_read,
    rederive_msps,
    unify_circular_nuc_calls_with_tf_calls,
    unify_nuc_calls_with_tf_calls,
)
from fiberhmm.inference.tagging import (
    split_intervals,
    unify_circular_nucs_with_tf_calls,
    unify_nucs_with_tf_calls,
)
from fiberhmm.inference.tf_recaller import build_scan_intervals, call_tfs_in_interval


@dataclass(frozen=True)
class _IntervalBounds:
    starts: list[int]
    ends: list[int]


@dataclass(frozen=True)
class _AnalyzedSpan:
    start: int
    end: int


@dataclass(frozen=True)
class _AnalyzedSpanRequest:
    apply_result: Mapping[str, Any]
    read_length: int
    kept: Sequence[Any]


@dataclass(frozen=True)
class _IntervalPairLists:
    starts: list
    lengths: list


@dataclass(frozen=True)
class _NucCallQualityLists:
    nq_values: list
    el_values: list
    er_values: list


@dataclass(frozen=True)
class _NucCallArrays:
    starts: np.ndarray
    lengths: np.ndarray


@dataclass(frozen=True)
class _TiledIntervalArrays:
    nuc_starts: Any
    nuc_lengths: Any
    msp_starts: Any
    msp_lengths: Any


@dataclass(frozen=True)
class _HmmApplyStageRequest:
    fiber_read: Mapping[str, Any]
    model: Any
    edge_trim: int
    circular: bool
    mode: str
    context_size: int
    msp_min_size: int
    nuc_min_size: int
    with_scores: bool


@dataclass(frozen=True)
class _TfRecallStageRequest:
    obs: Any
    ns: Sequence[int]
    nl: Sequence[int]
    msps: Sequence[int]
    msp_lengths: Sequence[int]
    read_length: int
    llr_hit: Any
    llr_miss: Any
    min_llr: float
    min_opps: int
    unify_threshold: int


@dataclass(frozen=True)
class _FusedRecallResultRequest:
    fiber_read: Mapping[str, Any]
    apply_result: Mapping[str, Any]
    llr_hit: Any
    llr_miss: Any
    min_llr: float
    min_opps: int
    unify_threshold: int
    with_scores: bool
    recall_nucs: bool = False
    split_min_llr: float = 4.0
    split_min_opps: int = 3
    nuc_min_size: int = 85
    msp_min_size: int = 0
    phase_nrl: int = 0


@dataclass(frozen=True)
class _OptionalApplyScoreFieldsRequest:
    apply_result: Mapping[str, Any]
    enabled: bool


@dataclass(frozen=True)
class _PromoteLargeTfNucsRequest:
    tf_calls: Any
    nuc_calls: Any
    obs: Any
    llr_hit: Any
    llr_miss: Any
    unify_threshold: int
    nuc_min_size: int


def _analyzed_span_from_request(
    request: _AnalyzedSpanRequest,
) -> _AnalyzedSpan:
    """Extent (lo, hi) the read was annotated over -- the union of the original
    HMM footprints/MSPs and the final nucleosomes -- used to tile MSPs."""
    bounds = _apply_result_interval_bounds(request.apply_result)
    starts = list(bounds.starts)
    ends = list(bounds.ends)
    for k in request.kept:
        starts.append(int(k.start))
        ends.append(int(k.start) + int(k.length))
    return _AnalyzedSpan(
        start=min(starts) if starts else 0,
        end=max(ends) if ends else int(request.read_length),
    )


def _analyzed_span(apply_result, read_length, kept):
    return _analyzed_span_from_request(
        _AnalyzedSpanRequest(
            apply_result=apply_result,
            read_length=read_length,
            kept=kept,
        )
    )


def _apply_result_interval_bounds(apply_result):
    starts, ends = [], []
    for ks, kl in (("ns", "nl"), ("as", "al")):
        for s, length in zip(apply_result.get(ks, ()), apply_result.get(kl, ())):
            starts.append(int(s))
            ends.append(int(s) + int(length))
    return _IntervalBounds(starts=starts, ends=ends)


def _interval_pair_lists(intervals):
    return _IntervalPairLists(
        starts=[s for s, _ in intervals],
        lengths=[length for _, length in intervals],
    )


def _interval_pairs(starts, lengths):
    return [(int(start), int(length)) for start, length in zip(starts, lengths)]


def _interval_ends(intervals):
    return [int(start) + int(length) for start, length in intervals]


def _tiled_interval_arrays(apply_result: Mapping[str, Any]):
    return _TiledIntervalArrays(
        nuc_starts=apply_result.get("tiled_ns", apply_result["ns"]),
        nuc_lengths=apply_result.get("tiled_nl", apply_result["nl"]),
        msp_starts=apply_result.get("tiled_as", apply_result["as"]),
        msp_lengths=apply_result.get("tiled_al", apply_result["al"]),
    )


def _apply_result_msp_pairs(apply_result: Mapping[str, Any]):
    return _interval_pairs(apply_result["as"], apply_result["al"])


def _nuc_call_quality_lists(nuc_calls):
    return _NucCallQualityLists(
        nq_values=[k.nq for k in nuc_calls],
        el_values=[k.el for k in nuc_calls],
        er_values=[k.er for k in nuc_calls],
    )


def _nuc_call_quality_fields(nuc_calls):
    qualities = _nuc_call_quality_lists(nuc_calls)
    return {
        "nq_for_kept_nucs": qualities.nq_values,
        "nuc_el_for_kept": qualities.el_values,
        "nuc_er_for_kept": qualities.er_values,
    }


def _nuc_call_arrays(nuc_calls):
    return _NucCallArrays(
        starts=np.asarray([k.start for k in nuc_calls], dtype=np.int32),
        lengths=np.asarray([k.length for k in nuc_calls], dtype=np.int32),
    )


def _nuc_call_start_length_lists(nuc_calls):
    return (
        [k.start for k in nuc_calls],
        [k.length for k in nuc_calls],
    )


def _optional_apply_scores(apply_result: Mapping[str, Any], key: str, enabled: bool):
    return apply_result.get(key) if enabled else None


def _optional_apply_score_fields_from_request(
    request: _OptionalApplyScoreFieldsRequest,
):
    return {
        "ns_scores": _optional_apply_scores(
            request.apply_result,
            "ns_scores",
            request.enabled,
        ),
        "as_scores": _optional_apply_scores(
            request.apply_result,
            "as_scores",
            request.enabled,
        ),
    }


def _optional_apply_score_fields(apply_result: Mapping[str, Any], enabled: bool):
    return _optional_apply_score_fields_from_request(
        _OptionalApplyScoreFieldsRequest(
            apply_result=apply_result,
            enabled=enabled,
        )
    )


def _promote_large_tf_nucs_from_request(
    request: _PromoteLargeTfNucsRequest,
):
    tf_calls, promoted = promote_large_tf_calls(
        request.tf_calls,
        request.obs,
        request.llr_hit,
        request.llr_miss,
        request.unify_threshold,
        request.nuc_min_size,
    )
    nuc_calls = drop_short_nucs_overlapping_promoted(
        request.nuc_calls,
        promoted,
        request.unify_threshold,
    ) + promoted
    return tf_calls, nuc_calls


def _promote_large_tf_nucs(
    tf_calls,
    nuc_calls,
    obs,
    llr_hit,
    llr_miss,
    unify_threshold: int,
    nuc_min_size: int,
):
    return _promote_large_tf_nucs_from_request(
        _PromoteLargeTfNucsRequest(
            tf_calls=tf_calls,
            nuc_calls=nuc_calls,
            obs=obs,
            llr_hit=llr_hit,
            llr_miss=llr_miss,
            unify_threshold=unify_threshold,
            nuc_min_size=nuc_min_size,
        )
    )


def _circular_read_length(fiber_read: Mapping[str, Any],
                          apply_result: Mapping[str, Any]) -> int:
    return int(
        apply_result.get("circular_read_length")
        or len(fiber_read["query_sequence"])
    )


def _apply_result_is_circular(apply_result: Mapping[str, Any]) -> bool:
    return bool(apply_result.get("circular"))


def apply_result_has_footprints(apply_result: Optional[Mapping[str, Any]]) -> bool:
    """Return whether an HMM apply result has annotations worth writing."""
    if apply_result is None:
        return False
    return len(apply_result["ns"]) > 0 or len(apply_result["as"]) > 0


def run_hmm_apply_stage_from_request(
    request: _HmmApplyStageRequest,
) -> Optional[dict]:
    """Run the HMM apply stage and keep encoded observations for recall."""
    return _process_single_read(
        request.fiber_read,
        request.model,
        request.edge_trim,
        request.circular,
        request.mode,
        request.context_size,
        request.msp_min_size,
        nuc_min_size=request.nuc_min_size,
        with_scores=request.with_scores,
        return_posteriors=False,
        include_encoded=True,
    )


def run_hmm_apply_stage(
    fiber_read: Mapping[str, Any],
    model,
    edge_trim: int,
    circular: bool,
    mode: str,
    context_size: int,
    msp_min_size: int,
    nuc_min_size: int,
    with_scores: bool,
) -> Optional[dict]:
    return run_hmm_apply_stage_from_request(
        _HmmApplyStageRequest(
            fiber_read=fiber_read,
            model=model,
            edge_trim=edge_trim,
            circular=circular,
            mode=mode,
            context_size=context_size,
            msp_min_size=msp_min_size,
            nuc_min_size=nuc_min_size,
            with_scores=with_scores,
        ),
    )


def run_tf_recall_stage_from_request(
    request: _TfRecallStageRequest,
) -> list:
    """Build recall scan intervals and run the TF LLR scan over each one."""
    intervals = build_scan_intervals(
        request.ns,
        request.nl,
        request.msps,
        request.msp_lengths,
        request.read_length,
        unify_threshold=request.unify_threshold,
    )
    tf_calls = []
    for lo, hi in intervals:
        tf_calls.extend(
            call_tfs_in_interval(
                request.obs,
                lo,
                hi,
                request.llr_hit,
                request.llr_miss,
                request.min_llr,
                request.min_opps,
            )
        )
    return tf_calls


def run_tf_recall_stage(
    obs,
    ns: Sequence[int],
    nl: Sequence[int],
    msps: Sequence[int],
    msp_lengths: Sequence[int],
    read_length: int,
    llr_hit,
    llr_miss,
    min_llr: float,
    min_opps: int,
    unify_threshold: int,
) -> list:
    return run_tf_recall_stage_from_request(
        _TfRecallStageRequest(
            obs=obs,
            ns=ns,
            nl=nl,
            msps=msps,
            msp_lengths=msp_lengths,
            read_length=read_length,
            llr_hit=llr_hit,
            llr_miss=llr_miss,
            min_llr=min_llr,
            min_opps=min_opps,
            unify_threshold=unify_threshold,
        ),
    )


def build_fused_recall_result(
    fiber_read: Mapping[str, Any],
    apply_result: Mapping[str, Any],
    llr_hit,
    llr_miss,
    min_llr: float,
    min_opps: int,
    unify_threshold: int,
    with_scores: bool,
    recall_nucs: bool = False,
    split_min_llr: float = 4.0,
    split_min_opps: int = 3,
    nuc_min_size: int = 85,
    msp_min_size: int = 0,
    phase_nrl: int = 0,
) -> dict:
    """Run TF recall and nucleosome/TF unification after an HMM apply result.

    When ``recall_nucs`` is True, the per-read nucleosome recaller runs FIRST:
    it splits over-merged footprints on accessible evidence and refines each
    nucleosome's conservative edges + quality (nuc+QQQ). MSPs are then re-derived
    from the new boundaries, and TF recall runs over the cleaner accessible
    space. Circular reads run the same flow in tiled coordinates and project the
    refined nucs/MSPs/TFs back to the molecule. ``recall_nucs=False`` (the
    default) is byte-for-byte the original behavior.
    """
    return build_fused_recall_result_from_request(
        _FusedRecallResultRequest(
            fiber_read=fiber_read,
            apply_result=apply_result,
            llr_hit=llr_hit,
            llr_miss=llr_miss,
            min_llr=min_llr,
            min_opps=min_opps,
            unify_threshold=unify_threshold,
            with_scores=with_scores,
            recall_nucs=recall_nucs,
            split_min_llr=split_min_llr,
            split_min_opps=split_min_opps,
            nuc_min_size=nuc_min_size,
            msp_min_size=msp_min_size,
            phase_nrl=phase_nrl,
        ),
    )


def build_fused_recall_result_from_request(
    request: _FusedRecallResultRequest,
) -> dict:
    is_circular = _apply_result_is_circular(request.apply_result)

    if request.recall_nucs:
        if is_circular:
            return _build_fused_recall_result_with_nucs_circular(
                request.fiber_read, request.apply_result, request.llr_hit,
                request.llr_miss, request.min_llr, request.min_opps,
                request.unify_threshold, request.split_min_llr,
                request.split_min_opps, request.nuc_min_size,
                request.msp_min_size, request.phase_nrl,
            )
        return _build_fused_recall_result_with_nucs(
            request.fiber_read, request.apply_result, request.llr_hit,
            request.llr_miss, request.min_llr, request.min_opps,
            request.unify_threshold, request.split_min_llr,
            request.split_min_opps, request.nuc_min_size,
            request.msp_min_size, request.phase_nrl,
        )

    if is_circular:
        return _build_fused_recall_result_without_nucs_circular(
            request.fiber_read, request.apply_result, request.llr_hit,
            request.llr_miss, request.min_llr, request.min_opps,
            request.unify_threshold, request.with_scores,
        )
    return _build_fused_recall_result_without_nucs_linear(
        request.fiber_read, request.apply_result, request.llr_hit,
        request.llr_miss, request.min_llr, request.min_opps,
        request.unify_threshold, request.with_scores,
    )


def _build_fused_recall_result_without_nucs_linear(
    fiber_read: Mapping[str, Any],
    apply_result: Mapping[str, Any],
    llr_hit,
    llr_miss,
    min_llr: float,
    min_opps: int,
    unify_threshold: int,
    with_scores: bool,
) -> dict:
    ns = apply_result["ns"]
    nl = apply_result["nl"]
    msps = apply_result["as"]
    msp_lengths = apply_result["al"]
    tf_calls = run_tf_recall_stage(
        apply_result["encoded"],
        ns,
        nl,
        msps,
        msp_lengths,
        len(fiber_read["query_sequence"]),
        llr_hit,
        llr_miss,
        min_llr,
        min_opps,
        unify_threshold,
    )

    score_fields = _optional_apply_score_fields_from_request(
        _OptionalApplyScoreFieldsRequest(
            apply_result=apply_result,
            enabled=with_scores,
        )
    )
    kept_nucs, nq_for_kept = unify_nucs_with_tf_calls(
        ns,
        nl,
        tf_calls,
        unify_threshold,
        score_fields["ns_scores"],
    )
    kept_starts, kept_lengths = split_intervals(kept_nucs)

    return {
        "ns": kept_starts,
        "nl": kept_lengths,
        "as": msps,
        "al": msp_lengths,
        **score_fields,
        "nq_for_kept_nucs": nq_for_kept,
        "tf_calls": tf_calls,
    }


def _build_fused_recall_result_without_nucs_circular(
    fiber_read: Mapping[str, Any],
    apply_result: Mapping[str, Any],
    llr_hit,
    llr_miss,
    min_llr: float,
    min_opps: int,
    unify_threshold: int,
    with_scores: bool,
) -> dict:
    read_length = _circular_read_length(fiber_read, apply_result)
    tiled = _tiled_interval_arrays(apply_result)
    tf_calls = run_tf_recall_stage(
        apply_result["encoded"],
        tiled.nuc_starts,
        tiled.nuc_lengths,
        tiled.msp_starts,
        tiled.msp_lengths,
        len(apply_result["encoded"]),
        llr_hit,
        llr_miss,
        min_llr,
        min_opps,
        unify_threshold,
    )
    tf_calls = project_center_tf_calls(tf_calls, read_length)
    kept_nucs, nq_for_kept = unify_circular_nucs_with_tf_calls(
        apply_result.get("circular_ns", []),
        tf_calls,
        unify_threshold,
        read_length,
        _optional_apply_scores(apply_result, "circular_ns_scores", with_scores),
    )
    kept_starts, kept_lengths, kept_scores = split_intervals_for_legacy(
        kept_nucs,
        read_length,
        _optional_apply_scores(apply_result, "circular_ns_scores", with_scores),
    )
    msp_starts, msp_lengths_split, msp_scores = split_intervals_for_legacy(
        apply_result.get("circular_as", []),
        read_length,
        _optional_apply_scores(apply_result, "circular_as_scores", with_scores),
    )
    return {
        "ns": kept_starts,
        "nl": kept_lengths,
        "as": msp_starts,
        "al": msp_lengths_split,
        "ns_scores": kept_scores,
        "as_scores": msp_scores,
        "nq_for_kept_nucs": nq_for_kept,
        "tf_calls": tf_calls,
        "circular": True,
        "circular_read_length": read_length,
        "circular_ns": kept_nucs,
        "circular_as": apply_result.get("circular_as", []),
    }


def _build_fused_recall_result_with_nucs(
    fiber_read: Mapping[str, Any],
    apply_result: Mapping[str, Any],
    llr_hit,
    llr_miss,
    min_llr: float,
    min_opps: int,
    unify_threshold: int,
    split_min_llr: float,
    split_min_opps: int,
    nuc_min_size: int,
    msp_min_size: int,
    phase_nrl: int = 0,
) -> dict:
    """nuc recall -> MSP re-derive -> TF recall (non-circular only)."""
    obs = apply_result["encoded"]
    read_length = len(fiber_read["query_sequence"])
    ns = apply_result["ns"]
    nl = apply_result["nl"]
    orig_msps = _apply_result_msp_pairs(apply_result)

    # 1) split + edge-refine footprints (+ optional Pass-2 phase prior)
    nuc_calls, access = recall_nucs_in_read(
        obs, ns, nl, read_length, llr_hit, llr_miss,
        split_min_llr=split_min_llr, split_min_opps=split_min_opps,
        nuc_min_size=nuc_min_size, phase_nrl=phase_nrl,
    )

    # 2) re-derive MSPs from the new nucleosome boundaries
    new_msps = rederive_msps(orig_msps, access, read_length, msp_min_size)
    msp_pairs = _interval_pair_lists(new_msps)

    # 3) TF recall over the cleaner accessible space + short refined nucs
    refined_ns, refined_nl = _nuc_call_start_length_lists(nuc_calls)
    tf_calls = run_tf_recall_stage(
        obs,
        refined_ns,
        refined_nl,
        msp_pairs.starts,
        msp_pairs.lengths,
        read_length,
        llr_hit, llr_miss, min_llr, min_opps, unify_threshold,
    )

    # 3b) promote nucleosome-sized TF leaks (>= unify_threshold) back to nuc+
    tf_calls, nuc_calls = _promote_large_tf_nucs_from_request(
        _PromoteLargeTfNucsRequest(
            tf_calls=tf_calls,
            nuc_calls=nuc_calls,
            obs=obs,
            llr_hit=llr_hit,
            llr_miss=llr_miss,
            unify_threshold=unify_threshold,
            nuc_min_size=nuc_min_size,
        )
    )

    # 4) unify: drop short refined nucs overlapped by a TF call (carry nq/el/er)
    kept = unify_nuc_calls_with_tf_calls(nuc_calls, tf_calls, unify_threshold)

    # 5) re-tile: split/phase/promotion can leave overlapping nucs + stale MSPs.
    # Clip to non-overlapping nucleosomes and derive complementary MSPs so
    # ns/nl + as/al tile cleanly (required by fibertools / FIRE).
    analyzed_span = _analyzed_span_from_request(
        _AnalyzedSpanRequest(
            apply_result=apply_result,
            read_length=read_length,
            kept=kept,
        )
    )
    kept, new_msps = assemble_nuc_msp_tiling(
        kept,
        analyzed_span.start,
        analyzed_span.end,
        msp_min_size,
        nuc_min_size,
    )
    msp_pairs = _interval_pair_lists(new_msps)

    nuc_arrays = _nuc_call_arrays(kept)
    return {
        "ns": nuc_arrays.starts,
        "nl": nuc_arrays.lengths,
        "as": np.asarray(msp_pairs.starts, dtype=np.int32),
        "al": np.asarray(msp_pairs.lengths, dtype=np.int32),
        "ns_scores": None,
        "as_scores": None,
        **_nuc_call_quality_fields(kept),
        "tf_calls": tf_calls,
    }


def _build_fused_recall_result_with_nucs_circular(
    fiber_read: Mapping[str, Any],
    apply_result: Mapping[str, Any],
    llr_hit,
    llr_miss,
    min_llr: float,
    min_opps: int,
    unify_threshold: int,
    split_min_llr: float,
    split_min_opps: int,
    nuc_min_size: int,
    msp_min_size: int,
    phase_nrl: int = 0,
) -> dict:
    """nuc recall for circular reads: split/refine in tiled space, then project
    the refined nucs, MSPs and TF calls back to molecule coordinates."""
    obs = apply_result["encoded"]                     # 3x tiled observations
    tiled_len = len(obs)
    read_length = _circular_read_length(fiber_read, apply_result)
    tiled = _tiled_interval_arrays(apply_result)
    tiled_msps = _interval_pairs(tiled.msp_starts, tiled.msp_lengths)

    # 1) split + edge-refine in tiled coordinates (+ optional Pass-2 phase prior)
    tiled_nucs, tiled_access = recall_nucs_in_read(
        obs, tiled.nuc_starts, tiled.nuc_lengths, tiled_len, llr_hit, llr_miss,
        split_min_llr=split_min_llr, split_min_opps=split_min_opps,
        nuc_min_size=nuc_min_size, phase_nrl=phase_nrl,
    )

    # 2) re-derive MSPs (still tiled), then 3) TF recall on the refined structure
    tiled_new_msps = rederive_msps(tiled_msps, tiled_access, tiled_len, msp_min_size)
    tiled_msp_pairs = _interval_pair_lists(tiled_new_msps)
    tiled_nuc_starts, tiled_nuc_lengths = _nuc_call_start_length_lists(tiled_nucs)
    tiled_tf = run_tf_recall_stage(
        obs,
        tiled_nuc_starts, tiled_nuc_lengths,
        tiled_msp_pairs.starts, tiled_msp_pairs.lengths,
        tiled_len, llr_hit, llr_miss, min_llr, min_opps, unify_threshold,
    )
    # 3b) promote nucleosome-sized TF leaks back to nuc+ (still tiled)
    tiled_tf, tiled_nucs = _promote_large_tf_nucs_from_request(
        _PromoteLargeTfNucsRequest(
            tf_calls=tiled_tf,
            nuc_calls=tiled_nucs,
            obs=obs,
            llr_hit=llr_hit,
            llr_miss=llr_miss,
            unify_threshold=unify_threshold,
            nuc_min_size=nuc_min_size,
        )
    )

    # 4) project everything from tiled -> molecule
    tf_calls = project_center_tf_calls(tiled_tf, read_length)
    proj_nucs = project_center_nuc_calls(tiled_nucs, read_length)
    proj_msps = project_center_runs(
        tiled_msp_pairs.starts,
        _interval_ends(tiled_new_msps),
        read_length,
    )

    # 5) unify (circular-aware), re-tile (non-overlapping nucs + complementary
    # MSPs), then lay out for emission.
    kept = unify_circular_nuc_calls_with_tf_calls(
        proj_nucs, tf_calls, unify_threshold, read_length)
    kept, proj_msps = assemble_circular_nuc_msp_tiling(
        kept, read_length, msp_min_size, nuc_min_size)
    circular_ns = [(k.start, k.length) for k in kept]
    kept_starts, kept_lengths, _ = split_intervals_for_legacy(
        circular_ns, read_length, None)
    msp_starts, msp_lengths_split, _ = split_intervals_for_legacy(
        proj_msps, read_length, None)

    return {
        "ns": kept_starts,
        "nl": kept_lengths,
        "as": msp_starts,
        "al": msp_lengths_split,
        "ns_scores": None,
        "as_scores": None,
        **_nuc_call_quality_fields(kept),
        "tf_calls": tf_calls,
        "circular": True,
        "circular_read_length": read_length,
        "circular_ns": circular_ns,
        "circular_as": proj_msps,
    }
