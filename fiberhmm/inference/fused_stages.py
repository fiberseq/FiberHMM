"""Stage helpers for fused HMM apply plus TF recall inference."""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

from fiberhmm.inference.circular import project_center_tf_calls, split_intervals_for_legacy
from fiberhmm.inference.engine import _process_single_read
from fiberhmm.inference.tagging import (
    split_intervals,
    unify_circular_nucs_with_tf_calls,
    unify_nucs_with_tf_calls,
)
from fiberhmm.inference.tf_recaller import build_scan_intervals, call_tfs_in_interval


def apply_result_has_footprints(apply_result: Optional[Mapping[str, Any]]) -> bool:
    """Return whether an HMM apply result has annotations worth writing."""
    if apply_result is None:
        return False
    return len(apply_result["ns"]) > 0 or len(apply_result["as"]) > 0


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
    """Run the HMM apply stage and keep encoded observations for recall."""
    return _process_single_read(
        fiber_read,
        model,
        edge_trim,
        circular,
        mode,
        context_size,
        msp_min_size,
        nuc_min_size=nuc_min_size,
        with_scores=with_scores,
        return_posteriors=False,
        include_encoded=True,
    )


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
    """Build recall scan intervals and run the TF LLR scan over each one."""
    intervals = build_scan_intervals(
        ns,
        nl,
        msps,
        msp_lengths,
        read_length,
        unify_threshold=unify_threshold,
    )
    tf_calls = []
    for lo, hi in intervals:
        tf_calls.extend(
            call_tfs_in_interval(obs, lo, hi, llr_hit, llr_miss, min_llr, min_opps)
        )
    return tf_calls


def build_fused_recall_result(
    fiber_read: Mapping[str, Any],
    apply_result: Mapping[str, Any],
    llr_hit,
    llr_miss,
    min_llr: float,
    min_opps: int,
    unify_threshold: int,
    with_scores: bool,
) -> dict:
    """Run TF recall and nucleosome/TF unification after an HMM apply result."""
    ns = apply_result["ns"]
    nl = apply_result["nl"]
    msps = apply_result["as"]
    msp_lengths = apply_result["al"]
    is_circular = bool(apply_result.get("circular"))

    recall_ns = apply_result.get("tiled_ns", ns) if is_circular else ns
    recall_nl = apply_result.get("tiled_nl", nl) if is_circular else nl
    recall_msps = apply_result.get("tiled_as", msps) if is_circular else msps
    recall_msp_lengths = apply_result.get("tiled_al", msp_lengths) if is_circular else msp_lengths
    recall_read_length = len(apply_result["encoded"]) if is_circular else len(fiber_read["query_sequence"])

    tf_calls = run_tf_recall_stage(
        apply_result["encoded"],
        recall_ns,
        recall_nl,
        recall_msps,
        recall_msp_lengths,
        recall_read_length,
        llr_hit,
        llr_miss,
        min_llr,
        min_opps,
        unify_threshold,
    )
    if is_circular:
        read_length = int(apply_result.get("circular_read_length") or len(fiber_read["query_sequence"]))
        tf_calls = project_center_tf_calls(tf_calls, read_length)
        kept_nucs, nq_for_kept = unify_circular_nucs_with_tf_calls(
            apply_result.get("circular_ns", []),
            tf_calls,
            unify_threshold,
            read_length,
            apply_result.get("circular_ns_scores") if with_scores else None,
        )
        kept_starts, kept_lengths, kept_scores = split_intervals_for_legacy(
            kept_nucs,
            read_length,
            apply_result.get("circular_ns_scores") if with_scores else None,
        )
        msp_starts, msp_lengths_split, msp_scores = split_intervals_for_legacy(
            apply_result.get("circular_as", []),
            read_length,
            apply_result.get("circular_as_scores") if with_scores else None,
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

    kept_nucs, nq_for_kept = unify_nucs_with_tf_calls(
        ns,
        nl,
        tf_calls,
        unify_threshold,
        apply_result.get("ns_scores") if with_scores else None,
    )
    kept_starts, kept_lengths = split_intervals(kept_nucs)

    return {
        "ns": kept_starts,
        "nl": kept_lengths,
        "as": msps,
        "al": msp_lengths,
        "ns_scores": apply_result.get("ns_scores") if with_scores else None,
        "as_scores": apply_result.get("as_scores") if with_scores else None,
        "nq_for_kept_nucs": nq_for_kept,
        "tf_calls": tf_calls,
    }
