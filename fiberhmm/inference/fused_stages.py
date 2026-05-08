"""Stage helpers for fused HMM apply plus TF recall inference."""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

from fiberhmm.inference.engine import _process_single_read
from fiberhmm.inference.tagging import split_intervals, unify_nucs_with_tf_calls
from fiberhmm.inference.tf_recaller import build_scan_intervals, call_tfs_in_interval


def _as_list(values: Any) -> list:
    if hasattr(values, "tolist"):
        return values.tolist()
    return list(values)


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

    ns_list = _as_list(ns)
    nl_list = _as_list(nl)
    msp_list = _as_list(msps)
    msp_length_list = _as_list(msp_lengths)

    tf_calls = run_tf_recall_stage(
        apply_result["encoded"],
        ns_list,
        nl_list,
        msp_list,
        msp_length_list,
        len(fiber_read["query_sequence"]),
        llr_hit,
        llr_miss,
        min_llr,
        min_opps,
        unify_threshold,
    )
    kept_nucs, nq_for_kept = unify_nucs_with_tf_calls(
        ns_list,
        nl_list,
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
