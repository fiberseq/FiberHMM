"""Stage helpers for fused HMM apply plus TF recall inference."""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

import numpy as np

from fiberhmm.inference.circular import project_center_tf_calls, split_intervals_for_legacy
from fiberhmm.inference.engine import _process_single_read
from fiberhmm.inference.nuc_recaller import (
    recall_nucs_in_read,
    rederive_msps,
    unify_nuc_calls_with_tf_calls,
)
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
    recall_nucs: bool = False,
    split_min_llr: float = 4.0,
    split_min_opps: int = 3,
    nuc_min_size: int = 85,
    msp_min_size: int = 0,
) -> dict:
    """Run TF recall and nucleosome/TF unification after an HMM apply result.

    When ``recall_nucs`` is True (and the read is not circular), the per-read
    nucleosome recaller runs FIRST: it splits over-merged footprints on
    accessible evidence and refines each nucleosome's conservative edges +
    quality (nuc+QQQ). MSPs are then re-derived from the new boundaries, and TF
    recall runs over the cleaner accessible space. ``recall_nucs=False`` (the
    default) is byte-for-byte the original behavior.
    """
    ns = apply_result["ns"]
    nl = apply_result["nl"]
    msps = apply_result["as"]
    msp_lengths = apply_result["al"]
    is_circular = bool(apply_result.get("circular"))

    if recall_nucs and not is_circular:
        return _build_fused_recall_result_with_nucs(
            fiber_read, apply_result, llr_hit, llr_miss,
            min_llr, min_opps, unify_threshold,
            split_min_llr, split_min_opps, nuc_min_size, msp_min_size,
        )

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
) -> dict:
    """nuc recall -> MSP re-derive -> TF recall (non-circular only)."""
    obs = apply_result["encoded"]
    read_length = len(fiber_read["query_sequence"])
    ns = apply_result["ns"]
    nl = apply_result["nl"]
    orig_msps = list(
        zip((int(s) for s in apply_result["as"]), (int(x) for x in apply_result["al"]))
    )

    # 1) split + edge-refine footprints
    nuc_calls, access = recall_nucs_in_read(
        obs, ns, nl, read_length, llr_hit, llr_miss,
        split_min_llr=split_min_llr, split_min_opps=split_min_opps,
        nuc_min_size=nuc_min_size,
    )

    # 2) re-derive MSPs from the new nucleosome boundaries
    new_msps = rederive_msps(orig_msps, access, read_length, msp_min_size)
    msp_starts = [s for s, _ in new_msps]
    msp_len = [length for _, length in new_msps]

    # 3) TF recall over the cleaner accessible space + short refined nucs
    refined_ns = [nc.start for nc in nuc_calls]
    refined_nl = [nc.length for nc in nuc_calls]
    tf_calls = run_tf_recall_stage(
        obs, refined_ns, refined_nl, msp_starts, msp_len, read_length,
        llr_hit, llr_miss, min_llr, min_opps, unify_threshold,
    )

    # 4) unify: drop short refined nucs overlapped by a TF call (carry nq/el/er)
    kept = unify_nuc_calls_with_tf_calls(nuc_calls, tf_calls, unify_threshold)

    kept_starts = np.asarray([k.start for k in kept], dtype=np.int32)
    kept_lengths = np.asarray([k.length for k in kept], dtype=np.int32)
    return {
        "ns": kept_starts,
        "nl": kept_lengths,
        "as": np.asarray(msp_starts, dtype=np.int32),
        "al": np.asarray(msp_len, dtype=np.int32),
        "ns_scores": None,
        "as_scores": None,
        "nq_for_kept_nucs": [k.nq for k in kept],
        "nuc_el_for_kept": [k.el for k in kept],
        "nuc_er_for_kept": [k.er for k in kept],
        "tf_calls": tf_calls,
    }
