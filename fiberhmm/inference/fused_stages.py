"""Stage helpers for fused HMM apply plus TF recall inference."""

from __future__ import annotations

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


def _analyzed_span(apply_result, read_length, kept):
    """Extent (lo, hi) the read was annotated over -- the union of the original
    HMM footprints/MSPs and the final nucleosomes -- used to tile MSPs."""
    starts, ends = [], []
    for ks, kl in (("ns", "nl"), ("as", "al")):
        for s, length in zip(apply_result.get(ks, ()), apply_result.get(kl, ())):
            starts.append(int(s))
            ends.append(int(s) + int(length))
    for k in kept:
        starts.append(int(k.start))
        ends.append(int(k.start) + int(k.length))
    return (min(starts) if starts else 0,
            max(ends) if ends else int(read_length))


def _interval_pair_lists(intervals):
    return [s for s, _ in intervals], [length for _, length in intervals]


def _nuc_call_quality_lists(nuc_calls):
    return (
        [k.nq for k in nuc_calls],
        [k.el for k in nuc_calls],
        [k.er for k in nuc_calls],
    )


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
    ns = apply_result["ns"]
    nl = apply_result["nl"]
    msps = apply_result["as"]
    msp_lengths = apply_result["al"]
    is_circular = bool(apply_result.get("circular"))

    if recall_nucs:
        if is_circular:
            return _build_fused_recall_result_with_nucs_circular(
                fiber_read, apply_result, llr_hit, llr_miss,
                min_llr, min_opps, unify_threshold,
                split_min_llr, split_min_opps, nuc_min_size, msp_min_size,
                phase_nrl,
            )
        return _build_fused_recall_result_with_nucs(
            fiber_read, apply_result, llr_hit, llr_miss,
            min_llr, min_opps, unify_threshold,
            split_min_llr, split_min_opps, nuc_min_size, msp_min_size,
            phase_nrl,
        )

    if is_circular:
        return _build_fused_recall_result_without_nucs_circular(
            fiber_read, apply_result, llr_hit, llr_miss,
            min_llr, min_opps, unify_threshold, with_scores,
        )
    return _build_fused_recall_result_without_nucs_linear(
        fiber_read, apply_result, llr_hit, llr_miss,
        min_llr, min_opps, unify_threshold, with_scores,
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
    read_length = int(apply_result.get("circular_read_length") or len(fiber_read["query_sequence"]))
    tf_calls = run_tf_recall_stage(
        apply_result["encoded"],
        apply_result.get("tiled_ns", apply_result["ns"]),
        apply_result.get("tiled_nl", apply_result["nl"]),
        apply_result.get("tiled_as", apply_result["as"]),
        apply_result.get("tiled_al", apply_result["al"]),
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
    orig_msps = list(
        zip((int(s) for s in apply_result["as"]), (int(x) for x in apply_result["al"]))
    )

    # 1) split + edge-refine footprints (+ optional Pass-2 phase prior)
    nuc_calls, access = recall_nucs_in_read(
        obs, ns, nl, read_length, llr_hit, llr_miss,
        split_min_llr=split_min_llr, split_min_opps=split_min_opps,
        nuc_min_size=nuc_min_size, phase_nrl=phase_nrl,
    )

    # 2) re-derive MSPs from the new nucleosome boundaries
    new_msps = rederive_msps(orig_msps, access, read_length, msp_min_size)
    msp_starts, msp_len = _interval_pair_lists(new_msps)

    # 3) TF recall over the cleaner accessible space + short refined nucs
    refined_ns = [nc.start for nc in nuc_calls]
    refined_nl = [nc.length for nc in nuc_calls]
    tf_calls = run_tf_recall_stage(
        obs, refined_ns, refined_nl, msp_starts, msp_len, read_length,
        llr_hit, llr_miss, min_llr, min_opps, unify_threshold,
    )

    # 3b) promote nucleosome-sized TF leaks (>= unify_threshold) back to nuc+
    tf_calls, promoted = promote_large_tf_calls(
        tf_calls, obs, llr_hit, llr_miss, unify_threshold, nuc_min_size)
    nuc_calls = drop_short_nucs_overlapping_promoted(
        nuc_calls, promoted, unify_threshold) + promoted

    # 4) unify: drop short refined nucs overlapped by a TF call (carry nq/el/er)
    kept = unify_nuc_calls_with_tf_calls(nuc_calls, tf_calls, unify_threshold)

    # 5) re-tile: split/phase/promotion can leave overlapping nucs + stale MSPs.
    # Clip to non-overlapping nucleosomes and derive complementary MSPs so
    # ns/nl + as/al tile cleanly (required by fibertools / FIRE).
    span_lo, span_hi = _analyzed_span(apply_result, read_length, kept)
    kept, new_msps = assemble_nuc_msp_tiling(
        kept, span_lo, span_hi, msp_min_size, nuc_min_size)
    msp_starts, msp_len = _interval_pair_lists(new_msps)

    nq_for_kept, nuc_el_for_kept, nuc_er_for_kept = _nuc_call_quality_lists(kept)
    return {
        "ns": np.asarray([k.start for k in kept], dtype=np.int32),
        "nl": np.asarray([k.length for k in kept], dtype=np.int32),
        "as": np.asarray(msp_starts, dtype=np.int32),
        "al": np.asarray(msp_len, dtype=np.int32),
        "ns_scores": None,
        "as_scores": None,
        "nq_for_kept_nucs": nq_for_kept,
        "nuc_el_for_kept": nuc_el_for_kept,
        "nuc_er_for_kept": nuc_er_for_kept,
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
    read_length = int(apply_result.get("circular_read_length")
                      or len(fiber_read["query_sequence"]))
    tiled_ns = apply_result.get("tiled_ns", apply_result["ns"])
    tiled_nl = apply_result.get("tiled_nl", apply_result["nl"])
    tiled_msps = list(zip(
        (int(s) for s in apply_result.get("tiled_as", apply_result["as"])),
        (int(x) for x in apply_result.get("tiled_al", apply_result["al"])),
    ))

    # 1) split + edge-refine in tiled coordinates (+ optional Pass-2 phase prior)
    tiled_nucs, tiled_access = recall_nucs_in_read(
        obs, tiled_ns, tiled_nl, tiled_len, llr_hit, llr_miss,
        split_min_llr=split_min_llr, split_min_opps=split_min_opps,
        nuc_min_size=nuc_min_size, phase_nrl=phase_nrl,
    )

    # 2) re-derive MSPs (still tiled), then 3) TF recall on the refined structure
    tiled_new_msps = rederive_msps(tiled_msps, tiled_access, tiled_len, msp_min_size)
    tiled_msp_starts, tiled_msp_lengths = _interval_pair_lists(tiled_new_msps)
    tiled_tf = run_tf_recall_stage(
        obs,
        [nc.start for nc in tiled_nucs], [nc.length for nc in tiled_nucs],
        tiled_msp_starts, tiled_msp_lengths,
        tiled_len, llr_hit, llr_miss, min_llr, min_opps, unify_threshold,
    )
    # 3b) promote nucleosome-sized TF leaks back to nuc+ (still tiled)
    tiled_tf, tiled_promoted = promote_large_tf_calls(
        tiled_tf, obs, llr_hit, llr_miss, unify_threshold, nuc_min_size)
    tiled_nucs = drop_short_nucs_overlapping_promoted(
        tiled_nucs, tiled_promoted, unify_threshold) + tiled_promoted

    # 4) project everything from tiled -> molecule
    tf_calls = project_center_tf_calls(tiled_tf, read_length)
    proj_nucs = project_center_nuc_calls(tiled_nucs, read_length)
    proj_msps = project_center_runs(
        tiled_msp_starts,
        [s + length for s, length in tiled_new_msps],
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

    nq_for_kept, nuc_el_for_kept, nuc_er_for_kept = _nuc_call_quality_lists(kept)
    return {
        "ns": kept_starts,
        "nl": kept_lengths,
        "as": msp_starts,
        "al": msp_lengths_split,
        "ns_scores": None,
        "as_scores": None,
        "nq_for_kept_nucs": nq_for_kept,
        "nuc_el_for_kept": nuc_el_for_kept,
        "nuc_er_for_kept": nuc_er_for_kept,
        "tf_calls": tf_calls,
        "circular": True,
        "circular_read_length": read_length,
        "circular_ns": circular_ns,
        "circular_as": proj_msps,
    }
