"""Stage helpers for fused HMM apply plus TF recall inference."""

from __future__ import annotations

from dataclasses import replace
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
    exclude_nucleosomes_from_msps,
    promote_large_tf_calls,
    recall_nucs_in_read,
    rederive_msps,
    unify_circular_nuc_calls_with_tf_calls,
    unify_nuc_calls_with_tf_calls,
    validate_radial_access_in_read,
)


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
from fiberhmm.inference.tagging import (
    split_intervals,
    unify_circular_nucs_with_tf_calls,
    unify_nucs_with_tf_calls,
)
from fiberhmm.inference.tf_recaller import (
    N_CTX,
    build_scan_intervals,
    call_tfs_in_interval,
)


def filter_nuc_derived_tf_calls(
    tf_calls,
    original_scan_intervals,
    obs,
    max_edge_ambiguity: int | None,
):
    """Require two-sided boundary evidence for TF calls exposed by nuc recall.

    Calls whose center was already in the HMM MSP/short-footprint scan space
    retain the ordinary TF-recaller contract. Calls made possible only because
    nucleosome refinement opened new scan space must have a deamination hit
    within ``max_edge_ambiguity`` bp on both sides. This prevents the DddA
    radial template's complement from turning every protected residue into a
    TF hypothesis while preserving the buried, sharply bracketed TFs that nuc
    recall is intended to recover. Boundary evidence is measured against the
    full molecule, not the derived scan interval: the ordinary recaller stops
    at interval edges and would otherwise report a false zero ambiguity there.
    """
    if max_edge_ambiguity is None:
        return list(tf_calls)
    threshold = max(0, int(max_edge_ambiguity))
    original = [(int(start), int(end)) for start, end in original_scan_intervals]
    observations = np.asarray(obs)

    def _ambiguity_to_hit(index, step):
        ambiguity = 0
        while 0 <= index < len(observations):
            code = int(observations[index])
            if 0 <= code < N_CTX:  # deamination/methylation hit code
                return ambiguity
            ambiguity += 1
            if ambiguity > threshold:
                break
            index += step
        return None

    kept = []
    for call in tf_calls:
        center = int(call.start) + int(call.length) // 2
        was_scannable = any(start <= center < end for start, end in original)
        if was_scannable:
            kept.append(call)
            continue
        left = _ambiguity_to_hit(int(call.start) - 1, -1)
        right = _ambiguity_to_hit(int(call.start) + int(call.length), 1)
        if left is not None and right is not None:
            # Correct any interval-edge-truncated ambiguity before AQ emission.
            kept.append(replace(
                call,
                left_ambiguity=int(left),
                right_ambiguity=int(right),
            ))
    return kept


def apply_result_has_footprints(apply_result: Optional[Mapping[str, Any]]) -> bool:
    """Return whether an HMM apply result has annotations worth writing."""
    if apply_result is None:
        return False
    return len(apply_result["ns"]) > 0 or len(apply_result["as"]) > 0


def payload_cpg_mask(payload: Mapping[str, Any], read_length: int,
                     policy: str):
    """DddA CpG mask for one slim payload (see ``make_apply_payload``).

    Same policy as ``fiberhmm-recall-tfs``: with ``unmethylated-only`` every
    CpG is excluded except inside the molecule's ``ddda_ucg`` island calls.
    """
    from fiberhmm.inference.tf_recaller import cpg_mask_from_intervals

    intervals = payload.get('_cpg_ma_intervals') or {}
    return cpg_mask_from_intervals(
        read_length, policy,
        intervals.get('ucg', ()), intervals.get('mcg', ()),
    )


def run_ddda_mcg_stage(
    observation_payload,
    apply_result: Mapping[str, Any],
    read_length: int,
):
    """Call per-read DddA mCG after apply and return its TF mask + spans.

    The preliminary HMM nucleosomes are the accessibility exclusion used by
    the validated standalone tagger. This stage deliberately runs before
    nucleosome/TF recall so the same-molecule mask can alter TF emissions in
    that recall pass.
    """
    if not observation_payload or read_length <= 0:
        return None, []
    from fiberhmm.daf.m5c import (
        DDDA_FIVE_PRIME_FACTORS,
        call_read_m5c,
        observations_from_ddda_mcg_payload,
    )

    excluded = [
        (int(start), int(start) + int(length))
        for start, length in zip(apply_result.get("ns", ()),
                                 apply_result.get("nl", ()))
        if int(length) > 0
    ]
    observations = observations_from_ddda_mcg_payload(
        observation_payload, excluded,
    )
    if not observations:
        return None, []
    result = call_read_m5c(
        observations,
        DDDA_FIVE_PRIME_FACTORS,
        expected_run_bp=5000.0,
        posterior_threshold=0.99,
        baseline_radius=250,
        min_other=10,
        min_call_cpg=2,
        max_call_gap_bp=5000.0,
    )
    spans = [(int(call.start), int(call.end)) for call in result.calls]
    if not spans:
        return None, []
    mask = np.zeros(int(read_length), dtype=bool)
    for lo, hi in spans:
        mask[max(0, lo):min(int(read_length), hi)] = True
    return mask, spans


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
    m5c_mask=None,
    m5c_llr_hit=None,
    m5c_llr_miss=None,
) -> list:
    """Build TF scan domains and decode native multi-interval configurations."""
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
        if m5c_mask is None:
            calls = call_tfs_in_interval(
                obs, lo, hi, llr_hit, llr_miss, min_llr, min_opps,
            )
        else:
            calls = call_tfs_in_interval(
                obs, lo, hi, llr_hit, llr_miss, min_llr, min_opps,
                m5c_mask=m5c_mask, m5c_llr_hit=m5c_llr_hit,
                m5c_llr_miss=m5c_llr_miss,
            )
        tf_calls.extend(calls)
    return tf_calls


def finalize_baseline_radial_nuc_configuration(
    obs,
    original_ns,
    original_nl,
    original_msps,
    radial_nucs,
    read_length,
    llr_hit,
    llr_miss,
    split_min_llr,
    split_min_opps,
    nuc_min_size,
    msp_min_size,
    nuc_profile=None,
    m5c_mask=None,
    m5c_llr_hit=None,
    m5c_llr_miss=None,
):
    """Finish molecule-local DddA nucleosome refinement before TF recall.

    The radial/rotational pass nominates dyads, edges, and accessible residue.
    This validation uses only the molecule's chemistry evidence and the HMM
    topology.  It deliberately receives no TF calls, TF families, strand
    consensus, or population prior: baseline inference is strictly

        HMM -> nucleosome refinement -> TF refinement.

    A later, coverage-gated consensus stage may test these baseline
    nucleosomes against a population-supported TF hypothesis and write an
    ``nuc_sr`` overlay.  That is a separate operation.
    """
    nuc_calls, accessible = validate_radial_access_in_read(
        obs,
        original_ns,
        original_nl,
        radial_nucs,
        (),
        read_length,
        llr_hit,
        llr_miss,
        min_llr=split_min_llr,
        min_opps=split_min_opps,
        nuc_min_size=nuc_min_size,
        nuc_profile=nuc_profile,
        m5c_mask=m5c_mask,
        m5c_llr_hit=m5c_llr_hit,
        m5c_llr_miss=m5c_llr_miss,
    )
    msps = rederive_msps(
        original_msps, accessible, read_length, msp_min_size,
    )
    msps = exclude_nucleosomes_from_msps(
        msps, nuc_calls, msp_min_size,
    )
    return nuc_calls, msps


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
    nuc_profile=None,
    nuc_recall_policy: str = "conservative",
    m5c_mask=None,
    m5c_llr_hit=None,
    m5c_llr_miss=None,
    derived_tf_max_edge_ambiguity: int | None = None,
    nuc_llr_hit=None,
    nuc_llr_miss=None,
    nuc_m5c_llr_hit=None,
    nuc_m5c_llr_miss=None,
) -> dict:
    """Run TF recall and nucleosome/TF unification after an HMM apply result.

    When ``recall_nucs`` is True, the per-read nucleosome recaller runs FIRST:
    it splits over-merged footprints on accessible evidence and resolves each
    nucleosome according to ``nuc_recall_policy`` (nuc+QQQ). In DddA mode this
    includes the molecule-local radial/rotational edge model. MSPs are then
    re-derived from the fixed baseline nucleosome configuration, and TF recall
    runs once over the resulting accessible space. No TF-family, strand, or
    population evidence is available to this baseline path. Circular reads run
    the same order in tiled coordinates and project the refined nucs/MSPs/TFs
    back to the molecule.

    Coverage-gated TF consensus and its subsequent nucleosome reconciliation
    are separate downstream operations that write ``tf_sr``/``nuc_sr`` while
    preserving these ordinary ``tf``/``nuc`` calls.
    ``recall_nucs=False`` (the default) is byte-for-byte the original behavior.
    """
    ns = apply_result["ns"]
    nl = apply_result["nl"]
    msps = apply_result["as"]
    msp_lengths = apply_result["al"]
    is_circular = bool(apply_result.get("circular"))

    if recall_nucs:
        # Backward-compatible fallback: callers that do not supply a locked
        # nucleosome likelihood model retain the compatibility shared-table
        # behavior. Bundled DddA callers supply independent frozen tables so a
        # TF-emission update cannot silently retune radial nuc refinement.
        if nuc_llr_hit is None:
            nuc_llr_hit = llr_hit
        if nuc_llr_miss is None:
            nuc_llr_miss = llr_miss
        if nuc_m5c_llr_hit is None:
            nuc_m5c_llr_hit = m5c_llr_hit
        if nuc_m5c_llr_miss is None:
            nuc_m5c_llr_miss = m5c_llr_miss
        if is_circular:
            return _build_fused_recall_result_with_nucs_circular(
                fiber_read, apply_result, llr_hit, llr_miss,
                min_llr, min_opps, unify_threshold,
                split_min_llr, split_min_opps, nuc_min_size, msp_min_size,
                phase_nrl,
                nuc_profile,
                nuc_recall_policy,
                m5c_mask, m5c_llr_hit, m5c_llr_miss,
                derived_tf_max_edge_ambiguity=derived_tf_max_edge_ambiguity,
                nuc_llr_hit=nuc_llr_hit,
                nuc_llr_miss=nuc_llr_miss,
                nuc_m5c_llr_hit=nuc_m5c_llr_hit,
                nuc_m5c_llr_miss=nuc_m5c_llr_miss,
            )
        return _build_fused_recall_result_with_nucs(
            fiber_read, apply_result, llr_hit, llr_miss,
            min_llr, min_opps, unify_threshold,
            split_min_llr, split_min_opps, nuc_min_size, msp_min_size,
            phase_nrl,
            nuc_profile,
            nuc_recall_policy,
            m5c_mask, m5c_llr_hit, m5c_llr_miss,
            derived_tf_max_edge_ambiguity=derived_tf_max_edge_ambiguity,
            nuc_llr_hit=nuc_llr_hit,
            nuc_llr_miss=nuc_llr_miss,
            nuc_m5c_llr_hit=nuc_m5c_llr_hit,
            nuc_m5c_llr_miss=nuc_m5c_llr_miss,
        )

    recall_ns = apply_result.get("tiled_ns", ns) if is_circular else ns
    recall_nl = apply_result.get("tiled_nl", nl) if is_circular else nl
    recall_msps = apply_result.get("tiled_as", msps) if is_circular else msps
    recall_msp_lengths = apply_result.get("tiled_al", msp_lengths) if is_circular else msp_lengths
    recall_read_length = len(apply_result["encoded"]) if is_circular else len(fiber_read["query_sequence"])
    recall_m5c_mask = m5c_mask
    if is_circular and m5c_mask is not None and len(m5c_mask) != recall_read_length:
        read_length = int(
            apply_result.get("circular_read_length")
            or len(fiber_read["query_sequence"])
        )
        if len(m5c_mask) != read_length or recall_read_length % read_length:
            raise ValueError(
                "circular m5c mask must match one molecule or tiled observations"
            )
        recall_m5c_mask = np.tile(
            np.asarray(m5c_mask, dtype=bool), recall_read_length // read_length
        )

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
        recall_m5c_mask, m5c_llr_hit, m5c_llr_miss,
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
    phase_nrl: int = 0,
    nuc_profile=None,
    nuc_recall_policy: str = "conservative",
    m5c_mask=None,
    m5c_llr_hit=None,
    m5c_llr_miss=None,
    derived_tf_max_edge_ambiguity: int | None = None,
    nuc_llr_hit=None,
    nuc_llr_miss=None,
    nuc_m5c_llr_hit=None,
    nuc_m5c_llr_miss=None,
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
        obs, ns, nl, read_length, nuc_llr_hit, nuc_llr_miss,
        split_min_llr=split_min_llr, split_min_opps=split_min_opps,
        nuc_min_size=nuc_min_size, phase_nrl=phase_nrl,
        nuc_profile=nuc_profile, recall_policy=nuc_recall_policy,
    )

    # 2) finish baseline nucleosome refinement without TF/population evidence,
    # then re-derive MSPs.  3) run baseline TF refinement exactly once on that
    # fixed nucleosome configuration.
    if nuc_profile is not None:
        nuc_calls, new_msps = finalize_baseline_radial_nuc_configuration(
            obs,
            ns,
            nl,
            orig_msps,
            nuc_calls,
            read_length,
            nuc_llr_hit,
            nuc_llr_miss,
            split_min_llr,
            split_min_opps,
            nuc_min_size,
            msp_min_size,
            nuc_profile=nuc_profile,
            m5c_mask=m5c_mask,
            m5c_llr_hit=nuc_m5c_llr_hit,
            m5c_llr_miss=nuc_m5c_llr_miss,
        )
    else:
        new_msps = rederive_msps(orig_msps, access, read_length, msp_min_size)
    tf_calls = run_tf_recall_stage(
        obs,
        [nc.start for nc in nuc_calls],
        [nc.length for nc in nuc_calls],
        [start for start, _ in new_msps],
        [length for _, length in new_msps],
        read_length,
        llr_hit,
        llr_miss,
        min_llr,
        min_opps,
        unify_threshold,
        m5c_mask,
        m5c_llr_hit,
        m5c_llr_miss,
    )
    # 3b) promote nucleosome-sized TF leaks (>= unify_threshold) back to nuc+
    tf_calls, promoted = promote_large_tf_calls(
        tf_calls, obs, llr_hit, llr_miss, unify_threshold, nuc_min_size,
        preserve_fragment=nuc_recall_policy == "topology")
    if nuc_profile is not None and derived_tf_max_edge_ambiguity is not None:
        original_scan = build_scan_intervals(
            ns,
            nl,
            [start for start, _ in orig_msps],
            [length for _, length in orig_msps],
            read_length,
            unify_threshold=unify_threshold,
        )
        tf_calls = filter_nuc_derived_tf_calls(
            tf_calls,
            original_scan,
            obs,
            derived_tf_max_edge_ambiguity,
        )
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
    msp_starts = [s for s, _ in new_msps]
    msp_len = [length for _, length in new_msps]

    return {
        "ns": np.asarray([k.start for k in kept], dtype=np.int32),
        "nl": np.asarray([k.length for k in kept], dtype=np.int32),
        "as": np.asarray(msp_starts, dtype=np.int32),
        "al": np.asarray(msp_len, dtype=np.int32),
        "ns_scores": None,
        "as_scores": None,
        "nq_for_kept_nucs": [k.nq for k in kept],
        "nuc_el_for_kept": [k.el for k in kept],
        "nuc_er_for_kept": [k.er for k in kept],
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
    nuc_profile=None,
    nuc_recall_policy: str = "conservative",
    m5c_mask=None,
    m5c_llr_hit=None,
    m5c_llr_miss=None,
    derived_tf_max_edge_ambiguity: int | None = None,
    nuc_llr_hit=None,
    nuc_llr_miss=None,
    nuc_m5c_llr_hit=None,
    nuc_m5c_llr_miss=None,
) -> dict:
    """nuc recall for circular reads: split/refine in tiled space, then project
    the refined nucs, MSPs and TF calls back to molecule coordinates."""
    obs = apply_result["encoded"]                     # 3x tiled observations
    tiled_len = len(obs)
    read_length = int(apply_result.get("circular_read_length")
                      or len(fiber_read["query_sequence"]))
    if derived_tf_max_edge_ambiguity is not None:
        required_tiled_keys = ("tiled_ns", "tiled_nl", "tiled_as", "tiled_al")
        missing = [key for key in required_tiled_keys if key not in apply_result]
        if missing:
            raise ValueError(
                "circular DddA derived-TF edge gating requires tiled HMM "
                "nucleosome/MSP coordinates; missing " + ", ".join(missing)
            )
    tiled_ns = apply_result.get("tiled_ns", apply_result["ns"])
    tiled_nl = apply_result.get("tiled_nl", apply_result["nl"])
    tiled_msps = list(zip(
        (int(s) for s in apply_result.get("tiled_as", apply_result["as"])),
        (int(x) for x in apply_result.get("tiled_al", apply_result["al"])),
    ))

    # 1) split + edge-refine in tiled coordinates (+ optional Pass-2 phase prior)
    tiled_nucs, tiled_access = recall_nucs_in_read(
        obs, tiled_ns, tiled_nl, tiled_len, nuc_llr_hit, nuc_llr_miss,
        split_min_llr=split_min_llr, split_min_opps=split_min_opps,
        nuc_min_size=nuc_min_size, phase_nrl=phase_nrl,
        nuc_profile=nuc_profile, recall_policy=nuc_recall_policy,
    )

    tiled_m5c_mask = m5c_mask
    if m5c_mask is not None and len(m5c_mask) != tiled_len:
        if len(m5c_mask) != read_length or tiled_len % read_length:
            raise ValueError("circular m5c mask must match one molecule or tiled observations")
        tiled_m5c_mask = np.tile(np.asarray(m5c_mask, dtype=bool),
                                 tiled_len // read_length)
    # 2) finish molecule-local baseline nucleosome refinement and re-derive
    # tiled MSPs. 3) run TF refinement once on that fixed configuration.
    if nuc_profile is not None:
        tiled_nucs, tiled_new_msps = finalize_baseline_radial_nuc_configuration(
            obs,
            tiled_ns,
            tiled_nl,
            tiled_msps,
            tiled_nucs,
            tiled_len,
            nuc_llr_hit,
            nuc_llr_miss,
            split_min_llr,
            split_min_opps,
            nuc_min_size,
            msp_min_size,
            nuc_profile=nuc_profile,
            m5c_mask=tiled_m5c_mask,
            m5c_llr_hit=nuc_m5c_llr_hit,
            m5c_llr_miss=nuc_m5c_llr_miss,
        )
    else:
        tiled_new_msps = rederive_msps(
            tiled_msps, tiled_access, tiled_len, msp_min_size,
        )
    tiled_tf = run_tf_recall_stage(
        obs,
        [nc.start for nc in tiled_nucs], [nc.length for nc in tiled_nucs],
        [s for s, _ in tiled_new_msps], [length for _, length in tiled_new_msps],
        tiled_len, llr_hit, llr_miss, min_llr, min_opps, unify_threshold,
        tiled_m5c_mask, m5c_llr_hit, m5c_llr_miss,
    )
    # 3b) promote nucleosome-sized TF leaks back to nuc+ (still tiled)
    tiled_tf, tiled_promoted = promote_large_tf_calls(
        tiled_tf, obs, llr_hit, llr_miss, unify_threshold, nuc_min_size,
        preserve_fragment=nuc_recall_policy == "topology")
    if nuc_profile is not None and derived_tf_max_edge_ambiguity is not None:
        original_scan = build_scan_intervals(
            tiled_ns,
            tiled_nl,
            [start for start, _ in tiled_msps],
            [length for _, length in tiled_msps],
            tiled_len,
            unify_threshold=unify_threshold,
        )
        tiled_tf = filter_nuc_derived_tf_calls(
            tiled_tf,
            original_scan,
            obs,
            derived_tf_max_edge_ambiguity,
        )
    tiled_nucs = drop_short_nucs_overlapping_promoted(
        tiled_nucs, tiled_promoted, unify_threshold) + tiled_promoted

    # 4) project everything from tiled -> molecule
    tf_calls = project_center_tf_calls(tiled_tf, read_length)
    proj_nucs = project_center_nuc_calls(tiled_nucs, read_length)
    proj_msps = project_center_runs(
        [s for s, _ in tiled_new_msps],
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

    return {
        "ns": kept_starts,
        "nl": kept_lengths,
        "as": msp_starts,
        "al": msp_lengths_split,
        "ns_scores": None,
        "as_scores": None,
        "nq_for_kept_nucs": [k.nq for k in kept],
        "nuc_el_for_kept": [k.el for k in kept],
        "nuc_er_for_kept": [k.er for k in kept],
        "tf_calls": tf_calls,
        "circular": True,
        "circular_read_length": read_length,
        "circular_ns": circular_ns,
        "circular_as": proj_msps,
    }
