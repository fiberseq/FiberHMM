"""Validation primitives for molecule-disjoint strand-rescue benchmarks.

These helpers deliberately perturb only held-out target molecules.  Population
site discovery and TF-class fitting remain the responsibility of the ordinary
strand-rescue engine, so benchmark labels cannot leak into the learned prior.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import replace
from typing import Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np

from fiberhmm.inference.strand_rescue import (
    IntervalCall,
    ReadEvidence,
    SiteTemplate,
    _group_site_indices,
    fit_site_state_model,
    match_direct_calls,
    match_direct_site_indices,
    opportunity_conditioned_call_fraction,
)


def stable_molecule_fold(
    dataset_id: str,
    molecule_id: Iterable[str],
    folds: int,
    seed: str,
) -> int:
    """Assign one molecule/PCR-family representative to a stable fold."""
    if folds < 2:
        raise ValueError("folds must be at least two")
    payload = "\x1f".join((seed, dataset_id, *(str(value) for value in molecule_id)))
    value = int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big")
    return value % folds


def merge_accessible_intervals(
    msps: Sequence[IntervalCall], masked_tfs: Sequence[IntervalCall]
) -> list[IntervalCall]:
    """Return sorted MSPs after replacing selected TF calls by accessibility.

    Touching intervals are merged because a missed footprint would be part of
    the surrounding accessible segment.  Synthetic MSPs intentionally omit
    molecular-coordinate and source-ordinal metadata: they are validation
    inputs and can never be materialized back into a BAM.
    """
    intervals = sorted(
        [(int(call.start), int(call.end)) for call in (*msps, *masked_tfs)],
        key=lambda value: (value[0], value[1]),
    )
    if not intervals:
        return []
    merged: list[list[int]] = []
    for start, end in intervals:
        if end <= start:
            raise ValueError("accessible intervals must have positive length")
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [
        IntervalCall(start, end, ordinal=index)
        for index, (start, end) in enumerate(merged)
    ]


def mask_tf_calls_as_accessible(
    read: ReadEvidence,
    tf_indices: Sequence[int],
    *,
    variant_name: str,
) -> ReadEvidence:
    """Hide selected ordinary TF calls and merge them into the MSP stream."""
    selected = {int(index) for index in tf_indices}
    if not selected:
        raise ValueError("at least one TF call must be selected for masking")
    if min(selected) < 0 or max(selected) >= len(read.tfs):
        raise IndexError("masked TF index is outside the read TF list")
    masked = [call for index, call in enumerate(read.tfs) if index in selected]
    retained = [call for index, call in enumerate(read.tfs) if index not in selected]
    return replace(
        read,
        name=str(variant_name),
        tfs=retained,
        msps=merge_accessible_intervals(read.msps, masked),
        # Synthetic accessible intervals have no trustworthy molecular-tag
        # ordinal.  The benchmark never emits action BAMs from these copies.
        molecular_tfs=None,
        molecular_msps=None,
    )


def thin_interval_opportunities(
    read: ReadEvidence,
    intervals: Sequence[Tuple[int, int]],
    retention: float,
    *,
    stable_key: str,
) -> tuple[ReadEvidence, dict]:
    """Deterministically retain a nested fraction of opportunities in intervals.

    The same ``stable_key`` yields a single hashed opportunity order, so lower
    retention fractions are strict subsets of higher fractions.  All aligned
    evidence arrays, including nucleosome steps, remain position-aligned.
    """
    if not math.isfinite(retention) or not 0.0 <= retention <= 1.0:
        raise ValueError("retention must be finite and in [0,1]")
    normalized = sorted((int(start), int(end)) for start, end in intervals)
    if any(end <= start for start, end in normalized):
        raise ValueError("thinning intervals must have positive length")

    inside = np.zeros(read.positions.shape[0], dtype=bool)
    for start, end in normalized:
        inside |= (read.positions >= start) & (read.positions < end)
    inside_indices = np.flatnonzero(inside)
    retain_count = int(math.floor(retention * inside_indices.size + 0.5))
    retain_count = max(0, min(int(inside_indices.size), retain_count))

    ranked = sorted(
        (int(index) for index in inside_indices),
        key=lambda index: hashlib.sha256(
            "\x1f".join(
                (
                    str(stable_key),
                    str(int(read.positions[index])),
                    str(index),
                )
            ).encode("utf-8")
        ).digest(),
    )
    retained_inside = set(ranked[:retain_count])
    keep = ~inside
    if retained_inside:
        keep[np.fromiter(sorted(retained_inside), dtype=np.int64)] = True

    original_hits = int(np.sum(read.hits[inside]))
    retained_hits = int(np.sum(read.hits[keep & inside]))
    thinned = replace(
        read,
        positions=np.asarray(read.positions[keep]).copy(),
        steps=np.asarray(read.steps[keep]).copy(),
        hits=np.asarray(read.hits[keep]).copy(),
        contexts=np.asarray(read.contexts[keep]).copy(),
        nuc_steps=(
            np.asarray(read.nuc_steps[keep]).copy()
            if read.nuc_steps is not None
            else None
        ),
    )
    return thinned, {
        "retention_requested": float(retention),
        "opportunities_original": int(inside_indices.size),
        "opportunities_retained": int(retain_count),
        "hits_original": original_hits,
        "hits_retained": retained_hits,
    }


def fit_source_supported_sites(
    training_reads: Sequence[ReadEvidence],
    sites: Sequence[SiteTemplate],
    *,
    min_source_support: int,
    min_source_local_enrichment: float,
    center_radius: int,
    edge_compatibility_bp: int,
) -> tuple[dict[str, set[int]], dict[str, dict[int, dict]]]:
    """Reproduce the engine's directional source-support gate."""
    strands = sorted({read.strand for read in training_reads})
    if len(strands) != 2:
        raise ValueError(f"source-support fitting requires two strands, observed {strands}")
    by_strand = {
        strand: [read for read in training_reads if read.strand == strand]
        for strand in strands
    }
    models = {
        strand: {
            index: fit_site_state_model(
                strand_reads,
                site,
                center_radius=center_radius,
                edge_compatibility_bp=edge_compatibility_bp,
            )
            for index, site in enumerate(sites)
        }
        for strand, strand_reads in by_strand.items()
    }
    supported: dict[str, set[int]] = {}
    for target in strands:
        source = strands[1] if target == strands[0] else strands[0]
        supported[target] = {
            index
            for index, site in enumerate(sites)
            if site.support.get(source, 0) >= min_source_support
            and float(
                (site.local_enrichment_by_strand or {}).get(
                    source, site.local_enrichment
                )
            )
            >= min_source_local_enrichment
            and opportunity_conditioned_call_fraction(
                site, source, models[source][index]
            )
            >= opportunity_conditioned_call_fraction(
                site, target, models[target][index]
            )
        }
    return supported, models


def positive_mask_events(
    read: ReadEvidence,
    sites: Sequence[SiteTemplate],
    source_supported: Mapping[str, set[int]],
    *,
    center_radius: int,
) -> list[dict]:
    """Return held-out baseline TF calls eligible for directional masking."""
    call_index = {id(call): index for index, call in enumerate(read.tfs)}
    eligible_calls = [call for call in read.tfs if call.geometry_eligible]
    events = []
    for call, site_index in match_direct_calls(
        eligible_calls, sites, center_radius=center_radius
    ):
        if site_index not in source_supported.get(read.strand, set()):
            continue
        site = sites[site_index]
        if not read.fully_maps(site.start, site.end):
            continue
        llr, opportunities, hits = read.interval_evidence(site.start, site.end)
        events.append(
            {
                "site_index": int(site_index),
                "site_id": site.site_id,
                "tf_index": int(call_index[id(call)]),
                "call_start": int(call.start),
                "call_end": int(call.end),
                "site_start": int(site.start),
                "site_end": int(site.end),
                "opportunities": int(opportunities),
                "hits": int(hits),
                "llr": float(llr),
            }
        )
    return sorted(events, key=lambda value: (value["site_start"], value["site_end"]))


def negative_accessible_events(
    read: ReadEvidence,
    sites: Sequence[SiteTemplate],
    source_supported: Mapping[str, set[int]],
    *,
    center_radius: int,
    accessible_site_gap: int,
) -> list[dict]:
    """Enumerate same-family baseline-negative MSP opportunities.

    This mirrors the analyzer's pre-likelihood containment, blocking, and
    full-mapping rules.  It intentionally does not require a positive LLR:
    local evidence is the tested outcome, not a matching covariate.
    """
    supported = set(source_supported.get(read.strand, set()))
    if not supported:
        return []
    direct = match_direct_site_indices(
        [call for call in read.tfs if call.geometry_eligible],
        sites,
        center_radius=center_radius,
    )
    existing_tf_overlaps = {
        index
        for index, site in enumerate(sites)
        if any(site.start < call.end and call.start < site.end for call in read.tfs)
    }
    nuc_overlaps = {
        index
        for index, site in enumerate(sites)
        if any(site.start < call.end and call.start < site.end for call in read.nucs)
    }
    blocked = existing_tf_overlaps | nuc_overlaps | {
        index
        for index, site in enumerate(sites)
        if any(
            site.start < sites[other].end and sites[other].start < site.end
            for other in direct
        )
    }
    by_site: Dict[int, dict] = {}
    for msp in read.msps:
        centered = [
            index
            for index in supported
            if index not in blocked and msp.start <= sites[index].center < msp.end
        ]
        contained = [
            index
            for index in centered
            if msp.start <= sites[index].start and sites[index].end <= msp.end
        ]
        relevant = [
            index
            for index in contained
            if read.fully_maps(sites[index].start, sites[index].end)
        ]
        for group in _group_site_indices(relevant, sites, accessible_site_gap):
            for site_index in group:
                site = sites[site_index]
                llr, opportunities, hits = read.interval_evidence(site.start, site.end)
                candidate = {
                    "site_index": int(site_index),
                    "site_id": site.site_id,
                    "site_start": int(site.start),
                    "site_end": int(site.end),
                    "msp_start": int(msp.start),
                    "msp_end": int(msp.end),
                    "msp_length": int(msp.end - msp.start),
                    "competing_sites": int(len(group)),
                    "opportunities": int(opportunities),
                    "hits": int(hits),
                    "llr": float(llr),
                }
                previous = by_site.get(site_index)
                if previous is None or (
                    candidate["msp_length"], candidate["msp_start"]
                ) < (previous["msp_length"], previous["msp_start"]):
                    by_site[site_index] = candidate
    return [by_site[index] for index in sorted(by_site)]


def site_prediction(
    decisions: Sequence[Mapping[str, object]], site: SiteTemplate
) -> dict:
    """Return the strongest decision that nominates one canonical site."""
    interval = [int(site.start), int(site.end)]
    matching = [
        decision
        for decision in decisions
        if interval in decision.get("proposed_site_intervals", [])
    ]
    if not matching:
        return {
            "q0": 0.0,
            "q1": 0.0,
            "q2": 0.0,
            "decision_id": None,
            "proposal_tier": None,
            "proposed_site_intervals": [],
        }
    best = max(
        matching,
        key=lambda value: (
            float(value["sr_hypothesis_probability"]),
            str(value["decision_id"]),
        ),
    )
    component_index = best["proposed_site_intervals"].index(interval)
    edge_confidence = best.get("proposed_site_edge_confidence", [])
    q1, q2 = (
        edge_confidence[component_index]
        if component_index < len(edge_confidence)
        else (0.0, 0.0)
    )
    return {
        "q0": float(best["sr_hypothesis_probability"]),
        "q1": float(q1),
        "q2": float(q2),
        "decision_id": str(best["decision_id"]),
        "proposal_tier": str(best["proposal_tier"]),
        "proposed_site_intervals": [
            [int(start), int(end)]
            for start, end in best["proposed_site_intervals"]
        ],
    }
