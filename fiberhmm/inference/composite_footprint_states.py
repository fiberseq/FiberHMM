"""Nominate recurrent component states compatible with a broader TF family.

This is deliberately a second layer beneath the stable parent-family call.
It tests whether the union of two or more narrower recurrent families can
reconstruct a broader recurrent family at the same locus.  The result is a
geometry/state nomination, not evidence of TF identity, stoichiometry, or
dimerization.  Independent motif and replicate evidence must be attached by
downstream analyses before a physical interpretation is made.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from typing import Mapping, Sequence


@dataclass(frozen=True)
class CompositeFootprintStateConfig:
    """Bounded controls for one targeted family catalog."""

    boundary_tolerance_bp: int = 18
    maximum_members: int = 12
    beam_width: int = 256
    maximum_nominations_per_envelope: int = 8
    maximum_component_width_fraction: float = 0.92
    minimum_width_difference_bp: int = 4
    minimum_union_coverage_fraction: float = 0.60
    minimum_unique_component_bp: int = 2
    minimum_geometry_score: float = 0.70


def _merged_intervals(
    intervals: Sequence[tuple[int, int]],
) -> tuple[tuple[int, int], ...]:
    merged: list[list[int]] = []
    for start, end in sorted((int(start), int(end)) for start, end in intervals):
        if end <= start:
            continue
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return tuple((start, end) for start, end in merged)


def _union_bp(intervals: Sequence[tuple[int, int]]) -> int:
    return sum(end - start for start, end in _merged_intervals(intervals))


def _family_key(family: Mapping[str, object]) -> str:
    return str(family.get("batch_family_id") or family["family_id"])


def _family_ref(family: Mapping[str, object]) -> dict:
    return {
        "family_id": _family_key(family),
        "source_family_id": str(family.get("family_id", _family_key(family))),
        "family_slot": int(family["family_slot"]),
        "contig": str(family["contig"]),
        "start": int(family["start"]),
        "end": int(family["end"]),
        "width": int(family["end"]) - int(family["start"]),
        "working_support_molecules": int(
            family.get("working_support_molecules", 0)
        ),
        "working_support_by_dataset": dict(
            family.get("working_support_by_dataset", {})
        ),
    }


def _clipped_interval(
    family: Mapping[str, object], envelope: Mapping[str, object]
) -> tuple[int, int] | None:
    start = max(int(family["start"]), int(envelope["start"]))
    end = min(int(family["end"]), int(envelope["end"]))
    return (start, end) if end > start else None


def _unique_component_bp(
    component_index: int, intervals: Sequence[tuple[int, int]]
) -> int:
    complete = _union_bp(intervals)
    without = _union_bp(
        interval
        for index, interval in enumerate(intervals)
        if index != component_index
    )
    return complete - without


def _support_summary(
    envelope: Mapping[str, object],
    components: Sequence[Mapping[str, object]],
    family_molecules: Mapping[str, set[str]],
    family_molecules_by_dataset: Mapping[str, Mapping[str, set[str]]],
) -> dict:
    roles = (envelope, *components)
    role_ids = [_family_key(family) for family in roles]
    role_sets = [set(family_molecules.get(family_id, set())) for family_id in role_ids]
    union = set().union(*role_sets) if role_sets else set()
    multi_state = {
        molecule_id
        for molecule_id in union
        if sum(molecule_id in values for values in role_sets) > 1
    }
    datasets = sorted(
        {
            dataset_id
            for family_id in role_ids
            for dataset_id in family_molecules_by_dataset.get(family_id, {})
        }
    )
    by_dataset = {}
    datasets_with_all_states = []
    for dataset_id in datasets:
        counts = [
            len(
                family_molecules_by_dataset
                .get(family_id, {})
                .get(dataset_id, set())
            )
            for family_id in role_ids
        ]
        if counts and all(count > 0 for count in counts):
            datasets_with_all_states.append(dataset_id)
        by_dataset[dataset_id] = {
            "envelope_molecules": counts[0],
            "component_molecules": counts[1:],
            "observed_state_count": sum(value > 0 for value in counts),
        }
    counts = [len(values) for values in role_sets]
    support_balance = (
        min(counts) / max(counts)
        if counts and min(counts) > 0 and max(counts) > 0
        else 0.0
    )
    exclusivity = (
        (len(union) - len(multi_state)) / len(union) if union else None
    )
    return {
        "envelope_molecules": counts[0] if counts else 0,
        "component_molecules": counts[1:],
        "union_molecules": len(union),
        "exclusive_state_molecules": len(union) - len(multi_state),
        "multi_state_molecules": len(multi_state),
        "state_exclusivity_fraction": (
            round(exclusivity, 4) if exclusivity is not None else None
        ),
        "support_balance": round(support_balance, 4),
        "by_dataset": by_dataset,
        "datasets_with_all_states": datasets_with_all_states,
    }


def _evaluate_subset(
    envelope: Mapping[str, object],
    components: Sequence[Mapping[str, object]],
    *,
    effective_tolerance: int,
    config: CompositeFootprintStateConfig,
    family_molecules: Mapping[str, set[str]],
    family_molecules_by_dataset: Mapping[str, Mapping[str, set[str]]],
) -> dict | None:
    envelope_start = int(envelope["start"])
    envelope_end = int(envelope["end"])
    envelope_width = envelope_end - envelope_start
    if envelope_width <= 0 or len(components) < 2:
        return None
    intervals = [_clipped_interval(family, envelope) for family in components]
    if any(interval is None for interval in intervals):
        return None
    component_intervals = [interval for interval in intervals if interval is not None]
    hull_start = min(int(family["start"]) for family in components)
    hull_end = max(int(family["end"]) for family in components)
    left_error = abs(hull_start - envelope_start)
    right_error = abs(hull_end - envelope_end)
    if left_error > effective_tolerance or right_error > effective_tolerance:
        return None
    unique_bp = [
        _unique_component_bp(index, component_intervals)
        for index in range(len(component_intervals))
    ]
    if any(value < config.minimum_unique_component_bp for value in unique_bp):
        return None
    union_bp = _union_bp(component_intervals)
    union_coverage = union_bp / envelope_width
    if union_coverage < config.minimum_union_coverage_fraction:
        return None
    summed_component_bp = sum(end - start for start, end in component_intervals)
    overlap_bp = max(0, summed_component_bp - union_bp)
    clipped_hull_start = max(envelope_start, hull_start)
    clipped_hull_end = min(envelope_end, hull_end)
    internal_gap_bp = max(0, clipped_hull_end - clipped_hull_start - union_bp)
    component_widths = [
        int(family["end"]) - int(family["start"]) for family in components
    ]
    edge_score = max(
        0.0,
        1.0 - (left_error + right_error) / max(1, 2 * effective_tolerance),
    )
    width_separation_score = min(
        1.0,
        (envelope_width - max(component_widths))
        / max(4.0, 0.20 * envelope_width),
    )
    redundancy_score = max(
        0.0, 1.0 - overlap_bp / max(1, summed_component_bp)
    )
    geometry_score = (
        0.35 * edge_score
        + 0.40 * min(1.0, union_coverage)
        + 0.15 * width_separation_score
        + 0.10 * redundancy_score
    )
    if geometry_score < config.minimum_geometry_score:
        return None
    support = _support_summary(
        envelope,
        components,
        family_molecules,
        family_molecules_by_dataset,
    )
    exclusivity = support["state_exclusivity_fraction"]
    if exclusivity is None:
        nomination_score = geometry_score
    else:
        assignment_score = (
            0.65 * float(exclusivity)
            + 0.35 * float(support["support_balance"])
        )
        nomination_score = 0.80 * geometry_score + 0.20 * assignment_score
    component_ids = [_family_key(family) for family in components]
    digest = hashlib.sha256(
        "\x1f".join((_family_key(envelope), *component_ids)).encode()
    ).hexdigest()[:12]
    return {
        "schema": "fiberhmm.composite_footprint_state.v1",
        "nomination_id": f"composite_state_{digest}",
        "interpretation": "composite-state-compatible",
        "evidence_level": (
            "geometry_plus_molecule_assignments"
            if support["envelope_molecules"]
            and all(support["component_molecules"])
            else "geometry_only"
        ),
        "envelope_family": _family_ref(envelope),
        "component_families": [_family_ref(family) for family in components],
        "component_count": len(components),
        "geometry": {
            "effective_boundary_tolerance_bp": effective_tolerance,
            "outer_boundary_error_bp": [left_error, right_error],
            "union_coverage_fraction": round(union_coverage, 4),
            "internal_gap_bp": internal_gap_bp,
            "component_overlap_bp": overlap_bp,
            "unique_component_bp": unique_bp,
            "geometry_score": round(geometry_score, 4),
        },
        "support": support,
        "nomination_score": round(nomination_score, 4),
        "claim_boundary": (
            "Recurrent interval geometry is compatible with a composite "
            "footprint state; factor identity, stoichiometry, dimerization, "
            "and other physical multimerization are not established."
        ),
    }


def _partial_path_priority(
    path: tuple[int, ...],
    components: Sequence[Mapping[str, object]],
    envelope: Mapping[str, object],
) -> tuple:
    selected = [components[index] for index in path]
    envelope_start = int(envelope["start"])
    envelope_width = max(1, int(envelope["end"]) - envelope_start)
    intervals = [
        interval
        for family in selected
        if (interval := _clipped_interval(family, envelope)) is not None
    ]
    coverage = _union_bp(intervals) / envelope_width
    right_progress = min(
        1.0,
        (max(int(family["end"]) for family in selected) - envelope_start)
        / envelope_width,
    )
    support = sum(
        int(family.get("working_support_molecules", 0)) for family in selected
    )
    return (-coverage, -right_progress, -support, len(path), path)


def nominate_composite_footprint_states(
    families: Sequence[Mapping[str, object]],
    *,
    family_molecules: Mapping[str, set[str]] | None = None,
    family_molecules_by_dataset: Mapping[
        str, Mapping[str, set[str]]
    ] | None = None,
    config: CompositeFootprintStateConfig | None = None,
) -> dict:
    """Return bounded, deterministic envelope/component nominations."""

    config = config or CompositeFootprintStateConfig()
    family_molecules = family_molecules or {}
    family_molecules_by_dataset = family_molecules_by_dataset or {}
    ordered = sorted(
        families,
        key=lambda family: (
            str(family["contig"]),
            int(family["start"]),
            int(family["end"]),
            _family_key(family),
        ),
    )
    by_contig: dict[str, list[Mapping[str, object]]] = {}
    for family in ordered:
        by_contig.setdefault(str(family["contig"]), []).append(family)
    nominations = []
    truncated_envelopes = []
    envelopes_considered = 0
    for envelope in ordered:
        envelope_width = int(envelope["end"]) - int(envelope["start"])
        if envelope_width <= config.minimum_width_difference_bp:
            continue
        tolerance = max(
            3,
            min(
                int(config.boundary_tolerance_bp),
                max(3, int(round(0.25 * envelope_width))),
            ),
        )
        components = [
            family
            for family in by_contig[str(envelope["contig"])]
            if _family_key(family) != _family_key(envelope)
            and int(family["end"]) > int(envelope["start"])
            and int(family["start"]) < int(envelope["end"])
            and int(family["start"]) >= int(envelope["start"]) - tolerance
            and int(family["end"]) <= int(envelope["end"]) + tolerance
            and int(family["end"]) - int(family["start"])
            <= config.maximum_component_width_fraction * envelope_width
            and envelope_width - (int(family["end"]) - int(family["start"]))
            >= config.minimum_width_difference_bp
        ]
        components.sort(
            key=lambda family: (
                int(family["start"]), int(family["end"]), _family_key(family)
            )
        )
        left_anchors = [
            index
            for index, family in enumerate(components)
            if abs(int(family["start"]) - int(envelope["start"])) <= tolerance
        ]
        if len(components) < 2 or not left_anchors:
            continue
        envelopes_considered += 1
        paths = [(index,) for index in left_anchors]
        candidates_by_subset = {}
        search_truncated = False
        maximum_depth = min(len(components), int(config.maximum_members))
        for _depth in range(1, maximum_depth + 1):
            expansions = set()
            for path in paths:
                if len(path) >= 2:
                    candidate = _evaluate_subset(
                        envelope,
                        [components[index] for index in path],
                        effective_tolerance=tolerance,
                        config=config,
                        family_molecules=family_molecules,
                        family_molecules_by_dataset=family_molecules_by_dataset,
                    )
                    if candidate is not None:
                        key = tuple(
                            family["family_id"]
                            for family in candidate["component_families"]
                        )
                        previous = candidates_by_subset.get(key)
                        if previous is None or candidate["nomination_score"] > previous[
                            "nomination_score"
                        ]:
                            candidates_by_subset[key] = candidate
                if len(path) >= maximum_depth:
                    continue
                current = [
                    _clipped_interval(components[index], envelope) for index in path
                ]
                current = [value for value in current if value is not None]
                current_bp = _union_bp(current)
                for next_index in range(path[-1] + 1, len(components)):
                    interval = _clipped_interval(components[next_index], envelope)
                    if interval is None:
                        continue
                    if _union_bp([*current, interval]) - current_bp < config.minimum_unique_component_bp:
                        continue
                    expansions.add((*path, next_index))
            if not expansions:
                break
            paths = sorted(
                expansions,
                key=lambda path: _partial_path_priority(path, components, envelope),
            )
            if len(paths) > config.beam_width:
                paths = paths[: config.beam_width]
                search_truncated = True
        if len(components) > maximum_depth:
            search_truncated = True
        ranked = sorted(
            candidates_by_subset.values(),
            key=lambda value: (
                -float(value["nomination_score"]),
                -float(value["geometry"]["geometry_score"]),
                int(value["component_count"]),
                value["nomination_id"],
            ),
        )
        candidate_count = len(ranked)
        if candidate_count > config.maximum_nominations_per_envelope:
            ranked = ranked[: config.maximum_nominations_per_envelope]
            search_truncated = True
        for rank, candidate in enumerate(ranked, start=1):
            candidate["rank_for_envelope"] = rank
            candidate["alternative_subset_count"] = candidate_count
            candidate["search_truncated_for_envelope"] = search_truncated
            nominations.append(candidate)
        if search_truncated:
            truncated_envelopes.append(_family_key(envelope))
    nominations.sort(
        key=lambda value: (
            -float(value["nomination_score"]),
            value["envelope_family"]["contig"],
            value["envelope_family"]["start"],
            value["nomination_id"],
        )
    )
    return {
        "schema": "fiberhmm.composite_footprint_state_nomination.v1",
        "nominations": nominations,
        "summary": {
            "family_count": len(ordered),
            "envelopes_considered": envelopes_considered,
            "nominated_envelopes": len(
                {value["envelope_family"]["family_id"] for value in nominations}
            ),
            "nomination_count": len(nominations),
            "search_truncated": bool(truncated_envelopes),
            "search_truncated_envelope_ids": truncated_envelopes,
        },
        "config": asdict(config),
        "contracts": {
            "parent_family_role": "stable_primary_call_preserved",
            "subset_cardinality": "two_or_more_variable_width_states",
            "support_role": "accepted_molecule_assignments_not_unbiased_occupancy",
            "claim_boundary": (
                "Nominations do not establish factor identity, stoichiometry, "
                "dimerization, or other physical multimerization."
            ),
        },
    }


__all__ = [
    "CompositeFootprintStateConfig",
    "nominate_composite_footprint_states",
]
