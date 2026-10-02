#!/usr/bin/env python3
"""Audit normalized ``nuc_sr``/``tf_sr`` MA/AQ/AN invariants."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import DefaultDict, List, Mapping, Optional, Sequence

import pysam

from fiberhmm.cli.strand_rescue_annotate import (
    ALL_HEADER_PREFIXES,
    HEADER_PREFIX,
    LEGACY_QUALITY_SPEC,
    LEGACY_HEADER_PREFIX,
    LAYER_ORDER,
    QUALITY_SPEC,
    V3_HEADER_PREFIX,
    V6_HEADER_PREFIX,
)
from fiberhmm.io.bam_header import declared_ma_types
from fiberhmm.io.ma_tags import parse_an_tag, parse_aq_array, parse_ma_tag
from fiberhmm.cli.tag_families import (
    FAMILY_HEADER_PREFIX,
    FAMILY_QUALITY_SPEC,
)


NAME_RE = re.compile(
    r"^(fhsr_[0-9a-f]{16})(?:_O(?P<origin>[0-9]+))?_"
    r"(?P<role>H|R(?P<index>[0-9]+))$"
)
LATENT_HEADER_PREFIX = "FIBERHMM-LATENT-TILING:v1:"
MULTIFAMILY_LATENT_HEADER_PREFIX = "FIBERHMM-LATENT-TILING:v2:"
POSTFAMILY_LATENT_HEADER_PREFIX = "FIBERHMM-LATENT-TILING:v3:"
LATENT_HEADER_PREFIXES = (
    LATENT_HEADER_PREFIX,
    MULTIFAMILY_LATENT_HEADER_PREFIX,
    POSTFAMILY_LATENT_HEADER_PREFIX,
)
LATENT_NAME_RE = re.compile(
    r"^(fhlt_[0-9a-f]{16})(?P<propagated>_P)?_"
    r"(?:R(?P<tf_index>[0-9]+)|N(?P<nuc_index>[0-9]+)_H)$"
)
THRESHOLDS = (0, 64, 128, 192, 255)
CONTRACT_PREFIXES = {
    6: V6_HEADER_PREFIX,
    4: HEADER_PREFIX,
    3: V3_HEADER_PREFIX,
    2: LEGACY_HEADER_PREFIX,
}
QUALITY_SPECS = {
    6: QUALITY_SPEC,
    4: QUALITY_SPEC,
    3: LEGACY_QUALITY_SPEC,
    2: LEGACY_QUALITY_SPEC,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _annotation_rows(read) -> List[dict]:
    if not read.has_tag("MA"):
        return []
    parsed = parse_ma_tag(read.get_tag("MA"))
    raw_types = parsed["raw_types"]
    aq = read.get_tag("AQ") if read.has_tag("AQ") else []
    expected = sum(
        len(specification) * len(intervals)
        for _name, _strand, specification, intervals in raw_types
    )
    if len(aq) != expected:
        raise ValueError(f"AQ has {len(aq)} bytes but MA requires {expected}")
    quality_rows = parse_aq_array(
        aq,
        [item[2] for item in raw_types],
        [len(item[3]) for item in raw_types],
    )
    annotation_count = sum(len(item[3]) for item in raw_types)
    if read.has_tag("AN"):
        names = parse_an_tag(read.get_tag("AN"))
        if len(names) != annotation_count:
            raise ValueError(
                f"AN has {len(names)} fields but MA has {annotation_count} annotations"
            )
    else:
        names = [""] * annotation_count
    rows = []
    cursor = 0
    for name, strand, specification, intervals in raw_types:
        for interval in intervals:
            rows.append(
                {
                    "type": name,
                    "strand": strand,
                    "quality_spec": specification,
                    "interval": tuple(int(value) for value in interval),
                    "qualities": tuple(int(value) for value in quality_rows[cursor]),
                    "annotation_name": names[cursor],
                }
            )
            cursor += 1
    return rows


def _reference_interval(read, annotation: Mapping[str, object], ma_read_length: int):
    """Project one MA interval with the same production path used by readers."""
    from fiberhmm.core.bam_reader import cigar_to_query_ref
    from fiberhmm.io.footprint_bam import _ma_interval_to_query, _project_query_interval

    start, length = annotation["interval"]
    query_start, query_end, complete = _ma_interval_to_query(
        start=int(start),
        length=int(length),
        ma_read_length=ma_read_length,
        read=read,
    )
    projected, _mapped_fraction, _endpoints = _project_query_interval(
        cigar_to_query_ref(read), query_start, query_end, int(length), complete
    )
    return projected


def audit_bam(path: str, *, max_errors: int = 100) -> dict:
    candidate = Path(path).expanduser().resolve()
    if not candidate.is_file():
        raise ValueError(f"missing BAM: {candidate}")
    if max_errors < 1:
        raise ValueError("max_errors must be positive")
    errors = []
    error_count = 0

    def error(read_name: str, decision: str, message: str) -> None:
        nonlocal error_count
        error_count += 1
        if len(errors) < max_errors:
            errors.append(
                {"read": read_name, "decision": decision, "message": message}
            )

    counts: Counter = Counter()
    tf_component_histogram: Counter = Counter()
    threshold_states = {
        str(threshold): {
            "all": {"SR": 0, "baseline": 0},
            "R": {"TF": 0, "A": 0},
            "H": {"SR_edges": 0, "baseline_edges": 0},
        }
        for threshold in THRESHOLDS
    }
    try:
        pysam.quickcheck(str(candidate))
        quickcheck = True
    except pysam.utils.SamtoolsError:
        quickcheck = False
        error("", "", "samtools quickcheck failed")

    with pysam.AlignmentFile(candidate, "rb", check_sq=False) as bam:
        try:
            indexed = bool(bam.has_index() and bam.check_index())
        except (AttributeError, OSError, ValueError):
            indexed = False
        if not indexed:
            error("", "", "BAM index is missing or cannot be opened")
        contracts = [
            str(comment)
            for comment in bam.header.to_dict().get("CO", [])
            if str(comment).startswith(ALL_HEADER_PREFIXES)
        ]
        contract = contracts[-1] if contracts else None
        family_contracts = [
            str(comment)
            for comment in bam.header.to_dict().get("CO", [])
            if str(comment).startswith(FAMILY_HEADER_PREFIX)
        ]
        family_contract = family_contracts[-1] if family_contracts else None
        family_extension = family_contract is not None
        latent_contracts = [
            str(comment)
            for comment in bam.header.to_dict().get("CO", [])
            if str(comment).startswith(LATENT_HEADER_PREFIXES)
        ]
        latent_contract = latent_contracts[-1] if latent_contracts else None
        latent_extension = latent_contract is not None
        latent_contract_version = (
            3
            if latent_contract is not None
            and latent_contract.startswith(POSTFAMILY_LATENT_HEADER_PREFIX)
            else 2
            if latent_contract is not None
            and latent_contract.startswith(MULTIFAMILY_LATENT_HEADER_PREFIX)
            else 1
            if latent_contract is not None
            else None
        )
        latent_versions = [
            3 if value.startswith(POSTFAMILY_LATENT_HEADER_PREFIX)
            else 2 if value.startswith(MULTIFAMILY_LATENT_HEADER_PREFIX)
            else 1
            for value in latent_contracts
        ]
        if len(latent_versions) != len(set(latent_versions)):
            error("", "", "multiple latent-tiling contracts for one version")
        if latent_contract is not None:
            required_latent_tokens = (
                (
                    "layers=nuc_sr,tf_sr",
                    "semantics=complete_advisory_consensus_shadow",
                    "baseline_nuc_tf_unchanged=true",
                    "nuc_sr_structural_replacement=true",
                    "msp_baseline_not_refined=true",
                    "v6_nuc_identity_cardinality_fixed_overridden_for_named_latent_actions=true",
                    "family_geometry_crossfit=false",
                    "family_geometry_max_expansion_bp=6",
                    "nucleosome_length_prior_crossfit=false",
                    "molecule_efficiency_calibration_crossfit=false",
                    "validated_result_scope=CT_only_at_NAPA_family_021",
                    "GA_supported_actions=0",
                    "q0=map_configuration_posterior_within_family_group_uint8_endpoints_reserved",
                    "decision_confidence=weakest_split_identity_crossfit_component_in_actions_sidecar",
                    "q1=molecular_left_edge_confidence",
                    "q2=molecular_right_edge_confidence",
                    "edge_confidence=molecule_conditioned_within_family_configuration_marginal",
                    "map_geometry=maximum_posterior_configuration_within_family_group",
                    "map_geometry_share=recorded_in_actions_sidecar",
                    "named_actions=fhlt_token_optional_P_Rn_tf_and_fhlt_token_optional_P_Nn_H_nuc",
                    "propagated_name_marker=fhlt_token_P",
                    "amplification_family_policy=propagate_only_to_compatible_exact_baseline_state",
                    "incompatible_amplification_siblings=unchanged",
                    "ambiguous_molecules_unchanged=true",
                )
                if latent_contract_version == 1
                else (
                    "layers=nuc_sr,tf_sr",
                    "semantics=complete_advisory_multifamily_configuration",
                    "baseline_nuc_tf_unchanged=true",
                    "multiple_named_tf_segments=true",
                    "tf_roles=R0_through_Rn_zero_based_contiguous",
                    "nuc_roles=N0H_through_NnH_zero_based_contiguous",
                    "q1q2=segment_specific_edge_marginals",
                    "named_actions=fhlt_token_optional_P_Rn_tf_and_fhlt_token_optional_P_Nn_H_nuc",
                ) if latent_contract_version == 2
                else (
                    "layers=nuc_sr,tf_sr",
                    "semantics=advisory_post_family_configuration",
                    "baseline_nuc_tf_unchanged=true",
                    "source_layer=nuc_sr",
                    "exactly_one_covering_nuc_sr_replaced=true",
                    "single_named_tf_segment=true",
                    "single_residual_nuc_segment=true",
                    "q0=equal_prior_weakest_grid_robust_hypothesis_probability_uint8_endpoints_reserved",
                    "v6_q0_semantics_overridden_for_named_postfamily_actions=true",
                    "q1q2=segment_specific_edge_marginals",
                    "named_actions=fhlt_token_optional_P_Rn_tf_and_fhlt_token_optional_P_Nn_H_nuc",
                    "selection=proxy_operating_point_plus_grid_stability",
                    "family_geometry_crossfit=opposite_assayed_strand",
                    "nucleosome_length_prior_crossfit=false",
                    "molecule_efficiency_calibration_crossfit=false",
                    "nomination=permissive_exact_family_assignment",
                    "parent_bam_sha256=",
                    "actions_sha256=",
                    "materializer_grid=",
                    "ambiguous_multifamily_blocks=unchanged",
                )
            )
            for token in required_latent_tokens:
                if token not in latent_contract:
                    error("", "", f"latent-tiling header contract lacks {token}")
            if latent_contract_version == 3:
                for field in ("parent_bam_sha256", "actions_sha256"):
                    match = re.search(
                        rf"(?:^|;){field}=([0-9a-f]{{64}})(?:;|$)",
                        latent_contract,
                    )
                    if match is None:
                        error(
                            "", "",
                            f"latent-tiling {field} is not 64 lowercase hex characters",
                        )
            if latent_contract_version == 2:
                stage_specific_q0 = "shared_decision_q0=false" in latent_contract
                versioned_tokens = (
                    (
                        "shared_decision_q0=false",
                        "first_stage_geometry_and_q0_preserved=true",
                        "new_R1_and_replaced_N0H_q0=multifamily_map_configuration_posterior",
                        "equal_prior_hybrid_hypothesis_probability=actions_sidecar_only",
                    )
                    if stage_specific_q0
                    else (
                        "shared_decision_q0=true",
                        "first_stage_geometry_preserved=true",
                    )
                )
                for token in versioned_tokens:
                    if token not in latent_contract:
                        error("", "", f"latent-tiling header contract lacks {token}")
        if len(family_contracts) > 1:
            error("", "", "multiple TF-family header contracts")
        if family_contract is not None:
            for token in (
                "layer=tf_sr",
                "quality_spec=QQQQQ",
                "q3=fi_local_repeating_family_id_uint8",
                "fi_zero=unassigned",
                "fi_identity=reference_neighborhood_plus_fi",
                "fi_reuse_separation_bp=24",
                "q4=fq_conditional_family_assignment_confidence",
                "fq_scale=round_255_times_confidence",
                "assignments_sha256=",
            ):
                if token not in family_contract:
                    error("", "", f"TF-family header contract lacks {token}")
            calibration_tokens = (
                (
                    "fq_producer_calibration=declared_in_assignment_metadata",
                    (
                        "assignment_scope=complete_post_multifamily_tf_sr_fi_gt_0"
                        if latent_contract_version == 2
                        else (
                            "assignment_scope=complete_postfamily_tf_sr_fi_gt_0"
                            if latent_contract_version == 3
                            else "assignment_scope=complete_post_latent_tf_sr_fi_gt_0"
                        )
                    ),
                    "assignment_metadata_sha256=",
                )
                if latent_extension else
                ("fq_producer_calibration=declared_in_assignment_artifact",)
            )
            for token in calibration_tokens:
                if token not in family_contract:
                    error("", "", f"TF-family header contract lacks {token}")
            digest_match = re.search(r"assignments_sha256=([0-9a-f]{64})(?:;|$)", family_contract)
            if digest_match is None:
                error("", "", "TF-family assignments_sha256 is not 64 lowercase hex characters")
            if latent_extension:
                metadata_match = re.search(
                    r"assignment_metadata_sha256=([0-9a-f]{64})(?:;|$)",
                    family_contract,
                )
                if metadata_match is None:
                    error("", "", "TF-family assignment_metadata_sha256 is not 64 lowercase hex characters")
        if contract is None:
            error("", "", "missing strand-rescue header contract")
            contract_version = None
        else:
            if len(contracts) != 1:
                error("", "", "multiple strand-rescue header contracts")
            contract_version = next(
                (
                    version
                    for version, prefix in CONTRACT_PREFIXES.items()
                    if contract.startswith(prefix)
                ),
                None,
            )
            if family_extension and contract_version not in {4, 6}:
                error("", "", "TF-family extension requires a v4/v6 strand-rescue contract")
            if latent_extension and contract_version != 6:
                error("", "", "latent-tiling extension requires a v6 strand-rescue contract")
            if latent_extension and not family_extension:
                error("", "", "latent-tiling extension requires a TF-family contract")
            if contract_version in {4, 6}:
                q0_token = (
                    "q0=exact_selected_configuration_probability_if_action_"
                    "set_complete_else_zero"
                    if contract_version == 6
                    else "q0=sr_alternative_probability_vs_ordinary_baseline"
                )
                tokens = (
                    "groups=nuc_sr,tf_sr",
                    "semantics=two_strand_state_and_edge_normalization",
                    "quality_spec=QQQ",
                    "q_scale=linear_unit_interval",
                    q0_token,
                    "q1=molecular_left_edge_confidence",
                    "q2=molecular_right_edge_confidence",
                    "display_sr_if=q0>=T",
                    "threshold_named_only=true",
                    "roles=Rn_tf_rescue,H_edge_normalized",
                    "r_q0_atomic=true",
                    "h_source_ordinal=true",
                    "nuc_sr_edge_only=true",
                    "nuc_identity_cardinality_fixed=true",
                    "layers_complementary=false",
                    "nuc_length_ceiling=none",
                    "unchanged_edge_q=255",
                    "baseline_row=255,0,0",
                    "baseline_q0_sentinel=true",
                )
                if contract_version == 6:
                    tokens += (
                        "h_q0=assignment_marginalized_canonical_geometry_probability",
                        "changed_edge_without_target_opportunity_q=0",
                    )
            elif contract_version == 3:
                tokens = (
                    "groups=nuc_sr,tf_sr",
                    "semantics=two_strand_state_and_edge_normalization",
                    "quality_spec=QQQQQ",
                    "q0=state_presence_probability",
                    "roles=Rn_tf_rescue,H_edge_normalized",
                    "h_source_ordinal=true",
                    "nuc_sr_edge_only=true",
                    "nuc_sr_q0=255",
                    "nuc_identity_cardinality_fixed=true",
                    "layers_complementary=false",
                    "nuc_length_ceiling=none",
                    "baseline_q0=255",
                )
            elif contract_version == 2:
                tokens = (
                    "groups=tf_sr",
                    "semantics=two_strand_tf_normalization",
                    "quality_spec=QQQQQ",
                    "q0=tf_presence_probability",
                    "roles=Rn_rescue,H_harmonized",
                    "nuc_occupancy_candidates=false",
                    "nuc_annotations_modified=false",
                    "baseline_q0=255",
                )
            else:
                tokens = ()
                error("", "", "unrecognized strand-rescue contract version")
            for token in tokens:
                if token not in contract:
                    error("", "", f"header contract lacks {token}")
        complete_shadow = contract_version in {3, 4, 6}
        expected_layers = set(LAYER_ORDER if complete_shadow else ("tf_sr",))
        missing_types = sorted(expected_layers - set(declared_ma_types(bam.header)))
        if missing_types:
            error("", "", "MA-TYPES lacks " + ",".join(missing_types))

        for read in bam.fetch(until_eof=True):
            counts["records"] += 1
            try:
                annotations = _annotation_rows(read)
                ma_read_length = (
                    int(parse_ma_tag(read.get_tag("MA"))["read_length"])
                    if read.has_tag("MA") else int(read.query_length or 0)
                )
            except (KeyError, TypeError, ValueError) as exc:
                error(read.query_name, "", str(exc))
                continue
            decisions: DefaultDict[str, List[dict]] = defaultdict(list)
            latent_decisions: DefaultDict[str, List[dict]] = defaultdict(list)
            sr_rows = []
            ordinary = {
                name: [row for row in annotations if row["type"] == name]
                for name in ("nuc", "tf", "msp")
            }
            source_intervals = {}
            for annotation in annotations:
                match = NAME_RE.fullmatch(annotation["annotation_name"])
                latent_match = LATENT_NAME_RE.fullmatch(annotation["annotation_name"])
                decision_name = (
                    match.group(1) if match is not None
                    else latent_match.group(1) if latent_match is not None
                    else ""
                )
                if annotation["type"] == "nuc_sr" and not complete_shadow:
                    error(read.query_name, "", "v2 output contains a nuc_sr annotation")
                    continue
                if match is not None and annotation["type"] not in LAYER_ORDER:
                    error(
                        read.query_name,
                        match.group(1),
                        f"paired AN label occurs on {annotation['type']}",
                    )
                    continue
                if latent_match is not None and annotation["type"] not in LAYER_ORDER:
                    error(
                        read.query_name,
                        latent_match.group(1),
                        f"latent AN label occurs on {annotation['type']}",
                    )
                    continue
                if annotation["type"] not in LAYER_ORDER:
                    continue
                counts["sr_annotations"] += 1
                counts[f"annotations_{annotation['type']}"] += 1
                sr_rows.append(annotation)
                expected_specification = (
                    FAMILY_QUALITY_SPEC
                    if family_extension and annotation["type"] == "tf_sr"
                    else QUALITY_SPECS.get(contract_version, QUALITY_SPEC)
                )
                if annotation["quality_spec"] != expected_specification:
                    error(
                        read.query_name,
                        decision_name,
                        f"{annotation['type']} does not use "
                        f"{expected_specification}",
                    )
                    continue
                if family_extension and annotation["type"] == "tf_sr":
                    family_id, family_q = annotation["qualities"][3:5]
                    if family_id == 0 and family_q != 0:
                        error(
                            read.query_name,
                            decision_name,
                            "unassigned fi=0 annotation has nonzero fq",
                        )
                    if family_id > 0:
                        if family_q == 0:
                            error(
                                read.query_name,
                                decision_name,
                                "assigned fi>0 annotation has zero fq",
                            )
                        counts["family_assigned_tf_sr"] += 1
                        counts[f"family_id_{family_id}"] += 1
                if (
                    contract_version == 3
                    and annotation["type"] == "nuc_sr"
                    and annotation["qualities"][0] != 255
                ):
                    error(read.query_name, "", "nuc_sr does not have fixed q0=255")
                if latent_match is not None:
                    if not latent_extension:
                        error(
                            read.query_name,
                            latent_match.group(1),
                            "latent AN label lacks latent-tiling header contract",
                        )
                    annotation["decision"] = latent_match.group(1)
                    if latent_match.group("tf_index") is not None:
                        annotation["role"] = f"R{latent_match.group('tf_index')}"
                        annotation["latent_kind"] = "tf"
                    else:
                        annotation["role"] = f"N{latent_match.group('nuc_index')}H"
                        annotation["latent_kind"] = "nuc"
                    annotation["origin"] = None
                    latent_decisions[latent_match.group(1)].append(annotation)
                    counts["latent_annotations"] += 1
                    continue
                if match is None:
                    counts["fixed_baseline_annotations"] += 1
                    baseline_row = (
                        (255, 0, 0)
                        if contract_version in {4, 6}
                        else (255, 0, 0, 0, 0)
                    )
                    if annotation["qualities"][:len(baseline_row)] != baseline_row:
                        error(
                            read.query_name,
                            "",
                            "unpaired SR annotation is not fixed baseline",
                        )
                    ordinary_type = annotation["type"].removesuffix("_sr")
                    if annotation["interval"] not in {
                        row["interval"] for row in ordinary[ordinary_type]
                    }:
                        error(
                            read.query_name,
                            "",
                            "unpaired SR annotation does not match ordinary call",
                        )
                    continue
                annotation["decision"] = match.group(1)
                annotation["role"] = match.group("role")
                annotation["origin"] = match.group("origin")
                decisions[match.group(1)].append(annotation)

            if decisions or latent_decisions:
                counts["records_with_decisions"] += 1
            if complete_shadow:
                nuc_shadow = [row for row in sr_rows if row["type"] == "nuc_sr"]
                tf_shadow = [row for row in sr_rows if row["type"] == "tf_sr"]
                latent_nuc_count = sum(
                    member.get("latent_kind") == "nuc"
                    for members in latent_decisions.values()
                    for member in members
                )
                expected_nuc_shadow = (
                    len(ordinary["nuc"])
                    - len(latent_decisions)
                    + latent_nuc_count
                )
                if len(nuc_shadow) != expected_nuc_shadow:
                    error(read.query_name, "", "nuc_sr cardinality differs from nuc")
                rescue_components = sum(
                    (
                        NAME_RE.fullmatch(row["annotation_name"]) is not None
                        and "_R" in row["annotation_name"]
                    )
                    or row.get("latent_kind") == "tf"
                    for row in tf_shadow
                )
                if len(tf_shadow) != len(ordinary["tf"]) + rescue_components:
                    error(
                        read.query_name,
                        "",
                        "tf_sr baseline cardinality differs from tf",
                    )
                represented_sources = {"nuc": set(), "tf": set()}
                postfamily_tiles = []
                for row in sr_rows:
                    if row.get("role") != "H":
                        continue
                    ordinary_type = row["type"].removesuffix("_sr")
                    origin = row.get("origin")
                    if origin is None:
                        error(
                            read.query_name,
                            row.get("decision", ""),
                            "H role lacks source annotation ordinal",
                        )
                        continue
                    source_index = int(origin)
                    if not 0 <= source_index < len(ordinary[ordinary_type]):
                        error(
                            read.query_name,
                            row.get("decision", ""),
                            "H source annotation ordinal is out of range",
                        )
                        continue
                    if source_index in represented_sources[ordinary_type]:
                        error(
                            read.query_name,
                            row.get("decision", ""),
                            "ordinary source annotation represented more than once",
                        )
                        continue
                    represented_sources[ordinary_type].add(source_index)
                    source_intervals[id(row)] = ordinary[ordinary_type][source_index][
                        "interval"
                    ]
                for decision_id, members in latent_decisions.items():
                    nuc_members = [
                        member for member in members
                        if member.get("latent_kind") == "nuc"
                    ]
                    tf_members = [
                        member for member in members
                        if member.get("latent_kind") == "tf"
                    ]
                    if not nuc_members or not tf_members:
                        error(
                            read.query_name,
                            decision_id,
                            "latent tiling must contain both TF and nucleosome segments",
                        )
                        continue
                    projected_members = [
                        (member, _reference_interval(read, member, ma_read_length))
                        for member in members
                    ]
                    if any(projected is None for _member, projected in projected_members):
                        error(
                            read.query_name,
                            decision_id,
                            "latent component is not reference-projectable",
                        )
                        continue
                    tiled = sorted(
                        projected_members,
                        key=lambda item: item[1],
                    )
                    tile_start = tiled[0][1][0]
                    tile_end = tile_start
                    valid_tiling = True
                    for _member, projected in tiled:
                        start, end = projected
                        if start < tile_end or end <= start:
                            valid_tiling = False
                            break
                        tile_end = end
                    if not valid_tiling:
                        error(
                            read.query_name,
                            decision_id,
                            "latent components are not one ordered nonoverlapping configuration",
                        )
                        continue
                    if latent_contract_version == 3:
                        # A v3 action replaces the conservative ``nuc_sr``
                        # state, not the original HMM ``nuc`` call.  Recover
                        # the corresponding ordinary source by elimination
                        # after all unchanged/H shadows have been matched.
                        # The parent/child materialization audit separately
                        # proves that exactly one input ``nuc_sr`` was removed.
                        postfamily_tiles.append(
                            (tile_start, tile_end, decision_id, nuc_members)
                        )
                        continue
                    candidates = [
                        index
                        for index, ordinary_row in enumerate(ordinary["nuc"])
                        if index not in represented_sources["nuc"]
                        and (
                            (projected := _reference_interval(
                                read, ordinary_row, ma_read_length
                            )) is not None
                            and abs(projected[0] - tile_start) <= 6
                            and abs(projected[1] - tile_end) <= 6
                        )
                    ]
                    if len(candidates) != 1:
                        error(
                            read.query_name,
                            decision_id,
                            "latent configuration does not replace exactly one nearby ordinary nucleosome",
                        )
                        continue
                    source_index = candidates[0]
                    represented_sources["nuc"].add(source_index)
                    source_interval = ordinary["nuc"][source_index]["interval"]
                    for member in nuc_members:
                        source_intervals[id(member)] = source_interval
                for row in sr_rows:
                    if row.get("role") is not None:
                        if row["role"].startswith("R") and row.get("origin") is not None:
                            error(
                                read.query_name,
                                row.get("decision", ""),
                                "rescue role unexpectedly has a source ordinal",
                            )
                        continue
                    ordinary_type = row["type"].removesuffix("_sr")
                    candidates = [
                        index
                        for index, ordinary_row in enumerate(ordinary[ordinary_type])
                        if index not in represented_sources[ordinary_type]
                        and ordinary_row["interval"] == row["interval"]
                    ]
                    if not candidates:
                        error(
                            read.query_name,
                            "",
                            "baseline shadow has no unused matching ordinary call",
                        )
                        continue
                    source_index = candidates[0]
                    represented_sources[ordinary_type].add(source_index)
                    source_intervals[id(row)] = ordinary[ordinary_type][source_index][
                        "interval"
                    ]
                if latent_contract_version == 3 and postfamily_tiles:
                    remaining_nuc_sources = [
                        index for index in range(len(ordinary["nuc"]))
                        if index not in represented_sources["nuc"]
                    ]
                    if len(remaining_nuc_sources) != len(postfamily_tiles):
                        error(
                            read.query_name,
                            "",
                            "post-family configurations do not replace one distinct nuc_sr source each",
                        )
                    else:
                        ordered_sources = sorted(
                            remaining_nuc_sources,
                            key=lambda index: (
                                _reference_interval(
                                    read, ordinary["nuc"][index], ma_read_length
                                ) or (10**18, 10**18)
                            ),
                        )
                        ordered_tiles = sorted(
                            postfamily_tiles,
                            key=lambda item: (item[0], item[1], item[2]),
                        )
                        for source_index, (
                            _tile_start, _tile_end, _decision_id, nuc_members
                        ) in zip(ordered_sources, ordered_tiles):
                            represented_sources["nuc"].add(source_index)
                            source_interval = ordinary["nuc"][source_index]["interval"]
                            for member in nuc_members:
                                source_intervals[id(member)] = source_interval
                for ordinary_type in ("nuc", "tf"):
                    if represented_sources[ordinary_type] != set(
                        range(len(ordinary[ordinary_type]))
                    ):
                        error(
                            read.query_name,
                            "",
                            f"{ordinary_type}_sr is not one-for-one with ordinary calls",
                        )
                for layer_name in LAYER_ORDER:
                    represented = [
                        row
                        for row in sr_rows
                        if row["type"] == layer_name and id(row) in source_intervals
                    ]
                    for left_index, left in enumerate(represented):
                        for right in represented[left_index + 1 :]:
                            left_old = source_intervals[id(left)]
                            right_old = source_intervals[id(right)]
                            if left_old[0] == right_old[0]:
                                continue
                            old_order = left_old[0] < right_old[0]
                            new_order = left["interval"][0] < right["interval"][0]
                            if old_order != new_order:
                                error(
                                    read.query_name,
                                    "",
                                    "SR shadow call order differs from ordinary calls",
                                )
                for left_index, left in enumerate(sr_rows):
                    left_start, left_length = left["interval"]
                    left_end = left_start + left_length
                    for right in sr_rows[left_index + 1 :]:
                        right_start, right_length = right["interval"]
                        right_end = right_start + right_length
                        if not (
                            left_start < right_end and right_start < left_end
                        ):
                            continue
                        left_source = source_intervals.get(id(left))
                        right_source = source_intervals.get(id(right))
                        grandfathered = False
                        if left_source is not None and right_source is not None:
                            left_old_start, left_old_length = left_source
                            right_old_start, right_old_length = right_source
                            grandfathered = (
                                left_old_start < right_old_start + right_old_length
                                and right_old_start < left_old_start + left_old_length
                            )
                        if not grandfathered:
                            error(
                                read.query_name,
                                "",
                                "new overlap in SR shadow layers",
                            )
            for decision_id, members in decisions.items():
                counts["named_groups"] += 1
                roles = [member["role"] for member in members]
                if len(set(roles)) != len(roles):
                    error(read.query_name, decision_id, "duplicate member role")
                    continue
                rows = {member["qualities"] for member in members}
                q0_values = {member["qualities"][0] for member in members}
                if contract_version in {4, 6} and len(q0_values) != 1:
                    error(
                        read.query_name,
                        decision_id,
                        "group components do not share one q0",
                    )
                    continue
                if contract_version not in {4, 6} and len(rows) != 1:
                    error(
                        read.query_name,
                        decision_id,
                        "legacy group components do not share one quality row",
                    )
                    continue
                q0 = next(iter(q0_values))
                if roles == ["H"]:
                    counts["geometry_harmonizations"] += 1
                    counts[f"edge_refinements_{members[0]['type']}"] += 1
                    row = members[0]["qualities"]
                    if contract_version not in {4, 6} and q0 != 255:
                        error(
                            read.query_name,
                            decision_id,
                            "edge-normalized ordinary call does not have q0=255",
                        )
                    if contract_version in {4, 6}:
                        source = source_intervals.get(id(members[0]))
                        if source is not None:
                            source_start, source_length = source
                            source_end = source_start + source_length
                            alternative_start, alternative_length = members[0][
                                "interval"
                            ]
                            alternative_end = alternative_start + alternative_length
                            left_changed = alternative_start != source_start
                            right_changed = alternative_end != source_end
                            if not left_changed and not right_changed:
                                error(
                                    read.query_name,
                                    decision_id,
                                    "H alternative does not change either edge",
                                )
                            if not left_changed and row[1] != 255:
                                error(
                                    read.query_name,
                                    decision_id,
                                    "unchanged molecular-left H edge is not q=255",
                                )
                            if not right_changed and row[2] != 255:
                                error(
                                    read.query_name,
                                    decision_id,
                                    "unchanged molecular-right H edge is not q=255",
                                )
                    for threshold in THRESHOLDS:
                        selected = q0 >= threshold
                        state = threshold_states[str(threshold)]
                        state["all"]["SR" if selected else "baseline"] += 1
                        state["H"][
                            "SR_edges" if selected else "baseline_edges"
                        ] += 1
                    continue
                if any(not role.startswith("R") for role in roles):
                    error(read.query_name, decision_id, "mixed H and R roles")
                    continue
                if any(member["type"] != "tf_sr" for member in members):
                    error(read.query_name, decision_id, "rescue member is not tf_sr")
                indices = sorted(int(role[1:]) for role in roles)
                if indices != list(range(len(indices))):
                    error(
                        read.query_name,
                        decision_id,
                        "rescue roles are not contiguous",
                    )
                    continue
                for left_index, left in enumerate(members):
                    left_start, left_length = left["interval"]
                    left_end = left_start + left_length
                    for right in members[left_index + 1 :]:
                        right_start, right_length = right["interval"]
                        right_end = right_start + right_length
                        if left_start < right_end and right_start < left_end:
                            error(
                                read.query_name,
                                decision_id,
                                "rescue components overlap each other",
                            )
                if not any(
                    all(
                        msp["interval"][0] <= member["interval"][0]
                        and member["interval"][0] + member["interval"][1]
                        <= msp["interval"][0] + msp["interval"][1]
                        for member in members
                    )
                    for msp in ordinary["msp"]
                ):
                    error(
                        read.query_name,
                        decision_id,
                        "rescue components are not contained by one ordinary MSP",
                    )
                overlap_found = False
                for member in members:
                    start, length = member["interval"]
                    end = start + length
                    for other in sr_rows:
                        if other is member or other.get("decision") == decision_id:
                            continue
                        other_start, other_length = other["interval"]
                        other_end = other_start + other_length
                        if start < other_end and other_start < end:
                            error(
                                read.query_name,
                                decision_id,
                                "rescue overlaps another tf_sr group",
                            )
                            counts["rescue_cross_group_overlaps"] += 1
                            overlap_found = True
                            break
                    if overlap_found:
                        break
                counts["msp_to_tf_rescues"] += 1
                tf_component_histogram[str(len(members))] += 1
                for threshold in THRESHOLDS:
                    selected = q0 >= threshold
                    state = threshold_states[str(threshold)]
                    state["all"]["SR" if selected else "baseline"] += 1
                    state["R"]["TF" if selected else "A"] += 1
            for decision_id, members in latent_decisions.items():
                counts["latent_tiling_groups"] += 1
                if any(
                    LATENT_NAME_RE.fullmatch(member["annotation_name"]).group("propagated")
                    for member in members
                ):
                    counts["latent_tiling_propagated_groups"] += 1
                q0_values = {member["qualities"][0] for member in members}
                stage_specific_q0 = (
                    latent_contract_version == 2
                    and latent_contract is not None
                    and "shared_decision_q0=false" in latent_contract
                )
                if stage_specific_q0:
                    first_stage = [
                        member for member in members
                        if member.get("latent_kind") == "tf" and member.get("role") == "R0"
                    ]
                    second_stage = [member for member in members if member not in first_stage]
                    if len(first_stage) != 1 or not second_stage:
                        error(
                            read.query_name,
                            decision_id,
                            "stage-specific multifamily action lacks one R0 and later-stage members",
                        )
                        continue
                    later_q0_values = {member["qualities"][0] for member in second_stage}
                    if len(later_q0_values) != 1:
                        error(
                            read.query_name,
                            decision_id,
                            "later-stage multifamily components do not share one q0",
                        )
                        continue
                    q0 = min(first_stage[0]["qualities"][0], next(iter(later_q0_values)))
                else:
                    if len(q0_values) != 1:
                        error(
                            read.query_name,
                            decision_id,
                            "latent components do not share one q0",
                        )
                        continue
                    q0 = next(iter(q0_values))
                if any(value in {0, 255} for value in q0_values):
                    error(
                        read.query_name,
                        decision_id,
                        "latent action q0 must reserve sentinel endpoints 0 and 255",
                    )
                tf_members = [
                    member for member in members
                    if member.get("latent_kind") == "tf"
                ]
                nuc_members = [
                    member for member in members
                    if member.get("latent_kind") == "nuc"
                ]
                if any(member["type"] != "tf_sr" for member in tf_members):
                    error(read.query_name, decision_id, "latent TF member is not tf_sr")
                if any(member["type"] != "nuc_sr" for member in nuc_members):
                    error(read.query_name, decision_id, "latent nuc member is not nuc_sr")
                tf_indices = sorted(int(member["role"][1:]) for member in tf_members)
                nuc_indices = sorted(int(member["role"][1:-1]) for member in nuc_members)
                if tf_indices != list(range(len(tf_indices))):
                    error(read.query_name, decision_id, "latent TF roles are not contiguous")
                if nuc_indices != list(range(len(nuc_indices))):
                    error(read.query_name, decision_id, "latent nuc roles are not contiguous")
                if latent_contract_version in {1, 3} and len(tf_members) != 1:
                    error(
                        read.query_name,
                        decision_id,
                        f"v{latent_contract_version} latent action must contain exactly one named-family TF",
                    )
                if latent_contract_version == 3 and len(nuc_members) != 1:
                    error(
                        read.query_name,
                        decision_id,
                        "v3 latent action must contain exactly one residual nucleosome",
                    )
                for threshold in THRESHOLDS:
                    selected = q0 >= threshold
                    state = threshold_states[str(threshold)]
                    state["all"]["SR" if selected else "baseline"] += 1
                    state["R"]["TF" if selected else "A"] += 1

    index_paths = [
        str(Path(str(candidate) + suffix))
        for suffix in (".bai", ".csi")
        if Path(str(candidate) + suffix).is_file()
    ]
    return {
        "path": str(candidate),
        "sha256": _sha256(candidate),
        "indexed": indexed,
        "quickcheck": quickcheck,
        "contract_version": contract_version,
        "family_contract": family_contract,
        "family_extension": family_extension,
        "latent_contract": latent_contract,
        "latent_extension": latent_extension,
        "index_paths": index_paths,
        "valid": error_count == 0,
        "error_count": error_count,
        "errors": errors,
        "counts": dict(sorted(counts.items())),
        "tf_component_histogram": dict(sorted(tf_component_histogram.items())),
        "threshold_states": threshold_states,
    }


def audit_bams(paths: Sequence[str], *, max_errors: int = 100) -> dict:
    files = [audit_bam(path, max_errors=max_errors) for path in paths]
    totals: Counter = Counter()
    threshold_states = {
        str(threshold): {
            "all": {"SR": 0, "baseline": 0},
            "R": {"TF": 0, "A": 0},
            "H": {"SR_edges": 0, "baseline_edges": 0},
        }
        for threshold in THRESHOLDS
    }
    for result in files:
        totals.update(result["counts"])
        for threshold, roles in result["threshold_states"].items():
            for role, states in roles.items():
                for state, value in states.items():
                    threshold_states[threshold][role][state] += int(value)
    return {
        "schema": "fiberhmm.strand_rescue.audit.v4",
        "valid": all(result["valid"] for result in files),
        "file_count": len(files),
        "totals": dict(sorted(totals.items())),
        "threshold_states": threshold_states,
        "files": files,
    }


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-i", "--bam", action="append", required=True)
    parser.add_argument("-o", "--output")
    parser.add_argument("--max-errors", type=int, default=100)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    from fiberhmm.cli.common import add_version_args
    add_version_args(parser)
    args = parser.parse_args(argv)
    from fiberhmm.cli.common import PathAliasError, check_path_aliases
    try:
        check_path_aliases(inputs={"--bam": args.bam},
                           outputs={"--output": args.output})
    except PathAliasError as exc:
        parser.error(str(exc))
    try:
        result = audit_bams(args.bam, max_errors=args.max_errors)
        if args.output:
            _atomic_json(Path(args.output).expanduser(), result)
    except (OSError, TypeError, ValueError, pysam.utils.SamtoolsError) as exc:
        parser.error(str(exc))
    print(
        json.dumps(
            {
                "valid": result["valid"],
                "file_count": result["file_count"],
                "totals": result["totals"],
                "output": (
                    str(Path(args.output).expanduser().resolve())
                    if args.output
                    else None
                ),
            },
            sort_keys=True,
        )
    )
    return 0 if result["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
