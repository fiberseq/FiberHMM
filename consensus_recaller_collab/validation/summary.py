"""Compact, assay-aware summaries for fine-TF validation panels."""
from __future__ import annotations

import math
from statistics import median
from typing import Dict, Mapping, Optional, Sequence, Tuple


def _record_bf(record: Mapping[str, object], sample: Mapping[str, object]) -> float:
    if bool(sample.get("negative_evidence_allowed", True)):
        return float(record.get("log_bayes_factor", 0.0))
    return float(record.get(
        "positive_log_bayes_factor",
        max(0.0, float(record.get("log_bayes_factor", 0.0))),
    ))


def _quantile(values: Sequence[float], probability: float) -> Optional[float]:
    """Deterministic linearly interpolated quantile without a NumPy scalar."""
    if not values:
        return None
    if not 0.0 <= probability <= 1.0:
        raise ValueError("quantile probability must lie in [0, 1]")
    ordered = sorted(float(value) for value in values)
    position = probability * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _empirical_upper_tail(value: float, null: Sequence[float]) -> Optional[float]:
    if not null:
        return None
    return (1.0 + sum(float(item) >= value for item in null)) / (len(null) + 1.0)


def summarize_fine_tf(
    evidence: Mapping[str, object],
    manifest: Mapping[str, object],
    truth: Optional[Mapping[str, object]] = None,
    *,
    support_log_bf: float = 5.0,
) -> dict:
    samples = {
        str(sample["sample_id"]): sample for sample in manifest.get("samples", [])
    }
    candidates = [
        candidate for candidate in evidence.get("candidates", [])
        if candidate.get("axis") == "fine_tf"
    ]
    records: Dict[Tuple[str, str], Mapping[str, object]] = {
        (str(record["candidate_id"]), str(record["sample_id"])): record
        for record in evidence.get("records", [])
        if record.get("axis") == "fine_tf"
    }
    controls_by_parent: Dict[str, list] = {}
    for control in evidence.get("controls", []):
        controls_by_parent.setdefault(
            str(control["parent_candidate_id"]), []
        ).append(control)
    control_metadata = {
        str(control["candidate_id"]): control
        for control in evidence.get("controls", [])
    }
    control_records: Dict[Tuple[str, str], Mapping[str, object]] = {
        (str(record["candidate_id"]), str(record["sample_id"])): record
        for record in evidence.get("control_records", [])
    }

    truth_by_candidate: Dict[str, Dict[str, Mapping[str, object]]] = {}
    if truth is not None:
        for result in truth.get("results", []):
            if result.get("axis") != "fine_tf":
                continue
            truth_by_candidate.setdefault(
                str(result["candidate_id"]), {}
            )[str(result.get("target_assay_family"))] = result

    rows = []
    for candidate in candidates:
        candidate_id = str(candidate["candidate_id"])
        sample_rows = {}
        for (record_candidate, sample_id), record in records.items():
            if record_candidate != candidate_id or sample_id not in samples:
                continue
            sample = samples[sample_id]
            real_bf = _record_bf(record, sample)
            informative = int(record.get("informative", 0))
            normalized_real_bf = real_bf / informative if informative else None
            matched_controls = []
            for control in controls_by_parent.get(candidate_id, []):
                control_id = str(control["candidate_id"])
                control_record = control_records.get(
                    (control_id, sample_id)
                )
                if control_record is not None:
                    control_bf = _record_bf(control_record, sample)
                    control_informative = int(control_record.get("informative", 0))
                    matched_controls.append({
                        "candidate_id": control_id,
                        "interval": [
                            int(control_metadata[control_id].get("start", 0)),
                            int(control_metadata[control_id].get("end", 0)),
                        ],
                        "offset": int(control_metadata[control_id].get("offset", 0)),
                        "opportunity_log_mismatch": float(
                            control_metadata[control_id].get(
                                "opportunity_log_mismatch", 0.0
                            )
                        ),
                        "informative": control_informative,
                        "site_opportunities": int(
                            control_record.get(
                                "site_opportunities", control_informative
                            )
                        ),
                        "log_bayes_factor": control_bf,
                        "normalized_log_bayes_factor": (
                            control_bf / control_informative
                            if control_informative else None
                        ),
                        "by_library": control_record.get("by_library", {}),
                    })
            matched_controls.sort(key=lambda row: row["candidate_id"])
            usable_controls = [
                row for row in matched_controls
                if row["normalized_log_bayes_factor"] is not None
                and int(row["site_opportunities"]) > 0
            ]
            control_bfs = [
                float(row["log_bayes_factor"]) for row in usable_controls
            ]
            normalized_control_bfs = [
                float(row["normalized_log_bayes_factor"])
                for row in usable_controls
            ]
            control_median = median(control_bfs) if control_bfs else None
            delta = real_bf - control_median if control_median is not None else None
            normalized_control_median = (
                median(normalized_control_bfs) if normalized_control_bfs else None
            )
            normalized_delta = (
                normalized_real_bf - normalized_control_median
                if normalized_real_bf is not None
                and normalized_control_median is not None else None
            )
            source_diagnostics = [
                diagnostic for diagnostic in candidate.get("source_diagnostics", [])
                if str(diagnostic.get("sample_id")) == sample_id
            ]
            library_rows = {}
            for library_id, library_record in sorted(
                record.get("by_library", {}).items()
            ):
                library_bf = _record_bf(library_record, sample)
                library_control_bfs = []
                for control in usable_controls:
                    control_library = control.get("by_library", {}).get(library_id)
                    if control_library is not None:
                        library_control_bfs.append(
                            _record_bf(control_library, sample)
                        )
                library_control_median = (
                    median(library_control_bfs) if library_control_bfs else None
                )
                library_delta = (
                    library_bf - library_control_median
                    if library_control_median is not None else None
                )
                library_rows[library_id] = {
                    "informative": int(library_record.get("informative", 0)),
                    "site_opportunities": int(
                        library_record.get("site_opportunities", 0)
                    ),
                    "explicit_tf_reads": int(
                        library_record.get("explicit_tf_reads", 0)
                    ),
                    "log_bayes_factor": library_bf,
                    "control_count": len(library_control_bfs),
                    "median_control_log_bayes_factor": library_control_median,
                    "local_delta_log_bayes_factor": library_delta,
                    "supported": library_bf >= support_log_bf,
                    "locally_enriched": (
                        library_delta > 0.0 if library_delta is not None else None
                    ),
                }
            sample_rows[sample_id] = {
                "assay_family": sample["assay_family"],
                "permission": sample.get("axis_permissions", {}).get("fine_tf"),
                "truth_vote": bool(sample.get("truth_vote", True)),
                "negative_evidence_allowed": bool(
                    sample.get("negative_evidence_allowed", True)
                ),
                "is_source_sample": sample_id in candidate.get("source_samples", []),
                "is_geometry_source": sample_id in candidate.get(
                    "geometry_source_samples", []
                ),
                "source_diagnostics": source_diagnostics,
                "libraries": library_rows,
                "library_count": len(library_rows),
                "library_supported_count": sum(
                    row["supported"] for row in library_rows.values()
                ),
                "library_locally_enriched_count": sum(
                    row["locally_enriched"] is True
                    for row in library_rows.values()
                ),
                "informative": informative,
                "log_bayes_factor": real_bf,
                "normalized_log_bayes_factor": normalized_real_bf,
                "supported": real_bf >= support_log_bf,
                "mean_tf_posterior": record.get("mean_posterior"),
                "explicit_tf_reads": record.get("explicit_tf_reads", 0),
                "site_opportunities": record.get("site_opportunities", 0),
                "control_count": len(control_bfs),
                "matched_control_count": len(matched_controls),
                "matched_controls": matched_controls,
                "median_control_log_bayes_factor": control_median,
                "maximum_control_log_bayes_factor": (
                    max(control_bfs) if control_bfs else None
                ),
                "control_q90_log_bayes_factor": _quantile(control_bfs, 0.90),
                "local_delta_log_bayes_factor": delta,
                "locally_enriched": delta > 0.0 if delta is not None else None,
                "beats_all_controls": (
                    real_bf > max(control_bfs) if control_bfs else None
                ),
                "local_rank": (
                    1 + sum(value >= real_bf for value in control_bfs)
                    if control_bfs else None
                ),
                "local_empirical_p": _empirical_upper_tail(real_bf, control_bfs),
                "median_control_normalized_log_bayes_factor": (
                    normalized_control_median
                ),
                "normalized_local_delta_log_bayes_factor": normalized_delta,
                "normalized_local_empirical_p": (
                    _empirical_upper_tail(
                        normalized_real_bf, normalized_control_bfs
                    )
                    if normalized_real_bf is not None else None
                ),
            }
        voting_rows = [
            row for sample_id, row in sample_rows.items()
            if bool(samples[sample_id].get("truth_vote", True))
            and samples[sample_id].get("axis_permissions", {}).get("fine_tf")
            not in ("none", "geometry_only")
        ]
        controlled = [
            row for row in voting_rows if row["locally_enriched"] is not None
        ]
        rows.append({
            "candidate_id": candidate_id,
            "cohort_id": candidate.get("cohort_id"),
            "locus_id": candidate.get("locus_id"),
            "interval": [candidate["start"], candidate["end"]],
            "geometry_tier": candidate.get("geometry_tier"),
            "source_samples": candidate.get("source_samples", []),
            "geometry_source_samples": candidate.get(
                "geometry_source_samples", []
            ),
            "source_support": candidate.get("source_support", 0),
            "source_diagnostics": candidate.get("source_diagnostics", []),
            "support_only_seed": int(candidate.get("geometry_tier", 0)) >= 99,
            "samples": sample_rows,
            "controlled_voting_samples": len(controlled),
            "locally_enriched_all_voting_samples": (
                bool(controlled)
                and len(controlled) == len(voting_rows)
                and all(row["locally_enriched"] for row in controlled)
            ),
            "locally_enriched_any_voting_sample": any(
                row["locally_enriched"] for row in controlled
            ),
            "leave_one_family_out": truth_by_candidate.get(candidate_id, {}),
        })

    sample_summaries = []
    for sample_id in sorted({key[1] for key in records}):
        if sample_id not in samples:
            continue
        sample_rows = [row["samples"][sample_id] for row in rows if sample_id in row["samples"]]
        controlled = [row for row in sample_rows if row["locally_enriched"] is not None]
        sample_summaries.append({
            "sample_id": sample_id,
            "assay_family": samples[sample_id]["assay_family"],
            "candidates": len(sample_rows),
            "supported": sum(row["supported"] for row in sample_rows),
            "controlled": len(controlled),
            "locally_enriched": sum(row["locally_enriched"] for row in controlled),
            "median_local_delta_log_bayes_factor": (
                median(row["local_delta_log_bayes_factor"] for row in controlled)
                if controlled else None
            ),
            "truth_vote": bool(samples[sample_id].get("truth_vote", True)),
        })

    truth_statuses: Dict[str, Dict[str, int]] = {}
    for target_results in truth_by_candidate.values():
        for target, result in target_results.items():
            status = str(result["status"])
            truth_statuses.setdefault(target, {})[status] = (
                truth_statuses.setdefault(target, {}).get(status, 0) + 1
            )

    return {
        "schema_version": 2,
        "source_evidence": {
            "producer": evidence.get("producer"),
            "parameters": evidence.get("parameters"),
            "input_files": evidence.get("input_files", []),
            "control_policy": evidence.get("control_policy"),
            "loci": evidence.get("loci", []),
        },
        "axis": "fine_tf",
        "support_log_bf": support_log_bf,
        "candidate_count": len(rows),
        "support_only_seed_count": sum(row["support_only_seed"] for row in rows),
        "controlled_candidate_count": sum(
            row["controlled_voting_samples"] > 0 for row in rows
        ),
        "locally_enriched_all_voting_samples": sum(
            row["locally_enriched_all_voting_samples"] for row in rows
        ),
        "locally_enriched_any_voting_sample": sum(
            row["locally_enriched_any_voting_sample"] for row in rows
        ),
        "sample_summaries": sample_summaries,
        "leave_one_family_out_statuses": truth_statuses,
        "support_only_seeds": [row for row in rows if row["support_only_seed"]],
        "candidates": rows,
    }
