"""Held-out empirical calibration and deterministic proposal tiers.

The source assay is selected for containing a focal call cluster, so its local
rank is diagnostic rather than an independent p-value.  Independent assays are
calibrated against pseudo-site deltas made from matched controls at *other*
loci of the same assay family.  This keeps depth and chemistry hierarchy
explicit and prevents a high-depth support assay from acquiring boundary
authority through sample size alone.
"""
from __future__ import annotations

import copy
import csv
import hashlib
import io
import json
import os
import tempfile
from pathlib import Path
from statistics import median
from typing import Dict, List, Mapping, MutableMapping, Optional, Sequence

from consensus_recaller_collab.validation import VALIDATION_VERSION


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "w") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _family_policy(
    policy: Mapping[str, object], family: str
) -> Mapping[str, object]:
    shared = dict(policy.get("evidence", {}))
    shared.update(policy.get("families", {}).get(family, {}))
    return shared


def _benjamini_hochberg(rows: Sequence[MutableMapping[str, object]]) -> None:
    """Attach monotone BH q-values in place to rows containing ``p``."""
    ordered = sorted(
        (row for row in rows if row.get("p") is not None),
        key=lambda row: (float(row["p"]), str(row["candidate_id"])),
    )
    count = len(ordered)
    running = 1.0
    for reverse_index, row in enumerate(reversed(ordered), start=1):
        rank = count - reverse_index + 1
        running = min(running, float(row["p"]) * count / rank)
        row["q"] = min(1.0, running)


def build_pseudo_site_nulls(
    summary: Mapping[str, object], *, minimum_controls: int
) -> List[dict]:
    """Make exchangeable null deltas by holding out each matched control.

    A control is compared with the median of its sibling controls.  Raw and
    per-informative-molecule deltas are retained, but calibration uses the
    normalized value so panels with different depths can share a chemistry-
    specific null.
    """
    raw_nulls: List[dict] = []
    for candidate in summary.get("candidates", []):
        for sample_id, sample_row in candidate.get("samples", {}).items():
            controls = [
                control for control in sample_row.get("matched_controls", [])
                if control.get("normalized_log_bayes_factor") is not None
                and int(control.get("site_opportunities", 0)) > 0
            ]
            if len(controls) < minimum_controls:
                continue
            for control in controls:
                siblings = [
                    other for other in controls
                    if other["candidate_id"] != control["candidate_id"]
                ]
                if not siblings:
                    continue
                raw_baseline = median(
                    float(other["log_bayes_factor"]) for other in siblings
                )
                normalized_baseline = median(
                    float(other["normalized_log_bayes_factor"])
                    for other in siblings
                )
                interval = control.get("interval")
                control_bin = None
                if (
                    isinstance(interval, list)
                    and len(interval) == 2
                    and int(interval[1]) > int(interval[0])
                ):
                    control_bin = int(round(
                        (int(interval[0]) + int(interval[1])) / 50.0
                    ))
                raw_nulls.append({
                    "cohort_id": candidate.get("cohort_id"),
                    "locus_id": candidate.get("locus_id"),
                    "parent_candidate_id": candidate["candidate_id"],
                    "control_candidate_id": control["candidate_id"],
                    "control_interval": interval,
                    "control_center_25bp_bin": control_bin,
                    "sample_id": sample_id,
                    "assay_family": sample_row["assay_family"],
                    "raw_delta": (
                        float(control["log_bayes_factor"]) - raw_baseline
                    ),
                    "normalized_delta": (
                        float(control["normalized_log_bayes_factor"])
                        - normalized_baseline
                    ),
                })
    # The same genomic decoy can be selected for several neighboring parents.
    # Aggregate those correlated reuses before empirical calibration instead
    # of pretending they are independent null observations.
    groups: Dict[tuple, List[dict]] = {}
    for row in raw_nulls:
        bin_key = row["control_center_25bp_bin"]
        if bin_key is None:
            bin_key = str(row["control_candidate_id"])
        key = (
            row["assay_family"], row["cohort_id"], row["locus_id"],
            row["sample_id"], bin_key,
        )
        groups.setdefault(key, []).append(row)
    nulls = []
    for key, rows in groups.items():
        representative = min(
            rows, key=lambda row: str(row["control_candidate_id"])
        )
        # Keep the representative's own delta.  Median-averaging the members of
        # a bin shrinks the null's variance while the observed statistic gets no
        # such averaging, which makes every downstream p-value anti-conservative.
        # Deduplication is about not counting one decoy several times, not about
        # smoothing distinct decoys together.
        nulls.append({
            **representative,
            "correlated_control_uses": len(rows),
            "distinct_parent_candidates": len({
                str(row["parent_candidate_id"]) for row in rows
            }),
        })
    _studentize_nulls(nulls)
    return sorted(nulls, key=lambda row: (
        str(row["assay_family"]), str(row["cohort_id"]),
        str(row["locus_id"]), str(row["parent_candidate_id"]),
        str(row["control_candidate_id"]), str(row["sample_id"]),
    ))


MINIMUM_SCALE_CONTROLS = 5
_MAD_TO_SIGMA = 1.4826


def _null_scale_key(row: Mapping[str, object]) -> tuple:
    return (
        str(row["assay_family"]), str(row.get("cohort_id")),
        str(row.get("locus_id")), str(row.get("sample_id")),
    )


def _robust_scale(values: Sequence[float]) -> Optional[float]:
    if len(values) < MINIMUM_SCALE_CONTROLS:
        return None
    centre = median(values)
    deviation = median(abs(value - centre) for value in values)
    return _MAD_TO_SIGMA * deviation if deviation > 0.0 else None


class NullScales:
    """Robust null spread per stratum, falling back to coarser strata.

    A stratum with too few decoys to estimate its own spread must not lose its
    p-value; it borrows the family's spread instead.  A stratum that cannot even
    do that is left unstudentized (scale 1.0), which reproduces the previous
    behaviour for that row rather than silently dropping it.
    """

    def __init__(self, nulls: Sequence[Mapping[str, object]]):
        by_stratum: Dict[tuple, List[float]] = {}
        by_family: Dict[str, List[float]] = {}
        for row in nulls:
            delta = float(row["normalized_delta"])
            by_stratum.setdefault(_null_scale_key(row), []).append(delta)
            by_family.setdefault(str(row["assay_family"]), []).append(delta)
        self._stratum = {
            key: scale for key, values in by_stratum.items()
            if (scale := _robust_scale(values)) is not None
        }
        self._family = {
            key: scale for key, values in by_family.items()
            if (scale := _robust_scale(values)) is not None
        }

    def get(self, key: tuple) -> float:
        if key in self._stratum:
            return self._stratum[key]
        return self._family.get(key[0], 1.0)

    def resolved(self, key: tuple) -> str:
        if key in self._stratum:
            return "stratum_mad"
        return "family_mad" if key[0] in self._family else "unstudentized"


def null_scales(nulls: Sequence[Mapping[str, object]]) -> NullScales:
    """Robust null spread for each (family, cohort, locus, sample) stratum."""
    return NullScales(nulls)


def _studentize_nulls(nulls: List[MutableMapping[str, object]]) -> None:
    """Put every locus's nulls on one scale before they are pooled.

    ``normalized_delta`` divides a log Bayes factor by the number of informative
    bases, but that is not depth-invariant: the BIC penalty per molecule and the
    sampling error of a per-molecule likelihood both scale with depth, so the
    null *width* differs about twofold across loci.  Pooling raw deltas makes
    the shared null too narrow for deep loci and too wide for shallow ones, and
    the resulting p-values are not super-uniform, so BH inherits no FDR
    guarantee.  Dividing each observation by its own stratum's robust spread
    restores exchangeability across loci.
    """
    scales = null_scales(nulls)
    for row in nulls:
        key = _null_scale_key(row)
        scale = scales.get(key)
        row["null_scale"] = scale
        row["null_scale_source"] = scales.resolved(key)
        row["studentized_delta"] = float(row["normalized_delta"]) / scale


def _heldout_pool(
    nulls: Sequence[Mapping[str, object]],
    *,
    family: str,
    cohort_id: object,
    locus_id: object,
    minimum_size: int,
) -> tuple[List[float], str]:
    def pooled(same_cohort: bool) -> List[float]:
        return [
            float(row["studentized_delta"]) for row in nulls
            if row["assay_family"] == family
            and row.get("locus_id") != locus_id
            and row.get("studentized_delta") is not None
            and (not same_cohort or row.get("cohort_id") == cohort_id)
        ]

    values = pooled(True)
    if len(values) >= minimum_size:
        return values, "cohort_family_leave_one_locus_out_studentized"
    values = pooled(False)
    if len(values) >= minimum_size:
        return values, "family_leave_one_locus_out_studentized"
    return [], "unavailable"


def _source_focality_pass(
    sample_row: Mapping[str, object], family_policy: Mapping[str, object]
) -> bool:
    diagnostics = sample_row.get("source_diagnostics", [])
    if not diagnostics:
        return False
    minimum_enrichment = float(family_policy.get("source_min_local_enrichment", 1.5))
    maximum_mad = float(family_policy.get("source_max_boundary_mad", 12.0))
    minimum_support = int(family_policy.get("source_min_support", 3))
    return any(
        float(item.get("local_enrichment", 0.0)) >= minimum_enrichment
        and max(
            float(item.get("start_mad", float("inf"))),
            float(item.get("end_mad", float("inf"))),
        ) <= maximum_mad
        and int(item.get("support", 0)) >= minimum_support
        for item in diagnostics
    )


def _set_local_status(
    row: MutableMapping[str, object],
    family_policy: Mapping[str, object],
) -> None:
    minimum_controls = int(family_policy.get("minimum_controls", 3))
    minimum_informative = int(family_policy.get("minimum_informative", 10))
    minimum_opportunities = int(family_policy.get("minimum_site_opportunities", 5))
    support_log_bf = float(family_policy.get("support_log_bf", 5.0))
    strong_q = float(family_policy.get("independent_strong_q", 0.05))
    review_q = float(family_policy.get("independent_review_q", 0.20))
    minimum_replicates = int(
        family_policy.get("minimum_replicate_local_enriched", 0)
    )

    row["source_focality_pass"] = _source_focality_pass(row, family_policy)
    library_count = int(row.get("library_count", 0))
    replicate_pass = (
        minimum_replicates <= 0
        or library_count <= 1
        or int(row.get("library_locally_enriched_count", 0))
        >= minimum_replicates
    )
    row["replicate_consistency_pass"] = replicate_pass
    if (
        int(row.get("informative", 0)) < minimum_informative
        or int(row.get("site_opportunities", 0)) < minimum_opportunities
    ):
        row["calibrated_status"] = "untestable_low_information"
        return
    if int(row.get("control_count", 0)) < minimum_controls:
        row["calibrated_status"] = "untestable_controls"
        return
    if bool(row.get("is_source_sample")):
        if (
            row["source_focality_pass"]
            and float(row["log_bayes_factor"]) >= support_log_bf
            and replicate_pass
        ):
            row["calibrated_status"] = "source_focal"
        elif not replicate_pass:
            row["calibrated_status"] = "replicate_inconsistent"
        elif float(row["log_bayes_factor"]) >= support_log_bf:
            row["calibrated_status"] = "source_review"
        else:
            row["calibrated_status"] = "source_weak"
        return
    if row.get("heldout_null_count", 0) == 0:
        if (
            float(row["log_bayes_factor"]) >= support_log_bf
            and bool(row.get("beats_all_controls"))
        ):
            row["calibrated_status"] = "uncalibrated_local_support"
        else:
            row["calibrated_status"] = "uncalibrated"
        return
    if float(row["log_bayes_factor"]) < support_log_bf:
        row["calibrated_status"] = "no_support"
        return
    if float(row.get("normalized_local_delta_log_bayes_factor", 0.0)) <= 0.0:
        row["calibrated_status"] = "nonfocal"
        return
    if not replicate_pass:
        row["calibrated_status"] = "replicate_inconsistent"
        return
    q_value = row.get("heldout_empirical_q")
    if bool(row.get("beats_all_controls")) and q_value is not None and float(q_value) <= strong_q:
        row["calibrated_status"] = "strong_local"
    elif q_value is not None and float(q_value) <= review_q:
        row["calibrated_status"] = "local_support"
    else:
        row["calibrated_status"] = "ambiguous_local"


def _candidate_tier(candidate: MutableMapping[str, object]) -> tuple[str, str]:
    rows = list(candidate.get("samples", {}).values())
    voting = [row for row in rows if bool(row.get("truth_vote", True))]
    independent = [row for row in voting if not bool(row.get("is_source_sample"))]
    independent_anchors = [
        row for row in independent if row.get("permission") == "anchor"
    ]
    independent_strong = [
        row for row in independent
        if row.get("calibrated_status") == "strong_local"
    ]
    independent_support = [
        row for row in independent
        if row.get("calibrated_status") in {
            "strong_local", "local_support", "uncalibrated_local_support"
        }
    ]
    anchor_evidence = [
        row for row in voting
        if row.get("permission") == "anchor"
        and row.get("calibrated_status") in {
            "source_focal", "strong_local", "local_support",
            "uncalibrated_local_support",
        }
    ]
    source_focal = [
        row for row in voting
        if bool(row.get("is_source_sample"))
        and row.get("calibrated_status") == "source_focal"
    ]
    source_focal_families = {
        str(row["assay_family"]) for row in source_focal
    }
    source_anchor_families = {
        str(row["assay_family"]) for row in source_focal
        if row.get("permission") == "anchor"
    }
    support_only_seed = bool(candidate.get("support_only_seed"))
    if support_only_seed:
        if any(row.get("calibrated_status") == "strong_local" for row in independent_anchors):
            return "strong", "support seed independently confirmed by an authoritative anchor"
        if any(row.get("calibrated_status") in {
            "local_support", "uncalibrated_local_support"
        } for row in independent_anchors):
            return "review", "support seed has focal anchor support below the strong calibration tier"
        powered_anchor_rows = [
            row for row in independent_anchors
            if not str(row.get("calibrated_status", "")).startswith("untestable")
        ]
        if any(
            row.get("calibrated_status") == "nonfocal"
            for row in powered_anchor_rows
        ):
            return "review", "support seed has anchor signal, but it is not focal against multiple controls"
        if powered_anchor_rows and all(
            row.get("calibrated_status") == "no_support"
            for row in powered_anchor_rows
        ):
            return "reject", "support seed lacks focal confirmation in an informative anchor"
        return "untestable", "support seed has no independently calibrated anchor"

    if len(source_focal_families) >= 2 and source_anchor_families:
        return (
            "strong",
            "independently nominated by multiple assay families with authoritative anchor geometry",
        )
    if anchor_evidence and independent_strong:
        return "strong", "authoritative geometry has independently calibrated focal support"
    if anchor_evidence and independent_support:
        return "review", "authoritative geometry has independent support below the strong tier"
    if anchor_evidence:
        return "review", "authoritative single-assay site; cross-assay evidence is absent or unresolved"
    if independent_strong:
        return "review", "strong independent signal without retained authoritative source evidence"
    tested = [
        row for row in independent
        if not str(row.get("calibrated_status", "")).startswith("untestable")
    ]
    if any(row.get("calibrated_status") == "nonfocal" for row in tested):
        return "review", "independent signal is present but not focal against multiple controls"
    if tested and all(
        row.get("calibrated_status") == "no_support"
        for row in tested
    ):
        return "reject", "no focal evidence in informative independent assays"
    return "untestable", "insufficient independently calibrated evidence"


def calibrate_fine_tf(
    summary: Mapping[str, object],
    manifest: Mapping[str, object],
    policy: Mapping[str, object],
) -> dict:
    """Calibrate independent local evidence and assign conservative tiers."""
    result = copy.deepcopy(summary)
    minimum_controls = int(policy.get("evidence", {}).get("minimum_controls", 3))
    minimum_null = int(policy.get("calibration", {}).get("minimum_heldout_null", 50))
    nulls = build_pseudo_site_nulls(result, minimum_controls=minimum_controls)
    scales = null_scales(nulls)

    q_rows_by_family: Dict[str, List[MutableMapping[str, object]]] = {}
    for candidate in result.get("candidates", []):
        cohort_id = candidate.get("cohort_id")
        locus_id = candidate.get("locus_id")
        for sample_id, row in candidate.get("samples", {}).items():
            family = str(row["assay_family"])
            pool, scope = _heldout_pool(
                nulls,
                family=family,
                cohort_id=cohort_id,
                locus_id=locus_id,
                minimum_size=minimum_null,
            )
            observed = row.get("normalized_local_delta_log_bayes_factor")
            # The candidate must enter the pooled null on the same scale the
            # pool was put on, which is its own locus/sample null spread.  Its
            # controls supply only that scale; they are never null draws for it.
            scale_key = (family, str(cohort_id), str(locus_id), str(sample_id))
            scale = scales.get(scale_key)
            studentized = (
                float(observed) / scale if observed is not None else None
            )
            p_value = None
            if studentized is not None and pool:
                p_value = (
                    1.0 + sum(value >= studentized for value in pool)
                ) / (len(pool) + 1.0)
            row.update({
                "heldout_null_scope": scope,
                "heldout_null_count": len(pool),
                "local_null_scale": scale,
                "local_null_scale_source": scales.resolved(scale_key),
                "studentized_local_delta": studentized,
                "heldout_empirical_p": p_value,
                "heldout_empirical_q": None,
                "formal_independent_test": not bool(row.get("is_source_sample")),
            })
            if p_value is not None and not bool(row.get("is_source_sample")):
                q_rows_by_family.setdefault(family, []).append({
                    "candidate_id": candidate["candidate_id"],
                    "sample_id": sample_id,
                    "p": p_value,
                    "row": row,
                })

    for family_rows in q_rows_by_family.values():
        _benjamini_hochberg(family_rows)
        for item in family_rows:
            item["row"]["heldout_empirical_q"] = item.get("q")

    tier_counts: Dict[str, int] = {}
    status_counts: Dict[str, int] = {}
    for candidate in result.get("candidates", []):
        for sample_id, row in candidate.get("samples", {}).items():
            family_policy = _family_policy(policy, str(row["assay_family"]))
            _set_local_status(row, family_policy)
            status = str(row["calibrated_status"])
            status_counts[status] = status_counts.get(status, 0) + 1
        candidate["source_focal_families"] = sorted({
            str(row["assay_family"])
            for row in candidate.get("samples", {}).values()
            if bool(row.get("is_source_sample"))
            and row.get("calibrated_status") == "source_focal"
        })
        tier, reason = _candidate_tier(candidate)
        candidate["proposal_tier"] = tier
        candidate["proposal_reason"] = reason
        tier_counts[tier] = tier_counts.get(tier, 0) + 1

    calibrated_sample_summaries = []
    sample_ids = sorted({
        sample_id
        for candidate in result.get("candidates", [])
        for sample_id in candidate.get("samples", {})
    })
    for sample_id in sample_ids:
        sample_rows = [
            candidate["samples"][sample_id]
            for candidate in result.get("candidates", [])
            if sample_id in candidate.get("samples", {})
        ]
        counts: Dict[str, int] = {}
        for row in sample_rows:
            status = str(row["calibrated_status"])
            counts[status] = counts.get(status, 0) + 1
        calibrated_sample_summaries.append({
            "sample_id": sample_id,
            "assay_family": sample_rows[0]["assay_family"],
            "candidates": len(sample_rows),
            "source_selected": sum(
                bool(row.get("is_source_sample")) for row in sample_rows
            ),
            "formal_independent_tests": sum(
                bool(row.get("formal_independent_test"))
                and row.get("heldout_empirical_p") is not None
                for row in sample_rows
            ),
            "formal_q_le_0_05": sum(
                row.get("heldout_empirical_q") is not None
                and float(row["heldout_empirical_q"]) <= 0.05
                for row in sample_rows
            ),
            "status_counts": counts,
        })
    locus_summaries = []
    for locus_id in sorted({
        str(candidate.get("locus_id"))
        for candidate in result.get("candidates", [])
    }):
        locus_rows = [
            candidate for candidate in result.get("candidates", [])
            if str(candidate.get("locus_id")) == locus_id
        ]
        locus_summaries.append({
            "locus_id": locus_id,
            "candidates": len(locus_rows),
            "proposal_tier_counts": {
                tier: sum(row["proposal_tier"] == tier for row in locus_rows)
                for tier in ("strong", "review", "reject", "untestable")
            },
        })

    return {
        "schema_version": 1,
        "valid": True,
        "errors": [],
        "producer": {
            "name": "fiberhmm-consensus-calibration",
            "version": VALIDATION_VERSION,
            "manifest_sha256": _canonical_sha256(manifest),
            "policy_sha256": _canonical_sha256(policy),
        },
        "axis": "fine_tf",
        "calibration_method": (
            "matched-control pseudo-sites; chemistry/cohort leave-one-locus-out; "
            "BH correction only for assays independent of candidate discovery"
        ),
        "policy": copy.deepcopy(policy),
        "pseudo_site_null_count": len(nulls),
        "pseudo_site_null_counts_by_family": {
            family: sum(row["assay_family"] == family for row in nulls)
            for family in sorted({str(row["assay_family"]) for row in nulls})
        },
        "candidate_count": len(result.get("candidates", [])),
        "proposal_tier_counts": tier_counts,
        "sample_status_counts": status_counts,
        "sample_summaries": calibrated_sample_summaries,
        "locus_summaries": locus_summaries,
        "source_reports": result.get("source_reports", []),
        "candidates": result.get("candidates", []),
    }


TSV_COLUMNS = (
    "candidate_id", "cohort_id", "locus_id", "start", "end",
    "geometry_tier", "support_only_seed", "proposal_tier", "proposal_reason",
    "sample_id", "assay_family", "permission", "truth_vote",
    "is_source_sample", "formal_independent_test", "calibrated_status",
    "informative", "site_opportunities", "explicit_tf_reads",
    "log_bayes_factor", "normalized_local_delta_log_bayes_factor",
    "control_count", "beats_all_controls", "heldout_null_scope",
    "heldout_null_count", "heldout_empirical_p", "heldout_empirical_q",
)


def write_proposal_tsv(report: Mapping[str, object], path: str) -> None:
    """Write stable one-candidate/one-sample rows for review and plotting."""
    output = Path(path)
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=TSV_COLUMNS, delimiter="\t")
    writer.writeheader()
    for candidate in sorted(
        report.get("candidates", []), key=lambda row: str(row["candidate_id"])
    ):
        start, end = candidate["interval"]
        for sample_id, sample in sorted(candidate.get("samples", {}).items()):
            row = {
                "candidate_id": candidate["candidate_id"],
                "cohort_id": candidate.get("cohort_id"),
                "locus_id": candidate.get("locus_id"),
                "start": start,
                "end": end,
                "geometry_tier": candidate.get("geometry_tier"),
                "support_only_seed": candidate.get("support_only_seed"),
                "proposal_tier": candidate.get("proposal_tier"),
                "proposal_reason": candidate.get("proposal_reason"),
                "sample_id": sample_id,
            }
            for key in TSV_COLUMNS:
                if key not in row:
                    row[key] = sample.get(key)
            writer.writerow(row)
    _atomic_write_text(output, buffer.getvalue())


def merge_summaries(summaries: Sequence[Mapping[str, object]]) -> dict:
    """Merge per-locus summaries, rejecting duplicate candidate identifiers."""
    if not summaries:
        raise ValueError("at least one summary is required")
    candidates = []
    seen = set()
    for summary in summaries:
        for candidate in summary.get("candidates", []):
            candidate_id = str(candidate["candidate_id"])
            if candidate_id in seen:
                raise ValueError(f"duplicate candidate across summaries: {candidate_id}")
            seen.add(candidate_id)
            candidates.append(copy.deepcopy(candidate))
    return {
        "schema_version": max(int(summary.get("schema_version", 1)) for summary in summaries),
        "axis": "fine_tf",
        "candidate_count": len(candidates),
        "sample_summaries": [],
        "source_reports": [
            copy.deepcopy(summary.get("source_evidence", {}))
            for summary in summaries
        ],
        "candidates": sorted(candidates, key=lambda row: str(row["candidate_id"])),
    }
