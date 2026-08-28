"""State-specific, leave-one-assay-out truth adjudication.

The model intentionally separates broad nucleosome structure, fine TF
geometry, and occupancy frequency. Assay families can be anchors for one axis
and merely supporting observations for another. A supporting assay cannot
create definitive truth by itself, and samples with positive-only
ascertainment cannot contribute negative evidence.
"""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


AXES = ("broad_nucleosome", "fine_tf", "occupancy_frequency")
PERMISSIONS = ("anchor", "support", "geometry_only", "none")


def load_json(path: str) -> dict:
    return json.loads(Path(path).read_text())


def _log_beta_binomial(k: float, n: float, alpha: float, beta: float) -> float:
    """Beta-binomial log mass, generalized to fractional effective counts."""
    if not 0.0 <= k <= n:
        raise ValueError("positive evidence must lie between zero and informative depth")
    if alpha <= 0.0 or beta <= 0.0:
        raise ValueError("beta-binomial parameters must be positive")
    return (
        math.lgamma(n + 1.0)
        - math.lgamma(k + 1.0)
        - math.lgamma(n - k + 1.0)
        + math.lgamma(k + alpha)
        + math.lgamma(n - k + beta)
        - math.lgamma(n + alpha + beta)
        + math.lgamma(alpha + beta)
        - math.lgamma(alpha)
        - math.lgamma(beta)
    )


def _effective_counts(positive: float, informative: float, cap: float) -> Tuple[float, float]:
    if positive < 0.0 or informative < 0.0 or positive > informative:
        raise ValueError("invalid positive/informative evidence counts")
    if informative == 0.0:
        return 0.0, 0.0
    effective_n = min(informative, cap)
    return effective_n * positive / informative, effective_n


def _permission(sample: Mapping[str, object], axis: str, default_role: str) -> str:
    permissions = sample.get("axis_permissions", {})
    if not isinstance(permissions, Mapping):
        raise ValueError("sample axis_permissions must be an object")
    value = str(permissions.get(axis, default_role))
    if value not in PERMISSIONS:
        raise ValueError(f"unknown {axis} permission: {value}")
    return value


def _posterior_from_log_odds(log_odds: float) -> float:
    if log_odds >= 0.0:
        return 1.0 / (1.0 + math.exp(-min(log_odds, 745.0)))
    exp_value = math.exp(max(log_odds, -745.0))
    return exp_value / (1.0 + exp_value)


def _weighted_median(values: Sequence[int], weights: Sequence[float]) -> int:
    if not values or len(values) != len(weights):
        raise ValueError("weighted median requires paired values and weights")
    ordered = sorted(zip(values, weights), key=lambda item: item[0])
    threshold = 0.5 * sum(weight for _, weight in ordered)
    cumulative = 0.0
    for value, weight in ordered:
        cumulative += weight
        if cumulative >= threshold:
            return int(value)
    return int(ordered[-1][0])


def validate_hierarchy(config: Mapping[str, object]) -> List[str]:
    errors: List[str] = []
    axes = config.get("axes")
    if not isinstance(axes, Mapping):
        return ["hierarchy.axes must be an object"]
    for axis in AXES:
        axis_config = axes.get(axis)
        if not isinstance(axis_config, Mapping):
            errors.append(f"missing hierarchy axis: {axis}")
            continue
        prior = axis_config.get("prior_present")
        if not isinstance(prior, (int, float)) or not 0.0 < float(prior) < 1.0:
            errors.append(f"{axis}.prior_present must lie in (0, 1)")
        families = axis_config.get("families")
        if not isinstance(families, Mapping) or not families:
            errors.append(f"{axis}.families must be a non-empty object")
            continue
        for family, family_config in families.items():
            if not isinstance(family_config, Mapping):
                errors.append(f"{axis}.{family} must be an object")
                continue
            if family_config.get("role") not in ("anchor", "support"):
                errors.append(f"{axis}.{family}.role must be anchor or support")
            for key in ("present_rate_beta", "absent_rate_beta"):
                beta = family_config.get(key)
                if (
                    not isinstance(beta, list)
                    or len(beta) != 2
                    or any(not isinstance(value, (int, float)) or value <= 0 for value in beta)
                ):
                    errors.append(f"{axis}.{family}.{key} must contain two positive values")
            cap = family_config.get("effective_depth_cap")
            if not isinstance(cap, (int, float)) or cap <= 0:
                errors.append(f"{axis}.{family}.effective_depth_cap must be positive")
    return errors


def _sample_log_bayes_factor(
    record: Mapping[str, object],
    sample: Mapping[str, object],
    family_config: Mapping[str, object],
    permission: str,
) -> Tuple[float, dict]:
    positive = float(record.get("positive", 0.0))
    informative = float(record.get("informative", 0.0))
    direct_log_bf = record.get("log_bayes_factor")
    if direct_log_bf is not None:
        detail = {
            "positive": positive,
            "informative": informative,
            "effective_positive": positive,
            "effective_informative": informative,
            "permission": permission,
        }
        if permission in ("none", "geometry_only") or informative == 0.0:
            detail["log_bayes_factor"] = 0.0
            return 0.0, detail
        negative_allowed = bool(sample.get("negative_evidence_allowed", True))
        if negative_allowed:
            raw_log_bf = float(direct_log_bf)
        else:
            raw_log_bf = float(record.get(
                "positive_log_bayes_factor", max(0.0, float(direct_log_bf))
            ))
        scale = float(family_config.get("log_bf_scale", 1.0))
        cap = float(family_config.get("max_abs_log_bf", 30.0))
        calibrated = max(-cap, min(cap, scale * raw_log_bf))
        detail.update({
            "negative_evidence_allowed": negative_allowed,
            "raw_log_bayes_factor": raw_log_bf,
            "log_bf_scale": scale,
            "log_bf_cap": cap,
            "log_bayes_factor": calibrated,
            "evidence_model": record.get("model", {}).get(
                "evidence_statistic", "direct log Bayes factor"
            ),
        })
        return calibrated, detail
    effective_k, effective_n = _effective_counts(
        positive,
        informative,
        float(family_config["effective_depth_cap"]),
    )
    detail = {
        "positive": positive,
        "informative": informative,
        "effective_positive": effective_k,
        "effective_informative": effective_n,
        "permission": permission,
    }
    if permission in ("none", "geometry_only") or effective_n == 0.0:
        detail["log_bayes_factor"] = 0.0
        return 0.0, detail

    present_alpha, present_beta = [float(value) for value in family_config["present_rate_beta"]]
    absent_alpha, absent_beta = [float(value) for value in family_config["absent_rate_beta"]]
    negative_allowed = bool(sample.get("negative_evidence_allowed", True))
    full_log_bf = _log_beta_binomial(
        effective_k, effective_n, present_alpha, present_beta
    ) - _log_beta_binomial(
        effective_k, effective_n, absent_alpha, absent_beta
    )
    # Targeted/amplified panels may be allowed to confirm presence without
    # defining absence.  They still have an assay-specific null distribution:
    # ordinary background posterior mass is not affirmative evidence.  Clamp
    # the calibrated likelihood ratio at zero instead of counting every soft
    # positive as an independent hit.
    log_bf = full_log_bf if negative_allowed else max(0.0, full_log_bf)
    detail["unclamped_log_bayes_factor"] = full_log_bf
    detail["negative_evidence_allowed"] = negative_allowed
    detail["log_bayes_factor"] = log_bf
    return log_bf, detail


def _consensus_geometry(
    records: Sequence[Mapping[str, object]],
    samples: Mapping[str, Mapping[str, object]],
    family_configs: Mapping[str, Mapping[str, object]],
    axis: str,
) -> Optional[dict]:
    candidates = []
    for record in records:
        interval = record.get("geometry_interval")
        if (
            not isinstance(interval, list)
            or len(interval) != 2
            or float(record.get("positive", 0.0)) <= 0.0
        ):
            continue
        sample = samples[str(record["sample_id"])]
        family = str(sample["assay_family"])
        config = family_configs.get(family)
        if config is None or config.get("geometry_tier") is None:
            continue
        permission = _permission(sample, axis, str(config["role"]))
        if permission not in ("anchor", "support", "geometry_only"):
            continue
        candidates.append((
            int(config["geometry_tier"]),
            int(interval[0]),
            int(interval[1]),
            max(1.0, float(record.get("positive", 0.0))),
            str(record["sample_id"]),
        ))
    if not candidates:
        return None
    best_tier = min(candidate[0] for candidate in candidates)
    selected = [candidate for candidate in candidates if candidate[0] == best_tier]
    start = _weighted_median(
        [candidate[1] for candidate in selected],
        [candidate[3] for candidate in selected],
    )
    end = _weighted_median(
        [candidate[2] for candidate in selected],
        [candidate[3] for candidate in selected],
    )
    return {
        "interval": [start, end],
        "geometry_tier": best_tier,
        "source_samples": sorted({candidate[4] for candidate in selected}),
    }


def adjudicate_candidate(
    records: Sequence[Mapping[str, object]],
    manifest: Mapping[str, object],
    hierarchy: Mapping[str, object],
    *,
    target_assay_family: Optional[str] = None,
) -> dict:
    """Infer one candidate's truth while optionally holding out an assay family."""
    if not records:
        raise ValueError("candidate has no evidence records")
    axes = {str(record["axis"]) for record in records}
    if len(axes) != 1:
        raise ValueError("candidate records must share one axis")
    axis = axes.pop()
    if axis not in AXES:
        raise ValueError(f"unknown validation axis: {axis}")

    sample_list = manifest.get("samples", [])
    if not isinstance(sample_list, list):
        raise ValueError("manifest.samples must be a list")
    samples = {str(sample["sample_id"]): sample for sample in sample_list}
    axis_config = hierarchy["axes"][axis]
    family_configs = axis_config["families"]
    prior = float(axis_config["prior_present"])
    log_odds = math.log(prior / (1.0 - prior))
    anchors = set()
    negative_anchors = set()
    support_families = set()
    included_records = []
    contributors = []

    for record in records:
        sample_id = str(record["sample_id"])
        if sample_id not in samples:
            raise ValueError(f"evidence references unknown sample: {sample_id}")
        sample = samples[sample_id]
        family = str(sample["assay_family"])
        if target_assay_family is not None and family == target_assay_family:
            continue
        if not bool(sample.get("truth_vote", True)):
            continue
        if str(sample.get("availability", "available")) != "available":
            continue
        family_config = family_configs.get(family)
        if family_config is None:
            continue
        permission = _permission(sample, axis, str(family_config["role"]))
        log_bf, detail = _sample_log_bayes_factor(
            record, sample, family_config, permission
        )
        log_odds += log_bf
        included_records.append(record)
        if permission == "anchor" and float(record.get("informative", 0.0)) > 0.0:
            anchors.add(family)
            if bool(sample.get("negative_evidence_allowed", True)):
                negative_anchors.add(family)
        elif permission == "support" and float(record.get("informative", 0.0)) > 0.0:
            support_families.add(family)
        contributors.append({
            "sample_id": sample_id,
            "assay_family": family,
            **detail,
        })

    posterior = _posterior_from_log_odds(log_odds)
    thresholds = hierarchy["thresholds"]
    if anchors and posterior >= float(thresholds["anchored_present"]):
        status = "anchored_present"
    elif anchors and negative_anchors and posterior <= float(thresholds["anchored_absent"]):
        status = "anchored_absent"
    elif (
        not anchors
        and len(support_families) >= int(thresholds["minimum_provisional_families"])
        and posterior >= float(thresholds["provisional_present"])
    ):
        status = "provisional_present"
    elif anchors:
        status = "unresolved"
    else:
        status = "unresolved_no_anchor"

    first = records[0]
    geometry = None
    if status in ("anchored_present", "provisional_present"):
        geometry = _consensus_geometry(
            included_records, samples, family_configs, axis
        )
    return {
        "cohort_id": first["cohort_id"],
        "locus_id": first["locus_id"],
        "candidate_id": first["candidate_id"],
        "axis": axis,
        "target_assay_family": target_assay_family,
        "posterior_present": posterior,
        "log_posterior_odds": log_odds,
        "status": status,
        "anchor_families": sorted(anchors),
        "support_families": sorted(support_families),
        "geometry": geometry,
        "contributors": contributors,
    }


def adjudicate_all(
    evidence: Mapping[str, object],
    manifest: Mapping[str, object],
    hierarchy: Mapping[str, object],
    *,
    leave_one_assay_out: bool = True,
) -> dict:
    records = evidence.get("records", [])
    if not isinstance(records, list):
        raise ValueError("evidence.records must be a list")
    groups: Dict[Tuple[str, str, str, str], List[Mapping[str, object]]] = {}
    for record in records:
        key = (
            str(record["cohort_id"]),
            str(record["locus_id"]),
            str(record["candidate_id"]),
            str(record["axis"]),
        )
        groups.setdefault(key, []).append(record)

    results = []
    for group_records in groups.values():
        target_families: Iterable[Optional[str]]
        if leave_one_assay_out:
            target_families = sorted({
                str(sample["assay_family"])
                for sample in manifest.get("samples", [])
                if str(sample.get("cohort_id")) == str(group_records[0]["cohort_id"])
            })
        else:
            target_families = [None]
        for target_family in target_families:
            results.append(adjudicate_candidate(
                group_records,
                manifest,
                hierarchy,
                target_assay_family=target_family,
            ))

    status_counts: Dict[str, int] = {}
    for result in results:
        status_counts[result["status"]] = status_counts.get(result["status"], 0) + 1
    return {
        "schema_version": 1,
        "leave_one_assay_out": leave_one_assay_out,
        "n_candidate_folds": len(results),
        "status_counts": status_counts,
        "results": results,
    }


def hierarchy_with_sample_overrides(
    hierarchy: Mapping[str, object],
    overrides: Mapping[str, object],
) -> dict:
    """Deep-copy helper used by calibration sweeps without mutating defaults."""
    result = copy.deepcopy(hierarchy)
    for axis, families in overrides.items():
        for family, values in families.items():
            result["axes"][axis]["families"][family].update(values)
    return result
