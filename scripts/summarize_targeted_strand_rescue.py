#!/usr/bin/env python3
"""Summarize a targeted strand-rescue production matrix without mutating it.

The production driver publishes a canonical ``status.json`` for each target
stage.  This tool opens attempt artifacts only when that status says
``complete`` and points at a validated receipt.  In particular, it never scans
or reads a running attempt directory.

By default the JSON summary is printed to stdout and no files are written.
JSON, TSV, and Markdown files are produced only when their corresponding
explicit output option is supplied::

    python scripts/summarize_targeted_strand_rescue.py \
      --json-out summary.json --tsv-out targets.tsv --markdown-out summary.md \
      --stdout none

The summary includes receipt counts, GNU time resource measurements, every
progress JSONL stage timing, v5 report action totals, and v4 BAM-audit totals.
"""

from __future__ import annotations

import argparse
import csv
import errno
import io
import json
import os
import sys
import time
import uuid
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    Set,
    Tuple,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MATRIX = (
    REPO_ROOT
    / "consensus_validation_outputs"
    / "strand_rescue_targeted_full_20260716"
    / "manifests"
    / "targeted_sr_run_matrix.json"
)

ACTION_KEYS = (
    "fetch_records",
    "action_records",
    "rescue_decisions",
    "rescue_components",
    "tf_edge_updates",
    "nuc_edge_updates",
)

CORE_AUDIT_KEYS = (
    "records",
    "records_with_decisions",
    "msp_to_tf_rescues",
    "edge_refinements_tf_sr",
    "edge_refinements_nuc_sr",
    "geometry_harmonizations",
    "named_groups",
    "annotations_tf_sr",
    "annotations_nuc_sr",
    "sr_annotations",
    "fixed_baseline_annotations",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _read_json(path: Path) -> Any:
    with path.open() as handle:
        return json.load(handle)


def _resolve_declared_path(value: str, declaring_file: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = declaring_file.parent / path
        return path.resolve()
    if path.exists():
        return path.resolve()

    # Production manifests preserve the absolute path used for the validated
    # run. When the complete validation tree is copied to another host (for
    # example from a Linux workstation to a laptop), relocate paths anchored at the
    # repository-owned consensus_validation_outputs directory. Do not guess
    # for arbitrary missing absolute paths.
    anchor = "consensus_validation_outputs"
    if anchor in path.parts:
        suffix = Path(*path.parts[path.parts.index(anchor):])
        for parent in declaring_file.resolve().parents:
            candidate = parent / suffix
            if candidate.exists():
                return candidate.resolve()
    return path.resolve()


def _within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _target_records(matrix_path: Path) -> Tuple[Mapping[str, Any], Path, List[Dict[str, Any]]]:
    matrix_path = matrix_path.expanduser().resolve()
    matrix = _read_json(matrix_path)
    if matrix.get("schema") != "fiberhmm.validation.targeted_strand_rescue_matrix.v1":
        raise ValueError(f"unsupported targeted SR matrix schema: {matrix_path}")
    output_root = _resolve_declared_path(str(matrix["output_root"]), matrix_path)
    targets: List[Dict[str, Any]] = []

    ddda = matrix["ddda"]
    for record in ddda["targets"]:
        targets.append(
            {
                "target": f"ddda:{record['id']}",
                "assay": "ddda",
                "target_id": str(record["id"]),
                "display_name": str(record.get("display_name", record["id"])),
                "preset": str(ddda["preset"]),
                "region": str(record["region_cli_zero_based_half_open"]),
                "bam_count": len(record["bams"]),
                "min_support": int(ddda["min_support"]),
            }
        )

    dddb_path = _resolve_declared_path(str(matrix["dddb_manifest"]), matrix_path)
    dddb = _read_json(dddb_path)
    if dddb.get("schema") != "fiberhmm.validation.dddb_wt_full_amplicons.v1":
        raise ValueError(f"unsupported DddB target manifest schema: {dddb_path}")
    min_support = int(dddb["run_recommendation"]["min_support"])
    for record in dddb["targets"]:
        targets.append(
            {
                "target": f"dddb:{record['id']}",
                "assay": "dddb",
                "target_id": str(record["id"]),
                "display_name": str(record.get("display_name", record["id"])),
                "preset": str(matrix["dddb"]["preset"]),
                "region": str(record["region_cli_zero_based_half_open"]),
                "bam_count": len(record["included_bam_ids"]),
                "min_support": min_support,
            }
        )

    keys = [record["target"] for record in targets]
    if len(keys) != len(set(keys)):
        raise ValueError("target matrix contains duplicate target keys")
    expected = int(matrix.get("expected_matrix", {}).get("total_targets", len(targets)))
    if len(targets) != expected:
        raise ValueError(f"matrix has {len(targets)} targets, expected {expected}")
    return matrix, output_root, targets


def _elapsed_seconds(value: str) -> float:
    fields = value.strip().split(":")
    try:
        if len(fields) == 3:
            hours, minutes, seconds = fields
            return int(hours) * 3600.0 + int(minutes) * 60.0 + float(seconds)
        if len(fields) == 2:
            minutes, seconds = fields
            return int(minutes) * 60.0 + float(seconds)
        if len(fields) == 1:
            return float(fields[0])
    except ValueError as error:
        raise ValueError(f"invalid GNU time elapsed value: {value!r}") from error
    raise ValueError(f"invalid GNU time elapsed value: {value!r}")


def parse_gnu_time(path: Path) -> Dict[str, Any]:
    """Parse the stable fields emitted by ``/usr/bin/time -v``."""
    labels = {
        "User time (seconds)": ("user_seconds", float),
        "System time (seconds)": ("system_seconds", float),
        "Percent of CPU this job got": ("cpu_percent", lambda value: float(value.rstrip("%"))),
        "Maximum resident set size (kbytes)": ("max_rss_kib", int),
        "Major (requiring I/O) page faults": ("major_page_faults", int),
        "Minor (reclaiming a frame) page faults": ("minor_page_faults", int),
        "Voluntary context switches": ("voluntary_context_switches", int),
        "Involuntary context switches": ("involuntary_context_switches", int),
        "File system inputs": ("file_system_inputs", int),
        "File system outputs": ("file_system_outputs", int),
        "Exit status": ("exit_status", int),
    }
    result: Dict[str, Any] = {"path": str(path.resolve())}
    for raw_line in path.read_text(errors="replace").splitlines():
        line = raw_line.strip()
        if line.startswith("Elapsed (wall clock) time"):
            separator = line.rfind(": ")
            if separator >= 0:
                result["wall_seconds"] = _elapsed_seconds(line[separator + 2 :])
            continue
        for label, (key, converter) in labels.items():
            prefix = label + ": "
            if line.startswith(prefix):
                result[key] = converter(line[len(prefix) :])
                break
    required = {"wall_seconds", "max_rss_kib", "exit_status"}
    missing = sorted(required - set(result))
    if missing:
        raise ValueError(f"incomplete GNU time log {path}: missing {', '.join(missing)}")
    result["max_rss_bytes"] = int(result["max_rss_kib"]) * 1024
    return result


def parse_progress(path: Path) -> Dict[str, Any]:
    """Return every stage event plus stage-name aggregates."""
    stages: List[Dict[str, Any]] = []
    by_name: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
    last_elapsed = 0.0
    with path.open() as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid progress JSON at {path}:{line_number}") from error
            if event.get("schema") != "fiberhmm.performance.progress.v1":
                raise ValueError(f"unexpected progress schema at {path}:{line_number}")
            stage = event.get("stage")
            if not isinstance(stage, dict) or not stage.get("name"):
                raise ValueError(f"progress event lacks a stage at {path}:{line_number}")
            record = {
                "name": str(stage["name"]),
                "occurrence": sum(value["name"] == stage["name"] for value in stages) + 1,
                "wall_seconds": float(stage.get("wall_seconds", 0.0)),
                "rss_bytes_after": stage.get("rss_bytes_after"),
                "process_peak_rss_bytes_after": stage.get("process_peak_rss_bytes_after"),
                "elapsed_wall_seconds": float(event.get("elapsed_wall_seconds", 0.0)),
                "details": stage.get("details", {}),
            }
            stages.append(record)
            last_elapsed = record["elapsed_wall_seconds"]
            aggregate = by_name.setdefault(
                record["name"],
                {
                    "occurrences": 0,
                    "wall_seconds": 0.0,
                    "max_rss_bytes_after": 0,
                    "max_process_peak_rss_bytes_after": 0,
                },
            )
            aggregate["occurrences"] += 1
            aggregate["wall_seconds"] += record["wall_seconds"]
            aggregate["max_rss_bytes_after"] = max(
                int(aggregate["max_rss_bytes_after"]), int(record["rss_bytes_after"] or 0)
            )
            aggregate["max_process_peak_rss_bytes_after"] = max(
                int(aggregate["max_process_peak_rss_bytes_after"]),
                int(record["process_peak_rss_bytes_after"] or 0),
            )
    slowest = None
    if by_name:
        slowest_name, slowest_value = max(
            by_name.items(), key=lambda item: float(item[1]["wall_seconds"])
        )
        slowest = {
            "name": slowest_name,
            "wall_seconds": float(slowest_value["wall_seconds"]),
        }
    return {
        "path": str(path.resolve()),
        "event_count": len(stages),
        "elapsed_wall_seconds": last_elapsed,
        "stages": stages,
        "by_name": dict(by_name),
        "slowest_stage": slowest,
    }


def _artifact_map(receipt: Mapping[str, Any]) -> Dict[str, Mapping[str, Any]]:
    result: Dict[str, Mapping[str, Any]] = {}
    artifacts = receipt.get("artifacts", [])
    if not isinstance(artifacts, list):
        raise ValueError("receipt artifacts are not a list")
    for record in artifacts:
        relative = str(record["relative_path"])
        if relative in result:
            raise ValueError(f"receipt repeats artifact {relative}")
        result[relative] = record
    return result


def _check_receipt_artifacts(
    attempt: Path, artifacts: Mapping[str, Mapping[str, Any]]
) -> Tuple[int, List[str]]:
    checked = 0
    warnings: List[str] = []
    for relative, record in artifacts.items():
        path = (attempt / relative).resolve()
        if not _within(path, attempt):
            warnings.append(f"receipt artifact escapes attempt: {relative}")
            continue
        try:
            size = path.stat().st_size
        except OSError as error:
            warnings.append(f"receipt artifact unavailable: {relative}: {error}")
            continue
        checked += 1
        if size != int(record.get("size_bytes", -1)):
            warnings.append(
                f"receipt artifact size drift: {relative}: {size} != {record.get('size_bytes')}"
            )
    return checked, warnings


def _status_identity(status: Mapping[str, Any]) -> Tuple[Any, ...]:
    return (
        status.get("state"),
        status.get("contract_sha256"),
        status.get("attempt_dir"),
        status.get("receipt"),
    )


def _completed_stage_chain(
    stage_root: Path, target: str, expected_schema: str
) -> Tuple[Dict[str, Any], List[str]]:
    """Read one completed stage, or return status only for a non-complete stage."""
    status_path = stage_root / "status.json"
    if not status_path.is_file():
        return {"state": "not_started", "status_path": str(status_path.resolve())}, []
    try:
        status = _read_json(status_path)
    except (OSError, json.JSONDecodeError) as error:
        return {"state": "invalid_status", "status_path": str(status_path.resolve())}, [str(error)]
    state = str(status.get("state", "unknown"))
    result: Dict[str, Any] = {
        "state": state,
        "status_path": str(status_path.resolve()),
        "updated_at": status.get("updated_at"),
    }
    # This is the boundary that protects active attempts: do not resolve, stat,
    # list, or open anything under an attempt unless the canonical state is complete.
    if state != "complete":
        result["attempt_dir"] = status.get("attempt_dir")
        result["last_completed_stage"] = status.get("last_completed_stage")
        result["elapsed_wall_seconds"] = status.get("elapsed_wall_seconds")
        result["error"] = status.get("error")
        return result, []

    warnings: List[str] = []
    attempt_value = status.get("attempt_dir")
    if not attempt_value:
        return {**result, "state": "complete_invalid"}, ["complete status lacks attempt_dir"]
    attempts_root = (stage_root / "attempts").resolve()
    declared_attempt = Path(str(attempt_value)).expanduser()
    attempt = declared_attempt.resolve()
    if declared_attempt.is_absolute() and not attempt.exists():
        relocated_attempt = (attempts_root / declared_attempt.name).resolve()
        if relocated_attempt.is_dir():
            attempt = relocated_attempt
    if not _within(attempt, attempts_root):
        return {**result, "state": "complete_invalid"}, [
            "complete attempt is outside stage attempts root"
        ]
    receipt_value = status.get("receipt", str(attempt / "validation_receipt.json"))
    declared_receipt = Path(str(receipt_value)).expanduser()
    receipt_path = declared_receipt.resolve()
    canonical_receipt = (attempt / "validation_receipt.json").resolve()
    if declared_receipt.is_absolute() and not receipt_path.exists():
        if declared_receipt.name == "validation_receipt.json":
            receipt_path = canonical_receipt
    if receipt_path != canonical_receipt:
        return {**result, "state": "complete_invalid"}, ["status points at a noncanonical receipt"]
    try:
        receipt = _read_json(receipt_path)
    except (OSError, json.JSONDecodeError) as error:
        return {**result, "state": "complete_invalid"}, [f"cannot read completed receipt: {error}"]
    if receipt.get("schema") != expected_schema:
        warnings.append(f"unexpected receipt schema: {receipt.get('schema')}")
    if receipt.get("validated") is not True:
        warnings.append("receipt is not validated")
    if receipt.get("target") != target:
        warnings.append(f"receipt target mismatch: {receipt.get('target')} != {target}")
    if receipt.get("contract_sha256") != status.get("contract_sha256"):
        warnings.append("receipt/status contract mismatch")
    try:
        artifacts = _artifact_map(receipt)
        checked, artifact_warnings = _check_receipt_artifacts(attempt, artifacts)
        warnings.extend(artifact_warnings)
    except (KeyError, TypeError, ValueError) as error:
        artifacts = {}
        checked = 0
        warnings.append(str(error))
    try:
        status_after = _read_json(status_path)
    except (OSError, json.JSONDecodeError) as error:
        warnings.append(f"status changed or became unreadable during snapshot: {error}")
    else:
        if _status_identity(status_after) != _status_identity(status):
            warnings.append("stage status chain changed during snapshot")
    result.update(
        {
            "attempt_dir": str(attempt),
            "attempt_id": attempt.name,
            "receipt_path": str(receipt_path),
            "receipt": receipt,
            "artifacts": artifacts,
            "receipt_artifact_size_checks": checked,
        }
    )
    if warnings:
        result["state"] = "complete_with_warnings"
    return result, warnings


def _required_artifact(stage: Mapping[str, Any], relative: str) -> Path:
    artifacts = stage.get("artifacts", {})
    if relative not in artifacts:
        raise ValueError(f"completed receipt does not declare {relative}")
    attempt = Path(str(stage["attempt_dir"]))
    return (attempt / relative).resolve()


def _numeric_mapping(value: Any) -> Dict[str, int]:
    if not isinstance(value, dict):
        return {}
    return {
        str(key): int(item)
        for key, item in value.items()
        if isinstance(item, (int, float)) and not isinstance(item, bool)
    }


def _extract_inference(stage: MutableMapping[str, Any], warnings: List[str]) -> None:
    if not str(stage.get("state", "")).startswith("complete"):
        return
    try:
        report_path = _required_artifact(stage, "report.json")
        progress_path = _required_artifact(stage, "progress.jsonl")
        time_path = _required_artifact(stage, "time.txt")
        report = _read_json(report_path)
        if report.get("schema") != "fiberhmm.strand_rescue.v5":
            warnings.append(f"unexpected report schema: {report.get('schema')}")
        action_storage = report.get("strand_rescue", {}).get("action_storage", {})
        actions = _numeric_mapping(action_storage.get("totals", {}))
        missing = sorted(set(ACTION_KEYS) - set(actions))
        if missing:
            warnings.append("report action totals missing: " + ", ".join(missing))
        receipt_summary = stage["receipt"].get("report_summary", {})
        aliases = {
            "action_records": "action_record_count",
            "rescue_decisions": "decision_count",
            "rescue_components": "rescue_component_count",
            "tf_edge_updates": "tf_harmonization_count",
            "nuc_edge_updates": "nuc_harmonization_count",
        }
        for action_key, receipt_key in aliases.items():
            if action_key in actions and receipt_key in receipt_summary:
                if actions[action_key] != int(receipt_summary[receipt_key]):
                    warnings.append(
                        f"report/receipt count mismatch for {action_key}: "
                        f"{actions[action_key]} != {receipt_summary[receipt_key]}"
                    )
        strand_rescue = report.get("strand_rescue", {})
        tf_sites = strand_rescue.get("sites", [])
        nuc_sites = strand_rescue.get("edge_refinement", {}).get("nuc", {}).get("sites", [])
        stage["report"] = {
            "path": str(report_path),
            "schema": report.get("schema"),
            "applicable": bool(strand_rescue.get("applicable")),
            "n_raw_reads": int(report.get("n_raw_reads", 0)),
            "n_analyzed_molecules": int(report.get("n_analyzed_molecules", 0)),
            "tf_site_count": len(tf_sites) if isinstance(tf_sites, list) else None,
            "nuc_site_count": len(nuc_sites) if isinstance(nuc_sites, list) else None,
            "actions": actions,
            "rescue_counts": _numeric_mapping(strand_rescue.get("counts", {})),
            "action_stream_count": len(action_storage.get("streams", [])),
            "action_streams": [
                {
                    key: stream.get(key)
                    for key in (
                        "input_index",
                        "input_id",
                        "path",
                        "fetch_record_count",
                        "action_record_count",
                        "rescue_decision_count",
                        "rescue_component_count",
                        "tf_edge_update_count",
                        "nuc_edge_update_count",
                        "compressed_size_bytes",
                        "uncompressed_size_bytes",
                    )
                }
                for stream in action_storage.get("streams", [])
            ],
        }
        stage["progress"] = parse_progress(progress_path)
        stage["gnu_time"] = parse_gnu_time(time_path)
        stage["receipt_summary"] = receipt_summary
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        warnings.append(f"inference summary extraction failed: {error}")


def _extract_materialization(stage: MutableMapping[str, Any], warnings: List[str]) -> None:
    if not str(stage.get("state", "")).startswith("complete"):
        return
    try:
        audit_path = _required_artifact(stage, "audit.json")
        annotate_time_path = _required_artifact(stage, "annotate.time.txt")
        audit_time_path = _required_artifact(stage, "audit.time.txt")
        audit = _read_json(audit_path)
        if audit.get("schema") != "fiberhmm.strand_rescue.audit.v4":
            warnings.append(f"unexpected audit schema: {audit.get('schema')}")
        if audit.get("valid") is not True:
            warnings.append("audit is not valid")
        totals = _numeric_mapping(audit.get("totals", {}))
        receipt_totals = stage["receipt"].get("audit_summary", {}).get("totals", {})
        if totals != _numeric_mapping(receipt_totals):
            warnings.append("audit totals differ from materialization receipt")
        stage["audit"] = {
            "path": str(audit_path),
            "schema": audit.get("schema"),
            "valid": audit.get("valid") is True,
            "file_count": int(audit.get("file_count", 0)),
            "totals": totals,
            "threshold_states": audit.get("threshold_states", {}),
        }
        stage["annotate_gnu_time"] = parse_gnu_time(annotate_time_path)
        stage["audit_gnu_time"] = parse_gnu_time(audit_time_path)
        stage["receipt_summary"] = stage["receipt"].get("audit_summary", {})
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        warnings.append(f"materialization summary extraction failed: {error}")


def _cross_check_target(record: Mapping[str, Any], warnings: List[str]) -> None:
    report = record.get("inference", {}).get("report", {})
    audit = record.get("materialization", {}).get("audit", {})
    actions = report.get("actions", {})
    totals = audit.get("totals", {})
    comparisons = (
        ("rescue_decisions", "msp_to_tf_rescues"),
        ("tf_edge_updates", "edge_refinements_tf_sr"),
        ("nuc_edge_updates", "edge_refinements_nuc_sr"),
        ("action_records", "records_with_decisions"),
    )
    for action_key, audit_key in comparisons:
        if action_key in actions and audit_key in totals:
            if int(actions[action_key]) != int(totals[audit_key]):
                warnings.append(
                    f"report/audit mismatch: {action_key}={actions[action_key]} "
                    f"but {audit_key}={totals[audit_key]}"
                )
    if "geometry_harmonizations" in totals:
        expected = int(actions.get("tf_edge_updates", 0)) + int(
            actions.get("nuc_edge_updates", 0)
        )
        if expected != int(totals["geometry_harmonizations"]):
            warnings.append("audit geometry_harmonizations differs from report H actions")


def _sum_flat_numeric(mappings: Iterable[Mapping[str, Any]]) -> Dict[str, int]:
    result: Dict[str, int] = {}
    for mapping in mappings:
        for key, value in mapping.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                result[str(key)] = result.get(str(key), 0) + int(value)
    return result


def _sum_nested_numeric(values: Iterable[Any]) -> Any:
    values = [value for value in values if isinstance(value, dict)]
    keys: Set[str] = set()
    for value in values:
        keys.update(str(key) for key in value)
    result: Dict[str, Any] = {}
    for key in sorted(keys):
        children = [value[key] for value in values if key in value]
        if children and all(isinstance(child, dict) for child in children):
            result[key] = _sum_nested_numeric(children)
        else:
            numeric = [
                child
                for child in children
                if isinstance(child, (int, float)) and not isinstance(child, bool)
            ]
            if numeric:
                result[key] = sum(numeric)
    return result


def _aggregate_rows(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    inference_complete = [row for row in rows if row["inference"]["state"] == "complete"]
    materialization_complete = [
        row for row in rows if row["materialization"]["state"] == "complete"
    ]
    fully_complete = [row for row in rows if row.get("fully_complete")]
    action_totals = _sum_flat_numeric(
        row["inference"].get("report", {}).get("actions", {})
        for row in inference_complete
    )
    audit_totals = _sum_flat_numeric(
        row["materialization"].get("audit", {}).get("totals", {})
        for row in materialization_complete
    )
    stage_totals: Dict[str, Dict[str, Any]] = {}
    for row in inference_complete:
        for name, value in row["inference"].get("progress", {}).get("by_name", {}).items():
            aggregate = stage_totals.setdefault(
                name,
                {
                    "target_count": 0,
                    "occurrences": 0,
                    "wall_seconds": 0.0,
                    "max_process_peak_rss_bytes_after": 0,
                },
            )
            aggregate["target_count"] += 1
            aggregate["occurrences"] += int(value.get("occurrences", 0))
            aggregate["wall_seconds"] += float(value.get("wall_seconds", 0.0))
            aggregate["max_process_peak_rss_bytes_after"] = max(
                int(aggregate["max_process_peak_rss_bytes_after"]),
                int(value.get("max_process_peak_rss_bytes_after", 0)),
            )

    inference_times = [
        row["inference"].get("gnu_time", {}) for row in inference_complete
    ]
    annotate_times = [
        row["materialization"].get("annotate_gnu_time", {})
        for row in materialization_complete
    ]
    audit_times = [
        row["materialization"].get("audit_gnu_time", {})
        for row in materialization_complete
    ]

    def timing(values: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        return {
            "completed_jobs": len([value for value in values if value]),
            "wall_seconds": sum(float(value.get("wall_seconds", 0.0)) for value in values),
            "user_seconds": sum(float(value.get("user_seconds", 0.0)) for value in values),
            "system_seconds": sum(float(value.get("system_seconds", 0.0)) for value in values),
            "max_rss_bytes": max(
                (int(value.get("max_rss_bytes", 0)) for value in values),
                default=0,
            ),
        }

    return {
        "target_count": len(rows),
        "inference_complete": len(inference_complete),
        "materialization_complete": len(materialization_complete),
        "fully_complete": len(fully_complete),
        "targets_with_warnings": sum(bool(row.get("warnings")) for row in rows),
        "incomplete_targets": [
            row["target"] for row in rows if not row.get("fully_complete")
        ],
        "action_totals": action_totals,
        "audit_totals": audit_totals,
        "audit_threshold_states": _sum_nested_numeric(
            row["materialization"].get("audit", {}).get("threshold_states", {})
            for row in materialization_complete
        ),
        "timing": {
            "inference": timing(inference_times),
            "annotation": timing(annotate_times),
            "audit": timing(audit_times),
            "end_to_end_wall_seconds": (
                timing(inference_times)["wall_seconds"]
                + timing(annotate_times)["wall_seconds"]
                + timing(audit_times)["wall_seconds"]
            ),
        },
        "progress_stage_totals": stage_totals,
    }


def build_summary(
    matrix_path: Path = DEFAULT_MATRIX,
    selected_targets: Optional[Set[str]] = None,
) -> Dict[str, Any]:
    matrix, output_root, targets = _target_records(matrix_path)
    if selected_targets:
        known = {record["target"] for record in targets}
        unknown = sorted(selected_targets - known)
        if unknown:
            raise ValueError("unknown target selector(s): " + ", ".join(unknown))
        targets = [record for record in targets if record["target"] in selected_targets]
    rows: List[Dict[str, Any]] = []
    for metadata in targets:
        target = metadata["target"]
        target_root = output_root / "runs" / metadata["assay"] / metadata["target_id"]
        inference, inference_warnings = _completed_stage_chain(
            target_root / "inference",
            target,
            "fiberhmm.validation.targeted_sr.inference_receipt.v1",
        )
        materialization, materialization_warnings = _completed_stage_chain(
            target_root / "materialization",
            target,
            "fiberhmm.validation.targeted_sr.materialization_receipt.v1",
        )
        warnings = inference_warnings + materialization_warnings
        _extract_inference(inference, warnings)
        _extract_materialization(materialization, warnings)
        row: Dict[str, Any] = {
            **metadata,
            "inference": inference,
            "materialization": materialization,
        }
        _cross_check_target(row, warnings)
        row["fully_complete"] = (
            inference.get("state") == "complete"
            and materialization.get("state") == "complete"
            and "report" in inference
            and materialization.get("audit", {}).get("valid") is True
        )
        row["warnings"] = warnings
        rows.append(row)
    aggregate = _aggregate_rows(rows)
    by_assay = {
        assay: _aggregate_rows([row for row in rows if row["assay"] == assay])
        for assay in ("ddda", "dddb")
        if any(row["assay"] == assay for row in rows)
    }
    return {
        "schema": "fiberhmm.validation.targeted_sr.production_summary.v1",
        "snapshot_at": _utc_now(),
        "matrix": str(Path(matrix_path).expanduser().resolve()),
        "matrix_schema": matrix.get("schema"),
        "output_root": str(output_root),
        "receipt_policy": (
            "Only canonical stages marked complete are opened; validated receipt hashes "
            "are trusted and declared artifact sizes are rechecked without rereading BAM bytes."
        ),
        "selection": [row["target"] for row in rows],
        "aggregate": aggregate,
        "by_assay": by_assay,
        "targets": rows,
    }


def _value(row: Mapping[str, Any], *keys: str) -> Any:
    value: Any = row
    for key in keys:
        if not isinstance(value, dict):
            return ""
        value = value.get(key, "")
    return value


def render_tsv(summary: Mapping[str, Any]) -> str:
    rows = summary["targets"]
    stage_names: List[str] = []
    audit_names: Set[str] = set()
    for row in rows:
        for name in _value(row, "inference", "progress", "by_name") or {}:
            if name not in stage_names:
                stage_names.append(name)
        audit_names.update((_value(row, "materialization", "audit", "totals") or {}).keys())
    base_fields = [
        "target",
        "assay",
        "target_id",
        "display_name",
        "region",
        "bam_count",
        "min_support",
        "inference_state",
        "materialization_state",
        "fully_complete",
        "n_raw_reads",
        "n_analyzed_molecules",
        "tf_site_count",
        "nuc_site_count",
        *ACTION_KEYS,
        "inference_wall_seconds",
        "inference_max_rss_bytes",
        "progress_elapsed_wall_seconds",
        "slowest_stage",
        "slowest_stage_seconds",
        "annotation_wall_seconds",
        "annotation_max_rss_bytes",
        "audit_wall_seconds",
        "audit_max_rss_bytes",
        "end_to_end_wall_seconds",
        "inference_attempt_dir",
        "materialization_attempt_dir",
        "warnings",
    ]
    audit_fields = [f"audit__{name}" for name in sorted(audit_names)]
    stage_fields = [f"stage_seconds__{name}" for name in stage_names]
    fields = base_fields + audit_fields + stage_fields
    handle = io.StringIO(newline="")
    writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        actions = _value(row, "inference", "report", "actions") or {}
        progress = _value(row, "inference", "progress") or {}
        slowest = progress.get("slowest_stage") or {}
        inference_time = _value(row, "inference", "gnu_time") or {}
        annotate_time = _value(row, "materialization", "annotate_gnu_time") or {}
        audit_time = _value(row, "materialization", "audit_gnu_time") or {}
        output: Dict[str, Any] = {
            key: row.get(key, "")
            for key in (
                "target",
                "assay",
                "target_id",
                "display_name",
                "region",
                "bam_count",
                "min_support",
                "fully_complete",
            )
        }
        output.update(
            {
                "inference_state": _value(row, "inference", "state"),
                "materialization_state": _value(row, "materialization", "state"),
                "n_raw_reads": _value(row, "inference", "report", "n_raw_reads"),
                "n_analyzed_molecules": _value(row, "inference", "report", "n_analyzed_molecules"),
                "tf_site_count": _value(row, "inference", "report", "tf_site_count"),
                "nuc_site_count": _value(row, "inference", "report", "nuc_site_count"),
                "inference_wall_seconds": inference_time.get("wall_seconds", ""),
                "inference_max_rss_bytes": inference_time.get("max_rss_bytes", ""),
                "progress_elapsed_wall_seconds": progress.get("elapsed_wall_seconds", ""),
                "slowest_stage": slowest.get("name", ""),
                "slowest_stage_seconds": slowest.get("wall_seconds", ""),
                "annotation_wall_seconds": annotate_time.get("wall_seconds", ""),
                "annotation_max_rss_bytes": annotate_time.get("max_rss_bytes", ""),
                "audit_wall_seconds": audit_time.get("wall_seconds", ""),
                "audit_max_rss_bytes": audit_time.get("max_rss_bytes", ""),
                "end_to_end_wall_seconds": sum(
                    float(value.get("wall_seconds", 0.0))
                    for value in (inference_time, annotate_time, audit_time)
                ) or "",
                "inference_attempt_dir": _value(row, "inference", "attempt_dir"),
                "materialization_attempt_dir": _value(row, "materialization", "attempt_dir"),
                "warnings": " | ".join(row.get("warnings", [])),
            }
        )
        output.update({key: actions.get(key, "") for key in ACTION_KEYS})
        audit_totals = _value(row, "materialization", "audit", "totals") or {}
        output.update({f"audit__{name}": audit_totals.get(name, "") for name in audit_names})
        by_name = progress.get("by_name", {})
        output.update(
            {
                f"stage_seconds__{name}": by_name.get(name, {}).get("wall_seconds", "")
                for name in stage_names
            }
        )
        writer.writerow(output)
    return handle.getvalue()


def _markdown_escape(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _format_seconds(value: Any) -> str:
    if value in (None, ""):
        return "—"
    return f"{float(value):,.2f}"


def _format_int(value: Any) -> str:
    if value in (None, ""):
        return "—"
    return f"{int(value):,}"


def render_markdown(summary: Mapping[str, Any]) -> str:
    aggregate = summary["aggregate"]
    lines = [
        "# Targeted strand-rescue production summary",
        "",
        f"Snapshot: `{summary['snapshot_at']}`  ",
        f"Matrix: `{summary['matrix']}`",
        "",
        (
            "Only canonical completed stages and their validated receipts are "
            "included in scientific and performance totals."
        ),
        "",
        "## Completion",
        "",
        f"- Targets: {aggregate['target_count']}",
        f"- Inference complete: {aggregate['inference_complete']}",
        f"- Materialization complete: {aggregate['materialization_complete']}",
        f"- Fully complete and audit-valid: {aggregate['fully_complete']}",
        f"- Targets with consistency warnings: {aggregate['targets_with_warnings']}",
        "",
        "## Per-target results",
        "",
        (
            "| Target | Inference | Materialization | Raw reads | Molecules | R | "
            "TF H | Nuc H | Inference s | Peak GiB | Slowest stage |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in summary["targets"]:
        report = _value(row, "inference", "report") or {}
        actions = report.get("actions", {})
        timing = _value(row, "inference", "gnu_time") or {}
        slowest = (_value(row, "inference", "progress", "slowest_stage") or {})
        peak_gib = (
            float(timing["max_rss_bytes"]) / (1024 ** 3)
            if timing.get("max_rss_bytes") not in (None, "")
            else None
        )
        slowest_text = "—"
        if slowest:
            slowest_text = (
                f"{slowest.get('name')} "
                f"({_format_seconds(slowest.get('wall_seconds'))} s)"
            )
        lines.append(
            "| "
            + " | ".join(
                [
                    _markdown_escape(row["target"]),
                    _markdown_escape(row["inference"]["state"]),
                    _markdown_escape(row["materialization"]["state"]),
                    _format_int(report.get("n_raw_reads")),
                    _format_int(report.get("n_analyzed_molecules")),
                    _format_int(actions.get("rescue_decisions")),
                    _format_int(actions.get("tf_edge_updates")),
                    _format_int(actions.get("nuc_edge_updates")),
                    _format_seconds(timing.get("wall_seconds")),
                    "—" if peak_gib is None else f"{peak_gib:.2f}",
                    _markdown_escape(slowest_text),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Aggregate action and audit totals",
            "",
            "| Metric | Count |",
            "|---|---:|",
        ]
    )
    for key in ACTION_KEYS:
        lines.append(f"| `{key}` | {_format_int(aggregate['action_totals'].get(key))} |")
    for key in CORE_AUDIT_KEYS:
        if key in aggregate["audit_totals"]:
            lines.append(f"| `audit.{key}` | {_format_int(aggregate['audit_totals'][key])} |")

    lines.extend(
        [
            "",
            "## Aggregate performance",
            "",
            "| Job | Completed | Wall s | Peak GiB |",
            "|---|---:|---:|---:|",
        ]
    )
    for name in ("inference", "annotation", "audit"):
        timing = aggregate["timing"][name]
        lines.append(
            f"| {name} | {timing['completed_jobs']} | {_format_seconds(timing['wall_seconds'])} | "
            f"{timing['max_rss_bytes'] / (1024 ** 3):.2f} |"
        )
    lines.extend(
        [
            "",
            "### Inference stage totals",
            "",
            "| Stage | Targets | Occurrences | Wall s | Share of completed inference |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    total_inference = float(aggregate["timing"]["inference"]["wall_seconds"])
    ordered = sorted(
        aggregate["progress_stage_totals"].items(),
        key=lambda item: float(item[1]["wall_seconds"]),
        reverse=True,
    )
    for name, value in ordered:
        share = 100.0 * float(value["wall_seconds"]) / total_inference if total_inference else 0.0
        lines.append(
            f"| `{_markdown_escape(name)}` | {value['target_count']} | {value['occurrences']} | "
            f"{_format_seconds(value['wall_seconds'])} | {share:.1f}% |"
        )
    if aggregate["incomplete_targets"]:
        lines.extend(
            [
                "",
                "## Incomplete targets",
                "",
                ", ".join(f"`{value}`" for value in aggregate["incomplete_targets"]),
            ]
        )
    warning_rows = [row for row in summary["targets"] if row.get("warnings")]
    if warning_rows:
        lines.extend(["", "## Consistency warnings", ""])
        for row in warning_rows:
            for warning in row["warnings"]:
                lines.append(f"- `{row['target']}`: {warning}")
    lines.append("")
    return "\n".join(lines)


def _atomic_write(path: Path, text: str) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.partial")
    with temporary.open("x") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    delay = 0.05
    for attempt in range(10):
        try:
            os.replace(temporary, path)
            break
        except PermissionError as error:
            if error.errno not in {errno.EACCES, errno.EPERM} or attempt == 9:
                raise
            time.sleep(delay)
            delay = min(0.75, delay * 2.0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument(
        "--target",
        action="append",
        help="Restrict to an exact assay:id target key; repeatable",
    )
    parser.add_argument("--json-out", type=Path, help="Explicit JSON output path")
    parser.add_argument("--tsv-out", type=Path, help="Explicit per-target TSV output path")
    parser.add_argument("--markdown-out", type=Path, help="Explicit Markdown output path")
    parser.add_argument(
        "--stdout",
        choices=("json", "tsv", "markdown", "none"),
        default="json",
        help="Format printed to stdout; defaults to JSON and never writes a file",
    )
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Exit nonzero unless every selected target is fully complete and audit-valid",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit nonzero when any completed target has a consistency warning",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        summary = build_summary(
            args.matrix,
            set(args.target) if args.target else None,
        )
        json_text = json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n"
        tsv_text = render_tsv(summary)
        markdown_text = render_markdown(summary)
        if args.json_out is not None:
            _atomic_write(args.json_out, json_text)
        if args.tsv_out is not None:
            _atomic_write(args.tsv_out, tsv_text)
        if args.markdown_out is not None:
            _atomic_write(args.markdown_out, markdown_text)
        stdout_text = {
            "json": json_text,
            "tsv": tsv_text,
            "markdown": markdown_text,
        }.get(args.stdout)
        if stdout_text is not None:
            sys.stdout.write(stdout_text)
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    if (
        args.require_complete
        and summary["aggregate"]["fully_complete"]
        != summary["aggregate"]["target_count"]
    ):
        return 3
    if args.strict and summary["aggregate"]["targets_with_warnings"]:
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
