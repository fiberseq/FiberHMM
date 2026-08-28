#!/usr/bin/env python3
"""Run the audited targeted DddA/DddB strand-rescue matrix reproducibly.

This is deliberately a serial, attempt-preserving production driver.  It keeps
inference separate from BAM materialization so a completed report/proposal can
be reused by a future compact action-store annotator.  No attempt directory is
deleted or overwritten.  ``--resume`` skips a stage only after a cryptographic
validation receipt and every artifact named by that receipt still validate.

Examples
--------
Validate the complete 2-DddA + 14-DddB command matrix without running it::

    python scripts/run_targeted_strand_rescue.py --dry-run

Run or resume the complete matrix, one target at a time::

    python scripts/run_targeted_strand_rescue.py --resume

Exercise one inference target before committing to the matrix::

    python scripts/run_targeted_strand_rescue.py \
        --one-target dddb:sna --stage inference --resume

Pass a newly implemented inference option without changing this driver::

    python scripts/run_targeted_strand_rescue.py \
        --one-target ddda:napa \
        --inference-extra='--future-option {attempt_dir}/future-output.bin'

Leading-dash extras must be supplied with ``=`` as shown.  The placeholders
``{attempt_dir}``, ``{report}``, ``{proposal}``, and ``{output_dir}`` are
expanded without invoking a shell.  The current FiberHMM parser must recognize
all expanded options, including during ``--dry-run``.
"""

from __future__ import annotations

import argparse
import csv
import errno
import functools
import hashlib
import importlib
import json
import os
import shlex
import subprocess
import sys
import time
import traceback
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MATRIX = (
    REPO_ROOT
    / "consensus_validation_outputs"
    / "strand_rescue_targeted_full_20260716"
    / "manifests"
    / "targeted_sr_run_matrix.json"
)
GNU_TIME = Path("/usr/bin/time")
SAMTOOLS = "samtools"

INFERENCE_MODULE = "fiberhmm.cli.strand_rescue"
ANNOTATE_MODULE = "fiberhmm.cli.strand_rescue_annotate"
AUDIT_MODULE = "fiberhmm.cli.strand_rescue_audit"

INFERENCE_REQUIRED_ARTIFACTS = (
    "report.json",
    "proposals.tsv",
    "progress.jsonl",
    "time.txt",
    "stdout.log",
    "stderr.log",
)

FORBIDDEN_INFERENCE_EXTRA = {
    "-i",
    "--bam",
    "--preset",
    "--region",
    "--min-mapq",
    "--min-support",
    "--minimum-geometry-support",
    "--max-auto-sites",
    "--max-auto-nuc-sites",
    "--nuc-min-support",
    "--strand-min-source-support",
    "--molecule-collapse",
    "--forced-sites-only",
    "--forced-nuc-sites-only",
    "--skip-nuc-edge-refinement",
    "--max-reads",
    "--model",
    "--nuc-model",
    "--report-layout",
    "--diagnostics",
    "--proposal-tsv",
    "-o",
    "--output",
}
FORBIDDEN_ANNOTATE_EXTRA = {
    "--report",
    "-i",
    "--bam",
    "-o",
    "--output",
    "--output-dir",
    "--region",
    "--minimum-posterior",
    "--io-threads",
    "--allow-input-drift",
}
FORBIDDEN_AUDIT_EXTRA = {"-i", "--bam", "-o", "--output"}

PROPOSAL_COLUMNS = (
    "decision_id",
    "library_id",
    "read",
    "target_strand",
    "source_prior_strand",
    "current",
    "current_start",
    "current_end",
    "proposed",
)


@dataclass(frozen=True)
class Target:
    assay: str
    target_id: str
    display_name: str
    preset: str
    region: str
    bams: Tuple[str, ...]
    min_support: int
    min_mapq: int
    minimum_geometry_support: int
    molecule_collapse: str
    max_auto_sites: int
    max_auto_nuc_sites: int
    annotation_minimum_posterior: float

    @property
    def key(self) -> str:
        return f"{self.assay}:{self.target_id}"


@dataclass(frozen=True)
class Matrix:
    path: Path
    sha256: str
    dddb_manifest_path: Path
    dddb_manifest_sha256: str
    output_root: Path
    targets: Tuple[Target, ...]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _attempt_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    return f"{stamp}-p{os.getpid()}-{uuid.uuid4().hex[:8]}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _read_json(path: Path) -> Any:
    with path.open() as handle:
        return json.load(handle)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    """Atomically publish JSON beside its destination without using /tmp."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.partial"
    )
    with temporary.open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
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


def _resolve_declared_path(value: str, declaring_file: Path) -> Path:
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = declaring_file.parent / candidate
    return candidate.resolve()


def _parse_region(region: str) -> Tuple[str, int, int]:
    try:
        chrom, values = region.rsplit(":", 1)
        start_text, end_text = values.split("-", 1)
        start, end = int(start_text), int(end_text)
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid zero-based half-open region: {region!r}") from error
    if not chrom or start < 0 or end <= start:
        raise ValueError(f"invalid zero-based half-open region: {region!r}")
    return chrom, start, end


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_matrix(path: Path) -> Matrix:
    path = path.expanduser().resolve()
    raw = _read_json(path)
    _require(
        raw.get("schema") == "fiberhmm.validation.targeted_strand_rescue_matrix.v1",
        f"unsupported run matrix schema in {path}",
    )
    output_root = _resolve_declared_path(str(raw["output_root"]), path)
    dddb_path = _resolve_declared_path(str(raw["dddb_manifest"]), path)
    dddb = _read_json(dddb_path)
    _require(
        dddb.get("schema") == "fiberhmm.validation.dddb_wt_full_amplicons.v1",
        f"unsupported audited DddB manifest schema in {dddb_path}",
    )
    _require(
        dddb.get("status") == "audited_ready_for_targeted_full_strand_rescue",
        "DddB manifest is not audited-ready",
    )
    reference = dddb.get("reference_contract", {})
    _require(reference.get("all_bams_match_sq_header") is True, "DddB SQ mismatch")
    _require(reference.get("all_bams_coordinate_sorted") is True, "DddB BAM unsorted")
    _require(reference.get("all_bams_indexed") is True, "DddB BAM missing index")
    _require(reference.get("samtools_quickcheck_all_passed") is True, "DddB quickcheck failed")

    shared = raw["shared_parameters"]
    _require(int(shared["max_auto_sites"]) == 0, "TF discovery must be uncapped")
    _require(
        int(shared["max_auto_nuc_sites"]) == 0,
        "nucleosome discovery must be uncapped",
    )
    targets: List[Target] = []
    ddda = raw["ddda"]
    for record in ddda["targets"]:
        bams = tuple(
            str(_resolve_declared_path(str(candidate), path))
            for candidate in record["bams"]
        )
        _require(len(bams) > 0, f"DddA target {record['id']} has no BAM")
        _parse_region(str(record["region_cli_zero_based_half_open"]))
        targets.append(
            Target(
                assay="ddda",
                target_id=str(record["id"]),
                display_name=str(record["display_name"]),
                preset=str(ddda["preset"]),
                region=str(record["region_cli_zero_based_half_open"]),
                bams=bams,
                min_support=int(ddda["min_support"]),
                min_mapq=int(shared["min_mapq"]),
                minimum_geometry_support=int(shared["minimum_geometry_support"]),
                molecule_collapse=str(shared["molecule_collapse"]),
                max_auto_sites=int(shared["max_auto_sites"]),
                max_auto_nuc_sites=int(shared["max_auto_nuc_sites"]),
                annotation_minimum_posterior=float(
                    shared["annotation_minimum_posterior"]
                ),
            )
        )

    bams_by_id = {str(record["id"]): record for record in dddb["bams"]}
    _require(
        len(bams_by_id) == len(dddb["bams"]), "duplicate BAM IDs in DddB manifest"
    )
    dddb_min_support = int(dddb["run_recommendation"]["min_support"])
    _require(dddb_min_support == 10, "audited DddB production default must be 10")
    for record in dddb["targets"]:
        included_ids = [str(value) for value in record["included_bam_ids"]]
        _require(included_ids, f"DddB target {record['id']} has no compatible BAMs")
        _require(
            len(included_ids) == len(set(included_ids)),
            f"DddB target {record['id']} repeats a BAM",
        )
        unknown = [value for value in included_ids if value not in bams_by_id]
        _require(not unknown, f"DddB target {record['id']} has unknown BAM IDs: {unknown}")
        bams = tuple(
            str(_resolve_declared_path(str(bams_by_id[value]["path"]), dddb_path))
            for value in included_ids
        )
        region = str(record["region_cli_zero_based_half_open"])
        _parse_region(region)
        targets.append(
            Target(
                assay="dddb",
                target_id=str(record["id"]),
                display_name=str(record.get("display_name", record["id"])),
                preset=str(raw["dddb"]["preset"]),
                region=region,
                bams=bams,
                min_support=dddb_min_support,
                min_mapq=int(dddb["run_recommendation"]["min_mapq"]),
                minimum_geometry_support=int(
                    dddb["run_recommendation"]["minimum_geometry_support"]
                ),
                molecule_collapse=str(
                    dddb["run_recommendation"]["molecule_collapse"]
                ),
                max_auto_sites=int(shared["max_auto_sites"]),
                max_auto_nuc_sites=int(shared["max_auto_nuc_sites"]),
                annotation_minimum_posterior=float(
                    shared["annotation_minimum_posterior"]
                ),
            )
        )

    keys = [target.key for target in targets]
    _require(len(keys) == len(set(keys)), "duplicate target keys in run matrix")
    expected = raw["expected_matrix"]
    _require(
        sum(target.assay == "ddda" for target in targets)
        == int(expected["ddda_targets"])
        == 2,
        "run matrix must contain exactly two DddA targets",
    )
    _require(
        sum(target.assay == "dddb" for target in targets)
        == int(expected["dddb_targets"])
        == 14,
        "run matrix must contain exactly fourteen DddB targets",
    )
    _require(
        len(targets) == int(expected["total_targets"]) == 16,
        "run matrix must contain exactly sixteen targets",
    )
    return Matrix(
        path=path,
        sha256=_sha256(path),
        dddb_manifest_path=dddb_path,
        dddb_manifest_sha256=_sha256(dddb_path),
        output_root=output_root,
        targets=tuple(targets),
    )


def _find_index(bam: Path) -> Optional[Path]:
    candidates = (
        Path(str(bam) + ".bai"),
        bam.with_suffix(".bai"),
        Path(str(bam) + ".csi"),
        bam.with_suffix(".csi"),
    )
    return next((path.resolve() for path in candidates if path.is_file()), None)


def _file_fingerprint(path: Path, *, include_sha256: bool = False) -> Dict[str, Any]:
    stat = path.stat()
    result: Dict[str, Any] = {
        "path": str(path.resolve()),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }
    if include_sha256:
        result["sha256"] = _sha256(path)
    return result


def _input_fingerprints(target: Target) -> List[Dict[str, Any]]:
    records = []
    for value in target.bams:
        bam = Path(value)
        _require(bam.is_file(), f"missing input BAM: {bam}")
        index = _find_index(bam)
        _require(index is not None, f"missing BAM index: {bam}")
        records.append(
            {
                "bam": _file_fingerprint(bam),
                "index": _file_fingerprint(index),
            }
        )
    return records


def _quickcheck_inputs(targets: Sequence[Target]) -> Dict[str, Any]:
    bams = []
    seen = set()
    for target in targets:
        for value in target.bams:
            resolved = str(Path(value).resolve())
            if resolved not in seen:
                seen.add(resolved)
                bams.append(resolved)
    result = subprocess.run(
        [SAMTOOLS, "quickcheck", "-v", *bams],
        cwd=str(REPO_ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    _require(
        result.returncode == 0,
        "samtools quickcheck failed for targeted inputs: "
        + (result.stdout + result.stderr).strip(),
    )
    return {"bam_count": len(bams), "samtools_quickcheck_passed": True}


def _source_fingerprint(paths: Sequence[Path]) -> List[Dict[str, Any]]:
    records = []
    for path in paths:
        if not path.is_file():
            continue
        stat = path.stat()
        records.append(
            _cached_source_record(
                str(path.resolve()), int(stat.st_size), int(stat.st_mtime_ns)
            )
        )
    return records


@functools.lru_cache(maxsize=None)
def _cached_source_record(path: str, size_bytes: int, mtime_ns: int) -> Dict[str, Any]:
    return {
        "path": path,
        "size_bytes": size_bytes,
        "mtime_ns": mtime_ns,
        "sha256": _sha256(Path(path)),
    }


def _inference_source_fingerprint() -> List[Dict[str, Any]]:
    preset_models = (
        REPO_ROOT / "models" / "ddda_TF.json",
        REPO_ROOT / "models" / "ddda_nuc.json",
        REPO_ROOT / "models" / "dddb_nanopore.json",
    )
    return _source_fingerprint(
        (
            REPO_ROOT / "fiberhmm" / "cli" / "strand_rescue.py",
            REPO_ROOT / "fiberhmm" / "inference" / "strand_rescue.py",
            REPO_ROOT / "fiberhmm" / "inference" / "tf_recaller.py",
            REPO_ROOT / "fiberhmm" / "core" / "bam_reader.py",
            REPO_ROOT / "fiberhmm" / "core" / "model_io.py",
            REPO_ROOT / "fiberhmm" / "io" / "ma_tags.py",
            REPO_ROOT / "fiberhmm" / "__init__.py",
            *preset_models,
        )
    )


def _materialization_source_fingerprint() -> List[Dict[str, Any]]:
    return _source_fingerprint(
        (
            REPO_ROOT / "fiberhmm" / "cli" / "strand_rescue_annotate.py",
            REPO_ROOT / "fiberhmm" / "cli" / "strand_rescue_audit.py",
            REPO_ROOT / "fiberhmm" / "io" / "bam_header.py",
            REPO_ROOT / "fiberhmm" / "io" / "ma_tags.py",
            REPO_ROOT / "fiberhmm" / "cli" / "strand_rescue.py",
            REPO_ROOT / "fiberhmm" / "__init__.py",
        )
    )


def _expand_extra(values: Sequence[str], replacements: Mapping[str, str]) -> List[str]:
    tokens: List[str] = []
    for value in values:
        expanded = value
        for key, replacement in replacements.items():
            expanded = expanded.replace("{" + key + "}", replacement)
        tokens.extend(shlex.split(expanded))
    return tokens


def _validate_extra(tokens: Sequence[str], forbidden: Iterable[str], label: str) -> None:
    blocked = set(forbidden)
    found = []
    for token in tokens:
        flag = token.partition("=")[0]
        if flag in blocked:
            found.append(flag)
    if found:
        raise ValueError(
            f"{label} extras cannot override driver contract option(s): "
            + ", ".join(sorted(set(found)))
        )


def _module_args(command: Sequence[str], module: str) -> List[str]:
    marker = list(command).index(module)
    return list(command[marker + 1 :])


def _validate_with_current_parser(module_name: str, args: Sequence[str]) -> None:
    module = importlib.import_module(module_name)
    parser = module.build_parser()
    try:
        parser.parse_args(list(args))
    except SystemExit as error:
        raise ValueError(
            f"current {module_name} parser rejected generated arguments"
        ) from error


def _inference_cli_args(
    target: Target,
    report: Path,
    proposal: Path,
    extras: Sequence[str],
) -> List[str]:
    args: List[str] = []
    for bam in target.bams:
        args.extend(("-i", bam))
    args.extend(
        (
            "--preset",
            target.preset,
            "--region",
            target.region,
            "--min-mapq",
            str(target.min_mapq),
            "--min-support",
            str(target.min_support),
            "--nuc-min-support",
            str(target.min_support),
            "--strand-min-source-support",
            str(target.min_support),
            "--minimum-geometry-support",
            str(target.minimum_geometry_support),
            "--molecule-collapse",
            target.molecule_collapse,
            "--max-auto-sites",
            str(target.max_auto_sites),
            "--max-auto-nuc-sites",
            str(target.max_auto_nuc_sites),
            "--report-layout",
            "stream",
        )
    )
    args.extend(extras)
    args.extend(("--proposal-tsv", str(proposal), "-o", str(report)))
    return args


def _annotate_cli_args(
    report: Path,
    output_dir: Path,
    minimum_posterior: float,
    io_threads: int,
    extras: Sequence[str],
) -> List[str]:
    args = [
        "--report",
        str(report),
        "--output-dir",
        str(output_dir),
        "--minimum-posterior",
        str(minimum_posterior),
        "--io-threads",
        str(io_threads),
    ]
    args.extend(extras)
    return args


def _audit_cli_args(bams: Sequence[Path], output: Path, extras: Sequence[str]) -> List[str]:
    args: List[str] = []
    for bam in bams:
        args.extend(("-i", str(bam)))
    args.extend(extras)
    args.extend(("-o", str(output)))
    return args


def _timed_command(module: str, args: Sequence[str], time_path: Path) -> List[str]:
    return [
        str(GNU_TIME),
        "-v",
        "-o",
        str(time_path),
        sys.executable,
        "-m",
        module,
        *args,
    ]


def _inference_contract(
    matrix: Matrix,
    target: Target,
    raw_extras: Sequence[str],
    input_fingerprints: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    value = {
        "schema": "fiberhmm.validation.targeted_sr.inference_contract.v1",
        "matrix_sha256": matrix.sha256,
        "dddb_manifest_sha256": matrix.dddb_manifest_sha256,
        "target": asdict(target),
        "input_fingerprints": list(input_fingerprints),
        "inference_source_fingerprint": _inference_source_fingerprint(),
        "raw_inference_extras": list(raw_extras),
        "python": str(Path(sys.executable).resolve()),
    }
    value["contract_sha256"] = _json_sha256(value)
    return value


def _materialization_contract(
    matrix: Matrix,
    target: Target,
    inference_receipt: Mapping[str, Any],
    raw_annotate_extras: Sequence[str],
    raw_audit_extras: Sequence[str],
    input_fingerprints: Sequence[Mapping[str, Any]],
    io_threads: int,
) -> Dict[str, Any]:
    value = {
        "schema": "fiberhmm.validation.targeted_sr.materialization_contract.v1",
        "matrix_sha256": matrix.sha256,
        "target": asdict(target),
        "input_fingerprints": list(input_fingerprints),
        "inference_contract_sha256": inference_receipt["contract_sha256"],
        "inference_report_sha256": inference_receipt["report_summary"][
            "report_sha256"
        ],
        "materialization_source_fingerprint": _materialization_source_fingerprint(),
        "raw_annotate_extras": list(raw_annotate_extras),
        "raw_audit_extras": list(raw_audit_extras),
        "annotation_minimum_posterior": target.annotation_minimum_posterior,
        "io_threads": int(io_threads),
        "python": str(Path(sys.executable).resolve()),
    }
    value["contract_sha256"] = _json_sha256(value)
    return value


def _last_json_object(path: Path) -> Mapping[str, Any]:
    result: Optional[Mapping[str, Any]] = None
    with path.open(errors="replace") as handle:
        for line in handle:
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                result = value
    if result is None:
        raise ValueError(f"no JSON object in {path}")
    return result


def _validate_time_file(path: Path) -> None:
    text = path.read_text(errors="replace")
    _require("Command being timed:" in text, f"incomplete GNU time log: {path}")
    _require("Exit status: 0" in text, f"nonzero or missing exit status in {path}")


def _validate_progress(path: Path, bam_count: int) -> Dict[str, Any]:
    events = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid progress JSON at {path}:{line_number}"
                ) from error
            _require(
                value.get("schema") == "fiberhmm.performance.progress.v1",
                f"wrong progress schema at {path}:{line_number}",
            )
            events.append(value)
    stages = [value.get("stage", {}) for value in events]
    names = [str(stage.get("name")) for stage in stages]
    required = {
        "model_loading",
        "efficiency_calibration",
        "molecule_collapse",
        "tf_site_discovery",
        "nuc_site_discovery",
        "action_detail_collection",
        "action_stream_finalization_and_publication",
        "report_assembly",
        "report_validation",
        "report_serialization_and_write",
    }
    _require(required.issubset(set(names)), f"incomplete progress stages in {path}")
    _require(
        names.count("bam_evidence_loading") == bam_count,
        f"expected {bam_count} BAM loading stages in {path}",
    )
    tf_stage = next(stage for stage in stages if stage.get("name") == "tf_site_discovery")
    nuc_stage = next(
        stage for stage in stages if stage.get("name") == "nuc_site_discovery"
    )
    _require(
        int(tf_stage.get("details", {}).get("max_auto_sites", -1)) == 0,
        "TF site discovery was not uncapped",
    )
    _require(
        int(nuc_stage.get("details", {}).get("max_auto_sites", -1)) == 0,
        "nucleosome site discovery was not uncapped",
    )
    return {
        "event_count": len(events),
        "stage_names": names,
        "elapsed_wall_seconds": float(events[-1]["elapsed_wall_seconds"]),
    }


def _validate_proposals(path: Path, expected_rows: int) -> Dict[str, Any]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fields = tuple(reader.fieldnames or ())
        _require(
            set(PROPOSAL_COLUMNS).issubset(fields),
            f"proposal TSV has an invalid header: {path}",
        )
        rows = sum(1 for _record in reader)
    _require(
        rows == expected_rows,
        f"proposal TSV row count {rows} != report decisions {expected_rows}",
    )
    return {"row_count": rows, "columns": list(fields)}


def _validate_v5_action_sidecars(
    report: Mapping[str, Any], report_path: Path
) -> Tuple[Mapping[str, Any], List[Path]]:
    """Validate the bounded v5 manifest and immutable sidecar bytes.

    Full record/source validation happens again during materialization.  At the
    inference receipt boundary we validate the complete manifest, all declared
    BGZF/GZI sizes and hashes, standard BGZF EOF blocks, and standard GZI
    structure.  The resulting receipt then pins every sidecar byte.
    """
    from fiberhmm.cli.strand_rescue_annotate import (
        _validate_v5_action_storage,
        _validate_v5_sidecar_files,
    )

    streams = _validate_v5_action_storage(report, report_path)
    paths: List[Path] = []
    for stream in streams:
        _validate_v5_sidecar_files(stream)
        paths.extend((stream.bgzf_path, stream.gzi_path))
    storage = report["strand_rescue"]["action_storage"]
    return storage["totals"], paths


def _artifact_record(path: Path, root: Path) -> Dict[str, Any]:
    _require(path.is_file(), f"missing artifact: {path}")
    return {
        "relative_path": str(path.resolve().relative_to(root.resolve())),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _validate_receipt(
    attempt: Path,
    receipt: Mapping[str, Any],
    contract_sha256: str,
) -> bool:
    if receipt.get("validated") is not True:
        return False
    if receipt.get("contract_sha256") != contract_sha256:
        return False
    artifacts = receipt.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        return False
    try:
        for record in artifacts:
            path = (attempt / str(record["relative_path"])).resolve()
            path.relative_to(attempt.resolve())
            if not path.is_file() or path.stat().st_size != int(record["size_bytes"]):
                return False
            if _sha256(path) != str(record["sha256"]):
                return False
    except (KeyError, OSError, TypeError, ValueError):
        return False
    return True


def _validate_inference_attempt(
    attempt: Path,
    target: Target,
    contract: Mapping[str, Any],
    *,
    publish_receipt: bool,
) -> Mapping[str, Any]:
    metadata = _read_json(attempt / "attempt.json")
    _require(
        metadata.get("contract_sha256") == contract["contract_sha256"],
        "inference attempt contract mismatch",
    )
    report_path = attempt / "report.json"
    proposal_path = attempt / "proposals.tsv"
    progress_path = attempt / "progress.jsonl"
    time_path = attempt / "time.txt"
    for name in INFERENCE_REQUIRED_ARTIFACTS:
        _require((attempt / name).is_file(), f"missing inference artifact: {name}")
    report = _read_json(report_path)
    _require(report.get("schema") == "fiberhmm.strand_rescue.v5", "wrong report schema")
    _require(report.get("schema_version") == 5, "wrong report schema version")
    report_input = report.get("input", {})
    observed_bams = [str(Path(value).resolve()) for value in report_input.get("bams", [])]
    expected_bams = [str(Path(value).resolve()) for value in target.bams]
    _require(observed_bams == expected_bams, "report BAM cohort/order mismatch")
    _require(report_input.get("preset") == target.preset, "report preset mismatch")
    chrom, start, end = _parse_region(target.region)
    _require(
        report_input.get("focal_region") == [chrom, start, end],
        "report focal region mismatch",
    )
    parameters = report.get("parameters", {})
    expected_parameters = {
        "min_mapq": target.min_mapq,
        "min_support": target.min_support,
        "nuc_min_support": target.min_support,
        "strand_min_source_support": target.min_support,
        "minimum_geometry_support": target.minimum_geometry_support,
        "max_reads": 0,
        "max_auto_sites": 0,
        "max_auto_nuc_sites": 0,
        "molecule_collapse": target.molecule_collapse,
        "report_layout_requested": "stream",
        "report_layout_selected": "stream",
    }
    for key, value in expected_parameters.items():
        _require(parameters.get(key) == value, f"report parameter mismatch: {key}")
    guardrails = report.get("guardrails", {})
    _require(guardrails.get("external_assay_prior_used") is False, "external prior used")
    _require(
        guardrails.get("nucleosome_identity_or_cardinality_modified") is False,
        "nucleosome identity/cardinality guardrail failed",
    )
    rescue = report.get("strand_rescue", {})
    _require("decisions" not in rescue, "v5 report contains inline decisions")
    action_totals, action_paths = _validate_v5_action_sidecars(report, report_path)
    decision_count = int(action_totals["rescue_decisions"])
    harmonization_count = int(action_totals["tf_edge_updates"]) + int(
        action_totals["nuc_edge_updates"]
    )
    proposal_summary = _validate_proposals(proposal_path, decision_count)
    progress_summary = _validate_progress(progress_path, len(target.bams))
    _validate_time_file(time_path)
    stdout_summary = _last_json_object(attempt / "stdout.log")
    _require(
        str(Path(stdout_summary.get("output", "")).resolve()) == str(report_path.resolve()),
        "inference stdout summary points to another report",
    )
    _require(
        stdout_summary.get("report_layout") == "stream",
        "inference stdout did not select the streamed report",
    )
    _require(
        int(stdout_summary.get("decisions", -1)) == decision_count,
        "decision count mismatch",
    )

    required_paths = [attempt / name for name in INFERENCE_REQUIRED_ARTIFACTS]
    extra_paths = [
        path
        for path in attempt.rglob("*")
        if path.is_file()
        and path.name
        not in {
            "attempt.json",
            "validation_receipt.json",
        }
        and path not in required_paths
    ]
    _require(
        set(action_paths).issubset(set(extra_paths)),
        "declared v5 action sidecars are not retained beside the report",
    )
    artifacts = [
        _artifact_record(path, attempt)
        for path in sorted(set(required_paths + extra_paths))
    ]
    report_sha256 = next(
        record["sha256"]
        for record in artifacts
        if record["relative_path"] == "report.json"
    )
    receipt: Dict[str, Any] = {
        "schema": "fiberhmm.validation.targeted_sr.inference_receipt.v1",
        "validated": True,
        "validated_at": _utc_now(),
        "contract_sha256": contract["contract_sha256"],
        "target": target.key,
        "artifacts": artifacts,
        "report_summary": {
            "report_sha256": report_sha256,
            "n_raw_reads": int(report.get("n_raw_reads", 0)),
            "n_analyzed_molecules": int(report.get("n_analyzed_molecules", 0)),
            "report_layout": "stream",
            "decision_count": decision_count,
            "rescue_component_count": int(action_totals["rescue_components"]),
            "harmonization_count": harmonization_count,
            "tf_harmonization_count": int(action_totals["tf_edge_updates"]),
            "nuc_harmonization_count": int(action_totals["nuc_edge_updates"]),
            "action_record_count": int(action_totals["action_records"]),
            "applicable": bool(report.get("strand_rescue", {}).get("applicable")),
        },
        "proposal_summary": proposal_summary,
        "progress_summary": progress_summary,
    }
    if publish_receipt:
        _atomic_json(attempt / "validation_receipt.json", receipt)
    return receipt


def _receipt_or_full_inference_validation(
    attempt: Path,
    target: Target,
    contract: Mapping[str, Any],
) -> Optional[Mapping[str, Any]]:
    receipt_path = attempt / "validation_receipt.json"
    try:
        if receipt_path.is_file():
            receipt = _read_json(receipt_path)
            if _validate_receipt(attempt, receipt, str(contract["contract_sha256"])):
                return receipt
        return _validate_inference_attempt(
            attempt, target, contract, publish_receipt=True
        )
    except (csv.Error, json.JSONDecodeError, OSError, TypeError, ValueError):
        return None


def _attempt_candidates(stage_root: Path) -> List[Path]:
    attempts = stage_root / "attempts"
    if not attempts.is_dir():
        return []
    return sorted(
        (path for path in attempts.iterdir() if path.is_dir()),
        key=lambda path: path.name,
        reverse=True,
    )


def _find_valid_inference(
    stage_root: Path,
    target: Target,
    contract: Mapping[str, Any],
) -> Optional[Tuple[Path, Mapping[str, Any]]]:
    for attempt in _attempt_candidates(stage_root):
        value = _receipt_or_full_inference_validation(attempt, target, contract)
        if value is not None:
            return attempt, value
    return None


def _run_timed(
    command: Sequence[str],
    *,
    stdout_path: Path,
    stderr_path: Path,
    progress_path: Optional[Path] = None,
    progress_callback=None,
) -> int:
    with stdout_path.open("x") as stdout_handle, stderr_path.open(
        "x"
    ) as stderr_handle:
        progress_handle = progress_path.open("x") if progress_path else None
        try:
            process = subprocess.Popen(
                list(command),
                cwd=str(REPO_ROOT),
                stdout=stdout_handle,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
            assert process.stderr is not None
            try:
                for line in process.stderr:
                    stderr_handle.write(line)
                    stderr_handle.flush()
                    if progress_handle is None:
                        continue
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if event.get("schema") != "fiberhmm.performance.progress.v1":
                        continue
                    progress_handle.write(json.dumps(event, sort_keys=True) + "\n")
                    progress_handle.flush()
                    os.fsync(progress_handle.fileno())
                    if progress_callback is not None:
                        progress_callback(event)
                return process.wait()
            except BaseException:
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                raise
        finally:
            if progress_handle is not None:
                progress_handle.close()


def _stage_status(
    stage: str,
    state: str,
    target: Target,
    contract: Mapping[str, Any],
    **extra: Any,
) -> Dict[str, Any]:
    value: Dict[str, Any] = {
        "schema": "fiberhmm.validation.targeted_sr.stage_status.v1",
        "stage": stage,
        "state": state,
        "target": target.key,
        "contract_sha256": contract["contract_sha256"],
        "updated_at": _utc_now(),
    }
    value.update(extra)
    return value


def run_inference(
    matrix: Matrix,
    target: Target,
    stage_root: Path,
    contract: Mapping[str, Any],
    raw_extras: Sequence[str],
) -> Tuple[Path, Mapping[str, Any]]:
    attempt = stage_root / "attempts" / _attempt_id()
    attempt.mkdir(parents=True, exist_ok=False)
    report = attempt / "report.json"
    proposal = attempt / "proposals.tsv"
    replacements = {
        "attempt_dir": str(attempt.resolve()),
        "report": str(report.resolve()),
        "proposal": str(proposal.resolve()),
        "output_dir": str((attempt / "extra_outputs").resolve()),
    }
    extras = _expand_extra(raw_extras, replacements)
    _validate_extra(extras, FORBIDDEN_INFERENCE_EXTRA, "inference")
    args = _inference_cli_args(target, report, proposal, extras)
    _validate_with_current_parser(INFERENCE_MODULE, args)
    command = _timed_command(INFERENCE_MODULE, args, attempt / "time.txt")
    attempt_metadata = {
        **dict(contract),
        "attempt_id": attempt.name,
        "attempt_dir": str(attempt.resolve()),
        "created_at": _utc_now(),
        "command": command,
        "command_display": shlex.join(command),
    }
    _atomic_json(attempt / "attempt.json", attempt_metadata)
    status_path = stage_root / "status.json"
    started = time.monotonic()
    _atomic_json(
        status_path,
        _stage_status(
            "inference",
            "running",
            target,
            contract,
            attempt_dir=str(attempt.resolve()),
            command_display=shlex.join(command),
            started_at=_utc_now(),
        ),
    )

    def progress_callback(event: Mapping[str, Any]) -> None:
        stage = event.get("stage", {})
        _atomic_json(
            status_path,
            _stage_status(
                "inference",
                "running",
                target,
                contract,
                attempt_dir=str(attempt.resolve()),
                started_at=attempt_metadata["created_at"],
                elapsed_wall_seconds=float(event.get("elapsed_wall_seconds", 0.0)),
                last_completed_stage=stage.get("name"),
                last_stage_details=stage.get("details", {}),
            ),
        )

    try:
        returncode = _run_timed(
            command,
            stdout_path=attempt / "stdout.log",
            stderr_path=attempt / "stderr.log",
            progress_path=attempt / "progress.jsonl",
            progress_callback=progress_callback,
        )
        _require(returncode == 0, f"inference exited {returncode}")
        receipt = _validate_inference_attempt(
            attempt, target, contract, publish_receipt=True
        )
        _atomic_json(
            status_path,
            _stage_status(
                "inference",
                "complete",
                target,
                contract,
                attempt_dir=str(attempt.resolve()),
                receipt=str((attempt / "validation_receipt.json").resolve()),
                elapsed_driver_seconds=time.monotonic() - started,
                report_summary=receipt["report_summary"],
            ),
        )
        return attempt, receipt
    except BaseException as error:
        _atomic_json(
            status_path,
            _stage_status(
                "inference",
                "failed",
                target,
                contract,
                attempt_dir=str(attempt.resolve()),
                elapsed_driver_seconds=time.monotonic() - started,
                error=f"{type(error).__name__}: {error}",
                traceback=traceback.format_exc(),
            ),
        )
        raise


def _parse_annotation_outputs(stdout_path: Path) -> Mapping[str, Any]:
    value = _last_json_object(stdout_path)
    _require(isinstance(value.get("outputs"), list), "annotation summary lacks outputs")
    return value


def _validate_materialization_attempt(
    attempt: Path,
    target: Target,
    contract: Mapping[str, Any],
    *,
    publish_receipt: bool,
) -> Mapping[str, Any]:
    metadata = _read_json(attempt / "attempt.json")
    _require(
        metadata.get("contract_sha256") == contract["contract_sha256"],
        "materialization attempt contract mismatch",
    )
    required = (
        attempt / "annotate.stdout.log",
        attempt / "annotate.stderr.log",
        attempt / "annotate.time.txt",
        attempt / "audit.stdout.log",
        attempt / "audit.stderr.log",
        attempt / "audit.time.txt",
        attempt / "audit.json",
    )
    for path in required:
        _require(path.is_file(), f"missing materialization artifact: {path.name}")
    _validate_time_file(attempt / "annotate.time.txt")
    _validate_time_file(attempt / "audit.time.txt")
    annotation = _parse_annotation_outputs(attempt / "annotate.stdout.log")
    _require(len(annotation["outputs"]) == len(target.bams), "annotation BAM count mismatch")
    observed_inputs = [
        str(Path(record["input"]).resolve()) for record in annotation["outputs"]
    ]
    expected_inputs = [str(Path(value).resolve()) for value in target.bams]
    _require(observed_inputs == expected_inputs, "annotation input cohort/order mismatch")
    overlay_bams = [Path(record["output"]).resolve() for record in annotation["outputs"]]
    streamed = annotation.get("streamed_actions") is True
    expected_report = metadata.get("inference_receipt", {}).get(
        "report_summary", {}
    )
    streamed_totals = {
        "action_records": 0,
        "rescue_decisions": 0,
        "rescue_components": 0,
        "tf_edge_updates": 0,
        "nuc_edge_updates": 0,
    }
    for record, bam in zip(annotation["outputs"], overlay_bams):
        _require(bam.is_file(), f"missing overlay BAM: {bam}")
        index = Path(str(bam) + ".bai")
        _require(index.is_file(), f"missing overlay index: {index}")
        if streamed:
            action_stream = record.get("action_stream", {})
            validation = action_stream.get("validation", {})
            available = int(record.get("rescue_decisions_available", -1))
            materialized = int(record.get("rescue_decisions_materialized", -1))
            thresholded = int(record.get("rescue_decisions_thresholded", -1))
            _require(
                available == int(validation.get("rescue_decision_count", -2)),
                "streamed rescue count differs from validated action stream",
            )
            _require(
                materialized + thresholded == available,
                "streamed rescue actions were not fully accounted for",
            )
            if target.annotation_minimum_posterior == 0.0:
                _require(thresholded == 0, "zero-threshold annotation dropped rescues")
            for record_key, validation_key, total_key in (
                (
                    "rescue_components_materialized",
                    "rescue_component_count",
                    "rescue_components",
                ),
                (
                    "tf_edge_updates_materialized",
                    "tf_edge_update_count",
                    "tf_edge_updates",
                ),
                (
                    "nuc_edge_updates_materialized",
                    "nuc_edge_update_count",
                    "nuc_edge_updates",
                ),
            ):
                observed = int(record.get(record_key, -1))
                expected = int(validation.get(validation_key, -2))
                _require(
                    observed == expected,
                    f"{record_key} differs from validated action stream",
                )
                streamed_totals[total_key] += expected
            streamed_totals["action_records"] += int(
                validation.get("action_record_count", -1)
            )
            streamed_totals["rescue_decisions"] += available
        else:
            _require(
                int(record.get("decisions_unmatched", -1)) == 0,
                "unmatched TF rescue",
            )
            _require(
                int(record.get("harmonizations_unmatched", -1)) == 0,
                "unmatched edge harmonization",
            )
    if streamed:
        expected_streamed_totals = {
            "action_records": int(expected_report.get("action_record_count", -1)),
            "rescue_decisions": int(expected_report.get("decision_count", -1)),
            "rescue_components": int(
                expected_report.get("rescue_component_count", -1)
            ),
            "tf_edge_updates": int(
                expected_report.get("tf_harmonization_count", -1)
            ),
            "nuc_edge_updates": int(
                expected_report.get("nuc_harmonization_count", -1)
            ),
        }
        _require(
            streamed_totals == expected_streamed_totals,
            "materialized action totals differ from inference receipt",
        )
    audit = _read_json(attempt / "audit.json")
    _require(audit.get("schema") == "fiberhmm.strand_rescue.audit.v4", "wrong audit schema")
    _require(audit.get("valid") is True, "strand-rescue BAM audit failed")
    _require(int(audit.get("file_count", -1)) == len(overlay_bams), "audit file count mismatch")
    audited_paths = [str(Path(record["path"]).resolve()) for record in audit.get("files", [])]
    _require(audited_paths == [str(path) for path in overlay_bams], "audit path/order mismatch")
    _require(all(record.get("valid") is True for record in audit["files"]), "invalid audit file")

    artifact_paths = list(required)
    for bam in overlay_bams:
        artifact_paths.extend((bam, Path(str(bam) + ".bai")))
    extra_paths = [
        path
        for path in attempt.rglob("*")
        if path.is_file()
        and path.name not in {"attempt.json", "validation_receipt.json"}
        and path not in artifact_paths
    ]
    artifacts = [
        _artifact_record(path, attempt)
        for path in sorted(set(artifact_paths + extra_paths))
    ]
    receipt: Dict[str, Any] = {
        "schema": "fiberhmm.validation.targeted_sr.materialization_receipt.v1",
        "validated": True,
        "validated_at": _utc_now(),
        "contract_sha256": contract["contract_sha256"],
        "target": target.key,
        "artifacts": artifacts,
        "overlay_bams": [str(path) for path in overlay_bams],
        "audit_summary": {
            "valid": True,
            "file_count": int(audit["file_count"]),
            "totals": audit.get("totals", {}),
        },
    }
    if publish_receipt:
        _atomic_json(attempt / "validation_receipt.json", receipt)
    return receipt


def _receipt_or_full_materialization_validation(
    attempt: Path,
    target: Target,
    contract: Mapping[str, Any],
) -> Optional[Mapping[str, Any]]:
    receipt_path = attempt / "validation_receipt.json"
    try:
        if receipt_path.is_file():
            receipt = _read_json(receipt_path)
            if _validate_receipt(attempt, receipt, str(contract["contract_sha256"])):
                return receipt
        return _validate_materialization_attempt(
            attempt, target, contract, publish_receipt=True
        )
    except (json.JSONDecodeError, OSError, TypeError, ValueError):
        return None


def _find_valid_materialization(
    stage_root: Path,
    target: Target,
    contract: Mapping[str, Any],
) -> Optional[Tuple[Path, Mapping[str, Any]]]:
    for attempt in _attempt_candidates(stage_root):
        value = _receipt_or_full_materialization_validation(attempt, target, contract)
        if value is not None:
            return attempt, value
    return None


def run_materialization(
    target: Target,
    stage_root: Path,
    inference_attempt: Path,
    inference_receipt: Mapping[str, Any],
    contract: Mapping[str, Any],
    raw_annotate_extras: Sequence[str],
    raw_audit_extras: Sequence[str],
    io_threads: int,
) -> Tuple[Path, Mapping[str, Any]]:
    attempt = stage_root / "attempts" / _attempt_id()
    attempt.mkdir(parents=True, exist_ok=False)
    output_dir = attempt / "overlays"
    report = inference_attempt / "report.json"
    replacements = {
        "attempt_dir": str(attempt.resolve()),
        "report": str(report.resolve()),
        "proposal": str((inference_attempt / "proposals.tsv").resolve()),
        "output_dir": str(output_dir.resolve()),
    }
    annotate_extras = _expand_extra(raw_annotate_extras, replacements)
    audit_extras = _expand_extra(raw_audit_extras, replacements)
    _validate_extra(annotate_extras, FORBIDDEN_ANNOTATE_EXTRA, "annotation")
    _validate_extra(audit_extras, FORBIDDEN_AUDIT_EXTRA, "audit")
    annotate_args = _annotate_cli_args(
        report,
        output_dir,
        target.annotation_minimum_posterior,
        io_threads,
        annotate_extras,
    )
    _validate_with_current_parser(ANNOTATE_MODULE, annotate_args)
    annotate_command = _timed_command(
        ANNOTATE_MODULE, annotate_args, attempt / "annotate.time.txt"
    )
    metadata = {
        **dict(contract),
        "attempt_id": attempt.name,
        "attempt_dir": str(attempt.resolve()),
        "created_at": _utc_now(),
        "inference_attempt": str(inference_attempt.resolve()),
        "inference_receipt": dict(inference_receipt),
        "annotate_command": annotate_command,
        "annotate_command_display": shlex.join(annotate_command),
    }
    _atomic_json(attempt / "attempt.json", metadata)
    status_path = stage_root / "status.json"
    started = time.monotonic()
    _atomic_json(
        status_path,
        _stage_status(
            "materialization",
            "running_annotation",
            target,
            contract,
            attempt_dir=str(attempt.resolve()),
            started_at=metadata["created_at"],
            inference_attempt=str(inference_attempt.resolve()),
        ),
    )
    try:
        returncode = _run_timed(
            annotate_command,
            stdout_path=attempt / "annotate.stdout.log",
            stderr_path=attempt / "annotate.stderr.log",
        )
        _require(returncode == 0, f"annotation exited {returncode}")
        annotation = _parse_annotation_outputs(attempt / "annotate.stdout.log")
        overlay_bams = [Path(record["output"]).resolve() for record in annotation["outputs"]]
        _require(len(overlay_bams) == len(target.bams), "annotation BAM count mismatch")
        audit_args = _audit_cli_args(
            overlay_bams, attempt / "audit.json", audit_extras
        )
        _validate_with_current_parser(AUDIT_MODULE, audit_args)
        audit_command = _timed_command(
            AUDIT_MODULE, audit_args, attempt / "audit.time.txt"
        )
        metadata["audit_command"] = audit_command
        metadata["audit_command_display"] = shlex.join(audit_command)
        _atomic_json(attempt / "attempt.json", metadata)
        _atomic_json(
            status_path,
            _stage_status(
                "materialization",
                "running_audit",
                target,
                contract,
                attempt_dir=str(attempt.resolve()),
                started_at=metadata["created_at"],
                elapsed_driver_seconds=time.monotonic() - started,
                overlay_bams=[str(path) for path in overlay_bams],
            ),
        )
        returncode = _run_timed(
            audit_command,
            stdout_path=attempt / "audit.stdout.log",
            stderr_path=attempt / "audit.stderr.log",
        )
        _require(returncode == 0, f"audit exited {returncode}")
        receipt = _validate_materialization_attempt(
            attempt, target, contract, publish_receipt=True
        )
        _atomic_json(
            status_path,
            _stage_status(
                "materialization",
                "complete",
                target,
                contract,
                attempt_dir=str(attempt.resolve()),
                receipt=str((attempt / "validation_receipt.json").resolve()),
                elapsed_driver_seconds=time.monotonic() - started,
                audit_summary=receipt["audit_summary"],
            ),
        )
        return attempt, receipt
    except BaseException as error:
        _atomic_json(
            status_path,
            _stage_status(
                "materialization",
                "failed",
                target,
                contract,
                attempt_dir=str(attempt.resolve()),
                elapsed_driver_seconds=time.monotonic() - started,
                error=f"{type(error).__name__}: {error}",
                traceback=traceback.format_exc(),
            ),
        )
        raise


def _select_targets(matrix: Matrix, args: argparse.Namespace) -> List[Target]:
    targets = list(matrix.targets)
    if args.assay:
        targets = [target for target in targets if target.assay in set(args.assay)]
    selectors = list(args.target or [])
    if args.one_target:
        _require(not selectors, "--one-target cannot be combined with --target")
        selectors = [args.one_target]
    if not selectors:
        return targets
    selected: List[Target] = []
    for selector in selectors:
        matches = [
            target
            for target in targets
            if target.key == selector or target.target_id == selector
        ]
        _require(len(matches) == 1, f"target selector must match exactly once: {selector}")
        if matches[0] not in selected:
            selected.append(matches[0])
    return selected


def _dry_run_record(
    matrix: Matrix,
    target: Target,
    args: argparse.Namespace,
    input_fingerprints: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    placeholder = matrix.output_root / "DRY_RUN_ATTEMPT"
    report = placeholder / "report.json"
    proposal = placeholder / "proposals.tsv"
    replacements = {
        "attempt_dir": str(placeholder),
        "report": str(report),
        "proposal": str(proposal),
        "output_dir": str(placeholder / "overlays"),
    }
    inference_extras = _expand_extra(args.inference_extra, replacements)
    annotate_extras = _expand_extra(args.annotate_extra, replacements)
    audit_extras = _expand_extra(args.audit_extra, replacements)
    _validate_extra(inference_extras, FORBIDDEN_INFERENCE_EXTRA, "inference")
    _validate_extra(annotate_extras, FORBIDDEN_ANNOTATE_EXTRA, "annotation")
    _validate_extra(audit_extras, FORBIDDEN_AUDIT_EXTRA, "audit")
    inference_args = _inference_cli_args(target, report, proposal, inference_extras)
    _validate_with_current_parser(INFERENCE_MODULE, inference_args)
    annotate_args = _annotate_cli_args(
        report,
        placeholder / "overlays",
        target.annotation_minimum_posterior,
        args.io_threads,
        annotate_extras,
    )
    _validate_with_current_parser(ANNOTATE_MODULE, annotate_args)
    dry_overlay_bams = [
        placeholder / "overlays" / f"input-{index}.strand-rescue.bam"
        for index, _bam in enumerate(target.bams)
    ]
    audit_args = _audit_cli_args(dry_overlay_bams, placeholder / "audit.json", audit_extras)
    _validate_with_current_parser(AUDIT_MODULE, audit_args)
    contract = _inference_contract(
        matrix, target, args.inference_extra, input_fingerprints
    )
    return {
        "target": target.key,
        "region_zero_based_half_open": target.region,
        "bam_count": len(target.bams),
        "bams": list(target.bams),
        "min_support": target.min_support,
        "max_auto_sites": target.max_auto_sites,
        "max_auto_nuc_sites": target.max_auto_nuc_sites,
        "inference_contract_sha256": contract["contract_sha256"],
        "inference_command": shlex.join(
            _timed_command(
                INFERENCE_MODULE, inference_args, placeholder / "time.txt"
            )
        ),
        "annotation_command": shlex.join(
            _timed_command(
                ANNOTATE_MODULE, annotate_args, placeholder / "annotate.time.txt"
            )
        ),
        "audit_command": shlex.join(
            _timed_command(AUDIT_MODULE, audit_args, placeholder / "audit.time.txt")
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument(
        "--stage",
        choices=("all", "inference", "materialization"),
        default="all",
        help="Run inference, materialization, or both as separately resumable stages",
    )
    parser.add_argument(
        "--one-target",
        metavar="ASSAY:ID",
        help="Run exactly one target (a unique bare ID is also accepted)",
    )
    parser.add_argument(
        "--target",
        action="append",
        help="Select a target; repeat for an explicit subset",
    )
    parser.add_argument(
        "--assay",
        action="append",
        choices=("ddda", "dddb"),
        help="Restrict the matrix by assay",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip a stage only when a matching attempt and receipt revalidate",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate manifests, inputs, and generated args with current parsers",
    )
    parser.add_argument(
        "--dry-run-output",
        type=Path,
        help="Atomically retain the validated dry-run manifest (requires --dry-run)",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop after the first failed target instead of preserving it and continuing",
    )
    parser.add_argument("--io-threads", type=int, default=4)
    parser.add_argument(
        "--inference-extra",
        action="append",
        default=[],
        metavar="ARGS",
        help="Additional current inference CLI args; repeatable, parsed with shlex",
    )
    parser.add_argument(
        "--annotate-extra",
        action="append",
        default=[],
        metavar="ARGS",
        help="Additional current annotation CLI args; repeatable, parsed with shlex",
    )
    parser.add_argument(
        "--audit-extra",
        action="append",
        default=[],
        metavar="ARGS",
        help="Additional current audit CLI args; repeatable, parsed with shlex",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.io_threads < 1:
        parser.error("--io-threads must be positive")
    if args.dry_run_output is not None and not args.dry_run:
        parser.error("--dry-run-output requires --dry-run")
    if not GNU_TIME.is_file():
        parser.error(f"GNU time is required: {GNU_TIME}")
    try:
        matrix = load_matrix(args.matrix)
        targets = _select_targets(matrix, args)
        _require(targets, "selection contains no targets")
        fingerprints = {
            target.key: _input_fingerprints(target) for target in targets
        }
        input_preflight = _quickcheck_inputs(targets)
    except (
        json.JSONDecodeError,
        KeyError,
        OSError,
        subprocess.SubprocessError,
        TypeError,
        ValueError,
    ) as error:
        parser.error(str(error))

    if args.dry_run:
        try:
            records = [
                _dry_run_record(matrix, target, args, fingerprints[target.key])
                for target in targets
            ]
        except (ImportError, OSError, SystemExit, TypeError, ValueError) as error:
            parser.error(str(error))
        result = {
            "schema": "fiberhmm.validation.targeted_sr.dry_run.v1",
            "valid": True,
            "matrix": str(matrix.path),
            "matrix_sha256": matrix.sha256,
            "dddb_manifest": str(matrix.dddb_manifest_path),
            "dddb_manifest_sha256": matrix.dddb_manifest_sha256,
            "input_preflight": input_preflight,
            "target_count": len(records),
            "targets": records,
        }
        if args.dry_run_output is not None:
            _atomic_json(args.dry_run_output.expanduser().resolve(), result)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    run_id = _attempt_id()
    run_dir = matrix.output_root / "driver_runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    run_status_path = run_dir / "status.json"
    run_status: Dict[str, Any] = {
        "schema": "fiberhmm.validation.targeted_sr.driver_run.v1",
        "run_id": run_id,
        "state": "running",
        "started_at": _utc_now(),
        "matrix": str(matrix.path),
        "matrix_sha256": matrix.sha256,
        "dddb_manifest": str(matrix.dddb_manifest_path),
        "dddb_manifest_sha256": matrix.dddb_manifest_sha256,
        "input_preflight": input_preflight,
        "requested_stage": args.stage,
        "resume": bool(args.resume),
        "selected_targets": [target.key for target in targets],
        "results": {},
    }
    _atomic_json(run_status_path, run_status)
    failures = 0
    started = time.monotonic()
    for target in targets:
        target_result: Dict[str, Any] = {
            "target": target.key,
            "state": "running",
            "started_at": _utc_now(),
        }
        run_status["results"][target.key] = target_result
        _atomic_json(run_status_path, run_status)
        target_root = matrix.output_root / "runs" / target.assay / target.target_id
        inference_root = target_root / "inference"
        materialization_root = target_root / "materialization"
        try:
            inference_contract = _inference_contract(
                matrix, target, args.inference_extra, fingerprints[target.key]
            )
            inference_value: Optional[Tuple[Path, Mapping[str, Any]]] = None
            if args.resume or args.stage == "materialization":
                inference_value = _find_valid_inference(
                    inference_root, target, inference_contract
                )
            if args.stage in {"all", "inference"}:
                if inference_value is not None:
                    target_result["inference"] = {
                        "state": "resume_skipped_valid",
                        "attempt_dir": str(inference_value[0].resolve()),
                    }
                    _atomic_json(
                        inference_root / "status.json",
                        _stage_status(
                            "inference",
                            "complete",
                            target,
                            inference_contract,
                            attempt_dir=str(inference_value[0].resolve()),
                            receipt=str(
                                (inference_value[0] / "validation_receipt.json").resolve()
                            ),
                            resume_revalidated_at=_utc_now(),
                            report_summary=inference_value[1].get("report_summary", {}),
                        ),
                    )
                else:
                    inference_value = run_inference(
                        matrix,
                        target,
                        inference_root,
                        inference_contract,
                        args.inference_extra,
                    )
                    target_result["inference"] = {
                        "state": "complete",
                        "attempt_dir": str(inference_value[0].resolve()),
                    }
                _atomic_json(run_status_path, run_status)
            if args.stage == "materialization" and inference_value is None:
                raise ValueError(
                    "materialization requested without a matching validated inference; "
                    "run --stage inference first with the same matrix/options"
                )
            if args.stage in {"all", "materialization"}:
                assert inference_value is not None
                inference_attempt, inference_receipt = inference_value
                materialization_contract = _materialization_contract(
                    matrix,
                    target,
                    inference_receipt,
                    args.annotate_extra,
                    args.audit_extra,
                    fingerprints[target.key],
                    args.io_threads,
                )
                materialization_value = None
                if args.resume:
                    materialization_value = _find_valid_materialization(
                        materialization_root, target, materialization_contract
                    )
                if materialization_value is not None:
                    target_result["materialization"] = {
                        "state": "resume_skipped_valid",
                        "attempt_dir": str(materialization_value[0].resolve()),
                    }
                    _atomic_json(
                        materialization_root / "status.json",
                        _stage_status(
                            "materialization",
                            "complete",
                            target,
                            materialization_contract,
                            attempt_dir=str(materialization_value[0].resolve()),
                            receipt=str(
                                (
                                    materialization_value[0]
                                    / "validation_receipt.json"
                                ).resolve()
                            ),
                            resume_revalidated_at=_utc_now(),
                            audit_summary=materialization_value[1].get(
                                "audit_summary", {}
                            ),
                        ),
                    )
                else:
                    materialization_value = run_materialization(
                        target,
                        materialization_root,
                        inference_attempt,
                        inference_receipt,
                        materialization_contract,
                        args.annotate_extra,
                        args.audit_extra,
                        args.io_threads,
                    )
                    target_result["materialization"] = {
                        "state": "complete",
                        "attempt_dir": str(materialization_value[0].resolve()),
                    }
            target_result["state"] = "complete"
            target_result["completed_at"] = _utc_now()
        except BaseException as error:
            failures += 1
            target_result["state"] = "failed"
            target_result["failed_at"] = _utc_now()
            target_result["error"] = f"{type(error).__name__}: {error}"
            target_result["traceback"] = traceback.format_exc()
            _atomic_json(run_status_path, run_status)
            print(f"{target.key}: {target_result['error']}", file=sys.stderr, flush=True)
            if args.fail_fast or isinstance(error, (KeyboardInterrupt, SystemExit)):
                break
        _atomic_json(run_status_path, run_status)

    complete = sum(
        result.get("state") == "complete" for result in run_status["results"].values()
    )
    run_status.update(
        {
            "state": "complete" if failures == 0 else "complete_with_failures",
            "completed_at": _utc_now(),
            "elapsed_driver_seconds": time.monotonic() - started,
            "targets_complete": complete,
            "targets_failed": failures,
            "targets_not_run": len(targets) - complete - failures,
        }
    )
    _atomic_json(run_status_path, run_status)
    _atomic_json(matrix.output_root / "driver_latest_status.json", run_status)
    print(json.dumps(run_status, indent=2, sort_keys=True))
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
