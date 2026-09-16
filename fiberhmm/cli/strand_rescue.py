#!/usr/bin/env python3
"""Normalize focal TF and nucleosome calls between physical/read strands.

The command learns canonical footprint geometries from every ordinary TF call,
uses opposite-strand occupancy to rescue weak-but-positive TFs only from MSPs,
and proposes shared edge normalization for accepted TF and nucleosome calls.
Nucleosome identity and cardinality are never changed. It is report-only:
input BAMs and baseline calls are never modified. Use
``fiberhmm-strand-rescue-annotate`` to create optional regional ``nuc_sr`` and
``tf_sr`` visualization layers.
"""
from __future__ import annotations

import argparse
import atexit
import csv
import errno
import hashlib
import heapq
import io
import itertools
import json
import math
import os
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import (
    Callable,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    TextIO,
    Tuple,
    TypeVar,
)

import pysam

try:  # ``resource`` is POSIX-only; timing remains available elsewhere.
    import resource as _resource
except ImportError:  # pragma: no cover - exercised only on non-POSIX hosts
    _resource = None

from fiberhmm import __version__ as FIBERHMM_VERSION
from fiberhmm.core.model_io import load_model_with_metadata
from fiberhmm.inference.strand_rescue import (
    DEFAULT_ACCESSIBLE_SITE_GAP,
    MIN_MAPPED_ANNOTATION_FRACTION,
    PRESETS,
    ReadEvidence,
    _deduplicate_decisions,
    analyze_strand_rescue,
    assign_global_efficiency_steps,
    build_edge_action_candidate,
    build_rescue_action_candidate,
    calibrate_cohort_efficiency,
    collapse_amplified_reads,
    discover_sites,
    discover_edge_sites,
    load_region_evidence,
    merge_forced_sites,
    finalize_alignment_actions,
    resolve_joint_edge_topology,
    resolve_rescue_edge_topology,
    resolve_resource_path,
    shift_site_templates,
)
from fiberhmm.inference.tf_recaller import build_llr_tables


def _current_rss_bytes() -> Optional[int]:
    """Return current process RSS on Linux without adding a dependency."""
    try:
        resident_pages = int(Path("/proc/self/statm").read_text().split()[1])
        return resident_pages * int(os.sysconf("SC_PAGE_SIZE"))
    except (IndexError, OSError, TypeError, ValueError):
        return None


def _process_peak_rss_bytes() -> Optional[int]:
    """Return the process RSS high-water mark with platform-correct units."""
    if _resource is None:
        return None
    try:
        value = int(_resource.getrusage(_resource.RUSAGE_SELF).ru_maxrss)
    except (OSError, TypeError, ValueError):
        return None
    # Linux and the other BSDs report KiB; macOS reports bytes.
    return value if sys.platform == "darwin" else value * 1024


class _StageProfiler:
    """Low-overhead, inference-neutral wall-time and RSS provenance."""

    def __init__(self, progress_stream=None) -> None:
        now = time.perf_counter()
        self._started = now
        self._last_mark = now
        self._stages: list[dict] = []
        self._progress_stream = (
            sys.stderr if progress_stream is None else progress_stream
        )

    def mark(self, name: str, details: Optional[Mapping[str, object]] = None) -> None:
        now = time.perf_counter()
        stage = {
            "name": str(name),
            "wall_seconds": float(max(0.0, now - self._last_mark)),
        }
        current_rss = _current_rss_bytes()
        peak_rss = _process_peak_rss_bytes()
        if current_rss is not None:
            stage["rss_bytes_after"] = int(current_rss)
        if peak_rss is not None:
            # This is deliberately labelled cumulative: ru_maxrss cannot be reset.
            stage["process_peak_rss_bytes_after"] = int(peak_rss)
        if details:
            stage["details"] = dict(details)
        self._stages.append(stage)
        self._last_mark = now
        progress = {
            "schema": "fiberhmm.performance.progress.v1",
            "event": "stage_complete",
            "elapsed_wall_seconds": float(max(0.0, now - self._started)),
            "stage": dict(stage),
        }
        try:
            print(
                json.dumps(progress, sort_keys=True),
                file=self._progress_stream,
                flush=True,
            )
        except (BrokenPipeError, OSError, TypeError, ValueError):
            # Progress is diagnostic and must never alter inference or output.
            pass

    def inference_callback(self, name: str, details: Mapping[str, object]) -> None:
        self.mark(name, details)

    def snapshot(self, *, scope: str) -> dict:
        peak_rss = _process_peak_rss_bytes()
        result = {
            "schema": "fiberhmm.performance.v1",
            "clock": "time.perf_counter",
            "scope": scope,
            "total_wall_seconds": float(
                max(0.0, time.perf_counter() - self._started)
            ),
            "stage_memory_semantics": (
                "rss_bytes_after_is_instantaneous_when_available; "
                "process_peak_rss_bytes_after_is_the_cumulative_process_high_water"
            ),
            "stages": [dict(stage) for stage in self._stages],
        }
        if peak_rss is not None:
            result["process_peak_rss_bytes"] = int(peak_rss)
        return result


def parse_region(value: str) -> Tuple[str, int, int]:
    try:
        chrom, coordinates = value.rsplit(":", 1)
        start_text, end_text = coordinates.replace(",", "").split("-", 1)
        start, end = int(start_text), int(end_text)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError(
            "region must be CHROM:START-END"
        ) from error
    if not chrom or start < 0 or end <= start:
        raise argparse.ArgumentTypeError(
            "region must satisfy 0 <= START < END"
        )
    return chrom, start, end


def parse_site_interval(value: str) -> Tuple[int, int]:
    try:
        start_text, end_text = value.split("-", 1)
        start, end = int(start_text), int(end_text)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError(
            "site interval must be START-END"
        ) from error
    if start < 0 or end <= start:
        raise argparse.ArgumentTypeError(
            "site interval must satisfy 0 <= START < END"
        )
    return start, end


def _file_metadata(path: str) -> dict:
    resolved = Path(path).expanduser().resolve()
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _resolved_path(path: object) -> Path:
    return Path(str(path)).expanduser().resolve()


def validate_generation_paths(
    bams: Sequence[str],
    models: Sequence[str],
    output: Path,
    proposal_tsv: Optional[Path],
) -> Tuple[List[Path], set[Path]]:
    """Reject canonical input duplication and every destructive destination."""
    resolved_bams = [_resolved_path(path) for path in bams]
    if len(resolved_bams) != len(set(resolved_bams)):
        raise ValueError(
            "input BAM paths must be canonically unique; repeated BAMs would "
            "double-count one library and make the v5 ordinal manifest invalid"
        )
    protected = set(resolved_bams)
    protected.update(_resolved_path(path) for path in models)
    for path in resolved_bams:
        # htslib accepts both ``sample.bam.bai``/``sample.bam.csi`` and the
        # suffix-replaced ``sample.bai``/``sample.csi`` conventions. Protect
        # every valid index name even when an index is not currently present.
        protected.update(
            {
                _resolved_path(f"{path}.bai"),
                _resolved_path(path.with_suffix(".bai")),
                _resolved_path(f"{path}.csi"),
                _resolved_path(path.with_suffix(".csi")),
            }
        )
    destinations = {"report": _resolved_path(output)}
    if proposal_tsv is not None:
        destinations["proposal TSV"] = _resolved_path(proposal_tsv)
    if len(set(destinations.values())) != len(destinations):
        raise ValueError("report output and proposal TSV must be different paths")
    for label, destination in destinations.items():
        if destination in protected:
            raise ValueError(
                f"{label} would overwrite an input BAM, BAM index, or model: "
                f"{destination}"
            )
    return resolved_bams, protected | set(destinations.values())


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _nonfinite_json_paths(value: object, path: str = "report") -> list[str]:
    if isinstance(value, float):
        return [] if math.isfinite(value) else [path]
    if isinstance(value, dict):
        return [
            result
            for key, child in value.items()
            for result in _nonfinite_json_paths(child, f"{path}.{key}")
        ]
    if isinstance(value, (list, tuple)):
        return [
            result
            for index, child in enumerate(value)
            for result in _nonfinite_json_paths(child, f"{path}[{index}]")
        ]
    return []


_PUBLISH_PERMISSION_ATTEMPTS = 10
_PUBLISH_PERMISSION_INITIAL_DELAY_SECONDS = 0.05
_PUBLISH_PERMISSION_MAX_DELAY_SECONDS = 0.75
_PermissionResult = TypeVar("_PermissionResult")


def _retry_transient_permission(
    operation: Callable[[], _PermissionResult],
) -> _PermissionResult:
    """Retry only bounded EACCES/EPERM failures from destination-local I/O."""
    delay = _PUBLISH_PERMISSION_INITIAL_DELAY_SECONDS
    for attempt in range(_PUBLISH_PERMISSION_ATTEMPTS):
        try:
            return operation()
        except PermissionError as error:
            if (
                error.errno not in {errno.EACCES, errno.EPERM}
                or attempt + 1 == _PUBLISH_PERMISSION_ATTEMPTS
            ):
                raise
            time.sleep(delay)
            delay = min(_PUBLISH_PERMISSION_MAX_DELAY_SECONDS, delay * 2.0)
    raise AssertionError("unreachable permission-retry state")


def _mkstemp_with_permission_retry(
    *, prefix: str, suffix: str, directory: Path
) -> Tuple[int, str]:
    """Create one destination-local stage despite a short sharing lock."""
    return _retry_transient_permission(
        lambda: tempfile.mkstemp(
            prefix=prefix,
            suffix=suffix,
            dir=str(directory),
        )
    )


def _open_text_stage_with_permission_retry(
    path: Path, *, newline: str
) -> TextIO:
    """Open an owned text stage, retrying only transient sharing denials."""
    return _retry_transient_permission(
        lambda: path.open("w", encoding="utf-8", newline=newline)
    )


def _open_bgzf_stage_with_permission_retry(
    path: Path, index_path: Path
):
    """Open owned BGZF/GZI stages through a bounded sharing-lock retry."""
    return _retry_transient_permission(
        lambda: pysam.BGZFile(str(path), "wb", index=str(index_path))
    )


def _replace_with_permission_retry(source: Path, destination: Path) -> None:
    _retry_transient_permission(lambda: os.replace(source, destination))


def _link_with_permission_retry(source: Path, destination: Path) -> None:
    _retry_transient_permission(lambda: os.link(source, destination))


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = _mkstemp_with_permission_retry(
        prefix=f".{path.name}.", suffix=".tmp", directory=path.parent
    )
    try:
        with os.fdopen(descriptor, "w") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        _replace_with_permission_retry(Path(temporary_name), path)
        _fsync_directory(path.parent)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


INLINE_DETAIL_LIMIT = 100_000
INLINE_JSON_LIMIT_BYTES = 256 * 1024 * 1024
V5_ACTION_STREAM_SCHEMA = "fiberhmm.strand_rescue.actions.v1"
V5_ACTION_LAYOUT = "per_input_bgzf_jsonl_v1"
_BGZF_EOF = bytes.fromhex(
    "1f8b08040000000000ff0600424302001b0003000000000000000000"
)
_PROPOSAL_COLUMNS = (
    "decision_id",
    "library_id",
    "read",
    "target_strand",
    "source_prior_strand",
    "current",
    "current_start",
    "current_end",
    "proposed",
    "sr_hypothesis_probability",
    "baseline_hypothesis_probability",
    "supported_tf_probability_vs_accessible",
    "configuration_posterior",
    "molecule_probability",
    "population_probability",
    "source_support_reliability",
    "log_bf_vs_current",
    "proposal_tier",
    "proposed_site_intervals",
    "proposed_site_edge_confidence",
)


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        + b"\n"
    )


def _proposal_tsv_row(decision: Mapping[str, object]) -> dict:
    start, end = decision["current_interval"]
    return {
        "decision_id": decision["decision_id"],
        "library_id": decision["library_id"],
        "read": decision["read"],
        "target_strand": decision["target_strand"],
        "source_prior_strand": decision["source_prior_strand"],
        "current": decision["current"],
        "current_start": start,
        "current_end": end,
        "proposed": decision["proposed"],
        "sr_hypothesis_probability": decision[
            "sr_hypothesis_probability"
        ],
        "baseline_hypothesis_probability": decision[
            "baseline_hypothesis_probability"
        ],
        "supported_tf_probability_vs_accessible": decision[
            "supported_tf_probability_vs_accessible"
        ],
        "configuration_posterior": decision[
            "best_configuration_posterior_given_tf"
        ],
        "molecule_probability": decision["molecule_probability"],
        "population_probability": decision["population_probability"],
        "source_support_reliability": decision["source_support_reliability"],
        "log_bf_vs_current": decision["log_bf_vs_current"],
        "proposal_tier": decision["proposal_tier"],
        "proposed_site_intervals": json.dumps(
            decision["proposed_site_intervals"], separators=(",", ":")
        ),
        "proposed_site_edge_confidence": json.dumps(
            decision["proposed_site_edge_confidence"], separators=(",", ":")
        ),
    }


def _write_tsv(report: dict, path: Path) -> None:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=_PROPOSAL_COLUMNS, delimiter="\t")
    writer.writeheader()
    for decision in report["strand_rescue"]["decisions"]:
        writer.writerow(_proposal_tsv_row(decision))
    _atomic_write(path, buffer.getvalue())


class _InlineReportLimitExceeded(ValueError):
    pass


def select_report_layout(
    requested: str,
    detail_count: int,
    estimated_json_bytes: int,
    *,
    detail_limit: int = INLINE_DETAIL_LIMIT,
    json_limit_bytes: int = INLINE_JSON_LIMIT_BYTES,
) -> str:
    """Choose v4 inline or v5 stream with inclusive, testable limits."""
    if requested not in {"auto", "inline", "stream"}:
        raise ValueError(f"unsupported report layout: {requested}")
    if requested == "stream":
        return "stream"
    over_limit = (
        int(detail_count) >= int(detail_limit)
        or int(estimated_json_bytes) >= int(json_limit_bytes)
    )
    if requested == "inline" and over_limit:
        raise _InlineReportLimitExceeded(
            "explicit inline report crossed the bounded v4 limit "
            f"({detail_count:,} details, {estimated_json_bytes:,} estimated "
            "compact JSON bytes); rerun with --report-layout stream"
        )
    return "stream" if over_limit else "inline"


class _StagingRegistry:
    """Track destination-local transient files and remove only unpublished ones."""

    def __init__(self) -> None:
        self.paths: set[Path] = set()
        self._registered = True
        atexit.register(self.cleanup)

    def create(self, directory: Path, label: str, suffix: str) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        descriptor, raw_path = _mkstemp_with_permission_retry(
            prefix=f".{label}.{uuid.uuid4().hex}.",
            suffix=suffix,
            directory=directory,
        )
        os.close(descriptor)
        path = Path(raw_path)
        self.paths.add(path)
        return path

    def discard(self, path: Optional[Path]) -> None:
        if path is None:
            return
        Path(path).unlink(missing_ok=True)
        self.paths.discard(Path(path))

    def published(self, staged: Path) -> None:
        self.paths.discard(Path(staged))

    def cleanup(self) -> None:
        for path in tuple(self.paths):
            path.unlink(missing_ok=True)
            self.paths.discard(path)
        if self._registered:
            atexit.unregister(self.cleanup)
            self._registered = False


class _GenerationSink:
    """Bounded callback sink for inline details, compact candidates, and TSV."""

    _KINDS = ("rescue", "tf", "nuc")

    def __init__(
        self,
        output: Path,
        requested_layout: str,
        registry: _StagingRegistry,
        proposal_tsv: Optional[Path] = None,
    ) -> None:
        self.output = output
        self.requested_layout = requested_layout
        self.registry = registry
        self.detail_count = 0
        self.estimated_json_bytes = 0
        self.spilled = requested_layout == "stream"
        self.candidate_rejections: Dict[str, int] = {}
        self.candidate_paths = {
            kind: registry.create(output.parent, f"{output.name}.{kind}", ".jsonl")
            for kind in self._KINDS
        }
        self.candidate_handles: Dict[str, TextIO] = {
            kind: _open_text_stage_with_permission_retry(path, newline="\n")
            for kind, path in self.candidate_paths.items()
        }
        self.raw_paths: Dict[str, Path] = {}
        self.raw_handles: Dict[str, TextIO] = {}
        self.proposal_path = proposal_tsv
        self.proposal_stage: Optional[Path] = None
        self.proposal_handle: Optional[TextIO] = None
        self.proposal_writer = None
        if proposal_tsv is not None:
            self.proposal_stage = registry.create(
                proposal_tsv.parent, proposal_tsv.name, ".tsv"
            )
            self.proposal_handle = _open_text_stage_with_permission_retry(
                self.proposal_stage, newline=""
            )
            self.proposal_writer = csv.DictWriter(
                self.proposal_handle,
                fieldnames=_PROPOSAL_COLUMNS,
                delimiter="\t",
            )
            self.proposal_writer.writeheader()

    def _write_candidate(self, kind: str, candidate: Mapping[str, object]) -> None:
        self.candidate_handles[kind].write(
            _canonical_json_bytes(candidate).decode("utf-8")
        )

    def _increment_rejection(self, reason: str) -> None:
        self.candidate_rejections[reason] = (
            self.candidate_rejections.get(reason, 0) + 1
        )

    def _close_raw(self, *, discard: bool) -> None:
        for handle in self.raw_handles.values():
            handle.close()
        self.raw_handles.clear()
        if discard:
            for path in tuple(self.raw_paths.values()):
                self.registry.discard(path)
            self.raw_paths.clear()

    def _retain_detail(self, kind: str, value: Mapping[str, object]) -> None:
        self.detail_count += 1
        if self.spilled:
            return
        raw = _canonical_json_bytes(value)
        self.estimated_json_bytes += len(raw)
        path = self.raw_paths.get(kind)
        if path is None:
            path = self.registry.create(
                self.output.parent, f"{self.output.name}.{kind}.inline", ".jsonl"
            )
            self.raw_paths[kind] = path
            self.raw_handles[kind] = _open_text_stage_with_permission_retry(
                path, newline="\n"
            )
        self.raw_handles[kind].write(raw.decode("utf-8"))
        selected = select_report_layout(
            self.requested_layout,
            self.detail_count,
            self.estimated_json_bytes,
        )
        if selected == "stream":
            self.spilled = True
            self._close_raw(discard=True)

    def decision_callback(self, read: ReadEvidence, decision: dict) -> None:
        self._retain_detail("rescue", decision)
        if self.proposal_writer is not None:
            self.proposal_writer.writerow(_proposal_tsv_row(decision))
        candidate, rejection = build_rescue_action_candidate(read, decision)
        if candidate is None:
            self._increment_rejection(str(rejection))
        else:
            self._write_candidate("rescue", candidate)

    def edge_callback(self, read: ReadEvidence, decision: dict) -> None:
        call_type = str(decision.get("call_type"))
        if call_type not in {"tf", "nuc"}:
            raise ValueError(f"unsupported edge callback call type: {call_type}")
        self._retain_detail(call_type, decision)
        if decision.get("status") != "edge_update":
            return
        candidate, rejection = build_edge_action_candidate(read, decision)
        if candidate is None:
            self._increment_rejection(f"{call_type}_{rejection}")
        else:
            self._write_candidate(call_type, candidate)

    def finish_callbacks(self) -> None:
        for handle in self.candidate_handles.values():
            handle.flush()
            os.fsync(handle.fileno())
            handle.close()
        self.candidate_handles.clear()
        self._close_raw(discard=False)
        if self.proposal_handle is not None:
            self.proposal_handle.flush()
            os.fsync(self.proposal_handle.fileno())
            self.proposal_handle.close()
            self.proposal_handle = None

    def final_layout(self) -> str:
        return select_report_layout(
            self.requested_layout,
            self.detail_count,
            self.estimated_json_bytes,
        )

    def load_inline(self, kind: str) -> List[dict]:
        if self.final_layout() != "inline":
            raise ValueError("cannot load inline details after streaming spill")
        path = self.raw_paths.get(kind)
        if path is None:
            return []
        with path.open("r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]

    def publish_proposal(self) -> None:
        if self.proposal_path is None or self.proposal_stage is None:
            return
        self.proposal_path.parent.mkdir(parents=True, exist_ok=True)
        _replace_with_permission_retry(self.proposal_stage, self.proposal_path)
        self.registry.published(self.proposal_stage)
        self.proposal_stage = None
        _fsync_directory(self.proposal_path.parent)


def _iter_candidate_spool(path: Path) -> Iterator[dict]:
    with path.open("r", encoding="utf-8") as handle:
        previous: Optional[Tuple[int, int]] = None
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            key = (int(value["input_index"]), int(value["ordinal"]))
            if previous is not None and key < previous:
                raise ValueError(
                    f"candidate spool {path.name}:{line_number} is out of order"
                )
            previous = key
            yield value


def _merged_candidate_groups(
    paths: Sequence[Path],
) -> Iterator[Tuple[Tuple[int, int], List[dict]]]:
    merged = heapq.merge(
        *(_iter_candidate_spool(path) for path in paths),
        key=lambda value: (int(value["input_index"]), int(value["ordinal"])),
    )
    for key, values in itertools.groupby(
        merged,
        key=lambda value: (int(value["input_index"]), int(value["ordinal"])),
    ):
        yield key, list(values)


def _hydrate_inline_details(strand_rescue: dict, sink: _GenerationSink) -> None:
    decisions = sink.load_inline("rescue")
    decisions.sort(
        key=lambda value: (
            value["library_id"] or "",
            value["read"],
            value["current_interval"],
            value["proposed"],
        )
    )
    decisions = _deduplicate_decisions(decisions)
    strand_rescue["decisions"] = decisions
    counts = strand_rescue.get("counts", {})
    counts["from_msp"] = len(decisions)
    for tier in ("strong", "review", "retain_current"):
        counts[tier] = sum(
            decision["proposal_tier"] == tier for decision in decisions
        )
    counts["templates_truncated"] = sum(
        bool(decision["templates_truncated"]) for decision in decisions
    )
    for call_type in ("tf", "nuc"):
        values = sink.load_inline(call_type)
        values.sort(
            key=lambda value: (
                value["library_id"] or "",
                value["read"],
                value["current_interval"],
            )
        )
        strand_rescue["edge_refinement"][call_type]["harmonizations"] = (
            _deduplicate_decisions(values)
        )
    resolve_joint_edge_topology(
        strand_rescue["edge_refinement"]["tf"],
        strand_rescue["edge_refinement"]["nuc"],
    )
    resolve_rescue_edge_topology(
        decisions,
        strand_rescue["edge_refinement"]["tf"],
        strand_rescue["edge_refinement"]["nuc"],
    )


def _fsync_path(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:  # pragma: no cover - platform/filesystem dependent
        return
    try:
        os.fsync(descriptor)
    except OSError:  # pragma: no cover - platform/filesystem dependent
        pass
    finally:
        os.close(descriptor)


def _validate_standard_gzi(path: Path, compressed_size: int) -> None:
    with path.open("rb") as handle:
        raw_count = handle.read(8)
        if len(raw_count) != 8:
            raise ValueError("generated GZI is shorter than its entry count")
        count = int.from_bytes(raw_count, "little", signed=False)
        previous_compressed = 0
        previous_uncompressed = 0
        for _index in range(count):
            entry = handle.read(16)
            if len(entry) != 16:
                raise ValueError("generated GZI has a truncated offset entry")
            compressed = int.from_bytes(entry[:8], "little")
            uncompressed = int.from_bytes(entry[8:], "little")
            if (
                compressed <= previous_compressed
                or compressed >= compressed_size
                or uncompressed <= previous_uncompressed
            ):
                raise ValueError("generated GZI offsets are not strictly increasing")
            previous_compressed = compressed
            previous_uncompressed = uncompressed
        if handle.read(1):
            raise ValueError("generated GZI has trailing bytes")


def _iter_bgzf_lines(handle, chunk_size: int = 64 * 1024) -> Iterator[bytes]:
    """Yield newline-preserving BGZF records with one-record bounded buffering."""
    buffered = b""
    while True:
        chunk = handle.read(chunk_size)
        if not chunk:
            break
        buffered += chunk
        while True:
            newline = buffered.find(b"\n")
            if newline < 0:
                break
            yield buffered[: newline + 1]
            buffered = buffered[newline + 1 :]
    if buffered:
        yield buffered


class _V5ActionWriter:
    """Write, validate, and publish one input's immutable action sidecar."""

    def __init__(
        self,
        output: Path,
        input_index: int,
        loaded_region: Sequence[object],
        registry: _StagingRegistry,
    ) -> None:
        self.output = output
        self.input_index = int(input_index)
        self.input_id = f"input{self.input_index:04d}"
        self.loaded_region = list(loaded_region)
        self.registry = registry
        self.stage = registry.create(
            output.parent, f"{output.name}.{self.input_id}.actions", ".jsonl.bgz"
        )
        self.gzi_stage = registry.create(
            output.parent, f"{output.name}.{self.input_id}.actions", ".jsonl.bgz.gzi"
        )
        self.handle = _open_bgzf_stage_with_permission_retry(
            self.stage, self.gzi_stage
        )
        self.digest = hashlib.sha256()
        self.uncompressed_size = 0
        self.previous_ordinal: Optional[int] = None
        self.counts = {
            "action_record_count": 0,
            "rescue_decision_count": 0,
            "rescue_component_count": 0,
            "tf_edge_update_count": 0,
            "nuc_edge_update_count": 0,
        }
        self.first_ordinal: Optional[int] = None
        self.last_ordinal: Optional[int] = None
        self._write(
            {
                "kind": "header",
                "schema": V5_ACTION_STREAM_SCHEMA,
                "input_index": self.input_index,
                "input_id": self.input_id,
                "loaded_region": self.loaded_region,
                "ordinal_base": 0,
            }
        )

    def _write(self, value: Mapping[str, object]) -> None:
        raw = _canonical_json_bytes(value)
        self.handle.write(raw)
        self.digest.update(raw)
        self.uncompressed_size += len(raw)

    def write_actions(
        self,
        ordinal: int,
        read: ReadEvidence,
        rescues: Sequence[Mapping[str, object]],
        edges: Sequence[Mapping[str, object]],
    ) -> None:
        ordinal = int(ordinal)
        if self.previous_ordinal is not None and ordinal <= self.previous_ordinal:
            raise ValueError("v5 action ordinals are not strictly increasing")
        if not rescues and not edges:
            raise ValueError("empty v5 action rows are forbidden")
        if not read.record_sha256 or len(read.record_sha256) != 64:
            raise ValueError(
                f"read {read.name!r} lacks its exact BAM-record SHA-256"
            )
        row = {
            "kind": "actions",
            "ordinal": ordinal,
            "read": str(read.name),
            "record_sha256": str(read.record_sha256),
            "rescues": [dict(value) for value in rescues],
            "edge_updates": [dict(value) for value in edges],
        }
        self._write(row)
        self.previous_ordinal = ordinal
        if self.first_ordinal is None:
            self.first_ordinal = ordinal
        self.last_ordinal = ordinal
        self.counts["action_record_count"] += 1
        self.counts["rescue_decision_count"] += len(rescues)
        self.counts["rescue_component_count"] += sum(
            len(value["components"]) for value in rescues
        )
        self.counts["tf_edge_update_count"] += sum(
            value["call_type"] == "tf" for value in edges
        )
        self.counts["nuc_edge_update_count"] += sum(
            value["call_type"] == "nuc" for value in edges
        )

    def _validate_decompressed(
        self, fetch_record_count: int, expected_jsonl_sha256: str
    ) -> None:
        digest = hashlib.sha256()
        size = 0
        observed_counts = {key: 0 for key in self.counts}
        first: Optional[int] = None
        last: Optional[int] = None
        with pysam.BGZFile(str(self.stage), "rb") as handle:
            lines = _iter_bgzf_lines(handle)
            raw = next(lines, None)
            if raw is None:
                raise ValueError("generated action stream lacks its header")
            if not raw.endswith(b"\n"):
                raise ValueError("generated action-stream row is not newline terminated")
            digest.update(raw)
            size += len(raw)
            header = json.loads(raw)
            if header != {
                "kind": "header",
                "schema": V5_ACTION_STREAM_SCHEMA,
                "input_index": self.input_index,
                "input_id": self.input_id,
                "loaded_region": self.loaded_region,
                "ordinal_base": 0,
            }:
                raise ValueError("generated action-stream header failed validation")
            previous: Optional[int] = None
            trailer = None
            for raw in lines:
                if not raw.endswith(b"\n"):
                    raise ValueError(
                        "generated action-stream row is not newline terminated"
                    )
                digest.update(raw)
                size += len(raw)
                row = json.loads(raw)
                if row.get("kind") == "trailer":
                    trailer = row
                    if next(lines, None) is not None:
                        raise ValueError(
                            "generated action stream has content after its trailer"
                        )
                    break
                if set(row) != {
                    "kind",
                    "ordinal",
                    "read",
                    "record_sha256",
                    "rescues",
                    "edge_updates",
                } or row.get("kind") != "actions":
                    raise ValueError("generated action-stream row failed validation")
                ordinal = int(row["ordinal"])
                if previous is not None and ordinal <= previous:
                    raise ValueError("generated action ordinals are not increasing")
                if ordinal < 0 or ordinal >= fetch_record_count:
                    raise ValueError("generated action ordinal is outside regional fetch")
                if not row["rescues"] and not row["edge_updates"]:
                    raise ValueError("generated action stream contains an empty row")
                previous = ordinal
                first = ordinal if first is None else first
                last = ordinal
                observed_counts["action_record_count"] += 1
                observed_counts["rescue_decision_count"] += len(row["rescues"])
                observed_counts["rescue_component_count"] += sum(
                    len(value["components"]) for value in row["rescues"]
                )
                observed_counts["tf_edge_update_count"] += sum(
                    value["call_type"] == "tf" for value in row["edge_updates"]
                )
                observed_counts["nuc_edge_update_count"] += sum(
                    value["call_type"] == "nuc" for value in row["edge_updates"]
                )
        if trailer is None:
            raise ValueError("generated action stream lacks its trailer")
        expected_trailer = {
            "kind": "trailer",
            "fetch_record_count": int(fetch_record_count),
            **self.counts,
            "first_action_ordinal": self.first_ordinal,
            "last_action_ordinal": self.last_ordinal,
        }
        # Stable stream order places first/last before payload counts; parsed
        # object equality deliberately ignores key insertion order.
        if trailer != expected_trailer:
            raise ValueError("generated action-stream trailer failed validation")
        if observed_counts != self.counts or (
            first,
            last,
        ) != (self.first_ordinal, self.last_ordinal):
            raise ValueError("generated action-stream observations differ from trailer")
        if size != self.uncompressed_size or digest.hexdigest() != expected_jsonl_sha256:
            raise ValueError("generated decompressed action-stream checksum mismatch")

    def finish(
        self,
        fetch_record_count: int,
        *,
        protected_paths: Sequence[Path] = (),
    ) -> dict:
        trailer = {
            "kind": "trailer",
            "fetch_record_count": int(fetch_record_count),
            "action_record_count": self.counts["action_record_count"],
            "first_action_ordinal": self.first_ordinal,
            "last_action_ordinal": self.last_ordinal,
            "rescue_decision_count": self.counts["rescue_decision_count"],
            "rescue_component_count": self.counts["rescue_component_count"],
            "tf_edge_update_count": self.counts["tf_edge_update_count"],
            "nuc_edge_update_count": self.counts["nuc_edge_update_count"],
        }
        self._write(trailer)
        self.handle.close()
        _fsync_path(self.stage)
        _fsync_path(self.gzi_stage)
        jsonl_sha256 = self.digest.hexdigest()
        compressed_size = self.stage.stat().st_size
        with self.stage.open("rb") as handle:
            handle.seek(-len(_BGZF_EOF), os.SEEK_END)
            if handle.read() != _BGZF_EOF:
                raise ValueError("generated action stream has no standard BGZF EOF")
        _validate_standard_gzi(self.gzi_stage, compressed_size)
        self._validate_decompressed(fetch_record_count, jsonl_sha256)
        bgzf_sha256 = _sha256_file(str(self.stage))
        gzi_sha256 = _sha256_file(str(self.gzi_stage))
        final_name = (
            f"{self.output.stem}.{self.input_id}.sr-actions."
            f"{jsonl_sha256[:12]}.jsonl.bgz"
        )
        final = self.output.parent / final_name
        final_gzi = Path(f"{final}.gzi")
        protected = {_resolved_path(path) for path in protected_paths}
        for destination in (final, final_gzi):
            if _resolved_path(destination) in protected:
                raise ValueError(
                    "derived action sidecar would collide with an input or "
                    f"requested output: {destination}"
                )
        for staged, destination, expected_sha256 in (
            (self.stage, final, bgzf_sha256),
            (self.gzi_stage, final_gzi, gzi_sha256),
        ):
            if destination.exists():
                if _sha256_file(str(destination)) != expected_sha256:
                    raise ValueError(
                        "content-addressed action sidecar exists with different bytes: "
                        f"{destination}"
                    )
                self.registry.discard(staged)
            else:
                try:
                    _link_with_permission_retry(staged, destination)
                except FileExistsError:
                    if _sha256_file(str(destination)) != expected_sha256:
                        raise ValueError(
                            "concurrently published action sidecar has different bytes: "
                            f"{destination}"
                        )
                self.registry.discard(staged)
        _fsync_directory(self.output.parent)
        return {
            "input_index": self.input_index,
            "input_id": self.input_id,
            "path": final.name,
            "gzi_path": final_gzi.name,
            "loaded_region": self.loaded_region,
            "fetch_record_count": int(fetch_record_count),
            **self.counts,
            "first_action_ordinal": self.first_ordinal,
            "last_action_ordinal": self.last_ordinal,
            "compressed_size_bytes": compressed_size,
            "uncompressed_size_bytes": self.uncompressed_size,
            "bgzf_sha256": bgzf_sha256,
            "jsonl_sha256": jsonl_sha256,
            "gzi_size_bytes": final_gzi.stat().st_size,
            "gzi_sha256": gzi_sha256,
        }


def _build_v5_action_storage(
    sink: _GenerationSink,
    target_reads: Sequence[ReadEvidence],
    input_fetch_counts: Sequence[int],
    loaded_region: Sequence[object],
    registry: _StagingRegistry,
    protected_paths: Sequence[Path] = (),
) -> Tuple[dict, Dict[str, int]]:
    writers = [
        _V5ActionWriter(
            sink.output,
            input_index,
            loaded_region,
            registry,
        )
        for input_index in range(len(input_fetch_counts))
    ]
    ordered_reads = iter(target_reads)
    current_read = next(ordered_reads, None)
    previous_read_key: Optional[Tuple[int, int]] = None
    topology_diagnostics: Dict[str, int] = {}
    for key, candidates in _merged_candidate_groups(
        [sink.candidate_paths[kind] for kind in _GenerationSink._KINDS]
    ):
        while current_read is not None:
            if current_read.input_record_ordinal is None:
                raise ValueError("eligible read lacks its regional input ordinal")
            read_key = (
                int(current_read.input_index),
                int(current_read.input_record_ordinal),
            )
            if previous_read_key is not None and read_key <= previous_read_key:
                raise ValueError("eligible target reads are not in input/ordinal order")
            if read_key >= key:
                break
            previous_read_key = read_key
            current_read = next(ordered_reads, None)
        if current_read is None:
            raise ValueError(f"action candidate {key} has no eligible input record")
        read_key = (
            int(current_read.input_index),
            int(current_read.input_record_ordinal),
        )
        if read_key != key:
            raise ValueError(
                f"action candidate {key} differs from next eligible record {read_key}"
            )
        rescues, edges, diagnostics = finalize_alignment_actions(
            current_read,
            [value for value in candidates if value["kind"] == "rescue_candidate"],
            [value for value in candidates if value["kind"] == "edge_candidate"],
        )
        for name, value in diagnostics.items():
            topology_diagnostics[name] = topology_diagnostics.get(name, 0) + int(value)
        raw_edges_by_token = {
            str(value["token"]): value
            for value in candidates
            if value["kind"] == "edge_candidate"
        }
        for edge in edges:
            call_type = str(edge["call_type"])
            aggregate = raw_edges_by_token[str(edge["token"])].get(
                "_aggregate", {}
            )
            for metric in ("prior_only", "chemistry_opposed", "extreme_edge_shift"):
                if aggregate.get(metric, False):
                    name = f"{call_type}_edge_accepted_{metric}"
                    topology_diagnostics[name] = topology_diagnostics.get(name, 0) + 1
        if rescues or edges:
            writers[key[0]].write_actions(key[1], current_read, rescues, edges)
        previous_read_key = read_key
        current_read = next(ordered_reads, None)
    streams = [
        writer.finish(
            input_fetch_counts[writer.input_index],
            protected_paths=protected_paths,
        )
        for writer in writers
    ]
    totals = {
        "fetch_records": sum(value["fetch_record_count"] for value in streams),
        "action_records": sum(value["action_record_count"] for value in streams),
        "rescue_decisions": sum(value["rescue_decision_count"] for value in streams),
        "rescue_components": sum(value["rescue_component_count"] for value in streams),
        "tf_edge_updates": sum(value["tf_edge_update_count"] for value in streams),
        "nuc_edge_updates": sum(value["nuc_edge_update_count"] for value in streams),
    }
    return (
        {
            "layout": V5_ACTION_LAYOUT,
            "stream_schema": V5_ACTION_STREAM_SCHEMA,
            "quality_encoding": "uint8_round_255_times_unit_probability",
            "coordinate_frame": "molecular_zero_based_start_length",
            "streams": streams,
            "totals": totals,
        },
        topology_diagnostics,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-i",
        "--bam",
        action="append",
        required=True,
        help=(
            "Post-TF/post-nuc BAM; repeat only to pool shards or compatible "
            "timepoints from one inference cohort"
        ),
    )
    parser.add_argument("--preset", choices=sorted(PRESETS), required=True)
    parser.add_argument("--region", required=True, type=parse_region)
    parser.add_argument("--model")
    parser.add_argument(
        "--nuc-model",
        help="Protected/accessibility model used only for nucleosome edge evidence",
    )
    parser.add_argument("--prob-threshold", type=int)
    parser.add_argument("--control-flank", type=int, default=2000)
    parser.add_argument("--min-mapq", type=int, default=20)
    parser.add_argument("--min-support", type=int, default=10)
    parser.add_argument("--minimum-geometry-support", type=int, default=3)
    parser.add_argument("--source-boundary-margin", type=int, default=10)
    parser.add_argument("--center-radius", type=int, default=10)
    parser.add_argument("--peak-distance", type=int, default=15)
    parser.add_argument(
        "--tf-edge-compatibility",
        type=int,
        default=12,
        help=(
            "Maximum within-family diameter in bp for each TF boundary; "
            "co-centered calls outside this bound form distinct TF models"
        ),
    )
    parser.add_argument("--max-boundary-mad", type=float, default=12.0)
    parser.add_argument("--min-local-enrichment", type=float, default=2.0)
    parser.add_argument("--local-background-radius", type=int, default=250)
    parser.add_argument(
        "--max-auto-sites",
        type=int,
        default=0,
        help="Optional TF-family cap after scoring; 0 keeps the exhaustive locus map",
    )
    parser.add_argument(
        "--site",
        action="append",
        type=parse_site_interval,
        default=[],
        help=(
            "Externally seed one zero-based half-open START-END site; repeatable. "
            "Support and canonical edges are recomputed from ordinary cohort calls."
        ),
    )
    parser.add_argument(
        "--forced-sites-only",
        action="store_true",
        help="Analyze only --site geometries rather than unioning automatic sites",
    )
    parser.add_argument(
        "--nuc-min-support",
        type=int,
        help="Required ordinary nuc calls on one strand (default: --min-support)",
    )
    parser.add_argument("--nuc-center-radius", type=int, default=25)
    parser.add_argument("--nuc-source-boundary-margin", type=int, default=20)
    parser.add_argument("--nuc-edge-assignment-radius", type=int, default=48)
    parser.add_argument("--nuc-max-boundary-mad", type=float, default=24.0)
    parser.add_argument(
        "--max-auto-nuc-sites",
        type=int,
        default=0,
        help="Optional nucleosome-family cap after scoring; 0 keeps all families",
    )
    parser.add_argument(
        "--nuc-site",
        action="append",
        type=parse_site_interval,
        default=[],
        help=(
            "Seed one zero-based half-open nucleosome population; repeatable. "
            "Edges are relearned from ordinary nuc calls and no length ceiling applies."
        ),
    )
    parser.add_argument(
        "--forced-nuc-sites-only",
        action="store_true",
        help="Refine only --nuc-site populations rather than automatic nuc sites",
    )
    parser.add_argument(
        "--skip-nuc-edge-refinement",
        dest="skip_nuc_edge_refinement",
        action="store_true",
        default=True,
        help=(
            "Do not run independent one-for-one nucleosome edge normalization "
            "(default; retained for command-line compatibility)"
        ),
    )
    parser.add_argument(
        "--independent-nuc-edge-refinement",
        dest="skip_nuc_edge_refinement",
        action="store_false",
        help=(
            "Experimental population nucleosome-edge normalization. "
            "This is not the TF-conditioned consensus-nuc reconciliation stage."
        ),
    )
    parser.add_argument(
        "--strand-min-source-support",
        type=int,
        help="Required source-strand ordinary TF calls (default: --min-support)",
    )
    parser.add_argument(
        "--strand-min-source-enrichment",
        type=float,
        default=1.5,
        help="Minimum source-strand focal enrichment over local background",
    )
    parser.add_argument("--strong-posterior", type=float, default=0.95)
    parser.add_argument("--review-posterior", type=float, default=0.5)
    parser.add_argument(
        "--tf-class-pseudocount",
        type=float,
        default=0.5,
        help=(
            "Symmetric per-configuration pseudocount for localized single/"
            "composite site-consensus priors"
        ),
    )
    parser.add_argument(
        "--tf-class-locus-gap",
        type=int,
        default=30,
        help="Maximum gap joining atomic footprint states into one consensus locus",
    )
    parser.add_argument(
        "--tf-class-max-span",
        type=int,
        default=250,
        help="Maximum span in bp of one localized site-consensus locus",
    )
    parser.add_argument(
        "--tf-class-max-sites",
        type=int,
        default=10,
        help=(
            "Maximum atomic states enumerated in a complete site-consensus action set; "
            "larger loci are reported as skipped"
        ),
    )
    parser.add_argument("--maximum-sites-per-decision", type=int, default=8)
    parser.add_argument(
        "--accessible-site-gap",
        type=int,
        default=DEFAULT_ACCESSIBLE_SITE_GAP,
        help="Maximum gap joining TF sites into one MSP-origin decision",
    )
    parser.add_argument(
        "--control-shifts",
        default="",
        help=(
            "Optional comma-separated target-coordinate shifts. Source priors "
            "remain at the true sites; controls are diagnostic, not an FDR null."
        ),
    )
    parser.add_argument("--max-reads", type=int, default=0)
    parser.add_argument(
        "--molecule-collapse",
        choices=("auto", "on", "off"),
        default="auto",
        help="Collapse amplified DAF PCR families; auto enables for DddA/DddB",
    )
    parser.add_argument("--molecule-min-jaccard", type=float, default=0.95)
    parser.add_argument("--molecule-min-deam", type=int, default=10)
    parser.add_argument(
        "--per-molecule-efficiency",
        dest="per_molecule_efficiency",
        action="store_true",
        default=True,
        help="Calibrate hard-call efficiency from each molecule's MSPs (default)",
    )
    parser.add_argument(
        "--global-efficiency",
        dest="per_molecule_efficiency",
        action="store_false",
        help="Use the model-wide accessible hard-call rate for every molecule",
    )
    parser.add_argument("--efficiency-pseudo-count", type=float, default=20.0)
    parser.add_argument("--efficiency-min-opportunities", type=int, default=20)
    parser.add_argument(
        "--report-layout",
        choices=("auto", "inline", "stream"),
        default="auto",
        help=(
            "Report action storage: auto spills at the bounded v4 limits; "
            "stream forces v5 BGZF action sidecars"
        ),
    )
    parser.add_argument(
        "--diagnostics",
        choices=("aggregate", "stream"),
        default="aggregate",
        help="Per-call diagnostic storage (aggregate is the production default)",
    )
    parser.add_argument("--proposal-tsv")
    parser.add_argument("-o", "--output", required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.diagnostics == "stream":
        parser.error(
            "--diagnostics stream is not implemented; use the production "
            "--diagnostics aggregate mode"
        )
    if args.forced_sites_only and not args.site:
        parser.error("--forced-sites-only requires at least one --site")
    if args.forced_nuc_sites_only and not args.nuc_site:
        parser.error("--forced-nuc-sites-only requires at least one --nuc-site")
    if args.skip_nuc_edge_refinement and (
        args.nuc_site or args.forced_nuc_sites_only
    ):
        parser.error("--skip-nuc-edge-refinement cannot be combined with nuc sites")
    if args.min_support < 1:
        parser.error("--min-support must be positive")
    if args.minimum_geometry_support < 1:
        parser.error("--minimum-geometry-support must be positive")
    if args.source_boundary_margin < 0 or args.nuc_source_boundary_margin < 0:
        parser.error("source boundary margins must be non-negative")
    if args.nuc_min_support is not None and args.nuc_min_support < 1:
        parser.error("--nuc-min-support must be positive")
    if (
        args.strand_min_source_support is not None
        and args.strand_min_source_support < 1
    ):
        parser.error("--strand-min-source-support must be positive")
    if args.center_radius < 0:
        parser.error("--center-radius must be non-negative")
    if args.tf_edge_compatibility < 0:
        parser.error("--tf-edge-compatibility must be non-negative")
    if args.nuc_center_radius < 0:
        parser.error("--nuc-center-radius must be non-negative")
    if args.peak_distance < 1:
        parser.error("--peak-distance must be positive")
    if args.nuc_edge_assignment_radius < 1:
        parser.error("--nuc-edge-assignment-radius must be positive")
    finite_parameters = {
        "--max-boundary-mad": args.max_boundary_mad,
        "--nuc-max-boundary-mad": args.nuc_max_boundary_mad,
        "--min-local-enrichment": args.min_local_enrichment,
        "--strand-min-source-enrichment": args.strand_min_source_enrichment,
        "--strong-posterior": args.strong_posterior,
        "--review-posterior": args.review_posterior,
        "--molecule-min-jaccard": args.molecule_min_jaccard,
        "--efficiency-pseudo-count": args.efficiency_pseudo_count,
        "--tf-class-pseudocount": args.tf_class_pseudocount,
    }
    nonfinite_parameters = [
        name for name, value in finite_parameters.items() if not math.isfinite(value)
    ]
    if nonfinite_parameters:
        parser.error(
            "numeric arguments must be finite: " + ", ".join(nonfinite_parameters)
        )
    if args.max_boundary_mad < 0.0:
        parser.error("--max-boundary-mad must be non-negative")
    if args.nuc_max_boundary_mad < 0.0:
        parser.error("--nuc-max-boundary-mad must be non-negative")
    if (
        args.min_local_enrichment < 0.0
        or args.strand_min_source_enrichment < 0.0
    ):
        parser.error("enrichment thresholds must be non-negative")
    if args.local_background_radius < 1:
        parser.error("--local-background-radius must be positive")
    if args.max_auto_sites < 0:
        parser.error("--max-auto-sites must be non-negative")
    if args.max_auto_nuc_sites < 0:
        parser.error("--max-auto-nuc-sites must be non-negative")
    if args.control_flank < 0:
        parser.error("--control-flank must be non-negative")
    if not 0 <= args.min_mapq <= 255:
        parser.error("--min-mapq must be in [0,255]")
    if args.prob_threshold is not None and not 0 <= args.prob_threshold <= 255:
        parser.error("--prob-threshold must be in [0,255]")
    if args.max_reads < 0:
        parser.error("--max-reads must be non-negative")
    if args.maximum_sites_per_decision < 1:
        parser.error("--maximum-sites-per-decision must be positive")
    if args.accessible_site_gap < 0:
        parser.error("--accessible-site-gap must be non-negative")
    if not 0.0 <= args.review_posterior <= args.strong_posterior <= 1.0:
        parser.error(
            "posterior thresholds must satisfy 0 <= review <= strong <= 1"
        )
    if args.efficiency_pseudo_count < 0.0:
        parser.error("--efficiency-pseudo-count must be non-negative")
    if args.tf_class_pseudocount < 0.0:
        parser.error("--tf-class-pseudocount must be non-negative")
    if args.tf_class_locus_gap < 0:
        parser.error("--tf-class-locus-gap must be non-negative")
    if args.tf_class_max_span < 1:
        parser.error("--tf-class-max-span must be positive")
    if args.tf_class_max_sites < 1:
        parser.error("--tf-class-max-sites must be positive")
    if args.efficiency_min_opportunities < 0:
        parser.error("--efficiency-min-opportunities must be non-negative")
    if not 0.0 < args.molecule_min_jaccard <= 1.0:
        parser.error("--molecule-min-jaccard must be in (0,1]")
    if args.molecule_min_deam < 1:
        parser.error("--molecule-min-deam must be positive")
    try:
        control_shifts = sorted(
            {
                int(value)
                for value in args.control_shifts.split(",")
                if value and int(value) != 0
            }
        )
    except ValueError:
        parser.error("--control-shifts must be comma-separated integers")

    profiler = _StageProfiler()
    preset = PRESETS[args.preset]
    model_path = resolve_resource_path(args.model or preset["model"])
    nuc_model_path = resolve_resource_path(
        args.nuc_model or preset.get("nuc_model", preset["model"])
    )
    output = Path(args.output).expanduser()
    proposal_path = (
        Path(args.proposal_tsv).expanduser() if args.proposal_tsv else None
    )
    missing = [
        path
        for path in [*args.bam, model_path, nuc_model_path]
        if not Path(path).is_file()
    ]
    if missing:
        parser.error("missing input file(s): " + ", ".join(missing))
    try:
        resolved_bams, protected_generation_paths = validate_generation_paths(
            args.bam,
            (model_path, nuc_model_path),
            output,
            proposal_path,
        )
    except ValueError as error:
        parser.error(str(error))
    model, context_size, mode = load_model_with_metadata(model_path)
    nuc_model, nuc_context_size, nuc_mode = load_model_with_metadata(nuc_model_path)
    if (nuc_context_size, nuc_mode) != (context_size, mode):
        parser.error("TF and nuc edge models must use the same context size and mode")
    llr_hit, llr_miss = build_llr_tables(model)
    probability_threshold = (
        args.prob_threshold
        if args.prob_threshold is not None
        else preset["prob_threshold"]
    )
    profiler.mark(
        "model_loading",
        {
            "tf_model": str(Path(model_path).resolve()),
            "nuc_model": str(Path(nuc_model_path).resolve()),
        },
    )
    chrom, focal_start, focal_end = args.region
    load_start = max(0, focal_start - args.control_flank)
    load_end = focal_end + args.control_flank

    reads: list[ReadEvidence] = []
    input_load_diagnostics: List[dict] = []
    for bam_index, bam_path in enumerate(args.bam):
        remaining = 0 if not args.max_reads else max(0, args.max_reads - len(reads))
        load_reads = not args.max_reads or remaining > 0
        diagnostics: dict = {}
        loaded_reads = load_region_evidence(
            bam_path,
            chrom,
            load_start,
            load_end,
            strand_mode=preset["strand_mode"],
            mode=mode,
            context_size=context_size,
            prob_threshold=probability_threshold,
            llr_hit=llr_hit,
            llr_miss=llr_miss,
            min_mapq=args.min_mapq,
            max_reads=remaining,
            input_index=bam_index,
            load_diagnostics=diagnostics,
            load_reads=load_reads,
        )
        reads.extend(loaded_reads)
        input_load_diagnostics.append(diagnostics)
        profiler.mark(
            "bam_evidence_loading",
            {
                "bam_index": bam_index,
                "bam": str(Path(bam_path).expanduser().resolve()),
                "raw_reads_added": len(loaded_reads),
                "raw_reads_cumulative": len(reads),
                "regional_fetch_records": int(
                    diagnostics.get("fetch_record_count", 0)
                ),
            },
        )
    raw_read_count = len(reads)
    target_reads = list(reads)
    if args.per_molecule_efficiency:
        efficiency = calibrate_cohort_efficiency(
            reads,
            model,
            pseudo_count=args.efficiency_pseudo_count,
            min_opportunities=args.efficiency_min_opportunities,
        )
        if Path(nuc_model_path).resolve() == Path(model_path).resolve():
            for read in reads:
                read.nuc_steps = read.steps.copy()
            nuc_efficiency = {
                **efficiency,
                "step_attribute": "nuc_steps",
                "shared_with_tf_model": True,
            }
        else:
            nuc_efficiency = calibrate_cohort_efficiency(
                reads,
                nuc_model,
                pseudo_count=args.efficiency_pseudo_count,
                min_opportunities=args.efficiency_min_opportunities,
                step_attribute="nuc_steps",
            )
            nuc_efficiency["shared_with_tf_model"] = False
    else:
        efficiency = {"enabled": False, "reads": len(reads)}
        nuc_efficiency = assign_global_efficiency_steps(
            reads, nuc_model, step_attribute="nuc_steps"
        )
        nuc_efficiency["shared_with_tf_model"] = (
            Path(nuc_model_path).resolve() == Path(model_path).resolve()
        )
    profiler.mark(
        "efficiency_calibration",
        {
            "raw_reads": len(reads),
            "per_molecule_efficiency": bool(args.per_molecule_efficiency),
            "shared_tf_nuc_model": bool(
                Path(nuc_model_path).resolve() == Path(model_path).resolve()
            ),
        },
    )

    collapse_enabled = args.molecule_collapse == "on" or (
        args.molecule_collapse == "auto" and args.preset in {"ddda", "dddb"}
    )
    if collapse_enabled:
        reads, molecule_diagnostics = collapse_amplified_reads(
            reads,
            min_jaccard=args.molecule_min_jaccard,
            min_deam=args.molecule_min_deam,
        )
    else:
        molecule_diagnostics = {
            "mode": "read",
            "raw_reads": raw_read_count,
            "analyzed_molecules": len(reads),
            "duplicate_reads_collapsed": 0,
        }
    profiler.mark(
        "molecule_collapse",
        {
            "enabled": bool(collapse_enabled),
            "raw_reads": raw_read_count,
            "analyzed_molecules": len(reads),
            "duplicate_reads_collapsed": int(
                molecule_diagnostics.get("duplicate_reads_collapsed", 0)
            ),
        },
    )

    tf_site_diagnostics = {}
    automatic_sites = discover_sites(
        reads,
        focal_start,
        focal_end,
        min_support=args.min_support,
        minimum_geometry_support=args.minimum_geometry_support,
        center_radius=args.center_radius,
        peak_distance=args.peak_distance,
        max_boundary_mad=args.max_boundary_mad,
        min_local_enrichment=args.min_local_enrichment,
        local_background_radius=args.local_background_radius,
        max_auto_sites=args.max_auto_sites,
        source_boundary_margin=args.source_boundary_margin,
        edge_compatibility_bp=args.tf_edge_compatibility,
        separate_strand_maps=preset["strand_mode"] == "daf",
        diagnostics=tf_site_diagnostics,
    )
    sites = merge_forced_sites(
        automatic_sites,
        args.site,
        reads,
        center_radius=args.center_radius,
        minimum_geometry_support=args.minimum_geometry_support,
        forced_only=args.forced_sites_only,
        source_boundary_margin=args.source_boundary_margin,
        edge_assignment_radius=args.tf_edge_compatibility,
    )
    profiler.mark(
        "tf_site_discovery",
        {
            **{
                key: value
                for key, value in tf_site_diagnostics.items()
                if key != "latent_pair_test_records"
            },
            "automatic_sites": len(automatic_sites),
            "selected_sites": len(sites),
            "max_auto_sites": args.max_auto_sites,
        },
    )
    nuc_minimum_support = (
        args.min_support if args.nuc_min_support is None else args.nuc_min_support
    )
    if args.skip_nuc_edge_refinement:
        nuc_sites = []
        nuc_site_diagnostics = {
            "enabled": False,
            "cap_bound": False,
            "retained_after_cap": 0,
        }
    else:
        nuc_site_diagnostics = {"enabled": True}
        automatic_nuc_sites = discover_edge_sites(
            reads,
            focal_start,
            focal_end,
            call_type="nuc",
            min_support=nuc_minimum_support,
            minimum_geometry_support=args.minimum_geometry_support,
            center_radius=args.nuc_center_radius,
            edge_assignment_radius=args.nuc_edge_assignment_radius,
            max_boundary_mad=args.nuc_max_boundary_mad,
            min_local_enrichment=1.0,
            local_background_radius=args.local_background_radius,
            max_auto_sites=args.max_auto_nuc_sites,
            boundary_reliability_scale=24.0,
            source_boundary_margin=args.nuc_source_boundary_margin,
            diagnostics=nuc_site_diagnostics,
        )
        nuc_sites = merge_forced_sites(
            automatic_nuc_sites,
            args.nuc_site,
            reads,
            center_radius=args.nuc_center_radius,
            minimum_geometry_support=args.minimum_geometry_support,
            forced_only=args.forced_nuc_sites_only,
            call_type="nuc",
            boundary_reliability_scale=24.0,
            source_boundary_margin=args.nuc_source_boundary_margin,
            edge_assignment_radius=args.nuc_edge_assignment_radius,
        )
    profiler.mark(
        "nuc_site_discovery",
        {
            **nuc_site_diagnostics,
            "enabled": not args.skip_nuc_edge_refinement,
            "selected_sites": len(nuc_sites),
            "max_auto_sites": args.max_auto_nuc_sites,
        },
    )

    minimum_source_support = (
        args.min_support
        if args.strand_min_source_support is None
        else args.strand_min_source_support
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    registry = _StagingRegistry()
    generation_sink = _GenerationSink(
        output,
        args.report_layout,
        registry,
        proposal_tsv=proposal_path,
    )
    try:
        strand_rescue = analyze_strand_rescue(
            reads,
            sites,
            target_reads=target_reads,
            nuc_sites=nuc_sites,
            min_source_support=minimum_source_support,
            min_source_local_enrichment=args.strand_min_source_enrichment,
            maximum_sites_per_decision=args.maximum_sites_per_decision,
            accessible_site_gap=args.accessible_site_gap,
            center_radius=args.center_radius,
            tf_edge_compatibility=args.tf_edge_compatibility,
            tf_class_pseudocount=args.tf_class_pseudocount,
            tf_class_locus_gap=args.tf_class_locus_gap,
            tf_class_max_span=args.tf_class_max_span,
            tf_class_max_sites=args.tf_class_max_sites,
            minimum_geometry_support=args.minimum_geometry_support,
            nuc_center_radius=args.nuc_center_radius,
            strong_posterior=args.strong_posterior,
            review_posterior=args.review_posterior,
            performance_callback=profiler.inference_callback,
            performance_stage_prefix="primary_",
            decision_callback=generation_sink.decision_callback,
            edge_callback=generation_sink.edge_callback,
            retain_per_record=False,
            stream_order=True,
        )
    except ValueError as error:
        parser.error(str(error))
    strand_rescue["site_discovery"] = {
        "tf": tf_site_diagnostics,
        "nuc": nuc_site_diagnostics,
    }
    generation_sink.finish_callbacks()
    try:
        report_layout = generation_sink.final_layout()
    except ValueError as error:
        parser.error(str(error))
    profiler.mark(
        "action_detail_collection",
        {
            "requested_layout": args.report_layout,
            "selected_layout": report_layout,
            "details_seen": generation_sink.detail_count,
            "estimated_compact_inline_json_bytes": (
                generation_sink.estimated_json_bytes
            ),
            "candidate_rejections": dict(
                sorted(generation_sink.candidate_rejections.items())
            ),
        },
    )

    controls = {}
    for shift in control_shifts:
        shifted = shift_site_templates(sites, shift)
        control = analyze_strand_rescue(
            reads,
            sites,
            target_reads=target_reads,
            nuc_sites=(),
            target_sites=shifted,
            min_source_support=minimum_source_support,
            min_source_local_enrichment=args.strand_min_source_enrichment,
            maximum_sites_per_decision=args.maximum_sites_per_decision,
            accessible_site_gap=args.accessible_site_gap,
            center_radius=args.center_radius,
            tf_edge_compatibility=args.tf_edge_compatibility,
            tf_class_pseudocount=args.tf_class_pseudocount,
            tf_class_locus_gap=args.tf_class_locus_gap,
            tf_class_max_span=args.tf_class_max_span,
            tf_class_max_sites=args.tf_class_max_sites,
            minimum_geometry_support=args.minimum_geometry_support,
            strong_posterior=args.strong_posterior,
            review_posterior=args.review_posterior,
            performance_callback=profiler.inference_callback,
            performance_stage_prefix=f"control_{shift}_",
            retain_per_record=False,
            stream_order=True,
        )
        controls[str(shift)] = {
            "target_sites": [
                [site.start, site.end] for site in shifted
            ],
            "applicable": control["applicable"],
            "counts": control["counts"],
        }
    strand_rescue["coordinate_controls"] = controls
    strand_rescue["coordinate_control_interpretation"] = (
        "target-geometry stress test with the true source prior retained; "
        "not an FDR null"
    )

    action_storage = None
    action_generation_diagnostics: Dict[str, object] = {
        "candidate_rejections": dict(
            sorted(generation_sink.candidate_rejections.items())
        )
    }
    if report_layout == "inline":
        try:
            _hydrate_inline_details(strand_rescue, generation_sink)
        except ValueError as error:
            parser.error(str(error))
    else:
        try:
            action_storage, topology_diagnostics = _build_v5_action_storage(
                generation_sink,
                target_reads,
                [
                    int(value.get("fetch_record_count", 0))
                    for value in input_load_diagnostics
                ],
                [chrom, load_start, load_end],
                registry,
                protected_paths=tuple(protected_generation_paths),
            )
        except (OSError, ValueError) as error:
            parser.error(str(error))
        action_generation_diagnostics["topology"] = dict(
            sorted(topology_diagnostics.items())
        )
        strand_rescue.pop("decisions", None)
        for call_type in ("tf", "nuc"):
            section = strand_rescue["edge_refinement"][call_type]
            section.pop("harmonizations", None)
            counts = section.get("counts", {})
            accepted = int(
                action_storage["totals"][f"{call_type}_edge_updates"]
            )
            counts["edge_updates"] = accepted
            counts["joint_topology_conflicts_retained"] = int(
                topology_diagnostics.get(
                    f"{call_type}_edge_joint_collision", 0
                )
            )
            counts["rescue_topology_conflicts_retained"] = int(
                topology_diagnostics.get(
                    f"{call_type}_edge_rescue_collision", 0
                )
            )
            counts["action_projection_rejected"] = sum(
                int(value)
                for name, value in generation_sink.candidate_rejections.items()
                if name.startswith(f"{call_type}_")
            )
            counts["prior_only_updates"] = int(
                topology_diagnostics.get(
                    f"{call_type}_edge_accepted_prior_only", 0
                )
            )
            counts["chemistry_opposed_updates"] = int(
                topology_diagnostics.get(
                    f"{call_type}_edge_accepted_chemistry_opposed", 0
                )
            )
            counts["extreme_edge_shifts"] = int(
                topology_diagnostics.get(
                    f"{call_type}_edge_accepted_extreme_edge_shift", 0
                )
            )
        strand_rescue["action_storage"] = action_storage
        strand_rescue["diagnostic_storage"] = {
            "mode": "aggregate",
            "action_generation": action_generation_diagnostics,
        }
        profiler.mark(
            "action_stream_finalization_and_publication",
            {
                **action_storage["totals"],
                "streams": len(action_storage["streams"]),
            },
        )

    report = {
        "schema": "fiberhmm.strand_rescue.v6",
        "schema_version": 6,
        "producer": {
            "name": "fiberhmm-strand-rescue",
            "fiberhmm_version": FIBERHMM_VERSION,
            "model_sha256": _sha256_file(model_path),
            "nuc_edge_model_sha256": _sha256_file(nuc_model_path),
        },
        "normalized_layer_contract": {
            "groups": ["nuc_sr.QQQ", "tf_sr.QQQ"],
            "source_call_groups_preserved": True,
            "source_evidence_preserved": True,
            "normalized_groups_are_optional_parallel_layers": True,
            "single_molecule_policy": (
                "retain_raw_calls_and_supported_per_molecule_variation"
            ),
            "q0": (
                "role_specific_R_exact_selected_configuration_probability_"
                "within_accessible_plus_all_supported_TF_configurations_or_"
                "zero_when_truncated_H_assignment_marginalized_canonical_"
                "geometry_probability_unnamed_baseline_sentinel"
            ),
            "q1": (
                "role_specific_R_canonical_left_boundary_reliability_H_"
                "molecular_left_edge_confidence_or_zero_for_a_changed_H_edge_"
                "without_target_opportunity"
            ),
            "q2": (
                "role_specific_R_canonical_right_boundary_reliability_H_"
                "molecular_right_edge_confidence_or_zero_for_a_changed_H_edge_"
                "without_target_opportunity"
            ),
            "probability_interpretation": (
                "conditional_same_cohort_model_probability_not_held_out_"
                "calibrated_biological_truth"
            ),
            "rescue_probability_partition": [
                "accessible_probability_within_supported_action_set",
                "selected_configuration_probability_vs_accessible_and_"
                "supported_tf",
                "other_supported_tf_configuration_probability",
                "unresolved_action_set_probability",
            ],
            "threshold_scope": "named_R_and_H_groups_only",
            "unnamed_baseline_row": [255, 0, 0],
            "nucleosome_identity_cardinality_fixed": True,
        },
        "input": {
            "bams": [str(path) for path in resolved_bams],
            "files": [_file_metadata(path) for path in args.bam],
            "preset": args.preset,
            "model": str(Path(model_path).resolve()),
            "nuc_edge_model": str(Path(nuc_model_path).resolve()),
            "mode": mode,
            "prob_threshold": probability_threshold,
            "focal_region": [chrom, focal_start, focal_end],
            "loaded_region": [chrom, load_start, load_end],
            "cohort_semantics": "explicitly_pooled_same_assay_population",
        },
        "parameters": {
            "min_mapq": args.min_mapq,
            "control_flank": args.control_flank,
            "min_support": args.min_support,
            "minimum_geometry_support": args.minimum_geometry_support,
            "source_boundary_margin": args.source_boundary_margin,
            "center_radius": args.center_radius,
            "peak_distance": args.peak_distance,
            "tf_edge_compatibility": args.tf_edge_compatibility,
            "max_boundary_mad": args.max_boundary_mad,
            "min_local_enrichment": args.min_local_enrichment,
            "local_background_radius": args.local_background_radius,
            "max_auto_sites": args.max_auto_sites,
            "nuc_min_support": nuc_minimum_support,
            "nuc_center_radius": args.nuc_center_radius,
            "nuc_source_boundary_margin": args.nuc_source_boundary_margin,
            "nuc_edge_assignment_radius": args.nuc_edge_assignment_radius,
            "nuc_max_boundary_mad": args.nuc_max_boundary_mad,
            "max_auto_nuc_sites": args.max_auto_nuc_sites,
            "forced_nuc_site_intervals": [
                list(interval) for interval in args.nuc_site
            ],
            "forced_nuc_sites_only": args.forced_nuc_sites_only,
            "skip_nuc_edge_refinement": args.skip_nuc_edge_refinement,
            "nuc_length_ceiling": None,
            "strand_min_source_support": minimum_source_support,
            "strand_min_source_enrichment": args.strand_min_source_enrichment,
            "strong_posterior": args.strong_posterior,
            "review_posterior": args.review_posterior,
            "tf_class_pseudocount": args.tf_class_pseudocount,
            "tf_class_locus_gap": args.tf_class_locus_gap,
            "tf_class_max_span": args.tf_class_max_span,
            "tf_class_max_sites": args.tf_class_max_sites,
            "maximum_sites_per_decision": args.maximum_sites_per_decision,
            "accessible_site_gap": args.accessible_site_gap,
            "forced_site_intervals": [list(interval) for interval in args.site],
            "forced_sites_only": args.forced_sites_only,
            "control_shifts": control_shifts,
            "molecule_collapse": args.molecule_collapse,
            "molecule_min_jaccard": args.molecule_min_jaccard,
            "molecule_min_deam": args.molecule_min_deam,
            "per_molecule_efficiency": args.per_molecule_efficiency,
            "efficiency_pseudo_count": args.efficiency_pseudo_count,
            "efficiency_min_opportunities": args.efficiency_min_opportunities,
            "max_reads": args.max_reads,
            "report_layout_requested": args.report_layout,
            "report_layout_selected": report_layout,
            "diagnostics": args.diagnostics,
            "minimum_mapped_annotation_fraction": (
                MIN_MAPPED_ANNOTATION_FRACTION
            ),
        },
        "molecule_diagnostics": molecule_diagnostics,
        "efficiency_calibration": efficiency,
        "nuc_edge_efficiency_calibration": nuc_efficiency,
        "n_raw_reads": raw_read_count,
        "n_analyzed_molecules": len(reads),
        "strand_rescue": strand_rescue,
        "guardrails": {
            "writes_bam": False,
            "reads_only_standard_hard_calls": True,
            "external_assay_prior_used": False,
            "input_bam_identity_partitions_prior": False,
            "pacbio_mode_exposed_by_this_command": False,
            "requires_positive_target_evidence_per_proposed_component": True,
            "source_state_model_includes_nucleosome_nuisance": True,
            "rescue_action_prior_conditions_on_non_nucleosome_states": True,
            "nucleosome_occupancy_candidates": False,
            "nucleosome_identity_or_cardinality_modified": False,
            "nucleosome_edges_may_be_normalized": True,
            "nucleosome_edge_length_ceiling": None,
            "ordinary_tf_tq_used_for_eligibility": False,
            "ordinary_tf_calls_anchor_source_model": True,
            "source_direction_uses_opportunity_conditioned_detection": True,
            "shared_geometry_balances_represented_strata": True,
            "short_reads_may_contribute_per_site": True,
            "amplified_daf_collapse_crosses_input_bams": False,
            "amplified_daf_collapse_removes_output_targets": False,
        },
    }
    profiler.mark(
        "report_assembly",
        {
            "raw_reads": raw_read_count,
            "analyzed_molecules": len(reads),
            "tf_sites": len(sites),
            "nuc_sites": len(nuc_sites),
        },
    )
    report["performance"] = profiler.snapshot(
        scope="report_generation_before_validation_serialization_and_write"
    )
    nonfinite = _nonfinite_json_paths(report)
    if nonfinite:
        parser.error(
            "report contains non-finite model values: "
            + ", ".join(nonfinite[:10])
        )
    profiler.mark("report_validation", {"nonfinite_values": 0})
    report["performance"] = profiler.snapshot(
        scope="report_generation_after_validation_before_serialization_and_write"
    )
    _atomic_write(output, json.dumps(report, indent=2, allow_nan=False) + "\n")
    generation_sink.publish_proposal()
    profiler.mark(
        "report_serialization_and_write",
        {
            "json_size_bytes": int(output.stat().st_size),
            "proposal_tsv_written": bool(args.proposal_tsv),
        },
    )
    completion_performance = profiler.snapshot(
        scope="complete_cli_before_summary_output"
    )
    print(
        json.dumps(
            {
                "output": str(output.resolve()),
                "raw_reads": raw_read_count,
                "analyzed_molecules": len(reads),
                "sites": len(sites),
                "nuc_sites": len(nuc_sites),
                "applicable": strand_rescue["applicable"],
                "decisions": (
                    int(action_storage["totals"]["rescue_decisions"])
                    if action_storage is not None
                    else len(strand_rescue["decisions"])
                ),
                "report_layout": report_layout,
                "counts": strand_rescue["counts"],
                "performance": {
                    "total_wall_seconds": completion_performance[
                        "total_wall_seconds"
                    ],
                    "process_peak_rss_bytes": completion_performance.get(
                        "process_peak_rss_bytes"
                    ),
                    "report_serialization_and_write_wall_seconds": (
                        completion_performance["stages"][-1]["wall_seconds"]
                    ),
                },
            },
            sort_keys=True,
        )
    )
    registry.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
