#!/usr/bin/env python3
"""Write normalized ``nuc_sr`` and ``tf_sr`` layers from an SR report.

The source BAM and its ordinary ``nuc``, ``msp``, and ``tf`` annotations are
never modified. Each output is a new sorted, indexed regional BAM. Existing TF
and nucleosome calls may receive canonical shared edges, and weak-but-positive
MSP footprints may be added. Nucleosome identity and cardinality remain fixed.
"""
from __future__ import annotations

import argparse
import array
import errno
import hashlib
import json
import math
import os
import re
import shlex
import sys
import tempfile
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, DefaultDict, Dict, List, Mapping, Optional, Sequence, Tuple

import pysam

from fiberhmm import __version__ as FIBERHMM_VERSION
from fiberhmm.cli.strand_rescue import parse_region
from fiberhmm.io.bam_header import append_ma_types, append_pg_record
from fiberhmm.io.ma_tags import (
    flip_interval_frame,
    format_an_tag,
    parse_an_tag,
    parse_aq_array,
    parse_ma_tag,
)


LAYER_ORDER = ("nuc_sr", "tf_sr")
REPLACED_LAYER_NAMES = frozenset(("nuc_sr", "tf_sr"))
QUALITY_SPEC = "QQQ"
LEGACY_QUALITY_SPEC = "QQQQQ"
HEADER_PREFIX = "FIBERHMM-STRAND-RESCUE:v4:"
V6_HEADER_PREFIX = "FIBERHMM-STRAND-RESCUE:v6:"
V3_HEADER_PREFIX = "FIBERHMM-STRAND-RESCUE:v3:"
LEGACY_HEADER_PREFIX = "FIBERHMM-STRAND-RESCUE:v2:"
ALL_HEADER_PREFIXES = (
    V6_HEADER_PREFIX,
    HEADER_PREFIX,
    V3_HEADER_PREFIX,
    LEGACY_HEADER_PREFIX,
)
REPORT_SCHEMAS = frozenset(
    (
        "fiberhmm.strand_rescue.v2",
        "fiberhmm.strand_rescue.v3",
        "fiberhmm.strand_rescue.v4",
        "fiberhmm.strand_rescue.v6",
    )
)
V5_REPORT_SCHEMA = "fiberhmm.strand_rescue.v5"
V6_REPORT_SCHEMA = "fiberhmm.strand_rescue.v6"
STREAM_REPORT_VERSIONS = {
    V5_REPORT_SCHEMA: 5,
    V6_REPORT_SCHEMA: 6,
}
V5_ACTION_STREAM_SCHEMA = "fiberhmm.strand_rescue.actions.v1"
V5_ACTION_LAYOUT = "per_input_bgzf_jsonl_v1"
_TOKEN_PATTERN = re.compile(r"^[0-9a-f]{16}$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_BGZF_EOF = bytes.fromhex(
    "1f8b08040000000000ff0600424302001b0003000000000000000000"
)


def _strict_json_loads(raw: object, context: str) -> object:
    def reject_constant(value: str) -> None:
        raise ValueError(f"{context} contains non-finite JSON value {value}")

    def unique_object(pairs: Sequence[Tuple[str, object]]) -> dict:
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{context} contains duplicate key {key!r}")
            result[key] = value
        return result

    return json.loads(
        raw,
        parse_constant=reject_constant,
        object_pairs_hook=unique_object,
    )


@dataclass(frozen=True)
class StreamRescueComponent:
    component_index: int
    interval: Tuple[int, int]
    left_q: int
    right_q: int


@dataclass(frozen=True)
class StreamRescueAction:
    token: str
    source_ordinal: int
    source_interval: Tuple[int, int]
    alternative_q: int
    components: Tuple[StreamRescueComponent, ...]


@dataclass(frozen=True)
class StreamEdgeAction:
    token: str
    call_type: str
    source_ordinal: int
    source_interval: Tuple[int, int]
    alternative_interval: Tuple[int, int]
    quality: Tuple[int, int, int]


@dataclass(frozen=True)
class StreamActionRecord:
    ordinal: int
    read_name: str
    record_sha256: str
    rescues: Tuple[StreamRescueAction, ...]
    edge_updates: Tuple[StreamEdgeAction, ...]


def _require_exact_keys(
    value: Mapping[str, object], expected: Sequence[str], context: str
) -> None:
    observed = set(value)
    wanted = set(expected)
    if observed != wanted:
        missing = sorted(wanted - observed)
        unknown = sorted(observed - wanted)
        raise ValueError(
            f"{context} keys differ from schema; missing={missing}, "
            f"unknown={unknown}"
        )


def _require_int(value: object, context: str, *, minimum: int = 0) -> int:
    if type(value) is not int or int(value) < minimum:
        raise ValueError(f"{context} must be an integer >= {minimum}")
    return int(value)


def _require_q(value: object, context: str) -> int:
    result = _require_int(value, context)
    if result > 255:
        raise ValueError(f"{context} must be in [0,255]")
    return result


def _require_interval(value: object, context: str) -> Tuple[int, int]:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{context} must be [molecular_start,length]")
    start = _require_int(value[0], f"{context}[0]")
    length = _require_int(value[1], f"{context}[1]", minimum=1)
    return start, length


def _require_token(value: object, context: str) -> str:
    if type(value) is not str:
        raise ValueError(f"{context} must be a string")
    token = value
    if not _TOKEN_PATTERN.fullmatch(token):
        raise ValueError(f"{context} must be 16 lowercase hexadecimal characters")
    return token


def _require_sha256(value: object, context: str) -> str:
    if type(value) is not str:
        raise ValueError(f"{context} must be a string")
    digest = value
    if not _SHA256_PATTERN.fullmatch(digest):
        raise ValueError(f"{context} must be 64 lowercase hexadecimal characters")
    return digest


def _parse_stream_rescue(value: object, context: str) -> StreamRescueAction:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be an object")
    _require_exact_keys(
        value,
        ("token", "source_ordinal", "source_interval", "q0", "components"),
        context,
    )
    source = _require_interval(value["source_interval"], f"{context}.source_interval")
    raw_components = value["components"]
    if not isinstance(raw_components, list) or not raw_components:
        raise ValueError(f"{context}.components must be a non-empty list")
    components = []
    source_end = source[0] + source[1]
    for index, raw in enumerate(raw_components):
        component_context = f"{context}.components[{index}]"
        if not isinstance(raw, Mapping):
            raise ValueError(f"{component_context} must be an object")
        _require_exact_keys(
            raw, ("component_index", "interval", "q1", "q2"), component_context
        )
        component_index = _require_int(
            raw["component_index"], f"{component_context}.component_index"
        )
        if component_index != index:
            raise ValueError(
                f"{context}.components must retain configuration order with "
                "consecutive component_index values"
            )
        interval = _require_interval(raw["interval"], f"{component_context}.interval")
        if interval[0] < source[0] or interval[0] + interval[1] > source_end:
            raise ValueError(f"{component_context} lies outside its source MSP")
        components.append(
            StreamRescueComponent(
                component_index=component_index,
                interval=interval,
                left_q=_require_q(raw["q1"], f"{component_context}.q1"),
                right_q=_require_q(raw["q2"], f"{component_context}.q2"),
            )
        )
    ordered_intervals = sorted(component.interval for component in components)
    if any(
        left[0] + left[1] > right[0]
        for left, right in zip(ordered_intervals, ordered_intervals[1:])
    ):
        raise ValueError(f"{context}.components must be non-overlapping")
    return StreamRescueAction(
        token=_require_token(value["token"], f"{context}.token"),
        source_ordinal=_require_int(
            value["source_ordinal"], f"{context}.source_ordinal"
        ),
        source_interval=source,
        alternative_q=_require_q(value["q0"], f"{context}.q0"),
        components=tuple(components),
    )


def _parse_stream_edge(value: object, context: str) -> StreamEdgeAction:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be an object")
    _require_exact_keys(
        value,
        (
            "token",
            "call_type",
            "source_ordinal",
            "source_interval",
            "alternative_interval",
            "q",
        ),
        context,
    )
    if type(value["call_type"]) is not str:
        raise ValueError(f"{context}.call_type must be a string")
    call_type = value["call_type"]
    if call_type not in {"tf", "nuc"}:
        raise ValueError(f"{context}.call_type must be tf or nuc")
    raw_q = value["q"]
    if not isinstance(raw_q, list) or len(raw_q) != 3:
        raise ValueError(f"{context}.q must be [q0,q1,q2]")
    return StreamEdgeAction(
        token=_require_token(value["token"], f"{context}.token"),
        call_type=call_type,
        source_ordinal=_require_int(
            value["source_ordinal"], f"{context}.source_ordinal"
        ),
        source_interval=_require_interval(
            value["source_interval"], f"{context}.source_interval"
        ),
        alternative_interval=_require_interval(
            value["alternative_interval"], f"{context}.alternative_interval"
        ),
        quality=tuple(
            _require_q(item, f"{context}.q[{index}]")
            for index, item in enumerate(raw_q)
        ),
    )


def parse_stream_action_record(value: object) -> StreamActionRecord:
    """Validate one actions-v1 record and return its bounded typed form."""
    if not isinstance(value, Mapping):
        raise ValueError("action line must be an object")
    _require_exact_keys(
        value,
        ("kind", "ordinal", "read", "record_sha256", "rescues", "edge_updates"),
        "action line",
    )
    if value["kind"] != "actions":
        raise ValueError("action line kind must be actions")
    if type(value["read"]) is not str:
        raise ValueError("action line read must be a string")
    read_name = value["read"]
    if not read_name:
        raise ValueError("action line read must be non-empty")
    raw_rescues = value["rescues"]
    raw_edges = value["edge_updates"]
    if not isinstance(raw_rescues, list) or not isinstance(raw_edges, list):
        raise ValueError("action line rescues and edge_updates must be lists")
    if not raw_rescues and not raw_edges:
        raise ValueError("empty action lines are forbidden")
    rescues = tuple(
        _parse_stream_rescue(raw, f"rescues[{index}]")
        for index, raw in enumerate(raw_rescues)
    )
    edges = tuple(
        _parse_stream_edge(raw, f"edge_updates[{index}]")
        for index, raw in enumerate(raw_edges)
    )
    names = [
        f"fhsr_{action.token}_R{component.component_index}"
        for action in rescues
        for component in action.components
    ] + [
        f"fhsr_{action.token}_O{action.source_ordinal}_H" for action in edges
    ]
    if len(names) != len(set(names)):
        raise ValueError("action line would emit duplicate AN names")
    edge_sources = [(action.call_type, action.source_ordinal) for action in edges]
    if len(edge_sources) != len(set(edge_sources)):
        raise ValueError("action line has multiple edge updates for one source call")
    return StreamActionRecord(
        ordinal=_require_int(value["ordinal"], "action line ordinal"),
        read_name=read_name,
        record_sha256=_require_sha256(
            value["record_sha256"], "action line record_sha256"
        ),
        rescues=rescues,
        edge_updates=edges,
    )


def _parse_stream_region(value: object, context: str) -> Tuple[str, int, int]:
    if (
        not isinstance(value, list)
        or len(value) != 3
        or type(value[0]) is not str
        or not value[0]
    ):
        raise ValueError(f"{context} must be [chrom,start,end]")
    start = _require_int(value[1], f"{context}[1]")
    end = _require_int(value[2], f"{context}[2]", minimum=1)
    if end <= start:
        raise ValueError(f"{context} must satisfy start < end")
    return value[0], start, end


class V5ActionStreamReader:
    """One-pass, bounded-memory validator for an actions-v1 BGZF stream."""

    _HEADER_KEYS = (
        "kind",
        "schema",
        "input_index",
        "input_id",
        "loaded_region",
        "ordinal_base",
    )
    _TRAILER_KEYS = (
        "kind",
        "fetch_record_count",
        "action_record_count",
        "first_action_ordinal",
        "last_action_ordinal",
        "rescue_decision_count",
        "rescue_component_count",
        "tf_edge_update_count",
        "nuc_edge_update_count",
    )
    _COUNT_KEYS = (
        "fetch_record_count",
        "action_record_count",
        "rescue_decision_count",
        "rescue_component_count",
        "tf_edge_update_count",
        "nuc_edge_update_count",
    )

    def __init__(self, path: Path, manifest: Mapping[str, object]) -> None:
        self.path = path
        self.manifest = manifest
        self.handle = None
        self.digest = hashlib.sha256()
        self.uncompressed_size = 0
        self.buffer = b""
        self.next_action: Optional[StreamActionRecord] = None
        self.trailer: Optional[Mapping[str, object]] = None
        self.previous_ordinal: Optional[int] = None
        self.observed = {
            "action_record_count": 0,
            "rescue_decision_count": 0,
            "rescue_component_count": 0,
            "tf_edge_update_count": 0,
            "nuc_edge_update_count": 0,
        }
        self.first_action_ordinal: Optional[int] = None
        self.last_action_ordinal: Optional[int] = None

    def __enter__(self):
        self.handle = pysam.BGZFile(str(self.path), "rb")
        try:
            header = self._read_json("action stream header")
            _require_exact_keys(header, self._HEADER_KEYS, "action stream header")
            if header["kind"] != "header":
                raise ValueError("action stream must begin with a header")
            if header["schema"] != V5_ACTION_STREAM_SCHEMA:
                raise ValueError("unsupported strand-rescue action stream schema")
            manifest_index = _require_int(
                self.manifest["input_index"], "manifest input_index"
            )
            if (
                _require_int(header["input_index"], "stream header input_index")
                != manifest_index
            ):
                raise ValueError("stream header input_index differs from manifest")
            if (
                type(header["input_id"]) is not str
                or type(self.manifest["input_id"]) is not str
                or header["input_id"] != self.manifest["input_id"]
            ):
                raise ValueError("stream header input_id differs from manifest")
            manifest_region = _parse_stream_region(
                self.manifest["loaded_region"], "manifest loaded_region"
            )
            if (
                _parse_stream_region(
                    header["loaded_region"], "stream header region"
                )
                != manifest_region
            ):
                raise ValueError("stream header region differs from manifest")
            if _require_int(header["ordinal_base"], "stream header ordinal_base") != 0:
                raise ValueError("action stream ordinal_base must be zero")
            self._advance()
            return self
        except BaseException:
            self.handle.close()
            self.handle = None
            raise

    def __exit__(self, exc_type, exc_value, traceback):
        if self.handle is not None:
            self.handle.close()
            self.handle = None

    def _readline(self) -> bytes:
        if self.handle is None:
            raise ValueError("action stream is not open")
        while b"\n" not in self.buffer:
            chunk = self.handle.read(64 * 1024)
            if not chunk:
                break
            self.buffer += chunk
        newline = self.buffer.find(b"\n")
        if newline >= 0:
            raw = self.buffer[: newline + 1]
            self.buffer = self.buffer[newline + 1 :]
        else:
            raw = self.buffer
            self.buffer = b""
        if raw:
            self.digest.update(raw)
            self.uncompressed_size += len(raw)
        return raw

    def _read_json(self, context: str) -> Mapping[str, object]:
        raw = self._readline()
        if not raw:
            raise ValueError(f"unexpected end of BGZF while reading {context}")
        if not raw.endswith(b"\n"):
            raise ValueError(f"{context} is not newline terminated")
        try:
            value = _strict_json_loads(raw.decode("utf-8"), context)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid {context}: {error}") from error
        if not isinstance(value, Mapping):
            raise ValueError(f"{context} must be a JSON object")
        return value

    def _advance(self) -> None:
        if self.trailer is not None:
            raise ValueError("action record occurs after stream trailer")
        value = self._read_json("action stream record")
        kind = value.get("kind")
        if kind == "trailer":
            _require_exact_keys(value, self._TRAILER_KEYS, "action stream trailer")
            self.trailer = value
            self.next_action = None
            if self._readline():
                raise ValueError("action stream has content after trailer")
            return
        record = parse_stream_action_record(value)
        if self.previous_ordinal is not None and record.ordinal <= self.previous_ordinal:
            raise ValueError("action ordinals must be strictly increasing")
        self.previous_ordinal = record.ordinal
        if self.first_action_ordinal is None:
            self.first_action_ordinal = record.ordinal
        self.last_action_ordinal = record.ordinal
        self.observed["action_record_count"] += 1
        self.observed["rescue_decision_count"] += len(record.rescues)
        self.observed["rescue_component_count"] += sum(
            len(action.components) for action in record.rescues
        )
        self.observed["tf_edge_update_count"] += sum(
            action.call_type == "tf" for action in record.edge_updates
        )
        self.observed["nuc_edge_update_count"] += sum(
            action.call_type == "nuc" for action in record.edge_updates
        )
        self.next_action = record

    def pop_for_ordinal(self, ordinal: int) -> Optional[StreamActionRecord]:
        if self.next_action is None:
            return None
        if self.next_action.ordinal < ordinal:
            raise ValueError(
                f"action ordinal {self.next_action.ordinal} was not present in BAM fetch"
            )
        if self.next_action.ordinal > ordinal:
            return None
        result = self.next_action
        self._advance()
        return result

    def finish(self, fetch_record_count: int) -> dict:
        if self.next_action is not None:
            raise ValueError(
                f"action ordinal {self.next_action.ordinal} lies beyond BAM fetch EOF"
            )
        if self.trailer is None:
            raise ValueError("action stream has no trailer")
        expected_fetch = _require_int(
            self.trailer["fetch_record_count"], "trailer fetch_record_count"
        )
        if fetch_record_count != expected_fetch:
            raise ValueError(
                "regional BAM fetch count differs from action stream trailer: "
                f"{fetch_record_count} != {expected_fetch}"
            )
        for key in self._COUNT_KEYS:
            trailer_value = _require_int(self.trailer[key], f"trailer {key}")
            manifest_value = _require_int(self.manifest[key], f"manifest {key}")
            observed_value = (
                fetch_record_count if key == "fetch_record_count" else self.observed[key]
            )
            if not (trailer_value == manifest_value == observed_value):
                raise ValueError(
                    f"{key} disagrees across stream, manifest, and observation"
                )
        for key, observed in (
            ("first_action_ordinal", self.first_action_ordinal),
            ("last_action_ordinal", self.last_action_ordinal),
        ):
            trailer_value = self.trailer[key]
            manifest_value = self.manifest[key]
            if trailer_value is not None:
                trailer_value = _require_int(trailer_value, f"trailer {key}")
            if manifest_value is not None:
                manifest_value = _require_int(manifest_value, f"manifest {key}")
            if not (trailer_value == manifest_value == observed):
                raise ValueError(f"{key} disagrees across stream and manifest")
        expected_size = _require_int(
            self.manifest["uncompressed_size_bytes"],
            "manifest uncompressed_size_bytes",
            minimum=1,
        )
        if self.uncompressed_size != expected_size:
            raise ValueError("decompressed action-stream size differs from manifest")
        observed_digest = self.digest.hexdigest()
        if observed_digest != _require_sha256(
            self.manifest["jsonl_sha256"], "manifest jsonl_sha256"
        ):
            raise ValueError("decompressed action-stream SHA-256 mismatch")
        return {
            **self.observed,
            "fetch_record_count": fetch_record_count,
            "first_action_ordinal": self.first_action_ordinal,
            "last_action_ordinal": self.last_action_ordinal,
            "jsonl_sha256": observed_digest,
            "uncompressed_size_bytes": self.uncompressed_size,
        }


@dataclass(frozen=True)
class StrandDecision:
    decision_id: str
    read_name: str
    library_id: str
    tier: str
    current_state: str
    current_interval: Tuple[int, int]
    tf_intervals: Tuple[Tuple[int, int], ...]
    tf_posterior: float
    current_posterior: float
    configuration_probability: float
    molecule_probability: float
    population_probability: float
    support_reliability: float
    geometry_reliability: float
    edge_confidences: Tuple[Tuple[float, float], ...] = ()
    alignment_reference_start: Optional[int] = None
    alignment_flag: Optional[int] = None
    alignment_cigar: Optional[str] = None
    alignment_record_sha256: Optional[str] = None
    alignment_occurrence: Optional[int] = None
    current_molecular_interval: Optional[Tuple[int, int]] = None
    current_annotation_ordinal: Optional[int] = None


@dataclass(frozen=True)
class GeometryDecision:
    decision_id: str
    read_name: str
    library_id: str
    current_interval: Tuple[int, int]
    canonical_interval: Tuple[int, int]
    assignment_probability: float
    molecule_probability: float
    population_probability: float
    geometry_reliability: float
    alternative_probability: float = 1.0
    left_edge_probability: float = 1.0
    right_edge_probability: float = 1.0
    call_type: str = "tf"
    alignment_reference_start: Optional[int] = None
    alignment_flag: Optional[int] = None
    alignment_cigar: Optional[str] = None
    alignment_record_sha256: Optional[str] = None
    alignment_occurrence: Optional[int] = None
    current_molecular_interval: Optional[Tuple[int, int]] = None
    current_annotation_ordinal: Optional[int] = None


def _bounded_probability(value: object, *, default: float = 0.5) -> float:
    if value is None:
        return default
    result = float(value)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"probability outside [0,1]: {result}")
    return result


def posterior_to_q(value: float) -> int:
    value = _bounded_probability(value)
    return max(0, min(255, int(round(255.0 * value))))


def collect_decisions(
    report: Mapping[str, object], *, minimum_posterior: float = 0.0
) -> List[StrandDecision]:
    """Convert the complete report decision surface into typed records."""
    schema = report.get("schema")
    if schema not in REPORT_SCHEMAS:
        raise ValueError("input is not a supported fiberhmm strand-rescue report")
    rescue = report.get("strand_rescue", {})
    if not isinstance(rescue, Mapping):
        raise ValueError("report has no strand_rescue object")
    decisions = []
    for raw in rescue.get("decisions", []):
        if schema in {
            "fiberhmm.strand_rescue.v4",
            "fiberhmm.strand_rescue.v6",
        }:
            alternative_value = raw.get(
                "sr_hypothesis_probability", raw.get("posterior")
            )
            if alternative_value is None:
                raise ValueError(
                    f"{schema} rescue decision has no alternative probability"
                )
            posterior = _bounded_probability(alternative_value)
            current = _bounded_probability(
                raw.get(
                    "baseline_hypothesis_probability",
                    raw.get("current_posterior"),
                ),
                default=1.0 - posterior,
            )
        else:
            if raw.get("posterior") is None or raw.get(
                "best_configuration_posterior_given_tf"
            ) is None:
                raise ValueError(
                    "legacy rescue decision lacks exact-configuration ingredients"
                )
            supported_tf = _bounded_probability(raw.get("posterior"))
            current = _bounded_probability(
                raw.get("current_posterior"), default=1.0 - supported_tf
            )
            configuration = _bounded_probability(
                raw.get("best_configuration_posterior_given_tf")
            )
            selected = supported_tf * configuration
            denominator = selected + current
            posterior = selected / denominator if denominator > 0.0 else 0.5
            current = 1.0 - posterior
        if posterior < minimum_posterior:
            continue
        if schema == "fiberhmm.strand_rescue.v6":
            other = _bounded_probability(
                raw.get("other_supported_tf_configuration_probability"),
                default=max(0.0, 1.0 - posterior - current),
            )
            unresolved = _bounded_probability(
                raw.get("unresolved_action_set_probability"), default=0.0
            )
            if not math.isclose(
                posterior + current + other + unresolved, 1.0, abs_tol=1e-8
            ):
                raise ValueError(
                    "v6 rescue action-set probabilities do not sum to one"
                )
        else:
            total = posterior + current
            if total <= 0.0:
                posterior = current = 0.5
            else:
                posterior /= total
                current = 1.0 - posterior
        tf_intervals = tuple(
            tuple(int(value) for value in interval)
            for interval in raw.get("proposed_site_intervals", [])
        )
        if not tf_intervals:
            continue
        raw_edge_confidences = raw.get("proposed_site_edge_confidence")
        if schema in {
            "fiberhmm.strand_rescue.v4",
            "fiberhmm.strand_rescue.v6",
        }:
            if not isinstance(raw_edge_confidences, list) or len(
                raw_edge_confidences
            ) != len(tf_intervals):
                raise ValueError(
                    "v4 rescue decision has no one-to-one edge confidences"
                )
            if any(
                not isinstance(value, (list, tuple)) or len(value) != 2
                for value in raw_edge_confidences
            ):
                raise ValueError(
                    "v4 rescue edge confidence must contain left and right"
                )
            edge_confidences = tuple(
                (
                    _bounded_probability(value[0]),
                    _bounded_probability(value[1]),
                )
                for value in raw_edge_confidences
            )
        else:
            fallback = _bounded_probability(
                raw.get("canonical_geometry_reliability"), default=0.0
            )
            edge_confidences = tuple(
                (fallback, fallback) for _interval in tf_intervals
            )
        decision = StrandDecision(
            decision_id=str(raw["decision_id"]),
            read_name=str(raw["read"]),
            library_id=str(raw.get("library_id") or ""),
            tier=str(raw.get("proposal_tier", "retain_current")),
            current_state=str(raw["current"]),
            current_interval=tuple(int(value) for value in raw["current_interval"]),
            tf_intervals=tf_intervals,
            tf_posterior=posterior,
            current_posterior=current,
            configuration_probability=_bounded_probability(
                raw.get("best_configuration_posterior_given_tf"), default=0.0
            ),
            molecule_probability=_bounded_probability(
                raw.get("molecule_probability"), default=0.5
            ),
            population_probability=_bounded_probability(
                raw.get("population_probability"), default=0.5
            ),
            support_reliability=_bounded_probability(
                raw.get("source_support_reliability"), default=0.0
            ),
            geometry_reliability=_bounded_probability(
                raw.get("canonical_geometry_reliability"), default=0.0
            ),
            edge_confidences=edge_confidences,
            alignment_reference_start=(
                int(raw.get("alignment", {}).get("reference_start"))
                if raw.get("alignment", {}).get("reference_start") is not None
                else None
            ),
            alignment_flag=(
                int(raw.get("alignment", {}).get("flag"))
                if raw.get("alignment", {}).get("flag") is not None
                else None
            ),
            alignment_cigar=(
                str(raw.get("alignment", {}).get("cigar"))
                if raw.get("alignment", {}).get("cigar") is not None
                else None
            ),
            alignment_record_sha256=(
                str(raw.get("alignment", {}).get("record_sha256"))
                if raw.get("alignment", {}).get("record_sha256") is not None
                else None
            ),
            alignment_occurrence=(
                int(raw.get("alignment", {}).get("occurrence"))
                if raw.get("alignment", {}).get("occurrence") is not None
                else None
            ),
            current_molecular_interval=(
                tuple(int(value) for value in raw["current_molecular_interval"])
                if raw.get("current_molecular_interval") is not None
                else None
            ),
            current_annotation_ordinal=(
                int(raw["current_annotation_ordinal"])
                if raw.get("current_annotation_ordinal") is not None
                else None
            ),
        )
        if decision.current_state != "A":
            raise ValueError("strand rescue accepts only MSP-origin decisions")
        decisions.append(decision)
    decisions.sort(
        key=lambda item: (
            item.library_id,
            item.read_name,
            item.current_interval,
            item.decision_id,
        )
    )
    return decisions


def collect_harmonizations(report: Mapping[str, object]) -> List[GeometryDecision]:
    schema = report.get("schema")
    if schema not in REPORT_SCHEMAS:
        raise ValueError("input is not a supported fiberhmm strand-rescue report")
    rescue = report.get("strand_rescue", {})
    if schema in {
        "fiberhmm.strand_rescue.v3",
        "fiberhmm.strand_rescue.v4",
        "fiberhmm.strand_rescue.v6",
    }:
        raw_values = [
            raw
            for call_type in ("tf", "nuc")
            for raw in rescue.get("edge_refinement", {})
            .get(call_type, {})
            .get("harmonizations", [])
        ]
        accepted_status = "edge_update"
    else:
        raw_values = rescue.get("geometry_harmonization", {}).get(
            "harmonizations", []
        )
        accepted_status = "geometry_update"
    decisions = []
    for raw in raw_values:
        if raw.get("status") != accepted_status:
            continue
        current_interval = tuple(int(value) for value in raw["current_interval"])
        canonical_interval = tuple(
            int(value) for value in raw["canonical_interval"]
        )
        if schema in {
            "fiberhmm.strand_rescue.v4",
            "fiberhmm.strand_rescue.v6",
        }:
            hypothesis = raw.get("edge_hypothesis")
            if not isinstance(hypothesis, Mapping):
                raise ValueError("edge refinement has no edge_hypothesis")
            left_hypothesis = hypothesis.get("left")
            right_hypothesis = hypothesis.get("right")
            if not isinstance(left_hypothesis, Mapping) or not isinstance(
                right_hypothesis, Mapping
            ):
                raise ValueError("edge hypothesis lacks per-edge values")
            alternative_probability = _bounded_probability(
                hypothesis.get("alternative_probability")
            )
            left_edge_probability = _bounded_probability(
                left_hypothesis.get("alternative_probability")
            )
            right_edge_probability = _bounded_probability(
                right_hypothesis.get("alternative_probability")
            )
            materialized_edge_confidence = raw.get("materialized_edge_confidence")
            if materialized_edge_confidence is not None:
                if (
                    not isinstance(materialized_edge_confidence, (list, tuple))
                    or len(materialized_edge_confidence) != 2
                ):
                    raise ValueError(
                        "edge refinement has invalid materialized_edge_confidence"
                    )
                left_edge_probability = _bounded_probability(
                    materialized_edge_confidence[0]
                )
                right_edge_probability = _bounded_probability(
                    materialized_edge_confidence[1]
                )
        else:
            alternative_probability = 1.0
            fallback = _bounded_probability(
                raw.get("geometry_reliability"), default=0.0
            )
            left_edge_probability = (
                1.0
                if current_interval[0] == canonical_interval[0]
                else fallback
            )
            right_edge_probability = (
                1.0
                if current_interval[1] == canonical_interval[1]
                else fallback
            )
        decision = GeometryDecision(
            decision_id=str(raw["decision_id"]),
            read_name=str(raw["read"]),
            library_id=str(raw.get("library_id") or ""),
            current_interval=current_interval,
            canonical_interval=canonical_interval,
            assignment_probability=_bounded_probability(
                raw.get("assignment_probability"), default=0.0
            ),
            molecule_probability=_bounded_probability(
                raw.get("molecule_probability"), default=0.5
            ),
            population_probability=_bounded_probability(
                raw.get("population_probability"), default=0.5
            ),
            geometry_reliability=_bounded_probability(
                raw.get("geometry_reliability"), default=0.0
            ),
            alternative_probability=alternative_probability,
            left_edge_probability=left_edge_probability,
            right_edge_probability=right_edge_probability,
            call_type=str(raw.get("call_type", "tf")),
            alignment_reference_start=(
                int(raw.get("alignment", {}).get("reference_start"))
                if raw.get("alignment", {}).get("reference_start") is not None
                else None
            ),
            alignment_flag=(
                int(raw.get("alignment", {}).get("flag"))
                if raw.get("alignment", {}).get("flag") is not None
                else None
            ),
            alignment_cigar=(
                str(raw.get("alignment", {}).get("cigar"))
                if raw.get("alignment", {}).get("cigar") is not None
                else None
            ),
            alignment_record_sha256=(
                str(raw.get("alignment", {}).get("record_sha256"))
                if raw.get("alignment", {}).get("record_sha256") is not None
                else None
            ),
            alignment_occurrence=(
                int(raw.get("alignment", {}).get("occurrence"))
                if raw.get("alignment", {}).get("occurrence") is not None
                else None
            ),
            current_molecular_interval=(
                tuple(int(value) for value in raw["current_molecular_interval"])
                if raw.get("current_molecular_interval") is not None
                else None
            ),
            current_annotation_ordinal=(
                int(raw["current_annotation_ordinal"])
                if raw.get("current_annotation_ordinal") is not None
                else None
            ),
        )
        if decision.call_type not in {"tf", "nuc"}:
            raise ValueError(f"unsupported edge call type: {decision.call_type}")
        decisions.append(decision)
    decisions.sort(
        key=lambda item: (
            item.library_id,
            item.read_name,
            item.current_interval,
            item.decision_id,
        )
    )
    return decisions


def rescue_quality_row(
    decision: StrandDecision,
    component_index: int = 0,
    *,
    reverse: bool = False,
) -> Tuple[int, ...]:
    if decision.edge_confidences:
        try:
            left, right = decision.edge_confidences[component_index]
        except IndexError as error:
            raise ValueError("rescue component has no edge confidence") from error
    else:
        left = right = decision.geometry_reliability
    if reverse:
        left, right = right, left
    return (
        posterior_to_q(decision.tf_posterior),
        posterior_to_q(left),
        posterior_to_q(right),
    )


def geometry_quality_row(
    decision: GeometryDecision, *, reverse: bool = False
) -> Tuple[int, ...]:
    left = decision.left_edge_probability
    right = decision.right_edge_probability
    if reverse:
        left, right = right, left
    return (
        posterior_to_q(decision.alternative_probability),
        posterior_to_q(left),
        posterior_to_q(right),
    )


def _read_length(read) -> int:
    length = read.query_length
    if not length and read.query_sequence:
        length = len(read.query_sequence)
    if not length and read.has_tag("MA"):
        try:
            length = int(str(read.get_tag("MA")).split(";", 1)[0])
        except (TypeError, ValueError):
            length = 0
    if not length:
        infer = getattr(read, "infer_read_length", None)
        if callable(infer):
            length = infer()
    if not length:
        raise ValueError(f"read {read.query_name!r} has no query length")
    return int(length)


def reference_interval_to_molecular(
    read, start: int, end: int
) -> Optional[Tuple[int, int]]:
    """Project a reference half-open interval to 0-based molecular MA space."""
    if end <= start:
        raise ValueError(f"invalid reference interval: {start}-{end}")
    reference_start = getattr(read, "reference_start", None)
    reference_end = getattr(read, "reference_end", None)
    if reference_start is not None and reference_end is not None and not (
        int(reference_start) <= start < end <= int(reference_end)
    ):
        return None
    reference_positions = read.get_reference_positions(full_length=True)
    mapped_positions = {
        int(reference_position)
        for reference_position in reference_positions
        if reference_position is not None
    }
    if start not in mapped_positions or end - 1 not in mapped_positions:
        return None
    mapped_fraction = sum(
        start <= position < end for position in mapped_positions
    ) / (end - start)
    if mapped_fraction < 0.95:
        return None
    query_positions = [
        index
        for index, reference_position in enumerate(reference_positions)
        if reference_position is not None and start <= int(reference_position) < end
    ]
    if not query_positions:
        return None
    sequence_start = min(query_positions)
    length = max(query_positions) + 1 - sequence_start
    if read.is_reverse:
        return flip_interval_frame(sequence_start, length, _read_length(read))
    return int(sequence_start), int(length)


def _format_ma(
    read_length: int,
    groups: Sequence[
        Tuple[str, str, str, Sequence[Tuple[int, int]], Sequence[Sequence[int]]]
    ],
) -> Tuple[str, array.array]:
    parts = [str(int(read_length))]
    qualities = array.array("B")
    for name, strand, quality_spec, intervals, quality_values in groups:
        if not intervals:
            continue
        if len(intervals) != len(quality_values):
            raise ValueError(f"{name} interval/quality count mismatch")
        tokens = []
        for (start, length), row in zip(intervals, quality_values):
            if start < 0 or length <= 0 or start + length > read_length:
                raise ValueError(
                    f"{name} interval outside read: {start}+{length}/{read_length}"
                )
            if len(row) != len(quality_spec):
                raise ValueError(f"{name} quality arity mismatch")
            tokens.append(f"{int(start) + 1}-{int(length)}")
            qualities.extend(max(0, min(255, int(value))) for value in row)
        parts.append(f"{name}{strand}{quality_spec}:" + ",".join(tokens))
    return ";".join(parts), qualities


def _matching_interval(
    intervals: Sequence[Tuple[int, int]], projected: Tuple[int, int]
) -> Optional[Tuple[int, int]]:
    if projected in intervals:
        return projected
    projected_start, projected_length = projected
    projected_end = projected_start + projected_length
    candidates = []
    for interval in intervals:
        start, length = interval
        end = start + length
        overlap = max(0, min(end, projected_end) - max(start, projected_start))
        union = max(end, projected_end) - min(start, projected_start)
        if overlap:
            candidates.append((overlap / union, overlap, interval))
    if not candidates:
        return None
    jaccard, _overlap, interval = max(candidates)
    return interval if len(candidates) == 1 and jaccard >= 0.8 else None


def _alignment_matches(
    read, decision: object, alignment_occurrence: Optional[int]
) -> bool:
    reference_start = getattr(decision, "alignment_reference_start", None)
    flag = getattr(decision, "alignment_flag", None)
    cigar = getattr(decision, "alignment_cigar", None)
    record_sha256 = getattr(decision, "alignment_record_sha256", None)
    occurrence = getattr(decision, "alignment_occurrence", None)
    if reference_start is not None and int(read.reference_start) != reference_start:
        return False
    if flag is not None and int(read.flag) != flag:
        return False
    if cigar is not None and str(read.cigarstring) != cigar:
        return False
    if record_sha256 is not None:
        to_string = getattr(read, "to_string", None)
        if not callable(to_string):
            return False
        observed = hashlib.sha256(to_string().encode("utf-8")).hexdigest()
        if observed != record_sha256:
            return False
    if occurrence is not None and alignment_occurrence != occurrence:
        return False
    return True


def _current_molecular_annotation(
    read,
    decision: object,
    originals: Sequence[Tuple[Tuple[int, int], Sequence[int]]],
) -> Optional[Tuple[int, Tuple[int, int]]]:
    exact = getattr(decision, "current_molecular_interval", None)
    ordinal = getattr(decision, "current_annotation_ordinal", None)
    if exact is not None:
        exact = tuple(int(value) for value in exact)
        if ordinal is not None:
            if not 0 <= ordinal < len(originals):
                return None
            return (ordinal, exact) if tuple(originals[ordinal][0]) == exact else None
        matches = [
            (index, exact)
            for index, value in enumerate(originals)
            if tuple(value[0]) == exact
        ]
        return matches[0] if len(matches) == 1 else None
    projected = reference_interval_to_molecular(read, *decision.current_interval)
    if projected is None:
        return None
    matched = _matching_interval(
        [tuple(value[0]) for value in originals], projected
    )
    if matched is None:
        return None
    matches = [
        (index, matched)
        for index, value in enumerate(originals)
        if tuple(value[0]) == matched
    ]
    return matches[0] if len(matches) == 1 else None


def _current_molecular_interval(
    read,
    decision: object,
    originals: Sequence[Tuple[Tuple[int, int], Sequence[int]]],
) -> Optional[Tuple[int, int]]:
    result = _current_molecular_annotation(read, decision, originals)
    return result[1] if result is not None else None


def _molecular_intervals_overlap(
    left: Tuple[int, int], right: Tuple[int, int]
) -> bool:
    return left[0] < right[0] + right[1] and right[0] < left[0] + left[1]


def add_strand_rescue_groups(
    read,
    decisions: Sequence[StrandDecision],
    harmonizations: Sequence[GeometryDecision] = (),
    *,
    stream_rescues: Sequence[StreamRescueAction] = (),
    stream_edges: Sequence[StreamEdgeAction] = (),
    alignment_occurrence: Optional[int] = None,
    diagnostics: Optional[DefaultDict[str, int]] = None,
    applied_decision_ids: Optional[set] = None,
    applied_harmonization_ids: Optional[set] = None,
) -> Dict[str, int]:
    """Replace stale SR groups and add complete normalized nuc/TF layers."""
    if not read.has_tag("MA"):
        if stream_rescues or stream_edges:
            raise ValueError("streamed action targets a BAM record with no MA tag")
        return {}
    read_length = _read_length(read)
    parsed = parse_ma_tag(read.get_tag("MA"))
    if int(parsed["read_length"]) != read_length:
        raise ValueError(
            f"read {read.query_name!r}: MA/query length mismatch"
        )
    raw_types = parsed["raw_types"]
    aq = read.get_tag("AQ") if read.has_tag("AQ") else []
    expected_aq = sum(
        len(specification) * len(intervals)
        for _name, _strand, specification, intervals in raw_types
    )
    if len(aq) != expected_aq:
        raise ValueError(
            f"read {read.query_name!r}: AQ has {len(aq)} bytes; MA needs {expected_aq}"
        )
    per_annotation = parse_aq_array(
        aq,
        [item[2] for item in raw_types],
        [len(item[3]) for item in raw_types],
    )
    had_an = read.has_tag("AN")
    old_names = parse_an_tag(read.get_tag("AN")) if had_an else []
    annotation_count = sum(len(item[3]) for item in raw_types)
    if had_an and len(old_names) != annotation_count:
        raise ValueError(
            f"read {read.query_name!r}: AN has {len(old_names)} names; "
            f"MA has {annotation_count} annotations"
        )

    preserved_groups = []
    preserved_names: List[str] = []
    original: Dict[str, List[Tuple[Tuple[int, int], Sequence[int]]]] = {
        "nuc": [],
        "tf": [],
        "msp": [],
    }
    cursor = 0
    for name, strand, specification, intervals in raw_types:
        count = len(intervals)
        rows = per_annotation[cursor : cursor + count]
        names = old_names[cursor : cursor + count]
        cursor += count
        if name in original:
            original[name].extend(zip(intervals, rows))
        if name in REPLACED_LAYER_NAMES:
            continue
        preserved_groups.append((name, strand, specification, list(intervals), rows))
        preserved_names.extend(names + [""] * (count - len(names)))

    baseline_row = (255, 0, 0)
    layer_values: Dict[
        str,
        Dict[
            Tuple[str, object],
            Tuple[Tuple[int, int], Tuple[int, ...], str, str],
        ],
    ] = {name: {} for name in LAYER_ORDER}
    for call_type, layer_name in (("nuc", "nuc_sr"), ("tf", "tf_sr")):
        for ordinal, (interval, _row) in enumerate(original[call_type]):
            layer_values[layer_name][("B", ordinal)] = (
                tuple(interval),
                baseline_row,
                "",
                "baseline",
            )

    candidates = {}
    stream_candidate_keys = set()
    for decision in sorted(
        harmonizations,
        key=lambda value: (
            value.call_type,
            value.current_interval,
            value.decision_id,
        ),
    ):
        if diagnostics is not None:
            diagnostics["harmonizations_seen"] += 1
        if not _alignment_matches(read, decision, alignment_occurrence):
            if diagnostics is not None:
                diagnostics["harmonization_alignment_mismatch"] += 1
            continue
        layer_name = f"{decision.call_type}_sr"
        current_annotation = _current_molecular_annotation(
            read, decision, original[decision.call_type]
        )
        canonical = reference_interval_to_molecular(
            read, *decision.canonical_interval
        )
        if current_annotation is None or canonical is None:
            if diagnostics is not None:
                diagnostics["harmonization_unprojectable"] += 1
            continue
        source_ordinal, current = current_annotation
        source_key = ("B", source_ordinal)
        if source_key not in layer_values[layer_name]:
            if diagnostics is not None:
                diagnostics["harmonization_current_unmatched"] += 1
            continue
        key = (layer_name, source_key)
        if key in candidates:
            if diagnostics is not None:
                diagnostics["harmonization_duplicate_source"] += 1
            continue
        token = hashlib.sha256(
            (
                f"sr-edge:{decision.call_type}:{decision.library_id}:"
                f"{decision.decision_id}"
            ).encode("utf-8")
        ).hexdigest()[:16]
        candidates[key] = {
            "decision": decision,
            "current": current,
            "canonical": canonical,
            "value": (
                geometry_quality_row(decision, reverse=bool(read.is_reverse)),
                f"fhsr_{token}_O{source_ordinal}_H",
                "harmonized",
            ),
        }

    for action in stream_edges:
        layer_name = f"{action.call_type}_sr"
        originals = original[action.call_type]
        if action.alternative_interval[0] + action.alternative_interval[1] > read_length:
            raise ValueError("streamed edge alternative lies outside the read")
        if not 0 <= action.source_ordinal < len(originals):
            raise ValueError(
                f"streamed {action.call_type} edge source ordinal is out of range"
            )
        observed_source = tuple(originals[action.source_ordinal][0])
        if observed_source != action.source_interval:
            raise ValueError(
                f"streamed {action.call_type} edge source interval differs from MA"
            )
        source_key = ("B", action.source_ordinal)
        key = (layer_name, source_key)
        if key in candidates:
            raise ValueError("multiple edge actions target one ordinary annotation")
        candidates[key] = {
            "decision": None,
            "current": action.source_interval,
            "canonical": action.alternative_interval,
            "value": (
                action.quality,
                f"fhsr_{action.token}_O{action.source_ordinal}_H",
                "harmonized",
            ),
            "stream_call_type": action.call_type,
        }
        stream_candidate_keys.add(key)

    valid = set(candidates)
    baseline_objects = [
        (layer_name, source_key)
        for layer_name in LAYER_ORDER
        for source_key in layer_values[layer_name]
    ]
    while valid:
        rejected = set()
        final_intervals = {
            key: (
                candidates[key]["canonical"]
                if key in valid
                else layer_values[key[0]][key[1]][0]
            )
            for key in baseline_objects
        }
        for left_index, left_key in enumerate(baseline_objects):
            for right_key in baseline_objects[left_index + 1 :]:
                new_overlap = _molecular_intervals_overlap(
                    final_intervals[left_key], final_intervals[right_key]
                )
                old_overlap = _molecular_intervals_overlap(
                    layer_values[left_key[0]][left_key[1]][0],
                    layer_values[right_key[0]][right_key[1]][0],
                )
                if new_overlap and not old_overlap:
                    rejected.update(
                        key for key in (left_key, right_key) if key in valid
                    )
        for layer_name in LAYER_ORDER:
            ordered = sorted(
                (key for key in baseline_objects if key[0] == layer_name),
                key=lambda key: (layer_values[key[0]][key[1]][0], key[1]),
            )
            for left_key, right_key in zip(ordered, ordered[1:]):
                if final_intervals[left_key][0] > final_intervals[right_key][0]:
                    rejected.update(
                        key for key in (left_key, right_key) if key in valid
                    )
        if not rejected:
            break
        valid.difference_update(rejected)
        if diagnostics is not None:
            diagnostics["harmonization_joint_collision"] += len(rejected)

    rejected_stream = stream_candidate_keys - valid
    if rejected_stream:
        raise ValueError(
            "streamed edge action violates normalized-layer topology"
        )

    for key in sorted(valid):
        layer_name, source_key = key
        candidate = candidates[key]
        canonical = candidate["canonical"]
        row, name, kind = candidate["value"]
        layer_values[layer_name][source_key] = (canonical, row, name, kind)
        decision = candidate["decision"]
        if decision is not None:
            if applied_harmonization_ids is not None:
                applied_harmonization_ids.add(decision.decision_id)
            call_type = decision.call_type
        else:
            call_type = candidate["stream_call_type"]
        if diagnostics is not None:
            diagnostics[f"{call_type}_intervals_harmonized"] += 1

    for decision in decisions:
        if diagnostics is not None:
            diagnostics["decisions_seen"] += 1
        if not _alignment_matches(read, decision, alignment_occurrence):
            if diagnostics is not None:
                diagnostics["decision_alignment_mismatch"] += 1
            continue
        current = _current_molecular_interval(read, decision, original["msp"])
        if current is None:
            if diagnostics is not None:
                diagnostics["current_unprojectable"] += 1
            continue
        projected_tfs = []
        projection_failed = False
        for reference_interval in decision.tf_intervals:
            interval = reference_interval_to_molecular(read, *reference_interval)
            if interval is None or interval in projected_tfs:
                projection_failed = True
                break
            projected_tfs.append(interval)
        current_start, current_length = current
        current_end = current_start + current_length
        if any(
            interval[0] < current_start
            or interval[0] + interval[1] > current_end
            for interval in projected_tfs
        ):
            projection_failed = True
        if projection_failed or len(projected_tfs) != len(decision.tf_intervals):
            if diagnostics is not None:
                diagnostics["tf_configuration_unprojectable"] += 1
            continue

        token = hashlib.sha256(
            f"sr:{decision.library_id}:{decision.decision_id}".encode("utf-8")
        ).hexdigest()[:16]
        prefix = f"fhsr_{token}"
        # Keep rescue groups atomic. A collision should be impossible for a
        # correctly blocked report, but never replace an accepted baseline TF
        # or materialize only part of a multi-TF alternative if it occurs.
        if any(
            _molecular_intervals_overlap(left, right)
            for index, left in enumerate(projected_tfs)
            for right in projected_tfs[index + 1 :]
        ) or any(
            _molecular_intervals_overlap(interval, existing)
            for interval in projected_tfs
            for layer_name in LAYER_ORDER
            for existing, _row, _name, _kind in layer_values[layer_name].values()
        ):
            if diagnostics is not None:
                diagnostics["rescue_collision"] += 1
            continue
        for index, interval in enumerate(projected_tfs):
            tf_row = rescue_quality_row(
                decision, index, reverse=bool(read.is_reverse)
            )
            layer_values["tf_sr"][("R", f"{decision.decision_id}:{index}")] = (
                interval,
                tf_row,
                f"{prefix}_R{index}",
                "rescued",
            )
        if applied_decision_ids is not None:
            applied_decision_ids.add(decision.decision_id)
        if diagnostics is not None:
            diagnostics["tf_intervals_added"] += len(projected_tfs)

    for action in stream_rescues:
        if not 0 <= action.source_ordinal < len(original["msp"]):
            raise ValueError("streamed rescue source MSP ordinal is out of range")
        observed_source = tuple(original["msp"][action.source_ordinal][0])
        if observed_source != action.source_interval:
            raise ValueError("streamed rescue source MSP interval differs from MA")
        projected_tfs = [component.interval for component in action.components]
        if any(
            _molecular_intervals_overlap(left, right)
            for index, left in enumerate(projected_tfs)
            for right in projected_tfs[index + 1 :]
        ) or any(
            _molecular_intervals_overlap(interval, existing)
            for interval in projected_tfs
            for layer_name in LAYER_ORDER
            for existing, _row, _name, _kind in layer_values[layer_name].values()
        ):
            raise ValueError("streamed rescue violates normalized-layer topology")
        for component in action.components:
            layer_values["tf_sr"][("R", f"{action.token}:{component.component_index}")] = (
                component.interval,
                (action.alternative_q, component.left_q, component.right_q),
                f"fhsr_{action.token}_R{component.component_index}",
                "rescued",
            )
        if diagnostics is not None:
            diagnostics["tf_intervals_added"] += len(action.components)

    new_names: List[str] = []
    counts = {}
    for layer_name in LAYER_ORDER:
        records = sorted(
            layer_values[layer_name].items(),
            key=lambda item: (item[1][0], str(item[0])),
        )
        if not records:
            continue
        intervals = [value[0] for _key, value in records]
        rows = [list(value[1]) for _key, value in records]
        preserved_groups.append((layer_name, ".", QUALITY_SPEC, intervals, rows))
        new_names.extend(value[2] for _key, value in records)
        counts[layer_name] = len(records)

    ma_value, aq_value = _format_ma(read_length, preserved_groups)
    read.set_tag("MA", ma_value, value_type="Z")
    if aq_value:
        read.set_tag("AQ", aq_value)
    elif read.has_tag("AQ"):
        read.set_tag("AQ", None)
    names = preserved_names + new_names
    if had_an or any(names):
        read.set_tag("AN", format_an_tag(names), value_type="Z")
    elif read.has_tag("AN"):
        read.set_tag("AN", None)
    return counts


def _canonical_path(path: str) -> str:
    return str(Path(path).expanduser().resolve())


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class ResolvedV5ActionStream:
    input_index: int
    input_path: str
    manifest: Mapping[str, object]
    bgzf_path: Path
    gzi_path: Path


_V5_STORAGE_KEYS = (
    "layout",
    "stream_schema",
    "quality_encoding",
    "coordinate_frame",
    "streams",
    "totals",
)
_V5_TOTAL_KEYS = (
    "fetch_records",
    "action_records",
    "rescue_decisions",
    "rescue_components",
    "tf_edge_updates",
    "nuc_edge_updates",
)
_V5_STREAM_KEYS = (
    "input_index",
    "input_id",
    "path",
    "gzi_path",
    "loaded_region",
    "fetch_record_count",
    "action_record_count",
    "first_action_ordinal",
    "last_action_ordinal",
    "rescue_decision_count",
    "rescue_component_count",
    "tf_edge_update_count",
    "nuc_edge_update_count",
    "compressed_size_bytes",
    "uncompressed_size_bytes",
    "bgzf_sha256",
    "jsonl_sha256",
    "gzi_size_bytes",
    "gzi_sha256",
)
_V5_TOTAL_TO_STREAM_COUNT = {
    "fetch_records": "fetch_record_count",
    "action_records": "action_record_count",
    "rescue_decisions": "rescue_decision_count",
    "rescue_components": "rescue_component_count",
    "tf_edge_updates": "tf_edge_update_count",
    "nuc_edge_updates": "nuc_edge_update_count",
}


def _resolve_v5_relative_path(
    report_directory: Path, raw_path: object, context: str
) -> Path:
    if type(raw_path) is not str or not raw_path:
        raise ValueError(f"{context} must be a non-empty relative path")
    relative = Path(raw_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{context} must not be absolute or contain '..'")
    base = report_directory.resolve()
    resolved = (base / relative).resolve()
    try:
        resolved.relative_to(base)
    except ValueError as error:
        raise ValueError(f"{context} resolves outside the report directory") from error
    return resolved


def _validate_v5_action_storage(
    report: Mapping[str, object], report_path: Path
) -> List[ResolvedV5ActionStream]:
    """Validate a streamed v5/v6 manifest without loading its actions."""
    schema = report.get("schema")
    expected_version = STREAM_REPORT_VERSIONS.get(schema)
    if expected_version is None:
        raise ValueError("input is not a streamed fiberhmm strand-rescue report")
    if (
        _require_int(report.get("schema_version"), "schema_version")
        != expected_version
    ):
        raise ValueError(
            f"strand-rescue v{expected_version} report must have "
            f"schema_version {expected_version}"
        )
    raw_input = report.get("input")
    if not isinstance(raw_input, Mapping):
        raise ValueError("v5 report has no input object")
    raw_bams = raw_input.get("bams")
    raw_files = raw_input.get("files")
    if not isinstance(raw_bams, list) or not raw_bams:
        raise ValueError("v5 report input.bams must be a non-empty list")
    if not isinstance(raw_files, list) or len(raw_files) != len(raw_bams):
        raise ValueError("v5 report input.files must align one-to-one with input.bams")
    report_region = _parse_stream_region(
        raw_input.get("loaded_region"), "report input.loaded_region"
    )
    input_paths = []
    for index, (bam, metadata) in enumerate(zip(raw_bams, raw_files)):
        if type(bam) is not str or not Path(bam).is_absolute():
            raise ValueError(f"input.bams[{index}] must be an absolute path")
        if (
            not isinstance(metadata, Mapping)
            or type(metadata.get("path")) is not str
            or not Path(metadata["path"]).is_absolute()
        ):
            raise ValueError(f"input.files[{index}] must contain a string path")
        _require_int(metadata.get("size_bytes"), f"input.files[{index}].size_bytes")
        _require_int(metadata.get("mtime_ns"), f"input.files[{index}].mtime_ns")
        canonical = _canonical_path(bam)
        if _canonical_path(metadata["path"]) != canonical:
            raise ValueError(f"input.files[{index}] path differs from input.bams")
        input_paths.append(canonical)
    if len(input_paths) != len(set(input_paths)):
        raise ValueError("v5 report input.bams contains duplicate paths")

    rescue = report.get("strand_rescue")
    if not isinstance(rescue, Mapping):
        raise ValueError("v5 report has no strand_rescue object")
    if "decisions" in rescue:
        raise ValueError("v5 report mixes streamed and inline rescue decisions")
    edge_refinement = rescue.get("edge_refinement", {})
    if not isinstance(edge_refinement, Mapping):
        raise ValueError("v5 strand_rescue.edge_refinement must be an object")
    for call_type in ("tf", "nuc"):
        section = edge_refinement.get(call_type, {})
        if not isinstance(section, Mapping):
            raise ValueError(f"v5 edge_refinement.{call_type} must be an object")
        if "harmonizations" in section:
            raise ValueError("v5 report mixes streamed and inline edge actions")

    storage = rescue.get("action_storage")
    if not isinstance(storage, Mapping):
        raise ValueError("v5 report has no strand_rescue.action_storage manifest")
    _require_exact_keys(storage, _V5_STORAGE_KEYS, "v5 action_storage")
    if storage["layout"] != V5_ACTION_LAYOUT:
        raise ValueError("unsupported v5 action-storage layout")
    if storage["stream_schema"] != V5_ACTION_STREAM_SCHEMA:
        raise ValueError("unsupported v5 action-stream schema")
    if storage["quality_encoding"] != "uint8_round_255_times_unit_probability":
        raise ValueError("unsupported v5 action quality encoding")
    if storage["coordinate_frame"] != "molecular_zero_based_start_length":
        raise ValueError("unsupported v5 action coordinate frame")
    raw_streams = storage["streams"]
    if not isinstance(raw_streams, list) or len(raw_streams) != len(input_paths):
        raise ValueError("v5 action streams must align one-to-one with input.bams")
    totals = storage["totals"]
    if not isinstance(totals, Mapping):
        raise ValueError("v5 action_storage.totals must be an object")
    _require_exact_keys(totals, _V5_TOTAL_KEYS, "v5 action_storage.totals")

    report_directory = report_path.parent
    resolved_streams = []
    seen_indices = set()
    seen_paths = set()
    for stream_position, manifest in enumerate(raw_streams):
        context = f"v5 action_storage.streams[{stream_position}]"
        if not isinstance(manifest, Mapping):
            raise ValueError(f"{context} must be an object")
        _require_exact_keys(manifest, _V5_STREAM_KEYS, context)
        input_index = _require_int(manifest["input_index"], f"{context}.input_index")
        if input_index >= len(input_paths) or input_index in seen_indices:
            raise ValueError(f"{context}.input_index is duplicate or out of range")
        seen_indices.add(input_index)
        expected_id = f"input{input_index:04d}"
        if type(manifest["input_id"]) is not str or manifest["input_id"] != expected_id:
            raise ValueError(f"{context}.input_id must equal {expected_id}")
        if (
            _parse_stream_region(
                manifest["loaded_region"], f"{context}.loaded_region"
            )
            != report_region
        ):
            raise ValueError(f"{context}.loaded_region differs from the main report")

        counts = {
            key: _require_int(manifest[key], f"{context}.{key}")
            for key in V5ActionStreamReader._COUNT_KEYS
        }
        action_count = counts["action_record_count"]
        fetch_count = counts["fetch_record_count"]
        first = manifest["first_action_ordinal"]
        last = manifest["last_action_ordinal"]
        payload_count = (
            counts["rescue_decision_count"]
            + counts["tf_edge_update_count"]
            + counts["nuc_edge_update_count"]
        )
        if action_count == 0:
            if first is not None or last is not None or payload_count != 0:
                raise ValueError(
                    f"{context} empty stream must have null ordinals and zero actions"
                )
        else:
            first = _require_int(first, f"{context}.first_action_ordinal")
            last = _require_int(last, f"{context}.last_action_ordinal")
            if (
                payload_count == 0
                or action_count > fetch_count
                or action_count > last - first + 1
                or not (0 <= first <= last < fetch_count)
            ):
                raise ValueError(
                    f"{context} action ordinals are inconsistent with fetch count"
                )
        if (
            counts["rescue_component_count"] < counts["rescue_decision_count"]
            or (
                counts["rescue_decision_count"] == 0
                and counts["rescue_component_count"] != 0
            )
        ):
            raise ValueError(f"{context} has fewer rescue components than decisions")

        compressed_size = _require_int(
            manifest["compressed_size_bytes"],
            f"{context}.compressed_size_bytes",
            minimum=1,
        )
        _require_int(
            manifest["uncompressed_size_bytes"],
            f"{context}.uncompressed_size_bytes",
            minimum=1,
        )
        _require_int(
            manifest["gzi_size_bytes"], f"{context}.gzi_size_bytes", minimum=1
        )
        _require_sha256(manifest["bgzf_sha256"], f"{context}.bgzf_sha256")
        jsonl_sha = _require_sha256(
            manifest["jsonl_sha256"], f"{context}.jsonl_sha256"
        )
        _require_sha256(manifest["gzi_sha256"], f"{context}.gzi_sha256")
        bgzf_path = _resolve_v5_relative_path(
            report_directory, manifest["path"], f"{context}.path"
        )
        gzi_path = _resolve_v5_relative_path(
            report_directory, manifest["gzi_path"], f"{context}.gzi_path"
        )
        if gzi_path == bgzf_path or str(manifest["gzi_path"]) != f"{manifest['path']}.gzi":
            raise ValueError(f"{context}.gzi_path must be path plus '.gzi'")
        if not str(manifest["path"]).endswith(
            f".{jsonl_sha[:12]}.jsonl.bgz"
        ):
            raise ValueError(f"{context}.path has the wrong content-address token")
        if compressed_size < len(_BGZF_EOF):
            raise ValueError(f"{context}.compressed_size_bytes is too small for BGZF")
        for path in (bgzf_path, gzi_path):
            if path in seen_paths:
                raise ValueError("v5 action streams reuse a sidecar path")
            seen_paths.add(path)
        resolved_streams.append(
            ResolvedV5ActionStream(
                input_index=input_index,
                input_path=input_paths[input_index],
                manifest=manifest,
                bgzf_path=bgzf_path,
                gzi_path=gzi_path,
            )
        )

    if seen_indices != set(range(len(input_paths))):
        raise ValueError("v5 action streams do not cover every input index")
    for total_key, stream_key in _V5_TOTAL_TO_STREAM_COUNT.items():
        observed = sum(int(stream.manifest[stream_key]) for stream in resolved_streams)
        expected = _require_int(totals[total_key], f"v5 totals.{total_key}")
        if observed != expected:
            raise ValueError(f"v5 total {total_key} differs from stream manifests")
    return sorted(resolved_streams, key=lambda stream: stream.input_index)


def _validate_standard_gzi(path: Path, *, compressed_size: int) -> None:
    with path.open("rb") as handle:
        raw_count = handle.read(8)
        if len(raw_count) != 8:
            raise ValueError("action-stream GZI is shorter than its entry count")
        count = int.from_bytes(raw_count, "little", signed=False)
        if path.stat().st_size != 8 + 16 * count:
            raise ValueError("action-stream GZI has a non-standard byte length")
        previous_compressed = 0
        previous_uncompressed = 0
        for _index in range(count):
            raw_entry = handle.read(16)
            if len(raw_entry) != 16:
                raise ValueError("action-stream GZI ended inside an index entry")
            compressed = int.from_bytes(raw_entry[:8], "little", signed=False)
            uncompressed = int.from_bytes(
                raw_entry[8:], "little", signed=False
            )
            if (
                compressed <= previous_compressed
                or compressed >= compressed_size
                or uncompressed <= previous_uncompressed
            ):
                raise ValueError(
                    "action-stream GZI offsets are not strictly increasing"
                )
            previous_compressed = compressed
            previous_uncompressed = uncompressed


def _validate_v5_sidecar_files(stream: ResolvedV5ActionStream) -> None:
    """Validate immutable compressed and GZI bytes before BAM staging begins."""
    manifest = stream.manifest
    for path, label in ((stream.bgzf_path, "BGZF"), (stream.gzi_path, "GZI")):
        if not path.is_file():
            raise ValueError(f"missing action-stream {label}: {path}")
    compressed_size = int(manifest["compressed_size_bytes"])
    if stream.bgzf_path.stat().st_size != compressed_size:
        raise ValueError("action-stream BGZF size differs from manifest")
    if stream.gzi_path.stat().st_size != int(manifest["gzi_size_bytes"]):
        raise ValueError("action-stream GZI size differs from manifest")
    if _sha256_file(stream.bgzf_path) != manifest["bgzf_sha256"]:
        raise ValueError("action-stream BGZF SHA-256 mismatch")
    if _sha256_file(stream.gzi_path) != manifest["gzi_sha256"]:
        raise ValueError("action-stream GZI SHA-256 mismatch")
    with stream.bgzf_path.open("rb") as handle:
        handle.seek(-len(_BGZF_EOF), os.SEEK_END)
        if handle.read() != _BGZF_EOF:
            raise ValueError("action-stream BGZF has no standard EOF block")
    _validate_standard_gzi(stream.gzi_path, compressed_size=compressed_size)


def _report_inputs(report: Mapping[str, object]) -> List[str]:
    return list(
        dict.fromkeys(
            _canonical_path(str(path))
            for path in report.get("input", {}).get("bams", [])
        )
    )


def _proposal_index(
    decisions: Sequence[object], report_inputs: Sequence[str]
) -> Dict[str, DefaultDict[str, List[object]]]:
    result: Dict[str, DefaultDict[str, List[object]]] = {
        path: defaultdict(list) for path in report_inputs
    }
    basenames: DefaultDict[str, List[str]] = defaultdict(list)
    seen: DefaultDict[Tuple[str, str], set[str]] = defaultdict(set)
    for path in report_inputs:
        basenames[Path(path).name].append(path)
    for decision in decisions:
        if decision.library_id:
            resolved = _canonical_path(decision.library_id)
            if resolved not in result:
                matches = basenames.get(Path(decision.library_id).name, [])
                if len(matches) != 1:
                    raise ValueError(
                        "decision library is not uniquely present in report inputs: "
                        + decision.library_id
                    )
                resolved = matches[0]
        elif len(report_inputs) == 1:
            resolved = report_inputs[0]
        else:
            raise ValueError(
                f"decision {decision.decision_id} lacks library_id in multi-BAM report"
            )
        key = (resolved, decision.read_name)
        if decision.decision_id in seen[key]:
            raise ValueError(
                f"duplicate decision ID for one input/read: {decision.decision_id}"
            )
        seen[key].add(decision.decision_id)
        result[resolved][decision.read_name].append(decision)
    return result


def _validate_input_file(
    path: str, report: Mapping[str, object], *, allow_drift: bool
) -> None:
    metadata = {
        _canonical_path(str(record["path"])): record
        for record in report.get("input", {}).get("files", [])
    }
    candidate = Path(path)
    if not candidate.is_file():
        raise ValueError(f"missing input BAM: {candidate}")
    expected = metadata.get(_canonical_path(path))
    if expected is None:
        return
    stat = candidate.stat()
    changed = (
        int(expected.get("size_bytes", stat.st_size)) != int(stat.st_size)
        or int(expected.get("mtime_ns", stat.st_mtime_ns)) != int(stat.st_mtime_ns)
    )
    if changed and not allow_drift:
        raise ValueError(
            f"input BAM changed since report generation: {candidate}; "
            "use --allow-input-drift only after checking provenance"
        )


def _header_with_provenance(
    header,
    *,
    report_sha256: str,
    minimum_posterior: float,
    command_line: str,
    action_bgzf_sha256: Optional[str] = None,
    action_jsonl_sha256: Optional[str] = None,
    action_stream_schema: Optional[str] = None,
    action_input_index: Optional[int] = None,
    quality_contract_version: int = 4,
):
    if quality_contract_version not in {4, 6}:
        raise ValueError("quality contract version must be 4 or 6")
    output = append_ma_types(header, LAYER_ORDER)
    header_dict = output.to_dict()
    comments = [
        str(comment)
        for comment in header_dict.get("CO", [])
        if not str(comment).startswith(ALL_HEADER_PREFIXES)
    ]
    provenance = f";report_sha256={report_sha256}"
    if action_bgzf_sha256 is not None:
        provenance += f";action_bgzf_sha256={action_bgzf_sha256}"
    if action_jsonl_sha256 is not None:
        provenance += f";action_jsonl_sha256={action_jsonl_sha256}"
    if action_stream_schema is not None:
        provenance += f";action_stream_schema={action_stream_schema}"
    if action_input_index is not None:
        provenance += f";action_input_index={action_input_index}"
    contract_prefix = (
        V6_HEADER_PREFIX if quality_contract_version == 6 else HEADER_PREFIX
    )
    q0_semantics = (
        "exact_selected_configuration_probability_if_action_set_complete_"
        "else_zero"
        if quality_contract_version == 6
        else "sr_alternative_probability_vs_ordinary_baseline"
    )
    edge_opportunity_contract = (
        ";changed_edge_without_target_opportunity_q=0"
        if quality_contract_version == 6
        else ""
    )
    comments.append(
        contract_prefix
        + "groups=nuc_sr,tf_sr;semantics=two_strand_state_and_edge_normalization"
        + ";quality_spec=QQQ;q_scale=linear_unit_interval"
        + f";q0={q0_semantics}"
        + ";h_q0=assignment_marginalized_canonical_geometry_probability"
        + ";q1=molecular_left_edge_confidence"
        + ";q2=molecular_right_edge_confidence"
        + ";display_sr_if=q0>=T;threshold_named_only=true"
        + ";roles=Rn_tf_rescue,H_edge_normalized"
        + ";r_q0_atomic=true;h_source_ordinal=true"
        + ";unchanged_edge_q=255"
        + edge_opportunity_contract
        + ";baseline_row=255,0,0;baseline_q0_sentinel=true"
        + ";nuc_sr_edge_only=true"
        + ";nuc_occupancy_candidates=false;nuc_identity_cardinality_fixed=true"
        + ";layers_complementary=false;nuc_length_ceiling=none"
        + f";minimum_posterior={minimum_posterior}"
        + provenance
    )
    header_dict["CO"] = comments
    output = pysam.AlignmentHeader.from_dict(header_dict)
    return append_pg_record(
        output,
        {
            "PN": "fiberhmm-strand-rescue-annotate",
            "VN": FIBERHMM_VERSION,
            "CL": command_line,
            "DS": (
                "chemistry-aware MSP-to-TF rescue plus shared TF/nuc edges; "
                "QQQ alternative probability/left-edge/right-edge confidence; "
                "nucleosome identity/cardinality fixed; "
                f"report_sha256={report_sha256}"
                + (
                    f"; action_bgzf_sha256={action_bgzf_sha256}; "
                    f"action_jsonl_sha256={action_jsonl_sha256}; "
                    f"action_stream_schema={action_stream_schema}; "
                    f"action_input_index={action_input_index}"
                    if action_bgzf_sha256 is not None
                    else ""
                )
            ),
        },
    )


def write_overlay_bam(
    input_path: str,
    output_path: Path,
    *,
    region: Tuple[str, int, int],
    decisions_by_read: Mapping[str, Sequence[StrandDecision]],
    harmonizations_by_read: Mapping[str, Sequence[GeometryDecision]],
    report_sha256: str,
    minimum_posterior: float,
    command_line: str,
    io_threads: int = 1,
    protected_paths: Sequence[str] = (),
    require_all_matches: bool = False,
    defer_publish: bool = False,
    quality_contract_version: int = 4,
) -> dict:
    if output_path.suffix != ".bam":
        raise ValueError(f"output must end in .bam: {output_path}")
    forbidden = {_canonical_path(input_path)}
    forbidden.update(_canonical_path(path) for path in protected_paths)
    forbidden.update(f"{path}.bai" for path in tuple(forbidden))
    for candidate in (str(output_path), f"{output_path}.bai"):
        if _canonical_path(candidate) in forbidden:
            raise ValueError(f"refusing to overwrite an input BAM: {candidate}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.stem}.", suffix=".bam", dir=str(output_path.parent)
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    temporary_index = Path(str(temporary) + ".bai")
    output_index = Path(str(output_path) + ".bai")
    counts = {name: 0 for name in LAYER_ORDER}
    applied = set()
    applied_harmonizations = set()
    reads_written = 0
    reads_with_layers = 0
    diagnostics: DefaultDict[str, int] = defaultdict(int)
    alignment_occurrences: DefaultDict[Tuple[object, ...], int] = defaultdict(int)
    chrom, start, end = region
    expected = {
        decision.decision_id
        for values in decisions_by_read.values()
        for decision in values
    }
    expected_harmonizations = {
        decision.decision_id
        for values in harmonizations_by_read.values()
        for decision in values
    }
    try:
        with pysam.AlignmentFile(
            input_path, "rb", check_sq=False, threads=io_threads
        ) as source:
            header = _header_with_provenance(
                source.header,
                report_sha256=report_sha256,
                minimum_posterior=minimum_posterior,
                command_line=command_line,
                quality_contract_version=quality_contract_version,
            )
            with pysam.AlignmentFile(
                str(temporary), "wb", header=header, threads=io_threads
            ) as destination:
                for read in source.fetch(chrom, start, end):
                    alignment_occurrence = None
                    if not (
                        read.is_unmapped
                        or read.is_secondary
                        or read.is_supplementary
                    ):
                        record_sha256 = hashlib.sha256(
                            read.to_string().encode("utf-8")
                        ).hexdigest()
                        occurrence_key = (
                            read.query_name,
                            int(read.reference_start),
                            int(read.flag),
                            read.cigarstring,
                            record_sha256,
                        )
                        alignment_occurrence = alignment_occurrences[occurrence_key]
                        alignment_occurrences[occurrence_key] += 1
                    read_decisions = (
                        ()
                        if read.is_unmapped
                        or read.is_secondary
                        or read.is_supplementary
                        else decisions_by_read.get(read.query_name, ())
                    )
                    read_harmonizations = (
                        ()
                        if read.is_unmapped
                        or read.is_secondary
                        or read.is_supplementary
                        else harmonizations_by_read.get(read.query_name, ())
                    )
                    layer_counts = add_strand_rescue_groups(
                        read,
                        read_decisions,
                        read_harmonizations,
                        alignment_occurrence=alignment_occurrence,
                        diagnostics=diagnostics,
                        applied_decision_ids=applied,
                        applied_harmonization_ids=applied_harmonizations,
                    )
                    if layer_counts:
                        reads_with_layers += 1
                        for name, value in layer_counts.items():
                            counts[name] += value
                    destination.write(read)
                    reads_written += 1
        if require_all_matches:
            unmatched_decisions = expected - applied
            unmatched_harmonizations = expected_harmonizations - applied_harmonizations
            if unmatched_decisions or unmatched_harmonizations:
                raise ValueError(
                    "report proposals could not be materialized exactly: "
                    f"{len(unmatched_decisions)} rescues, "
                    f"{len(unmatched_harmonizations)} edge refinements; "
                    f"application={dict(sorted(diagnostics.items()))}; "
                    f"example_edge_ids={sorted(unmatched_harmonizations)[:3]}"
                )
        pysam.index(str(temporary))
        if not defer_publish:
            os.replace(temporary, output_path)
            os.replace(temporary_index, output_index)
    except BaseException:
        temporary.unlink(missing_ok=True)
        temporary_index.unlink(missing_ok=True)
        raise
    summary = {
        "input": _canonical_path(input_path),
        "output": str(output_path.resolve()),
        "index": str(output_index.resolve()),
        "region": [chrom, start, end],
        "reads_written": reads_written,
        "reads_with_layers": reads_with_layers,
        "annotations": counts,
        "decisions_expected": len(expected),
        "decisions_matched": len(applied),
        "decisions_unmatched": len(expected - applied),
        "harmonizations_expected": len(expected_harmonizations),
        "harmonizations_matched": len(applied_harmonizations),
        "harmonizations_unmatched": len(
            expected_harmonizations - applied_harmonizations
        ),
        "application": dict(sorted(diagnostics.items())),
    }
    if defer_publish:
        summary["_staged_output"] = str(temporary)
        summary["_staged_index"] = str(temporary_index)
    return summary


def _validate_stream_action_against_read(
    read, action_record: StreamActionRecord
) -> None:
    if read.is_unmapped or read.is_secondary or read.is_supplementary:
        raise ValueError(
            "streamed action targets an unmapped, secondary, or supplementary record"
        )
    if read.query_name != action_record.read_name:
        raise ValueError("action-stream query name differs from BAM record")
    record_sha256 = hashlib.sha256(read.to_string().encode("utf-8")).hexdigest()
    if record_sha256 != action_record.record_sha256:
        raise ValueError("action-stream BAM-record SHA-256 mismatch")
    if not read.has_tag("MA"):
        raise ValueError("streamed action targets a BAM record with no MA tag")
    read_length = _read_length(read)
    parsed = parse_ma_tag(read.get_tag("MA"))
    if int(parsed["read_length"]) != read_length:
        raise ValueError(
            f"read {read.query_name!r}: MA/query length mismatch"
        )
    originals = {"msp": [], "tf": [], "nuc": []}
    for name, _strand, _specification, intervals in parsed["raw_types"]:
        if name in originals:
            originals[name].extend(tuple(interval) for interval in intervals)
    for action in action_record.rescues:
        if not 0 <= action.source_ordinal < len(originals["msp"]):
            raise ValueError("streamed rescue source MSP ordinal is out of range")
        if originals["msp"][action.source_ordinal] != action.source_interval:
            raise ValueError("streamed rescue source MSP interval differs from MA")
        for component in action.components:
            if component.interval[0] + component.interval[1] > read_length:
                raise ValueError("streamed rescue component lies outside the read")
    for action in action_record.edge_updates:
        source_calls = originals[action.call_type]
        if not 0 <= action.source_ordinal < len(source_calls):
            raise ValueError(
                f"streamed {action.call_type} edge source ordinal is out of range"
            )
        if source_calls[action.source_ordinal] != action.source_interval:
            raise ValueError(
                f"streamed {action.call_type} edge source interval differs from MA"
            )
        if action.alternative_interval[0] + action.alternative_interval[1] > read_length:
            raise ValueError("streamed edge alternative lies outside the read")


def write_streaming_overlay_bam(
    input_path: str,
    output_path: Path,
    *,
    action_stream: ResolvedV5ActionStream,
    report_sha256: str,
    minimum_posterior: float,
    command_line: str,
    io_threads: int = 1,
    protected_paths: Sequence[str] = (),
    defer_publish: bool = False,
    quality_contract_version: int = 4,
) -> dict:
    """Materialize one v5 action stream by an ordinal, bounded-memory join."""
    if output_path.suffix != ".bam":
        raise ValueError(f"output must end in .bam: {output_path}")
    if _canonical_path(input_path) != action_stream.input_path:
        raise ValueError("selected BAM does not match the action stream input index")
    _validate_v5_sidecar_files(action_stream)
    forbidden = {_canonical_path(input_path)}
    forbidden.update(_canonical_path(path) for path in protected_paths)
    forbidden.update(
        (
            _canonical_path(str(action_stream.bgzf_path)),
            _canonical_path(str(action_stream.gzi_path)),
        )
    )
    forbidden.update(f"{path}.bai" for path in tuple(forbidden))
    for candidate in (str(output_path), f"{output_path}.bai"):
        if _canonical_path(candidate) in forbidden:
            raise ValueError(f"refusing to overwrite an input or provenance file: {candidate}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.stem}.", suffix=".bam", dir=str(output_path.parent)
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    temporary_index = Path(str(temporary) + ".bai")
    output_index = Path(str(output_path) + ".bai")
    counts = {name: 0 for name in LAYER_ORDER}
    diagnostics: DefaultDict[str, int] = defaultdict(int)
    reads_written = 0
    reads_with_layers = 0
    reads_with_actions = 0
    rescues_materialized = 0
    rescue_components_materialized = 0
    stream_validation = {}
    manifest = action_stream.manifest
    region = _parse_stream_region(manifest["loaded_region"], "manifest loaded_region")
    chrom, start, end = region
    try:
        with V5ActionStreamReader(
            action_stream.bgzf_path, manifest
        ) as action_reader, pysam.AlignmentFile(
            input_path, "rb", check_sq=False, threads=io_threads
        ) as source:
            header = _header_with_provenance(
                source.header,
                report_sha256=report_sha256,
                minimum_posterior=minimum_posterior,
                command_line=command_line,
                action_bgzf_sha256=str(manifest["bgzf_sha256"]),
                action_jsonl_sha256=str(manifest["jsonl_sha256"]),
                action_stream_schema=V5_ACTION_STREAM_SCHEMA,
                action_input_index=action_stream.input_index,
                quality_contract_version=quality_contract_version,
            )
            with pysam.AlignmentFile(
                str(temporary), "wb", header=header, threads=io_threads
            ) as destination:
                for ordinal, read in enumerate(source.fetch(chrom, start, end)):
                    action_record = action_reader.pop_for_ordinal(ordinal)
                    stream_rescues: Sequence[StreamRescueAction] = ()
                    stream_edges: Sequence[StreamEdgeAction] = ()
                    if action_record is not None:
                        reads_with_actions += 1
                        _validate_stream_action_against_read(read, action_record)
                        stream_rescues = tuple(
                            action
                            for action in action_record.rescues
                            if action.alternative_q / 255.0 >= minimum_posterior
                        )
                        stream_edges = action_record.edge_updates
                        rescues_materialized += len(stream_rescues)
                        rescue_components_materialized += sum(
                            len(action.components) for action in stream_rescues
                        )
                    layer_counts = add_strand_rescue_groups(
                        read,
                        (),
                        (),
                        stream_rescues=stream_rescues,
                        stream_edges=stream_edges,
                        diagnostics=diagnostics,
                    )
                    if layer_counts:
                        reads_with_layers += 1
                        for name, value in layer_counts.items():
                            counts[name] += value
                    destination.write(read)
                    reads_written += 1
                stream_validation = action_reader.finish(reads_written)
        pysam.index(str(temporary))
        pysam.quickcheck(str(temporary))
        with pysam.AlignmentFile(str(temporary), "rb", check_sq=False) as validation:
            if not validation.check_index():
                raise ValueError("staged strand-rescue BAM index is not readable")
        if not defer_publish:
            os.replace(temporary, output_path)
            os.replace(temporary_index, output_index)
    except BaseException:
        temporary.unlink(missing_ok=True)
        temporary_index.unlink(missing_ok=True)
        raise
    summary = {
        "input": _canonical_path(input_path),
        "input_index": action_stream.input_index,
        "output": str(output_path.resolve()),
        "index": str(output_index.resolve()),
        "region": [chrom, start, end],
        "reads_written": reads_written,
        "reads_with_layers": reads_with_layers,
        "reads_with_actions": reads_with_actions,
        "annotations": counts,
        "rescue_decisions_available": int(manifest["rescue_decision_count"]),
        "rescue_decisions_materialized": rescues_materialized,
        "rescue_decisions_thresholded": (
            int(manifest["rescue_decision_count"]) - rescues_materialized
        ),
        "rescue_components_materialized": rescue_components_materialized,
        "tf_edge_updates_materialized": int(manifest["tf_edge_update_count"]),
        "nuc_edge_updates_materialized": int(manifest["nuc_edge_update_count"]),
        "application": dict(sorted(diagnostics.items())),
        "action_stream": {
            "schema": V5_ACTION_STREAM_SCHEMA,
            "bgzf": str(action_stream.bgzf_path),
            "gzi": str(action_stream.gzi_path),
            "bgzf_sha256": str(manifest["bgzf_sha256"]),
            "jsonl_sha256": str(manifest["jsonl_sha256"]),
            "gzi_sha256": str(manifest["gzi_sha256"]),
            "validation": stream_validation,
        },
    }
    if defer_publish:
        summary["_staged_output"] = str(temporary)
        summary["_staged_index"] = str(temporary_index)
    return summary


def _default_output_path(output_dir: Path, input_path: str, used: set[str]) -> Path:
    source = Path(input_path)
    stem = source.name[:-4] if source.name.endswith(".bam") else source.name
    name = f"{stem}.strand-rescue.bam"
    if name in used:
        token = hashlib.sha256(input_path.encode("utf-8")).hexdigest()[:8]
        name = f"{stem}.{token}.strand-rescue.bam"
    used.add(name)
    return output_dir / name


_PUBLISH_PERMISSION_ATTEMPTS = 10
_PUBLISH_PERMISSION_INITIAL_DELAY_SECONDS = 0.05
_PUBLISH_PERMISSION_MAX_DELAY_SECONDS = 0.75


def _retry_transient_permission(operation: Callable[[], None]) -> None:
    """Retry only bounded EACCES/EPERM failures from a destination-local move."""
    delay = _PUBLISH_PERMISSION_INITIAL_DELAY_SECONDS
    for attempt in range(_PUBLISH_PERMISSION_ATTEMPTS):
        try:
            operation()
            return
        except PermissionError as error:
            if (
                error.errno not in {errno.EACCES, errno.EPERM}
                or attempt + 1 == _PUBLISH_PERMISSION_ATTEMPTS
            ):
                raise
            time.sleep(delay)
            delay = min(
                _PUBLISH_PERMISSION_MAX_DELAY_SECONDS,
                delay * 2.0,
            )


def _replace_with_permission_retry(source: Path, destination: Path) -> None:
    _retry_transient_permission(lambda: os.replace(source, destination))


def _unlink_with_permission_retry(path: Path) -> None:
    _retry_transient_permission(lambda: path.unlink(missing_ok=True))


def _publish_staged_outputs(summaries: Sequence[Mapping[str, object]]) -> None:
    """Publish a validated cohort together, restoring old files on failure."""
    pairs = [
        (Path(str(summary[stage_key])), Path(str(summary[output_key])))
        for summary in summaries
        for stage_key, output_key in (
            ("_staged_output", "output"),
            ("_staged_index", "index"),
        )
    ]
    destinations = [destination.resolve() for _stage, destination in pairs]
    if len(destinations) != len(set(destinations)):
        raise ValueError("multiple staged files target the same output path")
    for stage, destination in pairs:
        if not stage.is_file():
            raise ValueError(f"missing staged output: {stage}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and not destination.is_file():
            raise ValueError(f"output path is not a regular file: {destination}")

    backups: Dict[Path, Path] = {}
    published: List[Path] = []
    try:
        for _stage, destination in pairs:
            if not destination.exists():
                continue
            descriptor, backup_name = tempfile.mkstemp(
                prefix=f".{destination.name}.",
                suffix=".strand-rescue-backup",
                dir=str(destination.parent),
            )
            os.close(descriptor)
            backup = Path(backup_name)
            _unlink_with_permission_retry(backup)
            _replace_with_permission_retry(destination, backup)
            backups[destination] = backup
        for stage, destination in pairs:
            _replace_with_permission_retry(stage, destination)
            published.append(destination)
    except BaseException as publication_error:
        rollback_errors = []
        # Replacing a newly published file directly with its backup avoids an
        # observable missing-file interval and is itself atomic. New outputs
        # without a predecessor are removed separately.
        for destination, backup in reversed(tuple(backups.items())):
            if not backup.exists():
                continue
            try:
                _replace_with_permission_retry(backup, destination)
            except BaseException as error:  # preserve the recoverable backup
                rollback_errors.append((backup, destination, error))
        for destination in reversed(published):
            if destination in backups:
                continue
            try:
                _unlink_with_permission_retry(destination)
            except BaseException as error:
                rollback_errors.append((destination, None, error))
        for stage, _destination in pairs:
            try:
                _unlink_with_permission_retry(stage)
            except BaseException as error:
                rollback_errors.append((stage, None, error))
        if rollback_errors:
            details = ", ".join(
                (
                    f"{source} -> {destination}: {error}"
                    if destination is not None
                    else f"cleanup {source}: {error}"
                )
                for source, destination, error in rollback_errors
            )
            raise RuntimeError(
                "strand-rescue publication failed and rollback was incomplete: "
                + details
            ) from publication_error
        raise
    else:
        for backup in backups.values():
            _unlink_with_permission_retry(backup)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True)
    parser.add_argument(
        "-i",
        "--bam",
        action="append",
        help="Report input BAM to materialize; default is every report BAM",
    )
    outputs = parser.add_mutually_exclusive_group(required=True)
    outputs.add_argument(
        "-o",
        "--output",
        help="Output BAM; requires exactly one selected input",
    )
    outputs.add_argument(
        "--output-dir",
        help="Directory receiving one regional BAM per selected input",
    )
    parser.add_argument("--region", type=parse_region)
    parser.add_argument(
        "--minimum-posterior",
        type=float,
        default=0.0,
        help="Optional output-size filter; default 0 preserves every decision",
    )
    parser.add_argument("--allow-input-drift", action="store_true")
    parser.add_argument("--io-threads", type=int, default=1)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.io_threads < 1:
        parser.error("--io-threads must be positive")
    if not 0.0 <= args.minimum_posterior <= 1.0:
        parser.error("--minimum-posterior must be in [0,1]")
    report_path = Path(args.report).expanduser().resolve()
    try:
        report = _strict_json_loads(
            report_path.read_text(encoding="utf-8"), "strand-rescue report"
        )
        if not isinstance(report, Mapping):
            raise ValueError("strand-rescue report must be a JSON object")
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        parser.error(f"cannot read report: {error}")
    rescue_storage = report.get("strand_rescue", {})
    is_v5 = report.get("schema") == V5_REPORT_SCHEMA or (
        report.get("schema") == V6_REPORT_SCHEMA
        and isinstance(rescue_storage, Mapping)
        and "action_storage" in rescue_storage
    )
    streams_by_input: Dict[str, ResolvedV5ActionStream] = {}
    index = {}
    geometry_index = {}
    decisions: Sequence[StrandDecision] = ()
    harmonizations: Sequence[GeometryDecision] = ()
    try:
        if is_v5:
            action_streams = _validate_v5_action_storage(report, report_path)
            report_inputs = [stream.input_path for stream in action_streams]
            streams_by_input = {
                stream.input_path: stream for stream in action_streams
            }
        else:
            decisions = collect_decisions(
                report, minimum_posterior=args.minimum_posterior
            )
            harmonizations = collect_harmonizations(report)
            report_inputs = _report_inputs(report)
            index = _proposal_index(decisions, report_inputs)
            geometry_index = _proposal_index(harmonizations, report_inputs)
        selected = (
            [_canonical_path(path) for path in args.bam]
            if args.bam
            else report_inputs
        )
        unknown = sorted(set(selected) - set(report_inputs))
        if unknown:
            raise ValueError(
                "BAM is not recorded by the report: " + ", ".join(unknown)
            )
        if not selected:
            raise ValueError("report contains no BAM inputs")
        if args.output and len(selected) != 1:
            raise ValueError("--output requires exactly one selected BAM")
        for path in selected:
            _validate_input_file(path, report, allow_drift=args.allow_input_drift)
    except (KeyError, TypeError, ValueError) as error:
        parser.error(str(error))

    loaded_region = report.get("input", {}).get("loaded_region")
    try:
        if is_v5:
            report_region = _parse_stream_region(
                loaded_region, "report input.loaded_region"
            )
            if args.region is not None and args.region != report_region:
                raise ValueError(
                    "--region cannot redefine v5 action-stream ordinal space"
                )
            region = report_region
        elif args.region is not None:
            region = args.region
        else:
            region = _parse_stream_region(
                loaded_region, "report input.loaded_region"
            )
    except ValueError as error:
        parser.error(str(error))

    report_sha256 = _sha256_file(report_path)
    command_line = " ".join(
        shlex.quote(value)
        for value in (["fiberhmm-strand-rescue-annotate"] + list(argv or sys.argv[1:]))
    )
    output_dir = Path(args.output_dir).expanduser() if args.output_dir else None
    used_names: set[str] = set()
    summaries = []
    staged_paths: List[Path] = []
    provenance_paths = [str(report_path)]
    quality_contract_version = (
        6 if report.get("schema") == V6_REPORT_SCHEMA else 4
    )
    if is_v5:
        provenance_paths.extend(
            str(path)
            for stream in streams_by_input.values()
            for path in (stream.bgzf_path, stream.gzi_path)
        )
    try:
        for path in selected:
            output = (
                Path(args.output).expanduser()
                if args.output
                else _default_output_path(output_dir, path, used_names)
            )
            if is_v5:
                summary = write_streaming_overlay_bam(
                    path,
                    output,
                    action_stream=streams_by_input[path],
                    report_sha256=report_sha256,
                    minimum_posterior=args.minimum_posterior,
                    command_line=command_line,
                    io_threads=args.io_threads,
                    protected_paths=(
                        tuple(report_inputs)
                        + tuple(selected)
                        + tuple(provenance_paths)
                    ),
                    defer_publish=True,
                    quality_contract_version=quality_contract_version,
                )
            else:
                summary = write_overlay_bam(
                    path,
                    output,
                    region=region,
                    decisions_by_read=index[path],
                    harmonizations_by_read=geometry_index[path],
                    report_sha256=report_sha256,
                    minimum_posterior=args.minimum_posterior,
                    command_line=command_line,
                    io_threads=args.io_threads,
                    protected_paths=tuple(report_inputs) + tuple(selected),
                    require_all_matches=(
                        report.get("schema")
                        in {
                            "fiberhmm.strand_rescue.v3",
                            "fiberhmm.strand_rescue.v4",
                        }
                    ),
                    defer_publish=True,
                    quality_contract_version=quality_contract_version,
                )
            summaries.append(summary)
            staged_paths.extend(
                (
                    Path(summary["_staged_output"]),
                    Path(summary["_staged_index"]),
                )
            )
        _publish_staged_outputs(summaries)
        for summary in summaries:
            summary.pop("_staged_output")
            summary.pop("_staged_index")
    except (OSError, ValueError, pysam.utils.SamtoolsError) as error:
        for path in staged_paths:
            path.unlink(missing_ok=True)
        parser.error(str(error))
    print(
        json.dumps(
            {
                "report": str(report_path),
                "report_sha256": report_sha256,
                "decisions": (
                    int(
                        report.get("strand_rescue", {})
                        .get("action_storage", {})
                        .get("totals", {})
                        .get("rescue_decisions", 0)
                    )
                    if is_v5
                    else len(decisions)
                ),
                "harmonizations": (
                    int(
                        report.get("strand_rescue", {})
                        .get("action_storage", {})
                        .get("totals", {})
                        .get("tf_edge_updates", 0)
                    )
                    + int(
                        report.get("strand_rescue", {})
                        .get("action_storage", {})
                        .get("totals", {})
                        .get("nuc_edge_updates", 0)
                    )
                    if is_v5
                    else len(harmonizations)
                ),
                "streamed_actions": is_v5,
                "outputs": summaries,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
