"""Normalize focal footprint populations across physical/read strands.

This module is intentionally separate from consensus reconstruction. Strand
rescue learns strand-balanced canonical TF and nucleosome geometries from
ordinary calls, then uses opposite-strand TF occupancy as a prior for
weak-but-positive hard modification evidence inside MSPs. Nucleosomes are
never occupancy candidates: their accepted identity and cardinality are fixed,
and only their edges can be normalized. It reads only standard BAM sequence,
hard MM/ML (or DAF mismatches), and existing MA annotations. It never uses
another assay or library as an inference prior.
"""
from __future__ import annotations

import hashlib
import heapq
import itertools
import json
import math
import re
from bisect import bisect_left, bisect_right
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pysam
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import linear_sum_assignment
from scipy.signal import find_peaks

from fiberhmm.cli.dedup import cluster_reads
from fiberhmm.cli.extract_tags import (
    _parse_all_ma_annotations,
    _parse_ma_annotations,
)
from fiberhmm.core.bam_reader import (
    cigar_to_query_ref,
    encode_from_query_sequence,
    parse_mm_tag_query_calls,
)
from fiberhmm.inference.tf_recaller import (
    N_CTX,
    UNMETH_OFFSET,
    extract_modification_calls,
)
from fiberhmm.inference.tf_sites import bounded_edge_components
from fiberhmm.io.bam_header import declared_ma_types
from fiberhmm.io.ma_tags import flip_interval_frame, parse_ma_tag


PRESETS = {
    "dddb": {
        "model": "fiberhmm/models/dddb_nanopore.json",
        "nuc_model": "fiberhmm/models/dddb_nanopore.json",
        "strand_mode": "daf",
        "prob_threshold": None,
    },
    "ddda": {
        # Package-qualified paths prevent a stale top-level ``models/`` copy
        # from changing the preset merely because FiberHMM was launched from
        # a source checkout instead of an installed environment.
        "model": "fiberhmm/models/ddda_TF.json",
        "nuc_model": "fiberhmm/models/ddda_nuc.json",
        "strand_mode": "daf",
        "prob_threshold": None,
    },
    "hia5-nanopore": {
        "model": "fiberhmm/models/hia5_nanopore.json",
        "nuc_model": "fiberhmm/models/hia5_nanopore.json",
        "strand_mode": "alignment",
        # Nanopore Hia5 is intentionally a strict hard-call assay here.
        "prob_threshold": 248,
    },
}

MIN_MAPPED_ANNOTATION_FRACTION = 0.8
DEFAULT_ACCESSIBLE_SITE_GAP = 220


def resolve_resource_path(path: str) -> str:
    """Resolve a user path or a path relative to the installed fiberhmm package."""
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return str(candidate)
    package_dir = Path(__file__).resolve().parents[1]
    if candidate.parts[:1] == ("fiberhmm",):
        # ``fiberhmm/...`` denotes a package resource, not a cwd-relative user
        # path. Resolve it from the installed package even if the working
        # directory happens to contain a shadow ``fiberhmm`` tree.
        packaged = package_dir / Path(*candidate.parts[1:])
        return str(packaged if packaged.exists() else candidate)
    if candidate.exists():
        return str(candidate)
    relative = candidate
    packaged = package_dir / relative
    return str(packaged if packaged.exists() else candidate)


@dataclass(frozen=True)
class IntervalCall:
    start: int
    end: int
    score: int = 0
    molecular_start: Optional[int] = None
    molecular_length: Optional[int] = None
    ordinal: Optional[int] = None
    mapped_fraction: float = 1.0
    geometry_eligible: bool = True

    @property
    def center(self) -> float:
        return (self.start + self.end) / 2.0


def _alignment_blocks_fully_map_brute(
    blocks: Sequence[Tuple[int, int]], start: int, end: int
) -> bool:
    """Evaluate mapped coverage with the original per-block semantics."""
    if end <= start:
        return False
    left_mapped = any(left <= start < right for left, right in blocks)
    right_mapped = any(left <= end - 1 < right for left, right in blocks)
    mapped = sum(
        max(0, min(end, right) - max(start, left))
        for left, right in blocks
    )
    return left_mapped and right_mapped and mapped / (end - start) >= 0.95


@dataclass(frozen=True)
class _AlignmentBlockIndex:
    """Immutable prefix index for sorted, nonoverlapping alignment blocks."""

    starts: Tuple[int, ...]
    ends: Tuple[int, ...]
    prefix_mapped: Tuple[int, ...]

    @classmethod
    def build(cls, blocks: object) -> Optional["_AlignmentBlockIndex"]:
        # BAM loading creates exactly this immutable shape.  Anything more
        # permissive keeps the brute path so unusual synthetic inputs retain
        # their established iteration and arithmetic behavior.
        if not isinstance(blocks, tuple):
            return None
        starts: List[int] = []
        ends: List[int] = []
        prefix_mapped = [0]
        previous_right: Optional[int] = None
        for block in blocks:
            if (
                not isinstance(block, tuple)
                or len(block) != 2
                or type(block[0]) is not int
                or type(block[1]) is not int
            ):
                return None
            left, right = block
            if left >= right or (
                previous_right is not None and left < previous_right
            ):
                return None
            starts.append(left)
            ends.append(right)
            prefix_mapped.append(prefix_mapped[-1] + right - left)
            previous_right = right
        return cls(
            starts=tuple(starts),
            ends=tuple(ends),
            prefix_mapped=tuple(prefix_mapped),
        )

    def _contains(self, position: int) -> bool:
        block_index = bisect_right(self.starts, position) - 1
        return block_index >= 0 and position < self.ends[block_index]

    def fully_maps(self, start: int, end: int) -> bool:
        if end <= start:
            return False
        if not self._contains(start) or not self._contains(end - 1):
            return False

        first = bisect_right(self.ends, start)
        stop = bisect_left(self.starts, end)
        mapped = self.prefix_mapped[stop] - self.prefix_mapped[first]
        if first < stop:
            mapped -= max(0, start - self.starts[first])
            mapped -= max(0, self.ends[stop - 1] - end)
        return mapped / (end - start) >= 0.95


@dataclass
class ReadEvidence:
    name: str
    strand: str
    ref_start: int
    ref_end: int
    positions: np.ndarray
    steps: np.ndarray
    hits: np.ndarray
    contexts: np.ndarray
    tfs: List[IntervalCall]
    nucs: List[IntervalCall]
    msps: List[IntervalCall]
    fingerprint_positions: Optional[np.ndarray] = None
    library_id: Optional[str] = None
    nuc_steps: Optional[np.ndarray] = None
    alignment_flag: int = 0
    cigar: Optional[str] = None
    record_sha256: Optional[str] = None
    alignment_occurrence: int = 0
    alignment_blocks: Optional[Tuple[Tuple[int, int], ...]] = None
    input_index: int = 0
    input_record_ordinal: Optional[int] = None
    query_length: int = 0
    cigar_tuples: Optional[Tuple[Tuple[int, int], ...]] = None
    molecular_tfs: Optional[Tuple[Tuple[int, int], ...]] = None
    molecular_nucs: Optional[Tuple[Tuple[int, int], ...]] = None
    molecular_msps: Optional[Tuple[Tuple[int, int], ...]] = None
    amplification_fingerprint_status: Optional[str] = None
    amplification_family_id: Optional[str] = None
    pair_partner: Optional[str] = None
    duplex_sources: Tuple[str, ...] = ()
    pairing_method: Optional[str] = None
    pairing_model: Optional[str] = None
    # Multiplicative per-molecule conversion/detection factor used when
    # rebuilding chemistry likelihoods outside ``steps`` (for example the
    # DddA radial nucleosome likelihood).  Cohort calibration updates this
    # alongside the context-aware protected/accessibility LLRs.
    efficiency_factor: float = 1.0

    @property
    def molecule_id(self) -> Tuple[str, str, str]:
        return (str(self.library_id or ""), str(self.name), str(self.strand))

    def spans(self, start: int, end: int) -> bool:
        return self.ref_start <= start and self.ref_end >= end

    def fully_maps(self, start: int, end: int) -> bool:
        if end <= start:
            return False
        if not self.spans(start, end):
            return False
        if self.alignment_blocks is None:
            return True
        blocks = self.alignment_blocks
        if (
            not isinstance(blocks, tuple)
            or len(blocks) <= 1
            or type(start) is not int
            or type(end) is not int
            or end <= start
        ):
            return _alignment_blocks_fully_map_brute(blocks, start, end)

        cached_source = getattr(self, "_alignment_block_index_source", None)
        if cached_source is not blocks:
            index = _AlignmentBlockIndex.build(blocks)
            # These are deliberately not dataclass fields: action/report
            # schemas remain unchanged, while each immutable BAM block tuple
            # gets one reusable per-read index.
            self._alignment_block_index_source = blocks
            self._alignment_block_index = index
        else:
            index = getattr(self, "_alignment_block_index", None)
        if index is None:
            return _alignment_blocks_fully_map_brute(blocks, start, end)
        return index.fully_maps(start, end)

    def interval_evidence(self, start: int, end: int) -> Tuple[float, int, int]:
        lo = int(np.searchsorted(self.positions, start, side="left"))
        hi = int(np.searchsorted(self.positions, end, side="left"))
        if hi <= lo:
            return 0.0, 0, 0
        return (
            float(np.sum(self.steps[lo:hi])),
            int(hi - lo),
            int(np.sum(self.hits[lo:hi])),
        )

    def _interval_evidence_batch(
        self,
        starts: np.ndarray,
        ends: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Vectorized immutable evidence lookup for a fixed interval grid.

        Prefix arrays are private computational caches, not model evidence.
        FiberHMM's calibration functions replace ``steps`` rather than
        modifying it in place, so object-identity invalidation keeps the cache
        synchronized while repeated catalog fits reuse the expensive prefix.
        """
        starts = np.asarray(starts, dtype=np.int64)
        ends = np.asarray(ends, dtype=np.int64)
        if starts.shape != ends.shape:
            raise ValueError("batched interval starts and ends must align")
        if np.any(ends <= starts):
            raise ValueError("batched intervals must have positive width")
        cached_sources = getattr(self, "_interval_prefix_sources", None)
        current_sources = (self.steps, self.hits)
        if (
            cached_sources is None
            or cached_sources[0] is not current_sources[0]
            or cached_sources[1] is not current_sources[1]
        ):
            step_prefix = np.empty(self.steps.size + 1, dtype=np.float64)
            step_prefix[0] = 0.0
            np.cumsum(self.steps, dtype=np.float64, out=step_prefix[1:])
            hit_prefix = np.empty(self.hits.size + 1, dtype=np.int64)
            hit_prefix[0] = 0
            np.cumsum(self.hits, dtype=np.int64, out=hit_prefix[1:])
            self._interval_prefix_sources = current_sources
            self._interval_step_prefix = step_prefix
            self._interval_hit_prefix = hit_prefix
        else:
            step_prefix = self._interval_step_prefix
            hit_prefix = self._interval_hit_prefix
        left = np.searchsorted(self.positions, starts, side="left")
        right = np.searchsorted(self.positions, ends, side="left")
        return (
            step_prefix[right] - step_prefix[left],
            right - left,
            hit_prefix[right] - hit_prefix[left],
            left,
            right,
        )


def probability_to_uint8(value: float) -> int:
    """Encode a unit probability exactly as one MA/AQ quality byte."""
    return max(0, min(255, int(round(255.0 * float(value)))))


def oriented_quality_bytes(
    alternative: float,
    reference_left: float,
    reference_right: float,
    *,
    reverse: bool,
) -> Tuple[int, int, int]:
    """Return final molecular-orientation ``q0,q1,q2`` bytes."""
    left, right = (
        (reference_right, reference_left)
        if reverse
        else (reference_left, reference_right)
    )
    return (
        probability_to_uint8(alternative),
        probability_to_uint8(left),
        probability_to_uint8(right),
    )


def _cigar_tuples(read: ReadEvidence) -> Tuple[Tuple[int, int], ...]:
    if read.cigar_tuples is not None:
        return read.cigar_tuples
    if not read.cigar:
        length = max(0, int(read.ref_end) - int(read.ref_start))
        return ((0, length),) if length else ()
    operation_codes = {
        "M": 0,
        "I": 1,
        "D": 2,
        "N": 3,
        "S": 4,
        "H": 5,
        "P": 6,
        "=": 7,
        "X": 8,
    }
    parsed = tuple(
        (operation_codes[operation], int(length))
        for length, operation in re.findall(r"(\d+)([MIDNSHP=X])", read.cigar)
    )
    return parsed


def project_reference_interval_to_molecular(
    read: ReadEvidence, start: int, end: int
) -> Optional[Tuple[int, int]]:
    """Project a reference interval exactly as the v4 annotator does."""
    if end <= start:
        raise ValueError(f"invalid reference interval: {start}-{end}")
    if not (read.ref_start <= start < end <= read.ref_end):
        return None
    query_position = 0
    reference_position = int(read.ref_start)
    mapped_bases = 0
    first_query_position: Optional[int] = None
    last_query_position: Optional[int] = None
    start_maps = False
    end_maps = False
    for operation, length in _cigar_tuples(read):
        length = int(length)
        if operation in {0, 7, 8}:
            segment_start = reference_position
            segment_end = reference_position + length
            overlap_start = max(start, segment_start)
            overlap_end = min(end, segment_end)
            if overlap_start < overlap_end:
                mapped_bases += overlap_end - overlap_start
                query_start = query_position + overlap_start - segment_start
                query_end = query_position + overlap_end - segment_start
                first_query_position = (
                    query_start
                    if first_query_position is None
                    else min(first_query_position, query_start)
                )
                last_query_position = (
                    query_end - 1
                    if last_query_position is None
                    else max(last_query_position, query_end - 1)
                )
                start_maps = start_maps or segment_start <= start < segment_end
                end_maps = end_maps or segment_start <= end - 1 < segment_end
            query_position += length
            reference_position += length
        elif operation in {1, 4}:
            query_position += length
        elif operation in {2, 3}:
            reference_position += length
        elif operation in {5, 6}:
            continue
        else:  # pragma: no cover - guarded by pysam/strict CIGAR parsing
            raise ValueError(f"unsupported CIGAR operation code: {operation}")
    if (
        not start_maps
        or not end_maps
        or first_query_position is None
        or last_query_position is None
        or mapped_bases / (end - start) < 0.95
    ):
        return None
    length = last_query_position + 1 - first_query_position
    if read.alignment_flag & 16:
        return (
            int(read.query_length) - (first_query_position + length),
            int(length),
        )
    return int(first_query_position), int(length)


def _molecular_intervals_for_type(
    read: ReadEvidence, call_type: str
) -> Tuple[Tuple[int, int], ...]:
    stored = {
        "tf": read.molecular_tfs,
        "nuc": read.molecular_nucs,
        "msp": read.molecular_msps,
    }.get(call_type)
    if stored is None:
        calls = {
            "tf": read.tfs,
            "nuc": read.nucs,
            "msp": read.msps,
        }.get(call_type)
        if calls is None:
            raise ValueError(f"unsupported call type: {call_type}")
        ordered = sorted(
            (
                call
                for call in calls
                if call.molecular_start is not None
                and call.molecular_length is not None
            ),
            key=lambda call: (
                call.ordinal if call.ordinal is not None else 1 << 60,
                call.molecular_start,
            ),
        )
        return tuple(
            (int(call.molecular_start), int(call.molecular_length))
            for call in ordered
        )
    return tuple((int(start), int(length)) for start, length in stored)


def _molecular_intervals_overlap(
    left: Tuple[int, int], right: Tuple[int, int]
) -> bool:
    return left[0] < right[0] + right[1] and right[0] < left[0] + left[1]


def build_rescue_action_candidate(
    read: ReadEvidence, decision: Mapping[str, object]
) -> Tuple[Optional[dict], Optional[str]]:
    """Project one v4 rescue decision into the compact v5 action surface."""
    alternative_probability = decision.get("sr_hypothesis_probability")
    if alternative_probability is None:
        alternative_probability = decision.get("posterior")
    if alternative_probability is None:
        return None, "rescue_quality_metadata_invalid"
    source_ordinal = decision.get("current_annotation_ordinal")
    source_interval = decision.get("current_molecular_interval")
    if source_ordinal is None or source_interval is None:
        return None, "rescue_source_unprojectable"
    source_ordinal = int(source_ordinal)
    source_interval = tuple(int(value) for value in source_interval)
    msps = _molecular_intervals_for_type(read, "msp")
    if not 0 <= source_ordinal < len(msps) or msps[source_ordinal] != source_interval:
        return None, "rescue_source_mismatch"
    intervals = decision.get("proposed_site_intervals", [])
    confidences = decision.get("proposed_site_edge_confidence", [])
    if not isinstance(intervals, list) or len(intervals) != len(confidences):
        return None, "rescue_component_metadata_invalid"
    reverse = bool(read.alignment_flag & 16)
    components = []
    for component_index, (interval, confidence) in enumerate(
        zip(intervals, confidences)
    ):
        if not isinstance(interval, (list, tuple)) or len(interval) != 2:
            return None, "rescue_component_metadata_invalid"
        if not isinstance(confidence, (list, tuple)) or len(confidence) != 2:
            return None, "rescue_component_metadata_invalid"
        projected = project_reference_interval_to_molecular(
            read, int(interval[0]), int(interval[1])
        )
        if projected is None:
            return None, "rescue_component_unprojectable"
        _q0, q1, q2 = oriented_quality_bytes(
            float(alternative_probability),
            float(confidence[0]),
            float(confidence[1]),
            reverse=reverse,
        )
        components.append(
            {
                "component_index": int(component_index),
                "interval": [int(projected[0]), int(projected[1])],
                "q1": q1,
                "q2": q2,
            }
        )
    if not components:
        return None, "rescue_component_metadata_invalid"
    projected_intervals = [tuple(value["interval"]) for value in components]
    source_start, source_length = source_interval
    source_end = source_start + source_length
    if (
        len(set(projected_intervals)) != len(projected_intervals)
        or any(
            _molecular_intervals_overlap(left, right)
            for index, left in enumerate(projected_intervals)
            for right in projected_intervals[index + 1 :]
        )
        or any(
            interval[0] < source_start
            or interval[0] + interval[1] > source_end
            for interval in projected_intervals
        )
    ):
        return None, "rescue_component_topology_invalid"
    decision_id = str(decision["decision_id"])
    library_id = str(decision.get("library_id") or "")
    q0 = probability_to_uint8(
        float(alternative_probability)
    )
    return (
        {
            "kind": "rescue_candidate",
            "input_index": int(read.input_index),
            "ordinal": int(read.input_record_ordinal),
            "decision_id": decision_id,
            "current_interval": [
                int(value) for value in decision.get("current_interval", [])
            ],
            "token": hashlib.sha256(
                f"sr:{library_id}:{decision_id}".encode("utf-8")
            ).hexdigest()[:16],
            "source_ordinal": source_ordinal,
            "source_interval": [source_interval[0], source_interval[1]],
            "q0": q0,
            "components": components,
        },
        None,
    )


def build_edge_action_candidate(
    read: ReadEvidence, decision: Mapping[str, object]
) -> Tuple[Optional[dict], Optional[str]]:
    """Project one accepted v4 edge decision into a provisional v5 action."""
    if decision.get("status") != "edge_update":
        return None, str(decision.get("status") or "edge_not_actionable")
    call_type = str(decision.get("call_type"))
    if call_type not in {"tf", "nuc"}:
        raise ValueError(f"unsupported edge call type: {call_type}")
    source_ordinal = decision.get("current_annotation_ordinal")
    source_interval = decision.get("current_molecular_interval")
    if source_ordinal is None or source_interval is None:
        return None, "edge_source_unprojectable"
    source_ordinal = int(source_ordinal)
    source_interval = tuple(int(value) for value in source_interval)
    originals = _molecular_intervals_for_type(read, call_type)
    if (
        not 0 <= source_ordinal < len(originals)
        or originals[source_ordinal] != source_interval
    ):
        return None, "edge_source_mismatch"
    canonical = decision.get("canonical_interval")
    if not isinstance(canonical, (list, tuple)) or len(canonical) != 2:
        return None, "edge_alternative_unprojectable"
    alternative = project_reference_interval_to_molecular(
        read, int(canonical[0]), int(canonical[1])
    )
    if alternative is None:
        return None, "edge_alternative_unprojectable"
    hypothesis = decision.get("edge_hypothesis")
    if not isinstance(hypothesis, Mapping):
        return None, "edge_quality_metadata_invalid"
    left = hypothesis.get("left")
    right = hypothesis.get("right")
    if not isinstance(left, Mapping) or not isinstance(right, Mapping):
        return None, "edge_quality_metadata_invalid"
    materialized_edge_confidence = decision.get("materialized_edge_confidence")
    if materialized_edge_confidence is None:
        reference_left = float(left["alternative_probability"])
        reference_right = float(right["alternative_probability"])
    elif (
        not isinstance(materialized_edge_confidence, (list, tuple))
        or len(materialized_edge_confidence) != 2
    ):
        return None, "edge_quality_metadata_invalid"
    else:
        reference_left = float(materialized_edge_confidence[0])
        reference_right = float(materialized_edge_confidence[1])
    q = oriented_quality_bytes(
        float(hypothesis["alternative_probability"]),
        reference_left,
        reference_right,
        reverse=bool(read.alignment_flag & 16),
    )
    decision_id = str(decision["decision_id"])
    library_id = str(decision.get("library_id") or "")
    return (
        {
            "kind": "edge_candidate",
            "input_index": int(read.input_index),
            "ordinal": int(read.input_record_ordinal),
            "decision_id": decision_id,
            "token": hashlib.sha256(
                (
                    f"sr-edge:{call_type}:{library_id}:{decision_id}"
                ).encode("utf-8")
            ).hexdigest()[:16],
            "call_type": call_type,
            "source_ordinal": source_ordinal,
            "source_interval": [source_interval[0], source_interval[1]],
            "alternative_interval": [int(alternative[0]), int(alternative[1])],
            "q": [int(value) for value in q],
            "_aggregate": {
                "prior_only": bool(
                    decision.get("target_edge_evidence", {}).get(
                        "changed_opportunities", 0
                    )
                    == 0
                ),
                "chemistry_opposed": bool(
                    float(decision.get("molecule_probability", 0.5)) < 0.5
                ),
                "extreme_edge_shift": bool(
                    decision.get("extreme_edge_shift", False)
                ),
            },
        },
        None,
    )


def finalize_alignment_actions(
    read: ReadEvidence,
    rescue_candidates: Sequence[Mapping[str, object]],
    edge_candidates: Sequence[Mapping[str, object]],
) -> Tuple[List[dict], List[dict], Dict[str, int]]:
    """Resolve all molecular topology for one alignment and release details."""
    diagnostics: Dict[str, int] = {
        "edge_duplicate_source": 0,
        "edge_joint_collision": 0,
        "edge_rescue_collision": 0,
        "rescue_collision": 0,
    }
    rescues = sorted(
        (dict(value) for value in rescue_candidates),
        key=lambda value: (
            value.get("current_interval", []),
            value["decision_id"],
        ),
    )
    edge_values = sorted(
        (dict(value) for value in edge_candidates),
        key=lambda value: (
            value["call_type"],
            value["source_interval"],
            value["decision_id"],
        ),
    )
    by_source = {}
    for edge in edge_values:
        key = (str(edge["call_type"]), int(edge["source_ordinal"]))
        if key in by_source:
            diagnostics["edge_duplicate_source"] += 1
            key_name = f"{edge['call_type']}_edge_duplicate_source"
            diagnostics[key_name] = diagnostics.get(key_name, 0) + 1
            continue
        by_source[key] = edge

    rescue_intervals = [
        tuple(component["interval"])
        for rescue in rescues
        for component in rescue["components"]
    ]
    baseline = {
        (call_type, ordinal): interval
        for call_type in ("nuc", "tf")
        for ordinal, interval in enumerate(
            _molecular_intervals_for_type(read, call_type)
        )
    }
    valid = set(by_source)
    baseline_keys = sorted(
        baseline,
        key=lambda key: (0 if key[0] == "nuc" else 1, key[1]),
    )
    while valid:
        rejected = set()
        final_intervals = {
            key: (
                tuple(by_source[key]["alternative_interval"])
                if key in valid
                else baseline[key]
            )
            for key in baseline_keys
        }
        for left_index, left_key in enumerate(baseline_keys):
            for right_key in baseline_keys[left_index + 1 :]:
                if (
                    _molecular_intervals_overlap(
                        final_intervals[left_key], final_intervals[right_key]
                    )
                    and not _molecular_intervals_overlap(
                        baseline[left_key], baseline[right_key]
                    )
                ):
                    candidate_keys = [
                        key for key in (left_key, right_key) if key in valid
                    ]
                    call_types = {left_key[0], right_key[0]}
                    if call_types == {"tf", "nuc"}:
                        # Stage order is TF consensus, then nucleosome
                        # reconciliation. A nuc edge proposal may not veto an
                        # already-supported TF edge proposal. Revert the nuc
                        # proposal first and re-evaluate the complete topology.
                        # If the TF still conflicts with the baseline nuc, the
                        # next fixed-point pass must leave that TF proposal
                        # unmaterialized; only the dedicated TF-conditioned nuc
                        # stage can represent such a split safely.
                        nuc_candidates = [
                            key for key in candidate_keys if key[0] == "nuc"
                        ]
                        rejected.update(nuc_candidates or candidate_keys)
                    else:
                        rejected.update(candidate_keys)
        # V4 compares every same-class H pair in its original annotation order.
        # The strict ``<`` relation is intentional: collapsing two distinct
        # starts to equality (or separating equal starts inconsistently) is an
        # order change too. Checking only adjacent pairs can miss a third member
        # of an inverted component after the first pair is rejected.
        ordered_candidates = sorted(valid, key=lambda key: (key[0], key[1]))
        for left_index, left_key in enumerate(ordered_candidates):
            for right_key in ordered_candidates[left_index + 1 :]:
                if left_key[0] != right_key[0]:
                    continue
                current_order = baseline[left_key][0] < baseline[right_key][0]
                alternative_order = (
                    final_intervals[left_key][0] < final_intervals[right_key][0]
                )
                if current_order != alternative_order:
                    rejected.update((left_key, right_key))
        if not rejected:
            break
        valid.difference_update(rejected)
        diagnostics["edge_joint_collision"] += len(rejected)
        for call_type, _source_ordinal in rejected:
            key_name = f"{call_type}_edge_joint_collision"
            diagnostics[key_name] = diagnostics.get(key_name, 0) + 1

    # Match the v4/spec precedence exactly: resolve the complete H candidate
    # set jointly first, then let retained R alternatives veto surviving H
    # expansions. Removing an R-conflicting H before the joint pass can
    # incorrectly rescue another member of a newly overlapping H component.
    for key in tuple(valid):
        edge = by_source[key]
        current = tuple(edge["source_interval"])
        alternative = tuple(edge["alternative_interval"])
        if any(
            _molecular_intervals_overlap(alternative, rescue)
            and not _molecular_intervals_overlap(current, rescue)
            for rescue in rescue_intervals
        ):
            valid.remove(key)
            diagnostics["edge_rescue_collision"] += 1
            key_name = f"{edge['call_type']}_edge_rescue_collision"
            diagnostics[key_name] = diagnostics.get(key_name, 0) + 1

    final_baseline = {
        key: (
            tuple(by_source[key]["alternative_interval"])
            if key in valid
            else baseline[key]
        )
        for key in baseline_keys
    }
    accepted_rescues = []
    occupied = list(final_baseline.values())
    for rescue in rescues:
        intervals = [tuple(value["interval"]) for value in rescue["components"]]
        if any(
            _molecular_intervals_overlap(interval, existing)
            for interval in intervals
            for existing in occupied
        ):
            diagnostics["rescue_collision"] += 1
            continue
        accepted_rescues.append(
            {
                "token": str(rescue["token"]),
                "source_ordinal": int(rescue["source_ordinal"]),
                "source_interval": [
                    int(value) for value in rescue["source_interval"]
                ],
                "q0": int(rescue["q0"]),
                "components": [dict(value) for value in rescue["components"]],
            }
        )
        occupied.extend(intervals)
    accepted_edges = [
        {
            "token": str(by_source[key]["token"]),
            "call_type": str(by_source[key]["call_type"]),
            "source_ordinal": int(by_source[key]["source_ordinal"]),
            "source_interval": [
                int(value) for value in by_source[key]["source_interval"]
            ],
            "alternative_interval": [
                int(value) for value in by_source[key]["alternative_interval"]
            ],
            "q": [int(value) for value in by_source[key]["q"]],
        }
        for key in sorted(valid)
    ]
    names = [
        f"fhsr_{rescue['token']}_R{component['component_index']}"
        for rescue in accepted_rescues
        for component in rescue["components"]
    ] + [
        f"fhsr_{edge['token']}_O{edge['source_ordinal']}_H"
        for edge in accepted_edges
    ]
    if len(names) != len(set(names)):
        raise ValueError(
            f"duplicate v5 action name token for read {read.name!r}"
        )
    return accepted_rescues, accepted_edges, diagnostics


@dataclass
class SiteTemplate:
    site_id: str
    start: int
    end: int
    center: int
    support: Dict[str, int]
    start_mad: float
    end_mad: float
    local_enrichment: float
    local_enrichment_by_strand: Optional[Dict[str, float]] = None
    strand_geometry: Optional[Dict[str, dict]] = None
    geometry_reliability: float = 0.0
    start_geometry_reliability: Optional[float] = None
    end_geometry_reliability: Optional[float] = None
    strand_start_disagreement: float = 0.0
    strand_end_disagreement: float = 0.0
    call_type: str = "tf"
    source_boundary_margin: int = 0
    boundary_excluded_support: Optional[Dict[str, int]] = None
    discovery_strata: Tuple[str, ...] = ()
    consolidation_status: str = "pooled"
    consolidation_evidence: Optional[dict] = None
    family_id: Optional[str] = None
    substate_id: Optional[str] = None
    seed_provenance_stratum: Optional[str] = None


@dataclass(frozen=True)
class Configuration:
    name: str
    site_indices: Tuple[int, ...]
    is_nucleosome: bool = False


def _logsumexp(values: np.ndarray, axis: int = -1) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    maximum = np.max(values, axis=axis, keepdims=True)
    result = maximum + np.log(
        np.sum(np.exp(values - maximum), axis=axis, keepdims=True)
    )
    return np.squeeze(result, axis=axis)


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = np.asarray(values, dtype=np.float64) - np.max(values)
    weights = np.exp(shifted)
    return weights / np.sum(weights)


def fit_mixture_weights(
    log_likelihoods: np.ndarray,
    *,
    max_iter: int = 500,
    tol: float = 1e-9,
    min_weight: float = 1e-9,
    pseudocount: float = 0.0,
) -> Tuple[np.ndarray, int]:
    """Mixture weights for fixed per-read likelihoods and a symmetric prior."""
    matrix = np.asarray(log_likelihoods, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("log_likelihoods must be a non-empty 2D array")
    if not math.isfinite(pseudocount) or pseudocount < 0.0:
        raise ValueError("pseudocount must be finite and non-negative")
    weights = np.full(matrix.shape[1], 1.0 / matrix.shape[1])
    for iteration in range(1, max_iter + 1):
        responsibilities = matrix + np.log(np.maximum(weights, min_weight))[None, :]
        responsibilities -= _logsumexp(responsibilities, axis=1)[:, None]
        updated = np.sum(np.exp(responsibilities), axis=0) + pseudocount
        updated /= matrix.shape[0] + pseudocount * matrix.shape[1]
        updated = np.maximum(updated, min_weight)
        updated /= np.sum(updated)
        if float(np.max(np.abs(updated - weights))) < tol:
            return updated, iteration
        weights = updated
    return weights, max_iter


def enumerate_configurations(
    sites: Sequence[SiteTemplate], *, include_nucleosome: bool = True
) -> List[Configuration]:
    """Return accessible, every compatible TF subset, and optionally N.

    Sites carrying the same ``family_id`` are alternative geometry substates
    of one latent footprint and therefore cannot co-occur in a configuration.
    """
    configurations = [Configuration("A", ())]
    for mask in range(1, 1 << len(sites)):
        indices = tuple(index for index in range(len(sites)) if mask & (1 << index))
        ordered = sorted(
            indices, key=lambda index: (sites[index].start, sites[index].end)
        )
        if not all(
            sites[left].end <= sites[right].start
            for left, right in zip(ordered, ordered[1:])
        ):
            continue
        family_ids = [
            str(sites[index].family_id or sites[index].site_id)
            for index in ordered
        ]
        if len(family_ids) != len(set(family_ids)):
            continue
        configurations.append(
            Configuration(
                "TF:" + ",".join(sites[index].site_id for index in ordered),
                tuple(ordered),
            )
        )
    if include_nucleosome:
        configurations.append(Configuration("N", (), is_nucleosome=True))
    return configurations


def match_direct_calls(
    calls: Sequence[IntervalCall],
    sites: Sequence[SiteTemplate],
    *,
    center_radius: int = 10,
) -> List[Tuple[IntervalCall, int]]:
    """Map each existing TF call to one best non-overlapping site template."""
    selected: set[int] = set()
    matches = []
    for call in sorted(calls, key=lambda item: (item.start, item.end)):
        candidates = sorted(
            (
                index
                for index, site in enumerate(sites)
                if abs(call.center - site.center) <= center_radius
            ),
            key=lambda index: (
                abs(call.center - sites[index].center),
                abs((call.end - call.start) - (sites[index].end - sites[index].start)),
            ),
        )
        for index in candidates:
            site = sites[index]
            if any(
                site.start < sites[other].end and sites[other].start < site.end
                for other in selected
            ):
                continue
            selected.add(index)
            matches.append((call, index))
            break
    return matches


def match_direct_site_indices(
    calls: Sequence[IntervalCall],
    sites: Sequence[SiteTemplate],
    *,
    center_radius: int = 10,
) -> set[int]:
    return {
        index
        for _call, index in match_direct_calls(
            calls, sites, center_radius=center_radius
        )
    }


def _mapped_annotations(
    read,
    target: str,
    reference_positions: Sequence[Optional[int]],
    *,
    retain_topology_only: bool = False,
    parsed_annotations: Optional[Mapping[str, Sequence[Mapping[str, object]]]] = None,
) -> List[IntervalCall]:
    try:
        annotations = (
            list(parsed_annotations.get(target, ()))
            if parsed_annotations is not None
            else (_parse_ma_annotations(read, target) or [])
        )
    except (KeyError, TypeError, ValueError):
        return []
    result = []
    for ordinal, annotation in enumerate(annotations):
        query_start = max(0, int(annotation["start"]))
        annotation_length = int(annotation["length"])
        query_end = min(
            len(reference_positions), query_start + annotation_length
        )
        interval_positions = reference_positions[query_start:query_end]
        if isinstance(interval_positions, np.ndarray):
            mapped_array = interval_positions[interval_positions >= 0]
            mapped_count = int(mapped_array.size)
            mapped_min = int(np.min(mapped_array)) if mapped_count else None
            mapped_max = int(np.max(mapped_array)) if mapped_count else None
        else:
            mapped = [
                position for position in interval_positions if position is not None
            ]
            mapped_count = len(mapped)
            mapped_min = min(mapped) if mapped else None
            mapped_max = max(mapped) if mapped else None
        mapped_fraction = (
            mapped_count / len(interval_positions) if len(interval_positions) else 0.0
        )
        if not mapped_count or (
            not retain_topology_only
            and mapped_fraction < MIN_MAPPED_ANNOTATION_FRACTION
        ):
            continue
        qualities = annotation.get("quals", [])
        molecular_start = (
            int(annotation.get("read_length", len(reference_positions)))
            - (query_start + annotation_length)
            if read.is_reverse
            else query_start
        )
        result.append(
            IntervalCall(
                mapped_min,
                mapped_max + 1,
                int(qualities[0]) if qualities else 0,
                molecular_start=molecular_start,
                molecular_length=annotation_length,
                ordinal=ordinal,
                mapped_fraction=float(mapped_fraction),
                geometry_eligible=(
                    mapped_fraction >= MIN_MAPPED_ANNOTATION_FRACTION
                ),
            )
        )
    return result


def _molecular_annotation_intervals(
    read,
    target: str,
    *,
    parsed_annotations: Optional[Mapping[str, Sequence[Mapping[str, object]]]] = None,
) -> Tuple[Tuple[int, int], ...]:
    """Return every ordinary target annotation in native MA order/frame."""
    try:
        annotations = (
            list(parsed_annotations.get(target, ()))
            if parsed_annotations is not None
            else (_parse_ma_annotations(read, target) or [])
        )
    except (KeyError, TypeError, ValueError):
        return ()
    intervals = []
    for annotation in annotations:
        query_start = int(annotation["start"])
        length = int(annotation["length"])
        read_length = int(annotation.get("read_length", read.query_length or 0))
        molecular_start = (
            read_length - (query_start + length)
            if read.is_reverse
            else query_start
        )
        intervals.append((int(molecular_start), int(length)))
    return tuple(intervals)


def _has_mapped_annotation_overlap(
    read,
    target: str,
    start: int,
    end: int,
    annotation_frame: str = 'molecular',
) -> bool:
    """Test one MA layer against a reference window without decoding evidence.

    BAM indices cannot query auxiliary tags, so the alignment still has to be
    decompressed.  This lightweight gate parses only MA and CIGAR and avoids
    sequence/modification likelihood construction for rejected records.
    """
    if end <= start:
        raise ValueError("annotation overlap window must have positive width")
    try:
        parsed = parse_ma_tag(read.get_tag("MA"))
    except (KeyError, TypeError, ValueError):
        return False
    query_to_reference = None
    read_length = int(parsed["read_length"])
    for name, _strand, _quality_spec, intervals in parsed["raw_types"]:
        if name != target:
            continue
        if query_to_reference is None:
            query_to_reference = cigar_to_query_ref(read)
        for raw_start, raw_length in intervals:
            query_start, length = (
                flip_interval_frame(
                    int(raw_start), int(raw_length), read_length
                )
                if read.is_reverse and annotation_frame == 'molecular'
                else (int(raw_start), int(raw_length))
            )
            query_end = query_start + length
            bounded_start = max(0, query_start)
            bounded_end = min(len(query_to_reference), query_end)
            if bounded_end <= bounded_start:
                continue
            positions = query_to_reference[bounded_start:bounded_end]
            mapped = positions[positions >= 0]
            if (
                mapped.size / max(1, length)
                < MIN_MAPPED_ANNOTATION_FRACTION
            ):
                continue
            if int(np.min(mapped)) < end and start < int(np.max(mapped)) + 1:
                return True
    return False


def hard_observations(
    read,
    strand_mode: str,
    mode: str,
    context_size: int,
    prob_threshold: Optional[int],
):
    """Encode hard assay calls and return the physical/read-strand group."""
    sequence = read.query_sequence
    if not sequence:
        return None
    symbol = "."
    if strand_mode == "daf" and read.has_tag('cs'):
        from ..crossstrand.recall import deam_regime_masks,decode_ry_consensus,encode_daf_both_strand
        masks=deam_regime_masks(read)
        sources=str(read.get_tag('cs')).split(';')
        if masks is None or not all(mask.any() for mask in masks) or len(sources)!=2 or len(set(sources))!=2 or not all(sources):
            raise ValueError('Joint duplex requires two source names and MA deam+/deam- coverage')
        if read.is_reverse:
            raise ValueError('Joint duplex evidence requires the forward reference-frame merge output')
        conv,ct,ga=decode_ry_consensus(sequence)
        return encode_daf_both_strand(conv,ct,ga,*masks,edge_trim=10,context_size=context_size), 'BOTH'
    # Bases an MM '?' entry leaves unlisted carry no call: they are encoded
    # non-target (as in fiberhmm-call), not as misses.
    if strand_mode == "daf":
        extracted = extract_modification_calls(read, "daf", context_size)
        if extracted is None:
            return None
        modification_positions, symbol, sequence, unknown_positions = extracted
        strand = "CT" if symbol == "+" else "GA"
    else:
        if not read.has_tag("MM") or not read.has_tag("ML"):
            return None
        modification_positions, unknown_positions = parse_mm_tag_query_calls(
            read.get_tag("MM"),
            read.get_tag("ML"),
            sequence,
            bool(read.is_reverse),
            prob_threshold=int(prob_threshold if prob_threshold is not None else 125),
            mode=mode,
        )
        strand = "REV" if read.is_reverse else "FWD"
    observations = encode_from_query_sequence(
        sequence,
        modification_positions,
        edge_trim=10,
        mode=mode,
        strand=symbol,
        context_size=context_size,
        is_reverse=bool(read.is_reverse),
        unknown_positions=unknown_positions,
    )
    return observations, strand


def load_region_evidence(
    bam_path: str,
    chrom: str,
    start: int,
    end: int,
    *,
    strand_mode: str,
    mode: str,
    context_size: int,
    prob_threshold: Optional[int],
    llr_hit: np.ndarray,
    llr_miss: np.ndarray,
    min_mapq: int,
    tf_layer: str = "tf",
    nuc_layer: str = "nuc",
    max_reads: int = 0,
    input_index: int = 0,
    load_diagnostics: Optional[Dict[str, object]] = None,
    load_reads: bool = True,
    required_annotation_overlap: Optional[Tuple[str, int, int]] = None,
    required_read_names: Optional[Sequence[str]] = None,
    projection: str = "full",
    evidence_scope: str = "region",
    legacy_annotation_frame: Optional[str] = None,
    ma_annotation_frame: str = 'molecular',
) -> List[ReadEvidence]:
    """Load hard-call evidence and selected TF/nucleosome MA layers.

    ``tf`` remains the inference default.  Post-inference family-catalog tools
    can select ``tf_sr`` so rescued and edge-normalized calls are classified
    without rewriting the raw molecule-specific intervals.

    ``nuc`` likewise remains the inference default.  A post-family advisory
    recaller can select ``nuc_sr`` to test the current consensus-shadow block
    without mutating or silently substituting the ordinary call layer.
    """
    if ma_annotation_frame not in ('auto', 'seq', 'molecular'):
        raise ValueError('Unknown MA annotation frame')
    if not tf_layer or "," in tf_layer or ":" in tf_layer:
        raise ValueError("tf_layer must be one MA annotation type name")
    if not nuc_layer or "," in nuc_layer or ":" in nuc_layer:
        raise ValueError("nuc_layer must be one MA annotation type name")
    if projection not in {"full", "targeted_family"}:
        raise ValueError("projection must be 'full' or 'targeted_family'")
    if evidence_scope not in {"region", "full-alignment"}:
        raise ValueError("evidence_scope must be 'region' or 'full-alignment'")
    if required_annotation_overlap is not None:
        annotation_name, annotation_start, annotation_end = (
            required_annotation_overlap
        )
        if (
            not annotation_name
            or "," in annotation_name
            or ":" in annotation_name
            or int(annotation_end) <= int(annotation_start)
        ):
            raise ValueError("invalid required annotation overlap filter")
    if isinstance(required_read_names, (str, bytes)):
        raise ValueError("required_read_names must be a sequence of read names")
    read_name_allowlist = (
        frozenset(str(value) for value in required_read_names)
        if required_read_names is not None
        else None
    )
    if read_name_allowlist is not None and "" in read_name_allowlist:
        raise ValueError("required_read_names cannot contain an empty name")
    reads: List[ReadEvidence] = []
    library_id = str(Path(bam_path).expanduser().resolve())
    alignment_occurrences: Dict[Tuple[object, ...], int] = {}
    fetch_record_count = 0
    annotation_overlap_excluded_count = 0
    read_name_excluded_count = 0
    with pysam.AlignmentFile(bam_path, "rb") as bam:
        if ma_annotation_frame == 'auto':
            from fiberhmm.io.annotation_frame import ma_annotation_frame as resolve_ma_frame
            ma_annotation_frame = resolve_ma_frame(bam.header)
        if load_diagnostics is not None:
            load_diagnostics['ma_annotation_frame'] = ma_annotation_frame
        if tf_layer != "tf" and tf_layer not in declared_ma_types(bam.header):
            raise ValueError(
                f"selected TF layer {tf_layer!r} is not declared in the BAM "
                "MA-TYPES header"
            )
        if nuc_layer != "nuc" and nuc_layer not in declared_ma_types(bam.header):
            raise ValueError(
                f"selected nucleosome layer {nuc_layer!r} is not declared in the BAM "
                "MA-TYPES header"
            )
        if not load_reads:
            iterator = ()
        else:
            iterator = bam.fetch(chrom, start, end)
        for input_record_ordinal, read in enumerate(iterator):
            fetch_record_count = input_record_ordinal + 1
            if read.is_unmapped or read.is_secondary or read.is_supplementary:
                continue
            if read.mapping_quality < min_mapq:
                continue
            if (
                read_name_allowlist is not None
                and str(read.query_name or "") not in read_name_allowlist
            ):
                read_name_excluded_count += 1
                continue
            if required_annotation_overlap is not None and not (
                _has_mapped_annotation_overlap(
                    read,
                    str(required_annotation_overlap[0]),
                    int(required_annotation_overlap[1]),
                    int(required_annotation_overlap[2]),
                    annotation_frame=ma_annotation_frame,
                )
            ):
                annotation_overlap_excluded_count += 1
                continue
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
            alignment_occurrence = alignment_occurrences.get(occurrence_key, 0)
            alignment_occurrences[occurrence_key] = alignment_occurrence + 1
            extracted = hard_observations(
                read, strand_mode, mode, context_size, prob_threshold
            )
            if extracted is None:
                continue
            observations, strand = extracted
            reference_positions = cigar_to_query_ref(read)
            query_length = int(
                read.query_length
                or len(reference_positions)
                or len(read.query_sequence or "")
            )
            observation_array = np.asarray(observations, dtype=np.int64)
            usable = min(reference_positions.size, observation_array.size)
            mapped_positions = reference_positions[:usable]
            codes = observation_array[:usable]
            mapped = mapped_positions >= 0
            hit_codes = (codes >= 0) & (codes < N_CTX)
            miss_codes = (
                (codes >= UNMETH_OFFSET)
                & (codes < UNMETH_OFFSET + N_CTX)
            )
            regional_evidence = mapped & (hit_codes | miss_codes)
            if evidence_scope == "region":
                regional_evidence &= (mapped_positions >= start) & (
                    mapped_positions < end
                )
            positions_array = np.asarray(
                mapped_positions[regional_evidence], dtype=np.int64
            )
            regional_codes = codes[regional_evidence]
            hits_array = np.asarray(regional_codes < N_CTX, dtype=bool)
            contexts_array = np.asarray(
                np.where(
                    hits_array,
                    regional_codes,
                    regional_codes - UNMETH_OFFSET,
                ),
                dtype=np.int64,
            )
            steps_array = np.asarray(
                np.where(
                    hits_array,
                    llr_hit[contexts_array],
                    llr_miss[contexts_array],
                ),
                dtype=np.float64,
            )
            fingerprint_positions_array = (
                np.asarray(mapped_positions[mapped & hit_codes], dtype=np.int64)
                if projection == "full"
                else None
            )
            order = (
                np.argsort(positions_array)
                if positions_array.size
                else np.asarray([], dtype=np.int64)
            )
            try:
                parsed_annotations = _parse_all_ma_annotations(read, annotation_frame=ma_annotation_frame) or {}
            except (KeyError, TypeError, ValueError):
                parsed_annotations = {}
            if legacy_annotation_frame is not None and not read.has_tag('MA'):
                from .legacy_annotations import legacy_annotations
                parsed_annotations = legacy_annotations(read, legacy_annotation_frame) or {}
            projected_full = projection == "full"
            reads.append(
                ReadEvidence(
                    name=read.query_name,
                    pair_partner=str(read.get_tag('mp')) if read.has_tag('mt') and read.get_tag('mt')=='P' and read.has_tag('mp') else None,
                    duplex_sources=tuple(str(read.get_tag('cs')).split(';')) if strand=='BOTH' else (),
                    pairing_method=str(read.get_tag('pm')) if read.has_tag('pm') else None,
                    pairing_model=str(read.get_tag('mv')) if read.has_tag('mv') else None,
                    strand=strand,
                    ref_start=int(read.reference_start),
                    ref_end=int(read.reference_end or read.reference_start),
                    positions=positions_array[order],
                    steps=steps_array[order],
                    hits=hits_array[order],
                    contexts=contexts_array[order],
                    tfs=(
                        _mapped_annotations(
                            read,
                            tf_layer,
                            reference_positions,
                            retain_topology_only=True,
                            parsed_annotations=parsed_annotations,
                        )
                        if projected_full else []
                    ),
                    nucs=(
                        _mapped_annotations(
                            read,
                            nuc_layer,
                            reference_positions,
                            retain_topology_only=True,
                            parsed_annotations=parsed_annotations,
                        )
                        if projected_full else []
                    ),
                    msps=_mapped_annotations(
                        read,
                        "msp",
                        reference_positions,
                        parsed_annotations=parsed_annotations,
                    ),
                    fingerprint_positions=fingerprint_positions_array,
                    library_id=library_id,
                    alignment_flag=int(read.flag),
                    cigar=read.cigarstring,
                    record_sha256=record_sha256,
                    alignment_occurrence=alignment_occurrence,
                    alignment_blocks=tuple(
                        (int(left), int(right)) for left, right in read.get_blocks()
                    ),
                    input_index=int(input_index),
                    input_record_ordinal=int(input_record_ordinal),
                    query_length=query_length if projected_full else 0,
                    cigar_tuples=(
                        tuple(
                            (int(operation), int(length))
                            for operation, length in (read.cigartuples or ())
                        )
                        if projected_full else None
                    ),
                    molecular_tfs=(
                        _molecular_annotation_intervals(
                            read,
                            tf_layer,
                            parsed_annotations=parsed_annotations,
                        )
                        if projected_full else None
                    ),
                    molecular_nucs=(
                        _molecular_annotation_intervals(
                            read,
                            nuc_layer,
                            parsed_annotations=parsed_annotations,
                        )
                        if projected_full else None
                    ),
                    molecular_msps=(
                        _molecular_annotation_intervals(
                            read,
                            "msp",
                            parsed_annotations=parsed_annotations,
                        )
                        if projected_full else None
                    ),
                )
            )
            # ``max_reads`` is a deterministic BAM-order cap.  Once the same
            # first N eligible records returned by the original loader have
            # been collected, continuing to decompress the rest of a deep
            # targeted region cannot change the returned evidence.
            if max_reads and len(reads) >= max_reads:
                break
    if load_diagnostics is not None:
        load_diagnostics.update(
            {
                "input_index": int(input_index),
                "library_id": library_id,
                "fetch_record_count": int(fetch_record_count),
                "eligible_read_count": len(reads),
                "truncated_at_max_reads": bool(
                    max_reads and len(reads) >= max_reads
                ),
                "regional_fetch_skipped": not load_reads,
                "evidence_scope": evidence_scope,
                "required_annotation_overlap": (
                    list(required_annotation_overlap)
                    if required_annotation_overlap is not None
                    else None
                ),
                "annotation_overlap_excluded_count": (
                    annotation_overlap_excluded_count
                ),
                "required_read_name_count": (
                    len(read_name_allowlist)
                    if read_name_allowlist is not None
                    else None
                ),
                "read_name_excluded_count": read_name_excluded_count,
                "tf_layer": tf_layer,
                "nuc_layer": nuc_layer,
                "projection": projection,
                "selected_tf_call_count": sum(len(read.tfs) for read in reads),
            }
        )
    return reads


def conditional_hit_probabilities(model) -> Tuple[np.ndarray, np.ndarray]:
    emissions = np.asarray(model.emissionprob_, dtype=np.float64)
    protected_hit = emissions[0, :N_CTX]
    protected_miss = emissions[0, UNMETH_OFFSET : UNMETH_OFFSET + N_CTX]
    accessible_hit = emissions[1, :N_CTX]
    accessible_miss = emissions[1, UNMETH_OFFSET : UNMETH_OFFSET + N_CTX]
    protected = protected_hit / np.maximum(protected_hit + protected_miss, 1e-30)
    accessible = accessible_hit / np.maximum(accessible_hit + accessible_miss, 1e-30)
    return (
        np.clip(protected, 1e-6, 1.0 - 1e-6),
        np.clip(accessible, 1e-6, 1.0 - 1e-6),
    )


def _prepare_efficiency_exclusion_union(
    excluded_intervals: Sequence[Tuple[int, int]],
) -> Tuple[np.ndarray, np.ndarray]:
    """Validate and merge calibration exclusions into sorted NumPy vectors."""

    merged: List[List[int]] = []
    for raw_start, raw_end in sorted(
        (int(start), int(end)) for start, end in excluded_intervals
    ):
        if raw_end <= raw_start:
            raise ValueError("efficiency exclusion intervals must be positive")
        if merged and raw_start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], raw_end)
        else:
            merged.append([raw_start, raw_end])
    return (
        np.asarray([value[0] for value in merged], dtype=np.int64),
        np.asarray([value[1] for value in merged], dtype=np.int64),
    )


def _efficiency_opportunity_mask(
    read: ReadEvidence,
    prepared_exclusions: Tuple[np.ndarray, np.ndarray],
) -> Tuple[np.ndarray, int]:
    """Return MSP opportunities after exact interval-union exclusion."""

    positions = np.asarray(read.positions, dtype=np.int64)
    difference = np.zeros(positions.size + 1, dtype=np.int32)
    if read.msps and positions.size:
        starts = np.fromiter(
            (int(call.start) for call in read.msps),
            dtype=np.int64,
            count=len(read.msps),
        )
        ends = np.fromiter(
            (int(call.end) for call in read.msps),
            dtype=np.int64,
            count=len(read.msps),
        )
        left = np.searchsorted(positions, starts, side="left")
        right = np.searchsorted(positions, ends, side="left")
        np.add.at(difference, left, 1)
        np.add.at(difference, right, -1)
    selected = np.cumsum(difference[:-1]) > 0
    initially_selected = int(np.sum(selected))
    exclusion_starts, exclusion_ends = prepared_exclusions
    if exclusion_starts.size and positions.size:
        ordinal = np.searchsorted(exclusion_starts, positions, side="right") - 1
        valid = ordinal >= 0
        safe_ordinal = np.maximum(ordinal, 0)
        excluded = valid & (positions < exclusion_ends[safe_ordinal])
        selected &= ~excluded
    return selected, initially_selected - int(np.sum(selected))


def calibrate_read_efficiency(
    read: ReadEvidence,
    protected_hit: np.ndarray,
    accessible_hit: np.ndarray,
    *,
    pseudo_count: float,
    min_opportunities: int,
    min_factor: float = 0.2,
    max_factor: float = 2.0,
    step_attribute: str = "steps",
    excluded_intervals: Sequence[Tuple[int, int]] = (),
    scale_protected_hit: bool = True,
    _prepared_exclusions: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> dict:
    """Calibrate hard-call efficiency from noncandidate baseline MSP sites."""
    if pseudo_count < 0.0:
        raise ValueError("pseudo-count must be non-negative")
    prepared_exclusions = (
        _prepare_efficiency_exclusion_union(excluded_intervals)
        if _prepared_exclusions is None
        else _prepared_exclusions
    )
    selected, excluded_opportunities = _efficiency_opportunity_mask(
        read, prepared_exclusions
    )
    opportunities = int(np.sum(selected))
    expected = (
        float(np.mean(accessible_hit[read.contexts[selected]]))
        if opportunities
        else float("nan")
    )
    if opportunities < min_opportunities or not expected > 0.0:
        factor = 1.0
        read.efficiency_factor = factor
        adjusted_accessible = accessible_hit
        adjusted_protected = protected_hit
        llr_hit = np.log(adjusted_protected) - np.log(adjusted_accessible)
        llr_miss = np.log1p(-adjusted_protected) - np.log1p(
            -adjusted_accessible
        )
        setattr(
            read,
            step_attribute,
            np.where(read.hits, llr_hit[read.contexts], llr_miss[read.contexts]).astype(
                np.float64
            ),
        )
        return {
            "calibrated": False,
            "accessible_opportunities": opportunities,
            "excluded_candidate_opportunities": excluded_opportunities,
            "factor": factor,
        }
    hit_count = int(np.sum(read.hits[selected]))
    shrunk_rate = (hit_count + pseudo_count * expected) / (
        opportunities + pseudo_count
    )
    factor = float(np.clip(shrunk_rate / expected, min_factor, max_factor))
    read.efficiency_factor = factor
    adjusted_accessible = np.clip(accessible_hit * factor, 1e-6, 1.0 - 1e-6)
    adjusted_protected = (
        np.clip(protected_hit * factor, 1e-6, 1.0 - 1e-6)
        if scale_protected_hit
        else protected_hit
    )
    llr_hit = np.log(adjusted_protected) - np.log(adjusted_accessible)
    llr_miss = np.log1p(-adjusted_protected) - np.log1p(
        -adjusted_accessible
    )
    setattr(
        read,
        step_attribute,
        np.where(read.hits, llr_hit[read.contexts], llr_miss[read.contexts]).astype(
            np.float64
        ),
    )
    return {
        "calibrated": True,
        "accessible_opportunities": opportunities,
        "excluded_candidate_opportunities": excluded_opportunities,
        "accessible_hits": hit_count,
        "observed_rate": hit_count / opportunities,
        "expected_rate": expected,
        "shrunk_rate": float(shrunk_rate),
        "factor": factor,
        "protected_hit_scaled_with_efficiency": bool(scale_protected_hit),
    }


def calibrate_cohort_efficiency(
    reads: Sequence[ReadEvidence],
    model,
    *,
    pseudo_count: float = 20.0,
    min_opportunities: int = 20,
    step_attribute: str = "steps",
    excluded_intervals: Sequence[Tuple[int, int]] = (),
    scale_protected_hit: bool = True,
) -> dict:
    protected_hit, accessible_hit = conditional_hit_probabilities(model)
    prepared_exclusions = _prepare_efficiency_exclusion_union(
        excluded_intervals
    )
    factors = []
    excluded_opportunities = 0
    for read in reads:
        result = calibrate_read_efficiency(
            read,
            protected_hit,
            accessible_hit,
            pseudo_count=pseudo_count,
            min_opportunities=min_opportunities,
            step_attribute=step_attribute,
            excluded_intervals=excluded_intervals,
            scale_protected_hit=scale_protected_hit,
            _prepared_exclusions=prepared_exclusions,
        )
        excluded_opportunities += int(
            result["excluded_candidate_opportunities"]
        )
        if result["calibrated"]:
            factors.append(float(result["factor"]))
    values = np.asarray(factors, dtype=np.float64)
    return {
        "enabled": True,
        "reads": len(reads),
        "calibrated_reads": len(factors),
        "uncalibrated_reads": len(reads) - len(factors),
        "pseudo_count": float(pseudo_count),
        "min_accessible_opportunities": int(min_opportunities),
        "step_attribute": step_attribute,
        "candidate_exclusion_intervals": [
            [int(start), int(end)] for start, end in excluded_intervals
        ],
        "excluded_candidate_opportunities": int(excluded_opportunities),
        "protected_hit_scaled_with_efficiency": bool(scale_protected_hit),
        "factor_quantiles_5_25_50_75_95": (
            [float(value) for value in np.percentile(values, [5, 25, 50, 75, 95])]
            if values.size
            else None
        ),
    }


def assign_global_efficiency_steps(
    reads: Sequence[ReadEvidence], model, *, step_attribute: str
) -> dict:
    """Assign model-wide hard-call LLRs without molecule calibration."""
    protected_hit, accessible_hit = conditional_hit_probabilities(model)
    llr_hit = np.log(protected_hit) - np.log(accessible_hit)
    llr_miss = np.log1p(-protected_hit) - np.log1p(-accessible_hit)
    for read in reads:
        setattr(
            read,
            step_attribute,
            np.where(read.hits, llr_hit[read.contexts], llr_miss[read.contexts]).astype(
                np.float64
            ),
        )
    return {
        "enabled": False,
        "reads": len(reads),
        "step_attribute": step_attribute,
        "factor": 1.0,
    }


def collapse_amplified_reads(
    reads: Sequence[ReadEvidence],
    *,
    min_jaccard: float = 0.95,
    min_deam: int = 10,
) -> Tuple[List[ReadEvidence], dict]:
    """Collapse DAF PCR families separately within every input BAM."""
    groups: Dict[str, List[ReadEvidence]] = {}
    for read in reads:
        groups.setdefault(str(read.library_id or ""), []).append(read)
    retained: List[ReadEvidence] = []
    by_input = {}
    for library_id, group in sorted(groups.items()):
        position_sets = []
        group_keys = []
        for read in group:
            fingerprint = (
                read.fingerprint_positions
                if read.fingerprint_positions is not None
                else read.positions[read.hits]
            )
            positions = frozenset(int(value) for value in fingerprint)
            if len(positions) < min_deam:
                position_sets.append(None)
                group_keys.append(None)
            else:
                position_sets.append(positions)
                group_keys.append((read.strand,))
        labels = cluster_reads(
            position_sets, group_keys, min_jaccard, 32, 8, 7
        )
        best_by_cluster: Dict[int, int] = {}
        cluster_sizes: Dict[int, int] = {}
        for index, label_value in enumerate(labels):
            label = int(label_value)
            if label < 0:
                continue
            cluster_sizes[label] = cluster_sizes.get(label, 0) + 1
            incumbent = best_by_cluster.get(label)
            quality = (
                len(group[index].positions),
                group[index].ref_end - group[index].ref_start,
            )
            if incumbent is None:
                best_by_cluster[label] = index
            else:
                incumbent_quality = (
                    len(group[incumbent].positions),
                    group[incumbent].ref_end - group[incumbent].ref_start,
                )
                if quality > incumbent_quality:
                    best_by_cluster[label] = index
        keep = {
            index for index, label_value in enumerate(labels) if int(label_value) < 0
        } | set(best_by_cluster.values())
        representative_indices = set(best_by_cluster.values())
        for index, label_value in enumerate(labels):
            label = int(label_value)
            if label < 0:
                group[index].amplification_fingerprint_status = (
                    "unfingerprintable"
                )
                group[index].amplification_family_id = None
            else:
                group[index].amplification_fingerprint_status = (
                    "fingerprintable_representative"
                    if index in representative_indices
                    else "fingerprintable_duplicate_discarded"
                )
                group[index].amplification_family_id = (
                    f"{library_id}:strand={group[index].strand}:cluster={label}"
                )
        retained.extend(read for index, read in enumerate(group) if index in keep)
        fingerprintable = sum(int(value) >= 0 for value in labels)
        duplicates = fingerprintable - len(best_by_cluster)
        by_input[library_id] = {
            "raw_reads": len(group),
            "analyzed_molecules": len(keep),
            "fingerprintable_reads": fingerprintable,
            "unfingerprintable_reads": len(group) - fingerprintable,
            "duplicate_reads_collapsed": duplicates,
            "largest_family": max(cluster_sizes.values(), default=0),
        }
    sum_keys = (
        "raw_reads",
        "analyzed_molecules",
        "fingerprintable_reads",
        "unfingerprintable_reads",
        "duplicate_reads_collapsed",
    )
    totals = {
        key: sum(int(record[key]) for record in by_input.values()) for key in sum_keys
    }
    return retained, {
        "mode": "deamination_fingerprint_per_input_bam",
        **totals,
        "duplication_fraction": (
            totals["duplicate_reads_collapsed"] / totals["fingerprintable_reads"]
            if totals["fingerprintable_reads"]
            else 0.0
        ),
        "largest_family": max(
            (record["largest_family"] for record in by_input.values()), default=0
        ),
        "min_jaccard": float(min_jaccard),
        "min_deam": int(min_deam),
        "by_input_bam": by_input,
    }


def _mad(values: Sequence[int]) -> float:
    if not values:
        return math.nan
    array = np.asarray(values, dtype=np.float64)
    return float(np.median(np.abs(array - np.median(array))))


class _RunningMedian:
    """Exact streaming median with logarithmic insertion cost."""

    def __init__(self) -> None:
        self._lower: List[int] = []
        self._upper: List[int] = []

    def add(self, value: int) -> None:
        value = int(value)
        if not self._lower or value <= -self._lower[0]:
            heapq.heappush(self._lower, -value)
        else:
            heapq.heappush(self._upper, value)
        if len(self._lower) > len(self._upper) + 1:
            heapq.heappush(self._upper, -heapq.heappop(self._lower))
        elif len(self._upper) > len(self._lower):
            heapq.heappush(self._lower, -heapq.heappop(self._upper))

    @property
    def value(self) -> float:
        if not self._lower:
            raise ValueError("median of an empty stream")
        if len(self._lower) == len(self._upper):
            return (-self._lower[0] + self._upper[0]) / 2.0
        return float(-self._lower[0])


class _DynamicMedianCoordinateIndex:
    """Exact spatial buckets for one evolving cluster coordinate."""

    def __init__(self, radius: int) -> None:
        self.radius = int(radius)
        self.bucket_width = max(1, int(radius))
        self._buckets: Dict[int, set[int]] = {}
        self._cluster_buckets: Dict[int, int] = {}

    def _bucket(self, coordinate: float) -> int:
        return int(math.floor(float(coordinate) / self.bucket_width))

    def add(self, cluster_index: int, coordinate: float) -> None:
        bucket = self._bucket(coordinate)
        self._buckets.setdefault(bucket, set()).add(cluster_index)
        self._cluster_buckets[cluster_index] = bucket

    def update(self, cluster_index: int, coordinate: float) -> None:
        old_bucket = self._cluster_buckets[cluster_index]
        new_bucket = self._bucket(coordinate)
        if old_bucket == new_bucket:
            return
        members = self._buckets[old_bucket]
        members.remove(cluster_index)
        if not members:
            del self._buckets[old_bucket]
        self._buckets.setdefault(new_bucket, set()).add(cluster_index)
        self._cluster_buckets[cluster_index] = new_bucket

    def candidate_window(self, coordinate: float) -> Tuple[int, int]:
        return (
            self._bucket(float(coordinate) - self.radius),
            self._bucket(float(coordinate) + self.radius),
        )

    def candidate_count(self, window: Tuple[int, int]) -> int:
        first, last = window
        return sum(
            len(self._buckets.get(bucket, ()))
            for bucket in range(first, last + 1)
        )

    def iter_candidates(self, window: Tuple[int, int]) -> Iterator[int]:
        first, last = window
        for bucket in range(first, last + 1):
            yield from self._buckets.get(bucket, ())

    def contains(
        self, cluster_index: int, window: Tuple[int, int]
    ) -> bool:
        first, last = window
        bucket = self._cluster_buckets[cluster_index]
        return first <= bucket <= last


class _DynamicMedianEdgeIndex:
    """Exact bucket intersection for evolving center/start/end medians."""

    def __init__(self, center_radius: int, edge_radius: int) -> None:
        self._center = _DynamicMedianCoordinateIndex(center_radius)
        self._start = _DynamicMedianCoordinateIndex(edge_radius)
        self._end = _DynamicMedianCoordinateIndex(edge_radius)

    @staticmethod
    def _center_coordinate(start: float, end: float) -> float:
        return (start + end) / 2.0

    def add(self, cluster_index: int, start: float, end: float) -> None:
        self._center.add(cluster_index, self._center_coordinate(start, end))
        self._start.add(cluster_index, start)
        self._end.add(cluster_index, end)

    def update(self, cluster_index: int, start: float, end: float) -> None:
        self._center.update(cluster_index, self._center_coordinate(start, end))
        self._start.update(cluster_index, start)
        self._end.update(cluster_index, end)

    def candidates(
        self, center: float, start: float, end: float
    ) -> List[int]:
        axes = (
            (self._center, self._center.candidate_window(center)),
            (self._start, self._start.candidate_window(start)),
            (self._end, self._end.candidate_window(end)),
        )
        populations = tuple(
            axis.candidate_count(window) for axis, window in axes
        )
        seed_position = min(
            range(len(axes)),
            key=lambda position: (populations[position], position),
        )
        if populations[seed_position] == 0:
            return []
        seed_axis, seed_window = axes[seed_position]
        candidates = [
            cluster_index
            for cluster_index in seed_axis.iter_candidates(seed_window)
            if all(
                axis.contains(cluster_index, window)
                for position, (axis, window) in enumerate(axes)
                if position != seed_position
            )
        ]
        candidates.sort()
        return candidates


def _cluster_edge_records(
    records: Sequence[
        Tuple[IntervalCall, Tuple[str, str, str], str]
    ],
    *,
    center_radius: int,
    edge_assignment_radius: int,
    use_spatial_index: bool,
) -> List[dict]:
    """Assign edge records while preserving original cluster/tie order."""
    clusters: List[dict] = []
    spatial_index = (
        _DynamicMedianEdgeIndex(center_radius, edge_assignment_radius)
        if use_spatial_index
        else None
    )
    for record in records:
        call = record[0]
        candidates = []
        cluster_indices = (
            spatial_index.candidates(call.center, call.start, call.end)
            if spatial_index is not None
            else range(len(clusters))
        )
        for index in cluster_indices:
            cluster = clusters[index]
            median_start = cluster["starts"].value
            median_end = cluster["ends"].value
            median_center = (median_start + median_end) / 2.0
            if (
                abs(call.center - median_center) <= center_radius
                and abs(call.start - median_start) <= edge_assignment_radius
                and abs(call.end - median_end) <= edge_assignment_radius
            ):
                distance = (
                    (
                        (call.start - median_start)
                        / max(1, edge_assignment_radius)
                    )
                    ** 2
                    + (
                        (call.end - median_end)
                        / max(1, edge_assignment_radius)
                    )
                    ** 2
                )
                candidates.append((distance, index))
        if candidates:
            cluster_index = min(candidates)[1]
            cluster = clusters[cluster_index]
            cluster["records"].append(record)
            cluster["starts"].add(call.start)
            cluster["ends"].add(call.end)
            if spatial_index is not None:
                spatial_index.update(
                    cluster_index,
                    cluster["starts"].value,
                    cluster["ends"].value,
                )
        else:
            starts = _RunningMedian()
            ends = _RunningMedian()
            starts.add(call.start)
            ends.add(call.end)
            clusters.append(
                {"records": [record], "starts": starts, "ends": ends}
            )
            if spatial_index is not None:
                spatial_index.add(
                    len(clusters) - 1, starts.value, ends.value
                )
    return clusters


def _source_interval_fully_maps(
    read: ReadEvidence, call: IntervalCall, margin: int
) -> bool:
    return read.fully_maps(max(0, call.start - margin), call.end + margin)


def _alignment_identity_token(read: ReadEvidence) -> str:
    fields = (
        read.library_id or "",
        read.name,
        str(read.ref_start),
        str(read.alignment_flag),
        read.cigar or "",
        read.record_sha256 or "",
        str(read.alignment_occurrence),
    )
    return hashlib.sha256("\x1f".join(fields).encode("utf-8")).hexdigest()[:16]


def _deduplicate_decisions(values: Sequence[dict]) -> List[dict]:
    """Collapse byte-identical BAM records while rejecting ID collisions."""
    retained: Dict[str, dict] = {}
    for value in values:
        decision_id = str(value["decision_id"])
        previous = retained.get(decision_id)
        if previous is not None and previous != value:
            raise ValueError(f"non-identical decisions share ID {decision_id!r}")
        retained.setdefault(decision_id, value)
    return list(retained.values())


def _calls_for_type(read: ReadEvidence, call_type: str) -> List[IntervalCall]:
    return [
        call
        for call in _all_calls_for_type(read, call_type)
        if call.geometry_eligible
    ]


def _all_calls_for_type(read: ReadEvidence, call_type: str) -> List[IntervalCall]:
    if call_type == "tf":
        return read.tfs
    if call_type == "nuc":
        return read.nucs
    raise ValueError(f"unsupported call type: {call_type}")


_CatalogRecord = Tuple[
    IntervalCall,
    Tuple[str, str, str],
    ReadEvidence,
]


@dataclass(frozen=True)
class _StrandCallCatalog:
    """Center-sorted ordinary calls for one physical/read strand."""

    centers: Tuple[float, ...]
    records: Tuple[_CatalogRecord, ...]

    def inclusive_center_window(
        self, center: float, radius: float
    ) -> Tuple[_CatalogRecord, ...]:
        left = bisect_left(self.centers, center - radius)
        right = bisect_right(self.centers, center + radius)
        return self.records[left:right]


@dataclass(frozen=True)
class _CallCatalog:
    """Reusable per-call-type, per-strand center index."""

    call_type: str
    by_strand: Mapping[str, _StrandCallCatalog]
    _source_coverage_cache: Dict[
        int, Dict[Tuple[int, int], bool]
    ] = field(default_factory=dict, compare=False, repr=False)

    def source_interval_fully_maps(
        self, read: ReadEvidence, call: IntervalCall, margin: int
    ) -> bool:
        """Cache immutable source-call coverage separately for every margin."""
        margin = int(margin)
        by_record = self._source_coverage_cache.setdefault(margin, {})
        # A call object can intentionally be shared by synthetic reads, so the
        # cache identity must include both objects.  The catalog retains every
        # read/call reference for its lifetime, preventing Python id reuse.
        key = (id(read), id(call))
        if key not in by_record:
            by_record[key] = _source_interval_fully_maps(read, call, margin)
        return by_record[key]

    def inclusive_center_window(
        self, strand: str, center: float, radius: float
    ) -> Tuple[_CatalogRecord, ...]:
        catalog = self.by_strand.get(strand)
        if catalog is None:
            return ()
        return catalog.inclusive_center_window(center, radius)


def _build_call_catalog(
    reads: Sequence[ReadEvidence], call_type: str
) -> _CallCatalog:
    """Index geometry-eligible calls once for repeated focal queries."""
    records_by_strand: Dict[str, List[_CatalogRecord]] = {
        strand: [] for strand in sorted({read.strand for read in reads})
    }
    for read in reads:
        records_by_strand[read.strand].extend(
            (call, read.molecule_id, read)
            for call in _calls_for_type(read, call_type)
        )
    by_strand = {}
    for strand, records in records_by_strand.items():
        records.sort(key=lambda record: record[0].center)
        by_strand[strand] = _StrandCallCatalog(
            centers=tuple(record[0].center for record in records),
            records=tuple(records),
        )
    return _CallCatalog(call_type=call_type, by_strand=by_strand)


def _build_site_template(
    reads: Sequence[ReadEvidence],
    *,
    center: int,
    center_radius: int,
    local_background_radius: int,
    minimum_geometry_support: int,
    site_id: str,
    fixed_interval: Optional[Tuple[int, int]] = None,
    call_type: str = "tf",
    boundary_reliability_scale: float = 12.0,
    source_boundary_margin: int = 0,
    seed_interval: Optional[Tuple[int, int]] = None,
    edge_assignment_radius: Optional[int] = None,
    call_catalog: Optional[_CallCatalog] = None,
    allowed_geometry_records: Optional[set[Tuple[int, int]]] = None,
    discovery_strata: Sequence[str] = (),
    consolidation_status: str = "pooled",
    consolidation_evidence: Optional[Mapping[str, object]] = None,
) -> SiteTemplate:
    """Build one strand-balanced canonical geometry from ordinary calls."""
    if boundary_reliability_scale <= 0.0:
        raise ValueError("boundary reliability scale must be positive")
    if source_boundary_margin < 0:
        raise ValueError("source boundary margin must be non-negative")
    if edge_assignment_radius is not None and edge_assignment_radius < 0:
        raise ValueError("edge assignment radius must be non-negative")
    if call_catalog is not None and call_catalog.call_type != call_type:
        raise ValueError("call catalog does not match requested call type")
    source_interval_fully_maps = (
        _source_interval_fully_maps
        if call_catalog is None
        else call_catalog.source_interval_fully_maps
    )
    strands = sorted({read.strand for read in reads})
    support: Dict[str, int] = {}
    enrichment: Dict[str, float] = {}
    strand_geometry: Dict[str, dict] = {}
    boundary_excluded_support: Dict[str, int] = {}
    inner = max(1, 2 * center_radius)
    background_width = max(1, 2 * (local_background_radius - inner))
    for strand in strands:
        if call_catalog is None:
            nearby_candidates = (
                (call, read.molecule_id, read)
                for read in reads
                if read.strand == strand
                for call in _calls_for_type(read, call_type)
            )
        else:
            nearby_candidates = call_catalog.inclusive_center_window(
                strand, center, center_radius
            )
        all_nearby = [
            (call, molecule, read)
            for call, molecule, read in nearby_candidates
            if abs(call.center - center) <= center_radius
            and (
                allowed_geometry_records is None
                or (id(read), id(call)) in allowed_geometry_records
            )
            and (
                seed_interval is None
                or edge_assignment_radius is None
                or (
                    abs(call.start - seed_interval[0]) <= edge_assignment_radius
                    and abs(call.end - seed_interval[1]) <= edge_assignment_radius
                )
            )
        ]
        nearby = [
            (call, molecule)
            for call, molecule, read in all_nearby
            if source_interval_fully_maps(read, call, source_boundary_margin)
        ]
        boundary_excluded_support[strand] = len(
            {molecule for _call, molecule, _read in all_nearby}
            - {molecule for _call, molecule in nearby}
        )
        starts = [call.start for call, _molecule in nearby]
        ends = [call.end for call, _molecule in nearby]
        support[strand] = len({molecule for _call, molecule in nearby})
        if nearby:
            strand_geometry[strand] = {
                "start": int(round(float(np.median(starts)))),
                "end": int(round(float(np.median(ends)))),
                "start_mad": _mad(starts),
                "end_mad": _mad(ends),
                "calls": len(nearby),
                "molecules": support[strand],
            }
        if call_catalog is None:
            background_candidates = (
                (call, read.molecule_id, read)
                for read in reads
                if read.strand == strand
                for call in _calls_for_type(read, call_type)
            )
        else:
            background_candidates = call_catalog.inclusive_center_window(
                strand, center, local_background_radius
            )
        background = {
            molecule
            for call, molecule, read in background_candidates
            if inner < abs(call.center - center) <= local_background_radius
            and source_interval_fully_maps(
                read, call, source_boundary_margin
            )
        }
        expected = len(background) * (2 * center_radius + 1) / background_width
        enrichment[strand] = float((support[strand] + 0.5) / (expected + 0.5))

    robust_geometry = [
        value
        for strand, value in strand_geometry.items()
        if support.get(strand, 0) >= minimum_geometry_support
    ]
    if fixed_interval is None and robust_geometry:
        # Each represented strand contributes one median, regardless of depth.
        start = int(
            round(float(np.median([value["start"] for value in robust_geometry])))
        )
        end = int(
            round(float(np.median([value["end"] for value in robust_geometry])))
        )
    elif fixed_interval is not None:
        start, end = fixed_interval
    else:
        start = end = center
    boundary_mads = [
        float(value[key])
        for value in strand_geometry.values()
        for key in ("start_mad", "end_mad")
    ]
    start_mad = max(
        (float(value["start_mad"]) for value in strand_geometry.values()),
        default=0.0,
    )
    end_mad = max(
        (float(value["end_mad"]) for value in strand_geometry.values()),
        default=0.0,
    )
    total_support = sum(support.values())
    support_reliability = min(1.0, math.log1p(total_support) / math.log(101.0))
    boundary_reliability = math.exp(
        -max(boundary_mads, default=0.0) / boundary_reliability_scale
    )
    strand_reliability = 1.0 if len(robust_geometry) == len(strands) else 0.75
    start_disagreement = float(
        max((value["start"] for value in robust_geometry), default=start)
        - min((value["start"] for value in robust_geometry), default=start)
    )
    end_disagreement = float(
        max((value["end"] for value in robust_geometry), default=end)
        - min((value["end"] for value in robust_geometry), default=end)
    )
    agreement_reliability = math.exp(
        -(start_disagreement + end_disagreement) / 24.0
    )
    start_geometry_reliability = float(
        support_reliability
        * math.exp(-start_mad / boundary_reliability_scale)
        * strand_reliability
        * math.exp(
            -start_disagreement / (2.0 * boundary_reliability_scale)
        )
    )
    end_geometry_reliability = float(
        support_reliability
        * math.exp(-end_mad / boundary_reliability_scale)
        * strand_reliability
        * math.exp(
            -end_disagreement / (2.0 * boundary_reliability_scale)
        )
    )
    return SiteTemplate(
        site_id=site_id,
        start=start,
        end=end,
        center=int(round((start + end) / 2.0)),
        support=support,
        start_mad=start_mad,
        end_mad=end_mad,
        local_enrichment=max(enrichment.values(), default=0.0),
        local_enrichment_by_strand=enrichment,
        strand_geometry=strand_geometry,
        geometry_reliability=float(
            support_reliability
            * boundary_reliability
            * strand_reliability
            * agreement_reliability
        ),
        start_geometry_reliability=start_geometry_reliability,
        end_geometry_reliability=end_geometry_reliability,
        strand_start_disagreement=start_disagreement,
        strand_end_disagreement=end_disagreement,
        call_type=call_type,
        source_boundary_margin=source_boundary_margin,
        boundary_excluded_support=boundary_excluded_support,
        discovery_strata=tuple(sorted(discovery_strata)),
        consolidation_status=str(consolidation_status),
        consolidation_evidence=(
            dict(consolidation_evidence)
            if consolidation_evidence is not None
            else None
        ),
    )


def _bounded_center_groups(
    centers: Sequence[int], maximum_diameter: int
) -> List[List[int]]:
    """Partition sorted center proposals without transitive chain bridging."""
    groups: List[List[int]] = []
    for center in sorted(int(value) for value in centers):
        if groups and center - groups[-1][0] <= maximum_diameter:
            groups[-1].append(center)
        else:
            groups.append([center])
    return groups


def _family_representative_records(family: Mapping[str, object]) -> List[tuple]:
    """Retain one deterministic geometry record per collapsed molecule."""
    site = family["site"]
    selected: Dict[Tuple[str, str, str], tuple] = {}
    for record in family["records"]:
        call, molecule, _read = record
        previous = selected.get(molecule)
        if previous is None:
            selected[molecule] = record
            continue
        previous_call = previous[0]
        if (
            abs(call.start - site.start) + abs(call.end - site.end),
            call.start,
            call.end,
            call.ordinal if call.ordinal is not None else -1,
        ) < (
            abs(previous_call.start - site.start)
            + abs(previous_call.end - site.end),
            previous_call.start,
            previous_call.end,
            previous_call.ordinal if previous_call.ordinal is not None else -1,
        ):
            selected[molecule] = record
    return [selected[key] for key in sorted(selected)]


def _aggregate_boundary_grid_scores(
    family: Mapping[str, object],
    intervals: Sequence[Tuple[int, int]],
) -> Tuple[np.ndarray, int, int]:
    """Sum calibrated protected/accessibility LLRs over one interval grid."""
    if not intervals:
        raise ValueError("latent boundary grid must not be empty")
    coordinates = np.asarray(
        sorted({value for interval in intervals for value in interval}),
        dtype=np.int64,
    )
    coordinate_index = {
        int(coordinate): index for index, coordinate in enumerate(coordinates)
    }
    aggregate_prefix = np.zeros(coordinates.size, dtype=np.float64)
    representatives = _family_representative_records(family)
    envelope_start = min(start for start, _end in intervals)
    envelope_end = max(end for _start, end in intervals)
    eligible = 0
    for _call, _molecule, read in representatives:
        if not read.fully_maps(envelope_start, envelope_end):
            continue
        prefix = np.concatenate(
            (np.asarray([0.0]), np.cumsum(read.steps, dtype=np.float64))
        )
        indices = np.searchsorted(read.positions, coordinates, side="left")
        aggregate_prefix += prefix[indices]
        eligible += 1
    values = np.asarray(
        [
            aggregate_prefix[coordinate_index[end]]
            - aggregate_prefix[coordinate_index[start]]
            for start, end in intervals
        ],
        dtype=np.float64,
    )
    return values, eligible, len(representatives)


def _aggregate_random_effect_boundary_scores(
    family: Mapping[str, object],
    intervals: Sequence[Tuple[int, int]],
    *,
    start_scale: float,
    end_scale: float,
) -> Tuple[np.ndarray, int, int]:
    """Integrate molecule-specific edges around every candidate class center.

    The latent start and end are independent discrete Gaussian random effects.
    Candidate grids are constructed so every start precedes every end, making
    the integral separable and avoiding a quadratic interval-by-interval loop
    for every molecule.
    """
    if not intervals:
        raise ValueError("latent boundary grid must not be empty")
    if start_scale <= 0.0 or end_scale <= 0.0:
        raise ValueError("latent boundary scales must be positive")
    starts = np.asarray(sorted({value[0] for value in intervals}), dtype=np.int64)
    ends = np.asarray(sorted({value[1] for value in intervals}), dtype=np.int64)
    if int(starts[-1]) >= int(ends[0]):
        raise ValueError(
            "random-effect boundary grid requires every start to precede every end"
        )

    def normalized_kernel(values: np.ndarray, scale: float) -> np.ndarray:
        differences = values[:, None] - values[None, :]
        kernel = -0.5 * (differences / float(scale)) ** 2
        return kernel - _logsumexp(kernel, axis=1)[:, None]

    start_kernel = normalized_kernel(starts, start_scale)
    end_kernel = normalized_kernel(ends, end_scale)
    coordinates = np.concatenate((starts, ends))
    aggregate_start = np.zeros(starts.size, dtype=np.float64)
    aggregate_end = np.zeros(ends.size, dtype=np.float64)
    representatives = _family_representative_records(family)
    envelope_start = int(starts[0])
    envelope_end = int(ends[-1])
    eligible = 0
    for _call, _molecule, read in representatives:
        if not read.fully_maps(envelope_start, envelope_end):
            continue
        prefix = np.concatenate(
            (np.asarray([0.0]), np.cumsum(read.steps, dtype=np.float64))
        )
        prefix_values = prefix[
            np.searchsorted(read.positions, coordinates, side="left")
        ]
        start_values = -prefix_values[: starts.size]
        end_values = prefix_values[starts.size :]
        aggregate_start += _logsumexp(
            start_kernel + start_values[None, :], axis=1
        )
        aggregate_end += _logsumexp(
            end_kernel + end_values[None, :], axis=1
        )
        eligible += 1
    start_index = {int(value): index for index, value in enumerate(starts)}
    end_index = {int(value): index for index, value in enumerate(ends)}
    scores = np.asarray(
        [
            aggregate_start[start_index[start]] + aggregate_end[end_index[end]]
            for start, end in intervals
        ],
        dtype=np.float64,
    )
    return scores, eligible, len(representatives)


def _differential_opportunity_summary(
    families: Sequence[Mapping[str, object]],
    left_interval: Tuple[int, int],
    right_interval: Tuple[int, int],
) -> Dict[str, dict]:
    """Describe actual possible sites whose protected membership differs."""
    by_strand: Dict[str, Dict[str, object]] = {}
    seen_molecules: set[Tuple[str, str, str]] = set()
    envelope = (
        min(left_interval[0], right_interval[0]),
        max(left_interval[1], right_interval[1]),
    )
    for family in families:
        for _call, molecule, read in _family_representative_records(family):
            if molecule in seen_molecules:
                continue
            seen_molecules.add(molecule)
            # Use the same complete-mapping population as the boundary-grid
            # likelihood. Otherwise a partially mapped molecule can decide an
            # identifiability label while contributing no matching evidence.
            if not read.fully_maps(*envelope):
                continue
            left_mask = (read.positions >= left_interval[0]) & (
                read.positions < left_interval[1]
            )
            right_mask = (read.positions >= right_interval[0]) & (
                read.positions < right_interval[1]
            )
            selected = read.positions[left_mask ^ right_mask]
            value = by_strand.setdefault(
                read.strand,
                {
                    "positions": set(),
                    "molecules": 0,
                    "molecule_opportunities": 0,
                    "hits": 0,
                },
            )
            if selected.size:
                selected_set = {int(position) for position in selected}
                value["positions"].update(selected_set)
                value["molecules"] = int(value["molecules"]) + 1
                value["molecule_opportunities"] = int(
                    value["molecule_opportunities"]
                ) + int(selected.size)
                value["hits"] = int(value["hits"]) + int(
                    np.sum(read.hits[left_mask ^ right_mask])
                )
    output = {}
    for strand, value in sorted(by_strand.items()):
        positions = sorted(value.pop("positions"))
        output[strand] = {
            "distinct_positions": len(positions),
            "positions": positions,
            "informative_molecules": int(value["molecules"]),
            "molecule_opportunities": int(value["molecule_opportunities"]),
            "hits": int(value["hits"]),
        }
    return output


def opportunity_lattice_projection(
    read: ReadEvidence, interval: Tuple[int, int]
) -> dict:
    """Project a reference interval onto one molecule's observable lattice.

    For hard DAF evidence, two reference-coordinate intervals are exactly
    observationally equivalent on a molecule when they select the same slice
    of ``read.positions``.  The returned half-open index signature is therefore
    the identifiable object; a single base-pair boundary is not identifiable
    inside a gap between consecutive possible deamination sites.

    ``equivalent_*_range`` gives the inclusive integer range of coordinates
    that preserves the corresponding search boundary.  A ``None`` endpoint is
    open because the molecule has no bracketing opportunity on that side.
    """
    start, end = (int(interval[0]), int(interval[1]))
    if end <= start:
        raise ValueError("opportunity projection requires a positive interval")
    left_index = int(np.searchsorted(read.positions, start, side="left"))
    right_index = int(np.searchsorted(read.positions, end, side="left"))
    count = int(right_index - left_index)

    def coordinate_range(index: int) -> List[Optional[int]]:
        lower = (
            int(read.positions[index - 1]) + 1 if index > 0 else None
        )
        upper = (
            int(read.positions[index])
            if index < int(read.positions.size)
            else None
        )
        return [lower, upper]

    selected = read.positions[left_index:right_index]
    selected_hits = read.hits[left_index:right_index]
    return {
        "interval": [start, end],
        "signature": [left_index, right_index],
        "opportunities": count,
        "hits": int(np.sum(selected_hits)) if count else 0,
        "first_opportunity": int(selected[0]) if count else None,
        "last_opportunity": int(selected[-1]) if count else None,
        "minimal_projected_interval": (
            [int(selected[0]), int(selected[-1]) + 1] if count else None
        ),
        "equivalent_start_range": coordinate_range(left_index),
        "equivalent_end_range": coordinate_range(right_index),
    }


def _lattice_projection_pair_summary(
    families: Sequence[Mapping[str, object]],
    left_interval: Tuple[int, int],
    right_interval: Tuple[int, int],
) -> Dict[str, dict]:
    """Summarize molecule-level equality of two opportunity projections."""
    by_strand: Dict[str, Dict[str, int]] = {}
    seen_molecules: set[Tuple[str, str, str]] = set()
    envelope = (
        min(left_interval[0], right_interval[0]),
        max(left_interval[1], right_interval[1]),
    )
    for family in families:
        for _call, molecule, read in _family_representative_records(family):
            if molecule in seen_molecules:
                continue
            seen_molecules.add(molecule)
            values = by_strand.setdefault(
                read.strand,
                {
                    "representative_molecules": 0,
                    "envelope_mapped_molecules": 0,
                    "equivalent_projection_molecules": 0,
                    "distinguishing_molecules": 0,
                },
            )
            values["representative_molecules"] += 1
            if not read.fully_maps(*envelope):
                continue
            values["envelope_mapped_molecules"] += 1
            left_projection = opportunity_lattice_projection(
                read, left_interval
            )
            right_projection = opportunity_lattice_projection(
                read, right_interval
            )
            if left_projection["signature"] == right_projection["signature"]:
                values["equivalent_projection_molecules"] += 1
            else:
                values["distinguishing_molecules"] += 1
    output = {}
    for strand, values in sorted(by_strand.items()):
        mapped = int(values["envelope_mapped_molecules"])
        output[strand] = {
            **{key: int(value) for key, value in values.items()},
            "equivalent_projection_fraction": (
                float(values["equivalent_projection_molecules"] / mapped)
                if mapped
                else None
            ),
        }
    return output


def _weighted_coordinate_interval(
    values: Sequence[int], probabilities: np.ndarray, mass: float = 0.95
) -> List[int]:
    """Small equal-tail integer credible interval for one grid coordinate."""
    if not 0.0 < mass < 1.0:
        raise ValueError("credible mass must be strictly between zero and one")
    grouped: Dict[int, float] = {}
    for value, probability in zip(values, probabilities):
        grouped[int(value)] = grouped.get(int(value), 0.0) + float(probability)
    ordered = sorted(grouped)
    cumulative = np.cumsum([grouped[value] for value in ordered])
    tail = (1.0 - mass) / 2.0
    left_index = int(np.searchsorted(cumulative, tail, side="left"))
    right_index = int(np.searchsorted(cumulative, 1.0 - tail, side="left"))
    return [
        int(ordered[min(left_index, len(ordered) - 1)]),
        int(ordered[min(right_index, len(ordered) - 1)]),
    ]


def latent_geometry_pair_evidence(
    left_family: Mapping[str, object],
    right_family: Mapping[str, object],
    *,
    boundary_search_radius: int,
) -> dict:
    """Compare one shared latent footprint with two strand-specific intervals.

    Scores are likelihood ratios relative to the all-accessible state, so the
    omitted accessible likelihood is the same under both hypotheses and
    cancels.  The independent-boundary marginal factorizes exactly.  Uniform
    priors are applied over the same bounded one-base grid under both models.
    This result is conditional on the two call-derived candidate families; it
    is not a calibrated probability that the biological TF identity is shared.
    """
    if boundary_search_radius < 0:
        raise ValueError("boundary search radius must be non-negative")
    left_site = left_family["site"]
    right_site = right_family["site"]
    represented_margins = [
        int(site.source_boundary_margin)
        for site in (left_site, right_site)
        if int(site.source_boundary_margin) > 0
    ]
    effective_radius = boundary_search_radius
    if represented_margins:
        effective_radius = min(effective_radius, min(represented_margins))
    start_min = min(left_site.start, right_site.start)
    start_max = max(left_site.start, right_site.start)
    end_min = min(left_site.end, right_site.end)
    end_max = max(left_site.end, right_site.end)
    effective_radius = min(
        effective_radius,
        max(0, (end_min - start_max - 1) // 2),
    )
    starts = range(
        start_min - effective_radius,
        start_max + effective_radius + 1,
    )
    ends = range(
        end_min - effective_radius,
        end_max + effective_radius + 1,
    )
    intervals = [
        (start, end)
        for start in starts
        for end in ends
        if end > start
    ]
    if not intervals:
        raise ValueError("latent boundary grid has no positive-length interval")
    fixed_left_scores, fixed_left_eligible, left_total = (
        _aggregate_boundary_grid_scores(
            left_family, intervals
        )
    )
    fixed_right_scores, fixed_right_eligible, right_total = (
        _aggregate_boundary_grid_scores(right_family, intervals)
    )
    fixed_shared_scores = fixed_left_scores + fixed_right_scores
    log_grid_size = math.log(len(intervals))
    fixed_left_log_marginal = float(
        _logsumexp(fixed_left_scores) - log_grid_size
    )
    fixed_right_log_marginal = float(
        _logsumexp(fixed_right_scores) - log_grid_size
    )
    fixed_shared_log_marginal = float(
        _logsumexp(fixed_shared_scores) - log_grid_size
    )
    fixed_log_bf = float(
        fixed_shared_log_marginal
        - fixed_left_log_marginal
        - fixed_right_log_marginal
    )
    fixed_shared_probability = _softmax(fixed_shared_scores)
    fixed_shared_index = int(np.argmax(fixed_shared_scores))
    fixed_left_index = int(np.argmax(fixed_left_scores))
    fixed_right_index = int(np.argmax(fixed_right_scores))

    start_scale = max(
        2.0,
        1.4826 * float(left_site.start_mad),
        1.4826 * float(right_site.start_mad),
    )
    end_scale = max(
        2.0,
        1.4826 * float(left_site.end_mad),
        1.4826 * float(right_site.end_mad),
    )
    unique_starts = sorted({interval[0] for interval in intervals})
    unique_ends = sorted({interval[1] for interval in intervals})
    random_effect_available = unique_starts[-1] < unique_ends[0]
    random_effect_diagnostic: Dict[str, object] = {
        "available": bool(random_effect_available),
        "decision_use": "diagnostic_only_not_used_for_matching",
        "distribution": "independent_discrete_gaussian",
        "start_scale_bp": float(start_scale),
        "end_scale_bp": float(end_scale),
    }
    random_log_bf: Optional[float] = None
    random_probability: Optional[float] = None
    if random_effect_available:
        random_left_scores, random_left_eligible, _left_total_again = (
            _aggregate_random_effect_boundary_scores(
                left_family,
                intervals,
                start_scale=start_scale,
                end_scale=end_scale,
            )
        )
        random_right_scores, random_right_eligible, _right_total_again = (
            _aggregate_random_effect_boundary_scores(
                right_family,
                intervals,
                start_scale=start_scale,
                end_scale=end_scale,
            )
        )
        if (
            fixed_left_eligible != random_left_eligible
            or fixed_right_eligible != random_right_eligible
        ):
            raise ValueError(
                "fixed and random-effect boundary eligibility diverged"
            )
        random_shared_scores = random_left_scores + random_right_scores
        random_left_log_marginal = float(
            _logsumexp(random_left_scores) - log_grid_size
        )
        random_right_log_marginal = float(
            _logsumexp(random_right_scores) - log_grid_size
        )
        random_shared_log_marginal = float(
            _logsumexp(random_shared_scores) - log_grid_size
        )
        random_log_bf = float(
            random_shared_log_marginal
            - random_left_log_marginal
            - random_right_log_marginal
        )
        random_probability = float(_logistic(random_log_bf))
        random_effect_diagnostic.update(
            {
                "left_map_interval": list(
                    intervals[int(np.argmax(random_left_scores))]
                ),
                "right_map_interval": list(
                    intervals[int(np.argmax(random_right_scores))]
                ),
                "shared_map_interval": list(
                    intervals[int(np.argmax(random_shared_scores))]
                ),
                "left_log_marginal": random_left_log_marginal,
                "right_log_marginal": random_right_log_marginal,
                "shared_log_marginal": random_shared_log_marginal,
                "log_bf_shared_vs_separate": random_log_bf,
                "conditional_equal_prior_probability": random_probability,
            }
        )
    else:
        random_effect_diagnostic["unavailable_reason"] = (
            "candidate_grid_is_not_a_complete_positive_interval_cartesian_grid"
        )
    differential = _differential_opportunity_summary(
        (left_family, right_family),
        (left_site.start, left_site.end),
        (right_site.start, right_site.end),
    )
    projection_summary = _lattice_projection_pair_summary(
        (left_family, right_family),
        (left_site.start, left_site.end),
        (right_site.start, right_site.end),
    )
    represented_strands = sorted(
        {
            record[2].strand
            for family in (left_family, right_family)
            for record in family["records"]
        }
    )
    zero_differential_strands = [
        strand
        for strand in represented_strands
        if differential.get(strand, {}).get("distinct_positions", 0) == 0
    ]
    identical_interval = (
        left_site.start == right_site.start
        and left_site.end == right_site.end
    )
    formally_nonidentifiable = bool(represented_strands) and len(
        zero_differential_strands
    ) == len(represented_strands)
    left_grid_likelihood_flat = bool(
        np.ptp(fixed_left_scores) <= 1e-12
    )
    right_grid_likelihood_flat = bool(
        np.ptp(fixed_right_scores) <= 1e-12
    )
    # Byte-identical candidate intervals are duplicate model states. Different
    # coordinates that happen to have identical seed projections remain an
    # explicit ambiguity; they are not converted into overwhelming merge
    # evidence because other coordinates on the candidate grid may be
    # distinguishable.
    insufficient_grid_coverage = (
        fixed_left_eligible == 0 or fixed_right_eligible == 0
    )
    # The molecule-boundary random-effect calculation above is retained as an
    # explicit development diagnostic, but it is not yet a valid assignment
    # score. It integrates physical edge jitter around one shared reference
    # coordinate without first projecting that coordinate onto each strand's
    # C/G opportunity lattice. On real DddA data that confounds lattice
    # snapping with a biological boundary shift. The fixed-boundary marginal
    # compares the two hypotheses on the same opportunity-aware chemistry grid
    # and is the audited conditional score used for this consolidation pass.
    matching_score_source = (
        "deterministic_byte_identical_duplicate"
        if identical_interval
        else "insufficient_grid_coverage"
        if insufficient_grid_coverage
        else "fixed_boundary_conditional_log_bf"
    )
    matching_log_score = (
        50.0
        if identical_interval
        else -1e12
        if insufficient_grid_coverage
        else fixed_log_bf
    )
    return {
        "model": "shared_vs_separate_latent_boundary_grid.v2",
        "left_interval": [int(left_site.start), int(left_site.end)],
        "right_interval": [int(right_site.start), int(right_site.end)],
        "boundary_search_radius": int(boundary_search_radius),
        "effective_boundary_search_radius": int(effective_radius),
        "candidate_grid_size": len(intervals),
        "left_representative_molecules": int(left_total),
        "right_representative_molecules": int(right_total),
        "left_grid_eligible_molecules": int(fixed_left_eligible),
        "right_grid_eligible_molecules": int(fixed_right_eligible),
        "left_map_interval": list(intervals[fixed_left_index]),
        "right_map_interval": list(intervals[fixed_right_index]),
        "shared_map_interval": list(intervals[fixed_shared_index]),
        "conditional_grid_shared_start_95_equal_tail_interval": (
            _weighted_coordinate_interval(
                [interval[0] for interval in intervals],
                fixed_shared_probability,
            )
        ),
        "conditional_grid_shared_end_95_equal_tail_interval": (
            _weighted_coordinate_interval(
                [interval[1] for interval in intervals],
                fixed_shared_probability,
            )
        ),
        "left_loss_at_shared_map": float(
            np.max(fixed_left_scores)
            - fixed_left_scores[fixed_shared_index]
        ),
        "right_loss_at_shared_map": float(
            np.max(fixed_right_scores)
            - fixed_right_scores[fixed_shared_index]
        ),
        "left_log_marginal": fixed_left_log_marginal,
        "right_log_marginal": fixed_right_log_marginal,
        "shared_log_marginal": fixed_shared_log_marginal,
        "log_bf_shared_vs_separate": fixed_log_bf,
        "fixed_boundary_log_bf_shared_vs_separate": fixed_log_bf,
        "matching_log_bf_upper_bound": float(log_grid_size),
        "molecule_boundary_random_effect": random_effect_diagnostic,
        "same_geometry_probability_equal_prior": float(
            _logistic(fixed_log_bf)
        ),
        "conditional_seed_pair_score_equal_prior_logistic": float(
            _logistic(fixed_log_bf)
        ),
        "experimental_random_effect_same_geometry_probability_equal_prior": (
            random_probability
        ),
        "matching_log_score": float(matching_log_score),
        "matching_score_source": matching_score_source,
        "insufficient_grid_coverage": bool(insufficient_grid_coverage),
        "identical_interval": bool(identical_interval),
        "formally_nonidentifiable_on_all_represented_strands": bool(
            formally_nonidentifiable
        ),
        "seed_intervals_opportunity_equivalent_on_all_represented_strands": (
            bool(formally_nonidentifiable)
        ),
        "left_candidate_grid_likelihood_flat": left_grid_likelihood_flat,
        "right_candidate_grid_likelihood_flat": right_grid_likelihood_flat,
        "candidate_grid_likelihood_flat_for_both_families": bool(
            left_grid_likelihood_flat and right_grid_likelihood_flat
        ),
        "differential_opportunities_by_strand": differential,
        "opportunity_lattice_projection_by_strand": projection_summary,
        "zero_differential_opportunity_strands": zero_differential_strands,
        "selection_conditioning": (
            "candidate_pair_seeded_by_ordinary_calls_and_robust_coordinate_gate"
        ),
        "probability_calibration": (
            "none_selection_conditioned_bounded_seed_pair_score"
        ),
    }


def _match_stratified_geometry_families(
    left: Sequence[dict],
    right: Sequence[dict],
    *,
    center_radius: int,
    edge_compatibility_bp: int,
    diagnostics: Optional[List[dict]] = None,
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """Match strand families under a shared latent-boundary model.

    Per-strand family formation already bounds raw edge diameter.  Applying
    the same bound to the *union* of two families makes one tail call veto an
    otherwise identical cross-strand mode.  Candidate gating therefore uses
    robust family medians.  Compatible pairs are then compared under two
    hypotheses over the same finite boundary grid: one shared latent interval
    versus independently placed strand intervals.

    The molecule likelihood is evaluated from calibrated hard-chemistry LLRs
    at represented opportunities.  A missing C/G (or A/T) opportunity adds
    exactly zero.  The returned matching includes an explicit unmatched null;
    positive pair evidence is therefore necessary but is not sufficient when
    several families compete for the same partner.
    """
    if not left or not right:
        return [], list(range(len(left))), list(range(len(right)))
    n_left = len(left)
    n_right = len(right)
    size = n_left + n_right
    incompatible = -1e12
    scores = np.full((size, size), incompatible, dtype=np.float64)
    pair_evidence: Dict[Tuple[int, int], dict] = {}
    for left_index, left_family in enumerate(left):
        for right_index, right_family in enumerate(right):
            left_site = left_family["site"]
            right_site = right_family["site"]
            center_distance = abs(
                float(left_family["center"])
                - float(right_family["center"])
            )
            start_distance = abs(left_site.start - right_site.start)
            end_distance = abs(left_site.end - right_site.end)
            if (
                center_distance <= center_radius
                and start_distance <= edge_compatibility_bp
                and end_distance <= edge_compatibility_bp
            ):
                evidence = latent_geometry_pair_evidence(
                    left_family,
                    right_family,
                    boundary_search_radius=min(2, edge_compatibility_bp),
                )
                evidence.update(
                    {
                        "left_family_index": int(left_index),
                        "right_family_index": int(right_index),
                        "center_distance": float(center_distance),
                        "start_distance": int(start_distance),
                        "end_distance": int(end_distance),
                    }
                )
                pair_evidence[(left_index, right_index)] = evidence
                scores[left_index, right_index] = float(
                    evidence["matching_log_score"]
                )
        scores[left_index, n_right + left_index] = 0.0
    for right_index in range(n_right):
        scores[n_left + right_index, right_index] = 0.0
    scores[n_left:, n_right:] = 0.0

    row_indices, column_indices = linear_sum_assignment(scores, maximize=True)
    proposed_matches = [
        (int(row), int(column))
        for row, column in zip(row_indices, column_indices)
        if row < n_left
        and column < n_right
        and scores[row, column] > incompatible
    ]
    matches = []
    for left_index, right_index in proposed_matches:
        evidence = pair_evidence[(left_index, right_index)]
        row_values = [
            scores[left_index, candidate]
            for candidate in range(n_right)
            if scores[left_index, candidate] > incompatible
        ]
        column_values = [
            scores[candidate, right_index]
            for candidate in range(n_left)
            if scores[candidate, right_index] > incompatible
        ]
        selected_score = float(scores[left_index, right_index])
        row_probability = float(
            math.exp(selected_score - _logsumexp(np.asarray([0.0, *row_values])))
        )
        column_probability = float(
            math.exp(
                selected_score
                - _logsumexp(np.asarray([0.0, *column_values]))
            )
        )
        assignment_probability = min(row_probability, column_probability)
        evidence["left_assignment_probability"] = row_probability
        evidence["right_assignment_probability"] = column_probability
        evidence["assignment_probability"] = assignment_probability
        evidence["assignment_confidence_heuristic"] = assignment_probability
        evidence["assignment_probability_semantics"] = (
            "minimum_of_row_and_column_local_softmax_heuristics_not_a_"
            "global_matching_posterior"
        )
        evidence["selected_by_global_assignment"] = True
        # Exactly 0.5 is a silent comparison and remains unresolved. Only a
        # byte-identical duplicate is collapsed without positive conditional
        # evidence; all other zero-information pairs stay separate for the
        # later molecule-configuration model.
        accepted = bool(evidence.get("identical_interval")) or (
            selected_score > 0.0
            and assignment_probability > 0.5
        )
        evidence["accepted"] = bool(accepted)
        evidence["decision"] = (
            "matched_shared_latent_geometry"
            if accepted
            else "unmatched_geometry_ambiguity"
        )
        if accepted:
            matches.append((left_index, right_index))

    for key, evidence in pair_evidence.items():
        if "selected_by_global_assignment" not in evidence:
            evidence["selected_by_global_assignment"] = False
            evidence["accepted"] = False
            evidence["decision"] = "not_selected_by_global_assignment"
        if diagnostics is not None:
            diagnostics.append(dict(evidence))
        left[key[0]].setdefault("pair_evidence", {})[key[1]] = evidence
        right[key[1]].setdefault("pair_evidence", {})[key[0]] = evidence
    matched_left = {left_index for left_index, _right_index in matches}
    matched_right = {right_index for _left_index, right_index in matches}
    return (
        matches,
        [index for index in range(n_left) if index not in matched_left],
        [index for index in range(n_right) if index not in matched_right],
    )


def discover_sites(
    reads: Sequence[ReadEvidence],
    chrom_start: int,
    chrom_end: int,
    *,
    min_support: int = 10,
    minimum_geometry_support: int = 3,
    center_radius: int = 10,
    peak_distance: int = 15,
    max_boundary_mad: float = 12.0,
    min_local_enrichment: float = 2.0,
    local_background_radius: int = 250,
    max_auto_sites: int = 0,
    call_type: str = "tf",
    smoothing_sigma: float = 3.0,
    boundary_reliability_scale: float = 12.0,
    source_boundary_margin: int = 0,
    edge_compatibility_bp: int = 12,
    separate_strand_maps: bool = False,
    diagnostics: Optional[Dict[str, object]] = None,
) -> List[SiteTemplate]:
    """Discover recurrent centers and bounded start/end footprint families."""
    if smoothing_sigma <= 0.0:
        raise ValueError("smoothing sigma must be positive")
    if edge_compatibility_bp < 0:
        raise ValueError("edge compatibility must be non-negative")
    strands = sorted({read.strand for read in reads})
    call_catalog = _build_call_catalog(reads, call_type)
    source_interval_fully_maps = call_catalog.source_interval_fully_maps
    calls_by_strand: Dict[str, List[Tuple[IntervalCall, Tuple[str, str, str]]]] = {
        strand: [] for strand in strands
    }
    for strand, strand_catalog in call_catalog.by_strand.items():
        for call, molecule, read in strand_catalog.records:
            if call.end <= chrom_start or call.start >= chrom_end:
                continue
            if (
                not source_interval_fully_maps(
                    read, call, source_boundary_margin
                )
            ):
                continue
            calls_by_strand[strand].append((call, molecule))

    proposed_by_strand: Dict[str, List[int]] = {
        strand: [] for strand in strands
    }
    proposal_min_support = (
        minimum_geometry_support if separate_strand_maps else min_support
    )
    width = chrom_end - chrom_start
    for strand, calls in calls_by_strand.items():
        if not calls or width <= 0:
            continue
        call_centers = tuple(call.center for call, _molecule in calls)
        histogram = np.zeros(width, dtype=np.float64)
        for call, _molecule in calls:
            index = int(round(call.center)) - chrom_start
            if 0 <= index < width:
                histogram[index] += 1.0
        smooth = gaussian_filter1d(histogram, smoothing_sigma)
        peaks, _ = find_peaks(
            smooth, distance=peak_distance, prominence=0.25, height=0.18
        )
        for peak in peaks:
            center = chrom_start + int(peak)
            left = bisect_left(call_centers, center - center_radius)
            right = bisect_right(call_centers, center + center_radius)
            nearby_records = calls[left:right]
            if (
                len({molecule for _call, molecule in nearby_records})
                >= proposal_min_support
            ):
                proposed_by_strand[strand].append(center)

    proposed = [
        center
        for strand_centers in proposed_by_strand.values()
        for center in strand_centers
    ]
    merged = _bounded_center_groups(proposed, center_radius)

    def record_identity(record: _CatalogRecord) -> tuple:
        return (
            *record[1],
            _alignment_identity_token(record[2]),
            record[0].start,
            record[0].end,
            record[0].ordinal if record[0].ordinal is not None else -1,
            record[0].molecular_start
            if record[0].molecular_start is not None
            else -1,
            record[0].molecular_length
            if record[0].molecular_length is not None
            else -1,
        )

    def families_near(
        center: int,
        selected_strands: Sequence[str],
        source_centers: Optional[Sequence[int]] = None,
    ) -> List[List[_CatalogRecord]]:
        represented_centers = tuple(
            int(value) for value in (source_centers or (center,))
        )
        query_center = (
            min(represented_centers) + max(represented_centers)
        ) / 2.0
        query_radius = center_radius + (
            max(represented_centers) - min(represented_centers)
        ) / 2.0
        nearby_records = [
            record
            for strand in selected_strands
            for record in call_catalog.inclusive_center_window(
                strand, query_center, query_radius
            )
            if any(
                abs(record[0].center - source_center) <= center_radius
                for source_center in represented_centers
            )
            and source_interval_fully_maps(
                record[2], record[0], source_boundary_margin
            )
        ]
        return bounded_edge_components(
            nearby_records,
            edge_compatibility_bp,
            interval_key=lambda record: (record[0].start, record[0].end),
            identity_key=record_identity,
        )

    def build_family_site(
        family: Sequence[_CatalogRecord],
        center: int,
        discovery_strata: Sequence[str],
        consolidation_status: str,
        consolidation_evidence: Optional[Mapping[str, object]] = None,
    ) -> SiteTemplate:
        canonical = (
            consolidation_evidence.get("canonical_interval")
            if consolidation_evidence is not None
            else None
        )
        site = _build_site_template(
            reads,
            center=center,
            center_radius=center_radius,
            local_background_radius=local_background_radius,
            minimum_geometry_support=minimum_geometry_support,
            site_id="pending",
            call_type=call_type,
            boundary_reliability_scale=boundary_reliability_scale,
            source_boundary_margin=source_boundary_margin,
            call_catalog=call_catalog,
            fixed_interval=(
                (int(canonical[0]), int(canonical[1]))
                if isinstance(canonical, (list, tuple))
                and len(canonical) == 2
                else None
            ),
            allowed_geometry_records={
                (id(record[2]), id(record[0])) for record in family
            },
            discovery_strata=discovery_strata,
            consolidation_status=consolidation_status,
            consolidation_evidence=consolidation_evidence,
        )
        if consolidation_evidence is not None:
            assignment_probability = consolidation_evidence.get(
                "assignment_probability"
            )
            if assignment_probability is not None:
                reliability_factor = max(
                    0.0, min(1.0, float(assignment_probability))
                )
                site.geometry_reliability *= reliability_factor
                if site.start_geometry_reliability is not None:
                    site.start_geometry_reliability *= reliability_factor
                if site.end_geometry_reliability is not None:
                    site.end_geometry_reliability *= reliability_factor
        return site

    def site_is_eligible(
        site: SiteTemplate, *, provisional_stratum_map: bool = False
    ) -> bool:
        if site.end <= site.start:
            return False
        if max(site.start_mad, site.end_mad) > max_boundary_mad:
            return False
        strongest = max(
            strands, key=lambda strand: site.support.get(strand, 0)
        )
        if provisional_stratum_map:
            # Defer the population-support gate until after cross-strand
            # consolidation. Two sub-threshold strand maps can jointly define
            # a supported latent class (for example 7 CT + 8 GA molecules).
            support_is_adequate = (
                site.support.get(strongest, 0)
                >= minimum_geometry_support
            )
        elif site.consolidation_status == "matched_cross_strand":
            support_is_adequate = sum(site.support.values()) >= min_support
        else:
            support_is_adequate = (
                site.support.get(strongest, 0) >= min_support
            )
        if not support_is_adequate:
            return False
        enrichment = site.local_enrichment_by_strand or {}
        return enrichment.get(strongest, 0.0) >= min_local_enrichment

    sites = []
    family_count = 0
    matched_family_count = 0
    per_strand_map_counts: Dict[str, int] = {}
    pair_diagnostics: List[dict] = []
    within_stratum_exact_duplicate_merges = 0
    if separate_strand_maps:
        if len(strands) > 2:
            raise ValueError(
                "separate strand-map consolidation supports at most two strata"
            )
        maps: Dict[str, List[dict]] = {strand: [] for strand in strands}
        for strand in strands:
            center_groups = _bounded_center_groups(
                proposed_by_strand[strand], center_radius
            )
            for centers in center_groups:
                center = int(round(float(np.median(centers))))
                families = families_near(center, [strand], centers)
                family_count += len(families)
                for family in families:
                    provisional = build_family_site(
                        family,
                        center,
                        [strand],
                        "unconsolidated_stratum_map",
                    )
                    if not site_is_eligible(
                        provisional, provisional_stratum_map=True
                    ):
                        continue
                    maps[strand].append(
                        {
                            "center": center,
                            "records": list(family),
                            "site": provisional,
                        }
                    )
            maps[strand].sort(
                key=lambda value: (
                    value["site"].start,
                    value["site"].end,
                    value["center"],
                )
            )
            # A hard bounded-run partition can split a single high-depth mode
            # into two model states with byte-identical robust geometry.  Such
            # states are observationally and geometrically identical and must
            # be collapsed before any cross-strand assignment.
            deduplicated = []
            for (_start, _end), duplicate_group in itertools.groupby(
                maps[strand],
                key=lambda value: (value["site"].start, value["site"].end),
            ):
                values = list(duplicate_group)
                if len(values) == 1:
                    deduplicated.append(values[0])
                    continue
                within_stratum_exact_duplicate_merges += len(values) - 1
                records = [
                    record for value in values for record in value["records"]
                ]
                center = int(
                    round(float(np.median([value["center"] for value in values])))
                )
                evidence = {
                    "model": "exact_within_stratum_duplicate.v1",
                    "stratum": strand,
                    "interval": [int(_start), int(_end)],
                    "merged_family_count": len(values),
                    "identical_interval": True,
                    "decision": "merged_exact_within_stratum_duplicate",
                }
                provisional = build_family_site(
                    records,
                    center,
                    [strand],
                    "merged_exact_within_stratum_duplicate",
                    evidence,
                )
                deduplicated.append(
                    {
                        "center": center,
                        "records": records,
                        "site": provisional,
                        "within_stratum_duplicate_evidence": evidence,
                    }
                )
            maps[strand] = deduplicated
            per_strand_map_counts[strand] = len(maps[strand])

        consolidated: List[
            Tuple[
                List[_CatalogRecord],
                int,
                Tuple[str, ...],
                str,
                Optional[dict],
            ]
        ] = []
        if len(strands) == 2:
            left_strand, right_strand = strands
            matches, unmatched_left, unmatched_right = (
                _match_stratified_geometry_families(
                    maps[left_strand],
                    maps[right_strand],
                    center_radius=center_radius,
                    edge_compatibility_bp=edge_compatibility_bp,
                    diagnostics=pair_diagnostics,
                )
            )
            matched_family_count = len(matches)
            for left_index, right_index in matches:
                left = maps[left_strand][left_index]
                right = maps[right_strand][right_index]
                evidence = dict(left.get("pair_evidence", {})[right_index])
                zero_strands = set(
                    evidence.get("zero_differential_opportunity_strands", [])
                )
                represented = {left_strand, right_strand}
                informative = sorted(represented - zero_strands)
                if len(zero_strands) == 1 and len(informative) == 1:
                    informative_strand = informative[0]
                    informative_family = (
                        left if informative_strand == left_strand else right
                    )
                    informative_site = informative_family["site"]
                    evidence["canonical_interval"] = [
                        int(informative_site.start),
                        int(informative_site.end),
                    ]
                    evidence["canonical_selection"] = (
                        "informative_stratum_when_other_stratum_has_zero_"
                        "differential_opportunities"
                    )
                    evidence["canonical_informative_stratum"] = (
                        informative_strand
                    )
                else:
                    evidence["canonical_selection"] = (
                        "strand_balanced_robust_median"
                    )
                consolidated.append(
                    (
                        [*left["records"], *right["records"]],
                        int(round(np.median([left["center"], right["center"]]))),
                        (left_strand, right_strand),
                        "matched_cross_strand",
                        evidence,
                    )
                )
            for index in unmatched_left:
                value = maps[left_strand][index]
                consolidated.append(
                    (
                        list(value["records"]),
                        int(value["center"]),
                        (left_strand,),
                        "one_sided_stratum_map",
                        value.get("within_stratum_duplicate_evidence"),
                    )
                )
            for index in unmatched_right:
                value = maps[right_strand][index]
                consolidated.append(
                    (
                        list(value["records"]),
                        int(value["center"]),
                        (right_strand,),
                        "one_sided_stratum_map",
                        value.get("within_stratum_duplicate_evidence"),
                    )
                )
        elif strands:
            strand = strands[0]
            consolidated.extend(
                (
                    list(value["records"]),
                    int(value["center"]),
                    (strand,),
                    "one_sided_stratum_map",
                    value.get("within_stratum_duplicate_evidence"),
                )
                for value in maps[strand]
            )
        for family, center, discovery_strata, status, evidence in consolidated:
            site = build_family_site(
                family,
                center,
                discovery_strata,
                status,
                evidence,
            )
            if site_is_eligible(site):
                sites.append(site)
    else:
        for centers in merged:
            center = int(round(float(np.median(centers))))
            families = families_near(center, strands, centers)
            family_count += len(families)
            for family in families:
                site = build_family_site(
                    family, center, strands, "pooled"
                )
                if site_is_eligible(site):
                    sites.append(site)
    sites.sort(
        key=lambda site: (
            max(site.support.values(), default=0)
            * site.local_enrichment
            / (1.0 + site.start_mad + site.end_mad)
        ),
        reverse=True,
    )
    eligible_before_cap = len(sites)
    cap_bound = max_auto_sites > 0 and eligible_before_cap > max_auto_sites
    if max_auto_sites > 0:
        sites = sites[:max_auto_sites]
    sites.sort(key=lambda site: (site.start, site.end))
    prefix = "site" if call_type == "tf" else f"{call_type}_site"
    for index, site in enumerate(sites, start=1):
        site.site_id = f"{prefix}{index}"
    if diagnostics is not None:
        values = {
            "center_modes": len(merged),
            "geometry_families": family_count,
            "separate_strand_maps": bool(separate_strand_maps),
            "per_strand_map_families": per_strand_map_counts,
            "matched_cross_strand_families": matched_family_count,
            "one_sided_stratum_families": sum(
                site.consolidation_status == "one_sided_stratum_map"
                for site in sites
            ),
            "eligible_before_cap": eligible_before_cap,
            "retained_after_cap": len(sites),
            "max_auto_sites": max_auto_sites,
            "cap_bound": cap_bound,
        }
        if separate_strand_maps:
            values.update(
                {
                    "latent_pair_tests": len(pair_diagnostics),
                    "latent_pair_test_records": pair_diagnostics,
                    "within_stratum_exact_duplicate_merges": (
                        within_stratum_exact_duplicate_merges
                    ),
                }
            )
        diagnostics.update(values)
    return sites


def discover_edge_sites(
    reads: Sequence[ReadEvidence],
    chrom_start: int,
    chrom_end: int,
    *,
    call_type: str,
    min_support: int,
    minimum_geometry_support: int,
    center_radius: int,
    edge_assignment_radius: int,
    max_boundary_mad: float,
    min_local_enrichment: float,
    local_background_radius: int,
    max_auto_sites: int,
    boundary_reliability_scale: float,
    source_boundary_margin: int,
    diagnostics: Optional[Dict[str, object]] = None,
) -> List[SiteTemplate]:
    """Discover multimodal geometry families jointly in start/end space."""
    call_catalog = _build_call_catalog(reads, call_type)
    source_interval_fully_maps = call_catalog.source_interval_fully_maps
    records = []
    for strand, strand_catalog in call_catalog.by_strand.items():
        for call, molecule, read in strand_catalog.records:
            if call.end <= chrom_start or call.start >= chrom_end:
                continue
            if (
                not source_interval_fully_maps(
                    read, call, source_boundary_margin
                )
            ):
                continue
            records.append((call, molecule, strand))
    records.sort(
        key=lambda value: (
            value[0].center,
            value[0].end - value[0].start,
            value[0].start,
            value[0].end,
            value[1],
        )
    )
    clusters = _cluster_edge_records(
        records,
        center_radius=center_radius,
        edge_assignment_radius=edge_assignment_radius,
        use_spatial_index=True,
    )

    strands = sorted({read.strand for read in reads})
    sites = []
    for cluster_state in clusters:
        cluster = cluster_state["records"]
        unique_support = {
            strand: len(
                {
                    molecule
                    for _call, molecule, value_strand in cluster
                    if value_strand == strand
                }
            )
            for strand in strands
        }
        if max(unique_support.values(), default=0) < min_support:
            continue
        starts = [value[0].start for value in cluster]
        ends = [value[0].end for value in cluster]
        seed = (
            int(round(float(np.median(starts)))),
            int(round(float(np.median(ends)))),
        )
        center = int(round((seed[0] + seed[1]) / 2.0))
        site = _build_site_template(
            reads,
            center=center,
            center_radius=center_radius,
            local_background_radius=local_background_radius,
            minimum_geometry_support=minimum_geometry_support,
            site_id="pending",
            call_type=call_type,
            boundary_reliability_scale=boundary_reliability_scale,
            source_boundary_margin=source_boundary_margin,
            seed_interval=seed,
            edge_assignment_radius=edge_assignment_radius,
            call_catalog=call_catalog,
        )
        if site.end <= site.start:
            continue
        if max(site.start_mad, site.end_mad) > max_boundary_mad:
            continue
        strongest = max(strands, key=lambda strand: site.support.get(strand, 0))
        if site.support.get(strongest, 0) < min_support:
            continue
        enrichment = site.local_enrichment_by_strand or {}
        if enrichment.get(strongest, 0.0) < min_local_enrichment:
            continue
        sites.append(site)
    sites.sort(
        key=lambda site: (
            max(site.support.values(), default=0)
            * site.local_enrichment
            / (1.0 + site.start_mad + site.end_mad)
        ),
        reverse=True,
    )
    eligible_before_cap = len(sites)
    cap_bound = max_auto_sites > 0 and eligible_before_cap > max_auto_sites
    if max_auto_sites > 0:
        sites = sites[:max_auto_sites]
    sites.sort(key=lambda site: (site.start, site.end))
    prefix = "site" if call_type == "tf" else f"{call_type}_site"
    for index, site in enumerate(sites, start=1):
        site.site_id = f"{prefix}{index}"
    if diagnostics is not None:
        diagnostics.update(
            {
                "edge_clusters": len(clusters),
                "eligible_before_cap": eligible_before_cap,
                "retained_after_cap": len(sites),
                "max_auto_sites": max_auto_sites,
                "cap_bound": cap_bound,
            }
        )
    return sites


def build_forced_site_template(
    reads: Sequence[ReadEvidence],
    interval: Tuple[int, int],
    *,
    site_id: str,
    center_radius: int,
    minimum_geometry_support: int = 3,
    call_type: str = "tf",
    boundary_reliability_scale: float = 12.0,
    source_boundary_margin: int = 0,
    edge_assignment_radius: Optional[int] = None,
) -> SiteTemplate:
    """Use an external seed while learning canonical edges from ordinary calls."""
    start, end = interval
    center = int(round((start + end) / 2.0))
    site = _build_site_template(
        reads,
        center=center,
        center_radius=center_radius,
        local_background_radius=250,
        minimum_geometry_support=minimum_geometry_support,
        site_id=site_id,
        call_type=call_type,
        boundary_reliability_scale=boundary_reliability_scale,
        source_boundary_margin=source_boundary_margin,
        seed_interval=interval,
        edge_assignment_radius=edge_assignment_radius,
    )
    if site.end <= site.start:
        site.start = start
        site.end = end
        site.center = center
    return site


def merge_forced_sites(
    automatic: Sequence[SiteTemplate],
    forced_intervals: Sequence[Tuple[int, int]],
    reads: Sequence[ReadEvidence],
    *,
    center_radius: int,
    minimum_geometry_support: int = 3,
    forced_only: bool,
    call_type: str = "tf",
    boundary_reliability_scale: float = 12.0,
    source_boundary_margin: int = 0,
    edge_assignment_radius: Optional[int] = None,
) -> List[SiteTemplate]:
    sites = [] if forced_only else list(automatic)
    for interval in forced_intervals:
        center = (interval[0] + interval[1]) / 2.0
        sites = [site for site in sites if abs(site.center - center) > center_radius]
        sites.append(
            build_forced_site_template(
                reads,
                interval,
                site_id="forced",
                center_radius=center_radius,
                minimum_geometry_support=minimum_geometry_support,
                call_type=call_type,
                boundary_reliability_scale=boundary_reliability_scale,
                source_boundary_margin=source_boundary_margin,
                edge_assignment_radius=edge_assignment_radius,
            )
        )
    sites.sort(key=lambda site: (site.start, site.end))
    prefix = "site" if call_type == "tf" else f"{call_type}_site"
    for index, site in enumerate(sites, start=1):
        site.site_id = f"{prefix}{index}"
    return sites


def shift_site_templates(
    sites: Sequence[SiteTemplate], shift: int
) -> List[SiteTemplate]:
    shifted = []
    for site in sites:
        start = site.start + int(shift)
        end = site.end + int(shift)
        center = site.center + int(shift)
        if start < 0:
            raise ValueError("shifted site starts before coordinate zero")
        shifted.append(
            SiteTemplate(
                **{
                    **asdict(site),
                    "start": start,
                    "end": end,
                    "center": center,
                }
            )
        )
    return shifted


def configuration_log_likelihoods(
    read: ReadEvidence,
    sites: Sequence[SiteTemplate],
    configurations: Sequence[Configuration],
    call: IntervalCall,
) -> Tuple[np.ndarray, List[Tuple[float, int, int]]]:
    site_evidence = [
        read.interval_evidence(max(call.start, site.start), min(call.end, site.end))
        for site in sites
    ]
    nuc_likelihood, _opportunities, _hits = read.interval_evidence(
        call.start, call.end
    )
    values = []
    for configuration in configurations:
        if configuration.is_nucleosome:
            values.append(nuc_likelihood)
        else:
            values.append(
                sum(site_evidence[index][0] for index in configuration.site_indices)
            )
    return np.asarray(values, dtype=np.float64), site_evidence


def fit_site_state_model(
    reads: Sequence[ReadEvidence],
    site: SiteTemplate,
    *,
    flank: int = 35,
    center_radius: int = 10,
    boundary_band: int = 12,
    edge_compatibility_bp: int = 12,
) -> dict:
    """Fit A/TF/N weights while anchoring already accepted TF and nuc calls."""
    if boundary_band < 0:
        raise ValueError("boundary band must be non-negative")
    if edge_compatibility_bp < 0:
        raise ValueError("edge compatibility must be non-negative")
    configurations = enumerate_configurations([site], include_nucleosome=True)
    eligible = [read for read in reads if read.fully_maps(site.start, site.end)]
    center_coverage = len(
        {
            read.molecule_id
            for read in reads
            if read.fully_maps(site.center, site.center + 1)
        }
    )
    site_coverage = len({read.molecule_id for read in eligible})
    tf_index = next(
        index
        for index, configuration in enumerate(configurations)
        if configuration.site_indices
    )
    nuc_index = next(
        index
        for index, configuration in enumerate(configurations)
        if configuration.is_nucleosome
    )
    matrix = []
    anchored_tf = 0
    anchored_nuc = 0
    canonical_explicit_tf_molecules = set()
    canonical_explicit_tf_with_opportunities = set()
    direct_tf_without_site_opportunities = set()
    opportunity_eligible_molecules = set()
    positive_evidence_molecules = set()
    tf_callable_molecules = set()
    opportunity_count_by_molecule: Dict[Tuple[str, str, str], int] = {}
    window = IntervalCall(max(0, site.start - flank), site.end + flank)
    for read in eligible:
        direct_tf = any(
            abs(call.center - site.center) <= center_radius
            and abs(call.start - site.start) <= edge_compatibility_bp
            and abs(call.end - site.end) <= edge_compatibility_bp
            for call in _calls_for_type(read, "tf")
        )
        direct_nuc = any(
            nuc.start <= site.center < nuc.end
            for nuc in _calls_for_type(read, "nuc")
        )
        site_llr, site_opportunities, _site_hits = read.interval_evidence(
            site.start, site.end
        )
        opportunity_count_by_molecule[read.molecule_id] = max(
            opportunity_count_by_molecule.get(read.molecule_id, 0),
            int(site_opportunities),
        )
        if site_opportunities > 0:
            opportunity_eligible_molecules.add(read.molecule_id)
        if site_llr > 0.0:
            positive_evidence_molecules.add(read.molecule_id)
        if direct_tf or site_opportunities > 0:
            tf_callable_molecules.add(read.molecule_id)
        if direct_tf:
            canonical_explicit_tf_molecules.add(read.molecule_id)
            if site_opportunities > 0:
                canonical_explicit_tf_with_opportunities.add(read.molecule_id)
            else:
                direct_tf_without_site_opportunities.add(read.molecule_id)
            values = np.full(len(configurations), -np.inf, dtype=np.float64)
            values[tf_index] = 0.0
            anchored_tf += 1
        elif direct_nuc:
            values = np.full(len(configurations), -np.inf, dtype=np.float64)
            values[nuc_index] = 0.0
            anchored_nuc += 1
        else:
            values, _evidence = configuration_log_likelihoods(
                read, [site], configurations, window
            )
        matrix.append(values)
    if matrix:
        weights, iterations = fit_mixture_weights(np.vstack(matrix))
    else:
        weights = np.full(len(configurations), 1.0 / len(configurations))
        iterations = 0
    by_state = {}
    for index, configuration in enumerate(configurations):
        state = (
            "N"
            if configuration.is_nucleosome
            else "TF"
            if configuration.site_indices
            else "A"
        )
        by_state[state] = float(weights[index])
    def opportunity_summary(start: int, end: int) -> dict:
        if end <= start:
            return {
                "interval": [int(start), int(end)],
                "fully_mapped_molecules": 0,
                "molecules_with_opportunity": 0,
                "molecules_with_positive_evidence": 0,
                "total_opportunities": 0,
            }
        coverage = set()
        with_opportunity = set()
        with_positive_evidence = set()
        opportunities_by_molecule: Dict[Tuple[str, str, str], int] = {}
        for read in reads:
            if not read.fully_maps(start, end):
                continue
            molecule_id = read.molecule_id
            coverage.add(molecule_id)
            llr, opportunities, _hits = read.interval_evidence(start, end)
            opportunities_by_molecule[molecule_id] = max(
                opportunities_by_molecule.get(molecule_id, 0),
                int(opportunities),
            )
            if opportunities > 0:
                with_opportunity.add(molecule_id)
            if llr > 0.0:
                with_positive_evidence.add(molecule_id)
        return {
            "interval": [int(start), int(end)],
            "fully_mapped_molecules": len(coverage),
            "molecules_with_opportunity": len(with_opportunity),
            "molecules_with_positive_evidence": len(with_positive_evidence),
            "total_opportunities": int(sum(opportunities_by_molecule.values())),
        }

    opportunity_by_segment = {
        "left_boundary": opportunity_summary(
            max(0, site.start - boundary_band), site.start
        ),
        "interior": opportunity_summary(site.start, site.end),
        "right_boundary": opportunity_summary(
            site.end, site.end + boundary_band
        ),
    }
    return {
        "coverage": len(eligible),
        "site_coverage": site_coverage,
        "center_coverage": int(center_coverage),
        "anchored_tf_calls": anchored_tf,
        "canonical_explicit_tf_molecules": len(
            canonical_explicit_tf_molecules
        ),
        "canonical_explicit_tf_with_opportunities": len(
            canonical_explicit_tf_with_opportunities
        ),
        "direct_tf_without_site_opportunities": len(
            direct_tf_without_site_opportunities
        ),
        "anchored_nuc_calls": anchored_nuc,
        "chemistry_only_reads": len(eligible) - anchored_tf - anchored_nuc,
        "opportunity_eligible_molecules": len(opportunity_eligible_molecules),
        "positive_evidence_molecules": len(positive_evidence_molecules),
        "tf_callable_molecules": len(tf_callable_molecules),
        "total_site_opportunities": int(
            sum(opportunity_count_by_molecule.values())
        ),
        "opportunity_by_segment": opportunity_by_segment,
        "iterations": iterations,
        "weights": by_state,
    }


def explicit_call_fraction(
    site: SiteTemplate,
    strand: str,
    site_model: Mapping[str, object],
) -> float:
    """Accepted-call fraction among molecules spanning the complete site."""
    site_coverage = int(
        site_model.get(
            "site_coverage",
            site_model.get("coverage", site_model.get("center_coverage", 0)),
        )
    )
    support = int(
        site_model.get(
            "canonical_explicit_tf_molecules",
            min(int(site.support.get(strand, 0)), site_coverage),
        )
    )
    if support > site_coverage:
        raise ValueError(
            "canonical explicit TF calls exceed fully-spanning molecule coverage"
        )
    denominator = site_coverage
    return support / denominator if denominator else 0.0


def opportunity_conditioned_call_fraction(
    site: SiteTemplate,
    strand: str,
    site_model: Mapping[str, object],
) -> float:
    """Accepted-call rate among molecules with raw site opportunities.

    This is an assay-detection quantity, not an occupancy estimate. The
    targeted rescue pass compares this rate between strands to choose the
    source-to-target direction; raw strand depth is not used for that gate. A
    direct call without a represented raw C/G opportunity remains separately
    audited and does not make this conditional detection rate look more
    favorable.
    """
    denominator = int(
        site_model.get(
            "opportunity_eligible_molecules",
            site_model.get("site_coverage", site_model.get("coverage", 0)),
        )
    )
    support = int(
        site_model.get(
            "canonical_explicit_tf_with_opportunities",
            min(int(site.support.get(strand, 0)), denominator),
        )
    )
    if support > denominator:
        raise ValueError(
            "opportunity-bearing explicit TF calls exceed opportunity coverage"
        )
    return support / denominator if denominator else 0.0


def fit_tf_configuration_class_model(
    reads: Sequence[ReadEvidence],
    sites: Sequence[SiteTemplate],
    configurations: Sequence[Configuration],
    *,
    center_radius: int = 10,
    pseudocount: float = 0.5,
    source_stratum: Optional[str] = None,
) -> dict:
    """Learn a localized prior over complete TF footprint configurations.

    A class can be one atomic footprint or a compatible set of smaller
    footprints. Accepted calls anchor observed classes. Molecules without a
    direct class assignment contribute chemistry likelihoods, while molecules
    carrying a nucleosome or an unmodeled local TF topology are excluded from
    this non-nucleosome action prior and remain auditable.
    """
    if not sites:
        raise ValueError("TF configuration class model requires at least one site")
    if not configurations or any(
        configuration.is_nucleosome for configuration in configurations
    ):
        raise ValueError(
            "TF configuration class model requires non-nucleosome configurations"
        )
    if not math.isfinite(pseudocount) or pseudocount < 0.0:
        raise ValueError("class-model pseudocount must be finite and non-negative")

    envelope = IntervalCall(
        min(site.start for site in sites), max(site.end for site in sites)
    )
    configuration_by_signature = {
        configuration.site_indices: index
        for index, configuration in enumerate(configurations)
    }
    eligible_by_molecule: Dict[Tuple[str, str, str], ReadEvidence] = {}
    for read in reads:
        if not read.fully_maps(envelope.start, envelope.end):
            continue
        previous = eligible_by_molecule.get(read.molecule_id)
        if previous is None:
            eligible_by_molecule[read.molecule_id] = read
            continue
        previous_opportunities = previous.interval_evidence(
            envelope.start, envelope.end
        )[1]
        current_opportunities = read.interval_evidence(
            envelope.start, envelope.end
        )[1]
        if (
            current_opportunities,
            -int(read.input_record_ordinal or 0),
        ) > (
            previous_opportunities,
            -int(previous.input_record_ordinal or 0),
        ):
            eligible_by_molecule[read.molecule_id] = read

    matrix = []
    hard_support = np.zeros(len(configurations), dtype=np.int64)
    anchored_molecules = 0
    chemistry_only_molecules = 0
    nucleosome_excluded_molecules = 0
    unmodeled_topology_molecules = 0
    total_opportunities = 0
    for molecule_id in sorted(eligible_by_molecule):
        read = eligible_by_molecule[molecule_id]
        local_calls = [
            call
            for call in _calls_for_type(read, "tf")
            if call.start < envelope.end and envelope.start < call.end
        ]
        matched_calls = match_geometry_calls(
            local_calls, sites, center_radius=center_radius
        )
        assigned_signature = tuple(
            sorted(
                site_index
                for _call, site_index, assigned in matched_calls
                if assigned
            )
        )
        has_unassigned_local_call = bool(local_calls) and (
            len(assigned_signature) != len(local_calls)
        )
        configuration_index = configuration_by_signature.get(
            assigned_signature
        )
        if has_unassigned_local_call or (
            assigned_signature and configuration_index is None
        ):
            unmodeled_topology_molecules += 1
            continue
        if any(
            nuc.start < envelope.end and envelope.start < nuc.end
            for nuc in _calls_for_type(read, "nuc")
        ):
            nucleosome_excluded_molecules += 1
            continue
        _llr, opportunities, _hits = read.interval_evidence(
            envelope.start, envelope.end
        )
        total_opportunities += int(opportunities)
        if assigned_signature:
            values = np.full(
                len(configurations), -np.inf, dtype=np.float64
            )
            values[int(configuration_index)] = 0.0
            hard_support[int(configuration_index)] += 1
            anchored_molecules += 1
        else:
            values, _site_evidence = configuration_log_likelihoods(
                read, sites, configurations, envelope
            )
            chemistry_only_molecules += 1
        matrix.append(values)

    if matrix:
        weights, iterations = fit_mixture_weights(
            np.vstack(matrix), pseudocount=pseudocount
        )
    else:
        weights = np.full(
            len(configurations), 1.0 / len(configurations), dtype=np.float64
        )
        iterations = 0
    weights = np.maximum(weights, 1e-12)
    weights /= np.sum(weights)

    model_payload = "\x1f".join(
        [
            str(source_stratum or "."),
            *(f"{site.site_id}:{site.start}-{site.end}" for site in sites),
        ]
    )
    model_id = "tfclassmodel_" + hashlib.sha256(
        model_payload.encode("utf-8")
    ).hexdigest()[:16]
    classes = []
    for index, configuration in enumerate(configurations):
        component_sites = [
            sites[site_index] for site_index in configuration.site_indices
        ]
        signature = ",".join(
            site.site_id for site in component_sites
        ) or "accessible"
        class_id = "tfclass_" + hashlib.sha256(
            f"{model_id}\x1f{signature}".encode("utf-8")
        ).hexdigest()[:16]
        classes.append(
            {
                "class_id": class_id,
                "configuration": configuration.name,
                "state": "TF" if component_sites else "A",
                "component_site_ids": [
                    site.site_id for site in component_sites
                ],
                "component_intervals": [
                    [site.start, site.end] for site in component_sites
                ],
                "envelope": (
                    [
                        min(site.start for site in component_sites),
                        max(site.end for site in component_sites),
                    ]
                    if component_sites
                    else [envelope.start, envelope.end]
                ),
                "probability": float(weights[index]),
                "anchored_molecule_support": int(hard_support[index]),
            }
        )
    return {
        "schema": "fiberhmm.tf_configuration_class_model.v1",
        "model_id": model_id,
        "source_stratum": source_stratum,
        "envelope": [envelope.start, envelope.end],
        "site_ids": [site.site_id for site in sites],
        "eligible_molecules": len(eligible_by_molecule),
        "modeled_molecules": len(matrix),
        "anchored_molecules": anchored_molecules,
        "chemistry_only_molecules": chemistry_only_molecules,
        "nucleosome_excluded_molecules": nucleosome_excluded_molecules,
        "unmodeled_topology_molecules": unmodeled_topology_molecules,
        "total_opportunities": total_opportunities,
        "pseudocount_per_configuration": float(pseudocount),
        "iterations": iterations,
        "configuration_probabilities": [
            float(value) for value in weights
        ],
        "classes": classes,
    }


@lru_cache(maxsize=512)
def _gauss_legendre_unit_interval(
    quadrature_points: int,
) -> Tuple[np.ndarray, np.ndarray]:
    nodes, weights = np.polynomial.legendre.leggauss(quadrature_points)
    return (nodes + 1.0) / 2.0, weights / 2.0


def _diffuse_unmodeled_configuration_log_likelihood(
    read: ReadEvidence,
    envelope: IntervalCall,
    *,
    quadrature_points: Optional[int] = None,
) -> float:
    """Marginal LLR for an unstructured protected-opportunity component.

    Conditional on ``rho``, every represented opportunity is independently
    protected with probability ``rho``.  Integrating ``rho`` under a uniform
    Beta(1, 1) prior gives a proper, deliberately diffuse alternative to the
    spatially coherent TF configurations.  It absorbs chemistry patterns that
    the current structured library cannot explain without using ordinary call
    labels as likelihood terms or teaching any TF boundary.
    """
    lo = int(np.searchsorted(read.positions, envelope.start, side="left"))
    hi = int(np.searchsorted(read.positions, envelope.end, side="left"))
    if hi <= lo:
        return 0.0
    steps = np.asarray(read.steps[lo:hi], dtype=np.float64)
    if quadrature_points is None:
        # The integrand is a degree-N polynomial in rho.  N-point
        # Gauss-Legendre quadrature integrates degree 2N-1 exactly, so this
        # choice removes the severe high-opportunity bias of a fixed 16-point
        # rule while retaining stable log-space evaluation.
        quadrature_points = max(16, (len(steps) + 2) // 2)
    if quadrature_points < 2:
        raise ValueError("unmodeled quadrature requires at least two points")
    rho, quadrature_weights = _gauss_legendre_unit_interval(
        int(quadrature_points)
    )
    conditional = np.sum(
        np.logaddexp(
            np.log1p(-rho)[:, None],
            np.log(rho)[:, None] + steps[None, :],
        ),
        axis=1,
    )
    return float(
        _logsumexp(
            conditional + np.log(quadrature_weights), axis=0
        )
    )


def _spatial_null_candidate_intervals(
    envelope: IntervalCall,
    structured_candidates: Sequence[Sequence[Tuple[int, int]]],
    *,
    minimum_width: int = 6,
    maximum_width: int = 80,
) -> List[Tuple[int, int]]:
    """Enumerate a de-nested, placement-marginalized spatial residual.

    Width has a discrete uniform prior over the stated physical TF-footprint
    range; start is uniform conditional on width.  Exact reference intervals
    present in any anchored candidate grid are excluded, so the residual is
    not nested with a named class.  Placements that project identically on a
    sparse opportunity lattice are deliberately retained: together they are
    the correct prior mass under a reference-coordinate prior.
    """
    if minimum_width < 1 or maximum_width < minimum_width:
        raise ValueError("invalid spatial-null width range")
    span = envelope.end - envelope.start
    widths = range(minimum_width, min(maximum_width, span) + 1)
    anchored = {
        (int(start), int(end))
        for candidates in structured_candidates
        for start, end in candidates
    }
    intervals = [
        (start, start + width)
        for width in widths
        for start in range(envelope.start, envelope.end - width + 1)
        if (start, start + width) not in anchored
    ]
    if not intervals:
        raise ValueError("spatial null requires at least one admissible interval")
    return intervals


def _spatial_null_configuration_log_likelihood(
    read: ReadEvidence,
    candidate_intervals: Sequence[Tuple[int, int]],
) -> float:
    """Marginal LLR for one unanchored, spatially coherent footprint.

    Each molecule may carry one contiguous protected interval, but its genomic
    placement is integrated out rather than optimized.  The log-mean-exp is
    the essential look-elsewhere penalty: a recurrent anchored TF class can
    outperform this component only when the same location repeatedly explains
    independent molecules.
    """
    if not candidate_intervals:
        raise ValueError("spatial null candidate interval set cannot be empty")
    starts, ends, log_prior = _prepare_spatial_null_interval_grid(
        candidate_intervals
    )
    scores = read._interval_evidence_batch(starts, ends)[0]
    return float(_logsumexp(scores + log_prior, axis=0))


def _prepare_spatial_null_interval_grid(
    candidate_intervals: Sequence[Tuple[int, int]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Prepare one shared float64 spatial-null grid for many molecules."""
    if not candidate_intervals:
        raise ValueError("spatial null candidate interval set cannot be empty")
    starts = np.fromiter(
        (start for start, _end in candidate_intervals), dtype=np.int64
    )
    ends = np.fromiter(
        (end for _start, end in candidate_intervals), dtype=np.int64
    )
    widths = ends - starts
    unique_widths, counts = np.unique(widths, return_counts=True)
    count_by_width = {
        int(width): int(count) for width, count in zip(unique_widths, counts)
    }
    log_prior = np.asarray(
        [
            -math.log(len(unique_widths)) - math.log(count_by_width[int(width)])
            for width in widths
        ],
        dtype=np.float64,
    )
    return starts, ends, log_prior


def _spatial_null_configuration_log_likelihoods(
    reads: Sequence[ReadEvidence],
    candidate_intervals: Sequence[Tuple[int, int]],
) -> np.ndarray:
    """Evaluate a prepared spatial-null grid without rebuilding it per read."""
    likelihoods, _localization = _spatial_null_configuration_evidence(
        reads, candidate_intervals
    )
    return likelihoods


def _spatial_null_log_likelihoods_from_prepared_grid(
    reads: Sequence[ReadEvidence],
    starts: np.ndarray,
    ends: np.ndarray,
    log_prior: np.ndarray,
) -> np.ndarray:
    """Reference CPU kernel for an already prepared spatial-null grid."""
    starts = np.asarray(starts, dtype=np.int64)
    ends = np.asarray(ends, dtype=np.int64)
    log_prior = np.asarray(log_prior, dtype=np.float64)
    if starts.shape != ends.shape or starts.shape != log_prior.shape:
        raise ValueError("prepared spatial-null arrays must align")
    return np.asarray(
        [
            float(
                _logsumexp(
                    read._interval_evidence_batch(starts, ends)[0] + log_prior,
                    axis=0,
                )
            )
            for read in reads
        ],
        dtype=np.float64,
    )


def _spatial_null_configuration_evidence(
    reads: Sequence[ReadEvidence],
    candidate_intervals: Sequence[Tuple[int, int]],
    *,
    localization_edges: Sequence[Tuple[int, int, int]] = (),
) -> Tuple[np.ndarray, np.ndarray]:
    """Return P0 likelihoods and edge-localized placement posterior masses.

    Each localization tuple is ``(coordinate_index, minimum, maximum)`` where
    coordinate index 0 selects interval starts, 1 selects interval ends, and
    -1 selects intervals whose center lies inside the supplied site envelope.
    This prevents the posterior of one unanchored footprint elsewhere in the
    envelope from being credited to every modeled site edge.
    """
    starts, ends, log_prior = _prepare_spatial_null_interval_grid(
        candidate_intervals
    )
    coordinates = (starts, ends)
    localization_masks = []
    for coordinate_index, minimum, maximum in localization_edges:
        if coordinate_index not in {-1, 0, 1} or maximum < minimum:
            raise ValueError("invalid spatial-null edge localization")
        if coordinate_index == -1:
            center_twice = starts + ends
            localization_masks.append(
                (center_twice >= 2 * int(minimum))
                & (center_twice <= 2 * int(maximum))
            )
        else:
            values = coordinates[coordinate_index]
            localization_masks.append(
                (values >= int(minimum)) & (values <= int(maximum))
            )
    likelihoods = np.empty(len(reads), dtype=np.float64)
    localization = np.zeros(
        (len(reads), len(localization_masks)), dtype=np.float64
    )
    for read_index, read in enumerate(reads):
        logits = read._interval_evidence_batch(starts, ends)[0] + log_prior
        total = float(_logsumexp(logits, axis=0))
        likelihoods[read_index] = total
        for localization_index, mask in enumerate(localization_masks):
            if np.any(mask):
                localization[read_index, localization_index] = float(
                    np.exp(_logsumexp(logits[mask], axis=0) - total)
                )
    return likelihoods, localization


def _interval_evidence_matrix(
    reads: Sequence[ReadEvidence],
    intervals: Sequence[Tuple[int, int]],
    *,
    summation_mode: str = "prefix",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return float64 interval scores and opportunity lattice projections."""
    if not intervals:
        raise ValueError("interval evidence matrix requires a non-empty grid")
    starts = np.fromiter((start for start, _end in intervals), dtype=np.int64)
    ends = np.fromiter((end for _start, end in intervals), dtype=np.int64)
    if summation_mode not in {"prefix", "slice_sum"}:
        raise ValueError("invalid interval evidence summation mode")
    score_matrix = np.empty((len(reads), len(intervals)), dtype=np.float64)
    left_matrix = np.empty((len(reads), len(intervals)), dtype=np.int64)
    right_matrix = np.empty((len(reads), len(intervals)), dtype=np.int64)
    for read_index, read in enumerate(reads):
        scores, _opportunities, _hits, left, right = (
            read._interval_evidence_batch(starts, ends)
        )
        score_matrix[read_index] = (
            scores
            if summation_mode == "prefix"
            else np.asarray(
                [
                    np.sum(read.steps[int(lo) : int(hi)], dtype=np.float64)
                    for lo, hi in zip(left, right)
                ],
                dtype=np.float64,
            )
        )
        left_matrix[read_index] = left
        right_matrix[read_index] = right
    return score_matrix, left_matrix, right_matrix


def _read_evidence_content_sha256(read: ReadEvidence) -> str:
    """Stable evidence identity for order-independent molecule tie-breaking."""
    if read.record_sha256:
        return str(read.record_sha256)
    digest = hashlib.sha256()
    for value in (
        str(read.library_id or ""),
        read.name,
        read.strand,
        str(read.ref_start),
        str(read.ref_end),
        str(read.alignment_flag),
        str(read.cigar or ""),
    ):
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
    for array_value, dtype in (
        (read.positions, "<i8"),
        (read.steps, "<f8"),
        (read.hits, "u1"),
        (read.contexts, "<i8"),
    ):
        payload = np.asarray(array_value, dtype=dtype).tobytes()
        digest.update(len(payload).to_bytes(8, "little"))
        digest.update(payload)
    for call_type, calls in (
        ("tf", read.tfs),
        ("nuc", read.nucs),
        ("msp", read.msps),
    ):
        digest.update(call_type.encode("ascii"))
        for call in calls:
            digest.update(int(call.start).to_bytes(8, "little", signed=True))
            digest.update(int(call.end).to_bytes(8, "little", signed=True))
    return digest.hexdigest()


def fit_iterative_tf_class_geometry_model(
    reads: Sequence[ReadEvidence],
    sites: Sequence[SiteTemplate],
    configurations: Optional[Sequence[Configuration]] = None,
    *,
    boundary_search_radius: int = 2,
    pseudocount: float = 0.5,
    center_radius: int = 10,
    max_iter: int = 100,
    tol: float = 1e-8,
    initialization_mode: str = "ordinary_calls",
    initial_geometry_intervals: Optional[Sequence[Tuple[int, int]]] = None,
    tie_break_mode: str = "seed",
    analysis_envelope: Optional[Tuple[int, int]] = None,
    spatial_null_exclusion_intervals: Optional[
        Sequence[Tuple[int, int]]
    ] = None,
    stratum_semantics: str = "unspecified",
    spatial_null_padding: int = 20,
    spatial_null_minimum_width: int = 1,
    spatial_null_maximum_width: int = 80,
    minimum_molecule_opportunities: int = 3,
    minimum_edge_effective_opportunities: float = 10.0,
    minimum_edge_information_spread: float = math.log(10.0),
    minimum_edge_q_margin_per_effective_molecule: float = 0.1,
    minimum_spatial_null_conflict_fraction: float = 0.1,
) -> dict:
    """Fit pooled molecule configurations and discrete TF-class boundaries.

    This is the report-only outer geometry iteration for targeted loci.  The
    latent assignment unit is one complete non-overlapping configuration per
    collapsed molecule. Ordinary calls initialize mixture weights and bound
    the candidate library, but their deterministic segmentation is not
    multiplied into the raw-chemistry likelihood and never hard-anchors an
    exact edge. Geometry updates maximize the expected complete-data
    likelihood over a fixed local integer grid. Coordinates tied on every
    represented opportunity lattice remain an explicit equivalence set.

    A placement-marginalized spatial null captures one coherent footprint at
    an arbitrary local position, while a proper diffuse opportunity-level
    component captures non-spatial patterns.  Together they keep every fully
    mapped molecule with the configured minimum represented opportunities in
    the likelihood and nominate residuals for the future outer proposal pass;
    neither component updates a TF edge.  Under-opportunity molecules are
    counted explicitly but cannot contribute support.

    The atomic class set and structured configuration topology are fixed in this stage.
    Held-out merge/split selection is deliberately a later outer operation;
    this function must not be used as a calibrated biological caller yet.
    """
    if not sites:
        raise ValueError("iterative TF geometry requires at least one site")
    if boundary_search_radius < 0:
        raise ValueError("boundary search radius must be non-negative")
    if not math.isfinite(pseudocount) or pseudocount < 0.0:
        raise ValueError("pseudocount must be finite and non-negative")
    if max_iter < 1:
        raise ValueError("maximum iterations must be positive")
    if not math.isfinite(tol) or tol <= 0.0:
        raise ValueError("convergence tolerance must be finite and positive")
    if initialization_mode not in {"ordinary_calls", "uniform"}:
        raise ValueError(
            "initialization mode must be 'ordinary_calls' or 'uniform'"
        )
    if tie_break_mode not in {"seed", "initial"}:
        raise ValueError("tie-break mode must be 'seed' or 'initial'")
    if stratum_semantics not in {
        "physical_complementary",
        "diagnostic_partition",
        "unspecified",
    }:
        raise ValueError("invalid stratum semantics")
    if spatial_null_padding < 1:
        raise ValueError("spatial-null padding must be positive")
    if spatial_null_minimum_width < 1 or (
        spatial_null_maximum_width < spatial_null_minimum_width
    ):
        raise ValueError("invalid spatial-null width range")
    if minimum_molecule_opportunities < 1:
        raise ValueError("minimum molecule opportunities must be positive")
    if minimum_edge_effective_opportunities < 0.0 or (
        not math.isfinite(minimum_edge_effective_opportunities)
    ):
        raise ValueError("invalid minimum edge effective opportunities")
    if minimum_edge_information_spread < 0.0 or (
        not math.isfinite(minimum_edge_information_spread)
    ):
        raise ValueError("invalid minimum edge information spread")
    if minimum_edge_q_margin_per_effective_molecule < 0.0 or (
        not math.isfinite(minimum_edge_q_margin_per_effective_molecule)
    ):
        raise ValueError("invalid minimum edge Q margin per effective molecule")
    if not 0.0 <= minimum_spatial_null_conflict_fraction <= 1.0 or (
        not math.isfinite(minimum_spatial_null_conflict_fraction)
    ):
        raise ValueError("invalid minimum spatial-null conflict fraction")
    input_sites = list(sites)
    if initial_geometry_intervals is not None and len(
        initial_geometry_intervals
    ) != len(input_sites):
        raise ValueError(
            "initial geometry interval count must match the site count"
        )
    site_order = sorted(
        range(len(input_sites)),
        key=lambda index: (
            input_sites[index].start,
            input_sites[index].end,
            input_sites[index].site_id,
            index,
        ),
    )
    old_to_new = {old: new for new, old in enumerate(site_order)}
    sites = [input_sites[index] for index in site_order]
    family_substate_keys = [
        (
            str(site.family_id or site.site_id),
            str(site.substate_id or site.site_id),
        )
        for site in sites
    ]
    if len(family_substate_keys) != len(set(family_substate_keys)):
        raise ValueError(
            "TF-family catalog contains a duplicate family/substate identity"
        )
    if initial_geometry_intervals is not None:
        initial_geometry_intervals = [
            tuple(initial_geometry_intervals[index]) for index in site_order
        ]
    if configurations is None:
        configuration_list = enumerate_configurations(
            sites, include_nucleosome=False
        )
    else:
        if any(
            site_index < 0 or site_index >= len(input_sites)
            for configuration in configurations
            for site_index in configuration.site_indices
        ):
            raise ValueError("configuration references an unavailable site")
        configuration_list = [
            Configuration(
                configuration.name,
                tuple(
                    sorted(old_to_new[site_index] for site_index in configuration.site_indices)
                ),
                is_nucleosome=configuration.is_nucleosome,
            )
            for configuration in configurations
        ]
    configuration_list = sorted(
        configuration_list,
        key=lambda configuration: (
            bool(configuration.site_indices),
            tuple(
                sites[index].site_id for index in configuration.site_indices
            ),
            configuration.name,
        ),
    )
    if not configuration_list or any(
        configuration.is_nucleosome
        for configuration in configuration_list
    ):
        raise ValueError(
            "iterative TF geometry requires non-nucleosome configurations"
        )
    if not any(not configuration.site_indices for configuration in configuration_list):
        raise ValueError("configuration library must contain an accessible state")
    normalized_configuration_signatures = [
        tuple(sorted(configuration.site_indices))
        for configuration in configuration_list
    ]
    if len(normalized_configuration_signatures) != len(
        set(normalized_configuration_signatures)
    ):
        raise ValueError("configuration library contains duplicate site sets")
    for configuration in configuration_list:
        configuration_family_ids = [
            str(sites[index].family_id or sites[index].site_id)
            for index in configuration.site_indices
        ]
        if len(configuration_family_ids) != len(
            set(configuration_family_ids)
        ):
            raise ValueError(
                "configuration contains multiple substates of one site-consensus state"
            )
        ordered_indices = sorted(
            configuration.site_indices,
            key=lambda index: (sites[index].start, sites[index].end, index),
        )
        if any(
            sites[left].end > sites[right].start
            for left, right in zip(ordered_indices, ordered_indices[1:])
        ):
            raise ValueError(
                "configuration contains overlapping atomic site intervals"
            )

    geometry_envelope = IntervalCall(
        min(site.start for site in sites) - boundary_search_radius,
        max(site.end for site in sites) + boundary_search_radius,
    )
    analysis_padding = max(boundary_search_radius, spatial_null_padding)
    default_envelope = IntervalCall(
        min(site.start for site in sites) - analysis_padding,
        max(site.end for site in sites) + analysis_padding,
    )
    if analysis_envelope is None:
        envelope = default_envelope
        analysis_envelope_explicit = False
    else:
        if len(analysis_envelope) != 2:
            raise ValueError("analysis envelope must contain start and end")
        envelope = IntervalCall(
            int(analysis_envelope[0]), int(analysis_envelope[1])
        )
        if envelope.end <= envelope.start:
            raise ValueError("analysis envelope end must exceed start")
        if (
            envelope.start > geometry_envelope.start
            or envelope.end < geometry_envelope.end
        ):
            raise ValueError(
                "analysis envelope must contain every geometry candidate"
            )
        analysis_envelope_explicit = True
    eligible_by_molecule: Dict[Tuple[str, str, str], ReadEvidence] = {}
    for read in reads:
        if not read.fully_maps(envelope.start, envelope.end):
            continue
        previous = eligible_by_molecule.get(read.molecule_id)
        if previous is None:
            eligible_by_molecule[read.molecule_id] = read
            continue
        previous_opportunities = previous.interval_evidence(
            envelope.start, envelope.end
        )[1]
        current_opportunities = read.interval_evidence(
            envelope.start, envelope.end
        )[1]
        if current_opportunities > previous_opportunities or (
            current_opportunities == previous_opportunities
            and _read_evidence_content_sha256(read)
            < _read_evidence_content_sha256(previous)
        ):
            eligible_by_molecule[read.molecule_id] = read

    configuration_by_signature = {
        tuple(sorted(configuration.site_indices)): index
        for index, configuration in enumerate(configuration_list)
    }
    accessible_index = next(
        index
        for index, configuration in enumerate(configuration_list)
        if not configuration.site_indices
    )
    retained_reads: List[ReadEvidence] = []
    spatial_null_component_index = len(configuration_list)
    unmodeled_component_index = len(configuration_list) + 1
    component_count = len(configuration_list) + 2
    seed_counts = np.zeros(component_count, dtype=np.float64)
    unmodeled_topology_molecules = 0
    nucleosome_seed_topology_molecules = 0
    uninformative_molecules = 0
    uninformative_molecules_by_strand: Dict[str, int] = {}
    initially_unmodeled_topology: List[bool] = []
    for molecule_id in sorted(eligible_by_molecule):
        read = eligible_by_molecule[molecule_id]
        represented_opportunities = read.interval_evidence(
            envelope.start, envelope.end
        )[1]
        if represented_opportunities < minimum_molecule_opportunities:
            uninformative_molecules += 1
            uninformative_molecules_by_strand[read.strand] = (
                uninformative_molecules_by_strand.get(read.strand, 0) + 1
            )
            continue
        local_calls = [
            call
            for call in _calls_for_type(read, "tf")
            if call.start < envelope.end and envelope.start < call.end
        ]
        matched_calls = match_geometry_calls(
            local_calls, sites, center_radius=center_radius
        )
        signature = tuple(
            sorted(
                site_index
                for _call, site_index, assigned in matched_calls
                if assigned
            )
        )
        has_unassigned = bool(local_calls) and len(signature) != len(local_calls)
        configuration_index = configuration_by_signature.get(signature)
        has_nucleosome_seed_topology = any(
            nuc.start < envelope.end and envelope.start < nuc.end
            for nuc in _calls_for_type(read, "nuc")
        )
        if has_nucleosome_seed_topology:
            nucleosome_seed_topology_molecules += 1
        topology_is_unmodeled = bool(
            has_nucleosome_seed_topology
            or has_unassigned
            or (signature and configuration_index is None)
        )
        if topology_is_unmodeled:
            unmodeled_topology_molecules += 1
        retained_reads.append(read)
        initially_unmodeled_topology.append(topology_is_unmodeled)
        if has_nucleosome_seed_topology:
            seed_counts[unmodeled_component_index] += 1.0
        elif topology_is_unmodeled:
            seed_counts[spatial_null_component_index] += 1.0
        else:
            seed_counts[
                accessible_index
                if configuration_index is None
                else configuration_index
            ] += 1.0

    candidate_intervals: List[List[Tuple[int, int]]] = []
    candidate_score_matrices: List[np.ndarray] = []
    candidate_left_index_matrices: List[np.ndarray] = []
    candidate_right_index_matrices: List[np.ndarray] = []
    for site in sites:
        values = [
            (start, end)
            for start in range(
                site.start - boundary_search_radius,
                site.start + boundary_search_radius + 1,
            )
            for end in range(
                site.end - boundary_search_radius,
                site.end + boundary_search_radius + 1,
            )
            if end > start
        ]
        candidate_intervals.append(values)
        score_matrix, left_matrix, right_matrix = _interval_evidence_matrix(
            retained_reads, values, summation_mode="slice_sum"
        )
        candidate_score_matrices.append(score_matrix)
        candidate_left_index_matrices.append(left_matrix)
        candidate_right_index_matrices.append(right_matrix)

    anchored_candidate_intervals = {
        (int(start), int(end))
        for candidates in candidate_intervals
        for start, end in candidates
    }
    anchored_candidate_widths = sorted(
        {end - start for start, end in anchored_candidate_intervals}
    )
    if spatial_null_exclusion_intervals is None:
        spatial_null_excluded = anchored_candidate_intervals
        spatial_null_exclusion_explicit = False
    else:
        spatial_null_excluded = {
            (int(start), int(end))
            for start, end in spatial_null_exclusion_intervals
        }
        invalid_exclusions = [
            interval
            for interval in spatial_null_excluded
            if interval[1] <= interval[0]
            or interval[0] < envelope.start
            or interval[1] > envelope.end
        ]
        if invalid_exclusions:
            raise ValueError(
                "spatial-null exclusion intervals must be valid inside the analysis envelope"
            )
        if not anchored_candidate_intervals <= spatial_null_excluded:
            raise ValueError(
                "spatial-null exclusion universe must include every anchored candidate"
            )
        spatial_null_exclusion_explicit = True
    spatial_null_excluded_sorted = sorted(spatial_null_excluded)
    spatial_null_exclusion_digest = hashlib.sha256(
        np.asarray(spatial_null_excluded_sorted, dtype="<i8").tobytes()
    ).hexdigest()
    spatial_null_intervals = _spatial_null_candidate_intervals(
        envelope,
        [spatial_null_excluded_sorted],
        minimum_width=spatial_null_minimum_width,
        maximum_width=spatial_null_maximum_width,
    )
    spatial_null_candidate_widths = sorted(
        {end - start for start, end in spatial_null_intervals}
    )
    spatial_null_candidate_width_set = set(spatial_null_candidate_widths)
    anchored_widths_outside_spatial_null = [
        width
        for width in anchored_candidate_widths
        if width not in spatial_null_candidate_width_set
    ]
    spatial_null_localization_edges = [
        (
            -1,
            min(interval[0] for interval in values),
            max(interval[1] for interval in values),
        )
        for values in candidate_intervals
        for _coordinate_index in (0, 1)
    ]
    (
        spatial_null_log_likelihoods,
        spatial_null_edge_localization_probabilities,
    ) = _spatial_null_configuration_evidence(
        retained_reads,
        spatial_null_intervals,
        localization_edges=spatial_null_localization_edges,
    )
    unmodeled_log_likelihoods = np.asarray(
        [
            _diffuse_unmodeled_configuration_log_likelihood(read, envelope)
            for read in retained_reads
        ],
        dtype=np.float64,
    )

    # Compute opportunity-projection classes once with integer search indices.
    # Expected scores are evaluated once per class below, so exact lattice ties
    # cannot be broken by BLAS accumulation noise at high depth.
    candidate_projection_groups: List[List[List[int]]] = []
    for site_index, values in enumerate(candidate_intervals):
        groups_by_signature: Dict[tuple, List[int]] = {}
        for candidate_index, (start, end) in enumerate(values):
            signature = tuple(
                (
                    int(
                        candidate_left_index_matrices[site_index][
                            read_index, candidate_index
                        ]
                    ),
                    int(
                        candidate_right_index_matrices[site_index][
                            read_index, candidate_index
                        ]
                    ),
                )
                for read_index, _read in enumerate(retained_reads)
            )
            groups_by_signature.setdefault(signature, []).append(candidate_index)
        candidate_projection_groups.append(list(groups_by_signature.values()))

    configuration_membership = np.zeros(
        (component_count, len(sites)), dtype=np.float64
    )
    for configuration_index, configuration in enumerate(configuration_list):
        configuration_membership[
            configuration_index, list(configuration.site_indices)
        ] = 1.0
    selected_candidate_indices = []
    initial_geometry_targets: List[Tuple[int, int]] = []
    for site_index, (site, values) in enumerate(
        zip(sites, candidate_intervals)
    ):
        initial = (
            (site.start, site.end)
            if initial_geometry_intervals is None
            else tuple(int(value) for value in initial_geometry_intervals[site_index])
        )
        if initial not in values:
            raise ValueError(
                "initial geometry interval must belong to the fixed candidate grid"
            )
        initial_geometry_targets.append(initial)
        selected_candidate_indices.append(
            min(
                range(len(values)),
                key=lambda index: (
                    abs(values[index][0] - initial[0])
                    + abs(values[index][1] - initial[1]),
                    values[index],
                ),
            )
        )

    for configuration in configuration_list:
        ordered_indices = sorted(
            configuration.site_indices,
            key=lambda index: (
                candidate_intervals[index][selected_candidate_indices[index]][0],
                candidate_intervals[index][selected_candidate_indices[index]][1],
                index,
            ),
        )
        if any(
            candidate_intervals[left][selected_candidate_indices[left]][1]
            > candidate_intervals[right][selected_candidate_indices[right]][0]
            for left, right in zip(ordered_indices, ordered_indices[1:])
        ):
            raise ValueError(
                "initial geometry overlaps within a structured configuration"
            )

    component_pseudocounts = np.zeros(component_count, dtype=np.float64)
    component_pseudocounts[accessible_index] = float(pseudocount)
    tf_configuration_indices = [
        index
        for index, configuration in enumerate(configuration_list)
        if configuration.site_indices
    ]
    tf_configuration_indices_by_family_set: Dict[
        Tuple[str, ...], List[int]
    ] = {}
    for index in tf_configuration_indices:
        family_set = tuple(
            sorted(
                {
                    str(sites[site_index].family_id or sites[site_index].site_id)
                    for site_index in configuration_list[index].site_indices
                }
            )
        )
        tf_configuration_indices_by_family_set.setdefault(
            family_set, []
        ).append(index)
    if tf_configuration_indices:
        family_set_mass = float(pseudocount) / len(
            tf_configuration_indices_by_family_set
        )
        for indices in tf_configuration_indices_by_family_set.values():
            component_pseudocounts[indices] = family_set_mass / len(indices)
    component_pseudocounts[spatial_null_component_index] = float(pseudocount)
    component_pseudocounts[unmodeled_component_index] = float(pseudocount)
    if initialization_mode == "ordinary_calls":
        weights = seed_counts + component_pseudocounts
    else:
        weights = np.ones(component_count, dtype=np.float64)
    if float(np.sum(weights)) == 0.0:
        weights = np.ones(component_count, dtype=np.float64)
    weights /= np.sum(weights)
    objective_trace: List[float] = []
    converged = False
    convergence_reason = "maximum_iterations"
    final_weight_change: Optional[float] = None
    final_objective_change_per_molecule: Optional[float] = None
    responsibilities = np.empty(
        (len(retained_reads), component_count), dtype=np.float64
    )

    def selected_site_score_matrix() -> np.ndarray:
        if not retained_reads:
            return np.zeros((0, len(sites)), dtype=np.float64)
        return np.column_stack(
            [
                candidate_score_matrices[site_index][
                    :, selected_candidate_indices[site_index]
                ]
                for site_index in range(len(sites))
            ]
        )

    def candidate_preserves_topology(
        site_index: int, candidate: Tuple[int, int]
    ) -> bool:
        for configuration in configuration_list:
            if site_index not in configuration.site_indices:
                continue
            for other_index in configuration.site_indices:
                if other_index == site_index:
                    continue
                other = candidate_intervals[other_index][
                    selected_candidate_indices[other_index]
                ]
                if candidate[0] < other[1] and other[0] < candidate[1]:
                    return False
        return True

    def projection_canonicalized_scores(
        site_index: int, scores: np.ndarray
    ) -> np.ndarray:
        result = np.asarray(scores, dtype=np.float64).copy()
        for members in candidate_projection_groups[site_index]:
            representative_score = float(result[members[0]])
            result[np.asarray(members, dtype=np.int64)] = representative_score
        return result

    def score_is_tied(value: float, maximum: float) -> bool:
        return math.isclose(
            float(value),
            float(maximum),
            rel_tol=1e-12,
            abs_tol=1e-10,
        )

    def evaluate_current_parameters() -> Tuple[float, float, np.ndarray]:
        site_scores = selected_site_score_matrix()
        configuration_scores = site_scores @ configuration_membership.T
        configuration_scores[:, spatial_null_component_index] = (
            spatial_null_log_likelihoods
        )
        configuration_scores[:, unmodeled_component_index] = (
            unmodeled_log_likelihoods
        )
        log_joint = configuration_scores + np.log(
            np.maximum(weights, 1e-300)
        )[None, :]
        log_normalizer = _logsumexp(log_joint, axis=1)
        current_responsibilities = np.exp(
            log_joint - log_normalizer[:, None]
        )
        raw_log_likelihood = float(np.sum(log_normalizer))
        penalized_objective = float(
            raw_log_likelihood
            + np.sum(
                component_pseudocounts
                * np.log(np.maximum(weights, 1e-300))
            )
        )
        return raw_log_likelihood, penalized_objective, current_responsibilities

    if retained_reads:
        (
            _initial_raw_log_likelihood,
            initial_objective,
            responsibilities,
        ) = evaluate_current_parameters()
        objective_trace.append(initial_objective)
        stable_geometry_iterations = 0
        for iteration in range(1, max_iter + 1):
            previous_weights = weights.copy()
            weights = np.sum(responsibilities, axis=0) + component_pseudocounts
            weights /= len(retained_reads) + float(np.sum(component_pseudocounts))
            weights = np.maximum(weights, 1e-300)
            weights /= np.sum(weights)

            class_responsibilities = (
                responsibilities @ configuration_membership
            )
            previous_candidates = tuple(selected_candidate_indices)
            for site_index, values in enumerate(candidate_intervals):
                expected_scores = projection_canonicalized_scores(
                    site_index,
                    class_responsibilities[:, site_index]
                    @ candidate_score_matrices[site_index],
                )
                valid = [
                    index
                    for index, candidate in enumerate(values)
                    if candidate_preserves_topology(site_index, candidate)
                ]
                if not valid:
                    continue
                maximum = max(float(expected_scores[index]) for index in valid)
                tied = [
                    index
                    for index in valid
                    if score_is_tied(float(expected_scores[index]), maximum)
                ]
                tie_target = (
                    (sites[site_index].start, sites[site_index].end)
                    if tie_break_mode == "seed"
                    else initial_geometry_targets[site_index]
                )
                selected_candidate_indices[site_index] = min(
                    tied,
                    key=lambda index: (
                        abs(values[index][0] - tie_target[0])
                        + abs(values[index][1] - tie_target[1]),
                        values[index],
                    ),
                )

            (
                _post_raw_log_likelihood,
                post_update_objective,
                post_update_responsibilities,
            ) = evaluate_current_parameters()
            objective_tolerance = 1e-9 * max(1, len(retained_reads))
            if post_update_objective + objective_tolerance < objective_trace[-1]:
                raise ValueError(
                    "iterative TF geometry generalized-EM objective decreased"
                )
            objective_trace.append(post_update_objective)
            responsibilities = post_update_responsibilities

            maximum_weight_change = float(
                np.max(np.abs(weights - previous_weights))
            )
            final_weight_change = maximum_weight_change
            geometry_unchanged = previous_candidates == tuple(
                selected_candidate_indices
            )
            stable_geometry_iterations = (
                stable_geometry_iterations + 1 if geometry_unchanged else 0
            )
            objective_change_per_molecule = (
                abs(post_update_objective - objective_trace[-2])
                / len(retained_reads)
            )
            final_objective_change_per_molecule = (
                float(objective_change_per_molecule)
                if math.isfinite(objective_change_per_molecule)
                else None
            )
            if stable_geometry_iterations >= 2 and (
                maximum_weight_change < tol
                or objective_change_per_molecule < tol
            ):
                converged = True
                convergence_reason = (
                    "stable_geometry_and_weight_tolerance"
                    if maximum_weight_change < tol
                    else "stable_geometry_and_objective_tolerance_per_molecule"
                )
                break
    else:
        iteration = 0
        weights = component_pseudocounts.copy()
        if float(np.sum(weights)) == 0.0:
            weights = np.ones(component_count, dtype=np.float64)
        weights /= np.sum(weights)
        responsibilities = np.zeros(
            (0, component_count), dtype=np.float64
        )

    # Re-evaluate responsibilities and per-class expected boundary objectives
    # at the final parameter values so the reported equivalence sets correspond
    # to the returned model rather than the penultimate E-step.
    if retained_reads:
        (
            final_raw_log_likelihood,
            final_objective,
            responsibilities,
        ) = evaluate_current_parameters()
        class_responsibilities = responsibilities @ configuration_membership
    else:
        final_objective = 0.0
        final_raw_log_likelihood = 0.0
        class_responsibilities = np.zeros((0, len(sites)), dtype=np.float64)

    geometry = []
    for site_index, site in enumerate(sites):
        values = candidate_intervals[site_index]
        if retained_reads:
            expected_scores = projection_canonicalized_scores(
                site_index,
                class_responsibilities[:, site_index]
                @ candidate_score_matrices[site_index],
            )
            valid = [
                index
                for index, candidate in enumerate(values)
                if candidate_preserves_topology(site_index, candidate)
            ]
            if not valid:
                raise ValueError(
                    "no topology-preserving boundary candidate at fitted geometry"
                )
            maximum = max(float(expected_scores[index]) for index in valid)
            maximizing = [
                values[index]
                for index in valid
                if score_is_tied(float(expected_scores[index]), maximum)
            ]
            effective_support = float(
                np.sum(class_responsibilities[:, site_index])
            )
            candidate_score_delta = [
                {
                    "interval": [int(values[index][0]), int(values[index][1])],
                    "expected_log_likelihood_delta_from_best": float(
                        expected_scores[index] - maximum
                    ),
                }
                for index in valid
            ]
            candidate_score_delta_by_strand = {}
            strand_expected_scores: Dict[str, np.ndarray] = {}
            residual_aware_strand_expected_scores: Dict[str, np.ndarray] = {}
            for strand in sorted({read.strand for read in retained_reads}):
                strand_mask = np.asarray(
                    [read.strand == strand for read in retained_reads],
                    dtype=bool,
                )
                strand_scores = projection_canonicalized_scores(
                    site_index,
                    class_responsibilities[strand_mask, site_index]
                    @ candidate_score_matrices[site_index][strand_mask],
                )
                strand_expected_scores[strand] = strand_scores
                strand_maximum = max(
                    float(strand_scores[index]) for index in valid
                )
                candidate_score_delta_by_strand[strand] = [
                    {
                        "interval": [
                            int(values[index][0]),
                            int(values[index][1]),
                        ],
                        "expected_log_likelihood_delta_from_strand_best": (
                            float(strand_scores[index] - strand_maximum)
                        ),
                    }
                    for index in valid
                ]
        else:
            valid = list(range(len(values)))
            expected_scores = np.zeros(len(values), dtype=np.float64)
            maximum = 0.0
            maximizing = list(values)
            effective_support = 0.0
            candidate_score_delta = [
                {
                    "interval": [int(start), int(end)],
                    "expected_log_likelihood_delta_from_best": 0.0,
                }
                for start, end in values
            ]
            candidate_score_delta_by_strand = {}
            strand_expected_scores = {}
            residual_aware_strand_expected_scores = {}
        selected = values[selected_candidate_indices[site_index]]

        def coordinate_identifiability(coordinate_index: int) -> dict:
            coordinate_name = "start" if coordinate_index == 0 else "end"
            candidate_coordinates = sorted(
                {int(values[index][coordinate_index]) for index in valid}
            )

            def maximizing_coordinates(scores: np.ndarray) -> List[int]:
                maximum_score = max(float(scores[index]) for index in valid)
                return sorted(
                    {
                        int(values[index][coordinate_index])
                        for index in valid
                        if score_is_tied(float(scores[index]), maximum_score)
                    }
                )

            def coordinate_profile(scores: np.ndarray) -> Dict[int, float]:
                """Profile the nuisance edge out before measuring this edge.

                The full 2-D interval surface can vary strongly because of the
                *other* boundary.  Using its unprofiled range would therefore
                label a blind start as informative whenever only the end was
                resolved (and vice versa).  The profiled expected complete-data
                Q surface retains the best candidate for each value of the
                coordinate being tested.  It is not an observed-data
                likelihood-ratio or Bayes-factor calculation.
                """
                profile: Dict[int, float] = {}
                for index in valid:
                    coordinate = int(values[index][coordinate_index])
                    profile[coordinate] = max(
                        profile.get(coordinate, -math.inf),
                        float(scores[index]),
                    )
                return profile

            def coordinate_projection_groups(
                coordinates: Sequence[int],
                selected_reads: Sequence[ReadEvidence],
            ) -> List[List[int]]:
                groups: Dict[Tuple[int, ...], List[int]] = {}
                for coordinate in coordinates:
                    signature = tuple(
                        int(
                            np.searchsorted(
                                read.positions, coordinate, side="left"
                            )
                        )
                        for read in selected_reads
                    )
                    groups.setdefault(signature, []).append(int(coordinate))
                return sorted(
                    (sorted(group) for group in groups.values()),
                    key=lambda group: (group[0], group),
                )

            def profiled_projection_information(
                scores: np.ndarray,
                selected_reads: Sequence[ReadEvidence],
            ) -> Tuple[float, float, int]:
                """Return profile range and best-vs-runner-up lattice margin.

                A large best-to-worst range does not establish an edge when a
                second non-equivalent projection is nearly as good as the
                maximum.  Operational information is therefore the margin to
                the strongest alternative opportunity-projection class.
                """
                profile = coordinate_profile(scores)
                groups = coordinate_projection_groups(
                    sorted(profile), selected_reads
                )
                profile_range = float(
                    max(profile.values()) - min(profile.values())
                )
                class_scores = sorted(
                    (
                        max(profile[coordinate] for coordinate in group)
                        for group in groups
                    ),
                    reverse=True,
                )
                information_margin = (
                    float(class_scores[0] - class_scores[1])
                    if len(class_scores) >= 2
                    else 0.0
                )
                return profile_range, information_margin, len(groups)

            if retained_reads:
                localized_spatial_null_responsibilities = (
                    responsibilities[:, spatial_null_component_index]
                    * spatial_null_edge_localization_probabilities[
                        :, site_index * 2 + coordinate_index
                    ]
                )
                residual_aware_strand_expected_scores = {}
                for strand in strand_expected_scores:
                    strand_mask = np.asarray(
                        [read.strand == strand for read in retained_reads],
                        dtype=bool,
                    )
                    residual_aware_weights = (
                        class_responsibilities[strand_mask, site_index]
                        + localized_spatial_null_responsibilities[strand_mask]
                    )
                    residual_aware_strand_expected_scores[strand] = (
                        projection_canonicalized_scores(
                            site_index,
                            residual_aware_weights
                            @ candidate_score_matrices[site_index][strand_mask],
                        )
                    )
                pooled_coordinates = maximizing_coordinates(expected_scores)
                by_strand_coordinates = {
                    strand: maximizing_coordinates(scores)
                    for strand, scores in strand_expected_scores.items()
                }
                pooled_projection_groups = coordinate_projection_groups(
                    pooled_coordinates, retained_reads
                )
                projection_groups_by_strand = {
                    strand: coordinate_projection_groups(
                        coordinates,
                        [read for read in retained_reads if read.strand == strand],
                    )
                    for strand, coordinates in by_strand_coordinates.items()
                }
                selected_coordinate = int(selected[coordinate_index])
                edge_window_start = min(candidate_coordinates)
                # Half-open boundaries at min..max can only differ on bases
                # min..max-1.  A base at ``max`` is on the same side of every
                # candidate and is not an edge-discriminating opportunity.
                edge_window_end = max(candidate_coordinates)
                (
                    pooled_profile_range,
                    pooled_information_spread,
                    pooled_candidate_projection_class_count,
                ) = profiled_projection_information(
                    expected_scores, retained_reads
                )
                pooled_effective_edge_opportunities = float(
                    sum(
                        class_responsibilities[read_index, site_index]
                        * read.interval_evidence(
                            edge_window_start, edge_window_end
                        )[1]
                        for read_index, read in enumerate(retained_reads)
                    )
                )
                pooled_information_margin_per_effective_molecule = (
                    float(pooled_information_spread / effective_support)
                    if effective_support > 0.0
                    else 0.0
                )
                information_spread_by_strand = {}
                information_margin_per_effective_molecule_by_strand = {}
                profile_range_by_strand = {}
                candidate_projection_class_count_by_strand = {}
                effective_edge_opportunities_by_strand = {}
                effective_molecule_support_by_strand = {}
                raw_edge_opportunities_by_strand = {}
                residual_aware_edge_opportunities_by_strand = {}
                residual_aware_information_margin_by_strand = {}
                residual_aware_information_margin_per_effective_molecule_by_strand = {}
                residual_aware_effective_molecule_support_by_strand = {}
                spatial_null_support_fraction_by_strand = {}
                residual_aware_maximizing_coordinates_by_strand = {}
                residual_aware_maximizing_projection_classes_by_strand = {}
                spatial_null_support_by_strand = {}
                total_spatial_null_support_by_strand = {}
                for strand, scores in strand_expected_scores.items():
                    strand_reads = [
                        read for read in retained_reads if read.strand == strand
                    ]
                    (
                        profile_range_by_strand[strand],
                        information_spread_by_strand[strand],
                        candidate_projection_class_count_by_strand[strand],
                    ) = profiled_projection_information(
                        scores, strand_reads
                    )
                    effective_molecule_support_by_strand[strand] = float(
                        sum(
                            class_responsibilities[read_index, site_index]
                            for read_index, read in enumerate(retained_reads)
                            if read.strand == strand
                        )
                    )
                    information_margin_per_effective_molecule_by_strand[
                        strand
                    ] = (
                        information_spread_by_strand[strand]
                        / effective_molecule_support_by_strand[strand]
                        if effective_molecule_support_by_strand[strand] > 0.0
                        else 0.0
                    )
                    effective_edge_opportunities_by_strand[strand] = float(
                        sum(
                            class_responsibilities[read_index, site_index]
                            * read.interval_evidence(
                                edge_window_start, edge_window_end
                            )[1]
                            for read_index, read in enumerate(retained_reads)
                            if read.strand == strand
                        )
                    )
                    raw_edge_opportunities_by_strand[strand] = int(
                        sum(
                            read.interval_evidence(
                                edge_window_start, edge_window_end
                            )[1]
                            for read in retained_reads
                            if read.strand == strand
                        )
                    )
                    residual_scores = residual_aware_strand_expected_scores[
                        strand
                    ]
                    residual_coordinates = maximizing_coordinates(
                        residual_scores
                    )
                    residual_aware_maximizing_coordinates_by_strand[
                        strand
                    ] = residual_coordinates
                    residual_groups = coordinate_projection_groups(
                        residual_coordinates, strand_reads
                    )
                    residual_aware_maximizing_projection_classes_by_strand[
                        strand
                    ] = residual_groups
                    (
                        _residual_profile_range,
                        residual_aware_information_margin_by_strand[strand],
                        _residual_candidate_projection_classes,
                    ) = profiled_projection_information(
                        residual_scores, strand_reads
                    )
                    residual_aware_edge_opportunities_by_strand[strand] = float(
                        sum(
                            (
                                class_responsibilities[read_index, site_index]
                                + localized_spatial_null_responsibilities[
                                    read_index
                                ]
                            )
                            * read.interval_evidence(
                                edge_window_start, edge_window_end
                            )[1]
                            for read_index, read in enumerate(retained_reads)
                            if read.strand == strand
                        )
                    )
                    residual_aware_effective_molecule_support_by_strand[
                        strand
                    ] = float(
                        sum(
                            class_responsibilities[read_index, site_index]
                            + localized_spatial_null_responsibilities[read_index]
                            for read_index, read in enumerate(retained_reads)
                            if read.strand == strand
                        )
                    )
                    residual_aware_information_margin_per_effective_molecule_by_strand[
                        strand
                    ] = (
                        residual_aware_information_margin_by_strand[strand]
                        / residual_aware_effective_molecule_support_by_strand[
                            strand
                        ]
                        if residual_aware_effective_molecule_support_by_strand[
                            strand
                        ]
                        > 0.0
                        else 0.0
                    )
                    spatial_null_support_by_strand[strand] = float(
                        sum(
                            localized_spatial_null_responsibilities[read_index]
                            for read_index, read in enumerate(retained_reads)
                            if read.strand == strand
                        )
                    )
                    total_spatial_null_support_by_strand[strand] = float(
                        sum(
                            responsibilities[
                                read_index, spatial_null_component_index
                            ]
                            for read_index, read in enumerate(retained_reads)
                            if read.strand == strand
                        )
                    )
                    spatial_null_support_fraction_by_strand[strand] = (
                        spatial_null_support_by_strand[strand]
                        / residual_aware_effective_molecule_support_by_strand[
                            strand
                        ]
                        if residual_aware_effective_molecule_support_by_strand[
                            strand
                        ]
                        > 0.0
                        else 0.0
                    )
                informative = [
                    strand
                    for strand in by_strand_coordinates
                    if effective_edge_opportunities_by_strand[strand]
                    >= minimum_edge_effective_opportunities
                    and information_margin_per_effective_molecule_by_strand[
                        strand
                    ]
                    >= minimum_edge_q_margin_per_effective_molecule
                    and information_spread_by_strand[strand]
                    >= minimum_edge_information_spread
                ]
                blind = [
                    strand
                    for strand in by_strand_coordinates
                    if strand not in informative
                ]
                ambiguous = [
                    strand
                    for strand, coordinates in by_strand_coordinates.items()
                    if len(coordinates) > 1
                ]
                uniquely_resolving = [
                    strand
                    for strand, coordinates in by_strand_coordinates.items()
                    if strand in informative
                    and coordinates == [selected_coordinate]
                ]
                informative_sets = [
                    set(by_strand_coordinates[strand]) for strand in informative
                ]
                informative_intersection = (
                    sorted(set.intersection(*informative_sets))
                    if informative_sets
                    else []
                )
                conflicting = (
                    len(informative_sets) >= 2 and not informative_intersection
                )
                residual_absorbed_strata = [
                    strand
                    for strand in by_strand_coordinates
                    if strand not in informative
                    and raw_edge_opportunities_by_strand[strand]
                    >= minimum_edge_effective_opportunities
                    and residual_aware_edge_opportunities_by_strand[strand]
                    >= minimum_edge_effective_opportunities
                    and residual_aware_information_margin_per_effective_molecule_by_strand[
                        strand
                    ]
                    >= minimum_edge_q_margin_per_effective_molecule
                    and residual_aware_information_margin_by_strand[strand]
                    >= minimum_edge_information_spread
                    and spatial_null_support_fraction_by_strand[strand]
                    >= minimum_spatial_null_conflict_fraction
                    and len(
                        residual_aware_maximizing_projection_classes_by_strand[
                            strand
                        ]
                    )
                    == 1
                    and not (
                        set(
                            residual_aware_maximizing_coordinates_by_strand[
                                strand
                            ]
                        )
                        & set(pooled_coordinates)
                    )
                ]
                pooled_information_qualified = bool(
                    pooled_effective_edge_opportunities
                    >= minimum_edge_effective_opportunities
                    and pooled_information_margin_per_effective_molecule
                    >= minimum_edge_q_margin_per_effective_molecule
                    and pooled_information_spread
                    >= minimum_edge_information_spread
                )
                if conflicting:
                    status = "conflicting_strata"
                elif residual_absorbed_strata:
                    status = "stratum_absorbed_by_residual_component"
                elif (
                    not informative
                    and pooled_information_qualified
                    and len(pooled_projection_groups) == 1
                ):
                    status = {
                        "physical_complementary": (
                            "resolved_by_pooled_complementary_evidence"
                        ),
                        "diagnostic_partition": (
                            "resolved_by_pooled_diagnostic_evidence"
                        ),
                        "unspecified": "resolved_by_pooled_evidence",
                    }[stratum_semantics]
                elif not informative:
                    status = "undercovered"
                elif (
                    len(informative) == 1
                    and informative[0] in uniquely_resolving
                ):
                    status = "resolved_informative_stratum"
                elif (
                    len(informative) >= 2
                    and all(strand in uniquely_resolving for strand in informative)
                ):
                    status = "resolved_both_strata"
                elif (
                    len(informative) >= 2
                    and all(strand in ambiguous for strand in informative)
                    and informative_intersection == [selected_coordinate]
                ):
                    # Each physical strand is ambiguous on its own, but the
                    # intersection of their opportunity-lattice equivalence
                    # sets contains one coordinate.  This is the central
                    # reason to pool complementary DAF strata.
                    status = {
                        "physical_complementary": (
                            "resolved_jointly_complementary_strata"
                        ),
                        "diagnostic_partition": (
                            "resolved_jointly_across_diagnostic_strata"
                        ),
                        "unspecified": "resolved_jointly_across_strata",
                    }[stratum_semantics]
                elif (
                    uniquely_resolving
                    and informative_intersection == [selected_coordinate]
                ):
                    status = (
                        "resolved_by_single_stratum_consistent_with_others"
                    )
                elif len(pooled_projection_groups) > 1:
                    status = "model_ambiguous"
                else:
                    status = "identified_up_to_opportunity_projection"
            else:
                pooled_coordinates = sorted(
                    {int(value[coordinate_index]) for value in values}
                )
                by_strand_coordinates = {}
                pooled_projection_groups = coordinate_projection_groups(
                    pooled_coordinates, []
                )
                projection_groups_by_strand = {}
                informative = []
                blind = []
                ambiguous = []
                uniquely_resolving = []
                information_spread_by_strand = {}
                profile_range_by_strand = {}
                candidate_projection_class_count_by_strand = {}
                effective_edge_opportunities_by_strand = {}
                raw_edge_opportunities_by_strand = {}
                residual_aware_edge_opportunities_by_strand = {}
                residual_aware_information_margin_by_strand = {}
                residual_aware_maximizing_coordinates_by_strand = {}
                residual_aware_maximizing_projection_classes_by_strand = {}
                spatial_null_support_by_strand = {}
                total_spatial_null_support_by_strand = {}
                spatial_null_support_fraction_by_strand = {}
                residual_absorbed_strata = []
                pooled_profile_range = 0.0
                pooled_information_spread = 0.0
                pooled_candidate_projection_class_count = 0
                pooled_effective_edge_opportunities = 0.0
                pooled_information_margin_per_effective_molecule = 0.0
                pooled_information_qualified = False
                information_margin_per_effective_molecule_by_strand = {}
                effective_molecule_support_by_strand = {}
                residual_aware_information_margin_per_effective_molecule_by_strand = {}
                residual_aware_effective_molecule_support_by_strand = {}
                informative_intersection = []
                status = "undercovered"
            return {
                "edge": coordinate_name,
                "status": status,
                "selected_coordinate": int(selected[coordinate_index]),
                "candidate_coordinate_range": [
                    min(candidate_coordinates),
                    max(candidate_coordinates),
                ],
                "selected_at_search_boundary": bool(
                    int(selected[coordinate_index])
                    in {min(candidate_coordinates), max(candidate_coordinates)}
                ),
                "pooled_maximizing_coordinates": pooled_coordinates,
                "maximizing_coordinates_by_strand": by_strand_coordinates,
                "pooled_maximizing_opportunity_projection_classes": (
                    pooled_projection_groups
                ),
                "pooled_maximizing_opportunity_projection_class_count": len(
                    pooled_projection_groups
                ),
                "maximizing_opportunity_projection_classes_by_strand": (
                    projection_groups_by_strand
                ),
                "informative_strands": informative,
                "blind_strands": blind,
                "ambiguous_strands": ambiguous,
                "uniquely_resolving_strands": uniquely_resolving,
                "informative_stratum_coordinate_intersection": (
                    informative_intersection
                ),
                "effective_edge_opportunities_by_strand": (
                    effective_edge_opportunities_by_strand
                ),
                "raw_edge_opportunities_by_strand": (
                    raw_edge_opportunities_by_strand
                ),
                "residual_aware_edge_opportunities_by_strand": (
                    residual_aware_edge_opportunities_by_strand
                ),
                "residual_aware_information_margin_nats_by_strand": (
                    residual_aware_information_margin_by_strand
                ),
                "residual_aware_information_margin_per_effective_molecule_nats_by_strand": (
                    residual_aware_information_margin_per_effective_molecule_by_strand
                ),
                "residual_aware_effective_molecule_support_by_strand": (
                    residual_aware_effective_molecule_support_by_strand
                ),
                "residual_aware_maximizing_coordinates_by_strand": (
                    residual_aware_maximizing_coordinates_by_strand
                ),
                "residual_aware_maximizing_opportunity_projection_classes_by_strand": (
                    residual_aware_maximizing_projection_classes_by_strand
                ),
                "spatial_null_effective_molecule_support_by_strand": (
                    spatial_null_support_by_strand
                ),
                "spatial_null_total_effective_molecule_support_by_strand": (
                    total_spatial_null_support_by_strand
                ),
                "spatial_null_edge_localization": {
                    "coordinate": coordinate_name,
                    "candidate_coordinate_range": [
                        min(candidate_coordinates),
                        max(candidate_coordinates),
                    ],
                    "placement_posterior_condition": (
                        "P0_interval_center_inside_site_candidate_envelope"
                    ),
                },
                "spatial_null_fraction_of_residual_aware_support_by_strand": (
                    spatial_null_support_fraction_by_strand
                ),
                "residual_absorbed_strands": residual_absorbed_strata,
                "information_spread_nats_by_strand": (
                    information_spread_by_strand
                ),
                "information_margin_to_best_non_equivalent_nats_by_strand": (
                    information_spread_by_strand
                ),
                "information_margin_per_effective_molecule_nats_by_strand": (
                    information_margin_per_effective_molecule_by_strand
                ),
                "effective_molecule_support_by_strand": (
                    effective_molecule_support_by_strand
                ),
                "profile_range_nats_by_strand": profile_range_by_strand,
                "candidate_opportunity_projection_class_count_by_strand": (
                    candidate_projection_class_count_by_strand
                ),
                "pooled_effective_edge_opportunities": float(
                    pooled_effective_edge_opportunities
                ),
                "pooled_profile_range_nats": float(pooled_profile_range),
                "pooled_information_spread_nats": float(
                    pooled_information_spread
                ),
                "pooled_information_margin_to_best_non_equivalent_nats": float(
                    pooled_information_spread
                ),
                "pooled_information_margin_per_effective_molecule_nats": float(
                    pooled_information_margin_per_effective_molecule
                ),
                "information_basis": (
                    "profiled_expected_complete_data_q_margin_to_best_"
                    "non_equivalent_edge_opportunity_projection;_qualification_"
                    "uses_margin_per_effective_site_molecule"
                ),
                "pooled_candidate_opportunity_projection_class_count": int(
                    pooled_candidate_projection_class_count
                ),
                "pooled_information_qualified": pooled_information_qualified,
                "operationally_identified": bool(
                    pooled_information_qualified
                    and len(pooled_projection_groups) == 1
                    and status
                    not in {
                        "conflicting_strata",
                        "model_ambiguous",
                        "stratum_absorbed_by_residual_component",
                        "undercovered",
                    }
                ),
                "minimum_effective_edge_opportunities": float(
                    minimum_edge_effective_opportunities
                ),
                "minimum_information_spread_nats": float(
                    minimum_edge_information_spread
                ),
                "minimum_information_margin_to_best_non_equivalent_nats": float(
                    minimum_edge_information_spread
                ),
                "minimum_information_margin_per_effective_molecule_nats": float(
                    minimum_edge_q_margin_per_effective_molecule
                ),
                "minimum_spatial_null_conflict_fraction": float(
                    minimum_spatial_null_conflict_fraction
                ),
            }

        edge_identifiability = {
            "start": coordinate_identifiability(0),
            "end": coordinate_identifiability(1),
        }
        selected_projection_members = next(
            members
            for members in candidate_projection_groups[site_index]
            if selected_candidate_indices[site_index] in members
        )
        tolerance_sets = {
            label: [
                [int(values[index][0]), int(values[index][1])]
                for index in valid
                if float(expected_scores[index])
                >= maximum - delta * effective_support
            ]
            for label, delta in (
                ("exact", 0.0),
                (
                    "within_log_10_per_effective_molecule",
                    math.log(10.0),
                ),
                (
                    "within_log_100_per_effective_molecule",
                    math.log(100.0),
                ),
            )
        }
        geometry.append(
            {
                "site_id": site.site_id,
                "family_id": site.family_id or site.site_id,
                "substate_id": site.substate_id or site.site_id,
                "seed_provenance_stratum": site.seed_provenance_stratum,
                "seed_interval": [int(site.start), int(site.end)],
                "selected_interval": [int(selected[0]), int(selected[1])],
                "maximizing_interval_count": len(maximizing),
                "maximizing_intervals": [
                    [int(start), int(end)] for start, end in maximizing
                ],
                "selected_opportunity_projection_equivalent_intervals": [
                    [
                        int(values[index][0]),
                        int(values[index][1]),
                    ]
                    for index in selected_projection_members
                    if index in valid
                ],
                "candidate_opportunity_projection_class_count": len(
                    candidate_projection_groups[site_index]
                ),
                "evidence_tolerance_interval_sets": tolerance_sets,
                "evidence_tolerance_basis": (
                    "profiled_expected_complete_data_q_delta_per_effective_"
                    "molecule"
                ),
                "start_equivalence_range": [
                    min(start for start, _end in maximizing),
                    max(start for start, _end in maximizing),
                ],
                "end_equivalence_range": [
                    min(end for _start, end in maximizing),
                    max(end for _start, end in maximizing),
                ],
                "boundary_status": (
                    "identified_on_candidate_grid"
                    if len(maximizing) == 1
                    else "identified_up_to_opportunity_projection"
                    if retained_reads
                    else "unidentified_no_eligible_molecules"
                ),
                "edge_identifiability": edge_identifiability,
                "effective_molecule_support": effective_support,
                "site_opportunity_bearing_molecules": int(
                    sum(
                        read.interval_evidence(site.start, site.end)[1] > 0
                        for read in eligible_by_molecule.values()
                    )
                ),
                "site_total_opportunities": int(
                    sum(
                        read.interval_evidence(site.start, site.end)[1]
                        for read in eligible_by_molecule.values()
                    )
                ),
                "site_opportunity_bearing_molecules_by_strand": {
                    strand: int(
                        sum(
                            read.strand == strand
                            and read.interval_evidence(site.start, site.end)[1] > 0
                            for read in eligible_by_molecule.values()
                        )
                    )
                    for strand in sorted(
                        {read.strand for read in eligible_by_molecule.values()}
                    )
                },
                "candidate_expected_log_likelihood_surface": (
                    candidate_score_delta
                ),
                "candidate_expected_complete_data_q_surface": (
                    candidate_score_delta
                ),
                "candidate_expected_log_likelihood_surface_by_strand": (
                    candidate_score_delta_by_strand
                ),
                "candidate_expected_complete_data_q_surface_by_strand": (
                    candidate_score_delta_by_strand
                ),
            }
        )

    # A mixture cannot identify separate weights for configurations that mark
    # exactly the same represented opportunities protected on every molecule.
    # Canonicalize that quotient explicitly without changing any molecule's
    # raw call or choosing between reference coordinates inside a lattice gap.
    selected_intervals = [
        candidate_intervals[index][selected_candidate_indices[index]]
        for index in range(len(sites))
    ]
    projection_groups: Dict[tuple, List[int]] = {}
    projection_hashes: Dict[int, str] = {}
    for configuration_index, configuration in enumerate(configuration_list):
        projection_parts = []
        digest = hashlib.sha256()
        for read in retained_reads:
            local_positions = read.positions[
                int(
                    np.searchsorted(
                        read.positions, envelope.start, side="left"
                    )
                ) : int(
                    np.searchsorted(read.positions, envelope.end, side="left")
                )
            ]
            protected = np.zeros(len(local_positions), dtype=np.uint8)
            for site_index in configuration.site_indices:
                start, end = selected_intervals[site_index]
                left = int(np.searchsorted(local_positions, start, side="left"))
                right = int(np.searchsorted(local_positions, end, side="left"))
                protected[left:right] = 1
            packed = np.packbits(protected, bitorder="little").tobytes()
            part = (len(protected), packed)
            projection_parts.append(part)
            digest.update(int(len(protected)).to_bytes(8, "little"))
            digest.update(int(len(packed)).to_bytes(8, "little"))
            digest.update(packed)
        signature = tuple(projection_parts)
        projection_groups.setdefault(signature, []).append(configuration_index)
        projection_hashes[configuration_index] = digest.hexdigest()

    configuration_equivalence_classes = []
    configuration_equivalence_ids: List[Optional[str]] = [
        None for _configuration in configuration_list
    ]
    for member_indices in projection_groups.values():
        class_payload = "\x1f".join(
            configuration_list[index].name for index in member_indices
        )
        equivalence_id = "tfprojection_" + hashlib.sha256(
            class_payload.encode("utf-8")
        ).hexdigest()[:16]
        for index in member_indices:
            configuration_equivalence_ids[index] = equivalence_id
        configuration_equivalence_classes.append(
            {
                "equivalence_class_id": equivalence_id,
                "configuration_indices": [
                    int(index) for index in member_indices
                ],
                "configuration_names": [
                    configuration_list[index].name for index in member_indices
                ],
                "opportunity_projection_sha256": projection_hashes[
                    member_indices[0]
                ],
                "probability_sum": float(
                    np.sum(weights[np.asarray(member_indices, dtype=np.int64)])
                ),
                "individual_configuration_weights_identifiable": (
                    len(member_indices) == 1
                ),
                "interpretation": (
                    "exact_same_protected_opportunity_assignment_on_all_"
                    "modeled_molecules"
                ),
            }
        )

    final_site_scores = selected_site_score_matrix()
    final_component_likelihoods = (
        final_site_scores @ configuration_membership.T
    )
    final_component_likelihoods[:, spatial_null_component_index] = (
        spatial_null_log_likelihoods
    )
    final_component_likelihoods[:, unmodeled_component_index] = (
        unmodeled_log_likelihoods
    )
    component_names = [
        *(configuration.name for configuration in configuration_list),
        "P0:unanchored_single_interval",
        "U:diffuse_iid_opportunity_protection",
    ]
    component_groups = [
        list(member_indices) for member_indices in projection_groups.values()
    ]
    for nuisance_index in (
        spatial_null_component_index,
        unmodeled_component_index,
    ):
        matched_group = next(
            (
                members
                for members in component_groups
                if np.allclose(
                    final_component_likelihoods[:, nuisance_index],
                    final_component_likelihoods[:, members[0]],
                    rtol=1e-14,
                    atol=1e-12,
                )
            ),
            None,
        )
        if matched_group is None:
            component_groups.append([nuisance_index])
        else:
            matched_group.append(nuisance_index)
    component_likelihood_equivalence_class_ids: List[Optional[str]] = [
        None for _name in component_names
    ]
    component_likelihood_equivalence_classes = []
    for members in component_groups:
        payload = "\x1f".join(sorted(component_names[index] for index in members))
        representative = np.asarray(
            final_component_likelihoods[:, members[0]], dtype="<f8"
        )
        likelihood_sha256 = hashlib.sha256(representative.tobytes()).hexdigest()
        equivalence_id = "tfemission_" + hashlib.sha256(
            f"{payload}\x1f{likelihood_sha256}".encode("utf-8")
        ).hexdigest()[:16]
        for index in members:
            component_likelihood_equivalence_class_ids[index] = equivalence_id
        component_likelihood_equivalence_classes.append(
            {
                "equivalence_class_id": equivalence_id,
                "component_indices": [int(index) for index in members],
                "component_names": [component_names[index] for index in members],
                "per_molecule_log_likelihood_sha256": likelihood_sha256,
                "probability_sum": float(
                    np.sum(weights[np.asarray(members, dtype=np.int64)])
                ),
                "individual_component_weights_identifiable": len(members) == 1,
            }
        )

    family_site_indices: Dict[str, List[int]] = {}
    for site_index, site in enumerate(sites):
        family_site_indices.setdefault(
            str(site.family_id or site.site_id), []
        ).append(site_index)
    tf_families = []

    def component_subset_probability_identifiable(
        component_indices: Sequence[int],
    ) -> bool:
        selected = set(component_indices)
        for record in component_likelihood_equivalence_classes:
            members = set(record["component_indices"])
            if members & selected and not members <= selected:
                return False
        return True

    for family_id, member_site_indices in sorted(family_site_indices.items()):
        member_set = set(member_site_indices)
        family_configuration_indices = [
            index
            for index, configuration in enumerate(configuration_list)
            if member_set & set(configuration.site_indices)
        ]
        substate_records = []
        for site_index in member_site_indices:
            site = sites[site_index]
            substate_configuration_indices = [
                index
                for index, configuration in enumerate(configuration_list)
                if site_index in configuration.site_indices
            ]
            substate_records.append(
                {
                    "site_id": site.site_id,
                    "substate_id": str(site.substate_id or site.site_id),
                    "seed_provenance_stratum": site.seed_provenance_stratum,
                    "seed_interval": [int(site.start), int(site.end)],
                    "selected_interval": geometry[site_index][
                        "selected_interval"
                    ],
                    "configuration_indices": substate_configuration_indices,
                    "configuration_names": [
                        configuration_list[index].name
                        for index in substate_configuration_indices
                    ],
                    "marginal_probability": float(
                        np.sum(
                            weights[
                                np.asarray(
                                    substate_configuration_indices,
                                    dtype=np.int64,
                                )
                            ]
                        )
                    ),
                    "marginal_probability_identifiable": (
                        component_subset_probability_identifiable(
                            substate_configuration_indices
                        )
                    ),
                    "component_likelihood_equivalence_class_ids": sorted(
                        {
                            str(
                                component_likelihood_equivalence_class_ids[
                                    index
                                ]
                            )
                            for index in substate_configuration_indices
                        }
                    ),
                }
            )
        tf_families.append(
            {
                "family_id": family_id,
                "site_indices": member_site_indices,
                "site_ids": [sites[index].site_id for index in member_site_indices],
                "configuration_indices": family_configuration_indices,
                "configuration_names": [
                    configuration_list[index].name
                    for index in family_configuration_indices
                ],
                "marginal_probability": float(
                    np.sum(
                        weights[
                            np.asarray(
                                family_configuration_indices, dtype=np.int64
                            )
                        ]
                    )
                ),
                "marginal_probability_identifiable": (
                    component_subset_probability_identifiable(
                        family_configuration_indices
                    )
                ),
                "substates": substate_records,
                "interpretation": (
                    "localized_family_probability_and_geometry_substates;_"
                    "substate_assignments_are_soft_and_source_molecule_calls_"
                    "remain_unchanged"
                ),
            }
        )

    structure_payload = "\x1f".join(
        [
            f"radius={int(boundary_search_radius)}",
            f"spatial_padding={int(spatial_null_padding)}",
            f"spatial_widths={int(spatial_null_minimum_width)}-{int(spatial_null_maximum_width)}",
            f"minimum_spatial_null_conflict_fraction={float(minimum_spatial_null_conflict_fraction):.17g}",
            f"analysis_envelope={int(envelope.start)}-{int(envelope.end)}",
            f"stratum_semantics={stratum_semantics}",
            f"spatial_exclusion_sha256={spatial_null_exclusion_digest}",
            "spatial_null=denested_uniform_width_then_reference_start",
            "spatial_null_edge_conflict=placement_posterior_boundary_localized",
            "unmodeled=beta_1_1_iid_opportunity_protection",
            "unmodeled_quadrature=gauss_legendre_degree_exact",
            "evidence_backend=numpy_float64_slice_candidate_prefix_spatial_v2",
            "weight_prior=hierarchical_A_TF_P0_U_family_set_then_substate",
            *(
                (
                    f"{site.site_id}:{site.start}-{site.end}:"
                    f"family={site.family_id or site.site_id}:"
                    f"substate={site.substate_id or site.site_id}:"
                    f"seed_stratum={site.seed_provenance_stratum or ''}"
                )
                for site in sorted(
                    sites, key=lambda value: (value.start, value.end, value.site_id)
                )
            ),
            *(
                "configuration="
                + ",".join(
                    sorted(sites[index].site_id for index in configuration.site_indices)
                )
                for configuration in sorted(
                    configuration_list,
                    key=lambda value: tuple(
                        sorted(sites[index].site_id for index in value.site_indices)
                    ),
                )
            ),
        ]
    )
    model_structure_id = "tfgeometrystructure_" + hashlib.sha256(
        structure_payload.encode("utf-8")
    ).hexdigest()[:16]
    training_digest = hashlib.sha256()
    for read in retained_reads:
        molecule_payload = "\x1f".join(read.molecule_id).encode("utf-8")
        training_digest.update(len(molecule_payload).to_bytes(8, "little"))
        training_digest.update(molecule_payload)
        training_digest.update(
            bytes.fromhex(_read_evidence_content_sha256(read))
        )
    training_data_sha256 = training_digest.hexdigest()
    fit_payload = json.dumps(
        {
            "structure_id": model_structure_id,
            "training_data_sha256": training_data_sha256,
            "pseudocount": float(pseudocount),
            "center_radius": int(center_radius),
            "max_iter": int(max_iter),
            "tol": float(tol),
            "minimum_molecule_opportunities": int(minimum_molecule_opportunities),
            "minimum_edge_effective_opportunities": float(
                minimum_edge_effective_opportunities
            ),
            "minimum_edge_information_spread": float(
                minimum_edge_information_spread
            ),
            "minimum_edge_q_margin_per_effective_molecule": float(
                minimum_edge_q_margin_per_effective_molecule
            ),
            "minimum_spatial_null_conflict_fraction": float(
                minimum_spatial_null_conflict_fraction
            ),
            "stratum_semantics": stratum_semantics,
            "initialization_mode": initialization_mode,
            "initial_geometry": initial_geometry_targets,
            "tie_break_mode": tie_break_mode,
            "selected_intervals": selected_intervals,
            "weights": [float(value) for value in weights],
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    model_id = "tfgeometrymodel_" + hashlib.sha256(
        fit_payload.encode("utf-8")
    ).hexdigest()[:16]
    return {
        "schema": "fiberhmm.iterative_tf_class_geometry_model.v2",
        "model_id": model_id,
        "model_structure_id": model_structure_id,
        "training_data_sha256": training_data_sha256,
        "status": "experimental_report_only",
        "inference_enabled": False,
        "assignment_unit": "collapsed_molecule_configuration",
        "model_role": "locus_catalog_and_per_molecule_soft_decoder",
        "raw_molecule_signal_policy": "immutable_source_evidence_and_calls",
        "evidence_backend": "numpy_float64_slice_candidate_prefix_spatial_v2",
        "evidence_backend_precision": "float64",
        "evidence_backend_deterministic_reductions": True,
        "model_projection_policy": (
            "separate_optional_layer_never_replaces_source_calls"
        ),
        "single_molecule_variation_policy": (
            "retain_supported_geometry_or_configuration_variation"
        ),
        "tf_family_substate_policy": (
            "localized_family_sums_soft_geometry_substates_without_rewriting_"
            "source_molecule_calls"
        ),
        "strand_treatment": "pooled_with_molecule_specific_opportunity_lattice",
        "stratum_semantics": stratum_semantics,
        "ordinary_call_role": "candidate_seed_and_weight_initialization_only",
        "configuration_assignment_policy": (
            "fully_soft_raw_chemistry_with_spatial_and_diffuse_null_components"
        ),
        "rescued_call_training": False,
        "boundary_search_radius": int(boundary_search_radius),
        "initialization_mode": initialization_mode,
        "tie_break_mode": tie_break_mode,
        "initial_geometry_intervals": [
            [int(start), int(end)]
            for start, end in (
                initial_geometry_intervals
                if initial_geometry_intervals is not None
                else [(site.start, site.end) for site in sites]
            )
        ],
        "envelope": [int(envelope.start), int(envelope.end)],
        "analysis_envelope_explicit": analysis_envelope_explicit,
        "spatial_null_exclusion_explicit": spatial_null_exclusion_explicit,
        "spatial_null_exclusion_intervals": [
            [int(start), int(end)]
            for start, end in spatial_null_excluded_sorted
        ],
        "spatial_null_exclusion_sha256": spatial_null_exclusion_digest,
        "geometry_envelope": [
            int(geometry_envelope.start),
            int(geometry_envelope.end),
        ],
        "spatial_null_padding": int(spatial_null_padding),
        "spatial_null_minimum_width": int(spatial_null_minimum_width),
        "spatial_null_maximum_width": int(spatial_null_maximum_width),
        "spatial_null_requested_width_range": [
            int(spatial_null_minimum_width),
            int(spatial_null_maximum_width),
        ],
        "spatial_null_effective_width_range": [
            int(min(spatial_null_candidate_widths)),
            int(max(spatial_null_candidate_widths)),
        ],
        "minimum_molecule_opportunities": int(
            minimum_molecule_opportunities
        ),
        "minimum_edge_effective_opportunities": float(
            minimum_edge_effective_opportunities
        ),
        "minimum_edge_information_spread_nats": float(
            minimum_edge_information_spread
        ),
        "minimum_edge_q_margin_to_best_non_equivalent_nats": float(
            minimum_edge_information_spread
        ),
        "minimum_edge_q_margin_per_effective_molecule_nats": float(
            minimum_edge_q_margin_per_effective_molecule
        ),
        "minimum_spatial_null_conflict_fraction": float(
            minimum_spatial_null_conflict_fraction
        ),
        "edge_q_margin_threshold_basis": (
            "dual_accumulated_margin_and_per_effective_site_molecule_margin"
        ),
        "legacy_field_aliases": {
            "candidate_expected_log_likelihood_surface": (
                "candidate_expected_complete_data_q_surface"
            ),
            "pooled_information_spread_nats": (
                "pooled_information_margin_to_best_non_equivalent_nats"
            ),
            "minimum_edge_information_spread_nats": (
                "minimum_edge_q_margin_to_best_non_equivalent_nats"
            ),
        },
        "candidate_class_set_fixed": True,
        "merge_split_enabled": False,
        "eligible_molecules": len(eligible_by_molecule),
        "modeled_molecules": len(retained_reads),
        "uninformative_molecules": int(uninformative_molecules),
        "uninformative_molecules_by_strand": dict(
            sorted(uninformative_molecules_by_strand.items())
        ),
        "unmodeled_topology_molecules": int(unmodeled_topology_molecules),
        "nucleosome_seed_topology_molecules": int(
            nucleosome_seed_topology_molecules
        ),
        "structured_seed_topology_coverage": (
            (len(retained_reads) - unmodeled_topology_molecules)
            / len(retained_reads)
            if retained_reads
            else 0.0
        ),
        "modeled_molecule_fraction": (
            len(retained_reads) / len(eligible_by_molecule)
            if eligible_by_molecule
            else 0.0
        ),
        "informative_molecule_fraction": (
            sum(
                any(
                    read.interval_evidence(site.start, site.end)[1] > 0
                    for site in sites
                )
                for read in eligible_by_molecule.values()
            )
            / len(eligible_by_molecule)
            if eligible_by_molecule
            else 0.0
        ),
        "informative_molecule_definition": (
            "fully_mapped_molecule_with_at_least_one_opportunity_inside_any_"
            "seeded_atomic_site"
        ),
        "spatial_null_component": {
            "name": "P0:unanchored_single_interval",
            "emission": (
                "one_contiguous_protected_interval_with_uniform_physical_"
                "width_then_uniform_reference_start_excluding_anchored_grid"
            ),
            "geometry_training": False,
            "candidate_interval_count": len(spatial_null_intervals),
            "candidate_widths": spatial_null_candidate_widths,
            "requested_width_range": [
                int(spatial_null_minimum_width),
                int(spatial_null_maximum_width),
            ],
            "effective_width_range": [
                int(min(spatial_null_candidate_widths)),
                int(max(spatial_null_candidate_widths)),
            ],
            "candidate_envelope": [int(envelope.start), int(envelope.end)],
            "anchored_reference_intervals_excluded": True,
            "anchored_candidate_width_range": [
                int(min(anchored_candidate_widths)),
                int(max(anchored_candidate_widths)),
            ],
            "can_represent_all_anchored_candidate_widths": (
                not anchored_widths_outside_spatial_null
            ),
            "anchored_candidate_widths_outside_null_prior": [
                int(width) for width in anchored_widths_outside_spatial_null
            ],
            "excluded_reference_interval_count": len(
                spatial_null_excluded_sorted
            ),
            "excluded_reference_intervals_sha256": (
                spatial_null_exclusion_digest
            ),
            "probability": float(weights[spatial_null_component_index]),
            "effective_molecule_support": (
                float(np.sum(responsibilities[:, spatial_null_component_index]))
                if retained_reads
                else 0.0
            ),
            "posterior_over_half_molecules": (
                int(
                    np.sum(
                        responsibilities[:, spatial_null_component_index] > 0.5
                    )
                )
                if retained_reads
                else 0
            ),
            "interpretation": (
                "structured_residual_null_with_explicit_look_elsewhere_penalty"
            ),
        },
        "unmodeled_component": {
            "name": "U:diffuse_iid_opportunity_protection",
            "emission": (
                "independent_protected_opportunities_integrated_under_beta_1_1"
            ),
            "geometry_training": False,
            "probability": float(weights[unmodeled_component_index]),
            "effective_molecule_support": (
                float(np.sum(responsibilities[:, unmodeled_component_index]))
                if retained_reads
                else 0.0
            ),
            "posterior_over_half_molecules": (
                int(
                    np.sum(
                        responsibilities[:, unmodeled_component_index] > 0.5
                    )
                )
                if retained_reads
                else 0
            ),
            "mean_responsibility_by_seed_topology": {
                "unmodeled": (
                    float(
                        np.mean(
                            responsibilities[
                                np.asarray(initially_unmodeled_topology, dtype=bool),
                                unmodeled_component_index,
                            ]
                        )
                    )
                    if any(initially_unmodeled_topology)
                    else None
                ),
                "represented": (
                    float(
                        np.mean(
                            responsibilities[
                                ~np.asarray(initially_unmodeled_topology, dtype=bool),
                                unmodeled_component_index,
                            ]
                        )
                    )
                    if retained_reads and not all(initially_unmodeled_topology)
                    else None
                ),
            },
        },
        "modeled_molecules_by_strand": {
            strand: sum(read.strand == strand for read in retained_reads)
            for strand in sorted({read.strand for read in retained_reads})
        },
        "seed_configuration_counts": [
            int(value) for value in seed_counts
        ],
        "hierarchical_weight_prior": {
            "family_pseudocount": float(pseudocount),
            "families": ["accessible", "anchored_tf", "spatial_null", "diffuse_null"],
            "component_pseudocounts": [
                float(value) for value in component_pseudocounts
            ],
            "anchored_tf_family_mass_invariant_to_configuration_count": True,
            "anchored_tf_substate_expansion_neutral": True,
            "anchored_tf_distribution_policy": (
                "equal_mass_over_unique_biological_family_sets_then_equal_"
                "mass_over_valid_substate_configurations_within_each_set"
            ),
            "anchored_tf_family_set_groups": [
                {
                    "family_ids": list(family_set),
                    "configuration_indices": [int(index) for index in indices],
                    "configuration_names": [
                        configuration_list[index].name for index in indices
                    ],
                    "total_pseudocount": float(
                        np.sum(
                            component_pseudocounts[
                                np.asarray(indices, dtype=np.int64)
                            ]
                        )
                    ),
                }
                for family_set, indices in sorted(
                    tf_configuration_indices_by_family_set.items()
                )
            ],
        },
        "configuration_names": component_names,
        "configuration_probabilities": [
            float(value) for value in weights
        ],
        "configuration_equivalence_class_ids": [
            *configuration_equivalence_ids,
            None,
            None,
        ],
        "configuration_equivalence_classes": sorted(
            configuration_equivalence_classes,
            key=lambda value: value["configuration_indices"],
        ),
        "configuration_equivalence_definition": (
            "identical_protected_opportunity_assignment_for_every_modeled_"
            "molecule_at_final_geometry"
        ),
        "component_likelihood_equivalence_class_ids": (
            component_likelihood_equivalence_class_ids
        ),
        "component_likelihood_equivalence_classes": (
            component_likelihood_equivalence_classes
        ),
        "component_likelihood_equivalence_definition": (
            "per_molecule_log_likelihood_vectors_numerically_indistinguishable_"
            "at_frozen_geometry_with_rtol_1e-14_atol_1e-12;_structured_"
            "configurations_first_quotiented_by_exact_opportunity_projection"
        ),
        "tf_families": tf_families,
        "iterations": int(iteration),
        "converged": bool(converged),
        "convergence_reason": convergence_reason,
        "convergence_tolerance": float(tol),
        "final_maximum_weight_change": final_weight_change,
        "final_objective_change_per_molecule": (
            final_objective_change_per_molecule
        ),
        "objective": final_objective,
        "objective_definition": (
            "raw_mixture_log_likelihood_plus_hierarchical_family_weight_log_prior"
        ),
        "raw_mixture_log_likelihood": final_raw_log_likelihood,
        "objective_trace": objective_trace,
        "geometry": geometry,
        "probability_calibration": "not_held_out_calibrated",
    }


def fit_multistart_iterative_tf_class_geometry_model(
    reads: Sequence[ReadEvidence],
    sites: Sequence[SiteTemplate],
    configurations: Optional[Sequence[Configuration]] = None,
    *,
    boundary_search_radius: int = 2,
    pseudocount: float = 0.5,
    center_radius: int = 10,
    max_iter: int = 100,
    tol: float = 1e-8,
    analysis_envelope: Optional[Tuple[int, int]] = None,
    spatial_null_exclusion_intervals: Optional[
        Sequence[Tuple[int, int]]
    ] = None,
    stratum_semantics: str = "unspecified",
    spatial_null_padding: int = 20,
    spatial_null_minimum_width: int = 1,
    spatial_null_maximum_width: int = 80,
    minimum_molecule_opportunities: int = 3,
    minimum_edge_effective_opportunities: float = 10.0,
    minimum_edge_information_spread: float = math.log(10.0),
    minimum_edge_q_margin_per_effective_molecule: float = 0.1,
    minimum_spatial_null_conflict_fraction: float = 0.1,
) -> dict:
    """Fit the fixed catalog from deterministic geometry/weight starts.

    The mixture-boundary objective is non-convex even for one atomic site.
    This wrapper makes that fact explicit and auditable: it fits a small set of
    predeclared starts, selects only by the common training objective, and
    retains every start's convergence and geometry summary.  Held-out scoring
    remains a separate operation and is never used to select an initialization.
    """
    if boundary_search_radius < 1:
        raise ValueError("multistart geometry requires a positive search radius")
    if not sites:
        raise ValueError("multistart geometry requires at least one site")
    configuration_list = list(
        enumerate_configurations(sites, include_nucleosome=False)
        if configurations is None
        else configurations
    )
    delta = min(2, boundary_search_radius)
    seed = [(int(site.start), int(site.end)) for site in sites]
    starts = [
        (
            "calls.seed",
            "ordinary_calls",
            seed,
        ),
        (
            "calls.shift_left",
            "ordinary_calls",
            [(start - delta, end - delta) for start, end in seed],
        ),
        (
            "calls.shift_right",
            "ordinary_calls",
            [(start + delta, end + delta) for start, end in seed],
        ),
        (
            "calls.expanded",
            "ordinary_calls",
            [(start - delta, end + delta) for start, end in seed],
        ),
        (
            "calls.contracted",
            "ordinary_calls",
            [
                (start + delta, end - delta)
                if end - start > 2 * delta
                else (start, end)
                for start, end in seed
            ],
        ),
        (
            "uniform.seed",
            "uniform",
            seed,
        ),
    ]
    def start_preserves_configuration_topology(
        initial_geometry: Sequence[Tuple[int, int]],
    ) -> bool:
        """Return whether a proposed start is valid for every configuration.

        A deterministic stress start can expand two otherwise compatible sites
        into one another.  That is a property of the start, not a failure of
        the fixed catalog, so multistart records and skips it rather than
        aborting all of the remaining valid starts.
        """
        for configuration in configuration_list:
            ordered_indices = sorted(
                configuration.site_indices,
                key=lambda index: (
                    initial_geometry[index][0],
                    initial_geometry[index][1],
                    index,
                ),
            )
            if any(
                initial_geometry[left][1] > initial_geometry[right][0]
                for left, right in zip(
                    ordered_indices, ordered_indices[1:]
                )
            ):
                return False
        return True

    fitted = []
    skipped = []
    seen_initializations = set()
    for start_name, initialization_mode, initial_geometry in starts:
        initialization_signature = (
            initialization_mode,
            tuple((int(start), int(end)) for start, end in initial_geometry),
        )
        if initialization_signature in seen_initializations:
            skipped.append(
                {
                    "start": start_name,
                    "initialization_mode": initialization_mode,
                    "initial_geometry_intervals": [
                        [int(start), int(end)]
                        for start, end in initial_geometry
                    ],
                    "status": "skipped_duplicate_initialization",
                    "reason": "same_weight_mode_and_geometry_as_an_earlier_start",
                }
            )
            continue
        seen_initializations.add(initialization_signature)
        if not start_preserves_configuration_topology(initial_geometry):
            skipped.append(
                {
                    "start": start_name,
                    "initialization_mode": initialization_mode,
                    "initial_geometry_intervals": [
                        [int(start), int(end)]
                        for start, end in initial_geometry
                    ],
                    "status": "skipped_invalid_configuration_topology",
                    "reason": (
                        "initial_geometry_overlaps_within_a_structured_"
                        "configuration"
                    ),
                }
            )
            continue
        model = fit_iterative_tf_class_geometry_model(
            reads,
            sites,
            configuration_list,
            boundary_search_radius=boundary_search_radius,
            pseudocount=pseudocount,
            center_radius=center_radius,
            max_iter=max_iter,
            tol=tol,
            initialization_mode=initialization_mode,
            initial_geometry_intervals=initial_geometry,
            tie_break_mode="initial",
            analysis_envelope=analysis_envelope,
            spatial_null_exclusion_intervals=(
                spatial_null_exclusion_intervals
            ),
            stratum_semantics=stratum_semantics,
            spatial_null_padding=spatial_null_padding,
            spatial_null_minimum_width=spatial_null_minimum_width,
            spatial_null_maximum_width=spatial_null_maximum_width,
            minimum_molecule_opportunities=minimum_molecule_opportunities,
            minimum_edge_effective_opportunities=(
                minimum_edge_effective_opportunities
            ),
            minimum_edge_information_spread=minimum_edge_information_spread,
            minimum_edge_q_margin_per_effective_molecule=(
                minimum_edge_q_margin_per_effective_molecule
            ),
            minimum_spatial_null_conflict_fraction=(
                minimum_spatial_null_conflict_fraction
            ),
        )
        fitted.append((start_name, model))
    if not fitted:
        raise ValueError(
            "all multistart initial geometries overlap within a structured configuration"
        )
    best_objective = max(float(value[1]["objective"]) for value in fitted)
    objective_ties = [
        value
        for value in fitted
        if math.isclose(
            float(value[1]["objective"]),
            best_objective,
            rel_tol=1e-12,
            abs_tol=1e-10,
        )
    ]
    best_raw_likelihood = max(
        float(value[1]["raw_mixture_log_likelihood"])
        for value in objective_ties
    )
    likelihood_ties = [
        value
        for value in objective_ties
        if math.isclose(
            float(value[1]["raw_mixture_log_likelihood"]),
            best_raw_likelihood,
            rel_tol=1e-12,
            abs_tol=1e-10,
        )
    ]
    selected_name, selected_model = min(
        likelihood_ties, key=lambda value: value[0]
    )
    best_objective = float(selected_model["objective"])
    result = dict(selected_model)
    result["initialization_strategy"] = (
        "deterministic_multistart_maximum_penalized_training_objective"
    )
    result["selected_multistart"] = selected_name
    result["multistart_attempted"] = len(starts)
    result["multistart_fitted"] = len(fitted)
    result["multistart_skipped"] = skipped
    result["multistart"] = [
        {
            "start": start_name,
            "initialization_mode": model["initialization_mode"],
            "initial_geometry_intervals": model["initial_geometry_intervals"],
            "selected_geometry": [
                record["selected_interval"] for record in model["geometry"]
            ],
            "objective": float(model["objective"]),
            "objective_delta_from_selected": float(
                model["objective"] - best_objective
            ),
            "raw_mixture_log_likelihood": float(
                model["raw_mixture_log_likelihood"]
            ),
            "configuration_probabilities": list(
                model["configuration_probabilities"]
            ),
            "converged": bool(model["converged"]),
            "iterations": int(model["iterations"]),
        }
        for start_name, model in fitted
    ]
    return result


def score_iterative_tf_class_geometry_model(
    reads: Sequence[ReadEvidence],
    sites: Sequence[SiteTemplate],
    configurations: Sequence[Configuration],
    model: Mapping[str, object],
    *,
    include_molecule_records: bool = False,
) -> dict:
    """Score a frozen v2 TF-geometry model on independent molecules.

    No weight, boundary, or candidate is updated here.  This separation is
    what makes molecule-disjoint model comparison and depth titration valid.
    Scores are log-likelihood ratios relative to the chemistry-specific
    accessible emission already encoded in each ``ReadEvidence.steps`` array.
    Consequently, model comparisons are meaningful only for the same held-out
    molecules and envelope.
    """
    if model.get("schema") != "fiberhmm.iterative_tf_class_geometry_model.v2":
        raise ValueError("frozen scorer requires a v2 iterative TF geometry model")
    if not sites or not configurations:
        raise ValueError("frozen scorer requires sites and configurations")
    if any(configuration.is_nucleosome for configuration in configurations):
        raise ValueError("frozen scorer does not accept nucleosome configurations")
    input_sites = list(sites)
    site_order = sorted(
        range(len(input_sites)),
        key=lambda index: (
            input_sites[index].start,
            input_sites[index].end,
            input_sites[index].site_id,
            index,
        ),
    )
    old_to_new = {old: new for new, old in enumerate(site_order)}
    sites = [input_sites[index] for index in site_order]
    configurations = [
        Configuration(
            configuration.name,
            tuple(
                sorted(old_to_new[index] for index in configuration.site_indices)
            ),
            is_nucleosome=False,
        )
        for configuration in configurations
    ]
    configurations = sorted(
        configurations,
        key=lambda configuration: (
            bool(configuration.site_indices),
            tuple(
                sites[index].site_id for index in configuration.site_indices
            ),
            configuration.name,
        ),
    )
    expected_structured_names = [
        configuration.name for configuration in configurations
    ]
    component_names = [str(value) for value in model["configuration_names"]]
    expected_names = [
        *expected_structured_names,
        "P0:unanchored_single_interval",
        "U:diffuse_iid_opportunity_protection",
    ]
    if component_names != expected_names:
        raise ValueError("frozen model configuration order does not match inputs")
    weights = np.asarray(model["configuration_probabilities"], dtype=np.float64)
    if len(weights) != len(component_names) or np.any(weights < 0.0):
        raise ValueError("frozen model has invalid configuration probabilities")
    if not math.isclose(float(np.sum(weights)), 1.0, abs_tol=1e-8):
        raise ValueError("frozen model configuration probabilities do not sum to one")

    envelope_values = model.get("envelope")
    if not isinstance(envelope_values, list) or len(envelope_values) != 2:
        raise ValueError("frozen model is missing its fitted envelope")
    envelope = IntervalCall(int(envelope_values[0]), int(envelope_values[1]))
    boundary_search_radius = int(model["boundary_search_radius"])
    geometry_records = model.get("geometry")
    if not isinstance(geometry_records, list) or len(geometry_records) != len(sites):
        raise ValueError("frozen model geometry does not match site count")
    selected_intervals: List[Tuple[int, int]] = []
    for site, record in zip(sites, geometry_records):
        if not isinstance(record, Mapping) or record.get("site_id") != site.site_id:
            raise ValueError("frozen model geometry site order does not match inputs")
        seed_interval = record.get("seed_interval")
        if seed_interval != [int(site.start), int(site.end)]:
            raise ValueError("frozen model seed coordinates do not match inputs")
        if str(record.get("family_id", site.site_id)) != str(
            site.family_id or site.site_id
        ):
            raise ValueError("frozen model family identity does not match inputs")
        if str(record.get("substate_id", site.site_id)) != str(
            site.substate_id or site.site_id
        ):
            raise ValueError("frozen model substate identity does not match inputs")
        interval = record.get("selected_interval")
        if not isinstance(interval, list) or len(interval) != 2:
            raise ValueError("frozen model geometry has an invalid interval")
        selected_intervals.append((int(interval[0]), int(interval[1])))

    eligible_by_molecule: Dict[Tuple[str, str, str], ReadEvidence] = {}
    for read in reads:
        if not read.fully_maps(envelope.start, envelope.end):
            continue
        previous = eligible_by_molecule.get(read.molecule_id)
        if previous is None:
            eligible_by_molecule[read.molecule_id] = read
            continue
        previous_opportunities = previous.interval_evidence(
            envelope.start, envelope.end
        )[1]
        current_opportunities = read.interval_evidence(
            envelope.start, envelope.end
        )[1]
        if current_opportunities > previous_opportunities or (
            current_opportunities == previous_opportunities
            and _read_evidence_content_sha256(read)
            < _read_evidence_content_sha256(previous)
        ):
            eligible_by_molecule[read.molecule_id] = read
    retained_reads = [
        eligible_by_molecule[molecule_id]
        for molecule_id in sorted(eligible_by_molecule)
        if eligible_by_molecule[molecule_id].interval_evidence(
            envelope.start, envelope.end
        )[1]
        >= int(model["minimum_molecule_opportunities"])
    ]

    structured_candidates = [
        [
            (start, end)
            for start in range(
                site.start - boundary_search_radius,
                site.start + boundary_search_radius + 1,
            )
            for end in range(
                site.end - boundary_search_radius,
                site.end + boundary_search_radius + 1,
            )
            if end > start
        ]
        for site in sites
    ]
    spatial_null_exclusions = model.get("spatial_null_exclusion_intervals")
    if spatial_null_exclusions is None:
        # Backward compatibility for early in-memory v2 models created before
        # the common null-universe field was added.
        spatial_null_exclusion_intervals = [
            interval
            for candidates in structured_candidates
            for interval in candidates
        ]
    else:
        if not isinstance(spatial_null_exclusions, list) or any(
            not isinstance(interval, list) or len(interval) != 2
            for interval in spatial_null_exclusions
        ):
            raise ValueError("frozen model has invalid spatial-null exclusions")
        spatial_null_exclusion_intervals = [
            (int(interval[0]), int(interval[1]))
            for interval in spatial_null_exclusions
        ]
        observed_exclusion_digest = hashlib.sha256(
            np.asarray(
                sorted(set(spatial_null_exclusion_intervals)), dtype="<i8"
            ).tobytes()
        ).hexdigest()
        expected_exclusion_digest = model.get(
            "spatial_null_exclusion_sha256"
        )
        if (
            expected_exclusion_digest is not None
            and observed_exclusion_digest != expected_exclusion_digest
        ):
            raise ValueError("frozen model spatial-null exclusions fail digest")
    current_anchored_intervals = {
        interval
        for candidates in structured_candidates
        for interval in candidates
    }
    if not current_anchored_intervals <= set(spatial_null_exclusion_intervals):
        raise ValueError(
            "frozen model spatial-null exclusions omit an anchored candidate"
        )
    spatial_null_intervals = _spatial_null_candidate_intervals(
        envelope,
        [spatial_null_exclusion_intervals],
        minimum_width=int(model["spatial_null_minimum_width"]),
        maximum_width=int(model["spatial_null_maximum_width"]),
    )
    component_count = len(component_names)
    spatial_null_index = len(configurations)
    diffuse_null_index = len(configurations) + 1
    likelihoods = np.zeros(
        (len(retained_reads), component_count), dtype=np.float64
    )
    opportunity_counts = np.zeros(len(retained_reads), dtype=np.int64)
    selected_site_scores = _interval_evidence_matrix(
        retained_reads,
        selected_intervals,
        summation_mode="slice_sum",
    )[0]
    spatial_null_scores = _spatial_null_configuration_log_likelihoods(
        retained_reads, spatial_null_intervals
    )
    for read_index, read in enumerate(retained_reads):
        site_scores = selected_site_scores[read_index]
        for configuration_index, configuration in enumerate(configurations):
            likelihoods[read_index, configuration_index] = float(
                np.sum(site_scores[list(configuration.site_indices)])
            )
        likelihoods[read_index, spatial_null_index] = spatial_null_scores[
            read_index
        ]
        likelihoods[read_index, diffuse_null_index] = (
            _diffuse_unmodeled_configuration_log_likelihood(read, envelope)
        )
        opportunity_counts[read_index] = read.interval_evidence(
            envelope.start, envelope.end
        )[1]

    if retained_reads:
        log_joint = likelihoods + np.log(np.maximum(weights, 1e-300))[None, :]
        log_normalizers = _logsumexp(log_joint, axis=1)
        responsibilities = np.exp(log_joint - log_normalizers[:, None])
    else:
        log_normalizers = np.zeros(0, dtype=np.float64)
        responsibilities = np.zeros((0, component_count), dtype=np.float64)
    total_log_likelihood = float(np.sum(log_normalizers))
    total_opportunities = int(np.sum(opportunity_counts))
    family_component_indices: Dict[str, List[int]] = {}
    for family in model.get("tf_families", []):
        if not isinstance(family, Mapping):
            raise ValueError("frozen model has an invalid TF-family record")
        indices = [int(value) for value in family["configuration_indices"]]
        if any(index < 0 or index >= len(configurations) for index in indices):
            raise ValueError("frozen model TF-family component index is invalid")
        family_component_indices[str(family["family_id"])] = indices
    molecule_records = []
    if include_molecule_records:
        for read_index, read in enumerate(retained_reads):
            maximum_index = int(np.argmax(responsibilities[read_index]))
            molecule_records.append(
                {
                    "molecule_id": list(read.molecule_id),
                    "strand": read.strand,
                    "opportunities": int(opportunity_counts[read_index]),
                    "log_likelihood_ratio": float(log_normalizers[read_index]),
                    "maximum_posterior_component": component_names[maximum_index],
                    "maximum_posterior_probability": float(
                        responsibilities[read_index, maximum_index]
                    ),
                    "component_posteriors": {
                        name: float(responsibilities[read_index, index])
                        for index, name in enumerate(component_names)
                    },
                    "family_posteriors": {
                        family_id: float(
                            np.sum(
                                responsibilities[
                                    read_index,
                                    np.asarray(indices, dtype=np.int64),
                                ]
                            )
                        )
                        for family_id, indices in family_component_indices.items()
                    },
                }
            )

    return {
        "schema": "fiberhmm.iterative_tf_class_geometry_score.v1",
        "model_id": str(model["model_id"]),
        "fit_schema": str(model["schema"]),
        "frozen_parameters": True,
        "envelope": [int(envelope.start), int(envelope.end)],
        "fully_mapped_molecules": len(eligible_by_molecule),
        "eligible_molecules": len(retained_reads),
        "uninformative_molecules": len(eligible_by_molecule) - len(retained_reads),
        "eligible_molecules_by_strand": {
            strand: sum(read.strand == strand for read in retained_reads)
            for strand in sorted({read.strand for read in retained_reads})
        },
        "total_opportunities": total_opportunities,
        "component_names": component_names,
        "component_probabilities": [float(value) for value in weights],
        "component_effective_molecule_support": {
            name: (
                float(np.sum(responsibilities[:, index]))
                if retained_reads
                else 0.0
            )
            for index, name in enumerate(component_names)
        },
        "family_effective_molecule_support": {
            family_id: (
                float(
                    np.sum(
                        responsibilities[
                            :,
                            np.asarray(indices, dtype=np.int64),
                        ]
                    )
                )
                if retained_reads
                else 0.0
            )
            for family_id, indices in family_component_indices.items()
        },
        "raw_mixture_log_likelihood_ratio": total_log_likelihood,
        "mean_log_likelihood_ratio_per_molecule": (
            total_log_likelihood / len(retained_reads) if retained_reads else None
        ),
        "mean_log_likelihood_ratio_per_opportunity": (
            total_log_likelihood / total_opportunities
            if total_opportunities
            else None
        ),
        "score_definition": (
            "frozen_mixture_predictive_log_likelihood_ratio_relative_to_"
            "chemistry_specific_accessible_emission"
        ),
        "configuration_equivalence_class_ids": list(
            model["configuration_equivalence_class_ids"]
        ),
        "molecules": molecule_records,
    }


def consolidate_tf_boundary_variant_sites(
    sites: Sequence[SiteTemplate],
    *,
    maximum_boundary_delta: int = 8,
    maximum_width_delta: int = 8,
    maximum_center_delta: float = 6.0,
    minimum_shorter_overlap_fraction: float = 0.75,
) -> List[List[SiteTemplate]]:
    """Group modest nested edge variants without merging nearby TF families.

    Compatibility is complete-linkage: every member of a returned group must
    be compatible with every other member.  This prevents a chain of small
    shifts from bridging two genuinely separated footprints.  The function is
    intentionally geometric proposal logic; raw chemistry is evaluated later
    by :func:`fit_boundary_marginalized_tf_family_model`.
    """
    if maximum_boundary_delta < 0:
        raise ValueError("maximum boundary delta must be non-negative")
    if maximum_width_delta < 0:
        raise ValueError("maximum width delta must be non-negative")
    if not math.isfinite(maximum_center_delta) or maximum_center_delta < 0.0:
        raise ValueError("maximum center delta must be finite and non-negative")
    if (
        not math.isfinite(minimum_shorter_overlap_fraction)
        or not 0.0 <= minimum_shorter_overlap_fraction <= 1.0
    ):
        raise ValueError("minimum shorter-overlap fraction must lie in [0,1]")

    ordered = sorted(
        sites,
        key=lambda site: (site.start, site.end, site.site_id),
    )

    def compatible(left: SiteTemplate, right: SiteTemplate) -> bool:
        overlap = max(
            0,
            min(left.end, right.end) - max(left.start, right.start),
        )
        shorter = min(left.end - left.start, right.end - right.start)
        if shorter <= 0:
            return False
        return bool(
            abs(left.start - right.start) <= maximum_boundary_delta
            and abs(left.end - right.end) <= maximum_boundary_delta
            and abs(
                (left.end - left.start) - (right.end - right.start)
            )
            <= maximum_width_delta
            and abs(left.center - right.center) <= maximum_center_delta
            and overlap / shorter >= minimum_shorter_overlap_fraction
        )

    groups: List[List[SiteTemplate]] = []
    for site in ordered:
        eligible = [
            index
            for index, group in enumerate(groups)
            if all(compatible(site, member) for member in group)
        ]
        if not eligible:
            groups.append([site])
            continue
        selected = min(
            eligible,
            key=lambda index: (
                sum(
                    abs(site.start - member.start)
                    + abs(site.end - member.end)
                    for member in groups[index]
                ),
                tuple(member.site_id for member in groups[index]),
            ),
        )
        groups[selected].append(site)
        groups[selected].sort(
            key=lambda member: (member.start, member.end, member.site_id)
        )
    groups.sort(
        key=lambda group: (
            min(site.start for site in group),
            max(site.end for site in group),
            tuple(site.site_id for site in group),
        )
    )
    return groups


def _boundary_family_candidate_intervals(
    seed_intervals: Sequence[Tuple[int, int]],
    boundary_search_radius: int,
) -> List[Tuple[int, int]]:
    seeds = sorted({(int(start), int(end)) for start, end in seed_intervals})
    if not seeds or any(end <= start for start, end in seeds):
        raise ValueError("boundary family requires positive-width seed intervals")
    if boundary_search_radius < 0:
        raise ValueError("boundary search radius must be non-negative")
    minimum_seed_width = min(end - start for start, end in seeds)
    maximum_seed_width = max(end - start for start, end in seeds)
    minimum_width = max(1, minimum_seed_width - boundary_search_radius)
    maximum_width = maximum_seed_width + boundary_search_radius
    start_minimum = min(start for start, _end in seeds) - boundary_search_radius
    start_maximum = max(start for start, _end in seeds) + boundary_search_radius
    end_minimum = min(end for _start, end in seeds) - boundary_search_radius
    end_maximum = max(end for _start, end in seeds) + boundary_search_radius
    return [
        (start, end)
        for start in range(start_minimum, start_maximum + 1)
        for end in range(end_minimum, end_maximum + 1)
        if minimum_width <= end - start <= maximum_width
    ]


def _boundary_family_seed_local_candidate_intervals(
    seed_intervals: Sequence[Tuple[int, int]],
    boundary_search_radius: int,
) -> List[Tuple[int, int]]:
    """Expand each observed substate locally without inventing hybrid edges.

    A family may contain compact and broad protected substates.  Combining the
    minimum/maximum edge ranges into one rectangle permits an unsupported
    interval whose left edge comes from one substate and right edge from
    another.  The union of seed-local grids preserves chemistry-specific edge
    uncertainty while retaining the observed width neighborhood of each
    substate.
    """

    seeds = sorted({(int(start), int(end)) for start, end in seed_intervals})
    if not seeds or any(end <= start for start, end in seeds):
        raise ValueError("boundary family requires positive-width seed intervals")
    if boundary_search_radius < 0:
        raise ValueError("boundary search radius must be non-negative")
    candidates = set()
    for seed_start, seed_end in seeds:
        seed_width = seed_end - seed_start
        minimum_width = max(1, seed_width - boundary_search_radius)
        maximum_width = seed_width + boundary_search_radius
        candidates.update(
            (start, end)
            for start in range(
                seed_start - boundary_search_radius,
                seed_start + boundary_search_radius + 1,
            )
            for end in range(
                seed_end - boundary_search_radius,
                seed_end + boundary_search_radius + 1,
            )
            if minimum_width <= end - start <= maximum_width
        )
    return sorted(candidates)


def _boundary_family_projection_groups(
    left_indices: np.ndarray,
    right_indices: np.ndarray,
) -> List[List[int]]:
    if left_indices.shape != right_indices.shape or left_indices.ndim != 2:
        raise ValueError("boundary projection indices must be matched matrices")
    groups: Dict[bytes, List[int]] = {}
    for candidate_index in range(left_indices.shape[1]):
        signature = np.column_stack(
            (
                left_indices[:, candidate_index],
                right_indices[:, candidate_index],
            )
        ).astype("<i8", copy=False).tobytes()
        groups.setdefault(signature, []).append(candidate_index)
    return sorted(groups.values(), key=lambda members: tuple(members))


def _weighted_integer_quantile(
    values: Sequence[int], weights: Sequence[float], quantile: float
) -> int:
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must lie in [0,1]")
    ordered = sorted(
        zip((int(value) for value in values), (float(weight) for weight in weights)),
        key=lambda item: item[0],
    )
    total = sum(weight for _value, weight in ordered)
    if total <= 0.0:
        raise ValueError("weighted quantile requires positive total weight")
    threshold = quantile * total
    accumulated = 0.0
    for value, weight in ordered:
        accumulated += weight
        if accumulated >= threshold:
            return value
    return ordered[-1][0]


def _boundary_family_distribution_summary(
    candidate_intervals: Sequence[Tuple[int, int]],
    projection_groups: Sequence[Sequence[int]],
    conditional_class_probabilities: Sequence[float],
    seed_intervals: Sequence[Tuple[int, int]],
) -> dict:
    physical_probabilities = np.zeros(len(candidate_intervals), dtype=np.float64)
    for members, probability in zip(
        projection_groups, conditional_class_probabilities
    ):
        if members:
            physical_probabilities[np.asarray(members, dtype=np.int64)] = (
                float(probability) / len(members)
            )
    if not math.isclose(
        float(np.sum(physical_probabilities)), 1.0, abs_tol=1e-8
    ):
        raise ValueError("boundary family physical probabilities do not sum to one")
    seed_start = float(np.median([start for start, _end in seed_intervals]))
    seed_end = float(np.median([end for _start, end in seed_intervals]))
    conditional = np.asarray(conditional_class_probabilities, dtype=np.float64)
    maximum_class_probability = float(np.max(conditional))
    maximum_class_indices = np.flatnonzero(
        np.isclose(
            conditional,
            maximum_class_probability,
            rtol=1e-14,
            atol=1e-15,
        )
    )
    selected_class_index = min(
        (int(index) for index in maximum_class_indices),
        key=lambda class_index: min(
            (
                abs(candidate_intervals[candidate_index][0] - seed_start)
                + abs(candidate_intervals[candidate_index][1] - seed_end),
                candidate_intervals[candidate_index],
            )
            for candidate_index in projection_groups[class_index]
        ),
    )
    selected_index = min(
        (int(index) for index in projection_groups[selected_class_index]),
        key=lambda index: (
            abs(candidate_intervals[index][0] - seed_start)
            + abs(candidate_intervals[index][1] - seed_end),
            candidate_intervals[index],
        ),
    )
    starts = [start for start, _end in candidate_intervals]
    ends = [end for _start, end in candidate_intervals]
    positive = conditional[conditional > 0.0]
    entropy = float(-np.sum(positive * np.log(positive))) if positive.size else 0.0
    return {
        "canonical_interval": [
            int(candidate_intervals[selected_index][0]),
            int(candidate_intervals[selected_index][1]),
        ],
        "canonical_interval_probability": float(
            physical_probabilities[selected_index]
        ),
        "canonical_projection_class_index": int(selected_class_index),
        "boundary_credible_envelope_95": {
            "start": [
                _weighted_integer_quantile(starts, physical_probabilities, 0.025),
                _weighted_integer_quantile(starts, physical_probabilities, 0.975),
            ],
            "end": [
                _weighted_integer_quantile(ends, physical_probabilities, 0.025),
                _weighted_integer_quantile(ends, physical_probabilities, 0.975),
            ],
        },
        "geometry_projection_class_entropy_nats": entropy,
        "geometry_projection_effective_class_count": float(math.exp(entropy)),
        "map_projection_class_probability": maximum_class_probability,
        "physical_interval_probabilities": [
            {
                "interval": [int(start), int(end)],
                "conditional_probability": float(physical_probabilities[index]),
            }
            for index, (start, end) in enumerate(candidate_intervals)
            if physical_probabilities[index] > 0.0
        ],
    }


def fit_boundary_marginalized_tf_family_model(
    reads: Sequence[ReadEvidence],
    family_id: str,
    seed_intervals: Sequence[Tuple[int, int]],
    *,
    boundary_search_radius: int = 3,
    candidate_interval_mode: str = "rectangular_boundary_grid",
    pseudocount: float = 0.5,
    analysis_envelope: Optional[Tuple[int, int]] = None,
    spatial_null_exclusion_intervals: Optional[
        Sequence[Tuple[int, int]]
    ] = None,
    spatial_null_padding: int = 20,
    spatial_null_minimum_width: int = 1,
    spatial_null_maximum_width: int = 80,
    minimum_molecule_opportunities: int = 3,
    max_iter: int = 500,
    tol: float = 1e-9,
    objective_tol_per_molecule: float = 1e-6,
    stratum_semantics: str = "unspecified",
    evidence_summation_mode: str = "prefix",
) -> dict:
    """Fit one site-consensus state while marginalizing modest boundary ambiguity.

    Candidate coordinates are first quotiented by their complete opportunity
    projection across the fitted molecules.  The anchored-TF pseudocount is
    split over those observable geometry classes, not over raw coordinates,
    so adding coordinates inside an opportunity gap cannot inflate the family
    prior.  Geometry-class probabilities sum to one conditional on the family;
    raw calls and evidence arrays are never modified.  ``prefix`` evaluates
    the same additive interval likelihood as the reference ``slice_sum``
    reference backend using immutable per-read cumulative-sum caches.  The
    backend is recorded in the frozen model, and ``slice_sum`` remains
    available for numerical regression and legacy reproduction.
    """
    if not family_id:
        raise ValueError("boundary family requires a non-empty family id")
    if not math.isfinite(pseudocount) or pseudocount < 0.0:
        raise ValueError("pseudocount must be finite and non-negative")
    if minimum_molecule_opportunities < 1:
        raise ValueError("minimum molecule opportunities must be positive")
    if max_iter < 1 or not math.isfinite(tol) or tol <= 0.0:
        raise ValueError("invalid boundary family convergence controls")
    if (
        not math.isfinite(objective_tol_per_molecule)
        or objective_tol_per_molecule <= 0.0
    ):
        raise ValueError("objective tolerance per molecule must be positive")
    if stratum_semantics not in {
        "physical_complementary",
        "diagnostic_partition",
        "unspecified",
    }:
        raise ValueError("invalid stratum semantics")
    if evidence_summation_mode not in {"prefix", "slice_sum"}:
        raise ValueError("invalid interval evidence summation mode")
    seeds = sorted({(int(start), int(end)) for start, end in seed_intervals})
    if candidate_interval_mode not in {
        "rectangular_boundary_grid",
        "seed_local_boundary_grid",
        "exact_seed_intervals",
    }:
        raise ValueError("invalid boundary-family candidate interval mode")
    if candidate_interval_mode == "exact_seed_intervals":
        if boundary_search_radius != 0:
            raise ValueError(
                "exact seed intervals require boundary_search_radius=0"
            )
        if not seeds or any(end <= start for start, end in seeds):
            raise ValueError("boundary family requires positive-width seed intervals")
        candidates = list(seeds)
    elif candidate_interval_mode == "seed_local_boundary_grid":
        candidates = _boundary_family_seed_local_candidate_intervals(
            seeds, boundary_search_radius
        )
    else:
        candidates = _boundary_family_candidate_intervals(
            seeds, boundary_search_radius
        )
    if analysis_envelope is None:
        envelope = IntervalCall(
            min(start for start, _end in candidates) - spatial_null_padding,
            max(end for _start, end in candidates) + spatial_null_padding,
        )
        analysis_envelope_explicit = False
    else:
        envelope = IntervalCall(
            int(analysis_envelope[0]), int(analysis_envelope[1])
        )
        if envelope.end <= envelope.start or any(
            start < envelope.start or end > envelope.end
            for start, end in candidates
        ):
            raise ValueError("analysis envelope must contain every family candidate")
        analysis_envelope_explicit = True

    eligible_by_molecule: Dict[Tuple[str, str, str], ReadEvidence] = {}
    for read in reads:
        if not read.fully_maps(envelope.start, envelope.end):
            continue
        previous = eligible_by_molecule.get(read.molecule_id)
        if previous is None:
            eligible_by_molecule[read.molecule_id] = read
            continue
        previous_opportunities = previous.interval_evidence(
            envelope.start, envelope.end
        )[1]
        current_opportunities = read.interval_evidence(
            envelope.start, envelope.end
        )[1]
        if current_opportunities > previous_opportunities or (
            current_opportunities == previous_opportunities
            and _read_evidence_content_sha256(read)
            < _read_evidence_content_sha256(previous)
        ):
            eligible_by_molecule[read.molecule_id] = read
    retained_reads = [
        eligible_by_molecule[molecule_id]
        for molecule_id in sorted(eligible_by_molecule)
        if eligible_by_molecule[molecule_id].interval_evidence(
            envelope.start, envelope.end
        )[1]
        >= minimum_molecule_opportunities
    ]
    if not retained_reads:
        raise ValueError("boundary family has no eligible molecules")

    score_matrix, left_indices, right_indices = _interval_evidence_matrix(
        retained_reads, candidates, summation_mode=evidence_summation_mode
    )
    projection_groups = _boundary_family_projection_groups(
        left_indices, right_indices
    )
    geometry_scores = np.column_stack(
        [score_matrix[:, members[0]] for members in projection_groups]
    )
    if spatial_null_exclusion_intervals is None:
        spatial_null_exclusions = list(candidates)
        spatial_null_exclusions_explicit = False
    else:
        spatial_null_exclusions = sorted(
            {
                (int(start), int(end))
                for start, end in spatial_null_exclusion_intervals
            }
        )
        if not spatial_null_exclusions or any(
            end <= start for start, end in spatial_null_exclusions
        ):
            raise ValueError(
                "spatial-null exclusion intervals must be non-empty and positive"
            )
        if any(
            start < envelope.start or end > envelope.end
            for start, end in spatial_null_exclusions
        ):
            raise ValueError(
                "spatial-null exclusion intervals must lie inside the analysis envelope"
            )
        if not set(candidates).issubset(spatial_null_exclusions):
            raise ValueError(
                "spatial-null exclusions must contain every anchored family candidate"
            )
        spatial_null_exclusions_explicit = True
    spatial_null_exclusion_sha256 = hashlib.sha256(
        np.asarray(spatial_null_exclusions, dtype="<i8").tobytes()
    ).hexdigest()
    spatial_intervals = _spatial_null_candidate_intervals(
        envelope,
        [spatial_null_exclusions],
        minimum_width=spatial_null_minimum_width,
        maximum_width=spatial_null_maximum_width,
    )
    spatial_scores = _spatial_null_configuration_log_likelihoods(
        retained_reads, spatial_intervals
    )
    diffuse_scores = np.asarray(
        [
            _diffuse_unmodeled_configuration_log_likelihood(read, envelope)
            for read in retained_reads
        ],
        dtype=np.float64,
    )
    geometry_count = len(projection_groups)
    component_names = [
        "A",
        *[
            f"TF:{family_id}:geometry_projection_{index + 1}"
            for index in range(geometry_count)
        ],
        "P0:unanchored_single_interval",
        "U:diffuse_iid_opportunity_protection",
    ]
    likelihoods = np.column_stack(
        (
            np.zeros(len(retained_reads), dtype=np.float64),
            geometry_scores,
            spatial_scores,
            diffuse_scores,
        )
    )
    component_pseudocounts = np.asarray(
        [
            pseudocount,
            *([pseudocount / geometry_count] * geometry_count),
            pseudocount,
            pseudocount,
        ],
        dtype=np.float64,
    )
    weights = np.asarray(
        [
            0.25,
            *([0.25 / geometry_count] * geometry_count),
            0.25,
            0.25,
        ],
        dtype=np.float64,
    )
    objective_trace = []
    converged = False
    convergence_reason = "maximum_iterations_reached"
    final_weight_change = None
    final_objective_change_per_molecule = None
    responsibilities = np.zeros_like(likelihoods)
    for iteration in range(1, max_iter + 1):
        log_joint = likelihoods + np.log(np.maximum(weights, 1e-300))[None, :]
        log_normalizers = _logsumexp(log_joint, axis=1)
        responsibilities = np.exp(log_joint - log_normalizers[:, None])
        objective = float(
            np.sum(log_normalizers)
            + np.sum(
                component_pseudocounts
                * np.log(np.maximum(weights, 1e-300))
            )
        )
        objective_trace.append(objective)
        if len(objective_trace) > 1:
            final_objective_change_per_molecule = float(
                (objective_trace[-1] - objective_trace[-2])
                / len(retained_reads)
            )
            if (
                abs(final_objective_change_per_molecule)
                < objective_tol_per_molecule
            ):
                converged = True
                convergence_reason = "penalized_objective_change_per_molecule"
                break
        updated = np.sum(responsibilities, axis=0) + component_pseudocounts
        updated /= len(retained_reads) + float(np.sum(component_pseudocounts))
        updated = np.maximum(updated, 1e-300)
        updated /= np.sum(updated)
        final_weight_change = float(np.max(np.abs(updated - weights)))
        weights = updated
        if final_weight_change < tol:
            converged = True
            convergence_reason = "maximum_absolute_weight_change"
            break

    log_joint = likelihoods + np.log(np.maximum(weights, 1e-300))[None, :]
    log_normalizers = _logsumexp(log_joint, axis=1)
    responsibilities = np.exp(log_joint - log_normalizers[:, None])
    raw_log_likelihood = float(np.sum(log_normalizers))
    final_objective = float(
        raw_log_likelihood
        + np.sum(
            component_pseudocounts
            * np.log(np.maximum(weights, 1e-300))
        )
    )
    if not objective_trace or not math.isclose(
        final_objective, objective_trace[-1], rel_tol=0.0, abs_tol=1e-12
    ):
        objective_trace.append(final_objective)
    family_slice = slice(1, 1 + geometry_count)
    family_probability = float(np.sum(weights[family_slice]))
    conditional_class_probabilities = (
        weights[family_slice] / family_probability
        if family_probability > 0.0
        else np.full(geometry_count, 1.0 / geometry_count)
    )
    distribution = _boundary_family_distribution_summary(
        candidates,
        projection_groups,
        conditional_class_probabilities,
        seeds,
    )
    geometry_classes = []
    for class_index, members in enumerate(projection_groups):
        member_intervals = [candidates[index] for index in members]
        digest = hashlib.sha256(
            np.asarray(member_intervals, dtype="<i8").tobytes()
        ).hexdigest()[:16]
        geometry_classes.append(
            {
                "geometry_class_id": f"tfgeometryprojection_{digest}",
                "member_candidate_indices": [int(index) for index in members],
                "member_intervals": [
                    [int(start), int(end)] for start, end in member_intervals
                ],
                "conditional_probability_given_family": float(
                    conditional_class_probabilities[class_index]
                ),
                "mixture_probability": float(weights[1 + class_index]),
                "effective_molecule_support": float(
                    np.sum(responsibilities[:, 1 + class_index])
                ),
            }
        )
    structure_tokens = [
        family_id,
        *(f"seed={start}-{end}" for start, end in seeds),
        *(record["geometry_class_id"] for record in geometry_classes),
        f"envelope={envelope.start}-{envelope.end}",
        f"boundary_search_radius={boundary_search_radius}",
        f"pseudocount={pseudocount:.17g}",
        f"minimum_molecule_opportunities={minimum_molecule_opportunities}",
        f"spatial_null_padding={spatial_null_padding}",
        f"spatial_null_minimum_width={spatial_null_minimum_width}",
        f"spatial_null_maximum_width={spatial_null_maximum_width}",
        f"max_iter={max_iter}",
        f"tol={tol:.17g}",
        f"objective_tol_per_molecule={objective_tol_per_molecule:.17g}",
        f"stratum_semantics={stratum_semantics}",
        "prior=equal_four_blocks_equal_projection_classes",
    ]
    if spatial_null_exclusions_explicit:
        structure_tokens.extend(
            (
                "spatial_null_exclusions=explicit_common_universe",
                f"spatial_null_exclusion_sha256={spatial_null_exclusion_sha256}",
            )
        )
    if candidate_interval_mode != "rectangular_boundary_grid":
        structure_tokens.append(
            f"candidate_interval_mode={candidate_interval_mode}"
        )
    model_structure_id = "tfboundaryfamily_" + hashlib.sha256(
        "|".join(
            structure_tokens
        ).encode("utf-8")
    ).hexdigest()[:16]
    training_cohort_evidence_sha256 = hashlib.sha256(
        "\n".join(
            _read_evidence_content_sha256(read) for read in retained_reads
        ).encode("ascii")
    ).hexdigest()
    model_fit_id = "tfboundaryfamilyfit_" + hashlib.sha256(
        b"\x1f".join(
            (
                model_structure_id.encode("ascii"),
                training_cohort_evidence_sha256.encode("ascii"),
                np.asarray(weights, dtype="<f8").tobytes(),
            )
        )
    ).hexdigest()[:16]
    return {
        "schema": "fiberhmm.boundary_marginalized_tf_family_model.v1",
        "status": "experimental_report_only_fixed_family",
        "family_id": str(family_id),
        "model_structure_id": model_structure_id,
        "model_fit_id": model_fit_id,
        "training_cohort_evidence_sha256": training_cohort_evidence_sha256,
        "interval_evidence_summation_mode": evidence_summation_mode,
        "interval_evidence_backend_semantics": (
            "float64_additive_log_likelihood_over_identical_opportunity_slices"
        ),
        "seed_intervals": [[start, end] for start, end in seeds],
        "candidate_intervals": [[start, end] for start, end in candidates],
        "candidate_interval_count": len(candidates),
        "opportunity_projection_class_count": geometry_count,
        "geometry_classes": geometry_classes,
        **distribution,
        "family_probability": family_probability,
        "family_effective_molecule_support": float(
            np.sum(responsibilities[:, family_slice])
        ),
        "component_names": component_names,
        "component_probabilities": [float(value) for value in weights],
        "component_effective_molecule_support": {
            name: float(np.sum(responsibilities[:, index]))
            for index, name in enumerate(component_names)
        },
        "hierarchical_prior": {
            "family_pseudocount": float(pseudocount),
            "blocks": ["accessible", "anchored_tf_family", "spatial_null", "diffuse_null"],
            "anchored_family_total_pseudocount": float(pseudocount),
            "geometry_distribution_policy": (
                "equal_mass_over_complete_opportunity_projection_classes_"
                "not_reference_coordinates"
            ),
            "component_pseudocounts": [
                float(value) for value in component_pseudocounts
            ],
        },
        "envelope": [int(envelope.start), int(envelope.end)],
        "analysis_envelope_explicit": analysis_envelope_explicit,
        "boundary_search_radius": int(boundary_search_radius),
        "candidate_interval_mode": candidate_interval_mode,
        "minimum_molecule_opportunities": int(minimum_molecule_opportunities),
        "spatial_null_minimum_width": int(spatial_null_minimum_width),
        "spatial_null_maximum_width": int(spatial_null_maximum_width),
        "spatial_null_candidate_interval_count": len(spatial_intervals),
        "spatial_null_exclusions_explicit": spatial_null_exclusions_explicit,
        "spatial_null_exclusion_sha256": spatial_null_exclusion_sha256,
        "spatial_null_exclusion_intervals": [
            [int(start), int(end)] for start, end in spatial_null_exclusions
        ],
        "eligible_molecules": len(retained_reads),
        "eligible_molecules_by_strand": {
            strand: sum(read.strand == strand for read in retained_reads)
            for strand in sorted({read.strand for read in retained_reads})
        },
        "iterations": int(iteration),
        "converged": bool(converged),
        "convergence_reason": convergence_reason,
        "final_maximum_weight_change": final_weight_change,
        "objective_tolerance_per_molecule": float(
            objective_tol_per_molecule
        ),
        "final_objective_change_per_molecule": (
            final_objective_change_per_molecule
        ),
        "objective_trace": objective_trace,
        "raw_mixture_log_likelihood": raw_log_likelihood,
        "stratum_semantics": stratum_semantics,
        "raw_molecule_signal_policy": (
            "source_calls_positions_hits_contexts_and_steps_are_never_modified"
        ),
        "family_interpretation": (
            "one_locus_family_with_marginalized_boundary_distribution;_"
            "geometry_modes_are_not_distinct_biological_states_without_"
            "independent_split_support"
        ),
    }


def score_boundary_marginalized_tf_family_model(
    reads: Sequence[ReadEvidence],
    model: Mapping[str, object],
    *,
    include_molecule_records: bool = False,
    evidence_summation_mode: Optional[str] = None,
    molecule_record_minimum_standardized_family_posterior: Optional[
        float
    ] = None,
    molecule_record_minimum_family_vs_null_log_bayes_factor: Optional[
        float
    ] = None,
) -> dict:
    """Score a frozen boundary-marginalized family on independent molecules.

    Models created before the optimized backend was recorded reproduce their
    reference ``slice_sum`` scoring by default.  Optional molecule-record
    thresholds only suppress expensive conditional-boundary construction for
    rejected molecules; every cohort summary is still computed over the full
    eligible set.
    """
    if model.get("schema") != "fiberhmm.boundary_marginalized_tf_family_model.v1":
        raise ValueError("frozen boundary-family scorer requires a v1 model")
    selected_summation_mode = evidence_summation_mode
    if selected_summation_mode is None:
        selected_summation_mode = str(
            model.get("interval_evidence_summation_mode", "slice_sum")
        )
    if selected_summation_mode not in {"prefix", "slice_sum"}:
        raise ValueError("invalid interval evidence summation mode")
    if (
        molecule_record_minimum_standardized_family_posterior is not None
        and (
            not math.isfinite(
                molecule_record_minimum_standardized_family_posterior
            )
            or not 0.0
            <= molecule_record_minimum_standardized_family_posterior
            <= 1.0
        )
    ):
        raise ValueError("molecule record posterior threshold must be in [0,1]")
    if (
        molecule_record_minimum_family_vs_null_log_bayes_factor is not None
        and not math.isfinite(
            molecule_record_minimum_family_vs_null_log_bayes_factor
        )
    ):
        raise ValueError("molecule record log Bayes-factor threshold must be finite")
    if not include_molecule_records and (
        molecule_record_minimum_standardized_family_posterior is not None
        or molecule_record_minimum_family_vs_null_log_bayes_factor is not None
    ):
        raise ValueError(
            "molecule record thresholds require include_molecule_records=True"
        )
    envelope = IntervalCall(int(model["envelope"][0]), int(model["envelope"][1]))
    candidates = [
        (int(interval[0]), int(interval[1]))
        for interval in model["candidate_intervals"]
    ]
    classes = list(model["geometry_classes"])
    weights = np.asarray(model["component_probabilities"], dtype=np.float64)
    expected_component_count = len(classes) + 3
    if weights.size != expected_component_count or not math.isclose(
        float(np.sum(weights)), 1.0, abs_tol=1e-8
    ):
        raise ValueError("frozen boundary-family model has invalid probabilities")

    eligible_by_molecule: Dict[Tuple[str, str, str], ReadEvidence] = {}
    for read in reads:
        if not read.fully_maps(envelope.start, envelope.end):
            continue
        previous = eligible_by_molecule.get(read.molecule_id)
        if previous is None:
            eligible_by_molecule[read.molecule_id] = read
            continue
        previous_opportunities = previous.interval_evidence(
            envelope.start, envelope.end
        )[1]
        current_opportunities = read.interval_evidence(
            envelope.start, envelope.end
        )[1]
        if current_opportunities > previous_opportunities or (
            current_opportunities == previous_opportunities
            and _read_evidence_content_sha256(read)
            < _read_evidence_content_sha256(previous)
        ):
            eligible_by_molecule[read.molecule_id] = read
    retained_reads = [
        eligible_by_molecule[molecule_id]
        for molecule_id in sorted(eligible_by_molecule)
        if eligible_by_molecule[molecule_id].interval_evidence(
            envelope.start, envelope.end
        )[1]
        >= int(model["minimum_molecule_opportunities"])
    ]
    if not retained_reads:
        return {
            "schema": "fiberhmm.boundary_marginalized_tf_family_score.v1",
            "status": "frozen_model_score_empty_target_cohort",
            "family_id": str(model["family_id"]),
            "model_structure_id": str(model["model_structure_id"]),
            "model_fit_id": str(model.get("model_fit_id", "")),
            "frozen_parameters": True,
            "interval_evidence_summation_mode": selected_summation_mode,
            "envelope": [int(envelope.start), int(envelope.end)],
            "eligible_molecules": 0,
            "eligible_molecules_by_strand": {},
            "total_opportunities": 0,
            "family_effective_molecule_support": 0.0,
            "standardized_family_effective_molecule_support_equal_prior": 0.0,
            "family_posterior_over_half_molecules": 0,
            "family_posterior_over_nine_tenths_molecules": 0,
            "standardized_family_posterior_over_half_molecules": 0,
            "standardized_family_posterior_over_nine_tenths_molecules": 0,
            "family_vs_null_log_bayes_factor_nonnegative_molecules": 0,
            "family_vs_null_log_bayes_factor_at_least_log_10_molecules": 0,
            "median_family_vs_null_log_bayes_factor": None,
            "evidence_standardization": (
                "family_vs_conditionally_normalized_A_P0_U_predictive_"
                "log_bayes_factor;_equal_family_null_prior_posterior"
            ),
            "raw_mixture_log_likelihood_ratio": 0.0,
            "mean_log_likelihood_ratio_per_opportunity": None,
            "molecules": [],
            "molecule_record_selection": {
                "eligible_molecules": 0,
                "emitted_molecules": 0,
                "minimum_standardized_family_posterior": (
                    molecule_record_minimum_standardized_family_posterior
                ),
                "minimum_family_vs_null_log_bayes_factor": (
                    molecule_record_minimum_family_vs_null_log_bayes_factor
                ),
            },
        }

    candidate_scores, left_indices, right_indices = _interval_evidence_matrix(
        retained_reads, candidates, summation_mode=selected_summation_mode
    )
    geometry_scores = np.empty((len(retained_reads), len(classes)), dtype=np.float64)
    class_members = []
    for class_index, record in enumerate(classes):
        members = np.asarray(record["member_candidate_indices"], dtype=np.int64)
        if members.size == 0 or np.any(members < 0) or np.any(members >= len(candidates)):
            raise ValueError("frozen boundary-family geometry class is invalid")
        class_members.append(members)
        values = candidate_scores[:, members]
        geometry_scores[:, class_index] = (
            _logsumexp(values, axis=1) - math.log(members.size)
        )
    spatial_intervals = _spatial_null_candidate_intervals(
        envelope,
        [[tuple(interval) for interval in model["spatial_null_exclusion_intervals"]]],
        minimum_width=int(model["spatial_null_minimum_width"]),
        maximum_width=int(model["spatial_null_maximum_width"]),
    )
    spatial_scores = _spatial_null_configuration_log_likelihoods(
        retained_reads, spatial_intervals
    )
    diffuse_scores = np.asarray(
        [
            _diffuse_unmodeled_configuration_log_likelihood(read, envelope)
            for read in retained_reads
        ],
        dtype=np.float64,
    )
    likelihoods = np.column_stack(
        (
            np.zeros(len(retained_reads), dtype=np.float64),
            geometry_scores,
            spatial_scores,
            diffuse_scores,
        )
    )
    log_joint = likelihoods + np.log(np.maximum(weights, 1e-300))[None, :]
    log_normalizers = _logsumexp(log_joint, axis=1)
    responsibilities = np.exp(log_joint - log_normalizers[:, None])
    geometry_slice = slice(1, 1 + len(classes))
    family_posteriors = np.sum(responsibilities[:, geometry_slice], axis=1)
    family_weights = weights[geometry_slice]
    family_weight_total = float(np.sum(family_weights))
    null_weights = np.asarray(
        [weights[0], weights[-2], weights[-1]], dtype=np.float64
    )
    null_weight_total = float(np.sum(null_weights))
    if family_weight_total <= 0.0 or null_weight_total <= 0.0:
        raise ValueError("frozen boundary-family model has a degenerate prior block")
    family_log_predictive = _logsumexp(
        geometry_scores
        + np.log(np.maximum(family_weights / family_weight_total, 1e-300))[
            None, :
        ],
        axis=1,
    )
    null_log_predictive = _logsumexp(
        np.column_stack(
            (
                np.zeros(len(retained_reads), dtype=np.float64),
                spatial_scores,
                diffuse_scores,
            )
        )
        + np.log(np.maximum(null_weights / null_weight_total, 1e-300))[None, :],
        axis=1,
    )
    family_vs_null_log_bayes_factors = (
        family_log_predictive - null_log_predictive
    )
    standardized_family_posteriors = np.empty(
        len(retained_reads), dtype=np.float64
    )
    nonnegative = family_vs_null_log_bayes_factors >= 0.0
    standardized_family_posteriors[nonnegative] = 1.0 / (
        1.0 + np.exp(-family_vs_null_log_bayes_factors[nonnegative])
    )
    negative_exponentials = np.exp(
        family_vs_null_log_bayes_factors[~nonnegative]
    )
    standardized_family_posteriors[~nonnegative] = (
        negative_exponentials / (1.0 + negative_exponentials)
    )
    molecule_records = []
    if include_molecule_records:
        record_mask = np.ones(len(retained_reads), dtype=bool)
        if molecule_record_minimum_standardized_family_posterior is not None:
            record_mask &= standardized_family_posteriors >= float(
                molecule_record_minimum_standardized_family_posterior
            )
        if molecule_record_minimum_family_vs_null_log_bayes_factor is not None:
            record_mask &= family_vs_null_log_bayes_factors >= float(
                molecule_record_minimum_family_vs_null_log_bayes_factor
            )
        for read_index in np.flatnonzero(record_mask):
            read = retained_reads[int(read_index)]
            family_probability = float(family_posteriors[read_index])
            physical = np.zeros(len(candidates), dtype=np.float64)
            if family_probability > 0.0:
                for class_index, members in enumerate(class_members):
                    class_conditional = float(
                        responsibilities[read_index, 1 + class_index]
                        / family_probability
                    )
                    member_values = candidate_scores[read_index, members]
                    member_probabilities = _softmax(member_values)
                    physical[members] += class_conditional * member_probabilities
            if float(np.sum(physical)) > 0.0:
                physical /= np.sum(physical)
                maximum = float(np.max(physical))
                map_index = int(
                    min(
                        np.flatnonzero(
                            np.isclose(physical, maximum, rtol=1e-14, atol=1e-15)
                        ),
                        key=lambda index: candidates[int(index)],
                    )
                )
                positive = physical[physical > 0.0]
                entropy = float(-np.sum(positive * np.log(positive)))
                starts = [start for start, _end in candidates]
                ends = [end for _start, end in candidates]
                credible = {
                    "start": [
                        _weighted_integer_quantile(starts, physical, 0.025),
                        _weighted_integer_quantile(starts, physical, 0.975),
                    ],
                    "end": [
                        _weighted_integer_quantile(ends, physical, 0.025),
                        _weighted_integer_quantile(ends, physical, 0.975),
                    ],
                }
                map_interval = [
                    int(candidates[map_index][0]), int(candidates[map_index][1])
                ]
            else:
                entropy = None
                credible = None
                map_interval = None
                maximum = None
            maximum_component_index = int(np.argmax(responsibilities[read_index]))
            component_names = list(model["component_names"])
            molecule_records.append(
                {
                    "molecule_id": list(read.molecule_id),
                    "strand": read.strand,
                    "opportunities": int(
                        read.interval_evidence(envelope.start, envelope.end)[1]
                    ),
                    "family_posterior": family_probability,
                    "standardized_family_posterior_equal_prior": float(
                        standardized_family_posteriors[read_index]
                    ),
                    "family_vs_null_log_bayes_factor": float(
                        family_vs_null_log_bayes_factors[read_index]
                    ),
                    "family_log_predictive_ratio_to_accessible": float(
                        family_log_predictive[read_index]
                    ),
                    "null_log_predictive_ratio_to_accessible": float(
                        null_log_predictive[read_index]
                    ),
                    "mixture_log_likelihood_ratio_to_accessible": float(
                        log_normalizers[read_index]
                    ),
                    "accessible_posterior": float(responsibilities[read_index, 0]),
                    "spatial_null_posterior": float(
                        responsibilities[read_index, -2]
                    ),
                    "diffuse_null_posterior": float(
                        responsibilities[read_index, -1]
                    ),
                    "maximum_posterior_component": component_names[
                        maximum_component_index
                    ],
                    "conditional_map_interval": map_interval,
                    "conditional_map_interval_probability": maximum,
                    "conditional_boundary_credible_envelope_95": credible,
                    "conditional_geometry_entropy_nats": entropy,
                    "molecule_opportunity_projection_class_count": len(
                        {
                            (
                                int(left_indices[read_index, candidate_index]),
                                int(right_indices[read_index, candidate_index]),
                            )
                            for candidate_index in range(len(candidates))
                        }
                    ),
                }
            )
    total_log_likelihood = float(np.sum(log_normalizers))
    total_opportunities = int(
        sum(
            read.interval_evidence(envelope.start, envelope.end)[1]
            for read in retained_reads
        )
    )
    return {
        "schema": "fiberhmm.boundary_marginalized_tf_family_score.v1",
        "status": "frozen_model_score",
        "family_id": str(model["family_id"]),
        "model_structure_id": str(model["model_structure_id"]),
        "model_fit_id": str(model.get("model_fit_id", "")),
        "frozen_parameters": True,
        "interval_evidence_summation_mode": selected_summation_mode,
        "envelope": [int(envelope.start), int(envelope.end)],
        "eligible_molecules": len(retained_reads),
        "eligible_molecules_by_strand": {
            strand: sum(read.strand == strand for read in retained_reads)
            for strand in sorted({read.strand for read in retained_reads})
        },
        "total_opportunities": total_opportunities,
        "family_effective_molecule_support": float(np.sum(family_posteriors)),
        "standardized_family_effective_molecule_support_equal_prior": float(
            np.sum(standardized_family_posteriors)
        ),
        "family_posterior_over_half_molecules": int(
            np.sum(family_posteriors >= 0.5)
        ),
        "family_posterior_over_nine_tenths_molecules": int(
            np.sum(family_posteriors >= 0.9)
        ),
        "standardized_family_posterior_over_half_molecules": int(
            np.sum(standardized_family_posteriors >= 0.5)
        ),
        "standardized_family_posterior_over_nine_tenths_molecules": int(
            np.sum(standardized_family_posteriors >= 0.9)
        ),
        "family_vs_null_log_bayes_factor_nonnegative_molecules": int(
            np.sum(family_vs_null_log_bayes_factors >= 0.0)
        ),
        "family_vs_null_log_bayes_factor_at_least_log_10_molecules": int(
            np.sum(family_vs_null_log_bayes_factors >= math.log(10.0))
        ),
        "median_family_vs_null_log_bayes_factor": float(
            np.median(family_vs_null_log_bayes_factors)
        ),
        "evidence_standardization": (
            "family_vs_conditionally_normalized_A_P0_U_predictive_"
            "log_bayes_factor;_equal_family_null_prior_posterior"
        ),
        "raw_mixture_log_likelihood_ratio": total_log_likelihood,
        "mean_log_likelihood_ratio_per_opportunity": (
            total_log_likelihood / total_opportunities
            if total_opportunities
            else None
        ),
        "molecules": molecule_records,
        "molecule_record_selection": {
            "eligible_molecules": len(retained_reads),
            "emitted_molecules": len(molecule_records),
            "minimum_standardized_family_posterior": (
                molecule_record_minimum_standardized_family_posterior
            ),
            "minimum_family_vs_null_log_bayes_factor": (
                molecule_record_minimum_family_vs_null_log_bayes_factor
            ),
        },
    }


def classify_strand_family_candidates(
    sites: Sequence[SiteTemplate],
    site_models: Mapping[str, Mapping[int, Mapping[str, object]]],
    strands: Sequence[str],
    *,
    minimum_support: int,
    minimum_coverage: Optional[int] = None,
) -> dict:
    """Classify cross-strand family evidence without changing inference.

    The classes intentionally distinguish observation from interpretation.
    In particular, a one-sided family with zero represented opportunities on
    the other strand is reported as opportunity-limited, not accepted as a
    rescue and not rejected as a false family.
    """
    if minimum_support < 1:
        raise ValueError("minimum support must be positive")
    if minimum_coverage is None:
        minimum_coverage = minimum_support
    if minimum_coverage < 1:
        raise ValueError("minimum coverage must be positive")
    if len(strands) != 2:
        raise ValueError("strand family classification requires two strata")

    records = []
    class_counts: Dict[str, int] = {}
    for site_index, site in enumerate(sites):
        by_strand = {}
        supported_strands = []
        under_covered_strands = []
        for strand in strands:
            model = site_models[strand][site_index]
            geometry_support = int(site.support.get(strand, 0))
            site_coverage = int(
                model.get("site_coverage", model.get("coverage", 0))
            )
            canonical_support = int(
                model.get(
                    "canonical_explicit_tf_molecules",
                    min(geometry_support, site_coverage),
                )
            )
            opportunity_molecules = int(
                model.get("opportunity_eligible_molecules", 0)
            )
            if canonical_support >= minimum_support:
                supported_strands.append(strand)
            if site_coverage < minimum_coverage:
                under_covered_strands.append(strand)
            by_strand[strand] = {
                "geometry_source_support": geometry_support,
                "canonical_explicit_tf_molecules": canonical_support,
                "geometry_map_supported": geometry_support >= minimum_support,
                "canonical_family_supported": (
                    canonical_support >= minimum_support
                ),
                "site_coverage": site_coverage,
                "opportunity_eligible_molecules": opportunity_molecules,
                "direct_tf_without_site_opportunities": int(
                    model.get("direct_tf_without_site_opportunities", 0)
                ),
                "explicit_call_fraction": explicit_call_fraction(
                    site, strand, model
                ),
                "opportunity_conditioned_call_fraction": (
                    opportunity_conditioned_call_fraction(
                        site, strand, model
                    )
                ),
                "opportunity_by_segment": model.get(
                    "opportunity_by_segment", {}
                ),
            }

        supported = set(supported_strands)
        if under_covered_strands:
            candidate_class = "under_covered"
            opportunity_limited_strand = None
        elif len(supported) == 2:
            candidate_class = "matched_supported"
            opportunity_limited_strand = None
        elif len(supported) == 1:
            unsupported = next(
                strand for strand in strands if strand not in supported
            )
            opportunity_limited_strand = unsupported
            if by_strand[unsupported]["opportunity_eligible_molecules"] == 0:
                candidate_class = "one_sided_zero_interior_opportunity"
            else:
                candidate_class = "one_sided_with_interior_opportunity"
        else:
            candidate_class = "under_supported"
            opportunity_limited_strand = None
        class_counts[candidate_class] = class_counts.get(candidate_class, 0) + 1
        records.append(
            {
                "site": site.site_id,
                "interval": [site.start, site.end],
                "candidate_class": candidate_class,
                "supported_strands": sorted(supported),
                "opportunity_limited_strand": opportunity_limited_strand,
                "by_strand": by_strand,
                "reporting_only": True,
                "inference_enabled": False,
            }
        )
    return {
        "classification_version": 1,
        "minimum_geometry_support_per_strand": int(minimum_support),
        "minimum_canonical_coverage_per_strand": int(minimum_coverage),
        "reporting_only": True,
        "inference_enabled": False,
        "class_counts": dict(sorted(class_counts.items())),
        "families": records,
    }


def factorized_configuration_prior(
    configurations: Sequence[Configuration],
    global_site_indices: Sequence[int],
    source_site_models: Mapping[int, Mapping[str, object]],
) -> np.ndarray:
    """Compose multi-site priors from short-read-safe source marginals."""
    tf_probabilities = []
    nuc_probabilities = []
    for global_index in global_site_indices:
        weights = source_site_models[global_index]["weights"]
        accessible = float(weights.get("A", 0.0))
        tf = float(weights.get("TF", 0.0))
        denominator = accessible + tf
        tf_probabilities.append(tf / denominator if denominator > 0.0 else 0.5)
        nuc_probabilities.append(float(weights.get("N", 0.0)))
    rho_nuc = float(np.mean(nuc_probabilities)) if nuc_probabilities else 0.0
    prior = np.zeros(len(configurations), dtype=np.float64)
    non_nuc = []
    for configuration_index, configuration in enumerate(configurations):
        if configuration.is_nucleosome:
            prior[configuration_index] = rho_nuc
            continue
        selected = set(configuration.site_indices)
        weight = 1.0
        for site_index, probability in enumerate(tf_probabilities):
            weight *= probability if site_index in selected else (1.0 - probability)
        prior[configuration_index] = weight
        non_nuc.append(configuration_index)
    total = float(np.sum(prior[non_nuc])) if non_nuc else 0.0
    if total > 0.0:
        non_nuc_mass = (
            1.0 - rho_nuc
            if any(c.is_nucleosome for c in configurations)
            else 1.0
        )
        prior[non_nuc] *= non_nuc_mass / total
    prior = np.maximum(prior, 1e-12)
    return prior / np.sum(prior)


def wilson_lower_bound(successes: int, total: int, z: float = 1.96) -> float:
    if total <= 0 or successes <= 0:
        return 0.0
    if successes > total:
        raise ValueError("successes cannot exceed total")
    fraction = successes / total
    z2 = z * z
    center = fraction + z2 / (2.0 * total)
    radius = z * math.sqrt(
        fraction * (1.0 - fraction) / total + z2 / (4.0 * total * total)
    )
    return max(0.0, min(1.0, (center - radius) / (1.0 + z2 / total)))


def _logistic(value: float) -> float:
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-min(value, 745.0)))
    exp_value = math.exp(max(value, -745.0))
    return exp_value / (1.0 + exp_value)


def _group_site_indices(
    indices: Sequence[int], sites: Sequence[SiteTemplate], maximum_gap: int
) -> List[List[int]]:
    ordered = sorted(indices, key=lambda index: (sites[index].start, sites[index].end))
    groups: List[List[int]] = []
    for index in ordered:
        if groups and sites[index].start - sites[groups[-1][-1]].end <= maximum_gap:
            groups[-1].append(index)
        else:
            groups.append([index])
    return groups


def _group_tf_class_loci(
    sites: Sequence[SiteTemplate],
    *,
    maximum_gap: int = 30,
    maximum_span: int = 250,
) -> List[List[int]]:
    """Build bounded local neighborhoods for single/composite TF classes."""
    if maximum_gap < 0 or maximum_span < 1:
        raise ValueError("TF class locus bounds are invalid")
    ordered = sorted(
        range(len(sites)), key=lambda index: (sites[index].start, sites[index].end)
    )
    groups: List[List[int]] = []
    for index in ordered:
        site = sites[index]
        if groups:
            group = groups[-1]
            group_start = min(sites[value].start for value in group)
            group_end = max(sites[value].end for value in group)
            if (
                site.start - group_end <= maximum_gap
                and max(group_end, site.end) - min(group_start, site.start)
                <= maximum_span
            ):
                group.append(index)
                continue
        groups.append([index])
    return groups


def _choose_sites(
    indices: Sequence[int],
    sites: Sequence[SiteTemplate],
    source: str,
    maximum: int,
) -> Tuple[List[int], List[int]]:
    if len(indices) <= maximum:
        return list(indices), []
    ranked = sorted(
        indices,
        key=lambda index: (
            sites[index].support.get(source, 0)
            * (sites[index].local_enrichment_by_strand or {}).get(
                source, sites[index].local_enrichment
            ),
            -sites[index].start,
        ),
        reverse=True,
    )[:maximum]
    selected = sorted(
        ranked, key=lambda index: (sites[index].start, sites[index].end)
    )
    selected_set = set(selected)
    dropped = sorted(
        (index for index in indices if index not in selected_set),
        key=lambda index: (sites[index].start, sites[index].end),
    )
    return selected, dropped


GEOMETRY_NULL_LOG_SCORE = -4.605170185988091


def _geometry_log_score(call: IntervalCall, site: SiteTemplate) -> float:
    start_scale = max(2.0, 1.4826 * float(site.start_mad))
    end_scale = max(2.0, 1.4826 * float(site.end_mad))
    return -0.5 * (
        ((call.start - site.start) / start_scale) ** 2
        + ((call.end - site.end) / end_scale) ** 2
    )


def _geometry_assignment(
    call: IntervalCall,
    sites: Sequence[SiteTemplate],
    selected_index: int,
    *,
    center_radius: int,
) -> Tuple[float, bool, float]:
    candidates = [
        index
        for index, site in enumerate(sites)
        if abs(call.center - site.center) <= center_radius
    ]
    if selected_index not in candidates:
        return 0.0, False, -math.inf
    scores = []
    for index in candidates:
        scores.append(_geometry_log_score(call, sites[index]))
    probabilities = _softmax(
        np.asarray([*scores, GEOMETRY_NULL_LOG_SCORE], dtype=np.float64)
    )
    selected_position = candidates.index(selected_index)
    selected_score = float(scores[selected_position])
    return (
        float(probabilities[selected_position]),
        selected_score > GEOMETRY_NULL_LOG_SCORE,
        selected_score,
    )


def match_geometry_calls(
    calls: Sequence[IntervalCall],
    sites: Sequence[SiteTemplate],
    *,
    center_radius: int,
) -> List[Tuple[IntervalCall, int, bool]]:
    """Jointly map calls to geometry families with one explicit null per call."""
    if not calls or not sites:
        return []
    call_list = list(calls)
    score_matrix = np.full(
        (len(call_list), len(sites) + len(call_list)), -1e12, dtype=np.float64
    )
    candidates: List[List[int]] = []
    for call_index, call in enumerate(call_list):
        nearby = [
            site_index
            for site_index, site in enumerate(sites)
            if abs(call.center - site.center) <= center_radius
        ]
        candidates.append(nearby)
        for site_index in nearby:
            score_matrix[call_index, site_index] = _geometry_log_score(
                call, sites[site_index]
            )
        score_matrix[call_index, len(sites) :] = GEOMETRY_NULL_LOG_SCORE
    rows, columns = linear_sum_assignment(score_matrix, maximize=True)
    assigned_columns = dict(zip(rows.tolist(), columns.tolist()))
    matches = []
    for call_index, nearby in enumerate(candidates):
        if not nearby:
            continue
        selected_column = assigned_columns[call_index]
        if selected_column < len(sites):
            selected = selected_column
            assigned = score_matrix[call_index, selected] > GEOMETRY_NULL_LOG_SCORE
        else:
            selected = max(
                nearby,
                key=lambda site_index: score_matrix[call_index, site_index],
            )
            assigned = False
        matches.append((call_list[call_index], selected, assigned))
    return matches


def geometry_edge_samples_by_site(
    reads: Sequence[ReadEvidence],
    sites: Sequence[SiteTemplate],
    *,
    strand: str,
    call_type: str,
    center_radius: int,
) -> Dict[int, List[IntervalCall]]:
    """Collect one molecule-collapsed source-strand call per geometry family."""
    grouped: Dict[Tuple[int, Tuple[str, str, str]], List[IntervalCall]] = {}
    for read in reads:
        if read.strand != strand:
            continue
        for call, site_index, assigned in match_geometry_calls(
            _calls_for_type(read, call_type),
            sites,
            center_radius=center_radius,
        ):
            if not assigned or not _source_interval_fully_maps(
                read, call, sites[site_index].source_boundary_margin
            ):
                continue
            grouped.setdefault((site_index, read.molecule_id), []).append(call)
    result: Dict[int, List[IntervalCall]] = {index: [] for index in range(len(sites))}
    for (site_index, _molecule), calls in grouped.items():
        result[site_index].append(
            IntervalCall(
                int(round(float(np.median([call.start for call in calls])))),
                int(round(float(np.median([call.end for call in calls])))),
            )
        )
    return result


def edge_log_bayes_factor(
    read: ReadEvidence,
    current: IntervalCall,
    canonical: IntervalCall,
    *,
    call_type: str = "tf",
) -> dict:
    """Compare two edge placements while holding footprint identity fixed.

    ``read.steps`` is log P(protected) / P(accessible). Bases newly included by
    the canonical interval therefore add their step, while bases removed from
    the current interval subtract it. Shared interior bases cancel exactly.
    """
    steps = read.nuc_steps if call_type == "nuc" else read.steps
    if steps is None:
        steps = read.steps
    current_mask = (read.positions >= current.start) & (read.positions < current.end)
    canonical_mask = (read.positions >= canonical.start) & (
        read.positions < canonical.end
    )
    added = canonical_mask & ~current_mask
    removed = current_mask & ~canonical_mask
    added_llr = float(np.sum(steps[added]))
    removed_llr = float(np.sum(steps[removed]))
    log_bf = added_llr - removed_llr
    left_changed = current.start != canonical.start
    right_changed = current.end != canonical.end
    left_mask = (read.positions >= min(current.start, canonical.start)) & (
        read.positions < max(current.start, canonical.start)
    )
    right_mask = (read.positions >= min(current.end, canonical.end)) & (
        read.positions < max(current.end, canonical.end)
    )
    left_sign = 1.0 if canonical.start < current.start else -1.0
    right_sign = 1.0 if canonical.end > current.end else -1.0
    left_llr = float(left_sign * np.sum(steps[left_mask])) if left_changed else 0.0
    right_llr = (
        float(right_sign * np.sum(steps[right_mask])) if right_changed else 0.0
    )
    changed_edge_probabilities = [
        _logistic(value)
        for value, changed in (
            (left_llr, left_changed),
            (right_llr, right_changed),
        )
        if changed
    ]
    return {
        "log_bf_canonical_vs_current": log_bf,
        "canonical_probability_equal_prior": _logistic(log_bf),
        "conservative_edge_probability": min(
            changed_edge_probabilities, default=1.0
        ),
        "changed_opportunities": int(np.sum(added) + np.sum(removed)),
        "changed_hits": int(np.sum(read.hits[added]) + np.sum(read.hits[removed])),
        "added": {
            "llr": added_llr,
            "opportunities": int(np.sum(added)),
            "hits": int(np.sum(read.hits[added])),
        },
        "removed": {
            "llr": removed_llr,
            "opportunities": int(np.sum(removed)),
            "hits": int(np.sum(read.hits[removed])),
        },
        "left_edge": {
            "changed": left_changed,
            "log_bf_canonical_vs_current": left_llr,
            "canonical_probability_equal_prior": (
                _logistic(left_llr) if left_changed else None
            ),
            "opportunities": int(np.sum(left_mask)) if left_changed else 0,
            "hits": int(np.sum(read.hits[left_mask])) if left_changed else 0,
        },
        "right_edge": {
            "changed": right_changed,
            "log_bf_canonical_vs_current": right_llr,
            "canonical_probability_equal_prior": (
                _logistic(right_llr) if right_changed else None
            ),
            "opportunities": int(np.sum(right_mask)) if right_changed else 0,
            "hits": int(np.sum(read.hits[right_mask])) if right_changed else 0,
        },
    }


def _site_edge_reliability(site: SiteTemplate, edge: str) -> float:
    """Return the population reliability for one canonical boundary."""
    value = (
        site.start_geometry_reliability
        if edge == "left"
        else site.end_geometry_reliability
    )
    if value is None:
        value = site.geometry_reliability
    return max(0.0, min(1.0, float(value)))


@dataclass(frozen=True)
class _EdgePopulationStatistics:
    """Reusable sufficient statistics for one opposite-strand edge family."""

    sample_count: int
    start_sum: int
    end_sum: int
    start_median: Optional[float]
    end_median: Optional[float]
    start_mad: float
    end_mad: float
    start_scale: float
    end_scale: float
    correlation: float
    covariance: np.ndarray
    inverse_covariance: np.ndarray


def _prepare_edge_population_statistics(
    source_calls: Sequence[IntervalCall],
) -> _EdgePopulationStatistics:
    """Prepare the depth-independent terms of the edge population model."""
    starts = [int(call.start) for call in source_calls]
    ends = [int(call.end) for call in source_calls]
    sample_count = len(starts)
    start_median = float(np.median(starts)) if starts else None
    end_median = float(np.median(ends)) if ends else None
    start_mad = _mad(starts) if starts else 0.0
    end_mad = _mad(ends) if ends else 0.0
    start_scale = max(2.0, 1.4826 * float(start_mad))
    end_scale = max(2.0, 1.4826 * float(end_mad))

    correlation = 0.0
    if sample_count >= 3:
        sample_matrix = np.asarray(list(zip(starts, ends)), dtype=np.float64)
        centered_start = sample_matrix[:, 0] - np.mean(sample_matrix[:, 0])
        centered_end = sample_matrix[:, 1] - np.mean(sample_matrix[:, 1])
        denominator = float(
            math.sqrt(
                float(centered_start @ centered_start)
                * float(centered_end @ centered_end)
            )
        )
        if denominator > 0.0:
            estimated_correlation = float(
                (centered_start @ centered_end) / denominator
            )
            if math.isfinite(estimated_correlation):
                correlation = max(-0.9, min(0.9, estimated_correlation))
    covariance = np.asarray(
        [
            [
                start_scale**2,
                correlation * start_scale * end_scale,
            ],
            [
                correlation * start_scale * end_scale,
                end_scale**2,
            ],
        ],
        dtype=np.float64,
    )
    return _EdgePopulationStatistics(
        sample_count=sample_count,
        start_sum=sum(starts),
        end_sum=sum(ends),
        start_median=start_median,
        end_median=end_median,
        start_mad=float(start_mad),
        end_mad=float(end_mad),
        start_scale=float(start_scale),
        end_scale=float(end_scale),
        correlation=float(correlation),
        covariance=covariance,
        inverse_covariance=np.linalg.inv(covariance),
    )


def edge_hypothesis_evidence(
    current: IntervalCall,
    canonical: IntervalCall,
    source_calls: Sequence[IntervalCall],
    chemistry: Mapping[str, object],
    *,
    population_statistics: Optional["_EdgePopulationStatistics"] = None,
) -> dict:
    """Score canonical edges against the ordinary edges under equal priors.

    The population term compares the canonical and ordinary edges under one
    regularized plug-in Gaussian predictive score fitted to molecule-collapsed
    source edges. Source depth therefore reduces uncertainty in the location
    without multiplying the same prior evidence once per source molecule. The
    predictive geometry term is combined with the changed-base chemistry log
    Bayes factor on the target molecule. Unchanged edges are shared by both
    hypotheses and receive confidence one without contributing evidence.
    """
    statistics = (
        _prepare_edge_population_statistics(source_calls)
        if population_statistics is None
        else population_statistics
    )
    values = {}
    for edge, current_value, canonical_value, median, mad, scale in (
        (
            "left",
            current.start,
            canonical.start,
            statistics.start_median,
            statistics.start_mad,
            statistics.start_scale,
        ),
        (
            "right",
            current.end,
            canonical.end,
            statistics.end_median,
            statistics.end_mad,
            statistics.end_scale,
        ),
    ):
        changed = current_value != canonical_value
        population_log_bf = 0.0
        predictive_scale = float(scale)
        if statistics.sample_count > 0:
            predictive_scale *= math.sqrt(
                1.0 + 1.0 / statistics.sample_count
            )
        if changed and median is not None and statistics.sample_count > 0:
            population_log_bf = 0.5 * float(
                ((current_value - median) / predictive_scale) ** 2
                - ((canonical_value - median) / predictive_scale) ** 2
            )
        chemistry_value = chemistry.get(f"{edge}_edge", {})
        chemistry_log_bf = (
            float(chemistry_value.get("log_bf_canonical_vs_current", 0.0))
            if isinstance(chemistry_value, Mapping)
            else 0.0
        )
        combined_log_bf = (
            chemistry_log_bf + population_log_bf if changed else 0.0
        )
        probability = _logistic(combined_log_bf) if changed else 1.0
        values[edge] = {
            "changed": changed,
            "current": int(current_value),
            "canonical": int(canonical_value),
            "opposite_strand_molecules": statistics.sample_count,
            "opposite_strand_median": median,
            "opposite_strand_mad": float(mad),
            "population_scale": float(scale),
            "population_predictive_scale": float(predictive_scale),
            "population_log_bf": float(population_log_bf),
            "chemistry_log_bf": float(chemistry_log_bf),
            "combined_log_bf": float(combined_log_bf),
            "alternative_probability": float(probability),
        }
    baseline_center = np.asarray([current.start, current.end], dtype=np.float64)
    canonical_center = np.asarray(
        [canonical.start, canonical.end], dtype=np.float64
    )
    predictive_multiplier = (
        1.0 + 1.0 / statistics.sample_count
        if statistics.sample_count > 0
        else 1.0
    )
    predictive_covariance = statistics.covariance * predictive_multiplier
    joint_population_log_bf = 0.0
    if (
        statistics.sample_count > 0
        and statistics.start_median is not None
        and statistics.end_median is not None
    ):
        population_center = np.asarray(
            [statistics.start_median, statistics.end_median],
            dtype=np.float64,
        )
        predictive_inverse = (
            statistics.inverse_covariance / predictive_multiplier
        )
        baseline_residual = baseline_center - population_center
        canonical_residual = canonical_center - population_center
        joint_population_log_bf = 0.5 * float(
            baseline_residual @ predictive_inverse @ baseline_residual
            - canonical_residual @ predictive_inverse @ canonical_residual
        )
    joint_chemistry_log_bf = float(
        chemistry.get("log_bf_canonical_vs_current", 0.0)
    )
    joint_log_bf = joint_population_log_bf + joint_chemistry_log_bf
    joint_probability = _logistic(joint_log_bf)
    return {
        "model": (
            "opposite_strand_regularized_gaussian_predictive_score_plus_"
            "target_chemistry"
        ),
        "prior_odds": 1.0,
        "opposite_strand_molecules": statistics.sample_count,
        "population_covariance": statistics.covariance.tolist(),
        "population_predictive_covariance": predictive_covariance.tolist(),
        "population_correlation": float(statistics.correlation),
        "joint_population_log_bf": float(joint_population_log_bf),
        "joint_chemistry_log_bf": float(joint_chemistry_log_bf),
        "joint_log_bf": float(joint_log_bf),
        "alternative_probability": float(joint_probability),
        "baseline_probability": float(1.0 - joint_probability),
        "left": values["left"],
        "right": values["right"],
    }


def _edge_conflicts(
    read: ReadEvidence,
    current: IntervalCall,
    canonical: IntervalCall,
    call_type: str,
) -> List[dict]:
    conflicts = []
    for other_type in ("tf", "nuc"):
        for other in _all_calls_for_type(read, other_type):
            if other_type == call_type and other is current:
                continue
            canonical_overlap = (
                canonical.start < other.end and other.start < canonical.end
            )
            current_overlap = current.start < other.end and other.start < current.end
            if canonical_overlap and not current_overlap:
                conflicts.append(
                    {
                        "call_type": other_type,
                        "interval": [other.start, other.end],
                    }
                )
    return conflicts


def _site_spanning_coverage(
    reads: Sequence[ReadEvidence], strand: str, start: int, end: int
) -> int:
    return len(
        {
            read.molecule_id
            for read in reads
            if read.strand == strand and read.fully_maps(start, end)
        }
    )


def analyze_shared_geometry(
    reads: Sequence[ReadEvidence],
    sites: Sequence[SiteTemplate],
    site_models: Optional[Mapping[str, Mapping[int, Mapping[str, object]]]] = None,
    *,
    population_reads: Optional[Sequence[ReadEvidence]] = None,
    call_type: str = "tf",
    center_radius: int = 10,
    minimum_geometry_support: int = 3,
    record_callback: Optional[Callable[[ReadEvidence, dict], None]] = None,
    retain_harmonizations: bool = True,
    stream_order: bool = False,
    evaluation_provenance: Optional[Mapping[str, object]] = None,
) -> dict:
    """Normalize accepted calls to a shared geometry without changing occupancy.

    ``prior_only_updates``, ``chemistry_opposed_updates``, and
    ``extreme_edge_shifts`` are audit counters, not hidden rejection gates.
    Acceptance remains graded in the emitted edge probabilities so downstream
    tools can choose an explicit q1/q2 threshold.
    """
    if call_type not in {"tf", "nuc"}:
        raise ValueError("call type must be tf or nuc")
    if any(site.call_type != call_type for site in sites):
        raise ValueError("geometry sites do not match the requested call type")
    strands = sorted({read.strand for read in reads})
    if len(strands) != 2:
        return {"harmonizations": [], "counts": {}}
    population_read_list = list(
        reads if population_reads is None else population_reads
    )
    spanning_coverages = {
        (strand, site_index): _site_spanning_coverage(
            population_read_list, strand, site.start, site.end
        )
        for strand in strands
        for site_index, site in enumerate(sites)
    }
    center_coverages = {
        (strand, site_index): _site_spanning_coverage(
            population_read_list, strand, site.center, site.center + 1
        )
        for strand in strands
        for site_index, site in enumerate(sites)
    }
    source_edge_statistics = {}
    for strand in strands:
        samples_by_site = geometry_edge_samples_by_site(
            population_read_list,
            sites,
            strand=strand,
            call_type=call_type,
            center_radius=center_radius,
        )
        source_edge_statistics[strand] = {
            site_index: _prepare_edge_population_statistics(samples)
            for site_index, samples in samples_by_site.items()
        }
    counts = {
        "geometry_eligible_calls": sum(
            call.geometry_eligible
            for read in reads
            for call in _all_calls_for_type(read, call_type)
        ),
        "topology_only_obstacles": sum(
            not call.geometry_eligible
            for read in reads
            for call in _all_calls_for_type(read, call_type)
        ),
        "matched_existing_calls": 0,
        "shared_geometry_calls": 0,
        "already_canonical": 0,
        "edge_updates": 0,
        "unassigned_retained": 0,
        "topology_conflicts_retained": 0,
        "canonical_interval_not_spanned": 0,
        "insufficient_shared_geometry": 0,
        "insufficient_opposite_geometry_samples": 0,
        "prior_only_updates": 0,
        "chemistry_opposed_updates": 0,
        "extreme_edge_shifts": 0,
        "edge_population_models_prepared": sum(
            len(by_site) for by_site in source_edge_statistics.values()
        ),
        "edge_population_source_samples_prepared": sum(
            statistics.sample_count
            for by_site in source_edge_statistics.values()
            for statistics in by_site.values()
        ),
        "edge_population_sufficient_stat_evaluations": 0,
        "edge_population_naive_sample_evaluations_avoided": 0,
    }
    harmonizations = []
    target_reads = (
        ((read.strand, read) for read in reads)
        if stream_order
        else (
            (target, read)
            for target in strands
            for read in reads
            if read.strand == target
        )
    )
    for target, read in target_reads:
        source = strands[1] if target == strands[0] else strands[0]
        for call, site_index, jointly_assigned in match_geometry_calls(
            _calls_for_type(read, call_type), sites, center_radius=center_radius
        ):
                counts["matched_existing_calls"] += 1
                site = sites[site_index]
                shared = all(
                    site.support.get(strand, 0) >= minimum_geometry_support
                    for strand in strands
                )
                if not shared:
                    counts["insufficient_shared_geometry"] += 1
                    continue
                population_statistics = source_edge_statistics[source][site_index]
                counts["shared_geometry_calls"] += 1
                canonical = IntervalCall(site.start, site.end)
                assignment_probability, individually_assigned, assignment_log_score = (
                    _geometry_assignment(
                        call,
                        sites,
                        site_index,
                        center_radius=center_radius,
                    )
                )
                assigned = jointly_assigned and individually_assigned
                status = "edge_update"
                conflicts = _edge_conflicts(read, call, canonical, call_type)
                if (call.start, call.end) == (canonical.start, canonical.end):
                    status = "already_canonical"
                    counts["already_canonical"] += 1
                elif not assigned:
                    status = "unassigned_retained"
                    counts["unassigned_retained"] += 1
                elif not read.fully_maps(canonical.start, canonical.end):
                    status = "canonical_interval_not_spanned"
                    counts["canonical_interval_not_spanned"] += 1
                elif conflicts:
                    status = "topology_conflict_retained"
                    counts["topology_conflicts_retained"] += 1
                elif population_statistics.sample_count < minimum_geometry_support:
                    status = "insufficient_opposite_geometry_samples"
                    counts["insufficient_opposite_geometry_samples"] += 1
                else:
                    counts["edge_updates"] += 1
                edge_evidence = edge_log_bayes_factor(
                    read, call, canonical, call_type=call_type
                )
                edge_hypothesis = edge_hypothesis_evidence(
                    call,
                    canonical,
                    (),
                    edge_evidence,
                    population_statistics=population_statistics,
                )
                if (call.start, call.end) != (
                    canonical.start,
                    canonical.end,
                ):
                    conditional_probability = float(
                        edge_hypothesis["alternative_probability"]
                    )
                    edge_hypothesis[
                        "conditional_alternative_probability_given_geometry_family"
                    ] = conditional_probability
                    edge_hypothesis["geometry_family_assignment_probability"] = (
                        float(assignment_probability)
                    )
                    edge_hypothesis["alternative_probability"] = float(
                        assignment_probability * conditional_probability
                    )
                    edge_hypothesis["baseline_probability"] = float(
                        assignment_probability * (1.0 - conditional_probability)
                    )
                    edge_hypothesis["unresolved_geometry_probability"] = float(
                        1.0 - assignment_probability
                    )
                    for edge_name in ("left", "right"):
                        edge_hypothesis_edge = edge_hypothesis[edge_name]
                        if not edge_hypothesis_edge["changed"]:
                            continue
                        conditional_edge_probability = float(
                            edge_hypothesis_edge["alternative_probability"]
                        )
                        edge_hypothesis_edge[
                            "conditional_alternative_probability_given_geometry_family"
                        ] = conditional_edge_probability
                        edge_hypothesis_edge[
                            "geometry_family_assignment_probability"
                        ] = float(assignment_probability)
                        edge_hypothesis_edge["alternative_probability"] = float(
                            assignment_probability * conditional_edge_probability
                        )
                        edge_hypothesis_edge[
                            "unresolved_geometry_probability"
                        ] = float(1.0 - assignment_probability)
                counts["edge_population_sufficient_stat_evaluations"] += 1
                counts[
                    "edge_population_naive_sample_evaluations_avoided"
                ] += population_statistics.sample_count
                if status == "edge_update":
                    counts["prior_only_updates"] += int(
                        edge_evidence["changed_opportunities"] == 0
                    )
                    counts["chemistry_opposed_updates"] += int(
                        edge_evidence["conservative_edge_probability"] < 0.5
                    )
                    counts["extreme_edge_shifts"] += int(
                        max(
                            abs(call.start - canonical.start),
                            abs(call.end - canonical.end),
                        )
                        > 2 * center_radius
                    )
                materialized_edge_confidence = []
                for edge_name in ("left", "right"):
                    hypothesis_edge = edge_hypothesis[edge_name]
                    evidence_edge = edge_evidence[f"{edge_name}_edge"]
                    if not hypothesis_edge["changed"]:
                        quality_probability = 1.0
                    elif int(evidence_edge["opportunities"]) == 0:
                        # A population family can nominate the edge, but it
                        # cannot establish this molecule's exact boundary when
                        # the changed span has no represented assay base.
                        quality_probability = 0.0
                    else:
                        quality_probability = float(
                            hypothesis_edge["alternative_probability"]
                        )
                    materialized_edge_confidence.append(quality_probability)
                geometry_support = int(site.support.get(source, 0))
                spanning_coverage = int(
                    spanning_coverages[(source, site_index)]
                )
                source_model = (
                    site_models.get(source, {}).get(site_index)
                    if site_models is not None
                    else None
                )
                model_spanning_coverage = int(
                    source_model.get("site_coverage", spanning_coverage)
                    if source_model is not None
                    else spanning_coverage
                )
                center_coverage = int(
                    source_model.get(
                        "center_coverage",
                        center_coverages[(source, site_index)],
                    )
                    if source_model is not None
                    else center_coverages[(source, site_index)]
                )
                canonical_support = int(
                    source_model.get(
                        "canonical_explicit_tf_molecules",
                        min(geometry_support, model_spanning_coverage),
                    )
                    if source_model is not None
                    else min(
                        population_statistics.sample_count,
                        model_spanning_coverage,
                    )
                )
                if canonical_support > model_spanning_coverage:
                    raise ValueError(
                        "canonical edge source support exceeds spanning coverage"
                    )
                if call_type == "tf" and source_model is not None:
                    weights = source_model["weights"]
                    accessible = float(weights.get("A", 0.0))
                    occupied = float(weights.get("TF", 0.0))
                    denominator = accessible + occupied
                    population_probability = (
                        occupied / denominator if denominator else 0.5
                    )
                else:
                    population_probability = (
                        (canonical_support + 0.5)
                        / (model_spanning_coverage + 1.0)
                        if model_spanning_coverage
                        else 0.5
                    )
                enrichment = float(
                    (site.local_enrichment_by_strand or {}).get(
                        source, site.local_enrichment
                    )
                )
                support_reliability = wilson_lower_bound(
                    canonical_support, model_spanning_coverage
                )
                decision_id = (
                    f"{read.name}@{_alignment_identity_token(read)}:"
                    f"{call.start}-{call.end}:ord{call.ordinal}:"
                    f"{call_type.upper()}->{site.site_id}:{target}<->{source}"
                )
                harmonization = {
                        "decision_id": decision_id,
                        "read": read.name,
                        "library_id": read.library_id,
                        "alignment": {
                            "reference_start": read.ref_start,
                            "flag": read.alignment_flag,
                            "cigar": read.cigar,
                            "record_sha256": read.record_sha256,
                            "occurrence": read.alignment_occurrence,
                        },
                        "target_strand": target,
                        "opposite_strand": source,
                        "call_type": call_type,
                        "site": site.site_id,
                        "status": status,
                        "current_interval": [call.start, call.end],
                        "current_molecular_interval": (
                            [call.molecular_start, call.molecular_length]
                            if call.molecular_start is not None
                            and call.molecular_length is not None
                            else None
                        ),
                        "current_annotation_ordinal": call.ordinal,
                        "canonical_interval": [canonical.start, canonical.end],
                        "topology_conflicts": conflicts,
                        "assignment_probability": assignment_probability,
                        "assignment_log_score": assignment_log_score,
                        "assignment_null_log_score": GEOMETRY_NULL_LOG_SCORE,
                        "assignment_beats_null": assigned,
                        "maximum_edge_shift": max(
                            abs(call.start - canonical.start),
                            abs(call.end - canonical.end),
                        ),
                        "extreme_edge_shift": max(
                            abs(call.start - canonical.start),
                            abs(call.end - canonical.end),
                        )
                        > 2 * center_radius,
                        "molecule_probability": edge_evidence[
                            "conservative_edge_probability"
                        ],
                        "population_probability": population_probability,
                        "geometry_reliability": float(site.geometry_reliability),
                        "edge_hypothesis": edge_hypothesis,
                        "materialized_edge_confidence": materialized_edge_confidence,
                        "target_edge_evidence": edge_evidence,
                        "source_evidence": {
                            "explicit_call_support": canonical_support,
                            "geometry_source_support": geometry_support,
                            "geometry_minus_canonical_support": (
                                geometry_support - canonical_support
                            ),
                            "explicit_call_fraction": (
                                canonical_support / model_spanning_coverage
                            )
                            if model_spanning_coverage
                            else 0.0,
                            "support_reliability": support_reliability,
                            "center_coverage": center_coverage,
                            "spanning_coverage": model_spanning_coverage,
                            "local_enrichment": enrichment,
                        },
                        "evaluation_provenance": dict(
                            evaluation_provenance or {}
                        ),
                    }
                if record_callback is not None:
                    record_callback(read, harmonization)
                if retain_harmonizations:
                    harmonizations.append(harmonization)
    if retain_harmonizations:
        harmonizations.sort(
            key=lambda value: (
                value["library_id"] or "",
                value["read"],
                value["current_interval"],
            )
        )
        harmonizations = _deduplicate_decisions(harmonizations)
    return {
        "call_type": call_type,
        "sites": [asdict(site) for site in sites],
        "harmonizations": harmonizations,
        "counts": counts,
        "topology_only_obstacles_by_strand": {
            strand: sum(
                not call.geometry_eligible
                for read in reads
                if read.strand == strand
                for call in _all_calls_for_type(read, call_type)
            )
            for strand in strands
        },
    }


def resolve_joint_edge_topology(*edge_results: dict) -> None:
    """Resolve edge collisions in TF-then-nucleosome stage order.

    Same-layer collisions remain symmetric. For a newly overlapping TF/nuc
    pair, retain the TF consensus proposal and reject the nuc proposal: the
    latter is the downstream reconciliation layer and cannot veto its input.
    """
    grouped: Dict[Tuple[object, ...], List[dict]] = {}
    for result in edge_results:
        for decision in result.get("harmonizations", []):
            if decision.get("status") != "edge_update":
                continue
            alignment = decision.get("alignment", {})
            key = (
                decision.get("library_id"),
                decision.get("read"),
                alignment.get("reference_start"),
                alignment.get("flag"),
                alignment.get("cigar"),
                alignment.get("record_sha256"),
                alignment.get("occurrence"),
            )
            grouped.setdefault(key, []).append(decision)
    rejected_ids = set()
    for decisions in grouped.values():
        for left_index, left in enumerate(decisions):
            for right in decisions[left_index + 1 :]:
                left_current = IntervalCall(*left["current_interval"])
                right_current = IntervalCall(*right["current_interval"])
                left_canonical = IntervalCall(*left["canonical_interval"])
                right_canonical = IntervalCall(*right["canonical_interval"])
                old_overlap = (
                    left_current.start < right_current.end
                    and right_current.start < left_current.end
                )
                new_overlap = (
                    left_canonical.start < right_canonical.end
                    and right_canonical.start < left_canonical.end
                )
                order_inverted = (
                    left["call_type"] == right["call_type"]
                    and (left_current.start < right_current.start)
                    != (left_canonical.start < right_canonical.start)
                )
                if (new_overlap and not old_overlap) or order_inverted:
                    call_types = {left["call_type"], right["call_type"]}
                    if call_types == {"tf", "nuc"}:
                        rejected_ids.add(id(
                            left if left["call_type"] == "nuc" else right
                        ))
                    else:
                        rejected_ids.update((id(left), id(right)))
    for result in edge_results:
        counts = result.get("counts", {})
        rejected = 0
        for decision in result.get("harmonizations", []):
            if id(decision) not in rejected_ids:
                continue
            decision["status"] = "joint_topology_conflict_retained"
            rejected += 1
        accepted = [
            decision
            for decision in result.get("harmonizations", [])
            if decision.get("status") == "edge_update"
        ]
        counts["edge_updates"] = len(accepted)
        counts["joint_topology_conflicts_retained"] = rejected
        counts["prior_only_updates"] = sum(
            decision["target_edge_evidence"]["changed_opportunities"] == 0
            for decision in accepted
        )
        counts["chemistry_opposed_updates"] = sum(
            decision["molecule_probability"] < 0.5 for decision in accepted
        )
        counts["extreme_edge_shifts"] = sum(
            decision["extreme_edge_shift"] for decision in accepted
        )


def resolve_rescue_edge_topology(
    decisions: Sequence[dict], *edge_results: dict
) -> None:
    """Retain baseline edges when a new edge would collide with a TF rescue."""
    rescues: Dict[Tuple[object, ...], List[Tuple[int, int]]] = {}
    for decision in decisions:
        alignment = decision.get("alignment", {})
        key = (
            decision.get("library_id"),
            decision.get("read"),
            alignment.get("reference_start"),
            alignment.get("flag"),
            alignment.get("cigar"),
            alignment.get("record_sha256"),
            alignment.get("occurrence"),
        )
        rescues.setdefault(key, []).extend(
            tuple(interval) for interval in decision.get("proposed_site_intervals", [])
        )
    for result in edge_results:
        rejected = 0
        for decision in result.get("harmonizations", []):
            if decision.get("status") != "edge_update":
                continue
            alignment = decision.get("alignment", {})
            key = (
                decision.get("library_id"),
                decision.get("read"),
                alignment.get("reference_start"),
                alignment.get("flag"),
                alignment.get("cigar"),
                alignment.get("record_sha256"),
                alignment.get("occurrence"),
            )
            current = tuple(decision["current_interval"])
            canonical = tuple(decision["canonical_interval"])
            conflict = any(
                canonical[0] < rescue[1]
                and rescue[0] < canonical[1]
                and not (current[0] < rescue[1] and rescue[0] < current[1])
                for rescue in rescues.get(key, [])
            )
            if conflict:
                decision["status"] = "rescue_topology_conflict_retained"
                rejected += 1
        accepted = [
            decision
            for decision in result.get("harmonizations", [])
            if decision.get("status") == "edge_update"
        ]
        counts = result.get("counts", {})
        counts["edge_updates"] = len(accepted)
        counts["rescue_topology_conflicts_retained"] = rejected
        counts["prior_only_updates"] = sum(
            decision["target_edge_evidence"]["changed_opportunities"] == 0
            for decision in accepted
        )
        counts["chemistry_opposed_updates"] = sum(
            decision["molecule_probability"] < 0.5 for decision in accepted
        )
        counts["extreme_edge_shifts"] = sum(
            decision["extreme_edge_shift"] for decision in accepted
        )


def _score_decision(
    read: ReadEvidence,
    call: IntervalCall,
    relevant_indices: Sequence[int],
    *,
    sites: Sequence[SiteTemplate],
    target_sites: Sequence[SiteTemplate],
    source: str,
    target: str,
    source_models: Mapping[int, Mapping[str, object]],
    source_reads: Sequence[ReadEvidence],
    class_model_cache: Dict[Tuple[str, Tuple[int, ...]], dict],
    center_radius: int,
    class_model_pseudocount: float,
    strong_posterior: float,
    review_posterior: float,
    maximum_sites: int,
) -> Optional[dict]:
    preliminary_evidence = {
        index: read.interval_evidence(
            max(call.start, target_sites[index].start),
            min(call.end, target_sites[index].end),
        )
        for index in relevant_indices
    }
    locally_supported_indices = [
        index
        for index in relevant_indices
        if preliminary_evidence[index][1] >= 1
        and preliminary_evidence[index][0] > 0.0
    ]
    if not locally_supported_indices:
        return None
    locally_supported_set = set(locally_supported_indices)
    selected_indices, dropped_indices = _choose_sites(
        locally_supported_indices, sites, source, maximum_sites
    )
    templates_truncated = bool(dropped_indices)
    local_sites = [target_sites[index] for index in selected_indices]
    configurations = enumerate_configurations(local_sites, include_nucleosome=False)
    factorized_prior = factorized_configuration_prior(
        configurations, selected_indices, source_models
    )
    source_local_sites = [sites[index] for index in selected_indices]
    source_configurations = enumerate_configurations(
        source_local_sites, include_nucleosome=False
    )
    if [
        configuration.site_indices for configuration in source_configurations
    ] != [configuration.site_indices for configuration in configurations]:
        raise ValueError(
            "source and target TF class configuration topologies differ"
        )
    class_model_key = (source, tuple(selected_indices))
    class_model = class_model_cache.get(class_model_key)
    if class_model is None:
        class_model = fit_tf_configuration_class_model(
            source_reads,
            source_local_sites,
            source_configurations,
            center_radius=center_radius,
            pseudocount=class_model_pseudocount,
            source_stratum=source,
        )
        class_model_cache[class_model_key] = class_model
    prior = np.asarray(
        class_model["configuration_probabilities"], dtype=np.float64
    )
    values, site_evidence = configuration_log_likelihoods(
        read, local_sites, configurations, call
    )
    log_scores = np.log(prior) + values
    posterior = _softmax(log_scores)
    current_index = next(
        index for index, configuration in enumerate(configurations)
        if configuration.name == "A"
    )
    supported_tf_indices = [
        index
        for index, configuration in enumerate(configurations)
        if configuration.site_indices
        and all(
            site_evidence[site_index][1] >= 1
            and site_evidence[site_index][0] > 0.0
            for site_index in configuration.site_indices
        )
    ]
    if not supported_tf_indices:
        return None
    tf_mass = float(sum(posterior[index] for index in supported_tf_indices))
    best_index = max(
        supported_tf_indices, key=lambda index: float(log_scores[index])
    )
    best = configurations[best_index]
    selected_mass = float(posterior[best_index])
    current_mass = float(posterior[current_index])
    log_posterior_odds_vs_current = float(
        math.log(float(prior[best_index]))
        + values[best_index]
        - math.log(float(prior[current_index]))
        - values[current_index]
    )
    pairwise_sr_probability = _logistic(log_posterior_odds_vs_current)
    pairwise_current_probability = 1.0 - pairwise_sr_probability
    supported_log_mass = float(
        _logsumexp(np.asarray([log_scores[index] for index in supported_tf_indices]))
    )
    supported_tf_probability = _logistic(
        supported_log_mass - float(log_scores[current_index])
    )
    configuration_probability = float(
        math.exp(float(log_scores[best_index]) - supported_log_mass)
    )
    conditional_selected_configuration_probability = float(
        supported_tf_probability * configuration_probability
    )
    conditional_accessible_action_probability = float(
        1.0 - supported_tf_probability
    )
    conditional_other_supported_configuration_probability = float(
        supported_tf_probability * (1.0 - configuration_probability)
    )
    action_set_complete = not templates_truncated
    selected_configuration_probability = (
        conditional_selected_configuration_probability
        if action_set_complete
        else 0.0
    )
    accessible_action_probability = (
        conditional_accessible_action_probability if action_set_complete else 0.0
    )
    other_supported_configuration_probability = (
        conditional_other_supported_configuration_probability
        if action_set_complete
        else 0.0
    )
    unresolved_action_set_probability = 0.0 if action_set_complete else 1.0
    tf_prior_mass = float(sum(prior[index] for index in supported_tf_indices))
    selected_prior = float(prior[best_index])
    selected_factorized_prior = float(factorized_prior[best_index])
    current_prior = float(prior[current_index])
    current_factorized_prior = float(factorized_prior[current_index])
    population_probability = selected_prior / (selected_prior + current_prior)
    supported_tf_population_probability = tf_prior_mass / (
        tf_prior_mass + current_prior
    )
    log_bf = float(values[best_index] - values[current_index])
    source_evidence = []
    reliability_values = []
    geometry_reliability_values = []
    for local_index in best.site_indices:
        global_index = selected_indices[local_index]
        site = sites[global_index]
        model = source_models[global_index]
        geometry_support = int(site.support.get(source, 0))
        spanning_coverage = int(
            model.get("site_coverage", model["coverage"])
        )
        center_coverage = int(model.get("center_coverage", 0))
        canonical_support = int(
            model.get(
                "canonical_explicit_tf_molecules",
                min(geometry_support, spanning_coverage),
            )
        )
        if canonical_support > spanning_coverage:
            raise ValueError(
                "canonical source support exceeds its spanning coverage"
            )
        enrichment = float(
            (site.local_enrichment_by_strand or {}).get(
                source, site.local_enrichment
            )
        )
        support_reliability = wilson_lower_bound(
            canonical_support, spanning_coverage
        )
        focal_reliability = max(0.0, min(1.0, 1.0 - 1.0 / max(1.0, enrichment)))
        reliability_values.append(min(support_reliability, focal_reliability))
        geometry_reliability_values.append(float(site.geometry_reliability))
        source_evidence.append(
            {
                "site": site.site_id,
                "source_strand": source,
                "explicit_call_support": canonical_support,
                "geometry_source_support": geometry_support,
                "geometry_minus_canonical_support": (
                    geometry_support - canonical_support
                ),
                "explicit_call_fraction": explicit_call_fraction(
                    site, source, model
                ),
                "opportunity_conditioned_call_fraction": (
                    opportunity_conditioned_call_fraction(
                        site, source, model
                    )
                ),
                "center_coverage": center_coverage,
                "spanning_coverage": spanning_coverage,
                "local_enrichment": enrichment,
                "state_model": model,
                "support_reliability": support_reliability,
                "canonical_geometry_reliability": site.geometry_reliability,
                "canonical_left_edge_reliability": _site_edge_reliability(
                    site, "left"
                ),
                "canonical_right_edge_reliability": _site_edge_reliability(
                    site, "right"
                ),
            }
        )
    tier = (
        "strong"
        if selected_configuration_probability >= strong_posterior
        else "review"
        if selected_configuration_probability >= review_posterior
        else "retain_current"
    )
    decision_id = (
        f"{read.name}@{_alignment_identity_token(read)}:"
        f"{call.start}-{call.end}:ord{call.ordinal}:A->{best.name}:"
        f"{target}<-{source}"
    )
    return {
        "decision_id": decision_id,
        "read": read.name,
        "library_id": read.library_id,
        "alignment": {
            "reference_start": read.ref_start,
            "flag": read.alignment_flag,
            "cigar": read.cigar,
            "record_sha256": read.record_sha256,
            "occurrence": read.alignment_occurrence,
        },
        "target_strand": target,
        "source_prior_strand": source,
        "proposal_tier": tier,
        "current": "A",
        "current_annotation": "msp",
        "current_interval": [call.start, call.end],
        "current_molecular_interval": (
            [call.molecular_start, call.molecular_length]
            if call.molecular_start is not None
            and call.molecular_length is not None
            else None
        ),
        "current_annotation_ordinal": call.ordinal,
        "proposed": best.name,
        "proposed_site_intervals": [
            [local_sites[index].start, local_sites[index].end]
            for index in best.site_indices
        ],
        "proposed_site_edge_confidence": [
            [
                _site_edge_reliability(local_sites[index], "left"),
                _site_edge_reliability(local_sites[index], "right"),
            ]
            for index in best.site_indices
        ],
        "posterior": selected_configuration_probability,
        "sr_hypothesis_probability": selected_configuration_probability,
        "current_posterior": accessible_action_probability,
        "baseline_hypothesis_probability": accessible_action_probability,
        "selected_configuration_probability_vs_accessible_and_supported_tf": (
            selected_configuration_probability
        ),
        "accessible_probability_within_supported_action_set": (
            accessible_action_probability
        ),
        "other_supported_tf_configuration_probability": (
            other_supported_configuration_probability
        ),
        "unresolved_action_set_probability": (
            unresolved_action_set_probability
        ),
        "q0_action_set_complete": action_set_complete,
        "conditional_selected_configuration_probability_within_considered_action_set": (
            conditional_selected_configuration_probability
        ),
        "conditional_accessible_probability_within_considered_action_set": (
            conditional_accessible_action_probability
        ),
        "conditional_other_supported_tf_probability_within_considered_action_set": (
            conditional_other_supported_configuration_probability
        ),
        "pairwise_selected_configuration_probability_vs_accessible": (
            pairwise_sr_probability
        ),
        "pairwise_accessible_probability_vs_selected_configuration": (
            pairwise_current_probability
        ),
        "supported_tf_probability_vs_accessible": supported_tf_probability,
        "best_configuration_posterior_given_tf": configuration_probability,
        "molecule_probability": _logistic(log_bf),
        "population_probability": population_probability,
        "supported_tf_population_probability": (
            supported_tf_population_probability
        ),
        "source_tf_class_model_id": class_model["model_id"],
        "selected_tf_class_id": class_model["classes"][best_index][
            "class_id"
        ],
        "selected_tf_class_anchored_support": class_model["classes"][
            best_index
        ]["anchored_molecule_support"],
        "source_tf_class_model_modeled_molecules": class_model[
            "modeled_molecules"
        ],
        "source_support_reliability": min(reliability_values, default=0.0),
        "canonical_geometry_reliability": min(
            geometry_reliability_values, default=0.0
        ),
        "log_bf_vs_current": log_bf,
        "log_posterior_odds_vs_current": log_posterior_odds_vs_current,
        "proposed_log_likelihood": float(values[best_index]),
        "current_log_likelihood": float(values[current_index]),
        "proposed_prior": selected_prior,
        "current_prior": current_prior,
        "proposed_factorized_site_prior_diagnostic": (
            selected_factorized_prior
        ),
        "current_factorized_site_prior_diagnostic": current_factorized_prior,
        "raw_selected_configuration_posterior_mass": selected_mass,
        "raw_supported_tf_posterior_mass": tf_mass,
        "raw_current_posterior_mass": current_mass,
        "templates_considered": len(local_sites),
        "locally_supported_templates_before_cap": len(
            locally_supported_indices
        ),
        "locally_unsupported_template_site_ids": [
            sites[index].site_id
            for index in relevant_indices
            if index not in locally_supported_set
        ],
        "templates_truncated": templates_truncated,
        "dropped_template_site_ids": [
            sites[index].site_id for index in dropped_indices
        ],
        "configuration_count": len(configurations),
        "site_evidence": [
            {
                "site": local_sites[index].site_id,
                "llr": float(site_evidence[index][0]),
                "opportunities": int(site_evidence[index][1]),
                "hits": int(site_evidence[index][2]),
            }
            for index in best.site_indices
        ],
        "source_prior_evidence": source_evidence,
    }


def analyze_strand_rescue(
    reads: Sequence[ReadEvidence],
    sites: Sequence[SiteTemplate],
    *,
    target_reads: Optional[Sequence[ReadEvidence]] = None,
    nuc_sites: Sequence[SiteTemplate] = (),
    target_sites: Optional[Sequence[SiteTemplate]] = None,
    min_source_support: int = 10,
    min_source_local_enrichment: float = 1.5,
    maximum_sites_per_decision: int = 8,
    accessible_site_gap: int = DEFAULT_ACCESSIBLE_SITE_GAP,
    center_radius: int = 10,
    tf_edge_compatibility: int = 12,
    tf_class_pseudocount: float = 0.5,
    tf_class_locus_gap: int = 30,
    tf_class_max_span: int = 250,
    tf_class_max_sites: int = 10,
    minimum_geometry_support: int = 3,
    nuc_center_radius: int = 25,
    nuc_minimum_geometry_support: Optional[int] = None,
    strong_posterior: float = 0.95,
    review_posterior: float = 0.5,
    performance_callback: Optional[
        Callable[[str, Mapping[str, object]], None]
    ] = None,
    performance_stage_prefix: str = "",
    decision_callback: Optional[Callable[[ReadEvidence, dict], None]] = None,
    edge_callback: Optional[Callable[[ReadEvidence, dict], None]] = None,
    retain_per_record: bool = True,
    stream_order: bool = False,
    evaluation_partition: Optional[Mapping[str, object]] = None,
) -> dict:
    """Rescue weak MSP TFs and normalize accepted TF/nuc edge populations."""
    if maximum_sites_per_decision < 1:
        raise ValueError("maximum sites per decision must be positive")
    if minimum_geometry_support < 1:
        raise ValueError("minimum geometry support must be positive")
    if tf_edge_compatibility < 0:
        raise ValueError("TF edge compatibility must be non-negative")
    if not math.isfinite(tf_class_pseudocount) or tf_class_pseudocount < 0.0:
        raise ValueError("TF class pseudocount must be finite and non-negative")
    if tf_class_locus_gap < 0:
        raise ValueError("TF class locus gap must be non-negative")
    if tf_class_max_span < 1:
        raise ValueError("TF class maximum span must be positive")
    if tf_class_max_sites < 1:
        raise ValueError("TF class maximum sites must be positive")
    if nuc_minimum_geometry_support is None:
        nuc_minimum_geometry_support = minimum_geometry_support
    if nuc_minimum_geometry_support < 1:
        raise ValueError("nuc minimum geometry support must be positive")
    if nuc_center_radius < 0:
        raise ValueError("nuc center radius must be non-negative")
    if not 0.0 <= review_posterior <= strong_posterior <= 1.0:
        raise ValueError("posterior thresholds must satisfy 0 <= review <= strong <= 1")
    def mark_performance(name: str, **details: object) -> None:
        if performance_callback is not None:
            performance_callback(f"{performance_stage_prefix}{name}", details)

    target_site_list = list(sites if target_sites is None else target_sites)
    target_read_list = list(reads if target_reads is None else target_reads)
    population_molecules = {read.molecule_id for read in reads}
    target_molecules = {read.molecule_id for read in target_read_list}
    overlapping_molecules = population_molecules & target_molecules
    evaluation_provenance = {
        "mode": (
            "held_out_population"
            if target_molecules and not overlapping_molecules
            else "same_cohort_or_overlapping_population"
        ),
        "population_molecules": len(population_molecules),
        "target_molecules": len(target_molecules),
        "overlapping_molecules": len(overlapping_molecules),
        "molecule_overlap_fraction_of_target": (
            len(overlapping_molecules) / len(target_molecules)
            if target_molecules
            else 0.0
        ),
        "population_prior_target_disjoint": bool(
            target_molecules and not overlapping_molecules
        ),
        "catalog_supplied_by_caller": True,
        "catalog_training_disjointness": "not_verifiable_by_analyzer",
        "declared_partition": dict(evaluation_partition or {}),
    }
    if len(target_site_list) != len(sites):
        raise ValueError("source and target site lists must have equal length")
    strands = sorted({read.strand for read in reads})
    if len(strands) != 2:
        mark_performance(
            "inference_not_applicable",
            reads=len(reads),
            strands=len(strands),
            tf_sites=len(sites),
            nuc_sites=len(nuc_sites),
        )
        return {
            "applicable": False,
            "reason": f"physical strand rescue requires two groups; observed {strands}",
            "strands": strands,
            "sites": [asdict(site) for site in sites],
            "site_models": {},
            "evaluation_provenance": evaluation_provenance,
            "decisions": [],
            "edge_refinement": {
                "tf": {"harmonizations": [], "counts": {}},
                "nuc": {"harmonizations": [], "counts": {}},
            },
            "counts": {},
        }
    by_strand = {
        strand: [read for read in reads if read.strand == strand] for strand in strands
    }
    target_by_strand = {
        strand: [read for read in target_read_list if read.strand == strand]
        for strand in strands
    }
    site_models = {
        strand: {
            index: fit_site_state_model(
                strand_reads,
                site,
                center_radius=center_radius,
                edge_compatibility_bp=tf_edge_compatibility,
            )
            for index, site in enumerate(sites)
        }
        for strand, strand_reads in by_strand.items()
    }
    strand_family_candidates = classify_strand_family_candidates(
        sites,
        site_models,
        strands,
        minimum_support=min_source_support,
    )
    mark_performance(
        "source_state_model_fitting",
        population_reads=len(reads),
        tf_sites=len(sites),
        strand_family_candidate_classes=strand_family_candidates[
            "class_counts"
        ],
    )
    tf_class_model_cache: Dict[Tuple[str, Tuple[int, ...]], dict] = {}
    tf_class_loci = []
    for group in _group_tf_class_loci(
        sites,
        maximum_gap=tf_class_locus_gap,
        maximum_span=tf_class_max_span,
    ):
        site_ids = [sites[index].site_id for index in group]
        locus_payload = "\x1f".join(site_ids)
        locus_id = "tfclasslocus_" + hashlib.sha256(
            locus_payload.encode("utf-8")
        ).hexdigest()[:16]
        locus_record = {
            "locus_id": locus_id,
            "interval": [
                min(sites[index].start for index in group),
                max(sites[index].end for index in group),
            ],
            "site_ids": site_ids,
            "site_count": len(group),
            "maximum_sites": tf_class_max_sites,
            "action_set_complete": len(group) <= tf_class_max_sites,
            "source_model_ids": {},
        }
        if len(group) > tf_class_max_sites:
            locus_record["status"] = "skipped_site_limit"
            locus_record["configuration_count"] = None
            tf_class_loci.append(locus_record)
            continue
        local_sites = [sites[index] for index in group]
        configurations = enumerate_configurations(
            local_sites, include_nucleosome=False
        )
        locus_record["status"] = "modeled"
        locus_record["configuration_count"] = len(configurations)
        locus_record["pooled_iterative_geometry_model"] = (
            fit_iterative_tf_class_geometry_model(
                reads,
                local_sites,
                configurations,
                boundary_search_radius=min(6, tf_edge_compatibility),
                pseudocount=tf_class_pseudocount,
                center_radius=center_radius,
                max_iter=50,
            )
        )
        for strand in strands:
            cache_key = (strand, tuple(group))
            model = fit_tf_configuration_class_model(
                by_strand[strand],
                local_sites,
                configurations,
                center_radius=center_radius,
                pseudocount=tf_class_pseudocount,
                source_stratum=strand,
            )
            tf_class_model_cache[cache_key] = model
            locus_record["source_model_ids"][strand] = model["model_id"]
        tf_class_loci.append(locus_record)
    mark_performance(
        "tf_configuration_class_model_fitting",
        loci=len(tf_class_loci),
        modeled_loci=sum(
            value["status"] == "modeled" for value in tf_class_loci
        ),
        skipped_site_limit=sum(
            value["status"] == "skipped_site_limit"
            for value in tf_class_loci
        ),
        models=len(tf_class_model_cache),
        pooled_iterative_geometry_models=sum(
            "pooled_iterative_geometry_model" in value
            for value in tf_class_loci
        ),
    )
    decisions = []
    counts = {
        "candidate_msp_groups": 0,
        "locally_unsupported": 0,
        "source_sites_inside_nucs_ignored": 0,
        "source_sites_already_covered_by_tf": 0,
        "template_not_contained_by_msp": 0,
        "template_not_fully_mapped": 0,
        "templates_truncated": 0,
        "strong": 0,
        "review": 0,
        "retain_current": 0,
        "from_msp": 0,
    }
    source_supported_by_target = {}
    for target in strands:
        source = strands[1] if target == strands[0] else strands[0]
        source_supported_by_target[target] = {
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
                site, source, site_models[source][index]
            )
            >= opportunity_conditioned_call_fraction(
                site, target, site_models[target][index]
            )
        }
    ordered_target_reads = (
        ((read.strand, read) for read in target_read_list)
        if stream_order
        else (
            (target, read)
            for target in strands
            for read in target_by_strand[target]
        )
    )
    for target, read in ordered_target_reads:
        source = strands[1] if target == strands[0] else strands[0]
        source_supported = source_supported_by_target[target]
        if not source_supported:
            continue
        direct = match_direct_site_indices(
            _calls_for_type(read, "tf"),
            target_site_list,
            center_radius=center_radius,
        )
        existing_tf_overlaps = {
            index
            for index, site in enumerate(target_site_list)
            if any(
                site.start < call.end and call.start < site.end
                for call in read.tfs
            )
        }
        nuc_overlaps = {
            index
            for index, site in enumerate(target_site_list)
            if any(
                site.start < nuc.end and nuc.start < site.end
                for nuc in read.nucs
            )
        }
        blocked = existing_tf_overlaps | nuc_overlaps | {
            index
            for index, site in enumerate(target_site_list)
            if any(
                site.start < target_site_list[other].end
                and target_site_list[other].start < site.end
                for other in direct
            )
        }
        counts["source_sites_already_covered_by_tf"] += len(
            source_supported & existing_tf_overlaps
        )
        counts["source_sites_inside_nucs_ignored"] += len(
            source_supported & nuc_overlaps
        )
        for call in read.msps:
            centered = [
                index
                for index in source_supported
                if index not in blocked
                and call.start <= target_site_list[index].center < call.end
            ]
            contained = [
                index
                for index in centered
                if call.start <= target_site_list[index].start
                and target_site_list[index].end <= call.end
            ]
            counts["template_not_contained_by_msp"] += len(centered) - len(
                contained
            )
            relevant = [
                index
                for index in contained
                if read.fully_maps(
                    target_site_list[index].start,
                    target_site_list[index].end,
                )
            ]
            counts["template_not_fully_mapped"] += len(contained) - len(relevant)
            for group in _group_site_indices(
                relevant, target_site_list, accessible_site_gap
            ):
                counts["candidate_msp_groups"] += 1
                decision = _score_decision(
                    read,
                    call,
                    group,
                    sites=sites,
                    target_sites=target_site_list,
                    source=source,
                    target=target,
                    source_models=site_models[source],
                    source_reads=by_strand[source],
                    class_model_cache=tf_class_model_cache,
                    center_radius=center_radius,
                    class_model_pseudocount=tf_class_pseudocount,
                    strong_posterior=strong_posterior,
                    review_posterior=review_posterior,
                    maximum_sites=maximum_sites_per_decision,
                )
                if decision is None:
                    counts["locally_unsupported"] += 1
                    continue
                decision["evaluation_provenance"] = dict(
                    evaluation_provenance
                )
                if decision_callback is not None:
                    decision_callback(read, decision)
                if retain_per_record:
                    decisions.append(decision)
                counts["from_msp"] += 1
                counts[decision["proposal_tier"]] += 1
                counts["templates_truncated"] += int(
                    decision["templates_truncated"]
                )
    if retain_per_record:
        decisions.sort(
            key=lambda decision: (
                decision["library_id"] or "",
                decision["read"],
                decision["current_interval"],
                decision["proposed"],
            )
        )
        decisions = _deduplicate_decisions(decisions)
        counts["from_msp"] = len(decisions)
        for tier in ("strong", "review", "retain_current"):
            counts[tier] = sum(
                decision["proposal_tier"] == tier for decision in decisions
            )
        counts["templates_truncated"] = sum(
            bool(decision["templates_truncated"]) for decision in decisions
        )
    mark_performance(
        "msp_tf_rescue_scoring",
        target_reads=len(target_read_list),
        candidate_msp_groups=counts["candidate_msp_groups"],
        decisions=counts["from_msp"],
    )
    tf_edge_refinement = analyze_shared_geometry(
        target_read_list,
        sites,
        site_models,
        population_reads=reads,
        call_type="tf",
        center_radius=center_radius,
        minimum_geometry_support=minimum_geometry_support,
        record_callback=edge_callback,
        retain_harmonizations=retain_per_record,
        stream_order=stream_order,
        evaluation_provenance=evaluation_provenance,
    )
    mark_performance(
        "tf_edge_normalization",
        target_reads=len(target_read_list),
        tf_sites=len(sites),
        harmonizations=len(tf_edge_refinement.get("harmonizations", [])),
        source_samples_prepared=int(
            tf_edge_refinement.get("counts", {}).get(
                "edge_population_source_samples_prepared", 0
            )
        ),
        naive_sample_evaluations_avoided=int(
            tf_edge_refinement.get("counts", {}).get(
                "edge_population_naive_sample_evaluations_avoided", 0
            )
        ),
    )
    nuc_edge_refinement = analyze_shared_geometry(
        target_read_list,
        nuc_sites,
        population_reads=reads,
        call_type="nuc",
        center_radius=nuc_center_radius,
        minimum_geometry_support=nuc_minimum_geometry_support,
        record_callback=edge_callback,
        retain_harmonizations=retain_per_record,
        stream_order=stream_order,
        evaluation_provenance=evaluation_provenance,
    )
    mark_performance(
        "nuc_edge_normalization",
        target_reads=len(target_read_list),
        nuc_sites=len(nuc_sites),
        harmonizations=len(nuc_edge_refinement.get("harmonizations", [])),
        source_samples_prepared=int(
            nuc_edge_refinement.get("counts", {}).get(
                "edge_population_source_samples_prepared", 0
            )
        ),
        naive_sample_evaluations_avoided=int(
            nuc_edge_refinement.get("counts", {}).get(
                "edge_population_naive_sample_evaluations_avoided", 0
            )
        ),
    )
    if retain_per_record:
        resolve_joint_edge_topology(tf_edge_refinement, nuc_edge_refinement)
        resolve_rescue_edge_topology(
            decisions, tf_edge_refinement, nuc_edge_refinement
        )
    mark_performance(
        "topology_resolution",
        decisions=counts["from_msp"],
        tf_harmonizations=len(tf_edge_refinement.get("harmonizations", [])),
        nuc_harmonizations=len(nuc_edge_refinement.get("harmonizations", [])),
    )
    return {
        "applicable": True,
        "strands": strands,
        "sites": [asdict(site) for site in sites],
        "target_sites": [asdict(site) for site in target_site_list],
        "site_models": {
            strand: {
                sites[index].site_id: model for index, model in models.items()
            }
            for strand, models in site_models.items()
        },
        "strand_family_candidates": strand_family_candidates,
        "tf_class_models": {
            model["model_id"]: model
            for _key, model in sorted(tf_class_model_cache.items())
        },
        "tf_class_loci": tf_class_loci,
        "evaluation_provenance": evaluation_provenance,
        "decisions": decisions,
        "edge_refinement": {
            "tf": tf_edge_refinement,
            "nuc": nuc_edge_refinement,
        },
        "counts": counts,
    }
