"""Population-level discovery of recurrent TF footprint geometries.

This module is deliberately independent of BAM parsing, chemistry models, and
the rescue implementations.  Callers provide already projected ordinary TF and
MSP annotations plus mapped reference blocks.  Every TF observation is retained.
An observation teaches consensus boundaries only when ``geometry_eligible`` is
true *and* its complete reference interval satisfies the configured mapping
fraction; other observations remain auditable and may be assigned to a geometry
learned from eligible calls.

The public result contains both site summaries and explicit call-to-site
assignments.  The latter are required for future leave-one-molecule-out rescue
models and make it possible to audit every source call.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass, replace
from hashlib import sha256
from math import ceil, exp, floor, isclose, isfinite
from numbers import Real
from typing import (
    Callable,
    Dict,
    Hashable,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
)

import numpy as np

try:  # Optional acceleration; the exact Python path remains the fallback.
    import numba as _numba
except ImportError:  # pragma: no cover - exercised in minimal installations.
    _numba = None

Interval = Tuple[int, int]
_MoleculeKey = Tuple[str, str, str]
_EdgeRecord = TypeVar("_EdgeRecord")


@dataclass(frozen=True)
class TFObservation:
    """One valid TF call in zero-based, half-open reference coordinates.

    ``geometry_eligible=False`` keeps the call as a population observation but
    prevents it from teaching the frozen site geometry.  It is a provenance
    control for later resolved layers, not a quality filter; there is
    intentionally no TQ field.
    """

    call_id: str
    start: int
    end: int
    geometry_eligible: bool = True

    @property
    def center(self) -> float:
        return (self.start + self.end) / 2.0


@dataclass(frozen=True)
class BaselineMolecule:
    """Annotation-only molecule record consumed by site discovery.

    ``molecule_id`` must already encode the caller's molecule-collapse policy.
    Repeated records with the same contig/stratum/ID are coalesced by unioning
    mapped blocks and annotations; adapters must therefore use the same ID only
    for duplicate or reference-collinear fragments of one denominator molecule.
    ``stratum`` is a physical/read strand label such as ``CT``, ``GA``, ``FWD``,
    or ``REV``; use ``.`` when no meaningful stratum exists. A molecule ID may
    not appear in multiple strata on the same contig.
    """

    molecule_id: str
    contig: str
    mapped_blocks: Tuple[Interval, ...]
    tfs: Tuple[TFObservation, ...]
    msps: Tuple[Interval, ...] = ()
    stratum: str = "."


@dataclass(frozen=True)
class SiteDiscoveryConfig:
    """Geometry and reporting parameters for baseline site discovery.

    ``edge_compatibility_bp`` is a maximum within-family diameter for *each*
    boundary, not a transitive pairwise-linkage radius.  Thus a chain of calls
    whose neighboring edges differ by the tolerance cannot bridge two more
    distant footprint geometries into one family.  The per-stratum geometry
    threshold decides which strata get equal votes in canonical boundaries;
    ``minimum_geometry_support`` and ``minimum_population_support`` separately
    label total geometry-teacher and TF-supporting population readiness.  None
    of these thresholds removes a discovered site.
    """

    smoothing_sigma_bp: float = 3.0
    peak_distance_bp: int = 15
    assignment_radius_bp: int = 10
    edge_compatibility_bp: int = 12
    minimum_geometry_support_per_stratum: int = 3
    minimum_geometry_support: int = 3
    minimum_population_support: int = 3
    minimum_mapped_fraction: float = 0.95

    def __post_init__(self) -> None:
        if (
            isinstance(self.smoothing_sigma_bp, bool)
            or not isinstance(self.smoothing_sigma_bp, Real)
            or not isfinite(float(self.smoothing_sigma_bp))
            or self.smoothing_sigma_bp <= 0
        ):
            raise ValueError("smoothing_sigma_bp must be finite and positive")
        for name in (
            "peak_distance_bp",
            "assignment_radius_bp",
            "edge_compatibility_bp",
            "minimum_geometry_support_per_stratum",
            "minimum_geometry_support",
            "minimum_population_support",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise ValueError(f"{name} must be an integer")
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.peak_distance_bp == 0:
            raise ValueError("peak_distance_bp must be positive")
        if (
            isinstance(self.minimum_mapped_fraction, bool)
            or not isinstance(self.minimum_mapped_fraction, Real)
            or not isfinite(float(self.minimum_mapped_fraction))
            or not 0 < self.minimum_mapped_fraction <= 1
        ):
            raise ValueError("minimum_mapped_fraction must be in (0, 1]")


@dataclass(frozen=True)
class StratumSiteSummary:
    """Geometry and population counts for one physical/read stratum."""

    stratum: str
    cluster_support_molecules: int
    n_fully_mapped: int
    n_tf: int
    n_msp: int
    n_tf_msp: int
    median_start: int
    median_end: int
    start_mad: float
    end_mad: float


@dataclass(frozen=True)
class TFPopulationSite:
    """One recurrent TF geometry and its baseline population measurements.

    The four ``n_*_(msp|no_msp)`` cells partition unique molecules fully
    mapping the complete canonical site.  Top-level boundary MADs conservatively
    retain the larger of within-stratum dispersion and between-stratum median
    disagreement; exact per-stratum values remain available in ``strata``.
    """

    site_id: str
    locus_id: str
    contig: str
    start: int
    end: int
    locus_summit: int
    summit: int
    family_index: int
    assigned_call_count: int
    geometry_call_count: int
    cluster_support_molecules: int
    n_fully_mapped: int
    n_tf: int
    n_msp: int
    n_tf_msp: int
    n_tf_no_msp: int
    n_no_tf_msp: int
    n_no_tf_no_msp: int
    start_mad: float
    end_mad: float
    geometry_ready: bool
    population_ready: bool
    strata: Tuple[StratumSiteSummary, ...]

    @property
    def occupancy_overall(self) -> Optional[float]:
        if self.n_fully_mapped == 0:
            return None
        return self.n_tf / self.n_fully_mapped

    @property
    def occupancy_given_msp(self) -> Optional[float]:
        if self.n_msp == 0:
            return None
        return self.n_tf_msp / self.n_msp

    @property
    def population_support(self) -> int:
        """Unique fully mapped molecules assigned to this hypothesis."""

        return self.n_tf

    @property
    def geometry_support(self) -> int:
        """Unique eligible molecules that established this geometry."""

        return self.cluster_support_molecules

    @property
    def analysis_ready(self) -> bool:
        """Whether both geometry and population support thresholds are met."""

        return self.geometry_ready and self.population_ready

    def as_record(self) -> Dict[str, object]:
        """Return a stable, serialization-friendly baseline record."""

        return {
            "contig": self.contig,
            "start": self.start,
            "end": self.end,
            "site_id": self.site_id,
            "locus_id": self.locus_id,
            "locus_summit": self.locus_summit,
            "summit": self.summit,
            "family_index": self.family_index,
            "start_mad": self.start_mad,
            "end_mad": self.end_mad,
            "assigned_call_count": self.assigned_call_count,
            "geometry_call_count": self.geometry_call_count,
            "cluster_support_molecules": self.cluster_support_molecules,
            "n_fully_mapped": self.n_fully_mapped,
            "n_tf": self.n_tf,
            "n_msp": self.n_msp,
            "n_tf_msp": self.n_tf_msp,
            "n_tf_no_msp": self.n_tf_no_msp,
            "n_no_tf_msp": self.n_no_tf_msp,
            "n_no_tf_no_msp": self.n_no_tf_no_msp,
            "occupancy_overall": self.occupancy_overall,
            "occupancy_given_msp": self.occupancy_given_msp,
            "population_support": self.population_support,
            "geometry_support": self.geometry_support,
            "geometry_ready": self.geometry_ready,
            "population_ready": self.population_ready,
            "analysis_ready": self.analysis_ready,
        }


@dataclass(frozen=True)
class TFCallAssignment:
    """Auditable assignment of one ordinary TF call to at most one site."""

    call_id: str
    molecule_id: str
    contig: str
    stratum: str
    start: int
    end: int
    site_id: Optional[str]
    used_for_geometry: bool
    fully_maps_site: bool
    msp_contains_site: bool


@dataclass(frozen=True)
class SiteDiscoveryDiagnostics:
    input_records: int
    unique_molecules: int
    raw_tf_calls: int
    unique_tf_calls: int
    geometry_tf_calls: int
    assigned_tf_calls: int
    unassigned_tf_calls: int


@dataclass(frozen=True)
class TFModelLocus:
    """One smoothed binding population with one or more geometry families."""

    locus_id: str
    contig: str
    start: int
    end: int
    summit: int
    family_site_ids: Tuple[str, ...]

    @property
    def family_count(self) -> int:
        return len(self.family_site_ids)


@dataclass(frozen=True)
class TFSiteCatalog:
    """Immutable TF-model catalog returned by :func:`build_tf_site_catalog`."""

    loci: Tuple[TFModelLocus, ...]
    sites: Tuple[TFPopulationSite, ...]
    assignments: Tuple[TFCallAssignment, ...]
    config: SiteDiscoveryConfig
    diagnostics: SiteDiscoveryDiagnostics


@dataclass(frozen=True)
class _NormalizedMolecule:
    key: _MoleculeKey
    molecule_id: str
    contig: str
    stratum: str
    mapped_blocks: Tuple[Interval, ...]
    mapped_block_starts: Tuple[int, ...]
    mapped_block_prefix_bases: Tuple[int, ...]
    tfs: Tuple[TFObservation, ...]
    msps: Tuple[Interval, ...]
    msp_starts: Tuple[int, ...]
    msp_prefix_max_ends: Tuple[int, ...]


@dataclass(frozen=True)
class _CallRecord:
    molecule: _NormalizedMolecule
    call: TFObservation

    @property
    def center(self) -> float:
        return self.call.center


@dataclass(frozen=True)
class _SiteGeometry:
    site_id: str
    locus_id: str
    contig: str
    start: int
    end: int
    locus_summit: int
    summit: int
    family_index: int
    records: Tuple[_CallRecord, ...]
    representatives: Tuple[_CallRecord, ...]
    start_mad: float
    end_mad: float
    stratum_geometry: Tuple[Tuple[str, int, int, int, float, float], ...]


@dataclass(frozen=True)
class _SiteIntervalNode:
    center: float
    overlap_by_start: Tuple[_SiteGeometry, ...]
    overlap_by_end: Tuple[_SiteGeometry, ...]
    left: Optional[_SiteIntervalNode]
    right: Optional[_SiteIntervalNode]


def _build_site_interval_index(
    geometries: Sequence[_SiteGeometry],
) -> Optional[_SiteIntervalNode]:
    if not geometries:
        return None
    center = float(
        np.median(
            np.asarray(
                [(geometry.start + geometry.end) / 2.0 for geometry in geometries],
                dtype=np.float64,
            )
        )
    )
    left = []
    right = []
    overlap = []
    for geometry in geometries:
        if geometry.end <= center:
            left.append(geometry)
        elif geometry.start > center:
            right.append(geometry)
        else:
            overlap.append(geometry)
    return _SiteIntervalNode(
        center=center,
        overlap_by_start=tuple(
            sorted(
                overlap,
                key=lambda value: (value.start, value.end, value.site_id),
            )
        ),
        overlap_by_end=tuple(
            sorted(
                overlap,
                key=lambda value: (-value.end, value.start, value.site_id),
            )
        ),
        left=_build_site_interval_index(left),
        right=_build_site_interval_index(right),
    )


def _query_site_interval_index(
    node: Optional[_SiteIntervalNode], start: int, end: int, result: set
) -> None:
    if node is None or end <= start:
        return
    if end <= node.center:
        for geometry in node.overlap_by_start:
            if geometry.start >= end:
                break
            result.add(geometry.site_id)
        _query_site_interval_index(node.left, start, end, result)
    elif start > node.center:
        for geometry in node.overlap_by_end:
            if geometry.end <= start:
                break
            result.add(geometry.site_id)
        _query_site_interval_index(node.right, start, end, result)
    else:
        result.update(geometry.site_id for geometry in node.overlap_by_start)
        _query_site_interval_index(node.left, start, end, result)
        _query_site_interval_index(node.right, start, end, result)


if _numba is not None:

    @_numba.njit(cache=True, parallel=True)
    def _site_coverage_counts_numba(
        site_starts,
        site_ends,
        block_offsets,
        block_starts,
        block_ends,
        msp_offsets,
        msp_starts,
        msp_ends,
        stratum_indices,
        stratum_count,
        minimum_fraction,
    ):
        """Count exact mapped/MSP denominators without Python pair loops."""
        site_count = site_starts.size
        molecule_count = stratum_indices.size
        mapped_counts = np.zeros((site_count, stratum_count), dtype=np.int64)
        msp_counts = np.zeros((site_count, stratum_count), dtype=np.int64)
        for site_index in _numba.prange(site_count):
            start = int(site_starts[site_index])
            end = int(site_ends[site_index])
            width = end - start
            if width <= 0:
                continue
            for molecule_index in range(molecule_count):
                first_block = int(block_offsets[molecule_index])
                stop_block = int(block_offsets[molecule_index + 1])
                left_found = False
                right_found = False
                mapped = 0
                for block_index in range(first_block, stop_block):
                    block_start = int(block_starts[block_index])
                    block_end = int(block_ends[block_index])
                    if block_start <= start < block_end:
                        left_found = True
                    if block_start <= end - 1 < block_end:
                        right_found = True
                    overlap_start = start if start > block_start else block_start
                    overlap_end = end if end < block_end else block_end
                    if overlap_end > overlap_start:
                        mapped += overlap_end - overlap_start
                if not left_found or not right_found:
                    continue
                if mapped / width < minimum_fraction:
                    continue
                stratum_index = int(stratum_indices[molecule_index])
                mapped_counts[site_index, stratum_index] += 1
                first_msp = int(msp_offsets[molecule_index])
                stop_msp = int(msp_offsets[molecule_index + 1])
                for msp_index in range(first_msp, stop_msp):
                    if (
                        int(msp_starts[msp_index]) <= start
                        and int(msp_ends[msp_index]) >= end
                    ):
                        msp_counts[site_index, stratum_index] += 1
                        break
        return mapped_counts, msp_counts

else:
    _site_coverage_counts_numba = None


def _validate_interval(interval: Interval, *, label: str) -> Interval:
    if len(interval) != 2:
        raise ValueError(f"{label} must be a (start, end) pair")
    raw_start, raw_end = interval
    if any(
        isinstance(value, bool) or not isinstance(value, (int, np.integer))
        for value in (raw_start, raw_end)
    ):
        raise ValueError(f"{label} coordinates must be integers")
    start, end = int(raw_start), int(raw_end)
    if start < 0 or end <= start:
        raise ValueError(f"invalid {label}: {start}-{end}")
    return start, end


def _identifier(value: object, *, label: str) -> str:
    if value is None:
        raise ValueError(f"{label} must not be None")
    text = str(value)
    if not text:
        raise ValueError(f"{label} must not be empty")
    if any(character in text for character in "\t\n\r"):
        raise ValueError(f"{label} must not contain tab or newline characters")
    return text


def _merge_blocks(blocks: Sequence[Interval]) -> Tuple[Interval, ...]:
    normalized = sorted(_validate_interval(block, label="mapped block") for block in blocks)
    merged: List[List[int]] = []
    for start, end in normalized:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return tuple((start, end) for start, end in merged)


def _coalesce_molecules(
    molecules: Iterable[BaselineMolecule],
) -> Tuple[Tuple[_NormalizedMolecule, ...], int, int]:
    grouped: Dict[_MoleculeKey, Dict[str, object]] = {}
    strata_by_molecule: Dict[Tuple[str, str], str] = {}
    input_records = 0
    raw_tf_calls = 0
    for molecule in molecules:
        input_records += 1
        raw_tf_calls += len(molecule.tfs)
        contig = _identifier(molecule.contig, label="contig")
        molecule_id = _identifier(molecule.molecule_id, label="molecule_id")
        stratum = _identifier(molecule.stratum or ".", label="stratum")
        molecule_scope = (contig, molecule_id)
        previous_stratum = strata_by_molecule.setdefault(molecule_scope, stratum)
        if previous_stratum != stratum:
            raise ValueError(
                f"molecule_id {molecule_id!r} appears in multiple strata "
                f"on {contig!r}: {previous_stratum!r} and {stratum!r}"
            )
        key = (contig, stratum, molecule_id)
        entry = grouped.setdefault(
            key,
            {
                "blocks": [],
                "msps": set(),
                "tfs": {},
            },
        )
        entry["blocks"].extend(molecule.mapped_blocks)  # type: ignore[union-attr]
        for interval in molecule.msps:
            entry["msps"].add(_validate_interval(interval, label="MSP"))  # type: ignore[union-attr]
        calls: Dict[str, TFObservation] = entry["tfs"]  # type: ignore[assignment]
        for call in molecule.tfs:
            call_id = _identifier(call.call_id, label="call_id")
            start, end = _validate_interval((call.start, call.end), label="TF call")
            normalized_call = TFObservation(
                call_id=call_id,
                start=start,
                end=end,
                geometry_eligible=bool(call.geometry_eligible),
            )
            previous = calls.get(call_id)
            if previous is not None and previous != normalized_call:
                raise ValueError(
                    f"conflicting duplicate TF call {call_id!r} on molecule {molecule_id!r}"
                )
            calls[call_id] = normalized_call

    result = []
    for key in sorted(grouped):
        contig, stratum, molecule_id = key
        entry = grouped[key]
        blocks = _merge_blocks(entry["blocks"])  # type: ignore[arg-type]
        msps = tuple(sorted(entry["msps"]))  # type: ignore[arg-type]
        msp_prefix_max_ends = []
        maximum_end = -1
        for _msp_start, msp_end in msps:
            maximum_end = max(maximum_end, msp_end)
            msp_prefix_max_ends.append(maximum_end)
        calls = entry["tfs"]  # type: ignore[assignment]
        tfs = tuple(sorted(calls.values(), key=lambda call: (call.start, call.end, call.call_id)))
        mapped_block_prefix_bases = [0]
        for block_start, block_end in blocks:
            mapped_block_prefix_bases.append(
                mapped_block_prefix_bases[-1] + block_end - block_start
            )
        result.append(
            _NormalizedMolecule(
                key=key,
                molecule_id=molecule_id,
                contig=contig,
                stratum=stratum,
                mapped_blocks=blocks,
                mapped_block_starts=tuple(start for start, _end in blocks),
                mapped_block_prefix_bases=tuple(mapped_block_prefix_bases),
                tfs=tfs,
                msps=msps,
                msp_starts=tuple(start for start, _end in msps),
                msp_prefix_max_ends=tuple(msp_prefix_max_ends),
            )
        )
    return tuple(result), input_records, raw_tf_calls


def _molecule_fully_maps(
    molecule: _NormalizedMolecule,
    start: int,
    end: int,
    minimum_fraction: float,
) -> bool:
    """Check complete interval coverage in logarithmic time.

    Normalization guarantees sorted, disjoint mapped blocks.  Binary searches
    locate the blocks containing the two required endpoints, while a cumulative
    block-length index gives the mapped bases between them without rescanning
    every CIGAR block for every candidate site.
    """

    if end <= start or not molecule.mapped_blocks:
        return False
    left_index = bisect_right(molecule.mapped_block_starts, start) - 1
    right_index = bisect_right(molecule.mapped_block_starts, end - 1) - 1
    if left_index < 0 or right_index < 0:
        return False
    left_block = molecule.mapped_blocks[left_index]
    right_block = molecule.mapped_blocks[right_index]
    if left_block[1] <= start or right_block[1] < end:
        return False
    if left_index == right_index:
        return True

    mapped = left_block[1] - start
    mapped += end - right_block[0]
    mapped += (
        molecule.mapped_block_prefix_bases[right_index]
        - molecule.mapped_block_prefix_bases[left_index + 1]
    )
    return mapped / (end - start) >= minimum_fraction


def _msp_contains(molecule: _NormalizedMolecule, start: int, end: int) -> bool:
    index = bisect_right(molecule.msp_starts, start) - 1
    return index >= 0 and molecule.msp_prefix_max_ends[index] >= end


def _interval_center_bin(start: int, end: int) -> int:
    return (start + end + 1) // 2


def _center_bin(record: _CallRecord) -> int:
    return _interval_center_bin(record.call.start, record.call.end)


def _unique_center_counts(records: Sequence[_CallRecord]) -> List[Tuple[int, int]]:
    """Count unique molecules per center without retaining molecule-center sets."""

    ordered = sorted(
        records,
        key=lambda record: (
            _center_bin(record),
            record.molecule.key,
            record.call.start,
            record.call.end,
            record.call.call_id,
        ),
    )
    counts: List[Tuple[int, int]] = []
    previous: Optional[Tuple[int, _MoleculeKey]] = None
    for record in ordered:
        center = _center_bin(record)
        contribution = (center, record.molecule.key)
        if contribution == previous:
            continue
        if counts and counts[-1][0] == center:
            counts[-1] = (center, counts[-1][1] + 1)
        else:
            counts.append((center, 1))
        previous = contribution
    return counts


def _iter_smoothed_center_density(
    center_counts: Sequence[Tuple[int, int]], config: SiteDiscoveryConfig
) -> Iterator[Tuple[Optional[int], float]]:
    """Stream finite-kernel density runs with a ``None`` separator.

    The implementation never materializes chromosome-length arrays or one
    Python object per kernel-touched base.  Runtime is proportional to the
    union of finite kernel supports, and active-center memory is bounded by the
    kernel width.
    """

    if not center_counts:
        return
    radius = max(1, int(ceil(4.0 * config.smoothing_sigma_bp)))
    sigma = config.smoothing_sigma_bp
    kernel = tuple(exp(-0.5 * (offset / sigma) ** 2) for offset in range(-radius, radius + 1))

    run_start = 0
    while run_start < len(center_counts):
        run_end = run_start
        while (
            run_end + 1 < len(center_counts)
            and center_counts[run_end + 1][0] - center_counts[run_end][0] <= 2 * radius + 1
        ):
            run_end += 1

        first_position = center_counts[run_start][0] - radius
        last_position = center_counts[run_end][0] + radius
        active_left = run_start
        active_right = run_start
        for position in range(first_position, last_position + 1):
            while active_left <= run_end and center_counts[active_left][0] < position - radius:
                active_left += 1
            active_right = max(active_right, active_left)
            while active_right <= run_end and center_counts[active_right][0] <= position + radius:
                active_right += 1
            density = 0.0
            for center, count in center_counts[active_left:active_right]:
                density += count * kernel[position - center + radius]
            yield position, density
        yield None, 0.0
        run_start = run_end + 1


def _local_density_maxima(
    center_counts: Sequence[Tuple[int, int]], config: SiteDiscoveryConfig
) -> List[Tuple[int, float]]:
    """Find local maximum plateaus, choosing each plateau's left midpoint."""

    maxima: List[Tuple[int, float]] = []
    plateau_start: Optional[int] = None
    plateau_end: Optional[int] = None
    plateau_value = 0.0
    left_value = 0.0

    def finish_plateau(right_value: float) -> None:
        if (
            plateau_start is not None
            and plateau_end is not None
            and plateau_value > left_value
            and plateau_value > right_value
        ):
            maxima.append(((plateau_start + plateau_end) // 2, plateau_value))

    for position, value in _iter_smoothed_center_density(center_counts, config):
        if position is None:
            finish_plateau(0.0)
            plateau_start = None
            plateau_end = None
            plateau_value = 0.0
            left_value = 0.0
            continue
        if plateau_start is None:
            plateau_start = position
            plateau_end = position
            plateau_value = value
        elif isclose(value, plateau_value, rel_tol=0.0, abs_tol=1e-12):
            plateau_end = position
        else:
            finish_plateau(value)
            left_value = plateau_value
            plateau_start = position
            plateau_end = position
            plateau_value = value
    finish_plateau(0.0)
    return maxima


def _center_modes(records: Sequence[_CallRecord], config: SiteDiscoveryConfig) -> List[int]:
    center_counts = _unique_center_counts(records)
    maxima = _local_density_maxima(center_counts, config)

    selected: List[int] = []
    selected_buckets: Dict[int, List[int]] = {}
    distance = config.peak_distance_bp
    for position, _value in sorted(maxima, key=lambda value: (-value[1], value[0])):
        bucket = position // distance
        neighbors = (
            selected_buckets.get(bucket - 1, [])
            + selected_buckets.get(bucket, [])
            + selected_buckets.get(bucket + 1, [])
        )
        if all(abs(position - previous) >= distance for previous in neighbors):
            selected.append(position)
            selected_buckets.setdefault(bucket, []).append(position)
    modes = sorted(selected)

    # A broad mode must not silently absorb or discard a valid distant call.
    # Add deterministic singleton/fallback modes until every call is within the
    # assignment radius of at least one mode.
    fallback_modes: List[int] = []
    for center, _count in center_counts:
        insertion = bisect_left(modes, center)
        selected_neighbors = modes[max(0, insertion - 1) : min(len(modes), insertion + 1)]
        selected_covers = any(
            abs(center - mode) <= config.assignment_radius_bp for mode in selected_neighbors
        )
        fallback_covers = bool(
            fallback_modes and center - fallback_modes[-1] <= config.assignment_radius_bp
        )
        if not selected_covers and not fallback_covers:
            fallback_modes.append(center)
    return sorted(set(modes) | set(fallback_modes))


def _assign_to_modes(
    records: Sequence[_CallRecord], modes: Sequence[int], radius: int
) -> Dict[int, List[_CallRecord]]:
    groups: Dict[int, List[_CallRecord]] = {mode: [] for mode in modes}
    for record in sorted(
        records,
        key=lambda value: (
            _center_bin(value),
            value.call.start,
            value.call.end,
            value.molecule.key,
            value.call.call_id,
        ),
    ):
        center = _center_bin(record)
        insertion = bisect_left(modes, center)
        neighbors = modes[max(0, insertion - 1) : min(len(modes), insertion + 1)]
        mode = min(neighbors, key=lambda candidate: (abs(center - candidate), candidate))
        if abs(center - mode) <= radius:
            groups[mode].append(record)
    return {mode: values for mode, values in groups.items() if values}


def bounded_edge_components(
    records: Sequence[_EdgeRecord],
    tolerance: int,
    *,
    interval_key: Callable[[_EdgeRecord], Interval],
    identity_key: Callable[[_EdgeRecord], Hashable],
) -> List[List[_EdgeRecord]]:
    """Deterministically partition a mode into bounded-diameter families.

    Families are constrained to contiguous runs in doubled-center/length order.
    Maximal bounded runs are built in both center and length orientations, then
    four candidate partitions are compared by family count, squared boundary
    diameter, and a coordinate-free membership signature. This makes the result
    invariant to reflection while retaining the hard diameter bound on both
    edges.
    """
    if tolerance < 0:
        raise ValueError("edge tolerance must be non-negative")

    def ordered_records(
        *, reverse_centers: bool, reverse_lengths: bool
    ) -> Tuple[_EdgeRecord, ...]:
        center_direction = -1 if reverse_centers else 1
        length_direction = -1 if reverse_lengths else 1
        return tuple(
            sorted(
                records,
                key=lambda record: (
                    center_direction * sum(interval_key(record)),
                    length_direction
                    * (interval_key(record)[1] - interval_key(record)[0]),
                    identity_key(record),
                ),
            )
        )

    def partition(
        ordered: Sequence[_EdgeRecord],
    ) -> Tuple[Tuple[_EdgeRecord, ...], ...]:
        families = []
        start = 0
        while start < len(ordered):
            start_value, end_value = interval_key(ordered[start])
            start_min = start_max = start_value
            end_min = end_max = end_value
            stop = start + 1
            while stop < len(ordered):
                record = ordered[stop]
                record_start, record_end = interval_key(record)
                next_start_min = min(start_min, record_start)
                next_start_max = max(start_max, record_start)
                next_end_min = min(end_min, record_end)
                next_end_max = max(end_max, record_end)
                if (
                    next_start_max - next_start_min > tolerance
                    or next_end_max - next_end_min > tolerance
                ):
                    break
                start_min, start_max = next_start_min, next_start_max
                end_min, end_max = next_end_min, next_end_max
                stop += 1
            families.append(tuple(ordered[start:stop]))
            start = stop
        return tuple(families)

    candidates = tuple(
        partition(
            ordered_records(
                reverse_centers=reverse_centers,
                reverse_lengths=reverse_lengths,
            )
        )
        for reverse_centers in (False, True)
        for reverse_lengths in (False, True)
    )

    def partition_key(families: Sequence[Sequence[_EdgeRecord]]) -> tuple:
        dispersion = 0
        signatures = []
        for family in families:
            intervals = [interval_key(record) for record in family]
            starts = [value[0] for value in intervals]
            ends = [value[1] for value in intervals]
            dispersion += (max(starts) - min(starts)) ** 2
            dispersion += (max(ends) - min(ends)) ** 2
            signatures.append(
                tuple(sorted(identity_key(record) for record in family))
            )
        return len(families), dispersion, tuple(sorted(signatures))

    grouped = min(candidates, key=partition_key)
    return sorted(
        (list(family) for family in grouped),
        key=lambda values: (
            float(np.median([interval_key(value)[0] for value in values])),
            float(np.median([interval_key(value)[1] for value in values])),
            tuple(sorted(identity_key(value) for value in values)),
        ),
    )


def _edge_components(
    records: Sequence[_CallRecord], tolerance: int
) -> List[List[_CallRecord]]:
    return bounded_edge_components(
        records,
        tolerance,
        interval_key=lambda record: (record.call.start, record.call.end),
        identity_key=lambda record: (*record.molecule.key, record.call.call_id),
    )


def _mad(values: Sequence[int]) -> float:
    if not values:
        return 0.0
    array = np.asarray(values, dtype=np.float64)
    median = float(np.median(array))
    return float(np.median(np.abs(array - median)))


def _rounded_median(values: Sequence[int]) -> int:
    return int(floor(float(np.median(np.asarray(values, dtype=np.float64))) + 0.5))


def _molecule_representatives(records: Sequence[_CallRecord]) -> Tuple[_CallRecord, ...]:
    preliminary_start = float(np.median([record.call.start for record in records]))
    preliminary_end = float(np.median([record.call.end for record in records]))
    by_molecule: Dict[_MoleculeKey, List[_CallRecord]] = {}
    for record in records:
        by_molecule.setdefault(record.molecule.key, []).append(record)
    representatives = []
    for key in sorted(by_molecule):
        representatives.append(
            min(
                by_molecule[key],
                key=lambda record: (
                    abs(record.call.start - preliminary_start)
                    + abs(record.call.end - preliminary_end),
                    record.call.start,
                    record.call.end,
                    record.call.call_id,
                ),
            )
        )
    return tuple(representatives)


def _component_summit(records: Sequence[_CallRecord], config: SiteDiscoveryConfig) -> int:
    maxima = _local_density_maxima(_unique_center_counts(records), config)
    return min(maxima, key=lambda value: (-value[1], value[0]))[0]


def _build_geometry(
    contig: str,
    records: Sequence[_CallRecord],
    config: SiteDiscoveryConfig,
    *,
    locus_summit: Optional[int] = None,
    family_index: int = 1,
) -> _SiteGeometry:
    representatives = _molecule_representatives(records)
    by_stratum: Dict[str, List[_CallRecord]] = {}
    for record in representatives:
        by_stratum.setdefault(record.molecule.stratum, []).append(record)

    stratum_geometry = []
    canonical_candidates = []
    for stratum in sorted(by_stratum):
        values = by_stratum[stratum]
        starts = [record.call.start for record in values]
        ends = [record.call.end for record in values]
        median_start = _rounded_median(starts)
        median_end = _rounded_median(ends)
        summary = (
            stratum,
            len(values),
            median_start,
            median_end,
            _mad(starts),
            _mad(ends),
        )
        stratum_geometry.append(summary)
        if len(values) >= config.minimum_geometry_support_per_stratum:
            canonical_candidates.append(summary)

    if canonical_candidates:
        start = _rounded_median([value[2] for value in canonical_candidates])
        end = _rounded_median([value[3] for value in canonical_candidates])
    else:
        start = _rounded_median([record.call.start for record in representatives])
        end = _rounded_median([record.call.end for record in representatives])
    if end <= start:  # Defensive; validated source intervals make this unlikely.
        end = start + 1
    summit = min(max(_component_summit(representatives, config), start), end - 1)
    if locus_summit is None:
        locus_summit = summit
    locus_digest = sha256(f"fiberhmm.tf-locus.v1\0{contig}\0{locus_summit}".encode()).hexdigest()[
        :16
    ]
    locus_id = f"tflocus_{locus_digest}"
    start_mad = max(
        max(value[4] for value in stratum_geometry),
        _mad([value[2] for value in stratum_geometry]),
    )
    end_mad = max(
        max(value[5] for value in stratum_geometry),
        _mad([value[3] for value in stratum_geometry]),
    )
    digest = sha256(
        f"fiberhmm.tf-site.v1\0{contig}\0{start}\0{end}\0{summit}".encode()
    ).hexdigest()[:16]
    return _SiteGeometry(
        site_id=f"tfsite_{digest}",
        locus_id=locus_id,
        contig=contig,
        start=start,
        end=end,
        locus_summit=locus_summit,
        summit=summit,
        family_index=family_index,
        records=tuple(records),
        representatives=representatives,
        start_mad=start_mad,
        end_mad=end_mad,
        stratum_geometry=tuple(stratum_geometry),
    )


def _discover_geometry(
    records: Sequence[_CallRecord], config: SiteDiscoveryConfig
) -> Tuple[_SiteGeometry, ...]:
    by_contig: Dict[str, List[_CallRecord]] = {}
    for record in records:
        by_contig.setdefault(record.molecule.contig, []).append(record)
    geometries = []
    for contig in sorted(by_contig):
        contig_records = by_contig[contig]
        modes = _center_modes(contig_records, config)
        mode_groups = _assign_to_modes(contig_records, modes, config.assignment_radius_bp)
        for mode, mode_records in sorted(mode_groups.items()):
            for component in _edge_components(mode_records, config.edge_compatibility_bp):
                geometries.append(
                    _build_geometry(
                        contig,
                        component,
                        config,
                        locus_summit=mode,
                    )
                )

    # Separate center modes can converge to identical robust geometry.  Merge
    # those deterministically so one call can never support duplicate sites.
    # Exact-geometry bridges also union their source modes transitively: every
    # family learned under any bridged mode belongs to the same parent locus.
    # The leftmost source mode is the stable canonical locus summit.
    _ModeKey = Tuple[str, int]
    parent: Dict[_ModeKey, _ModeKey] = {}

    def find(mode_key: _ModeKey) -> _ModeKey:
        root = parent.setdefault(mode_key, mode_key)
        while root != parent[root]:
            root = parent[root]
        while mode_key != root:
            next_key = parent[mode_key]
            parent[mode_key] = root
            mode_key = next_key
        return root

    def union(left: _ModeKey, right: _ModeKey) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        canonical, other = sorted((left_root, right_root))
        parent[other] = canonical

    by_exact_geometry: Dict[Tuple[str, int, int, int], List[_SiteGeometry]] = {}
    for geometry in geometries:
        mode_key = (geometry.contig, geometry.locus_summit)
        parent.setdefault(mode_key, mode_key)
        geometry_key = (
            geometry.contig,
            geometry.start,
            geometry.end,
            geometry.summit,
        )
        by_exact_geometry.setdefault(geometry_key, []).append(geometry)

    for equivalent_geometries in by_exact_geometry.values():
        source_modes = [
            (geometry.contig, geometry.locus_summit) for geometry in equivalent_geometries
        ]
        for source_mode in source_modes[1:]:
            union(source_modes[0], source_mode)

    final = []
    for (contig, _start, _end, _summit), equivalent_geometries in sorted(by_exact_geometry.items()):
        source_mode = (
            equivalent_geometries[0].contig,
            equivalent_geometries[0].locus_summit,
        )
        records_for_site = [
            record for geometry in equivalent_geometries for record in geometry.records
        ]
        final.append(
            _build_geometry(
                contig,
                records_for_site,
                config,
                locus_summit=find(source_mode)[1],
            )
        )

    by_locus: Dict[Tuple[str, str], List[_SiteGeometry]] = {}
    for geometry in final:
        by_locus.setdefault((geometry.contig, geometry.locus_id), []).append(geometry)
    indexed = []
    for locus_key in sorted(by_locus):
        families = sorted(
            by_locus[locus_key],
            key=lambda value: (value.start, value.end, value.summit, value.site_id),
        )
        indexed.extend(
            replace(geometry, family_index=family_index)
            for family_index, geometry in enumerate(families, start=1)
        )
    return tuple(
        sorted(
            indexed,
            key=lambda value: (
                value.contig,
                value.start,
                value.end,
                value.locus_summit,
                value.family_index,
            ),
        )
    )


def _assign_all_calls(
    molecules: Sequence[_NormalizedMolecule],
    geometries: Sequence[_SiteGeometry],
    geometry_call_keys: Dict[Tuple[_MoleculeKey, str], str],
    config: SiteDiscoveryConfig,
) -> Tuple[TFCallAssignment, ...]:
    site_by_id = {geometry.site_id: geometry for geometry in geometries}
    by_contig: Dict[str, Tuple[_SiteGeometry, ...]] = {}
    summits_by_contig: Dict[str, Tuple[int, ...]] = {}
    for contig in sorted({geometry.contig for geometry in geometries}):
        contig_sites = tuple(
            sorted(
                (geometry for geometry in geometries if geometry.contig == contig),
                key=lambda value: (value.summit, value.start, value.end, value.site_id),
            )
        )
        by_contig[contig] = contig_sites
        summits_by_contig[contig] = tuple(value.summit for value in contig_sites)

    assignments = []
    for molecule in molecules:
        contig_sites = by_contig.get(molecule.contig, ())
        contig_summits = summits_by_contig.get(molecule.contig, ())
        for call in molecule.tfs:
            call_center = _interval_center_bin(call.start, call.end)
            geometry_site_id = geometry_call_keys.get((molecule.key, call.call_id))
            site = site_by_id.get(geometry_site_id) if geometry_site_id else None
            if site is None:
                left = bisect_left(contig_summits, call_center - config.assignment_radius_bp)
                right = bisect_right(contig_summits, call_center + config.assignment_radius_bp)
                candidates = [
                    value
                    for value in contig_sites[left:right]
                    if abs(call.start - value.start) <= config.edge_compatibility_bp
                    and abs(call.end - value.end) <= config.edge_compatibility_bp
                ]
                if candidates:
                    site = min(
                        candidates,
                        key=lambda value: (
                            abs(call_center - value.summit),
                            abs(call.start - value.start) + abs(call.end - value.end),
                            value.start,
                            value.end,
                            value.site_id,
                        ),
                    )
            fully_maps_site = bool(
                site
                and _molecule_fully_maps(
                    molecule,
                    site.start,
                    site.end,
                    config.minimum_mapped_fraction,
                )
            )
            assignments.append(
                TFCallAssignment(
                    call_id=call.call_id,
                    molecule_id=molecule.molecule_id,
                    contig=molecule.contig,
                    stratum=molecule.stratum,
                    start=call.start,
                    end=call.end,
                    site_id=None if site is None else site.site_id,
                    used_for_geometry=(molecule.key, call.call_id) in geometry_call_keys,
                    fully_maps_site=fully_maps_site,
                    msp_contains_site=bool(
                        site and fully_maps_site and _msp_contains(molecule, site.start, site.end)
                    ),
                )
            )
    return tuple(
        sorted(
            assignments,
            key=lambda value: (
                value.contig,
                value.stratum,
                value.molecule_id,
                value.start,
                value.end,
                value.call_id,
            ),
        )
    )


def _accelerated_site_coverage_counts(
    molecules: Sequence[_NormalizedMolecule],
    geometries: Sequence[_SiteGeometry],
    minimum_fraction: float,
) -> Optional[
    Tuple[Dict[str, Dict[str, int]], Dict[str, Dict[str, int]]]
]:
    """Return exact per-site/stratum counts through the compiled kernel."""
    if _site_coverage_counts_numba is None or not molecules or not geometries:
        return None
    # Compilation/array setup is not worthwhile for tiny targeted examples.
    if len(molecules) * len(geometries) < 10_000:
        return None
    strata = tuple(sorted({molecule.stratum for molecule in molecules}))
    stratum_to_index = {stratum: index for index, stratum in enumerate(strata)}

    def flatten(attribute: str):
        offsets = [0]
        starts = []
        ends = []
        for molecule in molecules:
            intervals = getattr(molecule, attribute)
            starts.extend(int(start) for start, _end in intervals)
            ends.extend(int(end) for _start, end in intervals)
            offsets.append(len(starts))
        return (
            np.asarray(offsets, dtype=np.int64),
            np.asarray(starts, dtype=np.int64),
            np.asarray(ends, dtype=np.int64),
        )

    block_offsets, block_starts, block_ends = flatten("mapped_blocks")
    msp_offsets, msp_starts, msp_ends = flatten("msps")
    mapped, msp = _site_coverage_counts_numba(
        np.asarray([geometry.start for geometry in geometries], dtype=np.int64),
        np.asarray([geometry.end for geometry in geometries], dtype=np.int64),
        block_offsets,
        block_starts,
        block_ends,
        msp_offsets,
        msp_starts,
        msp_ends,
        np.asarray(
            [stratum_to_index[molecule.stratum] for molecule in molecules],
            dtype=np.int64,
        ),
        len(strata),
        float(minimum_fraction),
    )
    mapped_by_site = {}
    msp_by_site = {}
    for site_index, geometry in enumerate(geometries):
        mapped_by_site[geometry.site_id] = {
            stratum: int(mapped[site_index, stratum_index])
            for stratum_index, stratum in enumerate(strata)
            if mapped[site_index, stratum_index]
        }
        msp_by_site[geometry.site_id] = {
            stratum: int(msp[site_index, stratum_index])
            for stratum_index, stratum in enumerate(strata)
            if msp[site_index, stratum_index]
        }
    return mapped_by_site, msp_by_site


def _summarize_sites(
    molecules: Sequence[_NormalizedMolecule],
    geometries: Sequence[_SiteGeometry],
    assignments: Sequence[TFCallAssignment],
    config: SiteDiscoveryConfig,
) -> Tuple[TFPopulationSite, ...]:
    assignments_by_site: Dict[str, List[TFCallAssignment]] = {}
    for assignment in assignments:
        if assignment.site_id is not None:
            assignments_by_site.setdefault(assignment.site_id, []).append(assignment)

    accelerated_counts = _accelerated_site_coverage_counts(
        molecules,
        geometries,
        config.minimum_mapped_fraction,
    )
    if accelerated_counts is None:
        mapped_by_site: Dict[str, Dict[str, int]] = {
            geometry.site_id: {} for geometry in geometries
        }
        msp_by_site: Dict[str, Dict[str, int]] = {
            geometry.site_id: {} for geometry in geometries
        }
    else:
        mapped_by_site, msp_by_site = accelerated_counts
    geometry_by_id = {geometry.site_id: geometry for geometry in geometries}
    geometries_by_contig: Dict[str, List[_SiteGeometry]] = {}
    for geometry in geometries:
        geometries_by_contig.setdefault(geometry.contig, []).append(geometry)

    if accelerated_counts is None:
        # Query true interval overlaps; a single anomalously wide footprint
        # cannot expand every molecule's candidate window as a max-width
        # heuristic would.  This exact Python implementation is also the
        # reference fallback for installations without numba.
        site_indexes = {
            contig: _build_site_interval_index(contig_sites)
            for contig, contig_sites in geometries_by_contig.items()
        }
        for molecule in molecules:
            site_index = site_indexes.get(molecule.contig)
            if site_index is None or not molecule.mapped_blocks:
                continue
            candidate_site_ids = set()
            for block_start, block_end in molecule.mapped_blocks:
                _query_site_interval_index(
                    site_index,
                    block_start,
                    block_end,
                    candidate_site_ids,
                )
            for site_id in sorted(candidate_site_ids):
                geometry = geometry_by_id[site_id]
                if not _molecule_fully_maps(
                    molecule,
                    geometry.start,
                    geometry.end,
                    config.minimum_mapped_fraction,
                ):
                    continue
                mapped_counts = mapped_by_site[geometry.site_id]
                mapped_counts[molecule.stratum] = (
                    mapped_counts.get(molecule.stratum, 0) + 1
                )
                if _msp_contains(molecule, geometry.start, geometry.end):
                    msp_counts = msp_by_site[geometry.site_id]
                    msp_counts[molecule.stratum] = (
                        msp_counts.get(molecule.stratum, 0) + 1
                    )

    sites = []
    for geometry in geometries:
        mapped_by_stratum = mapped_by_site[geometry.site_id]
        msp_by_stratum = msp_by_site[geometry.site_id]

        site_assignments = assignments_by_site.get(geometry.site_id, [])
        tf_by_stratum: Dict[str, set] = {}
        tf_msp_by_stratum: Dict[str, set] = {}
        for assignment in site_assignments:
            if not assignment.fully_maps_site:
                continue
            key = (assignment.contig, assignment.stratum, assignment.molecule_id)
            tf_by_stratum.setdefault(assignment.stratum, set()).add(key)
            if assignment.msp_contains_site:
                tf_msp_by_stratum.setdefault(assignment.stratum, set()).add(key)

        all_strata = sorted(
            set(mapped_by_stratum)
            | set(msp_by_stratum)
            | set(tf_by_stratum)
            | set(tf_msp_by_stratum)
            | {value[0] for value in geometry.stratum_geometry}
        )
        geometry_by_stratum = {value[0]: value for value in geometry.stratum_geometry}
        stratum_summaries = []
        n_mapped_all = 0
        n_msp_all = 0
        n_tf_all = 0
        n_tf_msp_all = 0
        for stratum in all_strata:
            n_mapped = mapped_by_stratum.get(stratum, 0)
            n_msp = msp_by_stratum.get(stratum, 0)
            tf = tf_by_stratum.get(stratum, set())
            tf_msp = tf_msp_by_stratum.get(stratum, set())
            n_tf = len(tf)
            n_tf_msp = len(tf_msp)
            n_mapped_all += n_mapped
            n_msp_all += n_msp
            n_tf_all += n_tf
            n_tf_msp_all += n_tf_msp
            geometry_values = geometry_by_stratum.get(
                stratum,
                (stratum, 0, geometry.start, geometry.end, 0.0, 0.0),
            )
            stratum_summaries.append(
                StratumSiteSummary(
                    stratum=stratum,
                    cluster_support_molecules=int(geometry_values[1]),
                    n_fully_mapped=n_mapped,
                    n_tf=n_tf,
                    n_msp=n_msp,
                    n_tf_msp=n_tf_msp,
                    median_start=int(geometry_values[2]),
                    median_end=int(geometry_values[3]),
                    start_mad=float(geometry_values[4]),
                    end_mad=float(geometry_values[5]),
                )
            )

        n_tf_no_msp = n_tf_all - n_tf_msp_all
        n_no_tf_msp = n_msp_all - n_tf_msp_all
        n_no_tf_no_msp = n_mapped_all - n_tf_all - n_msp_all + n_tf_msp_all
        cluster_support = len({record.molecule.key for record in geometry.records})
        sites.append(
            TFPopulationSite(
                site_id=geometry.site_id,
                locus_id=geometry.locus_id,
                contig=geometry.contig,
                start=geometry.start,
                end=geometry.end,
                locus_summit=geometry.locus_summit,
                summit=geometry.summit,
                family_index=geometry.family_index,
                assigned_call_count=len(site_assignments),
                geometry_call_count=len(geometry.records),
                cluster_support_molecules=cluster_support,
                n_fully_mapped=n_mapped_all,
                n_tf=n_tf_all,
                n_msp=n_msp_all,
                n_tf_msp=n_tf_msp_all,
                n_tf_no_msp=n_tf_no_msp,
                n_no_tf_msp=n_no_tf_msp,
                n_no_tf_no_msp=n_no_tf_no_msp,
                start_mad=geometry.start_mad,
                end_mad=geometry.end_mad,
                geometry_ready=(cluster_support >= config.minimum_geometry_support),
                population_ready=n_tf_all >= config.minimum_population_support,
                strata=tuple(stratum_summaries),
            )
        )
    return tuple(sites)


def _build_model_loci(
    sites: Sequence[TFPopulationSite],
) -> Tuple[TFModelLocus, ...]:
    grouped: Dict[Tuple[str, str], List[TFPopulationSite]] = {}
    for site in sites:
        grouped.setdefault((site.contig, site.locus_id), []).append(site)
    loci = []
    for (contig, locus_id), families in sorted(grouped.items()):
        ordered = sorted(families, key=lambda value: value.family_index)
        if [family.family_index for family in ordered] != list(range(1, len(ordered) + 1)):
            raise ValueError(f"non-contiguous family indices for locus {locus_id}")
        summits = {family.locus_summit for family in ordered}
        if len(summits) != 1:
            raise ValueError(f"conflicting summits for locus {locus_id}")
        loci.append(
            TFModelLocus(
                locus_id=locus_id,
                contig=contig,
                start=min(family.start for family in ordered),
                end=max(family.end for family in ordered),
                summit=ordered[0].locus_summit,
                family_site_ids=tuple(family.site_id for family in ordered),
            )
        )
    return tuple(
        sorted(
            loci,
            key=lambda value: (
                value.contig,
                value.start,
                value.end,
                value.summit,
                value.locus_id,
            ),
        )
    )


def build_tf_site_catalog(
    molecules: Iterable[BaselineMolecule],
    *,
    config: SiteDiscoveryConfig = SiteDiscoveryConfig(),
) -> TFSiteCatalog:
    """Build a deterministic baseline TF population catalog.

    Site discovery uses only ordinary, geometry-eligible TF locations.  There is
    no TQ input, local-background rejection, support rejection, or top-N cap.
    Sparse eligible calls therefore remain valid support-one sites. Other calls
    remain auditable and may join a compatible eligible-learned site. Geometry
    and population readiness are separate descriptive labels; neither filters
    the catalog.
    """

    normalized, input_records, raw_tf_calls = _coalesce_molecules(molecules)
    unique_tf_calls = sum(len(molecule.tfs) for molecule in normalized)
    geometry_records = tuple(
        _CallRecord(molecule=molecule, call=call)
        for molecule in normalized
        for call in molecule.tfs
        if call.geometry_eligible
        and _molecule_fully_maps(
            molecule,
            call.start,
            call.end,
            config.minimum_mapped_fraction,
        )
    )
    geometries = _discover_geometry(geometry_records, config)
    geometry_call_keys = {
        (record.molecule.key, record.call.call_id): geometry.site_id
        for geometry in geometries
        for record in geometry.records
    }
    assignments = _assign_all_calls(normalized, geometries, geometry_call_keys, config)
    sites = _summarize_sites(normalized, geometries, assignments, config)
    loci = _build_model_loci(sites)
    assigned_tf_calls = sum(assignment.site_id is not None for assignment in assignments)
    return TFSiteCatalog(
        loci=loci,
        sites=sites,
        assignments=assignments,
        config=config,
        diagnostics=SiteDiscoveryDiagnostics(
            input_records=input_records,
            unique_molecules=len(normalized),
            raw_tf_calls=raw_tf_calls,
            unique_tf_calls=unique_tf_calls,
            geometry_tf_calls=len(geometry_records),
            assigned_tf_calls=assigned_tf_calls,
            unassigned_tf_calls=unique_tf_calls - assigned_tf_calls,
        ),
    )


# Preferred public vocabulary: the complete learned artifact is a footprint
# population model, while each geometry family is a TF-binding hypothesis.
# Earlier TF-model/site-discovery names remain aliases for prototype callers.
FootprintObservation = TFObservation
FootprintModelLocus = TFModelLocus
BindingHypothesis = TFPopulationSite
BindingHypothesisAssignment = TFCallAssignment
FootprintPopulationModel = TFSiteCatalog
TFModelFamily = TFPopulationSite
TFModelAssignment = TFCallAssignment
TFModelCatalog = TFSiteCatalog


def build_tf_model_catalog(
    molecules: Iterable[BaselineMolecule],
    *,
    config: SiteDiscoveryConfig = SiteDiscoveryConfig(),
) -> TFModelCatalog:
    """Build a data-derived TF model catalog from raw footprint observations."""

    return build_tf_site_catalog(molecules, config=config)


def build_footprint_population_model(
    molecules: Iterable[BaselineMolecule],
    *,
    config: SiteDiscoveryConfig = SiteDiscoveryConfig(),
) -> FootprintPopulationModel:
    """Infer data-derived TF-binding hypotheses from footprint observations."""

    return build_tf_site_catalog(molecules, config=config)


__all__ = [
    "BaselineMolecule",
    "BindingHypothesis",
    "BindingHypothesisAssignment",
    "FootprintModelLocus",
    "FootprintObservation",
    "FootprintPopulationModel",
    "SiteDiscoveryConfig",
    "SiteDiscoveryDiagnostics",
    "StratumSiteSummary",
    "TFCallAssignment",
    "TFModelAssignment",
    "TFModelCatalog",
    "TFModelFamily",
    "TFModelLocus",
    "TFObservation",
    "TFPopulationSite",
    "TFSiteCatalog",
    "build_tf_model_catalog",
    "build_tf_site_catalog",
    "build_footprint_population_model",
]
