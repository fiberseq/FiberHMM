"""Targeted, coverage-aware discovery of recurrent footprint families.

This module implements the first phase of the targeted-family workflow.  It
uses ordinary ``tf`` and ``msp`` annotations only, identifies short genomic
windows containing recurrent long MSPs, and learns family geometry from a
deterministic MSP-enriched molecule cohort.  The enriched cohort is explicitly
*not* an occupancy sample.  Quantification and molecule-level rescue must use
the complete unbiased cohort after the geometry catalog has been frozen.

Windows are independent discovery tasks and may be evaluated in separate
processes.  Reconciliation and compact family-ID allocation remain global,
serial operations so output is independent of worker completion order.
"""

from __future__ import annotations

import hashlib
import math
import multiprocessing
import time
from bisect import bisect_left, bisect_right
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, replace
from numbers import Real
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pysam

from fiberhmm.cli.dedup import cluster_reads
from fiberhmm.cli.extract_tags import _deam_positions_list
from fiberhmm.core.bam_reader import cigar_to_query_ref
from fiberhmm.inference.tf_sites import (
    BaselineMolecule,
    SiteDiscoveryConfig,
    build_footprint_population_model,
)
from fiberhmm.inference.mp_context import _MP_CONTEXT


Interval = Tuple[int, int]


@dataclass(frozen=True)
class TargetedFamilyChemistry:
    """Chemistry-aware geometry tolerances for targeted family discovery."""

    maximum_boundary_delta: int
    maximum_width_delta: int
    maximum_center_delta: float
    minimum_shorter_overlap_fraction: float
    boundary_search_radius: int
    minimum_molecule_opportunities: int


CHEMISTRY_PROFILES: Mapping[str, TargetedFamilyChemistry] = {
    "ddda": TargetedFamilyChemistry(18, 18, 12.0, 0.75, 4, 3),
    "dddb": TargetedFamilyChemistry(28, 32, 18.0, 0.55, 8, 2),
    "hia5-pacbio": TargetedFamilyChemistry(24, 28, 16.0, 0.60, 7, 2),
    "hia5-nanopore": TargetedFamilyChemistry(24, 28, 16.0, 0.60, 7, 2),
}


@dataclass(frozen=True)
class TargetedFamilyDiscoveryConfig:
    """Parameters for the annotation-only discovery phase.

    ``minimum_informative_fraction`` is applied to molecules that fully map a
    core.  A core is informative when at least the larger of that fraction and
    ``minimum_informative_molecules`` carry an MSP of at least
    ``minimum_nfr_length`` overlapping the core.  The default one-percent gate
    is deliberately sensitive; final evidence thresholds are applied only on
    the unbiased cohort.
    """

    core_size: int = 1000
    halo_size: int = 200
    minimum_nfr_length: int = 150
    minimum_informative_molecules: int = 3
    minimum_informative_fraction: float = 0.01
    maximum_discovery_molecules: int = 500
    minimum_family_support: int = 3
    minimum_family_fraction: float = 0.05
    seed: str = "fiberhmm-targeted-family-discovery-v1"

    def __post_init__(self) -> None:
        for name in (
            "core_size",
            "halo_size",
            "minimum_nfr_length",
            "minimum_informative_molecules",
            "maximum_discovery_molecules",
            "minimum_family_support",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.core_size < self.minimum_nfr_length:
            raise ValueError("core_size must be at least minimum_nfr_length")
        fraction = self.minimum_informative_fraction
        if (
            isinstance(fraction, bool)
            or not isinstance(fraction, Real)
            or not math.isfinite(float(fraction))
            or not 0.0 <= float(fraction) <= 1.0
        ):
            raise ValueError("minimum_informative_fraction must be in [0,1]")
        family_fraction = self.minimum_family_fraction
        if (
            isinstance(family_fraction, bool)
            or not isinstance(family_fraction, Real)
            or not math.isfinite(float(family_fraction))
            or not 0.0 <= float(family_fraction) <= 1.0
        ):
            raise ValueError("minimum_family_fraction must be in [0,1]")
        if not self.seed:
            raise ValueError("seed must be non-empty")


@dataclass(frozen=True)
class TargetedFamilyWindow:
    """One core, its evidence halo, and its annotation-only NFR gate."""

    ordinal: int
    contig: str
    core_start: int
    core_end: int
    halo_start: int
    halo_end: int
    fully_mapped_molecules: int
    long_msp_molecules: int
    required_long_msp_molecules: int
    informative: bool

    @property
    def label(self) -> str:
        return f"{self.contig}:{self.core_start}-{self.core_end}"


@dataclass(frozen=True)
class TargetedFootprintFamily:
    """One globally reconciled family geometry learned during discovery."""

    family_id: str
    contig: str
    start: int
    end: int
    seed_intervals: Tuple[Interval, ...]
    member_site_ids: Tuple[str, ...]
    discovery_support_molecules: int
    discovery_denominator_molecules: int
    source_window_ordinals: Tuple[int, ...]

    @property
    def center(self) -> float:
        return (self.start + self.end) / 2.0

    @property
    def width(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class TargetedWindowResult:
    """Deterministic result and provenance for one informative window."""

    window: TargetedFamilyWindow
    selected_molecule_ids: Tuple[str, ...]
    selected_strata: Tuple[Tuple[str, int], ...]
    available_discovery_molecules: int
    tf_call_count: int
    elapsed_seconds: float
    families: Tuple[TargetedFootprintFamily, ...]


@dataclass(frozen=True)
class TargetedFamilyDiscovery:
    """Frozen geometry catalog plus complete discovery-cohort provenance."""

    schema: str
    contig: str
    locus_start: int
    locus_end: int
    chemistry: str
    config: TargetedFamilyDiscoveryConfig
    windows: Tuple[TargetedFamilyWindow, ...]
    window_results: Tuple[TargetedWindowResult, ...]
    families: Tuple[TargetedFootprintFamily, ...]
    input_molecules: int

    def as_dict(self) -> dict:
        return {
            "schema": self.schema,
            "contig": self.contig,
            "locus": [self.locus_start, self.locus_end],
            "chemistry": self.chemistry,
            "config": asdict(self.config),
            "input_molecules": self.input_molecules,
            "contracts": {
                "discovery_cohort_role": "geometry_only_msp_enriched",
                "occupancy_cohort": "full_unbiased_required",
                "raw_calls_mutated": False,
                "window_parallelism": "independent_core_plus_halo_tasks",
                "global_reconciliation": "canonical_serial_order",
            },
            "windows": [asdict(window) for window in self.windows],
            "window_results": [
                {
                    "window_ordinal": result.window.ordinal,
                    "selected_molecule_ids": list(result.selected_molecule_ids),
                    "selected_strata": dict(result.selected_strata),
                    "available_discovery_molecules": result.available_discovery_molecules,
                    "tf_call_count": result.tf_call_count,
                    "elapsed_seconds": result.elapsed_seconds,
                    "family_count": len(result.families),
                }
                for result in self.window_results
            ],
            "families": [
                {
                    **asdict(family),
                    "seed_intervals": [list(value) for value in family.seed_intervals],
                    "source_window_ordinals": list(family.source_window_ordinals),
                }
                for family in self.families
            ],
        }


@dataclass(frozen=True)
class _BoundaryContribution:
    molecule_id: Tuple[str, str, str]
    strand: str
    opportunities: int
    family_log_predictive: float
    null_log_predictive: float
    log_bayes_factor: float
    cpu_replayed: bool = False


_PRESCRIBED_BOUNDARY_WORKER_MODEL = None
_PRESCRIBED_BOUNDARY_WORKER_MODE = None
_PRESCRIBED_BOUNDARY_WORKER_MODELS = None
_PRESCRIBED_BOUNDARY_PRECOMPUTED_SETTINGS = None


def _initialize_prescribed_boundary_worker(model, evidence_summation_mode):
    global _PRESCRIBED_BOUNDARY_WORKER_MODEL
    global _PRESCRIBED_BOUNDARY_WORKER_MODE
    _PRESCRIBED_BOUNDARY_WORKER_MODEL = model
    _PRESCRIBED_BOUNDARY_WORKER_MODE = evidence_summation_mode


def _initialize_prescribed_boundary_families_worker(
    models, evidence_summation_mode
):
    global _PRESCRIBED_BOUNDARY_WORKER_MODELS
    global _PRESCRIBED_BOUNDARY_WORKER_MODE
    _PRESCRIBED_BOUNDARY_WORKER_MODELS = tuple(models)
    _PRESCRIBED_BOUNDARY_WORKER_MODE = evidence_summation_mode


def _initialize_prescribed_boundary_precomputed_worker(
    models,
    evidence_summation_mode,
    cuda_interval_chunk_size,
    replay_cutoffs,
    cuda_replay_guard_nats,
):
    global _PRESCRIBED_BOUNDARY_WORKER_MODELS
    global _PRESCRIBED_BOUNDARY_WORKER_MODE
    global _PRESCRIBED_BOUNDARY_PRECOMPUTED_SETTINGS
    _PRESCRIBED_BOUNDARY_WORKER_MODELS = tuple(models)
    _PRESCRIBED_BOUNDARY_WORKER_MODE = evidence_summation_mode
    _PRESCRIBED_BOUNDARY_PRECOMPUTED_SETTINGS = (
        int(cuda_interval_chunk_size),
        tuple(replay_cutoffs),
        float(cuda_replay_guard_nats),
    )


def _prepare_prescribed_boundary_model(model):
    """Attach a private reusable spatial-null grid to one frozen model."""

    if model.get("_prepared_spatial_null_grid") is not None:
        return model
    from fiberhmm.inference.strand_rescue import (
        IntervalCall,
        _prepare_spatial_null_interval_grid,
        _spatial_null_candidate_intervals,
    )

    envelope = IntervalCall(int(model["envelope"][0]), int(model["envelope"][1]))
    spatial_intervals = _spatial_null_candidate_intervals(
        envelope,
        [[tuple(interval) for interval in model["spatial_null_exclusion_intervals"]]],
        minimum_width=int(model["spatial_null_minimum_width"]),
        maximum_width=int(model["spatial_null_maximum_width"]),
    )
    starts, ends, log_prior = _prepare_spatial_null_interval_grid(
        spatial_intervals
    )
    prepared = dict(model)
    prepared["_prepared_spatial_null_grid"] = (
        starts,
        ends,
        log_prior,
        len(spatial_intervals),
    )
    return prepared


def _eligible_boundary_reads(reads, model):
    from fiberhmm.inference.strand_rescue import (
        IntervalCall,
        _read_evidence_content_sha256,
    )

    envelope = IntervalCall(int(model["envelope"][0]), int(model["envelope"][1]))
    minimum_opportunities = int(model["minimum_molecule_opportunities"])
    by_molecule = {}
    for read in reads:
        if not read.fully_maps(envelope.start, envelope.end):
            continue
        opportunities = read.interval_evidence(envelope.start, envelope.end)[1]
        if opportunities < minimum_opportunities:
            continue
        previous = by_molecule.get(read.molecule_id)
        if previous is None:
            by_molecule[read.molecule_id] = read
            continue
        previous_opportunities = previous.interval_evidence(
            envelope.start, envelope.end
        )[1]
        if opportunities > previous_opportunities or (
            opportunities == previous_opportunities
            and _read_evidence_content_sha256(read)
            < _read_evidence_content_sha256(previous)
        ):
            by_molecule[read.molecule_id] = read
    return tuple(by_molecule[key] for key in sorted(by_molecule))


def _precompute_unique_boundary_eligibility(
    reads, models, precomputed_opportunities=None
):
    """Return exact eligibility masks and opportunity counts for unique reads.

    The joint production scorer has already established that every molecule ID
    occurs once. A vectorized reference-span rejection therefore avoids the
    dominant Python cost of asking every read about every family, while the
    surviving candidates still use :meth:`ReadEvidence.fully_maps` so gapped
    alignments retain the reference implementation's exact semantics.
    """

    read_count = len(reads)
    family_count = len(models)
    eligible = np.zeros((family_count, read_count), dtype=bool)
    opportunities = np.zeros((family_count, read_count), dtype=np.int32)
    if not reads or not models:
        return eligible, opportunities
    if precomputed_opportunities is not None:
        precomputed_opportunities = np.asarray(
            precomputed_opportunities, dtype=np.int64
        )
        if precomputed_opportunities.shape != (family_count, read_count):
            raise ValueError("precomputed opportunity matrix must align")

    # Most long-read alignments consist of one contiguous reference block. For
    # those, interval coverage is exactly a pair of vectorized comparisons.
    # Only genuinely gapped or unusual block encodings need the full reference
    # mapping implementation below.
    simple = np.zeros(read_count, dtype=bool)
    simple_starts = np.zeros(read_count, dtype=np.int64)
    simple_ends = np.zeros(read_count, dtype=np.int64)
    complex_rows = []
    for row_index, read in enumerate(reads):
        blocks = read.alignment_blocks
        if blocks is None:
            simple[row_index] = True
            simple_starts[row_index] = int(read.ref_start)
            simple_ends[row_index] = int(read.ref_end)
        elif (
            isinstance(blocks, tuple)
            and len(blocks) == 1
            and isinstance(blocks[0], tuple)
            and len(blocks[0]) == 2
            and type(blocks[0][0]) is int
            and type(blocks[0][1]) is int
            and blocks[0][0] < blocks[0][1]
        ):
            simple[row_index] = True
            simple_starts[row_index] = blocks[0][0]
            simple_ends[row_index] = blocks[0][1]
        else:
            complex_rows.append(row_index)
    for model_index, model in enumerate(models):
        start, end = (int(value) for value in model["envelope"])
        minimum = int(model["minimum_molecule_opportunities"])
        fully_mapped = simple & (simple_starts <= start) & (simple_ends >= end)
        for row_index in complex_rows:
            if reads[row_index].fully_maps(start, end):
                fully_mapped[row_index] = True
        candidate_rows = np.flatnonzero(fully_mapped)
        if precomputed_opportunities is not None:
            counts = precomputed_opportunities[model_index, candidate_rows]
            retained = candidate_rows[counts >= minimum]
            eligible[model_index, retained] = True
            opportunities[model_index, retained] = counts[counts >= minimum]
            continue
        for row_index in candidate_rows:
            read = reads[int(row_index)]
            left = int(np.searchsorted(read.positions, start, side="left"))
            right = int(np.searchsorted(read.positions, end, side="left"))
            count = right - left
            if count < minimum:
                continue
            eligible[model_index, row_index] = True
            opportunities[model_index, row_index] = count
    return eligible, opportunities


def _cuda_locality_family_batches(reads, models, maximum_span_bp):
    """Group nearby families with only reads that can overlap their span.

    A chromosome chunk is sparse: a fiber cannot support a family hundreds of
    kilobases away. Dense all-family/all-read CUDA matrices therefore waste
    both compute and VRAM. Families are owned by exactly one deterministic
    coordinate batch, while a long fiber may occur in multiple neighboring
    batches; each molecule-family likelihood is still evaluated exactly once.
    """

    if (
        isinstance(maximum_span_bp, bool)
        or not isinstance(maximum_span_bp, int)
        or maximum_span_bp < 1
    ):
        raise ValueError("CUDA family batch span must be a positive integer")
    ordered_indices = sorted(
        range(len(models)),
        key=lambda index: (
            int(models[index]["envelope"][0]),
            int(models[index]["envelope"][1]),
            index,
        ),
    )
    family_groups = []
    current = []
    group_start = None
    group_end = None
    for model_index in ordered_indices:
        start, end = (int(value) for value in models[model_index]["envelope"])
        if current and max(group_end, end) - group_start > maximum_span_bp:
            family_groups.append((tuple(current), group_start, group_end))
            current = []
            group_start = None
            group_end = None
        current.append(model_index)
        group_start = start if group_start is None else min(group_start, start)
        group_end = end if group_end is None else max(group_end, end)
    if current:
        family_groups.append((tuple(current), group_start, group_end))

    reference_starts = np.fromiter(
        (int(read.ref_start) for read in reads),
        dtype=np.int64,
        count=len(reads),
    )
    reference_ends = np.fromiter(
        (int(read.ref_end) for read in reads),
        dtype=np.int64,
        count=len(reads),
    )
    batches = []
    for model_indices, start, end in family_groups:
        read_rows = np.flatnonzero(
            (reference_starts < end) & (reference_ends > start)
        )
        batches.append(
            (
                model_indices,
                tuple(reads[int(index)] for index in read_rows),
                (int(start), int(end)),
            )
        )
    return tuple(batches)


def _prescribed_boundary_precomputed_contributions(
    reads,
    model,
    family_scores,
    spatial_scores,
    diffuse_scores,
    opportunities,
    *,
    summation_mode,
    cpu_replay_cutoffs,
    cpu_replay_guard_nats,
):
    """Assemble exact contributions after all three likelihoods ran on CUDA."""

    from fiberhmm.inference.strand_rescue import _logsumexp

    if not reads:
        return ()
    family_log_predictive = np.asarray(family_scores, dtype=np.float64)
    spatial_scores = np.asarray(spatial_scores, dtype=np.float64)
    diffuse_scores = np.asarray(diffuse_scores, dtype=np.float64)
    opportunities = np.asarray(opportunities, dtype=np.int32)
    expected = (len(reads),)
    if not all(
        values.shape == expected
        for values in (
            family_log_predictive,
            spatial_scores,
            diffuse_scores,
            opportunities,
        )
    ):
        raise ValueError("precomputed CUDA likelihood vectors must align")
    null_log_predictive = (
        _logsumexp(
            np.column_stack(
                (
                    np.zeros(len(reads), dtype=np.float64),
                    spatial_scores,
                    diffuse_scores,
                )
            ),
            axis=1,
        )
        - math.log(3.0)
    )
    log_bayes_factors = family_log_predictive - null_log_predictive
    cpu_replayed = np.zeros(len(reads), dtype=bool)
    if cpu_replay_cutoffs and cpu_replay_guard_nats > 0.0:
        replay_distance = np.full(len(reads), math.inf, dtype=np.float64)
        for cutoff in cpu_replay_cutoffs:
            if math.isfinite(float(cutoff)):
                replay_distance = np.minimum(
                    replay_distance,
                    np.abs(log_bayes_factors - float(cutoff)),
                )
        cpu_replayed = replay_distance <= cpu_replay_guard_nats
        if np.any(cpu_replayed):
            # Copy only when values will be changed. The common path remains a
            # zero-copy view of the detached device result.
            family_log_predictive = family_log_predictive.copy()
            null_log_predictive = null_log_predictive.copy()
            log_bayes_factors = log_bayes_factors.copy()
            replay_indices = np.flatnonzero(cpu_replayed)
            replay_reads = tuple(reads[int(index)] for index in replay_indices)
            replay_contributions, _unused_geometry = _prescribed_boundary_chunk(
                (
                    replay_reads,
                    model,
                    summation_mode,
                    False,
                    "cpu",
                )
            )
            replay_by_id = {
                value.molecule_id: value for value in replay_contributions
            }
            for replay_index in replay_indices:
                reference = replay_by_id[reads[int(replay_index)].molecule_id]
                family_log_predictive[replay_index] = reference.family_log_predictive
                null_log_predictive[replay_index] = reference.null_log_predictive
                log_bayes_factors[replay_index] = reference.log_bayes_factor
    return tuple(
        _BoundaryContribution(
            molecule_id=read.molecule_id,
            strand=read.strand,
            opportunities=int(opportunities[index]),
            family_log_predictive=float(family_log_predictive[index]),
            null_log_predictive=float(null_log_predictive[index]),
            log_bayes_factor=float(log_bayes_factors[index]),
            cpu_replayed=bool(cpu_replayed[index]),
        )
        for index, read in enumerate(reads)
    )


def _prescribed_boundary_chunk(payload):
    """Return sufficient statistics under prescribed, cohort-neutral weights."""

    reads, model, summation_mode, include_geometry = payload[:4]
    likelihood_backend = payload[4] if len(payload) >= 5 else "cpu"
    cuda_batch = payload[5] if len(payload) >= 6 else None
    cuda_interval_chunk_size = int(payload[6]) if len(payload) >= 7 else 512
    cpu_replay_cutoffs = tuple(payload[7]) if len(payload) >= 8 else ()
    cpu_replay_guard_nats = float(payload[8]) if len(payload) >= 9 else 0.0
    precomputed_spatial_scores = payload[9] if len(payload) >= 10 else None
    precomputed_family_scores = payload[10] if len(payload) >= 11 else None
    precomputed_diffuse_scores = payload[11] if len(payload) >= 12 else None
    from fiberhmm.inference.strand_rescue import (
        IntervalCall,
        _diffuse_unmodeled_configuration_log_likelihood,
        _interval_evidence_matrix,
        _logsumexp,
        _softmax,
        _spatial_null_candidate_intervals,
        _spatial_null_configuration_log_likelihoods,
        _spatial_null_log_likelihoods_from_prepared_grid,
        _weighted_integer_quantile,
    )

    if not reads:
        return (), ()
    envelope = IntervalCall(int(model["envelope"][0]), int(model["envelope"][1]))
    candidates = [
        (int(interval[0]), int(interval[1])) for interval in model["candidate_intervals"]
    ]
    classes = list(model["geometry_classes"])
    class_members = []
    for class_index, record in enumerate(classes):
        members = np.asarray(record["member_candidate_indices"], dtype=np.int64)
        if members.size == 0 or np.any(members < 0) or np.any(members >= len(candidates)):
            raise ValueError("frozen boundary-family geometry class is invalid")
        class_members.append(members)
    if precomputed_family_scores is not None and not include_geometry:
        precomputed_family_scores = np.asarray(
            precomputed_family_scores, dtype=np.float64
        )
        if cuda_batch is None:
            if precomputed_family_scores.shape != (len(reads),):
                raise ValueError(
                    "detached precomputed family scores must align with retained reads"
                )
            family_log_predictive = precomputed_family_scores
        else:
            family_log_predictive = precomputed_family_scores[
                cuda_batch.row_indices(reads)
            ]
        candidate_scores = left_indices = right_indices = geometry_scores = None
    else:
        candidate_scores, left_indices, right_indices = _interval_evidence_matrix(
            reads, candidates, summation_mode=summation_mode
        )
        geometry_scores = np.empty((len(reads), len(classes)), dtype=np.float64)
        for class_index, members in enumerate(class_members):
            geometry_scores[:, class_index] = (
                _logsumexp(candidate_scores[:, members], axis=1)
                - math.log(members.size)
            )
        family_log_predictive = (
            _logsumexp(geometry_scores, axis=1) - math.log(len(classes))
        )
    prepared_spatial = model.get("_prepared_spatial_null_grid")
    if prepared_spatial is None:
        spatial_intervals = _spatial_null_candidate_intervals(
            envelope,
            [[tuple(interval) for interval in model["spatial_null_exclusion_intervals"]]],
            minimum_width=int(model["spatial_null_minimum_width"]),
            maximum_width=int(model["spatial_null_maximum_width"]),
        )
        spatial_scores = _spatial_null_configuration_log_likelihoods(
            reads, spatial_intervals
        )
    elif likelihood_backend == "cuda":
        starts, ends, log_prior, _interval_count = prepared_spatial
        if precomputed_spatial_scores is not None:
            precomputed_spatial_scores = np.asarray(
                precomputed_spatial_scores, dtype=np.float64
            )
            if cuda_batch is None:
                if precomputed_spatial_scores.shape != (len(reads),):
                    raise ValueError(
                        "detached precomputed CUDA scores must align with retained reads"
                    )
                spatial_scores = precomputed_spatial_scores
            else:
                spatial_scores = precomputed_spatial_scores[
                    cuda_batch.row_indices(reads)
                ]
        else:
            if cuda_batch is None:
                from fiberhmm.inference.cuda_likelihood import (
                    prepare_torch_spatial_null_batch,
                )

                cuda_batch = prepare_torch_spatial_null_batch(reads)
            spatial_scores = cuda_batch.spatial_null_log_likelihoods(
                reads,
                starts,
                ends,
                log_prior,
                interval_chunk_size=cuda_interval_chunk_size,
            )
    else:
        starts, ends, log_prior, _interval_count = prepared_spatial
        spatial_scores = _spatial_null_log_likelihoods_from_prepared_grid(
            reads, starts, ends, log_prior
        )
    if precomputed_diffuse_scores is not None:
        precomputed_diffuse_scores = np.asarray(
            precomputed_diffuse_scores, dtype=np.float64
        )
        if cuda_batch is None:
            if precomputed_diffuse_scores.shape != (len(reads),):
                raise ValueError(
                    "detached precomputed diffuse scores must align with retained reads"
                )
            diffuse_scores = precomputed_diffuse_scores
        else:
            diffuse_scores = precomputed_diffuse_scores[
                cuda_batch.row_indices(reads)
            ]
    else:
        diffuse_scores = np.asarray(
            [
                _diffuse_unmodeled_configuration_log_likelihood(read, envelope)
                for read in reads
            ],
            dtype=np.float64,
        )
    null_log_predictive = (
        _logsumexp(
            np.column_stack(
                (
                    np.zeros(len(reads), dtype=np.float64),
                    spatial_scores,
                    diffuse_scores,
                )
            ),
            axis=1,
        )
        - math.log(3.0)
    )
    log_bayes_factors = family_log_predictive - null_log_predictive
    cpu_replayed = np.zeros(len(reads), dtype=bool)
    if (
        likelihood_backend == "cuda"
        and prepared_spatial is not None
        and cpu_replay_cutoffs
        and cpu_replay_guard_nats > 0.0
    ):
        replay_distance = np.full(len(reads), math.inf, dtype=np.float64)
        for cutoff in cpu_replay_cutoffs:
            if math.isfinite(float(cutoff)):
                replay_distance = np.minimum(
                    replay_distance, np.abs(log_bayes_factors - float(cutoff))
                )
        cpu_replayed = replay_distance <= cpu_replay_guard_nats
        if np.any(cpu_replayed):
            replay_indices = np.flatnonzero(cpu_replayed)
            replay_reads = tuple(reads[int(index)] for index in replay_indices)
            replay_contributions, _unused_geometry = _prescribed_boundary_chunk(
                (
                    replay_reads,
                    model,
                    summation_mode,
                    False,
                    "cpu",
                )
            )
            replay_by_id = {
                value.molecule_id: value for value in replay_contributions
            }
            for replay_index in replay_indices:
                reference = replay_by_id[reads[int(replay_index)].molecule_id]
                family_log_predictive[replay_index] = reference.family_log_predictive
                null_log_predictive[replay_index] = reference.null_log_predictive
                log_bayes_factors[replay_index] = reference.log_bayes_factor
    contributions = tuple(
        _BoundaryContribution(
            molecule_id=read.molecule_id,
            strand=read.strand,
            opportunities=int(
                read.interval_evidence(envelope.start, envelope.end)[1]
            ),
            family_log_predictive=float(family_log_predictive[index]),
            null_log_predictive=float(null_log_predictive[index]),
            log_bayes_factor=float(log_bayes_factors[index]),
            cpu_replayed=bool(cpu_replayed[index]),
        )
        for index, read in enumerate(reads)
    )
    geometry_records = []
    if include_geometry:
        starts = [start for start, _end in candidates]
        ends = [end for _start, end in candidates]
        for read_index, read in enumerate(reads):
            class_probabilities = _softmax(geometry_scores[read_index])
            physical = np.zeros(len(candidates), dtype=np.float64)
            for class_index, members in enumerate(class_members):
                physical[members] += class_probabilities[class_index] * _softmax(
                    candidate_scores[read_index, members]
                )
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
            geometry_records.append(
                {
                    "molecule_id": list(read.molecule_id),
                    "conditional_map_interval": list(candidates[map_index]),
                    "conditional_map_interval_probability": maximum,
                    "conditional_boundary_credible_envelope_95": {
                        "start": [
                            _weighted_integer_quantile(starts, physical, 0.025),
                            _weighted_integer_quantile(starts, physical, 0.975),
                        ],
                        "end": [
                            _weighted_integer_quantile(ends, physical, 0.025),
                            _weighted_integer_quantile(ends, physical, 0.975),
                        ],
                    },
                    "conditional_geometry_entropy_nats": float(
                        -np.sum(positive * np.log(positive))
                    ),
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
    return contributions, tuple(geometry_records)


def _prescribed_boundary_worker(reads):
    return _prescribed_boundary_chunk(
        (
            reads,
            _PRESCRIBED_BOUNDARY_WORKER_MODEL,
            _PRESCRIBED_BOUNDARY_WORKER_MODE,
            False,
        )
    )


def _prescribed_boundary_families_worker(reads):
    """Score one unique-molecule chunk against every frozen family.

    The same ``ReadEvidence`` objects are deliberately reused across families.
    Their immutable opportunity arrays therefore build one prefix cache per
    worker instead of being re-pickled and re-indexed for every family.
    """

    results = []
    for model in _PRESCRIBED_BOUNDARY_WORKER_MODELS:
        retained = _eligible_boundary_reads(reads, model)
        current, _geometry = _prescribed_boundary_chunk(
            (
                retained,
                model,
                _PRESCRIBED_BOUNDARY_WORKER_MODE,
                False,
            )
        )
        results.append(current)
    return tuple(results)


def _prescribed_boundary_families_precomputed_worker(payload):
    """Finish anchored/diffuse likelihoods on CPU after one GPU spatial pass."""

    (
        reads,
        family_scores_by_model,
        spatial_scores_by_model,
        diffuse_scores_by_model,
    ) = payload
    interval_chunk_size, replay_cutoffs, replay_guard = (
        _PRESCRIBED_BOUNDARY_PRECOMPUTED_SETTINGS
    )
    row_by_molecule_id = {
        read.molecule_id: index for index, read in enumerate(reads)
    }
    results = []
    eligibility_seconds = 0.0
    likelihood_seconds = 0.0
    for model_index, model in enumerate(_PRESCRIBED_BOUNDARY_WORKER_MODELS):
        phase_started = time.perf_counter()
        retained = _eligible_boundary_reads(reads, model)
        eligibility_seconds += time.perf_counter() - phase_started
        retained_spatial = np.asarray(
            [
                spatial_scores_by_model[model_index, row_by_molecule_id[read.molecule_id]]
                for read in retained
            ],
            dtype=np.float64,
        )
        retained_family = np.asarray(
            [
                family_scores_by_model[model_index, row_by_molecule_id[read.molecule_id]]
                for read in retained
            ],
            dtype=np.float64,
        )
        retained_diffuse = np.asarray(
            [
                diffuse_scores_by_model[
                    model_index, row_by_molecule_id[read.molecule_id]
                ]
                for read in retained
            ],
            dtype=np.float64,
        )
        phase_started = time.perf_counter()
        current, _geometry = _prescribed_boundary_chunk(
            (
                retained,
                model,
                _PRESCRIBED_BOUNDARY_WORKER_MODE,
                False,
                "cuda",
                None,
                interval_chunk_size,
                replay_cutoffs,
                replay_guard,
                retained_spatial,
                retained_family,
                retained_diffuse,
            )
        )
        likelihood_seconds += time.perf_counter() - phase_started
        results.append(current)
    return tuple(results), eligibility_seconds, likelihood_seconds


def _logistic(values):
    values = np.asarray(values, dtype=np.float64)
    result = np.empty_like(values)
    nonnegative = values >= 0.0
    result[nonnegative] = 1.0 / (1.0 + np.exp(-values[nonnegative]))
    exponential = np.exp(values[~nonnegative])
    result[~nonnegative] = exponential / (1.0 + exponential)
    return result


def _resolve_scoring_likelihood_backend(
    requested,
    *,
    cuda_interval_chunk_size,
    cuda_read_chunk_size,
    cuda_replay_guard_nats,
):
    if isinstance(cuda_interval_chunk_size, bool) or not isinstance(
        cuda_interval_chunk_size, int
    ) or cuda_interval_chunk_size < 1:
        raise ValueError("CUDA interval chunk size must be a positive integer")
    if isinstance(cuda_read_chunk_size, bool) or not isinstance(
        cuda_read_chunk_size, int
    ) or cuda_read_chunk_size < 0:
        raise ValueError("CUDA read chunk size must be a non-negative integer")
    if not math.isfinite(cuda_replay_guard_nats) or cuda_replay_guard_nats < 0.0:
        raise ValueError("CUDA replay guard must be finite and non-negative")
    from fiberhmm.inference.cuda_likelihood import resolve_likelihood_backend

    resolved, runtime = resolve_likelihood_backend(str(requested))
    return resolved, dict(runtime)


def _assignment_log_bayes_factor_cutoffs(
    minimum_assignment_standardized_posterior,
    minimum_assignment_log_bayes_factor,
):
    cutoffs = [float(minimum_assignment_log_bayes_factor)]
    posterior = float(minimum_assignment_standardized_posterior)
    if 0.0 < posterior < 1.0:
        cutoffs.append(math.log(posterior) - math.log1p(-posterior))
    return tuple(sorted(set(cutoffs)))


def _summarize_prescribed_boundary_family(
    contributions,
    retained_by_id,
    model,
    *,
    chunk_size,
    occupancy_pseudocount,
    occupancy_max_iter,
    occupancy_tolerance,
    minimum_assignment_standardized_posterior,
    minimum_assignment_log_bayes_factor,
    evidence_summation_mode,
    include_conditional_geometry,
):
    contributions = sorted(contributions, key=lambda value: value.molecule_id)
    if not contributions:
        return {
            "schema": "fiberhmm.unbiased_targeted_family_score.v1",
            "status": "empty_unbiased_cohort",
            "family_id": str(model["family_id"]),
            "model_structure_id": str(model["model_structure_id"]),
            "eligible_molecules": 0,
            "fitted_family_occupancy": None,
            "molecules": [],
        }
    log_bayes_factors = np.asarray(
        [value.log_bayes_factor for value in contributions], dtype=np.float64
    )
    occupancy = 0.5
    converged = False
    iteration = 0
    for iteration in range(1, occupancy_max_iter + 1):
        log_prior_odds = math.log(max(occupancy, 1e-300)) - math.log(
            max(1.0 - occupancy, 1e-300)
        )
        posteriors = _logistic(log_bayes_factors + log_prior_odds)
        updated = float(
            (np.sum(posteriors) + occupancy_pseudocount)
            / (len(posteriors) + 2.0 * occupancy_pseudocount)
        )
        if abs(updated - occupancy) < occupancy_tolerance:
            occupancy = updated
            converged = True
            break
        occupancy = updated
    log_prior_odds = math.log(max(occupancy, 1e-300)) - math.log(
        max(1.0 - occupancy, 1e-300)
    )
    fitted_posteriors = _logistic(log_bayes_factors + log_prior_odds)
    standardized_posteriors = _logistic(log_bayes_factors)
    # Assignment is a geometry/evidence decision and must not disappear merely
    # because the exploratory one-family prevalence fit shrinks toward zero.
    # The empirical-Bayes posterior remains reported as a cohort summary.
    accepted = np.flatnonzero(
        (standardized_posteriors >= minimum_assignment_standardized_posterior)
        & (log_bayes_factors >= minimum_assignment_log_bayes_factor)
    )
    geometry_by_id = {}
    if include_conditional_geometry:
        accepted_reads = tuple(
            retained_by_id[contributions[int(index)].molecule_id]
            for index in accepted
        )
        for start in range(0, len(accepted_reads), chunk_size):
            _unused, records = _prescribed_boundary_chunk(
                (
                    accepted_reads[start : start + chunk_size],
                    model,
                    evidence_summation_mode,
                    True,
                )
            )
            geometry_by_id.update(
                {tuple(record["molecule_id"]): record for record in records}
            )
    molecule_records = []
    for index in accepted:
        contribution = contributions[int(index)]
        geometry = geometry_by_id.get(contribution.molecule_id, {})
        molecule_records.append(
            {
                **geometry,
                "molecule_id": list(contribution.molecule_id),
                "strand": contribution.strand,
                "opportunities": contribution.opportunities,
                "fitted_family_posterior": float(fitted_posteriors[index]),
                "standardized_family_posterior_equal_prior": float(
                    standardized_posteriors[index]
                ),
                "family_vs_null_log_bayes_factor": float(log_bayes_factors[index]),
            }
        )
    mixture_log_likelihood = np.logaddexp(
        np.log(max(occupancy, 1e-300))
        + np.asarray(
            [value.family_log_predictive for value in contributions], dtype=np.float64
        ),
        np.log(max(1.0 - occupancy, 1e-300))
        + np.asarray(
            [value.null_log_predictive for value in contributions], dtype=np.float64
        ),
    )
    total_opportunities = sum(value.opportunities for value in contributions)
    strands = {
        strand: sum(value.strand == strand for value in contributions)
        for strand in sorted({value.strand for value in contributions})
    }
    return {
        "schema": "fiberhmm.unbiased_targeted_family_score.v1",
        "status": "complete_independent_family_screen",
        "family_id": str(model["family_id"]),
        "model_structure_id": str(model["model_structure_id"]),
        "model_fit_id": str(model.get("model_fit_id", "")),
        "eligible_molecules": len(contributions),
        "eligible_molecules_by_strand": strands,
        "total_opportunities": total_opportunities,
        "fitted_family_occupancy": occupancy,
        "fitted_family_effective_support": float(np.sum(fitted_posteriors)),
        "standardized_family_effective_support_equal_prior": float(
            np.sum(standardized_posteriors)
        ),
        "family_vs_null_log_bayes_factor_nonnegative_molecules": int(
            np.sum(log_bayes_factors >= 0.0)
        ),
        "family_vs_null_log_bayes_factor_at_least_log_10_molecules": int(
            np.sum(log_bayes_factors >= math.log(10.0))
        ),
        "median_family_vs_null_log_bayes_factor": float(
            np.median(log_bayes_factors)
        ),
        "cpu_replayed_near_threshold_molecules": sum(
            int(value.cpu_replayed) for value in contributions
        ),
        "mixture_log_likelihood": float(np.sum(mixture_log_likelihood)),
        "mean_mixture_log_likelihood_per_opportunity": (
            float(np.sum(mixture_log_likelihood)) / total_opportunities
            if total_opportunities
            else None
        ),
        "occupancy_fit": {
            "parameter": "exploratory_scalar_family_prevalence_on_complete_unbiased_cohort",
            "role": "cohort_summary_not_assignment_prior",
            "iterations": iteration,
            "converged": converged,
            "tolerance": occupancy_tolerance,
            "beta_pseudocount_per_state": occupancy_pseudocount,
        },
        "predictive_weight_contract": {
            "family_geometry_classes": "uniform_prescribed",
            "within_projection_class_candidates": "uniform_prescribed",
            "null_accessible_spatial_diffuse": "uniform_prescribed",
            "discovery_mixture_weights_used": False,
        },
        "assignment_selection": {
            "posterior": "standardized_equal_prior_family_vs_null",
            "minimum_standardized_family_posterior_equal_prior": (
                minimum_assignment_standardized_posterior
            ),
            "minimum_family_vs_null_log_bayes_factor": (
                minimum_assignment_log_bayes_factor
            ),
            "conditional_geometry_materialized": bool(
                include_conditional_geometry
            ),
            "emitted_molecules": len(molecule_records),
        },
        "molecules": molecule_records,
    }


def score_boundary_family_on_unbiased_cohort(
    reads,
    model: Mapping[str, object],
    *,
    chunk_size: int = 512,
    workers: int = 1,
    occupancy_pseudocount: float = 0.5,
    occupancy_max_iter: int = 500,
    occupancy_tolerance: float = 1e-10,
    minimum_assignment_standardized_posterior: float = 0.5,
    minimum_assignment_log_bayes_factor: float = 0.0,
    evidence_summation_mode: str = "prefix",
    include_conditional_geometry: bool = True,
    likelihood_backend: str = "cpu",
    cuda_interval_chunk_size: int = 512,
    cuda_read_chunk_size: int = 0,
    cuda_replay_guard_nats: float = 1e-8,
    progress: Optional[Callable[[int, int], None]] = None,
    minimum_assignment_posterior: Optional[float] = None,
) -> dict:
    """Score frozen geometry without leaking MSP-enriched mixture weights.

    Geometry-class weights are prescribed uniformly, as are the accessible,
    displaced-interval, and diffuse null components.  Only the scalar family
    prevalence is estimated on the complete unbiased cohort.  The returned
    equal-prior posterior therefore has the same meaning across discovery caps,
    while ``fitted_family_posterior`` supports cohort-specific quantification.
    """

    if model.get("schema") != "fiberhmm.boundary_marginalized_tf_family_model.v1":
        raise ValueError("unbiased scorer requires a v1 boundary-family model")
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer")
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be a positive integer")
    if not math.isfinite(occupancy_pseudocount) or occupancy_pseudocount < 0.0:
        raise ValueError("occupancy_pseudocount must be finite and non-negative")
    if (
        isinstance(occupancy_max_iter, bool)
        or not isinstance(occupancy_max_iter, int)
        or occupancy_max_iter < 1
    ):
        raise ValueError("occupancy_max_iter must be a positive integer")
    if not math.isfinite(occupancy_tolerance) or occupancy_tolerance <= 0.0:
        raise ValueError("occupancy_tolerance must be finite and positive")
    if not math.isfinite(minimum_assignment_log_bayes_factor):
        raise ValueError("minimum_assignment_log_bayes_factor must be finite")
    if evidence_summation_mode not in {"prefix", "slice_sum"}:
        raise ValueError("invalid interval evidence summation mode")
    if minimum_assignment_posterior is not None:
        if (
            minimum_assignment_standardized_posterior != 0.5
            and minimum_assignment_standardized_posterior
            != minimum_assignment_posterior
        ):
            raise ValueError(
                "conflicting standardized and legacy assignment posterior thresholds"
            )
        minimum_assignment_standardized_posterior = minimum_assignment_posterior
    if not 0.0 <= minimum_assignment_standardized_posterior <= 1.0:
        raise ValueError(
            "minimum_assignment_standardized_posterior must be in [0,1]"
        )
    resolved_backend, backend_runtime = _resolve_scoring_likelihood_backend(
        likelihood_backend,
        cuda_interval_chunk_size=cuda_interval_chunk_size,
        cuda_read_chunk_size=cuda_read_chunk_size,
        cuda_replay_guard_nats=cuda_replay_guard_nats,
    )
    replay_cutoffs = _assignment_log_bayes_factor_cutoffs(
        minimum_assignment_standardized_posterior,
        minimum_assignment_log_bayes_factor,
    )
    model = _prepare_prescribed_boundary_model(model)
    retained = _eligible_boundary_reads(reads, model)
    cuda_batch_plan = {"mode": "not_applicable_cpu"}
    if resolved_backend == "cuda" and cuda_read_chunk_size == 0:
        from fiberhmm.inference.cuda_likelihood import (
            recommend_cuda_read_chunk_size,
        )

        effective_chunk_size, cuda_batch_plan = recommend_cuda_read_chunk_size(
            retained,
            interval_chunk_size=cuda_interval_chunk_size,
            maximum_envelope_width=(
                int(model["envelope"][1]) - int(model["envelope"][0])
            ),
            minimum=chunk_size,
        )
    elif resolved_backend == "cuda":
        effective_chunk_size = max(chunk_size, cuda_read_chunk_size)
        cuda_batch_plan = {
            "mode": "explicit",
            "requested_rows": int(cuda_read_chunk_size),
            "effective_rows": int(effective_chunk_size),
        }
    else:
        effective_chunk_size = chunk_size
    chunks = tuple(
        retained[start : start + effective_chunk_size]
        for start in range(0, len(retained), effective_chunk_size)
    )
    contributions = []
    execution_workers = 1 if resolved_backend == "cuda" else workers
    if execution_workers == 1 or len(chunks) <= 1:
        for completed, chunk in enumerate(chunks, start=1):
            cuda_batch = None
            if resolved_backend == "cuda" and chunk:
                from fiberhmm.inference.cuda_likelihood import (
                    prepare_torch_spatial_null_batch,
                )

                cuda_batch = prepare_torch_spatial_null_batch(chunk)
            current, _geometry = _prescribed_boundary_chunk(
                (
                    chunk,
                    model,
                    evidence_summation_mode,
                    False,
                    resolved_backend,
                    cuda_batch,
                    cuda_interval_chunk_size,
                    replay_cutoffs,
                    cuda_replay_guard_nats,
                )
            )
            contributions.extend(current)
            if progress is not None:
                progress(completed, len(chunks))
    else:
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=_MP_CONTEXT,
            initializer=_initialize_prescribed_boundary_worker,
            initargs=(model, evidence_summation_mode),
        ) as executor:
            for completed, (current, _geometry) in enumerate(
                executor.map(_prescribed_boundary_worker, chunks), start=1
            ):
                contributions.extend(current)
                if progress is not None:
                    progress(completed, len(chunks))
    retained_by_id = {read.molecule_id: read for read in retained}
    summary = _summarize_prescribed_boundary_family(
        contributions,
        retained_by_id,
        model,
        chunk_size=chunk_size,
        occupancy_pseudocount=occupancy_pseudocount,
        occupancy_max_iter=occupancy_max_iter,
        occupancy_tolerance=occupancy_tolerance,
        minimum_assignment_standardized_posterior=(
            minimum_assignment_standardized_posterior
        ),
        minimum_assignment_log_bayes_factor=(
            minimum_assignment_log_bayes_factor
        ),
        evidence_summation_mode=evidence_summation_mode,
        include_conditional_geometry=include_conditional_geometry,
    )
    summary["likelihood_backend"] = {
        "requested": str(likelihood_backend),
        "resolved": resolved_backend,
        "runtime": backend_runtime,
        "requested_workers": workers,
        "scoring_workers": execution_workers,
        "cuda_interval_chunk_size": int(cuda_interval_chunk_size),
        "cuda_read_chunk_size": int(cuda_read_chunk_size),
        "effective_read_chunk_size": int(effective_chunk_size),
        "cuda_batch_plan": cuda_batch_plan,
        "cpu_replay_guard_nats": float(cuda_replay_guard_nats),
        "cpu_replay_cutoffs_log_bayes_factor": list(replay_cutoffs),
    }
    return summary


def score_boundary_families_on_unbiased_cohort(
    reads,
    models: Sequence[Mapping[str, object]],
    *,
    chunk_size: int = 512,
    workers: int = 1,
    occupancy_pseudocount: float = 0.5,
    occupancy_max_iter: int = 500,
    occupancy_tolerance: float = 1e-10,
    minimum_assignment_standardized_posterior: float = 0.5,
    minimum_assignment_log_bayes_factor: float = 0.0,
    evidence_summation_mode: str = "prefix",
    include_conditional_geometry: bool = False,
    likelihood_backend: str = "cpu",
    cuda_interval_chunk_size: int = 1024,
    cuda_read_chunk_size: int = 0,
    cuda_family_batch_span_bp: int = 25000,
    cuda_replay_guard_nats: float = 1e-8,
    progress: Optional[Callable[[int, int], None]] = None,
) -> Tuple[dict, ...]:
    """Score all families in one bounded, deterministic molecule stream.

    This is the production high-throughput backend.  The targeted-family
    allowlist normally supplies unique independent molecule IDs. If defensive
    input still contains repeated representations, the function falls back to
    exact per-family envelope deduplication so it cannot disagree with the
    single-family API. Each joint worker otherwise receives a read chunk once
    and scores every frozen family, retaining per-read prefix caches.
    """

    frozen_models = tuple(models)
    if not frozen_models:
        return ()
    for model in frozen_models:
        if model.get("schema") != "fiberhmm.boundary_marginalized_tf_family_model.v1":
            raise ValueError("unbiased scorer requires v1 boundary-family models")
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer")
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be a positive integer")
    if not math.isfinite(occupancy_pseudocount) or occupancy_pseudocount < 0.0:
        raise ValueError("occupancy_pseudocount must be finite and non-negative")
    if (
        isinstance(occupancy_max_iter, bool)
        or not isinstance(occupancy_max_iter, int)
        or occupancy_max_iter < 1
    ):
        raise ValueError("occupancy_max_iter must be a positive integer")
    if not math.isfinite(occupancy_tolerance) or occupancy_tolerance <= 0.0:
        raise ValueError("occupancy_tolerance must be finite and positive")
    if not math.isfinite(minimum_assignment_log_bayes_factor):
        raise ValueError("minimum_assignment_log_bayes_factor must be finite")
    if evidence_summation_mode not in {"prefix", "slice_sum"}:
        raise ValueError("invalid interval evidence summation mode")
    if not 0.0 <= minimum_assignment_standardized_posterior <= 1.0:
        raise ValueError(
            "minimum_assignment_standardized_posterior must be in [0,1]"
        )
    resolved_backend, backend_runtime = _resolve_scoring_likelihood_backend(
        likelihood_backend,
        cuda_interval_chunk_size=cuda_interval_chunk_size,
        cuda_read_chunk_size=cuda_read_chunk_size,
        cuda_replay_guard_nats=cuda_replay_guard_nats,
    )
    if resolved_backend == "cuda" and evidence_summation_mode != "prefix":
        raise ValueError("CUDA likelihood evaluation supports prefix summation only")
    if (
        isinstance(cuda_family_batch_span_bp, bool)
        or not isinstance(cuda_family_batch_span_bp, int)
        or cuda_family_batch_span_bp < 1
    ):
        raise ValueError("CUDA family batch span must be a positive integer")
    replay_cutoffs = _assignment_log_bayes_factor_cutoffs(
        minimum_assignment_standardized_posterior,
        minimum_assignment_log_bayes_factor,
    )
    input_reads = tuple(reads)
    input_record_count = len(input_reads)
    unique_molecule_ids = {read.molecule_id for read in input_reads}
    if len(unique_molecule_ids) != input_record_count:
        # A locus-wide representative can differ from the exact
        # family-envelope choice. Production allowlists are unique; preserve
        # exact single-family semantics for defensive duplicate input.
        scores = []
        for completed, model in enumerate(frozen_models, start=1):
            score = score_boundary_family_on_unbiased_cohort(
                input_reads,
                model,
                chunk_size=chunk_size,
                workers=workers,
                occupancy_pseudocount=occupancy_pseudocount,
                occupancy_max_iter=occupancy_max_iter,
                occupancy_tolerance=occupancy_tolerance,
                minimum_assignment_standardized_posterior=(
                    minimum_assignment_standardized_posterior
                ),
                minimum_assignment_log_bayes_factor=(
                    minimum_assignment_log_bayes_factor
                ),
                evidence_summation_mode=evidence_summation_mode,
                include_conditional_geometry=include_conditional_geometry,
                likelihood_backend=likelihood_backend,
                cuda_interval_chunk_size=cuda_interval_chunk_size,
                cuda_read_chunk_size=cuda_read_chunk_size,
                cuda_replay_guard_nats=cuda_replay_guard_nats,
            )
            score["joint_cohort_deduplication"] = {
                "policy": "per_family_envelope_opportunities_then_read_evidence_sha256",
                "execution": "exact_single_family_fallback_for_duplicate_representations",
                "input_records": input_record_count,
                "unique_independent_molecules": len(unique_molecule_ids),
                "discarded_duplicate_representations": (
                    input_record_count - len(unique_molecule_ids)
                ),
            }
            scores.append(score)
            if progress is not None:
                progress(completed, len(frozen_models))
        return tuple(scores)

    frozen_models = tuple(
        _prepare_prescribed_boundary_model(model) for model in frozen_models
    )
    retained_by_id = {read.molecule_id: read for read in input_reads}
    ordered_reads = tuple(retained_by_id[key] for key in sorted(retained_by_id))
    cuda_batch_plan = {"mode": "not_applicable_cpu"}
    if resolved_backend == "cuda" and cuda_read_chunk_size == 0:
        from fiberhmm.inference.cuda_likelihood import (
            recommend_cuda_read_chunk_size,
        )

        effective_chunk_size, cuda_batch_plan = recommend_cuda_read_chunk_size(
            ordered_reads,
            interval_chunk_size=cuda_interval_chunk_size,
            maximum_envelope_width=max(
                int(model["envelope"][1]) - int(model["envelope"][0])
                for model in frozen_models
            ),
            minimum=chunk_size,
        )
    elif resolved_backend == "cuda":
        effective_chunk_size = max(chunk_size, cuda_read_chunk_size)
        cuda_batch_plan = {
            "mode": "explicit",
            "requested_rows": int(cuda_read_chunk_size),
            "effective_rows": int(effective_chunk_size),
        }
    else:
        effective_chunk_size = chunk_size
    chunks = tuple(
        ordered_reads[start : start + effective_chunk_size]
        for start in range(0, len(ordered_reads), effective_chunk_size)
    )
    cuda_locality_batches = ()
    if resolved_backend == "cuda":
        cuda_locality_batches = _cuda_locality_family_batches(
            ordered_reads,
            frozen_models,
            cuda_family_batch_span_bp,
        )
        cuda_batch_plan = {
            **dict(cuda_batch_plan),
            "family_locality_span_bp": int(cuda_family_batch_span_bp),
            "family_locality_batches": len(cuda_locality_batches),
            "maximum_families_per_locality_batch": max(
                (len(value[0]) for value in cuda_locality_batches), default=0
            ),
            "maximum_candidate_reads_per_locality_batch": max(
                (len(value[1]) for value in cuda_locality_batches), default=0
            ),
        }
        cpu_work_chunk_count = max(
            1,
            sum(
                math.ceil(len(local_reads) / chunk_size)
                for _indices, local_reads, _span in cuda_locality_batches
            ),
        )
    else:
        cpu_work_chunk_count = max(1, math.ceil(len(ordered_reads) / chunk_size))
    contributions_by_model = [[] for _model in frozen_models]
    # CUDA is owned by the parent process. Once every likelihood component is
    # resident on the device, spawning Python workers only adds serialization
    # and interpreter-startup cost to the small deterministic assembly step.
    execution_workers = (
        1
        if resolved_backend == "cuda"
        else min(workers, cpu_work_chunk_count)
    )
    cuda_phase_seconds = {
        "resident_grid_preparation": 0.0,
        "resident_read_batch_preparation": 0.0,
        "family_spatial_diffuse_device_evaluation_and_transfer": 0.0,
        "family_eligibility": 0.0,
        "remaining_cpu_family_likelihood": 0.0,
    }
    if resolved_backend == "cuda":
        from fiberhmm.inference.cuda_likelihood import (
            prepare_torch_family_likelihood_grids,
            reset_cuda_peak_memory_stats,
        )

        reset_cuda_peak_memory_stats()

    def prepare_cuda_chunk(chunk, cuda_grids):
        from fiberhmm.inference.cuda_likelihood import (
            prepare_torch_spatial_null_batch,
        )

        phase_started = time.perf_counter()
        cuda_batch = prepare_torch_spatial_null_batch(chunk)
        cuda_phase_seconds["resident_read_batch_preparation"] += (
            time.perf_counter() - phase_started
        )
        phase_started = time.perf_counter()
        (
            cuda_family_scores,
            cuda_spatial_scores,
            cuda_diffuse_scores,
            cuda_opportunity_counts,
        ) = (
            cuda_batch.family_spatial_diffuse_log_likelihoods_many(
                cuda_grids,
                interval_chunk_size=cuda_interval_chunk_size,
            )
        )
        cuda_phase_seconds[
            "family_spatial_diffuse_device_evaluation_and_transfer"
        ] += (
            time.perf_counter() - phase_started
        )
        return (
            cuda_batch,
            cuda_family_scores,
            cuda_spatial_scores,
            cuda_diffuse_scores,
            cuda_opportunity_counts,
        )

    if resolved_backend == "cuda":
        # Evaluate all likelihoods in the largest safe device batch, then use
        # exact vectorized eligibility and bounded host assembly. This keeps
        # one CUDA owner and avoids paying spawn/pickle overhead for work that
        # is now much smaller than the likelihood kernel itself.
        completed = 0
        for model_indices, local_reads, _locality_span in cuda_locality_batches:
            local_models = tuple(frozen_models[index] for index in model_indices)
            phase_started = time.perf_counter()
            cuda_grids = prepare_torch_family_likelihood_grids(local_models)
            cuda_phase_seconds["resident_grid_preparation"] += (
                time.perf_counter() - phase_started
            )
            for cuda_offset in range(0, len(local_reads), effective_chunk_size):
                # Drop references before allocating the next resident batch so
                # the allocator never needs to accommodate two full batches.
                if "_cuda_batch" in locals():
                    del _cuda_batch
                    del cuda_family_scores
                    del cuda_spatial_scores
                    del cuda_diffuse_scores
                    del cuda_opportunity_counts
                chunk = local_reads[cuda_offset : cuda_offset + effective_chunk_size]
                (
                    _cuda_batch,
                    cuda_family_scores,
                    cuda_spatial_scores,
                    cuda_diffuse_scores,
                    cuda_opportunity_counts,
                ) = prepare_cuda_chunk(chunk, cuda_grids)
                phase_started = time.perf_counter()
                eligible, opportunity_counts = _precompute_unique_boundary_eligibility(
                    chunk,
                    local_models,
                    precomputed_opportunities=cuda_opportunity_counts,
                )
                cuda_phase_seconds["family_eligibility"] += (
                    time.perf_counter() - phase_started
                )
                for offset in range(0, len(chunk), chunk_size):
                    stop = min(offset + chunk_size, len(chunk))
                    phase_started = time.perf_counter()
                    for local_model_index, model_index in enumerate(model_indices):
                        model = frozen_models[model_index]
                        retained_rows = np.flatnonzero(
                            eligible[local_model_index, offset:stop]
                        ) + offset
                        retained = tuple(
                            chunk[int(index)] for index in retained_rows
                        )
                        current = _prescribed_boundary_precomputed_contributions(
                            retained,
                            model,
                            cuda_family_scores[local_model_index, retained_rows],
                            cuda_spatial_scores[local_model_index, retained_rows],
                            cuda_diffuse_scores[local_model_index, retained_rows],
                            opportunity_counts[local_model_index, retained_rows],
                            summation_mode=evidence_summation_mode,
                            cpu_replay_cutoffs=replay_cutoffs,
                            cpu_replay_guard_nats=cuda_replay_guard_nats,
                        )
                        contributions_by_model[model_index].extend(current)
                    cuda_phase_seconds["remaining_cpu_family_likelihood"] += (
                        time.perf_counter() - phase_started
                    )
                    completed += 1
                    if progress is not None:
                        progress(completed, cpu_work_chunk_count)
    elif execution_workers == 1 or len(chunks) <= 1:
        for completed, chunk in enumerate(chunks, start=1):
            current_by_model = []
            for model in frozen_models:
                phase_started = time.perf_counter()
                retained = _eligible_boundary_reads(chunk, model)
                cuda_phase_seconds["family_eligibility"] += (
                    time.perf_counter() - phase_started
                )
                phase_started = time.perf_counter()
                current, _geometry = _prescribed_boundary_chunk(
                    (
                        retained,
                        model,
                        evidence_summation_mode,
                        False,
                        "cpu",
                        None,
                        cuda_interval_chunk_size,
                        replay_cutoffs,
                        cuda_replay_guard_nats,
                        None,
                        None,
                        None,
                    )
                )
                cuda_phase_seconds["remaining_cpu_family_likelihood"] += (
                    time.perf_counter() - phase_started
                )
                current_by_model.append(current)
            for index, current in enumerate(current_by_model):
                contributions_by_model[index].extend(current)
            if progress is not None:
                progress(completed, len(chunks))
    else:
        with ProcessPoolExecutor(
            max_workers=min(execution_workers, len(chunks)),
            mp_context=_MP_CONTEXT,
            initializer=_initialize_prescribed_boundary_families_worker,
            initargs=(frozen_models, evidence_summation_mode),
        ) as executor:
            for completed, current_by_model in enumerate(
                executor.map(_prescribed_boundary_families_worker, chunks), start=1
            ):
                for index, current in enumerate(current_by_model):
                    contributions_by_model[index].extend(current)
                if progress is not None:
                    progress(completed, len(chunks))
    scores = tuple(
        _summarize_prescribed_boundary_family(
            contributions,
            retained_by_id,
            model,
            chunk_size=chunk_size,
            occupancy_pseudocount=occupancy_pseudocount,
            occupancy_max_iter=occupancy_max_iter,
            occupancy_tolerance=occupancy_tolerance,
            minimum_assignment_standardized_posterior=(
                minimum_assignment_standardized_posterior
            ),
            minimum_assignment_log_bayes_factor=(
                minimum_assignment_log_bayes_factor
            ),
            evidence_summation_mode=evidence_summation_mode,
            include_conditional_geometry=include_conditional_geometry,
        )
        for model, contributions in zip(frozen_models, contributions_by_model)
    )
    deduplication = {
        "policy": "unique_independent_molecule_allowlist_no_deduplication_required",
        "input_records": input_record_count,
        "unique_independent_molecules": len(retained_by_id),
        "discarded_duplicate_representations": (
            input_record_count - len(retained_by_id)
        ),
    }
    cuda_memory = {}
    if resolved_backend == "cuda":
        from fiberhmm.inference.cuda_likelihood import cuda_memory_snapshot

        cuda_memory = dict(cuda_memory_snapshot())
    for score in scores:
        score["joint_cohort_deduplication"] = deduplication
        score["likelihood_backend"] = {
            "requested": str(likelihood_backend),
            "resolved": resolved_backend,
            "runtime": backend_runtime,
            "requested_workers": workers,
            "scoring_workers": execution_workers,
            "cuda_interval_chunk_size": int(cuda_interval_chunk_size),
            "cuda_read_chunk_size": int(cuda_read_chunk_size),
            "cuda_family_batch_span_bp": int(cuda_family_batch_span_bp),
            "effective_read_chunk_size": int(effective_chunk_size),
            "cuda_batch_plan": cuda_batch_plan,
            "cpu_replay_guard_nats": float(cuda_replay_guard_nats),
            "cpu_replay_cutoffs_log_bayes_factor": list(replay_cutoffs),
            "cuda_phase_seconds": {
                key: float(value) for key, value in cuda_phase_seconds.items()
            },
            "cuda_memory": cuda_memory,
        }
    return scores


def _fully_maps(molecule: BaselineMolecule, start: int, end: int) -> bool:
    if end <= start:
        return False
    mapped = 0
    left_mapped = False
    right_mapped = False
    for left, right in molecule.mapped_blocks:
        left_mapped |= left <= start < right
        right_mapped |= left <= end - 1 < right
        mapped += max(0, min(end, right) - max(start, left))
    return bool(left_mapped and right_mapped and mapped / (end - start) >= 0.95)


def collapse_daf_amplification_families(
    input_bam: str | Path,
    molecules: Sequence[BaselineMolecule],
    *,
    contig: str,
    start: int,
    end: int,
    min_mapq: int = 20,
    minimum_jaccard: float = 0.95,
    minimum_deaminations: int = 10,
) -> Tuple[Tuple[BaselineMolecule, ...], Mapping[str, object]]:
    """Collapse DAF PCR families before population geometry discovery.

    Fingerprints are calculated separately for each input library.  Reads with
    too few deaminations cannot establish molecule independence and are
    excluded rather than counted as independent molecules.  This conservative
    exclusion can enrich the retained geometry cohort for accessible fibers;
    the manifest therefore records it and the cohort remains geometry-only.
    """

    if not 0.0 < float(minimum_jaccard) <= 1.0:
        raise ValueError("minimum_jaccard must be in (0,1]")
    if minimum_deaminations < 1:
        raise ValueError("minimum_deaminations must be positive")
    by_name = {molecule.molecule_id: molecule for molecule in molecules}
    if len(by_name) != len(molecules):
        raise ValueError("DAF amplification collapse requires unique read names")
    fingerprints: Dict[str, frozenset[int]] = {}
    strata: Dict[str, str] = {}
    endpoints: Dict[str, Tuple[int, int]] = {}
    with pysam.AlignmentFile(str(Path(input_bam).expanduser().resolve()), "rb") as bam:
        for read in bam.fetch(contig, start, end):
            name = str(read.query_name or "")
            if name not in by_name or (
                read.is_unmapped
                or read.is_secondary
                or read.is_supplementary
                or read.is_qcfail
                or read.is_duplicate
                or int(read.mapping_quality) < min_mapq
            ):
                continue
            observed = _deam_positions_list(read, cigar_to_query_ref(read), 0)
            fingerprints[name] = frozenset(
                int(position) for position, _flavor in observed
            )
            ct_hits = sum(int(flavor) == 1 for _position, flavor in observed)
            ga_hits = sum(int(flavor) == 0 for _position, flavor in observed)
            strata[name] = "CT" if ct_hits > ga_hits else "GA" if ga_hits > ct_hits else "."
            endpoints[name] = (
                int(read.reference_start),
                int(read.reference_end or read.reference_start),
            )

    ordered = sorted(
        (
            replace(molecule, stratum=strata.get(molecule.molecule_id, "."))
            for molecule in molecules
        ),
        key=lambda molecule: molecule.molecule_id,
    )
    position_sets = []
    group_keys = []
    alignment_endpoints = []
    for molecule in ordered:
        fingerprint = fingerprints.get(molecule.molecule_id, frozenset())
        if len(fingerprint) < minimum_deaminations:
            position_sets.append(None)
            group_keys.append(None)
            alignment_endpoints.append(None)
        else:
            position_sets.append(fingerprint)
            group_keys.append((molecule.contig, molecule.stratum))
            alignment_endpoints.append(endpoints.get(molecule.molecule_id))
    labels = cluster_reads(
        position_sets,
        group_keys,
        minimum_jaccard,
        32,
        8,
        7,
        endpoints=alignment_endpoints,
        max_end_diff=50,
    )
    best_by_cluster: Dict[int, int] = {}
    cluster_sizes: Dict[int, int] = {}
    for index, raw_label in enumerate(labels):
        label = int(raw_label)
        if label < 0:
            continue
        cluster_sizes[label] = cluster_sizes.get(label, 0) + 1
        incumbent = best_by_cluster.get(label)
        quality = (
            sum(
                max(0, min(end, right) - max(start, left))
                for left, right in ordered[index].mapped_blocks
            ),
            sum(right - left for left, right in ordered[index].mapped_blocks),
            ordered[index].molecule_id,
        )
        if incumbent is None:
            best_by_cluster[label] = index
            continue
        incumbent_quality = (
            sum(
                max(0, min(end, right) - max(start, left))
                for left, right in ordered[incumbent].mapped_blocks
            ),
            sum(right - left for left, right in ordered[incumbent].mapped_blocks),
            ordered[incumbent].molecule_id,
        )
        if quality > incumbent_quality:
            best_by_cluster[label] = index
    retained = tuple(ordered[index] for index in sorted(best_by_cluster.values()))
    fingerprintable = sum(int(value) >= 0 for value in labels)
    return retained, {
        "mode": "deamination_fingerprint_per_input_bam",
        "raw_ordinary_annotation_molecules": len(ordered),
        "fingerprintable_reads": fingerprintable,
        "unfingerprintable_reads": len(ordered) - fingerprintable,
        "retained_amplification_family_representatives": len(retained),
        "duplicate_reads_collapsed": fingerprintable - len(retained),
        "largest_family": max(cluster_sizes.values(), default=0),
        "minimum_jaccard": float(minimum_jaccard),
        "minimum_deaminations": int(minimum_deaminations),
        "unfingerprintable_policy": "excluded_independence_not_established",
    }


def _has_long_msp_overlap(
    molecule: BaselineMolecule,
    start: int,
    end: int,
    minimum_length: int,
) -> bool:
    return any(
        right - left >= minimum_length and left < end and start < right
        for left, right in molecule.msps
    )


def index_informative_windows(
    molecules: Sequence[BaselineMolecule],
    *,
    contig: str,
    locus_start: int,
    locus_end: int,
    config: TargetedFamilyDiscoveryConfig = TargetedFamilyDiscoveryConfig(),
) -> Tuple[TargetedFamilyWindow, ...]:
    """Build the recurrent-long-MSP window index in one molecule traversal."""

    if not contig or locus_start < 0 or locus_end <= locus_start:
        raise ValueError("invalid targeted locus")
    raw_windows = []
    for ordinal, core_start in enumerate(
        range(locus_start, locus_end, config.core_size)
    ):
        core_end = min(locus_end, core_start + config.core_size)
        raw_windows.append(
            (
                ordinal,
                core_start,
                core_end,
                max(0, core_start - config.halo_size),
                core_end + config.halo_size,
            )
        )
    denominators = [0] * len(raw_windows)
    long_msp_support = [0] * len(raw_windows)
    for molecule in molecules:
        if molecule.contig != contig or not molecule.mapped_blocks:
            continue
        # A fully mapped core must lie inside the molecule's outer alignment
        # bounds. Restrict the exact block-aware test to that local ordinal
        # range instead of scanning every locus window for every molecule.
        outer_start = min(left for left, _right in molecule.mapped_blocks)
        outer_end = max(right for _left, right in molecule.mapped_blocks)
        first = max(
            0,
            int(math.floor((outer_start - locus_start) / config.core_size)) - 1,
        )
        stop = min(
            len(raw_windows),
            int(math.floor((outer_end - locus_start) / config.core_size)) + 2,
        )
        for ordinal in range(first, stop):
            _ordinal, core_start, core_end, _halo_start, _halo_end = raw_windows[
                ordinal
            ]
            if not _fully_maps(molecule, core_start, core_end):
                continue
            denominators[ordinal] += 1
            if _has_long_msp_overlap(
                molecule,
                core_start,
                core_end,
                config.minimum_nfr_length,
            ):
                long_msp_support[ordinal] += 1

    windows = []
    for ordinal, core_start, core_end, halo_start, halo_end in raw_windows:
        required = max(
            config.minimum_informative_molecules,
            int(
                math.ceil(
                    config.minimum_informative_fraction * denominators[ordinal]
                )
            ),
        )
        windows.append(
            TargetedFamilyWindow(
                ordinal=ordinal,
                contig=contig,
                core_start=core_start,
                core_end=core_end,
                halo_start=halo_start,
                halo_end=halo_end,
                fully_mapped_molecules=denominators[ordinal],
                long_msp_molecules=long_msp_support[ordinal],
                required_long_msp_molecules=required,
                informative=bool(long_msp_support[ordinal] >= required),
            )
        )
    return tuple(windows)


def _molecules_by_window_halo(
    molecules: Sequence[BaselineMolecule],
    windows: Sequence[TargetedFamilyWindow],
) -> Tuple[Tuple[BaselineMolecule, ...], ...]:
    """Index molecules into only the core+halo tasks their blocks overlap."""

    if not windows:
        return ()
    halo_starts = [window.halo_start for window in windows]
    halo_ends = [window.halo_end for window in windows]
    indexed: List[List[BaselineMolecule]] = [[] for _window in windows]
    contig = windows[0].contig
    for molecule in molecules:
        if molecule.contig != contig:
            continue
        ordinals = set()
        for left, right in molecule.mapped_blocks:
            first = bisect_right(halo_ends, left)
            stop = bisect_left(halo_starts, right)
            ordinals.update(range(first, stop))
        for ordinal in sorted(ordinals):
            indexed[ordinal].append(molecule)
    return tuple(tuple(values) for values in indexed)


def _discovery_stratum(molecule: BaselineMolecule) -> str:
    source, separator, _name = molecule.molecule_id.partition("\x1f")
    source = source if separator else "."
    return f"{source}|{molecule.stratum}"


def _stable_molecule_key(seed: str, window: TargetedFamilyWindow, molecule_id: str):
    digest = hashlib.sha256(
        "\x1f".join((seed, window.label, molecule_id)).encode("utf-8")
    ).digest()
    return digest, molecule_id


def _proportional_quotas(group_sizes: Mapping[str, int], capacity: int) -> Dict[str, int]:
    total = sum(group_sizes.values())
    if capacity >= total:
        return dict(group_sizes)
    exact = {
        name: capacity * size / total for name, size in group_sizes.items()
    }
    quotas = {name: min(group_sizes[name], int(math.floor(value))) for name, value in exact.items()}
    remaining = capacity - sum(quotas.values())
    order = sorted(
        group_sizes,
        key=lambda name: (-(exact[name] - math.floor(exact[name])), name),
    )
    for name in order:
        if not remaining:
            break
        if quotas[name] < group_sizes[name]:
            quotas[name] += 1
            remaining -= 1
    return quotas


def select_discovery_molecules(
    molecules: Sequence[BaselineMolecule],
    window: TargetedFamilyWindow,
    *,
    config: TargetedFamilyDiscoveryConfig = TargetedFamilyDiscoveryConfig(),
) -> Tuple[BaselineMolecule, ...]:
    """Select a reproducible, source/strand-stratified MSP-enriched cohort."""

    if not window.informative:
        return ()
    groups: Dict[str, List[BaselineMolecule]] = {}
    for molecule in _eligible_discovery_molecules(
        molecules, window, config=config
    ):
        groups.setdefault(_discovery_stratum(molecule), []).append(molecule)
    quotas = _proportional_quotas(
        {name: len(values) for name, values in groups.items()},
        config.maximum_discovery_molecules,
    )
    selected = []
    for name in sorted(groups):
        ordered = sorted(
            groups[name],
            key=lambda molecule: _stable_molecule_key(
                config.seed, window, molecule.molecule_id
            ),
        )
        selected.extend(ordered[: quotas[name]])
    return tuple(sorted(selected, key=lambda molecule: molecule.molecule_id))


def _eligible_discovery_molecules(
    molecules: Sequence[BaselineMolecule],
    window: TargetedFamilyWindow,
    *,
    config: TargetedFamilyDiscoveryConfig,
) -> Tuple[BaselineMolecule, ...]:
    return tuple(
        molecule
        for molecule in molecules
        if molecule.contig == window.contig
        and _has_long_msp_overlap(
            molecule,
            window.core_start,
            window.core_end,
            config.minimum_nfr_length,
        )
        and any(
            left < window.halo_end and window.halo_start < right
            for left, right in molecule.mapped_blocks
        )
    )


def _weighted_integer_median(values: Sequence[int], weights: Sequence[int]) -> int:
    ordered = sorted(zip(values, weights))
    threshold = sum(weights) / 2.0
    cumulative = 0
    for value, weight in ordered:
        cumulative += weight
        if cumulative >= threshold:
            return int(value)
    return int(ordered[-1][0])


def _overlap_fraction(left: Interval, right: Interval) -> float:
    overlap = max(0, min(left[1], right[1]) - max(left[0], right[0]))
    shorter = min(left[1] - left[0], right[1] - right[0])
    return overlap / shorter if shorter > 0 else 0.0


def geometry_compatible(
    left: TargetedFootprintFamily,
    right: TargetedFootprintFamily,
    chemistry: TargetedFamilyChemistry,
) -> bool:
    """Return chemistry-aware complete-link compatibility for two geometries."""

    return bool(
        left.contig == right.contig
        and abs(left.start - right.start) <= chemistry.maximum_boundary_delta
        and abs(left.end - right.end) <= chemistry.maximum_boundary_delta
        and abs(left.width - right.width) <= chemistry.maximum_width_delta
        and abs(left.center - right.center) <= chemistry.maximum_center_delta
        and _overlap_fraction((left.start, left.end), (right.start, right.end))
        >= chemistry.minimum_shorter_overlap_fraction
    )


def _slice_molecules(
    molecules: Sequence[BaselineMolecule], window: TargetedFamilyWindow
) -> Tuple[BaselineMolecule, ...]:
    sliced = []
    for molecule in molecules:
        blocks = tuple(
            (max(window.halo_start, left), min(window.halo_end, right))
            for left, right in molecule.mapped_blocks
            if left < window.halo_end and window.halo_start < right
        )
        if not blocks:
            continue
        sliced.append(
            BaselineMolecule(
                molecule_id=molecule.molecule_id,
                contig=molecule.contig,
                mapped_blocks=blocks,
                tfs=tuple(
                    call
                    for call in molecule.tfs
                    if window.halo_start <= call.center < window.halo_end
                ),
                msps=tuple(
                    (left, right)
                    for left, right in molecule.msps
                    if left < window.halo_end and window.halo_start < right
                ),
                stratum=molecule.stratum,
            )
        )
    return tuple(sliced)


def _families_from_catalog(
    catalog,
    molecules: Sequence[BaselineMolecule],
    window: TargetedFamilyWindow,
    chemistry: TargetedFamilyChemistry,
    minimum_support: int,
    minimum_fraction: float,
) -> Tuple[TargetedFootprintFamily, ...]:
    sites = [
        site
        for site in catalog.sites
        if site.analysis_ready
        and site.population_support
        >= max(minimum_support, int(math.ceil(minimum_fraction * site.n_fully_mapped)))
    ]
    sites.sort(key=lambda site: (site.contig, site.start, site.end, site.site_id))
    assignments_by_site: Dict[str, set] = {}
    for assignment in catalog.assignments:
        if assignment.fully_maps_site:
            assignments_by_site.setdefault(assignment.site_id, set()).add(
                assignment.molecule_id
            )
    groups: List[List[object]] = []
    for site in sites:
        candidate = TargetedFootprintFamily(
            family_id=site.site_id,
            contig=site.contig,
            start=int(site.start),
            end=int(site.end),
            seed_intervals=((int(site.start), int(site.end)),),
            member_site_ids=(site.site_id,),
            discovery_support_molecules=int(site.population_support),
            discovery_denominator_molecules=int(site.n_fully_mapped),
            source_window_ordinals=(window.ordinal,),
        )
        eligible = [
            index
            for index, group in enumerate(groups)
            if all(geometry_compatible(candidate, member, chemistry) for member in group)
        ]
        if eligible:
            selected = min(
                eligible,
                key=lambda index: (
                    sum(
                        abs(candidate.start - member.start)
                        + abs(candidate.end - member.end)
                        for member in groups[index]
                    ),
                    tuple(member.family_id for member in groups[index]),
                ),
            )
            groups[selected].append(candidate)
        else:
            groups.append([candidate])

    families = []
    for group in groups:
        weights = [max(1, member.discovery_support_molecules) for member in group]
        start = _weighted_integer_median([member.start for member in group], weights)
        end = _weighted_integer_median([member.end for member in group], weights)
        if end <= start or not window.core_start <= (start + end) / 2.0 < window.core_end:
            continue
        group_site_ids = {
            value for member in group for value in member.member_site_ids
        }
        supporting_ids = set().union(
            *(assignments_by_site.get(site_id, set()) for site_id in group_site_ids)
        )
        denominator = sum(_fully_maps(molecule, start, end) for molecule in molecules)
        support = sum(
            molecule.molecule_id in supporting_ids and _fully_maps(molecule, start, end)
            for molecule in molecules
        )
        seeds = tuple(sorted({seed for member in group for seed in member.seed_intervals}))
        digest = hashlib.sha256(
            "|".join(f"{left}-{right}" for left, right in seeds).encode("ascii")
        ).hexdigest()[:12]
        families.append(
            TargetedFootprintFamily(
                family_id=f"family_{digest}",
                contig=group[0].contig,
                start=start,
                end=end,
                seed_intervals=seeds,
                member_site_ids=tuple(
                    sorted(group_site_ids)
                ),
                discovery_support_molecules=support,
                discovery_denominator_molecules=denominator,
                source_window_ordinals=(window.ordinal,),
            )
        )
    return tuple(sorted(families, key=lambda family: (family.start, family.end, family.family_id)))


def _discover_window_task(payload) -> TargetedWindowResult:
    window, selected, available, chemistry, config, site_config = payload
    started = time.perf_counter()
    sliced = _slice_molecules(selected, window)
    catalog = build_footprint_population_model(sliced, config=site_config)
    families = _families_from_catalog(
        catalog,
        sliced,
        window,
        chemistry,
        config.minimum_family_support,
        config.minimum_family_fraction,
    )
    stratum_counts: Dict[str, int] = {}
    for molecule in selected:
        name = _discovery_stratum(molecule)
        stratum_counts[name] = stratum_counts.get(name, 0) + 1
    return TargetedWindowResult(
        window=window,
        selected_molecule_ids=tuple(molecule.molecule_id for molecule in selected),
        selected_strata=tuple(sorted(stratum_counts.items())),
        available_discovery_molecules=available,
        tf_call_count=sum(len(molecule.tfs) for molecule in sliced),
        elapsed_seconds=time.perf_counter() - started,
        families=families,
    )


def _reconcile_families(
    candidates: Sequence[TargetedFootprintFamily],
    chemistry: TargetedFamilyChemistry,
) -> Tuple[TargetedFootprintFamily, ...]:
    groups: List[List[TargetedFootprintFamily]] = []
    active_group_indices: List[int] = []
    current_contig = None
    for family in sorted(
        candidates, key=lambda value: (value.contig, value.start, value.end, value.family_id)
    ):
        if family.contig != current_contig:
            current_contig = family.contig
            active_group_indices = []
        cutoff = family.start - chemistry.maximum_boundary_delta
        active_group_indices = [
            index
            for index in active_group_indices
            if max(member.start for member in groups[index]) >= cutoff
        ]
        eligible = [
            index
            for index in active_group_indices
            if all(
                geometry_compatible(family, member, chemistry)
                for member in groups[index]
            )
        ]
        if eligible:
            selected = min(
                eligible,
                key=lambda index: (
                    sum(
                        abs(family.start - member.start) + abs(family.end - member.end)
                        for member in groups[index]
                    ),
                    tuple(member.family_id for member in groups[index]),
                ),
            )
            groups[selected].append(family)
        else:
            groups.append([family])
            active_group_indices.append(len(groups) - 1)

    reconciled = []
    for group in groups:
        weights = [max(1, member.discovery_support_molecules) for member in group]
        start = _weighted_integer_median([member.start for member in group], weights)
        end = _weighted_integer_median([member.end for member in group], weights)
        seeds = tuple(sorted({value for member in group for value in member.seed_intervals}))
        windows = tuple(
            sorted({value for member in group for value in member.source_window_ordinals})
        )
        digest = hashlib.sha256(
            "|".join(
                (
                    group[0].contig,
                    *(f"{left}-{right}" for left, right in seeds),
                )
            ).encode("utf-8")
        ).hexdigest()[:12]
        reconciled.append(
            TargetedFootprintFamily(
                family_id=f"targeted_family_{digest}",
                contig=group[0].contig,
                start=start,
                end=end,
                seed_intervals=seeds,
                member_site_ids=tuple(
                    sorted({value for member in group for value in member.member_site_ids})
                ),
                discovery_support_molecules=max(
                    member.discovery_support_molecules for member in group
                ),
                discovery_denominator_molecules=max(
                    member.discovery_denominator_molecules for member in group
                ),
                source_window_ordinals=windows,
            )
        )
    return tuple(
        sorted(reconciled, key=lambda family: (family.contig, family.start, family.end, family.family_id))
    )


def discover_targeted_families(
    molecules: Sequence[BaselineMolecule],
    *,
    contig: str,
    locus_start: int,
    locus_end: int,
    chemistry: str,
    config: TargetedFamilyDiscoveryConfig = TargetedFamilyDiscoveryConfig(),
    site_config: Optional[SiteDiscoveryConfig] = None,
    workers: int = 1,
    progress: Optional[Callable[[int, int, TargetedWindowResult], None]] = None,
) -> TargetedFamilyDiscovery:
    """Discover a frozen family catalog with deterministic window parallelism."""

    if chemistry not in CHEMISTRY_PROFILES:
        raise ValueError(f"unsupported targeted-family chemistry: {chemistry}")
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be a positive integer")
    chemistry_profile = CHEMISTRY_PROFILES[chemistry]
    if site_config is None:
        site_config = SiteDiscoveryConfig(
            smoothing_sigma_bp=3.0,
            peak_distance_bp=15,
            assignment_radius_bp=max(
                10, int(round(chemistry_profile.maximum_center_delta))
            ),
            edge_compatibility_bp=chemistry_profile.maximum_boundary_delta,
            minimum_geometry_support_per_stratum=config.minimum_family_support,
            minimum_geometry_support=config.minimum_family_support,
            minimum_population_support=config.minimum_family_support,
            minimum_mapped_fraction=0.95,
        )
    windows = index_informative_windows(
        molecules,
        contig=contig,
        locus_start=locus_start,
        locus_end=locus_end,
        config=config,
    )
    molecules_by_window = _molecules_by_window_halo(molecules, windows)
    payloads = []
    for window in windows:
        if not window.informative:
            continue
        candidates = molecules_by_window[window.ordinal]
        eligible = _eligible_discovery_molecules(
            candidates, window, config=config
        )
        selected = select_discovery_molecules(eligible, window, config=config)
        payloads.append(
            (
                window,
                selected,
                len(eligible),
                chemistry_profile,
                config,
                site_config,
            )
        )

    results = []
    if workers == 1 or len(payloads) <= 1:
        iterator: Iterable[TargetedWindowResult] = map(_discover_window_task, payloads)
        for completed, result in enumerate(iterator, start=1):
            results.append(result)
            if progress is not None:
                progress(completed, len(payloads), result)
    else:
        # map() yields in task order even when workers finish out of order.  This
        # makes progress, reconciliation, and serialized artifacts reproducible.
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=_MP_CONTEXT,
        ) as executor:
            for completed, result in enumerate(
                executor.map(_discover_window_task, payloads), start=1
            ):
                results.append(result)
                if progress is not None:
                    progress(completed, len(payloads), result)
    results.sort(key=lambda result: result.window.ordinal)
    families = _reconcile_families(
        [family for result in results for family in result.families],
        chemistry_profile,
    )
    return TargetedFamilyDiscovery(
        schema="fiberhmm.targeted_family_discovery.v1",
        contig=contig,
        locus_start=locus_start,
        locus_end=locus_end,
        chemistry=chemistry,
        config=config,
        windows=windows,
        window_results=tuple(results),
        families=families,
        input_molecules=len(molecules),
    )


__all__ = [
    "CHEMISTRY_PROFILES",
    "TargetedFamilyChemistry",
    "TargetedFamilyDiscovery",
    "TargetedFamilyDiscoveryConfig",
    "TargetedFamilyWindow",
    "TargetedFootprintFamily",
    "TargetedWindowResult",
    "collapse_daf_amplification_families",
    "discover_targeted_families",
    "geometry_compatible",
    "index_informative_windows",
    "score_boundary_families_on_unbiased_cohort",
    "score_boundary_family_on_unbiased_cohort",
    "select_discovery_molecules",
]
