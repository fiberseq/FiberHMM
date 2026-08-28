"""Optional deterministic CUDA kernels for targeted-family likelihoods.

The CPU implementations in :mod:`fiberhmm.inference.strand_rescue` remain the
scientific reference.  This module is deliberately lazy: importing FiberHMM
does not import PyTorch or initialize CUDA, and installations without PyTorch
retain the complete CPU workflow.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence, Tuple

import numpy as np


class CudaLikelihoodUnavailable(ValueError):
    """Raised when an explicitly requested CUDA backend cannot be used."""


def cuda_runtime_status() -> Mapping[str, object]:
    """Return a non-throwing description of the optional PyTorch CUDA runtime."""

    try:
        import torch
    except (ImportError, OSError) as error:
        return {
            "available": False,
            "reason": f"PyTorch could not be imported: {error}",
        }
    try:
        available = bool(torch.cuda.is_available())
    except (OSError, RuntimeError) as error:
        return {
            "available": False,
            "torch_version": str(torch.__version__),
            "cuda_build": str(torch.version.cuda or ""),
            "reason": f"CUDA probing failed: {error}",
        }
    status = {
        "available": available,
        "torch_version": str(torch.__version__),
        "cuda_build": str(torch.version.cuda or ""),
    }
    if not available:
        status["reason"] = "PyTorch reports no accessible CUDA device"
        return status
    try:
        status["device_count"] = int(torch.cuda.device_count())
        status["device_name"] = str(torch.cuda.get_device_name(0))
    except (OSError, RuntimeError) as error:
        status["available"] = False
        status["reason"] = f"CUDA device inspection failed: {error}"
    return status


def resolve_likelihood_backend(requested: str) -> Tuple[str, Mapping[str, object]]:
    """Resolve ``auto|cpu|cuda`` without silently weakening explicit CUDA."""

    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError("likelihood backend must be 'auto', 'cpu', or 'cuda'")
    if requested == "cpu":
        return "cpu", {"available": True, "reason": "CPU explicitly selected"}
    status = cuda_runtime_status()
    if bool(status.get("available")):
        return "cuda", status
    if requested == "cuda":
        raise CudaLikelihoodUnavailable(
            "CUDA likelihood backend requested but unavailable: "
            + str(status.get("reason", "unknown CUDA runtime error"))
        )
    return "cpu", status


def recommend_cuda_read_chunk_size(
    reads: Sequence[object],
    *,
    interval_chunk_size: int,
    maximum_envelope_width: int,
    minimum: int,
    target_free_memory_fraction: float = 0.70,
    maximum: int = 4096,
) -> Tuple[int, Mapping[str, object]]:
    """Choose a conservative batch that intentionally uses available VRAM."""

    if not reads:
        return max(1, int(minimum)), {"mode": "empty_cohort"}
    if not 0.0 < target_free_memory_fraction < 1.0:
        raise ValueError("CUDA target memory fraction must be inside (0,1)")
    if minimum < 1 or maximum < 1 or maximum < minimum:
        raise ValueError("invalid CUDA read chunk bounds")
    if maximum_envelope_width < 1:
        raise ValueError("maximum CUDA family envelope width must be positive")
    try:
        import torch
    except (ImportError, OSError) as error:
        raise CudaLikelihoodUnavailable(
            f"PyTorch is required for CUDA likelihood evaluation: {error}"
        ) from error
    if not bool(torch.cuda.is_available()):
        raise CudaLikelihoodUnavailable(
            "CUDA likelihood backend requested but PyTorch reports no accessible device"
        )
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    maximum_opportunities = max(int(np.asarray(read.positions).size) for read in reads)
    # Persistent row storage is int64 positions plus float64 prefixes *and*
    # steps (24 bytes/opportunity). Spatial-null kernel storage includes two
    # expanded boundaries, two search indices, interval scores, and reduction
    # workspace. Diffuse integration transiently materializes several
    # (rows, quadrature points, envelope opportunities) float64 tensors, so its
    # peak is quadratic in envelope width. Four tensor-equivalents plus the
    # extra 25% guard conservatively cover broadcast temporaries, masks,
    # allocator fragmentation, and implementation details. Empirical RTX 5090
    # throughput peaks around 4k rows; larger batches run slower even when they
    # fit in VRAM.
    working_width = max(int(interval_chunk_size), 256)
    diffuse_quadrature_points = max(
        16, (int(maximum_envelope_width) + 2) // 2
    )
    diffuse_bytes_per_row = (
        32 * diffuse_quadrature_points * int(maximum_envelope_width)
    )
    estimated_bytes_per_row = int(
        1.25
        * (
            24 * (maximum_opportunities + 1)
            + 48 * working_width
            + diffuse_bytes_per_row
        )
    )
    target_bytes = int(float(free_bytes) * target_free_memory_fraction)
    budget_rows = max(1, target_bytes // max(1, estimated_bytes_per_row))
    recommended = min(maximum, budget_rows)
    return int(recommended), {
        "mode": "vram_aware",
        "free_bytes_at_planning": int(free_bytes),
        "total_bytes": int(total_bytes),
        "target_free_memory_fraction": float(target_free_memory_fraction),
        "target_bytes": int(target_bytes),
        "maximum_opportunities_per_molecule": int(maximum_opportunities),
        "maximum_family_envelope_width": int(maximum_envelope_width),
        "diffuse_quadrature_points": int(diffuse_quadrature_points),
        "estimated_diffuse_bytes_per_row": int(diffuse_bytes_per_row),
        "estimated_peak_bytes_per_row": int(estimated_bytes_per_row),
        "preferred_minimum_rows": int(minimum),
        "preferred_minimum_rows_satisfied": bool(recommended >= minimum),
        "maximum_rows": int(maximum),
        "budget_rows": int(budget_rows),
        "recommended_rows": int(recommended),
    }


def cuda_memory_snapshot() -> Mapping[str, object]:
    """Return allocator and device-memory diagnostics after a CUDA run."""

    try:
        import torch
    except (ImportError, OSError):
        return {}
    if not bool(torch.cuda.is_available()):
        return {}
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    return {
        "free_bytes": int(free_bytes),
        "total_bytes": int(total_bytes),
        "allocated_bytes": int(torch.cuda.memory_allocated()),
        "reserved_bytes": int(torch.cuda.memory_reserved()),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }


def reset_cuda_peak_memory_stats() -> None:
    """Reset run-local allocator peaks after backend resolution."""

    try:
        import torch
    except (ImportError, OSError):
        return
    if bool(torch.cuda.is_available()):
        torch.cuda.reset_peak_memory_stats()


@dataclass
class PreparedTorchSpatialNullBatch:
    """Opportunity-prefix arrays resident on one torch device.

    Molecule IDs, rather than Python object identity, map filtered family
    cohorts back to rows.  The production independent-molecule allowlist makes
    these IDs unique before a batch is prepared.
    """

    molecule_ids: Tuple[Tuple[str, str, str], ...]
    positions: Any
    prefixes: Any
    steps: Any
    row_by_molecule_id: Mapping[Tuple[str, str, str], int]
    torch: Any
    device: Any

    def row_indices(self, reads: Sequence[object]) -> np.ndarray:
        """Map a filtered family cohort to resident row ordinals."""

        try:
            return np.asarray(
                [self.row_by_molecule_id[read.molecule_id] for read in reads],
                dtype=np.int64,
            )
        except KeyError as error:
            raise ValueError("CUDA read subset is absent from the resident batch") from error

    def spatial_null_log_likelihoods(
        self,
        reads: Sequence[object],
        starts: np.ndarray,
        ends: np.ndarray,
        log_prior: np.ndarray,
        *,
        interval_chunk_size: int = 512,
    ) -> np.ndarray:
        """Evaluate a prepared spatial-null grid for a filtered read subset."""

        if isinstance(interval_chunk_size, bool) or not isinstance(
            interval_chunk_size, int
        ) or interval_chunk_size < 1:
            raise ValueError("CUDA interval chunk size must be a positive integer")
        starts = np.asarray(starts, dtype=np.int64)
        ends = np.asarray(ends, dtype=np.int64)
        log_prior = np.asarray(log_prior, dtype=np.float64)
        if starts.ndim != 1 or starts.shape != ends.shape or starts.shape != log_prior.shape:
            raise ValueError("prepared spatial-null arrays must be aligned vectors")
        if starts.size == 0:
            raise ValueError("prepared spatial-null grid must not be empty")
        if np.any(ends <= starts):
            raise ValueError("prepared spatial-null intervals must have positive width")
        if not reads:
            return np.empty(0, dtype=np.float64)
        rows = self.row_indices(reads)

        torch = self.torch
        row_tensor = torch.as_tensor(rows, dtype=torch.int64, device=self.device)
        positions = self.positions.index_select(0, row_tensor)
        prefixes = self.prefixes.index_select(0, row_tensor)
        starts_gpu = torch.as_tensor(starts, dtype=torch.int64, device=self.device)
        ends_gpu = torch.as_tensor(ends, dtype=torch.int64, device=self.device)
        log_prior_gpu = torch.as_tensor(
            log_prior, dtype=torch.float64, device=self.device
        )
        accumulated = torch.full(
            (len(reads),),
            -math.inf,
            dtype=torch.float64,
            device=self.device,
        )
        with torch.inference_mode():
            for offset in range(0, int(starts.size), interval_chunk_size):
                stop = min(offset + interval_chunk_size, int(starts.size))
                chunk_starts = starts_gpu[offset:stop].expand(len(reads), -1).contiguous()
                chunk_ends = ends_gpu[offset:stop].expand(len(reads), -1).contiguous()
                left = torch.searchsorted(positions, chunk_starts, right=False)
                right = torch.searchsorted(positions, chunk_ends, right=False)
                scores = torch.gather(prefixes, 1, right) - torch.gather(
                    prefixes, 1, left
                )
                chunk_value = torch.logsumexp(
                    scores + log_prior_gpu[offset:stop], dim=1
                )
                accumulated = torch.logaddexp(accumulated, chunk_value)
        return np.asarray(accumulated.cpu().numpy(), dtype=np.float64)

    def spatial_null_log_likelihoods_many(
        self,
        grids: "PreparedTorchSpatialNullGrids",
        *,
        interval_chunk_size: int = 512,
    ) -> np.ndarray:
        """Evaluate every resident family grid with one host synchronization."""

        if isinstance(interval_chunk_size, bool) or not isinstance(
            interval_chunk_size, int
        ) or interval_chunk_size < 1:
            raise ValueError("CUDA interval chunk size must be a positive integer")
        if str(grids.device) != str(self.device):
            raise ValueError("CUDA read and spatial-grid batches must share a device")
        torch = self.torch
        results = []
        with torch.inference_mode():
            for starts, ends, log_prior in grids.grids:
                accumulated = torch.full(
                    (len(self.molecule_ids),),
                    -math.inf,
                    dtype=torch.float64,
                    device=self.device,
                )
                interval_count = int(starts.numel())
                for offset in range(0, interval_count, interval_chunk_size):
                    stop = min(offset + interval_chunk_size, interval_count)
                    chunk_starts = starts[offset:stop].expand(
                        len(self.molecule_ids), -1
                    ).contiguous()
                    chunk_ends = ends[offset:stop].expand(
                        len(self.molecule_ids), -1
                    ).contiguous()
                    left = torch.searchsorted(
                        self.positions, chunk_starts, right=False
                    )
                    right = torch.searchsorted(
                        self.positions, chunk_ends, right=False
                    )
                    scores = torch.gather(self.prefixes, 1, right) - torch.gather(
                        self.prefixes, 1, left
                    )
                    chunk_value = torch.logsumexp(
                        scores + log_prior[offset:stop], dim=1
                    )
                    accumulated = torch.logaddexp(accumulated, chunk_value)
                results.append(accumulated)
            stacked = torch.stack(results, dim=0)
        return np.asarray(stacked.cpu().numpy(), dtype=np.float64)

    def family_spatial_diffuse_log_likelihoods_many(
        self,
        grids: "PreparedTorchSpatialNullGrids",
        *,
        interval_chunk_size: int = 512,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Evaluate three likelihoods and envelope opportunities in one epoch."""

        if not grids.anchored_grids or len(grids.anchored_grids) != len(grids.grids):
            raise ValueError("resident family and spatial-null grids must align")
        if isinstance(interval_chunk_size, bool) or not isinstance(
            interval_chunk_size, int
        ) or interval_chunk_size < 1:
            raise ValueError("CUDA interval chunk size must be a positive integer")
        if str(grids.device) != str(self.device):
            raise ValueError("CUDA read and family-grid batches must share a device")
        torch = self.torch
        family_results = []
        spatial_results = []
        diffuse_results = []
        opportunity_results = []
        quadrature_cache = {}
        with torch.inference_mode():
            for family_grid, spatial_grid in zip(
                grids.anchored_grids, grids.grids
            ):
                candidate_starts, candidate_ends, class_members, envelope = family_grid
                candidate_count = int(candidate_starts.numel())
                candidate_scores = torch.empty(
                    (len(self.molecule_ids), candidate_count),
                    dtype=torch.float64,
                    device=self.device,
                )
                for offset in range(0, candidate_count, interval_chunk_size):
                    stop = min(offset + interval_chunk_size, candidate_count)
                    expanded_starts = candidate_starts[offset:stop].expand(
                        len(self.molecule_ids), -1
                    ).contiguous()
                    expanded_ends = candidate_ends[offset:stop].expand(
                        len(self.molecule_ids), -1
                    ).contiguous()
                    left = torch.searchsorted(
                        self.positions, expanded_starts, right=False
                    )
                    right = torch.searchsorted(
                        self.positions, expanded_ends, right=False
                    )
                    candidate_scores[:, offset:stop] = torch.gather(
                        self.prefixes, 1, right
                    ) - torch.gather(self.prefixes, 1, left)
                class_scores = torch.stack(
                    [
                        torch.logsumexp(
                            candidate_scores.index_select(1, members), dim=1
                        )
                        - math.log(int(members.numel()))
                        for members in class_members
                    ],
                    dim=1,
                )
                family_results.append(
                    torch.logsumexp(class_scores, dim=1)
                    - math.log(len(class_members))
                )

                starts, ends, log_prior = spatial_grid
                accumulated = torch.full(
                    (len(self.molecule_ids),),
                    -math.inf,
                    dtype=torch.float64,
                    device=self.device,
                )
                interval_count = int(starts.numel())
                for offset in range(0, interval_count, interval_chunk_size):
                    stop = min(offset + interval_chunk_size, interval_count)
                    chunk_starts = starts[offset:stop].expand(
                        len(self.molecule_ids), -1
                    ).contiguous()
                    chunk_ends = ends[offset:stop].expand(
                        len(self.molecule_ids), -1
                    ).contiguous()
                    left = torch.searchsorted(
                        self.positions, chunk_starts, right=False
                    )
                    right = torch.searchsorted(
                        self.positions, chunk_ends, right=False
                    )
                    scores = torch.gather(self.prefixes, 1, right) - torch.gather(
                        self.prefixes, 1, left
                    )
                    accumulated = torch.logaddexp(
                        accumulated,
                        torch.logsumexp(
                            scores + log_prior[offset:stop], dim=1
                        ),
                    )
                spatial_results.append(accumulated)
                envelope_starts = torch.full(
                    (len(self.molecule_ids), 1),
                    int(envelope[0]),
                    dtype=torch.int64,
                    device=self.device,
                )
                envelope_ends = torch.full(
                    (len(self.molecule_ids), 1),
                    int(envelope[1]),
                    dtype=torch.int64,
                    device=self.device,
                )
                envelope_left = torch.searchsorted(
                    self.positions, envelope_starts, right=False
                ).squeeze(1)
                envelope_right = torch.searchsorted(
                    self.positions, envelope_ends, right=False
                ).squeeze(1)
                opportunity_counts = envelope_right - envelope_left
                opportunity_results.append(opportunity_counts)
                # One rule sized for the genomic envelope integrates every
                # row's lower-degree polynomial exactly as well. Padding is
                # masked to a zero log-factor. This removes a host sync and a
                # tiny kernel sequence for every distinct opportunity count.
                envelope_width = int(envelope[1]) - int(envelope[0])
                maximum_count = min(envelope_width, int(self.steps.shape[1]))
                row_numbers = torch.arange(
                    len(self.molecule_ids),
                    dtype=torch.int64,
                    device=self.device,
                )[:, None]
                ordinals = torch.arange(
                    maximum_count,
                    dtype=torch.int64,
                    device=self.device,
                )[None, :]
                valid = ordinals < opportunity_counts[:, None]
                offsets = torch.clamp(
                    envelope_left[:, None] + ordinals,
                    max=int(self.steps.shape[1]) - 1,
                )
                selected_steps = self.steps[row_numbers, offsets]
                quadrature_points = max(16, (maximum_count + 2) // 2)
                quadrature = quadrature_cache.get(quadrature_points)
                if quadrature is None:
                    nodes, weights = np.polynomial.legendre.leggauss(
                        quadrature_points
                    )
                    rho = (nodes + 1.0) / 2.0
                    weights = weights / 2.0
                    quadrature = (
                        torch.as_tensor(
                            np.log1p(-rho),
                            dtype=torch.float64,
                            device=self.device,
                        ),
                        torch.as_tensor(
                            np.log(rho),
                            dtype=torch.float64,
                            device=self.device,
                        ),
                        torch.as_tensor(
                            np.log(weights),
                            dtype=torch.float64,
                            device=self.device,
                        ),
                    )
                    quadrature_cache[quadrature_points] = quadrature
                log_one_minus_rho, log_rho, log_weights = quadrature
                conditional_terms = torch.logaddexp(
                    log_one_minus_rho[None, :, None],
                    log_rho[None, :, None] + selected_steps[:, None, :],
                )
                conditional = torch.where(
                    valid[:, None, :],
                    conditional_terms,
                    torch.zeros((), dtype=torch.float64, device=self.device),
                ).sum(dim=2)
                diffuse = torch.logsumexp(
                    conditional + log_weights[None, :], dim=1
                )
                diffuse = torch.where(
                    opportunity_counts > 0,
                    diffuse,
                    torch.zeros_like(diffuse),
                )
                diffuse_results.append(diffuse)
            family_matrix = torch.stack(family_results, dim=0)
            spatial_matrix = torch.stack(spatial_results, dim=0)
            diffuse_matrix = torch.stack(diffuse_results, dim=0)
            opportunity_matrix = torch.stack(opportunity_results, dim=0)
        return (
            np.asarray(family_matrix.cpu().numpy(), dtype=np.float64),
            np.asarray(spatial_matrix.cpu().numpy(), dtype=np.float64),
            np.asarray(diffuse_matrix.cpu().numpy(), dtype=np.float64),
            np.asarray(opportunity_matrix.cpu().numpy(), dtype=np.int64),
        )


@dataclass
class PreparedTorchSpatialNullGrids:
    """Family spatial-null grids transferred once for a scoring run."""

    grids: Tuple[Tuple[Any, Any, Any], ...]
    torch: Any
    device: Any
    anchored_grids: Tuple[
        Tuple[Any, Any, Tuple[Any, ...], Tuple[int, int]], ...
    ] = ()


def prepare_torch_spatial_null_batch(
    reads: Sequence[object],
    *,
    device: str = "cuda",
) -> PreparedTorchSpatialNullBatch:
    """Pack immutable opportunity arrays once and transfer them to ``device``.

    ``device='cpu'`` exists for dependency-light numerical tests of the exact
    torch kernel.  Production callers resolve and validate CUDA availability
    before requesting the default device.
    """

    try:
        import torch
    except (ImportError, OSError) as error:
        raise CudaLikelihoodUnavailable(
            f"PyTorch is required for CUDA likelihood evaluation: {error}"
        ) from error
    if not reads:
        raise ValueError("cannot prepare an empty likelihood batch")
    if device == "cuda" and not bool(torch.cuda.is_available()):
        raise CudaLikelihoodUnavailable(
            "CUDA likelihood backend requested but PyTorch reports no accessible device"
        )
    molecule_ids = tuple(read.molecule_id for read in reads)
    if len(set(molecule_ids)) != len(molecule_ids):
        raise ValueError("resident CUDA batches require unique molecule IDs")
    maximum_opportunities = max(int(np.asarray(read.positions).size) for read in reads)
    sentinel = np.iinfo(np.int64).max
    positions = np.full(
        (len(reads), maximum_opportunities), sentinel, dtype=np.int64
    )
    prefixes = np.zeros(
        (len(reads), maximum_opportunities + 1), dtype=np.float64
    )
    step_values = np.zeros(
        (len(reads), maximum_opportunities), dtype=np.float64
    )
    for row, read in enumerate(reads):
        current_positions = np.asarray(read.positions, dtype=np.int64)
        current_steps = np.asarray(read.steps, dtype=np.float64)
        if current_positions.ndim != 1 or current_steps.shape != current_positions.shape:
            raise ValueError("read opportunity positions and likelihood steps must align")
        if current_positions.size > 1 and np.any(np.diff(current_positions) <= 0):
            raise ValueError(
                "read opportunity positions must be strictly increasing"
            )
        size = int(current_positions.size)
        positions[row, :size] = current_positions
        step_values[row, :size] = current_steps
        np.cumsum(
            current_steps,
            dtype=np.float64,
            out=prefixes[row, 1 : size + 1],
        )
        prefixes[row, size + 1 :] = prefixes[row, size]
    torch_device = torch.device(device)
    return PreparedTorchSpatialNullBatch(
        molecule_ids=molecule_ids,
        positions=torch.as_tensor(positions, dtype=torch.int64, device=torch_device),
        prefixes=torch.as_tensor(prefixes, dtype=torch.float64, device=torch_device),
        steps=torch.as_tensor(step_values, dtype=torch.float64, device=torch_device),
        row_by_molecule_id={value: index for index, value in enumerate(molecule_ids)},
        torch=torch,
        device=torch_device,
    )


def prepare_torch_spatial_null_grids(
    grids: Sequence[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    *,
    device: str = "cuda",
) -> PreparedTorchSpatialNullGrids:
    """Transfer all frozen family grids once for repeated read batches."""

    try:
        import torch
    except (ImportError, OSError) as error:
        raise CudaLikelihoodUnavailable(
            f"PyTorch is required for CUDA likelihood evaluation: {error}"
        ) from error
    if not grids:
        raise ValueError("cannot prepare an empty spatial-null grid collection")
    if device == "cuda" and not bool(torch.cuda.is_available()):
        raise CudaLikelihoodUnavailable(
            "CUDA likelihood backend requested but PyTorch reports no accessible device"
        )
    torch_device = torch.device(device)
    resident = []
    for starts, ends, log_prior in grids:
        starts = np.asarray(starts, dtype=np.int64)
        ends = np.asarray(ends, dtype=np.int64)
        log_prior = np.asarray(log_prior, dtype=np.float64)
        if starts.ndim != 1 or starts.shape != ends.shape or starts.shape != log_prior.shape:
            raise ValueError("prepared spatial-null arrays must be aligned vectors")
        if starts.size == 0 or np.any(ends <= starts):
            raise ValueError("prepared spatial-null grids require positive-width intervals")
        resident.append(
            (
                torch.as_tensor(starts, dtype=torch.int64, device=torch_device),
                torch.as_tensor(ends, dtype=torch.int64, device=torch_device),
                torch.as_tensor(log_prior, dtype=torch.float64, device=torch_device),
            )
        )
    return PreparedTorchSpatialNullGrids(
        grids=tuple(resident),
        torch=torch,
        device=torch_device,
    )


def prepare_torch_family_likelihood_grids(
    models: Sequence[Mapping[str, object]],
    *,
    device: str = "cuda",
) -> PreparedTorchSpatialNullGrids:
    """Transfer anchored and spatial grids for every frozen family once."""

    spatial = []
    for model in models:
        prepared = model.get("_prepared_spatial_null_grid")
        if prepared is None:
            raise ValueError("CUDA family scoring requires prepared spatial-null grids")
        spatial.append((prepared[0], prepared[1], prepared[2]))
    result = prepare_torch_spatial_null_grids(spatial, device=device)
    torch = result.torch
    anchored = []
    for model in models:
        candidates = [
            (int(interval[0]), int(interval[1]))
            for interval in model["candidate_intervals"]
        ]
        if not candidates:
            raise ValueError("CUDA family scoring requires anchored candidates")
        starts = torch.as_tensor(
            [value[0] for value in candidates],
            dtype=torch.int64,
            device=result.device,
        )
        ends = torch.as_tensor(
            [value[1] for value in candidates],
            dtype=torch.int64,
            device=result.device,
        )
        classes = []
        for record in model["geometry_classes"]:
            members = np.asarray(record["member_candidate_indices"], dtype=np.int64)
            if members.size == 0 or np.any(members < 0) or np.any(
                members >= len(candidates)
            ):
                raise ValueError("frozen boundary-family geometry class is invalid")
            classes.append(
                torch.as_tensor(members, dtype=torch.int64, device=result.device)
            )
        if not classes:
            raise ValueError("CUDA family scoring requires geometry classes")
        envelope = tuple(int(value) for value in model["envelope"])
        if len(envelope) != 2 or envelope[1] <= envelope[0]:
            raise ValueError("CUDA family scoring requires a valid envelope")
        anchored.append((starts, ends, tuple(classes), envelope))
    result.anchored_grids = tuple(anchored)
    return result
