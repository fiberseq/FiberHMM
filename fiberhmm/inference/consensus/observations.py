# Numerical kernel promoted from the validated September 2026 consensus experiments.
"""Outcome-free local configuration catalog and native observation likelihoods.

The fixed genomic candidate grid is a computational approximation, not a claim
about assay resolution. EVERY supplied opportunity in the full domain enters
the likelihood, including observed modifications in gaps and flanks. Native
calls do not select units, boundaries, configurations, or observations here.

Configuration rows contain union-opportunity indices (a,b,c,d), with -1 padding:
empty=(-1,-1,-1,-1), single=(a,b,-1,-1), double=(a,b,c,d), a<b<c<d.
Strict b<c leaves at least one unprotected union opportunity. Adjacent protected
pieces are one observable run and are not duplicated as a two-piece state.

Returned log_likelihood_matrix is RELATIVE to the same full-domain accessible
baseline for every configuration. Add accessible_log_baseline[:,None] for
absolute observation log likelihood. Each original observation unit is one row,
regardless of its number of native calls. Empty-exposure rows remain present.
"""
from __future__ import annotations

import hashlib
from itertools import combinations
import math
from numbers import Integral

import numpy as np


DEFAULT_MAX_MATRIX_BYTES = 2 * 1024 ** 3


def _integer(value, name, minimum=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _domain(start, end, grid_bp, grid_offset, max_intervals):
    start, end = _integer(start, "start"), _integer(end, "end")
    grid_bp = _integer(grid_bp, "grid_bp", 1)
    grid_offset = _integer(grid_offset, "grid_offset", 0)
    max_intervals = _integer(max_intervals, "max_intervals", 0)
    if end <= start:
        raise ValueError("The full half-open observation domain must have positive length")
    if grid_offset >= grid_bp:
        raise ValueError("grid_offset must be smaller than grid_bp; it is a declared grid phase")
    if max_intervals > 2:
        raise ValueError("This bounded catalog supports at most two protected intervals")
    return start, end, grid_bp, grid_offset, max_intervals


def _catalog_dimensions(grid_positions, start, end, grid_bp, grid_offset, max_intervals):
    genomic_cuts = np.unique(np.r_[start, np.arange(start + grid_offset, end, grid_bp, dtype=np.int64), end]).astype(np.int64)
    projection = np.searchsorted(grid_positions, genomic_cuts).astype(np.int64)
    cuts, first = np.unique(projection, return_index=True)
    j = len(cuts)
    count = 1
    if max_intervals >= 1:
        count += math.comb(j, 2)
    if max_intervals >= 2 and j >= 4:
        count += math.comb(j, 4)
    return genomic_cuts, projection, cuts, first, count


def build_configuration_catalog(grid_positions, start, end, *, grid_bp=4,
                                grid_offset=0, max_intervals=2,
                                max_catalog_bytes=DEFAULT_MAX_MATRIX_BYTES):
    """Enumerate all grid-constrained zero/one/two-run observable states.

    Input is the full outcome-free cohort UNION opportunity lattice. Equal
    projected cuts are collapsed without looking at hit outcomes or raw calls.
    Boundary envelopes are all integer genomic coordinates with that same cut
    projection, clipped to the observation domain. Their endpoints are inclusive;
    protected intervals and the observation domain remain half-open.
    """
    start, end, grid_bp, grid_offset, max_intervals = _domain(start, end, grid_bp, grid_offset, max_intervals)
    budget = _integer(max_catalog_bytes, "max_catalog_bytes", 0)
    positions = np.asarray(grid_positions)
    if positions.ndim != 1 or (positions.size and positions.dtype.kind not in "iu"):
        raise ValueError("Opportunity positions must be a one-dimensional integer array")
    positions = np.asarray(positions, dtype=np.int64)
    if np.any(np.diff(positions) <= 0) or np.any((positions < start) | (positions >= end)):
        raise ValueError("The union lattice must be strictly increasing and inside the full domain")
    genomic_cuts, projection, cuts, first, count = _catalog_dimensions(positions, start, end, grid_bp, grid_offset, max_intervals)
    catalog_bytes = count * (4 * np.dtype(np.int64).itemsize + np.dtype(np.uint8).itemsize)
    if catalog_bytes > budget:
        raise MemoryError(f"Configuration catalog requires {catalog_bytes} bytes, exceeds {budget}; no truncation")
    configurations = np.full((count, 4), -1, dtype=np.int64)
    n_intervals = np.zeros(count, dtype=np.uint8)
    row = 1
    if max_intervals >= 1:
        for a, b in combinations(cuts.tolist(), 2):
            configurations[row, :2] = a, b
            n_intervals[row] = 1
            row += 1
    if max_intervals >= 2:
        for a, b, c, d in combinations(cuts.tolist(), 4):
            configurations[row] = a, b, c, d
            n_intervals[row] = 2
            row += 1
    if row != count:
        raise AssertionError("Catalog enumeration did not match its exact combinatorial count")
    coordinates = genomic_cuts[first].copy()
    envelopes = np.empty((len(cuts), 2), dtype=np.int64)
    coordinate_aliases = []
    for i, cut in enumerate(cuts):
        lower = start if cut == 0 else int(positions[cut - 1]) + 1
        upper = end if cut == len(positions) else int(positions[cut])
        envelopes[i] = lower, upper
        coordinate_aliases.append(genomic_cuts[projection == cut].copy())
        if cut == 0:
            coordinates[i] = start
        elif cut == len(positions):
            coordinates[i] = end
        if not lower <= coordinates[i] <= upper:
            raise AssertionError("Canonical grid coordinate lies outside its projection envelope")
    digest = hashlib.sha256()
    digest.update(np.asarray([start, end, grid_bp, grid_offset, max_intervals], dtype="<i8").tobytes())
    for value in (positions, genomic_cuts, cuts, configurations):
        digest.update(np.asarray(value.shape, dtype="<i8").tobytes())
        digest.update(value.astype("<i8", copy=False).tobytes())
    return {
        "grid_positions": positions, "genomic_grid_cuts": genomic_cuts,
        "cut_indices": cuts, "cut_coordinates": coordinates,
        "cut_coordinate_envelopes": envelopes, "cut_grid_coordinate_aliases": coordinate_aliases,
        "configurations": configurations, "n_intervals": n_intervals,
        "catalog_sha256": digest.hexdigest(), "catalog_bytes": catalog_bytes,
        "candidate_count": count, "grid_bp": grid_bp, "grid_offset": grid_offset,
        "max_intervals": max_intervals,
        "boundary_envelope_semantics": "inclusive min/max integer coordinate giving the same cohort-union cut index; not a fitted credible interval",
        "candidate_grid_semantics": "fixed outcome-free computational boundary grid; full observations are NOT binned or thinned",
    }


def configuration_log_likelihoods(log_lr, configurations, *,
                                 max_matrix_bytes=DEFAULT_MAX_MATRIX_BYTES, chunk_size=4096):
    """Prefix-sum all protected spans once; return a relative NxG matrix.

    Missing observations must already have neutral logLR zero. The matrix byte
    budget is enforced before allocation. Prefix arrays and a bounded column
    chunk add working memory, so max_matrix_bytes is not a process-RSS limit.
    """
    values = np.asarray(log_lr, dtype=float)
    configs = np.asarray(configurations)
    budget = _integer(max_matrix_bytes, "max_matrix_bytes", 0)
    chunk_size = _integer(chunk_size, "chunk_size", 1)
    if values.ndim != 2 or np.any(~np.isfinite(values)):
        raise ValueError("log_lr must be a finite units-by-opportunities matrix")
    if configs.ndim != 2 or configs.shape[1] != 4 or configs.dtype.kind not in "iu":
        raise ValueError("Configurations must be a Gx4 integer array")
    for a, b, c, d in configs:
        empty = a == b == c == d == -1
        single = 0 <= a < b <= values.shape[1] and c == d == -1
        double = 0 <= a < b < c < d <= values.shape[1]
        if not (empty or single or double):
            raise ValueError("Invalid configuration; adjacent/overlapping intervals must not duplicate protected observations")
    needed = values.shape[0] * len(configs) * np.dtype(np.float64).itemsize
    if needed > budget:
        raise MemoryError(f"Likelihood matrix requires {needed} bytes, exceeds {budget}; no truncation")
    prefix = np.c_[np.zeros(len(values)), np.cumsum(values, axis=1)]
    result = np.empty((len(values), len(configs)), dtype=np.float64)
    for first in range(0, len(configs), chunk_size):
        last = min(len(configs), first + chunk_size)
        boundaries = np.maximum(configs[first:last], 0)
        result[:, first:last] = prefix[:, boundaries[:, 1]] - prefix[:, boundaries[:, 0]]
        result[:, first:last] += prefix[:, boundaries[:, 3]] - prefix[:, boundaries[:, 2]]
    return result


def prepare_population(stratum, start, end, *, grid_bp=4, grid_offset=0,
                       max_intervals=0, max_matrix_bytes=DEFAULT_MAX_MATRIX_BYTES):
    """Prepare one stratum without selecting on calls, hit counts, or fit.

    ``units`` references the input list without copying or modifying its unit
    objects. Every row retains its unit and fold-group identity. Missing matrix
    cells have observed=False, hits=-1, pa/pp=NaN, and neutral logLR=0. Empty
    rows have accessible baseline zero and relative log likelihood zero for all
    configurations; they must not supply empirical support or precision.
    """
    start, end, grid_bp, grid_offset, max_intervals = _domain(start, end, grid_bp, grid_offset, max_intervals)
    budget = _integer(max_matrix_bytes, "max_matrix_bytes", 0)
    units = stratum["units"]
    unit_ids = [u["unit_id"] for u in units]
    if len(set(unit_ids)) != len(unit_ids):
        raise ValueError("Each observation unit must occupy exactly one row; duplicate unit IDs found")
    fold_group_ids = [u.get("fold_group_id", u["unit_id"]) for u in units]
    records = []
    for unit in units:
        positions = np.asarray(unit["positions"])
        hits = np.asarray(unit["hits"])
        pa = np.asarray(unit["p_accessible"], dtype=float)
        pp = np.asarray(unit["p_protected"], dtype=float)
        if positions.ndim != 1 or (positions.size and positions.dtype.kind not in "iu"):
            raise ValueError("Unit positions must be a one-dimensional integer array")
        if hits.shape != positions.shape or pa.shape != positions.shape or pp.shape != positions.shape:
            raise ValueError("One hit and two native probabilities are required per observation position")
        if np.any(np.diff(positions) <= 0) or np.any((hits != 0) & (hits != 1)):
            raise ValueError("Unit positions must be sorted/unique and hits must be binary")
        keep = (positions >= start) & (positions < end)
        positions = positions[keep].astype(np.int64)
        hits, pa, pp = hits[keep].astype(np.int8), pa[keep], pp[keep]
        if np.any(~np.isfinite(pa)) or np.any(~np.isfinite(pp)) or np.any((pa <= 0) | (pa >= 1) | (pp <= 0) | (pp >= 1)):
            raise ValueError("Every in-domain observed opportunity needs strict finite native probabilities")
        records.append((positions, hits, pa, pp))
    positions = np.unique(np.concatenate([r[0] for r in records])) if records else np.empty(0, dtype=np.int64)
    _, _, _, _, count = _catalog_dimensions(positions, start, end, grid_bp, grid_offset, max_intervals)
    needed = len(units) * count * np.dtype(np.float64).itemsize
    observation_bytes = len(units) * len(positions) * (8*3 + 2)
    if observation_bytes + needed > budget:
        raise MemoryError(f'Observation and evidence matrices require {observation_bytes + needed} bytes, exceeds {budget}; all units retained or fail, no sampling')
    if needed > budget:
        raise MemoryError(f"Likelihood matrix requires {needed} bytes, exceeds {budget}; no truncation")
    catalog = build_configuration_catalog(positions, start, end, grid_bp=grid_bp, grid_offset=grid_offset,
                                          max_intervals=max_intervals, max_catalog_bytes=max(budget, 33))
    shape = (len(units), len(positions))
    observed = np.zeros(shape, dtype=bool)
    hits = np.full(shape, -1, dtype=np.int8)
    pa = np.full(shape, np.nan)
    pp = np.full(shape, np.nan)
    log_lr = np.zeros(shape, dtype=float)
    baseline = np.zeros(len(units), dtype=float)
    for row, (local_positions, local_hits, local_pa, local_pp) in enumerate(records):
        columns = np.searchsorted(positions, local_positions)
        observed[row, columns] = True
        hits[row, columns], pa[row, columns], pp[row, columns] = local_hits, local_pa, local_pp
        accessible = np.where(local_hits, np.log(local_pa), np.log1p(-local_pa))
        protected = np.where(local_hits, np.log(local_pp), np.log1p(-local_pp))
        log_lr[row, columns] = protected - accessible
        baseline[row] = accessible.sum()
    likelihood = configuration_log_likelihoods(log_lr, catalog["configurations"], max_matrix_bytes=budget)
    nonempty = observed.any(axis=1)
    informative = np.any(likelihood != 0, axis=1)
    return {
        **catalog, "stratum_id": stratum.get("stratum_id"), "dataset_id": stratum.get("dataset_id"),
        "chemistry": stratum.get("chemistry"), "start": start, "end": end,
        "units": units, "unit_ids": unit_ids, "fold_group_ids": fold_group_ids,
        "observed": observed, "hits": hits, "p_accessible": pa, "p_protected": pp,
        "log_lr": log_lr, "accessible_log_baseline": baseline,
        "log_likelihood_matrix": likelihood, "relative_log_likelihood_matrix": likelihood,
        "likelihood_matrix_semantics": "protected-versus-accessible log likelihood relative to one fixed FULL-domain accessible base per row; add accessible_log_baseline[:,None] for absolute likelihood",
        "nonempty": nonempty, "informative": informative, "zero_observation_mask": ~nonempty,
        "native_observation_informative": np.any(log_lr != 0, axis=1),
        "observation_count_by_unit": observed.sum(axis=1),
        "observed_units": int(nonempty.sum()), "informative_units": int(informative.sum()),
        "total_units": len(units), "estimated_matrix_bytes": needed,
        "max_matrix_bytes": budget, "raw_call_filtering": False,
        "observation_thinning": False, "source_units_modified": False,
    }
