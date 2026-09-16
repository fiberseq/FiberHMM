"""Bounded endpoint allowance for an operational native profile statistic.

This is a joint box maximum of a density *penalty*, not a convolution or a
new probability distribution. Generative masses must remain separate. The
allowance is in integer base pairs, in addition to the native lattice cells.
"""
from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True)
def _range_max_columns(values, lower, upper):
    """Monotone inclusive row windows; O(rows * columns), including holes."""
    result = np.full(values.shape, -np.inf)
    queue = np.empty(values.shape[0], np.int64)
    for column in range(values.shape[1]):
        head = tail = cursor = 0
        for row in range(values.shape[0]):
            while cursor <= upper[row]:
                while tail > head and values[queue[tail-1], column] <= values[cursor, column]:
                    tail -= 1
                queue[tail] = cursor
                tail += 1
                cursor += 1
            while head < tail and queue[head] < lower[row]:
                head += 1
            if head < tail:
                result[row, column] = values[queue[head], column]
    return result


def bounded_edge_penalty(log_density, boundary_cells, tolerance_bp, *,
                         maximum_matrix_bytes=512*1024**2):
    """Exact joint profile over endpoint cells within a bounded bp distance.

    Cells are [left_low, left_high, right_low, right_high], inclusive integers,
    on one partition/refinement of the underlying opportunity lattice. For a
    target cell g, maximize d(h)-max(d) over ONE source cell h satisfying both
    endpoint distances <= tolerance_bp. This does not splice separate modes.

    The target-cell result is an existential profile maximum, not a constant
    density to assign to every coordinate in that cell. It is only used in a
    likelihood profile (and its identically computed simulation statistic).
    Keep the original cell masses for simulation and physical/core checks.
    """
    d = np.asarray(log_density, float)
    cells = np.asarray(boundary_cells)
    if (isinstance(tolerance_bp, (bool, np.bool_))
            or not isinstance(tolerance_bp, (int, np.integer)) or tolerance_bp < 0):
        raise ValueError('A nonnegative integer bp allowance is required')
    if (d.ndim != 1 or not len(d) or cells.shape != (len(d), 4)
            or np.any(~np.isfinite(cells)) or np.any(cells != np.rint(cells))
            or np.any(np.isnan(d) | np.isposinf(d)) or not np.isfinite(d).any()):
        raise ValueError('Finite integer boundary cells and valid log density required')
    cells = cells.astype(np.int64)
    if (np.any(cells[:, 0] > cells[:, 1]) or np.any(cells[:, 2] > cells[:, 3])
            or np.any(cells[:, 1] >= cells[:, 2])):
        raise ValueError('Every cell must contain only positive-width intervals')
    axes = []
    for pair in (cells[:, :2], cells[:, 2:]):
        # A valid endpoint partition has exactly one upper endpoint per lower
        # endpoint. Integer-vector sorting is much cheaper than NumPy's
        # structured two-column sort, and the consistency check retains the
        # same rejection of nonidentical overlapping bins.
        low, first, inverse = np.unique(pair[:, 0], return_index=True, return_inverse=True)
        high = pair[first, 1]
        if np.any(pair[:, 1] != high[inverse]):
            raise ValueError('Endpoint bins must form a disjoint integer partition')
        bins = np.c_[low, high]
        if np.any(bins[1:, 0] <= bins[:-1, 1]):
            raise ValueError('Endpoint bins must form a disjoint integer partition')
        lower = np.searchsorted(bins[:, 1], bins[:, 0]-tolerance_bp, side='left')
        upper = np.searchsorted(bins[:, 0], bins[:, 1]+tolerance_bp, side='right')-1
        axes.append((bins, inverse, lower, upper))
    left, right = axes
    shape = (len(left[0]), len(right[0]))
    if len(np.unique(left[1]*shape[1]+right[1])) != len(d):
        raise ValueError('Duplicate boundary rectangles must be quotiented first')
    penalty = d-d.max()
    if tolerance_bp == 0:
        return penalty
    # Two full matrices plus transpose-contiguous working space, a returned
    # vector, and index arrays. No genomic-span-sized or cell-pair matrix.
    estimated_bytes = 4*shape[0]*shape[1]*8 + 8*len(d) + 64*sum(shape)
    if maximum_matrix_bytes < estimated_bytes:
        raise MemoryError(f'Bounded edge profile needs {estimated_bytes} bytes; no geometry pruned')
    matrix = np.full(shape, -np.inf)
    matrix[left[1], right[1]] = penalty
    # Keep even nonphysical INTERMEDIATE cells: the source matrix contains only
    # physical geometries, and final targets are physical. Masking intermediate
    # cells would incorrectly truncate the joint rectangle query.
    first = _range_max_columns(matrix, left[2], left[3])
    del matrix
    second = _range_max_columns(first.T, right[2], right[3]).T
    return second[left[1], right[1]]
