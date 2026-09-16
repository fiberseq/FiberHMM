"""Exact summaries of fitted, positive-width INTEGER boundary cells.

Gaussian location/precision parameters describe an untruncated potential, not
the moments of its normalized finite positive-width distribution. In particular
the Gaussian location can have left >= right. Never use that location as an
observed footprint, a protected core, or a credible nomination envelope.
"""
from __future__ import annotations

import numpy as np
from scipy.special import logsumexp


def _marginal_quantiles(low, high, weights, probabilities):
    """Discrete uniform mixtures, without allocating the genomic span."""
    cuts, inverse = np.unique(np.r_[low, high+1], return_inverse=True)
    rate = weights/(high-low+1)
    changes = np.bincount(inverse, weights=np.r_[rate, -rate], minlength=len(cuts))
    density = np.maximum(0., np.cumsum(changes)[:-1])
    mass = density*np.diff(cuts)
    mass /= mass.sum()
    cdf = np.cumsum(mass); cdf[-1] = 1.
    result = []
    for probability in probabilities:
        i = min(len(mass)-1, int(np.searchsorted(cdf, probability)))
        before = cdf[i-1] if i else 0.
        # The quantile is an integer with P(boundary <= q) >= probability.
        offset = int(np.ceil(max(0., probability-before)/mass[i]*(cuts[i+1]-cuts[i])-1e-12))-1
        result.append(int(cuts[i]+np.clip(offset, 0, cuts[i+1]-cuts[i]-1)))
    return result


def summarize_boundary_cells(log_mass, left_low, left_high, right_low, right_high,
                             *, credible_levels=(.95, .999)):
    """Moments and conservative joint boxes of the ACTUAL normalized model.

    Each cell has constant density over its integer-coordinate rectangle.
    Boundary boxes use Bonferroni marginal tails; their actual joint mass is
    recomputed, not assumed. Width includes within-cell lattice uncertainty.
    This is conditional on a fitted model, not a frequentist coverage claim.
    """
    mass = np.asarray(log_mass, float)
    arrays = [np.asarray(v) for v in (left_low, left_high, right_low, right_high)]
    if (mass.ndim != 1 or not len(mass) or any(a.shape != mass.shape for a in arrays)
            or any(np.any(~np.isfinite(a)) or np.any(a != np.rint(a)) for a in arrays)
            or np.any(np.isnan(mass) | np.isposinf(mass)) or not np.isfinite(mass).any()):
        raise ValueError('Finite integer cells and normalizable log masses required')
    ll, lh, rl, rh = [a.astype(np.int64) for a in arrays]
    if np.any(ll > lh) or np.any(rl > rh) or np.any(lh >= rl):
        raise ValueError('Every rectangle must contain only positive-width intervals')
    if any(not np.isfinite(p) or not 0 < p < 1 for p in credible_levels):
        raise ValueError('Credible levels must lie strictly between zero and one')
    weights = np.exp(mass-logsumexp(mass))
    coordinates = np.c_[(ll+lh)/2., (rl+rh)/2.]
    # Subtract an origin before the moment calculation; genomic coordinates
    # squared directly would lose small variances to catastrophic cancellation.
    origin = coordinates[0]
    mean_delta = weights @ (coordinates-origin)
    mean = origin+mean_delta
    residual = coordinates-origin-mean_delta
    covariance = (residual*weights[:, None]).T @ residual
    covariance += np.diag(weights @ np.c_[((lh-ll+1.)**2-1)/12., ((rh-rl+1.)**2-1)/12.])
    boxes = {}
    for level in credible_levels:
        tail = (1-level)/4.
        left = _marginal_quantiles(ll, lh, weights, (tail, 1-tail))
        right = _marginal_quantiles(rl, rh, weights, (tail, 1-tail))
        lf = np.maximum(0, np.minimum(lh, left[1])-np.maximum(ll, left[0])+1)/(lh-ll+1.)
        rf = np.maximum(0, np.minimum(rh, right[1])-np.maximum(rl, right[0])+1)/(rh-rl+1.)
        contained = float(weights @ (lf*rf))
        if contained+1e-10 < level:
            raise AssertionError('Boundary box undercovers its stated conditional mass')
        boxes[f'{level:g}'] = dict(left=left, right=right, actual_mass=contained)
    return dict(mean=mean.tolist(), covariance=covariance.tolist(),
                mean_width=float(mean[1]-mean[0]), credible_boxes=boxes,
                geometry_cells=len(mass), integer_within_cell_uncertainty=True,
                semantics='normalized finite positive-width model; conditional on fit')


def summarize_grid_geometry(grid, log_mass):
    """Adapter for the shared opportunity-grid representation."""
    coordinates = grid['coordinates']
    return summarize_boundary_cells(log_mass,
        np.rint(2*coordinates[:, 0]-grid['left_hi']), grid['left_hi'],
        grid['right_lo'], np.rint(2*coordinates[:, 1]-grid['right_lo']))


def projection_coverage_probability(log_mass, starts, ends, n_positions):
    """P(geometry covers position j) on its exactly refined opportunity grid.

    Every cell must project identically onto these positions. Refine a foreign
    source grid at recipient positions before using this routine; do not round
    a cell midpoint onto a new lattice. This includes the complete distribution,
    not only the MAP or an intersection of representative intervals.
    """
    mass = np.asarray(log_mass, float)
    a, b = np.asarray(starts), np.asarray(ends)
    if (mass.ndim != 1 or a.shape != mass.shape or b.shape != mass.shape
            or a.dtype.kind not in 'iu' or b.dtype.kind not in 'iu'
            or np.any(a < 0) or np.any(a >= b) or np.any(b > n_positions)
            or np.any(np.isnan(mass) | np.isposinf(mass)) or not np.isfinite(mass).any()):
        raise ValueError('Normalized positive projection cells required')
    w = np.exp(mass-logsumexp(mass))
    delta = np.bincount(a, weights=w, minlength=n_positions+1)-np.bincount(b, weights=w, minlength=n_positions+1)
    result = np.clip(np.cumsum(delta)[:-1], 0., 1.)
    if not np.isclose(result.sum(), w @ (b-a), rtol=1e-10, atol=1e-10):
        raise AssertionError('Expected protected opportunity count changed')
    return result
