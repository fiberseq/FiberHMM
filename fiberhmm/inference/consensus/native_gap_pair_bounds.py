"""Positive, log-stable partition bounds for all separated internal-gap pairs.

This is a computational integration aid, not a top-k model or a biological
confidence score. An unevaluated pair remains in the declared model universe
until its total mass is bounded. Priors must be applied consistently by the
caller; these functions return unnormalized positive likelihood sums.
"""
from __future__ import annotations

import numpy as np
from scipy.special import logsumexp


class SeparatedGapUniverse:
    def __init__(self, gaps, minimum_flank_bp=1):
        h = np.asarray(gaps)
        if (h.ndim != 2 or h.shape[1] != 2 or not len(h)
                or np.any(~np.isfinite(h)) or np.any(h != np.floor(h))
                or np.any(h[:, 0] >= h[:, 1]) or len(np.unique(h, axis=0)) != len(h)):
            raise ValueError('Unique nonempty integer gap intervals required')
        if (isinstance(minimum_flank_bp, bool) or not isinstance(minimum_flank_bp, (int, np.integer))
                or minimum_flank_bp < 1):
            raise ValueError('Positive integer protected flank required')
        self.gaps = h.astype(np.int64).copy()
        self.gaps.setflags(write=False)
        self.flank = int(minimum_flank_bp)
        self.by_end = np.argsort(h[:, 1], kind='stable')
        self.by_start = np.argsort(h[:, 0], kind='stable')
        self.n_left = np.searchsorted(h[self.by_end, 1], h[:, 0]-self.flank, side='right')
        self.first_right = np.searchsorted(h[self.by_start, 0], h[:, 1]+self.flank, side='left')

    def _active(self, active):
        a = np.asarray(active)
        if a.shape != (len(self.gaps),) or a.dtype != bool:
            raise ValueError('Boolean active mask over the complete gap universe required')
        return a

    def _weights(self, values):
        x = np.asarray(values, float)
        if x.shape != (len(self.gaps),) or np.any(np.isnan(x) | np.isposinf(x)):
            raise ValueError('One finite or log-zero weight per complete gap candidate required')
        return x

    def count_pairs(self, active):
        a = self._active(active)
        prefix = np.r_[0, np.cumsum(a[self.by_end], dtype=np.int64)]
        return int(prefix[self.n_left][a].sum(dtype=np.int64))

    def pair_log_sum(self, left_weights, right_weights, active):
        """Sum exp(w_left(h1)+w_right(h2)) over h1.end+flank <= h2.start."""
        a = self._active(active)
        left, right = self._weights(left_weights), self._weights(right_weights)
        ordered = np.where(a, left, -np.inf)[self.by_end]
        prefix = np.r_[-np.inf, np.logaddexp.accumulate(ordered)]
        return float(logsumexp(np.where(a, right+prefix[self.n_left], -np.inf)))

    def incident_log_sums(self, left_weights, right_weights, active):
        """Each anchor's entire incident-pair bound, respecting orientation."""
        a = self._active(active)
        left, right = self._weights(left_weights), self._weights(right_weights)
        left_prefix = np.r_[-np.inf, np.logaddexp.accumulate(np.where(a, left, -np.inf)[self.by_end])]
        ordered = np.where(a, right, -np.inf)[self.by_start]
        right_suffix = np.r_[np.logaddexp.accumulate(ordered[::-1])[::-1], -np.inf]
        both = np.logaddexp(right+left_prefix[self.n_left], left+right_suffix[self.first_right])
        return np.where(a, both, -np.inf)

    def compatible_indices(self, anchor, active):
        a = self._active(active)
        if not isinstance(anchor, (int, np.integer)) or not 0 <= anchor < len(self.gaps):
            raise ValueError('Valid anchor index required')
        s, t = self.gaps[anchor]
        return np.flatnonzero(a & ((self.gaps[:, 1]+self.flank <= s) |
                                  (self.gaps[:, 0] >= t+self.flank)))


def certified_log_partition(evaluated_log_sum, omitted_log_upper):
    """Interval enclosing a positive partition and its omitted posterior mass.

    No claim of numerical interval-arithmetic certification is made: input
    likelihoods are floating-point native calculations. The omitted-*model*
    mass bound is rigorous conditional on those calculations and upper bounds.
    """
    lower, omitted = float(evaluated_log_sum), float(omitted_log_upper)
    if np.isnan(lower) or np.isnan(omitted) or np.isposinf(lower) or np.isposinf(omitted):
        raise ValueError('Finite or log-zero positive partition bounds required')
    upper = float(np.logaddexp(lower, omitted))
    if np.isneginf(upper):
        return dict(log_lower=lower, log_upper=upper, omitted_mass_upper=0.)
    mass = 0. if np.isneginf(omitted) else float(np.exp(omitted-upper))
    return dict(log_lower=lower, log_upper=upper, omitted_mass_upper=mass)


def predictive_log_ratio_bounds(full, train, *, exact_baseline_predictive=0.):
    """Separate full/train partitions; never average ratios with wrong weights."""
    z = float(exact_baseline_predictive)
    return [full['log_lower']-train['log_upper']-z,
            full['log_upper']-train['log_lower']-z]
