"""Vectorized native predictive reference: counter-based draws, reference scoring, optional tilt.

The experiment is the reference one: the same generative geometry (cdf over the
recipient's projection classes), the same per-site native probabilities, the same
profile-loss event with the same penalties and threshold. Two things change, both
declared:

1. Uniform variates come from a counter-based generator (numpy Philox keyed by the
   record's seed) filled as one (replicates x sites+1) matrix, instead of a sequential
   MT19937 stream inside numba. Same distribution, different numbers.
2. Optionally the generating projection is drawn from a tilted distribution and the
   event is importance-weighted. The estimator is unbiased for the same tail; its
   variance is smaller when most generative mass sits where an exceedance cannot
   happen, which is the regime of a 0.001 tail. The exported interval accounts for
   the weights.

Scoring per replicate is the reference algorithm verbatim (sliding-window maximum over
monotone projection ranges, penalty scan with early exit) reading pre-drawn uniforms.
"""
from __future__ import annotations

import numpy as np
from numba import njit

from .measurement_distribution import _monotone_projection_ranges


@njit(cache=True, nogil=True)
def _events(u, g_index, pa, pp, starts, ends, log_penalty, threshold, hit_step, miss_step,
            prefix, monotone, left, lower, upper, queue, order):
    """One event flag per replicate from pre-drawn uniforms and generating projections."""
    R = u.shape[0]; k = len(pa); target = threshold-1e-10
    out = np.empty(R, np.bool_)
    for i in range(R):
        g = g_index[i]
        for j in range(k):
            p = pp[j] if starts[g] <= j < ends[g] else pa[j]
            prefix[j+1] = prefix[j]+(hit_step[j] if u[i, j] < p else miss_step[j])
        best = -np.inf
        if monotone:
            head = tail = cursor = 0
            for j in range(len(left)):
                while cursor <= upper[j]:
                    while tail > head and prefix[queue[tail-1]] <= prefix[cursor]:
                        tail -= 1
                    queue[tail] = cursor; tail += 1; cursor += 1
                while head < tail and queue[head] < lower[j]:
                    head += 1
                best = max(best, prefix[queue[head]]-prefix[left[j]])
        else:
            for h in range(len(starts)):
                best = max(best, prefix[ends[h]]-prefix[starts[h]])
        exceeds = True
        for h in order:
            if best-(best+log_penalty[h]) >= target:
                break
            value = prefix[ends[h]]-prefix[starts[h]]
            if best-(value+log_penalty[h]) < target:
                exceeds = False
                break
        out[i] = exceeds
    return out


def vectorized_predictive(pa, pp, starts, ends, log_penalty, cdf, threshold, replicates, seed,
                          *, tilt=0., block=4096):
    """Return (estimated tail, lower, upper, effective_draws, diagnostics).

    ``tilt`` in [0, 1): 0 draws the generating projection from its generative mass q
    exactly as the reference does (plain Monte Carlo with a different generator). Above
    0 the sampling distribution is (1-tilt) q + tilt * uniform over projections with
    finite penalty, and each replicate carries the weight q[g]/q_tilted[g]. The tail
    estimate is the mean weighted event, unbiased for the reference tail.
    """
    pa = np.asarray(pa, float); pp = np.asarray(pp, float)
    starts = np.asarray(starts, np.int64); ends = np.asarray(ends, np.int64)
    log_penalty = np.asarray(log_penalty, float); cdf = np.asarray(cdf, float)
    k = len(pa); P = len(starts)
    hit_step = np.log(pp/pa); miss_step = np.log1p(-pp)-np.log1p(-pa)
    prefix = np.zeros(k+1)
    monotone, left, lower, upper = _monotone_projection_ranges(starts, ends)
    queue = np.empty(k+2, np.int64); order = np.argsort(-log_penalty)
    q = np.diff(np.r_[0., cdf]); q = np.maximum(q, 0.)
    tilted = (isinstance(tilt, str) and tilt.startswith('threshold')) or (not isinstance(tilt, str) and float(tilt) > 0.)
    if tilted and q.sum() <= 0.:
        tilted = False      # no generating mass to reweight; draw exactly as the reference does
    if not tilted:
        # Plain Monte Carlo: the generating projection comes from the reference's own cdf,
        # unit weights, and the interval is the reference's Wilson interval on the count.
        cdf_s = cdf; weight = np.ones(P)
    elif isinstance(tilt, str) and tilt.startswith('threshold'):
        # Threshold-aware tilt. Under the null from projection g the loss is at most
        # (best - value_g) - penalty_g, and best - value_g is a small excess, so an
        # exceedance needs -penalty_g >= threshold - margin. Spend draws there; keep
        # a floor elsewhere so the estimator stays unbiased and finite-variance.
        margin = float(tilt.split(':')[1]) if ':' in tilt else 4.
        plausible = (-log_penalty) >= threshold-margin
        if plausible.any() and not plausible.all():
            q_s = np.where(plausible, q, .02*q)
        else:
            q_s = q.copy()
        q_s = np.where(np.isfinite(log_penalty), q_s, 0.)
    else:
        finite = np.isfinite(log_penalty)
        uniform = finite/max(1, finite.sum())
        q_s = (1.-float(tilt))*q+float(tilt)*uniform
    if tilted:
        q_s = q_s/q_s.sum()
        cdf_s = np.cumsum(q_s); cdf_s[-1] = 1.
        with np.errstate(divide='ignore', invalid='ignore'):
            weight = np.where(q_s > 0, q/q_s, 0.)
    rng = np.random.Generator(np.random.Philox(key=int(seed)))
    wsum = 0.; w2sum = 0.; done = 0; count = 0
    while done < replicates:
        n = min(block, replicates-done)
        u = rng.random((n, k+1))
        g = np.searchsorted(cdf_s, u[:, 0])
        g = np.minimum(g, P-1)
        flags = _events(u[:, 1:], g, pa, pp, starts, ends, log_penalty, threshold, hit_step, miss_step,
                        prefix, monotone, left, lower, upper, queue, order)
        w = weight[g]*flags
        wsum += float(w.sum()); w2sum += float((w*w).sum()); count += int(flags.sum()); done += n
    R = float(replicates)
    if not tilted:
        from .measurement_distribution import predictive_count_record
        record = predictive_count_record({}, count, replicates)
        lower, upper = record['predictive_tail_interval']
        return record['predictive_tail'], lower, upper, replicates, dict(raw_events=count, weighted_events=float(count), tilt=tilt)
    tail = wsum/R
    # variance of the weighted mean
    var = max(0., (w2sum/R-tail*tail)/R)
    half = 1.959963984540054*np.sqrt(var)
    if not np.isfinite(tail):
        tail, half = 0., 0.
    return tail, max(0., tail-half), min(1., tail+half), replicates, dict(raw_events=count, weighted_events=wsum, tilt=tilt)
