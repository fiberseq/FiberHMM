"""Resolution of two specified interval states on one actual native lattice.

This is a forward-model diagnostic, not a matcher, caller, rescue, population
test, posterior, or shared-identity claim. Both states use the SAME supplied
observations and native per-position emissions. Shared observations cancel.
Missing opportunities never become modification misses or positive support.

Exact enumeration concerns the joint distribution of the differing informative
observations, under conditional-independent Bernoulli native emissions. Above
an explicit enumeration budget, all observations still contribute to the LR
and directional KL; exact TV/power are unavailable and information bounds are
returned. There is no scan, fitted gap, learned prior, or hidden truncation.
"""
from __future__ import annotations

import math
from numbers import Integral, Real
from typing import Mapping, Sequence

import numpy as np


DEFAULT_CONFIDENCE_LEVELS = (95., 99., 99.9)


def _intervals(items):
    spans = []
    for pair in items:
        if len(pair) != 2 or any(isinstance(v, (bool, np.bool_)) or not isinstance(v, Integral) for v in pair):
            raise ValueError('Intervals require integer half-open coordinates')
        a, b = map(int, pair)
        if a < 0 or b <= a or b > np.iinfo(np.int64).max:
            raise ValueError('Intervals require nonnegative, positive-width int64 spans')
        spans.append((a, b))
    merged = []
    for a, b in sorted(spans):
        if merged and a <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(b, merged[-1][1]))
        else:
            merged.append((a, b))
    return merged


def _membership(positions, intervals):
    result = np.zeros(len(positions), bool)
    for a, b in intervals:
        result |= (positions >= a) & (positions < b)
    return result


def _difference_intervals(left, right):
    """Set subtraction without expanding genomic base-pair ranges."""
    result = []
    for a, b in left:
        cursor = a
        for x, y in right:
            if y <= cursor:
                continue
            if x >= b:
                break
            if x > cursor:
                result.append((cursor, min(x, b)))
            cursor = max(cursor, y)
            if cursor >= b:
                break
        if cursor < b:
            result.append((cursor, b))
    return result


def _read(unit):
    p = np.asarray(unit['positions'])
    hits = np.asarray(unit['hits'])
    if p.ndim != 1 or (p.size and p.dtype.kind not in 'iu'):
        raise ValueError('One-dimensional integer opportunity coordinates required')
    if np.any(p < 0) or np.any(p > np.iinfo(np.int64).max) or np.any(p[1:] <= p[:-1]):
        raise ValueError('Opportunity coordinates must be nonnegative, unique and increasing')
    if hits.shape != p.shape or (hits.size and hits.dtype.kind not in 'biu') or np.any((hits != 0) & (hits != 1)):
        raise ValueError('One binary integer/bool observation per actual opportunity required')
    pa, pp = np.asarray(unit['p_accessible'], float), np.asarray(unit['p_protected'], float)
    if pa.shape != p.shape or pp.shape != p.shape:
        raise ValueError('Both native probability arrays must match the actual opportunity lattice')
    if any(np.any(~np.isfinite(v)) or np.any((v <= 0) | (v >= 1)) for v in (pa, pp)):
        raise ValueError('Finite native probabilities strictly between zero and one required')
    alignment = _intervals(unit['aligned_blocks']) if 'aligned_blocks' in unit else None
    if alignment is not None and not _membership(p, alignment).all():
        raise ValueError('An actual opportunity is outside the supplied alignment blocks')
    return p.astype(np.int64, copy=False), hits.astype(np.int8, copy=False), pa, pp, alignment


def _bernoulli_kl(p, q):
    """Stable directional KL, including nearly identical native emissions."""
    p, q = np.asarray(p, float), np.asarray(q, float)
    out = p*(np.log(p)-np.log(q))+(1-p)*(np.log1p(-p)-np.log1p(-q))
    delta = p-q
    close = np.abs(delta) < 1e-3*np.minimum(q, 1-q)
    if close.any():
        # D(p||q)=q*f(delta/q)+(1-q)*f(-delta/(1-q)),
        # f(x)=(1+x)*log(1+x)-x. Its convergent series avoids cancelling
        # two first-order terms. At |x|<1e-3, 12 terms exceed float precision.
        def f(x):
            result = np.zeros_like(x)
            for degree in range(2, 14):
                result += ((-1.)**degree)*x**degree/(degree*(degree-1))
            return result
        d, t = delta[close], q[close]
        out[close] = t*f(d/t)+(1-t)*f(-d/(1-t))
    if np.any(out < -1e-12):
        raise ArithmeticError('Native Bernoulli KL became materially negative')
    return np.maximum(out, 0.)


def _information_bounds(p0, p1, kl01, kl10):
    # Product Bhattacharyya affinity supplies Hellinger bounds; either
    # directional KL supplies Pinsker. These are mathematical information
    # bounds evaluated in floating point, not interval-arithmetic certificates.
    direct_log_affinity = np.logaddexp(.5*(np.log(p0)+np.log(p1)),
                                      .5*(np.log1p(-p0)+np.log1p(-p1)))
    # A direct log-affinity rounds to zero for sufficiently similar models.
    # Rationalized root differences retain its second-order information.
    root_delta = (p0-p1)/(np.sqrt(p0)+np.sqrt(p1))
    miss_root_delta = (p1-p0)/(np.sqrt(1-p0)+np.sqrt(1-p1))
    h2 = .5*(root_delta*root_delta+miss_root_delta*miss_root_delta)
    close = h2 < .25
    direct_log_affinity[close] = np.log1p(-h2[close])
    log_affinity = float(np.minimum(0., direct_log_affinity).sum())
    hellinger_lower = -math.expm1(log_affinity)
    hellinger_upper = math.sqrt(max(0., -math.expm1(2*log_affinity)))
    pinsker_upper = math.sqrt(max(0., min(kl01, kl10)/2))
    guard = 128*np.finfo(float).eps*max(1, len(p0))
    lower = max(0., hellinger_lower-guard)
    upper = min(1., hellinger_upper+guard, pinsker_upper+guard)
    return dict(total_variation_lower=lower, total_variation_upper=max(lower, upper),
        optimal_equal_prior_accuracy_lower=.5+.5*lower,
        optimal_equal_prior_accuracy_upper=.5+.5*max(lower, upper),
        log_Bhattacharyya_affinity=log_affinity,
        methods=['product_Bhattacharyya_Hellinger', 'bidirectional_Pinsker'],
        floating_point_guard=guard, interval_arithmetic_certified=False)


def _exact(p0, p1, hits, levels, include_outcomes):
    k = len(p0); count = 1 << k
    codes = np.arange(count, dtype=np.uint64)
    log0 = np.zeros(count); log1 = np.zeros(count); lr = np.zeros(count)
    observed_code = 0
    for i, (a, b, hit) in enumerate(zip(p0, p1, hits)):
        yes = ((codes >> np.uint64(i)) & np.uint64(1)).astype(bool)
        step0 = np.where(yes, math.log(a), math.log1p(-a))
        step1 = np.where(yes, math.log(b), math.log1p(-b))
        log0 += step0; log1 += step1; lr += step1-step0
        observed_code |= int(hit) << i
    prob0, prob1 = np.exp(log0), np.exp(log1)
    normalization_errors = [float(abs(prob0.sum()-1)), float(abs(prob1.sum()-1))]
    if max(normalization_errors) > 1e-8:
        raise ArithmeticError('Enumerated native outcome probabilities failed normalization')
    prob0 /= prob0.sum(); prob1 /= prob1.sum()
    # Treat machine-precision LR ties conservatively. Coarsening adjacent ties
    # preserves a nonrandomized size-controlled LR ordering; it never selects
    # arbitrary members of a tied pattern group to spend the remaining alpha.
    tolerance = 32*np.finfo(float).eps*max(1., float(np.abs(lr).max()))*max(1, k)
    order = np.argsort(lr, kind='stable'); ordered_lr = lr[order]
    group_starts = np.r_[0, np.flatnonzero(np.diff(ordered_lr) > tolerance)+1]
    group_index = np.repeat(np.arange(len(group_starts)), np.diff(np.r_[group_starts, count]))
    group0 = np.add.reduceat(prob0[order], group_starts)
    group1 = np.add.reduceat(prob1[order], group_starts)
    upper0 = np.cumsum(group0[::-1])[::-1]
    lower1 = np.cumsum(group1)
    pattern_group = np.empty(count, int); pattern_group[order] = group_index
    observed_group = int(pattern_group[observed_code])
    tests = []
    for level in levels:
        alpha = (100-level)/100
        reject_a = upper0 <= alpha
        reject_b = lower1 <= alpha
        tests.append(dict(confidence_percent=level, alpha=alpha,
            rejection_rate_under_a=float(group0[reject_a].sum()), power_under_b=float(group1[reject_a].sum()),
            rejection_rate_under_b=float(group1[reject_b].sum()), power_under_a=float(group0[reject_b].sum()),
            reject_a_rule='upper-tail log LR(B/A) under A <= alpha',
            reject_b_rule='lower-tail log LR(B/A) under B <= alpha', randomized=False))
    result = dict(available=True, outcomes_enumerated=count, informative_positions=k,
        total_variation=float(.5*np.abs(prob0-prob1).sum()),
        optimal_equal_prior_accuracy=float(.5+.25*np.abs(prob0-prob1).sum()),
        observed_upper_tail_under_a=min(1., float(upper0[observed_group])),
        observed_lower_tail_under_b=min(1., float(lower1[observed_group])),
        tests=tests, probability_normalization_absolute_errors=normalization_errors,
        LR_machine_precision_tie_tolerance_nats=tolerance,
        observed_log_lr_enumeration=float(lr[observed_code]), outcome_table=None)
    if include_outcomes:
        result['outcome_table'] = [dict(bit_code=int(code), probability_a=float(pa), probability_b=float(pb),
            log_lr_b_over_a=float(value), upper_tail_under_a=min(1.,float(upper0[group])),
            lower_tail_under_b=min(1.,float(lower1[group])))
            for code, pa, pb, value, group in zip(codes, prob0, prob1, lr, pattern_group)]
        result['outcome_bit_order'] = 'bit i is informative_positions[i]; 1 means observed modification'
    return result


def compare_interval_states(
    unit: Mapping,
    protected_a: Sequence[Sequence[int]],
    protected_b: Sequence[Sequence[int]],
    *,
    max_exact_outcomes: int = 65536,
    confidence_levels: Sequence[float] = DEFAULT_CONFIDENCE_LEVELS,
    include_position_details: bool = True,
    include_outcome_table: bool = False,
) -> dict:
    """Compare two specified protected interval unions on one native unit.

    Positive ``log_lr_b_over_a`` favors B. Coordinates are genomic half-open
    integer intervals; neither input is snapped to calls, a reference sequence,
    or another assay's opportunity lattice. Overlapping/touching intervals in
    each state are unioned, so an observation cannot be counted twice.

    ``max_exact_outcomes`` explicitly budgets 2**k outcomes, where k is the
    number of observed positions with different predicted probabilities.
    Beyond this budget, all k observations still contribute to the exact LR
    and directional KL. Exact distributional quantities become unavailable.

    An empty/difference-free observed lattice returns LR=1, KL=0 and optimal
    accuracy=.5, with ``supports_shared_identity=False``. Lack of information
    must never be counted as a supporting molecule or population recurrence.
    """
    if isinstance(max_exact_outcomes, (bool, np.bool_)) or not isinstance(max_exact_outcomes, Integral) or max_exact_outcomes < 1:
        raise ValueError('max_exact_outcomes must be a positive integer')
    levels = []
    for value in confidence_levels:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not math.isfinite(value) or not 0 < value < 100:
            raise ValueError('Confidence levels must be finite percentages strictly between 0 and 100')
        if float(value) not in levels:
            levels.append(float(value))
    a, b = _intervals(protected_a), _intervals(protected_b)
    positions, hits, pa, pp, alignment = _read(unit)
    state_a, state_b = _membership(positions, a), _membership(positions, b)
    changed = state_a != state_b
    informative = changed & (pa != pp)
    p0 = np.where(state_a[informative], pp[informative], pa[informative])
    p1 = np.where(state_b[informative], pp[informative], pa[informative])
    observed_hits = hits[informative]
    steps = np.where(observed_hits, np.log(p1)-np.log(p0), np.log1p(-p1)-np.log1p(-p0))
    log_lr = float(math.fsum(map(float, steps)))
    kl01 = float(math.fsum(map(float, _bernoulli_kl(p0, p1))))
    kl10 = float(math.fsum(map(float, _bernoulli_kl(p1, p0))))
    difference = _intervals(_difference_intervals(a, b)+_difference_intervals(b, a))
    observed_changed_bases = [(int(p), int(p)+1) for p in positions[changed]]
    missing = _difference_intervals(difference, observed_changed_bases)
    outside_alignment = _difference_intervals(difference, alignment) if alignment is not None else None
    aligned_missing = _difference_intervals(missing, outside_alignment) if alignment is not None else None
    k = int(informative.sum())
    identical = k == 0
    bounds = _information_bounds(p0, p1, kl01, kl10) if k else dict(
        total_variation_lower=0., total_variation_upper=0.,
        optimal_equal_prior_accuracy_lower=.5, optimal_equal_prior_accuracy_upper=.5,
        log_Bhattacharyya_affinity=0., methods=['identical_observed_distributions'],
        floating_point_guard=0., interval_arithmetic_certified=False)
    within_budget = k <= int(max_exact_outcomes).bit_length()-1
    can_enumerate = k < 63 and within_budget
    if can_enumerate:
        exact = _exact(p0, p1, observed_hits, levels, include_outcome_table)
        if abs(exact['observed_log_lr_enumeration']-log_lr) > 1e-10*max(1.,abs(log_lr)):
            raise ArithmeticError('Observed LR differs from exact enumerated reference')
    else:
        exact = dict(available=False,
            reason='exact_outcome_budget_exceeded' if not within_budget else 'exact_uint64_outcome_index_limit',
            outcomes_enumerated=0,
            informative_positions=k, total_variation=None, optimal_equal_prior_accuracy=None,
            observed_upper_tail_under_a=None, observed_lower_tail_under_b=None, outcome_table=None,
            tests=[dict(confidence_percent=level, alpha=(100-level)/100,
                rejection_rate_under_a=None, power_under_b=None, rejection_rate_under_b=None, power_under_a=None,
                power_under_b_upper_bound=min(1.,(100-level)/100+bounds['total_variation_upper']),
                power_under_a_upper_bound=min(1.,(100-level)/100+bounds['total_variation_upper']),
                bound='Any test with size <= alpha has power <= alpha + TV', randomized=False)
                for level in levels])
    reason = ('specified_interval_states_identical' if not difference else
              'no_observed_opportunity_in_changed_state' if not changed.any() else
              'changed_states_have_identical_native_emissions') if identical else None
    result = dict(contract_version='native_resolution_v1', unit_id=unit.get('unit_id'),
        status='identical_observed_distributions' if identical else ('exact_reference' if can_enumerate else 'information_bounds_only'),
        protected_intervals_a=[list(v) for v in a], protected_intervals_b=[list(v) for v in b],
        differing_state_intervals=[list(v) for v in difference],
        unobserved_differing_base_intervals=[list(v) for v in missing],
        differing_intervals_outside_alignment=[list(v) for v in outside_alignment] if outside_alignment is not None else None,
        aligned_differing_bases_without_recorded_opportunity=[list(v) for v in aligned_missing] if aligned_missing is not None else None,
        observed_opportunities=len(positions), unchanged_state_observations=int((~changed).sum()),
        changed_state_observations=int(changed.sum()), changed_state_equal_emission_observations=int((changed & ~informative).sum()),
        informative_observations=k, informative_hits=int(observed_hits.sum()), informative_misses=int(k-observed_hits.sum()),
        unobserved_differing_bases=sum(y-x for x,y in missing),
        informative_positions=positions[informative].tolist(), informative_probability_a=p0.tolist(), informative_probability_b=p1.tolist(),
        observed_log_lr_contributions=steps.tolist(), log_lr_b_over_a=log_lr,
        likelihood_ratio_b_over_a=math.exp(log_lr) if abs(log_lr)<700 else None,
        likelihood_ratio_requires_log_space=abs(log_lr)>=700,
        KL_a_vs_b_nats=kl01, KL_b_vs_a_nats=kl10,
        identical_observed_distributions=identical, identical_distribution_reason=reason,
        information_bounds=bounds, exact=exact, exact_max_outcomes=int(max_exact_outcomes),
        exact_required_log2_outcomes=k, exact_required_outcomes=(1<<k) if k<63 else None,
        max_exact_outcomes_includes_all_informative_positions=True,
        no_silent_observation_truncation=True, native_conditional_independence_assumed=True,
        specified_hypotheses_not_discovered_or_fitted=True,
        shared_observations_cancel_without_extra_likelihood_bonus=True,
        missing_genomic_bases_not_imputed_opportunities=True,
        supports_shared_identity=False, population_prevalence_estimated=False,
        cohort_confidence_not_available=True, calibrated_after_call_or_gap_selection=False)
    if include_position_details:
        result.update(unchanged_state_positions=positions[~changed].tolist(),
            changed_state_equal_emission_positions=positions[changed & ~informative].tolist(),
            changed_state_positions=positions[changed].tolist(),
            observed_informative_hits=observed_hits.tolist())
    return result
