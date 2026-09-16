"""Exact shared internal-pattern contrasts with the full frozen outer law.

A candidate gap is applied only to outer geometries that contain it with the
declared flanks. All other geometries remain continuous: no source mass is
discarded or renormalized to a tiny admissible tail. This changes an explicit
latent pattern model, not native calls/families, and carries no FDR/occupancy
interpretation. Parameter-estimation uncertainty is not supplied by this kernel.
"""
from __future__ import annotations

import numpy as np
from scipy.special import logsumexp


def _intervals(value, name):
    a = np.asarray(value)
    if (a.ndim != 2 or a.shape[1] != 2 or not np.issubdtype(a.dtype, np.number)
            or np.any(~np.isfinite(a)) or np.any(a != np.floor(a))
            or np.any(a[:, 0] >= a[:, 1])):
        raise ValueError(name + ' requires nonempty integer half-open intervals')
    return a.astype(np.int64)


def _corners(intervals, weights, gaps, flank):
    """Log sums over containing geometries and their disjoint complement.

    No subtraction of nearly equal cumulative sums. Duplicate finite states
    add their probability mass, rather than being counted as observations.
    """
    left, right = np.unique(intervals[:, 0]), np.unique(intervals[:, 1])
    w = np.full((len(left), len(right)), -np.inf)
    np.logaddexp.at(w, (np.searchsorted(left, intervals[:, 0]),
                       np.searchsorted(right, intervals[:, 1])), weights)
    prefix_left = np.logaddexp.accumulate(w, axis=0)
    upper_right = np.logaddexp.accumulate(prefix_left[:, ::-1], axis=1)[:, ::-1]
    upper_left = np.logaddexp.accumulate(prefix_left, axis=1)
    row_mass = np.logaddexp.reduce(w, axis=1)
    lower_rows = np.r_[np.logaddexp.accumulate(row_mass[::-1])[::-1], -np.inf]
    i = np.searchsorted(left, gaps[:, 0] - flank, side='right') - 1
    j = np.searchsorted(right, gaps[:, 1] + flank, side='left')
    admitted = np.full(len(gaps), -np.inf)
    has_corner = (i >= 0) & (j < len(right))
    admitted[has_corner] = upper_right[i[has_corner], j[has_corner]]
    # Complement = all rows a > s-flank plus rows a <= s-flank, b < t+flank.
    low_end = np.full(len(gaps), -np.inf)
    has_low = (i >= 0) & (j > 0)
    low_end[has_low] = upper_left[i[has_low], j[has_low]-1]
    complement = np.logaddexp(lower_rows[i+1], low_end)
    total = float(logsumexp(weights))
    if not np.allclose(np.logaddexp(admitted, complement), total, atol=5e-10, rtol=0):
        raise ArithmeticError('Containing and complementary geometry do not conserve total mass')
    return admitted, complement, total


def internal_gap_scores(observation, outer_intervals, log_outer_weights, gaps, *,
                        protected_background=(), minimum_flank_bp=1, existing_gaps=(),
                        moment_orders=()):
    """Marginalize every outer geometry for each specified shared gap candidate.

    S(g,h) = g minus h if h fits properly within g; otherwise S(g,h) = g.
    Optional existing gaps are applied by the same rule under BOTH hypotheses.
    Candidate gaps must be separated from these fixed gaps by the declared
    protected flank. This is an exact conditional extension, not a joint search
    or marginalization over the existing-gap parameters.
    The fixed background is protected under BOTH hypotheses. Missing positions
    never become observations. The caller must supply exact finite outer-state
    masses (refine native constant-density cells to integer states if needed).
    """
    g, h = _intervals(outer_intervals, 'outer'), _intervals(gaps, 'gaps')
    if not len(g) or not len(h):
        raise ValueError('At least one outer state and gap candidate required')
    if not isinstance(minimum_flank_bp, (int, np.integer)) or minimum_flank_bp < 1:
        raise ValueError('Positive integer minimum protected flank required')
    orders = np.asarray(moment_orders, float)
    if orders.ndim != 1 or np.any(~np.isfinite(orders)) or np.any(orders < 0):
        raise ValueError('Finite nonnegative moment orders required')
    existing = (_intervals(existing_gaps, 'existing gaps') if len(existing_gaps)
                else np.empty((0, 2), np.int64))
    for i, (s, t) in enumerate(existing):
        if np.any(~((h[:, 1]+minimum_flank_bp <= s) | (h[:, 0] >= t+minimum_flank_bp))):
            raise ValueError('Candidate and existing gaps require separated protected flanks')
        other = existing[i+1:]
        if np.any(~((other[:, 1]+minimum_flank_bp <= s) | (other[:, 0] >= t+minimum_flank_bp))):
            raise ValueError('Existing gaps require separated protected flanks')
    lm = np.asarray(log_outer_weights, float)
    if (lm.shape != (len(g),) or np.any(np.isnan(lm) | np.isposinf(lm))
            or not np.isfinite(lm).any()):
        raise ValueError('Finite nonzero outer probability mass required')
    input_normalizer = float(logsumexp(lm))
    lm = lm - input_normalizer
    p = np.asarray(observation['positions'])
    y = np.asarray(observation['hits'])
    pa, pp = (np.asarray(observation[k], float) for k in ('p_accessible', 'p_protected'))
    if (p.ndim != 1 or any(a.shape != p.shape for a in (y, pa, pp))
            or np.any(~np.isfinite(p)) or np.any(p != np.floor(p))
            or np.any(np.diff(p) <= 0) or np.any((y != 0) & (y != 1))
            or np.any(~np.isfinite(pa)) or np.any(~np.isfinite(pp))
            or np.any((pa <= 0) | (pa >= 1) | (pp <= 0) | (pp >= 1))):
        raise ValueError('Actual sorted unique opportunities and native probabilities required')
    p = p.astype(np.int64)
    frozen = np.zeros(len(p), bool)
    if len(protected_background):
        for a, b in _intervals(protected_background, 'background'):
            frozen |= (p >= a) & (p < b)
    values = np.where(y, np.log(pp/pa), np.log1p(-pp)-np.log1p(-pa))
    values[frozen] = 0.
    prefix = np.r_[0., np.cumsum(values)]
    left, right = np.searchsorted(p, g[:, 0]), np.searchsorted(p, g[:, 1])
    continuous_lr = prefix[right] - prefix[left]
    continuous = float(logsumexp(lm + continuous_lr))
    outer_lr = continuous_lr.copy()
    for s, t in existing:
        inside = (g[:, 0]+minimum_flank_bp <= s) & (g[:, 1] >= t+minimum_flank_bp)
        gap_value = prefix[np.searchsorted(p, t)]-prefix[np.searchsorted(p, s)]
        outer_lr[inside] -= gap_value
    a, c, b = _corners(g, lm + outer_lr, h, minimum_flank_bp)
    prior_a, _, _ = _corners(g, lm, h, minimum_flank_bp)
    hs, ht = np.searchsorted(p, h[:, 0]), np.searchsorted(p, h[:, 1])
    gap_lr = prefix[ht] - prefix[hs]
    refined = np.logaddexp(a - gap_lr, c)
    contrast = refined - b
    # This identity is structural, not merely a tolerance around exp/log noise.
    contrast[(gap_lr == 0.) | ~np.isfinite(a)] = 0.
    moments = np.array([np.logaddexp(a-order*gap_lr, c)-b for order in orders])
    if len(orders):
        moments[:, (gap_lr == 0.) | ~np.isfinite(a)] = 0.
        moments[orders == 0.] = 0.
    informative = np.r_[0, np.cumsum(~frozen & (pa != pp))]
    count = np.r_[0, np.cumsum(~frozen)]
    hits = np.r_[0, np.cumsum(~frozen & (y == 1))]
    baseline_contrast = b-continuous
    return dict(gaps=h, log_refined_vs_continuous=contrast+baseline_contrast,
                log_refined_vs_current_pattern=contrast,
                log_current_pattern_vs_continuous=baseline_contrast,
                log_current_pattern_vs_common_background=b,
                log_refined_vs_common_background=refined,
                log_continuous_vs_common_background=continuous,
                gap_protected_vs_accessible_log_lr=gap_lr,
                moment_orders=orders, gap_log_moments_vs_current_pattern=moments,
                gap_maximum_log_gain=np.where(np.isfinite(a), np.maximum(0., -gap_lr), 0.),
                log_outer_prior_mass_admitting_gap=prior_a,
                log_outer_posterior_mass_admitting_gap=np.minimum(a-b, 0.),
                gap_observed_opportunities=count[ht]-count[hs],
                gap_informative_opportunities=informative[ht]-informative[hs],
                gap_hits=hits[ht]-hits[hs],
                actual_observed_opportunities=len(p),
                common_background_opportunities=int(frozen.sum()),
                outer_input_log_normalizer=input_normalizer,
                outer_marginal_mass_preserved=True,
                unadmitted_outer_geometries_remain_continuous=True,
                conditioning_on_gap_admissibility=False,
                existing_gaps=existing,
                existing_gap_parameters_conditioned_not_marginalized=True,
                minimum_flank_bp=int(minimum_flank_bp), calibrated=False)


def shared_gap_posterior(log_ratios, *, evidence_groups, log_prior=None):
    """A conditional shared-coordinate model, not a variable-gap mixture fit.

    Evidence-group uniqueness is checked; physical independence must be justified
    separately. A held-out predictive score must use an outer model and a gap
    posterior whose training excludes the recipient, including indirect fitting.
    """
    values = np.asarray(log_ratios, float)
    if values.ndim != 2 or not values.shape[1] or np.any(~np.isfinite(values)):
        raise ValueError('Finite unit-by-gap marginal contrast matrix required')
    if len(evidence_groups) != len(values) or len(set(evidence_groups)) != len(values):
        raise ValueError('One distinct evidence group per matrix row required')
    prior = (np.full(values.shape[1], -np.log(values.shape[1])) if log_prior is None
             else np.asarray(log_prior, float))
    if (prior.shape != (values.shape[1],) or np.any(np.isnan(prior) | np.isposinf(prior))
            or not np.isfinite(prior).any()):
        raise ValueError('A declared proper gap prior is required')
    prior = prior - logsumexp(prior)
    joint = prior + values.sum(axis=0)
    z = float(logsumexp(joint))
    return dict(log_posterior=joint-z, log_prior=prior,
                log_shared_gap_vs_continuous=z, evidence_groups=list(evidence_groups),
                units=len(values), gap_candidates=values.shape[1],
                shared_gap_coordinates=True, calibrated=False,
                source_parameter_uncertainty_included=False)


def predictive_gap_log_ratio(log_ratios, log_gap_weights):
    values, weights = np.asarray(log_ratios, float), np.asarray(log_gap_weights, float)
    if (values.ndim != 1 or values.shape != weights.shape or not len(values)
            or np.any(~np.isfinite(values)) or np.any(np.isnan(weights) | np.isposinf(weights))
            or not np.isfinite(weights).any()):
        raise ValueError('Complete finite contrasts and nonzero gap probability required')
    return float(logsumexp(values + weights) - logsumexp(weights))
