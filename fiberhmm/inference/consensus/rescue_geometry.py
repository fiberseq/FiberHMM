# Numerical kernel promoted from the validated September 2026 consensus experiments.
"""Additional native-lattice geometry evidence required for de novo rescue.

This module does not modify native CR categorization. Population geometry is a
distribution over JOINT start/end projections, not a minimum-width rule or an
independent +/-bp acceptance box. The fitted mixture is explicitly conditional
on pre-existing called class members, not a calibrated biological prevalence.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import softmax
from scipy.special import logsumexp


def fit_geometry_mixture(conditional_geometry, base_q, unit_weights=None,
                         prior_units=1., max_iter=150, tolerance=1e-7,
                         conditional_prior=None):
    """Fit q using native conditional geometry likelihoods, marginalized over competitors.

    Input rows are P(g | family present, native data). Divide by the corresponding
    P(g | family present) from the SAME complete prior partition, not merely q0:
    compatibility with other families changes that prior. Their ratio is
    proportional to the properly normalized data likelihood conditional on g,
    with all competing states marginalized. A pseudo-unit preserves support.
    The q0 fallback is valid only for a standalone mixture with no competitors.
    """
    post = np.asarray(conditional_geometry, float)
    q0 = np.asarray(base_q, float)
    if post.ndim != 2 or post.shape[1] != len(q0):
        raise ValueError('Units by geometry matrix required')
    if np.any(~np.isfinite(post)) or np.any(post < 0) or np.any(q0 <= 0):
        raise ValueError('Finite nonnegative geometry masses and positive base prior required')
    if prior_units <= 0:
        raise ValueError('Positive pseudo-unit regularizer required')
    q0 = q0 / q0.sum()
    weights = np.ones(len(post)) if unit_weights is None else np.asarray(unit_weights, float)
    if weights.shape != (len(post),) or np.any(~np.isfinite(weights)) or np.any(weights < 0):
        raise ValueError('Nonnegative source-unit weights required')
    prior = np.broadcast_to(q0, post.shape) if conditional_prior is None else np.asarray(conditional_prior, float)
    if prior.shape != post.shape or np.any(~np.isfinite(prior)) or np.any(prior < 0):
        raise ValueError('One valid conditional geometry prior per source unit required')
    if np.any((prior == 0) & (post > 1e-14)):
        raise ValueError('Data posterior cannot assign mass to a prior-impossible geometry')
    usable = (post.sum(1) > 0) & (weights > 0)
    post, weights, prior = post[usable], weights[usable], prior[usable]
    if not len(post):
        return q0.copy(), dict(units=0, effective_units=0., iterations=0, converged=True)
    likelihood = np.divide(post, prior, out=np.zeros_like(post), where=prior > 0)
    likelihood /= likelihood.max(1, keepdims=True)
    q = q0.copy(); converged = False
    for iteration in range(max_iter):
        responsibility = likelihood * q
        responsibility /= responsibility.sum(1, keepdims=True)
        updated = ((weights[:, None] * responsibility).sum(0) + prior_units * q0) / (weights.sum() + prior_units)
        change = np.abs(updated - q).sum(); q = updated
        if change <= tolerance:
            converged = True
            break
    polish_info = None
    if not converged:
        total = weights.sum()+prior_units
        def objective(logits):
            candidate = softmax(logits)
            z = likelihood@candidate
            objective_value = -(weights@np.log(z)+prior_units*(q0@np.log(candidate)))
            gradient = -(candidate*(likelihood.T@(weights/z)) + prior_units*q0-total*candidate)
            return float(objective_value),gradient
        fit=minimize(objective,np.log(q),jac=True,method='L-BFGS-B',
                     options=dict(maxiter=1000,ftol=1e-12,gtol=1e-7,maxls=40))
        q=softmax(fit.x);converged=bool(fit.success)
        polish_info=dict(success=bool(fit.success),iterations=int(fit.nit),
                         message=str(fit.message),max_abs_gradient=float(np.max(np.abs(fit.jac))))
    return q, dict(units=int(len(post)), effective_units=float(weights.sum()),
                   iterations=iteration+1, converged=converged, final_l1_change=float(change),polish=polish_info)


def projected_typicality(positions, geometry_coordinates, geometry_q, candidate):
    """Smallest tie-inclusive recipient-projection HPD set containing the candidate.

    All cohort geometries that this recipient cannot distinguish are summed.
    Thus a bp-size deviation with IDENTICAL observed opportunities is not a
    geometry failure. No recipient modification outcome is used in this mapping.
    Zero-opportunity projections remain in the population prediction, rather than
    being silently renormalized away.
    """
    positions = np.asarray(positions, int)
    coordinates = np.asarray(geometry_coordinates, int)
    q = np.asarray(geometry_q, float)
    if coordinates.shape != (len(q), 2) or np.any(q < 0) or not np.isfinite(q).all() or q.sum() <= 0:
        raise ValueError('Valid geometry distribution required')
    q = q / q.sum()
    projections = np.searchsorted(positions, coordinates)
    unique, inverse = np.unique(projections, axis=0, return_inverse=True)
    masses = np.bincount(inverse, weights=q, minlength=len(unique))
    target = np.searchsorted(positions, candidate)
    same = (unique == target).all(1)
    if not same.any():
        return dict(projection_prior_mass=0., hpd_mass_before_candidate=1.,
                    in_population_projection_support=False)
    mass = float(masses[np.flatnonzero(same)[0]])
    # Include all ties at the HPD cutoff; uniform/unresolved geometries must not
    # be arbitrarily ranked by their coordinate order.
    before = float(masses[masses > mass + 1e-12].sum())
    return dict(projection_prior_mass=mass, hpd_mass_before_candidate=before,
                in_population_projection_support=True)


def population_predictive(llr, opportunity_count, physical_fraction, q):
    """Native evidence over complete population geometry; no favorable-sliver donation.

    Shape likelihood is conditioned on physical exposure and reported alongside
    its retained prior mass. The >=3-opportunity probability is measured against
    the ORIGINAL q, not the renormalized surviving sliver.
    """
    llr, opportunities, fraction, q = map(np.asarray, (llr, opportunity_count, physical_fraction, q))
    if not (llr.shape == opportunities.shape == fraction.shape == q.shape):
        raise ValueError('One value per geometry required')
    if np.any(fraction < 0) or np.any(fraction > 1+1e-9) or np.any(q < 0):
        raise ValueError('Invalid geometry mass or exposure')
    q = q / q.sum()
    mass = q * fraction
    exposed = float(mass.sum())
    observable = float(mass[opportunities >= 3].sum())
    if exposed == 0:
        return dict(population_log_bf=None, observable_log_bf=None,
                    physical_prior_mass=0., three_opportunity_prior_mass=0.)
    live = mass > 0
    log_bf = float(logsumexp(np.log(mass[live]) + llr[live]) - np.log(exposed))
    # The arbitrary-gap alternative is conditioned on >=3 opportunities. Match
    # that event for the SHAPE comparison only, avoiding dilution on just one
    # side. Absolute protection and mass gates retain the original population q.
    informative = live & (opportunities >= 3)
    observable_log_bf = (float(logsumexp(np.log(mass[informative]) + llr[informative]) - np.log(observable))
                         if observable > 0 else None)
    return dict(population_log_bf=log_bf, observable_log_bf=observable_log_bf, physical_prior_mass=exposed,
                three_opportunity_prior_mass=observable)


def arbitrary_gap_predictive(positions, native_steps, envelope, free_domains):
    """Alternative: any single protected gap inside the SAME local envelope.

    Uniform integer start/end prior, conditioned on positive width, physical
    exposure and >=3 actual opportunities. This is an auxiliary single-interval
    shape comparison, not a replacement for full-region family competition.
    All native observations on the envelope share one accessible base measure.
    """
    lo, hi = map(int, envelope)
    if hi <= lo:
        raise ValueError('Positive envelope required')
    a, b = np.triu_indices(hi-lo+1, k=1)
    a, b = a+lo, b+lo
    physical = np.zeros(len(a), bool)
    for start, end in free_domains:
        physical |= (a >= start) & (b <= end)
    ia, ib = np.searchsorted(positions, a), np.searchsorted(positions, b)
    physical &= ib-ia >= 3
    if not physical.any():
        return None
    prefix = np.r_[0., np.cumsum(native_steps)]
    llr = prefix[ib[physical]] - prefix[ia[physical]]
    return float(logsumexp(llr) - np.log(len(llr)))


def rescue_decision(record, credible_mass=.9, loss_odds=3., minimum_probability=.9,
                    minimum_testable_mass=.5, minimum_source_units=3):
    """Stricter, separately interpretable requirements; not a calibrated q value."""
    if not 0 < credible_mass <= 1 or loss_odds < 1:
        raise ValueError('Valid credible mass and likelihood-loss allowance required')
    failures = []
    if record['source_geometry_units'] < minimum_source_units:
        failures.append('insufficient_native_geometry_source')
    if record['updated_family_inclusion_mass'] < minimum_probability:
        failures.append('weak_whole_family_inclusion')
    if record['physical_prior_mass'] < minimum_testable_mass:
        failures.append('insufficient_population_geometry_exposure')
    if record['three_opportunity_prior_mass'] < minimum_testable_mass:
        failures.append('insufficient_population_geometry_information')
    if (not record['in_population_projection_support'] or
        record['hpd_mass_before_candidate'] >= credible_mass-1e-12):
        failures.append('atypical_joint_geometry_on_recipient_lattice')
    bf, patch = record['population_log_bf'], record['arbitrary_gap_log_bf']
    comparable_bf = record['observable_log_bf']
    if bf is None or bf <= 0:
        failures.append('no_native_whole_family_protection_evidence')
    if comparable_bf is None or patch is None or comparable_bf-patch < -np.log(loss_odds)-1e-12:
        failures.append('arbitrary_gap_fits_better_than_population_family')
    return dict(accepted=not failures, failures=failures)
