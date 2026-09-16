"""Latent boundary variation fitted THROUGH native observation likelihoods.

This is not a GMM on raw observed edges, nor a replacement methylation rate.
Native measurement noise remains in L_i(g). A discrete Gaussian density over
identifiable boundary cells models residual between-observation variation.
Finite-grid normalization and each call's admissibility conditioning are both
included in the fitted likelihood and analytic gradient.
"""
from __future__ import annotations

import numpy as np
from numba import njit
from scipy.optimize import minimize
from scipy.special import logsumexp


def _density(parameters, xy):
    mu = parameters[:2]
    a, b, c = np.exp(parameters[2]), parameters[3], np.exp(parameters[4])
    dx = xy-mu
    z = a*dx[:, 0]+b*dx[:, 1]
    w = c*dx[:, 1]
    log_density = -.5*(z*z+w*w)
    derivative = np.column_stack((a*z, b*z+c*w, -z*a*dx[:, 0], -z*dx[:, 1], -w*w))
    return log_density, derivative


def native_distribution_objective(parameters, xy, log_area, likelihood, allowed):
    """Negative conditional native likelihood and its EXACT gradient."""
    density, derivative = _density(parameters, xy)
    prior = np.where(allowed, density+log_area, -np.inf)
    joint = prior+likelihood
    z0, z = logsumexp(prior, axis=1), logsumexp(joint, axis=1)
    gradient = (np.exp(joint-z[:, None])-np.exp(prior-z0[:, None])).sum(0) @ derivative
    return -float((z-z0).sum()), -gradient


def _allowed_classes(allowed):
    """Byte-identical prior masks, with original row-order expansion."""
    seen={};rows=[];inverse=np.empty(len(allowed),dtype=np.int64)
    for i,row in enumerate(allowed):
        key=row.tobytes()
        if key not in seen:seen[key]=len(rows);rows.append(i)
        inverse[i]=seen[key]
    return allowed[rows],inverse


def _stable_cached_objective(density, derivative, log_area, likelihood, classes):
    """Same log-space row sums/reductions as the reference, reusing priors.

    Unlike mixing fast and stable row arithmetic, this retains the stable path
    for EVERY likelihood row. Only byte-identical admissibility priors are
    computed once, then expanded before the original ordered gradient reduction.
    """
    unique,inverse=classes
    prior=np.where(unique,density+log_area,-np.inf)
    z0=logsumexp(prior,axis=1)
    joint=prior[inverse]+likelihood
    z=logsumexp(joint,axis=1)
    prior_mass=np.exp(prior-z0[:,None])
    gradient=(np.exp(joint-z[:,None])-prior_mass[inverse]).sum(0) @ derivative
    return -float((z-z0[inverse]).sum()),-gradient


def _cached_objective(parameters, xy, log_area, likelihood, allowed, scaled_likelihood, offset, exposure, classes=None):
    """Algebraically identical likelihood/gradient, reusing native emissions.

    Parameter updates only change q. Avoid exponentiating an entire
    source-by-geometry native matrix at every optimizer step. Extreme proposals
    fall back to the stable log-space reference, rather than clipping evidence.
    """
    density, derivative = _density(parameters, xy)
    log_q = density+log_area
    log_q -= logsumexp(log_q)
    q = np.exp(log_q)
    z, z0 = scaled_likelihood @ q, exposure @ q
    if np.any(z < 1e-180) or np.any(z0 < 1e-180):
        return _stable_cached_objective(density,derivative,log_area,likelihood,
            _allowed_classes(allowed) if classes is None else classes)
    weights = q*(scaled_likelihood.T @ (1./z)-exposure.T @ (1./z0))
    return -float((np.log(z)-np.log(z0)+offset).sum()), -(weights @ derivative)


def fit_native_distribution(likelihood, allowed, coordinates, areas, *, reference,
                            max_iterations=100, initial=None, progress=None,
                            objective_backend='cpu', accelerator_bytes=256*1024**2,
                            smoothing_pseudo_units=4.):
    """A single latent shape distribution, not a fitted occupancy prior.

    Two width initializations guard against local optima. All source units and
    projection classes enter each fit. Width bounds are numerical limits, not
    bp search windows; the full caller-domain grid is retained at every width.
    Fit failure/nonconvergence is exported rather than concealed as confidence.
    """
    ll = np.asarray(likelihood, float); mask = np.asarray(allowed, bool)
    coords = np.asarray(coordinates, float); area = np.asarray(areas, float)
    reference = np.asarray(reference, float)
    if (ll.ndim != 2 or not len(ll) or mask.shape != ll.shape or coords.shape != (ll.shape[1], 2)
            or area.shape != (ll.shape[1],) or reference.shape != (2,)
            or np.any(~np.isfinite(ll)) or np.any(~mask.any(1))
            or np.any(~np.isfinite(coords)) or np.any(~np.isfinite(area)) or np.any(area <= 0)):
        raise ValueError('Complete finite native likelihoods and boundary cells required')
    # Coordinate scaling improves conditioning; it is NOT a 10-bp model width.
    xy = (coords-reference)/10.
    initials = ([np.asarray(initial, float)] if initial is not None else
                [np.array([0., 0., np.log(2.), 0., np.log(2.)]), np.zeros(5)])
    bounds = [(float(xy[:, j].min()), float(xy[:, j].max())) for j in range(2)]
    bounds += [(-6., 5.), (-100., 100.), (-6., 5.)]
    if objective_backend not in ('cpu', 'cuda', 'separable', 'nonparametric'):
        raise ValueError('Native fitting supports CPU, the separable CPU objective, the nonparametric mass, or experimental CUDA float64')
    nonparametric = objective_backend == 'nonparametric'
    if nonparametric:
        # The Gaussian seed is fitted with the separable objective where the size
        # gate admits it; the nonparametric mass is then maximized from that seed.
        objective_backend = 'separable'
    separable = None
    if objective_backend == 'separable':
        from .separable_fit import SeparableObjective
        try:
            separable = SeparableObjective(ll, mask)
        except ValueError:
            # Structure not met for this fit: the dense reference objective is used.
            objective_backend = 'cpu'
    dense = {}
    def dense_args():
        if 'args' not in dense:
            offset = np.where(mask, ll, -np.inf).max(1)
            scaled_likelihood = np.exp(np.where(mask, ll-offset[:, None], -np.inf))
            exposure = mask.astype(float)
            dense['args'] = (xy, np.log(area), ll, mask, scaled_likelihood, offset, exposure, _allowed_classes(mask))
        return dense['args']
    if separable is not None:
        def _dense_fallback(parameters):
            return _cached_objective(parameters, *dense_args())
        def objective(parameters, xy_, log_area_):
            return separable(parameters, xy_, log_area_, _dense_fallback)
        objective_args = (xy, np.log(area))
    else:
        objective_args = dense_args()
        objective = _cached_objective
    if objective_backend == 'cuda':
        xy, _, ll, mask, scaled_likelihood, offset, exposure, classes = objective_args
        from .accelerated_fit import CudaObjective
        accelerated = CudaObjective(xy, np.log(area), scaled_likelihood, offset, exposure,
                                    maximum_bytes=accelerator_bytes)
        def objective(parameters, *args):
            value = accelerated(parameters)
            return _cached_objective(parameters, *args) if value is None else value
    candidates = []
    for start_index,guess in enumerate(initials):
        iteration=0
        def checkpoint(_parameters):
            nonlocal iteration
            iteration+=1
            if progress is not None:
                progress(f'initialization {start_index+1}/{len(initials)}, optimizer iteration {iteration}')
        result = minimize(objective, guess,
            args=objective_args, method='L-BFGS-B', jac=True,
            bounds=bounds, callback=checkpoint if progress is not None else None,
            options=dict(maxiter=max_iterations, ftol=1e-9, gtol=1e-5, maxls=40))
        if np.isfinite(result.fun) and np.all(np.isfinite(result.x)):
            candidates.append(result)
    if not candidates:
        raise FloatingPointError('No finite native boundary-distribution fit')
    result = min(candidates, key=lambda r: r.fun)
    out = finalize_native_distribution(np.asarray(result.x, float), coordinates, areas, reference,
        objective=float(result.fun), converged=bool(result.success), iterations=int(result.nit),
        message=str(result.message), source_units=len(ll))
    if separable is not None:
        out['objective_backend'] = 'separable'
        out['separable_evaluations'] = int(separable.evaluations)
        out['separable_dense_fallbacks'] = int(separable.fallbacks)
    if nonparametric:
        from .nonparametric_fit import fit_nonparametric
        npf = fit_nonparametric(ll, mask, area, out['log_mass'], alpha=float(smoothing_pseudo_units))
        log_area = np.log(area)
        out.update(gaussian_seed=dict(parameters=out['parameters'].tolist(), center=out['center'].tolist(),
                                      covariance=out['covariance'].tolist(), objective=out['objective'],
                                      log_mass=out['log_mass'].tolist()),
                   log_mass=npf['log_mass'], log_density=npf['log_mass']-log_area,
                   objective=npf['objective'], converged=bool(out['converged'] and npf['converged']),
                   nonparametric_iterations=npf['iterations'], nonparametric_converged=npf['converged'],
                   smoothing_pseudo_units=npf['smoothing_pseudo_units'], objective_backend='nonparametric',
                   penalized_objective=npf['penalized_objective'],
                   prior_inadmissible_mass=npf['prior_inadmissible_mass'])
        # centre/covariance from the fitted mass over the cell midpoints; the
        # classification recomputes the full geometry summary on the exact cells.
        w = np.exp(npf['log_mass']-logsumexp(npf['log_mass']))
        mean = w @ coords; resid = coords-mean
        out['center'] = mean; out['covariance'] = (resid*w[:, None]).T @ resid
    return out


def finalize_native_distribution(parameters, coordinates, areas, reference, *, objective, converged,
                                 iterations, message, source_units):
    """Deterministic model outputs from fitted parameters; shared by fresh and cached fits."""
    coords = np.asarray(coordinates, float); area = np.asarray(areas, float)
    reference = np.asarray(reference, float)
    xy = (coords-reference)/10.
    x = np.asarray(parameters, float)
    density, _ = _density(x, xy)
    a, b, c = np.exp(x[2]), x[3], np.exp(x[4])
    precision = np.array([[a*a, a*b], [a*b, b*b+c*c]])
    covariance = np.linalg.inv(precision)*100.
    return dict(parameters=x, log_density=density,
                log_mass=density+np.log(area)-logsumexp(density+np.log(area)),
                center=reference+10*x[:2], covariance=covariance,
                objective=float(objective), converged=bool(converged), iterations=int(iterations),
                message=str(message), source_units=int(source_units))


def _shape_penalty(density, starts, ends, *, relax_left=False, relax_right=False,
                   boundary_cells=None, edge_tolerance_bp=None):
    """One penalty implementation shared by observations and simulations.

    Legacy Boolean profiling remains solely for frozen native-CR compatibility.
    XCR uses the explicitly bounded base-pair mode, without reference triggers.
    """
    if edge_tolerance_bp is not None:
        if relax_left or relax_right:
            raise ValueError('Bounded allowance cannot be combined with legacy edge profiling')
        from .measurement_edge_tolerance import bounded_edge_penalty
        return bounded_edge_penalty(density, boundary_cells, edge_tolerance_bp)
    if boundary_cells is not None:
        raise ValueError('Boundary cells require an explicit bounded bp allowance')
    penalty = density-density.max()
    adjusted = penalty.copy()
    if relax_left:
        by_end = np.full(int(ends.max())+1, -np.inf)
        np.maximum.at(by_end, ends, penalty)
        adjusted = np.maximum(adjusted, by_end[ends])
    if relax_right:
        by_start = np.full(int(ends.max())+1, -np.inf)
        np.maximum.at(by_start, starts, penalty)
        adjusted = np.maximum(adjusted, by_start[starts])
    if relax_left and relax_right:
        adjusted[:] = 0.
    return adjusted


def distribution_comparison(log_density, recipient, starts, ends, *, allowed,
                            relax_left=False, relax_right=False,
                            boundary_cells=None, edge_tolerance_bp=None):
    """Model-shape compatibility, with no class-frequency/normalizer penalty.

    Loss = max L_i - max_g[L_i(g) + log density(g)/max density].
    This is a penalized profile score, NOT the earlier exact-equality LR, a
    Bayes factor, posterior, confidence level or FDR. The density ratio (not
    geometry probability mass) avoids rewarding wide opportunity cells.
    Per-edge floor profiling is exported separately from the native result.
    """
    return _distribution_comparison(log_density, recipient, starts, ends, allowed=allowed,
        relax_left=relax_left, relax_right=relax_right,
        boundary_cells=boundary_cells, edge_tolerance_bp=edge_tolerance_bp)[0]


def _distribution_comparison(log_density, recipient, starts, ends, *, allowed,
                             relax_left=False, relax_right=False,
                             boundary_cells=None, edge_tolerance_bp=None,
                             _prepared_shape_penalty=None):
    """Return the score AND its fixed penalty for simulation-side reuse."""
    density, recipient = np.asarray(log_density, float), np.asarray(recipient, float)
    aa, bb, valid = np.asarray(starts), np.asarray(ends), np.asarray(allowed, bool)
    if not (density.shape == recipient.shape == aa.shape == bb.shape == valid.shape) or not valid.any():
        raise ValueError('One same-grid profile and validity flag per projection required')
    if (np.any(np.isnan(density) | np.isposinf(density)) or not np.isfinite(density).any()
            or np.any(~np.isfinite(recipient))):
        raise ValueError('Valid model density and finite recipient likelihood required')
    if not np.isfinite(density[valid]).any():
        raise ValueError('Recipient has no admissible model geometry')
    penalty = density-density.max()
    best = float(recipient[valid].max())
    native = max(0., best-float((recipient+penalty)[valid].max()))
    adjusted = (_shape_penalty(density, aa, bb, relax_left=relax_left, relax_right=relax_right,
        boundary_cells=boundary_cells, edge_tolerance_bp=edge_tolerance_bp)
        if _prepared_shape_penalty is None else _prepared_shape_penalty)
    effective = max(0., best-float((recipient+adjusted)[valid].max()))
    candidates = np.flatnonzero(valid)
    selected = int(candidates[np.argmax((recipient+adjusted)[valid])])
    return dict(native_loss=native, floor_adjusted_loss=effective,
                selected_projection=selected, recipient_optimum=best,
                selected_native_log_lr=float(recipient[selected]),
                selected_model_shape_penalty=float(-adjusted[selected])), adjusted


@njit(cache=True)
def _monotone_projection_ranges(starts, ends):
    """Recognize, never assume, contiguous monotone ranges of right edges.

    Native half-open interval constraints normally have this structure even
    after quotienting missing opportunities. Arbitrary/holey inputs retain
    the exhaustive maximum. The returned ranges include invisible a == b
    classes; no candidate is removed.
    """
    left = np.empty(len(starts), np.int64)
    lower = np.empty(len(starts), np.int64)
    upper = np.empty(len(starts), np.int64)
    count = 0
    for h in range(len(starts)):
        if count and starts[h] == left[count-1]:
            if ends[h] != upper[count-1]+1:
                return False, left[:0], lower[:0], upper[:0]
            upper[count-1] = ends[h]
        else:
            if count and starts[h] <= left[count-1]:
                return False, left[:0], lower[:0], upper[:0]
            left[count], lower[count], upper[count] = starts[h], ends[h], ends[h]
            count += 1
    for j in range(1, count):
        if lower[j] < lower[j-1] or upper[j] < upper[j-1]:
            return False, left[:0], lower[:0], upper[:0]
    return True, left[:count], lower[:count], upper[:count]


@njit(cache=True, nogil=True)
def _predictive_exceedances(pa, pp, starts, ends, log_penalty, cdf, threshold, replicates, seed):
    """Same random experiments and exact tail event, with bounded profile work.

    The unpenalized maximum uses a sliding range maximum where the complete
    projection universe permits it. A penalized candidate can change the tail
    event only if best - (value + penalty) < threshold - 1e-10. Since value
    cannot exceed best, sorted penalties provide an exact early stopping bound.
    Both tests use the original floating-point operations, not an approximate
    penalty cutoff. Every replicate still consumes every original random draw.
    """
    if (not len(starts) or len(ends) != len(starts) or len(log_penalty) != len(starts)
            or len(cdf) != len(starts) or len(pp) != len(pa)):
        raise ValueError('Nonempty aligned predictive projections and probabilities required')
    if (np.any(np.isnan(log_penalty) | np.isposinf(log_penalty)) or np.isnan(threshold)
            or np.any(starts < 0) or np.any(ends < starts) or np.any(ends > len(pa))):
        raise ValueError('Valid predictive penalties, threshold and interval indices required')
    np.random.seed(seed)
    count = 0
    # Native probabilities are fixed across replicates. Preserve scalar
    # arithmetic and RNG draw order, but compute each log step only once.
    hit_step = np.empty(len(pa)); miss_step = np.empty(len(pa))
    for j in range(len(pa)):
        if not (0 < pa[j] < 1 and 0 < pp[j] < 1):
            raise ValueError('Predictive probabilities must be strictly between zero and one')
        hit_step[j] = np.log(pp[j]/pa[j])
        miss_step[j] = np.log1p(-pp[j])-np.log1p(-pa[j])
        if not np.isfinite(hit_step[j]) or not np.isfinite(miss_step[j]):
            raise ValueError('Finite predictive log-likelihood increments required')
    prefix = np.zeros(len(pa)+1)
    monotone, left, lower, upper = _monotone_projection_ranges(starts, ends)
    queue = np.empty(len(prefix), np.int64)
    order = np.argsort(-log_penalty)
    target = threshold-1e-10
    # For a fixed experiment, a binary observation vector always has the same
    # likelihood/profile loss. Small lattices repeatedly draw the same vector.
    # Cache its EXACT event, not a probability approximation. Still consume
    # every original RNG draw and construct the same ordered prefix sums.
    memo = np.full(1 << len(pa) if len(pa) <= 16 else 0, -1, np.int8)
    for _ in range(replicates):
        g = np.searchsorted(cdf, np.random.random())
        pattern = 0
        for j in range(len(pa)):
            p = pp[j] if starts[g] <= j < ends[g] else pa[j]
            hit = np.random.random() < p
            step = hit_step[j] if hit else miss_step[j]
            prefix[j+1] = prefix[j]+step
            if len(memo) and hit:
                pattern |= 1 << j
        if len(memo) and memo[pattern] >= 0:
            count += memo[pattern]
            continue
        best = -np.inf
        if monotone:
            head = tail = cursor = 0
            for j in range(len(left)):
                while cursor <= upper[j]:
                    while tail > head and prefix[queue[tail-1]] <= prefix[cursor]:
                        tail -= 1
                    queue[tail] = cursor
                    tail += 1
                    cursor += 1
                while head < tail and queue[head] < lower[j]:
                    head += 1
                best = max(best, prefix[queue[head]]-prefix[left[j]])
        else:
            for h in range(len(starts)):
                best = max(best, prefix[ends[h]]-prefix[starts[h]])
        exceeds = True
        for h in order:
            if best-(best+log_penalty[h]) >= target:
                # Subsequent penalties are no larger. No remaining candidate
                # can defeat this event, even at its upper bound value=best.
                break
            value = prefix[ends[h]]-prefix[starts[h]]
            if best-(value+log_penalty[h]) < target:
                exceeds = False
                break
        count += exceeds
        if len(memo):
            memo[pattern] = int(exceeds)
    return count


def predictive_reference(log_density, log_mass, recipient, starts, ends, *, allowed,
                         observed, p_accessible, p_protected, replicates=4095, seed=1,
                         relax_left=False, relax_right=False,
                         boundary_cells=None, edge_tolerance_bp=None, _defer_simulation=False,
                         _prepared_shape_penalty=None):
    """Same-model predictive reference on THIS recipient's native lattice.

    This tests compatibility with a fitted family distribution. It does not
    rerun upstream nomination/HMM selection and therefore is NOT a validated
    false-split rate, an FDR q value, or the probability two families are equal.
    Finite simulation uncertainty and its resolution are exported explicitly.

    Identical recipient projections are exactly quotiented: their generative
    masses add, and their profile penalties take the maximum. No tail, unit or
    projection is sampled away. Missing sites are not simulated as misses.
    """
    if not isinstance(replicates, (int, np.integer)) or replicates < 1:
        raise ValueError('Positive integer native simulation count required')
    d, mass = np.asarray(log_density, float), np.asarray(log_mass, float)
    observed = np.asarray(observed, bool); pa, pp = np.asarray(p_accessible), np.asarray(p_protected)
    aa, bb = np.asarray(starts), np.asarray(ends)
    valid = np.asarray(allowed, bool)
    if pa.shape != observed.shape or pp.shape != pa.shape or np.any((pa[observed] <= 0) | (pa[observed] >= 1) | (pp[observed] <= 0) | (pp[observed] >= 1)):
        raise ValueError('Native probabilities required on every observed site')
    score, penalty = _distribution_comparison(d, recipient, aa, bb, allowed=valid,
        relax_left=relax_left, relax_right=relax_right,
        boundary_cells=boundary_cells, edge_tolerance_bp=edge_tolerance_bp,
        _prepared_shape_penalty=_prepared_shape_penalty)
    if score['floor_adjusted_loss'] <= 1e-10:
        return score | dict(predictive_tail=1., predictive_tail_interval=[1., 1.],
                            simulations=0, tail_exceedances=0, simulation_resolution=0.)
    projection = np.r_[0, np.cumsum(observed)]
    unique, inv = _unique_projection_pairs(projection[aa[valid]], projection[bb[valid]])
    # Invisible geometries can have equal cuts. They remain generative options,
    # NOT fabricated positive evidence for a footprint on this recipient.
    q = np.exp(mass[valid]-logsumexp(mass[valid]))
    q = np.bincount(inv, weights=q, minlength=len(unique))
    penalties = np.full(len(unique), -np.inf); np.maximum.at(penalties, inv, penalty[valid])
    cdf = np.cumsum(q); cdf[-1] = 1.
    request = (pa[observed].astype(float), pp[observed].astype(float),
        unique[:, 0], unique[:, 1], penalties, cdf, score['floor_adjusted_loss'], replicates, seed)
    if _defer_simulation:
        # Private execution carrier, never a persistable scientific record.
        # Complete it before filtering, summarization or serialization.
        return score | dict(_native_predictive_request=request)
    return complete_predictive_reference(score | dict(_native_predictive_request=request))


from contextvars import ContextVar
_predictive_kernel = ContextVar('consensus_native_predictive_kernel', default=('reference', 0.))


def set_default_predictive_kernel(name, tilt=0.):
    _predictive_kernel.set((str(name or 'reference'), float(tilt or 0.)))


def current_predictive_kernel():
    return _predictive_kernel.get()


def complete_predictive_reference(record):
    """Finish exactly one original-seed simulation, removing its work carrier.

    The reference kernel is the exact historical one. The declared 'vectorized'
    kernel runs the same experiment with a counter-based generator (and an
    optional importance tilt); its record says so and carries the weighted
    interval instead of the Wilson one.
    """
    if '_native_predictive_request' not in record:
        return record
    result = dict(record)
    request = result.pop('_native_predictive_request')
    kernel, tilt = _predictive_kernel.get()
    if kernel == 'vectorized':
        from .vectorized_predictive import vectorized_predictive
        tail, lower, upper, replicates, info = vectorized_predictive(*request, tilt=tilt)
        return result | dict(predictive_tail=float(tail), predictive_tail_interval=[float(lower), float(upper)],
            simulations=int(replicates), tail_exceedances=int(info['raw_events']),
            simulation_resolution=1./(replicates+1.), predictive_kernel='vectorized_philox',
            predictive_tilt=str(tilt), weighted_events=float(info['weighted_events']))
    count = int(_predictive_exceedances(*request))
    replicates = request[-2]
    return predictive_count_record(result, count, replicates)


def predictive_count_record(result, count, replicates):
    """Shared output arithmetic for CPU and validated accelerator counts."""
    # Plus-one reference and Wilson Monte Carlo interval; the interval is about
    # Monte Carlo precision only, not scientific-model or selection uncertainty.
    tail = (count+1.)/(replicates+1.)
    z = 1.959963984540054; freq = count/replicates
    den = 1+z*z/replicates
    center = (freq+z*z/(2*replicates))/den
    half = z*np.sqrt(freq*(1-freq)/replicates+z*z/(4*replicates**2))/den
    return result | dict(predictive_tail=float(tail), predictive_tail_interval=[float(max(0., center-half)), float(min(1., center+half))],
        simulations=replicates, tail_exceedances=count, simulation_resolution=1./(replicates+1.))


def _unique_projection_pairs(left, right):
    """Exactly the lexicographic np.unique(axis=0) quotient without record sort.

    This helper handles nonnegative integer opportunity indices only. Guard the
    packed radix product before multiplication; extremely large indices retain
    the structured reference implementation. Ordering, multiplicities, and hence
    generative CDF/RNG interpretation remain identical.
    """
    left=np.asarray(left,dtype=np.int64);right=np.asarray(right,dtype=np.int64)
    if not len(left):
        return np.empty((0,2),dtype=np.int64),np.empty(0,dtype=np.int64)
    radix=int(right.max())+1
    if left.min()<0 or right.min()<0 or radix>np.iinfo(np.int64).max or int(left.max())>(np.iinfo(np.int64).max-(radix-1))//radix:
        return np.unique(np.c_[left,right],axis=0,return_inverse=True)
    keys,inverse=np.unique(left*radix+right,return_inverse=True)
    return np.c_[keys//radix,keys%radix],inverse
