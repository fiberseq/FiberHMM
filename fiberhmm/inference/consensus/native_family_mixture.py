"""Training-only mixtures of two intact, frozen native boundary distributions.

The conditional mixture objective need not be concave: every source call has
its own admissible-geometry denominator. The scalar optimizer uses certified
interval upper bounds, admits zero weights, and reports its remaining global
objective gap. It never uses recipient-assay agreement to choose weights.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import heapq
import math

import numpy as np
from scipy.special import logsumexp

from .artifacts import digest
from .measurement_family import _native_values
from .native_family_consolidation import restore_consolidated_geometry


SCHEMA = 'native_explicit_two_component_parent_v1'


def fit_conditional_two_component_weights(log_numerator, log_denominator, *,
                                          objective_tolerance=1e-6, max_evaluations=20000):
    """Globally bound sum log(w A1+(1-w) A2)-log(w B1+(1-w) B2).

    Each row's ratio of positive affine functions is monotone, so the sum of
    its endpoint maxima bounds the full objective on every interval. A second
    bound uses f'' >= -sum(max_endpoint |A'/A|**2). No claim of concavity is
    needed. Returned bounds include a floating-point rounding allowance, not
    an inference confidence interval. Zero-denominator endpoints are not
    feasible fits, though their one-sided limits bound the open interval.
    """
    a, b = np.asarray(log_numerator, float), np.asarray(log_denominator, float)
    if (a.ndim != 2 or a.shape[1] != 2 or not len(a) or b.shape != a.shape
            or np.any(np.isnan(a) | np.isposinf(a) | np.isnan(b) | np.isposinf(b))
            or np.any(np.isfinite(a) != np.isfinite(b)) or np.any(~np.isfinite(b).any(axis=1))):
        raise ValueError('Two component numerator/denominator columns with matching finite support required')
    if (not math.isfinite(objective_tolerance) or objective_tolerance <= 0
            or isinstance(max_evaluations, bool) or not isinstance(max_evaluations, int) or max_evaluations < 3):
        raise ValueError('Positive objective tolerance and at least three evaluations required')
    conditional = np.zeros_like(a)
    np.subtract(a, b, out=conditional, where=np.isfinite(b))
    endpoint_feasible = [bool(np.isfinite(b[:, 1]).all()), bool(np.isfinite(b[:, 0]).all())]
    scaled_a = np.exp(a - np.max(a, axis=1, keepdims=True))
    delta_a = scaled_a[:, 0] - scaled_a[:, 1]
    evaluations = 0
    best = -np.inf; best_w = None

    def evaluate(w):
        nonlocal evaluations, best, best_w
        evaluations += 1
        if w == 0.:
            values = np.where(np.isfinite(b[:, 1]), conditional[:, 1], conditional[:, 0])
        elif w == 1.:
            values = np.where(np.isfinite(b[:, 0]), conditional[:, 0], conditional[:, 1])
        else:
            values = (np.logaddexp(math.log(w) + a[:, 0], math.log1p(-w) + a[:, 1])
                      - np.logaddexp(math.log(w) + b[:, 0], math.log1p(-w) + b[:, 1]))
        value = float(values.sum())
        feasible = endpoint_feasible[int(w)] if w in (0., 1.) else True
        if feasible and (value > best or (value == best and (best_w is None or w < best_w))):
            best, best_w = value, float(w)
        return values

    def upper(lo, hi, left, right):
        monotone = float(np.maximum(left, right).sum())
        with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
            dl = lo * scaled_a[:, 0] + (1 - lo) * scaled_a[:, 1]
            dr = hi * scaled_a[:, 0] + (1 - hi) * scaled_a[:, 1]
            sl = np.divide(np.abs(delta_a), dl, out=np.full(len(a), np.inf), where=dl > 0)
            sr = np.divide(np.abs(delta_a), dr, out=np.full(len(a), np.inf), where=dr > 0)
            curvature = float(np.square(np.maximum(sl, sr)).sum())
        smooth = max(float(left.sum()), float(right.sum())) + curvature * (hi - lo)**2 / 8.
        bound = min(monotone, smooth)
        rounding = 64 * np.finfo(float).eps * max(1., abs(bound), float(np.abs(left).sum()), float(np.abs(right).sum()))
        return bound + rounding

    points = np.linspace(0., 1., min(65, max_evaluations))
    rows = [evaluate(float(w)) for w in points]
    endpoint_limits = [float(rows[0].sum()), float(rows[-1].sum())]
    heap = []
    for lo, hi, left, right in zip(points[:-1], points[1:], rows[:-1], rows[1:]):
        heapq.heappush(heap, (-upper(lo, hi, left, right), float(lo), float(hi), left, right))
    while heap and -heap[0][0] - best > objective_tolerance and evaluations < max_evaluations:
        _, lo, hi, left, right = heapq.heappop(heap)
        middle = (lo + hi) / 2.
        if middle == lo or middle == hi:
            heapq.heappush(heap, (-upper(lo, hi, left, right), lo, hi, left, right))
            break
        values = evaluate(middle)
        heapq.heappush(heap, (-upper(lo, middle, left, values), lo, middle, left, values))
        heapq.heappush(heap, (-upper(middle, hi, values, right), middle, hi, values, right))
    ceiling = max(best, -heap[0][0]) if heap else best
    gap = max(0., ceiling - best)
    both = np.isfinite(b).all(axis=1)
    indistinguishable = bool(np.all(~both | np.isclose(conditional[:, 0], conditional[:, 1], rtol=0., atol=1e-12)))
    return dict(weights=[best_w, 1. - best_w], log_likelihood=best,
        objective=-best, global_objective_upper_bound=ceiling, global_objective_gap=gap,
        converged=bool(gap <= objective_tolerance), evaluations=evaluations,
        objective_tolerance=float(objective_tolerance), max_evaluations=max_evaluations,
        endpoint_weights_allowed=True, endpoint_feasible=endpoint_feasible,
        endpoint_log_likelihood_limits=endpoint_limits,
        profile_grid=[dict(weight_first=float(w), log_likelihood=float(v.sum()),
                           loss_from_fitted_best=max(0., best-float(v.sum())),
                           feasible=endpoint_feasible[int(w)] if w in (0., 1.) else True)
                      for w, v in zip(points, rows)],
        profile_grid_semantics='descriptive conditional likelihood profile, not a calibrated confidence interval',
        conditionally_indistinguishable_on_training=indistinguishable,
        identifiability='not_identified' if indistinguishable else 'not_certified_by_optimization',
        message='global scalar objective bounded' if gap <= objective_tolerance else 'explicit scalar evaluation/precision budget reached',
        objective_semantics='conditional native likelihood; includes per-call admissibility normalizers; no occupancy or OOF claim')


def native_component_evidence(likelihood, allowed, component_log_mass, *, batch_size=64):
    """Integrate intact component geometries for each source, without underflow."""
    ll, mask, q = np.asarray(likelihood, float), np.asarray(allowed, bool), np.asarray(component_log_mass, float)
    if (ll.ndim != 2 or mask.shape != ll.shape or q.shape != (2, ll.shape[1])
            or np.any(~np.isfinite(ll)) or np.any(np.isnan(q) | np.isposinf(q))
            or not np.allclose(logsumexp(q, axis=1), 0., atol=1e-8, rtol=0.)
            or isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1):
        raise ValueError('Finite native likelihood, allowed mask and two normalized component masses required')
    a = np.empty((len(ll), 2)); b = a.copy()
    for k in range(2):
        for lo in range(0, len(ll), batch_size):
            hi = min(len(ll), lo + batch_size)
            prior = np.where(mask[lo:hi], q[k], -np.inf)
            b[lo:hi, k] = logsumexp(prior, axis=1)
            a[lo:hi, k] = logsumexp(prior + ll[lo:hi], axis=1)
    return a, b


def _profiles(stratum, native_result, parent, grid):
    units = {u['unit_id']: u for u in stratum['units']}
    if len(units) != len(stratum['units']): raise ValueError('Duplicate source unit IDs')
    calls = native_result['calls']; rows = parent['source_calls']
    values, allowed, seen = [], [], set()
    aa, bb, p = grid['starts'], grid['ends'], grid['positions']
    for row in rows:
        c = calls[row['source_call_index']]
        if (c['unit_id'] != row['unit_id'] or c['ordinal'] != row['ordinal']
                or [c['start'], c['end']] != row['interval']):
            raise ValueError('Frozen parent source-call identity mismatch')
        u = units[c['unit_id']]; group = c.get('evidence_group_id', c['unit_id'])
        if (group in seen or group != u.get('fold_group_id', u['unit_id'])
                or list(u['representative_raw_tf_intervals'][c['ordinal']]) != row['interval']):
            raise ValueError('Duplicate source group or changed original source span')
        seen.add(group)
        steps, _ = _native_values(u, p, c); prefix = np.r_[0., np.cumsum(steps)]
        ca, cb = np.searchsorted(p, [c['start'], c['end']])
        valid = (aa < cb) & (bb > ca)
        previous, following = grid['domain']
        for x, y in list(u['representative_raw_tf_intervals']) + list(u.get('raw_nuc_intervals', [])):
            if y <= c['start']: previous = max(previous, y)
            if x >= c['end']: following = min(following, x)
        valid &= (grid['left_hi'] >= previous) & (grid['right_lo'] <= following)
        values.append(prefix[bb] - prefix[aa]); allowed.append(valid)
    return np.asarray(values), np.asarray(allowed)


def _pack_log(values):
    a = np.asarray(values, float); support = np.isfinite(a)
    if np.any(np.isnan(a) | np.isposinf(a)): raise ValueError('Invalid log density')
    return dict(finite_support_mask=support.tolist(), finite_values=a[support].tolist(),
                excluded_cells_mean='exact zero probability, not missing data')


def _unpack_log(value):
    support = np.asarray(value['finite_support_mask'])
    values = np.asarray(value['finite_values'], float)
    if support.ndim != 1 or support.dtype.kind != 'b' or values.shape != (int(support.sum()),) or np.any(~np.isfinite(values)):
        raise ValueError('Invalid explicit finite-density support encoding')
    output = np.full(len(support), -np.inf); output[support] = values
    return output


def fit_explicit_native_family_mixture(stratum, native_result, source_parent, frozen_children, *,
                                     parent_family_id, objective_tolerance=1e-6,
                                     max_evaluations=20000, maximum_matrix_bytes=1024**3):
    """Fit only mixture weights on the verified, once-per-group parent cohort.

    ``source_parent`` is the preceding explicit consolidation artifact; only
    its authenticated cohort/domain/grid are reused, not its fitted Gaussian.
    ``frozen_children`` are exact original source-grid objects. Their complete
    source-cell masses transfer losslessly to one refined union grid.
    """
    from .native_cross import boundary_grid, transfer_density, frozen_tabulated_geometry
    template = restore_consolidated_geometry(source_parent)
    if digest(native_result) != source_parent['provenance']['native_snapshot_digest']:
        raise ValueError('Native snapshot differs from the fixed parent cohort')
    children = sorted(frozen_children, key=lambda f: f['model']['family'])
    ids = [f['model']['family'] for f in children]
    if len(ids) != 2 or ids != source_parent['model']['child_family_ids']:
        raise ValueError('Exactly the same two original child families are required')
    if (not isinstance(parent_family_id, str) or not parent_family_id.startswith(stratum['dataset_id'] + ':')
            or parent_family_id in {m['family'] for m in native_result['family_models']}):
        raise ValueError('A new explicit parent ID in the same dataset is required')
    for child in children:
        if digest(child['model']) != source_parent['provenance']['child_model_digests'][child['model']['family']]:
            raise ValueError('Original child model changed')
    if isinstance(maximum_matrix_bytes, bool) or not isinstance(maximum_matrix_bytes, int) or maximum_matrix_bytes < 1:
        raise ValueError('Positive explicit matrix budget required')
    domain = template['grid']['domain']
    p = np.unique(np.concatenate([template['grid']['positions'], *[f['grid']['positions'] for f in children]]))
    grid = boundary_grid(p, domain, model_domains=[f['grid']['domain'] for f in children])
    estimate = int((len(grid['areas']) + len(template['grid']['areas'])) * len(source_parent['source_calls']) * 18
                   + len(grid['areas']) * (64 * 24 + 160))
    if estimate > maximum_matrix_bytes:
        raise MemoryError(f'Exact component fit needs approximately {estimate} bytes; no sources/geometries removed')
    old_ll, old_allowed = _profiles(stratum, native_result, source_parent, template['grid'])
    if (hashlib.sha256(old_ll.astype('<f8').tobytes()).hexdigest() != source_parent['provenance']['likelihood_profiles_sha256']
            or hashlib.sha256(old_allowed.tobytes()).hexdigest() != source_parent['provenance']['allowed_projections_sha256']):
        raise ValueError('Source native likelihood or neighbor conditioning changed')
    if all(np.array_equal(grid[k], template['grid'][k]) for k in grid):
        ll, allowed = old_ll, old_allowed
    else:
        ll, allowed = _profiles(stratum, native_result, source_parent, grid)
    masses = []; child_records = []
    for child in children:
        original = transfer_density(child, child['grid'])
        z = logsumexp(original + np.log(child['grid']['areas']))
        density = transfer_density(child, grid) - z
        mass = density + np.log(grid['areas'])
        if not np.isclose(logsumexp(mass), 0., atol=1e-8, rtol=0.):
            raise ValueError('Common grid does not losslessly preserve a complete child distribution')
        masses.append(mass)
        child_records.append(dict(family=child['model']['family'], source_model_digest=digest(child['model']),
            original_projection_grid_sha256=child['model']['fold_models']['full']['projection_grid_sha256'],
            common_grid_log_mass=_pack_log(mass)))
    masses = np.asarray(masses)
    a, b = native_component_evidence(ll, allowed, masses)
    fit = fit_conditional_two_component_weights(a, b, objective_tolerance=objective_tolerance, max_evaluations=max_evaluations)
    log_weights = np.full(2, -np.inf)
    positive = np.asarray(fit['weights']) > 0
    log_weights[positive] = np.log(np.asarray(fit['weights'])[positive])
    mixture_mass = logsumexp(masses + log_weights[:, None], axis=0)
    density = mixture_mass - np.log(grid['areas'])
    grid_hash = hashlib.sha256(np.c_[grid['coordinates'], grid['areas']].astype('<f8').tobytes()).hexdigest()
    model = dict(family=parent_family_id, status='fitted', family_model='explicit_frozen_native_mixture',
        source_units=len(ll), source_call_indices=source_parent['model']['source_call_indices'],
        reference_interval=source_parent['model']['reference_interval'], domain=domain,
        child_family_ids=ids, component_weights=dict(zip(ids, fit['weights'])),
        parameters_leave_evidence_group_out=False, proposal_discovery_out_of_fold=False, scoring_folds=0,
        model_role='explicit_multimodal_parent_comparison', explicit_projection_grid_required=True,
        identifiable_projections=len(grid['areas']),
        fold_models={'full': dict(component_family_ids=ids, weights=fit['weights'], projection_grid_sha256=grid_hash)},
        training_evidence_groups=source_parent['model']['training_evidence_groups'],
        fit_diagnostics={'full': dict(fit, source_units=len(ll), iterations=fit['evaluations'])})
    frozen = frozen_tabulated_geometry(model, grid, density)
    model['normalized_geometry'] = frozen['geometry_summary']
    output = dict(schema=SCHEMA, status='complete', model=model, grid=grid,
        geometry_summary=frozen['geometry_summary'], tabulated_source_log_density=_pack_log(density),
        source_log_density_sha256=frozen['source_log_density_sha256'], components=child_records,
        source_calls=source_parent['source_calls'], fit=fit,
        provenance=dict(source_parent_artifact_sha256=source_parent['artifact_sha256'],
            native_snapshot_digest=source_parent['provenance']['native_snapshot_digest'],
            source_selection=source_parent['provenance']['source_selection'],
            source_primary_union_calls=source_parent['provenance']['primary_union_calls'],
            source_groups_fit_once=len(ll), source_counts_by_strand=dict(Counter(r['strand'] for r in source_parent['source_calls'])),
            full_original_child_distributions_retained=True, recipient_assay_used_to_fit_weights=False,
            old_primary_counts_used_as_weights=False, positive_weight_floor=False,
            per_call_allowed_geometry_normalizers_included=True, original_calls_models_scores_unchanged=True,
            parent_per_molecule_assignments_published=False, source_profile_binding_verified=True,
            source_domain=domain, projection_grid_sha256=grid_hash,
            maximum_matrix_bytes=maximum_matrix_bytes, estimated_matrix_bytes=estimate))
    output['artifact_sha256'] = digest(output)
    return output


def restore_native_mixture_geometry(artifact):
    """Restore and cross-check the complete parent and every preserved component.

    Components are diagnostics for the same parent, not extra observations or
    independently fitted parent assignments. Even zero-weight components stay
    available and must contain a valid, normalized native distribution.
    """
    from .native_cross import frozen_tabulated_geometry
    if artifact.get('schema') != SCHEMA or digest({k: v for k, v in artifact.items() if k != 'artifact_sha256'}) != artifact.get('artifact_sha256'):
        raise ValueError('Mixture artifact schema/digest mismatch')
    frozen = frozen_tabulated_geometry(artifact['model'], artifact['grid'], _unpack_log(artifact['tabulated_source_log_density']))
    if frozen['source_log_density_sha256'] != artifact['source_log_density_sha256']:
        raise ValueError('Mixture tabulated density digest mismatch')
    components = artifact.get('components', [])
    ids = [c.get('family') for c in components]
    model = artifact['model']; weights_by_id = model.get('component_weights', {})
    if (len(ids) != 2 or any(not isinstance(f, str) for f in ids) or len(set(ids)) != 2
            or set(ids) != set(weights_by_id) or ids != model.get('child_family_ids')):
        raise ValueError('Exactly the declared two preserved mixture components are required')
    raw_weights = [weights_by_id[f] for f in ids]
    if any(isinstance(w, bool) or not isinstance(w, (int, float, np.integer, np.floating)) for w in raw_weights):
        raise ValueError('Finite nonnegative component weights required')
    weights = np.asarray(raw_weights, float)
    if (np.any(~np.isfinite(weights)) or np.any(weights < 0)
            or not np.isclose(weights.sum(), 1., rtol=0., atol=1e-12)):
        raise ValueError('Finite nonnegative component weights summing to one required')
    for declared in (artifact.get('fit', {}).get('weights'), model.get('fold_models', {}).get('full', {}).get('weights')):
        if declared is None or np.asarray(declared).shape != (2,) or not np.allclose(declared, weights, rtol=0., atol=1e-12):
            raise ValueError('Mixture weight metadata mismatch')
    grid = frozen['grid']; log_area = np.log(grid['areas'])
    masses = []; restored = []
    for component, weight in zip(components, weights):
        mass = _unpack_log(component['common_grid_log_mass'])
        if mass.shape != log_area.shape:
            raise ValueError('Component mass must cover the complete parent grid')
        # Do not invent an original child reference interval from the parent's
        # mean. These stubs identify exact preserved densities, not new fits.
        stub = dict(family=component['family'], model_role='preserved_native_parent_component',
                    parent_family_id=model['family'], source_model_digest=component.get('source_model_digest'),
                    original_projection_grid_sha256=component.get('original_projection_grid_sha256'))
        child = frozen_tabulated_geometry(stub, grid, mass-log_area)
        masses.append(mass)
        restored.append(dict(family=component['family'], weight=float(weight), frozen=child))
    log_weights = np.full(2, -np.inf)
    positive = weights > 0
    log_weights[positive] = np.log(weights[positive])
    combined = logsumexp(np.asarray(masses)+log_weights[:, None], axis=0)
    parent_mass = frozen['source_log_density']+log_area
    support = np.isfinite(parent_mass)
    if (not np.array_equal(np.isfinite(combined), support)
            or not np.allclose(combined[support], parent_mass[support], rtol=0., atol=1e-9)):
        raise ValueError('Weighted preserved component mass does not match stored parent density')
    frozen['mixture_components'] = restored
    return frozen
