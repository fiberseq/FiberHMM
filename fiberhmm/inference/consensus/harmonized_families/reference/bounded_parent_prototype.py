# Extracted reference kernels; see SOURCE_MANIFEST.json.
import hashlib
from collections import Counter
import numpy as np
from scipy.ndimage import maximum_filter
from fiberhmm.inference.consensus.measurement_family import _native_values
from fiberhmm.inference.consensus.measurement_distribution import fit_native_distribution, predictive_reference
from fiberhmm.inference.consensus.measurement_geometry import summarize_boundary_cells

def prepare_events(units, calls, region, maximum_bytes=1024 * 1024 ** 2):
    if not calls:
        raise ValueError('Existing called events required')
    groups = [c.get('evidence_group_id', c['unit_id']) for c in calls]
    if len(groups) != len(set(groups)):
        raise ValueError('Resolve duplicate physical groups explicitly; none silently removed')
    (lo, hi) = map(int, region)
    bounds = np.arange(lo, hi + 1)
    size = len(bounds)
    estimate = len(calls) * size * size * 9 + size * size * 8 * 30
    if estimate > maximum_bytes:
        raise MemoryError(f'Exact event grid estimate {estimate}; no event/grid thinning')
    (li, ri) = np.triu_indices(size, 1)
    left = bounds[li]
    right = bounds[ri]
    positions = np.unique(np.concatenate([np.asarray(units[c['unit_id']]['positions'], int) for c in calls]))
    positions = positions[(positions >= lo) & (positions < hi)]
    if not len(positions):
        raise ValueError('No native opportunities; parent is unassessed')
    cuts = np.searchsorted(positions, bounds)
    profiles = []
    allowed = []
    observations = []
    for c in calls:
        u = units[c['unit_id']]
        (values, observed) = _native_values(u, positions, c)
        (before, after) = (lo, hi)
        for (a, b) in list(u['representative_raw_tf_intervals']) + list(u.get('raw_nuc_intervals', [])):
            if b <= c['start']:
                before = max(before, b)
            if a >= c['end']:
                after = min(after, a)
        valid = (left < c['end']) & (right > c['start']) & (left >= before) & (right <= after)
        if not valid.any():
            raise ValueError('Event has no admissible geometry; do not silently drop it')
        prefix = np.r_[0.0, np.cumsum(values)]
        profiles.append(prefix[cuts[ri]] - prefix[cuts[li]])
        allowed.append(valid)
        idx = np.searchsorted(np.asarray(u['positions']), positions[observed])
        observations.append(dict(observed=observed, values=values, p_accessible=np.asarray(u['p_accessible'])[idx], p_protected=np.asarray(u['p_protected'])[idx], neighbor_limits=[before, after]))
    return dict(calls=calls, groups=groups, region=[lo, hi], bounds=bounds, size=size, li=li, ri=ri, left=left, right=right, positions=positions, starts=cuts[li], ends=cuts[ri], profiles=np.asarray(profiles), allowed=np.asarray(allowed), observations=observations, estimated_bytes=estimate)

def center_profiles(data, radius):
    """Existing exact 2D bounded-profile operation, with EVENT masks and no empty."""
    if not isinstance(radius, int) or radius < 0:
        raise ValueError('Nonnegative integer physical radius required')
    rows = []
    size = data['size']
    bounds = data['bounds']
    for (c, values, valid) in zip(data['calls'], data['profiles'], data['allowed']):
        matrix = np.full((size, size), -np.inf)
        matrix[data['li'], data['ri']] = np.where(valid, values, -np.inf)
        pooled = maximum_filter(matrix, size=2 * radius + 1, mode='constant', cval=-np.inf)
        good = (bounds[:, None] < bounds[None, :]) & (bounds[:, None] < c['end']) & (bounds[None, :] > c['start'])
        pooled[~good] = -np.inf
        rows.append(pooled[data['li'], data['ri']])
    return np.asarray(rows)

def fit_parent(data, rows, radius, pooled, center_strategy='source_median', center_bounds=None):
    if center_bounds is not None and center_strategy != 'source_median':
        raise ValueError('Conditional center bounds currently require source_median strategy')
    profiles = data['profiles']
    allowed = data['allowed']
    training = np.asarray(rows, int)
    reference = np.median([[data['calls'][i]['start'], data['calls'][i]['end']] for i in training], axis=0)
    if center_strategy == 'common_profile':
        common = np.isfinite(pooled[training]).all(0)
        if not common.any():
            return dict(status='no_admissible_common_center')
        total = np.where(np.isfinite(pooled[training]), pooled[training], 0).sum(0)
        best = total[common].max()
        ties = np.flatnonzero(common & (total >= best - 1e-09))
        k = min(ties, key=lambda j: ((data['left'][j] - reference[0]) ** 2 + (data['right'][j] - reference[1]) ** 2, int(j)))
        center = np.array([data['left'][k], data['right'][k]])
    elif center_strategy == 'source_median':
        center = np.array([int(np.floor(reference[0])), int(np.ceil(reference[1]))])
        best = None
    else:
        raise ValueError('Unknown experimental center strategy')
    if center_bounds is not None:
        box = np.asarray(center_bounds)
        if box.shape != (2, 2) or np.any(box != np.floor(box)) or np.any(box[:, 0] > box[:, 1]):
            raise ValueError('Two closed integer center bounds required')
        good = (data['left'] >= box[0, 0]) & (data['left'] <= box[0, 1]) & (data['right'] >= box[1, 0]) & (data['right'] <= box[1, 1])
        candidates = np.flatnonzero(good)
        if not len(candidates):
            return dict(status='no_positive_common_nomination_center')
        k = min(candidates, key=lambda j: ((data['left'][j] - reference[0]) ** 2 + (data['right'][j] - reference[1]) ** 2, int(j)))
        center = np.array([data['left'][k], data['right'][k]])
    support = (np.abs(data['left'] - center[0]) <= radius) & (np.abs(data['right'] - center[1]) <= radius)
    columns = np.flatnonzero(support)
    cells = bounded_cells(data, columns, training)
    representatives = np.array([cell['columns'][0] for cell in cells])
    areas = np.array([len(cell['columns']) for cell in cells], float)
    coords = np.array([[(c['left'][0] + c['left'][1]) / 2, (c['right'][0] + c['right'][1]) / 2] for c in cells])
    ll = profiles[np.ix_(training, representatives)]
    mask = allowed[np.ix_(training, representatives)]
    keep = mask.any(1) & np.array([center[0] < data['calls'][i]['end'] and center[1] > data['calls'][i]['start'] for i in training])
    excluded = [data['groups'][i] for i in training[~keep]]
    training = training[keep]
    ll = ll[keep]
    mask = mask[keep]
    if not len(training):
        return dict(status='no_admissible_training_events', excluded_groups=excluded)
    fit = fit_native_distribution(ll, mask, coords, areas, reference=center, max_iterations=100)
    if not fit['converged']:
        retry = fit_native_distribution(ll, mask, coords, areas, reference=center, max_iterations=500)
        if retry['objective'] <= fit['objective'] + 1e-07:
            fit = retry
    boxes = np.array([[*c['left'], *c['right']] for c in cells])
    geometry = summarize_boundary_cells(fit['log_mass'], boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3])
    density = np.full(len(data['left']), -np.inf)
    mass = density.copy()
    for (j, c) in enumerate(cells):
        density[c['columns']] = fit['log_density'][j]
        mass[c['columns']] = fit['log_mass'][j] - np.log(areas[j])
    return dict(status='fitted', anchor=center.tolist(), physical_radius=radius, columns=columns, log_density=density[columns], log_mass=mass[columns], geometry=geometry, boundary_cells=boxes, cell_log_mass=fit['log_mass'], within_cell_uniform=True, diagnostics={k: fit[k] for k in ['converged', 'iterations', 'objective', 'message', 'source_units']}, training_groups=[data['groups'][i] for i in training], excluded_groups=excluded, parent_mass_not_overlapping_anchor=float(np.exp(mass[(data['left'] >= center[1]) | (data['right'] <= center[0])]).sum()), exclusion_reason='no admissible geometry in parent box or no actual overlap with parent anchor', center_strategy=center_strategy, profile_objective=None if best is None else float(best), **dict(conditional_nomination_center_bounds=center_bounds) if center_bounds is not None else {})

def bounded_cells(data, columns, training=None):
    """Exact rectangles refined at native projections and fixed event barriers.

    All integer geometries remain represented. A density is constant inside each
    indistinguishable rectangle, preventing fictitious single-bp edge certainty.
    Empty-probe positive geometries are included as row rectangles, never misses.
    """
    training = np.arange(len(data['calls'])) if training is None else np.asarray(training, int)
    seen = np.any([data['observations'][i]['observed'] for i in training], axis=0)
    left_cuts = list(data['positions'][seen] + 1)
    right_cuts = list(data['positions'][seen] + 1)
    for i in training:
        (call, obs) = (data['calls'][i], data['observations'][i])
        left_cuts.extend([call['end'], obs['neighbor_limits'][0]])
        right_cuts.extend([call['start'] + 1, obs['neighbor_limits'][1] + 1])
    bins = np.c_[np.searchsorted(np.unique(left_cuts), data['left'][columns], side='right'), np.searchsorted(np.unique(right_cuts), data['right'][columns], side='right')]
    (_, inverse) = np.unique(bins, axis=0, return_inverse=True)
    inverse = inverse.reshape(-1)
    cells = []
    for key in range(inverse.max() + 1):
        cc = columns[inverse == key]
        ll = data['left'][cc]
        rr = data['right'][cc]
        groups = [cc] if ll.max() < rr.min() else [cc[ll == l] for l in np.unique(ll)]
        for g in groups:
            (a, b) = (int(data['left'][g].min()), int(data['left'][g].max()))
            (c, d) = (int(data['right'][g].min()), int(data['right'][g].max()))
            if not (b < c and len(g) == (b - a + 1) * (d - c + 1)):
                raise ValueError('Inexact boundary rectangle')
            if not np.all(data['allowed'][np.ix_(training, g)] == data['allowed'][np.ix_(training, g[:1])]):
                raise ValueError('Training eligibility changes within cell')
            if not np.all(data['profiles'][np.ix_(training, g)] == data['profiles'][np.ix_(training, g[:1])]):
                raise ValueError('Training likelihood changes within cell')
            cells.append(dict(columns=g, left=[a, b], right=[c, d]))
    if sum((len(c['columns']) for c in cells)) != len(columns):
        raise ValueError('Lost integer geometries')
    return cells

def score_event(data, i, fit, replicates, seed):
    if fit['status'] != 'fitted':
        return dict(status='unassessed_parent_fit', compatible=None)
    call = data['calls'][i]
    if not (call['start'] < fit['anchor'][1] and call['end'] > fit['anchor'][0]):
        return dict(status='outside_actual_parent_overlap', compatible=None)
    size = len(data['left'])
    d = np.full(size, -np.inf)
    mass = d.copy()
    d[fit['columns']] = fit['log_density']
    mass[fit['columns']] = fit['log_mass']
    valid = data['allowed'][i]
    if not np.isfinite(mass[valid]).any():
        return dict(status='no_admissible_parent_geometry', compatible=None)
    obs = data['observations'][i]
    pa = np.full(len(data['positions']), 0.5)
    pp = pa.copy()
    pa[obs['observed']] = obs['p_accessible']
    pp[obs['observed']] = obs['p_protected']
    result = predictive_reference(d, mass, data['profiles'][i], data['starts'], data['ends'], allowed=valid, observed=obs['observed'], p_accessible=pa, p_protected=pp, replicates=replicates, seed=seed)
    columns = fit['columns']
    weights = np.exp(fit['log_mass'])
    if not np.isclose(weights.sum(), 1.0):
        raise ValueError('Parent mass not normalized')
    delta = np.bincount(data['starts'][columns], weights=weights, minlength=len(data['positions']) + 1) - np.bincount(data['ends'][columns], weights=weights, minlength=len(data['positions']) + 1)
    coverage = np.clip(np.cumsum(delta)[:-1], 0, 1)
    call = data['calls'][i]
    core = obs['observed'] & (coverage >= 0.5) & (data['positions'] >= call['start']) & (data['positions'] < call['end'])
    core_lr = float(obs['values'][core].sum())
    bad = core.any() and core_lr < -np.log(100.0)
    result.update(status='core_contradicted' if bad else 'scored', compatible=bool(not bad and result['predictive_tail_interval'][1] >= 0.001), core_opportunities=int(core.sum()), core_protected_vs_accessible_log_lr=core_lr, additional_matching_floor_bp=0, calibrated_false_split_rate_claim=False)
    return result

def evaluate_parent(data, *, radius, folds=10, replicates=4095, seed=7123, center_strategy='source_median', center_bounds=None):
    n = len(data['calls'])
    if n < 2:
        raise ValueError('At least two physical groups needed for excluded-group scoring')
    ordered = sorted(range(n), key=lambda i: hashlib.sha256(('bounded-parent-fold-v1|' + data['groups'][i]).encode()).digest())
    fold = np.empty(n, int)
    for (j, i) in enumerate(ordered):
        fold[i] = j % min(folds, n)
    pooled = center_profiles(data, radius) if center_strategy == 'common_profile' else None
    full = fit_parent(data, list(range(n)), radius, pooled, center_strategy, center_bounds)
    fits = {str(f): fit_parent(data, np.flatnonzero(fold != f), radius, pooled, center_strategy, center_bounds) for f in sorted(set(fold))}
    records = []
    for (i, c) in enumerate(data['calls']):
        model = fits[str(fold[i])]
        if data['groups'][i] in model.get('training_groups', []):
            raise ValueError('Held-out physical group leaked into fit')
        s = int.from_bytes(hashlib.sha256(f"{seed}|{data['groups'][i]}".encode()).digest()[:4], 'little')
        result = score_event(data, i, model, replicates, s)
        (before, after) = data['observations'][i]['neighbor_limits']
        records.append(dict(call=c, evidence_group=data['groups'][i], fold=int(fold[i]), neighbor_limits=[before, after], left_limit_is_region_boundary=before == data['region'][0], right_limit_is_region_boundary=after == data['region'][1], called_interval_crosses_region=c['start'] < data['region'][0] or c['end'] > data['region'][1], fit_warning=model.get('diagnostics', {}).get('converged') is False, **result))
    return dict(schema='experimental_bounded_event_parent_v1', full_model=full, fold_models=fits, records=records, radius=radius, center_strategy=center_strategy, scoring_reference_percent=99.9, replicates=replicates, compatible=sum((r.get('compatible') is True for r in records)), events=n, rejected=sum((r.get('compatible') is False for r in records)), unassessed=sum((r.get('compatible') is None for r in records)), statuses=dict(Counter((r['status'] for r in records))), actual_simulations=sum((r.get('simulations', 0) for r in records)), gaussian_potential_fitted_only_within_physical_box=True, all_integer_boundaries_retained=True, no_extra_matching_floor=True, no_empty_event_escape=True, no_automatic_merge_decision=True, original_call_records_unchanged=True, calibration_after_llr_selection_not_established=True)
