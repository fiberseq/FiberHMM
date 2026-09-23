"""Reciprocal frozen native-family transfer; no merged cross-assay interval.

The source density is piecewise constant on its FITTED boundary cells. Transfer
refines those cells with recipient opportunities; it does not evaluate a new
Gaussian at a recipient-cell midpoint. This preserves source probability mass
under assay/lattice changes. Both models see the same recipient observations,
conditioned neighbours and geometric search universe.

These conditional predictive diagnostics classify EXISTING calls. They are not
new protection calls, a calibrated equivalence test/FDR, or an occupancy model.
All statuses remain provisional; count agreement never chooses a relationship.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import math

import numpy as np
from scipy.special import logsumexp

from .measurement_distribution import _density, predictive_reference
from .measurement_family import _native_values, _native_observation_arrays
from .measurement_geometry import summarize_grid_geometry, projection_coverage_probability


def boundary_grid(positions, domain, *, model_domains=()):
    """Exactly the integer-coordinate cells used by native-family fitting."""
    lo, hi = map(int, domain)
    # Artificial CUTS, not observed opportunities: split terminal source cells
    # exactly at each donor-domain limit. They carry no observation in scoring.
    cuts=[v for left,right in model_domains for v in (int(left)-1,int(right))]
    p = np.unique(np.r_[np.asarray(positions, np.int64),np.asarray(cuts,np.int64)])
    p = p[(p >= lo) & (p < hi)]
    if not len(p):
        raise ValueError('A boundary grid needs observed opportunities')
    a, b = np.triu_indices(len(p)+1, 1)
    ll, lh = np.r_[lo, p+1][a], p[a]
    rl, rh = p[b-1]+1, np.r_[p, hi][b]
    return dict(positions=p, starts=a, ends=b, left_hi=lh, right_lo=rl,
                coordinates=np.c_[(ll+lh)/2., (rl+rh)/2.],
                areas=(lh-ll+1.)*(rh-rl+1.), domain=[lo, hi])


def frozen_model_geometry(model, calls, units):
    """Rebuild and verify the source model's exact training-cohort grid."""
    lo, hi = model['domain']; left, right = model['reference_interval']
    recipients = [c for c in calls if c['start'] < right and c['end'] > left
                  and c['start'] < hi and c['end'] > lo]
    p = np.unique(np.concatenate([np.asarray(units[c['unit_index']]['positions'], np.int64)
                                  for c in recipients]))
    grid = boundary_grid(p, (lo, hi))
    digest = hashlib.sha256(np.c_[grid['coordinates'], grid['areas']].astype('<f8').tobytes()).hexdigest()
    expected = model['fold_models']['full']['projection_grid_sha256']
    if digest != expected:
        raise ValueError(f"{model['family']}: reconstructed native geometry does not match fitted grid")
    frozen = dict(model=model, grid=grid)
    full = model['fold_models']['full']
    if full.get('objective_backend') == 'nonparametric':
        density = np.asarray(full['tabulated_log_density'], float)
        frozen['source_log_density'] = density
        frozen['source_log_density_sha256'] = hashlib.sha256(density.astype('<f8').tobytes()).hexdigest()
        frozen['source_geometry_form'] = 'nonparametric_tabulated'
    frozen['geometry_summary'] = summarize_grid_geometry(
        grid, transfer_density(frozen, grid)+np.log(grid['areas']))
    return frozen


def transfer_density(frozen, grid, fold='full'):
    """Exact refinement of a fitted source-cell density, including zero mass."""
    model, source = frozen['model'], frozen['grid']
    if 'source_log_density' in frozen:
        if str(fold) != 'full':
            raise ValueError('Tabulated parent has no excluded-fold density')
        source_density = np.asarray(frozen['source_log_density'], float)
        expected_shape = (len(source['positions'])*(len(source['positions'])+1)//2,)
        if (source_density.shape != expected_shape or np.any(np.isnan(source_density) | np.isposinf(source_density))
                or not np.isfinite(source_density).any()):
            raise ValueError('Complete valid tabulated source-cell density required')
        expected = frozen.get('source_log_density_sha256')
        actual = hashlib.sha256(source_density.astype('<f8').tobytes()).hexdigest()
        if expected != actual:
            raise ValueError('Tabulated source density integrity mismatch')
    else:
        fit = model['fold_models'][str(fold)]
    coords, p = grid['coordinates'], source['positions']
    a = np.searchsorted(p, coords[:, 0]); b = np.searchsorted(p, coords[:, 1])
    valid = ((coords[:, 0] >= source['domain'][0]) & (coords[:, 1] <= source['domain'][1])
             & (a < b) & (a < len(p)) & (b > 0))
    result = np.full(len(coords), -np.inf)
    aa, bb = a[valid], b[valid]
    if 'source_log_density' in frozen:
        # Exact triangular row index in the original source grid. Never fit a
        # Gaussian to, or interpolate across, the retained component mixture.
        index = aa*(2*len(p)-aa+1)//2 + bb-aa-1
        result[valid] = source_density[index]
        return result
    ll = np.r_[source['domain'][0], p+1][aa]; lh = p[aa]
    rl = p[bb-1]+1; rh = np.r_[p, source['domain'][1]][bb]
    xy = (np.c_[(ll+lh)/2., (rl+rh)/2.] - np.asarray(fit['parameter_reference'])) / fit['parameter_coordinate_scale_bp']
    result[valid] = _density(np.asarray(fit['parameters']), xy)[0]
    return result


def frozen_tabulated_geometry(model, grid, source_log_density):
    """Validate a normalized, complete native-cell density for exact transfer.

    Used by explicit multimodal parent families. Component learning and its
    provenance belong to the constructor; this adapter validates the numerical
    grid, mass and immutable density vector. It does not average old scores.
    """
    canonical = boundary_grid(grid['positions'], grid['domain'])
    if any(not np.array_equal(grid.get(k), canonical[k]) for k in canonical):
        raise ValueError('Tabulated source must use the exact canonical native grid')
    d = np.asarray(source_log_density, float).copy()
    if (d.shape != canonical['areas'].shape or np.any(np.isnan(d) | np.isposinf(d))
            or not np.isfinite(d).any()):
        raise ValueError('Complete finite-or-zero-mass source density required')
    if not np.isclose(logsumexp(d+np.log(canonical['areas'])), 0., rtol=0., atol=1e-8):
        raise ValueError('Tabulated native source mass must already be normalized')
    d.setflags(write=False)
    return dict(model=model, grid=canonical, source_log_density=d,
        source_log_density_sha256=hashlib.sha256(d.astype('<f8').tobytes()).hexdigest(),
        geometry_summary=summarize_grid_geometry(canonical, d+np.log(canonical['areas'])))


def model_overlap(left, right):
    """Retain all overlapping model domains, not a hard credible-box match.

    Source geometry mass, observed-call attribution and native information are
    assessed downstream and remain in the ledger even when they fail. A fixed
    99.9% envelope is a display summary, not a hypothesis-nomination cutoff.
    """
    a, b = left['grid']['domain'], right['grid']['domain']
    return a[0] < b[1] and b[0] < a[1]


def _recipient_grid_positions(unit, call, grid, native=None):
    """Exact half-open raw-call/neighbor predicates, with no new observations.

    These boundaries are geometry cuts only. Refinement inherits the source
    density, preserving its full integer-coordinate probability mass.
    """
    cuts = [call['start'], call['end']-1]
    for x, y in list(unit['representative_raw_tf_intervals'])+list(unit.get('raw_nuc_intervals', [])):
        if y <= call['start']: cuts.append(y-1)
        if x >= call['end']: cuts.append(x)
    p = np.asarray(unit['positions'], np.int64) if native is None else native[0]
    lo, hi = grid['domain']
    # Positions outside this fixed domain were discarded by boundary_grid
    # anyway. Do not sort/copy an entire long Fiber-seq read for each test.
    positions = np.union1d(grid['positions'], p[(p >= lo) & (p < hi)])
    positions = np.union1d(positions, cuts)
    return positions[(positions >= lo) & (positions < hi)]


def refine_recipient_call_grid(unit, call, grid):
    return boundary_grid(_recipient_grid_positions(unit, call, grid), grid['domain'])


def _transfer_geometry(frozen, grid, unit, call, floor_bp, native, cache):
    if cache is not None:
        cache.validate(frozen, grid, floor_bp)
    positions = _recipient_grid_positions(unit, call, grid, native)
    key = positions.astype('<i8', copy=False).tobytes()
    prepared = cache.get(('grid', key)) if cache is not None else None
    if prepared is not None:
        return prepared
    refined = boundary_grid(positions, grid['domain'])
    d = transfer_density(frozen, refined)
    lm = d+np.log(refined['areas']); lm -= logsumexp(lm)
    prepared = dict(grid=refined, density=d, log_mass=lm, cache_key=key)
    if cache is not None:
        # Reserve space NOW for lazily populated penalty, boundary cells and
        # coverage. Admission never depends on mutable-entry undercounting.
        size = (sum(a.nbytes for a in refined.values() if isinstance(a, np.ndarray))
                + 8*d.nbytes + refined['positions'].nbytes + len(key) + 2048)
        cache.put(('grid', key), prepared, size)
    return prepared


def _reference_members(result, reference_percent, membership_loss_odds=1.):
    """Re-filter frozen evidence; keep immutable original call identities.

    With tie-set membership a call joins every compatible family within the
    declared margin, so a family that is nobody's nearest centre can still have
    a cohort and therefore be testable at all. Nothing about the call, its
    interval or its evidence changes.
    """
    from .native_presentation import member_families
    out = defaultdict(list)
    cut = 1.-reference_percent/100.
    for i, (call, scores) in enumerate(zip(result['calls'], result['call_family_evidence'])):
        accepted = [s for s in scores if s['status'] == 'scored'
                    and s.get('predictive_tail_interval', [0., 0.])[1] >= cut]
        if not accepted:
            continue
        accepted.sort(key=lambda s: (s['geometry_distance_sq'], s['floor_adjusted_loss'], s['family']))
        for fid in member_families(accepted, membership_loss_odds):
            out[fid].append(i)
    return out


def _recipient_constraints(unit, call, grid, require_call_overlap, limits=None):
    a = grid['starts']
    valid = ((grid['left_hi'] < call['end']) & (grid['right_lo'] > call['start'])
             if require_call_overlap else np.ones(len(a),bool))
    if limits is None:
        previous, following = grid['domain']
        for x, y in list(unit['representative_raw_tf_intervals'])+list(unit.get('raw_nuc_intervals', [])):
            if y <= call['start']: previous = max(previous, y)
            if x >= call['end']: following = min(following, x)
    else:
        previous, following = limits
    neighbors = (grid['left_hi'] >= previous) & (grid['right_lo'] <= following)
    valid &= neighbors
    return valid, neighbors


def _geometry_admission(prepared, unit, call, require_overlap, read_cache, geometry_cache):
    grid, lm = prepared['grid'], prepared['log_mass']
    limits = read_cache.limits(unit, call, grid['domain']) if read_cache is not None else None
    key = None
    if limits is not None and geometry_cache is not None:
        key = ('admission', prepared['cache_key'], call['start'], call['end'], *limits, require_overlap)
        cached = geometry_cache.get(key)
        if cached is not None:
            return cached
    allowed, neighbors = _recipient_constraints(unit, call, grid, require_overlap, limits)
    overlap = (grid['left_hi'] < call['end']) & (grid['right_lo'] > call['start'])
    value = (allowed, neighbors,
             float(np.exp(logsumexp(lm[allowed]))) if allowed.any() else 0.,
             float(np.exp(logsumexp(lm[overlap]))) if overlap.any() else 0.,
             float(np.exp(logsumexp(lm[neighbors]))) if neighbors.any() else 0.)
    if key is not None:
        geometry_cache.put(key, value, allowed.nbytes+neighbors.nbytes+len(prepared['cache_key'])+1024)
    return value


def _recipient_observations(unit, call, grid, *, require_call_overlap=True,
                            prepared=None, constraints=None):
    p, a, b = grid['positions'], grid['starts'], grid['ends']
    native = _native_observation_arrays(unit) if prepared is None else prepared
    values, observed = _native_values(unit, p, call, prepared=native)
    prefix = np.r_[0., np.cumsum(values)]
    valid, neighbors = (_recipient_constraints(unit, call, grid, require_call_overlap)
                        if constraints is None else constraints)
    indices = np.searchsorted(native[0], p[observed])
    pa = np.zeros(len(p)); pp = np.zeros(len(p))
    pa[observed] = native[3][indices]
    pp[observed] = native[4][indices]
    return dict(likelihood=prefix[b]-prefix[a], observed=observed, allowed=valid,
                p_accessible=pa, p_protected=pp, values=values, neighbor_allowed=neighbors)


def transferred_call(frozen, grid, unit, call, *, floor_bp=0, replicates=4095,
                     core_contradiction_odds=100., minimum_retained_mass=.05,
                     require_call_overlap=True,minimum_visible_geometry_mass=0.,
                     minimum_call_attribution_mass=None, _defer_simulation=False,
                     _read_cache=None, _geometry_cache=None):
    """Score a foreign model on an existing recipient call, preserving its span."""
    if (isinstance(floor_bp, (bool, np.bool_))
            or not isinstance(floor_bp, (int, np.integer)) or floor_bp < 0):
        raise ValueError('A nonnegative integer bp allowance is required')
    attribution_minimum = (minimum_retained_mass if minimum_call_attribution_mass is None
                           else minimum_call_attribution_mass)
    native = (_native_observation_arrays(unit) if _read_cache is None else _read_cache.native(unit))
    prepared_geometry = _transfer_geometry(frozen, grid, unit, call, floor_bp, native, _geometry_cache)
    grid, d, log_mass = (prepared_geometry[k] for k in ('grid', 'density', 'log_mass'))
    allowed, neighbors, kept, overlap_mass, neighbor_mass = _geometry_admission(
        prepared_geometry, unit, call, require_call_overlap, _read_cache, _geometry_cache)
    if _read_cache is not None:
        observed_count = _read_cache.visible_count(unit, call, grid['positions'][0], grid['positions'][-1])
    else:
        p, _, occupied, _, _ = native
        visible = ((p >= grid['positions'][0]) & (p <= grid['positions'][-1])
                   & ~(occupied & ~((p >= call['start']) & (p < call['end']))))
        observed_count = int(visible.sum())
    record = dict(unit_id=call['unit_id'], source_ordinal=call['ordinal'],
                  evidence_group_id=call.get('evidence_group_id',call['unit_id']),
                  interval=[call['start'], call['end']], strand=call['strand'],
                  physical_geometry_retention=kept, original_span_unchanged=True,
                  observed_opportunities=observed_count,minimum_pair_edge_tolerance_bp=floor_bp,
                  raw_call_overlap_required=require_call_overlap,
                  geometry_mass_overlapping_observed_call=overlap_mass,
                  geometry_mass_allowed_by_frozen_neighbors=neighbor_mass,
                  raw_call_attribution_uses_full_geometry_not_majority_core=True,
                  exact_raw_call_neighbor_cell_refinement=True,
                  edge_tolerance_semantics='bounded_joint_endpoint_cell_profile_v1',
                  edge_tolerance_changes_generative_mass=False,
                  recipient_raw_span_within_comparison_domain=(
                      grid['domain'][0] <= call['start'] and call['end'] <= grid['domain'][1]),
                  source_domain_within_comparison_domain=(
                      grid['domain'][0] <= frozen['grid']['domain'][0]
                      and frozen['grid']['domain'][1] <= grid['domain'][1]))
    if 'alignment_orientation' in call:record['alignment_orientation']=call['alignment_orientation']
    if not observed_count:
        return record | dict(status='no_recipient_information')
    if require_call_overlap and overlap_mass < attribution_minimum:
        return record | dict(status='hypothesis_not_attributable_to_this_call')
    if kept < minimum_retained_mass or not np.isfinite(d[allowed]).any():
        return record | dict(status='geometry_blocked_by_neighbours_or_domain')
    # The early gates and all their diagnostics are identical. Only passing
    # recipients need a likelihood over every geometry and padded probabilities.
    obs = _recipient_observations(unit, call, grid, require_call_overlap=require_call_overlap,
                                 prepared=native, constraints=(allowed, neighbors))
    observed = obs['observed']
    # Existence testability is over the SAME distributed geometries, not an
    # intersection/median core or information from accessible flanks alone.
    aa,bb=grid['starts'],grid['ends'];op=np.r_[0,np.cumsum(observed)]
    # Normalize in log space. ``np.where`` evaluates both branches, so the
    # previous exp(log_mass) / kept expression could overflow even where a
    # geometry was disallowed.  For allowed cells the normalized value is at
    # most one; subtracting log(kept) before exponentiation preserves that
    # invariant and avoids a noisy RuntimeWarning during otherwise valid runs.
    conditional=np.zeros(len(log_mass),dtype=float)
    conditional[allowed]=np.exp(log_mass[allowed]-math.log(kept))
    observed_mass=float(conditional @ ((op[bb]-op[aa])>0))
    pa,pp=obs['p_accessible'][observed],obs['p_protected'][observed]
    information=np.zeros(len(observed));best_steps=information.copy()
    information[observed]=pp*np.log(pp/pa)+(1-pp)*(np.log1p(-pp)-np.log1p(-pa))
    best_steps[observed]=np.maximum(np.log(pp/pa),np.log1p(-pp)-np.log1p(-pa))
    ip=np.r_[0.,np.cumsum(information)];mp=np.r_[0.,np.cumsum(best_steps)]
    record.update(geometry_observed_mass_fraction=observed_mass,
        expected_core_opportunities=float(conditional @ (op[bb]-op[aa])),
        expected_information_upper_bound_nats=float(conditional @ (ip[bb]-ip[aa])),
        maximum_attainable_model_log_lr=float(logsumexp((log_mass+mp[bb]-mp[aa])[allowed])-math.log(kept)),
        raw_call_overlap_required=require_call_overlap)
    if observed_mass < minimum_visible_geometry_mass:
        return record | dict(status='transferred_shape_unobserved')
    # Frozen neighbours are common to the likelihood and predictive simulation.
    # Report conditioning loss; never renormalize a tiny surviving shape tail
    # without making it visible and enforcing the physical-retention guard.
    # The allowance is a bounded geometric operation on the source density,
    # NOT a raw-reference Boolean that can erase an entire edge dimension.
    # Actual recipient likelihoods, generative mass and physical/core guards
    # remain unchanged. Exact endpoint cells retain inherent lattice ambiguity.
    if 'cells' not in prepared_geometry:
        prepared_geometry['cells'] = np.c_[np.rint(2*grid['coordinates'][:, 0]-grid['left_hi']), grid['left_hi'],
            grid['right_lo'], np.rint(2*grid['coordinates'][:, 1]-grid['right_lo'])]
    cells = prepared_geometry['cells']
    adjusted = None
    if _geometry_cache is not None:
        if 'shape_penalty' not in prepared_geometry:
            from .measurement_distribution import _shape_penalty
            prepared_geometry['shape_penalty'] = _shape_penalty(d, grid['starts'], grid['ends'],
                boundary_cells=cells, edge_tolerance_bp=floor_bp)
        adjusted = prepared_geometry['shape_penalty']
    seed_key = f"native-cross-v1|{frozen['model']['family']}|{call['unit_id']}|{call['ordinal']}"
    seed = int.from_bytes(hashlib.sha256(seed_key.encode()).digest()[:4], 'little')
    score = predictive_reference(d, log_mass, obs['likelihood'], grid['starts'], grid['ends'],
        allowed=allowed, observed=observed, p_accessible=obs['p_accessible'], p_protected=obs['p_protected'],
        replicates=replicates, seed=seed, boundary_cells=cells, edge_tolerance_bp=floor_bp,
        _defer_simulation=_defer_simulation, _prepared_shape_penalty=adjusted)
    geometry = frozen.get('geometry_summary') or summarize_grid_geometry(
        frozen['grid'], transfer_density(frozen, frozen['grid'])+np.log(frozen['grid']['areas']))
    mean = geometry['mean']
    if 'covered' not in prepared_geometry:
        prepared_geometry['covered'] = projection_coverage_probability(
            log_mass, grid['starts'], grid['ends'], len(grid['positions']))
    covered = prepared_geometry['covered']
    core = observed & (covered >= .5) & (grid['positions'] >= call['start']) & (grid['positions'] < call['end'])
    core_lr = float(obs['values'][core].sum())
    veto_lr = core_lr
    mixture_details = {}
    if 'source_log_density' in frozen:
        # A majority-coverage core must not force every mixture component to
        # protect that entire core. Integrate EACH geometry's native likelihood
        # on this SAME fixed observed core with its frozen probability weight.
        # This is neither an unweighted OR nor a second subtraction in the
        # profile statistic. An infinitesimal surviving mode cannot erase a
        # genuinely negative weighted marginal likelihood.
        prefix_core = np.r_[0., np.cumsum(np.where(core, obs['values'], 0.))]
        per_geometry = prefix_core[grid['ends']]-prefix_core[grid['starts']]
        veto_lr = float(logsumexp((log_mass+per_geometry)[allowed])-math.log(kept))
        mixture_details.update(mixture_core_native_log_lr=veto_lr,
            core_veto_semantics='native_mixture_marginal_on_fixed_observed_core',
            mixture_core_uses_frozen_weights=True)
        if frozen.get('mixture_components'):
            component_records = []; weighted_prior = []; weighted_evidence = []
            for component in frozen['mixture_components']:
                weight = float(component['weight'])
                cd = transfer_density(component['frozen'], grid)
                cm = cd+np.log(grid['areas'])
                prior = float(logsumexp(cm[allowed]))
                evidence = float(logsumexp((cm+obs['likelihood'])[allowed]))
                lw = math.log(weight) if weight > 0 else -np.inf
                weighted_prior.append(lw+prior); weighted_evidence.append(lw+evidence)
                component_records.append(dict(family=component['family'], source_mixture_weight=weight,
                    admissible_source_mass=float(np.exp(prior)),
                    native_pattern_log_lr=evidence-prior if np.isfinite(prior) else None))
            prior_z = logsumexp(weighted_prior); evidence_z = logsumexp(weighted_evidence)
            if not np.isfinite(prior_z) or not np.isfinite(evidence_z):
                raise ValueError('Mixture component diagnostics lack admissible native evidence')
            for i, component in enumerate(component_records):
                component.update(conditional_prior_weight=float(np.exp(weighted_prior[i]-prior_z)),
                    posterior_weight_given_native_observations=float(np.exp(weighted_evidence[i]-evidence_z)))
            mixture_details['mixture_component_evidence'] = component_records
    # Separate hard contradiction; don't subtract it from the statistic again.
    contradicted = core.any() and veto_lr < -math.log(core_contradiction_odds)
    return record | score | mixture_details | dict(status='core_contradicted' if contradicted else 'scored',
        mean_core_opportunities=int(core.sum()), mean_core_native_log_lr=core_lr,
        core_geometry_mean=mean, core_geometry_summary='normalized_positive_cell_coverage',
        core_minimum_geometry_coverage=.5,
        # A positive unrelaxed loss might already pass its predictive test.
        # Zeroing that loss is not proof that compatibility depends on the floor.
        floor_zeroed_loss=score['floor_adjusted_loss'] <= 1e-10 and score['native_loss'] > 1e-10,
        minimum_edge_floor_applied=score['floor_adjusted_loss'] < score['native_loss']-1e-10)


def summarize_direction(records, *, recipient_dataset, reference_percent=99.9):
    cut = 1.-reference_percent/100.
    compatible = [r for r in records if r['status'] == 'scored'
                  and r['predictive_tail_interval'][1] >= cut]
    scored = [r for r in records if r['status'] == 'scored']
    certain = [r for r in compatible if r['predictive_tail_interval'][0] >= cut]
    testable = [r for r in records if r['status'] in ('scored','core_contradicted')]
    reasons = Counter(r['status'] if r['status'] != 'scored' else
                      ('compatible' if r['predictive_tail_interval'][1] >= cut else 'predictive_rejected') for r in records)
    return dict(recipient_dataset=recipient_dataset, n_source_calls=len(records),
        n_scored=len(scored), n_testable=len(testable), n_compatible=len(compatible),
        n_compatible_mc_clear=len(certain), n_compatible_mc_borderline=len(compatible)-len(certain),
        compatible_fraction=len(compatible)/len(testable) if testable else None,
        compatible_mc_clear_fraction=len(certain)/len(testable) if testable else None,
        source_coverage_fraction=len(testable)/len(records) if records else None,
        compatible_all_source_fraction=len(compatible)/len(records) if records else None,
        source_units=len({r.get('evidence_group_id',r.get('unit_id')) for r in records}),
        testable_units=len({r.get('evidence_group_id',r.get('unit_id')) for r in testable}),
        compatible_units=len({r.get('evidence_group_id',r.get('unit_id')) for r in compatible}),
        floor_zeroed_compatible_calls=sum(r.get('floor_zeroed_loss',r.get('tolerance_only_compatibility',False)) for r in compatible),
        median_predictive_tail=float(np.median([r['predictive_tail'] for r in scored])) if scored else None,
        untestable_calls=sum(reasons[r] for r in ('no_recipient_information', 'geometry_blocked_by_neighbours_or_domain','transferred_shape_unobserved',
                                                 'hypothesis_not_attributable_to_this_call')),
        information_limited_calls=sum(reasons[r] for r in ('no_recipient_information','transferred_shape_unobserved')),
        geometry_limited_calls=sum(reasons[r] for r in ('geometry_blocked_by_neighbours_or_domain','hypothesis_not_attributable_to_this_call')),
        contradicted_calls=reasons['core_contradicted'], reason_counts=dict(reasons),
        by_strand={s: summarize_direction([r for r in records if r['strand'] == s],
                         recipient_dataset=recipient_dataset, reference_percent=reference_percent)
                   for s in sorted({r['strand'] for r in records})} if len({r['strand'] for r in records}) > 1 else {})


def agreement_summary(datasets, nodes, links, members, *, minimum_calls,
                      classification_reference_percent=99.9, bin_bp=10, excluded=None):
    """Per-dataset agreement against an interpretable denominator.

    The pair count is combinatorial: two independently nominated catalogues
    produce |A| x |B| candidate pairs, most of them families at different
    places. The interpretable figures are how many families had enough evidence
    to be tested at all, and how many of those matched. Reported per dataset
    alongside, never instead of, the per-pair ledger.
    """
    ids = sorted(datasets)
    catalogue = {ds: [n['family'] for n in nodes if n['dataset_id'] == ds] for ds in ids}
    verdict, compatible = {ds: set() for ds in ids}, {ds: set() for ds in ids}
    for link in links:
        if link['status'] not in ('provisional_reciprocal_compatible', 'provisional_not_reciprocally_compatible'):
            continue
        for ds, fid in link['families'].items():
            verdict[ds].add(fid)
            if link['status'] == 'provisional_reciprocal_compatible':
                compatible[ds].add(fid)
    cut = 1.-classification_reference_percent/100.
    sites, supported = {}, {}
    for ds in ids:
        counts = defaultdict(int)
        for call in datasets[ds]['result']['calls']:
            counts[int((call['start']+call['end'])//2)//bin_bp] += 1
        sites[ds] = {b for b, n in counts.items() if n >= minimum_calls}
        # How many calls each family is predictively compatible with, whether or
        # not it won them. A family with support but no cohort lost its calls to
        # a neighbour; a family with no support is unsupported by the data.
        tally = defaultdict(int)
        for scores in datasets[ds]['result']['call_family_evidence']:
            for score in scores:
                if score['status'] == 'scored' and score.get('predictive_tail_interval', [0., 0.])[1] >= cut:
                    tally[score['family']] += 1
        supported[ds] = tally
    out = {}
    for ds in ids:
        cohort = [f for f in catalogue[ds] if len(members[ds].get(f, [])) >= minimum_calls]
        partner_sites = set().union(*(sites[o] for o in ids if o != ds)) if len(ids) > 1 else set()
        shared = sites[ds] & partner_sites
        thin = len(sites[ds]) and len(shared) < .5*len(sites[ds])
        # Two different causes of an untestable family, distinguished rather than
        # merged: no partner depth at this site, or calls routed to a neighbour.
        stranded = sum(1 for f in catalogue[ds]
                       if len(members[ds].get(f, [])) < minimum_calls and supported[ds][f] >= minimum_calls)
        unsupported = sum(1 for f in catalogue[ds] if not supported[ds][f])
        out[ds] = dict(families=len(catalogue[ds])+len((excluded or {}).get(ds, [])),
            **(dict(families_in_graph=len(catalogue[ds]),
                    families_excluded_by_source_units=len(excluded.get(ds, [])),
                    excluded_semantics='families below the declared node minimum of source units stay in the '
                        'catalogue but enter no pair; they are counted here and nowhere else in this summary')
               if excluded is not None else {}),
            families_with_cohort=len(cohort),
            families_without_a_cohort_but_predictively_supported=stranded,
            families_with_no_compatible_call=unsupported,
            families_reaching_a_verdict=len(verdict[ds] & set(catalogue[ds])),
            families_compatible_in_any_pair=len(compatible[ds] & set(catalogue[ds])),
            compatible_fraction_of_families_with_cohort=(len(compatible[ds] & set(cohort))/len(cohort)
                                                         if cohort else None),
            sites_with_cohort=len(sites[ds]), sites_shared_with_a_partner=len(shared), site_bin_bp=int(bin_bp),
            families_without_a_cohort=sum(1 for f in catalogue[ds] if not members[ds].get(f)),
            limiting_factor=('partner_depth' if thin else
                             'catalogue_resolution' if unsupported > .25*max(1, len(catalogue[ds])) else
                             'assignment' if stranded > .1*max(1, len(catalogue[ds])) else 'none'),
            limiting_factor_semantics=('partner_depth: this dataset has cohort sites where the partner has none. '
                'catalogue_resolution: many nominated families have no predictively compatible call at all, so the '
                'catalogue is finer than the evidence supports. assignment: families are predictively supported but '
                'hold no cohort, so their calls went to a neighbour.'),
            semantics=('Pair counts are combinatorial across two independent catalogues; the interpretable '
                       'figure is the fraction of families with a cohort that matched. Not an FDR or a '
                       'calibrated agreement probability.'))
    return out


def node_concordance(nodes, links, members, agreement, *, minimum_calls):
    """One explanation per family of why it did or did not match a partner family.

    Verdict statuses come from the pair ledger. Untested families are split by
    the reason the pair statistic was never reached, so a reader can separate
    "the partner has no molecules here" from "this family holds no cohort" from
    "the shape was tested and disagreed". Diagnostic only; nothing upstream
    reads it.
    """
    by_family = defaultdict(list)
    for link in links:
        for ds, fid in link['families'].items():
            by_family[(ds, fid)].append(link)
    out = {}
    for node in nodes:
        key = (node['dataset_id'], node['family'])
        statuses = [l['status'] for l in by_family.get(key, [])]
        cohort = len(members[node['dataset_id']].get(node['family'], []))
        if 'provisional_reciprocal_compatible' in statuses:
            status, reason = 'matched', 'reciprocally compatible with at least one partner family'
        elif 'provisional_not_reciprocally_compatible' in statuses:
            status, reason = 'discordant', 'tested against a partner family and the shapes disagreed'
        elif not statuses:
            status, reason = 'untested', 'no partner family within the nominated distance'
        elif cohort < minimum_calls:
            status, reason = 'untested', f'this family holds {cohort} member calls, fewer than the {int(minimum_calls)} required'
        elif all(st == 'provisional_no_native_primary_cohort' for st in statuses):
            status, reason = 'untested', 'every nominated partner family holds no cohort'
        elif all(st in ('provisional_no_native_primary_cohort', 'provisional_sparse_native_class') for st in statuses):
            status, reason = 'untested', 'every nominated partner family is below the minimum cohort'
        else:
            status, reason = 'untested', 'too few member calls were testable on the partner geometry'
        row = dict(status=status, reason=reason, member_calls=cohort, partner_pairs=len(statuses),
                   pair_statuses=dict(Counter(statuses)))
        limiting = (agreement or {}).get(node['dataset_id'], {}).get('limiting_factor')
        if status == 'untested' and limiting:
            row['dataset_limiting_factor'] = limiting
        out[node['family']] = row
    return out


def reciprocal_summary_status(summaries, *, minimum_calls=3, minimum_fraction=.5, minimum_testable_fraction=.5):
    """Empty classes and failed attribution are not sequence-resolution proofs."""
    values = list(summaries.values())
    if not values or any(s['n_source_calls'] == 0 for s in values):
        return 'provisional_no_native_primary_cohort'
    if any(s['n_source_calls'] < minimum_calls for s in values):
        return 'provisional_sparse_native_class'
    if any(s['source_coverage_fraction'] is None or s['source_coverage_fraction'] < minimum_testable_fraction for s in values):
        return 'provisional_incomplete_reciprocal_assessment'
    if any(s['n_testable'] < minimum_calls for s in values):
        return 'provisional_sparse_native_class'
    return ('provisional_reciprocal_compatible' if all(s['compatible_fraction'] >= minimum_fraction for s in values)
            else 'provisional_not_reciprocally_compatible')


def _member_opportunities(data, indices, region):
    """Exact outcome-free opportunity union, once per family, not per edge.

    Repeated call ordinals on one unit do not change a set union. Chunk input
    arrays so this optimization does not construct a full cohort-by-position
    matrix. Coordinates outside the fixed analysis domain cannot enter any
    pair's grid and are removed before concatenating.
    """
    union = np.empty(0, np.int64); chunks = []; count = 0
    units = sorted({data['result']['calls'][i]['unit_index'] for i in indices})
    for m in units:
        positions = np.asarray(data['units'][m]['positions'], np.int64)
        a, b = np.searchsorted(positions, region)
        for first in range(a, b, 65536):
            block = positions[first:min(first+65536, b)]
            chunks.append(block); count += len(block)
            if count >= 65536:
                union = np.union1d(union, np.concatenate(chunks)); chunks = []; count = 0
    if chunks: union = np.union1d(union, np.concatenate(chunks))
    return union


def _agreement_block(datasets, nodes, links, members, minimum_calls, classification_reference_percent,
                     excluded=None):
    summary = agreement_summary(datasets, nodes, links, members, minimum_calls=minimum_calls,
                                classification_reference_percent=classification_reference_percent,
                                excluded=excluded)
    concordance = node_concordance(nodes, links, members, summary, minimum_calls=minimum_calls)
    for node in nodes:
        node['concordance'] = concordance[node['family']]
    tally = {ds: dict(Counter(concordance[n['family']]['status'] for n in nodes if n['dataset_id'] == ds))
             for ds in sorted(datasets)}
    for ds in tally:
        if excluded is not None and excluded.get(ds):
            tally[ds]['excluded_insufficient_source_units'] = len(excluded[ds])
        summary[ds]['families_by_concordance'] = tally[ds]
    return dict(agreement_summary=summary)


def reciprocal_native_graph(datasets, *, region, reference_percent=99.9,
                            replicates=4095, minimum_fraction=.5, minimum_calls=3,
                            maximum_matrix_bytes=512*1024**2, progress=None,
                            minimum_testable_fraction=.5, classification_reference_percent=99.9,
                            minimum_call_attribution_mass=.05, minimum_geometry_retention=.05,
                            minimum_visible_geometry_mass=.05, membership_loss_odds=1.,
                            pair_nomination_rule='all_overlapping_frozen_model_domains',
                            pair_nomination_gap_bp=20, summarize_agreement=False,
                            minimum_node_source_units=0, cores=1):
    """All plausible pairs; immutable nodes and complete per-call failure ledger.

    `datasets[id]` contains chemistry, units, and the frozen native-family result.
    Donor models are full source-assay fits: assays do not share evidence units.
    No recipient outcomes are used to fit a foreign model. Native discovery and
    choice of the edge universe remain caller-conditioned, not OOF inference.

    ``cores`` is an execution-only budget. Family pairs are independent given the
    frozen catalogs, and every predictive experiment is seeded by a content hash,
    so assessing pairs in separate single-threaded processes returns the identical
    graph in the identical order. No pair, recipient, gate, or draw changes.
    """
    progress = progress or (lambda _: None)
    if (not 50 <= reference_percent < 100 or not 50 <= classification_reference_percent < 100
            or not isinstance(replicates,int) or replicates<1):
        raise ValueError('Valid predictive reference and positive simulation count required')
    if any(not np.isfinite(x) or not 0 < x <= 1 for x in
           (minimum_call_attribution_mass,minimum_geometry_retention,minimum_visible_geometry_mass)):
        raise ValueError('Geometry attribution/retention and visibility fractions must be in (0,1]')
    if isinstance(cores, bool) or not isinstance(cores, (int, np.integer)) or cores < 1:
        raise ValueError('Positive integer pair-assessment worker budget required')
    zero_upper=3.841458820694124/(replicates+3.841458820694124)
    if zero_upper >= 1.-reference_percent/100.:
        raise ValueError(f'Predictive gate unresolved: {replicates} draws cannot reject at {reference_percent}%; '
                         f'zero-exceedance Monte Carlo upper bound is {zero_upper:.6g}')
    frozen, members, nodes = {}, {}, []
    member_opportunities = {}; excluded_nodes = {}; excluded_families = {}
    for ds, data in sorted(datasets.items()):
        result = data['result']; frozen[ds] = {}; member_opportunities[ds] = {}
        if (result.get('diagnostics',{}).get('family_model')!='latent_distribution'
                or not result.get('diagnostics',{}).get('predictive_replicates',0)):
            raise ValueError(f'{ds}: frozen latent-distribution models and predictive references are required')
        members[ds] = _reference_members(result, classification_reference_percent, membership_loss_odds)
        for model in result['family_models']:
            if model.get('status') != 'fitted' or 'fold_models' not in model:
                continue
            if int(model.get('source_units', 0)) < int(minimum_node_source_units):
                # Declared node gate: below the cohort minimum a family can never
                # reach a reciprocal verdict; it stays in the catalogue, not the graph.
                excluded_nodes[ds] = excluded_nodes.get(ds, 0)+1
                excluded_families.setdefault(ds, []).append(model['family'])
                continue
            fid = model['family']
            member_opportunities[ds][fid] = _member_opportunities(data, members[ds].get(fid, []), region)
            frozen[ds][fid] = frozen_model_geometry(model, result['calls'], data['units'])
            nodes.append(dict(dataset_id=ds, family=fid, interval=model['reference_interval'],
                fitted_center=frozen[ds][fid]['geometry_summary']['mean'],
                normalized_geometry=frozen[ds][fid]['geometry_summary'],
                untruncated_gaussian_location=model['fitted_distribution_center'], source_units=model['source_units'],
                primary_calls=len(members[ds].get(fid, [])),
                fit_diagnostics=model['fit_diagnostics']['full'], native_model_immutable=True))
    from .progress import report_work
    from .cross_preparation import NativeReadCache
    # Bounded caches share the declared matrix budget with active recipients.
    # Eviction only triggers recomputation; it never limits reads/families.
    cache_budget = maximum_matrix_bytes//8
    params = dict(region=region, reference_percent=reference_percent, replicates=replicates,
                  minimum_fraction=minimum_fraction, minimum_calls=minimum_calls,
                  maximum_matrix_bytes=maximum_matrix_bytes, minimum_testable_fraction=minimum_testable_fraction,
                  minimum_call_attribution_mass=minimum_call_attribution_mass,
                  minimum_geometry_retention=minimum_geometry_retention,
                  minimum_visible_geometry_mass=minimum_visible_geometry_mass, cache_budget=cache_budget)
    links = []; ids = sorted(datasets); not_nominated = {}
    for ia, da in enumerate(ids):
        for db in ids[ia+1:]:
            candidates = [(fa, fb) for fa, a in frozen[da].items() for fb, b in frozen[db].items()
                          if model_overlap(a, b)]
            if pair_nomination_rule == 'reference_interval_gap':
                # Two families at different places cannot produce a verdict: in
                # both directions the transferred geometry keeps too little mass
                # over the recipient call. Declining to assess them is a declared
                # nomination rule, not a silent filter on computed evidence.
                intervals = {da: {f: m['model']['reference_interval'] for f, m in frozen[da].items()},
                             db: {f: m['model']['reference_interval'] for f, m in frozen[db].items()}}
                def _gap(fa, fb):
                    (a0, a1), (b0, b1) = intervals[da][fa], intervals[db][fb]
                    return max(0, max(a0, b0)-min(a1, b1))
                considered = len(candidates)
                candidates = [(fa, fb) for fa, fb in candidates if _gap(fa, fb) <= pair_nomination_gap_bp]
                not_nominated[f'{da}/{db}'] = considered-len(candidates)
            report_work(progress,f'{da}/{db}: testing {len(candidates)} reciprocal family pairs',
                task=f'{da}/{db}',completed=0,total=len(candidates),unit='family pairs')
            workers = min(int(cores), len(candidates))
            if workers > 1:
                assessed = _assess_pairs_in_processes(datasets, frozen, members, member_opportunities,
                    params, da, db, candidates, workers, progress)
            else:
                read_cache = NativeReadCache(cache_budget)
                assessed = []
                for number, (fa, fb) in enumerate(candidates, 1):
                    assessed.append(_assess_pair(datasets, frozen, members, member_opportunities, params,
                        da, fa, db, fb, number, len(candidates), progress, read_cache))
                    report_work(progress,f'{da}/{db}: {number}/{len(candidates)} pairs completed',
                        task=f'{da}/{db}',completed=number,total=len(candidates),unit='family pairs')
            for link in assessed:
                links.append(dict(link_id=f'X{len(links)+1:06d}', **link))
    return dict(schema='fiberhmm.native_family_xcr.v1', status='complete', nodes=nodes, links=links,
        **(_agreement_block(datasets, nodes, links, members, minimum_calls, classification_reference_percent,
                            excluded=excluded_families if int(minimum_node_source_units) > 0 else None)
           if summarize_agreement else {}),
        parameters=dict(reference_percent=reference_percent, predictive_replicates=replicates,
                        minimum_fraction=minimum_fraction, minimum_calls=minimum_calls,
                        minimum_testable_fraction=minimum_testable_fraction,
                        classification_reference_percent=classification_reference_percent,
                        nomination_rule=pair_nomination_rule,
                        **(dict(pair_nomination_gap_bp=int(pair_nomination_gap_bp),
                                pairs_not_nominated=dict(sorted(not_nominated.items())),
                                nomination_rule_semantics='pairs whose native reference intervals lie farther apart '
                                    'than the declared gap are not assessed; they are absent, not failed')
                           if pair_nomination_rule == 'reference_interval_gap' else {}),
                        **(dict(membership_loss_odds=float(membership_loss_odds),
                                membership_rule='primary_plus_ties_within_loss_margin')
                           if membership_loss_odds > 1. else {}),
                        **(dict(minimum_node_source_units=int(minimum_node_source_units),
                                nodes_excluded_by_source_units=dict(sorted(excluded_nodes.items())),
                                families_excluded_by_source_units={ds: sorted(v) for ds, v in sorted(excluded_families.items())})
                           if int(minimum_node_source_units) > 0 else {}),
                        minimum_original_geometry_mass_for_call_attribution=minimum_call_attribution_mass,
                        minimum_physical_geometry_retention=minimum_geometry_retention,
                        minimum_visible_geometry_mass=minimum_visible_geometry_mass,
                        edge_tolerance_semantics='bounded_joint_endpoint_cell_profile_v1',
                        pair_floor_rule='maximum of the two predeclared native extra-edge tolerances'),
        diagnostics=dict(all_plausible_pairs_retained=True, counts_used_to_match=False,
            source_density_mass_preserved=True, merged_intervals=False, calibrated=False,
            grouping_is_not_new_call_rescue=True, recipient_outcomes_fit_foreign_models=False,
            raw_interval_overlap_is_not_a_cross_test_gate=False,
            observed_call_attribution_integrates_full_geometry=True,
            exact_recipient_boundary_neighbor_cuts=True,
            failed_assessment_not_automatically_a_resolution_claim=True,
            geometry_summary_uses_normalized_positive_cells=True,
            bounded_edge_allowance_without_reference_trigger=True,
            edge_allowance_changes_generative_mass=False,
            hia5_alignment_orientation_not_a_chemical_stratum=True))


def _assess_pair(datasets, frozen, members, member_opportunities, params, da, fa, db, fb,
                 number, total, progress, read_cache):
    """One reciprocal family pair, exactly as the historical inline loop body.

    The returned link has no ``link_id``: identifiers are assigned by the caller
    in candidate order, independent of which process or thread finished first.
    """
    from .progress import report_work
    from .scoring_execution import score_native_recipients
    from .measurement_distribution import complete_predictive_reference
    from .cross_preparation import TransferGeometryCache
    region = params['region']; maximum_matrix_bytes = params['maximum_matrix_bytes']
    cache_budget = params['cache_budget']; replicates = params['replicates']
    a, b = frozen[da][fa], frozen[db][fb]
    domain = [max(region[0], min(a['grid']['domain'][0], b['grid']['domain'][0])),
              min(region[1], max(a['grid']['domain'][1], b['grid']['domain'][1]))]
    positions = np.union1d(a['grid']['positions'], b['grid']['positions'])
    # Include all recipient opportunities, including any not present
    # in the subset of calls used to build the model's original grid.
    for ds, fid in ((da, fa), (db, fb)):
        positions = np.union1d(positions, member_opportunities[ds][fid])
    grid = boundary_grid(positions, domain,model_domains=[a['grid']['domain'],b['grid']['domain']])
    if len(grid['starts'])*8*18 > maximum_matrix_bytes:
        raise MemoryError(f'{da}:{fa} / {db}:{fb}: exact transfer grid exceeds budget; no edges dropped')
    directions, summaries = {}, {}
    # One declared operational tolerance in both transfer directions.
    # Native evidence losses remain separately exported. This floor
    # adds no observations and never bypasses the core veto.
    pair_floor=max(datasets[ds]['result']['diagnostics']['minimum_edge_tolerance_bp'] for ds in (da,db))
    for recipient, rf, donor, source in ((da, fa, db, b), (db, fb, da, a)):
        data = datasets[recipient]
        geometry_cache = TransferGeometryCache(source, grid, pair_floor, cache_budget)
        def recipient_progress(done, count):
            report_work(progress,f'{da}/{db}: {fa} ↔ {fb}; {recipient} recipients {done}/{count}',
                task=f'{da}/{db}',completed=number-1,total=total,unit='family pairs')
        def score_recipient(i):
            call = data['result']['calls'][i]
            if data['chemistry'].startswith('hia5'):
                call=dict(call,strand='pooled',alignment_orientation=call['strand'])
            return transferred_call(source, grid, data['units'][call['unit_index']], call,
                floor_bp=pair_floor, replicates=replicates,require_call_overlap=True,
                minimum_retained_mass=params['minimum_geometry_retention'],
                minimum_call_attribution_mass=params['minimum_call_attribution_mass'],
                minimum_visible_geometry_mass=params['minimum_visible_geometry_mass'],
                _defer_simulation=True, _read_cache=read_cache, _geometry_cache=geometry_cache)
        # Bound all raw-call/neighbour cuts, not just this call's
        # endpoints. Outcome-free member opportunity unions above
        # already include all recipient opportunity positions.
        indices = members[recipient].get(rf, [])
        recipient_units = {data['result']['calls'][i]['unit_index'] for i in indices}
        max_calls = max((len(data['units'][m]['representative_raw_tf_intervals'])+
                         len(data['units'][m].get('raw_nuc_intervals', []))
                         for m in recipient_units), default=0)
        refined_k = len(grid['positions'])+2+2*max_calls
        records = score_native_recipients(score_recipient, indices,
            maximum_bytes=max(1, maximum_matrix_bytes-2*cache_budget),
            bytes_per_item=refined_k*(refined_k+1)//2*8*64,
            finish=complete_predictive_reference,
            progress=recipient_progress if progress is not None else None)
        directions[recipient] = records
        summaries[recipient] = summarize_direction(records, recipient_dataset=recipient,
                                                   reference_percent=params['reference_percent'])
    status = reciprocal_summary_status(summaries,minimum_calls=params['minimum_calls'],
        minimum_fraction=params['minimum_fraction'],minimum_testable_fraction=params['minimum_testable_fraction'])
    compatible = status == 'provisional_reciprocal_compatible'
    return dict(datasets=[da, db], families={da:fa, db:fb},
        native_intervals={da:a['model']['reference_interval'], db:b['model']['reference_interval']},
        status=status, native_family_evidence=summaries, call_evidence=directions,
        compatibility_depends_on_monte_carlo_uncertainty=(compatible and any(
            s['compatible_mc_clear_fraction'] < params['minimum_fraction'] for s in summaries.values())),
        merged_interval=None, comparable=compatible, calibrated=False,pair_minimum_edge_tolerance_bp=pair_floor)


_PAIR_WORKER = {}
from .execution import register_worker_state as _register_worker_state
_register_worker_state(_PAIR_WORKER)
_UNIT_FIELDS = ('unit_id', 'strand', 'positions', 'hits', 'p_accessible', 'p_protected',
                'representative_raw_tf_intervals', 'raw_nuc_intervals')


def _pair_worker_payload(datasets, frozen, members, member_opportunities, params):
    """Immutable inputs a pair worker needs; observations as compact arrays.

    Only the fields read by transfer scoring are shipped. Numerical arrays are
    exactly the values the parent would have converted with ``np.asarray``.
    """
    slim = {}
    for ds, data in datasets.items():
        units = []
        for u in data['units']:
            unit = {k: u[k] for k in _UNIT_FIELDS if k in u}
            unit['positions'] = np.asarray(u['positions'], dtype=np.int64)
            unit['hits'] = np.asarray(u['hits'])
            unit['p_accessible'] = np.asarray(u['p_accessible'], float)
            unit['p_protected'] = np.asarray(u['p_protected'], float)
            units.append(unit)
        slim[ds] = dict(chemistry=data['chemistry'], units=units,
                        result=dict(calls=data['result']['calls'],
                                    diagnostics=dict(minimum_edge_tolerance_bp=
                                        data['result']['diagnostics']['minimum_edge_tolerance_bp'])))
    from .measurement_distribution import current_predictive_kernel
    return dict(datasets=slim, frozen=frozen, members=members,
                member_opportunities=member_opportunities, params=params,
                predictive_kernel=current_predictive_kernel())


def _load_pair_worker(path):
    import pickle
    with open(path, 'rb') as handle:
        return pickle.load(handle)


def _pair_task(path, index, da, fa, db, fb, total):
    from .execution import task_thread_budget, load_worker_state
    from .measurement_distribution import set_default_predictive_kernel
    task_thread_budget(1)
    state = load_worker_state(_PAIR_WORKER, path, _load_pair_worker)
    set_default_predictive_kernel(*state.get('predictive_kernel', ('reference', 0.)))
    if 'read_cache' not in state:
        from .cross_preparation import NativeReadCache
        state['read_cache'] = NativeReadCache(state['params']['cache_budget'])
    link = _assess_pair(state['datasets'], state['frozen'], state['members'], state['member_opportunities'],
                        state['params'], da, fa, db, fb, index+1, total, None, state['read_cache'])
    return index, link


def _assess_pairs_in_processes(datasets, frozen, members, member_opportunities, params,
                               da, db, candidates, workers, progress):
    """Independent pairs in spawned single-threaded workers; candidate order kept.

    Workers load one pickled immutable payload, own their bounded caches, and
    return complete links. The parent reports progress, assigns identifiers in
    candidate order, and cancels/drains the batch on any exception.
    """
    import pickle, tempfile
    from concurrent.futures import FIRST_COMPLETED, wait
    from pathlib import Path
    from .execution import stage_executor
    from .progress import report_work
    total = len(candidates)
    results = [None]*total
    with tempfile.TemporaryDirectory(prefix='fiberhmm-native-cross-') as directory:
        path = str(Path(directory)/'pairs.pkl')
        with open(path, 'wb') as handle:
            pickle.dump(_pair_worker_payload(datasets, frozen, members, member_opportunities, params),
                        handle, protocol=pickle.HIGHEST_PROTOCOL)
        executor, release = stage_executor(workers)
        pending = {}; remaining = iter(enumerate(candidates)); completed = 0
        def submit():
            item = next(remaining, None)
            if item is not None:
                index, (fa, fb) = item
                pending[executor.submit(_pair_task, path, index, da, fa, db, fb, total)] = index
        try:
            for _ in range(2*workers):
                submit()
            while pending:
                done, _ = wait(pending, timeout=.2, return_when=FIRST_COMPLETED)
                for future in done:
                    pending.pop(future)
                    index, link = future.result()
                    results[index] = link
                    completed += 1
                    submit()
                report_work(progress,f'{da}/{db}: {completed}/{total} pairs completed ({workers} workers)',
                    task=f'{da}/{db}',completed=completed,total=total,unit='family pairs')
        except BaseException:
            for future in pending:
                future.cancel()
            release(failed=True)
            raise
        release()
    if any(link is None for link in results):
        raise RuntimeError('Pair assessment finished without every candidate result')
    return results
