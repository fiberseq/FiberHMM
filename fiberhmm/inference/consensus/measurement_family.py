"""Native family-profile classification of already detected intervals.

A shared interval (diagnostic control), or a latent boundary distribution, is
fitted through a family's native observation likelihoods. A recipient is
compared with that model, not a percentile of noisy witnesses. Its complete
evidence group is excluded from fitting. The distribution comparison is a
penalized profile score with a conditional predictive reference, NOT an exact
shared-state LR, posterior, protection call, or occupancy estimate.

Proposal nomination remains caller-conditioned: full-cohort raw geometry can
nominate a model. Leave-group-out parameter fitting therefore is NOT advertised
as out-of-fold discovery or independently calibrated evidence.
"""
from __future__ import annotations

from collections import Counter, OrderedDict
from types import SimpleNamespace
from copy import deepcopy
import hashlib
import math

import numpy as np

from .native_presentation import member_families

from .measurement_grouping import _calls
from .measurement_compatibility import edge_floor_loss
from .measurement_geometry import summarize_boundary_cells, projection_coverage_probability


def profile_comparison(training, recipient, starts, ends, *,
                       recipient_allowed=None, training_allowed=None):
    """Exact shared/separate and single-edge-relaxed profiles on ONE grid.

    All inputs have the same accessible base measure and observation domain.
    In particular this does not compare two Bayes factors from different nulls.
    Starts/ends are projection-cut indices; duplicate projections are forbidden.
    No geometry multiplicity or fitted class-frequency term enters the score.
    """
    t, r = np.asarray(training, float), np.asarray(recipient, float)
    a, b = np.asarray(starts), np.asarray(ends)
    if (t.ndim != 1 or not len(t) or r.shape != t.shape or a.shape != t.shape or b.shape != t.shape
            or a.dtype.kind not in 'iu' or b.dtype.kind not in 'iu'
            or np.any(a < 0) or np.any(a >= b) or np.any(~np.isfinite(t)) or np.any(~np.isfinite(r))):
        raise ValueError('Finite same-grid native profiles and integer positive projections required')
    if len(np.unique(np.c_[a, b], axis=0)) != len(a):
        raise ValueError('Projection aliases must be quotiented before scoring')
    ta = np.ones(len(t), bool) if training_allowed is None else np.asarray(training_allowed, bool)
    ra = np.ones(len(t), bool) if recipient_allowed is None else np.asarray(recipient_allowed, bool)
    if ta.shape != t.shape or ra.shape != t.shape:
        raise ValueError('One eligibility flag per projection required')
    if not ta.any() or not ra.any() or not (ta & ra).any():
        return dict(status='no_common_projection', native_loss=None)
    mt, mr = float(t[ta].max()), float(r[ra].max())
    shared = float((t+r)[ta & ra].max())
    end_t = np.full(int(b.max())+1, -np.inf); end_r = end_t.copy()
    start_t = np.full(int(b.max())+1, -np.inf); start_r = start_t.copy()
    np.maximum.at(end_t, b[ta], t[ta]); np.maximum.at(end_r, b[ra], r[ra])
    np.maximum.at(start_t, a[ta], t[ta]); np.maximum.at(start_r, a[ra], r[ra])
    return dict(status='scored', native_loss=max(0., mt+mr-shared),
                left_edge_relaxed_loss=max(0., mt+mr-float((end_t+end_r).max())),
                right_edge_relaxed_loss=max(0., mt+mr-float((start_t+start_r).max())),
                source_optimum=mt, recipient_optimum=mr, shared_optimum=shared)


def _native_observation_arrays(unit):
    """Validated numerical context reused inside one classification run."""
    p = np.asarray(unit['positions'], dtype=np.int64)
    h = np.asarray(unit['hits'])
    pa, pp = np.asarray(unit['p_accessible'], float), np.asarray(unit['p_protected'], float)
    if (not (p.shape == h.shape == pa.shape == pp.shape) or np.any(np.diff(p) <= 0)
            or np.any((h != 0) & (h != 1)) or np.any(~np.isfinite(pa)) or np.any(~np.isfinite(pp))
            or np.any((pa <= 0) | (pa >= 1) | (pp <= 0) | (pp >= 1))):
        raise ValueError('Ordered native observations and valid probabilities required')
    frozen = np.zeros(len(p), bool)
    for x, y in list(unit['representative_raw_tf_intervals'])+list(unit.get('raw_nuc_intervals', [])):
        frozen |= (p >= x) & (p < y)
    values = np.where(h, np.log(pp/pa), np.log1p(-pp)-np.log1p(-pa))
    return p, values, frozen, pa, pp


def _native_values(unit, positions, own_call, *, prepared=None):
    """Likelihood relative to accessible background, with other calls frozen.

    Other already detected TF/nucleosome spans are held protected in BOTH
    hypotheses. Their observations cancel, except where they overlap this
    source interval (whose actual observations must always remain present).
    This avoids fitting the next footprint as part of this one. It is a
    caller-conditioned background, not a claim that raw calls are ground truth.
    Missing opportunities and conditioned background contribute zero, never a
    fabricated modification miss. Recipient information counts use this mask.
    """
    p, native, occupied, _, _ = _native_observation_arrays(unit) if prepared is None else prepared
    keep = (p >= positions[0]) & (p <= positions[-1])
    frozen = occupied & ~((p >= own_call['start']) & (p < own_call['end']))
    keep &= ~frozen
    columns = np.searchsorted(positions, p[keep])
    values = np.zeros(len(positions)); observed = np.zeros(len(positions), bool)
    values[columns] = native[keep]
    observed[columns] = True
    return values, observed


def _projection_coordinates(positions, a, b, lo, hi, reference):
    """A projection has a coordinate CELL, not a magically exact bp boundary."""
    left_lo = lo if a == 0 else int(positions[a-1])+1
    left_hi = int(positions[a])
    right_lo = int(positions[b-1])+1
    right_hi = hi if b == len(positions) else int(positions[b])
    return [int(np.clip(reference[0], left_lo, left_hi)),
            int(np.clip(reference[1], right_lo, right_hi))], [[left_lo, left_hi], [right_lo, right_hi]]


def classify_family_profiles(stratum, catalog, *, region, loss_odds_levels=(10., 100., 1000.),
                             minimum_edge_tolerance_bp=0, core_contradiction_odds=100.,
                             maximum_matrix_bytes=512*1024**2, progress=None,
                             family_model='shared_interval', max_fit_iterations=100,
                             predictive_replicates=0, scoring_folds=10, membership_loss_odds=1.,
                             fit_backend='cpu', nonparametric_pseudo_units=4., _frozen_result=None, cores=1,
                             edge_tolerance_mode='legacy_profile', training_evidence_groups=None,
                             retry_fit_iterations=0):
    """Fit a native geometry profile for every initial proposal, then classify.

    Every independently represented source unit contributes once. All
    identifiable geometries in each fixed caller-union domain are evaluated;
    the bp floor never limits that grid. Every original call survives unchanged.
    Full-span centroids nominate source groups and break classification ties;
    raw geometry is not counted a second time in the likelihood.
    """
    if (not loss_odds_levels or any(not math.isfinite(v) or v < 1 for v in loss_odds_levels)
            or not math.isfinite(core_contradiction_odds) or core_contradiction_odds < 1):
        raise ValueError('Finite loss allowances of at least one required')
    if (isinstance(minimum_edge_tolerance_bp, bool)
            or not isinstance(minimum_edge_tolerance_bp, (int, np.integer)) or minimum_edge_tolerance_bp < 0):
        raise ValueError('A nonnegative integer per-edge tolerance is required')
    if maximum_matrix_bytes < 1 or len(region) != 2 or region[0] >= region[1]:
        raise ValueError('Positive compute budget and analysis region required')
    if family_model not in ('shared_interval', 'latent_distribution'):
        raise ValueError('Unknown native family model')
    if edge_tolerance_mode not in ('legacy_profile', 'bounded'):
        raise ValueError('Unknown edge_tolerance_mode')
    if edge_tolerance_mode == 'bounded' and family_model != 'latent_distribution':
        raise ValueError('Bounded edge tolerance requires latent_distribution')
    if not isinstance(scoring_folds, int) or scoring_folds < 2:
        raise ValueError('At least two evidence-group folds required')
    progress = progress or (lambda _: None)
    if isinstance(cores, bool) or not isinstance(cores, (int, np.integer)) or cores < 1:
        raise ValueError('Positive integer family worker budget required')
    calls = _calls(stratum); units = stratum['units']
    if (isinstance(retry_fit_iterations, bool) or not isinstance(retry_fit_iterations, int)
            or (retry_fit_iterations != 0 and retry_fit_iterations <= max_fit_iterations)):
        raise ValueError('Retry iteration budget must be zero or exceed the initial fit budget')
    if training_evidence_groups is not None:
        if isinstance(training_evidence_groups, (str, bytes)):
            raise ValueError('Explicit training evidence-group collection required')
        training_evidence_groups = frozenset(training_evidence_groups)
        available_groups = {u.get('fold_group_id', u['unit_id']) for u in units}
        if not training_evidence_groups or not training_evidence_groups <= available_groups:
            raise ValueError('Nonempty known training evidence groups required')
    if _frozen_result is not None and (training_evidence_groups is not None or retry_fit_iterations):
        raise ValueError('Explicit training/retry mode cannot use legacy frozen composition')
    # This cache never survives a run or a change to the normalized intervals.
    native_arrays = _native_arrays_cache(units, maximum_matrix_bytes)
    spans = np.asarray([[c['start'], c['end']] for c in calls], dtype=np.int64).reshape(-1, 2)
    centers = np.asarray([[f['consensus_start'], f['consensus_end']] for f in catalog], dtype=np.int64).reshape(-1, 2)
    if not len(centers) or np.any(centers[:, 0] >= centers[:, 1]):
        raise ValueError('Nonempty positive-width initial proposals required')
    ids = [f['family'] for f in catalog]
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate family identities')
    active = (spans[:, 0] < region[1]) & (spans[:, 1] > region[0])
    homes = np.full(len(calls), -1, int)
    for i in np.flatnonzero(active):
        ff = np.flatnonzero((centers[:, 0] < spans[i, 1]) & (centers[:, 1] > spans[i, 0]))
        if len(ff):
            homes[i] = min(ff, key=lambda f: (int(np.square(spans[i]-centers[f]).sum()), ids[f]))
    summaries = [[] for _ in calls]; models = []
    reused = {}
    if _frozen_result is not None:
        frozen_options = _frozen_result.get('diagnostics', {})
        if (frozen_options.get('edge_tolerance_mode', 'legacy_profile') != edge_tolerance_mode
                or frozen_options.get('minimum_edge_tolerance_bp') != minimum_edge_tolerance_bp):
            raise ValueError('Frozen reuse cannot change edge tolerance mode or allowance')
        # Internal append-only workflow hook, not a disk/cache loading API.
        # The producer authenticates observations/options/implementation before
        # this call and validates BOTH bindings again during final composition.
        # Homes above MUST use the whole augmented catalog: new models retain
        # exactly the sources and folds of the former full-refit path.
        if (_frozen_result.get('status') != 'complete' or _frozen_result['calls'] != calls
                or len(_frozen_result['call_family_evidence']) != len(calls)):
            raise ValueError('Frozen reuse requires the identical complete call axis')
        reused = {m['family']: m for m in _frozen_result['family_models']}
        if not reused.keys() <= set(ids):
            raise ValueError('Frozen reuse cannot remove an existing family')
        summaries = deepcopy(_frozen_result['call_family_evidence'])
    from .progress import report_work
    ctx = SimpleNamespace(calls=calls, units=units, spans=spans, centers=centers, ids=ids, homes=homes,
        active=active, region=region, native_arrays=native_arrays,
        training_evidence_groups=training_evidence_groups, retry_fit_iterations=retry_fit_iterations,
        minimum_edge_tolerance_bp=minimum_edge_tolerance_bp, core_contradiction_odds=core_contradiction_odds,
        edge_tolerance_mode=edge_tolerance_mode,
        maximum_matrix_bytes=maximum_matrix_bytes, family_model=family_model,
        max_fit_iterations=max_fit_iterations, predictive_replicates=predictive_replicates,
        scoring_folds=scoring_folds, progress=progress)
    order = sorted(range(len(ids)), key=lambda j: ids[j])
    pending_families = [(number, f) for number, f in enumerate(order) if ids[f] not in reused]
    workers = min(int(cores), len(pending_families))
    outcomes = {}
    if workers > 1:
        # Families are independent given the frozen proposals: one worker
        # fits and scores a whole family with the unchanged serial kernels.
        # Results are merged in sorted family order, exactly as the loop did.
        outcomes = _classify_families_in_processes(ctx, pending_families, len(ids), workers, progress)
    for number, f in enumerate(order):
        if ids[f] in reused:
            model = reused[ids[f]]
            if model.get('status') == 'fitted' and model['reference_interval'] != centers[f].tolist():
                raise ValueError('Frozen reuse cannot change proposal geometry')
            models.append(deepcopy(model))
            continue
        if f in outcomes:
            model, entries = outcomes[f]
            report_work(progress,f'{ids[f]}: merged worker result',completed=number+1,total=len(ids),unit='families')
        else:
            model, entries = _classify_one_family(ctx, f, number, len(ids))
        models.append(model)
        for i, summary in entries:
            summaries[i].append(summary)
    report_work(progress,'All native family models tested; assembling frozen classifications',
        completed=len(ids),total=len(ids),unit='families')
    partitions = {}; predictive_partitions = {}
    settings = [('loss', v) for v in sorted(set(map(float, loss_odds_levels)))]
    if predictive_replicates:
        settings += [('predictive', v) for v in (95., 99., 99.9)]
    for setting, odds in settings:
        assignments = []; members = {}; primary_members = {}; provisional = {}
        for i in np.flatnonzero(active):
            compatible = sorted((s for s in summaries[i] if s['status'] == 'scored'
                and (s['floor_adjusted_loss'] <= math.log(odds)+1e-9 if setting == 'loss' else
                     s.get('predictive_tail_interval', [0., 0.])[1] >= 1.-odds/100.)),
                key=lambda s: (s['geometry_distance_sq'], s['floor_adjusted_loss'], s['family']))
            if compatible:
                primary = compatible[0]; fid = primary['family']; status = 'compatible_catalog_label'
            else:
                primary = None
                fid = stratum['dataset_id']+':unresolved_'+hashlib.sha256(f'{spans[i,0]}|{spans[i,1]}'.encode()).hexdigest()[:12]
                provisional[fid] = spans[i].tolist(); status = 'provisional_unresolved'
            assignment = dict(**calls[i], interval=spans[i].tolist(), family=fid,
                classification_status=status, primary_evidence=primary,
                compatible_alternatives=[s['family'] for s in compatible[1:]],
                source_geometry_home=ids[homes[i]] if homes[i] >= 0 else None)
            # Membership can widen to a declared tie set; the primary label above
            # is untouched, so every existing consumer of `family` is unaffected.
            joined = member_families(compatible, membership_loss_odds) if compatible else [fid]
            if membership_loss_odds > 1.:
                assignment['member_families'] = joined
            assignments.append(assignment)
            primary_members.setdefault(fid, []).append(int(i))
            for member in joined:
                members.setdefault(member, []).append(int(i))
        out_catalog = []; by_id = {f['family']: f for f in catalog}
        for fid, mm in sorted(members.items()):
            is_provisional = fid in provisional
            interval = provisional[fid] if is_provisional else [by_id[fid]['consensus_start'], by_id[fid]['consensus_end']]
            entry = dict(family=fid, representative_interval=interval, source_calls=len(mm),
                calls_by_strand=dict(Counter(calls[i]['strand'] for i in mm)),
                units_by_strand={s: len({calls[i]['evidence_group_id'] for i in mm if calls[i]['strand'] == s})
                                 for s in sorted({calls[i]['strand'] for i in mm})},
                member_indices=mm, provisional_unresolved=is_provisional,
                provisional_singleton=len(mm) == 1, native_proposal_preserved=not is_provisional)
            if membership_loss_odds > 1.:
                # source_calls now counts tie-set members; the winner-take-all
                # count stays visible so the two are never confused.
                entry['primary_calls'] = len(primary_members.get(fid, []))
                entry['primary_indices'] = primary_members.get(fid, [])
            out_catalog.append(entry)
        target = partitions if setting == 'loss' else predictive_partitions
        target[str(odds)] = dict(catalog=out_catalog, assignments=assignments, loss_odds=odds if setting == 'loss' else None,
            predictive_reference_percent=odds if setting == 'predictive' else None,
            classes=len(out_catalog), unresolved_calls=sum(a['classification_status'] == 'provisional_unresolved' for a in assignments),
            ambiguous_calls=sum(bool(a['compatible_alternatives']) for a in assignments),
            all_source_calls_retained=len(assignments) == int(active.sum()), boundaries_changed=False,
            family_profile_model=True,
            **(dict(membership_loss_odds=float(membership_loss_odds),
                    membership_rule='primary_plus_ties_within_loss_margin',
                    tie_set_member_calls=sum(len(a['member_families']) for a in assignments if 'member_families' in a))
               if membership_loss_odds > 1. else {}))
    return dict(status='complete', calls=calls, partitions=partitions, predictive_partitions=predictive_partitions,
                call_family_evidence=summaries,
                family_models=models, source_homes=[ids[f] if f >= 0 else None for f in homes],
                diagnostics=dict(source_calls=int(active.sum()), out_of_region_calls=int((~active).sum()),
                    source_units=len(units), original_catalog_families=len(catalog),
                    **(dict(explicit_training_evidence_groups=sorted(training_evidence_groups),
                            fit_grid_domain='fixed_region_outcome_free_union_opportunities')
                       if training_evidence_groups is not None else {}),
                    **(dict(retry_fit_iterations=retry_fit_iterations) if retry_fit_iterations else {}),
                    minimum_edge_tolerance_bp=int(minimum_edge_tolerance_bp),
                    **(dict(edge_tolerance_mode=edge_tolerance_mode,
                            edge_tolerance_semantics='bounded_joint_endpoint_cell_profile_v1',
                            edge_tolerance_changes_generative_mass=False)
                       if edge_tolerance_mode == 'bounded' else {}),
                    core_contradiction_odds=core_contradiction_odds,
                    calibrated_confidence=False, probabilities_changed=False, boundaries_changed=False,
                    family_profile_model=True, witness_quantile=None, class_frequency_prior=False,
                    parameters_leave_evidence_group_out=True, proposal_discovery_out_of_fold=False,
                    family_model=family_model,
                    **(dict(membership_loss_odds=float(membership_loss_odds)) if membership_loss_odds > 1. else {}),
                    **(dict(fit_backend=str(fit_backend)) if fit_backend != 'cpu' else {}),
                    **(dict(nonparametric_pseudo_units=float(nonparametric_pseudo_units))
                       if fit_backend == 'nonparametric' else {}),
                    predictive_replicates=predictive_replicates,
                    scoring_folds=scoring_folds, frozen_neighbor_crossing_forbidden=True,
                    mean_core_observation_required_for_existing_call_classification=False,
                    predictive_reference_uses_exported_score=True,
                    predictive_reference_semantics='native model compatibility reference; conditional proposals; not validated false-split rate or FDR',
                    statistic=('native likelihood plus independently fitted shape-density penalty; not confidence or FDR'
                               if family_model == 'latent_distribution' else
                               'incremental native shared-vs-separate family profile loss; not confidence or FDR')))


def _native_arrays_cache(units, maximum_matrix_bytes):
    """Bounded per-run validated observation cache; identical values either way."""
    observation_cache = OrderedDict()
    state = dict(bytes=0)
    cache_budget = min(256*1024**2, maximum_matrix_bytes//4)
    def native_arrays(index):
        if index in observation_cache:
            observation_cache.move_to_end(index)
            return observation_cache[index]
        arrays = _native_observation_arrays(units[index])
        size = sum(a.nbytes for a in arrays)
        if size <= cache_budget:
            while observation_cache and state['bytes']+size > cache_budget:
                _, old = observation_cache.popitem(last=False)
                state['bytes'] -= sum(a.nbytes for a in old)
            observation_cache[index] = arrays; state['bytes'] += size
        return arrays
    native_arrays.bytes = lambda: state['bytes']
    return native_arrays


def _classify_one_family(ctx, f, number, total):
    """Fit and score ONE proposal; return its model and (call index, summary) entries.

    This is the historical loop body verbatim with shared-list mutation replaced
    by returned values, so a family can be processed in any process.
    """
    from .progress import report_work
    from .scoring_execution import score_native_recipients, defer_native_simulations
    from .measurement_distribution import complete_predictive_reference
    calls, units, spans, centers, ids, homes, active = (ctx.calls, ctx.units, ctx.spans, ctx.centers, ctx.ids,
                                                        ctx.homes, ctx.active)
    entries = []
    cache_bytes = ctx.native_arrays.bytes()
    def family_progress(detail):
        report_work(ctx.progress,f'{ids[f]}: {detail}',completed=number,total=len(ids),unit='families',detail=detail)
    family_progress('preparing native observations')
    recipients = np.flatnonzero(active & (spans[:, 0] < centers[f, 1]) & (spans[:, 1] > centers[f, 0]))
    # Select by geometry alone, BEFORE looking at native likelihood.
    sources = {}
    for i in np.flatnonzero(homes == f):
        group = calls[i]['evidence_group_id']
        if ctx.training_evidence_groups is not None and group not in ctx.training_evidence_groups:
            continue
        key = (int(np.square(spans[i]-centers[f]).sum()), calls[i]['unit_id'], calls[i]['ordinal'])
        if group not in sources or key < sources[group][0]:
            sources[group] = (key, int(i))
    if not sources:
        return dict(family=ids[f], status='no_source_observations', source_units=0), entries
    source_indices = [v[1] for _, v in sorted(sources.items())]
    domain_lo = max(ctx.region[0], int(min(spans[recipients, 0].min(), centers[f, 0])))
    domain_hi = min(ctx.region[1], int(max(spans[recipients, 1].max(), centers[f, 1])))
    positions = np.unique(np.concatenate([np.asarray(units[calls[i]['unit_index']]['positions'], np.int64)
                                           for i in recipients]))
    positions = positions[(positions >= domain_lo) & (positions < domain_hi)]
    if ctx.training_evidence_groups is not None:
        # Confirmation hit/call changes cannot modify a training model's grid
        # or domain. All supplied positions are fixed measurement covariates;
        # neither confirmation call spans nor their selection define this grid.
        domain_lo, domain_hi = ctx.region
        positions = np.unique(np.concatenate([np.asarray(u['positions'], np.int64) for u in units]))
        positions = positions[(positions >= domain_lo) & (positions < domain_hi)]
    if not len(positions):
        return dict(family=ids[f], status='no_observed_opportunities', source_units=len(sources)), entries
    k = len(positions); ng = k*(k+1)//2
    estimate = ng*(len(source_indices)+8)*8 + len(recipients)*k*16
    if estimate > ctx.maximum_matrix_bytes:
        raise MemoryError(f'{ids[f]} needs {estimate} matrix bytes; budget {ctx.maximum_matrix_bytes}; no calls/projections removed')
    aa, bb = np.triu_indices(k+1, 1)
    left_cell_hi = positions[aa]
    right_cell_lo = positions[bb-1]+1
    profiles = []; allowed = []; observations = {}
    for i in recipients:
        unit = units[calls[i]['unit_index']]
        unit_index = calls[i]['unit_index']
        values, mask = _native_values(unit, positions, calls[i], prepared=ctx.native_arrays(unit_index))
        prefix = np.r_[0., np.cumsum(values)]
        ca, cb = np.searchsorted(positions, spans[i])
        valid = (aa < cb) & (bb > ca)
        # A geometry can change this call's edges, but cannot cross a
        # neighboring frozen call. This prevents cost-free growth into
        # observations canceled by the conditioned background.
        previous, following = domain_lo, domain_hi
        for x, y in list(unit['representative_raw_tf_intervals'])+list(unit.get('raw_nuc_intervals', [])):
            if y <= spans[i, 0]: previous = max(previous, y)
            if x >= spans[i, 1]: following = min(following, x)
        valid &= (left_cell_hi >= previous) & (right_cell_lo <= following)
        observations[int(i)] = (prefix, mask, valid)
    source_indices = [i for i in source_indices if observations[i][2].any()
                      and np.any(observations[i][1] & (np.diff(observations[i][0]) != 0.))]
    if not source_indices:
        return dict(family=ids[f], status='no_informative_source', source_units=0), entries
    sources = {calls[i]['evidence_group_id']: (None, i) for i in source_indices}
    for i in source_indices:
        prefix, mask, valid = observations[i]
        profiles.append(prefix[bb]-prefix[aa]); allowed.append(valid)
    profiles = np.asarray(profiles); allowed = np.asarray(allowed)
    total = profiles.sum(0); invalid = (~allowed).sum(0)
    good = invalid == 0
    if not good.any() and ctx.family_model == 'shared_interval':
        return dict(family=ids[f], status='no_common_source_projection', source_units=len(sources)), entries
    if not good.any():
        # A distribution can explain different admissible geometries on
        # different source molecules. Requiring one interval shared by all
        # would silently restore the rejected point-family model.
        good = allowed.any(0)
    maximum = total[good].max()
    # Native score first. Full geometry only resolves exact model ties.
    maxima = np.flatnonzero(good & (total >= maximum-1e-9))
    best = min(maxima, key=lambda g: (sum((np.asarray(_projection_coordinates(positions, aa[g], bb[g], domain_lo, domain_hi, centers[f])[0])-centers[f])**2), int(g)))
    fitted, cell = _projection_coordinates(positions, aa[best], bb[best], domain_lo, domain_hi, centers[f])
    model = dict(family=ids[f], status='fitted', source_units=len(sources),
                 reference_interval=centers[f].tolist(), fitted_interval=fitted, fitted_boundary_cells=cell,
                 domain=[domain_lo, domain_hi], identifiable_projections=ng,
                 source_call_indices=source_indices, best_native_log_lr=float(maximum),
                 parameters_leave_evidence_group_out=True,
                 proposal_discovery_out_of_fold=False)
    source_rows = {calls[i]['evidence_group_id']: j for j, i in enumerate(source_indices)}
    distributions = {}; fold_by_group = {}; coordinates = areas = None
    bounded_penalties = {}; boundary_cells = None
    if ctx.family_model == 'latent_distribution':
        from .measurement_distribution import distribution_comparison, predictive_reference
        from .fit_execution import fit_native_models
        left_lo = np.r_[domain_lo, positions+1][aa]
        left_hi = positions[aa]
        right_lo = positions[bb-1]+1
        right_hi = np.r_[positions, domain_hi][bb]
        coordinates = np.c_[(left_lo+left_hi)/2., (right_lo+right_hi)/2.]
        areas = (left_hi-left_lo+1.)*(right_hi-right_lo+1.)
        if ctx.edge_tolerance_mode == 'bounded':
            boundary_cells = np.c_[left_lo, left_hi, right_lo, right_hi]
        # Balanced deterministic local folds; a whole evidence group stays
        # together. Every source participates in the full display fit, and
        # no recipient trains its own scoring model. Full-fit parameters
        # are deliberately NOT used to initialize the excluded-fold fits.
        ordered_groups = sorted(source_rows, key=lambda g: hashlib.sha256(('native-shape-v1|'+g).encode()).digest())
        n_folds = min(ctx.scoring_folds, len(ordered_groups))
        fold_by_group = {g: j % n_folds for j, g in enumerate(ordered_groups)}
        row_sets = {'full': None}
        for fold in range(n_folds):
            rows = [j for g, j in source_rows.items() if fold_by_group[g] != fold]
            if rows:
                row_sets[fold] = rows
        distributions = fit_native_models(profiles, allowed, coordinates, areas,
            row_sets=row_sets, reference=centers[f], max_iterations=ctx.max_fit_iterations,
            resident_bytes=cache_bytes+sum(a.nbytes for obs in observations.values() for a in obs),
            progress=family_progress)
        retry_records = {}
        failed = {key: row_sets[key] for key, fit in distributions.items() if not fit['converged']}
        if failed and ctx.retry_fit_iterations:
            retried = fit_native_models(profiles, allowed, coordinates, areas,
                row_sets=failed, reference=centers[f], max_iterations=ctx.retry_fit_iterations,
                resident_bytes=cache_bytes+sum(a.nbytes for obs in observations.values() for a in obs),
                progress=family_progress)
            for key, candidate in retried.items():
                original = distributions[key]
                accepted = candidate['objective'] <= original['objective'] + 1e-7
                if accepted:
                    distributions[key] = candidate
                retry_records[str(key)] = dict(initial_objective=original['objective'],
                    initial_iterations=original['iterations'], initial_message=original['message'],
                    retry_objective=candidate['objective'], retry_converged=candidate['converged'],
                    retry_iterations=candidate['iterations'], accepted=bool(accepted))
        full = distributions['full']
        for fit in distributions.values():
            fit['geometry_summary'] = summarize_boundary_cells(
                fit['log_mass'], left_lo, left_hi, right_lo, right_hi)
            fit['geometry_coverage'] = projection_coverage_probability(fit['log_mass'], aa, bb, len(positions))
        if ctx.edge_tolerance_mode == 'bounded':
            from .measurement_edge_tolerance import bounded_edge_penalty
            bounded_penalties = {key: bounded_edge_penalty(fit['log_density'], boundary_cells,
                ctx.minimum_edge_tolerance_bp, maximum_matrix_bytes=ctx.maximum_matrix_bytes)
                for key, fit in distributions.items()}
        model.update(family_model=ctx.family_model, fitted_distribution_center=full['center'].tolist(),
            fitted_distribution_covariance=full['covariance'].tolist(),
            normalized_geometry=full['geometry_summary'],
            fold_models={str(key): dict(parameters=fit['parameters'].tolist(),
                center=fit['center'].tolist(), covariance=fit['covariance'].tolist(),
                normalized_geometry=fit['geometry_summary'],
                parameter_coordinate_scale_bp=10., parameter_reference=centers[f].tolist(),
                projection_grid_sha256=hashlib.sha256(
                    np.c_[coordinates, areas].astype('<f8').tobytes()).hexdigest(),
                # A nonparametric mass is transferred by exact cell refinement, never by
                # evaluating the Gaussian seed: tabulate it on the full fold.
                **(dict(objective_backend='nonparametric',
                        tabulated_log_density=np.asarray(fit['log_density'], float).tolist(),
                        smoothing_pseudo_units=fit.get('smoothing_pseudo_units'),
                        nonparametric_iterations=fit.get('nonparametric_iterations'))
                   if fit.get('objective_backend') == 'nonparametric' and str(key) == 'full' else
                   (dict(objective_backend=fit['objective_backend']) if fit.get('objective_backend') else {})))
                for key, fit in distributions.items()},
            fit_diagnostics={str(key): {name: fit[name] for name in
                ('objective', 'converged', 'iterations', 'message', 'source_units')}
                for key, fit in distributions.items()},
            scoring_folds=n_folds,
            training_evidence_groups={str(key): sorted(g for g in source_rows if key == 'full' or fold_by_group[g] != key)
                                      for key in distributions})
        if ctx.retry_fit_iterations:
            model['fit_retry_diagnostics'] = retry_records
    pending_predictions = []
    prediction_batch = max(1, min(32, ctx.maximum_matrix_bytes//max(1, ng*8*64)))
    def flush_predictions():
        if not pending_predictions:
            return
        finished = score_native_recipients(lambda record: record, pending_predictions,
            finish=complete_predictive_reference, maximum_bytes=ctx.maximum_matrix_bytes,
            bytes_per_item=ng*8*64,
            progress=lambda done, count: family_progress(f'completing native simulations {done}/{count} in current batch'))
        for record, value in zip(pending_predictions, finished):
            record.clear(); record.update(value)
        pending_predictions.clear()
    for recipient_number,i in enumerate(recipients):
        if len(pending_predictions) >= prediction_batch:
            flush_predictions()
        if recipient_number % 32 == 0:
            family_progress(f'testing native recipients {recipient_number}/{len(recipients)}')
        prefix, mask, valid = observations[int(i)]
        source_row = source_rows.get(calls[i]['evidence_group_id'])
        ns = len(sources)-(source_row is not None)
        if ns == 0:
            entries.append((int(i), dict(family=ids[f], status='no_independent_training_unit', source_units=0))); continue
        train = total if source_row is None else total-profiles[source_row]
        train_valid = good if source_row is None else (invalid-(~allowed[source_row])) == 0
        informative = int(np.sum(mask & (np.diff(prefix) != 0)))
        if not informative:
            entries.append((int(i), dict(family=ids[f], status='no_recipient_information', source_units=ns))); continue
        if ctx.family_model == 'shared_interval':
            scored = profile_comparison(train, prefix[bb]-prefix[aa], aa, bb,
                                        recipient_allowed=valid, training_allowed=train_valid)
        else:
            fold = fold_by_group.get(calls[i]['evidence_group_id'], 'full')
            fit = distributions[fold]
            if ctx.edge_tolerance_mode == 'bounded':
                options = dict(allowed=valid, boundary_cells=boundary_cells,
                               edge_tolerance_bp=ctx.minimum_edge_tolerance_bp)
            else:
                options = dict(allowed=valid,
                    relax_left=ctx.minimum_edge_tolerance_bp > 0 and abs(spans[i, 0]-centers[f, 0]) <= ctx.minimum_edge_tolerance_bp,
                    relax_right=ctx.minimum_edge_tolerance_bp > 0 and abs(spans[i, 1]-centers[f, 1]) <= ctx.minimum_edge_tolerance_bp)
            if ctx.predictive_replicates:
                unit = units[calls[i]['unit_index']]
                native_positions, _, _, native_pa, native_pp = ctx.native_arrays(calls[i]['unit_index'])
                native_index = np.searchsorted(native_positions, positions[mask])
                pa = np.zeros(k); pp = np.zeros(k)
                pa[mask] = native_pa[native_index]
                pp[mask] = native_pp[native_index]
                seed = int.from_bytes(hashlib.sha256(f'{ids[f]}|{calls[i]["unit_id"]}|{calls[i]["ordinal"]}|predictive-v1'.encode()).digest()[:4], 'little')
                scored = predictive_reference(fit['log_density'], fit['log_mass'], prefix[bb]-prefix[aa], aa, bb,
                    observed=mask, p_accessible=pa, p_protected=pp,
                    replicates=ctx.predictive_replicates, seed=seed,
                    _defer_simulation=defer_native_simulations(),
                    **(dict(_prepared_shape_penalty=bounded_penalties[fold])
                       if ctx.edge_tolerance_mode == 'bounded' else {}), **options)
            else:
                scored = distribution_comparison(fit['log_density'], prefix[bb]-prefix[aa], aa, bb, **options)
            scored.update(status='scored', scoring_fold=str(fold),
                          fit_converged=fit['converged'], model_shape_penalty=True)
            if ctx.edge_tolerance_mode == 'bounded':
                scored.update(edge_tolerance_mode='bounded',
                              edge_tolerance_bp=ctx.minimum_edge_tolerance_bp,
                              edge_tolerance_semantics='bounded_joint_endpoint_cell_profile_v1',
                              edge_tolerance_changes_generative_mass=False)
            ns = fit['source_units']
        if scored['native_loss'] is None:
            entries.append((int(i), dict(family=ids[f], status=scored['status'], source_units=ns))); continue
        # The veto is separate, not another term added to the profile loss.
        if ctx.family_model == 'latent_distribution':
            # This adequacy geometry is from the excluded-fold model, not
            # the full-data display fit containing the recipient.
            check_geometry = fit['geometry_summary']['mean']
        else:
            train_best = int(np.flatnonzero(train_valid)[np.argmax(train[train_valid])])
            check_geometry = _projection_coordinates(positions, aa[train_best], bb[train_best], domain_lo, domain_hi, centers[f])[0]
        if ctx.family_model == 'latent_distribution':
            core = mask & (fit['geometry_coverage'] >= .5) & (positions >= spans[i, 0]) & (positions < spans[i, 1])
            core_lr = float(np.diff(prefix)[core].sum()); core_n = int(core.sum())
        else:
            left, right = max(spans[i, 0], check_geometry[0]), min(spans[i, 1], check_geometry[1])
            ca, cb = np.searchsorted(positions, [left, max(left, right)])
            core_lr = float(prefix[cb]-prefix[ca]); core_n = int(mask[ca:cb].sum())
        bad = core_n > 0 and core_lr < -math.log(ctx.core_contradiction_odds)
        adjusted = scored.get('floor_adjusted_loss')
        if adjusted is None:
            adjusted = float(edge_floor_loss(scored['native_loss'], scored['left_edge_relaxed_loss'],
                              scored['right_edge_relaxed_loss'], abs(spans[i, 0]-centers[f, 0]),
                              abs(spans[i, 1]-centers[f, 1]), ctx.minimum_edge_tolerance_bp))
        # This labels an EXISTING call, not a newly rescued footprint.
        # Its full-domain observations determine shape compatibility.
        # A distribution's mean interval need not contain a recipient
        # opportunity: absence there is a diagnostic, not a second gate.
        # In particular NEVER overwrite the statistic after simulating
        # its reference distribution. Core contradiction is an independent
        # veto below; it does not change the score that was simulated.
        entries.append((int(i), dict(family=ids[f], status='core_contradicted' if bad else 'scored',
            **{key: value for key, value in scored.items() if key not in ('status', 'floor_adjusted_loss')},
            floor_adjusted_loss=adjusted, source_units=ns,
            recipient_informative_opportunities=informative, core_opportunities=core_n,
            mean_core_unobserved=core_n == 0,
            core_geometry_mean=list(check_geometry),
            core_geometry_summary='normalized_positive_cell_coverage' if ctx.family_model == 'latent_distribution' else 'shared_interval',
            core_minimum_geometry_coverage=.5 if ctx.family_model == 'latent_distribution' else None,
            minimum_edge_floor_applied=adjusted < scored['native_loss']-1e-10,
            core_protected_vs_accessible_log_lr=core_lr,
            geometry_distance_sq=int(np.square(centers[f]-spans[i]).sum()),
            own_evidence_excluded=source_row is not None)))
        if '_native_predictive_request' in entries[-1][1]:
            pending_predictions.append(entries[-1][1])
    flush_predictions()
    report_work(ctx.progress,f'{ids[f]}: {len(sources)} independent sources, {len(recipients)} recipients, {ng} native projections',
        completed=number+1,total=len(ids),unit='families')
    return model, entries


_FAMILY_WORKER = {}
from .execution import register_worker_state as _register_worker_state
_register_worker_state(_FAMILY_WORKER)
_FAMILY_UNIT_FIELDS = ('unit_id', 'strand', 'positions', 'hits', 'p_accessible', 'p_protected',
                       'representative_raw_tf_intervals', 'raw_nuc_intervals')


def _family_worker_payload(ctx):
    units = []
    for u in ctx.units:
        unit = {k: u[k] for k in _FAMILY_UNIT_FIELDS if k in u}
        unit['positions'] = np.asarray(u['positions'], dtype=np.int64)
        unit['hits'] = np.asarray(u['hits'])
        unit['p_accessible'] = np.asarray(u['p_accessible'], float)
        unit['p_protected'] = np.asarray(u['p_protected'], float)
        units.append(unit)
    return dict(calls=ctx.calls, units=units, spans=ctx.spans, centers=ctx.centers, ids=ctx.ids, homes=ctx.homes,
                training_evidence_groups=ctx.training_evidence_groups, retry_fit_iterations=ctx.retry_fit_iterations,
                active=ctx.active, region=ctx.region, minimum_edge_tolerance_bp=ctx.minimum_edge_tolerance_bp,
                edge_tolerance_mode=ctx.edge_tolerance_mode,
                core_contradiction_odds=ctx.core_contradiction_odds, maximum_matrix_bytes=ctx.maximum_matrix_bytes,
                family_model=ctx.family_model, max_fit_iterations=ctx.max_fit_iterations,
                predictive_replicates=ctx.predictive_replicates, scoring_folds=ctx.scoring_folds,
                fit_cache_dir=_current_fit_cache_dir(), fit_backend=_current_fit_backend(),
                nonparametric_pseudo_units=_current_fit_smoothing(),
                predictive_kernel=_current_predictive_kernel())


def _current_fit_cache_dir():
    from .fit_execution import _default_cache_dir
    return _default_cache_dir.get()


def _current_fit_backend():
    from .fit_execution import _default_fit_backend
    return _default_fit_backend.get()


def _current_fit_smoothing():
    from .fit_execution import _default_fit_smoothing
    return _default_fit_smoothing.get()


def _current_predictive_kernel():
    from .measurement_distribution import current_predictive_kernel
    return current_predictive_kernel()


def _load_family_worker(path):
    import pickle
    with open(path, 'rb') as handle:
        state = pickle.load(handle)
    from .fit_execution import set_default_fit_cache, set_default_fit_backend, set_default_fit_smoothing
    set_default_fit_cache(state.pop('fit_cache_dir', ''))
    set_default_fit_backend(state.pop('fit_backend', 'cpu'))
    set_default_fit_smoothing(state.pop('nonparametric_pseudo_units', 4.))
    from .measurement_distribution import set_default_predictive_kernel
    set_default_predictive_kernel(*state.pop('predictive_kernel', ('reference', 0.)))
    state['native_arrays'] = _native_arrays_cache(state['units'], state['maximum_matrix_bytes'])
    state['progress'] = None
    return dict(ctx=SimpleNamespace(**state))


def _family_task(path, f, number, total):
    from .execution import task_thread_budget, load_worker_state
    task_thread_budget(1)
    state = load_worker_state(_FAMILY_WORKER, path, _load_family_worker)
    model, entries = _classify_one_family(state['ctx'], f, number, total)
    return f, model, entries


def _classify_families_in_processes(ctx, pending, total, workers, progress):
    """Whole families in single-threaded spawned workers; merged in family order.

    Each worker runs the unchanged fit and scoring kernels serially with
    single-threaded BLAS, so every model, fold, score and predictive draw is
    identical to the historical in-process result. Only scheduling changes.
    """
    import pickle, tempfile
    from concurrent.futures import FIRST_COMPLETED, wait
    from pathlib import Path
    from .execution import stage_executor
    from .progress import report_work
    outcomes = {}
    with tempfile.TemporaryDirectory(prefix='fiberhmm-native-families-') as directory:
        path = str(Path(directory)/'families.pkl')
        with open(path, 'wb') as handle:
            pickle.dump(_family_worker_payload(ctx), handle, protocol=pickle.HIGHEST_PROTOCOL)
        executor, release = stage_executor(workers)
        # Largest expected workloads first reduce the tail; results are merged
        # by the caller in sorted family order regardless of completion order.
        spans, centers, active = ctx.spans, ctx.centers, ctx.active
        def weight(item):
            number, f = item
            return -int(np.sum(active & (spans[:, 0] < centers[f, 1]) & (spans[:, 1] > centers[f, 0])))
        queue = sorted(pending, key=weight)
        pending_futures = {}; remaining = iter(queue); completed = 0
        def submit():
            item = next(remaining, None)
            if item is not None:
                number, f = item
                pending_futures[executor.submit(_family_task, path, f, number, total)] = f
        try:
            for _ in range(workers+2):
                submit()
            while pending_futures:
                done, _ = wait(pending_futures, timeout=.2, return_when=FIRST_COMPLETED)
                for future in done:
                    pending_futures.pop(future)
                    f, model, entries = future.result()
                    outcomes[f] = (model, entries); completed += 1
                    submit()
                report_work(progress,f'native families: {completed}/{len(pending)} fitted and scored ({workers} workers)',
                    completed=completed,total=total,unit='families')
        except BaseException:
            for future in pending_futures:
                future.cancel()
            release(failed=True)
            raise
        release()
    if len(outcomes) != len(pending):
        raise RuntimeError('Family classification finished without every result')
    return outcomes
