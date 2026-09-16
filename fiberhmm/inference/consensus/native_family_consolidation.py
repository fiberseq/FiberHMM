"""Explicit, training-only consolidation of named native family hypotheses.

Original calls, child models, and their evidence remain immutable. A parent is
fitted once through the union of fixed primary source cohorts, not averaged
from child parameters. This is a caller-conditioned comparison model, not an
OOF assignment, recurrence certificate, automated merge, or occupancy model.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import math

import numpy as np

from .artifacts import digest
from .measurement_distribution import fit_native_distribution
from .measurement_family import _native_values
from .measurement_geometry import summarize_boundary_cells
from .native_presentation import classify_proposal


SCHEMA = 'native_explicit_family_consolidation_v1'


def _grid(positions, domain):
    lo, hi = map(int, domain)
    p = np.unique(np.asarray(positions, np.int64))
    p = p[(p >= lo) & (p < hi)]
    if not len(p):
        raise ValueError('No observed opportunities in the common domain')
    a, b = np.triu_indices(len(p) + 1, 1)
    ll, lh = np.r_[lo, p + 1][a], p[a]
    rl, rh = p[b - 1] + 1, np.r_[p, hi][b]
    return dict(positions=p, starts=a, ends=b, left_hi=lh, right_lo=rl,
                coordinates=np.c_[(ll + lh) / 2., (rl + rh) / 2.],
                areas=(lh - ll + 1.) * (rh - rl + 1.), domain=[lo, hi])


def _grid_digest(grid):
    return hashlib.sha256(np.c_[grid['coordinates'], grid['areas']].astype('<f8').tobytes()).hexdigest()


def _call_record(call, index, dataset, family=None):
    result = dict(source_call_index=int(index), unit_id=call['unit_id'],
        evidence_group_id=call.get('evidence_group_id', call['unit_id']),
        ordinal=int(call['ordinal']), interval=[int(call['start']), int(call['end'])],
        strand=call['strand'], source_call_id=f"{dataset}:{call['unit_id']}:{call['ordinal']}")
    if family is not None:
        result['original_primary_family'] = family
    return result


def consolidate_native_families(stratum, native_result, family_ids, *, parent_family_id,
                               region, reference_percent=99.9,
                               maximum_matrix_bytes=512 * 1024**2,
                               max_fit_iterations=100):
    """Return a separate frozen ``{model, grid, geometry_summary, ...}`` object.

    Source identities are reprojected from the frozen native candidate ledger
    at the explicitly recorded reference. Its data/receipt binding must be
    authenticated by the caller; this function cannot prove historical file
    provenance from an in-memory result. It binds exactly the supplied result
    and the observations actually used in this fit in its own output.

    The parameter reference is the coordinate-wise median of each evidence
    group's median original span. One candidate per group is then selected by
    squared full-span distance to that fixed reference, with coordinate/ID
    ties. No likelihood or child frequency enters duplicate selection.

    The common domain is the union of child caller domains within ``region``.
    Its lattice is the union of actual observations on original calls that
    overlap any child reference or are selected primary source candidates.
    Raw neighbours are conditioned exactly as in the existing native fitter.
    Each source likelihood is normalized over its own admissible geometries.
    No source or projection is downsampled to satisfy a resource budget.
    """
    ids = sorted(family_ids)
    dataset = stratum['dataset_id']
    if (len(ids) < 2 or len(set(ids)) != len(ids)
            or any(not isinstance(f, str) or not f.startswith(dataset + ':') for f in ids)):
        raise ValueError('At least two distinct family IDs from one dataset are required')
    if (not isinstance(parent_family_id, str) or not parent_family_id.startswith(dataset + ':')
            or not parent_family_id.split(':', 1)[1]):
        raise ValueError('An explicit parent family ID in the same dataset is required')
    if (len(region) != 2 or any(isinstance(v, bool) or not isinstance(v, (int, np.integer)) for v in region)
            or region[0] >= region[1]):
        raise ValueError('A positive integer-coordinate region is required')
    if (isinstance(reference_percent, bool) or not math.isfinite(reference_percent)
            or not 50 <= reference_percent <= 99.999):
        raise ValueError('Invalid classification reference')
    for name, value in [('matrix budget', maximum_matrix_bytes), ('fit iteration budget', max_fit_iterations)]:
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f'Positive integer {name} required')
    models = {m['family']: m for m in native_result['family_models']}
    if len(models) != len(native_result['family_models']):
        raise ValueError('Duplicate native model IDs')
    if parent_family_id in models:
        raise ValueError('The parent must not overwrite an existing native family')
    if any(f not in models for f in ids):
        raise ValueError('Unknown requested native family')
    children = [models[f] for f in ids]
    if any(m.get('status') != 'fitted' or m.get('family_model') != 'latent_distribution'
           or 'full' not in m.get('fold_models', {}) for m in children):
        raise ValueError('Consolidation requires fitted native boundary-distribution children')
    if any(len(m['domain']) != 2 or m['domain'][0] >= m['domain'][1]
           or m['reference_interval'][0] >= m['reference_interval'][1] for m in children):
        raise ValueError('Invalid child domain or reference interval')
    units = {u['unit_id']: u for u in stratum['units']}
    if len(units) != len(stratum['units']):
        raise ValueError('Duplicate evidence unit IDs')
    calls, evidence = native_result['calls'], native_result['call_family_evidence']
    if len(calls) != len(evidence):
        raise ValueError('Native call and evidence axes differ')
    expected = {(uid, j): tuple(span) for uid, u in units.items()
                for j, span in enumerate(u['representative_raw_tf_intervals'])}
    seen = set()
    primary = {}
    for i, (call, scores) in enumerate(zip(calls, evidence)):
        key = call['unit_id'], call['ordinal']
        if key in seen or key not in expected or expected[key] != (call['start'], call['end']):
            raise ValueError('Original call identity/span mismatch or duplicate')
        seen.add(key)
        unit = units[call['unit_id']]
        if (call.get('evidence_group_id', call['unit_id']) != unit.get('fold_group_id', unit['unit_id'])
                or call['strand'] != unit['strand']):
            raise ValueError('Native call evidence-group/strand identity mismatch')
        if call['start'] >= region[1] or call['end'] <= region[0]:
            continue
        selection = classify_proposal(dict(candidate_evidence=scores, unresolved_family='unresolved'), reference_percent)
        if selection['primary_evidence'] is not None and selection['family'] in ids:
            primary[i] = selection['family']
    if seen != set(expected):
        raise ValueError('Incomplete original call axis')
    if not primary:
        raise ValueError('No primary source calls for the requested family union')
    groups = defaultdict(list)
    for i in primary:
        groups[calls[i].get('evidence_group_id', calls[i]['unit_id'])].append(i)
    group_medians = [np.median([[calls[i]['start'], calls[i]['end']] for i in groups[g]], axis=0)
                     for g in sorted(groups)]
    reference = np.median(group_medians, axis=0)
    chosen, duplicates = [], []
    for group in sorted(groups):
        def key(i):
            c = calls[i]
            return (float(np.square(np.array([c['start'], c['end']]) - reference).sum()),
                    c['start'], c['end'], c['unit_id'], c['ordinal'])
        indices = sorted(groups[group], key=key)
        chosen.append(indices[0])
        duplicates.extend(_call_record(calls[i], i, dataset, primary[i]) |
                          dict(reason='duplicate_evidence_group', selected_source_call_id=
                               _call_record(calls[indices[0]], indices[0], dataset)['source_call_id']) for i in indices[1:])
    domain = [max(int(region[0]), min(int(m['domain'][0]) for m in children)),
              min(int(region[1]), max(int(m['domain'][1]) for m in children))]
    if domain[0] >= domain[1]:
        raise ValueError('Child domains do not overlap the requested analysis region')
    references = [m['reference_interval'] for m in children]
    recipients = [i for i, c in enumerate(calls)
                  if c['start'] < domain[1] and c['end'] > domain[0]
                  and (i in primary or any(c['start'] < b and c['end'] > a for a, b in references))]
    if any(i not in recipients for i in chosen):
        raise ValueError('A primary source lies outside the common child domain')
    grid_units = sorted({calls[i]['unit_id'] for i in recipients})
    positions = np.unique(np.concatenate([np.asarray(units[uid]['positions'], np.int64) for uid in grid_units]))
    positions = positions[(positions >= domain[0]) & (positions < domain[1])]
    if not len(positions):
        raise ValueError('No observed opportunities in the common domain')
    ng = len(positions) * (len(positions) + 1) // 2
    # Include likelihood/mask, cached exponentials/exposure and optimization
    # fallback work arrays. Python input objects are additional process memory.
    estimate = int(ng * (len(chosen) * 80 + 192) + len(positions) * len(chosen) * 24)
    if estimate > maximum_matrix_bytes:
        raise MemoryError(f'Exact parent fit needs approximately {estimate} matrix bytes; '
                          f'budget {maximum_matrix_bytes}; no sources/projections removed')
    grid = _grid(positions, domain)
    aa, bb = grid['starts'], grid['ends']
    profiles, allowed, included, excluded, row_details = [], [], [], [], []
    for i in chosen:
        call = calls[i]; unit = units[call['unit_id']]
        values, observed = _native_values(unit, grid['positions'], call)
        prefix = np.r_[0., np.cumsum(values)]
        ca, cb = np.searchsorted(positions, [call['start'], call['end']])
        valid = (aa < cb) & (bb > ca)
        previous, following = domain
        for x, y in list(unit['representative_raw_tf_intervals']) + list(unit.get('raw_nuc_intervals', [])):
            if y <= call['start']: previous = max(previous, y)
            if x >= call['end']: following = min(following, x)
        valid &= (grid['left_hi'] >= previous) & (grid['right_lo'] <= following)
        record = _call_record(call, i, dataset, primary[i])
        if not valid.any() or not np.any(observed & (values != 0.)):
            excluded.append(record | dict(reason='no_admissible_projection' if not valid.any() else 'no_native_information'))
            continue
        profiles.append(prefix[bb] - prefix[aa]); allowed.append(valid); included.append(i)
        row_details.append(record | dict(observed_opportunities=int(observed.sum()),
            informative_opportunities=int(np.sum(observed & (values != 0.))),
            admissible_projections=int(valid.sum()), frozen_neighbor_limits=[previous, following]))
    if not included:
        raise ValueError('No informative admissible source remains after the declared geometry-only selection')
    profiles, allowed = np.asarray(profiles), np.asarray(allowed)
    fit = fit_native_distribution(profiles, allowed, grid['coordinates'], grid['areas'],
                                  reference=reference, max_iterations=int(max_fit_iterations))
    left_lo = np.r_[domain[0], positions + 1][aa]
    right_hi = np.r_[positions, domain[1]][bb]
    summary = summarize_boundary_cells(fit['log_mass'], left_lo, grid['left_hi'], grid['right_lo'], right_hi)
    grid_hash = _grid_digest(grid)
    included_groups = [calls[i].get('evidence_group_id', calls[i]['unit_id']) for i in included]
    model = dict(family=parent_family_id, status='fitted', family_model='latent_distribution',
        source_units=len(included), reference_interval=reference.tolist(), domain=domain,
        identifiable_projections=ng, source_call_indices=included,
        fitted_distribution_center=fit['center'].tolist(), fitted_distribution_covariance=fit['covariance'].tolist(),
        normalized_geometry=summary, parameters_leave_evidence_group_out=False, proposal_discovery_out_of_fold=False,
        explicit_projection_grid_required=True, model_role='explicit_consolidated_parent_comparison',
        child_family_ids=ids, scoring_folds=0,
        fold_models={'full': dict(parameters=fit['parameters'].tolist(), center=fit['center'].tolist(),
            covariance=fit['covariance'].tolist(), normalized_geometry=summary,
            parameter_coordinate_scale_bp=10., parameter_reference=reference.tolist(),
            projection_grid_sha256=grid_hash)},
        fit_diagnostics={'full': {k: fit[k] for k in ('objective', 'converged', 'iterations', 'message', 'source_units')}},
        training_evidence_groups={'full': included_groups})
    observations = []
    for i in included:
        c = calls[i]; u = units[c['unit_id']]; p = np.asarray(u['positions'])
        keep = (p >= domain[0]) & (p < domain[1])
        observations.append(dict(source_call=_call_record(c, i, dataset),
            positions=p[keep], hits=np.asarray(u['hits'])[keep],
            p_accessible=np.asarray(u['p_accessible'])[keep], p_protected=np.asarray(u['p_protected'])[keep],
            original_tf_intervals=u['representative_raw_tf_intervals'], original_nuc_intervals=u.get('raw_nuc_intervals', [])))
    output = dict(schema=SCHEMA, status='complete', model=model, grid=grid, geometry_summary=summary,
        source_calls=row_details, excluded_duplicate_calls=duplicates, excluded_source_calls=excluded,
        provenance=dict(dataset_id=dataset, chemistry=stratum.get('chemistry'),
            native_snapshot_digest=digest(native_result), child_model_digests={m['family']: digest(m) for m in children},
            classification_reference_percent=float(reference_percent),
            primary_calls_by_child=dict(Counter(primary.values())), primary_union_calls=len(primary),
            primary_union_evidence_groups=len(groups), fit_sources_by_child=dict(Counter(primary[i] for i in included)),
            primary_source_call_ids=[_call_record(calls[i], i, dataset)['source_call_id'] for i in sorted(primary)],
            source_selection='group-balanced raw full-span median; one call/group by squared distance, then start/end/unit/ordinal',
            original_spans_preserved=True, original_catalog_unchanged=True, old_models_or_scores_averaged=False,
            parent_per_molecule_assignments_published=False, fit_uses_all_selected_informative_groups_once=True,
            historical_observation_binding_requires_caller_verification=True,
            common_domain_rule='union of child native caller domains, intersected with explicit analysis region',
            grid_recipient_source_call_indices=recipients, grid_unit_ids=grid_units,
            likelihood_domain_sha256=digest(observations), projection_grid_sha256=grid_hash,
            likelihood_profiles_sha256=hashlib.sha256(profiles.astype('<f8').tobytes()).hexdigest(),
            allowed_projections_sha256=hashlib.sha256(allowed.tobytes()).hexdigest(),
            neighbor_conditioning='existing _native_values background and native fitter whole-cell admissibility',
            maximum_matrix_bytes=int(maximum_matrix_bytes), estimated_matrix_bytes=estimate,
            max_fit_iterations=int(max_fit_iterations), no_projection_or_source_subsampling=True))
    output['artifact_sha256'] = digest(output)
    return output


def restore_consolidated_geometry(artifact):
    """Validate portable parent/grid integrity and restore native array types.

    Pass this frozen object directly to native transfer. Do not reconstruct a
    grid from the parent's new reference interval and an inferred recipient set.
    """
    if artifact.get('schema') != SCHEMA:
        raise ValueError('Unknown consolidation schema')
    unsigned = {k: v for k, v in artifact.items() if k != 'artifact_sha256'}
    if digest(unsigned) != artifact.get('artifact_sha256'):
        raise ValueError('Consolidated artifact digest mismatch')
    model, stored = artifact['model'], artifact['grid']
    grid = _grid(stored['positions'], stored['domain'])
    for key in grid:
        if not np.array_equal(grid[key], stored[key]):
            raise ValueError(f'Consolidated grid mismatch: {key}')
    expected = model['fold_models']['full']['projection_grid_sha256']
    if _grid_digest(grid) != expected or expected != artifact['provenance']['projection_grid_sha256']:
        raise ValueError('Consolidated native projection digest mismatch')
    return dict(model=model, grid=grid, geometry_summary=artifact['geometry_summary'])
