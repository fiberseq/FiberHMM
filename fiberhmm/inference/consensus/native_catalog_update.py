"""Pure, authenticated append composition for native family hypotheses.

No fitting, nomination, thresholds, emissions, source geometry or production
defaults change here. Bindings must be made by the producer at completion. For
legacy files, callers must FIRST authenticate their receipts/file hashes and
reconstruct the actual producer inputs/options. Post-hoc binding alone is not
evidence of historical provenance.

Alternative models may share observations. Their own source indices/folds are
authoritative; there is deliberately no replacement global source-home map.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
from numbers import Integral, Real
from typing import Any, Mapping

import numpy as np

from .native_presentation import classify_proposal

BINDING_SCHEMA = 'fiberhmm.native_catalog_binding.v1'
UPDATE_SCHEMA = 'fiberhmm.native_catalog_append.v1'
_DOWNSTREAM = frozenset(('cross', 'xcr', 'cross_graph', 'native_cross_graph',
                        'count_groups', 'rescue', 'auxiliary', 'split',
                        'comparability', 'curation', 'manual_groups'))
_REQUIRED_OPTIONS = frozenset(('family_model', 'minimum_edge_tolerance_bp',
    'core_contradiction_odds', 'max_fit_iterations', 'predictive_replicates',
    'scoring_folds', 'maximum_matrix_bytes', 'loss_odds_levels'))
_COMMON_BINDING = ('dataset_id', 'chemistry', 'region', 'unit_axis_sha256',
    'call_axis_sha256', 'observation_sha256', 'model_options',
    'implementation_contract', 'diagnostic_contract')
_PRESENTATION_DIAGNOSTICS = frozenset(('original_catalog_families',
    'residual_update_policy', 'loss_partitions_not_rebuilt',
    'models_share_training_observations', 'joint_mixture'))


@dataclass(frozen=True)
class NativeCatalogComposition:
    """Detached result and companions; no input container is returned by alias.

    With no added ID, ``result`` is exactly the initial result, including its
    existing presentation caches. ``report`` records that no expansion occurred.
    Bindings and model versions are separate to avoid altering original model
    or score payloads merely to annotate their provenance.
    """
    result: dict
    catalog: list
    binding: dict
    model_versions: dict
    report: dict


def _canonical(value):
    # Concrete-type dispatch first. ``json`` calls this adapter once per numpy
    # scalar in a large evidence tree, and every abstract-base ``isinstance``
    # costs an ABCMeta subclass check. Plain JSON types take the same branches
    # as the reference chain below, so the encoded bytes are unchanged.
    kind = type(value)
    if kind is str or kind is bool or kind is int or value is None:
        return value
    if kind is float:
        if math.isfinite(value):
            return value
        raise ValueError('Native provenance requires finite JSON-compatible values')
    if kind is dict:
        if any(not isinstance(k, str) for k in value):
            raise ValueError('Native provenance dictionaries require string keys')
        return {k: _canonical(v) for k, v in sorted(value.items())}
    if kind is list or kind is tuple:
        return [_canonical(v) for v in value]
    if isinstance(value, np.ndarray):
        return _canonical(value.tolist())
    if isinstance(value, np.generic):
        return _canonical(value.item())
    if isinstance(value, Mapping):
        if any(not isinstance(k, str) for k in value):
            raise ValueError('Native provenance dictionaries require string keys')
        return {k: _canonical(v) for k, v in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_canonical(v) for v in value]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise ValueError('Native provenance requires finite JSON-compatible values')


def _encoded(value):
    # Validate dictionary keys without copying/sorting the entire evidence
    # tree first. The C JSON encoder already sorts keys and rejects NaN/Inf;
    # _canonical remains the adapter for arrays/scalars/custom mappings.
    # Do not memoize mutable object identities: every binding is revalidated.
    def keys(root):
        # Iterative walk with concrete-type dispatch. Leaves are the bulk of a
        # 200 MB evidence tree; testing them against abstract bases cost more
        # than the C encoder itself. Behaviour is unchanged: only dict keys are
        # validated, only Mapping/list/tuple containers are descended.
        stack = [root]
        pop, push = stack.pop, stack.extend
        while stack:
            node = pop()
            kind = type(node)
            if kind is str or kind is int or kind is float or kind is bool or node is None:
                continue
            if kind is dict:
                for key in node:
                    if type(key) is not str and not isinstance(key, str):
                        raise ValueError('Native provenance dictionaries require string keys')
                push(node.values())
            elif kind is list or kind is tuple:
                push(node)
            elif isinstance(node, Mapping):
                for key in node:
                    if not isinstance(key, str):
                        raise ValueError('Native provenance dictionaries require string keys')
                push(node.values())
            elif isinstance(node, (list, tuple)):
                push(node)
    keys(value)
    try:
        return json.dumps(value, sort_keys=True, separators=(',', ':'),
                          allow_nan=False, default=_canonical).encode()
    except (TypeError, ValueError) as exc:
        raise ValueError('Native provenance requires finite JSON-compatible values') from exc


def _model_digest_task(item):
    fid, model = item
    return fid, (native_snapshot_digest(model),
                 {key: native_snapshot_digest(value) for key, value in model.get('fold_models', {}).items()},
                 {key: native_snapshot_digest(value) for key, value in model.get('training_evidence_groups', {}).items()})


def _model_digests_in_pool(items):
    """Per-model digests are pure functions of immutable models: compute them in
    the run's worker pool when one exists. Values are identical either way."""
    from .execution import _shared_pool
    pool = _shared_pool.get()
    if pool is None or len(items) < 8:
        return dict(_model_digest_task(item) for item in items)
    executor = pool.get()
    futures = [executor.submit(_model_digest_task, item) for item in items]
    return dict(f.result() for f in futures)


def native_snapshot_digest(value):
    """Deterministic value digest: equivalent list/array containers agree."""
    return hashlib.sha256(_encoded(value)).hexdigest()


def _integer(value, label, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f'{label} must be an integer >= {minimum}')
    return int(value)


def _sha256_string(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def _diagnostic_contract(diagnostics):
    return _canonical({k: v for k, v in diagnostics.items() if k not in _PRESENTATION_DIAGNOSTICS})


def _ids(rows, label, dataset_id):
    result = {}
    for row in rows:
        fid = row.get('family')
        if not isinstance(fid, str) or not fid.startswith(dataset_id+':'):
            raise ValueError(f'{label}: dataset-scoped family identity required')
        if fid in result:
            raise ValueError(f'{label}: duplicate family ID {fid}')
        result[fid] = row
    return result


def _region(region):
    if not isinstance(region, Mapping) or not isinstance(region.get('chrom'), str) or not region['chrom']:
        raise ValueError('Explicit chromosome and half-open analysis region required')
    start = _integer(region.get('start'), 'Region start')
    end = _integer(region.get('end'), 'Region end')
    if end <= start:
        raise ValueError('Positive-width analysis region required')
    return dict(chrom=region['chrom'], start=start, end=end)


def _source_axes(stratum):
    dataset_id = stratum.get('dataset_id')
    if not isinstance(dataset_id, str) or not dataset_id:
        raise ValueError('Explicit dataset ID required')
    chemistry = stratum.get('chemistry')
    if chemistry not in ('ddda', 'dddb', 'hia5-pacbio', 'hia5-nanopore'):
        raise ValueError('Explicit native chemistry required')
    units, calls, axis, seen = [], [], [], set()
    for m, original in enumerate(stratum['units']):
        fast = _fast_observation_axes(original)
        if fast is None:
            unit = _canonical(original)
        else:
            unit = _canonical({k: v for k, v in original.items() if k not in _OBSERVATION_FIELDS})
            unit.update(fast)
        uid = unit.get('unit_id')
        if not isinstance(uid, str) or not uid or uid in seen:
            raise ValueError('Unique nonempty source unit IDs required')
        seen.add(uid)
        group = unit.get('fold_group_id', uid)
        if not isinstance(group, str) or not group or not isinstance(unit.get('strand'), str):
            raise ValueError('Explicit evidence-group and strand identities required')
        if fast is None:
            pos = [_integer(v, 'Opportunity coordinate') for v in unit['positions']]
            hits = unit['hits']
            if any(v not in (0, 1) for v in hits):
                raise ValueError('Binary observed modifications required')
            pa, pp = unit['p_accessible'], unit['p_protected']
            if not len(pos) == len(hits) == len(pa) == len(pp) or any(a >= b for a, b in zip(pos, pos[1:])):
                raise ValueError('Same ordered observation/emission axes required')
            if any(not isinstance(v, Real) or isinstance(v, bool) or not 0 < v < 1 for v in [*pa, *pp]):
                raise ValueError('Strictly interior native emission probabilities required')
            unit.update(positions=pos, hits=[int(v) for v in hits],
                        p_accessible=list(map(float, pa)), p_protected=list(map(float, pp)))
        for ordinal, span in enumerate(unit['representative_raw_tf_intervals']):
            if len(span) != 2:
                raise ValueError('Original call needs two coordinates')
            a, b = (_integer(v, 'Call coordinate') for v in span)
            if a >= b:
                raise ValueError('Original calls must have positive width')
            calls.append(dict(unit_index=m, unit_id=uid, ordinal=ordinal, start=a,
                              end=b, strand=unit['strand'], evidence_group_id=group))
        axis.append(dict(unit_index=m, unit_id=uid, evidence_group_id=group,
                         strand=unit['strand'], read_name=unit.get('read_name')))
        units.append(unit)
    calls.sort(key=lambda c: (c['start'], c['end'], c['unit_id'], c['ordinal']))
    # All source fields, not only hits, are hashed: masks, aligned blocks,
    # qualities, frozen neighbors and producer metadata cannot drift unnoticed.
    complete = {k: _canonical(v) for k, v in stratum.items() if k != 'units'}
    complete['units'] = units
    return calls, axis, complete


_OBSERVATION_FIELDS = ('positions', 'hits', 'p_accessible', 'p_protected')


def _fast_observation_axes(unit):
    """Vectorized validation of the four large observation arrays, or None.

    Produces exactly the canonical Python ints/floats the element-wise path
    emits, with the same acceptance rules. Anything unusual (object dtypes,
    booleans, mixed types) returns None and the reference element-wise path
    validates and canonicalizes instead.
    """
    try:
        pos = np.asarray(unit['positions']); hits = np.asarray(unit['hits'])
        pa = np.asarray(unit['p_accessible']); pp = np.asarray(unit['p_protected'])
    except (KeyError, TypeError, ValueError):
        return None
    if (pos.ndim != 1 or pos.dtype.kind not in 'iu' or hits.ndim != 1 or hits.dtype.kind not in 'iu'
            or pa.ndim != 1 or pa.dtype.kind != 'f' or pp.ndim != 1 or pp.dtype.kind != 'f'):
        return None
    if np.any(pos < 0):
        raise ValueError('Opportunity coordinate must be an integer >= 0')
    if np.any((hits != 0) & (hits != 1)):
        raise ValueError('Binary observed modifications required')
    if not len(pos) == len(hits) == len(pa) == len(pp) or np.any(pos[1:] <= pos[:-1]):
        raise ValueError('Same ordered observation/emission axes required')
    if (np.any(~np.isfinite(pa)) or np.any(~np.isfinite(pp)) or np.any((pa <= 0) | (pa >= 1))
            or np.any((pp <= 0) | (pp >= 1))):
        raise ValueError('Strictly interior native emission probabilities required')
    return dict(positions=[int(v) for v in pos.tolist()], hits=[int(v) for v in hits.tolist()],
                p_accessible=[float(v) for v in pa.tolist()], p_protected=[float(v) for v in pp.tolist()])


def prepare_source_binding(stratum):
    """Immutable source axes and digests, computed once per stratum per workflow.

    The stratum's observations do not change between the initial and augmented
    bindings of one append workflow, so recomputing these axes and digests for
    each binding is pure repetition. Callers pass the result to
    ``bind_native_result``; passing a binding for a different stratum is caught
    by the dataset identity and call-axis checks.
    """
    calls, unit_axis, observations = _source_axes(stratum)
    return dict(dataset_id=stratum.get('dataset_id'), calls=calls, unit_count=len(unit_axis),
                unit_axis_sha256=native_snapshot_digest(unit_axis),
                call_axis_sha256=native_snapshot_digest(calls),
                observation_sha256=native_snapshot_digest(observations))


def _validate_options(options, diagnostics):
    if not isinstance(options, Mapping) or not _REQUIRED_OPTIONS <= options.keys():
        raise ValueError('Complete native fitter options, including numerical budgets, required')
    options = _canonical(options)
    mode = options.get('edge_tolerance_mode', 'legacy_profile')
    if mode not in ('legacy_profile', 'bounded') or mode != diagnostics.get('edge_tolerance_mode', 'legacy_profile'):
        raise ValueError('Native result disagrees with producer option edge_tolerance_mode')
    if options['family_model'] != 'latent_distribution':
        raise ValueError('Append presentation requires the native latent-distribution model')
    for key, minimum in [('minimum_edge_tolerance_bp', 0), ('max_fit_iterations', 1),
                         ('predictive_replicates', 1), ('scoring_folds', 2), ('maximum_matrix_bytes', 1)]:
        _integer(options[key], key, minimum)
    if 'membership_loss_odds' in options:
        value = options['membership_loss_odds']
        if isinstance(value, bool) or not isinstance(value, Real) or not value >= 1:
            raise ValueError('Native membership_loss_odds must be a number at least 1')
        if diagnostics.get('membership_loss_odds') != value:
            raise ValueError('Native result disagrees with producer option membership_loss_odds')
    elif diagnostics.get('membership_loss_odds', 1.) != 1.:
        raise ValueError('Native result declares tie-set membership that its producer options omit')
    if options.get('fit_backend', 'cpu') != diagnostics.get('fit_backend', 'cpu'):
        raise ValueError('Native result disagrees with producer option fit_backend')
    if options.get('fit_backend', 'cpu') == 'nonparametric':
        smoothing = options.get('nonparametric_pseudo_units')
        if isinstance(smoothing, bool) or not isinstance(smoothing, Real) or not smoothing >= 0:
            raise ValueError('Native nonparametric_pseudo_units must be a non-negative number')
        if diagnostics.get('nonparametric_pseudo_units') != smoothing:
            raise ValueError('Native result disagrees with producer option nonparametric_pseudo_units')
    odds = options['core_contradiction_odds']
    levels = options['loss_odds_levels']
    if (not isinstance(odds, Real) or isinstance(odds, bool) or odds < 1
            or not levels or any(not isinstance(v, Real) or isinstance(v, bool) or v < 1 for v in levels)):
        raise ValueError('Finite native loss/contradiction odds >=1 required')
    for key in ('family_model', 'minimum_edge_tolerance_bp', 'core_contradiction_odds',
                'predictive_replicates', 'scoring_folds'):
        if key not in diagnostics or diagnostics[key] != options[key]:
            raise ValueError(f'Native result disagrees with producer option {key}')
    return options


def _validate_payload(result, catalog, dataset_id, region):
    if result.get('status') != 'complete':
        raise ValueError('Only complete native fitted results may be composed')
    if _DOWNSTREAM.intersection(result):
        raise ValueError('Stale downstream XCR/auxiliary/curation payload must not enter native composition')
    models = _ids(result['family_models'], 'Models', dataset_id)
    proposals = _ids(catalog, 'Catalog', dataset_id)
    if set(models) != set(proposals):
        raise ValueError('Every proposal must retain exactly one fitted or unavailable model record')
    for fid, proposal in proposals.items():
        a = _integer(proposal.get('consensus_start'), 'Proposal start')
        b = _integer(proposal.get('consensus_end'), 'Proposal end')
        if a >= b:
            raise ValueError('Positive-width proposal required')
        if (models[fid].get('status') == 'fitted'
                and native_snapshot_digest(models[fid].get('reference_interval')) != native_snapshot_digest([a, b])):
            raise ValueError('Model reference geometry disagrees with its frozen proposal')
    calls, scores = result['calls'], result['call_family_evidence']
    if len(calls) != len(scores):
        raise ValueError('Missing candidate evidence row on the original call axis')
    model_groups = {}
    for fid, model in models.items():
        indices = model.get('source_call_indices', [])
        if len(set(indices)) != len(indices):
            raise ValueError('Duplicate model-specific source-call index')
        for i in indices:
            _integer(i, 'Model source index')
            if i >= len(calls):
                raise ValueError('Model source index outside the frozen call axis')
        groups = [calls[i]['evidence_group_id'] for i in indices]
        if len(set(groups)) != len(groups):
            raise ValueError('A model cannot count an evidence group more than once')
        model_groups[fid] = groups
        if model.get('status') != 'fitted':
            continue
        if not len(indices) or model.get('source_units') != len(indices):
            raise ValueError('Fitted model lacks complete model-specific source provenance')
        folds, training = model.get('fold_models', {}), model.get('training_evidence_groups', {})
        if 'full' not in folds or set(folds) != set(training):
            raise ValueError('Fitted model lacks its full/fold parameter dependencies')
        if set(training['full']) != set(groups):
            raise ValueError('Full model training groups disagree with its source indices')
        for fold, fit in folds.items():
            train = training[fold]
            if len(set(train)) != len(train) or not set(train) <= set(groups) or not train:
                raise ValueError('Invalid model-specific excluded-fold training groups')
            grid = fit.get('projection_grid_sha256')
            if not _sha256_string(grid):
                raise ValueError('Every native fold needs an explicit projection-grid digest')
            diagnostics = model.get('fit_diagnostics', {}).get(fold)
            if diagnostics is None or diagnostics.get('source_units') != len(train):
                raise ValueError('Fold source counts/fit diagnostics disagree')
    candidate_indices = {fid: set() for fid in models}
    for call_index, (call, evidence) in enumerate(zip(calls, scores)):
        indexed = _ids(evidence, 'Candidate evidence', dataset_id)
        if not set(indexed) <= set(models):
            raise ValueError('Candidate refers to an unknown model version')
        for fid, score in indexed.items():
            candidate_indices[fid].add(call_index)
            if score.get('status') not in ('scored', 'core_contradicted'):
                continue
            model = models[fid]
            fold = score.get('scoring_fold')
            train = model.get('training_evidence_groups', {}).get(fold)
            if model.get('status') != 'fitted' or train is None:
                raise ValueError('Scored candidate has no fitted fold-model dependency')
            if call['evidence_group_id'] in train:
                raise ValueError('Recipient evidence group occurs in its own scoring model')
            if score.get('source_units') != len(train):
                raise ValueError('Candidate source count disagrees with its scoring fold')
            if score.get('fit_converged') != model['fit_diagnostics'][fold]['converged']:
                raise ValueError('Candidate and scoring-fold convergence diagnostics disagree')
            tail = score.get('predictive_tail_interval')
            if tail is None or len(tail) != 2 or not 0 <= tail[0] <= tail[1] <= 1:
                raise ValueError('Scored native candidate needs valid predictive bounds')
            for name in ('floor_adjusted_loss', 'geometry_distance_sq'):
                value = score.get(name)
                if not isinstance(value, Real) or isinstance(value, bool) or not math.isfinite(value) or value < 0:
                    raise ValueError(f'Invalid native presentation score {name}')
    spans = np.asarray([[c['start'], c['end']] for c in calls], dtype=np.int64).reshape(-1, 2)
    active = (spans[:, 0] < region['end']) & (spans[:, 1] > region['start'])
    for fid, model in models.items():
        if model.get('status') == 'fitted':
            a, b = proposals[fid]['consensus_start'], proposals[fid]['consensus_end']
            expected = set(map(int, np.flatnonzero(active & (spans[:, 0] < b) & (spans[:, 1] > a))))
        else:
            expected = set()
        if candidate_indices[fid] != expected:
            raise ValueError('Missing or out-of-domain candidate evidence for a frozen native model')
    return models, proposals, model_groups


def _seal(binding):
    binding = {k: deepcopy(v) for k, v in binding.items() if k != 'snapshot_digest'}
    return dict(binding, snapshot_digest=native_snapshot_digest(binding))


def bind_native_result(result, *, stratum, catalog, region, model_options,
                       implementation_contract, nomination_parent_digest=None, source_binding=None):
    """Bind actual completed producer inputs; does not authenticate legacy files.

    ``model_options`` is the complete fitting keyword set without callbacks.
    ``implementation_contract`` is a nonempty source/version digest mapping.
    An augmented nomination binds to the initial ``snapshot_digest`` explicitly.
    All fields affecting the observation likelihood and call/fold axes are part
    of this descriptor, including masks/qualities present in the source units.
    """
    region = _region(region)
    if source_binding is None or source_binding.get('dataset_id') != stratum.get('dataset_id'):
        source_binding = prepare_source_binding(stratum)
    calls = source_binding['calls']
    if native_snapshot_digest(result['calls']) != native_snapshot_digest(calls):
        raise ValueError('Result call order, coordinates or unit/group identity differ from the source')
    _validate_payload(result, catalog, stratum['dataset_id'], region)
    diagnostics = result['diagnostics']
    options = _validate_options(model_options, diagnostics)
    active = sum(c['start'] < region['end'] and c['end'] > region['start'] for c in calls)
    if (diagnostics.get('source_units') != source_binding['unit_count'] or diagnostics.get('source_calls') != active
            or diagnostics.get('out_of_region_calls') != len(calls)-active
            or diagnostics.get('original_catalog_families') != len(catalog)):
        raise ValueError('Producer region/unit/catalog diagnostic axes disagree')
    if not isinstance(implementation_contract, Mapping) or not implementation_contract:
        raise ValueError('Explicit producer implementation contract required')
    if any(not _sha256_string(v) for v in implementation_contract.values()):
        raise ValueError('Implementation contract must name SHA256 version digests')
    if nomination_parent_digest is not None and not _sha256_string(nomination_parent_digest):
        raise ValueError('Explicit valid initial nomination snapshot digest required')
    return _seal(dict(schema=BINDING_SCHEMA, dataset_id=stratum['dataset_id'],
        chemistry=stratum['chemistry'], region=region,
        unit_axis_sha256=source_binding['unit_axis_sha256'], call_axis_sha256=source_binding['call_axis_sha256'],
        observation_sha256=source_binding['observation_sha256'], model_options=options,
        implementation_contract=_canonical(implementation_contract),
        diagnostic_contract=_diagnostic_contract(diagnostics),
        catalog_sha256=native_snapshot_digest(catalog), result_sha256=native_snapshot_digest(result),
        nomination_parent_digest=nomination_parent_digest,
        provenance_semantics='Producer binding; legacy files require independent receipt/file authentication before binding'))


def _verify_binding(binding, result, catalog):
    if binding.get('schema') != BINDING_SCHEMA or _seal(binding) != binding:
        raise ValueError('Invalid or stale native snapshot binding')
    if (binding['result_sha256'] != native_snapshot_digest(result)
            or binding['catalog_sha256'] != native_snapshot_digest(catalog)):
        raise ValueError('Model, evidence or proposal changed after its snapshot was bound')
    _validate_payload(result, catalog, binding['dataset_id'], binding['region'])


def _partition(dataset_id, calls, evidence, catalog, region, reference, membership_loss_odds=1.):
    """Same primary ordering and evidence gate as the production presentation."""
    by_id = {p['family']: p for p in catalog}
    assignments, members, primaries = [], {}, {}
    for i, (call, scores) in enumerate(zip(calls, evidence)):
        a, b = call['start'], call['end']
        if a >= region['end'] or b <= region['start']:
            continue
        unresolved = dataset_id+':unresolved_'+hashlib.sha256(f'{a}|{b}'.encode()).hexdigest()[:12]
        selected = classify_proposal(dict(candidate_evidence=scores, unresolved_family=unresolved),
                                     reference, membership_loss_odds)
        keys = ('family', 'classification_status', 'primary_evidence', 'compatible_alternatives')
        if 'member_families' in selected:
            keys += ('member_families',)
        # One-level copies: these records are serialized and compared, never
        # mutated, and the six partition twins per composition spent most of
        # their time deep-copying scalar call fields and score dictionaries.
        assignments.append(dict(call, interval=[a, b], **{key: (dict(selected[key]) if isinstance(selected[key], dict)
                                                              else list(selected[key]) if isinstance(selected[key], list)
                                                              else selected[key]) for key in keys}))
        primaries.setdefault(selected['family'], []).append(i)
        for member in selected.get('member_families', [selected['family']]):
            members.setdefault(member, []).append(i)
    entries = []
    for fid, indices in sorted(members.items()):
        unresolved = fid not in by_id
        interval = ([calls[indices[0]]['start'], calls[indices[0]]['end']] if unresolved
                    else [by_id[fid]['consensus_start'], by_id[fid]['consensus_end']])
        strands = sorted({calls[i]['strand'] for i in indices})
        entry = dict(family=fid, representative_interval=interval, source_calls=len(indices),
            calls_by_strand=dict(Counter(calls[i]['strand'] for i in indices)),
            units_by_strand={s: len({calls[i]['evidence_group_id'] for i in indices if calls[i]['strand'] == s}) for s in strands},
            member_indices=indices, provisional_unresolved=unresolved, provisional_singleton=len(indices) == 1,
            native_proposal_preserved=not unresolved)
        if membership_loss_odds > 1.:
            entry['primary_calls'] = len(primaries.get(fid, []))
            entry['primary_indices'] = primaries.get(fid, [])
        entries.append(entry)
    return dict(catalog=entries, assignments=assignments, loss_odds=None,
        predictive_reference_percent=float(reference), classes=len(entries),
        true_primary_families=sum(not p['provisional_unresolved'] for p in entries),
        unresolved_calls=sum(p['classification_status'] == 'provisional_unresolved' for p in assignments),
        ambiguous_calls=sum(bool(p['compatible_alternatives']) for p in assignments),
        all_source_calls_retained=True, boundaries_changed=False, family_profile_model=True,
        **(dict(membership_loss_odds=float(membership_loss_odds),
                membership_rule='primary_plus_ties_within_loss_margin',
                tie_set_member_calls=sum(len(a['member_families']) for a in assignments if 'member_families' in a))
           if membership_loss_odds > 1. else {}))


def _compatible(assignment):
    return ({assignment['family'], *assignment['compatible_alternatives']}
            if assignment['primary_evidence'] is not None else set())


def compose_append_frozen(initial, augmented, *, initial_binding, augmented_binding,
                          initial_catalog, augmented_catalog, reference_percents=(95., 99., 99.9)):
    """Keep initial model/score objects exactly; append only nominated new IDs.

    Refitted *old* augmented objects (including missing old recipient scores)
    are deliberately ignored. A distinct, viable new candidate may change an
    original primary; a missing/vetoed old candidate itself is never promoted.
    This function rejects stale downstream payloads rather than laundering an
    existing graph or manual state onto the changed catalog.
    """
    references = sorted(set(reference_percents))
    if not references:
        raise ValueError('At least one native presentation reference required')
    for value in references:
        classify_proposal(dict(candidate_evidence=[], unresolved_family='validation'), value)
    _verify_binding(initial_binding, initial, initial_catalog)
    _verify_binding(augmented_binding, augmented, augmented_catalog)
    for key in _COMMON_BINDING:
        if initial_binding[key] != augmented_binding[key]:
            raise ValueError(f'Append source/setting/implementation drift: {key}')
    if augmented_binding.get('nomination_parent_digest') != initial_binding['snapshot_digest']:
        raise ValueError('Augmented nomination has a stale or unidentified initial dependency')
    ds = initial_binding['dataset_id']
    old_models = _ids(initial['family_models'], 'Initial models', ds)
    aug_models = _ids(augmented['family_models'], 'Augmented models', ds)
    old_catalog = _ids(initial_catalog, 'Initial catalog', ds)
    aug_catalog = _ids(augmented_catalog, 'Augmented catalog', ds)
    if not old_models.keys() <= aug_models.keys():
        raise ValueError('Augmented catalog dropped an established family identity')
    for fid, proposal in old_catalog.items():
        if native_snapshot_digest(proposal) != native_snapshot_digest(aug_catalog[fid]):
            raise ValueError('Established proposal identity/geometry was overwritten')
    added = set(aug_models)-set(old_models)
    update = augmented.get('nomination_update', {})
    additions = _ids(update.get('additions', []), 'Nomination additions', ds)
    if (set(additions) != added or update.get('added_proposals', 0) != len(added)
            or update.get('initial_proposals', len(old_catalog)) != len(old_catalog)):
        raise ValueError('New model IDs lack exact nomination provenance')
    calls = initial['calls']
    call_lookup = {(c['unit_id'], c['ordinal']): c for c in calls}
    for fid, proposal in additions.items():
        if native_snapshot_digest(proposal) != native_snapshot_digest(aug_catalog[fid]):
            raise ValueError('New proposal disagrees with its nomination record')
        aliases = proposal.get('source_aliases', [])
        if not proposal.get('nomination_provenance') or not aliases:
            raise ValueError('New model has unidentified proposal provenance')
        alias_keys = set()
        for alias in aliases:
            key = (alias.get('unit_id'), alias.get('ordinal'))
            call = call_lookup.get(key)
            if (call is None or native_snapshot_digest(alias.get('interval')) != native_snapshot_digest([call['start'], call['end']])
                    or key in alias_keys):
                raise ValueError('Nomination source aliases differ from original calls')
            alias_keys.add(key)
    combined_models = [*initial['family_models'], *(m for m in augmented['family_models'] if m['family'] in added)]
    combined_evidence = [[*before, *(s for s in after if s['family'] in added)]
                         for before, after in zip(initial['call_family_evidence'], augmented['call_family_evidence'])]
    catalog = [*initial_catalog, *(p for p in augmented_catalog if p['family'] in added)]
    # One model digest plus a streaming call-index/score digest per family is a
    # compact dependency map for every candidate row, without mutating each score.
    score_hashes = {m['family']: hashlib.sha256() for m in combined_models}
    for index, row in enumerate(combined_evidence):
        for score in row:
            score_hashes[score['family']].update(_encoded([index, score])+b'\n')
    versions = {}
    model_digests = _model_digests_in_pool([(m['family'], m) for m in combined_models])
    for model in combined_models:
        fid = model['family']; binding = augmented_binding if fid in added else initial_binding
        indices = model.get('source_call_indices', [])
        source_ids = [f"{ds}:{calls[i]['unit_id']}:{calls[i]['ordinal']}" for i in indices]
        source_groups = [calls[i]['evidence_group_id'] for i in indices]
        raw_digest, fold_digests, group_digests = model_digests[fid]
        versions[fid] = dict(origin='augmented_new' if fid in added else 'initial_frozen',
            producer_snapshot_digest=binding['snapshot_digest'], model_sha256=raw_digest,
            model_version_sha256=native_snapshot_digest(dict(model=raw_digest,
                source_call_axis=binding['call_axis_sha256'], observations=binding['observation_sha256'],
                options=binding['model_options'], implementation=binding['implementation_contract'])),
            candidate_evidence_sha256=score_hashes[fid].hexdigest(),
            source_call_axis_sha256=binding['call_axis_sha256'], source_call_ids=source_ids,
            source_evidence_group_ids=source_groups,
            fold_model_sha256=fold_digests,
            training_groups_sha256=group_digests,
            source_selection='actual model-specific producer sources; not residual-only or mutually exclusive',
            candidate_dependency='family identifies this model version; scoring_fold identifies its frozen excluded-fold model')
    report = dict(schema=UPDATE_SCHEMA, policy='append_frozen', initial_snapshot_digest=initial_binding['snapshot_digest'],
        augmented_snapshot_digest=augmented_binding['snapshot_digest'], original_family_count=len(old_models),
        added_family_ids=sorted(added), added_family_count=len(added),
        added_candidate_records=sum(s['family'] in added for row in combined_evidence for s in row),
        original_models_exact=True, original_evidence_exact=True, added_models_and_evidence_exact=True,
        observations_changed=False, boundaries_changed=False, refit_performed_by_composer=False,
        models_may_share_training_observations=True, joint_mixture=False, calibrated_occupancy=False,
        downstream_requires_recomputation=bool(added), no_expansion=not added, references={})
    if not added:
        return NativeCatalogComposition(deepcopy(initial), deepcopy(initial_catalog),
                                        deepcopy(initial_binding), versions, report)
    partitions = {}
    for reference in references:
        odds = float(initial_binding.get('model_options', {}).get('membership_loss_odds', 1.))
        previous = _partition(ds, calls, initial['call_family_evidence'], initial_catalog,
                              initial_binding['region'], reference, odds)
        current = _partition(ds, calls, combined_evidence, catalog, initial_binding['region'], reference, odds)
        switches = new_calls = more_ambiguous = 0
        for before, after in zip(previous['assignments'], current['assignments']):
            if not _compatible(before) <= _compatible(after):
                raise ValueError('Append unexpectedly removed old compatible evidence')
            changed = before['primary_evidence'] is not None and before['family'] != after['family']
            if changed and after['family'] not in added:
                raise ValueError('An established primary switched to another old model')
            switches += changed
            new_calls += before['primary_evidence'] is None and after['primary_evidence'] is not None
            more_ambiguous += len(_compatible(after)) > len(_compatible(before))
        partitions[str(float(reference))] = current
        report['references'][str(float(reference))] = dict(primary_switches_to_new=switches,
            newly_classified_existing_calls=new_calls, compatible_set_growth_calls=more_ambiguous,
            old_compatible_subset=True, unresolved_calls=current['unresolved_calls'],
            ambiguous_calls=current['ambiguous_calls'])
    result = dict(status='complete', schema=UPDATE_SCHEMA, calls=deepcopy(calls),
        family_models=deepcopy(combined_models), call_family_evidence=deepcopy(combined_evidence),
        predictive_partitions=partitions, partitions={},
        initial_source_homes_historical=deepcopy(initial.get('source_homes', initial.get('initial_source_homes_historical'))),
        source_home_semantics='No replacement global partition; frozen source_call_indices per model are authoritative.',
        nomination_update=deepcopy(update),
        diagnostics=dict(deepcopy(initial['diagnostics']), original_catalog_families=len(catalog),
            residual_update_policy='append_frozen', loss_partitions_not_rebuilt=True,
            models_share_training_observations=True, joint_mixture=False),
        catalog_update=deepcopy(report), model_versions=deepcopy(versions))
    binding = _seal(dict(initial_binding, catalog_sha256=native_snapshot_digest(catalog),
        result_sha256=native_snapshot_digest(result), nomination_parent_digest=initial_binding['snapshot_digest'],
        diagnostic_contract=_diagnostic_contract(result['diagnostics'])))
    return NativeCatalogComposition(result, deepcopy(catalog), binding, versions, report)
