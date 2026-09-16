"""Explicit strict-recall/split auxiliaries beside frozen native-family CR.

The bounded legacy strict *detection* model is retained under its own name.
It neither discovers CR classes nor assigns/reassigns existing footprints.
Native-family membership supplies source provenance only; positive native
protection evidence, grouped source exclusion, physical exposure and diffuse
adequacy are independently required. An addition additionally passes an
independently trained native latent-distribution shape test with zero bp floor.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np

from .. import strand_boundary_normalization as sbn
from .adapter import read_adapter
from .artifacts import digest, write_json
from .geometry import physically_allowed, representative_geometries
from .lattice import RegionFamilyLattice
from .native_auxiliary_shape import NativeAuxiliaryShape, donor_mask, grouped_fold
from .native_presentation import browser_cr
from .observations import prepare_population
from .projection import bounded_projection
from .recall import (adequacy_mask, exposure, free_domains, projection_rectangles,
                     score_group, shape_record, source_geometry_model, source_recipe)
from .rescue_geometry import rescue_decision


AUXILIARY_MODEL = 'frozen_native_catalog_legacy_strict_detection_v1'
SOURCE_REFERENCE_PERCENT = 99.9


def grouped_source_tables(membership, eligible, data, recipient_strand, mode):
    """One donor per physical group/family, preferring a measured detection.

    Same-fold removal happens in source_recipe. All records of a physical group
    share that fold; CT/GA are never paired from observation similarity here.
    """
    source = donor_mask(data['strands'], recipient_strand, mode)
    member = np.zeros_like(membership, float)
    exposed = np.zeros_like(eligible, bool)
    groups = {}
    for m in np.flatnonzero(source):
        groups.setdefault(str(data['fold_group_ids'][m]), []).append(int(m))
    for indices in groups.values():
        for f in range(membership.shape[1]):
            valid = [m for m in indices if eligible[m, f]]
            if not valid:
                continue
            chosen = min(valid, key=lambda m: (-float(membership[m, f]), data['unit_ids'][m]))
            exposed[chosen, f] = True
            member[chosen, f] = membership[chosen, f]
    return member, exposed


def _source_detections(stratum, catalog, native, maximum_diffuse_odds):
    """A classification-compatible label alone is not positive rescue support."""
    presentation = browser_cr(stratum, catalog, native, SOURCE_REFERENCE_PERCENT)
    index = {f['family']: i for i, f in enumerate(catalog)}
    by_id = {u['unit_id']: i for i, u in enumerate(stratum['units'])}
    membership = np.zeros((len(by_id), len(index)))
    ledger = []
    for row in presentation['records']:
        m = by_id[row['unit_id']]
        unit = stratum['units'][m]
        read = read_adapter(unit)
        for call in row['proposals']:
            family = call['family']
            if family not in index:
                continue
            a, b = call['interval']
            qa, qb = np.searchsorted(read.positions, [a, b])
            evidence = sbn.endpoint_pattern_evidence(read, a, b)
            lr = evidence['protected_log_likelihood'] - evidence['accessible_log_likelihood']
            passed = (qb-qa >= 3 and lr > 0 and
                      evidence['protected_log_likelihood']-evidence['diffuse_log_likelihood'] >= -np.log(maximum_diffuse_odds))
            membership[m, index[family]] = max(membership[m, index[family]], float(passed))
            ledger.append(dict(unit_id=unit['unit_id'], family=family,
                source_call_id=call['source_call_id'], interval=call['interval'],
                opportunities=int(qb-qa), native_log_lr=float(lr),
                direct_native_detection=bool(passed), source_reference_percent=SOURCE_REFERENCE_PERCENT))
    return membership, presentation['records'], ledger


def _prepare(stratum, catalog, native, options, progress):
    start, end = stratum['units'][0]['_region']
    groups = [u.get('fold_group_id', u['unit_id']) for u in stratum['units']]
    data = dict(stratum_id=stratum.get('stratum_id'), dataset_id=stratum['dataset_id'],
        chemistry=stratum['chemistry'], units=stratum['units'], start=start, end=end,
        unit_ids=[u['unit_id'] for u in stratum['units']], fold_group_ids=groups,
        strands=np.asarray([u['strand'] for u in stratum['units']]),
        folds=np.asarray([grouped_fold(g) for g in groups]), progress=progress,
        rescue_options=options['rescue'])
    membership, records, ledger = _source_detections(stratum, catalog, native,
        options['rescue'].maximum_diffuse_odds)
    return dict(data=data, catalog=catalog, records=records, membership=membership,
                source_detection_ledger=ledger)


def _prepare_lattice(run, options):
    """Only the recall/comparability auxiliary needs a configuration lattice."""
    base = run['data']
    data = prepare_population(base, base['start'], base['end'], grid_bp=1, max_intervals=0,
        max_matrix_bytes=options['compute'].maximum_matrix_mb*1024**2)
    data.update({k: base[k] for k in ('strands', 'folds', 'progress', 'rescue_options')})
    if not len(data['grid_positions']):
        return dict(run, data=data, catalog=[], kernel=None, membership=np.zeros((len(data['units']), 0)),
                    eligible=np.zeros((len(data['units']), 0), bool),
                    unavailable=[dict(family=f['family'], reason='no_auxiliary_opportunity_projection') for f in run['catalog']])
    # Low-count native classes remain in CR. They cannot nominate a strict new
    # detection until the user's explicit source-support criterion is reached.
    # This is admission, not an arbitrary class-count/resource truncation.
    selected, unavailable = [], []
    for f, family in enumerate(run['catalog']):
        groups = {str(base['fold_group_ids'][m]) for m in np.flatnonzero(run['membership'][:, f])}
        if not options['comparability'].enabled and len(groups) < options['rescue'].minimum_source_units:
            unavailable.append(dict(family=family['family'], reason='insufficient_direct_native_source_groups', source_groups=len(groups)))
            continue
        center = [family['consensus_start'], family['consensus_end']]
        try:
            bounded_projection(data['grid_positions'], center, options['rescue'].proposal_edge_radius_bp)
        except ValueError as exc:
            if 'no visible nonempty opportunity projection' not in str(exc):
                raise
            unavailable.append(dict(family=family['family'], reason='no_auxiliary_opportunity_projection'))
            continue
        selected.append(f)
    catalog = [dict(run['catalog'][f], family_index=i) for i, f in enumerate(selected)]
    membership = run['membership'][:, selected]
    if not catalog:
        return dict(run, data=data, catalog=[], kernel=None, eligible=np.zeros_like(membership, bool),
                    membership=membership, unavailable=unavailable)
    # A bounded geometry is a named auxiliary *proposal detector*, not the
    # native family's fitted boundary distribution or a CR assignment engine.
    # Never change catalog IDs or substitute its MAP calls for native labels.
    centers = np.asarray([[f['consensus_start'], f['consensus_end']] for f in catalog], int)
    kernel = RegionFamilyLattice(data['grid_positions'], centers, options['rescue'].proposal_edge_radius_bp,
        maximum_nodes=options['compute'].maximum_nodes, maximum_edges=options['compute'].maximum_edges)
    data['family_ids'] = [f['family'] for f in catalog]
    coordinates = representative_geometries(kernel)
    prefix = np.c_[np.zeros(len(data['units'])), np.cumsum(data['observed'], axis=1)]
    opportunities = prefix[:, kernel.gb]-prefix[:, kernel.ga]
    eligible = np.zeros_like(membership, bool)
    for m, u in enumerate(data['units']):
        physical = physically_allowed(u, coordinates, opportunities[m], 3)
        eligible[m, np.unique(kernel.gf[physical])] = True
    return dict(run, data=data, kernel=kernel, catalog=catalog,
                membership=membership, eligible=eligible, unavailable=unavailable,
                # Fixed auxiliary configuration prior, not fitted CR activity.
                eta=np.zeros(len(catalog)))


def _strict_candidates(run, indices, eta, admitted, q, counts, candidates, options):
    """Existing strict geometry gates, without legacy CR reassignment."""
    data, kernel = run['data'], run['kernel']
    rects = projection_rectangles(kernel)
    coordinates = representative_geometries(kernel)
    centers = kernel.centers[kernel.gf]
    opt = options['rescue']
    ledger = []
    for begin in range(0, len(indices), options['compute'].batch_size):
        ix = indices[begin:begin+options['compute'].batch_size]
        if not len(ix):
            continue
        units = [data['units'][m] for m in ix]
        values = data['log_lr'][ix]
        prefix = np.c_[np.zeros(len(ix)), np.cumsum(values, axis=1)]
        observed = np.c_[np.zeros(len(ix)), np.cumsum(data['observed'][ix], axis=1)]
        llr = prefix[:, kernel.gb]-prefix[:, kernel.ga]
        opp = observed[:, kernel.gb]-observed[:, kernel.ga]
        fractions, coords, domains = [], [], []
        for unit in units:
            domain = free_domains(unit, unit['representative_raw_tf_intervals'])
            fraction, coordinate = exposure(rects, centers, domain)
            domains.append(domain); fractions.append(fraction); coords.append(coordinate)
        fractions = np.asarray(fractions)
        physical = (fractions > 0) & (opp >= 3)
        adjustment = np.log(np.maximum(fractions, 1e-300)) + np.log(q)-kernel.logq
        allowed = np.broadcast_to(admitted, (len(ix), kernel.f))
        mask = physical.copy()
        for j, unit in enumerate(units):
            mask[j], _ = adequacy_mask(read_adapter(unit), kernel, physical[j], llr[j], coords[j], opt.maximum_diffuse_odds)
        prior = kernel.evaluate(np.zeros_like(values), eta, allowed=allowed,
            geometry_allowed=physical, geometry_log_adjustment=adjustment)
        post = kernel.evaluate(values, eta, allowed=allowed,
            geometry_allowed=mask, geometry_log_adjustment=adjustment)
        for j, unit in enumerate(units):
            calls = []
            for call in candidates.get(unit['unit_id'], []):
                f = call['family_index']
                evidence = shape_record(unit, call, f, kernel, rects, coordinates, q,
                    domains[j], counts[f], post['family_inclusion'][j, f])
                decision = rescue_decision(evidence, credible_mass=opt.credible_mass,
                    loss_odds=opt.loss_odds, minimum_probability=opt.minimum_probability,
                    minimum_source_units=opt.minimum_source_units)
                calls.append(dict(call, geometry_validation=evidence,
                    geometry_decisions=dict(selected=decision), auxiliary_model=AUXILIARY_MODEL))
            ledger.append(dict(unit_id=unit['unit_id'], strand=unit['strand'], calls=calls,
                whole_region_learned_geometry_log_predictive=float(post['log_partition'][j]-prior['log_partition'][j])))
    return ledger


def _recall(run, stratum, native, options, output, progress):
    data, kernel, opt = run['data'], run['kernel'], options['rescue']
    if kernel is None:
        return dict(status='no_supported_auxiliary_candidates', records=[],
                    unavailable=run['unavailable'], native_cr_records_unchanged=True)
    if opt.source_mode == 'opposite_strand' and not {'CT', 'GA'} <= set(data['strands']):
        return dict(status='not_applicable', records=[],
                    reason='No opposite biochemical strand; use pooled recall explicitly')
    if opt.model != 'legacy_strict':
        return dict(status='not_available_in_native_mode', records=[],
                    reason='Population-assisted rescue has a different generative model; choose explicit legacy_strict auxiliary detection')
    validator = NativeAuxiliaryShape(stratum, native, mode=opt.source_mode,
        maximum_matrix_bytes=options['compute'].maximum_matrix_mb*1024**2,
        max_iterations=options['cr'].family_fit_iterations,
        predictive_replicates=options['cr'].predictive_replicates,
        minimum_source_units=opt.minimum_source_units, progress=progress)
    originals = {r['unit_id']: r for r in run['records']}
    rng = np.random.default_rng(options['cr'].seed+1)
    accepted, ledger, recipes, nulls = [], [], [], []
    for strand in sorted(set(data['strands'])):
        members, eligible = grouped_source_tables(run['membership'], run['eligible'], data, strand, opt.source_mode)
        directory = output / ('source_geometry_' + str(strand))
        directory.mkdir(exist_ok=True)
        # Masked donors and one source per physical group/family. The existing
        # routine then excludes the entire held fold. It cannot pool a forbidden
        # donor strand or use recipient outcomes to train source geometry.
        q, counts = source_geometry_model(directory, data, kernel, members, eligible,
            run['eta'], originals, opt.geometry_prior_units)
        for fold in (0, 1):
            indices = np.flatnonzero((data['strands'] == strand) & (data['folds'] == fold))
            eta, admitted, recipe = source_recipe(members, eligible, data['strands'], data['folds'],
                strand, fold, opt.source_mode, opt.prior_scale, opt.minimum_source_units)
            source = donor_mask(data['strands'], strand, opt.source_mode) & (data['folds'] != fold)
            recipe.update(recipient_strand=strand, held_fold=fold,
                eligible_training_groups=sorted({str(data['fold_group_ids'][m]) for m in np.flatnonzero(source & eligible.any(1))}),
                auxiliary_model=AUXILIARY_MODEL)
            recipes.append(recipe)
            proposed, control, vetoes = score_group(kernel, data, indices, eta, admitted,
                projection_rectangles(kernel), [opt.minimum_protection_mass], rng, opt.null_replicates, 'posterior_accuracy')
            nulls.append(dict(strand=strand, fold=fold, counts=[dict(v) for v in control], vetoes=vetoes,
                              scope='initial auxiliary detection selection only; not full final-call FDR'))
            candidates = {r['unit_id']: r['calls'] for r in proposed}
            validated = _strict_candidates(run, indices, eta, admitted, q[fold], counts[fold], candidates, options)
            units = {u['unit_id']: u for u in data['units']}
            for row in validated:
                additions = []
                for ordinal, call in enumerate(row['calls']):
                    decision = call['geometry_decisions']['selected']
                    if decision['accepted'] and call['protection_event_mass'] >= opt.minimum_protection_mass:
                        native_shape = validator.validate(units[row['unit_id']], call['interval'], call['family'], ordinal=ordinal)
                        call['native_family_validation'] = native_shape
                        if native_shape['accepted']:
                            additions.append(dict(call, provenance='strict_auxiliary_new_call_native_shape_validated',
                                auxiliary_call_id='aux_' + digest([stratum['dataset_id'], row['unit_id'], call['family'], call['interval']])[:24],
                                source_call_id=None, native_classification_changed=False, trains_native_catalog=False))
                        else:
                            decision['accepted'] = False
                            decision['failures'].append('independent_native_family_shape_not_supported')
                    else:
                        call['native_family_validation'] = dict(status='not_tested_strict_detection_failed', accepted=False)
                accepted.append(dict(unit_id=row['unit_id'], strand=row['strand'], calls=additions))
            ledger.extend(validated)
    write_json(output/'strict_auxiliary_ledger.json.gz', ledger)
    write_json(output/'independent_native_shape_fits.json.gz', validator.fit_manifest())
    return dict(status='complete', model='legacy_strict', auxiliary_model=AUXILIARY_MODEL,
        records=accepted, ledger=ledger, cr_records=[],
        proposed_calls=sum(len(r['calls']) for r in ledger),
        accepted_calls=sum(len(r['calls']) for r in accepted),
        source_recipes=recipes, conditional_accessible_nulls=nulls,
        independent_native_shape_fits=validator.fit_manifest(),
        unavailable=run['unavailable'],
        native_cr_records_unchanged=True, recalled_calls_train_catalog=False,
        recalled_calls_supply_cross_support=False, source_reference_percent=SOURCE_REFERENCE_PERCENT,
        bounded_proposal_radius_bp=kernel.ambiguity_bp,
        semantics='Explicit bounded strict detection auxiliary plus donor-mode/group-excluded native shape validation at zero extra bp; no old CR assignments, posterior calibration or FDR claim')


def _split(run, stratum, native, options, progress):
    from .workflow import _splits

    # The existing splitter is a native intact-versus-separator hypothesis
    # test, not a CR caller. Its CR-like piece matches are nominations only.
    member, _ = grouped_source_tables(run['membership'], run['membership'] > 0,
                                     run['data'], 'pooled', 'pooled')
    proposals = _splits(dict(run, proposal_membership=member), options['split'], progress)
    validator = NativeAuxiliaryShape(stratum, native, mode='pooled',
        maximum_matrix_bytes=options['compute'].maximum_matrix_mb*1024**2,
        max_iterations=options['cr'].family_fit_iterations,
        predictive_replicates=options['cr'].predictive_replicates,
        minimum_source_units=options['split'].minimum_source_units, progress=progress)
    units = {u['unit_id']: u for u in stratum['units']}
    accepted = []
    for row in proposals['ledger']:
        row['legacy_separator_hypothesis_accepted'] = row['accepted']
        row['auxiliary_model'] = 'frozen_native_catalog_intact_vs_separator_v1'
        validations = []
        if row['accepted']:
            for test in row['tests']:
                if test['model'] != 'native_TF':
                    continue
                for ordinal, piece in enumerate(test['matching_CR_pieces']):
                    if piece['family'] not in row['matching_robust_families']:
                        continue
                    unit = deepcopy(units[row['unit_id']])
                    unit['raw_nuc_intervals'] = [s for s in unit['raw_nuc_intervals'] if list(s) != list(row['original_nuc'])]
                    # Keep other proposed protected pieces fixed but leave the
                    # actual separator observations visible in both models.
                    unit['representative_raw_tf_intervals'] = list(unit['representative_raw_tf_intervals']) + [
                        p for p in test['pieces'] if list(p) != list(piece['piece'])]
                    validation = validator.validate(unit, piece['piece'], piece['family'], ordinal=ordinal)
                    validations.append(dict(validation, piece=piece['piece'], family=piece['family']))
        row['independent_native_piece_validations'] = validations
        row['accepted'] = any(v['accepted'] for v in validations)
        row['reason'] = ('independent_native_family_piece_supported' if row['accepted'] else
                         'native_shape_not_supported' if row['legacy_separator_hypothesis_accepted'] else 'separator_hypothesis_not_supported')
        if row['accepted']:
            accepted.append(row)
    return dict(proposals, records=accepted, accepted_spans=len(accepted),
        independent_native_shape_fits=validator.fit_manifest(), native_cr_records_unchanged=True,
        raw_nucleosomes_modified=False, split_pieces_train_catalog=False)


def _comparability(run, options, output, progress):
    """Named auxiliary population diagnostic; never relabel native CR calls."""
    from .population_comparison import native_evidence, compare_populations, simulation_power

    data, kernel = run['data'], run['kernel']
    if kernel is None:
        return dict(status='no_testable_auxiliary_candidates', records=[], unavailable=run['unavailable'])
    if not {'CT', 'GA'} <= set(data['strands']):
        return dict(status='not_applicable', records=[], reason='No biochemical CT/GA strata')
    keys = list(zip(map(str, data['fold_group_ids']), data['strands']))
    if len(set(keys)) != len(keys):
        return dict(status='not_available_grouped_replicates', records=[],
            reason='Conditional population factors need one observation model per group/strand; duplicated physical groups cannot be counted independently')
    members, eligible = grouped_source_tables(run['membership'], run['eligible'], data, 'pooled', 'pooled')
    q, _ = source_geometry_model(output, data, kernel, members, eligible, run['eta'],
        {r['unit_id']: r for r in run['records']}, options['rescue'].geometry_prior_units)
    evidence = native_evidence(output, data, kernel, run['eta'], q)
    records = compare_populations(output, data, kernel, evidence, options['comparability'])
    if options['comparability'].simulation_units:
        simulation_power(output, data, kernel, run['eta'], q, evidence, records, options['comparability'])
    return dict(status='complete', records=records, rescue_counts_used=False,
        auxiliary_model=AUXILIARY_MODEL, configuration_log_activities=run['eta'].tolist(),
        native_cr_records_unchanged=True, native_family_occupancy_established=False,
        semantics='Conditional auxiliary CT/GA population inference on a bounded detection model with a fixed configuration prior; not the native shape-classifier posterior or calibrated cross-assay occupancy')


def run_native_auxiliary(stratum, catalog, native, options, output_dir, progress=None):
    """Entrypoint returning enabled stage dictionaries, with immutable inputs."""
    progress = progress or (lambda *_: None)
    from numba import config, set_num_threads
    set_num_threads(min(options['compute'].cores, config.NUMBA_NUM_THREADS))
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    requested = [s for s in ('rescue', 'comparability', 'split') if options[s].enabled]
    if not requested:
        return {}
    before = digest([stratum, catalog, native])
    # Only private observation adapters receive auxiliary metadata. No caller
    # interval, native score, source member, or cross-dataset link is mutated.
    source = deepcopy(stratum)
    result = {}
    if not catalog or not source['units']:
        return {s: dict(status='empty_native_catalog', records=[]) for s in requested}
    run = _prepare(source, deepcopy(catalog), native, options, progress)
    lattice_run = None
    write_json(output/'source_detection_ledger.json.gz', run['source_detection_ledger'])
    for stage in requested:
        target = output/stage; target.mkdir(exist_ok=True)
        try:
            if (stage == 'rescue' and options['rescue'].source_mode == 'opposite_strand'
                    and not {'CT', 'GA'} <= set(run['data']['strands'])):
                result[stage] = dict(status='not_applicable', records=[],
                    reason='No opposite biochemical strand; use pooled recall explicitly')
                write_json(target/'result.json.gz', result[stage])
                continue
            if stage in ('rescue', 'comparability') and lattice_run is None:
                lattice_run = _prepare_lattice(run, options)
            if stage == 'rescue':
                result[stage] = _recall(lattice_run, source, native, options, target, progress)
            elif stage == 'split':
                result[stage] = _split(run, source, native, options, progress)
            else:
                result[stage] = _comparability(lattice_run, options, target, progress)
        except MemoryError as exc:
            result[stage] = dict(status='resource_limited', records=[], reason=str(exc),
                native_cr_records_unchanged=True, auxiliary_model=AUXILIARY_MODEL)
        write_json(target/'result.json.gz', result[stage])
    if digest([stratum, catalog, native]) != before:
        raise AssertionError('Auxiliary stage changed frozen native CR or observations')
    return result
