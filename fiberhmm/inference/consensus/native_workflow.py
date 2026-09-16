"""Production adapter for the approved native-family distribution CR engine."""
from __future__ import annotations
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import hashlib
import tempfile
import time
import numpy as np

from . import SCHEMA_VERSION
from .artifacts import digest, write_json
from .parameters import options_dict
from .stages import boundary_sr, extract_calls
from .observations import prepare_population
from .nomination import overlap_neighborhoods, discover_neighborhood, deduplicate
from .measurement_family import classify_family_profiles
from .measurement_nomination import augment_catalog
from .native_presentation import MODE, browser_cr
from .native_cross import reciprocal_native_graph
from .native_cross_counts import summarize_native_correspondences
from .native_auxiliary import run_native_auxiliary
from .measurement_grouping import _calls
from .native_catalog_update import bind_native_result, compose_append_frozen
from .progress import report, stage_progress


def nominate_catalog(stratum, region, options, compute, progress):
    """Use the accepted adaptive nominations; no obsolete global MAP assignment.

    Local predictive nomination is caller-conditioned. Residual support-one
    nomination follows native classification, so a local complexity decision
    cannot permanently prevent an unexplained existing call from proposing a
    class. The native-family model, not this seed geometry, classifies calls.
    """
    start,end=region
    report(progress,'cr_nomination','Preparing all-read nomination matrices',dataset_id=stratum['dataset_id'])
    data=prepare_population(stratum,start,end,grid_bp=1,max_intervals=0,
                            max_matrix_bytes=compute.maximum_matrix_mb*1024**2)
    calls=extract_calls(stratum,start,end)
    neighborhoods=overlap_neighborhoods(calls,start,end,options.neighborhood_bandwidth or options.ambiguity_bp)
    args=SimpleNamespace(**vars(options),start=start,end=end,neighborhood_seconds=compute.neighborhood_seconds,
        maximum_nodes=compute.maximum_nodes,maximum_edges=compute.maximum_edges,check=lambda:progress('cr',None))
    centers=[];aliases=[];ledger=[]
    cores=int(compute.cores)
    # Few large neighborhoods dominate: give each worker several numba threads
    # for the row-parallel lattice evaluation instead of one thread per worker.
    threads=2 if cores>=8 else 1
    # The dominant neighborhoods get up to four threads (the worker pool's
    # numba ceiling); the concurrency cap in the dispatcher keeps the total
    # running threads within the core budget.
    large_threads=4 if cores>=8 else threads
    workers=min(max(1,cores//threads),len(neighborhoods))
    if workers>1:
        from .nomination import discover_neighborhoods_in_processes
        def note(done):
            report(progress,'cr_nomination',f'{stratum["dataset_id"]}: native seed neighborhoods {done}/{len(neighborhoods)} ({workers} workers x {threads} threads)',
                dataset_id=stratum['dataset_id'],completed=done,total=len(neighborhoods),unit='neighborhoods')
        outcomes=discover_neighborhoods_in_processes(data,calls,neighborhoods,args,workers,note,threads_per_worker=threads,
                                                     cores=cores,large_threads=large_threads)
    else:
        outcomes=[]
        for number,ids in enumerate(neighborhoods):
            report(progress,'cr_nomination',f'{stratum["dataset_id"]}: native seed neighborhood {number+1}/{len(neighborhoods)}',
                dataset_id=stratum['dataset_id'],completed=number,total=len(neighborhoods),unit='neighborhoods')
            outcomes.append(discover_neighborhood(data,[calls[i] for i in ids],args,number))
    for number,(local,record) in enumerate(outcomes):
        centers.extend(local);aliases.extend([dict(neighborhood=number,local_family=f) for f in range(len(local))]);ledger.append(record)
    report(progress,'cr_nomination','Nomination complete; reconciling exact projection aliases',
        dataset_id=stratum['dataset_id'],completed=len(neighborhoods),total=len(neighborhoods),unit='neighborhoods')
    centers,aliases,unavailable=deduplicate(centers,aliases,data['grid_positions'],options.ambiguity_bp)
    catalog=[dict(family=f'{stratum["dataset_id"]}:F{i+1:04d}',family_index=i,
        consensus_start=int(c[0]),consensus_end=int(c[1]),aliases=aliases[i],
        nomination_provenance='adaptive_native_likelihood_geometry_seed') for i,c in enumerate(centers)]
    return catalog,dict(neighborhoods=ledger,unavailable=unavailable)


def _check_implementation(implementation):
    """A producer receipt cannot describe code that changed during its run."""
    current={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted(Path(__file__).parent.glob('*.py'))}
    if current!=implementation:
        raise ValueError('Native implementation changed during the run; no stale producer binding may be emitted')


def _fit_kwargs(stratum, options, compute):
    """Materialize every numerical/model default actually passed to the fitter."""
    floor=options.minimum_edge_tolerance_bp
    if floor<0:floor=10 if stratum['chemistry']=='ddda' else 0
    kwargs=dict(family_model='latent_distribution',minimum_edge_tolerance_bp=floor,
        loss_odds_levels=(10.,100.,1000.),core_contradiction_odds=100.,
        maximum_matrix_bytes=compute.maximum_matrix_mb*1024**2,
        max_fit_iterations=options.family_fit_iterations,predictive_replicates=options.predictive_replicates,
        scoring_folds=options.scoring_folds)
    if getattr(options, 'edge_tolerance_mode', 'legacy_profile') != 'legacy_profile':
        kwargs['edge_tolerance_mode'] = options.edge_tolerance_mode
    # Declared only when it changes behaviour, so a default run's producer
    # options, binding and artifacts stay byte-identical to the frozen reference.
    if getattr(options,'membership_loss_odds',1.)>1.:
        kwargs['membership_loss_odds']=float(options.membership_loss_odds)
    if getattr(compute,'fit_backend','cpu') not in ('cpu','cuda','mps','auto'):
        # A non-reference objective is part of the model's identity: it must be
        # bound, so its models are never composed with reference-fitted ones.
        kwargs['fit_backend']=str(compute.fit_backend)
        if str(compute.fit_backend) == 'nonparametric':
            kwargs['nonparametric_pseudo_units']=float(getattr(compute,'nonparametric_pseudo_units',4.))
    return kwargs


def _append_frozen_update(stratum, initial_catalog, initial, *, region, fit_options,
                          options, target, implementation, progress, cores=1):
    """Bind the actual producer BEFORE nomination; compose before any downstream use."""
    def persistence_progress(stage, detail):
        report(progress,stage,f'{stratum["dataset_id"]}: {detail}',dataset_id=stratum['dataset_id'])
    persistence_progress('native_provenance','validating and sealing the initial native models')
    _check_implementation(implementation)
    from .native_catalog_update import prepare_source_binding
    binding_args=dict(stratum=stratum,region=region,model_options=fit_options,
                      implementation_contract=implementation,source_binding=prepare_source_binding(stratum))
    initial_binding=bind_native_result(initial,catalog=initial_catalog,**binding_args)
    paths={name:target/filename for name,filename in dict(
        initial_model='native_family_initial.json.gz',initial_catalog='native_catalog_initial.json',
        initial_binding='native_initial_binding.json',augmented_model='native_family_augmented.json.gz',
        augmented_catalog='native_catalog_augmented.json',augmented_binding='native_augmented_binding.json',
        nomination_update='native_nomination_update.json',composed_binding='native_composed_binding.json',
        model_versions='native_model_versions.json',update_report='native_catalog_update.json').items()}
    for name,value in [('initial_model',initial),('initial_catalog',initial_catalog),('initial_binding',initial_binding)]:
        persistence_progress('saving_models',f'writing {name.replace("_", " ")}')
        write_json(paths[name],value)
    bounds=(region['start'],region['end'])
    if options.residual_nomination and initial_catalog:
        positions=np.unique(np.concatenate([np.asarray(u['positions'],np.int64) for u in stratum['units']]))
        catalog,update=augment_catalog(initial_catalog,initial,positions,
                                      dataset_id=stratum['dataset_id'],region=bounds)
    else:
        catalog=deepcopy(initial_catalog)
        update=dict(initial_proposals=len(initial_catalog),added_proposals=0,additions=[],
            status='disabled' if not options.residual_nomination else 'no_initial_source_family',
            raw_footprints_added=0,display_threshold_independent=True)
    write_json(paths['nomination_update'],update)
    if update['added_proposals']:
        progress('cr_classification', f'{stratum["dataset_id"]}: reusing {len(initial_catalog)} frozen models; '
                 f'fitting only {update["added_proposals"]} added models')
        augmented=classify_family_profiles(stratum,catalog,region=bounds,**fit_options,cores=cores,
            _frozen_result=initial,
            progress=stage_progress(progress,'cr_classification',prefix=f'{stratum["dataset_id"]}: ',
                dataset_id=stratum['dataset_id'],task='residual_update'))
    else:
        augmented=deepcopy(initial)
    # The real nomination dependency is part of the augmented producer record,
    # never retroactively attached to the immutable initial snapshot.
    augmented['nomination_update']=deepcopy(update)
    persistence_progress('native_provenance','validating and sealing the augmented native models')
    _check_implementation(implementation)
    augmented_binding=bind_native_result(augmented,catalog=catalog,
        nomination_parent_digest=initial_binding['snapshot_digest'],**binding_args)
    for name,value in [('augmented_model',augmented),('augmented_catalog',catalog),('augmented_binding',augmented_binding)]:
        persistence_progress('saving_models',f'writing {name.replace("_", " ")}')
        write_json(paths[name],value)
    persistence_progress('native_provenance','verifying frozen-model reuse and assembling the composed catalog')
    composed=compose_append_frozen(initial,augmented,initial_binding=initial_binding,
        augmented_binding=augmented_binding,initial_catalog=initial_catalog,augmented_catalog=catalog)
    for name,value in [('composed_binding',composed.binding),('model_versions',composed.model_versions),
                       ('update_report',composed.report)]:
        persistence_progress('saving_models',f'writing {name.replace("_", " ")}')
        write_json(paths[name],value)
    summary=dict(policy='append_frozen',original_family_count=len(initial_catalog),
        existing_models_reused_without_refit=len(initial_catalog),
        added_family_count=composed.report['added_family_count'],
        initial_snapshot_digest=initial_binding['snapshot_digest'],
        augmented_snapshot_digest=augmented_binding['snapshot_digest'],
        composed_snapshot_digest=composed.binding['snapshot_digest'],
        original_models_exact=True,original_evidence_exact=True,
        models_may_share_training_observations=True,joint_mixture=False,
        no_expansion=composed.report['no_expansion'],
        artifacts={name:str(path) for name,path in paths.items()})
    return composed.catalog,composed.result,composed.model_versions,summary


def _empty_native_result(stratum, region, fit_options):
    """Complete no-model producer record; not a fitted or rescued hypothesis."""
    calls=_calls(stratum)
    active=sum(c['start']<region[1] and c['end']>region[0] for c in calls)
    return dict(status='complete',calls=calls,call_family_evidence=[[] for _ in calls],family_models=[],
        partitions={},predictive_partitions={},source_homes=[None]*len(calls),
        diagnostics=dict(no_source_family=True,original_spans_preserved=True,source_calls=active,
            out_of_region_calls=len(calls)-active,source_units=len(stratum['units']),original_catalog_families=0,
            **(dict(edge_tolerance_mode=fit_options['edge_tolerance_mode'])
               if fit_options.get('edge_tolerance_mode', 'legacy_profile') != 'legacy_profile' else {}),
            **{k:fit_options[k] for k in ('family_model','minimum_edge_tolerance_bp','core_contradiction_odds',
                                         'predictive_replicates','scoring_folds')}))


def _parse_tilt(value):
    text = str(value or '0').strip()
    if text.startswith('threshold'):
        return text
    try:
        return float(text)
    except ValueError:
        raise ValueError(f'compute.predictive_tilt must be a number in [0,1) or threshold:M, got {value!r}')


def run_native_workflow(payload, options, output_dir=None, progress=None):
    # The native engine returns before the legacy workflow sets its thread
    # mask. Without this scope even a one-row prior used every host CPU.
    from .execution import numerical_thread_budget, shared_worker_pool
    from .fit_execution import native_fit_pool
    from .scoring_execution import native_scoring_pool
    with numerical_thread_budget(options['compute'].cores), shared_worker_pool(options['compute'].cores), native_fit_pool(
            options['compute'].cores, options['compute'].maximum_matrix_mb*1024**2,
            backend=options['compute'].fit_backend,
            accelerator_bytes=options['compute'].accelerator_mb*1024**2,
            cache_dir=getattr(options['compute'], 'fit_cache_dir', ''),
            smoothing_pseudo_units=float(getattr(options['compute'], 'nonparametric_pseudo_units', 4.))), native_scoring_pool(
            options['compute'].cores, backend=options['compute'].predictive_backend,
            tilt=_parse_tilt(getattr(options['compute'], 'predictive_tilt', '0')),
            accelerator_bytes=options['compute'].accelerator_mb*1024**2):
        return _run_native_workflow(payload, options, output_dir, progress)


def _run_native_workflow(payload, options, output_dir=None, progress=None):
    from .workflow import _pool
    progress=progress or (lambda stage,message:None);started=time.monotonic()
    out=Path(output_dir or tempfile.mkdtemp(prefix='fiberhmm-native-consensus-'));out.mkdir(parents=True,exist_ok=True)
    strata=_pool(payload,options)
    if options['cross'].enabled and len(strata)<2:raise ValueError('XCR requires two selected datasets')
    receipt=dict(schema=SCHEMA_VERSION,cr_mode=MODE,status='running',region=payload['region'],
        parameters=options_dict(options),input_digest=digest(payload),all_units=True,read_sample_cap=None,
        family_count_cap=None,native_source_modified=False,
        model_semantics='Caller-conditioned native shape compatibility; group-excluded parameters; not posterior/FDR or new-call rescue',
        datasets=[dict(dataset_id=s['dataset_id'],chemistry=s['chemistry'],units=len(s['units']),
            model=s.get('model_manifest'),evidence_units=s.get('evidence_units')) for s in strata],
        implementation_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path(__file__).parent.glob('*.py'))})
    from .execution import numerical_environment, warn_if_blas_multithreaded
    receipt['numerical_environment']=numerical_environment()
    if warn_if_blas_multithreaded(receipt['numerical_environment']):
        progress('native','WARNING: multi-threaded BLAS in this process; results may not reproduce the single-threaded reference')
    if payload.get('browser_sources'):receipt['browser_sources']=deepcopy(payload['browser_sources'])
    write_json(out/'manifest.json',receipt)
    results={};cross_inputs={};region=(payload['region']['start'],payload['region']['end'])
    for s in strata:
        name=s['dataset_id'];target=out/digest(name)[:16];target.mkdir(exist_ok=True)
        progress('native',f'{name}: all {len(s["units"])} evidence units')
        sr=boundary_sr(s,options['sr'],progress,cores=options['compute'].cores) if options['sr'].enabled else dict(status='disabled',records=[])
        result=dict(dataset_id=name,chemistry=s['chemistry'],cr_mode=MODE,
            units=[dict(unit_id=u['unit_id'],read_name=u['read_name'],strand=u['strand'],
                source_members=u.get('source_members',[]),native_intervals=u['native_multi_interval_tf_intervals']) for u in s['units']],sr=sr)
        results[name]=result
        if not options['cr'].enabled:continue
        opt=options['cr'];catalog,nomination=nominate_catalog(s,region,opt,options['compute'],progress)
        write_json(target/'nomination.json',nomination)
        fit_options=_fit_kwargs(s,opt,options['compute'])
        receipt.setdefault('native_fit_options',{})[name]=deepcopy(fit_options)
        if not catalog:
            if opt.residual_update_policy=='append_frozen':
                empty=_empty_native_result(s,region,fit_options)
                catalog,empty,versions,update_summary=_append_frozen_update(s,[],empty,
                    region=payload['region'],fit_options=fit_options,options=opt,target=target,
                    implementation=receipt['implementation_sha256'],progress=progress,cores=options['compute'].cores)
                write_json(target/'native_family_model.json.gz',empty)
                result['cr']=browser_cr(s,[],empty,region=region,model_versions=versions)
                result['cr']['model_artifact']=str(target/'native_family_model.json.gz')
            else:
                calls=_calls(s)
                empty=dict(calls=calls,call_family_evidence=[[] for _ in calls],family_models=[],
                           diagnostics=dict(no_source_family=True,original_spans_preserved=True))
                result['cr']=browser_cr(s,[],empty,region=region)
                update_summary=dict(policy='refit_catalog',original_family_count=0,added_family_count=0)
            result['cr']['native_catalog_update']=update_summary
            result['cr'].update(status='no_testable_family',reason='All original calls retained as unresolved; no source model available')
            # A selected dataset with no fitted catalog is unassessed, not an
            # omitted selection or evidence against the other dataset's states.
            cross_inputs[name]=dict(chemistry=s['chemistry'],units=s['units'],
                result=empty if opt.residual_update_policy=='append_frozen' else _empty_native_result(s,region,fit_options))
            continue
        initial_count=len(catalog)
        kwargs=dict(region=region,**fit_options,cores=options['compute'].cores,progress=stage_progress(progress,'cr_classification',
            prefix=f'{name}: ',dataset_id=name,task='initial_catalog'))
        native=classify_family_profiles(s,catalog,**kwargs)
        versions=None
        if opt.residual_update_policy=='append_frozen':
            catalog,native,versions,update_summary=_append_frozen_update(s,catalog,native,
                region=payload['region'],fit_options=fit_options,options=opt,target=target,
                implementation=receipt['implementation_sha256'],progress=progress,cores=options['compute'].cores)
        elif opt.residual_nomination:
            positions=np.unique(np.concatenate([np.asarray(u['positions'],np.int64) for u in s['units']]))
            augmented,update=augment_catalog(catalog,native,positions,dataset_id=name,region=region)
            if update['added_proposals']:
                catalog=augmented;native=classify_family_profiles(s,catalog,**(kwargs | dict(
                    progress=stage_progress(progress,'cr_classification',prefix=f'{name}: ',
                        dataset_id=name,task='residual_refit'))))
            native['nomination_update']=update
        if opt.residual_update_policy=='refit_catalog':
            update_summary=dict(policy='refit_catalog',original_family_count=initial_count,
                                added_family_count=len(catalog)-initial_count)
        report(progress,'saving_models',f'{name}: saving native models and building display records',dataset_id=name)
        write_json(target/'native_family_model.json.gz',native)
        result['cr']=browser_cr(s,catalog,native,region=region,model_versions=versions)
        result['cr']['native_catalog_update']=update_summary
        result['cr']['model_artifact']=str(target/'native_family_model.json.gz')
        cross_inputs[name]=dict(chemistry=s['chemistry'],units=s['units'],result=native)
        # Optional detection/splitting never substitutes legacy CR assignments.
        # It operates beside this frozen native catalog and validates additions
        # against independently trained, group-excluded native shape models.
        result.update(run_native_auxiliary(s,catalog,native,options,target/'auxiliary',progress))
    if options['cross'].enabled:
        graph=reciprocal_native_graph(cross_inputs,region=region,
            reference_percent=options['cross'].native_reference_percent,replicates=options['cross'].native_predictive_replicates,
            minimum_fraction=options['cross'].minimum_fraction,minimum_calls=options['cross'].minimum_support,
            minimum_call_attribution_mass=options['cross'].native_minimum_call_attribution_mass,
            minimum_geometry_retention=options['cross'].native_minimum_geometry_retention,
            minimum_visible_geometry_mass=options['cross'].native_minimum_visible_geometry_mass,
            minimum_testable_fraction=options['cross'].native_minimum_testable_fraction,
            membership_loss_odds=options['cr'].membership_loss_odds,
            pair_nomination_rule=options['cross'].pair_nomination_rule,
            pair_nomination_gap_bp=options['cross'].pair_nomination_gap_bp,
            summarize_agreement=options['cross'].agreement_summary,
            minimum_node_source_units=options['cross'].minimum_node_source_units,
            maximum_matrix_bytes=options['compute'].maximum_matrix_mb*1024**2,progress=stage_progress(progress,'xcr'),
            cores=options['compute'].cores)
        assessments={name:dict(
            fitted_native_models=sum(m.get('status')=='fitted' and 'fold_models' in m
                                     for m in data['result']['family_models']),
            supplied_evidence_units=len(data['units']),native_cr_status=results[name]['cr'].get('status','unrecorded'))
            for name,data in sorted(cross_inputs.items())}
        for assessment in assessments.values():
            assessment['status']='native_models_available' if assessment['fitted_native_models'] else 'no_fitted_native_models'
        graph['selected_dataset_ids']=sorted(cross_inputs)
        graph['dataset_assessments']=assessments
        graph['assessment_status']=('incomplete_no_native_models' if any(
            a['fitted_native_models']==0 for a in assessments.values()) else 'native_model_pairs_assessed')
        progress('finalizing','Saving reciprocal evidence and counting native-family events')
        write_json(out/'native_cross_graph.json.gz',graph)
        groups=summarize_native_correspondences(graph,cross_inputs)
        cross=dict(graph,edges=graph['links'],comparable_edges=sum(e['comparable'] for e in graph['links']),
                   cr_mode=MODE,count_groups=groups)
        cross.pop('links')
    else:cross=dict(status='disabled',edges=[],cr_mode=MODE)
    _check_implementation(receipt['implementation_sha256'])
    from .scoring_execution import scoring_execution_status
    from .fit_execution import fit_execution_status
    receipt['execution_backends'] = dict(predictive=scoring_execution_status(), native_fit=fit_execution_status())
    receipt.update(status='complete',seconds=time.monotonic()-started)
    result=dict(schema=SCHEMA_VERSION,cr_mode=MODE,manifest=receipt,datasets=results,cross=cross)
    progress('finalizing','Saving frozen result and provenance manifest')
    write_json(out/'result.json.gz',result);write_json(out/'manifest.json',receipt)
    progress('complete','Native-family classification and relationship layers ready')
    return result
