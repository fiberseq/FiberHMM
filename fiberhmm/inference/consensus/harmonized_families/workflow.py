"""Full-locus native LLR families with separately inspectable frozen stages.

The reference kernels are extracted, not imported from a benchmark directory.
Initial nomination is the explicitly versioned common-native-cell adapter; all
assignments and consolidation decisions use the accepted predictive machinery.
"""
from collections import defaultdict
from copy import deepcopy
import hashlib
from pathlib import Path
import tempfile
import time

from . import MODE
from .nomination import nominate
from .reference.run_native_locus_map import prepare_input, summarize
from .reference.run_bounded_parent_panel import call_key, evaluate_cohort
from .reference.cross_source_family_consolidation import combine_cases, foreign_child_scores, cross_annotation
from .reference.fitted_geometry_nomination import source_density_cells, nominate_fitted_parents
from .reference.run_native_cell_consolidation import candidate_inputs
from .reference.overlapping_family_update import extend_parent
from .reference.reuse_consensus_fits import reuse_consensus
from .reference.resolve_consensus_representatives import resolve_representatives
from ..artifacts import digest, read_json, write_json
from ..measurement_family import classify_family_profiles
from ..fit_execution import native_fit_pool
from ..scoring_execution import native_scoring_pool
from ..parameters import options_dict
from .presentation import browser_snapshot, presentation_context
from .checkpoints import Checkpoints, publish_source, restore_source
from .parallel import ordered_tasks, parent_task, foreign_task, parent_working_bytes
from ..execution import shared_worker_pool
from ..progress import report, stage_progress

STAGES = [('native', 'Native state fits'), ('parents', 'Shared-state fits'),
          ('consolidated', 'Consolidated hypotheses'), ('resolved', 'Final recurrent states')]


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, value)


def implementation_hashes():
    root=Path(__file__).parent.parent
    return {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob('*.py'))}


def prepare_sources(payload, options):
    """Preserve baseline/replay identity and reject duplicate physical evidence."""
    grouped = {}; physical = {}; identities = set()
    for original in payload['strata']:
        source = deepcopy(original); ds = source['dataset_id']
        if source['chemistry'] not in ('ddda', 'dddb', 'hia5-pacbio', 'hia5-nanopore'):
            raise ValueError('Explicit native chemistry required')
        if ds in grouped and (grouped[ds]['chemistry'] != source['chemistry'] or
                              grouped[ds].get('model_manifest') != source.get('model_manifest')):
            raise ValueError('Conflicting native models within a dataset')
        target = grouped.setdefault(ds, dict(source, units=[]))
        for u in source['units']:
            identity = (ds, u['unit_id'])
            if identity in identities: raise ValueError('Duplicate native evidence-unit ID')
            identities.add(identity)
            # Namespaced dataset IDs must not hide the same PacBio molecule
            # supplied twice. DAF amplification units retain upstream collapse.
            name = u.get('read_name', '')
            pieces = name.split('/')
            molecule = ('pacbio', *pieces[:2]) if len(pieces) >= 3 and pieces[1].isdigit() else ('read', name) if name else None
            keys = [molecule] if molecule else []
            for source_name in u.get('physical_source_names',[]):
                parts=source_name.split('/')
                keys.append(('pacbio',*parts[:2]) if len(parts)>=3 and parts[1].isdigit() else ('read',source_name))
            keys=list(set(keys))
            if u.get('physical_molecule_id'): keys.append(('physical', str(u['physical_molecule_id'])))
            for key in keys:
                if key in physical: raise ValueError('Repeated physical molecule needs joint-view grouping, not independent fitting')
                physical[key] = identity
            baseline = deepcopy(u.get('raw_tf_intervals', u['representative_raw_tf_intervals']))
            u['original_bam_tf_intervals'] = deepcopy(u.get('original_bam_tf_intervals', baseline))
            if options['input'].correct_native:
                if source['chemistry'].startswith('hia5') and options['families'].recall_hia5_nucleosomes and 'upstream_nuc_tf_recall' not in u:
                    raise ValueError('Hia5 nuc recall requested: reload BAM evidence or explicitly disable families.recall_hia5_nucleosomes for the saved scaffold')
                if 'native_multi_interval_tf_intervals' not in u:
                    raise ValueError('Explicit native replay requires BAM-prepared decoder results')
                u['raw_tf_intervals'] = deepcopy(u['native_multi_interval_tf_intervals'])
            else:
                u['raw_tf_intervals'] = baseline
            u['representative_raw_tf_intervals'] = deepcopy(u['raw_tf_intervals'])
            if source['chemistry'].startswith('hia5'): u['strand'] = 'pooled'
            target['units'].append(u)
    return list(grouped.values())


def fit_source(source, region, channel, folder, options, progress, seed_catalog=None):
    started = time.monotonic(); prepared, ledger, stats = prepare_input(source, region)
    if seed_catalog is None:
        catalog, nomination = nominate(prepared, region, options['families'].nomination_radius_bp)
    else:
        # Programmatic parity/testing input, never a motif-driven detector.
        catalog = deepcopy(seed_catalog)
        nomination = dict(method='supplied_existing_call_geometry_catalog', proposals=len(catalog))
    save(folder/'input.json.gz', dict(stratum=prepared, ledger=ledger, stats=stats))
    save(folder/'catalog.json', dict(catalog=catalog, nomination=nomination))
    progress('native_fit', f'{channel}: {len(catalog)} initial hypotheses, {len(ledger)} original calls')
    if catalog:
        native = classify_family_profiles(prepared, catalog, region=tuple(region),
            family_model='latent_distribution', edge_tolerance_mode='bounded', minimum_edge_tolerance_bp=2,
            predictive_replicates=4095, scoring_folds=10, max_fit_iterations=100, retry_fit_iterations=500,
            maximum_matrix_bytes=options['compute'].maximum_matrix_mb*1024**2,
            cores=options['compute'].cores,
            progress=stage_progress(progress,'native_fit',prefix=f'{channel}: ',dataset_id=source['dataset_id']))
        records, summary = summarize(native, ledger, 99.9, catalog)
    else:
        native = dict(calls=[], family_models=[], call_family_evidence=[])
        records = [dict(r, assignment_status='not_evaluated', compatible_families=[],
                        primary_display_family=None) for r in ledger]
        summary = dict(calls=len(ledger), nominated_models=0, actual_simulations=0)
    path = folder/'native.json.gz'; save(path, native)
    # Reference consumers use strict JSON digests and immutable plain records.
    native = read_json(path)
    models = [dict(family=m['family'], reference_interval=m['reference_interval'], domain=m['domain'],
        normalized_geometry=m['normalized_geometry'], native_projection_cell=m['fitted_boundary_cells'],
        fit_warning=any(not d['converged'] for d in m.get('fit_diagnostics', {}).values()),
        source_call_keys=[call_key(native['calls'][i]) for i in m['source_call_indices']])
        for m in native['family_models'] if m.get('status') == 'fitted' and m.get('normalized_geometry')]
    original = {source['dataset_id']+'::'+u['unit_id']:u for u in source['units']}
    case = dict(locus='browser', channel=channel, chemistry=source['chemistry'],
        dataset_id=source['dataset_id'], units={u['unit_id']:u for u in prepared['units']},
        calls=native['calls'], ledger=records, models=models, source_extent=list(region),
        native_cell_provenance=dict(path=str(path.resolve()), digest=digest(native)),
        browser_units=original, source_summary=summary, nomination=nomination)
    save(folder/'source.json.gz', case)
    return case, dict(seconds=time.monotonic()-started, **summary)


def attach_assignment_scores(snapshots, parts, results, foreign):
    """Retain actual frozen predictive scores, including reused-model aliases.

    No refitting, independent confidence invention, or nearest-state fallback.
    """
    scores = defaultdict(dict)
    def add(key, fid, score):
        scores[tuple(key)][fid] = {k:deepcopy(score[k]) for k in
            ('status', 'predictive_tail_interval') if k in score}
    for channel, part in parts.items():
        native = read_json(part['native_cell_provenance']['path'])
        if digest(native) != part['native_cell_provenance']['digest']:
            raise ValueError('Native scores changed since fitting')
        for call, evidence in zip(native['calls'], native.get('call_family_evidence', [])):
            for score in evidence:
                add(call_key(call), channel+'::'+score['family'], score)
    for score in foreign:
        add(score['call_key'], score['hypothesis'], score)
    for fid, result in results.items():
        for score in result['records']:
            add(call_key(score['call']), fid, score)
    for annotation in snapshots.values():
        hypotheses = {h['id']:h for h in annotation['hypotheses']}
        def root(fid):
            seen=set()
            while hypotheses.get(fid, {}).get('reused_from'):
                if fid in seen:
                    raise ValueError('Cyclic reused score model')
                seen.add(fid);fid=hypotheses[fid]['reused_from']
            return fid
        for row in annotation['records']:
            evidence = scores[(row['unit_id'], *row['interval'])]
            row['assignment_compatibility'] = {
                fid:deepcopy(evidence[root(fid)]) for fid in row['display_hypotheses']
                if root(fid) in evidence}


def consolidate_scope(parts, radius, folder, maximum_bytes, stop_after, progress, *, cache, scope_key, cores, minimum_retention_groups=2):
    case = combine_cases(parts); frozen = digest(case); snapshots = {}; timings = {}
    snapshots['native'] = cross_annotation(case, [], {}, radius, [])
    save(folder/'native.json.gz',snapshots['native'])
    if stop_after == 'native':
        attach_assignment_scores(snapshots, parts, {}, [])
        save(folder/'native.json.gz', snapshots['native'])
        return case, snapshots, timings
    started = time.monotonic()
    progress('foreign_scoring', 'Testing original calls against overlapping foreign native hypotheses')
    # Workers need native evidence, not copied Browser units or full ledgers.
    context=folder/'worker_context.json.gz'
    save(context,dict(case={k:case[k] for k in ('units','calls','models','source_extent','source_by_unit')},
        parts={channel:{k:part[k] for k in ('units','models','native_cell_provenance')}
               for channel,part in parts.items()}))
    foreign_by_channel={};tasks=[];foreign_keys={}
    foreign_started=time.monotonic()
    for channel in sorted(parts):
        key=cache.key('foreign',dict(scope=scope_key,channel=channel));foreign_keys[channel]=key
        value=cache.get('foreign',key)
        if value is None:tasks.append((channel,(str(context.resolve()),channel),128*1024**2))
        else:foreign_by_channel[channel]=value
    def save_foreign(channel,value):
        cache.put('foreign',foreign_keys[channel],value)
    foreign_by_channel.update(ordered_tasks(foreign_task,tasks,cores=cores,maximum_bytes=maximum_bytes,
        progress=progress,stage='foreign_scoring',on_result=save_foreign) if tasks else {})
    foreign=[r for channel in sorted(parts) for r in foreign_by_channel[channel]['records']]
    timings['foreign_scoring']=time.monotonic()-foreign_started
    save(folder/'foreign_scores.json.gz', foreign)
    supported = {f for row in case['ledger'] for f in row['compatible_families']}
    models = [m for m in case['models'] if m['family'] in supported and not m['fit_warning']]
    cells = source_density_cells(parts)
    proposals = nominate_fitted_parents(models, radius, cells, 'density_cell')
    save(folder/'nominations.json', proposals)
    results = {}; failures = []; tasks=[];parent_keys={}
    parent_started=time.monotonic()
    def install_parent(pid,value):
        if 'failure' in value:failures.append(value['failure'])
        else:
            results[pid]=value['result']
            save(folder/'parents'/(pid.split(':')[1]+'.json.gz'),value['result'])
    for i, proposal in enumerate(proposals):
        report(progress,'parent_cache',f'Checking shared-fit checkpoints {i+1}/{len(proposals)}',
               completed=i+1,total=len(proposals),unit='hypotheses')
        pid=proposal['id'];key=cache.key('parent',dict(scope=scope_key,proposal=proposal));parent_keys[pid]=key
        value=cache.get('parent',key)
        if value is None:
            tasks.append((pid,(str(context.resolve()),proposal,maximum_bytes),parent_working_bytes(case,proposal)))
        else:install_parent(pid,value)
    def save_parent(pid,value):
        cache.put('parent',parent_keys[pid],value);install_parent(pid,value)
    if tasks:ordered_tasks(parent_task,tasks,cores=cores,maximum_bytes=maximum_bytes,
        progress=progress,stage='parent_fit',on_result=save_parent)
    # Completion order never controls annotation, representative order or seeds.
    results={p['id']:results[p['id']] for p in proposals if p['id'] in results}
    failures.sort(key=lambda f:f['proposal'])
    timings['parent_fitting_and_scoring']=time.monotonic()-parent_started
    annotation = cross_annotation(case, proposals, results, radius, foreign)
    snapshots['parents'] = annotation; timings['parents'] = time.monotonic()-started
    save(folder/'parents.json.gz',annotation)
    save(folder/'parent_failures.json', failures)
    if stop_after != 'parents':
        started = time.monotonic(); progress('consolidation', 'Reusing native evidence for shared explanations')
        annotation, receipts = reuse_consensus(case, annotation, lambda fid: results[fid], nomination_mode='physical_support', minimum_retention_groups=minimum_retention_groups)
        snapshots['consolidated'] = annotation; timings['consolidated'] = time.monotonic()-started
        save(folder/'consolidated.json.gz',annotation)
        save(folder/'consolidation_receipts.json', receipts)
        if stop_after == 'resolved':
            started = time.monotonic(); progress('resolution', 'Resolving representatives without additional fits or simulations')
            annotation, receipts = resolve_representatives(case, annotation, lambda fid: results[fid], minimum_retention_groups=minimum_retention_groups)
            snapshots['resolved'] = annotation; timings['resolved'] = time.monotonic()-started
            save(folder/'resolution_receipts.json', receipts)
    attach_assignment_scores(snapshots, parts, results, foreign)
    for stage, snapshot in snapshots.items():
        if len(snapshot['records']) != len(case['ledger']): raise AssertionError('Original calls lost')
        save(folder/(stage+'.json.gz'), snapshot)
    if digest(case) != frozen: raise AssertionError('Source evidence changed')
    return case, snapshots, timings


def run_staged_families(payload, options, output_dir=None, progress=None):
    if not __debug__:raise RuntimeError('Staged reference invariants require assertions enabled; do not run with python -O')
    started = time.monotonic(); progress = progress or (lambda *_: None)
    implementation=implementation_hashes()
    from ..execution import numerical_environment
    environment=numerical_environment()
    out = Path(output_dir or tempfile.mkdtemp(prefix='fiberhmm-families-')); out.mkdir(parents=True, exist_ok=True)
    region = payload['region']; bounds = [region['start'], region['end']]
    if bounds[1] <= bounds[0] or bounds[1]-bounds[0] > options['compute'].maximum_region_bp:
        raise ValueError('Invalid region or maximum analysis span exceeded; no cropping')
    before = digest(payload); sources = prepare_sources(payload, options)
    save(out/'evidence.json.gz',payload)
    if options['cross'].enabled and len(sources) < 2: raise ValueError('XCR requires at least two datasets')
    mode = 'XCR' if options['cross'].enabled else 'SR' if options['sr'].enabled else 'CR'
    parts = {}; native_timings = {}; dataset_by_channel = {}
    budget = options['compute'].maximum_matrix_mb*1024**2
    cache = options['compute'].fit_cache_dir or str(out/'fit_cache')
    presentation_only={'cli.py','report.py','progress.py','bam.py','bam_export.py','regions.py',
                       'harmonized_families/presentation.py'}
    kernel_implementation={name:value for name,value in implementation.items() if name not in presentation_only}
    checkpoints=Checkpoints(Path(cache)/'staged',kernel_implementation);source_keys={}
    with shared_worker_pool(options['compute'].cores), native_fit_pool(options['compute'].cores, budget, cache_dir=cache), native_scoring_pool(options['compute'].cores):
        for source in sources:
            strands = ['pooled'] if source['chemistry'].startswith('hia5') else sorted({u['strand'] for u in source['units']})
            for strand in strands:
                selected = dict(source, units=[u for u in source['units'] if u['strand'] == strand])
                if not selected['units']: continue
                channel = source['dataset_id']+'::'+strand
                safe = hashlib.sha256(channel.encode()).hexdigest()[:16]
                source_key=checkpoints.key('native',dict(source=selected,region=bounds,
                    nomination_radius=options['families'].nomination_radius_bp))
                source_keys[channel]=source_key;folder=out/'sources'/safe;native_started=time.monotonic()
                progress('native_cache',f'{channel}: checking exact native-fit checkpoint')
                restored=restore_source(checkpoints,source_key,folder)
                if restored is None:
                    if options['compute'].require_native_cache:
                        raise ValueError('Missing or incompatible native-fit checkpoint for '+channel+'; start at native fitting or use the original parameters/cache')
                    parts[channel],native_timings[channel]=fit_source(selected,bounds,channel,folder,options,progress)
                    publish_source(checkpoints,source_key,folder,native_timings[channel])
                    native_timings[channel]['checkpoint_reused']=False
                else:
                    parts[channel],previous=restored
                    native_timings[channel]=dict(previous,original_compute_seconds=previous['seconds'],
                        seconds=time.monotonic()-native_started,checkpoint_reused=True)
                    progress('native_cache',f'{channel}: restored native fits and all Monte Carlo scores')
                dataset_by_channel[channel] = source['dataset_id']
        if mode == 'CR': scopes = {c:{c:p} for c,p in parts.items()}
        elif mode == 'SR': scopes = {s['dataset_id']:{c:p for c,p in parts.items() if dataset_by_channel[c] == s['dataset_id']} for s in sources}
        else: scopes = {'XCR':parts}
        stage_scopes = defaultdict(list); stage_times = defaultdict(float)
        for scope, selected in scopes.items():
            if not selected: continue
            folder = out/'scopes'/hashlib.sha256(scope.encode()).hexdigest()[:16]
            case, snapshots, timings = consolidate_scope(selected, options['families'].physical_radius_bp,
                folder, budget, options['families'].stop_after, progress,cache=checkpoints,
                scope_key=digest({c:source_keys[c] for c in sorted(selected)}),cores=options['compute'].cores,
                minimum_retention_groups=options['families'].minimum_retention_groups)
            for stage, snapshot in snapshots.items(): stage_scopes[stage].append((case, snapshot))
            for stage, seconds in timings.items(): stage_times[stage] += seconds
    # Empty datasets also retain an empty catalog/ledger rather than disappearing.
    requested_stages = [key for key,_ in STAGES[:1+[key for key,_ in STAGES].index(options['families'].stop_after)]]
    context=presentation_context(sources,compact=True)
    snapshots = {stage:browser_snapshot(
        stage_scopes[stage], sources, mode, stage, context,
        minimum_primary_units=options['families'].minimum_display_primary_units,
        minimum_primary_fraction=options['families'].minimum_display_primary_fraction,
        assignment_reference_percent=options['families'].assignment_reference_percent,
    ) for stage in requested_stages}
    realized_channels={s['dataset_id']:sorted({u['strand'] for u in s['units']}) for s in sources}
    populated=sum(bool(v) for v in realized_channels.values())
    observed_sr=any(s['chemistry'] in ('ddda','dddb') and len([c for c in realized_channels[s['dataset_id']] if c!='BOTH'])>1 for s in sources)
    realized_mode=('SR/XCR' if observed_sr else 'XCR') if populated>1 else 'SR' if observed_sr else 'CR' if populated else 'no_evidence'
    data_warnings=[]
    if options['cross'].enabled and populated<2: data_warnings.append('Fewer than two datasets have eligible evidence; cross-dataset support is not established.')
    if options['sr'].enabled and not observed_sr: data_warnings.append('No dataset has both chemical strands represented; strand-shared support is not established.')
    if any(s['chemistry'] in ('ddda','dddb') and not s.get('evidence_units',{}).get('physical_duplex_independence_established') for s in sources): data_warnings.append('Physical duplex independence is not established by this workflow; strand evidence must not be counted as proven independent duplex molecules.')
    final_stage = requested_stages[-1]
    stages = [dict(id=stage, label=dict(STAGES)[stage], seconds=(sum(t['seconds'] for t in native_timings.values()) if stage=='native' else stage_times[stage]),
        families=len({f['family'] for ds in snapshots[stage]['datasets'].values() for f in ds['cr']['catalog']}),
        original_calls=sum(len(r['proposals']) for ds in snapshots[stage]['datasets'].values() for r in ds['cr']['records']),
        assignments=sum(bool(p['family']) for ds in snapshots[stage]['datasets'].values() for r in ds['cr']['records'] for p in r['proposals'])) for stage in requested_stages]
    receipt = dict(schema='fiberhmm.consensus.v1', status='complete', cr_mode=MODE, region=region,
        parameters=options_dict(options), input_digest=before, mode=mode, mode_realized=realized_mode,realized_channels=realized_channels,data_warnings=data_warnings,seconds=time.monotonic()-started,
        all_units=True, read_sample_cap=None, family_count_cap=None, native_source_modified=False,
        nomination='common_native_cell_v1', numerical_policy=dict(native_edge_floor_bp=2, parent_matching_floor_bp=0,
        edge_tolerance_mode='bounded', reference_percent=99.9, predictive_replicates=4095, scoring_folds=10, fit_iterations=100, retry_iterations=500),
        stages=stages, native_timings=native_timings, last_stage=final_stage,
        checkpoints=checkpoints.statistics(),detailed_stage_seconds=dict(stage_times),
        datasets=[dict(dataset_id=s['dataset_id'], chemistry=s['chemistry'], units=len(s['units']),
            model=s.get('model_manifest'),evidence_units=s.get('evidence_units')) for s in sources],
        browser_sources=payload.get('browser_sources'), pooling=payload.get('pooling'), input_files=payload.get('input_files'),
        display_mode=('SR/XCR' if options['cross'].enabled and options['sr'].enabled else mode),
        numerical_environment=environment,implementation_sha256=implementation,
        recurrent_state_display=dict(
            minimum_primary_units=options['families'].minimum_display_primary_units,
            minimum_primary_fraction=options['families'].minimum_display_primary_fraction,
            fitted_alternatives_retained_in_audit_evidence=True,
        ),
        parameter_semantics='Active staged controls plus fixed reference policy; unused legacy controls remain default placeholders',
        presentation_revision='staged_browser_v2')
    result = dict(schema='fiberhmm.consensus.v1', cr_mode=MODE, manifest=receipt,
        stages=stages, stage_results=snapshots, final_stage=final_stage,
        evidence_encoding='shared_json_v1',evidence_pool=context['evidence_pool'],**snapshots[final_stage])
    if digest(payload) != before: raise AssertionError('Input payload mutated')
    if implementation_hashes()!=implementation:raise RuntimeError('Consensus implementation changed during this run; result not installed')
    save(out/'manifest.json', receipt); save(out/'result.json.gz', result)
    from ..report import write_report
    write_report(result,out)
    progress('complete', f'{stages[-1]["families"]} recurrent footprint states ready; all original calls retained')
    return result
