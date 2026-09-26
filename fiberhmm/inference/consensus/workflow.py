"""Portable staged consensus workflow used by FiberBrowser and batch replay."""
from __future__ import annotations
from collections import Counter
from copy import deepcopy
from itertools import combinations
from pathlib import Path
import hashlib
import tempfile
import time
import numpy as np
from . import SCHEMA_VERSION
from .artifacts import digest, write_json
from .parameters import parse_options, options_dict
from .stages import boundary_sr, discover_cr, assign_cr, make_kernel
from .geometry import merged
from .cross_evidence import prepare_shape_data, shape_predictive, direction, plausible_pairs, link_status
from .cross_joint import prediction, replacement_diagnostics
from .cross_quantification import summarize_correspondences
from .recall import source_geometry_model, projection_rectangles, source_recipe, score_group, score_candidates
from .population_comparison import native_evidence, compare_populations, simulation_power
from . import splitting


class ConsensusCancelled(Exception):
    """Cooperative cancellation; no partial layer is installed."""


def _pool(payload, options):
    region = payload['region']; start, end = int(region['start']), int(region['end'])
    if start < 0 or end <= start or end-start > options['compute'].maximum_region_bp:
        raise ValueError('Invalid region or explicit region budget exceeded; no automatic cropping')
    groups={}; ids=set()
    for source in payload['strata']:
        ds=source['dataset_id']; chemistry=source['chemistry']
        if chemistry not in ('ddda','dddb','hia5-pacbio','hia5-nanopore'):
            raise ValueError(f'Unknown native chemistry {chemistry!r}; declare the dataset datatype')
        if ds in groups and (groups[ds]['chemistry'] != chemistry or
                groups[ds].get('model_manifest') != source.get('model_manifest')):
            raise ValueError(f'Cannot pool different native models inside dataset {ds}')
        target=groups.setdefault(ds,{**source,'stratum_id':ds+':consensus','units':[]})
        for original in source['units']:
            u=deepcopy(original)
            if u['unit_id'] in ids: raise ValueError('Duplicate evidence-unit ID across input strata')
            ids.add(u['unit_id']);u['_region']=[start,end]
            u['original_bam_tf_intervals']=deepcopy(u.get('original_bam_tf_intervals',u['representative_raw_tf_intervals']))
            if options['input'].correct_native:
                if 'native_multi_interval_tf_intervals' not in u:
                    raise ValueError('Corrected native replay requested but actual-query replay is missing; use the BAM adapter or explicitly disable replay')
                u['representative_raw_tf_intervals']=deepcopy(u['native_multi_interval_tf_intervals'])
            else:
                u['native_multi_interval_tf_intervals']=deepcopy(u['representative_raw_tf_intervals'])
            # Alignment orientation is not a chemical strand in Hia5.
            if chemistry.startswith('hia5'): u['strand']='pooled'
            target['units'].append(u)
    return list(groups.values())


def _model_geometry(run, output, options, progress):
    if 'geometry_q' in run: return run['geometry_q'],run['geometry_counts']
    data=run['data'];data['progress']=progress;data['rescue_options']=options['rescue']
    originals={r['unit_id']:r for r in run['records']}
    q,counts=source_geometry_model(output,data,run['kernel'],run['proposal_membership'],
        run['eligible'],run['eta'],originals,options['rescue'].geometry_prior_units)
    run['geometry_q'],run['geometry_counts']=q,counts
    return q,counts


def _recall(run, output, options, progress):
    data=run['data']; opt=options['rescue']; kernel=run['kernel']
    if opt.source_mode=='opposite_strand' and not {'CT','GA'} <= set(data['strands']):
        return dict(status='not_applicable',reason='Opposite biochemical strand unavailable; choose pooled CR recall explicitly',records=[])
    q,counts=_model_geometry(run,output,options,progress)
    if opt.model=='population_assisted':
        from .population_rescue import run_population_rescue
        return run_population_rescue(run,output,options,q,progress)
    rects=projection_rectangles(kernel); candidates=[];nulls=[];recipes=[]
    rng=np.random.default_rng(options['cr'].seed+1)
    for strand in sorted(set(data['strands'])):
        for fold in (0,1):
            indices=np.flatnonzero((data['strands']==strand)&(data['folds']==fold))
            eta,admitted,recipe=source_recipe(run['proposal_membership'],run['eligible'],
                data['strands'],data['folds'],strand,fold,opt.source_mode,opt.prior_scale,
                minimum_source_units=opt.minimum_source_units)
            rows,control,veto=score_group(kernel,data,indices,eta,admitted,rects,
                [opt.minimum_protection_mass],rng,opt.null_replicates,'posterior_accuracy')
            candidates.extend(rows);nulls.append(dict(strand=strand,fold=fold,counts=[dict(v) for v in control],vetoes=veto))
            recipes.append(dict(strand=strand,fold=fold,**recipe))
    original={r['unit_id']:r for r in run['records']}; proposed={r['unit_id']:r for r in candidates}
    ledger=score_candidates(output,data,kernel,run['proposal_membership'],run['eligible'],
        original,proposed,q,counts)
    accepted=[dict(unit_id=r['unit_id'],strand=r['strand'],calls=[c for c in r['calls']
        if c['geometry_decisions']['selected']['accepted'] and c['protection_event_mass']>=opt.minimum_protection_mass]) for r in ledger]
    by_id={r['unit_id']:r['calls'] for r in accepted}
    spans=[u['representative_raw_tf_intervals']+[c['interval'] for c in by_id.get(u['unit_id'],[])] for u in data['units']]
    reassigned=assign_cr(data,kernel,run['eta'],options['cr'],options['compute'],progress,calls=spans)
    return dict(status='complete',records=accepted,cr_records=reassigned['records'],
        proposed_calls=sum(len(r['calls']) for r in candidates),accepted_calls=sum(len(r['calls']) for r in accepted),
        source_recipes=recipes,conditional_accessible_nulls=nulls,
        semantics='Fold-excluded source geometry and counts; full-data ancestor catalog/activities. Not fully OOF or calibrated FDR.')


def _splits(run, opt, progress):
    data=run['data']; catalogs=run['catalog']
    counts=(run['proposal_membership']>=.5).sum(0)
    families=[dict(f,source_units=int(counts[j])) for j,f in enumerate(catalogs) if counts[j]>=opt.minimum_source_units]
    nuc_tables=None
    if data['chemistry']=='ddda':
        from ...core.model_io import load_model_with_metadata
        path=Path(__file__).resolve().parents[2]/'models'/'ddda_nuc.json'
        model,_,_=load_model_with_metadata(str(path));nuc_tables=splitting.conditional_model_tables(model)
    rng=np.random.default_rng(20260906);ledger=[];accepted=[]
    for m,u in enumerate(data['units']):
        if m%24==0: progress('splitting',f'{data["dataset_id"]}: internal split tests {m}/{len(data["units"])}')
        positions=np.asarray(u['positions']);hits=np.asarray(u['hits']);contexts=np.asarray(u['contexts'])
        for span in u['raw_nuc_intervals']:
            a,b=span
            if a<data['start'] or b>data['end'] or not any(lo<=a and b<=hi for lo,hi in merged(u['aligned_blocks'])): continue
            fs=[f for f in families if f['consensus_start']<b and f['consensus_end']>a]
            if not fs: continue
            qa,qb=np.searchsorted(positions,[a,b]);pos=positions[qa:qb];h=hits[qa:qb]
            if len(pos)<9: continue
            g=splitting.geometry(pos,span,fs,opt.ambiguity_bp,opt.maximum_gap_bp)
            if g is None:continue
            pp=np.asarray(u['p_protected'])[qa:qb];pa=np.asarray(u['p_accessible'])[qa:qb]
            tf_steps=np.where(h,np.log(pp/pa),np.log1p(-pp)-np.log1p(-pa))
            models={'native_TF':(pp,pa)}
            if nuc_tables is not None:
                nuc_acc=nuc_tables[1][contexts[qa:qb]]
                models['installed_nuc']=(nuc_tables[0][contexts[qa:qb]],nuc_acc)
            tests=[];passing=[]
            for name,(pr,ac) in models.items():
                prefix=np.r_[0.,np.cumsum(np.where(h,np.log(ac/pr),np.log1p(-ac)-np.log1p(-pr)))]
                gain=prefix[g['ends']]-prefix[g['starts']]
                result=splitting.evaluate(g,gain,opt.activity)
                action=splitting.summarize_action(result,g,span,pos,h,gain,fs,opt.ambiguity_bp,tf_steps,
                    minimum_separator_opportunities=opt.minimum_separator_opportunities,minimum_separator_bf=opt.minimum_separator_bf)
                passed=result['log_bf_any_split']>=np.log(opt.minimum_split_bf) and action['strong_separator_and_CR_geometry']
                passing.append({f['family'] for f in action['matching_CR_pieces']} if passed else set())
                null=[]
                for _ in range(opt.null_replicates):
                    nh=rng.random(len(pos))<pr
                    npref=np.r_[0.,np.cumsum(np.where(nh,np.log(ac/pr),np.log1p(-ac)-np.log1p(-pr)))]
                    ng=npref[g['ends']]-npref[g['starts']]
                    nr=splitting.evaluate(g,ng,opt.activity)
                    ntf=np.where(nh,np.log(pp/pa),np.log1p(-pp)-np.log1p(-pa))
                    na=splitting.summarize_action(nr,g,span,pos,nh,ng,fs,opt.ambiguity_bp,ntf,
                        minimum_separator_opportunities=opt.minimum_separator_opportunities,minimum_separator_bf=opt.minimum_separator_bf)
                    null.append(dict(log_bf_any_split=nr['log_bf_any_split'],qualified=na['strong_separator_and_CR_geometry']))
                tests.append(dict(model=name,activity=opt.activity,**{k:v for k,v in result.items() if k not in ('selected','geometry_mass')},**action,intact_model_null=null))
            robust=passing[0] & passing[1] if opt.require_nuc_model and nuc_tables is not None else passing[0]
            row=dict(unit_id=u['unit_id'],strand=u['strand'],original_nuc=span,tests=tests,
                matching_robust_families=sorted(robust),accepted=bool(robust))
            ledger.append(row)
            if robust: accepted.append(row)
    return dict(status='complete',records=accepted,ledger=ledger,tested_spans=len(ledger),
        accepted_spans=len(accepted),outer_boundaries_unchanged=True,nucleosome_identity_inferred=False)


def _cross(runs, opt, cr, compute, progress):
    edges=[];prepared={};cache={};baselines={}
    for name,r in runs.items():
        if r['kernel'] is not None: prepared[name]=prepare_shape_data(r['data'],r['data']['start'],r['data']['end'])
    def shape(name,center):
        key=(name,tuple(map(int,center)))
        if key not in cache:cache[key]=shape_predictive(prepared[name],center,cr.ambiguity_bp,
            minimum_opportunities=cr.minimum_opportunities)
        return cache[key]
    def joint(name,f,replacement):
        r=runs[name];data=r['data'];ix=np.flatnonzero(r['allowed'][:,f])
        if not len(ix):return dict(status='untestable',retained_native_mass_fraction=None)
        if name not in baselines:baselines[name]=prediction(r['kernel'],data['log_lr'],r['eta'],r['allowed'])
        base,inc=baselines[name];centers=r['kernel'].centers.copy();centers[f]=replacement
        try:
            replacement_kernel=make_kernel(data,centers,cr,compute)
        except MemoryError as exc:
            return dict(status='resource_limited',retained_native_mass_fraction=None,reason=str(exc))
        except ValueError as exc:
            if 'no visible nonempty opportunity projection' not in str(exc):raise
            return dict(status='untestable',retained_native_mass_fraction=None,reason=str(exc))
        pred,post=prediction(replacement_kernel,data['log_lr'][ix],r['eta'],r['allowed'][ix])
        return replacement_diagnostics(pred-base[ix],inc[ix,f],post[:,f],opt.tolerance_odds)
    for left,right in combinations(prepared,2):
        l,r=runs[left],runs[right]
        for i,j in plausible_pairs(l['kernel'].centers,r['kernel'].centers,cr.ambiguity_bp):
            progress('xcr', f'{left} ↔ {right}: scoring relationship {len(edges)+1}')
            a,b=l['kernel'].centers[i],r['kernel'].centers[j]
            forward=direction(shape(right,b),shape(right,a),r['membership'][:,j],opt.tolerance_odds)
            reverse=direction(shape(left,a),shape(left,b),l['membership'][:,i],opt.tolerance_odds)
            status=link_status(forward,reverse,opt.minimum_support,opt.minimum_fraction)
            row=dict(edge_id=digest([left,i,right,j])[:24],left_dataset=left,right_dataset=right,
                left_family=l['catalog'][i]['family'],right_family=r['catalog'][j]['family'],
                left_interval=a.tolist(),right_interval=b.tolist(),status=status,
                forward=forward,reverse=reverse,comparability_mask=False,
                relationship='nested' if (a[0]<=b[0] and a[1]>=b[1]) or (b[0]<=a[0] and b[1]>=a[1]) else 'overlapping_alternative')
            if status=='reciprocal_shape_compatible' or opt.joint_validation=='all':
                row['joint_left']=joint(left,i,b);row['joint_right']=joint(right,j,a)
                row['comparability_mask']=bool(status=='reciprocal_shape_compatible' and all(
                    v['retained_native_mass_fraction'] is not None and v['retained_native_mass_fraction']>=opt.minimum_joint_fraction
                    for v in (row['joint_left'],row['joint_right'])))
                # Shape compatibility and preservation of an individual class
                # label are different scales. Dropping a shape edge because a
                # nearby class absorbs mass would fragment the coarse ANY-member
                # endpoint again. Keep that edge, but do not call its individual
                # family identity unambiguous without this additional test.
                row['target_role_comparability_mask']=bool(row['comparability_mask'] and all(
                    (v.get('jointly_retained_target_mass_fraction') or 0.)>=opt.minimum_target_retention
                    for v in (row['joint_left'],row['joint_right'])))
                row['joint_status']=('passed_full_model_and_target_retention' if row['target_role_comparability_mask'] else
                    'shape_compatible_target_role_unresolved' if row['comparability_mask'] else 'replacement_fit_not_retained')
            else: row['joint_status']='not_tested_provisional'
            edges.append(row)
    count_groups,annotations=summarize_correspondences(edges,runs,minimum_shared_support=opt.minimum_support)
    for edge in edges:
        edge.update(annotations.get(edge['edge_id'],dict(count_comparison='shape_comparability_not_established',
            individual_count_comparison_unambiguous=False)))
    return dict(status='complete',edges=edges,counts=dict(Counter(e['status'] for e in edges)),
        comparable_edges=sum(e['comparability_mask'] for e in edges),merged_intervals=False,
        count_groups=count_groups,
        count_semantics='Caller-conditioned physical decoded ANY-member events with exact group inclusion >=0.5; ANY-variant and canonical-core eligibility are separate. Shape compatibility is not calibrated occupancy equivalence or equal detection power.',
        semantics='Native population-shape relationship graph and full-model counterfactual sensitivity; not calibrated q/FDR or matched molecule identity.')


def run_workflow(payload, parameters=None, output_dir=None, progress=None):
    options=parse_options(parameters);started=time.monotonic()
    if options['cr'].engine == 'lattice_recaller':
        from .lattice_recaller import run_lattice_recaller
        return run_lattice_recaller(payload, options, output_dir, progress)
    if options['cr'].engine == 'staged_native_families':
        from .harmonized_families.workflow import run_staged_families
        return run_staged_families(payload, options, output_dir, progress)
    if options['cr'].engine == 'call_harmonization':
        from .call_clustering.workflow import run_harmonization
        return run_harmonization(payload, options, output_dir, progress)
    if options['cr'].engine == 'native_family_distribution':
        from .native_workflow import run_native_workflow
        return run_native_workflow(payload, options, output_dir, progress)
    out=Path(output_dir or tempfile.mkdtemp(prefix='fiberhmm-consensus-'));out.mkdir(parents=True,exist_ok=True)
    progress=progress or (lambda stage,message:None)
    strata=_pool(payload,options)
    if options['cross'].enabled and len(strata)<2:raise ValueError('XCR requires at least two selected datasets')
    from numba import set_num_threads, config
    set_num_threads(min(options['compute'].cores,config.NUMBA_NUM_THREADS))
    receipt=dict(schema=SCHEMA_VERSION,status='running',region=payload['region'],parameters=options_dict(options),
        input_digest=digest(payload),all_units=True,read_sample_cap=None,family_count_cap=None,
        native_source_modified=False,model_semantics='Exploratory full-cohort caller-conditioned CR; native per-opportunity emissions; not calibrated Q/FDR or fully out-of-fold inference',
        datasets=[dict(dataset_id=s['dataset_id'],chemistry=s['chemistry'],units=len(s['units']),model=s.get('model_manifest'),
            evidence_units=s.get('evidence_units')) for s in strata])
    receipt['implementation_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(Path(__file__).parent.glob('*.py'))}
    from .execution import numerical_environment, warn_if_blas_multithreaded
    receipt['numerical_environment']=numerical_environment()
    if warn_if_blas_multithreaded(receipt['numerical_environment']):
        progress('native','WARNING: multi-threaded BLAS in this process; results may not reproduce the single-threaded reference')
    if payload.get('browser_sources'):
        receipt['browser_sources']=deepcopy(payload['browser_sources'])
    write_json(out/'manifest.json',receipt)
    runs={};results={}
    for s in strata:
        name=s['dataset_id'];target=out/digest(name)[:16];target.mkdir(exist_ok=True)
        progress('native',f'{name}: all {len(s["units"])} evidence units')
        sr=boundary_sr(s,options['sr'],progress) if options['sr'].enabled else dict(status='disabled',records=[])
        result=dict(dataset_id=name,chemistry=s['chemistry'],units=[dict(unit_id=u['unit_id'],read_name=u['read_name'],strand=u['strand'],
            source_members=u.get('source_members',[]),native_intervals=u['native_multi_interval_tf_intervals']) for u in s['units']],sr=sr)
        results[name]=result
        if not options['cr'].enabled:continue
        run=discover_cr(s,payload['region']['start'],payload['region']['end'],options['cr'],options['compute'],progress)
        runs[name]=run; result['cr']=dict(status=run['status'],catalog=run['catalog'],records=run['records'])
        write_json(target/'nomination.json',run['ledger'])
        if run['kernel'] is None:continue
        run['data']['progress']=progress;run['data']['rescue_options']=options['rescue']
        baseline=assign_cr(run['data'],run['kernel'],run['eta'],options['cr'],options['compute'],progress,
            calls=[u['native_multi_interval_tf_intervals'] for u in s['units']]) if options['sr'].enabled else run
        run['native_proposal_membership']=baseline['proposal_membership']
        run['native_allowed']=baseline['allowed']
        for f,c in enumerate(run['catalog']):
            c['rates_by_strand']={strand:dict(eligible_units=int(run['eligible'][run['data']['strands']==strand,f].sum()),
                canonical_core_eligible_units=int(run['core_eligible'][run['data']['strands']==strand,f].sum()),
                native_assignments=int((baseline['proposal_membership'][run['data']['strands']==strand,f]>=.5).sum()),
                after_sr_assignments=int((run['proposal_membership'][run['data']['strands']==strand,f]>=.5).sum())) for strand in sorted(set(run['data']['strands']))}
        np.savez_compressed(target/'cr_model.npz',unit_ids=np.asarray(run['data']['unit_ids']),centers=run['kernel'].centers,
            log_activities=run['eta'],family_inclusion=run['membership'],proposal_membership=run['proposal_membership'],eligible=run['eligible'],
            core_eligible=run['core_eligible'],core_opportunities=run['core_opportunities'],
            native_proposal_membership=baseline['proposal_membership'],allowed=run['allowed'],native_allowed=baseline['allowed'])
        result['cr']['kernel']=run['kernel'].metadata()
        result['cr']['fits']=[{k:v for k,v in fit.items() if k!='eta'} for fit in run['fits']]
        result['cr']['fit_diagnostics']=run['fit_diagnostics']
        if options['rescue'].enabled:
            result['rescue']=_recall(run,target,options,progress)
            counts=Counter((r['strand'],p['family']) for r in result['rescue'].get('cr_records',[])
                for p in r['proposals'] if p['model_membership']>=.5)
            if result['rescue']['status']=='complete':
                for c in run['catalog']:
                    for strand,rates in c['rates_by_strand'].items():
                        rates['after_recall_assignments']=counts[strand,c['family']]
        if options['comparability'].enabled:
            if not {'CT','GA'} <= set(run['data']['strands']):result['comparability']=dict(status='not_applicable',records=[])
            else:
                q,_=_model_geometry(run,target,options,progress)
                ev=native_evidence(target,run['data'],run['kernel'],run['eta'],q)
                records=compare_populations(target,run['data'],run['kernel'],ev,options['comparability'])
                if options['comparability'].simulation_units:
                    simulation_power(target,run['data'],run['kernel'],run['eta'],q,ev,records,options['comparability'])
                result['comparability']=dict(status='complete',records=records,rescue_counts_used=False)
        if options['split'].enabled:result['split']=_splits(run,options['split'],progress)
    cross=_cross(runs,options['cross'],options['cr'],options['compute'],progress) if options['cross'].enabled else dict(status='disabled',edges=[])
    receipt.update(status='complete',seconds=time.monotonic()-started)
    result=dict(schema=SCHEMA_VERSION,manifest=receipt,datasets=results,cross=cross)
    write_json(out/'result.json.gz',result);write_json(out/'manifest.json',receipt)
    progress('complete','Consensus layers ready')
    return result


def run_analysis(payload, parameters=None, output_dir=None, progress=None):
    """Public Browser/CLI entry point: lattice recaller unless cr.engine names the staged engine; metadata-derived source mode."""
    from .regions import automatic_parameters
    return run_workflow(payload, automatic_parameters(parameters,payload['strata']), output_dir, progress)
