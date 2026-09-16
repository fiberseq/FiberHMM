"""Population-assisted new-call layer; never overwrites ordinary native CR.

Recurrence uses ALL source observations, the normalized joint configuration
likelihood, and a proper uncertain activity prior. A source's inferred count is
not turned into a recipient activity. Single-opportunity members survive.
Geometry/catalog ancestry remains explicitly caller-conditioned empirical Bayes.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import gc
import hashlib
import math
from pathlib import Path
import time
import numpy as np
from scipy.special import ndtri
from scipy.stats import qmc

from .artifacts import digest, read_json, write_json
from .context import physical_alias_exposure
from .geometry import overlap_mask
from .nomination import family_availability
from .population_posterior import PathCache, JointActivityPosterior, integrate_recipient
from .recall import projection_rectangles, exposure, free_domains
from .recall_action import decode_posterior_family_accuracy
from .adapter import read_adapter
from .. import strand_boundary_normalization as sbn


def training_indices(data, recipient_strand, held_fold, mode):
    mask=data['folds']!=held_fold
    if mode=='opposite_strand':mask &= data['strands']!=recipient_strand
    elif mode=='same_strand':mask &= data['strands']==recipient_strand
    elif mode!='pooled':raise ValueError('Unknown population source mode')
    training=np.flatnonzero(mask)
    groups=set(np.asarray(data['fold_group_ids'])[training])
    held=set(np.asarray(data['fold_group_ids'])[data['folds']==held_fold])
    if groups & held:raise AssertionError('Recipient fold-group leaked into activity training')
    return training


def context_arrays(data,kernel,indices,rectangles,q):
    fractions=[];coordinates=[]
    for i in indices:
        frac,coord=physical_alias_exposure(data['units'][i],kernel,rectangles)
        fractions.append(frac);coordinates.append(coord)
    fraction=np.asarray(fractions)
    # No observation-count or modification-outcome gate. Blind aliases stay in
    # both inference partitions. Lost physical mass is not reallocated within f.
    adjustment=np.log(np.maximum(fraction,1e-300))+np.log(q)[None,:]-kernel.logq[None,:]
    return fraction,fraction>0,adjustment,np.asarray(coordinates)


def build_pair(data,kernel,indices,rectangles,q,reference_eta,compute,report=None):
    size=len(indices)*len(kernel.dest)*8
    native=[];prior=[];used=0
    for first in range(0,len(indices),compute.batch_size):
        last=min(len(indices),first+compute.batch_size);ix=indices[first:last]
        _,physical,adj,_=context_arrays(data,kernel,ix,rectangles,q)
        a=PathCache.build(kernel,data['log_lr'][ix],reference_eta,physical,adj)
        b=PathCache.build(kernel,np.zeros_like(data['log_lr'][ix]),reference_eta,physical,adj)
        native.append(a);prior.append(b);used+=a.transition.nbytes+b.transition.nbytes
        if 2*used>compute.maximum_matrix_mb*1024**2:
            raise MemoryError('Exact sparse source cache exceeds configured allocation budget; no units/families dropped')
        if report and first%240==0:report('cache',dict(units=last,total=len(indices),MiB=used/1024**2))
    return PathCache.concatenate(native),PathCache.concatenate(prior)


def fit_source(data,kernel,indices,q,rectangles,opt,compute,out,seed,report):
    out.mkdir(parents=True,exist_ok=True)
    identity=dict(training_unit_ids=[data['unit_ids'][i] for i in indices],
                  training_group_ids=sorted(set(np.asarray(data['fold_group_ids'])[indices])),
                  geometry_digest=digest(q.tolist()),centers_digest=digest(kernel.centers.tolist()),
                  prior_mean=opt.population_prior_mean,prior_sd=opt.population_prior_sd,
                  posterior_draws=opt.population_draws,source_mode=opt.source_mode,seed=seed,
                  native_evidence_digest=hashlib.sha256(np.ascontiguousarray(data['log_lr'][indices]).tobytes()).hexdigest(),
                  physical_context_digest=digest([{k:u.get(k) for k in ['unit_id','aligned_blocks','msp_intervals','raw_nuc_intervals','_region']} for u in [data['units'][i] for i in indices]]),
                  integrator_version='joint_hmc_v1')
    token=digest(identity)
    if (out/'posterior.npz').exists():
        record=read_json(out/'source_model.json')
        if record['training_digest']!=token:raise ValueError('Stale population cache: training definition changed')
        with np.load(out/'posterior.npz') as z:
            return z['draws'].copy(),z['log_weights'].copy(),z['eta_map'].copy(),record
    if not len(indices):raise ValueError('No source evidence units outside the recipient fold')
    reference=np.full(kernel.f,opt.population_prior_mean)
    observed,prior=build_pair(data,kernel,indices,rectangles,q,reference,compute,report)
    model=JointActivityPosterior(observed,prior,opt.population_prior_mean,opt.population_prior_sd)
    best,fits=model.fit(max_iterations=opt.population_iterations,report=report)
    h,cov,hd=model.hessian(best['eta'],report=report)
    if kernel.f>32:
        # The all-family NAPA pilot demonstrated catastrophic high-dimensional
        # importance collapse (ESS~1). Use a target-corrected chain directly;
        # this threshold changes the integrator, never the family catalog.
        draws,lw,diagnostics=model.corrected_chain_draws(best['eta'],cov,number=max(128,opt.population_draws),seed=seed+1009,report=report)
    else:
        draws,lw,diagnostics=model.draws(best['eta'],cov,number=opt.population_draws,seed=seed,report=report)
        if diagnostics['effective_sample_size']<max(16,.2*opt.population_draws):
            initial=diagnostics
            draws,lw,diagnostics=model.corrected_chain_draws(best['eta'],cov,number=max(128,opt.population_draws),seed=seed+1009,report=report)
            diagnostics['initial_importance_proposal']=initial
    p0=prior.evaluate(best['eta'])['family_inclusion']
    p=observed.evaluate(best['eta'])['family_inclusion']
    # Proposal lineage, not support or a likelihood weight. No source confidence cut.
    admission=family_availability([data['units'][i] for i in indices],kernel.centers,kernel.ambiguity_bp).sum(0)
    record=dict(training_digest=token,**identity,fit=best,fit_starts=fits,hessian=hd,integration=diagnostics,
                numerical_provisional=diagnostics['effective_sample_size']<8 or diagnostics.get('maximum_split_Rhat',1.)>1.2,
                source_nomination_units=admission.tolist(),
                source_prior_inclusion_at_MAP=p0.mean(0).tolist(),
                source_posterior_expected_units_at_MAP=p.sum(0).tolist(),
                activity_standard_deviation=np.sqrt(np.diag(cov)).tolist(),
                activity_training='all source observations, exact Z_data/Z_prior; no call-selected count weights',
                ancestry='Frozen full-cohort CR catalog and fold-excluded empirical geometry; not fully OOF discovery')
    np.savez_compressed(out/'posterior.npz',draws=draws,log_weights=lw,eta_map=best['eta'],covariance=cov,hessian=h)
    write_json(out/'source_model.json',record)
    del model,observed,prior;gc.collect()
    return draws,lw,best['eta'],record


def transfer_draws(draws,scale,seed):
    """Proper directional discrepancy eta_recipient|eta_source ~ N(eta_source,s^2 I)."""
    if scale==0:return draws.copy()
    n,f=draws.shape
    u=qmc.Sobol(f,scramble=True,seed=seed).random_base2(int(math.log2(n)))
    return draws+scale*ndtri(np.clip(u,1e-12,1-1e-12))


def adequacy(unit,interval,maximum_diffuse_odds):
    positions=np.asarray(unit['positions']);a,b=np.searchsorted(positions,interval)
    if b-a<3:return dict(passed=True,test='insufficient_positions_for_diffuse_shape_test',log_protected_vs_diffuse=None)
    result=sbn.endpoint_pattern_evidence(read_adapter(unit),int(interval[0]),int(interval[1]))
    value=float(result['protected_log_likelihood']-result['diffuse_log_likelihood'])
    return dict(passed=value>=-math.log(maximum_diffuse_odds),
                test='native_diffuse_adequacy_output_gate_only',log_protected_vs_diffuse=value)


def score_batch(data,kernel,indices,q,rectangles,draws,log_weights,eta_map,
                source_record,opt,originals):
    fraction,physical,adjustment,coordinates=context_arrays(data,kernel,indices,rectangles,q)
    v=data['log_lr'][indices]
    observed=PathCache.build(kernel,v,eta_map,physical,adjustment)
    prior=PathCache.build(kernel,np.zeros_like(v),eta_map,physical,adjustment)
    integrated=integrate_recipient(prior,observed,draws,log_weights,export_geometry=True)
    # Explicit native-only reference: equal activities, SAME empirical geometry,
    # context, competitors and observation domain. It is conditional, not a p-value.
    refp=observed.evaluate(np.zeros(kernel.f))['family_inclusion']
    ref0=prior.evaluate(np.zeros(kernel.f))['family_inclusion']
    from .comparability import inclusion_log_bf
    native_bf,native_available,saturated=inclusion_log_bf(refp,ref0)
    prefix=lambda x:np.c_[np.zeros(len(x)),np.cumsum(x,axis=1)]
    lp=prefix(v);op=prefix(data['observed'][indices]);hp=prefix(data['hits'][indices]==1)
    llr=lp[:,kernel.gb]-lp[:,kernel.ga];opp=op[:,kernel.gb]-op[:,kernel.ga];hit=hp[:,kernel.gb]-hp[:,kernel.ga]
    free_fraction=[];free_coordinates=[]
    for m in indices:
        unit=data['units'][m]
        row=originals[unit['unit_id']]
        blocks=row['source_calls']+[c['interval'] for c in row['proposals']]
        domain=free_domains(unit,blocks)
        frac,coord=exposure(rectangles,kernel.centers[kernel.gf],domain)
        free_fraction.append(frac);free_coordinates.append(coord)
    free_fraction=np.asarray(free_fraction);free_coordinates=np.asarray(free_coordinates)
    ratio=np.divide(free_fraction,fraction,out=np.zeros_like(fraction),where=fraction>0)
    if np.any(ratio>1+1e-8):raise AssertionError('New-call exposure exceeds its parent physical context')
    ratio=np.clip(ratio,0,1)
    admitted=np.asarray(source_record['source_nomination_units'])>=1
    allowed=np.broadcast_to(admitted,(len(indices),kernel.f)).copy()
    for j,m in enumerate(indices):
        for c in originals[data['unit_ids'][m]]['proposals']:
            allowed[j,int(c['family_index'])]=False
    action=(free_fraction>0)&(opp>=1)&(llr>0)
    # Keep the integrated posterior untouched. Decoding only proposes geometries
    # compatible with existing calls; confidence is their family/free-geometry event.
    decoded=decode_posterior_family_accuracy(kernel,v,data['observed'][indices],integrated,eta_map,
                                      allowed,action,adjustment)['geometry_by_family']
    new_mass=np.zeros((len(indices),kernel.f));observable_mass=np.zeros_like(new_mass)
    for j in range(len(indices)):
        observable_mass[j]=np.bincount(kernel.gf,weights=q*fraction[j]*(opp[j]>=1),minlength=kernel.f)
        new_mass[j]=np.bincount(kernel.gf,weights=integrated['geometry_mass'][j]*ratio[j]*(opp[j]>=1),minlength=kernel.f)
    records=[];candidates=[]
    for j,m in enumerate(indices):
        unit=data['units'][m];calls=[];all_candidates=[]
        for f in np.flatnonzero(decoded[j]>=0):
            g=int(decoded[j,f]);interval=free_coordinates[j,g].tolist()
            check=adequacy(unit,interval,opt.maximum_diffuse_odds)
            native=float(native_bf[j,f]);combined=float(new_mass[j,f])
            call=dict(family=data['family_ids'][f],family_index=int(f),interval=interval,
                consensus=kernel.centers[f].tolist(),model_membership=combined,
                population_assisted_probability=combined,family_inclusion_mass=float(integrated['family_inclusion'][j,f]),
                prior_only_family_inclusion=float(integrated['prior_inclusion'][j,f]),
                native_equal_activity_log_bf=native,recipient_integrated_log_bf=float(integrated['recipient_log_bf'][j,f]),
                native_log_lr=float(llr[j,g]),opportunities=int(opp[j,g]),hits=int(hit[j,g]),
                source_training_digest=source_record['training_digest'],
                source_nomination_units=int(source_record['source_nomination_units'][f]),
                source_expected_units=float(source_record['source_posterior_expected_units_at_MAP'][f]),
                source_prior_inclusion_at_MAP=float(source_record['source_prior_inclusion_at_MAP'][f]),
                source_activity_sd=float(source_record['activity_standard_deviation'][f]),
                source_posterior_ESS=float(source_record['integration']['effective_sample_size']),
                recipient_parameter_ESS=float(integrated['effective_sample_size'][j]),
                numerical_provisional=bool(source_record['numerical_provisional'] or integrated['effective_sample_size'][j]<8),
                information_status='weak_population_assisted' if opp[j,g]<=2 or native<math.log(10) else 'recipient_supported',
                adequacy=check,provenance='population_assisted_new_call',new_call=True,
                decision_rule='observed_opportunity_categorical_family_label_loss',
                geometry_validation=dict(updated_family_inclusion_mass=combined),
                semantics='Model-conditional new-family/free-geometry probability; not calibrated accuracy, exact-edge confidence or FDR')
            all_candidates.append(call)
            if check['passed']:calls.append(call)
        calls.sort(key=lambda c:c['interval'])
        if any(a['interval'][1]>b['interval'][0] for a,b in zip(calls,calls[1:])):
            raise AssertionError('New output calls overlap after joint decoding')
        records.append(dict(unit_id=unit['unit_id'],strand=unit['strand'],held_fold=int(data['folds'][m]),calls=calls))
        candidates.append(dict(unit_id=unit['unit_id'],strand=unit['strand'],calls=all_candidates))
    return records,candidates,dict(prior=integrated['prior_inclusion'],combined=integrated['family_inclusion'],
        new_probability=new_mass,native_log_bf=native_bf,integrated_log_bf=integrated['recipient_log_bf'],
        observable_prior_mass=observable_mass,recipient_ess=integrated['effective_sample_size'],
        log_predictive=integrated['log_predictive'])


def run_population_rescue(run,out,options,q,progress):
    data=run['data'];kernel=run['kernel'];opt=options['rescue'];compute=options['compute']
    out=Path(out)/'population_rescue';out.mkdir(parents=True,exist_ok=True)
    rects=projection_rectangles(kernel)
    originals={r['unit_id']:r for r in run['records']}
    original_digest=digest(run['records'])
    n,f=len(data['units']),kernel.f
    matrices={key:np.zeros((n,f)) for key in ['prior','combined','new_probability','native_log_bf','integrated_log_bf','observable_prior_mass']}
    matrices['recipient_ess']=np.zeros(n);matrices['log_predictive']=np.zeros(n)
    rows=[];ledger=[];models=[];started=time.monotonic()
    for strand in sorted(set(data['strands'])):
        for held_fold in (0,1):
            recipients=np.flatnonzero((data['strands']==strand)&(data['folds']==held_fold))
            if not len(recipients):continue
            train=training_indices(data,strand,held_fold,opt.source_mode)
            geometry=(1-opt.population_geometry_floor)*q[held_fold]+opt.population_geometry_floor*np.exp(kernel.logq)
            tag=f'{strand}_held{held_fold}'
            def report(stage,values):
                if stage=='fit' and values.get('iteration',0)%8:return
                if progress:progress('population_rescue',f'{data["dataset_id"]} {tag}: {stage} {values}')
                print('POPULATION_RESCUE',data['dataset_id'],tag,stage,values,'seconds',round(time.monotonic()-started,1),flush=True)
            source_draws,lw,eta,model=fit_source(data,kernel,train,geometry,rects,opt,compute,out/tag,
                options['cr'].seed+held_fold*101+(1 if strand=='CT' else 2),report)
            models.append(dict(recipient_strand=strand,held_fold=held_fold,**model))
            draws=transfer_draws(source_draws,opt.population_transfer_sd,options['cr'].seed+held_fold*107+(3 if strand=='CT' else 4))
            for begin in range(0,len(recipients),compute.batch_size):
                ix=recipients[begin:begin+compute.batch_size]
                rr,cc,stats=score_batch(data,kernel,ix,geometry,rects,draws,lw,eta,model,opt,originals)
                rows.extend(rr);ledger.extend(cc)
                for key,value in stats.items():matrices[key][ix]=value
                if begin%120==0 or begin+compute.batch_size>=len(recipients):report('score',dict(units=begin+len(ix),total=len(recipients)))
            write_json(out/f'{tag}_records.json.gz',[r for r in rows if r['strand']==strand and r['held_fold']==held_fold])
    if len(rows)!=n or len({r['unit_id'] for r in rows})!=n:raise AssertionError('Not every evidence unit was scored exactly once')
    order={u:i for i,u in enumerate(data['unit_ids'])};rows.sort(key=lambda r:order[r['unit_id']])
    combined=deepcopy(run['records']);by_id={r['unit_id']:r for r in combined}
    for r in rows:
        for call in r['calls']:
            c=deepcopy(call);c['source_ordinals']=[];c['source_intervals']=[]
            c['boundary_changed']=False
            by_id[r['unit_id']]['proposals'].append(c)
        by_id[r['unit_id']]['proposals'].sort(key=lambda c:c['interval'])
    if digest(run['records'])!=original_digest:raise AssertionError('Ordinary native CR was mutated')
    np.savez_compressed(out/'recipient_evidence.npz',unit_ids=np.asarray(data['unit_ids']),
                        family_ids=np.asarray(data['family_ids']),folds=data['folds'],strands=data['strands'],**matrices)
    write_json(out/'candidate_ledger.json.gz',ledger);write_json(out/'source_models.json',models)
    result=dict(status='complete',model='population_assisted',records=rows,cr_records=combined,
        decision_rule='observed_opportunity_categorical_family_label_loss',
        proposed_calls=sum(len(r['calls']) for r in ledger),retained_calls=sum(len(r['calls']) for r in rows),
        accepted_calls=sum(c['model_membership']>=opt.minimum_probability for r in rows for c in r['calls']),
        reported_acceptance_probability=opt.minimum_probability,all_threshold_candidates_retained=True,
        source_recipes=models,conditional_accessible_nulls=[],native_CR_unchanged=True,
        semantics='Joint source activity posterior; exact lattice; target-corrected parameter integration with ESS/convergence diagnostics. '
                  'Frozen catalog/empirical geometry ancestry is not fully OOF; confidence is not calibrated accuracy/FDR.',
        numerical_provisional_source_models=sum(m['numerical_provisional'] for m in models),
        seconds=time.monotonic()-started)
    write_json(out/'result.json.gz',result)
    return result
