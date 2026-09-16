from __future__ import annotations
import time
import hashlib
import numpy as np
from .artifacts import write_json
from .recall import projection_rectangles, exposure, free_domains
from .comparability import inclusion_log_bf, quadrature, prevalence_posterior, compare_strands, wilson_interval, sample_conditional_configurations

def physical_model(units,folds,kernel,q,rects):
    # Native calls are NOT obstacles. Existing MSP/nucleosome context is held
    # fixed and explicitly conditions the analysis, rather than being called
    # outcome-free experimental ground truth.
    fraction=np.asarray([exposure(rects,kernel.centers[kernel.gf],free_domains(u,[],block_native=False))[0] for u in units])
    adjustment=np.log(np.maximum(fraction,1e-300))+np.log(q[folds])-kernel.logq
    return fraction,fraction>0,adjustment


def native_evidence(output,data,kernel,eta,q):
    n=len(data['units']);f=kernel.f;rects=projection_rectangles(kernel);started=time.monotonic()
    result={key:np.zeros((n,f)) for key in ['log_bf','prior_inclusion','posterior_inclusion','direct_expected_kl','physical_geometry_mass']}
    result['available']=np.zeros((n,f),bool);result['numerically_saturated']=np.zeros((n,f),bool)
    for begin in range(0,n,24):
        if data.get('progress'):data['progress']('comparability',f'Native evidence {begin}/{n}')
        end=min(n,begin+24);values=data['log_lr'][begin:end];folds=data['folds'][begin:end]
        fraction,physical,adjustment=physical_model(data['units'][begin:end],folds,kernel,q,rects)
        prior=kernel.evaluate(np.zeros_like(values),eta,geometry_allowed=physical,geometry_log_adjustment=adjustment,export_geometry=True)
        post=kernel.evaluate(values,eta,geometry_allowed=physical,geometry_log_adjustment=adjustment)
        bf,available,saturated=inclusion_log_bf(post['family_inclusion'],prior['family_inclusion'])
        pa=data['p_accessible'][begin:end];pp=data['p_protected'][begin:end];obs=data['observed'][begin:end]
        kl=np.where(obs,pp*np.log(pp/pa)+(1-pp)*np.log((1-pp)/(1-pa)),0.)
        prefix=np.c_[np.zeros(len(values)),np.cumsum(kl,axis=1)]
        gkl=prefix[:,kernel.gb]-prefix[:,kernel.ga]
        for j in range(len(values)):
            expected=np.bincount(kernel.gf,weights=prior['geometry_mass'][j]*gkl[j],minlength=f)
            result['direct_expected_kl'][begin+j]=np.divide(expected,prior['family_inclusion'][j],out=np.zeros(f),where=prior['family_inclusion'][j]>0)
            result['physical_geometry_mass'][begin+j]=np.bincount(kernel.gf,weights=q[folds[j]]*fraction[j],minlength=f)
        result['log_bf'][begin:end]=bf;result['available'][begin:end]=available
        result['numerically_saturated'][begin:end]=saturated
        result['prior_inclusion'][begin:end]=prior['family_inclusion']
        result['posterior_inclusion'][begin:end]=post['family_inclusion']
        if end%240==0 or end==n:print('NATIVE_STRAND_EVIDENCE',end,'/',n,'seconds',round(time.monotonic()-started,1),flush=True)
    np.savez_compressed(output/'native_evidence.npz',unit_ids=np.asarray(data['unit_ids']),strands=data['strands'],folds=data['folds'],centers=kernel.centers,**result)
    return result


def compare_populations(output,data,kernel,evidence,args):
    grid,weights=quadrature(args.quadrature_size);posterior_grid=np.r_[0.,grid]
    records=[];all_mass=np.zeros((2,kernel.f,len(posterior_grid)))
    for f,center in enumerate(kernel.centers):
        if data.get('progress'):data['progress']('comparability',f'Population comparison {f+1}/{kernel.f}')
        fits={};summaries={}
        for j,strand in enumerate(('CT','GA')):
            usable=(data['strands']==strand)&evidence['available'][:,f]&(evidence['direct_expected_kl'][:,f]>1e-8)
            fit=prevalence_posterior(evidence['log_bf'][usable,f],grid,weights);fits[strand]=fit
            all_mass[j,f]=fit['mass']
            summaries[strand]={k:v for k,v in fit.items() if k not in ('mass','grid')}
            summaries[strand].update(mean_direct_expected_kl=float(evidence['direct_expected_kl'][usable,f].mean()) if usable.any() else 0.,
                total_direct_expected_kl=float(evidence['direct_expected_kl'][usable,f].sum()),
                saturated_factors=int(evidence['numerically_saturated'][usable,f].sum()))
        comparison=compare_strands(fits['CT'],fits['GA'],posterior_grid,existence_floor=args.existence_floor,
            equivalence_margin=args.equivalence_margin,maximum_population_ci_width=args.maximum_population_ci_width,
            minimum_units=args.minimum_population_units,q_cap=args.q_cap)
        sensitivity={str(margin):compare_strands(fits['CT'],fits['GA'],posterior_grid,existence_floor=args.existence_floor,
            equivalence_margin=margin,maximum_population_ci_width=args.maximum_population_ci_width,
            minimum_units=args.minimum_population_units,q_cap=args.q_cap)['comparative_Q_model']
            for margin in (.05,.10,.20)}
        records.append(dict(family=data['family_ids'][f],family_index=f,consensus=center.tolist(),strands=summaries,
                            **comparison,margin_sensitivity=sensitivity))
    np.savez_compressed(output/'conditional_population_posteriors.npz',grid=posterior_grid,slab_quadrature_weights=weights,mass=all_mass,
                        families=np.asarray([r['family'] for r in records]))
    write_json(output/'family_scores.preview.json',records)
    print('POPULATION_COMPARISON_READY',len(records),'families',flush=True)
    return records


def simulation_power(output,data,kernel,eta,q,evidence,records,args):
    """Bounded exact-DAG conditional simulations; all families, no read filtering by calls.

This is a model-power preview over deterministic sampled real opportunity/
efficiency/exposure contexts. It is NOT empirical calibration or a population
CI. All real units entered the evidence fit above. Insufficient simulation
contexts remain explicit; the simulator never substitutes raw call overlap.
"""
    rects=projection_rectangles(kernel);details=[];started=time.monotonic()
    for f,record in enumerate(records):
        if data.get('progress'):data['progress']('comparability_power',f'Exact configuration power {f+1}/{kernel.f}')
        power={}
        for strand in ('CT','GA'):
            good=np.flatnonzero((data['strands']==strand)&evidence['available'][:,f]&(evidence['direct_expected_kl'][:,f]>1e-8))
            # Balance the two geometry-training folds, not observed hit patterns.
            chosen=[]
            for fold in (0,1):
                pool=[int(m) for m in good if data['folds'][m]==fold]
                pool.sort(key=lambda m:hashlib.sha256(f'cq-context-v1|{f}|{data["unit_ids"][m]}'.encode()).hexdigest())
                chosen+=pool[:(args.simulation_units+1-fold)//2]
            positives=[];negatives=[]
            for m in chosen:
                u=data['units'][m];fold=int(data['folds'][m])
                _,physical,adjustment=physical_model([u],np.asarray([fold]),kernel,q,rects)
                w=eta[kernel.gf]+kernel.logq+adjustment[0]
                for present in (False,True):
                    seed=int(hashlib.sha256(f'cq-simulation-v1|{f}|{u["unit_id"]}|{present}'.encode()).hexdigest()[:8],16)
                    values,selected,possible=sample_conditional_configurations(kernel.offsets,kernel.dest,kernel.edge_geo,
                        kernel.ga,kernel.gb,kernel.gf,w,physical[0],kernel.n_nodes,kernel.f,f,present,
                        data['p_accessible'][m],data['p_protected'][m],data['observed'][m],args.draws_per_state,seed)
                    if not possible:continue
                    assert np.all((selected[:,f]>=0)==present)
                    post=kernel.evaluate(values,eta,geometry_allowed=np.broadcast_to(physical, (len(values),len(kernel.ga))),
                                         geometry_log_adjustment=np.broadcast_to(adjustment,(len(values),len(kernel.ga))))
                    bf,valid,_=inclusion_log_bf(post['family_inclusion'][:,f],evidence['prior_inclusion'][m,f])
                    assert valid.all()
                    (positives if present else negatives).extend(bf.tolist())
            pos=np.asarray(positives);neg=np.asarray(negatives);threshold=np.log(args.evidence_odds)
            tp=int((pos>=threshold).sum());fp=int((neg>=threshold).sum())
            lo,hi=wilson_interval(tp,len(pos));flo,fhi=wilson_interval(fp,len(neg))
            power[strand]=dict(context_units=len(chosen),positive_draws=len(pos),negative_draws=len(neg),
                detection_probability=tp/len(pos) if len(pos) else None,detection_mc_lower95=lo,detection_mc_upper95=hi,
                false_positive_fraction=fp/len(neg) if len(neg) else None,false_positive_mc_lower95=flo,false_positive_mc_upper95=fhi,
                evidence_odds=args.evidence_odds,model_markov_false_positive_bound=1/args.evidence_odds,
                molecule_quantification_usable=bool(lo is not None and lo>=args.minimum_detection_power))
            details.append(dict(family=record['family'],strand=strand,context_unit_ids=[data['unit_ids'][m] for m in chosen],
                                positive_log_bf=positives,negative_log_bf=negatives,**power[strand]))
        record['molecule_power_preview']=power
        record['molecule_quantification_mask']={s:power[s]['molecule_quantification_usable'] for s in power}
        if (f+1)%10==0 or f+1==len(records):
            print('FULL_MODEL_POWER',f+1,'/',len(records),'families','seconds',round(time.monotonic()-started,1),flush=True)
    write_json(output/'simulation_power_details.json.gz',details)
