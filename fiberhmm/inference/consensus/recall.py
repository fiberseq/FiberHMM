from __future__ import annotations

import time
import hashlib
from collections import Counter, defaultdict
import numpy as np
from .artifacts import write_json
from .. import strand_boundary_normalization as sbn
from .adapter import read_adapter
from .geometry import representative_geometries
from .nomination import family_availability
from .recall_action import decode_posterior_accuracy
from .rescue_geometry import fit_geometry_mixture, projected_typicality, population_predictive, arbitrary_gap_predictive, rescue_decision

def projection_rectangles(kernel):
    """All actual integer aliases of each union projection, before obstacles."""
    pos=kernel.positions;x=kernel.ambiguity_bp;c=kernel.centers[kernel.gf]
    ga,gb=kernel.ga,kernel.gb
    sl=np.maximum(c[:,0]-x,np.where(ga==0,-10**12,pos[np.maximum(ga-1,0)]+1))
    sh=np.minimum(c[:,0]+x,np.where(ga==len(pos),10**12,pos[np.minimum(ga,len(pos)-1)]))
    el=np.maximum(c[:,1]-x,np.where(gb==0,-10**12,pos[np.maximum(gb-1,0)]+1))
    eh=np.minimum(c[:,1]+x,np.where(gb==len(pos),10**12,pos[np.minimum(gb,len(pos)-1)]))
    if np.any(sl>sh) or np.any(el>eh):raise AssertionError('Empty projection rectangle')
    return np.c_[sl,sh,el,eh]


def exposure(rectangles,centers_by_geometry,domains):
    """Integrate lost integer prior mass exactly; never donate it to a sliver."""
    sl,sh,el,eh=rectangles.T;den=(sh-sl+1)*(eh-el+1)
    counts=np.zeros(len(den));coordinates=np.zeros((len(den),2),dtype=int)
    for lo,hi in domains:
        al,ah=np.maximum(sl,lo),np.minimum(sh,hi-1)
        bl,bh=np.maximum(el,lo+1),np.minimum(eh,hi)
        num=np.maximum(0,ah-al+1)*np.maximum(0,bh-bl+1)
        live=num>0
        if np.any(live & (counts>0)):
            raise AssertionError('An observed nonempty projection cannot occupy two disjoint free domains')
        counts+=num
        coordinates[live,0]=np.clip(centers_by_geometry[live,0],al[live],ah[live])
        coordinates[live,1]=np.clip(centers_by_geometry[live,1],bl[live],bh[live])
    return counts/den,coordinates


def adequacy_mask(read,kernel,physical,llrs,coordinates,maximum_diffuse_odds=100.):
    """Native diffuse-core veto on EVERY positive physical projection, once."""
    allowed=physical.copy()
    positive=np.flatnonzero(physical & (llrs>0))
    if not len(positive):return allowed,0
    projection=np.searchsorted(read.positions,coordinates[positive])
    unique,inverse=np.unique(projection,axis=0,return_inverse=True)
    veto=np.zeros(len(unique),bool)
    for j,(a,b) in enumerate(unique):
        if b-a<3:raise AssertionError('Recall requires actual recipient opportunities')
        stats=sbn.endpoint_pattern_evidence(read,int(read.positions[a]),int(read.positions[b-1])+1)
        veto[j]=(stats['protected_log_likelihood']-stats['diffuse_log_likelihood'] < -np.log(maximum_diffuse_odds))
    allowed[positive[veto[inverse]]]=False
    return allowed,int(veto.sum())


def source_recipe(source_membership,eligible,strands,folds,recipient_strand,recipient_fold,mode,scale,minimum_source_units=3):
    source=(folds!=recipient_fold)
    if mode=='opposite_strand':source &= strands!=recipient_strand
    elif mode=='same_strand':source &= strands==recipient_strand
    elif mode!='pooled':raise ValueError(mode)
    counts=((source_membership[source]>=.5)&eligible[source]).sum(0)
    denominators=eligible[source].sum(0)
    activity=scale*np.divide(counts,denominators,out=np.zeros(len(counts)),where=denominators>0)
    admitted=counts>=minimum_source_units
    eta=np.log(np.maximum(activity,1e-12))
    return eta,admitted,dict(source_units=int(source.sum()),counts=counts.tolist(),
        eligible=denominators.tolist(),activity=activity.tolist(),admitted=admitted.tolist())


def score_group(kernel,data,indices,eta,admitted,rects,thresholds,rng,null_replicates,decoder):
    records=[];null_counts=[Counter() for _ in range(null_replicates)];vetoes=0
    gc=kernel.centers[kernel.gf]
    for first in range(0,len(indices),24):
        if data.get('progress'): data['progress']('rescue', f'Scoring recall units {first}/{len(indices)}')
        ix=indices[first:first+24];units=[data['units'][i] for i in ix]
        reads=[read_adapter(u) for u in units]
        values=data['log_lr'][ix]
        lp=np.c_[np.zeros(len(ix)),np.cumsum(values,axis=1)]
        op=np.c_[np.zeros(len(ix)),np.cumsum(data['observed'][ix],axis=1)]
        hp=np.c_[np.zeros(len(ix)),np.cumsum(data['hits'][ix]==1,axis=1)]
        llrs=lp[:,kernel.gb]-lp[:,kernel.ga];opps=op[:,kernel.gb]-op[:,kernel.ga]
        hit=hp[:,kernel.gb]-hp[:,kernel.ga]
        fraction=[];coords=[]
        for read,u in zip(reads,units):
            msps=[(max(a,data['start']),min(b,data['end'])) for a,b in u['msp_intervals'] if a<data['end'] and b>data['start']]
            domains=sbn._free_domains(read,msps,u['representative_raw_tf_intervals'])
            f,c=exposure(rects,gc,domains);fraction.append(f);coords.append(c)
        fraction=np.asarray(fraction);physical=(fraction>0)&(opps>=3)
        adjustment=np.log(np.maximum(fraction,1e-300))
        allowed=np.tile(admitted,(len(ix),1))
        prior=kernel.evaluate(np.zeros_like(values),eta,allowed=allowed,geometry_allowed=physical,
            geometry_log_adjustment=adjustment,export_geometry=True)
        data_mask=physical.copy()
        for m,read in enumerate(reads):
            data_mask[m],v=adequacy_mask(read,kernel,physical[m],llrs[m],coords[m],data['rescue_options'].maximum_diffuse_odds);vetoes+=v
        post=kernel.evaluate(values,eta,allowed=allowed,geometry_allowed=data_mask,
            geometry_log_adjustment=adjustment,export_geometry=True)
        if decoder=='geometry_map':
            decode=kernel.map_configuration(values,eta,allowed=allowed,geometry_allowed=data_mask&(llrs>0),
                geometry_log_adjustment=adjustment)['geometry_by_family']
        else:
            decode=decode_posterior_accuracy(kernel,values,data['observed'][ix],post,eta,allowed,
                data_mask&(llrs>0),adjustment)['geometry_by_family']
        for m,u in enumerate(units):
            profiles=[]
            for result in (post,prior):
                diff=np.zeros(kernel.k+1)
                np.add.at(diff,kernel.ga,result['geometry_mass'][m]);np.add.at(diff,kernel.gb,-result['geometry_mass'][m])
                profiles.append(np.cumsum(diff)[:-1])
            calls=[]
            for family in np.flatnonzero(decode[m]>=0):
                g=int(decode[m,family]);a,b=map(int,coords[m][g])
                qa,qb=np.searchsorted(reads[m].positions,[a,b]);anchor=int(reads[m].positions[(qa+qb-1)//2])
                aj=int(np.searchsorted(kernel.positions,anchor))
                event=float(np.clip(profiles[0][aj],0,1));prior_event=float(np.clip(profiles[1][aj],0,1))
                if event<min(thresholds):continue
                calls.append(dict(family=data['family_ids'][family],family_index=int(family),interval=[a,b],consensus=kernel.centers[family].tolist(),
                    native_log_lr=float(llrs[m,g]),opportunities=int(opps[m,g]),hits=int(hit[m,g]),
                    protection_event_mass=event,family_inclusion_mass=float(post['family_inclusion'][m,family]),
                    geometry_inclusion_mass=float(post['geometry_mass'][m,g]),
                    prior_only_protection_event_mass=prior_event,anchor=anchor))
            calls.sort(key=lambda c:c['interval'])
            if any(a['interval'][1]>b['interval'][0] for a,b in zip(calls,calls[1:])):raise AssertionError('Overlapping recall actions')
            records.append(dict(unit_id=u['unit_id'],strand=u['strand'],calls=calls))
        for rep in range(null_replicates):
            null_values=np.zeros_like(values);nr=[]
            for m,u in enumerate(units):
                hh=(rng.random(len(u['positions']))<np.asarray(u['p_accessible'])).astype(np.int8)
                read=read_adapter(u,hh);nr.append(read)
                keep=(read.positions>=data['start'])&(read.positions<data['end'])
                js=np.searchsorted(kernel.positions,read.positions[keep]);null_values[m,js]=np.diff(read.prefix)[keep]
            nlp=np.c_[np.zeros(len(ix)),np.cumsum(null_values,axis=1)]
            nl=nlp[:,kernel.gb]-nlp[:,kernel.ga];mask=physical.copy()
            for m,read in enumerate(nr):mask[m],_=adequacy_mask(read,kernel,physical[m],nl[m],coords[m],data['rescue_options'].maximum_diffuse_odds)
            ns=kernel.evaluate(null_values,eta,allowed=allowed,geometry_allowed=mask,
                geometry_log_adjustment=adjustment,export_geometry=True)
            if decoder=='geometry_map':
                nd=kernel.map_configuration(null_values,eta,allowed=allowed,geometry_allowed=mask&(nl>0),
                    geometry_log_adjustment=adjustment)['geometry_by_family']
            else:
                nd=decode_posterior_accuracy(kernel,null_values,data['observed'][ix],ns,eta,allowed,
                    mask&(nl>0),adjustment)['geometry_by_family']
            for m in range(len(ix)):
                diff=np.zeros(kernel.k+1);np.add.at(diff,kernel.ga,ns['geometry_mass'][m]);np.add.at(diff,kernel.gb,-ns['geometry_mass'][m])
                profile=np.cumsum(diff)[:-1]
                for f in np.flatnonzero(nd[m]>=0):
                    g=nd[m,f];a,b=coords[m][g];qa,qb=np.searchsorted(nr[m].positions,[a,b]);anchor=nr[m].positions[(qa+qb-1)//2]
                    event=profile[np.searchsorted(kernel.positions,anchor)]
                    for threshold in thresholds:
                        if event>=threshold:null_counts[rep][str(threshold)]+=1
    return records,null_counts,vetoes

def source_geometry_model(output,data,kernel,members,eligible,native_eta,originals,prior_units):
    """Called-member conditional geometry likelihood retains all native competitors.

    Physical source exposure is an explicit outcome-free model restriction in
    this auxiliary fit. Original CR outputs themselves are never re-evaluated or
    overwritten. Fold groups, including both strands, are excluded together.
    """
    start_time=time.monotonic();coordinates=representative_geometries(kernel)
    rects=projection_rectangles(kernel);gcenters=kernel.centers[kernel.gf]
    base_q=np.exp(kernel.logq)
    gsets=[np.flatnonzero(kernel.gf==f) for f in range(kernel.f)]
    called=[originals[u['unit_id']]['source_calls'] for u in data['units']]
    available=family_availability([{**u,'representative_raw_tf_intervals':cc}
                                  for u,cc in zip(data['units'],called)],kernel.centers,kernel.ambiguity_bp)
    records=defaultdict(list)
    for begin in range(0,len(data['units']),24):
        if data.get('progress'): data['progress']('source_geometry', f'Fitting source geometry {begin}/{len(data["units"])}')
        end=min(begin+24,len(data['units']));values=data['log_lr'][begin:end]
        op=np.c_[np.zeros(end-begin),np.cumsum(data['observed'][begin:end],axis=1)]
        opp=op[:,kernel.gb]-op[:,kernel.ga]
        fraction=np.asarray([exposure(rects,gcenters,free_domains(u,[],block_native=False))[0] for u in data['units'][begin:end]])
        physical=(fraction>0)&(opp>=1)
        adjustment=np.log(np.maximum(fraction,1e-300))
        prior=kernel.evaluate(np.zeros_like(values),native_eta,allowed=available[begin:end],geometry_allowed=physical,
                              geometry_log_adjustment=adjustment,export_geometry=True)
        post=kernel.evaluate(values,native_eta,allowed=available[begin:end],geometry_allowed=physical,
                             geometry_log_adjustment=adjustment,export_geometry=True)
        for j in range(end-begin):
            m=begin+j
            for f in np.flatnonzero((members[m]>=.5)&eligible[m]):
                gs=gsets[f];mass=post['geometry_mass'][j,gs];pmass=prior['geometry_mass'][j,gs]
                if mass.sum()>0 and pmass.sum()>0:
                    records[int(f)].append((m,(mass/mass.sum()).astype(np.float64),float(members[m,f]),
                                           (pmass/pmass.sum()).astype(np.float64)))
        if end%240==0 or end==len(data['units']):
            print('SOURCE_GEOMETRY',end,'units',round(time.monotonic()-start_time,1),'seconds',flush=True)
    q=np.tile(base_q,(2,1));counts=np.zeros((2,kernel.f),int);fit_rows=[];training_members=[]
    for held_fold in (0,1):
        for f,gs in enumerate(gsets):
            if data.get('progress'): data['progress']('source_geometry', None)
            rr=[r for r in records[f] if data['folds'][r[0]]!=held_fold]
            posterior=np.asarray([r[1] for r in rr]).reshape(-1,len(gs))
            weights=np.asarray([r[2] for r in rr])
            conditional_prior=np.asarray([r[3] for r in rr]).reshape(-1,len(gs))
            fitted,diagnostics=fit_geometry_mixture(posterior,base_q[gs],weights,prior_units=prior_units,
                                                   conditional_prior=conditional_prior)
            q[held_fold,gs]=fitted;counts[held_fold,f]=len(rr)
            fit_rows.append(dict(held_fold=held_fold,family=data['family_ids'][f],**diagnostics))
            training_members.append(dict(held_fold=held_fold,family=data['family_ids'][f],
                unit_ids=[data['unit_ids'][r[0]] for r in rr]))
    np.savez_compressed(output/'population_geometry.npz',q=q,source_counts=counts,base_q=base_q,
        geometry_family=kernel.gf,geometry_start=kernel.ga,geometry_end=kernel.gb,
        centers=kernel.centers,unit_ids=np.asarray(data['unit_ids']),folds=data['folds'])
    write_json(output/'geometry_fit_diagnostics.json',fit_rows)
    write_json(output/'geometry_training_members.json.gz',training_members)
    print('POPULATION_GEOMETRY_READY','fits',len(fit_rows),'unconverged',sum(not r['converged'] for r in fit_rows),
          'seconds',round(time.monotonic()-start_time,1),flush=True)
    return q,counts


def free_domains(u,block_calls,*,block_native=True):
    read=read_adapter(u)
    # Training/reference members MUST be allowed to occupy their own existing
    # footprint. For de novo recall, raw calls and normalized calls both remain
    # obstacles. The adapter includes raw calls even when extra obstacles=[];
    # make this choice explicit rather than accidentally training on flanks.
    if not block_native:
        read.calls=[]
    msps=[(max(a,u['_region'][0]),min(b,u['_region'][1])) for a,b in u['msp_intervals'] if a<u['_region'][1] and b>u['_region'][0]]
    return sbn._free_domains(read,msps,block_calls)


def native_steps(u):
    h=np.asarray(u['hits']);pa=np.asarray(u['p_accessible']);pp=np.asarray(u['p_protected'])
    return np.where(h,np.log(pp/pa),np.log1p(-pp)-np.log1p(-pa))


def shape_record(u,call,f,kernel,rects,coordinates,q,domains,source_count,family_mass):
    gs=np.flatnonzero(kernel.gf==f);localq=q[gs]
    fraction,physical_coords=exposure(rects[gs],np.tile(kernel.centers[f],(len(gs),1)),domains)
    pos=np.asarray(u['positions']);steps=native_steps(u);prefix=np.r_[0.,np.cumsum(steps)]
    projection=np.searchsorted(pos,coordinates[gs]);opp=projection[:,1]-projection[:,0]
    llr=prefix[projection[:,1]]-prefix[projection[:,0]]
    result=population_predictive(llr,opp,fraction,localq)
    result.update(projected_typicality(pos,coordinates[gs],localq,call['interval']))
    a,b=kernel.centers[f]
    result.update(arbitrary_gap_log_bf=arbitrary_gap_predictive(pos,steps,(a-kernel.ambiguity_bp,b+kernel.ambiguity_bp),domains),
                  source_geometry_units=int(source_count),updated_family_inclusion_mass=float(family_mass),
                  selected_width_bp=int(call['interval'][1]-call['interval'][0]))
    result['population_minus_gap_log_bf']=(None if result['observable_log_bf'] is None or result['arbitrary_gap_log_bf'] is None
                                         else result['observable_log_bf']-result['arbitrary_gap_log_bf'])
    return result


def score_candidates(output,data,kernel,members,eligible,originals,candidates,q,counts):
    rects=projection_rectangles(kernel);coordinates=representative_geometries(kernel)
    gcenters=kernel.centers[kernel.gf];ledger=[];recipes=[];started=time.monotonic()
    for strand in sorted(set(data['strands'])):
        for held_fold in (0,1):
            indices=np.flatnonzero((data['strands']==strand)&(data['folds']==held_fold))
            eta,admitted,recipe=source_recipe(members,eligible,data['strands'],data['folds'],strand,held_fold,data['rescue_options'].source_mode,data['rescue_options'].prior_scale, minimum_source_units=data['rescue_options'].minimum_source_units)
            recipes.append(dict(strand=strand,held_fold=held_fold,**recipe))
            shape_adjustment=np.log(q[held_fold])-kernel.logq
            for begin in range(0,len(indices),24):
                if data.get('progress'): data['progress']('rescue_geometry', f'Validating {strand} fold {held_fold}: {begin}/{len(indices)}')
                ix=indices[begin:begin+24];values=data['log_lr'][ix];units=[data['units'][m] for m in ix]
                lp=np.c_[np.zeros(len(ix)),np.cumsum(values,axis=1)]
                op=np.c_[np.zeros(len(ix)),np.cumsum(data['observed'][ix],axis=1)]
                llr=lp[:,kernel.gb]-lp[:,kernel.ga];opp=op[:,kernel.gb]-op[:,kernel.ga]
                fractions=[];coords=[];domains=[]
                for u in units:
                    domain=free_domains(u,originals[u['unit_id']]['source_calls']);domains.append(domain)
                    fraction,representative=exposure(rects,gcenters,domain)
                    fractions.append(fraction);coords.append(representative)
                fractions=np.asarray(fractions);physical=(fractions>0)&(opp>=3)
                adjustment=np.log(np.maximum(fractions,1e-300))+shape_adjustment
                allowed=np.tile(admitted,(len(ix),1));mask=physical.copy()
                for j,u in enumerate(units):mask[j],_=adequacy_mask(read_adapter(u),kernel,physical[j],llr[j],coords[j],data['rescue_options'].maximum_diffuse_odds)
                # Same learned q and outcome-free exposure in the two partitions.
                # The recipient-data adequacy veto applies only once, in data.
                prior=kernel.evaluate(np.zeros_like(values),eta,allowed=allowed,geometry_allowed=physical,
                                      geometry_log_adjustment=adjustment)
                post=kernel.evaluate(values,eta,allowed=allowed,geometry_allowed=mask,
                                     geometry_log_adjustment=adjustment)
                for j,m in enumerate(ix):
                    u=units[j];cc=[]
                    for call in candidates[u['unit_id']]['calls']:
                        f=call['family_index']
                        evidence=shape_record(u,call,f,kernel,rects,coordinates,q[held_fold],domains[j],counts[held_fold,f],
                                              post['family_inclusion'][j,f])
                        decisions={name:rescue_decision(evidence,credible_mass=cm,loss_odds=odds,minimum_probability=data['rescue_options'].minimum_probability,
                                                       minimum_source_units=data['rescue_options'].minimum_source_units)
                                   for name,(cm,odds) in {'selected':(data['rescue_options'].credible_mass,data['rescue_options'].loss_odds)}.items()}
                        cc.append(dict(**call,geometry_validation=evidence,geometry_decisions=decisions))
                    ledger.append(dict(unit_id=u['unit_id'],strand=strand,held_fold=held_fold,calls=cc,
                        whole_region_learned_geometry_log_predictive=float(post['log_partition'][j]-prior['log_partition'][j])))
            print('STRICT_RECALL_SCORED',strand,held_fold,len(indices),'units',round(time.monotonic()-started,1),'seconds',flush=True)
    write_json(output/'candidate_geometry_ledger.json.gz',ledger);write_json(output/'source_activity_recipes.json',recipes)
    return ledger
