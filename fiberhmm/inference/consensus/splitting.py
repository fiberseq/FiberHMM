from __future__ import annotations
import numpy as np
from numba import njit

def conditional_model_tables(model):
    # Same exact conditional-emission equation as tf_recaller, without importing
    # its unrelated BAM/UI modules into a dependency-light analysis process.
    ep=np.asarray(model.emissionprob_,dtype=float)
    if ep.shape[0]!=2 or ep.shape[1]<8193:raise ValueError('Complete two-state context emissions required')
    hit=np.clip(ep[:,:4096],0,None);miss=np.clip(ep[:,4097:8193],0,None)
    conditional=np.clip(hit/np.maximum(hit+miss,1e-12),1e-12,1-1e-12)
    return conditional[0],conditional[1]


@njit(cache=True)
def _dp(starts,ends,weights,eo,so,previous,following):
    n=len(weights);fw=np.zeros(n+1);bw=np.zeros(n+1)
    best=np.zeros(n+1);take=np.zeros(n,dtype=np.bool_)
    for j in range(n):
        i=eo[j];fw[j+1]=np.logaddexp(fw[j],weights[i]+fw[previous[i]])
        value=weights[i]+best[previous[i]]
        if value>best[j]+1e-12:best[j+1]=value;take[j]=True
        else:best[j+1]=best[j]
    for j in range(n-1,-1,-1):
        i=so[j];bw[j]=np.logaddexp(bw[j+1],weights[i]+bw[following[i]])
    inc=np.exp(weights+fw[previous]+bw[following]-fw[-1]);chosen=np.zeros(n,dtype=np.bool_);j=n
    while j:
        if take[j-1]:i=eo[j-1];chosen[i]=True;j=previous[i]
        else:j-=1
    return fw[-1],inc,chosen


def geometry(positions,span,families,x,max_gap):
    """Nomination uses only population TF edges, native span and opportunities."""
    a,b=span;raw=[]
    for fam in families:
        for edge,side in [(fam['consensus_start'],-1),(fam['consensus_end'],1)]:
            if edge+x<=a or edge-x>=b:continue
            cut,width=np.meshgrid(np.arange(edge-x,edge+x+1),np.arange(1,max_gap+1),indexing='ij')
            lo=cut-width if side<0 else cut;hi=cut if side<0 else cut+width
            keep=(lo>a)&(hi<b)
            raw.extend(zip(lo[keep].tolist(),hi[keep].tolist()))
    if not raw:return None
    raw=np.unique(np.asarray(raw,dtype=int),axis=0)
    proj=np.searchsorted(positions,raw)
    physical=(proj[:,0]>=3)&(proj[:,1]<=len(positions)-3)&(proj[:,1]>proj[:,0])
    if not physical.any():return None
    visible=raw[physical];p=proj[physical]
    unique,inverse,counts=np.unique(p,axis=0,return_inverse=True,return_counts=True)
    coords=np.zeros_like(unique)
    for j in range(len(unique)):
        choices=visible[inverse==j];coords[j]=choices[len(choices)//2]
    starts,ends=unique.T;eo=np.lexsort((starts,ends));so=np.lexsort((ends,starts))
    previous=np.searchsorted(ends[eo],starts-1,side='right')
    following=np.searchsorted(starts[so],ends+1,side='left')
    return dict(starts=starts,ends=ends,eo=eo,so=so,previous=previous,following=following,
        q=counts/counts.sum(),coordinates=coords,raw_candidates=len(raw),
        observable_integer_candidates=int(physical.sum()))


def evaluate(g,gain,activity):
    args=[g[k] for k in ('starts','ends')]
    order=[g[k] for k in ('eo','so','previous','following')]
    prior_weight=np.log(activity*g['q'])
    zp,_,_=_dp(*args,prior_weight,*order)
    zd,inc,_=_dp(*args,prior_weight+gain,*order)
    # Conditional nonempty split-model BF, with its own exact prior normalizer.
    log_nonempty_data=zd+np.log(-np.expm1(-zd)) if zd>0 else -np.inf
    log_nonempty_prior=zp+np.log(-np.expm1(-zp))
    profile=np.zeros(int(g['ends'].max())+1)
    np.add.at(profile,g['starts'],inc);np.add.at(profile,g['ends'],-inc)
    coverage=np.cumsum(profile)[:-1];utility=np.r_[0.,np.cumsum(2*coverage-1)]
    candidate_utility=utility[g['ends']]-utility[g['starts']]
    # Native-positive separator firewall in the action, not the prior.
    candidate_utility=np.where(gain>0,candidate_utility,-np.inf)
    _,_,selected=_dp(*args,candidate_utility,*order)
    return dict(log_bf_any_split=float(log_nonempty_data-log_nonempty_prior),
        posterior_any_gap=float(-np.expm1(-zd)),prior_any_gap=float(-np.expm1(-zp)),
        selected=np.flatnonzero(selected),geometry_mass=inc)


def summarize_action(result,g,span,positions,hits,gain,families,x,tf_steps,
                     minimum_separator_opportunities=3,minimum_separator_bf=10.):
    gaps=[]
    for i in result['selected']:
        ga,gb=g['starts'][i],g['ends'][i];a,b=g['coordinates'][i]
        gaps.append(dict(interval=[int(a),int(b)],opportunities=int(gb-ga),hits=int(hits[ga:gb].sum()),
            native_accessible_vs_protected_log_lr=float(gain[i]),geometry_mass=float(result['geometry_mass'][i])))
    gaps.sort(key=lambda c:c['interval'])
    pieces=[];at=span[0]
    for gap in gaps:
        pieces.append([at,gap['interval'][0]]);at=gap['interval'][1]
    pieces.append([at,span[1]])
    matches=[];geometry_only=[]
    for piece in pieces:
        qa,qb=np.searchsorted(positions,piece)
        if qb-qa<3:continue
        for f in families:
            if abs(piece[0]-f['consensus_start'])<=x and abs(piece[1]-f['consensus_end'])<=x:
                record=dict(piece=piece,family=f['family'],source_units=f['source_units'],
                    native_TF_protected_vs_accessible_log_lr=float(tf_steps[qa:qb].sum()),
                    opportunities=int(qb-qa),hits=int(hits[qa:qb].sum()))
                geometry_only.append(record)
                if record['native_TF_protected_vs_accessible_log_lr']>0:matches.append(record)
    qualified=(bool(matches) and bool(gaps) and all(c['opportunities']>=minimum_separator_opportunities and
        c['native_accessible_vs_protected_log_lr']>=np.log(minimum_separator_bf) for c in gaps))
    return dict(gaps=gaps,pieces=pieces,matching_CR_pieces=matches,geometry_only_CR_pieces=geometry_only,
        strong_separator_and_CR_geometry=qualified,
        summed_internal_separator_log_lr=float(sum(c['native_accessible_vs_protected_log_lr'] for c in gaps)))
