# Numerical kernel promoted from the validated September 2026 consensus experiments.
"""Native-model, family-level strand comparability; never a call-count balance score.

Family include/exclude likelihoods come from the SAME complete configuration
model with fixed competitor priors. Scalar prevalence fits are conditional
diagnostics, not a jointly fitted occupancy distribution across all families.
Q_model is a capped Phred transform of stated model probabilities, NOT an
empirically calibrated quality or FDR. Missing/underpowered is not Q=0.
"""
from __future__ import annotations

import math
import numpy as np
from numba import njit
from scipy.special import logsumexp,expit
from numpy.polynomial.legendre import leggauss
from scipy.interpolate import PchipInterpolator


def inclusion_log_bf(posterior, prior, numerical_epsilon=1e-12):
    """log[(Z_include(data)/Z_include(prior))/(Z_exclude(data)/Z_exclude(prior))].

The common accessible observation baseline cancels. Removing the inclusion
prior odds is essential. Fixed competitor priors remain inside both conditional
likelihoods. Saturated floating-point marginals are capped and explicitly marked;
this is not a statistical evidence cutoff. Prior-impossible events are unavailable.
"""
    posterior,prior=np.broadcast_arrays(np.asarray(posterior,float),np.asarray(prior,float))
    if np.any(~np.isfinite(posterior)) or np.any(~np.isfinite(prior)):
        raise ValueError('Finite marginals required')
    if np.any((posterior < -1e-7)|(posterior > 1+1e-7)|(prior < -1e-7)|(prior > 1+1e-7)):
        raise ValueError('Marginals must lie in [0,1]')
    available=(prior>numerical_epsilon)&(prior<1-numerical_epsilon)
    saturated=available&((posterior<numerical_epsilon)|(posterior>1-numerical_epsilon))
    p=np.clip(posterior,numerical_epsilon,1-numerical_epsilon)
    p0=np.clip(prior,numerical_epsilon,1-numerical_epsilon)
    value=(np.log(p)-np.log1p(-p))-(np.log(p0)-np.log1p(-p0))
    return np.where(available,value,0.),available,saturated


def prevalence_posterior(log_bf, grid, quadrature_weights):
    """Spike/slab population model: prior .5 at pi=0, .5 uniform Beta(1,1).

Every unit contributes one factor, including uncalled units and neutral BF=1.
The include/exclude likelihoods must use the same base measure and competitor
prior. No selected-molecule weights or rescue counts enter this fit.
"""
    values=np.asarray(log_bf,float);grid=np.asarray(grid,float);qw=np.asarray(quadrature_weights,float)
    if values.ndim!=1 or np.any(~np.isfinite(values)):
        raise ValueError('One finite BF per evidence unit required')
    if grid.shape!=qw.shape or np.any((grid<=0)|(grid>=1)) or np.any(qw<=0):
        raise ValueError('Interior quadrature nodes with positive weights required')
    ll=np.zeros(len(grid))
    for start in range(0,len(values),256):
        ll+=np.logaddexp(np.log1p(-grid)[None,:],np.log(grid)[None,:]+values[start:start+256,None]).sum(0)
    lp=ll+np.log(qw);log_evidence=float(logsumexp(lp)-np.log(qw.sum()))
    # A continuous prior alone would call pi>.01 with 99% prior probability
    # even on a completely blind strand. Retain an explicit no-family model.
    posterior_grid=np.r_[0.,grid]
    mass=np.r_[expit(-log_evidence),expit(log_evidence)*np.exp(lp-logsumexp(lp))]
    # Quadrature weights represent continuous mass, not atoms at every node.
    # Half-weight CDF nodes remove the otherwise first-order quantile bias.
    cumulative=np.r_[mass[0],mass[0]+np.cumsum(mass[1:])-.5*mass[1:],1.]
    quantiles=np.interp([.025,.5,.975],cumulative,np.r_[0.,grid,1.])
    return dict(mass=mass,grid=posterior_grid,log_marginal_bf=log_evidence,
                mean=float(mass@posterior_grid),median=float(quantiles[1]),
                lower95=float(quantiles[0]),upper95=float(quantiles[2]),
                width95=float(quantiles[2]-quantiles[0]),units=int(len(values)))


def quadrature(size=3072):
    x,w=leggauss(size)
    return (x+1)/2,w/2


def q_from_error(error, cap=40.):
    if not np.isfinite(error) or error < -1e-10 or error > 1+1e-10:
        raise ValueError('Probability required')
    return float(min(cap,-10*math.log10(max(float(error),10**(-cap/10)))))


def slab_cdf(grid,mass):
    """Continuous monotone CDF for a quadrature posterior's slab component.

Only grid[0]=0 is a genuine atom. Treating all quadrature nodes as atoms makes
the practical-equivalence boundary jump when numerical resolution changes.
The half-weight cumulative nodes approximate integration, not biological
smoothing. Native observations and likelihoods are never smoothed.
"""
    if grid[0]!=0 or len(grid)!=len(mass):raise ValueError('Spike followed by interior slab nodes required')
    return _slab_spline(grid,mass)


def _slab_spline(grid,mass):
    x=np.r_[0.,grid[1:],1.];total=mass[1:].sum()
    y=np.maximum.accumulate(np.clip(np.r_[0.,np.cumsum(mass[1:])-.5*mass[1:],total],0.,total))
    # Harmonic slopes safely tend to zero in underflow-scale posterior tails.
    with np.errstate(over='ignore',divide='ignore'):
        spline=PchipInterpolator(x,y,extrapolate=False)
    if not np.all(np.isfinite(spline.c)):raise ValueError('Nonfinite posterior CDF spline')
    return spline


def window_probability(first,second,grid,margin,floor=None):
    """Continuous window integration with no hard quadrature-node admission.

The interpolated CDF is piecewise cubic. Between its own and shifted knots,
the product of the first density and the second window mass has degree five.
Three-point Gauss integration is exact for that interpolant. Only pi=0 is an
atom; existence-floor and margin boundaries cannot turn grid nodes on/off.
"""
    a,b=slab_cdf(grid,first),slab_cdf(grid,second)
    lower=0. if floor is None else floor
    knots=np.unique(np.r_[lower,1.,a.x,b.x-margin,b.x+margin,lower+margin])
    knots=knots[(knots>=lower)&(knots<=1.)]
    half=np.diff(knots)/2;mid=(knots[1:]+knots[:-1])/2
    nodes=mid[:,None]+half[:,None]*np.array([-math.sqrt(3/5),0.,math.sqrt(3/5)])
    inner=np.maximum(0.,b(np.clip(nodes+margin,lower,1.))-b(np.clip(nodes-margin,lower,1.)))
    value=float(np.sum(half[:,None]*a.derivative()(nodes)*inner*np.array([5/9,8/9,5/9])))
    if floor is None:
        value+=first[0]*second[0]+first[0]*float(b(margin))+second[0]*float(a(margin))
    return float(np.clip(value,0,1))


def compare_strands(ct,ga,grid,*,existence_floor=.01,equivalence_margin=.10,
                    maximum_population_ci_width=.20,minimum_units=20,q_cap=40.):
    """Keep native existence Q, comparative Q, and quantification masks separate.

Positive comparison event: pi_CT and pi_GA exceed the existence floor AND their
absolute difference is inside the predeclared practical-equivalence margin.
An underpowered strand cannot contribute corroboration or a contradiction.
Population precision does not imply high per-molecule discrimination power.
"""
    if not 0<existence_floor<1 or not 0<equivalence_margin<1:
        raise ValueError('Interior existence floor and equivalence margin required')
    pc,pg=ct['mass'],ga['mass']
    native={s:q_from_error(p['mass'][0]+float(slab_cdf(grid,p['mass'])(existence_floor)),q_cap)
            for s,p in [('CT',ct),('GA',ga)]}
    agreement=.5*(window_probability(pc,pg,grid,equivalence_margin)+window_probability(pg,pc,grid,equivalence_margin))
    success=.5*(window_probability(pc,pg,grid,equivalence_margin,existence_floor)+
                window_probability(pg,pc,grid,equivalence_margin,existence_floor))
    error=float(np.clip(1-success,0,1))
    mask={s:bool(p['units']>=minimum_units and p['width95']<=maximum_population_ci_width)
          for s,p in [('CT',ct),('GA',ga)]}
    if all(mask.values()):
        cq=q_from_error(error,q_cap)
        if max(native.values())<=-10*math.log10(.95):
            status='no_recurrent_support_on_either_strand'
        elif agreement<=.05 and max(native.values())>=10:
            status='strand_model_discrepancy'
        elif cq>=10:
            status='two_strand_supported'
        else:
            status='two_strand_weak_or_unresolved_support'
    else:
        cq=None
        status=('CT_population_only' if mask['CT'] else 'GA_population_only' if mask['GA'] else 'population_underpowered')
    return dict(native_existence_Q_model={s:native[s] if mask[s] else None for s in mask},comparative_Q_model=cq,
                probability_within_margin=agreement if all(mask.values()) else None,
                probability_joint_support=1-error if all(mask.values()) else None,
                unqualified_model_probabilities=dict(within_margin=agreement,joint_support=1-error,
                    diagnostic_only=not all(mask.values())),
                population_quantification_mask=mask,status=status,
                equivalence_margin=equivalence_margin,existence_floor=existence_floor)


def wilson_interval(successes,total,z=1.95996398454):
    if total<=0:return (None,None)
    p=successes/total;den=1+z*z/total
    center=(p+z*z/(2*total))/den
    half=z*math.sqrt(p*(1-p)/total+z*z/(4*total*total))/den
    return max(0.,center-half),min(1.,center+half)


@njit(cache=True)
def _logadd(a,b):
    if a==-np.inf:return b
    if b==-np.inf:return a
    if a<b:a,b=b,a
    return a+math.log1p(math.exp(b-a))


@njit(cache=True)
def sample_conditional_configurations(offsets,dest,edge_geo,ga,gb,gf,weights,physical,
                                      n_nodes,nf,target,present,pa,pp,observed,draws,seed):
    """Exact full-DAG prior draws, conditioned on target inclusion or exclusion.

No independent-interval approximation and no target-only planted gap. Every
simulated unit contains compatible competing families drawn under the same
reference activities/geometry/exposure used to score its include/exclude BF.
"""
    np.random.seed(seed)
    bw=np.full((2,n_nodes),-np.inf);bw[0,n_nodes-1]=0.
    for v in range(n_nodes-2,-1,-1):
        for e in range(offsets[v],offsets[v+1]):
            g=edge_geo[e]
            if g>=0 and not physical[g]:continue
            w=0. if g<0 else weights[g]
            u=dest[e]
            if g>=0 and gf[g]==target:
                bw[1,v]=_logadd(bw[1,v],w+bw[0,u])
            else:
                bw[0,v]=_logadd(bw[0,v],w+bw[0,u])
                bw[1,v]=_logadd(bw[1,v],w+bw[1,u])
    need0=1 if present else 0
    values=np.zeros((draws,len(pa)));selected=np.full((draws,nf),-1,np.int64)
    if not np.isfinite(bw[need0,0]):
        return values,selected,False
    for m in range(draws):
        protected=np.zeros(len(pa),np.bool_);v=0;need=need0
        while v<n_nodes-1:
            pick=np.random.random();cumulative=0.;chosen=-1;next_need=need
            for e in range(offsets[v],offsets[v+1]):
                g=edge_geo[e]
                if g>=0 and not physical[g]:continue
                w=0. if g<0 else weights[g]
                target_edge=g>=0 and gf[g]==target
                if target_edge and need==0:continue
                after=0 if target_edge else need
                tail=w+bw[after,dest[e]]
                if not np.isfinite(tail):continue
                cumulative+=math.exp(tail-bw[need,v])
                chosen=e;next_need=after
                if pick<=cumulative:break
            if chosen<0:raise ValueError('Conditional sampling path has no continuation')
            g=edge_geo[chosen]
            if g>=0:
                selected[m,gf[g]]=g
                for j in range(ga[g],gb[g]):protected[j]=True
            v=dest[chosen];need=next_need
        if need!=0:raise ValueError('Conditioned inclusion was not satisfied')
        for j in range(len(pa)):
            if not observed[j]:continue
            probability=pp[j] if protected[j] else pa[j]
            hit=np.random.random()<probability
            values[m,j]=(math.log(pp[j]/pa[j]) if hit else math.log1p(-pp[j])-math.log1p(-pa[j]))
    return values,selected,True


@njit(cache=True)
def sample_conditional_family_event(offsets,dest,edge_geo,ga,gb,gf,weights,physical,
                                    n_nodes,nf,target_families,present,pa,pp,observed,draws,seed):
    """Exact full-DAG draws given ANY member of a family set, or NONE.

    The third backward state permits additional target members after the first.
    Reusing the one-family sampler would incorrectly condition on exactly one
    member when two group members can coexist. Every other competitor remains.
    """
    return sample_conditional_geometry_event(offsets,dest,edge_geo,ga,gb,gf,weights,physical,
        n_nodes,nf,target_families[gf],present,pa,pp,observed,draws,seed)


@njit(cache=True)
def sample_conditional_geometry_event(offsets,dest,edge_geo,ga,gb,gf,weights,physical,
                                      n_nodes,nf,target_geometries,present,pa,pp,observed,draws,seed):
    """Exact whole-configuration prior draws conditioned on ANY target geometry."""
    np.random.seed(seed)
    # 0: no target allowed; 1: still require a target; 2: unrestricted suffix.
    bw=np.full((3,n_nodes),-np.inf);bw[0,-1]=0.;bw[2,-1]=0.
    for v in range(n_nodes-2,-1,-1):
        for e in range(offsets[v],offsets[v+1]):
            g=edge_geo[e]
            if g>=0 and not physical[g]:continue
            w=0. if g<0 else weights[g];u=dest[e]
            target_edge=g>=0 and target_geometries[g]
            bw[2,v]=_logadd(bw[2,v],w+bw[2,u])
            if target_edge:bw[1,v]=_logadd(bw[1,v],w+bw[2,u])
            else:
                bw[0,v]=_logadd(bw[0,v],w+bw[0,u])
                bw[1,v]=_logadd(bw[1,v],w+bw[1,u])
    state0=1 if present else 0
    values=np.zeros((draws,len(pa)));selected=np.full((draws,nf),-1,np.int64)
    if not np.isfinite(bw[state0,0]):return values,selected,False
    for m in range(draws):
        protected=np.zeros(len(pa),np.bool_);v=0;state=state0
        while v<n_nodes-1:
            pick=np.random.random();cumulative=0.;chosen=-1;next_state=state
            for e in range(offsets[v],offsets[v+1]):
                g=edge_geo[e]
                if g>=0 and not physical[g]:continue
                target_edge=g>=0 and target_geometries[g]
                if state==0 and target_edge:continue
                after=2 if state==1 and target_edge else state
                w=0. if g<0 else weights[g];tail=w+bw[after,dest[e]]
                if not np.isfinite(tail):continue
                cumulative+=math.exp(tail-bw[state,v]);chosen=e;next_state=after
                if pick<=cumulative:break
            if chosen<0:raise ValueError('Conditional event sampler has no continuation')
            g=edge_geo[chosen]
            if g>=0:
                selected[m,gf[g]]=g
                for j in range(ga[g],gb[g]):protected[j]=True
            v=dest[chosen];state=next_state
        if state==1:raise ValueError('ANY-family condition was not satisfied')
        for j in range(len(pa)):
            if not observed[j]:continue
            probability=pp[j] if protected[j] else pa[j]
            hit=np.random.random()<probability
            values[m,j]=math.log(pp[j]/pa[j]) if hit else math.log1p(-pp[j])-math.log1p(-pa[j])
    return values,selected,True


@njit(cache=True)
def sample_conditional_alias_event(offsets,dest,edge_geo,ga,gb,gf,weights,physical,
                                   n_nodes,nf,event_fraction,present,pa,pp,observed,draws,seed):
    """ANY-event conditional draws retaining partially identified integer aliases.

    An event label can vary within a geometry's observationally identical alias
    class. It then changes event probability, never that geometry's emissions.
    """
    np.random.seed(seed);bw=np.full((3,n_nodes),-np.inf);bw[0,-1]=0.;bw[2,-1]=0.
    for v in range(n_nodes-2,-1,-1):
        for e in range(offsets[v],offsets[v+1]):
            g=edge_geo[e]
            if g>=0 and not physical[g]:continue
            w=0. if g<0 else weights[g];u=dest[e];p=0. if g<0 else event_fraction[g]
            bw[2,v]=_logadd(bw[2,v],w+bw[2,u])
            if p>0:bw[1,v]=_logadd(bw[1,v],w+math.log(p)+bw[2,u])
            if p<1:
                no=w+math.log1p(-p)
                bw[0,v]=_logadd(bw[0,v],no+bw[0,u]);bw[1,v]=_logadd(bw[1,v],no+bw[1,u])
    initial=1 if present else 0;values=np.zeros((draws,len(pa)));selected=np.full((draws,nf),-1,np.int64)
    if not np.isfinite(bw[initial,0]):return values,selected,False
    for m in range(draws):
        protected=np.zeros(len(pa),np.bool_);v=0;state=initial
        while v<n_nodes-1:
            pick=np.random.random();cumulative=0.;chosen=-1;next_state=state
            for e in range(offsets[v],offsets[v+1]):
                g=edge_geo[e]
                if g>=0 and not physical[g]:continue
                w=0. if g<0 else weights[g];p=0. if g<0 else event_fraction[g]
                for branch in range(2):
                    if state==2:
                        if branch:continue
                        factor=1.;after=2
                    elif branch:
                        if state==0:continue
                        factor=p;after=2
                    else:factor=1-p;after=state
                    if factor<=0:continue
                    tail=w+math.log(factor)+bw[after,dest[e]]
                    if not np.isfinite(tail):continue
                    cumulative+=math.exp(tail-bw[state,v]);chosen=e;next_state=after
                    if pick<=cumulative:break
                if pick<=cumulative:break
            if chosen<0:raise ValueError('Alias-event conditional path has no continuation')
            g=edge_geo[chosen]
            if g>=0:
                selected[m,gf[g]]=g
                for j in range(ga[g],gb[g]):protected[j]=True
            v=dest[chosen];state=next_state
        if state==1:raise ValueError('Alias event condition unsatisfied')
        for j in range(len(pa)):
            if not observed[j]:continue
            probability=pp[j] if protected[j] else pa[j];hit=np.random.random()<probability
            values[m,j]=math.log(pp[j]/pa[j]) if hit else math.log1p(-pp[j])-math.log1p(-pa[j])
    return values,selected,True
