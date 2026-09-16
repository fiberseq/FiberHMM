"""Exact fixed-competitor population-shape comparisons (experimental kernel).

One occurrence per family makes a target's include partition linear in its
geometry mixture. Cache exact full-configuration include/exclude partitions for
each source-trained template, then optimize a common mixture across datasets or
separate mixtures. Every model retains the same competitors and Z_prior.
This is a computational kernel, not an empirical q-value or equivalence test.
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit,logsumexp


def target_terms(kernel,values,eta,available,adjustments,groups,family,batch=24):
    """Return exact target-vs-exclude partition ratios under each row's model.

    Target activity is set to zero; it will be fitted independently per dataset.
    Competitor activities and all other geometry weights remain unchanged. The
    target is forbidden explicitly for the exclude partition; subtraction of
    nearly equal total and include partitions would be numerically unstable.
    """
    groups=np.asarray(groups,int);eta=np.asarray(eta,float).copy();eta[family]=0.
    values=np.asarray(values,float);allowed=np.asarray(available,bool)
    adjustments=np.asarray(adjustments,float)
    if groups.shape!=(len(values),) or allowed.shape!=(len(values),kernel.f):raise ValueError('One context per unit required')
    if adjustments.ndim!=2 or adjustments.shape[1]!=len(kernel.ga):raise ValueError('Geometry adjustments must match the full model')
    if np.any(groups<0) or np.any(groups>=len(adjustments)):raise ValueError('Unknown geometry-model group')
    ratios={k:np.zeros(len(values)) for k in ('data','prior','exclude_predictive')}
    for lo in range(0,len(values),batch):
        hi=min(len(values),lo+batch);mask=allowed[lo:hi];without=mask.copy();without[:,family]=False
        kw=dict(geometry_log_adjustment=adjustments[groups[lo:hi]])
        result={}
        for name,v in [('data',values[lo:hi]),('prior',np.zeros_like(values[lo:hi]))]:
            all_states=kernel.evaluate(v,eta,allowed=mask,**kw)
            exclude=kernel.evaluate(v,eta,allowed=without,**kw)['log_partition']
            inc=all_states['family_inclusion'][:,family]
            with np.errstate(divide='ignore'):included=all_states['log_partition']+np.log(inc)
            ratios[name][lo:hi]=included-exclude
            result[name]=exclude
        ratios['exclude_predictive'][lo:hi]=result['data']-result['prior']
    return ratios


def mixture_prediction(data,prior,weights,activity):
    """Normalized log predictive change relative to target-excluded model."""
    data=np.asarray(data,float);prior=np.asarray(prior,float);w=np.asarray(weights,float)
    if data.shape!=prior.shape or data.ndim!=2 or w.shape!=(data.shape[1],):raise ValueError('Same unit-by-template domain required')
    if np.any(w<0) or not np.isclose(w.sum(),1.) or np.any(np.isnan(data)) or np.any(np.isnan(prior)):
        raise ValueError('Normalized finite mixture and non-NaN partition ratios required')
    with np.errstate(divide='ignore'):logw=np.log(w)
    xd=activity+logsumexp(data+logw,axis=1);xp=activity+logsumexp(prior+logw,axis=1)
    return np.logaddexp(0.,xd)-np.logaddexp(0.,xp),expit(xd),expit(xp)


def mixture_objective(parameters,datasets,shared=True,weight_pseudocount=1.):
    """Mean conditional negative log likelihood and exact gradient.

    Each dataset has its own target activity. Mixture logits are either shared
    or dataset-specific. A predeclared symmetric weight pseudo-count stabilizes
    empty components; it is applied once per fitted mixture, not per molecule.
    All eligible and non-target units enter once, with their unchanged contexts.
    """
    k=len(datasets);t=datasets[0]['data'].shape[1];p=np.asarray(parameters,float)
    nm=1 if shared else k
    if p.shape!=(k+nm*t,):raise ValueError('Activity and mixture parameter dimensions differ')
    if weight_pseudocount<0:raise ValueError('Weight pseudo-count must be nonnegative')
    logits=p[k:].reshape(nm,t);logw=logits-logsumexp(logits,axis=1,keepdims=True);w=np.exp(logw)
    objective=0.;gradient=np.zeros_like(p);wg=gradient[k:].reshape(nm,t);n=0
    for j,dataset in enumerate(datasets):
        h=0 if shared else j;d=np.asarray(dataset['data']);r=np.asarray(dataset['prior'])
        if d.shape!=r.shape or d.ndim!=2 or d.shape[1]!=t or np.any(np.isnan(d)) or np.any(np.isnan(r)):
            raise ValueError('Every dataset requires matching unit-by-template include ratios')
        ld=logsumexp(d+logw[h],axis=1);lp=logsumexp(r+logw[h],axis=1)
        a=expit(p[j]+ld);b=expit(p[j]+lp)
        objective-=float((np.logaddexp(0.,p[j]+ld)-np.logaddexp(0.,p[j]+lp)).sum());n+=len(d)
        gradient[j]=-float((a-b).sum())
        with np.errstate(invalid='ignore'):
            rd=np.exp(d+logw[h]-ld[:,None]);rp=np.exp(r+logw[h]-lp[:,None])
        rd[~np.isfinite(rd)]=0.;rp[~np.isfinite(rp)]=0.
        wg[h]-=((a[:,None]*(rd-w[h]))-(b[:,None]*(rp-w[h]))).sum(0)
    if not n:raise ValueError('At least one evidence unit required')
    # Dirichlet-like pseudo-observations; neutral uniform target, not harmony.
    objective-=weight_pseudocount*float(logw.mean(axis=1).sum())
    wg+=weight_pseudocount*(w-1/t)
    return objective/n,gradient/n


def fit_mixture(datasets,shared=True,weight_pseudocount=1.,max_iter=300):
    k=len(datasets);t=datasets[0]['data'].shape[1];nm=1 if shared else k;fits=[]
    # Deterministic dispersed starts; no source is privileged by optimization.
    for start in range(t+1):
        x=np.zeros(k+nm*t);x[:k]=-1.
        if start:x[k:].reshape(nm,t)[:,start-1]=2.
        fit=minimize(lambda v:mixture_objective(v,datasets,shared,weight_pseudocount),x,jac=True,
            method='L-BFGS-B',bounds=[(-12.,12.)]*len(x),options={'maxiter':max_iter,'ftol':1e-12,'gtol':1e-7,'maxls':40})
        logits=fit.x[k:].reshape(nm,t);w=np.exp(logits-logsumexp(logits,axis=1,keepdims=True))
        grad=np.asarray(fit.jac).copy();grad[(fit.x<=-12+1e-7)&(grad>0)]=0.;grad[(fit.x>=12-1e-7)&(grad<0)]=0.
        fits.append(dict(objective=float(fit.fun),success=bool(fit.success),iterations=int(fit.nit),
            max_projected_gradient=float(np.max(np.abs(grad))),message=str(fit.message),
            activities=fit.x[:k].tolist(),weights=w.tolist(),
            activity_lower_bound_hit=(fit.x[:k]<=-12+1e-5).tolist(),
            activity_upper_bound_hit=(fit.x[:k]>=12-1e-5).tolist(),
            convergence_semantics='Numerical constrained optimum; a boundary solution may be scientifically uninformative'))
    return min(fits,key=lambda f:f['objective']),fits
