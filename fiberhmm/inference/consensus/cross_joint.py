from __future__ import annotations
import numpy as np


def replacement_diagnostics(delta,native_inclusion,replacement_inclusion,tolerance_odds):
    """Keep full-model fit and the target's own role as separate requirements.

    Unchanged predictive density is insufficient if another competitor absorbed
    the replaced family. Retention uses the same units and caps each unit's
    contribution at its native target mass; new mass elsewhere cannot repay it.
    """
    delta=np.asarray(delta,float);w=np.asarray(native_inclusion,float)
    other=np.asarray(replacement_inclusion,float)
    if delta.shape!=w.shape or w.shape!=other.shape or np.any(~np.isfinite(delta)):
        raise ValueError('One finite predictive change and target inclusion per tested unit required')
    if any(np.any(~np.isfinite(a)) or np.any(a<0) or np.any(a>1+1e-7) for a in (w,other)):
        raise ValueError('Target inclusion must be a finite mass in [0,1]')
    total=float(w.sum());within=delta>=-np.log(tolerance_odds)
    retained=np.minimum(w,other)
    return dict(status='tested',
        retained_native_mass_fraction=float((w*within).sum()/total) if total else None,
        retained_target_inclusion_fraction=float(retained.sum()/total) if total else None,
        jointly_retained_target_mass_fraction=float((retained*within).sum()/total) if total else None,
        sum_log_predictive_change=float(delta.sum()),mean_log_predictive_change=float(delta.mean()) if len(delta) else None,
        native_family_mass=total,replacement_family_mass=float(other.sum()),eligible_units=len(w))

def prediction(kernel, values, eta, available, batch=192):
    unique,_,inverse=kernel.prior_recipe(available,len(values))
    prior=[]
    for start in range(0,len(unique),batch):
        m=unique[start:start+batch]
        prior.extend(kernel.evaluate(np.zeros((len(m),kernel.k)),eta,allowed=m)['log_partition'])
    z=[];inclusion=[]
    for start in range(0,len(values),batch):
        out=kernel.evaluate(values[start:start+batch],eta,allowed=available[start:start+batch])
        z.extend(out['log_partition']);inclusion.append(out['family_inclusion'])
    return np.asarray(z)-np.asarray(prior)[inverse],np.concatenate(inclusion)


def prediction_with_geometry_groups(kernel,values,eta,available,adjustments,groups,batch=24):
    """Exact predictive density for fold-specific geometry weights.

    Both prior and data partitions use the same row's complete geometry model;
    no native competitor is dropped and no N-units × N-geometries copy is kept.
    """
    groups=np.asarray(groups,int);adjustments=np.asarray(adjustments,float)
    if groups.shape!=(len(values),) or adjustments.ndim!=2 or adjustments.shape[1]!=len(kernel.ga):
        raise ValueError('Geometry-model groups must match units and kernel projections')
    if np.any(groups<0) or np.any(groups>=len(adjustments)):raise ValueError('Unknown geometry-model group')
    recipe=np.c_[np.asarray(available,np.int64),groups]
    unique,inverse=np.unique(recipe,axis=0,return_inverse=True)
    prior=np.zeros(len(unique));z=np.zeros(len(values));inc=np.zeros((len(values),kernel.f))
    for lo in range(0,len(unique),batch):
        rows=unique[lo:lo+batch]
        prior[lo:lo+len(rows)]=kernel.evaluate(np.zeros((len(rows),kernel.k)),eta,allowed=rows[:,:-1].astype(bool),
            geometry_log_adjustment=adjustments[rows[:,-1]])['log_partition']
    for lo in range(0,len(values),batch):
        hi=min(len(values),lo+batch)
        out=kernel.evaluate(values[lo:hi],eta,allowed=available[lo:hi],geometry_log_adjustment=adjustments[groups[lo:hi]])
        z[lo:hi]=out['log_partition'];inc[lo:hi]=out['family_inclusion']
    return z-prior[inverse],inc
