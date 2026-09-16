# Numerical kernel promoted from the validated September 2026 consensus experiments.
"""Reciprocal *shape* diagnostics for immutable native CR families.

The conditional shape likelihood integrates exactly the same uniform integer
+/-X edge variation used by native CR, on each recipient's own opportunities
and emissions. It does not compare smoothed tracks or transplant methylation
probabilities. Both shapes use one common full-region accessible base measure;
observations outside their protected intervals cancel in the likelihood ratio.

These diagnostics describe existing, caller-conditioned CR assignments. Their
weights are frozen native membership masses, NOT independent OOF evidence,
population prevalence or calibrated correspondence probabilities. In particular,
this isolated-shape comparison does not replace full configuration competition.
It cannot create calls or change native intervals. Atomic links remain separate.
"""
from __future__ import annotations

import numpy as np
from scipy.special import logsumexp


def prefix(values):
    return np.c_[np.zeros(len(values)), np.cumsum(values, axis=1)]


def prepare_shape_data(data, start, end):
    """Caller-context geometry exposure plus complete native prefix matrices.

    Alignment is observed context; MSPs and nucleosomes are upstream inferred
    annotations, not outcome-free covariates.
    """
    valid = np.zeros((len(data['units']), end-start), dtype=bool)
    # At coordinate a, furthest end of ONE original MSP containing a. This
    # handles adjacent/overlapping MSP records without pretending their union
    # admits geometries which no original MSP contains (the native contract).
    msp_end=np.zeros((len(data['units']),end-start+1),dtype=np.int32)
    for m, u in enumerate(data['units']):
        aligned = np.zeros(end-start, bool)
        msp = np.zeros(end-start, bool)
        for lo, hi in u['aligned_blocks']:
            if lo < end and hi > start:
                aligned[max(lo,start)-start:min(hi,end)-start] = True
        for lo, hi in u['msp_intervals']:
            if lo < end and hi > start:
                msp[max(lo,start)-start:min(hi,end)-start] = True
                target=msp_end[m,max(lo,start)-start:min(hi,end)-start]
                np.maximum(target,min(hi,end)-start,out=target)
        valid[m] = aligned & msp
        for lo, hi in u['raw_nuc_intervals']:
            if lo < end and hi > start:
                valid[m,max(lo,start)-start:min(hi,end)-start] = False
    return dict(start=start, end=end, positions=data['grid_positions'],
        llr=prefix(data['log_lr']), observed=prefix(data['observed']),
        hits=prefix(data['hits']==1), invalid=prefix(~valid),msp_end=msp_end)


def shape_predictive(data, center, ambiguity_bp=10, batch_size=192, minimum_opportunities=3,
                     integer_weights=None, integer_intervals=None):
    """Exact sum over ALL integer edge variants, never only the best interval.

    Conditioning on physical aligned-MSP exposure is explicit and its original
    q mass is returned. Low retained mass cannot become cross support merely
    because the surviving sliver is renormalized. No observations are binned.
    Fewer than the requested minimum opportunities is kept for shape likelihood, but cannot
    contribute to positive-core support. No observed core => no support.
    """
    if int(ambiguity_bp) != ambiguity_bp or ambiguity_bp < 0:
        raise ValueError('Nonnegative integer edge variation required')
    if isinstance(minimum_opportunities,(bool,np.bool_)) or int(minimum_opportunities)!=minimum_opportunities or minimum_opportunities<1:
        raise ValueError('A positive integer minimum opportunity count is required')
    center = np.asarray(center, dtype=int)
    if center.shape != (2,) or center[0] >= center[1]:
        raise ValueError('Positive-width family center required')
    x = int(ambiguity_bp)
    a,b = np.meshgrid(np.arange(center[0]-x,center[0]+x+1),
                      np.arange(center[1]-x,center[1]+x+1), indexing='ij')
    keep = a < b
    a,b = a[keep],b[keep]
    if integer_intervals is not None:
        intervals=np.asarray(integer_intervals)
        if intervals.ndim!=2 or intervals.shape[1]!=2 or not len(intervals) or np.any(~np.isfinite(intervals)) or np.any(intervals!=np.rint(intervals)) or np.any(intervals[:,0]>=intervals[:,1]):
            raise ValueError('Finite positive-width integer interval array required')
        if integer_weights is None:raise ValueError('Explicit intervals require explicit source prior weights')
        a,b=intervals.astype(np.int64).T
    integer_alias_count=len(a)
    ga,gb = np.searchsorted(data['positions'], a),np.searchsorted(data['positions'], b)
    visible = ga < gb
    # Optional weights enumerate every positive-width integer interval in the
    # same row-major order as the mesh above. Source evidence is constant within
    # a source projection, so transfer distributes each class over its aliases.
    # Keep source mass lost at recipient-invisible aliases explicit; never donate
    # that mass to a favorable observable sliver.
    all_weight=None
    if integer_weights is not None:
        all_weight=np.asarray(integer_weights,float)
        if all_weight.ndim==1:all_weight=np.broadcast_to(all_weight,(len(data['llr']),len(a)))
        if all_weight.shape!=(len(data['llr']),len(a)) or np.any(~np.isfinite(all_weight)) or np.any(all_weight<0):
            raise ValueError('Finite nonnegative weight per positive-width integer alias (or unit/alias) required')
        total=all_weight.sum(1)
        if np.any(total<=0):raise ValueError('Every shape mixture must have positive total mass')
        all_weight=all_weight/total[:,None]
        all_weight=all_weight[:,visible]
    a,b,ga,gb = a[visible],b[visible],ga[visible],gb[visible]
    n = len(data['llr'])
    out = {k:np.zeros(n) for k in ('log_bf','physical_prior_mass',
        'three_opportunity_prior_mass','minimum_opportunity_prior_mass','positive_core_posterior_mass',
        'posterior_opportunities','posterior_hits','recipient_visible_prior_mass')}
    # Testability is measured on the original integer-interval prior. Filtering
    # invisible aliases must not make a tiny visible remainder look fully
    # testable. The conditional likelihood below is unchanged. Explicit uniform
    # weights and the default uniform path therefore have identical semantics,
    # including on a sparse recipient lattice.
    out['recipient_visible_prior_mass'][:]=len(a)/integer_alias_count if all_weight is None else all_weight.sum(1)
    if not len(a):
        out['log_bf'][:] = np.nan
        return out
    in_domain = (a >= data['start']) & (b <= data['end'])
    ca,cb = np.clip(a-data['start'],0,data['end']-data['start']),np.clip(b-data['start'],0,data['end']-data['start'])
    for lo in range(0,n,batch_size):
        hi = min(n,lo+batch_size)
        llr = data['llr'][lo:hi,gb]-data['llr'][lo:hi,ga]
        opp = data['observed'][lo:hi,gb]-data['observed'][lo:hi,ga]
        hit = data['hits'][lo:hi,gb]-data['hits'][lo:hi,ga]
        physical = in_domain & ((data['invalid'][lo:hi,cb]-data['invalid'][lo:hi,ca]) == 0)
        if 'msp_end' in data:physical &= data['msp_end'][lo:hi,ca]>=cb
        count = physical.sum(1)
        weight=None if all_weight is None else all_weight[lo:hi]
        if weight is None:
            score=np.where(physical,llr,-np.inf)
            retained=count/integer_alias_count
            normalization=np.log(np.maximum(count,1))
        else:
            score=np.where(physical&(weight>0),llr+np.log(np.maximum(weight,1e-300)),-np.inf)
            retained=(weight*physical).sum(1)
            normalization=np.log(np.maximum(retained,1e-300))
        z = logsumexp(score,axis=1)
        live = retained > 0
        posterior = np.zeros_like(score)
        posterior[live] = np.exp(score[live]-z[live,None])
        out['log_bf'][lo:hi] = np.where(live,z-normalization,np.nan)
        out['physical_prior_mass'][lo:hi] = retained
        out['three_opportunity_prior_mass'][lo:hi] = (physical&(opp>=3)).sum(1)/integer_alias_count if weight is None else (weight*physical*(opp>=3)).sum(1)
        out['minimum_opportunity_prior_mass'][lo:hi] = (physical&(opp>=minimum_opportunities)).sum(1)/integer_alias_count if weight is None else (weight*physical*(opp>=minimum_opportunities)).sum(1)
        out['positive_core_posterior_mass'][lo:hi] = (posterior*((opp>=minimum_opportunities)&(llr>0))).sum(1)
        out['posterior_opportunities'][lo:hi] = (posterior*opp).sum(1)
        out['posterior_hits'][lo:hi] = (posterior*hit).sum(1)
    return out


def weighted_quantile(values, weights, q):
    keep = np.isfinite(values) & (weights>0)
    if not keep.any():
        return None
    v,w = np.asarray(values)[keep],np.asarray(weights)[keep]
    order = np.argsort(v,kind='stable');v,w=v[order],w[order]
    return float(v[min(np.searchsorted(np.cumsum(w),q*w.sum()),len(v)-1)])


def direction(native, transferred, membership, tolerance_odds=10.):
    """How much frozen native class mass accepts a foreign geometry model?

    Tolerance is a likelihood-loss allowance (e.g. 10:1), not a 90%/99% error
    probability. Native and recipient positive evidence and geometry exposure
    are explicit. Scores use FRACTIONS of available mass, not saturating counts.
    """
    if tolerance_odds < 1:
        raise ValueError('Likelihood tolerance odds must be at least one')
    w = np.asarray(membership,dtype=float)
    if np.any(~np.isfinite(w)) or np.any(w<0) or np.any(w>1+1e-8):
        raise ValueError('Finite frozen native unit membership masses required')
    native_ok = (np.isfinite(native['log_bf']) & (native['log_bf']>0) &
                 (native['physical_prior_mass']>=.5))
    original_mass=float(w.sum())
    w = w*native_ok*native['positive_core_posterior_mass']
    total = float(w.sum())
    testable = (np.isfinite(transferred['log_bf']) &
                (transferred['physical_prior_mass']>=.5) &
                (transferred.get('minimum_opportunity_prior_mass',transferred['three_opportunity_prior_mass'])>=.5))
    loss = native['log_bf']-transferred['log_bf']
    positive = testable & (transferred['log_bf']>0) & (transferred['positive_core_posterior_mass']>=.5)
    acceptable = positive & (np.abs(loss) <= np.log(tolerance_odds)+1e-12)
    native_preferred=positive & (loss>np.log(tolerance_odds)+1e-12)
    foreign_preferred=positive & (loss < -np.log(tolerance_odds)-1e-12)
    weak=testable & ~positive
    accepted = w*acceptable
    tested = w*testable
    categories=dict(acceptable_native_mass_fraction=float(accepted.sum()/total) if total else 0.,
        native_shape_preferred_mass_fraction=float((w*native_preferred).sum()/total) if total else 0.,
        foreign_shape_preferred_mass_fraction=float((w*foreign_preferred).sum()/total) if total else 0.,
        weak_transfer_mass_fraction=float((w*weak).sum()/total) if total else 0.,
        untestable_mass_fraction=float((w*~testable).sum()/total) if total else 0.)
    if total and abs(sum(categories.values())-1)>1e-8:
        raise AssertionError('Cross-shape categories must partition native mass')
    return dict(native_effective_support=total,
        original_decoded_membership_mass=original_mass,
        native_evidence_excluded_mass=original_mass-total,
        native_support_units=int((w>0).sum()),
        native_weight_ess=float(total**2/(w@w)) if total else 0.,
        testable_native_mass_fraction=float(tested.sum()/total) if total else 0.,
        **categories,
        acceptable_effective_support=float(accepted.sum()),
        mean_native_minus_transfer_log_bf=float(np.nansum(tested*loss)/tested.sum()) if tested.sum() else None,
        median_native_minus_transfer_log_bf=weighted_quantile(loss,tested,.5),
        q10_native_minus_transfer_log_bf=weighted_quantile(loss,tested,.1),
        q90_native_minus_transfer_log_bf=weighted_quantile(loss,tested,.9),
        median_transferred_log_bf=weighted_quantile(transferred['log_bf'],tested,.5),
        median_transferred_opportunities=weighted_quantile(transferred['posterior_opportunities'],tested,.5),
        median_transferred_hits=weighted_quantile(transferred['posterior_hits'],tested,.5))


def plausible_pairs(left, right, ambiguity_bp):
    """ALL overlapping bounded envelopes; no Hungarian, nearest-only or cap."""
    for i,(a,b) in enumerate(left):
        for j,(c,d) in enumerate(right):
            if a-ambiguity_bp < d+ambiguity_bp and c-ambiguity_bp < b+ambiguity_bp:
                yield i,j


def link_status(forward, reverse, minimum_support=3., minimum_fraction=.5):
    if min(forward['native_effective_support'],reverse['native_effective_support']) < minimum_support:
        return 'sparse_native_support'
    if min(forward['testable_native_mass_fraction'],reverse['testable_native_mass_fraction']) < minimum_fraction:
        return 'resolution_or_exposure_limited'
    if min(forward['acceptable_native_mass_fraction'],reverse['acceptable_native_mass_fraction']) >= minimum_fraction:
        return 'reciprocal_shape_compatible'
    if max(forward['foreign_shape_preferred_mass_fraction'],reverse['foreign_shape_preferred_mass_fraction']) >= minimum_fraction:
        return 'foreign_shape_preferred'
    return 'shape_disagreement'
