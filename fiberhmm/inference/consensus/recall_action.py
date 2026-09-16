# Numerical kernel promoted from the validated September 2026 consensus experiments.
"""Bayes action for footprint recall under measured-opportunity Hamming loss.

The posterior is NOT changed. Calling opportunity j protected instead of
accessible improves expected 0/1 accuracy by 2 P(protected_j | data)-1.
Maximize that additive utility over the same legal family-configuration DAG.
Ties use the actual native geometry log weight. This is NOT a joint geometry
MAP probability and must not be labeled one. It avoids the empty-state artifact
when protection mass is spread over many plausible edge coordinates.
"""
from __future__ import annotations
import numpy as np
from numba import njit,prange


def coverage_profiles(kernel,geometry_mass):
    result=np.empty((len(geometry_mass),kernel.k))
    for m,mass in enumerate(geometry_mass):
        diff=np.zeros(kernel.k+1)
        np.add.at(diff,kernel.ga,mass);np.add.at(diff,kernel.gb,-mass)
        result[m]=np.clip(np.cumsum(diff)[:-1],0,1)
    return result


@njit(cache=True,parallel=True)
def _decode(utility,native,eta,allowed,physical,adjust,offsets,dest,edge_geo,ga,gb,gf,logq,n_nodes):
    n=len(utility);out=np.full((n,len(eta)),-1,dtype=np.int64)
    scores=np.zeros(n)
    for m in prange(n):
        best=np.full(n_nodes,-np.inf);tie=np.full(n_nodes,-np.inf)
        previous=np.full(n_nodes,-1,dtype=np.int64);choice=np.full(n_nodes,-1,dtype=np.int64)
        best[0]=0.;tie[0]=0.
        for v in range(n_nodes-1):
            if best[v]==-np.inf:continue
            for e in range(offsets[v],offsets[v+1]):
                g=edge_geo[e]
                if g>=0 and (not allowed[m,gf[g]] or not physical[m,g]):continue
                delta=0. if g<0 else utility[m,gb[g]]-utility[m,ga[g]]
                second=0. if g<0 else eta[gf[g]]+logq[g]+adjust[m,g]+native[m,gb[g]]-native[m,ga[g]]
                u=dest[e];candidate=best[v]+delta;secondary=tie[v]+second
                if candidate>best[u]+1e-12 or (abs(candidate-best[u])<=1e-12 and secondary>tie[u]):
                    best[u]=candidate;tie[u]=secondary;previous[u]=v;choice[u]=g
        scores[m]=best[n_nodes-1];v=n_nodes-1
        while v>0:
            g=choice[v]
            if g>=0:out[m,gf[g]]=g
            v=previous[v]
    return out,scores


def decode_posterior_accuracy(kernel,values,observed,posterior,eta,allowed,geometry_allowed,adjustment):
    coverage=coverage_profiles(kernel,posterior['geometry_mass'])
    utility=(2*coverage-1)*observed
    up=np.c_[np.zeros(len(utility)),np.cumsum(utility,axis=1)]
    lp=np.c_[np.zeros(len(values)),np.cumsum(values,axis=1)]
    calls,scores=_decode(up,lp,eta,allowed,geometry_allowed,adjustment,kernel.offsets,
        kernel.dest,kernel.edge_geo,kernel.ga,kernel.gb,kernel.gf,kernel.logq,kernel.n_nodes)
    return dict(geometry_by_family=calls,expected_accuracy_improvement=scores,coverage=coverage)


@njit(cache=True,parallel=True)
def _family_label_utilities(masses,protected,observed,ga,gb,order,offsets,n_families):
    """Delta correct-label probability: P(label=f) - P(label=accessible).

    Binary protection coverage credits the same bases to every covering family,
    including an almost impossible one. Here each family receives only its own
    posterior coverage. The accessible alternative is common to all labels.
    """
    utility=np.zeros_like(masses)
    for m in prange(len(masses)):
        for f in range(n_families):
            first,last=offsets[f],offsets[f+1]
            if first==last:continue
            lo=ga[order[first]];hi=gb[order[first]]
            for z in range(first,last):
                g=order[z];lo=min(lo,ga[g]);hi=max(hi,gb[g])
            difference=np.zeros(hi-lo+1)
            for z in range(first,last):
                g=order[z];difference[ga[g]-lo]+=masses[m,g];difference[gb[g]-lo]-=masses[m,g]
            prefix=np.zeros(hi-lo+1);family_mass=0.
            for x in range(hi-lo):
                family_mass+=difference[x]
                delta=(family_mass-(1.-protected[m,x+lo])) if observed[m,x+lo] else 0.
                prefix[x+1]=prefix[x]+delta
            for z in range(first,last):
                g=order[z];utility[m,g]=prefix[gb[g]-lo]-prefix[ga[g]-lo]
    return utility


@njit(cache=True,parallel=True)
def _decode_family_labels(utility,native,eta,allowed,physical,adjust,offsets,dest,edge_geo,ga,gb,gf,logq,n_nodes):
    n=len(utility);out=np.full((n,len(eta)),-1,dtype=np.int64);scores=np.zeros(n)
    for m in prange(n):
        best=np.full(n_nodes,-np.inf);tie=np.full(n_nodes,-np.inf)
        previous=np.full(n_nodes,-1,dtype=np.int64);choice=np.full(n_nodes,-1,dtype=np.int64)
        best[0]=0.;tie[0]=0.
        for v in range(n_nodes-1):
            if best[v]==-np.inf:continue
            for e in range(offsets[v],offsets[v+1]):
                g=edge_geo[e]
                if g>=0 and (not allowed[m,gf[g]] or not physical[m,g]):continue
                delta=0. if g<0 else utility[m,g]
                second=0. if g<0 else eta[gf[g]]+logq[g]+adjust[m,g]+native[m,gb[g]]-native[m,ga[g]]
                u=dest[e];candidate=best[v]+delta;secondary=tie[v]+second
                if candidate>best[u]+1e-12 or (abs(candidate-best[u])<=1e-12 and secondary>tie[u]):
                    best[u]=candidate;tie[u]=secondary;previous[u]=v;choice[u]=g
        scores[m]=best[n_nodes-1];v=n_nodes-1
        while v>0:
            g=choice[v]
            if g>=0:out[m,gf[g]]=g
            v=previous[v]
    return out,scores


def decode_posterior_family_accuracy(kernel,values,observed,posterior,eta,allowed,geometry_allowed,adjustment):
    """Joint Bayes action for the *family labels*, not just binary protection.

    Neither the posterior nor confidence changes here. Under per-opportunity
    categorical 0/1 loss, writing family f rather than accessible changes expected
    correctness by P(f covers j|data)-P(accessible j|data). Summing that utility
    and maximizing on the same DAG respects all physical/family exclusions.
    Native geometry weights are used only to break exact utility ties.
    """
    coverage=coverage_profiles(kernel,posterior['geometry_mass'])
    order=np.argsort(kernel.gf,kind='stable').astype(np.int64)
    offsets=np.searchsorted(kernel.gf[order],np.arange(kernel.f+1)).astype(np.int64)
    utility=_family_label_utilities(posterior['geometry_mass'],coverage,observed,
                                    kernel.ga,kernel.gb,order,offsets,kernel.f)
    lp=np.c_[np.zeros(len(values)),np.cumsum(values,axis=1)]
    calls,scores=_decode_family_labels(utility,lp,eta,allowed,geometry_allowed,adjustment,
        kernel.offsets,kernel.dest,kernel.edge_geo,kernel.ga,kernel.gb,kernel.gf,kernel.logq,kernel.n_nodes)
    return dict(geometry_by_family=calls,expected_accuracy_improvement=scores,coverage=coverage,
                decision_rule='observed_opportunity_categorical_family_label_loss')
