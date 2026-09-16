"""Exact integer-alias exposure under fixed caller-supplied context.

This is conditional on the MSP/nucleosome calls, not independent experimental
ground truth. It contains NO modification outcomes or positive-evidence veto.
Each interval must be inside one original MSP and one aligned continuous run,
and must avoid fixed nucleosome obstacles. Adjacent MSPs are not union-merged.
"""
from __future__ import annotations
import numpy as np
from numba import njit


def endpoint_limits(unit,start,end):
    aligned=np.zeros(end-start,bool);msp_end=np.zeros(end-start,np.int64)
    for a,b in unit['aligned_blocks']:
        if a<end and b>start:aligned[max(a,start)-start:min(b,end)-start]=True
    for a,b in unit['raw_nuc_intervals']:
        if a<end and b>start:aligned[max(a,start)-start:min(b,end)-start]=False
    for a,b in unit['msp_intervals']:
        if a<end and b>start:
            view=msp_end[max(a,start)-start:min(b,end)-start]
            np.maximum(view,min(b,end),out=view)
    # Furthest exclusive boundary within one uninterrupted aligned/unblocked run.
    run_end=np.zeros(len(aligned),np.int64);right=end
    for j in range(len(aligned)-1,-1,-1):
        if not aligned[j]:right=start+j
        else:run_end[j]=right
    return np.minimum(run_end,msp_end)


@njit(cache=True)
def _exposure(rectangles,centers,limits,start):
    fraction=np.zeros(len(rectangles));coordinates=np.zeros((len(rectangles),2),np.int64)
    for g in range(len(rectangles)):
        sl,sh,el,eh=rectangles[g];count=0;best=1e100
        for a in range(max(sl,start),min(sh,start+len(limits)-1)+1):
            bh=min(eh,limits[a-start]);bl=max(el,a+1)
            if bh<bl:continue
            count+=bh-bl+1
            b=min(max(centers[g,1],bl),bh)
            distance=(a-centers[g,0])**2+(b-centers[g,1])**2
            if distance<best:
                best=distance;coordinates[g,0]=a;coordinates[g,1]=b
        fraction[g]=count/((sh-sl+1)*(eh-el+1))
    return fraction,coordinates


def physical_alias_exposure(unit,kernel,rectangles):
    start,end=map(int,unit['_region'])
    return _exposure(rectangles,kernel.centers[kernel.gf],endpoint_limits(unit,start,end),start)


@njit(cache=True)
def _coverage_aliases(rectangles,limits,start,left,right,minimum_overlap):
    fraction=np.zeros(len(rectangles));event=np.zeros(len(rectangles))
    for g in range(len(rectangles)):
        sl,sh,el,eh=rectangles[g];total=0;positive=0
        for a in range(max(sl,start),min(sh,start+len(limits)-1)+1):
            bl=max(el,a+1);bh=min(eh,limits[a-start])
            if bh<bl:continue
            total+=bh-bl+1
            # overlap([a,b),[left,right)) >= minimum_overlap.
            required=max(a,left)+minimum_overlap
            if required<=right:positive+=max(0,bh-max(bl,required)+1)
        fraction[g]=total/((sh-sl+1)*(eh-el+1))
        event[g]=positive/total if total else 0.
    return fraction,event


def coverage_alias_exposure(unit,kernel,rectangles,window,minimum_overlap):
    """Physical q fraction and event fraction WITHIN each physical alias class."""
    left,right=map(int,window);start,end=map(int,unit['_region'])
    if not 0<minimum_overlap<=right-left:raise ValueError('Positive overlap no larger than the fixed window required')
    return _coverage_aliases(rectangles,endpoint_limits(unit,start,end),start,left,right,int(minimum_overlap))
