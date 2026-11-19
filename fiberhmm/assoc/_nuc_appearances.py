"""Numeric batch kernel for the standard nucleosome association appearances.

The public association engine owns state rules, masks, balancing, resampling
and inference. This kernel evaluates its unchanged per-molecule sums across
many loci without constructing a Python object for every appearance.
"""
import math
import numpy as np
from numba import njit


@njit(cache=True, nogil=True)
def appearance_batch(a, msps, nucs, dyads, cum_inf, ip, il, ih, mark_cum,
                     length, reverse, bins, period, slop, min_msp, min_sites,
                     flank_exclude, molecule_frame, summary_only, histogram_bins):
    n, nb = len(a), len(bins)
    nout = 1 if summary_only else nb
    state = np.full(n,-1,np.int64)
    masks = np.zeros(n,np.int64)
    stats = np.zeros((n,nout,7))
    density_hits, density_sites = np.zeros(n),np.zeros(n)
    hist = np.zeros((n,histogram_bins))
    for j in range(n):
        anchor = int(a[j])
        bs, be = anchor,anchor
        for k in range(len(msps)):
            s,e=msps[k,0],msps[k,1]
            if s<=anchor-slop and e>=anchor+slop and e-s>=min_msp:
                state[j]=1
                if molecule_frame:
                    bs,be=s,e
                break
        if state[j]<0:
            for k in range(len(nucs)):
                if nucs[k,0]<=anchor and nucs[k,1]>anchor:
                    state[j]=0
                    if molecule_frame:
                        bs,be=nucs[k,0],nucs[k,1]
                    break
        if state[j]<0:
            continue
        mask=0
        for b in range(nb):
            low,high=int(math.floor(bins[b,0])),int(math.ceil(bins[b,1]))
            if bs-high<0 or be+high>length:
                continue
            nr=cum_inf[be+high]-cum_inf[be+low]
            nl=cum_inf[bs-low]-cum_inf[bs-high]
            if nr>=min_sites and nl>=min_sites:
                mask|=1<<b
        masks[j]=mask
        if mask==0:
            continue
        ex0=np.searchsorted(ip,anchor-flank_exclude)
        ex1=np.searchsorted(ip,anchor+flank_exclude)
        for k in range(len(il)):
            left_hi=max(il[k],min(ih[k],ex0))
            right_lo=min(ih[k],max(il[k],ex1))
            density_sites[j]+=left_hi-il[k]+ih[k]-right_lo
            density_hits[j]+=mark_cum[left_hi]-mark_cum[il[k]]+mark_cum[ih[k]]-mark_cum[right_lo]
        for k in range(len(dyads)):
            d=dyads[k]
            if molecule_frame:
                if d>=be:
                    rel=d-be
                elif d<bs:
                    rel=d-bs
                else:
                    continue
            else:
                rel=d-anchor
            if reverse:
                rel=-rel
            distance=abs(rel)
            if distance<bins[0,0] or distance>=bins[nb-1,1]:
                continue
            theta=2*math.pi*rel/period
            re,im=math.cos(theta),math.sin(theta)
            for out in range(nout):
                b=nb-1 if summary_only else out
                if ((mask>>b)&1)==0 or not (bins[b,0]<=distance<bins[b,1]):
                    continue
                offset=0 if rel>0 else 2
                stats[j,out,offset]+=re
                stats[j,out,offset+1]+=im if rel>0 else -im
                stats[j,out,6]+=1
            if histogram_bins and state[j]==1 and ((mask>>(nb-1))&1):
                hb=int((distance-bins[0,0])//10)
                if 0<=hb<histogram_bins:
                    hist[j,hb]+=1
        for out in range(nout):
            stats[j,out,4]=stats[j,out,0]**2+stats[j,out,1]**2
            stats[j,out,5]=stats[j,out,2]**2+stats[j,out,3]**2
    return state,masks,stats,density_hits,density_sites,hist
