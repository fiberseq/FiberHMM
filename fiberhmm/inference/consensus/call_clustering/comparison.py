"""Native-only, resolution-aware comparison. No rescue, fitting or sampling.

Coarse events are explicit ANY-child events, not assertions that all child
geometries are one biological species. Every child retains its native identity.
Lattice eligibility and observed agreement are deliberately separate quantities.
"""
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations
import hashlib
import json
import math

import numpy as np

from . import overlap_fraction, window_physics, RescueOptions, median_interval


@dataclass(frozen=True)
class ComparisonOptions:
    overlap: float = .7
    maximum_span_bp: int = 100
    minimum_covered_units: int = 20
    minimum_capable_fraction: float = .8
    minimum_opportunities: int = 3
    agreement_difference: float = .1


def coarse_events(results, options=None, reads=None):
    """Bounded broad-anchor stars of native SR classes, symmetric in assay.

    Broad established geometries nominate the comparison scale; population
    prevalence and rate agreement never enter. No transitive closure: every
    child must overlap the immutable anchor directly. Each native family belongs
    to one event. An anchor may describe cooccurring fragments; counts stay ANY.
    """
    opt = options or ComparisonOptions()
    families = [dict(f, dataset=scope[3:]) for scope, fs in results.items()
                if scope.startswith('SR ') for f in fs]
    for f in families:
        f['fine_interval']=list(f['interval'])
        f['native_assay_intervals']={}
        if reads is not None:
            by_assay=defaultdict(list)
            for uid,ordinal in f['members']:
                by_assay[reads[uid]['dataset']].append((uid,ordinal))
            # A pooled median can hide the wider geometry of the lower-depth
            # assay. Use the widest recurrent ASSAY-native median as a comparison
            # anchor, without refitting or reassigning any fine memberships.
            for ds,members in sorted(by_assay.items()):
                if len({reads[uid]['group'] for uid,_ in members})>=3:
                    iv=median_interval([reads[uid]['calls'][ordinal] for uid,ordinal in members])
                    if 0<iv[1]-iv[0]<=opt.maximum_span_bp:
                        f['native_assay_intervals'][ds]=iv
            if f['native_assay_intervals']:
                ds,iv=min(f['native_assay_intervals'].items(),key=lambda item:(-(item[1][1]-item[1][0]),item[1],item[0]))
                f['interval']=iv;f['anchor_assay']=ds
    order = sorted(families, key=lambda f: (not f['established'],
                   -(f['interval'][1]-f['interval'][0]),f['interval'],f['dataset'],f['family_id']))
    used, events = set(), []
    for anchor in order:
        if anchor['family_id'] in used:
            continue
        children = [anchor];used.add(anchor['family_id'])
        a,b = anchor['interval']; envelope = [a,b]
        if anchor['established']:
            for child in order:
                if child['family_id'] in used:
                    continue
                c,d = child['interval']
                if (overlap_fraction((a,b),(c,d)) >= opt.overlap
                        and max(envelope[1],d)-min(envelope[0],c) <= opt.maximum_span_bp):
                    children.append(child);used.add(child['family_id'])
                    envelope = [min(envelope[0],c),max(envelope[1],d)]
        # No new caller interval is created: this envelope only defines the
        # common evaluation domain. Even cooccurring children count once/unit.
        span = [min(f['interval'][0] for f in children),max(f['interval'][1] for f in children)]
        by_ds = defaultdict(list)
        for f in children:
            by_ds[f['dataset']].append(f['family_id'])
        key = sorted(f['family_id'] for f in children)
        fid = 'XC_'+hashlib.sha256(json.dumps(key).encode()).hexdigest()[:16]
        members = sorted({(uid,ordinal) for f in children for uid,ordinal in f['members']})
        events.append(dict(family_id=fid, interval=span, anchor=anchor['family_id'],
            anchor_interval=anchor['interval'], established=any(f['established'] for f in children),
            anchor_assay=anchor.get('anchor_assay'),
            geometry_basis='widest_recurrent_assay_native_median' if reads is not None else 'supplied_family_median',
            children=dict(by_ds), members=[list(v) for v in members],
            child_intervals={f['family_id']:f['interval'] for f in children},
            fine_child_intervals={f['family_id']:f['fine_interval'] for f in children},
            native_assay_intervals={f['family_id']:f['native_assay_intervals'] for f in children},
            kind='coarse_any_native_class' if any(len(v)>1 for v in by_ds.values()) else 'native_class_correspondence',
            rate_agreement_used=False, transitive_union=False, native_calls_modified=False))
    return sorted(events,key=lambda f:(f['interval'],f['family_id']))


def wilson(k,n):
    if not n:
        return None
    z=1.959963984540054;centre=(k+z*z/2)/(n+z*z)
    half=z*math.sqrt(k*(1-k/n)+z*z/4)/(n+z*z)
    return [max(0.,centre-half),min(1.,centre+half)]


def _finish(c):
    c=dict(c)
    for key in ('covered_units','assigned_units','lattice_capable_units','capable_assigned_units',
                'physically_accessible_units','multiple_member_units','original_member_calls'):
        c.setdefault(key,0)
    n=c['covered_units'];m=c['lattice_capable_units'];k=c['assigned_units']
    c.update(fraction=k/n if n else None,lattice_capable_fraction=m/n if n else None,
             capable_fraction=c['capable_assigned_units']/m if m else None,
             fraction_interval=wilson(k,n),
             mean_opportunities=c.pop('opportunity_sum',0)/n if n else None,
             mean_ceiling=c.pop('ceiling_sum',0)/n if n else None)
    return c


def native_counts(reads, families, floors, options=None):
    """Identical aligned-span denominator across assays; exact unit membership.

    Lattice capability uses all recorded target opportunities, including those
    inside nucleosomes, independently of their observed modification outcomes.
    It is a necessary threshold-reachability check, NOT detection power/FDR.
    Physical accessibility is a separate diagnostic, never the rate denominator.
    """
    opt=options or ComparisonOptions();output=[]
    if not families:
        return output
    windows=np.asarray([f['interval'] for f in families]);lo,hi=windows.T
    keys=['covered_units','assigned_units','lattice_capable_units','capable_assigned_units',
          'multiple_member_units','original_member_calls','opportunity_sum','ceiling_sum','physically_accessible_units']
    stats={st:np.zeros((len(families),len(keys))) for st in sorted({r['stratum'] for r in reads.values()})}
    membership=defaultdict(Counter)
    for k,f in enumerate(families):
        for uid,ordinal in f['members']:
            membership[uid][k]+=1
    for uid,r in reads.items():
        blocks=np.asarray(r['blocks'],dtype=np.int64).reshape(-1,2)
        if not len(blocks):
            continue
        bs,be=blocks.T;cum=np.r_[0,np.cumsum(be-bs)]
        def coverage(x):
            j=np.searchsorted(be,x,side='right');v=cum[j].copy();inside=j<len(bs)
            v[inside]+=np.maximum(0,x[inside]-bs[j[inside]])
            return v
        valid=(r['span'][0]<=lo)&(hi<=r['span'][1])&(coverage(hi)-coverage(lo)==hi-lo)
        p=r.get('lattice_positions',r['positions']);prefix=r.get('lattice_ceiling_prefix',r['ceiling_prefix'])
        i,j=np.searchsorted(p,windows.T);ceiling=prefix[j]-prefix[i]
        capable=(j-i>=opt.minimum_opportunities)&(ceiling>floors[r['dataset']]+1e-10)
        m=np.zeros(len(families),dtype=np.int64)
        for k,n in membership[uid].items():
            m[k]=n
        physical=window_physics(r,windows,RescueOptions())==''
        values=np.column_stack((np.ones(len(families)),m>0,capable,capable&(m>0),m>1,m,j-i,ceiling,physical))
        stats[r['stratum']]+=values*valid[:,None]
    for k,f in enumerate(families):
        strata={st:Counter({name:(float(v) if name=='ceiling_sum' else int(v)) for name,v in zip(keys,values[k])})
                for st,values in stats.items()}
        datasets=defaultdict(Counter)
        for st,c in strata.items():
            datasets[st.rsplit(':',1)[0]].update(c)
        output.append(dict(family=f['family_id'],interval=f['interval'],established=f.get('established',False),
            by_stratum={s:_finish(c) for s,c in sorted(strata.items())},
            by_dataset={s:_finish(c) for s,c in sorted(datasets.items())}))
    return output


def compare_counts(left,right,options=None):
    opt=options or ComparisonOptions();reasons=[]
    for name,c in [('left',left),('right',right)]:
        if c.get('covered_units',0)<opt.minimum_covered_units:
            reasons.append(name+'_insufficient_coverage')
        if (c.get('lattice_capable_fraction') or 0)<opt.minimum_capable_fraction:
            reasons.append(name+'_lattice_limited')
    a,b=left.get('fraction'),right.get('fraction')
    difference=b-a if a is not None and b is not None else None
    if difference is None:
        agreement='unavailable'
    elif min(left.get('assigned_units',0),right.get('assigned_units',0))<3:
        agreement='low_detection_counts'
    else:
        agreement='similar_observed_fractions' if abs(difference)<=opt.agreement_difference else 'discordant_observed_fractions'
    return dict(comparability='not_comparable' if reasons else 'lattice_comparable',
        reasons=reasons,agreement=agreement,difference=difference,
        absolute_difference=abs(difference) if difference is not None else None,
        left=left,right=right,
        semantics='Lattice comparability is necessary threshold reachability, not equal sensitivity or calibrated occupancy; agreement is descriptive and does not select classes.')


def comparison_rows(counts,options=None):
    rows=[]
    for f in counts:
        pairs=[('assay',a,b,f['by_dataset'][a],f['by_dataset'][b]) for a,b in combinations(f['by_dataset'],2)]
        pairs += [('strand',a,b,f['by_stratum'][a],f['by_stratum'][b]) for a,b in combinations(f['by_stratum'],2)
                  if a.rsplit(':',1)[0]==b.rsplit(':',1)[0]]
        for kind,a,b,left,right in pairs:
            rows.append(dict(family=f['family'],interval=f['interval'],established=f.get('established',False),comparison_kind=kind,
                left_label=a,right_label=b,**compare_counts(left,right,options)))
    return rows
