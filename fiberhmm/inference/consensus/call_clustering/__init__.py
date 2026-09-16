"""Deterministic native-call harmonization; experimental rescue is opt-in only.

No population likelihood fit, pooled presence call, random draws, grid, or
depth-dependent component splitting. Repeated intervals carry integer weights
through average linkage. Native calls remain the only source of family support.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, replace
import gzip
import hashlib
import heapq
import json
import math
import time
import numpy as np


@dataclass(frozen=True)
class ClusterOptions:
    cut: float = .5
    min_calls: int = 3
    max_similarity: float = .7
    max_length_ratio: float = 3.
    lattice_tolerance: int = 2
    max_call_bp: int = 100
    edge_tolerance_bp: int = 10
    stratum_min_calls: int = 3

    def __post_init__(self):
        if not 0 <= self.cut < 1 or not 0 <= self.max_similarity <= 1:
            raise ValueError('Clustering fractions must be in [0,1], with cut < 1')
        if min(self.min_calls, self.stratum_min_calls, self.max_call_bp) < 1:
            raise ValueError('Call counts and maximum length must be positive')
        if self.max_length_ratio < 1 or min(self.edge_tolerance_bp, self.lattice_tolerance) < 0:
            raise ValueError('Invalid length ratio or lattice/edge tolerance')


@dataclass(frozen=True)
class RescueOptions:
    minimum_source_units: int = 3
    minimum_llr: float = math.log(2.)
    near_floor_nats: float = 1.
    allow_one_hit: bool = True
    allow_sparse: bool = True
    minimum_sites: int = 1
    maximum_alignment_gap_bp: int = 0
    minimum_msp_bp: int = 0
    source_mode: str = 'auto'
    supported_null_tail: float = .01

    def __post_init__(self):
        if self.minimum_source_units < 1 or self.minimum_sites < 1:
            raise ValueError('Source support and site counts must be positive')
        if not math.isfinite(self.minimum_llr) or self.minimum_llr <= 0:
            raise ValueError('Rescue must require positive, finite recipient evidence')
        if not math.isfinite(self.near_floor_nats) or self.near_floor_nats < 0:
            raise ValueError('Near-floor allowance must be finite and nonnegative')
        if min(self.maximum_alignment_gap_bp, self.minimum_msp_bp) < 0:
            raise ValueError('Physical domain limits must be nonnegative')
        if self.source_mode not in ('auto', 'same_strand', 'opposite_strand', 'same_dataset', 'all_datasets'):
            raise ValueError('Unknown source mode')
        if not 0 < self.supported_null_tail < 1:
            raise ValueError('Supported null-tail threshold must be in (0,1)')


def read_json(path):
    with (gzip.open(path, 'rt') if str(path).endswith('.gz') else open(path)) as handle:
        return json.load(handle)


def write_json(path, value):
    with (gzip.open(path, 'wt') if str(path).endswith('.gz') else open(path, 'w')) as handle:
        json.dump(value, handle, separators=(',', ':'), allow_nan=False)


def merged(intervals):
    result = []
    for a, b in sorted(intervals):
        if b <= a:
            raise ValueError('Intervals must be positive, half open spans')
        if result and a <= result[-1][1]:
            result[-1][1] = max(result[-1][1], b)
        else:
            result.append([int(a), int(b)])
    return result


def prepare(payloads):
    """Read-only transport. Reference-coordinate observations are not query replay."""
    if not payloads:
        raise ValueError('At least one evidence payload is required')
    chroms = {p['region'].get('chrom') for p in payloads}
    if len(chroms) != 1:
        raise ValueError('Cannot compare different chromosomes')
    lo = max(p['region']['start'] for p in payloads)
    hi = min(p['region']['end'] for p in payloads)
    if lo >= hi:
        raise ValueError('Payload regions do not intersect')
    reads = {}; floors = {}
    for payload in payloads:
        for s in payload['strata']:
            ds = s['dataset_id']; floor = s['model_manifest'].get('native_minimum_llr')
            if floor is None or not math.isfinite(floor) or floor < 0:
                raise ValueError(f'{ds}: a nonnegative native caller floor is required')
            if ds in floors and floors[ds] != floor:
                raise ValueError(f'{ds}: conflicting caller floors')
            floors[ds] = float(floor)
            for u in s['units']:
                uid = u['unit_id']
                if uid in reads:
                    raise ValueError(f'Duplicate evidence unit {uid}')
                p = np.asarray(u['positions'], dtype=np.int64)
                h = np.asarray(u['hits'], dtype=float)
                pa = np.asarray(u['p_accessible'], dtype=float)
                pp = np.asarray(u['p_protected'], dtype=float)
                if not (p.ndim == h.ndim == pa.ndim == pp.ndim == 1 and len(p) == len(h) == len(pa) == len(pp)):
                    raise ValueError(f'{uid}: malformed observations')
                if np.any(np.diff(p) <= 0) or np.any(~np.isfinite(h)):
                    raise ValueError(f'{uid}: positions must be unique and sorted, hits finite')
                if np.any(~np.isfinite(pa)) or np.any(~np.isfinite(pp)) or np.any((pa <= 0) | (pa >= 1) | (pp <= 0) | (pp >= 1)):
                    raise ValueError(f'{uid}: invalid conditional probabilities')
                nucs = merged(u.get('raw_nuc_intervals', []))
                regional = (p >= lo) & (p < hi)
                lattice_positions = p[regional].copy()
                lattice_ceiling = np.maximum(np.log(pp[regional])-np.log(pa[regional]),
                    np.log1p(-pp[regional])-np.log1p(-pa[regional]))
                keep = (p >= lo) & (p < hi)
                for a, b in nucs:
                    keep &= ~((p >= a) & (p < b))
                p, h, pa, pp = p[keep], h[keep] > 0, pa[keep], pp[keep]
                hit_llr = np.log(pp)-np.log(pa)
                miss_llr = np.log1p(-pp)-np.log1p(-pa)
                steps = np.where(h, hit_llr, miss_llr)
                # fold_group_id is the upstream independent-unit contract. No claim
                # of physical duplex pairing is inferred from CT/GA strand labels.
                group = str(u.get('fold_group_id', uid))
                reads[uid] = dict(unit=uid, dataset=ds, strand=u['strand'], stratum=f'{ds}:{u["strand"]}',
                    group=(ds, group), positions=p, hits=h, pa=pa, pp=pp, steps=steps,
                    lattice_positions=lattice_positions,lattice_ceiling_prefix=np.r_[0.,np.cumsum(lattice_ceiling)],
                    hit_llr=hit_llr, miss_llr=miss_llr, prefix=np.r_[0., np.cumsum(steps)],
                    ceiling_prefix=np.r_[0., np.cumsum(np.maximum(hit_llr, miss_llr))],
                    span=(int(u['reference_start']), int(u['reference_end'])),
                    nucs=nucs, msps=merged(u.get('msp_intervals', [])),
                    blocks=merged(u.get('aligned_blocks', [])),
                    calls=u.get('native_multi_interval_calls', []))
    return reads, floors, dict(chrom=next(iter(chroms)), start=int(lo), end=int(hi))


def overlap_fraction(a, b):
    return max(0, min(a[1], b[1])-max(a[0], b[0])) / min(a[1]-a[0], b[1]-b[0])


def median_interval(calls):
    return [int(np.median([c['interval'][0] for c in calls])),
            int(np.median([c['interval'][1] for c in calls]))]


def stratum_intervals(calls, interval, opt):
    lo, hi = interval; width = hi-lo; by = defaultdict(list)
    for c in calls:
        by[c['stratum']].append(c)
    result = {}
    for st, rows in sorted(by.items()):
        a, b = median_interval(rows) if len(rows) >= opt.stratum_min_calls else interval
        a = int(np.clip(a, lo-opt.edge_tolerance_bp, lo+opt.edge_tolerance_bp))
        b = int(np.clip(b, hi-opt.edge_tolerance_bp, hi+opt.edge_tolerance_bp))
        if b-a < max(1, math.ceil(width/2)):
            centre = int(np.clip(np.median([(c['interval'][0]+c['interval'][1])//2 for c in rows]),
                                 lo+width//2-opt.edge_tolerance_bp, lo+width//2+opt.edge_tolerance_bp))
            a = centre-width//2; b = a+width
        result[st] = [a, b]
    return result


def geometry_distance(a, b, rows_a, rows_b, reads, opt):
    share = overlap_fraction(a, b)
    if share == 0:
        return 1.
    if max(a[1]-a[0], b[1]-b[0]) <= opt.max_length_ratio*min(a[1]-a[0], b[1]-b[0]):
        return 1.-share
    short, long, rows = (a, b, rows_a) if a[1]-a[0] <= b[1]-b[0] else (b, a, rows_b)
    excused = 0
    for c in rows:
        p = reads[c['unit']]['positions']
        ia, ib = np.searchsorted(p, long)
        ja, jb = np.searchsorted(p, [max(short[0], long[0]), min(short[1], long[1])])
        excused += (ib-ia)-(jb-ja) <= opt.lattice_tolerance
    return 1.-share*excused/len(rows)


def weighted_clusters(calls, reads, opt):
    """Average linkage with sparse distance storage and repeated-span weights.

    Missing distances equal 1. Store ALL distances below 1, including distances
    above the cut: one may fall below the cut after a weighted update. Immutable
    initial coordinate ordering and heap generations make ties reproducible.
    """
    grouped = defaultdict(list)
    for c in calls:
        grouped[tuple(c['interval'])].append(c)
    ivs = sorted(grouped)
    members = [grouped[iv] for iv in ivs]
    masses = [len(v) for v in members]
    adj = [dict() for _ in ivs]; versions = [0]*len(ivs); live = set(range(len(ivs))); heap = []
    active = []
    for j, b in enumerate(ivs):
        active = [i for i in active if ivs[i][1] > b[0]]
        for i in active:
            d = geometry_distance(ivs[i], b, members[i], members[j], reads, opt)
            if d < 1:
                adj[i][j] = adj[j][i] = d
                if d <= opt.cut:
                    heap.append((d, i, j, 0, 0))
        active.append(j)
    heapq.heapify(heap)
    while heap:
        d, i, j, vi, vj = heapq.heappop(heap)
        if i not in live or j not in live or vi != versions[i] or vj != versions[j]:
            continue
        wi, wj = masses[i], masses[j]
        neighbors = (set(adj[i]) | set(adj[j]))-{i,j}
        updated = {k: (wi*adj[i].get(k,1.)+wj*adj[j].get(k,1.))/(wi+wj) for k in neighbors}
        for k in neighbors:
            adj[k].pop(i, None); adj[k].pop(j, None)
        adj[i] = {}; adj[j] = {}; versions[i] += 1; live.remove(j)
        members[i].extend(members[j]); members[j] = []; masses[i] += masses[j]
        for k, value in sorted(updated.items()):
            if value < 1:
                adj[i][k] = adj[k][i] = value
                if value <= opt.cut:
                    a,b = sorted((i,k))
                    heapq.heappush(heap, (value,a,b,versions[a],versions[b]))
    return [members[i] for i in sorted(live)]


def fold_clusters(groups, opt):
    """One star per survivor: a receiving anchor cannot subsequently fold.

    Every comparison uses the ORIGINAL group medians. A small group with no
    overlapping survivor remains a native-only, unestablished family.
    """
    intervals = [median_interval(g) for g in groups]
    order = sorted(range(len(groups)), key=lambda i: (-len(groups[i]), intervals[i],
                   sorted((c['unit'],c['ordinal']) for c in groups[i])))
    owner = {}; result = []
    for anchor in order:
        if anchor in owner:
            continue
        owner[anchor] = anchor; rows = list(groups[anchor])
        for donor in order:
            if donor in owner:
                continue
            share = overlap_fraction(intervals[anchor], intervals[donor])
            small = len(groups[donor]) < opt.min_calls
            threshold = 1.-opt.cut if small else opt.max_similarity
            a,b = intervals[anchor], intervals[donor]
            ratio_ok = max(a[1]-a[0],b[1]-b[0]) <= opt.max_length_ratio*min(a[1]-a[0],b[1]-b[0])
            if share > 0 and share >= threshold and (ratio_ok or share == 1):
                owner[donor] = anchor; rows.extend(groups[donor])
        result.append(rows)
    return result


def cluster_calls(reads, region, options=None, scopes=None):
    opt = options or ClusterOptions(); calls = []
    for uid, r in sorted(reads.items()):
        for ordinal, c in enumerate(r['calls']):
            a,b = map(int, c['interval'])
            if b <= a:
                raise ValueError('Malformed native call')
            # Partially covered calls cannot donate a complete regional geometry.
            if a < region['start'] or b > region['end'] or b-a > opt.max_call_bp:
                continue
            calls.append(dict(unit=uid, ordinal=ordinal, interval=[a,b], stratum=r['stratum'], dataset=r['dataset']))
    strata = sorted({r['stratum'] for r in reads.values()}); datasets = sorted({r['dataset'] for r in reads.values()})
    sets = {**{'CR '+s:[c for c in calls if c['stratum']==s] for s in strata},
            **{'SR '+s:[c for c in calls if c['dataset']==s] for s in datasets}, 'XCR':calls}
    result = {}
    for name, rows in sets.items():
        if scopes is not None and name not in scopes:
            continue
        families = []
        for group in fold_clusters(weighted_clusters(rows, reads, opt), opt):
            interval = median_interval(group)
            members = sorted([[c['unit'],c['ordinal']] for c in group])
            identity = hashlib.sha256(json.dumps([name,members],separators=(',',':')).encode()).hexdigest()[:16]
            families.append(dict(family_id='CC_'+identity, interval=interval, calls=len(group),
                units=len({c['unit'] for c in group}), established=len({reads[c['unit']]['group'] for c in group})>=opt.min_calls,
                interval_by_stratum=stratum_intervals(group,interval,opt),
                by_stratum=dict(sorted(Counter(c['stratum'] for c in group).items())), members=members))
        result[name] = sorted(families,key=lambda f:(f['interval'],f['family_id']))
    return dict(schema='call-clustering.v2', region=region, options=asdict(opt), results=result)


def score_window(read, interval):
    a,b = np.searchsorted(read['positions'], interval)
    return float(read['prefix'][b]-read['prefix'][a]), float(read['ceiling_prefix'][b]-read['ceiling_prefix'][a]), int(a), int(b)


def rescue_decision(steps, hit_llr, miss_llr, hits, floor, options=None):
    """A fixed-window compatibility rule; not a posterior or calibrated FDR.

    The one-hit counterfactual is a diagnostic only. The reported score is
    always the actual observed score; no observation is changed or imputed.
    """
    opt = options or RescueOptions(); evidence = float(np.sum(steps))
    ceiling = float(np.maximum(hit_llr,miss_llr).sum())
    gain = np.maximum(miss_llr-hit_llr,0.)
    one_hit_gain = float(np.max(gain[hits])) if np.any(hits) else 0.
    if len(steps) < opt.minimum_sites:
        return None, evidence, ceiling, one_hit_gain
    if evidence < opt.minimum_llr:
        return None, evidence, ceiling, one_hit_gain
    if evidence >= floor:
        reason = 'above_floor'
    elif evidence >= floor-opt.near_floor_nats:
        reason = 'near_floor'
    elif opt.allow_one_hit and one_hit_gain > 0 and evidence+one_hit_gain >= floor:
        reason = 'one_hit_short'
    elif opt.allow_sparse and ceiling < floor and not np.any(hits) and np.all(miss_llr > 0):
        reason = 'sparse_lattice'
    else:
        reason = None
    return reason,evidence,ceiling,one_hit_gain


def physical_reason(read, interval, opt):
    a,b = interval
    if any(x < b and y > a for x,y in read['nucs']):
        return 'nucleosome_overlap'
    if not any(x <= a and b <= y and y-x >= opt.minimum_msp_bp for x,y in read['msps']):
        return 'outside_msp'
    covered = sum(max(0,min(b,y)-max(a,x)) for x,y in read['blocks'])
    if covered == 0 or b-a-covered > opt.maximum_alignment_gap_bp:
        return 'alignment_gap'
    return None


def accessible_lower_tail(probabilities, observed_hits):
    """Exact P(K <= observed_hits) for independent Bernoulli opportunities.

    Truncated Poisson-binomial recursion, O(sites * (hits+1)); zero-hit windows
    are a product. This conditional-model diagnostic is neither a posterior nor
    an empirical FDR, and never converts source prevalence into presence.
    """
    p=np.asarray(probabilities,dtype=float);k=int(observed_hits)
    if k<0:return 0.
    if k>=len(p):return 1.
    if k==0:return float(np.exp(np.log1p(-p).sum()))
    mass=np.zeros(k+1);mass[0]=1.
    for probability in p:
        mass[1:]=mass[1:]*(1-probability)+mass[:-1]*probability
        mass[0]*=1-probability
    return float(np.clip(mass.sum(),0,1))


def source_support(family, recipient, reads, mode):
    sources = {uid for uid,_ in family['members']}
    groups = defaultdict(set)
    for uid in sources:
        r = reads[uid]
        if r['group'] == recipient['group']:
            continue
        same_ds = r['dataset'] == recipient['dataset']
        same_st = r['strand'] == recipient['strand']
        if mode == 'same_strand' and not (same_ds and same_st):
            continue
        if mode == 'opposite_strand' and not (same_ds and not same_st):
            continue
        if mode == 'same_dataset' and not same_ds:
            continue
        groups['same_strand' if same_ds and same_st else 'opposite_strand' if same_ds else 'other_dataset'].add(r['group'])
    return {key:len(value) for key,value in sorted(groups.items())}, len(set().union(*groups.values())) if groups else 0


def window_physics(read, windows, opt):
    """Vectorized physical eligibility; half-open intervals, no masked stitching."""
    n=len(windows);lo=windows[:,0];hi=windows[:,1]
    def contained(domains, minimum_width=0):
        if not domains:return np.zeros(n,bool)
        spans=np.asarray(domains);i=np.searchsorted(spans[:,0],lo,side='right')-1
        j=np.maximum(i,0)
        return (i>=0)&(spans[j,1]>=hi)&(spans[j,1]-spans[j,0]>=minimum_width)
    reasons=np.full(n,'',dtype=object)
    aligned=contained(read['blocks'])
    if opt.maximum_alignment_gap_bp:
        spans=np.asarray(read['blocks'])
        if len(spans):
            prefix=np.r_[0,np.cumsum(spans[:,1]-spans[:,0])]
            def coverage(x):
                i=np.searchsorted(spans[:,1],x,side='right')
                return prefix[i]+np.where(i<len(spans),np.maximum(0,x-spans[np.minimum(i,len(spans)-1),0]),0)
            covered=coverage(hi)-coverage(lo)
            aligned=(covered>0)&((hi-lo-covered)<=opt.maximum_alignment_gap_bp)
    reasons[~aligned]='alignment_gap'
    reasons[~contained(read['msps'],opt.minimum_msp_bp)]='outside_msp'
    if read['nucs']:
        nucs=np.asarray(read['nucs']);i=np.searchsorted(nucs[:,1],lo,side='right')
        overlap=(i<len(nucs))&(nucs[np.minimum(i,len(nucs)-1),0]<hi)
        reasons[overlap]='nucleosome_overlap'
    return reasons


def rescue_families(reads, floors, families, options=None, scope='XCR', retain_all=False):
    opt = options or RescueOptions()
    if opt.source_mode=='auto':
        opt=replace(opt,source_mode='all_datasets' if scope=='XCR' else 'same_strand' if scope.startswith('CR ') else 'same_dataset')
    seen=set()
    for family in families:
        if family['interval'][1]<=family['interval'][0]:raise ValueError('Invalid family interval')
        for uid,ordinal in family['members']:
            if uid not in reads or not 0<=ordinal<len(reads[uid]['calls']):
                raise ValueError('Family membership does not match this evidence payload')
            if (uid,ordinal) in seen:raise ValueError('A native call cannot be a member of two families within a scope')
            seen.add((uid,ordinal))
    # Membership is the call ledger, never an overlap test.
    member_sets = [{uid for uid,_ in f['members']} for f in families]
    summaries = [dict(family=f['family_id'],interval=f['interval'],counts=Counter(),by_stratum=defaultdict(Counter),
                     rescue_support=Counter(),rescue_support_by_stratum=defaultdict(Counter)) for f in families]
    records = []; total = Counter(); source_tables={}; windows_by_stratum={}
    states=['unmeasurable','unsupported','competing_call','member','rescued','ambiguous']
    state_code={state:i for i,state in enumerate(states)}
    # Counts are shared across recipients. Only the excluded source group varies.
    for st in sorted({r['stratum'] for r in reads.values()}):
        recipient=next(r for r in reads.values() if r['stratum']==st)
        windows_by_stratum[st]=np.array([f['interval_by_stratum'].get(st,f['interval']) for f in families],dtype=np.int64).reshape(-1,2)
        for k,f in enumerate(families):
            groups=defaultdict(set)
            for uid in {uid for uid,_ in f['members']}:
                r=reads[uid];same_ds=r['dataset']==recipient['dataset'];same_st=r['strand']==recipient['strand']
                if opt.source_mode=='same_strand' and not(same_ds and same_st):continue
                if opt.source_mode=='opposite_strand' and not(same_ds and not same_st):continue
                if opt.source_mode=='same_dataset' and not same_ds:continue
                groups['same_strand' if same_ds and same_st else 'opposite_strand' if same_ds else 'other_dataset'].add(r['group'])
            source_tables[(k,st)]=(groups,set().union(*groups.values()) if groups else set())
    for uid,r in sorted(reads.items()):
        if scope.startswith('CR ') and r['stratum'] != scope[3:]:
            continue
        if scope.startswith('SR ') and r['dataset'] != scope[3:]:
            continue
        rows = [];windows=windows_by_stratum[r['stratum']]
        physical_reasons=window_physics(r,windows,opt)
        starts=np.searchsorted(r['positions'],windows[:,0]);ends=np.searchsorted(r['positions'],windows[:,1])
        evidences=r['prefix'][ends]-r['prefix'][starts];ceilings=r['ceiling_prefix'][ends]-r['ceiling_prefix'][starts]
        overlapping=np.zeros(len(windows),bool)
        if r['calls']:
            for ca,cb in merged([c['interval'] for c in r['calls']]):
                overlapping|=(windows[:,0]<cb)&(windows[:,1]>ca)
        for k,f in enumerate(families):
            iv = f['interval_by_stratum'].get(r['stratum'],f['interval'])
            if r['span'][0] > iv[0] or r['span'][1] < iv[1]:
                continue
            ev,ce,a,b = float(evidences[k]),float(ceilings[k]),int(starts[k]),int(ends[k]); floor = floors[r['dataset']]
            row = dict(family=f['family_id'],interval=iv,evidence=ev,ceiling=ce,n_sites=b-a,
                       hits=int(r['hits'][a:b].sum()),floor=floor,margin=ev-floor)
            if uid in member_sets[k]:
                row.update(state='member',reason='native_call_member')
            elif overlapping[k]:
                row.update(state='competing_call',reason='native_call_in_another_family_or_scale')
            else:
                physical = physical_reasons[k]
                if physical or b-a < opt.minimum_sites:
                    row.update(state='unmeasurable',reason=physical or 'no_lattice_information')
                else:
                    groups,all_groups=source_tables[(k,r['stratum'])]
                    support={key:len(value)-(r['group'] in value) for key,value in groups.items()}
                    n=len(all_groups)-(r['group'] in all_groups)
                    reason,ev,ce,gain = rescue_decision(r['steps'][a:b],r['hit_llr'][a:b],r['miss_llr'][a:b],r['hits'][a:b],floor,opt)
                    row.update(evidence=ev,ceiling=ce,margin=ev-floor,source_support=support,
                               source_units=n,one_hit_gain=gain)
                    if n < opt.minimum_source_units:
                        row.update(state='unsupported',reason='insufficient_independent_source_support')
                    elif reason:
                        tail=accessible_lower_tail(r['pa'][a:b],int(r['hits'][a:b].sum()))
                        row.update(state='rescued',reason=reason,accessible_null_tail=tail,
                            rescue_support='supported' if tail<=opt.supported_null_tail else 'provisional')
                    else:
                        row.update(state='unsupported',reason='recipient_evidence_insufficient')
            row['_index']=k; rows.append(row)
        # Overlapping successful rescue windows do not create two presences.
        # Retain the ambiguity explicitly instead of picking a family by depth.
        candidates = [row for row in rows if row['state']=='rescued']
        ambiguous = set()
        for i,left in enumerate(candidates):
            for right in candidates[i+1:]:
                if left['interval'][0] < right['interval'][1] and right['interval'][0] < left['interval'][1]:
                    ambiguous.update([left['family'],right['family']])
        for row in rows:
            if row['family'] in ambiguous:
                row.update(state='ambiguous',rescue_reason=row['reason'],reason='overlapping_family_hypotheses')
            k=row.pop('_index'); state=row['state']; summaries[k]['counts'][state]+=1
            summaries[k]['by_stratum'][r['stratum']][state]+=1; total[state]+=1
            if state=='rescued':
                summaries[k]['rescue_support'][row['rescue_support']]+=1
                summaries[k]['rescue_support_by_stratum'][r['stratum']][row['rescue_support']]+=1
        records.append(dict(unit_id=uid,stratum=r['stratum'],
            family_indices=[next_k for next_k,f in enumerate(families) if r['span'][0]<=f['interval_by_stratum'].get(r['stratum'],f['interval'])[0]
                            and r['span'][1]>=f['interval_by_stratum'].get(r['stratum'],f['interval'])[1]],
            state_codes=[state_code[row['state']] for row in rows],
            observations=rows if retain_all else [row for row in rows if row['state'] in ('member','rescued','ambiguous')]))
    for summary in summaries:
        for counts in [summary['counts'],*summary['by_stratum'].values()]:
            # All covered observations are retained in the denominator ledger.
            counts['covered']=sum(counts.values())
        def fraction(counts,support):
            denominator=sum(counts.get(key,0) for key in ('member','rescued','unsupported'))
            return dict(denominator=denominator,native_only=counts.get('member',0)/denominator if denominator else None,
                with_supported_rescue=(counts.get('member',0)+support.get('supported',0))/denominator if denominator else None,
                including_provisional=(counts.get('member',0)+counts.get('rescued',0))/denominator if denominator else None)
        summary['detection_fraction']=fraction(summary['counts'],summary['rescue_support'])
        summary['detection_fraction_by_stratum']={st:fraction(counts,summary['rescue_support_by_stratum'].get(st,{}))
                                                for st,counts in summary['by_stratum'].items()}
    return dict(schema='call-clustering-rescue.v2',scope=scope,options=asdict(opt),floors=floors,
        semantics='Positive recipient evidence plus independent native source calls. Supported means exact conditional accessible hit-count tail <= the configured threshold; provisional candidates remain explicit. Detection fractions are not occupancy estimates, posteriors, FDR, or proof of absence. Unmeasurable, competing, and ambiguous cases are excluded from the detection denominator and separately counted.',
        counts=dict(total),state_codebook=states,observation_detail='all' if retain_all else 'member_rescued_ambiguous',
        families=summaries,records=records)


def cooccurrence(rescue):
    counts=defaultdict(Counter)
    for record in rescue['records']:
        rows=[r for r in record['observations'] if r['state'] in ('member','rescued')]
        rows.sort(key=lambda r:r['family'])
        for i,left in enumerate(rows):
            for right in rows[i+1:]:
                source=('native_native' if left['state']==right['state']=='member' else
                    'includes_provisional_rescue' if any(v.get('rescue_support')=='provisional' for v in (left,right)) else
                    'includes_supported_rescue')
                counts[(record['stratum'],left['family'],right['family'])][source]+=1
    return [dict(stratum=st,families=[a,b],**c) for (st,a,b),c in sorted(counts.items())]


def run_call_clustering(payloads, cluster_options=None, rescue_options=None, rescue_scopes=()):
    """Opt-in kernel entry point. Does not invoke the fitted/Monte Carlo workflow.

    CR, SR and XCR use the same implementation on different input scopes.
    Rescue scopes may be empty (identity/boundaries only), or any result keys.
    """
    reads,floors,region=prepare(payloads)
    result=cluster_calls(reads,region,cluster_options)
    result['rescue']={}
    for scope in rescue_scopes:
        if scope not in result['results']:raise ValueError(f'Unknown rescue scope {scope}')
        rescued=rescue_families(reads,floors,result['results'][scope],rescue_options,scope)
        rescued['cooccurrence']=cooccurrence(rescued)
        result['rescue'][scope]=rescued
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--evidence',nargs='+',required=True)
    ap.add_argument('--out',required=True)
    ap.add_argument('--families',help='Use an existing v2 family result')
    ap.add_argument('--rescue-out')
    ap.add_argument('--set',default='XCR')
    ap.add_argument('--cut',type=float,default=.5)
    ap.add_argument('--min-calls',type=int,default=3)
    ap.add_argument('--max-similarity',type=float,default=.7)
    ap.add_argument('--minimum-rescue-llr',type=float,default=math.log(2.))
    ap.add_argument('--near-floor-nats',type=float,default=1.)
    ap.add_argument('--minimum-source-units',type=int,default=3)
    ap.add_argument('--source-mode',choices=['auto','same_strand','opposite_strand','same_dataset','all_datasets'],default='auto')
    ap.add_argument('--supported-null-tail',type=float,default=.01)
    ap.add_argument('--no-one-hit',action='store_true')
    ap.add_argument('--no-sparse',action='store_true')
    args=ap.parse_args(); t=time.monotonic()
    reads,floors,region=prepare([read_json(p) for p in args.evidence])
    opt=ClusterOptions(cut=args.cut,min_calls=args.min_calls,max_similarity=args.max_similarity)
    result=read_json(args.families) if args.families else cluster_calls(reads,region,opt)
    write_json(args.out,result)
    print(json.dumps(dict(elapsed=round(time.monotonic()-t,3),families={k:len(v) for k,v in result['results'].items()})),flush=True)
    if args.rescue_out:
        ropt=RescueOptions(minimum_llr=args.minimum_rescue_llr,near_floor_nats=args.near_floor_nats,
            minimum_source_units=args.minimum_source_units,source_mode=args.source_mode,
            allow_one_hit=not args.no_one_hit,allow_sparse=not args.no_sparse,supported_null_tail=args.supported_null_tail)
        rescue=rescue_families(reads,floors,result['results'][args.set],ropt,args.set)
        rescue['cooccurrence']=cooccurrence(rescue)
        write_json(args.rescue_out,rescue)
        print(json.dumps(dict(elapsed=round(time.monotonic()-t,3),counts=rescue['counts'])),flush=True)


if __name__=='__main__':
    main()
