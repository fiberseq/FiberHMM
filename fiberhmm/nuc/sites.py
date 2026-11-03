"""Shared positioned sites and accessible barriers, discovered on held-out molecules.

Coordinates are reference bp, half open. Discovery uses hash half 0 only; all
columns use the same sites, and half 1 supplies the independent positioning
estimate. A position is callable only on aligned, non-no-call bases. Smoothed
dyad hits and callable bases have exactly the same kernel and grid.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from itertools import combinations
from concurrent.futures import ThreadPoolExecutor
import contextvars
import time

import numpy as np
from numba import njit
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, peak_widths
from scipy.stats import fisher_exact, norm, t as student_t

from fiberhmm.assoc.nucleosome import hash_half, molecule_key, nuc_state, condition_stats
from fiberhmm.nuc.density import edge_truncated
from fiberhmm.nuc.params import DEFAULT_PARAMS
from fiberhmm.assoc.control import cooperative


@dataclass(frozen=True)
class SitesParams:
    grid_bp: int = 5
    smooth_bp: float = 15
    min_separation_bp: int = 120
    min_coverage: int = 10
    positioning_bp: int = 20
    fuzziness_cap_bp: int = 60
    # Absolute molecular length, deliberately independent of coarse size calls.
    mono_length_window: tuple = (90, 230)
    anchor_separation_bp: int = 350
    anchor_min_fraction: float = .10
    baseline_positions: int = 256
    coverage_tolerance: float = .20
    phasing_grid_bp: int = 250
    phasing_grid_max: int = 256
    n_perm: int = 199
    n_boot: int = 199
    seed: int = 17

    def __post_init__(self):
        if (self.grid_bp <= 0 or self.smooth_bp <= 0 or self.min_coverage < 1
                or self.min_separation_bp < self.grid_bp or self.anchor_separation_bp < self.grid_bp
                or self.baseline_positions < 1 or self.phasing_grid_bp < 1 or self.phasing_grid_max < 2 or self.n_perm < 1 or self.n_boot < 1
                or not 0 <= self.coverage_tolerance < 1
                or not 0 < self.mono_length_window[0] < self.mono_length_window[1]):
            raise ValueError('Invalid sites parameters')


def molecular_dyads(mol, length_window=(90, 230)):
    iv = mol.nucs
    if not len(iv):
        return np.empty(0)
    length = iv[:, 1] - iv[:, 0]
    keep = (length >= length_window[0]) & (length <= length_window[1]) & ~edge_truncated(mol, iv)
    return iv[keep].mean(axis=1)


def callable_positions(mol, positions):
    """Round-trip projection rejects deletions (ref_to_mol alone bridges them)."""
    positions = np.asarray(positions, dtype=np.int64)
    q = mol.ref_to_mol(positions)
    valid = (q >= 0) & (mol.mol_to_ref(q) == positions)
    for a, b in mol.no_call:
        valid &= ~((q >= a) & (q < b))
    return valid


def site_state(mol, pos, params=DEFAULT_PARAMS):
    """open / bound / occluded / unclassified. Bound means open with a TF call.

    Open uses condition_stats' >=80 bp MSP spanning +/-30 bp rule, and a
    TF call must overlap that +/-30 bp anchor window. Fractions use all
    callable molecules as denominator, including unclassified molecules.
    """
    if not callable_positions(mol, [pos])[0]:
        return 'unclassified'
    a = int(mol.ref_to_mol([pos])[0])
    state = nuc_state(mol, a, params)
    if state == '1':
        tf = mol.tfs
        return 'bound' if len(tf) and np.any((tf[:, 0] < a + params.barrier_slop)
                                            & (tf[:, 1] > a - params.barrier_slop)) else 'open'
    return 'occluded' if state == '0' else 'unclassified'


def _anchor_states(mol, positions, valid, params):
    """Vectorized version of site_state and the selected occluder's reference dyad."""
    a = mol.ref_to_mol(positions)
    ms, nc, tf = mol.msps, mol.nucs, mol.tfs
    op = ((ms[None,:,0] <= a[:,None]-params.barrier_slop)
          & (ms[None,:,1] >= a[:,None]+params.barrier_slop)
          & ((ms[:,1]-ms[:,0])[None,:] >= params.barrier_min_msp)).any(1)
    bound = ((tf[None,:,0] < a[:,None]+params.barrier_slop)
              & (tf[None,:,1] > a[:,None]-params.barrier_slop)).any(1)
    cl = (nc[None,:,0] <= a[:,None]) & (nc[None,:,1] > a[:,None])
    closed = cl.any(1)
    state = np.where(valid & op,np.where(bound,1,0),np.where(valid & closed,2,3))
    occluder = np.full(len(a),np.nan)
    if len(nc):
        centres = mol.mol_to_ref(nc[cl.argmax(1)].mean(1))
        occluder = np.where((state==2) & (centres>=0),centres,np.nan)
    return state,occluder


def _nearest(dyads, positions, cap):
    if not len(dyads):
        return np.full(len(positions), np.nan)
    d = np.sort(dyads)
    i = np.searchsorted(d, positions)
    left, right = d[np.maximum(0, i - 1)], d[np.minimum(len(d) - 1, i)]
    near = np.where(abs(left - positions) <= abs(right - positions), left, right)
    return np.where(abs(near - positions) <= cap, near, np.nan)


@njit(cache=True,nogil=True)
def _callable_bins(blocks,no_call,reverse,length,start,end,step):
    measured=np.zeros(end-start,np.bool_)
    for i in range(len(blocks)):
        q,r,n=blocks[i]
        for j in range(max(start,r),min(end,r+n)):
            measured[j-start]=True
    for k in range(len(no_call)):
        a,b=no_call[k]
        if reverse:
            a,b=length-b,length-a
        for i in range(len(blocks)):
            q,r,n=blocks[i]
            lo,hi=max(q,a),min(q+n,b)
            for j in range(max(start,r+lo-q),min(end,r+hi-q)):
                measured[j-start]=False
    bp=np.zeros((end-start+step-1)//step,np.int64)
    for j in range(end-start):
        if measured[j]:
            bp[j//step]+=1
    return bp


def _bin_bases(mol,start,end,step):
    """Exact aligned callable bases per bin, excluding no-call bases and CIGAR gaps."""
    blocks=mol.blocks if mol.blocks is not None else np.empty((0,3),np.int64)
    return _callable_bins(blocks,mol.no_call,mol.is_reverse,mol.length,start,end,step)


def _add_profiles(ms, refs, rows, valid, states, anchor_column, window, checkpoint):
    """25 bp reference-aligned dyad density, using already decoded dyads.

    Each state uses its own exact callable-base exposure (gaps/no-call bases
    excluded). Binning each molecule once bounds memory independently of depth.
    No coverage is null, rather than a misleading zero-density segment.
    """
    step = 25
    start = max(0, window[1]-1025)
    end = window[2]+1025
    nb = (end-start+step-1)//step
    pending = []
    for j, row in enumerate(rows):
        centres = start + (np.arange(nb)+.5)*step - row['pos']
        indices = np.flatnonzero(abs(centres) <= 1000)
        groups = ('open', 'occluded') if row['kind']=='anchor' else ('all',)
        pending.append((j, indices, dict(x=centres[indices].tolist(), bin_bp=step,
                                       unit='dyads per callable kb',
                                       **{g: np.zeros((2,len(indices))) for g in groups})))
    for i, m in enumerate(ms):
        checkpoint()
        bp = _bin_bases(m, start, end, step)
        d = refs[i]
        d = d[(d>=start) & (d<end)]
        d = d[callable_positions(m,d)]
        counts = np.bincount(((d-start)//step).astype(int), minlength=nb)
        for j, indices, profile in pending:
            if not valid[i,j]:
                continue
            if rows[j]['kind']=='anchor':
                state = states[i,anchor_column[j]]
                if state>2:
                    continue
                group = 'open' if state<=1 else 'occluded'
            else:
                group = 'all'
            profile[group][0] += counts[indices]
            profile[group][1] += bp[indices]
    for j, _, profile in pending:
        for group in ('open','occluded') if rows[j]['kind']=='anchor' else ('all',):
            counts, bases = profile[group]
            profile[group] = np.divide(1000*counts,bases,out=np.full(len(bases),np.nan),where=bases>0).tolist()
        yield j, profile


def _fraction(hit, valid):
    n = int(valid.sum())
    k = int(np.sum(hit & valid))
    if not n:
        return dict(n=0, hits=0, value=None, ci=None)
    p = k / n
    z = 1.95996398454
    mid = (p + z*z/(2*n)) / (1+z*z/n)
    half = z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return dict(n=n, hits=k, value=p, ci=[max(0, mid-half), min(1, mid+half)])


def _mean(values, origin=0):
    a = np.asarray(values)
    a = a[np.isfinite(a)] - origin
    if not len(a):
        return dict(n=0, mean=None, ci=None, sd=None)
    sd = float(a.std(ddof=1)) if len(a) > 1 else None
    half = float(student_t.ppf(.975, len(a)-1)*sd/np.sqrt(len(a))) if sd is not None else None
    return dict(n=len(a), mean=float(a.mean()), sd=sd,
                ci=[float(a.mean()-half), float(a.mean()+half)] if half is not None else None)


def _fraction_difference(x,y):
    if not x['n'] or not y['n']:
        return None
    delta = y['value'] - x['value']
    # Newcombe score interval: retains uncertainty at 0/1 and with small counts.
    ci = [delta - np.hypot(y['value']-y['ci'][0], x['ci'][1]-x['value']),
          delta + np.hypot(y['ci'][1]-y['value'], x['value']-x['ci'][0])]
    p = float(fisher_exact([[x['hits'], x['n']-x['hits']], [y['hits'], y['n']-y['hits']]])[1])
    return dict(difference=delta,ci=ci,p=p)


def _contrast(a, b, chemistry):
    out = _fraction_difference(a['positioning'],b['positioning'])
    if out is None:
        return None
    out.update(metric='positioning', chemistry_contrast=chemistry,
               caveat='chemistry contrast, not biology' if chemistry else None)
    ma, mb = a['dyad_shift'], b['dyad_shift']
    if min(ma['n'], mb['n']) > 1:
        va, vb = ma['sd']**2/ma['n'], mb['sd']**2/mb['n']
        se = np.sqrt(va+vb)
        df = (va+vb)**2/(va*va/(ma['n']-1)+vb*vb/(mb['n']-1)) if se else 1
        shift = mb['mean']-ma['mean']
        half = float(student_t.ppf(.975, df)*se)
        out['shift'] = dict(difference=shift, ci=[shift-half, shift+half],
                            p=float(2*student_t.sf(abs(shift/se), df)) if se else float(shift == 0))
    return out


def _bh(rows, key='p', out='q'):
    valid = [r for r in rows if r.get(key) is not None and np.isfinite(r[key])]
    ordered = sorted(valid, key=lambda r: r[key], reverse=True)
    q = 1.
    for rank, row in zip(range(len(ordered), 0, -1), ordered):
        q = min(q, row[key]*len(ordered)/rank)
        row[out] = float(q)


def _phasing_evidence(sample,n_perm):
    phase=sample.get('phasing',{}).get('molecule',{})
    delta,prob=phase.get('delta'),phase.get('p')
    if delta is None or prob is None or not np.isfinite(delta) or not np.isfinite(prob):
        return 0.
    return max(0.,delta)*-np.log10(max(prob,1/(n_perm+1)))


@cooperative
def analyse_sites(observations, window, *, chemistries=None, periods=None, params=None,
                  condition_params=None, pool_keys=None, phasing=True, checkpoint=lambda: None):
    """Pool observations, discover once, compare columns. No disk I/O or global cache.

    ``pool_keys`` maps observation names to physical dataset identities, preventing
    overlapping read panels from counting a molecule twice in pooled discovery.
    Per-column estimates remain scoped to each supplied observation. A missing
    chemistry stays unknown, rather than being inferred from the sample name.
    """
    started = time.perf_counter()
    p = params or SitesParams()
    cp = condition_params or DEFAULT_PARAMS
    chrom, start, end = window
    if not chrom or start < 0 or end <= start or end-start > 2_000_000:
        raise ValueError('Give a nonempty sites window of at most 2,000,000 bp')
    chemistries, periods, pool_keys = chemistries or {}, periods or {}, pool_keys or {}
    # Longest record per physical molecule per column, deterministic under input reordering.
    obs = {}
    for key, mols in observations.items():
        chosen = {}
        for m in sorted(mols, key=lambda m: (-m.length, m.record_key())):
            if m.chrom == chrom:
                chosen.setdefault(molecule_key(m), m)
        obs[key] = list(chosen.values())
    pooled = {}
    for key, mols in obs.items():
        for m in mols:
            pooled.setdefault((pool_keys.get(key, key), molecule_key(m)), m)
    mols = list(pooled.values())
    x = np.arange(start, end, p.grid_bp)
    hits, bases, msp = np.zeros(len(x)), np.zeros(len(x)), np.zeros(len(x))
    dy = {}
    for m in {id(m): m for ms in obs.values() for m in ms}.values():
        dy[id(m)] = molecular_dyads(m, p.mono_length_window)
    for m in mols:
        if hash_half(m) != 0:
            continue
        checkpoint()
        bp = _bin_bases(m, start, end, p.grid_bp)
        bases += bp
        d = m.mol_to_ref(dy[id(m)])
        d = d[(d >= start) & (d < end)]
        d = d[callable_positions(m, d)]
        hits[np.unique((d-start)//p.grid_bp)] += 1
        acc = np.zeros(len(x), bool)
        for a, b in m.msps:
            if b-a < cp.barrier_min_msp:
                continue
            iv = m.mol_interval_to_ref(int(a), int(b))
            if iv:
                acc |= (x >= iv[0]) & (x < iv[1])
        msp += acc*bp
    sigma = p.smooth_bp/p.grid_bp
    den = gaussian_filter1d(bases, sigma)
    density = np.divide(gaussian_filter1d(hits, sigma), den, out=np.zeros(len(x)),
                        where=den >= p.min_coverage*p.grid_bp)
    accessible = np.divide(gaussian_filter1d(msp, sigma), den, out=np.zeros(len(x)),
                           where=den >= p.min_coverage*p.grid_bp)
    peaks, _ = find_peaks(density, distance=p.min_separation_bp//p.grid_bp, prominence=.00025, height=.0006)
    anchors, _ = find_peaks(accessible, distance=p.anchor_separation_bp//p.grid_bp,
                            prominence=.04, height=p.anchor_min_fraction)
    rows = [dict(kind='site', pos=int(x[i]), start=int(x[i]-p.positioning_bp), end=int(x[i]+p.positioning_bp+1))
            for i in peaks if start+100 <= x[i] < end-100]
    for i in anchors:
        if not start+100 <= x[i] < end-100:
            continue
        width = peak_widths(accessible, [i], rel_height=.5)
        a, b = max(start, int(start+width[2][0]*p.grid_bp)), min(end, int(start+width[3][0]*p.grid_bp)+1)
        rows.append(dict(kind='anchor', pos=int(x[i]), start=a, end=b,
                         minus1=max((r['pos'] for r in rows if r['kind']=='site' and r['pos'] < a), default=None),
                         plus1=min((r['pos'] for r in rows if r['kind']=='site' and r['pos'] >= b), default=None)))
    rng = np.random.default_rng(p.seed)
    random_pos = np.sort(rng.choice(np.arange(start, end), min(end-start, p.baseline_positions), replace=False))
    positions = np.r_[[r['pos'] for r in rows], random_pos].astype(np.int64)
    all_contrasts = []
    for row in rows:
        row.update(chrom=chrom, id=f"{row['kind']}:{chrom}:{row['pos']}", samples={})
    for key, ms in obs.items():
        checkpoint()
        valid = np.array([callable_positions(m, positions) for m in ms], bool).reshape(len(ms), len(positions))
        refs = [m.mol_to_ref(dy[id(m)]) for m in ms]
        refs = [d[d>=0] for d in refs]
        near = np.array([_nearest(d, positions, p.fuzziness_cap_bp) for d in refs]).reshape(len(ms), len(positions))
        near[~valid] = np.nan
        hit = abs(near-positions) <= p.positioning_bp
        held = np.array([hash_half(m) == 1 for m in ms],dtype=bool)
        anchor_index = [j for j,r in enumerate(rows) if r['kind']=='anchor']
        state_arrays = [_anchor_states(m,positions[anchor_index],valid[i,anchor_index],cp) for i,m in enumerate(ms)]
        state_matrix = np.array([v[0] for v in state_arrays]).reshape(len(ms),len(anchor_index))
        occluder_matrix = np.array([v[1] for v in state_arrays]).reshape(len(ms),len(anchor_index))
        anchor_column = {j:i for i,j in enumerate(anchor_index)}
        nrand = valid[:, len(rows):].sum(0)
        rand_fraction = np.divide(hit[:, len(rows):].sum(0), nrand,
                                  out=np.full(len(random_pos), np.nan), where=nrand > 0)
        nrand_held = valid[held,len(rows):].sum(0)
        rand_held = np.divide(hit[held,len(rows):].sum(0),nrand_held,
                              out=np.full(len(random_pos),np.nan),where=nrand_held>0)
        for j, row in enumerate(rows):
            z = _fraction(hit[:, j], valid[:, j])
            h = _fraction(hit[:, j], valid[:, j] & held)
            matched = (abs(nrand-z['n']) <= max(1, z['n']*p.coverage_tolerance)) & np.isfinite(rand_fraction)
            baseline = rand_fraction[matched]
            mean = float(np.mean(baseline)) if len(baseline) else None
            hmatched = (abs(nrand_held-h['n']) <= max(1,h['n']*p.coverage_tolerance)) & np.isfinite(rand_held)
            hbase = rand_held[hmatched]
            hmean = float(np.mean(hbase)) if len(hbase) else None
            sample = dict(positioning=z, held_out_positioning=h, baseline=mean,
                          held_out_baseline=hmean, held_out_baseline_n=len(hbase),
                          held_out_over_baseline=h['value']/hmean if hmean and h['value'] is not None else None,
                          held_out_percentile=float(100*np.mean(hbase<=h['value'])) if len(hbase) and h['value'] is not None else None,
                          positioning_over_baseline=z['value']/mean if mean and z['value'] is not None else None,
                          baseline_percentile=float(100*np.mean(baseline <= z['value'])) if len(baseline) and z['value'] is not None else None,
                          baseline_n=len(baseline), dyad_shift=_mean(near[:, j], row['pos']),
                          fuzziness_cap_bp=p.fuzziness_cap_bp)
            sample['fuzziness_sd_bp'] = sample['dyad_shift']['sd']
            if row['kind'] == 'anchor':
                col = anchor_column[j]
                sample['states'] = {s: _fraction(state_matrix[:,col] == i, valid[:, j])
                                    for i,s in enumerate(('open', 'bound', 'occluded', 'unclassified'))}
                sample['states']['open_total'] = _fraction(state_matrix[:,col]<=1, valid[:,j])
                sample['occluder_dyad'] = _mean(occluder_matrix[:,col], row['pos'])
                sample['minus1'] = row['minus1']
                sample['plus1'] = row['plus1']
                sample['flanks'] = {}
                sample['flank_positioning'] = {}
                for side in ('minus1','plus1'):
                    fj = next((jj for jj,rr in enumerate(rows) if rr['kind']=='site' and rr['pos']==row[side]),None)
                    sample['flanks'][side] = _mean(near[state_matrix[:,col]<=1,fj],row['pos']) if fj is not None else _mean([])
                    sample['flank_positioning'][side] = _fraction(hit[:,fj],valid[:,fj] & (state_matrix[:,col]<=1)) if fj is not None else _fraction(np.zeros(len(ms),bool),np.zeros(len(ms),bool))
            row['samples'][key] = sample
        for j, profile in _add_profiles(ms,refs,rows,valid,state_matrix,anchor_column,window,checkpoint):
            rows[j]['samples'][key]['profile'] = profile
    if phasing:
        # Columns are independent conditional analyses, with separate random
        # streams. Bound parallelism and propagate cooperative cancellation.
        with ThreadPoolExecutor(max_workers=min(2,max(1,len(obs)))) as executor:
            tasks = [executor.submit(contextvars.copy_context().run,_add_phasing,
                                     ms,key,rows,window,periods.get(key,189.),p,cp,checkpoint,dy)
                     for key,ms in obs.items()]
            for task in tasks:
                task.result()
    for row in rows:
        row['comparisons'] = []
        for a, b in combinations(obs, 2):
            chem = bool(chemistries.get(a) and chemistries.get(b) and chemistries[a] != chemistries[b])
            c = _contrast(row['samples'][a], row['samples'][b], chem)
            if c:
                c.update(a=a, b=b)
                if row['kind']=='anchor':
                    c['states'] = {state:_fraction_difference(row['samples'][a]['states'][state],row['samples'][b]['states'][state])
                                   for state in ('open','bound','occluded','open_total')}
                    pa = row['samples'][a].get('phasing',{}).get('molecule',{})
                    pb = row['samples'][b].get('phasing',{}).get('molecule',{})
                    ca,cb = pa.get('ci') or {},pb.get('ci') or {}
                    if np.isfinite(ca.get('se',np.nan)) and np.isfinite(cb.get('se',np.nan)):
                        delta = pb['delta']-pa['delta']
                        se = np.hypot(ca['se'],cb['se'])
                        c['phasing'] = dict(difference=delta,ci=[delta-1.96*se,delta+1.96*se],
                                             p=float(2*norm.sf(abs(delta/se))) if se else float(delta==0),
                                             method='independent physical-molecule bootstrap SEs, fixed locus')
                if pool_keys.get(a,a)==pool_keys.get(b,b) and set(molecule_key(m) for m in obs[a]) & set(molecule_key(m) for m in obs[b]):
                    c['overlap_note'] = 'Overlapping observations; differences are descriptive'
                    for value in [c,c.get('shift'),c.get('phasing'),*c.get('states',{}).values()]:
                        if value is not None:
                            value.update(ci=None,p=None,q=None)
                row['comparisons'].append(c)
                all_contrasts.append(c)
        row['rank_score'] = max((_phasing_evidence(s,p.n_perm)
                                 for s in row['samples'].values()), default=0) if row['kind']=='anchor' else max(
                                     (s['held_out_over_baseline'] or 0 for s in row['samples'].values()), default=0)
    _bh(all_contrasts)
    _bh([c['shift'] for c in all_contrasts if 'shift' in c])
    for state in ('open','bound','occluded','open_total'):
        _bh([c['states'][state] for c in all_contrasts if c.get('states',{}).get(state)])
    _bh([c['phasing'] for c in all_contrasts if 'phasing' in c])
    return dict(window=dict(chrom=chrom, start=start, end=end), rows=sorted(rows, key=lambda r: (r['kind']!='anchor', -r['rank_score'], r['pos'])),
                track=dict(x=x.tolist(), density=density.tolist(), accessible=accessible.tolist(), callable_bases=den.tolist()),
                discovery=dict(method='physical-molecule hash half 0; evaluation half 1',
                               molecules=len(mols), discovery_n=sum(hash_half(m)==0 for m in mols)),
                parameters=asdict(p), seconds=time.perf_counter()-started)


def _add_phasing(ms, key, rows, window, period, p, cp, checkpoint, dyads):
    """Use the association engine for both the anchor statistic and its locus null.

    Random backgrounds are attempted jointly, respecting separation between
    every discovered anchor. When none qualify, a reported regular locus grid
    is evaluated with the identical condition_stats statistic and state rule.
    No permutation SE is relabelled as an effect confidence interval.
    """
    from fiberhmm.assoc.nucleosome import random_backgrounds, vectorized_anchor_appearances
    from fiberhmm.assoc.condition import LocusTable
    from fiberhmm.assoc import strata as ST
    from fiberhmm.nuc.phasing import _osc_fit_histogram
    chrom, start, end = window
    anchors = [(chrom, r['pos']) for r in rows if r['kind']=='anchor']
    if not anchors or not ms:
        return
    # Make the engine's length eligibility agree exactly with SitesParams.
    cp = replace(cp, mono_frac=(p.mono_length_window[0]/period, p.mono_length_window[1]/period))
    span = int(cp.phase_x0 + cp.phase_summary_periods*period)
    cands, mb = random_backgrounds(ms, anchors, flank=span, span=span, seed=p.seed)
    matched = {anchors[i][1]: [cands[j] for j in js] for i, js in mb['sets']}
    # Assigned backgrounds can still lack both measurable states. Keep a
    # bounded locus grid available for that failure, reporting its actual step.
    grid_step = max(p.phasing_grid_bp,int(np.ceil(max(0,end-start-2*span)/p.phasing_grid_max)))
    grid = list(range(start+span, end-span, grid_step))
    positions = sorted(set([pos for _, pos in anchors] + grid + [pos for bg in matched.values() for _,pos in bg]))
    loci = LocusTable.from_sets([(chrom,pos) for pos in positions], [], [(i,[]) for i in range(len(positions))])
    edges = np.arange(cp.phase_x0,cp.phase_x0+cp.phase_summary_periods*period+10,10.)
    histograms = {frame:np.zeros((len(positions),len(edges)-1)) for frame in ('molecule','anchor')}
    prefetched = {frame: vectorized_anchor_appearances(ms,loci,frame=frame,P=period,params=cp,dyads=dyads,
                                                      summary_only=True,open_histogram=histograms[frame])
                  for frame in ('molecule','anchor')}
    position_index = {pos:i for i,pos in enumerate(positions)}

    def appearances(selected):
        out = {}
        indices = [position_index[pos] for _,pos in selected]
        for frame,(app,bins,info) in prefetched.items():
            sub = app.subset(np.isin(app.locus,indices))
            remap = np.full(len(positions),-1,dtype=np.int64)
            remap[indices] = np.arange(len(indices))
            sub.locus = remap[sub.locus]
            # Match a separate condition_stats call's density strata exactly.
            sub.stratum = ST.density_bins(sub.density,12) if len(sub.locus) else np.empty(0,np.int64)
            out[frame] = (sub,bins,dict(appearances=len(sub.locus),molecules=len(np.unique(sub.molecule))))
        return out

    def engine(anchor,bg=(),n_perm=0,n_boot=0):
        checkpoint()
        return condition_stats(ms, [anchor], P=period, params=cp,
                               prefetched_backgrounds=(list(bg), {'sets': [(0, list(range(len(bg))))]}),
                               prefetched_appearances=appearances([anchor,*bg]),
                               metrics=('R_minus_null',), n_perm=n_perm,n_role=n_perm,n_boot=n_boot,
                               per_anchor=False, seed=p.seed, bootstrap_unit='molecule',
                               cancel=lambda: bool(checkpoint()))
    grid_values = {}
    if grid:
        grid_values = {'molecule':[],'anchor':[]}
        for pos in grid:
            res = engine((chrom,pos))
            for frame,fr in res['frames'].items():
                d = fr['metrics']['R_minus_null']['bins'][-1]['delta_anchor']
                if np.isfinite(d):
                    grid_values[frame].append(d)
    for row in rows:
        if row['kind'] != 'anchor':
            continue
        checkpoint()
        pos = row['pos']
        bg = matched.get(pos, [])
        result = engine((chrom,pos), bg, p.n_perm, p.n_boot)
        frames = {}
        for frame, fr in result['frames'].items():
            bins = fr['metrics']['R_minus_null']['bins']
            r = bins[-1]
            ci = r.get('delta_anchor_ci')
            vals = grid_values.get(frame, [])
            delta = r.get('delta_anchor')
            fit = _osc_fit_histogram(histograms[frame][position_index[pos]],edges,period)
            frames[frame] = dict(delta=delta, ci=ci, p=r.get('p_within_anchor'), period=period,
                                 ell=fit.get('ell'), ell_ci=fit.get('ell_ci'), ell_state='open (including bound)',
                                 background_method='matched' if r.get('n_sets') else 'locus-grid',
                                 matched_backgrounds=len(bg) if r.get('n_sets') else 0,
                                 locus_grid_percentile=float(100*np.mean(np.asarray(vals)<=delta)) if vals and np.isfinite(delta) else None,
                                 locus_grid_n=len(vals), locus_grid_step_bp=grid_step,
                                 delta_site=r.get('delta_site'), delta_site_ci=r.get('delta_site_ci'),
                                 ci_scope='physical molecules conditional on this locus',
                                 bins=[dict(bin=b['bin'], delta=b['delta_anchor']) for b in bins],
                                 n_molecules=r.get('n_anchor_molecules'))
        row['samples'][key]['phasing'] = frames
