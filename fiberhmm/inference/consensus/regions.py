"""BED-directed cross-locus transport; no inferred anchors or orientations."""
from copy import deepcopy
from pathlib import Path
from .artifacts import digest


def load_bed(path, *, pooled=False):
    rows=[]; names=set()
    import gzip
    opener=gzip.open if str(path).endswith('.gz') else open
    with opener(path,'rt') as handle:
        for n,line in enumerate(handle,1):
            if not line.strip() or line.startswith(('#','track ','browser ')): continue
            f=line.split()
            if len(f)<3: raise ValueError(f'BED line {n}: expected at least 3 fields')
            a,b=int(f[1]),int(f[2]); strand=f[5] if len(f)>=6 else '+'
            if a<0 or b<=a: raise ValueError(f'BED line {n}: invalid half-open coordinates')
            if pooled and (len(f)<6 or strand not in ('+','-')):
                raise ValueError(f'BED line {n}: CL-CR requires explicit BED6 + or - orientation')
            name=f[3] if len(f)>=4 and f[3]!='.' else f'{f[0]}:{a}-{b}:{strand}'
            if name in names: raise ValueError(f'Duplicate BED name: {name}')
            names.add(name); rows.append(dict(chrom=f[0],start=a,end=b,name=name,strand=strand))
    if not rows: raise ValueError('BED contains no windows')
    if pooled and len({r['end']-r['start'] for r in rows})!=1:
        raise ValueError('CL-CR windows must have equal widths: choose matching oriented windows in BED6')
    return rows


def orient_unit(unit, window, *, origin=0):
    """Local zero is BED start on +, BED end on -; base i mirrors to end-1-i."""
    u=deepcopy(unit); a,b=window['start'],window['end']; reverse=window['strand']=='-'
    point=lambda x:(b-1-x if reverse else x-a)+origin
    interval=lambda iv:[b-iv[1]+origin,b-iv[0]+origin] if reverse else [iv[0]-a+origin,iv[1]-a+origin]
    order=sorted((i for i,p in enumerate(u['positions']) if a<=p<b),key=lambda i:point(u['positions'][i]))
    for key in ('positions','hits','contexts','p_accessible','p_protected','m5c_observations'):
        if key in u: u[key]=[u[key][i] for i in order]
    u['positions']=[point(p) for p in u['positions']]
    for key in ('raw_tf_intervals','representative_raw_tf_intervals','raw_nuc_intervals','msp_intervals',
                'aligned_blocks','original_bam_tf_intervals','original_bam_nuc_intervals','original_bam_msp_intervals',
                'native_multi_interval_tf_intervals'):
        if key in u: u[key]=sorted(interval(iv) for iv in u[key])
    for call in u.get('native_multi_interval_calls',[]): call['interval']=interval(call['interval'])
    if 'native_multi_interval_calls' in u: u['native_multi_interval_calls'].sort(key=lambda c:c['interval'])
    recalled=u.get('upstream_nuc_tf_recall')
    if recalled:
        for key in ('nucleosomes','msps','original_tag_msps_absent_from_loaded_scaffold'):
            if key in recalled: recalled[key]=sorted(interval(iv) for iv in recalled[key])
        for key in ('calls','excluded_calls'):
            for call in recalled.get(key,[]):
                if 'interval' in call: call['interval']=interval(call['interval'])
            if key in recalled: recalled[key].sort(key=lambda c:c.get('interval',[]))
    u['reference_start'],u['reference_end']=interval([unit['reference_start'],unit['reference_end']])
    u['_region']=[origin,b-a+origin]
    u['genomic_provenance']=dict(window=window,original_unit_id=unit['unit_id'],
        reference_start=unit['reference_start'],reference_end=unit['reference_end'],strand=unit['strand'])
    u['unit_id']=unit['unit_id']+'::'+digest(window)[:16]
    if reverse: u['strand']={'CT':'GA','GA':'CT','FWD':'REV','REV':'FWD'}.get(u['strand'],u['strand'])
    return u


def pool_payloads(payloads, windows):
    """One locus view per molecule; union PCR aliases across locus views."""
    if not windows or len(payloads)!=len(windows): raise ValueError('Supply one payload per BED window')
    if any(w['strand'] not in ('+','-') or w['start']<0 or w['end']<=w['start'] for w in windows):
        raise ValueError('Invalid oriented BED window')
    if len({w['end']-w['start'] for w in windows})!=1: raise ValueError('CL-CR requires equal window widths')
    groups={}; views=[]; parent={}; manifests=[]
    def physical(name):
        parts=name.split('/')
        return '/'.join(parts[:2]) if len(parts)>=3 and parts[1].isdigit() else name
    def find(key):
        parent.setdefault(key,key)
        while parent[key]!=key:
            parent[key]=parent[parent[key]];key=parent[key]
        return key
    def union(a,b):
        a,b=find(a),find(b)
        if a!=b: parent[max(a,b)]=min(a,b)
    for payload,window in zip(payloads,windows):
        for source in payload['strata']:
            ds=source['dataset_id']
            if ds in groups and groups[ds]['chemistry']!=source['chemistry']: raise ValueError('Dataset chemistry changed across loci')
            if ds not in groups: groups[ds]=dict({k:deepcopy(v) for k,v in source.items() if k!='units'},units=[])
            manifests.append(dict(window=window['name'],dataset_id=ds,model=source.get('model_manifest'),load=source.get('load')))
            for original in source['units']:
                u=orient_unit(original,window)
                name=u.get('read_name') or ds+'::'+original['unit_id']
                key=physical(name);find(key)
                if u.get('physical_molecule_id'): union(key,str(u['physical_molecule_id']))
                for source_name in u.get('physical_source_names',[]):
                    union(key,physical(source_name))
                for member in u.get('source_members',[]):
                    if member.get('read_name'): union(key,physical(member['read_name']))
                views.append((ds,u,key,window['name'],original['unit_id']))
    chosen={};excluded=[];window_views={}
    for ds,u,key,window,original_id in views:
        key=find(key);u['physical_molecule_id']=key;u['fold_group_id']=key
        same_window=(key,window)
        if same_window in window_views:
            raise ValueError('Repeated physical molecule within one window; by-strand CCS or duplicate inputs require explicit joint-molecule preparation')
        window_views[same_window]=u['unit_id']
        rank=digest([key,window,ds,original_id]); previous=chosen.get(key)
        if previous is None or rank<previous[0]:
            if previous: excluded.append(dict(unit_id=previous[2]['unit_id'],window=previous[2]['genomic_provenance']['window']['name'],reason='repeated_locus_view'))
            chosen[key]=(rank,ds,u)
        else: excluded.append(dict(unit_id=u['unit_id'],window=window,reason='repeated_locus_view'))
    for _,ds,u in sorted(chosen.values(),key=lambda x:(x[1],x[2]['unit_id'])): groups[ds]['units'].append(u)
    for source in groups.values():
        source['load']=dict(cross_locus=True,input_window_count=len(windows),retained_views=len(source['units']))
        source['evidence_units']=dict(analyzed_molecules=len(source['units']),cross_locus_deduplicated=True,
            physical_duplex_independence_established=bool(source['units']) and all(u.get('physical_source_names') for u in source['units']),
            joint_duplex_units=sum(bool(u.get('physical_source_names')) for u in source['units']))
    width=windows[0]['end']-windows[0]['start']
    return dict(region=dict(chrom='oriented_BED',start=0,end=width),strata=list(groups.values()),
        pooling=dict(schema='fiberhmm.clcr.bed.v1',windows=windows,coordinate_system='0-based oriented BED window',
            ownership='one deterministic locus view per physical molecule, union of recorded amplification aliases',
            strand_frame='oriented_window',provenance_frame='original_genomic',
            excluded_repeated_views=sorted(excluded,key=lambda x:x['unit_id']),retained_molecules=len(chosen),window_preparation=manifests),
        input_files=payloads[0].get('input_files',[]))


def machine_compute_defaults():
    """Resource defaults for this machine: cores = usable CPUs - 2 (at most 16)
    and a matrix budget of a quarter of physical memory within [2, 32] GiB.
    Execution only: both are recorded in the manifest and never change a result.
    Memory limits imposed by a scheduler (cgroups, SLURM) are not detected; set
    compute.maximum_matrix_mb explicitly there."""
    import os
    try:available=len(os.sched_getaffinity(0))  # respects taskset/cgroup cpusets (Linux)
    except AttributeError:available=os.cpu_count() or 2
    cores=max(1,min(16,available-2))
    try:memory_mb=os.sysconf('SC_PAGE_SIZE')*os.sysconf('SC_PHYS_PAGES')//1024**2
    except (ValueError,OSError,AttributeError):memory_mb=8192
    budget=max(2048,min(32768,memory_mb//4))
    return dict(cores=cores,maximum_matrix_mb=budget-budget%16)


def automatic_parameters(parameters, strata):
    values=deepcopy(parameters or {})
    # Unset compute controls follow the machine; explicit values always win.
    # Decision stopping leaves every predictive decision identical (see
    # measurement_distribution.predictive_decision_stop_record).
    compute=values.setdefault('compute',{})
    for key,value in machine_compute_defaults().items():compute.setdefault(key,value)
    from .parameters import CROptions
    engine=values.get('cr',{}).get('engine',CROptions().engine)
    if engine not in ('staged_native_families','lattice_recaller'):
        raise ValueError('run_analysis requires the staged or lattice-recaller engine; use run_workflow for historical engine replay')
    if engine=='staged_native_families':compute.setdefault('predictive_stopping','decision')
    values.setdefault('cr',{})['engine']=engine
    # SR/XCR follow the data unless set explicitly; an explicit value always wins.
    # (The lattice recaller records mode CR whatever these say: its classes are
    # shared across channels and datasets by construction.)
    values.setdefault('sr',{}).setdefault('enabled',any(s['chemistry'] in ('ddda','dddb') for s in strata))
    values.setdefault('cross',{}).setdefault('enabled',len({s['dataset_id'] for s in strata})>1)
    return values
