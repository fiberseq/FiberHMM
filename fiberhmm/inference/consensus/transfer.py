"""Portable frozen-family transfer using the same native paper scorers."""
from copy import deepcopy
from pathlib import Path
import hashlib
import math
import numpy as np
from .artifacts import digest,read_json,write_json
from .harmonized_families.reference.cross_source_family_consolidation import freeze_children,transfer_child
from .harmonized_families.reference.bounded_parent_prototype import prepare_events,score_event
from .harmonized_families.reference.overlapping_family_update import reindex_fit
from .harmonized_families.reference.run_native_locus_map import prepare_input
from .measurement_grouping import _calls
from .cross_preparation import NativeReadCache,TransferGeometryCache

SCHEMA='fiberhmm.frozen_families.v1'

def _encode(x):
    if isinstance(x,np.ndarray): return {'__array__':x.dtype.str,'values':_encode(x.tolist())}
    if isinstance(x,np.generic): return _encode(x.item())
    if isinstance(x,float) and not math.isfinite(x):
        if math.isnan(x): raise ValueError('NaN in frozen model')
        return {'__float__':'-inf' if x<0 else 'inf'}
    if isinstance(x,dict): return {str(k):_encode(v) for k,v in x.items()}
    if isinstance(x,(list,tuple,set)): return [_encode(v) for v in (sorted(x) if isinstance(x,set) else x)]
    return x

def _decode(x):
    if isinstance(x,dict):
        if set(x)=={'__array__','values'}:
            dtype=np.dtype(x['__array__'])
            if dtype.kind not in 'biuf': raise ValueError('Unsupported model array type')
            return np.asarray(_decode(x['values']),dtype=dtype)
        if set(x)=={'__float__'}:
            if x['__float__'] not in ('-inf','inf'): raise ValueError('Invalid model float')
            return float(x['__float__'])
        return {k:_decode(v) for k,v in x.items()}
    if isinstance(x,list):return [_decode(v) for v in x]
    return x

def molecule_keys(unit):
    keys=set()
    for field in ('physical_molecule','physical_molecule_id','fold_group_id'):
        if unit.get(field): keys.add(str(unit[field]))
    names=[unit.get('read_name','')]+list(unit.get('physical_source_names',[]))+[m.get('read_name','') for m in unit.get('source_members',[])]
    for name in names:
        if name:
            keys.add(name);parts=name.split('/')
            if len(parts)>=3 and parts[1].isdigit():keys.add('/'.join(parts[:2]))
    return {hashlib.sha256(k.encode()).hexdigest() for k in keys}

def save_bundle(path, models, region, training_units, provenance):
    if region[1]<=region[0]: raise ValueError('Invalid frozen window extent')
    if not models: raise ValueError('No fitted active families to export')
    body=_encode(dict(schema=SCHEMA,region=list(region),models=models,
        training_molecule_hashes=sorted(set().union(*(molecule_keys(u) for u in training_units))),
        provenance=provenance,score_semantics='predictive compatibility, not occupancy probability'))
    write_json(path,dict(body,content_sha256=digest(body)))
    return load_bundle(path)

def load_bundle(path):
    body=read_json(path);expected=body.pop('content_sha256',None)
    if body.get('schema')!=SCHEMA or expected!=digest(body): raise ValueError('Invalid frozen-family schema or content digest')
    value=_decode(body)
    if value['region'][1]<=value['region'][0]:raise ValueError('Invalid frozen window coordinates')
    if not value['models'] or len({m['id'] for m in value['models']})!=len(value['models']):raise ValueError('Nonempty unique family IDs required')
    for m in value['models']:
        if m['kind'] not in ('native_child','bounded_parent'):raise ValueError('Unsupported frozen family kind')
        if m['kind']=='bounded_parent':
            for key in ('columns','log_density','log_mass'):
                m['fit'][key]=np.asarray(m['fit'][key],dtype=int if key=='columns' else float)
    value['content_sha256']=expected
    return value

def export_run(run_dir, output):
    """Freeze displayed, converged final models; preserve aliases and sources."""
    root=Path(run_dir);manifest=read_json(root/'manifest.json');evidence=read_json(root/'evidence.json.gz')
    if not evidence.get('pooling'):raise ValueError('Transfer models require an oriented CL-CR run (--pool-loci)')
    parts={}
    for path in sorted((root/'sources').glob('*/source.json.gz')):
        part=read_json(path)
        part['native_cell_provenance']['path']=str(path.with_name('native.json.gz'))
        parts[part['channel']]=part
    mode=manifest['mode'];scopes=({c:{c:p} for c,p in parts.items()} if mode=='CR' else
        {ds:{c:p for c,p in parts.items() if p['dataset_id']==ds} for ds in {p['dataset_id'] for p in parts.values()}} if mode=='SR' else {'XCR':parts})
    models=[]
    for scope,selected in sorted(scopes.items()):
        folder=root/'scopes'/hashlib.sha256(scope.encode()).hexdigest()[:16]
        ann=read_json(folder/(manifest['last_stage']+'.json.gz'));allh={h['id']:h for h in ann['hypotheses']}
        children={c:freeze_children(p) for c,p in selected.items()}
        for h in ann['hypotheses']:
            if not h['display'] or h.get('fit_warning'):continue
            fid=h['id'];seen=set()
            while allh[fid].get('reused_from'):
                if fid in seen:raise ValueError('Cyclic fit alias')
                seen.add(fid);fid=allh[fid]['reused_from']
            base=dict(id=scope+'|'+h['id'],reference_interval=h['reference_interval'],geometry=h.get('geometry'),kind=h['kind'])
            if h['kind']=='native_child':
                matches=[(c,k) for c,fs in children.items() for k in fs if c+'::'+k==fid]
                if len(matches)!=1:raise ValueError('Cannot resolve native family '+fid)
                c,k=matches[0];base['frozen']=children[c][k]
            else:
                parent=read_json(folder/'parents'/(fid.split(':')[1]+'.json.gz'))
                if parent['full_model'].get('diagnostics',{}).get('converged') is False:continue
                base.update(kind='bounded_parent',fit=parent['full_model'],region=parent['records'][0]['scoring_region'])
            models.append(base)
    units=[u for s in evidence['strata'] for u in s['units']]
    return save_bundle(output,models,[evidence['region']['start'],evidence['region']['end']],units,
        dict(input_digest=manifest['input_digest'],implementation_sha256=manifest['implementation_sha256'],parameters=manifest['parameters']))

def score_payload(bundle,payload,*,progress=None,maximum_bytes=256*1024**2):
    """No fitting or family nomination; original target calls retain their span."""
    if [payload['region']['start'],payload['region']['end']]!=bundle['region']:raise ValueError('Target window width/frame differs from frozen model')
    training=set(bundle['training_molecule_hashes']);rows=[];denominators=[];excluded=[];seen=set()
    cache=NativeReadCache(32*1024**2)
    geometry={m['id']:TransferGeometryCache(m['frozen'],m['frozen']['grid'],2,8*1024**2) for m in bundle['models'] if m['kind']=='native_child'}
    for source in payload['strata']:
        raw=[]
        for original in source['units']:
            identity=molecule_keys(original)
            if identity&training:
                excluded.append(dict(dataset=source['dataset_id'],unit_id=original['unit_id'],reason='training_molecule'));continue
            if identity&seen:raise ValueError('Repeated physical molecule in target window; provide jointly prepared data')
            seen.update(identity)
            u=deepcopy(original)
            if 'native_multi_interval_tf_intervals' in u:
                u['raw_tf_intervals']=deepcopy(u['native_multi_interval_tf_intervals']);u['representative_raw_tf_intervals']=deepcopy(u['raw_tf_intervals'])
            raw.append(u)
        if not raw:
            denominators.extend(dict(dataset=source['dataset_id'],family=m['id'],eligible_units=[],count=0) for m in bundle['models'])
            continue
        prepared,ledger,_=prepare_input(dict(source,units=raw),bundle['region'])
        units={u['unit_id']:u for u in prepared['units']}
        originals={source['dataset_id']+'::'+u['unit_id']:u for u in raw}
        calls={(c['unit_id'],c['start'],c['end']):c for c in _calls(prepared)}
        for m in bundle['models']:
            mean=(m.get('geometry') or {}).get('mean') or m['reference_interval'];a,b=math.floor(mean[0]),math.ceil(mean[1])
            eligible=[uid for uid,u in units.items() if any(x<=a and b<=y for x,y in u['eligible_intervals'])]
            denominators.append(dict(dataset=source['dataset_id'],family=m['id'],eligible_units=eligible,count=len(eligible)))
        for i,row in enumerate(ledger):
            uid=row['unit_id'];a,b=row['interval'];scores=[]
            if row['inference_eligible']:
                call=calls[uid,a,b];u=units[uid]
                for m in bundle['models']:
                    if not(a<m['reference_interval'][1] and b>m['reference_interval'][0]):continue
                    if m['kind']=='native_child':s=transfer_child(m['frozen'],u,call,read_cache=cache,cache=geometry[m['id']])
                    else:
                        region=[min(m['region'][0],a),max(m['region'][1],b)]
                        data=prepare_events(units,[call],region,maximum_bytes=maximum_bytes)
                        seed=int.from_bytes(hashlib.sha256(f"7123|{u['fold_group_id']}".encode()).digest()[:4],'little')
                        s=score_event(data,0,reindex_fit(m['fit'],m['region'],region),4095,seed)
                    scores.append(dict(family=m['id'],**s))
            from .bam_export import genomic_interval
            chrom,span=genomic_interval([a,b],originals[uid],payload['region'])
            rows.append(dict(dataset=source['dataset_id'],unit_id=uid,interval=[a,b],chrom=chrom,genomic_interval=span,
                eligible=row['inference_eligible'],scores=scores,compatible=[s['family'] for s in scores if s.get('compatible') is True]))
            if progress:progress('transfer',f"{source['dataset_id']}: {i+1}/{len(ledger)} original calls")
    return dict(schema='fiberhmm.frozen_transfer.v1',model_sha256=bundle['content_sha256'],rows=rows,denominators=denominators,
        exclusions=excluded,model_refitted=False,parameters=dict(replicates=4095,reference_percent=99.9),
        denominator_semantics='aligned MSP covers fitted mean, excluding nucleosomes; opportunity assessment reported per call')
