"""Materialize frozen families using the existing MA/AQ/AN family convention.

Native layers are unchanged. Derived tf_consensus/tf_cross_consensus annotations
retain every compatible family, not only the display-primary assignment.
Lattice-recaller runs write tf_consensus (never tf_cross_consensus: the recaller
computes no XCR) and, optionally, a tf_recaller layer with the recaller's own
per-molecule class calls at every prevalence tier (see RECALLER_LAYER).
"""
from collections import defaultdict
from pathlib import Path
import array
import hashlib
import json
import os
import tempfile

import pysam

from ...io.bam_header import append_ma_types, append_pg_record
from ...io.ma_tags import parse_ma_tag, parse_an_tag, format_an_tag, llr_to_tq
from ..tf_family_ids import allocate_repeating_family_ids, TFFamilyInterval
from .artifacts import digest, write_json
from .native_presentation import Q0_SEMANTICS

CONTRACT = 'FIBERHMM-CONSENSUS-MA:v1:'
FAMILY = 'FIBERHMM-CONSENSUS-FAMILY:v1:'
RECALLER_MODE = 'lattice_recaller'
RECALLER_LAYER = 'tf_recaller'
OWNED_LAYERS = {'tf_consensus', 'tf_cross_consensus', RECALLER_LAYER}
QUALITY_NAMES = ['tq', 'fi', 'fq', 'op', 'sq', 'q0']
RECALLER_QUALITY_NAMES = ['tq', 'fi', 'tier', 'q0', 'lr', 'rr']
TIER_CODES = {'core': 1, 'edge': 2, 'loose': 3}
RECALLER_Q0_SEMANTICS = ("lattice_recaller_class_posterior_x255: the molecule's EM posterior for the labelled class on its "
                         "own channel (dataset x chemical strand), round(255*P), at least 1 for a labelled call; 0 = no class. "
                         "A mixture posterior, not a calibrated probability or the staged engine's class-evidence share")


def genomic_interval(interval, unit, region):
    provenance=unit.get('genomic_provenance')
    if not provenance: return region['chrom'],list(interval)
    w=provenance['window'];a,b=interval;origin=provenance.get('coordinate_origin',0);a-=origin;b-=origin
    return w['chrom'],[w['end']-b,w['end']-a] if w['strand']=='-' else [a+w['start'],b+w['start']]


def strand_quality(unit, chemistry, family):
    """AQ ``sq``: this molecule's own core protection ceiling for ``family``.

    The protection LLR the molecule would give if every lattice site in the
    family's core were unconverted, from its own sites and context emissions:
    what its chemical strand can say about this footprint at all. Encoded as
    1 + LLR*10 (saturated at 255) so a DAF molecule with no core site (1) is
    distinct from 0, which marks non-DAF chemistry or an unavailable lattice."""
    if chemistry not in ('ddda', 'dddb') or not family or 'p_accessible' not in unit:
        return 0
    import numpy as np
    positions = np.asarray(unit['positions'])
    core = (positions >= family['consensus_start']) & (positions < family['consensus_end'])
    pa = np.asarray(unit['p_accessible'], float)[core]; pp = np.asarray(unit['p_protected'], float)[core]
    ceiling = float(np.sum(np.log1p(-pp)-np.log1p(-pa)))
    return int(min(255, 1+max(0, round(10*ceiling))))


def _mapped_span(start, end, unit, region):
    """A class span (consensus coordinates) on the unit's genome, rounded outward to whole bases."""
    import math
    return genomic_interval([math.floor(start), math.ceil(end)], unit, region)


class ExportPlan:
    """Exact SAM-record/occurrence matches for every analysis, accumulated one analysis at a time.

    add() keeps only the planned annotation rows, the family catalog and a slim copy of each run's manifest, windows
    and input files, so a caller can drop each payload and result once it is added (the CLI streams BED windows this
    way). recaller_layer: also plan the tf_recaller layer for lattice-recaller runs."""

    def __init__(self, recaller_layer=False):
        self.recaller_layer = bool(recaller_layer)
        self.planned = defaultdict(lambda: defaultdict(list)); self.families = {}; self.file_stats = {}; self.chrom_cache = {}
        self.analyses = []; self.engines = set()

    def _chrom(self, path, chrom, interval):
        if (path, chrom) not in self.chrom_cache:
            from .bam import _resolve_bam_fetch_region
            self.chrom_cache[(path, chrom)] = _resolve_bam_fetch_region(path, chrom, *interval)[0]
        return self.chrom_cache[(path, chrom)]

    def _targets(self, member, paths):
        path = member.get('library_id')
        if path and str(Path(path).resolve()) in paths: return [str(Path(path).resolve())]
        if not path and len(paths) == 1: return paths
        raise ValueError('Cannot identify the source BAM for a family member')

    def add(self, result, payload):
        manifest = result['manifest']; recaller = result.get('cr_mode', manifest.get('cr_mode')) == RECALLER_MODE
        self.engines.add(RECALLER_MODE if recaller else 'other')
        if len(self.engines) > 1:
            raise ValueError('Export lattice-recaller and other-engine results to separate BAMs: their q0 semantics differ')
        run = manifest.get('family_identity_digest', manifest['input_digest']); region = payload['region']
        # The recaller computes no XCR (its classes are shared by construction): always tf_consensus.
        layer = 'tf_cross_consensus' if manifest['parameters']['cross']['enabled'] and not recaller else 'tf_consensus'
        units = {s['dataset_id']+'::'+u['unit_id']: u for s in payload['strata'] for u in s['units']}
        chemistry = {s['dataset_id']: s.get('chemistry') for s in payload['strata']}
        paths_by_dataset = defaultdict(list)
        for row in payload.get('input_files', []):
            path = str(Path(row['path']).resolve()); paths_by_dataset[row['dataset_id']].append(path)
            if path in self.file_stats and self.file_stats[path] != dict(row, path=path):
                # Dataset labels can differ, but a file cannot change during preparation.
                if any(self.file_stats[path].get(k) != row.get(k) for k in ('size', 'mtime_ns')):
                    raise ValueError('Source BAM changed across analyses: '+path)
            self.file_stats[path] = dict(row, path=path)
            self.planned[path]

        def family_entry(layer_name, token, family, actual_chrom, interval, unit, catalog, ds):
            key = (layer_name, token, actual_chrom)
            if recaller and family in catalog:
                # Class geometry, not the union of the calls that carry the label.
                fam = catalog[family]
                _, span = _mapped_span(fam['consensus_start'], fam['consensus_end'], unit, region)
                f = self.families.setdefault(key, dict(layer=layer_name, annotation_name=token, family_key=family, input_digest=run,
                    stage=result.get('final_stage'), chrom=actual_chrom, start=span[0], end=span[1], extent='class_consensus_span'))
                f['start'] = min(f['start'], span[0]); f['end'] = max(f['end'], span[1])
            else:
                f = self.families.setdefault(key, dict(layer=layer_name, annotation_name=token, family_key=family,
                    input_digest=run, stage=result.get('final_stage'), chrom=actual_chrom, start=interval[0], end=interval[1]))
                f['start'] = min(f['start'], interval[0]); f['end'] = max(f['end'], interval[1])
            if catalog.get(family, {}).get('strand_resolution'):
                f.setdefault('strand_resolution', {})[ds] = catalog[family]['strand_resolution']

        for ds, data in result['datasets'].items():
            catalog = {f['family']: f for f in data['cr'].get('catalog', [])}
            for record in data['cr']['records']:
                unit = units[record['unit_id']]
                members = unit.get('source_members') or [dict(
                    record_sha256=unit.get('provenance', {}).get('representative_record_sha256'),
                    alignment_occurrence=unit.get('provenance', {}).get('alignment_occurrence', 0))]
                rows = []
                for proposal in record['proposals']:
                    span = proposal['source_interval']
                    chrom, interval = genomic_interval(span, unit, region)
                    native = next((c for c in unit.get('native_multi_interval_calls', []) if tuple(c['interval']) == tuple(span)), {})
                    tq = llr_to_tq(native['llr']) if 'llr' in native else None
                    op = sum(span[0] <= p < span[1] for p in unit['positions'])
                    for family in proposal.get('compatible_families', []):
                        sq = strand_quality(unit, chemistry.get(ds), catalog.get(family))
                        # Each membership row carries that class's own support share (recaller: its class posterior).
                        q0 = int(proposal.get('member_q0', {}).get(family, proposal.get('q0', 0) if family == proposal.get('family') else 0))
                        token = ('fhxcr_' if layer == 'tf_cross_consensus' else 'fhcr_')+digest([run, family])[:24]
                        rows.append((layer, token, family, chrom, interval, dict(tq=tq, op=min(255, op), sq=sq, q0=q0)))
                if recaller and self.recaller_layer:
                    for call in record.get('recaller_calls', []):
                        if call.get('kind') != 'class' or not call.get('family') or call['family'] not in catalog:
                            continue                  # broader stretches have no class; they stay in result.json.gz
                        chrom, interval = genomic_interval(call['interval'], unit, region)
                        width = lambda r: min(255, max(0, int(r[1]) - int(r[0]))) if r else 0
                        ranges = call.get('edge_range') or [None, None]
                        native = next((c for c in unit.get('native_multi_interval_calls', [])
                                       if call.get('edge_source') == 'native' and list(c['interval']) == list(call['interval'])), {})
                        token = 'fhcr_'+digest([run, call['family']])[:24]
                        rows.append((RECALLER_LAYER, token, call['family'], chrom, interval, dict(
                            quals=[llr_to_tq(native['llr']) if 'llr' in native else 0, 0, TIER_CODES.get(call.get('tier'), 0),
                                   int(round(255*min(1., max(0., float(call.get('posterior') or 0.))))), width(ranges[0]), width(ranges[1])],
                            tolerant=True)))
                for member in members:
                    sha = member.get('record_sha256')
                    if not sha: raise ValueError('BAM materialization requires exact source SAM-record hashes; reload BAM evidence')
                    for path in self._targets(member, paths_by_dataset[ds]):
                        for layer_name, token, family, chrom, interval, values in rows:
                            actual_chrom = self._chrom(path, chrom, interval)
                            family_entry(layer_name, token, family, actual_chrom, interval, unit, catalog, ds)
                            self.planned[path][(sha, int(member.get('alignment_occurrence', 0)))].append(
                                dict(values, layer=layer_name, token=token, chrom=actual_chrom, interval=interval))
        slim_payload = {k: payload[k] for k in ('region', 'pooling', 'input_files') if k in payload}
        slim_result = dict(manifest=manifest, final_stage=result.get('final_stage'), cr_mode=result.get('cr_mode', manifest.get('cr_mode')))
        self.analyses.append((slim_result, slim_payload))
        return self

    def finish(self):
        for layer in {key[0] for key in self.families}:
            selected = {key: f for key, f in self.families.items() if key[0] == layer}
            slots = allocate_repeating_family_ids([TFFamilyInterval(digest(key), f['chrom'], f['start'], f['end']) for key, f in selected.items()])
            for key, f in selected.items(): f['fi'] = slots[digest(key)]
        for records in self.planned.values():
            for rows in records.values():
                for row in rows:
                    row['fi'] = self.families[(row['layer'], row['token'], row['chrom'])]['fi']
                    if 'quals' in row: row['quals'][1] = row['fi']
        return self.planned, list(self.families.values()), self.file_stats

    @property
    def recaller(self):
        return self.engines == {RECALLER_MODE}


def _as_plan(analyses, recaller_layer=False):
    if isinstance(analyses, ExportPlan): return analyses
    plan = ExportPlan(recaller_layer=recaller_layer)
    for result, payload in analyses: plan.add(result, payload)
    return plan


def assignment_plan(analyses):
    """Build exact SAM-record/occurrence matches; aliases keep representative provenance."""
    return _as_plan(analyses).finish()


def _project_to_molecule(read, interval, molecule_length, tolerant=False):
    """Molecular (start, length) of a reference interval on this read. tolerant: a recaller lattice call whose edge
    falls on unaligned reference maps to the aligned bases inside it (None if there are none)."""
    refs=read.get_reference_positions(full_length=True)
    if molecule_length!=len(refs):
        raise ValueError('MA/query length mismatch; cannot safely materialize family coordinates')
    a,b=interval;positions=[i for i,p in enumerate(refs) if p is not None and a<=p<b]
    if tolerant and not positions: return None
    if not tolerant and (not positions or refs[positions[0]]!=a or refs[positions[-1]]!=b-1):
        raise ValueError('Family span does not map completely to this alignment: '+str(interval))
    if read.has_tag('MA'):
        matches=[]
        for name,_,_,spans in parse_ma_tag(read.get_tag('MA'))['raw_types']:
            if name!='tf': continue
            for start,length in spans:
                lo=molecule_length-start-length if read.is_reverse else start
                mapped=[p for p in refs[lo:lo+length] if p is not None]
                if mapped and (min(mapped),max(mapped)+1)==(a,b): matches.append((start,length))
        if len(set(matches))==1: return matches[0]
        if len(set(matches))>1: raise ValueError('Ambiguous native molecular projection')
    lo,hi=positions[0],positions[-1]+1
    return (molecule_length-hi,hi-lo) if read.is_reverse else (lo,hi-lo)


def _append_annotations(read, rows):
    old=read.get_tag('MA') if read.has_tag('MA') else str(read.query_length)
    parsed=parse_ma_tag(old);types=parsed['raw_types']
    if any(name in {r['layer'] for r in rows} for name,_,_,_ in types):
        raise ValueError('Source already contains the target family layer; use the original source BAM')
    aq=list(read.get_tag('AQ')) if read.has_tag('AQ') else []
    expected=sum(len(q)*len(iv) for _,_,q,iv in types)
    if len(aq)!=expected: raise ValueError('Source MA/AQ length mismatch')
    count=sum(len(iv) for _,_,_,iv in types)
    names=parse_an_tag(read.get_tag('AN')) if read.has_tag('AN') else ['']*count
    if len(names)!=count: raise ValueError('Source MA/AN length mismatch')
    # Preserve an existing native TQ when no replay score was available.
    native_tq={};cursor=0
    for name,_,q,intervals in types:
        for molecular_start,length in intervals:
            if name=='tf' and q:
                native_tq[(molecular_start,length)]=aq[cursor]
            cursor+=len(q)
    groups=defaultdict(dict)
    for row in rows:
        if read.reference_name!=row['chrom']:
            # Preparation permits exact chr/no-chr aliases; compare against BAM here.
            # Callers normalize chrom to the source header before reaching here.
            raise ValueError('Family chromosome does not match source alignment')
        interval=_project_to_molecule(read,row['interval'],parsed['read_length'],tolerant=row.get('tolerant',False))
        if interval is None: continue
        key=(interval,row['token'])
        groups[row['layer']][key]=row
    suffix=[]
    for layer,values in sorted(groups.items()):
        tokens=[];width=None
        for (interval,token),row in sorted(values.items()):
            start,length=interval;tokens.append(f'{start+1}-{length}')
            quals=row['quals'] if 'quals' in row else [native_tq.get(interval,0) if row['tq'] is None else row['tq'],row['fi'],0,row['op'],
                       row.get('sq',0),row.get('q0',0)]
            width=len(quals);aq.extend(quals)
            names.append(token)
        suffix.append(layer+'.'+'Q'*width+':'+','.join(tokens))
    read.set_tag('MA',old+';'+ ';'.join(suffix),value_type='Z')
    read.set_tag('AQ',array.array('B',aq))
    read.set_tag('AN',format_an_tag(names),value_type='Z')
    return sum(len(values) for values in groups.values())


def _clear_owned_layers(read, layers):
    if not layers or not read.has_tag('MA'): return
    old=read.get_tag('MA');types=parse_ma_tag(old)['raw_types']
    aq=list(read.get_tag('AQ')) if read.has_tag('AQ') else []
    count=sum(len(iv) for _,_,_,iv in types)
    names=parse_an_tag(read.get_tag('AN')) if read.has_tag('AN') else ['']*count
    if len(aq)!=sum(len(q)*len(iv) for _,_,q,iv in types) or len(names)!=count:
        raise ValueError('Source MA/AQ/AN length mismatch')
    sections=old.split(';');kept=[sections[0]];qualities=[];annotations=[];qi=ni=0
    for section,(name,_,q,intervals) in zip(sections[1:],types):
        n=len(intervals);nq=n*len(q)
        if name not in layers:
            kept.append(section);qualities.extend(aq[qi:qi+nq]);annotations.extend(names[ni:ni+n])
        qi+=nq;ni+=n
    read.set_tag('MA',';'.join(kept),value_type='Z')
    read.set_tag('AQ',array.array('B',qualities) if qualities else None)
    read.set_tag('AN',format_an_tag(annotations) if any(annotations) else None)


def _source_windows(analyses, source):
    from .bam import _resolve_bam_fetch_region
    windows=defaultdict(list)
    for _,payload in analyses:
        if source not in {str(Path(f['path']).resolve()) for f in payload.get('input_files',[])}:
            continue
        for w in payload.get('pooling',{}).get('windows') or [payload['region']]:
            chrom,start,end,_=_resolve_bam_fetch_region(source,w['chrom'],w['start'],w['end'])
            if end>start: windows[chrom].append((start,end))
    merged={}
    for chrom,spans in windows.items():
        merged[chrom]=[]
        for start,end in sorted(spans):
            if merged[chrom] and start<=merged[chrom][-1][1]:
                merged[chrom][-1][1]=max(end,merged[chrom][-1][1])
            else: merged[chrom].append([start,end])
    return merged


def _export_reads(bam, windows, scope):
    if scope=='full':
        yield from bam.fetch(until_eof=True)
        return
    # Fetch union windows once. Long alignments spanning disjoint windows belong
    # to their first overlapping window; true duplicate SAM records are retained.
    for chrom in bam.references:
        spans=windows.get(chrom,[])
        for i,(start,end) in enumerate(spans):
            for read in bam.fetch(chrom,start,end):
                if any(read.reference_start<b and read.reference_end>a for a,b in spans[:i]):
                    continue
                yield read


def _contract(plan, layers, scope, windows):
    """The FIBERHMM-CONSENSUS-MA header contract: per-layer quality semantics for this export."""
    recaller=plan.recaller;analyses=plan.analyses
    value=dict(layers=layers,quality_names=QUALITY_NAMES,
        tq='native_LLR_times_10_saturated_255_or_zero_if_unavailable',
        fi='local_repeating_uint8_slot; AN_is_authoritative_family_identity',
        fq='zero_unavailable_no_calibrated_assignment_probability',
        op='representative_native_opportunities_saturated_255',
        sq='DAF_molecule_core_protection_ceiling: 1+LLR_times_10_saturated_255 (1 = no core site); 0 = not DAF or unavailable',
        q0=(RECALLER_Q0_SEMANTICS if recaller else Q0_SEMANTICS+'; per membership row, that class\'s share'),
        strand_resolution=('per-family catalog entry, per DAF dataset (lattice_recaller): trusted_strand (CT/GA/both/none; a strand is trusted when its expected evidence per molecule over the class reaches recaller.resolution_nats), supported_strands, per-strand resolution_nats; use it to choose which chemical strand to quantify'
            if recaller else 'per-family catalog entry, per dataset: trusted_strand (CT/GA/both/none; a strand is limited when its core ceiling is below the native floor), core_resolution, per-strand median core ceilings and native floor; use it to choose which chemical strand to quantify'),
        memberships='all_compatible_families_nonexclusive',
        export_scope=scope,export_windows=windows,
        coordinates='original_source_call_in_molecular_frame',
        projection='PCR_aliases_inherit_representative_family_membership',
        runs=[dict(input_digest=r['manifest']['input_digest'],parameters=r['manifest']['parameters'],
            mode=r['manifest'].get('display_mode'),stage=r.get('final_stage'),cr_mode=r.get('cr_mode'),
            family_identity_digest=r['manifest'].get('family_identity_digest'),
            numerical_policy=r['manifest'].get('numerical_policy'),
            implementation_sha256=r['manifest'].get('implementation_sha256')) for r,_ in analyses])
    if recaller:
        value.update(engine=RECALLER_MODE,
            family_extent='FAMILY start/end = the class consensus span (not the union of labelled calls)',
            labels='tf_consensus: native calls of member molecules (per-molecule BF label) that fit the class: width <= recaller.call_max_bp and lattice-censored edges reaching the class edge boxes',
            recaller_calls=('tf_recaller layer' if RECALLER_LAYER in layers else 'not exported; see result.json.gz recaller_calls'))
        if RECALLER_LAYER in layers:
            value['layer_quality_names']={RECALLER_LAYER:RECALLER_QUALITY_NAMES}
            value[RECALLER_LAYER]=dict(
                intervals="the recaller's own per-molecule class call (result.json.gz recaller_calls kind=class): the matching native call's edges when edge_source=native, else the lattice edges; clipped to aligned bases",
                tq='native_LLR_times_10 when the edges are a native call (edge_source=native), 0 = lattice edges',
                fi='local_repeating_uint8_slot; AN_is_authoritative_family_identity (same fhcr_ token as tf_consensus)',
                tier='1 = core (class member), 2 = edge (protected run lined up with a class edge), 3 = loose (clean class core under any protection)',
                q0="class posterior x255 (edge/loose calls are non-members, so usually low); 0 allowed",
                lr='left edge range width (bp, saturated 255) from the lattice; 0 = exact edge',
                rr='right edge range width (bp, saturated 255) from the lattice; 0 = exact edge',
                excluded='broader-protection stretches (no class) stay in result.json.gz and broader.tsv.gz')
    return value


def _export_source_bams(analyses, output_dir, scope):
    """Write indexed derivative BAMs atomically; source BAMs are never modified."""
    plan=_as_plan(analyses)
    planned,families,file_stats=plan.finish();analyses=plan.analyses
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True);outputs=[]
    for index,(source,records) in enumerate(sorted(planned.items()),1):
        source=Path(source)
        stat=source.stat();saved=file_stats[str(source)]
        if stat.st_size!=saved['size'] or stat.st_mtime_ns!=saved['mtime_ns']:
            raise ValueError('Source BAM changed since evidence preparation: '+str(source))
        target=out/f'{index:03d}_{source.stem}.families.bam'
        if target.exists() or Path(str(target)+'.csi').exists(): raise ValueError('BAM output already exists: '+str(target))
        fd,name=tempfile.mkstemp(prefix='.families-',suffix='.bam',dir=out);os.close(fd);temporary=Path(name)
        seen=defaultdict(int);matched=set();annotations=0;written=0
        windows=_source_windows(analyses,str(source))
        try:
            with pysam.AlignmentFile(str(source),'rb') as bam:
                layers=sorted({f['layer'] for f in families})
                header=append_ma_types(bam.header,layers)
                hd=header.to_dict();old_comments=list(hd.get('CO',[]))
                owned_layers=set()
                for comment in old_comments:
                    if comment.startswith(CONTRACT):
                        owned_layers.update(json.loads(comment[len(CONTRACT):]).get('layers',[]))
                owned_layers.intersection_update(OWNED_LAYERS)
                comments=[c for c in old_comments if not c.startswith((CONTRACT,FAMILY))]
                comments.append(CONTRACT+json.dumps(_contract(plan,layers,scope,windows),sort_keys=True))
                comments.extend(FAMILY+json.dumps(f,sort_keys=True) for f in families)
                hd['CO']=comments
                source_rg='fhconsensus_'+digest(str(source))[:16]
                owned_rg={r['ID'] for r in hd.get('RG',[]) if any(c.startswith(CONTRACT) for c in old_comments)
                    and r['ID'].startswith('fhconsensus_') and r.get('DS','').startswith('Original source BAM: ')}
                retained_rg=[dict(r) for r in hd.get('RG',[]) if r['ID'] not in owned_rg]
                for rg in retained_rg:
                    original=rg.get('DS','').split('; FiberHMM source BAM: ')[0]
                    rg['DS']=original+'; FiberHMM source BAM: '+str(source)
                existing_rg={r['ID'] for r in retained_rg}
                while source_rg in existing_rg: source_rg+='_'
                hd['RG']=[*retained_rg,dict(ID=source_rg,DS='Original source BAM: '+str(source))]
                header=append_pg_record(pysam.AlignmentHeader.from_dict(hd),dict(PN='fiberhmm-consensus',DS='Frozen staged family annotations in MA/AQ/AN'))
                with pysam.AlignmentFile(str(temporary),'wb',header=header) as dest:
                    for read in _export_reads(bam,windows,scope):
                        sha=hashlib.sha256(read.to_string().encode()).hexdigest()
                        key=(sha,seen[sha]);seen[sha]+=1
                        rows=records.get(key)
                        _clear_owned_layers(read,owned_layers)
                        if read.has_tag('RG') and read.get_tag('RG') in owned_rg: read.set_tag('RG',None)
                        if rows:
                            normalized=[]
                            for row in rows:
                                chrom=row['chrom']
                                if chrom not in bam.references:
                                    from .bam import _resolve_bam_fetch_region
                                    chrom,_,_,_=_resolve_bam_fetch_region(str(source),chrom,*row['interval'])
                                normalized.append(dict(row,chrom=chrom))
                            annotations+=_append_annotations(read,normalized);matched.add(key)
                        if not read.has_tag('RG'): read.set_tag('RG',source_rg,value_type='Z')
                        dest.write(read);written+=1
            if set(records)-matched: raise ValueError('Some frozen family assignments did not match exact source alignments')
            current=source.stat()
            if (current.st_size,current.st_mtime_ns)!=(stat.st_size,stat.st_mtime_ns): raise ValueError('Source changed while exporting')
            pysam.quickcheck(str(temporary));pysam.index('-c',str(temporary))
            os.replace(temporary,target);os.replace(str(temporary)+'.csi',str(target)+'.csi')
            outputs.append(dict(input=str(source),bam=str(target.resolve()),index=str(target.resolve())+'.csi',annotations=annotations,matched_alignments=len(matched),written_alignments=written,export_scope=scope,windows=windows))
        finally:
            temporary.unlink(missing_ok=True);Path(str(temporary)+'.csi').unlink(missing_ok=True)
    write_json(out/'bam_exports.json',dict(outputs=outputs,families=families))
    return outputs




def export_bams(analyses, output_dir, *, grouping='datasets', dataset_groups=None, progress=None, scope='regions', recaller_layer=False):
    """Group outputs by the current dataset view, or by original input file.

    analyses: [(result, payload)] or an ExportPlan built incrementally (then recaller_layer is the plan's own).
    recaller_layer: for lattice-recaller results, also write the tf_recaller layer."""
    if scope not in ('regions','full'): raise ValueError('BAM scope must be regions or full')
    if grouping not in ('datasets','files'): raise ValueError('BAM grouping must be datasets or files')
    out=Path(output_dir).expanduser().resolve()
    if out.exists() and any(out.iterdir()): raise ValueError('Choose a new or empty BAM output folder')
    analyses=_as_plan(analyses,recaller_layer=recaller_layer)
    inputs={str(Path(f['path']).resolve()) for _,p in analyses.analyses for f in p.get('input_files',[])}
    if not inputs: raise ValueError('No source BAM paths in this result; reload and run from BAM input')
    if dataset_groups is None:
        grouped=defaultdict(set)
        for _,p in analyses.analyses:
            for f in p.get('input_files',[]): grouped[f['dataset_id']].add(str(Path(f['path']).resolve()))
        dataset_groups=[dict(dataset_id=k,paths=sorted(v)) for k,v in sorted(grouped.items())]
    if grouping=='files': dataset_groups=[dict(dataset_id=Path(p).stem,paths=[p]) for p in sorted(inputs)]
    groups=[];owned=set()
    for g in dataset_groups:
        paths=[str(Path(p).resolve()) for p in g['paths'] if str(Path(p).resolve()) in inputs]
        if not paths: continue
        if owned.intersection(paths): raise ValueError('A source BAM occurs in multiple export groups')
        owned.update(paths);groups.append(dict(dataset_id=g['dataset_id'],paths=paths))
    if owned!=inputs: raise ValueError('Current dataset grouping does not include every source BAM from this run')
    out.mkdir(parents=True,exist_ok=True)
    outputs=[]
    with tempfile.TemporaryDirectory(prefix='.family-export-',dir=out) as folder:
        staging=Path(folder)
        if progress: progress('bam_export','Writing MA family annotations to source BAM copies')
        sources=_export_source_bams(analyses,staging/'sources',scope)
        by_path={row['input']:row for row in sources}
        ready=[]
        for i,g in enumerate(groups,1):
            if progress: progress('bam_export',f"Preparing dataset {i}/{len(groups)}: {g['dataset_id']}")
            import re
            stem=re.sub(r'[^A-Za-z0-9_.-]+','_',g['dataset_id'])[:100] or 'dataset'
            name=f'{i:03d}_{stem}.families.bam';target=staging/name
            rows=[by_path[p] for p in g['paths']]
            if len(rows)==1:
                os.replace(rows[0]['bam'],target);os.replace(rows[0]['index'],str(target)+'.csi')
            else:
                lengths={}
                for row in rows:
                    with pysam.AlignmentFile(row['bam'],'rb') as bam:
                        for chrom,length in zip(bam.references,bam.lengths):
                            if chrom in lengths and lengths[chrom]!=length:
                                raise ValueError('Merged BAMs have conflicting reference lengths: '+chrom)
                            lengths[chrom]=length
                merged=staging/f'merged_{i}.bam'
                # samtools preserves/relabels colliding RG/PG IDs and their read tags.
                pysam.merge('-o',str(merged),*[row['bam'] for row in rows])
                with pysam.AlignmentFile(str(merged),'rb') as combined:
                    hd=combined.header.to_dict()
                hd['CO']=list(dict.fromkeys(hd.get('CO',[])))
                header_file=staging/f'header_{i}.sam'
                header_file.write_text(str(pysam.AlignmentHeader.from_dict(hd)))
                reheadered=staging/f'reheadered_{i}.bam';reheadered.touch()
                pysam.reheader('-P',str(header_file),str(merged),save_stdout=str(reheadered))
                pysam.sort('-o',str(target),str(reheadered));pysam.index('-c',str(target))
                merged.unlink()
            pysam.quickcheck(str(target))
            outputs.append(dict(dataset_id=g['dataset_id'],inputs=g['paths'],bam=str(out/name),
                index=str(out/name)+'.csi',annotations=sum(r['annotations'] for r in rows),
                matched_alignments=sum(r['matched_alignments'] for r in rows),
                written_alignments=sum(r['written_alignments'] for r in rows),export_scope=scope,
                source_windows={r['input']:r['windows'] for r in rows}))
            ready.append((target,out/name))
        for src,dest in ready:
            os.replace(src,dest);os.replace(str(src)+'.csi',str(dest)+'.csi')
        receipt=json.loads((staging/'sources'/'bam_exports.json').read_text())
        write_json(out/'bam_exports.json',dict(receipt,grouping=grouping,outputs=outputs))
    return outputs


def read_family_catalog(header):
    """Recover family identity and score semantics without external tables."""
    values=header.to_dict() if hasattr(header,'to_dict') else header
    contracts=[];families={}
    for comment in values.get('CO',[]):
        if comment.startswith(CONTRACT):
            value=json.loads(comment[len(CONTRACT):])
            if value not in contracts: contracts.append(value)
        elif comment.startswith(FAMILY):
            value=json.loads(comment[len(FAMILY):])
            key=(value['layer'],value['annotation_name'],value['chrom'])
            if key in families:
                previous=families[key]
                semantic=lambda row:{k:v for k,v in row.items() if k not in ('start','end')}
                if semantic(previous)!=semantic(value): raise ValueError('Conflicting MA family catalog entries')
                previous['start']=min(previous['start'],value['start'])
                previous['end']=max(previous['end'],value['end'])
            else:
                families[key]=value
    return dict(contracts=contracts,families=list(families.values()))
