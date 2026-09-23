"""Shared, unsampled BAM preparation for CLI and FiberBrowser consensus."""
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
import hashlib
HARMONIZATION_MODE='call_harmonization'
STAGED_MODE='staged_native_families'


def loader_runtime():
    from fiberhmm.core.model_io import load_model_with_metadata
    from fiberhmm.inference.strand_rescue import PRESETS, collapse_amplified_reads, load_region_evidence, resolve_resource_path
    from fiberhmm.inference.tf_recaller import build_llr_tables
    return dict(PRESETS=PRESETS, collapse=collapse_amplified_reads, load_bam=load_region_evidence,
                resolve_resource=resolve_resource_path, load_model=load_model_with_metadata, build_llr=build_llr_tables)


def _flatten_paths(value):
    return [str(value)] if isinstance(value,(str,Path)) else [str(p) for p in value]


def _resolve_bam_fetch_region(path, chrom, start, end):
    import pysam
    with pysam.AlignmentFile(path,'rb') as bam:
        lengths=dict(zip(bam.references,bam.lengths))
    variants=[chrom, chrom[3:] if chrom.startswith('chr') else 'chr'+chrom]
    base=chrom[3:] if chrom.startswith('chr') else chrom
    romans=['I','II','III','IV','V','VI','VII','VIII','IX','X','XI','XII','XIII','XIV','XV','XVI','XVII','XVIII','XIX','XX']
    if base.isdigit() and 1<=int(base)<=20:
        roman=romans[int(base)-1];variants += [roman,'chr'+roman]
    if base.lower() in ('m','mt','mito','mitochondria','mitochondrion'): variants += ['M','MT','chrM','chrMT','chrMito']
    actual=next((c for c in variants if c in lengths),None)
    if actual is None: raise ValueError(f'{path}: chromosome {chrom!r} is absent')
    lo,hi=max(0,start),min(end,lengths[actual])
    if hi<=lo: raise ValueError('Window does not intersect the BAM chromosome')
    return actual,lo,hi,lengths[actual]


def _chemistry_runtime(runtime, chemistry, **unused):
    preset=runtime['PRESETS'].get(chemistry)
    if chemistry=='hia5-pacbio': preset=dict(model='fiberhmm/models/hia5_pacbio.json',strand_mode='alignment',prob_threshold=125)
    if preset is None: raise ValueError('Unsupported chemistry: '+chemistry)
    model,context,mode=runtime['load_model'](runtime['resolve_resource'](preset['model']))
    hit,miss=runtime['build_llr'](model)
    return preset,model,context,mode,hit,miss


def _load_dataset_evidence(state, dsid, chemistry, *, chrom, evidence_start, evidence_end,
                           runtime, minimum_mapq, legacy_annotation_frame=None, **unused):
    ds=state.get_dataset(dsid)
    preset,model,context,mode,hit,miss=_chemistry_runtime(runtime,chemistry)
    reads=[]; files=[]
    for i,path in enumerate(ds.paths):
        actual,lo,hi,length=_resolve_bam_fetch_region(path,chrom,evidence_start,evidence_end)
        diagnostics=dict(requested_chrom=chrom,bam_chrom=actual,fetch_start=lo,fetch_end=hi)
        reads.extend(runtime['load_bam'](path,actual,lo,hi,strand_mode=preset['strand_mode'],mode=mode,
            context_size=context,prob_threshold=preset.get('prob_threshold'),llr_hit=hit,llr_miss=miss,
            min_mapq=minimum_mapq,tf_layer='tf',nuc_layer='nuc',max_reads=0,input_index=i,ma_annotation_frame='auto',
            load_diagnostics=diagnostics,**({'legacy_annotation_frame':legacy_annotation_frame} if legacy_annotation_frame is not None else {})))
        files.append(diagnostics)
    return reads,model,preset,dict(source_type='bam',evidence_reads=len(reads),files=files)


def load_bam_payload(datasets, region, options=None, progress=None):
    """datasets: [{dataset_id, paths, chemistry?}]; metadata conflicts are errors."""
    from fiberhmm.cli.targeted_families import _resolve_chemistry, _bam_chemistry
    from .parameters import parse_options
    options=options or parse_options({'cr':{'engine':STAGED_MODE}})
    if options['cr'].engine!=STAGED_MODE:
        raise ValueError('Shared BAM preparation requires staged_native_families options')
    rows={};chemistry_records={}
    for item in datasets:
        dsid=item['dataset_id']
        if dsid in rows: raise ValueError('Duplicate dataset ID: '+dsid)
        paths=_flatten_paths(item['paths'])
        if not paths: raise ValueError('Dataset has no BAM paths: '+dsid)
        if not item.get('chemistry') and any(_bam_chemistry(Path(p))[0] is None for p in paths):
            raise ValueError('Every BAM needs chemistry metadata or an explicit dataset chemistry: '+dsid)
        chemistry,metadata=_resolve_chemistry([Path(p) for p in paths],item.get('chemistry'))
        chemistry_records[dsid]=dict(requested_chemistry=item.get('chemistry'),files=metadata)
        rows[dsid]=SimpleNamespace(paths=paths,path=paths[0],label=dsid,data_type='bam',
            family_scan_chemistry=chemistry,available_layers={})
    state=SimpleNamespace(get_dataset=rows.get)
    result=_load_payload(state,dict(region,dataset_ids=list(rows)),options,progress or (lambda *a:None))
    result['input_files']=[dict(dataset_id=k,path=str(Path(p).resolve()),size=Path(p).stat().st_size,
        mtime_ns=Path(p).stat().st_mtime_ns) for k,ds in rows.items() for p in ds.paths]
    for source in result['strata']: source['model_manifest']['chemistry_resolution']=chemistry_records[source['dataset_id']]
    return result


def _prefix_library(read, dataset_id: str) -> None:
    original = str(read.library_id or "")
    read.library_id = f"browser:{dataset_id}:{original}"
    read._browser_dataset_id = dataset_id
    read._browser_original_library_id = original


def _projection_member_key(read) -> tuple:
    """A retained alignment is not identified by QNAME/strand alone."""
    return (*read.molecule_id, str(getattr(read, 'record_sha256', '') or ''),
        int(getattr(read, 'alignment_occurrence', 0)), int(getattr(read, 'input_index', 0)),
        int(read.ref_start), int(read.ref_end))


def _projection_members(raw_reads: list, retained_reads: list, *, exact_records=False) -> dict[tuple, list[dict]]:
    clusters: dict[str, list] = defaultdict(list)
    for read in raw_reads:
        cluster = getattr(read, "amplification_family_id", None)
        if cluster:
            clusters[str(cluster)].append(read)
    result = {}
    for read in retained_reads:
        cluster = getattr(read, "amplification_family_id", None)
        members = clusters.get(str(cluster), ()) if cluster else (read,)
        key = _projection_member_key(read) if exact_records else read.molecule_id
        result[key] = [
            {
                "dataset_id": str(getattr(member, "_browser_dataset_id", "")),
                "read_name": str(member.name),
                "strand": str(member.strand),
            }
            for member in members
        ]
        if exact_records:
            for output, member in zip(result[key], members):
                output.update(library_id=str(getattr(member, '_browser_original_library_id', member.library_id) or ''),
                    reference_start=int(member.ref_start), reference_end=int(member.ref_end))
                if getattr(member, 'record_sha256', None):
                    output.update(record_sha256=str(member.record_sha256),
                        alignment_occurrence=int(member.alignment_occurrence),
                        alignment_orientation='reverse' if int(member.alignment_flag) & 16 else 'forward')
    return result


def _load_payload(state,request,options,progress):
    """Every regional read before collapse; no viewport read sample enters inference."""
    from fiberhmm.inference.consensus.adapter import evidence_unit,replay_alignment,condition_unit_on_m5c
    from fiberhmm.inference.tf_recaller import ENZYME_PRESETS
    runtime=loader_runtime();strata=[];start,end=request['start'],request['end'];chrom=request['chrom']
    for dsid in request['dataset_ids']:
        ds=state.get_dataset(dsid)
        chemistry=request.get('chemistry_overrides',{}).get(dsid) or getattr(ds,'family_scan_chemistry',None)
        if chemistry not in ('ddda','dddb','hia5-pacbio','hia5-nanopore'):
            raise ValueError(f'{ds.label}: choose a datatype in the Footprint panel; guessing the opportunity model is unsafe')
        from fiberhmm.inference.consensus.progress import report
        report(progress,'loading',f'{ds.label}: loading all regional observations ({chemistry})',dataset_id=dsid)
        reads,model,preset,diagnostics=_load_dataset_evidence(state,dsid,chemistry,chrom=chrom,
            evidence_start=start,evidence_end=end,maximum_reads=0,runtime=runtime,
            allow_population=True,require_nucleosomes=False,tf_layer='tf',minimum_mapq=options['input'].minimum_mapq,
            **({'legacy_annotation_frame':options['input'].legacy_hia5_annotation_frame} if chemistry.startswith('hia5') and options['cr'].engine in (HARMONIZATION_MODE,STAGED_MODE) else {}))
        if any(read.pair_partner for read in reads):
            raise ValueError('Paired source reads are not independent molecules. Run fiberhmm-merge --recall before population consensus; if pairs failed to merge, --pairs-only excludes those unresolved pairs.')
        for read in reads:_prefix_library(read,dsid)
        if chemistry in ('ddda','dddb'):
            representatives,collapse=runtime['collapse'](reads,min_jaccard=options['input'].molecule_min_jaccard,
                min_deam=options['input'].molecule_min_deam)
        else:
            representatives=reads;collapse=dict(raw_reads=len(reads),analyzed_molecules=len(reads),duplicate_reads_collapsed=0)
        projections=_projection_members(reads,representatives,exact_records=True)
        units=[evidence_unit(read,model,dsid,projections.get(_projection_member_key(read),[]),start,end) for read in representatives]
        use_m5c=chemistry=='ddda' and options['input'].ddda_m5c_correction
        if use_m5c and ds.data_type!='bam' and ds.available_layers.get('ddda_mcg'):
            raise ValueError('Tagged DddA mCG conditioning currently needs BAM query coordinates; use BAM or explicitly disable that input option')
        minimum_llr=None
        if options['input'].correct_native or (use_m5c and ds.data_type=='bam'):
            if ds.data_type!='bam':raise ValueError(f'{ds.label}: actual-query decoder replay requires BAM; disable native replay to use existing BigBed TF calls')
            import pysam
            lookup=defaultdict(list)
            for read,unit in zip(representatives,units):lookup[str(read.record_sha256)].append((read,unit))
            found=set()
            _,_,context_size,mode,_,_=_chemistry_runtime(runtime,chemistry,allow_population=True)
            enzyme='hia5' if chemistry.startswith('hia5') else chemistry
            minimum_llr=getattr(options['input'],chemistry.replace('-','_')+'_minimum_llr')
            if minimum_llr<0:minimum_llr=ENZYME_PRESETS[enzyme]['min_llr']
            for path in _flatten_paths(getattr(ds,'paths',None) or ds.path):
                actual,lo,hi,_=_resolve_bam_fetch_region(path,chrom,start,end)
                with pysam.AlignmentFile(path,'rb') as bam:
                    for alignment in bam.fetch(actual,lo,hi):
                        if alignment.is_secondary or alignment.is_supplementary:continue
                        key=hashlib.sha256(alignment.to_string().encode()).hexdigest()
                        if key not in lookup or key in found:continue
                        if len(found)%32==0:
                            report(progress,'native',f'{ds.label}: replaying the native caller {len(found)}/{len(lookup)} alignments',
                                dataset_id=dsid,completed=len(found),total=len(lookup),unit='alignments')
                        else:progress('native',None)
                        # Identical SAM records may be distinct retained evidence
                        # units. Replay every unit; never collapse them in this lookup.
                        for read,u in lookup[key]:
                            if use_m5c:condition_unit_on_m5c(alignment,u)
                            if options['input'].correct_native:
                                if options['cr'].engine==STAGED_MODE and chemistry.startswith('hia5') and options['families'].recall_hia5_nucleosomes:
                                    from fiberhmm.inference.consensus.upstream_recall import recall_hia5_alignment,install_recall
                                    recalled=recall_hia5_alignment(alignment,u,model,preset['strand_mode'],mode,context_size,preset.get('prob_threshold'),
                                        minimum_llr,minimum_opportunities=options['input'].native_minimum_opportunities,
                                        split_minimum_llr=options['families'].nuc_split_minimum_llr,
                                        maximum_alignment_gap_bp=options['input'].native_maximum_alignment_gap_bp,
                                        minimum_nfr_length=options['input'].minimum_nfr_length,
                                        legacy_annotation_frame=options['input'].legacy_hia5_annotation_frame)
                                    install_recall(u,recalled)
                                    continue
                                replay_unit=u
                                if options['cr'].engine==HARMONIZATION_MODE:
                                    domains=[[max(start,a),min(end,b)] for a,b in u['msp_intervals'] if a<end and b>start]
                                    replay_unit=dict(u,msp_intervals=domains)
                                    u['provenance']['native_replay_scope']=dict(region=[start,end],effective_msp_intersections=domains,
                                        minimum_nfr_length=options['input'].minimum_nfr_length,boundary_clipping_is_scope_not_quality=True)
                                calls=replay_alignment(alignment,replay_unit,model,preset['strand_mode'],mode,context_size,preset.get('prob_threshold'),
                                    minimum_llr,minimum_opportunities=options['input'].native_minimum_opportunities,
                                    minimum_nfr_length=options['input'].minimum_nfr_length,use_m5c=use_m5c,
                                    maximum_alignment_gap_bp=int(getattr(options['input'],'native_maximum_alignment_gap_bp',0)))
                                u['native_multi_interval_calls']=calls;u['native_multi_interval_tf_intervals']=[c['interval'] for c in calls]
                        found.add(key)
            if len(found)!=len(lookup):raise ValueError(f'{ds.label}: could not identify every representative alignment for native replay')
        # Keep chemical CT/GA order, but never split Hia5 by mapping orientation.
        units.sort(key=lambda u:((0 if u['strand']=='CT' else 1),u['unit_id']) if chemistry in ('ddda','dddb') else (0,u['unit_id']))
        model_hash=hashlib.sha256(model.emissionprob_.astype('<f8').tobytes()).hexdigest()
        strata.append(dict(dataset_id=dsid,stratum_id=dsid,chemistry=chemistry,units=units,
            model_manifest=dict(preset=chemistry,emissions_sha256=model_hash,efficiency_scaling=False,
                native_minimum_llr=minimum_llr if options['input'].correct_native else None,
                replay_scope=('query_nuc_recall_then_TF_replay' if any('upstream_nuc_tf_recall' in u for u in units) else
                    'fixed_MSP_intersection_with_analysis_region' if options['cr'].engine==HARMONIZATION_MODE else 'fixed_MSP_only') if options['input'].correct_native else 'existing_calls',
                legacy_hia5_annotation_frame=options['input'].legacy_hia5_annotation_frame if chemistry.startswith('hia5') else None,
                native_minimum_msp_bp=options['input'].minimum_nfr_length,
                native_minimum_opportunities=options['input'].native_minimum_opportunities,
                upstream_nuc_tf_recall=any('upstream_nuc_tf_recall' in u for u in units),
                upstream_nuc_tf_settings=next((u['upstream_nuc_tf_recall']['settings'] for u in units if 'upstream_nuc_tf_recall' in u),None),
                m5c_conditioned_opportunities=sum(u['provenance'].get('native_m5c_conditioned_opportunities',0) for u in units)),
            evidence_units=dict(**collapse,physical_duplex_independence_established=bool(units) and all(u.get('physical_source_names') for u in units),joint_duplex_units=sum(bool(u.get('physical_source_names')) for u in units)),load=diagnostics))
    return dict(region=dict(chrom=chrom,start=start,end=end),strata=strata)
