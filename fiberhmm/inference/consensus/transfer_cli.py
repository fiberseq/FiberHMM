"""Freeze a consensus run's classes (lattice_recaller) or families (staged_native_families) and apply them to new
evidence or oriented target BAM windows without rediscovery or refitting."""
import argparse
import csv
import html
import json
from pathlib import Path
from .artifacts import read_json,write_json,digest
from .transfer import export_run,load_bundle,score_payload,run_engine,RECALLER_MODE
from .cli import Progress, check_chemistry, report_chemistry_errors

STAGED_SCHEMA='fiberhmm.frozen_families.v1'
MODEL_FILES=('frozen_classes.json.gz','frozen_models.json.gz')


def _model_path(value):
    """--models accepts the frozen file or the --freeze-run output directory holding it."""
    path=Path(value)
    if path.is_dir():
        found=[path/n for n in MODEL_FILES if (path/n).is_file()]
        if len(found)!=1:raise ValueError(f'{path} must contain exactly one of {", ".join(MODEL_FILES)}')
        return found[0]
    return path


def main(argv=None):
    from fiberhmm.cli.common import run_reporting_input_errors
    return run_reporting_input_errors(
        'fiberhmm-transfer', lambda: report_chemistry_errors(lambda: _main(argv),'fiberhmm-transfer'))


def _main(argv=None):
    parser=argparse.ArgumentParser(prog='fiberhmm-transfer',description=__doc__)
    source=parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--freeze-run',help='Completed consensus result directory: a lattice_recaller run (any frame) or an oriented '
                        'staged_native_families CL-CR run; export its classes/models without refitting')
    source.add_argument('--models',help='frozen_classes.json.gz (lattice_recaller), frozen_models.json.gz (staged), or the --freeze-run output directory')
    inputs=parser.add_mutually_exclusive_group()
    inputs.add_argument('--bam',action='append',help='Target BAM; repeat for separate datasets (needs --bed)')
    inputs.add_argument('--datasets',help='JSON list of dataset_id and BAM paths')
    inputs.add_argument('--evidence',help='Saved oriented native evidence.json.gz (one window or pooled evidence)')
    parser.add_argument('--bed',help='Equal-width, explicitly oriented BED6 target windows')
    parser.add_argument('--chemistry',choices=['ddda','dddb','hia5-pacbio','hia5-nanopore'],help='Explicit missing-metadata declaration for --bam, as in fiberhmm-consensus')
    parser.add_argument('--parameters',help='BAM preparation parameters; target calls are replayed, families are never fitted')
    parser.add_argument('--chip-bed',help='Optional independent ChIP peaks, joined only after scoring')
    parser.add_argument('--no-bam',action='store_true',help='Do not write family-tagged BAMs')
    parser.add_argument('--bam-scope',choices=['regions','full'],default='regions',help='Export alignments overlapping the target windows (default), or the full source BAM')
    parser.add_argument('--bam-grouping',choices=['datasets','files'],default='datasets',help='One BAM per logical dataset (default) or per source file')
    parser.add_argument('--json-progress',action='store_true',help='Structured progress on stderr')
    recaller=parser.add_argument_group('lattice_recaller catalogs')
    recaller.add_argument('--cores',type=int,help='Worker processes for scoring (default: this machine\'s consensus default)')
    recaller.add_argument('--dataset-map',action='append',default=[],metavar='TARGET=SOURCE',
                          help='Use the frozen per-channel boxes and spots of source dataset SOURCE for target dataset TARGET '
                               '(needed when several source datasets share the target chemistry)')
    recaller.add_argument('--include-training-molecules',action='store_true',
                          help='Score molecules the catalog was trained on (default: exclude them, as for staged families); '
                               'use for self-application checks')
    parser.add_argument('--output',required=True,help='New or empty output directory')
    from fiberhmm.cli.common import add_version_args
    add_version_args(parser)
    args=parser.parse_args(argv);out=Path(args.output)
    if out.exists() and any(out.iterdir()):parser.error('Output directory must be empty')
    out.mkdir(parents=True,exist_ok=True);progress=Progress(args.json_progress)
    if args.freeze_run:
        if args.bam or args.datasets or args.evidence or args.bed or args.chip_bed:parser.error('--freeze-run accepts no target inputs')
        if run_engine(args.freeze_run)==RECALLER_MODE:
            from .lattice_recaller.frozen import CATALOG_NAME
            catalog=export_run(args.freeze_run,out/CATALOG_NAME)
            progress('complete',f"Exported {len(catalog['classes'])} frozen classes over {len(catalog['channels'])} channels")
            return
        bundle=export_run(args.freeze_run,out/'frozen_models.json.gz')
        progress('complete',f"Exported {len(bundle['models'])} frozen families")
        return
    if not (args.bam or args.datasets or args.evidence):parser.error('Supply --bam, --datasets or --evidence')
    if not args.models:parser.error('Supply --models (a --freeze-run output)')
    try:
        model_path=_model_path(args.models);document=read_json(model_path)
    except (UnicodeDecodeError,json.JSONDecodeError) as error:
        parser.error(f'--models {args.models} is not a JSON frozen class catalog ({error})')
    except ValueError as error:parser.error(f'--models {args.models}: {error}')
    except (OSError,EOFError) as error:parser.error(f'--models {args.models}: cannot read it as (gzipped) JSON ({error})')
    schema=document.get('schema') if isinstance(document,dict) else None
    from .lattice_recaller.frozen import SCHEMA as RECALLER_SCHEMA
    if schema==RECALLER_SCHEMA:
        return _apply_recaller(args,parser,out,progress,model_path)
    if schema!=STAGED_SCHEMA:
        parser.error(f'--models {args.models} is not a frozen class catalog (schema {schema!r}); give the frozen_classes.json.gz '
                     f'or frozen_models.json.gz that fiberhmm-transfer --freeze-run writes ({RECALLER_SCHEMA} or {STAGED_SCHEMA})')
    if args.cores is not None or args.dataset_map or args.include_training_molecules:
        parser.error('--cores, --dataset-map and --include-training-molecules apply to lattice_recaller catalogs only')
    args.models=str(model_path)
    if args.evidence and args.bed:parser.error('Saved evidence fixes its coordinate frame; omit --bed')
    if not args.evidence and not args.bed:parser.error('BAM transfer requires oriented BED6')
    if args.evidence and args.chip_bed:parser.error('ChIP evaluation requires explicit target BED windows')
    bundle=load_bundle(args.models)
    from .regions import load_bed,pool_payloads
    from .bam import load_bam_payload
    from .parameters import parse_options
    options=parse_options(dict(cr={'engine':'staged_native_families'}))
    if args.parameters:
        params=read_json(args.parameters);params.setdefault('cr',{})['engine']='staged_native_families';options=parse_options(params)
    datasets=read_json(args.datasets) if args.datasets else [dict(dataset_id=Path(p).stem,paths=[p],**({'chemistry':args.chemistry} if args.chemistry else {})) for p in (args.bam or [])]
    if len({d['dataset_id'] for d in datasets})!=len(datasets):parser.error('Duplicate dataset names; use --datasets with unique IDs')
    check_chemistry(parser,datasets)
    windows=load_bed(args.bed,pooled=True) if args.bed else [None]
    if any(w and w['end']-w['start']!=bundle['region'][1]-bundle['region'][0] for w in windows):parser.error('BED widths must match the frozen source window')
    analyses=[];all_results=[];summary=[]
    for index,w in enumerate(windows):
        progress('load',f'Window {index+1}/{len(windows)}')
        payload=read_json(args.evidence) if w is None else pool_payloads([load_bam_payload(datasets,dict(chrom=w['chrom'],start=w['start'],end=w['end']),options,progress)], [w])
        if w is not None and bundle['region'][0]:
            from .regions import orient_unit
            offset=bundle['region'][0]
            payload['strata']=[dict(source,units=[orient_unit(u,dict(chrom='oriented',start=0,end=w['end']-w['start'],strand='+',name=w['name']),origin=offset) for u in source['units']]) for source in payload['strata']]
            # Restore original genomic provenance; only the analysis origin changes.
            for source in payload['strata']:
                for u in source['units']:u['genomic_provenance']=dict(window=w,coordinate_origin=offset)
            payload['region'].update(start=bundle['region'][0],end=bundle['region'][1])
        result=score_payload(bundle,payload,progress=progress,maximum_bytes=options['compute'].maximum_matrix_mb*1024**2)
        result['window']=w or payload.get('pooling',{}).get('windows') or payload['region']
        result['input_digest']=digest(payload);write_json(out/f'window_{index+1:05d}.json.gz',result);all_results.append(result)
        for den in result['denominators']:
            eligible=set(den['eligible_units'])
            supported={r['unit_id'] for r in result['rows'] if r['dataset']==den['dataset'] and den['family'] in r['compatible']} & eligible
            assessed={r['unit_id'] for r in result['rows'] if r['dataset']==den['dataset'] and any(s['family']==den['family'] and s.get('compatible') is not None for s in r['scores'])} & eligible
            summary.append(dict(window=w['name'] if w else 'evidence',chrom=w['chrom'] if w else '',start=w['start'] if w else '',end=w['end'] if w else '',
                strand=w['strand'] if w else '',dataset=den['dataset'],family=den['family'],eligible_molecules=len(eligible),assessed_molecules=len(assessed),compatible_molecules=len(supported),
                compatibility_fraction=len(supported)/len(eligible) if eligible else None))
        if payload.get('input_files'):
            ds={s['dataset_id']:dict(cr=dict(records=[])) for s in payload['strata']}
            for row in result['rows']:
                ds[row['dataset']]['cr']['records'].append(dict(unit_id=row['unit_id'],proposals=[dict(source_interval=row['interval'],compatible_families=row['compatible'])]))
            export_result=dict(manifest=dict(input_digest=result['input_digest'],family_identity_digest=bundle['content_sha256'],parameters={'cross':{'enabled':True}},frozen_transfer=True),datasets=ds,final_stage='frozen_transfer')
            analyses.append((export_result,payload))
    # External labels never enter preparation, fitting, family selection or scoring.
    if args.chip_bed:
        if args.evidence:parser.error('ChIP evaluation requires explicit target BED windows')
        peaks=[]
        with open(args.chip_bed) as handle:
            for line in handle:
                if not line.strip() or line.startswith(('#','track ','browser ')):continue
                fields=line.split()
                if len(fields)<3 or int(fields[1])<0 or int(fields[2])<=int(fields[1]):raise ValueError('Invalid ChIP BED interval')
                peaks.append(dict(chrom=fields[0],start=int(fields[1]),end=int(fields[2])))
        for row in summary:row['chip_overlap']=int(any(p['chrom']==row['chrom'] and p['start']<row['end'] and p['end']>row['start'] for p in peaks))
    evaluation=[]
    if args.chip_bed:
        from sklearn.metrics import roc_auc_score,average_precision_score
        for dataset,family in sorted({(r['dataset'],r['family']) for r in summary}):
            subset=[r for r in summary if (r['dataset'],r['family'])==(dataset,family) and r['compatibility_fraction'] is not None]
            labels=[r['chip_overlap'] for r in subset];scores=[r['compatibility_fraction'] for r in subset]
            evaluation.append(dict(dataset=dataset,family=family,loci=len(subset),positives=sum(labels),
                auroc=float(roc_auc_score(labels,scores)) if len(set(labels))==2 else None,
                average_precision=float(average_precision_score(labels,scores)) if len(set(labels))==2 else None,
                fitting='none; direct frozen compatibility fraction',uncertainty='not estimated; loci may be correlated'))
        write_json(out/'chip_evaluation.json',evaluation)
    summary_fields=list(summary[0]) if summary else ['window','dataset','family','eligible_molecules','compatible_molecules']
    with (out/'families.tsv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=summary_fields,delimiter='\t');writer.writeheader();writer.writerows(summary)
    with (out/'calls.tsv').open('w',newline='') as handle:
        fields=['window','dataset','unit_id','chrom','start','end','eligible','compatible_families']
        writer=csv.DictWriter(handle,fieldnames=fields,delimiter='\t');writer.writeheader()
        for i,result in enumerate(all_results):
            for r in result['rows']:writer.writerow(dict(window=i+1,dataset=r['dataset'],unit_id=r['unit_id'],chrom=r['chrom'],start=r['genomic_interval'][0],end=r['genomic_interval'][1],eligible=r['eligible'],compatible_families=';'.join(r['compatible'])))
    if analyses and not args.no_bam:
        from .bam_export import export_bams
        export_bams(analyses,out/'bams',grouping=args.bam_grouping,scope=args.bam_scope,progress=progress)
    from .harmonized_families.workflow import implementation_hashes
    write_json(out/'manifest.json',dict(implementation_sha256=implementation_hashes(),preparation_parameters=read_json(args.parameters) if args.parameters else None,schema='fiberhmm.transfer_run.v1',model_sha256=bundle['content_sha256'],windows=len(windows),refitted=False,
        source_provenance=bundle['provenance'],input_digests=[r['input_digest'] for r in all_results],training_molecule_exclusions=sum(len(r['exclusions']) for r in all_results),
        chip_label='peak overlaps supplied window' if args.chip_bed else None))
    rows=''.join('<tr>'+''.join('<td>'+html.escape(str(r.get(k,'')))+'</td>' for k in summary_fields)+'</tr>' for r in summary)
    headers='<tr>'+''.join('<th>'+html.escape(k)+'</th>' for k in summary_fields)+'</tr>'
    lo,hi=bundle['region'];svg=['<svg xmlns="http://www.w3.org/2000/svg" width="900" height="'+str(50+35*len(bundle['models']))+'">','<rect width="100%" height="100%" fill="white"/>']
    for i,model in enumerate(bundle['models']):
        a,b=(model.get('geometry') or {}).get('mean') or model['reference_interval'];y=30+35*i
        svg.append(f'<text x="5" y="{y}" font-size="11">{html.escape(model["id"])}</text><rect x="{420+440*(a-lo)/(hi-lo)}" y="{y-12}" width="{440*(b-a)/(hi-lo)}" height="14" fill="#087e8b"/><text x="865" y="{y}" font-size="10">{b-a:.1f} bp</text>')
    svg.append('</svg>');(out/'families.svg').write_text(''.join(svg))
    (out/'report.html').write_text('<!doctype html><meta charset="utf-8"><title>Frozen footprint families</title><h1>Frozen footprint family transfer</h1><p>Source models were not refitted. Compatibility is not occupancy probability. Unassessed calls and training-molecule exclusions are retained in window JSON files.</p><img src="families.svg" alt="Frozen family mean spans in oriented analysis coordinates"><p><a href="families.tsv">Per-window family counts</a> · <a href="calls.tsv">Original call assignments</a></p>'+('<p><a href="chip_evaluation.json">Descriptive ChIP discrimination</a>: peak overlap with supplied windows; no target fitting or uncertainty estimate.</p>' if args.chip_bed else '')+'<table border="1">'+headers+rows+'</table>')
    progress('complete',f'{len(windows)} windows scored against {len(bundle["models"])} frozen families')


def _read_peaks(path):
    peaks=[]
    with open(path) as handle:
        for line in handle:
            if not line.strip() or line.startswith(('#','track ','browser ')):continue
            fields=line.split()
            if len(fields)<3 or int(fields[1])<0 or int(fields[2])<=int(fields[1]):raise ValueError('Invalid ChIP BED interval')
            peaks.append(dict(chrom=fields[0],start=int(fields[1]),end=int(fields[2])))
    return peaks


def _apply_recaller(args,parser,out,progress,model_path):
    """Score target evidence against a frozen lattice-recaller class catalog; each window is a normal recaller run."""
    from .lattice_recaller import frozen as F
    from .lattice_recaller.workflow import CLASS_FIELDS
    if args.evidence and args.bed:parser.error('Saved evidence fixes its coordinate frame; omit --bed')
    if not args.evidence and not args.bed:parser.error('BAM transfer requires oriented BED6')
    if args.evidence and args.chip_bed:parser.error('ChIP evaluation requires explicit target BED windows')
    catalog=F.load_catalog(model_path);frame=catalog['frame']['region'];width=frame['end']-frame['start']
    overrides=read_json(args.parameters) if args.parameters else None
    dataset_map={}
    for item in args.dataset_map:
        target,sep,source=item.partition('=')
        if not sep or not target or not source or target in dataset_map:parser.error('--dataset-map takes unique TARGET=SOURCE pairs')
        dataset_map[target]=source
    from .execution import single_threaded_blas
    jobs=[]
    if args.evidence:
        jobs.append((None,read_json(args.evidence)))
    else:
        from .regions import load_bed,pool_payloads
        from .bam import load_bam_payload
        datasets=read_json(args.datasets) if args.datasets else [dict(dataset_id=Path(p).stem,paths=[str(Path(p).resolve())],**({'chemistry':args.chemistry} if args.chemistry else {})) for p in (args.bam or [])]
        if len({d['dataset_id'] for d in datasets})!=len(datasets):parser.error('Duplicate dataset names; use --datasets with unique IDs')
        check_chemistry(parser,datasets)
        windows=load_bed(args.bed,pooled=True)
        if any(w['end']-w['start']!=width for w in windows):parser.error(f'BED widths must match the frozen frame ({width} bp)')
        load_options=F.transfer_options(catalog,[],overrides,1)
        for i,w in enumerate(windows):
            progress('load',f"Window {i+1}/{len(windows)} {w['name']}")
            pooled=pool_payloads([load_bam_payload(datasets,dict(chrom=w['chrom'],start=w['start'],end=w['end']),load_options,progress)],[w])
            same=not catalog['frame']['pooled'] and w['strand']=='+' and [w['start'],w['end']]==[frame['start'],frame['end']]
            region=dict(chrom=w['chrom'] if same else 'oriented_BED',start=frame['start'],end=frame['end'])
            jobs.append((w,F.shift_payload(pooled,frame['start'],region)))
    analyses=[];summary=[];runs=[]
    with single_threaded_blas():
        for i,(w,payload) in enumerate(jobs):
            folder=out if len(jobs)==1 else out/f'window_{i+1:06d}'
            progress('transfer',f"{w['name'] if w else 'evidence'}: scoring against {len(catalog['classes'])} frozen classes")
            result,target=F.apply_catalog(catalog,payload,folder,parameters=overrides,cores=args.cores,dataset_map=dataset_map,
                                          include_training=args.include_training_molecules,progress=progress,window=w)
            if target.get('input_files'):analyses.append((result,target))
            runs.append(dict(window=w,output=str(folder),input_digest=result['manifest']['input_digest'],
                             channel_map=result['transfer']['channel_map'],excluded_training_molecules=result['transfer']['excluded_training_molecules'],
                             classes=len(result['recaller']['rows'])))
            for r in result['recaller']['rows']:
                summary.append(dict(window=w['name'] if w else 'evidence',chrom=w['chrom'] if w else frame.get('chrom',''),
                    window_start=w['start'] if w else frame['start'],window_end=w['end'] if w else frame['end'],window_strand=w['strand'] if w else '',
                    source_channel=result['transfer']['channel_map'].get(r['channel']),**{k:r.get(k) for k in CLASS_FIELDS}))
    fields=['window','chrom','window_start','window_end','window_strand','source_channel']+CLASS_FIELDS
    with (out/'transfer_summary.tsv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields,delimiter='\t');writer.writeheader();writer.writerows(summary)
    evaluation=None
    if args.chip_bed:
        peaks=_read_peaks(args.chip_bed)
        for row in summary:row['chip_overlap']=int(any(p['chrom']==row['chrom'] and p['start']<row['window_end'] and p['end']>row['window_start'] for p in peaks))
        from sklearn.metrics import roc_auc_score,average_precision_score
        evaluation=[]
        for channel,cls in sorted({(r['channel'],r['class_id']) for r in summary}):
            subset=[r for r in summary if (r['channel'],r['class_id'])==(channel,cls)]
            labels=[r['chip_overlap'] for r in subset];scores=[r['prevalence'] for r in subset]
            evaluation.append(dict(channel=channel,class_id=cls,loci=len(subset),positives=sum(labels),
                auroc=float(roc_auc_score(labels,scores)) if len(set(labels))==2 else None,
                average_precision=float(average_precision_score(labels,scores)) if len(set(labels))==2 else None,
                fitting='none; frozen-class EM prevalence per window',uncertainty='not estimated; loci may be correlated'))
        write_json(out/'chip_evaluation.json',evaluation)
    bams=[]
    if analyses and not args.no_bam:
        from .bam_export import export_bams
        bams=export_bams(analyses,out/'bams',grouping=args.bam_grouping,scope=args.bam_scope,progress=progress)
    write_json(out/'transfer_manifest.json',dict(schema='fiberhmm.transfer_run.lattice_recaller.v1',catalog=str(Path(model_path).resolve()),
        catalog_sha256=catalog['content_sha256'],catalog_schema=catalog['schema'],source_provenance=catalog['provenance'],frame=catalog['frame'],
        refitted=False,rediscovered=False,preparation_parameters=overrides,dataset_map=dataset_map,
        include_training_molecules=args.include_training_molecules,windows=runs,bams=bams,apply_code=F.code_identity(),
        chip_label='peak overlaps supplied window' if args.chip_bed else None))
    if len(jobs)>1:
        (out/'report.html').write_text('<!doctype html><meta charset="utf-8"><title>Frozen class transfer</title><h1>Frozen class transfer</h1>'
            '<p>Classes, edge boxes and learned spots were fixed by the catalog; prevalences were estimated per window. '
            '<a href="transfer_summary.tsv">All windows</a></p>'+''.join('<p><a href="'+html.escape(str(Path(r['output']).relative_to(out)))+'/report.html">'
            +html.escape(r['window']['name'])+'</a></p>' for r in runs))
    progress('complete',f'{len(jobs)} window(s) scored against {len(catalog["classes"])} frozen classes')


if __name__=='__main__':main()
