"""Footprint-class consensus (default engine: lattice recaller) from BAM/BED or saved evidence.

The deprecated staged Monte Carlo engine (--engine staged_native_families) also resumes from native-fit checkpoints."""
import argparse
import json
import sys
import time
from pathlib import Path
from .artifacts import read_json, write_json
from .parameters import CROptions, parameter_schema, parse_options
from .workflow import run_analysis
from .regions import load_bed, pool_payloads, automatic_parameters


class Progress:
    def __init__(self, json_output=False):
        self.json_output=json_output;self.last_stage=None;self.last_time=0.
    def __call__(self, stage, message): self.report(stage,message)
    def report(self, stage, message, **work):
        if message is None: return
        now=time.monotonic()
        if not self.json_output and stage==self.last_stage and now-self.last_time<.25 and work.get('completed')!=work.get('total',-1): return
        self.last_stage=stage;self.last_time=now
        if self.json_output:
            print(json.dumps(dict(stage=stage,message=message,**work)),file=sys.stderr,flush=True)
        else:
            done,total=work.get('completed'),work.get('total');bar=''
            if done is not None and total:
                n=min(24,max(0,int(24*done/total)));bar='['+'='*n+'.'*(24-n)+f'] {done}/{total} '
            print(f'{stage}: {bar}{message}',file=sys.stderr,flush=True)


def check_chemistry(parser, datasets):
    """Resolve every BAM dataset's chemistry up front; a missing, unsupported or conflicting
    chemistry (or an unreadable BAM) is a one-line error (exit 2), not a traceback. It is an input
    problem, not a usage one, so no usage block is printed."""
    from fiberhmm.io.bam_header import ChemistryResolutionError
    from .bam import check_dataset_chemistries
    try: check_dataset_chemistries(datasets)
    except (ChemistryResolutionError, OSError) as error: parser.exit(2,f'{parser.prog}: error: {error}\n')


def report_chemistry_errors(run, prog):
    """Run ``run()``; a chemistry error raised later (e.g. from a --datasets entry) exits 2 with one line."""
    from fiberhmm.io.bam_header import ChemistryResolutionError
    try: return run()
    except ChemistryResolutionError as error:
        print(f'{prog}: error: {error}',file=sys.stderr);raise SystemExit(2)


def main(argv=None):
    return report_chemistry_errors(lambda: _main(argv),'fiberhmm-consensus')


def _main(argv=None):
    p=argparse.ArgumentParser(prog='fiberhmm-consensus',description=__doc__)
    p.add_argument('--schema',action='store_true',help='Print every parameter group with defaults and help as JSON (cr.engine selects the engine; the default is lattice_recaller)')
    source=p.add_mutually_exclusive_group()
    source.add_argument('--bam',action='append',help='Repeat for separate datasets; chemistry comes from BAM @CO metadata')
    source.add_argument('--datasets',help='JSON list of {dataset_id, paths: [BAMs], chemistry?: profile}')
    source.add_argument('--evidence',help='Saved evidence.json.gz, including pooled evidence')
    source.add_argument('--resume',help='Previous run directory containing evidence.json.gz, manifest.json and fit_cache')
    p.add_argument('--bed',help='BED3 windows for independent runs; BED6 with equal widths for CL-CR')
    p.add_argument('--region',action='append',help='Alternative CHROM:START-END, explicitly 0-based half-open')
    p.add_argument('--pool-loci',action='store_true',help='CL-CR: pool BED6 windows in their provided orientations')
    p.add_argument('--chemistry',choices=['ddda','dddb','hia5-pacbio','hia5-nanopore'],help='Explicit missing-metadata declaration for --bam; conflicts fail')
    p.add_argument('--parameters',help='JSON parameter groups; see --schema')
    p.add_argument('--engine',choices=['lattice_recaller','staged_native_families'],help='Consensus engine (default lattice_recaller; staged_native_families is the deprecated Monte Carlo engine). Overrides cr.engine from --parameters or --resume')
    p.add_argument('--consolidation-bp',type=int,help='staged_native_families only: shared-family edge allowance (default 10; 5 gives finer grouping)')
    p.add_argument('--stop-after',choices=['native','parents','consolidated','resolved'],help='staged_native_families only: last stage to compute (the lattice recaller runs in one pass)')
    p.add_argument('--start-at',choices=['native','consolidation'],default='native',help='staged_native_families only: consolidation restarts from saved native fits')
    p.add_argument('--cores',type=int,help='Worker processes (sets compute.cores; default 4)')
    p.add_argument('--cache',help='staged_native_families only: persistent exact native-fit cache directory')
    p.add_argument('--json-progress',action='store_true',help='Structured progress on stderr')
    p.add_argument('--daf-mask-runs',type=int,default=None,metavar='N',help='DAF only: thin targets in same-strand runs of >= N original C (CT) or G (GA) bases in lattices and native replay (2 = CC/GG and longer; 0 = off). Default: per dataset chemistry, DddA keep-one on runs >= 2 (duplex-validated), DddB off')
    p.add_argument('--daf-run-policy',choices=['keep-one','drop'],default='keep-one',help="With --daf-mask-runs: keep each run's 5'-most target (default) or drop the run")
    p.add_argument('--no-bam',action='store_true',help='Save frozen results/reports without materializing family-tagged BAMs')
    p.add_argument('--bam-scope',choices=['regions','full'],default='regions',help='Export whole alignments overlapping analyzed windows (default), or the full source BAM')
    p.add_argument('--bam-grouping',choices=['datasets','files'],default='datasets',help='One BAM per logical dataset (default) or original source file')
    p.add_argument('--bam-recaller-layer',action='store_true',help="lattice_recaller: also write the optional tf_recaller MA layer (the recaller's own per-molecule class calls at every prevalence tier; bytes tq,fi,tier,q0,lr,rr) to exported BAMs. Off by default; the calls are always in result.json.gz")
    p.add_argument('--output',help='New or empty result directory')
    from fiberhmm.cli.common import add_version_args
    add_version_args(p)
    args=p.parse_args(argv)
    from fiberhmm.core.bam_reader import configure_daf_run_mask
    # An explicit value applies to every DAF dataset; unset leaves each dataset
    # on its chemistry default (bam._dataset_daf_run_mask).
    if args.daf_mask_runs is not None:
        try: configure_daf_run_mask(args.daf_mask_runs,args.daf_run_policy)
        except ValueError as error: p.error(str(error))
    if args.schema:
        print(json.dumps(parameter_schema(),indent=2));return
    if not args.output or not any((args.bam,args.datasets,args.evidence,args.resume)): p.error('Supply BAMs/datasets, evidence or resume, and --output')
    out=Path(args.output).resolve()
    if out.exists() and any(out.iterdir()): p.error('Output directory must be empty; existing results are never overwritten')
    if args.bed and args.region: p.error('Use --bed or --region')
    if args.pool_loci and not args.bed and not (args.evidence or args.resume): p.error('--pool-loci requires oriented BED6')
    if args.pool_loci and (args.evidence or args.resume): p.error('Saved evidence fixes pooling; omit --pool-loci when replaying')
    if (args.evidence or args.resume) and (args.bed or args.region): p.error('Saved evidence already fixes the windows')
    if args.start_at=='consolidation' and not (args.resume or (args.evidence and args.cache)): p.error('Consolidation requires --resume or --evidence plus --cache')
    values={};resume=Path(args.resume).resolve() if args.resume else None
    if resume and not (resume/'manifest.json').is_file(): p.error('Resume an individual window_XXXXXX directory or a pooled run, not the batch parent directory')
    if resume: values=read_json(resume/'manifest.json')['parameters']
    if args.parameters:
        for k,v in read_json(args.parameters).items(): values.setdefault(k,{}).update(v)
    engine=args.engine or values.get('cr',{}).get('engine') or CROptions().engine
    if engine not in ('lattice_recaller','staged_native_families'): p.error('This command runs lattice_recaller or staged_native_families; historical engines require the replay API')
    if engine=='lattice_recaller':
        # Reject staged-only stage controls before any evidence is loaded.
        for flag,value in (('--start-at consolidation',args.start_at=='consolidation'),('--stop-after',args.stop_after not in (None,'resolved')),
                           ('--consolidation-bp',args.consolidation_bp is not None),('--cache',args.cache is not None)):
            if value: p.error(f'{flag} applies to --engine staged_native_families only; the lattice recaller discovers and scores classes in one pass '
                              '(every run saves evidence.json.gz for replay with --evidence or --resume)')
    values.setdefault('cr',{})['engine']=engine
    for group,name,value in [('families','physical_radius_bp',args.consolidation_bp),('families','stop_after',args.stop_after),
                             ('compute','cores',args.cores),('compute','fit_cache_dir',args.cache)]:
        if value is not None: values.setdefault(group,{})[name]=value
    if resume and not args.cache and engine=='staged_native_families':
        previous=read_json(resume/'manifest.json')
        values.setdefault('compute',{})['fit_cache_dir']=previous['parameters']['compute']['fit_cache_dir'] or str(resume/'fit_cache')
    values.setdefault('compute',{})['require_native_cache']=args.start_at=='consolidation'
    if args.start_at=='consolidation' and args.stop_after=='native': p.error('Consolidation cannot stop before consolidation starts')
    if resume and args.stop_after is None: values.setdefault('families',{})['stop_after']='resolved'
    if values.get('compute',{}).get('fit_cache_dir'):
        values['compute']['fit_cache_dir']=str(Path(values['compute']['fit_cache_dir']).resolve())
    try: options=parse_options(values)
    except ValueError as error: p.error(str(error))
    datasets=None
    if not (args.evidence or resume):
        datasets=read_json(args.datasets) if args.datasets else [dict(dataset_id=f'dataset_{i+1}',paths=[str(Path(path).resolve())],chemistry=args.chemistry) for i,path in enumerate(args.bam)]
        check_chemistry(p,datasets)
    progress=Progress(args.json_progress)
    out.mkdir(parents=True,exist_ok=True)
    from .execution import single_threaded_blas
    from .bam import load_bam_payload
    with single_threaded_blas():
        if args.evidence or resume:
            payload=read_json(resume/'evidence.json.gz' if resume else args.evidence)
            count=1;jobs=iter([('analysis',payload,out)])
        else:
            if args.bed: windows=load_bed(args.bed,pooled=args.pool_loci)
            elif args.region:
                windows=[]
                for i,value in enumerate(args.region):
                    chrom,span=value.rsplit(':',1);a,b=map(int,span.split('-'))
                    if a<0 or b<=a: raise ValueError('Invalid 0-based region: '+value)
                    windows.append(dict(chrom=chrom,start=a,end=b,name=f'region_{i+1}',strand='+'))
            else: p.error('BAM input requires --bed or --region')
            if any(w['end']-w['start']>options['compute'].maximum_region_bp for w in windows): raise ValueError('BED window exceeds maximum_region_bp')
            def load(i,w):
                progress.report('windows',f"Loading {w['name']}",completed=i,total=len(windows))
                return load_bam_payload(datasets,{k:w[k] for k in ('chrom','start','end')},options,progress)
            if args.pool_loci:
                # CL-CR pools every window into one analysis, so all payloads are needed together.
                count=1;jobs=iter([('pooled',pool_payloads([load(i,w) for i,w in enumerate(windows)],windows),out)])
            else:
                # Independent windows stream: load, analyse, record the BAM assignments, then drop the payload
                # and result, so memory does not grow with the number of BED rows.
                count=len(windows)
                jobs=((w['name'],load(i,w),out if len(windows)==1 else out/f'window_{i+1:06d}') for i,w in enumerate(windows))
        from .bam_export import ExportPlan
        summary=[];plan=ExportPlan(recaller_layer=args.bam_recaller_layer) if not args.no_bam else None;bam_outputs=[]
        for i,(name,payload,folder) in enumerate(jobs):
            progress.report('regions',name,completed=i,total=count)
            result=run_analysis(payload,values,folder,progress=progress)
            if plan is not None and payload.get('input_files'): plan.add(result,payload)
            summary.append(dict(name=name,output=str(folder),status=result['manifest']['status'],seconds=result['manifest']['seconds']))
            del result,payload
        if plan is not None and plan.analyses:
            from .bam_export import export_bams
            bam_outputs=export_bams(plan,out/'bams',grouping=args.bam_grouping,scope=args.bam_scope,progress=progress)
        if count>1:
            write_json(out/'regions.json',summary)
            import html
            (out/'report.html').write_text('<!doctype html><meta charset="utf-8"><h1>Consensus windows</h1>'+''.join(
                '<p><a href="'+str(Path(r['output']).relative_to(out))+'/report.html">'+html.escape(r['name'])+'</a></p>' for r in summary))
    print(json.dumps(dict(status='complete',output=str(out),regions=summary,bams=bam_outputs)))


if __name__=='__main__':main()
