"""Run a frozen native evidence payload through the Browser's staged engine."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from fiberhmm.inference.consensus.artifacts import read_json
from fiberhmm.inference.consensus.workflow import run_workflow


def main():
    p=argparse.ArgumentParser();p.add_argument('input',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--region',nargs=2,type=int);p.add_argument('--dataset');p.add_argument('--radius',type=int,default=10)
    p.add_argument('--nomination-radius',type=int,default=2)
    p.add_argument('--use-replayed-calls',action='store_true',help='Classify the separately preserved native replay layer instead of original BAM TF calls')
    p.add_argument('--no-hia5-nuc-recall',action='store_true',help='Explicitly reuse a saved Hia5 scaffold without upstream nuc recall')
    p.add_argument('--minimum-retention-groups',type=int,default=2)
    p.add_argument('--cache-dir',type=Path)
    p.add_argument('--stop-after',choices=['native','parents','consolidated','resolved'],default='resolved')
    p.add_argument('--mode',choices=['CR','SR','XCR'],default='XCR');p.add_argument('--cores',type=int,default=1)
    p.add_argument('--matrix-mb',type=int,default=2048);args=p.parse_args()
    if args.output.exists():p.error('Choose a fresh output directory')
    payload=read_json(args.input)
    if args.region:payload['region'].update(start=args.region[0],end=args.region[1])
    if args.dataset:payload['strata']=[s for s in payload['strata'] if s['dataset_id']==args.dataset]
    options=dict(input=dict(correct_native=args.use_replayed_calls),cr=dict(engine='staged_native_families'),
        families=dict(physical_radius_bp=args.radius,nomination_radius_bp=args.nomination_radius,stop_after=args.stop_after,
            minimum_retention_groups=args.minimum_retention_groups,recall_hia5_nucleosomes=not args.no_hia5_nuc_recall),sr=dict(enabled=args.mode in ('SR','XCR')),
        cross=dict(enabled=args.mode=='XCR'),rescue=dict(enabled=False),split=dict(enabled=False),
        comparability=dict(enabled=False),compute=dict(cores=args.cores,maximum_matrix_mb=args.matrix_mb,
            fit_cache_dir=str(args.cache_dir.resolve()) if args.cache_dir else ''))
    result=run_workflow(payload,options,args.output,lambda stage,message:print(stage,message,flush=True))
    print(result['stages'],flush=True)


if __name__=='__main__':main()
