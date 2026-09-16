"""Profile unchanged parent kernels on a saved real-locus native catalog."""
import argparse
import cProfile
from pathlib import Path
import pstats
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from fiberhmm.inference.consensus.artifacts import read_json,write_json,digest
from fiberhmm.inference.consensus.harmonized_families.reference.cross_source_family_consolidation import combine_cases
from fiberhmm.inference.consensus.harmonized_families.parallel import parent_task,parent_working_bytes,ordered_tasks


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--cores',type=int,default=2);p.add_argument('--limit',type=int,default=8)
    p.add_argument('--profile',action='store_true');args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    parts={}
    for path in sorted((args.run/'sources').glob('*/source.json.gz')):
        case=read_json(path);parts[case['channel']]=case
    case=combine_cases(parts)
    proposals=read_json(next((args.run/'scopes').glob('*/nominations.json')))
    # Include heavy real-data cohorts, not only the fastest tiny families.
    proposals=sorted(proposals,key=lambda v:parent_working_bytes(case,v),reverse=True)[:args.limit]
    context=args.output/'context.json.gz'
    write_json(context,dict(case={k:case[k] for k in ('units','calls','models','source_extent','source_by_unit')}))
    del parts
    tasks=[(p['id'],(str(context.resolve()),p,2*1024**3),parent_working_bytes(case,p)) for p in proposals]
    if args.profile:
        profile=cProfile.Profile();profile.enable();value=parent_task(*tasks[0][1]);profile.disable()
        write_json(args.output/'profile_result.json.gz',value)
        profile.dump_stats(str(args.output/'parent.prof'))
        pstats.Stats(profile).sort_stats('cumulative').print_stats(35)
    else:
        started=time.monotonic()
        results=ordered_tasks(parent_task,tasks,cores=args.cores,maximum_bytes=2*1024**3,
            progress=lambda stage,msg:None,stage='parent_fit',
            on_result=lambda key,value:write_json(args.output/(key.split(':')[1]+'.json.gz'),value))
        summary=dict(seconds=time.monotonic()-started,cores=args.cores,
            results={k:dict(seconds=v['seconds'],digest=digest(v.get('result',v.get('failure')))) for k,v in results.items()})
        write_json(args.output/'summary.json',summary);print(summary,flush=True)


if __name__=='__main__':main()
