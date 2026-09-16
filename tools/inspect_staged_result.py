"""Report a frozen staged run and optionally plot unchanged calls by family.

Sampling affects only displayed molecules. Every original call is counted.
No classification, geometry changes, or additional fitting takes place here.
"""
import argparse
from collections import Counter, defaultdict
import colorsys
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fiberhmm.inference.consensus.artifacts import read_json, write_json


def overlap(interval, view):
    return max(interval[0], view[0]) < min(interval[1], view[1])


def color(fid):
    hue = int(hashlib.sha256(fid.encode()).hexdigest()[:8], 16) / 2**32
    return colorsys.hsv_to_rgb(hue, .65, .72)


def summarize(result, view):
    counts = {}; signatures = []
    for stage, snapshot in result['stage_results'].items():
        cohorts = defaultdict(list); signature = []
        for ds, data in snapshot['datasets'].items():
            for row in data['cr']['records']:
                for call in row['proposals']:
                    signature.append((ds, row['unit_id'], call['source_call_id'], tuple(call['interval'])))
                    if overlap(call['interval'], view): cohorts[(ds, row['strand'])].append(call)
        signatures.append(sorted(signature))
        counts[stage] = {ds+' / '+strand:dict(calls=len(calls), eligible=sum(c['inference_eligible'] for c in calls),
            assigned=sum(bool(c['compatible_families']) for c in calls),
            multi_compatible=sum(len(c['compatible_families'])>1 for c in calls),
            statuses=dict(Counter(c['assessment_status'] for c in calls))) for (ds,strand),calls in cohorts.items()}
    if any(s != signatures[0] for s in signatures): raise AssertionError('Original-call identities changed between stages')
    families = {f['family']:f for ds in result['datasets'].values() for f in ds['cr']['catalog']
        if overlap([f['consensus_start'], f['consensus_end']], view)}
    return dict(view=list(view), original_calls_identical_across_stages=True, run_seconds=result['manifest']['seconds'],
        stages=result['stages'], in_view_counts=counts, final_hypotheses=[dict(id=f['family'],
            interval=[f['consensus_start'],f['consensus_end']], status=f['model_status'],
            edge_uncertainty=f['edge_uncertainty'],physical_edge_support=f['physical_edge_support']) for f in families.values()])


def plot(result, report, view, output, sample):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    source_rows = defaultdict(list)
    for ds, data in result['datasets'].items():
        for row in data['cr']['records']:
            if any(overlap(c['interval'],view) for c in row['proposals']): source_rows[(ds,row['strand'])].append(row)
    stages = list(result['stage_results'])
    fig, axes = plt.subplots(len(stages)+len(source_rows),1,figsize=(15,2*len(stages)+2.7*len(source_rows)),
        sharex=True, gridspec_kw={'height_ratios':[2]*len(stages)+[2.7]*len(source_rows)}, squeeze=False)
    axes = axes[:,0]
    for ax, stage in zip(axes, stages):
        families={f['family']:f for ds in result['stage_results'][stage]['datasets'].values() for f in ds['cr']['catalog']
            if overlap([f['consensus_start'],f['consensus_end']],view)}
        ends=[]
        for fid,f in sorted(families.items(), key=lambda p:(p[1]['consensus_start'],p[1]['consensus_end'],p[0])):
            left,right=f['consensus_start'],f['consensus_end']
            lane=next((i for i,end in enumerate(ends) if end<left),len(ends))
            if lane==len(ends): ends.append(right)
            else: ends[lane]=right
            ax.plot([left,right],[lane,lane],color=color(fid),lw=3,solid_capstyle='butt')
            box=f.get('edge_uncertainty') or {}
            for edge in ('left','right'):
                if edge in box:
                    a,b=box[edge];ax.add_patch(Rectangle((a,lane-.23),b-a,.46,color=color(fid),alpha=.18,lw=0))
        ax.set_ylim(-1,max(1,len(ends)));ax.set_yticks([])
        ax.set_title(f'{stage} · {len(families)} active hypotheses in view (including retained alternatives)',loc='left',fontsize=10)
    for ax,((ds,strand),rows) in zip(axes[len(stages):],sorted(source_rows.items())):
        selected=sorted(rows,key=lambda r:hashlib.sha256(r['unit_id'].encode()).hexdigest())[:sample]
        selected.sort(key=lambda r:r['unit_id'])
        counts=report['in_view_counts'][result['final_stage']][ds+' / '+strand]
        for y,row in enumerate(selected):
            ax.plot(view,[y,y],color='#edf0f3',lw=.4,zorder=0)
            for call in row['proposals']:
                if not overlap(call['interval'],view): continue
                ids=call['compatible_families']
                for i,fid in enumerate(ids or [None]):
                    offset=(i+.5)/max(1,len(ids))*.65-.325
                    ax.plot(call['interval'],[y+offset,y+offset],color=color(fid) if fid else '#9199a1',
                        lw=max(.25,2/max(1,len(ids))),solid_capstyle='butt')
        ax.set_ylim(-1,max(1,len(selected)));ax.set_yticks([])
        ax.set_title(f'{ds} / {strand} · {len(selected)} displayed call-bearing units · '
            f'{counts["assigned"]:,}/{counts["calls"]:,} original in-view calls assigned across all units',loc='left',fontsize=10)
    for ax in axes:
        ax.set_xlim(view);ax.spines[['top','right','left']].set_visible(False)
        ax.ticklabel_format(axis='x',style='plain',useOffset=False)
    axes[-1].set_xlabel('Genomic position (bp)')
    fig.suptitle('Lattice-aware CR / SR / XCR · frozen stage inspection',fontsize=17)
    fig.text(.5,.005,'Read bars retain original LLR spans. Multiple colors retain multiple compatible memberships; gray calls remain unassigned.\n'
        'Hypothesis bars show fitted mean edges with conditional 95% edge boxes. Display sampling does not limit inference.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.035,1,.97));fig.savefig(output,dpi=180);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('result',type=Path);parser.add_argument('--view',nargs=2,type=int)
    parser.add_argument('--output',type=Path);parser.add_argument('--sample',type=int,default=60)
    args=parser.parse_args();result=read_json(args.result)
    view=args.view or [result['manifest']['region'][k] for k in ('start','end')]
    if view[1]<=view[0] or args.sample<1:parser.error('Positive view and display sample required')
    report=summarize(result,view)
    if args.output:
        args.output.mkdir(parents=True,exist_ok=False)
        write_json(args.output/'summary.json',report)
        plot(result,report,view,args.output/'map.png',args.sample)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
