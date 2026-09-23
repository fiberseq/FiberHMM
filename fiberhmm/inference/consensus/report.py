"""Portable plot tables and a self-contained family geometry report."""
import csv
import html
import json
from pathlib import Path
from .artifacts import write_json


def write_report(result, output):
    out=Path(output); out.mkdir(parents=True,exist_ok=True)
    families=[]; calls=[]; stages=[]
    for stage,snapshot in result.get('stage_results',{result.get('final_stage','final'):result}).items():
        stages.append(dict(stage=stage,**next((s for s in result.get('stages',[]) if s['id']==stage),{})))
        for ds,data in snapshot['datasets'].items():
            units={u['unit_id']:u for u in data['units']}
            for f in data['cr']['catalog']:
                families.append(dict(stage=stage,dataset=ds,family=f['family'],start=f['consensus_start'],end=f['consensus_end'],
                    width=f['consensus_end']-f['consensus_start'],source_units=f.get('source_units',0),
                    edge_uncertainty=json.dumps(f.get('edge_uncertainty')),fit_flags=json.dumps(f.get('fit_flags',[])),
                    classification_counts=json.dumps(f.get('classification_counts',{})),
                    trusted_strand=(f.get('strand_resolution') or {}).get('trusted_strand',''),
                    core_resolution=(f.get('strand_resolution') or {}).get('core_resolution','')))
            for row in data['cr']['records']:
                u=units.get(row['unit_id'],{}); genomic=u.get('genomic_provenance',{})
                for p in row['proposals']:
                    from .bam_export import genomic_interval
                    chrom,genomic_span=genomic_interval(p['source_interval'],u,result['manifest']['region'])
                    calls.append(dict(chrom=chrom,genomic_start=genomic_span[0],genomic_end=genomic_span[1],stage=stage,dataset=ds,unit_id=row['unit_id'],read_name=u.get('read_name',''),
                        source_start=p['source_interval'][0],source_end=p['source_interval'][1],
                        compatible_families=json.dumps(p.get('compatible_families',[])),
                        status=p.get('assessment_status',''),window=json.dumps(genomic.get('window'))))
    def table(name,rows,fields):
        with (out/name).open('w',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=fields,delimiter='\t');writer.writeheader();writer.writerows(rows)
    table('families.tsv',families,['stage','dataset','family','start','end','width','source_units','edge_uncertainty','fit_flags','classification_counts','trusted_strand','core_resolution'])
    table('calls.tsv',calls,['chrom','genomic_start','genomic_end','stage','dataset','unit_id','read_name','source_start','source_end','compatible_families','status','window'])
    write_json(out/'report_data.json',dict(stages=stages,families=families,manifest=result['manifest']))
    final=[f for f in families if f['stage']==result.get('final_stage','final')]
    region=result['manifest']['region'];lo,hi=region['start'],region['end']; width=max(1,hi-lo)
    height=max(120,70+28*len(final)); shapes=[]
    for i,f in enumerate(final):
        y=45+28*i; x=300+680*(f['start']-lo)/width; w=680*f['width']/width
        label=html.escape(f"{f['dataset']} · {f['family']} ({f['source_units']} molecules)")
        shapes.append(f'<text x="10" y="{y+4}" font-size="10">{label}</text><rect x="{x}" y="{y-7}" width="{max(.5,w)}" height="14" fill="#0891b2"/>')
    svg=f'<svg xmlns="http://www.w3.org/2000/svg" width="1040" height="{height}" viewBox="0 0 1040 {height}"><rect width="100%" height="100%" fill="white"/><text x="300" y="18">{lo} — {hi} bp</text>'+''.join(shapes)+'</svg>'
    (out/'families.svg').write_text(svg)
    m=result['manifest']; mode=m.get('display_mode',m.get('mode','CR'));pool=m.get('pooling')
    title=('CL-' if pool else '')+mode+' footprint families'
    policy=m.get('numerical_policy',{})
    warnings=''.join('<p>'+html.escape(w)+'</p>' for w in m.get('data_warnings',[]))
    provenance='<table><tr><th>Dataset</th><th>Chemistry</th><th>Molecules</th><th>Replay</th></tr>'+''.join(
        '<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in [d['dataset_id'],d['chemistry'],d['units'],(d.get('model') or {}).get('replay_scope','saved_evidence')])+'</tr>'
        for d in m.get('datasets',[]))+'</table>'
    stage_text='Completed stage: '+result.get('final_stage','unknown')+'. '
    if result.get('final_stage') in ('consolidated','resolved'): stage_text+='Consolidation edge allowance ±'+str(m['parameters']['families']['physical_radius_bp'])+' bp.'
    else: stage_text+='Final consolidation has not been completed.'
    (out/'report.html').write_text('<!doctype html><meta charset="utf-8"><title>'+html.escape(title)+'</title>'
        '<style>body{font:16px system-ui;margin:36px;max-width:1200px}img{max-width:100%}td,th{padding:8px;text-align:left}</style>'
        '<h1>'+html.escape(title)+'</h1><p>'+f"{len(final)} dataset-family rows; elapsed inference {m.get('seconds',0):.1f} s. "
        +html.escape(stage_text)+'</p>'+provenance+warnings
        +f"<p>Full lattice-aware fits: {policy.get('predictive_replicates','unrecorded')} predictive draws, {policy.get('scoring_folds','unrecorded')} folds. "
        'Memberships may overlap; molecule counts are not exclusive occupancy estimates. '
        'Bars show fitted mean geometry, not original call boundaries. Edge uncertainty and fit warnings are in families.tsv.</p>'
        +('<p>Coordinates are relative to the oriented BED window. One locus view per physical molecule is retained.</p>' if pool else '')
        +'<img src="families.svg" alt="Final family geometry"><p><a href="families.tsv">Family plotting table</a> · '
        '<a href="calls.tsv">Call memberships</a> · <a href="report_data.json">Report data and provenance</a></p>')
