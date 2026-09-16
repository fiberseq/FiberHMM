"""Rebuild Browser-only views from saved stage ledgers, without refitting.

Keeps the historical fitting receipt; records presentation changes separately.
The source run directory is never modified.
"""
import argparse
from copy import deepcopy
from pathlib import Path
import hashlib
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from fiberhmm.inference.consensus.artifacts import read_json,write_json,digest
from fiberhmm.inference.consensus.harmonized_families import MODE,presentation


def rebuild(source,output):
    started=time.monotonic()
    receipt=read_json(source/'manifest.json');grouped={}
    for path in sorted((source/'sources').glob('*/source.json.gz')):
        case=read_json(path);ds=case['dataset_id']
        entry=grouped.setdefault(ds,dict(dataset_id=ds,chemistry=case['chemistry'],units={}))
        for uid,unit in case['browser_units'].items():
            if uid in entry['units']:raise ValueError('Repeated original source unit')
            entry['units'][uid]=unit
    sources=[dict(s,units=list(s['units'].values())) for s in grouped.values()]
    context=presentation.presentation_context(sources,compact=True);snapshots={}
    for stage in [s['id'] for s in receipt['stages']]:
        annotations=[read_json(p) for p in sorted((source/'scopes').glob('*/'+stage+'.json.gz'))]
        snapshots[stage]=presentation.browser_snapshot([(None,ann) for ann in annotations],sources,receipt['mode'],stage,context)
        original={(r['unit_id'],tuple(r['interval'])) for ann in annotations for r in ann['records']}
        projected={(r['unit_id'],tuple(c['interval'])) for ds in snapshots[stage]['datasets'].values()
                   for r in ds['cr']['records'] for c in r['proposals']}
        if projected!=original:raise AssertionError('Presentation lost original calls')
        print(stage,'projected',len(projected),'original calls',flush=True)
    updated=deepcopy(receipt)
    updated.update(presentation_revision='staged_browser_v2',presentation_rebuild=dict(
        historical_manifest_digest=digest(receipt),historical_manifest_path=str((source/'manifest.json').resolve()),
        implementation_sha256=hashlib.sha256(Path(presentation.__file__).read_bytes()).hexdigest(),
        seconds=time.monotonic()-started,additional_fits=0,additional_simulations=0,
        changed=['thin_projection_units','native_display_primary','qualified_evidence_ids',
                 'separate_primary_and_compatible_unit_counts','alignment_aware_eligibility','no_comparability_claim','lossless_shared_evidence_storage']))
    result=dict(schema='fiberhmm.consensus.v1',cr_mode=MODE,manifest=updated,stages=receipt['stages'],
        final_stage=receipt['last_stage'],stage_results=snapshots,evidence_encoding='shared_json_v1',
        evidence_pool=context['evidence_pool'],**snapshots[receipt['last_stage']])
    output.mkdir(parents=True,exist_ok=False)
    write_json(output/'manifest.json',updated);write_json(output/'result.json.gz',result)
    print('Browser view:',output/'result.json.gz',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source',type=Path);parser.add_argument('output',type=Path)
    args=parser.parse_args();rebuild(args.source,args.output)
