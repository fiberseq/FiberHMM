"""Browser/CLI adapter for native-only lightweight harmonization.

The old fitted engines are not called. Source calls and their identities remain
immutable; comparison groups never add detections or estimate occupancy.
"""
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import asdict
from itertools import combinations
from pathlib import Path
import hashlib
import tempfile
import time

import numpy as np

from . import ClusterOptions, prepare, cluster_calls, physical_reason, RescueOptions
from .comparison import ComparisonOptions, coarse_events, native_counts, comparison_rows
from ..artifacts import digest, write_json
from ..parameters import options_dict

MODE='call_harmonization'


def run_harmonization(payload, options, output_dir=None, progress=None):
    t=time.monotonic();progress=progress or (lambda *_:None)
    if options['rescue'].enabled or options['split'].enabled:
        raise ValueError('Native harmonization does not add or split footprints. Disable rescue and nucleosome splitting; detection is controlled by the native LLR.')
    source=payload
    if not options['input'].correct_native:
        source=deepcopy(payload)
        from ...tf_recaller import ENZYME_PRESETS
        for s in source['strata']:
            s.setdefault('model_manifest',{})
            if s['model_manifest'].get('native_minimum_llr') is None:
                enzyme='hia5' if s['chemistry'].startswith('hia5') else s['chemistry']
                s['model_manifest']['native_minimum_llr']=ENZYME_PRESETS[enzyme]['min_llr']
                s['model_manifest']['threshold_is_comparison_reference_only']=True
            for u in s['units']:
                u['native_multi_interval_calls']=[dict(interval=list(iv)) for iv in u['representative_raw_tf_intervals']]
    elif any('native_multi_interval_calls' not in u for s in source['strata'] for u in s['units']):
        raise ValueError('Corrected native replay requested but actual-query replay is missing')
    if options['input'].correct_native:
        for s in source['strata']:
            model=s.get('model_manifest',{})
            requested=getattr(options['input'],s['chemistry'].replace('-','_')+'_minimum_llr')
            if requested>=0 and requested!=model.get('native_minimum_llr'):
                raise ValueError('Native LLR differs from prepared evidence. Reprepare from BAM; a frozen-evidence harmonization run cannot rerun detection.')
            for declared,option in [('native_minimum_msp_bp','minimum_nfr_length'),('native_minimum_opportunities','native_minimum_opportunities')]:
                if declared in model and model[declared]!=getattr(options['input'],option):
                    raise ValueError(f'{option} differs from prepared evidence; reprepare from BAM rather than relabel the frozen scan.')
    if source['region']['end']-source['region']['start']>options['compute'].maximum_region_bp:
        raise ValueError('Explicit maximum analysis span exceeded; no automatic cropping')
    reads,floors,region=prepare([source])
    strata={}
    for s in source['strata']:
        ds=s['dataset_id']
        if ds not in strata:
            strata[ds]=dict(s,units=list(s['units']))
        else:
            if strata[ds]['chemistry']!=s['chemistry']:
                raise ValueError(f'{ds}: conflicting chemistries within a dataset')
            strata[ds]['units'].extend(s['units'])
    if options['cross'].enabled and len({r['dataset'] for r in reads.values()})<2:
        raise ValueError('XCR requires at least two datasets')
    cr=options['cr'];cross=options['cross']
    cluster_opt=ClusterOptions(cut=cr.harmonization_cut,max_similarity=cr.harmonization_fold_overlap,
        max_call_bp=cr.harmonization_max_call_bp,edge_tolerance_bp=cr.harmonization_edge_tolerance_bp)
    compare_opt=ComparisonOptions(overlap=cr.harmonization_fold_overlap,maximum_span_bp=cr.harmonization_max_call_bp,
        minimum_covered_units=cross.lattice_minimum_covered_units,minimum_capable_fraction=cross.lattice_capable_fraction,
        minimum_opportunities=options['input'].native_minimum_opportunities)
    progress('clustering','Harmonizing native call classes; no rescue or simulations')
    clustered=cluster_calls(reads,region,cluster_opt) if cr.enabled else dict(results={})
    # Keep the native per-dataset SR catalog in the export. When XCR is on,
    # displayed labels use the common call-level XCR partition, not a fit forced
    # to have equal prevalence. All source membership ordinals remain exact.
    if cross.enabled and cr.enabled:
        families=clustered['results']['XCR']
    else:
        scopes=[k for k in clustered['results'] if k.startswith('SR ' if options['sr'].enabled else 'CR ')]
        families=[f for k in scopes for f in clustered['results'][k]]
    progress('comparability','Counting native members and checking each measured lattice')
    counted=native_counts(reads,families,floors,compare_opt)
    count_map={r['family']:r for r in counted}
    fine_rows=[dict(r,resolution='native_class',child_families=[r['family']]) for r in comparison_rows(counted,compare_opt)
               if cross.enabled or r['comparison_kind']=='strand']
    coarse=[];coarse_rows=[]
    if cross.enabled and cross.coarse_native_events:
        # Coarsen the SAME fine partition, never silently substitute an
        # independently regrouped catalog in a before/after comparison.
        coarse=coarse_events({'SR comparison':families},compare_opt,reads=reads)
        coarse_map={f['family_id']:f for f in coarse}
        coarse_rows=[dict(r,resolution='coarse_any_child',child_families=coarse_map[r['family']]['children']['comparison'],
                         group_kind=coarse_map[r['family']]['kind'],geometry_basis=coarse_map[r['family']]['geometry_basis'],
                         anchor_assay=coarse_map[r['family']]['anchor_assay'],child_intervals=coarse_map[r['family']]['child_intervals'])
                     for r in comparison_rows(native_counts(reads,coarse,floors,compare_opt),compare_opt)]
    member_family={(uid,ordinal):f for f in families for uid,ordinal in f['members']}
    datasets={}
    for s in strata.values():
        ds=s['dataset_id'];catalog=[];records=[];sr_records=[];changed=0
        for f in families:
            member_ids={uid for uid,_ in f['members'] if reads[uid]['dataset']==ds}
            if not member_ids:
                continue
            counts=count_map[f['family_id']];by_strand={st.rsplit(':',1)[1]:c for st,c in counts['by_stratum'].items() if st.rsplit(':',1)[0]==ds}
            call_counts=Counter(reads[uid]['strand'] for uid,_ in f['members'] if reads[uid]['dataset']==ds)
            catalog.append(dict(family=f['family_id'],consensus_start=f['interval'][0],consensus_end=f['interval'][1],
                interval_by_stratum=f['interval_by_stratum'],established=f['established'],source_units=len(member_ids),
                model_status='native_call_harmonization',
                classification_counts={st:dict(original_calls=call_counts[st],primary_calls=call_counts[st],compatible_calls=call_counts[st],
                    eligible_units=c['covered_units'],primary_units=c['assigned_units']) for st,c in by_strand.items()},
                rates_by_strand={st:dict(eligible_units=c['covered_units'],native_assignments=c['assigned_units'],
                    after_sr_assignments=c['assigned_units']) for st,c in by_strand.items()},
                evidence_summary=dict(native_counts=by_strand,calibrated_occupancy=False)))
        units=[]
        for u in s['units']:
            uid=u['unit_id'];r=reads[uid];proposals=[];sr_calls=[]
            intervals=[c['interval'] for c in r['calls']]
            units.append(dict(unit_id=uid,read_name=u.get('read_name',uid),strand=u['strand'],
                reference_start=u['reference_start'],reference_end=u['reference_end'],
                alignment_orientation=u.get('alignment_orientation'),source_members=u.get('source_members',[]),
                native_intervals=intervals,provenance=u.get('provenance',{})))
            for ordinal,c in enumerate(r['calls']):
                f=member_family.get((uid,ordinal));iv=c['interval'];target=f['interval_by_stratum'].get(r['stratum'],f['interval']) if f else iv
                # Only measurement-equivalent, physically valid coordinate
                # changes are shown as SR. CR always retains source spans.
                same=np.array_equal(np.searchsorted(r['positions'],iv),np.searchsorted(r['positions'],target))
                other=any(j!=ordinal and target[0]<v['interval'][1] and target[1]>v['interval'][0] for j,v in enumerate(r['calls']))
                normalized=bool(options['sr'].enabled and f and same and not other and not physical_reason(r,target,RescueOptions()))
                shown=target if normalized else iv;changed+=int(shown!=iv)
                sr_calls.append(dict(raw=iv,interval=shown,action='projection_equivalent_normalization' if shown!=iv else 'unchanged',
                                     family=f['family_id'] if f else None))
                proposals.append(dict(source_call_id=f'{ds}:{uid}:{ordinal}',source_ordinal=ordinal,source_interval=iv,
                    source_intervals=[iv],interval=iv,family=f['family_id'] if f else None,
                    classification_status='harmonized_native_call' if f else 'provisional_unresolved',
                    cr_mode=MODE,native_llr=c.get('llr'),native_opportunities=c.get('opportunities'),
                    harmonized_interval=target,compatible_alternatives=[],new_call=False))
            records.append(dict(unit_id=uid,strand=r['strand'],source_calls=intervals,proposals=proposals))
            sr_records.append(dict(unit_id=uid,strand=r['strand'],calls=sr_calls))
        datasets[ds]=dict(chemistry=s['chemistry'],units=units,
            sr=dict(status='complete' if options['sr'].enabled else 'disabled',records=sr_records,changed=changed),
            cr=dict(status='complete' if cr.enabled else 'disabled',cr_mode=MODE,catalog=catalog,records=records),
            rescue=dict(status='disabled',accepted_calls=0,records=[]),split=dict(status='disabled',accepted_spans=0,records=[]),
            comparability=dict(status='complete',records=[]))
    edges=[]
    if cross.enabled:
        for row in fine_rows:
            if row['comparison_kind']!='assay':
                continue
            f=row['family'];left=row['left_label'];right=row['right_label']
            # Retain one-sided comparisons in the diagnostics, not as a
            # relationship to a nonexistent catalog node.
            if not row['left']['original_member_calls'] or not row['right']['original_member_calls']:
                continue
            edges.append(dict(edge_id=f'{f}:{left}:{right}',left_dataset=left,right_dataset=right,left_family=f,right_family=f,
                left_interval=row['interval'],right_interval=row['interval'],status=row['comparability'],
                comparability_mask=row['comparability']=='lattice_comparable',comparison=row,
                rate_agreement_used_for_selection=False))
    receipt=dict(schema='fiberhmm.consensus.v1',status='complete',cr_mode=MODE,region=region,parameters=options_dict(options),
        datasets=[dict(dataset_id=s['dataset_id'],chemistry=s['chemistry'],units=len(s['units']),model=s.get('model_manifest')) for s in strata.values()],
        browser_sources=source.get('browser_sources',{}),input_digest=digest(source),all_units=True,read_sample_cap=None,
        family_count_cap=None,native_source_modified=False,rescue=False,
        implementation_sha256={str(p.relative_to(Path(__file__).parents[3])):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(list(Path(__file__).parent.glob('*.py'))+
                [Path(__file__).parents[1]/name for name in ('adapter.py','parameters.py','workflow.py')]+
                [Path(__file__).parents[2]/name for name in ('tf_recaller.py','strand_rescue.py','legacy_annotations.py')])},
        seconds=time.monotonic()-t)
    result=dict(schema='fiberhmm.consensus.v1',cr_mode=MODE,manifest=receipt,datasets=datasets,
        cross=dict(status='complete' if cross.enabled else 'disabled',edges=edges,count_groups=[],
                   comparable_edges=sum(e['comparability_mask'] for e in edges)),
        harmonization=clustered,comparison=dict(options=asdict(compare_opt),rows=fine_rows+coarse_rows,coarse_events=coarse,
            counts=Counter(r['comparability'] for r in fine_rows),
            semantics='Native-only counts; common aligned-span denominator; lattice capability and observed agreement are separate. Coarse events are ANY-child detections, not additional footprints.'))
    out=Path(output_dir or tempfile.mkdtemp(prefix='fiberhmm-harmonization-'))
    out.mkdir(parents=True, exist_ok=True)
    progress('saving','Saving native-only harmonization and per-class comparisons')
    write_json(out/'result.json.gz',result);write_json(out/'manifest.json',receipt)
    return result
