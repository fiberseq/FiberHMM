from copy import deepcopy
import json
from pathlib import Path

import pytest

from fiberhmm.inference.consensus.parameters import parse_options
from fiberhmm.inference.consensus.workflow import run_workflow
from fiberhmm.inference.consensus.harmonized_families.workflow import prepare_sources
from fiberhmm.inference.consensus.harmonized_families.presentation import browser_snapshot,presentation_context,eligible_units


def payload():
    p=list(range(0,81,3)); units=[]
    for i,(a,b) in enumerate([(12,51),(11,52),(12,50),(12,27),(11,28),(12,26)]):
        units.append(dict(unit_id=f'u{i}', strand='CT', positions=p,
            hits=[int(x<a or x>=b) for x in p], p_accessible=[.85]*len(p), p_protected=[.02]*len(p),
            raw_tf_intervals=[[a,b]], representative_raw_tf_intervals=[[a,b]], raw_nuc_intervals=[],
            msp_intervals=[[0,81]], aligned_blocks=[[0,81]], reference_start=0, reference_end=81))
    return dict(region=dict(chrom='chrTest',start=0,end=81), strata=[dict(dataset_id='test',chemistry='ddda',units=units,model_manifest={})])


def parameters(**families):
    return dict(cr=dict(engine='staged_native_families'), input=dict(correct_native=False),
                families=families, sr=dict(enabled=False), cross=dict(enabled=False),
                rescue=dict(enabled=False), split=dict(enabled=False), comparability=dict(enabled=False),
                compute=dict(cores=1,maximum_matrix_mb=128))


def test_stages_preserve_calls_and_use_real_mc(tmp_path):
    source=payload(); before=deepcopy(source)
    result=run_workflow(source,parameters(physical_radius_bp=5),tmp_path)
    assert source==before
    assert [s['id'] for s in result['stages']]==['native','parents','consolidated','resolved']
    expected=sorted(tuple(iv) for u in source['strata'][0]['units'] for iv in u['raw_tf_intervals'])
    for snapshot in result['stage_results'].values():
        ds=snapshot['datasets']['test']; calls=[p for r in ds['cr']['records'] for p in r['proposals']]
        assert sorted(tuple(p['interval']) for p in calls)==expected
        assert len({p['source_call_id'] for p in calls})==6
        known={f['family'] for f in ds['cr']['catalog']}
        assert all(set(p['compatible_families'])<=known for p in calls)
    assert sum(v['actual_simulations'] for v in result['manifest']['native_timings'].values())>0
    assert (tmp_path/'result.json.gz').is_file()
    assert (tmp_path/'evidence.json.gz').is_file()
    assert 'measurement_family.py' in result['manifest']['implementation_sha256']
    assert result['manifest']['parameters']['cr']['edge_tolerance_mode']=='bounded'
    assert all('positions' not in u for u in result['datasets']['test']['units'])


def test_hia5_pools_directions_and_rejects_duplicate_physical_molecules():
    data=payload(); s=data['strata'][0]; s['chemistry']='hia5-pacbio'
    for i,u in enumerate(s['units']): u.update(strand='FWD' if i%2 else 'REV',read_name=f'movie/{i}/ccs')
    options=parse_options(parameters())
    assert {u['strand'] for u in prepare_sources(data,options)[0]['units']}=={'pooled'}
    other=deepcopy(s);other['dataset_id']='other';data['strata'].append(other)
    with pytest.raises(ValueError,match='Repeated physical molecule'): prepare_sources(data,options)


def test_empty_calls_still_have_complete_empty_stages(tmp_path):
    data=payload()
    for u in data['strata'][0]['units']:u['raw_tf_intervals']=u['representative_raw_tf_intervals']=[]
    result=run_workflow(data,parameters(),tmp_path)
    assert len(result['datasets']['test']['units'])==6
    assert result['datasets']['test']['cr']['catalog']==[]
    assert all(s['original_calls']==0 for s in result['stages'])


def test_checkpoints_resume_before_consolidation_without_new_fits(tmp_path,monkeypatch):
    from fiberhmm.inference.consensus.harmonized_families import workflow
    data=payload()
    for i,u in enumerate(data['strata'][0]['units']):
        a,b=(12,30) if i<3 else (18,36)
        u['raw_tf_intervals']=u['representative_raw_tf_intervals']=[[a,b]]
        u['hits']=[int(x<a or x>=b) for x in u['positions']]
    values=parameters(physical_radius_bp=5,stop_after='parents')
    values['compute']['fit_cache_dir']=str(tmp_path/'cache')
    first=run_workflow(data,values,tmp_path/'first')
    assert first['final_stage']=='parents'
    assert first['manifest']['checkpoints']['misses']['native']==1
    def forbidden(*args,**kwargs):raise AssertionError('A cached stage recomputed evidence')
    monkeypatch.setattr(workflow,'fit_source',forbidden)
    monkeypatch.setattr(workflow,'ordered_tasks',forbidden)
    values['families']['stop_after']='resolved'
    second=run_workflow(data,values,tmp_path/'second')
    assert second['final_stage']=='resolved'
    assert second['manifest']['checkpoints']['hits']['native']==1
    assert second['manifest']['checkpoints']['hits']['foreign']==1
    assert all(v['checkpoint_reused'] for v in second['manifest']['native_timings'].values())
    def memberships(result):
        return [(c['source_call_id'],c['interval'],c['compatible_families'])
            for r in result['stage_results']['parents']['datasets']['test']['cr']['records'] for c in r['proposals']]
    assert memberships(first)==memberships(second)
    values['families']['minimum_retention_groups']=5
    third=run_workflow(data,values,tmp_path/'third')
    assert third['manifest']['checkpoints']['hits']['native']==1
    assert third['manifest']['checkpoints']['hits']['foreign']==1
    assert memberships(first)==memberships(third)


def test_unsupported_detector_stages_rejected():
    values=parameters();values['rescue']['enabled']=True
    with pytest.raises(ValueError,match='existing LLR calls'):parse_options(values)


@pytest.mark.parametrize('group,name,value',[('cr','ambiguity_bp',5),
    ('cross','native_predictive_replicates',8191),('sr','loss_odds',10),('compute','predictive_tilt','0.5')])
def test_unused_legacy_knobs_do_not_silently_pass(group,name,value):
    values=parameters();values.setdefault(group,{})[name]=value
    with pytest.raises(ValueError,match='not used by staged families'):parse_options(values)


def test_reference_scoring_policy_is_explicit_in_parameters():
    options=parse_options(parameters())
    assert options['cr'].edge_tolerance_mode=='bounded' and options['cr'].minimum_edge_tolerance_bp==2
    values=parameters();values['cr']['minimum_edge_tolerance_bp']=5
    with pytest.raises(ValueError,match='fixed to 2'):parse_options(values)


@pytest.mark.parametrize('mode',['SR','XCR'])
def test_shared_modes_keep_native_lattices_and_original_memberships(tmp_path,mode):
    data=payload();first=data['strata'][0];second=deepcopy(first)
    for i,u in enumerate(second['units']):
        u['unit_id']='foreign'+str(i);u['strand']='GA'
        p=list(range(1,81,4));a,b=u['raw_tf_intervals'][0]
        u.update(positions=p,hits=[int(x<a or x>=b) for x in p],p_accessible=[.7]*len(p),p_protected=[.04]*len(p))
    if mode=='SR': first['units'].extend(second['units'])
    else:second.update(dataset_id='hia5',chemistry='hia5-pacbio');data['strata'].append(second)
    before=deepcopy(data);values=parameters();values['sr']['enabled']=True;values['cross']['enabled']=mode=='XCR'
    result=run_workflow(data,values,tmp_path)
    assert data==before and all(s['original_calls']==12 for s in result['stages'])
    calls=[p for ds in result['datasets'].values() for r in ds['cr']['records'] for p in r['proposals']]
    assert len({c['source_call_id'] for c in calls})==12
    assert all(c['interval']==c['source_interval'] for c in calls)
    if mode=='XCR':
        assert result['cross']['edges']
        assert all(e['left_family']==e['right_family'] for e in result['cross']['edges'])
        assert all(e['shared_hypothesis'] and not e['comparability_mask'] for e in result['cross']['edges'])
        assert {r['strand'] for r in result['datasets']['hia5']['cr']['records']}=={'pooled'}


def test_primary_display_and_counts_preserve_native_order_and_ambiguity():
    source=payload()['strata'][0];source['units']=source['units'][:1]
    original=dict(compatible_families=['c::B','c::A'],primary_display_family='B',
        primary_evidence=dict(family='B'),inference_eligible=True)
    record=dict(unit_id='test::u0',interval=[12,51],display_hypotheses=['c::A','c::B'],
        original=original,status='multi_compatible',source_channel='c')
    annotation=dict(records=[record],hypotheses=[dict(id=f,reference_interval=[12,51],display=True,status='retained_alternative') for f in ['c::A','c::B']])
    result=browser_snapshot([(None,annotation)],[source],'CR','native')['datasets']['test']['cr']
    call=result['records'][0]['proposals'][0]
    assert call['family']=='c::B' and call['compatible_alternatives']==['c::A']
    assert call['family_evidence']['original']['primary_evidence']['family']=='c::B'
    assert original['primary_evidence']['family']=='B'
    counts={f['family']:f['classification_counts']['CT'] for f in result['catalog']}
    assert counts['c::A']['primary_units']==0 and counts['c::A']['compatible_units']==1
    assert counts['c::B']['primary_units']==1


def test_mean_span_eligibility_respects_alignment_gaps_and_nucleosomes():
    source=payload()['strata'][0];source['units']=source['units'][:1]
    unit=source['units'][0];unit['aligned_blocks']=[[0,20],[22,81]];unit['raw_nuc_intervals']=[[40,50]]
    context=presentation_context([source])
    assert eligible_units(context,'test','CT',10,19)=={'test::u0'}
    assert not eligible_units(context,'test','CT',10,25)
    assert not eligible_units(context,'test','CT',30,45)
