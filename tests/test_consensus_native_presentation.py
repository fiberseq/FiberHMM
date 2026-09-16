from copy import deepcopy
import pytest
from fiberhmm.inference.consensus.native_presentation import browser_cr,reclassify_records,MODE


def example():
    units=[dict(unit_id='u',strand='GA',representative_raw_tf_intervals=[[10,40],[60,80]],_region=[0,100])]
    calls=[dict(unit_index=0,unit_id='u',ordinal=i,start=s,end=e,strand='GA') for i,(s,e) in enumerate([[10,40],[60,80]])]
    score=lambda fid,upper,distance,status='scored':dict(family=fid,status=status,predictive_tail_interval=[0.,upper],
        geometry_distance_sq=distance,floor_adjusted_loss=1.)
    result=dict(calls=calls,call_family_evidence=[[score('d:F1',.2,100),score('d:F2',.005,0),
        score('d:F3',1.,0,'core_contradicted')],[]],family_models=[],diagnostics={})
    cat=[dict(family='d:'+f,consensus_start=10,consensus_end=40) for f in ['F1','F2','F3']]
    return dict(dataset_id='d',units=units),cat,result


def test_reference_expands_compatible_set_never_moves_or_drops_original_calls():
    s,c,r=example();cr=browser_cr(s,c,r);before=deepcopy(cr)
    strict=reclassify_records(cr,95)[0]['proposals'];loose=reclassify_records(cr,99.9)[0]['proposals']
    assert strict[0]['family']=='d:F1' and loose[0]['family']=='d:F2'
    assert loose[0]['compatible_alternatives']==['d:F1']
    assert [p['interval'] for p in strict]==[p['interval'] for p in loose]==[[10,40],[60,80]]
    assert [p['source_call_id'] for p in strict]==['d:u:0','d:u:1']
    assert loose[1]['primary_evidence'] is None and loose[1]['classification_status']=='provisional_unresolved'
    assert all('model_membership' not in p for p in loose)
    assert cr==before


def test_empty_catalog_retains_every_original_call():
    s,c,r=example();r['call_family_evidence']=[[],[]]
    cr=browser_cr(s,[],r)
    assert len(cr['records'][0]['proposals'])==2 and cr['catalog']==[]
    assert all(p['primary_evidence'] is None for p in cr['records'][0]['proposals'])


def test_candidate_testability_and_counts_are_not_posterior_or_occupancy():
    s,c,r=example();cr=browser_cr(s,c,r)
    counts={f['family']:f['classification_counts'] for f in cr['catalog']}
    assert counts['d:F2']['GA']['primary_calls']==1
    assert counts['d:F1']['GA']['compatible_calls']==1 and counts['d:F1']['GA']['primary_calls']==0
    assert counts['d:F3']['GA']['eligible_units']==1 and counts['d:F3']['GA']['compatible_calls']==0


def test_hia5_alignment_orientation_is_not_a_chemical_stratum():
    s,c,r=example();s['chemistry']='hia5-pacbio';s['units'][0]['strand']='REV'
    cr=browser_cr(s,c,r)
    assert cr['records'][0]['strand']=='pooled'
    assert cr['records'][0]['alignment_orientation']=='REV'
    assert list(cr['catalog'][0]['classification_counts'])==['pooled']


@pytest.mark.parametrize('value',[True,float('nan'),49.9,100.])
def test_invalid_reference_rejected(value):
    s,c,r=example()
    with pytest.raises(ValueError):reclassify_records(browser_cr(s,c,r),value)


def test_default_native_workflow_uses_real_span_preserving_engine(tmp_path):
    from test_consensus_workflow import payload
    from fiberhmm.inference.consensus.workflow import run_workflow
    source=payload();before=deepcopy(source)
    result=run_workflow(source,{'input':{'correct_native':False},'sr':{'enabled':False},
        'cr':{'engine':'native_family_distribution','ambiguity_bp':2,'local_iterations':10,'predictive_replicates':63,'scoring_folds':2,
              'family_fit_iterations':30},
        'rescue':{'enabled':True,'null_replicates':0}},tmp_path)
    assert result['cr_mode']==MODE and source==before
    for ds,data in result['datasets'].items():
        original={u['unit_id']:u['representative_raw_tf_intervals'] for s in source['strata'] if s['dataset_id']==ds for u in s['units']}
        assert data['cr']['cr_mode']==MODE and data['cr']['new_calls']==0
        for row in data['cr']['records']:
            assert [p['interval'] for p in row['proposals']]==original[row['unit_id']]
            assert all('model_membership' not in p for p in row['proposals'])
        assert data['cr']['diagnostics']['family_model']=='latent_distribution'
        assert data['cr']['diagnostics']['minimum_edge_tolerance_bp']==(10 if ds=='daf' else 0)
        assert data['rescue']['status'] not in {'not_available_in_native_mode','resource_limited'}
        if ds!='daf':
            assert data['rescue']['status']=='not_applicable'
    assert (tmp_path/'result.json.gz').is_file()
