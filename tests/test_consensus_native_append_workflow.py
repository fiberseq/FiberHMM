"""Genuine tiny native fits; nomination/expensive downstream calls are bounded fixtures."""
from copy import deepcopy
from pathlib import Path

import pytest

from fiberhmm.inference.consensus.artifacts import digest, read_json
from fiberhmm.inference.consensus.parameters import parse_options, parameter_schema
from fiberhmm.inference.consensus.workflow import run_workflow
from fiberhmm.inference.consensus import native_workflow as nw


def source(two=False):
    from test_consensus_measurement_family import fixture
    stratum,_=fixture()
    stratum.update(chemistry='ddda',stratum_id='d')
    for unit in stratum['units']:
        unit.update(read_name=unit['unit_id'],fold_group_id=unit['unit_id'],
            aligned_blocks=[[0,81]],msp_intervals=[[0,81]])
    strata=[stratum]
    if two:
        other=deepcopy(stratum);other.update(dataset_id='h',stratum_id='h',chemistry='hia5-pacbio')
        for unit in other['units']:
            unit.update(unit_id='h'+unit['unit_id'],read_name='h'+unit['read_name'],
                        fold_group_id='h'+unit['fold_group_id'],strand='+')
        strata.append(other)
    return dict(region=dict(chrom='chr1',start=0,end=81),strata=strata)


def options(policy='append_frozen', **cr):
    out=dict(input=dict(correct_native=False),sr=dict(enabled=False),
        cr=dict(engine="native_family_distribution",predictive_replicates=31,scoring_folds=2,family_fit_iterations=25,**cr),
        compute=dict(maximum_matrix_mb=32))
    if policy is not None:out['cr']['residual_update_policy']=policy
    return out


def install(monkeypatch, output, *, add=True, corrupt=False, empty=False):
    calls=[];initial_before_nomination=[];downstream=[]
    real_fit=nw.classify_family_profiles
    def nominate(s,*args):
        return ([] if empty else [dict(family=s['dataset_id']+':small',consensus_start=12,consensus_end=27)]),{}
    def fit(s,cat,**kw):
        result=real_fit(s,cat,**kw)
        calls.append(dict(dataset=s['dataset_id'],catalog=deepcopy(cat),result=deepcopy(result),kwargs=kw))
        return result
    def augment(cat,native,positions,*,dataset_id,region):
        target=output/digest(dataset_id)[:16]
        initial_before_nomination.append((target/'native_initial_binding.json').is_file())
        additions=[]
        if add:
            call=next(c for c in native['calls'] if c['end']-c['start']>30)
            additions=[dict(family=dataset_id+':RN_broad',consensus_start=call['start'],consensus_end=call['end'],
                nomination_provenance='bounded_test_nomination_from_actual_original_call',
                source_aliases=[dict(unit_id=call['unit_id'],ordinal=call['ordinal'],interval=[call['start'],call['end']])])]
        if corrupt:
            score=next(s for row in native['call_family_evidence'] for s in row if s['status']=='scored')
            score['geometry_distance_sq']+=.25
        return [*cat,*additions],dict(initial_proposals=len(cat),added_proposals=len(additions),additions=additions)
    def auxiliary(s,cat,native,*args):
        downstream.append(('auxiliary',s['dataset_id'],native))
        return dict(rescue=dict(status='disabled'),split=dict(status='disabled'))
    monkeypatch.setattr(nw,'nominate_catalog',nominate)
    monkeypatch.setattr(nw,'classify_family_profiles',fit)
    monkeypatch.setattr(nw,'augment_catalog',augment)
    monkeypatch.setattr(nw,'run_native_auxiliary',auxiliary)
    return calls,initial_before_nomination,downstream


def test_policy_schema_default_and_native_only_validation():
    assert parse_options()['cr'].residual_update_policy=='refit_catalog'
    field=next(f for f in parameter_schema()['cr'] if f['name']=='residual_update_policy')
    assert field['default']=='refit_catalog' and field['choices']==['refit_catalog','append_frozen']
    for cr in [dict(residual_update_policy='append'),
               dict(engine='legacy_lattice',residual_update_policy='append_frozen')]:
        with pytest.raises(ValueError):parse_options(dict(cr=cr))


def test_append_binds_before_nomination_and_all_downstream_get_composed_result(tmp_path,monkeypatch):
    data=source(two=True);before=deepcopy(data)
    fits,bound,downstream=install(monkeypatch,tmp_path)
    graph_inputs=[]
    def graph(inputs,**kwargs):
        graph_inputs.append(inputs)
        return dict(status='complete',links=[],nodes=[])
    monkeypatch.setattr(nw,'reciprocal_native_graph',graph)
    monkeypatch.setattr(nw,'summarize_native_correspondences',lambda *args:[])
    opts=options();opts['cross']=dict(enabled=True)
    result=run_workflow(data,opts,tmp_path)
    assert data==before and bound==[True,True]
    assert len(fits)==4 and len(graph_inputs)==1
    for ds,shown in result['datasets'].items():
        target=tmp_path/digest(ds)[:16]
        initial=read_json(target/'native_family_initial.json.gz')
        augmented=read_json(target/'native_family_augmented.json.gz')
        final=read_json(target/'native_family_model.json.gz')
        ib=read_json(target/'native_initial_binding.json');ab=read_json(target/'native_augmented_binding.json')
        assert 'nomination_update' not in initial and augmented['nomination_update']['added_proposals']==1
        assert ab['nomination_parent_digest']==ib['snapshot_digest']
        assert ib['model_options']['core_contradiction_odds']==100
        assert ib['model_options']['loss_odds_levels']==[10.,100.,1000.]
        assert ib['model_options']['max_fit_iterations']==25
        assert ib['implementation_contract']==result['manifest']['implementation_sha256']
        assert final['family_models'][0]==initial['family_models'][0]
        new=next(m for m in augmented['family_models'] if m['family']==ds+':RN_broad')
        assert final['family_models'][-1]==new
        for old,refit,current in zip(initial['call_family_evidence'],augmented['call_family_evidence'],final['call_family_evidence']):
            assert current==[*old,*(s for s in refit if s['family']==ds+':RN_broad')]
        assert graph_inputs[0][ds]['result']==final
        assert next(n for kind,name,n in downstream if name==ds)==final
        assert shown['cr']['native_catalog_update']['policy']=='append_frozen'
        assert shown['cr']['native_catalog_update']['added_family_count']==1
        origins={f['family']:f['model_provenance']['origin'] for f in shown['cr']['catalog']}
        assert origins=={ds+':small':'initial_frozen',ds+':RN_broad':'augmented_new'}
        for family in shown['cr']['catalog']:
            assert len(family['model_provenance']['model_version_sha256'])==64
        assert final['partitions']=={}
    assert result['manifest']['native_fit_options']['d']['max_fit_iterations']==25


@pytest.mark.parametrize('residual',[True,False])
def test_no_update_identity_and_no_extra_fit(tmp_path,monkeypatch,residual):
    fits,bound,_=install(monkeypatch,tmp_path,add=False)
    result=run_workflow(source(),options(residual_nomination=residual),tmp_path)
    target=tmp_path/digest('d')[:16]
    assert len(fits)==1
    assert read_json(target/'native_family_model.json.gz')==read_json(target/'native_family_initial.json.gz')
    assert read_json(target/'native_composed_binding.json')==read_json(target/'native_initial_binding.json')
    assert result['datasets']['d']['cr']['native_catalog_update']['no_expansion']
    assert bound==([True] if residual else [])


def test_refit_catalog_remains_default_and_uses_augmented_fit(tmp_path,monkeypatch):
    fits,bound,_=install(monkeypatch,tmp_path)
    result=run_workflow(source(),options(None),tmp_path)
    assert len(fits)==2 and bound==[False]
    target=tmp_path/digest('d')[:16]
    final=read_json(target/'native_family_model.json.gz')
    expected=deepcopy(fits[-1]['result']);expected['nomination_update']=final['nomination_update']
    assert final==expected
    assert not (target/'native_initial_binding.json').exists()
    assert result['manifest']['parameters']['cr']['residual_update_policy']=='refit_catalog'
    assert result['datasets']['d']['cr']['native_catalog_update']==dict(policy='refit_catalog',original_family_count=1,added_family_count=1)
    assert all('model_provenance' not in f for f in result['datasets']['d']['cr']['catalog'])


def test_empty_catalog_is_bound_no_op_with_all_calls_unresolved(tmp_path,monkeypatch):
    fits,bound,_=install(monkeypatch,tmp_path,empty=True)
    result=run_workflow(source(),options(),tmp_path)
    cr=result['datasets']['d']['cr'];target=tmp_path/digest('d')[:16]
    assert fits==[] and bound==[] and cr['catalog']==[]
    assert cr['status']=='no_testable_family' and cr['native_catalog_update']['no_expansion']
    assert sum(len(r['proposals']) for r in cr['records'])==6
    assert all(p['primary_evidence'] is None for r in cr['records'] for p in r['proposals'])
    assert read_json(target/'native_family_model.json.gz')==read_json(target/'native_family_initial.json.gz')


def test_mutating_initial_after_producer_binding_rejects_stale_dependency(tmp_path,monkeypatch):
    install(monkeypatch,tmp_path,corrupt=True)
    with pytest.raises(ValueError,match='changed after its snapshot'):
        run_workflow(source(),options(),tmp_path)
    assert not (tmp_path/'result.json.gz').exists()


def test_receipt_implementation_drift_is_rejected():
    with pytest.raises(ValueError,match='implementation changed'):
        nw._check_implementation({'native_workflow.py':'0'*64})


@pytest.mark.parametrize('policy',['refit_catalog','append_frozen'])
def test_empty_selected_dataset_is_retained_in_real_xcr_assessment(tmp_path,monkeypatch,policy):
    fits,_,_=install(monkeypatch,tmp_path,add=False)
    monkeypatch.setattr(nw,'nominate_catalog',lambda s,*args:(
        [dict(family='d:small',consensus_start=12,consensus_end=27)] if s['dataset_id']=='d' else [],{}))
    opts=options(policy,residual_nomination=False);opts['cross']=dict(enabled=True)
    result=run_workflow(source(two=True),opts,tmp_path)
    cross=result['cross']
    assert cross['selected_dataset_ids']==['d','h']
    assert cross['assessment_status']=='incomplete_no_native_models'
    assert cross['dataset_assessments']['h']==dict(fitted_native_models=0,supplied_evidence_units=6,
        native_cr_status='no_testable_family',status='no_fitted_native_models')
    assert cross['dataset_assessments']['d']['status']=='native_models_available'
    assert cross['edges']==[] and cross['comparable_edges']==0 and cross['count_groups']==[]
    assert len(fits)==1
    assert sum(len(r['proposals']) for r in result['datasets']['h']['cr']['records'])==6
