from copy import deepcopy
import numpy as np
import pytest
from fiberhmm.inference.consensus.parameters import parse_options,parameter_schema
from fiberhmm.inference.consensus.workflow import run_workflow,ConsensusCancelled


def payload():
    strata=[]
    for dataset,chemistry in [('daf','ddda'),('fiber','hia5-pacbio')]:
        units=[];positions=list(range(100,200,2))
        for i in range(24):
            a,b=(120,142) if i%3 else (156,180)
            units.append(dict(unit_id=f'{dataset}{i}',read_name=f'{dataset}_read{i}',fold_group_id=f'{dataset}{i}',
                strand=('CT' if i%2 else 'GA') if dataset=='daf' else ('+' if i%2 else '-'),
                positions=positions,contexts=[0]*len(positions),hits=[int(not a<=p<b) for p in positions],
                p_accessible=[.8]*len(positions),p_protected=[.1]*len(positions),
                representative_raw_tf_intervals=[[a,b]],raw_nuc_intervals=[],msp_intervals=[[100,200]],
                aligned_blocks=[[100,200]],reference_start=100,reference_end=200))
        strata.append(dict(dataset_id=dataset,stratum_id=dataset,chemistry=chemistry,units=units))
    return dict(region=dict(chrom='chr1',start=100,end=200),strata=strata)


def test_parameter_contract_rejects_typos_nonfinite_and_hidden_caps():
    assert set(parameter_schema())=={'input','sr','cr','cross','rescue','comparability','split','compute','families','recaller'}
    for bad in [{'cr':{'max_families':3}},{'cr':{'ambiguity_bp':1.2}},{'rescue':{'prior_scale':float('nan')}},
                {'sr':{'enabled':'false'}},{'cross':{'enabled':True},'cr':{'enabled':False}}]:
        with pytest.raises(ValueError):parse_options(bad)
    assert not any('reads' in f['name'] or 'max_families'==f['name'] for g in parameter_schema().values() for f in g)


def test_full_workflow_mixed_assays_and_immutable_native_source(tmp_path):
    source=payload();before=deepcopy(source)
    result=run_workflow(source,{'input':{'correct_native':False},
        'cr':{'engine':'legacy_lattice','ambiguity_bp':2,'local_iterations':10,'global_iterations':20},
        'cross':{'enabled':True},'rescue':{'enabled':True},
        'comparability':{'enabled':True,'quadrature_size':128,'simulation_units':0,'minimum_population_units':2},
        'split':{'enabled':True}},tmp_path)
    assert source==before
    assert result['datasets']['daf']['sr']['status']=='complete'
    assert result['datasets']['fiber']['sr']['status']=='not_applicable'
    for ds in result['datasets'].values():
        assert len(ds['units'])==24
        assert len(ds['cr']['catalog'])==2
        assert sum(len(r['proposals']) for r in ds['cr']['records'])==24
    assert result['cross']['comparable_edges']==2
    assert all('merged_interval' not in e for e in result['cross']['edges'])
    groups=result['cross']['count_groups']
    assert len(groups)==2 and all(g['kind']=='one_to_one' for g in groups)
    assert sorted(g['left_counts']['assigned_units'] for g in groups)==[8,16]
    for g in groups:
        assert not g['rescue_counts_used']
        assert not g['quantification_equivalence_established']
        assert g['left_counts']['assigned_units']==g['right_counts']['assigned_units']
        assert g['left_counts']['native_assigned_units']==g['left_counts']['assigned_units']
        assert g['left_counts']['eligible_units']==g['right_counts']['eligible_units']==24
        assert 'merged_interval' not in g
    assert result['datasets']['fiber']['rescue']['status']=='not_applicable'
    assert result['manifest']['read_sample_cap'] is None
    assert (tmp_path/'result.json.gz').is_file()


def test_cancellation_does_not_emit_complete_result(tmp_path):
    def stop(stage,message):raise ConsensusCancelled('cancel')
    with pytest.raises(ConsensusCancelled):run_workflow(payload(),{'input':{'correct_native':False}},tmp_path,stop)
    assert not (tmp_path/'result.json.gz').exists()


def test_requires_actual_query_replay_never_fabricates_it(tmp_path):
    with pytest.raises(ValueError,match='actual-query replay'):run_workflow(payload(),{},tmp_path)


def test_joint_resource_limit_is_an_unresolved_edge_not_loss_of_all_native_layers(tmp_path,monkeypatch):
    from fiberhmm.inference.consensus import workflow
    def limited(*_):raise MemoryError('Exact frontier budget exceeded: test fixture')
    monkeypatch.setattr(workflow,'make_kernel',limited)
    result=run_workflow(payload(),{'input':{'correct_native':False},
        'cr':{'engine':'legacy_lattice','ambiguity_bp':2,'local_iterations':10,'global_iterations':20},
        'cross':{'enabled':True,'joint_validation':'all'}},tmp_path)
    assert result['manifest']['status']=='complete'
    assert result['cross']['comparable_edges']==0
    assert result['cross']['edges']
    for edge in result['cross']['edges']:
        assert edge['joint_left']['status']=='resource_limited'
        assert edge['joint_left']['retained_native_mass_fraction'] is None
    assert len(result['datasets']['daf']['cr']['catalog'])==2
