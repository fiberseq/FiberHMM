from copy import deepcopy
import gzip
import json

import numpy as np
import pytest

from fiberhmm.inference.consensus.call_clustering import prepare
from fiberhmm.inference.consensus.call_clustering.comparison import (
    ComparisonOptions, coarse_events, native_counts, compare_counts, comparison_rows,
)
from fiberhmm.inference.consensus.parameters import parse_options
from fiberhmm.inference.consensus.workflow import run_workflow, ConsensusCancelled
from test_consensus_call_clustering import unit, payload


def source():
    p = payload([unit('a', calls=[(20,30)]), unit('b','GA',calls=[(20,30)]),
                 unit('c',calls=[(20,30)]), unit('absent','GA',hits=(0,0,0,1))])
    p['strata'][0]['chemistry'] = 'ddda'
    p['strata'].append(dict(dataset_id='h', chemistry='hia5-pacbio',
        model_manifest=dict(native_minimum_llr=5.), units=[unit('h1','FWD',calls=[(20,30)]),
            unit('h2','REV',calls=[(20,30)]), unit('h3','FWD',calls=[(20,30)])]))
    return p


def family(name, span, members=()):
    return dict(family_id=name, interval=list(span), members=[list(m) for m in members], established=True)


def test_current_default_is_all_msp_native_harmonization():
    options = parse_options()
    assert options['input'].minimum_nfr_length == 0
    assert options['cr'].engine == 'call_harmonization'
    assert not options['rescue'].enabled and not options['split'].enabled


def test_harmonization_export_keeps_every_native_call_and_never_rescues(tmp_path):
    p=source();before=deepcopy(p)
    result=run_workflow(p, {'cross':{'enabled':True}}, tmp_path/'new'/'run')
    assert p==before
    assert result['cr_mode']=='call_harmonization'
    assert result['manifest']['native_source_modified'] is False
    assert result['manifest']['read_sample_cap'] is None
    assert result['manifest']['rescue'] is False
    for s in p['strata']:
        ds=result['datasets'][s['dataset_id']]
        assert ds['rescue']['status']==ds['split']['status']=='disabled'
        for u,row,sr in zip(s['units'],ds['cr']['records'],ds['sr']['records']):
            assert len(row['proposals'])==len(u['native_multi_interval_calls'])
            for i,(call,original) in enumerate(zip(row['proposals'],u['native_multi_interval_calls'])):
                assert call['interval']==original['interval']==call['source_interval']
                assert call['source_ordinal']==i and not call['new_call']
                assert 'model_membership' not in call and 'predictive_tail' not in call
            for call in sr['calls']:
                assert np.array_equal(np.searchsorted(u['positions'],call['raw']),
                                      np.searchsorted(u['positions'],call['interval']))
    assert result['datasets']['d']['cr']['records'][-1]['proposals']==[]
    assert {'native_class','coarse_any_child'}=={r['resolution'] for r in result['comparison']['rows']}
    assert all(e['rate_agreement_used_for_selection'] is False for e in result['cross']['edges'])
    with gzip.open(tmp_path/'new'/'run'/'result.json.gz','rt') as f:
        assert json.load(f)==result


def test_coarse_counts_are_any_child_not_sum_and_lattice_ignores_outcomes_and_nuc_mask():
    a=unit('a',calls=[(20,24),(26,30)])
    b=unit('b','GA',hits=(1,1,1,1));b['raw_nuc_intervals']=[[20,30]];b['msp_intervals']=[]
    gap=unit('gap',calls=[(20,24)]);gap['aligned_blocks']=[[0,25],[26,200]]
    reads,floors,_=prepare([payload([a,b,gap])])
    fs=[family('wide',[20,30],[('a',0),('a',1),('gap',0)])]
    count=native_counts(reads,fs,floors)[0]['by_dataset']['d']
    assert count['covered_units']==2 and count['assigned_units']==1
    assert count['fraction']==.5 and count['original_member_calls']==2
    assert count['multiple_member_units']==1
    assert count['lattice_capable_units']==2 and count['lattice_capable_fraction']==1
    assert count['physically_accessible_units']==1
    b['hits']=[0]*4;b['raw_nuc_intervals']=[];b['msp_intervals']=[[0,200]]
    changed,_,_=prepare([payload([a,b,gap])])
    next_count=native_counts(changed,fs,floors)[0]['by_dataset']['d']
    for key in ('covered_units','assigned_units','fraction','lattice_capable_units'):
        assert next_count[key]==count[key]
    assert next_count['physically_accessible_units']==2


def test_coarse_comparison_uses_direct_bounded_anchor_not_transitive_chains():
    fs=[family('a',[0,10]),family('b',[3,13]),family('c',[6,16])]
    result=coarse_events({'SR d':fs})
    assert [f['children']['d'] for f in result]==[['a','b'],['c']]
    assert coarse_events({'SR d':list(reversed(fs))})==result
    for f in fs:f['calls']=10000 if f['family_id']=='c' else 1
    assert coarse_events({'SR d':fs})==result  # No rate/count-agreement selection.
    assert all(not f['transitive_union'] and not f['native_calls_modified'] for f in result)


def test_coarse_comparison_preserves_four_fragments_below_a_broad_assay_class():
    fs=[family('broad',[10,50],[('h',0)])]+[
        family(f'f{i}',[10+i*10,20+i*10],[('d',i)]) for i in range(4)]
    result=coarse_events({'SR common':fs})
    assert len(result)==1 and len(result[0]['children']['common'])==5
    assert result[0]['members']==[['d',0],['d',1],['d',2],['d',3],['h',0]]
    assert result[0]['kind']=='coarse_any_native_class'
    assert fs[1]['interval']==[10,20]


def test_comparability_and_observed_agreement_are_independent():
    left=dict(covered_units=100,assigned_units=20,fraction=.2,lattice_capable_fraction=1.)
    right=dict(left,assigned_units=60,fraction=.6)
    row=compare_counts(left,right)
    assert row['comparability']=='lattice_comparable'
    assert row['agreement']=='discordant_observed_fractions'
    right.update(assigned_units=20,fraction=.2,lattice_capable_fraction=.1)
    row=compare_counts(left,right)
    assert row['comparability']=='not_comparable' and row['agreement']=='similar_observed_fractions'
    assert row['reasons']==['right_lattice_limited']
    assert compare_counts(left,dict(right,assigned_units=0,fraction=0))['agreement']=='low_detection_counts'
    assert compare_counts(left,dict(right,covered_units=0,fraction=None))['agreement']=='unavailable'


def test_coarse_scale_does_not_lose_low_depth_assay_width_to_a_pooled_median():
    p=payload([unit(f'd{i}',calls=[(20,24),(30,34)]) for i in range(30)])
    p['strata'].append(dict(dataset_id='h',model_manifest=dict(native_minimum_llr=5.),
        units=[unit(f'h{i}',calls=[(19,35)]) for i in range(3)]))
    reads,_,_=prepare([p])
    a=family('a',[20,24],[(f'd{i}',0) for i in range(30)]+[(f'h{i}',0) for i in range(3)])
    b=family('b',[30,34],[(f'd{i}',1) for i in range(30)])
    before=deepcopy([a,b])
    result=coarse_events({'SR common':[a,b]},reads=reads)
    assert len(result)==1 and result[0]['interval']==[19,35]
    assert result[0]['anchor_assay']=='h'
    assert result[0]['children']['common']==['a','b']
    assert [a,b]==before  # No assignment is moved between fine classes.


def test_workflow_combines_strata_without_overwriting_units(tmp_path):
    p=source();s=p['strata'][0];p['strata'].append(dict(s,units=s['units'][2:]));s['units']=s['units'][:2]
    result=run_workflow(p, output_dir=tmp_path)
    assert len(result['datasets']['d']['units'])==4
    assert len(result['datasets']['d']['cr']['records'])==4


@pytest.mark.parametrize('stage',['rescue','split'])
def test_new_algorithm_rejects_detection_auxiliaries(stage,tmp_path):
    with pytest.raises(ValueError,match='does not add or split'):
        run_workflow(source(),{stage:{'enabled':True}},tmp_path)
    assert not (tmp_path/'result.json.gz').exists()


def test_harmonization_cancellation_and_region_budget_fail_without_complete_artifact(tmp_path):
    def stop(*_):raise ConsensusCancelled('cancel')
    with pytest.raises(ConsensusCancelled):run_workflow(source(),output_dir=tmp_path,progress=stop)
    with pytest.raises(ValueError,match='maximum analysis span'):
        run_workflow(source(),{'compute':{'maximum_region_bp':100}},tmp_path)
    assert not (tmp_path/'result.json.gz').exists()


def test_raw_call_mode_is_explicit_and_zero_native_llr_is_supported(tmp_path):
    p=source()
    for s in p['strata']:
        s['model_manifest']['native_minimum_llr']=0
        for u in s['units']:u['representative_raw_tf_intervals']=[c['interval'] for c in u.pop('native_multi_interval_calls')]
    result=run_workflow(p,{'input':{'correct_native':False}},tmp_path)
    assert result['manifest']['parameters']['input']['correct_native'] is False
    assert len(result['datasets']['d']['cr']['records'][0]['proposals'])==1


def test_frozen_evidence_cannot_pretend_to_change_native_llr_or_search_domain(tmp_path):
    with pytest.raises(ValueError,match='Reprepare from BAM'):
        run_workflow(source(),{'input':{'ddda_minimum_llr':2}},tmp_path)
    p=source();p['strata'][0]['model_manifest']['native_minimum_msp_bp']=150
    with pytest.raises(ValueError,match='reprepare from BAM'):
        run_workflow(p,output_dir=tmp_path)
