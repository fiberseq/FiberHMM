from copy import deepcopy
from types import SimpleNamespace
import numpy as np
import pysam
import pytest
from fiberhmm.inference.consensus.upstream_recall import recall_hia5_alignment,install_recall
from fiberhmm.inference import strand_rescue
from fiberhmm.inference.consensus.parameters import parse_options
from fiberhmm.inference.consensus.harmonized_families.workflow import prepare_sources


def setup_case(monkeypatch,mode):
    read=pysam.AlignedSegment();read.query_sequence='A'*300
    read.reference_id=0;read.reference_start=100;read.cigarstring='300M'
    read.set_tag('ns',[40]);read.set_tag('nl',[210]);read.set_tag('as',[0,250]);read.set_tag('al',[40,50])
    obs=np.full(300,4096,np.int32);obs[::2]=0
    obs[40:75:2]=4097;obs[96:250:2]=4097
    if mode=='continuous':obs[40:250:2]=4097
    if mode=='blind_gap':obs[76:96:2]=4096
    if mode=='accessible':obs[::2]=0
    monkeypatch.setattr(strand_rescue,'hard_observations',lambda *_:(obs,'FWD'))
    ep=np.zeros((2,8193));ep[0,:4096]=.02;ep[0,4097:]=.98;ep[1,:4096]=.8;ep[1,4097:]=.2
    unit=dict(unit_id='u',_region=[100,400],raw_nuc_intervals=[[140,350]],msp_intervals=[[100,140],[350,400]],
        raw_tf_intervals=[[110,120]],aligned_blocks=[[100,400]])
    return read,unit,SimpleNamespace(emissionprob_=ep)


@pytest.mark.parametrize('mode',['continuous','resolved_gap','blind_gap','accessible'])
def test_native_gap_controls(monkeypatch,mode):
    read,unit,model=setup_case(monkeypatch,mode);before=deepcopy(unit)
    result=recall_hia5_alignment(read,unit,model,'alignment','pacbio-fiber',3,125,5,legacy_annotation_frame='seq')
    released=any(c['interval'][0]<=140 and c['interval'][1]>=173 for c in result['calls'])
    assert released==(mode=='resolved_gap')
    assert unit==before
    install_recall(unit,result)
    assert unit['original_bam_nuc_intervals']==before['raw_nuc_intervals']
    assert unit['original_bam_msp_intervals']==before['msp_intervals']
    assert unit['original_bam_tf_intervals']==before['raw_tf_intervals']
    with pytest.raises(ValueError,match='recursively'):install_recall(unit,result)


def test_scaffold_mismatch_is_not_silently_reinterpreted(monkeypatch):
    read,unit,model=setup_case(monkeypatch,'resolved_gap');unit['raw_nuc_intervals']=[[150,350]]
    with pytest.raises(ValueError,match='scaffold'):
        recall_hia5_alignment(read,unit,model,'alignment','pacbio-fiber',3,125,5,legacy_annotation_frame='seq')


def test_old_cached_hia5_payload_requires_explicit_scaffold_choice():
    from test_staged_families import payload,parameters
    data=payload();data['strata'][0]['chemistry']='hia5-pacbio'
    for u in data['strata'][0]['units']:u['native_multi_interval_tf_intervals']=deepcopy(u['raw_tf_intervals'])
    opts=parameters();opts['input']['correct_native']=True
    with pytest.raises(ValueError,match='reload BAM'):prepare_sources(data,parse_options(opts))
    opts['families']['recall_hia5_nucleosomes']=False
    assert prepare_sources(data,parse_options(opts))


def test_missing_scaffold_does_not_become_an_accessible_read(monkeypatch):
    read,unit,model=setup_case(monkeypatch,'resolved_gap')
    unit.update(raw_nuc_intervals=[],msp_intervals=[])
    result=recall_hia5_alignment(read,unit,model,'alignment','pacbio-fiber',3,125,5)
    assert result['status']=='no_original_scaffold'
    assert result['calls']==result['msps']==result['nucleosomes']==[]


def test_scaffold_mismatch_message_is_a_declared_per_molecule_failure(monkeypatch):
    from fiberhmm.inference.consensus.upstream_recall import MOLECULE_RECALL_FAILURES
    read,unit,model=setup_case(monkeypatch,'resolved_gap');unit['raw_nuc_intervals']=[[150,350]]
    with pytest.raises(ValueError) as error:
        recall_hia5_alignment(read,unit,model,'alignment','pacbio-fiber',3,125,5,legacy_annotation_frame='seq')
    assert str(error.value).startswith(MOLECULE_RECALL_FAILURES)
    # configuration errors stay fatal: they are not per-molecule inconsistencies
    assert not 'Nuc recall needs MA annotations or an explicit legacy tag frame'.startswith(MOLECULE_RECALL_FAILURES)


def test_bad_molecules_are_excluded_with_receipts_but_systematic_failure_raises():
    from fiberhmm.inference.consensus.bam import _exclude_recall_failures
    units=[dict(unit_id=f'u{i:03d}') for i in range(200)]
    bad={id(units[7]):dict(unit_id='u007',read_name='r7',reason='Upstream TF overlaps a final nucleosome')}
    kept,diagnostics=_exclude_recall_failures(units,bad,'hia5',dict(files=[]))
    assert len(kept)==199 and all(u['unit_id']!='u007' for u in kept)
    assert diagnostics['hia5_recall_excluded']==[bad[id(units[7])]] and diagnostics['files']==[]
    assert _exclude_recall_failures(units,{},'hia5',{})==(units,{})
    small=units[:50]
    kept,_=_exclude_recall_failures(small,{id(small[0]):dict(unit_id='u000',read_name='r0',reason='x')},'hia5',{})
    assert len(kept)==49  # one bad read never aborts a small window
    many={id(u):dict(unit_id=u['unit_id'],read_name='r',reason='x') for u in units[:3]}
    with pytest.raises(ValueError,match='3 of 200 molecules'):
        _exclude_recall_failures(units,many,'hia5',{})
