from types import SimpleNamespace
from copy import deepcopy
import numpy as np
import pysam
import pytest
from fiberhmm.inference.consensus import adapter
from fiberhmm.inference import strand_rescue,tf_recaller


def model():
    ep=np.zeros((2,8193));ep[0,:4096]=.1/4096;ep[1,:4096]=.9/4096
    ep[0,4097:]=.9/4096;ep[1,4097:]=.1/4096
    return SimpleNamespace(emissionprob_=ep)


def unit(end):
    return dict(msp_intervals=[[100,end]],raw_nuc_intervals=[],aligned_blocks=[[100,end]])


def test_default_preparation_scans_short_msps_without_changing_llr(monkeypatch):
    from fiberhmm.inference.consensus.parameters import InputOptions
    opts = InputOptions()
    assert opts.minimum_nfr_length == 0
    obs = np.full(80, 4097, dtype=np.int32)
    monkeypatch.setattr(strand_rescue, 'hard_observations', lambda *_: (obs, 'CT'))
    monkeypatch.setattr(strand_rescue, 'cigar_to_query_ref', lambda _: np.arange(80)+100)
    args = (object(), unit(180), model(), 'CT', 'daf', 3, None, 7)
    calls = adapter.replay_alignment(*args, minimum_nfr_length=opts.minimum_nfr_length)
    assert len(calls) == 1 and calls[0]['interval'] == [100, 180]
    assert calls[0]['llr'] > 7 and calls[0]['opportunities'] == 80
    assert adapter.replay_alignment(*args, minimum_nfr_length=150) == []


def test_replay_clips_actual_encoder_cigar_domain_before_native_wrapper(monkeypatch):
    obs=np.full(50,4097,dtype=np.int32)
    monkeypatch.setattr(strand_rescue,'hard_observations',lambda *_:(obs,'CT'))
    monkeypatch.setattr(strand_rescue,'cigar_to_query_ref',lambda _:np.arange(70)+100)
    calls=adapter.replay_alignment(object(),unit(180),model(),'CT','daf',3,None,7)
    expected=tf_recaller.call_tfs_in_interval(obs,0,50,*tf_recaller.build_llr_tables(model()),7,3)
    assert [(c['interval'],c['llr']) for c in calls]==[([100+c.start,100+c.start+c.length],c.llr) for c in expected]
    assert all(c['interval'][1]<=150 for c in calls)


def test_m5c_conditioning_matches_native_tables_only_at_tagged_cpgs(monkeypatch):
    monkeypatch.setattr(strand_rescue,'cigar_to_query_ref',lambda _:np.arange(6)+100)
    monkeypatch.setattr(adapter,'m5c_query_mask',lambda *_:np.array([True,True,False,False,False,True]))
    u=dict(positions=list(range(100,106)),hits=[0,1,0,1,0,1],contexts=[48,0,48,0,48,48],p_accessible=[.9]*6,
        p_protected=[.1]*6,provenance={})
    adapter.condition_unit_on_m5c(object(),u)
    assert u['positions']==[101,102,103,104]
    assert u['hits']==[1,0,1,0]
    assert u['contexts']==[0,48,0,48]
    assert u['p_accessible']==[.9]*4
    assert 'm5c_observations' not in u
    assert u['provenance']['native_m5c_excluded_opportunities']==2
    assert u['provenance']['native_emissions_unchanged'] is True


def test_native_replay_honors_m5c_tables(monkeypatch):
    obs=np.full(20,4097+48,dtype=np.int32)
    monkeypatch.setattr(strand_rescue,'hard_observations',lambda *_:(obs,'CT'))
    monkeypatch.setattr(strand_rescue,'cigar_to_query_ref',lambda _:np.arange(20)+100)
    monkeypatch.setattr(adapter,'m5c_query_mask',lambda _,length:np.ones(length,bool))
    regular=tf_recaller.build_llr_tables(model())[1][48]*20
    conditioned=tf_recaller.build_m5c_llr_tables(model())[1][48]*20
    threshold=(regular+conditioned)/2
    assert regular>conditioned
    assert adapter.replay_alignment(object(),unit(120),model(),'CT','daf',3,None,threshold)
    assert not adapter.replay_alignment(object(),unit(120),model(),'CT','daf',3,None,threshold,use_m5c=True)


@pytest.mark.parametrize('reverse',[False,True])
@pytest.mark.parametrize('cigar,expected',[
    ([(0,10),(1,3),(0,20)],[100,130]),
    ([(4,5),(0,10),(1,2),(0,20),(4,5)],[100,130]),
    ([(0,10),(0,20)],[100,130]),
    ([(0,10),(2,2),(0,20)],None),
    ([(0,10),(3,50),(0,20)],None),
])
def test_replay_contiguous_reference_coverage_not_single_cigar_block(monkeypatch,cigar,expected,reverse):
    read=pysam.AlignedSegment()
    length=sum(n for op,n in cigar if op in (0,1,4,7,8))
    read.query_sequence='C'*length
    read.reference_id=0;read.reference_start=100;read.cigartuples=cigar
    read.flag=16 if reverse else 0
    obs=np.full(length,4097,dtype=np.int32)
    monkeypatch.setattr(strand_rescue,'hard_observations',lambda *_:(obs,'CT'))
    u=dict(msp_intervals=[[100,read.reference_end]],raw_nuc_intervals=[],
           aligned_blocks=[list(v) for v in read.get_blocks()])
    before=deepcopy(u)
    calls=adapter.replay_alignment(read,u,model(),'CT','daf',3,None,7)
    assert u==before
    assert [c['interval'] for c in calls]==([] if expected is None else [expected])
    # Actual query opportunities, including insertions, remain in the decoder.
    if expected is not None:
        assert calls[0]['opportunities']==sum(n for op,n in cigar if op in (0,1,7,8))


def test_replay_drops_tfs_overlapping_a_no_call_block(monkeypatch):
    """Production calling drops TF footprints overlapping a long unaligned
    block; native replay must not feed one to the recaller either."""
    read=pysam.AlignedSegment()
    cigar=[(0,50),(1,100),(0,50)]
    read.query_sequence='C'*200
    read.reference_id=0;read.reference_start=1000;read.cigartuples=cigar;read.flag=0
    obs=np.full(200,4097,dtype=np.int32)
    monkeypatch.setattr(strand_rescue,'hard_observations',lambda *_:(obs,'CT'))
    u=dict(msp_intervals=[[1000,read.reference_end]],raw_nuc_intervals=[],
           aligned_blocks=[list(v) for v in read.get_blocks()])
    calls=adapter.replay_alignment(read,u,model(),'CT','daf',3,None,7)
    assert all(c['query_interval'][1]<=50 or c['query_interval'][0]>=150 for c in calls)


def test_replay_keeps_block_tfs_in_circular_mode(monkeypatch):
    """Circular-mode production calling keeps calls inside no-call blocks."""
    read=pysam.AlignedSegment()
    read.query_sequence='C'*200
    read.reference_id=0;read.reference_start=1000;read.cigartuples=[(0,50),(1,100),(0,50)];read.flag=0
    obs=np.full(200,4097,dtype=np.int32)
    monkeypatch.setattr(strand_rescue,'hard_observations',lambda *_:(obs,'CT'))
    u=dict(msp_intervals=[[1000,read.reference_end]],raw_nuc_intervals=[],
           aligned_blocks=[list(v) for v in read.get_blocks()])
    linear=adapter.replay_alignment(read,deepcopy(u),model(),'CT','daf',3,None,7)
    circular=adapter.replay_alignment(read,deepcopy(u),model(),'CT','daf',3,None,7,circular=True)
    assert any(c['query_interval'][0]<150 and c['query_interval'][1]>50 for c in circular)
    assert len(circular)>=len(linear)


def test_called_circular_reads_the_call_command_line():
    def header(cl):
        return pysam.AlignmentHeader.from_dict({'SQ':[{'SN':'p','LN':1000}],
            'PG':[{'ID':'fiberhmm-call','PN':'fiberhmm-call','CL':cl}]})
    assert adapter.called_circular(header('fiberhmm-call -i a.bam -o b.bam -r --enzyme ddda'))
    assert adapter.called_circular(header('fiberhmm-call -i a.bam -o b.bam --circular'))
    assert not adapter.called_circular(header('fiberhmm-call -i a.bam -o b.bam --enzyme ddda'))
