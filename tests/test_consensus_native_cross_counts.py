from copy import deepcopy
import pytest
from fiberhmm.inference.consensus.native_cross_counts import maximal_rectangles,count_native_event,covers


def test_incomplete_chain_never_becomes_one_cross_site():
    assert maximal_rectangles([('a','x'),('b','x'),('b','y'),('c','y')])==[
        (('a','b'),('x',)),(('b',),('x','y')),(('b','c'),('y',))]
    assert maximal_rectangles([('a','x'),('a','y'),('b','x'),('b','y')])==[(('a','b'),('x','y'))]


def test_concept_budget_fails_instead_of_silently_dropping_classes():
    with pytest.raises(MemoryError,match='no groups silently dropped'):
        maximal_rectangles([('a','x'),('a','y'),('b','y'),('b','z')],maximum_concepts=1)


def test_actual_alignment_blocks_define_denominator():
    assert covers([(0,10),(10,20)],(3,18))
    assert not covers([(0,10),(12,20)],(3,18))


def test_unique_events_expose_alternative_vs_cooccurring_fragment_counts():
    units=[dict(unit_id='u',fold_group_id='g',strand='CT',positions=[2,4,8],aligned_blocks=[[0,20]]),
           dict(unit_id='copy',fold_group_id='g',strand='CT',positions=[2,4,8],aligned_blocks=[[0,20]]),
           dict(unit_id='v',strand='GA',positions=[2,4,8],aligned_blocks=[[0,20]]),
           dict(unit_id='gap',strand='GA',positions=[2,8],aligned_blocks=[[0,3],[7,20]])]
    calls=[dict(unit_id='u',evidence_group_id='g',ordinal=0),dict(unit_id='u',evidence_group_id='g',ordinal=1),
           dict(unit_id='copy',evidence_group_id='g',ordinal=0),dict(unit_id='v',ordinal=0)]
    scores=[[dict(family=f,status='scored',predictive_tail_interval=[.5,1],geometry_distance_sq=0,floor_adjusted_loss=0)]
            for f in ['a','b','a','a']]
    data=dict(chemistry='ddda',units=units,result=dict(calls=calls,call_family_evidence=scores))
    before=deepcopy(data);out=count_native_event(data,['a','b'],(1,10))
    assert data==before
    assert out['eligible_units']==2 and out['assigned_units']==2 and out['original_member_calls']==4
    assert out['all_member_units']==1 and out['multiple_member_units']==1
    assert out['by_strand']['CT']['assigned_units']==1 and out['by_strand']['GA']['assigned_units']==1
    assert out['repeated_groups_collapsed']==1
    data['chemistry']='hia5-pacbio'
    assert list(count_native_event(data,['a','b'],(1,10))['by_strand'])==['pooled']
