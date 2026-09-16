import math

import numpy as np

from fiberhmm.inference.geometry_uncertainty import (
    UncertaintyGeometry as G, information_lattice, group_uncertain_geometries,
    geometry_conflict_classes,
)


def test_information_uses_model_and_opportunities_not_hit_counts():
    a={"positions":[1,4],"p_accessible":[.9,.8],"p_protected":[.1,.2],"hits":[0,0]}
    b={**a,"hits":[1,1]}
    pa,ia=information_lattice([a]); pb,ib=information_lattice([b])
    np.testing.assert_array_equal(pa,pb)
    np.testing.assert_allclose(ia,ib)
    np.testing.assert_allclose(ia[1],-math.log(.6),rtol=0,atol=1e-14)


def test_minor_edges_group_but_informative_extension_remains_separate():
    positions=np.arange(30)
    prefix=np.arange(31)*.5
    geometries=[G("short1",5,12),G("short2",5,13),G("long1",5,23),G("long2",4,23)]
    evidence=np.array([[12,11,-10,-11],[-2,0,25,26]])
    groups=group_uncertain_geometries(geometries,positions,prefix,evidence)
    assert {frozenset(g["geometry_indices"]) for g in groups} == {frozenset([0,1]),frozenset([2,3])}


def test_actual_two_way_evidence_can_override_theoretical_group_radius():
    gs=[G("a",0,10),G("b",0,11)]
    # A deliberately weak-distance stratum still cannot merge two observed,
    # strongly distinguishable patterns merely because interval overlap is high.
    groups=group_uncertain_geometries(gs,np.arange(12),np.arange(13)*.01,
                                      [[20,10],[10,20]])
    assert len(groups)==2


def test_complete_link_prevents_chain_merging_and_keeps_all_alias_nodes():
    gs=[G("a",0,4),G("b",1,5),G("c",2,6)]
    groups=group_uncertain_geometries(gs,np.arange(8),np.arange(9)*.5,
                                      [[1,1,1]],distance_limit=1.1)
    assert len(groups)==2
    assert sorted(j for g in groups for j in g["geometry_indices"])==[0,1,2]


def test_no_information_does_not_create_an_equivalence_claim():
    gs=[G("a",0,4),G("b",1,5)]
    groups=group_uncertain_geometries(gs,[],[0.],[[0,0]])
    assert len(groups)==2


def test_mixed_geometry_blocks_do_not_equal_entire_overlap_components():
    gs=[G("as",0,4),G("al",0,8),G("b",6,10),G("c",9,12)]
    groups=[{"geometry_indices":[0,1]},{"geometry_indices":[2]},{"geometry_indices":[3]}]
    graph=geometry_conflict_classes(gs,groups)
    assert graph["always_conflicting"]==[(1,2)]
    assert graph["mixed_pairs"]==[(0,1)]
    assert graph["mixed_blocks"]==[[0,1],[2]]
