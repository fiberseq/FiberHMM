"""Small exact regressions for cross-stage evidence/counting seams."""
from copy import deepcopy
from types import SimpleNamespace
import numpy as np
import pytest

from fiberhmm.inference.consensus.lattice import RegionFamilyLattice
from fiberhmm.inference.consensus.cross_evidence import prepare_shape_data,shape_predictive,direction
from fiberhmm.inference.consensus.cross_quantification import summarize_correspondences
from fiberhmm.inference.consensus.stages import assign_cr


def test_exact_any_event_does_not_sum_cooccurring_marginals():
    k=RegionFamilyLattice(np.arange(30),[[2,7],[15,20]],0)
    values=np.zeros((1,30));eta=np.zeros(2)
    marg=k.evaluate(values,eta)['family_inclusion']
    np.testing.assert_allclose(marg,[[.5,.5]])
    np.testing.assert_allclose(k.any_family_inclusion(values,eta,[0,1]),[.75])
    np.testing.assert_allclose(k.any_family_inclusion(values,eta,[0]),marg[:,0])
    np.testing.assert_allclose(k.any_family_inclusion(values,eta,[]),[0])
    with pytest.raises(ValueError):k.any_family_inclusion(values,eta,[2])


def test_exact_any_event_respects_same_geometry_mask_and_prior_weights():
    k=RegionFamilyLattice(np.arange(30),[[2,7],[15,20]],0)
    values=np.zeros((2,30));eta=np.log([2.,3.]);mask=np.array([[1,0],[0,0]],bool)
    np.testing.assert_allclose(k.any_family_inclusion(values,eta,[0,1],geometry_allowed=mask),[2/3,0])


def event_run(prefix,classes):
    centers=np.array([[5,15]]*classes);k=RegionFamilyLattice(np.arange(30),centers,0)
    values=np.full((4,30),-2.);values[:,5:15]=2.
    # Splitting one state's prior weight among exactly duplicate labels must
    # not alter the protected-event distribution or its count.
    eta=np.full(classes,-np.log(classes));allowed=np.ones((4,classes),bool)
    mass=k.evaluate(values,eta,allowed=allowed)['family_inclusion']
    proposal=np.zeros_like(mass);proposal[:,0]=mass[:,0]
    return dict(kernel=k,data=dict(log_lr=values),eta=eta,allowed=allowed,native_allowed=allowed.copy(),
        catalog=[dict(family=f'{prefix}{i}',family_index=i,consensus_start=5,consensus_end=15) for i in range(classes)],
        eligible=np.ones_like(allowed),core_eligible=np.ones_like(allowed),
        membership=mass,proposal_membership=proposal,native_proposal_membership=proposal.copy())


def test_coarse_counts_survive_splitting_probability_among_three_exclusive_labels():
    runs={'D':event_run('D',1),'H':event_run('H',3)}
    links=[dict(edge_id=f'e{i}',left_dataset='D',right_dataset='H',left_family='D0',right_family=f'H{i}',comparability_mask=True) for i in range(3)]
    g=summarize_correspondences(links,runs)[0][0]
    assert g['left_counts']['assigned_units']==g['right_counts']['assigned_units']==4
    assert g['right_counts']['per_member_threshold_assigned_units']==0
    assert g['right_counts']['group_event_mass_computed']
    assert g['right_counts']['native_assigned_units']==4
    assert g['right_counts']['multiple_member_units']==0


def test_a_qualifying_coarse_event_still_needs_a_physical_selected_action():
    runs={'D':event_run('D',1),'H':event_run('H',1)}
    runs['H']['proposal_membership'][:]=0
    link=dict(edge_id='e',left_dataset='D',right_dataset='H',left_family='D0',right_family='H0',comparability_mask=True)
    g=summarize_correspondences([link],runs)[0][0]
    assert g['right_counts']['assigned_units']==0
    assert g['right_counts']['group_mass_passing_without_selected_action']==4


def test_unmatched_recurrent_sibling_is_explicit_not_an_unambiguous_count_match():
    runs={'D':event_run('D',2),'H':event_run('H',1)}
    runs['D']['proposal_membership'][:]=.8
    link=dict(edge_id='e',left_dataset='D',right_dataset='H',left_family='D0',right_family='H0',comparability_mask=True)
    groups,ann=summarize_correspondences([link],runs)
    assert groups[0]['unrepresented_recurrent_overlap']
    assert groups[0]['omitted_overlapping_native_families']['D'][0]['family']=='D1'
    assert not ann['e']['individual_count_comparison_unambiguous']
    assert groups[0]['kind']=='one_to_one' # graph topology remains inspectable


def shape_data(msps):
    return prepare_shape_data(dict(units=[dict(msp_intervals=msps,aligned_blocks=[[0,30]],raw_nuc_intervals=[])],
        grid_positions=np.arange(30),log_lr=np.full((1,30),2.),observed=np.ones((1,30),bool),hits=np.zeros((1,30))),0,30)


@pytest.mark.parametrize('msps',[[[0,10],[10,30]],[[0,12],[10,30]]])
def test_adjacent_and_overlapping_msps_do_not_abort_or_union_native_domains(msps):
    d=shape_data(msps)
    assert shape_predictive(d,[2,7],0)['physical_prior_mass'][0]==1
    assert shape_predictive(d,[8,15],0)['physical_prior_mass'][0]==0
    assert shape_predictive(d,[12,17],0)['physical_prior_mass'][0]==1


def test_configurable_opportunity_floor_is_used_by_cross_testability():
    d=shape_data([[0,30]])
    three=shape_predictive(d,[10,13],0,minimum_opportunities=3)
    four=shape_predictive(d,[10,13],0,minimum_opportunities=4)
    assert three['positive_core_posterior_mass'][0]==1
    assert four['positive_core_posterior_mass'][0]==0
    assert four['three_opportunity_prior_mass'][0]==1 # legacy diagnostic remains literal
    assert four['minimum_opportunity_prior_mass'][0]==0
    assert direction(three,four,np.ones(1))['untestable_mass_fraction']==1


def test_canonical_core_eligibility_cannot_be_borrowed_from_wide_edge_variant():
    pos=np.array([10,12,14,16,28]);k=RegionFamilyLattice(pos,[[18,20]],10)
    u=dict(unit_id='u',strand='CT',representative_raw_tf_intervals=[[10,17]],
           aligned_blocks=[[0,40]],msp_intervals=[[0,40]],raw_nuc_intervals=[])
    data=dict(units=[u],dataset_id='D',log_lr=np.array([[2.,2.,2.,2.,-2.]]),
        observed=np.ones((1,5),bool),hits=np.array([[0,0,0,0,1]]),family_ids=['D0'])
    result=assign_cr(data,k,np.zeros(1),SimpleNamespace(minimum_opportunities=3),
        SimpleNamespace(batch_size=4,maximum_matrix_mb=8),lambda *_:None)
    assert result['eligible'][0,0]
    assert not result['core_eligible'][0,0]
    assert result['core_opportunities'][0,0]==0
    assert result['records'][0]['proposals']
    p=result['records'][0]['proposals'][0]
    assert p['opportunities']>=3 and p['canonical_core_opportunities']==0
    assert not p['canonical_core_eligible']
def test_physical_integer_alias_exposure_matches_exhaustive_without_joining_msps():
    from fiberhmm.inference.consensus.context import physical_alias_exposure
    from fiberhmm.inference.consensus.recall import projection_rectangles
    from fiberhmm.inference.consensus.geometry import physically_allowed
    from fiberhmm.inference.consensus.lattice import RegionFamilyLattice
    k=RegionFamilyLattice(np.array([1,3,5,8,11,15,19]),[[3,8],[8,15]],3)
    rects=projection_rectangles(k)
    u=dict(_region=[0,22],aligned_blocks=[[0,10],[10,17],[18,22]],
        msp_intervals=[[0,8],[8,12],[9,22]],raw_nuc_intervals=[[13,14]])
    fraction,coords=physical_alias_exposure(u,k,rects)
    for g,(sl,sh,el,eh) in enumerate(rects):
        intervals=np.array([(a,b) for a in range(sl,sh+1) for b in range(el,eh+1)])
        assert np.all(intervals[:,0]<intervals[:,1])
        permitted=physically_allowed(u,intervals,np.ones(len(intervals)),1)
        np.testing.assert_allclose(fraction[g],permitted.mean())
        if fraction[g]:
            assert list(coords[g]) in intervals[permitted].tolist()
            np.testing.assert_array_equal(np.searchsorted(k.positions,coords[g]),[k.ga[g],k.gb[g]])


def test_coverage_event_integrates_integer_aliases_exactly():
    from fiberhmm.inference.consensus.context import coverage_alias_exposure
    from fiberhmm.inference.consensus.recall import projection_rectangles
    from fiberhmm.inference.consensus.geometry import physically_allowed
    from fiberhmm.inference.consensus.lattice import RegionFamilyLattice
    k=RegionFamilyLattice(np.array([1,5,11,15,19]),[[3,13]],4);rects=projection_rectangles(k)
    u=dict(_region=[0,23],aligned_blocks=[[0,23]],msp_intervals=[[0,9],[9,23]],raw_nuc_intervals=[])
    frac,event=coverage_alias_exposure(u,k,rects,[7,14],4)
    for g,(sl,sh,el,eh) in enumerate(rects):
        spans=np.array([(a,b) for a in range(sl,sh+1) for b in range(el,eh+1)])
        valid=physically_allowed(u,spans,np.ones(len(spans)),1)
        predicate=np.maximum(0,np.minimum(spans[:,1],14)-np.maximum(spans[:,0],7))>=4
        np.testing.assert_allclose(frac[g],valid.mean())
        np.testing.assert_allclose(event[g],predicate[valid].mean() if valid.any() else 0)
