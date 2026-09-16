import numpy as np
import pytest
from scipy.special import logsumexp

from fiberhmm.inference.consensus.measurement_distribution import _density
from fiberhmm.inference.consensus.native_cross import (
    boundary_grid, transfer_density, transferred_call, summarize_direction,reciprocal_native_graph,
    model_overlap, reciprocal_summary_status)


def frozen(positions, domain=(0, 60), reference=(12, 40)):
    grid = boundary_grid(positions, domain)
    fit = dict(parameters=[0., 0., 1., 0., 1.], parameter_reference=list(reference),
               parameter_coordinate_scale_bp=10.)
    return dict(grid=grid, model=dict(family='donor:F1', reference_interval=list(reference),
        fitted_distribution_center=list(reference), fold_models={'full':fit}, domain=list(domain)))


def test_transfer_preserves_source_cell_mass_on_denser_recipient_lattice():
    source = frozen([2, 11, 25, 41, 55])
    original = source['grid']
    refined = boundary_grid(np.union1d(original['positions'], np.arange(60)), (0, 60))
    density = transfer_density(source, original)
    expected = _density(np.array([0., 0., 1., 0., 1.]),
                        (original['coordinates']-[12, 40])/10.)[0]
    assert np.allclose(density, expected)
    fine = transfer_density(source, refined)
    assert logsumexp(density+np.log(original['areas'])) == pytest.approx(
        logsumexp(fine+np.log(refined['areas'])), abs=1e-10)
    # No Gaussian extrapolation into intervals invisible in the fitted source.
    assert np.isneginf(fine).any()


def test_unequal_domains_preserve_terminal_source_cell_mass():
    source=frozen([11,20,35,49],domain=(8,52),reference=(20,35))
    original=source['grid']
    # Source limits deliberately lie inside coarse recipient terminal cells.
    refined=boundary_grid([2,11,17,20,35,45,49,59],(0,70),model_domains=[(8,52)])
    a=transfer_density(source,original);b=transfer_density(source,refined)
    assert logsumexp(a+np.log(original['areas']))==pytest.approx(logsumexp(b+np.log(refined['areas'])),abs=1e-10)


def unit(hits, spans=((12, 40),)):
    p = np.arange(1, 60, 2)
    return dict(positions=p, hits=np.asarray(hits), p_accessible=np.full(len(p), .6),
        p_protected=np.full(len(p), .01), representative_raw_tf_intervals=list(spans),
        raw_nuc_intervals=[])


def test_actual_modified_core_veto_cannot_be_erased_by_edge_floor():
    p = np.arange(1, 60, 2); model = frozen(p)
    call = dict(unit_id='r', ordinal=0, start=12, end=40, strand='GA')
    out = transferred_call(model, model['grid'], unit(np.ones(len(p))), call,
                           floor_bp=10, replicates=63)
    assert out['status'] == 'core_contradicted'
    # A bounded floor must not erase the entire model penalty. The actual
    # modified-core veto remains mandatory regardless of the adjusted loss.
    assert 0 <= out['floor_adjusted_loss'] <= out['native_loss']
    assert out['edge_tolerance_semantics'] == 'bounded_joint_endpoint_cell_profile_v1'
    assert out['mean_core_native_log_lr'] < -np.log(100)
    assert out['interval'] == [12, 40]


@pytest.mark.parametrize('require_overlap',[True,False])
def test_broad_transfer_blocked_by_real_neighbour_remains_explicit(require_overlap):
    p = np.arange(1, 60, 2); model = frozen(p, reference=(12, 50))
    call = dict(unit_id='r', ordinal=0, start=12, end=22, strand='CT')
    u = unit((p < 12) | (p >= 22), spans=((12,22), (25,50)))
    out = transferred_call(model, model['grid'], u, call, replicates=63,
                           require_call_overlap=require_overlap)
    assert out['status'] == 'geometry_blocked_by_neighbours_or_domain'
    assert out['physical_geometry_retention'] < .05


def test_missing_opportunities_never_become_simulated_misses():
    p = np.arange(1,60,2); model=frozen(p)
    u=unit(np.zeros(len(p)))
    u['positions']=np.array([70]); u['hits']=np.array([0])
    u['p_accessible']=np.array([.6]); u['p_protected']=np.array([.01])
    call=dict(unit_id='r',ordinal=0,start=12,end=40,strand='CT')
    out=transferred_call(model,model['grid'],u,call,replicates=63)
    assert out['status']=='no_recipient_information'


def test_direction_reports_strands_and_does_not_drop_failed_calls():
    rows=[dict(status='scored',strand='CT',predictive_tail=.8,predictive_tail_interval=[.7,.9]),
          dict(status='core_contradicted',strand='GA'),
          dict(status='geometry_blocked_by_neighbours_or_domain',strand='GA')]
    out=summarize_direction(rows,recipient_dataset='B')
    assert out['n_source_calls']==3 and out['compatible_fraction']==pytest.approx(1/2)
    assert out['source_coverage_fraction']==pytest.approx(2/3)
    assert out['by_strand']['CT']['compatible_fraction']==1
    assert out['by_strand']['GA']['compatible_fraction']==0
    assert out['contradicted_calls']==1 and out['untestable_calls']==1


def test_all_untestable_is_not_disagreement():
    out=summarize_direction([dict(status='no_recipient_information',strand='GA')],recipient_dataset='B')
    assert out['compatible_fraction'] is None and out['n_testable']==0
    assert out['source_coverage_fraction']==0


def test_unresolvable_simulation_gate_and_missing_frozen_model_fail_explicitly():
    with pytest.raises(ValueError,match='Predictive gate unresolved'):
        reciprocal_native_graph({},region=(0,100),reference_percent=99.9,replicates=511)
    with pytest.raises(ValueError,match='Predictive gate unresolved'):
        reciprocal_native_graph({},region=(0,100),reference_percent=99.999,replicates=4095)
    with pytest.raises(ValueError,match='frozen latent-distribution'):
        reciprocal_native_graph({'A':{'result':{'diagnostics':{}}}},region=(0,100))


def test_spatially_displaced_model_is_not_attributed_to_an_unrelated_call():
    p=np.arange(1,60,2);source=frozen(p,reference=(12,22))
    # Parameters encode log precision, not an SD: this makes the physical
    # edge SD 1 bp (coordinate scale / exp(log precision)). The broader
    # default fixture still has substantial mass overlapping the caller alias.
    source['model']['fold_models']['full']['parameters']=[0.,0.,np.log(10.),0.,np.log(10.)]
    call=dict(unit_id='r',ordinal=0,start=26,end=34,strand='CT')
    u=unit((p<12)|(p>=34),spans=((26,34),))
    strict=transferred_call(source,source['grid'],u,call,replicates=127)
    crossed=transferred_call(source,source['grid'],u,call,replicates=127,require_call_overlap=False)
    assert strict['status']=='hypothesis_not_attributable_to_this_call'
    assert crossed['status']=='scored' and crossed['physical_geometry_retention']==pytest.approx(1.)
    assert crossed['interval']==[26,34] and crossed['raw_call_overlap_required'] is False
    # An unrestricted site readout remains available diagnostically. It must
    # not label a spatially separate original call merely by non-rejection.
    assert crossed['native_loss']>0


def test_unobserved_distributed_core_is_censored_not_supported_by_flanks():
    p=np.arange(1,60,2);source=frozen(p)
    u=unit(np.zeros(len(p)))
    u.update(positions=np.array([1,59]),hits=np.array([1,1]),
        p_accessible=np.array([.6,.6]),p_protected=np.array([.01,.01]))
    call=dict(unit_id='r',ordinal=0,start=12,end=40,strand='CT')
    out=transferred_call(source,source['grid'],u,call,replicates=63,
        require_call_overlap=False,minimum_visible_geometry_mass=.05)
    assert out['observed_opportunities']==2
    assert out['status']=='transferred_shape_unobserved'
    assert out['geometry_observed_mass_fraction']<.05


def test_raw_call_attribution_integrates_lattice_aliases_not_a_tiny_mean_core():
    source=frozen([2,11,25,41,55], reference=(18,26))
    source['model']['fold_models']['full']['parameters']=[0.,0.,np.log(10),0.,np.log(10)]
    # Most source mass lies in a boundary cell with right edge26..41. The
    # Gaussian location26 is NOT the normalized positive-cell boundary mean.
    call=dict(unit_id='r',ordinal=0,start=30,end=34,strand='CT')
    u=unit(np.zeros(30),spans=((30,34),))
    out=transferred_call(source,source['grid'],u,call,replicates=63)
    exact=boundary_grid(np.arange(60),(0,60))
    lm=transfer_density(source,exact)+np.log(exact['areas'])
    xy=exact['coordinates'];mask=(xy[:,0]<34)&(xy[:,1]>30)
    expected=np.exp(logsumexp(lm[mask])-logsumexp(lm))
    assert out['geometry_mass_overlapping_observed_call']==pytest.approx(expected,abs=1e-14)
    assert expected>.05
    assert out['status']=='scored'


def test_exact_following_neighbor_retention_matches_integer_enumeration():
    p=np.arange(1,60,2);source=frozen(p,reference=(12,50))
    call=dict(unit_id='r',ordinal=0,start=12,end=50,strand='CT')
    out=transferred_call(source,source['grid'],unit(np.zeros(len(p)),spans=((12,50),(52,58))),call,replicates=63)
    exact=boundary_grid(np.arange(60),(0,60));lm=transfer_density(source,exact)+np.log(exact['areas'])
    a,b=exact['coordinates'].T;valid=(a<50)&(b>12)&(b<=52)
    expected=np.exp(logsumexp(lm[valid])-logsumexp(lm))
    assert out['physical_geometry_retention']==pytest.approx(expected,abs=1e-14)
    assert out['exact_raw_call_neighbor_cell_refinement']


def test_domain_nomination_does_not_delete_a_displaced_hypothesis_before_scoring():
    a=frozen([1,5,15,30,45,59],reference=(5,10))
    b=frozen([1,5,15,30,45,59],reference=(45,50))
    assert model_overlap(a,b)
    other=frozen([70,80,90],domain=(60,100),reference=(75,85))
    assert not model_overlap(a,other)


def test_empty_and_unattributable_cohorts_are_not_called_resolution_limited():
    empty=summarize_direction([],recipient_dataset='A')
    usable=summarize_direction([dict(status='scored',strand='CT',predictive_tail=.8,
        predictive_tail_interval=[.7,.9]) for _ in range(5)],recipient_dataset='B')
    assert reciprocal_summary_status({'A':empty,'B':usable})=='provisional_no_native_primary_cohort'
    displaced=summarize_direction([dict(status='hypothesis_not_attributable_to_this_call',strand='GA')
                                  for _ in range(5)],recipient_dataset='A')
    assert displaced['geometry_limited_calls']==5 and displaced['information_limited_calls']==0
    assert reciprocal_summary_status({'A':displaced,'B':usable})=='provisional_incomplete_reciprocal_assessment'
    assert reciprocal_summary_status({'A':usable,'B':usable})=='provisional_reciprocal_compatible'


def test_monte_carlo_borderline_is_separate_from_clear_non_rejection():
    rows=[dict(status='scored',strand='CT',predictive_tail=.002,predictive_tail_interval=[.0015,.003]),
          dict(status='scored',strand='CT',predictive_tail=.0008,predictive_tail_interval=[.0004,.0015]),
          dict(status='scored',strand='CT',predictive_tail=.0001,predictive_tail_interval=[0.,.0005])]
    out=summarize_direction(rows,recipient_dataset='B',reference_percent=99.9)
    assert out['n_compatible']==2
    assert out['n_compatible_mc_clear']==out['n_compatible_mc_borderline']==1
    assert out['compatible_fraction']==pytest.approx(2/3)
    assert out['compatible_mc_clear_fraction']==pytest.approx(1/3)


def test_zero_or_invalid_geometry_mass_gate_is_not_an_information_bypass():
    for name in ('minimum_call_attribution_mass','minimum_geometry_retention','minimum_visible_geometry_mass'):
        with pytest.raises(ValueError,match='fractions'):
            reciprocal_native_graph({},region=(0,100),**{name:0.})
