import copy
import itertools

import numpy as np
import pytest

from fiberhmm.inference.consensus.native_compound import compound_profile_statistic, compound_predictive_reference, score_existing_compound
from fiberhmm.inference.consensus.native_cross import boundary_grid, transferred_call, transfer_density
from scipy.special import logsumexp


def fixture(hits=None):
    p = np.arange(1, 60, 2); grid = boundary_grid(p, (0, 60))
    model = dict(family='donor:broad', reference_interval=[12, 50], fitted_distribution_center=[12, 50],
        fold_models={'full':dict(parameters=[0., 0., np.log(10), 0., np.log(10)],
            parameter_reference=[12, 50], parameter_coordinate_scale_bp=10.)}, domain=[0, 60])
    u = dict(positions=p, hits=((p < 12) | (p >= 50)).astype(int) if hits is None else hits,
        p_accessible=np.full(len(p), .6), p_protected=np.full(len(p), .01),
        representative_raw_tf_intervals=[[12, 22], [28, 50]], raw_nuc_intervals=[])
    calls = [dict(unit_id='u', ordinal=i, start=a, end=b, strand='CT') for i, (a, b) in enumerate(u['representative_raw_tf_intervals'])]
    return dict(model=model, grid=grid), u, calls


def test_internal_pattern_scan_matches_every_binary_configuration():
    steps = np.array([1., -2., 3., -4., 2.])
    aa, bb = np.triu_indices(6, 1); penalty = -.1*np.arange(len(aa)); penalty -= penalty.max()
    value, shape, gaps = compound_profile_statistic(steps, aa, bb, penalty, penalty)
    continuous = max(sum(steps[a:b])+q for a,b,q in zip(aa,bb,penalty))
    punctuated = max(sum(steps[a:b]*np.array(c))+q for a,b,q in zip(aa,bb,penalty)
                     for c in itertools.product((0,1), repeat=b-a))
    assert gaps == pytest.approx(punctuated-continuous)
    assert value == pytest.approx(max(shape, gaps))


def test_unfreezing_compound_retains_all_observations_and_does_not_change_atoms():
    f, u, calls = fixture(); old = copy.deepcopy(u)
    atoms = [transferred_call(f, f['grid'], u, c, replicates=63, require_call_overlap=False) for c in calls]
    assert all(a['status'] == 'geometry_blocked_by_neighbours_or_domain' for a in atoms)
    whole = score_existing_compound(f, u, calls, replicates=255)
    assert whole['status'] == 'scored' and whole['predictive_tail'] > .05
    assert whole['geometry_mass_overlapping_all_pieces'] > .9
    assert whole['member_observations_unfrozen'] and whole['merged_interval'] is None
    assert whole['source_intervals'] == [[12, 22], [28, 50]]
    for name in ('positions', 'hits', 'p_accessible', 'p_protected'): assert np.array_equal(u[name], old[name])
    assert u['representative_raw_tf_intervals'] == old['representative_raw_tf_intervals']
    assert atoms == [transferred_call(f, f['grid'], u, c, replicates=63, require_call_overlap=False) for c in calls]


def test_strong_real_internal_modifications_reject_without_chosen_gap_veto():
    f, u, calls = fixture(); p = u['positions']; u['hits'][(p >= 22) & (p < 42)] = 1
    out = score_existing_compound(f, u, calls, replicates=4095, floor_bp=10)
    assert out['status'] == 'scored' and out['predictive_tail_interval'][1] < .001
    assert out['internal_pattern_gain'] > 20
    assert out['fixed_selected_gap_veto'] is False


def test_nucleosome_cannot_be_unfrozen_by_tf_compound():
    f, u, calls = fixture(); u['raw_nuc_intervals'] = [[20, 150]]
    assert score_existing_compound(f, u, calls)['status'] == 'frozen_nucleosome_blocks_compound'


def test_missing_readout_never_becomes_an_internal_miss():
    f, u, calls = fixture(); u.update(positions=np.array([70]), hits=np.array([1]),
        p_accessible=np.array([.6]), p_protected=np.array([.01]))
    assert score_existing_compound(f, u, calls)['status'] == 'no_recipient_information'


def test_equal_native_observations_give_identical_joint_reference():
    n = 12; a,b = np.triu_indices(n+1,1)
    d = -10.*np.square(a-2)-10.*np.square(b-10); lm=d-np.logaddexp.reduce(d)
    pa=np.full(n,.6); pp=np.full(n,.01); hit=np.ones(n,bool); hit[2:10]=False; hit[6]=True
    values=np.where(hit,np.log(pp/pa),np.log1p(-pp)-np.log1p(-pa))
    args=dict(allowed=np.ones(len(a),bool),observed=np.ones(n,bool),p_accessible=pa,p_protected=pp,
              replicates=1023,seed=234)
    one=compound_predictive_reference(d,lm,values,a,b,**args)
    two=compound_predictive_reference(d+100,lm+100,values,a,b,**args)
    assert one == two
    assert one['internal_pattern_scan_repeated_in_reference']
    # One internal hit is not assigned the nominal single-site miss/hit rate
    # as a p-value; all possible internal patterns are rescanned.
    assert one['predictive_tail'] > .01


def test_following_nonlattice_neighbor_excludes_every_crossing_integer_boundary():
    f, u, calls = fixture()
    u['representative_raw_tf_intervals'].append([52, 58])
    out = score_existing_compound(f, u, calls, replicates=63)
    # Independent enumeration of every positive integer boundary pair. Do
    # not copy the implementation's neighbor cuts into this reference.
    exact = boundary_grid(np.arange(0, 60), (0, 60))
    mass = transfer_density(f, exact)+np.log(exact['areas'])
    left, right = exact['coordinates'].T
    valid = (right <= 52)
    for c in calls: valid &= (left < c['end']) & (right > c['start'])
    expected = np.exp(logsumexp(mass[valid])-logsumexp(mass))
    assert out['status'] == 'scored'
    assert out['geometry_mass_overlapping_all_pieces'] == pytest.approx(expected, abs=1e-14)


def test_adding_unobserved_source_cell_cuts_preserves_reference(monkeypatch):
    import fiberhmm.inference.consensus.native_compound as module
    f, u, calls = fixture(); u['representative_raw_tf_intervals'].append([52, 58])
    original = score_existing_compound(f, u, calls, replicates=255)
    build = module.boundary_grid
    monkeypatch.setattr(module, 'boundary_grid',
        lambda positions, domain: build(np.union1d(positions, np.arange(*domain)), domain))
    refined = score_existing_compound(f, u, calls, replicates=255)
    for key in ('geometry_mass_overlapping_all_pieces', 'compound_statistic',
                'native_shape_loss', 'internal_pattern_gain', 'predictive_tail'):
        assert original[key] == pytest.approx(refined[key], abs=1e-13)
    assert original['predictive_tail_interval'] == refined['predictive_tail_interval']


def test_edge_floor_does_not_erase_internal_pattern_evidence():
    n = 12; a, b = np.triu_indices(n+1, 1)
    d = -10.*np.square(a-2)-10.*np.square(b-10); lm=d-logsumexp(d)
    pa=np.full(n,.6); pp=np.full(n,.01); hit=np.ones(n,bool); hit[2:10]=False; hit[6]=True
    values=np.where(hit,np.log(pp/pa),np.log1p(-pp)-np.log1p(-pa))
    args=dict(allowed=np.ones(len(a),bool),observed=np.ones(n,bool),p_accessible=pa,p_protected=pp,
              replicates=np.int64(255),seed=123)
    regular=compound_predictive_reference(d,lm,values,a,b,**args)
    relaxed=compound_predictive_reference(d,lm,values,a,b,relax_left=True,relax_right=True,**args)
    assert relaxed['native_shape_loss'] == 0
    assert relaxed['internal_pattern_gain'] == regular['internal_pattern_gain'] > 0


def test_unselected_overlapping_call_is_not_silently_unfrozen():
    f, u, calls = fixture(); u['representative_raw_tf_intervals'].append([18, 32])
    assert score_existing_compound(f,u,calls)['status'] == 'unselected_overlapping_call_blocks_compound'


def test_zero_statistic_has_uniform_monte_carlo_schema():
    out=compound_predictive_reference(np.array([0.]),np.array([0.]),np.array([1.]),
        np.array([0]),np.array([1]),allowed=np.array([True]),observed=np.array([True]),
        p_accessible=np.array([.6]),p_protected=np.array([.01]),replicates=np.int64(63))
    assert out['compound_statistic'] == out['simulation_resolution'] == out['simulations'] == 0


def test_gap_diagnostics_are_readouts_not_fixed_gap_pvalues():
    f, u, calls = fixture(); u['hits'][(u['positions'] >= 22) & (u['positions'] < 28)] = 1
    out=score_existing_compound(f,u,calls,replicates=63)
    assert out['inter_piece_observed_hits'] == out['inter_piece_observed_opportunities'] == 3
    assert out['inter_piece_gaps'][0]['native_log_lr'] == pytest.approx(3*np.log(.01/.6))
    assert out['inter_piece_gaps'][0]['fixed_gap_pvalue'] is None
    assert out['member_observed_opportunities_outside_source_domain'] == 0
