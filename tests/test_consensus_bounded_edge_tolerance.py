import copy
import hashlib

import numpy as np
import pytest
from scipy.special import logsumexp

from fiberhmm.inference.consensus.measurement_edge_tolerance import bounded_edge_penalty
from fiberhmm.inference.consensus.measurement_distribution import (
    distribution_comparison, predictive_reference)
from fiberhmm.inference.consensus.native_cross import boundary_grid, transferred_call


def cells_for(grid):
    return np.c_[np.rint(2*grid['coordinates'][:, 0]-grid['left_hi']), grid['left_hi'],
                 grid['right_lo'], np.rint(2*grid['coordinates'][:, 1]-grid['right_lo'])]


def brute(d, cells, radius):
    values = []
    for ll, lh, rl, rh in cells:
        keep = ((cells[:, 0] <= lh+radius) & (cells[:, 1] >= ll-radius)
                & (cells[:, 2] <= rh+radius) & (cells[:, 3] >= rl-radius))
        values.append(d[keep].max()-d.max())
    return np.asarray(values)


@pytest.mark.parametrize('radius', [0, 1, 3, 10, 200])
def test_exact_joint_range_max_on_irregular_integer_cells(radius):
    grid = boundary_grid([3, 12, 14, 28, 52, 67], (0, 70))
    cells = cells_for(grid)
    rng = np.random.default_rng(19)
    d = rng.normal(size=len(cells)); d[::4] = -np.inf
    result = bounded_edge_penalty(d, cells, radius)
    assert np.array_equal(result, brute(d, cells, radius))
    assert np.allclose(result, bounded_edge_penalty(d+31, cells, radius), rtol=0, atol=8e-15)


def test_allowance_is_bp_not_projection_count_and_not_unlimited():
    cells = np.array([[0, 0, 40, 40], [5, 5, 45, 45], [30, 30, 60, 60]])
    d = np.array([0., -10., -40.])
    result = bounded_edge_penalty(d, cells, 10)
    assert np.array_equal(result, [0., 0., -40.])
    assert np.array_equal(bounded_edge_penalty(d, cells, 0), d)


def test_joint_query_does_not_splice_two_incompatible_source_modes():
    cells = np.array([[0, 0, 60, 60], [20, 20, 40, 40], [0, 0, 40, 40]])
    d = np.array([0., 0., -50.])
    # Separate coordinate maxima would fabricate a source mode at (0,40).
    assert bounded_edge_penalty(d, cells, 5)[2] == -50.


def test_adjacent_integer_bins_have_distance_one_and_native_cell_ambiguity_survives():
    cells = np.array([[0, 20, 40, 45], [21, 25, 40, 45]])
    d = np.array([0., -30.])
    assert np.array_equal(bounded_edge_penalty(d, cells, 0), d)
    assert np.array_equal(bounded_edge_penalty(d, cells, 1), [0., 0.])


def test_monotone_loss_without_erasing_distant_modification_evidence():
    grid = boundary_grid(np.arange(80), (0, 80))
    cells = cells_for(grid); xy = grid['coordinates']
    d = -2*np.square(xy-[20, 60]).sum(1)
    likelihood = -2*np.square(xy-[40, 60]).sum(1)
    losses = []
    for radius in (0, 2, 5, 10):
        out = distribution_comparison(d, likelihood, grid['starts'], grid['ends'],
            allowed=np.ones(len(d), bool), boundary_cells=cells, edge_tolerance_bp=radius)
        losses.append(out['floor_adjusted_loss'])
        assert out['native_loss'] == pytest.approx(400.)
    assert np.all(np.diff(losses) <= 0)
    assert losses[-1] > 0


def test_predictive_simulation_uses_same_bounded_penalty_and_unmodified_mass(monkeypatch):
    from fiberhmm.inference.consensus import measurement_distribution as md
    grid = boundary_grid(np.arange(12), (0, 12)); a, b = grid['starts'], grid['ends']
    cells = cells_for(grid); xy = grid['coordinates']
    d = -10*np.square(xy-[2, 5]).sum(1)
    mass = d+np.log(grid['areas']); mass -= logsumexp(mass)
    values = np.full(12, -2.); values[7:10] = 5.
    prefix = np.r_[0., np.cumsum(values)]; likelihood = prefix[b]-prefix[a]
    observed = np.ones(12, bool)
    expected_penalty = bounded_edge_penalty(d, cells, 1)
    unique, inv = np.unique(np.c_[a, b], axis=0, return_inverse=True)
    expected = np.full(len(unique), -np.inf); np.maximum.at(expected, inv, expected_penalty)
    q = np.bincount(inv, weights=np.exp(mass), minlength=len(unique))

    def fake(pa, pp, starts, ends, penalty, cdf, threshold, replicates, seed):
        assert np.array_equal(penalty, expected)
        assert np.allclose(cdf, np.cumsum(q))
        direct = distribution_comparison(d, likelihood, a, b, allowed=np.ones(len(a), bool),
            boundary_cells=cells, edge_tolerance_bp=1)
        assert threshold == direct['floor_adjusted_loss'] and threshold > 0
        return 7

    monkeypatch.setattr(md, '_predictive_exceedances', fake)
    out = predictive_reference(d, mass, likelihood, a, b,
        allowed=np.ones(len(a), bool), observed=observed,
        p_accessible=np.full(12, .6), p_protected=np.full(12, .01),
        boundary_cells=cells, edge_tolerance_bp=1, replicates=63)
    assert out['simulations'] == 63 and out['tail_exceedances'] == 7


def test_reference_display_shift_does_not_change_transfer_score():
    p = np.arange(60); grid = boundary_grid(p, (0, 60))
    source = dict(grid=grid, model=dict(family='D:F1', reference_interval=[12, 40],
        domain=[0, 60], fold_models={'full':dict(parameters=[0., 0., 1., 0., 1.],
        parameter_reference=[12, 40], parameter_coordinate_scale_bp=10.)}))
    u = dict(positions=p, hits=((p < 23) | (p >= 40)).astype(int),
        p_accessible=np.full(len(p), .6), p_protected=np.full(len(p), .01),
        representative_raw_tf_intervals=[[23, 40]], raw_nuc_intervals=[])
    call = dict(unit_id='r', ordinal=0, start=23, end=40, strand='CT')
    before = transferred_call(source, grid, u, call, floor_bp=10, replicates=127)
    renamed = copy.deepcopy(source)
    # Keep fitted density identical. A display/proposal reference must not
    # toggle unrestricted left-edge profiling at distance10 versus11.
    renamed['model']['reference_interval'] = [13, 40]
    after = transferred_call(renamed, grid, u, call, floor_bp=10, replicates=127)
    assert before == after
    assert before['edge_tolerance_semantics'] == 'bounded_joint_endpoint_cell_profile_v1'
    assert before['edge_tolerance_changes_generative_mass'] is False


def test_profile_refinement_invariance_not_false_pointwise_density_invariance():
    coarse = np.array([[0, 8, 30, 40], [9, 15, 30, 40]])
    density = np.array([0., -20.])
    fine = np.array([[0, 3, 30, 34], [0, 3, 35, 40], [4, 8, 30, 34], [4, 8, 35, 40],
                     [9, 12, 30, 34], [9, 12, 35, 40], [13, 15, 30, 34], [13, 15, 35, 40]])
    fine_d = np.repeat(density, 4)
    a = bounded_edge_penalty(density, coarse, 2)
    b = bounded_edge_penalty(fine_d, fine, 2)
    assert np.array_equal(a, b.reshape(2, 4).max(1))


@pytest.mark.parametrize('radius', [-1, 1.5, True])
def test_invalid_allowance_fails(radius):
    with pytest.raises(ValueError, match='integer'):
        bounded_edge_penalty([0.], [[0, 0, 10, 10]], radius)


def test_explicit_memory_budget_does_not_silently_prune():
    with pytest.raises(MemoryError, match='no geometry pruned'):
        bounded_edge_penalty([0.], [[0, 0, 10, 10]], 1, maximum_matrix_bytes=1)


def test_bounded_and_legacy_allowances_cannot_be_mixed():
    with pytest.raises(ValueError, match='cannot be combined'):
        distribution_comparison([0.], [0.], [0], [1], allowed=[True],
            relax_left=True, boundary_cells=[[0, 0, 10, 10]], edge_tolerance_bp=10)


@pytest.mark.parametrize('accessible,protected', [(.4, .01), (.6, .01), (.9, .1)])
def test_zero_bp_transfer_is_exact_legacy_score_at_low_middle_and_high_rates(accessible, protected):
    from fiberhmm.inference.consensus.native_cross import (
        refine_recipient_call_grid, transfer_density, _recipient_observations)
    p = np.arange(1, 60, 3); grid = boundary_grid(p, (0, 60))
    source = dict(grid=grid, model=dict(family='D:F1', reference_interval=[12, 40],
        domain=[0, 60], fold_models={'full':dict(parameters=[0., 0., 1., 0., 1.],
        parameter_reference=[12, 40], parameter_coordinate_scale_bp=10.)}))
    u = dict(positions=p, hits=((p < 18) | (p >= 43)).astype(int),
        p_accessible=np.full(len(p), accessible), p_protected=np.full(len(p), protected),
        representative_raw_tf_intervals=[[18, 43]], raw_nuc_intervals=[])
    call = dict(unit_id='r', ordinal=0, start=18, end=43, strand='CT')
    result = transferred_call(source, grid, u, call, floor_bp=0, replicates=255)
    exact = refine_recipient_call_grid(u, call, grid)
    density = transfer_density(source, exact)
    mass = density+np.log(exact['areas']); mass -= logsumexp(mass)
    obs = _recipient_observations(u, call, exact)
    seed = int.from_bytes(hashlib.sha256(b'native-cross-v1|D:F1|r|0').digest()[:4], 'little')
    legacy = predictive_reference(density, mass, obs['likelihood'], exact['starts'], exact['ends'],
        allowed=obs['allowed'], observed=obs['observed'], p_accessible=obs['p_accessible'],
        p_protected=obs['p_protected'], replicates=255, seed=seed)
    for k, value in legacy.items():
        assert result[k] == value
