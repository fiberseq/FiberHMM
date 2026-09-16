import numpy as np
import pytest

from fiberhmm.inference.consensus.measurement_geometry import summarize_boundary_cells, summarize_grid_geometry, projection_coverage_probability
from fiberhmm.inference.consensus.native_cross import boundary_grid, transfer_density


def test_actual_positive_cell_moments_not_untruncated_gaussian_location():
    grid = boundary_grid([3, 8, 16, 25], (0, 30))
    frozen = dict(grid=grid, model=dict(fold_models={'full':dict(parameters=[1., -1., 1., 0., 1.],
        parameter_reference=[12, 12], parameter_coordinate_scale_bp=10.)}))
    # Untruncated location is [22,2]: not a possible positive-width footprint.
    summary = summarize_grid_geometry(grid, transfer_density(frozen, grid)+np.log(grid['areas']))
    assert 0 <= summary['mean'][0] < summary['mean'][1] <= 30
    for p, box in summary['credible_boxes'].items():
        assert box['actual_mass'] >= float(p)-1e-10
    assert summary['mean_width'] > 0


def test_moments_and_joint_boxes_match_explicit_integer_enumeration():
    ll, lh, rl, rh = np.array([[1, 3], [2, 5], [8, 12], [10, 14]])
    weights = np.array([.3, .7])
    s = summarize_boundary_cells(np.log(weights), ll, lh, rl, rh)
    points, w = [], []
    for i in range(2):
        for a in range(ll[i], lh[i]+1):
            for b in range(rl[i], rh[i]+1):
                points.append([a, b]); w.append(weights[i]/((lh[i]-ll[i]+1)*(rh[i]-rl[i]+1)))
    p, w = np.array(points), np.array(w)
    mean = w @ p
    assert np.allclose(s['mean'], mean)
    assert np.allclose(s['covariance'], ((p-mean)*w[:, None]).T @ (p-mean))


def test_grid_refinement_preserves_moments_and_credible_boxes():
    grid = boundary_grid([2, 11, 25, 41, 55], (0, 60))
    frozen = dict(grid=grid, model=dict(fold_models={'full':dict(parameters=[0., 0., 1., 0., 1.],
        parameter_reference=[12, 40], parameter_coordinate_scale_bp=10.)}))
    fine = boundary_grid(np.arange(60), (0, 60))
    a, b = [summarize_grid_geometry(g, transfer_density(frozen, g)+np.log(g['areas'])) for g in (grid, fine)]
    assert np.allclose(a['mean'], b['mean'], atol=1e-10)
    assert np.allclose(a['covariance'], b['covariance'], atol=1e-10)
    for key, box in a['credible_boxes'].items():
        assert box['left'] == b['credible_boxes'][key]['left']
        assert box['right'] == b['credible_boxes'][key]['right']
        assert box['actual_mass'] == pytest.approx(b['credible_boxes'][key]['actual_mass'])


def test_huge_genomic_origin_does_not_destroy_small_within_cell_variance():
    origin = 50000000
    a = summarize_boundary_cells([0.], [origin], [origin+2], [origin+7], [origin+10])
    assert a['mean'] == [origin+1., origin+8.5]
    assert np.diag(a['covariance']) == pytest.approx([2/3, 1.25])


def test_invalid_or_empty_positive_cells_fail_explicitly():
    with pytest.raises(ValueError, match='positive-width'):
        summarize_boundary_cells([0.], [10], [20], [19], [30])
    with pytest.raises(ValueError, match='normalizable'):
        summarize_boundary_cells([-np.inf], [10], [20], [21], [30])


def test_coverage_profile_is_exact_and_refinement_invariant():
    grid = boundary_grid([2, 11, 25, 41, 55], (0, 60))
    frozen = dict(grid=grid, model=dict(fold_models={'full':dict(parameters=[0., 0., 1., 0., 1.],
        parameter_reference=[12, 40], parameter_coordinate_scale_bp=10.)}))
    fine = boundary_grid(np.arange(60), (0, 60))
    profiles = []
    for g in (grid, fine):
        lm = transfer_density(frozen, g)+np.log(g['areas'])
        profiles.append(projection_coverage_probability(lm, g['starts'], g['ends'], len(g['positions'])))
    assert np.allclose(profiles[0], profiles[1][grid['positions']])
    # On the per-base grid the number of protected opportunities is bp width.
    summary = summarize_grid_geometry(fine, transfer_density(frozen, fine)+np.log(fine['areas']))
    assert profiles[1].sum() == pytest.approx(summary['mean_width'])
