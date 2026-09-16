import copy
import numpy as np
import pytest
from scipy.special import logsumexp

from fiberhmm.inference.consensus.native_cross import (
    boundary_grid, frozen_tabulated_geometry, transfer_density, transferred_call)


def source():
    grid = boundary_grid([2, 11, 25, 41, 55], (0, 60))
    xy = grid['coordinates']
    left = -.4*np.square(xy-[10., 42.]).sum(1)
    right = -.4*np.square(xy-[26., 42.]).sum(1)
    area = np.log(grid['areas'])
    left -= logsumexp(left+area); right -= logsumexp(right+area)
    d = np.logaddexp(np.log(.7)+left, np.log(.3)+right)
    model = dict(family='D:parent', reference_interval=[10, 42], domain=[0, 60])
    return frozen_tabulated_geometry(model, grid, d)


def test_tabulated_multimodal_source_survives_exact_dense_lattice_transfer():
    frozen = source(); grid = frozen['grid']
    assert np.array_equal(transfer_density(frozen, grid), frozen['source_log_density'])
    dense = boundary_grid(np.arange(60), (0, 60))
    transferred = transfer_density(frozen, dense)
    assert logsumexp(transferred+np.log(dense['areas'])) == pytest.approx(0., abs=1e-12)
    assert np.isneginf(transferred).any()
    p = grid['positions']
    a = np.searchsorted(p, dense['coordinates'][:, 0]); b = np.searchsorted(p, dense['coordinates'][:, 1])
    valid = np.isfinite(transferred)
    index = a[valid]*(2*len(p)-a[valid]+1)//2+b[valid]-a[valid]-1
    assert np.array_equal(transferred[valid], frozen['source_log_density'][index])


def test_tabulated_transfer_is_not_an_unchecked_vector_or_implicit_renormalization():
    frozen = source()
    with pytest.raises(ValueError, match='already be normalized'):
        frozen_tabulated_geometry(frozen['model'], frozen['grid'], frozen['source_log_density']+1)
    altered = dict(frozen, source_log_density=frozen['source_log_density'].copy())
    altered['source_log_density'][0] += 1
    with pytest.raises(ValueError, match='integrity'):
        transfer_density(altered, altered['grid'])
    with pytest.raises(ValueError, match='excluded-fold'):
        transfer_density(frozen, frozen['grid'], fold=0)
    changed = copy.deepcopy(frozen['grid']); changed['coordinates'][0, 0] += .5
    with pytest.raises(ValueError, match='canonical'):
        frozen_tabulated_geometry(frozen['model'], changed, frozen['source_log_density'])


def test_actual_tabulated_parent_uses_the_same_bounded_native_transfer_path():
    frozen = source(); p = np.arange(60)
    u = dict(positions=p, hits=((p < 20) | (p >= 42)).astype(int),
        p_accessible=np.full(60, .6), p_protected=np.full(60, .01),
        representative_raw_tf_intervals=[[20, 42]], raw_nuc_intervals=[])
    call = dict(unit_id='H:r', ordinal=0, start=20, end=42, strand='pooled')
    result = transferred_call(frozen, frozen['grid'], u, call, floor_bp=10, replicates=127)
    assert result['status'] == 'scored'
    assert result['interval'] == [20, 42]
    assert result['edge_tolerance_changes_generative_mass'] is False
    assert result['edge_tolerance_semantics'] == 'bounded_joint_endpoint_cell_profile_v1'
    assert result['core_geometry_summary'] == 'normalized_positive_cell_coverage'


@pytest.mark.parametrize('compact_weight,all_modified,expected_status',
                         [(.1, False, 'scored'), (1e-12, False, 'core_contradicted'),
                          (.1, True, 'core_contradicted')])
def test_weighted_mixture_core_veto_neither_forces_majority_shape_nor_uses_boolean_or(
        compact_weight, all_modified, expected_status):
    p = np.arange(11); grid = boundary_grid(p, (0, 11))
    a, b = grid['starts'], grid['ends']
    broad = np.flatnonzero((a == 0) & (b == 11))[0]
    compact = np.flatnonzero((a == 8) & (b == 11))[0]
    log_mass = np.full(len(a), -np.inf)
    log_mass[broad] = np.log1p(-compact_weight)
    log_mass[compact] = np.log(compact_weight)
    model = dict(family='D:parent', reference_interval=[0, 11], domain=[0, 11])
    frozen = frozen_tabulated_geometry(model, grid, log_mass-np.log(grid['areas']))
    parts = []
    for name, index, weight in [('broad', broad, 1-compact_weight), ('compact', compact, compact_weight)]:
        lm = np.full(len(a), -np.inf); lm[index] = 0.
        cf = frozen_tabulated_geometry(dict(family='D:'+name), grid, lm-np.log(grid['areas']))
        parts.append(dict(family='D:'+name, weight=weight, frozen=cf))
    frozen['mixture_components'] = parts
    hits = np.ones(11, int) if all_modified else (p < 8).astype(int)
    unit = dict(positions=p, hits=hits, p_accessible=np.full(11, .9), p_protected=np.full(11, .1),
                representative_raw_tf_intervals=[[0, 11]], raw_nuc_intervals=[])
    call = dict(unit_id='H:r', ordinal=0, start=0, end=11, strand='pooled')
    result = transferred_call(frozen, grid, unit, call, floor_bp=0, replicates=127)
    assert result['status'] == expected_status
    assert result['mean_core_native_log_lr'] < -np.log(100.)  # old majority-core veto
    broad_lr = (-11 if all_modified else -5)*np.log(9.)
    compact_lr = (-3 if all_modified else 3)*np.log(9.)
    exact = np.logaddexp(np.log1p(-compact_weight)+broad_lr, np.log(compact_weight)+compact_lr)
    assert result['mixture_core_native_log_lr'] == pytest.approx(exact)
    evidence = result['mixture_component_evidence']
    assert sum(r['conditional_prior_weight'] for r in evidence) == pytest.approx(1.)
    assert sum(r['posterior_weight_given_native_observations'] for r in evidence) == pytest.approx(1.)
    if compact_weight == .1 and not all_modified:
        assert exact > 4.
        assert evidence[1]['posterior_weight_given_native_observations'] > .99999
