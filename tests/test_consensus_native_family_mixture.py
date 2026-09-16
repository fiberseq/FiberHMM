import copy
import hashlib
import json

import numpy as np
import pytest

from fiberhmm.inference.consensus.artifacts import digest, json_default
from fiberhmm.inference.consensus.native_family_mixture import (
    SCHEMA, fit_conditional_two_component_weights, native_component_evidence,
    fit_explicit_native_family_mixture, restore_native_mixture_geometry, _pack_log)


def known_mixture():
    ll = np.log([[.9, .1]] * 75 + [[.1, .9]] * 25)
    q = np.array([[0., -np.inf], [-np.inf, 0.]])
    return native_component_evidence(ll, np.ones(ll.shape, bool), q)


def test_known_weight_is_native_mle_not_raw_class_frequency():
    a, b = known_mixture()
    fit = fit_conditional_two_component_weights(a, b)
    assert fit['converged']
    assert fit['weights'] == pytest.approx([.8125, .1875], abs=1e-5)
    assert fit['global_objective_gap'] <= 1e-6


def test_conditioning_normalizer_cancels_unequal_exposure_when_data_uninformative():
    b = np.log([[.99, .01], [.4, .6], [.001, .9]])
    fit = fit_conditional_two_component_weights(b, b)
    assert fit['log_likelihood'] == pytest.approx(0., abs=1e-12)
    assert fit['converged'] and fit['conditionally_indistinguishable_on_training']
    assert fit['weights'] == [0., 1.]  # Deterministic tie; no forced positive floor.


def test_zero_component_weight_is_allowed_when_supported_by_likelihood():
    a = np.log([[.9, .1]] * 25); b = np.zeros_like(a)
    fit = fit_conditional_two_component_weights(a, b)
    assert fit['weights'] == [1., 0.]
    assert fit['converged']


def test_component_and_source_order_invariance():
    a, b = known_mixture()
    first = fit_conditional_two_component_weights(a, b)
    second = fit_conditional_two_component_weights(a[::-1, ::-1], b[::-1, ::-1])
    assert second['weights'][::-1] == pytest.approx(first['weights'], abs=1e-5)
    assert second['log_likelihood'] == pytest.approx(first['log_likelihood'], abs=1e-6)


def test_global_bounds_cover_dense_profiles_without_concavity_assumption():
    rng = np.random.default_rng(84932)
    w = np.linspace(0., 1., 10001)
    for _ in range(8):
        b = -rng.uniform(0., 7., size=(7, 2))
        a = b + rng.normal(0., 4., size=b.shape)
        fit = fit_conditional_two_component_weights(a, b)
        with np.errstate(divide='ignore'):
            values = (np.logaddexp(np.log(w[:, None]) + a[:, 0], np.log1p(-w[:, None]) + a[:, 1])
                      - np.logaddexp(np.log(w[:, None]) + b[:, 0], np.log1p(-w[:, None]) + b[:, 1])).sum(axis=1)
        assert values.max() <= fit['global_objective_upper_bound'] + 1e-9
        assert fit['log_likelihood'] >= values.max() - 1e-6
        assert fit['converged']


def test_explicit_evaluation_limit_is_not_called_converged():
    a, b = known_mixture()
    fit = fit_conditional_two_component_weights(a, b, max_evaluations=3)
    assert fit['evaluations'] == 3
    assert not fit['converged']
    assert fit['global_objective_gap'] > fit['objective_tolerance']


def test_extreme_likelihoods_do_not_underflow_into_missing_or_nan():
    fit = fit_conditional_two_component_weights([[1000., -1000.], [-1000., 1000.]], np.zeros((2, 2)))
    assert fit['converged']
    assert fit['weights'] == pytest.approx([.5, .5], abs=1e-6)
    assert fit['log_likelihood'] == pytest.approx(2000. - 2 * np.log(2.))


def test_impossible_conditioning_endpoint_cannot_be_selected_as_exact_zero():
    a = np.array([[0., -np.inf], [0., 2.]])
    b = np.array([[0., -np.inf], [0., 0.]])
    fit = fit_conditional_two_component_weights(a, b)
    assert fit['endpoint_feasible'] == [False, True]
    assert fit['weights'][0] > 0
    assert fit['converged']


def test_integrated_components_match_direct_enumeration_with_zero_cells_and_masks():
    q = np.array([[.2, .8, 0.], [.6, 0., .4]])
    ll = np.log([[2., 3., .5], [.4, 2., 5.]])
    allowed = np.array([[True, True, False], [True, False, True]])
    with np.errstate(divide='ignore'): a, b = native_component_evidence(ll, allowed, np.log(q), batch_size=1)
    for i in range(2):
        for k in range(2):
            assert np.exp(a[i, k]) == pytest.approx(sum(q[k, g] * np.exp(ll[i, g]) for g in range(3) if allowed[i, g]))
            assert np.exp(b[i, k]) == pytest.approx(sum(q[k, g] for g in range(3) if allowed[i, g]))


def test_portable_zero_density_support_is_exact_and_integrity_checked():
    from fiberhmm.inference.consensus.native_cross import boundary_grid, frozen_tabulated_geometry
    grid = boundary_grid([1, 3], (0, 5))
    density = np.full(3, -np.inf); density[1] = -np.log(grid['areas'][1])
    model = dict(family='d:mixture', fold_models={'full': {'weights': [1., 0.]}},
                 reference_interval=[1, 4], domain=[0, 5], child_family_ids=['d:A', 'd:B'],
                 component_weights={'d:A': 1., 'd:B': 0.})
    frozen = frozen_tabulated_geometry(model, grid, density)
    mass = density+np.log(grid['areas'])
    other = np.full(3, -np.inf); other[0] = 0.
    record = dict(schema=SCHEMA, model=model, grid=grid, tabulated_source_log_density=_pack_log(density),
                  source_log_density_sha256=frozen['source_log_density_sha256'], fit={'weights': [1., 0.]},
                  components=[{'family': 'd:A', 'common_grid_log_mass': _pack_log(mass)},
                              {'family': 'd:B', 'common_grid_log_mass': _pack_log(other)}])
    record['artifact_sha256'] = digest(record)
    portable = json.loads(json.dumps(record, default=json_default, allow_nan=False))
    restored = restore_native_mixture_geometry(portable)
    assert np.array_equal(restored['source_log_density'], density)
    assert not restored['source_log_density'].flags.writeable
    assert [c['weight'] for c in restored['mixture_components']] == [1., 0.]
    assert np.array_equal(restored['mixture_components'][1]['frozen']['source_log_density']+np.log(grid['areas']), other)
    # Re-sign intentionally inconsistent payloads: structural checks must catch
    # them independently of the outer artifact hash, including zero-weight q.
    invalid = copy.deepcopy(portable)
    invalid['components'][1]['common_grid_log_mass']['finite_values'][0] += .2
    invalid['artifact_sha256'] = digest({k: v for k, v in invalid.items() if k != 'artifact_sha256'})
    with pytest.raises(ValueError, match='normalized'): restore_native_mixture_geometry(invalid)
    invalid = copy.deepcopy(portable)
    invalid['model']['component_weights'] = {'d:A': .5, 'd:B': .5}
    invalid['model']['fold_models']['full']['weights'] = invalid['fit']['weights'] = [.5, .5]
    invalid['artifact_sha256'] = digest({k: v for k, v in invalid.items() if k != 'artifact_sha256'})
    with pytest.raises(ValueError, match='does not match'): restore_native_mixture_geometry(invalid)
    invalid = copy.deepcopy(portable)
    invalid['model']['component_weights'] = {'d:A': 1., 'd:B': -.01}
    invalid['artifact_sha256'] = digest({k: v for k, v in invalid.items() if k != 'artifact_sha256'})
    with pytest.raises(ValueError, match='nonnegative'): restore_native_mixture_geometry(invalid)
    portable['tabulated_source_log_density']['finite_values'][0] += .01
    with pytest.raises(ValueError, match='digest'): restore_native_mixture_geometry(portable)


def test_full_parent_wrapper_preserves_children_sources_and_grid_binding():
    from test_consensus_native_family_consolidation import fixture, run
    from fiberhmm.inference.consensus.native_cross import boundary_grid
    data, native = fixture()
    grid = boundary_grid(data['units'][0]['positions'], (0, 61))
    grid_hash = hashlib.sha256(np.c_[grid['coordinates'], grid['areas']].astype('<f8').tobytes()).hexdigest()
    for model in native['family_models']:
        model['fold_models']['full'].update(parameter_reference=model['reference_interval'],
            parameter_coordinate_scale_bp=10., projection_grid_sha256=grid_hash)
    template = run(data, native)
    children = [dict(model=m, grid=grid) for m in native['family_models']]
    before = digest([data, native, template, children])
    out = fit_explicit_native_family_mixture(data, native, template, children,
                                            parent_family_id='d:mixture')
    assert out['fit']['converged']
    assert out['provenance']['source_groups_fit_once'] == 6
    assert out['model']['source_call_indices'] == template['model']['source_call_indices']
    assert out['provenance']['parent_per_molecule_assignments_published'] is False
    assert digest([data, native, template, children]) == before
    restore_native_mixture_geometry(json.loads(json.dumps(out, default=json_default)))
    data['units'][0]['hits'][6] = 1
    with pytest.raises(ValueError, match='likelihood or neighbor conditioning'):
        fit_explicit_native_family_mixture(data, native, template, children, parent_family_id='d:other')


def test_invalid_or_unnormalized_components_rejected():
    with pytest.raises(ValueError, match='normalized'):
        native_component_evidence(np.zeros((2, 2)), np.ones((2, 2), bool), np.zeros((2, 2)))
    with pytest.raises(ValueError, match='support'):
        fit_conditional_two_component_weights([[0., -np.inf]], [[0., 0.]])
