import numpy as np
import pytest
from scipy.optimize import check_grad

from fiberhmm.inference.consensus.measurement_distribution import (
    native_distribution_objective, fit_native_distribution, distribution_comparison, predictive_reference, _cached_objective)


def test_native_fit_gradient_includes_projection_and_admissibility_normalizers():
    rng = np.random.default_rng(42)
    xy = rng.normal(size=(18, 2)); area = np.log(rng.uniform(.1, 5, 18))
    ll = rng.normal(size=(9, 18)); mask = rng.random((9, 18)) > .2
    theta = np.array([.2, -.1, -.5, .2, .1])
    fun = lambda p: native_distribution_objective(p, xy, area, ll, mask)[0]
    jac = lambda p: native_distribution_objective(p, xy, area, ll, mask)[1]
    assert check_grad(fun, jac, theta) < 1e-5
    offset = np.where(mask, ll, -np.inf).max(1)
    scaled = np.exp(np.where(mask, ll-offset[:, None], -np.inf))
    value, gradient = _cached_objective(theta, xy, area, ll, mask, scaled, offset, mask.astype(float))
    assert value == pytest.approx(fun(theta), abs=1e-10)
    assert np.allclose(gradient, jac(theta), atol=1e-10)


def test_native_likelihood_recovers_latent_shape_not_raw_edge_jitter():
    rng = np.random.default_rng(42)
    x, y = np.meshgrid(np.arange(-12, 13, 2), np.arange(30, 55, 2))
    xy = np.c_[x.ravel(), y.ravel()].astype(float)
    latent = rng.normal([0., 42.], [3., 3.], size=(300, 2))
    observations = latent+rng.normal(0., 4., size=(300, 2))
    # Analytically known measurement noise lives in the native likelihood.
    ll = -.5*np.square((xy[None]-observations[:, None])/4.).sum(2)
    fit = fit_native_distribution(ll, np.ones(ll.shape, bool), xy, np.ones(len(xy)), reference=[0., 42.])
    assert np.allclose(fit['center'], [0., 42.], atol=1.)
    assert np.all(np.sqrt(np.diag(fit['covariance'])) < 4.5)
    assert np.all(np.sqrt(np.diag(fit['covariance'])) > 1.5)


def test_shape_density_not_geometry_mass_and_single_edge_floor():
    aa, bb = np.triu_indices(5, 1)
    density = -3.*np.square(aa)-.2*np.square(bb-3)
    r = -7.*np.square(aa-1)-.2*np.square(bb-4)
    native = distribution_comparison(density, r, aa, bb, allowed=np.ones(len(aa), bool))
    left = distribution_comparison(density, r, aa, bb, allowed=np.ones(len(aa), bool), relax_left=True)
    both = distribution_comparison(density, r, aa, bb, allowed=np.ones(len(aa), bool), relax_left=True, relax_right=True)
    assert 0 <= left['floor_adjusted_loss'] < native['native_loss']
    assert both['floor_adjusted_loss'] == pytest.approx(0.)
    assert left['native_loss'] == native['native_loss']
    # Adding a normalization constant cannot change the density-ratio score.
    shifted = distribution_comparison(density+17, r, aa, bb, allowed=np.ones(len(aa), bool))
    assert shifted['native_loss'] == pytest.approx(native['native_loss'])


def test_repeated_more_informative_observations_accumulate_native_distinction():
    aa, bb = np.triu_indices(5, 1)
    density = -30*np.square(aa)-30*np.square(bb-3)
    weak = -.1*np.square(aa-2)-.1*np.square(bb-4)
    lo = distribution_comparison(density, weak, aa, bb, allowed=np.ones(len(aa), bool))
    hi = distribution_comparison(density, 20*weak, aa, bb, allowed=np.ones(len(aa), bool))
    assert hi['native_loss'] > lo['native_loss']


def test_same_five_extra_misses_are_plausible_at_low_rate_not_high_rate():
    n = 26
    aa, bb = np.triu_indices(n+1, 1)
    density = -100.*np.square(aa-8)-100.*np.square(bb-14)
    mass = density-np.logaddexp.reduce(density)
    hit = np.ones(n, bool); hit[8:19] = False
    results = []
    for probability in (.4, .9):
        pa = np.full(n, probability); pp = np.full(n, .01)
        values = np.where(hit, np.log(pp/pa), np.log1p(-pp)-np.log1p(-pa))
        pref = np.r_[0., np.cumsum(values)]
        results.append(predictive_reference(density, mass, pref[bb]-pref[aa], aa, bb,
            allowed=np.ones(len(aa), bool), observed=np.ones(n, bool), p_accessible=pa, p_protected=pp,
            replicates=4095, seed=812))
    assert results[0]['predictive_tail'] > .01
    assert results[1]['predictive_tail'] < .01
    assert results[0]['native_loss'] < results[1]['native_loss']


def test_eight_actual_modifications_reject_protected_extension():
    n = 30; aa, bb = np.triu_indices(n+1, 1)
    density = -100.*np.square(aa-8)-100.*np.square(bb-24)
    mass = density-np.logaddexp.reduce(density)
    hit = np.ones(n, bool); hit[8:16] = False
    pa = np.full(n, .6); pp = np.full(n, .01)
    values = np.where(hit, np.log(pp/pa), np.log1p(-pp)-np.log1p(-pa))
    pref = np.r_[0., np.cumsum(values)]
    out = predictive_reference(density, mass, pref[bb]-pref[aa], aa, bb,
        allowed=np.ones(len(aa), bool), observed=np.ones(n, bool), p_accessible=pa, p_protected=pp,
        replicates=4095, seed=534)
    assert out['predictive_tail_interval'][1] < .001


def test_predictive_zero_loss_needs_no_simulation_and_missing_is_not_miss():
    aa, bb = np.triu_indices(4, 1)
    out = predictive_reference(np.zeros(len(aa)), np.full(len(aa), -np.log(len(aa))),
        np.zeros(len(aa)), aa, bb, allowed=np.ones(len(aa), bool), observed=np.array([True, False, True]),
        p_accessible=np.array([.5, 0., .5]), p_protected=np.array([.01, 0., .01]))
    assert out['predictive_tail'] == 1.
    assert out['simulations'] == 0
