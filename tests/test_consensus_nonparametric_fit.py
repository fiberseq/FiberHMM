"""Nonparametric family geometry: monotone MM fit, unique optimum, seed-mass semantics."""
import numpy as np
import pytest
from scipy.special import logsumexp

from fiberhmm.inference.consensus.measurement_distribution import fit_native_distribution
from fiberhmm.inference.consensus.nonparametric_fit import fit_nonparametric


def _synthetic(units=12, k=12, seed=5):
    """Units with a real footprint: prefix-difference likelihoods peaking at the footprint cell,
    rectangular admissible sets around it, and corner cells no unit admits."""
    rs = np.random.RandomState(seed)
    a, b = np.triu_indices(k+1, 1)
    positions = np.arange(k)*3.
    lo, hi = np.r_[-1., positions+1][a], positions[a]
    rl, rh = positions[b-1]+1, np.r_[positions, positions[-1]+3][b]
    coordinates = np.c_[(lo+hi)/2., (rl+rh)/2.]
    areas = (hi-lo+1.)*(rh-rl+1.)
    ll = np.empty((units, len(a))); allowed = np.zeros((units, len(a)), bool)
    for u in range(units):
        start, end = rs.randint(3, 6), rs.randint(8, 11)
        log_odds = np.where((np.arange(k) >= start) & (np.arange(k) < end), 1.5, -1.)+rs.normal(0, .5, k)
        prefix = np.r_[0., np.cumsum(log_odds)]
        ll[u] = prefix[b]-prefix[a]
        alo, ahi = rs.randint(1, 3), rs.randint(6, 8)
        blo, bhi = rs.randint(7, 9), rs.randint(11, 13)
        allowed[u] = (a >= alo) & (a < ahi) & (b >= blo) & (b < bhi) & (a < b)
    return ll, allowed, coordinates, areas, np.array([positions[4], positions[9]])


def _seed():
    ll, allowed, coordinates, areas, reference = _synthetic()
    gaussian = fit_native_distribution(ll, allowed, coordinates, areas, reference=reference)
    return ll, allowed, coordinates, areas, reference, gaussian


def _conditional_loglik(log_mass, ll, allowed):
    total = 0.
    for u in range(ll.shape[0]):
        m = allowed[u]
        total += logsumexp(log_mass[m]+ll[u, m])-logsumexp(log_mass[m])
    return total


def test_penalized_objective_is_monotone_and_stationary():
    ll, allowed, _, areas, _, gaussian = _seed()
    history = []
    fit = fit_nonparametric(ll, allowed, areas, gaussian['log_mass'], alpha=2., history=history)
    assert fit['converged']
    assert np.all(np.diff(history) >= -1e-9*np.maximum(1., np.abs(history[1:])))
    assert fit['optimality_gap'] <= 1e-6
    # The exported objective is the same conditional log-likelihood the Gaussian reports,
    # evaluated on the returned mass, and the seed is a feasible point of the same problem.
    assert abs(fit['objective']-_conditional_loglik(fit['log_mass'], ll, allowed)) <= 1e-8*max(1., abs(fit['objective']))
    assert fit['objective'] >= gaussian['objective']-1e-9
    assert np.isfinite(fit['log_mass']).all() and abs(logsumexp(fit['log_mass'])) <= 1e-12


def test_optimum_does_not_depend_on_the_start():
    ll, allowed, _, areas, _, gaussian = _seed()
    admissible = allowed.any(0)
    rs = np.random.RandomState(11)
    starts = [None, np.where(admissible, 0., -np.inf)]
    for _ in range(2):
        start = np.full(ll.shape[1], -np.inf); start[admissible] = np.log(rs.dirichlet(np.ones(int(admissible.sum()))))
        starts.append(start)
    fits = [fit_nonparametric(ll, allowed, areas, gaussian['log_mass'], alpha=2., initial_log_mass=s,
                              tolerance=1e-12, max_iterations=5000) for s in starts]
    values = np.array([f['penalized_objective'] for f in fits])
    assert all(f['converged'] for f in fits)
    assert values.max()-values.min() <= 1e-7*max(1., abs(values.max()))
    masses = [np.exp(f['log_mass']) for f in fits]
    for m in masses[1:]:
        assert 0.5*np.abs(m-masses[0]).sum() <= 1e-5


def test_cells_no_unit_admits_keep_the_seed_mass():
    ll, allowed, _, areas, _, gaussian = _seed()
    admissible = allowed.any(0)
    assert (~admissible).any()
    prior = np.exp(gaussian['log_mass']-logsumexp(gaussian['log_mass']))
    fit = fit_nonparametric(ll, allowed, areas, gaussian['log_mass'], alpha=1.)
    mass = np.exp(fit['log_mass'])
    outside = ~admissible & (prior > 1e-200)
    assert outside.any()
    assert np.allclose(mass[outside], prior[outside], rtol=1e-9, atol=0.)
    assert abs(mass[admissible].sum()-prior[admissible].sum()) <= 1e-12
    assert abs(fit['prior_inadmissible_mass']-prior[~admissible].sum()) <= 1e-12
    assert fit['seed_conditioning'] == 'seed_on_admissible_cells'
    # Without smoothing the split is unidentified and the fit stays on the admissible cells.
    bare = fit_nonparametric(ll, allowed, areas, gaussian['log_mass'], alpha=0.)
    assert np.all(bare['log_mass'][~admissible] <= -600.) and np.isfinite(bare['log_mass']).all()
    assert abs(np.exp(bare['log_mass'][admissible]).sum()-1.) <= 1e-12


def test_fit_native_distribution_exports_seed_and_tabulated_geometry():
    ll, allowed, coordinates, areas, reference, gaussian = _seed()
    fit = fit_native_distribution(ll, allowed, coordinates, areas, reference=reference,
                                  objective_backend='nonparametric', smoothing_pseudo_units=2.)
    assert fit['objective_backend'] == 'nonparametric'
    assert fit['smoothing_pseudo_units'] == 2.
    assert fit['nonparametric_converged'] and fit['converged']
    seed = fit['gaussian_seed']
    assert np.allclose(seed['parameters'], gaussian['parameters'])
    assert abs(seed['objective']-gaussian['objective']) <= 1e-9*max(1., abs(gaussian['objective']))
    assert fit['objective'] >= seed['objective']-1e-9
    log_mass = np.asarray(fit['log_mass']); finite = np.isfinite(log_mass)
    assert finite.all() and abs(logsumexp(log_mass)) <= 1e-12
    assert np.allclose(fit['log_density'][finite], log_mass[finite]-np.log(areas)[finite])
    weights = np.exp(log_mass[finite])
    assert np.allclose(fit['center'], weights @ coordinates[finite])
    assert fit['penalized_objective'] <= fit['objective']+1e-9   # the smoothing term is non-positive on the simplex


def test_smoothing_option_is_validated():
    ll, allowed, _, areas, _, gaussian = _seed()
    with pytest.raises(ValueError):
        fit_native_distribution(ll, allowed, np.zeros((ll.shape[1], 2)), areas, reference=np.zeros(2),
                                objective_backend='unknown')
    strong = fit_nonparametric(ll, allowed, areas, gaussian['log_mass'], alpha=1e6)
    prior = np.exp(gaussian['log_mass']-logsumexp(gaussian['log_mass']))
    assert 0.5*np.abs(np.exp(strong['log_mass'])-prior).sum() <= 1e-3   # infinite pseudo-units return the seed
