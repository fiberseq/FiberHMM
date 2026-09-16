"""Separable fit backend: same algebra as the dense objective, declared non-identical."""
import numpy as np
import pytest

from fiberhmm.inference.consensus.measurement_distribution import (
    _allowed_classes, _cached_objective, fit_native_distribution)
from fiberhmm.inference.consensus.separable_fit import SeparableObjective


def _synthetic(units=7, k=12, seed=3):
    """Prefix-difference likelihoods on a triangular grid with rectangular admissible sets."""
    rs = np.random.RandomState(seed)
    a, b = np.triu_indices(k+1, 1)
    positions = np.arange(k)*3.
    lo, hi = np.r_[-1., positions+1][a], positions[a]
    rl, rh = positions[b-1]+1, np.r_[positions, positions[-1]+3][b]
    coordinates = np.c_[(lo+hi)/2., (rl+rh)/2.]
    areas = (hi-lo+1.)*(rh-rl+1.)
    ll = np.empty((units, len(a))); allowed = np.zeros((units, len(a)), bool)
    for u in range(units):
        prefix = np.r_[0., np.cumsum(rs.normal(0, 1.5, k))]
        ll[u] = prefix[b]-prefix[a]
        # Rectangles at least 2 wide in each index and overlapping the centre,
        # where a fitted geometry puts its mass, so evaluations are not all in
        # the underflow regime that legitimately falls back to the dense path.
        alo, blo = rs.randint(k//4, k//2-1), rs.randint(k//2+1, 3*k//4)
        ahi, bhi = alo+rs.randint(2, 4), min(k+1, blo+rs.randint(2, 4))
        allowed[u] = (a >= alo) & (a < ahi) & (b >= blo) & (b < bhi) & (a < b)
    return ll, allowed, coordinates, areas, np.array([positions.mean(), positions.mean()+5.])


def test_separable_matches_dense_value_and_gradient():
    ll, allowed, coordinates, areas, reference = _synthetic()
    xy = (coordinates-reference)/10.; log_area = np.log(areas)
    offset = np.where(allowed, ll, -np.inf).max(1)
    scaled = np.exp(np.where(allowed, ll-offset[:, None], -np.inf))
    args = (xy, log_area, ll, allowed, scaled, offset, allowed.astype(float), _allowed_classes(allowed))
    sep = SeparableObjective(ll, allowed, minimum_elements=0)
    for p in (np.zeros(5), np.array([.5, -.3, np.log(2.), .2, np.log(.7)])):
        f0, g0 = _cached_objective(p, *args)
        f1, g1 = sep(p, xy, log_area, lambda q: _cached_objective(q, *args))
        assert abs(f1-f0) <= 1e-9*max(1., abs(f0))
        assert np.max(np.abs(g1-g0)) <= 1e-9*max(1., np.max(np.abs(g0)))
    assert sep.fallbacks == 0


def test_structure_violations_fall_back_to_dense():
    ll, allowed, coordinates, areas, reference = _synthetic()
    broken = allowed.copy(); broken[0, np.flatnonzero(broken[0])[0]] = False   # no longer a rectangle
    with pytest.raises(ValueError):
        SeparableObjective(ll, broken, minimum_elements=0)
    # Perturb an interior cell: not on the first-start row nor the last-end
    # column that the reconstruction uses, so the cross-check must catch it.
    a, b = np.triu_indices(int(round((1+np.sqrt(1+8*ll.shape[1]))/2)), 1)
    idx = np.flatnonzero(allowed[1]); alo, bhi = a[idx].min(), b[idx].max()
    interior = [p for p in idx if a[p] != alo and b[p] != bhi]
    assert interior, 'synthetic unit 1 must have an interior cell'
    skew = ll.copy(); skew[1, interior[0]] += 1.   # no longer a prefix difference
    with pytest.raises(ValueError):
        SeparableObjective(skew, allowed, minimum_elements=0)
    fit = fit_native_distribution(ll, broken, coordinates, areas, reference=reference,
                                  max_iterations=20, objective_backend='separable')
    assert 'objective_backend' not in fit   # silently used the dense reference


def test_small_matrices_stay_dense_by_default():
    ll, allowed, coordinates, areas, reference = _synthetic()
    fit = fit_native_distribution(ll, allowed, coordinates, areas, reference=reference,
                                  max_iterations=20, objective_backend='separable')
    assert 'objective_backend' not in fit
    with pytest.raises(ValueError):
        SeparableObjective(ll, allowed)


def test_separable_fit_reaches_the_dense_optimum():
    ll, allowed, coordinates, areas, reference = _synthetic(units=9, k=14, seed=11)
    dense = fit_native_distribution(ll, allowed, coordinates, areas, reference=reference, max_iterations=100)
    import fiberhmm.inference.consensus.separable_fit as module
    original = module.SeparableObjective.__init__
    module.SeparableObjective.__init__ = lambda self, l, m, **kw: original(self, l, m, minimum_elements=0)
    try:
        sep = fit_native_distribution(ll, allowed, coordinates, areas, reference=reference,
                                      max_iterations=100, objective_backend='separable')
    finally:
        module.SeparableObjective.__init__ = original
    assert sep['objective_backend'] == 'separable'
    assert sep['separable_evaluations'] > sep['separable_dense_fallbacks']
    assert abs(sep['objective']-dense['objective']) <= 1e-6*max(1., abs(dense['objective']))
    assert np.max(np.abs(np.asarray(sep['parameters'])-np.asarray(dense['parameters']))) <= 1e-3
