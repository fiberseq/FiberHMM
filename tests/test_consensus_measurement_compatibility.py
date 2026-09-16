from itertools import combinations
import math

import numpy as np
import pytest

from fiberhmm.inference.consensus.measurement_compatibility import (
    FootprintObservation, _best_interval, compare, score_pairs)


def observation(interval, *, pa=.4, pp=.001, positions=None, hits=None):
    positions = np.arange(24) if positions is None else np.asarray(positions)
    if hits is None:
        hits = ~((positions >= interval[0]) & (positions < interval[1]))
    return FootprintObservation(positions, hits, pa, pp, interval)


def brute(values, sa, ea):
    candidates = [(float(values[a:b].sum()), a, b)
                  for a, b in combinations(range(len(values) + 1), 2)
                  if a < ea and b > sa]
    return max(candidates, default=(-np.inf, -1, -1), key=lambda v: v[0])


def test_linear_prefix_maximum_matches_every_interval():
    rng = np.random.default_rng(732)
    for n in range(1, 15):
        for sa, ea in combinations(range(n + 1), 2):
            values = rng.normal(size=n)
            score, a, b = _best_interval(values, sa, ea)
            assert score == pytest.approx(brute(values, sa, ea)[0], abs=1e-12)
            assert a < ea and b > sa
            assert values[a:b].sum() == pytest.approx(score)


def test_shared_profile_matches_exhaustive_and_is_symmetric():
    rng = np.random.default_rng(701)
    for n in range(4, 15):
        values = rng.normal(size=(2, n))
        observed = np.ones_like(values, bool)
        sa, ea, sb, eb = 0, n - 1, 1, n
        got = score_pairs(values, observed, [0], [1], [0], [n], [sa], [ea], [sb], [eb])
        expected = (brute(values[0], sa, ea)[0] + brute(values[1], sb, eb)[0]
                    - brute(values.sum(0), max(sa, sb), min(ea, eb))[0])
        assert got['native_loss'][0] == pytest.approx(expected, abs=1e-12)
        reverse = score_pairs(values, observed, [1], [0], [0], [n], [sb], [eb], [sa], [ea])
        assert got['native_loss'][0] == pytest.approx(reverse['native_loss'][0], abs=1e-12)


def test_each_edge_profile_matches_exhaustive_two_interval_search():
    rng=np.random.default_rng(48)
    for n in range(3,9):
        v=rng.normal(size=(2,n));obs=np.ones(v.shape,bool)
        aa=[(a,b,v[0,a:b].sum()) for a,b in combinations(range(n+1),2) if a<n-1]
        bb=[(a,b,v[1,a:b].sum()) for a,b in combinations(range(n+1),2) if b>1]
        out=score_pairs(v,obs,[0],[1],[0],[n],[0],[n-1],[1],[n])
        separate=max(t[2] for t in aa)+max(t[2] for t in bb)
        end=max(x[2]+y[2] for x in aa for y in bb if x[1]==y[1])
        start=max(x[2]+y[2] for x in aa for y in bb if x[0]==y[0])
        assert out['left_edge_relaxed_loss'][0]==pytest.approx(separate-end,abs=1e-12)
        assert out['right_edge_relaxed_loss'][0]==pytest.approx(separate-start,abs=1e-12)


def test_floor_on_one_edge_does_not_cap_native_uncertainty_on_other():
    pos=np.array([1,10,11,12,13,14,20,30,70,80,100])
    a=observation((10,81),positions=pos,pa=.9)
    b=observation((15,31),positions=pos,pa=.9)
    # Five high-information left misses distinguish the raw shapes. The far
    # right discrepancy has only two opportunities and is naturally tolerated.
    zero=compare(a,b)
    ten=compare(a,b,minimum_edge_tolerance_bp=10)
    assert zero['native_loss']>math.log(100)
    assert ten['edge_discrepancies']==[5,50]
    assert ten['compatible']
    assert ten['floor_adjusted_loss']<=math.log(100)
    assert ten['native_loss']==zero['native_loss']


def test_probability_and_lattice_variation_remain_at_zero_bp():
    low = compare(observation((4, 20)), observation((4, 14)))
    high = compare(observation((4, 20), pa=.9), observation((4, 14), pa=.9))
    assert low['native_loss'] == pytest.approx(6 * math.log(.999 / .6))
    assert high['native_loss'] == pytest.approx(6 * math.log(.999 / .1))
    assert low['native_compatible'] and low['compatible']
    assert not high['compatible']
    assert not low['edge_floor_compatible']


def test_sparse_lattice_allows_large_bp_difference_without_bp_allowance():
    positions = np.array([1, 10, 20, 30, 70, 80, 100])
    result = compare(observation((10, 81), positions=positions),
                     observation((10, 31), positions=positions))
    assert result['edge_discrepancies'] == [0, 50]
    assert result['native_compatible']
    assert result['minimum_edge_tolerance_bp'] == 0


def test_noop_bp_change_has_zero_native_loss():
    positions = [1, 4, 8, 12, 18, 22]
    a = observation((4, 20), positions=positions)
    b = observation((4, 21), positions=positions)
    result = compare(a, b)
    assert result['native_loss'] == pytest.approx(0., abs=1e-12)
    assert result['compatible']


def test_floor_is_union_not_cap_or_emission_change():
    a, b = observation((4, 20), pa=.9), observation((4, 14), pa=.9)
    zero, ten = compare(a, b), compare(a, b, minimum_edge_tolerance_bp=10)
    assert zero['native_loss'] == ten['native_loss']
    assert not zero['native_compatible'] and not ten['native_compatible']
    assert ten['edge_floor_compatible'] and ten['compatible']
    # A narrower floor does not shrink an already >10bp native tolerance.
    sparse = np.array([1, 10, 20, 30, 70, 80, 100])
    c, d = observation((10, 81), positions=sparse), observation((10, 31), positions=sparse)
    assert compare(c, d, minimum_edge_tolerance_bp=10)['compatible']
    # The floor does not donate 10bp to a discrepancy beyond its own width.
    c, d = observation((2, 23), pa=.9), observation((2, 8), pa=.9)
    assert not compare(c, d, minimum_edge_tolerance_bp=10)['compatible']


def test_floor_does_not_overrule_contradicted_shared_interior():
    positions = np.arange(24)
    a = observation((4, 20), pa=.9, hits=np.ones(24, dtype=int))
    b = observation((4, 18), pa=.9)
    result = compare(a, b, minimum_edge_tolerance_bp=10)
    assert result['core_contradicted']
    assert not result['compatible']
    assert not result['edge_floor_compatible']


def test_different_native_models_and_disjoint_strand_lattices():
    a = observation((4, 20), pa=.9, pp=.02, positions=np.arange(0, 24, 2))
    b = observation((5, 21), pa=.4, pp=.001, positions=np.arange(1, 24, 2))
    result = compare(a, b)
    reverse = compare(b, a)
    assert result['n_shared'] == 0
    assert result['n_a'] and result['n_b']
    assert result['native_loss'] == pytest.approx(reverse['native_loss'])
    assert result['compatible']


def test_missing_columns_do_not_add_states_or_evidence():
    values = np.array([[2., -1., 3., -1.], [2., 1., -2., 1.]])
    obs = np.ones_like(values, bool)
    base = score_pairs(values, obs, [0], [1], [0], [4], [0], [3], [1], [4])
    wide = np.full((2, 8), np.nan)
    wide[:, ::2] = values
    mask = np.zeros_like(wide, bool); mask[:, ::2] = True
    got = score_pairs(wide, mask, [0], [1], [0], [8], [0], [6], [2], [8])
    assert got['native_loss'][0] == pytest.approx(base['native_loss'][0])
    assert got['n_union'][0] == 4


def test_no_information_is_not_shared_support():
    a = observation((4, 20), positions=np.array([], dtype=int))
    b = observation((4, 20))
    assert compare(a, b)['status'] == 'measurement_unavailable'
    equal = observation((4, 20), pa=.4, pp=.4)
    assert compare(equal, b)['status'] == 'measurement_unavailable'


def test_genomic_translation_and_parameter_validation():
    a, b = observation((4, 20)), observation((4, 14))
    moved = [FootprintObservation(o.positions + 12345678, o.hits, o.p_accessible,
                                 o.p_protected, tuple(v + 12345678 for v in o.interval))
             for o in (a, b)]
    assert compare(*moved)['native_loss'] == pytest.approx(compare(a, b)['native_loss'])
    for x in (-1, True, 1.5):
        with pytest.raises(ValueError):
            compare(a, b, minimum_edge_tolerance_bp=x)
    with pytest.raises(ValueError):
        compare(a, b, loss_odds=.9)
    assert compare(observation((1, 5)), observation((8, 12)))['status'] == 'nonoverlapping'


def test_strongly_distinguishable_substates_remain_separate():
    a = observation((2, 22), pa=.9, pp=.01)
    b = observation((2, 10), pa=.9, pp=.01)
    result = compare(a, b, minimum_edge_tolerance_bp=10)
    assert result['native_loss'] > math.log(1000)
    assert not result['compatible']
