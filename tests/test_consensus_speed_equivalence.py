"""Performance changes must preserve evidence, multiplicity, and decisions."""
from copy import deepcopy

import numpy as np
from numba import njit
import pytest

from fiberhmm.inference.consensus import lattice as lm
from fiberhmm.inference.consensus.measurement_distribution import _predictive_exceedances
from fiberhmm.inference.consensus.measurement_family import classify_family_profiles, _native_values


@njit
def old_predictive(pa, pp, starts, ends, penalty, cdf, threshold, replicates, seed):
    """Frozen pre-optimization arithmetic and RNG order (no reference import)."""
    np.random.seed(seed)
    count = 0
    for _ in range(replicates):
        g = np.searchsorted(cdf, np.random.random())
        prefix = np.zeros(len(pa)+1)
        for j in range(len(pa)):
            p = pp[j] if starts[g] <= j < ends[g] else pa[j]
            hit = np.random.random() < p
            step = np.log(pp[j]/pa[j]) if hit else np.log1p(-pp[j])-np.log1p(-pa[j])
            prefix[j+1] = prefix[j]+step
        best = -np.inf; explained = -np.inf
        for h in range(len(starts)):
            value = prefix[ends[h]]-prefix[starts[h]]
            best = max(best, value)
            explained = max(explained, value+penalty[h])
        count += best-explained >= threshold-1e-10
    return count


@pytest.mark.parametrize('n', [1, 5, 30])
@pytest.mark.parametrize('seed', [1, 391, 4294967295])
def test_simulation_hoisting_preserves_every_exceedance(n, seed):
    rng = np.random.default_rng(seed)
    pa = rng.uniform(.25, .99, n); pp = rng.uniform(.001, .2, n)
    aa, bb = np.triu_indices(n+1, 1)
    q = rng.uniform(.01, 1, len(aa)); cdf = np.cumsum(q/q.sum()); cdf[-1] = 1
    penalty = -rng.uniform(0, 9, len(aa))
    for threshold in (0., .5, 2., 7.):
        args = (pa, pp, aa, bb, penalty, cdf, threshold, 4095, seed)
        assert _predictive_exceedances(*args) == old_predictive(*args)


@pytest.mark.parametrize('restricted', [False, True])
def test_cached_fit_objective_and_gradient_are_bit_identical(monkeypatch, restricted):
    rng = np.random.default_rng(654)
    k = lm.RegionFamilyLattice(np.arange(12), [[1, 5], [4, 8], [8, 11]], 1)
    pattern = rng.normal(size=(4, 12))
    v = pattern[[1, 2, 2, 3, 0, 1, 1, 0, 3, 2, 0]]
    allowed = np.ones((len(v), 3), bool); allowed[1, 1] = False; allowed[-2:] = False
    kw = dict(allowed=allowed)
    if restricted:
        masks = np.ones((len(v), len(k.ga)), bool); masks[0, :3] = False
        adjustment = np.zeros(masks.shape); adjustment[4, :2] = -.1
        kw.update(geometry_allowed=masks, geometry_log_adjustment=adjustment)
    reference = lm.minimize
    evaluated = []
    def checked(fun, initial, **options):
        def objective(eta):
            actual = fun(eta); expected = k.objective(v, eta, **kw)
            assert actual[0] == expected[0]
            np.testing.assert_array_equal(actual[1], expected[1])
            evaluated.append(eta.copy())
            return actual
        return reference(objective, initial, **options)
    monkeypatch.setattr(lm, 'minimize', checked)
    k.fit(v, max_iter=30, **kw)
    assert len(evaluated) > 10


def test_serial_and_parallel_dispatchers_have_separate_cache_and_exact_results():
    from numba import get_num_threads, set_num_threads
    k = lm.RegionFamilyLattice(np.arange(13), [[0, 4], [3, 9], [8, 12]], 2)
    rng = np.random.default_rng(91); v = rng.normal(size=(15, 13)); eta = np.array([-.7, .4, .2])
    allowed = rng.random((15, 3)) > .3; allowed[-1] = False
    masks = rng.random((15, len(k.ga))) > .2
    inputs = k._evaluation_inputs(v, allowed, masks, np.log(rng.uniform(.1, 1, masks.shape)))
    args = (*inputs[:1], eta, *inputs[1:], k.offsets, k.dest, k.edge_geo,
            k.ga, k.gb, k.gf, k.logq, k.n_nodes, True)
    serial = lm._evaluate_serial(*args)
    for value, got in zip(serial, lm._evaluate(*args)):
        np.testing.assert_array_equal(value, got)
    assert lm._evaluate.py_func.__qualname__ != lm._evaluate_serial.py_func.__qualname__
    assert lm._evaluate._cache._cache_file._index_path != lm._evaluate_serial._cache._cache_file._index_path
    assert lm._evaluate.targetoptions['parallel']
    assert not lm._evaluate_serial.targetoptions.get('parallel', False)


@pytest.mark.parametrize('fail', [False, True])
def test_native_entry_point_honors_and_restores_worker_budget(monkeypatch, fail):
    from numba import get_num_threads
    from fiberhmm.inference.consensus import native_workflow as nw
    from fiberhmm.inference.consensus.parameters import parse_options
    previous = get_num_threads()
    def work(*args):
        assert get_num_threads() == 1
        if fail: raise RuntimeError('test cancellation')
        return 'complete'
    monkeypatch.setattr(nw, '_run_native_workflow', work)
    options = parse_options(dict(compute=dict(cores=1)))
    if fail:
        with pytest.raises(RuntimeError, match='cancellation'): nw.run_native_workflow({}, options)
    else:
        assert nw.run_native_workflow({}, options) == 'complete'
    assert get_num_threads() == previous


def test_append_skips_old_fits_but_keeps_augmented_source_homes_and_scores(monkeypatch):
    from test_consensus_measurement_family import fixture
    from fiberhmm.inference.consensus import measurement_distribution as md
    s, catalog = fixture()
    kw = dict(region=(0, 81), family_model='latent_distribution', scoring_folds=2,
              predictive_replicates=31, max_fit_iterations=25)
    initial = classify_family_profiles(s, catalog[1:], **kw)
    full = classify_family_profiles(s, catalog, **kw)
    original = deepcopy(initial); observed = []; real_fit = md.fit_native_distribution
    def record(*args, **kwargs):
        observed.append(kwargs['reference'].tolist())
        return real_fit(*args, **kwargs)
    monkeypatch.setattr(md, 'fit_native_distribution', record)
    appended = classify_family_profiles(s, catalog, _frozen_result=initial, **kw)
    assert initial == original
    assert len(observed) == 3  # full plus two excluded folds, NEW model only
    assert all(v == [12, 51] for v in observed)
    assert appended['source_homes'] == full['source_homes']
    assert appended['family_models'][0] == full['family_models'][0]
    assert appended['family_models'][1] == initial['family_models'][0]
    for actual, expected, before in zip(appended['call_family_evidence'], full['call_family_evidence'], initial['call_family_evidence']):
        assert [v for v in actual if v['family'] == 'd:broad'] == [v for v in expected if v['family'] == 'd:broad']
        assert [v for v in actual if v['family'] == 'd:small'] == before


def test_observation_cache_matches_original_neighbor_conditioning():
    from test_consensus_measurement_family import fixture
    s, _ = fixture(); u = s['units'][0]
    u['representative_raw_tf_intervals'] += [[45, 75], [12, 51]]
    u['raw_nuc_intervals'] = [[0, 16], [60, 81]]
    rng = np.random.default_rng(843)
    p = np.asarray(u['positions']); h = np.asarray(u['hits'])
    pa = rng.uniform(.3, .95, len(p)); pp = rng.uniform(.001, .1, len(p))
    u['p_accessible'] = pa.tolist(); u['p_protected'] = pp.tolist()
    for a, b in u['representative_raw_tf_intervals']:
        frozen = np.zeros(len(p), bool)
        for x, y in u['representative_raw_tf_intervals']+u['raw_nuc_intervals']:
            if [x, y] != [a, b]: frozen |= (p >= x) & (p < y)
        keep = ~(frozen & ~((p >= a) & (p < b)))
        expected = np.zeros(len(p))
        expected[keep] = np.where(h[keep], np.log(pp[keep]/pa[keep]), np.log1p(-pp[keep])-np.log1p(-pa[keep]))
        values, mask = _native_values(u, p, dict(start=a, end=b))
        np.testing.assert_array_equal(mask, keep)
        np.testing.assert_array_equal(values, expected)


def test_cached_member_union_keeps_all_coordinates_and_ignores_duplicate_ordinals():
    from fiberhmm.inference.consensus.native_cross import _member_opportunities
    data = dict(units=[dict(positions=list(range(0, 200000, 3))),
                       dict(positions=[0, 2, 5, 17, 180001, 200002])],
                result=dict(calls=[dict(unit_index=0),dict(unit_index=1),dict(unit_index=0)]))
    region = (5, 190000)
    for indices in ([], [1], [0, 1, 2], [2, 0, 1, 0]):
        expected = np.empty(0, np.int64)
        for i in indices:
            expected = np.union1d(expected, data['units'][data['result']['calls'][i]['unit_index']]['positions'])
        expected = expected[(expected >= region[0]) & (expected < region[1])]
        np.testing.assert_array_equal(_member_opportunities(data,indices,region),expected)


def test_single_observation_build_also_supplies_the_exact_neighbor_only_mask():
    from test_consensus_native_cross import unit, frozen
    from fiberhmm.inference.consensus.native_cross import _recipient_observations, refine_recipient_call_grid
    p = np.arange(1,60,2); model = frozen(p)
    u = unit(np.zeros(len(p)), spans=((12, 27),(40, 58)))
    call = dict(unit_id='u', ordinal=0, start=12, end=27, strand='CT')
    grid = refine_recipient_call_grid(u,call,model['grid'])
    a = _recipient_observations(u,call,grid)
    b = _recipient_observations(u,call,grid,require_call_overlap=False)
    np.testing.assert_array_equal(a['neighbor_allowed'],b['allowed'])
    for key in ('likelihood','observed','values','p_accessible','p_protected'):
        np.testing.assert_array_equal(a[key],b[key])
