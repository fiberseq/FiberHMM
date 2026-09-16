"""Exact preparation reuse and earlier rejection preserve the full ledger."""
from copy import deepcopy

import numpy as np
import pytest

from fiberhmm.inference.consensus.cross_preparation import (
    ByteLRU, NativeReadCache, TransferGeometryCache)
from fiberhmm.inference.consensus import native_cross as nc


@pytest.mark.parametrize('floor', [0, 10])
@pytest.mark.parametrize('budget', [0, 1024, 2**20])
def test_cached_records_exact_under_hits_and_eviction(floor, budget):
    from test_consensus_native_cross import frozen, unit
    p = np.arange(1, 60, 2); source = frozen(p)
    reads = NativeReadCache(budget)
    geometries = TransferGeometryCache(source, source['grid'], floor, budget)
    rng = np.random.default_rng(420)
    units = [unit(rng.random(len(p)) < .5, spans=((12, 40), (45, 55))) for _ in range(12)]
    for i in list(range(12))*2:
        call = dict(unit_id=f'u{i}', ordinal=0, start=12, end=40, strand='GA')
        kw = dict(floor_bp=floor, replicates=127)
        expected = nc.transferred_call(source, source['grid'], units[i], call, **kw)
        actual = nc.transferred_call(source, source['grid'], units[i], call,
                                     _read_cache=reads, _geometry_cache=geometries, **kw)
        assert actual == expected
        assert reads.bytes <= budget and geometries.bytes <= budget
    if budget == 2**20:
        assert reads.hits >= 12 and geometries.hits >= 23


def test_cached_grid_binding_rejects_other_model_or_allowance():
    from test_consensus_native_cross import frozen, unit
    p = np.arange(1, 60, 2); source = frozen(p)
    cache = TransferGeometryCache(source, source['grid'], 0, 2**20)
    call = dict(unit_id='u', ordinal=0, start=12, end=40, strand='CT')
    with pytest.raises(ValueError, match='cannot be reused'):
        nc.transferred_call(source, source['grid'], unit(np.zeros(len(p))), call,
                            floor_bp=10, _geometry_cache=cache)
    with pytest.raises(ValueError, match='cannot be reused'):
        nc.transferred_call(deepcopy(source), source['grid'], unit(np.zeros(len(p))), call,
                            _geometry_cache=cache)


def test_preparation_early_reject_does_not_construct_geometry_likelihood(monkeypatch):
    from test_consensus_native_cross import frozen, unit
    p = np.arange(1, 60, 2); source = frozen(p, reference=(12, 22))
    source['model']['fold_models']['full']['parameters'] = [0., 0., np.log(10.), 0., np.log(10.)]
    call = dict(unit_id='u', ordinal=0, start=40, end=50, strand='CT')
    u = unit(np.zeros(len(p)), spans=((40, 50),))
    def forbidden(*args, **kwargs):
        raise AssertionError('Rejected transfer need not construct full likelihood')
    monkeypatch.setattr(nc, '_recipient_observations', forbidden)
    result = nc.transferred_call(source, source['grid'], u, call, replicates=127)
    assert result['status'] == 'hypothesis_not_attributable_to_this_call'
    assert result['observed_opportunities'] == len(p)
    # No-observation precedence over geometric rejection is unchanged.
    u.update(positions=np.array([70]), hits=np.array([0]),
             p_accessible=np.array([.6]), p_protected=np.array([.01]))
    result = nc.transferred_call(source, source['grid'], u, call, replicates=127)
    assert result['status'] == 'no_recipient_information'


def test_grid_cache_does_not_mix_distinct_neighbours_lattices_or_raw_edges():
    from test_consensus_native_cross import frozen, unit
    p = np.arange(1, 60, 2); source = frozen(p)
    cache = TransferGeometryCache(source, source['grid'], 10, 2**20)
    cases = []
    for span, neighbor in (((12, 40), (45, 55)), ((12, 40), (42, 55)), ((14, 40), (45, 55))):
        u = unit(np.zeros(len(p)), spans=(span, neighbor))
        cases.append((u, dict(unit_id='u', ordinal=0, start=span[0], end=span[1], strand='CT')))
    u, call = deepcopy(cases[0])
    for key in ('positions', 'hits', 'p_accessible', 'p_protected'):
        u[key] = np.asarray(u[key])[::2]
    cases.append((u, call))
    for u, call in cases:
        expected = nc.transferred_call(source, source['grid'], u, call, floor_bp=10, replicates=127)
        actual = nc.transferred_call(source, source['grid'], u, call, floor_bp=10,
                                    replicates=127, _geometry_cache=cache)
        assert expected == actual
    # Some lattices legitimately refine to exactly the same grid. Observed
    # masks remain per-read even when that geometry is shared.


def test_lru_admission_recomputes_oversize_and_retains_only_budgeted_entries():
    cache = ByteLRU(10)
    assert cache.put('a', [1], 6) == [1]
    cache.put('b', [2], 6)
    assert cache.get('a') is None and cache.get('b') == [2]
    assert cache.put('large', [3], 20) == [3]
    assert cache.get('large') is None and cache.bytes == 6
    cache.put('b', [4], 3)
    assert cache.get('b') == [4] and cache.bytes == 3


def test_prefix_visibility_and_sorted_neighbours_match_exhaustive_masks():
    from test_consensus_native_cross import unit
    rng = np.random.default_rng(732)
    u = unit(np.zeros(30), spans=((3, 12), (8, 18), (22, 24), (25, 42), (50, 59)))
    cache = NativeReadCache(2**20)
    native = cache.native(u); p = native[0]
    for _ in range(100):
        lo, hi = sorted(rng.choice(np.arange(-5, 66), 2, replace=False))
        a, b = sorted(rng.choice(np.arange(-5, 66), 2, replace=False))
        call = dict(start=int(a), end=int(b))
        actual = cache.visible_count(u, call, lo, hi)
        visible = (p >= lo) & (p <= hi) & ~(native[2] & ~((p >= a) & (p < b)))
        assert actual == int(visible.sum())
        previous, following = int(lo), int(hi)
        for x, y in u['representative_raw_tf_intervals']:
            if y <= a: previous = max(previous, y)
            if x >= b: following = min(following, x)
        assert cache.limits(u, call, (lo, hi)) == (previous, following)


def test_read_cache_owns_readonly_arrays_without_mutating_caller_flags():
    from test_consensus_native_cross import unit
    u = unit(np.zeros(30)); cache = NativeReadCache(2**20)
    arrays = cache.native(u)
    for a in arrays:
        assert not a.flags.writeable
    assert u['positions'].flags.writeable and u['p_accessible'].flags.writeable
    with pytest.raises(ValueError): arrays[0][0] = 500
    # Subsequent runs use new snapshots, not a stale process-global cache.
    u['p_accessible'][0] = .7
    assert NativeReadCache(2**20).native(u)[3][0] == .7
    assert arrays[3][0] == .6
