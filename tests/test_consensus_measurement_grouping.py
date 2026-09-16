import copy
import numpy as np
import pytest

from fiberhmm.inference.consensus.measurement_grouping import group_observations


def fixture(pa):
    pos = np.arange(32)
    units = []
    for i, span in enumerate([[3, 27]]*3+[[3, 12]]*3):
        hits = ~((pos >= span[0]) & (pos < span[1]))
        units.append(dict(unit_id=f'unit{i}', fold_group_id=f'unit{i}', strand='CT' if i % 2 else 'GA',
                          positions=pos.tolist(), hits=hits.astype(int).tolist(),
                          p_accessible=[pa]*len(pos), p_protected=[.001]*len(pos),
                          representative_raw_tf_intervals=[span]))
    return dict(dataset_id='test', chemistry='dddb', units=units)


def test_every_call_survives_and_low_efficiency_is_not_more_fragmented():
    source = fixture(.2); before = copy.deepcopy(source)
    low = group_observations(source)
    high = group_observations(fixture(.9))
    assert source == before
    assert low['diagnostics']['tested_pairs'] == 15
    assert low['partitions']['100.0']['classes'] == 1
    assert high['partitions']['100.0']['classes'] == 2
    for run in (low, high):
        for part in run['partitions'].values():
            assert part['all_source_calls_retained']
            assert len(part['assignments']) == 6
            assert not part['boundaries_changed']
            for call in part['assignments']:
                assert call['interval'] == source['units'][call['unit_index']]['representative_raw_tf_intervals'][call['ordinal']]


def test_no_nonoverlap_chaining_or_same_unit_double_counting():
    source = fixture(.2)
    # Overlapping calls from the same evidence unit remain separate source
    # observations; their pair cannot manufacture a population merge.
    source['units'][0]['representative_raw_tf_intervals'].append([6, 29])
    source['units'][1]['representative_raw_tf_intervals'].append([29, 31])
    result = group_observations(source)
    assert result['diagnostics']['same_evidence_group_pairs_excluded'] == 1
    assert result['diagnostics']['source_calls'] == 8
    calls = result['partitions']['100.0']['assignments']
    u0 = [c for c in calls if c['unit_index'] == 0]
    assert len(u0) == 2 and u0[0]['family'] != u0[1]['family']
    for a in calls:
        for b in calls:
            if a['family'] == b['family']:
                assert a['start'] < b['end'] and b['start'] < a['end']


def test_explicit_budget_failure_no_silent_call_cap():
    with pytest.raises(MemoryError, match='No reads/classes were omitted'):
        group_observations(fixture(.4), maximum_matrix_bytes=100)


def test_grouping_is_invariant_to_input_unit_order():
    source = fixture(.4)
    one = group_observations(source)
    shuffled = {**source, 'units': list(reversed(source['units']))}
    two = group_observations(shuffled)
    key = lambda result: {c['unit_id']: c['family'] for c in result['partitions']['100.0']['assignments']}
    assert key(one) == key(two)


def test_comparison_only_uses_identical_evidence_without_clustering():
    full=group_observations(fixture(.4),minimum_edge_tolerance_bp=10)
    fast=group_observations(fixture(.4),minimum_edge_tolerance_bp=10,comparison_only=True)
    assert fast['partitions']=={}
    assert fast['calls']==full['calls']
    for key in full['pairs']:
        np.testing.assert_array_equal(full['pairs'][key],fast['pairs'][key])
