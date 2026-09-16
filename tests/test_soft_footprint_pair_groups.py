"""Soft evidence aggregation without a perfect pairwise-compatibility clique."""
import itertools
import json
import math

import numpy as np
import pytest

from fiberhmm.inference.soft_footprint_pair_groups import cluster_signed_pairs


def run(weights, starts=None, ends=None, **kwargs):
    weights = np.asarray(weights, dtype=float)
    count = len(weights)
    a, b = np.triu_indices(count, 1)
    keep = np.isfinite(weights[a, b])
    a, b = a[keep], b[keep]
    confidence = kwargs.pop('confidence', .999)
    threshold = math.log(confidence / (1-confidence))
    return cluster_signed_pairs(np.zeros(count, dtype=int) if starts is None else starts,
                                np.full(count, 100, dtype=int) if ends is None else ends,
                                a, b, threshold - weights[a, b], confidence,
                                observation_ids=[f'observation_{i:04d}' for i in range(count)], **kwargs)


def test_isolated_conflict_does_not_fragment_dense_concordant_population():
    weights = np.full((20, 20), 4.)
    weights[0, 1] = weights[1, 0] = -15.
    labels, diagnostics = run(weights)
    assert len(set(labels)) == 1
    assert diagnostics['within_negative_residual_pairs'] == 1
    assert diagnostics['node_within_negative_count'][:2] == [1, 1]
    assert diagnostics['objective'] == pytest.approx(190*4-19)
    assert diagnostics['selected_start_converged']
    assert min(diagnostics['node_alternative_margin']) > 0


def test_two_coherent_blocks_do_not_single_link_through_ambiguous_bridge():
    count = 21
    weights = np.full((count, count), -8.)
    weights[:10, :10] = weights[10:20, 10:20] = 4.
    weights[20, :] = weights[:, 20] = 1.
    labels, diagnostics = run(weights)
    assert len(set(labels[:10])) == len(set(labels[10:20])) == 1
    assert labels[0] != labels[10]
    assert labels[20] in (labels[0], labels[10])
    assert diagnostics['node_alternative_margin'][20] == pytest.approx(0)
    assert diagnostics['within_negative_residual_pairs'] == 0


def test_unknown_edges_are_neither_support_nor_automatic_separation():
    weights = np.full((3, 3), np.nan)
    weights[0, 1] = weights[1, 0] = 5.
    labels, diagnostics = run(weights)
    assert labels[0] == labels[1] != labels[2]
    assert diagnostics['tested_pairs'] == 1
    assert diagnostics['node_within_tested_count'][2] == 0
    assert diagnostics['node_alternative_margin'][2] == 0
    # A-C is unknown, but supported A-B and B-C can place all three in a
    # coherent population. This is explicitly not an evidence clique.
    weights[1, 2] = weights[2, 1] = 5.
    labels, diagnostics = run(weights)
    assert len(set(labels)) == 1
    assert diagnostics['within_tested_pairs'] == 2


def test_spatial_anchor_prevents_grouping_disjoint_sites_by_a_bridge():
    weights = np.full((3, 3), np.nan)
    weights[0, 1] = weights[1, 0] = 5.
    weights[1, 2] = weights[2, 1] = 5.
    labels, diagnostics = run(weights, starts=[0, 5, 10], ends=[10, 15, 20])
    assert labels[0] != labels[2]
    assert len(set(labels)) == 2
    assert all(a['common_start'] < a['common_end'] for a in diagnostics['class_anchors'])


def test_all_negative_edges_keep_singletons_and_all_nodes_are_retained():
    labels, diagnostics = run(np.full((12, 12), -3.))
    assert len(labels) == len(set(labels)) == 12
    assert min(labels) == 1
    assert diagnostics['objective'] == 0
    assert diagnostics['observations_retained'] == 12
    json.dumps(diagnostics, allow_nan=False)


def test_node_and_pair_permutation_invariance_with_stable_observation_ids():
    rng = np.random.default_rng(387)
    count = 23
    a, b = np.triu_indices(count, 1)
    bf = rng.normal(6, 10, len(a))
    starts = rng.integers(0, 10, count)
    ends = starts + 25
    identities = np.array([f'physical-observation-{i}' for i in range(count)])
    first, diag_a = cluster_signed_pairs(starts, ends, a, b, bf, .999, observation_ids=identities)
    order = rng.permutation(count)
    inverse = np.empty(count, dtype=int)
    inverse[order] = np.arange(count)
    pair_order = rng.permutation(len(a))
    second, diag_b = cluster_signed_pairs(starts[order], ends[order], inverse[b[pair_order]], inverse[a[pair_order]],
                                          bf[pair_order], .999, observation_ids=identities[order])
    assert np.array_equal(first[order], second)
    assert diag_a['objective'] == diag_b['objective']
    assert diag_a['starts'] == diag_b['starts']
    assert np.array(diag_a['node_alternative_margin'])[order] == pytest.approx(diag_b['node_alternative_margin'])


def test_objective_is_exact_saved_edge_sum_and_passes_are_monotonic():
    rng = np.random.default_rng(487)
    weights = rng.normal(0, 6, (45, 45))
    labels, diagnostics = run(weights)
    a, b = np.triu_indices(len(weights), 1)
    assert diagnostics['objective'] == pytest.approx(weights[a[labels[a] == labels[b]], b[labels[a] == labels[b]]].sum())
    assert diagnostics['objective'] == pytest.approx(max(s['objective'] for s in diagnostics['starts']), abs=1e-9)
    for start in diagnostics['starts']:
        assert np.all(np.diff(start['objective_by_pass']) >= -1e-8)
    if diagnostics['selected_start_converged']:
        assert max(diagnostics['node_best_alternative_gain']) < 1e-8


def test_pass_budget_is_explicit_not_misreported_as_convergence():
    labels, diagnostics = run(np.full((10, 10), 3.), max_passes=1)
    assert len(set(labels)) == 1
    assert not diagnostics['selected_start_converged']
    assert all(s['passes'] == 1 for s in diagnostics['starts'])


def test_empty_singleton_and_identical_deterministic_runs():
    labels, diagnostics = cluster_signed_pairs([], [], [], [], [], .99, observation_ids=[])
    assert len(labels) == 0 and diagnostics['classes'] == 0
    labels, diagnostics = run(np.zeros((1, 1)))
    assert labels.tolist() == [1]
    first = run(np.full((9, 9), 2.))
    second = run(np.full((9, 9), 2.))
    assert np.array_equal(first[0], second[0]) and first[1] == second[1]


def test_invalid_and_duplicated_evidence_is_never_silently_reweighted():
    with pytest.raises(ValueError, match='Duplicate'):
        cluster_signed_pairs([0, 0], [10, 10], [0, 1], [1, 0], [2., 2.], .99)
    with pytest.raises(ValueError, match='self-paired'):
        cluster_signed_pairs([0], [10], [0], [0], [2.], .99)
    with pytest.raises(ValueError, match='finite'):
        cluster_signed_pairs([0, 0], [10, 10], [0], [1], [np.inf], .99)
    with pytest.raises(ValueError, match='unique'):
        cluster_signed_pairs([0, 0], [10, 10], [0], [1], [2.], .99, observation_ids=['x', 'x'])
