import copy
import json
import numpy as np
import pytest

from fiberhmm.inference.consensus.measurement_family import (
    profile_comparison, classify_family_profiles, _native_values)


def test_exact_profile_kernel_including_single_edge_relaxation():
    rng = np.random.default_rng(821)
    for n in range(2, 8):
        aa, bb = np.triu_indices(n+1, 1)
        for _ in range(12):
            t, r = rng.normal(size=(2, len(aa)))
            valid = rng.random((2, len(aa))) > .2
            if not (valid[0] & valid[1]).any():
                continue
            out = profile_comparison(t, r, aa, bb,
                training_allowed=valid[0], recipient_allowed=valid[1])
            total = t[valid[0]].max()+r[valid[1]].max()
            shared = max(t[i]+r[i] for i in range(len(aa)) if valid[0, i] and valid[1, i])
            left = max(t[i]+r[j] for i in range(len(aa)) for j in range(len(aa))
                       if valid[0, i] and valid[1, j] and bb[i] == bb[j])
            right = max(t[i]+r[j] for i in range(len(aa)) for j in range(len(aa))
                        if valid[0, i] and valid[1, j] and aa[i] == aa[j])
            assert out['native_loss'] == pytest.approx(total-shared)
            assert out['left_edge_relaxed_loss'] == pytest.approx(total-left)
            assert out['right_edge_relaxed_loss'] == pytest.approx(total-right)
            reverse = profile_comparison(r, t, aa, bb, training_allowed=valid[1], recipient_allowed=valid[0])
            assert reverse['native_loss'] == pytest.approx(out['native_loss'])


def test_duplicate_geometry_rejected_and_no_common_geometry_explicit():
    with pytest.raises(ValueError, match='aliases'):
        profile_comparison([1., 2.], [2., 1.], [0, 0], [1, 1])
    out = profile_comparison([1., 2.], [2., 1.], [0, 1], [1, 2],
                             training_allowed=[True, False], recipient_allowed=[False, True])
    assert out['status'] == 'no_common_projection'
    assert out['native_loss'] is None


def fixture():
    units = []
    p = np.arange(0, 81, 3)
    for i, (a, b) in enumerate([(12, 51), (11, 52), (12, 50), (12, 27), (11, 28), (12, 26)]):
        h = ((p < a) | (p >= b)).astype(int)
        units.append(dict(unit_id=f'u{i}', strand='CT', positions=p.tolist(), hits=h.tolist(),
            p_accessible=np.full(len(p), .85).tolist(), p_protected=np.full(len(p), .02).tolist(),
            representative_raw_tf_intervals=[[a, b]], raw_nuc_intervals=[]))
    catalog = [dict(family='d:broad', consensus_start=12, consensus_end=51),
               dict(family='d:small', consensus_start=12, consensus_end=27)]
    return dict(dataset_id='d', units=units), catalog


def test_native_models_keep_full_spans_and_distinct_broad_small_states():
    s, cat = fixture(); frozen = copy.deepcopy(s)
    result = classify_family_profiles(s, cat, region=(0, 81))
    assigned = {r['unit_id']: r for r in result['partitions']['100.0']['assignments']}
    for i in range(6):
        assert assigned[f'u{i}']['family'] == ('d:broad' if i < 3 else 'd:small')
        assert assigned[f'u{i}']['interval'] == s['units'][i]['representative_raw_tf_intervals'][0]
        assert assigned[f'u{i}']['primary_evidence']['own_evidence_excluded']
        assert assigned[f'u{i}']['primary_evidence']['source_units'] == 2
    assert s == frozen
    json.dumps(result, allow_nan=False)


def test_same_group_never_trains_itself_and_all_source_ordinals_survive():
    s, cat = fixture()
    for u in s['units']:
        u['fold_group_id'] = 'one_group'
    out = classify_family_profiles(s, cat, region=(0, 81))
    assert out['partitions']['100.0']['unresolved_calls'] == 6
    assert out['partitions']['100.0']['all_source_calls_retained']


def test_order_invariance_and_no_silent_budget_truncation():
    s, cat = fixture()
    a = classify_family_profiles(s, cat, region=(0, 81))
    s['units'].reverse()
    b = classify_family_profiles(s, cat[::-1], region=(0, 81))
    for level in a['partitions']:
        labels = lambda obj: {v['unit_id']: v['family'] for v in obj['partitions'][level]['assignments']}
        assert labels(a) == labels(b)
    with pytest.raises(MemoryError, match='no calls/projections removed'):
        classify_family_profiles(s, cat, region=(0, 81), maximum_matrix_bytes=1)


def test_neighbor_conditioning_keeps_own_core_observations():
    s, _ = fixture(); u = s['units'][0]
    u['representative_raw_tf_intervals'].append([45, 75])
    values, observed = _native_values(u, np.asarray(u['positions']), dict(start=12, end=51))
    assert observed[16]  # position 48 remains inside the source call
    assert not observed[20]  # position 60 is a frozen neighboring footprint
    assert values[20] == 0


def test_depth_does_not_multiply_recipient_evidence():
    aa, bb = np.triu_indices(4, 1)
    t = np.array([0., 1., 4., 0., 2., 0.])
    r = np.array([0., 1., 1., 4., 2., 0.])
    losses = [profile_comparison(n*t, r, aa, bb)['native_loss'] for n in (1, 10, 100, 1000)]
    assert max(losses) <= r.max()-r.min()+1e-10
    assert losses[-1] == pytest.approx(losses[-2])


def test_untruncated_gaussian_location_cannot_replace_actual_geometry(monkeypatch):
    from fiberhmm.inference.consensus import measurement_distribution as md
    real_fit = md.fit_native_distribution

    def fit_with_unobserved_mean(*args, **kwargs):
        fit = real_fit(*args, **kwargs)
        # Deliberately wrong untruncated location. The normalized positive-cell
        # model and its recipient scoring/simulation are unchanged.
        fit['center'] = np.array([75., 78.])
        return fit

    monkeypatch.setattr(md, 'fit_native_distribution', fit_with_unobserved_mean)
    s, cat = fixture()
    out = classify_family_profiles(s, cat, region=(0, 81), family_model='latent_distribution',
        minimum_edge_tolerance_bp=10, predictive_replicates=31)
    tested = 0
    for summaries in out['call_family_evidence']:
        for evidence in summaries:
            if evidence['status'] != 'scored':
                continue
            tested += 1
            assert evidence['core_geometry_summary'] == 'normalized_positive_cell_coverage'
            assert evidence['core_geometry_mean'] != [75., 78.]
            expected = max(0., evidence['recipient_optimum']-evidence['selected_native_log_lr']
                           + evidence['selected_model_shape_penalty'])
            assert evidence['floor_adjusted_loss'] == pytest.approx(expected)
            if expected <= 1e-10:
                assert evidence['predictive_tail'] == 1.
                assert evidence['simulations'] == 0
    assert tested > 0
    for model in out['family_models']:
        if model['status'] == 'fitted':
            assert set(model['fold_models']) == set(model['training_evidence_groups'])
            assert all(len(v['parameters']) == 5 for v in model['fold_models'].values())


def test_predictive_score_not_overwritten_when_actual_mean_core_unobserved():
    s, cat = fixture()
    sparse = copy.deepcopy(s['units'][0]); sparse['unit_id'] = 'sparse'
    p = np.array([0, 3, 6, 57, 60, 63, 66, 69, 72, 75, 78])
    sparse.update(positions=p.tolist(), hits=np.ones(len(p), int).tolist(),
        p_accessible=np.full(len(p), .85).tolist(), p_protected=np.full(len(p), .02).tolist())
    # Retain an observed flank in the union domain; no observations anywhere
    # in the domain would correctly be no_recipient_information, not scored.
    s['units'][0]['representative_raw_tf_intervals'] = [[6, 57]]
    s['units'].append(sparse)
    out = classify_family_profiles(s, cat, region=(0, 81), family_model='latent_distribution',
        minimum_edge_tolerance_bp=10, predictive_replicates=31)
    tested = 0
    for call, evidence in zip(out['calls'], out['call_family_evidence']):
        if call['unit_id'] != 'sparse': continue
        for score in evidence:
            if score['status'] != 'scored': continue
            tested += 1
            assert score['mean_core_unobserved']
            expected = max(0., score['recipient_optimum']-score['selected_native_log_lr']
                           + score['selected_model_shape_penalty'])
            assert score['floor_adjusted_loss'] == pytest.approx(expected)
            if expected <= 1e-10:
                assert score['predictive_tail'] == 1.
                assert score['simulations'] == 0
    assert tested > 0
