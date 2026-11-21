"""Boundary numerics and limits of opportunity-based comparative diagnostics."""
import numpy as np
import pytest
from scipy.special import logsumexp

from fiberhmm.inference.consensus.lattice_recaller import model as M
from fiberhmm.inference.consensus.parameters import RecallerOptions


def test_stable_log_bf_agrees_with_odds_away_from_boundaries():
    rng = np.random.default_rng(812)
    matrix = rng.normal(0, 2, (100, 7))
    weights = rng.dirichlet(np.ones(7))
    logits = matrix + np.log(weights)
    post = np.exp(logits - logsumexp(logits, axis=1, keepdims=True))
    expected = np.log(post) - np.log1p(-post) - (np.log(weights) - np.log1p(-weights))
    np.testing.assert_allclose(M.log_bayes_factors(matrix, weights), expected, atol=1e-12)


@pytest.mark.parametrize('signal,verdict', [(-1000., 'non_member'), (-50., 'non_member'),
                                          (20., 'member'), (1000., 'member')])
def test_boundary_posteriors_do_not_erase_likelihood_evidence(signal, verdict):
    matrix = np.tile([signal, 0., 0., 0.], (60, 1))
    weights, post = M.em(matrix)
    lbf = M.log_bayes_factors(matrix, weights)
    assert lbf[:, 0] == pytest.approx(np.full(60, signal))
    for p, evidence in zip(post[:, 0], lbf[:, 0]):
        assert M.label(p, weights[0], 3., log_bf=evidence)[0] == verdict


def test_log_bf_is_invariant_to_row_offsets_and_uses_mixture_alternatives():
    matrix = np.array([[1000., 999., -1000.], [-1000., -1001., -1002.]])
    weights = np.array([.2, .3, .5])
    actual = M.log_bayes_factors(matrix, weights)
    expected = matrix[:, 0] - logsumexp(matrix[:, 1:] + np.log(weights[1:]/weights[1:].sum()), axis=1)
    np.testing.assert_allclose(actual[:, 0], expected, atol=1e-12)
    np.testing.assert_allclose(actual, M.log_bayes_factors(matrix + np.array([[1e4], [-1e4]]), weights), atol=1e-11)


def test_neutral_evidence_remains_neutral_at_zero_or_one_weights():
    matrix = np.zeros((3, 4))
    for weights in (np.array([0., 0., 0., 1.]), np.array([1., 0., 0., 0.])):
        np.testing.assert_allclose(M.log_bayes_factors(matrix, weights), 0., atol=1e-12)


def unit(pos, hits, pa=.85):
    pos = np.asarray(pos)
    hits = np.asarray(hits, bool)
    p = np.full(len(pos), pa); q = np.full(len(pos), .02)
    return dict(uid='u', pos=pos, hit=hits, pa=p, pp=q,
                d=np.where(hits, np.log(q/p), np.log1p(-q)-np.log1p(-p)))


def test_interior_resolution_does_not_prove_geometric_identifiability():
    pos = np.arange(0, 200, 4)
    u = unit(pos, (pos < 80) | (pos >= 112))
    # Different bp edges, exactly the same observed protected targets.
    a = dict(L=[80,80], R=[112,112], span=[80,112])
    b = dict(L=[79,79], R=[111,111], span=[79,111])
    assert M.expected_evidence([u], a) > 5.
    assert M.expected_evidence([u], b) > 5.
    scores = M.joint(u, [a,b], 1., RecallerOptions())
    assert scores[0] == pytest.approx(scores[1])


def test_resolved_absent_haplotype_is_a_valid_comparison_arm():
    pos = np.arange(0, 200, 4)
    g = dict(L=[80,80], R=[112,112], span=[80,112])
    rng = np.random.default_rng(45)
    units = [dict(unit(pos, rng.random(len(pos)) < .85), uid=f'u{i}') for i in range(100)]
    result = M.fit_channel(units, [g], 1., RecallerOptions(learned_spots=False))
    assert result['resolution'][0] > 5.
    assert result['support_gain'][0] < 5.
    assert result['w'][0] < .02


def test_posterior_threshold_depends_on_channel_prevalence():
    def posterior(prior, bf):
        return prior * bf / (1 - prior + prior * bf)
    low, high = posterior(.01, 4), posterior(.3, 4)
    assert low < .5 < high
    assert M.label(low, .01, 3)[0] == M.label(high, .3, 3)[0] == 'member'


def test_ct_rich_ga_poor_lattices_can_have_different_resolution():
    pos = np.arange(0, 200, 4)
    g = dict(L=[80,80], R=[112,112], span=[80,112])
    ct = unit(pos, np.ones(len(pos)))
    ga_pos = pos[(pos < 80) | (pos >= 112)]
    ga = unit(ga_pos, np.ones(len(ga_pos)))
    assert M.expected_evidence([ct], g) > 5.
    assert M.expected_evidence([ga], g) == 0.


def test_opportunity_diagnostics_use_scored_units_and_ignore_observed_hits():
    pos = np.arange(0, 200, 4)
    g = dict(L=[80,80], R=[112,112], span=[80,112])
    opt = RecallerOptions(flank_bp=20)
    protected = unit(pos, np.zeros(len(pos)))
    accessible = unit(pos, np.ones(len(pos)))
    a = M.opportunity_diagnostics([protected], g, opt)
    assert a == M.opportunity_diagnostics([accessible], g, opt)
    assert a['median_span_opportunities'] == 8
    assert a['median_left_flank_opportunities'] == a['median_right_flank_opportunities'] == 5
    assert a['scored_span_accessible_nats'] == pytest.approx(M.expected_evidence([protected], g))
    expected = 8 * (.02*np.log(.02/.85) + .98*np.log(.98/.15))
    assert a['scored_span_protected_nats'] == pytest.approx(expected)
    assert set(M.opportunity_diagnostics([], g, opt).values()) == {None}


def test_fit_and_frozen_scoring_share_stable_labels_and_tiers():
    from fiberhmm.inference.consensus.lattice_recaller.frozen import score_channel
    pos = np.arange(0, 200, 4)
    g = dict(L=[80,80], R=[112,112], span=[80,112])
    units = [dict(unit(pos, (pos < 80) | (pos >= 112)), uid=f'u{i}') for i in range(60)]
    opt = RecallerOptions(learned_spots=False, edge_contraction=False)
    fit = M.fit_channel(units, [g], 1., opt)
    frozen = score_channel(units, [g], 1., opt, dict(boxes=[g], spots=[{}], edges=['']))
    matrix, _ = M.Scorer(units, [g], 1., opt).rows()
    expected = M.log_bayes_factors(matrix, fit['w'])
    for result in (fit, frozen):
        np.testing.assert_allclose(result['log_bf'], expected)
        assert result['tiers'] == M.prevalence_tiers(units, result['P'], result['w'], [g], [{}], 3., expected)
        assert all(M.label(p, result['w'][0], 3., log_bf=v)[0] == 'member'
                   for p, v in zip(result['P'][:, 0], result['log_bf'][:, 0]))


@pytest.mark.parametrize('variant', [80, 88])
def test_shared_variant_mask_equalizes_edges_and_core_tiers(variant):
    pos = np.arange(0, 200, 4)
    g = dict(L=[76,84], R=[108,116], span=[80,112])
    ref = (pos < 80) | (pos >= 112)
    alt = ref.copy(); alt[pos == variant] = True
    def score(p, h):
        u = unit(p, h)
        sc = M.Scorer([u], [g], 1., RecallerOptions())
        return M.molecule_edges(sc.items[0], 0, g), M.tier_match(u, g)
    before, changed = score(pos, ref), score(pos, alt)
    if variant == 80:
        assert before[0][2] == (77, 80) and changed[0][2] == (81, 84)
    else:
        assert before[1] is not None and changed[1] is None
    use = pos != variant
    assert score(pos[use], ref[use]) == score(pos[use], alt[use])
    assert int(((pos[use] >= 80) & (pos[use] < 112)).sum()) == 7


def test_saturated_evidence_reaches_written_outputs_and_frozen_transfer(tmp_path, monkeypatch):
    import csv
    import gzip
    from fiberhmm.inference.consensus.workflow import run_workflow
    from fiberhmm.inference.consensus.lattice_recaller import frozen
    from fiberhmm.inference.consensus.transfer import export_run
    from test_consensus_lattice_recaller import planted_payload
    # Inject a known likelihood boundary at the scorer seam. All downstream
    # EM, labelling, tiers, snapshot and file writers are the production paths.
    original = M.Scorer.rows
    def saturated(self, prof=None):
        matrix, keep = original(self, prof)
        assert matrix.shape[1] == 4
        matrix[:] = 0.; matrix[:, 0] = 20.
        return matrix, keep
    monkeypatch.setattr(M.Scorer, 'rows', saturated)
    payload = planted_payload()
    result = run_workflow(payload, {'compute': {'cores': 1},
                                   'recaller': {'learned_spots': False, 'edge_contraction': False}}, tmp_path/'run')
    catalog = export_run(tmp_path/'run', tmp_path/'frozen.json.gz')
    transferred, _ = frozen.apply_catalog(catalog, payload, tmp_path/'transfer', include_training=True, cores=1)
    assert M.label(1., 1., 3.)[0] == 'abstain'  # legacy failure this test guards
    for folder, value in ((tmp_path/'run', result), (tmp_path/'transfer', transferred)):
        with gzip.open(folder/'molecules.tsv.gz', 'rt') as handle:
            rows = list(csv.DictReader(handle, delimiter='\t'))
        assert rows and {r['label'] for r in rows} == {'member'}
        assert {float(r['log_bf']) for r in rows} == {20.}
        assert {r['tier'] for r in rows} == {'core'}
        assert all(r['edge_range'] for r in rows)
        assert value['manifest']['recaller']['label_semantics'] == M.LABEL_SEMANTICS
        assert all(r['prevalence'] == r['prevalence_edge'] == r['prevalence_loose'] == 1.
                   for r in value['recaller']['rows'])
        with open(folder/'classes.tsv') as handle:
            summaries = list(csv.DictReader(handle, delimiter='\t'))
        assert all(float(r['median_span_opportunities']) > 0 for r in summaries)
        calls = [c for record in value['datasets']['planted']['cr']['records'] for c in record.get('recaller_calls', [])]
        assert calls and all(c['tier'] == 'core' for c in calls)
    assert transferred['transfer']['label_semantics'] == M.LABEL_SEMANTICS


def test_floored_class_weight_labels_absent_channel_non_member_in_molecules_tsv(tmp_path):
    """End to end: a class bound on CT and absent on GA. The GA EM weight hits
    the floor, where clipped posterior odds cancelled to log-BF 0 and labelled
    every GA molecule "abstain"; the likelihood ratio labels them non_member."""
    import csv
    import gzip
    import json
    from test_consensus_lattice_recaller import _payload, _units_for
    from fiberhmm.inference.consensus.workflow import run_workflow
    ct = _units_for(80, [(1180, 1210, .5)], strands=('CT',), seed=7)
    ga = _units_for(80, [], strands=('GA',), seed=8)
    for i, u in enumerate(ga):
        u['unit_id'], u['read_name'] = f'ga{i:04d}', f'garead{i:04d}'
    res = run_workflow(_payload((1050, 1350), ct + ga),
                       {'cr': {'engine': 'lattice_recaller'}, 'recaller': {'learned_spots': False}}, tmp_path)
    rows = {r['channel']: r for r in res['recaller']['rows']}
    assert rows['d::CT']['supported'] and not rows['d::GA']['supported']
    assert rows['d::GA']['prevalence'] == 0.0
    with gzip.open(tmp_path / 'molecules.tsv.gz', 'rt') as fh:
        molecules = [r for r in csv.DictReader(fh, delimiter='\t') if r['channel'] == 'd::GA']
    assert len(molecules) == 80
    threshold = RecallerOptions().bf_threshold
    for r in molecules:
        # Legacy clipped posterior odds: abstain. Direct likelihood ratio: non_member.
        assert M.label(float(r['posterior']), rows['d::GA']['prevalence'], threshold)[0] == 'abstain'
        assert r['label'] == 'non_member' and float(r['log_bf']) <= -np.log(threshold)
    manifest = json.loads((tmp_path / 'manifest.json').read_text())
    assert manifest['recaller']['label_semantics'] == M.LABEL_SEMANTICS == 'likelihood_ratio_v2'
