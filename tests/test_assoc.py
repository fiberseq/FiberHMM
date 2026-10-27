"""fiberhmm.assoc (feature hub association statistics): M1-M3, resampling, conditional logistic, groups.

Synthetic data only. The calibration gate itself (2,000 confounded nulls per mode) is evidence outside the suite;
these tests check the machinery and small planted effects.
"""
import itertools

import numpy as np
import pytest

from fiberhmm.assoc import resample as R
from fiberhmm.assoc import strata as ST
from fiberhmm.assoc.background import candidate_positions, matched_backgrounds
from fiberhmm.assoc.clogit import clogit, score_bootstrap
from fiberhmm.assoc.condition import Appearances, LocusTable, condition, mean_metric
from fiberhmm.assoc.cooccur import FeatureStates, MoleculeTable, cooccurrence, find_pairs, fold_states
from fiberhmm.assoc.groups import group_contrast, group_labels, post_selection_check
from fiberhmm.assoc.rate import RateData, compress, edge_association, rate_ratio
from fiberhmm.assoc.simulate import simulate_m1, simulate_m2, simulate_m3


# ---------------------------------------------------------------- resampling and dependence
def test_locus_groups_join_units_that_share_molecules_transitively():
    # units 0-1 share molecule 10, 1-2 share 11, unit 3 alone
    unit = [0, 0, 1, 1, 2, 3]
    mol = [10, 20, 10, 11, 11, 30]
    g = R.locus_groups(unit, mol, 4)
    assert g[0] == g[1] == g[2] and g[3] != g[0]
    assert list(g) == [0, 0, 0, 1]


def test_block_bootstrap_never_cuts_a_group_and_falls_back_when_one_group_dominates():
    unit = np.repeat(np.arange(40), 3)
    mol = np.arange(120)
    g = R.locus_groups(unit, mol, 40)
    bb = R.BlockBootstrap(g, unit, mol, seed=1)
    assert bb.mode == 'groups'
    W = bb.item_weights(5)
    # items of one unit always share a weight
    for u in range(40):
        assert np.all(W[:, unit == u] == W[:, unit == u][:, :1])
    # one molecule in every unit: a single group -> fallback
    mol2 = np.where(np.arange(120) % 3 == 0, 999, np.arange(120))
    g2 = R.locus_groups(unit, mol2, 40)
    bb2 = R.BlockBootstrap(g2, unit, mol2, seed=1)
    assert bb2.mode == 'twoway' and bb2.largest_share == 1.0
    bb3 = R.BlockBootstrap(g2, unit, mol2, seed=1, fallback='pigeonhole')
    assert bb3.item_weights(3).shape == (3, 120)


def test_twoway_variance_matches_the_mean_variance_for_independent_items():
    rng = np.random.default_rng(0)
    n = 400
    x = rng.normal(size=n)
    unit = np.arange(n) // 2
    mol = np.arange(n)
    g = np.zeros(n // 2, np.int64)          # one group -> twoway
    bb = R.BlockBootstrap(g, unit, mol, seed=3)
    boot = bb.collect(lambda W: {'m': (W @ x) / W.sum(1)}, 400)
    se = boot.summary('m', x.mean())['se']
    assert abs(se - x.std() / np.sqrt(n)) / (x.std() / np.sqrt(n)) < 0.25


def test_perm_p_counts_ties_and_random_effects_pool_replicates():
    assert R.perm_p(1.0, [1.0, 0.0, 2.0], 'two') == pytest.approx(3 / 4)
    re = R.random_effects([0.5, 0.7, 0.6], [0.1, 0.1, 0.1])
    assert re['k'] == 3 and 0.5 < re['estimate'] < 0.7 and re['tau2'] == 0
    assert np.isnan(R.bh([0.01, np.nan, 0.04])[1])


# ---------------------------------------------------------------- conditional logistic
def test_clogit_equals_brute_force_conditional_likelihood():
    from scipy.optimize import minimize
    rng = np.random.default_rng(4)
    X, y, st = [], [], []
    for s in range(25):
        n = int(rng.integers(2, 7))
        x = rng.normal(size=(n, 2))
        yy = (rng.random(n) < 1 / (1 + np.exp(-(rng.normal() + x @ [0.8, -0.4])))).astype(int)
        X.append(x); y.append(yy); st.append(np.full(n, s))
    X, y, st = np.vstack(X), np.concatenate(y), np.concatenate(st)

    def nll(b):
        tot = 0.0
        for s in np.unique(st):
            m = st == s
            k, n = y[m].sum(), m.sum()
            if 0 < k < n:
                eta = X[m] @ b
                den = sum(np.exp(eta[list(c)].sum()) for c in itertools.combinations(range(n), k))
                tot += eta[y[m] == 1].sum() - np.log(den)
        return -tot
    ref = minimize(nll, np.zeros(2), method='BFGS').x
    fit = clogit(X, y, st, l2=0)
    assert np.allclose(fit['beta'], ref, atol=1e-5)
    # conditional case probabilities sum to the cases per stratum; score contributions sum to zero
    for s in np.unique(st):
        m = (st == s) & np.isfinite(fit['pi'])
        if m.any():
            assert np.nansum(fit['pi'][m]) == pytest.approx(y[st == s].sum())
    assert np.allclose(fit['u'].sum(0), 0, atol=1e-8)
    draws = score_bootstrap(fit, [np.ones((3, len(y)))])
    assert np.allclose(draws, fit['beta'][0])


# ---------------------------------------------------------------- M1
def _tiny_states(states_a, states_b):
    """One A hit and one B hit 100 bp apart; states_*: {molecule: state}."""
    mols = sorted(set(states_a) | set(states_b))
    mt = MoleculeTable(np.array([f'm{i}' for i in mols], dtype=object))
    ia = [(0, mols.index(m), s) for m, s in states_a.items()]
    ib = [(0, mols.index(m), s) for m, s in states_b.items()]
    A = FeatureStates('nfr.region', ['a0'], ['c'], [1000], [1200], [1100], [r[0] for r in ia], [r[1] for r in ia],
                      [r[2] for r in ia])
    B = FeatureStates('fp.class', ['b0'], ['c'], [1300], [1320], [1310], [r[0] for r in ib], [r[1] for r in ib],
                      [r[2] for r in ib])
    return A, B, mt


def test_fold_states_follows_the_record_fold_rule():
    h, m, s = fold_states([0, 0, 0, 0, 1, 1], [5, 5, 6, 6, 7, 7], [1, 1, 1, 0, 0, 0])
    assert list(zip(h, m, s)) == [(0, 5, 1), (1, 7, 0)]        # mixed records of molecule 6 -> '.'


def test_m1_per_pair_counts_exclusions_and_directional_readouts():
    sa = {i: int(i < 20) for i in range(40)}
    sb = {i: int(i < 15 or 30 <= i < 32) for i in range(38)} | {50: 1}
    A, B, mt = _tiny_states(sa, sb)
    res = cooccurrence(A, B, mt, n_boot=0, effect=False)
    row = res['pairs'][0]
    assert row['n'] == 38 and row['excluded_no_b_state'] == 2 and row['excluded_no_a_state'] == 1
    assert row['table'] == [15, 5, 2, 16]
    assert row['p_b1_given_a1'] == pytest.approx(15 / 20) and row['p_b1_given_a0'] == pytest.approx(2 / 18)
    assert row['p_exact'] < 1e-3


def test_m1_same_kind_overlapping_pairs_are_skipped_and_nesting_flagged():
    mt = MoleculeTable(np.array(['m0'], dtype=object))
    A = FeatureStates('fp.class', ['a', 'b'], ['c', 'c'], [100, 110], [130, 140], [115, 125], [0, 1], [0, 0], [1, 1])
    pairs = find_pairs(A, A, 5000)
    assert pairs == [(0, 1, '', 'overlap')]
    B = FeatureStates('nfr.region', ['n'], ['c'], [50], [400], [225], [0], [0], [1])
    assert find_pairs(A, B, 5000)[0][2] == 'inside'


def test_m1_pooled_detects_a_planted_cooccurrence_and_not_a_null():
    sim = simulate_m1(3, effect=1.2)
    res = cooccurrence(sim['a'], sim['b'], sim['molecules'], n_boot=100)
    assert res['pooled']['p'] < 0.01 and res['pooled']['or'] > 1.5
    assert res['pooled']['or_lo'] < res['pooled']['or'] < res['pooled']['or_hi']
    null = cooccurrence(**{k: v for k, v in simulate_m1(4).items()}, n_boot=0, effect=False)
    assert null['pooled']['p'] > 0.01


def test_m1_small_samples_use_a_t_reference():
    sim = simulate_m1(5, chrom_len=150_000, coverage=15, n_a=20, n_b=30)
    res = cooccurrence(sim['a'], sim['b'], sim['molecules'], n_boot=0, effect=False)
    assert res['pooled']['n_molecules'] < 200 and 't(' in res['pooled']['test']


def test_m1_haplotype_contrast_leaves_unphased_out_and_reports_them():
    sim = simulate_m1(6, unphased_fraction=0.3, effect=0.6)
    res = cooccurrence(sim['a'], sim['b'], sim['molecules'], group='haplotype', group_kind='haplotype', n_boot=50,
                       per_pair=False)
    assert res['group_info']['groups'] == ['1', '2'] and res['group_info']['unphased'] > 0
    assert 'unphased' in res and 'p' in res['interaction']
    n_phased = sum(1 for h in sim['molecules'].labels['haplotype'] if h != 0)
    assert res['pooled']['n_molecules'] <= n_phased


def test_m1_strata_test_matches_mh_with_one_stratum():
    from fiberhmm.assoc import coaccess as C
    rng = np.random.default_rng(1)
    sa = {i: int(rng.random() < 0.5) for i in range(60)}
    sb = {i: int(rng.random() < 0.5) for i in range(60)}
    A, B, mt = _tiny_states(sa, sb)
    res = cooccurrence(A, B, mt, test='strata', n_boot=0)
    x = np.array([sa[i] for i in range(60)]); y = np.array([sb[i] for i in range(60)])
    l2, _lo, _hi = C.mantel_haenszel(x, y, np.zeros(60))
    assert res['pooled']['log_or'] == pytest.approx(l2 * np.log(2))


# ---------------------------------------------------------------- M2
def test_matched_backgrounds_respect_separation_and_covariate_bins():
    anchors = [('c', 10_000), ('c', 50_000)]
    cand = candidate_positions([('c', 0, 100_000)], step=500)
    cov = lambda pos: np.array([0.0 if p < 60_000 else 1.0 for _c, p in pos])
    mb = matched_backgrounds(anchors, cand, anchor_covariates={'cov': cov(anchors)},
                             candidate_covariates={'cov': cov(cand)}, k=3, min_sep=3000, n_bins=2, seed=1)
    chosen = [cand[j][1] for _a, js in mb['sets'] for j in js]
    assert len(chosen) == 6 and mb['shortfall'] == 0
    assert all(min(abs(p - 10_000), abs(p - 50_000)) >= 3000 for p in chosen)
    assert all(p < 60_000 for p in chosen)                       # same covariate bin as the anchors
    assert all(abs(a - b) >= 3000 for a, b in itertools.combinations(chosen, 2))


def test_m2_balance_removes_a_pure_density_difference():
    # at one anchor the response depends only on the density stratum; states are unbalanced across strata
    rows = []
    for s in range(4):
        for k in range(20):
            st = int(k < (4 + 4 * s))
            rows.append((0, len(rows), st, s, 1, 10.0 * s))
    loc, mol, st, strat, mask, y = map(np.array, zip(*rows))
    app = Appearances(loc, mol, st, strat, mask, np.stack([y, np.ones_like(y)], -1)[:, None, :])
    loci = LocusTable(['a'], ['c'], [0], [True], [-1])
    res = condition(app, loci, mean_metric(), n_perm=200, n_role=0, n_boot=0)
    assert res['bins'][0]['delta_anchor'] == pytest.approx(0.0)
    unbal = condition(app, loci, mean_metric(), n_perm=0, n_role=0, n_boot=0, balance=False)
    assert unbal['bins'][0]['delta_anchor'] > 1.0


def test_m2_detects_a_site_effect_and_the_role_permutation_keeps_self_seeding():
    sim = simulate_m2(2, self_seed=0.4, site_effect=0.8, n_anchors=80)
    res = condition(sim['app'], sim['loci'], sim['metric'], bins=sim['bins'], n_boot=100, per_anchor=True)
    b = res['bins'][0]
    assert b['p_site'] < 0.01 and b['delta_site'] > 0.3
    assert b['delta_random'] > 0.15                                  # self-seeding at the backgrounds
    assert b['delta_site_ci']['lo'] < b['delta_site'] < b['delta_site_ci']['hi']
    null = condition(**{k: simulate_m2(7, self_seed=0.4)[k] for k in ('app', 'loci', 'metric')},
                     n_boot=0, n_perm=0)
    assert null['bins'][0]['p_site'] > 0.01


def test_m2_per_anchor_rows_need_eight_molecules_per_state():
    sim = simulate_m2(3, self_seed=0.5, n_anchors=40, coverage=40)
    res = condition(sim['app'], sim['loci'], sim['metric'], bins=sim['bins'], n_boot=0, n_role=0, n_perm=200)
    rows = res.get('per_anchor', [])
    assert rows and all(r['n_state1'] >= 8 and r['n_state0'] >= 8 for r in rows)
    assert all(0 <= r['q'] <= 1 for r in rows) and all(np.isfinite(r['delta_shrunk']) for r in rows)


def test_m2_group_contrast_and_haplotypes():
    sim = simulate_m2(4, self_seed=0.3, site_effect=0.3, n_datasets=2, group_effect={0: 0.0, 1: 2.0}, n_anchors=150,
                      unphased_fraction=0.2)
    res = condition(sim['app'], sim['loci'], sim['metric'], bins=sim['bins'], n_perm=0, n_role=0, n_boot=200,
                    group='dataset')
    inter = res['bins'][0]['interaction']
    assert inter['statistic'] == 'delta_site' and inter['p'] < 0.05
    hap = condition(sim['app'], sim['loci'], sim['metric'], bins=sim['bins'], n_perm=0, n_role=0, n_boot=0,
                    group='haplotype', group_kind='haplotype')
    assert 'unphased' in hap and hap['group_info']['unphased'] > 0


# ---------------------------------------------------------------- M3
def test_compress_maps_intervals_points_and_junctions():
    L, iv, pts, junc = compress([(0, 100), (200, 300)], [(50, 250)], [10, 150, 260])
    assert L == 200 and iv.tolist() == [[50, 150]] and pts.tolist() == [10, 160] and junc.tolist() == [100]


def test_m3_conditional_rate_ratio_is_identified_within_molecules():
    # two molecules with very different B rates; inside/outside ratio 1 on each -> beta = 0 despite the crude
    d = RateData([1000, 1000], [np.array([[0, 500]]), np.array([[0, 100]])],
                 [np.arange(0, 1000, 10), np.arange(0, 1000, 100)])
    res = rate_ratio(d, n_shift=200, n_boot=0)
    assert abs(res['log_rate_ratio']) < 1e-6


def test_m3_shift_null_preserves_exposure_and_detects_depletion():
    sim = simulate_m3(1, beta=-0.6)
    res = rate_ratio(sim, n_boot=100)
    assert res['p_depleted'] < 0.01 and res['rate_ratio'] < 0.75
    assert res['rate_ratio_lo'] < res['rate_ratio'] < res['rate_ratio_hi']
    null = rate_ratio(simulate_m3(2), n_boot=0)
    assert null['p_shift'] > 0.01


def test_m3_junction_redraw_rule_keeps_cuts_below_ten_percent():
    from fiberhmm.assoc.rate import _allowed_junction_points
    ok = _allowed_junction_points(1000, np.array([[100, 300]]))
    inside = ok[(ok > 100) & (ok < 300)]
    assert inside.min() >= 101 and inside.max() <= 299
    assert np.all(np.minimum(inside - 100, 300 - inside) <= 20)


def test_m3_groups_and_edges():
    sim = simulate_m3(3, beta_by_dataset={0: 0.0, 1: -0.8}, n_datasets=2, n_molecules=500)
    res = rate_ratio(sim, n_shift=100, n_boot=200, group='dataset')
    assert res['interaction']['p'] < 0.05
    e = edge_association(sim, n_shift=100)
    assert e['n_boundaries'] > 0 and 0 < e['p_two_sided'] <= 1


# ---------------------------------------------------------------- groups
def test_group_labels_d14_and_post_selection():
    lab, info = group_labels([1, 2, 0, None, '1', 'unphased'], 'haplotype')
    assert info['groups'] == ['1', '2'] and info['unphased'] == 3 and list(lab) == ['1', '2', None, None, '1', None]
    ok = post_selection_check([('c', 1000, 2000)], [('c', 10_000, 11_000)], flank=500)
    assert ok['status'] == 'ok'
    bad = post_selection_check([('c', 1000, 2000)], [('c', 2300, 2600)], flank=500)
    assert bad['status'] == 'refused'
    assert post_selection_check([('c', 1000, 2000)], [('c', 2300, 2600)], 500, True)['status'] == 'post-selection'


def test_group_contrast_from_draws():
    rng = np.random.default_rng(0)
    out = group_contrast({'x': 0.0, 'y': 1.0}, {'x': rng.normal(0, 0.1, 500), 'y': rng.normal(1, 0.1, 500)})
    assert out['differences'][0]['difference'] == pytest.approx(1.0) and out['p'] < 1e-6


# ---------------------------------------------------------------- review regressions (Codex round 1, group A)
def test_m1_small_samples_use_t_and_tiny_pairs_are_not_pooled():
    """A1, A2: one pair of six molecules never rejects at 0.05 for any equally likely table (the sign-flip did
    at 10%); pairs below min_pool_molecules stay out of the pooled test."""
    from fiberhmm.assoc.cooccur import _clustered
    x = np.array([0, 0, 0, 1, 1, 1.])
    rej = 0
    for ones in itertools.combinations(range(6), 3):
        y = np.zeros(6)
        y[list(ones)] = 1
        u = (x - x.mean()) * (y - y.mean())
        rej += _clustered(u, np.array([np.sqrt(6 / 5)]), np.zeros(6, int), np.arange(6), u.sum(), 200, 't')['p'] < .05
    assert rej == 0
    sa = {i: int(i < 2) for i in range(4)}
    sb = {i: int(i % 2) for i in range(4)}
    A, B, mt = _tiny_states(sa, sb)
    res = cooccurrence(A, B, mt, n_boot=0, effect=False, min_pool_molecules=5)
    assert res['pooled'].get('n_molecules', 0) == 0


def test_clogit_large_balanced_stratum_does_not_overflow():
    """A3."""
    from math import comb, log
    f = clogit(np.tile([0, 0, 1, 1], 300), np.tile([0, 1, 0, 1], 300), np.zeros(1200), max_iter=2, l2=0)
    assert abs(f['beta'][0]) < 1e-8 and f['loglik'] == pytest.approx(-log(comb(1200, 600)), abs=1e-6)
    assert np.nansum(f['pi']) == pytest.approx(600)


def test_m2_bootstrap_recomputes_balance_weights():
    """A4: a contrast that is zero by balance stays zero in every bootstrap draw."""
    rows = []
    for s_ in range(2):
        for k in range(20):
            rows.append((0, len(rows), k % 2, s_, 1, 10.0 * s_))
    loc, mol, st, strat, mask, y = map(np.array, zip(*rows))
    app = Appearances(loc, mol, st, strat, mask, np.stack([y, np.ones_like(y)], -1)[:, None, :])
    loci = LocusTable(['a', 'b'], ['c', 'c'], [0, 10_000], [True, True], [-1, -1])
    r = condition(app, loci, mean_metric(), n_perm=0, n_role=0, n_boot=100)
    assert r['bins'][0]['delta_anchor'] == pytest.approx(0) and r['bins'][0]['delta_anchor_ci']['se'] < 1e-9 or \
        np.isnan(r['bins'][0]['delta_anchor_ci']['se'])


def test_twoway_covariance_is_positive_semidefinite_and_contrasts_reject_negative_statistics():
    """A5, A6."""
    rng = np.random.default_rng(0)
    e = rng.choice([-1.0, 1.0], size=(4000, 3))
    mk = lambda C: e @ np.linalg.cholesky(C + 1e-9 * np.eye(3)).T
    parts = {}
    for name, C in (('units', np.array([[1, 1, 0], [1, 1, 0], [0, 0, 1.]])),
                    ('molecules', np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1.]])), ('items', 10 * np.eye(3))):
        d = mk(C)
        parts[name] = {'a': d[:, 0], 'b': d[:, 1], 'c': d[:, 2]}
    boot = R.BootDraws('twoway', parts=parts, info={'clusters': 50, 'molecules': 50})
    c = boot.cov(['a', 'b', 'c'])
    assert np.linalg.eigvalsh(c).min() > -1e-9
    out = group_contrast({'a': 0.0, 'b': 1.0, 'c': 1.0}, boot)
    assert not (out.get('chi2', 0) < 0 and out['p'] == 1)
    assert R.BootDraws('groups', draws={'x': np.array([-1., 0, 1])}, info={'clusters': 2}).correction() == (2.0, 1.0)


def test_m3_bootstrap_keeps_a_physical_molecule_together_and_boundaries():
    """A7, A9, A10."""
    d = RateData([1000, 1000, 1000], [np.array([[0, 500]])] * 3, [np.array([100, 700])] * 3,
                 molecule_ids=np.array(['m1', 'm1', 'm2'], dtype=object))
    res = rate_ratio(d, n_shift=0, n_boot=20)
    assert res['bootstrap']['molecules'] == 2
    assert rate_ratio(RateData([100], [np.array([[0, 50]])], [np.array([80])]), n_shift=0,
                      n_boot=0)['rate_ratio'] == 0.0
    d3 = RateData([100] * 3, [np.array([[0, 50]])] * 3, [np.array([80, 10])] * 3,
                  labels={'hap': np.array([1, 2, 0], dtype=object)})
    r = rate_ratio(d3, n_shift=0, n_boot=0, group='hap', group_kind='haplotype')
    assert r['unphased']['n_molecules'] == 1 and set(r['groups']) == {'1', '2'}


def test_float_zero_haplotype_is_unphased():
    """A8."""
    lab, info = group_labels([1.0, 2.0, 0.0, '0.0'], 'haplotype')
    assert info['unphased'] == 2 and lab[2] is None and lab[3] is None


# ---------------------------------------------------------------- review regressions (Codex round 2, group A)
def test_clogit_mixed_degree_strata_do_not_underflow():
    X = np.tile([0, 0, 1, 1], 600)
    y = np.r_[np.tile([0, 1, 0, 1], 300), 1, np.zeros(1199, int)]
    f = clogit(X, y, np.repeat([0, 1], 1200), max_iter=5)
    assert np.isfinite(f['beta']).all() and np.isfinite(f['loglik'])
    assert np.nansum(f['pi']) == pytest.approx(601)


def test_single_contrast_with_two_clusters_uses_t1():
    b = R.BootDraws('groups', draws={'a': np.zeros(3), 'b': np.array([-1., 0, 1])}, info={'clusters': 2})
    r = group_contrast({'a': 0., 'b': 3.}, b)
    assert r['p'] == pytest.approx(r['differences'][0]['p_wald'])


def test_per_pair_rows_do_not_depend_on_the_pooling_threshold():
    A, B, mt = _tiny_states({i: i % 2 for i in range(4)}, {i: int(i < 2) for i in range(4)})
    res = cooccurrence(A, B, mt, min_molecules=2, n_boot=0, effect=False)
    assert res['n_pair_rows'] == 1 and res['pooled'].get('n_molecules', 0) == 0
