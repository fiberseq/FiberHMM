"""Lattice recaller discovery: a class's stability is its prediction strength averaged over every split-half.

Up to 3.0.0 nominate() keyed each split's per-cluster strengths by rounded test-half centroid and gave a final class
the values under the single nearest key; keys rarely coincide across splits, so one split decided a class's stability.
"""
import dataclasses

import numpy as np
import pytest

from fiberhmm.inference.consensus.lattice_recaller import discovery as D
from fiberhmm.inference.consensus.parameters import RecallerOptions

SITES = np.arange(0, 1000, 4)
PA, PP = 0.8, 0.02


def _units(n, footprints, jitter, seed=11, prefix='m'):
    """n molecules, one native call each from one of the footprints (round robin), edges jittered by up to +-jitter bp;
    sites are marked everywhere outside the call so the censored edges sit on the call."""
    rng = np.random.default_rng(seed); out = []
    for i in range(n):
        a, b = footprints[i % len(footprints)]
        a += int(rng.integers(-jitter, jitter + 1)); b += int(rng.integers(-jitter, jitter + 1))
        hit = (SITES < a) | (SITES >= b)
        d = np.where(hit, np.log(PP) - np.log(PA), np.log1p(-PP) - np.log1p(-PA))
        out.append(dict(uid=f'{prefix}{i}', ch='ds::CT', pos=SITES, hit=hit, d=d, calls=[(a, b, 50.)]))
    return out


SEPARATED = [(100, 130), (300, 340), (600, 625)]
CROWDED = [(100, 130), (110, 145), (125, 150), (140, 170)]


def _opt(**kw):
    return dataclasses.replace(RecallerOptions(), **kw)


def _by_span(cands):
    return sorted((tuple(np.round(c['X'].mean(0), 6)), len(c['calls']), c['stability']) for c in cands)


# ---------------------------------------------------------------- cluster_stability (the matching rule)
def test_stability_is_membership_weighted_mean_over_every_split():
    members = [0, 1, 2, 3]
    splits = [(np.array([0, 1, 5]), np.array([1, 1, 0]), np.array([.2, .9])),      # both held calls in test cluster 1
              (np.array([2, 3, 6]), np.array([0, 1, 1]), np.array([.5, 1.])),      # split across clusters 0 and 1
              (np.array([5, 6]), np.array([0, 0]), np.array([.1, .1]))]            # no held-out call: no evidence
    assert np.isclose(D.cluster_stability(members, splits), np.mean([.9, (.5 + 1.)/2]))
    # unequal shares and unequal held-out counts: 2 of 3 held calls in cluster 0, 1 in cluster 2; then one call
    splits = [(np.array([0, 1, 2, 9]), np.array([0, 0, 2, 1]), np.array([.6, .1, .9])),
              (np.array([3, 8]), np.array([1, 0]), np.array([.2, .7]))]
    assert np.isclose(D.cluster_stability(members, splits), np.mean([(2*.6 + .9)/3, .7]))
    # an empty or singleton test cluster scores 0 and pulls a class it holds down
    splits = [(np.array([0, 1]), np.array([0, 1]), np.array([1., 0.]))]
    assert np.isclose(D.cluster_stability([0, 1], splits), .5)


def test_stability_ignores_call_order_cluster_numbering_and_split_order():
    rng = np.random.default_rng(3); n, k = 60, 3; members = np.arange(0, 60, 3)
    splits = []
    for _ in range(6):
        rows = np.sort(rng.choice(n, 30, replace=False))
        splits.append((rows, rng.integers(0, k, len(rows)), rng.random(k)))
    ref = D.cluster_stability(members, splits)
    relabelled = []
    for rows, lab, ps in splits[::-1]:
        perm = rng.permutation(k); order = rng.permutation(len(rows))      # new cluster numbers, shuffled calls
        relabelled.append((rows[order], perm[lab][order], ps[np.argsort(perm)]))
    assert np.isclose(D.cluster_stability(rng.permutation(members), relabelled), ref, rtol=0, atol=1e-12)


def test_stability_without_held_out_calls_is_zero_not_one():
    splits = [(np.array([4, 5]), np.array([0, 1]), np.array([1., 1.]))]
    assert D.cluster_stability([0, 1, 2], splits) == 0.0
    assert D.cluster_stability([0, 1, 2], []) == 0.0


def test_too_few_calls_for_k_has_no_strength_and_no_splits():
    X = np.array([[100., 130.], [300., 340.], [600., 625.]])
    assert D.prediction_strength(X, ['a', 'b', 'c'], 3, 1, 6) == (0.0, [])


# ---------------------------------------------------------------- nominate
def test_every_split_contributes_to_a_class_stability():
    units = _units(240, CROWDED, jitter=6); opt = _opt(stringency=.5)
    cands, k, _choice = D.nominate(units, opt)
    assert k > 1 and cands
    X, meta = D.features(units, opt.censor_bp); groups = [units[m[0]]['uid'] for m in meta]
    _, splits = D.prediction_strength(X, groups, k, opt.seed, opt.prediction_splits)
    assert len(splits) == opt.prediction_splits
    lab = D._fit(X, k, opt.seed)[1](X)
    varied = False
    for c in cands:
        j = int(c['id'][1:]); members = set(np.flatnonzero(lab == j).tolist())
        per_split = []
        for rows, labels, ps in splits:      # independent re-derivation of the rule, call by call
            held = [ps[t] for r, t in zip(rows, labels) if int(r) in members]
            if held:
                per_split.append(sum(held)/len(held))
        assert len(per_split) == opt.prediction_splits     # every split holds some of the class's calls
        assert np.isclose(c['stability'], np.mean(per_split))
        varied |= np.ptp(per_split) > 1e-6
    assert varied   # the splits disagree, so a single-split value would differ from the average


def test_chosen_k_reuses_its_split_results():
    units = _units(240, CROWDED, jitter=6); opt = _opt(stringency=.5)
    cands, k, _choice = D.nominate(units, opt)
    X, meta = D.features(units, opt.censor_bp); groups = [units[m[0]]['uid'] for m in meta]
    _, again = D.prediction_strength(X, groups, k, opt.seed, opt.prediction_splits)
    lab = D._fit(X, k, opt.seed)[1](X)
    for c in cands:
        assert c['stability'] == D.cluster_stability(np.flatnonzero(lab == int(c['id'][1:])), again)


def test_nominate_is_deterministic():
    units = _units(240, CROWDED, jitter=6); opt = _opt(stringency=.5)
    a, b = D.nominate(units, opt), D.nominate(units, opt)
    assert a[1] == b[1] and a[2] == b[2] and _by_span(a[0]) == _by_span(b[0])


def test_stability_does_not_depend_on_read_order():
    """Splits are keyed by molecule ID, so when k-means finds the same partitions from any call order the stabilities
    are identical. (k-means++ starts do depend on call order: on crowded data a different order can choose another k.
    That is the read-order sensitivity `--robust` measures; the stability rule adds none of its own.)"""
    units = _units(150, SEPARATED, jitter=4); opt = _opt()
    ref = D.nominate(units, opt)
    assert ref[1] == 3 and len(ref[0]) == 3
    for seed in (1, 2, 3):
        shuffled = [units[i] for i in np.random.default_rng(seed).permutation(len(units))]
        got = D.nominate(shuffled, opt)
        assert got[1] == ref[1] and _by_span(got[0]) == _by_span(ref[0])
    assert all(c['stability'] == 1.0 for c in ref[0])


@pytest.mark.filterwarnings('ignore:Number of distinct clusters')   # identical calls: k >= 2 has empty clusters
def test_single_cluster_is_fully_stable():
    units = _units(60, [(300, 340)], jitter=0); opt = _opt(stringency=1.)
    cands, k, choice = D.nominate(units, opt)
    assert k == 1 and len(cands) == 1 and cands[0]['stability'] == 1.0 and choice[0] == (1, 1.0)


# ---------------------------------------------------------------- core rule: re-split instead of drop
NEIGHBOURS = [(300, 330), (345, 380)]     # pooled: the middle half of calls shares no protected core


def _pooled(units, opt, cid='c0'):
    X, meta = D.features(units, opt.censor_bp)
    return dict(id=cid, members=[cid], calls=meta, X=X, stability=1.0)


def test_candidate_failing_the_core_rule_is_split_into_its_footprints():
    units = _units(240, NEIGHBOURS, jitter=2); opt = _opt()
    lumped = _pooled(units, opt)
    assert D.core_bp(lumped, opt) < opt.minimum_core_bp
    log = []
    kids = D.resplit(lumped, units, opt, opt.core_resplit_depth, log)
    assert len(kids) == 2 and all(k['id'].startswith('c0/') for k in kids)
    spans = sorted(tuple(np.round(D.boxes(k)['span'])) for k in kids)
    assert all(abs(s[0] - a) <= 3 and abs(s[1] - b) <= 3 for s, (a, b) in zip(spans, NEIGHBOURS))
    assert all(D.core_bp(k, opt) >= opt.minimum_core_bp and k['stability'] >= opt.stringency for k in kids)
    assert sorted(m for k in kids for m in k['calls']) == sorted(lumped['calls'])    # every call kept, once
    assert log[0]['parent'] == 'c0' and log[0]['k'] >= 2     # finer k-means clusters of one footprint merge back (identity test)


def test_resplit_leaves_a_single_dispersed_class_and_respects_its_switches():
    opt = _opt()
    wide = _units(120, [(300, 320)], jitter=20)       # one footprint whose edges scatter: no core, but one class
    c = _pooled(wide, opt)
    assert D.core_bp(c, opt) < opt.minimum_core_bp
    assert D.resplit(c, wide, opt, 2, []) == [c]       # still dropped by the core rule, as before
    units = _units(240, NEIGHBOURS, jitter=2); lumped = _pooled(units, opt)
    assert D.resplit(lumped, units, opt, 0, []) == [lumped]
    assert D.resplit(lumped, units, _opt(minimum_core_bp=-1000), 2, []) == [lumped]
    unstable = dict(lumped, stability=.5)
    assert D.resplit(unstable, units, opt, 2, []) == [unstable]


def test_discover_tile_keeps_the_footprints_of_a_lumped_candidate(monkeypatch):
    units = _units(240, NEIGHBOURS, jitter=2); opt = _opt()
    monkeypatch.setattr(D, 'nominate', lambda u, o: ([_pooled(u, o)], 1, [(1, 1.0)]))
    off, _ = D.discover_tile(units, _opt(core_resplit_depth=0))
    assert len(off) == 1 and off[0]['core_bp'] < opt.minimum_core_bp          # dropped later by the core rule
    on, diag = D.discover_tile(units, opt)
    assert len(on) == 2 and all(g['core_bp'] >= opt.minimum_core_bp for g in on)
    assert diag['resplits'][0]['parent'] == 'c0'
    again, _ = D.discover_tile(units, opt)
    assert [(g['span'], g['stability'], g['core_bp']) for g in again] == [(g['span'], g['stability'], g['core_bp']) for g in on]
