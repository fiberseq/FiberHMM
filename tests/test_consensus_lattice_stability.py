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


def _units(n, footprints, jitter, seed=11, prefix='m'):
    """n molecules, one native call each from one of the footprints (round robin), edges jittered by up to +-jitter bp;
    sites are marked everywhere outside the call so the censored edges sit on the call."""
    rng = np.random.default_rng(seed); out = []
    for i in range(n):
        a, b = footprints[i % len(footprints)]
        a += int(rng.integers(-jitter, jitter + 1)); b += int(rng.integers(-jitter, jitter + 1))
        out.append(dict(uid=f'{prefix}{i}', pos=SITES, hit=(SITES < a) | (SITES >= b), calls=[(a, b, 50.)]))
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
    assert D.cluster_stability(rng.permutation(members), relabelled) == ref


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


def test_nominate_is_deterministic():
    units = _units(240, CROWDED, jitter=6); opt = _opt(stringency=.5)
    a, b = D.nominate(units, opt), D.nominate(units, opt)
    assert a[1] == b[1] and a[2] == b[2] and _by_span(a[0]) == _by_span(b[0])


def test_stability_does_not_depend_on_read_order():
    """Splits are keyed by molecule ID, so with clusters k-means finds from any start the stabilities are identical."""
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
