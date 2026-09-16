"""Brute-force and generative gates for the experimental batch kernel."""
import itertools
import math

import numpy as np
import pytest
from scipy.special import logsumexp

from fiberhmm.inference.hierarchical_state_model import FixedStateCatalog, StateInterval, interval_evidence, posterior_quality


@pytest.mark.parametrize("n", [0, 1, 4, 8, 12])
def test_batch_against_bruteforce(n):
    rng = np.random.default_rng(151 + n)
    starts = rng.integers(0, 25, n)
    states = [StateInterval(str(i), int(s), int(s+rng.integers(1, 8))) for i,s in enumerate(starts)]
    cat = FixedStateCatalog(states)
    weights = rng.normal(0, 2, (3,n))
    configs = []
    for mask in itertools.product((0,1), repeat=n):
        iv = sorted((s.start,s.end) for s,k in zip(states,mask) if k)
        if all(a[1] <= b[0] for a,b in zip(iv,iv[1:])):
            configs.append(mask)
    masks = np.asarray(configs).reshape(len(configs),n)
    logs = weights @ masks.T
    z = logsumexp(logs, axis=1)
    expected = np.exp(logs-z[:,None]) @ masks
    got = cat.infer(weights)
    np.testing.assert_allclose(got.log_z, z, atol=1e-12, rtol=0)
    np.testing.assert_allclose(got.marginals, expected, atol=1e-12, rtol=0)


def test_prevalence_is_not_logistic_activity():
    cat = FixedStateCatalog([StateInterval("a",0,10), StateInterval("b",5,15)])
    np.testing.assert_allclose(cat.infer([0,0]).marginals, [[1/3,1/3]], atol=1e-15)
    np.testing.assert_allclose(cat.infer([1000,-1000]).marginals, [[1,0]], atol=1e-15)
    assert cat.map_indices([1,2]) == (1,)
    assert cat.map_indices([1,2], forbidden=[1]) == (0,)


def test_literal_lattice_likelihood_missingness_and_information():
    got = interval_evidence([2,8,11], [0,1,0], [.8,.6,.7], [.1,.2,.1], [0,3,10], [10,7,12])
    assert got["opportunities"].tolist() == [2,0,1]
    assert got["log_likelihood_ratio"][0] == pytest.approx(math.log(.9/.2)+math.log(.2/.6))
    assert got["log_likelihood_ratio"][1] == 0
    assert got["maximum_attainable_log_ratio"][1] == 0
    assert np.all(got["expected_protected_information"] >= -1e-12)
    with pytest.raises(ValueError):
        interval_evidence([2,2],[0,1],[.8,.8],[.1,.1],[0],[4])


def test_fitted_prior_and_molecule_likelihood_remain_separate():
    # Deliberately weak per-unit evidence but enough units to estimate prevalence.
    rng = np.random.default_rng(405)
    truth = rng.random(6000) < .3
    y = rng.random(6000) < np.where(truth,.4,.6)
    evidence = np.where(y,math.log(.4/.6),math.log(.6/.4))[:,None]
    cat = FixedStateCatalog([StateInterval("a",0,10)])
    fit = cat.fit_activities(evidence)
    assert fit["converged"]
    assert fit["prior_marginals"][0] == pytest.approx(.3, abs=.06)
    equal = cat.infer(evidence).marginals
    combined = cat.infer(evidence+np.asarray(fit["activities"])).marginals
    assert equal.max() <= .600000000001
    assert combined.max() < .5
    assert fit["log_likelihood_ratio"] > 20
    # No per-molecule high-quality assignment is needed for population gain.
    assert np.max(posterior_quality(combined)) < 3.1


def test_quality_thresholds_are_monotone_without_redecoding():
    q = posterior_quality([0,.1,.5,.9,.99,1])
    np.testing.assert_allclose(q[2:5],[-10*math.log10(.5),10,20],atol=1e-12)
    previous = set(range(len(q)))
    for cutoff in (0,3,10,20):
        visible = {i for i,value in enumerate(q) if value >= cutoff and i != 4}
        assert visible.issubset(previous)
        assert 4 not in visible
        previous = visible
