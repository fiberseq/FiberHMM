"""Small exact reference gates; no empirical calibration claim."""
import itertools
import math

import numpy as np
import pytest

from fiberhmm.inference.state_geometry import GeometryFamily, StateGeometry, infer_joint_geometry


def test_geometry_conditioning_cannot_be_replaced_by_independent_integrals():
    a = GeometryFamily("A", (StateGeometry("short", 0, 4, 0, .5),
                             StateGeometry("long", 0, 8, math.log(100), .5)))
    b = GeometryFamily("B", (StateGeometry("only", 6, 10, 0),))
    r = infer_joint_geometry([a, b])
    p = dict(zip(r.configurations, r.configuration_posteriors))
    assert p[("A", "B")] == pytest.approx(1 / 53.5, abs=1e-14)
    assert p[("A", "B")] != pytest.approx(50.5 / 103)
    assert dict(zip(r.configurations, r.configuration_priors))[("A", "B")] == .25


def test_observation_likelihood_is_normalized_with_joint_geometry():
    total = 0.0
    positions = np.array([1, 3, 6, 8])
    for outcome in itertools.product((0, 1), repeat=4):
        y = np.array(outcome)
        steps = np.where(y, math.log(.1/.7), math.log(.9/.3))
        def g(name, lo, hi, weight=1):
            return StateGeometry(name, lo, hi, float(steps[(positions >= lo) & (positions < hi)].sum()), weight)
        r = infer_joint_geometry([GeometryFamily("A", (g("short", 0, 4, .5), g("long", 0, 8, .5))),
                                 GeometryFamily("B", (g("only", 6, 10),))])
        base = float(np.prod(np.where(y, .7, .3)))
        total += base * math.exp(r.log_marginal_likelihood_ratio)
    assert total == pytest.approx(1, abs=1e-12)


def test_one_geometry_per_family_veto_decision_does_not_change_inference():
    f = GeometryFamily("A", (StateGeometry("a", 0, 2, 5), StateGeometry("b", 4, 6, 5)))
    r = infer_joint_geometry([f])
    assert r.configurations == ((), ("A",))
    assert r.map_state_ids == ("A",)
    assert r.decision_configuration(["A"]) == ()
    assert r.group_probability(["A"]) == pytest.approx(1/(1+math.exp(-5)))
    assert r.map_state_ids == ("A",)


def test_equal_geometry_weights_do_not_multiply_family_prior():
    f = GeometryFamily("A", tuple(StateGeometry(str(i), i*2, i*2+1, 0) for i in range(6)))
    r = infer_joint_geometry([f])
    assert r.prior_marginals == (.5,)
    assert r.posterior_marginals == (.5,)


def test_order_reflection_and_budget():
    a = GeometryFamily("A", (StateGeometry("g", 0, 8, 1),))
    b = GeometryFamily("B", (StateGeometry("g", 5, 10, 2),))
    r = infer_joint_geometry([a, b])
    assert r == infer_joint_geometry([b, a])
    reflected = [GeometryFamily(f.state_id, tuple(StateGeometry(g.geometry_id, -g.end, -g.start,
                                  g.log_likelihood_ratio, g.weight) for g in f.geometries)) for f in (a,b)]
    rr = infer_joint_geometry(reflected)
    assert r.configuration_posteriors == rr.configuration_posteriors
    assert ("A", "B") not in r.configurations
    with pytest.raises(ValueError, match="budget"):
        infer_joint_geometry([a,b], max_geometry_choices=2)


def test_empty_and_invalid():
    r = infer_joint_geometry([])
    assert r.configurations == ((),)
    assert r.log_marginal_likelihood_ratio == 0
    assert r.map_state_ids == ()
    with pytest.raises(ValueError):
        StateGeometry("bad", 3, 3, 0)
    with pytest.raises(ValueError):
        infer_joint_geometry([], {"not_a_state": 0})
