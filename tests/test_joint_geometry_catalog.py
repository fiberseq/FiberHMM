"""Exact mathematical gates for the compiled joint family/geometry kernel."""
import itertools
import math

import numpy as np
import pytest
from scipy.special import logsumexp

from fiberhmm.inference.joint_geometry_catalog import (
    ExactGeometryBudgetError, FamilyGeometry, GeometryInterval, JointGeometryCatalog,
)
from fiberhmm.inference.state_geometry import GeometryFamily, StateGeometry, infer_joint_geometry


def _reference(geometries, families, evidence, eta):
    native = []
    for family in families:
        weights = family.weights or (1.0,) * len(family.geometry_indices)
        native.append(GeometryFamily(family.family_id, tuple(
            StateGeometry(geometries[g].geometry_id, geometries[g].start, geometries[g].end, evidence[g], weight)
            for g, weight in zip(family.geometry_indices, weights))))
    return infer_joint_geometry(native, dict(zip((f.family_id for f in families), eta)))


def _geometry_brute(geometries, families, evidence, eta):
    by_config = {}
    for combination in itertools.product(*[[None, *f.geometry_indices] for f in families]):
        selected = [g for g in combination if g is not None]
        if any(geometries[a].start < geometries[b].end and geometries[b].start < geometries[a].end
               for a, b in itertools.combinations(selected, 2)):
            continue
        config = tuple(i for i, g in enumerate(combination) if g is not None)
        weight = 1.0
        for i in config:
            weights = families[i].weights or (1.0,) * len(families[i].geometry_indices)
            weight *= weights[families[i].geometry_indices.index(combination[i])]
        by_config.setdefault(config, []).append((selected, weight))
    all_tiles = []
    for config, tiles in by_config.items():
        denominator = sum(w for _, w in tiles)
        for selected, weight in tiles:
            log_score = sum(eta[i] for i in config) + math.log(weight / denominator) + sum(evidence[g] for g in selected)
            all_tiles.append((selected, log_score))
    z = logsumexp([score for _, score in all_tiles])
    return [sum(math.exp(score - z) for selected, score in all_tiles if g in selected)
            for g in range(len(geometries))]


def _fixture():
    geometries = [GeometryInterval("A_short", 0, 4), GeometryInterval("A_long", 0, 8),
                  GeometryInterval("B", 6, 10)]
    families = [FamilyGeometry("A", (0, 1), (.5, .5)), FamilyGeometry("B", (2,))]
    return geometries, families


def test_mandatory_geometry_denominator_and_conditional_map():
    geometry, families = _fixture()
    catalog = JointGeometryCatalog(geometry, families)
    result = catalog.infer([[0, math.log(100), 0]])
    assert catalog.compile_summary[0]["mode"] == "exact_compatible_tilings"
    assert result.configuration_probability(["A", "B"]) == pytest.approx(1 / 53.5, abs=1e-14)
    assert result.configuration_probability(["A", "B"], prior=True) == pytest.approx(.25)
    assert result.configuration_probability(["A", "B"]) != pytest.approx(50.5 / 103)
    assert result.map_family_ids == (("A",),)
    assert result.map_geometry_ids == (("A_long",),)
    assert result.family_marginals[0, 0] == pytest.approx((50.5 + 1) / 53.5)
    # When B receives sufficient evidence, AB is best, but its only compatible
    # geometry uses A_short, despite A_long being A's independent best shape.
    constrained = catalog.infer([[0, math.log(100), math.log(1000)]], activities=[2, 0])
    assert constrained.map_family_ids == (("A", "B"),)
    assert constrained.map_geometry_ids == (("A_short", "B"),)


@pytest.mark.parametrize("seed", range(8))
def test_batch_matches_scalar_reference_and_geometry_brute(seed):
    rng = np.random.default_rng(seed)
    geometries, families = [], []
    for f in range(4):
        indices = []
        for g in range(int(rng.integers(1, 4))):
            start = int(rng.integers(-3, 15))
            indices.append(len(geometries))
            geometries.append(GeometryInterval(f"g{f}_{g}", start, start + int(rng.integers(1, 8))))
        families.append(FamilyGeometry(f"f{f}", tuple(indices), tuple(rng.uniform(.1, 2, len(indices)))))
    evidence, eta = rng.uniform(-4, 4, (5, len(geometries))), rng.uniform(-1, 1, len(families))
    catalog = JointGeometryCatalog(geometries, families)
    result = catalog.infer(evidence, eta, batch_size=2)
    for row in range(len(evidence)):
        reference = _reference(geometries, families, evidence[row], eta)
        assert result.log_z_observation[row] == pytest.approx(reference.log_z_observation, abs=1e-12)
        assert result.log_z_prior == pytest.approx(reference.log_z_prior, abs=1e-12)
        assert result.family_marginals[row] == pytest.approx(reference.posterior_marginals, abs=1e-12)
        assert result.prior_family_marginals == pytest.approx(reference.prior_marginals, abs=1e-12)
        assert result.geometry_marginals[row] == pytest.approx(_geometry_brute(geometries, families, evidence[row], eta), abs=1e-12)
        assert result.prior_geometry_marginals == pytest.approx(_geometry_brute(geometries, families, np.zeros(len(geometries)), eta), abs=1e-12)
        equal = _reference(geometries, families, evidence[row], np.zeros(len(families)))
        assert result.equal_prior_family_marginals[row] == pytest.approx(equal.posterior_marginals, abs=1e-12)
        assert result.equal_prior_geometry_marginals[row] == pytest.approx(_geometry_brute(geometries, families, evidence[row], np.zeros(len(families))), abs=1e-12)
        assert result.map_family_ids[row] == reference.map_state_ids
        for config, posterior, prior in zip(reference.configurations, reference.configuration_posteriors, reference.configuration_priors):
            assert result.configuration_probability(config, row) == pytest.approx(posterior, abs=1e-12)
            assert result.configuration_probability(config, row, prior=True) == pytest.approx(prior, abs=1e-12)
        chosen = [geometries[g] for g in result.map_geometry_indices[row]]
        assert all(not (a.start < b.end and b.start < a.end) for a, b in itertools.combinations(chosen, 2))


def test_normalized_observation_distribution_with_mixed_geometry():
    geometries, families = _fixture()
    catalog = JointGeometryCatalog(geometries, families)
    positions = np.array([1, 3, 6, 8])
    evidence, bases = [], []
    for observation in itertools.product((0, 1), repeat=4):
        y = np.array(observation)
        steps = np.where(y, math.log(.1 / .7), math.log(.9 / .3))
        evidence.append([steps[(positions >= g.start) & (positions < g.end)].sum() for g in geometries])
        bases.append(np.prod(np.where(y, .7, .3)))
    result = catalog.infer(evidence, [.8, -.3])
    assert np.dot(bases, np.exp(result.log_marginal_likelihood_ratio)) == pytest.approx(1, abs=1e-12)


def test_duplicate_geometry_weight_split_does_not_change_family_evidence():
    geometries, families = _fixture()
    original = JointGeometryCatalog(geometries, families).infer([[.1, 2.3, .8]], [.7, -.1])
    expanded = [*geometries, GeometryInterval("A_long_duplicate", 0, 8)]
    split = [FamilyGeometry("A", (0, 1, 3), (.5, .25, .25)), families[1]]
    result = JointGeometryCatalog(expanded, split).infer([[.1, 2.3, .8, 2.3]], [.7, -.1])
    assert result.log_marginal_likelihood_ratio == pytest.approx(original.log_marginal_likelihood_ratio, abs=1e-12)
    assert result.family_marginals == pytest.approx(original.family_marginals, abs=1e-12)
    assert result.geometry_marginals[0, 1] + result.geometry_marginals[0, 3] == pytest.approx(original.geometry_marginals[0, 1], abs=1e-12)
    assert result.prior_family_marginals == pytest.approx(original.prior_family_marginals, abs=1e-12)


def test_family_alternatives_cannot_cooccur_even_if_their_coordinates_do_not_overlap():
    geometry = [GeometryInterval("left", 0, 2), GeometryInterval("right", 10, 12)]
    result = JointGeometryCatalog(geometry, [FamilyGeometry("A", (0, 1))]).infer([[5, 5]])
    assert result.geometry_marginals.sum() == pytest.approx(result.family_marginals[0, 0])
    assert len(result.map_geometry_indices[0]) == 1
    assert result.prior_family_marginals == pytest.approx([.5])


def test_proven_factorization_uses_representatives_not_false_envelope_overlap():
    geometry = [GeometryInterval("left", 0, 2), GeometryInterval("right", 10, 12), GeometryInterval("middle", 5, 7)]
    families = [FamilyGeometry("A", (0, 1)), FamilyGeometry("B", (2,))]
    catalog = JointGeometryCatalog(geometry, families, max_enumerated_families=1)
    result = catalog.infer([[0, 0, 0]])
    assert all(c["mode"] == "proven_factorized_interval_dp" for c in catalog.compile_summary)
    assert result.configuration_probability(["A", "B"]) == pytest.approx(.25)


def test_large_singleton_chain_uses_exact_fast_path_and_implicit_probabilities():
    geometry = [GeometryInterval(f"g{i:03}", i * 2, i * 2 + 3) for i in range(120)]
    families = [FamilyGeometry(f"f{i:03}", (i,)) for i in range(len(geometry))]
    catalog = JointGeometryCatalog(geometry, families, max_enumerated_families=2)
    result = catalog.infer(np.zeros((2, len(geometry))), batch_size=1)
    assert len(catalog.compile_summary) == 1
    assert catalog.compile_summary[0]["mode"] == "proven_factorized_interval_dp"
    assert result.log_marginal_likelihood_ratio == pytest.approx([0, 0], abs=1e-12)
    assert result.components[0].configuration_posteriors is None
    assert result.configuration_probability(["f000", "f001"]) == 0
    assert result.configuration_probability([]) > 0


def test_activity_gradient_and_fit_use_normalized_family_prior():
    geometry, families = _fixture()
    catalog = JointGeometryCatalog(geometry, families)
    evidence = np.array([[2, 4, .5], [-2, -1, .7], [.1, -.9, -1.3]])
    eta = np.array([.3, -.7])
    result = catalog.infer(evidence, eta)
    h = 1e-5
    for i in range(2):
        plus, minus = eta.copy(), eta.copy()
        plus[i] += h; minus[i] -= h
        finite_difference = (catalog.infer(evidence, plus).log_marginal_likelihood_ratio.sum() -
                             catalog.infer(evidence, minus).log_marginal_likelihood_ratio.sum()) / (2 * h)
        assert result.activity_gradient[:, i].sum() == pytest.approx(finite_difference, abs=1e-9)
    fit = catalog.fit_activities(evidence, regularization=.7, prior_center=.2)
    assert fit["converged"]
    assert fit["activity_prior_center"] == [.2, .2]
    fitted = catalog.infer(evidence, fit["activities"])
    gradient = fitted.activity_gradient.sum(axis=0) - .7 * (np.array(fit["activities"]) - .2)
    assert np.max(np.abs(gradient)) < 1e-5


def test_long_state_wins_for_unmodified_extension_but_loses_for_modified_extension():
    geometry = [GeometryInterval("short", 0, 3), GeometryInterval("long", 0, 9)]
    families = [FamilyGeometry("short_family", (0,)), FamilyGeometry("long_family", (1,))]
    catalog = JointGeometryCatalog(geometry, families)
    miss, hit = math.log(.9 / .3), math.log(.1 / .7)
    result = catalog.infer([[3 * miss, 9 * miss], [3 * miss, 3 * miss + 6 * hit]])
    assert result.map_family_ids == (("long_family",), ("short_family",))
    assert result.family_marginals[0, 1] > .99
    assert result.family_marginals[1, 0] > .9


def test_negative_infinite_evidence_and_empty_catalog_are_well_defined():
    geometry, families = _fixture()
    result = JointGeometryCatalog(geometry, families).infer([[-math.inf] * 3])
    assert result.log_z_observation == pytest.approx([0])
    assert result.family_marginals == pytest.approx(np.zeros((1, 2)))
    assert result.geometry_marginals == pytest.approx(np.zeros((1, 3)))
    assert result.map_family_ids == ((),)
    empty = JointGeometryCatalog([], []).infer(np.zeros((2, 0)))
    assert empty.log_marginal_likelihood_ratio.tolist() == [0, 0]
    assert empty.configuration_probability([]) == 1


def test_batch_size_order_and_reflection_invariance():
    geometry, families = _fixture()
    evidence = np.array([[1, 3, 2], [2, -1, .3]])
    original = JointGeometryCatalog(geometry, families).infer(evidence, [.2, -.7], batch_size=1)
    reflected = [GeometryInterval(g.geometry_id, -g.end, -g.start) for g in geometry]
    changed = JointGeometryCatalog(reflected, list(reversed(families))).infer(evidence, [-.7, .2], batch_size=9)
    assert changed.log_z_observation == pytest.approx(original.log_z_observation, abs=1e-12)
    assert changed.family_marginals[:, ::-1] == pytest.approx(original.family_marginals, abs=1e-12)
    assert changed.geometry_marginals == pytest.approx(original.geometry_marginals, abs=1e-12)
    assert changed.map_geometry_ids == original.map_geometry_ids


def test_exact_budget_failures_never_silently_change_the_model():
    geometry, families = _fixture()
    with pytest.raises(ExactGeometryBudgetError, match="budget"):
        JointGeometryCatalog(geometry, families, max_tilings_per_component=2)
    with pytest.raises(ExactGeometryBudgetError):
        JointGeometryCatalog(geometry, families, max_enumerated_families=1)
    catalog = JointGeometryCatalog(geometry, families, max_output_cells=1)
    with pytest.raises(ExactGeometryBudgetError, match="output"):
        catalog.infer([[0, 0, 0]])
    with pytest.raises(ExactGeometryBudgetError, match="Evidence"):
        JointGeometryCatalog(geometry, families, max_evidence_cells=2).infer([[0, 0, 0]])


def test_invalid_partition_weights_and_activities():
    geometry, families = _fixture()
    with pytest.raises(ValueError, match="partition"):
        JointGeometryCatalog(geometry, [families[0], FamilyGeometry("B", (1, 2))])
    with pytest.raises(ValueError, match="Every geometry"):
        JointGeometryCatalog(geometry, [families[0]])
    with pytest.raises(ValueError, match="weights"):
        JointGeometryCatalog(geometry, [FamilyGeometry("A", (0, 1), (.5, 0)), families[1]])
    catalog = JointGeometryCatalog(geometry, families)
    with pytest.raises(ValueError, match="Activities"):
        catalog.infer([[0, 0, 0]], [math.inf, 0])
    with pytest.raises(ValueError, match="Evidence"):
        catalog.infer([[0, math.nan, 0]])
