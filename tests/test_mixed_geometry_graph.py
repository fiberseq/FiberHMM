"""Exact mixed-block factorization gates against global geometry enumeration."""
import itertools
import math

import numpy as np
import pytest

from fiberhmm.inference.joint_geometry_catalog import (
    ExactGeometryBudgetError, FamilyGeometry, GeometryInterval, JointGeometryCatalog,
)
from fiberhmm.inference.mixed_geometry_graph import MixedGeometryGraphCatalog, _checked_probabilities
from fiberhmm.inference.state_geometry import GeometryFamily, StateGeometry, infer_joint_geometry


def _reference(geometries, families, evidence, activities):
    return infer_joint_geometry([
        GeometryFamily(f.family_id, tuple(StateGeometry(geometries[g].geometry_id,
                        geometries[g].start, geometries[g].end, evidence[g], w)
                       for g, w in zip(f.geometry_indices, f.weights or (1,) * len(f.geometry_indices))))
        for f in families], dict(zip((f.family_id for f in families), activities)))


def _two_blocks():
    geometries = [GeometryInterval("As", 0, 4), GeometryInterval("Al", 0, 8),
                  GeometryInterval("B", 6, 10), GeometryInterval("Cs", 8, 12),
                  GeometryInterval("Cl", 8, 16), GeometryInterval("D", 14, 18)]
    families = [FamilyGeometry("A", (0, 1), (.3, .7)), FamilyGeometry("B", (2,)),
                FamilyGeometry("C", (3, 4), (.6, .4)), FamilyGeometry("D", (5,))]
    return geometries, families


def test_two_mixed_blocks_with_always_conflict_have_exact_global_normalization():
    geometries, families = _two_blocks()
    catalog = MixedGeometryGraphCatalog(geometries, families)
    assert [len(b.families) for b in catalog.blocks] == [2, 2]
    assert catalog.always_conflicting_family_pairs == ((1, 2),)
    evidence = np.array([[0, math.log(100), .3, 1.2, -2, .1], [2, -1, 3, 1, 4, -.3]])
    activities = np.array([.3, -.2, .7, -.8])
    result = catalog.infer(evidence, activities, batch_size=1)
    global_reference = JointGeometryCatalog(geometries, families).infer(evidence, activities)
    assert result.log_z_observation == pytest.approx(global_reference.log_z_observation, abs=1e-12)
    assert result.log_z_prior == pytest.approx(global_reference.log_z_prior, abs=1e-12)
    assert result.family_marginals == pytest.approx(global_reference.family_marginals, abs=1e-12)
    assert result.geometry_marginals == pytest.approx(global_reference.geometry_marginals, abs=1e-12)
    assert result.prior_geometry_marginals == pytest.approx(global_reference.prior_geometry_marginals, abs=1e-12)
    assert result.equal_prior_family_marginals == pytest.approx(global_reference.equal_prior_family_marginals, abs=1e-12)
    assert result.equal_prior_geometry_marginals == pytest.approx(global_reference.equal_prior_geometry_marginals, abs=1e-12)
    for row in range(len(evidence)):
        reference = _reference(geometries, families, evidence[row], activities)
        for config, probability, prior in zip(reference.configurations, reference.configuration_posteriors, reference.configuration_priors):
            assert result.configuration_probability(config, row) == pytest.approx(probability, abs=1e-12)
            assert result.configuration_probability(config, row, prior=True) == pytest.approx(prior, abs=1e-12)
        assert result.map_family_ids[row] == reference.map_state_ids
    assert result.configuration_probability(["B", "C"]) == 0


@pytest.mark.parametrize("seed", range(8))
def test_random_mixed_blocks_match_global_reference(seed):
    rng = np.random.default_rng(seed + 610)
    geometries, families = [], []
    for f in range(5):
        indices = []
        for g in range(int(rng.integers(1, 4))):
            start = int(rng.integers(-10, 35))
            indices.append(len(geometries))
            geometries.append(GeometryInterval(f"g{f}_{g}", start, start + int(rng.integers(1, 10))))
        families.append(FamilyGeometry(f"f{f}", tuple(indices), tuple(rng.uniform(.1, 2, len(indices)))))
    evidence, activities = rng.uniform(-5, 5, (3, len(geometries))), rng.uniform(-1, 1, len(families))
    catalog = MixedGeometryGraphCatalog(geometries, families)
    result = catalog.infer(evidence, activities, batch_size=2)
    global_result = JointGeometryCatalog(geometries, families).infer(evidence, activities)
    assert result.log_marginal_likelihood_ratio == pytest.approx(global_result.log_marginal_likelihood_ratio, abs=1e-12)
    assert result.family_marginals == pytest.approx(global_result.family_marginals, abs=1e-12)
    assert result.geometry_marginals == pytest.approx(global_result.geometry_marginals, abs=1e-12)
    for row in range(len(evidence)):
        reference = _reference(geometries, families, evidence[row], activities)
        assert result.map_family_ids[row] == reference.map_state_ids
        for config, probability in zip(reference.configurations, reference.configuration_posteriors):
            assert result.configuration_probability(config, row) == pytest.approx(probability, abs=1e-12)
        selected = [geometries[g] for g in result.map_geometry_indices[row]]
        assert all(not (a.start < b.end and b.start < a.end) for a, b in itertools.combinations(selected, 2))


def test_short_long_fixture_and_best_geometry_ignore_activity_within_subset():
    geometry = [GeometryInterval("short", 0, 4), GeometryInterval("long", 0, 8), GeometryInterval("B", 6, 10)]
    families = [FamilyGeometry("A", (0, 1), (.5, .5)), FamilyGeometry("B", (2,))]
    catalog = MixedGeometryGraphCatalog(geometry, families)
    result = catalog.infer([[0, math.log(100), 0]])
    assert result.configuration_probability(["A", "B"]) == pytest.approx(1 / 53.5, abs=1e-14)
    assert result.configuration_probability(["A", "B"], prior=True) == pytest.approx(.25)
    assert result.map_geometry_ids == (("long",),)
    forced = catalog.infer([[0, math.log(100), math.log(1000)]], [2, 0])
    assert forced.map_family_ids == (("A", "B"),)
    assert forced.map_geometry_ids == (("B", "short"),)


def test_duplicate_weight_split_invariance():
    geometry, families = _two_blocks()
    evidence = np.array([[1, 2, 3, -1, 1, .3]])
    original = MixedGeometryGraphCatalog(geometry, families).infer(evidence)
    duplicated = geometry + [GeometryInterval("Al_alias", 0, 8)]
    changed_families = [FamilyGeometry("A", (0, 1, 6), (.3, .35, .35)), *families[1:]]
    changed = MixedGeometryGraphCatalog(duplicated, changed_families).infer(np.c_[evidence, evidence[:, 1]])
    assert changed.log_marginal_likelihood_ratio == pytest.approx(original.log_marginal_likelihood_ratio, abs=1e-12)
    assert changed.family_marginals == pytest.approx(original.family_marginals, abs=1e-12)
    assert changed.geometry_marginals[0, 1] + changed.geometry_marginals[0, 6] == pytest.approx(original.geometry_marginals[0, 1], abs=1e-12)


def test_fifty_eight_family_block_enumerates_feasible_subsets_not_dense_power_set():
    # Every family can occupy left OR right, so at most two families coexist.
    # There are 1 + 58 + C(58,2) feasible family subsets, never 2**58 masks.
    geometries, families = [], []
    for f in range(58):
        geometries.extend([GeometryInterval(f"g{f:02}L", 0, 2), GeometryInterval(f"g{f:02}R", 10, 12)])
        families.append(FamilyGeometry(f"f{f:02}", (2 * f, 2 * f + 1)))
    catalog = MixedGeometryGraphCatalog(geometries, families)
    assert len(catalog.blocks) == 1
    assert len(catalog.blocks[0].families) == 58
    # All nonempty subsets have the same empty external neighborhood. Their
    # exact sum becomes one node, without merging family identities.
    assert len(catalog.nodes) == 1
    assert len(catalog.nodes[0].configuration_indices) == 58 + math.comb(58, 2)
    assert catalog.total_tilings == 1 + 2 * 58 + 2 * math.comb(58, 2)
    result = catalog.infer(np.zeros((1, len(geometries))), include_map=False)
    assert result.log_z_prior == pytest.approx(math.log(1 + 58 + math.comb(58, 2)), abs=1e-12)
    assert result.log_marginal_likelihood_ratio == pytest.approx([0], abs=1e-12)


def test_compressed_group_sum_map_is_not_the_family_configuration_map():
    geometries = [GeometryInterval("short", 0, 4), GeometryInterval("long", 0, 8),
                  GeometryInterval("B", 6, 10)]
    families = [FamilyGeometry("A", (0, 1)), FamilyGeometry("B", (2,))]
    catalog = MixedGeometryGraphCatalog(geometries, families)
    assert len(catalog.nodes) == 1
    result = catalog.infer([[-.7, -.7, -.7]])
    # Sum over three nonempty configurations exceeds the empty weight, but
    # every individual family configuration is worse than empty.
    assert result.node_log_likelihood_ratios[0, 0] > 0
    assert catalog.graph.map_indices(result.node_log_likelihood_ratios)[0] == (0,)
    assert result.map_family_ids == ((),)
    reference = JointGeometryCatalog(geometries, families).infer([[-.7, -.7, -.7]])
    assert result.family_marginals == pytest.approx(reference.family_marginals, abs=1e-12)
    assert result.map_geometry_ids == reference.map_geometry_ids


def test_gradient_fitting_and_explicit_uniform_center():
    geometry, families = _two_blocks()
    evidence = np.array([[1, 3, .1, -.2, 1, 2], [-2, .4, 2, .3, -.5, 1], [.7, 2, -.4, .2, 1, -3]])
    catalog = MixedGeometryGraphCatalog(geometry, families)
    eta = np.array([.2, -.4, .7, -.1])
    result = catalog.infer(evidence, eta)
    for f in range(len(families)):
        plus, minus = eta.copy(), eta.copy()
        plus[f] += 1e-5; minus[f] -= 1e-5
        numeric = (catalog.infer(evidence, plus).log_marginal_likelihood_ratio.sum() -
                   catalog.infer(evidence, minus).log_marginal_likelihood_ratio.sum()) / 2e-5
        assert result.activity_gradient[:, f].sum() == pytest.approx(numeric, abs=1e-9)
    fit = catalog.fit_activities(evidence, prior_center=0, regularization=.8, batch_size=1)
    assert fit["converged"]
    assert fit["activity_prior_center"] == [0] * len(families)
    fitted = catalog.infer(evidence, fit["activities"])
    assert np.max(np.abs(fitted.activity_gradient.sum(axis=0) - .8 * np.array(fit["activities"]))) < 1e-5


def test_native_long_short_evidence_and_equal_prior_remain_separate_from_population_prior():
    geometry = [GeometryInterval("short", 0, 3), GeometryInterval("long", 0, 11)]
    families = [FamilyGeometry("short", (0,)), FamilyGeometry("long", (1,))]
    catalog = MixedGeometryGraphCatalog(geometry, families)
    miss, hit = math.log(.9 / .3), math.log(.1 / .7)
    evidence = [[3 * miss, 11 * miss], [3 * miss, 3 * miss + 8 * hit]]
    native = catalog.infer(evidence)
    biased = catalog.infer(evidence, [8, -8])
    assert native.map_family_ids == (("long",), ("short",))
    assert biased.equal_prior_family_marginals == pytest.approx(native.family_marginals, abs=1e-12)
    assert not np.allclose(biased.family_marginals, native.family_marginals)


def test_zero_activities_reuse_equal_graph_without_changing_outputs(monkeypatch):
    geometry, families = _two_blocks()
    catalog = MixedGeometryGraphCatalog(geometry, families)
    evidence = [[1, 2, -.3, .7, -1, .4]]
    calls = []
    original = catalog.graph.infer
    def tracked(weights):
        calls.append(np.asarray(weights).copy())
        return original(weights)
    monkeypatch.setattr(catalog.graph, "infer", tracked)
    full = catalog.infer(evidence, include_equal_prior=True)
    assert len(calls) == 2  # prior plus one shared native/equal observation pass
    assert full.equal_prior_family_marginals == pytest.approx(full.family_marginals)
    assert full.equal_prior_geometry_marginals == pytest.approx(full.geometry_marginals)
    calls.clear()
    compact = catalog.infer(evidence, include_equal_prior=False)
    assert len(calls) == 2
    assert compact.equal_prior_family_marginals is None
    assert compact.family_marginals == pytest.approx(full.family_marginals)
    assert compact.map_geometry_ids == full.map_geometry_ids


def test_empty_infinite_and_batching():
    empty = MixedGeometryGraphCatalog([], []).infer(np.zeros((2, 0)))
    assert empty.log_z_observation.tolist() == [0, 0]
    assert empty.configuration_probability([]) == 1
    geometry, families = _two_blocks()
    catalog = MixedGeometryGraphCatalog(geometry, families)
    impossible = catalog.infer([[-math.inf] * len(geometry)])
    assert impossible.family_marginals.sum() == 0
    assert impossible.geometry_marginals.sum() == 0
    assert impossible.map_family_ids == ((),)
    values = np.arange(18).reshape(3, 6) / 10
    a, b = catalog.infer(values, batch_size=1), catalog.infer(values, batch_size=128)
    assert a.family_marginals == pytest.approx(b.family_marginals, abs=1e-12)
    assert a.geometry_marginals == pytest.approx(b.geometry_marginals, abs=1e-12)
    assert a.map_geometry_ids == b.map_geometry_ids


def test_budget_and_partition_failures_are_explicit():
    geometry, families = _two_blocks()
    with pytest.raises(ExactGeometryBudgetError):
        MixedGeometryGraphCatalog(geometry, families, max_tilings_per_block=2)
    with pytest.raises(ExactGeometryBudgetError):
        MixedGeometryGraphCatalog(geometry, families, max_superstates=1)
    with pytest.raises(ValueError, match="partition"):
        MixedGeometryGraphCatalog(geometry, [families[0], FamilyGeometry("bad", (0, 2, 3, 4, 5))])
    with pytest.raises(ExactGeometryBudgetError):
        MixedGeometryGraphCatalog(geometry, families, max_output_cells=1).infer([[0] * len(geometry)])


def test_probability_roundoff_repair_is_bounded_and_does_not_hide_invalid_results():
    values = np.array([-2e-15, 0, .37, 1, 1 + 2e-15])
    assert _checked_probabilities(values, "test") == 2
    assert values.tolist() == [0, 0, .37, 1, 1]
    for invalid in (-1e-8, 1 + 1e-8, np.nan, np.inf, -np.inf):
        with pytest.raises(ValueError, match="beyond numerical tolerance"):
            _checked_probabilities(np.array([invalid]), "test")


def test_extreme_likelihoods_return_valid_probability_domain():
    geometry, families = _two_blocks()
    catalog = MixedGeometryGraphCatalog(geometry, families)
    result = catalog.infer([[700, -700, 700, -700, 700, -700], [-700, 700, -700, 700, -700, 700]])
    for field in ("family_marginals", "geometry_marginals", "prior_family_marginals",
                  "prior_geometry_marginals", "node_marginals", "prior_node_marginals"):
        values = getattr(result, field)
        assert np.all(np.isfinite(values))
        assert np.all((values >= 0) & (values <= 1))
