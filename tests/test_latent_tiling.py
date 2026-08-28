import math

import numpy as np
import pytest

from fiberhmm.inference.latent_tiling import (
    DddARadialChemistryScorer,
    TilingConfiguration,
    TilingSegment,
    family_quantification_weights,
    normalize_configuration_priors,
    score_tiling_configurations,
    score_tiling_groups,
)
from fiberhmm.inference.nuc_recaller import NucProfile
from fiberhmm.inference.strand_rescue import ReadEvidence


def _read(*, positions=(5, 15, 25), steps=(2.0, 2.0, 2.0)):
    return ReadEvidence(
        name="molecule",
        strand="CT",
        ref_start=0,
        ref_end=40,
        positions=np.asarray(positions, dtype=np.int64),
        steps=np.asarray(steps, dtype=np.float64),
        hits=np.asarray([True] * len(positions), dtype=bool),
        contexts=np.zeros(len(positions), dtype=np.int64),
        tfs=[],
        nucs=[],
        msps=[],
        library_id="library",
    )


def test_contiguous_tf_nuc_and_broad_nuc_are_chemistry_equivalent():
    read = _read()
    result = score_tiling_configurations(
        read,
        [
            TilingConfiguration(
                "broad_nuc",
                (TilingSegment("nuc", 0, 40),),
                log_structural_weight=0.0,
            ),
            TilingConfiguration(
                "family_plus_nuc",
                (
                    TilingSegment("tf_family", 0, 10, "family_1"),
                    TilingSegment("nuc", 10, 40),
                ),
                log_structural_weight=math.log(20.0),
            ),
        ],
    )

    records = {row["configuration_id"]: row for row in result["configurations"]}
    assert records["broad_nuc"]["chemistry_log_likelihood"] == 6.0
    assert records["family_plus_nuc"]["chemistry_log_likelihood"] == 6.0
    assert (
        records["broad_nuc"]["chemistry_equivalence_class"]
        == records["family_plus_nuc"]["chemistry_equivalence_class"]
    )
    assert result["decision"] == "family_plus_nuc"
    assert result["decision_basis"].startswith("population_topology")
    assert result["family_marginal_posteriors"]["family_1"] == pytest.approx(
        20.0 / 21.0
    )


def test_accessible_gap_can_make_tilings_chemically_distinct():
    read = _read(positions=(5, 15, 25), steps=(4.0, -5.0, 4.0))
    result = score_tiling_configurations(
        read,
        [
            TilingConfiguration(
                "broad_nuc", (TilingSegment("nuc", 0, 30),)
            ),
            TilingConfiguration(
                "family_gap_nuc",
                (
                    TilingSegment("tf_family", 0, 10, "family_1"),
                    TilingSegment("accessible", 10, 20),
                    TilingSegment("nuc", 20, 30),
                ),
            ),
        ],
    )

    assert result["decision"] == "family_gap_nuc"
    assert result["decision_basis"] == "chemistry_and_population_topology"
    records = {row["configuration_id"]: row for row in result["configurations"]}
    assert records["broad_nuc"]["chemistry_log_likelihood"] == 3.0
    assert records["family_gap_nuc"]["chemistry_log_likelihood"] == 8.0


def test_family_marginal_sums_every_configuration_containing_family_once():
    result = score_tiling_configurations(
        _read(),
        [
            TilingConfiguration("nuc", (TilingSegment("nuc", 0, 40),)),
            TilingConfiguration(
                "f1_nuc",
                (
                    TilingSegment("tf_family", 0, 10, "f1"),
                    TilingSegment("nuc", 10, 40),
                ),
            ),
            TilingConfiguration(
                "f1_f2_nuc",
                (
                    TilingSegment("tf_family", 0, 10, "f1"),
                    TilingSegment("tf_family", 10, 20, "f2"),
                    TilingSegment("nuc", 20, 40),
                ),
            ),
        ],
        minimum_resolved_posterior=0.9,
    )

    assert result["decision"] == "unresolved"
    assert result["family_marginal_posteriors"]["f1"] == pytest.approx(2.0 / 3.0)
    assert result["family_marginal_posteriors"]["f2"] == pytest.approx(1.0 / 3.0)


def test_custom_chemistry_scorer_can_distinguish_segment_types():
    configurations = [
        TilingConfiguration("nuc", (TilingSegment("nuc", 0, 40),)),
        TilingConfiguration(
            "family_nuc",
            (
                TilingSegment("tf_family", 0, 10, "f1"),
                TilingSegment("nuc", 10, 40),
            ),
        ),
    ]

    def scorer(_read, configuration):
        value = 0.0 if configuration.configuration_id == "nuc" else 5.0
        return value, 3, configuration.configuration_id

    result = score_tiling_configurations(
        _read(), configurations, chemistry_scorer=scorer
    )
    assert result["decision"] == "family_nuc"
    assert result["decision_basis"] == "chemistry_and_population_topology"


def test_custom_chemistry_scorer_declares_population_only_equivalence():
    configurations = [
        TilingConfiguration("nuc", (TilingSegment("nuc", 0, 40),)),
        TilingConfiguration(
            "family_nuc",
            (
                TilingSegment("tf_family", 0, 10, "f1"),
                TilingSegment("nuc", 10, 40),
            ),
            log_structural_weight=math.log(20),
        ),
    ]

    result = score_tiling_configurations(
        _read(),
        configurations,
        chemistry_scorer=lambda _read, _configuration: (3.0, 3, "same"),
    )
    assert result["decision"] == "family_nuc"
    assert result["decision_basis"].startswith("population_topology")


def test_ddda_radial_scorer_uses_distinct_tf_and_nucleosome_emissions():
    read = _read(positions=(5, 15, 25), steps=(3.0, 3.0, 3.0))
    profile = NucProfile(
        radial=np.asarray([0.1] * 30),
        linker=0.5,
        half=29,
        min_sep=20,
        edge_frac=0.8,
    )
    scorer = DddARadialChemistryScorer(profile)
    tf = TilingConfiguration(
        "tf", (TilingSegment("tf_family", 0, 30, "f1"),)
    )
    nuc = TilingConfiguration("nuc", (TilingSegment("nuc", 0, 30),))
    tf_score, tf_opportunities, tf_key = scorer(read, tf)
    nuc_score, nuc_opportunities, nuc_key = scorer(read, nuc)
    assert tf_score == 9.0
    assert nuc_score == pytest.approx(3.0 * math.log(0.1 / 0.5))
    assert tf_opportunities == nuc_opportunities == 3
    assert tf_key != nuc_key


def test_ddda_radial_scorer_applies_read_efficiency_to_both_rates():
    read = _read(positions=(5, 15), steps=(1.0, 1.0))
    read.hits = np.asarray([True, False], dtype=bool)
    read.efficiency_factor = 0.5
    profile = NucProfile(
        radial=np.asarray([0.2] * 30),
        linker=0.6,
        half=29,
        min_sep=20,
        edge_frac=0.8,
    )
    score, opportunities, _key = DddARadialChemistryScorer(profile)(
        read, TilingConfiguration("nuc", (TilingSegment("nuc", 0, 30),))
    )
    expected = math.log(0.1 / 0.3) + math.log(0.9 / 0.7)
    assert score == pytest.approx(expected)
    assert opportunities == 2


def test_ddda_radial_scorer_can_share_context_aware_accessible_baseline():
    read = _read(positions=(5, 15), steps=(1.0, 1.0))
    read.hits = np.asarray([True, False], dtype=bool)
    read.efficiency_factor = 0.5
    profile = NucProfile(
        radial=np.asarray([0.2] * 30),
        linker=0.6,
        half=29,
        min_sep=20,
        edge_frac=0.8,
    )
    scorer = DddARadialChemistryScorer(
        profile, accessible_hit_rates=np.asarray([0.8])
    )
    score, _opportunities, _key = scorer(
        read, TilingConfiguration("nuc", (TilingSegment("nuc", 0, 30),))
    )
    expected = math.log(0.1 / 0.4) + math.log(0.9 / 0.6)
    assert score == pytest.approx(expected)


def test_ddda_radial_scorer_scores_full_variable_span_around_explicit_dyad():
    read = _read(positions=(5, 15, 25, 35), steps=(1.0,) * 4)
    read.hits = np.asarray([True, True, True, True], dtype=bool)
    profile = NucProfile(
        radial=np.asarray([0.1, 0.2, 0.3, 0.4] + [0.4] * 40),
        linker=0.5,
        half=3,
        min_sep=4,
        edge_frac=0.8,
    )
    score, opportunities, key = DddARadialChemistryScorer(profile)(
        read,
        TilingConfiguration(
            "broad_nuc",
            (TilingSegment("nuc", 0, 40, dyad=15),),
        ),
    )
    expected = sum(math.log(rate / 0.5) for rate in (0.4, 0.1, 0.4, 0.4))
    assert score == pytest.approx(expected)
    assert opportunities == 4
    assert key == (("nuc_radial", 0, 40, 15.0),)


def test_explicit_nucleosome_dyad_must_be_inside_span():
    with pytest.raises(ValueError, match="dyad must lie inside"):
        TilingSegment("nuc", 0, 40, dyad=40)
    with pytest.raises(ValueError, match="only nucleosome"):
        TilingSegment("tf_family", 0, 10, "f1", dyad=5)


def test_ddda_radial_scorer_rejects_saturated_efficiency_adjustment():
    read = _read()
    read.efficiency_factor = 2.0
    profile = NucProfile(
        radial=np.asarray([0.1] * 30),
        linker=0.6,
        half=29,
        min_sep=20,
        edge_frac=0.8,
    )
    with pytest.raises(ValueError, match="linker rate"):
        DddARadialChemistryScorer(profile)(
            read, TilingConfiguration("nuc", (TilingSegment("nuc", 0, 30),))
        )


def test_incomplete_mapping_is_ineligible():
    read = _read()
    result = score_tiling_configurations(
        read,
        [TilingConfiguration("nuc", (TilingSegment("nuc", -1, 40),))],
    )
    assert result["status"] == "ineligible_incomplete_mapping"
    assert result["decision"] == "ineligible"


def test_family_quantification_reports_soft_and_conservative_counts():
    summary = family_quantification_weights(
        [
            {"family_marginal_posteriors": {"f1": 0.95, "f2": 0.2}},
            {"family_marginal_posteriors": {"f1": 0.8, "f2": 0.91}},
        ],
        minimum_marginal_posterior=0.9,
    )
    assert summary["f1"]["posterior_weighted_occupancy"] == pytest.approx(1.75)
    assert summary["f1"]["conservative_occupied_molecules"] == 1
    assert summary["f2"]["conservative_occupied_molecules"] == 1


def test_invalid_overlapping_protected_segments_are_rejected():
    with pytest.raises(ValueError, match="may not overlap"):
        TilingConfiguration(
            "overlap",
            (
                TilingSegment("tf_family", 0, 15, "f1"),
                TilingSegment("nuc", 10, 30),
            ),
        )


def test_unanchored_tf_is_protected_but_has_no_family_marginal():
    result = score_tiling_configurations(
        _read(),
        [
            TilingConfiguration(
                "spatial_null", (TilingSegment("unanchored_tf", 0, 10),)
            ),
            TilingConfiguration("nuc", (TilingSegment("nuc", 0, 40),)),
        ],
    )
    assert result["family_marginal_posteriors"] == {}
    spatial = next(
        row for row in result["configurations"]
        if row["configuration_id"] == "spatial_null"
    )
    assert spatial["chemistry_log_likelihood"] == 2.0


def test_configuration_group_normalization_restores_unit_mass_after_drops():
    normalized, raw_log_mass = normalize_configuration_priors(
        [
            TilingConfiguration(
                "a",
                (TilingSegment("nuc", 0, 20),),
                log_structural_weight=math.log(0.1),
            ),
            TilingConfiguration(
                "b",
                (TilingSegment("nuc", 20, 40),),
                log_structural_weight=math.log(0.2),
            ),
        ]
    )
    assert raw_log_mass == pytest.approx(math.log(0.3))
    assert sum(math.exp(value.log_structural_weight) for value in normalized) == pytest.approx(1.0)


def test_streaming_group_score_matches_explicit_group_evidence():
    read = _read(positions=(5, 15, 25), steps=(2.0, -1.0, 3.0))
    groups = {
        "nuc": normalize_configuration_priors(
            [TilingConfiguration("n", (TilingSegment("nuc", 0, 30),))]
        )[0],
        "family": normalize_configuration_priors(
            [
                TilingConfiguration(
                    "f_left", (TilingSegment("tf_family", 0, 10, "f1"),)
                ),
                TilingConfiguration(
                    "f_right", (TilingSegment("tf_family", 20, 30, "f1"),)
                ),
            ]
        )[0],
    }
    streamed = score_tiling_groups(read, groups)
    explicit = score_tiling_configurations(
        read, [*groups["nuc"], *groups["family"]]
    )
    expected = {}
    for group_name, ids in {
        "nuc": {"n"},
        "family": {"f_left", "f_right"},
    }.items():
        joints = [
            row["log_joint"]
            for row in explicit["configurations"]
            if row["configuration_id"] in ids
        ]
        expected[group_name] = math.log(sum(math.exp(value) for value in joints))
    assert streamed["status"] == "scored_normalized_groups"
    assert streamed["groups"]["nuc"]["log_evidence"] == pytest.approx(expected["nuc"])
    assert streamed["groups"]["family"]["log_evidence"] == pytest.approx(expected["family"])
    assert sum(group["posterior"] for group in streamed["groups"].values()) == pytest.approx(1.0)


def test_segment_cached_radial_group_score_matches_uncached_scorer():
    read = _read(positions=(5, 15, 25, 35), steps=(2.0, -1.0, 3.0, 0.5))
    read.hits = np.asarray([True, False, True, False], dtype=bool)
    profile = NucProfile(
        radial=np.linspace(0.08, 0.25, 50),
        linker=0.5,
        half=20,
        min_sep=30,
        edge_frac=0.8,
    )
    scorer = DddARadialChemistryScorer(profile)
    shared = TilingSegment("nuc", 10, 40, dyad=25)
    groups = {
        "nuc": normalize_configuration_priors(
            [TilingConfiguration("n", (TilingSegment("nuc", 0, 40, dyad=20),))]
        )[0],
        "family": normalize_configuration_priors(
            [
                TilingConfiguration(
                    "f1", (TilingSegment("tf_family", 0, 10, "f"), shared)
                ),
                TilingConfiguration(
                    "f2", (TilingSegment("tf_family", 0, 8, "f"), shared)
                ),
            ]
        )[0],
    }
    cached = score_tiling_groups(read, groups, chemistry_scorer=scorer)
    uncached = score_tiling_groups(
        read,
        groups,
        chemistry_scorer=lambda current_read, configuration: scorer(
            current_read, configuration
        ),
    )
    for group_name in groups:
        assert cached["groups"][group_name]["log_evidence"] == pytest.approx(
            uncached["groups"][group_name]["log_evidence"]
        )
        assert (
            cached["groups"][group_name]["map_configuration_id"]
            == uncached["groups"][group_name]["map_configuration_id"]
        )
    assert cached["groups"]["family"]["map_segments"][-1]["dyad"] == 25
