from __future__ import annotations

import random

import pytest

import fiberhmm.inference.tf_sites as tf_sites

from fiberhmm.inference.tf_sites import (
    BaselineMolecule,
    SiteDiscoveryConfig,
    TFObservation,
    build_tf_site_catalog,
)


def test_accelerated_site_denominators_match_exact_python_fallback(monkeypatch):
    if tf_sites._site_coverage_counts_numba is None:
        pytest.skip("numba acceleration is not installed")
    molecules = []
    for family in range(80):
        start = 20 + 12 * family
        for replicate in range(3):
            # Mix contiguous and gapped mappings plus containing/non-containing
            # MSPs so both accelerated count matrices are exercised.
            blocks = (
                ((0, 1000),)
                if replicate == 0
                else ((0, start + 5), (start + 6, 1000))
            )
            molecules.append(
                _molecule(
                    f"family-{family}-replicate-{replicate}",
                    tfs=((start, start + 10, True),),
                    msps=((start - 2, start + 12),) if replicate != 2 else (),
                    blocks=blocks,
                    stratum="CT" if replicate % 2 else "GA",
                )
            )

    accelerated = build_tf_site_catalog(molecules)
    monkeypatch.setattr(tf_sites, "_site_coverage_counts_numba", None)
    reference = build_tf_site_catalog(molecules)

    assert accelerated == reference


def _molecule(
    name,
    *,
    tfs=(),
    msps=(),
    blocks=((0, 1000),),
    stratum=".",
    contig="chr1",
):
    return BaselineMolecule(
        molecule_id=name,
        contig=contig,
        stratum=stratum,
        mapped_blocks=tuple(blocks),
        tfs=tuple(
            TFObservation(
                call_id=f"{name}-tf-{index}",
                start=start,
                end=end,
                geometry_eligible=geometry_eligible,
            )
            for index, (start, end, geometry_eligible) in enumerate(tfs)
        ),
        msps=tuple(msps),
    )


def test_center_smoothing_is_subdivided_by_compatible_edges():
    molecules = [_molecule(f"short-{index}", tfs=((100, 120, True),)) for index in range(6)] + [
        _molecule(f"wide-{index}", tfs=((90, 130, True),)) for index in range(6)
    ]

    catalog = build_tf_site_catalog(
        molecules,
        config=SiteDiscoveryConfig(edge_compatibility_bp=5),
    )

    assert [(site.start, site.end, site.summit) for site in catalog.sites] == [
        (90, 130, 110),
        (100, 120, 110),
    ]
    assert [site.cluster_support_molecules for site in catalog.sites] == [6, 6]
    assert all(site.population_ready for site in catalog.sites)
    assert catalog.diagnostics.unassigned_tf_calls == 0


def test_nested_geometry_families_share_a_stable_parent_locus():
    molecules = [_molecule(f"short-{index}", tfs=((100, 120, True),)) for index in range(6)] + [
        _molecule(f"wide-{index}", tfs=((90, 130, True),)) for index in range(6)
    ]

    catalog = build_tf_site_catalog(
        molecules,
        config=SiteDiscoveryConfig(edge_compatibility_bp=5),
    )

    assert len(catalog.loci) == 1
    locus = catalog.loci[0]
    assert (locus.start, locus.summit, locus.end) == (90, 110, 130)
    assert [site.family_index for site in catalog.sites] == [1, 2]
    assert {site.locus_id for site in catalog.sites} == {locus.locus_id}
    assert {site.locus_summit for site in catalog.sites} == {110}
    assert locus.family_site_ids == tuple(site.site_id for site in catalog.sites)

    # Site IDs predate the parent-locus model and are durable foreign keys.
    assert [site.site_id for site in catalog.sites] == [
        "tfsite_dd6d908eeadb48b1",
        "tfsite_468f24183b9832e1",
    ]
    record = catalog.sites[0].as_record()
    assert {key: record[key] for key in ("locus_id", "locus_summit", "family_index")} == {
        "locus_id": locus.locus_id,
        "locus_summit": 110,
        "family_index": 1,
    }


def test_one_molecule_can_support_overlapping_families_in_one_locus():
    molecules = [
        *[_molecule(f"short-{index}", tfs=((100, 120, True),)) for index in range(3)],
        *[_molecule(f"wide-{index}", tfs=((90, 130, True),)) for index in range(3)],
        _molecule(
            "dual",
            tfs=((90, 130, True), (100, 120, True)),
        ),
    ]

    catalog = build_tf_site_catalog(
        molecules,
        config=SiteDiscoveryConfig(edge_compatibility_bp=5),
    )
    dual = [assignment for assignment in catalog.assignments if assignment.molecule_id == "dual"]

    assert len(dual) == 2
    assert len({assignment.site_id for assignment in dual}) == 2
    assert {assignment.site_id for assignment in dual} == {site.site_id for site in catalog.sites}
    assert len({site.locus_id for site in catalog.sites}) == 1


def test_edge_compatibility_threshold_is_inclusive():
    base = [_molecule(f"base-{index}", tfs=((100, 120, True),)) for index in range(3)]
    at_threshold = [_molecule(f"edge-{index}", tfs=((105, 125, True),)) for index in range(3)]

    joined = build_tf_site_catalog(
        base + at_threshold,
        config=SiteDiscoveryConfig(edge_compatibility_bp=5),
    )
    split = build_tf_site_catalog(
        base + [_molecule(f"outside-{index}", tfs=((106, 126, True),)) for index in range(3)],
        config=SiteDiscoveryConfig(edge_compatibility_bp=5),
    )

    assert len(joined.sites) == 1
    assert len(split.sites) == 2


def test_edge_compatibility_cannot_chain_distant_boundaries():
    molecules = []
    for family, (start, end) in enumerate(((90, 130), (95, 125), (100, 120))):
        molecules.extend(
            _molecule(
                f"family-{family}-{replicate}",
                tfs=((start, end, True),),
            )
            for replicate in range(3)
        )

    catalog = build_tf_site_catalog(
        molecules,
        config=SiteDiscoveryConfig(edge_compatibility_bp=5),
    )

    assert len(catalog.sites) == 2
    for site in catalog.sites:
        members = [
            assignment for assignment in catalog.assignments if assignment.site_id == site.site_id
        ]
        assert max(value.start for value in members) - min(value.start for value in members) <= 5
        assert max(value.end for value in members) - min(value.end for value in members) <= 5
    assert catalog.diagnostics.assigned_tf_calls == 9


def test_geometry_family_partition_is_reflection_invariant():
    calls = (("left", 90, 115), ("middle", 94, 119), ("right", 96, 121))
    reflected = tuple((name, 300 - end, 300 - start) for name, start, end in calls)

    def memberships(values):
        catalog = build_tf_site_catalog(
            [_molecule(name, tfs=((start, end, True),)) for name, start, end in values],
            config=SiteDiscoveryConfig(edge_compatibility_bp=5),
        )
        return {
            frozenset(
                assignment.molecule_id
                for assignment in catalog.assignments
                if assignment.site_id == site.site_id
            )
            for site in catalog.sites
        }

    expected = {frozenset({"left"}), frozenset({"middle", "right"})}
    assert memberships(calls) == expected
    assert memberships(reflected) == expected


def test_geometry_family_partition_is_not_biased_toward_short_calls():
    molecules = [
        _molecule("short", tfs=((100, 120, True),)),
        _molecule("middle", tfs=((96, 124, True),)),
        _molecule("long", tfs=((94, 126, True),)),
    ]
    catalog = build_tf_site_catalog(
        molecules,
        config=SiteDiscoveryConfig(edge_compatibility_bp=5),
    )
    memberships = {
        frozenset(
            assignment.molecule_id
            for assignment in catalog.assignments
            if assignment.site_id == site.site_id
        )
        for site in catalog.sites
    }

    assert memberships == {
        frozenset({"short"}),
        frozenset({"middle", "long"}),
    }


def test_one_base_translation_moves_every_model_coordinate_once():
    original = [
        BaselineMolecule(
            molecule_id=f"read-{index}",
            contig="chr1",
            mapped_blocks=((0, 100),),
            tfs=(TFObservation(f"call-{index}", 9, 20),),
            msps=((5, 30),),
        )
        for index in range(4)
    ]
    shifted = [
        BaselineMolecule(
            molecule_id=molecule.molecule_id,
            contig=molecule.contig,
            mapped_blocks=tuple((start + 1, end + 1) for start, end in molecule.mapped_blocks),
            tfs=tuple(
                TFObservation(call.call_id, call.start + 1, call.end + 1) for call in molecule.tfs
            ),
            msps=tuple((start + 1, end + 1) for start, end in molecule.msps),
        )
        for molecule in original
    ]

    before = build_tf_site_catalog(original)
    after = build_tf_site_catalog(shifted)

    assert len(before.sites) == len(after.sites) == 1
    before_site, after_site = before.sites[0], after.sites[0]
    assert (after_site.start, after_site.end, after_site.summit, after_site.locus_summit) == (
        before_site.start + 1,
        before_site.end + 1,
        before_site.summit + 1,
        before_site.locus_summit + 1,
    )
    assert (
        after_site.n_fully_mapped,
        after_site.n_tf,
        after_site.n_msp,
        after_site.n_tf_msp,
    ) == (
        before_site.n_fully_mapped,
        before_site.n_tf,
        before_site.n_msp,
        before_site.n_tf_msp,
    )
    assert [(value.start, value.end) for value in after.assignments] == [
        (value.start + 1, value.end + 1) for value in before.assignments
    ]


def test_population_table_uses_unique_fully_mapped_molecules():
    molecules = [
        # Duplicate calls and a duplicate fetched record must not inflate any
        # molecule-level count.
        _molecule(
            "r1",
            tfs=((100, 120, True), (100, 120, True)),
            msps=((90, 130),),
        ),
        _molecule("r1", tfs=((100, 120, True),), msps=((90, 130),)),
        _molecule("r2", tfs=((100, 120, True),)),
        _molecule("r3", msps=((90, 130),)),
        _molecule("r4"),
        _molecule("r5", blocks=((109, 111),), msps=((90, 130),)),
        _molecule(
            "r6",
            blocks=((90, 109), (111, 130)),
            msps=((90, 130),),
        ),
    ]

    catalog = build_tf_site_catalog(molecules)
    site = catalog.sites[0]

    assert (site.start, site.end) == (100, 120)
    assert site.assigned_call_count == 3
    assert site.geometry_call_count == 3
    assert site.cluster_support_molecules == 2
    assert site.n_fully_mapped == 4
    assert site.n_tf == 2
    assert site.n_msp == 2
    assert site.n_tf_msp == 1
    assert site.n_tf_no_msp == 1
    assert site.n_no_tf_msp == 1
    assert site.n_no_tf_no_msp == 1
    assert site.occupancy_overall == pytest.approx(0.5)
    assert site.occupancy_given_msp == pytest.approx(0.5)
    assert catalog.diagnostics.raw_tf_calls == 4
    assert catalog.diagnostics.unique_tf_calls == 3


def test_population_mapping_index_preserves_gap_fraction_and_endpoint_rules():
    molecules = [
        _molecule("teacher", tfs=((100, 120, True),), blocks=((90, 130),)),
        # Normalization merges adjacent blocks, retaining complete coverage.
        _molecule("adjacent", blocks=((90, 110), (110, 130))),
        # One missing base is exactly the default 95% mapped-fraction threshold.
        _molecule("at-threshold", blocks=((90, 110), (111, 130))),
        _molecule("below-threshold", blocks=((90, 109), (111, 130))),
        # Mapped fraction alone is insufficient: both site endpoints are required.
        _molecule("missing-left-endpoint", blocks=((101, 130),)),
        _molecule("missing-right-endpoint", blocks=((90, 119),)),
    ]

    default_site = build_tf_site_catalog(molecules).sites[0]
    relaxed_site = build_tf_site_catalog(
        molecules,
        config=SiteDiscoveryConfig(minimum_mapped_fraction=0.90),
    ).sites[0]

    assert default_site.n_fully_mapped == 3
    assert relaxed_site.n_fully_mapped == 4


def test_msp_conditioning_requires_complete_site_containment():
    molecules = [
        _molecule("source", tfs=((100, 120, True),), msps=((90, 130),)),
        _molecule("exact", msps=((100, 120),)),
        _molecule("left-short", msps=((99, 119),)),
        _molecule("right-short", msps=((101, 121),)),
        _molecule("overlap", msps=((110, 130),)),
    ]

    site = build_tf_site_catalog(molecules).sites[0]

    assert site.n_fully_mapped == 5
    assert site.n_msp == 2
    assert site.n_tf_msp == 1
    assert site.occupancy_given_msp == pytest.approx(0.5)

    no_msp_site = build_tf_site_catalog([_molecule("only-tf", tfs=((200, 220, True),))]).sites[0]
    assert no_msp_site.n_msp == 0
    assert no_msp_site.occupancy_given_msp is None


def test_every_call_is_accounted_for_without_density_or_support_rejection():
    molecules = []
    for center_index, start in enumerate((100, 116, 132)):
        molecules.extend(
            _molecule(
                f"dense-{center_index}-{replicate}",
                tfs=((start, start + 10, True),),
            )
            for replicate in range(3)
        )
    molecules.append(
        _molecule(
            "sparse",
            tfs=((1_000_000_000, 1_000_000_010, True),),
            blocks=((999_999_900, 1_000_000_100),),
        )
    )

    catalog = build_tf_site_catalog(
        molecules,
        config=SiteDiscoveryConfig(edge_compatibility_bp=4),
    )

    assert len(catalog.sites) == 4
    assert sorted(site.cluster_support_molecules for site in catalog.sites) == [1, 3, 3, 3]
    assert sum(site.assigned_call_count for site in catalog.sites) == 10
    assert len(catalog.assignments) == 10
    assert all(assignment.site_id is not None for assignment in catalog.assignments)
    assert catalog.diagnostics.unassigned_tf_calls == 0
    assert not next(site for site in catalog.sites if site.start >= 1_000_000_000).population_ready


def test_strata_get_equal_votes_in_canonical_geometry():
    molecules = [
        _molecule(
            f"fwd-{index}",
            tfs=((100, 120, True),),
            stratum="FWD",
        )
        for index in range(20)
    ] + [
        _molecule(
            f"rev-{index}",
            tfs=((104, 126, True),),
            stratum="REV",
        )
        for index in range(5)
    ]

    site = build_tf_site_catalog(molecules).sites[0]

    assert (site.start, site.end) == (102, 123)
    assert site.start_mad == pytest.approx(2.0)
    assert site.end_mad == pytest.approx(3.0)
    assert {summary.stratum: summary.cluster_support_molecules for summary in site.strata} == {
        "FWD": 20,
        "REV": 5,
    }


def test_geometry_ineligible_tf_is_valid_population_observation():
    molecules = [_molecule(f"teacher-{index}", tfs=((100, 120, True),)) for index in range(3)] + [
        _molecule("topology-only", tfs=((100, 120, False),), msps=((90, 130),))
    ]

    catalog = build_tf_site_catalog(molecules)
    site = catalog.sites[0]
    assignment = next(
        value for value in catalog.assignments if value.molecule_id == "topology-only"
    )

    assert site.geometry_call_count == 3
    assert site.assigned_call_count == 4
    assert site.n_tf == 4
    assert not assignment.used_for_geometry
    assert assignment.site_id == site.site_id


def test_geometry_and_population_readiness_are_separate():
    molecules = [
        _molecule("teacher", tfs=((100, 120, True),)),
        _molecule("resolved-1", tfs=((100, 120, False),)),
        _molecule("resolved-2", tfs=((100, 120, False),)),
    ]

    site = build_tf_site_catalog(
        molecules,
        config=SiteDiscoveryConfig(minimum_population_support=3),
    ).sites[0]

    assert site.cluster_support_molecules == 1
    assert site.n_tf == 3
    assert site.geometry_support == 1
    assert site.population_support == 3
    assert not site.geometry_ready
    assert site.population_ready
    assert not site.analysis_ready


def test_unprojectable_geometry_call_remains_auditable():
    catalog = build_tf_site_catalog(
        [
            _molecule(
                "partial",
                tfs=((100, 120, True),),
                blocks=((100, 109), (111, 120)),
            )
        ]
    )

    assert catalog.sites == ()
    assert len(catalog.assignments) == 1
    assert catalog.assignments[0].site_id is None
    assert not catalog.assignments[0].used_for_geometry
    assert catalog.diagnostics.raw_tf_calls == 1
    assert catalog.diagnostics.unique_tf_calls == 1
    assert catalog.diagnostics.geometry_tf_calls == 0
    assert catalog.diagnostics.unassigned_tf_calls == 1


@pytest.mark.parametrize(("start", "end", "summit"), [(9, 20, 15), (10, 21, 16)])
def test_zero_radius_uses_the_same_half_base_center_rounding_for_assignment(
    start,
    end,
    summit,
):
    molecules = [
        _molecule(f"teacher-{index}", tfs=((start, end, True),)) for index in range(3)
    ] + [_molecule("resolved", tfs=((start, end, False),))]

    catalog = build_tf_site_catalog(
        molecules,
        config=SiteDiscoveryConfig(assignment_radius_bp=0),
    )
    resolved = next(
        assignment for assignment in catalog.assignments if assignment.molecule_id == "resolved"
    )

    assert catalog.sites[0].summit == summit
    assert resolved.site_id == catalog.sites[0].site_id


def test_summit_is_inside_a_one_base_half_open_site():
    site = build_tf_site_catalog([_molecule("single-base", tfs=((1, 2, True),))]).sites[0]

    assert (site.start, site.summit, site.end) == (1, 1, 2)


def test_coverage_interval_index_handles_an_anomalously_wide_site():
    molecules = [
        *[
            _molecule(
                f"site-a-{index}",
                tfs=((100, 120, True),),
                blocks=((90, 130),),
            )
            for index in range(3)
        ],
        *[
            _molecule(
                f"site-b-{index}",
                tfs=((1000, 1030, True),),
                blocks=((990, 1040),),
            )
            for index in range(3)
        ],
        _molecule("wide-source", tfs=((0, 5000, True),), blocks=((0, 5000),)),
        _molecule("near-a", blocks=((90, 130),)),
        _molecule("near-b", blocks=((990, 1040),)),
        _molecule("both", blocks=((90, 130), (990, 1040))),
        _molecule("wide-cover", blocks=((0, 5000),)),
    ]

    catalog = build_tf_site_catalog(molecules)
    mapped_by_geometry = {(site.start, site.end): site.n_fully_mapped for site in catalog.sites}

    assert mapped_by_geometry == {
        (0, 5000): 2,
        (100, 120): 7,
        (1000, 1030): 7,
    }


@pytest.mark.parametrize("sigma", [float("nan"), float("inf"), 0.0, -1.0])
def test_smoothing_sigma_must_be_finite_and_positive(sigma):
    with pytest.raises(ValueError, match="finite and positive"):
        SiteDiscoveryConfig(smoothing_sigma_bp=sigma)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("peak_distance_bp", float("nan")),
        ("assignment_radius_bp", 2.5),
        ("edge_compatibility_bp", "5"),
        ("minimum_geometry_support_per_stratum", True),
        ("minimum_geometry_support", float("inf")),
        ("minimum_population_support", None),
    ],
)
def test_discrete_config_fields_require_integers(field, value):
    with pytest.raises(ValueError, match="must be an integer"):
        SiteDiscoveryConfig(**{field: value})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("molecule_id", None),
        ("molecule_id", "read\talias"),
        ("contig", None),
        ("contig", "chr1\nchr2"),
    ],
)
def test_molecule_identifiers_must_be_losslessly_serializable(field, value):
    molecule = _molecule("valid", tfs=((100, 120, True),))
    values = {
        "molecule_id": molecule.molecule_id,
        "contig": molecule.contig,
        "mapped_blocks": molecule.mapped_blocks,
        "tfs": molecule.tfs,
        "msps": molecule.msps,
        "stratum": molecule.stratum,
    }
    values[field] = value

    with pytest.raises(ValueError, match=field):
        build_tf_site_catalog([BaselineMolecule(**values)])


def test_call_identifier_must_not_be_none_or_contain_control_characters():
    for call_id in (None, "call\nother"):
        molecule = BaselineMolecule(
            molecule_id="read",
            contig="chr1",
            mapped_blocks=((0, 1000),),
            tfs=(TFObservation(call_id=call_id, start=100, end=120),),
        )
        with pytest.raises(ValueError, match="call_id"):
            build_tf_site_catalog([molecule])


def test_coordinates_must_be_exact_integers():
    with pytest.raises(ValueError, match="TF call coordinates must be integers"):
        build_tf_site_catalog(
            [
                BaselineMolecule(
                    molecule_id="read",
                    contig="chr1",
                    mapped_blocks=((0, 1000),),
                    tfs=(TFObservation("call", 100.9, 120.9),),
                )
            ]
        )
    with pytest.raises(ValueError, match="mapped block coordinates must be integers"):
        build_tf_site_catalog(
            [
                BaselineMolecule(
                    molecule_id="read",
                    contig="chr1",
                    mapped_blocks=((0.0, 1000.0),),
                    tfs=(),
                )
            ]
        )


def test_one_molecule_id_cannot_occupy_multiple_strata_on_a_contig():
    molecules = [
        _molecule("same-read", stratum="FWD", tfs=((100, 120, True),)),
        _molecule("same-read", stratum="REV", tfs=((100, 120, True),)),
    ]

    with pytest.raises(ValueError, match="appears in multiple strata"):
        build_tf_site_catalog(molecules)


def test_equal_center_modes_and_multiple_contigs_are_deterministic():
    molecules = []
    for contig in ("chr1", "chr2"):
        for center_start in (95, 107):
            molecules.extend(
                _molecule(
                    f"{contig}-{center_start}-{replicate}",
                    contig=contig,
                    tfs=((center_start, center_start + 10, True),),
                )
                for replicate in range(3)
            )

    expected = build_tf_site_catalog(molecules)
    shuffled = list(molecules)
    random.Random(17).shuffle(shuffled)

    assert [(site.contig, site.summit) for site in expected.sites] == [
        ("chr1", 100),
        ("chr1", 112),
        ("chr2", 100),
        ("chr2", 112),
    ]
    assert len(expected.loci) == 4
    assert all(site.family_index == 1 for site in expected.sites)
    assert len({site.locus_id for site in expected.sites}) == 4
    assert build_tf_site_catalog(shuffled) == expected


def test_input_order_does_not_change_sites_ids_or_assignments():
    molecules = [
        _molecule(
            f"molecule-{index}",
            tfs=((100 + index % 3, 120 + index % 3, True),),
            msps=((90, 140),) if index % 2 else (),
            stratum="FWD" if index % 2 else "REV",
        )
        for index in range(12)
    ]
    expected = build_tf_site_catalog(molecules)
    shuffled = list(molecules)
    random.Random(20260722).shuffle(shuffled)
    observed = build_tf_site_catalog(shuffled)

    assert observed == expected
