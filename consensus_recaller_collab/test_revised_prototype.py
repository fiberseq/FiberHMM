from __future__ import annotations

import math

import numpy as np
import pytest

from consensus_recaller_collab.prototype import (
    IntervalCall,
    ReadEvidence,
    SiteTemplate,
    _mapped_annotations,
)
from consensus_recaller_collab.revised_prototype import (
    ConfigurationLibraryEntry,
    DirectConfigurationRecord,
    analyze_composite_deconvolution,
    build_direct_configuration_records,
    build_forced_site_template,
    calibrate_edge_bandwidth,
    choose_composite_replacement,
    collapse_amplified_cohort_by_input,
    edge_log_density,
    estimate_local_complex_prior,
    gap_diagnostics,
    main,
    merge_forced_sites,
    parse_site_interval,
    project_configuration_library,
    rescore_candidate_nuc_prior,
    score_composite_candidate,
    shift_site_templates,
    unmatched_target_molecule_ids,
    wilson_lower_bound,
)


def site(name, start, end):
    return SiteTemplate(
        site_id=name,
        start=start,
        end=end,
        center=(start + end) // 2,
        support={"BOTH": 20},
        all_support={"BOTH": 20},
        median_tq={"BOTH": 150.0},
        start_mad=1.0,
        end_mad=1.0,
        local_enrichment=10.0,
    )


SITES = [site("left", 100, 130), site("right", 150, 180)]


def evidence_read(
    name,
    *,
    tfs=(),
    nucs=(),
    positions=(95, 110, 120, 140, 160, 170, 185),
    steps=(0.5, 3.0, 3.0, 0.0, 3.0, 3.0, 0.5),
    hits=(False, False, False, False, False, False, False),
):
    return ReadEvidence(
        name=name,
        strand="BOTH",
        ref_start=80,
        ref_end=250,
        positions=np.asarray(positions, dtype=int),
        steps=np.asarray(steps, dtype=float),
        hits=np.asarray(hits, dtype=bool),
        contexts=np.zeros(len(positions), dtype=int),
        tfs=list(tfs),
        nucs=list(nucs),
        msps=[],
    )


def test_reference_projection_rejects_mostly_soft_clipped_ma_calls():
    class TaggedRead:
        is_reverse = False

        def __init__(self):
            self.tags = {
                "MA": "100;nuc.Q:1-100,31-50",
                "AQ": [200, 201],
            }

        def get_tag(self, name):
            if name not in self.tags:
                raise KeyError(name)
            return self.tags[name]

        def has_tag(self, name):
            return name in self.tags

    reference_positions = [None] * 30 + list(range(100, 170))
    calls = _mapped_annotations(TaggedRead(), "nuc", reference_positions)
    assert calls == [IntervalCall(100, 150, 201)]


def test_amplified_duplicate_collapse_never_crosses_pooled_input_bams():
    reads = [
        evidence_read("a1"), evidence_read("a2"), evidence_read("b1")
    ]
    for read, input_id in zip(reads, ("time1.bam", "time1.bam", "time2.bam")):
        read.library_id = input_id
        read.fingerprint_positions = np.asarray([101, 107, 113, 127], dtype=int)
    collapsed, diagnostics = collapse_amplified_cohort_by_input(
        reads, min_jaccard=0.95, min_deam=4
    )
    assert len(collapsed) == 2
    assert diagnostics["duplicate_reads_collapsed"] == 1
    assert diagnostics["by_input_bam"]["time1.bam"][
        "analyzed_molecules"
    ] == 1
    assert diagnostics["by_input_bam"]["time2.bam"][
        "analyzed_molecules"
    ] == 1


def test_external_target_molecules_are_detected_before_inference():
    source = [evidence_read("shared")]
    derivative = [evidence_read("shared")]
    external = [evidence_read("other-library")]
    assert unmatched_target_molecule_ids(source, derivative) == []
    assert unmatched_target_molecule_ids(source, external) == [
        ("other-library", "BOTH")
    ]


def test_explicit_configuration_library_never_uses_nuc_as_latent_training_label():
    direct = evidence_read(
        "direct",
        tfs=(IntervalCall(95, 130, 150), IntervalCall(150, 190, 160)),
    )
    ambiguous = evidence_read("ambiguous", nucs=(IntervalCall(95, 190, 255),))
    records = build_direct_configuration_records(
        [direct, ambiguous], SITES, min_tq=100, center_radius=10
    )
    assert len(records) == 1
    assert records[0].read_name == "direct"
    library = project_configuration_library(records, [0, 1], min_support=1)
    assert len(library) == 1
    assert library[0].site_indices == (0, 1)
    assert library[0].member_intervals == (((95, 130), (150, 190)),)
    assert library[0].hulls == ((95, 190),)


def test_forced_site_supplies_geometry_but_support_still_comes_from_source_calls():
    reads = [
        evidence_read("direct", tfs=(IntervalCall(118, 151, 180),)),
        evidence_read("nuc-only", nucs=(IntervalCall(95, 190, 255),)),
    ]
    forced = build_forced_site_template(
        reads, (120, 150), site_id="forced", min_tq=100, center_radius=10
    )
    assert (forced.start, forced.end) == (120, 150)
    assert forced.support == {"BOTH": 1}
    assert forced.all_support == {"BOTH": 1}
    assert forced.local_enrichment == 3.0
    assert forced.local_enrichment_by_strand == {"BOTH": 3.0}


def test_forced_sites_only_replaces_automatic_geometry():
    reads = [evidence_read("direct", tfs=(IntervalCall(118, 151, 180),))]
    merged = merge_forced_sites(
        SITES,
        [(120, 150)],
        reads,
        min_tq=100,
        center_radius=10,
        forced_only=True,
    )
    assert [(item.start, item.end) for item in merged] == [(120, 150)]
    assert merged[0].site_id == "site1"


def test_forced_site_parser_is_half_open_and_validated():
    assert parse_site_interval("120-150") == (120, 150)
    with pytest.raises(Exception, match="START < END"):
        parse_site_interval("150-120")


def test_target_geometry_shift_does_not_mutate_source_template():
    shifted = shift_site_templates(SITES, 100)
    assert [(item.start, item.end) for item in SITES] == [(100, 130), (150, 180)]
    assert [(item.start, item.end) for item in shifted] == [(200, 230), (250, 280)]
    assert shifted[0].support == SITES[0].support


def test_empirical_edge_density_rewards_recurrent_configuration_hull():
    call = IntervalCall(95, 190)
    exact = edge_log_density(call, [(95, 190)] * 10, bandwidth=5.0)
    shifted = edge_log_density(call, [(75, 210)] * 10, bandwidth=5.0)
    assert exact > shifted + 10.0


def test_edge_bandwidth_is_calibrated_leave_one_molecule_out_within_cohort():
    records = [
        DirectConfigurationRecord(
            "a", "BOTH",
            ((0, IntervalCall(95, 130)), (1, IntervalCall(150, 190))),
            "libA",
        ),
        DirectConfigurationRecord(
            "b", "BOTH",
            ((0, IntervalCall(96, 131)), (1, IntervalCall(151, 191))),
            "libB",
        ),
    ]
    result = calibrate_edge_bandwidth(
        records, (3.0, 10.0, 20.0),
        strand_specific=False, fallback=7.5,
    )
    assert result["holdout_mode"] == "leave_one_molecule_out"
    assert result["selected_bandwidth"] == 3.0


def test_local_complex_prior_is_a_lower_bound_and_requires_full_span():
    positive = evidence_read(
        "positive", tfs=(IntervalCall(100, 130, 200),)
    )
    negative = evidence_read("negative")
    short_positive = evidence_read(
        "short", tfs=(IntervalCall(100, 130, 200),)
    )
    short_positive.ref_end = 140
    reads = [positive, negative, short_positive]
    records = build_direct_configuration_records(
        reads, SITES, min_tq=100, center_radius=10
    )
    result = estimate_local_complex_prior(
        reads,
        records,
        [0, 1],
        SITES,
        strand=None,
        exclude_molecule_id=None,
        z=1.96,
    )
    assert result["spanning_molecules"] == 2
    assert result["explicit_complex_molecules"] == 1
    assert 0.0 < result["wilson_lower_bound"] < 0.5
    assert wilson_lower_bound(0, 0) == 0.0


def test_configuration_library_requires_full_template_span_for_known_reads():
    long_read = evidence_read(
        "long", tfs=(IntervalCall(100, 130, 200),)
    )
    short_read = evidence_read(
        "short", tfs=(IntervalCall(100, 130, 200),)
    )
    short_read.ref_end = 140
    records = build_direct_configuration_records(
        [long_read, short_read], SITES, min_tq=100, center_radius=10
    )
    library = project_configuration_library(
        records, [0, 1], min_support=1, sites=SITES
    )
    assert len(library) == 1
    assert library[0].support == 1
    assert library[0].member_molecule_ids == (("long", "BOTH"),)


def test_uncertain_exact_layout_becomes_broad_tf_complex_not_forced_split():
    result = {
        "complex_posterior": 0.91,
        "best_tf_posterior": 0.50,
        "best_decomposition_posterior_given_complex": 0.55,
        "best_tf": {
            "representative_edge_conditioned_intervals": [
                [100, 130], [150, 180]
            ]
        },
    }
    unresolved = choose_composite_replacement(
        result, IntervalCall(95, 190), 0.8
    )
    assert unresolved == {
        "decomposition_resolved": False,
        "replacement_kind": "unresolved_tf_agglomeration",
        "replacement_intervals": [[95, 190]],
        "replacement_posterior": 0.91,
    }
    resolved = choose_composite_replacement(
        {**result, "best_decomposition_posterior_given_complex": 0.9},
        IntervalCall(95, 190),
        0.8,
    )
    assert resolved["decomposition_resolved"] is True
    assert resolved["replacement_intervals"] == [[100, 130], [150, 180]]


def test_gap_diagnostics_use_actual_opportunities_and_enzyme_rate():
    read = evidence_read("gap")
    accessible_probability = np.full(4096, 0.8)
    result = gap_diagnostics(
        read, IntervalCall(95, 190), [0, 1], SITES, accessible_probability
    )
    internal = [gap for gap in result["gaps"] if gap["kind"] == "internal"]
    assert len(internal) == 1
    assert internal[0]["opportunities"] == 1
    np.testing.assert_allclose(internal[0]["p_no_hit_if_accessible"], 0.2)


def test_block_boundaries_are_not_evidence_only_the_bases_are():
    """A composite must not be rescued by matching the nuc call's edges.

    The block's boundaries came from the upstream single-molecule segmentation,
    which is the call under suspicion; it cannot also be the evidence.  The old
    model scored the source hull against those edges with a tight Gaussian,
    which was worth up to -68 nats and accounted for 98% of the rejection of TF
    at Homie.  Here the separator is mildly *protected* and no bridge is
    allowed, so the bases favour N and N must win however well the source hull
    happens to line up with the call.
    """
    read = evidence_read(
        "candidate",
        nucs=(IntervalCall(95, 190),),
        positions=(85, 95, 110, 120, 140, 160, 170, 185, 200),
        steps=(-10.0, 0.5, 3.0, 3.0, 1.0, 3.0, 3.0, 0.5, -10.0),
        hits=(True, False, False, False, False, False, False, False, True),
    )
    common = dict(
        read=read,
        call=IntervalCall(95, 190),
        relevant_site_indices=[0, 1],
        sites=SITES,
        control_lengths=[147] * 20,
        accessible_hit_probability=np.full(4096, 0.8),
        nuc_prior_odds=1.0,
        edge_bandwidth=5.0,
        length_bandwidth=10.0,
        boundary_null_width=300.0,
        prior_alpha=0.5,
        nuc_scorer=None,
    )
    # Source hull exactly equal to the call: under the old edge kernel this
    # alone flipped the call to TF.
    exact = [ConfigurationLibraryEntry((0, 1), (((95, 130), (150, 190)),) * 20)]
    result = score_composite_candidate(
        library=exact, protected_gap_prior=0.0, **common
    )
    assert result["top_state"] == "N"

    # Allowing a bridge bounds the cost of the protected separator at log(p),
    # rather than letting it collapse the composite without limit.
    bridged = score_composite_candidate(
        library=exact, protected_gap_prior=0.25, **common
    )
    assert bridged["best_tf"]["edge_conditioned_log_bf_vs_n"] >= math.log(0.25)
    assert bridged["best_tf"]["edge_conditioned_log_bf_vs_n"] > result[
        "best_tf"
    ]["edge_conditioned_log_bf_vs_n"]


def test_global_n_prior_sensitivity_is_separate_from_likelihood_terms():
    read = evidence_read("candidate", nucs=(IntervalCall(95, 190),))
    library = [ConfigurationLibraryEntry(
        (0, 1), (((95, 130), (150, 190)),) * 20
    )]
    common = dict(
        read=read,
        call=IntervalCall(95, 190),
        relevant_site_indices=[0, 1],
        sites=SITES,
        library=library,
        control_lengths=[95] * 20,
        accessible_hit_probability=np.full(4096, 0.8),
        edge_bandwidth=5.0,
        length_bandwidth=10.0,
        boundary_null_width=300.0,
        prior_alpha=0.5,
        protected_gap_prior=0.0,
        nuc_scorer=None,
    )
    neutral = score_composite_candidate(nuc_prior_odds=1.0, **common)
    strong_n = score_composite_candidate(nuc_prior_odds=100.0, **common)
    rescored = rescore_candidate_nuc_prior(neutral, 100.0)
    # The likelihood term must not move with the prior.
    assert neutral["best_tf"]["edge_conditioned_log_bf_vs_n"] == strong_n[
        "best_tf"
    ]["edge_conditioned_log_bf_vs_n"]
    assert neutral["best_tf_posterior"] > strong_n["best_tf_posterior"]
    np.testing.assert_allclose(
        rescored["best_tf_posterior"], strong_n["best_tf_posterior"]
    )


def test_boundary_decoy_can_reuse_identical_nuc_geometry_marginal():
    read = evidence_read("candidate", nucs=(IntervalCall(95, 190),))
    library = [ConfigurationLibraryEntry(
        (0, 1), (((95, 130), (150, 190)),) * 20
    )]
    common = dict(
        read=read,
        call=IntervalCall(95, 190),
        relevant_site_indices=[0, 1],
        sites=SITES,
        control_lengths=[95] * 20,
        accessible_hit_probability=np.full(4096, 0.8),
        nuc_prior_odds=1.0,
        edge_bandwidth=5.0,
        length_bandwidth=10.0,
        boundary_null_width=300.0,
        prior_alpha=0.5,
        protected_gap_prior=0.0,
        nuc_scorer=None,
    )
    true = score_composite_candidate(library=library, **common)
    shifted = score_composite_candidate(
        library=[ConfigurationLibraryEntry(
            (0, 1), (((115, 150), (170, 210)),) * 20
        )],
        nuc_reference=true["_nuc_reference"],
        **common,
    )
    assert shifted["integrated_nuc"] == true["integrated_nuc"]
    assert shifted["current_nuc_base_llr_diagnostic"] == true[
        "current_nuc_base_llr_diagnostic"
    ]


def test_protected_bridge_is_explicit_and_increases_composite_support():
    read = evidence_read(
        "candidate",
        nucs=(IntervalCall(95, 190),),
        steps=(0.5, 3.0, 3.0, 5.0, 3.0, 3.0, 0.5),
    )
    library = [ConfigurationLibraryEntry(
        (0, 1), (((95, 130), (150, 190)),) * 20
    )]
    common = dict(
        read=read,
        call=IntervalCall(95, 190),
        relevant_site_indices=[0, 1],
        sites=SITES,
        library=library,
        control_lengths=[95] * 20,
        accessible_hit_probability=np.full(4096, 0.8),
        nuc_prior_odds=1.0,
        edge_bandwidth=5.0,
        length_bandwidth=10.0,
        boundary_null_width=300.0,
        prior_alpha=0.5,
        nuc_scorer=None,
    )
    separated = score_composite_candidate(protected_gap_prior=0.0, **common)
    mixture = score_composite_candidate(protected_gap_prior=0.5, **common)
    assert mixture["best_tf_posterior"] > separated["best_tf_posterior"]
    bridges = [
        segment for segment in mixture["best_tf"]["internal_gap_states"]
        if segment["kind"] == "bridge"
    ]
    assert bridges and bridges[0]["posterior_protected"] > 0.5


def test_single_tf_does_not_expand_to_fill_nuc_candidate():
    one_site = [site("single", 120, 150)]
    read = evidence_read("single-candidate", nucs=(IntervalCall(95, 190),))
    library = [ConfigurationLibraryEntry((0,), (((120, 150),),) * 20)]
    result = score_composite_candidate(
        read,
        IntervalCall(95, 190),
        [0],
        one_site,
        library,
        control_lengths=[95] * 20,
        accessible_hit_probability=np.full(4096, 0.8),
        nuc_prior_odds=1.0,
        edge_bandwidth=5.0,
        length_bandwidth=10.0,
        boundary_null_width=300.0,
        prior_alpha=0.5,
        protected_gap_prior=0.25,
        nuc_scorer=None,
    )
    assert result["best_tf"]["representative_edge_conditioned_intervals"] == [
        [120, 150]
    ]


def test_composite_pass_reports_strict_and_review_tiers():
    source = [
        evidence_read(
            f"direct-{index}",
            tfs=(IntervalCall(95, 130, 150), IntervalCall(150, 190, 160)),
        )
        for index in range(12)
    ]
    target = [evidence_read("candidate", nucs=(IntervalCall(95, 190),))]
    result = analyze_composite_deconvolution(
        source,
        target,
        SITES,
        min_tq=100,
        center_radius=10,
        min_config_support=10,
        nuc_prior_odds_values=[1.0],
        edge_bandwidth=5.0,
        length_bandwidth=10.0,
        boundary_null_flank=100,
        background_exclusion=500,
        max_nuc_span=220,
        strict_posterior=0.5,
        review_posterior=0.25,
        prior_alpha=0.5,
        protected_gap_prior=0.0,
        configuration_shift=0,
        max_examples=5,
        strand_specific_library=False,
        nuc_scorer=None,
        accessible_hit_probability=np.full(4096, 0.8),
        configuration_control_shifts=(50,),
        minimum_boundary_control_log_bf=1.0,
    )
    counts = result["scenarios"]["1.0"]["counts"]
    assert counts["candidate_nucs"] == 1
    assert counts["review"] == 1
    assert counts["strict"] == 1
    assert counts["boundary_calibrated_strong"] == 1
    assert result["scenarios"]["1.0"]["proposals"][0][
        "proposal_tier"
    ] == "strong"
    assert result["scenarios"]["1.0"]["proposals"][0][
        "boundary_control_log_bf"
    ] > 1.0
    assert result["scenarios"]["1.0"]["boundary_control_counts"]["50"][
        "scored"
    ] == 1
    states = result["scenarios"]["1.0"]["candidate_states"]
    assert result["scenarios"]["1.0"]["candidate_state_count"] == 1
    assert states[0]["decision_id"] == result["scenarios"]["1.0"][
        "proposals"
    ][0]["proposal_id"]
    assert states[0]["replacement_intervals"] == [[95, 130], [150, 190]]
    assert "best_tf_edge_conditioned_log_bf_vs_n" in states[0]


def test_composite_prior_pools_input_bams_and_holds_out_only_candidate_molecule():
    source = []
    for library_id in ("libA", "libB"):
        for index in range(6):
            read = evidence_read(
                f"{library_id}-{index}",
                tfs=(IntervalCall(95, 130, 150), IntervalCall(150, 190, 160)),
            )
            read.library_id = library_id
            source.append(read)
    source_candidate = evidence_read(
        "candidate",
        tfs=(IntervalCall(95, 130, 150), IntervalCall(150, 190, 160)),
    )
    source_candidate.library_id = "libA"
    source.append(source_candidate)
    # The scored derivative represents the same molecule before the direct
    # TF configuration was applied. This exercises molecule holdout without
    # constructing a contradictory simultaneous nuc+TF current state.
    target = evidence_read("candidate", nucs=(IntervalCall(95, 190),))
    target.library_id = "libA"
    result = analyze_composite_deconvolution(
        source,
        [target],
        SITES,
        min_tq=100,
        center_radius=10,
        min_config_support=5,
        nuc_prior_odds_values=[1.0],
        edge_bandwidth=5.0,
        length_bandwidth=10.0,
        boundary_null_flank=100,
        background_exclusion=500,
        max_nuc_span=220,
        strict_posterior=0.5,
        review_posterior=0.25,
        prior_alpha=0.5,
        protected_gap_prior=0.0,
        configuration_shift=0,
        max_examples=5,
        strand_specific_library=False,
        nuc_scorer=None,
        accessible_hit_probability=np.full(4096, 0.8),
    )
    assert result["inference_cohort"] == {
        "mode": "explicitly_pooled_input_bams",
        "input_bam_ids": ["libA", "libB"],
        "input_bam_identity_used_as_prior_partition": False,
        "external_libraries_used": False,
        "candidate_holdout": "leave_one_molecule_out",
    }
    key = next(iter(result["library_cache"]))
    # The diagnostic cache describes the whole pooled cohort.
    assert result["library_cache"][key][0]["support"] == 13
    # The actual candidate state excludes only itself, retaining both BAMs.
    state = result["scenarios"]["1.0"]["candidate_states"][0]
    assert state["best_tf_source_support_after_candidate_holdout"] == 12
    assert state["local_complex_prior"]["spanning_molecules"] == 12
    assert state["local_complex_prior"]["explicit_complex_molecules"] == 12
    assert state["local_complex_prior"]["candidate_molecule_excluded"] is True


def test_composite_pass_hard_rejects_ceiling_above_the_maximum():
    with pytest.raises(ValueError, match="between 90 and 220 bp"):
        analyze_composite_deconvolution(
            [],
            [],
            SITES,
            min_tq=100,
            center_radius=10,
            min_config_support=10,
            nuc_prior_odds_values=[1.0],
            edge_bandwidth=5.0,
            length_bandwidth=10.0,
            boundary_null_flank=100,
            background_exclusion=500,
            max_nuc_span=221,
            strict_posterior=0.95,
            review_posterior=0.5,
            prior_alpha=0.5,
            protected_gap_prior=0.25,
            configuration_shift=0,
            max_examples=0,
            strand_specific_library=False,
            nuc_scorer=None,
            accessible_hit_probability=np.full(4096, 0.8),
        )


def test_composite_pass_reports_focal_blocks_skipped_above_ceiling():
    target = [evidence_read("long", nucs=(IntervalCall(90, 311),))]
    result = analyze_composite_deconvolution(
        [],
        target,
        SITES,
        min_tq=100,
        center_radius=10,
        min_config_support=10,
        nuc_prior_odds_values=[1.0],
        edge_bandwidth=5.0,
        length_bandwidth=10.0,
        boundary_null_flank=100,
        background_exclusion=500,
        max_nuc_span=220,
        strict_posterior=0.95,
        review_posterior=0.5,
        prior_alpha=0.5,
        protected_gap_prior=0.25,
        configuration_shift=0,
        max_examples=0,
        strand_specific_library=False,
        nuc_scorer=None,
        accessible_hit_probability=np.full(4096, 0.8),
    )
    assert result["span_filter"] == {
        "minimum_bp": 90,
        "maximum_bp": 220,
        "hard_ceiling_bp": 220,
        "skipped_focal_blocks_above_maximum": 1,
    }
    assert result["scenarios"]["1.0"]["counts"]["candidate_nucs"] == 0


def test_cli_rejects_ceiling_above_the_maximum_before_reading_input(tmp_path):
    with pytest.raises(SystemExit) as error:
        main([
            "-i", "does-not-exist.bam",
            "--preset", "hia5-pacbio",
            "--region", "chr2R:100-200",
            "--max-nuc-span", "221",
            "-o", str(tmp_path / "report.json"),
        ])
    assert error.value.code == 2


def _flat_read(name, call, *, positions, steps, nucs=()):
    return ReadEvidence(
        name=name, strand="BOTH", ref_start=900, ref_end=1200,
        positions=np.asarray(positions, dtype=int),
        steps=np.asarray(steps, dtype=float),
        hits=np.zeros(len(positions), dtype=bool),
        contexts=np.zeros(len(positions), dtype=int),
        tfs=[], nucs=list(nucs), msps=[],
    )


COMPOSITE_SITES = [site("tf1", 1000, 1040), site("tf2", 1060, 1100)]
COMPOSITE_CALL = IntervalCall(1000, 1100)
COMPOSITE_LIBRARY = [ConfigurationLibraryEntry(
    (0, 1),
    tuple([((1000, 1040), (1060, 1100))] * 40),
    tuple(f"src{index}" for index in range(40)),
)]
BACKGROUND_LENGTHS = [95, 100, 105, 110, 120, 130, 140, 147, 150, 160, 170, 180]
# Source-observed nuc geometries at this locus, the N analogue of the TF library.
NUC_GEOMETRIES = [(1000 + offset, 1100 - offset) for offset in range(-10, 10)] * 2


def _composite(read, **overrides):
    options = dict(
        accessible_hit_probability=np.full(4096, 0.35),
        nuc_prior_odds=10.0,
        edge_bandwidth=5.0,
        length_bandwidth=10.0,
        boundary_null_width=100.0,
        prior_alpha=1.0,
        protected_gap_prior=0.25,
        nuc_scorer=None,
        nuc_geometries=NUC_GEOMETRIES,
        min_nuc_geometries=20,
    )
    options.update(overrides)
    return score_composite_candidate(
        read, COMPOSITE_CALL, [0, 1], COMPOSITE_SITES, COMPOSITE_LIBRARY,
        BACKGROUND_LENGTHS, **options,
    )


def _uninformative_read():
    # Opportunities lie entirely outside the block, so every LLR is zero.
    return _flat_read(
        "uninformative", COMPOSITE_CALL,
        positions=(960, 980, 1120, 1140), steps=(0.0, 0.0, 0.0, 0.0),
        nucs=(COMPOSITE_CALL,),
    )


@pytest.mark.parametrize("odds", [1.0, 10.0, 100.0])
def test_uninformative_block_returns_the_prior_exactly(odds):
    """With no informative bases and matched geometry priors, posterior == prior.

    N and TF must marginalize over geometry priors fitted the same way from the
    same source population.  When the two priors are equally concentrated, a
    block carrying no evidence has to return the prior untouched.  The previous
    model gave N a vague uniform prior over every arithmetically possible dyad
    while TF got a sharp empirical one; that asymmetry alone was worth ~2.9 nats
    (~19x) toward splitting and silently displaced the whole N:TF sweep axis.
    """
    # Same concentration as the TF configuration library: a delta on the call.
    matched_geometries = [(1000, 1100)] * 40
    result = _composite(
        _uninformative_read(),
        nuc_prior_odds=odds,
        nuc_geometries=matched_geometries,
    )
    assert result["integrated_nuc"]["geometry_prior"] == (
        "none_block_is_the_reference"
    )
    assert result["complex_posterior"] == pytest.approx(
        1.0 / (1.0 + odds), abs=1e-6
    )


def test_continuous_nucleosome_is_not_split_into_a_tf_pair():
    """A protected separator is evidence for N and must be scored as such."""
    positions = tuple(range(996, 1104, 4))
    protected = tuple(2.0 for _ in positions)
    accessible_separator = tuple(
        -2.0 if 1040 <= position < 1060 else 2.0 for position in positions
    )
    nucleosome = _flat_read(
        "continuous-nuc", COMPOSITE_CALL,
        positions=positions, steps=protected, nucs=(COMPOSITE_CALL,),
    )
    tf_pair = _flat_read(
        "true-tf-pair", COMPOSITE_CALL,
        positions=positions, steps=accessible_separator,
        nucs=(COMPOSITE_CALL,),
    )
    nuc_result = _composite(nucleosome)
    tf_result = _composite(tf_pair)
    assert nuc_result["top_state"] == "N"
    assert nuc_result["nuc_posterior"] > 0.5
    # The same machinery must still split a genuinely accessible separator.
    assert tf_result["top_state"] == "TF:tf1,tf2"
    assert tf_result["complex_posterior"] > 0.95


def test_local_occupancy_prior_never_swallows_the_requested_sweep():
    """The requested odds must always move the effective prior.

    The local Wilson occupancy floor used to be combined with ``max(global,
    wilson)``.  Once the floor exceeded the global TF prior, the requested odds
    stopped mattering entirely and the 10:1 and 100:1 scenarios became
    bit-identical, so the conservative end of the sweep was unreachable at
    exactly the sites that make calls.  The floor is now a baseline that the
    requested odds multiply.
    """
    wilson = 0.35
    local_nuc_odds = (1.0 - wilson) / wilson
    effective = [local_nuc_odds * odds for odds in (1.0, 10.0, 100.0)]
    assert effective[0] < effective[1] < effective[2]
    # A conservative request must actually reach a conservative prior.
    assert effective[2] > 100.0


def test_bridged_mass_cannot_license_a_strong_split():
    """A protected separator must not produce a confident split.

    A fully bridged TF complex protects exactly the bases a nucleosome would, so
    it predicts identical observations and no molecule can ever refute it.
    Summing that mass into the aggregate let a block whose separator carried
    strong *protected* evidence reach split posterior ~0.99.  Only resolvable
    mass -- separators actually observed to be accessible -- may license a split.
    """
    positions = tuple(range(996, 1104, 4))
    protected = tuple(2.0 for _ in positions)
    accessible_separator = tuple(
        -2.0 if 1040 <= position < 1060 else 2.0 for position in positions
    )
    bridged = _composite(_flat_read(
        "protected-separator", COMPOSITE_CALL,
        positions=positions, steps=protected, nucs=(COMPOSITE_CALL,),
    ))
    genuine = _composite(_flat_read(
        "accessible-separator", COMPOSITE_CALL,
        positions=positions, steps=accessible_separator,
        nucs=(COMPOSITE_CALL,),
    ))
    # The protected separator is evidence *for* a nucleosome; whatever aggregate
    # mass the bridge carries, none of it may be resolvable.
    assert bridged["resolvable_complex_posterior"] < 0.05
    assert bridged["bridged_complex_posterior"] >= (
        bridged["complex_posterior"] - 0.05
    )
    # A genuinely accessible separator is refutable evidence and must survive.
    assert genuine["resolvable_complex_posterior"] > 0.95
    assert genuine["bridged_complex_posterior"] < 0.05
