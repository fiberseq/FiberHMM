from __future__ import annotations

from array import array

import numpy as np
import pysam

from consensus_recaller_collab.prototype import (
    DddaRadialNucScorer,
    IntervalCall,
    PRESETS,
    ReadEvidence,
    SiteTemplate,
    _hard_observations,
    analyze_window,
    enumerate_configurations,
    fit_site_state_model,
    fit_mixture_weights,
    local_call_likelihoods,
    match_direct_site_indices,
    marginalize_configuration_prior,
    posterior_with_prior_multiplier,
)
from fiberhmm.inference.nuc_recaller import NucProfile


def test_nanopore_hia5_uses_strict_hard_call_threshold():
    assert PRESETS["hia5-nanopore"]["prob_threshold"] == 248


def test_pacbio_treats_both_mm_strands_as_one_molecule_group():
    assert PRESETS["hia5-pacbio"]["strand_mode"] == "pacbio-duplex"
    assert PRESETS["hia5-pacbio"]["consensus_mode"] == "population"


def test_pacbio_duplex_observation_contains_a_and_t_mm_channels():
    read = pysam.AlignedSegment()
    read.query_name = "duplex"
    read.query_sequence = "CCCCCCCCCCACCCCCCTCCCCCCCCCC"
    read.flag = 0
    read.set_tag("MM", "A+a.,0;T-a.,0;")
    read.set_tag("ML", array("B", [255, 255]))
    obs, group = _hard_observations(
        read, "pacbio-duplex", "pacbio-fiber", 3, 125
    )
    assert group == "BOTH"
    assert 0 <= obs[10] < 4096
    assert 0 <= obs[17] < 4096


def _site(site_id, start, end):
    return SiteTemplate(
        site_id=site_id,
        start=start,
        end=end,
        center=(start + end) // 2,
        support={"FWD": 10, "REV": 0},
        all_support={"FWD": 10, "REV": 0},
        median_tq={"FWD": 120.0, "REV": 0.0},
        start_mad=2.0,
        end_mad=2.0,
        local_enrichment=10.0,
    )


def test_configuration_enumeration_allows_arbitrary_nonoverlapping_subsets():
    sites = [_site("a", 10, 20), _site("b", 30, 40), _site("c", 50, 60)]
    names = {config.name for config in enumerate_configurations(sites)}
    assert {"A", "N", "TF:a", "TF:b", "TF:c", "TF:a,b", "TF:a,c", "TF:b,c", "TF:a,b,c"} <= names


def test_configuration_enumeration_rejects_overlapping_tf_states():
    sites = [_site("a", 10, 30), _site("b", 20, 40)]
    names = {config.name for config in enumerate_configurations(sites)}
    assert "TF:a,b" not in names


def test_configuration_enumeration_can_omit_nucleosome_state():
    configs = enumerate_configurations([_site("a", 10, 20)], include_nucleosome=False)
    assert all(not config.is_nucleosome for config in configs)


def test_existing_call_maps_to_only_one_overlapping_template():
    sites = [_site("broad", 10, 40), _site("left", 10, 25), _site("right", 27, 40)]
    matched = match_direct_site_indices([IntervalCall(10, 25)], sites)
    assert matched == {1}


def test_em_recovers_dominant_latent_states():
    # Three states; the first 80 reads strongly prefer state 0 and the final
    # 20 strongly prefer state 1. State 2 has no support.
    ll = np.vstack([
        np.tile([8.0, 0.0, 0.0], (80, 1)),
        np.tile([0.0, 8.0, 0.0], (20, 1)),
    ])
    weights, _ = fit_mixture_weights(ll)
    np.testing.assert_allclose(weights[:2], [0.8, 0.2], atol=0.01)
    assert weights[2] < 0.01


def test_nucleosome_multiplier_changes_posterior_not_likelihood():
    sites = [_site("a", 10, 20)]
    configs = enumerate_configurations(sites)
    prior = np.full(len(configs), 1.0 / len(configs))
    ll = np.zeros(len(configs))
    base = posterior_with_prior_multiplier(ll, prior, configs, 1.0)
    boosted = posterior_with_prior_multiplier(ll, prior, configs, 100.0)
    nuc = next(i for i, config in enumerate(configs) if config.is_nucleosome)
    assert boosted[nuc] > base[nuc]


def test_global_prior_marginalizes_occupancy_outside_local_call():
    sites = [_site("a", 10, 20), _site("b", 30, 40)]
    global_configs = enumerate_configurations(sites)
    by_name = {config.name: i for i, config in enumerate(global_configs)}
    prior = np.zeros(len(global_configs))
    prior[by_name["A"]] = 0.10
    prior[by_name["TF:a"]] = 0.20
    prior[by_name["TF:b"]] = 0.15
    prior[by_name["TF:a,b"]] = 0.25
    prior[by_name["N"]] = 0.30
    local_configs = enumerate_configurations([sites[0]])
    local = marginalize_configuration_prior(global_configs, prior, [0], local_configs)
    local_by_name = {config.name: local[i] for i, config in enumerate(local_configs)}
    np.testing.assert_allclose(local_by_name["A"], 0.25)
    np.testing.assert_allclose(local_by_name["TF:a"], 0.45)
    np.testing.assert_allclose(local_by_name["N"], 0.30)


def test_local_nuc_likelihood_uses_exact_current_call_span():
    read = ReadEvidence(
        name="r",
        strand="FWD",
        ref_start=0,
        ref_end=100,
        positions=np.array([10, 15, 25, 35]),
        steps=np.array([2.0, 3.0, -4.0, -5.0]),
        hits=np.array([False, False, True, True]),
        contexts=np.zeros(4, dtype=int),
        tfs=[],
        nucs=[],
        msps=[],
    )
    sites = [_site("a", 10, 20)]
    configs = enumerate_configurations(sites)
    values, evidence = local_call_likelihoods(
        read, sites, configs, IntervalCall(10, 30))
    by_name = {config.name: values[i] for i, config in enumerate(configs)}
    assert by_name["A"] == 0.0
    assert by_name["TF:a"] == 5.0
    assert by_name["N"] == 1.0
    assert evidence[0][:2] == (5.0, 2)


def test_ddda_radial_nuc_likelihood_rewards_fingerprint_misses():
    scorer = DddaRadialNucScorer(
        profile=NucProfile(
            radial=np.full(74, 0.10), linker=0.75, half=73,
            min_sep=150, edge_frac=0.82,
        ),
        context_llr=np.zeros(4096),
        accessible_hit_probability=np.full(4096, 0.80),
    )
    protected = ReadEvidence(
        name="protected", strand="CT", ref_start=0, ref_end=200,
        positions=np.array([80, 90, 100, 110]),
        steps=np.zeros(4),
        hits=np.array([True, False, False, False]),
        contexts=np.zeros(4, dtype=int),
        tfs=[], nucs=[], msps=[],
    )
    accessible = ReadEvidence(
        name="accessible", strand="CT", ref_start=0, ref_end=200,
        positions=protected.positions.copy(), steps=np.zeros(4),
        hits=np.ones(4, dtype=bool), contexts=np.zeros(4, dtype=int),
        tfs=[], nucs=[], msps=[],
    )
    call = IntervalCall(27, 174)
    assert scorer.score(protected, call) > 0.0
    assert scorer.score(accessible, call) < 0.0
    configs = enumerate_configurations([_site("a", 90, 110)])
    values, _ = local_call_likelihoods(
        protected, [_site("a", 90, 110)], configs, call,
        nuc_scorer=scorer,
    )
    nuc_index = next(i for i, config in enumerate(configs) if config.is_nucleosome)
    assert values[nuc_index] == scorer.score(protected, call)


def test_site_model_does_not_require_read_to_span_flanking_window():
    site = _site("short", 100, 120)
    read = ReadEvidence(
        name="short-read", strand="FWD", ref_start=100, ref_end=120,
        positions=np.array([105, 115]), steps=np.array([3.0, 3.0]),
        hits=np.array([False, False]), contexts=np.zeros(2, dtype=int),
        tfs=[], nucs=[], msps=[],
    )
    fitted = fit_site_state_model(
        [read], site, flank=100, include_nucleosome=False)
    assert fitted["coverage"] == 1
    assert fitted["weights"]["TF"] > 0.99


def test_held_out_targets_do_not_enter_source_site_model():
    site = _site("held-out", 100, 120)

    def read(name, strand):
        return ReadEvidence(
            name=name, strand=strand, ref_start=90, ref_end=130,
            positions=np.array([], dtype=int), steps=np.array([], dtype=float),
            hits=np.array([], dtype=bool), contexts=np.array([], dtype=int),
            tfs=[], nucs=[], msps=[],
        )

    prior = [read("prior-fwd", "FWD"), read("prior-rev", "REV")]
    targets = [
        read("target-fwd-1", "FWD"),
        read("target-fwd-2", "FWD"),
        read("target-rev", "REV"),
    ]
    result = analyze_window(
        prior, [site], target_reads=targets, flank=35,
        posterior_threshold=0.95, nuc_multipliers=[100.0],
        max_examples=2, allow_nuc_rescue=False, min_source_support=1,
    )
    assert result["site_models"]["FWD"]["held-out"]["coverage"] == 1
    assert result["site_models"]["REV"]["held-out"]["coverage"] == 1
    assert result["cross_strand"]["FWD"]["scenarios"]["100.0"][
        "configuration_reads"
    ] == 2
    assert result["cross_strand"]["REV"]["scenarios"]["100.0"][
        "configuration_reads"
    ] == 1


def test_nonfocal_source_calls_cannot_create_strand_rescue_prior():
    site = _site("nonfocal", 100, 120)
    site.local_enrichment_by_strand = {"FWD": 0.5, "REV": 0.5}

    def read(name, strand, *, msp=False):
        return ReadEvidence(
            name=name, strand=strand, ref_start=90, ref_end=130,
            positions=np.array([105, 115]), steps=np.array([3.0, 3.0]),
            hits=np.array([False, False]), contexts=np.zeros(2, dtype=int),
            tfs=[], nucs=[], msps=[IntervalCall(90, 130)] if msp else [],
        )

    source_fwd = read("source", "FWD")
    prior_rev = read("prior-rev", "REV")
    target_rev = read("target", "REV", msp=True)
    result = analyze_window(
        [source_fwd, prior_rev], [site], target_reads=[target_rev], flank=35,
        posterior_threshold=0.95, nuc_multipliers=[1.0], max_examples=2,
        allow_nuc_rescue=False, include_nucleosome_state=True,
        min_source_support=1, min_source_local_enrichment=1.5,
    )
    counts = result["cross_strand"]["REV"]["scenarios"]["1.0"]["counts"]
    assert counts["eligible"] == 0
    assert counts["proposed"] == 0


def test_pacbio_population_model_never_creates_alignment_strand_comparison():
    site = _site("duplex", 100, 120)
    site.support = {"BOTH": 10}
    site.all_support = {"BOTH": 10}
    site.median_tq = {"BOTH": 120.0}

    def read(name):
        return ReadEvidence(
            name=name, strand="BOTH", ref_start=90, ref_end=130,
            positions=np.array([105, 115]), steps=np.array([3.0, 3.0]),
            hits=np.array([False, False]), contexts=np.zeros(2, dtype=int),
            tfs=[], nucs=[], msps=[],
        )

    result = analyze_window(
        [read("prior-1"), read("prior-2")], [site],
        target_reads=[read("held-out")], flank=35,
        posterior_threshold=0.95, nuc_multipliers=[1.0],
        max_examples=2, allow_nuc_rescue=False, min_source_support=1,
        consensus_mode="population",
    )
    assert result["cross_strand"] == {}
    assert result["joint_models"]["BOTH"]["coverage"] == 2
    assert "BOTH" in result["population_consensus"]


def test_population_model_reports_at_limited_ambiguous_nuc_for_review():
    site = _site("focal", 120, 160)
    site.support = {"BOTH": 10}
    site.all_support = {"BOTH": 10}
    site.median_tq = {"BOTH": 120.0}
    prior = ReadEvidence(
        name="prior", strand="BOTH", ref_start=80, ref_end=200,
        positions=np.array([100, 130, 150, 180]),
        steps=np.array([-3.0, 3.0, 3.0, -3.0]),
        hits=np.array([True, False, False, True]),
        contexts=np.zeros(4, dtype=int), tfs=[], nucs=[], msps=[],
    )
    target = ReadEvidence(
        name="target", strand="BOTH", ref_start=80, ref_end=200,
        positions=np.array([100, 130, 150]),
        steps=np.array([1.0, 3.0, 3.0]),
        hits=np.array([False, False, False]),
        contexts=np.zeros(3, dtype=int), tfs=[],
        nucs=[IntervalCall(90, 190)], msps=[],
    )
    result = analyze_window(
        [prior], [site], target_reads=[target], flank=35,
        posterior_threshold=0.95, nuc_multipliers=[1.0],
        max_examples=2, allow_nuc_rescue=True, min_source_support=1,
        consensus_mode="population",
    )
    scenario = result["population_consensus"]["BOTH"]["scenarios"]["1.0"]
    assert scenario["candidate_state_count"] == 1
    aggressive = scenario["candidate_states"][0]
    assert aggressive["current"] == "N"
    assert aggressive["proposal_tier"] == "retain_current"
    assert aggressive["posterior"] + aggressive["current_posterior"] == 1.0
    assert aggressive["site_evidence"][0]["opportunities"] >= 1
    assert scenario["counts"]["nuc_likelihood_ambiguous"] == 1
    assert scenario["counts"]["nuc_consensus_review"] == 1
    review = scenario["nuc_review_examples"][0]
    assert review["outside_tf_opportunities"] == 1
    assert review["log_bf_candidate_vs_nuc"] == -1.0


def test_strand_rescue_skips_nucleosome_targets_above_220_bp():
    site = _site("focal", 120, 160)
    site.support = {"BOTH": 10}
    site.all_support = {"BOTH": 10}
    site.median_tq = {"BOTH": 120.0}
    prior = ReadEvidence(
        name="prior", strand="BOTH", ref_start=50, ref_end=400,
        positions=np.array([130, 150]), steps=np.array([3.0, 3.0]),
        hits=np.array([False, False]), contexts=np.zeros(2, dtype=int),
        tfs=[], nucs=[], msps=[],
    )
    target = ReadEvidence(
        name="target", strand="BOTH", ref_start=50, ref_end=400,
        positions=np.array([130, 150]), steps=np.array([3.0, 3.0]),
        hits=np.array([False, False]), contexts=np.zeros(2, dtype=int),
        tfs=[], nucs=[IntervalCall(80, 301)], msps=[],
    )
    result = analyze_window(
        [prior], [site], target_reads=[target], flank=35,
        posterior_threshold=0.95, nuc_multipliers=[1.0],
        max_examples=2, allow_nuc_rescue=True, min_source_support=1,
        max_nuc_span=220, consensus_mode="population",
    )
    scenario = result["population_consensus"]["BOTH"]["scenarios"]["1.0"]
    assert scenario["counts"]["nuc_above_max_span"] == 1
    assert scenario["counts"]["eligible_nuc"] == 0
    assert scenario["candidate_state_count"] == 0


def test_strand_rescue_skips_nuc_already_overwritten_by_existing_tf():
    site = _site("focal", 120, 160)
    site.support = {"BOTH": 10}
    site.all_support = {"BOTH": 10}
    site.median_tq = {"BOTH": 120.0}
    prior = ReadEvidence(
        name="prior", strand="BOTH", ref_start=50, ref_end=250,
        positions=np.array([130, 150]), steps=np.array([3.0, 3.0]),
        hits=np.array([False, False]), contexts=np.zeros(2, dtype=int),
        tfs=[], nucs=[], msps=[],
    )
    target = ReadEvidence(
        name="target", strand="BOTH", ref_start=50, ref_end=250,
        positions=np.array([130, 150]), steps=np.array([3.0, 3.0]),
        hits=np.array([False, False]), contexts=np.zeros(2, dtype=int),
        # This direct TF overlaps the nuc but not the focal site itself.
        tfs=[IntervalCall(80, 100, 150)],
        nucs=[IntervalCall(90, 190)], msps=[],
    )
    result = analyze_window(
        [prior], [site], target_reads=[target], flank=35,
        posterior_threshold=0.95, nuc_multipliers=[1.0],
        max_examples=2, allow_nuc_rescue=True, min_source_support=1,
        consensus_mode="population",
    )
    scenario = result["population_consensus"]["BOTH"]["scenarios"]["1.0"]
    assert scenario["counts"]["nuc_overlaps_existing_tf"] == 1
    assert scenario["candidate_state_count"] == 0
