from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from consensus_recaller_collab.prototype import IntervalCall, ReadEvidence
from consensus_recaller_collab.prototype import resolve_resource_path
from consensus_recaller_collab.validation.evidence import (
    Candidate,
    _collapse_molecule_families,
    _fine_read_posterior,
    _fit_state_model,
    _resolve_bam as resolve_evidence_bam,
    build_shift_controls,
    discover_candidates,
)
from consensus_recaller_collab.validation.calibration import (
    build_pseudo_site_nulls,
    calibrate_fine_tf,
)
from consensus_recaller_collab.validation.hierarchy import (
    adjudicate_candidate,
    load_json,
    validate_hierarchy,
)
from consensus_recaller_collab.validation.pipeline import (
    _resolve_bam as resolve_manifest_bam,
    validate_manifest_structure,
)
from consensus_recaller_collab.validation.summary import summarize_fine_tf


VALIDATION_DIR = Path(__file__).with_name("validation")
HIERARCHY = load_json(str(VALIDATION_DIR / "default_hierarchy.json"))


def sample(sample_id, family, *, permission, negative=True, truth_vote=True):
    return {
        "sample_id": sample_id,
        "cohort_id": "cohort",
        "assay_family": family,
        "availability": "available",
        "negative_evidence_allowed": negative,
        "truth_vote": truth_vote,
        "axis_permissions": {
            "broad_nucleosome": "none",
            "fine_tf": permission,
            "occupancy_frequency": "none",
        },
    }


def record(sample_id, positive, informative, interval=None):
    result = {
        "cohort_id": "cohort",
        "locus_id": "locus",
        "candidate_id": "tf1",
        "axis": "fine_tf",
        "sample_id": sample_id,
        "positive": positive,
        "informative": informative,
    }
    if interval is not None:
        result["interval"] = list(interval)
        result["geometry_interval"] = list(interval)
    return result


def test_default_hierarchy_and_local_manifest_are_structurally_valid():
    assert validate_hierarchy(HIERARCHY) == []
    manifest = load_json(str(VALIDATION_DIR / "manifests" / "core_local.json"))
    errors, warnings = validate_manifest_structure(manifest)
    assert errors == []
    assert len(warnings) == 2


def test_core_manifest_uses_authoritative_primary_panels():
    manifest = load_json(str(VALIDATION_DIR / "manifests" / "core_local.json"))
    samples = {sample["sample_id"]: sample for sample in manifest["samples"]}
    assert samples["gm12878_hia5_pacbio_full"]["bams"] == [
        "/mnt/g/gm12878/GM12878.fiberhmm.bam"
    ]
    assert set(samples["fly_hia5_pacbio_2_4hr"]["bams"]) == {
        f"/mnt/g/v3seg_mp/2-4hr_{suffix}.bam"
        for suffix in (11, 4, 6, 7, 9)
    }
    assert samples["fly_dddb_yw_2_4hr_pooled"]["bams"] == [
        "/mnt/g/Dropbox/Fiber-NET-seq/Drosophila_phase2/Datasets/DAF-seq/"
        "spacetime_updated/fp_update/yw_2-4_recalled.bam"
    ]
    assert all(sample["assay_family"] != "scdaf" for sample in manifest["samples"])
    nanopore = [
        sample for sample in manifest["samples"]
        if sample["assay_family"] == "hia5_nanopore"
    ]
    assert len(nanopore) == 2
    assert all(sample["truth_vote"] is False for sample in nanopore)


def test_installed_resource_and_manifest_relative_paths_resolve(tmp_path):
    model = Path(resolve_resource_path("fiberhmm/models/hia5_pacbio.json"))
    assert model.is_file()
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")
    bam = tmp_path / "relative.bam"
    bam.write_bytes(b"")
    assert resolve_manifest_bam("relative.bam", str(manifest)) == bam


def test_nested_manifest_resolves_repository_relative_inputs(tmp_path):
    repository = tmp_path / "repository"
    manifest = repository / "consensus" / "validation" / "manifests" / "core.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("{}")
    bam = repository / "ddda_nuc_output" / "target.bam"
    bam.parent.mkdir()
    bam.write_bytes(b"")

    relative = "ddda_nuc_output/target.bam"
    assert resolve_manifest_bam(relative, str(manifest)) == bam
    assert resolve_evidence_bam(relative, base_dir=manifest.parent) == bam


def test_ddda_defines_held_out_pacbio_fine_geometry():
    manifest = {
        "samples": [
            sample("pb", "hia5_pacbio", permission="anchor"),
            sample("ddda", "ddda", permission="anchor", negative=False),
        ]
    }
    result = adjudicate_candidate(
        [
            record("pb", 0, 20, (105, 135)),
            record("ddda", 19, 20, (100, 130)),
        ],
        manifest,
        HIERARCHY,
        target_assay_family="hia5_pacbio",
    )
    assert result["status"] == "anchored_present"
    assert result["anchor_families"] == ["ddda"]
    assert result["geometry"] == {
        "interval": [100, 130],
        "geometry_tier": 1,
        "source_samples": ["ddda"],
    }


def test_pacbio_defines_held_out_ddda_fine_geometry_at_lower_resolution_tier():
    manifest = {
        "samples": [
            sample("pb", "hia5_pacbio", permission="anchor"),
            sample("ddda", "ddda", permission="anchor"),
        ]
    }
    result = adjudicate_candidate(
        [
            record("pb", 17, 20, (104, 137)),
            record("ddda", 0, 20, (100, 130)),
        ],
        manifest,
        HIERARCHY,
        target_assay_family="ddda",
    )
    assert result["status"] == "anchored_present"
    assert result["geometry"]["interval"] == [104, 137]
    assert result["geometry"]["geometry_tier"] == 2


def test_one_low_tier_assay_cannot_create_truth():
    manifest = {
        "samples": [sample("dddb", "dddb", permission="support")]
    }
    result = adjudicate_candidate(
        [record("dddb", 80, 80)], manifest, HIERARCHY
    )
    assert result["status"] == "unresolved_no_anchor"
    assert result["posterior_present"] > 0.99


def test_two_low_tier_assays_can_only_create_provisional_presence():
    manifest = {
        "samples": [
            sample("dddb", "dddb", permission="support"),
            sample("np", "hia5_nanopore", permission="support"),
        ]
    }
    result = adjudicate_candidate(
        [record("dddb", 80, 80), record("np", 60, 60)],
        manifest,
        HIERARCHY,
    )
    assert result["status"] == "provisional_present"
    assert result["geometry"] is None


def test_positive_only_anchor_cannot_turn_missing_calls_into_negative_truth():
    manifest = {
        "samples": [
            sample("ddda", "ddda", permission="anchor", negative=False)
        ]
    }
    result = adjudicate_candidate(
        [record("ddda", 0, 100)], manifest, HIERARCHY
    )
    assert result["status"] == "unresolved"
    assert result["posterior_present"] == pytest.approx(
        HIERARCHY["axes"]["fine_tf"]["prior_present"]
    )


def test_positive_only_anchor_does_not_promote_ordinary_background_mass():
    manifest = {
        "samples": [
            sample("ddda", "ddda", permission="anchor", negative=False)
        ]
    }
    result = adjudicate_candidate(
        [record("ddda", 10, 100)], manifest, HIERARCHY
    )
    assert result["status"] == "unresolved"
    assert result["posterior_present"] == pytest.approx(
        HIERARCHY["axes"]["fine_tf"]["prior_present"]
    )
    assert result["contributors"][0]["unclamped_log_bayes_factor"] < 0.0


def test_exploratory_assay_never_votes_even_with_strong_signal():
    manifest = {
        "samples": [
            sample(
                "np", "hia5_nanopore", permission="support", truth_vote=False
            )
        ]
    }
    result = adjudicate_candidate(
        [record("np", 60, 60)], manifest, HIERARCHY
    )
    assert result["status"] == "unresolved_no_anchor"
    assert result["contributors"] == []
    assert result["posterior_present"] == pytest.approx(
        HIERARCHY["axes"]["fine_tf"]["prior_present"]
    )


def _read(name, positions, hits, steps, *, strand="CT", tfs=None):
    positions = np.asarray(positions, dtype=np.int64)
    return ReadEvidence(
        name=name,
        strand=strand,
        ref_start=int(positions.min()),
        ref_end=int(positions.max()) + 1,
        positions=positions,
        steps=np.asarray(steps, dtype=np.float64),
        hits=np.asarray(hits, dtype=bool),
        contexts=np.zeros(len(positions), dtype=np.int64),
        tfs=list(tfs or []),
        nucs=[],
        msps=[],
    )


def test_amplicon_collapse_counts_deamination_families_not_pcr_copies():
    shared_hits = list(range(100, 110))
    first = _read("copy1", shared_hits, [True] * 10, [1.0] * 10)
    second = _read(
        "copy2",
        [95, *shared_hits, 115],
        [False, *([True] * 10), False],
        [-1.0, *([1.0] * 10), -1.0],
    )
    independent = _read(
        "molecule2", list(range(200, 210)), [True] * 10, [1.0] * 10
    )
    collapsed, diagnostics = _collapse_molecule_families(
        [first, second, independent],
        min_jaccard=0.95,
        min_deam=10,
        ignore_strand=False,
        num_hashes=32,
        bands=8,
        seed=7,
    )
    assert [read.name for read in collapsed] == ["copy2", "molecule2"]
    assert diagnostics["raw_reads"] == 3
    assert diagnostics["analyzed_molecules"] == 2
    assert diagnostics["duplicate_reads_collapsed"] == 1


def test_short_read_can_support_focal_tf_with_only_one_observed_flank():
    read = _read(
        "short",
        [102, 106, 112, 118],
        [True, True, False, False],
        [2.0, 2.0, -2.0, -2.0],
    )
    candidate = Candidate(
        cohort_id="cohort",
        locus_id="locus",
        candidate_id="locus.tf001",
        axis="fine_tf",
        start=100,
        end=110,
        center=105,
        geometry_tier=1,
        source_samples=("anchor",),
        source_support=10,
    )
    posterior = _fine_read_posterior(read, candidate, flank=25)
    assert posterior is not None
    assert posterior > 0.95


def test_nested_mixture_detects_low_occupancy_when_depth_is_strong():
    # Five percent of molecules strongly favor TF; the rest favor accessible.
    # Site existence is therefore supported without assuming high occupancy.
    likelihoods = (
        [np.asarray([0.0, 8.0, -8.0])] * 50
        + [np.asarray([0.0, -8.0, -8.0])] * 950
    )
    fitted = _fit_state_model(
        likelihoods, null_columns=(0, 2), positive_column=1
    )
    assert fitted["state_weights"][1] == pytest.approx(0.05, abs=0.005)
    assert fitted["log_bayes_factor"] > 100.0


def test_strong_support_assay_can_seed_a_candidate_without_defining_truth_geometry():
    reads = [
        _read(
            f"read{index}",
            [100, 110, 120, 130],
            [False, False, False, False],
            [-1.0, -1.0, -1.0, -1.0],
            tfs=[IntervalCall(104, 126, 200)],
        )
        for index in range(6)
    ]
    sample = {
        "sample_id": "dddb",
        "cohort_id": "cohort",
        "assay_family": "dddb",
        "truth_vote": True,
        "axis_permissions": {
            "broad_nucleosome": "support",
            "fine_tf": "support",
            "occupancy_frequency": "support",
        },
        "discovery": {
            "min_tq": 100,
            "min_tf_support": 3,
            "allow_support_seed": True,
            "candidate_geometry_tier": 99,
        },
    }
    candidates = discover_candidates(
        {"samples": [sample]},
        HIERARCHY,
        {"cohort_id": "cohort", "locus_id": "locus", "region": "chr1:0-300"},
        {"dddb": reads},
    )
    assert len(candidates) == 1
    assert candidates[0].geometry_tier == 99
    assert candidates[0].source_samples == ("dddb",)
    assert candidates[0].geometry_source_samples == ("dddb",)
    assert candidates[0].source_diagnostics[0]["local_enrichment"] > 1.5
    assert candidates[0].source_diagnostics[0]["support_by_strand"] == {"CT": 6}


def test_shift_control_is_nonoverlapping_and_opportunity_matched():
    reads = [
        _read(
            "read",
            [55, 100, 110, 150, 160, 205, 215],
            [False] * 7,
            [-1.0] * 7,
        )
    ]
    parent = Candidate(
        "cohort", "locus", "locus.tf001", "fine_tf",
        100, 120, 110, 1, ("anchor",), 10,
    )
    neighbor = Candidate(
        "cohort", "locus", "locus.tf002", "fine_tf",
        200, 220, 210, 1, ("anchor",), 10,
    )
    controls, metadata = build_shift_controls(
        [parent, neighbor],
        {"region": "chr1:0-500"},
        {"anchor": reads},
        controls_per_candidate=1,
        offsets=(-50, 50, 100),
        exclusion_buffer=15,
    )
    parent_control = next(
        item for item in controls if item.candidate_id.startswith("locus.tf001")
    )
    parent_metadata = next(
        item for item in metadata if item["parent_candidate_id"] == "locus.tf001"
    )
    assert (parent_control.start, parent_control.end) == (150, 170)
    assert parent_metadata["offset"] == 50
    assert parent_metadata["source_opportunities"] == 2
    assert parent_metadata["parent_source_opportunities"] == 2


def test_multiple_shift_controls_do_not_overlap_one_another():
    parent = Candidate(
        "cohort", "locus", "locus.tf001", "fine_tf",
        100, 120, 110, 1, ("anchor",), 10,
    )
    controls, _ = build_shift_controls(
        [parent],
        {"region": "chr1:0-500"},
        {"anchor": []},
        controls_per_candidate=3,
        offsets=(50, 75, 100, 125, 150),
        exclusion_buffer=15,
    )
    intervals = sorted((control.start, control.end) for control in controls)
    assert intervals == [(150, 170), (200, 220), (250, 270)]


def test_compact_summary_keeps_support_and_local_control_separate():
    manifest = {
        "samples": [sample("pb", "hia5_pacbio", permission="anchor")]
    }
    evidence = {
        "candidates": [{
            "candidate_id": "tf1",
            "axis": "fine_tf",
            "start": 100,
            "end": 120,
            "geometry_tier": 2,
            "source_samples": ["pb"],
            "source_support": 10,
        }],
        "records": [{
            **record("pb", 5, 20, (100, 120)),
            "log_bayes_factor": 12.0,
            "mean_posterior": 0.25,
            "explicit_tf_reads": 5,
            "by_library": {
                "lib1": {
                    "informative": 10, "site_opportunities": 10,
                    "log_bayes_factor": 8.0, "explicit_tf_reads": 4,
                },
                "lib2": {
                    "informative": 10, "site_opportunities": 10,
                    "log_bayes_factor": -1.0, "explicit_tf_reads": 1,
                },
            },
        }],
        "controls": [{
            "candidate_id": "tf1.control01",
            "parent_candidate_id": "tf1",
        }],
        "control_records": [{
            **record("pb", 2, 20, (150, 170)),
            "candidate_id": "tf1.control01",
            "log_bayes_factor": 4.0,
            "mean_posterior": 0.10,
            "by_library": {
                "lib1": {
                    "informative": 10, "site_opportunities": 10,
                    "log_bayes_factor": 2.0,
                },
                "lib2": {
                    "informative": 10, "site_opportunities": 10,
                    "log_bayes_factor": 0.0,
                },
            },
        }],
    }
    result = summarize_fine_tf(evidence, manifest)
    assert result["candidate_count"] == 1
    assert result["sample_summaries"][0]["supported"] == 1
    assert result["sample_summaries"][0]["locally_enriched"] == 1
    assert result["candidates"][0]["samples"]["pb"][
        "local_delta_log_bayes_factor"
    ] == 8.0
    assert result["candidates"][0]["samples"]["pb"]["local_rank"] == 1
    assert result["candidates"][0]["samples"]["pb"][
        "normalized_local_delta_log_bayes_factor"
    ] == pytest.approx(0.4)
    assert result["candidates"][0]["samples"]["pb"][
        "library_supported_count"
    ] == 1
    assert result["candidates"][0]["samples"]["pb"][
        "library_locally_enriched_count"
    ] == 1


def _calibration_sample_row(
    family, *, source, permission, observed=2.0, diagnostics=None
):
    controls = [
        {
            "candidate_id": f"control{index}",
            "informative": 20,
            "site_opportunities": 50,
            "log_bayes_factor": value * 20,
            "normalized_log_bayes_factor": value,
        }
        for index, value in enumerate((-0.1, 0.0, 0.1), start=1)
    ]
    return {
        "assay_family": family,
        "permission": permission,
        "truth_vote": True,
        "negative_evidence_allowed": True,
        "is_source_sample": source,
        "is_geometry_source": source,
        "source_diagnostics": diagnostics or [],
        "informative": 20,
        "site_opportunities": 50,
        "explicit_tf_reads": 5,
        "log_bayes_factor": observed * 20,
        "normalized_log_bayes_factor": observed,
        "control_count": 3,
        "matched_controls": controls,
        "beats_all_controls": observed > 0.1,
        "normalized_local_delta_log_bayes_factor": observed,
    }


def test_calibration_never_treats_source_selection_as_formal_validation():
    source_diagnostic = [{
        "support": 10,
        "local_enrichment": 3.0,
        "start_mad": 2.0,
        "end_mad": 2.0,
    }]
    candidates = []
    for locus in ("locus1", "locus2"):
        candidates.append({
            "candidate_id": f"{locus}.tf001",
            "cohort_id": "cohort",
            "locus_id": locus,
            "interval": [100, 120],
            "geometry_tier": 2,
            "support_only_seed": False,
            "samples": {
                "pb": _calibration_sample_row(
                    "hia5_pacbio", source=True, permission="anchor",
                    diagnostics=source_diagnostic,
                ),
                "dddb": _calibration_sample_row(
                    "dddb", source=False, permission="support"
                ),
            },
        })
    policy = {
        "evidence": {
            "minimum_controls": 3,
            "minimum_informative": 10,
            "minimum_site_opportunities": 5,
            "support_log_bf": 5.0,
            "source_min_local_enrichment": 1.5,
            "source_max_boundary_mad": 12.0,
            "source_min_support": 3,
            "independent_strong_q": 0.30,
            "independent_review_q": 0.50,
        },
        "calibration": {"minimum_heldout_null": 3},
        "families": {},
    }
    report = calibrate_fine_tf(
        {"candidates": candidates, "sample_summaries": []},
        {"samples": []},
        policy,
    )
    first = report["candidates"][0]
    assert first["samples"]["pb"]["formal_independent_test"] is False
    assert first["samples"]["pb"]["heldout_empirical_q"] is None
    assert first["samples"]["pb"]["calibrated_status"] == "source_focal"
    assert first["samples"]["dddb"]["formal_independent_test"] is True
    assert first["samples"]["dddb"]["heldout_empirical_q"] is not None
    assert first["samples"]["dddb"]["calibrated_status"] == "strong_local"
    assert first["proposal_tier"] == "strong"


def test_independent_source_nomination_is_cross_assay_replication():
    diagnostic = [{
        "support": 12,
        "local_enrichment": 4.0,
        "start_mad": 1.0,
        "end_mad": 1.0,
    }]
    candidate = {
        "candidate_id": "locus.tf001",
        "cohort_id": "cohort",
        "locus_id": "locus",
        "interval": [100, 120],
        "geometry_tier": 2,
        "support_only_seed": False,
        "samples": {
            "pb": _calibration_sample_row(
                "hia5_pacbio", source=True, permission="anchor",
                diagnostics=diagnostic,
            ),
            "dddb": _calibration_sample_row(
                "dddb", source=True, permission="support",
                diagnostics=diagnostic,
            ),
        },
    }
    policy = {
        "evidence": {
            "minimum_controls": 3,
            "minimum_informative": 10,
            "minimum_site_opportunities": 5,
            "support_log_bf": 5.0,
            "source_min_local_enrichment": 1.5,
            "source_max_boundary_mad": 12.0,
            "source_min_support": 3,
            "independent_strong_q": 0.05,
            "independent_review_q": 0.20,
        },
        "calibration": {"minimum_heldout_null": 50},
        "families": {},
    }
    report = calibrate_fine_tf(
        {"candidates": [candidate], "sample_summaries": []},
        {"samples": []},
        policy,
    )
    result = report["candidates"][0]
    assert result["source_focal_families"] == ["dddb", "hia5_pacbio"]
    assert result["proposal_tier"] == "strong"
    assert "multiple assay families" in result["proposal_reason"]


def test_reused_genomic_controls_are_not_counted_as_independent_nulls():
    candidates = []
    for parent in ("tf1", "tf2"):
        sample_row = _calibration_sample_row(
            "dddb", source=False, permission="support"
        )
        for index, control in enumerate(sample_row["matched_controls"]):
            control["candidate_id"] = f"{parent}.control{index}"
            control["interval"] = [100 + 50 * index, 120 + 50 * index]
        candidates.append({
            "candidate_id": parent,
            "cohort_id": "cohort",
            "locus_id": "locus",
            "samples": {"dddb": sample_row},
        })
    nulls = build_pseudo_site_nulls(
        {"candidates": candidates}, minimum_controls=3
    )
    assert len(nulls) == 3
    assert all(row["correlated_control_uses"] == 2 for row in nulls)
