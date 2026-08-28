from __future__ import annotations

import importlib.util
import json
from collections import Counter
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest


WORKSPACE = Path(__file__).resolve().parents[2]


def _load(name: str, relative: str):
    path = WORKSPACE / relative
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


stability = _load(
    "iterative_tf_geometry_stability_driver",
    "paper/analysis/strand_consensus/run_iterative_tf_geometry_stability.py",
)
stability_plot = _load(
    "iterative_tf_geometry_stability_plot",
    "paper/analysis/strand_consensus/plot_iterative_tf_geometry_stability.py",
)
catalog_compare = _load(
    "iterative_tf_geometry_catalog_compare",
    "paper/analysis/strand_consensus/compare_tf_geometry_catalogs.py",
)


class _Read:
    def __init__(self, strand: str, index: int):
        self.strand = strand
        self.molecule_id = (strand, str(index))


def test_stratified_nested_order_preserves_proportions_at_every_prefix():
    reads = [_Read("CT", index) for index in range(80)] + [
        _Read("GA", index) for index in range(20)
    ]
    ordered = stability.stratified_nested_order(
        reads,
        seed="test",
        partition_id="locus",
        replicate=0,
    )

    assert len(ordered) == len(reads)
    assert len({read.molecule_id for read in ordered}) == len(reads)
    for depth in range(1, len(reads) + 1):
        counts = Counter(read.strand for read in ordered[:depth])
        assert abs(counts["CT"] - 0.8 * depth) <= 1.0
        assert abs(counts["GA"] - 0.2 * depth) <= 1.0


def test_first_persistent_row_rejects_a_later_failure():
    rows = [
        {"training_depth": 40, "qualified": True},
        {"training_depth": 80, "qualified": True},
        {"training_depth": 100, "qualified": False},
    ]
    assert (
        stability.first_persistent_row(rows, lambda row: row["qualified"])
        is None
    )
    rows[-1]["qualified"] = True
    assert stability.first_persistent_row(
        rows, lambda row: row["qualified"]
    )["training_depth"] == 40


def test_persistent_threshold_rejects_first_success_above_fraction_cap():
    rows = [
        {"training_depth": 40, "qualified": False, "eligible": True},
        {"training_depth": 60, "qualified": True, "eligible": False},
        {"training_depth": 100, "qualified": True, "eligible": False},
    ]

    assert stability.first_persistent_row(
        rows,
        lambda row: row["qualified"],
        lambda row: row["eligible"],
    ) is None


def test_wilson_gate_requires_sufficient_replicate_certainty():
    assert stability.minimum_all_success_trials_for_wilson_lower(0.8) == 16
    assert stability.wilson_lower_bound(15, 15) < 0.8
    assert stability.wilson_lower_bound(16, 16) > 0.8
    assert stability.wilson_lower_bound(20, 20) > 0.8
    assert stability.wilson_lower_bound(19, 20) < 0.8
    assert stability.wilson_lower_bound(4, 5) < 0.8


def test_clone_reads_with_frozen_steps_preserves_source_evidence():
    class Evidence:
        pass

    read = Evidence()
    read.steps = np.asarray([1.0, 2.0], dtype=np.float64)
    read.label = "source"
    frozen = np.asarray([-3.0, 4.0], dtype=np.float64)

    cloned = stability.clone_reads_with_frozen_steps(
        [read], {id(read): frozen}
    )[0]

    np.testing.assert_array_equal(read.steps, [1.0, 2.0])
    np.testing.assert_array_equal(cloned.steps, frozen)
    assert cloned is not read
    assert cloned.label == "source"


def test_prefix_overlap_summary_reports_shared_molecule_fraction():
    first = [_Read("CT", index) for index in (0, 1, 2, 3)]
    second = [_Read("CT", index) for index in (0, 1, 4, 5)]

    observed = stability.prefix_overlap_summary([first, second], 3)

    assert observed["pairwise_shared_molecule_fraction_median"] == pytest.approx(
        2 / 3
    )
    assert observed["pairwise_shared_molecule_fraction_p95"] == pytest.approx(
        2 / 3
    )


def test_edge_metric_minima_keep_accumulated_q_distinct_from_effect_size():
    metrics = stability.minimum_edge_metrics(
        [
            {
                "pooled_effective_edge_opportunities": 14.0,
                "pooled_information_margin_to_best_non_equivalent_nats": 23.0,
                "pooled_information_margin_per_effective_molecule_nats": 0.7,
            },
            {
                "pooled_effective_edge_opportunities": 19.0,
                "pooled_information_margin_to_best_non_equivalent_nats": 31.0,
                "pooled_information_margin_per_effective_molecule_nats": 1.2,
            },
        ]
    )

    assert metrics == {
        "minimum_pooled_effective_edge_opportunities": 14.0,
        "minimum_pooled_edge_information_nats": 23.0,
        "minimum_pooled_edge_q_margin_nats": 23.0,
        "minimum_pooled_edge_q_margin_per_effective_molecule_nats": 0.7,
    }


def test_model_edge_extraction_maps_accumulated_q_and_effect_size_fields():
    edge = {
        "status": "resolved_both_strata",
        "selected_coordinate": 101,
        "candidate_coordinate_range": [98, 102],
        "selected_at_search_boundary": False,
        "pooled_effective_edge_opportunities": 14.0,
        "pooled_information_spread_nats": 23.0,
        "pooled_information_margin_to_best_non_equivalent_nats": 23.0,
        "pooled_information_margin_per_effective_molecule_nats": 0.7,
        "pooled_profile_range_nats": 99.0,
        "information_basis": "profiled_q",
        "effective_molecule_support_by_strand": {"CT": 10.0},
        "pooled_information_qualified": True,
        "operationally_identified": True,
        "pooled_maximizing_opportunity_projection_class_count": 1,
        "effective_edge_opportunities_by_strand": {"CT": 14.0},
        "raw_edge_opportunities_by_strand": {"CT": 20},
        "residual_aware_edge_opportunities_by_strand": {"CT": 14.0},
        "residual_aware_information_margin_nats_by_strand": {"CT": 2.0},
        "spatial_null_effective_molecule_support_by_strand": {"CT": 0.0},
        "residual_absorbed_strands": [],
        "information_spread_nats_by_strand": {"CT": 23.0},
        "informative_strands": ["CT"],
    }
    records = stability.model_edge_information(
        {
            "geometry": [
                {
                    "site_id": "TF1",
                    "effective_molecule_support": 10.0,
                    "edge_identifiability": {"start": edge, "end": edge},
                }
            ]
        }
    )

    assert len(records) == 2
    assert records[0][
        "pooled_information_margin_to_best_non_equivalent_nats"
    ] == 23.0
    assert records[0][
        "pooled_information_margin_per_effective_molecule_nats"
    ] == 0.7
    assert records[0]["pooled_profile_range_nats"] == 99.0


def test_sensitivity_summary_keeps_variant_and_replicate_axes_distinct():
    rows = [
        {
            "training_depth": 20,
            "variant": "left",
            "all_site_projections_match_reference": value,
            "configuration_quotient_total_variation": tv,
            "heldout_delta_vs_full_per_opportunity": delta,
        }
        for value, tv, delta in (
            (True, 0.04, -0.01),
            (False, 0.12, -0.03),
        )
    ] + [
        {
            "training_depth": 20,
            "variant": "right",
            "all_site_projections_match_reference": True,
            "configuration_quotient_total_variation": 0.02,
            "heldout_delta_vs_full_per_opportunity": 0.01,
        }
    ]

    summary = stability_plot.summarize_sensitivity_by_depth(
        rows, ["left", "right"]
    )

    assert [record["variant"] for record in summary] == ["left", "right"]
    assert summary[0]["replicates"] == 2
    assert summary[0]["exact_projection_match_fraction"] == 0.5
    assert summary[0]["configuration_tv_median"] == pytest.approx(0.08)
    assert summary[0][
        "heldout_predictive_loss_per_opportunity_median"
    ] == pytest.approx(0.02)
    assert summary[1]["replicates"] == 1


def test_cross_chemistry_concordance_requires_both_edges_operational():
    def bundle(coordinates, operational):
        edge = {
            "selected_coordinate": coordinates[0],
            "pooled_maximizing_coordinates": coordinates,
            "operationally_identified": operational,
        }
        return {
            "model": {
                "geometry": [
                    {
                        "site_id": "TF1",
                        "edge_identifiability": {
                            "start": edge,
                            "end": edge,
                        },
                    }
                ]
            },
            "summary": {
                "fixed_site_catalog": [
                    {"site_id": "TF1", "start": 100, "end": 120}
                ]
            },
            "runs": [
                {
                    "variant": "complete.multistart",
                    "is_full_training_depth": False,
                    "threshold_estimation_eligible": True,
                    "training_depth": 20,
                    "replicate": 0,
                    "maximizing_intervals": "[[[100,120]]]",
                }
            ],
            "edges": [
                {
                    "variant": "complete.multistart",
                    "training_depth": 20,
                    "replicate": 0,
                    "site_id": "TF1",
                    "edge": edge_name,
                    "operationally_identified": operational,
                }
                for edge_name in ("start", "end")
            ],
        }

    entries = [
        ("NAPA DddA", "ddda", "#000", "NAPA", bundle([100], True)),
        (
            "NAPA Fiber-seq",
            "hia5",
            "#111",
            "NAPA",
            bundle([100, 101], False),
        ),
    ]

    full, subsamples = stability_plot.cross_chemistry_concordance(entries)

    assert {row["concordance_status"] for row in full} == {
        "insufficient_or_conflicted_evidence"
    }
    assert all(not row["both_operational"] for row in subsamples)
    assert all(not row["operational_concordant"] for row in subsamples)


def test_catalog_comparison_is_paired_by_molecule_and_opportunity():
    left = {
        "molecules": {
            ("read", "1"): {"log_likelihood_ratio": 5.0, "opportunities": 10},
            ("read", "2"): {"log_likelihood_ratio": 2.0, "opportunities": 5},
        }
    }
    right = {
        "molecules": {
            ("read", "1"): {"log_likelihood_ratio": 3.0, "opportunities": 10},
            ("read", "2"): {"log_likelihood_ratio": 0.0, "opportunities": 5},
        }
    }

    deltas, opportunities = catalog_compare.paired_arrays(left, right)
    observed, lower, upper = catalog_compare.bootstrap_ratio(
        deltas,
        opportunities,
        seed=17,
        replicates=200,
    )

    assert observed == pytest.approx(4 / 15)
    assert lower > 0
    assert upper > lower


def test_catalog_bundle_spec_allows_cardinality_in_label():
    label, path = catalog_compare.parse_bundle_spec("raw K=4=/tmp/catalog")

    assert label == "raw K=4"
    assert path == Path("/tmp/catalog")


def test_catalog_comparison_rejects_changed_null_universe():
    base_summary = {field: "same" for field in catalog_compare.COMPARABILITY_FIELDS}
    entries = [
        {
            "label": "K1",
            "summary": deepcopy(base_summary),
            "molecules": {("read", "1"): {"opportunities": 10}},
        },
        {
            "label": "K2",
            "summary": deepcopy(base_summary),
            "molecules": {("read", "1"): {"opportunities": 10}},
        },
    ]
    entries[1]["summary"]["spatial_null_exclusion_sha256"] = "changed"

    with pytest.raises(RuntimeError, match="spatial_null_exclusion_sha256"):
        catalog_compare.assert_comparable(entries)


def _write_completed_bundle(root: Path, name: str, *, schema: str = "fiberhmm.iterative_tf_geometry_stability.v3") -> Path:
    bundle = root / name
    (bundle / "models").mkdir(parents=True)
    summary = {
        "schema": schema,
        "status": catalog_compare.RUNNER_SUMMARY_STATUS,
        "dataset_id": name,
        "completion_manifest": "completion_manifest.json",
        "reference_model": "models/full_reference.json",
        "reference_heldout_score": "models/full_reference.heldout-score.json",
        "heldout_molecules": 1,
        "skipped_variant_runs": 0,
        "skipped_variants_table": None,
    }
    files = {
        "summary.json": json.dumps(summary) + "\n",
        "models/full_reference.json": json.dumps(
            {
                "model_id": "model",
                "configuration_names": [
                    "A",
                    "P0:unanchored_single_interval",
                    "U:diffuse_iid_opportunity_protection",
                ],
                "geometry": [],
            }
        )
        + "\n",
        "models/full_reference.heldout-score.json": json.dumps(
            {
                "model_id": "model",
                "molecules": [
                    {
                        "molecule_id": ["library", "read", "CT"],
                        "log_likelihood_ratio": 1.0,
                        "opportunities": 5,
                    }
                ],
            }
        )
        + "\n",
        "stability_summary.tsv": "column\n",
        "stability_runs.tsv": "column\n",
        "edge_information.tsv": "column\n",
        "edge_stability_summary.tsv": "column\n",
        "site_stability_summary.tsv": "column\n",
    }
    for relative, content in files.items():
        (bundle / relative).write_text(content)
    artifacts = []
    for relative in files:
        path = bundle / relative
        artifacts.append(
            {
                "relative_path": relative,
                "bytes": path.stat().st_size,
                "sha256": stability_plot.sha256_file(path),
                "data_rows": 0 if relative.endswith(".tsv") else None,
            }
        )
    manifest = {
        "schema": "fiberhmm.iterative_tf_geometry_bundle_completion.v1",
        "status": "complete",
        "dataset_id": name,
        "reference_model_id": "model",
        "reference_score_model_id": "model",
        "artifacts": artifacts,
    }
    (bundle / "completion_manifest.json").write_text(json.dumps(manifest) + "\n")
    return bundle


def test_catalog_comparator_loads_real_runner_status(tmp_path: Path):
    bundle = _write_completed_bundle(tmp_path, "actual-status")

    loaded = catalog_compare.load_bundle("actual", bundle)

    assert loaded["summary"]["status"] == catalog_compare.RUNNER_SUMMARY_STATUS
    assert len(loaded["molecules"]) == 1


def test_plot_rejects_stale_pre_v3_bundle(tmp_path: Path):
    _write_completed_bundle(
        tmp_path,
        "stale",
        schema="fiberhmm.iterative_tf_geometry_stability.v2",
    )

    with pytest.raises(RuntimeError, match="not a current v3 stability bundle"):
        stability_plot.load_bundle(tmp_path, "stale")


def test_plot_rejects_partially_overwritten_completed_bundle(tmp_path: Path):
    bundle = _write_completed_bundle(tmp_path, "tampered")
    (bundle / "models/full_reference.json").write_text(
        json.dumps({"model_id": "new-partial-model"}) + "\n"
    )

    with pytest.raises(RuntimeError, match="size mismatch|digest mismatch"):
        stability_plot.load_bundle(tmp_path, "tampered")


def test_figure_commit_never_overwrites_a_completed_output(tmp_path: Path):
    staging = tmp_path / ".figure-staging"
    final = tmp_path / "figure"
    staging.mkdir()
    final.mkdir()
    (staging / "new.png").write_bytes(b"new")
    (final / "old.png").write_bytes(b"old")

    with pytest.raises(RuntimeError, match="already exists"):
        stability_plot.commit_fresh_directory(staging, final)

    assert (final / "old.png").read_bytes() == b"old"
    assert (staging / "new.png").read_bytes() == b"new"


def _comparison_entry(label: str, directory: str, chemistry: str, semantics: str):
    catalog = [{"site_id": "TF1", "start": 100, "end": 120}]
    summary = {
        "fiberhmm_version": "2.16.3",
        "runner_sha256": "runner",
        "strand_rescue_source_sha256": "core",
        "dataset_id": directory,
        "chemistry": chemistry,
        "stratum_semantics": semantics,
        "fixed_site_catalog": catalog,
        "region": ["chr19", 80, 140],
        "reference_contig": {
            "name": "chr19",
            "length": 58617616,
            "md5": None,
            "assembly": None,
            "uri": None,
        },
        "catalog_source_sha256": "catalog-source",
        "catalog_key": "napa_daf_catalog",
        "catalog_source_record": {
            "derivation_assay": "GM12878 targeted DddA DAF-seq",
            "region": ["chr19", 80, 140],
            "analysis_envelope": [80, 140],
            "sites": catalog,
        },
        "analysis_envelope": [80, 140],
        "boundary_search_radius": 6,
        "spatial_null_exclusion_catalog": catalog,
        "spatial_null_exclusion_radius": 8,
        "spatial_null_exclusion_interval_count": 225,
        "spatial_null_exclusion_sha256": "shared",
        "spatial_null_padding": 20,
        "spatial_null_width_range": [1, 80],
        "spatial_null_requested_width_range": [1, 80],
        "spatial_null_effective_width_range": [1, 60],
        "minimum_molecule_opportunities": 3,
        "minimum_mapping_quality": 20,
        "hierarchical_prior": {"family_pseudocount": 0.5},
        "fit_algorithm_controls": {"maximum_iterations": 100},
    }
    model = {
        "geometry": [
            {"site_id": "TF1", "seed_interval": [100, 120]}
        ],
        "envelope": [80, 140],
        "boundary_search_radius": 6,
        "spatial_null_exclusion_sha256": "shared",
        "stratum_semantics": semantics,
    }
    return (label, directory, "#000000", "NAPA", {"summary": summary, "model": model})


@pytest.mark.parametrize(
    "mismatch",
    [
        "catalog",
        "analysis_envelope",
        "p0_digest",
        "region",
        "catalog_source",
        "implementation",
    ],
)
def test_plot_rejects_cross_chemistry_comparison_invariant_mismatch(mismatch):
    ddda = _comparison_entry(
        "NAPA DddA",
        "napa_ddda_daf_catalog",
        "ddda",
        "physical_complementary",
    )
    hia5 = _comparison_entry(
        "NAPA Fiber-seq",
        "napa_hia5_daf_catalog",
        "hia5-pacbio",
        "diagnostic_partition",
    )
    hia5 = (*hia5[:4], deepcopy(hia5[4]))
    if mismatch == "catalog":
        hia5[4]["summary"]["fixed_site_catalog"][0]["start"] = 101
        hia5[4]["model"]["geometry"][0]["seed_interval"][0] = 101
    elif mismatch == "analysis_envelope":
        hia5[4]["summary"]["analysis_envelope"] = [79, 140]
        hia5[4]["model"]["envelope"] = [79, 140]
    elif mismatch == "p0_digest":
        hia5[4]["summary"]["spatial_null_exclusion_sha256"] = "different"
        hia5[4]["model"]["spatial_null_exclusion_sha256"] = "different"
    elif mismatch == "region":
        hia5[4]["summary"]["region"][0] = "chr20"
    elif mismatch == "catalog_source":
        hia5[4]["summary"]["catalog_source_sha256"] = "different"
    else:
        hia5[4]["summary"]["runner_sha256"] = "different"

    with pytest.raises(RuntimeError, match="mismatch|differs|implementation digest"):
        stability_plot.validate_entries([ddda, hia5])
