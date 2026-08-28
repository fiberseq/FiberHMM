from __future__ import annotations

import array
import csv
from dataclasses import replace

import json
from pathlib import Path
import numpy as np
import pysam
import pytest

from fiberhmm.inference.targeted_families import (
    TargetedFamilyDiscoveryConfig,
    discover_targeted_families,
    index_informative_windows,
    score_boundary_families_on_unbiased_cohort,
    score_boundary_family_on_unbiased_cohort,
    select_discovery_molecules,
)
from fiberhmm.inference.tf_sites import BaselineMolecule, TFObservation


def _molecule(
    index: int,
    *,
    stratum: str = "CT",
    source: str = "input0",
    msp=(100, 400),
    tf=(210, 228),
):
    return BaselineMolecule(
        molecule_id=f"{source}\x1fread{index:04d}",
        contig="chr1",
        mapped_blocks=((0, 2000),),
        tfs=(TFObservation(f"tf.{index}", *tf),) if tf else (),
        msps=(msp,) if msp else (),
        stratum=stratum,
    )


def test_informative_window_gate_skips_closed_core():
    molecules = tuple(_molecule(index) for index in range(10))
    config = TargetedFamilyDiscoveryConfig(
        core_size=1000,
        minimum_nfr_length=150,
        minimum_informative_molecules=3,
        minimum_informative_fraction=0.1,
    )
    windows = index_informative_windows(
        molecules,
        contig="chr1",
        locus_start=0,
        locus_end=2000,
        config=config,
    )
    assert len(windows) == 2
    assert windows[0].fully_mapped_molecules == 10
    assert windows[0].long_msp_molecules == 10
    assert windows[0].informative
    assert windows[1].fully_mapped_molecules == 10
    assert windows[1].long_msp_molecules == 0
    assert not windows[1].informative


def test_window_range_index_matches_exhaustive_block_aware_reference():
    molecules = (
        BaselineMolecule(
            molecule_id="split",
            contig="chr1",
            mapped_blocks=((100, 680), (720, 2900)),
            tfs=(),
            msps=((250, 460), (1550, 1800)),
            stratum="CT",
        ),
        BaselineMolecule(
            molecule_id="edge",
            contig="chr1",
            mapped_blocks=((0, 2000),),
            tfs=(),
            msps=((900, 1100),),
            stratum="GA",
        ),
        BaselineMolecule(
            molecule_id="other-contig",
            contig="chr2",
            mapped_blocks=((0, 3000),),
            tfs=(),
            msps=((100, 500),),
            stratum="CT",
        ),
    )
    config = TargetedFamilyDiscoveryConfig(
        core_size=500,
        minimum_nfr_length=150,
        minimum_informative_molecules=1,
        minimum_informative_fraction=0,
    )
    observed = index_informative_windows(
        molecules,
        contig="chr1",
        locus_start=0,
        locus_end=3000,
        config=config,
    )
    expected = []
    for core_start in range(0, 3000, 500):
        core_end = min(3000, core_start + 500)
        mapped = [
            molecule
            for molecule in molecules
            if molecule.contig == "chr1"
            and sum(
                max(0, min(core_end, right) - max(core_start, left))
                for left, right in molecule.mapped_blocks
            ) / (core_end - core_start) >= 0.95
            and any(left <= core_start < right for left, right in molecule.mapped_blocks)
            and any(left <= core_end - 1 < right for left, right in molecule.mapped_blocks)
        ]
        long_msp = sum(
            any(
                right - left >= 150
                and left < core_end
                and core_start < right
                for left, right in molecule.msps
            )
            for molecule in mapped
        )
        expected.append((len(mapped), long_msp))
    assert [
        (window.fully_mapped_molecules, window.long_msp_molecules)
        for window in observed
    ] == expected


def test_discovery_cap_is_deterministic_and_source_strand_stratified():
    molecules = tuple(
        _molecule(
            index,
            source="input0" if index < 40 else "input1",
            stratum="CT" if index % 2 else "GA",
        )
        for index in range(80)
    )
    config = TargetedFamilyDiscoveryConfig(
        maximum_discovery_molecules=20,
        minimum_informative_molecules=1,
        minimum_informative_fraction=0.0,
        seed="deterministic-test",
    )
    window = index_informative_windows(
        molecules,
        contig="chr1",
        locus_start=0,
        locus_end=1000,
        config=config,
    )[0]
    forward = select_discovery_molecules(molecules, window, config=config)
    reverse = select_discovery_molecules(tuple(reversed(molecules)), window, config=config)
    assert [value.molecule_id for value in forward] == [
        value.molecule_id for value in reverse
    ]
    assert len(forward) == 20
    counts = {}
    for molecule in forward:
        key = (molecule.molecule_id.split("\x1f", 1)[0], molecule.stratum)
        counts[key] = counts.get(key, 0) + 1
    assert counts == {
        ("input0", "CT"): 5,
        ("input0", "GA"): 5,
        ("input1", "CT"): 5,
        ("input1", "GA"): 5,
    }


def test_parallel_and_serial_discovery_have_identical_catalogs():
    molecules = tuple(
        _molecule(
            index,
            source="input0" if index < 12 else "input1",
            stratum="CT" if index % 2 else "GA",
            tf=(210 + (index % 3), 228 + (index % 3)),
        )
        for index in range(24)
    )
    config = TargetedFamilyDiscoveryConfig(
        core_size=500,
        halo_size=200,
        minimum_informative_molecules=3,
        minimum_informative_fraction=0.0,
        maximum_discovery_molecules=12,
        minimum_family_support=3,
        seed="parallel-test",
    )
    serial = discover_targeted_families(
        molecules,
        contig="chr1",
        locus_start=0,
        locus_end=1000,
        chemistry="ddda",
        config=config,
        workers=1,
    )
    parallel = discover_targeted_families(
        tuple(reversed(molecules)),
        contig="chr1",
        locus_start=0,
        locus_end=1000,
        chemistry="ddda",
        config=config,
        workers=2,
    )
    assert serial.families == parallel.families
    assert tuple(
        (result.window.ordinal, result.selected_molecule_ids, result.families)
        for result in serial.window_results
    ) == tuple(
        (result.window.ordinal, result.selected_molecule_ids, result.families)
        for result in parallel.window_results
    )
    assert len(serial.families) == 1
    assert serial.families[0].discovery_support_molecules >= 3
    assert serial.as_dict()["contracts"]["occupancy_cohort"] == "full_unbiased_required"


def test_selection_changes_do_not_mutate_input_molecules():
    molecules = tuple(_molecule(index) for index in range(8))
    snapshot = tuple(replace(molecule) for molecule in molecules)
    config = TargetedFamilyDiscoveryConfig(
        maximum_discovery_molecules=4,
        minimum_informative_molecules=1,
        minimum_informative_fraction=0.0,
    )
    window = index_informative_windows(
        molecules,
        contig="chr1",
        locus_start=0,
        locus_end=1000,
        config=config,
    )[0]
    select_discovery_molecules(molecules, window, config=config)
    assert molecules == snapshot


def test_efficiency_exclusion_covers_full_boundary_candidate_union():
    from fiberhmm.cli.targeted_families import (
        _family_candidate_exclusion_intervals,
    )

    families = [
        {
            "family_id": "a",
            "start": 100,
            "end": 125,
            "seed_intervals": [[100, 120], [105, 125]],
        },
        {
            "family_id": "b",
            "start": 130,
            "end": 150,
            "seed_intervals": [[130, 150]],
        },
        {
            "family_id": "c",
            "start": 300,
            "end": 320,
            "seed_intervals": [],
        },
    ]
    # DddB searches +/-8 bp at both boundaries.  The first two candidate
    # envelopes overlap and therefore form one calibration exclusion block.
    assert _family_candidate_exclusion_intervals(families, "dddb") == (
        (92, 158),
        (292, 328),
    )


def test_discover_cli_writes_restartable_catalog(tmp_path, capsys):
    from fiberhmm.cli.targeted_families import main

    bam = tmp_path / "targeted.bam"
    header = {
        "HD": {"VN": "1.6", "SO": "coordinate"},
        "SQ": [{"SN": "chr1", "LN": 2000}],
    }
    with pysam.AlignmentFile(str(bam), "wb", header=header) as handle:
        for index in range(8):
            read = pysam.AlignedSegment()
            read.query_name = f"read{index}"
            read.query_sequence = "A" * 1000
            read.query_qualities = pysam.qualitystring_to_array("I" * 1000)
            read.reference_id = 0
            read.reference_start = 0
            read.mapping_quality = 60
            read.cigartuples = [(0, 1000)]
            read.set_tag(
                "MA",
                f"1000;msp.:101-300;tf.QQQ:{201 + index % 2}-18",
                value_type="Z",
            )
            modified_positions = list(range(index % 3, 1000, 10))
            skips = []
            previous_a_index = -1
            for position in modified_positions:
                skips.append(position - previous_a_index - 1)
                previous_a_index = position
            read.set_tag("MM", "A+a," + ",".join(map(str, skips)) + ";")
            read.set_tag("ML", array.array("B", [240] * len(modified_positions)))
            handle.write(read)
    pysam.index(str(bam))
    output = tmp_path / "catalog"
    assert main(
        [
            "discover",
            "-i",
            str(bam),
            "--region",
            "chr1:0-1000",
            "--chemistry",
            "hia5-pacbio",
            "--minimum-informative-fraction",
            "0",
            "--minimum-family-support",
            "3",
            "--discovery-reads",
            "6",
            "-o",
            str(output),
        ]
    ) == 0
    catalog = json.loads((output / "catalog.json").read_text())
    assert catalog["schema"] == "fiberhmm.targeted_family_discovery.v1"
    assert catalog["contracts"]["discovery_cohort_role"] == "geometry_only_msp_enriched"
    assert catalog["contracts"]["occupancy_cohort"] == "full_unbiased_required"
    assert len(catalog["families"]) == 1
    assert (output / "families.tsv").is_file()
    assert (output / "windows.tsv").is_file()
    assert (output / "discovery_molecules.tsv").is_file()
    assert (output / "independent_molecules.tsv").is_file()
    quantified = tmp_path / "quantified"
    assert main(
        [
            "quantify",
            "--catalog",
            str(output),
            "-i",
            str(bam),
            "--tf-layer",
            "tf",
            "--nuc-layer",
            "nuc",
            "--chunk-size",
            "3",
            "--minimum-assignment-posterior",
            "0",
            "--minimum-assignment-log-bayes-factor",
            "-100",
            "-o",
            str(quantified),
        ]
    ) == 0
    manifest = json.loads((quantified / "manifest.json").read_text())
    assert manifest["status"] == "complete_unbiased_independent_family_screen"
    assert manifest["full_cohort_molecules"] == 8
    assert manifest["contracts"]["discovery_mixture_weights_used_for_quantification"] is False
    assert (quantified / "family_scores.tsv").is_file()
    assert (quantified / "molecule_family_scores.tsv").is_file()
    with pytest.raises(SystemExit):
        main(
            [
                "quantify",
                "--catalog",
                str(output),
                "-i",
                str(bam),
                "--tf-layer",
                "tf",
                "--nuc-layer",
                "nuc",
                "--min-mapq",
                "61",
                "--reuse-models-from",
                str(quantified),
                "-o",
                str(tmp_path / "incompatible_rescore"),
            ]
        )
    assert not (tmp_path / "incompatible_rescore").exists()
    rescored = tmp_path / "rescored"
    assert main(
        [
            "quantify",
            "--catalog",
            str(output),
            "-i",
            str(bam),
            "--tf-layer",
            "tf",
            "--nuc-layer",
            "nuc",
            "--chunk-size",
            "3",
            "--minimum-assignment-posterior",
            "0",
            "--minimum-assignment-log-bayes-factor",
            "-100",
            "--reuse-models-from",
            str(quantified),
            "-o",
            str(rescored),
        ]
    ) == 0
    rescored_manifest = json.loads((rescored / "manifest.json").read_text())
    assert rescored_manifest["timing_seconds"]["parallel_family_fit"] == 0
    assert rescored_manifest["reused_model_provenance"]["source_directory"] == str(
        quantified.resolve()
    )
    assert (rescored / "family_scores.tsv").read_text() == (
        quantified / "family_scores.tsv"
    ).read_text()
    assert (rescored / "molecule_family_scores.tsv").read_text() == (
        quantified / "molecule_family_scores.tsv"
    ).read_text()

    bed = tmp_path / "targets.bed"
    bed.write_text(
        "chr1\t180\t240\tpeak_a\n"
        "chr1\t700\t750\tpeak_b\n"
        "chr1\t1500\t1600\tpeak_without_reads\n"
    )
    batch = tmp_path / "batch"
    batch_values = [
        "batch",
        "--bed",
        str(bed),
        "-i",
        str(bam),
        "--chemistry",
        "hia5-pacbio",
        "--site-padding",
        "100",
        "--merge-gap",
        "0",
        "--max-work-unit-bp",
        "1000",
        "--window-size",
        "200",
        "--minimum-informative-fraction",
        "0",
        "--minimum-family-support",
        "3",
        "--discovery-reads",
        "6",
        "--tf-layer",
        "tf",
        "--nuc-layer",
        "nuc",
        "--minimum-assignment-posterior",
        "0",
        "--minimum-assignment-log-bayes-factor",
        "-100",
        "-o",
        str(batch),
    ]
    assert main(batch_values) == 0
    batch_manifest = json.loads((batch / "manifest.json").read_text())
    assert batch_manifest["status"] == "complete"
    assert batch_manifest["total_units"] == 3
    assert batch_manifest["aggregate_counts"]["families"] >= 1
    assert (batch / "targets.tsv").is_file()
    assert (batch / "families.tsv").is_file()
    assert (batch / "family_scores.tsv").is_file()
    assert (batch / "molecule_family_scores.tsv").is_file()
    assert (batch / "composite_states.tsv").is_file()
    assert (batch / "composite_states.json").is_file()
    assert (batch / "oriented_target_families.tsv").is_file()
    assert (batch / "efficiency_exclusion_intervals.bed").is_file()
    composite_analysis = json.loads((batch / "composite_states.json").read_text())
    assert composite_analysis["contracts"]["parent_family_role"] == (
        "stable_primary_call_preserved"
    )
    oriented_rows = list(
        csv.DictReader(
            (batch / "oriented_target_families.tsv").open(), delimiter="\t"
        )
    )
    assert all(row["parent_calls_changed"] == "False" for row in oriented_rows)
    batch_scores = list(
        csv.DictReader((batch / "family_scores.tsv").open(), delimiter="\t")
    )
    assert {row["score_status"] for row in batch_scores} <= {
        "scored",
        "unscorable",
    }
    quantification_dirs = list((batch / "units").glob("*/quantification"))
    assert quantification_dirs
    assert all((path / "models.jsonl").is_file() for path in quantification_dirs)
    assert all((path / "scores.jsonl").is_file() for path in quantification_dirs)
    assert all((path / "training_molecules.tsv").is_file() for path in quantification_dirs)
    assert all(not (path / "models").exists() for path in quantification_dirs)
    quantification_manifest = json.loads(
        (quantification_dirs[0] / "manifest.json").read_text()
    )
    assert quantification_manifest["model_fit_config"][
        "efficiency_evidence_scope"
    ] == "full-alignment"
    assert quantification_manifest["efficiency_calibration"][
        "calibration_opportunities_before_local_projection"
    ] > quantification_manifest["efficiency_calibration"][
        "scoring_opportunities_after_local_projection"
    ]
    unit_states = [
        json.loads(path.read_text())
        for path in (batch / "units").glob("*/unit.json")
    ]
    assert any(
        state["status"] == "complete_no_eligible_molecules"
        for state in unit_states
    )
    scored_state = next(
        state for state in unit_states if state.get("quantification_directory")
    )
    compact_rescore = tmp_path / "compact_rescore"
    assert main(
        [
            "quantify",
            "--catalog",
            scored_state["targeted_catalog"],
            "-i",
            str(bam),
            "--tf-layer",
            "tf",
            "--nuc-layer",
            "nuc",
            "--efficiency-exclusion-bed",
            str(batch / "efficiency_exclusion_intervals.bed"),
            "--efficiency-exclusion-padding",
            "0",
            "--efficiency-evidence-scope",
            "full-alignment",
            "--minimum-assignment-posterior",
            "0",
            "--minimum-assignment-log-bayes-factor",
            "-100",
            "--reuse-models-from",
            scored_state["quantification_directory"],
            "--compact-artifacts",
            "-o",
            str(compact_rescore),
        ]
    ) == 0
    assert json.loads((compact_rescore / "manifest.json").read_text())[
        "reused_model_provenance"
    ]["model_artifact_format"] == "jsonl"
    assert (compact_rescore / "family_scores.tsv").read_text() == (
        Path(scored_state["quantification_directory"]) / "family_scores.tsv"
    ).read_text()

    # A completed unit whose frozen calibration-mask provenance no longer
    # matches the batch-wide mask must be reopened and written to a retry
    # directory rather than mixed into the aggregate.
    original_quantification = Path(scored_state["quantification_directory"])
    damaged_manifest_path = original_quantification / "manifest.json"
    damaged_manifest = json.loads(damaged_manifest_path.read_text())
    damaged_manifest["model_fit_config"][
        "efficiency_exclusion_bed_sha256"
    ] = "intentionally-invalid-test-digest"
    damaged_manifest_path.write_text(
        json.dumps(damaged_manifest, indent=2, sort_keys=True) + "\n"
    )
    capsys.readouterr()
    assert main(
        [
            *batch_values,
            "--resume",
            "--unit-workers",
            "1",
            "-c",
            "2",
        ]
    ) == 0
    resumed_output = capsys.readouterr()
    assert json.loads(resumed_output.out)["status"] == "complete"
    resumed_states = [
        json.loads(path.read_text())
        for path in (batch / "units").glob("*/unit.json")
    ]
    resumed_scored_state = next(
        state for state in resumed_states if state.get("quantification_directory")
    )
    assert ".retry_" in resumed_scored_state["quantification_directory"]
    assert json.loads(
        (
            Path(resumed_scored_state["quantification_directory"])
            / "manifest.json"
        ).read_text()
    )["model_fit_config"]["efficiency_exclusion_bed_sha256"] == json.loads(
        (batch / "manifest.json").read_text()
    )["efficiency_exclusion"]["sha256"]

    # Aggregate materialization and scheduling controls are safe to change on
    # resume because neither is part of the scientific run signature.
    assert main(
        [
            *batch_values,
            "--resume",
            "--unit-workers",
            "1",
            "--no-aggregate-molecule-assignments",
        ]
    ) == 0
    assert not (batch / "molecule_family_scores.tsv").exists()
    assert main([*batch_values, "--resume", "--unit-workers", "1"]) == 0
    assert (batch / "molecule_family_scores.tsv").is_file()


def _evidence(index, *, duplicate_name=None, position_step=2):
    from fiberhmm.inference.strand_rescue import ReadEvidence

    positions = np.arange(0, 100, position_step, dtype=np.int64)
    protected = (positions >= 40) & (positions < 56)
    steps = np.where(protected, 1.2 if index % 3 else 0.5, -0.2).astype(float)
    return ReadEvidence(
        name=duplicate_name or f"read{index:03d}",
        strand="CT" if index % 2 else "GA",
        ref_start=0,
        ref_end=100,
        positions=positions,
        steps=steps,
        hits=protected.copy(),
        contexts=np.zeros(len(positions), dtype=np.int64),
        tfs=[],
        nucs=[],
        msps=[],
        library_id="library",
        alignment_blocks=((0, 100),),
    )


def test_bed_work_unit_planning_merges_without_duplicate_edges(tmp_path):
    from fiberhmm.inference.targeted_family_batch import (
        family_target_memberships,
        load_bed_targets,
        plan_targeted_family_work_units,
    )

    bed = tmp_path / "sites.bed"
    bed.write_text(
        "chr1\t1000\t1100\tleft\t42.5\t-\n"
        "chr1\t1450\t1500\tright\n"
        "chr1\t10000\t10100\tdistant\n"
        "chr2\t10\t30\tedge\n"
    )
    lengths = {"chr1": 20000, "chr2": 1000}
    targets = load_bed_targets(bed, lengths)
    assert targets[0].score == "42.5"
    assert targets[0].strand == "-"
    assert targets[1].score == "."
    assert targets[1].strand == "."
    units = plan_targeted_family_work_units(
        targets,
        lengths,
        padding=200,
        merge_gap=100,
        maximum_work_unit_bp=2000,
    )
    assert [(unit.contig, unit.start, unit.end) for unit in units] == [
        ("chr1", 800, 1700),
        ("chr1", 9800, 10300),
        ("chr2", 0, 230),
    ]
    assert units[0].target_ordinals == (0, 1)
    assert all(not unit.oversized_connected_component for unit in units)
    matched = family_target_memberships(
        {"start": 1280, "end": 1300},
        units[0],
        targets,
        lengths,
        padding=200,
    )
    assert [target.name for target in matched] == ["left", "right"]


def test_bed_work_unit_planning_keeps_oversized_overlap_component_together(tmp_path):
    from fiberhmm.inference.targeted_family_batch import (
        load_bed_targets,
        plan_targeted_family_work_units,
    )

    bed = tmp_path / "connected.bed"
    bed.write_text("chr1\t100\t200\ta\nchr1\t900\t1000\tb\n")
    targets = load_bed_targets(bed, {"chr1": 5000})
    units = plan_targeted_family_work_units(
        targets,
        {"chr1": 5000},
        padding=500,
        merge_gap=0,
        maximum_work_unit_bp=700,
    )
    assert len(units) == 1
    assert units[0].oversized_connected_component is True


def test_torch_spatial_null_batch_matches_numpy_reference_on_cpu():
    torch = pytest.importorskip("torch")
    del torch
    from fiberhmm.inference.cuda_likelihood import (
        prepare_torch_spatial_null_batch,
    )
    from fiberhmm.inference.strand_rescue import (
        _spatial_null_log_likelihoods_from_prepared_grid,
    )

    reads = [_evidence(index, position_step=1 + index % 3) for index in range(7)]
    starts = np.asarray([0, 7, 19, 37, 40, 52, 76], dtype=np.int64)
    ends = np.asarray([9, 18, 31, 49, 56, 70, 99], dtype=np.int64)
    log_prior = np.log(
        np.asarray([0.08, 0.12, 0.14, 0.18, 0.2, 0.16, 0.12], dtype=np.float64)
    )
    expected = _spatial_null_log_likelihoods_from_prepared_grid(
        list(reversed(reads)), starts, ends, log_prior
    )
    resident = prepare_torch_spatial_null_batch(reads, device="cpu")
    observed = resident.spatial_null_log_likelihoods(
        list(reversed(reads)),
        starts,
        ends,
        log_prior,
        interval_chunk_size=3,
    )
    np.testing.assert_allclose(observed, expected, rtol=1e-14, atol=1e-12)


def test_torch_resident_batch_rejects_duplicate_opportunity_positions():
    pytest.importorskip("torch")
    from fiberhmm.inference.cuda_likelihood import (
        prepare_torch_spatial_null_batch,
    )

    read = _evidence(0)
    read.positions[2] = read.positions[1]
    with pytest.raises(ValueError, match="strictly increasing"):
        prepare_torch_spatial_null_batch([read], device="cpu")


def test_cuda_chunk_planner_accounts_for_quadratic_diffuse_workspace(monkeypatch):
    torch = pytest.importorskip("torch")
    from fiberhmm.inference.cuda_likelihood import (
        recommend_cuda_read_chunk_size,
    )

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (1_000_000_000, 2_000_000_000))
    reads = [_evidence(index) for index in range(4)]
    narrow_rows, narrow = recommend_cuda_read_chunk_size(
        reads,
        interval_chunk_size=256,
        maximum_envelope_width=100,
        minimum=1,
    )
    wide_rows, wide = recommend_cuda_read_chunk_size(
        reads,
        interval_chunk_size=256,
        maximum_envelope_width=1000,
        minimum=1,
    )
    assert wide["estimated_diffuse_bytes_per_row"] > 90 * narrow[
        "estimated_diffuse_bytes_per_row"
    ]
    assert wide_rows < narrow_rows


def test_likelihood_backend_resolution_is_explicit(monkeypatch):
    from fiberhmm.inference import cuda_likelihood

    monkeypatch.setattr(
        cuda_likelihood,
        "cuda_runtime_status",
        lambda: {"available": False, "reason": "test device is hidden"},
    )
    assert cuda_likelihood.resolve_likelihood_backend("cpu")[0] == "cpu"
    resolved, status = cuda_likelihood.resolve_likelihood_backend("auto")
    assert resolved == "cpu"
    assert status["reason"] == "test device is hidden"
    with pytest.raises(ValueError, match="requested but unavailable"):
        cuda_likelihood.resolve_likelihood_backend("cuda")
    with pytest.raises(ValueError, match="auto.*cpu.*cuda"):
        cuda_likelihood.resolve_likelihood_backend("gpu")


def test_joint_torch_backend_matches_cpu_via_cpu_device(monkeypatch):
    pytest.importorskip("torch")
    from fiberhmm.inference import cuda_likelihood
    from fiberhmm.inference.strand_rescue import (
        fit_boundary_marginalized_tf_family_model,
    )

    reads = [
        _evidence(index, position_step=1 + index % 3)
        for index in range(19)
    ]
    for index in range(0, len(reads), 5):
        # Exercise exact complex-block eligibility as well as ragged resident
        # opportunity padding in the joint backend.
        reads[index].alignment_blocks = ((0, 45), (46, 100))
    models = (
        fit_boundary_marginalized_tf_family_model(
            reads[:12],
            "family_cuda_a",
            [(40, 56), (41, 57)],
            boundary_search_radius=2,
            minimum_molecule_opportunities=2,
        ),
        fit_boundary_marginalized_tf_family_model(
            reads[:12],
            "family_cuda_b",
            [(45, 62), (46, 63)],
            boundary_search_radius=2,
            minimum_molecule_opportunities=2,
        ),
    )
    cpu = score_boundary_families_on_unbiased_cohort(
        reads,
        models,
        chunk_size=6,
        workers=2,
        minimum_assignment_standardized_posterior=0.0,
        minimum_assignment_log_bayes_factor=-100.0,
        likelihood_backend="cpu",
    )
    real_prepare = cuda_likelihood.prepare_torch_spatial_null_batch
    real_prepare_family_grids = (
        cuda_likelihood.prepare_torch_family_likelihood_grids
    )
    monkeypatch.setattr(
        cuda_likelihood,
        "cuda_runtime_status",
        lambda: {
            "available": True,
            "device_name": "torch CPU equivalence oracle",
        },
    )
    monkeypatch.setattr(
        cuda_likelihood,
        "prepare_torch_spatial_null_batch",
        lambda current_reads: real_prepare(current_reads, device="cpu"),
    )
    monkeypatch.setattr(
        cuda_likelihood,
        "prepare_torch_family_likelihood_grids",
        lambda current_models: real_prepare_family_grids(
            current_models, device="cpu"
        ),
    )
    accelerated = score_boundary_families_on_unbiased_cohort(
        list(reversed(reads)),
        models,
        chunk_size=5,
        workers=2,
        minimum_assignment_standardized_posterior=0.0,
        minimum_assignment_log_bayes_factor=-100.0,
        likelihood_backend="cuda",
        cuda_interval_chunk_size=7,
        cuda_read_chunk_size=5,
        cuda_family_batch_span_bp=1,
    )
    for expected, observed in zip(cpu, accelerated):
        assert observed["likelihood_backend"]["resolved"] == "cuda"
        # CUDA owns the likelihood stream in the parent; requested CPU workers
        # remain available to loading/fitting but are not spawned for the
        # lightweight post-device contribution assembly.
        assert observed["likelihood_backend"]["requested_workers"] == 2
        assert observed["likelihood_backend"]["scoring_workers"] == 1
        assert observed["likelihood_backend"]["cuda_batch_plan"][
            "family_locality_batches"
        ] == 2
        assert observed["eligible_molecules"] == expected["eligible_molecules"]
        assert observed["fitted_family_occupancy"] == pytest.approx(
            expected["fitted_family_occupancy"], abs=1e-13
        )
        assert observed["mixture_log_likelihood"] == pytest.approx(
            expected["mixture_log_likelihood"], abs=1e-11
        )
        for expected_molecule, observed_molecule in zip(
            expected["molecules"], observed["molecules"]
        ):
            assert observed_molecule["molecule_id"] == expected_molecule["molecule_id"]
            assert observed_molecule[
                "family_vs_null_log_bayes_factor"
            ] == pytest.approx(
                expected_molecule["family_vs_null_log_bayes_factor"], abs=1e-12
            )


def test_training_window_index_matches_exhaustive_family_selection():
    from fiberhmm.cli.targeted_families import (
        _index_family_training_reads,
        _select_family_training_reads,
    )
    from fiberhmm.inference.strand_rescue import IntervalCall

    reads = [_evidence(index) for index in range(8)]
    for index, read in enumerate(reads):
        read.msps = [
            IntervalCall(90, 310) if index < 6 else IntervalCall(1100, 1300)
        ]
    windows = [
        {"ordinal": 0, "core_start": 0, "core_end": 1000},
        {"ordinal": 1, "core_start": 1000, "core_end": 2000},
    ]
    family = {
        "family_id": "indexed",
        "source_window_ordinals": [0],
        "discovery_minimum_nfr_length": 150,
    }
    exhaustive, exhaustive_available = _select_family_training_reads(
        reads, family, windows, maximum=20, seed="index-test"
    )
    indexed = _index_family_training_reads(
        reads, windows, minimum_nfr_length=150
    )
    selected, available = _select_family_training_reads(
        reads,
        family,
        windows,
        maximum=20,
        seed="index-test",
        eligible_reads=indexed[0],
    )
    assert available == exhaustive_available == 6
    assert [read.molecule_id for read in selected] == [
        read.molecule_id for read in exhaustive
    ]


def test_unbiased_boundary_scorer_is_chunk_and_worker_invariant():
    from fiberhmm.inference.strand_rescue import (
        fit_boundary_marginalized_tf_family_model,
    )

    reads = [_evidence(index) for index in range(17)]
    # The duplicate with more opportunities must win globally even when the
    # two representations fall in different chunks.
    reads.extend(
        [
            _evidence(30, duplicate_name="duplicate", position_step=4),
            _evidence(32, duplicate_name="duplicate", position_step=1),
        ]
    )
    model = fit_boundary_marginalized_tf_family_model(
        reads[:12],
        "family_test",
        [(40, 56), (41, 57)],
        boundary_search_radius=2,
        minimum_molecule_opportunities=2,
    )
    snapshots = [(read.positions.copy(), read.steps.copy()) for read in reads]
    serial = score_boundary_family_on_unbiased_cohort(
        reads,
        model,
        chunk_size=5,
        workers=1,
        minimum_assignment_posterior=0.0,
        minimum_assignment_log_bayes_factor=-100.0,
    )
    parallel = score_boundary_family_on_unbiased_cohort(
        list(reversed(reads)),
        model,
        chunk_size=3,
        workers=2,
        minimum_assignment_posterior=0.0,
        minimum_assignment_log_bayes_factor=-100.0,
    )
    assert serial["eligible_molecules"] == parallel["eligible_molecules"] == 18
    assert serial["eligible_molecules_by_strand"] == parallel["eligible_molecules_by_strand"]
    assert serial["total_opportunities"] == parallel["total_opportunities"]
    assert serial["fitted_family_occupancy"] == pytest.approx(
        parallel["fitted_family_occupancy"], abs=1e-14
    )
    assert serial["median_family_vs_null_log_bayes_factor"] == pytest.approx(
        parallel["median_family_vs_null_log_bayes_factor"], abs=1e-14
    )
    assert serial["molecules"] == parallel["molecules"]
    assert serial["predictive_weight_contract"]["discovery_mixture_weights_used"] is False
    duplicate = [
        record for record in serial["molecules"] if record["molecule_id"][1] == "duplicate"
    ]
    assert len(duplicate) == 1
    assert duplicate[0]["opportunities"] > 32
    for read, (positions, steps) in zip(reads, snapshots):
        np.testing.assert_array_equal(read.positions, positions)
        np.testing.assert_array_equal(read.steps, steps)


def test_composite_footprint_state_preserves_parent_and_variable_components():
    from fiberhmm.inference.composite_footprint_states import (
        nominate_composite_footprint_states,
    )

    def family(family_id, slot, start, end, support):
        return {
            "batch_family_id": f"unit:{family_id}",
            "family_id": family_id,
            "family_slot": slot,
            "contig": "chr1",
            "start": start,
            "end": end,
            "working_support_molecules": support,
        }

    families = [
        family("envelope", 3, 100, 170, 9),
        family("left", 1, 100, 123, 5),
        family("right", 2, 121, 170, 4),
    ]
    molecules = {
        "unit:envelope": {"rep1\x1fe1", "rep2\x1fe2"},
        "unit:left": {"rep1\x1fl1", "rep2\x1fl2"},
        "unit:right": {"rep1\x1fr1", "rep2\x1fr2"},
    }
    by_dataset = {
        family_id: {
            "rep1": {value for value in values if value.startswith("rep1")},
            "rep2": {value for value in values if value.startswith("rep2")},
        }
        for family_id, values in molecules.items()
    }

    result = nominate_composite_footprint_states(
        families,
        family_molecules=molecules,
        family_molecules_by_dataset=by_dataset,
    )
    nomination = result["nominations"][0]
    assert nomination["envelope_family"]["family_id"] == "unit:envelope"
    assert [
        value["family_id"] for value in nomination["component_families"]
    ] == ["unit:left", "unit:right"]
    assert nomination["geometry"]["union_coverage_fraction"] == 1.0
    assert nomination["support"]["datasets_with_all_states"] == ["rep1", "rep2"]
    assert "dimerization" in nomination["claim_boundary"]
    assert result["contracts"]["parent_family_role"] == (
        "stable_primary_call_preserved"
    )


def test_joint_family_scorer_matches_independent_scores_and_worker_counts():
    from fiberhmm.inference.strand_rescue import (
        fit_boundary_marginalized_tf_family_model,
    )

    reads = [_evidence(index) for index in range(24)]
    models = (
        fit_boundary_marginalized_tf_family_model(
            reads[:12],
            "family_a",
            [(40, 56), (41, 57)],
            boundary_search_radius=2,
            minimum_molecule_opportunities=2,
        ),
        fit_boundary_marginalized_tf_family_model(
            reads[:12],
            "family_b",
            [(45, 62), (46, 63)],
            boundary_search_radius=2,
            minimum_molecule_opportunities=2,
        ),
    )
    independent = tuple(
        score_boundary_family_on_unbiased_cohort(
            reads,
            model,
            chunk_size=7,
            workers=1,
            minimum_assignment_standardized_posterior=0.0,
            minimum_assignment_log_bayes_factor=-100.0,
            include_conditional_geometry=False,
        )
        for model in models
    )
    joint_serial = score_boundary_families_on_unbiased_cohort(
        list(reversed(reads)),
        models,
        chunk_size=5,
        workers=1,
        minimum_assignment_standardized_posterior=0.0,
        minimum_assignment_log_bayes_factor=-100.0,
    )
    joint_parallel = score_boundary_families_on_unbiased_cohort(
        reads,
        models,
        chunk_size=4,
        workers=2,
        minimum_assignment_standardized_posterior=0.0,
        minimum_assignment_log_bayes_factor=-100.0,
    )
    for expected, serial, parallel in zip(
        independent, joint_serial, joint_parallel
    ):
        assert serial["eligible_molecules"] == expected["eligible_molecules"]
        assert serial["fitted_family_occupancy"] == pytest.approx(
            expected["fitted_family_occupancy"], abs=1e-14
        )
        assert serial["molecules"] == expected["molecules"]
        assert parallel["fitted_family_occupancy"] == pytest.approx(
            expected["fitted_family_occupancy"], abs=1e-14
        )
        assert parallel["molecules"] == expected["molecules"]
        assert serial["assignment_selection"]["posterior"] == (
            "standardized_equal_prior_family_vs_null"
        )

    with_duplicates = reads + [
        _evidence(30, duplicate_name="duplicate", position_step=4),
        _evidence(32, duplicate_name="duplicate", position_step=1),
    ]
    deduplicated = score_boundary_families_on_unbiased_cohort(
        with_duplicates,
        models,
        chunk_size=6,
        workers=2,
        minimum_assignment_standardized_posterior=0.0,
        minimum_assignment_log_bayes_factor=-100.0,
    )
    assert deduplicated[0]["joint_cohort_deduplication"] == {
        "policy": "per_family_envelope_opportunities_then_read_evidence_sha256",
        "execution": "exact_single_family_fallback_for_duplicate_representations",
        "input_records": 26,
        "unique_independent_molecules": 25,
        "discarded_duplicate_representations": 1,
    }
    duplicate_reference = tuple(
        score_boundary_family_on_unbiased_cohort(
            with_duplicates,
            model,
            chunk_size=6,
            workers=2,
            minimum_assignment_standardized_posterior=0.0,
            minimum_assignment_log_bayes_factor=-100.0,
            include_conditional_geometry=False,
        )
        for model in models
    )
    for expected, observed in zip(duplicate_reference, deduplicated):
        assert observed["molecules"] == expected["molecules"]
        assert observed["fitted_family_occupancy"] == pytest.approx(
            expected["fitted_family_occupancy"], abs=1e-14
        )
    duplicate = [
        record
        for record in deduplicated[0]["molecules"]
        if record["molecule_id"][1] == "duplicate"
    ]
    assert len(duplicate) == 1
    assert duplicate[0]["opportunities"] > 32

    for scorer, scorer_models in (
        (score_boundary_family_on_unbiased_cohort, models[0]),
        (score_boundary_families_on_unbiased_cohort, models),
    ):
        with pytest.raises(ValueError, match="occupancy_max_iter"):
            scorer(reads, scorer_models, occupancy_max_iter=0)
        with pytest.raises(ValueError, match="occupancy_tolerance"):
            scorer(reads, scorer_models, occupancy_tolerance=float("nan"))
        with pytest.raises(ValueError, match="log_bayes_factor"):
            scorer(
                reads,
                scorer_models,
                minimum_assignment_log_bayes_factor=float("nan"),
            )
