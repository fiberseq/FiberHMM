"""Tests for bounded FiberHMM QC."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pysam
import pytest

from fiberhmm.cli.qc import _input_paths, parse_args, resolve_output_dir
from fiberhmm.qc.core import (
    _curve_comparison,
    _periodicity_score,
    _range_score,
    analyze_sample,
    format_terminal,
    load_control_curves,
    load_control_examples,
    load_references,
    pair_distance_histogram,
    phasogram_metrics,
    reference_profile_for_assay,
    run_multi_qc,
    run_qc,
    sample_bam_reads,
)


class FakeRead:
    def __init__(self, sequence: str, ma: str):
        self.query_sequence = sequence
        self.is_reverse = False
        self.tags = {"MA": ma, "st": "CT"}

    def has_tag(self, tag):
        return tag in self.tags

    def get_tag(self, tag):
        if tag not in self.tags:
            raise KeyError(tag)
        return self.tags[tag]


def _write_unindexed_iupac_bam(path: Path, n_reads: int = 100) -> None:
    header = pysam.AlignmentHeader.from_dict(
        {
            "HD": {"VN": "1.6", "SO": "unsorted"},
            "SQ": [{"SN": "chr1", "LN": 1_000_000}],
            "PG": [
                {
                    "ID": "fiberhmm-call",
                    "PN": "fiberhmm-call",
                    "DS": "mode=daf enzyme=dddb coord=molecular",
                }
            ],
        }
    )
    with pysam.AlignmentFile(path, "wb", header=header) as bam:
        for index in range(n_reads):
            sequence = list("CG" * 750)
            for position in range(95, len(sequence), 190):
                sequence[position] = "Y"
            read = pysam.AlignedSegment(header)
            read.query_name = f"read_{index:04d}"
            read.query_sequence = "".join(sequence)
            read.flag = 0
            read.reference_id = 0
            read.reference_start = (index * 1700) % 900_000
            read.mapping_quality = 60
            read.cigar = [(0, len(sequence))]
            read.set_tag("st", "CT", value_type="Z")
            read.set_tag(
                "MA",
                f"{len(sequence)};nuc.Q:1-147,201-150;tf.QQQ:500-24,800-41",
                value_type="Z",
            )
            bam.write(read)


def test_periodicity_recovers_190_bp_repeat():
    positions = [np.arange(20, 1920, 190) for _ in range(100)]
    histogram = pair_distance_histogram(positions)
    _curve, nrl, strength = phasogram_metrics(histogram)
    assert nrl == 190
    assert strength > 0.2


def test_rate_score_uses_drawn_iqr_and_fifth_to_ninety_fifth_bands():
    reference = load_references()["profiles"]["dddb"]["rate"]
    q05, q25, median, q75, q95 = reference["reference_quantiles"]
    assert _range_score(median, reference) == 100
    assert _range_score(q25, reference) >= 70
    assert _range_score(q75, reference) >= 70
    assert 35 <= _range_score((q05 + q25) / 2, reference) < 70
    assert 35 <= _range_score((q75 + q95) / 2, reference) < 70
    assert _range_score(q05 * 0.99, reference) < 35
    assert _range_score(q95 * 1.01, reference) < 35


def test_reference_pattern_amplitude_prevents_flat_false_pass():
    controls = load_control_curves()["profiles"]["dddb"]
    periodicity_reference = load_references()["profiles"]["dddb"]["periodicity"]
    lags = np.asarray(controls["phasogram"]["lags_bp"], dtype=int)
    values = np.asarray(
        controls["phasogram"]["detrended_pair_frequency"], dtype=float
    )
    matched = np.zeros(1001, dtype=float)
    matched[lags] = values
    flat = 0.40 * matched

    matched_r, matched_amplitude = _curve_comparison(matched, controls)
    flat_r, flat_amplitude = _curve_comparison(flat, controls)
    assert matched_r == pytest.approx(1.0)
    assert matched_amplitude == pytest.approx(1.0)
    assert flat_r == pytest.approx(1.0)
    assert flat_amplitude == pytest.approx(0.40)
    assert _periodicity_score(
        194,
        periodicity_reference["reference_strength"],
        periodicity_reference,
        matched_r,
        matched_amplitude,
    ) >= 70
    assert _periodicity_score(
        194,
        periodicity_reference["reference_strength"],
        periodicity_reference,
        flat_r,
        flat_amplitude,
    ) < 35


@pytest.mark.parametrize(
    ("mode", "enzyme", "expected"),
    [
        ("daf", "dddb", "dddb"),
        ("daf", "ddda", "ddda"),
        ("pacbio-fiber", "hia5", "hia5_pacbio"),
        ("nanopore-fiber", "hia5", "hia5_nanopore"),
    ],
)
def test_reference_profile_is_locked_to_assay(mode, enzyme, expected):
    assert reference_profile_for_assay(mode, enzyme) == expected


def test_reference_profile_rejects_incompatible_assay():
    with pytest.raises(ValueError, match="incompatible QC assay"):
        reference_profile_for_assay("daf", "hia5")


def test_packaged_controls_are_aggregate_curves_only():
    controls = load_control_curves()
    assert controls["contains_individual_read_data"] is False
    assert set(controls["profiles"]) == {
        "ddda",
        "dddb",
        "hia5_nanopore",
        "hia5_pacbio",
    }
    for profile in controls["profiles"].values():
        assert len(profile["rate_ecdf"]["probabilities"]) == 99
        assert len(profile["rate_ecdf"]["rates"]) == 99
        assert len(profile["phasogram"]["lags_bp"]) == 741
        assert len(profile["phasogram"]["detrended_pair_frequency"]) == 741
        assert "footprint_sizes" in profile
        assert profile["footprint_sizes"]["nucleosome"] is not None
        nuc = profile["footprint_sizes"]["nucleosome"]
        assert len(nuc["bin_edges_bp"]) == len(nuc["fraction_per_bin"]) + 1
        assert nuc["n_calls_total"] > 0
        encoded = str(profile).lower()
        assert "query_sequence" not in encoded
        assert "read_id" not in encoded


def test_packaged_control_examples_are_anonymous_visual_hatchmarks():
    examples = load_control_examples()
    assert examples["contains_bams"] is False
    assert examples["contains_sequences"] is False
    assert examples["contains_read_identifiers"] is False
    assert examples["contains_genomic_coordinates"] is False
    assert set(examples["profiles"]) == {
        "ddda",
        "dddb",
        "hia5_nanopore",
        "hia5_pacbio",
    }
    for profile in examples["profiles"].values():
        assert profile["groups"]
        for group in profile["groups"]:
            assert group["reads"]
            for read in group["reads"]:
                assert set(read) == {
                    "signal_positions_bp",
                    "signal_rate",
                    "span_bp",
                }
                assert all(0 <= value <= read["span_bp"] for value in read["signal_positions_bp"])


def test_analyze_sample_reports_upstream_dedup_tags_and_flags():
    reads = [FakeRead("CG" * 750, "1500;nuc.Q:1-147") for _ in range(4)]
    for index, read in enumerate(reads):
        read.tags.update({"di": 7, "ds": 4})
        read.is_duplicate = index > 0
    result, _arrays = analyze_sample(
        reads,
        mode="daf",
        reference_profile=None,
        min_opportunities=100,
    )
    assert result["deduplication"]["detected"] is True
    assert result["deduplication"]["mode"] == "flagged"
    assert result["deduplication"]["duplicate_fraction"] == pytest.approx(0.75)


def test_analyze_sample_uses_ma_nuc_and_tf_lengths():
    sequence = list("CG" * 1000)
    for position in range(20, 1920, 190):
        sequence[position] = "Y"
    reads = [
        FakeRead(
            "".join(sequence),
            "2000;nuc.Q:1-147,201-150;tf.QQQ:500-24,800-41",
        )
        for _ in range(100)
    ]
    result, _arrays = analyze_sample(
        reads,
        mode="daf",
        reference_profile=None,
        min_opportunities=100,
    )
    assert result["signal"]["n_rate_reads"] == 100
    assert result["footprints"]["n_nucleosomes"] == 200
    assert result["footprints"]["median_nucleosome_bp"] == 148.5
    assert result["footprints"]["fraction_nucleosome_85_250_bp"] == 1.0
    assert result["footprints"]["fraction_nucleosome_over_300_bp"] == 0.0
    assert result["footprints"]["fraction_nucleosome_over_1000_bp"] == 0.0
    assert result["footprints"]["n_tf_footprints"] == 200
    assert result["footprints"]["median_tf_footprint_bp"] == 32.5


def test_unindexed_sampler_is_bounded_and_deterministic(tmp_path):
    bam_path = tmp_path / "unindexed.bam"
    _write_unindexed_iupac_bam(bam_path)
    first = sample_bam_reads(str(bam_path), sample_reads=5, seed=7, scan_limit=30)
    second = sample_bam_reads(str(bam_path), sample_reads=5, seed=7, scan_limit=30)
    assert first.records_examined == 30
    assert len(first.reads) == 5
    assert [read.query_name for read in first.reads] == [
        read.query_name for read in second.reads
    ]
    assert "bounded reservoir" in first.strategy


def test_run_qc_writes_reports_without_whole_bam_scan(tmp_path):
    bam_path = tmp_path / "input.bam"
    prefix = tmp_path / "sample"
    _write_unindexed_iupac_bam(bam_path, n_reads=40)
    result = run_qc(
        str(bam_path),
        output_prefix=str(prefix),
        mode="daf",
        enzyme="dddb",
        reference_profile="none",
        sample_reads=10,
        stream=None,
    )
    assert result["sampling"]["sampled_reads"] == 10
    assert result["sampling"]["whole_bam_scanned"] is False
    assert result["assay"]["packaged_control_curve"] is False
    assert (tmp_path / "sample.qc.json").exists()
    assert (tmp_path / "sample.qc.tsv").exists()
    assert result["outputs"]["pdf"] == str((tmp_path / "sample.qc.pdf").resolve())
    pdf_bytes = (tmp_path / "sample.qc.pdf").read_bytes()
    assert b"/FontFile2" in pdf_bytes  # TrueType text remains editable in Illustrator.


def test_run_qc_reports_automatic_low_coverage_snp_skip(tmp_path):
    bam_path = tmp_path / "low_coverage.bam"
    prefix = tmp_path / "low_coverage"
    _write_unindexed_iupac_bam(bam_path, n_reads=20)
    preflight = {
        "run": False,
        "reason": "insufficient_depth",
        "estimated_genome_coverage": 0.04,
        "max_local_depth": 7,
        "max_alignment_start_bin_reads": 7,
        "supported_start_bin_fraction": 0.02,
        "minimum_depth": 10,
        "records_examined": 20,
    }
    result = run_qc(
        str(bam_path),
        output_prefix=str(prefix),
        mode="daf",
        enzyme="dddb",
        reference_profile="none",
        sample_reads=10,
        snp_preflight_summary=preflight,
        stream=None,
    )
    assert result["variant_masking"]["screening_preflight"] == preflight
    assert "automatic screen skipped" in result["variant_masking"]["note"]
    assert "automatic low-coverage skip" in format_terminal(result)


def test_multi_qc_writes_individual_and_combined_reports(tmp_path):
    first = tmp_path / "first.bam"
    second = tmp_path / "second.bam"
    output_dir = tmp_path / "qc"
    _write_unindexed_iupac_bam(first, n_reads=30)
    _write_unindexed_iupac_bam(second, n_reads=30)
    payload = run_multi_qc(
        [str(first), str(second)],
        output_dir=str(output_dir),
        mode="daf",
        enzyme="dddb",
        reference_profile="none",
        sample_reads=10,
        stream=None,
    )
    assert payload["n_samples"] == 2
    assert all("_plot_arrays" not in sample for sample in payload["samples"])
    for filename in (
        "first.qc.json",
        "first.qc.png",
        "first.qc.pdf",
        "second.qc.json",
        "second.qc.png",
        "second.qc.pdf",
        "combined.qc.json",
        "combined.qc.tsv",
        "combined.qc.png",
        "combined.qc.pdf",
        "combined.qc.html",
    ):
        assert (output_dir / filename).exists()
    html_text = (output_dir / "combined.qc.html").read_text()
    assert "first.bam" in html_text
    assert "second.bam" in html_text


def test_qc_cli_accepts_multiple_inputs_and_defaults_to_shared_qc_dir(tmp_path):
    first = tmp_path / "first.bam"
    second = tmp_path / "second.bam"
    args = parse_args(["-i", str(first), str(second)])
    inputs = _input_paths(args.input)
    assert inputs == [str(first), str(second)]
    assert resolve_output_dir(inputs, None) == tmp_path / "qc"


def test_qc_cli_requires_output_dir_for_inputs_in_different_directories(tmp_path):
    inputs = [str(tmp_path / "one" / "a.bam"), str(tmp_path / "two" / "b.bam")]
    with pytest.raises(ValueError, match="require -o/--output-dir"):
        resolve_output_dir(inputs, None)
