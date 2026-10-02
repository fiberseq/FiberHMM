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
    infer_assay,
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
        ("pacbio-fiber", "ecogii", ""),
        ("nanopore-fiber", "ecogii", ""),
    ],
)
def test_reference_profile_is_locked_to_assay(mode, enzyme, expected):
    assert reference_profile_for_assay(mode, enzyme) == expected


def test_reference_profile_rejects_incompatible_assay():
    with pytest.raises(ValueError, match="incompatible QC assay"):
        reference_profile_for_assay("daf", "hia5")


def test_ecogii_qc_does_not_borrow_hia5_calibration():
    assert reference_profile_for_assay("pacbio-fiber", "ecogii") == ""
    assert reference_profile_for_assay("nanopore-fiber", "ecogii") == ""


def test_ecogii_nanopore_header_infers_descriptive_qc(tmp_path):
    path = tmp_path / "ecogii.bam"
    header = pysam.AlignmentHeader.from_dict({
        "HD": {"VN": "1.6"},
        "SQ": [{"SN": "chr1", "LN": 1000}],
        "PG": [{
            "ID": "fiberhmm-call",
            "PN": "fiberhmm-call",
            "DS": "mode=nanopore-fiber enzyme=ecogii coord=molecular",
        }],
    })
    with pysam.AlignmentFile(path, "wb", header=header):
        pass
    assert infer_assay(str(path), [], mode="auto", enzyme="auto") == (
        "nanopore-fiber", "ecogii", ""
    )


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
    pytest.importorskip("matplotlib")  # PDF report needs the optional [plots] extra
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
    pytest.importorskip("matplotlib")  # PNG/PDF reports need the optional [plots] extra
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


# ---------------------------------------------------------------------------
# MM '?' (unknown) specs: unlisted target bases are not opportunities
# ---------------------------------------------------------------------------

def _mm_read(header, sequence, *, reverse, specs, name):
    """Aligned read whose MM lists ``specs`` = [(base, strand, flag, n_listed,
    n_hits)]: the first ``n_listed`` target bases (original frame) are listed,
    the first ``n_hits`` of them with ML 255, the rest with ML 0."""
    original = (
        sequence.translate(str.maketrans("ACGT", "TGCA"))[::-1]
        if reverse else sequence
    )
    mm_parts, ml = [], []
    for base, strand, flag, n_listed, n_hits in specs:
        total = original.count(base)
        n_listed = total if n_listed is None else n_listed
        mm_parts.append(f"{base}{strand}a{flag}" + "".join([",0"] * n_listed))
        ml += [255] * n_hits + [0] * (n_listed - n_hits)
    read = pysam.AlignedSegment(header)
    read.query_name = name
    read.query_sequence = sequence
    read.flag = 16 if reverse else 0
    read.reference_id = 0
    read.reference_start = 0
    read.mapping_quality = 60
    read.cigartuples = [(0, len(sequence))]
    read.set_tag("MM", ";".join(mm_parts) + ";")
    read.set_tag("ML", ml)
    return read


def _random_sequence(length, seed):
    rng = np.random.default_rng(seed)
    return "".join(rng.choice(list("ACGT"), size=length))


@pytest.mark.parametrize("reverse", [False, True])
def test_qc_excludes_question_unlisted_bases_nanopore(reverse):
    from fiberhmm.qc.core import _signal_profile

    header = pysam.AlignmentHeader.from_dict(
        {"HD": {"VN": "1.6"}, "SQ": [{"SN": "chr1", "LN": 10_000}]})
    sequence = _random_sequence(1200, seed=11)
    listed = _mm_read(header, sequence, reverse=reverse, name="dot",
                      specs=[("A", "+", ".", None, 20)])
    unknown = _mm_read(header, sequence, reverse=reverse, name="question",
                       specs=[("A", "+", "?", 100, 20)])

    dot_positions, dot_opportunities, _, _ = _signal_profile(
        listed, "nanopore-fiber")
    positions, opportunities, _, _ = _signal_profile(unknown, "nanopore-fiber")
    # '.': every basecalled-forward A (SEQ T on a reverse read) is an
    # opportunity.
    original_a = sequence.count("T" if reverse else "A")
    assert dot_opportunities == original_a
    assert len(dot_positions) == 20
    # '?': only the 100 listed bases were observed.
    assert opportunities == 100
    np.testing.assert_array_equal(positions, dot_positions)

    result, _ = analyze_sample([unknown], "nanopore-fiber", None,
                               min_opportunities=50)
    assert result["signal"]["n_opportunities"] == 100
    assert result["signal"]["aggregate_rate"] == pytest.approx(0.2)


def test_qc_excludes_question_unlisted_bases_pacbio():
    from fiberhmm.qc.core import _signal_profile

    header = pysam.AlignmentHeader.from_dict(
        {"HD": {"VN": "1.6"}, "SQ": [{"SN": "chr1", "LN": 10_000}]})
    sequence = _random_sequence(1200, seed=12)
    read = _mm_read(header, sequence, reverse=False, name="question",
                    specs=[("A", "+", "?", 60, 10), ("T", "-", "?", 40, 5)])
    positions, opportunities, _, _ = _signal_profile(read, "pacbio-fiber")
    assert opportunities == 100
    assert len(positions) == 15
    # A fully-listed '?' read is identical to '.'.
    full = _mm_read(header, sequence, reverse=False, name="full",
                    specs=[("A", "+", "?", None, 10), ("T", "-", "?", None, 5)])
    dot = _mm_read(header, sequence, reverse=False, name="dot",
                   specs=[("A", "+", ".", None, 10), ("T", "-", ".", None, 5)])
    assert _signal_profile(full, "pacbio-fiber")[1] == \
        _signal_profile(dot, "pacbio-fiber")[1] == \
        sequence.count("A") + sequence.count("T")


@pytest.mark.parametrize("with_md", [False, True])
def test_qc_excludes_question_unlisted_bases_daf_mm(with_md):
    """DAF calls carried only in MM/ML (no R/Y): a '?' spec's unlisted bases
    are not opportunities; a '.' spec keeps the legacy C/G count. With an MD
    tag (reference-conditioned path, no SEQ mismatches) and without one."""
    from fiberhmm.qc.core import _signal_profile

    header = pysam.AlignmentHeader.from_dict(
        {"HD": {"VN": "1.6"}, "SQ": [{"SN": "chr1", "LN": 10_000}]})
    sequence = _random_sequence(1200, seed=13)

    def daf_read(flag, n_listed, name):
        read = _mm_read(header, sequence, reverse=False, name=name,
                        specs=[("C", "+", flag, n_listed, 10)])
        read.set_tag("st", "CT")
        if with_md:
            read.set_tag("MD", str(len(sequence)))
        return read

    legacy = sequence.count("C") + sequence.count("G")
    positions, opportunities, _, _ = _signal_profile(
        daf_read(".", None, "dot"), "daf")
    assert (len(positions), opportunities) == (10, legacy)
    positions, opportunities, _, _ = _signal_profile(
        daf_read("?", 50, "question"), "daf")
    assert len(positions) == 10
    assert opportunities == sequence.count("G") + 50


def _revcomp(sequence):
    return sequence.translate(str.maketrans("ACGT", "TGCA"))[::-1]


@pytest.mark.parametrize("flag", [".", "?"])
def test_qc_nanopore_orientation_invariant(flag):
    """Regression: ONT QC counted SEQ A as opportunities on reverse-aligned
    reads, i.e. the opposite strand to the basecalled-forward A's the MM
    calls refer to. The same molecule aligned forward or reverse must give
    identical opportunities, events and rate."""
    from fiberhmm.qc.core import _signal_profile

    header = pysam.AlignmentHeader.from_dict(
        {"HD": {"VN": "1.6"}, "SQ": [{"SN": "chr1", "LN": 10_000}]})
    # A/T-skewed molecule so the two strands' A counts clearly differ.
    rng = np.random.default_rng(21)
    molecule = "".join(rng.choice(list("ACGT"), p=[0.4, 0.2, 0.2, 0.2],
                                  size=1500))
    assert molecule.count("A") != molecule.count("T")
    n_listed = None if flag == "." else 300
    forward = _mm_read(header, molecule, reverse=False, name="fwd",
                       specs=[("A", "+", flag, n_listed, 40)])
    reverse = _mm_read(header, _revcomp(molecule), reverse=True, name="rev",
                       specs=[("A", "+", flag, n_listed, 40)])
    assert forward.get_tag("MM") == reverse.get_tag("MM")

    fwd_positions, fwd_opportunities, _, _ = _signal_profile(
        forward, "nanopore-fiber")
    rev_positions, rev_opportunities, _, _ = _signal_profile(
        reverse, "nanopore-fiber")
    expected = molecule.count("A") if flag == "." else 300
    assert fwd_opportunities == rev_opportunities == expected
    assert len(fwd_positions) == len(rev_positions) == 40
    fwd, _ = analyze_sample([forward], "nanopore-fiber", None,
                            min_opportunities=50)
    rev, _ = analyze_sample([reverse], "nanopore-fiber", None,
                            min_opportunities=50)
    assert fwd["signal"] == rev["signal"]


def test_qc_pacbio_counts_both_strands_in_either_orientation():
    from fiberhmm.qc.core import _signal_profile

    header = pysam.AlignmentHeader.from_dict(
        {"HD": {"VN": "1.6"}, "SQ": [{"SN": "chr1", "LN": 10_000}]})
    molecule = _random_sequence(1500, seed=22)
    specs = [("A", "+", ".", None, 30), ("T", "-", ".", None, 10)]
    forward = _mm_read(header, molecule, reverse=False, name="f", specs=specs)
    reverse = _mm_read(header, _revcomp(molecule), reverse=True, name="r",
                       specs=specs)
    both = molecule.count("A") + molecule.count("T")
    assert _signal_profile(forward, "pacbio-fiber")[1] == both
    assert _signal_profile(reverse, "pacbio-fiber")[1] == both


# --- assay inference reads FiberHMM's own records only (audit H1) -----------

def _header_only_bam(path, programs, comments=()):
    header = pysam.AlignmentHeader.from_dict({
        "HD": {"VN": "1.6", "SO": "coordinate"},
        "SQ": [{"SN": "chr1", "LN": 1000}],
        "PG": programs,
        "CO": list(comments),
    })
    with pysam.AlignmentFile(str(path), "wb", header=header):
        pass
    return str(path)


_DEDUP_PG = {
    "ID": "fiberhmm-dedup", "PN": "fiberhmm-dedup",
    "DS": "DAF duplicate marking; grouping=deamination_flavour min_jaccard=0.95 "
          "prob_threshold=128 mode=flag",
    "CL": "fiberhmm-dedup -i in.bam -o out.bam",
}
_DDDB_CALL_PG = {
    "ID": "fiberhmm-call", "PN": "fiberhmm-call", "PP": "fiberhmm-dedup",
    "DS": "FiberHMM fused apply+recall; coord=molecular; mode=daf enzyme=dddb "
          "prob_threshold=None",
    "CL": "fiberhmm-call -i in.bam -o out.bam --enzyme dddb",
}


@pytest.mark.parametrize("dedup_mode", ["flag", "collapse"])
def test_deduplicated_daf_bam_is_daf_not_dedup_mode(tmp_path, dedup_mode):
    dedup = dict(_DEDUP_PG, DS=_DEDUP_PG["DS"].replace("mode=flag", f"mode={dedup_mode}"))
    path = _header_only_bam(
        tmp_path / "dddb.bam", [dedup, _DDDB_CALL_PG],
        ["FIBERHMM-CHEMISTRY:v1:assay=daf;enzyme=dddb;platform=nanopore;mode=daf"])
    assert infer_assay(path, [], mode="auto", enzyme="auto") == ("daf", "dddb", "dddb")


def test_deduplicated_daf_bam_without_declaration_uses_the_call_record(tmp_path):
    path = _header_only_bam(tmp_path / "dddb.bam", [_DEDUP_PG, _DDDB_CALL_PG])
    assert infer_assay(path, [], mode="auto", enzyme="auto") == ("daf", "dddb", "dddb")


def test_unrelated_program_text_never_decides_the_assay(tmp_path):
    # Codex2 #5: an earlier record mentions mode=daf enzyme=dddb; the BAM's
    # FiberHMM writer called Nanopore Hia5.
    other = {"ID": "custom-tool", "PN": "custom-tool", "DS": "mode=daf enzyme=dddb"}
    call = {"ID": "fiberhmm-call", "PN": "fiberhmm-call", "PP": "custom-tool",
            "DS": "FiberHMM fused apply+recall; mode=nanopore-fiber enzyme=hia5",
            "CL": "fiberhmm-call -i a.bam -o b.bam --enzyme hia5 --seq nanopore"}
    path = _header_only_bam(tmp_path / "ont.bam", [other, call])
    assert infer_assay(path, [], mode="auto", enzyme="auto") == (
        "nanopore-fiber", "hia5", "hia5_nanopore")
    declared = _header_only_bam(
        tmp_path / "ont_declared.bam", [other, call],
        ["FIBERHMM-CHEMISTRY:v1:assay=fiber-seq;enzyme=hia5;platform=nanopore;"
         "mode=nanopore-fiber"])
    assert infer_assay(declared, [], mode="auto", enzyme="auto") == (
        "nanopore-fiber", "hia5", "hia5_nanopore")


def test_explicit_mode_takes_the_declared_enzyme(tmp_path):
    path = _header_only_bam(
        tmp_path / "dddb.bam", [_DEDUP_PG, _DDDB_CALL_PG],
        ["FIBERHMM-CHEMISTRY:v1:assay=daf;enzyme=dddb;platform=nanopore;mode=daf"])
    assert infer_assay(path, [], mode="daf", enzyme="auto") == ("daf", "dddb", "dddb")


def test_standalone_qc_grades_deduplicated_daf_bam_as_daf(tmp_path):
    bam_path = tmp_path / "dddb.calls.bam"
    _write_unindexed_iupac_bam(bam_path, n_reads=40)
    with pysam.AlignmentFile(str(bam_path)) as bam:
        header = bam.header.to_dict()
        reads = [read.to_dict() for read in bam]
    header["PG"] = [_DEDUP_PG, _DDDB_CALL_PG]
    header["CO"] = ["FIBERHMM-CHEMISTRY:v1:assay=daf;enzyme=dddb;platform=nanopore;mode=daf"]
    out_header = pysam.AlignmentHeader.from_dict(header)
    with pysam.AlignmentFile(str(bam_path), "wb", header=out_header) as out:
        for read in reads:
            out.write(pysam.AlignedSegment.from_dict(read, out_header))
    result = run_qc(str(bam_path), output_prefix=str(tmp_path / "x"),
                    sample_reads=20, stream=None)
    assert result["assay"]["mode"] == "daf"
    assert result["assay"]["enzyme"] == "dddb"


# --- unaligned output is sampled (audit M1) ---------------------------------

def _write_unaligned_mm_bam(path, n_reads=30, with_sq=False):
    header = {"HD": {"VN": "1.6", "SO": "unknown"}}
    if with_sq:
        header["SQ"] = [{"SN": "chr1", "LN": 100_000}]
    header = pysam.AlignmentHeader.from_dict(header)
    with pysam.AlignmentFile(str(path), "wb", header=header) as bam:
        for index in range(n_reads):
            sequence = _random_sequence(3000, index)
            n_a = sequence.count("A")
            read = pysam.AlignedSegment(header)
            read.query_name = f"read_{index:03d}"
            read.query_sequence = sequence
            read.flag = 4
            read.reference_id = -1
            read.reference_start = -1
            read.mapping_quality = 0
            read.set_tag("MM", "A+a," + ",".join(["4"] * (n_a // 5)) + ";", value_type="Z")
            read.set_tag("ML", array_module.array("B", [255] * (n_a // 5)))
            bam.write(read)
    return str(path)


import array as array_module  # noqa: E402


@pytest.mark.parametrize("with_sq", [False, True])
def test_qc_samples_unaligned_reads(tmp_path, with_sq):
    path = _write_unaligned_mm_bam(tmp_path / "ubam.calls.bam", with_sq=with_sq)
    sampled = sample_bam_reads(path, sample_reads=20)
    assert len(sampled.reads) == 20
    assert "unaligned" in sampled.strategy


def test_qc_on_aligned_bam_still_skips_unmapped_records(tmp_path):
    path = tmp_path / "mixed.bam"
    header = pysam.AlignmentHeader.from_dict(
        {"HD": {"VN": "1.6"}, "SQ": [{"SN": "chr1", "LN": 100_000}]})
    with pysam.AlignmentFile(str(path), "wb", header=header) as bam:
        for index in range(10):
            read = pysam.AlignedSegment(header)
            read.query_name = f"r{index}"
            read.query_sequence = "ACGT" * 50
            if index < 5:
                read.flag, read.reference_id, read.reference_start = 0, 0, index * 300
                read.mapping_quality, read.cigar = 60, [(0, 200)]
            else:
                read.flag, read.reference_id, read.reference_start = 4, -1, -1
            bam.write(read)
    sampled = sample_bam_reads(str(path), sample_reads=50)
    assert sorted(read.query_name for read in sampled.reads) == [f"r{i}" for i in range(5)]


# --- user errors are one line, exit 2 (audit M2) ----------------------------

def test_qc_cli_incompatible_enzyme_is_a_clean_error(tmp_path, capsys):
    from fiberhmm.cli.qc import main

    path = _header_only_bam(
        tmp_path / "hia5.bam",
        [{"ID": "fiberhmm-call", "PN": "fiberhmm-call",
          "DS": "mode=pacbio-fiber enzyme=hia5", "CL": "fiberhmm-call --enzyme hia5"}],
        ["FIBERHMM-CHEMISTRY:v1:assay=fiber-seq;enzyme=hia5;platform=pacbio;"
         "mode=pacbio-fiber"])
    assert main(["-i", path, "-o", str(tmp_path / "qc"), "--enzyme", "dddb"]) == 2
    err = capsys.readouterr().err
    assert "incompatible QC assay" in err and "Traceback" not in err


def test_qc_cli_missing_input_is_a_clean_error(tmp_path, capsys):
    from fiberhmm.cli.qc import main

    assert main(["-i", str(tmp_path / "missing.bam"), "-o", str(tmp_path / "qc")]) == 2
    assert "cannot open BAM/CRAM" in capsys.readouterr().err
