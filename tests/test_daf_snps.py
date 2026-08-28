"""Tests for optional opposite-conversion DAF SNP calling and masking."""
from __future__ import annotations

from inspect import signature
from pathlib import Path

import pysam

from fiberhmm.daf.encoder import get_daf_positions
from fiberhmm.cli.daf_snps import parse_args as parse_snp_args
from fiberhmm.daf.snps import (
    DEFAULT_SNP_MIN_ALT_FIBERS,
    DEFAULT_SNP_MIN_DEPTH,
    DEFAULT_SNP_MIN_FRACTION,
    VALIDATED_SNP_POLICY_NAME,
    call_opposite_conversion_snps,
    describe_snp_threshold_policy,
)


def test_validated_policy_is_canonical_but_cli_thresholds_are_overridable():
    policy = describe_snp_threshold_policy(
        DEFAULT_SNP_MIN_FRACTION,
        DEFAULT_SNP_MIN_DEPTH,
        DEFAULT_SNP_MIN_ALT_FIBERS,
    )
    assert policy["name"] == VALIDATED_SNP_POLICY_NAME
    assert policy["uses_validated_defaults"] is True
    assert describe_snp_threshold_policy(0.30, 10, 7)["name"] == "custom"

    api_defaults = signature(call_opposite_conversion_snps).parameters
    assert api_defaults["min_fraction"].default == DEFAULT_SNP_MIN_FRACTION
    assert api_defaults["min_depth"].default == DEFAULT_SNP_MIN_DEPTH
    assert api_defaults["min_alt_fibers"].default == DEFAULT_SNP_MIN_ALT_FIBERS

    defaults = parse_snp_args(["-i", "input.bam"])
    assert defaults.min_fraction == DEFAULT_SNP_MIN_FRACTION
    assert defaults.min_depth == DEFAULT_SNP_MIN_DEPTH
    assert defaults.min_alt_fibers == DEFAULT_SNP_MIN_ALT_FIBERS
    custom = parse_snp_args(
        [
            "-i",
            "input.bam",
            "--min-fraction",
            "0.3",
            "--min-depth",
            "10",
            "--min-alt-fibers",
            "7",
        ]
    )
    assert (custom.min_fraction, custom.min_depth, custom.min_alt_fibers) == (
        0.3,
        10,
        7,
    )


def _md_tag(reference: str, query: str) -> tuple[str, int]:
    fields = []
    matches = 0
    mismatches = 0
    for reference_base, query_base in zip(reference, query):
        if reference_base == query_base:
            matches += 1
        else:
            fields.extend((str(matches), reference_base))
            matches = 0
            mismatches += 1
    fields.append(str(matches))
    return "".join(fields), mismatches


def _write_snp_bam(path: Path) -> None:
    reference = "C" * 50 + "G" * 50
    header = pysam.AlignmentHeader.from_dict(
        {"HD": {"VN": "1.6"}, "SQ": [{"SN": "chr1", "LN": 1000}]}
    )
    with pysam.AlignmentFile(path, "wb", header=header) as bam:
        for index in range(20):
            query = list(reference)
            for position in range(50, 60):
                query[position] = "A"
            if index < 10:
                query[10] = "T"
            if index < 8:
                query[11] = "T"
            if index < 12:
                query[70] = "A"
            sequence = "".join(query)
            md, nm = _md_tag(reference, sequence)
            read = pysam.AlignedSegment(header)
            read.query_name = f"ga_{index}"
            read.query_sequence = sequence
            read.reference_id = 0
            read.reference_start = 100
            read.mapping_quality = 60
            read.cigar = [(0, len(sequence))]
            read.set_tag("MD", md)
            read.set_tag("NM", nm)
            bam.write(read)
        for index in range(20):
            query = list(reference)
            for position in range(20, 30):
                query[position] = "T"
            if index < 8:
                query[70] = "A"
            if index < 12:
                query[10] = "T"
            sequence = "".join(query)
            md, nm = _md_tag(reference, sequence)
            read = pysam.AlignedSegment(header)
            read.query_name = f"ct_{index}"
            read.query_sequence = sequence
            read.reference_id = 0
            read.reference_start = 100
            read.mapping_quality = 60
            read.cigar = [(0, len(sequence))]
            read.set_tag("MD", md)
            read.set_tag("NM", nm)
            bam.write(read)


def test_opposite_conversion_snp_caller_and_encoder_mask(tmp_path):
    bam_path = tmp_path / "snps.bam"
    _write_snp_bam(bam_path)
    payload = call_opposite_conversion_snps(
        str(bam_path),
        min_fraction=0.30,
        min_depth=10,
        min_alt_fibers=3,
        min_dominant_events=5,
        min_dominant_purity=0.80,
    )
    assert payload["threshold_policy"]["name"] == "custom"
    assert payload["threshold_policy"]["validated_defaults"] == {
        "min_fraction_each_direction": 0.20,
        "min_depth_each_direction": 5,
        "min_mismatch_fibers_each_direction": 5,
        "bidirectional_support_required": True,
    }
    calls = {
        (call["position_0based"], call["reference"], call["alternate"]): call
        for call in payload["calls"]
    }
    assert (110, "C", "T") in calls
    assert calls[(110, "C", "T")]["alternate_fraction"] == 0.5
    assert (170, "G", "A") in calls
    assert calls[(170, "G", "A")]["alternate_fraction"] == 0.4
    assert (111, "C", "T") not in calls
    landscape = {
        (site["position_0based"], site["reference"], site["alternate"]): site
        for site in payload["site_distribution"]
    }
    assert landscape[(110, "C", "T")]["expected_direction_mismatch_fraction"] == 0.6
    assert landscape[(110, "C", "T")]["opposite_direction_mismatch_fraction"] == 0.5
    assert landscape[(110, "C", "T")]["called_as_snp"] is True
    assert landscape[(170, "G", "A")]["expected_direction_mismatch_fraction"] == 0.6
    assert landscape[(170, "G", "A")]["opposite_direction_mismatch_fraction"] == 0.4
    assert payload["dominant_amplicon"] == {
        "chrom": "chr1",
        "start_0based": 100,
        "end_0based_exclusive": 200,
        "overlapping_dominant_fibers": 40,
        "selection": "highest-coverage discovered amplicon with >= 20 aligned reads",
    }
    assert payload["n_discovered_amplicons"] == 1
    assert payload["amplicons"][0]["total_aligned_reads"] == 40
    assert payload["amplicons"][0]["consensus_length_bp"] == 100
    assert payload["amplicons"][0]["n_called_snps"] == 2
    assert {
        (site["change"], site["relative_position_bp"])
        for site in payload["amplicons"][0]["snp_positions"]
    } == {("C>T", 10), ("G>A", 70)}

    with pysam.AlignmentFile(bam_path, "rb") as bam:
        read = next(bam.fetch(until_eof=True))
        unmasked = get_daf_positions(read)
        masked = get_daf_positions(read, excluded_reference_positions={110})
    assert unmasked is not None and masked is not None
    assert 10 in unmasked[0]
    assert 10 not in masked[0]
