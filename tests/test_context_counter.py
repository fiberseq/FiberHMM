"""Tests for probability context counting primitives."""

import numpy as np

from fiberhmm.probabilities.context_counter import (
    ContextCounter,
    _count_ratio,
    _daf_c_context_from_strand_context,
    _daf_reconstruction_bases,
    _position_weight,
    _reconstruct_deaminated_sequence,
    _trim_context,
)


def test_reconstruct_deaminated_sequence_only_reverts_matching_bases():
    assert _reconstruct_deaminated_sequence("ATTA", {1, 2, 20}, "T", "C") == "ACCA"
    assert _reconstruct_deaminated_sequence("ATTA", {0, 3}, "T", "C") == "ATTA"


def test_position_weight_defaults_out_of_range_to_zero():
    weights = np.array([0.25, 0.75])

    assert _position_weight(weights, 1) == 0.75
    assert _position_weight(weights, 2) == 0.0


def test_count_ratio_handles_zero_total():
    assert _count_ratio(2, 6) == 0.25
    assert _count_ratio(0, 0) == 0.0


def test_trim_context_extracts_centered_subcontext():
    assert _trim_context("ACGTA", 0) == "ACGTA"
    assert _trim_context("AACGTAA", 2) == "CGT"


def test_daf_reconstruction_bases_by_strand():
    assert _daf_reconstruction_bases("+") == ("T", "C")
    assert _daf_reconstruction_bases(".") == ("T", "C")
    assert _daf_reconstruction_bases("-") == ("A", "G")


def test_daf_c_context_orientation_by_strand():
    assert _daf_c_context_from_strand_context("ACA", "+") == "ACA"
    assert _daf_c_context_from_strand_context("ACA", ".") == "ACA"
    assert _daf_c_context_from_strand_context("TGA", "-") == "TCA"


def test_add_position_records_valid_center_contexts():
    counter = ContextCounter(max_context=1, center_base="A")

    counter.add_position("CACAC", 1, True)
    counter.add_position("CACAC", 3, False)
    counter.add_position("CANAC", 1, True)

    assert counter.counts["CAC"] == [1, 1]
    assert counter.total_positions == 2
    assert counter.total_modified == 1


def test_region_and_weighted_region_share_context_accounting():
    unweighted = ContextCounter(max_context=1, center_base="A")
    weighted = ContextCounter(max_context=1, center_base="A")

    unweighted.add_region("CACAC", {1}, 0, 5, edge_trim=0)
    weighted.add_weighted_region(
        "CACAC",
        {1},
        0,
        5,
        weights=np.array([0.0, 0.25, 0.0, 0.75, 0.0]),
        edge_trim=0,
    )

    assert unweighted.counts["CAC"] == [1, 1]
    assert unweighted.total_positions == 2
    assert weighted.counts["CAC"] == [0.25, 0.75]
    assert weighted.total_positions == 1.0
    assert weighted.total_modified == 0.25


def test_daf_plus_strand_reconstructs_converted_c_contexts():
    counter = ContextCounter(max_context=1, center_base="C")

    counter.process_read_daf("ATA", {1}, strand="+", edge_trim=0)

    assert counter.counts["ACA"] == [1, 0]
    assert counter.total_positions == 1
    assert counter.total_modified == 1


def test_daf_minus_strand_reverse_complements_g_contexts():
    counter = ContextCounter(max_context=1, center_base="C")

    counter.process_read_daf("TAA", {1}, strand="-", edge_trim=0)

    assert counter.counts["TCA"] == [1, 0]
    assert counter.total_positions == 1
    assert counter.total_modified == 1
