"""Tests for probability context counting primitives."""

import numpy as np
import pytest

from fiberhmm.probabilities.context_counter import (
    ContextCounter,
    _aggregate_context_counts,
    _context_to_code,
    _count_ratio,
    _daf_c_context_from_strand_context,
    _daf_reconstruction_bases,
    _missing_probability_rows,
    _position_weight,
    _probability_dataframe_from_counts,
    _probability_row,
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


def test_probability_row_casts_counts_and_ratio():
    assert _probability_row("ACA", np.array([2, 6])) == {
        "context": "ACA",
        "hit": 2,
        "nohit": 6,
        "ratio": 0.25,
    }


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


def test_context_encoding_helpers_sort_codes_and_create_missing_rows():
    context_to_code = _context_to_code(["CCC", "AAA", "AAC"])

    assert context_to_code == {"AAA": 0, "AAC": 1, "CCC": 2}
    assert _missing_probability_rows(
        ["CCC", "AAA", "AAC"],
        {"AAA"},
        context_to_code,
    ) == [
        {"context": "CCC", "hit": 0, "nohit": 0, "ratio": 0.0, "encode": 2},
        {"context": "AAC", "hit": 0, "nohit": 0, "ratio": 0.0, "encode": 1},
    ]


def test_aggregate_context_counts_trims_and_merges_centered_contexts():
    assert _aggregate_context_counts(
        {
            "AACAA": [2, 3],
            "TACAT": [5, 7],
            "GAGAG": [11, 13],
        },
        max_context=2,
        context_size=1,
    ) == {
        "ACA": [7, 10],
        "AGA": [11, 13],
    }

    with pytest.raises(ValueError, match="Requested context size"):
        _aggregate_context_counts({}, max_context=1, context_size=2)


def test_probability_dataframe_from_counts_sorts_rows_and_adds_encode():
    df = _probability_dataframe_from_counts(
        {
            "TAT": [2, 2],
            "CAC": [1, 3],
        }
    )

    assert df.to_dict("records") == [
        {"context": "CAC", "hit": 1, "nohit": 3, "ratio": 0.25, "encode": 0},
        {"context": "TAT", "hit": 2, "nohit": 2, "ratio": 0.5, "encode": 1},
    ]


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
