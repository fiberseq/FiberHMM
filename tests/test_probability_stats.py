"""Tests for probability statistics report helpers."""

import io

import numpy as np
import pandas as pd

from fiberhmm.probabilities import stats


class _FakeCounter:
    def __init__(self, total_positions, total_modified, counts, table):
        self.total_positions = total_positions
        self.total_modified = total_modified
        self.counts = counts
        self._table = table
        self.context_sizes = []

    def get_encoding_table(self, context_size):
        self.context_sizes.append(context_size)
        return None, self._table


def test_probability_stats_summary_helpers_filter_contexts_with_data(tmp_path):
    table = pd.DataFrame({
        "context": ["AAA", "AAC", "AAG"],
        "ratio": [0.2, 0.4, 0.9],
        "hit": [1, 0, 3],
        "nohit": [4, 0, 0],
    })

    assert stats._modification_rate(_FakeCounter(0, 5, {}, table)) == 5.0
    assert stats._context_observation_totals(table).tolist() == [5, 0, 3]
    assert stats._probability_ratios_with_data(table).tolist() == [0.2, 0.9]


def test_counter_rate_summary_and_writer_format_counts_and_rate():
    table = pd.DataFrame({"context": [], "ratio": [], "hit": [], "nohit": []})
    counter = _FakeCounter(1000, 250, {"AAA": 5, "AAC": 2}, table)
    summary = stats._counter_rate_summary(counter)
    handle = io.StringIO()

    assert summary == {
        "total_positions": 1000,
        "total_modified": 250,
        "rate": 0.25,
        "unique_contexts": 2,
    }

    stats._write_counter_rate_summary(handle, "Accessible", summary)

    assert handle.getvalue() == (
        "\nAccessible:\n"
        "  Total positions:     1,000\n"
        "  Modified positions:  250\n"
        "  Modification rate:   0.2500 (25.00%)\n"
        "  Unique contexts:     2\n"
    )


def test_fold_enrichment_uses_floor_for_small_inaccessible_rate():
    assert stats._fold_enrichment(0.25, 0.1) == 2.5
    assert stats._fold_enrichment(0.25, 0.0) == 250.0


def test_probability_context_label_formats_kmer_width():
    assert stats._probability_context_label(2) == "k=2, 5-mer"
    assert stats._probability_context_label(4) == "k=4, 9-mer"


def test_context_observation_totals_accepts_merged_column_names():
    table = pd.DataFrame({
        "hit_acc": [1, 2],
        "nohit_acc": [3, 4],
    })

    assert stats._context_observation_totals(
        table, "hit_acc", "nohit_acc",
    ).tolist() == [4, 6]


def test_probability_tables_for_base_fetches_accessible_and_inaccessible_tables():
    acc_table = pd.DataFrame({"context": ["AAA"], "ratio": [0.8]})
    inacc_table = pd.DataFrame({"context": ["AAA"], "ratio": [0.2]})
    acc = _FakeCounter(0, 0, {}, acc_table)
    inacc = _FakeCounter(0, 0, {}, inacc_table)

    got_acc, got_inacc = stats._probability_tables_for_base(
        {"A": acc}, {"A": inacc}, "A", context_size=4,
    )

    assert got_acc is acc_table
    assert got_inacc is inacc_table
    assert acc.context_sizes == [4]
    assert inacc.context_sizes == [4]


def test_write_probability_ratio_summary_formats_nonempty_ratios_only():
    handle = io.StringIO()

    stats._write_probability_ratio_summary(
        handle,
        "Accessible",
        np.array([0.2, 0.8]),
    )
    stats._write_probability_ratio_summary(handle, "Empty", np.array([]))

    text = handle.getvalue()
    assert "Accessible contexts with data: 2" in text
    assert "Prob range:  0.2000 - 0.8000" in text
    assert "Prob median: 0.5000" in text
    assert "Empty contexts" not in text


def test_write_base_probability_summary_formats_rates_and_context_stats():
    acc_table = pd.DataFrame({
        "context": ["AAA", "AAC"],
        "ratio": [0.2, 0.8],
        "hit": [1, 4],
        "nohit": [4, 1],
    })
    inacc_table = pd.DataFrame({
        "context": ["AAA", "AAC"],
        "ratio": [0.1, 0.0],
        "hit": [1, 0],
        "nohit": [9, 0],
    })
    handle = io.StringIO()

    stats._write_base_probability_summary(
        handle,
        "A",
        _FakeCounter(100, 25, {"AAA": 5}, acc_table),
        _FakeCounter(200, 20, {"AAA": 10, "AAC": 2}, inacc_table),
        acc_table,
        inacc_table,
        context_size=2,
    )

    text = handle.getvalue()
    assert "A-centered Contexts" in text
    assert "Rate difference:     0.1500" in text
    assert "Fold enrichment:     2.50x" in text
    assert "Per-context statistics (k=2, 5-mer):" in text
    assert "Accessible contexts with data: 2" in text
    assert "Inaccessible contexts with data: 1" in text


def test_write_probability_stats_summary(tmp_path):
    acc_table = pd.DataFrame({
        "context": ["AAA", "AAC", "AAG"],
        "ratio": [0.2, 0.4, 0.9],
        "hit": [1, 0, 3],
        "nohit": [4, 0, 0],
    })
    inacc_table = pd.DataFrame({
        "context": ["AAA", "AAC"],
        "ratio": [0.1, 0.3],
        "hit": [1, 2],
        "nohit": [9, 0],
    })
    summary_path = tmp_path / "prob_stats.txt"

    stats._write_probability_stats_summary(
        str(summary_path),
        {"A": _FakeCounter(100, 25, {"AAA": 5}, acc_table)},
        {"A": _FakeCounter(200, 20, {"AAA": 10, "AAC": 2}, inacc_table)},
        context_size=2,
        title_prefix="Test",
    )

    text = summary_path.read_text()
    assert "Test Emission Probability Statistics (k=2, 5-mer)" in text
    assert "A-centered Contexts" in text
    assert "Modification rate:   0.2500 (25.00%)" in text
    assert "Modification rate:   0.1000 (10.00%)" in text
    assert "Fold enrichment:     2.50x" in text
    assert "Accessible contexts with data: 2" in text
    assert "Prob range:  0.2000 - 0.9000" in text


def test_probability_stats_output_path_uses_context_stem():
    assert stats._probability_stats_output_path("plots", "run", 4, "pdf") == (
        "plots/run_k4_stats.pdf"
    )


def test_merged_probability_table_aligns_contexts_and_totals():
    acc = pd.DataFrame({
        "context": ["AAA", "AAC"],
        "ratio": [0.7, 0.2],
        "hit": [7, 1],
        "nohit": [3, 9],
    })
    inacc = pd.DataFrame({
        "context": ["AAA", "AAG"],
        "ratio": [0.1, 0.5],
        "hit": [1, 5],
        "nohit": [9, 5],
    })

    merged = stats._merged_probability_table(acc, inacc)

    assert merged["context"].tolist() == ["AAA"]
    assert merged["ratio_acc"].tolist() == [0.7]
    assert merged["ratio_inacc"].tolist() == [0.1]
    assert merged["total_acc"].tolist() == [10]
    assert merged["total_inacc"].tolist() == [10]


def test_probability_plot_helpers_filter_log_odds_and_rank_contexts():
    merged = pd.DataFrame({
        "context": ["AAA", "AAC", "AAG"],
        "ratio_acc": [0.8, 0.01, 1.0],
        "ratio_inacc": [0.2, 0.5, 0.0],
    })

    log_odds = stats._filtered_log_odds(merged, eps=0.01)

    assert log_odds.round(3).tolist() == [2.0, -5.644, 6.629]

    top = stats._top_differentiating_contexts(merged, n=2)

    assert top["context"].tolist() == ["AAG", "AAA"]
    assert top["diff"].round(1).tolist() == [1.0, 0.6]
    assert "diff" not in merged.columns


def test_contexts_with_min_observations_requires_both_totals_above_threshold():
    merged = pd.DataFrame({
        "total_acc": [10, 11, 20],
        "total_inacc": [11, 10, 30],
    })

    assert stats._contexts_with_min_observations(merged, 10).tolist() == [
        False,
        False,
        True,
    ]


def test_cumulative_observation_percentages_sorts_and_handles_empty_totals():
    percentages = stats._cumulative_observation_percentages([1, 4, 0])

    assert percentages.tolist() == [80.0, 100.0, 100.0]
    assert stats._cumulative_observation_percentages([0, 0]).tolist() == []


def test_positive_log10_observations_filters_zero_totals():
    logged = stats._positive_log10_observations([0, 9, 99])

    assert logged.round(3).tolist() == [1.0, 2.0]
