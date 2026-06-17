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

    def get_encoding_table(self, context_size):
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


def test_context_observation_totals_accepts_merged_column_names():
    table = pd.DataFrame({
        "hit_acc": [1, 2],
        "nohit_acc": [3, 4],
    })

    assert stats._context_observation_totals(
        table, "hit_acc", "nohit_acc",
    ).tolist() == [4, 6]


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
