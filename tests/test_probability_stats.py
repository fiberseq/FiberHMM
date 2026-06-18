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
    assert stats._probability_distribution_plot_path("plots", "run", "A", 4) == (
        "plots/run_A_k4_distribution.png"
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


def test_probability_ratio_histograms_share_bins_colors_and_optional_medians():
    class FakeAxis:
        def __init__(self):
            self.hist_calls = []
            self.vlines = []

        def hist(self, values, **kwargs):
            self.hist_calls.append((values, kwargs))

        def axvline(self, value, **kwargs):
            self.vlines.append((value, kwargs))

    ax = FakeAxis()
    stats._plot_probability_ratio_histograms(
        ax,
        np.array([0.0, 0.2, 0.8]),
        np.array([0.0, 0.1, 0.3]),
    )

    assert len(ax.hist_calls) == 2
    np.testing.assert_array_equal(ax.hist_calls[0][0], [0.0, 0.2, 0.8])
    np.testing.assert_allclose(ax.hist_calls[0][1]["bins"], np.linspace(0, 1, 51))
    assert ax.hist_calls[0][1]["label"] == "Accessible"
    assert ax.hist_calls[0][1]["color"] == "forestgreen"
    assert ax.hist_calls[1][1]["label"] == "Inaccessible"
    assert ax.hist_calls[1][1]["color"] == "firebrick"
    assert ax.vlines == [
        (0.5, {"color": "green", "linestyle": "--", "linewidth": 2}),
        (0.2, {"color": "red", "linestyle": "--", "linewidth": 2}),
    ]

    no_median_ax = FakeAxis()
    stats._plot_probability_ratio_histograms(
        no_median_ax,
        np.array([0.1]),
        np.array([0.2]),
        accessible_label="A",
        inaccessible_label="B",
        show_positive_medians=False,
    )

    assert no_median_ax.hist_calls[0][1]["label"] == "A"
    assert no_median_ax.hist_calls[1][1]["label"] == "B"
    assert no_median_ax.vlines == []


def test_accessible_inaccessible_probability_scatter_filters_and_labels():
    class FakeAxis:
        def __init__(self):
            self.scatter_calls = []
            self.plot_calls = []
            self.xlabel = None
            self.ylabel = None
            self.title = None
            self.xlim = None
            self.ylim = None
            self.legend_called = False

        def scatter(self, x_values, y_values, **kwargs):
            self.scatter_calls.append((x_values, y_values, kwargs))

        def plot(self, x_values, y_values, style, **kwargs):
            self.plot_calls.append((x_values, y_values, style, kwargs))

        def set_xlabel(self, value):
            self.xlabel = value

        def set_ylabel(self, value):
            self.ylabel = value

        def set_title(self, value):
            self.title = value

        def set_xlim(self, start, end):
            self.xlim = (start, end)

        def set_ylim(self, start, end):
            self.ylim = (start, end)

        def legend(self):
            self.legend_called = True

    merged = pd.DataFrame({
        "ratio_acc": [0.8, 0.4],
        "ratio_inacc": [0.2, 0.3],
        "total_acc": [20, 10],
        "total_inacc": [30, 50],
    })
    ax = FakeAxis()

    stats._plot_accessible_inaccessible_probability_scatter(ax, merged)

    x_values, y_values, scatter_kwargs = ax.scatter_calls[0]
    assert x_values.tolist() == [0.2]
    assert y_values.tolist() == [0.8]
    assert scatter_kwargs == {"alpha": 0.5, "s": 10, "c": "steelblue"}
    assert ax.plot_calls == [([0, 1], [0, 1], "k--", {"alpha": 0.3, "label": "y=x"})]
    assert ax.xlabel == "P(m | inaccessible)"
    assert ax.ylabel == "P(m | accessible)"
    assert ax.title == "Accessible vs Inaccessible (1 contexts)"
    assert ax.xlim == (0, 1)
    assert ax.ylim == (0, 1)
    assert ax.legend_called


def test_log_odds_distribution_plots_filtered_values_and_empty_noop():
    class FakeAxis:
        def __init__(self):
            self.hist_calls = []
            self.vlines = []
            self.xlabel = None
            self.ylabel = None
            self.title = None
            self.legend_called = False

        def hist(self, values, **kwargs):
            self.hist_calls.append((values, kwargs))

        def axvline(self, value, **kwargs):
            self.vlines.append((value, kwargs))

        def set_xlabel(self, value):
            self.xlabel = value

        def set_ylabel(self, value):
            self.ylabel = value

        def set_title(self, value):
            self.title = value

        def legend(self):
            self.legend_called = True

    merged = pd.DataFrame({
        "ratio_acc": [0.8, 0.1],
        "ratio_inacc": [0.2, 0.4],
    })
    ax = FakeAxis()

    stats._plot_log_odds_distribution(ax, merged)

    values, hist_kwargs = ax.hist_calls[0]
    np.testing.assert_allclose(values, [2.0, -2.0])
    np.testing.assert_allclose(hist_kwargs["bins"], np.linspace(-5, 10, 51))
    assert hist_kwargs["color"] == "purple"
    assert hist_kwargs["alpha"] == 0.7
    assert hist_kwargs["edgecolor"] == "white"
    assert ax.vlines == [
        (0, {"color": "black", "linestyle": "-", "linewidth": 1}),
        (0.0, {"color": "red", "linestyle": "--", "label": "Median: 0.00"}),
    ]
    assert ax.xlabel == "Log2(P_accessible / P_inaccessible)"
    assert ax.ylabel == "Number of contexts"
    assert ax.title == "Separation Score (Log-Odds)"
    assert ax.legend_called

    empty_ax = FakeAxis()
    stats._plot_log_odds_distribution(empty_ax, pd.DataFrame())
    assert empty_ax.hist_calls == []
    assert empty_ax.vlines == []


def test_top_differentiating_contexts_plot_ranks_and_labels():
    class FakeAxis:
        def __init__(self):
            self.barh_calls = []
            self.yticks = None
            self.yticklabels = None
            self.xlabel = None
            self.title = None
            self.legend_kwargs = None
            self.xlim = None

        def barh(self, y_pos, values, width, **kwargs):
            self.barh_calls.append((y_pos, values, width, kwargs))

        def set_yticks(self, values):
            self.yticks = values

        def set_yticklabels(self, values, **kwargs):
            self.yticklabels = (values, kwargs)

        def set_xlabel(self, value):
            self.xlabel = value

        def set_title(self, value):
            self.title = value

        def legend(self, **kwargs):
            self.legend_kwargs = kwargs

        def set_xlim(self, start, end):
            self.xlim = (start, end)

    merged = pd.DataFrame({
        "context": ["AAA", "AAC", "AAG"],
        "ratio_acc": [0.8, 0.2, 0.7],
        "ratio_inacc": [0.2, 0.1, 0.6],
    })
    ax = FakeAxis()

    stats._plot_top_differentiating_contexts(ax, merged)

    inacc_y, inacc_values, width, inacc_kwargs = ax.barh_calls[0]
    acc_y, acc_values, _, acc_kwargs = ax.barh_calls[1]
    np.testing.assert_allclose(inacc_y, [-0.175, 0.825, 1.825])
    np.testing.assert_allclose(acc_y, [0.175, 1.175, 2.175])
    np.testing.assert_allclose(inacc_values, [0.2, 0.1, 0.6])
    np.testing.assert_allclose(acc_values, [0.8, 0.2, 0.7])
    assert width == 0.35
    assert inacc_kwargs == {
        "label": "Inaccessible",
        "color": "firebrick",
        "alpha": 0.8,
    }
    assert acc_kwargs == {
        "label": "Accessible",
        "color": "forestgreen",
        "alpha": 0.8,
    }
    np.testing.assert_array_equal(ax.yticks, [0, 1, 2])
    values, kwargs = ax.yticklabels
    assert values.tolist() == ["AAA", "AAC", "AAG"]
    assert kwargs == {"fontsize": 7, "fontfamily": "monospace"}
    assert ax.xlabel == "P(methylation)"
    assert ax.title == "Top Differentiating Contexts"
    assert ax.legend_kwargs == {"loc": "lower right"}
    assert ax.xlim == (0, 1)

    empty_ax = FakeAxis()
    stats._plot_top_differentiating_contexts(empty_ax, pd.DataFrame())
    assert empty_ax.barh_calls == []


def test_observations_per_context_plot_uses_positive_log_totals():
    class FakeAxis:
        def __init__(self):
            self.hist_calls = []
            self.xlabel = None
            self.ylabel = None
            self.title = None
            self.legend_called = False

        def hist(self, values, **kwargs):
            self.hist_calls.append((values, kwargs))

        def set_xlabel(self, value):
            self.xlabel = value

        def set_ylabel(self, value):
            self.ylabel = value

        def set_title(self, value):
            self.title = value

        def legend(self):
            self.legend_called = True

    ax = FakeAxis()

    stats._plot_observations_per_context(ax, [0, 9, 99], [4, 0])

    np.testing.assert_allclose(ax.hist_calls[0][0], [1.0, 2.0])
    assert ax.hist_calls[0][1] == {
        "bins": 50,
        "alpha": 0.6,
        "label": "Accessible",
        "color": "forestgreen",
    }
    np.testing.assert_allclose(ax.hist_calls[1][0], [np.log10(5)])
    assert ax.hist_calls[1][1] == {
        "bins": 50,
        "alpha": 0.6,
        "label": "Inaccessible",
        "color": "firebrick",
    }
    assert ax.xlabel == "Log10(observations + 1)"
    assert ax.ylabel == "Number of contexts"
    assert ax.title == "Observations per Context"
    assert ax.legend_called


def test_context_coverage_plot_uses_cumulative_percentages_and_guide():
    class FakeAxis:
        def __init__(self):
            self.plot_calls = []
            self.hlines = []
            self.xlabel = None
            self.ylabel = None
            self.title = None
            self.legend_called = False

        def plot(self, x_values, y_values, **kwargs):
            self.plot_calls.append((x_values, y_values, kwargs))

        def axhline(self, value, **kwargs):
            self.hlines.append((value, kwargs))

        def set_xlabel(self, value):
            self.xlabel = value

        def set_ylabel(self, value):
            self.ylabel = value

        def set_title(self, value):
            self.title = value

        def legend(self):
            self.legend_called = True

    ax = FakeAxis()

    stats._plot_context_coverage(ax, [1, 4, 0], [2, 2])

    acc_x, acc_y, acc_kwargs = ax.plot_calls[0]
    inacc_x, inacc_y, inacc_kwargs = ax.plot_calls[1]
    assert list(acc_x) == [0, 1, 2]
    assert list(inacc_x) == [0, 1]
    np.testing.assert_allclose(acc_y, [80.0, 100.0, 100.0])
    np.testing.assert_allclose(inacc_y, [50.0, 100.0])
    assert acc_kwargs == {"label": "Accessible", "color": "forestgreen"}
    assert inacc_kwargs == {"label": "Inaccessible", "color": "firebrick"}
    assert ax.hlines == [(90, {"color": "gray", "linestyle": "--", "alpha": 0.5})]
    assert ax.xlabel == "Number of contexts (ranked)"
    assert ax.ylabel == "Cumulative % of observations"
    assert ax.title == "Context Coverage (Lorenz-like)"
    assert ax.legend_called


def test_probability_vs_coverage_plot_filters_high_coverage_contexts():
    class FakeAxis:
        def __init__(self):
            self.scatter_calls = []
            self.xscale = None
            self.xlabel = None
            self.ylabel = None
            self.title = None
            self.legend_called = False

        def scatter(self, x_values, y_values, **kwargs):
            self.scatter_calls.append((x_values, y_values, kwargs))

        def set_xscale(self, value):
            self.xscale = value

        def set_xlabel(self, value):
            self.xlabel = value

        def set_ylabel(self, value):
            self.ylabel = value

        def set_title(self, value):
            self.title = value

        def legend(self):
            self.legend_called = True

    merged = pd.DataFrame({
        "total_acc": [101, 100, 200],
        "total_inacc": [150, 300, 99],
        "ratio_acc": [0.8, 0.4, 0.6],
        "ratio_inacc": [0.2, 0.3, 0.5],
    })
    ax = FakeAxis()

    stats._plot_probability_vs_coverage(ax, merged)

    acc_x, acc_y, acc_kwargs = ax.scatter_calls[0]
    inacc_x, inacc_y, inacc_kwargs = ax.scatter_calls[1]
    assert acc_x.tolist() == [101]
    assert acc_y.tolist() == [0.8]
    assert inacc_x.tolist() == [150]
    assert inacc_y.tolist() == [0.2]
    assert acc_kwargs == {
        "alpha": 0.5,
        "s": 10,
        "c": "forestgreen",
        "label": "Accessible",
    }
    assert inacc_kwargs == {
        "alpha": 0.5,
        "s": 10,
        "c": "firebrick",
        "label": "Inaccessible",
    }
    assert ax.xscale == "log"
    assert ax.xlabel == "Total observations (log scale)"
    assert ax.ylabel == "Methylation probability"
    assert ax.title == "Probability vs Coverage"
    assert ax.legend_called

    no_match_ax = FakeAxis()
    stats._plot_probability_vs_coverage(
        no_match_ax,
        pd.DataFrame({
            "total_acc": [100],
            "total_inacc": [101],
        }),
    )
    assert no_match_ax.scatter_calls == []

    empty_ax = FakeAxis()
    stats._plot_probability_vs_coverage(empty_ax, pd.DataFrame())
    assert empty_ax.scatter_calls == []


def test_context_frequency_comparison_plot_scales_counts_and_identity_line():
    class FakeAxis:
        def __init__(self):
            self.scatter_calls = []
            self.plot_calls = []
            self.xlabel = None
            self.ylabel = None
            self.title = None
            self.xscale = None
            self.yscale = None

        def scatter(self, x_values, y_values, **kwargs):
            self.scatter_calls.append((x_values, y_values, kwargs))

        def plot(self, x_values, y_values, style, **kwargs):
            self.plot_calls.append((x_values, y_values, style, kwargs))

        def set_xlabel(self, value):
            self.xlabel = value

        def set_ylabel(self, value):
            self.ylabel = value

        def set_title(self, value):
            self.title = value

        def set_xscale(self, value):
            self.xscale = value

        def set_yscale(self, value):
            self.yscale = value

    merged = pd.DataFrame({
        "total_acc": [10, 0, 30],
        "total_inacc": [20, 5, 40],
    })
    ax = FakeAxis()

    stats._plot_context_frequency_comparison(ax, merged)

    x_values, y_values, scatter_kwargs = ax.scatter_calls[0]
    assert x_values.tolist() == [20, 40]
    assert y_values.tolist() == [10, 30]
    assert scatter_kwargs == {"alpha": 0.5, "s": 10, "c": "steelblue"}
    assert ax.plot_calls == [([0, 40], [0, 40], "k--", {"alpha": 0.3})]
    assert ax.xlabel == "Inaccessible observations"
    assert ax.ylabel == "Accessible observations"
    assert ax.title == "Context Frequency Comparison"
    assert ax.xscale == "log"
    assert ax.yscale == "log"

    no_match_ax = FakeAxis()
    stats._plot_context_frequency_comparison(
        no_match_ax,
        pd.DataFrame({"total_acc": [0], "total_inacc": [1]}),
    )
    assert no_match_ax.scatter_calls == []

    empty_ax = FakeAxis()
    stats._plot_context_frequency_comparison(empty_ax, pd.DataFrame())
    assert empty_ax.scatter_calls == []


def test_probability_distribution_png_axis_uses_counter_labels_without_medians():
    class FakeAxis:
        def __init__(self):
            self.hist_calls = []
            self.vlines = []
            self.xlabel = None
            self.ylabel = None
            self.title = None
            self.legend_kwargs = None

        def hist(self, values, **kwargs):
            self.hist_calls.append((values, kwargs))

        def axvline(self, value, **kwargs):
            self.vlines.append((value, kwargs))

        def set_xlabel(self, value, **kwargs):
            self.xlabel = (value, kwargs)

        def set_ylabel(self, value, **kwargs):
            self.ylabel = (value, kwargs)

        def set_title(self, value, **kwargs):
            self.title = (value, kwargs)

        def legend(self, **kwargs):
            self.legend_kwargs = kwargs

    acc_probs = pd.DataFrame({"ratio": [0.2, 0.8]})
    inacc_probs = pd.DataFrame({"ratio": [0.1, 0.3]})
    ax = FakeAxis()

    stats._plot_probability_distribution_png_axis(
        ax,
        _FakeCounter(1000, 250, {}, acc_probs),
        _FakeCounter(2000, 300, {}, inacc_probs),
        acc_probs,
        inacc_probs,
        base="A",
        context_size=3,
    )

    assert ax.hist_calls[0][1]["label"] == "Accessible (n=1,000)"
    assert ax.hist_calls[1][1]["label"] == "Inaccessible (n=2,000)"
    assert ax.vlines == []
    assert ax.xlabel == ("P(methylation | context)", {"fontsize": 12})
    assert ax.ylabel == ("Number of contexts", {"fontsize": 12})
    assert ax.title == (
        "A-centered Emission Probability Distributions (k=3)",
        {"fontsize": 14},
    )
    assert ax.legend_kwargs == {"fontsize": 11}
