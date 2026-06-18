from __future__ import annotations

import io

import numpy as np
import pytest

from fiberhmm.inference import stats as stats_module


class _FakeRead:
    is_unmapped = False
    is_secondary = False
    is_supplementary = False
    query_length = 100
    is_reverse = False

    def __init__(self, tags=None):
        self._tags = tags or {}

    def get_tag(self, tag):
        if tag not in self._tags:
            raise KeyError(tag)
        return self._tags[tag]


def _fake_read(**attrs):
    read = _FakeRead()
    for name, value in attrs.items():
        setattr(read, name, value)
    return read


class _FakeBam:
    def __init__(self, reads=(), *, fail_iter: bool = False):
        self._reads = list(reads)
        self._fail_iter = fail_iter
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
        return False

    def __iter__(self):
        if self._fail_iter:
            raise RuntimeError("stats iteration failed")
        return iter(self._reads)

    def close(self):
        self.closed = True


def test_collect_stats_closes_second_bam_when_sampling_fails(monkeypatch):
    opened = []

    def fake_alignment_file(*args, **kwargs):
        handle = _FakeBam([_FakeRead()], fail_iter=len(opened) == 1)
        opened.append(handle)
        return handle

    monkeypatch.setattr(stats_module.pysam, "AlignmentFile", fake_alignment_file)

    with pytest.raises(RuntimeError, match="stats iteration failed"):
        stats_module.collect_stats_from_bam("input.bam", n_samples=1)

    assert len(opened) == 2
    assert all(handle.closed for handle in opened)


def test_count_primary_mapped_reads_filters_alignments_and_closes(monkeypatch):
    handle = _FakeBam([
        _fake_read(),
        _fake_read(is_unmapped=True),
        _fake_read(is_secondary=True),
    ])

    monkeypatch.setattr(
        stats_module.pysam,
        "AlignmentFile",
        lambda *args, **kwargs: handle,
    )

    assert stats_module._count_primary_mapped_reads("input.bam") == 1
    assert handle.closed


def test_positive_gaps_between_intervals_sorts_and_skips_overlaps():
    gaps = stats_module._positive_gaps_between_intervals(
        np.array([50, 0, 20]),
        np.array([10, 15, 40]),
    )

    assert gaps == [5]


def test_stats_tag_helpers_flip_intervals_and_scale_scores():
    read = _FakeRead({"ns": [10], "nl": [5], "nq": [255, 0]})

    intervals = stats_module._flipped_interval_tag_arrays(
        read, "ns", "nl", (None, None),
    )
    missing = stats_module._flipped_interval_tag_arrays(
        read, "as", "al", ("missing", "missing"),
    )
    scores = stats_module._scaled_score_tag(read, "nq")

    np.testing.assert_array_equal(intervals.starts, [10])
    np.testing.assert_array_equal(intervals.lengths, [5])
    assert missing.starts == "missing"
    assert missing.lengths == "missing"
    np.testing.assert_allclose(scores, [1.0, 0.0])
    assert stats_module._scaled_score_tag(read, "aq") is None


def test_stats_read_signal_arrays_loads_intervals_and_optional_scores():
    read = _FakeRead({
        "ns": [10],
        "nl": [5],
        "as": [20],
        "al": [7],
        "nq": [255],
        "aq": [128],
    })

    without_scores = stats_module._stats_read_signal_arrays(read, with_scores=False)
    with_scores = stats_module._stats_read_signal_arrays(read, with_scores=True)

    np.testing.assert_array_equal(without_scores.nuc_starts, [10])
    np.testing.assert_array_equal(without_scores.nuc_lengths, [5])
    np.testing.assert_array_equal(without_scores.msp_starts, [20])
    np.testing.assert_array_equal(without_scores.msp_lengths, [7])
    assert without_scores.nuc_scores is None
    assert without_scores.msp_scores is None
    np.testing.assert_allclose(with_scores.nuc_scores, [1.0])
    np.testing.assert_allclose(with_scores.msp_scores, [128 / 255])


def test_add_read_to_footprint_stats_updates_accumulator():
    stats = stats_module.FootprintStats()
    read = _FakeRead({
        "ns": [10],
        "nl": [5],
        "as": [20],
        "al": [7],
        "nq": [255],
        "aq": [128],
    })

    stats_module._add_read_to_footprint_stats(stats, read, with_scores=True)

    assert stats.total_reads_sampled == 1
    assert stats.read_lengths == [100]
    assert stats.footprint_sizes == [5]
    assert stats.msp_sizes == [7]
    np.testing.assert_allclose(stats.footprint_scores, [1.0])
    np.testing.assert_allclose(stats.msp_scores, [128 / 255])


def test_footprint_size_bin_counts_use_stable_labels():
    labels, counts = stats_module._footprint_size_bin_counts(
        [0, 19, 20, 149, 500, 9999],
    )

    assert labels == [
        "0-20",
        "20-40",
        "40-60",
        "60-80",
        "80-100",
        "100-150",
        "150-200",
        "200-300",
        "300-500",
        "500+",
    ]
    assert counts.tolist() == [2, 1, 0, 0, 0, 1, 0, 0, 0, 2]


def test_positive_counts_filters_zero_counts_without_reordering():
    assert stats_module._positive_counts([0, 3, 1, 0, 2]) == [3, 1, 2]


def test_footprint_coverage_fraction_handles_zero_read_length():
    assert stats_module._footprint_coverage_fraction(25, 100) == 0.25
    assert stats_module._footprint_coverage_fraction(25, 0) == 0


def test_initial_footprint_summary_handles_empty_and_nonempty_reads():
    assert stats_module._initial_footprint_summary(0, 0) == {
        "total_reads_sampled": 0,
        "reads_with_footprints": 0,
        "pct_reads_with_footprints": 0,
    }
    assert stats_module._initial_footprint_summary(4, 3) == {
        "total_reads_sampled": 4,
        "reads_with_footprints": 3,
        "pct_reads_with_footprints": 75.0,
    }


def test_add_numeric_summary_writes_requested_keys_and_skips_empty_values():
    summary = {}

    stats_module._add_numeric_summary(
        summary,
        "example",
        [1, 2, 5],
        total_key="total_examples",
        include_std=True,
        include_minmax=True,
        include_iqr=True,
    )
    stats_module._add_numeric_summary(summary, "empty", [])

    assert summary["total_examples"] == 3
    assert summary["example_median"] == 2
    assert summary["example_mean"] == np.mean([1, 2, 5])
    assert summary["example_std"] == np.std([1, 2, 5])
    assert summary["example_min"] == 1
    assert summary["example_max"] == 5
    assert summary["example_q25"] == np.percentile([1, 2, 5], 25)
    assert summary["example_q75"] == np.percentile([1, 2, 5], 75)
    assert "empty_median" not in summary


def test_add_positive_count_summary_ignores_zero_values():
    summary = {}

    stats_module._add_positive_count_summary(
        summary, "footprints_per_read", [0, 3, 1, 0],
    )
    stats_module._add_positive_count_summary(summary, "empty", [0, 0])

    assert summary["footprints_per_read_median"] == 2
    assert summary["footprints_per_read_mean"] == 2
    assert "empty_median" not in summary


def test_write_read_stats_section_formats_counts_and_lengths():
    handle = io.StringIO()

    stats_module._write_read_stats_section(
        handle,
        {
            "total_reads_sampled": 1000,
            "reads_with_footprints": 250,
            "pct_reads_with_footprints": 25.0,
            "read_length_median": 1200,
            "read_length_mean": 1500,
            "read_length_std": 250,
        },
    )

    assert handle.getvalue() == (
        "Read Statistics\n"
        "------------------------------\n"
        "Total reads sampled:        1,000\n"
        "Reads with footprints:      250 (25.0%)\n"
        "Read length (median):       1200 bp\n"
        "Read length (mean ± std):   1500 ± 250 bp\n"
        "\n"
    )


def test_write_footprint_stats_section_formats_size_and_coverage():
    handle = io.StringIO()

    stats_module._write_footprint_stats_section(
        handle,
        {
            "total_footprints": 1234,
            "footprint_size_median": 147,
            "footprint_size_mean": 151.25,
            "footprint_size_std": 12.5,
            "footprint_size_min": 80,
            "footprint_size_max": 220,
            "footprint_size_q25": 130,
            "footprint_size_q75": 170,
            "footprints_per_read_median": 3,
            "footprints_per_read_mean": 3.5,
            "footprint_coverage_median": 0.42,
        },
    )

    assert handle.getvalue() == (
        "Footprint Statistics\n"
        "------------------------------\n"
        "Total footprints:           1,234\n"
        "Size (median):              147 bp\n"
        "Size (mean ± std):          151.2 ± 12.5 bp\n"
        "Size (range):               80 - 220 bp\n"
        "Size (IQR):                 130 - 170 bp\n"
        "Per read (median):          3.0\n"
        "Per read (mean):            3.5\n"
        "Read coverage (median):     42.0%\n"
        "\n"
    )


def test_write_gap_stats_section_formats_when_present_only():
    handle = io.StringIO()

    stats_module._write_gap_stats_section(
        handle,
        {
            "total_gaps": 42,
            "gap_size_median": 75,
            "gap_size_mean": 80.25,
            "gap_size_std": 12.5,
        },
    )

    assert handle.getvalue() == (
        "Gap (Accessible Region) Statistics\n"
        "------------------------------\n"
        "Total gaps:                 42\n"
        "Gap size (median):          75 bp\n"
        "Gap size (mean ± std):      80.2 ± 12.5 bp\n"
        "\n"
    )

    empty_handle = io.StringIO()
    stats_module._write_gap_stats_section(empty_handle, {})
    assert empty_handle.getvalue() == ""


def test_write_msp_stats_section_formats_when_present_only():
    handle = io.StringIO()

    stats_module._write_msp_stats_section(
        handle,
        {
            "total_msps": 123,
            "msp_size_median": 225,
            "msp_size_mean": 250.5,
        },
    )

    assert handle.getvalue() == (
        "MSP (Large Accessible) Statistics\n"
        "------------------------------\n"
        "Total MSPs:                 123\n"
        "MSP size (median):          225 bp\n"
        "MSP size (mean):            250.5 bp\n"
        "\n"
    )

    empty_handle = io.StringIO()
    stats_module._write_msp_stats_section(empty_handle, {})
    assert empty_handle.getvalue() == ""


def test_write_footprint_quality_section_formats_when_present_only():
    handle = io.StringIO()

    stats_module._write_footprint_quality_section(
        handle,
        {
            "footprint_score_median": 0.5,
            "footprint_score_mean": 0.625,
            "footprint_score_std": 0.125,
        },
    )

    assert handle.getvalue() == (
        "Footprint Quality Scores\n"
        "------------------------------\n"
        "Score (median):             0.500\n"
        "Score (mean ± std):         0.625 ± 0.125\n"
    )

    empty_handle = io.StringIO()
    stats_module._write_footprint_quality_section(empty_handle, {})
    assert empty_handle.getvalue() == ""


def test_plot_median_histogram_formats_axis_and_optional_range():
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

    ax = FakeAxis()

    stats_module._plot_median_histogram(
        ax,
        [10, 20, 30],
        bins=50,
        hist_range=(0, 100),
        color="steelblue",
        xlabel="Footprint Size (bp)",
        title="Footprint Size Distribution",
        median_format=".0f",
        median_suffix=" bp",
    )

    values, hist_kwargs = ax.hist_calls[0]
    np.testing.assert_array_equal(values, [10, 20, 30])
    assert hist_kwargs == {
        "bins": 50,
        "color": "steelblue",
        "edgecolor": "white",
        "alpha": 0.8,
        "range": (0, 100),
    }
    assert ax.vlines == [
        (
            20.0,
            {"color": "red", "linestyle": "--", "label": "Median: 20 bp"},
        )
    ]
    assert ax.xlabel == "Footprint Size (bp)"
    assert ax.ylabel == "Count"
    assert ax.title == "Footprint Size Distribution"
    assert ax.legend_called

    no_range_ax = FakeAxis()
    stats_module._plot_median_histogram(
        no_range_ax,
        [0.1, 0.3],
        bins=10,
        color="gold",
        xlabel="Score",
        title="Score Distribution",
        median_format=".2f",
    )

    assert "range" not in no_range_ax.hist_calls[0][1]
    assert no_range_ax.vlines[0][1]["label"] == "Median: 0.20"


def test_plot_no_data_message_centers_text_and_sets_title():
    transform = object()

    class FakeAxis:
        def __init__(self):
            self.transAxes = transform
            self.text_calls = []
            self.title = None

        def text(self, x, y, message, **kwargs):
            self.text_calls.append((x, y, message, kwargs))

        def set_title(self, value):
            self.title = value

    ax = FakeAxis()

    stats_module._plot_no_data_message(ax, "No gap data", "Gap Size Distribution")

    assert ax.text_calls == [
        (
            0.5,
            0.5,
            "No gap data",
            {"ha": "center", "va": "center", "transform": transform},
        )
    ]
    assert ax.title == "Gap Size Distribution"


def test_plot_footprint_size_bins_uses_threshold_and_labels():
    transform = object()

    class FakeAxis:
        def __init__(self):
            self.transAxes = transform
            self.barh_calls = []
            self.yticks = None
            self.yticklabels = None
            self.xlabel = None
            self.ylabel = None
            self.title = None
            self.text_calls = []

        def barh(self, y_pos, values, **kwargs):
            self.barh_calls.append((y_pos, values, kwargs))

        def set_yticks(self, values):
            self.yticks = values

        def set_yticklabels(self, values):
            self.yticklabels = values

        def set_xlabel(self, value):
            self.xlabel = value

        def set_ylabel(self, value):
            self.ylabel = value

        def set_title(self, value):
            self.title = value

        def text(self, x, y, message, **kwargs):
            self.text_calls.append((x, y, message, kwargs))

    ax = FakeAxis()

    stats_module._plot_footprint_size_bins(ax, [10] * 101)

    y_pos, counts, kwargs = ax.barh_calls[0]
    assert list(y_pos) == list(range(10))
    assert counts.tolist() == [101, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    assert kwargs == {"color": "steelblue", "alpha": 0.8}
    assert list(ax.yticks) == list(range(10))
    assert ax.yticklabels == [
        "0-20",
        "20-40",
        "40-60",
        "60-80",
        "80-100",
        "100-150",
        "150-200",
        "200-300",
        "300-500",
        "500+",
    ]
    assert ax.xlabel == "Count"
    assert ax.ylabel == "Size Range (bp)"
    assert ax.title == "Footprint Size Bins"

    small_ax = FakeAxis()
    stats_module._plot_footprint_size_bins(small_ax, [10] * 100)

    assert small_ax.barh_calls == []
    assert small_ax.text_calls == [
        (
            0.5,
            0.5,
            "Insufficient data",
            {"ha": "center", "va": "center", "transform": transform},
        )
    ]
    assert small_ax.title == "Footprint Size Bins"


def test_plot_footprint_size_png_axis_uses_export_styling():
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

    ax = FakeAxis()

    stats_module._plot_footprint_size_png_axis(ax, [10, 20, 30])

    values, hist_kwargs = ax.hist_calls[0]
    np.testing.assert_array_equal(values, [10, 20, 30])
    assert hist_kwargs == {
        "bins": 100,
        "range": (0, min(500, np.percentile([10, 20, 30], 99))),
        "color": "steelblue",
        "edgecolor": "white",
        "alpha": 0.8,
    }
    assert ax.vlines == [
        (
            20.0,
            {
                "color": "red",
                "linestyle": "--",
                "linewidth": 2,
                "label": "Median: 20 bp",
            },
        )
    ]
    assert ax.xlabel == ("Footprint Size (bp)", {"fontsize": 12})
    assert ax.ylabel == ("Count", {"fontsize": 12})
    assert ax.title == ("Footprint Size Distribution", {"fontsize": 14})
    assert ax.legend_kwargs == {"fontsize": 11}


class _FakeFigure:
    def __init__(self):
        self.suptitles = []

    def suptitle(self, *args, **kwargs):
        self.suptitles.append((args, kwargs))


class _FakePyplot:
    def __init__(self):
        self.subplots_calls = []
        self.tight_layout_calls = 0
        self.closed = []
        self.fig = _FakeFigure()
        self.axes = np.array([["00", "01"], ["10", "11"]], dtype=object)

    def subplots(self, *args, **kwargs):
        self.subplots_calls.append((args, kwargs))
        return self.fig, self.axes

    def tight_layout(self):
        self.tight_layout_calls += 1

    def close(self, fig):
        self.closed.append(fig)


class _FakePdf:
    def __init__(self):
        self.saved = []

    def savefig(self, fig):
        self.saved.append(fig)


class _FailingPdf:
    def savefig(self, fig):
        raise RuntimeError("pdf save failed")


class _FakePngPyplot:
    def __init__(self):
        self.subplots_calls = []
        self.tight_layout_calls = 0
        self.savefig_calls = []
        self.closed = []
        self.fig = _FakeFigure()
        self.ax = "png-axis"

    def subplots(self, *args, **kwargs):
        self.subplots_calls.append((args, kwargs))
        return self.fig, self.ax

    def tight_layout(self):
        self.tight_layout_calls += 1

    def savefig(self, *args, **kwargs):
        self.savefig_calls.append((args, kwargs))

    def close(self, fig):
        self.closed.append(fig)


class _FailingPngPyplot(_FakePngPyplot):
    def savefig(self, *args, **kwargs):
        raise RuntimeError("png save failed")


def test_plot_footprint_overview_pdf_page_builds_all_panels(monkeypatch):
    plot_calls = []
    no_data_calls = []

    monkeypatch.setattr(
        stats_module,
        "_plot_median_histogram",
        lambda ax, values, **kwargs: plot_calls.append((ax, values, kwargs)),
    )
    monkeypatch.setattr(
        stats_module,
        "_plot_no_data_message",
        lambda *args: no_data_calls.append(args),
    )
    stats = stats_module.FootprintStats()
    stats.footprint_sizes = [10, 20, 30]
    stats.gap_sizes = [5, 15]
    stats.footprints_per_read = [0, 2, 3]
    stats.footprint_coverage = [0.1, 0.3]
    plt = _FakePyplot()
    pdf = _FakePdf()

    stats_module._plot_footprint_overview_pdf_page(stats, plt, pdf)

    assert plt.subplots_calls == [((2, 2), {"figsize": (10, 8)})]
    assert plt.fig.suptitles == [
        (
            ("FiberHMM Footprint Statistics",),
            {"fontsize": 14, "fontweight": "bold"},
        )
    ]
    assert [call[0] for call in plot_calls] == ["00", "01", "10", "11"]
    assert [call[2]["title"] for call in plot_calls] == [
        "Footprint Size Distribution",
        "Gap (Accessible) Size Distribution",
        "Footprints per Read",
        "Read Coverage by Footprints",
    ]
    np.testing.assert_array_equal(plot_calls[3][1], [10, 30])
    assert no_data_calls == []
    assert plt.tight_layout_calls == 1
    assert pdf.saved == [plt.fig]
    assert plt.closed == [plt.fig]


def test_plot_footprint_overview_pdf_page_closes_on_save_failure(monkeypatch):
    calls = []
    for name in (
        "_plot_footprint_size_pdf_panel",
        "_plot_gap_size_pdf_panel",
        "_plot_footprints_per_read_pdf_panel",
        "_plot_footprint_coverage_pdf_panel",
    ):
        monkeypatch.setattr(
            stats_module,
            name,
            lambda *args, _name=name: calls.append((_name, args)),
        )

    stats = stats_module.FootprintStats()
    stats.footprint_sizes = [10]
    plt = _FakePyplot()

    with pytest.raises(RuntimeError, match="pdf save failed"):
        stats_module._plot_footprint_overview_pdf_page(
            stats, plt, _FailingPdf(),
        )

    assert [call[0] for call in calls] == [
        "_plot_footprint_size_pdf_panel",
        "_plot_gap_size_pdf_panel",
        "_plot_footprints_per_read_pdf_panel",
        "_plot_footprint_coverage_pdf_panel",
    ]
    assert plt.closed == [plt.fig]


def test_plot_footprint_overview_pdf_page_skips_empty_and_marks_missing_gap(monkeypatch):
    plot_calls = []
    no_data_calls = []

    monkeypatch.setattr(
        stats_module,
        "_plot_median_histogram",
        lambda ax, values, **kwargs: plot_calls.append((ax, values, kwargs)),
    )
    monkeypatch.setattr(
        stats_module,
        "_plot_no_data_message",
        lambda *args: no_data_calls.append(args),
    )
    plt = _FakePyplot()
    pdf = _FakePdf()

    stats_module._plot_footprint_overview_pdf_page(
        stats_module.FootprintStats(), plt, pdf,
    )

    assert plt.subplots_calls == []
    assert pdf.saved == []

    stats = stats_module.FootprintStats()
    stats.footprint_sizes = [10, 20]
    stats_module._plot_footprint_overview_pdf_page(stats, plt, pdf)

    assert [call[2]["title"] for call in plot_calls] == [
        "Footprint Size Distribution",
    ]
    assert no_data_calls == [("01", "No gap data", "Gap Size Distribution")]
    assert pdf.saved == [plt.fig]


def test_plot_quality_msp_pdf_page_builds_metric_panels(monkeypatch):
    plot_calls = []
    no_data_calls = []
    bins_calls = []

    monkeypatch.setattr(
        stats_module,
        "_plot_median_histogram",
        lambda ax, values, **kwargs: plot_calls.append((ax, values, kwargs)),
    )
    monkeypatch.setattr(
        stats_module,
        "_plot_no_data_message",
        lambda *args: no_data_calls.append(args),
    )
    monkeypatch.setattr(
        stats_module,
        "_plot_footprint_size_bins",
        lambda ax, values: bins_calls.append((ax, values)),
    )
    stats = stats_module.FootprintStats()
    stats.footprint_scores = [0.1, 0.5]
    stats.msp_sizes = [100, 200]
    stats.read_lengths = [1000, 2000]
    stats.footprint_sizes = [10, 20]
    plt = _FakePyplot()
    pdf = _FakePdf()

    stats_module._plot_quality_msp_pdf_page(stats, plt, pdf)

    assert plt.subplots_calls == [((2, 2), {"figsize": (10, 8)})]
    assert plt.fig.suptitles == [
        (
            ("FiberHMM Quality and MSP Statistics",),
            {"fontsize": 14, "fontweight": "bold"},
        )
    ]
    assert [call[0] for call in plot_calls] == ["00", "01", "10"]
    assert [call[2]["title"] for call in plot_calls] == [
        "Footprint Quality Distribution",
        "MSP Size Distribution",
        "Read Length Distribution",
    ]
    assert no_data_calls == []
    assert bins_calls == [("11", stats.footprint_sizes)]
    assert plt.tight_layout_calls == 1
    assert pdf.saved == [plt.fig]
    assert plt.closed == [plt.fig]


def test_plot_metric_panels_accept_numpy_arrays(monkeypatch):
    plot_calls = []
    no_data_calls = []

    monkeypatch.setattr(
        stats_module,
        "_plot_median_histogram",
        lambda ax, values, **kwargs: plot_calls.append((ax, values, kwargs)),
    )
    monkeypatch.setattr(
        stats_module,
        "_plot_no_data_message",
        lambda *args: no_data_calls.append(args),
    )

    stats_module._plot_gap_size_pdf_panel("gap", np.asarray([10, 20]))
    stats_module._plot_footprint_coverage_pdf_panel("coverage", np.asarray([0.1, 0.2]))
    stats_module._plot_footprint_quality_pdf_panel("quality", np.asarray([0.7, 0.9]))
    stats_module._plot_msp_size_pdf_panel("msp", np.asarray([100, 200]))
    stats_module._plot_read_length_pdf_panel("read", np.asarray([1000, 2000]))

    assert [call[0] for call in plot_calls] == [
        "gap",
        "coverage",
        "quality",
        "msp",
        "read",
    ]
    assert [call[2]["title"] for call in plot_calls] == [
        "Gap (Accessible) Size Distribution",
        "Read Coverage by Footprints",
        "Footprint Quality Distribution",
        "MSP Size Distribution",
        "Read Length Distribution",
    ]
    assert no_data_calls == []


def test_plot_quality_msp_pdf_page_marks_missing_score_and_msp_data(monkeypatch):
    plot_calls = []
    no_data_calls = []
    bins_calls = []

    monkeypatch.setattr(
        stats_module,
        "_plot_median_histogram",
        lambda ax, values, **kwargs: plot_calls.append((ax, values, kwargs)),
    )
    monkeypatch.setattr(
        stats_module,
        "_plot_no_data_message",
        lambda *args: no_data_calls.append(args),
    )
    monkeypatch.setattr(
        stats_module,
        "_plot_footprint_size_bins",
        lambda ax, values: bins_calls.append((ax, values)),
    )
    stats = stats_module.FootprintStats()
    plt = _FakePyplot()
    pdf = _FakePdf()

    stats_module._plot_quality_msp_pdf_page(stats, plt, pdf)

    assert plot_calls == []
    assert no_data_calls == [
        (
            "00",
            "No score data\n(use --scores flag)",
            "Footprint Quality Distribution",
        ),
        ("01", "No MSP data", "MSP Size Distribution"),
    ]
    assert bins_calls == [("11", [])]
    assert pdf.saved == [plt.fig]


def test_plot_quality_msp_pdf_page_closes_on_save_failure(monkeypatch):
    calls = []
    for name in (
        "_plot_footprint_quality_pdf_panel",
        "_plot_msp_size_pdf_panel",
        "_plot_read_length_pdf_panel",
        "_plot_footprint_size_bins",
    ):
        monkeypatch.setattr(
            stats_module,
            name,
            lambda *args, _name=name: calls.append((_name, args)),
        )

    stats = stats_module.FootprintStats()
    plt = _FakePyplot()

    with pytest.raises(RuntimeError, match="pdf save failed"):
        stats_module._plot_quality_msp_pdf_page(stats, plt, _FailingPdf())

    assert [call[0] for call in calls] == [
        "_plot_footprint_quality_pdf_panel",
        "_plot_msp_size_pdf_panel",
        "_plot_read_length_pdf_panel",
        "_plot_footprint_size_bins",
    ]
    assert plt.closed == [plt.fig]


def test_save_footprint_size_png_writes_only_when_sizes_exist(monkeypatch):
    plot_calls = []
    monkeypatch.setattr(
        stats_module,
        "_plot_footprint_size_png_axis",
        lambda ax, values: plot_calls.append((ax, values)),
    )
    stats = stats_module.FootprintStats()
    plt = _FakePngPyplot()

    assert not stats_module._save_footprint_size_png(stats, plt, "out/run")
    assert plt.subplots_calls == []
    assert plot_calls == []

    stats.footprint_sizes = [10, 20, 30]
    assert stats_module._save_footprint_size_png(stats, plt, "out/run")

    assert plt.subplots_calls == [((), {"figsize": (8, 5)})]
    assert plot_calls == [("png-axis", [10, 20, 30])]
    assert plt.tight_layout_calls == 1
    assert plt.savefig_calls == [
        (("out/run_footprint_sizes.png",), {"dpi": 150})
    ]
    assert plt.closed == [plt.fig]


def test_save_footprint_size_png_closes_on_save_failure(monkeypatch):
    plot_calls = []
    monkeypatch.setattr(
        stats_module,
        "_plot_footprint_size_png_axis",
        lambda ax, values: plot_calls.append((ax, values)),
    )
    stats = stats_module.FootprintStats()
    stats.footprint_sizes = [10, 20, 30]
    plt = _FailingPngPyplot()

    with pytest.raises(RuntimeError, match="png save failed"):
        stats_module._save_footprint_size_png(stats, plt, "out/run")

    assert plot_calls == [("png-axis", [10, 20, 30])]
    assert plt.closed == [plt.fig]


def test_stats_sampling_probability_handles_full_and_partial_samples():
    assert stats_module._stats_sampling_probability(10, 100) == 1.0
    assert stats_module._stats_sampling_probability(100, 10) == 0.1
    assert stats_module._stats_sampling_probability(100, 0) == 0.0
