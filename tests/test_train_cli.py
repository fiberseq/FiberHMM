from types import SimpleNamespace

import pytest
import numpy as np

from fiberhmm.cli import train
from fiberhmm.core import bam_reader

pysam = pytest.importorskip("pysam")


def test_load_legacy_probability_table_normalizes_encode_and_ratio(tmp_path):
    probs_path = tmp_path / "accessible.tsv"
    probs_path.write_text(
        "Unnamed: 0\thit\tnohit\n"
        "0\t2\t2\n"
        "1\t0\t0\n"
    )

    probs = train.load_probability_file(str(probs_path))

    assert probs["encode"].tolist() == [0, 1]
    assert probs["ratio"].tolist() == [0.5, 0.0]


def test_load_legacy_probability_table_requires_ratio_or_counts(tmp_path):
    probs_path = tmp_path / "bad.tsv"
    probs_path.write_text("encode\tother\n0\t1\n")

    with pytest.raises(ValueError, match="Cannot find probability values"):
        train.load_probability_file(str(probs_path))


def test_state_runs_groups_contiguous_states():
    assert train._state_runs([0, 0, 1, 1, 1, 0]) == [
        (0, 2, 0),
        (2, 5, 1),
        (5, 6, 0),
    ]
    assert train._state_runs([]) == []


def test_state_run_lengths_splits_footprint_and_msp_lengths():
    footprint_sizes, msp_sizes = train._state_run_lengths([0, 0, 1, 1, 1, 0])

    assert footprint_sizes == [2, 1]
    assert msp_sizes == [3]


def test_state_block_specs_include_plot_widths_and_colors():
    assert train._state_block_specs([]) == []
    assert train._state_block_specs([0, 0, 1, 0]) == [
        (0, 2, "forestgreen"),
        (2, 1, "white"),
        (3, 1, "forestgreen"),
    ]


def test_expected_model_durations_from_self_transitions():
    durations = train._expected_model_durations(np.array([[0.8, 0.2], [0.1, 1.0]]))

    assert durations[0] == pytest.approx(5.0)
    assert durations[1] == float('inf')


def test_viterbi_state_size_stats_skips_empty_reads():
    class FakeModel:
        def predict(self, encoded):
            return encoded

    footprint_sizes, msp_sizes, states = train._viterbi_state_size_stats(
        FakeModel(),
        [
            np.array([0, 0, 1, 1]),
            np.array([], dtype=int),
            np.array([1, 0, 0]),
        ],
    )

    assert footprint_sizes == [2, 2]
    assert msp_sizes == [2, 1]
    assert len(states) == 2
    np.testing.assert_array_equal(states[0], [0, 0, 1, 1])
    np.testing.assert_array_equal(states[1], [1, 0, 0])


def test_nonzero_emissions_by_state_filters_zero_entries():
    fp, msp = train._nonzero_emissions_by_state(
        np.array([[0.0, 0.2, 0.0, 0.5], [0.1, 0.0, 0.3, 0.0]])
    )

    np.testing.assert_array_equal(fp, [0.2, 0.5])
    np.testing.assert_array_equal(msp, [0.1, 0.3])


def test_shuffled_training_arrays_are_deterministic_int_arrays():
    encoded_reads = [
        np.array([1, 1], dtype=np.int64),
        np.array([2], dtype=np.int64),
        np.array([3, 3], dtype=np.int64),
    ]

    first = train._shuffled_training_arrays(encoded_reads, n_iterations=3)
    second = train._shuffled_training_arrays(encoded_reads, n_iterations=3)

    assert list(first) == [0, 1, 2]
    for iteration, values in first.items():
        np.testing.assert_array_equal(values, second[iteration])
        assert np.issubdtype(values.dtype, np.integer)
        assert sorted(values.tolist()) == [1, 1, 2, 3, 3]


def test_training_strand_for_read_detects_only_daf_mode(monkeypatch):
    read = SimpleNamespace(query_sequence="ACGT", m6a_query_positions={1})

    monkeypatch.setattr(train, "detect_daf_strand", lambda seq, positions: "-")
    assert train._training_strand_for_read(read, "daf") == "-"
    assert train._training_strand_for_read(read, "pacbio-fiber") == "."
    assert train._training_strand_for_read(read, "nanopore-fiber") == "."


def test_encode_training_read_passes_mode_strand_context_and_reverse(monkeypatch):
    read = SimpleNamespace(
        query_sequence="ACGT",
        m6a_query_positions={1},
        is_reverse=True,
    )
    captured = {}

    monkeypatch.setattr(train, "_training_strand_for_read", lambda fiber_read, mode: "+")

    def fake_encode(sequence, mod_positions, edge_trim, **kwargs):
        captured.update(
            sequence=sequence,
            mod_positions=mod_positions,
            edge_trim=edge_trim,
            kwargs=kwargs,
        )
        return np.array([1, 2, 3], dtype=np.int32)

    monkeypatch.setattr(train, "encode_from_query_sequence", fake_encode)

    encoded = train._encode_training_read(
        read,
        edge_trim=12,
        mode="daf",
        context_size=5,
    )

    np.testing.assert_array_equal(encoded, [1, 2, 3])
    assert captured == {
        "sequence": "ACGT",
        "mod_positions": {1},
        "edge_trim": 12,
        "kwargs": {
            "mode": "daf",
            "strand": "+",
            "context_size": 5,
            "is_reverse": True,
        },
    }


def test_training_zoom_window_starts_are_deterministic_and_bounded():
    assert train._training_zoom_window_starts(
        seq_len=500, window_size=1000, n_windows=3, seed=0,
    ) == [0, 0, 0]

    starts = train._training_zoom_window_starts(
        seq_len=6000, window_size=1000, n_windows=3, seed=7,
    )
    assert starts == train._training_zoom_window_starts(
        seq_len=6000, window_size=1000, n_windows=3, seed=7,
    )
    assert len(starts) == 3
    assert all(0 <= start <= 5000 for start in starts)
    assert starts[0] < 2000
    assert 2000 <= starts[1] < 4000
    assert 4000 <= starts[2] <= 5000


def test_training_zoom_window_bounds_add_end_and_length():
    bounds = train._training_zoom_window_bounds(
        seq_len=500,
        window_size=1000,
        n_windows=3,
        seed=0,
    )

    assert bounds == [(0, 500, 500), (0, 500, 500), (0, 500, 500)]

    starts = train._training_zoom_window_starts(
        seq_len=6000,
        window_size=1000,
        n_windows=3,
        seed=7,
    )
    assert train._training_zoom_window_bounds(
        seq_len=6000,
        window_size=1000,
        n_windows=3,
        seed=7,
    ) == [(start, start + 1000, 1000) for start in starts]


def test_relative_positions_in_window_filters_and_offsets():
    assert train._relative_positions_in_window([4, 5, 9, 10], 5, 10) == [0, 4]


def test_training_size_summary_formats_nonempty_and_empty_sizes():
    assert train._training_size_summary(
        "footprints", "Footprint sizes", [10, 30],
    ) == (
        "  Total footprints: 2\n"
        "  Footprint sizes: mean=20.0, median=20.0, range=10-30\n"
    )
    assert train._training_size_summary("MSPs", "MSP sizes", []) == ""


def test_training_example_plot_data_sorts_positions_and_uses_footprint_probs():
    class FakeModel:
        def predict_proba(self, encoded):
            np.testing.assert_array_equal(encoded, [3, 4])
            return np.array([[0.2, 0.8], [0.7, 0.3]])

    read = SimpleNamespace(query_sequence="AC", m6a_query_positions={1, 0})

    seq_len, m6a_positions, footprint_prob = train._training_example_plot_data(
        FakeModel(), read, np.array([3, 4]),
    )

    assert seq_len == 2
    assert m6a_positions == [0, 1]
    np.testing.assert_array_equal(footprint_prob, [0.2, 0.7])


def test_training_example_title_formats_read_context():
    read = SimpleNamespace(
        read_id="read-abcdefghijklmnopqrstuvwxyz0123456789",
        chrom="chr2L",
        ref_start=1234,
        ref_end=5678,
    )

    assert train._training_example_title(2, read, 12345, [1, 2, 3]) == (
        "Example Read 3: read-abcdefghijklmnopqrstuvwxyz0123456789...\n"
        "Length: 12,345bp | Chromosome: chr2L:1,234-5,678 | m6A calls: 3"
    )


def test_add_training_zoom_highlight_marks_all_overview_axes():
    class FakeAxis:
        def __init__(self):
            self.spans = []

        def axvspan(self, *args, **kwargs):
            self.spans.append((args, kwargs))

    axes = (FakeAxis(), FakeAxis(), FakeAxis())

    train._add_training_zoom_highlight(axes, 10, 20, "red")

    assert [axis.spans for axis in axes] == [
        [((10, 20), {"alpha": 0.15, "color": "red"})],
        [((10, 20), {"alpha": 0.15, "color": "red"})],
        [((10, 20), {"alpha": 0.15, "color": "red"})],
    ]


def test_plot_training_probability_area_defaults_to_filled_threshold():
    class FakeAxis:
        def __init__(self):
            self.fill_between_calls = []
            self.plot_calls = []
            self.hlines = []

        def fill_between(self, *args, **kwargs):
            self.fill_between_calls.append((args, kwargs))

        def plot(self, *args, **kwargs):
            self.plot_calls.append((args, kwargs))

        def axhline(self, *args, **kwargs):
            self.hlines.append((args, kwargs))

    ax = FakeAxis()
    prob = np.array([0.2, 0.8])

    train._plot_training_probability_area(ax, prob)

    assert ax.fill_between_calls[0][0][0] == range(2)
    assert ax.fill_between_calls[0][0][1] == 0
    np.testing.assert_array_equal(ax.fill_between_calls[0][0][2], prob)
    assert ax.fill_between_calls[0][1] == {
        "color": "forestgreen",
        "alpha": 0.4,
        "step": "mid",
    }
    assert ax.plot_calls == []
    assert ax.hlines == [
        ((0.5,), {"color": "gray", "linestyle": "--", "linewidth": 0.5})
    ]


def test_plot_training_probability_area_can_overlay_line_and_alpha_threshold():
    class FakeAxis:
        def __init__(self):
            self.fill_between_calls = []
            self.plot_calls = []
            self.hlines = []

        def fill_between(self, *args, **kwargs):
            self.fill_between_calls.append((args, kwargs))

        def plot(self, *args, **kwargs):
            self.plot_calls.append((args, kwargs))

        def axhline(self, *args, **kwargs):
            self.hlines.append((args, kwargs))

    ax = FakeAxis()
    prob = np.array([0.1, 0.4, 0.9])

    train._plot_training_probability_area(
        ax, prob, show_line=True, threshold_alpha=0.5,
    )

    assert ax.plot_calls[0][0][0] == range(3)
    np.testing.assert_array_equal(ax.plot_calls[0][0][1], prob)
    assert ax.plot_calls[0][1] == {
        "color": "forestgreen",
        "linewidth": 0.5,
        "alpha": 0.8,
    }
    assert ax.hlines == [
        (
            (0.5,),
            {
                "color": "gray",
                "linestyle": "--",
                "linewidth": 0.5,
                "alpha": 0.5,
            },
        )
    ]


def test_add_training_state_blocks_uses_state_specs_and_sets_limits():
    rectangles = []
    collections = []

    class FakeRectangle:
        def __init__(self, xy, width, height):
            self.xy = xy
            self.width = width
            self.height = height
            rectangles.append(self)

    class FakePatchCollection:
        def __init__(self, patches, **kwargs):
            self.patches = patches
            self.kwargs = kwargs
            collections.append(self)

    class FakeAxis:
        def __init__(self):
            self.added = []
            self.xlim = None
            self.ylim = None

        def add_collection(self, collection):
            self.added.append(collection)

        def set_xlim(self, start, end):
            self.xlim = (start, end)

        def set_ylim(self, start, end):
            self.ylim = (start, end)

    ax = FakeAxis()
    train._add_training_state_blocks(
        ax,
        [0, 0, 1],
        10,
        FakeRectangle,
        FakePatchCollection,
    )

    assert [(rect.xy, rect.width, rect.height) for rect in rectangles] == [
        ((0, 0), 2, 1),
        ((2, 0), 1, 1),
    ]
    assert collections[0].patches == rectangles
    assert collections[0].kwargs == {
        "facecolors": ["forestgreen", "white"],
        "edgecolors": "lightgray",
        "linewidths": 0.3,
    }
    assert ax.added == collections
    assert ax.xlim == (0, 10)
    assert ax.ylim == (0, 1)


def test_plot_training_size_distribution_uses_shared_histogram_layout():
    class FakeAxis:
        def __init__(self):
            self.hist_calls = []
            self.vlines = []
            self.xlabel = None
            self.ylabel = None
            self.title = None
            self.legend_called = False
            self.xlim = None

        def hist(self, sizes, **kwargs):
            self.hist_calls.append((sizes, kwargs))

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

        def set_xlim(self, start, end):
            self.xlim = (start, end)

    ax = FakeAxis()
    train._plot_training_size_distribution(
        ax,
        [10, 30],
        color="firebrick",
        x_label="Footprint Size (bp)",
        title_prefix="Footprint Sizes",
        include_nucleosome_marker=True,
    )

    sizes, hist_kwargs = ax.hist_calls[0]
    assert sizes == [10, 30]
    np.testing.assert_array_equal(hist_kwargs["bins"], [0, 10, 20, 30])
    assert hist_kwargs["color"] == "firebrick"
    assert hist_kwargs["alpha"] == 0.7
    assert hist_kwargs["edgecolor"] == "white"
    assert ax.vlines == [
        (
            20.0,
            {"color": "black", "linestyle": "--", "label": "Median: 20bp"},
        ),
        (
            147,
            {
                "color": "blue",
                "linestyle": ":",
                "alpha": 0.7,
                "label": "Nucleosome (147bp)",
            },
        ),
    ]
    assert ax.xlabel == "Footprint Size (bp)"
    assert ax.ylabel == "Count"
    assert ax.title == "Footprint Sizes (n=2)"
    assert ax.legend_called
    assert ax.xlim == (0, 500)

    empty_ax = FakeAxis()
    train._plot_training_size_distribution(
        empty_ax,
        [],
        color="forestgreen",
        x_label="MSP Size (bp)",
        title_prefix="MSP Sizes",
    )
    assert empty_ax.hist_calls == []
    assert empty_ax.vlines == []


def test_plot_training_emission_distribution_uses_nonzero_contexts():
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

    train._plot_training_emission_distribution(
        ax,
        np.array([[0.0, 0.2, 0.5], [0.1, 0.0, 0.3]]),
    )

    fp_values, fp_kwargs = ax.hist_calls[0]
    msp_values, msp_kwargs = ax.hist_calls[1]
    np.testing.assert_array_equal(fp_values, [0.2, 0.5])
    np.testing.assert_array_equal(msp_values, [0.1, 0.3])
    np.testing.assert_array_equal(fp_kwargs["bins"], np.linspace(0, 1, 51))
    np.testing.assert_array_equal(msp_kwargs["bins"], np.linspace(0, 1, 51))
    assert fp_kwargs["alpha"] == 0.6
    assert fp_kwargs["label"] == "Footprint (n=2)"
    assert fp_kwargs["color"] == "firebrick"
    assert msp_kwargs["alpha"] == 0.6
    assert msp_kwargs["label"] == "Accessible (n=2)"
    assert msp_kwargs["color"] == "forestgreen"
    assert ax.xlabel == "P(methylation | state, context)"
    assert ax.ylabel == "Number of contexts"
    assert ax.title == "Emission Probability Distribution"
    assert ax.legend_called


def test_plot_training_transition_matrix_formats_heatmap_and_annotations():
    transform = object()

    class FakeAxis:
        def __init__(self):
            self.transAxes = transform
            self.imshow_call = None
            self.xticks = None
            self.yticks = None
            self.xticklabels = None
            self.yticklabels = None
            self.xlabel = None
            self.ylabel = None
            self.title = None
            self.text_calls = []

        def imshow(self, matrix, **kwargs):
            self.imshow_call = (matrix, kwargs)

        def set_xticks(self, values):
            self.xticks = values

        def set_yticks(self, values):
            self.yticks = values

        def set_xticklabels(self, values):
            self.xticklabels = values

        def set_yticklabels(self, values):
            self.yticklabels = values

        def set_xlabel(self, value):
            self.xlabel = value

        def set_ylabel(self, value):
            self.ylabel = value

        def set_title(self, value):
            self.title = value

        def text(self, x, y, value, **kwargs):
            self.text_calls.append((x, y, value, kwargs))

    ax = FakeAxis()
    trans = np.array([[0.8, 0.2], [0.25, 0.75]])

    train._plot_training_transition_matrix(ax, trans)

    matrix, imshow_kwargs = ax.imshow_call
    np.testing.assert_array_equal(matrix, trans)
    assert imshow_kwargs == {"cmap": "Blues", "vmin": 0, "vmax": 1}
    assert ax.xticks == [0, 1]
    assert ax.yticks == [0, 1]
    assert ax.xticklabels == ["Footprint", "Accessible"]
    assert ax.yticklabels == ["Footprint", "Accessible"]
    assert ax.xlabel == "To State"
    assert ax.ylabel == "From State"
    assert ax.title == "Transition Probabilities"
    assert [(x, y, value, kwargs["color"]) for x, y, value, kwargs in ax.text_calls[:4]] == [
        (0, 0, "0.8000", "white"),
        (1, 0, "0.2000", "black"),
        (0, 1, "0.2500", "black"),
        (1, 1, "0.7500", "white"),
    ]
    assert ax.text_calls[4] == (
        0.5,
        -0.3,
        "Expected durations: Footprint=5bp, MSP=4bp",
        {"transform": transform, "ha": "center", "fontsize": 10},
    )


def test_training_stats_paths_are_under_plots_dir():
    assert train._training_stats_paths("/tmp/out") == {
        "plots_dir": "/tmp/out/plots",
        "pdf": "/tmp/out/plots/training_stats.pdf",
        "summary": "/tmp/out/plots/training_stats.txt",
    }


def test_write_training_stats_summary(tmp_path):
    model = SimpleNamespace(
        transmat_=np.array([[0.8, 0.2], [0.25, 0.75]]),
    )
    summary_path = tmp_path / "training_stats.txt"

    train._write_training_stats_summary(
        str(summary_path),
        model,
        emission_probs=np.array([[0.0, 0.2, 0.4], [0.1, 0.0, 0.9]]),
        sampled_reads=[object(), object()],
        all_footprint_sizes=[10, 30],
        all_msp_sizes=[20],
    )

    text = summary_path.read_text()
    assert "FiberHMM Training Statistics" in text
    assert "Footprint → Footprint: 0.800000" in text
    assert "Expected footprint duration: 5.0 bp" in text
    assert "Footprint contexts: 2 (mean=0.3000, median=0.3000)" in text
    assert "Reads used: 2" in text
    assert "Footprint sizes: mean=20.0, median=20.0, range=10-30" in text
    assert "MSP sizes: mean=20.0, median=20.0, range=20-20" in text


def test_save_training_model_parameter_page_orchestrates_plots(monkeypatch):
    axes = np.array([
        [object(), object()],
        [object(), object()],
    ], dtype=object)

    class FakeFigure:
        def __init__(self):
            self.suptitle_call = None

        def suptitle(self, *args, **kwargs):
            self.suptitle_call = (args, kwargs)

    class FakePlt:
        def __init__(self):
            self.fig = FakeFigure()
            self.subplots_call = None
            self.tight_layout_called = False
            self.closed = []

        def subplots(self, *args, **kwargs):
            self.subplots_call = (args, kwargs)
            return self.fig, axes

        def tight_layout(self):
            self.tight_layout_called = True

        def close(self, fig):
            self.closed.append(fig)

    class FakePdf:
        def __init__(self):
            self.saved = []

        def savefig(self, fig):
            self.saved.append(fig)

    calls = []

    monkeypatch.setattr(
        train,
        "_plot_training_transition_matrix",
        lambda ax, trans: calls.append(("trans", ax, trans)),
    )
    monkeypatch.setattr(
        train,
        "_plot_training_emission_distribution",
        lambda ax, emissions: calls.append(("emissions", ax, emissions)),
    )

    def fake_size_plot(ax, sizes, **kwargs):
        calls.append(("sizes", ax, sizes, kwargs))

    monkeypatch.setattr(train, "_plot_training_size_distribution", fake_size_plot)

    plt = FakePlt()
    pdf = FakePdf()
    model = SimpleNamespace(transmat_=np.array([[0.8, 0.2], [0.1, 0.9]]))
    emission_probs = np.array([[0.2], [0.3]])

    train._save_training_model_parameter_page(
        plt,
        pdf,
        model,
        emission_probs,
        [10, 20],
        [30, 40],
    )

    assert plt.subplots_call == ((2, 2), {"figsize": (11, 8.5)})
    assert plt.fig.suptitle_call == (
        ("FiberHMM Training Results",),
        {"fontsize": 14, "fontweight": "bold"},
    )
    assert calls[0][0:2] == ("trans", axes[0, 0])
    np.testing.assert_array_equal(calls[0][2], model.transmat_)
    assert calls[1][0:2] == ("emissions", axes[0, 1])
    np.testing.assert_array_equal(calls[1][2], emission_probs)
    assert calls[2] == (
        "sizes",
        axes[1, 0],
        [10, 20],
        {
            "color": "firebrick",
            "x_label": "Footprint Size (bp)",
            "title_prefix": "Footprint Sizes",
            "include_nucleosome_marker": True,
        },
    )
    assert calls[3] == (
        "sizes",
        axes[1, 1],
        [30, 40],
        {
            "color": "forestgreen",
            "x_label": "MSP Size (bp)",
            "title_prefix": "MSP Sizes",
        },
    )
    assert plt.tight_layout_called
    assert pdf.saved == [plt.fig]
    assert plt.closed == [plt.fig]


def test_save_training_example_png_orchestrates_panels(monkeypatch, tmp_path):
    class FakeAxis:
        def __init__(self):
            self.eventplots = []
            self.fill_between_calls = []
            self.hlines = []
            self.xlim = None
            self.ylim = None
            self.ylabel = None
            self.xlabel = None
            self.yticks = None
            self.xticklabels = None
            self.title = None

        def eventplot(self, *args, **kwargs):
            self.eventplots.append((args, kwargs))

        def fill_between(self, *args, **kwargs):
            self.fill_between_calls.append((args, kwargs))

        def axhline(self, *args, **kwargs):
            self.hlines.append((args, kwargs))

        def set_xlim(self, start, end):
            self.xlim = (start, end)

        def set_ylim(self, start, end):
            self.ylim = (start, end)

        def set_ylabel(self, value):
            self.ylabel = value

        def set_xlabel(self, value):
            self.xlabel = value

        def set_yticks(self, values):
            self.yticks = values

        def set_xticklabels(self, values):
            self.xticklabels = values

        def set_title(self, *args, **kwargs):
            self.title = (args, kwargs)

    class FakePlt:
        def __init__(self):
            self.fig = object()
            self.axes = [FakeAxis(), FakeAxis(), FakeAxis()]
            self.subplots_call = None
            self.tight_layout_called = False
            self.saved = []
            self.closed = []

        def subplots(self, *args, **kwargs):
            self.subplots_call = (args, kwargs)
            return self.fig, self.axes

        def tight_layout(self):
            self.tight_layout_called = True

        def savefig(self, *args, **kwargs):
            self.saved.append((args, kwargs))

        def close(self, fig):
            self.closed.append(fig)

    model = object()
    read = SimpleNamespace(read_id="read123")
    states = np.array([0, 1, 0])
    encoded = np.array([1, 2, 3])
    state_block_calls = []

    monkeypatch.setattr(
        train,
        "_training_example_plot_data",
        lambda model_arg, read_arg, encoded_arg: (
            4,
            [1, 3],
            np.array([0.1, 0.6, 0.2, 0.8]),
        ),
    )
    monkeypatch.setattr(
        train,
        "_add_training_state_blocks",
        lambda *args: state_block_calls.append(args),
    )

    plt = FakePlt()
    png_path = train._save_training_example_png(
        plt,
        str(tmp_path),
        model,
        read,
        states,
        encoded,
        "Rectangle",
        "PatchCollection",
    )

    assert png_path == str(tmp_path / "example_read.png")
    assert plt.subplots_call == (
        (3, 1),
        {"figsize": (14, 6), "gridspec_kw": {"height_ratios": [1, 1.5, 1]}},
    )
    assert plt.axes[0].eventplots == [
        (
            ([[1, 3]],),
            {
                "colors": "purple",
                "lineoffsets": 0.5,
                "linelengths": 0.8,
                "linewidths": 0.3,
            },
        )
    ]
    assert plt.axes[0].title == (("Example: read123... (4bp)",), {"fontsize": 10})
    assert state_block_calls[0][0] is plt.axes[1]
    np.testing.assert_array_equal(state_block_calls[0][1], states)
    assert state_block_calls[0][2:] == (4, "Rectangle", "PatchCollection")
    assert plt.axes[2].fill_between_calls[0][0][0] == range(4)
    np.testing.assert_array_equal(
        plt.axes[2].fill_between_calls[0][0][2],
        [0.1, 0.6, 0.2, 0.8],
    )
    assert plt.axes[2].hlines == [
        ((0.5,), {"color": "gray", "linestyle": "--", "linewidth": 0.5})
    ]
    assert plt.tight_layout_called
    assert plt.saved == [((png_path,), {"dpi": 150})]
    assert plt.closed == [plt.fig]


def test_save_training_example_pdf_page_writes_pdf(tmp_path):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Patch, Rectangle

    class FakeModel:
        def predict_proba(self, encoded):
            footprint = np.linspace(0.1, 0.9, len(encoded))
            return np.column_stack([footprint, 1 - footprint])

    read = SimpleNamespace(
        read_id="read-1",
        query_sequence="A" * 50,
        m6a_query_positions={1, 10, 49},
        chrom="chr2L",
        ref_start=100,
        ref_end=150,
    )
    states = np.array([0, 0, 1, 1, 0] * 10)
    encoded = np.arange(50)
    pdf_path = tmp_path / "example.pdf"

    with PdfPages(str(pdf_path)) as pdf:
        train._save_training_example_pdf_page(
            plt,
            pdf,
            FakeModel(),
            read,
            states,
            encoded,
            0,
            Rectangle,
            PatchCollection,
            Patch,
        )

    plt.close("all")
    assert pdf_path.stat().st_size > 0


def test_save_training_stats_pdf_pages_orchestrates_model_and_examples(monkeypatch):
    calls = []
    monkeypatch.setattr(
        train,
        "_save_training_model_parameter_page",
        lambda *args: calls.append(("model", args)),
    )
    monkeypatch.setattr(
        train,
        "_save_training_example_pdf_page",
        lambda *args: calls.append(("example", args)),
    )
    plt = object()
    pdf = object()
    model = object()
    emission_probs = np.ones((2, 4))
    sampled_reads = ["read-a", "read-b", "read-c"]
    encoded_reads = ["enc-a", "enc-b"]
    all_states = ["states-a", "states-b"]

    n_plotted = train._save_training_stats_pdf_pages(
        plt,
        pdf,
        model,
        emission_probs,
        sampled_reads,
        encoded_reads,
        all_states,
        [10],
        [20],
        n_examples=5,
        rectangle_cls="Rectangle",
        patch_collection_cls="PatchCollection",
        patch_cls="Patch",
    )

    assert n_plotted == 2
    assert calls[0] == (
        "model",
        (plt, pdf, model, emission_probs, [10], [20]),
    )
    assert calls[1] == (
        "example",
        (
            plt,
            pdf,
            model,
            "read-a",
            "states-a",
            "enc-a",
            0,
            "Rectangle",
            "PatchCollection",
            "Patch",
        ),
    )
    assert calls[2][1][3:7] == ("read-b", "states-b", "enc-b", 1)


def test_save_training_stats_example_png_writes_only_when_examples_exist(
    monkeypatch,
    capsys,
):
    calls = []
    monkeypatch.setattr(
        train,
        "_save_training_example_png",
        lambda *args: calls.append(args) or "/tmp/example.png",
    )

    train._save_training_stats_example_png(
        "plt",
        "/tmp/plots",
        "model",
        ["read-a"],
        ["enc-a"],
        ["states-a"],
        "Rectangle",
        "PatchCollection",
    )
    train._save_training_stats_example_png(
        "plt",
        "/tmp/plots",
        "model",
        [],
        [],
        [],
        "Rectangle",
        "PatchCollection",
    )

    assert calls == [(
        "plt",
        "/tmp/plots",
        "model",
        "read-a",
        "states-a",
        "enc-a",
        "Rectangle",
        "PatchCollection",
    )]
    assert "Saved: /tmp/example.png" in capsys.readouterr().out


def test_write_training_stats_summary_report_delegates_and_reports(monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(
        train,
        "_write_training_stats_summary",
        lambda *args: calls.append(args),
    )

    train._write_training_stats_summary_report(
        "summary.txt",
        "model",
        "emissions",
        ["read-a"],
        [10],
        [20],
    )

    assert calls == [("summary.txt", "model", "emissions", ["read-a"], [10], [20])]
    assert "Saved: summary.txt" in capsys.readouterr().out


def test_training_config_includes_base_model_only_when_used():
    args = SimpleNamespace(
        context_size=3,
        mode="daf",
        edge_trim=10,
        prob_adjust=1.25,
        base_model=None,
    )

    assert train._training_config(args) == {
        "context_size": 3,
        "mode": "daf",
        "edge_trim": 10,
        "prob_adjust": 1.25,
    }

    args.base_model = "base-model.json"
    assert train._training_config(args)["base_model"] == "base-model.json"


def test_training_output_paths_use_canonical_filenames():
    assert train._training_output_paths("/tmp/out") == {
        "best_json": "/tmp/out/best-model.json",
        "best_npz": "/tmp/out/best-model.npz",
        "all_models": "/tmp/out/all_models.json",
        "train_reads": "/tmp/out/training-reads.tsv",
        "config": "/tmp/out/model_config.json",
    }


def test_model_json_record_uses_plain_lists():
    model = SimpleNamespace(
        n_states=2,
        startprob_=np.array([0.4, 0.6]),
        transmat_=np.array([[0.9, 0.1], [0.2, 0.8]]),
        emissionprob_=np.array([[0.1, 0.2], [0.8, 0.9]]),
    )

    record = train._model_json_record(model)

    assert record == {
        "n_states": 2,
        "startprob": [0.4, 0.6],
        "transmat": [[0.9, 0.1], [0.2, 0.8]],
        "emissionprob": [[0.1, 0.2], [0.8, 0.9]],
    }


def test_load_training_emission_probs_delegates_and_reports_shape(monkeypatch, capsys):
    calls = []
    emission_probs = np.ones((2, 4))
    monkeypatch.setattr(
        train,
        "make_emission_probs",
        lambda *args, **kwargs: calls.append((args, kwargs)) or emission_probs,
    )
    args = SimpleNamespace(
        probs=["accessible.tsv", "inaccessible.tsv"],
        context_size=3,
        prob_adjust=1.25,
    )

    assert train._load_training_emission_probs(args) is emission_probs
    assert calls == [(
        ("accessible.tsv", "inaccessible.tsv"),
        {"context_size": 3, "prob_adjust": 1.25},
    )]
    assert "Emission matrix: (2, 4)" in capsys.readouterr().out


def test_run_training_or_base_model_uses_base_model_path(monkeypatch):
    best_model = object()
    monkeypatch.setattr(
        train,
        "_build_model_from_base",
        lambda path, emissions, context_size: (best_model, [best_model]),
    )
    args = SimpleNamespace(base_model="base.json", context_size=3)
    emission_probs = np.ones((2, 4))

    assert train._run_training_or_base_model(args, emission_probs) == (
        best_model,
        [best_model],
        [],
        [],
        [],
    )


def test_run_training_or_base_model_samples_encodes_and_trains(monkeypatch, capsys):
    sampled = ["read-a", "read-b"]
    train_arrays = [np.array([1, 2])]
    train_rids = ["read-a"]
    encoded_reads = ["encoded-a"]
    valid_reads = ["read-a"]
    best_model = object()
    calls = []
    monkeypatch.setattr(
        train,
        "sample_reads",
        lambda *args, **kwargs: calls.append(("sample", args, kwargs)) or sampled,
    )
    monkeypatch.setattr(
        train,
        "generate_training_arrays",
        lambda *args: calls.append(("arrays", args)) or (
            train_arrays,
            train_rids,
            encoded_reads,
            valid_reads,
        ),
    )
    monkeypatch.setattr(
        train,
        "train_hmm",
        lambda *args: calls.append(("train", args)) or (best_model, [best_model]),
    )
    args = SimpleNamespace(
        base_model=None,
        input=["a.bam", "b.bam"],
        read_count=100,
        seed=7,
        mode="daf",
        min_mapq=20,
        prob_threshold=128,
        min_read_length=1000,
        edge_trim=10,
        iterations=5,
        context_size=3,
        use_hmmlearn=False,
    )
    emission_probs = np.ones((2, 4))

    assert train._run_training_or_base_model(args, emission_probs) == (
        best_model,
        [best_model],
        train_rids,
        valid_reads,
        encoded_reads,
    )
    assert calls[0] == (
        "sample",
        (["a.bam", "b.bam"], 100, 7),
        {
            "mode": "daf",
            "min_mapq": 20,
            "prob_threshold": 128,
            "min_read_length": 1000,
        },
    )
    assert calls[1] == (
        "arrays",
        (sampled, 10, 5, "daf", 3),
    )
    assert calls[2] == ("train", (emission_probs, train_arrays, False))
    out = capsys.readouterr().out
    assert "Sampling reads from 2 BAM file(s)" in out
    assert "Training HMM (5 iterations)" in out


def test_maybe_generate_training_stats_handles_data_and_base_skip(monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(
        train,
        "generate_training_stats",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    model = object()
    emission_probs = np.ones((2, 4))
    args = SimpleNamespace(
        stats=True,
        base_model=None,
        outdir="out",
        n_examples=3,
        mode="daf",
    )

    train._maybe_generate_training_stats(
        args,
        model,
        valid_reads=["read-a"],
        encoded_reads=["encoded-a"],
        emission_probs=emission_probs,
    )
    assert calls == [(
        (model, ["read-a"], ["encoded-a"], emission_probs, "out"),
        {"n_examples": 3, "mode": "daf"},
    )]
    assert "Generating training statistics" in capsys.readouterr().out

    args.base_model = "base.json"
    train._maybe_generate_training_stats(
        args,
        model,
        valid_reads=[],
        encoded_reads=[],
        emission_probs=emission_probs,
    )
    assert "stats skipped" in capsys.readouterr().out


def test_training_chrom_is_sampleable_filters_short_and_scaffold_names():
    assert train._training_chrom_is_sampleable("chr2L", 1_000_000)
    assert not train._training_chrom_is_sampleable("tiny", 99_999)
    assert not train._training_chrom_is_sampleable("chrUn_random", 1_000_000)
    assert not train._training_chrom_is_sampleable("CHR_ALT_1", 1_000_000)


def test_training_sampling_chrom_lengths_filters_scaffolds_and_falls_back():
    ref_lengths = {
        "chr1": 1_000_000,
        "chrUn_random": 2_000_000,
        "tiny": 10_000,
    }

    assert train._training_sampling_chrom_lengths(ref_lengths) == {"chr1": 1_000_000}
    assert train._training_sampling_chrom_lengths({"tiny": 10_000}) == {
        "tiny": 10_000,
    }


def test_chrom_pos_from_genome_offset_maps_cumulative_lengths():
    chroms = ["chr1", "chr2", "chr3"]
    cum_lengths = np.array([100, 250, 300])

    assert train._chrom_pos_from_genome_offset(chroms, cum_lengths, 50) == (
        "chr1", 50,
    )
    assert train._chrom_pos_from_genome_offset(chroms, cum_lengths, 125) == (
        "chr2", 25,
    )


def test_training_sampling_index_filters_and_accumulates_lengths():
    chroms, cum_lengths, total_length = train._training_sampling_index({
        "chr1": 1_000_000,
        "chrUn_random": 2_000_000,
        "chr2": 3_000_000,
    })

    assert chroms == ["chr1", "chr2"]
    np.testing.assert_array_equal(cum_lengths, [1_000_000, 4_000_000])
    assert total_length == 4_000_000


def test_training_sample_filter_helper_applies_alignment_filters():
    def read(**overrides):
        attrs = {
            "is_unmapped": False,
            "is_secondary": False,
            "is_supplementary": False,
            "mapping_quality": 30,
            "query_sequence": "ACGT",
            "reference_start": 10,
            "reference_end": 14,
        }
        attrs.update(overrides)
        return SimpleNamespace(**attrs)

    assert train._passes_training_sample_filters(read(), 20, 4) is True
    assert train._passes_training_sample_filters(
        read(mapping_quality=19), 20, 4,
    ) is False
    assert train._passes_training_sample_filters(
        read(query_sequence=None), 20, 4,
    ) is False
    assert train._passes_training_sample_filters(
        read(reference_end=13), 20, 4,
    ) is False
    assert train._training_reference_span(read(reference_start=None)) is None
    assert train._passes_training_sample_filters(
        read(reference_start=None), 20, 4,
    ) is False
    assert train._passes_training_sample_filters(
        read(is_supplementary=True), 20, 4,
    ) is False


def test_training_mod_query_positions_prefers_pysam_then_mm_fallback(monkeypatch):
    class TaggedRead:
        query_sequence = "AAAA"
        is_reverse = False

        def __init__(self, tags):
            self.tags = tags

        def has_tag(self, tag):
            return tag in self.tags

        def get_tag(self, tag):
            return self.tags[tag]

    read = TaggedRead({"MM": "A+a,0;", "ML": b"\xc8"})

    monkeypatch.setattr(bam_reader, "get_modified_positions_pysam", lambda *args: {3})
    monkeypatch.setattr(
        bam_reader,
        "parse_mm_tag_query_positions",
        lambda *args, **kwargs: pytest.fail("MM/ML fallback should not run"),
    )
    assert train._training_mod_query_positions(read, 125, "pacbio-fiber") == {3}

    captured = {}

    def fake_parse(mm_tag, ml_tag, sequence, is_reverse, prob_threshold, mode):
        captured["ml_tag"] = ml_tag
        return {1}

    monkeypatch.setattr(bam_reader, "get_modified_positions_pysam", lambda *args: set())
    monkeypatch.setattr(bam_reader, "parse_mm_tag_query_positions", fake_parse)

    assert train._training_mod_query_positions(read, 125, "pacbio-fiber") == {1}
    assert train._training_mm_ml_query_positions(read, 125, "pacbio-fiber") == {1}
    assert captured["ml_tag"] == b"\xc8"


def test_training_sample_candidate_filters_duplicates_and_requires_mods(monkeypatch):
    read = SimpleNamespace(query_name="read1")
    fiber = object()

    monkeypatch.setattr(
        train,
        "_passes_training_sample_filters",
        lambda got_read, min_mapq, min_read_length: got_read is read,
    )
    monkeypatch.setattr(
        train,
        "_training_mod_query_positions",
        lambda got_read, prob_threshold, mode: {2},
    )
    monkeypatch.setattr(
        train,
        "_training_fiber_read_from_segment",
        lambda got_read, mods: fiber,
    )

    assert train._training_sample_candidate(
        read, set(), 20, 1000, 125, "pacbio-fiber",
    ) is fiber
    assert train._training_sample_candidate(
        read, {"read1"}, 20, 1000, 125, "pacbio-fiber",
    ) is None
    assert train._training_sample_candidate(
        SimpleNamespace(query_name="read2"), set(), 20, 1000, 125, "pacbio-fiber",
    ) is None

    monkeypatch.setattr(
        train,
        "_training_mod_query_positions",
        lambda got_read, prob_threshold, mode: set(),
    )
    assert train._training_sample_candidate(
        read, set(), 20, 1000, 125, "pacbio-fiber",
    ) is None


def test_sample_training_read_at_position_returns_first_valid_candidate(monkeypatch):
    reads = [SimpleNamespace(query_name="skip"), SimpleNamespace(query_name="keep")]
    fiber = object()

    class FakeBam:
        def fetch(self, chrom, start, end):
            assert (chrom, start, end) == ("chr1", 0, 150)
            return iter(reads)

    def fake_candidate(read, seen, min_mapq, min_read_length, prob_threshold, mode):
        assert seen == set()
        assert (min_mapq, min_read_length, prob_threshold, mode) == (
            20,
            1000,
            125,
            "pacbio-fiber",
        )
        return fiber if read.query_name == "keep" else None

    monkeypatch.setattr(train, "_training_sample_candidate", fake_candidate)
    seen = set()

    assert train._sample_training_read_at_position(
        FakeBam(), "chr1", 50, seen, 20, 1000, 125, "pacbio-fiber",
    ) is fiber
    assert seen == {"keep"}


def test_sample_training_read_at_position_handles_invalid_region():
    class FakeBam:
        def fetch(self, chrom, start, end):
            raise ValueError("bad region")

    assert train._sample_training_read_at_position(
        FakeBam(), "chrMissing", 10, set(), 20, 1000, 125, "pacbio-fiber",
    ) is None


def test_sample_training_reads_for_positions_appends_until_target(monkeypatch):
    sampled = []
    seen = set()
    calls = []

    def fake_sample(
        bam,
        chrom,
        pos,
        seen_read_ids,
        min_mapq,
        min_read_length,
        prob_threshold,
        mode,
    ):
        calls.append((chrom, pos, seen_read_ids, min_mapq, min_read_length,
                      prob_threshold, mode))
        return f"{chrom}:{pos}"

    monkeypatch.setattr(train, "_sample_training_read_at_position", fake_sample)

    attempts = train._sample_training_reads_for_positions(
        object(),
        np.asarray([50, 125, 175]),
        sampled,
        2,
        ["chr1", "chr2"],
        np.asarray([100, 200]),
        seen,
        20,
        1000,
        125,
        "pacbio-fiber",
    )

    assert attempts == 2
    assert sampled == ["chr1:50", "chr2:25"]
    assert [call[:2] for call in calls] == [("chr1", 50), ("chr2", 25)]
    assert all(call[2] is seen for call in calls)


def test_reads_per_training_file_has_one_read_minimum():
    assert train._reads_per_training_file(100, 4) == 25
    assert train._reads_per_training_file(2, 5) == 1


def test_top_up_training_reads_truncates_or_samples_with_replacement():
    assert train._top_up_training_reads([], 3) == []
    assert train._top_up_training_reads(["a", "b", "c"], 2) == ["a", "b"]

    np.random.seed(0)
    topped = train._top_up_training_reads(["a", "b"], 5)

    assert len(topped) == 5
    assert topped[:2] == ["a", "b"]
    assert set(topped) <= {"a", "b"}


def test_sample_training_reads_from_file_delegates_and_reports(monkeypatch, capsys):
    calls = []

    monkeypatch.setattr(
        train,
        "sample_reads_indexed",
        lambda *args, **kwargs: calls.append((args, kwargs)) or ["read1", "read2"],
    )

    reads = train._sample_training_reads_from_file(
        "/tmp/example.bam",
        reads_per_file=3,
        seed=7,
        mode="daf",
        sample_kwargs={"min_mapq": 10},
    )

    assert reads == ["read1", "read2"]
    assert calls == [
        (("/tmp/example.bam", 3, 7), {"mode": "daf", "min_mapq": 10}),
    ]
    assert "Sampled 2 from example.bam" in capsys.readouterr().out


def test_build_model_from_base_copies_transitions_and_replaces_emissions(monkeypatch):
    base_model = SimpleNamespace(
        startprob_=np.array([0.7, 0.3]),
        transmat_=np.array([[0.95, 0.05], [0.1, 0.9]]),
        emissionprob_=np.zeros((2, 4)),
    )
    emission_probs = np.full((2, 4), 0.25)

    monkeypatch.setattr(
        train,
        "load_model",
        lambda path, normalize=False: base_model,
    )

    model, all_models = train._build_model_from_base(
        "base-model.json",
        emission_probs,
        context_size=3,
    )

    assert all_models == [model]
    np.testing.assert_array_equal(model.startprob_, [0.7, 0.3])
    np.testing.assert_array_equal(model.transmat_, [[0.95, 0.05], [0.1, 0.9]])
    assert model.emissionprob_ is emission_probs

    base_model.startprob_[0] = 0.1
    base_model.transmat_[0, 0] = 0.1
    assert model.startprob_[0] == 0.7
    assert model.transmat_[0, 0] == 0.95


def test_sample_reads_indexed_preserves_reverse_flag(monkeypatch):
    class FakeRead:
        is_unmapped = False
        is_secondary = False
        is_supplementary = False
        mapping_quality = 60
        query_sequence = "ACGT" * 300
        reference_start = 100
        reference_end = reference_start + len(query_sequence)
        query_name = "reverse-read"
        reference_name = "chr1"
        is_reverse = True

        def has_tag(self, tag):
            return False

        def get_tag(self, tag):
            raise KeyError(tag)

        def get_aligned_pairs(self):
            return [
                (query_pos, self.reference_start + query_pos)
                for query_pos in range(len(self.query_sequence))
            ]

    class FakeBam:
        references = ("chr1",)
        lengths = (2000,)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def fetch(self, chrom, start, end):
            return iter([FakeRead()])

    monkeypatch.setattr(pysam, "AlignmentFile", lambda *args, **kwargs: FakeBam())
    monkeypatch.setattr(bam_reader, "get_modified_positions_pysam", lambda *args, **kwargs: {5})

    reads = train.sample_reads_indexed(
        "fake.bam",
        n_samples=1,
        seed=1,
        min_mapq=0,
        min_read_length=0,
    )

    assert len(reads) == 1
    assert reads[0].strand == "-"
    assert reads[0].is_reverse is True
