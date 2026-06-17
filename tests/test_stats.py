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

    starts, lengths = stats_module._flipped_interval_tag_arrays(
        read, "ns", "nl", (None, None),
    )
    missing = stats_module._flipped_interval_tag_arrays(
        read, "as", "al", ("missing", "missing"),
    )
    scores = stats_module._scaled_score_tag(read, "nq")

    np.testing.assert_array_equal(starts, [10])
    np.testing.assert_array_equal(lengths, [5])
    assert missing == ("missing", "missing")
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

    np.testing.assert_array_equal(without_scores[0], [10])
    np.testing.assert_array_equal(without_scores[1], [5])
    np.testing.assert_array_equal(without_scores[2], [20])
    np.testing.assert_array_equal(without_scores[3], [7])
    assert without_scores[4:] == (None, None)
    np.testing.assert_allclose(with_scores[4], [1.0])
    np.testing.assert_allclose(with_scores[5], [128 / 255])


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


def test_stats_sampling_probability_handles_full_and_partial_samples():
    assert stats_module._stats_sampling_probability(10, 100) == 1.0
    assert stats_module._stats_sampling_probability(100, 10) == 0.1
    assert stats_module._stats_sampling_probability(100, 0) == 0.0
