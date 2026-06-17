from __future__ import annotations

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


def test_stats_sampling_probability_handles_full_and_partial_samples():
    assert stats_module._stats_sampling_probability(10, 100) == 1.0
    assert stats_module._stats_sampling_probability(100, 10) == 0.1
    assert stats_module._stats_sampling_probability(100, 0) == 0.0
