from __future__ import annotations

import numpy as np
import pytest

from fiberhmm.inference import stats as stats_module


class _FakeRead:
    is_unmapped = False
    is_secondary = False
    is_supplementary = False
    query_length = 100


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


def test_positive_gaps_between_intervals_sorts_and_skips_overlaps():
    gaps = stats_module._positive_gaps_between_intervals(
        np.array([50, 0, 20]),
        np.array([10, 15, 40]),
    )

    assert gaps == [5]
