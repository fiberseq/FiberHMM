"""Peak smoothing (3.2.1/3.3): the boxcar mean keeps np.convolve(mode='same')
window semantics, keeps the signal length, and sums in float64, so a window
sum is the exact sum of its float32 inputs (no platform-dependent rounding).
"""
from __future__ import annotations

import numpy as np
import pytest

from fiberhmm.scan.peaks_core import smooth_scan_signal


def _reference(signal, w):
    """np.convolve(mode='same') in float64 with the kernel clamped to the signal length."""
    n = len(signal)
    w = min(max(1, int(w)), n)
    return np.convolve(np.asarray(signal, dtype=np.float64), np.ones(w) / w, mode='same')


@pytest.mark.parametrize('n', [1, 2, 7, 50])
@pytest.mark.parametrize('w', [0, 1, 2, 3, 4, 9, 10, 51, 1000])
def test_matches_convolve_same_window_semantics(n, w):
    rng = np.random.default_rng(1000 * n + w)
    signal = rng.integers(0, 40, n).astype(np.float32) / np.float32(40)
    got = smooth_scan_signal(signal, w)
    assert got.dtype == np.float32 and got.shape == signal.shape
    want = _reference(signal, w)
    # the result is the float64 value rounded once to float32
    np.testing.assert_array_equal(got, want.astype(np.float32))


def test_even_width_window_is_left_heavy_like_convolve():
    signal = np.zeros(9, dtype=np.float32)
    signal[4] = 1
    # w=4: output k averages signal[k-2 : k+2]
    got = smooth_scan_signal(signal, 4)
    np.testing.assert_array_equal(np.nonzero(got)[0], [3, 4, 5, 6])
    assert np.all(got[[3, 4, 5, 6]] == np.float32(0.25))


def test_empty_signal_and_width_clamp():
    empty = smooth_scan_signal(np.zeros(0, dtype=np.float32), 5)
    assert empty.dtype == np.float32 and empty.size == 0
    ones = np.ones(3, dtype=np.float32)
    # a kernel wider than the viewport is clamped to it: length kept, divisor = n
    np.testing.assert_array_equal(smooth_scan_signal(ones, 50), np.array([2 / 3, 1, 2 / 3], dtype=np.float32))


def test_window_sums_are_exact_float32_sums_where_float32_accumulation_rounds():
    # 1/coverage ratios over a long viewport: a float32 running sum drifts, the float64 prefix sum does not.
    coverage = 37
    rng = np.random.default_rng(7)
    signal = (rng.integers(0, coverage + 1, 200_000) / coverage).astype(np.float32)
    w = 151
    got = smooth_scan_signal(signal, w)
    for k in (75, 1_000, 123_456, 199_924):
        lo = k - w // 2
        exact = float(np.sum(signal[lo:lo + w].astype(np.float64))) / w
        assert got[k] == np.float32(exact), k
