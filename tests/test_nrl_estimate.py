"""Tests for phase-NRL estimate post-processing helpers."""

import numpy as np

from fiberhmm.inference import nrl_estimate


def test_phase_nrl_peak_spacings_filters_primary_repeat_window():
    np.testing.assert_array_equal(
        nrl_estimate._phase_nrl_peak_spacings([90, 120, 185, 260, 300]),
        np.array([120, 185, 260], dtype=np.float64),
    )


def test_phase_nrl_result_uses_anchor_when_peak_has_too_few_pairs():
    assert nrl_estimate._phase_nrl_result(
        [90, 185, 300],
        n_reads=12,
        anchor=185,
        clamp_lo=150,
        clamp_hi=215,
        min_pairs=2,
    ) == {
        "nrl": 185,
        "ci": None,
        "n_pairs": 1,
        "n_reads": 12,
        "source": "anchor",
    }


def test_phase_nrl_result_estimates_and_clamps_from_peak_spacings():
    result = nrl_estimate._phase_nrl_result(
        [90, 180, 182, 184, 185, 186, 188, 300],
        n_reads=4,
        anchor=185,
        clamp_lo=150,
        clamp_hi=215,
        min_pairs=3,
    )

    assert result["nrl"] == 185
    assert result["n_pairs"] == 6
    assert result["n_reads"] == 4
    assert result["source"] == "estimated"
    assert result["ci"][0] < result["ci"][1]
