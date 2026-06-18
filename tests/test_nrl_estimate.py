"""Tests for phase-NRL estimate post-processing helpers."""

from types import SimpleNamespace

import numpy as np

from fiberhmm.inference import nrl_estimate


def test_phase_nrl_peak_spacings_filters_primary_repeat_window():
    np.testing.assert_array_equal(
        nrl_estimate._phase_nrl_peak_spacings([90, 120, 185, 260, 300]),
        np.array([120, 185, 260], dtype=np.float64),
    )


def test_nuc_center_spacings_sorts_calls_and_uses_centers():
    nucs = [
        SimpleNamespace(start=260, length=80),
        SimpleNamespace(start=10, length=100),
        SimpleNamespace(start=110, length=90),
    ]

    assert nrl_estimate._nuc_center_spacings(nucs) == [95.0, 145.0]


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


def test_phase_nrl_spacings_for_read_skips_non_primary_alignment():
    read = SimpleNamespace(
        is_unmapped=True,
        is_secondary=False,
        is_supplementary=False,
    )

    assert nrl_estimate._phase_nrl_spacings_for_read(
        read,
        model=object(),
        llr_hit=object(),
        llr_miss=object(),
        mode="daf",
        context_size=3,
        split_min_llr=4.0,
        split_min_opps=3,
        nuc_min_size=85,
        msp_min_size=0,
        prob_threshold=128,
        edge_trim=10,
    ) is None


def test_phase_nrl_spacings_for_read_runs_apply_and_recall(monkeypatch):
    read = SimpleNamespace(
        is_unmapped=False,
        is_secondary=False,
        is_supplementary=False,
    )
    model = object()
    llr_hit = object()
    llr_miss = object()
    fiber_read = {"query_sequence": "ACGT"}
    apply_result = {
        "encoded": np.zeros(300, dtype=np.int32),
        "ns": np.asarray([10], dtype=np.int32),
        "nl": np.asarray([100], dtype=np.int32),
    }
    nucs = [
        SimpleNamespace(start=10, length=100),
        SimpleNamespace(start=190, length=80),
    ]
    seen = {}

    def fake_extract(*args):
        seen["extract"] = args
        return fiber_read

    def fake_apply(*args):
        seen["apply"] = args
        return apply_result

    def fake_recall(*args, **kwargs):
        seen["recall"] = (args, kwargs)
        return nucs, []

    monkeypatch.setattr(nrl_estimate, "_extract_fiber_read_from_pysam", fake_extract)
    monkeypatch.setattr(nrl_estimate, "run_hmm_apply_stage", fake_apply)
    monkeypatch.setattr(nrl_estimate, "recall_nucs_in_read", fake_recall)

    assert nrl_estimate._phase_nrl_spacings_for_read(
        read,
        model,
        llr_hit,
        llr_miss,
        mode="daf",
        context_size=3,
        split_min_llr=4.0,
        split_min_opps=3,
        nuc_min_size=85,
        msp_min_size=0,
        prob_threshold=128,
        edge_trim=10,
    ) == [170.0]
    assert seen["extract"] == (read, "daf", 128)
    assert seen["apply"] == (
        fiber_read, model, 10, False, "daf", 3, 0, 85, False,
    )
    assert seen["recall"][0] == (
        apply_result["encoded"],
        apply_result["ns"],
        apply_result["nl"],
        300,
        llr_hit,
        llr_miss,
    )
    assert seen["recall"][1] == {
        "split_min_llr": 4.0,
        "split_min_opps": 3,
        "nuc_min_size": 85,
    }
