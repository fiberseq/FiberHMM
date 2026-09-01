"""Tests for fused apply/recall stage boundaries."""

from __future__ import annotations

import numpy as np
import pytest
from types import SimpleNamespace

from fiberhmm.inference import fused_stages
from fiberhmm.inference.nuc_recaller import NucCall
from fiberhmm.inference.tf_recaller import TFCall


def test_apply_result_has_footprints_detects_nucs_or_msps():
    empty = {
        "ns": np.asarray([], dtype=np.int32),
        "as": np.asarray([], dtype=np.int32),
    }
    nuc_only = {
        "ns": np.asarray([10], dtype=np.int32),
        "as": np.asarray([], dtype=np.int32),
    }
    msp_only = {
        "ns": np.asarray([], dtype=np.int32),
        "as": np.asarray([20], dtype=np.int32),
    }

    assert fused_stages.apply_result_has_footprints(None) is False
    assert fused_stages.apply_result_has_footprints(empty) is False
    assert fused_stages.apply_result_has_footprints(nuc_only) is True
    assert fused_stages.apply_result_has_footprints(msp_only) is True


def test_nuc_derived_tf_edge_gate_preserves_original_scan_space():
    calls = [
        TFCall(10, 15, 6.0, 3, 40, 40),
        TFCall(60, 15, 6.0, 3, 12, 12),
        TFCall(90, 15, 6.0, 3, 13, 2),
    ]

    obs = np.full(120, 4096, dtype=np.int32)
    obs[[47, 87, 118]] = 0
    kept = fused_stages.filter_nuc_derived_tf_calls(
        calls,
        original_scan_intervals=[(0, 30)],
        obs=obs,
        max_edge_ambiguity=12,
    )

    # The first call keeps the ordinary recaller contract because its centre
    # was HMM-accessible. Newly exposed calls require a nearby hit on both
    # sides, so 12/12 passes and 13/2 fails.
    assert kept == calls[:2]
    assert kept[0] is calls[0]
    assert (kept[1].left_ambiguity, kept[1].right_ambiguity) == (12, 12)


def test_nuc_derived_tf_edge_gate_rewrites_full_molecule_ambiguity():
    call = TFCall(60, 15, 6.0, 3, 255, 255)
    obs = np.full(100, 4096, dtype=np.int32)
    obs[[58, 77]] = 0

    kept = fused_stages.filter_nuc_derived_tf_calls(
        [call], original_scan_intervals=[], obs=obs, max_edge_ambiguity=12,
    )

    assert len(kept) == 1
    assert (kept[0].left_ambiguity, kept[0].right_ambiguity) == (1, 2)


def test_nuc_derived_tf_edge_gate_can_be_disabled():
    calls = [TFCall(90, 15, 6.0, 3, 255, 255)]

    assert fused_stages.filter_nuc_derived_tf_calls(
        calls, original_scan_intervals=[], obs=np.zeros(1),
        max_edge_ambiguity=None,
    ) == calls


def test_circular_derived_tf_gate_requires_tiled_hmm_coordinates():
    apply_result = {
        "ns": np.asarray([], dtype=np.int32),
        "nl": np.asarray([], dtype=np.int32),
        "as": np.asarray([], dtype=np.int32),
        "al": np.asarray([], dtype=np.int32),
        "encoded": np.zeros(300, dtype=np.int32),
        "circular": True,
        "circular_read_length": 100,
    }

    with pytest.raises(ValueError, match="requires tiled HMM"):
        fused_stages.build_fused_recall_result(
            {"query_sequence": "A" * 100},
            apply_result,
            llr_hit="hit",
            llr_miss="miss",
            min_llr=4.0,
            min_opps=3,
            unify_threshold=90,
            with_scores=True,
            recall_nucs=True,
            nuc_profile=object(),
            derived_tf_max_edge_ambiguity=12,
        )


def test_run_ddda_mcg_stage_excludes_apply_nucs_and_builds_mask(monkeypatch):
    from fiberhmm.daf import m5c

    captured = {}

    def fake_call(observations, factors, **kwargs):
        captured["observations"] = observations
        captured["factors"] = factors
        captured["kwargs"] = kwargs
        return SimpleNamespace(calls=(SimpleNamespace(start=30, end=51),))

    monkeypatch.setattr(m5c, "call_read_m5c", fake_call)
    payload = {
        "query_pos": np.array([10, 20, 30, 40], dtype=np.int32),
        "reference_pos": np.array([110, 120, 130, 140], dtype=np.int64),
        "is_cpg": np.array([False, True, False, True]),
        "deaminated": np.array([True, False, True, False]),
        "five_prime_base": np.zeros(4, dtype=np.int8),
    }
    apply_result = {
        "ns": np.array([0], dtype=np.int32),
        "nl": np.array([25], dtype=np.int32),
    }
    mask, spans = fused_stages.run_ddda_mcg_stage(
        payload, apply_result, read_length=100,
    )
    assert [obs.query_pos for obs in captured["observations"]] == [30, 40]
    assert spans == [(30, 51)]
    assert mask.sum() == 21
    assert mask[30] and mask[50] and not mask[51]


def test_build_fused_recall_result_runs_recall_and_aligns_kept_scores(monkeypatch):
    seen = {"interval_args": None, "scan_args": []}

    def fake_build_scan_intervals(
        ns, nl, msps, msp_lengths, read_length, unify_threshold
    ):
        seen["interval_args"] = (ns, nl, msps, msp_lengths, read_length, unify_threshold)
        return [(10, 30), (100, 130)]

    def fake_call_tfs_in_interval(obs, lo, hi, llr_hit, llr_miss, min_llr, min_opps):
        seen["scan_args"].append((obs, lo, hi, llr_hit, llr_miss, min_llr, min_opps))
        if lo == 10:
            return [
                TFCall(
                    start=8,
                    length=10,
                    llr=6.0,
                    n_opps=4,
                    left_ambiguity=1,
                    right_ambiguity=2,
                )
            ]
        return []

    monkeypatch.setattr(fused_stages, "build_scan_intervals", fake_build_scan_intervals)
    monkeypatch.setattr(fused_stages, "call_tfs_in_interval", fake_call_tfs_in_interval)

    obs = np.asarray([0, 1, 2, 3], dtype=np.int64)
    msps = np.asarray([12], dtype=np.int32)
    msp_lengths = np.asarray([8], dtype=np.int32)
    apply_result = {
        "ns": np.asarray([5, 100], dtype=np.int32),
        "nl": np.asarray([20, 100], dtype=np.int32),
        "as": msps,
        "al": msp_lengths,
        "encoded": obs,
        "ns_scores": np.asarray([0.25, 1.0]),
        "as_scores": np.asarray([0.5]),
    }

    result = fused_stages.build_fused_recall_result(
        {"query_sequence": "A" * 150},
        apply_result,
        llr_hit="hit",
        llr_miss="miss",
        min_llr=4.0,
        min_opps=3,
        unify_threshold=90,
        with_scores=True,
    )

    interval_args = seen["interval_args"]
    assert interval_args[0] is apply_result["ns"]
    assert interval_args[1] is apply_result["nl"]
    assert interval_args[2] is msps
    assert interval_args[3] is msp_lengths
    assert interval_args[4:] == (150, 90)
    scan_args = seen["scan_args"]
    assert len(scan_args) == 2
    assert scan_args[0][0] is obs
    assert scan_args[0][1:] == (10, 30, "hit", "miss", 4.0, 3)
    assert scan_args[1][0] is obs
    assert scan_args[1][1:] == (100, 130, "hit", "miss", 4.0, 3)
    assert result["ns"].tolist() == [100]
    assert result["nl"].tolist() == [100]
    assert result["as"] is msps
    assert result["al"] is msp_lengths
    assert result["nq_for_kept_nucs"] == [255]
    assert len(result["tf_calls"]) == 1


def test_build_fused_recall_result_projects_circular_tf_calls(monkeypatch):
    captured = {}

    def fake_build_scan_intervals(ns, nl, msps, msp_lengths, read_length, unify_threshold):
        assert read_length == 300
        assert list(ns) == [195]
        assert list(nl) == [20]
        assert list(msps) == [190]
        assert list(msp_lengths) == [40]
        return [(190, 230)]

    def fake_call_tfs_in_interval(
        obs, lo, hi, llr_hit, llr_miss, min_llr, min_opps, **kwargs,
    ):
        captured.update(kwargs)
        return [
            TFCall(
                start=195,
                length=20,
                llr=6.0,
                n_opps=4,
                left_ambiguity=1,
                right_ambiguity=2,
            )
        ]

    monkeypatch.setattr(fused_stages, "build_scan_intervals", fake_build_scan_intervals)
    monkeypatch.setattr(fused_stages, "call_tfs_in_interval", fake_call_tfs_in_interval)

    apply_result = {
        "ns": np.asarray([0, 95], dtype=np.int32),
        "nl": np.asarray([15, 5], dtype=np.int32),
        "as": np.asarray([0, 90], dtype=np.int32),
        "al": np.asarray([30, 10], dtype=np.int32),
        "encoded": np.zeros(300, dtype=np.int32),
        "circular": True,
        "circular_read_length": 100,
        "circular_ns": [(95, 20)],
        "circular_as": [(90, 40)],
        "circular_ns_scores": np.asarray([0.75], dtype=np.float32),
        "circular_as_scores": np.asarray([0.25], dtype=np.float32),
        "tiled_ns": np.asarray([195], dtype=np.int32),
        "tiled_nl": np.asarray([20], dtype=np.int32),
        "tiled_as": np.asarray([190], dtype=np.int32),
        "tiled_al": np.asarray([40], dtype=np.int32),
    }

    result = fused_stages.build_fused_recall_result(
        {"query_sequence": "A" * 100},
        apply_result,
        llr_hit="hit",
        llr_miss="miss",
        min_llr=4.0,
        min_opps=3,
        unify_threshold=90,
        with_scores=True,
        m5c_mask=np.arange(100) % 7 == 0,
        m5c_llr_hit="m5c-hit",
        m5c_llr_miss="m5c-miss",
    )

    assert result["circular"] is True
    assert result["tf_calls"] == [
        TFCall(start=95, length=20, llr=6.0, n_opps=4,
               left_ambiguity=1, right_ambiguity=2)
    ]
    assert result["circular_ns"] == []
    assert result["circular_as"] == [(90, 40)]
    assert result["ns"].tolist() == []
    assert result["nl"].tolist() == []
    assert result["nq_for_kept_nucs"] == []
    assert np.array_equal(
        captured["m5c_mask"], np.tile(np.arange(100) % 7 == 0, 3),
    )


def test_run_tf_recall_stage_forwards_m5c_tables_and_mask(monkeypatch):
    captured = {}

    monkeypatch.setattr(
        fused_stages, "build_scan_intervals", lambda *_args, **_kwargs: [(0, 4)],
    )

    def fake_call(*args, **kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr(fused_stages, "call_tfs_in_interval", fake_call)
    mask = np.array([False, True, True, False])
    fused_stages.run_tf_recall_stage(
        np.zeros(4, dtype=np.int32), [], [], [0], [4], 4,
        "hit", "miss", 5.0, 3, 90,
        m5c_mask=mask, m5c_llr_hit="m5c-hit", m5c_llr_miss="m5c-miss",
    )
    assert captured["m5c_mask"] is mask
    assert captured["m5c_llr_hit"] == "m5c-hit"
    assert captured["m5c_llr_miss"] == "m5c-miss"


def test_baseline_radial_nuc_finalization_never_receives_tf_calls(monkeypatch):
    radial = [NucCall(20, 120, 200, 10, 12, dyad=80)]
    profile = object()
    seen = {}

    def fake_validate(*args, **kwargs):
        seen["provisional_tf_calls"] = args[4]
        seen["nuc_profile"] = kwargs.get("nuc_profile")
        return radial, [(140, 20)]

    def fake_rederive(original_msps, accessible, read_length, msp_min_size):
        seen["rederive"] = (
            list(original_msps), list(accessible), read_length, msp_min_size,
        )
        return [(140, 20)]

    def fake_exclude(msps, nucs, msp_min_size):
        seen["exclude"] = (list(msps), list(nucs), msp_min_size)
        return list(msps)

    monkeypatch.setattr(
        fused_stages, "validate_radial_access_in_read", fake_validate,
    )
    monkeypatch.setattr(fused_stages, "rederive_msps", fake_rederive)
    monkeypatch.setattr(
        fused_stages, "exclude_nucleosomes_from_msps", fake_exclude,
    )

    nucs, msps = fused_stages.finalize_baseline_radial_nuc_configuration(
        np.zeros(200, dtype=np.int32),
        [20],
        [120],
        [(140, 20)],
        radial,
        200,
        np.zeros(1),
        np.zeros(1),
        4.0,
        3,
        85,
        0,
        nuc_profile=profile,
    )

    assert seen["provisional_tf_calls"] == ()
    assert seen["nuc_profile"] is profile
    assert seen["rederive"] == ([(140, 20)], [(140, 20)], 200, 0)
    assert seen["exclude"] == ([(140, 20)], radial, 0)
    assert nucs == radial
    assert msps == [(140, 20)]


def test_ddda_baseline_stage_order_is_nuc_then_tf(monkeypatch):
    events = []
    radial = [NucCall(20, 120, 200, 10, 12, dyad=80)]

    def fake_recall_nucs(*args, **kwargs):
        events.append("nuc_refine")
        return radial, [(140, 20)]

    profile = object()

    def fake_finalize(*args, **kwargs):
        assert events == ["nuc_refine"]
        assert kwargs["nuc_profile"] is profile
        events.append("nuc_finalize")
        return radial, [(140, 20)]

    def fake_recall_tfs(*args, **kwargs):
        assert events == ["nuc_refine", "nuc_finalize"]
        events.append("tf_refine")
        return []

    monkeypatch.setattr(fused_stages, "recall_nucs_in_read", fake_recall_nucs)
    monkeypatch.setattr(
        fused_stages,
        "finalize_baseline_radial_nuc_configuration",
        fake_finalize,
    )
    monkeypatch.setattr(fused_stages, "run_tf_recall_stage", fake_recall_tfs)
    monkeypatch.setattr(
        fused_stages,
        "promote_large_tf_calls",
        lambda calls, *_args, **_kwargs: (list(calls), []),
    )

    result = fused_stages.build_fused_recall_result(
        {"query_sequence": "A" * 200},
        {
            "ns": np.asarray([20], dtype=np.int32),
            "nl": np.asarray([120], dtype=np.int32),
            "as": np.asarray([140], dtype=np.int32),
            "al": np.asarray([20], dtype=np.int32),
            "encoded": np.zeros(200, dtype=np.int32),
        },
        np.zeros(1),
        np.zeros(1),
        5.0,
        3,
        90,
        True,
        recall_nucs=True,
        nuc_profile=profile,
    )

    assert events == ["nuc_refine", "nuc_finalize", "tf_refine"]
    assert result["tf_calls"] == []


def test_fused_ddda_uses_separate_tf_and_nuc_likelihood_tables(monkeypatch):
    seen = {}
    radial = [NucCall(20, 120, 200, 10, 12, dyad=80)]

    def fake_recall_nucs(*args, **_kwargs):
        seen["initial_nuc"] = args[4:6]
        return radial, [(140, 20)]

    def fake_finalize(*args, **kwargs):
        seen["final_nuc"] = args[6:8]
        seen["final_nuc_m5c"] = (
            kwargs["m5c_llr_hit"], kwargs["m5c_llr_miss"],
        )
        return radial, [(140, 20)]

    def fake_tf(*args, **_kwargs):
        seen["tf"] = args[6:8]
        return []

    monkeypatch.setattr(fused_stages, "recall_nucs_in_read", fake_recall_nucs)
    monkeypatch.setattr(
        fused_stages, "finalize_baseline_radial_nuc_configuration", fake_finalize,
    )
    monkeypatch.setattr(fused_stages, "run_tf_recall_stage", fake_tf)
    monkeypatch.setattr(
        fused_stages,
        "promote_large_tf_calls",
        lambda calls, *_args, **_kwargs: (list(calls), []),
    )

    tf_hit, tf_miss = object(), object()
    nuc_hit, nuc_miss = object(), object()
    nuc_m5c_hit, nuc_m5c_miss = object(), object()
    fused_stages.build_fused_recall_result(
        {"query_sequence": "A" * 200},
        {
            "ns": np.asarray([20], dtype=np.int32),
            "nl": np.asarray([120], dtype=np.int32),
            "as": np.asarray([140], dtype=np.int32),
            "al": np.asarray([20], dtype=np.int32),
            "encoded": np.zeros(200, dtype=np.int32),
        },
        tf_hit,
        tf_miss,
        5.0,
        3,
        90,
        True,
        recall_nucs=True,
        nuc_profile=object(),
        nuc_llr_hit=nuc_hit,
        nuc_llr_miss=nuc_miss,
        nuc_m5c_llr_hit=nuc_m5c_hit,
        nuc_m5c_llr_miss=nuc_m5c_miss,
    )

    assert seen["initial_nuc"] == (nuc_hit, nuc_miss)
    assert seen["final_nuc"] == (nuc_hit, nuc_miss)
    assert seen["final_nuc_m5c"] == (nuc_m5c_hit, nuc_m5c_miss)
    assert seen["tf"] == (tf_hit, tf_miss)


@pytest.mark.parametrize("circular", [False, True])
def test_ddda_driver_threads_nuc_profile_to_finalizer(monkeypatch, circular):
    """Both production drivers must activate HMM-crossing model comparison."""
    profile = object()

    class ProfileObserved(RuntimeError):
        pass

    monkeypatch.setattr(
        fused_stages,
        "recall_nucs_in_read",
        lambda *_args, **_kwargs: ([], []),
    )

    def fake_finalize(*_args, **kwargs):
        assert kwargs["nuc_profile"] is profile
        raise ProfileObserved

    monkeypatch.setattr(
        fused_stages,
        "finalize_baseline_radial_nuc_configuration",
        fake_finalize,
    )
    encoded_length = 300 if circular else 100
    apply_result = {
        "ns": np.asarray([], dtype=np.int32),
        "nl": np.asarray([], dtype=np.int32),
        "as": np.asarray([], dtype=np.int32),
        "al": np.asarray([], dtype=np.int32),
        "encoded": np.zeros(encoded_length, dtype=np.int32),
    }
    if circular:
        apply_result.update({
            "circular": True,
            "circular_read_length": 100,
            "tiled_ns": np.asarray([], dtype=np.int32),
            "tiled_nl": np.asarray([], dtype=np.int32),
            "tiled_as": np.asarray([], dtype=np.int32),
            "tiled_al": np.asarray([], dtype=np.int32),
        })

    with pytest.raises(ProfileObserved):
        fused_stages.build_fused_recall_result(
            {"query_sequence": "A" * 100},
            apply_result,
            llr_hit=np.zeros(1),
            llr_miss=np.zeros(1),
            min_llr=5.0,
            min_opps=3,
            unify_threshold=90,
            with_scores=True,
            recall_nucs=True,
            nuc_profile=profile,
        )
