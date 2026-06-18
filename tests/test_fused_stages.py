"""Tests for fused apply/recall stage boundaries."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from fiberhmm.inference import fused_stages
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


def test_run_hmm_apply_stage_request_delegates_single_read(monkeypatch):
    calls = []
    fiber_read = {"query_sequence": "AAAA"}

    monkeypatch.setattr(
        fused_stages,
        "_process_single_read",
        lambda *args, **kwargs: calls.append((args, kwargs)) or {"ns": [1]},
    )

    request = fused_stages._HmmApplyStageRequest(
        fiber_read=fiber_read,
        model="model",
        edge_trim=10,
        circular=True,
        mode="daf",
        context_size=5,
        msp_min_size=0,
        nuc_min_size=85,
        with_scores=True,
    )

    assert fused_stages.run_hmm_apply_stage_from_request(request) == {"ns": [1]}
    assert calls == [
        (
            (fiber_read, "model", 10, True, "daf", 5, 0),
            {
                "nuc_min_size": 85,
                "with_scores": True,
                "return_posteriors": False,
                "include_encoded": True,
            },
        ),
    ]
    calls.clear()

    assert fused_stages.run_hmm_apply_stage(
        fiber_read,
        "model",
        edge_trim=10,
        circular=True,
        mode="daf",
        context_size=5,
        msp_min_size=0,
        nuc_min_size=85,
        with_scores=True,
    ) == {"ns": [1]}
    assert len(calls) == 1


def test_apply_result_interval_bounds_collects_nucs_and_msps():
    bounds = fused_stages._apply_result_interval_bounds({
        "ns": np.asarray([10], dtype=np.int32),
        "nl": np.asarray([20], dtype=np.int64),
        "as": np.asarray([40], dtype=np.int32),
        "al": np.asarray([5], dtype=np.int64),
    })

    assert bounds == fused_stages._IntervalBounds(
        starts=[10, 40],
        ends=[30, 45],
    )
    assert all(type(value) is int for value in bounds.starts + bounds.ends)


def test_analyzed_span_covers_apply_intervals_kept_nucs_and_read_fallback():
    empty_span = fused_stages._analyzed_span_from_request(
        fused_stages._AnalyzedSpanRequest(
            apply_result={"ns": [], "nl": [], "as": [], "al": []},
            read_length=200,
            kept=[],
        )
    )
    span = fused_stages._analyzed_span(
        {
            "ns": np.asarray([10], dtype=np.int32),
            "nl": np.asarray([20], dtype=np.int64),
            "as": np.asarray([70], dtype=np.int32),
            "al": np.asarray([10], dtype=np.int64),
        },
        read_length=200,
        kept=[SimpleNamespace(start=100, length=40)],
    )

    assert empty_span == fused_stages._AnalyzedSpan(start=0, end=200)
    assert span.start == 10
    assert span.end == 140


def test_nuc_call_quality_lists_preserve_nq_el_er_order():
    nucs = [
        SimpleNamespace(nq=1, el=2, er=3),
        SimpleNamespace(nq=4, el=5, er=6),
    ]

    assert fused_stages._nuc_call_quality_lists(nucs) == fused_stages._NucCallQualityLists(
        nq_values=[1, 4],
        el_values=[2, 5],
        er_values=[3, 6],
    )


def test_nuc_call_quality_fields_names_quality_lists():
    nucs = [
        SimpleNamespace(nq=1, el=2, er=3),
        SimpleNamespace(nq=4, el=5, er=6),
    ]

    assert fused_stages._nuc_call_quality_fields(nucs) == {
        "nq_for_kept_nucs": [1, 4],
        "nuc_el_for_kept": [2, 5],
        "nuc_er_for_kept": [3, 6],
    }


def test_nuc_call_arrays_preserve_start_length_order_and_dtype():
    nucs = [
        SimpleNamespace(start=10, length=20),
        SimpleNamespace(start=40, length=5),
    ]

    arrays = fused_stages._nuc_call_arrays(nucs)

    assert arrays.starts.dtype == np.int32
    assert arrays.lengths.dtype == np.int32
    np.testing.assert_array_equal(arrays.starts, [10, 40])
    np.testing.assert_array_equal(arrays.lengths, [20, 5])


def test_nuc_call_start_length_lists_preserve_plain_values():
    nucs = [
        SimpleNamespace(start=np.int32(10), length=np.int64(20)),
        SimpleNamespace(start=40, length=5),
    ]

    starts, lengths = fused_stages._nuc_call_start_length_lists(nucs)

    assert starts == [np.int32(10), 40]
    assert lengths == [np.int64(20), 5]


def test_optional_apply_scores_respects_enabled_flag():
    apply_result = {"ns_scores": np.asarray([0.25], dtype=np.float32)}

    assert fused_stages._optional_apply_scores(
        apply_result, "ns_scores", enabled=True,
    ) is apply_result["ns_scores"]
    assert fused_stages._optional_apply_scores(
        apply_result, "ns_scores", enabled=False,
    ) is None
    assert fused_stages._optional_apply_scores(
        apply_result, "as_scores", enabled=True,
    ) is None


def test_optional_apply_score_fields_respects_enabled_flag():
    apply_result = {
        "ns_scores": np.asarray([0.25], dtype=np.float32),
        "as_scores": np.asarray([0.5], dtype=np.float32),
    }

    enabled = fused_stages._optional_apply_score_fields_from_request(
        fused_stages._OptionalApplyScoreFieldsRequest(
            apply_result=apply_result,
            enabled=True,
        )
    )
    disabled = fused_stages._optional_apply_score_fields(apply_result, enabled=False)

    assert enabled["ns_scores"] is apply_result["ns_scores"]
    assert enabled["as_scores"] is apply_result["as_scores"]
    assert disabled == {"ns_scores": None, "as_scores": None}


def test_promote_large_tf_nucs_request_appends_promoted_after_kept(monkeypatch):
    calls = {}

    def fake_promote(tf_calls, obs, llr_hit, llr_miss, threshold, floor):
        calls["promote"] = (tf_calls, obs, llr_hit, llr_miss, threshold, floor)
        return ["small_tf"], ["promoted_nuc"]

    def fake_drop(nuc_calls, promoted, threshold):
        calls["drop"] = (nuc_calls, promoted, threshold)
        return ["kept_nuc"]

    monkeypatch.setattr(fused_stages, "promote_large_tf_calls", fake_promote)
    monkeypatch.setattr(
        fused_stages,
        "drop_short_nucs_overlapping_promoted",
        fake_drop,
    )

    result = fused_stages._promote_large_tf_nucs_from_request(
        fused_stages._PromoteLargeTfNucsRequest(
            tf_calls=["tf"],
            nuc_calls=["nuc"],
            obs="obs",
            llr_hit="hit",
            llr_miss="miss",
            unify_threshold=90,
            nuc_min_size=85,
        )
    )

    assert result == (["small_tf"], ["kept_nuc", "promoted_nuc"])
    assert calls["promote"] == (["tf"], "obs", "hit", "miss", 90, 85)
    assert calls["drop"] == (["nuc"], ["promoted_nuc"], 90)

    calls.clear()
    assert fused_stages._promote_large_tf_nucs(
        ["tf"],
        ["nuc"],
        "obs",
        "hit",
        "miss",
        unify_threshold=90,
        nuc_min_size=85,
    ) == (["small_tf"], ["kept_nuc", "promoted_nuc"])
    assert calls["promote"] == (["tf"], "obs", "hit", "miss", 90, 85)
    assert calls["drop"] == (["nuc"], ["promoted_nuc"], 90)


def test_interval_pairs_casts_parallel_arrays_to_int_pairs():
    pairs = fused_stages._interval_pairs(
        np.asarray([1, 2], dtype=np.int32),
        np.asarray([10, 20], dtype=np.int64),
    )

    assert pairs == [(1, 10), (2, 20)]
    assert all(type(value) is int for pair in pairs for value in pair)


def test_interval_pair_lists_names_parallel_start_length_values():
    pairs = fused_stages._interval_pair_lists([(1, 10), (2, 20)])

    assert pairs == fused_stages._IntervalPairLists(
        starts=[1, 2],
        lengths=[10, 20],
    )


def test_interval_ends_casts_and_adds_starts_lengths():
    ends = fused_stages._interval_ends([
        (np.int32(1), np.int64(10)),
        (2, 20),
    ])

    assert ends == [11, 22]
    assert all(type(value) is int for value in ends)


def test_tiled_interval_arrays_prefers_tiled_values():
    base = {
        "ns": np.asarray([1], dtype=np.int32),
        "nl": np.asarray([10], dtype=np.int32),
        "as": np.asarray([20], dtype=np.int32),
        "al": np.asarray([30], dtype=np.int32),
    }
    result = fused_stages._tiled_interval_arrays(base)
    assert result.nuc_starts is base["ns"]
    assert result.nuc_lengths is base["nl"]
    assert result.msp_starts is base["as"]
    assert result.msp_lengths is base["al"]

    tiled = {
        **base,
        "tiled_ns": np.asarray([101], dtype=np.int32),
        "tiled_nl": np.asarray([110], dtype=np.int32),
        "tiled_as": np.asarray([120], dtype=np.int32),
        "tiled_al": np.asarray([130], dtype=np.int32),
    }
    result = fused_stages._tiled_interval_arrays(tiled)
    assert result.nuc_starts is tiled["tiled_ns"]
    assert result.nuc_lengths is tiled["tiled_nl"]
    assert result.msp_starts is tiled["tiled_as"]
    assert result.msp_lengths is tiled["tiled_al"]


def test_apply_result_msp_pairs_reads_apply_msp_arrays():
    pairs = fused_stages._apply_result_msp_pairs({
        "as": np.asarray([3, 7], dtype=np.int32),
        "al": np.asarray([4, 8], dtype=np.int64),
    })

    assert pairs == [(3, 4), (7, 8)]
    assert all(type(value) is int for pair in pairs for value in pair)


def test_circular_read_length_prefers_apply_metadata():
    fiber_read = {"query_sequence": "A" * 100}

    assert fused_stages._circular_read_length(
        fiber_read, {"circular_read_length": 75},
    ) == 75
    assert fused_stages._circular_read_length(fiber_read, {}) == 100


def test_apply_result_is_circular_uses_truthy_metadata_flag():
    assert fused_stages._apply_result_is_circular({"circular": True}) is True
    assert fused_stages._apply_result_is_circular({"circular": 1}) is True
    assert fused_stages._apply_result_is_circular({"circular": False}) is False
    assert fused_stages._apply_result_is_circular({}) is False


def test_run_tf_recall_stage_request_scans_intervals(monkeypatch):
    calls = []
    obs = np.asarray([0, 1, 2], dtype=np.int64)
    ns = np.asarray([10], dtype=np.int32)
    nl = np.asarray([20], dtype=np.int32)
    msps = np.asarray([30], dtype=np.int32)
    msp_lengths = np.asarray([5], dtype=np.int32)

    def fake_build_scan_intervals(*args, **kwargs):
        calls.append(("intervals", args, kwargs))
        return [(10, 20), (30, 40)]

    def fake_call_tfs_in_interval(*args):
        calls.append(("scan", args))
        return [args[1]]

    monkeypatch.setattr(
        fused_stages,
        "build_scan_intervals",
        fake_build_scan_intervals,
    )
    monkeypatch.setattr(
        fused_stages,
        "call_tfs_in_interval",
        fake_call_tfs_in_interval,
    )

    request = fused_stages._TfRecallStageRequest(
        obs=obs,
        ns=ns,
        nl=nl,
        msps=msps,
        msp_lengths=msp_lengths,
        read_length=100,
        llr_hit="hit",
        llr_miss="miss",
        min_llr=4.0,
        min_opps=3,
        unify_threshold=90,
    )

    assert fused_stages.run_tf_recall_stage_from_request(request) == [10, 30]
    label, interval_args, interval_kwargs = calls[0]
    assert label == "intervals"
    assert interval_args[0] is ns
    assert interval_args[1] is nl
    assert interval_args[2] is msps
    assert interval_args[3] is msp_lengths
    assert interval_args[4] == 100
    assert interval_kwargs == {"unify_threshold": 90}
    assert calls[1][0] == "scan"
    assert calls[1][1][0] is obs
    assert calls[1][1][1:] == (10, 20, "hit", "miss", 4.0, 3)
    assert calls[2][0] == "scan"
    assert calls[2][1][0] is obs
    assert calls[2][1][1:] == (30, 40, "hit", "miss", 4.0, 3)
    calls.clear()

    assert fused_stages.run_tf_recall_stage(
        obs,
        ns,
        nl,
        msps,
        msp_lengths,
        100,
        "hit",
        "miss",
        4.0,
        3,
        90,
    ) == [10, 30]
    assert len(calls) == 3


def test_build_fused_recall_result_builds_request(monkeypatch):
    seen = {}
    fiber_read = {"query_sequence": "AAAA"}
    apply_result = {"circular": True}

    def fake_build_from_request(request):
        seen["request"] = request
        return {"ok": True}

    monkeypatch.setattr(
        fused_stages,
        "build_fused_recall_result_from_request",
        fake_build_from_request,
    )

    assert fused_stages.build_fused_recall_result(
        fiber_read,
        apply_result,
        llr_hit="hit",
        llr_miss="miss",
        min_llr=4.0,
        min_opps=3,
        unify_threshold=90,
        with_scores=True,
        recall_nucs=True,
        split_min_llr=5.0,
        split_min_opps=4,
        nuc_min_size=75,
        msp_min_size=2,
        phase_nrl=147,
    ) == {"ok": True}

    request = seen["request"]
    assert request.fiber_read is fiber_read
    assert request.apply_result is apply_result
    assert request.llr_hit == "hit"
    assert request.llr_miss == "miss"
    assert request.min_llr == 4.0
    assert request.min_opps == 3
    assert request.unify_threshold == 90
    assert request.with_scores is True
    assert request.recall_nucs is True
    assert request.split_min_llr == 5.0
    assert request.split_min_opps == 4
    assert request.nuc_min_size == 75
    assert request.msp_min_size == 2
    assert request.phase_nrl == 147


def test_build_fused_recall_result_request_dispatches(monkeypatch):
    calls = []
    fiber_read = {"query_sequence": "AAAA"}

    def fake_request_branch(label):
        def inner(request):
            calls.append((label, request))
            return {"branch": label}

        return inner

    monkeypatch.setattr(
        fused_stages,
        "_build_fused_recall_result_without_nucs_linear_from_request",
        fake_request_branch("linear"),
    )
    monkeypatch.setattr(
        fused_stages,
        "_build_fused_recall_result_without_nucs_circular_from_request",
        fake_request_branch("circular"),
    )
    monkeypatch.setattr(
        fused_stages,
        "_build_fused_recall_result_with_nucs_from_request",
        fake_request_branch("nucs"),
    )
    monkeypatch.setattr(
        fused_stages,
        "_build_fused_recall_result_with_nucs_circular_from_request",
        fake_request_branch("nucs_circular"),
    )

    def request_for(apply_result, recall_nucs):
        return fused_stages._FusedRecallResultRequest(
            fiber_read=fiber_read,
            apply_result=apply_result,
            llr_hit="hit",
            llr_miss="miss",
            min_llr=4.0,
            min_opps=3,
            unify_threshold=90,
            with_scores=True,
            recall_nucs=recall_nucs,
            split_min_llr=5.0,
            split_min_opps=4,
            nuc_min_size=75,
            msp_min_size=2,
            phase_nrl=147,
        )

    def assert_no_nucs_request(call, label, apply_result):
        assert call[0] == label
        request = call[1]
        assert request == fused_stages._FusedNoNucsResultRequest(
            fiber_read=fiber_read,
            apply_result=apply_result,
            llr_hit="hit",
            llr_miss="miss",
            min_llr=4.0,
            min_opps=3,
            unify_threshold=90,
            with_scores=True,
        )
        assert request.fiber_read is fiber_read
        assert request.apply_result is apply_result

    def assert_nucs_request(call, label, apply_result):
        assert call[0] == label
        request = call[1]
        assert request == fused_stages._FusedNucsResultRequest(
            fiber_read=fiber_read,
            apply_result=apply_result,
            llr_hit="hit",
            llr_miss="miss",
            min_llr=4.0,
            min_opps=3,
            unify_threshold=90,
            split_min_llr=5.0,
            split_min_opps=4,
            nuc_min_size=75,
            msp_min_size=2,
            phase_nrl=147,
        )
        assert request.fiber_read is fiber_read
        assert request.apply_result is apply_result

    linear_apply = {"circular": False}
    circular_apply = {"circular": True}

    assert fused_stages.build_fused_recall_result_from_request(
        request_for(linear_apply, recall_nucs=False),
    ) == {"branch": "linear"}
    assert_no_nucs_request(calls[-1], "linear", linear_apply)

    assert fused_stages.build_fused_recall_result_from_request(
        request_for(circular_apply, recall_nucs=False),
    ) == {"branch": "circular"}
    assert_no_nucs_request(calls[-1], "circular", circular_apply)

    assert fused_stages.build_fused_recall_result_from_request(
        request_for(linear_apply, recall_nucs=True),
    ) == {"branch": "nucs"}
    assert_nucs_request(calls[-1], "nucs", linear_apply)

    assert fused_stages.build_fused_recall_result_from_request(
        request_for(circular_apply, recall_nucs=True),
    ) == {"branch": "nucs_circular"}
    assert_nucs_request(calls[-1], "nucs_circular", circular_apply)


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
    def fake_build_scan_intervals(ns, nl, msps, msp_lengths, read_length, unify_threshold):
        assert read_length == 300
        assert list(ns) == [195]
        assert list(nl) == [20]
        assert list(msps) == [190]
        assert list(msp_lengths) == [40]
        return [(190, 230)]

    def fake_call_tfs_in_interval(obs, lo, hi, llr_hit, llr_miss, min_llr, min_opps):
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
