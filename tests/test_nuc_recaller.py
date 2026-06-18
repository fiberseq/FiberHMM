"""Tests for the per-read nucleosome recaller (split + edge refine + nuc+QQQ)."""
from __future__ import annotations

import numpy as np

from fiberhmm.inference.circular import project_center_nuc_calls
from fiberhmm.inference.nuc_recaller import (
    NucCall,
    _AccessibleSplitRequest,
    _assemble_circular_nuc_msp_tiling_from_request,
    _assemble_nuc_msp_tiling_from_request,
    _bounded_interval,
    _BoundedInterval,
    _BoundedNucSpansRecallRequest,
    _circular_uncovered_cut,
    _CircularNucMspTilingRequest,
    _CircularTilingFrame,
    _clip_ordered_nuc_calls_for_tiling,
    _keep_nuc_against_circular_intervals,
    _msp_gaps_between_nucs,
    _nuc_from_protected_calls,
    _NucMspTilingRequest,
    _NucRecallParams,
    _NucRecallResult,
    _NucsInReadRecallRequest,
    _NucSpanRecallRequest,
    _ordered_positive_nuc_calls,
    _phase_cut_window,
    _phase_or_unsplit_subfragments,
    _phase_subfragments,
    _phase_subfragments_from_request,
    _PhaseCutWindow,
    _PhaseSplitRequest,
    _promoted_nuc_from_tf_call,
    _recall_bounded_nuc_spans,
    _recall_bounded_nuc_spans_from_request,
    _recall_nuc_params,
    _recall_nuc_span,
    _recall_nuc_span_from_request,
    _recall_nuc_tables,
    _rederive_msps_from_request,
    _ReDerivedMspsRequest,
    _refine_fragment,
    _refine_fragment_from_request,
    _RefineFragmentRequest,
    _residue_intervals_around_nuc,
    _restore_circular_tiling_frame,
    _rotate_circular_nuc_calls,
    _split_on_accessible_cuts,
    _split_on_accessible_cuts_from_request,
    _total_call_llr,
    _whole_molecule_nuc_candidate,
    assemble_circular_nuc_msp_tiling,
    assemble_nuc_msp_tiling,
    drop_short_nucs_overlapping_promoted,
    promote_large_tf_calls,
    recall_nucs_in_read,
    recall_nucs_in_read_from_request,
    rederive_msps,
    unify_circular_nuc_calls_with_tf_calls,
    unify_nuc_calls_with_tf_calls,
)
from fiberhmm.inference.tf_recaller import N_CTX, UNMETH_OFFSET, TFCall
from fiberhmm.io.ma_tags import format_aq_array, parse_aq_array

HIT = 0                 # ctx 0, modified (accessible evidence)
MISS = UNMETH_OFFSET    # ctx 0, unmodified (protected evidence)
NONTARGET = N_CTX       # code 4096: not an opportunity (e.g. non-C/G base)


def _llr_tables():
    """Protected-favoring tables: a miss favors protected, a hit favors accessible."""
    llr_hit = np.full(N_CTX, -3.0, dtype=np.float64)
    llr_miss = np.full(N_CTX, 0.3, dtype=np.float64)
    return llr_hit, llr_miss


def _obs(*runs):
    """Build an obs array from (code, count) runs."""
    parts = [np.full(n, code, dtype=np.int32) for code, n in runs]
    return np.concatenate(parts)


def test_split_separates_two_nucs_at_a_hit_cluster():
    # 60 misses | 6 hits | 60 misses -> the hit cluster is a cut between 2 nucs
    obs = _obs((MISS, 60), (HIT, 6), (MISS, 60))
    llr_hit, llr_miss = _llr_tables()
    nucs, access = recall_nucs_in_read_from_request(
        _NucsInReadRecallRequest(
            obs=obs,
            ns=[0],
            nl=[len(obs)],
            read_length=len(obs),
            llr_hit=llr_hit,
            llr_miss=llr_miss,
            split_min_llr=4.0,
            split_min_opps=3,
            nuc_min_size=40,
        )
    )
    assert len(nucs) == 2, [(n.start, n.length) for n in nucs]
    # a cut (accessible run) was recorded over the hit cluster
    assert any(60 <= s <= 66 for s, _ in access)
    # both nucleosomes carry quality + edge bytes
    for nc in nucs:
        assert nc.nq > 0
        assert 0 <= nc.el <= 255 and 0 <= nc.er <= 255


def test_no_split_when_no_accessible_evidence():
    obs = _obs((MISS, 150))
    llr_hit, llr_miss = _llr_tables()
    nucs, access = recall_nucs_in_read(
        obs, ns=[0], nl=[len(obs)], read_length=len(obs),
        llr_hit=llr_hit, llr_miss=llr_miss,
        split_min_llr=4.0, split_min_opps=3, nuc_min_size=40,
    )
    assert len(nucs) == 1
    assert nucs[0].length == 150
    assert access == []


def test_split_on_accessible_cuts_returns_fragments_and_cut_access():
    obs = _obs((MISS, 60), (HIT, 6), (MISS, 60))
    llr_hit, llr_miss = _llr_tables()
    request = _AccessibleSplitRequest(
        obs=obs,
        start=0,
        end=len(obs),
        nhit=-llr_hit,
        nmiss=-llr_miss,
        min_llr=4.0,
        min_opps=3,
    )

    split = _split_on_accessible_cuts_from_request(request)
    adapted = _split_on_accessible_cuts(
        obs, 0, len(obs), -llr_hit, -llr_miss,
        split_min_llr=4.0, split_min_opps=3,
    )

    assert len(split.fragments) == 2
    assert split.access
    assert any(60 <= s <= 66 for s, _ in split.access)
    assert adapted == split


def test_recall_nuc_span_matches_single_read_wrapper():
    obs = _obs((MISS, 60), (HIT, 6), (MISS, 60))
    llr_hit, llr_miss = _llr_tables()
    tables = _recall_nuc_tables(llr_hit, llr_miss)
    params = _NucRecallParams(
        split_min_llr=4.0,
        split_min_opps=3,
        nuc_min_size=40,
        edge_min_llr=2.0,
        edge_min_opps=2,
        phase_nrl=0,
        phase_min_llr=1.0,
        phase_min_opps=1,
        phase_window=35,
    )

    span_result = _recall_nuc_span_from_request(
        _NucSpanRecallRequest(
            obs=obs,
            start=0,
            end=len(obs),
            tables=tables,
            params=params,
        )
    )
    adapted_span_result = _recall_nuc_span(obs, 0, len(obs), tables, params)
    read_nucs, read_access = recall_nucs_in_read(
        obs,
        ns=[0],
        nl=[len(obs)],
        read_length=len(obs),
        llr_hit=llr_hit,
        llr_miss=llr_miss,
        split_min_llr=4.0,
        split_min_opps=3,
        nuc_min_size=40,
    )

    assert span_result.nucs == read_nucs
    assert span_result.access == read_access
    assert adapted_span_result == span_result


def test_recall_nuc_params_captures_thresholds():
    params = _recall_nuc_params(
        split_min_llr=4.5,
        split_min_opps=5,
        nuc_min_size=90,
        edge_min_llr=2.5,
        edge_min_opps=4,
        phase_nrl=185,
        phase_min_llr=1.5,
        phase_min_opps=2,
        phase_window=40,
    )

    assert params == _NucRecallParams(
        split_min_llr=4.5,
        split_min_opps=5,
        nuc_min_size=90,
        edge_min_llr=2.5,
        edge_min_opps=4,
        phase_nrl=185,
        phase_min_llr=1.5,
        phase_min_opps=2,
        phase_window=40,
    )


def test_recall_bounded_nuc_spans_skips_invalid_spans(monkeypatch):
    obs = np.zeros(10, dtype=np.int32)
    llr_hit, llr_miss = _llr_tables()
    params = _recall_nuc_params(
        split_min_llr=4.0,
        split_min_opps=3,
        nuc_min_size=40,
        edge_min_llr=2.0,
        edge_min_opps=2,
        phase_nrl=0,
        phase_min_llr=1.0,
        phase_min_opps=1,
        phase_window=35,
    )
    calls = []

    def fake_span(request):
        calls.append((
            request.obs,
            request.start,
            request.end,
            request.tables,
            request.params,
        ))
        return _NucRecallResult(
            nucs=[
                NucCall(request.start, request.end - request.start, 1, 2, 3),
            ],
            access=[(request.start, 1)],
        )

    monkeypatch.setattr(
        "fiberhmm.inference.nuc_recaller._recall_nuc_span_from_request",
        fake_span,
    )

    tables = _recall_nuc_tables(llr_hit, llr_miss)
    result = _recall_bounded_nuc_spans_from_request(
        _BoundedNucSpansRecallRequest(
            obs=obs,
            ns=[0, 5, 9, 12],
            nl=[3, 0, 5, 2],
            read_length=10,
            tables=tables,
            params=params,
        )
    )
    adapted_result = _recall_bounded_nuc_spans(
        obs,
        ns=[9],
        nl=[1],
        read_length=10,
        tables=tables,
        params=params,
    )

    assert [(call[1], call[2]) for call in calls] == [(0, 3), (9, 10), (9, 10)]
    assert result.nucs == [NucCall(0, 3, 1, 2, 3), NucCall(9, 1, 1, 2, 3)]
    assert result.access == [(0, 1), (9, 1)]
    assert adapted_result.nucs == [NucCall(9, 1, 1, 2, 3)]


def test_total_call_llr_sums_call_scores():
    calls = [
        TFCall(start=0, length=5, llr=1.5, n_opps=2,
               left_ambiguity=0, right_ambiguity=0),
        TFCall(start=10, length=5, llr=2.25, n_opps=2,
               left_ambiguity=0, right_ambiguity=0),
    ]

    assert _total_call_llr(calls) == 3.75


def test_nuc_from_protected_calls_uses_outer_span_and_floor():
    calls = [
        TFCall(start=20, length=5, llr=2.25, n_opps=2,
               left_ambiguity=0, right_ambiguity=3),
        TFCall(start=10, length=5, llr=1.5, n_opps=2,
               left_ambiguity=2, right_ambiguity=0),
    ]

    nuc = _nuc_from_protected_calls(calls, nuc_min_size=15)

    assert nuc is not None
    assert (nuc.start, nuc.length) == (10, 15)
    assert nuc.nq > 0
    assert nuc.el > 0
    assert nuc.er > 0
    assert _nuc_from_protected_calls(calls, nuc_min_size=16) is None


def test_residue_intervals_around_nuc_reports_flanks_only():
    nuc = NucCall(10, 20, 200, 255, 255)

    assert _residue_intervals_around_nuc(0, 40, nuc) == [(0, 10), (30, 10)]
    assert _residue_intervals_around_nuc(10, 30, nuc) == []


def test_refine_fragment_request_matches_adapter():
    obs = _obs((MISS, 85))
    llr_hit, llr_miss = _llr_tables()
    request = _RefineFragmentRequest(
        obs=obs,
        start=0,
        end=len(obs),
        llr_hit=llr_hit,
        llr_miss=llr_miss,
        nuc_min_size=40,
        edge_min_llr=2.0,
        edge_min_opps=2,
    )

    requested = _refine_fragment_from_request(request)
    adapted = _refine_fragment(
        obs,
        0,
        len(obs),
        llr_hit,
        llr_miss,
        40,
        2.0,
        2,
    )

    assert requested == adapted
    assert isinstance(requested.nuc, NucCall)
    assert requested.nuc.length == 85
    assert requested.access == []


def test_bounded_interval_handles_clamping_and_invalid_spans():
    assert _bounded_interval(10, 20, 25, clamp_start=False) == _BoundedInterval(
        start=10,
        end=25,
    )
    assert _bounded_interval(10, 0, 25, clamp_start=False) is None
    assert _bounded_interval(30, 5, 25, clamp_start=False) is None
    assert _bounded_interval(-5, 10, 25, clamp_start=True) == _BoundedInterval(
        start=0,
        end=5,
    )
    assert _bounded_interval(-5, 10, 25, clamp_start=False) == _BoundedInterval(
        start=-5,
        end=5,
    )


def test_subnucleosome_fragment_demoted_to_accessible():
    # 20 misses | 6 hits | 100 misses -> left flank (20bp) is below nuc_min_size
    obs = _obs((MISS, 20), (HIT, 6), (MISS, 100))
    llr_hit, llr_miss = _llr_tables()
    nucs, access = recall_nucs_in_read(
        obs, ns=[0], nl=[len(obs)], read_length=len(obs),
        llr_hit=llr_hit, llr_miss=llr_miss,
        split_min_llr=4.0, split_min_opps=3, nuc_min_size=40,
    )
    assert len(nucs) == 1
    assert nucs[0].start >= 26  # only the long right flank survives as a nuc
    # the 20bp flank shows up as accessible residue
    assert any(s == 0 and length == 20 for s, length in access)


def test_refined_core_below_floor_is_demoted_not_emitted():
    # 100bp fragment, but only a 20bp protected island (the rest is non-target,
    # so nothing splits and no nucleosome-sized protected core exists).
    obs = _obs((NONTARGET, 40), (MISS, 20), (NONTARGET, 40))
    llr_hit, llr_miss = _llr_tables()
    nucs, access = recall_nucs_in_read(
        obs, ns=[0], nl=[len(obs)], read_length=len(obs),
        llr_hit=llr_hit, llr_miss=llr_miss,
        split_min_llr=4.0, split_min_opps=3, nuc_min_size=85,
    )
    # the 20bp core must NOT be emitted as a nuc; whole fragment -> accessible
    assert nucs == []
    assert (0, 100) in access


def test_genuine_85bp_nuc_survives_edge_pass():
    obs = _obs((MISS, 85))
    llr_hit, llr_miss = _llr_tables()
    nucs, _ = recall_nucs_in_read(
        obs, ns=[0], nl=[len(obs)], read_length=len(obs),
        llr_hit=llr_hit, llr_miss=llr_miss,
        split_min_llr=4.0, split_min_opps=3, nuc_min_size=85,
    )
    assert len(nucs) == 1 and nucs[0].length >= 85


def test_phase_prior_splits_long_footprint_at_single_event():
    # 380bp footprint, all protected except ONE deamination near the predicted
    # linker (~190). Pass 1 (min_opps=3) can't split a single event; the phase
    # prior (nrl=185) lowers the bar there and splits into two ~nucleosomes.
    obs = _obs((MISS, 190), (HIT, 1), (MISS, 189))
    llr_hit, llr_miss = _llr_tables()
    kw = dict(ns=[0], nl=[len(obs)], read_length=len(obs),
              llr_hit=llr_hit, llr_miss=llr_miss,
              split_min_llr=4.0, split_min_opps=3, nuc_min_size=85)
    nucs_off, _ = recall_nucs_in_read(obs, phase_nrl=0, **kw)
    assert len(nucs_off) == 1
    nucs_on, _ = recall_nucs_in_read(obs, phase_nrl=185, **kw)
    assert len(nucs_on) == 2


def test_phase_subfragments_request_matches_adapter():
    obs = _obs((MISS, 190), (HIT, 1), (MISS, 189))
    llr_hit, llr_miss = _llr_tables()
    request = _PhaseSplitRequest(
        obs=obs,
        start=0,
        end=len(obs),
        nhit=-llr_hit,
        nmiss=-llr_miss,
        nrl=185,
        min_llr=1.0,
        min_opps=1,
        window=35,
    )

    requested = _phase_subfragments_from_request(request)
    adapted = _phase_subfragments(
        obs,
        0,
        len(obs),
        -llr_hit,
        -llr_miss,
        185,
        1.0,
        1,
        35,
    )

    assert requested == adapted
    assert len(requested.fragments) == 2
    assert requested.cuts


def test_phase_prior_never_splits_signal_desert():
    # long fully-protected footprint with ZERO deamination -> no split even with
    # the phase prior on (the prior lowers the threshold but evidence is still
    # required at the predicted linker).
    obs = _obs((MISS, 380))
    llr_hit, llr_miss = _llr_tables()
    nucs, _ = recall_nucs_in_read(
        obs, ns=[0], nl=[len(obs)], read_length=len(obs),
        llr_hit=llr_hit, llr_miss=llr_miss,
        split_min_llr=4.0, split_min_opps=3, nuc_min_size=85, phase_nrl=185)
    assert len(nucs) == 1


def test_phase_cut_window_clips_to_fragment_and_rejects_tiny_windows():
    assert _phase_cut_window(
        10, 100, pred=50, phase_window=15,
    ) == _PhaseCutWindow(start=35, end=65)
    assert _phase_cut_window(
        10, 100, pred=12, phase_window=15,
    ) == _PhaseCutWindow(start=10, end=27)
    assert _phase_cut_window(10, 11, pred=10, phase_window=15) is None


def test_phase_or_unsplit_subfragments_returns_original_when_disabled():
    phase = _phase_or_unsplit_subfragments(
        np.array([], dtype=np.int32),
        10,
        90,
        np.array([], dtype=np.float64),
        np.array([], dtype=np.float64),
        phase_nrl=0,
        phase_min_llr=1.0,
        phase_min_opps=1,
        phase_window=35,
    )

    assert phase.fragments == [(10, 90)]
    assert phase.cuts == []


def test_rederive_msps_merges_and_filters():
    request = _ReDerivedMspsRequest(
        original_msps=[(0, 10)],
        accessible_from_splits=[(10, 5), (200, 3)],
        read_length=300,
        msp_min_size=4,
    )

    msps = _rederive_msps_from_request(request)
    adapted = rederive_msps(
        original_msps=request.original_msps,
        accessible_from_splits=request.accessible_from_splits,
        read_length=300, msp_min_size=4,
    )

    # (0,10)+(10,5) merge into one 15bp MSP; the 3bp patch is filtered out
    assert adapted == msps
    assert (0, 15) in msps
    assert all(length >= 4 for _, length in msps)


def test_unify_drops_short_nuc_overlapping_tf():
    nucs = [NucCall(0, 50, 200, 255, 255), NucCall(100, 30, 150, 200, 200)]
    tf = [TFCall(start=105, length=20, llr=6.0, n_opps=4,
                 left_ambiguity=1, right_ambiguity=1)]
    kept = unify_nuc_calls_with_tf_calls(nucs, tf, unify_threshold=85)
    # the 30bp nuc overlapping the TF call is dropped; the 50bp one is kept
    # (both are < threshold, so overlap is what decides)
    assert [(k.start, k.length) for k in kept] == [(0, 50)]


def test_project_center_nuc_calls_keeps_quality_and_picks_center():
    n = 100
    calls = [
        NucCall(start=110, length=30, nq=200, el=255, er=128),  # center tile -> (10,30)
        NucCall(start=10, length=30, nq=1, el=1, er=1),         # first tile -> dropped
        NucCall(start=50, length=200, nq=150, el=64, er=64),    # covers middle -> (0,100)
    ]
    proj = project_center_nuc_calls(calls, n)
    by_start = {p.start: p for p in proj}
    assert set(by_start) == {10, 0}
    assert (
        by_start[10].length,
        by_start[10].nq,
        by_start[10].el,
        by_start[10].er,
    ) == (30, 200, 255, 128)
    assert by_start[0].length == 100  # whole-molecule projection


def test_unify_circular_drops_short_nuc_overlapping_wrapped_tf():
    n = 100
    # a wrapped TF call near the origin: covers [95,100) and [0,5)
    tf = [TFCall(start=95, length=10, llr=6.0, n_opps=4,
                 left_ambiguity=1, right_ambiguity=1)]
    nucs = [
        NucCall(start=2, length=20, nq=100, el=0, er=0),    # short, overlaps wrap -> drop
        NucCall(start=40, length=30, nq=100, el=0, er=0),   # short, no overlap -> keep
    ]
    kept = unify_circular_nuc_calls_with_tf_calls(nucs, tf, unify_threshold=85,
                                                  read_length=n)
    assert [(k.start, k.length) for k in kept] == [(40, 30)]


def test_keep_nuc_against_circular_intervals_keeps_large_nucs():
    wrapped_tf = [(95, 10)]

    assert not _keep_nuc_against_circular_intervals(
        NucCall(start=2, length=20, nq=100, el=0, er=0),
        wrapped_tf,
        unify_threshold=85,
        read_length=100,
    )
    assert _keep_nuc_against_circular_intervals(
        NucCall(start=2, length=85, nq=100, el=0, er=0),
        wrapped_tf,
        unify_threshold=85,
        read_length=100,
    )


def test_promote_large_tf_to_nuc():
    # a nucleosome-sized protected TF call (>= threshold) is promoted to nuc+
    # with edges; a small TF stays in tf+.
    obs = _obs((MISS, 300))
    llr_hit, llr_miss = _llr_tables()
    tf = [
        TFCall(start=0, length=120, llr=10.0, n_opps=20,
               left_ambiguity=1, right_ambiguity=1),   # nucleosome-sized
        TFCall(start=200, length=20, llr=6.0, n_opps=4,
               left_ambiguity=1, right_ambiguity=1),   # real small footprint
    ]
    remaining, promoted = promote_large_tf_calls(
        tf, obs, llr_hit, llr_miss, threshold=90, nuc_min_size=85)
    assert len(promoted) == 1 and promoted[0].length >= 85
    assert [c.start for c in remaining] == [200]


def test_promoted_nuc_from_tf_call_refines_large_tf_span():
    obs = _obs((MISS, 150))
    llr_hit, llr_miss = _llr_tables()
    tf_call = TFCall(start=0, length=120, llr=10.0, n_opps=20,
                     left_ambiguity=1, right_ambiguity=1)

    nuc = _promoted_nuc_from_tf_call(
        tf_call,
        obs,
        llr_hit,
        llr_miss,
        nuc_min_size=85,
        edge_min_llr=2.0,
        edge_min_opps=2,
    )

    assert isinstance(nuc, NucCall)
    assert nuc.length >= 85


def test_drop_short_nuc_overlapping_promoted():
    # Codex repro: a short (< unify_threshold) nuc starting slightly BEFORE a
    # promoted nucleosome must be dropped, so the start-order tiling does not
    # keep the short one and clip/discard the promoted one.
    promoted = [NucCall(5, 100, 200, 255, 255)]            # [5,105)
    short = [NucCall(0, 85, 100, 255, 255)]                # 85 < 90, overlaps
    assert drop_short_nucs_overlapping_promoted(short, promoted, 90) == []
    # full path: drop + add promoted, then tile -> promoted survives whole
    kept, _ = assemble_nuc_msp_tiling(
        drop_short_nucs_overlapping_promoted(short, promoted, 90) + promoted,
        span_lo=0, span_hi=300, msp_min_size=0, nuc_min_size=85)
    assert [(k.start, k.length) for k in kept] == [(5, 100)]
    # a long (>= threshold) nuc is NOT dropped; a non-overlapping short is kept
    long_nuc = [NucCall(0, 90, 100, 255, 255)]
    assert drop_short_nucs_overlapping_promoted(long_nuc, promoted, 90) == long_nuc
    far = [NucCall(300, 85, 100, 255, 255)]
    assert drop_short_nucs_overlapping_promoted(far, promoted, 90) == far


def test_msp_gaps_between_nucs_respects_floor():
    kept = [
        NucCall(10, 20, 200, 255, 255),
        NucCall(50, 10, 200, 255, 255),
    ]

    assert _msp_gaps_between_nucs(kept, span_lo=0, span_hi=70, floor=5) == [
        (0, 10),
        (30, 20),
        (60, 10),
    ]
    assert _msp_gaps_between_nucs(kept, span_lo=0, span_hi=70, floor=15) == [
        (30, 20),
    ]


def test_ordered_positive_nuc_calls_drops_empty_and_prefers_longer_same_start():
    calls = [
        NucCall(10, 20, 1, 2, 3),
        NucCall(10, 40, 4, 5, 6),
        NucCall(5, 0, 7, 8, 9),
        NucCall(30, 10, 10, 11, 12),
    ]

    ordered = _ordered_positive_nuc_calls(calls)

    assert [(n.start, n.length) for n in ordered] == [
        (10, 40),
        (10, 20),
        (30, 10),
    ]


def test_clip_ordered_nuc_calls_for_tiling_clips_overlap_and_drops_short():
    ordered = [
        NucCall(10, 50, 1, 2, 3),
        NucCall(40, 50, 4, 5, 6),
        NucCall(95, 10, 7, 8, 9),
    ]

    kept = _clip_ordered_nuc_calls_for_tiling(
        ordered, span_lo=0, nuc_floor=20,
    )

    assert kept == [
        NucCall(10, 50, 1, 2, 3),
        NucCall(60, 30, 4, 0, 6),
    ]


def test_assemble_nuc_msp_tiling_request_matches_adapter():
    nucs = [
        NucCall(10, 50, 1, 2, 3),
        NucCall(40, 50, 4, 5, 6),
        NucCall(95, 10, 7, 8, 9),
    ]
    request = _NucMspTilingRequest(
        nuc_calls=nucs,
        span_lo=0,
        span_hi=120,
        msp_min_size=5,
        nuc_min_size=20,
    )

    requested = _assemble_nuc_msp_tiling_from_request(request)
    adapted = assemble_nuc_msp_tiling(
        nucs,
        span_lo=0,
        span_hi=120,
        msp_min_size=5,
        nuc_min_size=20,
    )

    assert requested == adapted
    kept, msps = requested
    assert kept == [
        NucCall(10, 50, 1, 2, 3),
        NucCall(60, 30, 4, 0, 6),
    ]
    assert msps == [(0, 10), (90, 30)]


def test_circular_tiling_no_overlap_for_wrapped_nuc():
    # Codex repro (High): a nucleosome wrapping the origin must not get an MSP gap
    # derived linearly over its wrapped tail. A nuc [180,200)+[0,80) on a 200 bp
    # circle should leave a single MSP over the uncovered arc [80,180), with nucs
    # and MSPs tiling the circle exactly (no overlap, no gap).
    rl = 200
    nucs = [NucCall(180, 100, 200, 255, 255)]   # wraps: [180,200) + [0,80)
    request = _CircularNucMspTilingRequest(
        nuc_calls=nucs,
        read_length=rl,
        msp_min_size=1,
        nuc_min_size=85,
    )

    kept, msps = _assemble_circular_nuc_msp_tiling_from_request(request)
    assert assemble_circular_nuc_msp_tiling(
        nucs,
        rl,
        msp_min_size=1,
        nuc_min_size=85,
    ) == (kept, msps)
    assert [(k.start, k.length) for k in kept] == [(180, 100)]
    assert msps == [(80, 100)]
    cov = [0] * rl
    for s, length in [(k.start, k.length) for k in kept] + msps:
        for off in range(length):
            cov[(s + off) % rl] += 1
    assert all(c == 1 for c in cov)   # exact circular tiling: no overlap, no gap


def test_circular_rotation_helpers_choose_cut_and_split_straddlers():
    rl = 200
    wrapped = [NucCall(180, 100, 200, 255, 128)]

    assert _circular_uncovered_cut(wrapped, rl) == 80
    assert [
        (n.start, n.length, n.nq, n.el, n.er)
        for n in _rotate_circular_nuc_calls(wrapped, cut=190, read_length=rl)
    ] == [
        (190, 10, 200, 255, 0),
        (0, 90, 200, 0, 128),
    ]
    assert _restore_circular_tiling_frame(
        [NucCall(0, 90, 200, 0, 128)],
        [(90, 100)],
        cut=80,
        read_length=rl,
    ) == _CircularTilingFrame(
        nucs=[NucCall(80, 90, 200, 0, 128)],
        msps=[(170, 100)],
    )


def test_circular_tiling_empty_nucs_gives_whole_molecule_msp():
    # Codex repro (High): if circular unification drops all nucleosomes, the read
    # must tile as one accessible MSP over the whole molecule, not lose it.
    kept, msps = assemble_circular_nuc_msp_tiling([], 200, msp_min_size=0)
    assert kept == []
    assert msps == [(0, 200)]


def test_circular_tiling_fully_covered_overlap_tiles_exactly():
    # Codex repro (High): a fully-covered circle (no uncovered cut point) with
    # overlapping nucs must still tile exactly -- the origin can fall inside a
    # wrapped call, which previously produced double-covered bases.
    rl = 200
    nucs = [NucCall(0, 120, 200, 255, 255), NucCall(100, 120, 200, 255, 255)]
    kept, msps = assemble_circular_nuc_msp_tiling(nucs, rl, msp_min_size=0)
    cov = [0] * rl
    for s, length in [(k.start, k.length) for k in kept] + msps:
        for off in range(length):
            cov[(s + off) % rl] += 1
    assert all(c == 1 for c in cov)   # exact tiling: no base covered twice, none missed


def test_circular_tiling_whole_molecule_nuc_normalizes_start():
    # A projected center-copy run can become a whole-molecule nuc with nonzero
    # start. It must stay a full nuc, not split at the origin and demote the
    # short piece to MSP by the nuc_min_size floor.
    kept, msps = assemble_circular_nuc_msp_tiling(
        [NucCall(50, 200, 201, 123, 45)], 200, msp_min_size=0, nuc_min_size=85)
    assert [(k.start, k.length, k.nq, k.el, k.er) for k in kept] == [
        (0, 200, 201, 123, 45)
    ]
    assert msps == []


def test_whole_molecule_nuc_candidate_returns_first_covering_call():
    short = NucCall(10, 199, 1, 2, 3)
    whole = NucCall(50, 200, 4, 5, 6)
    later = NucCall(0, 220, 7, 8, 9)

    assert _whole_molecule_nuc_candidate([short, whole, later], 200) is whole
    assert _whole_molecule_nuc_candidate([short], 200) is None


def test_nuc_qqq_aq_roundtrip():
    # nuc+QQQ (2 nucs) then tf+QQQ (1 tf): parse back to per-annotation triples
    aq = format_aq_array(
        nq_values=[255, 150], tf_q_values=[100], tf_lq_values=[255], tf_rq_values=[0],
        nuc_lq_values=[246, 200], nuc_rq_values=[255, 64],
    )
    parsed = parse_aq_array(aq, ["QQQ", "", "QQQ"], [2, 1, 1])
    assert parsed[0] == [255, 246, 255]   # nuc 0 (nq, el, er)
    assert parsed[1] == [150, 200, 64]    # nuc 1
    assert parsed[2] == []                # msp (no bytes)
    assert parsed[3] == [100, 255, 0]     # tf
