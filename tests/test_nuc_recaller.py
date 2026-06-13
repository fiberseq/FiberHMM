"""Tests for the per-read nucleosome recaller (split + edge refine + nuc+QQQ)."""
from __future__ import annotations

import numpy as np

from fiberhmm.inference.circular import project_center_nuc_calls
from fiberhmm.inference.nuc_recaller import (
    NucCall,
    assemble_nuc_msp_tiling,
    drop_short_nucs_overlapping_promoted,
    promote_large_tf_calls,
    recall_nucs_in_read,
    rederive_msps,
    unify_circular_nuc_calls_with_tf_calls,
    unify_nuc_calls_with_tf_calls,
)
from fiberhmm.inference.tf_recaller import TFCall, N_CTX, UNMETH_OFFSET
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
    nucs, access = recall_nucs_in_read(
        obs, ns=[0], nl=[len(obs)], read_length=len(obs),
        llr_hit=llr_hit, llr_miss=llr_miss,
        split_min_llr=4.0, split_min_opps=3, nuc_min_size=40,
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


def test_rederive_msps_merges_and_filters():
    msps = rederive_msps(
        original_msps=[(0, 10)],
        accessible_from_splits=[(10, 5), (200, 3)],
        read_length=300, msp_min_size=4,
    )
    # (0,10)+(10,5) merge into one 15bp MSP; the 3bp patch is filtered out
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
    assert (by_start[10].length, by_start[10].nq, by_start[10].el, by_start[10].er) == (30, 200, 255, 128)
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
