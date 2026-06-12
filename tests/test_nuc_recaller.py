"""Tests for the per-read nucleosome recaller (split + edge refine + nuc+QQQ)."""
from __future__ import annotations

import numpy as np

from fiberhmm.inference.nuc_recaller import (
    NucCall,
    recall_nucs_in_read,
    rederive_msps,
    unify_nuc_calls_with_tf_calls,
)
from fiberhmm.inference.tf_recaller import TFCall, N_CTX, UNMETH_OFFSET
from fiberhmm.io.ma_tags import format_aq_array, parse_aq_array

HIT = 0                 # ctx 0, modified (accessible evidence)
MISS = UNMETH_OFFSET    # ctx 0, unmodified (protected evidence)


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
