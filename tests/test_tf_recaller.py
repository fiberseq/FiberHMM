"""Tests for the LLR TF recaller (fiberhmm.inference.tf_recaller +
fiberhmm.io.ma_tags).
"""
import numpy as np
import pytest

from fiberhmm.io.ma_tags import (
    EDGE_AMBIGUITY_SAT,
    TQ_SCALE,
    ambiguity_to_edge,
    format_aq_array,
    format_ma_tag,
    llr_to_tq,
    parse_aq_array,
    parse_ma_tag,
    tq_to_llr,
)
from fiberhmm.inference.tf_recaller import (
    ENZYME_PRESETS,
    N_CTX,
    UNMETH_OFFSET,
    build_scan_intervals,
    call_tfs_in_interval,
    merge_intervals,
)


# ------------------- ma_tags ---------------------------------------

def test_llr_to_tq_roundtrip():
    for llr in [0, 1, 5, 10, 25.5, 100]:
        tq = llr_to_tq(llr)
        assert 0 <= tq <= 255
    assert llr_to_tq(0) == 0
    assert llr_to_tq(5) == 50
    assert llr_to_tq(100) == 255  # saturated
    assert tq_to_llr(255) == 25.5


def test_ambiguity_to_edge():
    assert ambiguity_to_edge(0) == 255
    assert ambiguity_to_edge(EDGE_AMBIGUITY_SAT) == 0
    assert ambiguity_to_edge(EDGE_AMBIGUITY_SAT + 5) == 0
    assert 0 < ambiguity_to_edge(15) < 255


def test_format_and_parse_ma_tag():
    nucs = [(43, 147), (216, 155)]
    msps = [(1, 42)]
    tfs = [(50, 20)]
    ma = format_ma_tag(read_length=4521,
                       nuc_intervals=nucs,
                       msp_intervals=msps,
                       tf_intervals=tfs)
    assert ma.startswith('4521;')
    assert 'nuc+Q:44-147,217-155' in ma
    assert 'msp+:2-42' in ma
    assert 'tf+QQQ:51-20' in ma
    parsed = parse_ma_tag(ma)
    assert parsed['read_length'] == 4521
    assert parsed['nuc'] == nucs
    assert parsed['msp'] == msps
    assert parsed['tf'] == tfs


def test_aq_layout():
    nq = [200, 180]
    tq = [45]; el = [180]; er = [220]
    aq = format_aq_array(nq, tq, el, er)
    # 2 bytes (nucs) + 3 bytes (tf) = 5 bytes total
    assert list(aq) == [200, 180, 45, 180, 220]
    parsed = parse_aq_array(aq, ['Q', '', 'QQQ'], [2, 0, 1])
    assert parsed == [[200], [180], [45, 180, 220]]


# ------------------- merge_intervals -------------------------------

def test_merge_intervals():
    assert merge_intervals([]) == []
    assert merge_intervals([(0, 10), (5, 15)]) == [(0, 15)]
    assert merge_intervals([(0, 10), (10, 20)]) == [(0, 20)]  # touching = merge
    assert merge_intervals([(0, 10), (20, 30)]) == [(0, 10), (20, 30)]


def test_build_scan_intervals_includes_short_nucs():
    # MSPs at [0, 100), short nuc at [200, 240) (nl=40), big nuc at [400, 600)
    iv = build_scan_intervals(
        ns=[200, 400], nl=[40, 200],
        as_=[0], al=[100],
        read_len=1000, unify_threshold=90,
    )
    # Should include MSP + short nuc, NOT the big nuc
    assert (0, 100) in iv
    assert (200, 240) in iv
    assert all(start != 400 for start, _ in iv)


# ------------------- Kadane scan -----------------------------------

def _make_obs(seq):
    """Helper: convert a string of chars (h=hit@ctx0, m=miss@ctx0, .=non-target)
    to an obs np.ndarray."""
    arr = []
    for c in seq:
        if c == 'h': arr.append(0)
        elif c == 'm': arr.append(UNMETH_OFFSET)
        else: arr.append(N_CTX)  # non-target
    return np.array(arr, dtype=np.int32)


def test_kadane_clean_miss_run():
    llr_hit = np.full(N_CTX, -2.0)
    llr_miss = np.full(N_CTX, 0.5)
    obs = _make_obs('hhmmmmmmmmmmmhh')  # 5 hits, 11 misses... no wait
    obs = _make_obs('hhmmmmmmmmmmhh')   # 2 hits, 10 misses, 2 hits
    calls = call_tfs_in_interval(obs, 0, len(obs), llr_hit, llr_miss,
                                  min_llr=3.0, min_opps=3)
    assert len(calls) == 1
    c = calls[0]
    assert c.start == 2
    assert c.length == 10
    assert c.llr == pytest.approx(5.0)
    assert c.left_ambiguity == 0      # hit immediately at left
    assert c.right_ambiguity == 0     # hit immediately at right


def test_kadane_absorbs_one_rogue_hit():
    """A 20-bp miss run with 1 hit in the middle should remain a single call,
    because Kadane allows the running LLR to dip without resetting if there's
    enough surrounding positive evidence."""
    llr_hit = np.full(N_CTX, -2.0)
    llr_miss = np.full(N_CTX, 0.5)
    # 5 hits + 10 misses + 1 hit + 10 misses + 5 hits
    obs = _make_obs('h' * 5 + 'm' * 10 + 'h' + 'm' * 10 + 'h' * 5)
    calls = call_tfs_in_interval(obs, 0, len(obs), llr_hit, llr_miss,
                                  min_llr=3.0, min_opps=3)
    assert len(calls) == 1
    c = calls[0]
    # Call should span both miss runs + the rogue hit
    assert c.length >= 20
    # LLR = 20 * 0.5 - 1 * 2.0 = 8.0
    assert c.llr == pytest.approx(8.0)


def test_kadane_subthreshold_rejected():
    llr_hit = np.full(N_CTX, -2.0)
    llr_miss = np.full(N_CTX, 0.5)
    # Only 4 misses -> LLR = 2 < 3
    obs = _make_obs('hhmmmmhh')
    calls = call_tfs_in_interval(obs, 0, len(obs), llr_hit, llr_miss,
                                  min_llr=3.0, min_opps=3)
    assert calls == []


def test_kadane_ambiguous_edges():
    """Non-target padding between miss run and hit -> ambiguity > 0."""
    llr_hit = np.full(N_CTX, -2.0)
    llr_miss = np.full(N_CTX, 0.5)
    # hit + 5 non-target + 10 misses + 5 non-target + hit
    obs = _make_obs('h' + '.' * 5 + 'm' * 10 + '.' * 5 + 'h')
    calls = call_tfs_in_interval(obs, 0, len(obs), llr_hit, llr_miss,
                                  min_llr=3.0, min_opps=3)
    assert len(calls) == 1
    c = calls[0]
    assert c.left_ambiguity == 5
    assert c.right_ambiguity == 5


# ------------------- enzyme presets --------------------------------

def test_enzyme_presets_present():
    for enz in ('hia5', 'dddb', 'ddda'):
        assert enz in ENZYME_PRESETS
        for k in ('min_llr', 'emission_uplift'):
            assert k in ENZYME_PRESETS[enz]
    # All presets use uplift=1.0; DddA gets a pre-uplifted model file
    # (ddda_TF.json) rather than a runtime power transform.
    for enz in ('hia5', 'dddb', 'ddda'):
        assert ENZYME_PRESETS[enz]['emission_uplift'] == 1.0
    # DddB uses lower min_llr than Hia5 (single-strand evidence)
    assert ENZYME_PRESETS['dddb']['min_llr'] < ENZYME_PRESETS['hia5']['min_llr']
