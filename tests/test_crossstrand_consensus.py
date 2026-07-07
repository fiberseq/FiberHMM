"""Tests for both-strand consensus construction (fiberhmm-merge core)."""
import pysam
import pytest

from fiberhmm.crossstrand.consensus import build_consensus, format_deam_ma_tag


def _header():
    return pysam.AlignmentHeader.from_dict(
        {'HD': {'VN': '1.6', 'SO': 'coordinate'},
         'SQ': [{'SN': 'chr1', 'LN': 1000}]})


def _read(header, name, start, seq):
    r = pysam.AlignedSegment(header)
    r.query_name = name
    r.flag = 0
    r.reference_id = 0
    r.reference_start = start
    r.mapping_quality = 60
    r.cigartuples = [(0, len(seq))]  # all-M
    r.query_sequence = seq
    r.query_qualities = pysam.qualitystring_to_array('I' * len(seq))
    return r


@pytest.fixture
def pair():
    """CT read chr1:100-160 with Y at ref 110/120/130; GA read chr1:120-180
    with R at ref 125/135. Deaminations are R/Y-encoded in the sequence."""
    h = _header()
    ct_seq = list('A' * 60)
    for q in (10, 20, 30):          # ref 110/120/130
        ct_seq[q] = 'Y'
    ga_seq = list('A' * 60)
    for q in (5, 15):               # ref 125/135
        ga_seq[q] = 'R'
    ct = _read(h, 'ct', 100, ''.join(ct_seq))
    ga = _read(h, 'ga', 120, ''.join(ga_seq))
    return ct, ga


def test_consensus_spans_union_and_encodes_both_strands(pair):
    ct, ga = pair
    cons = build_consensus(ct, ga)
    assert cons is not None
    assert cons.ref_start == 100
    assert cons.length == 80                      # union 100..180
    assert cons.both_start == 120 and cons.both_end == 160
    s = cons.seq
    # C->T deaminations (Y) at ref 110/120/130 -> query 10/20/30
    assert s[10] == 'Y' and s[20] == 'Y' and s[30] == 'Y'
    # G->A deaminations (R) at ref 125/135 -> query 25/35
    assert s[25] == 'R' and s[35] == 'R'
    assert cons.nm == 5
    # one read carries both strands' deaminations
    assert 'Y' in s and 'R' in s


def test_consensus_ma_regime_tag(pair):
    ct, ga = pair
    cons = build_consensus(ct, ga)
    # deam+ = CT span (query 0, len 60); deam- = GA span (query 20, len 60)
    assert cons.ma == '80;deam+:1-60;deam-:21-60'


def test_format_deam_ma_tag_is_1based():
    # 0-based query starts -> 1-based spec starts
    assert format_deam_ma_tag(500, 0, 300, 120, 380) == '500;deam+:1-300;deam-:121-380'
