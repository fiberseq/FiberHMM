"""Tests for both-strand consensus construction (fiberhmm-merge core)."""
import pysam
import pytest

from fiberhmm.cli.merge import run_merge
from fiberhmm.crossstrand.consensus import build_consensus, format_deam_ma_tag
from fiberhmm.crossstrand.recall import deam_regime_masks


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
    for q in (25, 35):              # canonical G opposite GA deaminations
        ct_seq[q] = 'G'
    ga_seq = list('A' * 60)
    for q in (0, 10):               # canonical C opposite CT deaminations
        ga_seq[q] = 'C'
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
    assert cons.deam_count == 5
    assert cons.base_conflicts == 0
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


def test_consensus_coverage_excludes_source_deletions():
    h = _header()
    ct = _read(h, 'ct-del', 100, 'AAAAACCCCC')
    ct.cigartuples = [(0, 5), (2, 2), (0, 5)]  # no CT evidence at ref 105-106
    ga = _read(h, 'ga-full', 100, 'AAAAACCAAAAA')
    cons = build_consensus(ct, ga)
    assert cons is not None
    assert cons.ma == '12;deam+:1-5,8-5;deam-:1-12'

    seg = _read(h, 'joint', 100, cons.seq)
    seg.set_tag('MA', cons.ma)
    plus, minus = deam_regime_masks(seg)
    assert plus.tolist() == [True] * 5 + [False] * 2 + [True] * 5
    assert minus.tolist() == [True] * 12


def test_consensus_masks_disagreeing_source_bases():
    h = _header()
    ct = _read(h, 'ct-disagree', 100, 'AACAA')
    ga = _read(h, 'ga-disagree', 100, 'AAGAA')
    cons = build_consensus(ct, ga)
    assert cons is not None
    assert cons.seq == 'AANAA'
    assert cons.base_conflicts == 1


def test_merge_fails_closed_on_duplicate_paired_names(tmp_path):
    h = _header()

    def paired(name, mate, marker):
        read = _read(h, name, 100, marker + 'A' * 99)
        read.set_tag('MA', '100;nuc.:1-100')
        read.set_tag('mt', 'P', value_type='A')
        read.set_tag('mp', mate)
        return read

    input_bam = tmp_path / 'duplicate-pairs.bam'
    with pysam.AlignmentFile(input_bam, 'wb', header=h) as out:
        out.write(paired('duplicate', 'ga', 'Y'))
        out.write(paired('duplicate', 'ga', 'Y'))
        out.write(paired('ga', 'duplicate', 'R'))

    with pytest.raises(ValueError, match='duplicate paired name'):
        run_merge(str(input_bam), str(tmp_path / 'merged.bam'), io_threads=1)
    assert not (tmp_path / 'merged.bam.unsorted.bam').exists()


def test_merge_fails_closed_on_nonreciprocal_pair_tags(tmp_path):
    h = _header()

    def paired(name, mate, marker):
        read = _read(h, name, 100, marker + 'A' * 99)
        read.set_tag('MA', '100;nuc.:1-100')
        read.set_tag('mt', 'P', value_type='A')
        read.set_tag('mp', mate)
        return read

    input_bam = tmp_path / 'nonreciprocal.bam'
    with pysam.AlignmentFile(input_bam, 'wb', header=h) as out:
        out.write(paired('ct', 'ga', 'Y'))
        out.write(paired('ga', 'someone-else', 'R'))

    with pytest.raises(ValueError, match='not reciprocal'):
        run_merge(str(input_bam), str(tmp_path / 'merged.bam'), io_threads=1)
