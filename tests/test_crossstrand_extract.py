"""Tests for the both-strand overlay extraction (fiberhmm-extract --both-strand)."""
import io

import pysam

from fiberhmm.cli.extract_tags import _extract_both_strand, _interval_intersection


def test_interval_intersection():
    assert _interval_intersection([(0, 100)], [(50, 150)]) == [(50, 100)]
    assert _interval_intersection([(0, 40), (60, 100)], [(30, 70)]) == [(30, 40), (60, 70)]
    assert _interval_intersection([(0, 50)], [(50, 100)]) == []      # touching, no overlap
    assert _interval_intersection([(0, 100)], []) == []


def _consensus_read(ma):
    h = pysam.AlignmentHeader.from_dict(
        {'HD': {'VN': '1.6'}, 'SQ': [{'SN': 'chr1', 'LN': 100000}]})
    r = pysam.AlignedSegment(h)
    r.query_name = 'm.cs'
    r.flag = 0
    r.reference_id = 0
    r.reference_start = 1000
    r.cigartuples = [(0, 200)]          # all-M -> query pos q maps to ref 1000+q
    r.query_sequence = 'A' * 200
    r.query_qualities = pysam.qualitystring_to_array('I' * 200)
    r.set_tag('MA', ma, value_type='Z')
    return r


def test_both_strand_extract_intersection_to_bed():
    # deam+ = query [0,100), deam- = query [50,150) -> both = [50,100) -> ref [1050,1100)
    r = _consensus_read('200;deam+:1-100;deam-:51-100')
    out = io.StringIO()
    n = _extract_both_strand(r, out)
    assert n == 1
    fields = out.getvalue().strip().split('\t')
    assert fields[0] == 'chr1'
    assert int(fields[1]) == 1050 and int(fields[2]) == 1100   # both-strand ref span
    assert fields[3] == 'm.cs'
    assert fields[9] == '1' and fields[10].startswith('50')    # one 50bp block
    assert fields[-1] == '0'                                  # isDuplicate


def test_both_strand_extract_appends_haplotype_then_duplicate_fields():
    r = _consensus_read('200;deam+:1-100;deam-:51-100')
    r.set_tag('HP', 2, value_type='i')
    r.set_tag('PS', 4242, value_type='i')
    r.flag |= 0x400
    out = io.StringIO()

    assert _extract_both_strand(r, out, haplotype_fields=True) == 1
    fields = out.getvalue().strip().split('\t')
    assert len(fields) == 15
    assert fields[-3:] == ['2', '4242', '1']


def test_non_consensus_read_yields_nothing():
    # Only a nucleosome track, no deam+/deam- -> no both-strand feature.
    r = _consensus_read('200;nuc.:1-100')
    out = io.StringIO()
    assert _extract_both_strand(r, out) == 0
    assert out.getvalue() == ''
