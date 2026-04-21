"""Tests for the --block-scores (BED12+N) path of fiberhmm-extract.

Covers:
  - autoSQL schemas emit the right extra int[blockCount] columns per type
  - each extractor appends comma-separated arrays of length == blockCount
  - BED12-only (default) output is unchanged when the flag is off
  - EXTRA_FIELD_COUNTS matches the number of comma arrays written
"""
from __future__ import annotations

import io
from array import array

import numpy as np
import pytest

from fiberhmm.io.autosql import (
    AUTOSQL_SCHEMAS,
    EXTRA_FIELD_COUNTS,
    get_schema,
    write_autosql_for,
)
from fiberhmm.cli.extract_tags import (
    _extract_footprints,
    _extract_msps,
    _extract_ry,
    _extract_tfs,
)


class _FakeRead:
    """Minimal pysam-like read backed by an identity query->ref mapping.

    Only implements the tag/identity surface the extractors use. We avoid
    pysam fixtures to keep the test fast and hermetic.
    """
    def __init__(self, read_id='r', ref_name='chr1', is_reverse=False,
                 query_len=2000, ref_start=1_000_000):
        self._tags = {}
        self.query_name = read_id
        self.reference_name = ref_name
        self.is_reverse = is_reverse
        self._query_len = query_len
        self._ref_start = ref_start

    def has_tag(self, t):
        return t in self._tags

    def get_tag(self, t):
        if t not in self._tags:
            raise KeyError(t)
        return self._tags[t]

    def set_tag(self, t, v, value_type=None):
        if v is None:
            self._tags.pop(t, None)
        else:
            self._tags[t] = v


def _identity_map(read):
    """Identity query->ref map offset by ref_start. Matches what
    cigar_to_query_ref would produce for an ungapped alignment.
    """
    return np.arange(read._ref_start, read._ref_start + read._query_len,
                     dtype=np.int64)


# ------------------- autoSQL schemas --------------------------------

def test_autosql_default_is_bed12_only():
    """AUTOSQL_SCHEMAS has the classic BED12 shape (12 columns)."""
    for t, schema in AUTOSQL_SCHEMAS.items():
        # Classic BED12 fields; no per-block score arrays.
        assert 'chromStarts' in schema
        assert 'blockNq' not in schema
        assert 'blockAq' not in schema
        assert 'blockTq' not in schema
        assert 'blockMl' not in schema


def test_autosql_block_scores_adds_per_type_columns():
    foot = get_schema('footprint', block_scores=True)
    assert 'blockNq' in foot
    assert 'blockAq' not in foot

    msp = get_schema('msp', block_scores=True)
    assert 'blockAq' in msp
    assert 'blockNq' not in msp

    tf = get_schema('tf', block_scores=True)
    assert 'blockTq' in tf
    assert 'blockEl' in tf
    assert 'blockEr' in tf

    for t in ('m6a', 'm5c'):
        schema = get_schema(t, block_scores=True)
        assert 'blockMl' in schema


def test_extra_field_counts_match_written_arrays():
    """EXTRA_FIELD_COUNTS must match the number of int[blockCount] fields
    in the block_scores schema -- this is the number passed as
    ``-type=bed12+N`` to bedToBigBed."""
    assert EXTRA_FIELD_COUNTS['footprint'] == 1
    assert EXTRA_FIELD_COUNTS['msp'] == 1
    assert EXTRA_FIELD_COUNTS['tf'] == 3
    assert EXTRA_FIELD_COUNTS['m6a'] == 1
    assert EXTRA_FIELD_COUNTS['m5c'] == 1

    for t, n in EXTRA_FIELD_COUNTS.items():
        schema = get_schema(t, block_scores=True)
        assert schema.count('int[blockCount]') == 2 + n, (
            f'{t}: expected 2 base arrays + {n} extras, got '
            f'{schema.count("int[blockCount]")}')


def test_write_autosql_for_writes_both_variants(tmp_path):
    classic = write_autosql_for('tf', out_dir=str(tmp_path), block_scores=False)
    bs = write_autosql_for('tf', out_dir=str(tmp_path), block_scores=True)
    assert classic != bs
    with open(classic) as f:
        assert 'blockTq' not in f.read()
    with open(bs) as f:
        content = f.read()
        assert 'blockTq' in content
        assert 'blockEl' in content
        assert 'blockEr' in content


# ------------------- footprint --------------------------------------

def test_footprint_bed12_when_flag_off():
    read = _FakeRead()
    read.set_tag('ns', array('I', [100, 500, 900]))
    read.set_tag('nl', array('I', [120, 140, 100]))
    read.set_tag('nq', array('B', [180, 220, 160]))

    buf = io.StringIO()
    n = _extract_footprints(read, buf, with_scores=True,
                            query_to_ref=_identity_map(read),
                            block_scores=False)
    assert n == 3
    line = buf.getvalue().rstrip('\n')
    cols = line.split('\t')
    assert len(cols) == 12  # classic BED12
    assert int(cols[9]) == 3   # blockCount
    assert cols[10].count(',') == 2  # blockSizes (3 entries)


def test_footprint_bed12_plus_nq_when_flag_on():
    read = _FakeRead()
    read.set_tag('ns', array('I', [100, 500, 900]))
    read.set_tag('nl', array('I', [120, 140, 100]))
    read.set_tag('nq', array('B', [180, 220, 160]))

    buf = io.StringIO()
    n = _extract_footprints(read, buf, with_scores=False,
                            query_to_ref=_identity_map(read),
                            block_scores=True)
    assert n == 3
    cols = buf.getvalue().rstrip('\n').split('\t')
    assert len(cols) == 13
    block_count = int(cols[9])
    nq_arr = cols[12].split(',')
    assert len(nq_arr) == block_count == 3
    # with_scores=False leaves the mean score column at 0, but per-block nq
    # should still carry real values from the nq tag.
    assert int(cols[4]) == 0
    assert [int(v) for v in nq_arr] == [180, 220, 160]


def test_footprint_missing_nq_when_flag_on_writes_zeros():
    """If the BAM has no nq tag, blockNq array falls back to zeros."""
    read = _FakeRead()
    read.set_tag('ns', array('I', [100, 500]))
    read.set_tag('nl', array('I', [120, 140]))
    # NO nq tag on this read.

    buf = io.StringIO()
    n = _extract_footprints(read, buf, with_scores=False,
                            query_to_ref=_identity_map(read),
                            block_scores=True)
    assert n == 2
    cols = buf.getvalue().rstrip('\n').split('\t')
    assert len(cols) == 13
    assert [int(v) for v in cols[12].split(',')] == [0, 0]


# ------------------- msp --------------------------------------------

def test_msp_bed12_plus_aq_when_flag_on():
    read = _FakeRead()
    read.set_tag('as', array('I', [200, 700]))
    read.set_tag('al', array('I', [150, 180]))
    read.set_tag('aq', array('B', [90, 255]))

    buf = io.StringIO()
    n = _extract_msps(read, buf, with_scores=True,
                      query_to_ref=_identity_map(read),
                      block_scores=True)
    assert n == 2
    cols = buf.getvalue().rstrip('\n').split('\t')
    assert len(cols) == 13
    assert [int(v) for v in cols[12].split(',')] == [90, 255]


# ------------------- tf (MA/AQ tf+QQQ) ------------------------------

def _build_ma_aq(nuc_intervals, tf_intervals, nq_values, tf_qqqs,
                 read_length=2000):
    """Build MA:Z + AQ:B:C tag values matching the Molecular-annotation spec.

    ``tf_qqqs`` is a list of (tq, el, er) triplets, one per tf interval.
    """
    from fiberhmm.io.ma_tags import format_ma_tag, format_aq_array
    ma = format_ma_tag(read_length=read_length,
                       nuc_intervals=nuc_intervals,
                       msp_intervals=(),
                       tf_intervals=tf_intervals)
    tq_vals = [q[0] for q in tf_qqqs]
    el_vals = [q[1] for q in tf_qqqs]
    er_vals = [q[2] for q in tf_qqqs]
    aq = format_aq_array(nq_values=nq_values,
                         tf_q_values=tq_vals,
                         tf_lq_values=el_vals,
                         tf_rq_values=er_vals)
    return ma, aq


def test_tf_bed12_plus_qqq_when_flag_on():
    read = _FakeRead()
    # 1 nuc + 2 tfs, spec-mode MA + AQ
    ma, aq = _build_ma_aq(
        nuc_intervals=[(50, 120)],
        tf_intervals=[(300, 30), (800, 25)],
        nq_values=[200],
        tf_qqqs=[(150, 30, 25), (220, 10, 5)],
    )
    read.set_tag('MA', ma)
    read.set_tag('AQ', aq)

    buf = io.StringIO()
    n = _extract_tfs(read, buf, with_scores=True, min_tq=0,
                     query_to_ref=_identity_map(read),
                     block_scores=True)
    assert n == 2
    cols = buf.getvalue().rstrip('\n').split('\t')
    assert len(cols) == 15   # 12 + 3 (tq, el, er)

    block_count = int(cols[9])
    tq_arr = [int(v) for v in cols[12].split(',')]
    el_arr = [int(v) for v in cols[13].split(',')]
    er_arr = [int(v) for v in cols[14].split(',')]
    assert len(tq_arr) == len(el_arr) == len(er_arr) == block_count == 2
    # Blocks are sorted by ref start; our two TFs were already in order.
    assert tq_arr == [150, 220]
    assert el_arr == [30, 10]
    assert er_arr == [25, 5]


def test_tf_min_tq_filter_drops_low_quality_and_shrinks_blocks():
    read = _FakeRead()
    ma, aq = _build_ma_aq(
        nuc_intervals=[],
        tf_intervals=[(100, 30), (500, 25)],
        nq_values=[],
        tf_qqqs=[(40, 10, 10), (200, 20, 20)],  # first TF is tq=40
    )
    read.set_tag('MA', ma)
    read.set_tag('AQ', aq)

    buf = io.StringIO()
    n = _extract_tfs(read, buf, with_scores=True, min_tq=50,
                     query_to_ref=_identity_map(read),
                     block_scores=True)
    assert n == 1  # only the tq=200 TF survives
    cols = buf.getvalue().rstrip('\n').split('\t')
    assert len(cols) == 15
    assert int(cols[9]) == 1
    assert [int(v) for v in cols[12].split(',')] == [200]


def test_tf_bed12_only_unchanged_when_flag_off():
    """Sanity: BED12-only output has exactly 12 tab-separated columns."""
    read = _FakeRead()
    ma, aq = _build_ma_aq(
        nuc_intervals=[],
        tf_intervals=[(100, 30), (500, 25)],
        nq_values=[],
        tf_qqqs=[(150, 30, 25), (220, 10, 5)],
    )
    read.set_tag('MA', ma)
    read.set_tag('AQ', aq)

    buf = io.StringIO()
    n = _extract_tfs(read, buf, with_scores=True, min_tq=0,
                     query_to_ref=_identity_map(read),
                     block_scores=False)
    assert n == 2
    cols = buf.getvalue().rstrip('\n').split('\t')
    assert len(cols) == 12


# ------------------- ry (DAF IUPAC R/Y) -----------------------------

class _FakeReadWithSeq(_FakeRead):
    """_FakeRead with a mutable query_sequence attribute so we can test
    the R/Y scan against known IUPAC-encoded strings."""
    def __init__(self, seq, **kw):
        super().__init__(**kw)
        self.query_sequence = seq
        self._query_len = len(seq)


def test_ry_extracts_both_codes_and_sorts_by_ref_position():
    """A read with interleaved R and Y should produce one BED row per
    read, with each R/Y as a 1 bp block sorted by reference position."""
    # query: ACGYRTAYR  -> R/Y at positions 3,4,7,8
    seq = 'ACGYRTAYR'
    read = _FakeReadWithSeq(seq, ref_start=1000)

    buf = io.StringIO()
    n = _extract_ry(read, buf, query_to_ref=_identity_map(read),
                    block_scores=False)
    assert n == 4
    cols = buf.getvalue().rstrip('\n').split('\t')
    assert len(cols) == 12
    # chromStart = first mod position = 1003 ; chromEnd = last + 1 = 1009
    assert int(cols[1]) == 1003
    assert int(cols[2]) == 1009
    # score is the 255 sentinel (not probability)
    assert int(cols[4]) == 255
    assert int(cols[9]) == 4  # blockCount
    assert cols[10] == '1,1,1,1'
    assert cols[11] == '0,1,4,5'  # offsets relative to chromStart=1003


def test_ry_block_scores_disambiguates_r_vs_y():
    """With block_scores=True, the blockMod column must encode
    0 for R (GA-strand) and 1 for Y (CT-strand)."""
    seq = 'ACGYYRRYN'   # R/Y at 3,4,5,6,7 -> Y,Y,R,R,Y
    read = _FakeReadWithSeq(seq, ref_start=500)

    buf = io.StringIO()
    n = _extract_ry(read, buf, query_to_ref=_identity_map(read),
                    block_scores=True)
    assert n == 5
    cols = buf.getvalue().rstrip('\n').split('\t')
    assert len(cols) == 13
    assert [int(v) for v in cols[12].split(',')] == [1, 1, 0, 0, 1]


def test_ry_empty_sequence_returns_zero():
    read = _FakeReadWithSeq('ACGT' * 50, ref_start=1_000_000)
    buf = io.StringIO()
    n = _extract_ry(read, buf, query_to_ref=_identity_map(read),
                    block_scores=True)
    assert n == 0
    assert buf.getvalue() == ''


def test_ry_skips_positions_with_no_ref_mapping():
    """Insertion bases (query position with no ref mapping) must not
    produce BED rows with negative or None coordinates."""
    seq = 'ACYGR'  # Y at 2, R at 4
    read = _FakeReadWithSeq(seq, ref_start=2000)
    # Force query position 2 (the Y) to be an insertion (-1).
    q2r = _identity_map(read).copy()
    q2r[2] = -1
    buf = io.StringIO()
    n = _extract_ry(read, buf, query_to_ref=q2r, block_scores=True)
    # Only the R at q=4 should survive.
    assert n == 1
    cols = buf.getvalue().rstrip('\n').split('\t')
    assert int(cols[9]) == 1
    assert cols[12] == '0'  # code 0 = R


def test_ry_schema_has_blockmod_with_block_scores():
    schema = get_schema('ry', block_scores=True)
    # Check for the field declaration specifically, not a substring in the
    # description (which references the field name in prose).
    assert 'int[blockCount] blockMod' in schema
    classic = get_schema('ry', block_scores=False)
    assert 'int[blockCount] blockMod' not in classic


def test_ry_extra_field_count_is_one():
    assert EXTRA_FIELD_COUNTS['ry'] == 1
