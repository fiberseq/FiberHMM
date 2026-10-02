"""fiberhmm-extract keeps intervals whose edge bases are soft-clipped or inserted.

An interval's first or last query base often has no reference position on
minimap2 ONT/DAF alignments: the terminal MSP of a soft-clipped read starts
inside the clip, and 1-bp insertions land on interval edges. Extract used to
drop such intervals (11-16% of ONT MSPs); it now maps each interval to the
aligned bases inside it and drops it only when none is aligned.
"""
from __future__ import annotations

import io

import pysam

from fiberhmm.cli.extract_tags import (
    _extract_both_strand,
    _extract_footprints,
    _extract_msps,
    _extract_tfs,
)
from fiberhmm.core.bam_reader import cigar_to_query_ref
from fiberhmm.io.ma_tags import format_aq_array, format_ma_tag

REF_START = 1000
# 20S 100M 1I 100M 20S: query 0-19 clipped, 120 inserted, 221-240 clipped.
CIGAR = [(4, 20), (0, 100), (1, 1), (0, 100), (4, 20)]
READ_LENGTH = 241


def _read(is_reverse=False):
    header = pysam.AlignmentHeader.from_dict(
        {'HD': {'VN': '1.6'}, 'SQ': [{'SN': 'chr1', 'LN': 10_000}]})
    read = pysam.AlignedSegment(header)
    read.query_name = 'r1'
    read.query_sequence = 'A' * READ_LENGTH
    read.flag = 16 if is_reverse else 0
    read.reference_id = 0
    read.reference_start = REF_START
    read.mapping_quality = 60
    read.cigartuples = CIGAR
    return read


def _blocks(row):
    fields = row.rstrip('\n').split('\t')
    start = int(fields[1])
    sizes = [int(x) for x in fields[10].split(',') if x]
    offsets = [int(x) for x in fields[11].split(',') if x]
    return [(start + o, start + o + s) for o, s in zip(offsets, sizes)]


def test_span_snaps_to_aligned_bases_and_drops_only_unaligned_spans():
    from fiberhmm.cli.extract_tags import _query_span_to_ref_block

    q2r = cigar_to_query_ref(_read())
    # Starts in the 5' clip: snaps to the first aligned base.
    assert _query_span_to_ref_block(q2r, 0, 50) == (REF_START, REF_START + 30)
    # Ends on the insertion: snaps to the last aligned base before it.
    assert _query_span_to_ref_block(q2r, 60, 121) == (REF_START + 40, REF_START + 100)
    # Starts on the insertion.
    assert _query_span_to_ref_block(q2r, 120, 130) == (REF_START + 100, REF_START + 109)
    # Runs into the 3' clip.
    assert _query_span_to_ref_block(q2r, 200, 241) == (REF_START + 179, REF_START + 200)
    # Entirely clipped or inserted: dropped.
    assert _query_span_to_ref_block(q2r, 0, 20) is None
    assert _query_span_to_ref_block(q2r, 120, 121) is None
    assert _query_span_to_ref_block(q2r, 225, 241) is None
    # Fully aligned interval: unchanged from the old endpoint lookup.
    assert _query_span_to_ref_block(q2r, 30, 60) == (REF_START + 10, REF_START + 40)


def test_ma_intervals_with_clipped_or_inserted_edges_are_extracted():
    read = _read()
    # (start, length) in molecular == SEQ frame (forward read).
    nucs = [(0, 60), (121, 100)]          # starts in the 5' clip; ends at the 3' clip
    msps = [(60, 61), (221, 20)]          # ends on the insertion; entirely clipped
    tfs = [(20, 10)]
    read.set_tag('MA', format_ma_tag(READ_LENGTH, nucs, msps, tfs))
    read.set_tag('AQ', format_aq_array([200, 210], [100], [1], [2]))
    q2r = cigar_to_query_ref(read)

    out = io.StringIO()
    assert _extract_msps(read, out, False, q2r) == 1
    assert _blocks(out.getvalue()) == [(REF_START + 40, REF_START + 100)]

    out = io.StringIO()
    assert _extract_footprints(read, out, False, q2r) == 2
    assert _blocks(out.getvalue()) == [
        (REF_START, REF_START + 40), (REF_START + 100, REF_START + 200)]

    out = io.StringIO()
    assert _extract_tfs(read, out, False, 0, q2r) == 1


def test_legacy_tags_with_clipped_edges_are_extracted():
    read = _read()
    read.set_tag('ns', [0, 121])
    read.set_tag('nl', [60, 120])        # second nuc runs into the 3' clip
    read.set_tag('as', [60])
    read.set_tag('al', [61])             # ends on the insertion
    q2r = cigar_to_query_ref(read)

    out = io.StringIO()
    assert _extract_footprints(read, out, False, q2r) == 2
    assert _blocks(out.getvalue()) == [
        (REF_START, REF_START + 40), (REF_START + 100, REF_START + 200)]

    out = io.StringIO()
    assert _extract_msps(read, out, False, q2r) == 1
    assert _blocks(out.getvalue()) == [(REF_START + 40, REF_START + 100)]


def test_reverse_read_legacy_tags_with_clipped_edges_are_extracted():
    # Molecular-frame tags on a reverse read: [0, 60) molecular is the SEQ
    # span [181, 241), which runs into the 3' clip of the stored sequence.
    read = _read(is_reverse=True)
    read.set_tag('ns', [0])
    read.set_tag('nl', [60])
    q2r = cigar_to_query_ref(read)

    out = io.StringIO()
    assert _extract_footprints(read, out, False, q2r) == 1
    assert _blocks(out.getvalue()) == [(REF_START + 160, REF_START + 200)]


def test_duplex_both_strand_span_with_clipped_edge_is_extracted():
    read = _read()
    read.set_tag(
        'MA', f'{READ_LENGTH};deam+.:1-60;deam-.:1-50')
    q2r = cigar_to_query_ref(read)

    out = io.StringIO()
    assert _extract_both_strand(read, out, q2r) == 1
    assert _blocks(out.getvalue()) == [(REF_START, REF_START + 30)]
