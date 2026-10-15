"""Frames of footprint tags when a pipeline stripped @PG (and @CO)."""
import array

import pysam
import pytest

from fiberhmm.io.annotation_frame import (
    MOLECULAR,
    SEQ,
    frame_assumed_warning,
    legacy_tag_frame,
    legacy_tag_frame_report,
    ma_annotation_frame,
    ma_string_is_molecular,
    read_legacy_frame,
    read_ma_frame,
)

L = 1000
STRIPPED = {'HD': {'VN': '1.6'}, 'SQ': [{'SN': 'chr1', 'LN': 10_000}]}


def _reverse_read(ma=None, legacy=True):
    header = pysam.AlignmentHeader.from_dict(STRIPPED)
    read = pysam.AlignedSegment(header)
    read.query_name = 'rev'
    read.flag = 16
    read.reference_id = 0
    read.reference_start = 100
    read.query_sequence = 'A' * L
    read.cigarstring = f'{L}M'
    if legacy:   # molecular-frame nucleosome at molecule 100-149 (SEQ 850-899)
        read.set_tag('ns', array.array('I', [100]))
        read.set_tag('nl', array.array('I', [50]))
        read.set_tag('as', array.array('I', [300]))
        read.set_tag('al', array.array('I', [200]))
    if ma:
        read.set_tag('MA', ma, value_type='Z')
    return read


def test_dot_strand_ma_is_molecular_plus_follows_the_header():
    assert ma_string_is_molecular('1000;nuc.Q:101-50;msp.:301-200')
    assert ma_string_is_molecular('1000;ddda_ucg.:5-20')
    assert not ma_string_is_molecular('1000;nuc+Q:101-50;msp+:301-200')
    assert not ma_string_is_molecular(None)
    dot = _reverse_read('1000;nuc.Q:101-50;msp.:301-200')
    plus = _reverse_read('1000;nuc+Q:101-50;msp+:301-200')
    assert ma_annotation_frame(STRIPPED) == SEQ
    assert read_ma_frame(dot, SEQ) == MOLECULAR
    assert read_ma_frame(plus, SEQ) == SEQ
    assert read_ma_frame(plus, MOLECULAR) == MOLECULAR


def test_chemistry_declaration_decides_the_legacy_frame_first():
    header = dict(STRIPPED, CO=['FIBERHMM-CHEMISTRY:v1:assay=fiber-seq;enzyme=hia5;'
                                'platform=pacbio;mode=pacbio-fiber;model=hia5_pacbio'])
    report = legacy_tag_frame_report(header)
    assert report['source'] == 'chemistry' and report['frame'] == MOLECULAR
    assert legacy_tag_frame(header)[0] == MOLECULAR
    # Even over a merged history whose writers disagree.
    header['PG'] = [{'ID': 'fiberhmm-apply', 'PN': 'fiberhmm-apply', 'DS': 'coord=molecular'},
                    {'ID': 'fiberhmm-call', 'PN': 'fiberhmm-call', 'DS': 'mode=daf'}]
    assert legacy_tag_frame(header)[0] == MOLECULAR
    assert frame_assumed_warning('t', legacy_tag_frame_report(header)) is None


def test_no_header_evidence_assumes_seq_except_for_dot_ma_reads():
    report = legacy_tag_frame_report(STRIPPED)
    assert report['source'] == 'default' and report['frame'] == SEQ
    assert 'assumed SEQ' in frame_assumed_warning('fiberhmm-recall-tfs', report)
    assert read_legacy_frame(_reverse_read('1000;nuc.Q:101-50'), SEQ, True) == MOLECULAR
    assert read_legacy_frame(_reverse_read('1000;nuc+Q:101-50'), SEQ, True) == SEQ
    assert read_legacy_frame(_reverse_read(), SEQ, True) == SEQ
    # With header evidence the header decides.
    assert read_legacy_frame(_reverse_read('1000;nuc.Q:101-50'), SEQ, False) == SEQ


def test_qc_reads_a_stripped_reverse_read_in_its_molecular_frame():
    from fiberhmm.qc.states import msp_intervals_from_tags, tag_frames

    frames = tag_frames(STRIPPED)
    assert frames['legacy'] == SEQ and frames['legacy_assumed']
    # '.' MA (molecular 300-499 = SEQ 500-699 on a 1000 bp reverse read).
    dot = _reverse_read('1000;nuc.Q:101-50;msp.:301-200', legacy=False)
    assert [tuple(i[:2]) for i in msp_intervals_from_tags(dot, frames)] == [(500, 200)]
    plus = _reverse_read('1000;nuc+Q:101-50;msp+:301-200', legacy=False)
    assert [tuple(i[:2]) for i in msp_intervals_from_tags(plus, frames)] == [(300, 200)]
    # Legacy tags of a read whose MA has only '.' m5C groups: molecular too.
    legacy_dot = _reverse_read('1000;ddda_ucg.:5-20')
    assert [tuple(i[:2]) for i in msp_intervals_from_tags(legacy_dot, frames)] == [(500, 200)]
    legacy_plain = _reverse_read()
    assert [tuple(i[:2]) for i in msp_intervals_from_tags(legacy_plain, frames)] == [(300, 200)]


def test_recall_takes_dot_ma_reads_as_molecular_on_a_stripped_header(monkeypatch, capsys):
    from types import SimpleNamespace

    from fiberhmm.cli import recall_tfs

    monkeypatch.setattr(recall_tfs, '_UNDECIDED_FRAME', {'reason': '', 'assumed': False})
    monkeypatch.setattr(recall_tfs, '_MA_FRAME', {'molecular': False})
    frame = recall_tfs._resolve_input_molecular_frame(SimpleNamespace(input_frame='auto'),
                                                      STRIPPED)
    assert frame is False
    assert 'assumed SEQ' in capsys.readouterr().err
    dot = recall_tfs._make_payload(_reverse_read('1000;ddda_ucg.:5-20'), 'daf', frame)
    assert dot['_input_molecular_frame'] is True
    plain = recall_tfs._make_payload(_reverse_read(), 'daf', frame)
    assert '_input_molecular_frame' not in plain
    # An MA-only read with '.' types converts its molecular MA to the run frame.
    ma_only = _reverse_read('1000;nuc.Q:101-50;msp.:301-200', legacy=False)
    payload = recall_tfs._make_payload(ma_only, 'daf', frame)
    assert payload['tags']['ns'] == [100] and payload['_input_molecular_frame'] is True
    # The worker then flips the read's legacy tags to SEQ as for a molecular run.
    monkeypatch.setitem(recall_tfs._WORKER, 'input_molecular_frame', False)
    assert payload.get('_input_molecular_frame',
                       recall_tfs._WORKER['input_molecular_frame']) is True


def test_header_evidence_still_wins(monkeypatch):
    from types import SimpleNamespace

    from fiberhmm.cli import recall_tfs

    monkeypatch.setattr(recall_tfs, '_UNDECIDED_FRAME', {'reason': '', 'assumed': False})
    header = dict(STRIPPED, PG=[{'ID': 'fiberhmm-call', 'PN': 'fiberhmm-call',
                                 'DS': 'FiberHMM fused apply+recall; mode=daf'}])
    frame = recall_tfs._resolve_input_molecular_frame(SimpleNamespace(input_frame='auto'),
                                                      header)
    assert frame is False
    payload = recall_tfs._make_payload(_reverse_read('1000;nuc.Q:101-50'), 'daf', frame)
    assert '_input_molecular_frame' not in payload
    _ = pytest
