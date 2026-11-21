"""The MSP bigBed schema names aq as fibertools FIRE only when it may be.

A BAM with a ``fiberhmm-apply`` @PG and no FIBERHMM-ANNOTATION-CONTRACT line
comes from FiberHMM <= 3.2.1, whose ``apply --scores`` wrote its MSP posterior
as ``aq``; extract then describes the score as ambiguous.
"""
from __future__ import annotations

from array import array

import pysam

from fiberhmm.cli import extract_tags
from fiberhmm.io.autosql import get_schema, write_autosql_for
from fiberhmm.io.bam_header import ANNOTATION_CONTRACT_V2


def test_schema_text():
    plain = get_schema('msp', block_scores=True)
    amb = get_schema('msp', block_scores=True, msp_aq_ambiguous=True)
    assert 'mean fibertools FIRE aq' in plain and 'ambiguous' not in plain
    assert 'ambiguous' in amb and 'mean fibertools FIRE aq' not in amb
    assert amb.count('int[blockCount]') == plain.count('int[blockCount]')
    # only the MSP schema changes
    assert get_schema('tf', msp_aq_ambiguous=True) == get_schema('tf')


def test_written_variants_do_not_collide(tmp_path):
    a = write_autosql_for('msp', out_dir=str(tmp_path))
    b = write_autosql_for('msp', out_dir=str(tmp_path), msp_aq_ambiguous=True)
    assert a != b and 'ambiguous' in open(b).read()
    assert 'ambiguous' not in open(a).read()


def _bam(path, pgs, co=()):
    header = {'HD': {'VN': '1.6'}, 'SQ': [{'SN': 'chrT', 'LN': 10_000}], 'PG': pgs}
    if co:
        header['CO'] = list(co)
    with pysam.AlignmentFile(str(path), 'wb', header=header) as out:
        r = pysam.AlignedSegment(out.header)
        r.query_name = 'r1'
        r.reference_id = 0
        r.reference_start = 100
        r.mapping_quality = 60
        r.cigarstring = '400M'
        r.query_sequence = 'A' * 400
        r.set_tag('as', array('I', [0, 250]))
        r.set_tag('al', array('I', [100, 150]))
        r.set_tag('aq', array('B', [0, 229]))
        out.write(r)
    pysam.index(str(path))


OLD_APPLY = {'ID': 'fiberhmm-apply', 'PN': 'fiberhmm-apply', 'VN': '3.2.1',
             'CL': 'fiberhmm-apply -i in.bam -o out --scores'}
FT = {'ID': 'ft', 'PN': 'ft', 'VN': '0.6.2', 'CL': 'ft fire in.bam out.bam'}


def _schema_seen(tmp_path, monkeypatch, pgs, co=()):
    tmp_path.mkdir(exist_ok=True)
    bam = tmp_path / 'in.bam'
    _bam(bam, pgs, co)
    seen = {}

    def fake(bed_path, bb_path, chrom_sizes, bed_type, **kw):
        if kw.get('extract_type') == 'msp':
            seen['ambiguous'] = kw.get('msp_aq_ambiguous')
        return True

    monkeypatch.setattr(extract_tags, 'bed_to_bigbed', fake)
    monkeypatch.setattr('sys.argv', ['fiberhmm-extract', '-i', str(bam), '-o',
                                     str(tmp_path / 'out'), '--msp', '-c', '1'])
    try:
        extract_tags.main()
    except SystemExit as exc:
        assert not exc.code
    return seen['ambiguous']


def test_old_fiberhmm_apply_source_is_ambiguous(tmp_path, monkeypatch):
    assert _schema_seen(tmp_path, monkeypatch, [OLD_APPLY, FT]) is True


def test_fibertools_or_contract_source_is_fire(tmp_path, monkeypatch):
    assert _schema_seen(tmp_path, monkeypatch, [FT]) is False
    assert _schema_seen(tmp_path / 'c', monkeypatch, [OLD_APPLY, FT],
                        co=[ANNOTATION_CONTRACT_V2]) is False
