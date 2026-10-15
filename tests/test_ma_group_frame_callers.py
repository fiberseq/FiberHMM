"""Every reader of MA groups resolves their frame per group through
annotation_frame.MAGroupFrame (docs/reference/footprint-tag-frame.md).

Reads are reverse 400M alignments at chr1:1000 unless stated otherwise: a
query interval [s, e) maps to the reference [1000+s, 1000+e).
"""
import array
from pathlib import Path

import numpy as np
import pysam
import pytest

FX = Path(__file__).parent / 'fixtures' / 'fibertools_frame'
OLD_CALL = {'ID': 'fiberhmm-call', 'PN': 'fiberhmm-call', 'VN': '2.12.0',
            'CL': 'fiberhmm-call -i in.bam -o out.bam'}


def _header(pg=(), co=()):
    header = {'HD': {'VN': '1.6', 'SO': 'coordinate'}, 'SQ': [{'SN': 'chr1', 'LN': 10_000}]}
    if pg:
        header['PG'] = [dict(p) for p in pg]
    if co:
        header['CO'] = list(co)
    return pysam.AlignmentHeader.from_dict(header)


def _read(header, tags, reverse=True):
    read = pysam.AlignedSegment(header)
    read.query_name, read.flag, read.reference_id = 'rev', 16 if reverse else 0, 0
    read.reference_start, read.mapping_quality = 1000, 60
    read.query_sequence, read.cigarstring = 'A' * 400, '400M'
    read.query_qualities = pysam.qualitystring_to_array('I' * 400)
    for tag, value in tags.items():
        read.set_tag(tag, value)
    return read


# --- 1. fiberhmm.daf.m5c.ma_strand_intervals ---------------------------------

@pytest.mark.parametrize('pg, co, nuc, msp', [
    ([OLD_CALL], (), (10, 110), (120, 190)),                          # unmarked: SEQ
    ([OLD_CALL], ('fiberhmm:coord=molecular',), (290, 390), (210, 280)),
    ((), (), (10, 110), (120, 190)),                                  # no evidence, '+' groups
])
def test_m5c_reader_follows_the_group_frame(pg, co, nuc, msp):
    from fiberhmm.daf.m5c import ma_intervals, ma_strand_intervals

    read = _read(_header(pg, co), {'MA': '400;nuc+:11-100;msp+:121-70;ddda_mcg.:291-100'})
    assert ma_intervals(read, 'nuc') == [nuc]
    assert ma_intervals(read, 'msp') == [msp]
    assert ma_intervals(read, 'ddda_mcg') == [(10, 110)]      # m5C: always molecular
    assert ma_strand_intervals(read, 'nuc') == {'+': [nuc]}


@pytest.mark.parametrize('co', [(), ('fiberhmm:coord=molecular',)])
def test_m5c_deam_reader_agrees_with_the_duplex_mask(co):
    from fiberhmm.crossstrand.recall import deam_regime_masks
    from fiberhmm.daf.m5c import ma_strand_intervals

    read = _read(_header([OLD_CALL], co), {'MA': '400;deam+:11-100;deam-:51-100'})
    plus, minus = deam_regime_masks(read)
    coverage = ma_strand_intervals(read, 'deam')
    for strand, mask in (('+', plus), ('-', minus)):
        (start, end), = coverage[strand]
        assert list(np.flatnonzero(mask)) == list(range(start, end))


def test_consensus_hia5_replay_accepts_an_unmarked_scaffold(tmp_path):
    """The replay re-reads the scaffold with ma_intervals; on an unmarked
    FiberHMM header it must read MA as the loader does (SEQ frame)."""
    from fiberhmm.core.model_io import load_model_with_metadata
    from fiberhmm.inference.consensus.adapter import evidence_unit
    from fiberhmm.inference.consensus.upstream_recall import recall_hia5_alignment
    from fiberhmm.inference.strand_rescue import load_region_evidence
    from fiberhmm.inference.tf_recaller import build_llr_tables

    source = FX / 'fiberhmm3.0.0_call.bam'
    path = tmp_path / 'unmarked.bam'
    with pysam.AlignmentFile(str(source)) as bam:
        hd = bam.header.to_dict()
        hd['PG'] = [OLD_CALL]
        hd['CO'] = [c for c in hd.get('CO', []) if 'coord=molecular' not in c]
        header = pysam.AlignmentHeader.from_dict(hd)
        with pysam.AlignmentFile(str(path), 'wb', header=header) as out:
            for read in bam:
                out.write(pysam.AlignedSegment.from_dict(read.to_dict(), header))
    pysam.index(str(path))
    model, k, mode = load_model_with_metadata(
        str(Path(__file__).resolve().parents[1] / 'fiberhmm' / 'models' / 'hia5_pacbio.json'))
    hit, miss = build_llr_tables(model)
    reads = load_region_evidence(
        str(path), 'chrT', 0, 8000, strand_mode='alignment', mode=mode, context_size=k,
        prob_threshold=125, llr_hit=hit, llr_miss=miss, min_mapq=0,
        ma_annotation_frame='auto', legacy_annotation_frame='seq')
    by_name = {r.name: r for r in reads}
    with pysam.AlignmentFile(str(path)) as bam:
        for alignment in bam:
            unit = evidence_unit(by_name[alignment.query_name], model, 'd', [], 0, 8000)
            result = recall_hia5_alignment(alignment, unit, model, 'alignment', mode, k,
                                           125, 5.0, legacy_annotation_frame='seq')
            assert result['status'] == 'recalled', alignment.query_name


# --- 2. consensus export of fibertools Ma ------------------------------------

def test_export_keeps_ma_groups_where_the_output_header_reads_them():
    from fiberhmm.cli.extract_tags import _parse_all_ma_annotations
    from fiberhmm.inference.consensus.bam_export import EXPORT_DS, _append_annotations
    from fiberhmm.io.annotation_frame import MAGroupFrame

    contract = ('FIBERHMM-CONSENSUS-MA:v1:{"coordinates": '
                '"original_source_call_in_molecular_frame", "layers": ["fam"]}')
    for pg, molecular in (((), True),                     # no evidence: '.' molecular
                          ([OLD_CALL], False)):           # unmarked writer: SEQ
        header = _header([*pg, {'ID': 'fiberhmm-consensus', 'PN': 'fiberhmm-consensus',
                                'DS': EXPORT_DS}], [contract])
        read = _read(header, {'Ma': '400;nuc.:11-100'})
        frame = MAGroupFrame(header)
        before = _parse_all_ma_annotations(read)['nuc'][0]          # Ma: molecular
        assert (before['start'], before['length']) == (290, 100)
        rows = [dict(chrom='chr1', interval=[1100, 1120], layer='fam', token='t1',
                     tq=50, fi=1, op=2)]
        assert _append_annotations(read, rows, frame) == 1
        assert read.get_tag('MA').startswith('400;nuc.:11-100' if molecular
                                              else '400;nuc.:291-100')
        parsed = _parse_all_ma_annotations(read, annotation_frame=frame.header_frame,
                                           group_frame=frame)
        nuc, fam = parsed['nuc'][0], parsed['fam'][0]
        assert (nuc['start'], nuc['length']) == (290, 100)          # still chr1:1290-1390
        assert (fam['start'], fam['start'] + fam['length']) == (100, 120)


# --- 3. consensus / strand-rescue loader -------------------------------------

def test_loader_resolves_mixed_qualifiers_per_group(tmp_path):
    from fiberhmm.inference.strand_rescue import N_CTX, load_region_evidence

    path = tmp_path / 'mixed.bam'
    header = _header()
    seq = list('A' * 400)
    with pysam.AlignmentFile(str(path), 'wb', header=header) as out:
        read = _read(out.header, {'MA': '400;tf.:11-20;tf+:101-20'})
        read.query_sequence = ''.join(seq)
        read.query_qualities = pysam.qualitystring_to_array('I' * 400)
        read.set_tag('MM', 'A+a.;')
        read.set_tag('ML', array.array('B', []))
        out.write(read)
    pysam.index(str(path))

    def load(window):
        return load_region_evidence(
            str(path), 'chr1', 900, 1500, strand_mode='alignment', mode='pacbio-fiber',
            context_size=3, prob_threshold=125, llr_hit=np.zeros(N_CTX),
            llr_miss=np.zeros(N_CTX), min_mapq=0, ma_annotation_frame='auto',
            required_annotation_overlap=('tf', *window))

    assert len(load((1370, 1390))) == 1       # the dotted TF (molecular)
    assert len(load((1100, 1120))) == 1       # the '+' TF (SEQ)
    assert len(load((1010, 1030))) == 0       # where a per-name rule put the dotted TF


# --- other MA readers --------------------------------------------------------

@pytest.mark.parametrize('pg, expected', [([OLD_CALL], 1060), ([], 1340)])
def test_pair_dyads_follow_the_group_frame(pg, expected):
    from fiberhmm.crossstrand.pairing import nuc_dyads_ref

    strand = '+' if pg else '.'
    read = _read(_header(pg), {'MA': f'400;nuc{strand}:11-100'})
    assert list(nuc_dyads_ref(read)) == [expected]


@pytest.mark.parametrize('pg, start', [([OLD_CALL], 10), ([], 10),
                                       ([dict(OLD_CALL, VN='3.0.0', DS='coord=molecular')], 290)])
def test_utils_reference_bam_reads_ns_in_its_frame(tmp_path, monkeypatch, pg, start):
    """fiberhmm-utils transfer --reference-bam reads ns/nl by the provenance rule."""
    from types import SimpleNamespace

    from fiberhmm.cli import utils

    path = tmp_path / 'ref.bam'
    with pysam.AlignmentFile(str(path), 'wb', header=_header(pg)) as out:
        out.write(_read(out.header, {'ns': array.array('I', [10]), 'nl': array.array('I', [100])}))
    pysam.index(str(path))
    seen = []

    class Counter:
        def __init__(self, *args):
            pass

        def process_read_with_footprints(self, sequence, mask, edge_trim):
            seen.append(np.flatnonzero(mask))

    monkeypatch.setattr(utils, 'AccessibilityCounter', Counter)
    utils._process_reference_bam(str(path), 3, SimpleNamespace(
        min_mapq=0, edge_trim=0, max_reads=0))
    got = seen[0]
    assert (got[0], got[-1] + 1) == (start, start + 100)
