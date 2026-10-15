"""fiberhmm-extract reads footprint tags by the shared frame rules
(fiberhmm.io.annotation_frame; docs/reference/footprint-tag-frame.md).

One reverse read, 400 bp at chr1:1000, with a 100 bp footprint at query 10-110
(SEQ frame): genomic 1010-1110. Read as molecular, the same numbers mirror to
1290-1390.
"""
import array
import os
import subprocess
import sys
from pathlib import Path

import pysam
import pytest

from fiberhmm.cli.extract_tags import ExtractFrameError, extract_tags_parallel

ROOT = Path(__file__).resolve().parents[1]
SEQ_SPAN = ['1010', '1110']
MIRRORED = ['1290', '1390']

OLD_APPLY = {'ID': 'fiberhmm-apply', 'PN': 'fiberhmm-apply', 'VN': '2.12.0'}
NEW_APPLY = {'ID': 'fiberhmm-apply', 'PN': 'fiberhmm-apply', 'VN': '3.0.0',
             'DS': 'FiberHMM apply; coord=molecular'}
FIBERTOOLS = {'ID': 'ft', 'CL': 'ft add-nucleosomes in.bam out.bam'}
# samtools merge of an old FiberHMM file and a fibertools file: two @PG
# chains whose footprint writers disagree, and no coord=molecular anywhere.
MERGED = [{'ID': 'pbmm2', 'PN': 'pbmm2'}, dict(OLD_APPLY, PP='pbmm2'),
          dict(FIBERTOOLS, ID='ft.1', PP='pbmm2')]


def _bam(path, pg=(), ma=None, msp=False):
    header = {'HD': {'VN': '1.6', 'SO': 'coordinate'},
              'SQ': [{'SN': 'chr1', 'LN': 10_000}]}
    if pg:
        header['PG'] = [dict(p) for p in pg]
    with pysam.AlignmentFile(str(path), 'wb', header=header) as out:
        read = pysam.AlignedSegment(out.header)
        read.query_name = 'rev'
        read.flag = 16
        read.reference_id = 0
        read.reference_start = 1000
        read.mapping_quality = 60
        read.query_sequence = 'A' * 400
        read.cigarstring = '400M'
        if ma:
            read.set_tag('MA', ma)
        start_tag, length_tag = ('as', 'al') if msp else ('ns', 'nl')
        read.set_tag(start_tag, array.array('I', [10]))
        read.set_tag(length_tag, array.array('I', [100]))
        out.write(read)
    pysam.index(str(path))
    return str(path)


def _span(path, tmp_path, kind='nucleosome', **kwargs):
    bed = tmp_path / f'{kind}.bed'
    extract_tags_parallel(path, {kind: str(bed)}, [kind], with_scores=False, **kwargs)
    return bed.read_text().split('\t')[1:3]


@pytest.mark.parametrize('pg, expected', [
    ([OLD_APPLY], SEQ_SPAN),             # unmarked FiberHMM <= 2.12: SEQ
    ([NEW_APPLY], MIRRORED),             # declared coord=molecular
    ([FIBERTOOLS], MIRRORED),            # fibertools writes molecular
])
@pytest.mark.parametrize('kind', ['nucleosome', 'msp'])
def test_legacy_tags_follow_provenance(tmp_path, pg, expected, kind):
    path = _bam(tmp_path / 'in.bam', pg, msp=kind == 'msp')
    assert _span(path, tmp_path, kind) == expected


def test_ma_follows_the_header_rule(tmp_path):
    # MA nuc 11-100 (1-based) is query 10-110.
    unmarked = _bam(tmp_path / 'old.bam', [OLD_APPLY], ma='400;nuc+Q:11-100')
    assert _span(unmarked, tmp_path) == SEQ_SPAN
    declared = _bam(tmp_path / 'new.bam', [NEW_APPLY], ma='400;nuc.Q:11-100')
    assert _span(declared, tmp_path) == MIRRORED


def test_header_without_evidence_reads_dot_groups_as_molecular(tmp_path, capsys):
    # No @PG at all: '.' MA groups are molecular, '+' groups SEQ, and legacy
    # tags (read only without MA) are SEQ, with a warning.
    dot = _bam(tmp_path / 'dot.bam', ma='400;nuc.Q:11-100')
    assert _span(dot, tmp_path) == MIRRORED
    plus = _bam(tmp_path / 'plus.bam', ma='400;nuc+Q:11-100')
    assert _span(plus, tmp_path) == SEQ_SPAN
    bare = _bam(tmp_path / 'bare.bam')
    assert _span(bare, tmp_path) == SEQ_SPAN
    assert 'fiberhmm-extract: warning: the header records no footprint writer' in capsys.readouterr().err


def test_explicit_input_frame_overrides_provenance(tmp_path):
    path = _bam(tmp_path / 'in.bam', [OLD_APPLY])
    assert _span(path, tmp_path, input_frame='molecular') == MIRRORED
    path = _bam(tmp_path / 'new.bam', [NEW_APPLY])
    assert _span(path, tmp_path, input_frame='query') == SEQ_SPAN


def test_undecided_merged_history_is_refused(tmp_path):
    path = _bam(tmp_path / 'merged.bam', MERGED)
    with pytest.raises(ExtractFrameError, match='--input-frame'):
        _span(path, tmp_path)
    # Types that never read legacy tags are not affected.
    m6a = tmp_path / 'm6a.bed'
    extract_tags_parallel(path, {'m6a': str(m6a)}, ['m6a'])
    assert _span(path, tmp_path, input_frame='query') == SEQ_SPAN


def test_cli_undecided_frame_exits_2_without_traceback(tmp_path):
    path = _bam(tmp_path / 'merged.bam', MERGED)
    done = subprocess.run(
        [sys.executable, '-c', 'from fiberhmm.cli.extract_tags import main; main()',
         '-i', path, '-o', str(tmp_path / 'out'), '--nucleosome', '--bed-only'],
        capture_output=True, text=True, cwd=ROOT,
        env=dict(os.environ, PYTHONPATH=str(ROOT), FIBERHMM_NO_UPDATE_CHECK='1'))
    assert done.returncode == 2, done.stderr
    assert 'fiberhmm-extract: error:' in done.stderr and '--input-frame' in done.stderr
    assert 'Traceback' not in done.stderr


TAG_M5C = {'ID': 'fiberhmm-tag-m5c', 'PN': 'fiberhmm-tag-m5c', 'VN': '3.0.1',
           'PP': 'fiberhmm-apply',
           'DS': 'DddA molecule-specific mCG; mode=whole_cpg_island; ddda_mcg_frame=molecular'}


def _reverse_read_bam(path, pg, set_tags):
    header = {'HD': {'VN': '1.6', 'SO': 'coordinate'},
              'SQ': [{'SN': 'chr1', 'LN': 10_000}]}
    if pg:
        header['PG'] = [dict(p) for p in pg]
    with pysam.AlignmentFile(str(path), 'wb', header=header) as out:
        read = pysam.AlignedSegment(out.header)
        read.query_name = 'rev'
        read.flag = 16
        read.reference_id = 0
        read.reference_start = 1000
        read.mapping_quality = 60
        read.query_sequence = 'A' * 400
        read.cigarstring = '400M'
        set_tags(read)
        out.write(read)
    pysam.index(str(path))
    return str(path)


@pytest.mark.parametrize('pg', [
    [OLD_APPLY, TAG_M5C],            # SEQ-frame footprints, then tag-m5c
    [NEW_APPLY, dict(TAG_M5C, PP='fiberhmm-apply')],
    [],                              # stripped header: the '.' group decides
])
def test_m5c_groups_are_molecular_as_tag_m5c_writes_them(tmp_path, pg):
    """tag-m5c writes ddda_mcg molecular (ddda_mcg_frame=molecular) whatever
    frame the footprints use; extract reads them as the CpG mask does."""
    from fiberhmm.daf.m5c import add_m5c_ma_tag, ma_intervals

    def tags(read):
        add_m5c_ma_tag(read, [(10, 110)])          # SEQ-frame query span
        assert read.get_tag('MA') == '400;ddda_mcg.:291-100'
        assert ma_intervals(read, 'ddda_mcg') == [(10, 110)]

    path = _reverse_read_bam(tmp_path / 'm5c.bam', pg, tags)
    assert _span(path, tmp_path, 'm5c') == SEQ_SPAN


@pytest.mark.parametrize('pg, expected', [
    ([], ['1050', '1110']),                   # no evidence: '+'/'-' groups are SEQ
    ([OLD_APPLY], ['1050', '1110']),          # unmarked writer: SEQ
    ([NEW_APPLY], ['1290', '1350']),          # declared molecular
])
def test_both_strand_groups_follow_the_ma_rule(tmp_path, pg, expected):
    from fiberhmm.crossstrand.recall import deam_regime_masks

    def tags(read):
        read.set_tag('MA', '400;deam+:11-100;deam-:51-100')

    path = _reverse_read_bam(tmp_path / 'deam.bam', pg, tags)
    assert _span(path, tmp_path, 'bothstrand') == expected
    if not pg:
        # The duplex recall mask reads these groups in SEQ frame too.
        with pysam.AlignmentFile(path) as bam:
            plus, minus = deam_regime_masks(next(iter(bam)))
        both = [i for i in range(400) if plus[i] and minus[i]]
        assert (1000 + both[0], 1000 + both[-1] + 1) == (1050, 1110)


@pytest.mark.parametrize('pg', [[], [NEW_APPLY]])
def test_both_strand_of_a_pair_consensus_read(tmp_path, pg):
    """fiberhmm-pair/-merge consensus reads are forward: the both-strand
    region is the CT/GA overlap whatever the header says."""
    from fiberhmm.cli.merge import _make_consensus_segment
    from fiberhmm.crossstrand.consensus import build_consensus

    header = pysam.AlignmentHeader.from_dict(
        {'HD': {'VN': '1.6', 'SO': 'coordinate'}, 'SQ': [{'SN': 'chr1', 'LN': 10_000}],
         **({'PG': [dict(p) for p in pg]} if pg else {})})

    def source(name, start, seq):
        read = pysam.AlignedSegment(header)
        read.query_name, read.flag, read.reference_id = name, 0, 0
        read.reference_start, read.mapping_quality = start, 60
        read.cigartuples = [(0, len(seq))]
        read.query_sequence = seq
        read.query_qualities = pysam.qualitystring_to_array('I' * len(seq))
        return read

    ct = source('ct', 1100, 'A' * 10 + 'Y' + 'A' * 49)
    ga = source('ga', 1120, 'A' * 5 + 'R' + 'A' * 54)
    cons = build_consensus(ct, ga)
    assert (cons.both_start, cons.both_end) == (1120, 1160)
    path = str(tmp_path / 'pair.bam')
    with pysam.AlignmentFile(path, 'wb', header=header) as out:
        out.write(_make_consensus_segment(cons, out.header, 0))
    pysam.index(path)
    assert _span(path, tmp_path, 'bothstrand') == ['1120', '1160']


# --- m5C groups x header state x orientation, against the CpG-mask reader ---

M5C_HEADERS = {
    'stripped': [],
    'unmarked_writer': [OLD_APPLY],
    'declared': [NEW_APPLY],
    'tag_m5c_only': [dict(TAG_M5C, PP=None)],
}


def _m5c_writers():
    from fiberhmm.daf import m5c
    spans = [(10, 110)]
    other = [(150, 170)]
    return {
        'mcg.': lambda r: m5c.add_m5c_ma_tag(r, spans),
        'island.': lambda r: m5c.add_island_m5c_ma_tags(r, spans, other),
        'paired+-': lambda r: m5c.add_paired_m5c_ma_tags(r, spans, other, [(200, 220)], [(300, 330)]),
        'paired+': lambda r: m5c.add_paired_m5c_ma_tags(r, spans, [], [], []),
        'ucg+-': lambda r: m5c._replace_m5c_ma_groups(
            r, [('ddda_ucg', '+', spans), ('ddda_ucg', '-', other)]),
    }


@pytest.mark.parametrize('state', sorted(M5C_HEADERS))
@pytest.mark.parametrize('writer', sorted(_m5c_writers()))
@pytest.mark.parametrize('reverse', [False, True])
def test_m5c_groups_read_exactly_as_the_cpg_mask_reads_them(state, writer, reverse):
    from fiberhmm.cli.extract_tags import _parse_ma_annotations, resolve_extract_frames
    from fiberhmm.daf.m5c import ma_strand_intervals

    pg = [{k: v for k, v in p.items() if v is not None} for p in M5C_HEADERS[state]]
    header = pysam.AlignmentHeader.from_dict(
        {'HD': {'VN': '1.6'}, 'SQ': [{'SN': 'chr1', 'LN': 10_000}], **({'PG': pg} if pg else {})})
    read = pysam.AlignedSegment(header)
    read.query_name, read.flag, read.reference_id = 'r', 16 if reverse else 0, 0
    read.reference_start, read.query_sequence, read.cigarstring = 1000, 'A' * 400, '400M'
    _m5c_writers()[writer](read)
    frames = resolve_extract_frames(header)
    for feature in ('ddda_mcg', 'ddda_ucg', 'ddda_mcg_hemi'):
        expected = sorted(iv for values in ma_strand_intervals(read, feature).values()
                          for iv in values)
        got = sorted((a['start'], a['start'] + a['length'])
                     for a in (_parse_ma_annotations(read, feature, frames) or []))
        assert got == expected, (feature, read.get_tag('MA'))
    # The writers take SEQ-frame spans: the first one reads back unchanged.
    assert (10, 110) in sorted((a['start'], a['start'] + a['length'])
                               for f in ('ddda_mcg', 'ddda_ucg')
                               for a in (_parse_ma_annotations(read, f, frames) or []))


def test_signed_m5c_span_on_a_stripped_header_end_to_end(tmp_path):
    from fiberhmm.daf.m5c import add_paired_m5c_ma_tags

    def tags(read):
        add_paired_m5c_ma_tags(read, [(10, 110)], [], [], [])
        assert read.get_tag('MA') == '400;ddda_mcg+:291-100'

    path = _reverse_read_bam(tmp_path / 'signed.bam', [], tags)
    assert _span(path, tmp_path, 'm5c') == SEQ_SPAN
    assert _span(path, tmp_path, 'm5c', input_frame='query') == SEQ_SPAN


@pytest.mark.parametrize('header_extra, expected', [
    ({'CO': ['fiberhmm:coord=molecular']}, (1290, 1350)),   # molecular groups
    ({}, (1050, 1110)),                                      # no evidence: SEQ
    ({'PG': [OLD_APPLY]}, (1050, 1110)),                     # unmarked writer: SEQ
])
def test_duplex_mask_and_both_strand_extract_agree(tmp_path, header_extra, expected):
    from fiberhmm.crossstrand.recall import deam_regime_masks

    path = str(tmp_path / 'deam.bam')
    header = {'HD': {'VN': '1.6', 'SO': 'coordinate'},
              'SQ': [{'SN': 'chr1', 'LN': 10_000}], **header_extra}
    with pysam.AlignmentFile(path, 'wb', header=header) as out:
        read = pysam.AlignedSegment(out.header)
        read.query_name, read.flag, read.reference_id = 'rev', 16, 0
        read.reference_start, read.mapping_quality = 1000, 60
        read.query_sequence, read.cigarstring = 'A' * 400, '400M'
        read.set_tag('MA', '400;deam+:11-100;deam-:51-100')
        out.write(read)
    pysam.index(path)
    with pysam.AlignmentFile(path) as bam:
        plus, minus = deam_regime_masks(next(iter(bam)))
    both = [i for i in range(400) if plus[i] and minus[i]]
    assert (1000 + both[0], 1000 + both[-1] + 1) == expected
    assert _span(path, tmp_path, 'bothstrand') == [str(v) for v in expected]
