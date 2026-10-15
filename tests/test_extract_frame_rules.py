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
