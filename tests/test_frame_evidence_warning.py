"""One warning per input file whose header gives no frame evidence
(annotation_frame.warn_if_frames_unverifiable); silent for 3.0 output,
fibertools output and files without footprint annotations."""
import array
import shutil
from pathlib import Path

import pysam
import pytest

from fiberhmm.io import annotation_frame as af

FX = Path(__file__).parent / 'fixtures' / 'fibertools_frame'


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    monkeypatch.setattr(af, '_WARNED_PATHS', set())
    monkeypatch.setattr(af, '_CHECKED_PATHS', set())


def _bam(path, header_extra=None, tags=None):
    header = {'HD': {'VN': '1.6', 'SO': 'coordinate'}, 'SQ': [{'SN': 'chr1', 'LN': 10_000}],
              **(header_extra or {})}
    with pysam.AlignmentFile(str(path), 'wb', header=header) as out:
        for i in range(3):
            read = pysam.AlignedSegment(out.header)
            read.query_name, read.flag, read.reference_id = f'r{i}', 16, 0
            read.reference_start, read.mapping_quality = 1000 + i, 60
            read.query_sequence, read.cigarstring = 'A' * 400, '400M'
            for tag, value in (tags or {}).items():
                read.set_tag(tag, value)
            out.write(read)
    pysam.index(str(path))
    return str(path)


def test_stripped_header_with_footprints_warns_once(tmp_path, capsys):
    path = _bam(tmp_path / 'stripped.bam', tags={'MA': '400;nuc+Q:11-100', 'AQ': array.array('B', [9])})
    assert af.warn_if_frames_unverifiable(path)
    assert not af.warn_if_frames_unverifiable(path)          # once per file
    err = capsys.readouterr().err
    assert err.count('warning:') == 1
    assert f"warning: {path}: header has no FiberHMM/fibertools provenance" in err
    legacy = _bam(tmp_path / 'legacy.bam', tags={'ns': array.array('I', [10]), 'nl': array.array('I', [5])})
    assert af.warn_if_frames_unverifiable(legacy)


@pytest.mark.parametrize('name', ['fiberhmm3.0.0_call.bam', 'ft0.13_addnuc_fire.bam',
                                  'ft0.6.2_addnuc_fire.bam', 'ft0.13_predict_m6a.bam'])
def test_3_0_call_output_and_fibertools_output_are_silent(name, capsys):
    assert not af.warn_if_frames_unverifiable(str(FX / name))
    assert capsys.readouterr().err == ''


def test_no_footprints_or_only_always_molecular_annotations_are_silent(tmp_path, capsys):
    raw = _bam(tmp_path / 'raw.bam')                                       # aligned reads
    m5c = _bam(tmp_path / 'm5c.bam', tags={'MA': '400;ddda_mcg.:11-100'})   # m5C: molecular
    ft = _bam(tmp_path / 'ma.bam', tags={'Ma': '400;nuc.:11-100'})          # fibertools Ma
    evidence = _bam(tmp_path / 'ev.bam', header_extra={'CO': ['fiberhmm:coord=molecular']},
                    tags={'MA': '400;nuc.:11-100'})
    for path in (raw, m5c, ft, evidence, '-'):
        assert not af.warn_if_frames_unverifiable(path)
    assert capsys.readouterr().err == ''


def test_commands_warn_once_through_the_shared_helper(tmp_path, capsys):
    """extract (and every reader that calls the helper) warns once per file."""
    from fiberhmm.cli.extract_tags import extract_tags_parallel

    path = _bam(tmp_path / 'stripped.bam', tags={'MA': '400;nuc+Q:11-100', 'AQ': array.array('B', [9])})
    extract_tags_parallel(path, {'nucleosome': str(tmp_path / 'n.bed'), 'tf': str(tmp_path / 't.bed')},
                          ['nucleosome', 'tf'])
    extract_tags_parallel(path, {'nucleosome': str(tmp_path / 'n2.bed')}, ['nucleosome'])
    assert capsys.readouterr().err.count('header has no FiberHMM/fibertools provenance') == 1
