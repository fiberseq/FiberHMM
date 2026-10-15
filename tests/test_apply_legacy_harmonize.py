"""fiberhmm-apply --harmonize-legacy-tags (opt-in; default output unchanged)."""
import numpy as np
import pysam

from fiberhmm.inference.streaming_drain import strip_for_apply
from fiberhmm.inference.tagging import set_legacy_apply_tags
from fiberhmm.io.ma_tags import parse_ma_tag

# 180-219 is a TF-sized footprint inside the MSP 150-259.
RESULT = {'ns': np.array([10, 180, 260]), 'nl': np.array([140, 40, 120]),
          'ns_scores': np.array([0.9, 0.5, 0.8]),
          'as': np.array([150]), 'al': np.array([110]), 'as_scores': np.array([0.7])}


def _read(ma=None):
    header = pysam.AlignmentHeader.from_dict(
        {'HD': {'VN': '1.6'}, 'SQ': [{'SN': 'chr1', 'LN': 10_000}]})
    read = pysam.AlignedSegment(header)
    read.query_name = 'r'
    read.flag = 0
    read.reference_id = 0
    read.reference_start = 0
    read.query_sequence = 'A' * 400
    read.cigarstring = '400M'
    if ma:
        read.set_tag('MA', ma, value_type='Z')
    return read


def _legacy(read):
    return {tag: list(read.get_tag(tag)) for tag in ('ns', 'nl', 'nq', 'as', 'al', 'aq')
            if read.has_tag(tag)}


def test_default_keeps_every_footprint_in_ns():
    read = _read()
    set_legacy_apply_tags(read, RESULT, with_scores=True)
    assert _legacy(read)['ns'] == [10, 180, 260]
    assert not read.has_tag('MA')


def test_harmonized_ns_is_the_ma_nucleosome_view():
    raw, harmonized = _read(), _read()
    set_legacy_apply_tags(raw, RESULT, with_scores=True)
    set_legacy_apply_tags(harmonized, RESULT, with_scores=True, harmonize=True)
    a, b = _legacy(raw), _legacy(harmonized)
    assert b['ns'] == [10, 260] and b['nl'] == [140, 120]
    assert b['nq'] == [a['nq'][0], a['nq'][2]]
    for tag in ('as', 'al', 'aq'):
        assert a[tag] == b[tag]
    assert not harmonized.has_tag('MA')


def test_harmonized_island_read_legacy_equals_ma_and_ma_is_unchanged():
    island = '400;ddda_ucg.:101-100'
    raw, harmonized = _read(island), _read(island)
    for read, flag in ((raw, False), (harmonized, True)):
        strip_for_apply(read)
        set_legacy_apply_tags(read, RESULT, with_scores=True, harmonize=flag)
    assert raw.get_tag('MA') == harmonized.get_tag('MA')
    assert list(raw.get_tag('AQ')) == list(harmonized.get_tag('AQ'))
    nucs = sorted(map(tuple, parse_ma_tag(harmonized.get_tag('MA'))['nuc']))
    assert sorted(zip(harmonized.get_tag('ns'), harmonized.get_tag('nl'))) == nucs
    msps = sorted(map(tuple, parse_ma_tag(harmonized.get_tag('MA'))['msp']))
    assert sorted(zip(harmonized.get_tag('as'), harmonized.get_tag('al'))) == msps


def test_without_msps_harmonizing_changes_nothing():
    raw, harmonized = _read(), _read()
    set_legacy_apply_tags(raw, RESULT, with_scores=True, write_msps=False)
    set_legacy_apply_tags(harmonized, RESULT, with_scores=True, write_msps=False,
                          harmonize=True)
    assert _legacy(raw) == _legacy(harmonized)


def test_recall_refuses_harmonized_apply_output():
    from fiberhmm.cli.recall_tfs import _harmonized_apply_input

    def header(*records):
        return {'PG': [dict(record, ID=f"p{i}") for i, record in enumerate(records)]}

    harmonized = {'PN': 'fiberhmm-apply', 'DS': 'FiberHMM apply; legacy_tags=harmonized'}
    raw = {'PN': 'fiberhmm-apply', 'DS': 'FiberHMM apply; cpg_mask=off'}
    call = {'PN': 'fiberhmm-call', 'DS': 'FiberHMM fused apply+recall'}
    assert _harmonized_apply_input(header(harmonized))
    assert not _harmonized_apply_input(header(raw))
    assert not _harmonized_apply_input(header(harmonized, call))   # re-called since
    assert _harmonized_apply_input(header(call, harmonized))
    assert not _harmonized_apply_input(header())


def test_recall_tfs_cli_exits_2_on_harmonized_input(tmp_path, capsys):
    import sys

    import pytest

    from fiberhmm.cli import recall_tfs

    bam = tmp_path / 'h.bam'
    header = {'HD': {'VN': '1.6'}, 'SQ': [{'SN': 'chr1', 'LN': 1000}],
              'PG': [{'ID': 'fiberhmm-apply', 'PN': 'fiberhmm-apply',
                      'DS': 'FiberHMM apply; mode=pacbio-fiber legacy_tags=harmonized'}]}
    with pysam.AlignmentFile(str(bam), 'wb', header=header):
        pass
    argv = sys.argv
    sys.argv = ['fiberhmm-recall-tfs', '-i', str(bam), '-o', str(tmp_path / 'o.bam'),
                '--enzyme', 'hia5']
    try:
        with pytest.raises(SystemExit) as exc:
            recall_tfs.main()
    finally:
        sys.argv = argv
    assert exc.value.code == 2
    assert '--harmonize-legacy-tags' in capsys.readouterr().err
