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



def test_harmonized_reads_carry_a_marker_that_recall_refuses(tmp_path):
    from fiberhmm.cli.recall_tfs import HarmonizedInputError, _refuse_harmonized_read
    from fiberhmm.inference.read_filters import HARMONIZED_LEGACY_TAG
    from fiberhmm.inference.streaming_drain import strip_for_apply, strip_for_fused_call

    read = _read()
    set_legacy_apply_tags(read, RESULT, with_scores=True, harmonize=True)
    assert read.get_tag(HARMONIZED_LEGACY_TAG) == 1
    import pytest
    with pytest.raises(HarmonizedInputError):
        _refuse_harmonized_read(read)
    # A raw apply or a fiberhmm-call pass rewrites ns/nl and drops the marker.
    for strip in (strip_for_apply, strip_for_fused_call):
        marked = _read()
        set_legacy_apply_tags(marked, RESULT, with_scores=True, harmonize=True)
        strip(marked)
        assert not marked.has_tag(HARMONIZED_LEGACY_TAG)
    raw = _read()
    set_legacy_apply_tags(raw, RESULT, with_scores=True)
    assert not raw.has_tag(HARMONIZED_LEGACY_TAG)
    _refuse_harmonized_read(raw)


def _isolate_recall_state(monkeypatch):
    """main() configures recall_tfs's per-process worker state; restore it."""
    import copy

    from fiberhmm.cli import recall_tfs

    for name in dir(recall_tfs):
        value = getattr(recall_tfs, name)
        if name.startswith('_') and name.lstrip('_').isupper() and isinstance(value, dict):
            monkeypatch.setattr(recall_tfs, name, copy.deepcopy(value))


def test_recall_refuses_a_marked_read_whatever_the_header(tmp_path, capsys, monkeypatch):
    import sys

    import pytest

    from fiberhmm.cli import recall_tfs

    _isolate_recall_state(monkeypatch)

    header = {'HD': {'VN': '1.6'}, 'SQ': [{'SN': 'chr1', 'LN': 10_000}],
              'PG': [{'ID': 'fiberhmm-call', 'PN': 'fiberhmm-call',
                      'DS': 'FiberHMM fused apply+recall; mode=pacbio-fiber enzyme=hia5'}]}
    bam = tmp_path / 'm.bam'
    with pysam.AlignmentFile(str(bam), 'wb', header=header) as out:
        read = pysam.AlignedSegment(out.header)
        read.query_name = 'r'
        read.flag = 0
        read.reference_id = 0
        read.reference_start = 0
        read.query_sequence = 'A' * 400
        read.cigarstring = '400M'
        set_legacy_apply_tags(read, RESULT, with_scores=False, harmonize=True)
        out.write(read)
    argv = sys.argv
    sys.argv = ['fiberhmm-recall-tfs', '-i', str(bam), '-o', str(tmp_path / 'o.bam'),
                '--enzyme', 'hia5', '-c', '1', '--force-seq']
    try:
        with pytest.raises(SystemExit) as exc:
            recall_tfs.main()
    finally:
        sys.argv = argv
    assert exc.value.code == 2
    assert 'carries hl' in capsys.readouterr().err
    assert not (tmp_path / 'o.bam').exists()


def test_region_api_output_carries_the_harmonization_provenance(tmp_path):
    """process_bam_for_footprints(region_parallel=True) writes the run's @PG
    (legacy_tags=harmonized), and every harmonized read is marked."""
    from fiberhmm.inference.parallel import process_bam_for_footprints
    from fiberhmm.inference.read_filters import HARMONIZED_LEGACY_TAG
    from fiberhmm.models import get_model_path
    from conftest import make_synthetic_bam

    bam = str(tmp_path / 'in.bam')
    make_synthetic_bam(bam, n_reads=6, read_length=1500, n_chroms=1,
                       chrom_length=20_000, seed=3)
    pysam.index(bam)
    out = str(tmp_path / 'out.bam')
    pg = {'PN': 'fiberhmm-apply', 'VN': '3.0.0', 'CL': 'fiberhmm-apply',
          'DS': 'FiberHMM apply; coord=molecular legacy_tags=harmonized'}
    process_bam_for_footprints(
        bam, out, get_model_path('hia5', tool='apply', seq='pacbio'), set(), 10, False,
        'pacbio-fiber', 3, 0, n_cores=1, region_parallel=True, pg_record=pg,
        harmonize_legacy_tags=True)
    with pysam.AlignmentFile(out, check_sq=False) as result:
        assert any('legacy_tags=harmonized' in str(p.get('DS', ''))
                   for p in result.header.to_dict()['PG'])
        reads = list(result)
    called = [r for r in reads if r.has_tag('ns')]
    assert called and all(r.has_tag(HARMONIZED_LEGACY_TAG) for r in called)

