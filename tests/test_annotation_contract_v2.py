"""FIBERHMM-ANNOTATION-CONTRACT:v2: FiberHMM writes no MSP score, leaves no
stale fibertools annotations after rewriting MSPs, and revalidates FIRE groups.

Fixtures in ``tests/fixtures/fire_contract`` (one forward read with FIRE
elements, one reverse read with no m6A, so ``ft fire`` writes no ``fire``
group for it), made from ``fibertools_frame/fiberhmm3.0.0_call.bam`` with the
second read's ML zeroed, then ``fiberhmm-call --enzyme hia5 --seq pacbio``
(3.2.1), then:

* ``fiberhmm3.2.1_call_ft0.13_fire.bam`` -- ``ft fire`` 0.13.0: every MA group
  moved into ``Ma``/``Aq``, plus ``fire.Q`` on the forward read only;
* ``fiberhmm3.2.1_call_ft0.6.2_fire.bam`` -- ``ft fire`` 0.6.2: ``aq`` parallel
  to ``as``/``al`` (0 for MSPs that are not FIRE elements), MA left alone.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from array import array
from pathlib import Path

import pysam
import pytest

from fiberhmm.io.bam_header import (
    ANNOTATION_CONTRACT_V2,
    append_annotation_contract,
    header_annotation_contract,
)

FX = Path(__file__).parent / 'fixtures' / 'fire_contract'
FT13 = FX / 'fiberhmm3.2.1_call_ft0.13_fire.bam'
FT062 = FX / 'fiberhmm3.2.1_call_ft0.6.2_fire.bam'
REPO = Path(__file__).resolve().parents[1]
STALE = ('Ma', 'Aq', 'An', 'aq')


def _env():
    env = dict(os.environ)
    env.setdefault('FIBERHMM_NO_UPDATE_CHECK', '1')
    env['PYTHONPATH'] = os.pathsep.join(filter(None, [str(REPO), env.get('PYTHONPATH')]))
    return env


def _run(code_or_module, *args, module=True):
    cmd = [sys.executable, '-m', code_or_module] if module else [sys.executable, '-c', code_or_module]
    return subprocess.run([*cmd, *map(str, args)], capture_output=True, text=True,
                          env=_env(), cwd=REPO, timeout=300)


def _groups(read, tag='MA'):
    return {re.split(r'[.+-]', g.partition(':')[0])[0]: g.partition(':')[2].split(',')
            for g in str(read.get_tag(tag)).split(';')[1:] if g}


def _contract_lines(path):
    with pysam.AlignmentFile(str(path)) as bam:
        return [c for c in bam.header.to_dict().get('CO', [])
                if str(c).startswith('FIBERHMM-ANNOTATION-CONTRACT:')]


def _header():
    return pysam.AlignmentHeader.from_dict(
        {'HD': {'VN': '1.6'}, 'SQ': [{'SN': 'chrT', 'LN': 10000}]})


def _read(header, *, reverse=False, length=1000):
    r = pysam.AlignedSegment(header)
    r.query_name = 'r1'
    r.flag = 16 if reverse else 0
    r.reference_id = 0
    r.reference_start = 0
    r.mapping_quality = 60
    r.cigarstring = f'{length}M'
    r.query_sequence = 'A' * length
    return r


# --------------------------------------------------------------------------- #
#  fixtures: what the tools wrote                                               #
# --------------------------------------------------------------------------- #

def test_fixtures_show_the_fibertools_layouts():
    with pysam.AlignmentFile(str(FT13)) as bam:
        fwd, rev = list(bam)
    assert not fwd.has_tag('MA') and 'fire' in _groups(fwd, 'Ma')
    assert set(_groups(fwd, 'Ma')['fire']) <= set(_groups(fwd, 'Ma')['msp'])
    assert 'fire' not in _groups(rev, 'Ma')          # no FIRE element: no group at all
    with pysam.AlignmentFile(str(FT062)) as bam:
        fwd, rev = list(bam)
    assert len(fwd.get_tag('aq')) == len(fwd.get_tag('as'))
    assert set(rev.get_tag('aq')) == {0}


# --------------------------------------------------------------------------- #
#  no FiberHMM MSP score                                                        #
# --------------------------------------------------------------------------- #

def test_set_legacy_apply_tags_with_scores_writes_nq_but_no_aq():
    from fiberhmm.inference.tagging import set_legacy_apply_tags
    read = _read(_header())
    read.set_tag('aq', array('B', [7, 7]))
    result = {'ns': [100, 400], 'nl': [150, 150], 'ns_scores': [0.9, 0.8],
              'as': [0, 250, 550], 'al': [100, 150, 450], 'as_scores': [0.5, 0.6, 0.7]}
    set_legacy_apply_tags(read, result, with_scores=True)
    assert read.has_tag('nq') and list(read.get_tag('as')) == [0, 250, 550]
    assert not read.has_tag('aq')


@pytest.mark.parametrize('fixture', [FT13, FT062], ids=['ft0.13', 'ft0.6.2'])
def test_apply_scores_warns_and_writes_no_msp_score(tmp_path, fixture):
    proc = _run('fiberhmm.cli.apply', '-i', fixture, '-o', tmp_path, '--enzyme', 'hia5',
                '--seq', 'pacbio', '--force-seq', '--scores', '-c', '1')
    assert proc.returncode == 0, proc.stderr
    assert 'no longer writes an MSP score' in proc.stderr
    out = tmp_path / (fixture.name.replace('.bam', '') + '_footprints.bam')
    with pysam.AlignmentFile(str(out)) as bam:
        reads = list(bam)
    assert any(r.has_tag('as') for r in reads) and any(r.has_tag('nq') for r in reads)
    for read in reads:
        assert not any(read.has_tag(t) for t in STALE), read.query_name
    assert _contract_lines(out) == [ANNOTATION_CONTRACT_V2]


# --------------------------------------------------------------------------- #
#  MSP rewriters: no stale fibertools annotations, non-call groups kept         #
# --------------------------------------------------------------------------- #

def test_strip_stale_call_tags_keeps_m5c_groups_that_ft_moved_into_ma():
    """ft fire 0.13 moves every MA group into Ma, DddA m5C groups included;
    call/apply keep those groups (in MA) and drop the rest of Ma/Aq/An."""
    from fiberhmm.inference.read_filters import strip_stale_call_tags
    read = _read(_header())
    read.set_tag('Ma', '1000;nuc.:101-150;msp.:1-100,251-150;fire.Q:251-150;ddda_mcg.:401-20')
    read.set_tag('Aq', array('B', [240]))
    read.set_tag('aq', array('B', [0, 240]))
    strip_stale_call_tags(read)
    assert read.get_tag('MA') == '1000;ddda_mcg.:401-20'
    assert not any(read.has_tag(t) for t in (*STALE, 'AQ'))


def _write_ma(read, msps, *, legacy=True):
    from fiberhmm.inference.tf_recaller import write_ma_tags
    write_ma_tags(read, read.query_length, [], [(100, 150)], msps,
                  nq_for_kept_nucs=[200], also_write_legacy=legacy)


@pytest.mark.parametrize('reverse', [False, True])
def test_write_ma_tags_keeps_a_fire_group_that_matches_the_msps(reverse):
    read = _read(_header(), reverse=reverse)
    # molecular MSPs [0,100) and [250,400); on a reverse read the same MSPs in
    # SEQ frame are [900,1000) and [600,750)
    seq_msps = [(900, 100), (600, 150)] if reverse else [(0, 100), (250, 150)]
    read.set_tag('Ma', '1000;msp.:1-100,251-150;fire.Q:251-150;deam+:11-5')
    read.set_tag('Aq', array('B', [233]))
    read.set_tag('An', 'a,b,f1,d1')
    _write_ma(read, seq_msps)
    groups = _groups(read)
    assert groups['fire'] == ['251-150'] and groups['deam'] == ['11-5']
    assert list(read.get_tag('AQ'))[-1] == 233
    assert read.get_tag('AN').split(',')[-2:] == ['f1', 'd1']
    assert not any(read.has_tag(t) for t in STALE)


def test_write_ma_tags_drops_a_fire_group_whose_msps_changed():
    read = _read(_header())
    read.set_tag('Ma', '1000;msp.:1-100,251-150;fire.Q:251-150;deam+Q:11-5')
    read.set_tag('Aq', array('B', [233, 44]))
    read.set_tag('An', 'a,b,f1,d1')
    _write_ma(read, [(0, 100), (250, 140)])                  # MSP 2 now ends 10 bp earlier
    groups = _groups(read)
    assert 'fire' not in groups and groups['deam'] == ['11-5']
    assert 233 not in list(read.get_tag('AQ')) and list(read.get_tag('AQ'))[-1] == 44
    assert 'f1' not in read.get_tag('AN').split(',')


def test_write_ma_tags_without_legacy_tags_still_drops_legacy_aq():
    read = _read(_header())
    read.set_tag('as', array('I', [0, 250]))
    read.set_tag('al', array('I', [100, 150]))
    read.set_tag('aq', array('B', [0, 245]))
    _write_ma(read, [(0, 100), (250, 150)], legacy=False)
    assert read.has_tag('as') and not read.has_tag('aq')


@pytest.mark.parametrize('tool', ['recall_tfs', 'recall_tfs --no-legacy-tags', 'recall_nucs'])
def test_recall_on_ft013_output_keeps_valid_fire_and_no_stale_tags(tmp_path, tool):
    name, *extra = tool.split()
    out = tmp_path / 'o.bam'
    entry = 'main_recall_nucs' if name == 'recall_nucs' else 'main'
    code = ('import sys; from fiberhmm.cli import recall_tfs as m; '
            f'sys.argv[0] = "fiberhmm-{name.replace("_", "-")}"; sys.exit(m.{entry}())')
    proc = _run(code, '-i', FT13, '-o', out, '--enzyme', 'hia5', '--seq', 'pacbio',
                '--force-seq', '-c', '1', *extra, module=False)
    assert proc.returncode == 0, proc.stderr
    with pysam.AlignmentFile(str(out)) as bam:
        fwd, rev = list(bam)
    for read in (fwd, rev):
        assert not any(read.has_tag(t) for t in STALE), read.query_name
        groups = _groups(read)
        if 'fire' in groups:
            assert set(groups['fire']) <= set(groups['msp'])
    assert 'fire' in _groups(fwd) and 'fire' not in _groups(rev)
    assert _contract_lines(out) == [ANNOTATION_CONTRACT_V2]


def test_recall_on_ft062_output_drops_the_fire_aq_of_replaced_msps(tmp_path):
    out = tmp_path / 'o.bam'
    proc = _run('fiberhmm.cli.recall_tfs', '-i', FT062, '-o', out, '--enzyme', 'hia5',
                '--seq', 'pacbio', '--force-seq', '-c', '1', '--no-legacy-tags')
    assert proc.returncode == 0, proc.stderr
    with pysam.AlignmentFile(str(out)) as bam:
        for read in bam:
            assert not any(read.has_tag(t) for t in STALE), read.query_name


def test_call_on_ft013_output_leaves_no_fibertools_annotations(tmp_path):
    out = tmp_path / 'o.bam'
    code = ('import sys; from fiberhmm.cli import call; '
            'sys.argv[0] = "fiberhmm-call"; sys.exit(call.main())')
    proc = _run(code, '-i', FT13, '-o', out, '--enzyme', 'hia5', '--seq', 'pacbio',
                '--force-seq', '-c', '1', '--no-qc', module=False)
    assert proc.returncode == 0, proc.stderr
    with pysam.AlignmentFile(str(out)) as bam:
        for read in bam:
            assert not any(read.has_tag(t) for t in STALE), read.query_name
            assert 'fire' not in _groups(read)
    assert _contract_lines(out) == [ANNOTATION_CONTRACT_V2]


# --------------------------------------------------------------------------- #
#  pass-through writers remove Ma after copying it into MA                      #
# --------------------------------------------------------------------------- #

def test_tag_m5c_seed_removes_fibertools_ma_after_copying():
    from fiberhmm.daf.m5c import _seed_ma_from_fibertools
    with pysam.AlignmentFile(str(FT13)) as bam:
        read = next(iter(bam))
    ma = read.get_tag('Ma')
    _seed_ma_from_fibertools(read, 'molecular')
    assert read.get_tag('MA') == ma
    assert not any(read.has_tag(t) for t in ('Ma', 'Aq', 'An'))


def test_consensus_export_removes_fibertools_ma_after_copying():
    from fiberhmm.inference.consensus.bam_export import _append_annotations
    with pysam.AlignmentFile(str(FT13)) as bam:
        read = next(iter(bam))
    ma = read.get_tag('Ma')
    rows = [dict(chrom='chrT', interval=[100, 120], layer='fam', token='t1', tq=50, fi=1, op=2)]
    assert _append_annotations(read, rows, 'molecular') == 1
    assert read.get_tag('MA').startswith(ma + ';fam.')
    assert not any(read.has_tag(t) for t in ('Ma', 'Aq', 'An'))


def test_scan_groups_written_from_fibertools_ma_remove_it():
    from fiberhmm.io.scan_ma import replace_scan_groups
    with pysam.AlignmentFile(str(FT13)) as bam:
        read = next(iter(bam))
    ma = read.get_tag('Ma')
    replace_scan_groups(read, {'nucsz': ('Q', [(100, 150, [3], 'n1')])})
    assert read.get_tag('MA').startswith(ma)
    assert not any(read.has_tag(t) for t in ('Ma', 'Aq', 'An'))


# --------------------------------------------------------------------------- #
#  the header line                                                              #
# --------------------------------------------------------------------------- #

def test_contract_line_is_added_once_and_replaces_other_versions():
    header = {'HD': {'VN': '1.6'}, 'CO': ['FIBERHMM-ANNOTATION-CONTRACT:v1', 'keep me']}
    once = append_annotation_contract(header)
    twice = append_annotation_contract(once)
    co = twice.to_dict()['CO']
    assert co.count(ANNOTATION_CONTRACT_V2) == 1 and 'keep me' in co
    assert not any(c == 'FIBERHMM-ANNOTATION-CONTRACT:v1' for c in co)
    assert header_annotation_contract(twice) == 'v2'
    assert header_annotation_contract({'HD': {'VN': '1.6'}}) is None


def test_producer_header_carries_the_contract_once_on_rerun():
    from fiberhmm.cli.provenance import output_header_with_provenance
    record = {'PN': 'fiberhmm-recall-tfs', 'VN': 'test'}
    first = output_header_with_provenance(_header(), record)
    second = output_header_with_provenance(first, record)
    assert second.to_dict()['CO'].count(ANNOTATION_CONTRACT_V2) == 1


# --------------------------------------------------------------------------- #
#  records passed through unchanged: an aq that may be FiberHMM's goes          #
# --------------------------------------------------------------------------- #

def test_aq_is_ambiguous_only_after_an_old_fiberhmm_apply_without_the_contract():
    from fiberhmm.inference.read_filters import header_aq_may_be_fiberhmm
    apply_pg = {'ID': 'fiberhmm-apply', 'PN': 'fiberhmm-apply', 'VN': '3.2.1'}
    ft_pg = {'ID': 'ft.1', 'PN': 'fibertools-rs', 'VN': '0.6.2'}
    assert header_aq_may_be_fiberhmm({'HD': {'VN': '1.6'}, 'PG': [apply_pg, ft_pg]})
    assert not header_aq_may_be_fiberhmm({'HD': {'VN': '1.6'}, 'PG': [ft_pg]})
    assert not header_aq_may_be_fiberhmm({'HD': {'VN': '1.6'}, 'PG': [apply_pg, ft_pg],
                                          'CO': [ANNOTATION_CONTRACT_V2]})


def _with_old_apply_pg(src, dst, *, old_apply=True, m5c=False):
    with pysam.AlignmentFile(str(src)) as bam:
        header = bam.header.to_dict()
        if old_apply:
            header['PG'] = [{'ID': 'fiberhmm-apply', 'PN': 'fiberhmm-apply', 'VN': '3.2.1'},
                            *header['PG']]
        with pysam.AlignmentFile(str(dst), 'wb', header=header) as out:
            for read in bam:
                if m5c:   # a DddA CpG-island group (molecular, no AQ bytes)
                    read.set_tag('MA', str(read.get_tag('MA')) + ';ddda_ucg.:101-50')
                out.write(read)
    pysam.index(str(dst))


@pytest.mark.parametrize('m5c', [False, True], ids=['no-m5c', 'm5c'])
@pytest.mark.parametrize('old_apply', [False, True])
def test_apply_no_msps_keeps_fibertools_aq_but_not_an_ambiguous_one(tmp_path, old_apply, m5c):
    src = FT062
    if old_apply or m5c:
        src = tmp_path / 'in.bam'
        _with_old_apply_pg(FT062, src, old_apply=old_apply, m5c=m5c)
    with pysam.AlignmentFile(str(src)) as bam:
        before = {r.is_reverse: r for r in bam}
    proc = _run('fiberhmm.cli.apply', '-i', src, '-o', tmp_path / 'out', '--enzyme', 'hia5',
                '--seq', 'pacbio', '--force-seq', '--no-msps', '-c', '1')
    assert proc.returncode == 0, proc.stderr
    out = tmp_path / 'out' / (src.name.replace('.bam', '') + '_footprints.bam')
    with pysam.AlignmentFile(str(out)) as bam:
        reads = list(bam)
    assert all(r.has_tag('as') for r in reads)            # --no-msps keeps the input's MSPs
    assert all(r.has_tag('aq') != old_apply for r in reads)
    for read in reads:
        orig = before[read.is_reverse]
        assert list(read.get_tag('as')) == list(orig.get_tag('as'))
        if not old_apply:
            assert list(read.get_tag('aq')) == list(orig.get_tag('aq'))
        if m5c:
            # MA (present for the kept island) carries the retained MSPs,
            # molecular frame like as/al, and nucleosomes only outside them
            groups = _groups(read)
            assert groups['ddda_ucg'] == ['101-50']
            want = [f'{s + 1}-{n}' for s, n in zip(orig.get_tag('as'), orig.get_tag('al'))
                    if n > 0]
            assert sorted(groups['msp']) == sorted(want), read.is_reverse
            msps = [(s, s + n) for s, n in zip(orig.get_tag('as'), orig.get_tag('al'))]
            for item in groups.get('nuc', []):
                if not item:
                    continue
                a, n = (int(v) for v in item.split('-'))
                assert not any(a - 1 < e and s < a - 1 + n for s, e in msps)
