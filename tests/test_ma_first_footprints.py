"""Nucleosome/MSP calls come from MA first; legacy ns/nl/as/al only as fallback (3.3.0).

``fiberhmm.io.footprint_calls`` is the rule: a record's MA (else fibertools
Ma) ``nuc``/``msp`` groups are authoritative; legacy pairs are read only when
neither tag has such a group. recall-tfs used to take its v2 input from the
legacy tags, so an MA refreshed without its legacy tags (recall-nucs
``--no-legacy-tags``, other writers) was recalled from stale calls.

The reads are real FiberHMM call output (tests/fixtures/fire_contract).
"""
from __future__ import annotations

import array
import os
import subprocess
import sys
from pathlib import Path

import pysam
import pytest

from fiberhmm.io.footprint_calls import (
    footprint_annotation_tag,
    ma_has_nuc_msp_groups,
    resolve_footprint_tags,
)

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / 'tests' / 'fixtures' / 'fire_contract' / 'fiberhmm3.2.1_call_ft0.6.2_fire.bam'
LEGACY = ('ns', 'nl', 'nq', 'as', 'al', 'aq')


def _bare(reverse=False, length=1000):
    header = pysam.AlignmentHeader.from_dict({
        'HD': {'VN': '1.6'}, 'SQ': [{'SN': 'c', 'LN': 10_000}],
        'CO': ['fiberhmm:coord=molecular']})
    read = pysam.AlignedSegment(header)
    read.query_name = 'r'
    read.flag = 16 if reverse else 0
    read.reference_id = 0
    read.reference_start = 0
    read.query_sequence = 'A' * length
    read.cigarstring = f'{length}M'
    return read


def _set_legacy(read, ns, nl, as_=None, al=None, nq=None):
    read.set_tag('ns', array.array('I', ns))
    read.set_tag('nl', array.array('I', nl))
    if nq is not None:
        read.set_tag('nq', array.array('B', nq))
    if as_ is not None:
        read.set_tag('as', array.array('I', as_))
        read.set_tag('al', array.array('I', al))


def _lists(tags):
    return {k: [int(v) for v in tags[k]] for k in ('ns', 'nl', 'as', 'al', 'nq') if k in tags}


# --------------------------------------------------------------------------
# The rule
# --------------------------------------------------------------------------

def test_ma_nuc_msp_groups_win_over_disagreeing_legacy_tags():
    read = _bare()
    read.set_tag('MA', '1000;nuc.Q:101-150;msp.:301-200')
    read.set_tag('AQ', array.array('B', [77]))
    _set_legacy(read, [500], [140], [10], [20], nq=[9])
    got = resolve_footprint_tags(read, True)
    assert got['source'] == 'MA'
    assert _lists(got) == {'ns': [100], 'nl': [150], 'as': [300], 'al': [200], 'nq': [77]}


def test_legacy_is_the_fallback_only_without_nuc_msp_groups():
    read = _bare()
    _set_legacy(read, [500], [140], [10], [20], nq=[9])
    # An m5C-only (or FIRE-only, or tf-only) MA does not call nucleosomes/MSPs.
    for ma in ('1000;ddda_mcg.:11-5', '1000;tf.:11-5'):
        read.set_tag('MA', ma)
        assert footprint_annotation_tag(read) is None
        got = resolve_footprint_tags(read, True)
        assert got['source'] == 'legacy'
        assert _lists(got) == {'ns': [500], 'nl': [140], 'as': [10], 'al': [20], 'nq': [9]}
    # An empty nuc group is authoritative: no nucleosomes, not the legacy ones.
    read.set_tag('MA', '1000;nuc.:;msp.:301-200')
    got = resolve_footprint_tags(read, True)
    assert got['source'] == 'MA' and _lists(got)['ns'] == []
    assert ma_has_nuc_msp_groups('1000;nuc.:') and not ma_has_nuc_msp_groups('1000;fire.Q:1-5')


def test_fibertools_ma_used_when_fiberhmm_ma_has_only_other_groups():
    read = _bare(reverse=True)
    read.set_tag('MA', '1000;ddda_mcg.:11-5')
    read.set_tag('Ma', '1000;nuc.:101-150;msp.:301-200')
    _set_legacy(read, [500], [140])
    got = resolve_footprint_tags(read, False)   # Ma is molecular: flipped to SEQ
    assert got['source'] == 'Ma'
    assert _lists(got) == {'ns': [750], 'nl': [150], 'as': [500], 'al': [200]}


def test_legacy_nq_reused_only_when_legacy_lists_the_ma_nucleosomes():
    read = _bare()
    read.set_tag('MA', '1000;nuc.:101-150,401-147')
    _set_legacy(read, [400, 100], [147, 150], nq=[5, 6])
    assert _lists(resolve_footprint_tags(read, True))['nq'] == [6, 5]
    _set_legacy(read, [400, 100], [147, 151], nq=[5, 6])   # stale: one nucleosome differs
    assert 'nq' not in resolve_footprint_tags(read, True)


# --------------------------------------------------------------------------
# recall-tfs: live recall_read and the worker payload path
# --------------------------------------------------------------------------

@pytest.fixture(scope='module')
def tables():
    from fiberhmm.core.model_io import load_model_with_metadata
    from fiberhmm.models import get_model_path
    from fiberhmm.inference.tf_recaller import build_llr_tables

    model, _, _ = load_model_with_metadata(get_model_path('hia5', seq='pacbio'))
    return build_llr_tables(model)


def _variants(reverse):
    """(name, read) per fixture read: as written (MA + agreeing legacy), MA
    only, MA beside stale legacy tags, and the stale legacy tags alone."""
    out = []
    with pysam.AlignmentFile(str(SRC)) as bam:
        for index, read in enumerate(bam):
            if reverse:
                read.flag ^= 16
            full = pysam.AlignedSegment.fromstring(read.to_string(), bam.header)
            ma_only = pysam.AlignedSegment.fromstring(read.to_string(), bam.header)
            for tag in LEGACY:
                ma_only.set_tag(tag, None)
            stale = pysam.AlignedSegment.fromstring(ma_only.to_string(), bam.header)
            ns = [int(s) for s in read.get_tag('ns')]
            nl = [int(n) for n in read.get_tag('nl')]
            # Shifted, one nucleosome dropped, MSPs collapsed: not the MA calls.
            _set_legacy(stale, [s + 60 for s in ns[1:]] or [100], nl[1:] or [150],
                        [0], [min(200, read.query_length)], nq=[3] * max(1, len(ns) - 1))
            legacy_only = pysam.AlignedSegment.fromstring(stale.to_string(), bam.header)
            for tag in ('MA', 'AQ', 'AN'):
                legacy_only.set_tag(tag, None)
            out.append((index, {'full': full, 'ma_only': ma_only, 'stale': stale,
                                'legacy_only': legacy_only}))
    return out


@pytest.mark.parametrize('reverse', [False, True])
def test_recall_read_live_reads_ma_first(tables, reverse):
    from fiberhmm.inference.tf_recaller import recall_read

    hit, miss = tables
    for _index, reads in _variants(reverse):
        got = {name: recall_read(read, hit, miss, 'pacbio-fiber', 3, 5.0, 3, 90)
               for name, read in reads.items()}
        assert got['ma_only'] == got['full'] == got['stale']
        assert got['full'][1]            # nucleosomes were read
        assert got['legacy_only'] != got['full']   # the stale tags really differ


@pytest.mark.parametrize('reverse', [False, True])
def test_recall_worker_payload_reads_ma_first(tables, reverse):
    from fiberhmm.cli import recall_tfs

    hit, miss = tables
    recall_tfs._worker_init(hit, miss, 'pacbio-fiber', 3, 5.0, 3, 90)
    for _index, reads in _variants(reverse):
        got = {name: recall_tfs._process_payload_record(
                   recall_tfs._make_payload(read, 'pacbio-fiber', True))[0]
               for name, read in reads.items()}
        assert got['ma_only'] == got['full'] == got['stale']
        assert got['full'][1]
        assert got['legacy_only'] != got['full']


def test_payload_with_undecided_legacy_frame_proceeds_on_ma_calls(monkeypatch):
    from fiberhmm.cli import recall_tfs

    # Module-level frame state is per run; earlier in-process recall-tfs tests can leave it set.
    monkeypatch.setitem(recall_tfs._UNDECIDED_FRAME, 'assumed', False)
    monkeypatch.setitem(recall_tfs._UNDECIDED_FRAME, 'reason', '')
    read = _bare(reverse=True)
    read.set_tag('MA', '1000;nuc.:101-150;msp.:301-200')
    _set_legacy(read, [500], [140])
    payload = recall_tfs._make_payload(read, None, None)   # legacy frame undecided
    assert payload['_input_molecular_frame'] is False
    assert _lists(payload['tags']) == {'ns': [750], 'nl': [150], 'as': [500], 'al': [200]}


# --------------------------------------------------------------------------
# call -> recall-nucs -> recall-tfs through the real CLIs
# --------------------------------------------------------------------------

def _env(tmp_path):
    env = dict(os.environ, FIBERHMM_NO_UPDATE_CHECK='1', HOME=str(tmp_path))
    env['PYTHONPATH'] = os.pathsep.join(filter(None, [str(REPO), env.get('PYTHONPATH')]))
    return env


def _cli(tmp_path, entry, *args):
    code = (f'import sys; from fiberhmm.cli import recall_tfs as m; '
            f'sys.argv[0] = "fiberhmm-x"; sys.exit(m.{entry}())')
    proc = subprocess.run([sys.executable, '-c', code, *args, '--enzyme', 'hia5', '--seq', 'pacbio',
                           '-c', '1'], capture_output=True, text=True, env=_env(tmp_path), cwd=REPO,
                          timeout=600)
    assert proc.returncode == 0, proc.stderr[-2000:]


def _records(path):
    with pysam.AlignmentFile(str(path)) as bam:
        return {r.query_name: r for r in bam}


def _ma_nucs(read):
    tags = resolve_footprint_tags(read, True)
    return sorted(zip(tags['ns'], tags['nl']))


def test_recall_tfs_after_recall_nucs_uses_the_nucleosomes_recall_nucs_wrote(tmp_path):
    nucs = tmp_path / 'nucs.bam'
    _cli(tmp_path, 'main_recall_nucs', '-i', str(SRC), '-o', str(nucs))
    # Put unmistakably stale legacy tags beside the MA recall-nucs wrote (as a
    # 3.2 recall-nucs --no-legacy-tags left call's), then recall TFs from that
    # BAM and from an MA-only copy.
    stale, ma_only = tmp_path / 'stale.bam', tmp_path / 'ma_only.bam'
    with pysam.AlignmentFile(str(nucs)) as bam:
        with pysam.AlignmentFile(str(stale), 'wb', header=bam.header) as a, \
                pysam.AlignmentFile(str(ma_only), 'wb', header=bam.header) as b:
            for r in bam:
                _set_legacy(r, [5], [150], [0], [5])
                a.write(r)
                for tag in LEGACY:
                    r.set_tag(tag, None)
                b.write(r)
    outputs = {}
    for name, src in (('stale', stale), ('ma_only', ma_only)):
        out = tmp_path / f'tfs_{name}.bam'
        _cli(tmp_path, 'main', '-i', str(src), '-o', str(out))
        outputs[name] = _records(out)
    written = _records(nucs)
    assert {k: str(r.get_tag('MA')) for k, r in outputs['stale'].items()} == \
        {k: str(r.get_tag('MA')) for k, r in outputs['ma_only'].items()}
    for name, read in outputs['stale'].items():
        kept = set(_ma_nucs(read))
        before = set(_ma_nucs(written[name]))
        # recall-tfs keeps recall-nucs' nucleosomes (it only demotes short ones under TF calls).
        assert kept <= before and {n for n in before if n[1] >= 90} <= kept
        assert (5, 150) not in kept
