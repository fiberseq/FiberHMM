"""fiberhmm-merge and annotation contract v2 (3.3): the merged BAM declares the
contract once, and a passed-through (unmerged) record loses an ``aq`` that may
be an older FiberHMM's MSP posterior, while a fibertools FIRE ``aq`` stays.

Runs the real ``fiberhmm-merge`` on real fibertools output (tests/fixtures/fire_contract).
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pysam
import pytest

from fiberhmm.io.bam_header import ANNOTATION_CONTRACT_V2

REPO = Path(__file__).resolve().parents[1]
FT062 = REPO / 'tests' / 'fixtures' / 'fire_contract' / 'fiberhmm3.2.1_call_ft0.6.2_fire.bam'


def _merge(src, out, *extra):
    env = dict(os.environ, FIBERHMM_NO_UPDATE_CHECK='1')
    env['PYTHONPATH'] = os.pathsep.join(filter(None, [str(REPO), env.get('PYTHONPATH')]))
    return subprocess.run([sys.executable, '-m', 'fiberhmm.cli.merge', '-i', str(src), '-o', str(out), *extra],
                          capture_output=True, text=True, env=env, cwd=REPO, timeout=300)


def _with_header(src, dst, *, old_apply=False, contract=None):
    with pysam.AlignmentFile(str(src)) as bam:
        header = bam.header.to_dict()
        if old_apply:
            header['PG'] = [{'ID': 'fiberhmm-apply', 'PN': 'fiberhmm-apply', 'VN': '3.2.1'}, *header.get('PG', [])]
        if contract is not None:
            header['CO'] = [*header.get('CO', []), contract]
        with pysam.AlignmentFile(str(dst), 'wb', header=header) as out:
            for read in bam:
                out.write(read)
    pysam.index(str(dst))


def _contract_lines(path):
    with pysam.AlignmentFile(str(path)) as bam:
        return [c for c in bam.header.to_dict().get('CO', []) if str(c).startswith('FIBERHMM-ANNOTATION-CONTRACT:')]


@pytest.mark.parametrize('old_apply', [False, True], ids=['fibertools-aq', 'ambiguous-aq'])
def test_merge_passthrough_declares_contract_and_drops_only_an_ambiguous_aq(tmp_path, old_apply):
    src = tmp_path / 'in.bam'
    # The ambiguous case is an old fiberhmm-apply @PG with no contract line; the other carries an older contract
    # version, which the merge must replace with v2.
    _with_header(FT062, src, old_apply=old_apply, contract=None if old_apply else 'FIBERHMM-ANNOTATION-CONTRACT:v1')
    with pysam.AlignmentFile(str(src)) as bam:
        before = {r.query_name: r.has_tag('aq') for r in bam}
    assert any(before.values()), 'fixture has no aq'
    out = tmp_path / 'merged.bam'
    proc = _merge(src, out)
    assert proc.returncode == 0, proc.stderr
    assert 'reads passed through (unmerged): 2' in proc.stderr
    # exactly one contract line, the v2 one (an older version is replaced)
    assert _contract_lines(out) == [ANNOTATION_CONTRACT_V2]
    with pysam.AlignmentFile(str(out)) as bam:
        reads = {r.query_name: r for r in bam}
    assert set(reads) == set(before)
    for name, r in reads.items():
        if old_apply:
            assert not r.has_tag('aq'), f'{name}: an aq after an old fiberhmm-apply may be its MSP posterior'
        else:
            assert r.has_tag('aq') == before[name], f'{name}: a fibertools FIRE aq must stay'
        assert r.has_tag('as'), name          # the MSPs themselves pass through


def test_merge_pairs_only_writes_header_contract_with_no_records(tmp_path):
    src = tmp_path / 'in.bam'
    _with_header(FT062, src)
    out = tmp_path / 'merged.bam'
    proc = _merge(src, out, '--pairs-only')
    assert proc.returncode == 0, proc.stderr
    assert _contract_lines(out) == [ANNOTATION_CONTRACT_V2]
    with pysam.AlignmentFile(str(out)) as bam:
        assert not list(bam)
