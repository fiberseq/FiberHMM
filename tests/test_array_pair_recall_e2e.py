"""Array-pair tag rule (3.2.1), end to end through fiberhmm-recall-tfs.

A Dorado ``ns:i`` (a scalar: the read's signal sample count) is not a
legacy footprint call: a legacy pair is two integer arrays of equal length,
otherwise it is absent. The reads here are real FiberHMM 3.2.1 + fibertools
output (tests/fixtures/fire_contract) re-called by the real CLI.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pysam
import pytest

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / 'tests' / 'fixtures' / 'fire_contract' / 'fiberhmm3.2.1_call_ft0.6.2_fire.bam'
RECALL = 'import sys; from fiberhmm.cli import recall_tfs as m; sys.argv[0] = "fiberhmm-recall-tfs"; sys.exit(m.main())'


def _env():
    env = dict(os.environ, FIBERHMM_NO_UPDATE_CHECK='1')
    env['PYTHONPATH'] = os.pathsep.join(filter(None, [str(REPO), env.get('PYTHONPATH')]))
    return env


def _rewrite(dst, *, drop=(), scalar_ns=False):
    with pysam.AlignmentFile(str(SRC)) as bam:
        header = bam.header
        with pysam.AlignmentFile(str(dst), 'wb', header=header) as out:
            for r in bam:
                for t in drop:
                    r.set_tag(t, None)
                if scalar_ns:
                    r.set_tag('ns', 12345, value_type='i')
                out.write(r)
    pysam.index(str(dst))
    return dst


def _recall(src, out):
    proc = subprocess.run([sys.executable, '-c', RECALL, '-i', str(src), '-o', str(out), '--enzyme', 'hia5',
                           '--seq', 'pacbio', '-c', '1'], capture_output=True, text=True, env=_env(), cwd=REPO, timeout=600)
    return proc


def _ma(path):
    with pysam.AlignmentFile(str(path)) as bam:
        return {r.query_name: str(r.get_tag('MA')) for r in bam}


def test_dorado_scalar_ns_beside_ma_only_calls_recalls_like_ma_only(tmp_path):
    legacy = ('ns', 'nl', 'as', 'al', 'nq')
    plain = _rewrite(tmp_path / 'ma_only.bam', drop=legacy)
    dorado = _rewrite(tmp_path / 'ma_dorado_ns.bam', drop=legacy, scalar_ns=True)
    for src in (plain, dorado):
        proc = _recall(src, tmp_path / f'rec_{src.name}')
        assert proc.returncode == 0, proc.stderr[-2000:]
    a, b = _ma(tmp_path / 'rec_ma_only.bam'), _ma(tmp_path / 'rec_ma_dorado_ns.bam')
    assert a == b
    assert all('nuc.' in v and 'msp.' in v for v in a.values()), a


def test_scalar_ns_beside_an_nl_array_is_an_absent_pair_in_recall(tmp_path):
    reference = _rewrite(tmp_path / 'msp_pair.bam', drop=('ns', 'nl', 'nq'))
    broken = _rewrite(tmp_path / 'scalar_ns_nl.bam', drop=('nq',), scalar_ns=True)
    for src in (reference, broken):
        proc = _recall(src, tmp_path / f'rec_{src.name}')
        assert proc.returncode == 0, proc.stderr[-2000:]
    assert _ma(tmp_path / 'rec_scalar_ns_nl.bam') == _ma(tmp_path / 'rec_msp_pair.bam')
