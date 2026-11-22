"""MAPQ default 0 (3.2): a low-MAPQ read is called by default, and only an
explicit --min-mapq leaves it uncalled. Runs the real fiberhmm-call on real
FiberHMM + fibertools output (tests/fixtures/fire_contract) with MAPQ set to 3.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pysam
import pytest

from fiberhmm import defaults

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / 'tests' / 'fixtures' / 'fire_contract' / 'fiberhmm3.2.1_call_ft0.6.2_fire.bam'
CALL_TAGS = ('MA', 'ns', 'nl', 'as', 'al')


@pytest.fixture(scope='module')
def low_mapq_bam(tmp_path_factory):
    d = tmp_path_factory.mktemp('mapq')
    path = d / 'low_mapq.bam'
    with pysam.AlignmentFile(str(SRC)) as bam:
        header = bam.header.to_dict()
        reads = list(bam)
    with pysam.AlignmentFile(str(path), 'wb', header=header) as out:
        for r in reads:
            r.mapping_quality = 3
            out.write(r)
    pysam.index(str(path))
    return path


def _call(src, out, *extra):
    env = dict(os.environ, FIBERHMM_NO_UPDATE_CHECK='1')
    env['PYTHONPATH'] = os.pathsep.join(filter(None, [str(REPO), env.get('PYTHONPATH')]))
    code = 'import sys; from fiberhmm.cli import call; sys.argv[0] = "fiberhmm-call"; sys.exit(call.main())'
    return subprocess.run([sys.executable, '-c', code, '-i', str(src), '-o', str(out), '--enzyme', 'hia5',
                           '--seq', 'pacbio', '--force-seq', '-c', '1', '--no-qc', *extra],
                          capture_output=True, text=True, env=env, cwd=REPO, timeout=600)


def test_default_min_mapq_is_zero():
    assert defaults.MIN_MAPQ == 0


def test_low_mapq_reads_are_called_by_default_and_left_uncalled_with_an_explicit_threshold(low_mapq_bam, tmp_path):
    default_out = tmp_path / 'default.bam'
    proc = _call(low_mapq_bam, default_out)
    assert proc.returncode == 0, proc.stderr[-2000:]
    with pysam.AlignmentFile(str(default_out)) as bam:
        reads = list(bam)
    assert len(reads) == 2
    for r in reads:
        assert r.mapping_quality == 3
        assert all(r.has_tag(t) for t in CALL_TAGS), (r.query_name, [t for t, _ in r.get_tags()])

    strict_out = tmp_path / 'mapq20.bam'
    proc = _call(low_mapq_bam, strict_out, '--min-mapq', '20')
    assert proc.returncode == 0, proc.stderr[-2000:]
    with pysam.AlignmentFile(str(strict_out)) as bam:
        reads = list(bam)
    assert len(reads) == 2, 'a read below --min-mapq is kept in the output, uncalled'
    for r in reads:
        assert not any(r.has_tag(t) for t in CALL_TAGS), r.query_name
        assert r.has_tag('MM') and r.has_tag('ML')
