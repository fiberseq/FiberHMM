"""fibertools ``ft fire`` reads FiberHMM 3.3's default (MA-only) output.

FiberHMM 3.3 writes no legacy ``ns/nl/as/al`` by default. The standard
pipeline is ``fiberhmm-call | ft fire``, so this runs the real fibertools
binary on a call of the synthetic demo data, with and without
``--legacy-tags``, and checks that FIRE elements are found on the MSPs and
that the two inputs give the same FIRE calls.

Needs fibertools-rs >= 0.13 (``ft`` on PATH, or ``FIBERHMM_TEST_FT`` naming
the binary); skips cleanly otherwise. CI installs a pinned release binary.
fibertools <= 0.12 reads only the legacy ``as/al`` tags (use
``--legacy-tags``) and is not covered here.
"""
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pysam
import pytest

REPO = Path(__file__).resolve().parents[1]
LEGACY = ('ns', 'nl', 'nq', 'as', 'al', 'aq')


def _ft_binary():
    path = os.environ.get('FIBERHMM_TEST_FT') or shutil.which('ft')
    if not path:
        return None
    try:
        out = subprocess.run([path, '--version'], capture_output=True, text=True, timeout=60).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    match = re.search(r'(\d+)\.(\d+)\.(\d+)', out)
    if not match or (int(match.group(1)), int(match.group(2))) < (0, 13):
        return None
    return path


FT = _ft_binary()
# CI sets FIBERHMM_REQUIRE_FT=1 so a missing or too-old ft fails instead of skipping.
REQUIRED = os.environ.get('FIBERHMM_REQUIRE_FT') == '1'
pytestmark = [pytest.mark.fibertools,
              pytest.mark.skipif(FT is None and not REQUIRED, reason='fibertools-rs >= 0.13 (ft) not installed')]


def test_ft_is_available_when_required():
    assert FT is not None, 'FIBERHMM_REQUIRE_FT=1 but no fibertools-rs >= 0.13 ft was found'


def _intervals(body):
    out = set()
    for item in body.split(','):
        if item:
            start, length = item.split('-')
            out.add((int(start), int(length)))
    return out


def _groups(record):
    """{group head: interval set} of the record's MA-style tag (ft writes Ma)."""
    tag = 'Ma' if record.has_tag('Ma') else 'MA' if record.has_tag('MA') else None
    if tag is None:
        return {}
    groups = {}
    for group in str(record.get_tag(tag)).split(';')[1:]:
        head, _, body = group.partition(':')
        groups[head] = _intervals(body)
    return groups


def _fire_by_read(path):
    out = {}
    with pysam.AlignmentFile(str(path), check_sq=False) as bam:
        for record in bam:
            groups = _groups(record)
            fire = next((v for k, v in groups.items() if k.startswith('fire')), set())
            msp = next((v for k, v in groups.items() if k.startswith('msp')), set())
            qual = list(record.get_tag('Aq')) if record.has_tag('Aq') else []
            out[record.query_name] = (frozenset(fire), frozenset(msp), tuple(qual))
    return out


@pytest.fixture(scope='module')
def calls(tmp_path_factory):
    work = tmp_path_factory.mktemp('ft_fire')
    env = dict(os.environ, HOME=str(work), PYTHONPATH=str(REPO), FIBERHMM_NO_UPDATE_CHECK='1')
    subprocess.run([sys.executable, str(REPO / 'docs' / 'examples' / 'make_demo_data.py'), str(work / 'demo')],
                   check=True, capture_output=True, env=env)
    out = {}
    for label, extra in (('default', []), ('legacy', ['--legacy-tags'])):
        called = work / f'{label}.bam'
        subprocess.run([sys.executable, '-m', 'fiberhmm.cli.call', '-i', str(work / 'demo' / 'hia5_pacbio.bam'),
                        '-o', str(called), '--enzyme', 'hia5', '--seq', 'pacbio', '-c', '1', '--no-qc', *extra],
                       check=True, capture_output=True, env=env)
        fired = work / f'{label}.fire.bam'
        proc = subprocess.run([FT, 'fire', '-t', '1', str(called), str(fired)], capture_output=True, text=True)
        assert proc.returncode == 0, proc.stderr[-2000:]
        out[label] = (called, fired)
    return out


def test_default_call_writes_ma_only(calls):
    called, _ = calls['default']
    with pysam.AlignmentFile(str(called), check_sq=False) as bam:
        records = list(bam)
    assert records and all(r.has_tag('MA') for r in records)
    assert not any(r.has_tag(t) for r in records for t in LEGACY)
    assert sum(1 for r in records if 'msp' in str(r.get_tag('MA'))) > len(records) // 2


def test_ft_fire_finds_fire_elements_on_the_msps_of_ma_only_output(calls):
    fire = _fire_by_read(calls['default'][1])
    with_fire = [name for name, (elements, _msp, _q) in fire.items() if elements]
    assert len(with_fire) > len(fire) // 2, (len(with_fire), len(fire))
    for name in with_fire:
        elements, msps, _q = fire[name]
        assert elements <= msps, name


def test_ft_fire_gives_the_same_calls_with_and_without_legacy_tags(calls):
    assert _fire_by_read(calls['default'][1]) == _fire_by_read(calls['legacy'][1])
