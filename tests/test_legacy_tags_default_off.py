"""Writers emit no legacy ns/nl/nq, as/al by default (3.3.0); --legacy-tags restores them.

MA/AQ carry every call, and every FiberHMM reader takes nucleosomes and MSPs
from MA first (fiberhmm.io.footprint_calls), so a chain without legacy tags
must give the same calls as one with them. The input is real FiberHMM call
output (tests/fixtures/fire_contract). The default call output itself is
checked against the --legacy-tags run in tests/test_chimera_default_golden.py.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pysam

from fiberhmm import defaults as D

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / 'tests' / 'fixtures' / 'fire_contract' / 'fiberhmm3.2.1_call_ft0.6.2_fire.bam'
LEGACY = ('ns', 'nl', 'nq', 'as', 'al', 'aq')


def _env(tmp_path):
    env = dict(os.environ, FIBERHMM_NO_UPDATE_CHECK='1', HOME=str(tmp_path))
    env['PYTHONPATH'] = os.pathsep.join(filter(None, [str(REPO), env.get('PYTHONPATH')]))
    return env


def _run(tmp_path, module, entry, *args):
    code = (f'import sys; from fiberhmm.cli import {module} as m; '
            f'sys.argv[0] = "fiberhmm-x"; sys.exit(m.{entry}())')
    proc = subprocess.run([sys.executable, '-c', code, *args], capture_output=True, text=True,
                          env=_env(tmp_path), cwd=REPO, timeout=600)
    assert proc.returncode == 0, proc.stderr[-2000:]
    return proc


def _recall(tmp_path, entry, src, out, *extra):
    return _run(tmp_path, 'recall_tfs', entry, '-i', str(src), '-o', str(out),
                '--enzyme', 'hia5', '--seq', 'pacbio', '-c', '1', *extra)


def _tags(path):
    with pysam.AlignmentFile(str(path)) as bam:
        return {r.query_name: dict(r.get_tags()) for r in bam}


def _content(path):
    import gzip
    data = path.read_bytes()
    if data[:2] == b'\x1f\x8b':
        data = gzip.decompress(data)
    return data.decode(errors='replace')


def test_default_is_off():
    assert D.WRITE_LEGACY_TAGS is False


def test_recall_chain_without_legacy_tags_gives_the_same_ma(tmp_path):
    outs = {}
    for name, extra in (('default', ()), ('legacy', ('--legacy-tags',))):
        nucs, tfs = tmp_path / f'{name}.nucs.bam', tmp_path / f'{name}.tfs.bam'
        _recall(tmp_path, 'main_recall_nucs', SRC, nucs, *extra)
        _recall(tmp_path, 'main', nucs, tfs, *extra)
        outs[name] = (_tags(nucs), _tags(tfs))
    for stage in (0, 1):
        default, legacy = outs['default'][stage], outs['legacy'][stage]
        assert {k: v['MA'] for k, v in default.items()} == {k: v['MA'] for k, v in legacy.items()}
        assert {k: list(v['AQ']) for k, v in default.items()} == \
            {k: list(v['AQ']) for k, v in legacy.items()}
        # The input carries legacy tags: the default run removes them, --legacy-tags rewrites them.
        assert not any(tag in v for v in default.values() for tag in LEGACY)
        assert all('ns' in v and 'as' in v for v in legacy.values())


def test_deprecated_no_legacy_tags_is_accepted(tmp_path):
    out = tmp_path / 'tfs.bam'
    proc = _recall(tmp_path, 'main', SRC, out, '--no-legacy-tags')
    assert 'deprecated' in proc.stderr
    assert not any(tag in v for v in _tags(out).values() for tag in LEGACY)


def test_extract_reads_ma_only_output_like_legacy_output(tmp_path):
    ma_only = tmp_path / 'ma_only.bam'
    with pysam.AlignmentFile(str(SRC)) as bam:
        with pysam.AlignmentFile(str(ma_only), 'wb', header=bam.header) as out:
            for r in bam:
                for tag in LEGACY:
                    r.set_tag(tag, None)
                out.write(r)
    pysam.index(str(ma_only))
    beds = {}
    for name, src in (('legacy', SRC), ('ma_only', ma_only)):
        outdir = tmp_path / name
        outdir.mkdir()
        _run(tmp_path, 'extract_tags', 'main', '-i', str(src), '-o', str(outdir),
             '--nucleosome', '--msp', '--bed-only')
        beds[name] = sorted((f.name.replace(src.stem, 'X'), _content(f))
                            for f in outdir.rglob('*') if f.is_file())
    assert beds['legacy'] and beds['legacy'] == beds['ma_only']
    assert any(text.strip() for _name, text in beds['legacy'])
