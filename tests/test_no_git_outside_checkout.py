"""An installed FiberHMM never runs git.

git runs only when the package is imported from a source checkout (``.git``
beside the package directory). Elsewhere the commit comes from
``fiberhmm/_build_info.py`` (written when a wheel is built), else it is
omitted. On a Mac without the command-line developer tools, every ``git`` call
pops the "install developer tools" dialog, so an installed wheel must make no
git call at all. This test copies the package to a directory with no ``.git``
(like site-packages), puts a fake ``git`` that records its calls first on
PATH, and exercises every code path that records the running commit.
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

PROBE = r'''
import json, sys
from fiberhmm import identity
from fiberhmm.inference.accessibility.workflow import _version
from fiberhmm.inference.consensus.lattice_recaller.frozen import code_identity
print(json.dumps(dict(commit=identity.fiberhmm_commit(), nfr=_version().get('git'),
                      frozen=code_identity()['git_commit'])))
'''


def _fake_git(bindir, log):
    git = bindir / 'git'
    git.write_text(f'#!/bin/sh\necho "$@" >> "{log}"\nexit 1\n')
    git.chmod(0o755)


def _run(site, bindir, tmp_path):
    env = dict(os.environ, PATH=f'{bindir}{os.pathsep}{os.environ.get("PATH", "")}',
               PYTHONPATH=str(site), PYTHONDONTWRITEBYTECODE='1', HOME=str(tmp_path))
    out = subprocess.run([sys.executable, '-c', PROBE], env=env, cwd=str(tmp_path),
                         capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr[-3000:]
    return json.loads(out.stdout.strip().splitlines()[-1])


@pytest.mark.skipif(os.name == 'nt', reason='shell fake git')
@pytest.mark.parametrize('build_info', [None, 'a' * 40])
def test_installed_package_makes_no_git_calls(tmp_path, build_info):
    site = tmp_path / 'site-packages'
    shutil.copytree(REPO / 'fiberhmm', site / 'fiberhmm',
                    ignore=shutil.ignore_patterns('__pycache__', '_build_info.py'))
    if build_info:
        (site / 'fiberhmm' / '_build_info.py').write_text(f'COMMIT = "{build_info}"\n')
    bindir = tmp_path / 'bin'
    bindir.mkdir()
    log = tmp_path / 'git_calls.log'
    _fake_git(bindir, log)
    got = _run(site, bindir, tmp_path)
    assert not log.exists() or log.read_text() == '', log.read_text()
    assert got == dict(commit=build_info, nfr=build_info, frozen=build_info)


@pytest.mark.skipif(os.name == 'nt', reason='shell fake git')
def test_source_checkout_still_asks_git(tmp_path):
    if not (REPO / '.git').exists():
        pytest.skip('not running from a git checkout')
    bindir = tmp_path / 'bin'
    bindir.mkdir()
    log = tmp_path / 'git_calls.log'
    _fake_git(bindir, log)
    _run(REPO, bindir, tmp_path)
    assert log.exists() and 'rev-parse' in log.read_text()
