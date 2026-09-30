"""Build hook: record the git commit a wheel is built from.

All package metadata lives in pyproject.toml. This file only extends
``build_py`` and ``sdist`` so a wheel or sdist built from a git checkout (and a
wheel built from such an sdist, as ``python -m build`` does) carries
``fiberhmm/_build_info.py`` (``COMMIT = "<sha>"``, ``+dirty`` when files under
``fiberhmm/`` differ from it). ``fiberhmm.identity.fiberhmm_commit`` records it
in every BAM's ``FIBERHMM-CHEMISTRY`` declaration. Builds without git
metadata (e.g. from an sdist) write nothing, and the commit is then omitted.
"""
import os
import subprocess

from setuptools import setup
from setuptools.command.build_py import build_py
from setuptools.command.sdist import sdist

_ROOT = os.path.dirname(os.path.abspath(__file__))


def _git(*args):
    try:
        result = subprocess.run(["git", "-C", _ROOT, *args], capture_output=True,
                                text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def _source_commit():
    if not os.path.exists(os.path.join(_ROOT, ".git")):
        return None
    commit = (_git("rev-parse", "HEAD") or "").lower()
    if len(commit) != 40 or any(c not in "0123456789abcdef" for c in commit):
        return None
    status = _git("status", "--porcelain", "--untracked-files=normal", "--", "fiberhmm")
    return commit + ("+dirty" if status else "")


def _write_build_info(package_dir, commit):
    target = os.path.join(package_dir, "fiberhmm", "_build_info.py")
    os.makedirs(os.path.dirname(target), exist_ok=True)
    if os.path.exists(target):
        os.remove(target)  # may be a hard link into the source tree (sdist)
    with open(target, "w") as handle:
        handle.write('"""Written by setup.py at build time; do not edit."""\n')
        handle.write(f"COMMIT = {commit!r}\n")


class build_py_with_commit(build_py):
    def run(self):
        super().run()
        commit = _source_commit()
        if commit and not self.dry_run:
            _write_build_info(self.build_lib, commit)
        # Without git metadata (a wheel built from an sdist) the sdist's own
        # _build_info.py, if any, is copied like any other module.


class sdist_with_commit(sdist):
    def make_release_tree(self, base_dir, files):
        super().make_release_tree(base_dir, files)
        commit = _source_commit()
        if commit and not self.dry_run:
            _write_build_info(base_dir, commit)


setup(cmdclass={"build_py": build_py_with_commit, "sdist": sdist_with_commit})
