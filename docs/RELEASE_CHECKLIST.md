# Release checklist (v3.0.0)

Run from a clean checkout of the release commit.

1. **Version.** `pyproject.toml` `version = "3.0.0"`, and
   `python -c "import fiberhmm; print(fiberhmm.__version__)"` prints `3.0.0`.
   The top `CHANGELOG.md` entry is `## 3.0.0` and the README banner names 3.0.0.
2. **Docs in sync.** `python tools/gen_cli_reference.py --check` exits 0
   (regenerate with `python tools/gen_cli_reference.py` after any help-text
   change).
3. **CI green** on the release commit: the `test` jobs (Linux/macOS,
   Python 3.10 and 3.12) and the `package` wheel-smoke jobs in
   `.github/workflows/ci.yml`. Locally:
   `python -m pytest tests -q -p no:cacheprovider`.
4. **Wheel smoke.** Build and install into a fresh venv with core dependencies
   only, outside the source tree:

   ```bash
   python -m pip install --upgrade build twine
   python -m build                      # sdist + wheel into dist/
   python -m twine check dist/*
   python -m venv /tmp/fh3 && /tmp/fh3/bin/pip install dist/fiberhmm-3.0.0-*.whl
   cd /tmp && for c in call apply recall-tfs recall-nucs qc extract dedup pair \
       merge consensus transfer; do /tmp/fh3/bin/fiberhmm-$c --help >/dev/null \
       || echo "FAIL $c"; done
   ```

   The wheel must contain only the `fiberhmm` package plus its bundled models
   (`fiberhmm/models/*.json`, `legacy/*.json`) and QC references.
5. **Tag.** `git tag -a v3.0.0 -m "FiberHMM 3.0.0"` on the release commit and
   push the tag.
6. **PyPI.** `python -m twine upload dist/fiberhmm-3.0.0*` (TestPyPI first if
   desired: `--repository testpypi`). Confirm `pip install fiberhmm==3.0.0` in a
   clean venv.
7. **GitHub release.** Create the v3.0.0 release from the tag; paste the
   `## 3.0.0` section of `CHANGELOG.md` as the notes, keeping the Nanopore Hia5
   re-run notice at the top. Attach the sdist and wheel.
8. **FiberBrowser.** Release FiberBrowser 3.0.0 with its dependency pinned to
   `fiberhmm[consensus]>=3.0,<4`, after `fiberhmm==3.0.0` is on PyPI, and
   confirm a fresh FiberBrowser install resolves FiberHMM 3.0.0.
