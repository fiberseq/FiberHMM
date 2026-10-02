# Contributing

Bug reports and pull requests are welcome at
[github.com/fiberseq/FiberHMM](https://github.com/fiberseq/FiberHMM/issues).
For a bug, include the command, the full error output (with tracebacks), the
FiberHMM version, and if possible a small BAM that reproduces it; the
[demo data generator](../getting-started/quickstart.md#get-the-demo-data)
is a good starting point for a synthetic reproduction.

## Development setup

```bash
git clone https://github.com/fiberseq/FiberHMM.git
cd FiberHMM
pip install -e ".[all,dev]"
```

## Tests

```bash
OPENBLAS_NUM_THREADS=1 FIBERHMM_NO_UPDATE_CHECK=1 python -m pytest tests -q -p no:cacheprovider
```

Single-threaded BLAS keeps the consensus reference tests exact. Tests that
need private data sets skip when the data are absent. Benchmarks are
excluded by default (`-m benchmark tests/benchmarks` runs them). The
`fiberhmm-pipeline` end-to-end tests need `minimap2` and some tests need
`samtools` on `PATH`; they skip without them. CI runs the suite on Linux and
macOS with Python 3.10, 3.12 and 3.13 (with `minimap2` and `samtools`
installed and the full git history) and smoke-tests a wheel install
(`.github/workflows/ci.yml`). Code style: `black` and `ruff`, line
length 100.

Every fix should come with a test that fails before it and passes after.

## Documentation

The documentation is this site, built with
[MkDocs](https://www.mkdocs.org/) and
[Material for MkDocs](https://squidfunk.github.io/mkdocs-material/) from
`docs/` and `mkdocs.yml`:

```bash
pip install "mkdocs==1.6.1" "mkdocs-material==9.7.7"
mkdocs serve                  # live preview at http://127.0.0.1:8000
mkdocs build --strict         # what CI runs; warnings fail the build
```

The site is deployed to GitHub Pages by `.github/workflows/docs.yml` on
pushes to `main`; pull requests only build it.

The [command-line reference](../reference/cli.md) is generated from the
commands' argument parsers. After changing any option or help text, run:

```bash
python tools/gen_cli_reference.py            # rewrite docs/reference/cli.md
python tools/gen_cli_reference.py --check    # exit 1 if it is out of date
```

`tests/test_cli_reference.py` fails when the page and the parsers disagree.
Examples in the docs use the synthetic data from
`docs/examples/make_demo_data.py`; please check new examples against it.

## Release checklist

Run from a clean checkout of the release commit.

1. **Version.** `pyproject.toml` has the new version, and
   `python -c "import fiberhmm; print(fiberhmm.__version__)"` prints it. The
   top `CHANGELOG.md` entry names it, and `CITATION.cff` has the same
   `version` (`tests/test_packaging.py` checks it).
2. **Docs in sync.** `python tools/gen_cli_reference.py --check` exits 0 and
   `mkdocs build --strict` succeeds.
   **Advisories.** Every change in this release that alters results has an
   entry in `fiberhmm/advisories.json` (read by
   [`fiberhmm-check`](../reference/advisories.md)); then
   `python tools/build_advisory_index.py` refreshes the table digests and
   the commit table up to the release commit.
3. **CI green** on the release commit (tests and wheel smoke on Linux/macOS,
   Python 3.10, 3.12 and 3.13).
4. **Wheel smoke.** Build and install into a fresh venv with core
   dependencies only, outside the source tree:

    ```bash
    python -m pip install --upgrade build twine
    python -m build                      # sdist + wheel into dist/
    python -m twine check dist/*
    python -m venv /tmp/fh && /tmp/fh/bin/pip install dist/fiberhmm-*.whl
    cd /tmp && for c in call apply recall-tfs recall-nucs qc extract dedup pair \
        merge consensus transfer; do /tmp/fh/bin/fiberhmm-$c --help >/dev/null \
        || echo "FAIL $c"; done
    ```

    The wheel must contain only the `fiberhmm` package with its bundled models
    (`fiberhmm/models/*.json`, `legacy/*.json`), QC references,
    `fiberhmm/advisories.json` and `fiberhmm/_build_info.py` (the release
    commit, written by `setup.py`).
5. **Tag** the release commit (`git tag -a vX.Y.Z -m "FiberHMM X.Y.Z"`) and
   push the tag; the docs workflow publishes the site.
6. **PyPI.** `python -m twine upload dist/fiberhmm-X.Y.Z*` (TestPyPI first
   with `--repository testpypi` if desired); confirm
   `pip install fiberhmm==X.Y.Z` in a clean venv.
7. **GitHub release** from the tag, with the version's `CHANGELOG.md` section
   as the notes (for 3.0.0, keep the Nanopore Hia5 re-run notice at the top)
   and the sdist and wheel attached.
8. **FiberBrowser.** Release the matching FiberBrowser with its dependency
   pinned to `fiberhmm[consensus]>=X.0,<X+1` after FiberHMM is on PyPI, and
   confirm a fresh FiberBrowser install resolves it.
