# Environment variables

| Variable | Effect |
|---|---|
| `FIBERHMM_NO_UPDATE_CHECK` | Any non-empty value turns off the update reminder. Otherwise each command (except `fiberhmm-consensus` and `fiberhmm-transfer`, which never check) looks up the latest release on PyPI at most once a day (1.5 s timeout, cached in `$XDG_CACHE_HOME/fiberhmm/update_check.json`, default `~/.cache/…`) and prints a one-line reminder on **stderr** while a newer version exists. It never writes to stdout and never fails a run. |
| `FIBERHMM_MP_CONTEXT` | Worker start method: `fork`, `spawn` or `forkserver`. Default: `spawn` on Python ≥ 3.14, `fork` otherwise. Set `spawn` if workers crash with a segmentation fault. |
| `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS`, `OMP_NUM_THREADS` | BLAS threads. Consensus runs are reproducible to the last bit only with single-threaded BLAS; set `OPENBLAS_NUM_THREADS=1` (or the equivalent for your BLAS) before starting. A run with multi-threaded BLAS warns and records the thread state in its manifest. |
| `NUMBA_CACHE_DIR` | Numba's own setting: where compiled kernels are cached. Useful when the installation directory is read-only or synchronized (Dropbox, network drives). Worker processes turn caching off to avoid lock contention. |
| `PYTORCH_MPS_FAST_MATH` | With the `[cuda]` extra on Apple silicon, `1` disables the MPS backend of the staged engine's predictive kernels (fast math breaks their numerical check). |
| `XDG_CACHE_HOME` | Base directory of the update-check cache. |

Internal: `FIBERHMM_DAF_RUN_MASK` and `FIBERHMM_DAF_RUN_POLICY` carry the
`--daf-mask-runs` / `--daf-run-policy` setting to worker processes. The CLI
sets them; do not set them yourself (an inherited value is reported as a
warning).

For a clean test run from a copied or synchronized source tree, use fresh
cache directories:

```bash
OPENBLAS_NUM_THREADS=1 PYTHONPYCACHEPREFIX=/tmp/fiberhmm-pycache \
NUMBA_CACHE_DIR=/tmp/fiberhmm-numba FIBERHMM_NO_UPDATE_CHECK=1 \
python -m pytest tests -q -p no:cacheprovider
```
