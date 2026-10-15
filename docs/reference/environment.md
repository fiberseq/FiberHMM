# Environment variables

| Variable | Effect |
|---|---|
| `FIBERHMM_NO_UPDATE_CHECK` | Any non-empty value turns off the update reminder. Otherwise each command (except `fiberhmm-consensus`, `fiberhmm-transfer` and `fiberhmm-nfr`, which never check) looks up the latest release on PyPI at most once a day (1.5 s timeout, cached in `$XDG_CACHE_HOME/fiberhmm/update_check.json`, default `~/.cache/…`) and prints a one-line reminder on **stderr** while a newer version exists. It never writes to stdout and never fails a run. |
| `FIBERHMM_MP_CONTEXT` | Worker start method: `fork`, `spawn` or `forkserver`. Default: `spawn` on Python ≥ 3.14, `fork` otherwise. Set `spawn` if workers crash with a segmentation fault. |
| `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS`, `OMP_NUM_THREADS` | BLAS threads. Consensus runs are reproducible to the last bit only with single-threaded BLAS; set `OPENBLAS_NUM_THREADS=1` (or the equivalent for your BLAS) before starting. A run with multi-threaded BLAS warns and records the thread state in its manifest. |
| `NUMBA_CACHE_DIR` | Numba's own setting: where compiled kernels are cached. Useful when the installation directory is read-only or synchronized (Dropbox, network drives). Worker processes inherit it unchanged, so set it before starting a run to keep caches out of the installation; unset, numba caches beside the installed sources. |
| `PYTORCH_MPS_FAST_MATH` | With the `[cuda]` extra on Apple silicon, `1` disables the MPS backend of the staged engine's predictive kernels (fast math breaks their numerical check). |
| `XDG_CACHE_HOME` | Base directory of the update-check cache. |
| `FIBERHMM_DORADO` | `fiberhmm-pipeline`: the dorado program (or its install folder) when `--dorado` is not given. Searched before `PATH` and the usual install locations. |
| `FIBERHMM_DORADO_MODELS_DIR` | `fiberhmm-pipeline`: where dorado keeps downloaded models when `--dorado-models-dir` is not given. Default `~/.fiberhmm/dorado_models`. |
| `FIBERHMM_MINIMAP2_INDEX_DIR` | `fiberhmm-pipeline`: where minimap2 reference indexes are cached. Default `~/.fiberhmm/minimap2_index`. |
| `FIBERHMM_DEBUG` | Any non-empty value shows the full traceback for input errors (a missing path, a file that is not a readable BAM) instead of the one-line `error:` message with exit status 2. |

Internal: `FIBERHMM_DAF_RUN_MASK` and `FIBERHMM_DAF_RUN_POLICY` carry the
`--daf-mask-runs` / `--daf-run-policy` setting to worker processes, and
`FIBERHMM_DAF_MASK_UNALIGNED` carries `--daf-mask-unaligned` /
`--no-daf-mask-unaligned`. The CLI sets them; do not set them yourself (an
inherited run-mask value is reported as a warning).

For a clean test run from a copied or synchronized source tree, use fresh
cache directories:

```bash
OPENBLAS_NUM_THREADS=1 PYTHONPYCACHEPREFIX=/tmp/fiberhmm-pycache \
NUMBA_CACHE_DIR=/tmp/fiberhmm-numba FIBERHMM_NO_UPDATE_CHECK=1 \
python -m pytest tests -q -p no:cacheprovider
```
