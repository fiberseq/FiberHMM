# FiberHMM Refactor Audit

Audit branch: `refactor/audit-regression-baseline`

Date: 2026-05-07

## Baseline

- Branch was created from a clean `main`.
- `python -m pytest`: 297 passed in 68.91s.
- `python -m pytest -m "not benchmark"`: 277 passed, 20 deselected in 6.89s.
- `python -m compileall -q fiberhmm tests`: passed.
- `python -m ruff check fiberhmm tests`: not runnable in this environment because `ruff` is not installed.
- Coverage with pytest-cov is not currently a reliable full-suite command in this sandbox. A coverage run produced 268 passes and 29 failures, mostly in multiprocessing paths with `PermissionError: os.sysconf("SC_SEM_NSEMS_MAX")`, plus benchmark-memory failures under coverage overhead. The normal non-coverage suite passed.

## Implemented In This Branch

- Split correctness tests from benchmark tests in `pyproject.toml`. `python -m pytest` now defaults to `-m 'not benchmark'`, while benchmarks are still available with `python -m pytest -m benchmark tests/benchmarks`.
- Added fused `fiberhmm-call` characterization coverage for streaming output tags, `--with-scores` nucleosome quality tags, and one-region streaming/region-parallel tag equivalence.
- Extracted shared inference tagging helpers into `fiberhmm/inference/tagging.py`.
- Reused shared helpers for legacy apply tag writing in both legacy chunk paths.
- Reused shared helpers for fused nucleosome/TF unification in both streaming and region-parallel fused call paths.
- Fixed fused `--with-scores` output so kept nucleosome calls carry `nq` values through the unification step.
- Changed region-parallel temporary directories to unique per-run directories under the output directory to avoid concurrent-run collisions.
- Preserved the legacy `write_msps=False` behavior for the final partial chunk in the legacy apply pipeline.
- Added a CLI stdout characterization test proving `fiberhmm-call -o -` keeps logs on stderr and produces a readable BAM stream on stdout.
- Added DAF CLI input-source characterization tests for IUPAC-encoded BAMs, explicit `--reference` fallback, and actionable failure when no deamination source is present.
- Added fused DAF streaming output characterization proving raw-MD BAMs and raw BAMs with `--reference` produce the same output tags as matched IUPAC-encoded BAMs.
- Added `fiberhmm-call` model-resolution tests, including DddA's separate nuc/apply and TF/recall bundled models.
- Extracted shared streaming read skip/filter policy into `fiberhmm/inference/read_filters.py` and reused it in the legacy and fused streaming paths.
- Removed shell-based BED sorting from bigBed conversion helpers; sorting now uses list-form subprocess calls and has fake-command fallback tests.
- Extracted active region BAM `samtools cat -b` list-file handling into `fiberhmm/inference/bam_output.py`; both apply and fused region paths now clean list files on success and failure.
- Removed unused private BAM merge helpers from `fiberhmm/inference/parallel.py`; active region merge paths remain unchanged.

## Current Verification

- `python -m pytest tests/test_call_pipeline.py::test_daf_raw_md_and_reference_streaming_match_iupac_output`: 1 passed in 2.72s.
- `python -m pytest tests/test_call_pipeline.py`: 4 passed in 4.54s.
- `python -m pytest tests/test_call_cli.py`: 7 passed in 1.95s.
- `python -m pytest tests/test_bam_output.py`: 5 passed in 0.80s.
- `python -m pytest tests/test_bam_output.py tests/test_call_cli.py`: 9 passed in 1.61s.
- `python -m pytest tests/test_bam_output.py tests/test_mode_equivalence.py tests/test_call_pipeline.py`: 13 passed in 4.72s.
- `python -m pytest tests/test_call_pipeline.py tests/test_call_cli.py tests/test_daf_iupac.py tests/test_extract_block_scores.py`: 62 passed in 5.67s.
- `python -m pytest`: 300 passed, 20 deselected in 11.13s.
- `python -m compileall -q fiberhmm tests`: passed.
- `python -m pytest -m benchmark tests/benchmarks`: 20 passed in 61.24s.
- `python -m ruff check fiberhmm tests`: not runnable in this environment because `ruff` is not installed.

## Current Shape

Largest tracked Python files:

- `fiberhmm/inference/parallel.py`: 2686 lines.
- `fiberhmm/cli/extract_tags.py`: 1389 lines.
- `fiberhmm/core/bam_reader.py`: 1246 lines.
- `fiberhmm/core/hmm.py`: 1089 lines.
- `fiberhmm/cli/train.py`: 1024 lines.
- `fiberhmm/cli/utils.py`: 894 lines.
- `fiberhmm/cli/export_posteriors.py`: 811 lines.

Ignored local build artifacts exist (`build/`, `fiberhmm.egg-info/`) but are not tracked.

## Refactor Priorities

1. `fiberhmm/inference/parallel.py` is the main refactor target.

   It mixes multiprocessing context selection, worker initialization, region partitioning, BAM concatenation, streaming pipelines, legacy chunk mode, fused apply/recall, posterior export, tag writing, progress reporting, and skip accounting. This makes performance work risky because small behavior changes can affect output order, pass-through reads, tag semantics, or process lifetime.

2. Continue shrinking `fiberhmm/inference/parallel.py`.

   The first shared tagging/unification slice, streaming read-filter slice, and active `samtools cat` helper extraction are complete, but the module still mixes process lifecycle, output merge/indexing fallback behavior, posterior export, and orchestration. The next low-risk slices are additional BAM output helpers and typed payload/result containers.

3. Add remaining `fiberhmm-call` characterization tests.

   The fused streaming, fused region-parallel, legacy tags, `MA`/`AQ`, stdout/stderr behavior, DAF input-source sniffing, DAF raw MD/reference streaming output, model resolution, and `--with-scores` nucleosome quality behavior now have direct tests. Remaining gaps are broader fake-failure tests for external tools.

4. Mode-specific encoding should be split.

   `encode_from_query_sequence` handles PacBio, Nanopore, and DAF in one long function. The PacBio/Nanopore path has a numba single-pass encoder; the DAF path still builds context index matrices for target positions. Splitting into mode strategy functions would make performance changes safer and make a DAF numba encoder feasible.

5. Per-read exception handling hides regressions.

   Worker functions intentionally catch broad exceptions and return `None` so production runs continue. That is reasonable operational behavior, but tests need counters or debug diagnostics that prove exceptions are not silently increasing after refactors.

6. CLI coverage remains weak.

   Most CLI modules have little or no direct coverage. The package has good parser-helper tests and core integration tests, but there are no end-to-end `fiberhmm-call` CLI tests for fused streaming, fused region-parallel, stdout/stderr behavior, DAF `--reference`, or `--with-scores`.

7. Benchmarks are now explicit.

   Benchmark files are still collected by pytest, but the default marker expression deselects them. Keep running `python -m pytest -m benchmark tests/benchmarks` before and after performance-sensitive work.

8. External-tool wrappers need hardening.

   BAM sort/index and bigBed conversion call `samtools`, `sort`, and `bedToBigBed`. Known shell-based sorting has been removed, and `rg "shell=True" fiberhmm tests` currently returns no matches. BAM-list cleanup is covered for `samtools cat`; remaining work is broader fake-failure coverage around sort/index and merge fallback paths.

9. Temporary output paths are unique in the main region-parallel paths.

   Apply BAM, BED, and fused region-parallel modes now use per-run temp directories under the output directory. Future output helpers should keep this property and add direct tests around cleanup after worker failures.

10. Public compatibility surfaces must stay stable.

   Top-level wrapper scripts and legacy model loading are intentional compatibility surfaces. They should not be removed during the first refactor; they should be isolated behind thin adapters.

## Regression Gates

Before moving code:

- Keep `python -m pytest` green.
- Use `python -m pytest` as the fast non-benchmark correctness gate.
- Use `python -m pytest -m benchmark tests/benchmarks` as the explicit performance gate.
- Add `fiberhmm-call` characterization tests:
  - fused streaming output has expected `MA`/`AQ` plus legacy tags; done.
  - fused region-parallel matches fused streaming on synthetic BAMs; done.
  - `--with-scores` behavior writes aligned `nq` for kept nucleosomes; done.
  - stdout mode keeps BAM bytes on stdout and logs on stderr; done.
  - DAF IUPAC, raw MD, and raw `--reference` output paths are covered in fused streaming.
- Add golden-tag comparison helpers that hash/read `ns`, `nl`, `as`, `al`, `nq`, `aq`, `MA`, and `AQ` per read.
- Add tests for temporary directory uniqueness and external-tool fallback behavior.

## Refactor Plan

Phase 1: safety and characterization.

- Split benchmark execution from correctness execution without deleting benchmarks.
- Add missing `fiberhmm-call` tests and score/DAF characterization.
- Add debug-visible per-read failure counters.

Phase 2: mechanical modularization.

- Extract result/tag writing into `fiberhmm/inference/tagging.py`.
- Extract shared skip/filter logic into a small read-filter module.
- Extract BAM merge/sort/index helpers into one output module.
- Introduce typed payload/result containers for apply, fused, and recall paths.

Phase 3: pipeline split.

- Split `parallel.py` into streaming, region, workers, and orchestration modules while preserving `process_bam_for_footprints` as the public compatibility wrapper.
- Keep old import paths re-exporting during the transition.

Phase 4: performance work.

- Implement a DAF single-pass context encoder and compare against current vectorized output byte-for-byte.
- Reduce repeated list/array conversions in fused unification and tag writing.
- Benchmark chunk size, inflight depth, multiprocessing context, and I/O threads with the existing benchmark suite.

Phase 5: CLI and packaging cleanup.

- Centralize bundled model resolution and CLI defaults.
- Keep top-level compatibility wrappers, but ensure tests cover the console-script entry points.
- Document legacy pickle loading risk and keep it isolated.

## Invariants To Preserve

- MM/ML reverse-read query positions must match `pysam.modified_bases`.
- Base code order must remain A=0, C=1, T=2, G=3.
- `ns`/`nl` and `as`/`al` must be unsigned 32-bit BAM arrays.
- Stale `nq`/`aq` must be cleared when refreshed tag lengths differ.
- Streaming mode must preserve input read order.
- Region-parallel mode must preserve coordinate sort without a sort pass when assumptions hold.
- Skipped/unprocessable reads must pass through unchanged.
- DddA must continue using separate nuc and TF models.
- `fiberhmm-call` must keep logs off stdout when writing BAM to stdout.
