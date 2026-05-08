# FiberHMM Refactor Audit

Audit branch: `refactor/audit-regression-baseline`

Date: 2026-05-07

## Baseline

- Branch was created from a clean `main`.
- `python -m pytest`: 297 passed in 68.91s.
- `python -m pytest -m "not benchmark"`: 277 passed, 20 deselected in 6.89s.
- `python -m compileall -q fiberhmm tests`: passed.
- `python -m ruff check fiberhmm tests`: initially not runnable because `ruff` was not installed; `ruff` is now installed from the existing `dev` optional dependency and part of the current gate.
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
- Extracted the active `samtools merge -b` last-resort fallback into the shared BAM output helpers with list-file cleanup tests.
- Extracted `samtools index`/`samtools sort` command wrappers and added fake-command tests for direct indexing, unsorted BAM sorting, missing `samtools`, and pysam sort fallback.
- Extracted the full region BAM concatenation fallback ladder into `fiberhmm/inference/bam_output.py`; apply and fused region paths now share empty-output, single-copy, `samtools cat`, pysam fallback, partial-output cleanup, output-dir probing, and final `samtools merge` behavior.
- Added typed region-parallel worker contracts in `fiberhmm/inference/region_types.py`; apply BAM, fused BAM, and BED region workers now use named work-item/result containers while preserving legacy tuple coercion at private boundaries.
- Added shared region result aggregation containers for BAM and BED workers; apply BAM, fused BAM, and BED region-parallel loops now share count/skip/temp-path accumulation behavior.
- Removed unused private BAM merge helpers from `fiberhmm/inference/parallel.py`; active merge fallback coverage now lives with the shared BAM output helper tests.
- Added higher-level region worker failure cleanup tests covering apply BAM, BED, and fused BAM temp-directory cleanup after executor/future failures.
- Extracted fused HMM apply, TF recall scan, and nucleosome/TF unification stage helpers into `fiberhmm/inference/fused_stages.py`; fused streaming and region-parallel workers now share those stage boundaries.
- Avoided per-read Python list copies in the fused TF recall stage and removed a redundant score-byte cast in shared tag writing.
- Added a single-pass numba DAF context encoder with vectorized-fallback equivalence tests for CT and GA strands.
- Added an explicit DAF encoder benchmark that compares the single-pass fast path to the vectorized fallback oracle.
- Added a single-pass numba m6A context encoder for PacBio and Nanopore modes that emits final methylated/unmethylated HMM observation codes directly, with vectorized-fallback equivalence tests and benchmarks.
- Installed `ruff` and cleared the configured lint gate across `fiberhmm` and `tests` with import sorting, whitespace cleanup, unused-import cleanup, protected package re-exports, and scoped style fixes.
- Added structured worker chunk results so streaming apply and fused apply+recall workers still pass failed reads through unchanged but now report per-read worker failure counts to the drain/final summary.
- Reduced Viterbi memory traffic by keeping only rolling state scores plus backpointers instead of full per-state score arrays, with a dedicated warm prediction benchmark.
- Avoided per-call observation copies for integer HMM input arrays and warmed worker numba Viterbi signatures with int32 observations to match encoder output.
- Split `encode_from_query_sequence` into mode-specific PacBio m6A, Nanopore m6A, and DAF helpers while preserving the public API and keeping vectorized fallbacks as numba equivalence oracles.
- Added explicit read-only HMM log-probability freezing for inference workers, while keeping default public `predict()` behavior dynamic for direct in-place model mutations.
- Added model-load safeguards so legacy pickled models always return with dynamic log recomputation unless an inference worker explicitly freezes them again.
- Reused the shared footprint/MSP extraction helper inside `predict_footprints_and_msps`, removing duplicate state-to-region post-processing logic.
- Aligned the I/O-vs-compute benchmark with the frozen inference model path used by apply workers.
- Added a numba-backed footprint run scanner that avoids per-read padded state-array and diff allocations while preserving a numpy fallback when numba is unavailable.
- Added direct footprint-run oracle coverage for empty, all-accessible, all-footprint, edge, and dtype-varied state arrays.

## Current Verification

- `python -m pytest tests/test_call_pipeline.py::test_daf_raw_md_and_reference_streaming_match_iupac_output`: 1 passed in 2.72s.
- `python -m pytest tests/test_call_pipeline.py`: 4 passed in 4.54s.
- `python -m pytest tests/test_call_cli.py`: 7 passed in 1.95s.
- `python -m pytest tests/test_bam_output.py`: 18 passed in 0.90s.
- `python -m pytest tests/test_region_types.py`: 6 passed in 0.67s.
- `python -m pytest tests/test_bam_output.py tests/test_call_cli.py`: 9 passed in 1.61s.
- `python -m pytest tests/test_bam_output.py tests/test_mode_equivalence.py tests/test_call_pipeline.py`: 26 passed in 4.60s.
- `python -m pytest tests/test_region_types.py tests/test_mode_equivalence.py tests/test_call_pipeline.py`: 14 passed in 4.74s.
- `python -m pytest tests/test_region_cleanup.py`: 3 passed in 0.68s.
- `python -m pytest tests/test_region_cleanup.py tests/test_region_types.py tests/test_mode_equivalence.py tests/test_call_pipeline.py`: 17 passed in 4.72s.
- `python -m pytest tests/test_fused_stages.py tests/test_call_pipeline.py`: 6 passed in 4.45s.
- `python -m pytest tests/test_fused_stages.py tests/test_call_pipeline.py tests/test_mode_equivalence.py tests/test_region_cleanup.py`: 13 passed in 6.58s.
- `python -m pytest tests/test_tagging.py tests/test_fused_stages.py tests/test_call_pipeline.py tests/test_tf_recaller.py`: 29 passed in 4.57s.
- DAF encoder local timing check on synthetic CT sequence: vectorized fallback 0.0318s, single-pass fast path 0.0061s, 5.24x speedup.
- `python -m pytest tests/test_bam_reader.py::TestEncodingConsistency::test_daf_numba_fast_path_matches_vectorized tests/test_daf_iupac.py`: 23 passed in 1.01s.
- `python -m pytest tests/test_bam_reader.py tests/test_daf_iupac.py tests/test_call_pipeline.py tests/test_call_cli.py`: 61 passed in 5.07s.
- `python -m pytest -m benchmark tests/benchmarks/bench_encoding.py`: 1 passed in 0.68s.
- `python -m pytest tests/test_call_pipeline.py tests/test_call_cli.py tests/test_daf_iupac.py tests/test_extract_block_scores.py`: 62 passed in 5.67s.
- `python -m pytest tests/test_package_consistency.py`: 19 passed in 1.50s.
- `python -m ruff check fiberhmm tests`: passed.
- `python -m pytest tests/test_inference_parallel.py tests/test_streaming_pipeline.py tests/test_call_pipeline.py tests/test_fused_stages.py`: 69 passed in 11.32s.
- m6A encoder local timing on 2,000 synthetic PacBio reads: encode step improved from 0.3961s (5,049 reads/s) to 0.2744s (7,289 reads/s), 1.44x faster.
- `python -m pytest tests/test_bam_reader.py::TestEncodingConsistency`: 7 passed in 1.22s.
- `python -m pytest -s -m benchmark tests/benchmarks/bench_encoding.py`: 4 passed in 0.92s; m6A single-pass speedups vs vectorized fallback were 9.86x PacBio, 6.35x Nanopore forward, 11.69x Nanopore reverse.
- `python -m pytest tests/test_bam_reader.py tests/test_daf_iupac.py tests/test_call_pipeline.py tests/test_call_cli.py tests/test_streaming_pipeline.py`: 80 passed in 11.16s.
- Warm Viterbi local timing on 2,000 synthetic 5 kb reads: 0.1350s, 14,811 reads/s.
- `python -m pytest -s -m benchmark tests/benchmarks/bench_hmm.py`: 1 passed in 0.94s; warm Viterbi benchmark reported 12,990 reads/s for 5 kb observations.
- `python -m pytest tests/test_hmm.py tests/test_inference_engine.py tests/test_bam_reader.py tests/test_call_pipeline.py tests/test_mode_equivalence.py`: 85 passed in 5.42s.
- `python -m pytest tests/test_hmm.py tests/test_inference_engine.py tests/test_streaming_pipeline.py tests/test_call_pipeline.py`: 65 passed in 8.19s.
- `python -m pytest -s -m benchmark tests/benchmarks/bench_hmm.py`: 1 passed in 0.80s; warm Viterbi benchmark reported 12,947 reads/s for 5 kb observations.
- `python -m pytest`: 335 passed, 25 deselected in 11.24s.
- `python -m compileall -q fiberhmm tests`: passed.
- `python -m pytest -m benchmark tests/benchmarks`: 25 passed in 56.22s.
- `python -m pytest tests/test_bam_reader.py tests/test_daf_iupac.py tests/test_call_pipeline.py tests/test_call_cli.py tests/test_streaming_pipeline.py`: 80 passed in 10.55s.
- `python -m pytest`: 335 passed, 25 deselected in 11.41s.
- `python -m pytest -s -m benchmark tests/benchmarks/bench_encoding.py`: 4 passed in 0.77s; m6A single-pass speedups vs vectorized fallback were 9.62x PacBio, 6.29x Nanopore forward, 11.28x Nanopore reverse, and DAF speedup was 5.34x.
- `python -m compileall -q fiberhmm tests`: passed.
- `python -m pytest -m benchmark tests/benchmarks`: 25 passed in 55.25s.
- `python -m pytest tests/test_hmm.py tests/test_model_io.py tests/test_inference_engine.py tests/test_streaming_pipeline.py tests/test_call_pipeline.py tests/test_fused_stages.py tests/test_package_consistency.py`: 100 passed in 8.56s.
- `python -m pytest -s -m benchmark tests/benchmarks/bench_hmm.py`: 2 passed in 0.94s; dynamic repeated Viterbi prediction was 0.0160s, frozen read-only prediction was 0.0074s, 2.16x faster for 200 x 5 kb observations.
- `python -m pytest -s -m benchmark tests/benchmarks/bench_hmm.py tests/benchmarks/bench_throughput.py`: 8 passed in 12.36s; streaming throughput was 5,955 reads/s (1-core), 7,550 reads/s (2-core), 9,063 reads/s (4-core), region-parallel throughput was 7,834 reads/s (2-core) and 10,250 reads/s (4-core), legacy 1-core was 4,773 reads/s.
- `python -m ruff check fiberhmm tests`: passed.
- `python -m compileall -q fiberhmm tests`: passed.
- `python -m pytest`: 339 passed, 26 deselected in 11.29s.
- `python -m pytest -m benchmark tests/benchmarks`: 26 passed in 52.75s.
- `python -m pytest tests/test_inference_engine.py tests/test_fused_stages.py tests/test_call_pipeline.py tests/test_streaming_pipeline.py tests/test_mode_equivalence.py`: 43 passed in 9.88s.
- `python -m pytest -s -m benchmark tests/benchmarks/bench_io_vs_compute.py tests/benchmarks/bench_throughput.py`: 8 passed in 12.40s; frozen compute-only timing was 0.73s for 5,000 reads (6,810 reads/s), streaming throughput was 6,774 reads/s (1-core), 10,132 reads/s (2-core), 12,417 reads/s (4-core), region-parallel throughput was 9,908 reads/s (2-core) and 13,507 reads/s (4-core), legacy 1-core was 5,499 reads/s.
- `python -m ruff check fiberhmm tests`: passed.
- `python -m compileall -q fiberhmm tests`: passed.
- `python -m pytest`: 339 passed, 26 deselected in 10.97s.
- `python -m pytest -m benchmark tests/benchmarks`: 26 passed in 52.64s.
- `python -m ruff check fiberhmm/inference/engine.py tests/test_inference_engine.py`: passed.
- `python -m pytest tests/test_inference_engine.py tests/test_fused_stages.py tests/test_call_pipeline.py tests/test_streaming_pipeline.py tests/test_mode_equivalence.py`: 43 passed in 8.92s.
- `python -m pytest -s -m benchmark tests/benchmarks/bench_io_vs_compute.py tests/benchmarks/bench_throughput.py`: 8 passed in 12.31s; frozen compute-only timing was 0.71s for 5,000 reads (6,995 reads/s), streaming throughput was 6,974 reads/s (1-core), 9,522 reads/s (2-core), 12,517 reads/s (4-core), region-parallel throughput was 9,785 reads/s (2-core) and 13,694 reads/s (4-core), legacy 1-core was 5,722 reads/s.
- `python -m ruff check fiberhmm tests`: passed.
- `python -m compileall -q fiberhmm tests`: passed.
- `python -m pytest`: 339 passed, 26 deselected in 11.15s.
- `python -m pytest -m benchmark tests/benchmarks`: 26 passed in 51.90s.
- `python -m pytest tests/test_inference_engine.py`: 22 passed in 1.48s.
- `python -m ruff check fiberhmm tests`: passed.
- `python -m compileall -q fiberhmm tests`: passed.
- `python -m pytest`: 344 passed, 26 deselected in 11.04s.

## Current Shape

Largest tracked Python files:

- `fiberhmm/inference/parallel.py`: 2588 lines.
- `fiberhmm/cli/extract_tags.py`: 1389 lines.
- `fiberhmm/core/bam_reader.py`: 1340 lines.
- `fiberhmm/core/hmm.py`: 1123 lines.
- `fiberhmm/cli/train.py`: 1024 lines.
- `fiberhmm/cli/utils.py`: 894 lines.
- `fiberhmm/cli/export_posteriors.py`: 816 lines.

Ignored local build artifacts exist (`build/`, `fiberhmm.egg-info/`) but are not tracked.

## Refactor Priorities

1. `fiberhmm/inference/parallel.py` is the main refactor target.

   It mixes multiprocessing context selection, worker initialization, region partitioning, BAM concatenation, streaming pipelines, legacy chunk mode, fused apply/recall, posterior export, tag writing, progress reporting, and skip accounting. This makes performance work risky because small behavior changes can affect output order, pass-through reads, tag semantics, or process lifetime.

2. Continue shrinking `fiberhmm/inference/parallel.py`.

   The first shared tagging/unification slice, streaming read-filter slice, active BAM output helper extractions, region worker contracts, aggregation helpers, higher-level worker cleanup tests, and fused stage-boundary extraction are complete, but the module still mixes process lifecycle, posterior export, and orchestration. The next work should shift toward measured speed/stability improvements while continuing to keep `parallel.py` shrinking.

3. Add remaining `fiberhmm-call` characterization tests.

   The fused streaming, fused region-parallel, legacy tags, `MA`/`AQ`, stdout/stderr behavior, DAF input-source sniffing, DAF raw MD/reference streaming output, model resolution, `--with-scores` nucleosome quality behavior, and higher-level worker failure cleanup behavior now have direct tests.

4. Mode-specific encoding has explicit helper boundaries.

   `encode_from_query_sequence` now dispatches to PacBio, Nanopore, and DAF private helpers. PacBio, Nanopore, and DAF have single-pass numba encoders that emit final HMM observation codes, with vectorized fallbacks retained as equivalence oracles. Future performance work can stay mode-local.

5. Per-read exception handling should stay visible.

   Streaming worker functions intentionally catch broad exceptions and return `None` so production runs continue. They now return structured chunk results with per-read failure counts, and the streaming drain paths report those counts while preserving pass-through behavior. Continue extending the same pattern if region worker internals add per-read recovery paths.

6. CLI coverage remains weak.

   Most CLI modules have little or no direct coverage. `fiberhmm-call` now has direct coverage for stdout/stderr behavior, DAF input-source handling, model resolution, fused streaming/region-parallel output, and `--with-scores`, but other CLI entry points still need focused tests.

7. Benchmarks are now explicit.

   Benchmark files are still collected by pytest, but the default marker expression deselects them. Keep running `python -m pytest -m benchmark tests/benchmarks` before and after performance-sensitive work.

8. External-tool wrappers need hardening.

   BAM sort/index and bigBed conversion call `samtools`, `sort`, and `bedToBigBed`. Known shell-based sorting has been removed, and `rg "shell=True" fiberhmm tests` currently returns no matches. BAM-list cleanup, merge fallback cleanup, partial-output cleanup, output-dir probing, sort/index fallbacks, and higher-level worker-failure cleanup now have coverage.

9. Temporary output paths are unique in the main region-parallel paths.

   Apply BAM, BED, and fused region-parallel modes now use per-run temp directories under the output directory, and higher-level worker-failure tests verify cleanup. Future output helpers should keep this property.

10. Public compatibility surfaces must stay stable.

   Top-level wrapper scripts and legacy model loading are intentional compatibility surfaces. They should not be removed during the first refactor; they should be isolated behind thin adapters.

## Regression Gates

Before moving code:

- Keep `python -m pytest` green.
- Keep `python -m ruff check fiberhmm tests` green.
- Use `python -m pytest` as the fast non-benchmark correctness gate.
- Use `python -m pytest -m benchmark tests/benchmarks` as the explicit performance gate.
- Add `fiberhmm-call` characterization tests:
  - fused streaming output has expected `MA`/`AQ` plus legacy tags; done.
  - fused region-parallel matches fused streaming on synthetic BAMs; done.
  - `--with-scores` behavior writes aligned `nq` for kept nucleosomes; done.
  - stdout mode keeps BAM bytes on stdout and logs on stderr; done.
  - DAF IUPAC, raw MD, and raw `--reference` output paths are covered in fused streaming.
- Add golden-tag comparison helpers that hash/read `ns`, `nl`, `as`, `al`, `nq`, `aq`, `MA`, and `AQ` per read.
- Add tests for temporary directory uniqueness and external-tool fallback behavior; direct external-tool fallback coverage and worker-failure cleanup coverage are in place.

## Refactor Plan

Phase 1: safety and characterization.

- Split benchmark execution from correctness execution without deleting benchmarks.
- Add missing `fiberhmm-call` tests and score/DAF characterization.
- Add debug-visible per-read failure counters; done for streaming apply and fused apply+recall workers.

Phase 2: mechanical modularization.

- Extract result/tag writing into `fiberhmm/inference/tagging.py`.
- Extract shared skip/filter logic into a small read-filter module.
- Extract BAM merge/sort/index helpers into one output module.
- Introduce typed payload/result containers for apply, fused, and recall paths.
- Extract explicit fused HMM apply, TF recall, and label-writing stage boundaries.
- Split mode-specific read encoding behind private strategy helpers.

Phase 3: pipeline split.

- Split `parallel.py` into streaming, region, workers, and orchestration modules while preserving `process_bam_for_footprints` as the public compatibility wrapper.
- Keep old import paths re-exporting during the transition.

Phase 4: performance work.

- Keep DAF and m6A single-pass context encoding compared against the vectorized fallback byte-for-byte.
- Keep HMM state-path changes benchmarked and covered by mode-equivalence tests.
- Use explicit read-only HMM log freezing only in inference paths so public mutable-model semantics remain intact.
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
