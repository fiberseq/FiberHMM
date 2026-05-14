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
- Reused the shared legacy apply tag writer in the region BAM worker and changed unsigned 32-bit BAM tag-array construction to avoid materializing Python integer lists.
- Removed the extra state-path scan in public `predict_footprints`; the centralized run extractor now handles the no-footprint case directly.
- Closed the fused streaming DAF reference FASTA handle after reference-backed raw DAF processing and added a regression test for the close call.
- Made `fiberhmm-recall-tfs` close input and output BAM handles in a `finally` block, with CLI-level coverage for processing failures.
- Extracted genome region planning helpers into `fiberhmm/inference/region_planning.py`, keeping compatibility re-exports from `parallel.py` and adding direct region splitting/filtering coverage.
- Extracted order-preserving streaming drain helpers into `fiberhmm/inference/streaming_drain.py`; streaming apply and fused apply+recall now share drain behavior outside `parallel.py`.
- Extracted streaming worker initializers and chunk worker entry points into `fiberhmm/inference/streaming_workers.py`, while keeping compatibility imports in `parallel.py`.
- Extracted region-parallel worker initializers and BAM/BED/fused region worker entry points into `fiberhmm/inference/region_workers.py`, while keeping compatibility imports in `parallel.py`.
- Extracted multiprocessing start-method selection into `fiberhmm/inference/mp_context.py`, while keeping compatibility imports in `parallel.py`.
- Extracted streaming apply and fused apply/recall pipeline coordinators into `fiberhmm/inference/streaming_pipeline.py`, while keeping compatibility imports in `parallel.py`.
- Extracted apply BAM, BED, and fused apply/recall region-parallel pipeline coordinators into `fiberhmm/inference/region_pipeline.py`, while keeping compatibility imports in `parallel.py`.
- Extracted the legacy chunked apply pipeline into `fiberhmm/inference/legacy_pipeline.py`, while keeping compatibility imports in `parallel.py`.
- Closed inline posterior writers from apply processing `finally` blocks in both streaming and legacy paths, with failure-path regression coverage.
- Closed fused DAF streaming reference FASTA handles from the processing `finally` block, including failure-path coverage.
- Extracted region-parallel posterior TSV formatting, output-path resolution, and merge ordering into `fiberhmm/posteriors/region_tsv.py` with direct coverage.
- Avoided per-read Python list materialization for ML BAM tags in legacy BAM reading and TF recall extraction; the shared manual MM/ML parser now receives raw tag containers directly.
- Made TSV posterior export close its writer from a `finally` block when region processing fails, with CLI helper coverage.
- Made posterior TSV conversion and concatenation use context-managed plain/gzip handles, with failure-path coverage for conversion and concatenation cleanup.
- Made the HDF5 posterior writer close its underlying file handle even when finalization fails, with direct failure-path coverage.
- Made extract-tag region workers close already-opened temporary BED handles if a later per-type output open fails, with worker-level failure coverage.
- Made QC stats BAM sampling use context-managed BAM handles across both passes, with failure-path coverage for second-pass cleanup.
- Made DAF BAM encoding close input BAM and reference FASTA handles when pre-output MD validation fails, with direct lifecycle coverage.
- Made tagged-BAM BED extraction close its input BAM if output BED opening fails, with direct shared-helper coverage.
- Reduced DAF reference-fallback FASTA calls by fetching each read's reference span once instead of fetching one base per aligned position, with direct helper coverage.
- Made score-database creation and append helpers close SQLite connections on record parsing failures, with shared-helper coverage.
- Reused the shared read-filter policy in the legacy chunked apply loop and removed its unused write counter, preserving mode-equivalence coverage while adding the same `None` read-length guard used by streaming paths.
- Reused the shared read-filter policy for owned reads in apply and fused region BAM workers while preserving pre-ownership pass-through behavior for unmapped and secondary/supplementary reads.
- Deferred dispatcher-side model loading for streaming, region-parallel, and legacy multi-core model-path runs so worker initializers own model loading; legacy single-core dispatch still loads the model directly before processing.
- Made `fiberhmm-consensus-tfs` build each read's query-to-reference position map at most once while extracting TF calls, with direct coverage for repeated annotations, quality-filtered skips, insertion-only calls, and malformed MA tags.
- Made MA/AQ parsing consume quality bytes directly from indexable BAM tag containers instead of materializing the whole `AQ` array up front, preserving short-array behavior with direct parser coverage.
- Made `fiberhmm-recall-tfs` count per-read recall failures and pass failed records through unchanged in both single-thread and worker-chunk paths, with direct coverage for failure accounting and pass-through writes.
- Kept legacy recall payload tag arrays compact through `_make_payload` and let `recall_read` consume array-backed tag sequences directly, avoiding redundant Python list materialization in the TF recall path.
- Removed extra full-list materialization from `fiberhmm-extract` TF `AQ` parsing and MM/ML modified-position extraction, with regression tests that exercise indexable `AQ` containers and parser arrays without `.tolist()`.

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
- Streaming tuning check on 12,000 synthetic reads with 4 cores: `chunk_size=500` remained the best default among 250/500/1000/2000 at fixed `max_inflight=8`; raising in-flight depth from 8 to 12 only improved 9,973 reads/s to 10,119 reads/s, so no default change was made.
- `python -m pytest tests/test_tagging.py tests/test_call_pipeline.py tests/test_mode_equivalence.py tests/test_streaming_pipeline.py`: 27 passed in 7.83s.
- `python -m pytest -s -m benchmark tests/benchmarks/bench_io_vs_compute.py tests/benchmarks/bench_throughput.py`: 8 passed in 11.69s; frozen compute-only timing was 0.70s for 5,000 reads (7,135 reads/s), streaming throughput was 7,181 reads/s (1-core), 10,535 reads/s (2-core), 12,947 reads/s (4-core), region-parallel throughput was 10,348 reads/s (2-core) and 14,825 reads/s (4-core), legacy 1-core was 5,798 reads/s.
- `python -m ruff check fiberhmm tests`: passed.
- `python -m compileall -q fiberhmm tests`: passed.
- `python -m pytest`: 346 passed, 26 deselected in 11.37s.
- `python -m pytest -m benchmark tests/benchmarks`: 26 passed in 51.78s.
- `python -m pytest tests/test_inference_engine.py tests/test_fused_stages.py tests/test_call_pipeline.py tests/test_streaming_pipeline.py tests/test_mode_equivalence.py`: 48 passed in 8.88s.
- `python -m ruff check fiberhmm tests`: passed.
- `python -m compileall -q fiberhmm tests`: passed.
- `python -m pytest`: 346 passed, 26 deselected in 10.78s.
- `python -m pytest -m benchmark tests/benchmarks`: 26 passed in 52.11s.
- `python -m pytest tests/test_call_pipeline.py tests/test_call_cli.py tests/test_fused_stages.py tests/test_mode_equivalence.py`: 18 passed in 7.85s.
- `python -m ruff check fiberhmm tests`: passed.
- `python -m compileall -q fiberhmm tests`: passed.
- `python -m pytest`: 347 passed, 26 deselected in 10.78s.
- `python -m pytest -m benchmark tests/benchmarks`: 26 passed in 50.98s.
- `python -m pytest tests/test_recall_tfs_cli.py tests/test_tf_recaller.py`: 23 passed in 0.83s.
- `python -m ruff check fiberhmm tests`: passed.
- `python -m compileall -q fiberhmm tests`: passed.
- `python -m pytest`: 348 passed, 26 deselected in 11.04s.
- `python -m pytest -m benchmark tests/benchmarks`: 26 passed in 51.63s.
- `python -m pytest tests/test_region_planning.py tests/test_inference_parallel.py tests/test_region_types.py tests/test_region_cleanup.py tests/test_call_pipeline.py tests/test_mode_equivalence.py`: 66 passed in 6.02s.
- `python -m ruff check fiberhmm tests`: passed.
- `python -m compileall -q fiberhmm tests`: passed.
- `python -m pytest`: 349 passed, 26 deselected in 9.88s.
- `python -m pytest -m benchmark tests/benchmarks`: 26 passed in 49.21s.
- `python -m ruff check fiberhmm/inference/parallel.py fiberhmm/inference/streaming_drain.py tests/test_inference_parallel.py`: passed.
- `python -m pytest tests/test_inference_parallel.py tests/test_streaming_pipeline.py tests/test_call_pipeline.py tests/test_mode_equivalence.py`: 72 passed in 11.54s.
- `python -m ruff check fiberhmm tests`: passed.
- `python -m compileall -q fiberhmm tests`: passed.
- `python -m pytest`: 349 passed, 26 deselected in 9.76s.
- `python -m pytest -m benchmark tests/benchmarks`: 26 passed in 49.08s.
- `python -m ruff check fiberhmm/inference/parallel.py tests/test_posterior_lifecycle.py`: passed.
- `python -m pytest tests/test_posterior_lifecycle.py`: 2 passed in 0.55s.
- `python -m pytest tests/test_posterior_lifecycle.py tests/test_streaming_pipeline.py tests/test_call_pipeline.py tests/test_mode_equivalence.py tests/test_inference_parallel.py`: 74 passed in 11.76s.
- `python -m ruff check fiberhmm tests`: passed.
- `python -m compileall -q fiberhmm tests`: passed.
- `python -m pytest`: 351 passed, 26 deselected in 9.62s.
- `python -m pytest -m benchmark tests/benchmarks`: 26 passed in 48.75s.
- `python -m ruff check fiberhmm/inference/parallel.py tests/test_posterior_lifecycle.py`: passed.
- `python -m pytest tests/test_posterior_lifecycle.py`: 3 passed in 0.57s.
- `python -m pytest tests/test_posterior_lifecycle.py tests/test_streaming_pipeline.py tests/test_call_pipeline.py tests/test_mode_equivalence.py tests/test_inference_parallel.py`: 75 passed in 11.78s.
- `python -m ruff check fiberhmm tests`: passed.
- `python -m compileall -q fiberhmm tests`: passed.
- `python -m pytest`: 352 passed, 26 deselected in 9.76s.
- `python -m pytest -m benchmark tests/benchmarks`: 26 passed in 49.31s.
- `python -m ruff check fiberhmm/inference/parallel.py fiberhmm/posteriors/region_tsv.py tests/test_region_posteriors.py`: passed.
- `python -m pytest tests/test_region_posteriors.py tests/test_package_consistency.py`: 22 passed in 1.00s.
- `python -m pytest tests/test_region_posteriors.py tests/test_package_consistency.py tests/test_region_types.py tests/test_call_pipeline.py`: 33 passed in 3.67s.
- `python -m ruff check fiberhmm tests`: passed.
- `python -m compileall -q fiberhmm tests`: passed.
- `python -m pytest`: 355 passed, 26 deselected in 9.44s.
- `python -m pytest -m benchmark tests/benchmarks`: 26 passed in 48.85s.
- `python -m ruff check fiberhmm/core/bam_reader.py fiberhmm/inference/tf_recaller.py tests/test_bam_reader.py tests/test_tf_recaller.py`: passed.
- `python -m pytest tests/test_bam_reader.py tests/test_tf_recaller.py tests/test_mm_parser_vs_pysam.py`: 66 passed in 2.12s.
- `python -m ruff check fiberhmm tests`: passed.
- `python -m compileall -q fiberhmm tests`: passed.
- `python -m pytest`: 357 passed, 26 deselected in 10.49s.
- `python -m pytest -m benchmark tests/benchmarks`: 26 passed in 50.49s.
- `python -m ruff check fiberhmm/cli/export_posteriors.py tests/test_export_posteriors_cli.py`: passed.
- `python -m pytest tests/test_export_posteriors_cli.py tests/test_package_consistency.py`: 20 passed in 1.16s.
- `python -m ruff check fiberhmm tests`: passed.
- `python -m compileall -q fiberhmm tests`: passed.
- `python -m pytest`: 358 passed, 26 deselected in 9.92s.
- `python -m pytest -m benchmark tests/benchmarks`: 26 passed in 49.30s.
- `python -m ruff check fiberhmm/posteriors/tsv_backend.py tests/test_tsv_backend.py`: passed.
- `python -m pytest tests/test_tsv_backend.py tests/test_region_posteriors.py tests/test_export_posteriors_cli.py`: 6 passed in 0.67s.
- `python -m ruff check fiberhmm tests`: passed.
- `python -m compileall -q fiberhmm tests`: passed.
- `python -m pytest`: 360 passed, 26 deselected in 10.45s.
- `python -m pytest -m benchmark tests/benchmarks`: 26 passed in 49.52s.
- `python -m ruff check fiberhmm/posteriors/hdf5_backend.py tests/test_posterior_lifecycle.py`: passed.
- `python -m pytest tests/test_posterior_lifecycle.py tests/test_package_consistency.py`: 23 passed in 1.30s.
- `python -m ruff check fiberhmm tests`: passed.
- `python -m compileall -q fiberhmm tests`: passed.
- `python -m pytest`: 361 passed, 26 deselected in 10.06s.
- `python -m pytest -m benchmark tests/benchmarks`: 26 passed in 49.99s.
- `python -m ruff check fiberhmm/cli/extract_tags.py tests/test_extract_block_scores.py`: passed.
- `python -m pytest tests/test_extract_block_scores.py tests/test_package_consistency.py`: 53 passed in 1.19s.
- `python -m ruff check fiberhmm tests`: passed.
- `python -m compileall -q fiberhmm tests`: passed.
- `python -m pytest`: 362 passed, 26 deselected in 10.00s.
- `python -m pytest -m benchmark tests/benchmarks`: 26 passed in 49.61s.
- `python -m ruff check fiberhmm/inference/stats.py tests/test_stats.py`: passed.
- `python -m pytest tests/test_stats.py tests/test_package_consistency.py`: 20 passed in 1.09s.
- `python -m ruff check fiberhmm tests`: passed.
- `python -m compileall -q fiberhmm tests`: passed.
- `python -m pytest`: 363 passed, 26 deselected in 9.63s.
- `python -m pytest -m benchmark tests/benchmarks`: 26 passed in 49.23s.
- `python -m ruff check fiberhmm/daf/encoder.py tests/test_daf_encoder_lifecycle.py`: passed.
- `python -m pytest tests/test_daf_encoder_lifecycle.py tests/test_daf_iupac.py tests/test_call_cli.py`: 29 passed in 1.42s.
- `python -m ruff check fiberhmm tests`: passed.
- `python -m compileall -q fiberhmm tests`: passed.
- `python -m pytest`: 364 passed, 26 deselected in 10.00s.
- `python -m pytest -m benchmark tests/benchmarks`: 26 passed in 50.51s.
- `python -m ruff check fiberhmm/inference/bam_output.py tests/test_bam_output.py`: passed.
- `python -m pytest tests/test_bam_output.py tests/test_package_consistency.py`: 38 passed in 1.24s.
- `python -m ruff check fiberhmm tests`: passed.
- `python -m compileall -q fiberhmm tests`: passed.
- `python -m pytest`: 365 passed, 26 deselected in 9.80s.
- `python -m pytest -m benchmark tests/benchmarks`: 26 passed in 49.44s.
- `python -m ruff check fiberhmm/daf/encoder.py tests/test_daf_encoder_lifecycle.py`: passed.
- `python -m pytest tests/test_daf_encoder_lifecycle.py tests/test_daf_iupac.py tests/test_call_pipeline.py::test_daf_raw_md_and_reference_streaming_match_iupac_output`: 24 passed in 1.97s.
- `python -m pytest -m benchmark tests/benchmarks/bench_encoding.py`: 4 passed in 0.61s.
- `python -m ruff check fiberhmm tests`: passed.
- `python -m compileall -q fiberhmm tests`: passed.
- `python -m pytest`: 366 passed, 26 deselected in 9.77s.
- `python -m pytest -m benchmark tests/benchmarks`: 26 passed in 49.60s.
- `python -m ruff check fiberhmm/inference/bam_output.py tests/test_bam_output.py`: passed.
- `python -m pytest tests/test_bam_output.py tests/test_package_consistency.py`: 40 passed in 1.25s.
- `python -m ruff check fiberhmm tests`: passed.
- `python -m compileall -q fiberhmm tests`: passed.
- `python -m pytest`: 368 passed, 26 deselected in 9.73s.
- `python -m pytest -m benchmark tests/benchmarks`: 26 passed in 49.52s.
- `python -m ruff check fiberhmm/inference/parallel.py fiberhmm/inference/streaming_workers.py tests/test_inference_parallel.py`: passed.
- `python -m compileall -q fiberhmm/inference/parallel.py fiberhmm/inference/streaming_workers.py tests/test_inference_parallel.py`: passed.
- `python -m pytest tests/test_inference_parallel.py`: 48 passed in 0.77s.
- `python -m pytest tests/test_streaming_pipeline.py tests/test_call_pipeline.py tests/test_mode_equivalence.py`: 25 passed in 18.45s.
- `python -m ruff check fiberhmm tests`: passed.
- `python -m compileall -q fiberhmm tests`: passed.
- `python -m pytest`: 369 passed, 26 deselected in 11.84s.
- `python -m pytest -m benchmark tests/benchmarks`: 26 passed in 53.89s.
- `python -m ruff check fiberhmm/inference/parallel.py fiberhmm/inference/region_workers.py tests/test_inference_parallel.py tests/test_region_cleanup.py tests/test_call_pipeline.py`: passed.
- `python -m compileall -q fiberhmm/inference/parallel.py fiberhmm/inference/region_workers.py tests/test_inference_parallel.py`: passed.
- `python -m pytest tests/test_inference_parallel.py tests/test_region_cleanup.py tests/test_region_types.py tests/test_call_pipeline.py tests/test_mode_equivalence.py`: 67 passed in 7.19s.
- `python -m ruff check fiberhmm tests`: passed.
- `python -m compileall -q fiberhmm tests`: passed.
- `python -m pytest`: 370 passed, 26 deselected in 12.29s.
- `python -m pytest -m benchmark tests/benchmarks`: 26 passed in 51.76s.
- `python -m ruff check fiberhmm/inference/parallel.py fiberhmm/inference/mp_context.py tests/test_inference_parallel.py`: passed.
- `python -m compileall -q fiberhmm/inference/parallel.py fiberhmm/inference/mp_context.py tests/test_inference_parallel.py`: passed.
- `python -m pytest tests/test_inference_parallel.py tests/test_region_cleanup.py tests/test_streaming_pipeline.py tests/test_call_pipeline.py tests/test_mode_equivalence.py`: 78 passed in 14.10s.
- `python -m ruff check fiberhmm tests`: passed.
- `python -m compileall -q fiberhmm tests`: passed.
- `python -m pytest`: 371 passed, 26 deselected in 11.91s.
- `python -m pytest -m benchmark tests/benchmarks`: 26 passed in 53.07s.
- `python -m ruff check fiberhmm/inference/parallel.py fiberhmm/inference/streaming_pipeline.py tests/test_inference_parallel.py tests/test_posterior_lifecycle.py tests/test_call_pipeline.py`: passed.
- `python -m compileall -q fiberhmm/inference/parallel.py fiberhmm/inference/streaming_pipeline.py tests/test_inference_parallel.py tests/test_posterior_lifecycle.py`: passed.
- `python -m pytest tests/test_inference_parallel.py tests/test_posterior_lifecycle.py tests/test_streaming_pipeline.py tests/test_call_pipeline.py tests/test_mode_equivalence.py`: 80 passed in 14.16s.
- `python -m ruff check fiberhmm tests`: passed.
- `python -m compileall -q fiberhmm tests`: passed.
- `python -m pytest`: 372 passed, 26 deselected in 12.67s.
- `python -m pytest -m benchmark tests/benchmarks`: 26 passed in 52.33s.
- `python -m ruff check fiberhmm/inference/parallel.py fiberhmm/inference/region_pipeline.py tests/test_inference_parallel.py tests/test_region_cleanup.py tests/test_call_pipeline.py`: passed.
- `python -m compileall -q fiberhmm/inference/parallel.py fiberhmm/inference/region_pipeline.py tests/test_inference_parallel.py tests/test_region_cleanup.py tests/test_call_pipeline.py`: passed.
- `python -m pytest tests/test_inference_parallel.py tests/test_region_cleanup.py tests/test_region_types.py tests/test_call_pipeline.py tests/test_mode_equivalence.py`: 70 passed in 7.28s.
- `python -m ruff check fiberhmm tests`: passed.
- `python -m compileall -q fiberhmm tests`: passed.
- `python -m pytest`: 373 passed, 26 deselected in 11.90s.
- `python -m pytest -m benchmark tests/benchmarks`: 26 passed in 51.54s.
- `python -m ruff check fiberhmm/inference/parallel.py fiberhmm/inference/legacy_pipeline.py tests/test_inference_parallel.py tests/test_posterior_lifecycle.py tests/test_streaming_pipeline.py tests/test_mode_equivalence.py`: passed.
- `python -m compileall -q fiberhmm/inference/parallel.py fiberhmm/inference/legacy_pipeline.py tests/test_inference_parallel.py tests/test_posterior_lifecycle.py tests/test_streaming_pipeline.py tests/test_mode_equivalence.py`: passed.
- `python -m pytest tests/test_inference_parallel.py tests/test_posterior_lifecycle.py tests/test_streaming_pipeline.py tests/test_call_pipeline.py tests/test_mode_equivalence.py`: 82 passed in 14.03s.
- `python -m ruff check fiberhmm tests`: passed.
- `python -m compileall -q fiberhmm tests`: passed.
- `python -m pytest`: 374 passed, 26 deselected in 11.45s.
- `python -m pytest -m benchmark tests/benchmarks`: 26 passed in 51.62s.
- `python -m ruff check fiberhmm/inference/legacy_pipeline.py tests/test_mode_equivalence.py tests/test_streaming_pipeline.py tests/test_inference_parallel.py`: passed.
- `python -m compileall -q fiberhmm/inference/legacy_pipeline.py tests/test_mode_equivalence.py tests/test_streaming_pipeline.py tests/test_inference_parallel.py`: passed.
- `python -m pytest tests/test_mode_equivalence.py tests/test_streaming_pipeline.py tests/test_inference_parallel.py tests/test_posterior_lifecycle.py`: 77 passed in 5.02s.
- `python -m ruff check fiberhmm tests`: passed.
- `python -m compileall -q fiberhmm tests`: passed.
- `python -m pytest`: 374 passed, 26 deselected in 11.49s.
- `python -m pytest -m benchmark tests/benchmarks`: 26 passed in 51.35s.
- `python -m ruff check fiberhmm/inference/region_workers.py tests/test_region_cleanup.py tests/test_call_pipeline.py tests/test_mode_equivalence.py tests/test_inference_parallel.py`: passed.
- `python -m compileall -q fiberhmm/inference/region_workers.py tests/test_region_cleanup.py tests/test_call_pipeline.py tests/test_mode_equivalence.py tests/test_inference_parallel.py`: passed.
- `python -m pytest tests/test_region_cleanup.py tests/test_region_types.py tests/test_call_pipeline.py tests/test_mode_equivalence.py tests/test_inference_parallel.py`: 71 passed in 7.19s.
- `python -m ruff check fiberhmm tests`: passed.
- `python -m compileall -q fiberhmm tests`: passed.
- `python -m pytest`: 374 passed, 26 deselected in 11.44s.
- `python -m pytest -m benchmark tests/benchmarks`: 26 passed in 51.85s.
- `python -m ruff check fiberhmm/inference/parallel.py tests/test_inference_parallel.py tests/test_streaming_pipeline.py tests/test_mode_equivalence.py`: passed.
- `python -m compileall -q fiberhmm/inference/parallel.py tests/test_inference_parallel.py tests/test_streaming_pipeline.py tests/test_mode_equivalence.py`: passed.
- `python -m pytest tests/test_inference_parallel.py tests/test_streaming_pipeline.py tests/test_call_pipeline.py tests/test_mode_equivalence.py tests/test_posterior_lifecycle.py`: 86 passed in 14.18s.
- `python -m ruff check fiberhmm tests`: passed.
- `python -m compileall -q fiberhmm tests`: passed.
- `python -m pytest`: 378 passed, 26 deselected in 11.45s.
- `python -m pytest -m benchmark tests/benchmarks`: 26 passed in 51.18s.
- `python -m ruff check fiberhmm/cli/consensus_tfs.py tests/test_consensus_tfs.py`: passed.
- `python -m compileall -q fiberhmm/cli/consensus_tfs.py tests/test_consensus_tfs.py`: passed.
- `python -m pytest tests/test_consensus_tfs.py tests/test_tf_recaller.py`: 27 passed in 1.34s.
- `python -m ruff check fiberhmm tests`: passed.
- `python -m compileall -q fiberhmm tests`: passed.
- `python -m pytest`: 382 passed, 26 deselected in 12.31s.
- `python -m pytest -m benchmark tests/benchmarks`: 26 passed in 51.52s.
- `python -m ruff check fiberhmm/io/ma_tags.py tests/test_tf_recaller.py tests/test_consensus_tfs.py`: passed.
- `python -m compileall -q fiberhmm/io/ma_tags.py tests/test_tf_recaller.py tests/test_consensus_tfs.py`: passed.
- `python -m pytest tests/test_tf_recaller.py tests/test_consensus_tfs.py tests/test_call_pipeline.py`: 34 passed in 5.09s.
- `python -m ruff check fiberhmm tests`: passed.
- `python -m compileall -q fiberhmm tests`: passed.
- `python -m pytest`: 384 passed, 26 deselected in 11.79s.
- `python -m pytest -m benchmark tests/benchmarks`: 26 passed in 51.60s.
- `python -m ruff check fiberhmm/cli/recall_tfs.py tests/test_recall_tfs_cli.py`: passed.
- `python -m compileall -q fiberhmm/cli/recall_tfs.py tests/test_recall_tfs_cli.py`: passed.
- `python -m pytest tests/test_recall_tfs_cli.py tests/test_tf_recaller.py tests/test_call_pipeline.py`: 33 passed in 4.73s.
- `python -m ruff check fiberhmm tests`: passed.
- `python -m compileall -q fiberhmm tests`: passed.
- `python -m pytest`: 386 passed, 26 deselected in 12.20s.
- `python -m pytest -m benchmark tests/benchmarks`: 26 passed in 53.47s.
- `python -m ruff check fiberhmm/cli/recall_tfs.py fiberhmm/inference/tf_recaller.py tests/test_recall_tfs_cli.py tests/test_tf_recaller.py`: passed.
- `python -m compileall -q fiberhmm/cli/recall_tfs.py fiberhmm/inference/tf_recaller.py tests/test_recall_tfs_cli.py tests/test_tf_recaller.py`: passed.
- `python -m pytest tests/test_recall_tfs_cli.py tests/test_tf_recaller.py tests/test_call_pipeline.py`: 35 passed in 5.19s.
- `python -m ruff check fiberhmm tests`: passed.
- `python -m compileall -q fiberhmm tests`: passed.
- `python -m pytest`: 388 passed, 26 deselected in 12.57s.
- `python -m pytest -m benchmark tests/benchmarks`: 26 passed in 52.18s.
- `python -m ruff check fiberhmm/cli/extract_tags.py tests/test_extract_block_scores.py`: passed.
- `python -m compileall -q fiberhmm/cli/extract_tags.py tests/test_extract_block_scores.py`: passed.
- `python -m pytest tests/test_extract_block_scores.py tests/test_tf_recaller.py tests/test_call_pipeline.py`: 67 passed in 4.47s.
- `python -m ruff check fiberhmm tests`: passed.
- `python -m compileall -q fiberhmm tests`: passed.
- `python -m pytest`: 390 passed, 26 deselected in 12.13s.
- `python -m pytest -m benchmark tests/benchmarks`: 26 passed in 52.00s.

## Current Shape

Largest tracked Python files:

- `fiberhmm/cli/extract_tags.py`: 1395 lines.
- `fiberhmm/core/bam_reader.py`: 1340 lines.
- `fiberhmm/core/hmm.py`: 1123 lines.
- `fiberhmm/cli/train.py`: 1032 lines.
- `fiberhmm/cli/utils.py`: 894 lines.
- `fiberhmm/cli/export_posteriors.py`: 817 lines.
- `fiberhmm/inference/bam_output.py`: 788 lines.

`fiberhmm/inference/parallel.py` is now 189 lines after the streaming, region, and legacy pipeline extractions.

Ignored local build artifacts exist (`build/`, `fiberhmm.egg-info/`) but are not tracked.

## Refactor Priorities

1. `fiberhmm/inference/parallel.py` is now the public dispatcher and compatibility surface.

   The large streaming, region, worker, and legacy apply responsibilities have been moved into focused modules. The remaining file intentionally preserves old private import paths and `process_bam_for_footprints` dispatch behavior.

2. Phase 3 modularization is complete enough to shift toward measured speed/stability work.

   The first shared tagging/unification slice, streaming read-filter slice, active BAM output helper extractions, region worker contracts, aggregation helpers, higher-level worker cleanup tests, fused stage-boundary extraction, streaming drain extraction, streaming worker extraction, region worker extraction, multiprocessing context extraction, streaming pipeline extraction, region pipeline extraction, and legacy pipeline extraction are complete. Keep future cleanup small and tied to either a benchmark or a lifecycle/stability test.

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
- Extract streaming worker initialization and chunk worker entry points; done for the streaming apply and fused apply+recall worker functions.
- Extract region worker initialization and per-region worker entry points; done for apply BAM, BED, and fused BAM workers.
- Extract shared multiprocessing context selection; done.
- Extract streaming pipeline coordinators; done for apply streaming and fused apply/recall streaming.
- Extract region pipeline coordinators; done for apply BAM, BED, and fused apply/recall region-parallel paths.
- Extract legacy chunked apply pipeline; done.
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
