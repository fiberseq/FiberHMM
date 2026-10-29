# Scan and association performance

Baseline code is `63333f3`; the after measurements use the uncommitted scan-final worktree. These measurements use three supplied GRCh38 demo slices, pooled as one dataset, on the fixed 50 kb / 25 kb genome grid. The original grid covers 5.4 Mb in 216 units, with 423,196,086 aligned bp (78.37× mean aligned coverage). Primary and supplementary blocks are counted; secondary and duplicate records are excluded. All benchmark outputs are under `/tmp/fh-final-perf`; source BAMs are read only.

A deterministic whole-molecule hash subset near 30× was measured separately: 5.375 Mb, 215 units, 160,587,982 aligned bp, 29.88× mean aligned coverage. The same molecule decision applies to all of its records and slices. Baseline and changed code use the same model resources in this comparison.

Host: Darwin 25.6 arm64, 16 logical CPUs, Python 3.10.13; NumPy 1.26.4, SciPy 1.15.3, scikit-learn 1.6.1, pysam 0.24.0, numba 0.61.2, joblib 1.4.2. Each module runs in a fresh process, sequentially, with four workers and one BLAS/OpenMP thread per worker. Timings cover the scan engine: calibration, unit work, finalization, compression and indexing. Input surveying to select grid zones is excluded. Processes share a disk JIT cache. RSS is the maximum of coordinator and largest-worker high-water RSS, not simultaneous aggregate RSS. Host load is not controlled; the original-depth run had a pronounced long-tail episode.

## P1 at approximately 30×

Extrapolation to 3 Gb holds the measured dataset calibration fixed and scales the remaining elapsed time by aligned-base equivalents. The ideal 16-worker column optimistically divides all non-calibration time by four, including serial finalization. It is a lower-bound thought experiment, not measured or validated 16-worker scaling. The design target is an overnight pooled scan on a 16-core workstation.

| Module | Before wall, s | After wall, s | After peak RSS, GiB | After s/Mb | 3 Gb / 30× / 4 workers, h | Ideal 16 workers, h |
|---|---:|---:|---:|---:|---:|---:|
| fp | 1005.51 | 787.46 | 1.23 | 146.50 | 122.59 | 30.65 |
| nfr | 61.13 | 64.94 | 0.65 | 12.08 | 10.11 | 2.53 |
| peaks | 7.28 | 7.13 | 0.51 | 1.33 | 1.11 | 0.28 |
| nuc | 235.16 | 207.74 | 4.72 | 38.65 | 19.81 | 4.97 |
| Sum of separate module runs | 1309.08 | 1067.26 | — | — | 153.62 | 38.42 |

The combined ideal 16-worker projection is 38.42 h. It exceeds a 12-hour overnight reference window; the target is not met by that assumption.

These are linear projections from small slices. Feature complexity, depth, molecule lengths, I/O and calibration population alter cost. Footprint discovery is nonlinear in depth, which is why the near-30× measurements are preferable to directly scaling the original-depth data. Separate-module sums approximate a combined run, with repeated startup overhead.

| Module | Calibration, s | Unit worker s/Mb | Finalization, s | Stitch/reconcile within finalization, s |
|---|---:|---:|---:|---:|
| fp | 0.00 | 567.56 | 0.50 | 0.00 |
| nfr | 0.00 | 39.89 | 0.73 | 0.00 |
| peaks | 0.00 | 1.67 | 1.50 | 0.53 |
| nuc | 80.61 | 75.67 | 19.95 | 4.11 |

Unit times sum overlapping worker wall time; they are not elapsed scan time. Stitch time includes lazy track I/O in the changed nuc implementation.

## Original-depth measurements

| Module | Before wall, s | After wall, s | Before / after peak RSS, GiB |
|---|---:|---:|---:|
| fp | 2191.96 | 2431.01 | 2.32 / 2.45 |
| nfr | 183.05 | 203.31 | 1.14 / 1.07 |
| peaks | 10.69 | 11.34 | 0.45 / 0.50 |
| nuc | 478.76 | 449.12 | 6.85 / 4.74 |

The original-depth fp and nfr wall measurements do not establish a speedup. The fp paired-unit median was lower, but several long stalls dominated the elapsed comparison. No such median is substituted for the complete measured wall time.

## Output preservation and hotspots

All scan tables and indexes are compared as literal bytes, including empty tables. Annotation finalization was also compared on the Alu genome-grid units using saved unit results.
The complete comparison covers 78 table/index pairs and nine annotated BAMs plus their CSI indexes. There are no unexpected byte differences. In the original-depth nuc run only, calibration JSON and its manifest digest/scan ID differ because the archived package and worktree resolve bundled models to different absolute paths. Model contents and every other calibration field were verified equal. The near-30× runs share model-resource paths and do not require this exception.

| Change / probe | Before | After | Identity check |
|---|---:|---:|---|
| M2 matching, 100,000 anchors | 396.69 s | 0.366 s | complete result SHA-256 equal |
| BED membership, 10,000 anchors and intervals | 5.903 s | 0.00308 s | identical selection mask |
| 100 context loads, 100,000-unit plan | 27.745 s | 0.475 s | public scan outputs preserved |
| 1,500 reference intervals on real reads | 4.727 s | 0.0772 s | projected interval SHA-256 equal |
| Conditional-logistic DP probe, time / peak RSS | 3.065 s / 0.696 GiB | 2.997 s / 0.517 GiB | DP arrays byte-identical |
| M2, 10,000 anchors, fixed-layout/bounded batches | 31.68 s | 26.26 s | same random draws; floating reduction roundoff can differ |
| FP annotation finalization, saved Alu units | 2.831 s | 1.275 s | BAM, CSI and table bytes identical |

Background matching consumes shuffled pools once and uses spatial bins. BED membership uses interval bisection, including earlier enclosing intervals. Scan tasks send a 220-byte descriptor instead of a 9.39 MB plan payload in the 100,000-unit probe, plus one 2.82 MB compressed shared context. Repeated deterministic discovery fits and nuc record outputs use bounded caches; their combined effects are included in the scan rows above.

Table sorting spills 50,000-row batches with stable 32-way merges, preserving exact tie order. Nuc stitching expands only requested tracks and patch slices, with a bounded unit-array cache. Some statistical work arrays still span a whole contig. The changes do not promise constant memory for arbitrarily large contigs or arbitrary dataset counts. Indexed annotated reads merge requested windows, deduplicate by BAM offset and test interval overlap by bisection. M2 permutation batches and conditional-logistic workspace are bounded; rebatching retains the same random draws but can alter floating reduction roundoff.

## Association scaling

Simulations use seed 71. Each case uses 1,000 inference draws: M1 bootstrap; M2 permutations, role permutations and bootstrap; M3 shifts and bootstrap; M4 relocations. M1 includes density, openness and channel covariates and its default conditional-logistic effect fit. M1/M2 have 30 molecules per locus; M2 has one matched background per anchor, one response bin and one supplied density stratum. M3 sizes are molecule counts with 30 events in 1.5 kb admissible exposure per molecule. M4 sizes are hits in each set, sparsely spaced over simulated territory. Array construction is excluded from time; RSS includes it.

| Mode | 10,000 time, s | 100,000 time, s | Peak RSS, 10,000 / 100,000, GiB |
|---|---:|---:|---:|
| M1 | 46.38 | 441.14 | 1.47 / 5.55 |
| M2 | 26.26 | 491.62 | 1.63 / 3.31 |
| M3 | 2.22 | 20.04 | 0.71 / 3.45 |
| M4 | 4.06 | 70.86 | 0.58 / 0.87 |

The 10,000-anchor M1/M2, 10,000-molecule M3 and 100,000-hit M4 simulations fit the design's minutes-scale budgets. Sharing between loci, covariates, physical length and geometry affect these results; they are not universal bounds.

## Validation and reproduction

Final full suite: `3982 passed, 28 skipped, 26 deselected, 7 warnings in 1151.48s (0:19:11)`. Baseline: 3,949 passed. Tests include annotated M1–M3, a real nuc writer/table equivalence check, random transferable backgrounds, reference density/index guards, signal predicates and snapshot identity, stable spilled sorting and lazy track equivalence. CLI reference regeneration and private-string checks are separate checks.

```bash
PYTHONPATH=. python tools/benchmark_scan.py \
  --bam chr1_slice.bam --bam alu_slice.bam --bam mhc_slice.bam \
  --module fp --cores 4 -o /tmp/scan-benchmark/fp

PYTHONPATH=. python tools/benchmark_assoc.py --mode M1 --size 100000 \
  --covariates --draws 1000 --output /tmp/assoc-m1.json
```

Run modules and sizes sequentially in fresh processes. The scan script records its profile and benchmark under `work/`. The association script records time, peak RSS and a full-result digest.

Not measured here: a real 3 Gb data set, actual 16-worker scaling, browser paging/query/overlay latency, and live-tile p50/p95. Browser and e2e validation remain with the coordinator. Signal input support intentionally covers called schema 5 BAMs; legacy signal schemas 3/4 are outside the documented narrow input type. The hub contract outside this worktree was not edited; the v1.12 mirror is in the repository root.

Pytest has reported the result above, but interpreter shutdown remains pending. Existing worker-cancellation tests encounter a sandbox denial of macOS process enumeration in loky/psutil; the initial suite showed the same shutdown stall. Processes have been left untouched under the no-kill constraint. The pytest session exit status and thread/child snapshot are recorded separately under `/tmp/fh-final-perf`.
