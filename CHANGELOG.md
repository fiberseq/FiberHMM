# Changelog

## 3.0.0

FiberHMM 3.0 is the third generation of the FiberHMM family, released together
with FiberBrowser 3.0.0 (which requires `fiberhmm>=3.0,<4`).

> **Nanopore Hia5 users: re-run your calls.** The Nanopore Hia5 emission table
> shipped in every 2.x release was context-swapped (see *Fixed*). The DddB
> table had the same kind of error until this release.

### New

- **Lattice recaller is the default consensus engine.** `fiberhmm-consensus`
  discovers footprint classes from confident native calls and scores every
  molecule's own modification lattice against them with EM, per chemical
  channel (no Monte Carlo). It reports per-channel support and resolution
  verdicts, per-molecule member/non-member/abstain labels and the recaller's
  own per-molecule calls. BAM export labels native calls with their class
  (`tf_consensus`, `q0` = the molecule's class posterior ×255), records a
  per-dataset `trusted_strand` for DAF classes, and can write the recaller's
  calls as an opt-in `tf_recaller` layer (`--bam-recaller-layer`). See
  [docs/CONSENSUS_WORKFLOW.md](docs/CONSENSUS_WORKFLOW.md).
- **Prevalence tiers.** Each class × channel reports core, edge and loose
  prevalence plus the Wilson lower bound of the core prevalence.
- **Recaller transfer.** `fiberhmm-transfer --freeze-run` on a lattice-recaller
  run writes a versioned, digest-checked frozen class catalog;
  `fiberhmm-transfer --models` scores new data or loci against it without
  rediscovery (EM prevalence, tiers and per-molecule labels are recomputed on
  the target). See [docs/CONSENSUS_TRANSFER.md](docs/CONSENSUS_TRANSFER.md).
- `fiberhmm-consensus` and `fiberhmm-transfer` work on a plain
  `pip install fiberhmm`.
- `fiberhmm-recall-tfs`/`-recall-nucs` accept `--prob-threshold`;
  `fiberhmm-call` accepts `--use-m5c`/`--no-use-m5c` and `--cpg-mask-policy`;
  `fiberhmm-pair`/`-merge` accept `--use-m5c`/`--no-use-m5c`.
- `fiberhmm-call`: `--scores` (as in `fiberhmm-apply`) and `-c 0` for all
  CPUs; `@PG` records the ML threshold, primary-only setting and CpG masking.
- GitHub Actions CI (Linux/macOS, Python 3.10/3.12) with a wheel smoke test.

### Fixed

- **Nanopore Hia5 emission table.** The bundled table was indexed in
  alphabetical (ACGT) context order while the encoder uses A, C, T, G, so every
  context containing G or T read another context's emission in all 2.x
  releases. The table is reindexed (emission values unchanged); ONT Hia5 calls
  from 2.x should be re-run with 3.0. About 75–83% of calls keep both edges
  within 5 bp; call totals are about unchanged. The 2.x table is kept as
  `fiberhmm/models/legacy/hia5_nanopore_gt_swapped_legacy.json` to reproduce
  old calls. The model builder now always numbers contexts in encoder order.
- **DddB emission table** reindexed from ACGT to encoder context order (the
  same error). The old table is kept as
  `legacy/dddb_nanopore_gt_swapped_legacy.json`.
- **Calling entry points.** `fiberhmm-apply` no longer crashes on DAF
  strand-swap chimeras at the default `-c 1`. Unaligned (uBAM) and stdin input
  is called automatically instead of being skipped as unmapped, and a run that
  skips >90% of records as unmapped fails. A missing `--seq` is detected from
  the input instead of silently assuming PacBio. A custom `--model` inherits
  the input's chemistry; conflicts stop with a one-line fix
  (`--replace-chemistry` to re-declare). Hard-clipped records with
  inconsistent MM/ML are skipped. Region-parallel mode keeps every record
  exactly once and refuses unusable plans. Worker failures above 1% fail the
  run. Outputs (call, apply, recall, merge, pair, extract) are published
  atomically, so a failed run leaves no valid-looking partial BAM. Re-calling
  removes stale call tags from skipped reads, and `-k` is validated against the
  model.
- **DAF tools and parsing.** The MM parser steps ML correctly after multi-code
  entries, filters modification codes, and treats `?`-mode unlisted bases as
  unknown. R/Y input gets the chimera filter; MM/ML dU input gets the SNP mask.
  `fiberhmm-tag-m5c` writes unmethylated islands as `ddda_ucg`, so
  `recall-tfs --use-m5c` no longer masks every CpG. `fiberhmm-daf-encode -i -`
  streams stdin. `fiberhmm-pair` skips `0x400` duplicates, and
  `--from-paired` rejects pairing options. Recall keeps MA groups it does not
  regenerate (`deam+`/`deam-`) and warns when the call's SNP mask or reference
  cannot be re-applied.
- **Posteriors, training.** Posteriors decode reverse reads in the call's
  frame and handle DAF; `fiberhmm-probs`, `fiberhmm-train` and
  `fiberhmm-utils transfer` read DAF like `fiberhmm-call` and exit non-zero on
  zero reads; Baum-Welch trains per read; all-zero emission columns are
  neutral.
- **Consensus.** Looser prevalence tiers are a coherent union (a non-member
  adds only its remaining `1 − P`; previously a tier could exceed 1). Staged
  XCR units merge by complete linkage. The declared enzyme is honoured, and
  EcoGII/custom BAMs never resolve to Hia5. Each engine rejects knobs it
  ignores; explicit `sr`/`cross` settings are honoured. Worker pools stop on
  errors; multi-window BEDs stream.

### Changed defaults

These change numbers relative to 2.x.

- **ML threshold per chemistry.** Hia5 on Nanopore (`--seq nanopore`, given or
  detected) calls m6A at ML ≥ 248 in `fiberhmm-call`, `-apply`,
  `-recall-tfs`/`-recall-nucs`, `-extract` and `-qc` (read from the BAM's
  chemistry declaration where the tool has no `--enzyme`); 248 is also the
  threshold the Hia5 Nanopore QC reference is calibrated at, so automatic QC
  after an ONT Hia5 call no longer caps the rate score at WARN for a
  threshold mismatch. Other chemistries keep 128
  (`call`, `apply`) or 125 (`recall`, `extract`, `qc`). An explicit
  `--prob-threshold` always wins.
- **Primary alignments only.** `fiberhmm-call` and `fiberhmm-apply` pass
  secondary and supplementary records through uncalled (`--no-primary` to call
  them).
- **DddA CpG-aware recall everywhere.** `fiberhmm-call` and the joint recall
  of `fiberhmm-pair`/`-merge` now exclude CpG observations from nucleosome and
  TF recall for DddA, except inside confident unmethylated islands
  (`ddda_ucg`), exactly as `fiberhmm-recall-tfs` does (`--no-use-m5c` to turn
  off).
- **DAF tools read MM/ML dU at ML ≥ 128** (`fiberhmm-dedup`, `-pair`, `-merge`;
  was 0). R/Y and MD input is binary and unaffected. `fiberhmm-call`'s
  integrated dedup follows the calling threshold.
- **Duplicates by deamination flavour.** DAF dedup groups reads by C→T vs G→A
  flavour instead of alignment orientation (more duplicates are marked; e.g.
  about 5.8% → 9.9% on one DddA dataset).
- **Python ≥ 3.10.** Numba, scikit-learn, joblib and threadpoolctl are core
  dependencies (`[consensus]` and `[numba]` remain as empty aliases).
- `fiberhmm-apply` rejects `--chroms`, `--skip-scaffolds`, `--region-size`,
  `--scores-db` and `-l`, which never had an effect there.

### Removed

- `fiberhmm-run` (use `fiberhmm-call`, piped into `ft fire` if needed).
- The top-level script shims (`apply_model.py`, `extract_tags.py`,
  `train_model.py`, `generate_probs.py`, `export_posteriors.py`,
  `fiberhmm_utils.py`); use the `fiberhmm-*` commands.
- `fiberhmm-site-consensus` and targeted families (use `fiberhmm-consensus`).
- The staged Monte Carlo consensus engine (`--engine staged_native_families`)
  is deprecated but still available.

### Known issues

- Tools that re-read MM/ML from an existing BAM (`recall-tfs`, `extract`,
  `qc`) use 125 for non-Nanopore chemistries while `call`/`apply` use 128.
- `recaller.abutting` is experimental and biases prevalence and support upward
  when enabled.
- `fiberhmm-tag-consensus` needs an assignment table that no shipped command
  produces; its format is documented in the command's help.
- Legacy `.pickle` models execute code when loaded; load only trusted files.
