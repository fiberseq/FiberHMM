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
  [Footprint classes](https://fiberseq.github.io/FiberHMM/workflows/consensus/).
- **Prevalence tiers.** Each class × channel reports core, edge and loose
  prevalence plus the Wilson lower bound of the core prevalence.
- **Recaller transfer.** `fiberhmm-transfer --freeze-run` on a lattice-recaller
  run writes a versioned, digest-checked frozen class catalog;
  `fiberhmm-transfer --models` scores new data or loci against it without
  rediscovery (EM prevalence, tiers and per-molecule labels are recomputed on
  the target). See [Transferring classes](https://fiberseq.github.io/FiberHMM/workflows/transfer/).
- `fiberhmm-consensus` and `fiberhmm-transfer` work on a plain
  `pip install fiberhmm`.
- `fiberhmm-recall-tfs`/`-recall-nucs` accept `--prob-threshold`;
  `fiberhmm-call` accepts `--use-m5c`/`--no-use-m5c` and `--cpg-mask-policy`;
  `fiberhmm-pair`/`-merge` accept `--use-m5c`/`--no-use-m5c`.
- `fiberhmm-call`: `--scores` (as in `fiberhmm-apply`) and `-c 0` for all
  CPUs; `@PG` records the ML threshold, primary-only setting and CpG masking.
- Every `fiberhmm-*` command accepts `--version` (prints `fiberhmm <version>`).
- GitHub Actions CI (Linux/macOS, Python 3.10/3.12) with a wheel smoke test.
- Documentation site (MkDocs, published to GitHub Pages from `docs/`) with
  getting-started, concept, workflow and reference pages, a synthetic demo
  data generator (`docs/examples/make_demo_data.py`) and a command-line
  reference generated from the argument parsers.

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
  atomically, so a failed run leaves no valid-looking partial BAM. Every
  writer creates a missing output directory (`fiberhmm-call`, `-recall-tfs`,
  `-dedup`, `-tag-m5c`, `-call-m5c` and `-posteriors` stopped with
  `FileNotFoundError`).
  Re-calling removes stale call tags from skipped reads, and `-k` is validated
  against the model.
- **DAF tools and parsing.** The MM parser steps ML correctly after multi-code
  entries, filters modification codes, and treats `?`-mode unlisted bases as
  unknown. R/Y input gets the chimera filter; MM/ML dU input gets the SNP mask.
  `fiberhmm-tag-m5c` writes unmethylated islands as `ddda_ucg`, so
  `recall-tfs --use-m5c` no longer masks every CpG. `fiberhmm-daf-encode -i -`
  streams stdin. `fiberhmm-pair` skips `0x400` duplicates, and
  `--from-paired` rejects pairing options. Recall keeps MA groups it does not
  regenerate (`deam+`/`deam-`) and warns when the call's SNP mask or reference
  cannot be re-applied.
- **`fiberhmm-daf-encode` output is atomic.** The BAM is encoded, sorted and
  indexed as a hidden temporary and published with its index only on success;
  a failed run leaves an earlier output and its index untouched. `-o -` still
  streams.
- **`fiberhmm-apply` chemistry.** A custom `-m` without `--enzyme` on an input
  that declares a supported enzyme now gets that enzyme's defaults (DddA:
  CC/GG keep-one run mask; ML threshold), exactly as `fiberhmm-call` and
  `--enzyme <declared>` with the same table; previously the run mask was
  chosen from no enzyme (off). A chemistry that contradicts the input's
  declaration stops with exit 2 instead of writing footprints under a header
  that declares another chemistry.
- **`fiberhmm-apply` provenance.** Apply output carries an `@PG` record
  (resolved mode, enzyme, `k`, ML threshold, primary-only, DAF run mask;
  `coord=molecular`) and a `FIBERHMM-CHEMISTRY` declaration, written by the
  same helpers as `fiberhmm-call`, so recall, extract and QC read its
  chemistry and a refit `-m` inherits it. A custom `-m` on declared stdin
  input stops with exit 2 (pass `--enzyme`), as in `fiberhmm-call`.
- **QC opportunities.** `fiberhmm-qc` no longer counts bases an MM `?` entry
  leaves unlisted (no call made) as unmodified opportunities, and Nanopore
  Hia5 QC counts the basecalled-forward A's (SEQ T on reverse-aligned reads)
  instead of the opposite strand, so the same molecule gives the same rate in
  either orientation. PacBio, R/Y and MD QC output is unchanged.
- **Posteriors, training.** Posteriors decode reverse reads in the call's
  frame and handle DAF, and resolve `--seq` and the ML threshold like
  `fiberhmm-call`; `fiberhmm-probs`, `fiberhmm-train` and
  `fiberhmm-utils transfer` read DAF like `fiberhmm-call` and exit non-zero on
  zero reads; Baum-Welch trains per read; all-zero emission columns are
  neutral. `fiberhmm-utils transfer` no longer stops with
  `KeyError: 'total'`, and a saved `--accessibility-priors` table serves every
  smaller `-k`.
- **Consensus.** Looser prevalence tiers are a coherent union (a non-member
  adds only its remaining `1 − P`; previously a tier could exceed 1). Staged
  XCR units merge by complete linkage. The declared enzyme is honoured, and
  EcoGII/custom BAMs never resolve to Hia5; a missing, unsupported or
  conflicting chemistry stops `fiberhmm-consensus` and `fiberhmm-transfer`
  with a one-line error (exit 2) before any results are written, instead of a
  traceback. Each engine rejects knobs it ignores; explicit `sr`/`cross` settings are honoured. Worker pools stop on
  errors; multi-window BEDs stream.

### Changed defaults

These change numbers relative to 2.x.

- **ML threshold per chemistry.** Hia5 on Nanopore (`--seq nanopore`, given or
  detected) calls m6A at ML ≥ 248 in `fiberhmm-call`, `-apply`,
  `-recall-tfs`/`-recall-nucs`, `-extract`, `-qc` and `-posteriors` (read from the BAM's
  chemistry declaration where the tool has no `--enzyme`); 248 is also the
  threshold the Hia5 Nanopore QC reference is calibrated at, so automatic QC
  after an ONT Hia5 call no longer caps the rate score at WARN for a
  threshold mismatch. Other chemistries keep 128
  (`call`, `apply`, `posteriors`) or 125 (`recall`, `extract`, `qc`). An explicit
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
- **`fiberhmm-apply -m` on DddA-declared input** applies the DddA CC/GG
  keep-one run mask (see *Fixed*); nucleosome calls from such runs change.
- **Nanopore QC rates.** Per-read ONT Hia5 rates on reverse-aligned reads use
  the correct strand's A count, and `?`-mode reads exclude unlisted bases, so
  `fiberhmm-qc` rates for ONT data shift (about ±1–2% at the quartiles on the
  Drosophila control; `?` data more).

### Removed

- `fiberhmm-run` (use `fiberhmm-call`, piped into `ft fire` if needed).
- The top-level script shims (`apply_model.py`, `extract_tags.py`,
  `train_model.py`, `generate_probs.py`, `export_posteriors.py`,
  `fiberhmm_utils.py`); use the `fiberhmm-*` commands.
- `fiberhmm-site-consensus` and targeted families (use `fiberhmm-consensus`).
- The staged Monte Carlo consensus engine (`--engine staged_native_families`)
  is deprecated but still available.
- The `recaller.abutting` option (added during development; its configuration
  weights were not a normalized prior). Molecules whose protected run lines up
  with one class edge are reported in the "+ edge" prevalence tier; for
  footprints against a nucleosome use `recaller.linker=either`.

### Known issues

- The packaged `hia5_nanopore` QC reference (rate ECDF, rate quantiles and
  rate-stratified exemplars) was computed with the old Nanopore opportunity
  count. Recomputed on the same bounded control sample with the fixed count,
  the 5/25/50/75/95% rate quantiles move from 0.00209/0.0388/0.0735/0.1111/
  0.1832 to 0.00200/0.0392/0.0742/0.1103/0.1876; the phasogram and
  footprint-size references are unaffected. The PASS/WARN rate intervals are
  therefore off by at most a few percent until the reference is regenerated.

- Tools that re-read MM/ML from an existing BAM (`recall-tfs`, `extract`,
  `qc`) use 125 for non-Nanopore chemistries while `call`/`apply` use 128.
- `fiberhmm-tag-consensus` needs an assignment table that no shipped command
  produces; its format is documented in the command's help.
- Legacy `.pickle` models execute code when loaded; load only trusted files.
