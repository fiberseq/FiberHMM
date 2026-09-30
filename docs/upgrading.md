# Upgrading from 2.x

FiberHMM 3.0 is the third generation of the FiberHMM family, released
together with FiberBrowser 3.0 (which requires `fiberhmm>=3.0,<4`). This page
lists what 2.x users need to act on. The full list of changes is in the
[Changelog](changelog.md).

```bash
pip install --upgrade fiberhmm      # needs Python >= 3.10
```

## Re-run Nanopore Hia5 calls

!!! danger "Nanopore Hia5 emission table"
    The Nanopore Hia5 emission table shipped in **every 2.x release** was
    indexed in alphabetical (ACGT) context order, while the encoder numbers
    bases A, C, T, G. Every context containing a G or a T therefore read
    another context's emission probabilities. 3.0 ships the table reindexed
    (the emission values themselves are unchanged).

    **Re-run ONT Hia5 calls made with 2.x.** About 75–83% of calls keep both
    edges within 5 bp and call totals are about unchanged, but individual
    calls move. The PacBio Hia5 table was not affected. The DddB table had
    the same kind of error and is also reindexed.

The 2.x tables are kept as
`fiberhmm/models/legacy/hia5_nanopore_gt_swapped_legacy.json` and
`legacy/dddb_nanopore_gt_swapped_legacy.json` to reproduce old calls
([Bundled models](reference/models.md#legacy-tables)). The model builder now
always numbers contexts in encoder order, so custom tables built with
`fiberhmm-probs` in 3.0 are correct; **rebuild custom m6A tables made with
the 2.x builder**, which had the same ordering error.

## Defaults that change numbers

| Change | 2.x | 3.0 | To get the 2.x behaviour |
|---|---|---|---|
| ML threshold for Hia5 Nanopore (`call`, `apply`, `recall-tfs`/`-nucs`, `extract`, `qc`, `posteriors`) | 128 / 125 | **248** | `--prob-threshold 128` (or 125) |
| Alignments called by `call`/`apply` | all | **primary only** | `--no-primary` |
| DddA CpG-aware recall in `call` and `pair`/`merge` joint recall | off | **on** (`ddda_ucg` islands exempt), as in `recall-tfs` | `--no-use-m5c` |
| DAF tools reading MM/ML dU (`dedup`, `pair`, `merge`) | ML 0 | **ML 128** | `--prob-threshold 0` |
| DAF duplicate grouping | by alignment orientation | **by deamination flavour** (more duplicates found; for example 5.8% → 9.9% on one DddA data set) | none |
| Looser consensus prevalence tiers (`prevalence_edge`, `prevalence_loose`) | could double-count and exceed 1 | coherent union; drops by up to about 0.14 (typically 0.002–0.04); core unchanged | none (the old value was an error) |

R/Y- and MD-encoded deaminations are binary and unaffected by ML thresholds.

## Behaviour that used to fail silently

These now either work or stop with a clear message:

- Unaligned (uBAM) and stdin input are called instead of being skipped as
  unmapped; a run that skips more than 90% of records as unmapped fails.
- A missing `--seq` is detected from the input instead of assuming PacBio
  (so ONT Hia5 is no longer silently called as PacBio).
- A custom `--model` inherits the input's declared chemistry; conflicts stop
  with a one-line fix (`--replace-chemistry` to re-declare).
- Worker failures above 1% of reads fail the run; outputs are published
  atomically, so a failed run leaves no valid-looking partial BAM.
- Hard-clipped supplementary records whose `MM`/`ML` cannot match `SEQ` are
  skipped instead of being called on the wrong bases.
- `fiberhmm-tag-m5c` writes unmethylated islands (`ddda_ucg`), so
  `recall-tfs --use-m5c` no longer masks every CpG.
- `fiberhmm-apply -c 1` no longer crashes on DAF strand-swap chimeras.
- `fiberhmm-posteriors` decodes reverse reads in the right frame and handles
  DAF.
- EcoGII and custom-enzyme BAMs are never scored with Hia5 emissions in
  consensus.

## Removed and renamed

| 2.x | 3.0 |
|---|---|
| `fiberhmm-run` | `fiberhmm-call`, piped into `ft fire` if needed |
| `python apply_model.py`, `extract_tags.py`, `train_model.py`, `generate_probs.py`, `export_posteriors.py`, `fiberhmm_utils.py` | the `fiberhmm-*` commands |
| `fiberhmm-site-consensus`, targeted families | `fiberhmm-consensus` |
| `fiberhmm-crossstrand` | `fiberhmm-pair` |
| `fiberhmm-merge` | still installed but deprecated: `fiberhmm-pair --from-paired` |
| `fiberhmm-call --ddda-mcg` | `fiberhmm-tag-m5c`, then `fiberhmm-recall-tfs` ([DAF-seq](workflows/daf-seq.md#ddda-cpg-island-methylation)) |
| `fiberhmm-apply --chroms`, `--skip-scaffolds`, `--region-size`, `--scores-db`, `-l` | rejected (they never had an effect); use `fiberhmm-call --region-parallel` |
| `recaller.abutting` | removed; use the "+ edge" tier or `recaller.linker=either` |
| `--engine staged_native_families` (consensus) | deprecated but available |
| `fiberhmm[consensus]`, `fiberhmm[numba]` | still accepted; their packages are core dependencies now |

`fiberhmm-call` gained `--scores` (the `fiberhmm-apply` spelling; the old
`--with-scores` still works) and `-c 0` for all CPUs.

## New in 3.0

- The [lattice recaller](workflows/consensus.md) is the default consensus
  engine, with prevalence tiers, per-channel support and resolution, and
  per-molecule labels; [transfer](workflows/transfer.md) works with it.
- `fiberhmm-consensus` and `fiberhmm-transfer` work on a plain
  `pip install fiberhmm`.
- `fiberhmm-recall-tfs`/`-recall-nucs` accept `--prob-threshold`;
  `fiberhmm-call` accepts `--use-m5c`/`--no-use-m5c` and `--cpg-mask-policy`.
- This documentation site, with a generated
  [command-line reference](reference/cli.md).

## FiberBrowser

FiberBrowser 3.0 requires FiberHMM 3.x. Reload BAMs called with 3.0 to get
the corrected Nanopore Hia5 calls and the lattice-recaller class layers.
