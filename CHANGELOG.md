# Changelog

## 3.0.0

FiberHMM 3.0 is the third generation of the FiberHMM family, released together
with FiberBrowser 3.0.0 (which requires `fiberhmm>=3.0,<4`).

> **Nanopore Hia5 users: re-run your calls.** The Nanopore Hia5 emission table
> shipped in every 2.x release was context-swapped (see *Fixed*). The DddB
> table had the same kind of error until this release; re-run DddB calls
> too. `fiberhmm-check <outputs>` lists which of your BAMs, QC reports,
> posteriors files and consensus results need re-running.

### New

- **New commands.** `fiberhmm-pipeline` (reads + reference to a called BAM),
  `fiberhmm-check` (which outputs need re-running), `fiberhmm-consensus` and
  `fiberhmm-transfer` (footprint classes), `fiberhmm-pair` (scDAF duplexes;
  `fiberhmm-merge` is its deprecated merge step), `fiberhmm-strand-rescue`,
  `-strand-rescue-annotate` and `-strand-rescue-audit`,
  `fiberhmm-tag-consensus`, `fiberhmm-footprint-model`, and
  `fiberhmm-tag-m5c`/`fiberhmm-call-m5c` (DddA CpG-island methylation). None
  of these was in a 2.x release; each is described below.
- **`fiberhmm-pipeline`: reads + reference to footprints in one command.**
  FASTQ (or unaligned/aligned BAM) and a FASTA or plasmid map (SnapGene
  `.dna`, GenBank, EMBL) in; a called BAM ready for FiberBrowser, QC and an
  `outputs.json` saying what to open, out. Aligns with minimap2 (the program,
  or the `mappy` module) using the DAF-seq standard `-ax map-ont --MD -Y`,
  with a cached index; keeps primary MAPQ ≥ 20 alignments; joins reads that
  run through the origin of a circular plasmid into one record (the SAM
  circular-reference form) instead of truncating them; calls with
  `fiberhmm-call` defaults and runs `fiberhmm-qc`. A plasmid map's contig is
  named as FiberBrowser names the map, and the header records the reference
  (`@SQ M5`/`TP:circular`, `@CO FIBERHMM-REFERENCE:v1:`). Completed steps are
  skipped on a rerun only while their outputs keep their recorded SHA-256; a
  changed input, reference, setting or `--call-args` file is refused before
  the output directory changes; one run owns an output directory at a time;
  `--progress-json` streams progress for a GUI. A
  Plasmidsaurus-sized run takes one to two minutes. See
  [From a Plasmidsaurus run to footprints in minutes](https://fiberseq.github.io/FiberHMM/getting-started/quick-daf-seq/)
  and [Plasmids](https://fiberseq.github.io/FiberHMM/workflows/plasmids/).
- `fiberhmm-qc` also writes `<prefix>.qc.curves.json`
  (`fiberhmm.qc.curves.v1`): the per-read signal rates and ECDF with the
  bundled reference, the phasogram, footprint-size histograms and duplicate
  cluster sizes behind the QC plot.
- `fiberhmm-extract` splits rows at the origin of `@SQ TP:circular` contigs,
  so reads stored across a plasmid's origin give valid BED/bigBed.

- **Footprint classes: `fiberhmm-consensus`, with the lattice recaller as
  its default engine.** `fiberhmm-consensus` (window, region or BED input;
  several datasets and chemistries at once)
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
- **Consensus BAM export and strand trust.** Exported BAMs carry
  `tf_consensus.QQQQQQ` (`tq`, `fi`, `fq`, `op`, `sq`, `q0`): the class slot,
  the DAF molecule's own core protection ceiling (`sq`) and the class support
  (`q0`); the header's class catalog records, per DAF dataset, which strand
  reports each class reliably (`trusted_strand`, `core_resolution`). See
  [BAM tags](https://fiberseq.github.io/FiberHMM/reference/bam-tags/). The
  earlier staged Monte Carlo engine (`--engine staged_native_families`) is
  shipped as a deprecated alternative; its cross-chemistry comparisons group
  classes into units at the resolution of the coarser chemistry
  (`units.tsv`).
- **scDAF duplexes: `fiberhmm-pair`.** Finds the two reads (CT and GA) of one
  DddA molecule, by sequence or, without shared sequence evidence, with a
  bundled duplex ranker (`ddda_duplex_v1.json`,
  `ddda_duplex_rotational_v1.json`; experimental), merges them into one
  both-strand molecule and re-calls footprints on the joint evidence.
  `--stop-after pair|merge|recall`, `--from-paired`, `--pairs-tsv`,
  `--receipt-json`. `fiberhmm-extract --both-strand` writes the region both
  strands cover (`_bothstrand` BED/bigBed). See
  [Duplex](https://fiberseq.github.io/FiberHMM/workflows/duplex/).
- **Strand rescue.** `fiberhmm-strand-rescue` is an optional, focal
  secondary caller for one-strand chemistries (DddA, DddB, Nanopore Hia5): it
  recovers TF calls missed on one strand from the opposite strand's
  population and moves accepted calls onto a strand-balanced geometry, without
  rewriting the ordinary calls. `fiberhmm-strand-rescue-annotate` writes the
  result as shadow layers (`nuc_sr`, `tf_sr`), `fiberhmm-strand-rescue-audit`
  checks them, and `fiberhmm-tag-consensus` adds a family slot and
  confidence (`fi`, `fq`) to `tf_sr`. See
  [Strand rescue](https://fiberseq.github.io/FiberHMM/workflows/strand-rescue/).
- **`fiberhmm-footprint-model`** summarizes a called BAM's TF calls into
  recurrent loci and geometry families (TSV catalog, BED/bigBed track)
  without re-scoring or filtering the calls.
- **DddA CpG-island methylation.** `fiberhmm-tag-m5c` calls each molecule's
  complete CpG islands as methylated or unmethylated (genome-wide DddA
  DAF-seq only) and writes them as `ddda_mcg`/`ddda_ucg` `MA` intervals,
  which CpG-aware recall uses; `fiberhmm-call-m5c` calls methylated and
  unmethylated domains across all molecules of a region (BED6). See
  [DAF-seq](https://fiberseq.github.io/FiberHMM/workflows/daf-seq/#ddda-cpg-island-methylation).
- **Adjacent-target thinning** for DAF input: `--daf-mask-runs N` and
  `--daf-run-policy keep-one|drop` thin runs of N or more same-strand targets
  (CC on CT reads, GG on GA reads) in the HMM, both recallers, duplex recall
  and the consensus lattices; recorded in `@PG`. On by default for DddA (see
  *Changed defaults*).
- **`MA-TYPES:v1` header lines** advertise the `MA` group names a BAM may
  contain, so a viewer finds rare layers without scanning reads;
  `fiberhmm-utils ma-types` adds them to older BAMs in place (`--types` or
  `--scan`).
- `fiberhmm-recall-tfs`/`-recall-nucs` accept `--prob-threshold`;
  `fiberhmm-call` accepts `--use-m5c`/`--no-use-m5c` and `--cpg-mask-policy`;
  `fiberhmm-pair`/`-merge` accept `--use-m5c`/`--no-use-m5c`;
  `fiberhmm-posteriors` accepts `--prob-threshold`, `--daf-snp-mask` and the
  chimera options (`--keep-chimeras`, `--chimera-*`).
- Model files for development chemistries are shipped for method work only
  (custom `-m`, not accepted by `--enzyme`, not validated):
  `ecogii_pacbio.json` (EcoGII m6A) and `cpg_nanopore.json` (CpG
  methyltransferase), with the `gpc`/`cpg` 5mC observation modes in
  `fiberhmm-probs`, `-train` and `fiberhmm-utils transfer`. See
  [Chemistries](https://fiberseq.github.io/FiberHMM/concepts/chemistries/).
- `fiberhmm-call`: `--scores` (as in `fiberhmm-apply`) and `-c 0` for all
  CPUs; `@PG` records the ML threshold, primary-only setting and CpG masking.
- Every `fiberhmm-*` command accepts `--version` (prints `fiberhmm <version>`).
- **`fiberhmm-check`: which outputs need re-running.** Reads the provenance
  of BAMs (header and a bounded record sample), QC reports, posteriors files
  and consensus result directories and lists the fixes and default changes
  in this release that apply, each with severity (`rerun-required`,
  `rerun-recommended`, `info`), the evidence matched and the exact command to
  re-run; `--json` for scripts and GUIs; exit status 3 when something needs
  re-running. Merged (`samtools merge`), concatenated (`samtools cat`) and
  cyclic `@PG` histories are never reported clean: every calling branch is
  checked, and what the header cannot settle is at least "possibly
  affected". The advisory list ships as `fiberhmm/advisories.json` with the
  digest of every historical Nanopore Hia5 and DddB table; Python API
  `fiberhmm.advisories` (`report`, `check_path`, `check_bam`,
  `check_header`). See
  [Checking outputs for re-runs](https://fiberseq.github.io/FiberHMM/reference/advisories/).
- **Run identity in the chemistry declaration.** Each run's
  `FIBERHMM-CHEMISTRY` line also records the sha256 of the emission tables it
  read (`apply_sha256`, `recall_sha256`, `nuc_model_sha256`),
  `fiberhmm_version`, `fiberhmm_commit` (from a git checkout, or from
  `fiberhmm/_build_info.py`, which a wheel built from a git checkout now
  carries) and `pg`, the `@PG` ID of the run. The required fields are
  unchanged. `fiberhmm-dedup` and `fiberhmm-merge` write an `@PG` record
  (merge's joint recall also a chemistry declaration), and QC reports and
  posteriors files record `fiberhmm_version`.
- **Resumable `fiberhmm-call --region-parallel` runs.** Finished regions are
  kept in a work directory (`.<output>.fiberhmm-work`, or `--work-dir`) with a
  manifest of the input BAM identity (content SHA-256) and every effective
  parameter; `--resume` reuses regions whose BAM digest still matches, reruns
  missing, partial or altered ones and publishes records identical to an
  uninterrupted run, and refuses a changed input or parameter set, even one
  whose size and date were preserved. A work directory has one owner (a
  lock; a second run on it is refused while the first is alive). `SIGTERM`/`SIGHUP` stop a run as cleanly as Ctrl-C (never a
  half-published output). `--progress-json` writes machine-readable progress
  lines (regions done/total, reads/s, ETA) for GUIs and runners. See
  [Long runs and resuming](https://fiberseq.github.io/FiberHMM/workflows/calling/#long-runs-and-resuming).
- **Restartable multi-window consensus runs.** `fiberhmm-consensus --bam …
  --bed/--region` writes each window with an atomic completion marker and
  the run contract in `consensus_run.json`; `--continue` finishes an
  interrupted run in place (skips completed windows, redoes partial ones,
  rebuilds `regions.json`, the report and the BAMs, refuses changed inputs
  (content digests), parameters or DAF run mask, including one inherited
  from `FIBERHMM_DAF_RUN_MASK`). Independent windows now run in parallel (`--window-jobs`,
  default automatic within `--cores`) with per-window logs and a
  windows-done/ETA progress line; results do not depend on the schedule.
  `--resume` keeps its meaning (a new run from saved evidence). See
  [Long runs and resuming](https://fiberseq.github.io/FiberHMM/workflows/consensus/#long-runs-and-resuming).
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
  same error, also in every 2.x release); DddB calls from 2.x should be
  re-run. The old table is kept as
  `legacy/dddb_nanopore_gt_swapped_legacy.json`. 3.0 also ships a new
  in-vivo DddB table: it keeps the (reindexed) naked table's protected state,
  transitions and start probabilities and re-estimates only the accessible
  state's per-context rates in vivo. The naked table is kept as
  `legacy/dddb_nanopore_naked_2f10003c.json`.
- **fibertools BAMs.** `fiberhmm-recall-tfs`/`-recall-nucs --input-frame
  auto` read `ns/nl/as/al` written by fibertools (`ft predict-m6a`,
  `add-nucleosomes`, `fire`, and the older `ft predict`/`ft add`) in molecular
  frame; they were read in SEQ frame, so reverse-read nucleosomes and MSPs
  were mirrored. The frame is decided by one rule shared with FiberBrowser
  (the last footprint writer on each `@PG` chain), and a merged history whose
  chains disagree stops the run instead of guessing. fibertools ≥ 0.13 `Ma`
  tags (nucleosomes and MSPs) are read by recall, consensus and extract, and
  kept by consensus export. Tools that pass footprint tags through (dedup,
  pair, merge, tag-m5c, call-m5c, tag-consensus, strand-rescue-annotate)
  record their frame (`coord=molecular`) in `@PG`. See
  [Footprint-tag frame](https://fiberseq.github.io/FiberHMM/reference/footprint-tag-frame/).
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
  cannot be re-applied. The DddA joint recall of `fiberhmm-merge` stops on a
  DddB-declared input with the chemistry conflict.
- **The DAF SNP screen is deterministic.** For a read whose `MD` is shorter
  than its alignment, pysam fills the rest of the "reference" from undefined
  memory, so a screen without a FASTA (`fiberhmm-daf-snps`, `fiberhmm-call`,
  `fiberhmm-pipeline`) could call different SNPs on every run. Reads whose
  `MD` does not match the CIGAR now use the FASTA, or are skipped without one,
  as encoding and dedup already did; QC does the same. Where reads' `MD` tags
  disagree about a site's base (C in some, G in others), the kept base was
  chosen by hash order, or by which sites the background profile happened
  to sample; it is now the base reported by more classified reads at every
  site (a tie keeps C), and the report counts such sites. An `MD` whose deletion
  run covers a CIGAR insertion is treated the same way. `fiberhmm-pair`'s
  sequence signature (without a FASTA) and `fiberhmm-pipeline`'s check of
  aligned input against the reference never read such an `MD` either (the
  pipeline realigns that input). Called sites are unchanged on well-formed
  input; per-site counts can change where malformed-`MD` reads were used.
- **Circular contigs in the SNP screen and masks.** Positions past a contig's
  end fold back onto it only when the contig is circular (`@SQ TP:circular`,
  or `topology=circular` in the pipeline's `FIBERHMM-REFERENCE` line), on
  every turn of a record that wraps more than once; SNP masks apply after the
  origin too. On linear contigs, positions past the end are dropped instead
  of being relocated onto valid sites.
- **Replacing an existing output.** A BAM and its index are published
  together: the previous BAM and all its indexes are backed up first (hard
  link or verified copy) and restored if publication fails, an obsolete index
  that cannot be removed stops publication, and a failed rerun keeps the
  previous output.
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
  errors; multi-window BEDs load one window per concurrent job.

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
- **TF recall decoder.** The TF recaller finds the best set of
  non-overlapping protected intervals in each scan interval exactly (maximize
  the summed interval LLR minus `--min-llr` per interval) instead of taking
  one maximum per positive-score excursion, so a modified gap can separate two
  adjacent footprints. Emission tables and per-call LLRs are unchanged;
  `--min-llr` is now the per-interval cost. TF calls change for every
  chemistry. Recorded in `@PG` (`tf_decoder=multi_interval_v1`).
- **TF `--min-llr` 5.0 for every preset** (DddB was 4.0).
- **DddA models and nucleosome recall.** `ddda_TF.json` is recalibrated on
  physical scDAF duplexes (the TF-recall table only; the HMM table
  `ddda_nuc.json` is unchanged). Radial nucleosome recall infers each edge
  with a phase-aware posterior (the locked `ddda_nuc_profile.json`,
  `ddda_phase_posterior_v1`) and reads its likelihoods from a separate
  internal table, `ddda_nuc_refine.json`, so a TF recalibration never retunes
  it. That table is context-independent (one hit probability per state;
  sequence context enters through the rotational profile), because a
  per-context pattern frozen from the v2.6.0 TF table did not track SsDddA
  context rates. DddA TF and nucleosome calls change.
- **DddA adjacent-target thinning on.** For DddA, runs of two or more
  same-strand targets (CC on CT reads, GG on GA reads) keep only their 5'-most
  target in `fiberhmm-call`, `-apply`, `-recall-tfs`/`-recall-nucs`, `-pair`,
  `-merge` and the consensus lattices (`daf_run_mask=>=2/keep-one` in `@PG`).
  On scDAF duplexes this gave fewer, more precise TF calls. `--daf-mask-runs
  0` restores the 2.x behaviour; DddB stays off.
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
- The staged Monte Carlo consensus engine (`--engine staged_native_families`)
  is deprecated but still available.

Removed before release (present only in development builds after 2.16.8):

- `fiberhmm-site-consensus` and targeted families, superseded by
  `fiberhmm-consensus` (strand rescue and `fiberhmm-tag-consensus` stay).
- `fiberhmm-crossstrand` and `fiberhmm-duplex`; use `fiberhmm-pair`.
- The per-CpG `fiberhmm-call --ddda-mcg` caller; it now stops and prints the
  `fiberhmm-tag-m5c` workflow.
- The `--enzyme ecogii` and `--enzyme sssi` presets; their model files remain
  for custom `-m` (see *New*).
- The `recaller.abutting` option (its configuration weights were not a
  normalized prior). Molecules whose protected run lines up with one class
  edge are reported in the "+ edge" prevalence tier; for footprints against a
  nucleosome use `recaller.linker=either`. Settings saved with
  `abutting=false` still load; `abutting=true` is refused.

### Upgrading from 2.x

```bash
pip install --upgrade fiberhmm        # Python >= 3.10
fiberhmm-check data/*.bam qc/*.qc.json consensus_out/
```

- **Re-run** Nanopore Hia5 and DddB calls made with any 2.x release (the
  context-swapped tables), Nanopore Hia5 reads called without `--seq` before
  3.0 (they were called as PacBio), and 2.x `fiberhmm-posteriors` output.
  `fiberhmm-check` lists these and the recommended re-runs (DAF duplicates,
  QC, custom tables, `tag-m5c`, recaller tiers) per file, with the command
  to use; it exits 3 when something needs re-running.
- Re-calling with 3.0 also applies the new defaults above (ML 248 for Hia5
  Nanopore, primary-only, the TF decoder, DddA models, thinning and CpG-aware
  recall), so numbers move even where no fix applies. Most have an option to
  get the 2.x behaviour back (the TF decoder and the DddA tables do not); the
  context-swapped 2.x tables are kept under `fiberhmm/models/legacy/`.
- Replace `fiberhmm-run` and the `python *.py` scripts with the `fiberhmm-*`
  commands.
- FiberBrowser 3.0 requires FiberHMM 3.x.

The full guide, with a table of every default change and how to reproduce
2.x numbers, is
[Upgrading from 2.x](https://fiberseq.github.io/FiberHMM/upgrading/).

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

## Earlier versions

Releases up to 2.16.8 have no entries here; they are tagged in git
(`v2.16.8`, `v2.16.7`, …; see the
[repository tags](https://github.com/fiberseq/FiberHMM/tags)).
