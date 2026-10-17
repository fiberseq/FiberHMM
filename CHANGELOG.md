# Changelog

## Unreleased

### Added

- **`fiberhmm-lift`** lifts a called BAM to another assembly through a
  pangenome graph (vg + fibertools-rs `ft pg-pansn`, the stages of `ft
  pg-lift --bam`, which fails on fibertools-rs ≤ 0.13), with footprints in
  molecular coordinates. It repairs what the lift leaves stale: supplementary
  flags, hard clips (filled from the primary), `st` on reads that change
  strand, `MD`/`NM`/`SA` (recomputed with `--target-fasta`, else removed) and
  the `@PG` history. The source `MAPQ` is kept and recorded in `Yq`, the
  source locus in `Yp`. `--repair-only --source` repairs a BAM lifted with
  `ft pg-lift` directly; `--bed` lifts BED12 rows block by block. See
  [Lifting to another assembly](https://fiberseq.github.io/FiberHMM/workflows/lifting/).
- `fiberhmm-check` reports BAMs lifted through a graph without that repair
  (`graph-lift-unrepaired`).

### Fixed

- **fibertools-rs ≥ 0.13 `Ma`/`Aq` input is read everywhere.** `fiberhmm-recall-tfs`
  carries fibertools' `fire.Q` (and other non-footprint) groups into the `MA`
  it writes, and `fiberhmm-call`/`-recall-tfs` drop the superseded `Ma`/`Aq`/`An`
  instead of leaving them beside the new `MA`. `fiberhmm-qc` counts
  nucleosomes and TF footprints from `Ma`, `fiberhmm-footprint-model` loads
  `Ma` molecules, and the `fiberhmm-extract` tag sniff, `fiberhmm-utils
  ma-types --scan` and `fiberhmm-check` see `Ma` calls.

## 3.0.2

### Fixed

- **`fiberhmm-qc` grades DddA against a somatic DddA control.** The packaged
  DddA reference was built from a sperm DAF-seq sample (0.7% median
  deamination, no nucleosome phasing), so normal DddA libraries failed the
  signal rate and the periodicity grade compared them with a flat curve. The
  rate, phasogram, footprint-size and example-molecule references now come
  from the NAPA and UBA1 DddA amplicons (14.6% median deamination, NRL 202
  bp). DddA periodicity allows 15 bp around the reference NRL, since amplicon
  loci differ in repeat length. Re-run `fiberhmm-qc` on DddA BAMs; calls are
  unchanged, and DddB and Hia5 grades are unchanged.
- The nucleosome-size legend no longer covers the long-span note in QC plots.

## 3.0.1

### Fixed

- **`fiberhmm-extract` reads footprint tags in the frame the BAM records**,
  by the same rule as recall, QC and consensus. Nucleosome, MSP and TF
  tracks from BAMs called with FiberHMM 2.12 or earlier were mirrored on
  reverse-strand reads; extract those BAMs again. Tracks from BAMs called
  with 3.0 are unchanged. New `--input-frame auto|molecular|query` sets the
  frame of legacy `ns`/`nl`/`as`/`al`; a merged BAM whose histories
  disagree now stops nucleosome and MSP extraction and asks for it.
- `fiberhmm-extract --both-strand` reads `deam+`/`deam-` groups by the same
  rule. On reverse-strand reads of BAMs without `coord=molecular` the
  both-strand region was mirrored; `fiberhmm-pair` output (forward
  consensus reads) is unchanged.
- The other readers of `MA` groups follow the same per-group frame rule.
  On reverse-strand reads of BAMs called with FiberHMM 2.12 or earlier (or
  whose header lost its `@PG` lines): `fiberhmm-tag-m5c`/`-call-m5c` read
  nucleosome and MSP footprints mirrored; consensus loading placed a `.`
  group wrongly when the same feature also had a `+`/`-` group; consensus
  export moved fibertools `Ma` footprints on headers without frame
  evidence; `fiberhmm-pair` placed nucleosome dyads mirrored; and
  `fiberhmm-utils transfer --reference-bam` read SEQ-frame `ns`/`nl` as
  molecular. Re-run those steps on such BAMs; BAMs called with 3.0 are
  unchanged. The same applies to `fiberhmm-footprint-model` input loading,
  `fiberhmm-strand-rescue`, `-strand-rescue-annotate`, `-strand-rescue-audit`,
  `fiberhmm-tag-consensus` and the native TF quality that consensus export
  copies onto a family.
- Writers keep the annotations they carry through at their original
  positions. `fiberhmm-apply` (`--no-msps` MSPs), `-call`
  (`--no-legacy-tags`) and `-recall-tfs`/`-nucs` (preserved `MA` groups such
  as `tf_sr` and `deam`, kept legacy tags, uncalled reads) declare molecular
  output and now convert what they carry from the input's frame; on reverse
  reads of FiberHMM 2.12 or `@PG`-stripped BAMs these were mirrored. On
  3.0 output their results are unchanged.
- `fiberhmm-tag-m5c` and `fiberhmm-call-m5c --tag-mode locus` on fibertools
  output keep its native `Ma` footprints: the `MA` they create (which wins
  over `Ma`) now carries them.
- Every command that reads or carries footprint annotations prints one
  warning per input file whose header has no FiberHMM/fibertools provenance
  (stripped `@PG`, no `coord=molecular` marker): its reverse-read coordinates
  cannot be confirmed. Results are unchanged.
- The QC that `fiberhmm-call` runs after calling follows `--no-use-m5c` and
  `--cpg-mask-policy` for the sampled reads it calls itself.
- Writing apply tags without footprints keeps the `hl:i:1` marker on reads
  whose harmonized `ns`/`nl` stay in place (Python API; `fiberhmm-apply`
  output is unchanged).
- `fiberhmm-transfer --parameters` naming a parameter group the frozen
  catalog fixes exits 2 with one error line instead of a traceback.
- `fiberhmm-call-m5c` path conflicts exit 2 with a
  `fiberhmm-call-m5c: error:` line, as `fiberhmm-tag-m5c` does.
- `fiberhmm-check` describes the 2.x Nanopore Hia5 and pre-3.0 DddB tables
  as earlier emission tables. Its findings are unchanged.

### Supported inputs

- Standalone pass-through tools (`fiberhmm-dedup`, `-pair`, `-merge`,
  `-tag-m5c`, `-call-m5c`, `-tag-consensus`, `-strand-rescue-annotate`) are
  supported on FiberHMM 3.0 output. On pre-3.0 or header-stripped BAMs,
  re-call with FiberHMM 3.0 first.

### Documentation

- `--input-frame` help (`recall-tfs`/`-nucs`, `tag-m5c`, `call-m5c
  --tag-input-frame`) states the full frame rule.
- `fiberhmm-probs` no longer names the combined `*_probs.tsv` as
  `fiberhmm-train` input; `-p` takes the accessible and inaccessible tables.
- The `fiberhmm-recall-tfs` description no longer mentions `--phase-nrl`.
- CI tests Python 3.14.

## 3.0.0

FiberHMM 3.0 is the third generation of the FiberHMM family, released together
with FiberBrowser 3.0.0 (which requires `fiberhmm>=3.0,<4`).

> The emission tables have been updated, so re-run calls made with earlier
> versions. `fiberhmm-check <outputs>` lists which of your BAMs, QC reports,
> posteriors files and consensus results need re-running.

### Highlights

- **DddA nucleosomes follow the helical phase.** DddA (SsDddA, scDAF-seq)
  also deaminates inside nucleosomes, in step with the ~10-bp helical phase
  of the wrapped DNA. `fiberhmm-call --enzyme ddda` now places nucleosomes
  with a phase-aware rotational model, so internal deaminations no longer
  break nucleosomes into pieces.
- **scDAF duplexes.** `fiberhmm-pair` finds the two strands (CT and GA reads)
  of one DddA molecule and calls footprints on both strands jointly. The DddA
  TF emission table is recalibrated in chromatin on physical scDAF duplexes.
- **CpG-island methylation from deamination.** `fiberhmm-tag-m5c` calls each
  molecule's CpG islands as methylated or unmethylated, and
  `fiberhmm-call-m5c` calls methylated and unmethylated domains across
  molecules. For DddA, CpGs are masked by default outside confidently
  unmethylated islands, in the first-pass HMM and in footprint recall
  (`--use-m5c`, `--cpg-mask-policy`).
- **Cleaner DAF input.** Adjacent CC/GG targets are thinned to one (keep-one,
  on for DddA), strand-swap chimeras are filtered, and PCR duplicates are
  marked by deamination pattern.
- **`fiberhmm-pipeline`** goes from reads and a reference (FASTA or plasmid
  map) to a called BAM ready for FiberBrowser in one command.
- **`fiberhmm-check`** reads your existing outputs and lists which ones need
  re-running, with the command to use.
- **Updated emission tables.** The Nanopore Hia5 table is rebuilt from
  matched controls and DddB has a new in-vivo table.
- **Footprint classes.** `fiberhmm-consensus` finds recurrent footprint
  classes across molecules and measures them per molecule, and
  `fiberhmm-transfer` measures frozen classes in new data.
- **Python 3.10 to 3.14.**

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
- **`fiberhmm-pipeline` keeps basecaller provenance.** Unaligned BAM inputs
  (dorado; PacBio `ccs`/`jasmine`/fibertools) keep their `@RG` lines, `@PG`
  chain and `@CO` lines in the aligned and called BAMs, with minimap2 and
  `fiberhmm-pipeline` chained after them; read-group IDs are made unique
  across inputs and every read's `RG` tag follows its group. RG and
  minimap2's own tags in FASTQ comments are no longer copied by `-y` (they
  gave records a duplicate `RG`). The basecaller, its version, basecalling
  model and modification model(s) (from `@RG DS`, then the basecaller's
  `@PG CL`, then per-read read-group names) are recorded in the
  `fiberhmm-pipeline` and `fiberhmm-call` `@PG DS` (`basecaller=`,
  `basecaller_version=`, `basecall_model=`, `modbase_models=`,
  `basecaller_sources=`) and in `outputs.json` (`settings.basecaller`).
  FASTQ carries none: `--basecaller-info` and `--modbase-model` (also on
  `fiberhmm-call`) record it. Platform detection reads it, so a BAM aligned
  by the pipeline is still known to be Nanopore. Python:
  `fiberhmm.io.provenance.basecaller_provenance(header, reads=...)`.
- **`fiberhmm-pipeline` basecalls raw Nanopore data.** POD5 files or folders
  are basecalled with an installed dorado (not bundled; found on `PATH`, in
  the usual install folders, or with `--dorado`) before alignment: `sup` with
  the 6mA model for Hia5, plain basecalling for DAF-seq (`--dorado-model`,
  `--dorado-modified-bases`, `--dorado-device`, ... change it). The step
  resumes an interrupted dorado run and never repeats a finished one; dorado's
  version, models and command are recorded. See
  [Run everything](https://fiberseq.github.io/FiberHMM/getting-started/run-everything/).
- **Snakemake template.** `workflows/snakemake` runs one `fiberhmm-pipeline`
  per sample of a sample sheet (FASTQ, unaligned BAM or POD5), with local and
  SLURM profiles.
- **DAF calls inside recurrent insertions.** An insertion carried by many
  reads (an amplicon: tens to thousands) gets real evidence instead of the
  no-evidence mask. A pre-pass of `fiberhmm-call` and
  `fiberhmm-recall-tfs`/`-nucs` (`--daf-insert-consensus auto`, the default
  for file input) groups CIGAR insertions of at least 50 bp by breakpoint
  (±30 bp) and length (±25%), builds a deamination-aware consensus of each
  group with at least `--daf-insert-min-carriers` (20) carriers, and
  re-encodes every carrier's inserted bases against it. A column that is
  sometimes C and sometimes T on C->T reads is a C (deaminated in some
  molecules), one that is always T is a T; G/A likewise on G->A reads, and
  each strand reads the other's bases unconverted (the logic of the SNP
  screen), so an open C deaminated in most molecules is still called a C,
  where a majority vote calls it T. Columns are called by a profile
  likelihood with a Phred quality; only confident target columns give
  evidence, a column two alleles share is never confident, and a carrier
  aligning below 85% identity gets none. On synthetic DddB amplicons with a
  1.5 kb insert (open half deaminated at 65-85%, packed half not), the
  consensus matched the insert at 99.3-100% identity from 5 to 500 carriers
  with both strands (every confident column right), and the insert was
  called as the truth (open half >= 99% MSP/TF, packed half 96-98%
  nucleosome) where the mask leaves it uncalled and the 2.x encoding called
  it all nucleosome; with one strand only, open C columns stay right but
  fewer columns are confident (64-89%). Clusters, consensus sequences and
  the fraction of confident columns go to
  `qc/<output>.insert_consensus.json`; `@PG` records
  `daf_insert_consensus=on/<used>of<clusters>/min<N>`. Soft-clipped arms are
  not grouped: with supplementary calling a clipped arm that aligns
  elsewhere (a TE copy) is called against that copy.
- `fiberhmm-qc` also writes `<prefix>.qc.curves.json`
  (`fiberhmm.qc.curves.v1`): the per-read signal rates and ECDF with the
  bundled reference, the phasogram, footprint-size histograms and duplicate
  cluster sizes behind the QC plot.
- **State-aware QC rates.** `fiberhmm-qc` splits the modification rate by
  FiberHMM state: the rate inside MSPs of at least 85 bp (enzyme
  efficiency), the rate outside them (nucleosomes, linkers and shorter gaps;
  background), their ratio, and the share of read length in MSPs. States come
  from the BAM's own calls (`MA`, fibertools `Ma`, legacy `as/al`); an uncalled
  BAM gets a bounded *light call* (the declared chemistry's bundled apply
  HMM on at most 400 sampled reads / a 60 s soft budget, no nucleosome or
  TF recall), and the report says which. Opportunities are the calling
  encoding's own (DAF: deaminated-strand targets with the call's run mask,
  SNP-masked sites removed; Hia5
  PacBio A/T; Hia5 Nanopore basecalled A). Counts and opportunities are
  reported per compartment so confidence intervals can be computed. New
  report keys (`schema_minor_version: 1`, additive to schema 1):
  `state_rates`, `efficiency`, `background`, `overall.components`,
  `overall.verdict_basis`; `.qc.tsv`, `combined.qc.*`, `.qc.curves.json` and
  `fiberhmm-pipeline`'s `outputs.json` (`qc.verdicts`) carry them. New
  options: `--state-source`, `--min-msp-bp`, `--light-call-reads`,
  `--light-call-seconds`. See [QC](https://fiberseq.github.io/FiberHMM/workflows/qc/#state-aware-rates).
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
- **Optional read-order robustness check for footprint classes:**
  `fiberhmm-consensus --robust N` (`recaller.order_replicates`, default 0 =
  off) reruns class discovery and scoring under N other deterministic read
  orders and adds `order_robustness` (the fraction of the N+1 orders that
  find the class supported) and `robust` (every order, or the share set by
  `recaller.order_robust_fraction`) to `classes.tsv`, the result and the
  manifest. The classes themselves are unchanged; the check takes roughly
  N+1 times as long and nothing changes when it is off. See
  [Reproducibility](https://fiberseq.github.io/FiberHMM/workflows/consensus/#reproducibility).
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
  `fiberhmm-call`, `-apply`, `-posteriors`, `-recall-tfs`/`-recall-nucs`,
  `-pair`/`-merge` and `-pipeline` accept `--use-m5c`/`--no-use-m5c` and
  `--cpg-mask-policy`;
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
  of BAMs (header and a bounded record sample), QC reports, posteriors files,
  consensus result directories and `fiberhmm-pipeline` output directories and
  lists the fixes and default changes in this release that apply, each with
  severity (`rerun-required`, `rerun-recommended`, `unverifiable`, `info`),
  the evidence matched and the exact command to re-run; `--json` for scripts
  and GUIs; exit status 3 when something needs re-running, 4 when calls carry
  no FiberHMM provenance and cannot be verified (re-run them if they came from
  FiberHMM < 3.0), 2 when a path cannot be checked. A BAM with nothing
  FiberHMM made is reported as not a FiberHMM output rather than clean. Merged (`samtools merge`), concatenated (`samtools cat`) and
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
- GitHub Actions CI (Linux/macOS, Python 3.10/3.12/3.13) with a wheel smoke test.
- Documentation site (MkDocs, published to GitHub Pages from `docs/`) with
  getting-started, concept, workflow and reference pages, a synthetic demo
  data generator (`docs/examples/make_demo_data.py`) and a command-line
  reference generated from the argument parsers.

### Fixed

- **Dedup keeps every candidate pair.** Deamination-fingerprint dedup dropped
  qualifying candidate edges once a read had matched a bucket representative,
  so families bridged only through non-representatives were left split.
  Candidate components now keep all members and merge every matching family,
  without quadratic fingerprint comparisons for families of identical reads.
  The default Jaccard threshold stays 0.95; `fiberhmm-call
  --dedup-min-jaccard 0.90` and `fiberhmm-dedup --min-jaccard 0.90` are
  documented options.
- **Re-deduplicating recomputes the duplicate marks.** `fiberhmm-dedup` and
  `fiberhmm-call`'s integrated dedup only ever set `0x400` and the `di`/`ds`
  tags, so re-calling a BAM that already carried marks (a 2.x or earlier 3.0
  output) kept the old ones: the log reported the new duplicate count while
  the BAM held both sets. Earlier marks are now cleared on every record before
  marking (`prior_marks=cleared` in `@PG`), also when no read can be
  fingerprinted, and `fiberhmm-check` no longer recommends re-running dedup
  when such a run follows the old one.
- **`fiberhmm-tag-m5c` island inference on hg38 headers.** Without
  `--cpg-islands`, every `@SQ` contig had to be in the reference FASTA, so the
  command failed on BAMs listing `chrEBV` (absent from UCSC `hg38.fa`) even
  when no read mapped there. Islands are now inferred only for contigs that
  have reads (faster too); a read contig missing from the FASTA is skipped with
  a note.
- **`fiberhmm-tag-m5c` and `fiberhmm-call-m5c` outputs are atomic and
  guarded.** Each command publishes all its outputs as one group only when
  the run succeeds (tag-m5c: the BAM, `--calls-tsv` and `--write-cpg-islands`;
  call-m5c: the BED, the `--tag-output` BAM in either `--tag-mode` and
  `--write-cpg-islands`). If any publication step fails, every earlier output is
  restored, and an index of an earlier BAM is removed. No output may name an
  input BAM, the reference, the island catalog, any of their indexes or
  another output.
- **`fiberhmm-apply` keeps `ddda_ucg`/`ddda_mcg`.** Apply dropped every `MA`
  group, so recalling an applied tag-m5c BAM lost the island calls the CpG
  mask depends on; region-parallel apply also read them only after stripping.
  The island groups are kept, with this run's nucleosomes (footprints outside
  MSPs) and MSPs in `MA`; legacy `ns/nl` still lists every HMM footprint, as
  `fiberhmm-recall-tfs` expects, so it can hold more footprints than `MA`.
- **`fiberhmm-apply --harmonize-legacy-tags` (opt-in).** Writes `ns/nl` as the
  nucleosomes `MA` would hold (footprints outside MSPs), so the legacy tags tile
  with `as/al` as in `fiberhmm-call` output; `as/al`, `MA` and the calls are
  unchanged, and `@PG` records `legacy_tags=harmonized`. It is not the default
  because recall reads the raw footprints from `ns/nl`: recalling harmonized
  arrays changed recall output on 147 of 3,202 NAPA DddA reads, 42 of 169 Hia5
  PacBio and 358 of 400 Hia5 Nanopore reads. `fiberhmm-recall-tfs`/`-nucs`
  therefore refuse (exit 2) any read carrying `hl:i:1`, which every harmonized
  read gets. A later writer that replaces `ns/nl` (call, apply, recall)
  removes the tag; one that rewrites only MSPs or `MA` keeps it. The
  region-parallel apply API writes the same `@PG` record. The default apply
  output is unchanged.
- **Posteriors API.** `fiberhmm.cli.export_posteriors.export_posteriors_from_bam`
  accepts `enzyme`, `use_m5c` and `cpg_mask_policy`, resolves the DddA run
  thinning and CpG mask like `fiberhmm-posteriors`, and now applies
  `min_mapq` (default 0).
- **Footprint frames of files whose `@PG` was stripped.** `fiberhmm-call`
  output now also carries `@CO fiberhmm:coord=molecular` (it was only in the
  `@PG DS`). Wherever the header has frame evidence (a footprint writer on a
  `@PG` chain, or `coord=molecular`), frames are resolved exactly as before.
  Only without evidence: each `MA` group's own strand decides that group's
  frame (`.` molecular, `+`/`-` the header rule); legacy `ns/nl/as/al` follow a `.` nuc/msp/tf `MA`
  group on the same read, else SEQ with a warning (recall, tag-m5c, call-m5c
  and QC); a chemistry declaration decides nothing. On NAPA, scDAF and Hia5
  ONT call output with `@PG`/`@CO` stripped, recall-tfs now matches the
  unstripped run on every read (before: 1,601, 53 and 195 reads differed).
- `--cpg-mask-policy` given for a non-DddA run, or with `--no-use-m5c`, now
  warns instead of being ignored silently. Worker processes no longer reset
  `NUMBA_CACHE_DIR`, which made numba write its cache next to the installed
  sources. Progress lines redirected to a log are written as whole lines.
  The QC summary line names periodicity among the verdict's components.

- **DAF insertions and soft clips are no longer called as protected.**
  Deaminations are read-versus-reference mismatches, so bases with no
  reference counterpart (CIGAR insertions and soft clips) can never carry one,
  yet their unconverted C (CT strand) / G (GA strand) were counted as
  unmodified, i.e. protected, targets. An insertion or clip was therefore
  called nucleosome-packed whatever its chromatin: on synthetic 2 kb inserts
  with an open patch, DddA called a nucleosome array (85-90% of the open
  patch nucleosome) and DddB Nanopore one 2.1 kb "nucleosome". These bases
  are now masked like SNP sites (no evidence either way), in every DAF input
  form (R/Y, MD or reference, MM/ML) and in `fiberhmm-call`,
  `fiberhmm-recall-tfs`/`-nucs`, strand rescue, consensus replay and QC. An
  unaligned stretch of 50 bp or more is left uncalled (no nucleosome, MSP or
  TF; a nucleosome or MSP running into it is trimmed at its edge, and a
  trimmed nucleosome edge gets edge quality 0). The SNP mask itself had the
  same flaw: it removed a masked site's deamination but left an unconverted
  masked C/G counted as protected; masked sites are now no evidence too.
  Reads without insertions or clips are called exactly as before (NAPA DddA
  demo, 3,202 reads: all 534 reads without I/S identical; 317 of 2,582 reads
  with small indels or clips changed, about 0.9% of their nucleosome and MSP
  calls and 0.3% of TF calls; 85 of 86 reads with >= 50 bp clips lose the
  calls in the clip). On by default, recorded in `@PG` as
  `daf_unaligned_mask=on`; `--no-daf-mask-unaligned` restores the old
  encoding. `fiberhmm-qc` reports the masked share of sampled DAF bases
  (`unaligned_masking`). Fiber-seq m6A is read-intrinsic and unchanged.
  `fiberhmm-check` flags affected calls (`daf-unaligned-evidence`).
- **QC assay detection.** `fiberhmm-qc` (and the QC step of
  `fiberhmm-pipeline`) took the assay from the first `mode=` anywhere in the
  BAM header; on a deduplicated DAF BAM that was the dedup step's
  `mode=flag`, so every such report was graded INSUFFICIENT with "m6A
  labeling 0%". The assay now comes from the chemistry declaration (then the
  newest FiberHMM call record), the pipeline passes `--mode`/`--enzyme`
  explicitly, and `fiberhmm-check` flags old reports graded under a
  non-assay mode (`qc-assay-misdetected`; re-run QC, or `--redo qc`).
- **Footprint classes no longer depend on where the BAM lives or what the
  dataset is called.** Evidence units were named by a hash that included the
  BAM's absolute path and the dataset label, and that name orders the
  molecules class discovery sees and picks its split-halves and folds: the
  same BAM opened from two folders could give a different number of classes
  (on the demo window 16, 18 or 20). Units are now named by the dataset's
  position in the run, the file's position in the dataset and the alignment
  record, so the same BAMs and parameters give identical classes on any
  machine (the order of datasets and files is part of the input). The same
  holds for `--pool-loci` view selection and for the deprecated
  `staged_native_families` engine, which now runs on positional dataset names
  internally (its family IDs differ from earlier runs and its fit
  checkpoints from earlier runs are not reused). Class
  counts from earlier runs may differ for classes near the thresholds;
  well-supported classes are unchanged.
- **Footprint-class discovery keeps footprints k-means pooled with a
  neighbour.** When a tile's k was held down by its least stable cluster,
  k-means could pool the calls of neighbouring footprints into one
  candidate. That candidate failed the core rule and was dropped, and every
  footprint in it was lost. At NAPA's secondary NFR this dropped the
  strongest class (47,518,363–386) under some read orders. On ind Hia5,
  whose footprints are narrow and close together, it dropped 15 of 18
  candidates and left no supported class. Such a candidate is now clustered
  again on its own calls under the same rules (`recaller.core_resplit_depth`,
  default 2; 0 restores the old drop). Tiles are now deduplicated after the
  core rule, so a dropped geometry no longer hides the same class found
  valid in another tile. Separately, a class's stability is
  now its prediction strength averaged over every split-half. It used to
  come from one split, because per-split values were keyed by rounded
  centroids that rarely matched.
- **Nanopore Hia5 emission table updated.** 3.0 replaces the bundled table
  with one rebuilt from matched controls (next entry),
  so ONT Hia5 calls from 2.x should be re-run with 3.0. The 2.x table is kept
  as `fiberhmm/models/legacy/hia5_nanopore_gt_swapped_legacy.json` to
  reproduce old calls. The model builder now always numbers contexts in the
  encoder's order (A, C, T, G).
- **New Nanopore Hia5 emission table from matched controls.** The bundled
  table is rebuilt with `fiberhmm-probs` from Hia5-treated naked DNA
  (accessible state) and untreated DNA (protected state), both Drosophila
  2–4 h embryo genomic DNA sequenced on R10.4.1 with dorado
  `sup@v5.2.0` and the `6mA@v1` model, counted at the preset's ML ≥ 248.
  Naked-DNA reads in a low-methylation component (carryover) are excluded.
  Start and transition probabilities are the Hia5 PacBio model's. On 2–4 h
  embryo reads the median nucleosome call moves from 166 to 151 bp and the
  fraction of bases in MSPs from 15% to 24%. The reindexed 2.9 values are
  kept as `legacy/hia5_nanopore_v2.9_reindexed_legacy.json`.
- **DddB emission table updated.** The bundled DddB table is
  re-calibrated; DddB calls from 2.x should be re-run. The
  old table is kept as `legacy/dddb_nanopore_gt_swapped_legacy.json`. 3.0
  also ships a new in-vivo DddB table: it keeps the re-calibrated naked
  table's protected state,
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
  smaller `-k`. `fiberhmm-train --base-model` now matches the new emission
  rows to the base model's accessible state: with a base whose state 0 is
  the footprint (such as the Nanopore Hia5 model bundled in 2.x), every 2.x release
  paired the inherited transitions with inverted states, so the resulting
  model called accessible DNA as footprint. Re-train models built with
  `--base-model` on such a base.
- **Consensus.** Looser prevalence tiers are a coherent union (a non-member
  adds only its remaining `1 − P`; previously a tier could exceed 1). Staged
  XCR units merge by complete linkage. The declared enzyme is honoured, and
  EcoGII/custom BAMs never resolve to Hia5; a missing, unsupported or
  conflicting chemistry stops `fiberhmm-consensus` and `fiberhmm-transfer`
  with a one-line error (exit 2) before any results are written, instead of a
  traceback. Each engine rejects knobs it ignores; explicit `sr`/`cross` settings are honoured. Worker pools stop on
  errors; multi-window BEDs load one window per concurrent job.
- **Shared results are readable.** Consensus and transfer outputs (result
  JSONs, family BAMs), strand-rescue reports, BAMs and audits, and
  `fiberhmm-tag-consensus` BAMs were created owner-only (0600) whatever the
  umask; they now get the permissions any other file the user writes gets.
- **Publishing BAM and index together.** `fiberhmm-tag-consensus` and
  `fiberhmm-utils ma-types` publish the BAM and its index as one transaction
  (a failure restores the previous pair). `fiberhmm-utils fix-bigbed` exits 1
  when an input is missing or cannot be converted, and replaces the bigBed
  atomically beside it.
- **Custom models and edge cases.** TF/nucleosome recall refuses emission
  tables that are not k=3 (a k=4 model silently gave wrong calls), so
  `fiberhmm-call`, which always recalls TFs, now stops before any work on a
  custom model with another context size. 5mC
  (`gpc`/`cpg`) reverse-aligned reads are encoded in the same C-centred
  context as forward reads. The HMM no longer returns NaN for a state that a
  zero start or transition probability makes unreachable.
- **Remembered file digests are not reused for racily clean files.** The
  digest memo behind `--resume`, `--continue`, pipeline step markers,
  reference digests and provenance header digests reused a SHA-256 while a
  file's device, inode, size, mtime and ctime were unchanged. File timestamps
  only advance once per clock tick (a few ms on Linux, 2 s on FAT), so a
  same-size rewrite made right after hashing could leave every field equal
  and return the old digest (seen on WSL2 ext4 in every immediate trial). A
  digest is now remembered only when the file's mtime and ctime were at least
  3 s older than the start of hashing (git's "racily clean" rule); a file
  hashed while fresh is hashed again on its next lookup. Old files are still
  served from the memo. Memo entries written by earlier versions carry no hash
  time and are rehashed once. A remembered digest is also not returned after
  the clock was set back to within the margin of the file's timestamps, and a
  file that changes while it is hashed is read again instead of yielding a
  digest of mixed bytes. A reference `.fai` newer than its FASTA is reused
  only while it lists the FASTA's contig names and lengths.
- **Smaller fixes.** `--region-parallel` progress and logs go to stderr
  (stdout carries data only), and outputs no longer carry a `samtools cat`
  `@PG` listing the temporary work directory. `fiberhmm-call-m5c` refuses
  `-o -` with `--tag-output -`; the DddA preflight of `fiberhmm-tag-m5c`/
  `-call-m5c` reads the declared enzyme instead of any "hia5" in a path.
  `fiberhmm-strand-rescue` names a `--region` contig the BAM lacks (exit 2).
  The SNP report's per-call `amplicon_ids` no longer collapse when two
  amplicons swap places during renumbering.
- **`fiberhmm-extract` keeps intervals with clipped or inserted edges.** An
  interval whose first or last base was soft-clipped or an insertion was
  dropped from every BED/bigBed track: on minimap2 ONT/DAF alignments 11–16%
  of MSPs (every terminal MSP of a soft-clipped read), 1–7% of nucleosomes
  and about a quarter of TF calls. Such intervals now span their aligned
  bases; only intervals with no aligned base are dropped. Tracks, counts and
  aggregates built from extract output gain these features; features that
  were already extracted are unchanged. The BAM tags and FiberBrowser are
  unaffected. Extract also uses an existing CSI index and indexes an
  unindexed BAM into a temporary file instead of writing a `.bai` next to it.
- **MSP `aq` is a confidence.** `fiberhmm-apply --scores` wrote
  `aq` = mean P(footprint) over each MSP, so confidently accessible MSPs
  scored near 0 (demo median 5). It is now mean P(accessible) ×255 (demo
  median 249), as documented. Extract MSP BED scores and `blockAq` change
  with it; `fiberhmm-call` writes no `aq`.
- **Recall of `--no-legacy-tags` output.** `fiberhmm-recall-tfs` on a BAM
  whose footprints are only in `MA` deleted every annotation and exited 0;
  `-recall-nucs` left it unchanged. Both now read the `MA` footprints and give
  the same result as on the same call with legacy tags.
- **DddA re-calls.** A recall of DddA `fiberhmm-call` output does not
  reproduce the call (its TF scan space comes from HMM footprints the output
  does not keep; on the demo, TF calls change on 370 of 377 reads). The
  recallers now warn on such input, and the CpG-island workflow re-runs
  `fiberhmm-call` on the `fiberhmm-tag-m5c` output instead of
  `fiberhmm-recall-tfs`. Hia5 and DddB recalls reproduce the call at the
  call's ML threshold (for Hia5 PacBio pass `--prob-threshold 128`; see
  Known issues). The DddA radial recaller no longer fails on reads shorter
  than 41 bp.
- **Input checks.** An explicit `--seq` that the reads' MM specs contradict
  (for example `--seq pacbio` on Nanopore reads, ~100× more TF calls) now
  stops `call`, `apply`, `recall-tfs`/`-nucs` and `posteriors` with exit 2;
  `--force-seq` keeps it. Hia5 on reads without m6A calls stops instead of
  writing no footprints. A missing or non-BAM input, a JSON that is not a
  model, and bad `fiberhmm-consensus --region` values are one-line errors
  (exit 2) instead of tracebacks; `--prob-threshold` must be 0–255;
  `--chroms`/`--skip-scaffolds` need `--region-parallel`; `.sam`/`.cram`
  output names are refused; a custom `-m` of the other assay than
  `--enzyme` is refused; `fiberhmm-merge` accepts an empty BAM, and
  `fiberhmm-dedup` exits 1 when it writes nothing.

### Changed defaults

These change numbers relative to 2.x.

- **QC verdict uses in-MSP efficiency and outside-MSP background.** Where
  the profile has a state-aware reference (DddB, DddA, Hia5 PacBio), the
  overall QC status combines the in-MSP rate, the outside-MSP rate and
  periodicity; the overall signal rate is still reported and graded in
  `signal` but no longer decides the verdict, because it also reflects how
  accessible the sample's chromatin is (an amplicon at an open locus labels
  more of its length than genome-wide data at the same enzyme efficiency).
  Hia5 Nanopore, without a state-aware reference yet, keeps the
  overall-rate verdict, as does `--state-source none`. Both rates are graded
  relative to the reference median: PASS within 20% of it, WARN 20–30% off
  (below for efficiency, above for background), FAIL beyond 30%.

- **ML threshold per chemistry.** Hia5 on Nanopore (`--seq nanopore`, given or
  detected) calls m6A at ML ≥ 248 in `fiberhmm-call`, `-apply`,
  `-recall-tfs`/`-recall-nucs`, `-extract`, `-qc` and `-posteriors` (read from the BAM's
  chemistry declaration where the tool has no `--enzyme`); 248 is also the
  threshold the Hia5 Nanopore QC reference is calibrated at, so automatic QC
  after an ONT Hia5 call no longer caps the rate score at WARN for a
  threshold mismatch. Other chemistries keep 128
  (`call`, `apply`, `posteriors`) or 125 (`recall`, `extract`, `qc`). An explicit
  `--prob-threshold` always wins.
- **`fiberhmm-posteriors` trims 10 bases at read ends** (was 100), as
  `fiberhmm-call` and `fiberhmm-apply` do, so posteriors within 100 bases of
  a read end change. `--edge-trim 100` restores the 2.x behaviour.
- **Primary and supplementary alignments.** `fiberhmm-call` and
  `fiberhmm-apply` call primary and supplementary records and pass secondary
  records through uncalled (`--alignments`, default `primary-supplementary`).
  A supplementary record is another part of the same read (the far side of a
  structural variant, an insertion's transposon copy elsewhere); for DAF it
  carries real deamination evidence against that copy's reference, which
  2.x never used (DAF supplementary records were skipped even with
  `--no-primary`). A supplementary record is called on its aligned bases
  only: its soft clips are the primary record's sequence. `--primary` calls
  primary records only; `--no-primary` (`--alignments all`) every record.
  `fiberhmm-dedup` flags (or, collapsing, drops) a duplicate's supplementary
  and secondary records with it. Supplementary records shorter than
  `--min-read-length` aligned bases are skipped like any record. On a DddB
  Nanopore Drosophila BAM (4,203 records, 446 supplementary) 198 more
  records were called, +0.2% output size and about +4% run time; on a
  minimap2 Hia5 Nanopore BAM without `-Y` the supplementary records are
  hard-clipped and stay skipped (`hard_clipped_mm`).
- **`fiberhmm-pipeline` keeps split reads and soft clips.** The aligner step
  keeps a read's supplementary records (and its `SA` tag) on linear contigs,
  and DAF reads keep their soft clips there (calling treats them as no
  evidence; they are the read's own sequence for structural-variant views).
  On circular contigs DAF reads are hard-clipped as before (concatemer arms
  beyond one full circle) and supplementary records are joined across the
  origin or dropped, so a plasmid molecule is not annotated twice.
  `--hard-clip` clips everywhere, `--keep-soft-clips` nowhere,
  `--alignments primary` keeps the primary record only. `outputs.json`
  settings gain `alignments` and `daf_unaligned_mask`; `hard_clip` is now
  `on`/`off`/`circular` and `primary_only` is true only for
  `--alignments primary`.
- **TF recall decoder.** The TF recaller finds the best set of
  non-overlapping protected intervals in each scan interval exactly (maximize
  the summed interval LLR minus `--min-llr` per interval) instead of taking
  one maximum per positive-score excursion, so a modified gap can separate two
  adjacent footprints. Emission tables and per-call LLRs are unchanged;
  `--min-llr` is now the per-interval cost. TF calls change for every
  chemistry. Recorded in `@PG` (`tf_decoder=multi_interval_v1`).
- **TF `--min-llr` 5.0 for every preset** (DddB was 4.0).
- **One nucleosome-recall mode for Hia5 and DddB: the accessible-cut split,
  without a periodicity prior.** 2.x and 3.0 development builds split long protected
  blocks with a periodicity prior (estimated nucleosome repeat length) and,
  for Nanopore Hia5, used a `topology` recall policy. Both are retired: the
  prior split blocks at least 1.5 repeat lengths long on a single mark, which
  left a step at that length in nucleosome call lengths and most one-base
  MSPs; `topology` kept the HMM's edges and an 85-bp floor, which left a
  pile-up just above 85 bp and the HMM's ~10-bp edge comb. Nucleosome calls
  change for Hia5 and DddB; DddA radial recall is unchanged. The
  `--phase-nrl` and `--nuc-recall-policy` options are retired.
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
- **DddA CpG-aware calling everywhere.** For DddA, `fiberhmm-call`,
  `fiberhmm-apply`, `fiberhmm-posteriors` and the joint recall of
  `fiberhmm-pair`/`-merge` exclude CpG observations, except inside confident
  unmethylated islands (`ddda_ucg`), from the first-pass HMM as well as from
  nucleosome and TF recall (`fiberhmm-recall-tfs` applies the same recall
  mask). The segmentation then does not depend on CpG methylation, and every
  tool segments a read the same way. The first-pass mask changes about half
  of DddA reads: on the NAPA demo (3,202 reads) 1,640 reads change and TF
  calls fall by 2.1% (18,836 to 18,437); nucleosome and MSP calls stay close
  (per-read Jaccard 0.98-0.996 on NAPA, UBA1 and HG002 scDAF). Recorded in
  `@PG` as `hmm_cpg_mask=on`; `--no-use-m5c` turns both masks off and
  `--cpg-mask-policy methylated-only` masks only `ddda_mcg` spans.
  `fiberhmm-pipeline` passes both options to its call, and the light call of
  `fiberhmm-qc` on an uncalled DddA BAM applies the same first-pass mask.
  `fiberhmm-check` flags DddA calls and posteriors made before the
  first-pass mask (`ddda-hmm-cpg-mask`).
- **DAF tools read MM/ML dU at ML ≥ 128** (`fiberhmm-dedup`, `-pair`, `-merge`;
  was 0). R/Y and MD input is binary and unaffected. `fiberhmm-call`'s
  integrated dedup follows the calling threshold.
- **Duplicates by deamination flavour.** DAF dedup groups reads by C→T vs G→A
  flavour instead of alignment orientation (more duplicates are marked; e.g.
  about 5.8% → 9.9% on one DddA dataset).
- **Python 3.10 to 3.14.** Numba, scikit-learn, joblib and threadpoolctl are core
  dependencies (`[consensus]` and `[numba]` remain as compatibility aliases
  that add nothing beyond the core install).
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
  tables are updated), Nanopore Hia5 reads called without `--seq` before
  3.0 (they were called as PacBio), and 2.x `fiberhmm-posteriors` output.
  `fiberhmm-check` lists these and the recommended re-runs (DAF duplicates,
  QC, custom tables, `tag-m5c`, recaller tiers) per file, with the command
  to use; it exits 3 when something needs re-running and 4 for calls it
  cannot verify (no FiberHMM provenance in the header).
- Re-calling with 3.0 also applies the new defaults above (ML 248 for Hia5
  Nanopore, primary and supplementary alignments, the TF decoder, nucleosome
  recall without a periodicity prior, DddA models, thinning and CpG-aware
  calling), so numbers move even where no fix applies. Most have an option to
  get the 2.x behaviour back (the TF decoder, nucleosome recall and the DddA
  tables do not); the
  2.x Nanopore Hia5 and DddB tables are kept under `fiberhmm/models/legacy/`.
- **Re-train** models built with 2.x `fiberhmm-train --base-model` on a base
  whose state 0 is the footprint, such as the Nanopore Hia5 model bundled in 2.x
  (`fiberhmm-check` does not flag these).
- Replace `fiberhmm-run` and the `python *.py` scripts with the `fiberhmm-*`
  commands.
- FiberBrowser 3.0 requires FiberHMM 3.x.

The full guide, with a table of every default change and how to reproduce
2.x numbers, is
[Upgrading from 2.x](https://fiberseq.github.io/FiberHMM/upgrading/).

### Known issues

- The `hia5_nanopore` QC profile has no state-aware reference (its source
  control is not available for recalibration): in-MSP and outside-MSP rates
  are reported but not graded, and the verdict uses the overall rate.

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

### Experimental previews (included)

These preview tools and options ship in the 3.0.0 package. Their outputs,
parameters and formats may change in later releases without a deprecation
period.

- Frozen lattice transfer streams BED evidence one window at a time and keeps
  only compact BAM export plans, avoiding all-window evidence accumulation.
  Scoring workers persist across windows, sharing imports and JIT warm-up.
- NFR accepts independent `--bed` windows with persistent workers (`--cores`,
  `--window-jobs`), preserving single-region outputs and BED ordering.
- Sampled-loader top-ups cache only native replay changes, once per molecule;
  decoded arrays and immutable interval calls are shared while collapse metadata
  remains isolated. This avoids copying whole evidence units every tranche.
  Refinements that load no new units reuse prepared sources while refreshing
  loader receipts, avoiding a second full evidence copy.
- Reuse lattice scoring configurations across validation folds and the final
  fit; count prediction-strength pairs from label frequencies and reuse
  molecule split assignments without changing scientific results.
- Molecule caps are off by default (discovery and quantification both 0 = all).
  For deep data with thousands of reads per site, 2000 is a suggested opt-in
  cap for each phase.

- Consensus, frozen transfer and NFR can now sample BAM read names before tag
  decoding and native replay when all active molecule budgets are finite.
  Stable per-window hash frontiers oversample, top up after eligibility filters,
  and cache decoded evidence across tranches. Loader receipts report
  seen/decoded/retained counts and shortages; zero budgets preserve the full
  loader path.
- Add configurable molecule depth budgets to lattice-recaller footprint classes
  and NFR accessibility: `--max-discovery-molecules` (0 = all, per dataset/chemical
  channel), `--max-quantification-molecules` (0, all), and `--sample-seed` (1).
  Either cap at 0 disables that sampling phase. Stable hash priorities balance
  datasets and DAF CT/GA channels across overlapping tiles. Quantification uses
  the fixed discovered catalogue and selected eligible denominators; manifests
  record counts, budgets and capping notes. Both caps at 0 retain legacy output
  bytes. The browser schema exposes the consensus controls in `recaller`; NFR
  uses the same field names in its parameter group. Historical fitted consensus
  engines retain their existing behavior.

- **EXPERIMENTAL: `fiberhmm-nfr`, NFR variants and element co-accessibility
  (`fiberhmm.inference.accessibility`).** A preview for testing in
  FiberBrowser; outputs, parameters and formats (schema
  `fiberhmm.accessibility.preview.v0`) may change without notice. For a region:
  per-read NFRs (gaps between consecutive >= 90-bp nucleosome calls; factor-sized
  protections inside do not split them), NFR variants discovered with the
  lattice recaller's recipe (prediction-strength k, held-out identity merges and
  support; `--stringency`, default 0.9), per-read membership, prevalence as a
  strict-to-EM range with conditional bootstrap intervals, Timer depth states
  (`--mode depth`), and element co-accessibility between variants and footprint
  classes of a `fiberhmm-consensus` run (`--classes`): spanning reads only,
  Timer's shared-opening exclusion, an exact test stratified by per-read
  openness x channel, the Mantel-Haenszel odds ratio and BH. Writes
  variants.tsv, configurations.tsv, molecules.tsv.gz, coaccess.tsv, combos.tsv,
  result.json and manifest.json; deterministic.
- **EXPERIMENTAL: analysis views of an NFR run** (schema
  `fiberhmm.accessibility.preview.v1`; a v0 result still validates, but a v0
  run folder has no `context.json.gz`, so `load_result` asks for a re-run
  (`NeedsRerun`) before these views).
  - The result now stores, per molecule, its span, its nucleosome calls within
    the window +- 1 kb, its openness span and its gap edge features. It also
    stores the read states each pair test used, and each NFR's frozen variant
    catalogue (in result.json). The TSV outputs are unchanged.
  - `fiberhmm.inference.accessibility.analysis` splits a pair's reads into
    A+B+ / A+B- / A-B+ / A-B- and not-informative reads. It uses the test's own
    eligibility (`coaccess.pair_eligibility`), so the four groups are the 2x2
    table. It also splits combination patterns, and gives per-stratum tables,
    the spacing between two openings on one molecule, and per-group
    accessibility profiles.
  - It gives variant mini profiles; opening widths; the left-edge x right-edge
    density; the -1 / +1 boundary nucleosomes; centre and coverage V-plots;
    and nucleosome phasing on member vs other reads.
  - It also computes the frozen-catalogue prevalence on any read group or
    another payload (`quantify_frozen`, `group_prevalence`, `transfer`).
- **EXPERIMENTAL: raw-rate NFR detection** (`fiberhmm-nfr`, default without
  `--nfr`). `--detect-method raw` calls positions with a >= 175-bp gap in at
  least `--min-fraction` (default 0.10) of callable reads, at least
  `--min-open-reads` (10) open reads and 20 callable reads; regions need 60 bp.
  Runs still split by opening membership. Background (window q20, floored at
  0.005), fold and binomial BH q are informative only. Unusable backgrounds
  remain null with an explanation. `--detect-method enrichment` retains the
  prior fold/q gates and 0.05 default share. `fixed` retains its strict >0.15
  rule; a saved `detect_threshold` alone still selects fixed.
- **Optional gap assignment**: `--clip-gaps-to-region` (default off) assigns
  each gap to the region with largest overlap, ties to the leftmost region,
  before discovery, quantification and overflow truncation. Observed edges
  and gap widths are preserved. Frozen catalogues store `assignment_regions`
  so transfer quantification follows the same rule.
- **Descriptive opportunity diagnostic**: `detection.opportunity_diagnostic`
  compares complete >=175-bp gaps outside NFRs (`background`) with those
  overlapping NFRs (`nfr`). Each group reports `gaps`,
  `median_opportunities_per_100bp` and `sparse_share` (fewer than
  `sparse_below=10` opportunities per gap). Uses native, assay/strand-specific
  `unit.positions`, including unmarked sites and respecting native context
  masks. Reports missing and partial gaps separately; absent summaries are
  null. This does not establish that background gaps arise by chance.
  `detection.enrichment_test` continues to indicate whether the test was
  computed; new `enrichment_gates` indicates whether it gates detection.
  No existing result fields were renamed.

## Earlier versions

Releases up to 2.16.8 have no entries here; they are tagged in git
(`v2.16.8`, `v2.16.7`, …; see the
[repository tags](https://github.com/fiberseq/FiberHMM/tags)).
