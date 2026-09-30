# FiberHMM

FiberHMM calls chromatin footprints on single DNA molecules. From the
modification marks of a Fiber-seq (m6A) or DAF-seq (deamination) read it finds
the protected stretches (nucleosomes, and transcription-factor or Pol II
footprints) and the accessible stretches between them (methylase-sensitive
patches, MSPs). Its calls are written back into the BAM as standard tags that
[fibertools](https://github.com/fiberseq/fibertools-rs), FIRE and
[FiberBrowser](https://github.com/mtcicero26/FiberBrowser) read.

Across many molecules, FiberHMM also finds recurrent footprint classes at a
locus and measures how often each molecule carries each one.

!!! note "FiberHMM 3.0"
    3.0 is the third generation of the FiberHMM family and is released together
    with FiberBrowser 3.0. **Nanopore Hia5 users:** the Nanopore Hia5 emission
    table shipped in every 2.x release was context-swapped; re-run ONT Hia5
    calls made with 2.x. See [Upgrading from 2.x](upgrading.md).

## What it does

| Stage | Command | Page |
|---|---|---|
| Call nucleosomes, MSPs and TF footprints on every read | `fiberhmm-call` | [Calling footprints](workflows/calling.md) |
| Re-call TFs or nucleosomes on an already called BAM | `fiberhmm-recall-tfs`, `fiberhmm-recall-nucs` | [Re-calling](workflows/recalling.md) |
| DAF-seq extras: duplicates, SNPs, R/Y encoding, CpG-island methylation | `fiberhmm-dedup`, `-daf-snps`, `-daf-encode`, `-tag-m5c`, `-call-m5c` | [DAF-seq](workflows/daf-seq.md) |
| Pair the two strands of one scDAF molecule and call them jointly | `fiberhmm-pair` | [Duplex](workflows/duplex.md) |
| Check a run against reference data sets | `fiberhmm-qc` (also automatic after `fiberhmm-call`) | [Quality control](workflows/qc.md) |
| Export calls as BED12 / bigBed tracks | `fiberhmm-extract` | [Extracting tracks](workflows/extracting.md) |
| Find footprint classes across molecules and measure them per molecule | `fiberhmm-consensus` | [Footprint classes](workflows/consensus.md) |
| Measure frozen classes in new data or at other loci | `fiberhmm-transfer` | [Transferring classes](workflows/transfer.md) |
| Rescue footprints missed on one strand, using the other strand | `fiberhmm-strand-rescue` | [Strand rescue](workflows/strand-rescue.md) |
| Footprint population model from per-read TF calls | `fiberhmm-footprint-model` | [Footprint population model](workflows/footprint-model.md) |
| Per-position HMM posteriors | `fiberhmm-posteriors` | [Posteriors](workflows/posteriors.md) |
| Build a model for a new chemistry or condition | `fiberhmm-probs`, `fiberhmm-train`, `fiberhmm-utils` | [Training a model](workflows/training.md) |

## Supported data

| Chemistry | Platform | `fiberhmm-call` preset |
|---|---|---|
| Hia5 Fiber-seq (m6A) | PacBio | `--enzyme hia5 --seq pacbio` |
| Hia5 Fiber-seq (m6A) | Oxford Nanopore | `--enzyme hia5 --seq nanopore` |
| DddB DAF-seq (deamination) | any (usually Nanopore) | `--enzyme dddb` |
| DddA DAF-seq (deamination) | any (usually PacBio) | `--enzyme ddda` |

Models for these four are bundled. Other chemistries (EcoGII, GpC/CpG
methyltransferases) are development-only; see
[Chemistries and platforms](concepts/chemistries.md).

## Where to start

- New to FiberHMM: [Installation](getting-started/installation.md), then the
  [Quick start](getting-started/quickstart.md), which runs every chemistry on
  a small synthetic data set.
- Not sure which `--enzyme`/`--seq` to use:
  [Choosing the chemistry](getting-started/choosing-chemistry.md).
- Reading FiberHMM output in your own code: [BAM tags](reference/bam-tags.md)
  and [Python API](reference/python-api.md).
- Every option of every command: [Command-line reference](reference/cli.md).
- Something failed: [Troubleshooting](troubleshooting.md).
