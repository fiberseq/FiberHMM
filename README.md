# FiberHMM

Chromatin footprint calling on single DNA molecules from Fiber-seq (m6A) and
DAF-seq (deamination) data. For every read, FiberHMM calls nucleosomes,
transcription-factor and other sub-nucleosomal footprints, and the
methylase-sensitive patches (MSPs) between them, using sequence-context-aware
emission models; across molecules, it finds recurrent footprint classes and
measures how often each molecule carries them. Calls are written into the BAM
as fibertools-compatible `ns`/`nl`/`as`/`al` and Molecular-annotation `MA`/`AQ`
tags, readable by `ft`, FIRE and FiberBrowser.

**Documentation: [fiberseq.github.io/FiberHMM](https://fiberseq.github.io/FiberHMM/)**
(source in [`docs/`](https://github.com/fiberseq/FiberHMM/tree/main/docs)).

> **FiberHMM 3.0** is the third generation of the FiberHMM family, released
> with FiberBrowser 3.0.0. The emission tables have been updated, so re-run
> calls made with earlier versions. `fiberhmm-check <your outputs>` lists which
> files need re-running. See [Upgrading from 2.x](https://fiberseq.github.io/FiberHMM/upgrading/)
> and the [CHANGELOG](https://github.com/fiberseq/FiberHMM/blob/main/CHANGELOG.md).

## What's new in 3.0

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

## Install

```bash
pip install fiberhmm              # Python 3.10 to 3.14
pip install "fiberhmm[all]"       # plus QC figures (matplotlib) and HDF5 posteriors (h5py)
```

## Quick start

```bash
# DAF-seq amplicons / plasmids: reads + reference in, FiberBrowser-ready calls out
fiberhmm-pipeline reads.fastq.gz --reference plasmid.dna --enzyme dddb -o run/

# Aligned BAMs
fiberhmm-call -i aligned.bam -o calls.bam --enzyme hia5 --seq pacbio -c 8 --region-parallel   # Hia5 PacBio
fiberhmm-call -i aligned.bam -o calls.bam --enzyme hia5 --seq nanopore -c 8 --region-parallel # Hia5 Nanopore
fiberhmm-call -i aligned.bam -o calls.bam --enzyme dddb -c 8 --region-parallel                # DddB DAF-seq (DddA: --enzyme ddda)
fiberhmm-extract -i calls.bam -o tracks/                                                     # BED12 / bigBed tracks
```

A synthetic demo data set and a walk-through of every chemistry are in the
[Quick start](https://fiberseq.github.io/FiberHMM/getting-started/quickstart/).

## Documentation

- [Getting started](https://fiberseq.github.io/FiberHMM/getting-started/installation/):
  installation, quick start, choosing `--enzyme` and `--seq`
- [Concepts](https://fiberseq.github.io/FiberHMM/concepts/how-it-works/):
  how FiberHMM works, chemistries and platforms, coordinate frames, tags and scores
- [Workflows](https://fiberseq.github.io/FiberHMM/workflows/calling/):
  calling, re-calling, DAF-seq, plasmids, duplex pairing, tracks, posteriors,
  QC, footprint classes (`fiberhmm-consensus`), transfer, strand rescue,
  training
- [Reference](https://fiberseq.github.io/FiberHMM/reference/cli/):
  every command and option, BAM tags, header declarations, footprint-tag
  frames, checking outputs for re-runs (`fiberhmm-check`), bundled models,
  consensus output files, Python API, environment variables
- [Troubleshooting](https://fiberseq.github.io/FiberHMM/troubleshooting/)

## License

MIT; see [LICENSE](https://github.com/fiberseq/FiberHMM/blob/main/LICENSE). [Citing FiberHMM](https://fiberseq.github.io/FiberHMM/about/citing/).
