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

> **FiberHMM 3.0.0** is the third generation of the FiberHMM family, released
> with FiberBrowser 3.0.0. **Nanopore Hia5 and DddB users:** the Nanopore Hia5
> and DddB emission tables shipped in 2.x releases were context-swapped; re-run
> those calls made with 2.x. `fiberhmm-check <your outputs>` lists which files
> need re-running. See [Upgrading from 2.x](https://fiberseq.github.io/FiberHMM/upgrading/)
> and the [CHANGELOG](https://github.com/fiberseq/FiberHMM/blob/main/CHANGELOG.md).

## Install

```bash
pip install fiberhmm              # Python >= 3.10
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
  calling, re-calling, DAF-seq, duplex pairing, tracks, QC, footprint classes
  (`fiberhmm-consensus`), transfer, strand rescue, training
- [Reference](https://fiberseq.github.io/FiberHMM/reference/cli/):
  every command and option, BAM tags, header declarations, bundled models,
  consensus output files, Python API
- [Troubleshooting](https://fiberseq.github.io/FiberHMM/troubleshooting/)

## License

MIT; see [LICENSE](https://github.com/fiberseq/FiberHMM/blob/main/LICENSE). [Citing FiberHMM](https://fiberseq.github.io/FiberHMM/about/citing/).
