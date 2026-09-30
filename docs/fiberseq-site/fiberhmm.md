# FiberHMM: chromatin footprint calling

[![GitHub](https://img.shields.io/github/v/release/fiberseq/FiberHMM?color=green)](https://github.com/fiberseq/FiberHMM)
[![PyPI](https://img.shields.io/pypi/v/fiberhmm)](https://pypi.org/project/fiberhmm/)

FiberHMM calls chromatin footprints on single molecules from Fiber-seq (m6A,
Hia5) and DAF-seq (deamination, DddA and DddB) data, on PacBio and Oxford
Nanopore reads. For each read it finds nucleosomes, transcription-factor and
other sub-nucleosomal footprints, and the methylase-sensitive patches (MSPs)
between them, and writes them back into the BAM as fibertools-compatible
`ns`/`nl`/`as`/`al` tags and Molecular-annotation `MA`/`AQ` tags that `ft`,
FIRE and FiberBrowser read.

Its emission probabilities are learned per sequence context, which corrects
the large context bias of the marking enzymes and makes footprints the size
of a transcription factor callable, not just nucleosomes. Across molecules,
FiberHMM also finds recurrent footprint classes at a locus and measures how
often each molecule carries them.

This chapter is a short overview:

- [Installation and usage](usage.md)
- [Pre-trained models](models.md)
- [How it works](methods.md)

The full documentation is at
[fiberseq.github.io/FiberHMM](https://fiberseq.github.io/FiberHMM/); the
source is on [GitHub](https://github.com/fiberseq/FiberHMM).

> **FiberHMM 3.0** corrects the Nanopore Hia5 emission table, which was
> context-swapped in every 2.x release. Re-run ONT Hia5 calls made with 2.x
> ([upgrading from 2.x](https://fiberseq.github.io/FiberHMM/upgrading/)).
