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

> **FiberHMM 3.1** adds `fiberhmm-footprint` (below) and lowers
> `fiberhmm-consensus` memory use.
>
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

## Motif footprints: `fiberhmm-footprint`

`fiberhmm-footprint` asks, for every motif instance and every molecule that
spans it, "is this motif occupied on this molecule?". It is the FiberHMM
analogue of fibertools `ft footprint`, but it scores FiberHMM's own calls. A
molecule counts as footprinted at a motif when:

1. the motif lies inside an MSP on that molecule (the molecule is eligible),
2. one FiberHMM footprint call's outer edges (from the `MA` edge bytes)
   contain the whole motif,
3. the call's width is within `[--min-width, --max-width]`, and
4. optionally, with `--require-unmodified`, no m6A falls inside the motif.

The width window is where the gain over ft comes from. Without it,
FiberHMM's calls score within about 0.01 AUROC of ft. Broad protections that are not
motif-sized make up most of the false positives, and the window removes them.
The default window, 0-74 bp, needs no labels: 74 bp is the 95th percentile of
footprint calls that overlap motifs. Where ChIP-seq peaks exist,
`fiberhmm-footprint calibrate` learns a window per factor. The learned window
usually scores better and carried over to cell types it was not learned on.
Calibrate prints a ready-to-paste `--min-width/--max-width` line and writes a
settings file for other cell types.

Input is a called BAM, or the call bigBeds from `fiberhmm-extract`. Motifs
come from a BED or a motif bigBed such as JASPAR's TFBS tracks (filter with
`--motif-name` and `--min-score`). Output is an ft-style per-motif table and,
optionally, per-molecule rows. It can also add `fp_<name>` / `fpneg_<name>`
MA layers to a BAM copy, which FiberBrowser shows as one layer per motif set,
or write per-read bigBed tracks. `fiberhmm-utils ma-list` and `ma-drop`
list and remove such layers.

```bash
# learn the width window on cell type A (ChIP-seq peaks as labels) ...
fiberhmm-footprint calibrate --bed ctcf_motifs.bed --bam A.calls.bam --chip A_ctcf_peaks.bed \
    --require-unmodified -o ctcf_A
#   use: fiberhmm-footprint --min-width 25 --max-width 55 --require-unmodified
# ... and apply it to cell type B, which has no ChIP-seq
fiberhmm-footprint --bam B.calls.bam --bed ctcf_motifs.bed --settings ctcf_A.settings.json \
    -o B_ctcf.tsv --out-bam B.ctcf.bam
```

`fiberhmm-footprint` is the simple, rule-based tool. `fiberhmm-consensus` is
the modelling one: it learns each locus's footprint classes and scores
molecules against them. Use footprint to measure occupancy of known motifs
across many sites, and consensus to discover and quantify the footprint
classes at loci of interest.

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
  QC, motif footprints (`fiberhmm-footprint`), footprint classes
  (`fiberhmm-consensus`), transfer, strand rescue, training
- [Reference](https://fiberseq.github.io/FiberHMM/reference/cli/):
  every command and option, BAM tags, header declarations, footprint-tag
  frames, checking outputs for re-runs (`fiberhmm-check`), bundled models,
  consensus output files, Python API, environment variables
- [Troubleshooting](https://fiberseq.github.io/FiberHMM/troubleshooting/)

## License

MIT; see [LICENSE](https://github.com/fiberseq/FiberHMM/blob/main/LICENSE). [Citing FiberHMM](https://fiberseq.github.io/FiberHMM/about/citing/).
