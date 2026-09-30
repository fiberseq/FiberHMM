# Installation and usage

## Install

```bash
pip install fiberhmm            # Python >= 3.10
pip install "fiberhmm[all]"     # plus matplotlib (QC figures) and h5py (HDF5 posteriors)
```

UCSC `bedToBigBed` is needed for bigBed tracks, and `samtools` is
recommended for speed. See
[Installation](https://fiberseq.github.io/FiberHMM/getting-started/installation/).

## Call footprints

`fiberhmm-call` runs the HMM, nucleosome recall and TF recall in one pass.
Pick the chemistry; the bundled model follows:

```bash
# Hia5 Fiber-seq, PacBio (sorted, indexed BAM with MM/ML)
fiberhmm-call -i aligned.bam -o calls.bam --enzyme hia5 --seq pacbio -c 8 --region-parallel

# Hia5 Fiber-seq, Nanopore (--seq is detected from the MM tags when omitted)
fiberhmm-call -i aligned.bam -o calls.bam --enzyme hia5 --seq nanopore -c 8 --region-parallel

# DAF-seq, DddB or DddA (aligned with MD tags, e.g. minimap2 --MD)
fiberhmm-call -i aligned.bam -o calls.bam --enzyme dddb -c 8 --region-parallel
fiberhmm-call -i aligned.bam -o calls.bam --enzyme ddda -c 8 --region-parallel

# Unaligned BAM or stdin: stream, optionally into FIRE (fibertools' ft)
fiberhmm-call -i reads.bam -o - --enzyme hia5 --seq pacbio -c 8 | ft fire - fire.bam
```

For DAF input, PCR-duplicate marking and SNP screening run automatically, and
every file output gets a bounded QC report in `qc/` next to the BAM.

## Look at the calls

```bash
fiberhmm-extract -i calls.bam -o tracks/                 # BED12 / bigBed per feature type
fiberhmm-qc -i calls.bam other.bam -o qc_compare/         # QC and a combined comparison
fiberhmm-consensus --bam calls.bam --region chr1:1000000-1000350 --output classes/
```

## Output tags

| Tag | Content |
|-----|---------|
| `ns` / `nl` | nucleosome starts / lengths |
| `as` / `al` | MSP starts / lengths |
| `nq` | nucleosome evidence (LLR ×10) |
| `MA` / `AQ` | nucleosomes (`nuc.QQQ`), MSPs (`msp.`) and TF footprints (`tf.QQQ`) with quality bytes |

Intervals are in the molecule's own orientation, like fibertools. TF calls
are in `MA`/`AQ` only; `--downstream-compat` also puts them into `ns`/`nl`.
See [BAM tags](https://fiberseq.github.io/FiberHMM/reference/bam-tags/).

## Train a custom model

```bash
fiberhmm-probs -a accessible_control.bam -u inaccessible_control.bam -o probs/sample --mode pacbio-fiber -k 3
fiberhmm-train -i sample.bam -p probs/sample/tables/sample_accessible_A_k3.tsv \
    probs/sample/tables/sample_inaccessible_A_k3.tsv -o model/ -k 3
fiberhmm-call -i sample.bam -o calls.bam -m model/best-model.json --region-parallel
```

See [Training a model](https://fiberseq.github.io/FiberHMM/workflows/training/)
and the full [command-line reference](https://fiberseq.github.io/FiberHMM/reference/cli/).
