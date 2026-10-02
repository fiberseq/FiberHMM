# Quick start

`fiberhmm-call` is the command for almost every run: it calls nucleosomes,
MSPs and TF footprints in one pass and writes them into a new BAM. Models are
bundled, so you only choose the chemistry:

```bash
fiberhmm-call -i sample.bam -o calls.bam --enzyme hia5 --seq pacbio -c 8 --region-parallel
```

This page runs each chemistry end to end on a small synthetic data set, so
you can see what goes in and what comes out before you use your own data.

## Get the demo data

The generator needs only numpy and pysam, which FiberHMM already installed.
From a source checkout:

```bash
python docs/examples/make_demo_data.py demo
```

or, after `pip install fiberhmm`:

```bash
curl -O https://raw.githubusercontent.com/fiberseq/FiberHMM/v3.0.0/docs/examples/make_demo_data.py
python make_demo_data.py demo
```

It writes a 30 kb reference and six BAMs into `demo/`:

| File | Contents |
|---|---|
| `ref.fa`, `ref.fa.fai` | one contig, `chrDemo` |
| `hia5_pacbio.bam` | 300 Hia5 Fiber-seq reads with PacBio-style MM/ML (`A+a` and `T-a`) |
| `hia5_nanopore.bam` | 300 Hia5 reads with Nanopore-style MM/ML (`A+a` only) |
| `hia5_pacbio.unaligned.bam` | the PacBio reads as an unaligned BAM |
| `hia5_pacbio.naked.bam` | Hia5 reads of naked DNA (an accessible control for [training](../workflows/training.md)) |
| `dddb.bam` | 300 DddB DAF-seq reads, aligned with `MD` tags |
| `ddda.bam` | DddA DAF-seq reads with `MD` tags, including PCR duplicates and both strands of some molecules |

The molecules carry a phased nucleosome array and two planted TF sites
(24 bp at 10,040 and 16 bp at 10,110, 0-based). The data are synthetic: they
exercise the commands, and QC grades them against real reference data sets,
so expect QC to report WARN or FAIL on them.

All commands below run from the directory that contains `demo/`, and write
into `out/` (FiberHMM commands create missing output directories).

## Hia5 Fiber-seq, PacBio

```bash
fiberhmm-call -i demo/hia5_pacbio.bam -o out/pacbio.calls.bam \
    --enzyme hia5 --seq pacbio -c 2 --region-parallel
```

The banner states the resolved settings (abridged):

```text
  fiberhmm-call — fused apply + recall-tfs (region-parallel)
  apply model:  .../fiberhmm/models/hia5_pacbio.json
  mode=pacbio-fiber k=3 enzyme=hia5 prob-threshold=128 alignments=primary-supplementary
  min_llr=5.0 min_opps=3 unify_threshold=90 uplift=1.0
  nuc-recall-policy=conservative phase-nrl=193
...
  Total: 300 reads, 300 with footprints
```

Outputs:

```text
out/pacbio.calls.bam                 calls (sorted, indexed)
out/pacbio.calls.bam.bai
out/qc/pacbio.calls.qc.json          QC report (also .tsv; .png and .pdf with fiberhmm[plots])
```

Each called read now carries `ns`/`nl` (nucleosomes), `as`/`al` (MSPs) and an
`MA`/`AQ` pair that holds nucleosomes, MSPs and TF footprints with their
quality bytes:

```bash
samtools view out/pacbio.calls.bam | head -1 | tr '\t' '\n' | grep -E '^(MA|ns|as):' | cut -c1-90
```

```text
MA:Z:3744;nuc.QQQ:87-146,481-145,675-149,871-147,1063-156,1256-148,1444-147,1640-144,1836-
ns:B:I,86,480,674,870,1062,1255,1443,1639,1835,2231,2416,2608,3000,3183,3382,3587
as:B:I,0,232,625,823,1017,1218,1403,1590,1783,1986,2378,2564,2755,3144,3331,3529,3733
```

The header records what was run and on which chemistry:

```text
@CO	FIBERHMM-CHEMISTRY:v1:assay=fiber-seq;enzyme=hia5;platform=pacbio;mode=pacbio-fiber;model=hia5_pacbio
@CO	MA-TYPES:v1:nuc,msp,tf
```

[Annotations and scores](../concepts/annotations.md) explains the tags, and
[Coordinate frames](../concepts/coordinates.md) explains why intervals are in
the molecule's own orientation.

## Hia5 Fiber-seq, Nanopore

```bash
fiberhmm-call -i demo/hia5_nanopore.bam -o out/ont.calls.bam \
    --enzyme hia5 -c 2 --region-parallel
```

`--seq` was left out on purpose: FiberHMM detects the platform from the
input and says so.

```text
NOTE: --seq not given; using --seq nanopore (detected from MM specs of 200 read(s) (A+a only, no T-a)).
  mode=nanopore-fiber k=3 enzyme=hia5 prob-threshold=248 alignments=primary-supplementary
  nuc-recall-policy=topology phase-nrl=193
```

Two things differ from PacBio: m6A calls need ML ≥ 248 (the demo's
sub-threshold calls at ML 200 are ignored), and nucleosome recall uses the
single-strand-aware `topology` policy. See
[Choosing the chemistry](choosing-chemistry.md).

## DddB DAF-seq

```bash
fiberhmm-call -i demo/dddb.bam -o out/dddb.calls.bam \
    --enzyme dddb -c 2 --region-parallel
```

For a DAF enzyme on a file input, `fiberhmm-call` runs three extra stages
around the footprint call, and reports each:

```text
  --dedup: detecting PCR duplicates by deamination fingerprint (Jaccard >= 0.95, ends ±50 bp, mark/retain) BEFORE footprinting...
Clustering (Jaccard >= 0.95; ends ±50 bp): 300 reads -> 300 molecules | 0 duplicates (0.0%) | mean 1.00 copies/molecule [1s]
  automatic DAF SNP screen enabled: bounded preflight found genome coverage support (...)
  DAF SNP mask: 0 sites -> out/qc/dddb.calls.daf_snps.bed
...
  mode=daf k=3 enzyme=dddb prob-threshold=128 alignments=primary-supplementary
```

Deaminations are read from the `MD` tags (from `minimap2 --MD` or
`samtools calmd`). The run adds the SNP-screen outputs
(`out/qc/dddb.calls.daf_snps.{bed,vcf,json,amplicons.tsv}`) and a duplicate
report (`out/qc/dddb.calls.dedup.json`) to the QC directory. See
[DAF-seq](../workflows/daf-seq.md).

## DddA DAF-seq

```bash
fiberhmm-call -i demo/ddda.bam -o out/ddda.calls.bam \
    --enzyme ddda -c 2 --region-parallel
```

DddA uses separate models for the nucleosome HMM and for TF recall, the
phase-aware radial nucleosome caller, CpG-aware recall and keep-one masking of
adjacent targets; the banner shows them:

```text
  NOTE: DddA nucleosome recall uses phase-aware radial inference (bundled ddda_nuc_profile.json).
Clustering (Jaccard >= 0.95; ends ±50 bp): 377 reads -> 325 molecules | 52 duplicates (13.8%) | mean 1.16 copies/molecule [1s]
  apply model:  .../fiberhmm/models/ddda_nuc.json
  recall model: .../fiberhmm/models/ddda_TF.json
  nuc likelihood model: .../fiberhmm/models/ddda_nuc_refine.json
  cores=2 io-threads=8 cpg_mask=unmethylated-only daf_run_mask=>=2/keep-one
```

The 52 PCR copies planted in the demo are found and flagged (`0x400`, with
`di`/`ds` cluster tags); every read is kept.

## Unaligned input and pipes

`fiberhmm-call` also reads unaligned BAMs and stdin. Without
`--region-parallel` it streams:

```bash
fiberhmm-call -i demo/hia5_pacbio.unaligned.bam -o out/ubam.calls.bam \
    --enzyme hia5 --seq pacbio -c 2
```

```text
  NOTE: calling unmapped reads (unaligned input (no @SQ reference sequences)); pass --no-process-unmapped to pass them through instead.
```

Streaming to stdout lets you pipe into FIRE (`ft` from
[fibertools](https://github.com/fiberseq/fibertools-rs), installed separately):

```bash
fiberhmm-call -i demo/hia5_pacbio.bam -o - --enzyme hia5 --seq pacbio -c 2 \
    | ft fire - out/pacbio.fire.bam
```

QC is skipped for stdout output; run `fiberhmm-qc` on the saved BAM.

## Look at the results

```bash
# BED12 / bigBed tracks per feature type
fiberhmm-extract -i out/pacbio.calls.bam -o out/tracks -c 2

# QC for several BAMs, with a combined comparison
fiberhmm-qc -i out/pacbio.calls.bam out/dddb.calls.bam -o out/qc_compare

# Footprint classes at the planted TF sites
fiberhmm-consensus --bam out/pacbio.calls.bam --region chrDemo:9900-10250 \
    --cores 2 --output out/classes
```

`fiberhmm-extract` writes `out/tracks/pacbio.calls_{nucleosome,msp,tf,m6a}.bb`
(bigBed if UCSC `bedToBigBed` is installed, BED otherwise). The consensus run
finds the two planted sites:

```text
class_id   channel            start    end      status     molecules  prevalence
class_001  dataset_1::pooled  10041.0  10064.5  supported  176        0.5555
class_002  dataset_1::pooled  10110.0  10126.0  supported  176        0.4108
```

(columns selected from `out/classes/classes.tsv`; the planted occupancies were
0.7 and 0.5).

## Where to go next

- [Choosing the chemistry](choosing-chemistry.md): `--enzyme`, `--seq` and
  what is detected automatically.
- [Calling footprints](../workflows/calling.md): region-parallel versus
  streaming, filters, duplicates and SNPs, QC and every calling option.
- [How FiberHMM works](../concepts/how-it-works.md): the model behind the
  calls.
- [Footprint classes](../workflows/consensus.md): classes, prevalence tiers and
  per-molecule labels.
