# Run everything

One command takes a sequencing run to a called BAM ready for FiberBrowser:
`fiberhmm-pipeline` basecalls raw Nanopore data (optional), aligns, calls
footprints with `fiberhmm-call` and runs QC. For many samples, the
[Snakemake template](#many-samples-with-snakemake) runs one pipeline per
sample.

Every run writes `OUTDIR/<sample>.fiberhmm.bam` (+ `.bai`), `qc/` and
`outputs.json`, which says what to open in FiberBrowser. Running the same
command again continues where an earlier run stopped.

## What you need

- FiberHMM: `pip install "fiberhmm[plots]"`.
- minimap2, as a program (`brew install minimap2`, `conda install -c bioconda
  minimap2`) or the `mappy` module (`pip install mappy`).
- For raw Nanopore data (POD5) only: Oxford Nanopore's
  [dorado](https://github.com/nanoporetech/dorado). FiberHMM does not bundle
  it (it is distributed by Oxford Nanopore under its own licence and needs a
  GPU for practical speed). Unpack the release for your system and put its
  `bin/` on `PATH`, or pass `--dorado /path/to/dorado`.

## Nanopore: POD5 to called BAM

```bash
# Fiber-seq (Hia5, m6A): dorado sup + 6mA, then align and call
fiberhmm-pipeline run1/pod5/ --reference hg38.fa --enzyme hia5 -o run1_out/

# DAF-seq (DddB): plain basecalling (deaminations are read from the sequence)
fiberhmm-pipeline run2/pod5/ --reference dm6.fa --enzyme dddb -o run2_out/
```

A POD5 file, or a folder of them (searched recursively), is basecalled
first with

```text
dorado basecaller sup <pod5> --models-directory ~/.fiberhmm/dorado_models [--modified-bases 6mA]
```

into `OUTDIR/<sample>.basecalled.bam`, an unaligned BAM that then goes
through the usual steps. Defaults follow `--enzyme`: `sup` with the 6mA
model for `hia5`, no modification model for `ddda`/`dddb`. Change them with:

| Option | Meaning |
|---|---|
| `--dorado-model` | `fast`/`hac`/`sup[@vX.Y.Z]`, a complex such as `sup,6mA`, or a model folder |
| `--dorado-modified-bases` | modification codes (`6mA`), or `none` |
| `--dorado-modbase-models` | modification model names or paths (instead of the codes) |
| `--dorado-device` | `auto` (default), `metal`, `cuda:all`, `cuda:0`, `cpu` |
| `--dorado-batchsize` | dorado `--batchsize` |
| `--dorado-models-dir` | where models are downloaded (default `~/.fiberhmm/dorado_models`) |
| `--dorado-args` | other dorado options, quoted as one string |

Basecalling resumes: if dorado stops (a cancelled job, a crash), the next
run keeps the reads already written and asks dorado to continue
(`--resume-from`); finished basecalling is never repeated. Only
`--redo basecall` basecalls again (`--redo all` redoes every step after it).
Current dorado versions read POD5 only; convert FAST5 with
`pod5 convert fast5` first.

## Unaligned BAM to called BAM

```bash
# A dorado BAM (Nanopore Fiber-seq or DAF-seq)
fiberhmm-pipeline calls.bam --reference hg38.fa --enzyme hia5 -o out/

# PacBio HiFi reads with m6A calls (jasmine / fibertools), several movies
fiberhmm-pipeline m1.hifi_reads.bam m2.hifi_reads.bam --reference hg38.fa --enzyme hia5 -o out/

# FASTQ (record the basecaller yourself: FASTQ has no header)
fiberhmm-pipeline reads.fastq.gz --reference construct.dna --enzyme dddb -o out/ \
    --basecaller-info "program=dorado version=0.9.6 basecall_model=dna_r10.4.1_e8.2_400bps_sup@v5.0.0" \
    --modbase-model none
```

An unaligned BAM keeps its provenance: each input's `@RG` lines, `@PG`
chain and `@CO` lines go into `<sample>.aligned.bam`, and from there into
the called BAM; minimap2 and `fiberhmm-pipeline` are chained after them
(`@PG PP`). Read-group IDs stay unique across inputs (an ID used by two
inputs with different lines gets `-2`, `-3`...; identical lines are merged)
and every read's `RG` tag names its group under the new ID. Reads from FASTQ
belong to the pipeline's own read group, named after the sample.

The basecaller, its version, the basecalling model and the modification
model(s) are read from the inputs (dorado's `@RG DS`, then its `@PG CL`, then
the per-read read-group names) and recorded in the `fiberhmm-pipeline` and
`fiberhmm-call` `@PG DS` (see [Header declarations](../reference/headers.md#basecaller-provenance))
and in `outputs.json` under `settings.basecaller`. When the inputs do not
record them (FASTQ), the run says so; `--basecaller-info` and
`--modbase-model` record them, and take precedence over the headers.

## Many samples with Snakemake

The template in
[`workflows/snakemake`](https://github.com/fiberseq/FiberHMM/tree/main/workflows/snakemake)
runs one `fiberhmm-pipeline` per row of a sample sheet:

```text
sample	reads	enzyme	reference	seq	extra
ont_fiberseq	/data/run1/pod5	hia5
pacbio	/data/m1.hifi_reads.bam,/data/m2.hifi_reads.bam	hia5		pacbio
daf_plasmid	/data/reads.fastq.gz	dddb	/data/construct.dna		--tracks
```

```bash
pip install snakemake            # Snakemake >= 8
snakemake -n                     # dry run: what would run
snakemake --cores 16             # this machine
snakemake --profile profiles/slurm   # SLURM (pip install snakemake-executor-plugin-slurm)
```

POD5 samples get the `basecall` resources from `config.yaml` (a GPU on
SLURM). Rerunning after a failure continues each sample where it stopped.
See the template's README for the columns and settings.
