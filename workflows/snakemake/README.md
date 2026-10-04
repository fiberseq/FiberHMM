# FiberHMM Snakemake template

Runs one `fiberhmm-pipeline` per sample. The pipeline does the work
(basecalling POD5 with dorado when asked, alignment, calling, QC); this
workflow only fans it out over a sample sheet and gives each run resources.

## Files

| File | What it is |
|---|---|
| `Snakefile` | one rule, `fiberhmm_pipeline`, per sample |
| `config.yaml` | reference, output folder, threads, dorado settings, resources |
| `samples.tsv` | the sample sheet (tab-separated) |
| `profiles/local/config.yaml` | one machine (`--profile profiles/local`) |
| `profiles/slurm/config.yaml` | a SLURM cluster (Snakemake ≥ 8 executor plugin) |

## Use

1. Install FiberHMM and Snakemake (≥ 8) in one environment, e.g.
   `pip install fiberhmm snakemake`, and minimap2 (`conda install -c bioconda
   minimap2` or `pip install mappy`). For POD5 samples install dorado (not
   bundled with FiberHMM; https://github.com/nanoporetech/dorado).
2. Copy this folder next to your data and edit `samples.tsv` and
   `config.yaml` (at least `reference`).
3. Check what would run, then run:

   ```bash
   snakemake -n                         # dry run
   snakemake --cores 16                 # this machine
   snakemake --profile profiles/slurm   # SLURM (pip install snakemake-executor-plugin-slurm)
   ```

## Sample sheet

| Column | Required | Meaning |
|---|---|---|
| `sample` | yes | sample name (output files and read group) |
| `reads` | yes | one or more inputs, comma-separated: FASTQ, unaligned or aligned BAM, or a POD5 file/folder |
| `enzyme` | yes | `hia5` (Fiber-seq), `ddda` or `dddb` (DAF-seq) |
| `reference` | no | FASTA or plasmid map for this sample (default: `config.yaml`'s) |
| `seq` | no | `pacbio` or `nanopore` (default: detected from the reads) |
| `extra` | no | more `fiberhmm-pipeline` options for this sample |

A sample whose reads include POD5 is basecalled first (default: `sup` with
6mA for `hia5`, plain basecalling for `ddda`/`dddb`) and gets the
`basecall` resources from `config.yaml` (a GPU on SLURM via `slurm_extra`).

## Outputs

`results/<sample>/` holds what `fiberhmm-pipeline` writes: the called BAM,
QC, and `outputs.json` (what to open in FiberBrowser, and the basecaller
provenance under `settings.basecaller`). Logs: `results/logs/<sample>.log`
and `results/logs/<sample>.progress.jsonl`.

Rerunning after a failure or a cancelled job continues where each sample
stopped: the pipeline skips finished steps (including basecalling) and
resumes an interrupted dorado or fiberhmm-call run.
