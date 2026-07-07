# Multi-assay corpus & new FiberHMM modes

Goal: compile train/test single-molecule datasets across footprinting chemistries,
then add FiberHMM modes for the ones our long-read architecture can model.

Source survey: `../../` compass artifact (Ramani/Krebs/Timp/Altemose lineages).
Machine-readable dataset table: [`corpus_manifest.tsv`](corpus_manifest.tsv).

## Chemistry → FiberHMM mode map

| Chemistry | Enzyme | Mod / motif | Existing mode? | Work needed |
|---|---|---|---|---|
| m6A @ A, PacBio | EcoGII/Hia5 | 6mA / any A | **`pacbio-fiber`** | none (new model only) |
| m6A @ A, nanopore | EcoGII/Hia5 | 6mA / any A | **`nanopore-fiber`** | none (new model only) |
| 5mC @ GpC, nanopore | M.CviPI | 5mC / GpC | ✗ | **new `gpc` mode** (C-target, GpC-restricted, methylation=accessible) |
| 5mC @ CpG, nanopore | M.SssI | 5mC / CpG | ✗ | **new `cpg` mode** (yeast/Drosophila only; endogenous CpG confounds mammals) |
| m6A @ GATC, nanopore | Dam | 6mA / GATC | ✗ | **new `dam` mode** (A-target, GATC-restricted, sparse) |
| 5mC @ GpC/CpG, Illumina bisulfite | M.CviPI/M.SssI | 5mC, <=300bp reads | ✗ | future **short-fiber** class (relaxed spacing priors) — out of scope now |

The two m6A rows reuse existing emission machinery — only a trained `.json` model
is new. The cytosine/Dam rows need real code: a new target base + motif restriction
in `bam_reader.py` context encoding and `generate_probs.py`, mirroring how `daf`
mode already special-cases C/G. Polarity note: like Fiber-seq m6A (and unlike naive
intuition), **methylation marks ACCESSIBLE** in all MTase assays — same emission
polarity as existing modes, so no state-inversion needed.

## Format → trainable modBAM (the actual pipeline)

Every nanopore set is deposited as **raw signal** (fast5/pod5, `OXFORDNANOPORE_NATIVE`) —
no basecalled modBAM mirror. PacBio sets are subreads/CCS with kinetics. So:

- **nanopore m6A** (SMAC, SAM-seq, STAM, DiMeLo): `dorado basecall` with a 6mA model
  → `dorado aligner`/`minimap2` → modBAM (MM/ML). GPU: RTX 5090 present.
- **nanopore 5mC** (nanoNOMe, yeast M.SssI): `dorado basecall` 5mC(all-C) → align →
  restrict to GpC or CpG motif in our encoder. Endogenous CpG must be separated from
  exogenous GpC for nanoNOMe.
- **PacBio m6A** (SAMOSA, SAMOSA-Tag, RASAM): `ccs --hifi-kinetics` → `ft predict-m6a`
  (fibertools already installed) → `pbmm2` align → modBAM.

### Toolchain status (this machine)
Present: SRA-toolkit, samtools, minimap2, pbmm2, **fibertools (`ft`)**, pysam, `ccs-kinetics-bystrandify`, RTX 5090.
**Missing (installable):** `dorado` (ONT CDN reachable, 2.6 GB), `ccs`/`pbccs` (pbbioconda), `modkit`.

## Corpus layout (proposed)

```
<CORPUS_ROOT>/
  raw/<id>/            # downloaded fast5/pod5 or PacBio subreads (SUBSET, not full archive)
  modbam/<id>/         # basecalled+aligned modBAM (MM/ML) — the trainable artifact
  refs/                # genomes: hg38/T2T, mm10/39, sacCer3, TAIR10, dm6
  splits/<id>/         # train.bam / test.bam read-id lists
  models/<mode>/       # trained FiberHMM .json per chemistry
```
Disk is tight (~250–280 GB free/drive), full raw corpus is >1 TB → pull **per-dataset
subsets** sufficient for train (~500 reads default) + held-out test + controls, not mirrors.

## Pre-called shortcuts (no basecalling needed)

Verified deposits that are already methylation-called — skip dorado/ccs, just write a
format adapter into FiberHMM's per-molecule mod representation:

| Family | Source | Format | Controls? | Size | Adapter needed |
|---|---|---|---|---|---|
| SAMOSA (PacBio m6A, accessibility) | Zenodo 3834705 | `*_bingmm.pickle` = per-molecule binarized m6A matrix; `*_zmwinfo.pickle` | yes (`plusM` treated vs `neg`/`minusM` naked) | most <150 MB | pickle → mod-positions |
| DiMeLo (nanopore m6A, directed) | GSE208125 | `*.mod_mappings.sorted.bam` (MM/ML, Megalodon) | in-situ | Droso 3.2 GB; GM12878 54–64 GB | **none** (drop-in modBAM) |
| nanoNOMe (GpC/CpG 5mC) | Zenodo 3969567 | per-read `*_gpc_runs.txt.gz` / `*_cpg_calls.txt.gz` (region-restricted: TSS, CTCF) | yes (4-way) | 1–2 GB each | txt → mod-positions |

These cover 3 of 4 chemistry families (m6A-PacBio, m6A-nanopore, 5mC-GpC) without a
basecaller. DiMeLo modBAM validates the existing `nanopore-fiber` reader on non-Hia5
data immediately. SAMOSA pickles give PacBio-m6A accessibility + naked control in a
compact form — the fastest route to a real `pacbio-fiber` benchmark. Raw+dorado stays
the path only for genome-wide accessibility m6A (SMAC/SAM-seq) and genome-wide GpC.

## Priority order

- **P0 — reuse existing modes, validate the whole pipeline:**
  nanopore m6A (SMAC `PRJNA594057`, SAM-seq `PRJEB69301`) and PacBio m6A
  (SAMOSA-Tag `PRJNA863422`, RASAM `GSE245558`). Ramani HMM calls = benchmark.
- **P1 — first new mode (`gpc`):** nanoNOMe `PRJNA510783`.
- **P2 — `cpg` / repeats:** yeast single-fiber M.SssI (NAR 2024), dual-enzyme Biophys J 2025, STAM-seq.
- **P3 — directed / sparse (validation, not accessibility):** DiMeLo `PRJNA752170`, Nanopore-DamID `GSE203109`, in-vivo yeast Dam/M.SssI.
- **X — short-read bisulfite:** deferred to a future short-fiber architecture.

## Open verification items (chase from each paper's Data Availability)
- SAMOSA `GSE162410`: resolve the **PacBio** SRP (ENA surfaced ONT runs under the SuperSeries — the m6A PacBio subseries ID is unconfirmed).
- yeast single-fiber Fiber-Seq (NAR 2024 52(1):166) accession.
- in-vivo yeast M.SssI/Dam (NSMB 2025 32:247) accession.
- Nanopore-DamID `GSE203109` final peer-reviewed venue.
- SAMOSA-Tag: which open (non-dbGaP) samples carry a matched fully-methylated naked-DNA control.
