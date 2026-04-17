# FiberHMM

Hidden Markov Model toolkit for calling chromatin footprints from Fiber-seq, DAF-seq, and other single-molecule footprinting data.

FiberHMM identifies protected regions (footprints) and accessible regions (methylase-sensitive patches, MSPs) from single-molecule DNA modification data, including m6A methylation (fiber-seq) and deamination marks (DAF-seq).

---

> ## ⚠️ Upgrade to v2.9.1 or later
>
> **Versions 2.6 – 2.8.2** had a parser bug that placed m6A/m5C modifications at the wrong query positions on reverse-aligned reads (positions shifted by ~24 bp median). This shifted footprint/MSP/TF calls on reverse reads relative to where they should be. Forward reads (~50% of data) unaffected. **DAF with IUPAC (`R`/`Y`) encoding unaffected** at all versions — it bypasses the buggy code path. **v2.9.0** fixed the parser, validated byte-for-byte against `pysam.modified_bases` on both strands across 500+ real reads.
>
> **v2.9.1 adds two fixes on top of 2.9.0:**
> 1. **`ft validate` compatibility** — `fiberhmm-apply`/`fiberhmm-call` now drop stale `nq`/`aq` quality arrays when refreshing `ns/nl`/`as/al` on BAMs that already carry ft/modkit footprint tags. Without this, fibertools-rs asserts on `len(nq) != len(ns)`.
> 2. **Refreshed Hia5 Nanopore emissions** — the bundled `hia5_nanopore.json` model was refit from naked/untreated controls with the fixed parser. Accessible-state per-hexamer P(meth) dropped from ~0.61 → ~0.32 (the old emissions were systematically over-calling accessibility on Nanopore Hia5). Transitions and startprob preserved.
>
> **Action:** `pip install --upgrade fiberhmm`. Re-run samples if you need position-correct reverse-read calls or if your input BAM was pre-tagged by ft/modkit. See the [v2.9.1 release notes](https://github.com/fiberseq/FiberHMM/releases/tag/v2.9.1) for details.

---

## Key Features

- **`fiberhmm-call`** — **recommended**, runs nucleosome/MSP HMM + TF recall fused in one process with region-parallel scaling. Coordinate-sorted input → coordinate-sorted + indexed output, no sort pass needed.
- **No genome context files** — hexamer context computed directly from read sequences
- **Fibertools-compatible output** — tagged BAM with `ns`/`nl`/`as`/`al` legacy tags AND `MA`/`AQ` [Molecular-annotation spec](https://github.com/fiberseq/Molecular-annotation-spec) tags with `tf+QQQ` TF scoring
- **Native HMM implementation** — no hmmlearn dependency; Numba JIT enabled by default for ~10× speedup
- **Multi-platform** — PacBio fiber-seq, Nanopore fiber-seq, DAF-seq (DddB, DddA)
- **Region-parallel** — near-linear scaling with `--cores`, up to chromosome count
- **Legacy model support** — loads old hmmlearn-trained pickle/NPZ models seamlessly

## Installation

### Using pip

```bash
pip install fiberhmm
```

### From source

```bash
git clone https://github.com/fiberseq/FiberHMM.git
cd FiberHMM
pip install -e .
```

### Optional dependencies

```bash
pip install numba        # ~10x faster HMM computation
pip install matplotlib   # --stats visualization
pip install h5py         # HDF5 posteriors export
```

For bigBed output, install [`bedToBigBed`](https://hgdownload.soe.ucsc.edu/admin/exe/) from UCSC tools.

## Quick Start

**`fiberhmm-call` is the recommended entry point.** It runs nucleosome/MSP + TF calling fused in one process. Two modes depending on input:

### Sorted + indexed BAM → `--region-parallel` (fastest)

Near-linear scaling with `--cores` (up to chromosome count), preserves coordinate sort, no sort pass needed.

```bash
# Hia5 PacBio on a sorted+indexed BAM
fiberhmm-call -i experiment.bam -o recalled.bam \
              --enzyme hia5 --seq pacbio \
              -c 8 --io-threads 16 \
              --region-parallel --skip-scaffolds

# DddB DAF-seq
fiberhmm-call -i aligned.bam -o recalled.bam \
              --enzyme dddb \
              -c 8 --io-threads 16 \
              --region-parallel --skip-scaffolds

# If you also want FIRE scoring: second step after, on the sorted+indexed output
ft fire recalled.bam final.bam
```

### Unaligned / unsorted BAM or stdin → streaming mode, pipe to FIRE

For unaligned basecaller output, stdin, or any BAM without an index, drop `--region-parallel`. Streaming mode supports stdin (`-i -`) and stdout (`-o -`) and pipes cleanly into `ft fire`:

```bash
# PacBio fiber-seq, unaligned, all the way to FIRE-scored output in one pipeline
fiberhmm-call -i unaligned.bam -o - --enzyme hia5 --seq pacbio \
              -c 8 --io-threads 16 \
    | ft fire - final.bam

# DddB DAF from a live basecaller stream
some_basecaller ... | \
    fiberhmm-call -i - -o - --enzyme dddb -c 8 --io-threads 16 \
    | ft fire - final.bam
```

This is the intended path for workflows where the input isn't coordinate-sorted (e.g., direct-from-basecaller pipelines). Fusion still eliminates the apply → recall-tfs serialization, so the only remaining pipe is `fiberhmm-call → ft fire`.

### DddA amplicons

```bash
fiberhmm-call -i aligned.bam -o recalled.bam --enzyme ddda \
              -c 8 --io-threads 16 --region-parallel
# or streaming to fire:
fiberhmm-call -i aligned.bam -o - --enzyme ddda -c 8 | ft fire - final.bam
```

Pre-trained models are **bundled with the package** — `--enzyme` selects automatically, `-m` only needed for custom models.

### Extract BED12 / bigBed

```bash
fiberhmm-extract -i recalled.bam --footprint --msp --tf --bigbed
```

---

### Which tool do I run?

| Situation | Command |
|-----------|---------|
| **Default: full pipeline on a sorted+indexed BAM** | `fiberhmm-call --region-parallel` |
| Unaligned / unsorted BAM, or reading from stdin | `fiberhmm-call` (streaming mode, no `--region-parallel`) |
| Only want nucleosome/MSP calls, no TF recall | `fiberhmm-apply` |
| Already have apply-tagged BAM, want to add TF calls | `fiberhmm-recall-tfs` |
| DddB/DddA that hasn't been IUPAC-encoded yet | `fiberhmm-daf-encode \| fiberhmm-call` |

> **Note:** `fiberhmm-run` was removed in v2.8.0 — it ran apply + recall + fire as separate subprocess stages connected by pipes (slow, per-read BAM serialization at every hop). `fiberhmm-call` fuses those Python stages in-process and is 2–9× faster. If you had scripts using `fiberhmm-run`, replace with `fiberhmm-call [| ft fire]`.

### DddA note — two-model workflow is still required

DddA uses distinct models for nucleosomes (`ddda_nuc.json`) and TF recall (`ddda_TF.json`) — `fiberhmm-call --enzyme ddda` handles this automatically. If you want to run them separately (e.g., for QC), use `fiberhmm-apply --enzyme ddda` followed by `fiberhmm-recall-tfs --enzyme ddda`. Without the recall pass, sub-nucleosomal TF/Pol II calls are not emitted.

### If your BAM is missing MD tags (DAF only)

```bash
samtools calmd -b aligned.bam ref.fa | fiberhmm-daf-encode -i - -o - --enzyme dddb -
fiberhmm-call -i encoded.bam -o recalled.bam --enzyme dddb --region-parallel
```

Or pass the reference FASTA directly to `fiberhmm-daf-encode --reference ref.fa` (slower but single-pass).

### Extract to BED12/bigBed

```bash
fiberhmm-extract -i output/experiment_footprints.bam
```

### TF consensus size BED (EXPERIMENTAL — untested)

> ⚠️ **This tool is experimental and largely untested.** Output format and
> defaults may change without notice. Do not use in production pipelines or
> cite results from it without independent validation.

`fiberhmm-consensus-tfs` sweeps a recalled BAM and outputs one line per TF
locus with consensus footprint size and single-molecule occupancy statistics.
Designed to feed into motif discovery tools (MEME) and occupancy analyses.

```bash
fiberhmm-consensus-tfs -i recalled.bam -o consensus_tfs.bed --min-tq 50
```

Output columns: `chr start end consensus_len MAD read_count spanning_reads fwd_reads rev_reads`

Single-molecule occupancy = `read_count / spanning_reads`. For DAF-seq,
`fwd_reads` / `rev_reads` expose strand bias at C-poor motifs.

## Pre-trained Models

Pre-trained models are **bundled with the pip package** — no separate download.
Select the right one with `--enzyme` (+ `--seq` for Hia5); `-m` is only needed for custom models.

| Model | `--enzyme` | `--seq` | Enzyme | Platform | Mode | Used by |
|-------|-----------|---------|--------|----------|------|---------|
| `hia5_pacbio.json` | `hia5` | `pacbio` | Hia5 (m6A) | PacBio | `pacbio-fiber` | `fiberhmm-apply` / `fiberhmm-recall-tfs` |
| `hia5_nanopore.json` | `hia5` | `nanopore` | Hia5 (m6A) | Nanopore | `nanopore-fiber` | `fiberhmm-apply` / `fiberhmm-recall-tfs` |
| `ddda_nuc.json` | `ddda` | — | DddA (deamination) | — | `daf` | `fiberhmm-apply` — **nucleosomes only** |
| `ddda_TF.json` | `ddda` | — | DddA (deamination) | — | `daf` | `fiberhmm-recall-tfs` — **REQUIRED 2nd pass** |
| `dddb_nanopore.json` | `dddb` | — | DddB (deamination) | Nanopore | `daf` | `fiberhmm-apply` / `fiberhmm-recall-tfs` |

Older models are in `models/legacy/` (reproducibility only). Custom models can always be specified with `-m /path/to/model.json`.

### DddA workflow -- two passes, two models

DddA requires a two-step workflow to get clean nucleosomes AND well-scored
TF/Pol II footprints:

1. **`ddda_nuc.json`** -- HMM nucleosome caller (`fiberhmm-apply`). Uses
   DddB-Nanopore transitions plus biophysics-derived emissions (linker
   P(meth)=0.5 from DddA's kinetic limit; nucleosome P(meth | ctx) =
   per-context FP rate + 0.05 breathing). Deliberately tuned to call
   only nucleosomes cleanly.

2. **`ddda_TF.json`** -- LLR TF recaller (`fiberhmm-recall-tfs`). The
   DddB-Nanopore emission table with a baked-in 2.0× efficiency uplift
   to match DddA's higher per-position deamination rate. **Without this
   2nd pass, sub-nucleosomal calls (TFs, Pol II) are not emitted.**

```bash
# Step 1: nucleosomes (DddA)
fiberhmm-apply -i tagged.bam --enzyme ddda -o out/ --mode daf

# Step 2: TF/Pol II recall (REQUIRED for DddA)
fiberhmm-recall-tfs -i out/tagged_footprints.bam -o out/recalled.bam \
                    --enzyme ddda
```

For Hia5 and DddB, the single trained model already captures both nucs
and small footprints, so the recall pass is optional refinement rather
than required.

## Analysis Modes

| Mode | Flag | Description | Target bases |
|------|------|-------------|--------------|
| **PacBio fiber-seq** | `--mode pacbio-fiber` | Default. m6A at A and T (both strands) | A, T (with RC) |
| **Nanopore fiber-seq** | `--mode nanopore-fiber` | m6A at A only (single strand) | A only |
| **DAF-seq** | `--mode daf` | Deamination at C/G (strand-specific) | C or G |

**Note on mode selection:** The `pacbio-fiber` vs `nanopore-fiber` distinction only matters for Hia5 (m6A), where PacBio detects modifications on both strands while Nanopore detects only one. For deaminase-based methods (DddA, DddB), `--mode daf` is always used regardless of sequencing platform -- the chemistry is inherently strand-specific.

## CLI Tools Reference

### fiberhmm-daf-encode

Preprocess plain aligned DAF-seq BAMs for FiberHMM. Identifies C→T and G→A deamination mismatches via the MD tag, encodes them as IUPAC Y/R in the query sequence, and adds `st:Z` strand tags.

```bash
fiberhmm-daf-encode -i aligned.bam -o encoded.bam
```

| Flag | Default | Description |
|------|---------|-------------|
| `-i/--input` | required | Input BAM, or `-` for stdin |
| `-o/--output` | required | Output BAM, or `-` for stdout |
| `--reference` | none | Reference FASTA (fallback if MD tag missing) |
| `-q/--min-mapq` | 20 | Min mapping quality |
| `--min-read-length` | 1000 | Min aligned read length (bp) |
| `--io-threads` | 4 | htslib I/O threads |
| `--strand` | auto | Force strand: `CT`, `GA`, or `auto` (per-read consensus) |

The encoder determines conversion strand per read by counting which mismatch type (C→T vs G→A) dominates. Reads with no mismatches or equal counts are skipped. All C→T or G→A mismatches are encoded (the HMM handles noise via emission probabilities). Existing tags (including MM/ML for dual-labeling) are preserved.

### fiberhmm-call

**Recommended one-command pipeline.** Runs nucleosome/MSP HMM + TF recall fused in a single Python process. Two modes:

| Mode | When to use | Flag |
|------|-------------|------|
| **Region-parallel** | Coordinate-sorted + indexed input BAM. Fastest — near-linear scaling up to chromosome count. Writes sorted+indexed output directly. | `--region-parallel` |
| **Streaming** | Unaligned BAM, unsorted BAM, or stdin (`-i -`). Supports piping to stdout (`-o -`) for clean composition with `ft fire`. | *(default, no flag)* |

```bash
# Sorted+indexed Hia5 PacBio — region-parallel (fastest)
fiberhmm-call -i sorted.bam -o out.bam --enzyme hia5 --seq pacbio \
              -c 8 --io-threads 16 --region-parallel --skip-scaffolds

# Streaming unaligned BAM → pipe into FIRE
fiberhmm-call -i unaligned.bam -o - --enzyme hia5 --seq pacbio \
              -c 8 --io-threads 16 \
    | ft fire - final.bam

# DddB DAF-seq (streaming from basecaller into FIRE)
some_basecaller ... | \
    fiberhmm-call -i - -o - --enzyme dddb -c 8 --io-threads 16 \
    | ft fire - final.bam
```

Key flags:

| Flag | Default | Description |
|------|---------|-------------|
| `-i/--input` | required | Input BAM or `-` for stdin. |
| `-o/--output` | required | Output BAM or `-` for stdout. |
| `--enzyme` | — | `hia5`, `dddb`, or `ddda` (auto-selects bundled model). |
| `--seq` | — | `pacbio` or `nanopore` (required for Hia5). |
| `-c/--cores` | 4 | Worker processes. |
| `--io-threads` | 8 | htslib I/O threads per stage. |
| `--region-parallel` | off | Per-region worker pool (requires sorted+indexed input). |
| `--skip-scaffolds` | off | Drop small scaffolds in region-parallel mode. |
| `--chroms chr1 chr2 …` | all | Restrict to specific chromosomes (region-parallel). |
| `--min-llr` | enzyme preset | Override TF LLR threshold. |
| `--no-legacy-tags` | off | Emit only MA/AQ spec tags, skip `ns/nl/as/al`. |
| `--downstream-compat` | off | Write TF calls into legacy `ns/nl` (skip MA/AQ). |

FIRE scoring: `ft fire` is a separate Rust binary from fibertools-rs; pipe `fiberhmm-call -o -` into it directly, or run as a second step on the region-parallel output.

> **`fiberhmm-run` has been removed** (v2.8.0). Running it now prints a migration message pointing to `fiberhmm-call`. The old tool chained apply + recall + fire as separate subprocess stages connected by pipes, which was much slower than the fused path.

### fiberhmm-apply

Apply a trained HMM to call footprints. Uses a streaming pipeline that scales with `-c` cores and supports stdin/stdout piping.

```bash
# Bundled model (recommended)
fiberhmm-apply -i experiment.bam --enzyme hia5 --seq pacbio -o output/ -c 8

# Custom model override
fiberhmm-apply -i experiment.bam -m custom.json -o output/ -c 8

# Stdin/stdout piping
fiberhmm-apply -i - --enzyme dddb -o output/
```

| Flag | Default | Description |
|------|---------|-------------|
| `-i/--input` | required | Input BAM, or `-` for stdin |
| `-m/--model` | optional | Custom HMM model (.json, .npz, or .pickle). If omitted, uses the bundled model for `--enzyme`/`--seq`. |
| `--enzyme` | optional | Select bundled model: `hia5`, `dddb`, or `ddda`. Required unless `-m` is given. |
| `--seq` | optional | Sequencing platform: `pacbio` or `nanopore`. Required for `--enzyme hia5`; ignored for dddb/ddda. |
| `-o/--outdir` | required | Output directory, or `-` for stdout BAM |
| `--mode` | from model | Analysis mode (see table above) |
| `-c/--cores` | 1 | CPU cores (0 = auto-detect) |
| `--io-threads` | 4 | htslib I/O threads |
| `-q/--min-mapq` | 20 | Min mapping quality |
| `--min-read-length` | 1000 | Min aligned read length (bp) |
| `-e/--edge-trim` | 10 | Edge masking (bp) |
| `--scores` | false | Compute per-footprint confidence scores |
| `--skip-scaffolds` | false | Skip scaffold/contig chromosomes |
| `--chroms` | all | Process only these chromosomes |
| `--msp-min-size` | 60 | Minimum MSP region size (bp) |
| `--primary` | false | Only process primary alignments |

#### Read Filtering

By default, reads are skipped (written unchanged without footprint tags) when:

| Reason | Default | Override |
|--------|---------|----------|
| **MAPQ too low** | < 20 | `--min-mapq 0` |
| **Aligned length too short** | < 1000 bp | `--min-read-length 0` |
| **No MM/ML modification tags** | -- | Cannot override (no data to process) |
| **Unmapped** | -- | Cannot override |
| **No footprints detected** | HMM found nothing | Cannot override |

### fiberhmm-probs

Generate emission probability tables from accessible and inaccessible control BAMs.

```bash
fiberhmm-probs \
    -a accessible_control.bam \
    -u inaccessible_control.bam \
    -o probs/ \
    --mode pacbio-fiber \
    -k 3 4 5 6 \
    --stats
```

### fiberhmm-train

Train the HMM on BAM data using precomputed emission probabilities.

```bash
fiberhmm-train \
    -i sample.bam \
    -p probs/tables/accessible_A_k3.tsv probs/tables/inaccessible_A_k3.tsv \
    -o models/ \
    -k 3 \
    --stats
```

Output:
```
models/
├── best-model.json      # Primary format (recommended)
├── best-model.npz       # NumPy format
├── all_models.json      # All training iterations
├── training-reads.tsv   # Read IDs used for training
├── config.json          # Training parameters
└── plots/               # (with --stats)
```

### fiberhmm-extract

Extract footprint/MSP/m6A/m5C features from tagged BAMs to BED12/bigBed.

```bash
fiberhmm-extract -i output/sample_footprints.bam -o output/ -c 8

# Extract only footprints
fiberhmm-extract -i output/sample_footprints.bam --footprint

# Keep BED files alongside bigBed
fiberhmm-extract -i output/sample_footprints.bam --keep-bed
```

### fiberhmm-recall-tfs (BETA)

> **Beta feature.** First shipped in fiberhmm 2.6.0. Algorithm and tag
> schema are stable; defaults are validated on Hia5 PacBio, DddB DAF,
> and DddA amplicons.
>
> Tag output follows the [fiberseq Molecular-annotation
> spec](https://github.com/fiberseq/Molecular-annotation-spec). FiberBrowser
> support for the `tf+QQQ` annotation type is shipping in the next
> release. File issues at
> https://github.com/fiberseq/FiberHMM/issues

LLR-based TF footprint recaller. Runs as an optional second pass on a BAM
already tagged by `fiberhmm-apply`. The HMM is excellent at calling
nucleosomes (≥90 bp protected stretches), but its global transition
penalty leaves smaller TF/Pol II footprints scored only as part of the
MSP/short-nuc tracks. The recaller scans those regions with a per-context
log-likelihood ratio test using the **same emission table the HMM was
trained with**, and emits sub-nucleosomal calls with proper statistical
scoring + boundary-ambiguity quality bytes.

For DddA users this pass is **required** (not optional): the shipped
`ddda_nuc.json` model deliberately does not emit TF calls. Running
`fiberhmm-apply` with any DddA model prints a reminder about this step.

```bash
# Bundled model (recommended)
fiberhmm-recall-tfs -i tagged.bam -o recalled.bam --enzyme hia5 --seq pacbio

# Custom model override
fiberhmm-recall-tfs -i tagged.bam -o recalled.bam -m custom.json --enzyme hia5
```

| Flag | Default | Description |
|------|---------|-------------|
| `-i/--in-bam` | required | Input BAM tagged by `fiberhmm-apply` (carries `ns`/`nl`/`as`/`al`). Use `-` for stdin. |
| `-o/--out-bam` | required | Output BAM with `MA`/`AQ` tags + refreshed legacy tags. Use `-` for stdout (pipe-friendly). |
| `-m/--model` | optional | Custom FiberHMM model JSON. If omitted, uses the bundled model for `--enzyme`/`--seq`. |
| `--enzyme` | optional | Select bundled model + tuned preset: `hia5`, `dddb`, or `ddda`. Required unless `-m` is given. Sets `--min-llr` defaults. |
| `--seq` | optional | Sequencing platform: `pacbio` or `nanopore`. Required for `--enzyme hia5`; ignored for dddb/ddda. |
| `--min-llr` | enzyme preset | Min cumulative LLR (nats) per call |
| `--min-opps` | 3 | Min informative target positions per call |
| `--emission-uplift` | 1.0 | Power transform on emission probabilities. Rarely needed -- use a pre-uplifted model file (e.g. `ddda_TF.json`) instead. |
| `--unify-threshold` | 90 | v2 footprints with `nl < this` may be demoted to `tf+`; ≥ this stay as nucleosomes |
| `--no-legacy-tags` | off | Skip refreshed `ns/nl/as/al` -- emit only `MA`/`AQ` (spec mode) |
| `--downstream-compat` | off | **Downstream-compat mode**: write TF calls into legacy `ns/nl` alongside nucleosomes; skip `MA`/`AQ` entirely. Use for tools that don't speak the Molecular-annotation spec. |
| `-c/--cores` | 1 | Worker processes. 0 = auto-detect |
| `--chunk-size` | 256 | Reads per worker chunk |
| `--io-threads` | 4 | htslib BAM compression threads |
| `--mode` | from model | Override observation mode (`pacbio-fiber` / `nanopore-fiber` / `daf`) |
| `--context-size` | from model | Override context size (default: read from model JSON) |
| `--max-reads` | 0 | 0 = no limit |

`fiberhmm-recall-tfs` JIT-compiles the Kadane scoring loop via Numba when
available (`pip install numba`) and parallelizes per-read work across
`--cores` workers. Typical throughput: ~5k reads/sec single-core with
JIT; ~15k reads/sec with 4 cores (scaling is I/O bound above that).

#### Output modes: spec (default) vs downstream-compat

The recaller supports two mutually-exclusive output modes. Pick based on
what your downstream tooling can read.

**Spec mode (default)** -- write `MA`/`AQ` tags per the
[fiberseq Molecular-annotation
spec](https://github.com/fiberseq/Molecular-annotation-spec):

```bash
fiberhmm-recall-tfs -i tagged.bam -o recalled.bam --enzyme hia5 --seq pacbio
```

- `MA`/`AQ` tags carry `nuc+Q`, `msp+`, `tf+QQQ` annotations with full
  LLR + edge-ambiguity scoring.
- Legacy `ns`/`nl` is also refreshed, but contains **nucleosomes only** --
  TF calls live exclusively in `MA`/`AQ`.
- Required tooling: an MA/AQ-aware consumer (FiberBrowser ≥ the release
  that ships alongside fiberhmm 2.6.0; future fibertools-rs). Tools that
  only read `ns`/`nl` will NOT see TF calls in this mode -- they appear
  blind to the sub-nucleosomal track.
- Recommended if you are the primary author of your downstream pipeline
  and can update the reader.

**Downstream-compat mode** -- put TF calls into legacy `ns`/`nl` alongside
nucleosomes, no `MA`/`AQ` written:

```bash
fiberhmm-recall-tfs -i tagged.bam -o recalled.bam \
                    --enzyme hia5 --seq pacbio --downstream-compat
```

- Legacy `ns`/`nl` contains **all footprints** (nucleosomes + TFs), sorted
  by start position. Entries < `--unify-threshold` (default 90 bp) are
  the TF calls; entries ≥ that threshold are nucleosomes. Downstream
  tools filter by size.
- `MA`/`AQ` tags are NOT written. Any pre-existing `MA`/`AQ` on the input
  is stripped so consumers that check both don't see a stale view.
- Per-TF quality (tq, el, er) is **lost** in this mode -- only positions
  and lengths are preserved.
- Recommended when your downstream pipeline (fibertools-rs, custom
  scripts, older FiberBrowser) reads only `ns`/`nl`.

The runtime banner makes the current mode explicit. Switching modes is a
pure re-run of the recaller on the same HMM-tagged input BAM.

#### Per-enzyme presets

The `--enzyme` flag picks tuned defaults validated on the FiberHMM
benchmark BAMs (Hia5 PacBio embryo, DddB DAF time-course, DddA amplicons):

| `--enzyme` | `--seq` needed? | `--min-llr` | Bundled model | Why |
|---|---|---|---|---|
| `hia5` | yes (`pacbio` or `nanopore`) | 5.0 | `hia5_pacbio.json` / `hia5_nanopore.json` | Trained Hia5 model is already calibrated |
| `dddb` | no | 4.0 | `dddb_nanopore.json` | DAF uses one strand only → ~3× sparser per-position evidence; lower threshold |
| `ddda` | no | 5.0 | `ddda_TF.json` (recall) / `ddda_nuc.json` (apply) | DddB-Nanopore emissions with a baked-in 2.0× efficiency uplift for DddA |

Override `--min-llr` if your data needs different calibration.
For cross-enzyme experimentation, `--emission-uplift` applies a power
transform to the loaded model's emissions at runtime (default 1.0).

#### MA/AQ tag schema

The recaller writes one MA tag and one AQ tag per processed read, per the
[fiberseq Molecular-annotation
spec](https://github.com/fiberseq/Molecular-annotation-spec).

```
MA:Z:<read_length>;nuc+Q:s1-l1,s2-l2,...;msp+:s1-l1,...;tf+QQQ:s1-l1,...
AQ:B:C: nq, nq, ..., tq, el, er, tq, el, er, ...
```

Coordinates are 1-based (per the spec); internal storage stays 0-based.

| Annotation | Quality bytes | Meaning |
|---|---|---|
| `nuc+Q` | `nq` | Nucleosomes (`nl ≥ unify_threshold` or v2 short-nucs the recaller did not match). `nq` carries v2's posterior mean (0 sentinel for unverified entries). |
| `msp+` | none | Methylase-sensitive patches (v2 MSPs unchanged) |
| `tf+QQQ` | `tq, el, er` | Recaller TF calls. See encoding table below. |

##### `tq` -- LLR-based confidence (0-255)

```
tq = clip(round(LLR_nats * 10), 0, 255)
```

Mnemonic: every **23 tq points = one order of magnitude** of likelihood
ratio. Recommended thresholds:
- `tq ≥ 50`  (LLR ≥ 5 nats, LR ≈ 148:1) -- soft floor
- `tq ≥ 100` (LLR ≥ 10 nats, LR ≈ 22,000:1) -- high confidence
- `tq = 255` -- saturated (LLR ≥ 25.5, LR ≥ 1.2e11)

##### `el` / `er` -- edge sharpness (0-255)

The recaller emits a **conservative** boundary at each edge (immediately
past the last informative miss). The true boundary may extend up to the
terminating hit. `el` and `er` encode that ambiguity:

```
el = round(255 * max(0, 1 - left_ambiguity_bp / 30))
er = round(255 * max(0, 1 - right_ambiguity_bp / 30))
```

- `255` -- a hit sits immediately adjacent (sharp edge, size estimate is exact)
- `0` -- the bracketing hit is ≥30 bp away (edge could extend further; size is a lower bound)

Use these to flag calls whose endpoints might shift in browser
visualization or downstream size analysis.

#### Default behavior: `--unify` (always on)

By default, the recaller produces a clean partition: every v2 short-nuc
(`nl < --unify-threshold`) overlapped by a recaller call is **dropped**
from `nuc+`. The recaller version, with proper `tq`/`el`/`er` scoring,
replaces it in `tf+`. v2 short-nucs the recaller did *not* match are kept
in `nuc+` as fallback entries with `nq=0` (sentinel for "unverified").
v2 nucleosomes (`nl ≥ --unify-threshold`) are always preserved untouched.

This preserves the full information content while giving you a clean
1:1 partition between nucleosomes (`nuc+`) and TF/Pol II calls (`tf+`).

#### Reading the output

```python
import pysam
from fiberhmm.io.ma_tags import parse_ma_tag, parse_aq_array, tq_to_llr

bam = pysam.AlignmentFile('recalled.bam', 'rb', check_sq=False)
for read in bam:
    if not read.has_tag('MA'):
        continue
    parsed = parse_ma_tag(read.get_tag('MA'))
    aq = read.get_tag('AQ')
    qual_specs = [rt[2] for rt in parsed['raw_types']]
    n_per_type  = [len(rt[3]) for rt in parsed['raw_types']]
    per_ann = parse_aq_array(aq, qual_specs, n_per_type)
    # parsed['nuc'] = [(start, length), ...]   -- 0-based query coords
    # parsed['msp'] = [(start, length), ...]
    # parsed['tf']  = [(start, length), ...]
    # per_ann is a flat list, one sublist per annotation in MA order:
    #   first len(nucs) sublists are [nq]
    #   next  len(msps) sublists are []
    #   next  len(tfs)  sublists are [tq, el, er]
```

Or use the legacy tags directly (refreshed by the recaller to reflect the
unified call set):

```python
ns = list(read.get_tag('ns'))   # nuc starts (only nucs >=90bp + unmatched short)
nl = list(read.get_tag('nl'))
as_ = list(read.get_tag('as'))  # MSPs
al = list(read.get_tag('al'))
```

TF calls live only in MA/AQ -- legacy tags do not carry them, by
design (avoids inventing non-spec tag names).

#### Pipeline patterns

```bash
# Two-step (most common)
fiberhmm-apply -i input.bam --enzyme hia5 --seq pacbio -o tmp/ -c 8
fiberhmm-recall-tfs -i tmp/input_footprints.bam -o output/recalled.bam \
                    --enzyme hia5 --seq pacbio -c 8

# Stream apply -> recall (no intermediate file)
fiberhmm-apply -i input.bam --enzyme hia5 --seq pacbio -o - -c 8 | \
    fiberhmm-recall-tfs -i - -o recalled.bam \
                        --enzyme hia5 --seq pacbio -c 8

# Full chain: apply -> recall -> fire -> sort
fiberhmm-apply -i input.bam --enzyme hia5 --seq pacbio -o - -c 8 | \
    fiberhmm-recall-tfs -i - -o - --enzyme hia5 --seq pacbio -c 8 | \
    ft fire - - | samtools sort -o output.bam && samtools index output.bam

# Downstream-compat mode for tools that only read ns/nl
fiberhmm-recall-tfs -i tagged.bam -o recalled.bam \
                    --enzyme hia5 --seq pacbio -c 8 --downstream-compat
```

For DddA, the recall pass is **required** (the nucleosome model
`ddda_nuc.json` does not emit sub-nuc TF calls). The bundled `ddda_TF.json`
with the DddA efficiency adjustment is selected automatically:

```bash
fiberhmm-recall-tfs -i tagged.bam -o recalled.bam --enzyme ddda
```

### fiberhmm-consensus-tfs (EXPERIMENTAL — untested)

> ⚠️ **Experimental and largely untested.** Output format and defaults subject to change.

Sweeps a recalled BAM (output of `fiberhmm-recall-tfs`) and outputs one line per TF locus with consensus footprint size and occupancy statistics.

```bash
fiberhmm-consensus-tfs -i recalled.bam -o consensus_tfs.bed --min-tq 50 --min-cov 5
```

| Flag | Default | Description |
|------|---------|-------------|
| `-i/--input` | required | Recalled BAM (sorted + indexed, must have MA/AQ tags) |
| `-o/--output` | required | Output BED file, or `-` for stdout |
| `-q/--min-mapq` | 20 | Min mapping quality |
| `--min-tq` | 0 | Min TF quality score (0–255); 50 ≈ LLR 5 nats |
| `--min-cov` | 5 | Min TF-calling reads per locus to emit a line |
| `--kde-sigma` | 3.0 | Gaussian smoothing sigma for center KDE (bp) |
| `--peak-distance` | 15 | Min distance between adjacent peaks (bp) |

Output columns: `chr start end consensus_len MAD read_count spanning_reads fwd_reads rev_reads`

### fiberhmm-posteriors

Export per-position HMM posterior probabilities for downstream analysis (e.g., CNN training, custom scoring). Runs the HMM forward-backward algorithm on each read to produce P(footprint) at every position.

**Input is the same BAM you would pass to `fiberhmm-apply`** (any BAM with MM/ML modification tags or IUPAC-encoded DAF-seq reads). This is a parallel pipeline, not a downstream step.

```bash
# Export to gzipped TSV (no extra deps)
fiberhmm-posteriors -i experiment.bam --enzyme hia5 --seq pacbio -o posteriors.tsv.gz -c 4

# Export to HDF5 (requires: pip install h5py)
fiberhmm-posteriors -i experiment.bam --enzyme hia5 --seq pacbio -o posteriors.h5 -c 4

# Custom model override
fiberhmm-posteriors -i experiment.bam -m custom.json -o posteriors.tsv.gz -c 4
```

### fiberhmm-utils

Consolidated utility for model and probability management.

**convert** -- Convert legacy pickle/NPZ models to JSON:
```bash
fiberhmm-utils convert old_model.pickle new_model.json
```

**inspect** -- Print model metadata, parameters, and emission statistics:
```bash
fiberhmm-utils inspect model.json
fiberhmm-utils inspect model.json --full   # full emission table
```

**transfer** -- Transfer emission probabilities between modalities (e.g., fiber-seq to DAF-seq) using accessibility priors from a matched cell type:
```bash
fiberhmm-utils transfer \
    --target daf_sample.bam \
    --reference-bam fiberseq_footprints.bam \
    -o daf_probs/ \
    --mode daf \
    --stats
```

**adjust** -- Scale emission probabilities in a model (clamped to [0, 1]):
```bash
fiberhmm-utils adjust model.json --state accessible --scale 1.1 -o adjusted.json
```

## Output

### BAM Tags

`fiberhmm-apply` adds footprint tags compatible with the fibertools ecosystem:

| Tag | Type | Description |
|-----|------|-------------|
| `ns` | B,I | Nucleosome/footprint starts (0-based query coords) |
| `nl` | B,I | Nucleosome/footprint lengths |
| `as` | B,I | Accessible/MSP starts |
| `al` | B,I | Accessible/MSP lengths |
| `nq` | B,C | Footprint quality scores (0-255, with `--scores`) |
| `aq` | B,C | MSP quality scores (0-255, with `--scores`) |

`fiberhmm-recall-tfs` (optional 2nd pass) adds Molecular-annotation
spec-compliant tags carrying TF/Pol II footprints with proper LLR scoring:

| Tag | Type | Description |
|-----|------|-------------|
| `MA` | Z   | Annotation string: `<readlen>;nuc+Q:...;msp+:...;tf+QQQ:...` (1-based coords per spec) |
| `AQ` | B,C | Quality bytes interleaved per annotation: `nq` for nucs; `tq, el, er` for TFs (no bytes for MSPs) |

Legacy `ns`/`nl`/`as`/`al` are also rewritten to reflect the unified
call set (v2 short-nucs absorbed into TF calls are removed). TF calls
live only in `MA`/`AQ` -- by design we do not invent non-spec
nucleosome-track tag names. Use `--no-legacy-tags` to skip the legacy
refresh and emit only `MA`/`AQ`.

For DAF-seq, `fiberhmm-daf-encode` also adds:

| Tag | Type | Description |
|-----|------|-------------|
| `st` | Z | Conversion strand: `CT` (+ strand, C→T) or `GA` (- strand, G→A) |

## Streaming Pipelines

All FiberHMM tools support `-` for stdin/stdout, enabling Unix-style piping with no intermediate files:

```bash
# DAF-seq: align → encode → call footprints
minimap2 --MD -a ref.fa reads.fq | samtools view -b | \
    fiberhmm-daf-encode -i - -o - | \
    fiberhmm-apply --mode daf -i - --enzyme dddb -o output/

# Fiber-seq: call footprints from stdin
samtools view -b -h input.bam chr1 | \
    fiberhmm-apply -i - --enzyme hia5 --seq pacbio -o output/
```

When writing to stdout (`-o -`), the output is unsorted BAM. Sort and index downstream if needed:

```bash
fiberhmm-apply -i input.bam --enzyme hia5 --seq pacbio -o - | \
    samtools sort -o sorted.bam && samtools index sorted.bam
```

Pipe FiberHMM footprints directly to `ft fire` for FIRE element calling with no intermediate files:

```bash
fiberhmm-apply -i input.bam --enzyme hia5 --seq pacbio -o - -c 8 | \
    ft fire - output_fire.bam
```

When writing to a file, FiberHMM automatically sorts and indexes the output BAM.

## Fibertools Integration

FiberHMM produces BAM output with the same tag conventions used by [fibertools](https://github.com/fiberseq/fibertools-rs):

- `ns`/`nl` for nucleosome footprints
- `as`/`al` for methylase-sensitive patches (MSPs)
- `nq`/`aq` for quality scores

This means FiberHMM output can be used directly with any tool in the fibertools ecosystem, including `ft extract`, downstream analysis pipelines, and genome browsers that support fibertools-style BAM tags.

## Performance Tips

1. **Use multiple cores**: `-c 8` (or more) for parallel processing
2. **Use `--io-threads`**: for BAM compression/decompression (default: 4)
3. **Skip scaffolds**: `--skip-scaffolds` to avoid thousands of small contigs
4. **Install numba**: `pip install numba` for ~10x faster HMM computation
5. **Pipe directly**: use `-o -` to pipe to downstream tools (e.g., `ft fire`) with no intermediate files

## Model Formats

| Format | Extension | Description |
|--------|-----------|-------------|
| **JSON** | `.json` | Primary format -- portable, human-readable |
| NPZ | `.npz` | NumPy archive -- supported for loading |
| Pickle | `.pickle` | Legacy format -- supported for loading |

New models are always saved in JSON. Convert legacy models with:

```bash
fiberhmm-utils convert old_model.pickle new_model.json
```

## License

MIT License. See [LICENSE](LICENSE) for details.
