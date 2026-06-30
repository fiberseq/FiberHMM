# FiberHMM

Hidden Markov Model toolkit for calling chromatin footprints from Fiber-seq,
DAF-seq, and other single-molecule footprinting data.

FiberHMM identifies protected regions (nucleosomes, TF/Pol II footprints) and
accessible regions (methylase-sensitive patches, MSPs) from single-molecule DNA
modification data — m6A methylation (fiber-seq) and deamination marks (DAF-seq).

- [Installation](#installation)
- [Quick start](#quick-start)
- [Choosing a command](#choosing-a-command)
- [Workflows](#workflows)
- [Command reference](#command-reference)
- [Output tags](#output-tags)
- [Pre-trained models](#pre-trained-models)
- [Performance tips](#performance-tips)
- [Deep reference](docs/reference.md) — MA/AQ schema, LLR scoring model, tag glossary

## Key features

- **`fiberhmm-call`** — recommended one-command pipeline: nucleosome/MSP HMM +
  nucleosome recall + TF recall fused in one process, with region-parallel
  scaling. Coordinate-sorted input → sorted + indexed output, no separate sort.
- **Nucleosome recaller (on by default)** — splits over-merged nucleosomes on
  accessible evidence, refines edges, runs an evidence-gated periodicity prior.
- **`fiberhmm-dedup`** — PCR-duplicate removal for amplicon/UMI-less DAF-seq by
  deamination-pattern fingerprint, where coordinate dedup can't work.
- **No genome context files** — hexamer context computed from read sequences.
- **Spec-compliant tags** — `ns`/`nl`/`as`/`al` legacy tags plus `MA`/`AQ`
  [Molecular-annotation spec](https://github.com/fiberseq/Molecular-annotation-spec)
  tags with `nuc.QQQ` / `tf.QQQ` scoring.
- **Multi-platform** — PacBio fiber-seq, Nanopore fiber-seq, DAF-seq (DddB, DddA).
- **Native, fast** — no hmmlearn dependency; Numba JIT for ~10× speedup.

## Installation

```bash
pip install fiberhmm
```

From source:

```bash
git clone https://github.com/fiberseq/FiberHMM.git
cd FiberHMM && pip install -e .
```

Optional extras:

```bash
pip install numba        # ~10x faster HMM computation (recommended)
pip install matplotlib   # --stats visualization
pip install h5py         # HDF5 posteriors export
```

For bigBed output, install [UCSC tools](https://hgdownload.soe.ucsc.edu/admin/exe/)
(`bedToBigBed`, and `bigBedInfo`/`bigBedToBed` for `fiberhmm-utils fix-bigbed`).

## Quick start

`fiberhmm-call` is the entry point for almost everything. Pre-trained models are
bundled — `--enzyme` selects automatically; `-m` is only for custom models.

```bash
# Fiber-seq (Hia5), sorted+indexed BAM — region-parallel is fastest
fiberhmm-call -i sorted.bam -o calls.bam --enzyme hia5 --seq pacbio \
              -c 8 --region-parallel --skip-scaffolds

# DAF-seq (DddB), raw aligned BAM — --mode daf reads MD tags directly
fiberhmm-call -i aligned.bam -o calls.bam --enzyme dddb --mode daf \
              -c 8 --region-parallel

# DAF-seq amplicons (DddA) with PCR-duplicate removal
fiberhmm-call -i aligned.bam -o calls.bam --enzyme ddda \
              -c 8 --region-parallel --dedup

# Unaligned / stdin → streaming mode, pipe straight into FIRE
fiberhmm-call -i unaligned.bam -o - --enzyme hia5 --seq pacbio -c 8 \
    | ft fire - final.bam

# Extract calls to BED12 / bigBed for browsing
fiberhmm-extract -i calls.bam --nucleosome --msp --tf
```

## Choosing a command

| Situation | Command |
|-----------|---------|
| **Full pipeline, sorted+indexed BAM** (default) | `fiberhmm-call --region-parallel` |
| Unaligned/unsorted BAM, or reading from stdin | `fiberhmm-call` (streaming, no `--region-parallel`) |
| DAF-seq amplicons with PCR duplicates | `fiberhmm-call --dedup` (or `fiberhmm-dedup` first) |
| Only nucleosome/MSP calls, no TF recall | `fiberhmm-apply` |
| Already have an apply-tagged BAM, add TF calls | `fiberhmm-recall-tfs` |
| Apply-tagged BAM, full recall without re-running the HMM | `fiberhmm-recall-nucs` |
| Calls → BED12 / bigBed | `fiberhmm-extract` |

`fiberhmm-call` has two modes: **region-parallel** (`--region-parallel`, requires
a coordinate-sorted + indexed BAM; near-linear scaling up to chromosome count,
writes sorted+indexed output) and **streaming** (default; accepts unaligned/
unsorted BAM or stdin `-i -`, and pipes to stdout `-o -` for `ft fire`).

> `fiberhmm-run` was removed in 2.8.0 — it chained apply + recall + fire as
> separate piped subprocesses. `fiberhmm-call` fuses those stages in-process and
> is 2–9× faster. Replace `fiberhmm-run` with `fiberhmm-call [| ft fire]`.

## Workflows

### Fiber-seq (Hia5)

```bash
fiberhmm-call -i sorted.bam -o calls.bam --enzyme hia5 --seq pacbio \
              -c 8 --region-parallel --skip-scaffolds
```

`--seq` (`pacbio` or `nanopore`) is required for Hia5 — PacBio detects m6A on both
strands, Nanopore on one. Add FIRE scoring as a second step: `ft fire calls.bam
final.bam`, or stream it (see Quick start).

### DAF-seq (DddB)

`fiberhmm-call --mode daf` finds deamination calls per read, in priority order:

1. **R/Y IUPAC codes** in the stored sequence (from `fiberhmm-daf-encode`) — fast path
2. **MD tag** on a raw aligned BAM — parsed on the fly, no encode step needed
3. **`--reference ref.fa`** — fallback when the BAM has neither R/Y nor MD

```bash
# Raw DAF BAM with MD tags (from `minimap2 --MD` or `samtools calmd`)
fiberhmm-call -i aligned.bam -o calls.bam --enzyme dddb --mode daf --region-parallel

# No MD tags: supply a reference
fiberhmm-call -i aligned.bam -o calls.bam --enzyme dddb --mode daf \
              --reference ref.fa --region-parallel
```

`fiberhmm-call` fast-fails in under a second if none of R/Y, MD, or `--reference`
is available, instead of silently skipping every read. Running `fiberhmm-daf-encode`
first is optional and produces byte-identical calls — use it only if you also want
R/Y stamped into the stored sequence for downstream R/Y-aware tools.

### DAF-seq amplicons (DddA)

`--enzyme ddda` handles DddA's specifics automatically:

- **Two models** — `ddda_nuc.json` for nucleosomes and `ddda_TF.json` for TF/Pol II
  recall — are selected and run in one pass. (For QC you can run them separately:
  `fiberhmm-apply --enzyme ddda --mode daf` then `fiberhmm-recall-tfs --enzyme ddda`.)
- **Radial nucleosome recall** is **on by default** (DddA deaminates *inside*
  nucleosomes, so the standard accessible-cut split would shatter them; instead a
  radial deamination template places dyads). Use `--no-recall-nucs` for raw HMM
  nucleosomes.
- **Strand-swap chimera filter** is on by default (reads deaminated C→T in one
  segment and G→A in another are dropped and counted). `--keep-chimeras` to
  disable; `--chimera-min-seg` / `--chimera-purity` to tune.

```bash
fiberhmm-call -i aligned.bam -o calls.bam --enzyme ddda \
              -c 8 --region-parallel --dedup
```

> **DddA amplicons are typically ~80% PCR-duplicated**, and coordinate dedup
> (Picard/markdup) does **not** apply — every read piles up on the same locus with
> primer-fixed ends. `--dedup` collapses duplicates by deamination fingerprint
> *before* footprinting (see [`fiberhmm-dedup`](#fiberhmm-dedup)). It is opt-in
> because it removes reads.

> ⚠️ The DddA radial nucleosome recaller (added 2.14.0) is still under active
> validation — inspect nucleosome calls before relying on them, and
> [report issues](https://github.com/fiberseq/FiberHMM/issues).

### Second-pass recall on an apply-tagged BAM

If you already have a BAM tagged by `fiberhmm-apply` and want to add calls without
re-running the HMM, use the recallers. Both reconstruct the per-base observations
from each read's `MM`/`ML`+sequence and reuse the existing `ns`/`nl`/`as`/`al`
tags — the HMM is **not** re-run.

```bash
# TF recall only (over the original apply MSPs + short nucs)
fiberhmm-recall-tfs  -i apply.bam -o recalled.bam --enzyme hia5 --seq pacbio -c 8

# Full recall: nucleosome refine → MSP re-derive → TF recall → promotion
fiberhmm-recall-nucs -i apply.bam -o recalled.bam --enzyme hia5 --seq pacbio -c 8
```

`fiberhmm-recall-nucs` is byte-identical to `fiberhmm-call --recall-nucs` for a
matched `--phase-nrl`. **Linear reads only** — circular reads must use
`fiberhmm-call -r --recall-nucs`.

### PCR deduplication

```bash
# Collapse to one representative read per molecule (default)
fiberhmm-dedup -i sample.bam -o sample.dedup.bam

# Mark duplicates instead of removing them (0x400 + di/ds tags)
fiberhmm-dedup -i sample.bam -o sample.markdup.bam --flag-only
```

See [`fiberhmm-dedup`](#fiberhmm-dedup) for how it works and when to use it.

## Command reference

### fiberhmm-call

Fused apply + nucleosome recall + TF recall in one process. See
[Choosing a command](#choosing-a-command) for the region-parallel vs streaming
modes.

| Flag | Default | Description |
|------|---------|-------------|
| `-i/--input` | required | Input BAM, or `-` for stdin. |
| `-o/--output` | required | Output BAM, or `-` for stdout (unsorted). |
| `--enzyme` | — | `hia5`, `dddb`, or `ddda` (auto-selects bundled model). |
| `--seq` | — | `pacbio` or `nanopore` (required for Hia5). |
| `--mode` | from model | `pacbio-fiber` / `nanopore-fiber` / `daf`. |
| `--reference` | — | FASTA fallback for `--mode daf` when no R/Y or MD. |
| `-c/--cores` | 4 | Worker processes. |
| `--io-threads` | 8 | htslib I/O threads. |
| `--region-parallel` | off | Per-region worker pool (requires sorted+indexed input). |
| `--skip-scaffolds` | off | Drop small scaffolds (region-parallel). |
| `--chroms chr1 …` | all | Restrict to specific chromosomes (region-parallel). |
| `--no-recall-nucs` | recall on | Disable nucleosome recall (baseline HMM `nuc.Q`). |
| `--phase-nrl` | `auto` | Periodicity prior: `auto` (estimate, ~150–215 bp), `off`, or a fixed bp. |
| `--min-llr` | enzyme preset | Override TF LLR threshold. |
| `-r/--circular` | off | Circular molecule mode (see [reference](docs/reference.md#circular-molecules)). |
| `--keep-chimeras` | off | DAF: keep strand-swap chimeric reads (default: filter + count). |
| `--no-legacy-tags` | off | Emit only `MA`/`AQ`, skip `ns/nl/as/al`. |
| `--downstream-compat` | off | Write TF calls into legacy `ns/nl` (skip `MA/AQ`). |
| `--dedup` | off | **DAF only.** PCR-dedup before footprinting (see below). Ignored for `hia5`. |

`--dedup` tunables (forwarded to the dedup pass): `--dedup-min-jaccard` (0.95),
`--dedup-flag-only`, `--dedup-min-deam`, `--dedup-prob-threshold`,
`--dedup-ignore-strand`, `--dedup-stats-tsv`. MinHash internals stay at defaults —
use standalone `fiberhmm-dedup` to tune those.

### fiberhmm-apply

Apply a trained HMM to call nucleosomes/MSPs (no TF recall). Streaming pipeline
with stdin/stdout support.

```bash
fiberhmm-apply -i experiment.bam --enzyme hia5 --seq pacbio -o output/ -c 8
```

| Flag | Default | Description |
|------|---------|-------------|
| `-i/--input` | required | Input BAM, or `-` for stdin. |
| `-m/--model` | optional | Custom model (`.json`/`.npz`/`.pickle`); overrides `--enzyme`. |
| `--enzyme` | optional | `hia5`, `dddb`, or `ddda`. Required unless `-m` is given. |
| `--seq` | optional | `pacbio`/`nanopore` (required for Hia5; ignored for dddb/ddda). |
| `-o/--outdir` | required | Output directory, or `-` for stdout BAM. |
| `--mode` | from model | Analysis mode. |
| `-c/--cores` | 1 | CPU cores (0 = auto). |
| `--io-threads` | 4 | htslib I/O threads. |
| `-q/--min-mapq` | 0 | Min mapping quality (`0` = no filtering). |
| `--min-read-length` | 1000 | Min aligned read length (`0` to disable). |
| `-e/--edge-trim` | 10 | Edge masking (bp). |
| `--msp-min-size` | 0 | Minimum MSP region size (bp). |
| `--scores` | off | Compute per-footprint confidence scores (`nq`/`aq`). |
| `-r/--circular` | off | Circular molecule mode. |
| `--skip-scaffolds` / `--chroms` / `--primary` | — | As in `fiberhmm-call`. |

Reads are passed through unchanged (no footprint tags) when MAPQ or length is
below threshold, when there are no MM/ML tags, when unmapped, or when the HMM
finds no footprints.

### fiberhmm-recall-tfs / fiberhmm-recall-nucs

LLR-based second-pass recallers over an apply-tagged BAM. `recall-tfs` adds TF/Pol
II footprints; `recall-nucs` additionally refines nucleosomes (= `recall-tfs
--recall-nucs`). For DddA the TF recall pass is **required** (the nucleosome model
doesn't emit sub-nucleosomal calls) and `ddda_TF.json` is selected automatically.

```bash
fiberhmm-recall-tfs -i apply.bam -o recalled.bam --enzyme hia5 --seq pacbio -c 8
```

| Flag | Default | Description |
|------|---------|-------------|
| `-i/--in-bam` | required | Input BAM tagged by `fiberhmm-apply`. `-` for stdin. |
| `-o/--out-bam` | required | Output BAM (`MA`/`AQ` + refreshed legacy tags). `-` for stdout. |
| `-m/--model` | optional | Custom model JSON; overrides `--enzyme`. |
| `--enzyme` | optional | `hia5`/`dddb`/`ddda` — sets the model + `--min-llr` preset. |
| `--seq` | optional | `pacbio`/`nanopore` (Hia5 only). |
| `--min-llr` | preset | Min cumulative LLR (nats) per call (hia5 5.0, dddb 4.0, ddda 5.0). |
| `--min-opps` | 3 | Min informative target positions per call. |
| `--unify-threshold` | 90 | Footprints with `nl <` this may be demoted to `tf.`. |
| `--no-legacy-tags` | off | Emit only `MA`/`AQ`. |
| `--downstream-compat` | off | TF calls into legacy `ns/nl`, no `MA/AQ` (per-TF quality lost). |
| `-c/--cores` | 1 | Worker processes (0 = auto). |
| `--io-threads` | 4 | htslib threads. |

See the [deep reference](docs/reference.md) for the MA/AQ schema, the `tq`/`el`/`er`
quality bytes, output modes, circular molecules, and how to parse the output.

### fiberhmm-extract

Extract nucleosome/MSP/TF/m6A/m5C/deamination features from tagged BAMs to
BED12 / bigBed (bigBed by default; one file per feature type).

```bash
fiberhmm-extract -i calls.bam -o output/ -c 8        # all types
fiberhmm-extract -i calls.bam --nucleosome --msp --tf
fiberhmm-extract -i calls.bam --keep-bed             # keep BED alongside bigBed
fiberhmm-extract -i calls.bam --tf --msp --circular-groups   # FiberBrowser grouping
```

| Flag | Default | Description |
|------|---------|-------------|
| `--nucleosome` / `--msp` / `--tf` / `--m6a` / `--m5c` / `--deam` | all | Feature types to extract (default: all). |
| `--bed-only` / `--keep-bed` | off | BED only / keep BED beside bigBed. |
| `--block-scores` | off | Append per-block quality columns (BED12+N). |
| `--circular-groups` | off | Emit circular grouping fields for FiberBrowser. |
| `--sample-name` | BAM stem | Sample tag embedded in each bigBed's autoSQL. |
| `-S/--sort-mem` | `1G` | Buffer for the BED sort (`sort -S`; e.g. `8G`). |
| `--sort-parallel` | `--cores` | Sort threads (GNU sort; feature-detected). |
| `-c/--cores` | 1 | Worker processes. |

The post-extract sort runs under `LC_ALL=C` (a large speedup on its own); `-S` and
`--sort-parallel` help further on deep/whole-genome BAMs. Each bigBed embeds a
`Sample:` autoSQL tag (sanitized to a dot/space-free token) that FiberBrowser uses
to group a sample's layers; repair older bigBeds with
[`fiberhmm-utils fix-bigbed`](#fiberhmm-utils).

### fiberhmm-dedup

PCR-duplicate detection for DAF-seq via the per-read **deamination pattern**.
DAF-seq amplicons pile up on one locus with primer-fixed ends, so coordinate dedup
(Picard/markdup) has no positional signal and there are no UMIs. The molecular
fingerprint is the set of reference positions deaminated by the enzyme (R/Y, MM/ML
dU, or MD mismatch — same sources as `fiberhmm-extract --deam`). PCR copies share
that pattern but rarely *exactly* (sequencing error and missed/over-called
deaminations perturb a handful of the hundreds of calls), so exact-match dedup
misses most duplicates.

`fiberhmm-dedup` clusters reads whose deamination sets match within a Jaccard
threshold (MinHash + LSH, near-linear; ~20 s for 75k reads) and **collapses each
cluster to one representative by default** (highest MAPQ / most-complete). The
representative carries `ds` = number of copies it represents.

```bash
fiberhmm-dedup -i sample.bam -o sample.dedup.bam              # collapse (default)
fiberhmm-dedup -i sample.bam -o sample.markdup.bam --flag-only # mark only
```

| Flag | Default | Description |
|------|---------|-------------|
| `-i/--input` | required | Input DAF-seq BAM (R/Y-, MM/ML-dU-, or MD-encoded). |
| `-o/--output` | required | Output BAM (stays coordinate-sorted if the input was). |
| `--flag-only` | off | Mark duplicates (`0x400` + `di`/`ds`) instead of collapsing. |
| `--min-jaccard` | 0.95 | Min deamination-set Jaccard to call two reads the same molecule (bimodal gap ~0.90–0.95). |
| `--min-deam` | 10 | Reads with fewer calls aren't fingerprinted; passed through. |
| `--ignore-strand` | off | Allow opposite-strand reads to be duplicates. |
| `-p/--prob-threshold` | 0 | Min ML probability for MM/ML-native dU calls. |
| `--stats-tsv` | — | Write a `cluster_id<TAB>n_reads` table. |

### fiberhmm-daf-encode

Preprocess plain aligned DAF-seq BAMs: identify C→T / G→A mismatches via the MD
tag, encode them as IUPAC Y/R in the query sequence, and add an `st:Z` strand tag.
**Optional** — `fiberhmm-call --mode daf` reads MD directly. Use it only if you
want R/Y in the stored sequence for downstream R/Y-aware tools.

```bash
fiberhmm-daf-encode -i aligned.bam -o encoded.bam
```

Key flags: `--reference` (fallback if MD missing), `-q/--min-mapq` (20),
`--min-read-length` (1000), `--strand` (`CT`/`GA`/`auto`), `--io-threads` (4).

### fiberhmm-posteriors

Export per-position HMM posterior P(footprint) for downstream analysis (CNN
training, custom scoring). Input is the same BAM you'd pass to `fiberhmm-apply`.

```bash
fiberhmm-posteriors -i experiment.bam --enzyme hia5 --seq pacbio -o post.tsv.gz -c 4
fiberhmm-posteriors -i experiment.bam --enzyme hia5 --seq pacbio -o post.h5 -c 4  # needs h5py
```

### fiberhmm-probs / fiberhmm-train

Train custom models. `fiberhmm-probs` builds emission tables from accessible /
inaccessible control BAMs; `fiberhmm-train` fits the HMM using them.

```bash
fiberhmm-probs -a accessible.bam -u inaccessible.bam -o probs/ --mode pacbio-fiber -k 3 4 5 6 --stats
fiberhmm-train -i sample.bam -p probs/tables/accessible_A_k3.tsv probs/tables/inaccessible_A_k3.tsv -o models/ -k 3 --stats
```

`fiberhmm-train` writes `best-model.json` (recommended), `.npz`, all iterations,
training read IDs, config, and `--stats` plots.

### fiberhmm-utils

Model and bigBed utilities:

```bash
fiberhmm-utils convert old_model.pickle new_model.json   # legacy → JSON
fiberhmm-utils inspect model.json [--full]               # metadata + emissions
fiberhmm-utils transfer --target daf.bam --reference-bam fiber.bam -o probs/ --mode daf
fiberhmm-utils adjust model.json --state accessible --scale 1.1 -o adjusted.json
fiberhmm-utils fix-bigbed sample.filtered_T_*.bb sample.filtered_GA_*.bb --in-place
```

`fix-bigbed` repairs the embedded `Sample:` autoSQL tag in existing bigBeds (use
when split/genotype-filtered pools loaded side-by-side in FiberBrowser had layers
go missing because their tags weren't distinct). Rebuilds via `bigBedToBed →
bedToBigBed`; needs UCSC `bigBedInfo`/`bigBedToBed`/`bedToBigBed`.

## Output tags

`fiberhmm-apply` writes fibertools-style legacy tags (`ns`/`nl` nucleosomes,
`as`/`al` MSPs, `nq`/`aq` quality). The TF recaller adds spec-compliant `MA`/`AQ`
tags carrying `nuc.Q` / `msp.` / `tf.QQQ` with full LLR scoring. **TF calls live
only in `MA`/`AQ`** (legacy tags carry nucleosomes only, by design); use
`--downstream-compat` to fold TFs into `ns`/`nl` for tools that read only legacy
tags. Full schema, byte layouts, and parsing examples are in the
[deep reference](docs/reference.md).

This makes FiberHMM output directly usable across the
[fibertools](https://github.com/fiberseq/fibertools-rs) ecosystem (`ft extract`,
`ft fire`, FiberBrowser).

## Pre-trained models

Bundled with the package — `--enzyme` (+ `--seq` for Hia5) selects automatically;
`-m` only for custom models.

| Model | `--enzyme` | `--seq` | Mode | Used by |
|-------|-----------|---------|------|---------|
| `hia5_pacbio.json` | `hia5` | `pacbio` | `pacbio-fiber` | apply / recall-tfs |
| `hia5_nanopore.json` | `hia5` | `nanopore` | `nanopore-fiber` | apply / recall-tfs |
| `ddda_nuc.json` | `ddda` | — | `daf` | apply — **nucleosomes only** |
| `ddda_TF.json` | `ddda` | — | `daf` | recall-tfs — **required 2nd pass** |
| `dddb_nanopore.json` | `dddb` | — | `daf` | apply / recall-tfs |

For DddA, `fiberhmm-call --enzyme ddda` runs both models in one pass. For Hia5 and
DddB a single model captures both nucleosomes and small footprints, so the recall
pass is optional refinement. Older models live in `models/legacy/` (reproducibility
only); custom models load with `-m`. Formats: `.json` (primary), `.npz`, `.pickle`
(legacy, load-only) — convert with `fiberhmm-utils convert`.

## Performance tips

1. **Multiple cores** — `-c 8` (or more).
2. **`--io-threads`** — for BAM (de)compression.
3. **`--skip-scaffolds`** — avoid thousands of small contigs in region-parallel mode.
4. **`pip install numba`** — ~10× faster HMM computation.
5. **Pipe directly** — `-o -` into `ft fire`/`samtools` with no intermediate files.

## License

MIT License. See [LICENSE](LICENSE).
