# FiberHMM

Hidden Markov Model toolkit for calling chromatin footprints from Fiber-seq, DAF-seq, and other single-molecule footprinting data.

FiberHMM identifies protected regions (footprints) and accessible regions (methylase-sensitive patches, MSPs) from single-molecule DNA modification data, including m6A methylation (fiber-seq) and deamination marks (DAF-seq).

## Key Features

- **No genome context files** -- hexamer context computed directly from read sequences
- **Fibertools-compatible output** -- tagged BAM with `ns`/`nl`/`as`/`al` tags, ready for downstream tools
- **Native HMM implementation** -- no hmmlearn dependency; Numba JIT optional for ~10x speedup
- **Region-parallel processing** -- scales linearly with cores for large genomes
- **Streaming pipeline** -- supports stdin/stdout piping for composable workflows
- **Multi-platform** -- supports PacBio fiber-seq, Nanopore fiber-seq, and DAF-seq
- **DAF-seq preprocessing** -- `fiberhmm-daf-encode` converts plain aligned BAMs to IUPAC R/Y encoded BAMs
- **Legacy model support** -- loads old hmmlearn-trained pickle/NPZ models seamlessly

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

### Fiber-seq (PacBio or Nanopore)

If you have a BAM with m6A modification tags (MM/ML), you can call footprints directly with a pre-trained model:

```bash
fiberhmm-apply -i experiment.bam -m models/hia5_pacbio.json -o output/ -c 8
```

### DAF-seq

DAF-seq BAMs from minimap2 contain deamination events as C→T / G→A mismatches but lack the IUPAC encoding and strand tags that FiberHMM expects. Use `fiberhmm-daf-encode` to preprocess, then call footprints:

```bash
# Two-step
fiberhmm-daf-encode -i aligned.bam -o encoded.bam
fiberhmm-apply --mode daf -i encoded.bam -m models/ddda_pacbio.json -o output/ -c 8

# Or as a streaming pipeline (no intermediate file)
fiberhmm-daf-encode -i aligned.bam -o - | \
    fiberhmm-apply --mode daf --streaming -i - -m models/ddda_pacbio.json -o output/
```

If your BAM is missing MD tags (minimap2 wasn't run with `--MD`), add them first:

```bash
samtools calmd -b aligned.bam ref.fa | \
    fiberhmm-daf-encode -i - -o - | \
    fiberhmm-apply --mode daf --streaming -i - -m models/ddda_pacbio.json -o output/
```

Or pass the reference FASTA directly (slower):

```bash
fiberhmm-daf-encode -i aligned.bam -o encoded.bam --reference ref.fa
```

### Extract to BED12/bigBed

```bash
fiberhmm-extract -i output/experiment_footprints.bam
```

## Pre-trained Models

FiberHMM ships with pre-trained models in `models/` ready for immediate use:

| Model | File | Enzyme | Platform | Mode |
|-------|------|--------|----------|------|
| **Hia5 PacBio** | `hia5_pacbio.json` | Hia5 (m6A) | PacBio | `pacbio-fiber` |
| **Hia5 Nanopore** | `hia5_nanopore.json` | Hia5 (m6A) | Nanopore | `nanopore-fiber` |
| **DddA PacBio** | `ddda_pacbio.json` | DddA (deamination) | PacBio | `daf` |
| **DddB Nanopore** | `dddb_nanopore.json` | DddB (deamination) | Nanopore | `daf` |

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

### fiberhmm-apply

Apply a trained HMM to call footprints. Supports region-parallel and streaming modes.

```bash
# Region-parallel (indexed BAM)
fiberhmm-apply -i experiment.bam -m model.json -o output/ -c 8

# Streaming (stdin/stdout, unindexed BAM)
fiberhmm-apply --streaming -i - -m model.json -o output/
```

| Flag | Default | Description |
|------|---------|-------------|
| `-i/--input` | required | Input BAM, or `-` for stdin |
| `-m/--model` | required | Trained HMM model (.json, .npz, or .pickle) |
| `-o/--outdir` | required | Output directory, or `-` for stdout BAM |
| `--mode` | auto-detect | Analysis mode (see table above) |
| `-c/--cores` | 1 | CPU cores (0 = auto-detect) |
| `--streaming` | false | Streaming pipeline mode (for unindexed/stdin input) |
| `--io-threads` | 4 | htslib I/O threads |
| `-q/--min-mapq` | 20 | Min mapping quality |
| `--min-read-length` | 1000 | Min aligned read length (bp) |
| `-e/--edge-trim` | 10 | Edge masking (bp) |
| `--scores` | false | Compute per-footprint confidence scores |
| `--region-size` | 10000000 | Region size for parallel chunks |
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

### fiberhmm-posteriors

Export per-position HMM posterior probabilities for downstream analysis (e.g., CNN training, custom scoring). Runs the HMM forward-backward algorithm on each read to produce P(footprint) at every position.

**Input is the same BAM you would pass to `fiberhmm-apply`** (any BAM with MM/ML modification tags or IUPAC-encoded DAF-seq reads). This is a parallel pipeline, not a downstream step.

```bash
# Export to gzipped TSV (no extra deps)
fiberhmm-posteriors -i experiment.bam -m model.json -o posteriors.tsv.gz -c 4

# Export to HDF5 (requires: pip install h5py)
fiberhmm-posteriors -i experiment.bam -m model.json -o posteriors.h5 -c 4
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
    fiberhmm-apply --mode daf --streaming -i - -m models/ddda_pacbio.json -o output/

# Fiber-seq: call footprints from stdin
samtools view -b -h input.bam chr1 | \
    fiberhmm-apply --streaming -i - -m model.json -o output/
```

When writing to stdout (`-o -`), the output is unsorted BAM. Sort and index downstream if needed:

```bash
fiberhmm-apply --streaming -i input.bam -m model.json -o - | \
    samtools sort -o sorted.bam && samtools index sorted.bam
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
3. **Adjust region size for small genomes**:
   - Yeast (~12 MB): `--region-size 500000`
   - Drosophila (~140 MB): `--region-size 2000000`
   - Human/mammalian: default 10 MB is fine
4. **Skip scaffolds**: `--skip-scaffolds` to avoid thousands of small contigs
5. **Install numba**: `pip install numba` for ~10x faster HMM computation
6. **Streaming mode**: use `--streaming` with piped input to avoid writing intermediate files

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
