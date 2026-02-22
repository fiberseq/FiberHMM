# FiberHMM

Hidden Markov Model toolkit for calling chromatin footprints from fiber-seq and DAF-seq single-molecule data.

FiberHMM identifies protected regions (footprints) and accessible regions (methylase-sensitive patches, MSPs) from single-molecule DNA modification data, including m6A methylation (fiber-seq) and deamination marks (DAF-seq).

## Key Features

- **No genome context files** -- hexamer context computed directly from read sequences
- **Fibertools-compatible output** -- tagged BAM with `ns`/`nl`/`as`/`al` tags, ready for downstream tools
- **Native HMM implementation** -- no hmmlearn dependency; Numba JIT optional for ~10x speedup
- **Region-parallel processing** -- scales linearly with cores for large genomes
- **Multi-platform** -- supports PacBio fiber-seq, Nanopore fiber-seq, and DAF-seq
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
```

For bigBed output, install [`bedToBigBed`](https://hgdownload.soe.ucsc.edu/admin/exe/) from UCSC tools.

## Quick Start

### 1. Generate emission probabilities

Requires accessible (naked DNA) and inaccessible (native chromatin) control BAMs:

```bash
python generate_probs.py \
    -a accessible_control.bam \
    -u inaccessible_control.bam \
    -o probs/ \
    --stats
```

### 2. Train HMM

```bash
python train_model.py \
    -i sample.bam \
    -p probs/tables/accessible_A_k3.tsv probs/tables/inaccessible_A_k3.tsv \
    -o models/ \
    --stats
```

### 3. Call footprints

```bash
python apply_model.py \
    -i experiment.bam \
    -m models/best-model.json \
    -o output/ \
    -c 8 \
    --scores
```

### 4. Extract to BED12/bigBed

```bash
python extract_tags.py -i output/experiment_footprints.bam
```

## Pre-trained Models

FiberHMM ships with pre-trained models in `models/` ready for immediate use:

| Model | File | Enzyme | Platform | Mode |
|-------|------|--------|----------|------|
| **Hia5 PacBio** | `hia5_pacbio.json` | Hia5 (m6A) | PacBio | `pacbio-fiber` |
| **Hia5 Nanopore** | `hia5_nanopore.json` | Hia5 (m6A) | Nanopore | `nanopore-fiber` |
| **DddA PacBio** | `ddda_pacbio.json` | DddA (deamination) | PacBio | `daf` |
| **DddB Nanopore** | `dddb_nanopore.json` | DddB (deamination) | Nanopore | `daf` |

```bash
# Example: call footprints with a pre-trained model
python apply_model.py -i experiment.bam -m models/hia5_pacbio.json -o output/ -c 8
```

## Analysis Modes

| Mode | Flag | Description | Target bases |
|------|------|-------------|--------------|
| **PacBio fiber-seq** | `--mode pacbio-fiber` | Default. m6A at A and T (both strands) | A, T (with RC) |
| **Nanopore fiber-seq** | `--mode nanopore-fiber` | m6A at A only (single strand) | A only |
| **DAF-seq** | `--mode daf` | Deamination at C/G (strand-specific) | C or G |

All scripts accept `--mode`; context size `-k` (3-10) determines the hexamer table size.

**Note on mode selection:** The `pacbio-fiber` vs `nanopore-fiber` distinction only matters for Hia5 (m6A), where PacBio detects modifications on both strands while Nanopore detects only one. For deaminase-based methods (DddA, DddB), `--mode daf` is always used regardless of sequencing platform -- the chemistry is inherently strand-specific. Accuracy may differ between platforms, but the mode is the same.

## Output

### BAM Tags

`apply_model.py` adds footprint tags compatible with the fibertools ecosystem:

| Tag | Type | Description |
|-----|------|-------------|
| `ns` | B,I | Nucleosome/footprint starts (0-based query coords) |
| `nl` | B,I | Nucleosome/footprint lengths |
| `as` | B,I | Accessible/MSP starts |
| `al` | B,I | Accessible/MSP lengths |
| `nq` | B,C | Footprint quality scores (0-255, with `--scores`) |
| `aq` | B,C | MSP quality scores (0-255, with `--scores`) |

### BED12/bigBed Extraction

Use `extract_tags.py` to extract features from tagged BAMs for browser visualization:

```bash
# Extract all feature types to bigBed
python extract_tags.py -i output/sample_footprints.bam

# Extract only footprints
python extract_tags.py -i output/sample_footprints.bam --footprint

# Keep BED files alongside bigBed
python extract_tags.py -i output/sample_footprints.bam --keep-bed
```

## Scripts Reference

### generate_probs.py / `fiberhmm-probs`

Generate emission probability tables from accessible and inaccessible control BAMs.

```bash
python generate_probs.py \
    -a accessible_control.bam \
    -u inaccessible_control.bam \
    -o probs/ \
    --mode pacbio-fiber \
    -k 3 4 5 6 \
    --stats
```

### train_model.py / `fiberhmm-train`

Train the HMM on BAM data using precomputed emission probabilities.

```bash
python train_model.py \
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

### apply_model.py / `fiberhmm-apply`

Apply a trained model to call footprints. Supports region-parallel processing.

```bash
python apply_model.py \
    -i experiment.bam \
    -m models/best-model.json \
    -o output/ \
    -c 8 \
    --scores
```

Key options:

| Flag | Default | Description |
|------|---------|-------------|
| `-c/--cores` | 1 | CPU cores (0 = auto-detect) |
| `--region-size` | 10000000 | Region size for parallel chunks |
| `--skip-scaffolds` | false | Skip scaffold/contig chromosomes |
| `--chroms` | all | Process only these chromosomes |
| `--scores` | false | Compute per-footprint confidence scores |
| `-q/--min-mapq` | 20 | Min mapping quality |
| `-e/--edge-trim` | 10 | Edge masking (bp) |

### extract_tags.py / `fiberhmm-extract`

Extract footprint/MSP/m6A/m5C features from tagged BAMs to BED12/bigBed.

```bash
python extract_tags.py -i output/sample_footprints.bam -o output/ -c 8
```

### fiberhmm-utils

Consolidated utility for model and probability management. Four subcommands:

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

## Fibertools Integration

FiberHMM produces BAM output with the same tag conventions used by [fibertools](https://github.com/fiberseq/fibertools-rs):

- `ns`/`nl` for nucleosome footprints
- `as`/`al` for methylase-sensitive patches (MSPs)
- `nq`/`aq` for quality scores

This means FiberHMM output can be used directly with any tool in the fibertools ecosystem, including `ft extract`, downstream analysis pipelines, and genome browsers that support fibertools-style BAM tags.

## FiberBrowser

FiberBrowser, a dedicated genome browser for single-molecule chromatin data, is coming soon.

## Performance Tips

1. **Use multiple cores**: `-c 8` (or more) for parallel processing
2. **Adjust region size for small genomes**:
   - Yeast (~12 MB): `--region-size 500000`
   - Drosophila (~140 MB): `--region-size 2000000`
   - Human/mammalian: default 10 MB is fine
3. **Skip scaffolds**: `--skip-scaffolds` to avoid thousands of small contigs
4. **Install numba**: `pip install numba` for ~10x faster HMM training

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
