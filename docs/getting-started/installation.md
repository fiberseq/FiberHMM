# Installation

## Requirements

- Python 3.10 or later (3.10–3.13 are supported, and CI runs 3.10 and 3.12;
  3.9 does not work).
- Linux or macOS.
- A C toolchain and htslib headers only if `pip` has to build `pysam` from
  source (see [Troubleshooting](#troubleshooting)); prebuilt wheels cover the
  common platforms.

## Install with pip

```bash
pip install fiberhmm
```

This installs every command, including `fiberhmm-consensus` and
`fiberhmm-transfer`, and the bundled models and QC reference data. The core
dependencies are numpy, scipy, pandas, pysam, tqdm, Numba (JIT-compiled HMM
kernels), scikit-learn, joblib and threadpoolctl.

Optional extras:

| Extra | Adds | Needed for |
|---|---|---|
| `fiberhmm[plots]` | matplotlib | QC figures (`.qc.png`, `.qc.pdf`) and `--stats` plots |
| `fiberhmm[posteriors]` | h5py | HDF5 output of `fiberhmm-posteriors` |
| `fiberhmm[all]` | both of the above | |
| `fiberhmm[cuda]` | PyTorch | GPU predictive kernels of the deprecated staged consensus engine only |
| `fiberhmm[dev]` | pytest, pytest-cov, black, ruff | development |

```bash
pip install "fiberhmm[all]"
```

`fiberhmm[consensus]` and `fiberhmm[numba]` still resolve (FiberBrowser pins
`fiberhmm[consensus]>=3.0,<4`) but add nothing beyond the core install.

We recommend a virtual environment:

```bash
python3 -m venv fiberhmm-env
source fiberhmm-env/bin/activate
pip install "fiberhmm[all]"
```

or with conda:

```bash
conda create -n fiberhmm python=3.12
conda activate fiberhmm
pip install "fiberhmm[all]"
```

## Install from source

```bash
git clone https://github.com/fiberseq/FiberHMM.git
cd FiberHMM
pip install -e ".[all,dev]"
```

## External tools

| Tool | Used by | Without it |
|---|---|---|
| [`samtools`](https://www.htslib.org/) | sorting, indexing and concatenating BAMs | FiberHMM falls back to pysam (slower) |
| UCSC [`bedToBigBed`](https://hgdownload.soe.ucsc.edu/admin/exe/) | bigBed output of `fiberhmm-extract` and `fiberhmm-footprint-model --bigbed` | `fiberhmm-extract` writes BED only |
| UCSC `bigBedInfo`, `bigBedToBed` | `fiberhmm-utils fix-bigbed` | the command stops |
| [`ft`](https://github.com/fiberseq/fibertools-rs) (fibertools) | FIRE scoring after calling (`ft fire`) | not needed by FiberHMM itself |

```bash
# UCSC bedToBigBed on Linux (on a Mac use the macOSX.x86_64 build)
wget https://hgdownload.soe.ucsc.edu/admin/exe/linux.x86_64/bedToBigBed
chmod +x bedToBigBed && mv bedToBigBed ~/bin/   # any directory on your PATH
```

## Check the installation

```bash
python -c "import fiberhmm; print(fiberhmm.__version__)"   # 3.0.0
fiberhmm-call --help
```

Then run the [Quick start](quickstart.md) on the synthetic demo data.

Each command checks PyPI at most once a day for a newer release and prints a
one-line reminder on stderr. Set `FIBERHMM_NO_UPDATE_CHECK=1` to turn this off
(see [Environment variables](../reference/environment.md)).

## Updating and removing

```bash
pip install --upgrade fiberhmm
pip uninstall fiberhmm
```

## Troubleshooting

**`pysam` fails to build.** Install the htslib development files and retry:

```bash
# Debian/Ubuntu
sudo apt-get install -y python3-dev libhts-dev zlib1g-dev libbz2-dev liblzma-dev libcurl4-openssl-dev
# CentOS/RHEL
sudo yum install -y python3-devel htslib-devel zlib-devel bzip2-devel xz-devel libcurl-devel
# macOS
xcode-select --install && brew install htslib
```

**Numba import errors.** Numba supports a limited range of numpy versions:
`pip install --upgrade numpy numba`.

**`ERROR: Package 'fiberhmm' requires a different Python`.** FiberHMM 3.x needs
Python 3.10 or later; create an environment with a newer Python.

More in [Troubleshooting](../troubleshooting.md).
