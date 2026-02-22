# FiberHMM Installation Guide

This guide covers installation of FiberHMM on Linux and macOS systems.

## Requirements

- Python 3.9 or later
- C compiler (gcc on Linux, Xcode command line tools on macOS)
- htslib development headers (for pysam)

## Quick Install

### Using pip (recommended)

```bash
pip install fiberhmm
```

### From source

```bash
git clone https://github.com/fiberseq/FiberHMM.git
cd FiberHMM
pip install -e .
```

## Detailed Installation

### 1. System Dependencies

#### Linux (Ubuntu/Debian)

```bash
sudo apt-get update
sudo apt-get install -y \
    python3-dev \
    python3-pip \
    libhts-dev \
    zlib1g-dev \
    libbz2-dev \
    liblzma-dev \
    libcurl4-openssl-dev
```

#### Linux (CentOS/RHEL)

```bash
sudo yum install -y \
    python3-devel \
    htslib-devel \
    zlib-devel \
    bzip2-devel \
    xz-devel \
    libcurl-devel
```

#### macOS

```bash
# Install Xcode command line tools
xcode-select --install

# Install htslib via Homebrew
brew install htslib
```

### 2. Python Environment (recommended)

We recommend using a virtual environment:

```bash
# Using venv
python3 -m venv fiberhmm-env
source fiberhmm-env/bin/activate

# Or using conda
conda create -n fiberhmm python=3.11
conda activate fiberhmm
```

### 3. Install FiberHMM

#### Basic installation (core functionality)

```bash
pip install fiberhmm
```

#### With performance optimizations

```bash
pip install "fiberhmm[numba]"
```

#### With visualization support

```bash
pip install "fiberhmm[plots]"
```

#### Full installation (all features)

```bash
pip install "fiberhmm[all]"
```

### 4. Verify Installation

```bash
# Test core functionality
python -c "from fiberhmm import FiberHMM; print('FiberHMM loaded successfully')"

# Test CLI entry points
fiberhmm-apply --help
fiberhmm-train --help
fiberhmm-probs --help
fiberhmm-extract --help
fiberhmm-utils --help
```

## Optional Dependencies

### Numba (Recommended)

Numba provides ~10x speedup for HMM computations:

```bash
pip install numba
```

### UCSC tools (for bigBed output)

For bigBed format output, install `bedToBigBed`:

```bash
# Linux
wget https://hgdownload.soe.ucsc.edu/admin/exe/linux.x86_64/bedToBigBed
chmod +x bedToBigBed
sudo mv bedToBigBed /usr/local/bin/

# macOS
wget https://hgdownload.soe.ucsc.edu/admin/exe/macOSX.x86_64/bedToBigBed
chmod +x bedToBigBed
sudo mv bedToBigBed /usr/local/bin/
```

## Troubleshooting

### pysam installation fails

This usually means htslib headers are missing:

```bash
# Ubuntu/Debian
sudo apt-get install libhts-dev

# macOS
brew install htslib
```

### numba import errors

Numba requires specific numpy versions. Try:

```bash
pip install --upgrade numpy numba
```

### Permission denied errors

If you see permission errors, try installing with `--user`:

```bash
pip install --user fiberhmm
```

## Development Installation

For contributing to FiberHMM:

```bash
git clone https://github.com/fiberseq/FiberHMM.git
cd FiberHMM

# Install with development dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v
```

## Updating

```bash
pip install --upgrade fiberhmm
```

## Uninstalling

```bash
pip uninstall fiberhmm
```
