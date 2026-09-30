# Posteriors

`fiberhmm-posteriors` exports, for every read, the HMM posterior probability
that each position is in the protected (footprint) state, for downstream
modelling such as training a neural network or custom scoring. It runs the
same observation encoding as `fiberhmm-call` on the original input BAM; it is
a parallel export, not a step after calling.

```bash
fiberhmm-posteriors -i demo/hia5_pacbio.bam --enzyme hia5 --seq pacbio \
    -o out/pacbio.posteriors.tsv.gz -c 2
```

```text
Using bundled model: .../fiberhmm/models/hia5_pacbio.json
Loaded model: mode=pacbio-fiber, context_size=3
Processing 1 regions from demo/hia5_pacbio.bam
Using 2 cores, output format: TSV
Wrote out/pacbio.posteriors.tsv.gz (0.5 MB, 300 fibers)
```

## Output formats

The format follows the extension (`--format` overrides it).

**TSV** (`.tsv.gz`, no extra dependency): a `#metadata:` JSON line, a header,
then one line per read:

```text
#metadata:{"mode": "pacbio-fiber", "context_size": 3, "edge_trim": 10, "source_bam": "hia5_pacbio.bam", "format_version": 1}
#read_id	chrom	start	end	strand	posteriors_b64	fp_starts	fp_sizes
```

| Column | Content |
|---|---|
| `read_id`, `chrom`, `start`, `end` | read name and aligned reference span |
| `strand` | `+`/`-` alignment strand for Fiber-seq; the deamination strand for DAF |
| `posteriors_b64` | base64 of one uint8 per **query** position: P(protected) × 255 |
| `fp_starts`, `fp_sizes` | Viterbi footprints as comma-separated **reference** intervals (0-based, half-open) |

Decode a line in Python:

```python
from fiberhmm.posteriors.tsv_backend import parse_posteriors_line
import gzip

with gzip.open("out/pacbio.posteriors.tsv.gz", "rt") as handle:
    for line in handle:
        record = parse_posteriors_line(line)   # None for comment lines
        if record:
            p = record["posteriors"]            # float32 array, 0..1, query order
```

`python -m fiberhmm.posteriors.tsv_backend tsv2h5 in.tsv.gz out.h5` converts
a TSV file to HDF5.

**HDF5** (`.h5`/`.hdf5`, needs `pip install "fiberhmm[posteriors]"`): the same
per-read arrays in batched datasets (`--batch-size`, 1000 reads per write).

## Which reads are exported

Aligned primary reads of at least 100 bp with at least 10 marked targets
(m6A calls or deaminations). Unmapped, secondary and supplementary records, hard-clipped
records whose MM/ML cannot match SEQ, and DAF strand-swap chimeras are
skipped. Region-parallel processing (the default) owns each read by its
start, so reads crossing a region boundary are exported once; `--streaming`
processes unindexed input in one pass.

## Options

The DAF options (`--keep-chimeras`, `--chimera-*`, `--daf-snp-mask`,
`--daf-mask-runs`, `--daf-run-policy`) behave as in `fiberhmm-call`, so the
posteriors describe the observations `fiberhmm-call` decodes. Two
differences from `fiberhmm-call`:

- `--seq` is not detected from the input; without it Hia5 uses PacBio.
- `--prob-threshold` defaults to 128 for every chemistry, including Hia5
  Nanopore (where `fiberhmm-call` uses 248); pass `--prob-threshold 248` to
  match a Nanopore call.

Every option: [`fiberhmm-posteriors`](../reference/cli.md#fiberhmm-posteriors).
