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
Wrote out/pacbio.posteriors.tsv.gz (0.5 MB, 300 fibers)
```

`-v` also prints the model, region and core details and a progress bar.

## Output formats

The format follows the extension (`--format` overrides it).

**TSV** (`.tsv.gz`, no extra dependency): a `#metadata:` JSON line, a header,
then one line per read:

```text
#metadata:{"mode": "pacbio-fiber", "context_size": 3, "edge_trim": 10, "source_bam": "hia5_pacbio.bam", "format_version": 1, "fiberhmm_version": "3.0.0", "model": "hia5_pacbio.json", "model_sha256": "13342f5f...", "prob_threshold": 128, "filter_chimeras": true, "chimera_min_seg": 5, "chimera_purity": 0.8, "daf_mask_runs": 0, "daf_run_policy": "keep-one", "cpg_mask_policy": "off", "fiberhmm_commit": "...", "enzyme": "hia5", "platform": "pacbio", "fiberhmm_mode": "pacbio-fiber"}
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
posteriors describe the observations `fiberhmm-call` decodes. For DddA the
HMM sees `fiberhmm-call`'s first-pass CpG mask: CpGs are hidden except inside
the `ddda_ucg` islands the input carries (`--use-m5c`/`--no-use-m5c`,
`--cpg-mask-policy`; on for `--enzyme ddda`). `--daf-snp-mask` is not applied
automatically; pass the BED that `fiberhmm-call` wrote
(`qc/<stem>.daf_snps.bed`) to match its calls. As in `fiberhmm-call`, a
missing `--seq` is detected from the input, and `--prob-threshold` defaults
to the chemistry preset (248 for Hia5 Nanopore, 128 otherwise).

`-c/--cores` (4) workers process `--region-size` (5 Mb) regions; `-c 0` uses
every CPU. The model, ML threshold, chimera, SNP-mask, run-mask and CpG-mask
settings are recorded in the file's metadata, which `fiberhmm-check` reads.

Every option: [`fiberhmm-posteriors`](../reference/cli.md#fiberhmm-posteriors).
