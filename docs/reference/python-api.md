# Python API

FiberHMM is mainly used through its commands, but the functions below are
the documented Python entry points. Everything else under `fiberhmm.` is
internal and may change between releases.

## Package top level

```python
import fiberhmm
fiberhmm.__version__          # "3.0.0"
```

| Name | Signature | Purpose |
|---|---|---|
| `fiberhmm.load_model` | `(filepath, normalize=True) -> FiberHMM` | load a JSON (or `.npz`/`.pickle`) model |
| `fiberhmm.load_model_with_metadata` | `(filepath, normalize=True) -> (FiberHMM, context_size, mode)` | the model plus its context size and observation mode |
| `fiberhmm.save_model` | `(model, filepath, context_size=3, mode='pacbio-fiber')` | write a model as JSON |
| `fiberhmm.FiberHMM` | `(n_states=2)` | the two-state HMM: `predict` (Viterbi), `predict_proba` (posteriors), `fit`, `score` |
| `fiberhmm.ContextEncoder` | | builds and caches context lookup tables |
| `fiberhmm.FiberRead` | | one read with its modification calls and encoding |
| `fiberhmm.read_bam` | `(bam_path, region=None, min_mapq=20, prob_threshold=125, min_read_length=1000, mode='pacbio-fiber')` | iterate `FiberRead`s from a BAM |

Loading a `.pickle` model executes code; load only trusted files.

## Bundled models and defaults: `fiberhmm.models`

```python
from fiberhmm.models import get_model_path, default_prob_threshold

get_model_path("hia5", tool="apply", seq="nanopore")   # path to hia5_nanopore.json
get_model_path("ddda", tool="recall")                  # path to ddda_TF.json
default_prob_threshold("hia5", "nanopore")             # 248
default_prob_threshold("dddb", None)                   # 128
```

- `get_model_path(enzyme, tool='recall', seq=None)`: `tool` is `apply`
  (HMM), `recall` (TF recall) or `nuc_refine`.
- `default_prob_threshold(enzyme, seq, fallback=128)` and
  `resolve_prob_threshold(explicit, enzyme, seq, fallback=128)`: the
  chemistry-dependent ML threshold.
- `SUPPORTED_ENZYMES`: `("ddda", "dddb", "hia5")`.

## Re-run advisories: `fiberhmm.advisories`

```python
from fiberhmm.advisories import report, check_path, check_bam, check_header

report("calls.bam")["status"]      # "clean" | "not-fiberhmm" | "info" | "unverifiable" | "rerun-recommended" | "rerun-required" | "error"
check_bam("calls.bam", scan_records=0)   # list[Advisory], header only
```

Which fixes and default changes apply to an existing output; the JSON shape
and the evidence rules are in [Checking outputs for re-runs](advisories.md).

## Reading calls: `fiberhmm.io.ma_tags`

| Function | Purpose |
|---|---|
| `parse_ma_tag(ma_string) -> dict` | parse `MA:Z`: `read_length`, `raw_types` (name, strand, quality spec, intervals per group) and convenience keys `nuc`, `msp`, `tf`, `ddda_mcg`, `ddda_ucg` with 0-based `(start, length)` lists |
| `parse_aq_array(aq, qual_spec_per_type, n_annotations_per_type) -> list` | split the flat `AQ` array into one byte list per annotation |
| `flip_intervals_to_seq(starts, lengths, read) -> (starts, lengths)` | molecular frame to query frame (identity on forward reads) |

A complete example is in [Annotations and scores](../concepts/annotations.md#reading-the-output-in-python).

## Header declarations: `fiberhmm.io.bam_header`

| Function | Purpose |
|---|---|
| `declared_chemistries(header) -> list[dict]` | valid `FIBERHMM-CHEMISTRY` declarations, in order |
| `declared_ma_types(header) -> list[str]` | the union of `MA-TYPES` names |
| `append_ma_types(header, annotation_names)` | a header with one more `MA-TYPES` line for new names |
| `resolve_bam_chemistry(paths, requested)` | the consensus chemistry profile of a set of BAMs (errors on conflicts and unsupported enzymes) |

## Basecaller provenance: `fiberhmm.io.provenance`

| Function | Purpose |
|---|---|
| `basecaller_provenance(header, reads=None, override=None) -> dict` | basecaller program, version, basecalling model and modification models of a BAM, with the source of each (`@RG DS` > `@PG CL` > per-read read group; an override first). `modbase_models` is `None` when unknown and `[]` when none was used. See [Header declarations](headers.md#basecaller-provenance) |
| `bam_provenance(path, override=None) -> dict` | the same for a BAM file (header and first reads) |
| `parse_override(basecaller_info, modbase_model) -> dict` | the override from `--basecaller-info` / `--modbase-model` values |
| `ds_tokens(prov)` / `parse_ds_tokens(ds)` | the `basecaller=...` tokens FiberHMM writes into `@PG DS`, and back |

## Consensus

```python
from fiberhmm.inference.consensus.parameters import parse_options
from fiberhmm.inference.consensus.bam import load_bam_payload
from fiberhmm.inference.consensus.workflow import run_analysis

values = {"compute": {"cores": 2}}                 # parameter groups, as in --parameters
options = parse_options(values)                    # engine defaults to lattice_recaller
payload = load_bam_payload(
    [{"dataset_id": "demo", "paths": ["out/pacbio.calls.bam"]}],
    {"chrom": "chrDemo", "start": 9900, "end": 10250},   # 0-based, half-open
    options,
)
result = run_analysis(payload, values, "out/py_consensus")
result["cr_mode"]                                  # "lattice_recaller"
```

- `parse_options(values)` validates parameter groups (the same rules as the
  CLI, including rejection of knobs the engine ignores);
  `fiberhmm.inference.consensus.parameters.parameter_schema()` returns what
  `--schema` prints.
- `load_bam_payload(datasets, region, options=None, progress=None)` prepares
  the evidence. Pass `options`: without it the function prepares evidence
  for the deprecated staged engine.
- `run_analysis(payload, parameters=None, output_dir=None, progress=None)` runs
  the analysis and writes the run directory; it returns the result
  dictionary (`result.json.gz`). It does not write family BAMs.
- `fiberhmm.inference.consensus.bam_export.export_bams(analyses, output_dir, *, grouping='datasets', dataset_groups=None, progress=None, scope='regions', recaller_layer=False)`
  writes them, and `read_family_catalog(header)` reads the catalog back from
  an exported BAM's header (`{"contracts": [...], "families": [...]}`).

## Footprint population model

```python
from fiberhmm.inference import build_footprint_population_model
from fiberhmm.io import write_footprint_model_bundle, convert_footprint_model_bundle_to_bigbed
```

See [Footprint population model](../workflows/footprint-model.md#output-bundle).

## Posteriors

```python
from fiberhmm.posteriors.tsv_backend import parse_posteriors_line, tsv_to_h5
```

See [Posteriors](../workflows/posteriors.md).

## Consensus-state tagging

`fiberhmm.cli.tag_families.load_family_assignments(path)` validates an
assignment TSV, and `tag_tf_families(input_bam, output_bam, assignment_tsv, *, force=False)`
runs [`fiberhmm-tag-consensus`](../workflows/tag-consensus.md).
