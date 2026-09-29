# Apply frozen classes to new data (`fiberhmm-transfer`)

`fiberhmm-transfer` takes the footprint classes found by one consensus run and
scores other molecules against them without discovering or fitting classes
again. Use it to measure the same classes in new datasets at the same locus, or
at other loci that you align to the same frame with oriented BED windows.

It is a two-step command. The form is the same for both consensus engines:

```bash
# 1. Freeze a completed consensus run
fiberhmm-transfer --freeze-run consensus_run --output frozen

# 2a. Apply to saved evidence in the same frame
fiberhmm-transfer --models frozen --evidence target/evidence.json.gz --output transferred

# 2b. Apply to BAMs, with one oriented BED6 window per target locus
fiberhmm-transfer --models frozen --bam target.fiberhmm.bam --bed oriented_targets.bed --output transferred
```

`--freeze-run` checks the run's `cr_mode` and writes the matching file:

| Source run | Frozen file | Applied with |
|---|---|---|
| `lattice_recaller` (the default engine) | `frozen_classes.json.gz`, schema `fiberhmm.frozen_classes.lattice_recaller.v1` | recaller quantification with fixed classes |
| `staged_native_families` (`--engine staged_native_families`, oriented CL-CR run) | `frozen_models.json.gz`, schema `fiberhmm.frozen_families.v1` | native transfer and bounded-parent predictive kernels |

`--models` accepts either the frozen file or the `--freeze-run` output folder.
The command reads the schema to choose the scorer. Runs from any other engine,
unknown schemas, and edited or truncated files are rejected with an error that
says why.

## Lattice recaller catalogs

### What the catalog contains

The catalog is versioned JSON with a content digest. It has no executable
serialization. It holds:

- **Frame.** The run's analysis region. For an oriented CL-CR pool (`--pool-loci`)
  this is the oriented window (0 to width) and its BED windows. For a single-region
  run it is the reference region.
- **Discovery tiles, and every recaller and preparation parameter of the run.**
- **Every discovered class.** Each class has its id, span, pooled left and right
  edge boxes, call count and stability. Classes that no channel supported are
  frozen too, because they are part of the EM mixture. Removing one would change
  the prevalence of every other class in its overlap group. They stay hidden in
  the catalog display, as in the source run.
- **Per source channel** (`dataset::strand`, with its chemistry). For each class:
  - the final edge boxes, with chemistry jitter and any kept edge contraction applied;
  - the learned internal spots (positions and protected-state mark rates) at full precision;
  - the source estimates, for reference (prevalence tiers, support, resolution and molecule count).
- **Hashed identities of the training molecules.**
- **Provenance.** The source run path, the digest of its manifest, its input
  digest and mode, and the FiberHMM version, git commit (when frozen from a
  checkout) and recaller source hashes at freeze time. There is no timestamp, so
  freezing the same run twice with the same code gives the same digest. The recaller run manifest
  does not record which code version produced the run, so the catalog records
  the code that froze it.

Runs made before full-precision spot rates were stored in `result.json.gz`
(field `spot_rates`) are frozen from the 3-decimal `spots` column. The catalog
then reports `spot_precision: 3_decimals`.

### What is fixed and what is re-estimated

Transfer runs the recaller's own quantification (`quantify`) with discovery
turned off. The code path for each molecule is the same as in a normal run.

- **Fixed by the catalog:** the class set, class spans and edge boxes, the
  per-channel boxes and learned spot rates, the tiles, and all recaller options
  (linker, flank, weighting, abutting, BF threshold, support and resolution
  thresholds).
- **Estimated from the target molecules:**
  - EM prevalence and the three prevalence tiers;
  - posteriors, per-molecule labels, edges and edge ranges;
  - broader-protection stretches;
  - held-out support gain and the `supported` and `resolved` flags;
  - each channel's unknown-state accessible fraction;
  - per-channel efficiency, only if the source run enabled `efficiency_calibration`.

  These describe the target population and its assay, not the classes.

A transfer may change only the `input`, `families` and `compute` parameter
groups (`--parameters`, `--cores`). These control molecule preparation and
execution. Passing any other group, such as `recaller` or `cr`, is an error.

### Channel mapping

Frozen boxes and spots belong to a chemistry and strand. For each target channel
the source channel is chosen as follows:

1. `--dataset-map TARGET=SOURCE`, if given. Chemistries must match.
2. Otherwise, the same `dataset::strand` with the same chemistry.
3. Otherwise, the one source channel with the same chemistry and strand.
   Hia5 alignment directions are pooled, so the strand is `pooled`.
4. If several source datasets share that chemistry, the command stops and asks
   for `--dataset-map`. It never picks one silently.
5. If no source channel has that chemistry, the target channel is scored with
   the catalog class geometry, the target chemistry's jitter and no learned spots.

The mapping is recorded as `transfer.channel_map` in `manifest.json` and
`result.json.gz`, and as the `source_channel` column of `transfer_summary.tsv`.
When the boxes come from another channel, the `edge_contraction` column says
`from <source channel>: ...`.

### Frames, loci and orientation

Positions in the catalog are coordinates in the source frame.

- **`--evidence`** must share that frame: its region start and end must equal the
  catalog's. Evidence saved by `fiberhmm-consensus` for the same region, or for the
  same oriented pool, qualifies.
- **`--bam` with `--bed`** needs BED6 windows of the frame's width, each with an
  explicit `+` or `-` strand. Each window is loaded with the shared BAM loader and
  mapped into the frame. The BED start (on `+`) or BED end (on `-`) goes to the
  frame start, and a nonzero frame start is handled. Reverse windows swap DAF
  strands, as in CL-CR pooling. A `+` window with the same coordinates as a
  single-region source run maps every base to itself.

You choose anchors that make biological sense (a motif, a TSS, the same locus in
another sample) and the same reference assembly. Nothing is inferred
automatically: no motif search, no offset search, and no resizing. Each BED window
is its own analysis with its own prevalences. To pool target loci into one
estimate, prepare a pooled evidence file in the same oriented frame and pass it
with `--evidence`.

### Training molecules

Molecules whose identities match the training molecules are excluded by default,
as in staged transfer. The match uses physical molecule IDs, read names, PacBio
molecule names and source members. Excluded molecules are listed in
`transfer_exclusions.json` and counted in the transfer block.
`--include-training-molecules` scores them anyway. Use it for self-application
checks: applying a run's catalog to the run's own evidence with this flag gives
the same class rows, `molecules.tsv.gz` and `broader.tsv.gz` as the run. The test
suite checks this.

### Outputs

Each window is written as a normal lattice-recaller run. With one window or
`--evidence`, it goes in the output folder itself. With several windows, it goes
in `window_000001/`, `window_000002/` and so on.

- `classes.tsv`, `molecules.tsv.gz`, `broader.tsv.gz`, `result.json.gz`,
  `manifest.json`, `evidence.json.gz`, and the standard `report.html`,
  `families.tsv` and `calls.tsv`. The columns are the same as a recaller run.
  `result.json.gz` keeps `schema: fiberhmm.consensus.v1` and
  `cr_mode: lattice_recaller`, and adds a `transfer` block with:
  - the catalog digest and schema;
  - the source provenance and frame, and the target window;
  - the channel map and the dataset map;
  - the number of excluded training molecules;
  - the code identity at apply time.
- `transfer_exclusions.json`, written when training molecules were excluded.

The top level also has:

- `transfer_summary.tsv`: one row per window, class and channel, with the window
  coordinates, the source channel and every `classes.tsv` column.
- `transfer_manifest.json` (schema `fiberhmm.transfer_run.lattice_recaller.v1`):
  - the catalog path and digest, the source provenance and the frame;
  - the preparation parameters and the dataset map;
  - each window's output folder and input digest;
  - `refitted: false` and `rediscovered: false`.
- `bams/`: the shared family BAM export, only when inputs came from BAMs. Turn it
  off with `--no-bam`. The export options are the same as for `fiberhmm-consensus`.
- `chip_evaluation.json`, with `--chip-bed`. It gives the AUROC and average
  precision of each class and channel's per-window EM prevalence against overlap
  of a peak with the window. The metrics are descriptive, with no fitting and no
  uncertainty estimate.

`result.json.gz` has the recaller's format, schema and `cr_mode`. FiberBrowser's
result import accepts only results whose recorded Browser sources match the
open datasets. CLI runs record none, whether transfer or a normal recaller run.
From the CLI, reload the family BAM export in `bams/` instead. That export
carries the class labels of native calls. It does not carry the recaller's own
per-molecule calls, which are in `result.json.gz` only.

### Runtime

Transfer skips discovery (k-means, prediction strength and identity merging),
edge contraction and spot selection. Only the EM mixture and its held-out
support are computed. On the planted test fixture, a full recaller run took
2.0 s and transfer to the same molecules took 0.4 s (1 core). On the NAPA N1
window (2,840 molecules, 8 classes, 3 channels), a full run took 52 s on 1 core
and 38 s on 4 cores. Transfer took 17 s on either, with identical rows and
molecule tables.

## Staged-family bundles (`staged_native_families`)

These are the steps for runs made with `--engine staged_native_families`. They
have not changed.

Export the final converged models from an oriented CL-CR run
(`--pool-loci`), then score target windows:

```bash
fiberhmm-transfer --freeze-run pooled_result --output frozen_catalog
fiberhmm-transfer --models frozen_catalog/frozen_models.json.gz \
  --bam target.fiberhmm.bam --bed oriented_targets.bed --output transferred
```

The model bundle is versioned JSON with a content digest. Numerical arrays are
typed explicitly, and there is no executable Python serialization. The bundle
contains:

- the exact native-cell or bounded-parent model;
- hashed training-molecule identities.

Final displayed aliases are resolved to the models that were actually fitted.
Unconverged models are left out. The source CL-CR run must keep its native,
source and parent artifacts until the export step. After that, the bundle is
self-contained.

Target BED6 windows must give an explicit `+` or `-` orientation and must have
the same width as the source analysis. The BED start on `+` and the BED end on
`-` define the shared oriented frame. Older bundles may declare a nonzero
analysis origin. Transfer handles it explicitly. You choose anchors that make
biological sense and reference assemblies that match. No gene, motif or RNA
landmark is inferred automatically.

Input options:

- **Multiple `--bam` arguments** are separate datasets.
- **`--datasets`** takes the same dataset JSON as `fiberhmm-consensus` and allows
  several paths per dataset.
- **BAM preparation** uses the shared native loader, chemistry comments and
  native replay. `--parameters` exposes its preparation and memory controls.
- **`--evidence`** takes saved oriented native observations and replays them
  exactly. It does not reread BAMs or regenerate their emissions.
- **`--json-progress`** writes structured progress updates to stderr.

Scoring uses the existing native transfer and bounded-parent predictive kernels
with 4,095 Monte Carlo replicates and the established 99.9% reference policy.
Target families are never nominated, consolidated or refitted.

Training molecules are excluded, matched by physical identifiers, PacBio
molecule names and source-member aliases. If the same molecule appears twice in
a target window, transfer fails rather than inflating the denominators.
Observations from different windows are reported separately and are not claimed
to be independent.

Outputs:

- families.tsv: eligible, assessed and compatible molecule counts per family and window.
- calls.tsv: original spans and all compatible family identities.
- window_*.json.gz: complete scores, reasons calls were not assessed, and exclusions.
- families.svg and report.html: frozen mean spans and summary tables.
- manifest.json: frozen-model digest, input digests and source provenance.
- bams/: indexed MA/AQ/AN subset exports, written by default:
  - `--no-bam` turns them off;
  - `--bam-scope full` keeps the full source BAMs;
  - `--bam-grouping files` writes one export per file.

A molecule is in the eligible denominator when an aligned MSP covers the fitted
mean and no nucleosome overlaps it. An eligible molecule can still carry too
little information for a predictive decision, so assessed counts are reported
separately. The compatible fraction is an empirical feature. It is not a
calibrated ChIP occupancy probability.

You can supply `--chip-bed peaks.bed` to check against an external label.
Overlap of a peak with the supplied genomic window is joined only after scoring.
`chip_evaluation.json` reports the AUROC and average precision of the direct
compatibility fraction for each dataset and family, when both label classes
exist. These metrics are descriptive, with no confidence intervals and no target
fitting. Choose held-out windows and suitable matched controls before reading
them as discrimination. The paper's frozen logistic predictor and genomic-block
bootstrap are separate analyses. This command does not fit a new classifier.
