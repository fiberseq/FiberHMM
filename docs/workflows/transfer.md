# Transferring classes: `fiberhmm-transfer`

`fiberhmm-transfer` takes the footprint classes of a finished
[consensus run](consensus.md) and measures them in other molecules without
discovering or fitting classes again: new datasets at the same locus, or
other loci aligned to the same frame with oriented BED windows. It works for
both consensus engines.

## Two steps

```bash
# 1. Freeze a finished run
fiberhmm-transfer --freeze-run out/classes --output out/frozen
```

```text
complete: Exported 2 frozen classes over 1 channels
```

```bash
# 2a. Apply to saved evidence in the same frame
fiberhmm-transfer --models out/frozen --evidence out/classes/evidence.json.gz \
    --include-training-molecules --output out/transferred_self

# 2b. Apply to BAMs, one oriented BED6 window per target locus
printf 'chrDemo\t9900\t10250\tsite_plus\t0\t+\nchrDemo\t14000\t14350\tother_locus\t0\t-\n' > out/targets.bed
fiberhmm-transfer --models out/frozen --bam out/ont.calls.bam --bed out/targets.bed \
    --output out/transferred
```

```text
complete: 2 window(s) scored against 2 frozen classes
```

`--freeze-run` reads the run's engine and writes the matching file;
`--models` accepts that file or the whole `--freeze-run` output directory.

| Source run | Frozen file | Schema |
|---|---|---|
| lattice recaller (default) | `frozen_classes.json.gz` | `fiberhmm.frozen_classes.lattice_recaller.v1` |
| `staged_native_families`, oriented CL-CR run | `frozen_models.json.gz` | `fiberhmm.frozen_families.v1` |

Runs of any other engine, unknown schemas, and edited or truncated files are
rejected with an error that says why.

## Lattice-recaller catalogs

### What is frozen

The catalog is versioned JSON with a content digest and no executable
serialization. It holds:

- the **frame**: the run's region, or for an oriented pooled run the oriented
  window (`0 … width`) and its BED windows;
- the discovery tiles and every recaller and preparation parameter;
- **every discovered class**: id, span, pooled edge boxes, call count and
  stability. Unsupported classes are frozen too, because they are part of the
  EM mixture (removing one would change the prevalence of every other class in
  its group); they stay hidden in the catalog display;
- **per source channel** (`dataset::strand` with its chemistry), for each
  class: the final edge boxes (with jitter and any kept edge contraction), the
  learned spots at full precision, and the source estimates for reference;
- hashed identities of the training molecules;
- provenance: the source run path, its manifest and input digests, and the
  FiberHMM version, git commit (from a checkout) and recaller source hashes
  at freeze time. There is no timestamp, so freezing the same run twice with
  the same code gives the same digest.

### Fixed and re-estimated

Transfer runs the recaller's own quantification with discovery switched off;
each molecule goes through the same code path as in a normal run.

- **Fixed by the catalog**: the class set, spans and edge boxes, per-channel
  boxes and spot rates, the tiles, and every recaller option (linker, flank,
  weighting, BF threshold, support and resolution thresholds).
- **Estimated on the target**: EM prevalence and the three tiers, posteriors,
  per-molecule labels, edges and edge ranges, broader-protection stretches,
  held-out support and the `supported`/`resolved` flags, each channel's
  accessible fraction for unknown sites, and (only if the source run used it)
  per-channel efficiency.

A transfer may change only the `input`, `families` and `compute` parameter
groups (`--parameters`, `--cores`); these control molecule preparation and
execution. Any other group, such as `recaller`, is an error.

### Channel mapping

Frozen boxes and spots belong to a chemistry and strand. For each target
channel the source channel is:

1. `--dataset-map TARGET=SOURCE`, if given (chemistries must match);
2. otherwise the same `dataset::strand` with the same chemistry;
3. otherwise the one source channel with the same chemistry and strand (Hia5
   is `pooled`);
4. if several source datasets share that chemistry, the command stops and
   asks for `--dataset-map`;
5. if no source channel has that chemistry, the target is scored with the
   catalog geometry, the target chemistry's jitter and no learned spots.

The mapping is recorded as `transfer.channel_map` in `manifest.json` and
`result.json.gz`, and in the `source_channel` column of
`transfer_summary.tsv`.

### Frames and orientation

Catalog positions are coordinates in the source frame.

- `--evidence` must share that frame (same region start and end), for example
  evidence saved by `fiberhmm-consensus` for the same region or the same
  oriented pool.
- `--bam` with `--bed` needs BED6 windows of the frame's width with an
  explicit `+` or `-` strand. Each window is loaded with the shared BAM loader
  and mapped into the frame: the BED start (on `+`) or end (on `-`) goes to
  the frame start. Minus windows swap DAF strands. A `+` window with the
  coordinates of a single-region source run maps every base to itself.

Nothing is inferred: no motif search, offset search or resizing. Choose
anchors that make biological sense and the same assembly. Each BED window is
its own analysis with its own prevalences; to pool target loci, prepare
pooled evidence in the same oriented frame and pass `--evidence`.

### Training molecules

Molecules the catalog was trained on are excluded by default (matched by
physical molecule ids, read names, PacBio molecule names and source members),
listed in `transfer_exclusions.json` and counted in the transfer block.
`--include-training-molecules` scores them anyway; applying a run's catalog to
its own evidence with it reproduces the run's class rows,
`molecules.tsv.gz` and `broader.tsv.gz` exactly (a test checks this).

### Outputs

Each window is written as a normal lattice-recaller run (in the output
directory itself for one window or `--evidence`, otherwise in
`window_000001/`, …) with the same files and columns as
[`fiberhmm-consensus`](../reference/consensus-outputs.md).
`result.json.gz` keeps `schema: fiberhmm.consensus.v1` and
`cr_mode: lattice_recaller` and adds a `transfer` block (catalog digest and
schema, source provenance and frame, target window, channel and dataset maps,
excluded training molecules, code identity).

At the top level:

- `transfer_summary.tsv`: one row per window × class × channel, with the
  window coordinates and strand, the source channel and every `classes.tsv`
  column;
- `transfer_manifest.json` (schema
  `fiberhmm.transfer_run.lattice_recaller.v1`): catalog path and digest,
  source provenance, frame, preparation parameters, dataset map, each
  window's directory and input digest, and `refitted: false`,
  `rediscovered: false`;
- `bams/`: the family BAM export (off with `--no-bam`; `--bam-scope`,
  `--bam-grouping` as in `fiberhmm-consensus`; no `--bam-recaller-layer`);
- `chip_evaluation.json`, with `--chip-bed peaks.bed`: AUROC and average
  precision of each class and channel's per-window prevalence against peak
  overlap. These are descriptive, with no fitting and no uncertainty
  estimate.

FiberBrowser imports a `result.json.gz` only when its recorded browser
sources match the open datasets, and CLI runs record none. From the command
line, load the family BAM export in `bams/` into FiberBrowser instead; it
carries the class labels of native calls but not the recaller's own
per-molecule calls, which are in `result.json.gz`.

### Runtime

Transfer skips discovery (k-means, prediction strength, identity merging),
edge contraction and spot selection. On the NAPA N1 window (2,840 molecules,
8 classes, 3 channels) a full recaller run took 52 s on one core and transfer
17 s, with identical rows and molecule tables.

## Staged-engine bundles

For runs made with `--engine staged_native_families`, freeze an oriented
CL-CR run (`--pool-loci`) and score target windows the same way:

```bash
fiberhmm-transfer --freeze-run pooled_result --output frozen_catalog
fiberhmm-transfer --models frozen_catalog/frozen_models.json.gz \
    --bam target.bam --bed oriented_targets.bed --output transferred
```

The bundle is versioned JSON with a content digest, holding the exact native
or bounded-parent models (unconverged models are left out) and hashed
training-molecule identities; the source run must keep its native, source and
parent artifacts until export. Target windows must have the source width and
an explicit strand. Scoring uses the native transfer and bounded-parent
predictive kernels (4,095 Monte Carlo replicates, 99.9% reference); families
are never nominated or refitted. A molecule appearing twice in one window
fails the run. Outputs are `families.tsv` (eligible, assessed and compatible
counts per family and window), `calls.tsv`, `window_*.json.gz`,
`families.svg`, `report.html`, `manifest.json` and `bams/`. A molecule is
eligible when an aligned MSP covers the fitted mean and no nucleosome
overlaps it; the compatible fraction is an empirical feature, not a calibrated
ChIP occupancy.

Every option: [`fiberhmm-transfer`](../reference/cli.md#fiberhmm-transfer).
