# Footprint classes: `fiberhmm-consensus`

`fiberhmm-consensus` finds recurrent footprint **classes** in a window (for
example the sites where transcription factors bind), and measures, for each
chemical channel, how often molecules carry each class. Per-molecule calls
are noisy; a class is a footprint geometry that many molecules share, and
its prevalence is estimated from every molecule's own marks, including
molecules on which the per-read caller made no call.

FiberBrowser runs the same engine through the same entry point, so a class
found from the command line and one found in the browser are the same
computation.

## Quick start

```bash
fiberhmm-consensus --bam out/pacbio.calls.bam --region chrDemo:9900-10250 \
    --cores 2 --output out/classes
```

```text
recaller_discovery: tile 1/1 9900-10250: 178 molecules
complete: 2 classes; 2 class x channel estimates
bam_export: Writing MA family annotations to source BAM copies
```

```text
class_id   channel            start    end      status     molecules  prevalence  prevalence_edge  prevalence_loose
class_001  dataset_1::pooled  10041.0  10064.5  supported  176        0.5555      0.6776           0.7912
class_002  dataset_1::pooled  10110.0  10126.0  supported  176        0.4807      0.5386           0.5386
```

(selected columns of `out/classes/classes.tsv`; the demo planted a 24 bp
site at 10,040 with occupancy 0.7 and a 16 bp site at 10,110 with occupancy
0.5).

## Inputs

- **BAMs** called by FiberHMM, coordinate-sorted and indexed. Repeat
  `--bam` for separate datasets; group files with `--datasets datasets.json`:

    ```json
    [
      {"dataset_id": "replicate1", "paths": ["/data/a.bam", "/data/b.bam"]},
      {"dataset_id": "replicate2", "paths": ["/data/c.bam"], "chemistry": "ddda"}
    ]
    ```

    All files of one dataset must have compatible chemistry. A dataset
    called from `--bam` is named `dataset_1`, `dataset_2`, ….
- **Chemistry** comes from the BAMs' `FIBERHMM-CHEMISTRY` header lines (or
  a `fiberhmm-call` `@PG` record for older BAMs). Consensus supports `ddda`,
  `dddb`, `hia5-pacbio` and `hia5-nanopore`. For a BAM without chemistry
  metadata, or one called with a custom `--model` and no `--enzyme`
  (`enzyme=custom`), pass `--chemistry` (or a dataset's `chemistry`); it
  states which emission model to use and cannot override conflicting
  metadata. A BAM that declares another enzyme (for example EcoGII) is
  rejected; Hia5 emissions are never substituted for another enzyme.
- **Windows**: `--region CHROM:START-END`, or `--bed` (BED3, one independent
  analysis per row). Coordinates are **0-based, half-open**. A window may be
  at most `compute.maximum_region_bp` (50 kb).
- `--evidence evidence.json.gz` or `--resume RUN_DIR` replays saved evidence
  (below).

No reads are sampled and there is no cap on the number of classes. MAPQ
(`input.minimum_mapq`, 20), native-opportunity and DAF molecule-collapse
filters apply and are recorded in the manifest. Empty evidence gives an empty
result, not an invented class. Source BAMs are never modified; output
directories must be new or empty.

## How the lattice recaller works

The default engine, the **lattice recaller** (`cr.engine=lattice_recaller`),
discovers class geometries from confident native calls and then scores every
molecule's own lattice of marks against them with EM. It uses no Monte Carlo.

**Channels.** A channel is one dataset × chemical strand: the CT and GA
strands of DAF data are separate channels; Hia5 alignment orientations,
including ONT, are pooled into one channel (`pooled`), because orientation is
not an independent chemical measurement. Channels are labels, not features:
one class catalog is discovered from all channels and datasets together, and
each class is then scored on each channel.

1. **Evidence.** Native calls are replayed with the multi-interval TF decoder
   (`input.*` settings); for Hia5, nucleosome and TF boundaries are replayed
   first (`families.recall_hia5_nucleosomes`). The complete prepared payload
   is saved as `evidence.json.gz`.
2. **Discovery, per tile.** The window is cut into tiles of `tile_bp` (350)
   every `tile_step_bp` (250). Discovery uses native calls with LLR ≥
   `call_min_llr` (5.0) and width ≤ `call_max_bp` (100) that lie inside one
   tile. Each call's edges are censored by the lattice (an edge could move up
   to the nearest mark, capped at `censor_bp`, 15). k-means clusters the
   censored edge midpoints; k is the largest value whose split-half
   prediction strength reaches `stringency` (0.9). Overlapping candidates
   merge while two geometries beat one by less than `identity_nats` of
   held-out likelihood, and classes less stable than `stringency` are dropped.
   Each class gets left and right **edge boxes** (the `edge_quantile_low`/
   `_high`, 10/90, quantiles of its calls' censored edge ranges) and a span
   (the median edges). The core rule drops classes whose boxes (at the
   `core_quantile_low`/`_high` quantiles, 25/75) leave less than
   `minimum_core_bp` (3) of protected DNA. Classes found in two overlapping
   tiles are deduplicated. `call_max_bp` may not exceed the tile overlap
   (`tile_bp − tile_step_bp`); larger values are rejected because wide
   footprints straddling an overlap could not be discovered.
3. **Scoring, per overlap group × channel.** Classes whose spans overlap form
   a group. Every molecule that spans the group's edge boxes plus `flank_bp`
   (25) is scored against: each class; broader protection (one interval over
   every edge box, for example a nucleosome); any other single protected
   interval; and all-accessible. A class needs accessible linker DNA just
   beyond both edges (`linker=both`) or at least one (`either`). EM gives
   each class's mixture weight (its prevalence) and each molecule's
   posterior. Two optional refinements are kept only when they raise
   held-out likelihood: **edge contraction** (`edge_contraction`, off),
   which pulls an edge box inward past sites that members almost always mark
   on that channel; and **learned internal spots** (`learned_spots`, on),
   interior positions that are sometimes marked while the class is bound.
4. **Verdicts, per class × channel.** A channel **supports** a class when
   including it raises two-fold held-out likelihood by at least
   `support_gain_nats` (5) and the Wilson lower bound of its prevalence is at
   least `support_minimum_lower_bound` (0.02). A channel **resolves** a class
   when the expected evidence per molecule over the class span reaches
   `resolution_nats` (5); an unresolved estimate is reported but flagged.
5. **Per-molecule labels.** A molecule is a `member` when its posterior odds
   exceed the class's prior odds by `bf_threshold` (3), a `non_member` below
   the inverse, and otherwise `abstain`. Prevalence comes from EM and does
   not use the labels.

The run's mode is always CR: classes are shared by construction, and no SR
boundary normalization or XCR relationship graph is computed (`sr.enabled`
and `cross.enabled` are accepted and change nothing).

## Reading the results

### Class status

| Status | Meaning | In the catalog and BAM labels |
|---|---|---|
| `supported` | at least one channel supports it | yes |
| `unsupported` | scored on at least one channel, supported on none (for example short calls inside a nucleosome that scoring assigns to broader protection) | only with `recaller.report_unsupported_classes=true` |
| `unscored` | no channel could score it: no molecule spans its scoring window (edge boxes ± `flank_bp`), or a channel has fewer than `minimum_channel_units` (20) molecules. Common at contig, amplicon or data ends; not a verdict on the class | no; listed with the reason per channel |

`classes.tsv` has one row per class × channel, including unscored pairs
(with the reason and empty estimate columns). The manifest lists
`recaller.unsupported_classes`, `recaller.unscored_classes` and
`recaller.unscored`.

### Prevalence tiers

Each class × channel reports three prevalences, from strict to loose:

- **core** (`prevalence`): the EM class weight, the mean class posterior.
  Both edges fit the class.
- **edge** (`prevalence_edge`): core plus non-member molecules whose class
  core is clean (unmarked, with at least 3 nats of protection evidence) and
  whose protected run lines up with a class edge (inside the edge box, up to
  5 bp into the core) on at least one side.
- **loose** (`prevalence_loose`): edge plus non-member molecules with a clean
  core under any protection.

"Non-member" means the per-molecule label is not `member`. Each tier is a
coherent union: a qualifying molecule adds only its remaining `1 − P` of
class mass, so

```text
tier = mean over molecules of [ P_i + (1 − P_i) × 1{molecule i is in the tier} ]  ≤ 1
```

(before 3.0 a full `1/n` was added, which counted `P_i` twice and could
exceed 1). The molecules a tier adds are those that `molecules.tsv.gz` labels
`edge` or `loose`. `prevalence_lower_bound` is the Wilson 95% lower bound of
the core prevalence; `broader`, `other_shape` and `accessible` are the
remaining mixture weights.

None of these is a calibrated ChIP occupancy. Fractions from different
classes overlap; do not sum them as exclusive abundances.

The column-by-column description of `classes.tsv`, `molecules.tsv.gz`,
`broader.tsv.gz`, `result.json.gz` and `manifest.json` is in
[Consensus output files](../reference/consensus-outputs.md).

## Parameters

`fiberhmm-consensus --schema` prints every parameter group with defaults,
ranges and help as JSON. `--parameters options.json` sets them:

```json
{
  "recaller": {"stringency": 0.8, "linker": "either", "jitter_hia5_bp": 10},
  "input": {"minimum_mapq": 20},
  "compute": {"cores": 4}
}
```

The recaller reads the whole `recaller` group; `input.*` (except
`correct_native`, which must stay `true`); `families.recall_hia5_nucleosomes`
and `families.nuc_split_minimum_llr`; and `compute.cores`,
`compute.maximum_region_bp` and `compute.maximum_matrix_mb`. **A non-default
value anywhere else is rejected**, not silently ignored:

```text
fiberhmm-consensus: error: cr.seed is not used by the lattice recaller; reset it to its default (the recaller's own controls are in the recaller group)
```

The `recaller` group:

| Control | Default | Effect |
|---|---|---|
| `stringency` | 0.9 | prediction strength needed for k and for class stability; lower (0.6–0.7) finds more, finer classes |
| `kmax`, `prediction_splits`, `seed`, `minimum_candidate_calls` | 40, 6, 1, 5 | k-means search |
| `call_min_llr`, `call_max_bp` | 5.0, 100 | which native calls drive discovery (and can carry labels) |
| `censor_bp` | 15 | edge censoring cap |
| `identity_nats`, `identity_folds`, `identity_pad_bp`, `identity_wide_bp`, `identity_max_width_bp` | 5.0, 3, 6, 40, 150 | held-out identity test that merges candidates |
| `edge_quantile_low` / `_high` | 10 / 90 | edge boxes; 25/75 gives tighter boxes |
| `minimum_core_bp`, `core_quantile_low` / `_high` | 3, 25 / 75 | core rule (−1000 disables it) |
| `jitter_ddda_bp`, `jitter_dddb_bp`, `jitter_hia5_bp` | 0, 0, 0 | widen edge boxes outward per chemistry (about 10 helped Hia5 at one test site) |
| `linker`, `linker_bp`, `flank_bp` | `both`, 5, 25 | accessible-linker rule and scoring window; `either` suits footprints against a nucleosome and sparse lattices such as DddB |
| `class_weighting` | `bp` | weight configurations by the edge positions they cover (`bp`) or uniformly (`configurations`) |
| `edge_contraction`, `edge_contraction_rate`, `edge_minimum_members`, `edge_gain_nats` | off, 0.5, 20, 5.0 | per-channel edge contraction |
| `learned_spots`, `spot_minimum_evidence_nats`, `spot_gain_nats`, `spot_cap`, `spot_pseudo_units`, `spot_edge_bp`, `spot_iterations` | on, 10, 5, 0.35, 30, 5, 10 | learned internal spots |
| `support_gain_nats`, `support_minimum_lower_bound` | 5.0, 0.02 | support verdict |
| `resolution_nats` | 5.0 | resolution verdict (and DAF strand trust in BAMs) |
| `bf_threshold` | 3.0 | per-molecule member / non-member / abstain |
| `report_unsupported_classes` | off | also catalog unsupported classes |
| `efficiency_calibration` | off | scale each channel's accessible rate by its most-marked molecules |
| `tile_bp`, `tile_step_bp`, `minimum_channel_units` | 350, 250, 20 | tiling and minimum molecules per channel |
| `order_replicates`, `order_robust_fraction` | 0 (off), 1.0 | optional read-order robustness check (`--robust N`); see [Reproducibility](#reproducibility) |

`recaller.abutting` was removed in 3.0; old manifests with `abutting=false`
still load, and asking for `true` is refused with the alternatives (the
"+ edge" tier, or `linker=either`).

`--daf-mask-runs` / `--daf-run-policy` set adjacent-target thinning for every
DAF dataset (default per chemistry: DddA keep-one on runs ≥ 2, DddB off).
`--cores` sets `compute.cores` (default 4). `--json-progress` writes
structured progress events to stderr.

## Reproducibility

**Same inputs, same classes.** Since 3.0, the same BAMs with the same
parameters give identical classes on any machine (with single-threaded BLAS;
see [Environment variables](../reference/environment.md)), from any folder
and under any dataset names: a molecule is identified by its dataset's
position in the run, its file's position in that dataset and the alignment
record itself, never by a path or label. The order of the datasets and of
the files within a dataset is part of the input; reorder them and you have a
different (equally valid) run.

**Near-threshold classes can depend on read order.** Discovery is
deterministic, but the order of the molecules seeds k-means and decides the
split-halves and folds of its held-out tests. A class well above the
thresholds is found under any order; one close to them (a rare footprint, or
two geometries a few bp apart) may appear under some orders and not others.
On the 4 kb NAPA promoter window of FiberBrowser's demo (DddA-seq, hg38,
chr19:47514000-47518000, the window of the example below) the default order
finds 19 supported classes; other
orders find between 14 and 19, and `--robust 2` (like `--robust 4`) marks 12
of the 19 robust.

**`--robust N` marks which classes survive reordering.** It reruns discovery
and scoring under N other deterministic read orders and adds two columns to
`classes.tsv`: `order_robustness`, the fraction of the N+1 orders (the
default one included) in which a class of the same geometry was found and
supported, and `robust`, whether that fraction reaches
`recaller.order_robust_fraction` (default 1: every order). The classes, their
estimates and the per-molecule labels are always those of the default order;
the check only annotates them. It takes roughly N+1 times as long as
discovery alone (`--robust 2`: about 3×). Off by default.

```bash
fiberhmm-consensus --bam calls.bam --region chr19:47514000-47518000 --robust 2 --output out/classes_robust
```

FiberBrowser's *Check robustness to read order* option runs the same check
with N = 2.

## Several windows

```bash
printf 'chrDemo\t9900\t10250\nchrDemo\t14000\t14350\n' > out/windows.bed
fiberhmm-consensus --bam out/pacbio.calls.bam --bed out/windows.bed --cores 2 --output out/classes_bed
```

Each BED3 row is an independent analysis in genomic coordinates. Each window
is loaded, analysed and written by itself, so memory grows with the number of
windows running at once, not with the number of rows. The output has an index
`report.html`, `regions.json` and one `window_000001/`, `window_000002/`, …
directory per row, each a complete run.

Independent windows run in parallel: `--window-jobs N` analyses *N* windows at
a time, each with `--cores / N` workers (default `0` = automatic: up to
`--cores` windows at once; one at a time for the deprecated
`staged_native_families` engine, whose per-window `compute.maximum_matrix_mb`
budget is a hard limit). `--window-jobs 1` runs one window at a time with
every core, which suits a few large windows or memory-heavy runs (each
concurrent window holds its own evidence). Results do not depend on it. Each window logs to `logs/window_NNNNNN.log`; the terminal shows
windows done/total and an ETA (`--json-progress`, alias `--progress-json`: `stage: "windows"` events
with `completed`, `total`, `reused` and `eta_seconds`).

## Long runs and resuming

A BED with many windows is a restartable batch. Each window writes its outputs
into its own directory and, last, an atomic `unit_complete.json` marker
holding a digest of the run's inputs and parameters, the window and the
consensus code version, plus the size and SHA-256 of every file it wrote. The
run's contract (BAM and index path, size, modification time and content
SHA-256, windows, parameters, BAM-export options, and the DAF run mask in
force: `--daf-mask-runs`, or an inherited `FIBERHMM_DAF_RUN_MASK`) is saved as
`consensus_run.json`. A run holds a lock on its output directory while it runs,
so two runs never write the same windows. If a run is interrupted,
rerun the same command with `--continue`:

```bash
fiberhmm-consensus --bam calls.bam --bed sites.bed --cores 8 --output out/sites
# ... interrupted ...
fiberhmm-consensus --bam calls.bam --bed sites.bed --cores 8 --output out/sites --continue
```

- Windows whose marker matches and whose files still have their recorded
  digests are kept; missing, partial, damaged or altered windows and windows
  computed by other code are rerun.
- `regions.json`, the top-level `report.html` and the family-tagged BAMs are
  rebuilt from all completed windows at the end, so the outputs are the same
  whether windows ran one at a time, in parallel, or across an interrupted
  and continued run (apart from timing fields and the per-window
  `compute.cores`).
- Different BAMs (or a modified BAM or index, even one whose size and date
  were preserved), windows, parameters or DAF run mask are refused with the
  fields that differ. Only `--cores`, `--window-jobs` and
  `--json-progress` may change between attempts.
- Continuing a finished run reruns nothing and rebuilds the aggregates.
- Without `--continue` the output directory must be empty, as always.
- A failing window stops the run and the other windows' workers; completed
  windows are kept for `--continue` once the cause is fixed. The failing
  window's log is named in the error.

!!! note "`--continue` versus `--resume`"
    `--continue` finishes an interrupted multi-window run **in place**, in
    its own `--output`. `--resume RUN_DIR` starts a **new** analysis, in a new
    `--output`, from one finished window or pooled run's saved evidence (for
    example with different `--parameters`; see [Replay](#replay)).
    `--continue` does not apply to `--pool-loci`, `--evidence` or `--resume`
    runs, which are single analyses: rerun them.

## Pooling loci (CL-CR)

```bash
printf 'chrDemo\t9900\t10250\tsite_a\t0\t+\nchrDemo\t14000\t14350\tsite_b\t0\t-\n' > out/oriented.bed
fiberhmm-consensus --bam out/pacbio.calls.bam --bed out/oriented.bed --pool-loci \
    --cores 2 --output out/classes_pooled
```

With `--pool-loci`, BED6 windows with unique names, explicit `+`/`-` strands
and equal widths are analysed as one population in a shared oriented frame
`0 … width`: a minus window maps base *i* to `end − 1 − i` and interval
`[a, b)` to `[end − b, end − a)`. Positions, hits, context-specific emissions
and m5C masks stay paired, and DAF strands swap on minus windows. You choose
the windows and orientations (put the landmark, a motif or TSS, at the same
oriented offset); nothing is inferred. Lattices stay individual observations
and are not resampled. A molecule seen in several windows contributes one
deterministic view; the pooling receipt lists the excluded views. The result
is not an occupancy estimate for any one locus. All windows are loaded
together.

## Replay

`--evidence evidence.json.gz` or `--resume RUN_DIR` reruns an analysis on
saved evidence, for example with different `--parameters`, without reading
the BAMs again. `--resume` takes one window or pooled run directory, not a
multi-window parent (to finish an interrupted multi-window run, use
[`--continue`](#long-runs-and-resuming)). The recaller runs in one pass, so `--stop-after`,
`--start-at`, `--consolidation-bp` and `--cache` (staged-engine options) are
rejected before any work starts:

```text
fiberhmm-consensus: error: --stop-after applies to --engine staged_native_families only; the lattice recaller discovers and scores classes in one pass (every run saves evidence.json.gz for replay with --evidence or --resume)
```

## Family-tagged BAMs

BAM input produces indexed derivative BAMs in `<output>/bams/`
(`001_dataset_1.families.bam` …, plus `bam_exports.json`):

- `--bam-scope regions` (default) keeps whole alignments overlapping the
  analysed windows, including reads with no class; `full` keeps every source
  record.
- `--bam-grouping datasets` (default) writes one BAM per dataset; `files` one
  per source file.
- `--no-bam` writes only the reports and frozen results.

Source BAMs are never overwritten, and native `nuc`, `msp`, `tf` and other
`MA` groups stay intact. A re-export replaces this producer's own layers and
header entries. Alignments that cannot be matched exactly to the analysed
records fail the export rather than being matched by read name.

The lattice-recaller export adds:

- **`tf_consensus.QQQQQQ`**: native TF calls labelled with their class. A
  native call gets a label only if the molecule is a member **and** the call
  fits the class (width ≤ `call_max_bp`, lattice-censored edges reaching the
  class edge boxes, widened by the chemistry's jitter); a member's
  nucleosome-sized call over a small class carries no label. Bytes
  `tq, fi, fq, op, sq, q0`: native LLR ×10, class slot, `fq` (0 =
  unavailable), opportunities, the DAF core protection ceiling, and `q0` =
  the molecule's EM class posterior ×255 (1–255 for a label; 0 = no class).
  `q0` is a mixture posterior, not a calibrated probability. `AN` holds the
  class token (`fhcr_…`).
- **`tf_recaller.QQQQQQ`**, only with `--bam-recaller-layer`: the recaller's
  own class calls at every tier, including molecules with no native call.
  Bytes `tq, fi, tier, q0, lr, rr`: native LLR ×10 when the edges come from a
  native call (0 for lattice edges), class slot, tier (1 core, 2 edge,
  3 loose), class posterior ×255 (usually low for edge/loose calls, which are
  non-members), and left/right edge-range widths in bp (0 = exact edge).
  Without the option, these calls are only in `result.json.gz`.
- **Header**: one `FIBERHMM-CONSENSUS-FAMILY:v1:` line per class and layer,
  with the class consensus span (`extent: class_consensus_span`) and, for
  DAF classes, a per-dataset `strand_resolution` with `trusted_strand`
  (`CT`, `GA`, `both` or `none`): a strand is trusted when its expected
  evidence per molecule over the class reaches `recaller.resolution_nats`.
  Hia5 classes carry no strand verdict because orientations are pooled. One
  `FIBERHMM-CONSENSUS-MA:v1:` line records the engine and the meaning of
  every byte. See [Header declarations](../reference/headers.md#consensus-headers).

`read_family_catalog(bam.header)` in
`fiberhmm.inference.consensus.bam_export` reads the embedded catalog without
external tables. FiberBrowser's **Write family-tagged BAMs** uses the same
exporter. Recaller and staged-engine results cannot share one export because
their `q0` means different things.

## Paired duplex molecules

Run [`fiberhmm-pair`](duplex.md) on a DddA BAM before consensus. The merged
molecules keep their `cs` source identities and count once; the `deam+` and
`deam-` coverage decides which C/G positions are observed, and missing
channel coverage or source deletions do not become protected observations.
Source reads that still carry live `mt:P`/`mp` tags without being merged
cannot enter consensus as independent molecules; use
`fiberhmm-pair --pairs-only` to keep only merged molecules.

## Transfer

[`fiberhmm-transfer`](transfer.md) freezes the classes of a finished run and
measures them in other datasets or at other loci without rediscovery.

## Deprecated engine: `staged_native_families`

`--engine staged_native_families` selects the earlier staged Monte Carlo
engine. It fits native family distributions, shared parents, consolidates
hypotheses and resolves final representatives, using 4,095 predictive draws,
10 folds, a 2 bp native edge allowance, 100 fit iterations (500 on retry) and
99.9% predictive compatibility (these are not confidence or accuracy
percentages). Its mode follows the data unless `sr.enabled`/`cross.enabled`
are set: one Hia5 dataset uses CR, DAF uses SR, several datasets XCR.

It reads `families.*` (for example `physical_radius_bp`, ±10 bp, or
`--consolidation-bp 5` for finer grouping), `input.*` (including
`correct_native=false`, which classifies the original BAM calls) and the
compute controls, and rejects non-default `recaller.*` values. It can stop
and resume between stages:

```bash
fiberhmm-consensus --engine staged_native_families --bam calls.bam --bed windows.bed \
    --stop-after native --output native_run
fiberhmm-consensus --resume native_run --start-at consolidation --consolidation-bp 5 \
    --output consolidated_run
```

`--start-at consolidation` requires exact native checkpoints and fails rather
than refitting; `--cache DIR` keeps a persistent native-fit cache. Its BAMs
use `tf_consensus` (CR/SR) or `tf_cross_consensus` (XCR), where `q0` is the
class's share of the call's evidence among the classes it was scored against
(×255; 0 = unresolved) and `sq` the DAF molecule's own core protection
ceiling; its family extent is the union of the labelled calls.
`fiberhmm[cuda]` provides GPU predictive kernels for this engine only.
