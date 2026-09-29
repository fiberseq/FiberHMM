# Footprint-class consensus: `fiberhmm-consensus`

`fiberhmm-consensus` finds recurrent footprint classes in a window (or a set of
oriented windows) and measures, per chemical channel, how often each molecule
carries each class. FiberBrowser runs the same engine through the same entry
point (`run_analysis`) and the same BAM preparation (`load_bam_payload`).

The default engine is the **lattice recaller** (`cr.engine=lattice_recaller`).
It discovers class geometries from confident native calls, then scores every
molecule's own modification lattice against them with EM. It uses no Monte
Carlo. The earlier staged Monte Carlo engine (`--engine
staged_native_families`) is deprecated; it is described at the end of this
page. The historical call-harmonization engine can only be replayed through the
library API.

## Quick start

```bash
pip install -e '.[consensus]'
fiberhmm-consensus --bam calls.bam --region chr19:47514980-47515330 --output classes
fiberhmm-consensus --bam calls.bam --bed windows.bed --cores 4 --output classes
```

BAMs must be indexed. Chemistry comes from the `FIBERHMM-CHEMISTRY:v1:` BAM
`@CO` comments that `fiberhmm-call` writes. Supported legacy producer metadata
is also recognized. If the metadata is missing, or the BAM was called with a
custom `--model` and no `--enzyme`, supply `--chemistry ddda`, `dddb`,
`hia5-pacbio` or `hia5-nanopore`. An explicit setting cannot override
conflicting metadata. Consensus supports only these four chemistries. A BAM
that declares another enzyme (for example EcoGII) is rejected with an error:
Hia5 emissions are never substituted for another enzyme. Analysis never
modifies source BAMs.

Repeat `--bam` for separate datasets. To group several files as one dataset,
use `--datasets datasets.json`:

```json
[
  {"dataset_id": "replicate1", "paths": ["/data/a.bam", "/data/b.bam"]},
  {"dataset_id": "replicate2", "paths": ["/data/c.bam"]}
]
```

Each dataset can declare its `chemistry` when the metadata is absent. All files
in one dataset must have compatible chemistry. No read sampling or class-count
cap is used. MAPQ, native-opportunity and DAF molecule-collapse filters still
apply, and the manifest records them. Empty evidence gives an empty result, not
an invented class.

BED and `--region` coordinates are **zero-based, half-open**. Without
`--pool-loci`, each BED row is an independent analysis in genomic coordinates.
Windows are streamed: each is loaded, analysed and its BAM assignments recorded
before the next is loaded, so memory does not grow with the number of rows.
With more than one window, the output has an index `report.html`, `regions.json`
and one `window_000001`, `window_000002`, ... directory per row.

## How the lattice recaller works

**Channels.** A channel is one dataset × chemical strand. DAF strands (CT, GA)
are separate channels. Hia5 alignment orientations (including ONT) are pooled
into one channel, because orientation is not an independent chemical
measurement. Channels are labels, never features: one class catalogue is
discovered from every channel and dataset together, and each class is then
scored on each channel. The run's mode is therefore always **CR**, with classes
shared by construction. No SR boundary normalization or XCR relationship graph
is computed. `sr.enabled` and `cross.enabled` are accepted but change nothing.

1. **Evidence.** Native calls are replayed with the multi-interval decoder
   (`input.*`). Hia5 nucleosome/TF boundaries are replayed first unless
   `families.recall_hia5_nucleosomes=false`. The complete payload is saved as
   `evidence.json.gz`.
2. **Discovery (per tile).** The window is cut into tiles of `tile_bp`,
   `tile_step_bp` apart. The native calls used are those with
   LLR ≥ `call_min_llr` and width ≤ `call_max_bp` that lie entirely inside one
   tile. `call_max_bp` must not exceed the tile overlap
   (`tile_bp - tile_step_bp`); otherwise a wide footprint straddling an overlap
   would never be discovered, so this setting is rejected. Each call's edges are
   censored by the lattice: an edge is uncertain up to the nearest mark, capped
   at `censor_bp`. k-means clusters the censored edge midpoints, and k is the
   largest value whose split-half prediction strength reaches `stringency`.
   Overlapping candidates merge while two geometries beat one by less than
   `identity_nats` of held-out likelihood. Classes less stable than
   `stringency` are dropped. Each class gets left and right **edge boxes**
   (quantiles `edge_quantile_low`/`_high` of its calls' censored edge ranges)
   and a span (the median edges). The core rule drops classes whose core-rule
   boxes (`core_quantile_low`/`_high`) leave less than `minimum_core_bp` of
   protected DNA between them. Classes found in two overlapping tiles are
   deduplicated.
3. **Scoring (per overlap group × channel).** Classes whose spans overlap form
   a group. Molecules that span the group's edge boxes plus `flank_bp` are
   scored against these hypotheses: each class; broader protection (one
   interval covering every edge box); any other single protected interval; and
   all accessible. A class needs accessible linker DNA beyond both edges
   (`linker=both`) or at least one (`either`). EM gives the mixture weight
   (prevalence) and each molecule's posterior. Optional steps:
   - edge contraction (`edge_contraction`) pulls a box inward past sites that
     members almost always mark. The boxes are proposed on each validation
     fold's training half and scored on its held-out half.
   - learned internal spots (`learned_spots`) are interior positions that are
     sometimes marked while bound. They are kept only if they raise held-out
     likelihood.
4. **Verdicts.** A channel **supports** a class when including the class raises
   2-fold held-out likelihood by ≥ `support_gain_nats` and the prevalence's
   Wilson lower bound is ≥ `support_minimum_lower_bound`. A channel
   **resolves** a class when the expected evidence per molecule over the class
   span is ≥ `resolution_nats`. An unresolved class is still reported, but it is
   flagged.
5. **Per-molecule labels.** A molecule is a `member` when its posterior odds
   exceed the class's prior odds by `bf_threshold`, a `non_member` below the
   inverse, and otherwise `abstain`.

### Class status: supported, unsupported, unscored

| status | meaning | shown in the catalogue / BAM labels |
|---|---|---|
| `supported` | at least one channel supports it | yes |
| `unsupported` | scored on at least one channel, supported on none (e.g. short calls inside a nucleosome that scoring assigns to broader protection) | no, unless `recaller.report_unsupported_classes=true` |
| `unscored` | no channel could score it: no molecule spans its scoring window (edge boxes ± `flank_bp`), or a channel has fewer than `minimum_channel_units` molecules in the window. This is common at contig, amplicon or data ends. It is not a verdict on the class. | no (no estimate exists); listed with the reason per channel |

`manifest.json` lists `recaller.unsupported_classes`,
`recaller.unscored_classes` and `recaller.unscored` (class, channel, reason,
molecules). `classes.tsv` has one row per class × channel, including unscored
pairs.

### Prevalence tiers

Each class × channel has three prevalences, from conservative to loose:

- **core** (`prevalence`): the EM class weight, which is the mean class
  posterior. Both edges fit the class.
- **edge** (`prevalence_edge`): core plus non-member molecules whose class core
  is clean (unmarked, ≥ 3 nats of protection evidence) and whose protected run
  lines up with a class edge (inside the edge box, up to 5 bp into the core) on
  at least one side.
- **loose** (`prevalence_loose`): edge plus non-member molecules with a clean
  core under any protection.

"Non-member" means the per-molecule label is not `member` (the same
`bf_threshold` label as `molecules.tsv.gz`). The molecules a tier adds are
therefore exactly those that `molecules.tsv.gz` labels `edge` or `loose`. Each
tier is a coherent union: a qualifying molecule adds only its remaining
`1 - P` class mass, so
`tier = mean_i [P_i + (1 - P_i) × 1{molecule i is in the tier}] ≤ 1`.
Before 3.0, a full `1/n` was added per molecule, which counted `P_i` twice and
could exceed 1. The core tier is unchanged.

`prevalence_lower_bound` is the Wilson 95% lower bound of the core prevalence.
`broader`, `other_shape` and `accessible` are the remaining mixture weights.
None of these is a calibrated ChIP occupancy.

### Parameters

`fiberhmm-consensus --schema` prints every group with its defaults and help.
`--parameters options.json` sets them, for example:

```json
{
  "recaller": {"stringency": 0.8, "linker": "either", "jitter_hia5_bp": 10},
  "input": {"minimum_mapq": 20},
  "families": {"recall_hia5_nucleosomes": true},
  "compute": {"cores": 4}
}
```

The lattice recaller reads:

- the whole `recaller` group;
- `input.*`, except that `correct_native` must stay `true` (the recaller
  discovers and scores from replayed native calls and their LLRs);
- `families.recall_hia5_nucleosomes` and `families.nuc_split_minimum_llr`;
- `compute.cores`, `compute.maximum_region_bp` and `compute.maximum_matrix_mb`.

A non-default value anywhere else is rejected rather than silently ignored.
Examples are `cr.seed`, `families.physical_radius_bp`, `--consolidation-bp`,
`--stop-after`, `--cache` and device backends. The other engines likewise
reject non-default `recaller.*` values.

The `recaller` controls:

| control | default | effect |
|---|---|---|
| `stringency` | 0.9 | prediction strength needed for k, and class stability. Lower (0.6–0.7) finds more, finer classes |
| `kmax`, `prediction_splits`, `seed`, `minimum_candidate_calls` | 40, 6, 1, 5 | k-means search |
| `call_min_llr`, `call_max_bp` | 5.0, 100 | discovery calls (also the calls labelled in records) |
| `censor_bp` | 15 | edge censoring cap |
| `identity_nats`, `identity_folds`, `identity_pad_bp`, `identity_wide_bp`, `identity_max_width_bp` | 5, 3, 6, 40, 150 | held-out identity test that merges candidates |
| `edge_quantile_low`/`_high` | 10/90 | edge boxes. Use 25/75 for tighter boxes |
| `minimum_core_bp`, `core_quantile_low`/`_high` | 3, 25/75 | core rule (−1000 disables it) |
| `jitter_ddda_bp`, `jitter_dddb_bp`, `jitter_hia5_bp` | 0 | widen edge boxes outward per chemistry |
| `linker`, `linker_bp`, `flank_bp` | both, 5, 25 | scoring window and accessible-linker rule |
| `class_weighting` | bp | weight configurations by the edge positions they cover in the boxes, or uniformly |
| `edge_contraction`, `edge_contraction_rate`, `edge_minimum_members`, `edge_gain_nats` | off, 0.5, 20, 5 | per-channel edge contraction |
| `learned_spots`, `spot_*` | on | learned internal spots |
| `support_gain_nats`, `support_minimum_lower_bound` | 5, 0.02 | support verdict |
| `resolution_nats` | 5 | resolution verdict (and BAM strand trust) |
| `bf_threshold` | 3 | per-molecule member/non-member/abstain label |
| `report_unsupported_classes` | off | also catalogue unsupported classes |
| `efficiency_calibration` | off | scale each channel's accessible rate by its most-marked molecules |
| `tile_bp`, `tile_step_bp`, `minimum_channel_units` | 350, 250, 20 | tiling and minimum molecules per channel |
| `abutting` | off | **Experimental.** Calls a protected stretch that runs on past the class, with one edge in a class box, as the class with something abutting it. Its configuration weights are not a normalized prior: the class gains likelihood from the number of possible extensions even without chemical evidence. Runs that enable it get a warning and a `data_warnings` entry, and their prevalence and support are biased upward. |

## Outputs

Every run directory holds:

- `manifest.json`: parameters, mode (`CR`), realized channels,
  `data_warnings`, timings, and the `recaller` block (class counts,
  unsupported and unscored classes, tiles, per-tile discovery diagnostics).
- `evidence.json.gz`: the complete prepared payload, for replay.
- `result.json.gz`: the frozen result (schema `fiberhmm.consensus.v1`,
  `cr_mode: lattice_recaller`). It contains the per-dataset catalogue and
  records that FiberBrowser reads, and `recaller.classes` (geometry and
  `status`), `recaller.rows` (the scored rows of `classes.tsv`) and
  `recaller.unscored`. Each record has `proposals` (native calls with any class
  label) and `recaller_calls`: the recaller's own per-molecule calls. A class
  call has `tier` (core/edge/loose), `interval`, `lattice_interval`,
  `consensus_interval`, `edge_range` and `edge_source` (native when a labelled
  native call supplies the edges). A `broader` call is a wider-protection
  stretch.
- `classes.tsv`, `molecules.tsv.gz` and `broader.tsv.gz` (columns below).
- `families.tsv`, `calls.tsv`, `families.svg`, `report_data.json` and
  `report.html`: the standard report.
- `bams/`: family-tagged BAMs when the input was BAM (see below).

**`classes.tsv`**: one row per class × channel.

| column | meaning |
|---|---|
| `class_id`, `group` | class and its overlap group |
| `channel`, `dataset`, `strand` | `dataset::strand` (`pooled` for Hia5) |
| `start`, `end` | class span (median discovery edges) |
| `L0`, `L1`, `R0`, `R1` | left and right edge boxes on this channel (contracted if edge contraction was kept) |
| `status`, `unscored_reason` | `supported` / `unsupported` / `unscored`, and why a pair is unscored |
| `calls`, `stability` | discovery calls in the class; prediction strength |
| `molecules` | molecules scored (for unscored rows: molecules in the window) |
| `prevalence`, `prevalence_edge`, `prevalence_loose`, `prevalence_lower_bound` | the tiers and the core Wilson lower bound |
| `broader`, `other_shape`, `accessible` | remaining mixture weights |
| `support_gain_nats`, `supported` | held-out gain and the support verdict |
| `resolution_nats`, `resolved` | expected evidence per molecule and the resolution verdict |
| `spots`, `edge_contraction` | learned spots (`position:rate`); edge contraction record, or why it was rejected |
| `unknown_accessible_fraction`, `efficiency` | the channel's accessible fraction for unknown sites; efficiency factor if calibrated |

Unscored rows leave the estimate columns empty.

**`molecules.tsv.gz`**: one row per class × channel × scored molecule.

| column | meaning |
|---|---|
| `class_id`, `channel`, `unit_id` | class, channel and molecule |
| `posterior`, `log_bf`, `label` | EM class posterior; log posterior-over-prior odds; member / non_member / abstain |
| `tier` | `core` for members; `edge` / `loose` for non-members counted in those tiers; empty otherwise |
| `start`, `end` | the molecule's own call for the class (lattice midpoint edges), when it has one |
| `edge_range` | `l0-l1,r0-r1`: the ranges each edge can move before a mark contradicts it |

**`broader.tsv.gz`**: molecules best explained (posterior ≥ 0.5) by protection
wider than every class of a group, for example a nucleosome over it.

| column | meaning |
|---|---|
| `group`, `classes`, `channel`, `unit_id`, `posterior` | group, its classes (`;`), channel, molecule and broader-protection posterior |
| `start`, `end`, `edge_range` | the best broader stretch and its edge ranges |

Counts and fractions from different classes overlap. Do not sum them as
exclusive abundances.

## Cross-locus CR (CL-CR)

```bash
fiberhmm-consensus --bam calls.bam --bed oriented_windows.bed \
  --pool-loci --cores 4 --output pooled_classes
```

Provide BED6 with unique names, explicit `+`/`-` strands and equal window
widths. Choose the windows and orientations yourself: no motif lookup,
recentering or strand inference happens. A minus window maps base `i` to
`end-1-i` and interval `[a,b)` to `[end-b,end-a)`. Positions, hits,
context-specific emissions and m5C masks stay paired. The original genomic
window and molecule identities are retained. Local zero is the oriented
window's first base, and the pooled axis is `0..window_width`. To centre
motifs or TSSs, supply windows with the landmark at the same oriented offset.

Opportunity lattices stay individual observations and are not resampled onto a
dense average lattice. A physical molecule seen in several windows contributes
one deterministic window view, so evidence is not duplicated. The pooling
receipt lists the excluded views. This is not an occupancy estimator for every
locus. All windows are loaded together, because they are analysed as one.

## Replay

`--evidence evidence.json.gz` or `--resume run_dir` reruns an analysis on saved
evidence, for example with different `--parameters`. The recaller runs in one
pass: `--stop-after`, `--start-at consolidation`, `--consolidation-bp` and
`--cache` belong to the deprecated staged engine and are rejected before any
work starts. `--resume` addresses one window or pooled run directory, not a
batch parent. Progress goes to stderr (`--json-progress` gives structured
events). Output directories must be new or empty.

## Family-tagged BAMs

BAM input produces indexed derivative BAMs in `output/bams` by default.

- `--bam-scope regions` (the default) keeps whole alignments that overlap the
  analysed windows, including reads without class assignments.
  `--bam-scope full` keeps every source record.
- `--bam-grouping datasets` (the default) writes one BAM per logical dataset.
  `--bam-grouping files` writes one per source file.
- `--no-bam` keeps only reports and frozen artifacts.

Source BAMs are never overwritten. Native `tf`, `nuc`, MSP and other MA
annotations stay intact. Reruns replace this producer's own layers and header
catalogue. Unknown, changed or incompletely mapped source alignments fail the
export rather than being matched by read name. FiberBrowser's **Write
family-tagged BAMs** uses the same exporter.

**Lattice-recaller BAMs** carry:

- `tf_consensus.QQQQQQ` (never `tf_cross_consensus`: the recaller computes no
  XCR) with AQ bytes `tq,fi,fq,op,sq,q0`. A native call is labelled with a
  class only if the molecule is a member **and** the call fits the class:
  width ≤ `recaller.call_max_bp`, and its lattice-censored edges reach the
  class's edge boxes (widened by the chemistry's jitter). A member's
  nucleosome-sized call over a small class carries no label. AN is the class
  token (`fhcr_…`).
  - `q0` is the molecule's EM class posterior on its channel, `round(255·P)`
    (1–255 for a label; 0 = no class). It is a mixture posterior, not the
    staged engine's class-evidence share, and not a calibrated probability.
  - `tq` is the native call LLR ×10 (saturated at 255). `op` is the call's
    opportunities. `sq` is the DAF core protection ceiling, as in the staged
    engine. `fq = 0` means unavailable.
- **Strand trust.** Each DAF class's `FIBERHMM-CONSENSUS-FAMILY` entry has
  `strand_resolution` per dataset: `trusted_strand` (CT/GA/both/none),
  `trusted_strands`, `supported_strands`, per-strand `resolution_nats` and the
  threshold. A strand is trusted when its expected evidence per molecule over
  the class reaches `recaller.resolution_nats`. Use it to choose which strand
  to quantify. Hia5 classes carry no strand verdict, because orientations are
  pooled.
- **Class extent.** The FAMILY entry's `start`/`end` is the class consensus
  span (`extent: class_consensus_span`), not the union of the labelled calls.
- **Optional `tf_recaller.QQQQQQ` layer** (off by default: `--bam-recaller-layer`
  in the CLI, `recaller_layer=True` in the library `export_bams`; without it the
  header contract says the recaller calls are in `result.json.gz` only). It holds the recaller's own class calls at every
  tier, including molecules with no native call. AN is the same class token as
  `tf_consensus`. The AQ bytes (`layer_quality_names` in the header contract)
  are:
  - `tq`: native LLR ×10 when the edges come from a native call, 0 for lattice
    edges.
  - `fi`: slot.
  - `tier`: 1 core, 2 edge, 3 loose.
  - `q0`: class posterior ×255. Edge and loose calls are non-members, so it is
    usually low.
  - `lr`, `rr`: left and right edge-range widths in bp, saturated at 255.
    0 means an exact edge.

  The exact edge-range positions and the broader-protection stretches stay in
  `result.json.gz` (and `broader.tsv.gz`).
- The `FIBERHMM-CONSENSUS-MA:v1:` contract records the engine and the
  semantics of every byte. Recaller and staged results cannot share one
  export, because their `q0` semantics differ.

`read_family_catalog(bam.header)` in `fiberhmm.inference.consensus.bam_export`
reads the embedded catalogue without external tables.

## Transfer

`fiberhmm-transfer` freezes the classes of a finished run (either engine) and
scores other molecules against them without rediscovering classes: new datasets
at the same locus, or other loci aligned to the same frame with oriented BED6
windows. See [CONSENSUS_TRANSFER.md](CONSENSUS_TRANSFER.md) for the frozen-class
catalogue format, what stays fixed and what is re-estimated, and the outputs.

## Footprint-paired duplex molecules

Run `fiberhmm-pair` (pair → merge → recall, the default) on the called,
coordinate-sorted source BAM before population consensus. The merge recaller uses
both assay channels together, including the rotational nucleosome recaller.
Consensus preparation preserves the `cs` source identities and counts the
merged read once. The `deam+` and `deam-` MA coverage masks determine which C/G
opportunities are observed. Missing channel coverage and source deletions do
not become protected observations. The merged BAM retains `pm`, `dm`, `mg`
and `mv` pairing provenance.

Unmerged records with live `mt:P`/`mp` pair annotations cannot enter population
consensus as independent molecules. Merge preserves failed pairs by default, so
source data are not silently discarded; check its failure count.
`fiberhmm-pair --pairs-only` produces only successfully merged joint molecules.

## Validation on copied or synchronized source trees

Copied Python bytecode can retain filenames from a previous drive, and Numba
cache replacement can fail on synchronized Windows folders. Use fresh local
cache directories for release checks:

```bash
OPENBLAS_NUM_THREADS=1 \
PYTHONPYCACHEPREFIX=/tmp/fiberhmm-pycache \
NUMBA_CACHE_DIR=/tmp/fiberhmm-numba \
python -m pytest -q
```

## Deprecated engine: `staged_native_families`

`--engine staged_native_families` selects the staged Monte Carlo engine. It
fits native family distributions, fits shared parents, consolidates hypotheses
and resolves final representatives. Its fixed reference policy is 4,095
predictive draws, 10 folds, a bounded native edge allowance of 2 bp, 100 fit
iterations with a 500-iteration retry, and 99.9% predictive compatibility.
These are not confidence or accuracy percentages. Its mode follows the data
unless `sr.enabled`/`cross.enabled` are set explicitly: one Hia5 dataset uses
CR, DAF uses SR, and several datasets use XCR (SR/XCR with DAF strands). An
explicit value always wins.

It reads `families.*` (consolidation `physical_radius_bp`, default ±10 bp; use
5 bp for finer grouping), `input.*` (including `correct_native=false`, which
classifies the original BAM calls) and the compute controls. Non-default values
in its unused groups, including `recaller.*`, are rejected.

```bash
fiberhmm-consensus --engine staged_native_families --bam calls.bam --bed windows.bed \
  --stop-after native --output native_run
fiberhmm-consensus --resume native_run --start-at consolidation \
  --consolidation-bp 5 --output consolidated_run
```

Resume defaults to the final resolved stage. `--start-at consolidation`
requires exact native checkpoints and fails if any are missing or incompatible;
it never silently refits. `--evidence evidence.json.gz --cache fit_cache` also
permits replay. Source, evidence, implementation and numerical-library
signatures protect checkpoint reuse, so keep the cache with the run.

Its BAMs use `tf_consensus` (CR/SR) or `tf_cross_consensus` (XCR). Their
`q0` and `sq` bytes mean:

- `q0` is the class's share of the call's evidence among every class it was
  scored against, ×255: `w_k = exp(recipient_optimum_k − floor_adjusted_loss_k)`
  with a uniform prior. It is a relative profile-likelihood share, not a
  calibrated probability. 0 means unresolved.
- `sq` is the DAF molecule's own core protection ceiling (1 + LLR × 10). The
  family's `strand_resolution` is the strand verdict: a strand is limited when
  its ceiling is below the native floor. On HG002 scDAF duplexes, a limited
  strand's calls were confirmed by the complementary strand at only about 0.65
  precision.

Its FAMILY extent is the union of the labelled calls.
