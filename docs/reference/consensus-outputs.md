# Consensus output files

The files written by [`fiberhmm-consensus`](../workflows/consensus.md) (lattice
recaller) and [`fiberhmm-transfer`](../workflows/transfer.md). Coordinates are
reference, 0-based half-open, except pooled (`--pool-loci`) runs, which use
the oriented window frame `0 … width`.

## Run directory

```text
out/classes/
  classes.tsv          one row per class × channel
  molecules.tsv.gz     one row per class × channel × scored molecule
  broader.tsv.gz       molecules best explained by wider protection
  families.tsv         the standard report table (one row per class and dataset)
  calls.tsv            every native call and its class
  families.svg         class spans
  report.html          report (with report_data.json)
  manifest.json        parameters, mode, channels, warnings, timings, recaller summary
  evidence.json.gz     the complete prepared payload, for replay
  result.json.gz       the frozen result
  bams/                family-tagged BAMs (unless --no-bam)
```

With several BED windows the top level holds `report.html`, `regions.json`
(one entry per window: `name`, `output`, `status`, `seconds`) and one
`window_NNNNNN/` run directory per window. `bams/` holds
`NNN_<dataset>.families.bam` (CSI-indexed) and `bam_exports.json` (per-output
counts of annotations, matched and written alignments, scope and source
windows; and the exported families).

## `classes.tsv`

| Column | Meaning |
|---|---|
| `class_id`, `group` | class and its overlap group |
| `channel`, `dataset`, `strand` | `dataset::strand`; `strand` is `CT`/`GA` for DAF, `pooled` for Hia5 |
| `start`, `end` | class span (median discovery edges) |
| `L0`, `L1`, `R0`, `R1` | left and right edge boxes on this channel (contracted if edge contraction was kept) |
| `status`, `unscored_reason` | `supported` / `unsupported` / `unscored`, and why a pair is unscored |
| `calls`, `stability` | discovery calls in the class; prediction strength |
| `molecules` | molecules scored (for unscored rows: molecules in the window) |
| `prevalence`, `prevalence_edge`, `prevalence_loose` | core, edge and loose tiers |
| `prevalence_lower_bound` | Wilson 95% lower bound of the core prevalence |
| `broader`, `other_shape`, `accessible` | the remaining mixture weights |
| `support_gain_nats`, `supported` | held-out likelihood gain and the support verdict |
| `resolution_nats`, `resolved` | expected evidence per molecule and the resolution verdict |
| `spots` | learned internal spots, `position:rate` |
| `edge_contraction` | the kept contraction, or why it was rejected (`from <channel>: …` in a transfer) |
| `unknown_accessible_fraction` | the channel's accessible fraction for unknown sites |
| `efficiency` | per-channel efficiency factor, when `efficiency_calibration` is on |

Unscored rows leave the estimate columns empty.

## `molecules.tsv.gz`

| Column | Meaning |
|---|---|
| `class_id`, `channel`, `unit_id` | class, channel and molecule |
| `posterior` | EM class posterior |
| `log_bf` | log of posterior odds over prior odds |
| `label` | `member`, `non_member` or `abstain` (threshold `recaller.bf_threshold`) |
| `tier` | `core` for members; `edge` / `loose` for non-members counted in those tiers; empty otherwise |
| `start`, `end` | the molecule's own call for the class (lattice midpoint edges), when it has one |
| `edge_range` | `l0-l1,r0-r1`: how far each edge can move before a mark contradicts it |

## `broader.tsv.gz`

Molecules best explained (posterior ≥ 0.5) by protection wider than every
class of a group, for example a nucleosome over the site.

| Column | Meaning |
|---|---|
| `group`, `classes` | overlap group and its classes (`;`-separated) |
| `channel`, `unit_id`, `posterior` | channel, molecule, broader-protection posterior |
| `start`, `end`, `edge_range` | the best broader stretch and its edge ranges |

## `families.tsv` and `calls.tsv`

The standard report tables shared with FiberBrowser.

- `families.tsv`: `stage`, `dataset`, `family` (class id), `start`, `end`,
  `width`, `source_units` (members), `edge_uncertainty` (edge boxes as JSON),
  `fit_flags`, `classification_counts` (per channel: eligible units, i.e.
  molecules spanning the scoring window, compatible and primary units and
  calls), `trusted_strand`, `core_resolution`.
- `calls.tsv`: one row per native call: `chrom`, `genomic_start`,
  `genomic_end`, `stage`, `dataset`, `unit_id`, `read_name`, `source_start`,
  `source_end`, `compatible_families`, `status` (for example
  `lattice_unassigned`), `window`.

## `manifest.json`

Top-level keys include `schema`, `status`, `cr_mode` (`lattice_recaller`),
`region`, `parameters` (every group, with defaults filled in),
`input_digest`, `mode` / `mode_realized` (`CR`), `realized_channels`,
`data_warnings`, `seconds`, `datasets`, `input_files`, `pooling` (the pooling
receipt of a CL-CR run) and `recaller`:

- `classes`, `unsupported_classes`, `unscored_classes`, `unscored` (class,
  channel, reason, molecules);
- `dropped_by_core_rule` (span and core width of each dropped candidate);
- `tiles` and per-tile `discovery` diagnostics (k chosen, prediction
  strength per k, merges).

## `result.json.gz`

The frozen result FiberBrowser reads: schema `fiberhmm.consensus.v1`,
`cr_mode: lattice_recaller`, the manifest, the per-dataset catalog and
records, and `recaller`:

- `classes`: geometry and `status` of every class;
- `rows`: the scored rows of `classes.tsv`, with full-precision spot rates;
- `unscored`.

Each record has `proposals` (native calls with any class label) and
`recaller_calls`, the recaller's own per-molecule calls: a class call has
`tier` (core/edge/loose), `interval`, `lattice_interval`,
`consensus_interval`, `edge_range` and `edge_source` (`native` when a
labelled native call supplies the edges); a `broader` call is a
wider-protection stretch. A transfer adds a `transfer` block.

## Transfer files

| File | Content |
|---|---|
| `frozen_classes.json.gz` | lattice-recaller catalog, schema `fiberhmm.frozen_classes.lattice_recaller.v1`: `frame`, `parameters`, `tiles`, `classes`, `datasets`, `channels` (per-channel boxes and spots), `spot_precision`, `training_molecule_hashes`, `provenance`, `score_semantics`, `content_sha256` |
| `frozen_models.json.gz` | staged-engine bundle, schema `fiberhmm.frozen_families.v1` |
| `transfer_summary.tsv` | one row per window × class × channel: `window`, `chrom`, `window_start`, `window_end`, `window_strand`, `source_channel`, then every `classes.tsv` column |
| `transfer_manifest.json` | schema `fiberhmm.transfer_run.lattice_recaller.v1`: `catalog`, `catalog_sha256`, `catalog_schema`, `source_provenance`, `frame`, `refitted` (false), `rediscovered` (false), `preparation_parameters`, `dataset_map`, `include_training_molecules`, `windows`, `bams`, `apply_code` |
| `transfer_exclusions.json` | training molecules excluded from scoring |
| `chip_evaluation.json` | with `--chip-bed`: AUROC and average precision per class and channel |

The schema string `fiberhmm.consensus.v1` is shared by the lattice recaller
and the staged engine; FiberBrowser checks it, and `cr_mode` tells the two
engines apart.
