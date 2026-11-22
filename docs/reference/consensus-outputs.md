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
  read_posteriors.tsv.gz  with --read-table: every read x class posterior
```

With `--read-table`, BED/`--region` runs also write the merged `read_table/`
directory; see [Per-read tables, cross-fitting and role rules](consensus-read-table.md).

With several BED windows the top level holds `report.html`, `regions.json`
(one entry per window: `name`, `output`, `status`, `seconds`) and one
`window_NNNNNN/` run directory per window. Every BAM + BED/`--region` run also
writes `consensus_run.json` (the run contract `--continue` checks, with one
entry per attempt), `logs/window_NNNNNN.log` (one log per window) and, in each
window directory (the output directory itself for a single window),
`unit_complete.json`: the completion marker written last, with the unit
digest, window, status, seconds and the size and SHA-256 of every file the
window wrote.
See [Long runs and resuming](../workflows/consensus.md#long-runs-and-resuming). `bams/` holds
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
| `calls`, `stability` | discovery calls in the class; its prediction strength, averaged over the split-halves |
| `molecules` | molecules scored (for unscored rows: molecules in the window) |
| `prevalence`, `prevalence_edge`, `prevalence_loose` | core, edge and loose tiers |
| `prevalence_lower_bound` | Wilson-formula lower bound using `molecules × prevalence`; a heuristic support screen, not an uncertainty-calibrated interval for latent occupancy |
| `broader`, `other_shape`, `accessible` | the remaining mixture weights |
| `support_gain_nats`, `supported` | held-out likelihood gain and the support verdict |
| `resolution_nats`, `resolved` | median span `KL(p_accessible || p_protected)` over molecules with sites spanning the span ±30 bp, and the resolution verdict; computed before channel edge contraction |
| `spots` | learned internal spots, `position:rate` |
| `edge_contraction` | the kept contraction, or why it was rejected (`from <channel>: …` in a transfer) |
| `unknown_accessible_fraction` | the channel's accessible fraction for unknown sites |
| `efficiency` | per-channel efficiency factor, when `efficiency_calibration` is on |
| `median_span_opportunities` | median usable target count in `[start, end)` among the actual scored molecules |
| `median_left_flank_opportunities`, `median_right_flank_opportunities` | median usable target count in `[L0 − flank_bp, L0)` and `[R1, R1 + flank_bp)` among scored molecules |
| `scored_span_accessible_nats`, `scored_span_protected_nats` | median summed span KL in the accessible→protected and protected→accessible directions, respectively, among scored molecules |
| `order_robustness`, `robust` | only with `--robust N` (`recaller.order_replicates`): the fraction of the N+1 read orders in which the class was found and supported, and whether that reaches `recaller.order_robust_fraction` (default 1: every order). See [Reproducibility](../workflows/consensus.md#reproducibility) |

Unscored rows leave the estimate columns empty.

The opportunity summaries use the actual scored denominator and final channel
edge boxes. This differs from `resolution_nats`'s spanning-molecule set. Both
KL directions use the raw model emissions, without learned spots. These are
observability diagnostics, not extra support thresholds, a guarantee of edge
resolution, or an empirical calibration test. See
[Comparing haplotypes and samples](../workflows/consensus-comparisons.md).

## `molecules.tsv.gz`

| Column | Meaning |
|---|---|
| `class_id`, `channel`, `unit_id` | class, channel and molecule |
| `posterior` | EM class posterior |
| `log_bf` | log class likelihood divided by the fitted mixture likelihood of all other hypotheses, with alternative weights renormalized |
| `label` | `member`, `non_member` or `abstain` (threshold `recaller.bf_threshold`) |
| `tier` | `core` for members; `edge` / `loose` for non-members counted in those tiers; empty otherwise |
| `start`, `end` | the molecule's own call for the class (lattice midpoint edges), when it has one |
| `edge_range` | `l0-l1,r0-r1`: left/right edge ranges of the selected call; for core members, the chosen lattice configuration intersected with the class edge boxes; not a confidence interval |

`log_bf` is calculated directly from log likelihoods, using the same `1e-12`
mixture-weight floor as EM; alternatives whose weights all reach the floor
are consequently weighted equally. This equals posterior odds divided by effective
prior odds away from numerical boundaries. Older runs recovered it from
clipped posterior/prior odds, which could erase evidence near zero or one.
New manifests record `recaller.label_semantics = likelihood_ratio_v2` (also
under `transfer` for transfers). Labels, molecule edges and label-gated tiers
can change at these boundaries; the EM point estimates are unchanged.
The alternative mixture remains channel-dependent. A `member` is an evidence
label and can have a low posterior; see the
[counting discussion](../workflows/consensus-comparisons.md#confident-counts-and-fitted-prevalence-are-different-endpoints).

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
`region`, `parameters` (every group, with defaults filled in; the read-order
controls `order_replicates` and `order_robust_fraction` only when that check
is on),
`input_digest`, `mode` / `mode_realized` (`CR`), `realized_channels`,
`data_warnings`, `seconds`, `datasets`, `input_files`, `pooling` (the pooling
receipt of a CL-CR run) and `recaller`:

- `classes`, `unsupported_classes`, `unscored_classes`, `unscored` (class,
  channel, reason, molecules);
- `dropped_by_core_rule` (span and core width of each dropped candidate);
- `tiles` and per-tile `discovery` diagnostics (k chosen, prediction
  strength per k, merges, and `resplits`: each candidate that failed the core
  rule and was clustered again, with its core width, k and children);
- `order_robustness`, only with `--robust N`: `orderings` (N+1),
  `replicates` (N), `salts` (0 is the default order), `robust_fraction`,
  `rule`, `matching`, `discovered_per_ordering`, `supported_per_ordering`,
  `robust_classes`, `robust_supported_classes`, `supported_classes` and
  `seconds`.

## `result.json.gz`

The frozen result FiberBrowser reads: schema `fiberhmm.consensus.v1`,
`cr_mode: lattice_recaller`, the manifest, the per-dataset catalog and
records, and `recaller`:

- `classes`: geometry and `status` of every class (and `order_robustness`
  and `robust` with `--robust N`, also on each catalog family);
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
| `transfer_manifest.json` | schema `fiberhmm.transfer_run.lattice_recaller.v1`: `catalog`, `catalog_sha256`, `catalog_schema`, `source_provenance`, `frame`, `refitted` (false), `rediscovered` (false), `preparation_parameters`, `dataset_map`, `include_training_molecules`, `windows`, `bams`, `chip_label`, `apply_code` |
| `transfer_exclusions.json` | training molecules excluded from scoring |
| `chip_evaluation.json` | with `--chip-bed`: AUROC and average precision per class and channel |

The schema string `fiberhmm.consensus.v1` is shared by the lattice recaller
and the staged engine; FiberBrowser checks it, and `cr_mode` tells the two
engines apart.

Frozen lattice transfer loads and scores one BED window at a time, retaining
only compact annotation plans for aggregate BAM export. Raw evidence memory no
longer grows with the number of target windows. Its scoring worker pool is
reused across windows, so imports and JIT warm-up are paid per worker.

### fiberhmm-nfr: landscape runs, cross-fitting and many windows

`fiberhmm-nfr` (experimental preview) writes `variants.tsv`, `configurations.tsv`,
`molecules.tsv.gz` (per-read membership), `coaccess.tsv` (element pair tests),
`combos.tsv` (combination patterns), `result.json`, `context.json.gz` and
`manifest.json`.

**Skipping stages.** `--skip coaccess,combinations` leaves out the pair tests
and the combination patterns; `--fast-landscape` is the same preset. NFR
detection, variant discovery, the configuration EM and per-read membership
run unchanged, so `variants.tsv`, `configurations.tsv`, `molecules.tsv.gz` and
`context.json.gz` are byte-identical to a full run with the same inputs and
parameters. A skipped stage writes no file (an empty table would read as
"tested, nothing found"); `result.json` and `manifest.json` list it under
`skipped_stages`, and `result.json` has `pairs: []` with `family.stage:
skipped`. The defaults live in `fiberhmm.defaults` (`NFR_STAGES`, `NFR_SKIP`,
`NFR_FAST_LANDSCAPE_SKIP`).

**Parameters.** `--parameters` takes a JSON file or an inline JSON object
(text starting with `{`). Quote inline JSON for the shell, which otherwise
strips the double quotes:
`--parameters '{"nfr":{"min_gap_bp":150}}'`.

**Execution.** With `--region`, `--cores N` loads the datasets in parallel
(one worker process per dataset) and fits the NFR regions in parallel
(discovery, configuration EM, bootstrap and robustness reruns of one region
in one worker). The regions are assembled in order, so every output is
byte-identical to `--cores 1`. The loader keeps only the fields the NFR
analysis reads: positions and marks as arrays, without the lattice contexts
and emission probabilities. On an 18-kb window with about 8,300 molecules,
this cut peak memory from about 12 GB to 4 GB.

**Cross-fitted discovery.** `--discover-fold-spec "hash=md5,mod=10,discover=0-4"`
detects the NFR regions and discovers the variants (k-means, prediction
strength, held-out merges and support) on the reads whose fold is listed.
Every read, held out or not, then gets its membership from the configuration
EM. A read's fold is
`int(md5(read_name).hexdigest(), 16) % mod` (`fiberhmm.io.folds`, shared with
other tools). Several folds are joined with `+` (`discover=0+2+4`). With a fold
spec, `molecules.tsv.gz` gains a `fold` and a `discovery` column and
`result.json` a `cross_fit` block (spec, folds, molecule counts).
`--fold-weights heldout` fits the EM's configuration weights on the held-out
folds only (default `all`). A spec whose discover folds cover every fold gives
the same variants, configurations and memberships as a run without one.

**Variant columns: stability and edge consistency.**

- `stability` is the split-averaged prediction strength of the variant's
  k-means cell. For each of `splits` molecule split-halves, both halves are
  clustered at the chosen k. A test-half cluster scores the fraction of its
  gap pairs that the other half's fit also puts together. The variant's value
  is the mean over splits of those scores, weighted by where its gaps fall.
  It measures whether the *partition* is reproducible: with few well-separated
  cells it is near 1 even when each cell's edges are broad. It does not
  measure edge tightness.
- `Lsd` / `Rsd`: standard deviations of the variant's fitted edge Gaussian
  (robust MinCovDet covariance of its members, floored at 5 bp).
- `member_gaps`: the summed membership weight of the gaps assigned to the
  variant. A read's gap belongs to variant j with the read's posterior summed
  over the configurations that label that gap j.
- `edge_within`: the weighted share of member gaps whose left and right edge
  features both lie within `--edge-tolerance-bp` (default 25) of the
  variant's centre edges (`L`, `R`).
- `edge_mad_L` / `edge_mad_R`: the weighted median absolute deviation (bp) of
  the member gaps' left and right edge features from `L` and `R`.
- `edge_dispersion`: `(edge_mad_L + edge_mad_R) / width`.

Edge features are the midpoints of the edge ranges the EM uses (from the
nucleosome boundary to the first or last mark inside the gap, capped at 30 bp).

**Many windows.** `--bed sites.bed` (alias `--regions-bed`) analyses each
BED3/BED4 window exactly as `--region` would, in `window_XXXXXX/` (BED order).
Up to `--window-jobs` windows (default: `--cores`) run at once, one core each,
in worker processes that keep their imports across windows. Each window opens
the BAMs through their index and decodes only the reads that overlap it.
Windows share no decoded reads, so a read that spans two overlapping windows
is decoded in both. A finished window writes `window_complete.json` last,
recording a digest of the run inputs, parameters and window plus the size
and SHA-256 of each output. `--continue` in the same `--output` skips intact
finished windows and reruns the rest. It refuses when the BAMs (path, size,
mtime), the BED or the parameters differ from `nfr_run.json`. A window with no
molecules is recorded as `no_molecules`. A window that fails is recorded as
`failed` with its error while the other windows still run; the command then
exits with status 1, and `--continue` redoes only those windows. Every
attempt rebuilds `regions.json` and `regions.tsv` (one row per window: name,
region, status, NFRs, variants, molecules, seconds, error) and the combined
`variants.tsv` and `configurations.tsv` (window name and region prepended)
from the finished windows. NFR detection is independent in each window;
explicit `--nfr` and `--classes` need a single `--region`.

### Molecule depth budgets

The lattice recaller samples independently within each dataset and chemical channel
(CT/GA for DAF; pooled orientations for Hia5). `recaller.max_discovery_molecules`
defaults to 0 (all molecules) per discovery tile, `recaller.max_quantification_molecules` defaults
to 0 (all eligible molecules) per class-group scoring window, and
`recaller.sample_seed` defaults to 1. Both caps are off by default.
For deep data (thousands of reads per site), 2000 is a good opt-in cap for each
phase; set either cap to 0 to use all eligible molecules for that phase.
The corresponding CLI flags use hyphens. These controls apply to every independent
BED/region job and to tiles within pooled analyses. Historical fitted engines do
not implement these sampling controls; explicit CLI sampling flags require
`--engine lattice_recaller`.

Sampling keeps the lowest SHA-256 priorities of the seed and read name (unit ID
when no name exists), with unit ID breaking equal-name ties. Priorities do not
include tile coordinates or input order. Overlapping tiles with the same eligible
molecules therefore select the same set; when their candidate sets differ, the
priorities remain the same, but their lowest-N sets can differ. Shallow strata keep
all eligible molecules. The final cap follows input filters and duplicate collapse.
Tile overlap/site-count checks, and quantification spanning checks, precede the
cut; likelihood arrays are constructed only for the selected molecules.

When all active phases have finite budgets, BAM preparation first scans only
names, flags, MAPQ and alignment spans. Each dataset/tile retains a prefix of the
same read-name hashes, initially 1.5 times its discovery budget. The internal
`loader_sampling.margin` argument can change this margin. All alignments with a
selected name are retained together, so full-record identities and equal-name
tie breaks remain unchanged. Unselected names bypass tag/sequence decoding and
native replay. BAM blocks still need decompression; the BAM index cannot index
read-name hashes.

Hia5 has one pooled chemical channel. DAF CT/GA cannot reliably be inferred from
alignment flags without inspecting assay tags. DAF therefore advances a common
hash frontier (initially provisioning both channels) until each chemical channel
has enough usable molecules, or that
window's candidates are exhausted. It never guesses CT/GA from orientation. Rare or absent channels, heavy PCR
collapse, or stringent spanning filters can require the full candidate population.
After collapse and filtering, short windows take the next deterministic tranche.
Decoded reads and native replay results are cached across tranches. Selected raw
BAM records are reused for native replay, avoiding another regional BAM scan. When the
projected sample already covers at least 90% of candidates, preparation loads
them all without those caches; this avoids sampling overhead near the budget. Once classes
are known, their actual scoring windows can request further tranches before
quantification, including frozen transfer's fixed windows. NFR preparation does
the equivalent for detected or explicitly supplied NFRs. The final phase cap
still selects exactly the lowest N eligible molecules. Changing the margin does
not change that selection in the absence of sample-dependent duplicate families or repeated locus views.

Duplicate collapse is restricted to the sampled reads: aliases outside the
sample are not collapsed against it. Each distinct read name receives an
independent, evidence-independent hash, so duplicate aliases have independent
inclusion priorities and the retained aliases can still be collapsed normally.
This is a sampling approximation, not full-population PCR family reconstruction:
an omitted alias can change a family's representative or membership. Such
families are an exception to exact agreement with full-load post-collapse capping.

An uncapped active phase requires all observations. Thus `2000/0`, `0/2000` and
`0/0` keep the original loader path; frozen transfer needs only its quantification
budget. Full-population efficiency calibration also disables loader sampling.
Sampled evidence exports contain only decoded molecules; increasing a budget or
requesting unlimited analysis requires reloading the BAM, not replaying that
saved evidence as if it represented the whole population.

`manifest.molecule_sampling` is a list of window records. Each records `phase`,
`window`, the two caps and seed, and `strata`: `stratum` (`dataset::channel`),
`available`, `discovery`, `quantification`, and per-phase `short` flags. A count is null for the other phase
because discovery tiles and class-group scoring windows need not coincide.
Quantification records also identify `group`. The `denominator` explains that
prevalence uses the eligible molecules actually quantified, also recorded in
`classes.tsv:molecules`, not the full dataset depth. Capping does not rediscover
the catalogue. Warnings state how many molecules were used out of those available.
`read_sample_cap` records the controls and `all_units` reflects selection.

`manifest.loader_sampling` (NFR: `detection.loader_sampling`) contains the loader
receipts, also saved under each source's `load.loader_sampling` in evidence.
`reads_seen` counts fetched regional BAM records, `eligible_reads` counts records
after the cheap flag/MAPQ gate, and `reads_decoded` counts records sent to the
expensive decoder. `decode_attempts` counts decoding across all tranches (cached
reads are not decoded again). `usable_after_collapse` counts retained evidence
units; `windows[].strata` gives eligible counts, filled budgets (`used`), and
`short`/`exhausted` flags for each loader scope. Actual discovery/quantification
use is in `molecule_sampling`, whose available denominator is now the loaded
eligible sample, not an estimate of full BAM depth. DAF's pre-decode per-channel
`reads_seen` is null because its channel is unknown; shared dataset counts and
exact decoded CT/GA counts are reported instead. No full-depth chemical counts
are invented from alignment orientation. Pooled BED receipts preserve the
original genomic loading windows in `pooling.window_preparation`.

When both caps are 0, sampling metadata and controls are omitted to preserve
legacy output bytes (apart from the existing wall-clock timing fields). Absent
sampling controls in a saved receipt mean uncapped on resume/frozen-catalog replay;
new runs also default to no sampling. For deep data (thousands of reads per site), 2000 is a good cap for each phase. Frozen-class transfer may override only
these three `recaller` controls; class geometry and spots remain fixed.

The NFR engine exposes the same controls in `NFROptions` / the `nfr` JSON group.
It samples the run's detection profile and each NFR's variant discovery, then
independently samples callable molecules for quantification of the fixed variant
catalogue. Non-callable reads do not consume a variant budget. Its result and
manifest store equivalent window/stratum records in
`detection.molecule_sampling`, with `purpose` and (for variants) `nfr` identifying
the scope. `nfrs[].callable` and variant `by_channel[].n` are the actual sampled
prevalence denominators. The detection profile's denominator is the sampled
callable coverage at each position. Both caps at zero preserve legacy NFR bytes.
