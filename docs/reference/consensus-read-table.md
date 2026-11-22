# Per-read tables, cross-fitting and role rules

`fiberhmm-consensus --read-table` writes every read's posterior for every
discovered class. Over a BED of promoter windows it is a per-promoter batch:
each window is discovered and scored on its own, and the windows are merged
into one table keyed by read name and window. Optional pieces:

- **Cross-fitted discovery** (`--discover-fold-spec`): classes come from some
  read-name hash folds only, and every read is scored.
- **Role rules** (`--role-rules`): label classes by position relative to each
  window's anchor (for example the TSS) and width.
- **NFR-state conditioning** (`--nfr-states`): class prevalence within each
  `fiberhmm-nfr` variant state.

```bash
# promoters.bed: BED6, TSS-300..+800 in transcription orientation, one row per promoter
fiberhmm-consensus --bam calls.bam --bed promoters.bed --cores 8 --no-bam \
  --read-table --discover-fold-spec "hash=md5,mod=10,discover=0-4" \
  --role-rules roles.json --anchor-offset 300 --output out/promoters
```

The run is the ordinary [multi-window batch](../workflows/consensus.md#several-windows).
Each window is loaded, analysed, written and freed by itself, so memory is set
by the largest window and the number of windows running at once, not by the
number of BED rows. `--continue` finishes an interrupted run, and
`--max-memory-gb` lowers `--window-jobs` (then `--cores`) so the largest window
fits.

## Cross-fitted discovery

`--discover-fold-spec SPEC` (parameter `recaller.discover_fold_spec`) uses the
same fold helper and syntax as `fiberhmm-nfr` (`fiberhmm.io.folds`), so both
tools put a read in the same fold:

```text
fold = int(md5(read_name), 16) % mod
hash=md5,mod=10,discover=0-4     folds 0-4 discover, 5-9 are held out
mod=5,discover=0+2+4             '+' lists folds, '-' gives a range
```

- **Discovery** (k-means, prediction strength, identity tests) sees only the
  reads in discovery folds. DAF strands of one read share its name, so they
  share its fold. `--max-discovery-molecules` then applies within the discovery
  folds. A molecule-capped run loads every molecule instead of a read-name hash
  prefix, so the cap can be filled from the discovery folds.
- **Scoring** gives every read a posterior: discovery folds and held-out folds
  alike.
- `--fold-weights all` (default) fits each scoring group's EM weights, learned
  spots and edge contraction on all reads. `heldout` fits them on the held-out
  folds only, and every read's posterior uses that fit, as `fiberhmm-nfr
  --fold-weights heldout` does.
- The class geometry is fixed by discovery. With `--fold-weights all`,
  per-channel learned spots (`recaller.learned_spots`, on by default) and edge
  contraction (`recaller.edge_contraction`, off by default) are refined on all
  scored reads. Use `heldout`, or turn them off in `--parameters`, to keep
  held-out reads out of every fitted quantity.
- `manifest.json` records a `cross_fit` block: the canonical spec, the
  discovery and held-out unit counts, and the weights mode.

Fold columns use the BAM read name exactly. A copy of the BAM whose reads were
renamed (for example by a trimming script that appends the window) gets
different folds from `fiberhmm-nfr` run on the original BAM. Run both tools on
the original BAM; `--evidence-pad` (default 250 bp) already crops each read to
the window for memory.

## `read_posteriors.tsv.gz` (each window)

One row per evidence unit × discovered class, so every read of the window has a
row for every class, scored or not. A window with no classes writes the header
only.

| Column | Meaning |
|---|---|
| `read_name` | BAM read name (the join key with `fiberhmm-nfr`) |
| `fold`, `discovery` | hash fold, and whether that fold discovered (with no cross-fitting: md5 mod 10, and `True` for every read) |
| `dataset`, `strand`, `unit_id` | dataset, channel strand (`pooled` for Hia5; `CT`/`GA` for DAF) and evidence unit |
| `group`, `class_id`, `class_start`, `class_end` | the class's overlap group and reference span |
| `class_status` | the class on this channel: `supported`, `unsupported` or `unscored` |
| `scored` | whether the read spans the group's scoring window on its channel |
| `posterior`, `log_bf`, `label` | class posterior, likelihood ratio and three-way label, as in `molecules.tsv.gz` |
| `p_broader`, `p_other_shape`, `p_accessible` | the group's three non-class components for this read |
| `loglik`, `ll_broader`, `ll_other_shape`, `ll_accessible` | the read's log-likelihood (nats) under the class and under the three alternatives |

Within one read and group, the class posteriors plus `p_broader`,
`p_other_shape` and `p_accessible` sum to 1 (to rounding). Classes in different
groups do not overlap and are scored separately. The log-likelihoods are the
EM's own molecule × hypothesis matrix, so prevalence can be refitted on any
subset of reads with the classes fixed: run EM on the subset's rows of
`[loglik of each class in the group, ll_broader, ll_other_shape, ll_accessible]`.

## Merged tables (`read_table/`)

A `--bed`/`--region` run with `--read-table` also writes `read_table/` in the
output directory. Each table is streamed one window at a time, and the whole
directory is rebuilt from the saved window tables at the end of every run and
every `--continue`.

| File | Contents |
|---|---|
| `read_posteriors.tsv.gz` | every window's rows with window columns (`window_index`, `window`, `chrom`, `window_start`, `window_end`, `window_strand`, `anchor`), then `class_rel_start`, `class_rel_end`, `class_width`, `role` and `broader_role` |
| `class_roles.tsv` | one row per window × class: relative geometry (`class_rel_start`, `class_rel_end`, `class_center`, `class_width`), `role`, the group's `broader_role`, `status`, `supported_channels`, `max_prevalence` |
| `read_roles.tsv.gz` | one row per window × read: `groups_scored` and `p_<role>` for every role in the rule file |
| `class_by_nfr_state.tsv` | with `--nfr-states` (see below) |
| `read_table.json` | rows, windows, role names, the rule file's path and SHA-256 (of its parsed rules), and the anchor offset |

**Anchor.** Positions are relative to each window's anchor, in transcription
orientation: `anchor = window 5' end + --anchor-offset` (the BED start on `+`,
the BED end on `-`). For TSS−300..+800 windows use `--anchor-offset 300`.
`class_rel_start` is always the upstream edge, so on a `-` window it comes from
the class's reference end.

**`p_<role>`.** Classes of one group are mutually exclusive, so a role's
posteriors add within a group. Across groups, the value is the largest group
total, the best-supported instance of the role on that read. A broader role
adds the group's `p_broader`. A role whose classes the read was not scored for
is empty.

## Role rule files

Roles are your labels; none are built in. A rule file maps class geometry to
role names:

```json
{
  "schema": "fiberhmm.consensus.roles.v1",
  "description": "optional free text",
  "rules": [
    {"role": "upstream_small", "end": {"le": -100}, "width": [null, 50]},
    {"role": "over_anchor",    "start": {"lt": 0}, "end": {"gt": 0}},
    {"role": "downstream",     "start": {"ge": 0}, "center": [null, 200], "width": [20, 80]},
    {"role": "wide_over_downstream", "component": "broader", "group_roles": ["downstream"]}
  ]
}
```

- `start`, `end`, `center` and `width` are in bp, relative to the anchor in
  transcription orientation.
- Each condition is an inclusive `[min, max]` (`null` leaves that side open),
  or an object with any of `gt`, `ge`, `lt`, `le`.
- Rules are tried in file order, and the first match wins.
- `component` is `class` (the default: the rule labels discovered classes) or
  `broader`. A broader rule labels a group's broader-protection component,
  meaning reads best explained by protection wider than every class of the
  group, such as a nucleosome over it. Its geometry is the span of the group's
  classes. `group_roles` optionally requires a class of the group to carry one
  of the listed roles.
- Discovery uses native calls up to `recaller.call_max_bp` (default 100 bp).
  Nucleosome-sized protection over a class therefore appears as that group's
  broader component, not as a class.

Unknown keys, empty role names, `min > max` and malformed conditions are
refused before the run starts. Roles are applied when `read_table/` is built,
so a changed rule file needs no recompute: rerun the same command with
`--continue --role-rules new.json`. Complete windows are kept and only the
merged tables are rewritten.

## Conditioning on `fiberhmm-nfr` states

`--nfr-states DIR` takes a `fiberhmm-nfr` output: one run, or a `--bed` run
with `window_*` directories. Each NFR record of a read that is `callable`
(nucleosomes on both sides) gives it a state, the `map` configuration from
`molecules.tsv.gz`. Reads are joined by read name, plus the chemical strand for
DAF.

For every consensus window, every NFR of each nfr run whose region overlaps the
window, every state (plus `all`, every joined read) and every class and
channel, `class_by_nfr_state.tsv` gives:

| Column | Meaning |
|---|---|
| `nfr_run`, `nfr`, `nfr_start`, `nfr_end`, `state` | the NFR and the reads' state |
| `dataset`, `strand`, `group`, `class_id`, `role` | the class and channel |
| `reads` | reads in that state scored for the class's group |
| `mean_posterior`, `members` | from the window's pooled fit |
| `em_prevalence` | class prevalence refitted by EM on those reads alone, with the classes and every read's log-likelihoods fixed |

The classes are the same in every state, because they come from the window's
discovery. Only their prevalence is re-estimated, so states are compared on one
footprint catalog. `read_table.json` records the join rate per window. A run in
which no read joins is an error, not an empty table.
