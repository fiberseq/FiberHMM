# Footprint population model

`fiberhmm-footprint-model` turns ordinary single-molecule TF calls into a
reusable population model without replacing or filtering them. The complete
artifact is a **footprint population model**; each learned geometry family is
a **TF-binding hypothesis**. When a frozen model is later used for rescue,
its population frequencies can serve as empirical priors, but the
data-derived model itself is not a prior.

The model has two levels:

- a **locus** is a recurrent binding population found from a smoothed density
  of footprint centres;
- a **family** is one recurrent footprint geometry at that locus, learned
  from compatible left and right boundaries.

A locus may contain overlapping or nested families. A molecule may support
more than one family at the same locus when it has more than one TF call;
each call is assigned to at most one family.

This is a different analysis from [footprint classes](consensus.md): the
population model summarizes the per-read calls genome-wide or per region
without re-scoring molecules, while `fiberhmm-consensus` scores every
molecule's own marks against discovered classes in a window.

## BAM workflow

The command reads the ordinary `tf` and `msp` groups of a called BAM's `MA`
tags:

```bash
fiberhmm-footprint-model -i out/pacbio.calls.bam -o out/fpm/demo --genome demo --bigbed
```

It prints a JSON summary of what was read and projected
(`bam_diagnostics`), the files written and the scope, and writes:

```text
out/fpm/demo.footprint-model.tsv               population catalog
out/fpm/demo.footprint-model.bed / .as / .bb   population track
out/fpm/demo.footprint-model.strata.tsv        per-stratum counts and boundaries
out/fpm/demo.footprint-model.assignments.tsv   call-to-site_id audit
out/fpm/demo.footprint-model.fiberlayers.bed / .as / .bb / .json   FiberBrowser overlay
```

`-o` is a filename prefix, not a directory (missing directories are created).
`--force` replaces an earlier bundle with the same prefix.

The complete BAM can be streamed without an index. A targeted analysis uses
repeatable zero-based, half-open regions and requires an index:

```bash
fiberhmm-footprint-model -i recalled.bam -o results/window \
  --region chr2R:9974319-9984319
```

Region scope is defined by projected TF center: a call is retained when its
center lies in the requested half-open interval. Full mapped blocks and MSPs
remain available for retained molecules. Pad a target interval when hypotheses
near its boundary must be discovered from calls centered just outside it.

Only mapped primary, non-QC-failed, nonduplicate records with a valid `MA` tag
enter the baseline by default. A missing `MA` tag means that the record is not
known to have passed footprint calling; it is not interpreted as an unoccupied
molecule. A valid `MA` with no `tf` group is retained and contributes to
denominators. Secondary and supplementary alignments are excluded, MAPQ
defaults to zero, and `--include-duplicates` is available for an intentional
uncollapsed analysis.

This denominator rule deliberately trusts the presence of any valid `MA` as
evidence that footprint calling ran. A mixed BAM in which some records carry
only an unrelated MA group can therefore be inappropriate input; use a
consistently footprint-called/recalled cohort rather than combining caller
stages.

For DAF reads, `st:Z:CT` and `st:Z:GA` define the physical strata. If `st` is
absent, alignment orientation supplies `FWD`/`REV`; a present value other than
`CT` or `GA` is treated as malformed rather than silently reclassified. A
whole-BAM run with no analyzable MA records or no projectable ordinary TF calls
fails with a recalled-BAM hint. An explicitly targeted region may validly be
empty and writes an empty, auditable model with a warning.

MA intervals are converted from molecular coordinates to the stored BAM query
frame and then projected through the CIGAR. Partially mapped TFs are retained
when at least one base projects, but unreliable edges do not teach canonical
geometry. MSPs must meet the mapped-fraction requirement. All projection and
filter counts are recorded in the FiberLayers manifest and printed as JSON by
the command. `--bigbed` requires UCSC `bedToBigBed`; chromosome sizes are taken
from the BAM header.

Linked `AN` pieces from circular-molecule calling are not reconstructed. The
adapter rejects records carrying ordinary TF/MSP annotations together with
`AN` rather than silently interpreting wrapped pieces as independent linear
sites. Linear genomic DddB/DddA and fiber-seq BAMs are the supported input
geometry.

## Baseline discovery

`build_footprint_population_model()` accepts projected TF calls, mapped
reference blocks, and optional MSP intervals grouped by molecule. Discovery is
deterministic:

1. Count each molecule once at each footprint-center position.
2. Smooth the center counts with a finite Gaussian kernel and find local modes.
3. Assign nearby calls to a mode.
4. Split each mode into boundary-compatible geometry families. Compatibility
   is a maximum diameter for both edges, so a chain of marginally similar calls
   cannot bridge two distant geometries.
5. Estimate canonical boundaries from molecule-level representatives and give
   physical/read strata equal votes when they meet the geometry-support
   threshold.
6. Assign every source call to its learned family when possible and summarize
   the eligible population.

Every ordinary TF call is valid. There is intentionally no TQ cutoff, local
background-enrichment test, top-N limit, or minimum-support rejection.
`geometry_eligible` only controls whether a call can teach the frozen geometry;
an ineligible call can join a compatible family learned from eligible calls,
but cannot create a family by itself. Calls that cannot join remain explicit
unassigned rows in the assignment audit and are absent from the FiberLayers
overlay.
`geometry_ready` and `population_ready` are descriptive readiness flags, not
filters.

For supported-population reporting, require both readiness flags. Population
support is the exact unique-molecule count `n_tf`; geometry support is
`cluster_support_molecules`. The standard population BED/BigBed `score` is
`min(n_tf, 1000)`, so generic ranking reflects recurrence support rather than
occupancy. Occupancy remains available only in its explicit fields.

An isolated geometry-eligible call that lies outside every smoothed mode's
assignment radius is retained by creating a deterministic fallback locus at its
center. Consequently, every geometry-eligible call gets a hypothesis, but a
support-one locus is not necessarily a local maximum of the multi-molecule
smoothed density. Every projectable call remains auditable whether assigned or
not.

## Population measurements

A molecule enters a family's denominator only when both family endpoints map
and its mapped blocks cover the complete canonical interval at the configured
mapped fraction. TF and MSP status then form a four-cell table:

| | MSP contains family | No containing MSP |
| --- | ---: | ---: |
| TF assigned | `n_tf_msp` | `n_tf_no_msp` |
| No TF assigned | `n_no_tf_msp` | `n_no_tf_no_msp` |

The baseline catalog reports:

```text
occupancy_overall   = n_tf / n_fully_mapped
occupancy_given_msp = n_tf_msp / n_msp
```

The second value conditions on molecules whose MSP contains the entire family;
reads without such an MSP are not counted in that estimand. These are direct
population fractions, not enrichment against a genomic background.

## Identity

- `site_id` is the durable family key derived from contig and canonical family
  geometry.
- `locus_id` is the parent key derived from contig and the smoothed locus
  summit.
- `family_index` is a deterministic, 1-based display order within a locus.

`site_id` is the foreign key carried by call assignments and visualization
blocks. `family_index` may renumber if a refit gains or loses a family. IDs are
stable when their defining fitted geometry is unchanged; a materially changed
refit is a new model and may produce new IDs.

## Output bundle

`write_footprint_model_bundle()` writes complementary analysis and
visualization artifacts from one model:

| Artifact | Purpose |
| --- | --- |
| population TSV | Lossless family catalog and occupancy measurements |
| population BED + autoSQL | Portable genome-browser and interval-analysis track |
| strata TSV | Per-family physical/read-stratum counts and boundary estimates |
| assignments TSV | Auditable raw-call-to-`site_id` mapping |
| FiberLayers BED12+5 + autoSQL | Per-read footprint assignments for FiberBrowser |
| FiberLayers manifest | One logical layer, provenance, extension declarations, and counts |
| optional population/FiberLayers BigBeds | Indexed active-window access after UCSC conversion |

The library-level workflow is:

```python
from fiberhmm.inference import build_footprint_population_model
from fiberhmm.io import (
    convert_footprint_model_bundle_to_bigbed,
    write_footprint_model_bundle,
)

# molecules: an iterable of fiberhmm.inference.tf_sites.BaselineMolecule
model = build_footprint_population_model(molecules)
paths = write_footprint_model_bundle(
    model,
    "sample",
    source_dataset={"id": "sample", "path": "sample.bam"},
    genome="dm6",
    genomewide=True,
)

# Optional indexed tracks for genome-wide, active-window loading.
bigbeds = convert_footprint_model_bundle_to_bigbed(paths, "dm6.chrom.sizes")
```

Scope metadata defaults to `genomewide=False`; scanning every record in a BAM
does not prove that the library itself sampled the whole genome. Pass
`--genomewide` (or `genomewide=True` in Python) only as an explicit scope
assertion. Targeted callers should record their analyzed `regions`.

The output prefix expands to:

```text
sample.footprint-model.tsv
sample.footprint-model.bed
sample.footprint-model.as
sample.footprint-model.strata.tsv
sample.footprint-model.assignments.tsv
sample.footprint-model.fiberlayers.bed
sample.footprint-model.fiberlayers.as
sample.footprint-model.fiberlayers.json
```

The FiberLayers BED/BigBed payload uses one fixed class,
`footprint_model_assignments`; site IDs do **not** become UI layer toggles. Each
block uses the observed raw footprint coordinates and has an aligned
`blockSiteIds` value. The population BED supplies the canonical model geometry.

BED12 cannot encode overlapping blocks in one row for BigBed conversion.
Therefore, overlapping calls on the same read are greedily partitioned into
non-overlapping physical rows. Those rows retain the same read ID and logical
class. Identity-aware FiberBrowser support uses two loader extensions declared
by the manifest:

1. parse `blockSiteIds` into block-aligned annotation names;
2. concatenate duplicate `(read_id, class_id)` rows instead of overwriting
   them.

The paired FiberBrowser backend supports both transport extensions, so current
bundles load as one generic derived layer without losing overlapping lanes or
block-aligned site IDs. It does not yet resolve the population catalog or offer
model-aware support filtering, cross-highlighting, or local model summaries.
Those controls are the intended next FiberBrowser integration. Genome-wide IDs
remain data entities rather than thousands of persistent controls.

## Rescue integration

The baseline model is deliberately independent of consensus rescue and strand
rescue. Later rescue stages can consume the normalized `site_id` assignments,
and global reporting can optionally run on a selected post-rescue call layer.
The safe default is to freeze geometry from ordinary calls and mark rescued
calls `geometry_eligible=False`, so they can contribute to occupancy without
teaching the model that rescued them. A deliberate post-rescue refit should be
versioned as a new model. Rescue scoring should also exclude the target
molecule from its own population evidence.

The exported catalog contains TF-call assignments and aggregate denominator
counts, not every denominator-only molecule-site membership. A leave-one-out
rescuer must therefore retain or reread the target molecule's mapped blocks and
MSPs; the bundle alone is not a standalone leave-one-out evidence database.

Every option: [`fiberhmm-footprint-model`](../reference/cli.md#fiberhmm-footprint-model).
