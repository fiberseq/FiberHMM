# Footprint population model: implementation and handoff

**Status date:** 2026-07-23  
**Implementation status:** ready for baseline BAM use and a first downstream
family-caller integration  
**Primary downstream test:** OCT family caller

This document is the implementation handoff for FiberHMM's data-derived
footprint population model. It describes what exists now, the scientific and
file-format contracts a downstream caller can rely on, the verified real-data
behavior, and the work that remains outside this baseline implementation.

The Dropbox-synchronized example bundle is in
[`handoff/footprint_population_model`](../handoff/footprint_population_model/README.md).
Use that bundle—not the `/tmp` paths from development runs—when transferring
the first OCT integration to another machine.

The conceptual and user-facing model documentation is in
[`footprint-models.md`](footprint-models.md). This handoff is intentionally more
operational and should be read before changing schemas or using the model as
input to another caller.

## Executive status

| Component | Status | Notes |
| --- | --- | --- |
| Population site discovery | Implemented and tested | Finds smoothed center loci and overlapping/nested geometry families. |
| Ordinary BAM adapter | Implemented and tested | Reads lowercase `tf` and `msp` MA groups; TQ is not used. |
| Population support scoring | Implemented and tested | BED score is `min(n_tf, 1000)`, not occupancy. |
| Readiness filtering | Implemented in data contract | Recommended supported set is `geometry_ready == 1 AND population_ready == 1`. |
| Portable analysis bundle | Implemented and tested | Population, strata, and call-assignment TSVs plus BED/autoSQL. |
| BigBed conversion | Implemented and tested with UCSC tools | Conversion uses staged temporary files; failed conversion does not publish the staged BigBeds or update the manifest. |
| Per-read FiberBrowser overlay | Serialization and generic loading verified | Stable site IDs remain block-aligned; dedicated support-filter UI controls are not implemented in FiberHMM. |
| OCT/family labels | Not part of this model | The OCT caller should add an orthogonal sidecar keyed by model identity and `site_id`. |
| Consensus/strand rescue | Deliberately not integrated yet | The baseline model is suitable input for later rescue but does not perform rescue. |
| Circular linked MA annotations | Explicitly rejected | `AN`-linked ordinary TF/MSP pieces are not reconstructed as linear sites. |

In short: the baseline producer is ready to hand off. The OCT caller can be its
first direct consumer. A specialized FiberBrowser filtering panel and rescue
feedback are follow-on work, not hidden parts of the current implementation.

## Scientific object and terminology

The complete artifact is a **footprint population model**. It contains:

- **loci**, discovered from smoothed footprint-center density;
- one or more possibly overlapping **TF-binding hypotheses** at each locus,
  separated by compatible left and right footprint geometry;
- explicit assignments from observed per-read footprint calls to hypotheses;
- population denominators and overall/MSP-conditioned occupancy summaries.

A TF-binding hypothesis is a recurrent protected geometry. It is not, by
itself, an identification of a TF protein or family. An OCT caller must add
sequence, motif, perturbation, ChIP, or other orthogonal evidence before
assigning an OCT label.

Every valid ordinary TF call remains valid evidence. The discovery process has
no TQ cutoff, local genomic-background rejection, top-N cap, or null-derived
p-value. Sparse hypotheses remain in the exhaustive catalog for audit and
future rescue; readiness fields distinguish the supported analysis set.

## Support and readiness contract

Support and occupancy answer different questions and must not be conflated.

### Population support

`n_tf` is the primary population-support value:

```text
population_support = n_tf
```

It counts unique, fully mapped molecules with at least one call assigned to the
hypothesis. The Python API exposes the same value as
`site.population_support`.

The standard population BED/BigBed `score` is:

```text
score = min(n_tf, 1000)
```

This is an exact molecule count until the BED score ceiling. The uncapped
`nTf` field remains present in BigBed and `n_tf` remains present in the TSV.
This deliberately prevents a singleton observed on one covered molecule from
ranking above a recurrent low-occupancy population.

### Geometry support

`cluster_support_molecules` counts unique eligible molecules that established
the canonical footprint geometry. The Python API exposes it as
`site.geometry_support`.

### Readiness

The configured thresholds produce two independent flags:

```text
geometry_ready   = cluster_support_molecules >= minimum_geometry_support
population_ready = n_tf >= minimum_population_support
analysis_ready   = geometry_ready and population_ready
```

`analysis_ready` is a Python property and a manifest-declared expression. In
the flat TSV and BigBed, downstream tools should implement it by requiring both
stored readiness fields to equal one.

The default thresholds are three molecules for geometry and three for
population support. They are descriptive and configurable:

```bash
--minimum-geometry-support 5 \
--minimum-population-support 5
```

Changing a threshold changes readiness labels, not the discovered catalog.
This preserves the exhaustive evidence while allowing conservative reporting.

### Occupancy

Occupancy remains a separate prevalence measurement:

```text
occupancy_overall   = n_tf / n_fully_mapped
occupancy_given_msp = n_tf_msp / n_msp
```

The second estimand includes only molecules whose MSP contains the complete
hypothesis. A missing MSP is not a genomic background observation.

There is no null-background significance score. The only plausible null would
ask whether valid footprints align into the same geometry by chance after
preserving coverage, MSP structure, footprint burden, and length distribution.
That requires strong placement assumptions and is intentionally outside this
baseline model.

## Implementation pipeline

The end-to-end path is:

```text
recalled BAM
    |
    v
MA adapter and CIGAR projection
    |
    v
per-molecule mapped blocks, ordinary TFs, and MSPs
    |
    v
smoothed footprint-center loci
    |
    v
boundary-compatible geometry families
    |
    v
call-to-site assignment and molecule denominators
    |
    +--> population/strata/assignment tables
    |
    +--> population BigBed
    |
    +--> per-read FiberLayers BigBed + manifest
```

The main implementation modules are:

- `fiberhmm/io/footprint_bam.py`: BAM validation, filtering, MA parsing, and
  reference projection;
- `fiberhmm/inference/tf_sites.py`: deterministic discovery, assignment,
  support, and occupancy;
- `fiberhmm/io/tf_models.py`: portable TSV/BED/BigBed and FiberLayers bundle;
- `fiberhmm/cli/footprint_model.py`: end-user command and provenance.

### BAM intake

The adapter consumes ordinary lowercase `tf` and `msp` groups from `MA`.
Quality bytes and TQ are not used.

Default record rules:

- mapped primary alignments only;
- secondary, supplementary, QC-failed, and duplicate-flagged records excluded;
- MAPQ minimum zero unless changed with `--min-mapq`;
- records missing `MA` excluded because they are not known to have completed
  footprint calling;
- valid MA records with no TF retained as denominator molecules;
- `st:Z:CT` and `st:Z:GA` define DAF strata;
- when `st` is absent, alignment orientation gives `FWD` or `REV`;
- a present invalid `st` value is an error.

A whole-BAM run fails clearly if it finds no analyzable MA records or no
projectable ordinary TF calls. An explicitly requested empty region is allowed
and produces an auditable empty model with a warning.

The adapter intentionally trusts any valid MA tag as evidence that ordinary
footprint calling ran. Use a consistently footprint-called/recalled cohort;
records carrying only unrelated MA groups in a mixed-stage BAM would otherwise
be treated as denominator molecules.

MA coordinates are molecular coordinates. Reverse reads are flipped into the
stored BAM query frame before CIGAR projection. Hard clips, soft clips, indels,
deletions, and skipped reference spans are handled. A partially projectable TF
remains an observation, but only sufficiently complete calls with mapped
endpoints can establish geometry. MSPs must satisfy the configured projection
fraction.

Targeted scope uses each projected TF center and exact half-open region
membership. Pad a requested interval when boundary hypotheses should be
informed by calls centered just outside it.

### Site discovery

For each contig:

1. Count each molecule once at each rounded TF-center position.
2. Apply finite Gaussian smoothing.
3. Find local center-density maxima separated by the configured peak distance.
4. Add deterministic fallback modes so isolated geometry-eligible calls are
   not discarded.
5. Assign nearby geometry-eligible calls to modes.
6. Partition each mode into families with bounded diameter at both edges.
7. Estimate canonical boundaries from one representative per molecule and
   equal-vote eligible strata.
8. Assign all projectable calls, including geometry-ineligible calls, to a
   compatible learned hypothesis when possible. Calls that do not match remain
   explicit unassigned audit rows and are not placed in the FiberLayers overlay.
9. Count complete mapped-molecule and MSP denominators.

Molecule/site coverage queries use indexed mapped-block starts and cumulative
mapped bases. On a profiled DddB chromosome this reduced unprofiled discovery
time from 10.96 seconds to 2.35 seconds without changing the catalog.

## Output bundle and authority

Open the FiberLayers manifest first:

```text
<prefix>.footprint-model.fiberlayers.json
```

Require:

```text
model.schema     = fiberhmm.footprint_population_model.v1
model.row_object = tf_binding_hypothesis
scope.coordinate_system = 0-based-half-open
model.support.bed_score.formula = min(population_support, 1000)
```

Resolve filenames relative to the manifest. Retain the manifest path or digest,
assembly, scope, configuration, adapter filters, and exporter version as the
model identity. Do not infer the contract only from filename suffixes.
Prototype manifests lacking `model.support` used occupancy-valued BED scores
and must be treated as legacy. The corrected support semantics are pre-release
under the current v1 schema.

| Artifact | Authority and use |
| --- | --- |
| `.footprint-model.tsv` | Authoritative lossless table: one hypothesis per row. Primary OCT/family-caller input. |
| `.footprint-model.strata.tsv` | Per-stratum support, geometry, denominators, and occupancy. |
| `.footprint-model.assignments.tsv` | Auditable raw-call-to-hypothesis mapping and per-read propagation input. |
| `.footprint-model.bed/.bb` | Active-window population lookup and browser visualization. Standard score is support. |
| `.footprint-model.fiberlayers.bed/.bb` | Assigned observed calls grouped by read for FiberBrowser. |
| `.footprint-model.fiberlayers.as` | FiberLayers BigBed schema. |
| `.footprint-model.fiberlayers.json` | Bundle manifest, model semantics, provenance, files, and statistics. |

The FiberLayers overlay uses one fixed class,
`footprint_model_assignments`. Site IDs are data keys, not thousands of UI
layers. Each observed footprint block has an aligned `blockSiteIds` value.
Overlapping per-read intervals are partitioned into multiple physical BED rows;
the manifest declares that matching read/class rows must be concatenated.

`blockQuality=255` is categorical serialization and is not TQ, support, or
confidence. To support-filter an overlay, resolve `blockSiteIds` against the
population catalog and retain sites passing the readiness or support rule.

## Identity and joins

Use the following key rules:

- `site_id` is the hypothesis foreign key within one model bundle.
- Namespace `site_id` with a model reference or manifest digest in external
  products.
- `locus_id` is the parent for overlapping/nested hypotheses.
- `family_index` is display order only and may renumber after refitting.
- Assignment identity is
  `(contig, stratum, molecule_id, call_id)`; `call_id` alone repeats across
  molecules.
- Join assignments to the population table on non-missing `site_id`.

Site IDs remain stable only when the assembly and fitted geometry remain
unchanged. Do not assume IDs are universal across cohorts, assemblies,
different adapter filters, or changed discovery configurations.

## OCT family caller integration

The recommended first integration is artifact-based rather than importing
internal classes. That keeps the OCT caller decoupled from FiberHMM's Python
implementation while preserving a complete provenance boundary.

### Inputs

1. Read and validate the manifest.
2. Read the population TSV.
3. For supported family reporting, retain:

   ```text
   geometry_ready == 1 and population_ready == 1
   ```

4. Rank by `n_tf` descending, with `cluster_support_molecules` and boundary MAD
   values available as secondary evidence.
5. Use `contig`, `start`, `end`, and `summit` to fetch sequence or intersect
   motif calls.
6. Scan both motif strands. The footprint hypothesis itself is unstranded, and
   footprint edges are not motif edges.
7. Optionally use the strata table to check CT/GA support without treating
   single-stratum hypotheses as invalid.
8. Join the assignments table on `site_id` when OCT labels should propagate to
   individual observed footprint calls.

For MSP-conditioned claims, additionally require `n_msp > 0` and choose an
explicit minimum MSP denominator. `population_ready` measures TF-supporting
molecules; it does not guarantee a precise conditional-occupancy estimate.

### Recommended OCT output

Write a sidecar rather than rewriting FiberHMM site identities:

```text
model_ref
site_id
locus_id
family_label
family_score
caller_version
evidence
motif_id
motif_contig
motif_start
motif_end
motif_strand
```

The minimum foreign key is `(model_ref, site_id)`. Carry `locus_id` for grouped
analysis. For per-read propagation also carry
`contig`, `stratum`, `molecule_id`, and `call_id`.

The OCT caller must not assume:

- one geometry family equals one TF protein;
- hypotheses within a locus are mutually exclusive;
- occupancies of overlapping hypotheses sum to one;
- support-one hypotheses are population-density peaks;
- occupancy is enrichment, posterior probability, or significance;
- a missing assignment is a molecule-level negative;
- a missing MSP is background;
- rescue has already been applied.

### Suggested first acceptance test

Use the verified small DddB fixture:

```bash
python -m fiberhmm.cli.footprint_model \
  --input "/Users/tt7739/Dropbox/Fiber-NET-seq/Drosophila_phase2/Datasets/DAF-seq/spacetime_updated/kr_compare/fp/YWMJBX_4_siGAF_4.5-5.5hr_kr_recalled.bam" \
  --output-prefix /tmp/ywmjbx4-footprint-model \
  --genome dm6 \
  --bigbed \
  --force
```

Then:

1. Have the OCT caller consume the manifest and population TSV.
2. Record the full and supported hypothesis counts.
3. Require every OCT output `site_id` to exist in the source model.
4. Confirm support ranking uses `n_tf`, not occupancy.
5. Join OCT labels to assignments and require every propagated row to retain
   the original molecule/call identity.
6. Load the FiberLayers BigBed and verify `blockSiteIds` can resolve the OCT
   sidecar within an active window.
7. Report how many supported hypotheses are OCT-labeled, unlabeled, or
   ambiguous; do not force a label for every geometry.

This test directly exercises the intended architecture:

```text
raw valid footprint
    -> shared population hypothesis
    -> orthogonal OCT-family label
    -> per-read propagation and visualization
```

## Verified real-data behavior

The small DddB fixture above has:

- 915 emitted MA-analyzed denominator molecules;
- 4,178 raw ordinary TF annotations;
- 4,170 projectable TF calls;
- 4,130 assigned calls and 40 auditable unassigned calls;
- 601 population loci;
- 1,829 TF-binding hypotheses;
- 434 geometry-ready hypotheses;
- 443 population-ready hypotheses;
- 432 hypotheses passing both default readiness rules.

Both population and FiberLayers BigBeds pass UCSC validation. A FiberBrowser
active-window round trip on chr2R preserved 3,814 blocks and 3,814 aligned site
IDs across 738 reads.

The larger 84 MB
`spacetime_updated/fp/1-1.5_recalled.bam` completed whole-BAM model construction
in 33.9 seconds after mapped-block indexing. It produced 7,119 denominator
molecules, 36,522 projected TF calls, 2,817 loci, and 11,563 hypotheses.

## Validation and tests

Focused coverage includes:

- forward/reverse molecular-coordinate projection;
- hard/soft clipping and partial projection;
- valid zero-TF denominator molecules;
- missing-MA exclusion;
- flag, MAPQ, duplicate, region, and index behavior;
- malformed MA and malformed stratum failure;
- overlapping/nested family discovery;
- deterministic IDs and assignments;
- mapped-block gap/fraction/endpoint rules;
- support-based ranking versus occupancy inversion;
- TSV/BED/autoSQL/manifest identity and schema;
- overlapping per-read FiberLayers lane partitioning;
- real UCSC BigBed conversion with long block-aligned site-ID payloads;
- empty/wrong BAM diagnostics and converter preflight.

The package builds as both sdist and wheel, and the console entry point is
`fiberhmm-footprint-model`.

The latest repository regression run completed with 793 passing tests and
3 environment/dependency skips; 27 existing external-artifact receipt
selections were excluded from that run. The footprint-model-focused suite
completed with 62 passing tests.

## Known limitations and next work

These are explicit non-features, not implicit promises:

1. FiberHMM writes a filterable support contract, but a dedicated
   FiberBrowser model-aware filter/ranking panel still belongs in FiberBrowser.
2. The OCT caller has not yet completed the first consumer round trip described
   above.
3. Consensus rescue and strand rescue do not yet consume or refit this model.
4. The bundle does not export denominator-only molecule-to-site membership.
   Leave-one-out rescue must reread mapped blocks and MSPs from the BAM.
5. Circular `AN`-linked ordinary TF/MSP pieces are not reconstructed; the BAM
   adapter rejects such records explicitly.
6. Legacy-only `ns/nl/as/al` BAMs are not accepted; the adapter is deliberately
   MA-first.
7. There is no cross-cohort merge, differential occupancy model, null p-value,
   or FDR layer.
8. A model refit can change site IDs when canonical geometry changes.
9. The eight plain-text artifacts are not published as one filesystem
   transaction. If a late write/conversion error occurs, a partial new text
   generation can remain and should not be treated as complete without a valid
   manifest and requested BigBeds.

The next highest-value validation is the OCT integration, because it will test
whether a downstream biological classifier can consume stable population
hypotheses, attach an orthogonal family label, propagate that label to reads,
and preserve the model's audit trail.
