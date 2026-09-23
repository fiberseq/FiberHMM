# CR, SR, XCR and cross-locus CR

The normal `fiberhmm-consensus` command and FiberBrowser use the same full
FiberHMM engine (`run_analysis`) and the same BAM preparation (`load_bam_payload`).
The lightweight historical harmonization engine remains available only through
the lower-level replay API; it is not the normal CLI or Browser analysis.

## Quick start

```bash
pip install -e '.[consensus]'
fiberhmm-consensus --bam calls.bam --bed windows.bed --output families
```

BAMs must be indexed. Chemistry is read from the established
`FIBERHMM-CHEMISTRY:v1:` BAM `@CO` comments (supported legacy producer metadata
is also recognized). If missing, supply `--chemistry ddda`, `dddb`,
`hia5-pacbio`, or `hia5-nanopore`. An explicit setting cannot override conflicting
metadata. CLI analysis does not modify source BAMs. FiberBrowser's Dataset info
panel can optionally persist the canonical comments for future use.

Repeat `--bam` for separate datasets. To group several files as one dataset,
use `--datasets datasets.json` instead:

```json
[
  {"dataset_id": "replicate1", "paths": ["/data/a.bam", "/data/b.bam"]},
  {"dataset_id": "replicate2", "paths": ["/data/c.bam"]}
]
```

Each dataset can include an explicit `chemistry` when metadata is absent. All
files in one dataset must have compatible chemistry. No viewport read sampling
or family-count cap is used. MAPQ, native opportunity, and DAF molecule-collapse
filters still apply and are recorded. Empty evidence produces an empty report,
not an invented family.

Mode is automatic: one Hia5 dataset uses CR, DAF uses SR, multiple datasets use
XCR (reported as SR/XCR when DAF strands are present). The current validated
Hia5 engine pools alignment directions, including ONT: alignment direction is
not treated as an independent chemical-strand measurement. Mode is not a CLI
switch. The result records both the engine and mode.

## Cross-locus CR (CL-CR)

```bash
fiberhmm-consensus --bam calls.bam --bed oriented_windows.bed \
  --pool-loci --consolidation-bp 10 --cores 4 --output pooled_families
```

Provide BED6 with unique names, explicit `+`/`-` strands, and equal window
widths. BED coordinates are **zero-based, half-open**. Choose the windows and
orientations yourself: no motif lookup, recentering or inferred strand occurs.
A minus window reverses base `i` to `end-1-i` and interval `[a,b)` to
`[end-b,end-a)`. Positions, hits, context-specific emissions and m5C masks remain
paired. The original genomic window and molecule identities are retained.
Local zero is the oriented window's first base; the pooled report axis is
`0..window_width`. To center motifs/TSSs, supply windows with the desired
landmark at the same oriented offset. No genome-wide BAM copy is required.

Original variable opportunity lattices remain individual observations. Pooled
views are not resampled onto a dense average lattice. A physical molecule seen
in multiple windows contributes one deterministic window view, following the
paper adapter, to avoid duplicated evidence. The pooling receipt lists excluded
views. This is not an occupancy estimator for every locus.

Without `--pool-loci`, each BED row is analyzed independently in genomic
coordinates. Multiple-window output includes an index report and numbered
subdirectories. `--region chr1:10000-10200` is an alternative with the same
zero-based half-open convention.

## Steps, parameters and resume

The workflow prepares native calls, fits native family distributions, fits
shared parents, consolidates hypotheses and resolves final representatives.
Default consolidation edge allowance is ±10 bp; use 5 bp for finer grouping.
The full reference policy remains 4,095 predictive draws, 10 folds, bounded
native edge allowance 2 bp, 100 fit iterations with a 500-iteration retry, and
99.9% predictive compatibility. These are not confidence/accuracy percentages.
Unsupported legacy knobs are rejected rather than silently ignored.

```bash
fiberhmm-consensus --schema > parameter_schema.json
fiberhmm-consensus --bam calls.bam --bed windows.bed \
  --stop-after native --output native_run
fiberhmm-consensus --resume native_run --start-at consolidation \
  --consolidation-bp 5 --output consolidated_run
```

`--resume` addresses a single-window or pooled run directory. For a multi-window
batch, resume its individual `window_000001` etc. directories. Resume defaults
to the final resolved stage. `--start-at consolidation` requires exact native
checkpoints and fails if any are missing/incompatible; it never silently refits.
Shared-parent fitting may still be required, especially after changing the edge
allowance. `--resume` without this restriction permits safe recomputation of
invalidated checkpoints. Source/evidence, implementation and numerical-library
signatures protect checkpoint reuse. Keep the cache with the run.

`--evidence evidence.json.gz --cache /path/to/fit_cache` also permits replay.
`--parameters options.json` exposes all supported stage groups, e.g.:

```json
{
  "input": {"minimum_mapq": 20, "correct_native": true},
  "families": {"physical_radius_bp": 10, "minimum_retention_groups": 2},
  "compute": {"cores": 4, "maximum_matrix_mb": 2048}
}
```

Common controls have direct flags (`--cores`, `--consolidation-bp`,
`--stop-after`). Mode fields are derived from the input even if an old parameter
file contains stale switches. Progress is printed on stderr with work counters;
`--json-progress` emits structured events. Counters indicate work, not ETA.
Existing output directories must be empty; runs are never overwritten.

## Reports and plotting data

Open `report.html`. It links to `families.svg`, `families.tsv`, `calls.tsv` and
`report_data.json`. Tables contain all completed stages, fitted geometry,
uncertainty, fit warnings, dataset/strand counts and multi-compatible call
memberships. Pooled call rows retain the original window. `manifest.json`
records parameters, engine hashes, timings, numerical environment and cache
hits; `evidence.json.gz` and `result.json.gz` preserve replayable evidence and
frozen results. A family bar is its fitted mean geometry, not the original call.
Counts may overlap between families and must not be summed as exclusive
abundances or interpreted as calibrated ChIP occupancy probabilities.

FiberBrowser requires a compatible FiberHMM providing this API and displays an
installation command for its own Python environment if unavailable. Restart
Browser after upgrading. It does not substitute a lighter engine. For numerical
reproducibility launch the Browser with single-threaded BLAS; the CLI scopes
BLAS to one thread itself. The environment is included in the manifest.

## Reproducible family-tagged BAMs

BAM input produces indexed derivative BAMs in output/bams by default.
The default --bam-scope regions retains whole alignments overlapping the union
of analyzed genomic windows, including reads without family assignments.
Overlapping or disjoint windows do not duplicate a spanning alignment.
CL-CR uses the original BED windows, not its synthetic oriented coordinates.
Use --bam-scope full to retain all source records, including unmapped records.
Use --no-bam to retain only reports and frozen artifacts. By default,
multiple paths within one logical dataset are merged into one coordinate-sorted
BAM; --bam-grouping files writes one BAM per source file. Original BAMs are
never overwritten. Merged exports retain/reconcile read groups and program
records; reads lacking a read group receive an explicit source-provenance group.

In FiberBrowser, open **Write family-tagged BAMs** after a completed run, choose
a new or empty output folder, then choose **One BAM per dataset as currently
grouped** (default) or **One BAM per original source file**. Grouping is read
at export time, so merging or unmerging the view after inference is respected
for the run's source files. Export uses the frozen automatic final stage, not
display filtering or manual grouping edits. The **BAM contents** selector defaults to **Analyzed regions only**;
**Full source BAMs** is also available. Only analyzed calls receive annotations.
The export receipt records the scope, source windows, and written alignment count.

The established molecular-annotation convention is used:

* Native tf, nuc, MSP and other MA annotations remain intact.
* Reruns replace this producer's previous family layers and catalog, and rebuild
  its generated source read groups. Original sequencing/library RGs survive,
  with source provenance in DS. Unknown producers' target layers are not overwritten.
* Derived tf_consensus.QQQQQQ (CR/SR) or tf_cross_consensus.QQQQQQ (XCR) uses
  AQ dimensions tq,fi,fq,op,sq,q0; AN carries the full stable family token.
  (Earlier exports used QQQQ without sq/q0, then QQQQQ without q0.)
* q0 (class support) is the class's share of the call's evidence among every
  displayed class the call was scored against, x255: w_k = exp(recipient_optimum_k
  - floor_adjusted_loss_k), uniform prior. It is a relative profile-likelihood
  share, not a calibrated probability; it does not change with the assignment
  stringency. Each membership row carries that class's own share; 0 = unresolved.
* sq is the DAF molecule's own core protection ceiling for the family:
  1 + LLR x 10 (saturated at 255), where LLR is the protection log-likelihood a
  fully protected molecule would give from its own lattice sites and context
  emissions; 1 means no site in the core and 0 means non-DAF or unavailable.
  Each family's header catalog entry carries strand_resolution per dataset
  (trusted_strand CT/GA/both/none, core_resolution, per-strand median ceilings
  and the native floor); use it, not sq alone, to choose which strand to
  quantify. A strand is limited when its ceiling is below the native floor.
  On HG002 scDAF duplexes, a limited strand's calls were confirmed by the
  complementary strand at only ~0.65 precision (0.96 at >= 10 nats), and
  its errors were mainly extra calls, so its class rate should not be used.
* tq is native footprint LLR times 10 (saturated at 255), or the original native
  TQ if replay was disabled and available. Zero denotes unavailable when no
  source score exists. fi is the established locally reusable byte slot;
  AN plus the embedded catalog gives authoritative family identity across loci.
* **fq=0 explicitly means unavailable for this producer.** The staged engine
  has predictive compatibility, not a calibrated assignment probability.
  No confidence value is manufactured. op is the representative call's
  opportunity count, saturated at 255.
* All compatible family memberships are preserved as separate named
  annotations. They are nonexclusive and must not be summed as molecule counts.
* FIBERHMM-CONSENSUS-MA:v1: and FIBERHMM-CONSENSUS-FAMILY:v1: header comments
  preserve score semantics, model keys, family-slot mappings, run parameters,
  implementation hashes and stage. Existing chemistry comments are retained.
* Molecular coordinates follow the existing MA convention, including reverse
  alignments. Pooled BED coordinates are inverted to the original locus first.
  PCR aliases inherit the representative's classification; this is declared
  in the BAM contract.
* Unknown, changed or incompletely mapped source alignments fail export rather
  than being silently matched by read name. The output folder must be empty.

The shared read_family_catalog(bam.header) helper in
fiberhmm.inference.consensus.bam_export reads the embedded catalog without
external tables. FiberBrowser decodes the MA/AQ/AN family layers and discovers
them from MA-TYPES comments on reload. Frozen JSON retains full-precision
predictive evidence and alternatives beyond the compact BAM encoding.

## Apply frozen families to new windows

Use the same fitted families across loci or datasets without fitting recipient
families. First export the final converged models from an oriented CL-CR run:

```bash
fiberhmm-transfer --freeze-run pooled_result --output frozen_catalog
fiberhmm-transfer --models frozen_catalog/frozen_models.json.gz \
  --bam target.fiberhmm.bam --bed oriented_targets.bed --output transferred
```

The model bundle is versioned, digest-checked JSON (including explicitly typed
numerical arrays), not executable Python serialization. It contains the exact
native-cell or bounded-parent model and hashed training-molecule identities.
Final displayed aliases are resolved to their actual fitted models. Unconverged
models are excluded. A source CL-CR run must retain its native/source/parent
artifacts for the export step; the resulting bundle is self-contained.

Target BED6 must provide explicit + or - orientations and windows of the same
width as the source analysis. BED start on + and BED end on - define the shared
oriented frame. Historical bundles may declare a nonzero analysis origin;
transport accounts for it explicitly. The user chooses biologically meaningful
anchors and compatible reference assemblies. No gene, motif or RNA landmark is
inferred automatically.

Multiple --bam arguments represent separate datasets. --datasets accepts the
same dataset JSON as fiberhmm-consensus and supports multiple paths per dataset.
BAM preparation uses the shared native loader, chemistry comments and native
replay. --parameters exposes its preparation/memory controls. --evidence accepts
saved oriented native observations for exact replay. It does not reread BAMs or
regenerate their emissions. --json-progress emits structured stderr updates.

Scoring uses the existing native transfer and bounded-parent predictive kernels,
4,095 Monte Carlo replicates, and the established 99.9% reference policy. There
is no target family nomination, consolidation or refitting. Training molecules
are excluded using physical identifiers, PacBio molecule names and source-member
aliases; duplicate target molecules within a window fail instead of inflating
denominators. Cross-window observations are reported separately and are not
claimed to be independent.

Outputs include:

- families.tsv: eligible, assessed and compatible molecule counts per family/window.
- calls.tsv: original spans and all compatible family identities.
- window_*.json.gz: complete scores, unassessed reasons and exclusions.
- families.svg and report.html: frozen mean spans and summary tables.
- manifest.json: frozen-model digest, input digests and source provenance.
- bams/: indexed MA/AQ/AN subset exports by default; --no-bam disables them,
  --bam-scope full retains full source BAMs, and --bam-grouping files separates files.

The eligible denominator is coverage of the fitted mean by an aligned MSP without
nucleosome overlap. An eligible molecule can still have insufficient information
for a predictive decision; assessed counts are reported separately. The compatible
fraction is an empirical feature, not a calibrated ChIP occupancy probability.

Optionally supply --chip-bed peaks.bed for an external-label check. Peak overlap
with the supplied genomic window is joined after scoring. chip_evaluation.json
reports per-dataset/per-family AUROC and average precision of the direct
compatibility fraction, when both label classes exist. These are descriptive
metrics without confidence intervals or target fitting. Choose held-out windows
and appropriate matched controls before interpreting discrimination. The paper's
frozen logistic predictor and genomic-block bootstrap remain separate analyses;
this command does not silently fit a new classifier.

### Footprint-paired duplex molecules

Run `fiberhmm-pair` (pair -> merge -> recall, the default) on the called,
coordinate-sorted source BAM before population consensus. The merge recaller uses
both assay channels together (including the rotational nucleosome recaller).
Consensus preparation preserves the `cs` source identities and counts the
merged read once; `deam+` and `deam-` MA coverage masks determine which C/G
opportunities are observed. Missing channel coverage and source deletions do
not become protected observations. The merged BAM retains `pm`, `dm`, `mg`,
and `mv` pairing provenance.

Unmerged records with live `mt:P`/`mp` pair annotations cannot enter population
CR as independent molecules. Merge intentionally preserves failed pairs by
default so source data are not silently discarded. Check its failure count;
`fiberhmm-pair --pairs-only` produces only successfully merged joint
molecules and excludes unresolved pairs and ordinary unpaired reads. Keep the
source BAM for inspection. Use the merge recaller for joint reads; ordinary
single-strand calling commands are not a supported way to recall this output.

### Validation on copied or synchronized source trees

Copied Python bytecode can retain filenames from a previous drive, and Numba
cache replacement can fail on synchronized Windows folders. Use fresh local
cache directories for release checks, for example:

```bash
OPENBLAS_NUM_THREADS=1 \
PYTHONPYCACHEPREFIX=/tmp/fiberhmm-pycache \
NUMBA_CACHE_DIR=/tmp/fiberhmm-numba \
python -m pytest -q
```

Build a release wheel from a clean staging copy without `build/`, `*.egg-info`,
`__pycache__`, or Numba cache files. This avoids importing copied artifacts or
reusing a locked build directory; it does not change the numerical algorithm.
