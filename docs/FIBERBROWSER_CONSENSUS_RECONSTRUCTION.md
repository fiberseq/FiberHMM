# FiberBrowser implementation specification: consensus reconstruction

Status: implementation handoff, 2026-07-14  
Owner: FiberBrowser (regional analysis and interaction), not FiberHMM inference

## 1. Purpose and ownership

Consensus reconstruction (CR) is an **alternative-annotation hypothesis** for a
baseline nucleosome call. It asks:

> Can this protected interval be reconstructed from one or more recurrent TF
> footprint geometries observed on other molecules at the same locus?

CR does not re-evaluate methylation or deamination, does not replace the
baseline FiberHMM caller, and does not claim that a nucleosome is wrong. It is
an aggressive, secondary, focal analysis intended for high-depth loci where
the browser can show both interpretations and let the user move between them.

FiberBrowser owns:

- region and cohort selection;
- full-depth regional execution and caching;
- the connected N/TF threshold control;
- inspection of alternative configurations; and
- optional export of a regional alternative-hypothesis BAM.

The computational kernel should be a deterministic, testable module rather
than UI code. It may live in the FiberBrowser backend or a small standalone
library used by that backend. It must not import FiberHMM's chemistry models.

FiberHMM continues to own baseline `nuc`, `msp`, and `tf` calls and the
chemistry-aware MSP-to-TF plus TF/nuc edge-normalization layers (`nuc_sr` and
`tf_sr`). CR is deliberately absent from the FiberHMM command-line interface
and release package. The complete SR contract is
[`strand_rescue.md`](./strand_rescue.md).

### Production FiberHMM SR v4 handoff

FiberBrowser should display `nuc_sr` and `tf_sr` as one thresholded SR view, but
must parse it independently from CR. The authoritative v4 BAM header comment
starts with `FIBERHMM-STRAND-RESCUE:v4:` and declares `quality_spec=QQQ`,
`threshold_named_only=true`, and `display_sr_if=q0>=T`. `MA-TYPES:v1` must also
advertise both `nuc_sr` and `tf_sr`. Do not apply these rules to legacy v2/v3
`QQQQQ` layers merely because their group names are similar.

The ordinary `nuc`, `msp`, and `tf` groups are the immutable baseline
hypothesis. The SR groups are complete shadows plus explicit alternatives:

- `nuc_sr` contains exactly one shadow for every ordinary `nuc`;
- `tf_sr` contains exactly one shadow for every ordinary `tf`, plus zero or
  more rescued TF components; and
- an unnamed shadow has the same interval as its ordinary source and the
  sentinel quality row `(255,0,0)`.

The sentinel `q0=255` is **not** evidence favoring an SR alternative. It means
that there is no alternative for this call. This is why the header says
`threshold_named_only=true`: only named `H` and `Rn` annotations enter the SR
threshold decision.

#### Positional parsing and role names

Parse `MA`, `AQ`, and `AN` positionally. Each annotation consumes the number of
bytes in its MA quality specification, and `AN` must contain one field per MA
annotation (empty/`.` is unnamed). A named v4 SR annotation must match:

```text
^(fhsr_[0-9a-f]{16})(?:_O([0-9]+))?_(H|R([0-9]+))$
```

Interpret the fields as follows:

- `fhsr_<token>` is the decision/group ID;
- `fhsr_<token>_O<n>_H` is a singleton same-class edge alternative whose
  source is ordinary annotation ordinal `n`; and
- `fhsr_<token>_R0`, `_R1`, ... are the components of one MSP-to-TF rescue.

An `H` source ordinal is zero-based within the corresponding ordinary call
type, in the order those calls occur across all MA groups of that type on the
record. Thus a `nuc_sr` `_O2_H` points to ordinary `nuc` number 2, and a
`tf_sr` `_O2_H` points to ordinary `tf` number 2. Use this ordinal rather than
interval equality: duplicate ordinary intervals are legal. Each ordinary
source may be represented only once in its shadow layer. An `Rn` name must not
carry an `_O<n>` field because it adds a TF configuration to an ordinary MSP
rather than replacing an ordinary TF shadow.

Reject or quarantine a malformed SR view instead of guessing if any of these
invariants fail: an `H` lacks a valid unique source ordinal; an `H` appears in
the wrong shadow class; one decision mixes `H` and `Rn`; `Rn` indices are not
unique and contiguous from zero; or components of one rescue do not share one
`q0`.

#### `QQQ` semantics and threshold behavior

All three bytes are linear `[0,1]` model scores encoded as `round(255 * p)`:

| byte | meaning |
|---|---|
| `q0` | probability of choosing this named SR hypothesis over its ordinary baseline hypothesis |
| `q1` | confidence in this component's molecular-left alternative edge |
| `q2` | confidence in this component's molecular-right alternative edge |

For an `H`, `q0` compares the displayed alternative edges with the explicitly
linked ordinary source edges. For an `Rn` group, `q0` compares the complete
selected TF configuration with leaving the ordinary MSP unpromoted. `q1` and
`q2` are component-local diagnostics: different members of a multi-TF rescue
may have different edge confidences even though every member must share the
same `q0`. An unchanged edge of an `H` is encoded as 255 by convention. Do not
multiply the edge bytes into `q0`, and do not threshold components separately
on `q1` or `q2`.

For slider threshold `T` in `[0,255]`, construct the normalized view as:

```text
for each ordinary nuc or tf source:
    if it has a named same-class H shadow and H.q0 >= T:
        display the H interval
    else:
        display the ordinary source interval

for each decision token containing R0..R(k-1):
    assert every component has the same q0
    if q0 >= T:
        display all k tf_sr components
    else:
        display none of them              # the ordinary MSP remains baseline
```

The comparison is inclusive (`>=`). A multi-component `R` is atomic even when
its members have different `q1`/`q2`: the browser must never show only the
best-edged component. Ordinary baseline layers should remain separately
available for inspection, but the normalized SR view must not draw both an
ordinary source and its selected `H` replacement as if they were two calls.

`H` is strictly same-class, one-for-one edge normalization:

- ordinary `tf` -> `tf_sr` alternative edges;
- ordinary `nuc` -> `nuc_sr` alternative edges; and
- never TF <-> nucleosome reclassification, nucleosome demotion, splitting,
  merging, creation, or deletion.

Consequently `nuc_sr` cardinality always equals ordinary `nuc` cardinality.
An `H` in `nuc_sr` does not pair with a TF in `tf_sr`, and an `H` in `tf_sr`
does not imply removal of a nucleosome.

#### Molecular edge orientation

MA intervals and `q1`/`q2` are in original-molecule 5'->3' coordinates, not
reference-left/reference-right coordinates. Keep each quality attached to its
molecular boundary while projecting through the alignment. For a reverse-
mapped record, the browser's genomic-left tooltip is therefore `q2` and its
genomic-right tooltip is `q1`; for a forward-mapped record they are `q1` and
`q2`, respectively. Do not swap the bytes merely while parsing MA--FiberHMM
has already encoded them in molecular orientation.

`Rn` role indices identify configuration membership, not rendered left-to-
right order. Sort rectangles by projected coordinates for drawing while
retaining each member's role and its own quality row.

#### SR is not the CR connected state pair

SR alternatives are paired with ordinary baseline annotations by the rules
above, **not** with each other. Never require or compute
`nuc_sr.q0 + tf_sr.q0 == 255`, never connect an SR nucleosome to an SR TF by
spatial overlap, and do not route SR through the CR state slider in section 9.
One SR slider may consistently apply the named-alternative rule to both `H`
and `R`, but its fallback is always the ordinary baseline hypothesis.

CR export is deliberately different: `nuc_cr.QQQQQ` and `tf_cr.QQQQQ` are
explicit complementary N-versus-reconstruction members of one decision;
their state bytes sum to 255, and all TF components of the selected CR
configuration share the paired decision row. Preserve that CR contract and
its separate UI state.

## 2. Non-goals and hard prohibitions

CR must not:

- read or score `MM`, `ML`, IPD, pulse, raw-current, deamination, or sequence
  context evidence;
- use a PWM, motif score, external occupancy prior, or HiddenFoot-style motif
  prior;
- use another library or assay as an inference prior;
- use the browser's display-downsampled reads as the analysis cohort;
- treat forward/reverse PacBio alignment orientation as physical strands;
- impose a biological 220 bp maximum on the baseline block;
- silently convert the result into authoritative baseline calls; or
- require a configuration to have been observed intact on one source read.

Independent assays are valuable validation controls, but never inference
inputs. DddA, DddB, PacBio Hia5, and Nanopore Hia5 may each run CR on their own
annotations; their reads and priors must not be mixed.

## 3. Input contract

### 3.1 Cohort

One CR request contains one explicitly named inference cohort. Repeated BAMs
are allowed only when they are shards or compatible timepoints intended to be
pooled as one population. BAM identity is retained for read lookup and export,
not used to partition or borrow priors.

Required per input:

- stable file identifier and canonical path/URL;
- size, modification time, and preferably SHA-256 or an equivalent immutable
  object version;
- indexed, coordinate-sorted BAM/CRAM access;
- a post-TF/post-nucleosome `MA` annotation containing ordinary `nuc` and `tf`
  groups; and
- reference assembly/contig identity.

The request coordinate system is **0-based, half-open reference coordinates**.
MA coordinates remain molecular coordinates and must be projected through the
alignment. An annotation with less than 80% of its molecular bases mapped to
reference is ineligible as a source template or target block.

### 3.2 Request schema

The backend-facing request should follow this shape:

```json
{
  "schema": "fiberbrowser.cr.request.v1",
  "cohort_id": "user-visible stable identifier",
  "assembly": "dm6",
  "region": {"contig": "chr2R", "start": 9988750, "end": 9989118},
  "inputs": [
    {
      "file_id": "2-4hr_4",
      "uri": "/path/to/2-4hr_4.bam",
      "size_bytes": 123,
      "mtime_ns": 456,
      "sha256": "optional-but-preferred"
    }
  ],
  "parameters": {
    "min_tf_tq": 100,
    "minimum_template_support": 10,
    "minimum_local_enrichment": 1.5,
    "cluster_center_radius_bp": 10,
    "maximum_boundary_mad_bp": 12,
    "candidate_flank_bp": 75,
    "maximum_templates_per_candidate": 12,
    "maximum_configurations_per_candidate": 512,
    "exact_configuration_backoff": 10.0,
    "pseudocount": 0.5
  }
}
```

Every effective parameter, including defaults, must be returned in the
response and included in the cache key.

## 4. Analysis cohort versus displayed reads

The browser often downsamples molecules for rendering. CR must instead fetch
**all eligible records in the requested analysis interval** from every selected
cohort input. Changing the viewport's visual read limit, sorting, clustering,
or hidden-read filters must not change a cached CR result unless the user
explicitly promotes that filter into the analysis request.

Short reads are valid. A molecule contributes only to templates and coverage
whose relevant interval it maps. It is never required to span the entire
viewport or locus.

PCR/amplification handling is an upstream cohort decision. If the source BAM
contains duplicate-family annotations, the backend should count one selected
representative per family. Otherwise it must clearly report that reads, rather
than inferred original molecules, were counted. CR must not perform chemistry-
based duplicate discovery itself.

## 5. Elementary footprint templates

Templates are learned only from ordinary, explicit, high-confidence `tf`
annotations in the selected cohort.

1. Project each eligible TF call to reference coordinates.
2. Retain calls with `tq >= min_tf_tq`.
3. Cluster calls deterministically by center and compatible edges. A practical
   v1 implementation is center clustering within 10 bp followed by robust edge
   estimation within each center cluster.
4. For each cluster, record robust median start/end, start/end MAD, all member
   intervals, unique-molecule support, strand-neutral spanning depth, and local
   enrichment over an opportunity-matched surrounding window.
5. Reject clusters below support, enrichment, or boundary-stability minima.
6. Sort accepted templates by reference start, end, then a stable content hash;
   assign IDs from that order.

Calls on the candidate molecule must be held out when scoring that candidate.
The holdout affects template support, edge distributions, exact configuration
counts, and occupancy counts. If the holdout drops a template below eligibility,
that template is unavailable for that candidate.

Do not require all members of a proposed reconstruction to appear together on
one read. The motivating observation may be `TF1` on some reads, `TF2` on
others, and a fused protected interval on the target read.

## 6. Candidate nucleosome blocks

Every valid baseline `nuc` interval that overlaps the focal template envelope
is a candidate. Length is evidence, not eligibility:

- there is **no biological 220 bp ceiling**;
- there is no short-length trigger within the ordinary nucleosome range; and
- unusually long blocks may be analyzed when recurrent annotations genuinely
  provide a reconstruction.

Generic safety limits are permitted but must be reported as computational
limits, never biological conclusions. Examples are a maximum number of nearby
templates or configurations. A skipped candidate must return a structured
`not_evaluated` reason such as `configuration_limit`, not silently remain an
evaluated N call.

The candidate molecule is excluded from every population count used for its
own decision. Existing ordinary TFs already overlapping the same baseline nuc
make the target annotation internally contradictory; report that condition and
do not manufacture a second CR decision for it.

## 7. Configuration generation

For each candidate, gather eligible elementary templates that overlap the
candidate or fall within the configured boundary flank. Construct a
compatibility graph in which templates can coexist when their reference
intervals do not overlap. Enumerate every compatible non-empty subset, subject
only to explicit computational caps.

The state space must support:

- one TF;
- two or more TFs;
- three or more TFs when supported;
- gaps of arbitrary size between components;
- configurations never observed intact on a single read; and
- more than one plausible configuration for the same block.

Do not add a synthetic broad `TF_COMPLEX` rectangle merely because exact
decomposition is uncertain. The response can expose aggregate reconstruction
probability while retaining several concrete alternatives. Rendering may show
an unresolved envelope as a UI affordance, but the model output must preserve
the alternatives that created its probability mass.

When exhaustive enumeration exceeds the configured cap, use a deterministic
beam ordered by preliminary population weight and retain the top K states.
Return both the number generated and the number retained. Mark the posterior
as truncated so it is not presented as fully normalized without qualification.

## 8. Probabilistic model

CR is a call-space empirical-Bayes comparison between:

- `H_N`: retain the baseline nucleosome interpretation; and
- `H_R(C)`: reconstruct the interval as configuration `C`.

It intentionally contains no molecule chemistry likelihood.

### 8.1 Population configuration weight

For each candidate-held-out source cohort, calculate:

- `n_span(C)`: molecules spanning the configuration hull;
- `n_exact(C)`: spanning molecules whose explicit TF calls match `C` without
  an incompatible additional template inside the hull;
- `p_i`: smoothed marginal explicit-call probability for each component; and
- `p_ind(C)`: the product of selected `p_i` values and unselected `(1-p_i)`
  values for eligible templates inside the same hull.

Use an exact-count estimate backed off toward the marginal composition:

```text
p_config(C) =
  (n_exact(C) + kappa * p_ind(C)) / (n_span(C) + kappa)
```

where `kappa` is `exact_configuration_backoff`. This preserves directly
observed configurations while allowing an unseen combination of separately
well-supported components to receive non-zero weight.

### 8.2 Local nucleosome weight

Count candidate-held-out, spanning molecules whose baseline state across the
candidate center is an ordinary nucleosome and molecules with an explicit
compatible TF configuration. Use a symmetric Beta/Dirichlet pseudocount to
form a **local** N-versus-reconstruction state weight. Do not impose a fixed
genome-wide 10:1 N:TF ratio. Focal insulator or regulatory sites are allowed to
have a state balance radically different from the rest of the genome.

An ordinary TF annotation takes precedence over a contradictory overlapping
ordinary nuc when forming these counts. Uncalled or non-spanning molecules do
not become positive TF observations.

### 8.3 Boundary likelihood

The key geometric asymmetry is that a TF union predicts the protected block's
envelope, while a nucleosome at the locus may have a broader distribution of
positions and canonical lengths.

For each configuration, integrate the candidate start/end against empirical
component edge distributions. Use a smooth heavy-tailed kernel (Student-t is
recommended) whose scale is estimated from held-out explicit TF boundaries and
floored at a small sequencing/alignment tolerance. Do not use a hard 20 bp
pass/fail rule.

For `H_N`, use a broad position term and a smooth nucleosome-length density.
The length density may be estimated from valid same-cohort background nucs in
the loaded regional flank, excluding the focal template envelope. If regional
background is insufficient, use a documented weak canonical-length model.
Never fit the N geometry from the target call itself.

### 8.4 Normalization

For each candidate:

```text
w_N    = p_local(N) * p_nuc_length(length) * p_nuc_position(start, end)
w_R(C) = p_local(R) * p_config(C) * p_boundary(start, end | C)

P(R) = sum_C w_R(C) / (w_N + sum_C w_R(C))
P(N) = 1 - P(R)
P(C | R) = w_R(C) / sum_C w_R(C)
```

All calculations should occur in log space. Return raw log weights and all
counts so that calibration can be audited. Until held-out calibration is
complete, label `P(R)` a **normalized model probability**, not an empirically
calibrated probability of biological truth.

## 9. Connected quality score

Each decision exposes five linear 0--255 bytes:

| byte | name | definition |
|---|---|---|
| `q0` | state | `round(255 * P(reconstruction))`; the N side is exactly `255-q0` |
| `q1` | configuration | `round(255 * max_C P(C | reconstruction))` |
| `q2` | boundary | normalized compatibility of the candidate edges with the best configuration envelope |
| `q3` | population | reliability of the best configuration's held-out population support |
| `q4` | completeness | fraction of reconstruction mass retained after enumeration/caps, with 255 meaning complete |

`q0` is the only byte used by the connected state slider. The others are
diagnostics and optional filters. They must never be multiplied together after
the fact to create a different hidden decision score.

For browser threshold `T` in `[0,255]`:

```text
show reconstruction if q0_reconstruction >= T
otherwise show nucleosome
```

For a paired export, `q0_nuc + q0_tf` must equal 255 exactly. All TF components
of one configuration share the same five-byte row and switch atomically.

## 10. Response schema

The canonical result is JSON, independent of BAM export:

```json
{
  "schema": "fiberbrowser.cr.result.v1",
  "engine_version": "cr-callspace-v1",
  "request_sha256": "...",
  "cohort": {"cohort_id": "...", "inputs": []},
  "region": {"contig": "chr2R", "start": 9988750, "end": 9989118},
  "parameters": {},
  "templates": [
    {
      "template_id": "crt_0001",
      "interval": [9988800, 9988821],
      "support_molecules": 200,
      "spanning_molecules": 900,
      "start_mad": 2.0,
      "end_mad": 3.0,
      "local_enrichment": 18.4
    }
  ],
  "decisions": [
    {
      "decision_id": "stable-content-hash",
      "source": {
        "file_id": "2-4hr_4",
        "read_name": "molecule-name",
        "baseline_type": "nuc",
        "reference_interval": [9988794, 9988872],
        "molecular_interval": [511, 78]
      },
      "status": "evaluated",
      "state_probability": {"nuc": 0.31, "reconstruction": 0.69},
      "quality": {"nuc": [79, 220, 201, 188, 255], "reconstruction": [176, 220, 201, 188, 255]},
      "log_weight_nuc": -8.1,
      "log_weight_reconstruction_total": -7.3,
      "best_configuration_id": "crc_0004",
      "alternatives": [
        {
          "configuration_id": "crc_0004",
          "template_ids": ["crt_0001", "crt_0003"],
          "component_intervals": [[9988800, 9988821], [9988845, 9988867]],
          "hull": [9988800, 9988867],
          "probability_given_reconstruction": 0.86,
          "log_weight": -7.45,
          "support": {"spanning": 899, "exact": 73, "backoff_probability": 0.09},
          "edge_residuals": {"left_bp": -6, "right_bp": 5}
        }
      ],
      "diagnostics": {
        "candidate_held_out": true,
        "configuration_count_generated": 7,
        "configuration_count_retained": 7,
        "posterior_truncated": false
      }
    }
  ],
  "summary": {},
  "warnings": []
}
```

`status` is one of `evaluated`, `not_evaluated`, or `invalid_input`. A
non-evaluated decision must have a machine-readable reason.

## 11. Browser behavior

CR appears as an explicit opt-in regional analysis, not a default data layer.
Recommended interaction:

1. User selects one cohort and a region/viewport.
2. Browser states that full regional depth, not displayed reads, will be used.
3. Analysis runs in a cancellable worker/backend job.
4. A connected N/TF slider appears after results load.
5. Selecting a candidate reveals alternative configurations, source counts,
   edge residuals, held-out status, and all five quality values.
6. The baseline `nuc`/`tf` tracks remain available and unchanged.

At a threshold, show only one state for each decision. A multi-component
reconstruction is atomic: never show only TF1 because TF2 fell below an
independent threshold. If the user selects a non-best alternative manually,
label that as manual configuration inspection; it does not change `q0`.

## 12. Optional regional BAM export

Export is a FiberBrowser/backend responsibility and must create a new sorted,
indexed regional BAM. Never edit the source BAM.

Use MA-valid custom groups:

```text
nuc_cr.QQQQQ
tf_cr.QQQQQ
```

Export semantics:

- preserve original `nuc`, `msp`, and `tf` groups and qualities;
- copy unaffected baseline nucs/TFs into complete shadow layers with
  `(255,0,0,0,0)` and no paired `AN` label;
- give a challenged N and every component of its selected reconstruction the
  complementary/shared quality rows described above;
- link one N member and all TF members with an `AN` prefix such as
  `fbcr_<16hex>_N`, `fbcr_<16hex>_T0`, `fbcr_<16hex>_T1`;
- advertise `nuc_cr,tf_cr` through `MA-TYPES:v1`;
- record engine version, request hash, source fingerprints, quality meanings,
  and threshold rule in `@CO`/`@PG`; and
- reject export when a decision cannot be projected back to the exact source
  record and molecular interval.

The export format is a portable visualization artifact, not the canonical CR
result. JSON remains authoritative because it retains multiple alternatives
and detailed support.

## 13. Caching, limits, and reproducibility

Cache key:

```text
hash(engine_version, complete request, input object versions, region, parameters)
```

Recommended first implementation limits are configurable 100 kb regions,
50,000 fetched records, 12 templates per candidate, and 512 retained
configurations. These are engineering defaults, not biological thresholds.
Return a warning or structured `not_evaluated` state whenever a limit changes
the result surface.

Results must be deterministic across thread count, display downsampling, input
iteration order, and repeated execution. Stable sorting and content-derived
IDs are required.

## 14. Required tests

Unit tests:

- one-template reconstruction;
- two templates observed separately reconstruct one fused block;
- unseen combination receives marginal-backoff support;
- one TF plus a gap, three TFs, and competing configurations;
- a candidate longer than 220 bp remains eligible;
- candidate leave-one-molecule-out changes every applicable count;
- changing/removing MM/ML produces byte-identical CR JSON;
- another assay cannot enter the inference cohort;
- short reads vote only where they span;
- a canonical background nuc is retained;
- no hard boundary-tolerance discontinuity;
- exhaustive and capped enumeration report completeness correctly; and
- `q0_nuc + q0_reconstruction == 255` for every evaluated decision.

Integration tests:

- visual downsampling does not change results;
- cache invalidates on any input or parameter change;
- atomic connected-slider behavior for multi-TF alternatives;
- reverse-read molecular-coordinate export;
- MA/AQ/AN positional integrity and idempotent re-export;
- source BAM bytes remain unchanged; and
- exported BAM quickcheck, coordinate sort, and index access all pass.

Biological validation should include Homie, Nhomie, SF1, and SF2 using the five
pooled 2--4 hr fly PacBio BAMs. Use held-out synthetic fusions and ordinary
background nucleosomes for calibration; do not tune solely until the named
positive loci look desirable. DddA/DddB/Nanopore may be evaluated separately
as annotation-quality stress tests, never pooled with PacBio.

## 15. Acceptance criteria for v1

The FiberBrowser implementation is ready for experimental use when:

- CR reads only post-call annotation geometry;
- all regional reads, not display samples, drive inference;
- unseen combinations of separately observed templates are supported;
- no biological 220 bp ceiling exists;
- all counts and log weights are inspectable in canonical JSON;
- connected state scores are complementary and multi-TF states atomic;
- results are deterministic and candidate-held-out;
- an optional regional BAM export is MA/AQ/AN-valid and non-destructive; and
- cross-assay data can be used for evaluation without any code path that lets
  it alter the inference result.
