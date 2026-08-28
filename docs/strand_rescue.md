# Two-strand TF rescue and edge normalization

`fiberhmm-strand-rescue` is a focal secondary caller for assays in which each
alignment reports one biochemical/read strand: DddA DAF-seq, DddB DAF-seq, and
Nanopore Hia5 Fiber-seq.

It is an optional exploratory pass, not a replacement for ordinary FiberHMM.
Ordinary `nuc`, `msp`, and `tf` annotations remain unchanged and authoritative.
SR emits a complete shadow layer of alternatives, including ambiguous
alternatives, so downstream tools can select a quality threshold without
destroying or rewriting the baseline interpretation.

SR treats the two strand populations as imperfect observations of shared
biological footprints. It performs three related operations:

1. recover a TF missed on one strand when it remains an MSP there, has
   weak-but-positive hard chemistry evidence, and is recurrent on the opposite
   strand;
2. normalize accepted TF calls onto a strand-balanced shared footprint
   geometry; and
3. normalize accepted nucleosome calls onto a strand-balanced shared edge
   geometry.

The third operation is edge-only. SR never promotes a TF from a nucleosome and
never promotes, demotes, splits, merges, creates, or removes a nucleosome. Each
ordinary nucleosome has exactly one call in the normalized shadow layer. No
nucleosome-length ceiling is imposed.

This is independent of consensus reconstruction. SR reads molecule chemistry
and belongs in FiberHMM. CR compares alternative annotation combinations
without chemistry and belongs in FiberBrowser.

## Supported inputs

Inputs must be coordinate-sorted, indexed, and contain post-nucleosome/post-TF
Molecular Annotation (`MA`) calls.

| preset | strand groups | hard observations | TF model | nuc-edge model | default hard-call threshold |
|---|---|---|---|---|---|
| `ddda` | `CT`, `GA` | standard DAF mismatches/native hard calls | `ddda_TF.json` | `ddda_nuc.json` | n/a |
| `dddb` | `CT`, `GA` | standard DAF mismatches/native hard calls | `dddb_nanopore.json` | same | n/a |
| `hia5-nanopore` | `FWD`, `REV` | standard Dorado `MM`/`ML` m6A | `hia5_nanopore.json` | same | `ML >= 248` |

`--model` and `--nuc-model` can override those defaults. The separate DddA
nucleosome model is used only for evidence at changed nucleosome edges; it does
not add a nucleosome state to TF rescue.

PacBio Hia5 is intentionally unsupported. A HiFi molecule already exposes both
A and T channels; its alignment flag is not a biochemical strand.

SR reads standard aligned sequence, hard MM/ML or DAF calls, and MA. It never
reads IPD, pulse, raw-current, or sub-threshold modification evidence.

## Cohort and molecule semantics

Repeated `--bam` inputs are one explicitly pooled inference population. Use
them for shards or compatible timepoints of the same assay. BAM identity is
retained only for output routing and to prevent PCR-family collapse across
independently amplified inputs.

Another assay or biological library must not be supplied as a prior.
Cross-assay overlap is validation, not inference.

For amplified DddA/DddB, `--molecule-collapse auto` clusters full-read hard
deamination fingerprints independently within each BAM. One representative per
inferred family teaches population support and geometry, but every raw input
alignment remains an output target. Collapse therefore cannot make PCR-family
members disappear from a normalized BAM.

Short reads contribute wherever they fully map a footprint. They never need to
span the complete requested locus. A source call is excluded from geometry
learning when its alignment does not extend beyond both call edges by the
configured margin. The defaults are 10 bp for TFs
(`--source-boundary-margin`) and 20 bp for nucleosomes
(`--nuc-source-boundary-margin`). This prevents clipped calls from teaching a
false boundary while preserving their use at sites they genuinely span. The
report records `boundary_excluded_support` separately for each strand.
An ordinary TF or nuc annotation with less than 80% mapped molecular span is
also excluded from geometry learning, but retained as a topology-only obstacle:
an edge alternative can never overwrite it merely because most of the call is
soft-clipped.

## Shared TF vocabulary

Every geometry-eligible ordinary `tf` annotation can support a recurrent
footprint, regardless of TQ. The baseline recaller has already made the
call/no-call decision; its `tq` is not an additional source-call validity
threshold for SR. Calls with less than 80% mapped molecular span remain
topology-only obstacles, and calls failing the source boundary margin do not
teach geometry.

TF site construction is:

1. project ordinary TF intervals to reference coordinates;
2. cluster recurrent centers independently on each strand;
3. merge matching strand clusters into one footprint family;
4. record unique-molecule support, local enrichment, and robust start/end
   geometry separately for each strand; and
5. calculate one canonical start/end from the median of the represented
   strand-specific medians.

Each strand receives one geometry vote regardless of read depth once it has
`--minimum-geometry-support` calls (default 3). Thus a deeply sequenced strand
cannot drag the canonical interval toward itself merely by having more reads.
The report retains each strand's medians, MADs, call counts, clipped-source
exclusions, and disagreement with the canonical interval.

`--site START-END` is a seed, not an authoritative footprint size. Its center
nominates the family; support and canonical edges are learned again from all
ordinary calls in the selected cohort. If the cohort has no supporting calls,
the seed cannot manufacture an opposite-strand prior.

## Nucleosome edge populations

Ordinary nucleosomes can be multimodal at one center. SR therefore discovers
nucleosome geometry families jointly in start/end space rather than forcing all
nearby calls onto one median. `--nuc-edge-assignment-radius` controls the
maximum edge displacement considered during family construction, while
`--nuc-center-radius` limits center displacement. Support, local enrichment,
strand-specific medians, boundary MAD, and the strand-balanced canonical edges
are then calculated independently for each family.

`--nuc-site START-END` seeds one family and relearns its edges from ordinary
nucleosome calls. `--forced-nuc-sites-only` disables automatic nuc-family
discovery. Neither automatic nor seeded discovery imposes a 220 bp or other
biological length ceiling.

## Source-strand occupancy model

For each TF family and strand, SR fits a local `A / TF / N` mixture from the
hard observations of molecules spanning that family:

```text
log L(A)  = 0
log L(TF) = sum protected-vs-accessible LLR inside the canonical footprint
log L(N)  = sum protected-vs-accessible LLR across the local protected window
```

A geometry-eligible ordinary TF on a molecule spanning the family is anchored
to the TF component; SR does not use its chemistry to re-test that accepted
call. A geometry-eligible ordinary nuc covering the family center anchors the
source-side N nuisance component. Topology-only annotations never anchor the
model. Uncalled molecules, including MSPs, remain chemistry-scored so the
mixture can still represent missed footprints.

The N component remains in this source-population fit so generally protected
regions do not become artificial TF priors. When the target annotation is an
MSP, inference conditions on the relevant `A / TF` alternatives; N is not a
target state.

For multiple nearby families, SR composes the source per-site marginals into
every compatible non-overlapping subset. This permits one TF, one plus a gap,
two, three, or more TFs without requiring a short source read to span the
complete configuration. It does not claim to estimate pairwise co-occupancy.

## MSP-only missed-call recovery

For target strand `s`, a TF family may supply a prior from the opposite strand
only when it meets the source support and focal-enrichment requirements and is
at least as well represented there as on `s`. Representation is the accepted
TF-call fraction among reads that fully map the family, not its raw call count
or center-only depth. Center-only coverage is retained as a diagnostic but
cannot dilute the source fraction. This prevents unequal strand depth and
short edge-clipped reads from choosing the source direction.
The absolute `--strand-min-source-support` requirement still ensures that a
focal prior is backed by recurrent calls.

SR examines only ordinary MSPs. A canonical footprint must be completely
contained by the MSP and fully mapped by the alignment; center-only overlap is
insufficient. A family that overlaps any ordinary TF on the target molecule is
blocked, even if shifted edges placed that TF outside the center-radius matcher.
SR never adds a second, overlapping version of an already accepted footprint.

Each selected TF component must have:

- at least one informative target opportunity; and
- a strictly positive protected-versus-accessible LLR.

This is the “borderline, not unsupported” rule. The target molecule does not
need to pass the baseline caller's full TF threshold by itself. SR combines its
weak positive likelihood with the opposite-strand occupancy prior.

An ordinary nucleosome containing a source-supported TF family is counted as
`source_sites_inside_nucs_ignored` and receives no TF-rescue decision. The
separate nuc-edge pass can still normalize that nucleosome one-for-one; it does
not reinterpret the nucleosome as TF occupancy. TQ and DddA subnucleosomal
fingerprints do not enter SR eligibility.

## Per-molecule enzyme calibration

Absence of a hard modification is meaningful only relative to that molecule's
accessible hard-call rate. By default, SR estimates the rate from the
molecule's ordinary MSPs, accounts for context composition, and shrinks it
toward the model rate with a 20-opportunity pseudocount.

This matters especially for DddB and Nanopore Hia5, where incomplete
saturation can otherwise make an inefficient molecule appear protected. DddA
TF rescue and DddA nuc-edge evidence are calibrated against their respective
models.

## Existing-call edge normalization

An existing TF or nucleosome call is eligible for shared edges only when both
strands have at least the configured geometry support assigned to the same
family. Candidate calls are matched jointly and one-for-one to nearby families.
Assignment uses robust start/end residuals and includes an explicit unmatched
null, so a call is retained at baseline rather than forced onto an implausible
population.

For an assigned call, SR compares the canonical and current edges while holding
footprint identity fixed. Shared interior bases cancel. Protected-versus-
accessible log-likelihood steps are added only for bases newly included by the
canonical interval and subtracted only for bases removed from it. Left- and
right-edge Bayes factors, opportunities, hard hits, and the joint Bayes factor
are all retained in the report.

Edge materialization is intentionally aggressive. Family assignment must beat
the explicit null, and the canonical interval must have mapped endpoints and at
least 95% mapped reference coverage. Chemistry quality is reported but is not a
second acceptance gate: prior-only changes, chemistry-opposed changes, and
large shifts remain visible and are counted explicitly.

SR rejects every member of an edge-update component that would introduce a new
overlap anywhere in the combined TF/nuc shadow callset. It also rejects updates
that would invert call order within one layer. If a newly expanded edge would
collide with an otherwise valid MSP-to-TF rescue, the edge retains its baseline
coordinates and the rescue remains available. Existing baseline overlaps are
grandfathered. The original edge is retained for unassigned, incompletely
mapped, or topology-conflicting calls.

Nucleosome identity and cardinality remain fixed throughout this process. A
nucleosome may move or resize only in `nuc_sr`; ordinary `nuc` stays unchanged.
No edge update can become a nuc-to-TF conversion, split, merge, or demotion.

## Probability report

For a locally supported MSP, let `H*` be the exact nonoverlapping TF
configuration selected by SR, let `S` contain all locally supported TF
configurations, and let `A` be the ordinary MSP baseline. The v6 rescue
probability is the selected configuration's mass in the complete supported
action set:

```text
P(H* | A or S) = exp(log score(H*))
                 --------------------------------------------
                 exp(log score(A)) + sum[H in S] exp(log score(H))
```

where each log score combines the opposite-strand population prior with the
target molecule's hard-chemistry likelihood. A strong TF-versus-accessible
signal therefore cannot give one exact footprint family high quality when
several size or position families remain plausible. The report retains the
probability partition among the accessible state, the selected family, and all
other supported families, plus the legacy pairwise selected-versus-accessible
value for diagnosis.

The proposal tiers use the exact-configuration probability:

- `strong`: `P(H* | A or S) >= 0.95`;
- `review`: `0.5 <= P(H* | A or S) < 0.95`; and
- `retain_current`: `P(H* | A or S) < 0.5`.

These tiers are summaries, not inference gates. Low-quality alternatives are
part of the intended exploratory output. The current JSON schema is
`fiberhmm.strand_rescue.v6`. It records all parameters, input and model
provenance, molecule calibration/collapse diagnostics, TF and nuc geometry
families, source models, MSP decisions, and
`strand_rescue.edge_refinement.{tf,nuc}` decisions. Edge counts distinguish
updates, already-canonical calls, explicit-null/unassigned calls, incomplete
spans, topology conflicts, prior-only updates, chemistry-opposed updates, and
extreme shifts.

The posterior and edge probabilities are normalized model probabilities, not
yet held-out calibrated probabilities of biological truth.

## Normalized MA layers

`fiberhmm-strand-rescue-annotate` creates new sorted, indexed regional BAMs
containing two complete shadow layers:

```text
nuc_sr.QQQ
tf_sr.QQQ
```

`nuc_sr` contains exactly one interval for each ordinary `nuc`. `tf_sr` contains
one interval for each ordinary `tf`, plus every materialized MSP-to-TF rescue.
An unchanged shadow call is unnamed and receives `(255,0,0)`. This row is a
completeness sentinel, not a probability claim. An accepted one-for-one edge
update is a named singleton `H` group. Rescued TF components use one shared
name prefix with contiguous `R0`, `R1`, ... roles. They share `q0` and switch
atomically, while each component retains its own edge confidences. `R` roles
are never valid in `nuc_sr`.

The three linear 0--255 bytes have one role-independent display contract:

| byte | rescued TF (`Rn`) | edge-normalized TF or nuc (`H`) | unchanged baseline |
|---|---|---|---|
| `q0` | exact selected TF configuration within accessible plus all supported TF configurations; `0` if the action set was truncated | canonical shared edges versus this exact ordinary interval | sentinel `255`; ignore for thresholding |
| `q1` | canonical molecular-left edge reliability for this TF component | marginal canonical-versus-current molecular-left edge posterior; `0` if that changed edge has no target opportunity | sentinel `0` |
| `q2` | canonical molecular-right edge reliability for this TF component | marginal canonical-versus-current molecular-right edge posterior; `0` if that changed edge has no target opportunity | sentinel `0` |

For `R`, all components share the exact-configuration `q0` defined above.
`q1` and `q2` are component-specific canonical-boundary reliability scores
derived from molecule support, boundary MAD, whether both strands contributed,
and strand-to-strand boundary agreement. They are confidence scores, not a
second rescue gate. The stored alternative byte is
`q0 = round(255 * P(H* | A or S))`.
If `--maximum-sites-per-decision` truncates a local family set, the proposed
exploratory alternative is retained with `q0=0`; the report records every
dropped site ID, `q0_action_set_complete=false`, the conditional score within
the considered subset, and unresolved probability one. This prevents a
computational safety cap from masquerading as confidence.

For `H`, `q0` is an equal-prior posterior for the bivariate canonical
start/end hypothesis versus the ordinary start/end hypothesis. Its population
log Bayes factor is a regularized plug-in Gaussian predictive score fitted to
molecule-collapsed opposite-strand start/end pairs using robust marginal scales
and a regularized edge correlation, then added to the target molecule's joint
changed-base chemistry log Bayes factor. Source depth reduces uncertainty in
the fitted population location but does not multiply the same prior evidence
once per molecule. `q1` and `q2` use the corresponding marginal predictive
evidence plus that edge's target chemistry. An edge shared by both hypotheses
contributes no evidence and receives `255` confidence.
This probability is conditional on the selected geometry family and the fitted
same-cohort model. The BAM `q0` then multiplies that conditional probability by
the call's explicit geometry-family assignment probability; competing families
or the geometry null therefore reduce confidence in the exact edge alternative.
The report preserves the conditional, assigned-baseline, and unresolved-family
parts separately.

```text
q0 = round(255 * logistic(log BF(population, start+end)
                          + log BF(target chemistry, start+end)))
q1 = round(255 * logistic(log BF(population, left)
                          + log BF(target chemistry, left)))
q2 = round(255 * logistic(log BF(population, right)
                          + log BF(target chemistry, right)))
```

The bivariate population covariance uses MAD-derived marginal scales with a
2-bp floor, clips the empirical edge correlation to `[-0.9, 0.9]`, and uses
`Sigma_predictive = Sigma * (1 + 1/n)`. The
per-edge formula is replaced by `255` when that edge is unchanged.
For a changed edge with zero represented target-molecule opportunities, q1 or
q2 is instead `0`: the population can nominate an exploratory canonical edge,
but cannot establish that molecule's exact boundary. q0 retains the complete
population-plus-chemistry geometry posterior, and the report retains both the
model posterior and the materialized edge confidences.
Assignment against the explicit unmatched null and topology checks determine
which `H` alternatives are materialized; these probabilities expose the
aggressive alternatives for downstream thresholding rather than silently
removing them.

All three values are linear 0--255 quantities, not Phred scores, ordinary TF
`tq`, or nucleosome `nq`, and they are not yet held-out calibrated probabilities
of biological truth. `q1` always means molecular-left and `q2` molecular-right.
Because MA coordinates are in molecular orientation, the annotator swaps the
reference-left and reference-right confidence values on reverse alignments.

FiberBrowser must apply threshold `T` only to named `R` and `H` groups. When a
named group's `q0 >= T`, show the SR interval or atomic interval set; otherwise
show its ordinary baseline. The baseline is the containing ordinary MSP for an
`R` group and the exact same-class ordinary `tf` or `nuc` identified by the
source ordinal for an `H` group. Unnamed `(255,0,0)` shadow calls are already-
baseline sentinels and must never enter this threshold comparison. Thus one
connected SR slider can move named calls between SR and baseline hypotheses,
but `nuc_sr` and `tf_sr` are not complementary N/TF states and must not be
treated as the CR nucleosome-versus-reconstruction slider.

Ordinary `nuc`, `msp`, and `tf` groups remain in the output and the source BAM
is never edited. Stale `nuc_sr`/`tf_sr` groups are rebuilt rather than appended.
Each shadow interval consumes exactly three positional `AQ` bytes. `AN` retains
one positional token for every MA interval and uses `.` for unnamed baseline
calls, so later named `H`/`Rn` roles cannot shift out of alignment.
The v6 `@CO` declaration records the two groups, exact-family `q0` meaning, roles,
non-complementarity, fixed nucleosome identity/cardinality, and absence of a
nucleosome-length ceiling.

Report proposals are tied to the canonical input path, query name, reference
start, alignment flag, CIGAR, SHA-256 of the original SAM record, exact
molecular interval, annotation ordinal, and occurrence number among
byte-identical alignment records. Duplicate query names, duplicate ordinary
intervals, and byte-identical BAM records are therefore handled one-for-one.
Each `H` annotation name records its source ordinary-annotation ordinal. The
default `--minimum-posterior 0` preserves every rescue decision. For a v6
report, projection and topology diagnostics account for any proposed rescue or
accepted edge that cannot be materialized exactly.
All BAMs in a pooled cohort are staged and validated before any staged result
is published. A stale v2 header contract and any old SR groups are replaced
when v6 layers are written.

## Audit contract

`fiberhmm-strand-rescue-audit` validates:

- the v6 header and `MA-TYPES` declaration of both layers;
- exact MA/AQ/AN positional alignment and three-byte rows;
- `nuc_sr` cardinality equal to ordinary `nuc` cardinality;
- `tf_sr` cardinality equal to ordinary TFs plus all `Rn` components;
- fixed unnamed `(255,0,0)` rows for unchanged calls;
- singleton `H` roles with a modeled alternative-versus-baseline `q0`;
- an explicit, unique ordinary-annotation source ordinal for every `H` role;
- atomic, contiguous `Rn` groups confined to `tf_sr` and sharing one `q0`;
- `255` on every unchanged edge of a named `H` group;
- absence of newly introduced shadow-call overlaps; and
- quickcheck and BAM-index integrity.

The annotator retains compatibility with v2--v5 reports and rematerializes
legacy inputs under their compatible three-byte contract. The auditor can also
validate existing v2--v5 BAM contracts. Every newly generated report and BAM
uses v6.

## Validation and interpretation

The v6 paper benchmark uses five-fold molecule-disjoint mask-and-recover.
Training molecules learn each locus catalog and TF-class prior. Eligible
ordinary TF calls are then hidden only on held-out molecules, merged into their
ordinary MSP context, and recovered from the unchanged raw chemistry. Nested
opportunity thinning tests the expected loss of information. Same-family
ordinary accessible events provide a conservative baseline-negative
comparator; they are not false-positive truth because some may be biological
footprints missed by the baseline caller. An outcome-blind sensitivity analysis
matches hidden and comparator events without replacement on fold, family,
strand, opportunity stratum, and MSP length.

This experiment measures recovery of an ordinary FiberHMM call—a silver
label—not biological sensitivity, calibrated FDR, TF identity, or calibrated
posterior probability. Coordinate-displacement controls are descriptive
specificity tests only, because moving a catalog changes its recurrent source
support and local sequence opportunity. Cross-assay overlap is population
concordance rather than molecule-level truth.

The exact event, configuration, matching, bootstrap, audit, figure-source, and
artifact-receipt tables are documented in the
[two-locus benchmark bundle](../../paper/analysis/strand_consensus/two_locus_20260824/README.md).
The current development contract and claim boundary are in
[`STRAND_RESCUE_V6_DEVELOPMENT.md`](./STRAND_RESCUE_V6_DEVELOPMENT.md).

## Commands

```bash
fiberhmm-strand-rescue \
  -i timepoint1.bam -i timepoint2.bam \
  --preset dddb \
  --region chr3L:15039880-15040260 \
  --site 15039948-15040022 \
  --nuc-site 15039920-15040110 \
  -o ind.strand-rescue.json \
  --proposal-tsv ind.strand-rescue.tsv

fiberhmm-strand-rescue-annotate \
  --report ind.strand-rescue.json \
  --output-dir ind_sr_bams

fiberhmm-strand-rescue-audit \
  -i ind_sr_bams/timepoint1.strand-rescue.bam \
  -i ind_sr_bams/timepoint2.strand-rescue.bam \
  -o ind_sr_bams/audit.json
```

For Nanopore Hia5, `--preset hia5-nanopore` automatically applies the hard
threshold 248. The reproducible DddA, DddB, and Nanopore checks are recorded in
[`STRAND_RESCUE_VALIDATION.md`](./STRAND_RESCUE_VALIDATION.md).
