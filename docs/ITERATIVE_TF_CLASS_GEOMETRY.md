# Iterative TF-class geometry model

Status: experimental report-only v4 implementation, 2026-08-25. The
opportunity-lattice projection, fixed-class pooled generalized-EM geometry
pass, deterministic multistart fitting, structured residual nulls, exact
projection quotients, mutually exclusive family geometry substates, frozen
molecule-disjoint scoring, a global-efficiency sensitivity arm, and a prepared
float64 evidence backend are implemented.
Held-out merge/split selection and an outer catalog proposal pass are not
implemented, and the output is not yet a production result.

## Question being estimated

The latent object is a locus-specific TF configuration, not an observed HMM
edge. An atomic TF class is a half-open protected interval `B_k = [l_k, r_k)`.
A configuration is either accessible or a compatible non-overlapping subset of
atomic classes. A broad footprint and two adjacent small footprints are
therefore distinct configurations even when their envelopes are similar.

A locus family is a biological footprint hypothesis with one or more mutually
exclusive geometry substates. For example, NAPA has one localized family with
`narrow` and `broad` substates. The family posterior is the sum of the unique
configuration posteriors containing any of its substates. The substate
posteriors remain separate. Configurations containing two substates of the same
family are rejected, and indistinguishable substates are reported as an
opportunity-projection equivalence class rather than assigned arbitrary
individual weights. Seed-strand provenance is metadata, not a strand-specific
binding identity.

For molecule `i`, the assay exposes possible modification positions `O_i` and
hard outcomes at those positions. CT and GA DAF strands generally expose
different lattices. The projection of an interval onto molecule `i` is

```text
Pi_i([l,r)) = {p in O_i : l <= p < r}.
```

Two coordinates are exactly non-identifiable on that molecule when their
projections are equal. FiberHMM must report the resulting coordinate
equivalence set rather than manufacture base-pair resolution inside a gap in
`O_i`.

The scientific question is whether one latent configuration, after this
strand- and molecule-specific projection and calibrated assay detection, could
plausibly generate the observed evidence on both strands. It is not whether
the two ordinary callers emitted byte-identical reference edges.

## Likelihood

For a candidate configuration `C`, let `U(C)` be the union of its atomic
protected intervals. Existing calibrated `ReadEvidence.steps` are per-position
log likelihood ratios for protected versus accessible emission. Terms under
the all-accessible state cancel, giving

```text
log L_i(C) / L_i(A) = sum_{p in O_i intersect U(C)} step_i(p).
```

This automatically gives a missing C/G opportunity zero evidential weight.
The same formulation applies to DddA, DddB, and Hia5 after their own calibrated
emission models. PacBio Fiber-seq pools molecules without a CT/GA consolidation
step but uses the same opportunity projection for finite m6A sampling.

In the reference benchmark, emission efficiency is calibrated per molecule
from that molecule's baseline MSP calls before the population split. The full
candidate footprint plus boundary-search window is excluded from calibration,
preventing the tested TF hypothesis from helping estimate its own efficiency.
The factor scales both protected and accessible hit probabilities before new
LLR steps are computed. This does not share measurements across train and
holdout molecules, but it remains caller-conditioned because ordinary MSP
segmentation supplies the calibration mask. The benchmark therefore always
fits a full-depth sensitivity arm using frozen factor-1 steps computed from the
same emission model. On the corrected 2026-08-25 NAPA smoke test, both arms
selected the same broad and narrow geometries, while held-out mean
LLR/opportunity was 0.0713 with per-molecule calibration and 0.0286 with the
factor-1 comparator. The geometry was robust in this test, but the emission
calibration was predictively material and cannot be described as independent
raw chemistry.

Ordinary TF calls seed the candidate library and define a bounded search
region. They are deterministic products of the same chemistry and are not
multiplied into the likelihood as independent observations. An optional caller
projection kernel can constrain implausible assignments, but it must be learned
on disjoint molecules or controlled simulation and must operate on opportunity
signatures before physical edge residuals.

## Unit of assignment

The E-step assigns one complete configuration per collapsed molecule. It must
not independently assign each call, because that would:

- count a molecule more than once;
- confuse one broad footprint with two small footprints;
- allow mutually overlapping atomic classes on one molecule; and
- overstate evidence in amplified DAF data.

For molecule `i` and configuration `C`,

```text
r_iC proportional to pi_C * L_i(C),
```

where `pi_C` is the regularized population weight. Accepted ordinary calls can
restrict a molecule to configurations whose projected topology is compatible,
but small edge shifts that preserve the opportunity signature remain soft and
do not create separate hard labels.

The current report-only fixed-class fitter takes the more permissive option: assignment
among the complete seeded configuration library is fully soft under raw
chemistry. It includes two explicit residual components. `P0` is one coherent,
unanchored protected interval, marginalized over a wider locus window with a
uniform physical-width prior followed by a uniform start prior. Exact members
of the anchored candidate grids are excluded from `P0`, preventing a duplicate
TF component while retaining a look-elsewhere placement penalty. `U` is a
proper diffuse component in which represented opportunities are independently
protected with rate `rho`, integrated under a Beta(1,1) prior using
Gauss-Legendre quadrature sized to integrate the opportunity-count polynomial
exactly. The benchmark requests a default `P0` width prior of 1--80 bp so small
anchored footprints are representable by the null; every fit separately reports
the requested and realized envelope-limited width range and anchored widths
outside its realized support. Molecules carrying an ordinary topology
that the fixed library
cannot represent remain in the fit; their seed mismatch is counted and their
posterior responsibility for `P0` or `U` nominates residuals for future
new-model proposals. Neither residual component updates a TF boundary. A
future cross-fitted projection kernel may constrain compatible configurations,
but it must preserve
broad-versus-two-small ambiguity whenever the molecule's opportunity lattice
cannot distinguish those states.

## Iteration

1. Seed atomic intervals from bounded ordinary-call families on each physical
   strand. Retain all plausible small, broad, and composite alternatives.
2. Collapse exact duplicate states and mark pairs that are exactly
   opportunity-equivalent on every represented stratum.
3. Enumerate accessible and non-overlapping TF configurations within a bounded
   locus. Large loci are split before enumeration and the truncation is
   reported.
4. E-step: calculate molecule-configuration responsibilities from raw chemistry,
   current class weights, and any cross-fitted caller-projection kernel.
5. M-step for weights: update `pi_C` with a hierarchical pseudocount. Accessible,
   anchored-TF, spatial-residual, and diffuse-residual families receive equal
   prior mass; anchored-TF mass is divided across configurations so enlarging
   the configuration library does not silently enlarge its prior.
6. M-step for geometry: coordinate-ascent over a bounded integer edge grid.
   Maximize expected complete-data log likelihood while holding other atomic
   intervals fixed. All tied coordinates with identical projected evidence are
   retained as an equivalence set.
7. Propose one merge or split at a time. Refit responsibilities and geometry,
   then accept the move only under molecule-disjoint predictive validation and
   the topology safeguards below.
8. Stop when class identity, opportunity-equivalence sets, and weights are
   stable and the objective improvement is below tolerance. Enforce a maximum
   iteration count and run multiple deterministic initializations.
9. Freeze the fitted catalog. Only then score rescues and harmonized edges.
   Rescued or harmonized calls never teach the model that produced them.

This is generalized EM: each accepted coordinate or model move must not lower
the training objective, while model cardinality is selected using held-out
predictive performance rather than the training objective alone.

The implemented fixed-class pass records the penalized observed-data objective
after every weights-plus-geometry update and raises if it decreases beyond
floating-point tolerance. It exports pooled and per-strand expected
complete-data Q boundary surfaces, exact maximizing coordinate sets, and edge-level
identifiability labels. The local edge grid is configurable and is six bases
around each supplied seed in the current NAPA/UBA1 benchmark. Multiple
deterministic geometry and weight starts are fit, invalid topology starts are
reported and skipped, and the highest common penalized training objective is
selected without consulting held-out molecules.

## Merge and split safeguards

A merge is eligible only when candidate intervals are locally compatible and
one of the following holds:

- their opportunity projections are identical on every adequately covered
  represented stratum;
- one stratum is exactly blind to the difference and the informative stratum
  supports one shared class; or
- a shared model has adequate held-out predictive support over separate
  models.

No merge may collapse two classes that reproducibly co-occur as separate
non-overlapping calls on the same molecules. Compatibility is complete-linkage:
every member of a merged component must be compatible, preventing transitive
chain bridging.

A split requires a reproducible multimodal residual or configuration pattern,
minimum support in each child, and held-out predictive improvement after the
complexity penalty. A training-only likelihood gain is insufficient at
targeted depth because even negligible misspecification can become highly
significant.

## Identifiability labels

Every canonical class carries one of these boundary states:

- `resolved_both_strata`: both strands contain differential opportunities and
  favor the same equivalence set;
- `resolved_jointly_complementary_strata`: neither strand uniquely resolves
  the edge alone, but the intersection of their opportunity-lattice
  equivalence sets contains one coordinate;
- `resolved_informative_stratum`: one strand is exactly blind and the other
  determines the canonical equivalence set;
- `resolved_by_single_stratum_consistent_with_others`: one stratum uniquely
  resolves the edge and other informative strata do not contradict it;
- `resolved_by_pooled_complementary_evidence`: neither physical-strand stratum
  reaches the operational support threshold alone, but pooled evidence reaches
  the effective-opportunity, accumulated-Q-margin, and per-effective-molecule
  Q-margin gates;
- `resolved_by_pooled_diagnostic_evidence`: the analogous pooled result for
  diagnostic partitions, without implying physical cross-strand rescue;
- `resolved_jointly_across_diagnostic_strata`: diagnostic partitions jointly
  resolve an edge without implying physical cross-strand rescue;
- `identified_up_to_opportunity_projection`: the selected edge is unique only
  after quotienting coordinates that select identical opportunities in all
  analyzed molecules;
- `conflicting_strata`: informative strands favor incompatible classes;
- `stratum_absorbed_by_residual_component`: a well-covered stratum favors a
  non-equivalent edge through `P0` and must not be mislabeled blind. The gate
  uses the fraction `P0 / (site + P0)` of residual-aware effective molecule
  support, not an absolute expected-molecule count, so the diagnostic does not
  change merely because depth is multiplied. Each molecule's `P0` support is
  first localized by its posterior probability that the unanchored interval
  center falls inside that site's candidate envelope; a distant residual is
  therefore not credited to every modeled site;
- `undercovered`: mapping or opportunity support is inadequate; or
- `model_ambiguous`: more than one class/configuration retains substantial
  posterior mass.

The displayed canonical interval is a deterministic representative of an
equivalence set, not a claim that both base-pair edges were measured. BAM edge
quality remains zero on a changed edge when the target molecule has no changed
edge opportunity. These are edge-level `status` values; the enclosing site has
a separate `boundary_status` of `identified_on_candidate_grid`,
`identified_up_to_opportunity_projection`, or
`unidentified_no_eligible_molecules`.

Edge coverage is not raw read depth. For each start and end, the implementation
reports posterior-weighted opportunity count and the profiled expected
complete-data Q margin in nats to the best non-equivalent edge-opportunity
projection after maximizing over the opposite boundary. Operational evidence
must pass both an accumulated Q-margin gate and a minimum Q-margin effect size
per effective site molecule, in addition to the effective-opportunity gate.
This prevents one strong molecule from supplying all support while also
preventing arbitrarily large depth from certifying a vanishing effect. This is an EM
diagnostic, not an observed-data likelihood ratio or Bayes factor. Profiling is
essential:
evidence that resolves an end must not make a completely unobserved start look
informative. A boundary can be reproducible under repeated subsampling yet
remain `undercovered` when its selected coordinate is merely retained from the
seed inside an evidence tie.

## Prespecified stability benchmark

The targeted benchmark uses one prespecified stable-hash molecule split, fits
on 80% of fingerprintable collapsed DAF family representatives or independent
PacBio CCS molecules, and
freezes the catalog before scoring the remaining 20%. This is a fixed
evaluation holdout, not fivefold cross-validation. Nested training subsets are
repeated under stable within-stratum molecule orderings and combined by a
deterministic weighted-fair interleave that keeps each prefix close to the full
CT/GA or FWD/REV composition. The primary fit is deterministic
multistart; sensitivity fits vary the initial geometry, initial weights,
boundary radius, hierarchical prior strength, and `P0` width prior.

Each depth reports both molecule count and the number of usable
chemistry-specific opportunity observations. The stability-only threshold is
the first depth at which at least 80% of subsamples reproduce the full-cohort
held-out opportunity projection and median configuration-quotient total
variation is at most 0.1. The operational threshold additionally requires
every supplied site edge to pass the pooled effective-opportunity, accumulated
profiled-Q-margin, and per-effective-molecule Q-margin thresholds in at least
80% of subsamples. Thus an
unsupported catalog alternative can have a stability-only depth while having
no operational depth. The full cohort is an internal stability reference, not
biological truth. A threshold must remain qualified at every greater evaluated
depth; full-depth exact projection agreement is tautological, but full-depth
edge identifiability is still required for an operational threshold.

At least 16 repeated subsamples are required before an all-success 95% Wilson
lower bound can reach 0.8; production runs use 20. Three-repeat runs are
explicit development pilots and are structurally forbidden from reporting a
coverage threshold.

Direct catalog comparisons fix the analysis envelope, holdout partition, and
the union-catalog `P0` exclusion universe across every competing fit. This
prevents a catalog from gaining likelihood merely by changing which molecules,
opportunities, or residual-null placements are available. The four-family
hierarchical prior has total pseudo-molecule mass `4 * pseudocount`; the
benchmark exports that mass and its fraction of the weight denominator at each
depth.

For DAF validation, only retained amplification-family representatives with at
least ten fingerprint deaminations enter the train/holdout partition. Retained
reads that cannot be fingerprinted remain available to ordinary FiberHMM use
but are excluded from this predictive benchmark because physical PCR-family
independence cannot be verified for them.

DddA CT and GA are independent population molecules sharing reference
geometry. They are legitimate complementary physical-strand strata. PacBio
FWD/REV are alignment-orientation diagnostics only; they are never interpreted
as physical cross-strand rescue. Cross-chemistry comparisons are concordance
between independent assays conditional on the supplied catalog, not truth
labels.

The current v4 NAPA/UBA catalogs retain verified derivation-receipt hashes, but
the candidate seeds were viewed on the full DddA locus cohorts. Consequently,
their DddA predictive catalog comparisons remain exploratory even when the
fit/score partition is molecule-disjoint. A manuscript-grade discovery claim
requires proposal discovery inside a discovery fold followed by geometry and
cardinality selection on disjoint molecules. Independent Hia5 transfer is an
external concordance check, not a repair for an underpowered holdout.

## NAPA diagnostic

At `chr19:47,515,123-47,515,167`, the ordinary maps contain CT
`47,515,134-47,515,151` (253 molecules) and GA
`47,515,133-47,515,153` (106 molecules). The CT opportunity projection is
identical for the two intervals, so CT cannot distinguish their edges. GA has
differential opportunities and supplies the canonical representative. The
current pairwise initializer consolidates this pair and records the CT
non-identifiability explicitly.

The conditional v3 catalog pilot found that raw K=2 had a small but reproducible
held-out advantage over either K=1 seed. The v4 representation therefore keeps
one NAPA family with two mutually exclusive geometry substates instead of
averaging the edge variation into one interval. In the prepared-backend v4
post-audit smoke fit, the selected broad interval was
`47,515,135-47,515,152`, the narrow interval was
`47,515,136-47,515,149`, and their combined family probability was 0.1796
(0.0728 broad and 0.1068 narrow). The family and each substate were separately
identifiable under the fitted component-likelihood quotient in this cohort.
These are exploratory conditional-catalog results, not ground truth or a
coverage threshold.

For a `G`-interval uniform boundary grid, the initializer's shared-versus-
separate log Bayes factor is bounded above by `log(G)` but is unbounded against
sharing. It is therefore only a selection-conditioned seed tie-breaker. It is
not a class posterior or the authoritative merge criterion at targeted depth;
held-out predictive model comparison supplies that criterion in the planned
outer merge/split pass.

An experimental Gaussian molecule-edge random effect gave the wrong answer on
this example because it placed jitter in reference-coordinate space before
opportunity projection. It remains report-only diagnostic output and is not
used for matching.

## Prepared evidence and genome-wide execution

The CPU reference now prepares each interval grid once, reuses float64 prefix
sums for spatial-null marginalization, and reuses opportunity-lattice indices
for exact projection classes. A superseded pre-audit three-repeat NAPA K=2
pilot fell from 115.8 s to 27.7 s (about 4.2-fold). The more rigorous post-audit
one-repeat smoke took 57.6 s after adding candidate-excluded symmetric
calibration, placement-localized residual accounting, and the clean factor-1
sensitivity arm; those wall times are not direct performance comparators.

CUDA is not the default for one targeted locus. In the corrected fair
prepared-input comparison on 2,637 real NAPA molecules and 2,483 spatial-null
intervals, the resident RTX 5090 float64 kernel took a median 1.78 ms versus
83.36 ms for NumPy (46.8-fold), but one-time CUDA context startup took 293 ms.
At an eight-locus-equivalent batch (21,096 molecule rows), the resident speedup
increased to 84.9-fold (7.91 versus 671.12 ms). CPU/CUDA spatial-null
likelihoods agreed within `1.78e-15` nats. A separate 169-interval candidate
geometry reduction selected the same opportunity-projection class; its maximum
expected-Q error was below 1% of the fitter tie tolerance at both batch sizes.

The proposed genome-wide policy is therefore:

1. use the CPU reference for targeted loci and all final near-tie decisions;
2. scan or tile candidate windows first rather than enumerating families at
   every genomic base;
3. pack many prepared ragged opportunity lattices into resident GPU batches;
4. compute interval evidence and residual marginalization in deterministic
   float64 CUDA chunks; and
5. replay every boundary/cardinality decision whose objective margin is within
   a declared numerical guard band on CPU.

This benchmark covers the dominant prepared spatial-null kernel, not the full
EM pipeline. Genome-wide deployment still requires bounded-memory streaming,
candidate-window ownership rules, cross-window family reconciliation, and
end-to-end CPU/CUDA decision-equivalence tests.

## Validation required before biological claims

- exact controlled truth spanning edge shift, footprint width, class
  cardinality, opportunity density, efficiency, depth, and strand imbalance;
- broad-versus-two-small configuration truth and same-molecule co-occurrence
  controls;
- null classes separated by zero, one, and several differential opportunities;
- nested, stratum-proportion-preserving molecule subsampling plus explicit
  opportunity-count and profiled-Q-margin titration on a molecule-disjoint
  holdout;
- bootstrap stability of class count, exact opportunity-equivalence set,
  configuration quotient, and configuration weight;
- held-out predictive comparison for every accepted merge/split;
- NAPA and UBA replication with predeclared loci; and
- PacBio/Hia5 concordance reported as population agreement rather than
  molecule-level ground truth.

Low-coverage scDAF population fitting remains out of scope. scDAF uses strict
physical-duplex pairing, not this population-consensus model.

## Preservation of single-molecule signal

The fitted population catalog is a decoder, not an averaging operator. Every
molecule retains its possible sites, observed modifications, and original
FiberHMM calls unchanged. The model adds a soft configuration assignment and,
when justified, an optional projected call in the separate `tf_sr`/`nuc_sr`
layers. It never overwrites the source call layer.

A projected edge is eligible only when the molecule cannot distinguish the
raw and canonical coordinates on its own opportunity lattice, or when its own
chemistry plus the frozen population model supports the alternative. Genuine
per-molecule boundary or broad-versus-composite variation remains visible and
is assigned to a distinct or ambiguous configuration. The report must expose
the raw interval, projected interval, molecule likelihoods, posterior mass,
and coordinate-equivalence set together.

For targeted NAPA/UBA data, CT and GA reads are independent molecules drawn
from a shared locus-level configuration distribution. They are pooled to learn
that catalog but are never paired as copies of one physical molecule. Strict
joint emissions across strands apply only to explicitly identified duplex
families in the separate scDAF duplex workflow.
