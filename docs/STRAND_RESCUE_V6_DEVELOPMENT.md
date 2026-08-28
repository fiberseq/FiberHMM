# Strand rescue v6 development contract

Status date: 2026-08-24

Strand rescue v6 is an optional, exploratory pass over completed ordinary
FiberHMM calls. It does not replace the HMM, TF recaller, nucleosome recaller,
or ordinary `nuc`, `msp`, and `tf` annotations.

## Intended locus model

Targeted DAF-seq commonly provides thousands to tens of thousands of molecules
over a 10--20 kb locus. The intended product is therefore an exhaustive locus
map rather than a top-N site list. Automatic TF and nucleosome family caps are
unlimited by default; an explicitly requested cap and whether it bound are
recorded in the report.

For DAF, CT and GA observations retain separate geometry, opportunity, and
detection summaries. Automatic discovery builds the two family maps separately,
then a bounded one-to-one cross-strand consolidation pass nominates a shared
latent footprint family when both maps support compatible start/end classes or
when one map is absent for an opportunity-limited reason. A one-sided family is
not automatically an artifact: sparse reference C or G can make the other
strand unable to resolve the same footprint regardless of sequencing depth.

For PacBio Fiber-seq, the locus family model is learned from one pooled
population. Alignment orientation is not a biochemical strand and no CT/GA-like
consolidation is performed.

Paired-duplex scDAF consensus is a separate operation. It combines evidence
from inferred copies of the same physical duplex and can support a molecule
when one copy has zero local opportunities. A population prior from independent
molecules cannot establish molecule-specific occupancy with zero local
evidence.

## Candidate classes

Cross-strand family candidates should be reported as:

- matched on both strands;
- one-sided and opportunity-limited on the other strand;
- one-sided with one or more represented interior opportunities on the other
  strand (adequacy is not asserted until calibrated); or
- under-covered/indeterminate.

Opportunity summaries include complete mapped coverage, informative bases
inside the footprint, informative bases in each boundary band, molecule-level
efficiency, accepted-call rate, and weak positive chemistry evidence. These are
detection quantities, not occupancy estimates.
These candidate classes are reporting-only in v6: they do not yet loosen the
positive target-evidence gate or authorize a molecule-specific rescue.

## Cross-fitting

The paper validation must learn the locus catalog, geometry, and population
prior on training molecules and score held-out molecules against the frozen
map. Rotating folds is practical for targeted DAF and prevents a target call
from teaching its own geometry or prior. Low-coverage genome-wide datasets
should fail closed to ordinary FiberHMM calls instead of forcing population
refinement.

## BAM quality contract

The optional output contains complete `nuc_sr.QQQ` and `tf_sr.QQQ` shadow
layers while preserving ordinary calls. Named low-confidence alternatives are
intentional and can be thresholded by FiberBrowser.

- `q0` is role-specific. For a rescue it is the selected configuration's mass
  within the accessible state plus every supported TF configuration, so
  competing footprint families reduce `q0`. If the local action set is
  truncated by the computational safety cap, rescue `q0` is zero and the
  report preserves the conditional subset score and dropped family IDs. For an
  existing-TF harmonization it is instead the assignment-marginalized
  probability of the canonical edge geometry rather than the source geometry.
- `q1`/`q2` are role-specific boundary scores. For a rescue they are the
  learned canonical molecular-left/right boundary reliabilities of that class
  component. For an existing-call harmonization they are molecule-specific
  molecular-left/right canonical-versus-source edge confidences.

The bytes are linear unit-interval scores, not Phred scores. Reverse alignments
swap reference-edge scores so `q1` and `q2` always retain molecular orientation.
Opportunity class, prior source, fold, and full probability decomposition stay
in report/sidecar provenance keyed to the named alternative; the three `AQ`
bytes are not overloaded with categorical metadata.

Within a named harmonization, an unchanged edge is encoded as 255 because the
source and canonical hypotheses agree. A changed edge with no informative
opportunity on the target molecule is encoded as zero even when the population
prior makes `q0` high. This deliberately preserves a prior-supported proposal
for exploratory viewing while preventing an edge-aware filter from treating
the molecule-specific boundary as measured. An untouched baseline shadow row
uses the distinct `(255,0,0)` sentinel.

The zero-opportunity materialization rule is specific to changed `H` edges.
Rescue (`R`) boundary bytes summarize the learned class geometry and do not
claim that the target molecule measured each boundary separately. Consequently
an edge-aware browser filter is a conservative mixed-role view, not one common
calibrated boundary-accuracy threshold.

Edge-family population evidence uses one regularized plug-in predictive score, not
a likelihood product over all source molecules. Confidence therefore converges
with depth rather than saturating merely because the locus has more reads.

## Validation hierarchy

1. Synthetic and controlled edit truth for exact event recovery and boundary
   error.
2. Molecule-disjoint mask-and-recover of held-out ordinary TF calls, with
   same-family baseline-negative comparators and outcome-blind opportunity/MSP
   matching.
3. Source-depth titration and split-half/cross-fold stability. Coordinate
   displacement is retained only as a descriptive specificity stress test: it
   does not preserve the source prior or local C/G opportunity and is not an
   FDR null.
4. Held-out DAF molecule recovery under nested target-opportunity thinning,
   stratified by CT/GA opportunities.
5. Paired-duplex masking where an inferred scDAF mate is available.
6. PacBio/Fiber-seq and orthogonal assays as population concordance, not
   molecule-level truth.

The historical v4 production results in `STRAND_RESCUE_VALIDATION.md` remain a
software/materialization validation snapshot. Their depth-accumulating edge
Bayes factors and pairwise rescue `q0` are not the v6 scientific contract.

## 2026-08-24 development evidence

The inspectable NAPA DddA bundle is in
`paper/analysis/strand_consensus/napa_dev_20260824/`. The full-cohort run has
3,202 records, 2,886 collapsed population representatives, 56 site families,
46 CT/GA class models, 1,226 locally supported rescue decisions, and 1,627
bounded existing-TF edge proposals. The report and target molecules overlap,
so this run is an implementation diagnostic rather than a benchmark.

A five-fold molecule-disjoint run learned every site catalog from training
molecules and produced 1,422 held-out decisions. Relative to the full-cohort
run only as a stability reference, 1,032 decisions matched; median
configuration IoU was 1.0, 94.7% had IoU at least 0.5, and median absolute
`q0` difference was 0.0044. This does not calibrate biological accuracy.

The primary molecule-local benchmark assigns folds before hiding eligible
ordinary TF calls on held-out molecules. It merges each hidden call into its
ordinary MSP context, retains the actual locus and raw chemistry, and asks the
frozen training-only model to recover it. A same-family ordinary accessible
event is a baseline-negative comparator, not false-positive truth: it may be a
real footprint missed by the baseline caller. At `q0 >= 128`, full-information
recovery was 61.4% for 3,524 hidden NAPA calls versus 5.4% for 18,717
comparators, and 58.6% for 6,681 hidden UBA1 calls versus 12.7% for 13,816
comparators. Outcome-blind one-to-one matching on fold, family, strand,
opportunity stratum, and a twofold MSP-length caliper retained 2,655 NAPA and
3,171 UBA1 pairs; the corresponding nomination-rate ratios were 12.93 and
5.03. Nested opportunity thinning reduced NAPA hidden-call recovery from 61.4%
to 40.2% and UBA1 recovery from 58.6% to 50.6% at 25% retention. These are
silver-label recovery measurements, not biological sensitivity, FDR, TF
identity, or probability calibration.

Controlled-truth class simulations favor the joint configuration model over a
factorized atomic-family ablation in every chemistry/detection stratum tested.
Deleting one discriminating opportunity raises class entropy and lowers the
top-class probability, as required for opportunity-aware ambiguity.

An adversarial edge simulation also demonstrates why `q1`/`q2` cannot merely
copy the population-supported posterior. With a canonical source population,
half of target molecules retaining their source edge, a 3-bp shift, and ten
source molecules, the false-positive rate at probability at least 0.9 is 97.7%
when the target has no changed-edge opportunities. Four opportunities reduce
it to 29.3% but do not calibrate it. The final zero-opportunity edge-byte rule
is a direct response to this failure mode.

The remaining scientific work is paired-duplex masking,
orthogonal-occupancy/biological-negative evaluation, and cross-chemistry
population concordance. These are required before interpreting the scores as
calibrated biological truth.
