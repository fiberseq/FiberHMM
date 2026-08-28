# Hierarchical consensus-recaller validation

This directory implements the installed `fiberhmm-consensus-validate`
reporting pipeline for the focal consensus recaller. It does not modify BAMs.
Every likelihood is computed from standard aligned sequence, hard MM/ML calls,
and existing MA calls used only for candidate/diagnostic geometry. Raw IPD and
IDLI data are neither required nor read.

## Primary panels

`manifests/core_local.json` deliberately contains only the locally verified
primary comparisons.

| Cohort | Assay | Role | Local input |
|---|---|---|---|
| GM12878 | full PacBio Hia5 Fiber-seq | broad-nucleosome anchor; fine-TF anchor at lower geometric resolution | `/mnt/g/gm12878/GM12878.fiberhmm.bam` |
| GM12878 NAPA | targeted DddA DAF | fine-TF anchor | `ddda_nuc_output/napa_recaller_TF.bam` |
| GM12878 UBA1 | targeted DddA DAF | fine-TF anchor | `ddda_profile/uba1_ddda_recall.bam` |
| fly 2–4 hr | pooled PacBio Hia5 Fiber-seq | broad-nucleosome and fine-TF anchor | all five BAMs in `/mnt/g/v3seg_mp` |
| fly 2–4 hr | pooled targeted DddB DAF | fine-TF support and candidate nomination | `.../spacetime_updated/fp_update/yw_2-4_recalled.bam` |
| fly siGAF | Nanopore Hia5 | exploratory support only | both BAMs in `.../Fiber-seq/2-point_timecourse/bam` |

The low-depth GM12878 scDAF subset is not in the primary manifest. The two fly
Nanopore samples are perturbed siGAF data, use the requested hard `ML >= 248`
threshold, and have `truth_vote: false`.

## Assay hierarchy

Truth is not a flat vote and is split into independent axes.

- `broad_nucleosome`: PacBio Hia5 is the geometric anchor. DddB and Nanopore
  can support presence but do not set boundaries. DddA broad-nucleosome voting
  is disabled until its radial fingerprint likelihood is calibrated here.
- `fine_tf`: DddA is the highest-resolution anchor; PacBio is an independent
  high-specificity anchor with lower opportunity/resolution. DddB and Nanopore
  are support assays.
- `occupancy_frequency`: reserved in the hierarchy but not yet calibrated.
  Current results concern focal state existence and per-assay mixture weights,
  not quantitative cross-assay occupancy equivalence.

A support assay may nominate a focal interval for testing. Nomination does not
make it truth: a support-only candidate has geometry tier 99, and definitive
presence still requires an anchor's raw likelihood. One low-tier assay can
never define truth. Two independent support families can define only
provisional presence and never authoritative geometry.

## Probability model

For each focal TF interval and each informative molecule, the builder computes
chemistry-specific hard-call likelihoods for three local states:

1. accessible (`A`),
2. focal TF protection (`TF`), and
3. local nucleosome protection (`N`).

The population test compares the nested mixtures `A + N` and `A + TF + N`.
Mixture weights are fit from independent molecules, and the reported site
statistic is the log-likelihood gain minus the BIC penalty for the added TF
mixture parameter. Thus a reproducible 5% TF component can be strong at high
depth without assuming that a true site must occupy most molecules.

Stranded assays retain CT/GA or FWD/REV fits. Positive-only amplified panels
may contribute positive strand evidence without using a weak strand as a
negative veto. PacBio fibers are duplex: A and T opportunities are combined on
every molecule, and alignment orientation is never treated as a biochemical
strand.

Broad-nucleosome candidates use the corresponding nested no-N versus N
mixture. The separate composite pass still applies the hard 90–220 bp focal
block range and never breaks blocks merely because a population TF site
overlaps them.

## Amplified DAF independence

Targeted DAF reads are collapsed in memory by full-read hard-deamination
fingerprint at Jaccard 0.95, matching the logic of `fiberhmm-dedup`. One
maximally informative representative is retained per family; low-call reads
remain separate. The source BAM is never rewritten. Reports contain raw read,
fingerprintable read, inferred molecule, duplicate, and largest-family counts.

## Short Nanopore reads

A read does not have to span a locus or amplicon. It is informative for a TF
test when it has at least one opportunity in the focal interval and at least
one opportunity in either flank. Missing sequence contributes no likelihood
term. This is intentional for the very short fly Nanopore fibers.

## Local controls

Absolute protection Bayes factors are not sufficient in dense regulatory
regions. With `--controls-per-candidate N`, the builder chooses up to `N`
non-overlapping local shifts whose hard-call opportunity count most closely
matches the parent interval. Controls are kept in separate `controls` and
`control_records` arrays and never enter truth adjudication.

The production panel requests five controls per candidate, rejects controls
outside a fourfold source-opportunity match, and refuses overlapping controls
within a parent. Each control is compared with its siblings to form pseudo-site
null deltas. Reused genomic decoys are aggregated in 25-bp bins, then candidates
are calibrated against other loci of the same cohort/chemistry. Source-selected
assays are diagnostic only; BH q-values are assigned only to independent assay
evidence. Controls can still land on undiscovered biology, so every control is
retained rather than collapsed into an assumed biological negative label.

## Commands

The examples retain all scientific products under the project-relative,
Dropbox-backed `consensus_validation_outputs/` tree. Use a similarly persistent
location for other projects; disposable system temporary directories are not a
reproducible output target.

Audit paths, indexes, tags, and per-locus coverage:

```bash
fiberhmm-consensus-validate audit \
  --manifest consensus_recaller_collab/validation/manifests/core_local.json \
  -o consensus_validation_outputs/core_local/manifest_audit.json
```

Build molecule-aware evidence and five local controls for NAPA:

```bash
fiberhmm-consensus-validate build-evidence \
  --manifest consensus_recaller_collab/validation/manifests/core_local.json \
  --locus gm_napa --exclude-exploratory --controls-per-candidate 5 \
  -o consensus_validation_outputs/core_local/gm_napa_evidence.json
```

Construct leave-one-assay-family-out truth:

```bash
fiberhmm-consensus-validate adjudicate \
  --manifest consensus_recaller_collab/validation/manifests/core_local.json \
  --evidence consensus_validation_outputs/core_local/gm_napa_evidence.json \
  -o consensus_validation_outputs/core_local/gm_napa_truth.json
```

Create the same compact support/control/truth summary used by the batch report:

```bash
fiberhmm-consensus-validate summarize \
  --manifest consensus_recaller_collab/validation/manifests/core_local.json \
  --evidence consensus_validation_outputs/core_local/gm_napa_evidence.json \
  --truth consensus_validation_outputs/core_local/gm_napa_truth.json \
  -o consensus_validation_outputs/core_local/gm_napa_summary.json
```

Calibrate several independently built locus batches and emit deterministic
sample-level TSV rows:

```bash
fiberhmm-consensus-validate calibrate \
  --manifest consensus_recaller_collab/validation/manifests/core_local.json \
  --evidence consensus_validation_outputs/batches/gm_evidence.json \
  --evidence consensus_validation_outputs/batches/fly_batch_a.json \
  --evidence consensus_validation_outputs/batches/fly_batch_b.json \
  --tsv consensus_validation_outputs/core_local/consensus_proposals.tsv \
  -o consensus_validation_outputs/core_local/consensus_calibrated.json
```

Audit amplified-DAF family counts without rewriting the BAM:

```bash
fiberhmm-consensus-validate dedup-sensitivity \
  --manifest consensus_recaller_collab/validation/manifests/core_local.json \
  --locus gm_napa --sample gm12878_ddda_napa_targeted \
  -o consensus_validation_outputs/core_local/napa_dedup_sensitivity.json
```

The fallback beta-binomial settings in `default_hierarchy.json` remain for
legacy/synthetic records. Production proposal tiers use the direct nested
mixture statistic and `default_proposal_policy.json`.

## Production guardrails

- Calibrate against multiple matched shifts, held-out loci, and held-out
  biological replicates rather than choosing a threshold from one amplicon.
- Require explicit focality in at least one assay plus raw likelihood support;
  broad protection alone is not a TF candidate.
- Keep `N` in the source state space even when the strand pass is forbidden to
  rewrite nuc calls. Otherwise nucleosomes inflate the inferred TF prior.
- Keep support nomination, presence evidence, boundary authority, and
  per-molecule rewriting as separate decisions.
- Retain ambiguity/review output. Cross-assay disagreement is a validation
  category, not permission to force convergence.

The completed panel and operating-point results are documented in
[`../PRODUCTION_VALIDATION.md`](../PRODUCTION_VALIDATION.md).
