# Comparing footprint classes across haplotypes or samples

Consensus discovers recurrent footprint geometries and estimates their mixture
fractions. Transfer measures a fixed catalogue in new observations. Neither
command currently implements a calibrated differential-binding test. A motif
or orthogonal measurement is needed to identify the bound factor.

## Separate support, observability and assignment

| Output | What it answers | What it does not establish |
|---|---|---|
| `supported` | Does adding the class improve held-out fit enough and pass the prevalence screen? | Calibrated false-discovery rate, or evidence of absence when false |
| `resolved`, `resolution_nats` | Is the model's expected accessible-versus-protected information over the class span above the threshold? | Edge precision, separation from another protected geometry, emission calibration or adequate precision for a contrast |
| BF-based `member` | Does this molecule's evidence favor the class relative to the fitted alternative mixture? | High absolute membership probability |
| `posterior` | What class probability does the fitted mixture assign this molecule? | An independently calibrated probability of a named TF being bound |
| `prevalence` | What class fraction does EM estimate among scored molecules? | A hard count of known members or an uncertainty-calibrated biological occupancy |

The existing resolution statistic is the median, over its spanning-molecule
set, of the sum of `KL(p_accessible || p_protected)` inside the class span.
This is information expected under accessibility, not under protection. It
uses model emission probabilities: a sample with poorly calibrated deamination
rates can pass the resolution threshold. Two geometries with exactly the same
observable targets can both pass it while remaining indistinguishable.

The `prevalence_lower_bound` field applies a Wilson formula to `n * prevalence`.
It is a heuristic used by the support screen. It does not propagate latent
assignment, fitted-mixture, geometry-discovery or emission uncertainty and
should not be used as a confidence interval for a haplotype difference.

## Choosing CT, GA or both

CT and GA observe different reference targets. A C-rich, G-poor interval can
be informative on CT and poorly informative on GA. Examine usable targets
inside the class, its edge boxes and flanks, after masking and target thinning.
Motif composition alone is insufficient.

`classes.tsv` now reports `median_span_opportunities`, left/right flank
opportunity counts, and expected information in both accessible/protected
directions on the actual scored denominator. These complement the existing
resolution statistic; see the [output definitions](../reference/consensus-outputs.md).
They do not measure how well neighboring protected geometries are separated.

For a CT-only comparison, both haplotypes must be adequately measurable on
CT. Comparing H1 on CT with H2 on GA confounds haplotype and chemistry channel.
Use sequence and observability to choose channels, without selecting the one
with the desired difference.

- Support on one strand with an uninformative opposite strand can justify a
  candidate comparison. Report that the other strand could not corroborate it.
- When both strands are informative, retain both estimates and investigate
  disagreement; it cannot be dismissed as missing information solely because
  one strand failed the support screen.
- Do not require target support on both haplotypes. A resolved but absent
  class on one haplotype is a legitimate comparison arm.
- Strand agreement is corroboration, not additional biological replication.
  CT and GA records are not automatically independent physical molecules.

A coverage cutoff should apply to each sample × haplotype × tested strand.
State the actual denominator, exclusions and missing strata. A minimum
number of eligible samples may be a useful prespecified design choice, but
such a threshold is not a FiberHMM calibration guarantee. Haplotype labels must refer
to corresponding alleles across samples or phase blocks.

## Keep the compared class definitions and observations comparable

A frozen catalogue keeps all classes, including unsupported ones that remain
competitors in the mixture. Keep the complete catalogue when scoring; report
an independently selected subset afterward.

Frozen transfer can map targets to different source channels, each with its
own learned spots and contracted edge boxes. Inspect `transfer.channel_map`
and `transfer_summary.tsv`'s `source_channel`. For paired comparisons, mapping
both haplotypes to the same source dataset is an explicit way to share the
source model within each chemistry and chemical strand:

```bash
fiberhmm-transfer --models frozen --datasets haplotypes.json --bed windows.bed \
    --dataset-map H1=discovery --dataset-map H2=discovery \
    --daf-snp-mask common_variants.bed --output transferred
```

Here `H1`, `H2` and `discovery` are dataset IDs, not BAM HP-tag values. The
source should be selected independently of the contrast. This command does
not establish that the selected source model is well calibrated for either
target. Matching class geometry does not equalize target emission rates,
opportunity lattices or the independently fitted alternative-mixture weights.

SNP masking must be explicit in BAM-based consensus and transfer; an earlier
`fiberhmm-call` does not permanently mask the BAM's sequence or MD tag. Use a
common reviewed union of relevant variant positions on both haplotypes.
The built-in DAF SNP screen detects recurrent conversion-like variants, not
every heterozygous variant. Other variants can create or remove assay targets.
Masking removes observations, never converts them into evidence of protection.
See [SNP masking](consensus.md#optional-daf-snp-masking).
Consensus's mask does not rebuild the BAM's existing HMM/MSP scaffold. If
that scaffold defines eligibility, also apply the reviewed mask in upstream
`fiberhmm-call` and regenerate it before consensus/transfer.

A point mask does not remove genuine allele differences in neighboring
sequence contexts, CpG substrate status or adjacent-target thinning. It also
does not make neighbors invariant to the masked site's conversion state:
after a masked C→T call is removed, a T remains in the query context, whereas
an unconverted C remains C. A masked C in a CC run can still affect which
neighbor survives thinning. Point masking alone therefore cannot guarantee
unbiased haplotype comparisons. For important hits, inspect those neighborhoods and test a common expanded mask
covering affected contexts and runs. Report the resulting loss of information;
there is no universal padding distance that resolves every sequence effect.
Expanded BED intervals are already accepted by `--daf-snp-mask`; no separate
padding option is needed. Choose them in genomic coordinates before any
minus-window transformation.

## Confident counts and fitted prevalence are different endpoints

For a class prior weight `pi`, away from numerical boundaries:

```text
posterior odds = BF × prior odds
P = BF × pi / (1 − pi + BF × pi)
```

For `pi=0.01`, BF=3 gives P≈0.029. Requiring P≥0.5 requires BF≥99 at that
prior, versus BF≥1 when `pi=0.5`. Combining BF≥3 and P≥0.5 therefore imposes
a prevalence-dependent assignment threshold. Channels fit their prevalences
separately, so this is not a common detection threshold across haplotypes.
Low-posterior BF-positive molecules are possible without a numerical bug.

Hard-threshold counts depend on sensitivity and false-positive assignment as
well as the biological fraction. The direction of bias need not be downward.
Lower deamination or fewer usable targets can change them even with equal
true fractions. A per-sample random intercept in a mixed model does not automatically
correct allele-specific detection bias.

Use fitted `prevalence` as the model-based point estimate, but do not simply
replace counts with `sum(posterior)` and treat those fractional counts as
ordinary binomial successes. The posteriors share fitted parameters. A
molecule-level bootstrap that refits EM can assess conditional uncertainty
for a frozen catalogue, subject to the correct independent resampling unit.
It does not automatically include discovery uncertainty or fix misspecified
emissions. A full differential model and its error calibration require
separate validation.

The `edge` and `loose` tiers are label-gated sensitivity summaries, not extra
independent observations. Within a fit, core ≤ edge ≤ loose. Across fits,
changing labels can change which molecules receive the extra tier mass, so
tiers are not necessarily monotone in evidence strength. Do not pool their
counts or use them interchangeably as occupancy estimates.

## Linker footprints and long accessible regions

Requiring both chemical strands is not an MSP-length filter. A short linker
can contain real protection but have few informative flanking sites. A
long-accessible-region context is also not a guarantee of a real TF footprint.
Inspect boundaries and competing broader/other-shape explanations directly.

If the scientific question concerns binding within a specified accessible
context, define that context explicitly and apply the same rule to numerator
and denominator. `input.minimum_nfr_length` restricts native nomination/replay;
it does not, by itself, restrict transfer's scored-molecule denominator to
molecules with a minimum MSP length. Distinguish a fraction among all eligible spanning molecules
from a fraction conditional on a chosen accessibility context.

## Validation before interpreting differential binding

1. Check source-channel mapping, effective opportunities, geometry ambiguity,
   deamination calibration, variant handling and independent molecule counts.
2. Use equal-occupancy simulations on the actual haplotype lattices, preserving
   sequence differences, depth and chemistry. Include broader protection,
   heterogeneous per-molecule efficiency and overlapping alternatives.
3. Inject edge/core variants and compare unmasked, common point-mask and
   context/run sensitivity-mask results.
4. Compare fitted fractions with thresholded counts across efficiency and
   opportunity strata. Repeating a direction across samples does not rule out
   a shared allele-linked measurement bias.
5. Use within-sample label permutations only as an exchangeability check of
   the testing machinery. They mix genotype-specific lattices and cannot
   validate away sequence-dependent allele detection bias.
6. Prespecify the eligible class set and handle multiple testing. Same-data
   catalogue discovery and selection can require cross-fitting or validation
   on held-out molecules/independent samples.

The checked-in [validation script](https://github.com/fiberseq/FiberHMM/blob/main/scripts/validation/validate_consensus_comparisons.py)
exercises a fixed-geometry synthetic setting; its results are not a calibration
of any particular study's statistical model or an end-to-end genome-wide
false-discovery rate.
