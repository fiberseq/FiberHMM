# Consensus comparison audit, 2026-11-11

This audit examines fixed-catalogue comparisons between haplotypes, motivated
by concerns about strand support, deamination efficiency, BF/posterior count
filters and conversion-like variants. It tests implementation behavior and
controlled measurement biases. No user's BAMs, motif instances or GLMM design
were available, so it does not validate a biological hit or its significance.

## Conclusions for comparative analyses

- Both-strand support is too restrictive as a universal eligibility rule.
  CT-only support can be usable when the same strand is adequately observable
  on both haplotypes. An uninformative GA channel is different from an
  informative GA channel that disagrees. C-rich sequence makes asymmetry
  plausible; usable targets and their geometry must establish it locally.
- `resolved` measures expected interior accessible-versus-protected
  information from the supplied model. It does not establish precise edges,
  separation from competing protected shapes, or emission calibration.
  A class absent on one haplotype can still be resolved there; do not require
  support on both comparison arms.
- Fifty scored molecules and four eligible samples out of six are study
  design choices, not validated power/error thresholds. Apply depth rules to
  each sample × haplotype × tested strand, specify the denominator, and report
  missing strata. Shared directions across samples can reflect shared
  sequence-linked detection bias. Align haplotype orientation across samples.
- A BF-positive, low-posterior molecule is not intrinsically a numerical
  failure. At prior 0.01, BF 3 gives posterior 0.029; adding posterior ≥0.5
  effectively requires BF ≥99. Hard counts therefore use different detection
  thresholds when channel prevalence or assay sensitivity differs.
- EM fractions are model-based point estimates. Posterior sums are not known
  binomial successes, and membership draws at fitted parameters do not supply
  full inferential uncertainty. A calibrated differential model remains work
  beyond this patch. Permuting haplotype labels cannot validate allele-linked
  sensitivity because it mixes the two sequence/opportunity distributions.
- Both-strand support is not an MSP150 context filter. `minimum_nfr_length`
  affects native nomination/replay, not the complete transfer denominator.
- `fiberhmm-call` can screen conversion-like SNPs automatically when its depth
  preflight allows it; consensus itself does not call SNPs. Earlier masks did
  not survive BAM reconstruction. The new explicit option reapplies the BED.
  Use a reviewed common union, including external variant information where
  available; the built-in screen is not a complete heterozygous genotype set.
- Edge/core variants can alter usable targets, molecule edge ranges, tiers
  and counts. A point mask removes direct observations, but leaves possible
  neighbor context and run-thinning effects. Use expanded common masks as a
  sensitivity analysis, covering affected contexts and runs. A shared frozen
  catalogue also needs matched source-channel boxes and spots within the
  compared chemistry/strand; inspect the transfer mapping.

The user-facing [comparison guide](../../docs/workflows/consensus-comparisons.md)
develops these distinctions and gives a transfer command.

## Implementation changes and checks

The branch starts from FiberHMM 3.0.4 (`e8589f4`). Commit `d3b947e` adds
explicit `--daf-snp-mask` preparation to consensus and BAM transfer, including
content fingerprints, parallel/restart handling and refusal to remask saved
evidence. Targets must provide their own mask; source-coordinate masks are
not inherited from frozen catalogues.

Subsequent changes:

1. Compute BF labels directly from log likelihoods, normalizing the floored
   alternative mixture. The old clipped-odds calculation can turn likelihood
   evidence of +20 or −50 nats into an abstention near mixture boundaries.
   Prefix/suffix log sums avoid cancellation. EM estimates are unchanged.
   `label_semantics=likelihood_ratio_v2` records the change; label-gated tiers,
   molecule edges and downstream labels can change.
2. Remove masked duplex conversion calls before context reconstruction and
   target thinning, matching the single-strand path. Invalidate cached MD
   strand selection when new exclusions apply.
3. Emit span/flank opportunity medians and expected information in both KL
   directions on the actual scored denominator. These are diagnostics, with
   no new eligibility threshold or change to the old resolution statistic.
4. Document support versus observability, posterior-dependent thresholds,
   alternative-mixture/source-model differences, tier interpretation and
   point-mask limitations.

Tests cover CT/GA, R/Y/MD/MM, reverse alignments, duplex reads, CIGAR offsets,
context sizes, run policies, native replay, unchanged BAM bytes, source-mask
non-inheritance, minus-oriented transfer, parallel windows and restart masks.
Numerical properties cover interior odds equivalence, extreme log likelihoods,
zero/one weights, row-offset invariance and neutral evidence. An injected
likelihood-boundary integration test verifies actual written TSV labels,
tiers, edge ranges, snapshot calls, diagnostics and frozen transfer.
Separate cases demonstrate geometry ambiguity despite high resolution,
observable absence, and variant-dependent edge/core behavior.

The existing batch code hashes every consensus Python module into each unit
digest. `--continue` recomputes units after the label implementation changes;
it does not silently combine old and new semantics.

## Reproducible sensitivity experiment

```bash
PYTHONPATH=. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python scripts/validation/validate_consensus_comparisons.py \
  --replicates 500 --molecules 200 --workers 2 --seed 20261111 \
  --output sensitivity-500.json

PYTHONPATH=. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python scripts/validation/validate_consensus_comparisons.py \
  --replicates 40 --molecules 200 --workers 1 --seed 20261111 \
  --unknown-accessible-fraction .95 --output sensitivity-f095.json
```

The main experiment has 12 scenarios × 500 replicates × 2 haplotypes × 200
molecules: **2.4 million simulated molecule observations**. Each scenario
reuses paired seeds for comparison; these are not 12 independent validations.
The JSON retains replicate outputs, seed, code/script hashes and semantics.

This is one fixed 32-bp class, independent Bernoulli sites, no discovery,
learned spots or edge contraction. The baseline uses accessible rate 0.85,
protected rate 0.02 and scoring `f=1`. This deliberately favorable value is
outside production's unknown-fraction upper clamp; the second run checks
`f=.95`. Broader-background and heterogeneous-efficiency rows are stress
tests, not claims of complete model specification. The injected variant is
an always-marked edge target; the site-level simulation does not model k-mer
or adjacent-run effects (the separate encoder tests address those).

Entries are mean H1/H2 fractions. "Confident" means direct BF ≥3 **and**
posterior ≥0.5. Truth is the realized simulated fraction, not just the input
probability. H2 resolution uses the supplied scoring model.

| Scenario | Truth | EM fraction | Confident fraction | H2 resolution (nats) |
|---|---:|---:|---:|---:|
| Equal dense targets | .350/.350 | .350/.350 | .350/.350 | 23.24 |
| H2 efficiency .45, correct rates supplied | .350/.350 | .350/.344 | .350/.301 | 8.67 |
| H2 efficiency .45, scored as .85 | .350/.350 | .350/.112 | .350/.104 | 23.24 |
| H2 two interior targets | .350/.350 | .350/.350 | .350/.351 | 5.81 |
| H2 one interior target | .350/.350 | .350/.352 | .350/.441 | 2.91 |
| H2 unmasked edge variant | .350/.350 | .350/.255 | .350/.284 | 23.24 |
| Same variant, shared site mask | .350/.350 | .350/.350 | .351/.351 | 20.34 |
| True fraction difference | .150/.548 | .150/.548 | .150/.548 | 23.24 |
| Class absent on H1 | .000/.350 | .000/.350 | .000/.350 | 23.24 |
| Broader background; H2 lower efficiency | .350/.350 | .334/.278 | .334/.292 | 8.67 |
| H2 heterogeneous efficiency, mean rates supplied | .350/.350 | .350/.332 | .350/.332 | 23.24 |
| H2 heterogeneous efficiency, molecule rates supplied | .350/.350 | .350/.350 | .350/.349 | 26.58 |

The principal findings persist with `f=.95`: calibrated lower-efficiency
H2 has EM .345 but confident fraction .300; miscalibrated H2 has EM .131;
the edge-variant H2 has EM .273 versus .347 after a shared mask. Broader
protection still biases the comparison (EM .331/.280). Thus EM is not a
general correction for misspecification or geometry confusion.

The script records a naive Fisher count test for exploration. That is not
the user's GLMM, and no calibrated false-positive/FDR claim follows from
these results. Real-lattice nulls, competing geometry, clustered molecules,
phasing, uncertainty propagation and selection/multiple-testing calibration
are still required for a biological analysis.

## Independent advice and decisions

AGY and Claude were queried independently with code and the proposed changes,
then shown reproductions, simulation results and the final patch. Advice was
checked against code and tests, not accepted by vote.

Both ultimately endorsed direct BF arithmetic, consistent production
propagation, explicit observability diagnostics and the distinction between
support, posterior probability and differential inference. Initial suggestions
to treat low-posterior BF members as bugs or to rely on label permutations
were corrected after inspecting counterexamples.

One disagreement remained about point masks. AGY accepted the documented
scope limitation after the encoder tests. Claude preferred restoring a
reference target base for neighbor contexts/runs. That would impose a C/G
context on a true T/A allele, changing the substrate model. We retained the
existing point-mask semantics, narrowed the direct-observation claim and
documented the conversion-state dependence. The tests demonstrate a common
expanded mask removing it in a short-run example. This is a deliberate
limitation, not a claim that point masks make allele comparisons unbiased.

Deferred: haplotype-aware context restoration, automatic context/run padding,
per-dataset excluded-observation counts, automatic mapping/mask warnings,
and a calibrated joint differential model. Existing source-channel mapping
and mask fingerprints remain inspectable. No scientific threshold or tier
definition was changed.

## Validation record

- Baseline at `d3b947e`: 3,641 passed, 23 skipped, 26 benchmark tests deselected.
- Focused consensus/transfer regression suite: 78 passed, 1 skipped.
- Expanded numerical/mask/integration checks: 51 passed.
- Full regression run after implementation: 3,667 passed, 23 skipped,
  26 benchmark tests deselected (916 seconds). Seven cases added during final
  review after that run collected its tests are included in the separate
  51-passing expanded run above.
- CLI-reference regeneration check and strict MkDocs build passed.

Large replicate files, reviewer transcripts and test logs are retained outside
the repository in the working session's `consensus_review_20261111` directory.
The commands above reproduce the simulations without those local artifacts.
