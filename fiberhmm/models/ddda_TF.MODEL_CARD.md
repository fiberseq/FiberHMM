# DddA TF likelihood model

`ddda_TF.json` supplies the context-specific observation probabilities used by
the DddA protein-sized-footprint likelihood-ratio scan. It is not the
first-pass nucleosome HMM.

## Scoring role

- `fiberhmm.inference.tf_recaller.build_llr_tables()` consumes only the two
  rows of `emissionprob` after state normalization.
- Each normalized row has 8,194 categorical entries: 4,096 context-indexed
  event codes, two sentinel positions, and 4,096 context-indexed no-event
  codes. The paired event/no-event entries recover one conditional Bernoulli
  probability for each of the 4,096 six-base contexts and states.
- For each callable non-CpG cytosine opportunity, the scan accumulates either
  `log P(event|protected) - log P(event|accessible)` or
  `log P(no event|protected) - log P(no event|accessible)`.
- DddA TF recall uses the interval cost shared by every preset,
  `--min-llr 5.0` nats, with at least three informative opportunities (the
  earlier DddA operating point was LLR >= 7). BAM quality is
  `TQ = min(255, round(10 * LLR))`.
- `TQ` is evidence under this exact model, not a posterior probability or a
  calibrated false-discovery rate.

## Container-only fields

The shared FiberHMM JSON loader requires `startprob` and `transmat`. State
identity is derived from the normalized emission rows; the complete container
is then permuted consistently into accessible/protected order. Thus,
`startprob` and `transmat` follow the emission-derived order but do not
determine it. They do not enter TF LLR accumulation and impose no transition or
footprint-length prior on the local scan. First-pass DddA segmentation instead uses the separate
`ddda_nuc.json` model. The chemistry-specific phase-aware nucleosome refiner also
uses its separately frozen likelihood inputs; changing `ddda_TF.json` does not
change the supported `--enzyme ddda` nucleosome path.

## Calibration and provenance

The emission table was fit from untouched mates of experimentally paired CT/GA
scDAF duplexes in chr1:1--40,000,000 from each of twelve libraries. This
prespecified bounded interval made the calibration tractable; every reported
denominator refers to it. Query-only rules nominated strict
accessible or protected intervals; observations on the physical mate were
then used for context-specific fitting. Model selection and the LLR operating
point used leave-one-library-out transfer and full-pipeline geometry checks.
The first-pass HMM was held fixed.

- Model SHA-256:
  `2b23b0905189d0638b682845ea5adae32144cf10e7a5e90b8f0bac199dbbbffe`
- Promotion receipt:
  `paper/analysis/benchmark/results/ddda_duplex_model_promotion_20260901/receipt.json`
- Efficiency-selection audit:
  `paper/analysis/benchmark/results/ddda_duplex_efficiency_selection_exact12_20260901_v1/`

The fitted probabilities are conditional observation-model parameters for this
caller. They should not be interpreted as assay-wide biochemical conversion
fractions. Each serialized categorical row sums to one, but
`P(event|accessible, context)` and `P(event|protected, context)` are separately
estimated state-conditional probabilities and are not complements. Scores from another emission table are not numerically
interchangeable without recalibration.
