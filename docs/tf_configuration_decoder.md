# Native TF configuration decoder

The TF recaller now uses `multi_interval_v1`. Its observations, context-specific
emissions, opportunity definition, and output MA/AQ layout are unchanged. The
change is selection of a complete footprint configuration instead of one peak
per positive-score excursion. This is native TF inference, not SR, CR, or XCR.

## Why the old scan could miss strong footprints

The single-excursion scan remembers one maximum until its running sum falls to
zero. If footprint A builds 15.66 nats, an accessible gap reduces the running
sum to 2.71, and footprint B supplies another 9.31 nats, the second peak reaches
only 12.02. A is emitted and B is omitted, although B alone exceeds the native
DddA score setting of 7. This example was reproduced from an actual NAPA GA BAM
record, not only a synthetic construction. The same failure occurs in Hia5.

## Objective and algorithm

On each read's native target lattice, let `ell[j]` be its protected-versus-
accessible emission log likelihood ratio. For disjoint intervals C, maximize

```text
sum over I in C of sum(ell[j] inside I) − lambda * number of intervals
```

The all-accessible configuration has objective zero. Every interval must contain
at least `min_opps` informative targets and begin and end on positive evidence.
Neutral bases contribute neither evidence nor opportunity count. The current
`lambda` is the existing `min_llr` setting: 5.0 by default for every preset
(Hia5, DddB and DddA).
There is no learned occupancy/family-frequency prior and no fixed count cap.

This is a MAP-style segmentation with a constant cost per protected interval.
LLRs are the likelihood representation, not an alternative to probabilistic
segmentation. A prior proportional to `exp(-lambda * number_of_intervals)` on
this fixed configuration domain gives the stated objective; its normalizer is
constant across configurations for this decode. No posterior or partition
function is reported by this routine.

For prefix log likelihood S and optimal prefix score F, an interval ending at t
has candidate score `S[t] - lambda + max_s(F[s] - S[s])`, with admissible starts
`s <= t - min_opps`. A running maximum and backtrace solve the objective in
linear time and memory. The empty configuration is allowed. Numerical ties
prefer fewer intervals, then less protected span; an interval whose LLR is
exactly the cost is not forced into the output.

A weak internal modification can remain inside one footprint. Splitting requires
enough negative evidence in the intervening gap to justify another interval,
and enough evidence in both resulting parts. There is no fixed-bp splitting
rule, smoothing, emission weakening, or removal of observed modifications.

## Scores, compatibility, and scope

- Reported `TFCall.llr` remains the sum of native emission evidence inside the
  interval, without subtracting the interval cost. MA/AQ score encoding and edge
  ambiguity calculation are unchanged.
- `min_llr` is an inference setting, not a display slider or calibrated FDR.
  Changing it can change segmentation. Changing a downstream display threshold
  must not rerun this inference or silently alter intervals.
- `call_tfs_in_interval(..., decoder="single_excursion")` retains the old scan
  for controlled comparisons. The default is `"multi_interval"`.
- Nucleosome refinement explicitly imports `call_single_excursion_intervals`.
  Its split/edge scanner remains unchanged. Existing pipeline rules that use TF
  results for promotion/demotion are not redesigned here; a full pipeline rerun
  is distinct from a diagnostic with frozen nucleosome calls.
- `fiberhmm-call` and the second-pass caller record `tf_decoder=multi_interval_v1`
  in program descriptions. This source change does not update existing BAMs or
  restart a running FiberBrowser server.

The retained native numerical settings are starting operating points, not a
transfer of old calibration to a new search procedure. Validation includes an
independent exhaustive configuration oracle, reversal tests, the actual NAPA
misses, and paired model-generated accessible-null trials. Those nulls are not
naked-DNA measurements or empirical FDR calibration.
