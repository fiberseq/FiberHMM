# Paired focal consensus recall plan

This is the implementation contract for the high-sensitivity consensus pass.
It is deliberately secondary to the ordinary FiberHMM `nuc`/`tf` callers:
ordinary calls remain unchanged, while consensus alternatives are written as
custom MA annotation types in regional derivative BAMs.

Genome-wide execution is out of scope until the focal implementation and its
calibration are satisfactory.  The command accepts indexed BAMs and a focal
region, so users do not have to make physical BAM subsets.  Repeated input BAMs
form one inference cohort only when explicitly supplied together; independent
assays remain independent inference runs and are used for validation unless a
future, explicit meta-analysis mode is added.

The consumer-side implementation contract, including parsing pseudocode and
acceptance tests, is in
[`FIBERBROWSER_HANDOFF.md`](./FIBERBROWSER_HANDOFF.md).

## Recall policy

### Composite recall (CR)

Every 90--220 bp nucleosome overlapping a source-supported focal
configuration is retained in the complete decision table.  The aggressive MA
writer emits both the original nucleosome and the best TF-complex alternative,
even when the nucleosome is the MAP state.  Exact TF components are used when
the decomposition is resolved; otherwise the original envelope is emitted as
one unresolved TF-complex interval.  Posterior/tier gates remain annotations,
not output gates.

### Strand rescue (SR)

Every source-supported target interval with local target evidence is retained
in an analogous decision table.  Both accessible/MSP and nucleosome targets
are scored, with the same hard 90--220 bp limit for current nucleosome targets
as CR. Longer protected blocks must first be resolved by the baseline nuc
recaller. A nucleosome target receives a paired N/TF alternative; an
accessible target receives a one-sided TF alternative because accessibility
is already represented by the ordinary `msp` layer.  Strong and review tiers
remain available as presets, but the aggressive writer is not limited to
those tiers.

## MA/AQ/AN contract

The custom types are `nuc_cr`, `tf_cr`, `nuc_sr`, and `tf_sr`.  Paired output
uses the spec-valid quality layout `QQQQQ` (five linearly scaled bytes):

| index | CR meaning | SR meaning |
|---:|---|---|
| `q0` | posterior of the represented N or aggregate TF-complex state | posterior of the represented current or aggregate TF state, conditional on current-vs-TF |
| `q1` | best exact configuration probability conditional on TF complex | best TF configuration probability conditional on TF |
| `q2` | equal-odds probability obtained from the prior-independent molecular log likelihood ratio | equal-odds probability obtained from the molecular log likelihood ratio versus the current state |
| `q3` | selected local TF-complex prior probability | source-population TF prior probability versus the current state |
| `q4` | equal-odds probability from the true-vs-best-shifted-boundary log Bayes factor; 128 when unavailable | minimum explicit source-support fraction across selected sites; 128 when unavailable |

For each N/TF pair, `q0_tf` is rounded first and `q0_nuc = 255 - q0_tf`, so
the stored bytes sum to exactly 255.  Auxiliary zeroes on carried baseline
calls mean "not evaluated by consensus", not zero biological support.

`AN` supplies stable, unique labels:

```text
fhcr_<decision_hash>_N
fhcr_<decision_hash>_T0
fhcr_<decision_hash>_T1
```

The shared prefix links a decision; suffixes keep its individual annotations
distinct.  Multi-TF alternatives share the same `q0` and switch atomically.
One-sided accessible/MSP strand-rescue alternatives use `AT0`, `AT1`, ...
suffixes so a missing or corrupt N member cannot be mistaken for an
intentional accessible-origin decision.
Empty `AN` fields, rather than a non-spec placeholder, are used for unnamed
annotations.

MA itself treats the N and TF records as ordinary overlapping custom
annotations.  Their alternative-state interpretation is a FiberHMM/
FiberBrowser convention declared in BAM header comments.

## FiberBrowser linked threshold

For slider byte threshold `T`, the browser renders:

```text
TF alternative: q0_tf >= T
N alternative:  q0_nuc >= 256 - T
```

Equivalently, it can group by the `AN` decision prefix and render TF when
`q0_tf >= T`, otherwise N.  Exactly one paired state is visible.  `T=128`
gives the MAP split, lower thresholds are increasingly aggressive, and higher
thresholds are increasingly nucleosome-conservative.  Unpaired baseline calls
remain fixed; one-sided SR additions obey the TF threshold without hiding an
ordinary MSP.

The regional JSON/TSV report remains authoritative for raw probabilities,
log likelihood ratios, opportunity/hit counts, source depth, shifted controls,
and all other diagnostics that should not be compressed into bytes.

## Validation before genome-wide work

1. Preserve the existing unit and workflow tests.
2. Add round-trip tests against the MA grammar for `QQQQQ`, positional `AQ`,
   empty `AN` fields, overlaps, and complementary `q0` values.
3. Re-run Homie, Nhomie, SF1, SF2, focal fly DddB, DddA, and threshold-248
   Nanopore examples.
4. Audit that every paired N/TF decision has one shared prefix, complementary
   `q0`, identical auxiliary qualities, and atomic multi-TF components.
5. Keep selected-state legacy output available until the paired browser path
   is validated.
