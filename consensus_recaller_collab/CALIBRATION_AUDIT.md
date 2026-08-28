# Calibration audit and fixes — 2026-07-14

This supersedes the operating points frozen in
[`PRODUCTION_VALIDATION.md`](./PRODUCTION_VALIDATION.md). Three defects made
both passes systematically over-call, and the existing control panel was
structurally unable to detect any of them. All three are fixed; this document
records what was wrong, what the fix cost in calls, and what is still open.

The two passes fail differently and are treated separately throughout:

- **`sr` — strand rescue.** Transfers site-level evidence between physical
  strands (DAF, stranded Nanopore Hia5).
- **`cr` — consensus/composite recall.** Tests whether a nuc-like protected
  block is better explained by a population-recurrent TF configuration.

## 1. `sr`: every molecule was scored at one global deamination rate

A protected call is evidenced by the *absence* of deamination, so its LLR is a
direct function of `P(hit | accessible)`. The caller used a single global value
for every molecule. It has no per-read rate control of any kind.

Estimating each molecule's own `P(hit | accessible)` from its own `msp` calls
at the DddB `ind` locus (n = 2,950 molecules):

| | 5th | median | 95th |
|---|---:|---:|---:|
| real per-molecule `P(hit \| accessible)` | 0.093 | 0.387 | 0.517 |

The model's single global value is **0.497 — the 91st percentile of that
distribution.** It credited nearly every molecule as if it deaminated like the
top decile, worth a fixed `+0.666` nats per missed base. Eight zero-hit
opportunities therefore always scored `+5.33` nats, which is essentially the
entire observed median strong-rescue logBF of `+5.59`. **100% of strong rescues
are zero-hit calls**, so the pass was maximally exposed.

### Fix

`prototype.py::calibrate_cohort_deamination` estimates `P(hit | accessible)`
per molecule from that molecule's own accessible calls, shrunk empirical-Bayes
toward the model expectation for its own context composition (so the factor
measures enzyme efficiency, not sequence content). It runs before any state
model, prior, or proposal is fit, so both passes see the same LLRs. On by
default; `--global-deamination` restores the old behaviour.

### Cost

Re-running the DddB `ind` panel:

| | global (before) | per-molecule (after) |
|---|---:|---:|
| strong rescues | 452 | **89** |
| review rescues | 792 | 1,127 |
| median strong logBF | +5.71 | +3.66 |
| min strong logBF | **+0.18** | +3.00 |

Only **30 of 206** unique strong rescues survive (15%). The 176 demoted calls
had a median *old* logBF of `+5.59` — they were not marginal; the global rate
was systematically overcrediting them. Molecule efficiency factors run
0.29–1.26 (median 0.83): most molecules deaminate *less* efficiently than the
global assumption.

Note the old `min strong logBF` of `+0.18`. The `sr` likelihood gate is a sign
test (`values[top] > values[current]`, plus `LLR > 0` at every site), so under
the global rate a molecule could reach "strong" on ~0.2 nats of evidence. It
cannot now.

**Caveat:** 4,282 of 8,683 reads have fewer than 20 accessible opportunities in
their own `msp` calls and fall back to the global rate
(`--deamination-min-opportunities`). Those retain the old behaviour.

## 2. `cr`: N and TF had incomparable geometry priors

`integrated_nuc_log_likelihood` marginalized N over a **uniform prior across
every arithmetically possible (length, start)** — 887 geometries in a typical
case. The TF hypothesis marginalized over its **empirical source hulls**, all
sitting in one place. N therefore paid an Occam penalty of ~3.8 nats purely for
having its prior *written down vaguely*, and TF paid nothing comparable.

On a block with **no informative opportunities** — every molecular LLR
identically zero — the posterior must return the prior. It did not:

| nominal N:TF prior | prior implies | old model | fixed |
|---|---:|---:|---:|
| 1:1 | 0.500 | 0.949 | 0.597 |
| 10:1 | 0.091 | **0.652** | 0.129 |
| 100:1 | 0.010 | 0.158 | 0.015 |

The entire `+2.93` nat Bayes factor was geometry bookkeeping, not biology. The
frozen "10:1 N:TF conservative baseline" was really operating at **~1.9:1 in
favour of splitting.** Worse, the displacement scales with the number of
eligible N geometries, which is clipped by read span — 34 vs 887 geometries
gives TF posterior 0.15 vs 0.65 on *identical* molecular evidence. Short-read
chemistries were therefore systematically more conservative than PacBio, a
chemistry-correlated confound in exactly the cross-assay comparisons the panel
was making. Relabelling the prior axis could not have fixed this.

### Fix

`collect_local_nuc_geometry_records` gives N an **empirical geometry prior**
drawn from source molecules carrying an explicit nuc call over the same sites —
the exact analogue of the TF configuration library, with the same
`-log(support)` normalization, the same Gaussian call-edge kernel, and the same
leave-one-molecule-out holdout. Both hypotheses now carry comparably sharp,
data-fitted priors. The uniform enumeration is retained only as a fallback when
a locus has too few observed nuc geometries (`--min-nuc-geometries`, default
20); the path taken is reported in `integrated_nuc.geometry_prior`.

`test_uninformative_block_returns_the_prior_exactly` locks the invariant: with
matched geometry-prior concentration and no evidence, posterior == prior to
1e-6. `test_continuous_nucleosome_is_not_split_into_a_tf_pair` locks the
negative direction, which nothing previously tested.

### What was *not* wrong

The **protected-bridge state is not a bug.** A fully bridged TF pair predicts
the same protection pattern as a nucleosome — that is physically honest, and
`interval_evidence` being additive makes the bridged term algebraically equal to
the nuc-at-call LLR. But the bridge's evidence floor (`log(protected_gap_prior)`,
≈ −1.4 nats at 0.25) only binds when separator protection evidence exceeds that,
and with realistic dropout physics (a missed opportunity is worth only ~+0.43
nats, an observed hit ~−2.9) a short separator cannot reach it. Split support
falls monotonically as missed separator opportunities accumulate, exactly as
intended. An earlier draft of this audit wrongly called this fatal.

The **`sr` prior cannot overturn a molecule** (`prototype.py:1338`), and in the
production reports its prior odds are neutral (median 1.09 DddB, 0.91 UBA1 —
UBA1's prior actively *disfavours* the rescues it makes). That safeguard is real.

## 3. The control panel could not detect either defect

Every control in `PRODUCTION_VALIDATION.md` — the ±25/±50 bp boundary decoys,
the opportunity-matched local shifts, the pseudo-site nulls — is a **TF-geometry
control**. It compares a TF hypothesis against a *displaced TF hypothesis*, so
any bias shared by both arms cancels exactly.

A genuine continuous nucleosome run through the full production gate **passes
the decoy gate with a 25-log-unit margin** (the gate requires 3). The "4 strict
calls across 15,912 decoy scores" headline is a real specificity result about
*placement*, and says nothing about *splitting*.

There was **no true-nucleosome negative control anywhere** — which is precisely
the quantity the conservative end of the prior sweep exists to control.

### Fix

A random-genomic-region negative panel: dm6 is overwhelmingly nucleosomal, so
strong composite splits in randomly chosen well-covered regions are false splits
at high probability. Results in [§5](#5-negative-control-panel).

## 4. Validation q-values were not valid FDR quantities

Two independent problems in `validation/calibration.py`:

- **The null was not exchangeable across loci.** `normalized_delta` is
  `logBF / informative`, which is not depth-invariant: the BIC penalty per
  molecule and the sampling error of a per-molecule likelihood both scale with
  depth, so per-locus null *widths* varied about twofold. Pooling them made the
  shared null too narrow for some loci and too wide for others, so the p-values
  were not super-uniform and BH inherited no guarantee.
- **The 25 bp dedup median-averaged distinct decoys.** The code comment assumed
  the same genomic decoy is reused across neighbouring parents, but in the
  production data there are **zero exact-duplicate control intervals**. It was
  median-collapsing genuinely distinct nulls, shrinking null variance while the
  observed statistic got no such averaging.

### Fix

Studentize each observation by its own stratum's robust null spread (MAD, with
a family-level fallback) before pooling, and keep one representative null per
bin instead of median-averaging. Measured per-locus type-I error at the pooled
5% threshold, on the 827-candidate production panel:

| family / locus | before | after |
|---|---:|---:|
| hia5_pacbio / gm_napa | 14.3% | 4.3% |
| dddb / chr3l_0146 | 11.2% | 8.2% |
| hia5_pacbio / gm_uba1 | 10.1% | 3.8% |
| ddda / gm_napa | 10.0% | 8.6% |
| **max deviation from nominal 5%** | **9.3%** | **5.0%** |

**This is materially better, not perfect.** A few loci remain near 8% because
the null *shapes* differ across loci, not only their scales. The q-values are
much closer to valid but should still not be presented as exact FDR without a
per-locus null or a rank-based alternative.

The BH implementation itself was always correct (max |Δq| = 1.1e-16 against
`statsmodels`), leave-one-locus-out genuinely holds out the locus, and 0/1125
controls overlap a real site.

## 5. The local Wilson floor made the conservative end of the sweep unreachable

Found while building the negative panel. The site-local occupancy prior was
combined with the requested global prior as `max(global_prior, wilson_lower_bound)`.
Once the Wilson floor exceeded the global TF prior — which it does at essentially
every site that makes calls — **the requested prior stopped mattering entirely.**

In a random 20 kb region, at the requested 100:1 conservative prior, **all 2,249
candidates had their prior overridden by the floor**, and the effective odds
maxed out at 17:1 (median 6:1). The 10:1 and 100:1 scenarios produced
*bit-identical* counts (34 MAP / 14 strict / 9 strong). The conservative end of
the sweep did not exist.

This is a one-way ratchet: the local prior could only ever raise TF, never lower
it, and it swallowed the axis you were sweeping.

### Fix

The requested odds are now a **skepticism multiplier** on the site's own
occupancy prior rather than a competitor to it:

```
local_nuc_odds       = (1 - wilson) / wilson      # the site's own N:TF odds
effective_nuc_odds   = local_nuc_odds x requested_odds
```

The floor sets the baseline; the sweep always moves it. 1:1 means "trust the
local occupancy"; 100:1 means "demand 100x more evidence than local occupancy
alone implies." `test_local_occupancy_prior_never_swallows_the_requested_sweep`
locks the monotonicity.

## 6. Negative control panel

Six 20 kb random dm6 regions, five pooled PacBio BAMs, production parameters,
auto site discovery. dm6 is overwhelmingly nucleosomal, so a decoy-gated strong
split in a randomly chosen region is a false split at high probability. 7,881
candidate 90–220 bp blocks tested.

| requested N:TF | tested | MAP complex | strict | decoy-gated strong | false-split rate |
|---:|---:|---:|---:|---:|---:|
| 1:1 | 7,881 | 97 | 41 | 30 | **0.38%** |
| 10:1 | 7,881 | 47 | 17 | 13 | **0.16%** |
| 100:1 | 7,881 | 21 | 10 | 8 | **0.10%** |

The sweep is now monotone and the conservative end genuinely suppresses false
splits. Before these fixes the same panel gave 0.44% at 10:1 and an *identical*
0.40% at 100:1 — the prior had no purchase.

Isolating the geometry fix (§2) alone at 10:1 moved the rate only 0.44% → 0.38%,
because the Wilson saturation (§5) was pinning the effective prior near 6:1
regardless of what was requested. The two fixes had to land together.

**Read this as an upper bound on the true false-split rate, not a point
estimate.** Random dm6 regions do contain real regulatory footprints —
promoters, insulators — so some of the 8–30 strong calls are likely genuine.
The number to watch is the *ratio* across the sweep and against future model
changes, not its absolute value.

## 7. Still open

- **Bridged mass is still summed into `complex_posterior`.** A bridged TF
  complex and a nucleosome are observationally equivalent, so that component of
  the stage-1 aggregate carries information no data can ever move. It should be
  reported separately from resolvable-split mass (accessible separator), so a
  reader can see how much of a split's support is refutable in principle.
- **Residual q-value miscalibration** (§4): a few loci still sit near 8% type-I
  error at nominal 5% because null *shapes* differ across loci, not only scales.
- **Molecules with sparse accessible calls** still fall back to the global
  deamination rate (§1) — 4,282 of 8,683 reads at `ind`.
- The 220 bp ceiling and 90 bp floor remain hard bounds, contrary to the
  project's usual scoring-plus-`nq` preference. Blocks outside them are counted
  and skipped, never silently interpreted.
- The negative panel is 6 regions on one chemistry. It should be extended to
  DddA/DddB/Nanopore and to more regions before any operating point is re-frozen.

## 8. Re-validation required

Every panel in `PRODUCTION_VALIDATION.md`, `RESULTS.md`, `REVISED_RESULTS.md`
and `PAIRED_VALIDATION.md` was produced by the pre-fix model. The four boundary
loci (Nhomie, Homie, SF1, SF2), the cross-assay hierarchy panel, and the paired
shadow-callset BAMs all need regenerating before any of their counts are cited
again.

## Verification

- Consensus tests: 87 passed (81 before, plus 6 new invariant tests).
- Full FiberHMM regression: 645 passed, 3 skipped, 26 benchmark tests deselected.
- `pyproject.toml` `testpaths` now includes `consensus_recaller_collab`; the
  81 consensus tests were previously excluded from the default `pytest` run
  despite shipping as four console scripts.
