# ddda_nuc.json — DddA nucleosome caller

2-state FiberHMM for DddA DAF-seq data, specifically for nucleosome /
MSP / NFR calling. **Not a TF caller** — sub-nucleosomal calls
(footprints < 90 bp in the output) should be discarded; a separate
TF caller is run on top to identify Pol II / TF footprints inside
the MSPs.

## Why this exists

The legacy one-pass HMM (`legacy/ddda_pacbio.json`) was trained per-enzyme and
doesn't cleanly handle DddA's nuc/linker emissions. v3 (Poisson +
penetration) gave scattered positional calls compared to the HMM's
trained priors. This model takes a hybrid approach:

- **Transitions from `dddb_nanopore.json`** (known-good fly-chromatin
  state durations: nuc ~110 bp, linker ~14 bp means)
- **Emissions built from biological parameters**, not trained:
  - **Linker `P(methylated) = 1 − FN = 0.5`** — DddA accesses ~50%
    of C positions on open DNA (kinetic limit).
  - **Nuc `P(methylated | context) = FP[context] × fp_scale +
    breathing = ~0.06`** — per-context FP floor from calibrated
    untreated control + 5% breathing rate inside the octamer body.

These are *guessed* from biology, not trained. They were chosen by
sweeping FN and breathing and picking values that give ~10 nucs per
2.9 kb PS01499 amplicon with ~51% mono-nuc-sized calls and ~16%
long MSPs (plausible NFRs). See
`v3-caller/v3caller/analyses/ground_truth_validation/figures/ddda/`
for metaprofiles and snapshots at NAPA / UBA1 / PS01499.

## Parameters

```
fn         = 0.5       # linker false-negative rate
fp_scale   = 1.0       # multiplier on per-context FP rate
breathing  = 0.05      # in-nuc breathing rate (DddA-specific)
source     = models/dddb_nanopore.json  (transitions only)
fp_model   = phase0/data/fp_models/ct_nanopore_fp_3mer.json
```

## Rebuild

The build script lives in the research repository and is not shipped with
the package. It combined the DddB transitions (`dddb_nanopore.json`) with a
C->T false-positive context model using the parameters above.

## Usage

Normally the model is used through the preset, which also runs the dedicated
TF recaller (`ddda_TF.json`) and DddA nucleosome refinement in one pass:

```bash
fiberhmm-call -i <ddda.bam> -o calls.bam --enzyme ddda -c 4
```

To run the nucleosome HMM alone:

```bash
fiberhmm-apply -i <ddda.bam> --enzyme ddda -o out/ -c 4
```

After `fiberhmm-apply` alone, **filter output** to discard sub-nucleosomal
calls (footprints < 90 bp from ns/nl) before downstream analysis, or run
`fiberhmm-recall-tfs --enzyme ddda` on it: those positions are re-called by
the dedicated TF recaller.

## Limitations / future work

- No training data used — all parameters biologically guessed.
- Transitions borrowed from DddB Nanopore; may not perfectly match
  fly chromatin if DddA samples are from a different species /
  cell type.
- `fp_scale` is currently 1.0 (flat). Could tune per-context later.
- Validation is qualitative (metaprofile / snapshots on 3 amplicons).
  Better validation needs orthogonal MNase/ChIP data.
