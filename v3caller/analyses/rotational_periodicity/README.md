# Rotational phase periodicity (10.4 bp)

## Question

DAF-seq pair-correlation shows a ~10.4 bp oscillation — hits
cluster at helical-pitch intervals. Is this:
- **Enzyme-intrinsic face preference**: the enzyme blocks one face
  of the helix regardless of context? → applies to any DAF data
- **Chromatin-phased accessibility**: the enzyme only shows
  periodicity when DNA is wrapped around a nucleosome? → depends
  on local chromatin context

This matters for calling TFs near nucleosome edges: if the FP
varies with helical phase, a few consecutive misses at d=5-10 bp
from a nuc edge are biased differently than at d=15-20 bp.

## Key results

Calibration figure: `figures/periodicity_edge_vs_far_FINAL.png`
(DO NOT OVERWRITE — locked-in reference).

### Core finding

**10.4 bp periodicity is chromatin-mediated, not enzyme-intrinsic.**

Evidence:
| sample | cosine amplitude A | behavior |
|---|---|---|
| DddB spacetime (chromatin) | +0.21 | strong oscillation |
| DddB naked (top 10% deam) | +0.010 | flat (null) |
| DddB naked (top 20%) | +0.009 | flat |
| Hia5 naked | -0.006 | flat |
| Hia5 chromatinized | smooth decay, no osc | bulk enrichment only |

DddB shows 20× drop in oscillation amplitude from chromatinized
→ naked. Hia5 shows no oscillation even on chromatin (accesses
both faces). So the 10 bp periodicity requires:
1. A face-sensitive enzyme (DddA/DddB)
2. DNA wrapped against a histone (imposes rotational register)

Neither alone produces it.

### Edge-anchored profile

From `periodicity_edge_vs_far_FINAL.png` (scDAF, 3.9M nuc-edge
anchors, ≥200 bp-from-nuc control):

- At d=+5 bp from edge: 0.847× baseline (below — partial wrap)
- At d=+10 bp: **1.515× baseline** (first helical peak)
- At d=+15 bp: 1.095×
- At d=+20 bp: 1.109×
- At d=+30 bp: 1.008× (decayed to baseline)
- ≥200 bp control: 1.09-1.17 (flat, no oscillation)

**Decay to baseline by d=+30 bp** — the rotational imprint
extends ~30 bp into accessible DNA past a nuc edge.

### Amplitude by region

From `periodicity_by_region.png`:
| region | cosine A | peak@10 |
|---|---|---|
| short linker (≤80 bp) | +0.092 | 1.46 |
| mid linker (80-200) | +0.061 | 1.27 |
| large NFR (≥200 bp) | +0.027 | 1.28 (but flat after 50 bp trim) |

Decay length ~60 bp from nearest nuc; both flanking nucs
contribute additively when the region is short.

## Modeling implication (iter-17)

TF significance (tq) is corrected per-miss by:
```
rate_profile(d) = 1 + A × edge_quality × cos(2π·d/10.4) × exp(-d/τ)
```
with A=0.35, τ=15 bp, dampened by the nearest nuc edge's quality
(lq/rq) to prevent inversion on ambiguous edges.

See `CHECKPOINT_ITER17.md` §4.2 for the full recipe.

## How to reproduce

```bash
# Cross-enzyme comparison (DddA vs DddB vs Hia5, chromatin vs naked)
python scripts/periodicity_compare.py

# Nuc-edge-anchored profile (THE calibration figure)
python scripts/periodicity_anchor_control.py \
    --bam scdaf_iter17.bam --label scDAF --max-reads 40000

# Region-stratified (linker vs NFR)
python scripts/periodicity_by_region.py

# Distance-from-nearest-nuc decay curve
python scripts/periodicity_anchored.py
```

## Caveats

- **Amplitude (A=0.35) is calibrated from scDAF** (DddA PacBio).
  DddB/Hia5 might differ — needs re-calibration when we wire the
  rotational correction into those paths.
- **Damping constant (τ=15 bp)** was fit globally; real decay
  may depend on local chromatin state (arrayed vs isolated nucs).
- **Phase offset φ ≈ 0** (peak at d=10 ≈ one full turn) assumed;
  may shift for different nuc positioning mechanisms.
