# Ground-truth validation (PRO-seq + ChIP-nexus)

## Question

Outside reviewer's prediction: v3's TF calls that have no v2
counterpart (90% of v3 TFs are novel on DddB) will align with
published ChIP-nexus / CUT&RUN peaks for pioneer factors (Zelda,
GAF, Twist) at the sna / eve / ftz loci. v2 HMM "swallowed" these
into nucleosome boundaries due to its geometric state-duration
prior penalizing rapid transitions.

Similarly, v3's paused Pol II reads (fiberCNN heuristic output,
35–65 bp footprint in TSS+10..+50) should concentrate at published
PRO-seq / NET-seq Pol II pause peaks.

This is the first analysis that moves v2-vs-v3 from "which matches
a v2-calibrated heuristic" to "which better predicts orthogonal
biology."

## Required data (already in-house)

- **PRO-seq or NET-seq** at 2–4 hr *Drosophila* embryos, mapped
  to dm6. Ideally the Zeitlinger or Lis lab publications.
  - Path: TBD (ask user)
- **ChIP-nexus** for pioneer factors at sna / eve / ftz:
  - Zelda (Zld): most relevant for early zygotic genes
  - GAF / Trl: widespread pioneer
  - Twist (Twi): mesoderm-specific, overlaps sna
  - Path: TBD (ask user)

## Planned analyses

### 1. TF-call vs ChIP-nexus overlap

For each v3 TF call on DddB spacetime reads:
- Convert query-coordinate TF center to reference coordinate
- Annotate: `v3_only` / `v2_only` / `shared` (from caller_comparison)
- Bed-intersect with ChIP-nexus peaks for each factor
- Measure overlap rate per category

**Success criterion**: v3-only TFs show *higher* ChIP-nexus overlap
rate than v2-only TFs. Per the reviewer, v3-only TFs should be the
boundary-adjacent hits that v2 HMM folded into nucs.

### 2. Paused Pol II vs PRO-seq

For each read flagged "paused" by fiberCNN (35–65 bp footprint
in TSS+10..+50):
- Project the footprint center to reference coordinate
- Bed-intersect with PRO-seq pause peaks (top quartile by signal)
- Measure concordance: fraction of paused calls that fall within
  a PRO-seq peak window (±50 bp)

**Comparison**: compute this for v2-only paused calls, v3-only,
and shared. If v3 is correctly finding paused Pol II that v2
missed, v3-only should have high PRO-seq concordance.

### 3. Merge-sensitivity control

Re-run DddB with different merge settings (strict Poisson pen=0,
size-prior nbm=0.05, nbm=0.10) and compute PRO-seq / ChIP-nexus
concordance per setting. The setting with best concordance wins.

## Scripts to write

- `scripts/project_calls_to_ref.py` — take a BAM with MA tags,
  convert nuc / tf / fp_v2 centers from query to reference,
  output BED per call category.
- `scripts/intersect_with_peaks.py` — bedtools-style intersect
  between caller-output BED and external peak BEDs; report per-
  category enrichment.
- `scripts/plot_concordance.py` — figures comparing v2/v3
  concordance with ground truth across states.

## Directory layout

```
ground_truth_validation/
├── README.md               (this file)
├── scripts/                (analysis code — TBD)
├── figures/                (concordance plots)
└── data/
    ├── external/           (symlinks to PRO-seq / ChIP-nexus BEDs)
    ├── v3_tf_calls_ref.bed (projected from BAMs)
    └── overlap_summary.tsv
```

## Status

Pending ground-truth data locations. Once provided, estimate
~2 days to build the pipeline + generate the definitive v2-vs-v3
validation figures.
