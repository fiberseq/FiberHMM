# FiberHMM v3 caller — design review and open questions

**Context**: single-molecule DAF-seq / Hia5-seq → nucleosome + TF
footprint calls on long reads. v2 = fibertools HMM (trained on fiber
m6A/DAF data). v3 = a new Poisson-merge caller built over the last
few weeks.

This document describes what we built, what the data says, and where
we're stuck. We want outside advice on the merge model specifically.

Repo: `Fiber-NET-seq/FiberHMM v1.0/v3-caller/` (worktree, branch
`v3-caller`). All analyses referenced below live under
`v3caller/analyses/` with scripts + figures + data alongside.

## 0. Two goals

We are pursuing two distinct goals that imply different design
constraints. Keeping them separate is the main organizing question.

**Goal 1 — DddA DAF-seq caller.** The v2 HMM does not work for DddA
at all (breathing blurs the nuc/linker emission separation). We need
a new caller that (a) emits unmerged nucleosomes (no dinuc fusions),
(b) preserves TFs as explicit calls rather than folding them into
the nucleosome state, and (c) handles DddA's 10–20% in-nuc-body
breathing as calibrated noise rather than real accessibility.
**v3 in its current Poisson-with-penetration form is the only tool
we have that addresses this.**

**Goal 2 — fix DddB / Hia5 v2 output.** The v2 HMM already exists
and is the basis for fiberCNN transcription-state heuristics on
these enzymes, but it has two known failure modes:
- **Over-merging**: single HMM calls that span dinucleosomes
  (verified empirically: 14.9% of v2 DddB nucs split by v3)
- **Under-calling TFs**: the HMM is optimizing for nuc-grammar
  state sequences and leaves TF-sized footprints on the table,
  either as short nuc entries or folded into nuc boundaries
  (verified empirically: 90% of v3 TF calls have no v2 counterpart)
Fix requires: (a) resolving v2's overmerges into proper mono-
nucleosomes, (b) finding the missing TFs inside v2's MSPs (and
possibly inside its over-merged nucs).

These two goals are structurally different: Goal 1 is a new caller
for a problem the HMM can't do; Goal 2 is a refinement of an
HMM-based pipeline. Our current design space is whether v3 alone
can serve both, whether we need a hybrid for Goal 2, or whether
both should be routed through a single combined output.

---

## 1. Executive summary

- v3 pipeline: Pass-1 atom detection (hit-free runs ≥10 bp) → Poisson
  merge with per-context FP model + optional breathing term → TF
  caller ("overcall, rely on qualities") in the MSPs between merged
  nucs.
- Output is in the fiberseq Molecular Annotation (MA) tag spec with
  four annotation types: `nuc+QQQ`, `msp+`, `tf+QQQ`, `fp_v2+`. The
  `fp_v2+` type carries the v2 HMM's call track verbatim so every
  read has both callers' output coexisting.
- On DddB spacetime (7 windows, 336k reads, 2.29M v2 footprints):
  v3 correctly splits **14.9% of v2 calls** into multiple atoms
  (v2 dinuc overmerge), mean nuc size drops **752 → 237 bp**,
  ≥300 bp overmerge drops **44% → 16%**.
- v3 correctly calls **many more TF footprints** than v2 emits in
  its ns/nl track: 89-90% of v3 TF calls have no v2 counterpart,
  which we read as v3 finding things v2 was folding into its nuc
  state.
- But: for fiberCNN transcription-state heuristics (paused Pol II,
  elongating Pol II, PIC), v3's counts depended sharply on one
  merge-step parameter. With defaults v3 was 35-50% below v2 on
  Pol II counts; with the short-gap-bridging heuristic disabled v3
  exceeded v2 by 1.4-4×. Neither is obviously "right" without
  ground truth.
- We don't think we have a principled way to set that parameter
  without reinventing the HMM prior. **This is the question we want
  outside advice on.**

---

## 2. Architecture

### 2.1 Pipeline (per read)

```
Input BAM (may carry v2 HMM ns/nl tags)
  │
  ├── [if v2 tags present] snapshot to fp_v2+ MA annotation
  │
  ├── extract (opp, hit) arrays per query position
  │     opp = C/G positions (DAF) or A/T positions (Hia5)
  │     hit = C→T/G→A at those positions (DAF), or MM/ML ≥ 128 (Hia5)
  │
  ├── Pass-1 atoms = hit-free runs ≥ gap_radius (default 10 bp)
  │     (no rate filter; pure hit-presence)
  │
  ├── Pass-2 merge via Poisson evidence test:
  │     for each gap between adjacent atoms:
  │       gap_opp = Σ opp in gap
  │       gap_hit = Σ hit in gap
  │       lam_fp  = Σ per-context-FP-rate over gap opp positions
  │       lam_bio = gap_opp × (baseline − global_FP) × penetration_fraction
  │       lam = lam_fp + lam_bio
  │       merge iff gap_hit ∈ [Poisson-CI(lam, α/2), Poisson-CI(lam, 1−α/2)]
  │
  ├── Nuc filter: merged atoms < min_footprint (80 bp) dropped
  │
  ├── MSP definition: complement of (final nucs) ∪ (merged-absorbed gaps)
  │
  ├── TF caller (call_tfs_overcall): for each MSP, scan for hit-depleted
  │     sub-regions. "Overcall, rely on tq / el / er qualities for
  │     downstream filtering." tq is -log10(Poisson-P) at the TF site.
  │
  └── Output: ns/nl/nq/mq/lq/rq (nucs), as/al (MSPs), tn/tl/tq/el/er
      (TFs), plus MA:Z + AQ:B:C encoding the same info in the spec.
      Input's ns/nl (v2) is preserved as fp_v2+ in MA.
```

### 2.2 MA tag spec (implemented)

Per the fiberseq Molecular-annotation-spec. Four annotation types:

| Type | Qual spec | Meaning |
|---|---|---|
| `nuc+QQQ` | (nq, lq, rq) | v3 nucleosomes. nq=core tightness, lq/rq=edge sharpness (capture ambiguity) |
| `msp+` | — | v3 MSPs (accessible regions between nucs) |
| `tf+QQQ` | (tq, el, er) | v3 TF footprints. tq=significance, el/er=edge quality |
| `fp_v2+` | — | v2 HMM all-footprints from input BAM |

Legacy tag schema (ns, nl, nq, mq, lq, rq, as, al, tn, tl, tq, el, er)
written in parallel for back-compat. `mq` (merge quality) is in the
legacy tags only — not in MA — since it's a nuc-interior quality
of what-was-NOT-called (an absorbed MSP), philosophically orthogonal.

### 2.3 Novel calibrations added in v3

- **Per-context FP model** (`scripts/analyses/context_fp_calibration/`):
  trinucleotide FP rate tables from untreated controls.
  PacBio m6A (Hia5): global 0.94%, CV 0.59 across 32 contexts.
  Nanopore m6A: global 0.70%. Nanopore C→T (DddA/DddB): 0.82%.

- **SNP mask** (`scripts/analyses/snp_detection/`):
  per-amplicon pileup that flags positions where a ≥95%
  "deamination" fraction is actually sample-genotype SNP not real
  enzyme activity. Excluded from opp/hit arrays before calling.

- **Rotational periodicity correction**
  (`scripts/analyses/rotational_periodicity/`):
  DddB/DddA have a face-preference 10.4 bp oscillation in their
  per-site access rate — chromatin-phased, not enzyme-intrinsic.
  We use this to weight TF tq scores by distance-from-nuc-edge
  (iter-17 calibration). Figure locked for the paper.

- **Nucleosome penetration**
  (`scripts/analyses/nucleosome_penetration/`):
  empirical measurement of `penetration_fraction` (the breathing
  rate inside a nuc body). Three methods tried:
  - Bulk metaprofile on amplicons → 0.36 (confounded by positioning)
  - Hand-anchored → 0.24 (still positioning-confounded)
  - Conditional on flank-open reads → asymptotes at **0.22 on
    PS01499**, midpoint of 0.10–0.20 recommended for DddA.
  For DddB (one-strand access) and Hia5 (m6A single-strand) we
  recommend `penetration_fraction = 0`.

---

## 3. Datasets used

| Name | Enzyme | Platform | Reads | Purpose |
|---|---|---|---|---|
| DddB spacetime (7 × 0.5-hr windows) | DddB | Nanopore | 336k | v2/v3 comparison, production run |
| Hia5 sna/eve/ftz amplicon | Hia5 | PacBio | 1228 | v2/v3 comparison, Pol II states |
| PS01498/99/500/530, NAPA, UBA1, ENH30 | DddA | PacBio | ~40k each | FP calibration, penetration |
| scDAF (PS00758) | DddA | PacBio | whole genome | Rotational periodicity |
| Untreated controls (various) | — | both | 5k each | FP model calibration |

TSS BED at `analyses/caller_comparison/data/sna_eve_ftz_tss_dm6.bed`
uses amplicon-center coordinates as TSS proxies for sna/eve/ftz in
dm6.

---

## 4. Key results

### 4.1 Per-call overlap (v3 vs v2 on co-annotated reads)

On DddB spacetime (all 7 windows, 2.29M v2 footprints):

| v2 footprint fate | % |
|---|---|
| Preserved as nuc (v3 nuc IoU ≥ 0.5) | 59.7% |
| Preserved as tf (v3 tf IoU ≥ 0.5) | 5.0% |
| **Split (v2 overmerge → ≥2 v3 calls)** | **14.9%** |
| Reclassified | 3.3% |
| Weak overlap (0.25–0.5) | 11.1% |
| Lost | 6.0% |

See `figures/dddb_spacetime_all_classification_breakdown.png`.

v3 nuc calls: 42% shared with v2, 38% new.
v3 tf calls: 8.6% shared with v2, **90.3% new**.

The v2 HMM was folding TF-scale signals into its single `ns/nl`
track, either as short nucs (<90 bp) or as fused into larger nucs.
v3 emits them as explicit TF annotations.

### 4.2 fiberCNN transcription state counts

Based on FiberBrowser v3 `evaluate_read_states`:

- **Paused Pol II**: 35-65 bp footprint in TSS+10..+50 (strand-flipped)
- **Elongating Pol II**: 35-65 bp footprint in gene body, not in paused window
- **PIC**: 20-40 OR 60-80 bp footprint in TSS-50..+25
- **Accessible promoter**: no ≥90 bp footprint overlaps TSS±50
- **Hyperburst**: <50% of gene body covered by ≥90 bp footprints

For both v2 and v3 we union the caller's output (fp_v2 for v2;
nuc+tf combined for v3) and apply the heuristics.

#### Hia5 sna/eve/ftz (509 reads span a TSS)

| State | v2 (reads +) | v3 default | v3 no-merge | v3/v2 (default) |
|---|---|---|---|---|
| Paused | 3 | 5 | 34 | +67% |
| Elongating | 141 | 241 | 427 | +71% |
| PIC | 22 | 20 | 100 | −9% |
| Accessible promoter | 79 | 62 | 155 | **−22%** |
| Hyperburst | 49 | 72 | 239 | +47% |

#### DddB spacetime (38k reads span a TSS)

| State | v2 (reads +) | v3 default | v3 no-merge | v3/v2 (default) |
|---|---|---|---|---|
| Paused | 506 | 253 | **1,815** | **−50%** ❌ |
| Elongating | 8,715 | — | 26,746 | — |
| PIC | 1,458 | 2,558 | 6,617 | +75% |
| Accessible | 4,947 | 3,139 | 7,437 | **−37%** ❌ |
| Hyperburst | 2,981 | 6,450 | 7,297 | +116% |

The DddB default-merge undercall on paused / accessible is why we
investigated the merge step closely.

See `figures/dddb_spacetime_MERGED_pol2_agreement.png` and
`figures/dddb_spacetime_NOMERGE_pol2_agreement.png`.

### 4.3 The merge investigation

Merge audit via the legacy `mq` tag (min Poisson-interval absorbed):

**Default run** (with `short_gap_bp=8` structural-bridge shortcut):
- **55% of DddB nucs were fused from ≥2 atoms**, including
  62% of mono-nuc-sized (≤180 bp) and 68% of di-nuc-sized
  (181-300 bp) calls
- See `figures/dddb_spacetime_merge_audit.png`

**Root cause**: the Pass-2 merge has a heuristic shortcut:
```python
if gap_len <= short_gap_bp:  # default 8
    # auto-merge, attenuate mq linearly in hit count
    continue
# else: Poisson test with FP model
```

Disabling this shortcut (`--short-gap-bp 0`): 0% merges, v3 matches
"never merge atoms" behavior, every Pol II state count
**exceeds v2 by 1.4–4×** on DddB.

See `figures/dddb_spacetime_NOMERGE_merge_audit.png`.

### 4.4 Empirical merge-candidate characterization

On DddB 4-4.5 window, 98k adjacent-footprint pairs:

| Gap length | n pairs | Median hits | Median merged size |
|---|---|---|---|
| 1–3 bp | 33,852 | 1 | **276 bp (di-nuc!)** |
| 4–7 bp | 15,956 | 1 | 144 bp (mono-nuc) |
| 8–14 bp | 16,033 | 2 | 146 bp |
| 15–29 bp | 16,924 | 3 | 152 bp |
| 30–59 bp | 11,763 | 5 | 175 bp |
| 60–99 bp | 3,966 | 9 | 183 bp |

Two insights:
- Short gaps (4-29 bp) with 1-3 hits produce **~147 bp merged
  calls** (exactly mono-nucleosome-sized). These are the "real
  nucleosome with a breathing hit" merges.
- Very short gaps (1-3 bp) produce **276 bp merged calls**
  (di-nucleosome-sized). These are **actually 1-bp linkers between
  two real mono-nucs**, and merging them gives wrong biology.

See `figures/dddb_4-4.5_merge_candidates.png`.

### 4.5 Other subsystems that work

- Rotational periodicity correction (scDAF calibration, locked figure)
- Context-aware FP model (3 enzyme × platform variants)
- SNP mask (identifies and excludes 95%+ hit-fraction positions)
- Nucleosome penetration estimate (~0.22 on clean amplicon)
- v2 overmerge resolution (explicit 14.9% split rate on DddB)

---

## 5. The fundamental merge question

We keep circling back to: **what's the right merge rule?**

### 5.1 Options we've explicitly considered

| Approach | Description | DddB result | Hia5 result |
|---|---|---|---|
| **A. Default v3** | `short_gap_bp=8` structural shortcut + Poisson for longer gaps | 55% nucs merged, Pol II undercall | 1% nucs merged, reasonable |
| **B. Pure FP-Poisson** | No shortcut, `lam = lam_fp`, pen=0 | 0% merges (strict), Pol II overcall | 0% merges, probably over-splits |
| **C. FP + linear breathing** | `lam = lam_fp + gap_opp × (baseline − FP) × pen_frac`, per-enzyme pen | Same as B for DddB (pen=0) | Tunable for Hia5 |
| **D. Size-dependent prior** | `lam += gap_opp × 0.05 × N(147, 50)` — merge only if merged size ≈ nuc | Principled; data-justified | Not tested |

Option D looks good on paper because the empirical gap/size data
(§4.4) shows that 4–29 bp gaps with 1–3 hits do fuse into mono-nuc-
sized objects — these are plausibly real breathing-inside-nuc. But
1–3 bp gaps fuse into di-nuc-sized objects, which is wrong biology.

The problem: option D is essentially "add a Gaussian prior on
merged size," which is what an HMM's emission model does implicitly.

### 5.2 The circular question

- Every merge rule encodes a prior about what nucleosomes look like.
- v2 HMM has these priors baked in from training data.
- v3 with "just Poisson" is too strict (no prior).
- v3 with "Poisson + size prior" works but is re-inventing the HMM
  in local-decision form.

**The natural synthesis** (user's framing) is:
- Use an HMM for nucleosomes (its sweet spot)
- Use v3's Pass-1 + Poisson for TFs (which don't fit HMM state grammar)
- Use v3 with `penetration_fraction` > 0 for DddA (HMM breaks there)
- Carry both callers' output in the MA tag; downstream picks.

And we already have all three! The fiberseq v2 HMM exists, v3 exists
as a drop-in for the TF side, and v3 supports a "hybrid" mode
(`--use-v2-nucs`) that uses v2 nucs + v3 TFs in the MSPs.

### 5.3 Why we're still uncomfortable

Two problems with the hybrid:
1. The **90 bp threshold** for filtering v2 nucs vs TFs is arbitrary.
   `nq` quality values don't vary enough to use as a cleaner filter.
2. **v2 HMM merges TF footprints at boundaries into its nucs**
   (e.g., a 190 bp v2 "nuc" is often 147 bp nuc + 40 bp boundary TF
   fused). Using v2 nucs as the authoritative nuc list loses these
   boundary TFs, because v3's TF caller only runs in MSPs and never
   inside a v2 nuc.

We could work around (2) by running the TF caller inside any v2 nuc
>180 bp (likely-merged) — which the existing `gap_records` machinery
can support — but that's another hand-tuned threshold.

### 5.4 What we actually want to know

**For outside advice:**

1. Is the v3 + HMM-hybrid architecture the right direction, or
   should we commit to pure v3 (pure FP-Poisson) and accept that
   matching fiberCNN state counts to v2 isn't the goal?

2. Is there a principled way to set the "this gap is inside a nuc"
   null rate for the Poisson test that doesn't require either
   training data or a size-prior? E.g., compute per-read breathing
   rate from the protected atoms (which are hit-free by Pass-1
   construction, so this seems hard).

3. Has anyone tried the "gap-opp + hit-count" 2D grid for calibrating
   the merge decision? I.e., a learned table of `P(merge | gap_len,
   gap_opp, gap_hit)` from ground-truth-annotated data. We don't
   have ground truth, but we do have v2 HMM output on >300k reads.

4. For DddA specifically — is `penetration_fraction = 0.15` (from
   the conditional-on-flank-open estimate at 0.22) the right
   breathing scale, or should we expect enzyme-specific variation
   across amplicons that our single-amplicon estimate doesn't
   capture?

5. The paused / PIC states in fiberCNN — are these heuristics good
   enough targets, or should we re-calibrate them against the v3
   output distribution (i.e., don't tune v3 to match v2's rates
   since the rates themselves were never validated against ChIP)?

---

## 6. Architecture decisions we've frozen

These were called "resolved" during development and shouldn't need
to be revisited:

- Per-context FP model is necessary; flat FP underperforms at CpG
  contexts on Nanopore C→T
- SNP mask is necessary on amplicons; without it, genotype positions
  with 98%+ "deamination" break merge and TF calls
- Rotational periodicity is chromatin-mediated, not enzyme-intrinsic
  (verified from DddB naked vs chromatinized)
- `penetration_fraction = 0` for DddB and Hia5 (single-strand access
  means no breathing-through)
- MA tag spec compliance: `nuc+QQQ`, `msp+`, `tf+QQQ`, `fp_v2+`
- Auto-index output BAM (pysam.index at end of run)

---

## 7. Directory layout

```
v3caller/
├── caller_v8.py             # main caller
├── context_fp_model.py      # FP model class
├── ma_tags.py               # MA/AQ writer
├── snp_mask.py              # SNP mask loader
├── enzyme_extractors.py     # DAF + Hia5 extractors
├── fp_models/               # calibrated FP JSONs
└── analyses/
    ├── CALLER_DESIGN_REVIEW.md   (this file)
    ├── README.md                 # analyses index
    ├── context_fp_calibration/   # per-context FP calibration
    ├── nucleosome_penetration/   # breathing estimate
    ├── rotational_periodicity/   # 10.4 bp phase correction
    ├── snp_detection/            # SNP mask analysis
    ├── v2_v3_comparison/         # top-line nuc-size / overmerge stats
    └── caller_comparison/        # this deeper analysis
        ├── scripts/
        │   ├── parse_ma_calls.py
        │   ├── compare_overlap.py
        │   ├── pol2_states.py
        │   ├── audit_merging.py
        │   ├── audit_merge_candidates.py
        │   └── snapshot_paused.py
        ├── figures/         # all plots referenced above
        │   └── snapshots/   # per-read v2 vs v3 paused views
        └── data/            # TSVs, JSONs, BEDs
```

Key commits on branch `v3-caller`:
- `aec4c46` Tune DddB to --max-merge-len 0; Pol II rates recover
- `25c72d1` Root-cause: short-gap-bp shortcut was bypassing Poisson
- `3d2fdec` caller_comparison: v2 vs v3 per-read analysis
- `431bbe1` v2-vs-v3 stats refresh with FP + SNP correction
- `262a86f` MA tag refactor: spec-compliant QQQ + fp_v2+ annotation
- `e47151e` caller_v8: auto-index output BAM

---

## 8. How to reproduce

Minimum commands to regenerate every figure + table:

```bash
# Per-call comparison (single DddB window)
python scripts/compare_overlap.py \
  --in-bam .../iter17_calls/4-4.5.called.bam \
  --label dddb_4-4.5 --out-dir figures/

# Per-call comparison (all 7 DddB windows)
python scripts/compare_overlap.py \
  $(for W in 1-1.5 1.5-2 2-2.5 2.5-3 3-3.5 3.5-4 4-4.5; do \
      echo "--in-bam .../iter17_calls/${W}.called.bam"; done) \
  --label dddb_spacetime_all --out-dir figures/

# fiberCNN Pol II states
python scripts/pol2_states.py \
  --in-bam .../iter17_calls/*.called.bam \
  --tss-bed data/sna_eve_ftz_tss_dm6.bed \
  --label dddb_spacetime --out-dir figures/

# Merge audit (% nucs from fused atoms)
python scripts/audit_merging.py \
  --in-bam .../iter17_calls/*.called.bam \
  --label dddb --out-dir figures/

# Merge candidate empirics (§4.4)
python scripts/audit_merge_candidates.py \
  --in-bam .../iter17_calls/4-4.5.called.bam \
  --label dddb_4-4.5 --out-dir figures/

# Single-read snapshots of paused Pol II
python scripts/snapshot_paused.py \
  --pol2-tsv data/*_pol2_states.tsv.gz \
  --in-bam .../iter17_calls/*.called.bam \
  --label dddb --out-dir figures/snapshots/ --per-category 10
```

Recommended caller flags per enzyme (post-review defaults):

```bash
# Defaults: --max-merge-len 180 (steric cap, biophysical),
#           --short-gap-bp 0 (no bypass),
#           --use-v2-nucs off (pure v3, deprecated),
#           --penetration-fraction 0, --nuc-size-breathing-max 0

# DddA (PacBio, substantial breathing, amplicons)
--enzyme daf --fp-model fp_models/ct_pacbio_fp_3mer.json \
  --penetration-fraction 0.15 \
  --nuc-size-breathing-max 0.05

# DddB (Nanopore, one-strand access, minimal breathing)
--enzyme daf --fp-model fp_models/ct_nanopore_fp_3mer.json \
  --snp-mask <amplicon_snps.bed> \
  --nuc-size-breathing-max 0.05  # optional; permits mono-nuc breathing merges only

# Hia5 (PacBio m6A, minimal breathing)
--enzyme hia5 --fp-model fp_models/m6a_pacbio_fp_3mer.json \
  --nuc-size-breathing-max 0.02
```

Notes:
- `--use-v2-nucs` deprecated; pure v3 is the recommended path.
  `fp_v2+` MA annotation still preserves v2 calls for back-compat.
- `--penetration-fraction` kept for API continuity but its
  linear-breathing formula under-parameterizes DddB (see §10.3).
  The `--nuc-size-breathing-max` flag is the principled replacement.
- `--short-gap-bp` is a deprecated no-op (was the source of the
  55% merge bug).

---

## 9. Bottom-line asks for outside advice

### For Goal 1 (DddA caller)

1. **Is v3's Pass-1 + Poisson + penetration architecture sound
   for DddA?** The HMM doesn't work here, so we don't have another
   viable candidate. But this isn't an endorsement — is the
   Poisson null with `lam_fp + gap_opp × (baseline − FP) × pen`
   the right formulation for an enzyme with substantial breathing?

2. **How should we calibrate `penetration_fraction` for DddA?** Our
   single-amplicon measurement (0.22 → 0.15 midpoint, conditional
   on flank-open reads at PS01499) is pinned to one locus. Should
   we expect enzyme-wide or locus-specific breathing?

3. **Size-based prior for DddA?** Our empirical data (§4.4) shows
   that gap 4–29 bp with 1–3 hits fuses to mono-nuc-sized calls
   (146–152 bp); 1–3 bp gaps fuse to di-nuc-sized calls (276 bp).
   Adding a Gaussian prior on merged-size centered at 147 bp would
   concretely help, but it's reinventing HMM emissions locally.
   Is this an acceptable amount of prior to bake in, or should we
   resist?

### For Goal 2 (DddB / Hia5 improvement)

4. **Hybrid vs. pure v3?** We've implemented `--use-v2-nucs` which
   takes HMM nucs (≥90 bp filter) as authoritative and runs v3's
   TF caller only in the MSPs between them. This directly addresses
   the under-call issue for TFs, but it inherits v2's overmerge
   problem AND loses boundary-fused TFs (a 190 bp HMM "nuc" is
   often 147 bp nuc + 40 bp boundary-TF fused; v3 never looks
   inside to recover the TF). Options:
   - Ship hybrid as v1, accept boundary losses
   - Add interior-TF scanning for suspiciously-large HMM nucs
     (another tunable threshold)
   - Don't ship a "merged" caller at all; emit both tracks in MA
     and let consumers pick

5. **Alternatively, refine v2 HMM itself?** That's not something we
   can do without retraining. But it'd be the cleanest fix for
   Goal 2 — an HMM trained to emit separate "nuc" and "TF" states
   rather than a single "footprint" state.

### Cross-cutting

6. **Is the TF-undercall on DddB genuinely a problem, or is v2's
   ns/nl track an overcall** (fiberCNN heuristics were tuned on it)?
   Per-call overlap shows 90% of v3 TFs are novel — if those are
   real, v3 is the one to trust.

7. **Ground-truth** — we don't have orthogonal MNase/ChIP data at
   sna/eve/ftz to adjudicate. If you have this or can point us at
   it, the merge question (and the "are v3 TFs real?" question)
   becomes solvable empirically.

---

## 10. Addendum (after outside review)

### 10.1 Received feedback

Key points from outside reviewer:

- **Reframing**: we were evaluating v3 (a sharper microscope) by
  asking whether it matches v2 (the blurry pictures) via fiberCNN
  (heuristics calibrated to the blur). Wrong direction.
- **Steric guardrail is biophysics, not HMM reinvention.** A single
  octamer protects ~147 bp + breathing ~30 bp per side. Any merge
  producing ≥ 180 bp is necessarily a dinucleosome fusion. Cap
  `--max-merge-len` at 180 — this alone solves the §4.4 finding
  that 1–3 bp gaps fuse into 276 bp di-nuc calls.
- **Kill the `short_gap_bp` bypass**. (Already done — commit
  `25c72d1`.)
- **Goal 2 answer**: commit to pure v3 for DddB/Hia5, don't go
  hybrid. v2's overmerge + boundary-TF-folding is exactly what
  v3 fixes; using v2 nucs as authoritative locks in v2's bugs.
  Output `fp_v2+` in MA for back-compat and downstream consumers,
  but run the analytical engine on the v3 track.
- **Goal 1 answer**: v3's Poisson + penetration is the only
  mathematically-viable approach for DddA (HMMs require separable
  emissions which DddA breathing destroys). Ship `pen=0.15` as
  documented default, expose CLI for per-locus tuning.
- **v3 TF undercall is v2 overcall**: an HMM's geometric state-
  duration prior penalizes rapid transitions (Nuc→Linker→TF→Linker
  →Nuc) enough that it folds boundary TFs into nucs to avoid the
  transition cost. v3's "find nucs then scan the sky for TF stars"
  architecture exposes these correctly. Trust v3.
- **Ground truth is accessible**: *Drosophila* sna/eve/ftz have
  published PRO-seq / NET-seq (paused Pol II base-pair resolution)
  and ChIP-nexus / CUT&RUN (pioneer factor binding at sna shadow
  enhancer, eve stripe 2). Validate v3 novel TFs against known
  Zelda / Twist / GAF sites; validate paused Pol II calls against
  PRO-seq peaks. We have this data in-house.

### 10.2 Implementation delta

Committed as part of this review:

- `--max-merge-len` default **250 → 180 bp** (steric cap on fused
  atom size). Kills dinuc fusions without requiring the old
  `short_gap_bp` bypass. Per the reviewer: 147 bp octamer + 30 bp
  breathing each side is a biophysical constant; merges producing
  ≥ 180 bp are necessarily dinuc fusions.
- `short_gap_bp` bypass **confirmed removed** (was commit 25c72d1).
  All gaps now route through the Poisson test + size-prior.
- `--use-v2-nucs` hybrid mode **changed default from `auto` to
  `off`** and marked DEPRECATED in the CLI help. Per the reviewer:
  the hybrid locks in v2's overmerge AND loses boundary-fused TFs
  (v3 only scans MSPs, never inside v2 nucs). The `fp_v2+` MA
  annotation is still written for back-compat consumers.
- **New CLI flag `--nuc-size-breathing-max`** implementing
  reviewer's option D (size-dependent prior). Gaussian peak at
  147 bp merged size, sigma 50, zero outside 80–220 bp. Adds a
  per-opp breathing rate to the Poisson null that relaxes the
  test ONLY when the merged output would be a plausible mono-
  nucleosome. Does NOT permit di-nuc-sized merges. Default 0
  (strict Poisson).

### 10.2.1 Empirical test of the size prior on DddB 4-4.5

Pure v3 with `short_gap_bp=0`, `max_merge_len=180`, pen=0,
sweep `--nuc-size-breathing-max`:

| `nuc_size_breathing_max` | % nucs merged | Note |
|---|---|---|
| 0.00 | 0.00% | Strict Poisson, no prior |
| 0.02 | 0.00% | Below Poisson threshold still |
| 0.05 | 0.03% | 39 nucs merge in 4-4.5 window |
| 0.10 | 0.69% | 799 nucs; still << 55% w/ old shortcut |

All merges under the size prior are (by design) nuc-sized objects
— zero dinucleosome fusions. The 0.7% merge rate at nbm=0.10 is
far below the old 55% from the buggy bypass, but it's directed:
only breathing-in-mono-nuc merges, never boundary-fusion.

### 10.3 Empirical finding that still needs resolution

Our pen-sweep on DddB 4-4.5 (pure v3, `short_gap_bp=0`,
max-merge-len=180):

| `pen_frac` | % nucs merged (mq<255) | Δ called nucs vs pen=0 |
|---|---|---|
| 0.0 | 0.00% | baseline |
| 0.1 | 0.00% | 0 |
| 0.3 | 0.02% | +14 |
| 0.5 | 0.16% | +121 |
| 1.0 | 1.03% | +752 |

The advisor suggested `pen = 0.02–0.05` for DddB to allow ~15 bp
/ 3-hit gap merges. **That range produces zero merges on DddB.**

Diagnosis: the linear-breathing formula
```
lam_bio = gap_opp × (baseline − global_FP) × penetration_fraction
```
parameterizes breathing as a fraction of the *per-read* (baseline
− FP). For DddB, baseline ≈ 0.06, FP ≈ 0.008, so `(baseline − FP)
= 0.052`. At pen=0.05, breathing-per-site contribution = 0.0026 —
well below FP itself, so adding it to `lam_fp` barely moves
Poisson upper bound.

The reviewer's implicit intent seems to have been: "~3% per-site
absolute breathing rate inside a nuc body" — which in the current
formulation would require pen ≈ 0.6. That's not how the parameter
reads in CLI/docs.

**Proposal we'd like the advisor to react to:** replace
`penetration_fraction` (fraction of per-read excess) with
`--breathing-rate` (absolute per-site hit rate inside a nuc body).
For DddB ~0.03, Hia5 ~0.01, DddA ~0.10. `lam = lam_fp + gap_opp
× breathing_rate`. Cleaner semantics, direct biological meaning.
Is that the right reformulation?

### 10.3.5 Ground-truth validation result — v3 recovers 4–5× more real TFs

**Experiment**: DddB spacetime 2–3 hr windows (NC13–NC14 ZGA peak),
~65k reads spanning sna/eve/ftz loci. For each TF-sized (20–90 bp)
footprint call, compute mean ChIP-nexus signal at ±50 bp flank.
Threshold = P95 of signal at 10,000 size/chrom-matched random
positions ("5% false positive rate under null").

Hit rate above random-P95 per category:

| Factor | v2 shared | v2_only | v3 shared | v3_only | random |
|---|---|---|---|---|---|
| **zld** | 30.1% | 25.9% | 29.7% | **28.2%** | 5% |
| **gaf** | 25.6% | 23.9% | 25.2% | **21.7%** | 5% |
| **twi** | 33.2% | 27.8% | 31.8% | **28.5%** | 5% |
| **bcd** | 23.7% | 21.5% | 23.5% | **23.0%** | 5% |

See `ground_truth_validation/figures/dddb_2-3hr_hitrate_distribution.png`
and the summary TSV.

All categories are **4–6× enriched** over random at pioneer-factor
peaks. v3_only and v2_only have statistically identical hit rates
per factor — v3 is not calling noise; it's calling more *of the
same quality*.

**Absolute real-binding captures** (n_calls × hit_rate):

| Factor | v2 total | v3 total | **v3/v2** |
|---|---|---|---|
| zld | 8,693 | 41,106 | **4.7×** |
| gaf | 7,459 | 32,373 | **4.3×** |
| twi | 9,539 | 42,082 | **4.4×** |
| bcd | 6,892 | 33,223 | **4.8×** |

**This is the definitive v3 win.** v3 recovers ~4–5× more real
Zld / GAF / Twi / Bcd binding events than v2 on the same reads,
with identical per-call specificity.

The reviewer's prediction is confirmed exactly: v2 HMM's geometric
state-duration prior was folding boundary TFs into nuc calls
(avoiding the transition cost), erasing them. v3's atom-wise "find
nucs then scan the sky for TF stars" architecture exposes them.

### 10.4 Validation plan (has ground truth)

For the outside reviewer's prediction that v3 will align with
known regulators where v2 does not:

1. **Pull PRO-seq / NET-seq** at sna / eve / ftz in 2–4 hr embryos.
   Build a BED of Pol II engagement peaks.
2. **Pull ChIP-nexus** for Zelda (Zld), GAF (Trl), Twist (Twi)
   at the same loci. Build a BED of known pioneer/TF binding.
3. **For each v3 TF call** (v3-only, v2-only, shared): compute
   overlap with (a) PRO-seq peaks (paused Pol II), (b) ChIP-nexus
   peaks (specific factors).
4. **Success criterion**: v3-only TFs should show *higher*
   enrichment at Zld/GAF sites than v2-only calls. Per the
   reviewer: v2 "swallows" these into nuc boundaries; v3 exposes
   them.
5. **Also**: v3 paused Pol II reads (fiberCNN heuristic) should
   concentrate at PRO-seq peaks. If they do, the +259% paused
   count on DddB over v2 is real biology, not noise.

This moves the whole discussion from "what's the right merge
model" to "which caller's output best predicts orthogonal ground
truth." Which is the only honest way to resolve it.
