# LLR TF Recaller — checkpoint

**Date:** 2026-04-13
**Scope:** second-pass TF footprint caller that runs on top of v2 FiberHMM
output. Intended as a lightweight replacement for the v3-caller pipeline
for this manuscript. Not a new nucleosome segmenter.

---

## 1. Motivation

The v3 caller (`v3caller/caller_v8.py`, ~1200 LOC) aimed to replace the
v2 HMM with a pure-Poisson segmenter. V-plots showed a catastrophic noise
floor explosion (see `v3caller/analyses/ground_truth_validation/figures/`
earlier runs): v2 produced clean TSS-centered Pol II peaks, v3 produced
static. Root cause: pure per-position evidence scoring with no spatial
prior flags every random miss-streak as a footprint. v2's HMM transition
matrix is doing load-bearing regularization that the v3 approach discards.

Decision: shelve v3 for this paper, distill to a minimal **LLR recaller**
that reuses v2's emission table as the per-context null and scans **only
inside v2's already-called MSPs + short-nucs**. This preserves v2's
spatial segmentation and only adds TF-specific scoring where it's safe.

---

## 2. Architecture

### 2.1 The core LLR test ("Viterbi bypass")

v2's HMM ships a 2-state emission table `emissionprob_` of shape
`(2, 8194)`:

- Column `c ∈ [0, 4096)`: encoded hexamer with **hit** (methylated target)
- Column `4096`: non-target position
- Column `4097 + c`: same hexamer with **miss** (unmethylated target)

Because `load_model()` applies `normalize_states()`, state 0 is always
*protected/inaccessible* and state 1 is *accessible*. Given a read's
per-position `(is_target, ctx, is_hit)` array, the per-position log-
likelihood-ratio for protected vs accessible is a direct lookup:

```
llr_miss[c]  = log P(miss | c, protected) - log P(miss | c, accessible)
llr_hit[c]   = log P(hit  | c, protected) - log P(hit  | c, accessible)
```

For Hia5 pacbio, `median(llr_miss) ≈ 1.67` nats (positive — misses mildly
favor the protected state), `median(llr_hit) ≈ -4.34` nats (large
negative — hits strongly reject the protected hypothesis).

Because we reuse v2's trained emission table, per-enzyme calibration is
implicit: DAF (sparser labeling, `median(llr_miss) ≈ 1.0`) and Hia5
(dense, `1.67`) auto-scale without any per-enzyme tuning of the recaller.

### 2.2 Kadane-local scoring

Inside each scan interval, maintain a running LLR. On each target
position, add `llr_miss[ctx]` or `llr_hit[ctx]`. Non-target positions
contribute 0. When the running sum drops to ≤ 0, flush: if the peak
observed since the last reset is ≥ `min_llr` *and* the number of
informative target positions in the peak is ≥ `min_opps`, emit a TF call
spanning the climb-start to the peak position.

This Kadane form (vs. strict hit-termination) was suggested by Gemini to
let a real footprint absorb one rogue hit without being shattered. Works
for DAF (per-hit penalty ~-3.2 vs. per-miss ~+0.3 is near-symmetric);
works less well for Hia5 where the -4.34 / +1.67 asymmetry means a single
hit inside a real footprint usually kills the run. See §6 for the length
disparity this produces and a possible fix.

### 2.3 Scan space (per read)

**Three sources, merged and clipped to read bounds:**

1. All MSP spans from `as/al`.
2. Full span of every v2 nuc with `nl < --long-nuc-min` (default 90 bp).
   These are v2's already-called sub-nucleosomal footprints; rescoring
   them with the LLR is the primary sanity check.
3. **DISABLED BY DEFAULT:** outer `--boundary-sweep` bp of each nuc with
   `nl >= long_nuc_min`. See §5 for why.

### 2.4 Tags written

Only new tags; v2's `ns/nl/nq/as/al/aq` are passed through byte-identical
(asserted by `--preserve-check` on the first 100 reads).

| tag | type | meaning |
|-----|------|---------|
| `tn` | `B:I` | TF start positions (0-based query coords) |
| `tl` | `B:I` | TF lengths (bp) |
| `ts` | `B:C` | TF score = `round(LLR * 5)` clamped to 0-255 (saturates at LLR=51) |

Downstream filtering uses `--min-ts` (e.g. `75` = LLR ≥ 15 nats).

### 2.5 Modification extraction

Uses v2's pure-Python `parse_mm_tag_query_positions()` rather than
`pysam.read.modified_bases`. The latter **segfaults** on some long/dense
Hia5 reads (e.g. 26 kb read with unusual MM structure); SIGSEGV is not
Python-catchable. The manual parser is a drop-in replacement with no
known crash modes.

IUPAC-encoded DAF BAMs (`Y`/`R` codes) are handled via
`extract_daf_iupac_positions()` which also returns the converted sequence.

---

## 3. Defaults that shipped

| flag | default | notes |
|------|---------|-------|
| `--min-llr` | `5.0` | ~5 nats. Good for DAF; for Hia5 equivalent is ~10. |
| `--auto-threshold` | off | When on, sets `min_llr = 6 * median(llr_miss)`. |
| `--per-read-baseline` | off | Per-read null shift. **Do not use.** See §6. |
| `--min-opps` | `3` | Min target positions in a call. |
| `--boundary-sweep` | `0` | **Disabled** due to artifact, see §5. |
| `--long-nuc-min` | `90` | v2 `nl < 90` treated as short-nuc (TF); `>=90` as real nuc. |
| `--preserve-check` | `100` | Assert v2 tag bytes unchanged on first N reads. |

---

## 4. Benchmark results (Hia5 test BAM, 1228 reads, 3 TSSs)

**Input:** `Release v2.0.0/phase0/data/test_hia5_2-4hr_sna_eve_ftz.bam`
**Model:** `models/hia5_pacbio.json` (k=3 hexamer, mode `pacbio-fiber`)

### 4.1 Recovery (sanity check vs v2's own short-nuc calls)

| bin | recall (with boundary-sweep=0, defaults) |
|-----|---|
| 20-35 bp | 90.8% |
| **35-65 bp (Pol II)** | **98.0%** |
| 65-90 bp | 99.1% |

### 4.2 Overlap metrics at different ts cutoffs

| metric | ts≥0 (full) | ts≥75 | ts≥150 |
|---|---|---|---|
| n_v2_short | 36,649 | 36,649 | 36,649 |
| n_recaller | 44,978 | 17,356 | 10,158 |
| Recall (v2→rc) | **92.5%** | 35.6% | 19.2% |
| Precision (rc→v2) | **74.8%** | 70.3% | 60.6% |
| IoU median (matched) | **0.84** | 0.87 | 0.39 |
| Coverage Pearson (10 bp bins) | **0.89** | 0.84 | 0.78 |
| Coverage Spearman | **0.90** | 0.80 | 0.73 |
| Length Pearson (matched pairs) | 0.21 | -0.13 | -0.31 |

### 4.3 Classification of the 25% unmatched recaller calls (ts=0)

| category | count | pct |
|---|---|---|
| matches v2 short-nuc | 33,644 | 74.8% |
| inside v2 MSP, no short-nuc overlap (true rescue) | 11,292 | **25.1%** |
| inside v2 big nuc (artifact) | 42 | 0.1% |
| unassigned | 0 | 0.0% |

**1.7%** of matched v2 short-nucs are overlapped by ≥2 recaller calls
("v2 merged multiple features" hypothesis) — a minor effect. Matched-
pair length disparity is **median 2 bp** — recaller and v2 agree on
length for 1:1 matches.

**Interpretation**: recaller (a) reproduces v2's short-nuc track with
high position fidelity; (b) adds ~25% net new calls in MSPs that v2's
HMM missed (likely small TFs below the HMM's effective floor); (c) does
not currently rescue TFs over-merged into big nucs — that would require
the boundary-sweep fix in §5.

---

## 5. Known issue: `--boundary-sweep` is disabled

### 5.1 What it was supposed to do

Per the `CLAUDE.md` memory "LLR recaller gotchas": v2 sometimes over-merges
TFs into adjacent big nucs (a 200 bp "nuc" that's really 147 bp nuc + 50 bp
TF). `--boundary-sweep 30` scans the outer 30 bp of each `nl >= 90` nuc
to rescue these.

### 5.2 Why it produced an artifact

Inside a real dense nucleosome, the outer 30 bp are hit-free. Kadane
climbs monotonically with ~+1.67 nats per target position and never
drops. At end-of-interval the flush emits a spurious 30 bp call at every
nucleosome edge. V-plots showed a bright horizontal stripe at ~30 bp
across the entire ±2 kb window. Disabling boundary-sweep eliminated the
stripe completely (189K TFs → 45K TFs on the Hia5 BAM) with **no change
in recall**. Diagnosis credited to Gemini review.

### 5.3 How to safely re-enable (future work)

Modify `call_tfs_in_interval` so that intervals flagged as
"boundary-sweep-derived" do **not** flush at end-of-interval — only emit
on a natural hit-bounded drop of the running LLR inside the sweep
window. Effectively: require a 3-sided-bounded call (climb, peak, drop)
before emission. Pass a per-interval "synthetic boundary" flag to the
scanner. ~20 lines of change.

---

## 6. Other issues and things we tried

### 6.1 `--per-read-baseline` hurts more than helps

Correction shifts `llr_miss` and `llr_hit` by
`log((1-p_model_acc)/(1-p_obs))` and `log(p_model_acc/p_obs)` per read.
Mathematically correct: if a read has lower hit rate than model, misses
are less surprising, LLRs should shrink. But in practice this pushes
marginal true-positive Pol II calls below threshold, dropping 35-65 bp
recovery from 98% → 74% (alone) and → 51% (combined with auto-threshold).
**Not recommended.** Flag exists but should not be used. The underlying
problem it was trying to solve — "Hia5 has too many false positives" —
turned out to be the boundary-sweep artifact, not miscalibrated baselines.

### 6.2 Kadane length bias for Hia5

Hia5's per-hit LLR (-4.34) is ~2.6× the per-miss LLR (+1.67). Even with
Kadane's rogue-hit absorption, a single hit inside a real footprint
almost always resets the run because 1 hit cancels ~2.6 misses of
positive evidence. This pushes recaller calls toward the characteristic
*between-hit* gap length in accessible regions, which at Hia5's ~0.29
miss rate is ~30 bp. Matched-pair lengths agree well (median diff 2 bp)
when v2's short-nucs are also ~30 bp, but longer v2 short-nucs
(60-90 bp) tend to be matched by shorter recaller cores (~30 bp).

**Fix options:** (a) apply a sliding-window average to per-position LLR
*before* Kadane — a 20-30 bp smoother acts as a cheap transition-prior
surrogate; (b) use a proper 2-state HMM inside each scan interval (this
is basically what v2 does; at that point why not just re-tune v2).

Not pursued for this paper. Current behavior documented and calibrated
instead.

### 6.3 `--auto-threshold` marginal

Scales `min_llr` by `6 × median(llr_miss)`: DAF → ~6, Hia5 → ~10.
Produces a small reduction in the 20-35 bp false-positive bin and a
tiny drop in Pol II recall. Included as a flag; not a strong recommendation
one way or the other. Using `--min-ts` post-hoc is equivalent and lets
you keep a single, full-yield BAM.

---

## 7. File inventory

All paths relative to `/Users/tt7739/Dropbox/Fiber-NET-seq/FiberHMM v1.0/v3-caller/`.

### 7.1 Scripts

| file | purpose |
|------|---------|
| `tf_recaller.py` | the recaller (270 LOC) |
| `vplot_recaller.py` | aggregate V-plot (v2 vs recaller, all anchors pooled) |
| `vplot_per_gene.py` | per-gene V-plot with enhancer annotations + bigwig overlays |
| `overlap_metrics.py` | recall/precision/IoU/coverage-correlation analysis |
| `classify_extras.py` | classifies unmatched recaller calls (§4.3) |

### 7.2 Outputs persisted for FiberBrowser / downstream

| file | description |
|------|---|
| `v3caller/analyses/ground_truth_validation/output/hia5_recaller_nosweep.bam` | **Primary output.** Recaller on Hia5 test BAM, boundary-sweep disabled. Carries v2 tags + new tn/tl/ts. |
| `v3caller/analyses/ground_truth_validation/output/hia5_recaller_default_WITH_ARTIFACT.bam` | Kept for A/B reference: same input but with boundary-sweep=30, which produces the ~30 bp V-plot stripe. Useful for documentation / a supplementary figure showing the failure mode. |

Original v2 input:
`/Users/tt7739/Dropbox/Fiber-NET-seq/FiberHMM v1.0/Release v2.0.0/phase0/data/test_hia5_2-4hr_sna_eve_ftz.bam`

### 7.3 Figures

`v3caller/analyses/ground_truth_validation/figures/recaller/`

- `hia5_tss_nosweep_vplot.png` — aggregate V-plot (v2 ≈ recaller)
- `hia5_tss_recaller_vplot.png` — aggregate V-plot with the artifact (for comparison)
- `hia5_tss_recaller_ts75_vplot.png` / `_ts150_vplot.png` — ts-filtered
- `hia5_tss_auto_vplot.png` — auto-threshold variant
- `hia5_tss_perread_vplot.png` — per-read-baseline variant
- `hia5_tss_both_vplot.png` — combined
- `per_gene_nosweep/ts{0,75}/pergene_{sna,eve,ftz}.png` — per-gene, 8 kb upstream + 2 kb downstream, with known enhancer zones (P1/P2/C1/C2 for sna, stripe2/facilitator/late_enh for eve, Z/prox_UPS/R for ftz) annotated and with zld / h3k27ac / dl / PRO-seq overlays
- `per_gene/ts{75,100,150}/pergene_*.png` — per-gene with the boundary-sweep artifact present (shows the uniform 30 bp carpet for comparison)
- `overlap/hia5_ts{0,75,150}_overlap.png` — 6-panel overlap metrics
- `overlap/hia5_nosweep_classify.png` — unmatched-call classification pie + N:1 analysis

---

## 8. How to run

```bash
cd /Users/tt7739/Dropbox/Fiber-NET-seq/FiberHMM\ v1.0/v3-caller

# Recaller
python tf_recaller.py \
  --in-bam input.bam \
  --out-bam output.bam \
  --model models/hia5_pacbio.json

# Sort + index for viewing / fetch-based scripts
samtools sort -@ 4 -o sorted.bam output.bam
samtools index sorted.bam

# Per-gene V-plot with enhancer annotations
python vplot_per_gene.py \
  --in-bam sorted.bam \
  --anchors-bed v3caller/analyses/ground_truth_validation/data/tss_points_sna_eve_ftz.bed \
  --out-dir my_figs/ \
  --upstream 8000 --downstream 2000 \
  --min-ts 0 \
  --bw "zld:/path/zld.bw" \
  --bw "h3k27ac:/path/h3k27ac.bw" \
  --enhancer "P1:sna:-2800:-2300" ...

# Overlap metrics vs v2 short-nucs
python overlap_metrics.py \
  --in-bam sorted.bam \
  --out-prefix my_metrics/hia5

# Classification of unmatched calls
python classify_extras.py \
  --in-bam sorted.bam \
  --out-prefix my_metrics/hia5
```

---

## 8.5 DAF (DddB) tuning — Drosophila spacetime time-course

**Data**: `/Users/tt7739/Dropbox/Fiber-NET-seq/Drosophila_phase2/Datasets/DAF-seq/spacetime/fp/fiberhmm_*.m6a_footprints.bam`
**Model**: `/Users/tt7739/Dropbox/Fiber-NET-seq/Drosophila_phase2/dddb_optimization/dm6_dddb.json` (mode=daf, k=3)

### Strand handling

Each read's MM tag targets either `T` (`T-a,...` = + strand DAF, C→T deamination)
or `A` (`A-a,...` = − strand, G→A). My recaller's daf-mode encoder
reconstructs the template C/G from the deaminated T/A and does C-centered
context lookup (with RC for − strand). No code change needed — already
correct. `detect_daf_strand` picks strand correctly per read.

### Why `min_llr=5.0` was too strict for DAF

DAF uses only ONE strand's target positions. Combined with dddb-daf's
`median(llr_miss) = 0.52` (vs Hia5 pacbio 1.67), each per-position
contribution is ~3× smaller. True Pol II footprints accumulate less
total LLR at the default 5.0 threshold.

### Threshold sweep (10K reads each; WT 2.5-3hr + Dl- 2.5-3.5hr)

| min_llr | Pol II recall | WT TSS density (TFs/kb/read) | Dl- sna gene-body density | **SNR** |
|---|---|---|---|---|
| 2.0 | 96.9% | 4.44 | 0.50 | 8.94× |
| 3.0 | 94.2% | 3.49 | 0.37 | 9.40× |
| **4.0** | **89.5%** | **2.83** | **0.28** | **10.12×** ← best |
| 5.0 | 80.3% | 1.99 | 0.22 | 8.95× |

Specificity metric: `(WT TSS TFs/1kb/read) / (Dl- sna gene-body TFs/1kb/read)`.
TSS window = chr2L:15478160-15478360 (Pol II); gene body = chr2L:15473260-15478260.

### Full-BAM results at min_llr=4.0

| BAM | reads | TFs emitted | Pol II recovery |
|---|---|---|---|
| WT 2.5-3hr | 58,843 | 127,380 | 88.5% |
| Dl- 2.5-3.5hr | 19,823 | 99,998 | 90.4% |

### Per-gene plots

`v3caller/analyses/ground_truth_validation/figures/recaller/daf_per_gene/`
- `wt_2.5-3/pergene_{sna,eve,ftz}.png` — all three genes
- `dl_2.5-3.5/pergene_sna.png` — sna only (Dl- BAM targets chr2L)

Biology visible: WT sna has clear Pol II at TSS plus strong recaller
hits in the P1/P2 enhancer zones (co-localizing with H3K27ac peaks).
Dl- sna shows **dramatically reduced P1/P2 enhancer signal** — exactly
the dorsal-loss-of-function signature.

### Recommendation

Use `--min-llr 4.0` for all DddB DAF runs. Hia5 stays at default 5.0.
Saved as memory `project_recaller_daf_tuning`.

---

## 8.6 DddA tuning — using dddb model with emission uplift

**Data**: `/Users/tt7739/Dropbox/Fiber-NET-seq/FiberHMM v1.0/Release v2.0.0/ddda_nuc_output/{napa,uba1,ps01499}.bam` (IUPAC-encoded, ~500k reads each; no dedicated DddA model).

**Model**: reused `dm6_dddb.json` with a new `--emission-uplift` flag because DddA is ~3× higher efficiency than DddB. Without uplift, the dddb emissions underestimate DddA's per-position signal and the recaller misses real footprints.

### The `--emission-uplift` transform

Per context c, with `p_hit_acc(c) = P(hit | c, accessible)` from the model:
```
p_hit_acc_new(c)  = 1 − (1 − p_hit_acc(c))^uplift
p_hit_prot_new(c) =      p_hit_prot(c)^uplift
```
Then rebuild per-context `llr_miss` and `llr_hit` tables from the
transformed probabilities. `uplift = 1.0` is identity; `uplift = 2-3`
moves the accessible state's P(hit) toward 1 and protected's toward 0.

### 2-way sweep (uba1.bam, 5k-10k reads; dddb model)

| uplift | min_llr | Pol II recovery | TFs/read |
|---|---|---|---|
| **1.0** | 5.0 | **76.6% ⚠️** | 2.2 |
| 2.0 | 5.0 | 99.1% | 5.8 |
| 2.0 | 7.0 | 96.5% | 4.3 |
| 2.5 | 7.0 | 99.1% | 5.3 |
| 3.0 | 10.0 | 98.2% | 4.5 |

Uplift=1 (plain dddb emissions) drops Pol II recall to 77%. Uplift≥2 recovers full signal and per-amplicon v-plots show specific peaks (not saturated carpets). Heatmap figure: `v3caller/analyses/ground_truth_validation/figures/recaller/ddda_sweep_heatmap.png`.

### Recommendation

For DddA: `--emission-uplift 2.0 --min-llr 5.0`. Per-amplicon v-plots at the top 4 settings are in `v3caller/analyses/ground_truth_validation/figures/recaller/ddda_sweep/`.

---

## 9. Ideas / next steps

### 9.1 For this manuscript

- [ ] Run on the 7 DAF-seq time-course BAMs at
  `/Users/tt7739/Dropbox/Fiber-NET-seq/Drosophila_phase2/DAF-seq/spacetime/fp/*.bam`
  with `models/ddda_pacbio.json`.
- [ ] Spot-check 5-10 "MSP-rescue" calls (the 25% unmatched) on IGV to
  confirm they look like real footprints.
- [ ] Generate per-gene v-plots for hb, zen, tll, chrb (in `CLAUDE.md` of
  `pioneer_analysis`) — same recipe, different BEDs.
- [ ] Methods text: "Log-likelihood-ratio test applied per position
  within v2-called MSPs, using v2's k=3 hexamer emission table as the
  null. TF footprints emitted as Kadane-maximum intervals with LLR ≥ X
  nats and ≥ Y target positions. Pass-through v2 tags; new tags tn/tl/ts
  carry TF position, length, and 0-255 scaled LLR score."

### 9.2 Safely re-enable boundary-sweep

- [ ] Mark sweep-derived intervals with a flag.
- [ ] In `call_tfs_in_interval`, skip end-of-interval flush for those
  intervals. Only emit if `running` drops to ≤ 0 within the interval
  (natural hit-bounded termination).
- [ ] Re-run benchmark; expected effect: recover additional ~2-5% TFs
  that v2 over-merged into big nucs without reintroducing the 30 bp
  stripe.

### 9.3 Length-bias fix (optional, for Hia5 specifically)

- [ ] Apply a 20-30 bp sliding-window mean to per-position LLR before
  Kadane. Cheap transition-prior surrogate; should let longer footprints
  survive a stray hit.
- [ ] Compare matched-pair length Pearson before/after. Current 0.21 →
  expected 0.5-0.7 if the smoothing works. Doesn't really change
  biological conclusions — just makes recaller lengths match v2's for
  figure consistency.

### 9.4 Deferred (future paper)

- 1D CNN / U-Net trained on (opp, hit, ctx) triples with v2 output as
  weak labels. Gemini framed this as the "DeepVariant moment" — hand-
  engineered scoring beaten by letting the net learn the spatial prior.
  6-month project, explicitly not this manuscript.

---

## 10. Collaborator credits

- **Gemini** flagged the boundary-sweep artifact by inspection of the
  V-plot images alone ("the stripe of death"). Correct diagnosis on
  first try; fix confirmed within one run.
- **Gemini** also proposed the Kadane form over strict hit-termination
  (§2.2) and vetoed the permutation-null approach (which would have
  decoupled hits from sequence context and invalidated the context-
  aware math). Both calls held up.
