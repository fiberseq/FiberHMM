# Consensus recaller production validation — 2026-07-12

> **Superseded by the 2026-07-14 calibration audit.** Every proposal count and
> operating point below was produced by a model with five defects that have
> since been fixed. The counts are retained as a historical record; do not cite
> them. Regenerated panels and BAMs are in `consensus_validation_outputs/`
> `regenerated_20260714/` and `consensus_visualization_outputs/regenerated_20260714/`.
> See [`CHANGES_2026_07_14.md`](./CHANGES_2026_07_14.md) for what changed,
> what it cost, and which claims here no longer hold. In short:
>
> - `sr` scored every molecule at one global accessible hit rate, which sits at
>   the 91st percentile of the real per-molecule distribution. Only 15% of the
>   strong DddB rescues below survive per-molecule calibration.
> - `cr` gave N a vague uniform geometry prior while TF got a fitted empirical
>   one, worth ~2.9 nats (~19x) toward splitting on a block with no evidence.
>   The "10:1 N:TF" baseline below was really operating at ~1.9:1 *toward* TF.
> - The site-local Wilson floor combined as `max(global, wilson)`, so it
>   swallowed the prior sweep: the 10:1 and 100:1 scenarios below are
>   bit-identical at most sites. The conservative end did not exist.
> - Bridged TF complexes (protected separator) predict the same bases as N and
>   cannot be refuted, yet their mass licensed strong splits.
> - The ±25/±50 bp decoy panel cannot detect any of these: it compares TF
>   against displaced TF, so a shared bias cancels in both arms. There was no
>   true-nucleosome negative control anywhere in this document.
>
> Regenerated: 48 strong calls across the four boundaries become **13**, and all
> 13 now rest on genuinely accessible separators.

> This document freezes the conservative selected-state v2 validation. The
> subsequent aggressive paired-output implementation is specified in
> [`PAIRED_RECALL_PLAN.md`](./PAIRED_RECALL_PLAN.md) and
> [`VISUALIZATION.md`](./VISUALIZATION.md). Its v3 BAMs intentionally retain
> linked N/TF alternatives and therefore use pair-integrity audits rather than
> the zero-overlap invariant reported below.
> The completed focal paired panel is reported in
> [`PAIRED_VALIDATION.md`](./PAIRED_VALIDATION.md).

This document freezes the validated, **report-authoritative** operating point.
The caller reads standard aligned sequence, hard `MM`/`ML`, and existing `MA`
annotations and does not rewrite BAMs. A separate visualization command writes
complete shadow callsets only to new regional derivative BAMs. The installed commands
are:

```bash
fiberhmm-consensus-recall --help
fiberhmm-consensus-validate --help
fiberhmm-consensus-annotate --help
```

JSON and optional TSV proposal outputs are deterministic and contain model and
input-file provenance. JSON/TSV writes are atomic. A strong proposal is suitable
for downstream review. `fiberhmm-consensus-annotate` can expose strong and
review proposals as posterior-scored `nuc_cr`/`tf_cr` and `nuc_sr`/`tf_sr` MA
shadow callsets without changing the original `nuc`/`tf` calls; authoritative call
replacement remains out of scope.

## Frozen hierarchy and operating points

- DddA supplies the highest-resolution fine-TF geometry.
- PacBio Hia5 supplies authoritative broad-nucleosome geometry and a
  high-specificity, lower-resolution fine-TF anchor. Both A and T channels are
  read from every HiFi molecule; alignment orientation is never treated as a
  biochemical strand.
- DddB and Nanopore Hia5 can nominate/support focal sites but cannot acquire
  boundary authority through depth.
- Source-strand rescue requires explicit high-TQ support, source-strand local
  enrichment of at least 1.5, positive target-molecule likelihood, and
  posterior at least 0.95. Posterior 0.5–0.95 is review-only. `N` always remains
  in the source mixture.
- Composite deconvolution tests only current 90–220 bp nuc calls. The global
  baseline is 10:1 N:TF, but at a focal site the TF-complex prior may rise to
  the same-cohort 95% Wilson lower bound on explicit configurations among
  spanning molecules. Uncalled molecules stay in the denominator, short reads
  that do not span the template cannot vote, and the candidate molecule is
  held out of its own occupancy and configuration geometry. Repeated input
  BAMs are an explicitly pooled cohort; BAM identity never partitions the
  prior. Independent libraries and assays are validation-only.
- The decision is hierarchical: stage 1 compares canonical N with the sum over
  all supported TF configurations; stage 2 requires at least 0.8 conditional
  posterior before emitting an exact split. Otherwise the block is demoted to
  one unresolved TF-complex envelope. Strong status requires aggregate
  posterior at least 0.95 and a true-geometry advantage of at least 3 log units
  over the best ±25/±50 bp boundary decoy.
- Amplified DAF is collapsed by the same full-read hard-deamination fingerprint
  logic and Jaccard 0.95 default as `fiberhmm-dedup`. Collapse is performed
  separately within each input BAM/timepoint before the surviving molecules
  enter the pooled inference cohort.
- Nanopore Hia5 uses hard `ML >= 248`; no read is required to span a locus.

## Multi-control cross-assay panel

The panel uses full GM12878 PacBio, both targeted GM12878 DddA amplicons, all
five fly 2–4 hr PacBio BAMs, and all nine targeted fly DddB islands. scDAF is
excluded; the two perturbed fly Nanopore BAMs are exploratory and never vote.
This entire panel evaluates frozen, self-contained calls. No independent
library or assay supplies a focal prior, TF configuration, boundary, or call
to `fiberhmm-consensus-recall`.

- 827 fine-TF candidates were evaluated: 134 GM12878 and 693 fly.
- Five non-overlapping, opportunity-matched local controls were requested per
  candidate. All 693 fly and all 51 NAPA candidates received five; UBA1 received
  406/415 possible controls, with 80/83 candidates receiving all five.
- Reused genomic controls were collapsed into 25-bp bins before calibration,
  leaving 1,663 PacBio, 1,514 DddB, and 120 DddA pseudo-site null observations.
- Source-selected assays never receive formal p/q values. Independent evidence
  is calibrated on other loci of the same cohort/chemistry and BH-corrected
  within assay family.
- The conservative classification is 12 strong, 799 review, 1 reject, and 15
  untestable. “Review” includes authoritative single-assay sites and assay
  disagreement; it does not mean absent.
- Of ten DddB-only discovery seeds, three have strong independent PacBio
  confirmation (`ind.tf016`, `ind.tf077`, `zen.tf077`), three are nonfocal
  review sites, and four lack a calibrated anchor/control comparison.
- Fine-coordinate DddB deltas are close to their local-null distribution after
  control matching. This quantitatively supports the intended rule that DddB
  depth may nominate a site but cannot define its boundary.

## PacBio composite nuc deconvolution

Nhomie, Homie, SF1, and SF2 were run with five pooled PacBio BAMs, at least 50
high-TQ calls per site, at least 20 source configurations, automatic
within-cohort leave-one-molecule-out edge-bandwidth selection, and the
hierarchical focal-prior rule above.

| Boundary | Tested 90–220 bp calls | MAP TF complex | Review or strong | Strong | Unresolved MAP complexes | Edge bandwidth |
|---|---:|---:|---:|---:|---:|---:|
| Nhomie | 198 | 58 | 57 | 15 | 6 | 3 bp |
| Homie | 574 | 59 | 59 | 31 | 0 | 3 bp |
| SF1 | 1,590 | 95 | 93 | 41 | 0 | 5 bp |
| SF2 | 1,616 | 157 | 153 | 52 | 52 | 5 bp |
| **Total** | **3,978** | **369** | **362** | **139** | **58** | — |

The four ±25/±50 bp libraries produced **4 posterior-strict calls across 15,912
decoy scores**, versus 143 at the true geometries; 61 decoys reached review
posterior versus 369 true-site MAP complexes. The explicit best-decoy gate
retains 139 strong true-site calls. Blocks above 220 bp were counted and skipped (1,886 total),
never silently interpreted or split.

At Homie, the main class remains the paired-site configuration; 32 calls pass
the strict posterior before the decoy gate and 31 remain strong. At SF2, where
exact layouts genuinely compete, 52 MAP demotions
are intentionally broad unresolved complexes rather than forced splits.

The visualization integrity audit is exact. Across the four combined BAMs,
108,436/93,733/115,143/124,295 original nucleosomes survive in `nuc_cr`; the
59/58/95/157 selected blocks are absent and their 115/110/197/275 replacement
intervals are present in `tf_cr`. Every tested source N matched, and there are
zero exact or partial `nuc_cr`/`tf_cr` overlaps.

## Strand rescue

The source prior is a fitted `A/TF/N` mixture on the opposite physical/read
strand. A current target MSP can become TF only when its own hard calls provide
positive evidence; the prior cannot reverse a nonpositive molecule-level BF.
Forced geometry is allowed, but source support and source focal enrichment are
always recomputed from the input BAM.

| Chemistry / site | Source high-TQ calls | Source focal enrichment | Strong rescues | Review |
|---|---:|---:|---:|---:|
| fly DddB `ind` site 1, GA→CT | 386 | 19.97 | 309 | 166 |
| fly DddB `ind` site 2, CT→GA | 333 | 14.61 | 420 | 108 |
| GM UBA1 DddA, GA→CT | 340 | 13.56 | 111 | 23 |
| GM NAPA DddA, CT→GA | 610 | 3.59 | 0 | 89 |
| fly Nanopore site 1, REV→FWD | 5 | 4.16 | 0 | 27 |
| fly Nanopore site 2, REV→FWD | 7 | 6.58 | 0 | 42 |

The DddB figures use 21,663 deduplicated molecules pooled from five timepoints
plus the independent `yw_2-4` pool. The 22,057 raw reads are fingerprint-
collapsed separately within each input BAM, never across timepoints. This
separation is desirable: UBA1 and the high-depth DddB examples contain
actionable weak-strand molecules, whereas the sequence-limited NAPA and weak,
short Nanopore examples remain review-only. Source-prior-preserving coordinate
shifts are reported as a counterfactual stress test, not an FDR null: applying
a known true-site occupancy prior to a false coordinate is deliberately the
wrong generative model. Source-site focality is instead calibrated by the
multi-control site-existence panel above.

## Molecule-collapse sensitivity

At Jaccard 0.90 relative to the 0.95 default, inferred molecule counts change
by −2.7% (NAPA), −1.7% (UBA1), and −1.2% (pooled fly DddB). At the
deliberately strict 0.98 setting they change by +28.5%, +11.2%, and +1.3%,
respectively. Reports
therefore retain the threshold and raw/family counts, and the dedicated
`dedup-sensitivity` command makes DddA threshold dependence explicit.

Proposal counts at Jaccard 0.90/0.95/0.98 are:

- NAPA DddA: 0/0/0 strong and 80/89/108 review;
- UBA1 DddA: 111/111/128 strong and 22/23/25 review; and
- pooled fly DddB pair: 711/729/754 strong and 269/274/286 review.

The actionable classification is unchanged at 0.90 versus 0.95 and changes
gradually at the deliberately strict 0.98 setting. The 0.95 setting remains
aligned with the production `fiberhmm-dedup` default and its observed
0.90–0.95 similarity gap.

## Remaining boundaries of the claim

- Cross-assay occupancy *frequency* is not calibrated; mixture weights are
  diagnostic and must not be presented as assay-equivalent occupancy.
- The available Nanopore panel is weak, short, and perturbed. It validates safe
  review behavior, not sensitivity.
- Full-depth DddA radial composite deconvolution is not yet production-speed:
  two initial amplicon runs spent more than ten minutes in geometry
  marginalization and were stopped. Reusing the N marginal across boundary
  decoys removes a fivefold redundancy, but a subsequent full UBA1 run still
  did not finish within several minutes. DddA cross-strand rescue is validated
  separately; the DddA composite pass should be skipped until the remaining
  radial likelihood is vectorized.
- Coordinate controls can land on real regulatory footprints. They are retained
  individually rather than converted into a simplistic universal background.
- No strong result grants a low-resolution assay geometric authority.
- The inference command changes no BAM tags. The optional annotator writes new,
  indexed regional BAMs, preserves ordinary `nuc`/`tf`, and applies decisions
  only to complete shadow callsets; it is not an in-place or canonical call
  replacement stage.

## Release checks

- Consensus tests: 73 passed.
- Full FiberHMM regression: 559 passed, 3 skipped, 26 benchmark tests
  deselected.
- A clean wheel (SHA-256
  `131aab3a88abf6ece7c474ca3c5f31867b5774b42fd5515f9d84b248dea84531`)
  includes all three console scripts, hierarchy/policy JSON, and bundled
  models; all installed entry points completed smoke tests outside the source
  tree.
- From outside the source tree, the installed validator resolved and audited
  the full external manifest with zero errors, including its
  repository-relative DddA inputs.
- Two independent runs of the corrected Nhomie command produced byte-identical
  JSON (SHA-256 `2b29b931f4e88295ca9028ed94c724256a9ea64fb9f4ea7d5b074d0938d50b65`)
  and TSV (`e653d713583e22cac4fc911fb4d65b1d49115e6e50e658fcdb2379f2b092221c`).
- Thirty-six self-contained regional/combined shadow-callset BAMs were indexed
  and quick-checked for pooled PacBio, DddA, DddB, and short threshold-248
  Nanopore data; every retained proposal matched its report input record. All
  36 passed a complete cross-layer interval-overlap audit across 103,693
  records.
- No merge implementation file was modified.
