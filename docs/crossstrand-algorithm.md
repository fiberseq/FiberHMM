# Cross-strand DAF consensus: algorithm & accuracy

Reference for the `fiberhmm-crossstrand` pipeline (pair → merge → re-call).
Companion to [`crossstrand-fiberbrowser.md`](crossstrand-fiberbrowser.md), which
covers the output tags and browser tracks.

- [Biology & goal](#biology--goal)
- [Pipeline overview](#pipeline-overview)
- [Step 1 — pairing](#step-1--pairing)
  - [Signal A: sequence at deamination-safe A/T sites](#signal-a-sequence)
  - [Signal B: nucleosome footprint overlap](#signal-b-footprint-overlap)
  - [Assignment order & guards](#assignment-order--guards)
- [Step 2 — merge](#step-2--merge)
- [Step 3 — re-call](#step-3--re-call)
- [Accuracy](#accuracy)
- [Parameters](#parameters)

---

## Biology & goal

DddA is a double-strand cytosine deaminase: it converts C→U on **both** strands
of a duplex. After denaturation the two strands are amplified and sequenced
independently, so one molecule appears as two reads of opposite **flavor**:

- **CT** read — C→T conversions (deaminated cytosines on the top strand). Only
  reference **C** positions are informative.
- **GA** read — G→A conversions (top-strand view of bottom-strand C→U). Only
  reference **G** positions are informative.

A footprint (nucleosome, TF, Pol II) is read out as *protection* — a run of
target bases that were **not** deaminated because a protein blocked the enzyme.
Because each strand only reports at its own base (C for CT, G for GA), a single
strand is blind wherever its base is sparse. Pairing the two strands and merging
them makes **both C and G informative** in the overlap (~2× the footprinting
density), which is the whole point of the method.

The obstacle: the two strands of one molecule **cannot be matched by their
deamination pattern** (they mark disjoint bases). We match them another way.

## Pipeline overview

```
fiberhmm-crossstrand -i calls.bam -o consensus.bam -r hg38.fa
    │
    ├─ 1. fiberhmm-pair   pair each CT read with its GA strand-mate
    ├─ 2. fiberhmm-merge  build one both-strand consensus read per pair
    └─ 3. --recall        re-call footprints on the consensus (both strands)
```

Input is a footprint-called DAF BAM (from `fiberhmm-call --enzyme ddda`),
coordinate-sorted + indexed, deaminations encoded as IUPAC R/Y in the sequence.

---

## Step 1 — pairing

Two independent signals separate the true strand-mate from a wrong homolog.
Pairing is done **within each chromosome** and **within local overlap
components** — never as a global chromosome-wide matching.

### Signal A: sequence

At reference **A/T** positions DddA does nothing (it only acts on C/G), so the
query base there is the untouched genotype. For a heterozygous sample (HG002)
those positions carry **het SNPs that distinguish the two homologs**. The two
reads of one molecule are genotype-identical at shared A/T sites; reads from
different homologs differ at het SNPs.

- `_sequence_signature(read, reference)` returns the (ref-position, query-base)
  pairs at ref A/T sites only. Reference C/G sites are dropped (DddA-altered);
  IUPAC Y/R in the query are canonicalized to C/G so genuine alternate alleles
  still count.
- `score_sequence(a, b)` intersects the two reads' A/T sites and returns
  `(shared_bases, mismatches, rate)`. Same molecule → rate ≈ 0
  (sequencing/consensus error, ~0.1%); different homolog → het-SNP-scale rate.

This is essentially ground truth **where a distinguishing het SNP falls in the
overlap**. Its limitation is recall, not precision: many overlaps contain no
het SNP (the homologs are locally identical), and a lone 1+1 locus has no
competing edge to be reciprocal against.

### Signal B: footprint overlap

The nucleosome array is a physical property of the duplex, shared by both
strands, so two strand-mates have correlated dyad positions. This is the
original cross-strand signal and the fallback when sequence is uninformative.

**Per-read signal.** Each read's MA `nuc` dyad centers are mapped to reference
coordinates and rasterized into a **Gaussian dyad-density track** on a fixed
reference grid:

- grid resolution `grid_bp = 10` bp;
- each dyad contributes a Gaussian bump of width `sigma_bp = 30` bp (absorbs the
  cross-strand edge variance — the two strands place a dyad at slightly
  different positions).

So read *r* becomes a 1-D signal `s_r(x)` over its reference span.

**Pair score — `score_pair(a, b)`** (normalized, lag-tolerant cross-correlation
over the genomic overlap):

1. Overlap window `[lo, hi) = [max(start_a, start_b), min(end_a, end_b))`.
   Require `hi − lo ≥ min_overlap_bp` (1500) and at least `min_nucs` (4) dyads
   from **each** read inside the window (few-nucleosome overlaps are too noisy —
   see accuracy).
2. Slice both density tracks to the window and mean-center them:
   `x = s_a − mean`, `y = s_b − mean`.
3. Compute the normalized cross-correlation at each integer lag in
   `±max_lag_bp` (±60 bp, i.e. ±6 grid steps — a whole-array phase offset must
   not be penalized):

   ```
   score = max over lag of   Σ x[i]·y[i+lag]  /  (‖x‖ · ‖y‖)
   ```

   Returns a value in `[-1, 1]`. Same molecule → high (median ~0.5–0.6 at good
   overlap); different homolog → low (~0.1–0.2).

**Assignment — reciprocal-best with a null-floor margin gate.** For every
genomically overlapping opposite-flavor pair, `score_pair` is evaluated (unless
sequence-vetoed, below). For each read we track its best and second-best
partner. A pair `(i, j)` is accepted iff:

- **reciprocal**: `j` is `i`'s best partner *and* `i` is `j`'s best partner;
- **floor**: `score ≥ min_score` (0.25);
- **margin**: `min( score − comp_i , score − comp_j ) ≥ min_margin` (0.05),
  where the *competitor* `comp = max(second_best, null_floor)`.

The `null_floor` (0.24 ≈ the empirical wrong-pair p90) is a **virtual
competitor**. At a 2×2 locus the real second-best dominates the margin (relative
gate); at a lone 1+1 locus (no second-best) the null_floor dominates, so a lone
pair must beat the wrong-pair null to merge. This is why the absolute floor is
deliberately low — the *relative* comparison does the precision work.

**Why not a global matching.** Reads tile the genome in staggered overlaps; a
chromosome-wide optimal matching would let one weak local edge force a long
chain of downstream assignments. We only ever resolve reciprocal-best edges and
complete local 2×2 loci, so a mistake stays local.

### Assignment order & guards

`assign_pairs` runs the two signals in this order:

1. **Sequence, reciprocal-best** — mutually-lowest-mismatch CT↔GA edge, with a
   two-sided rate margin (`min_sequence_margin`) and a rate cap
   (`max_sequence_pair_rate`). Requires ≥2 candidates (competition).
2. **Sequence, constrained 2×2** — at a complete 2-CT/2-GA component with all
   four edges sequence-comparable, pick the better diagonal, **but only commit
   when a rejected edge is grossly discordant** (`> min_component_discordance_rate`,
   i.e. a clear het-SNP conflict rules one diagonal out). Ordinary SNP-scale
   noise is left to footprints — residual consensus errors make it unreliable.
3. **Footprint fallback** on everything sequence didn't resolve — the reciprocal
   best + null-floor gate above, **with a sequence veto**: a candidate whose
   genotype mismatch rate exceeds `max_sequence_mismatch_rate` (0.2%) is removed
   before scoring, no matter how well footprints correlate.

Every accepted read is tagged with its `method` (`S` sequence / `F` footprint),
mate, scores, and (for sequence) the shared-base / mismatch counts — see the
[tag reference](crossstrand-fiberbrowser.md).

---

## Step 2 — merge

For each resolved pair, `fiberhmm-merge` builds one consensus read spanning the
**union** of the two reference spans:

- reference-frame, all-`M` CIGAR (deaminations are substitutions; small source
  indels are dropped);
- deaminations re-encoded as IUPAC **Y** (C→T) and **R** (G→A) — one read now
  carries *both*;
- the strand-coverage regime is written spec-natively into `MA` as a custom
  `deam` type: `deam+` (CT-read coverage) and `deam-` (GA-read coverage).
  **`deam+ ∩ deam-` is the both-strand region.**

The two source reads are replaced by their consensus; all other reads pass
through unchanged.

## Step 3 — re-call

`--recall` runs the **full canonical FiberHMM stack** — HMM apply + nucleosome
recaller (radial split, edge refine, NRL-196 phasing) + TF/Pol II LLR recaller —
on a **both-strand observation**, reusing `build_fused_recall_result` unchanged.
The observation is built by merging two single-strand encodings:

- a `+` pass (target = C, C-centered context codes) masked to `deam+`;
- a `−` pass (target = G, reverse-complemented into the same C-centered codes)
  masked to `deam−`.

Reference C and G positions are disjoint, so the two never collide. In the
both-strand core both are informative (double density); in a single-strand flank
the absent strand's bases are correctly left non-target (absence of data, not
protection). Because both the HMM emission table and the LLR tables index the
same C-centered code space, **no recall math is special-cased** — the both-strand
structure is carried entirely by the observation encoding.

---

## Accuracy

**Pairing (SRR33130342, chr1:1–40 Mb, 1,391 pairs):**

| path | share | evidence |
|---|---|---|
| sequence (`S`) | 7% | median **0** mismatches over ~1,450 A/T bases; 59% exactly 0 → genotype identity, ≈ground truth |
| footprint (`F`) | 93% | corr median 0.58 (null ≈ 0.1–0.2); 99% sequence-comparable, **100% pass the ≤0.2% genotype veto** |

Sequence directly resolves only the minority with a distinguishing het SNP *and*
reciprocal competition, but its **veto guards all footprint pairs** — a
wrong-homolog footprint pair with a het SNP in the overlap is rejected. Where no
het SNP exists, footprints decide (~90% correct at ambiguous 2×2 loci); there the
homologs are locally sequence-identical, so an error mixes chromatin, not
genotype.

**Footprint score depends on overlap length** (2×2 ground truth, true vs null):

| overlap | true corr (med) | null corr (med, p95) | separation |
|---|---|---|---|
| 1–2 kb | 0.454 | 0.271 (p95 0.62) | 0.184 |
| 10–25 kb | 0.351 | 0.103 (p95 0.23) | 0.247 |
| >25 kb | 0.335 | 0.070 (p95 0.16) | 0.265 |

Absolute correlation drops with overlap (more nucleosomes → regress to the honest
value), but the true/null **separation grows** and the null tail collapses — long
overlaps are the reliable calls; **short overlaps are the risk** (inflated,
few-nucleosome variance). The gate is therefore low-floor + relative-margin, not
a high absolute threshold.

**Re-call:** the both-strand emission is verified correct on real reads (0
strand-swap disagreements, 0 C/G overlap, all deaminations captured, LLR 100%
finite). The caller is length-stable (nq/tq and call densities flat from 1 kb to
>100 kb).

**Dataset-level (12 HG002 cells):** 2-strand genome coverage 5.4–20.9% per cell;
~12% of fibers become both-strand consensus; ~4–5% gain length (typ. +1–2 kb).
The product is footprint-calling *density*, not read length.

## Strand asymmetry — why both strands matter

A CT read reports footprints only at reference **C** positions; a GA read only at
reference **G** positions. So a footprint in a base-skewed window is resolvable
on essentially one strand. Re-calling 800 consensus reads three ways
(both strands / CT-only / GA-only) and binning each both-strand TF call by the
C/G composition of its window:

| window sequence | CT-only resolves | GA-only resolves |
|---|---|---|
| strong G-rich | **24%** | 55% |
| mild G | 35% | 44% |
| balanced | 40% | 39% |
| mild C | 45% | 35% |
| strong C-rich | 53% | **25%** |

The gradient is monotonic and matches the mechanism exactly: in G-rich (C-poor)
windows the CT strand is largely blind; in C-rich windows the GA strand is.
Both-strand recovers the union — **~2× the TF-call yield of either single
strand**, with roughly half of all calls resolved by only one strand.

**TF families this matters for.** G-rich (reference) motifs are resolved mainly on
the **GA** strand and missed by CT: the GC-box / CpG-island promoter factors —
**SP1/SP3, KLF, EGR1/ZIF268, WT1, MAZ, E2F, NRF1, ZBTB, and the G-rich core of
CTCF**. Their C-rich reverse-complement contexts resolve on **CT**. Both-strand
is specifically valuable at these loci. Conversely, **A/T-rich** motifs
(**TATA/TBP, homeobox, FOX, GATA, SOX/HMG**) have few C *and* few G, so neither
strand resolves them well and both-strand does not help — a real limitation.

**Caveat.** The ~2× yield increase tracks the ~2× informative-position density,
but without an orthogonal truth set we cannot yet certify every extra call as a
real footprint rather than a threshold-crossing marginal. The *gradient* above is
robust (it is a within-call strand comparison); the *magnitude* needs a truth set
(motif/ChIP occupancy, ATAC, or concordance of the two strands' independent
calls). That benchmark is the headline validation still to run.

## Parameters

| param | default | role |
|---|---|---|
| `min_overlap_bp` | 1500 | min genomic overlap to score a footprint pair |
| `min_nucs` | 4 | min dyads per read in the overlap |
| `grid_bp` / `sigma_bp` | 10 / 30 | dyad-density resolution / Gaussian width |
| `max_lag_bp` | 60 | ± register shift searched in the correlation |
| `min_score` | 0.25 | absolute correlation floor |
| `min_margin` | 0.05 | best − competitor, both reads |
| `null_floor` | 0.24 | virtual competitor for lone (1+1) pairs (≈ data null p90) |
| `min_sequence_bases` | 500 | min shared A/T sites to use sequence |
| `max_sequence_mismatch_rate` | 0.002 | footprint-pair genotype veto |
| `max_sequence_pair_rate` | 0.01 | max rate on a sequence-selected pair |
| `min_component_discordance_rate` | 0.02 | rejected-edge conflict needed to commit a 2×2 |
| `min_sequence_margin` | 0.001 | min rate advantage for a sequence choice |
