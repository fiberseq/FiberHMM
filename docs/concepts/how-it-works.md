# How FiberHMM works

Single-molecule chromatin assays mark accessible DNA: Hia5 methylates
adenines (m6A), DddA and DddB deaminate cytosines (read out as C→T or G→A).
Protected DNA (under a nucleosome or a bound protein) is marked much less. A
footprint caller has to find the protected stretches of each molecule from
the pattern of marked and unmarked target bases.

The difficulty is that the enzymes do not mark every accessible base equally.
Their efficiency depends strongly on the surrounding sequence: in naked DNA
some contexts are marked most of the time and others rarely. A run of
unmarked bases in a poor context says little; the same run in a good context
is strong evidence of protection. FiberHMM models this explicitly, which is
what makes footprints the size of a transcription factor (10–30 bp) callable,
not just nucleosomes.

`fiberhmm-call` runs four steps on every read:

1. encode each target base with its sequence context and whether it was
   marked;
2. decode protected and accessible stretches with a two-state HMM;
3. re-examine nucleosome-sized footprints (split over-merged ones, refine
   edges);
4. call TF-sized footprints inside the accessible stretches with a
   likelihood-ratio recaller.

## 1. Context-specific emissions

For every target base, FiberHMM takes the *k* bases on each side from the
read's own sequence (no reference or context files are needed) and turns the
(2*k*+1)-mer into an integer code. With the bundled *k* = 3 (7-mers) there are
4^6 = 4,096 contexts. Each context has two observation symbols, marked and
unmarked; bases that are not targets have their own symbols.

What counts as a target depends on the observation mode:

| Mode | Target bases | Marked means |
|---|---|---|
| `pacbio-fiber` | A on the read strand and T (A on the opposite strand) | m6A call at ML ≥ threshold |
| `nanopore-fiber` | A on the sequenced strand | m6A call at ML ≥ threshold |
| `daf` | C on C→T (CT) reads, G on G→A (GA) reads | deaminated |

The emission table stores, for each context, the probability of a mark in
the protected state and in the accessible state. It is estimated from control
data: an accessible control (naked DNA treated with the enzyme) and an
inaccessible control, one table per context size (see
[Training a model](../workflows/training.md)).

Two DAF-specific adjustments change which observations enter the model:

- **Adjacent-target thinning (keep-one).** Adjacent targets on the deaminated
  strand (CC on CT reads, GG on GA reads) do not convert independently. With
  `--daf-mask-runs N`, runs of *N* or more targets keep only their 5'-most
  target (`--daf-run-policy keep-one`) or are dropped (`drop`). The default
  is `N = 2`, keep-one, for DddA and off for DddB; the setting applies to the
  HMM, both recallers, consensus lattices and the duplex joint recall, and is
  recorded in `@PG` (`daf_run_mask=>=2/keep-one`). On 15,557 HG002 scDAF
  duplexes, keep-one raised the precision of DddA TF calls, read on the
  complementary strand of the same molecule, in every stratum (for example
  0.95 → 0.98 for calls containing runs) at the cost of 16–27% fewer calls.
  DddB has not been validated.
- **CpG-aware recall (DddA).** 5-methylcytosine slows DddA, so a methylated
  CpG looks protected. For DddA, CpG observations are excluded from both
  recallers except inside islands called confidently unmethylated
  (`MA:ddda_ucg`, from [`fiberhmm-tag-m5c`](../workflows/daf-seq.md#ddda-cpg-island-methylation)).
  Excluded sites contribute no likelihood, do not count toward minimum
  evidence and cannot define a boundary.

## 2. The HMM

The HMM has two states, protected and accessible, with the context-specific
emissions above and learned transition probabilities. Viterbi decoding gives
the most likely state path; each run of the protected state is a footprint.

- Footprints of at least `--nuc-min-size` (85 bp) are nucleosome-sized. The
  accessible stretches between them are the **MSPs** (methylase-sensitive
  patches); smaller footprints do not split an MSP. `--msp-min-size` (0)
  drops shorter MSPs.
- Footprints shorter than `--unify-threshold` (90 bp) are candidates for the
  TF recaller (step 4); a recaller call that overlaps one replaces it.
- `--edge-trim` (10) masks the read ends, where calls are unreliable.

`fiberhmm-apply` stops here and writes `ns`/`nl`/`as`/`al` only.

## 3. Nucleosome recall

The HMM tends to merge adjacent nucleosomes when the linker between them
happens to carry few marks. Nucleosome recall (on by default,
`--no-recall-nucs` to skip) re-examines every nucleosome-sized footprint.

It uses the same likelihood-ratio framework as the TF recaller (below) with
the opposite sign: it looks for accessible stretches *inside* a protected
footprint. A stretch whose accessible evidence exceeds `--split-min-llr` (4.0
nats) over at least `--split-min-opps` (3) informative positions is a buried
linker, and the footprint is split there. The geometry is set by
`--nuc-recall-policy`:

- `conservative` treats every qualifying accessible run as a cut and then
  places conservative inner nucleosome edges.
- `topology` accepts a set of cuts only if every fragment stays at least
  `--nuc-min-size`, keeps the post-cut HMM fragments as the nucleosome
  intervals, and records unresolved edges with zero edge bytes, so sparse
  single-strand data cannot turn an unresolved edge into apparent
  accessibility.
- `auto` (default) uses `topology` for Nanopore Hia5 and `conservative`
  otherwise.

A **periodicity prior** (`--phase-nrl`) lowers the split threshold near
linkers predicted by the nucleosome repeat length. `auto` estimates the
repeat length from the sample (clamped to about 150–215 bp); `off` disables
it; a number fixes it. Splits made this way still need at least one local
mark, so a signal desert is never split. Finally, nucleosome-sized protected
calls exposed by the TF scan are promoted back to nucleosomes.

### DddA: phase-aware radial nucleosome recall

DddA also deaminates *inside* nucleosomes, in a helically phased pattern, so
the accessible-cut split above would shatter DddA nucleosomes. For DddA, nucleosome recall instead:

1. nominates nucleosome dyads from the radial (helically phased) deamination
   profile (`ddda_nuc_profile.json`);
2. scores candidate edges around each dyad with sequence-context-aware
   likelihoods, marginalizing the uncertain helical register and a local
   9–12 bp pitch, so on-phase internal deaminations remain compatible with
   wrapping;
3. places each edge at the median of its posterior and encodes the width of
   the central 90% interval in the edge bytes (`el`/`er`);
4. validates the final configuration against the HMM topology with a
   molecule-local Bayes-factor test: direct linker evidence can keep the HMM
   edge, protected evidence can move it.

The radial caller uses its own internal likelihood table
(`ddda_nuc_refine.json`, internal radial-nucleosome likelihoods); updating the
TF table does not change nucleosome calls. HMM footprints with no radial dyad
are kept as fallback calls with zero quality. TF recall then runs once on the
resulting MSPs; TF calls exposed only by the new nucleosome geometry need a
deamination within `--ddda-derived-tf-max-edge-gap` (12 bp) on both sides
(`-1` disables this check).

The profile (`ddda_phase_posterior_v1`) was locked after validation on twelve
independent HG002 scDAF libraries and the GM12878 NAPA and UBA1 targeted
cohorts. Its identity and file digest are recorded in the output header
(`nuc_model`, `nuc_sha256`).

## 4. TF recall: the likelihood-ratio recaller

For a context *c* the emission table gives the probability of a mark in each
state. From it FiberHMM precomputes two log-likelihood ratios (protected over
accessible):

- ℓ_hit(*c*) = log P(marked | protected, *c*) − log P(marked | accessible, *c*),
  negative: a mark is evidence of accessibility;
- ℓ_miss(*c*) = log P(unmarked | protected, *c*) − log P(unmarked | accessible, *c*),
  positive: an unmarked target is evidence of protection.

On each read's lattice of target bases, the recaller chooses the set of
disjoint intervals that maximizes

```text
sum over intervals I of ( sum of ℓ inside I )  −  λ × (number of intervals)
```

where every interval contains at least `--min-opps` (3) informative targets
and starts and ends on protected evidence. The all-accessible configuration
scores zero. λ is `--min-llr`, the per-interval cost: **5.0 nats for every
bundled preset** (Hia5, DddB, DddA). This is an exact linear-time dynamic
program (`multi_interval_v1`), so a marked gap can separate two footprints
even when the first did not use up its evidence. The recaller scans the MSPs
and the short HMM footprints.

For each call it reports:

- the LLR of the interval (without subtracting the cost), as
  `tq = min(255, round(10 × LLR))`;
- the conservative edge (just past the last informative unmarked target) and,
  from the distance to the bracketing mark, the edge sharpness `el`/`er`.

λ is a regularizer, not an FDR threshold, and `tq` is continuous evidence
under the emission table, not a posterior probability. Keep all calls and
filter on `tq` downstream if you need a stricter set; `--min-llr` changes the
segmentation itself. See [Annotations and scores](annotations.md).

## What the output means

After the four steps every called read carries:

- nucleosomes (`MA` group `nuc`, and `ns`/`nl`), with an evidence byte and
  edge bytes when nucleosome recall ran;
- MSPs (`msp`, and `as`/`al`);
- TF and other sub-nucleosomal footprints (`tf`), with `tq`, `el`, `er`.

All intervals are in the molecule's own orientation
([Coordinate frames](coordinates.md)). The per-read calls are the input to
everything downstream: tracks, QC, footprint classes, strand rescue and the
footprint population model.
