# FiberHMM reference

Deep reference for FiberHMM 3.0: defaults, the BAM tag schema, scoring, output
formats and every command's flags. For installation and day-to-day usage see
the [README](../README.md).

- [Analysis modes](#analysis-modes)
- [Chemistry defaults](#chemistry-defaults)
- [BAM tag glossary](#bam-tag-glossary)
- [MA/AQ molecular-annotation schema](#maaq-molecular-annotation-schema)
- [MA type discovery header](#ma-type-discovery-header)
- [Chemistry declaration header](#chemistry-declaration-header)
- [Quality bytes: tq / el / er](#quality-bytes-tq--el--er)
- [The log-likelihood-ratio recaller](#the-log-likelihood-ratio-recaller)
- [DAF adjacent-target runs](#daf-adjacent-target-runs---daf-mask-runs-keep-one)
- [DddA CpG-island methylation](#ddda-cpg-island-methylation)
- [recall-tfs output modes](#recall-tfs-output-modes)
- [Circular molecules](#circular-molecules)
- [Haplotype fields in BED / bigBed extraction](#haplotype-fields-in-bed--bigbed-extraction)
- [Paired-duplex tags](#paired-duplex-tags)
- [Consensus BAM contract](#consensus-bam-contract)
- [Command notes](#command-notes): QC, DAF SNP masking, deduplication,
  extraction, utilities and training
- [Reading the output](#reading-the-output)
- [Command-line reference](#command-line-reference) (generated from argparse)

---

## Analysis modes

| Mode | Normal selection | Description | Target bases |
|------|------------------|-------------|--------------|
| **PacBio fiber-seq** | `--enzyme hia5 --seq pacbio` | m6A at A and T (both strands) | A, T (with RC) |
| **Nanopore fiber-seq** | `--enzyme hia5 --seq nanopore` | m6A at A only (single strand) | A only |
| **DAF-seq** | `--enzyme dddb` or `--enzyme ddda` | Deamination at C/G (strand-specific) | C or G |

The `pacbio-fiber` vs `nanopore-fiber` distinction only matters for Hia5 (m6A),
where PacBio detects modifications on both strands while Nanopore detects only
one. For deaminase methods (DddA, DddB), DAF mode is selected regardless of
sequencing platform. High-level commands infer this from `--enzyme`/`--seq`;
custom models use their embedded mode metadata.

The legacy high-level `--mode` flag is hidden but still accepted for old scripts
and recovery from incorrect custom-model metadata. An explicit value wins even
when it contradicts normal inference, and emits a warning. New commands should
select the chemistry and platform instead. Low-level model-building tools still
take an explicit mode because it is an input to constructing the model.

## Chemistry defaults

Settings that depend on the resolved chemistry (`--enzyme`, and for Hia5 the
given or detected `--seq`; tools that read an existing FiberHMM BAM and have no
`--enzyme` use its `FIBERHMM-CHEMISTRY` declaration):

| Setting | Hia5 PacBio | Hia5 Nanopore | DddB | DddA |
|---|---|---|---|---|
| ML threshold, `fiberhmm-call` / `-apply` | 128 | **248** | 128 | 128 |
| ML threshold, `recall-tfs`/`-nucs`, `extract`, `qc` | 125 | **248** | 125 | 125 |
| ML threshold for MM/ML dU, `dedup` / `pair` / `merge` | — | — | 128 | 128 |
| TF interval cost `--min-llr` | 5.0 | 5.0 | 5.0 | 5.0 |
| Nucleosome recall policy (`auto`) | conservative | topology | conservative | radial (phase-aware) |
| CpG-aware recall (`--use-m5c`) | off | off | off | **on** (call, recall, pair/merge) |
| Adjacent-target thinning (`--daf-mask-runs`) | — | — | off | 2, keep-one |
| Duplicate marking, SNP screen (file input) | — | — | on | on |

An explicit `--prob-threshold` always wins. The threshold applies only to
MM/ML calls: R/Y- and MD-encoded deaminations are binary. The Hia5 Nanopore
value (248) is the threshold its bundled QC reference and the strand-rescue
`hia5-nanopore` preset are calibrated at. The resolver is
`fiberhmm.models.default_prob_threshold`.

`fiberhmm-call` and `fiberhmm-apply` call primary alignments only by default:
secondary and supplementary records are written through uncalled
(`--no-primary` calls them too). Records whose MM/ML cannot match SEQ (hard
clips without a matching `MN` tag) are always skipped (`hard_clipped_mm`).
Unmapped reads are called automatically for stdin, unindexed and unaligned
input. Every skipped record is written through without this run's call tags.

## BAM tag glossary

All interval tags are written in the **molecular (original-fiber) frame**: for
a reverse-aligned read, an interval `[s, s+l)` in SEQ (query) coordinates is
written as `[L-(s+l), L-s)` and edge bytes (`el`/`er`) are swapped; every list
is sorted by molecular start. Forward reads are unaffected. `fiberhmm-call`
states this with the `coord=molecular` token in its `@PG` description;
`fiberhmm-apply` and `recall-tfs`/`-nucs` add `@CO fiberhmm:coord=molecular`.
Readers accept either; a BAM with neither (FiberHMM 1.x) is treated as SEQ
frame. Use `fiberhmm.io.ma_tags.flip_intervals_to_seq` to get SEQ coordinates.

**Footprint calls** (`fiberhmm-call`, `-apply`, `-recall-tfs`/`-nucs`, and the
joint recall of `fiberhmm-pair`):

| Tag | Type | Description |
|-----|------|-------------|
| `ns` / `nl` | B,I | Nucleosome starts / lengths (0-based, molecular frame). With `--downstream-compat`, TF calls are included. |
| `as` / `al` | B,I | MSP starts / lengths (0-based, molecular frame). `fiberhmm-apply --no-msps` omits them. |
| `nq` | B,C | One byte per `ns` entry: with nucleosome recall, LLR ×10 like `tq` (0 = no protected evidence); otherwise the HMM posterior mean ×255 (`--scores`), the input's `nq` carried through, or 0. |
| `aq` | B,C | MSP posterior mean ×255; written only by `fiberhmm-apply --scores`. |
| `MA` | Z | Molecular annotations (next section). |
| `AQ` | B,C | Quality bytes for the `MA` annotations. |
| `AN` | Z | Annotation names; written only when a call wraps a circular origin or preserved groups carry names (`fh_*`, `fhw_*` for wrapped pieces). |

`--no-legacy-tags` writes only `MA`/`AQ` (existing legacy tags are left as
they were); `--downstream-compat` writes TFs into `ns`/`nl` and no
`MA`/`AQ`/`AN`. Records a run skips (filters, primary-only, no observations)
are written without the previous run's call tags; DddA island groups
(`ddda_mcg`, `ddda_ucg`) are kept.

**Other per-read tags:**

| Tag | Type | Writer | Description |
|-----|------|--------|-------------|
| `st` | Z | `fiberhmm-daf-encode` | Conversion strand: `CT` (C→T) or `GA` (G→A); SEQ carries Y/R codes. The one-pass DAF path in `fiberhmm-call` derives strand internally and does not write `st`. |
| `di` | i | `fiberhmm-dedup`, `fiberhmm-call` dedup | Duplicate-cluster id, on every member of a cluster of two or more (only the representative remains after a collapse). |
| `ds` | i | same | Cluster size (copies the molecule has in the input). |
| `mt`, `mp`, `pm`, `pa`, `sb`, `sd`, `sr`, `sg`, `mc`, `dm`, `mg`, `mv`, `cs`, `dc`, `bc` | | `fiberhmm-pair` / `-merge` | See [Paired-duplex tags](#paired-duplex-tags). |

Duplicates are also flagged `0x400` (`fiberhmm-dedup --flag-only` and
`fiberhmm-call`'s default mark-and-retain dedup).

**Header lines:**

| Line | Writer | Content |
|------|--------|---------|
| `@PG` | call, recall, pair, tag-m5c, consensus, … | Provenance. `fiberhmm-call`'s `DS` carries `coord=molecular` and the resolved settings (`mode`, `enzyme`, `prob_threshold`, `primary_only`, `tf_interval_penalty`, `recall_nucs`, `nuc_recall_policy`, `phase_nrl`, `chimera_filter`, `dedup`, `daf_snp_mask`, `daf_run_mask`, `cpg_mask`, …); recall's `DS` has the recall subset. |
| `@CO FIBERHMM-CHEMISTRY:v1:` | call, recall-tfs/-nucs | Authoritative chemistry ([below](#chemistry-declaration-header)). |
| `@CO MA-TYPES:v1:` | MA-producing commands | Advisory list of MA names ([below](#ma-type-discovery-header)). |
| `@CO fiberhmm:coord=molecular` | apply, recall-tfs/-nucs | Frame marker. |
| `@CO FIBERHMM-CONSENSUS-MA:v1:`, `FIBERHMM-CONSENSUS-FAMILY:v1:` | consensus, transfer | [Consensus BAM contract](#consensus-bam-contract). |
| `@CO FIBERHMM-STRAND-RESCUE:v4:`/`v6:`, `FIBERHMM-TF-FAMILY:v1:` | strand-rescue-annotate, tag-consensus | Shadow-layer contracts ([strand_rescue.md](strand_rescue.md)). |

## MA/AQ molecular-annotation schema

The recallers write one `MA` tag and one `AQ` tag per processed read
([spec](https://github.com/fiberseq/Molecular-annotation-spec)):

```
MA:Z:<read_length>;nuc.QQQ:s1-l1,s2-l2,...;msp.:s1-l1,...;tf.QQQ:s1-l1,...;ddda_ucg.:s1-l1
AQ:B:C:nq,el,er, nq,el,er, ..., tq,el,er, tq,el,er, ...
```

Each interval is `<1-based start>-<length>` in the molecular frame; the strand
field is always `.`. Groups are written in the order `nuc`, `msp`, `tf`, then
any groups the command preserves (for example `ddda_mcg`, `ddda_ucg`,
`deam+`/`deam-`) unchanged with their AQ bytes. `AQ` holds the quality bytes of
every annotation in `MA` order; groups without a quality spec contribute none.

| Annotation | Quality bytes | Writer | Meaning |
|---|---|---|---|
| `nuc.QQQ` | `nq, el, er` | nucleosome recall (the `fiberhmm-call` default, `recall-nucs`, pair recall) | Recalled nucleosomes. `nq` = LLR ×10 of the retained configuration (0 = unresolved); `el`/`er` as for TFs. |
| `nuc.Q` | `nq` | `recall-tfs`, `call --no-recall-nucs` | Nucleosomes (`nl ≥ --unify-threshold`, or short nucs the TF recaller did not match); `nq` carried from the input or 0. |
| `msp.` | none | all | Methylase-sensitive patches. |
| `tf.QQQ` | `tq, el, er` | all recallers | TF/Pol II footprints (below). |
| `ddda_mcg.` | none | `fiberhmm-tag-m5c`, `fiberhmm-call-m5c --tag-bam` | Complete CpG island with a confident molecule-specific methylated state. |
| `ddda_ucg.` | none | same | Complete CpG island with a confident unmethylated state; the CpG-aware recall whitelist. |
| `deam+` / `deam-` | none | `fiberhmm-pair` merge | CT- and GA-source aligned coverage of a joint molecule; their intersection is the both-strand region. |
| `nuc_sr.QQQ`, `tf_sr.QQQ` | SR alternative (`q0`), molecular-left edge, molecular-right edge | `fiberhmm-strand-rescue-annotate` | Complete shadow layers (below). |
| `tf_sr.QQQQQ` | as above plus `fi`, `fq` | `fiberhmm-tag-consensus` | Consensus-state slot (0 = unassigned) and producer-declared assignment confidence ×255. |
| `tf_consensus.QQQQQQ`, `tf_cross_consensus.QQQQQQ`, `tf_recaller.QQQQQQ` | six bytes | `fiberhmm-consensus`, `fiberhmm-transfer` | [Consensus BAM contract](#consensus-bam-contract). |

`ddda_mcg+`/`-` and `ddda_mcg_hemi+`/`-` from older experimental per-CpG
callers are still read but no longer written.

For DddA phase-aware radial recall, `nq` scores the topology-changing linker
configuration, not the much easier question of protected DNA versus open
linker. A retained split or inward edge uses the molecule-local
accessible-residue log Bayes factor. Baseline nucleosome refinement receives no
provisional TF calls or population prior; TF recall runs once afterward. An
unresolved boundary or a retained HMM footprint with no radial dyad receives
`nq=0`. This avoids the uninformative saturation produced by scoring a full
protected radial window as though it verified the boundary topology.

### Strand-rescue shadow layers (`nuc_sr`, `tf_sr`)

For an MSP-to-TF rescue (`R`), `q0 = round(255 * P(H* | A or S))`, where
`H*` is the exact selected TF configuration, `A` is the exact ordinary MSP
baseline, and `S` is the complete supported local TF-configuration action set.
The log scores combine the opposite-strand population prior and the target
molecule's hard-chemistry likelihood. All
components of a multi-TF rescue share one `AN` prefix with `R0`, `R1`, ...
roles and one `q0`, so they switch atomically. Their component-specific `q1`
and `q2` are canonical molecular-left and molecular-right boundary reliability
scores. An `R` role is valid only in `tf_sr`.

An accepted ordinary TF or nuc edge alternative is a singleton `H` role. Its
`q0` is the equal-prior posterior for the bivariate canonical start/end
hypothesis versus that exact ordinary interval. The log odds combine
molecule-collapsed opposite-strand start/end population evidence under a
regularized bivariate geometry model with the target molecule's joint
changed-base chemistry evidence. `q1` and `q2` are corresponding marginal
molecular-left and molecular-right edge posteriors; an unchanged edge is shared
by both hypotheses and receives `255`. A changed edge with no target-molecule
opportunity receives `0` in q1/q2 even when the population nominates the
canonical geometry; the full population-plus-chemistry posterior remains in
q0 and the report. Family assignment still has to beat an
explicit unmatched null, and overlap/order checks retain the ordinary edges on
topology conflicts. An `H` changes edges only and always remains the same class
as its source ordinary call.

These are linear 0--255 quantities, not Phred scores, ordinary TF `tq`, or
nucleosome `nq`. Ordinary `tq` is not used to decide which source TF calls
count. `q1` and `q2` are always in molecular orientation; on reverse
alignments the annotator swaps the reference-left and reference-right values.

Unchanged calls in either complete shadow layer are unnamed (`AN` token `.`)
and receive `(255,0,0)`. This is a completeness sentinel, not a high-confidence
SR alternative; their original evidence qualities remain on ordinary `nuc` and
`tf`.

FiberBrowser applies one threshold only to named `R` and `H` groups. At
`q0 >= T`, it displays the named SR interval or atomic interval set; below the
threshold it displays the exact ordinary baseline: the containing MSP for `R`,
or the same-class ordinary TF/nuc identified by the `H` source ordinal.
Unnamed sentinel rows must be ignored by the threshold. `nuc_sr` and `tf_sr`
are complete shadows, not complementary N/TF alternatives and not the CR
nucleosome-versus-reconstruction slider.

Ordinary nuc, MSP, and TF intervals, quality rows, and annotation identities
remain unchanged. Edge updates that would introduce a new overlap in the
combined shadow callset, or invert order within one layer, retain their baseline
edges. An edge expansion also yields to a nonoverlapping-baseline rescued TF
rather than hiding it. Existing baseline overlaps are grandfathered. No
nucleosome can be split, merged, promoted, demoted, or removed, and no
nucleosome-length ceiling applies.

Every `H` annotation name carries the type-local ordinal of the ordinary call
it replaces in the shadow layer. Report identity also includes the canonical
input path, alignment fields, source-record SHA-256, occurrence among
byte-identical records, molecular interval, and annotation ordinal. Duplicate
ordinary intervals and duplicate alignment records therefore retain literal
one-for-one shadow identity.

`fiberhmm-strand-rescue-annotate` writes only new indexed regional BAMs.
`fiberhmm-strand-rescue-audit` validates the v4 two-layer cardinality, quality,
role, MA/AQ/AN alignment, overlap, header contract, and index integrity. The
annotator accepts v2/v3 reports and emits the v4 `QQQ` contract; the auditor can
also validate existing v2/v3 BAM contracts.

Repeated SR `--bam` inputs explicitly define one pooled same-assay cohort. BAM
identity is retained for output routing and per-input amplified-DAF duplicate
collapse, never used as a cross-library prior. Independent assays do not enter
inference. SR consumes only standard sequence, hard MM/ML (or DAF mismatches),
and existing MA calls. The `hia5-nanopore` preset reads standard Dorado m6A and
uses the strict hard-call threshold `ML >= 248`. Consensus reconstruction from
annotation combinations is described in
[`CONSENSUS_WORKFLOW.md`](./CONSENSUS_WORKFLOW.md).

### DddA island annotations

`ddda_mcg.` is a molecular interval annotation, not a native per-base `MM:C+m`
modification call. DAF amplification removes that native channel. Each span is
a complete CpG island assigned a confident methylated state on one molecule;
the caller does not infer transitions or boundaries within the island. The
model and its calibration are DddA-specific and must not be applied to DddB.

## MA type discovery header

MA names are extensible and otherwise appear only inside per-read `MA:Z`
values. FiberHMM therefore advertises the logical annotation names an output
BAM may contain with this optional SAM header-comment convention:

```text
@CO<TAB>MA-TYPES:v1:nuc,msp,tf,ddda_mcg
```

`<TAB>` above means one literal ASCII tab (`0x09`), as required between `@CO`
and its text in a SAM header; the five characters `<TAB>` are not written.
`pysam` exposes the comment text itself as `MA-TYPES:v1:...`.

Declarations contain names only—never the per-annotation strand (`.`, `+`,
`-`) or quality suffix (`Q`, `QQQ`). Names are case-sensitive and follow the MA
grammar `[A-Za-z0-9_]+`. Multiple declarations are valid:

```text
@CO<TAB>MA-TYPES:v1:nuc,msp
@CO<TAB>MA-TYPES:v1:tf,ddda_mcg
```

Readers take the ordered union, discard duplicates, and preserve first-seen
ordering. FiberHMM producers retain existing comments and append one new
declaration containing only names that were not already validly declared. When
emitting MA, the main caller and recall tools advertise `nuc,msp,tf`; the DddA
island caller additionally advertises `ddda_mcg,ddda_ucg`.

This metadata is a discovery hint, not part of MA correctness:

- Records remain authoritative. Readers accept observed names that were not
  declared and may add those layers dynamically.
- A missing name does not establish biological absence, and declarations must
  never determine `AQ` arity.
- Missing, stale, malformed, or header-tool-removed declarations are ignored.
- Advertising a name never creates an empty per-read section such as
  `ddda_mcg.:`.

To repair an older BAM in place, either state the known logical names or scan
every alignment (not a sample):

```bash
fiberhmm-utils ma-types calls.bam --types nuc,msp,tf,ddda_mcg
fiberhmm-utils ma-types calls.bam --scan
```

Because a compressed BAM header cannot generally grow byte-for-byte in place,
the utility writes and validates a temporary BAM beside the original, rebuilds
any existing BAI/CSI index, and then atomically replaces each file. Per-read
tags are copied unchanged. If all requested names are already declared, it
does not rewrite the BAM.

## Chemistry declaration header

FiberHMM records the scientific observation model independently of filenames
and free-text command provenance with a versioned SAM header comment:

```text
@CO<TAB>FIBERHMM-CHEMISTRY:v1:assay=daf;enzyme=ddda;platform=pacbio;mode=daf;model=ddda_TF;nuc_model=ddda_phase_posterior_v1;nuc_sha256=c86b05dc07e45392880e3460cf7f8880593ecad174e0a338d36ac53b7d0172d6
```

The required v1 fields are `assay`, `enzyme`, `platform`, and `mode`. Producers
may append fields such as `model`; `fiberhmm-call` and the standalone recall
tools also record `nuc_model` and the exact profile-file digest `nuc_sha256`
when a distinct nucleosome profile is active. Field names match
`[a-z][a-z0-9_]*`; values are non-empty tokens matching
`[A-Za-z0-9_.+-]+`. The supported vocabulary is:

- `assay=daf`, `enzyme=ddda|dddb`, `mode=daf`, with the actual sequencing
  `platform=pacbio|nanopore` (or `unknown` only when unavailable);
- `assay=fiber-seq`, `enzyme=hia5`, and either
  `platform=pacbio;mode=pacbio-fiber` or
  `platform=nanopore;mode=nanopore-fiber`;
- `custom` for an explicitly custom assay, enzyme, or mode.

Valid v1 declarations are authoritative scientific metadata. Multiple lines
with the same four required fields are allowed (for example after successive
models add distinct `model` values); incompatible required fields constitute a
conflict and FiberHMM producers refuse to silently relabel them. Readers ignore
malformed lines and unknown convention versions. For BAMs made before this
contract, tools may report lower-confidence compatibility inference from a
`fiberhmm-call` `@PG` record, but filename inference is never equivalent to a
declaration.

`fiberhmm-call` and `fiberhmm-recall-tfs`/`-recall-nucs` emit this line
together with their `@PG` provenance, reconciling it with the input's: a custom
`--model` without `--enzyme` inherits the input's assay, enzyme and platform
when the observation mode matches; any other disagreement stops the run unless
`--replace-chemistry` re-declares the output. Tools that read a FiberHMM BAM
(recall, extract, qc, consensus) take the chemistry, and with it the default ML
threshold, from this line. Downstream BAM transformations retain it as part of
the copied header.

## Quality bytes: tq / el / er

**`tq` — LLR-based confidence (0–255)**

```
tq = clip(round(LLR_nats * 10), 0, 255)
```

Every **23 tq points = one order of magnitude** of likelihood ratio. Recommended
thresholds:
- `tq ≥ 50` (LLR ≥ 5 nats, LR ≈ 148:1) — soft floor
- `tq ≥ 100` (LLR ≥ 10 nats, LR ≈ 22,000:1) — high confidence
- `tq = 255` — saturated (LLR ≥ 25.5, LR ≥ 1.2e11)

**`el` / `er` — edge sharpness (0–255)**

The recaller emits a **conservative** boundary at each edge (immediately past
the last informative miss). The true boundary may extend up to the terminating
hit. `el`/`er` encode that ambiguity:

```
el = round(255 * max(0, 1 - left_ambiguity_bp / 30))
er = round(255 * max(0, 1 - right_ambiguity_bp / 30))
```

- `255` — a hit sits immediately adjacent (sharp edge; size estimate is exact)
- `0` — the bracketing hit is ≥30 bp away (edge could extend further; size is a lower bound)

The interval (`ns`/`nl`) is written at the **conservative (strict) boundary**;
the edge-sharpness bytes recover the loose boundary. A DddA dyad-nominated raw
nucleosome edge is the median of its continuous phase-marginal posterior, and
`el`/`er` encode the width of the central 90% interval using the same 30-bp
saturation convention. Edge quality never chooses between raw-edge estimators.
The final emitted coordinate can nevertheless be constrained by the uniform
molecule-local comparison with the HMM configuration and by non-overlapping
tiling. HMM calls with no radial dyad and non-radial promoted or fallback calls
are outside this posterior-edge contract; unresolved final boundaries are
explicitly Q0.

## The log-likelihood-ratio recaller

Both footprint recallers — the transcription-factor (TF) recaller and the
nucleosome recaller — operate within a common likelihood-ratio framework derived
from the trained two-state emission model.

**Statistical model.** The model distinguishes a *protected* state (φ;
nucleosome or protein footprint) from an *accessible* state (α; linker or
methylation-sensitive patch). For each *k*-mer sequence context *c*, the emission
table specifies the probability of observing a modification — N6-methyladenine
for fiber-seq, cytosine/guanine deamination for DAF-seq — conditional on the
state. From these, two per-position log-likelihood ratios are precomputed for
every context:

> ℓ_hit(*c*)  = log P(modified ∣ φ, *c*) − log P(modified ∣ α, *c*)
> ℓ_miss(*c*) = log P(unmodified ∣ φ, *c*) − log P(unmodified ∣ α, *c*)

where a *hit* denotes an observed modification and a *miss* an unmodified
instance of the target base. Because the modifying enzyme acts preferentially on
accessible DNA, hits are evidence for the accessible state (ℓ_hit < 0) and misses
for the protected state (ℓ_miss > 0).

CpG-aware DddA recall (on by default for DddA in `fiberhmm-call`,
`recall-tfs`/`-nucs` and the joint recall of `fiberhmm-pair`) uses CpG
observations only within confidently unmethylated whole-island annotations
(`MA:ddda_ucg`) and excludes all other CpGs from the opportunity lattice of
both the nucleosome and the TF recaller. Excluded sites contribute no
likelihood, do not count toward minimum evidence, and cannot define footprint
boundaries or boundary uncertainty; non-CpG observations retain their ordinary
emissions. The `--cpg-mask-policy methylated-only` compatibility setting
reproduces the former behavior of excluding CpGs only within `MA:ddda_mcg`
spans. See [DddA CpG-island methylation](#ddda-cpg-island-methylation).

**Maximal-segment inference.** Over a candidate interval the recaller accumulates
the per-position log-likelihood ratio and identifies the contiguous sub-interval
of maximal cumulative score by a linear-time maximum-subarray procedure. A
sub-interval is reported when its cumulative score exceeds a threshold
(`min_llr`) over a minimum number of informative positions (`min_opps`). For each
call it records the distance from the terminal informative position to the
nearest opposing observation on either flank, yielding a conservative inner
boundary and a bound on the true (loose) boundary.

**Dual application.** The two recallers correspond to the two signs of the same
statistic. The TF recaller scans accessible regions for protected segments
(positive ℓ), reporting sub-nucleosomal footprints. The nucleosome recaller scans
an over-merged protected footprint for accessible segments (negative ℓ); a
sufficiently supported accessible segment denotes a buried linker at which the
footprint may be divided.

The nucleosome geometry is controlled by `--nuc-recall-policy`:

- `conservative` treats every qualifying accessible run as
  a cut, after which the positive-sign scan defines conservative inner
  nucleosome boundaries. On sparse single-strand data, this can turn unresolved
  sequence into apparent accessibility.
- `topology` accepts a set of cuts only when every outer and intervening
  fragment remains at least `--nuc-min-size`. It retains each post-cut HMM
  fragment as the occupancy interval and records unresolved edges with zero
  edge-sharpness bytes. Thus isolated events cannot shatter one nucleosome and
  neutral edge ambiguity is not reported as an NFR.
- `auto` (the CLI default) selects `topology` for `nanopore-fiber` models and
  `conservative` otherwise. Either behavior can be forced explicitly.

The topology policy still recalls over-merged nucleosomes: supported internal
linkers divide long footprints, and the maximum-total-LLR compatible cut chain
is selected when several candidate linkers occur.

**DddA phase-aware radial configuration validation.** DddA internal
deaminations make the ordinary accessible-cut pass unsuitable, so the radial
template nominates protected dyads and candidate gaps. At each dyad, the caller
scores raw candidate edges with the chemistry's sequence-context emissions
while marginalizing a calibrated grid of uncertain helical registers and local
9–12-bp pitch. On-phase internal deaminations can remain compatible with
wrapping, and missing one or several rotational opportunities does not force an
edge. A weak, broad particle-extent prior is applied identically to every
molecule. The posterior median defines each dyad-nominated raw edge and
posterior width affects only `el`/`er`; there is no confidence-selected
coordinate switch.

Final configuration validation starts from a non-overlapping tiling constrained
by the HMM nucleosome topology. Whenever a raw posterior edge would reclaim
HMM-accessible sequence, the same molecule-local, sequence-context and
phase-aware configuration Bayes-factor test is applied at every posterior
width. Direct linker evidence can retain the HMM edge; protected evidence can
accept the posterior crossing. A supported internal linker separates adjacent
phase-supported particles. If the intervening state is unresolved, the
particles remain separate with facing Q0 edges and the residue is withheld from
TF scan space rather than being averaged into one giant particle or declared
accessible. An HMM footprint with no radial dyad is retained as a Q0 fallback.

This baseline pass receives no TF calls, strand consensus, or population prior.
After it fixes the nucleosome configuration and rebuilds MSPs, TF recall runs
once. Nucleosome-sized protected calls exposed in that scan can be promoted back
to nucleosomes, and `--ddda-derived-tf-max-edge-gap` provides a secondary
two-sided evidence check for small TF calls created only by the new scan space.
Coverage-gated strand/family consensus and its optional `nuc_sr` alternative are
later analyses; they do not alter the baseline call.

The production profile (`ddda_phase_posterior_v1`) was locked after validation
on deterministic whole-genome samples from twelve independent HG002 scDAF
libraries (35,727 primary reads) and independent GM12878 NAPA and UBA1 targeted
molecules (3,016 and 5,539 reads). Unsmoothed one-base size distributions were
inspected per library and jointly for estimator cliffs and residual one-sided
10-bp combs. This is a distributional and implementation validation, not a
claim that population size is ground truth; coordinates remain determined from
each molecule's chemistry likelihood.

## DAF adjacent-target runs (`--daf-mask-runs`, keep-one)

Adjacent targets on the deaminated strand (CC on CT reads, GG on GA reads; runs of two or more) do not convert independently. The per-site emission model assumes they do.

`--daf-mask-runs N` thins every same-strand run of N or more original targets. Runs are measured on the molecule's original sequence, so a run is thinned whether or not it converted. With `--daf-run-policy keep-one` (the default policy), each run keeps only its 5'-most target on the deaminated strand, with its own observation. `drop` removes the whole run.

The thinning happens in the DAF observation encoder, so it applies at every level:
- HMM, nucleosome and TF LLR calling (`fiberhmm-call`, `apply`, `recall-tfs`/`recall-nucs`, `pair`/`merge` recall);
- the consensus (CR) lattices;
- the opposite-strand and duplex-mate lattices.

**Defaults:**

| chemistry | default |
|---|---|
| DddA (`--enzyme ddda`, consensus `ddda` datasets) | `N=2`, keep-one |
| DddB and others | off |

**Why DddA defaults to keep-one.** On 15,557 sequence-assigned HG002 scDAF duplexes, precision of native TF calls rose in every stratum, read on the complementary strand of the same molecule:
- 0.95 → 0.98 for calls containing runs;
- 0.93 → 0.95–0.96 for calls without runs;
- Youden J (any protection) rose from 0.56 to 0.63.

The cost is 16–27% fewer calls. DddB has not been validated.

**Overriding.** `--daf-mask-runs 0` disables the mask. An explicit value on `fiberhmm-consensus` applies to every DAF dataset.

**Provenance.** The setting is recorded in `@PG` (`daf_run_mask=>=2/keep-one` or `off`) and in the consensus model manifest.

**Native calls must be replayed.** Consensus refuses a masked lattice when native calls are not replayed (`input.correct_native`), because existing BAM calls were decoded on a different lattice.

## DddA CpG-island methylation

Genome-wide DddA amplification removes native 5mC tags, but methylated CpGs
keep a strong DddA rate signature. FiberHMM reports it at the scale of complete
CpG islands, after ordinary calling:

```bash
fiberhmm-call -i aligned.bam -o calls.initial.bam --enzyme ddda -c 8 --region-parallel
fiberhmm-tag-m5c -i calls.initial.bam -o calls.m5c.bam -r reference.fa --enzyme ddda \
                 --write-cpg-islands islands.used.bed --calls-tsv island_calls.tsv
fiberhmm-recall-tfs -i calls.m5c.bam -o calls.bam --enzyme ddda -c 8
```

By default the tagger derives islands from 200-bp reference windows stepped by
10 bp, keeping windows with GC fraction ≥ 0.50 and CpG observed/expected ≥ 0.60
and merging overlaps; `--cpg-islands` accepts a merged BED catalog instead. Only
CpG and non-CpG observations inside the molecule's initial MSP contribute.
Calls need ≥ 15 CpGs, ≥ 10 non-CpGs and a posterior ≥ 0.99 (methylated,
`ddda_mcg.`) or ≤ 0.01 (unmethylated, `ddda_ucg.`); other overlaps are recorded
as uninformative in the audit table. The tagger always reads the FASTA, even
when R/Y or `MD` is present.

The CpG-aware recall policy is shared by `fiberhmm-call`,
`fiberhmm-recall-tfs`/`-recall-nucs` and the joint recall of
`fiberhmm-pair`/`-merge`: for DddA it is on by default and excludes every CpG
observation from nucleosome and TF recall except inside `ddda_ucg` spans the
read already carries (`--cpg-mask-policy methylated-only` reproduces the older
policy of excluding only `ddda_mcg` spans; `--no-use-m5c` turns it off). A
first `fiberhmm-call` on fresh data therefore neutralizes all CpGs; recalling
after `fiberhmm-tag-m5c` restores the CpGs of confidently unmethylated
islands. For a paired duplex, an island called on either source read is
projected onto the joint molecule. The retired `fiberhmm-call --ddda-mcg`
exits with a migration message.

This caller and its emission correction are calibrated for genome-wide DddA
DAF-seq only; do not apply them to DddB or to targeted/amplicon DddA.

## recall-tfs output modes

The recaller supports two mutually-exclusive output modes; pick based on what
your downstream tooling can read. The runtime banner makes the active mode
explicit, and switching is a pure re-run on the same HMM-tagged input.

**Spec mode (default)** — write `MA`/`AQ` tags per the spec:
- `MA`/`AQ` carry `nuc.Q` (`nuc.QQQ` with `--recall-nucs`), `msp.`, `tf.QQQ`
  with full LLR + edge-ambiguity scoring.
- Legacy `ns`/`nl` is also refreshed but contains **nucleosomes only** — TF calls
  live exclusively in `MA`/`AQ`.
- Requires an MA/AQ-aware consumer (FiberBrowser, fibertools-rs). Tools that read
  only `ns`/`nl` will not see TF calls in this mode.

**Downstream-compat mode** (`--downstream-compat`) — put TF calls into legacy
`ns`/`nl` alongside nucleosomes, no `MA`/`AQ` written:
- Legacy `ns`/`nl` contains **all footprints** (nucleosomes + TFs), sorted by
  start. Entries `< --unify-threshold` (default 90 bp) are TFs; `≥` are
  nucleosomes. Downstream tools filter by size.
- Any pre-existing `MA`/`AQ` is stripped so consumers don't see a stale view.
- Per-TF quality (`tq`, `el`, `er`) is **lost** — only positions/lengths survive.
- Use when your pipeline (fibertools-rs, custom scripts, older browsers) reads
  only `ns`/`nl`.

**`--unify` (always on).** Every v2 short-nuc (`nl < --unify-threshold`)
overlapped by a recaller call is dropped from `nuc.`; the recaller version (with
`tq`/`el`/`er`) replaces it in `tf.`. Unmatched short-nucs stay in `nuc.` as
fallback entries with `nq=0`. v2 nucleosomes (`nl ≥ --unify-threshold`) are
preserved untouched.

## Circular molecules

`--circular` (`-r`) is for plasmids, mitochondrial genomes, and other circular
molecules where a feature can cross the arbitrary read origin. FiberHMM tiles
each read 3× internally for calling, then projects features back to the original
molecule before writing output (tiled coordinates are never written to BAM).

Wrapped features are serialized as two spec-valid clipped `MA` intervals, one at
each end of the read. The optional `AN:Z` tag gives both pieces the same
annotation name so circular-aware tools can fuse them:

```
MA:Z:1000;tf.QQQ:1-45,971-30
AQ:B:C:180,20,30,180,20,30
AN:Z:fhw_tf_0,fhw_tf_0
```

Tools that ignore `AN` still see valid linear `MA/AQ` intervals at both ends;
FiberBrowser and `fiberhmm-extract --circular-groups` use `AN` to reconstruct the
single wrapped feature. Legacy `ns/nl` and `as/al` are also split to stay
coordinate-valid but do not carry the fused identity.

## Haplotype fields in BED / bigBed extraction

`fiberhmm-extract --haplotype-fields` copies the source BAM record's scalar
`HP:i` (haplotype) and `PS:i` (phase set) tags into every emitted feature row.
The option is off by default, so existing BED text and bigBed autoSQL schemas
remain byte/schema compatible unless it is requested.

Optional columns always have a deterministic order:

```
BED12 | per-block scores | circular grouping | hp | ps
```

Both appended autoSQL fields are signed integers. `-1` means that tag was absent
or not integer-valued; the sentinels are independent, so `HP:i:1` without `PS`
is written as `1, -1`. Valid HP values are positive and valid PS identifiers are
non-negative. Wrapped circular pieces each repeat the source read's same HP/PS
values. Extraction only propagates tags: it does not phase reads, infer missing
values, or alter calls.

## Paired-duplex tags

`fiberhmm-pair` (pairing stage) tags the source reads; the merge stage writes
one joint molecule per pair. See [paired-duplex.md](paired-duplex.md) for the
algorithm and [duplex.md](duplex.md) for the sequence-free model.

| Tag | Type | On | Meaning |
|-----|------|----|---------|
| `mt` | A | source reads | `P` accepted pair member, `U` had candidates but failed the gate, `.` no opposite-flavor candidate |
| `mp` | Z | pair members | reciprocal mate query name |
| `pm` | A | pair members, joint molecule | pairing route: `S` sequence-supported, `D` sequence-free model (`F` = pre-3.0 footprint pairer, still accepted by the merger) |
| `pa` | A | sequence pairs | `R` reciprocal sequence edge, `C` constrained complete 2×2 |
| `sb` | i | sequence pairs | shared deamination-safe (reference A/T) bases |
| `sd` | i | sequence pairs | sequence differences at those bases |
| `sr` | i | sequence pairs | difference rate ×1,000,000 |
| `sg` | i | sequence pairs | sequence assignment margin ×1,000,000 |
| `mc` | i | sequence pairs (optional) | nucleosome dyad cross-correlation ×1000 |
| `dm` | i | model pairs | sequence-free decision score ×1000 |
| `mg` | i | model pairs | reciprocal decision margin ×1000 |
| `mv` | Z | model pairs | frozen model identifier |
| `cs` | Z | joint molecule | source read names, `<ct_name>;<ga_name>` |
| `dc` | i | joint molecule | reference-frame deamination count (C→T and G→A) |
| `bc` | i | joint molecule | canonical source-base conflicts written as `N` |

The joint molecule (`<ct_name>.cs`) is a forward, all-`M` reference-frame record
spanning the union of the two sources, with `deam+`/`deam-` MA coverage, MAPQ
equal to the lower source MAPQ, no base qualities and no `mt`/`mp`. It carries
the pair-evidence tags of its route.

## Consensus BAM contract

`fiberhmm-consensus` (and FiberBrowser's **Write family-tagged BAMs**) writes new
BAMs; source BAMs are never modified and native `nuc`, `msp` and `tf`
annotations stay intact. The full contract is in
[CONSENSUS_WORKFLOW.md](CONSENSUS_WORKFLOW.md#family-tagged-bams); in brief, for the
default lattice recaller:

| Item | Content |
|------|---------|
| `tf_consensus.QQQQQQ` | Native TF calls that belong to a class. AQ bytes `tq, fi, fq, op, sq, q0`: native LLR ×10, class slot, `fq` (0 = unavailable), opportunities, DAF core protection ceiling, and `q0` = the molecule's EM class posterior ×255 (1–255 for a label, 0 = no class; a mixture posterior, not a calibrated probability). A call is labelled only if the molecule is a member and the call fits the class edge boxes. `AN` is the class token (`fhcr_…`). |
| `tf_recaller.QQQQQQ` | Opt-in (`--bam-recaller-layer`): the recaller's own class calls at every prevalence tier, including molecules with no native call. AQ bytes `tq, fi, tier (1 core, 2 edge, 3 loose), q0, lr, rr` (edge-range widths in bp). Without it the recaller calls are only in `result.json.gz`. |
| `@CO FIBERHMM-CONSENSUS-FAMILY` | One entry per class: consensus span, and for DAF classes a per-dataset `strand_resolution` with `trusted_strand` (`CT`/`GA`/`both`/`none`); Hia5 classes have no strand verdict because orientations are pooled. |
| `@CO FIBERHMM-CONSENSUS-MA:v1:` | The engine and the meaning of every AQ byte. |

The deprecated staged engine writes the same layer names with different `q0`
semantics (class-evidence share), so recaller and staged results cannot share
one export. `read_family_catalog(bam.header)` in
`fiberhmm.inference.consensus.bam_export` reads the embedded catalogue.
Consensus input needs a supported chemistry (`ddda`, `dddb`, `hia5-pacbio`,
`hia5-nanopore`): from the BAM's declaration, or `--chemistry` for BAMs called
with a custom `--model` and no `--enzyme` or without FiberHMM metadata. Other
enzymes (for example EcoGII) are rejected.

## Command notes

Behaviour that does not fit in a flag description. Every flag is listed in the
[command-line reference](#command-line-reference).

### QC (`fiberhmm-qc`, and automatically after `fiberhmm-call`)

QC is bounded: indexed BAMs are sampled from seeded random genomic windows
(default target 2,000 reads); unindexed BAMs use a reservoir over at most a
bounded prefix (10× the requested sample). The JSON records the strategy, seed,
records examined and `whole_bam_scanned: false`. QC never deduplicates.

With one input, outputs are `<BAM directory>/qc/<BAM stem>.qc.json` and
`.qc.tsv`, plus `.qc.png` and an Illustrator-compatible `.qc.pdf` (editable
TrueType text) when matplotlib is installed (`fiberhmm[plots]`). Several inputs
add `combined.qc.{json,tsv,png,pdf,html}`; inputs from different directories
need `-o`. After `fiberhmm-call` the prefix is
`<output BAM directory>/qc/<BAM stem>` unless `--qc-output-prefix` is set.

Each figure shows the matched control ECDF/median and phasogram, control
nucleosome/TF size distributions, PCR duplication, a mismatch-percentage
landscape, an amplicon SNP-location map and example molecules. The scorecard
has separate signal-rate and nucleosome-periodicity scores and an overall
PASS/WARN/FAIL (`INSUFFICIENT` for low-evidence samples). Rate: the control
IQR is PASS, the rest of the 5th–95th percentile interval is WARN, outside is
FAIL. Periodicity is conjunctive (peak strength, NRL, reference-curve
correlation and the amplitude of the reference-shaped component must all
support the grade). If the run's ML threshold differs by more than 5 from the
reference's calibration threshold, the rate score is capped at WARN; the
default thresholds match the references (Hia5 Nanopore 248, others 125).
Size panels need `MA` (or legacy `nl`).

Automatic QC after `fiberhmm-call` selects its reference from the resolved
enzyme/platform: Drosophila embryo DddB DAF-seq, human DddA DAF-seq, Drosophila
embryo PacBio Fiber-seq or Drosophila embryo Nanopore Fiber-seq. Explicit
profile overrides exist only in standalone `fiberhmm-qc`; incompatible
combinations are rejected. When integrated dedup ran, a `.dedup.json` sidecar
gives QC the exact full-run duplicate fraction. Only aggregate control curves
and a few anonymous molecule exemplars are packaged; see
[`fiberhmm/qc/README.md`](../fiberhmm/qc/README.md). Reference intervals are
screening diagnostics, not biological exclusion criteria.

### DAF SNP masking (`fiberhmm-daf-snps`, `--daf-call-snps`)

DAF conversions and C/T or G/A variants are confounded. The caller classifies
each informative fiber by its dominant conversion direction and calls a
recurrent C→T mismatch only across otherwise G→A-dominant fibers (and G→A only
across C→T-dominant fibers), so molecule-specific deamination stays out of the
SNP denominator. Duplicate-flagged reads are excluded.

The production policy `bidirectional_five_fiber_v1` requires ≥ 20% mismatch
fraction, depth ≥ 5 and ≥ 5 mismatch fibers independently in each
conversion-direction class. It was selected by the downsampling validation in
`scripts/validation/validate_daf_snp_downsampling.py` (98.4% of prespecified
unambiguous sites retained at an expected minimum bidirectional depth of 10.5,
versus 61.2% with a separate depth-10 cliff; neither rule called a site in the
high-depth/low-mismatch background class). Amplicons need 20 primary
MAPQ-filtered reads. Overrides are recorded as a `custom` policy in the JSON.

Masking is observational: sequence and `MD` are preserved; listed reference
positions are excluded from DAF emissions and QC rate/periodicity. Outputs are a
BED mask, VCF, JSON report, `.amplicons.tsv` and QC panels (mismatch landscape
and amplicon SNP map). `fiberhmm-call` runs the caller automatically for
file-based DddA/DddB input after duplicate marking when a bounded depth
preflight finds adequate coverage; `--daf-snp-mask BED` applies a reviewed mask.

### PCR deduplication (`fiberhmm-dedup`, `--dedup`)

DAF-seq amplicons pile up on one locus with primer-fixed ends and no UMIs, so
coordinate dedup does not apply. The fingerprint is the set of reference
positions deaminated on the read (R/Y, MM/ML dU at ML ≥ 128, or `MD`
mismatches). Reads must agree at both aligned reference ends within 50 bp and
have the same deamination flavour (C→T vs G→A; alignment orientation is not
used; `--ignore-strand` clusters across flavours), then are clustered by
deamination-set Jaccard ≥ 0.95 (MinHash + LSH). Reads with fewer than 10 calls
are not fingerprinted and pass through.

Standalone `fiberhmm-dedup` collapses each cluster to one representative
(highest MAPQ, most complete) by default; `--flag-only` keeps every read. In
`fiberhmm-call` the integrated pass is nondestructive by default (every read is
kept; duplicates get `0x400`); `--dedup-collapse` removes them. Members of a
duplicate cluster carry `di` (cluster id) and `ds` (copies represented);
singletons carry neither.

### Extraction (`fiberhmm-extract`)

One pass writes one BED12/bigBed per feature type (`nucleosome`, `msp`, `tf`,
`m6a`, `m5c`, `deam`, `bothstrand`; all by default, and `deam` is skipped
automatically on fiber-seq BAMs). TF extraction applies `--min-tq` (default
50). Native MM/ML m6A/5mC/dU positions use `-p/--prob-threshold` (248 for BAMs
declaring Hia5 Nanopore, 125 otherwise). Every schema appends `isDuplicate`
(from flag `0x400`); FiberBrowser hides those rows by default. The sort runs
under `LC_ALL=C`; `-S` and `--sort-parallel` help on deep BAMs. Each bigBed
embeds a `Sample:` autoSQL tag used by FiberBrowser to group layers; repair older
files with `fiberhmm-utils fix-bigbed`. `--block-scores`, `--circular-groups`
and `--haplotype-fields` append optional columns (see
[Haplotype fields](#haplotype-fields-in-bed--bigbed-extraction)).

### Utilities (`fiberhmm-utils`)

```bash
fiberhmm-utils convert old_model.pickle new_model.json   # legacy -> JSON
fiberhmm-utils inspect model.json [--full]               # metadata + emissions
fiberhmm-utils transfer --target daf.bam --reference-bam fiber.bam -o probs/ --mode daf
fiberhmm-utils adjust model.json --state accessible --scale 1.1 -o adjusted.json
fiberhmm-utils ma-types calls.bam --types nuc,msp,tf,ddda_mcg
fiberhmm-utils ma-types calls.bam --scan
fiberhmm-utils fix-bigbed sample.filtered_T_*.bb sample.filtered_GA_*.bb --in-place
```

`ma-types` rewrites the header declaration through a temporary BAM and
rebuilds an existing index; `--scan` visits every alignment. `fix-bigbed`
rebuilds bigBeds with distinct `Sample:` tags (needs UCSC `bigBedInfo`,
`bigBedToBed`, `bedToBigBed`). Legacy `.pickle` models execute code when
loaded; convert only trusted files.

### Posteriors and training

```bash
fiberhmm-posteriors -i experiment.bam --enzyme hia5 --seq pacbio -o post.tsv.gz -c 4
fiberhmm-posteriors -i experiment.bam --enzyme hia5 --seq pacbio -o post.h5 -c 4   # fiberhmm[posteriors]
fiberhmm-probs -a accessible.bam -u inaccessible.bam -o probs/ --mode pacbio-fiber -k 3 4 5 6 --stats
fiberhmm-train -i sample.bam -p probs/tables/accessible_A_k3.tsv probs/tables/inaccessible_A_k3.tsv \
               -o models/ -k 3 --stats
```

`fiberhmm-posteriors` decodes the same observations as `fiberhmm-call`.
`fiberhmm-probs` builds emission tables from accessible/inaccessible control
BAMs, always numbering contexts in the encoder's order; `fiberhmm-train` fits
the HMM and writes `best-model.json` (recommended), `.npz`, every iteration,
the training read IDs, its config and `--stats` plots. Both exit non-zero when
no reads pass their filters. Low-level model-building commands take `--mode`
explicitly because it is an input to the model.

## Reading the output

```python
import pysam
from fiberhmm.io.ma_tags import flip_intervals_to_seq, parse_aq_array, parse_ma_tag

bam = pysam.AlignmentFile('calls.bam', 'rb', check_sq=False)
for read in bam:
    if not read.has_tag('MA'):
        continue
    parsed = parse_ma_tag(read.get_tag('MA'))
    qual_specs = [rt[2] for rt in parsed['raw_types']]
    n_per_type = [len(rt[3]) for rt in parsed['raw_types']]
    per_ann = parse_aq_array(read.get_tag('AQ'), qual_specs, n_per_type)
    # parsed['nuc'] / ['msp'] / ['tf'] = [(start, length), ...]:
    #   0-based, molecular frame.
    # per_ann: one list of quality bytes per annotation, in MA order
    #   ([nq, el, er] per nuc.QQQ call, [] per msp, [tq, el, er] per tf).
    # SEQ (query) frame, e.g. to map onto the reference:
    starts, lengths = zip(*parsed['tf']) if parsed['tf'] else ((), ())
    tf_seq = flip_intervals_to_seq(starts, lengths, read)
```

The legacy tags are refreshed to the same call set (molecular frame; TF calls
are only in `MA`/`AQ` unless `--downstream-compat`):

```python
ns, nl = list(read.get_tag('ns')), list(read.get_tag('nl'))   # nucleosomes
as_, al = list(read.get_tag('as')), list(read.get_tag('al'))  # MSPs
```

## Command-line reference

<!-- BEGIN GENERATED CLI REFERENCE (tools/gen_cli_reference.py) -->

Generated from each command's argparse definition by `python tools/gen_cli_reference.py`; do not edit by hand. Hidden compatibility options are omitted. `auto` means the value is resolved at run time as the description says.

### fiberhmm-call

| Flag | Default | Description |
|------|---------|-------------|
| `-i` / `--input` | required | Input BAM. Use "-" for stdin (streaming mode). |
| `-o` / `--output` | required | Output BAM path or "-" for stdout (unsorted). |
| `-m` / `--model` | — | Apply HMM model JSON. If omitted, bundled model for --enzyme/--seq is used. |
| `--recall-model` | — | Separate model for TF LLR tables. Default: reuse apply model. |
| `--enzyme` | — | Bundled enzyme preset. Choices: `ddda`, `dddb`, `hia5`. |
| `--seq` | — | Sequencing platform. For Hia5 it selects the model; when omitted it is detected from the input (MM specs: PacBio T-a vs Nanopore A+a only; header records) and the run stops if the evidence conflicts. For dddb/ddda it only sets the declared platform. Choices: `pacbio`, `nanopore`. |
| `--replace-chemistry` | off | Replace, instead of reconcile with, the input BAM's FIBERHMM-CHEMISTRY declaration (re-calling a BAM with a deliberately different chemistry). |
| `--reference` | — | Reference FASTA for DAF-seq BAMs that lack both R/Y IUPAC encoding and MD tags. When present, acts as a fallback source for deamination-site detection. Required and always used by --ddda-mcg to determine CpG and DddA sequence context. |
| `-k` / `--context-size` | — | Context size override. Default: from model. |
| `--edge-trim` | `10` | Bases to mask at edges (default 10) |
| `--min-mapq` | `0` | Min mapping quality (default 0) |
| `--prob-threshold` | — | Min MM/ML modification probability 0-255. Default: chemistry preset -- 248 for Hia5 Nanopore (--seq nanopore, given or detected), 128 otherwise. R/Y- and MD-encoded DAF input is binary and ignores it. |
| `--min-read-length` | `1000` | Min aligned read length (default 1000 — matches fiberhmm-apply) |
| `--msp-min-size` | `0` | Min MSP size (default 0) |
| `--nuc-min-size` | `85` | Min footprint size to count as nucleosome (default 85) |
| `--with-scores` / `--scores` | off | Write the HMM posterior-mean nq score of baseline nucleosomes (with --no-recall-nucs; nucleosome recall writes its own LLR-based nq). No aq is written. --scores is the fiberhmm-apply spelling. |
| `-r` / `--circular` | off | Enable circular molecule mode (3x tile internally, emit wrapped MA/AQ/AN annotations). |
| `--process-unmapped` / `--no-process-unmapped` | auto | Call unmapped reads that carry SEQ + MM/ML. Default: automatic -- on for stdin, unindexed and unaligned (uBAM) input, off (pass-through) for indexed aligned BAMs. A run that skips >90% of records as unmapped fails unless --no-process-unmapped is given. |
| `--primary` / `--no-primary` | on | Call primary alignments only (default); secondary and supplementary records are passed through uncalled. --no-primary also calls them. Hard-clipped records whose MM/ML cannot match SEQ are always skipped (hard_clipped_mm). |
| `--min-llr` | — | Native LLR cost per TF interval in joint decoding (default: enzyme preset; not a calibrated FDR threshold). |
| `--min-opps` | `3` | Min informative target positions per TF call (default 3). |
| `--unify-threshold` | `90` | v2 nucs with nl < this may be demoted to tf+ (default 90). |
| `--emission-uplift` | — | Emission power transform. Default: enzyme preset. |
| `--use-m5c` / `--no-use-m5c` | auto | DddA CpG-aware recall, as in fiberhmm-recall-tfs: CpG observations are excluded from nucleosome and TF recall except inside confident unmethylated island calls (MA ddda_ucg from fiberhmm-tag-m5c) the input already carries. Default: on for --enzyme ddda, off otherwise; --no-use-m5c for an ablation. |
| `--cpg-mask-policy` | `unmethylated-only` | With CpG-aware recall: keep CpGs only inside ddda_ucg islands (default), or mask only ddda_mcg spans (the former behaviour). Choices: `unmethylated-only`, `methylated-only`. |
| `--no-legacy-tags` | off | Skip ns/nl/as/al, emit only MA/AQ. |
| `--downstream-compat` | off | Skip MA/AQ; write TF calls into legacy ns/nl track. |
| `--recall-nucs` / `--no-recall-nucs` | auto | Split over-merged nucleosomes + resolve platform-aware edges (emits nuc.QQQ), promote nucleosome-sized TF leaks to nuc, and run the Pass-2 phase prior. ON by default for all enzymes (DddA uses phase-aware radial inference, others the accessible-cut Kadane split). Use --no-recall-nucs for baseline HMM nucleosomes (nuc.Q). |
| `--split-min-llr` | `4.0` | Min accessible-run LLR to split a nucleosome; for DddA, the molecule-local linker-residue configuration LLR (default 4.0). |
| `--split-min-opps` | `3` | Min informative positions in a nucleosome-splitting cut or DddA linker residue (default 3). |
| `--ddda-derived-tf-max-edge-gap` | `12` | DddA phase-aware radial recall only: TF calls exposed solely by nucleosome refinement must have a deamination hit within BP on both sides (default 12). Original HMM-accessible TF scan space is unchanged. Use -1 to disable the safeguard. |
| `--nuc-recall-policy` | `auto` | Nucleosome-recaller geometry policy. "auto" (default) uses topology-constrained, ambiguity-preserving recall for Nanopore and the conservative-edge policy otherwise. "topology" only accepts cuts that leave nucleosome-sized pieces and does not turn unresolved edge ambiguity into accessibility. Choices: `auto`, `conservative`, `topology`. |
| `--phase-nrl` | `auto` | Pass-2 periodicity prior (with --recall-nucs): "auto" (default; estimate the nucleosome repeat length from this sample after Pass 1, clamped to ~150-215 bp anchored at 185), "off", or a fixed bp value (e.g. 185). Long footprints are split at phase-predicted linkers using a lowered threshold gated on >=1 local deamination event (never splits a signal-desert). |
| `--keep-chimeras` | off | DAF only: keep strand-swap chimeric reads (C->T in one segment + G->A in another). Default: filter them out and report the count. |
| `--chimera-min-seg` | `5` | DAF chimera: min same-strand deamination events per segment to call a swap (default 5). |
| `--chimera-purity` | `0.8` | DAF chimera: min same-strand purity per segment (default 0.8). |
| `--daf-mask-runs` | — | DAF only: thin targets lying in same-strand runs of >= N original C (CT) or G (GA) bases (CC/GG and longer at N=2; see --daf-run-policy). Adjacent conversions are coupled and do not follow the per-site emission model. Default: 2 with keep-one for --enzyme ddda (duplex-validated), off otherwise; 0 disables. |
| `--daf-run-policy` | `keep-one` | With --daf-mask-runs: keep the 5'-most target of each run (keep-one, default) or remove the whole run (drop). Choices: `keep-one`, `drop`. |
| `--daf-snp-mask` | — | DAF only: 0-based BED of recurrent C>T/G>A SNP sites to exclude from deamination observations. MD is preserved. |
| `--daf-call-snps` | auto | DAF only: force two-pass recurrent opposite-conversion SNP masking. By default file-based DddA/DddB runs screen automatically after a bounded depth preflight. |
| `--no-daf-call-snps` | auto | Disable automatic recurrent SNP screening. |
| `--daf-snp-output-prefix` | — | Output prefix for --daf-call-snps (default: qc/<output BAM stem>.daf_snps). |
| `--daf-snp-min-fraction` | `0.2` | Minimum opposite-direction fiber fraction for --daf-call-snps (validated default 0.2). |
| `--daf-snp-min-depth` | `5` | Minimum fiber depth in each conversion-direction class for --daf-call-snps (validated default 5; all thresholds must pass in both classes). |
| `--daf-snp-min-alt-fibers` | `5` | Minimum alternate fibers in each direction for --daf-call-snps (validated default 5). |
| `--daf-snp-min-dominant-events` | `5` | Minimum dominant conversions to classify a fiber for --daf-call-snps (default 5). |
| `--daf-snp-min-dominant-purity` | `0.8` | Minimum conversion-direction purity for --daf-call-snps (default 0.80). |
| `--daf-snp-min-amplicon-reads` | `20` | Minimum aligned reads required to discover and plot an amplicon consensus in SNP QC (default 20). |
| `--ddda-mcg` | off | Deprecated integrated per-CpG mode; retained only to emit a clear migration error. Run fiberhmm-call, then fiberhmm-tag-m5c (whole CpG islands), then fiberhmm-recall-tfs --use-m5c. |
| `--dedup` | auto | DAF (ddda/dddb) only: force PCR duplicate detection (already automatic for file-based DddA/DddB calls). Detect by deamination-pattern fingerprint (see fiberhmm-dedup) and similar alignment ends BEFORE footprinting. The integrated default is nondestructive: retain every read and set 0x400 plus di/ds cluster tags. Amplicon/UMI-less DAF libraries can be heavily PCR-duplicated and coordinate dedup does not apply. Requires a file input (not stdin). Ignored for fiber-seq (hia5). |
| `--no-dedup` | auto | Disable automatic DddA/DddB duplicate marking. |
| `--dedup-min-jaccard` | `0.95` | Deamination-set Jaccard threshold for --dedup (default 0.95). |
| `--dedup-flag-only` | off | Deprecated compatibility spelling for the nondestructive integrated default (0x400 + di/ds; retain every read). |
| `--dedup-collapse` | off | With --dedup: destructively collapse each duplicate cluster to one representative. Default: mark and retain all reads. |
| `--dedup-min-deam` | `10` | With --dedup: reads with fewer deamination calls are not fingerprinted and pass through (default 10). |
| `--dedup-prob-threshold` | — | With --dedup: min ML probability for MM/ML-native dU calls, 0-255 (default: the calling --prob-threshold, 128). R/Y and MD inputs are binary and ignore it. |
| `--dedup-ignore-strand` | off | With --dedup: cluster reads across deamination flavours (C->T with G->A reads). Default: only reads of the same flavour can be duplicates. |
| `--dedup-max-end-diff` | `50` | With --dedup: maximum difference at both aligned reference ends for duplicate matching (default 50 bp). |
| `--dedup-stats-tsv` | — | With --dedup: write a cluster_id<TAB>n_reads table. |
| `-c` / `--cores` | `4` | Worker processes (0 = all CPUs; default 4). |
| `--chunk-size` | `500` | Reads per worker chunk (default 500; streaming mode only). |
| `--io-threads` | `8` | htslib I/O threads per stage (default 8). |
| `--max-reads` | `0` | 0 = no limit (default; streaming mode only) |
| `--qc` / `--no-qc` | on | Run bounded signal/periodicity/footprint QC after a file output (default: on; use --no-qc to disable). |
| `--qc-sample-reads` | `2000` | Target random-window QC sample size (default 2000). |
| `--qc-seed` | `20260824` | Deterministic QC sampler seed (default 20260824). |
| `--qc-min-mapq` | `20` | Minimum mapping quality for the QC sample (default 20). |
| `--qc-output-prefix` | — | QC output prefix (default: qc/<BAM stem> beside output BAM). |
| `--region-parallel` | off | Process genomic regions in parallel (one worker per region). Scales linearly with --cores up to chromosome count. Requires coordinate-sorted + indexed input BAM. Recommended for full-genome runs; use streaming for stdin/unaligned. |
| `--region-size` | `10000000` | Region size in bp for --region-parallel (default 10 Mb). |
| `--skip-scaffolds` | off | Skip scaffold/contig chromosomes in region-parallel mode. |
| `--chroms` | — | Only process these chromosomes (region-parallel mode). |

### fiberhmm-apply

| Flag | Default | Description |
|------|---------|-------------|
| `-i` / `--input` | required | Input BAM file with modification calls (must be indexed) |
| `-m` / `--model` | — | Path to trained HMM model (.json, .npz, or .pickle). If omitted, the bundled model for --enzyme/--seq is used. |
| `-o` / `--outdir` | required | Output directory, or "-" to write BAM to stdout (for piping) |
| `--enzyme` | — | Auto-select a supported bundled chemistry model. Use --seq pacbio\|nanopore for Hia5. Choices: `ddda`, `dddb`, `hia5`. |
| `--seq` | — | Hia5 sequencing platform. When omitted it is detected from the input (MM specs: PacBio T-a vs Nanopore A+a only; header records); conflicting evidence stops the run. Ignored for dddb/ddda. Choices: `pacbio`, `nanopore`. |
| `-k` / `--context-size` | — | Context size (auto-detected from model if not specified) |
| `--cores` / `-c` | `1` | Number of CPU cores (0=auto, default: 1) |
| `--io-threads` | `4` | Number of htslib decompression/compression threads for BAM I/O (default: 4) |
| `--streaming` | off | Use streaming pipeline mode (works with unaligned/unindexed BAMs and stdin). Recommended for unaligned data or when reading from pipes. |
| `--chunk-size` | `500` | Reads per compute chunk in streaming mode (default: 500) |
| `--min-mapq` / `-q` | `0` | Minimum mapping quality; reads below this are written to output unchanged without footprint/nucleosome tags. Default 0 (call on all mapped reads). Pass a positive value to filter. |
| `--prob-threshold` | — | Minimum MM/ML probability (0-255) to call a modification. Default: chemistry preset -- 248 for Hia5 Nanopore (--seq nanopore, given or detected), 128 otherwise. R/Y- and MD-encoded DAF input is binary and ignores it. |
| `--min-read-length` | `1000` | Minimum aligned read length in bp; shorter reads are written to output unchanged without footprint/nucleosome tags. Set to 0 to attempt calling on all reads regardless of length (default: 1000) |
| `-t` / `--train-reads` | — | TSV file of read IDs used in training (to exclude) |
| `--primary` / `--no-primary` | on | Call primary alignments only (default); secondary and supplementary records are passed through uncalled. --no-primary also calls them. Hard-clipped records whose MM/ML cannot match SEQ are always skipped (hard_clipped_mm). |
| `--process-unmapped` / `--no-process-unmapped` | auto | Process unmapped reads that have sequences and modification tags. Default: automatic -- on for stdin, unindexed and unaligned (uBAM) input. A run that skips >90% of records as unmapped fails unless --no-process-unmapped is given. |
| `--edge-trim` / `-e` | `10` | Bases to trim from read edges (default: 10) |
| `-r` / `--circular` | off | Enable circular mode (tiles reads 3x) |
| `--scores` | off | Compute per-footprint confidence scores (slower but more informative) |
| `--msp-min-size` | `0` | Minimum size for MSP regions in bp. Default 0 (emit every accessible run; matches fibertools, which does not impose an MSP size filter at this stage). Pass a positive value to filter. |
| `--nuc-min-size` | `85` | Minimum footprint size (bp) to count as nucleosome-sized for MSP boundary detection. Only footprints >= this size split MSPs; smaller footprints are absorbed (default: 85) |
| `--no-msps` | off | Do not write MSP tags (as/al/aq) to output BAM. Useful for Fiber-seq where MSPs are computed differently by fibertools |
| `--stats` | off | Generate summary statistics and plots |
| `--stats-sample` | `10000` | Number of reads to sample for statistics (default: 10000) |
| `--stats-seed` | `42` | Random seed for sampling (default: 42) |
| `--output-posteriors` | — | Export HMM posteriors to file (H5 or TSV) |
| `--debug-timing` | off | Show per-read timing breakdown |
| `--daf-mask-runs` | — | Thin DAF targets in same-strand runs of >= N original C (CT) or G (GA) bases (N=2: CC/GG and longer). Default: 2 with keep-one for DddA (duplex-validated), off otherwise; 0 disables. |
| `--daf-run-policy` | `keep-one` | With --daf-mask-runs: keep each run's 5'-most target (default) or drop the run. Choices: `keep-one`, `drop`. |

### fiberhmm-recall-tfs

| Flag | Default | Description |
|------|---------|-------------|
| `-i` / `--in-bam` | required | Input BAM tagged by fiberhmm-apply (has ns/nl/as/al). Use "-" for stdin. |
| `-o` / `--out-bam` | required | Output BAM with MA/AQ + refreshed legacy tags. Use "-" for stdout (for piping to ft fire, samtools, etc). |
| `-m` / `--model` | — | FiberHMM model JSON. If omitted, the bundled model for --enzyme/--seq is used automatically. |
| `--enzyme` | — | Enzyme preset: auto-selects the bundled model and min-llr/emission-uplift defaults (ddda, dddb, hia5). Choices: `ddda`, `dddb`, `hia5`. |
| `--seq` | — | Hia5 sequencing platform. When omitted it is taken from the input's FIBERHMM-CHEMISTRY declaration or detected from its MM specs (PacBio T-a vs Nanopore A+a only); conflicting evidence stops the run. Ignored for dddb/ddda. Choices: `pacbio`, `nanopore`. |
| `--replace-chemistry` | off | Replace, instead of reconcile with, the input BAM's FIBERHMM-CHEMISTRY declaration. By default a custom --model inherits the input's enzyme/platform when its observation mode matches, and a conflicting --enzyme/--seq is refused. |
| `--daf-mask-runs` | — | DAF only: thin targets lying in same-strand runs of >= N original C (CT) or G (GA) bases (CC/GG and longer at N=2; see --daf-run-policy). Adjacent conversions are coupled and do not follow the per-site emission model. Default: 2 with keep-one for --enzyme ddda (duplex-validated), off otherwise; 0 disables. |
| `--daf-run-policy` | `keep-one` | With --daf-mask-runs: keep each run's 5'-most target (default) or drop the run. Choices: `keep-one`, `drop`. |
| `--min-llr` | — | Override native LLR cost per TF interval in joint decoding (nats; default: enzyme preset; not an FDR threshold). |
| `--prob-threshold` | — | Min MM/ML probability 0-255 for re-reading modification calls. Default: chemistry preset -- 248 for Hia5 Nanopore (--seq nanopore, or the input's declared chemistry), 125 otherwise. R/Y- and MD-encoded DAF input is binary and ignores it. |
| `--min-opps` | `3` | Min informative target positions per call (default 3) |
| `--emission-uplift` | — | Power transform on emission probabilities. Default 1.0 (identity). Use a pre-uplifted model file (e.g. ddda_TF.json) for DddA rather than setting this. |
| `--use-m5c` / `--no-use-m5c` | auto | Enable CpG-aware DddA recall. By default, retain CpGs only inside confident ddda_ucg MA spans. Enabled automatically for DddA, disabled for other enzymes; use --use-m5c explicitly with a custom DddA model. |
| `--cpg-mask-policy` | `unmethylated-only` | CpG-aware policy: retain CpGs only inside confident unmethylated islands (default), or reproduce the former behavior that masks only ddda_mcg spans. Choices: `unmethylated-only`, `methylated-only`. |
| `--unify-threshold` | `90` | v2 nucs with nl < this are scanned + may be demoted to tf+ if overlapped by a recaller call (default 90) |
| `--input-frame` | `auto` | Coordinate frame of the input ns/nl/as/al tags. "auto" (default) detects the @CO fiberhmm:coord=molecular marker: present -> molecular (current FiberHMM), absent -> query/seq (legacy v1.0). Force with molecular/query. Wrong frame mis-places reverse-strand calls. Choices: `auto`, `molecular`, `query`. |
| `--no-legacy-tags` | off | Skip refreshed ns/nl/as/al -- emit only MA/AQ. |
| `--downstream-compat` | off | Downstream-compatibility mode: skip MA/AQ entirely and write TF calls INTO the legacy ns/nl tag alongside nucleosomes (sorted by start). Use for older tools that do not understand the Molecular-annotation spec. Loses per-TF quality scoring (tq/el/er) -- positions and lengths only. |
| `-c` / `--cores` | `1` | Worker processes (0 = all CPUs; default 1). |
| `--chunk-size` | `1024` | Reads per worker chunk (default 1024). Larger values reduce IPC overhead; decrease if RAM is constrained (each chunk holds reads in memory). |
| `--io-threads` | `4` | htslib BAM compression threads (default 4). |
| `--context-size` | — | Override context size. Default: read from model. |
| `--max-reads` | `0` | 0 = no limit (default) |
| `--recall-nucs` / `--no-recall-nucs` | off | Enable nucleosome recall before TF recall. (Default on for fiberhmm-recall-nucs.) |
| `--split-min-llr` | `4.0` | Min accessible-cut LLR to split a footprint; for DddA, the molecule-local linker-residue configuration LLR (default 4.0) |
| `--split-min-opps` | `3` | Min informative positions for a split cut or DddA linker residue (default 3) |
| `--ddda-derived-tf-max-edge-gap` | `12` | DddA phase-aware radial recall only: require TF scan space opened solely by nucleosome refinement to have a deamination hit within BP on both sides (default 12; -1 disables). |
| `--nuc-recall-policy` | `auto` | "auto" uses topology-constrained, ambiguity-preserving recall for Nanopore and conservative edges otherwise. Choices: `auto`, `conservative`, `topology`. |
| `--nuc-min-size` | `85` | Min refined nucleosome size; smaller footprints are demoted to accessible/MSP (default 85) |
| `--msp-min-size` | `0` | Min re-derived MSP size to keep (default 0) |
| `--phase-nrl` | `auto` | Pass-2 periodicity prior: off / auto / fixed bp. "auto" (default) estimates the nucleosome repeat length from the input BAM's existing nuc tags (no HMM re-run). Lowers the split bar near phase-predicted linkers in long footprints. |

### fiberhmm-recall-nucs

| Flag | Default | Description |
|------|---------|-------------|
| `-i` / `--in-bam` | required | Input BAM tagged by fiberhmm-apply (has ns/nl/as/al). Use "-" for stdin. |
| `-o` / `--out-bam` | required | Output BAM with MA/AQ + refreshed legacy tags. Use "-" for stdout (for piping to ft fire, samtools, etc). |
| `-m` / `--model` | — | FiberHMM model JSON. If omitted, the bundled model for --enzyme/--seq is used automatically. |
| `--enzyme` | — | Enzyme preset: auto-selects the bundled model and min-llr/emission-uplift defaults (ddda, dddb, hia5). Choices: `ddda`, `dddb`, `hia5`. |
| `--seq` | — | Hia5 sequencing platform. When omitted it is taken from the input's FIBERHMM-CHEMISTRY declaration or detected from its MM specs (PacBio T-a vs Nanopore A+a only); conflicting evidence stops the run. Ignored for dddb/ddda. Choices: `pacbio`, `nanopore`. |
| `--replace-chemistry` | off | Replace, instead of reconcile with, the input BAM's FIBERHMM-CHEMISTRY declaration. By default a custom --model inherits the input's enzyme/platform when its observation mode matches, and a conflicting --enzyme/--seq is refused. |
| `--daf-mask-runs` | — | DAF only: thin targets lying in same-strand runs of >= N original C (CT) or G (GA) bases (CC/GG and longer at N=2; see --daf-run-policy). Adjacent conversions are coupled and do not follow the per-site emission model. Default: 2 with keep-one for --enzyme ddda (duplex-validated), off otherwise; 0 disables. |
| `--daf-run-policy` | `keep-one` | With --daf-mask-runs: keep each run's 5'-most target (default) or drop the run. Choices: `keep-one`, `drop`. |
| `--min-llr` | — | Override native LLR cost per TF interval in joint decoding (nats; default: enzyme preset; not an FDR threshold). |
| `--prob-threshold` | — | Min MM/ML probability 0-255 for re-reading modification calls. Default: chemistry preset -- 248 for Hia5 Nanopore (--seq nanopore, or the input's declared chemistry), 125 otherwise. R/Y- and MD-encoded DAF input is binary and ignores it. |
| `--min-opps` | `3` | Min informative target positions per call (default 3) |
| `--emission-uplift` | — | Power transform on emission probabilities. Default 1.0 (identity). Use a pre-uplifted model file (e.g. ddda_TF.json) for DddA rather than setting this. |
| `--use-m5c` / `--no-use-m5c` | auto | Enable CpG-aware DddA recall. By default, retain CpGs only inside confident ddda_ucg MA spans. Enabled automatically for DddA, disabled for other enzymes; use --use-m5c explicitly with a custom DddA model. |
| `--cpg-mask-policy` | `unmethylated-only` | CpG-aware policy: retain CpGs only inside confident unmethylated islands (default), or reproduce the former behavior that masks only ddda_mcg spans. Choices: `unmethylated-only`, `methylated-only`. |
| `--unify-threshold` | `90` | v2 nucs with nl < this are scanned + may be demoted to tf+ if overlapped by a recaller call (default 90) |
| `--input-frame` | `auto` | Coordinate frame of the input ns/nl/as/al tags. "auto" (default) detects the @CO fiberhmm:coord=molecular marker: present -> molecular (current FiberHMM), absent -> query/seq (legacy v1.0). Force with molecular/query. Wrong frame mis-places reverse-strand calls. Choices: `auto`, `molecular`, `query`. |
| `--no-legacy-tags` | off | Skip refreshed ns/nl/as/al -- emit only MA/AQ. |
| `--downstream-compat` | off | Downstream-compatibility mode: skip MA/AQ entirely and write TF calls INTO the legacy ns/nl tag alongside nucleosomes (sorted by start). Use for older tools that do not understand the Molecular-annotation spec. Loses per-TF quality scoring (tq/el/er) -- positions and lengths only. |
| `-c` / `--cores` | `1` | Worker processes (0 = all CPUs; default 1). |
| `--chunk-size` | `1024` | Reads per worker chunk (default 1024). Larger values reduce IPC overhead; decrease if RAM is constrained (each chunk holds reads in memory). |
| `--io-threads` | `4` | htslib BAM compression threads (default 4). |
| `--context-size` | — | Override context size. Default: read from model. |
| `--max-reads` | `0` | 0 = no limit (default) |
| `--recall-nucs` / `--no-recall-nucs` | on | Enable nucleosome recall before TF recall. (Default on for fiberhmm-recall-nucs.) |
| `--split-min-llr` | `4.0` | Min accessible-cut LLR to split a footprint; for DddA, the molecule-local linker-residue configuration LLR (default 4.0) |
| `--split-min-opps` | `3` | Min informative positions for a split cut or DddA linker residue (default 3) |
| `--ddda-derived-tf-max-edge-gap` | `12` | DddA phase-aware radial recall only: require TF scan space opened solely by nucleosome refinement to have a deamination hit within BP on both sides (default 12; -1 disables). |
| `--nuc-recall-policy` | `auto` | "auto" uses topology-constrained, ambiguity-preserving recall for Nanopore and conservative edges otherwise. Choices: `auto`, `conservative`, `topology`. |
| `--nuc-min-size` | `85` | Min refined nucleosome size; smaller footprints are demoted to accessible/MSP (default 85) |
| `--msp-min-size` | `0` | Min re-derived MSP size to keep (default 0) |
| `--phase-nrl` | `auto` | Pass-2 periodicity prior: off / auto / fixed bp. "auto" (default) estimates the nucleosome repeat length from the input BAM's existing nuc tags (no HMM re-run). Lowers the split bar near phase-predicted linkers in long footprints. |

### fiberhmm-qc

| Flag | Default | Description |
|------|---------|-------------|
| `-i` / `--input` | required | One or more FiberHMM-compatible BAM/CRAM files; -i may be repeated |
| `-o` / `--output-dir` | — | QC output directory (default: qc/ beside the input BAMs) |
| `--mode` | `auto` | Observation mode (default: infer from @PG/tags) Choices: `auto`, `daf`, `pacbio-fiber`, `nanopore-fiber`. |
| `--enzyme` | `auto` | Enzyme (default: infer from @PG) Choices: `auto`, `hia5`, `ddda`, `dddb`. |
| `--reference-profile` | `auto` | Empirical QC reference (default: assay-aware auto selection) Choices: `auto`, `none`, `ddda`, `dddb`, `hia5_nanopore`, `hia5_pacbio`. |
| `--reference` | — | Indexed FASTA fallback for raw DAF BAMs lacking MD/R/Y |
| `--sample-reads` | `2000` | Target bounded sample size (default 2,000) |
| `--seed` | `20260824` | Deterministic sampler seed (default 20260824) |
| `--min-mapq` | `20` | Minimum mapping quality (default 20) |
| `--prob-threshold` | — | Minimum MM/ML probability, 0-255 (default: 248 for Hia5 Nanopore, the threshold its QC reference is calibrated at; 125 otherwise) |
| `--min-opportunities` | `200` | Minimum target sites per read for rate QC (default 200) |
| `--snp-mask` | — | Applied DAF SNP-mask BED to summarize (single input only) |
| `--snp-report` | — | fiberhmm-daf-snps JSON to plot (single input only) |
| `--fail-on-qc` | off | Exit 2 when the final status is FAIL |

### fiberhmm-extract

| Flag | Default | Description |
|------|---------|-------------|
| `-i` / `--input` | required | Input tagged BAM file |
| `-o` / `--outdir` | — | Output directory (default: same as input) |
| `-c` / `--cores` | `1` | Number of CPU cores |
| `--nucleosome` / `--footprint` | off | Extract nucleosomes (ns/nl or MA nuc tags). (--footprint is a deprecated alias.) |
| `--msp` | off | Extract MSPs (as/al tags) |
| `--tf` | off | Extract TF/Pol II footprints from MA/AQ tag (tf.QQQ). Requires a BAM produced by fiberhmm-recall-tfs in default (spec) mode. See --min-tq for the quality floor. |
| `--min-tq` | `50` | Minimum TF quality (tq) to extract. 0-255 scale where tq = min(255, round(LLR * 10)). Default 50 (LLR >= 5 nats, ~148:1 likelihood ratio); enzyme presets may use different call floors (for example DddA uses 70). Set to 0 for every emitted call, 100+ for a stricter view. |
| `--m6a` | off | Extract m6A positions |
| `--m5c` | off | Extract DddA MA ddda_mcg spans, or native MM/ML 5mC positions |
| `--deam` | off | Extract DAF-seq deamination calls. Priority: (1) MM/ML-native dU calls (mod code "u" or ChEBI 55797); (2) IUPAC R/Y codes in the query sequence (fiberhmm-daf-encode output); (3) MD-tag ref mismatches as a fallback for raw DAF BAMs. First non-empty source wins per read. blockMod: 0 = R/GA-dea, 1 = Y/CT-dea, matching FiberBrowser flavor codes. |
| `--both-strand` / `--bothstrand` | off | Extract the deam+ / deam- intersection of paired DddA consensus reads as a BED12 coverage overlay. |
| `--all` | off | Extract all tag types (default if none specified) |
| `--bed-only` | off | Output BED only (no bigBed) |
| `--keep-bed` | off | Keep BED files when creating bigBed |
| `-q` / `--min-mapq` | `0` | Min mapping quality (default: 0, no filtering) |
| `-p` / `--prob-threshold` | — | Min probability for native MM/ML m6a/m5c/dU calls (0-255). Default: 248 when the BAM declares Hia5 Nanopore (FIBERHMM-CHEMISTRY header), 125 otherwise. Not applied to DddA MA ddda_mcg spans or to R/Y/MD deaminations (binary). |
| `--no-scores` | off | Omit scores from output |
| `--block-scores` | off | Append per-block quality as extra BED column(s) (BED12+N). nucleosome -> blockNq/blockEl/blockEr, msp -> blockAq, m6a/native-m5c -> blockMl (DddA ddda_mcg spans -> 0), tf -> blockTq/blockEl/blockEr. bigBed uses -type=bed12+N and the matching autoSQL schema so FiberBrowser/UCSC can surface per-feature quality without a sidecar database. |
| `--circular-groups` | off | Append circId/circPart/circParts/molStart/molLength columns and preserve MA/AN groups for circular wrapped nucleosome, MSP, and TF features. |
| `--haplotype-fields` | off | Append scalar hp and ps columns copied from the source BAM HP/PS tags after every other optional BED field. Missing or non-integer tags are written as -1. Off by default to preserve the existing BED/bigBed schema byte-for-byte. |
| `--sample-name` | — | Sample/dataset identifier to embed in the autoSQL description of every output bigBed ("Sample: <name>. ..."). Default: BAM basename stem. Lets downstream tools match a bigBed to its source without filename parsing. |
| `--region-size` | `10000000` | Region size for parallel |
| `--skip-scaffolds` | off | Skip scaffold chromosomes |
| `--chroms` | — | Comma-separated chromosomes to process |
| `--sort-mem` / `-S` | `1G` | Memory buffer for the BED sort, passed to `sort -S` (e.g. 4G, 8G, or 50% on GNU sort). Bigger = fewer temp-file merge passes = faster. Default 1G; pass an empty string to disable. The sort also runs under LC_ALL=C regardless, which alone is a large speedup. |
| `--sort-parallel` | `0` | Threads for the BED sort (GNU coreutils only; ignored on BSD/macOS sort). 0 = use --cores. Feature-detected, so safe to leave on. |

### fiberhmm-dedup

| Flag | Default | Description |
|------|---------|-------------|
| `-i` / `--input` | required | Input DAF-seq BAM (indexed not required) |
| `-o` / `--output` | required | Output BAM |
| `--min-jaccard` | `0.95` | Min deamination-set Jaccard to call two reads the same molecule (default 0.95; the bimodal gap sits ~0.90-0.95). Lower = more aggressive collapsing. |
| `--flag-only` | off | Keep all reads and only mark duplicates (set the 0x400 duplicate flag + di/ds tags on non-representatives). Default: collapse each cluster to one representative read. |
| `--min-deam` | `10` | Reads with fewer than this many deamination calls are not fingerprintable; passed through untouched (default 10). |
| `--max-end-diff` | `50` | Maximum difference at both aligned reference ends for two reads to be duplicates (default 50 bp). |
| `--ignore-strand` | off | Cluster across deamination flavours. Default: only reads with the same dominant flavour (C->T vs G->A, i.e. the same template strand) can be duplicates; alignment orientation is never used. |
| `-p` / `--prob-threshold` | `128` | Min ML probability for MM/ML-native dU calls (0-255; default 128, the same as fiberhmm-call). R/Y- and MD-encoded input is binary and ignores it. |
| `--num-hashes` | `32` | MinHash signature width (default 32). |
| `--bands` | `8` | LSH bands; rows = num-hashes / bands (default 8 -> rows 4). More bands = higher recall, more candidate pairs. |
| `--seed` | `7` | MinHash RNG seed (default 7). |
| `--stats-tsv` | — | Write a cluster_id<TAB>n_reads table to this path. |
| `--io-threads` | `4` | htslib BAM compression threads for output (default 4). |

### fiberhmm-daf-encode

| Flag | Default | Description |
|------|---------|-------------|
| `-i` / `--input` | required | Input BAM file, or "-" for stdin |
| `-o` / `--output` | required | Output BAM path, or "-" for stdout |
| `--reference` | — | Reference FASTA (fallback if MD tag is missing) |
| `-q` / `--min-mapq` | `20` | Minimum mapping quality (default: 20) |
| `--min-read-length` | `1000` | Minimum aligned read length in bp (default: 1000) |
| `--io-threads` | `4` | htslib I/O threads for BAM compression/decompression (default: 4) |
| `--strand` | `auto` | Force conversion strand: CT (+ strand), GA (- strand), or auto (per-read consensus, default: auto) Choices: `CT`, `GA`, `auto`. |

### fiberhmm-daf-snps

| Flag | Default | Description |
|------|---------|-------------|
| `-i` / `--input` | required | Input aligned BAM with MD tags |
| `-o` / `--output-prefix` | — | Output prefix (default: <BAM stem>.daf_snps) |
| `--min-fraction` | `0.2` | Minimum mismatch fraction in each direction (validated default 0.2) |
| `--min-depth` | `5` | Minimum depth in each conversion-direction class (validated default 5; all thresholds must pass in both classes) |
| `--min-alt-fibers` | `5` | Minimum mismatch-supporting fibers in each direction (validated default 5) |
| `--min-dominant-events` | `5` | Minimum dominant conversions per classifiable fiber (default 5) |
| `--min-dominant-purity` | `0.8` | Minimum dominant-direction purity (default 0.80) |
| `--min-mapq` | `20` | Minimum mapping quality (default 20) |
| `--min-amplicon-reads` | `20` | Minimum aligned reads required for an amplicon consensus (default 20) |
| `--reference` | — | Indexed FASTA fallback for BAMs lacking MD |

### fiberhmm-pair

| Flag | Default | Description |
|------|---------|-------------|
| `-i` / `--input` | required | Coordinate-sorted, indexed FiberHMM-called DddA BAM |
| `-o` / `--output` | required | Output BAM: joint duplex molecules by default; paired source reads with --stop-after pair |
| `-r` / `--reference` | — | Matching indexed FASTA. Required by the default sequence-free score; optional with --sequence-only when MD+CIGAR is available |
| `--sequence-only` | off | Accept only direct A/T sequence-supported pairs; disable the sequence-free model |
| `--pairs-tsv` | — | Write selected-pair evidence to this TSV |
| `--receipt-json` | — | Write a machine-readable pairing receipt |
| `--pairs-only` / `--paired-only` | off | Write only paired records: merged duplex molecules (default), or paired source reads with --stop-after pair. Otherwise unpaired reads pass through unchanged. |
| `--stop-after` | `recall` | Last stage to run: pair (tag pairs only), merge (both-strand molecules without re-calling) or recall (default) Choices: `pair`, `merge`, `recall`. |
| `--from-paired` | off | Input is already pair-tagged (mt/mp from an earlier --stop-after pair run): skip pairing and start at merge. Pairing-stage options (--reference, --sequence-only, --pairs-tsv, --model, --min-* ...) are rejected here. |
| `--model` | — | Override the bundled frozen sequence-free model JSON |
| `--call-layer` | `auto` | Nucleosome calibration for the sequence-free model (default auto) Choices: `auto`, `input-ma`, `rotational-recall`. |
| `--min-margin` | `1.0` | Minimum two-sided sequence-free model margin (default 1.0) |
| `--null-floor` | `0.0` | Virtual null model score for a lone candidate (default 0.0) |
| `--min-overlap` | `1500` | Min genomic overlap bp (default 1500) |
| `--min-nucs` | `4` | Min nucleosome dyads within the overlap, each read (default 4) |
| `--min-sequence-bases` | `500` | Min shared reference-A/T bases for a sequence edge (default 500) |
| `--min-component-discordance-rate` | `0.02` | Min rejected-edge difference rate to constrain a 2x2 (default 0.02) |
| `--max-sequence-pair-rate` | `0.01` | Max difference rate on a sequence-selected pair (default 0.01) |
| `--min-sequence-margin` | `0.002` | Min sequence preference/assignment margin (default 0.002) |
| `-p` / `--prob-threshold` | `128` | Min ML probability for MM/ML-native dU calls (0-255; default 128, the same as fiberhmm-call). R/Y- and MD-encoded input is binary and ignores it. |
| `--max-component` | `10000` | Safety ceiling for a complete overlap component (default 10000) |
| `--io-threads` | `4` | htslib compression threads for output (default 4) |
| `--no-index` | off | Do not index a paired-source output |
| `--phase-nrl` | `196` | Nucleosome repeat length for consensus recall (default 196) |
| `--nuc-recall-policy` | `conservative` | Nucleosome policy for consensus recall Choices: `conservative`, `topology`. |
| `--ddda-derived-tf-max-edge-gap` | `12` | Edge-evidence requirement for TF calls exposed only by DddA nucleosome refinement (default 12; -1 disables) |
| `--use-m5c` / `--no-use-m5c` | auto | Joint recall: DddA CpG-aware recall, as in fiberhmm-call and fiberhmm-recall-tfs -- CpG observations are excluded except inside the source reads' ddda_ucg islands (fiberhmm-tag-m5c). Default: on; --no-use-m5c for an ablation. |
| `--daf-mask-runs` | — | Thin DAF targets in same-strand runs of >= N original C (CT) or G (GA) bases (N=2: CC/GG and longer). Default: 2 with keep-one for DddA (duplex-validated), off otherwise; 0 disables. |
| `--daf-run-policy` | `keep-one` | With --daf-mask-runs: keep each run's 5'-most target (default) or drop the run. Choices: `keep-one`, `drop`. |

### fiberhmm-merge

| Flag | Default | Description |
|------|---------|-------------|
| `-i` / `--input` | required | BAM from fiberhmm-pair (mt/mp tags) |
| `-o` / `--output` | required | Output consensus BAM (sorted + indexed) |
| `--pairs-only` | off | Emit only consensus reads (default: also pass through unmerged reads) |
| `--recall` | off | Re-call footprints on each both-strand consensus read (HMM layer over both strands; writes ns/nl/as/al + MA nuc./msp.). Reads the deam+/deam- regime and uses C and G targets jointly. |
| `--enzyme` | `ddda` | Model preset for --recall (default ddda) |
| `--phase-nrl` | `196` | Nucleosome repeat length for consensus recall (default 196) |
| `--nuc-recall-policy` | `conservative` | Nucleosome geometry policy for consensus recall Choices: `conservative`, `topology`. |
| `--ddda-derived-tf-max-edge-gap` | `12` | With --recall, require TF calls exposed solely by DddA radial nucleosome refinement to have a deamination hit within BP on both sides (default 12; -1 disables). |
| `--use-m5c` / `--no-use-m5c` | auto | With --recall: DddA CpG-aware recall, as in fiberhmm-call and fiberhmm-recall-tfs (CpGs excluded except inside the source reads' ddda_ucg islands). Default: on for --enzyme ddda. |
| `-p` / `--prob-threshold` | `128` | Min ML probability for MM/ML-native dU calls (0-255; default 128, the same as fiberhmm-call). R/Y- and MD-encoded input is binary and ignores it. |
| `--io-threads` | `4` | htslib compression threads (default 4) |
| `--daf-mask-runs` | — | Thin DAF targets in same-strand runs of >= N original C (CT) or G (GA) bases (N=2: CC/GG and longer). Default: 2 with keep-one for DddA (duplex-validated), off otherwise; 0 disables. |
| `--daf-run-policy` | `keep-one` | With --daf-mask-runs: keep each run's 5'-most target (default) or drop the run. Choices: `keep-one`, `drop`. |

### fiberhmm-tag-m5c

| Flag | Default | Description |
|------|---------|-------------|
| `-i` / `--input` | required | R/Y-encoded, FiberHMM-tagged coordinate BAM |
| `-o` / `--output` | required | Output BAM |
| `-r` / `--reference` | required | Reference FASTA |
| `--enzyme` | required | Required chemistry assertion. Only DddA DAF-seq is supported. Choices: `ddda`. |
| `--posterior` | `0.99` |  |
| `--min-other` | `10` |  |
| `--cpg-islands` | — | Optional BED3 of merged, non-overlapping CpG islands. Default: infer islands from the reference sequence. |
| `--write-cpg-islands` | — | Optional BED or BED.GZ containing the exact island catalog used. |
| `--min-island-cpg` | `15` | Minimum CpG observations for a whole-island call (default 15). |
| `--calls-tsv` | — | Optional whole-island audit table containing methylated, unmethylated and uninformative molecule/island overlaps. |
| `--cpg-island-window` | `200` | Reference inference window in bp (default 200). |
| `--cpg-island-step` | `10` | Reference inference step in bp (default 10). |
| `--cpg-island-min-gc` | `0.5` | Minimum GC fraction for inferred islands (default 0.50). |
| `--cpg-island-min-oe` | `0.6` | Minimum CpG observed/expected for inferred islands (default 0.60). |
| `--five-prime-factors` | — | Comma-separated A,C,G,T factors; default calibrated DddA values |
| `--estimate-factors` | off | Estimate 5' factors from this BAM instead of using calibrated values |
| `--factor-sample-reads` | `5000` |  |
| `--input-frame` | `auto` | Frame of legacy ns/nl tags when MA is absent; auto uses the FiberHMM header marker Choices: `auto`, `molecular`, `query`. |
| `--io-threads` | `4` |  |

### fiberhmm-call-m5c

| Flag | Default | Description |
|------|---------|-------------|
| `-i` / `--input` | required | One or more coordinate-sorted, indexed DAF FiberHMM BAMs |
| `-r` / `--reference` | required | Indexed reference FASTA |
| `-o` / `--output` | required | Output BED6, or - for stdout |
| `--region` | required | contig[:start-end], 1-based display syntax |
| `--enzyme` | required | Required chemistry assertion. Only DddA DAF-seq is supported. Choices: `ddda`. |
| `--window` | `1000` |  |
| `--chunk-bp` | `5000000` | Bound observation collection in this many bp; the HMM still runs once across the complete region |
| `--min-other` | `10` |  |
| `--min-cpg` | `10` |  |
| `--posterior` | `0.99` |  |
| `--max-gap` | `1000` |  |
| `--five-prime-factors` | — | Comma-separated A,C,G,T factors; default calibrated DddA values |
| `--estimate-factors` | off | Estimate 5' factors from the complete supplied region (explicit high-memory audit option) |
| `--tag-bam` | — | Optionally add whole-island annotations to this BAM |
| `--tag-output` | — | Output BAM for --tag-bam (required with --tag-bam) |
| `--tag-mode` | `island` | island: one molecule state per complete CpG island (default); locus: copy aggregate validation domains Choices: `island`, `locus`. |
| `--cpg-islands` | — | Optional merged BED3 island catalog; default: infer from reference |
| `--write-cpg-islands` | — | Optional BED/BED.GZ recording the exact catalog used |
| `--read-posterior` | `0.99` | Whole-island posterior threshold (default: 0.99) |
| `--read-min-island-cpg` | `15` | Minimum CpGs per molecule-island call (default: 15) |
| `--tag-input-frame` | `auto` | Frame of tag-BAM legacy ns/nl when MA is absent Choices: `auto`, `molecular`, `query`. |
| `--io-threads` | `4` |  |

### fiberhmm-consensus

| Flag | Default | Description |
|------|---------|-------------|
| `--schema` | off | Print every parameter group with defaults and help as JSON (cr.engine selects the engine; the default is lattice_recaller) |
| `--bam` | — | Repeat for separate datasets; chemistry comes from BAM @CO metadata |
| `--datasets` | — | JSON list of {dataset_id, paths: [BAMs], chemistry?: profile} |
| `--evidence` | — | Saved evidence.json.gz, including pooled evidence |
| `--resume` | — | Previous run directory containing evidence.json.gz, manifest.json and fit_cache |
| `--bed` | — | BED3 windows for independent runs; BED6 with equal widths for CL-CR |
| `--region` | — | Alternative CHROM:START-END, explicitly 0-based half-open |
| `--pool-loci` | off | CL-CR: pool BED6 windows in their provided orientations |
| `--chemistry` | — | Explicit missing-metadata declaration for --bam; conflicts fail Choices: `ddda`, `dddb`, `hia5-pacbio`, `hia5-nanopore`. |
| `--parameters` | — | JSON parameter groups; see --schema |
| `--engine` | — | Consensus engine (default lattice_recaller; staged_native_families is the deprecated Monte Carlo engine). Overrides cr.engine from --parameters or --resume Choices: `lattice_recaller`, `staged_native_families`. |
| `--consolidation-bp` | — | staged_native_families only: shared-family edge allowance (default 10; 5 gives finer grouping) |
| `--stop-after` | — | staged_native_families only: last stage to compute (the lattice recaller runs in one pass) Choices: `native`, `parents`, `consolidated`, `resolved`. |
| `--start-at` | `native` | staged_native_families only: consolidation restarts from saved native fits Choices: `native`, `consolidation`. |
| `--cores` | — |  |
| `--cache` | — | staged_native_families only: persistent exact native-fit cache directory |
| `--json-progress` | off | Structured progress on stderr |
| `--daf-mask-runs` | — | DAF only: thin targets in same-strand runs of >= N original C (CT) or G (GA) bases in lattices and native replay (2 = CC/GG and longer; 0 = off). Default: per dataset chemistry, DddA keep-one on runs >= 2 (duplex-validated), DddB off |
| `--daf-run-policy` | `keep-one` | With --daf-mask-runs: keep each run's 5'-most target (default) or drop the run Choices: `keep-one`, `drop`. |
| `--no-bam` | off | Save frozen results/reports without materializing family-tagged BAMs |
| `--bam-scope` | `regions` | Export whole alignments overlapping analyzed windows (default), or the full source BAM Choices: `regions`, `full`. |
| `--bam-grouping` | `datasets` | One BAM per logical dataset (default) or original source file Choices: `datasets`, `files`. |
| `--bam-recaller-layer` | off | lattice_recaller: also write the optional tf_recaller MA layer (the recaller's own per-molecule class calls at every prevalence tier; bytes tq,fi,tier,q0,lr,rr) to exported BAMs. Off by default; the calls are always in result.json.gz |
| `--output` | — | New or empty result directory |

### fiberhmm-transfer

| Flag | Default | Description |
|------|---------|-------------|
| `--freeze-run` | — | Completed consensus result directory: a lattice_recaller run (any frame) or an oriented staged_native_families CL-CR run; export its classes/models without refitting |
| `--models` | — | frozen_classes.json.gz (lattice_recaller), frozen_models.json.gz (staged), or the --freeze-run output directory |
| `--bam` | — |  |
| `--datasets` | — | JSON list of dataset_id and BAM paths |
| `--evidence` | — | Saved oriented native evidence.json.gz (one window or pooled evidence) |
| `--bed` | — | Equal-width, explicitly oriented BED6 target windows |
| `--chemistry` | — | Choices: `ddda`, `dddb`, `hia5-pacbio`, `hia5-nanopore`. |
| `--parameters` | — | BAM preparation parameters; target calls are replayed, families are never fitted |
| `--chip-bed` | — | Optional independent ChIP peaks, joined only after scoring |
| `--no-bam` | off |  |
| `--bam-scope` | `regions` | Choices: `regions`, `full`. |
| `--bam-grouping` | `datasets` | Choices: `datasets`, `files`. |
| `--json-progress` | off |  |
| `--cores` | — | Worker processes for scoring (default: this machine's consensus default) |
| `--dataset-map` | — | Use the frozen per-channel boxes and spots of source dataset SOURCE for target dataset TARGET (needed when several source datasets share the target chemistry) |
| `--include-training-molecules` | off | Score molecules the catalog was trained on (default: exclude them, as for staged families); use for self-application checks |
| `--output` | required | New or empty output directory |

### fiberhmm-footprint-model

| Flag | Default | Description |
|------|---------|-------------|
| `-i` / `--input` | required | Recalled BAM containing ordinary tf and msp MA groups |
| `-o` / `--output-prefix` | required | Output filename prefix (not a directory) |
| `--region` | — | Analyze a zero-based, half-open region; repeatable and requires a BAM index. A bare CONTIG selects that complete contig. |
| `--genome` | — | Genome/assembly label recorded in provenance (for example dm6) |
| `--genomewide` | off | Assert that scanning the complete BAM represents a genome-wide analysis |
| `--bigbed` | off | Also create indexed population and FiberBrowser BigBeds |
| `--bed-to-bigbed` | — | Path to UCSC bedToBigBed; implies --bigbed |
| `--force` | off | Replace existing artifacts for this output prefix |
| `-q` / `--min-mapq` | `0` | Minimum alignment MAPQ (default: 0) |
| `--include-duplicates` | off | Include records carrying the BAM duplicate flag (excluded by default) |
| `--smoothing-sigma` | `3.0` | Footprint-center Gaussian sigma in bp (default: 3) |
| `--peak-distance` | `15` | Minimum distance between center modes (default: 15) |
| `--assignment-radius` | `10` | Maximum center-to-mode assignment radius (default: 10) |
| `--edge-compatibility` | `12` | Maximum within-family diameter for each boundary (default: 12) |
| `--minimum-geometry-support-per-stratum` | `3` | Molecules per stratum needed for that stratum to vote on canonical geometry (default: 3) |
| `--minimum-geometry-support` | `3` | Descriptive geometry-ready threshold (default: 3) |
| `--minimum-population-support` | `3` | Descriptive population-ready threshold (default: 3) |
| `--minimum-mapped-fraction` | `0.95` | Required mapped fraction for geometry, MSP projection, and site denominators (default: 0.95) |

### fiberhmm-strand-rescue

| Flag | Default | Description |
|------|---------|-------------|
| `-i` / `--bam` | required | Post-TF/post-nuc BAM; repeat only to pool shards or compatible timepoints from one inference cohort |
| `--preset` | required | Choices: `ddda`, `dddb`, `hia5-nanopore`. |
| `--region` | required |  |
| `--model` | — |  |
| `--nuc-model` | — | Protected/accessibility model used only for nucleosome edge evidence |
| `--prob-threshold` | — |  |
| `--control-flank` | `2000` |  |
| `--min-mapq` | `20` |  |
| `--min-support` | `10` |  |
| `--minimum-geometry-support` | `3` |  |
| `--source-boundary-margin` | `10` |  |
| `--center-radius` | `10` |  |
| `--peak-distance` | `15` |  |
| `--tf-edge-compatibility` | `12` | Maximum within-family diameter in bp for each TF boundary; co-centered calls outside this bound form distinct TF models |
| `--max-boundary-mad` | `12.0` |  |
| `--min-local-enrichment` | `2.0` |  |
| `--local-background-radius` | `250` |  |
| `--max-auto-sites` | `0` | Optional TF-family cap after scoring; 0 keeps the exhaustive locus map |
| `--site` | — | Externally seed one zero-based half-open START-END site; repeatable. Support and canonical edges are recomputed from ordinary cohort calls. |
| `--forced-sites-only` | off | Analyze only --site geometries rather than unioning automatic sites |
| `--nuc-min-support` | — | Required ordinary nuc calls on one strand (default: --min-support) |
| `--nuc-center-radius` | `25` |  |
| `--nuc-source-boundary-margin` | `20` |  |
| `--nuc-edge-assignment-radius` | `48` |  |
| `--nuc-max-boundary-mad` | `24.0` |  |
| `--max-auto-nuc-sites` | `0` | Optional nucleosome-family cap after scoring; 0 keeps all families |
| `--nuc-site` | — | Seed one zero-based half-open nucleosome population; repeatable. Edges are relearned from ordinary nuc calls and no length ceiling applies. |
| `--forced-nuc-sites-only` | off | Refine only --nuc-site populations rather than automatic nuc sites |
| `--skip-nuc-edge-refinement` | on | Do not run independent one-for-one nucleosome edge normalization (default; retained for command-line compatibility) |
| `--independent-nuc-edge-refinement` | off | Experimental population nucleosome-edge normalization. This is not the TF-conditioned consensus-nuc reconciliation stage. |
| `--strand-min-source-support` | — | Required source-strand ordinary TF calls (default: --min-support) |
| `--strand-min-source-enrichment` | `1.5` | Minimum source-strand focal enrichment over local background |
| `--strong-posterior` | `0.95` |  |
| `--review-posterior` | `0.5` |  |
| `--tf-class-pseudocount` | `0.5` | Symmetric per-configuration pseudocount for localized single/composite site-consensus priors |
| `--tf-class-locus-gap` | `30` | Maximum gap joining atomic footprint states into one consensus locus |
| `--tf-class-max-span` | `250` | Maximum span in bp of one localized site-consensus locus |
| `--tf-class-max-sites` | `10` | Maximum atomic states enumerated in a complete site-consensus action set; larger loci are reported as skipped |
| `--maximum-sites-per-decision` | `8` |  |
| `--accessible-site-gap` | `220` | Maximum gap joining TF sites into one MSP-origin decision |
| `--control-shifts` | `` | Optional comma-separated target-coordinate shifts. Source priors remain at the true sites; controls are diagnostic, not an FDR null. |
| `--max-reads` | `0` |  |
| `--molecule-collapse` | `auto` | Collapse amplified DAF PCR families; auto enables for DddA/DddB Choices: `auto`, `on`, `off`. |
| `--molecule-min-jaccard` | `0.95` |  |
| `--molecule-min-deam` | `10` |  |
| `--per-molecule-efficiency` | on | Calibrate hard-call efficiency from each molecule's MSPs (default) |
| `--global-efficiency` | off | Use the model-wide accessible hard-call rate for every molecule |
| `--efficiency-pseudo-count` | `20.0` |  |
| `--efficiency-min-opportunities` | `20` |  |
| `--report-layout` | `auto` | Report action storage: auto spills at the bounded v4 limits; stream forces v5 BGZF action sidecars Choices: `auto`, `inline`, `stream`. |
| `--diagnostics` | `aggregate` | Per-call diagnostic storage (aggregate is the production default) Choices: `aggregate`, `stream`. |
| `--proposal-tsv` | — |  |
| `-o` / `--output` | required |  |

### fiberhmm-strand-rescue-annotate

| Flag | Default | Description |
|------|---------|-------------|
| `--report` | required |  |
| `-i` / `--bam` | — | Report input BAM to materialize; default is every report BAM |
| `-o` / `--output` | — | Output BAM; requires exactly one selected input |
| `--output-dir` | — | Directory receiving one regional BAM per selected input |
| `--region` | — |  |
| `--minimum-posterior` | `0.0` | Optional output-size filter; default 0 preserves every decision |
| `--allow-input-drift` | off |  |
| `--io-threads` | `1` |  |

### fiberhmm-strand-rescue-audit

| Flag | Default | Description |
|------|---------|-------------|
| `-i` / `--bam` | required |  |
| `-o` / `--output` | — |  |
| `--max-errors` | `100` |  |

### fiberhmm-tag-consensus

| Flag | Default | Description |
|------|---------|-------------|
| `-i` / `--input` | required | Input BAM with tf_sr.QQQ |
| `-o` / `--output` | required | New sorted/indexed BAM |
| `-a` / `--assignments` | required | v1/v2 family assignment TSV |
| `--force` | off | Replace an existing output |

### fiberhmm-posteriors

| Flag | Default | Description |
|------|---------|-------------|
| `-i` / `--input` | required | Input BAM file |
| `-m` / `--model` | — | FiberHMM model file. If omitted, uses the bundled model for --enzyme/--seq. |
| `-o` / `--output` | required | Output file (.tsv.gz for TSV, .h5/.hdf5 for HDF5) |
| `--format` | `auto` | Output format (default: auto-detect from extension) (default: auto) Choices: `auto`, `hdf5`, `tsv`. |
| `--enzyme` | — | Auto-select a bundled enzyme model. (default: None) Choices: `ddda`, `dddb`, `hia5`. |
| `--seq` | — | Hia5 platform; omission warns and defaults to pacbio. Ignored for dddb/ddda. (default: None) Choices: `pacbio`, `nanopore`. |
| `--edge-trim` / `-e` | `10` | Bases to trim from read edges (default: 10) (default: 10) |
| `--prob-threshold` | `128` | Min ML probability (0-255) for an MM/ML modification call (same default as fiberhmm-call) (default: 128) |
| `--keep-chimeras` | off | Do not drop DAF strand-swap chimeric reads |
| `--chimera-min-seg` | `5` | DAF chimera: min same-strand deamination events per segment (default: 5) |
| `--chimera-purity` | `0.8` | DAF chimera: min same-strand purity per segment (default: 0.8) |
| `--daf-snp-mask` | — | 0-based BED of reference positions whose conversions are ignored (e.g. the mask fiberhmm-call used) |
| `--daf-mask-runs` | — | Thin DAF targets in same-strand runs of >= N original C (CT) or G (GA) bases (N=2: CC/GG and longer). Default: 2 with keep-one for DddA (duplex-validated), off otherwise; 0 disables. |
| `--daf-run-policy` | `keep-one` | With --daf-mask-runs: keep each run's 5'-most target (default) or drop the run. (default: keep-one) Choices: `keep-one`, `drop`. |
| `--cores` / `-c` | `4` | Number of CPU cores (0=auto, default: 4) (default: 4) |
| `--region-size` | `5000000` | Region size in bp for parallel processing (default: 5,000,000) (default: 5000000) |
| `--skip-scaffolds` | off | Skip scaffold/contig chromosomes |
| `--chroms` | — | Only process these chromosomes |
| `--io-threads` | `4` | Number of htslib decompression/compression threads for BAM I/O (default: 4) (default: 4) |
| `--streaming` | off | Use streaming pipeline mode (works with unaligned/unindexed BAMs and stdin). Recommended for unaligned data or when reading from pipes. |
| `--chunk-size` | `500` | Reads per compute chunk in streaming mode (default: 500) (default: 500) |
| `--batch-size` | `1000` | Fibers per HDF5 write batch (default: 1000) |
| `-v` / `--verbose` | off | Verbose output |

### fiberhmm-probs

| Flag | Default | Description |
|------|---------|-------------|
| `--accessible` / `-a` | required | BAM file(s) from accessible/naked DNA (dechromatinized, MTase-treated) |
| `--inaccessible` / `-u` | required | BAM file(s) from inaccessible/untreated samples (native chromatin) |
| `-o` / `--output` | required | Output file prefix (will create _accessible.tsv and _inaccessible.tsv) |
| `-k` / `--context-sizes` | `3 4 5 6` | Context size(s) to compute (bases on each side: 3=7mer, 6=13mer). Single value or list. Default: 3 4 5 6 (default: [3, 4, 5, 6]) |
| `--mode` | `pacbio-fiber` | Analysis mode: pacbio-fiber (PacBio), nanopore-fiber (Nanopore), daf (DAF-seq) (default: pacbio-fiber) Choices: `pacbio-fiber`, `nanopore-fiber`, `daf`, `gpc`, `cpg`. |
| `-n` / `--max-reads` | `100000` | Maximum reads to process per sample type (0 = all) (default: 100000) |
| `-s` / `--seed` | `42` | Random seed for sampling (default: 42) |
| `-q` / `--min-mapq` | `20` | Minimum mapping quality (default: 20) |
| `-p` / `--prob-threshold` | `128` | Minimum ML probability for modification call (0-255) (default: 128) |
| `--min-read-length` | `1000` | Minimum aligned read length (default: 1000) |
| `-e` / `--edge-trim` | `10` | Bases to exclude at read edges (default: 10) |
| `--save-interval` | `10000` | Save intermediate results every N reads (default: 10000) |
| `--stats` | off | Generate summary statistics and QC plots |
| `--verbose` / `-v` | off | Show detailed filter statistics per BAM file |

### fiberhmm-train

| Flag | Default | Description |
|------|---------|-------------|
| `-i` / `--input` | — | Input BAM file(s) for training (not required with --base-model) |
| `-p` / `--probs` | required | Accessible and inaccessible probability files (.tsv or .probs.pkl) |
| `--base-model` | — | Use transitions from existing model with new emissions (skip training) |
| `-o` / `--outdir` | required | Output directory |
| `--mode` | `pacbio-fiber` | Analysis mode: pacbio-fiber (PacBio), nanopore-fiber (Nanopore), daf (DAF-seq) (default: pacbio-fiber) Choices: `pacbio-fiber`, `nanopore-fiber`, `daf`, `gpc`, `cpg`. |
| `-k` / `--context-size` | `3` | Context size (bases on each side): 3=7mer, 5=11mer, etc. (default: 3) |
| `-c` / `--iterations` | `10` | Training iterations (random initializations) (default: 10) |
| `-r` / `--read-count` | `500` | Total reads to sample for training (default: 500) |
| `-s` / `--seed` | `42` | Random seed (default: 42) |
| `-e` / `--edge-trim` | `10` | Edge masking (default: 10) |
| `-q` / `--min-mapq` | `20` | Min mapping quality (default: 20) |
| `--prob-threshold` | `125` | Min ML probability (default matches ft-extract) (default: 125) |
| `--min-read-length` | `1000` | Min aligned length (default: 1000) |
| `-a` / `--prob-adjust` | `1.0` | Accessible probability adjustment factor (default: 1.0) |
| `--use-hmmlearn` | off | Use hmmlearn instead of native implementation (for legacy compatibility) |
| `--stats` | off | Generate training statistics and example plots |
| `--n-examples` | `5` | Number of example reads to plot (with --stats) (default: 5) |
| `--daf-mask-runs` | — | Thin DAF targets in same-strand runs of >= N original C (CT) or G (GA) bases (N=2: CC/GG and longer). Default: 2 with keep-one for DddA (duplex-validated), off otherwise; 0 disables. |
| `--daf-run-policy` | `keep-one` | With --daf-mask-runs: keep each run's 5'-most target (default) or drop the run. (default: keep-one) Choices: `keep-one`, `drop`. |

### fiberhmm-utils

#### fiberhmm-utils convert

| Flag | Default | Description |
|------|---------|-------------|
| `input` | required | Input model file (.pickle, .pkl, or .npz) |
| `output` | required | Output JSON file |

#### fiberhmm-utils inspect

| Flag | Default | Description |
|------|---------|-------------|
| `model` | required | Model file to inspect (.json, .npz, .pickle) |
| `--full` | off | Print full emission probability table |

#### fiberhmm-utils transfer

| Flag | Default | Description |
|------|---------|-------------|
| `--target` / `-t` | required | Target BAM file (e.g., DAF-seq) |
| `-o` / `--output` | required | Output directory |
| `--mode` | `daf` | Analysis mode for target data (default: daf) Choices: `pacbio-fiber`, `nanopore-fiber`, `daf`, `gpc`, `cpg`. |
| `--reference-bam` / `-rb` | — | Reference BAM with footprint tags (ns/nl) |
| `--accessibility-priors` / `-ap` | — | Pre-computed P(accessible\|context) TSV |
| `-k` / `--context-sizes` | `3 4 5 6` | Context size(s) (default: [3, 4, 5, 6]) |
| `-n` / `--max-reads` | `100000` | Max reads to process (0 = all) (default: 100000) |
| `-q` / `--min-mapq` | `20` | Min mapping quality (default: 20) |
| `-p` / `--prob-threshold` | `128` | Min ML probability for modification call (default: 128) |
| `--min-read-length` | `1000` | Min aligned read length (default: 1000) |
| `-e` / `--edge-trim` | `10` | Bases to exclude at read edges (default: 10) |
| `--min-observations` | `100` | Min observations per context for regression (default: 100) |
| `--stats` | off | Generate diagnostic plots |

#### fiberhmm-utils adjust

| Flag | Default | Description |
|------|---------|-------------|
| `model` | required | Input model file (.json) |
| `--state` | required | Which state(s) to adjust Choices: `accessible`, `inaccessible`, `both`. |
| `--scale` | required | Multiplier for emission probabilities |
| `-o` / `--output` | required | Output model file (.json) |

#### fiberhmm-utils ma-types

| Flag | Default | Description |
|------|---------|-------------|
| `bam` | required | BAM to update in place |
| `--types` | — | Logical MA name(s), comma- or space-separated (no strand/quality suffixes) |
| `--scan` | off | Exhaustively scan every alignment and discover non-empty MA types |
| `--io-threads` | `4` | BAM compression and index threads (default: 4) |

#### fiberhmm-utils fix-bigbed

| Flag | Default | Description |
|------|---------|-------------|
| `inputs` | required | Input bigBed file(s) |
| `--sample-name` | — | Explicit sample name to embed (sanitized to a dot/space-free token). Default: derived per file from the filename (stem minus the _<layer> suffix). |
| `--in-place` | off | Overwrite the input bigBed(s) in place. |
| `-o` / `--output` | — | Output path (single input only). Default: write <name>.fixed.bb alongside each input. |

<!-- END GENERATED CLI REFERENCE -->
