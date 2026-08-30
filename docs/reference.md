# FiberHMM reference

Deep reference for the FiberHMM tag schema, scoring model, and output formats.
For installation and day-to-day usage see the [README](../README.md).

- [Analysis modes](#analysis-modes)
- [BAM tag glossary](#bam-tag-glossary)
- [MA/AQ molecular-annotation schema](#maaq-molecular-annotation-schema)
- [MA type discovery header](#ma-type-discovery-header)
- [Quality bytes: tq / el / er](#quality-bytes-tq--el--er)
- [The log-likelihood-ratio recaller](#the-log-likelihood-ratio-recaller)
- [recall-tfs output modes](#recall-tfs-output-modes)
- [Circular molecules](#circular-molecules)
- [Reading the output](#reading-the-output)

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

## BAM tag glossary

`fiberhmm-apply` (and the apply stage of `fiberhmm-call`) write fibertools-style
legacy tags:

| Tag | Type | Description |
|-----|------|-------------|
| `ns` | B,I | Nucleosome/footprint starts (0-based query coords) |
| `nl` | B,I | Nucleosome/footprint lengths |
| `as` | B,I | Accessible/MSP starts |
| `al` | B,I | Accessible/MSP lengths |
| `nq` | B,C | Footprint quality scores (0–255, with `--scores`) |
| `aq` | B,C | MSP quality scores (0–255, with `--scores`) |

The TF recaller (`fiberhmm-recall-tfs`, or the recall stage of `fiberhmm-call`)
adds Molecular-annotation [spec](https://github.com/fiberseq/Molecular-annotation-spec)
tags carrying TF/Pol II footprints with full LLR scoring:

| Tag | Type | Description |
|-----|------|-------------|
| `MA` | Z | Annotation string: `<readlen>;nuc.Q:...;msp.:...;tf.QQQ:...;ddda_mcg.:...` (plus the strand-resolved DddA and optional strand-rescue groups below; 1-based coords per spec) |
| `AQ` | B,C | Quality bytes interleaved per annotation: `nq` for nucs; `tq, el, er` for ordinary TFs; three bytes for each normalized strand-rescue nuc or TF (no bytes for MSPs) |

Legacy `ns`/`nl`/`as`/`al` are rewritten to reflect the unified call set (v2
short-nucs absorbed into TF calls are removed). **TF calls live only in
`MA`/`AQ`** — by design FiberHMM does not invent non-spec nucleosome-track tag
names. Use `--no-legacy-tags` to skip the legacy refresh and emit only `MA`/`AQ`.

`fiberhmm-daf-encode` (optional) additionally writes:

| Tag | Type | Description |
|-----|------|-------------|
| `st` | Z | Conversion strand: `CT` (+ strand, C→T) or `GA` (− strand, G→A) |

The DAF one-pass path in `fiberhmm-call` does **not** write `st:Z` — it
derives strand internally from MD and doesn't modify the stored sequence.

`fiberhmm-dedup` writes:

| Tag | Type | Description |
|-----|------|-------------|
| `di` | i | Duplicate-cluster id |
| `ds` | i | Cluster size (number of PCR copies the read represents) |

## MA/AQ molecular-annotation schema

The recaller writes one `MA` tag and one `AQ` tag per processed read:

```
MA:Z:<read_length>;nuc.Q:s1-l1,s2-l2,...;msp.:s1-l1,...;tf.QQQ:s1-l1,...;ddda_mcg.:s1-l1,...
AQ:B:C: nq, nq, ..., tq, el, er, tq, el, er, ...
```

Coordinates are 1-based (per the spec); internal storage stays 0-based.

| Annotation | Quality bytes | Meaning |
|---|---|---|
| `nuc.Q` | `nq` | Nucleosomes (`nl ≥ unify_threshold`, or v2 short-nucs the recaller did not match). `nq` carries v2's posterior mean (0 sentinel for unverified entries). |
| `msp.` | none | Methylase-sensitive patches (v2 MSPs unchanged) |
| `tf.QQQ` | `tq, el, er` | Recaller TF calls (see below). |
| `ddda_mcg.` | none | Conservative molecule-specific methylated-CpG runs inferred from DddA deamination contrast. |
| `ddda_mcg+` / `ddda_mcg-` | none | Methylated runs on the CT/reference-C or GA/reference-G channel of a merged cross-strand DAF molecule. |
| `ddda_mcg_hemi+` / `ddda_mcg_hemi-` | none | High-confidence hemimethylated runs where both channels are observed; the qualifier identifies the methylated channel. |
| `nuc_sr.QQQ` | `SR alternative, molecular-left edge, molecular-right edge` | Complete optional nucleosome shadow layer from `fiberhmm-strand-rescue-annotate`: exactly one same-class call per ordinary nuc, with one-for-one shared edges where accepted. Identity and cardinality are fixed; SR cannot reclassify, split, merge, promote, demote, create, or remove a nuc. |
| `tf_sr.QQQ` | `SR alternative, molecular-left edge, molecular-right edge` | Complete optional TF shadow layer: one call per ordinary TF plus atomic weak-positive MSP-to-TF rescues. Accepted ordinary TFs may receive shared same-class edges. |

The recalled nucleosome track from the nucleosome recaller is `nuc.QQQ` =
`(nq, el, er)` — same byte layout as `tf.QQQ`.

For DddA phase-aware radial recall, `nq` scores the topology-changing linker
configuration, not the much easier question of protected DNA versus open
linker. A retained split or inward edge uses the molecule-local
accessible-residue log Bayes factor. Baseline nucleosome refinement receives no
provisional TF calls or population prior; TF recall runs once afterward. An
unresolved boundary or a retained HMM footprint with no radial dyad receives
`nq=0`. This avoids the uninformative saturation produced by scoring a full
protected radial window as though it verified the boundary topology.

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
annotation combinations is specified
separately for FiberBrowser in
[`FIBERBROWSER_CONSENSUS_RECONSTRUCTION.md`](./FIBERBROWSER_CONSENSUS_RECONSTRUCTION.md).

`ddda_mcg.` is a molecular interval annotation, not a native per-base `MM:C+m`
modification call. DAF amplification removes that native channel. The span
marks a run whose observed CpGs jointly support the methylated state; it is
called independently for each read unless the caller's explicit locus mode is
used. The model and its calibration are DddA-specific and must not be applied
to DddB DAF-seq. It is an experimental opt-in for genome-wide DddA data, not a
default stage of the targeted DddA workflow. Enable the integrated path with
`fiberhmm-call --enzyme ddda --ddda-mcg --reference ref.fa`; without that flag,
the ordinary DddA call is unchanged.

Merged cross-strand DAF reads declare CT and GA source coverage with `deam+`
and `deam-` MA groups. `fiberhmm-tag-m5c` detects these groups automatically,
keeps the two non-CpG baselines separate, and runs an equal-odds four-state HMM
(`UU`, `UM`, `MU`, `MM`) over canonical CpG dyads. Strand-resolved mCG may be
called wherever its source channel is observed; `ddda_mcg_hemi+/-` is emitted
only across multiple CpGs with scored evidence from both channels. BAM reverse
alignment flags are not used as chemical-strand identity.

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
emitting MA, the main caller and recall tools advertise `nuc,msp,tf`; DddA mCG
paths additionally advertise `ddda_mcg` and, where cross-strand calling is
possible, `ddda_mcg_hemi`.

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

`fiberhmm-call` emits this line together with its ordinary `@PG` provenance.
Downstream BAM transformations retain it as part of the copied header.

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

For DddA reads carrying a `ddda_mcg.` span, TF recall adjusts only CpG contexts
inside that span. If the fitted accessible-state deamination probability is
`p`, the methylated value is `1 - (1 - p)^(F/U)`, with calibrated
`F/U = 0.167/1.113`. This is a rate-scale correction: protected-state and
non-CpG emissions remain unchanged. Consequently an undeaminated methylated
CpG supplies less false evidence for a footprint, while an observed
deamination is also a less absolute accessible-state veto.

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

- `conservative` is the historical policy. Every qualifying accessible run is
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

## recall-tfs output modes

The recaller supports two mutually-exclusive output modes; pick based on what
your downstream tooling can read. The runtime banner makes the active mode
explicit, and switching is a pure re-run on the same HMM-tagged input.

**Spec mode (default)** — write `MA`/`AQ` tags per the spec:
- `MA`/`AQ` carry `nuc.Q`, `msp.`, `tf.QQQ` with full LLR + edge-ambiguity scoring.
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

## Reading the output

```python
import pysam
from fiberhmm.io.ma_tags import parse_ma_tag, parse_aq_array, tq_to_llr

bam = pysam.AlignmentFile('recalled.bam', 'rb', check_sq=False)
for read in bam:
    if not read.has_tag('MA'):
        continue
    parsed = parse_ma_tag(read.get_tag('MA'))
    aq = read.get_tag('AQ')
    qual_specs = [rt[2] for rt in parsed['raw_types']]
    n_per_type = [len(rt[3]) for rt in parsed['raw_types']]
    per_ann = parse_aq_array(aq, qual_specs, n_per_type)
    # parsed['nuc'] = [(start, length), ...]   -- 0-based query coords
    # parsed['msp'] = [(start, length), ...]
    # parsed['tf']  = [(start, length), ...]
    # per_ann is a flat list, one sublist per annotation in MA order:
    #   first len(nucs) sublists are [nq]
    #   next  len(msps) sublists are []
    #   next  len(tfs)  sublists are [tq, el, er]
```

Or use the legacy tags directly (refreshed by the recaller to reflect the
unified call set):

```python
ns = list(read.get_tag('ns'))   # nuc starts (nucs >=90bp + unmatched short-nucs)
nl = list(read.get_tag('nl'))
as_ = list(read.get_tag('as'))  # MSP starts
al = list(read.get_tag('al'))
```

TF calls live only in `MA`/`AQ` — legacy tags do not carry them, by design.
