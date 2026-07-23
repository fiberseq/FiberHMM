# FiberHMM reference

Deep reference for the FiberHMM tag schema, scoring model, and output formats.
For installation and day-to-day usage see the [README](../README.md).

- [Analysis modes](#analysis-modes)
- [BAM tag glossary](#bam-tag-glossary)
- [MA/AQ molecular-annotation schema](#maaq-molecular-annotation-schema)
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
| `MA` | Z | Annotation string: `<readlen>;nuc.Q:...;msp.:...;tf.QQQ:...` (1-based coords per spec) |
| `AQ` | B,C | Quality bytes interleaved per annotation: `nq` for nucs; `tq, el, er` for TFs (no bytes for MSPs) |

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
MA:Z:<read_length>;nuc.Q:s1-l1,s2-l2,...;msp.:s1-l1,...;tf.QQQ:s1-l1,...
AQ:B:C: nq, nq, ..., tq, el, er, tq, el, er, ...
```

Coordinates are 1-based (per the spec); internal storage stays 0-based.

| Annotation | Quality bytes | Meaning |
|---|---|---|
| `nuc.Q` | `nq` | Nucleosomes (`nl ≥ unify_threshold`, or v2 short-nucs the recaller did not match). `nq` carries v2's posterior mean (0 sentinel for unverified entries). |
| `msp.` | none | Methylase-sensitive patches (v2 MSPs unchanged) |
| `tf.QQQ` | `tq, el, er` | Recaller TF calls (see below). |

The recalled nucleosome track from the nucleosome recaller is `nuc.QQQ` =
`(nq, el, er)` — same byte layout as `tf.QQQ`.

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
the edge-sharpness bytes recover the loose boundary.

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
footprint is divided, after which the positive-sign scan re-estimates each
resulting nucleosome's conservative boundaries and confidence.

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
