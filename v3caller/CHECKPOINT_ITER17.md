# FiberHMM v3 Caller — iter-17 Checkpoint

**Date**: 2026-04-12
**Caller**: `v3caller/caller_v8.py`
**Tests**: `v3caller/tests/test_caller_v8.py` — 62 tests passing
**Key support files**: `v3caller/ma_tags.py`, `v3caller/best_guess.py`

---

## 1. Architecture overview

The caller runs a two-pass pipeline on each long read:

```
Read → Extract (opp, hit) arrays → Pass 1 (nucleosome atoms)
     → Pass 2 (Poisson evidence merge) → TF caller (overcall)
     → Quality scoring → BAM tag emit (legacy + MA)
```

### Pass 1: nucleosome atom detection

Two algorithms, selected via `--first-pass`:

- **`gap_cdf`** (default, recommended): finds every continuous hit-free
  stretch ≥ `gap_radius` bp (default 10). Simpler, faster, lower
  overmerge rate. The benchmark winner across 10 datasets.
- **`protected_runs`**: v7's windowed-rate approach. Computes a rolling
  hit rate in a W=40 bp window, finds stretches where the rate is
  significantly below the read's baseline.

### Pass 2: Poisson evidence merge

Adjacent Pass 1 atoms are merged if the gap between them has a hit
count consistent with the read's baseline under a two-sided Poisson
interval test (central 90%, quantiles 0.05/0.95). Guards:

- **Cascade guard**: never create merged atoms > 250 bp
- **Short-gap structural merge**: gaps ≤ 8 bp always merge (too short
  to be a real linker). mq degrades linearly with absorbed hit count:
  `mq = max(0, 255 - 30 * gap_hit_count)`. Zero-hit gaps → mq=255.
- **Low-power gate**: gaps with < 5 opportunities → default to split

Each merge records a **merge quality (mq)**:
- mq = 255: pure Pass-1 atom, no internal merges
- mq = 255 * (1 - 2 * |CDF(gap_hit | λ) - 0.5|): Poisson-test merge
- Lower mq = less confident merge (gap hit count was toward the
  tails of the interval)

### TF footprint caller (overcall)

Runs on two sources of accessible DNA:
1. **Complement of final nucleosomes** (linker regions)
2. **Merged-over gaps** inside merged atoms (alt-hypothesis: "maybe
   this merge was wrong and there's a TF inside")

Emits every uninterrupted MISS run meeting the configurable floor
filter (`--min-tf-bp 5 --min-tf-tq 10` by default). These defaults
cut pure noise (2-3 bp runs with near-zero significance) while
remaining highly permissive. Set both to 0 for full overcall mode.
Each call carries three quality scores described in §2.3 below.

### Rotational phase correction (iter-17)

The TF significance score includes a per-position correction for
the 10.4 bp helical face-phasing near nucleosome edges. DAF
deaminases (DddA/DddB) are blocked by histone contacts on the
wrapped face, creating a ~50% oscillation in hit rate that extends
~30 bp past each nuc edge.

For each miss at reference position p, the local hit rate is:

```
d = distance from p to nearest nuc edge (bp)
edge_q = quality (0-1) of that nearest nuc edge (lq or rq / 255)
amp = 0.35 * edge_q
rate_profile = 1.0 + amp * cos(2π * d / 10.4) * exp(-d / 15.0)
local_rate = baseline * rate_profile    (clamped to [0.001, 0.999])
```

The corrected P-value is: `P = ∏ (1 - local_rate(d_i))` across
all misses in the run, instead of the uniform `(1 - baseline)^k`.

**Edge-quality dampening**: the correction amplitude is scaled by
the edge quality of the nearest nucleosome boundary. Sharp edges
(lq/rq ≈ 255) get the full calibrated amplitude (0.35); ambiguous
edges (lq/rq < 128) get a proportionally reduced amplitude. This
prevents applying an out-of-phase cosine correction when the nuc
boundary position is uncertain — a 5 bp misplacement would invert
the correction. When edge quality is 0, the correction reverts
entirely to the uniform baseline model.

This correction only fires within 30 bp of a nuc edge. Beyond that,
`rate_profile ≈ 1.0` and the formula reduces to the uniform model.

**Only applies to DAF enzymes** (DddA, DddB). Hia5 passes
`nuc_edges=None` → correction disabled.

Calibrated from 40,000 scDAF reads (PS00758) using edge-anchored
pair-correlation, validated against naked DddB (no periodicity)
and ≥200 bp from-any-nuc control (flat). See
`bench/output/periodicity_edge_vs_far_FINAL.png`.

### Enzyme extraction

The `DAFExtractor` auto-detects four encoding styles per read via
majority vote:
- IUPAC-encoded: C→Y (forward) or G→R (reverse)
- Raw mismatch: C→T (forward) or G→A (reverse)

This lets a single extractor handle any DAF BAM (scDAF with IUPAC
encoding, DddB spacetime with raw mismatches) without per-dataset
configuration.

The `Hia5Extractor` parses MM/ML tags for m6A detection and does
NOT require a reference FASTA or MD tags — the caller passes a
dummy ref_seq for Hia5 reads when neither is available.

---

## 2. BAM tag encoding

The caller writes **two independent tag schemas** simultaneously
(controlled by `--tags {legacy, ma, both}`, default `both`):

### 2.1 Legacy tags (v7-compatible)

These are simple integer arrays in SAM `B:I` format. One array per
annotation type. Lengths of parallel arrays must match.

#### Nucleosomes

| tag | type | description |
|-----|------|-------------|
| `ns` | `B:I` | Nucleosome **start** positions (0-based query coords) |
| `nl` | `B:I` | Nucleosome **lengths** (bp). End = ns[i] + nl[i] |
| `nq` | `B:I` | **Protection quality** (0-255). How deeply the region is protected vs the read's baseline. 255 = near-zero internal hit rate, 0 = hit rate ≈ baseline |
| `mq` | `B:I` | **Merge quality** (0-255). Confidence in the merge step. 255 = pure Pass-1 atom (no merges), lower = borderline merge. See §2.3 for full semantics |
| `lq` | `B:I` | **Left edge quality** (0-255). Distance from the called left boundary to the first internal HIT, scaled linearly over a 50 bp breathing window. 255 = HIT right at boundary (sharp edge), 0 = no HIT within 50 bp (ambiguous edge) |
| `rq` | `B:I` | **Right edge quality** (0-255). Mirror of `lq` for the right boundary |

All six arrays have the same length = number of nucleosomes called.

#### MSPs (accessible regions)

| tag | type | description |
|-----|------|-------------|
| `as` | `B:I` | MSP **start** positions (0-based query coords) |
| `al` | `B:I` | MSP **lengths** (bp) |

MSPs are the "confident" accessible regions: the complement of the
final nucleosome list, PLUS any low-confidence merge gaps (gaps
inside merged atoms with mq < 128 that the merge step absorbed but
that might actually be accessible). MSPs can overlap with
nucleosome spans in query coords when the merge was borderline.

#### TF footprints

| tag | type | description |
|-----|------|-------------|
| `tn` | `B:I` | TF **start** positions (0-based query coords) |
| `tl` | `B:I` | TF **lengths** (bp) |
| `tq` | `B:I` | **Significance quality** (0-255). See §2.3 |
| `el` | `B:I` | **Left edge sharpness** (0-255). 255 = a HIT sits immediately adjacent to the TF's left boundary (within 2 bp). 0 = the TF extends to the MSP boundary with no nearby HIT (ambiguous edge). Linear decay over 10 bp |
| `er` | `B:I` | **Right edge sharpness** (0-255). Mirror of `el` |

All five arrays have the same length = number of TF calls.

TF calls are subject to a configurable floor filter before BAM
emission (`--min-tf-bp 5 --min-tf-tq 10` by default). Set both
to 0 to emit all candidates.

**Important**: TF calls can sit INSIDE nucleosome spans. This
happens when the TF caller scans a merged-over gap (a gap inside a
merged atom). The TF is an "alternative hypothesis" for the
nucleosome call at that position — the caller is saying "this
region was merged into a nucleosome, but there's TF-like signal
inside that questions the merge." TFs never sit inside pure Pass-1
atoms (mq = 255), only inside merged atoms (mq < 255).

### 2.2 Molecular Annotation (MA) tags

Per the fiberseq Molecular-annotation-spec
(https://github.com/fiberseq/Molecular-annotation-spec).

| tag | type | description |
|-----|------|-------------|
| `MA` | `Z` (string) | Annotation string. Format: `<read_length>;<type1>:<intervals1>;<type2>:<intervals2>;...` |
| `AQ` | `B:C` (u8 array) | Quality values, interleaved per annotation in MA order |

#### MA string format

```
<read_length>;nuc+QQQQ:<s1>-<l1>,<s2>-<l2>,...;msp+:<s1>-<l1>,...;tf+QQQ:<s1>-<l1>,...
```

- Coordinates are **1-based** (SAM convention). Internal storage is
  0-based; the writer adds 1 on output, the parser subtracts 1 on
  input.
- Lengths are in bp (same as stored).
- The `+` after each type name indicates forward strand (always `+`
  for our caller — we annotate in query coords).

#### Annotation types

| type | qual spec | meaning |
|------|-----------|---------|
| `nuc+QQQQ` | 4 quality values per annotation | Nucleosome. Quals: (nq, mq, lq, rq) |
| `msp+` | no quality values | Accessible region (MSP) |
| `tf+QQQ` | 3 quality values per annotation | TF footprint. Quals: (tq, el, er) |

#### AQ array layout

The AQ array is a flat `u8` (unsigned byte, 0-255) array. Values
are interleaved per annotation in the same order as they appear in
the MA string:

```
[nq₀, mq₀, lq₀, rq₀,   # nuc 0
 nq₁, mq₁, lq₁, rq₁,   # nuc 1
 ...
 tq₀, el₀, er₀,          # tf 0
 tq₁, el₁, er₁,          # tf 1
 ...]
```

MSPs contribute zero values to AQ (they have no quality spec).

**To parse**: read the MA string to determine the type order, count
annotations per type, and quality-spec length per type. Then walk
the AQ array consuming `len(qual_spec)` values per annotation.

#### Example

```
MA:Z:4503;nuc+QQQQ:7-138,235-140;msp+:1-6,145-90;tf+QQQ:160-25
AQ:B:C:200,255,128,180,  180,230,200,150,  45,204,230
         ^^^^^^^^^^^^     ^^^^^^^^^^^^       ^^^^^^^^^
         nuc 0: nq=200    nuc 1: nq=180     tf 0: tq=45
                mq=255           mq=230            el=204
                lq=128           lq=200            er=230
                rq=180           rq=150
```

### 2.3 Quality score reference

#### nq — nucleosome protection quality (0-255)

How deeply the nucleosome body is protected relative to the read's
baseline hit rate. Computed from the minimum windowed rate inside
the nucleosome:

```
protection = 1.0 - min_rate / baseline
nq = round(255 * clamp(protection, 0, 1))
```

- nq = 255: near-zero internal hit rate (deep protection)
- nq = 128: ~50% of baseline (moderate protection)
- nq = 0: hit rate ≈ baseline (not really protected)

#### mq — merge quality (0-255)

Confidence that the nucleosome was correctly assembled by the merge
step:

- mq = 255: **pure Pass-1 atom** — no internal merges, the entire
  nucleosome came from a single continuous protected stretch. No
  TF calls will ever appear inside.
- mq = 128-254: high-confidence merge. Internal gaps had hit counts
  close to the Poisson median for the read's baseline.
- mq = 1-127: borderline merge. Internal gaps were within the
  Poisson interval but toward the tails. TF calls may appear inside
  these atoms as alternative hypotheses.
- mq = 0: structural merge only (gap was ≤8 bp with many absorbed
  hits).

For short-gap structural merges, mq degrades linearly with absorbed
hit count: `mq = max(0, 255 - 30 * n_hits_absorbed)`.

#### lq, rq — edge ambiguity quality (0-255)

How precisely the nucleosome's boundary is defined:

```
breath_distance = distance from called boundary to first internal HIT
lq (or rq) = round(255 * max(0, 1 - breath_distance / 50))
```

- 255: a HIT sits right at the boundary (sharp, well-defined edge)
- 128: first HIT is ~25 bp inside (moderate breathing)
- 0: no HIT within 50 bp of the boundary (deep breathing zone,
  boundary placement is ambiguous)

#### tq — TF significance quality (0-255)

Statistical significance of the miss run, corrected for rotational
face-phasing near nucleosome edges (DAF enzymes only):

```
For each miss at position p:
  d = distance to nearest nuc edge
  edge_q = quality of that nearest edge (lq or rq, 0-1 scale)
  amp = 0.35 * edge_q
  local_rate = baseline * (1 + amp * cos(2π*d/10.4) * exp(-d/15))
  (clamped to [0.001, 0.999])

P = ∏ (1 - local_rate(d_i))   across all misses
tq = clip(255 * -log10(P) / 3.0, 0, 255)
```

- tq = 255: P ≤ 10⁻³ (extremely unlikely under baseline — strong
  footprint evidence)
- tq = 128: P ≈ 0.03
- tq = 21: P ≈ 0.58 (2 misses at baseline 0.24 — weakest call)
- tq = 0: not used (min_misses = 2 guarantees tq > 0)

For Hia5 or when `nuc_edges` is not available, the formula reduces
to the uniform `P = (1 - baseline)^k`.

#### el, er — TF edge sharpness (0-255)

How well each TF boundary is pinned by an adjacent HIT:

```
d = distance from TF boundary to the nearest HIT in opp positions
el (or er) = round(255 * max(0, 1 - d / 10))
```

- 255: HIT immediately adjacent (within 1-2 bp)
- 128: HIT ~5 bp away
- 0: TF extends to MSP boundary with no nearby HIT (ambiguous,
  the footprint could extend further)

---

## 3. Best-guess filtering

`v3caller/best_guess.py` provides a one-line filter for users who
want "just the confident calls" without reasoning about quality
scores.

### Thresholds (iter-16b, loosened for DAF biology)

| parameter | value | meaning |
|-----------|-------|---------|
| `nuc_mq_min` | **0** | Keep all nucleosomes (mq is still in the tag for post-hoc filtering) |
| `tf_edge_min` | **128** | Both TF edges must be pinned by a HIT |
| `tf_tq_min` (baseline ≥ 0.20) | **22** | ~2 consecutive misses at baseline 0.24 |
| `tf_tq_min` (baseline < 0.20) | **12** | ~3 consecutive misses at baseline 0.10 |

The baseline threshold (0.20) distinguishes high-density DAF
(amplicon/NAPA, scDAF at high enzyme loading) from low-density
single-cell DAF. The per-read baseline is computed by
`best_guess.py` via the DAFExtractor.

### Usage

**CLI**:
```bash
python best_guess.py --in-bam output.bam --out-bam filtered.bam
```

**Programmatic**:
```python
from best_guess import best_guess_calls
calls = best_guess_calls(read, baseline=0.24)
# calls = {'nucs': [(s, l), ...], 'msps': [...], 'tfs': [...]}
```

### Typical per-read counts after filtering

| dataset | baseline | raw nucs | filtered nucs | raw TFs | filtered TFs |
|---------|----------|----------|---------------|---------|--------------|
| NAPA | 0.30 | 18.8 | 18.8 | 24.6 | 19.4 |
| scDAF | 0.18 | 53.8 | 53.8 | 40.5 | 28.6 |
| ENH30 | 0.44 | 13.7 | 13.7 | 13.8 | 10.3 |

---

## 4. Key design decisions and rationale

### 4.1 TFs inside merged nucleosomes (relaxed cascade)

TF calls can appear inside merged nucleosomes (mq < 255) but never
inside pure Pass-1 atoms (mq = 255). This is intentional:

- A merged atom is the caller's BEST GUESS but it might be wrong
- A TF signal inside a merge is an **alternative hypothesis**: "maybe
  this region isn't one big nucleosome, maybe it's two closely-spaced
  TFs"
- Users can resolve the ambiguity downstream via cross-strand rescue,
  aggregate evidence across reads, or positional filtering
- Pure Pass-1 atoms are unambiguously protected (no internal gaps
  were ever present) so no TF can exist there

### 4.2 Rotational phase correction (DAF only)

DAF deaminases (DddA/DddB) are blocked by histone contacts on one
face of wrapped DNA, creating a 10.4 bp oscillation in hit rate
near nucleosome edges. This was measured via:

1. **Naked DddB** (data/dddb/): A ≈ 0 (no periodicity on free helix)
2. **Chromatinized DddB** (spacetime): A ≈ 0.21 (strong oscillation)
3. **Hia5 chromatinized**: smooth monotonic decay, NO oscillation
   (Hia5 accesses both faces even on chromatin)
4. **scDAF edge-anchored**: peak 1.515 at d=+10 from nuc edge,
   flat control (≥200 bp from nuc) at 1.166
5. **Decay range**: ~30 bp from nuc edge

The correction adjusts tq by ~±10-20% for TFs within 30 bp of a
nuc edge: upward for misses on the "exposed face" (genuinely
surprising), downward for misses on the "quiet face" (less
surprising than uniform baseline assumes). Beyond 30 bp, the
correction is a no-op.

**Edge-quality dampening**: the amplitude (0.35) is multiplied by
the nearest nuc edge's quality (lq or rq, scaled to 0-1). This
guards against applying an out-of-phase correction when the nuc
boundary is uncertain — a ~5 bp misplacement would invert the
cosine. Sharp edges get full correction; ambiguous edges revert
to the uniform model.

### 4.3 Dual tag output

Both legacy (ns/nl/nq/mq/...) and MA (MA:Z/AQ:B:C) tags are
written simultaneously. They encode the SAME information in
different formats:

- **Legacy**: simple, fast to parse, compatible with existing tools
  (FiberBrowser, custom scripts). Each quality is a separate tag.
  `mq` is now included as a legacy tag (was MA-only prior to
  iter-17 post-review fixes).
- **MA**: fiberseq spec-compliant, carries all quality information in
  a structured format. Enables interop with Mitch Vollger's tools
  and the broader fiberseq ecosystem.

A single BAM file can carry both without conflict — the tag names
don't collide.

### 4.4 Short-gap merge mq degradation

Structural merges (gaps ≤ 8 bp) always merge but degrade mq based
on absorbed hit count. This prevents labeling a merged atom as
"pure Pass-1" (mq=255) when it absorbed real hits — which would
mislead the TF caller into thinking the atom is unambiguously
protected.

### 4.5 TF emission floor filter

TF calls shorter than `--min-tf-bp` (default 5 bp) or with
`tq < --min-tf-tq` (default 10) are dropped before BAM emission.
This prevents BAM bloat from pure noise: at typical DAF baseline
0.24, the probability of ≥2 consecutive misses by chance is ~58%,
and a 3 bp run with tq=15 carries no biological signal. The floor
is intentionally very permissive — `best_guess.py` handles the
strict downstream filtering. Set both to 0 to restore full overcall.

### 4.6 DAFExtractor auto-encoding detection

The DAFExtractor handles both IUPAC-encoded (R/Y) and raw-mismatch
(C→T / G→A) BAMs via a per-read majority vote across four encoding
styles. This removes the need to pre-encode BAMs with
`fiberhmm-daf-encode` before calling — the caller works directly
on raw aligned BAMs (e.g. DddB spacetime data) as well as
IUPAC-encoded ones (e.g. scDAF, NAPA).

### 4.7 Hia5 without reference FASTA

Hia5 reads use MM/ML tags for m6A detection — the Hia5Extractor
does not use the reference sequence at all. When no FASTA or MD
tags are available, the caller passes a dummy ref_seq for Hia5
reads instead of skipping them. This makes the Hia5 path work
on BAMs that lack MD tags without requiring a reference genome.

---

## 5. CLI reference

```bash
python caller_v8.py \
  --in-bam INPUT.bam \
  --out-bam OUTPUT.bam \
  --fa REFERENCE.fa \          # or '' for MD tags / Hia5 dummy
  --enzyme {daf,hia5} \
  --first-pass {gap_cdf,protected_runs} \  # default: gap_cdf
  --gap-radius 10 \            # gap_cdf only
  --W 40 \                     # scan window width
  --min-read-rate 0.05 \       # skip low-signal reads (0.005 for DddB)
  --max-merge-len 250 \        # cascade guard
  --min-footprint 80 \         # reject merged atoms < 80 bp
  --min-tf-bp 5 \              # TF emission floor: min bp length
  --min-tf-tq 10 \             # TF emission floor: min tq score
  --tags {legacy,ma,both} \    # default: both
  --max-reads 0 \              # 0 = no limit
  --strip-mods                 # drop MM/ML tags to shrink output
```

---

## 6. File inventory

| file | description |
|------|-------------|
| `caller_v8.py` | Main caller (~1160 lines) |
| `caller_v7.py` | v7 primitives (windowed_rate, find_pass1_atoms, etc.) |
| `ma_tags.py` | Pure-Python MA/AQ tag writer + parser |
| `best_guess.py` | Baseline-aware filter for "best guess" calls |
| `enzyme_extractors.py` | DAFExtractor (auto-encoding) + Hia5Extractor |
| `tests/test_caller_v8.py` | 62 unit tests |

### Bench scripts (v3caller/bench/)

| script | purpose |
|--------|---------|
| `run_v8_ma_integration.py` | Regenerate all 10 datasets with iter-17 schema |
| `plot_best_guess_snapshots.py` | Per-read nq/mq/best-guess 3-column snapshots |
| `plot_mq_snapshots.py` | Per-read mq-colored snapshots (v7/v8/gap_cdf) |
| `periodicity_compare.py` | DddB vs DddA vs Hia5 pair-correlation |
| `periodicity_by_region.py` | Region-stratified (linker/NFR) periodicity |
| `periodicity_anchored.py` | Nuc-edge-anchored intra/extra profile |
| `periodicity_anchor_control.py` | Distance-from-nuc control for decay validation |
| `diagnose_tf_overcall.py` | TF quality distribution diagnostic |

### Key output figures

| figure | what it shows |
|--------|---------------|
| `periodicity_edge_vs_far_FINAL.png` | **DO NOT OVERWRITE.** Calibration figure: edge-anchored (orange) vs ≥200 bp control (blue) |
| `periodicity_compare.png` | DddB/DddA/Hia5 pair-correlation comparison |
| `best_guess_snapshots/*.png` | Per-dataset read snapshots |

---

## 7. For FiberBrowser integration

### Reading the tags

**Use the legacy tags directly** (simplest path — all information
is now available without parsing MA/AQ):

```python
# Nucleosomes (6 parallel arrays)
ns = read.get_tag('ns')  # starts (0-based query coords)
nl = read.get_tag('nl')  # lengths
nq = read.get_tag('nq')  # protection quality
mq = read.get_tag('mq')  # merge quality
lq = read.get_tag('lq')  # left edge quality
rq = read.get_tag('rq')  # right edge quality

# MSPs (2 arrays)
as_ = read.get_tag('as') # starts
al = read.get_tag('al')  # lengths

# TF footprints (5 arrays)
tn = read.get_tag('tn')  # starts
tl = read.get_tag('tl')  # lengths
tq = read.get_tag('tq')  # significance quality
el = read.get_tag('el')  # left edge sharpness
er = read.get_tag('er')  # right edge sharpness
```

All tags are `B:I` (32-bit integer arrays). All quality values are
0-255. TF tags may be absent if no TFs were called on the read.

**OR** parse the MA/AQ tags (for tools built on the fiberseq spec):

```python
from ma_tags import parse_ma_tag, parse_aq_array

ma_str = read.get_tag('MA')
parsed = parse_ma_tag(ma_str)
# parsed['read_length'] → int
# parsed['nuc'] → [(start_0based, length), ...]
# parsed['msp'] → [(start_0based, length), ...]
# parsed['raw_types'] → [(name, strand, qual_spec, intervals), ...]

aq = list(read.get_tag('AQ'))
qual_specs = [rt[2] for rt in parsed['raw_types']]
n_per_type = [len(rt[3]) for rt in parsed['raw_types']]
per_annotation = parse_aq_array(aq, qual_specs, n_per_type)
# per_annotation is a flat list, one sublist per annotation in MA order.
# Nucleosome i → per_annotation[i] = [nq, mq, lq, rq]
# MSP j → per_annotation[n_nucs + j] = []  (empty, no qualities)
# TF k → per_annotation[n_nucs + n_msps + k] = [tq, el, er]
```

### Suggested visualization

- **Nucleosomes**: colored rectangles. Suggested color gradient:
  - By `mq`: dark green (255, pure) → magenta (128, borderline) →
    light pink (0, tail merge)
  - By `nq`: dark blue (255, deep) → light orange (0, weak)
  - Edge ambiguity: draw dotted/faded edges when `lq` or `rq` < 128

- **TF footprints**: small colored rectangles or diamonds:
  - By `tq`: bright yellow (255, strong) → dim gray (0, weak)
  - Edge sharpness: solid outline when `el`/`er` ≥ 128, dashed when
    < 128
  - TFs inside nucleosome spans (alt-hypothesis) could be drawn in
    a different color or with a special marker

- **MSPs**: background shading behind the read in accessible regions

---

## 8. Validation results (v2 HMM → v3 comparison)

### DddB spacetime (Drosophila, 7 time windows, ~297k reads called)

| metric | v2 (HMM) | v3 (iter-17) |
|--------|----------|--------------|
| Reads called | ~242k | ~297k (+23%) |
| Nucs/read | 11-13 | 19-22 (~2×) |
| Median nuc size | 169-238 bp | 202-215 bp |
| Mean nuc size | 641 bp | 282 bp |
| ≥300 bp (overmerge) | 37.1% | 23.0% |
| TFs/read | n/a (2.1 as <90bp nucs) | 4-5 (explicit) |

### Hia5 (Drosophila fly embryo, sna/eve/ftz loci)

| metric | v2 (HMM) | v3 (iter-17) |
|--------|----------|--------------|
| Nucs (≥90 bp)/read | 64.6 | 87.0 |
| Median nuc size | 165 bp | 169 bp |
| ≥300 bp (overmerge) | 24.1% | 3.9% |
| ≥500 bp | 5.7% | 0.5% |
| TFs (<90 bp in ns/nl)/read | 29.9 | 2.9 |
| TFs (explicit tn/tl)/read | n/a | 133.1 |
