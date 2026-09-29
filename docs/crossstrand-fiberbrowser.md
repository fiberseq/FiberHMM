# FiberBrowser spec: cross-strand DAF consensus (MA tags + extract tracks)

Quick reference for rendering `fiberhmm-crossstrand` output in FiberBrowser.
Covers the consensus-read BAM layout, the `MA` tag additions, and the
`fiberhmm-extract` track files.

## Background

scDAF-seq (DddA) deaminates both strands of a duplex; each molecule is sequenced
as two reads of opposite flavor — **CT** (C→T, forward strand) and **GA** (G→A,
reverse strand). `fiberhmm-crossstrand` pairs the two strands of a molecule,
merges them into one **both-strand consensus read**, and re-calls footprints
using both strands jointly. Where both strands cover, every C **and** every G is
informative (~2× the deamination sampling), so footprint calls there are the
highest-confidence.

A cross-strand output BAM (`<dataset>.crossstrand.bam`) contains:

- **consensus reads** — one per confident CT/GA pair (identified by the `cs` tag)
- **passthrough reads** — every read not merged, unchanged

## Consensus read anatomy

- **Forward only** (SAM flag 0). Molecular frame == query frame == genomic
  orientation (no reverse-flip needed).
- **Reference-frame alignment**: CIGAR is all-`M`. Deaminations are
  substitutions; small indels from the source reads are dropped.
- **Sequence** carries deaminations as IUPAC codes: **`Y`** = C→T, **`R`** = G→A.
  Unlike a single-strand DAF read, a consensus read contains **both `R` and `Y`**.
- **No `MD` tag** — deaminations are read from the R/Y sequence.

### Tags

| tag | type | meaning |
|-----|------|---------|
| `MA` | `Z` | molecular annotations (see below) |
| `AQ` | `B,C` | quality bytes for `MA` (`nuc.QQQ`, `tf.QQQ`) |
| `ns` / `nl` | `B,I` | legacy nucleosome starts / lengths (query coords) |
| `as` / `al` | `B,I` | legacy MSP starts / lengths |
| `cs` | `Z` | consensus sources: `<CT_read_name>;<GA_read_name>` |
| `mc` | `i` | pairing cross-correlation × 1000 (0–1000; higher = better pairing) |

Pairing metadata (`mp` mate name, `mg` margin×1000, `mt` status `P`/`U`/`.`)
lives on reads in the intermediate `fiberhmm-pair` output; on the final
consensus only `mc` is carried.

## `MA` tag schema

Standard fiberseq [Molecular-annotation](https://github.com/fiberseq/Molecular-annotation-spec)
string, **1-based** coordinates:

```
MA:Z:<readlen>;nuc.QQQ:s-l,...;msp.:s-l,...;tf.QQQ:s-l,...;deam+:s-l;deam-:s-l
```

| annotation | strand | AQ bytes | meaning |
|---|---|---|---|
| `nuc.QQQ` | `.` | `nq, el, er` | nucleosomes |
| `msp.` | `.` | none | methylase-sensitive patches (accessible) |
| `tf.QQQ` | `.` | `tq, el, er` | TF / Pol II footprints |
| **`deam`** `+` | `+` | none | reference span where the **CT strand** (C→T) contributed coverage |
| **`deam`** `-` | `-` | none | reference span where the **GA strand** (G→A) contributed coverage |

`deam+` / `deam-` are a **custom annotation type** `deam` — permitted by the spec
(arbitrary type names; strand is per-annotation). They carry **no AQ bytes**.
Typically one interval each: `deam+` = the CT source read's span, `deam-` = the
GA source read's span, in consensus query coordinates.

### Strand-coverage regime

For any position along a consensus read:

- in **`deam+ ∩ deam-`** → **both strands** (C and G both informative) — the
  **highest-confidence** footprint region
- in `deam+` only → C-strand only
- in `deam-` only → G-strand only

A browser that ignores unknown `MA` types loses nothing — `nuc`/`msp`/`tf` still
render. For the both-strand overlay, prefer the extracted `_bothstrand.bb` track
(below) over parsing `deam` live.

## `fiberhmm-extract` tracks

```
fiberhmm-extract -i <dataset>.crossstrand.bam --nucleosome --msp --tf --both-strand
```

Writes one bigBed per type (`<dataset>_<type>.bb`), **BED12**, **genomic
(reference) coordinates**, one feature per read (blocks = the intervals):

| track file | flag | content |
|---|---|---|
| `<d>_nucleosome.bb` | `--nucleosome` | nucleosomes; per-block `nq` with `--block-scores` |
| `<d>_msp.bb` | `--msp` | MSPs (accessible) |
| `<d>_tf.bb` | `--tf` | TF / Pol II footprints; per-block `tq` (filter with `--min-tq`) |
| **`<d>_bothstrand.bb`** | `--both-strand` | **both-strand region (`deam+ ∩ deam-`) per read** |
| `<d>_deam.bb` | `--deam` | individual deamination sites (`R`/`Y`); `blockMod` 0 = GA, 1 = CT |

(`--all`, or no track flag, produces every track. BED-only with `--bed-only`.)

### The both-strand overlay (`_bothstrand.bb`)

One BED12 feature per consensus read:

```
chrom  start  end  read_name  1000  strand  start  end  0  blockCount  blockSizes  blockStarts
```

where the blocks are the `deam+ ∩ deam-` intervals in reference coordinates —
the stretch of the fiber where footprint calls draw on **both** strands.
Non-consensus reads produce no feature.

**Suggested rendering:** a toggleable highlight band behind the `nuc`/`tf`/`msp`
tracks, keyed by `read_name` so it aligns to the same fiber. Toggling it on marks
where a nucleosome / TF / MSP call is "extra-good" (both-strand supported); off,
the tracks read exactly like a normal single-strand DAF fiber.

## Notes

- Consensus reads are forward-only, so no molecular↔SEQ frame flipping is needed
  when reading `MA` intervals.
- `mc` (pairing correlation × 1000) can drive per-read confidence shading of the
  consensus itself, independent of the per-footprint `nq`/`tq` qualities.
- The both-strand region is a property of *coverage*, not of any single call, so
  it applies uniformly to whichever calls (nuc/tf/msp) fall inside it.
