# Motif footprints (`fiberhmm-footprint`)

`fiberhmm-footprint` is the FiberHMM analogue of fibertools `ft footprint`.
Given a BAM with FiberHMM calls and a BED of motif instances, it reports for
every motif how many molecules span it, how many are eligible (the motif lies
in an accessible patch) and how many are footprinted. It reads the calls
already in the BAM; nothing is re-decoded.

It is not the same command as `fiberhmm-footprint-model` (a population model
of recurrent footprint geometries; see
[Footprint population model](footprint-model.md)). For a posterior-based
alternative that scores each molecule's own marks against footprint classes
discovered in a window, use [`fiberhmm-consensus`](consensus.md).

```bash
fiberhmm-footprint --bam calls.bam --bed ctcf_motifs.bed -o ctcf.footprint.tsv \
    --per-molecule ctcf.molecules.tsv --out-bam calls.ctcf.bam --cores 8
```

## Call input: BAM or call bigBeds

Calls come from `--bam`, or from the per-read call bigBeds that
`fiberhmm-extract` writes (`--calls-bb PREFIX` for `PREFIX_tf.bb`,
`PREFIX_msp.bb`, `PREFIX_nucleosome.bb`, `PREFIX_m6a.bb`, or explicit
`--tf-bb`/`--msp-bb`/`--nuc-bb`/`--m6a-bb`). The bigBeds are queried per region,
never loaded whole. Extract them with `--block-scores` so they carry what the
rule needs:

```bash
fiberhmm-extract -i calls.bam -o bb/ --tf --msp --nucleosome --m6a --block-scores -p 125
fiberhmm-footprint --calls-bb bb/calls --bed ctcf.bed -o ctcf.tsv --out-bb bb/calls
```

| Needed | In the bigBed | Without `--block-scores` |
|---|---|---|
| per-call LLR (`--min-llr`) | `blockTq` | only the floor `fiberhmm-extract --min-tq` applied (default 50); LLR reported as `.` |
| outer edges | `blockEl`/`blockEr` (genomic left/right) | unavailable: conservative edges are used, with a warning |
| m6A ML (`--require-unmodified`) | `blockMl` | every extracted position, i.e. those above `--prob-threshold` |

What the format cannot carry, whatever the options:

- A row spans its first to last call, not the alignment, so a read's extent is
  the union of its rows. Reads with no FiberHMM calls are absent. Eligibility
  and footprint calls are unaffected; `n_spanning_fibers` counts only
  annotated reads.
- Outer edges are applied on the reference rather than in read coordinates,
  so they differ when an insertion or deletion lies inside the ambiguity zone.

On 5,369 sites of a Hia5 PacBio library (`--require-unmodified`, default
width cap), bigBed input matched BAM input for every motif's eligible count
(90,232) and for 11,031 of 11,036 footprinted molecules. The five that differ
have an indel inside the edge-ambiguity zone. The 15,365 molecules missing from
the bigBed run are reads without FiberHMM calls (`unannotated` in the BAM
run).

`--out-bb PREFIX` (either input) writes `PREFIX_fp_NAME.bb` per motif set:
one BED12 row per read, one block per footprinted motif instance, with
`blockTq`, `blockCw`, `blockEf` and `isDuplicate` (autoSQL `fiberhmm_fp`) and
a `Sample:` marker so viewers group it with the dataset's other tracks.
`fiberhmm-consensus` still needs a BAM: its native replay re-decodes each
molecule's MSPs from the m6A calls and sequence, which the bigBeds do not
carry.

## Motif input

`--bed` takes a BED (BED3+, BED6 strand honoured, `.gz` allowed) or a bigBed,
recognised by its magic number whatever the extension. A bigBed, such as the
JASPAR genome-wide TFBS tracks (hundreds of millions of sites), is never
expanded: it is queried in 1 Mb pieces, only where the BAM has reads, or only
in the `--region CHROM[:START-END]` given (repeatable). Results for a bigBed
are written in genomic order.

- `--motif-name NAME[,NAME...]` keeps motifs whose name, TF-name field (for
  JASPAR, `TFName`) or matrix ID matches, case-insensitively; `MA0139`
  matches `MA0139.2`.
- `--min-score` filters on the score column. JASPAR's UCSC tracks encode the
  motif p-value there (score = -log10(p) x 100, so `--min-score 400` keeps
  p <= 1e-4). `--max-pvalue` applies when the file has its own p-value field.
- For a bigBed, layers are named from the TF-name field when the file has one
  (`fp_CTCF`), else from the name field (`fp_MA0139_2`).
- More than 20 motif sets (layer pairs) in one run, e.g. a multi-TF bigBed with
  no `--motif-name`, is refused unless `--layer-name` merges them or
  `--allow-many-layers` is given.

Reading bigBed needs pyBigWig (`pip install 'fiberhmm[bigbed]'`) or, as a
fallback, the UCSC `bigBedToBed` binary on PATH; neither is a FiberHMM
dependency.

```bash
fiberhmm-footprint --bam calls.bam --bed JASPAR2024.bb --motif-name CTCF --min-score 500 -o ctcf.tsv
```

## Rule

For each molecule and motif, in reference coordinates:

| Step | Test |
|---|---|
| spanning | the alignment spans the motif: start < motif start and end >= motif end (ft's `spans` rule) |
| eligible | an MSP of the molecule spans the motif by the same rule. MSPs are the BAM's FiberHMM accessibility calls (`MA` `msp`; else fibertools `Ma`; else legacy `as`/`al`) |
| footprinted | one `tf` call with LLR >= `--min-llr` (default 5, i.e. `tq` >= 50) and conservative width <= `--max-width` whose bounds contain the whole motif (call start <= motif start and call end >= motif end) |

Bounds (`--edges`):

- `outer` (default): the conservative call extended by the edge ambiguity in
  its `el`/`er` bytes, i.e. through the non-informative bases up to the next
  informative position on each side. `--outer-bound exclusive` (default) stops
  just before that terminating position; `--outer-bound inclusive` includes it
  (one more base on each side). The stored outer edge never includes the
  methylated terminating base, so exclusive is the faithful reading. A saturated edge byte (ambiguity >= 30 bp) is
  read as 30 bp, a lower bound.
- `conservative`: the call exactly as stored, which ends at the last
  informative unmethylated base. Motifs with G/C ends are then often missed by
  one or two bases.

The width bounds apply to the conservative reference width of the call:
`--min-width` (default 0, off) and `--max-width`, both inclusive; 0 disables
either. The default, 74 bp, was set without ChIP labels:
it is the 95th percentile of the conservative widths of 168,943 FiberHMM TF
calls (LLR >= 5) whose outer bounds overlap a motif in an MSP-eligible
molecule, pooled over CTCF, NRF1, REST and USF1 motifs in five cell lines
(K562, GM12878, H1, H9, HepG2). The 90th and 99th percentiles are 61 and
87 bp. Calls wider than this are mostly broad protections that are not
motif-sized.

**Calibrate the window with ChIP-seq when you can.** The label-free window is
a safe default, but a window learned from ChIP-seq labels usually scores
better. In the benchmark it raised held-out AUROC over the 0-74 bp default by
+0.013 (CTCF), +0.020 (NRF1) and +0.050 (REST), and only +0.007 for USF1. It
also held up on cell types it was not learned on: in leave-one-cell-type-out
tests, windows learned on the other cell lines gave those gains on the held-out
line. See [Calibrating the width window](#calibrating-the-width-window).

`--require-unmodified` additionally requires no m6A call (ML >= `--ml`,
default 125) inside the motif, which is ft's footprint criterion. It reads
MM/ML m6A calls only.

## Calibrating the width window

`fiberhmm-footprint calibrate` learns `[--min-width, --max-width]` from ChIP-seq
labels, using the benchmark's method:

- **Score.** Per motif, the footprinted fraction of eligible molecules, pooled
  over a cell type's BAMs.
- **Objective.** The mean of the per-cell-type AUROCs against the ChIP labels;
  molecules are never pooled across cell types.
- **Search.** A 2D grid of conservative call widths: min 0-80 bp and max
  10-150 bp, in 5 bp steps.
- **Held out.** Leave-one-cell-type-out when several `--bam`/`--chip` pairs are
  given, otherwise odd against even chromosomes. A fold's window is always
  chosen on its training data alone.

```bash
fiberhmm-footprint calibrate --bed ctcf_motifs.bed \
    --bam h1_a.bam,h1_b.bam --chip h1_ctcf_peaks.bed --cell H1 \
    --bam k562.bam --chip k562_ctcf_peaks.bed --cell K562 \
    --require-unmodified -o ctcf_window
```

It prints the optimal window, a robust range (the bounds of every window
within 0.005 AUROC of the optimum, for choosing conservative bounds), the
held-out AUROC per fold, and a ready-to-paste line:

```text
use: fiberhmm-footprint --min-width 25 --max-width 55 --require-unmodified
  or: fiberhmm-footprint --settings ctcf_window.settings.json
```

It writes:

- `PREFIX.surface.tsv`: the full 2D surface. It holds the per-cell and mean
  AUROC, plus a `masked` flag where too few molecules or motifs are positive
  for AUROC to mean much.
- `PREFIX.heldout.tsv`: the per-fold results, including in-sample minus
  held-out optimism.
- `PREFIX.settings.json`: the window and rule options, for `--settings`.
- `PREFIX.heatmap.png`: the surface as a heatmap, when matplotlib is
  installed (`fiberhmm[plots]`).

A window learned in one cell type can be applied to another with `--settings`.
The window is specific to the factor: report the held-out AUROC, not the
in-sample one, which overstated AUROC by 0.006-0.034 in the benchmark. The
benchmark windows were learned with `--require-unmodified`.

## Outputs

**Per-motif table** (`-o`, default stdout), modelled on ft footprint's:

| Column | Meaning |
|---|---|
| `#chrom start end strand name layer` | motif (BED) and its MA layer |
| `n_spanning_fibers` | molecules spanning the motif |
| `n_spanning_msps` | of those, eligible (an MSP spans the motif) |
| `n_footprinted` | eligible and footprinted (code 3) |
| `fraction_footprinted` | `n_footprinted / n_spanning_msps` |
| `n_overlapping_nucs` | spanning molecules with a FiberHMM nucleosome overlapping the motif |
| `n_not_spanning` | molecules that overlap but do not span the motif (not scored) |
| `footprint_codes fiber_names haplotype` | per spanning molecule, comma-separated, in the same order |

Codes follow ft's bit layout for a one-module motif: bit 0 = an MSP spans the
motif, bit 1 = footprinted. `0` not eligible, `1` eligible and not
footprinted, `2` footprint call but no MSP over the motif, `3` footprinted.

**Per-molecule rows** (`--per-molecule`): motif, `motif_id`, `read_name`,
`status` (`footprinted`, `not_footprinted`, `no_msp`, `unannotated` = no
FiberHMM calls on the read, `not_spanning`), `code` (-1 when not spanning),
`reason` for a negative (`no_call`, `partial_overlap`, `too_wide`,
`too_narrow`, `modified`), the chosen call's conservative and outer reference bounds, width
and LLR, which bounds contained the motif (`edge`), nucleosome overlap and
haplotype. When several calls qualify, the narrowest is reported. Rows come in
genomic order of the motifs (the table keeps BED order).

**Annotated BAM** (`--out-bam`): a copy of the input with two MA layers per
motif set, so several motif sets stack as separate FiberBrowser layers:

| Layer | Bytes | Intervals |
|---|---|---|
| `fp_NAME.QQQ` | `tq` of the qualifying call, its conservative width (bp, max 255), edge flag (1 = conservative bounds contain the motif, 2 = only the outer bounds do) | footprinted molecules (code 3) |
| `fpneg_NAME.Q` | reason: 1 no call, 2 partial overlap, 3 too wide, 4 m6A in motif, 5 too narrow (every containing call below `--min-width`) | eligible, not footprinted (code 1) |

Molecules that are not eligible get neither layer, so "not footprinted"
(`fpneg_`) is distinct from "not observed" (no interval). Each interval covers
the motif's aligned bases on the molecule, in the same frame as the read's own
calls; `AN` names it by motif id (`chrom:start-end:strand`). Layers are
declared in `@CO MA-TYPES:v1` and described in a
`@CO FIBERHMM-FOOTPRINT-MA:v1:` JSON record with the run's parameters.

`NAME` is the BED name column (one layer pair per distinct name) or
`--layer-name`. It is reduced to the characters an MA type name allows,
`[A-Za-z0-9_]`: the spec's type string is name + strand (`+`, `-` or `.`) +
quality letters, so `.`, `+` and `-` cannot appear in a name. Other characters
become `_` (`MA0139.1` becomes `MA0139_1`).

Running the command again on the output BAM adds new layers and keeps every
existing one (`nuc`, `msp`, `tf`, consensus layers, earlier `fp_*` layers)
byte for byte. A layer name that already exists is replaced in every read, not
duplicated.

### Listing and removing layers

`fiberhmm-utils ma-list` shows a BAM's MA layers with their full type string
(`fp_REST.QQQ`, `nuc.QQQ`, `msp.`), the molecules carrying each, the interval
count and whether `@CO MA-TYPES` declares it. It reads the first 1000
alignments by default; `--scan` reads all of them, `--region` limits the
scan.

`fiberhmm-utils ma-drop` removes layers by logical name or glob from every
read, with their `AQ` bytes and `AN` names, and removes them from the
`MA-TYPES` declaration. All other layers and tags stay byte-identical, so
adding a layer with `fiberhmm-footprint` and dropping it again gives back the
original records. Core layers (`nuc`, `msp`, `tf`) are refused unless
`--force`.

```bash
fiberhmm-utils ma-list calls.ctcf.bam --scan
fiberhmm-utils ma-drop calls.ctcf.bam --layers 'fp_*,fpneg_*' -o calls.clean.bam
fiberhmm-utils ma-drop calls.ctcf.bam --layers fp_REST --in-place
```

### Memory

Each alignment is parsed, scored against the motifs it overlaps and released
at once; only the per-motif results are kept, and the annotated BAM is written
by scoring each alignment again as it is copied. Memory therefore does not grow
with depth, read length or the number of motifs in a region: on 5,369 sites of
a Hia5 PacBio library (4 cores, with `--per-molecule` and `--out-bam`) the
peak was 414 MB in total, down from 967 MB when whole regions were buffered.
No flank is needed: eligibility and footprint calls only use intervals that
overlap the motif.

## Differences from `ft footprint`

- **Eligibility** comes from FiberHMM's own MSPs (`ft footprint` also uses the
  input BAM's MSPs; on a FiberHMM-called BAM these are the same calls).
- **Footprinted** means a FiberHMM footprint call contains the motif, with
  bounds from the HMM and a width cap, instead of "no m6A in the motif".
  `--require-unmodified` adds ft's criterion on top.
- `n_footprinted` counts eligible molecules only (code 3); ft's module column
  also counts code 2.
- One module per motif; ft's YAML modules and `motif_span`, FIRE columns and
  `fire_quals` are not reproduced.
- Secondary, supplementary and unmapped records are skipped by default
  (`-F 0x904`; ft's default is 0).
- Nucleosome overlap uses ordinary half-open overlap.
- `--msp-bam` (advanced) takes MSPs from another BAM, matched by read name,
  for like-for-like comparisons with an ft run on a different call set.
