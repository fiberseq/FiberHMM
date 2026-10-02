# Extracting tracks

`fiberhmm-extract` turns the per-read calls of a FiberHMM BAM into BED12 and
bigBed tracks in reference coordinates, one file per feature type, for genome
browsers, FiberBrowser and interval tools.

```bash
fiberhmm-extract -i out/pacbio.calls.bam -o out/tracks -c 2
```

```text
Completed in 1.0s: 300 reads -> nucleosome: ..., msp: ..., tf: ..., m6a: ..., m5c: 0, bothstrand: 0
  [nucleosome] bigBed: out/tracks/pacbio.calls_nucleosome.bb
  [msp] bigBed: out/tracks/pacbio.calls_msp.bb
  [tf] bigBed: out/tracks/pacbio.calls_tf.bb
  [m6a] bigBed: out/tracks/pacbio.calls_m6a.bb
  [m5c] no features, skipping
  [bothstrand] no features, skipping
```

Output files are `<output dir>/<BAM stem>_<type>.bb` (bigBed, needs UCSC
`bedToBigBed` on the `PATH`) or `.bed`. Without `bedToBigBed`, BED files are
written instead. `-o` defaults to the BAM's directory.

## Feature types

With no type flag, every type is extracted (`--all`); `deam` is skipped on
Fiber-seq BAMs.

| Flag | File suffix | Source | Block score |
|---|---|---|---|
| `--nucleosome` | `_nucleosome` | `MA` `nuc` (or `ns`/`nl`) | `nq` |
| `--msp` | `_msp` | `as`/`al` | `aq` |
| `--tf` | `_tf` | `MA` `tf.QQQ` with `tq` ≥ `--min-tq` (50) | `tq` |
| `--m6a` | `_m6a` | `MM`/`ML` m6A at ML ≥ `-p` | ML |
| `--m5c` | `_m5c` | DddA `ddda_mcg` spans, or native `MM`/`ML` 5mC | ML (0 for `ddda_mcg`) |
| `--deam` | `_deam` | DAF deaminations: MM/ML dU, else R/Y, else `MD` mismatches (first non-empty source per read) | |
| `--both-strand` | `_bothstrand` | the `deam+` ∩ `deam-` region of joint duplex molecules | |

`-p/--prob-threshold` defaults to 248 for BAMs that declare Hia5 Nanopore
and 125 otherwise; it does not apply to `ddda_mcg` spans or to R/Y/`MD`
deaminations. `--min-tq 0` keeps every TF call; 100 or more gives a stricter
set. In the `deam` track, `blockMod` is 0 for G→A (R) and 1 for C→T (Y),
matching FiberBrowser's flavour codes.

## Row format

One BED12 row per read and feature type (one row per feature with
`--circular-groups`, below): the row spans the read's features, each feature
is a block, `name` is the read name, `strand` the alignment strand, and
`score` the mean quality byte of the read's features (for `tf`, the mean
`tq`). Every schema ends with `isDuplicate` (1 when the read carried
flag `0x400`); FiberBrowser hides those rows by default.

```text
chrDemo  186  5125  pacbio_0277  137  -  186  5125  0  9  34,7,7,5,8,6,83,61,30  0,464,1415,2015,2592,2643,3958,4044,4909  0
```

Optional columns go between the 12 BED columns and `isDuplicate`, always in
this order:

```text
BED12 | per-block scores | circular grouping | hp | ps | isDuplicate
```

- `--block-scores`: per-block quality (`blockNq`/`blockEl`/`blockEr` for
  nucleosomes, `blockAq` for MSPs, `blockMl` for m6A/5mC,
  `blockTq`/`blockEl`/`blockEr` for TFs), so a browser can show per-feature
  quality without a sidecar.
- `--circular-groups`: changes the row model to **one row per feature**
  (a single block; `score` is that feature's own quality byte) and adds
  `circId`, `circPart`, `circParts`, `molStart`, `molLength`, to reassemble
  features that wrap the origin of circular molecules. A feature that wraps
  is written as its pieces, each named `<read>|<type>|<circId>|<part>/<parts>`;
  other features keep the read name and `circId` `.`. Expect many more rows
  than reads.
- `--haplotype-fields`: the read's `HP` and `PS` tags as signed integers,
  `-1` when absent. Extraction only copies them; it does not phase.

On a contig declared `@SQ TP:circular`, a read stored across the origin
(`fiberhmm-pipeline`, see [Plasmids](plasmids.md#reads-through-the-origin))
gives features past the contig end. Their rows are split at the origin into
two rows with the same name, one at each end of the contig; a block that
crosses the origin is cut in two, and per-block columns follow their blocks.

Each bigBed embeds its autoSQL schema, whose description starts with
`Sample: <name>.` (the BAM stem, or `--sample-name`, with dots and spaces
replaced). FiberBrowser uses it to group a sample's layers.

## Performance

`fiberhmm-extract` processes regions in parallel (`-c`, `--region-size`,
`--chroms`, `--skip-scaffolds`) and then sorts the BED under `LC_ALL=C`.
`-S/--sort-mem` (1G) and `--sort-parallel` (GNU sort only) speed up the sort
on deep BAMs. `--bed-only` skips bigBed conversion; `--keep-bed` keeps the
BED next to the bigBed.

If `bedToBigBed` is installed but fails for a type, the command exits 1 and
keeps that type's BED.

## Repairing bigBed sample names

Older bigBeds, or several filtered pools extracted with the same stem, may
carry colliding `Sample:` names, so FiberBrowser merges their layers.
`fiberhmm-utils fix-bigbed` rewrites the embedded name (needs UCSC
`bigBedInfo`, `bigBedToBed` and `bedToBigBed`):

```bash
fiberhmm-utils fix-bigbed out/tracks/pacbio.calls_tf.bb --sample-name demo_pacbio \
    -o out/tracks/pacbio.calls_tf.fixed.bb
fiberhmm-utils fix-bigbed sample.filtered_T_*.bb sample.filtered_GA_*.bb --in-place
```

Without `--sample-name`, each file's name is derived from its filename (the
stem minus the `_<layer>` suffix). Without `--in-place` or `-o`, it writes
`<name>.fixed.bb` next to each input.

Every option: [`fiberhmm-extract`](../reference/cli.md#fiberhmm-extract),
[`fiberhmm-utils fix-bigbed`](../reference/cli.md#fiberhmm-utils-fix-bigbed).
