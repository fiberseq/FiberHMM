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
Fiber-seq BAMs. `--all` includes `sv`, and includes `var` when a VCF is
available: `--variants VCF`, or the DAF SNP screen VCF found next to a DAF
BAM (logged). Without a VCF, `--all` skips `var` with a one-line note.

| Flag | File suffix | Source | Block score |
|---|---|---|---|
| `--nucleosome` | `_nucleosome` | `MA` `nuc` (or `ns`/`nl`) | `nq` |
| `--msp` | `_msp` | `as`/`al` | `aq` (fibertools ≤ 0.12 FIRE; FiberHMM writes none)¹ |
| `--tf` | `_tf` | `MA` `tf.QQQ` with `tq` ≥ `--min-tq` (50) | `tq` |
| `--m6a` | `_m6a` | `MM`/`ML` m6A at ML ≥ `-p` | ML |
| `--m5c` | `_m5c` | DddA `ddda_mcg` spans, or native `MM`/`ML` 5mC | ML (0 for `ddda_mcg`) |
| `--deam` | `_deam` | DAF deaminations: MM/ML dU, else R/Y, else `MD` mismatches (first non-empty source per read) | |
| `--both-strand` | `_bothstrand` | the `deam+` ∩ `deam-` region of joint duplex molecules | |
| `--variants VCF` | `_var` | the allele each read carries at the sites of a VCF/BCF ([below](#variant-alleles-var)) | `blockAllele`, `blockBq` (always written) |
| `--sv` | `_sv` | CIGAR deletions/insertions, long clips and the `SA` tag ([below](#structural-events-sv)) | `blockType`, `blockLen` (always written) |

¹ A BAM whose header has a `fiberhmm-apply` `@PG` but no
`FIBERHMM-ANNOTATION-CONTRACT` line comes from FiberHMM 3.2.1 or older, where
`apply --scores` wrote its own MSP posterior as `aq`. For such a BAM the
`_msp` bigBed's autoSQL description calls the `aq` score ambiguous instead of
fibertools FIRE.

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
flag `0x400`); FiberBrowser hides those rows by default. A `tf` row from the
demo call:

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

## Variant alleles (`var`)

`--variants VCF` writes, for every primary alignment, the allele the read
carries at each site of a VCF or BCF it covers. Any standard file works:
DeepVariant or Clair3 calls, `vg deconstruct` output, the DAF SNP screen
VCF. An unindexed or plain-text VCF is indexed into a temporary copy.

- **Sites.** SNVs, MNVs and indels whose REF and ALT are at most
  `--variants-max-len` (50) bp plus the VCF anchor base. Symbolic and
  breakend alleles, `*`, and longer alleles are never matched, and a record
  without a usable ALT is skipped. `--variants-pass-only` (on by default)
  keeps records whose FILTER is `PASS` or `.`. `--variants-sample NAME`
  keeps only sites where that sample's GT carries an ALT allele, and
  `--variants-het-only` keeps only its heterozygous sites.
- **Genotyping.** The rule is pileup-style, as in WhatsHap and longphase
  haplotag. An SNV takes the aligned base. An MNV or indel takes the
  read's bases over the REF span, with CIGAR insertions after each REF base
  included (after the last REF base too, for indel sites). That string is
  compared with each allele. There is no realignment: an indel the aligner
  placed differently from the VCF's normalized position (homopolymers,
  tandem repeats) comes out as `-1`. A site is reported only when its REF
  span lies inside the alignment (an indel site also needs one aligned base
  after it) and not in an `N` skip.
- **DAF.** On a C→T read, a T may be a converted C; on a G→A read, an A may
  be a converted G. R/Y codes are read as the converted base (Y = T,
  R = A). If more than one allele fits once conversion is allowed, the site
  is `-3`. At a C/T site, a C on a C→T read is REF (informative) and a T is
  `-3`. If exactly one allele fits, the site takes that allele, so a T at a
  C/G site on a C→T read counts as C (REF). The read's conversion strand follows FiberHMM's DAF rule (the `st`
  tag on R/Y input, else the read's R/Y marks or reference mismatches). A
  DAF read whose strand cannot be decided allows both conversions.
  `--variants-daf auto` (the default) applies this to BAMs whose header
  declares DAF (`fiberhmm-daf-encode`, or `fiberhmm-call` with `mode=daf`)
  and to reads that carry R/Y or `st`. `on` applies it to every read, `off`
  to none.
- **`--variants auto`** uses the DAF SNP screen VCF of the BAM. It looks
  first for the prefix or mask recorded on the `fiberhmm-call` `@PG`, then
  for `qc/<stem>.daf_snps.vcf` and `<stem>.daf_snps.vcf` next to the BAM.
  It exits 2 if it finds none.

Each row is a read (more than one row with the same name when sites
overlap, since bigBed blocks may not overlap). Each block is a site, with
size = REF length. `score` is the mean base quality of the sites.

| Column | Type | Meaning |
|---|---|---|
| `blockAllele` | `int[blockCount]` | `k ≥ 0` VCF allele index (0 REF, 1.. ALT); `-1` other sequence; `-2` the site's REF bases are all deleted in the read (spanning deletion); `-3` ambiguous because of DAF conversion |
| `blockBq` | `int[blockCount]` | base quality of the SNV base, or the minimum over the compared bases of an MNV/indel; `-1` if none |

The autoSQL description records the VCF:
`Sample: <name>. VCF: name=<basename>; md5=<md5 of the file>; sample=<NAME or *>. ...`.

## Structural events (`sv`)

`--sv` writes the per-read signatures long-read SV callers (Sniffles2,
cuteSV) start from. One row is written per alignment record, and
supplementary records get their own rows (secondary records are skipped).

| Event | `blockType` | Block | `blockLen` |
|---|---|---|---|
| CIGAR `D` ≥ `--sv-min-len` (30) | 0 DEL | the deleted reference span | deleted bp |
| CIGAR `I` ≥ `--sv-min-len` | 1 INS | the 1 bp reference base before the insertion (VCF anchor) | inserted bases |
| clip ≥ `--sv-min-clip` (100) at the left end | 2 CLIP_LEFT | the first aligned base | soft + hard clipped bases |
| clip ≥ `--sv-min-clip` at the right end | 3 CLIP_RIGHT | the last aligned base | soft + hard clipped bases |

Left and right are in reference orientation. `N` skips are not deletions.
After the two per-block arrays come the record's `mapq`, `readLen` (full
read length including soft and hard clips), `flag` (SAM flag; 0x800 marks a
supplementary record) and `sa` (the `SA` tag verbatim, `.` if none; a tag
longer than 4000 characters keeps its leading whole entries and ends in
`...`). `score` is the MAPQ. As in `var`, events that overlap (an insertion
next to a deletion, a clip next to an indel) go to a second row with the
same name.

### Schema

```text
var: BED12 | blockAllele | blockBq | [hp | ps] | isDuplicate
sv:  BED12 | blockType | blockLen | mapq | readLen | flag | sa | [hp | ps] | isDuplicate
```

Table names are `fiberhmm_var` and `fiberhmm_sv`. These columns are written
whatever `--block-scores` says; `--circular-groups` does not apply. bigBed
rows must start at their first block and end at their last, so
`chromStart`/`chromEnd` span the read's sites or events, not its whole
alignment (the same rule as every other layer). A coordinate column for
the alignment span would not survive a graph lift, so none is written.

**SCHEMA note (3.2 draft).** The layout differs from the original draft spec
in five ways:

1. `chromStart`/`chromEnd` span the blocks, not the read alignment.
   bedToBigBed rejects rows whose first block does not start at
   `chromStart` and whose last block does not end at `chromEnd`.
2. `isDuplicate` is the last column, as in every FiberHMM layer.
3. A read can have more than one row (overlapping sites or events).
4. `readLen` includes hard clips.
5. An INS block is the anchor base before the insertion.

## Keeping bigBeds instead of BAMs

For most browsing and per-molecule questions, the bigBeds carry what the
BAM was kept for: the call layers, m6A, haplotype tags (`--haplotype-fields`),
alleles at known sites (`var`) and split/indel signatures (`sv`). A 1 Mb
slice of a GM12878 PacBio Fiber-seq BAM (351 reads, 6.3 MB) gives 1.7 MB of
bigBeds for every layer, m6A being most of it; `var` (18 test sites) is 26 KB
and `sv` 42 KB.

(a) DAF, with the SNP VCF from the call step (add `--daf-snp-all-snvs` to
`fiberhmm-call` for non-C>T/G>A sites too):

```bash
fiberhmm-extract -i out/daf.calls.bam -o out/tracks --all --haplotype-fields -c 8
# var layer: using the DAF SNP screen VCF out/qc/daf.calls.daf_snps.vcf
```

(b) A DeepVariant or Clair3 VCF, restricted to the sample's heterozygous
sites:

```bash
fiberhmm-extract -i sample.calls.bam -o tracks --all \
    --variants sample.deepvariant.vcf.gz --variants-sample SAMPLE --variants-het-only \
    --haplotype-fields -c 8
```

(c) Sites from a pangenome graph with `vg deconstruct`. Use the reference
path the BAM is aligned to and keep the contig names identical:

```bash
vg deconstruct -P GRCh38 -a -t 8 graph.gbz > graph.vcf
bgzip graph.vcf && tabix -p vcf graph.vcf.gz
fiberhmm-extract -i sample.calls.bam -o tracks --variants graph.vcf.gz --sv -c 8
```

`vg deconstruct` writes nested and multi-kilobase alleles. Those beyond
`--variants-max-len` are skipped, and overlapping records give extra rows.
`--variants-sample` takes one of its haplotype samples.

**Graph lift.** `var` and `sv` rows are plain BED12 whose per-block columns
come right after `chromStarts`, so `fiberhmm-lift --bed` (graphlift's
`liftbed`) lifts them like any other layer. Blocks keep their codes; with
`--bed12-partial keep` (the default), blocks that do not lift are dropped together with
their codes, and a row lifted to the opposite strand reverses its per-block
columns with its blocks. Allele indices still refer to the source VCF.
CLIP_LEFT/CLIP_RIGHT name the source-reference ends, so on a lift to the
opposite strand they swap physical sides. In current `feat/pg-lift`, BED
reassembly treats any comma-separated column with `blockCount` items as
per-block. The verbatim `sa` column (5k+1 comma-separated items for k
entries) can be mistaken for one when the counts happen to match, so the
lift should skip columns that contain `;`.

## Coordinate frames

On reverse-strand reads, footprint tags are read in the frame the BAM
records, by the [shared rule](../reference/footprint-tag-frame.md) that
recall, QC and consensus use: `MA` is SEQ frame unless the header declares
`coord=molecular`, and legacy `ns`/`nl`/`as`/`al` follow the `@PG` history
(SEQ for FiberHMM 2.12 and earlier, molecular for fibertools and 3.0).
DddA m5C groups (`ddda_mcg`, `ddda_ucg`, any qualifier) are always molecular,
as `fiberhmm-tag-m5c` writes them. The `deam+`/`deam-` groups of `--both-strand` follow the `MA`
rule, as the duplex recall mask reads them; `fiberhmm-pair` consensus reads
are forward, so either frame reads them the same.
`--input-frame query|molecular` sets the legacy-tag frame instead. If merged
`@PG` histories disagree and nothing declares `coord=molecular`, nucleosome
and MSP extraction stops and asks for `--input-frame`.

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
