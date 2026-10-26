# Lifting to another assembly

`fiberhmm-lift` moves a called BAM from the assembly it was aligned to (a
personal or haplotype assembly, say) to another assembly in the same
pangenome graph, keeping every footprint on the same bases of the molecule,
and repairs what a lift leaves stale. The primary route is native: graphlift's
`liftbam` (BAM) and `liftbed` (BED). vg is an optional fallback.

**Native** (the default when a map is given): graphlift's `liftbam` lifts the
records through a graphlift chain map, which graphlift's `gbz2chain` derives
from the graph once per source/target pair. No vg and no container.

```bash
# once per graph: GRCh38 <- both HG002 haplotypes, contigs named as in the BAM
gbz2chain -g hprc.gbz -t 'GRCh38#0' -s HG002 --hap-suffix 1=_PATERNAL,2=_MATERNAL \
    --source-sizes hg002v1.1.fa.fai \
    -o hprc.GRCh38.HG002.chain
fiberhmm-lift -i calls.hg002.bam -o calls.grch38.bam \
    --map hprc.GRCh38.HG002.chain.graphlift.json --target-fasta GRCh38.fa -t 16
```

The input's contig names must be the map's source names (`gbz2chain
--hap-suffix` or `--source-rename` set them), with the same lengths.
Aligned bases are projected through the chains that are primary in the
target net; source bases with no target counterpart become insertions (soft
clips at the ends) and target bases between two projected bases deletions.
A record is split into supplementaries, each with the whole `SEQ`
soft-clipped, at a chain break, a strand change or a gap of `--split-gap`
bp or more (default 10,000). Records that cannot be lifted (no projection,
contig not in the map) are counted by reason, and with `--unlifted` written
to a BAM in source coordinates (`Yu:Z:<reason>`). `--alt fallback` lifts a
record through an alternative chain when no primary chain projects it. A
second placement of mostly the same read bases (a duplication) is written as
a secondary alignment. Paired-end records are not lifted
(`paired_end_unsupported`).

### Installing graphlift

graphlift (<https://github.com/mtcicero26/graphlift>) is built from source
with Rust (the version is pinned in its `rust-toolchain.toml`) and a C
compiler. `liftbam` is built only with graphlift's `bam` feature:

```bash
git clone https://github.com/mtcicero26/graphlift
cd graphlift
cargo build --release --locked --features bam   # gbz2chain, liftbed and liftbam
```

The binaries land in `target/release/`. Put that directory on the `PATH`, or
name the binaries with `--liftbam` and `--liftbed` (`liftbed` is also found
next to an explicit `--liftbam`). A build without `--features bam` has
`liftbed` and `gbz2chain` but no `liftbam`; `fiberhmm-lift` then stops with
the command above.

**vg** (optional fallback, without a map): `vg inject` and `vg surject` on
the graph itself, with fibertools-rs `ft pg-pansn` adding the source panSN
prefix before and stripping it after.

```bash
fiberhmm-lift -i calls.hg002_mat.bam -o calls.grch38.bam \
    -g hprc.gbz --prefix 'HG002#1#' --target GRCh38 \
    --target-fasta GRCh38.fa -t 16
```

`--prefix` is the panSN prefix of the source assembly's paths in the graph
and `--target` the sample to surject onto. `vg` and `ft` (fibertools-rs ≥
0.13, for `pg-pansn`) must be on the `PATH`, or named with `--vg`/`--ft`;
`--vg` may be a wrapper script that runs vg in a container.

On the same graph the two engines place 99.95–99.97% of footprint intervals
at exactly the same target coordinates (HG002 → GRCh38 AluY and chr1 windows). Either way the output is sorted and indexed.

## What the repair does

Both engines copy `SEQ`, `QUAL` and every tag unchanged (`SEQ` and `QUAL`
reversed on a strand change; `MM`/`ML` are in original read orientation and
need no change), and footprints in molecular coordinates stay right, even on
reads whose strand changes. Everything that depends on the source alignment
is repaired (the table lists what the vg stages leave; `liftbam` keeps
`0x800` and hard clips itself):

| Lifted as | Repaired |
|---|---|
| supplementary alignments lose `0x800` and become extra primaries | `FLAG` restored from the source (one primary per read) |
| hard clips are dropped (`649M1845H` → `649M`) | hard-clipped supplementaries get the clipped bases from their primary before the lift, as soft clips (see below); one whose primary is not in the input is left out and counted |
| `st` keeps its value on reads that change strand | `st` switched `CT` ↔ `GA` (DAF readers otherwise see no deamination) |
| footprint tags in SEQ frame (FiberHMM ≤ 2.12) end up mirrored on those reads | converted to molecular frame before the lift |
| `MD`, `NM`, `SA` describe the source alignment | recomputed against `--target-fasta` (`samtools calmd`; `SA` rebuilt from the lifted parts); without it they are removed |
| `MAPQ` | kept from the source alignment by default (see [MAPQ](#mapq)) |
| `@PG` lines of the lift unlinked, output unsorted | one `@PG` chain ending in `fiberhmm-lift`; sorted and indexed |

A hard-clipped supplementary keeps its own calls: FiberHMM and fibertools
call each supplementary on its own aligned bases, so its footprints describe
its segment of the molecule, not the primary's. Its `SEQ` keeps its own
aligned bases (DAF R/Y codes are per alignment) with the clipped flanks from
the primary. Its footprint tags (`MA`/`AQ`/`AN`, `Ma`/`Aq`/`An`,
`ns`/`nl`/`nq`, `as`/`al`/`aq`) are rebased onto the filled sequence: the
read length in `MA` (else `Ma`) says whether they describe the stored bases
(moved by the clip, read length set to the molecule's) or the whole molecule
(kept), and the legacy tags follow it. Its `MM`/`ML` are its own, moved onto
the filled sequence, when they are valid; otherwise the primary's calls are
re-encoded on it, and a call on a base the supplementary changed (C → Y) is
dropped (`mods_dropped`). A supplementary without footprint tags stays
without. One whose tags cannot be placed unambiguously is left out and
counted (`excluded_unverifiable`).

Secondary and unmapped records are not lifted. Supplementaries whose primary
is not in the input are left out before the lift
(`orphan_supplementary_no_primary_in_input`), and those whose primary did not
lift are dropped after it (`orphan_supplementary_dropped`). The counts are printed and recorded in the
`fiberhmm-lift` `@PG`, which also declares `coord=molecular`:

```text
graph lift repair; mode=lift; engine=vg; source=HG002 target=GRCh38; graph=hprc.gbz; calmd=yes; excluded_supplementary=0; filled_supplementary=12; dropped_secondary=0; dropped_unmapped=0; unlifted=7; orphan_supplementary_dropped=0; st_swapped=563; mapq=source provenance_tags=Yq,Yp; coord=molecular (footprint tags carried over from the input)
```

Each lifted record carries `Yq:i` (source `MAPQ`) and `Yp:Z` (source locus,
`<contig>:<1-based POS>:<strand>`). Aligner scores other than `AS` (minimap2
`tp`, `cm`, `s1`, `de`, ...) still describe the source alignment.

## MAPQ

A lifted record keeps the `MAPQ` of its source alignment (`--mapq source`,
the default, as CrossMap does): it says how uniquely the read
placed on the source assembly, not on the target. Reads aligned to a diploid
assembly (both haplotypes in one reference) get low `MAPQ` wherever the two
haplotypes are identical, and keep it after the lift: in one HG002 data set
43% of the lifted chr1 primaries had `MAPQ` < 20, against none when the same
reads were aligned to GRCh38 directly. A `MAPQ` filter (`fiberhmm-qc
--min-mapq`, which prints a note on lifted BAMs) then drops those reads.
Alternatives: select reads by their haplotype tags (`HP`) instead of `MAPQ`,
or set a fixed value with `--mapq fixed:N` (for example `fixed:60`). The
source `MAPQ` stays in `Yq` either way, and the `fiberhmm-lift` `@PG` records
the mode (`mapq=source` or `mapq=fixed:N`).

## A BAM lifted by another vg-based tool

A BAM lifted with `vg inject`/`vg surject` outside `fiberhmm-lift` (for
example with fibertools-rs `ft pg-pansn` around them) has the stale fields in
the table above. `fiberhmm-check` reports such a BAM (`graph-lift-unrepaired`). Repair it from
the BAM it was lifted from:

```bash
fiberhmm-lift --repair-only -i lifted.bam --source calls.hg002_mat.bam \
    -o lifted.repaired.bam --target-fasta GRCh38.fa
```

Each lifted record is matched one-to-one with its source record (same read
name and sequence; the aligned part of the molecule decides between
supplementaries). Records that match no source record, or several equally
well (checked again each time a source record is taken), are dropped and
counted.
Hard clips the lift dropped are added back as hard clips.

`PS` (phase set) and `HP` are copied unchanged by both engines. `PS` names a
phase block; readers only compare it for equality. Phasing tools that phase
against a reference and then tag reads aligned to a personal assembly
already put reference coordinates in it, so remapping it would change a
correct value. On a diploid source, blocks of the two haplotypes can share a
`PS` value on one target contig; `Yp` (the source locus) tells them apart.

## BED files

`fiberhmm-lift --bed` lifts BED files. With `--map` it runs graphlift's
`liftbed`, which lifts each interval as `liftbam` would lift a read aligned
over the same bases (same chains, same primary net, same splitting rules),
so a lifted BED and a lifted BAM agree.

```bash
# fiberhmm-extract BED12 rows
fiberhmm-lift --bed -i calls_nucleosome.bed.gz -o calls_nucleosome.grch38.bed.gz \
    --map hprc.GRCh38.HG002.chain.graphlift.json --unlifted calls_nucleosome.unlifted.bed

# narrowPeak (summits lifted too)
fiberhmm-lift --bed -i peaks.hg002.narrowPeak -o peaks.grch38.narrowPeak \
    --map hprc.GRCh38.HG002.chain.graphlift.json

# from stdin
zcat peaks.hg002.bed.gz | fiberhmm-lift --bed -i - -o peaks.grch38.bed \
    --map hprc.GRCh38.HG002.chain.graphlift.json

# vg fallback, without a map (needs the source FASTA)
fiberhmm-lift --bed -i calls_nucleosome.bed -o calls_nucleosome.grch38.bed \
    -g hprc.gbz --prefix 'HG002#1#' --target GRCh38 --source-fasta hg002.fa
```

- **Input and output.** The input may be plain, gzip or bgzip, or `-` for
  stdin. An output (and an `--unlifted` file) whose name ends in `.gz` is
  written bgzip-compressed. Rows keep the input order; sort them before
  `tabix`.
- **Formats** (`--bed-format`, default `auto`):
  - `bed12`: rows are lifted block by block and rebuilt, since the engines
    move only an interval's start and end. `auto` picks it when every row has
    12 or more columns.
  - `narrowPeak`: the summit (column 10, an offset from the start) is lifted
    as a 1 bp interval and rewritten as an offset from the new start. A
    summit that does not lift, or lands outside the lifted peak, becomes -1;
    -1 stays -1. `auto` picks it for a `.narrowPeak` or `.narrowPeak.gz`
    input or a `track type=narrowPeak` line, not from the column count.
  - `bed`: rows are lifted one by one; columns after the third are copied.
  - thickStart/thickEnd (columns 7–8, `bed` and `bed12`): a thick interval
    that spans the row spans the lifted row, an empty one stays empty, and
    one inside the row is lifted on its own and clipped to the lifted row (it
    becomes empty if it does not lift). For `bed` this applies when every row
    has integer columns 7–8 within the row.
- **BED12 rows with unliftable blocks** (`--bed12-partial`). By default
  (`drop`) a row is dropped when one of its blocks does not lift. `keep`
  keeps the blocks that lift, recomputes blockCount, sizes, starts and the
  row bounds (per-block columns are subset too), and writes each dropped
  block to `--unlifted` as a BED6 row in source coordinates, named
  `<row name>|<block index>`, after a `#partial_<reason>` line. Either way a
  row is dropped when no block lifts, its blocks land on different contigs,
  or they come back out of order (`block_no_projection`, `split_contigs`,
  `rearranged`, ...). Lifting the BAM and running `fiberhmm-extract` on it
  gives the same blocks as `keep`, except where the BAM lift splits a
  molecule into a primary and a supplementary (across a large target-only
  gap, say): `keep` writes the blocks of both pieces in one row, while
  `fiberhmm-extract` writes the primary only.
- **Native rules.** An interval runs from its first to its last projected
  base, so bases with no target counterpart (an insertion in the source)
  shrink it. An interval that lands in several pieces (across an inversion
  breakpoint, say) keeps the piece with the most projected bases. On a `-`
  chain the strand column is swapped. Unlifted rows are counted by reason
  and, with `--unlifted`, written liftOver style.
- **`--min-match`.** The default, 0, lifts any interval with a projected
  base, as vg does. An interval or block that is mostly an insertion in the
  source then lifts as a remnant of a few bp: in HG002 → GRCh38 test tracks
  0.07–0.5% of nucleosome blocks lifted to less than half their size, and
  1 in 2,000–15,000 to 10 bp or less. `--min-match 0.5` drops intervals of which
  less than half projects, so those remnants are not written; they count as
  unlifted, and with `--bed12-partial drop` their whole row goes with them.
  With `--bed12-partial keep`, `--min-match 0.5` drops only the remnant
  blocks.
- **vg** (fallback: `--engine vg`, or `--graph` without `--map`). The
  source contigs (from `--source-fasta`) are cut into 50 kb reads that
  overlap by half, where intervals lie; `vg inject` puts them on the graph
  and `vg surject` onto the target. Each interval takes the target span of
  its own bases in the read where it lies most centrally (an interval longer
  than 25 kb gets a read of its own). A reverse-strand surjection swaps the
  strand column. fibertools-rs is not needed. Intervals that do not surject are
  counted and, with `--unlifted`, written out (`no_projection`,
  `contig_not_in_fasta`).

Lifting the BAM and extracting from the lifted BAM is the better route when
the BAM is at hand.
