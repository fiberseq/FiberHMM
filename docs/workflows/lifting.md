# Lifting to another assembly

`fiberhmm-lift` moves a called BAM from the assembly it was aligned to (a
personal or haplotype assembly, say) to another assembly in the same
pangenome graph, keeping every footprint on the same bases of the molecule.
It runs the stages of fibertools-rs `ft pg-lift --bam` (`ft pg-pansn`,
`vg inject`, `vg surject`, `ft pg-pansn --strip`) and repairs what that lift
leaves stale.

```bash
fiberhmm-lift -i calls.hg002_mat.bam -o calls.grch38.bam \
    -g hprc.gbz --prefix 'HG002#1#' --target GRCh38 \
    --target-fasta GRCh38.fa -t 16
```

`--prefix` is the panSN prefix of the source assembly's paths in the graph
and `--target` the sample to surject onto, as for `ft pg-lift`. The output
is sorted and indexed. `vg` and `ft` (fibertools-rs ≥ 0.13) must be on the
`PATH`, or named with `--vg`/`--ft`; `--vg` may be a wrapper script that
runs vg in a container.

## What the repair does

The lift copies `SEQ`, `QUAL` and every tag unchanged, and footprints in
molecular coordinates stay right, even on reads whose strand changes.
Everything that depends on the source alignment is repaired:

| Lifted as | Repaired |
|---|---|
| supplementary alignments lose `0x800` and become extra primaries | `FLAG` restored from the source (one primary per read) |
| hard clips are dropped (`649M1845H` → `649M`) | hard-clipped supplementaries get the clipped bases from their primary before the lift, as soft clips; one whose primary is not in the input is left out and counted |
| `st` keeps its value on reads that change strand | `st` switched `CT` ↔ `GA` (DAF readers otherwise see no deamination) |
| footprint tags in SEQ frame (FiberHMM ≤ 2.12) end up mirrored on those reads | converted to molecular frame before the lift |
| `MD`, `NM`, `SA` describe the source alignment | recomputed against `--target-fasta` (`samtools calmd`; `SA` rebuilt from the lifted parts); without it they are removed |
| `MAPQ` | kept: how uniquely the read placed on the source assembly is what the lift inherits |
| `@PG` lines of the lift unlinked, output unsorted | one `@PG` chain ending in `fiberhmm-lift`; sorted and indexed |

Secondary and unmapped records are not lifted, and supplementaries whose
primary did not lift are dropped. The counts are printed and recorded in the
`fiberhmm-lift` `@PG`, which also declares `coord=molecular`:

```text
graph lift repair; mode=lift; source=HG002 target=GRCh38; graph=hprc.gbz; calmd=yes; excluded_supplementary=0; filled_supplementary=12; dropped_secondary=0; dropped_unmapped=0; unlifted=7; orphan_supplementary_dropped=0; st_swapped=563; mapq=source provenance_tags=Yq,Yp; coord=molecular (footprint tags carried over from the input)
```

Each lifted record carries `Yq:i` (source `MAPQ`) and `Yp:Z` (source locus,
`<contig>:<1-based POS>:<strand>`). Aligner scores other than `AS` (minimap2
`tp`, `cm`, `s1`, `de`, ...) still describe the source alignment.

## A BAM lifted with `ft pg-lift`

`fiberhmm-check` reports such a BAM (`graph-lift-unrepaired`). Repair it from
the BAM it was lifted from:

```bash
fiberhmm-lift --repair-only -i lifted.bam --source calls.hg002_mat.bam \
    -o lifted.repaired.bam --target-fasta GRCh38.fa
```

Each lifted record is matched with its source record (same read name and
sequence; the aligned part of the molecule decides between supplementaries).
Hard clips the lift dropped are added back as hard clips.

## BED12 tracks

`ft pg-lift`'s BED mode moves only each row's start and end, so the blocks of
`fiberhmm-extract`'s BED12 rows stay at source offsets. `fiberhmm-lift --bed`
lifts every block as its own feature and rebuilds the rows:

```bash
fiberhmm-lift --bed -i calls_nucleosome.bed -o calls_nucleosome.grch38.bed \
    -g hprc.gbz --prefix 'HG002#1#' --target GRCh38 --source-fasta hg002.fa
```

A row is dropped when one of its blocks does not lift, or its blocks land on
different contigs or out of order. `ft pg-lift`'s BED mode currently lifts
nothing on segments that change strand. Lifting the BAM and extracting from
the lifted BAM avoids both limits and is the better route when the BAM is at
hand.
