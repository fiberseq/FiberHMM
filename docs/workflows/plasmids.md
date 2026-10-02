# Plasmids

A plasmid is a circle, but its map and FASTA are linear: they start at an
arbitrary position (often the first base of `ori`). `fiberhmm-pipeline` takes
the map itself as the reference and handles the two consequences, naming and
the origin, so a plasmid run needs no hand-made FASTA, no rotated reference
and no renaming:

```bash
fiberhmm-pipeline reads.fastq.gz --reference construct.dna --enzyme dddb -o run/
```

## The reference from a map

Supported maps: SnapGene `.dna`, GenBank `.gb`/`.gbk`/`.genbank`, EMBL
`.embl`/`.emb`. The pipeline writes `run/<contig>.fa` (+ `.fai`) and copies
the map into `run/`. The contig is named the way FiberBrowser names the map
when you load it, so the BAM and the map line up without renaming:

| Map | Contig name | Example |
|---|---|---|
| SnapGene `.dna` | file name without extension, cut at the first space, characters outside `A–Z a–z 0–9 _ . -` replaced by `_`, leading/trailing `_` removed | `L-HH (v2).dna` → `L-HH` |
| GenBank | the `LOCUS` name (same character rule, not stripped) | `LOCUS  pUC19 ...` → `pUC19` |
| EMBL | the `ID` name | `ID  pX1; ...` → `pX1` |
| FASTA (as a map) | the first header word | `>pRef plasmid` → `pRef` |

The rule is `fiberhmm.pipeline.reference.fiberbrowser_contig_name`.

The map's topology decides whether the contig is circular: SnapGene's
topology flag, `circular`/`linear` on the GenBank `LOCUS` or EMBL `ID` line.
`--topology circular` makes every contig circular (for a plasmid given as a
FASTA); `--topology linear` turns the origin handling off.

## Reads through the origin

Whole-plasmid libraries are cut at random positions, so many reads run
through the position where the map starts. minimap2 returns such a read in
two pieces, one ending at the last base of the map and one starting at its
first base. A linear pipeline keeps only the longer piece and trims the rest;
in a Plasmidsaurus library that loses part of about a third of the reads, and
coverage dips next to the origin. Rotating the reference (the "orirot" FASTAs
some projects make) only moves the cut.

The pipeline joins the two pieces into one record instead, in the form the
SAM specification prefers for circular references (SAMv1 §1.4, "Circular
reference sequences"): the contig is declared `@SQ ... TP:circular`, `POS`
lies within the contig, and the alignment may run past its end; a position
`p` past the end means `p - length`. The two pieces are joined when

- they are on the same strand, one ends within 100 bp of the contig end and
  the other starts within 100 bp of its start, and
- they are consecutive on the read (a gap or overlap of at most 100 bp).

Bases the two pieces both claim keep the primary's placement; a gap becomes
aligned, inserted or deleted bases, and `MD`/`NM` are recomputed against the
reference, so the join is scored like any other part of the alignment. A
molecule covers at most one full circle: anything beyond is concatemer
sequence and is trimmed with the other unaligned arms. The joined record
keeps the primary's flags and MAPQ; the supplementary record is dropped.
Other supplementary records on a circular contig (concatemer copies of the
same plasmid sequence) are dropped too, and DAF reads there are hard-clipped,
so one molecule is not annotated twice; on linear contigs the pipeline keeps
supplementary records and soft clips.

`fiberhmm-call` then calls the whole molecule. FiberBrowser draws it
continuously across the origin in a circular view (a view that runs past the
end of the plasmid), and `fiberhmm-extract` splits each track row at the
origin, so bigBed files stay valid.

`--no-origin-merge` keeps only the primary piece (the linear behaviour).

### What it changes, on real data

Two DddB runs of the Tohn collaboration, against their earlier processing
(minimap2, primary MAPQ ≥ 20, hard clip, `fiberhmm-call`, on a reference
rotated to cut in the middle of `ori`):

| | D plasmid, 11.4 kb | N1 plasmid, 12.2 kb |
|---|---|---|
| reads kept (primary, MAPQ ≥ 20) | 1,072 (same) | 1,271 (same) |
| reads joined across the origin | 346 (32%) | 1,159 (91%) |
| median aligned span | 3,403 bp (was 3,251 bp) | 12,179 bp, one full circle (was 11,765 bp) |
| coverage of the first kb (after the cut) | 353 (was 212) | 1,183 (was 693) |
| min/max coverage of 500 bp bins | 0.83 (was 0.63) | — |

The N1 library is a PCR product that runs around the whole construct, so the
earlier processing cut 414 bp off nearly every molecule; joined, each read is
one complete circle.

## In the BAM header

```text
@SQ	SN:N1	LN:12179	M5:1276ace935ddb3051497a9b2b956a0c2	TP:circular
@CO	FIBERHMM-REFERENCE:v1:contig=N1;length=12179;md5=1276ace935ddb3051497a9b2b956a0c2;topology=circular;source=N1.dna;source_format=snapgene;source_sha256=02bb9705b987b8df586ff0723c7d24de49f5683a85654c669a5bfc8e80243d88
```

`M5` is the MD5 of the contig sequence (the SAM standard) and the `@CO` line
names the map it came from and that file's SHA-256, so a viewer can pair a
BAM with its map even after files are renamed
([Header declarations](../reference/headers.md#fiberhmm-reference)).

## Several constructs or a plasmid plus a genome

Keep one reference per run. When a construct shares sequence with the genome
(a genomic insert, `mini-white`, an `hsp70` promoter), a combined
genome-plus-plasmid reference pushes the shared reads to MAPQ 0 and they are
dropped. Run the plasmid and the genome separately; reads that do not map to
the given reference are counted as unmapped in the log.
