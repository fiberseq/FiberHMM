# Coordinate frames

FiberHMM output uses three coordinate systems. Mixing them up misplaces every
call on reverse-strand reads, so it is worth knowing which is which.

| Frame | Origin and direction | Used by |
|---|---|---|
| **Molecular** (original fiber) | the first base of the molecule as sequenced, 5'→3' of the read | per-read tags: `ns`/`nl`, `as`/`al`, `MA`/`AQ`/`AN`, and fibertools' `Ma`/`Aq`/`An` |
| **Query** (SEQ) | the first base of the `SEQ` field as stored in the BAM | `MM`/`ML` positions after decoding, pysam `query_*` indexes, posteriors arrays |
| **Reference** | the reference contig, 0-based half-open | BED/bigBed tracks, consensus tables, SNP masks, `--region`/`--bed` inputs |

## Molecular versus query

For a forward-aligned read the two are identical. For a reverse-aligned read
(flag `0x10`) the BAM stores the reverse complement of the molecule, so the
query frame runs backwards along the molecule. An interval `[s, s+l)` in
query coordinates is written in molecular coordinates as

```text
[L − (s + l), L − s)        L = read length
```

and the left and right edge bytes (`el`, `er`) are swapped, because the
molecule's left edge is the query's right edge. Every list is sorted by
molecular start. This matches the
[Molecular-annotation spec](https://github.com/fiberseq/Molecular-annotation-spec)
and fibertools.

## How FiberHMM tells the frame of a BAM

The `ns`/`nl`/`as`/`al` arrays carry no frame of their own, and two
generations of writers used different frames: FiberHMM 2.12 and earlier
wrote query frame, while FiberHMM 2.13 and later and every fibertools
version write molecular frame. FiberHMM therefore reads the frame from the
header's provenance:

1. Any `@PG` or `@CO` line containing `coord=molecular` means molecular.
   `fiberhmm-call` puts it in its `@PG` `DS` field; `fiberhmm-apply` and
   `fiberhmm-recall-tfs`/`-recall-nucs` add `@CO fiberhmm:coord=molecular`.
2. Otherwise the `@PG` lines are read in order:
    - a fibertools-rs record whose command writes nucleosomes (`ft
      predict-m6a` or `m6a`, `ft add-nucleosomes`, `ft fire`, `ft fiber-hmm`,
      and the older names `ft predict` and `ft add`) means molecular;
    - a FiberHMM record (`fiberhmm-*`) whose `DS` says `coord=seq` means
      query frame;
    - a FiberHMM record with no `coord=` token copied the tags unchanged and
      leaves the frame as it was.
3. If nothing above decides it, the frame is unknown.

Tools that copy footprint tags without rewriting them (`fiberhmm-dedup`,
`-pair`, `-merge`, `-tag-m5c`, `-call-m5c`, `-tag-consensus`,
`-strand-rescue-annotate`) add `coord=molecular` or `coord=seq` to their own
`@PG` `DS` to record the frame they carried over. If they could not tell the
frame, they add nothing.

An unknown frame is not guessed. Output of FiberHMM 2.12 or earlier (no
FiberHMM `@PG`, query frame) looks the same as a fibertools BAM whose `@PG`
history was lost (molecular frame). In that case `fiberhmm-recall-tfs` and
`-recall-nucs` stop and ask for `--input-frame query` or `--input-frame
molecular`. Consensus, which reads legacy tags only for Hia5 input without
`MA`, stops and asks you to set its legacy Hia5 annotation frame. It applies
molecular frame by itself only when fibertools wrote the tags and no FiberHMM
caller ran after it: reads a FiberHMM caller left without `MA` are ones it
skipped, and they keep whatever tags they had before. `fiberhmm-tag-m5c --input-frame` and `fiberhmm-call-m5c
--tag-input-frame` force the frame of their DAF inputs in the same way.

One case cannot be detected from the header. FiberHMM 2.12 or earlier run on
a BAM that had already been through `ft predict-m6a` wrote query-frame tags,
but the header still shows only the fibertools command. Pass
`--input-frame query` for such files.

`MA`, `AQ` and `AN` are written only by FiberHMM, so their frame comes from
the `coord=molecular` declaration alone (unmarked `MA` is query frame).

### fibertools `Ma` tags

fibertools-rs 0.13 and later no longer write `ns`/`nl`/`as`/`al`. Nucleosomes,
MSPs and FIRE elements go into `Ma`/`Aq`/`An` instead, for example
`Ma:Z:8000;nuc.:297-154,...;msp.:451-16,...;fire.Q:804-397,...` with one
`Aq` byte per `fire.Q` element. These tags follow the same
Molecular-annotation spec as FiberHMM's `MA` and are always molecular frame.
FiberHMM reads `Ma` wherever it reads `MA` or the legacy arrays: in
recall-tfs/recall-nucs, consensus evidence loading and extract. If a read
has both, FiberHMM's own `MA` is used. Recall writes its calls to
`MA`/`AQ` (and `ns`/`nl`/`as`/`al`) and leaves the fibertools `Ma` tag in
place.

## Converting to query or reference coordinates

```python
import pysam
from fiberhmm.io.ma_tags import flip_intervals_to_seq, parse_ma_tag

for read in pysam.AlignmentFile("calls.bam"):
    if not read.has_tag("MA"):
        continue
    tf = parse_ma_tag(read.get_tag("MA"))["tf"]        # [(start, length)], 0-based, molecular
    if not tf:
        continue
    starts, lengths = flip_intervals_to_seq(*zip(*tf), read)   # query frame
    q2r = read.get_reference_positions(full_length=True)    # query index -> reference
    ref_intervals = [(q2r[s], q2r[s + l - 1] + 1) for s, l in zip(starts, lengths)
                     if q2r[s] is not None and q2r[s + l - 1] is not None]
```

`flip_intervals_to_seq` returns forward reads' intervals unchanged and
applies the formula above to reverse reads (the formula is its own inverse).
`fiberhmm-extract` does this projection for you and writes reference
coordinates.

## MA strings are 1-based

Inside the `MA` tag, each interval is written `start-length` with a
**1-based** start, as the spec requires. `parse_ma_tag` returns 0-based
starts. The legacy `ns`/`as` arrays are 0-based.

```text
MA:Z:3744;nuc.QQQ:87-146,...        # first nucleosome: 1-based start 87
ns:B:I,86,...                       # the same nucleosome: 0-based start 86
```

## Other frames

- **Posteriors** (`fiberhmm-posteriors`): one value per query position; the
  footprint columns are reference intervals.
- **Consensus** (`fiberhmm-consensus`): all tables are reference coordinates,
  except pooled cross-locus runs, which use the oriented window frame
  `0 … width` (a minus-strand window maps base *i* to `end − 1 − i`).
- **Strand rescue** edge bytes `q1`/`q2` are molecular-left and
  molecular-right.
- **Circular molecules** (`fiberhmm-call -r`): a feature that crosses the
  read origin is written as two intervals, one at each end of the read, that
  share an `AN` name (see [Annotations](annotations.md#circular-molecules)).
