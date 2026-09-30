# Coordinate frames

FiberHMM output uses three coordinate systems. Mixing them up misplaces every
call on reverse-strand reads, so it is worth knowing which is which.

| Frame | Origin and direction | Used by |
|---|---|---|
| **Molecular** (original fiber) | the first base of the molecule as sequenced, 5'→3' of the read | per-read tags: `ns`/`nl`, `as`/`al`, `MA`/`AQ`/`AN` |
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

Writers mark the frame in the header:

- `fiberhmm-call` puts `coord=molecular` in its `@PG` `DS` field;
- `fiberhmm-apply` and `fiberhmm-recall-tfs`/`-recall-nucs` add
  `@CO fiberhmm:coord=molecular`.

Readers accept either marker. A BAM with neither (FiberHMM 1.x output) is
treated as query frame; `fiberhmm-recall-tfs --input-frame` and
`fiberhmm-tag-m5c --input-frame` let you force `molecular` or `query` for
BAMs whose header was stripped.

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
