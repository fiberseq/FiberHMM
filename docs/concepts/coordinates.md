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
header's `@PG` provenance, with the rule it shares with FiberBrowser
([Footprint-tag coordinate frame](../reference/footprint-tag-frame.md)):

- The `@PG` history is followed through its `PP` links, not header order.
  Each chain (`samtools merge` keeps one per input) votes with the frame of
  its last footprint writer. fibertools-rs nucleosome commands (`ft
  predict-m6a`, `m6a`, `add-nucleosomes`, `fire`, `fiber-hmm`, the older
  `predict` and `add`) write molecular frame. `fiberhmm-call`, `-apply`,
  `-recall-tfs` and `-recall-nucs` write molecular frame when they declare
  `coord=molecular`, otherwise query frame. Other FiberHMM tools pass the
  tags through and do not vote.
- If the votes agree, they decide. With no votes, a `coord=molecular`
  declaration anywhere means molecular, otherwise query frame (unmarked
  FiberHMM 2.12 output).
- Merged histories whose votes disagree are molecular when `coord=molecular`
  is declared. Otherwise FiberHMM does not guess: `fiberhmm-recall-tfs` and
  `-recall-nucs` stop at the first read that carries `ns`/`nl`/`as`/`al` and
  ask for `--input-frame query` or `--input-frame molecular`. Reads whose
  only footprints are fibertools `Ma` tags need no frame.

Tools that copy footprint tags without rewriting them (`fiberhmm-dedup`,
`-pair`, `-merge`, `-tag-m5c`, `-call-m5c`, `-tag-consensus`,
`-strand-rescue-annotate`) add `coord=molecular` to their own `@PG` `DS` when
the tags they carried are molecular. They record nothing otherwise.

Consensus reads legacy tags only for Hia5 input without `MA`. It applies
molecular frame by itself only when fibertools is the last writer on every
chain. Reads a FiberHMM caller left without `MA` are ones it skipped, and
they keep whatever tags they had before. Otherwise it asks you to set its
legacy Hia5 annotation frame. `fiberhmm-tag-m5c --input-frame` and
`fiberhmm-call-m5c --tag-input-frame` force the frame of their DAF inputs.

One case cannot be detected from the header. FiberHMM 2.12 or earlier wrote
query-frame tags and no `@PG`. Run on a BAM that had already been through
`ft predict-m6a`, it leaves a header that shows only the fibertools command.
Pass `--input-frame query` for such files.

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
place. Consensus BAM export copies a read's `Ma` annotations into the `MA`
it writes, so the family layers do not hide them. The copy is in the file's
`MA` frame, which is query frame unless the header declares
`coord=molecular`. The family layers are molecular, as their
`FIBERHMM-CONSENSUS-MA` contract declares.

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
