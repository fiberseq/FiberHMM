# Coordinate frame of footprint tags

This is the rule FiberBrowser (`browser/services/bam_tags.py`,
`legacy_tag_frame_report`) and FiberHMM (`fiberhmm.io.annotation_frame`,
`legacy_tag_frame_report`; used by `fiberhmm-recall-tfs` /
`fiberhmm-recall-nucs --input-frame auto`, the pass-through tools' `@PG`
records, `fiberhmm-tag-m5c` / `-call-m5c` auto frames and consensus) use to
decide how to read a BAM's footprint tags on reverse-strand reads. Both
implementations must follow it; the same page is in both repositories.

## Frames

- **Molecular frame:** offsets count from the start of the original molecule,
  so a reverse read's offsets count from its genomic right end. To get SEQ
  offsets, use `seq_start = read_length - (start + length)`.
- **SEQ frame:** offsets count along the BAM SEQ (the reference-forward
  orientation). v1.0 FiberHMM wrote this frame and marked nothing.

| Tags | Frame |
|---|---|
| `Ma`/`Aq`/`An` (fibertools-rs ≥ 0.13) | Always molecular. |
| `MA`/`AQ`/`AN` | Molecular when `coord=molecular` appears anywhere in `@PG` or `@CO`, otherwise SEQ. This is FiberHMM's `ma_annotation_frame`, and this rule does not change it. |
| Legacy `ns/nl`, `as/al` and the parallel `nq`, `aq`, `lq`, `rq` | Decided by the rule below. The quality arrays follow their intervals' order. |

**Consensus family layers** in `MA` carry their own frame. The
`@CO FIBERHMM-CONSENSUS-MA:v1:` contract lists them with
`"coordinates": "original_source_call_in_molecular_frame"`, and both readers
treat those layers as molecular ahead of the header rule: FiberBrowser in
its compact read loader, FiberHMM in `consensus_molecular_layers`. The rest
of an exported `MA` follows the header rule. FiberHMM's export copies a
fibertools read's `Ma` groups into the `MA` it writes, because `MA` wins over
`Ma`. It writes the copy in that file's `MA` frame, which is SEQ unless
`coord=molecular` is declared, and adds no `coord=` token of its own.

Evidence that fibertools-rs writes molecular frame:

- In the 0.6.2 source, `add_nucleosomes_to_record` gets its input from
  `m6a.get_forward_starts()`. `Ranges::new` reads `ns`/`as` as forward starts,
  and `fire.rs` reverses `aq` on reverse reads.
- The 0.13 binary writes `Ma` at molecular offsets. `convert-tags` and
  `extract -r` read legacy `ns` as molecular.

See `tests/fixtures/fibertools_frame`.

## Only one declaration token exists

The token is `coord=molecular`, matched case-insensitively. It has two scopes:

- **Record level:** the token appears in a FiberHMM `@PG` record, in its `DS`
  field for `fiberhmm-call`, `-apply` and `-recall-*`.
- **Header level:** an `@CO` line contains it, for example
  `@CO fiberhmm:coord=molecular`.

No token declares SEQ frame.

## 1. Producer identity of an `@PG` record

Names are matched case-insensitively. Collision suffixes added by samtools and
pysam (`\.\d+` or `-[0-9a-f]{6,8}`, stripped repeatedly) are removed from `ID`
only when classifying a record. They are never removed when resolving `PP`.

1. **If `PN` is present, it alone decides:**
   - If it contains `fibertools`, the record is fibertools.
   - If it matches `^fiberhmm(-[a-z0-9-]+)?$`, the record is FiberHMM.
   - Anything else is another program, whatever `ID` or `CL` say. For example,
     `PN:samtools ID:samtools-fiberhmm-output` is not FiberHMM.
2. **If `PN` is absent, use the `ID` base:**
   - `ft`, or a base starting with `fibertools`, is fibertools.
   - A base matching `^fiberhmm(-…)?$` is FiberHMM.
3. **If that fails, use the basename of the first `CL` token:**
   - `ft`, `fibertools` or `fibertools-rs` is fibertools.
   - `fiberhmm(-…)?` is FiberHMM.
4. **Otherwise** the record is another program and never writes footprint tags.

## 2. Footprint writers

- **fibertools record:** it is a writer when any `CL` token after the program
  token is one of these subcommands:
  - `predict-m6a`, `m6a`, `predict` (fibertools-rs 0.1–0.3)
  - `add-nucleosomes`, `add-nucleosome`, `add` (0.2)
  - `fire`
  - `fiber-hmm`

  The token may also be an unambiguous prefix of one of them, as fibertools
  itself accepts: at least `add` for add-nucleosomes, `pred` for predict-m6a,
  `fir` for fire, and `fiber` for fiber-hmm. `ft add-nuc` appears in real
  headers.

  Its frame is always **molecular**. A fibertools record without `CL`, or with
  any other subcommand (`extract`, `convert-tags`, `pileup`, …), is not a
  writer.
- **FiberHMM record:** it is a writer when either of these holds:
  - The record contains `coord=molecular`.
  - One of its names (`PN`, the `ID` base, or the basename of the first `CL`
    token) is `fiberhmm-apply`, `fiberhmm-call`, `fiberhmm-recall-tfs` or
    `fiberhmm-recall-nucs`.

  Its frame is **molecular** if the record or an `@CO` line declares
  `coord=molecular`, and **SEQ** otherwise.

  All other FiberHMM programs pass the legacy tags through unchanged and are
  not writers: `tag-m5c`, `call-m5c`, `dedup`, `merge`, `pair`,
  `strand-rescue-annotate`, `tag-consensus`, `pipeline`, and so on.

## 3. Chains

**Parent links:**
- If **any** `@PG` record has a `PP` field, `PP` links define ancestry. A record
  without `PP` is a root.
- If a `PP` names an `ID` that several records share, it links to the nearest
  such record earlier in the header, or to the last one if none is earlier.
- A `PP` that names a missing `ID`, or the record itself, ends the chain.

**Fallback:** only when **no** record has a `PP` field is header order the
single chain, with each record the child of the one before it.

**Walking the chains:**
- A **leaf** is a record that no other record names as its parent.
- From each leaf, walk up the parents. The walk stops at a root, a missing
  parent or an already-visited record, so cycles terminate.
- If every record is some record's parent (a pure cycle), the last record in
  the header is the only leaf.
- A chain's vote is the frame of the **first writer met walking from the
  leaf**, which is the last writer that ran. A chain with no writer does not
  vote.

## 4. Decision

| Votes | Frame | Source | Ambiguous |
|---|---|---|---|
| All agree | That frame | `provenance` | No |
| None, but `coord=molecular` appears anywhere | Molecular | `declared` | No |
| None, no declaration | SEQ | `default` (unmarked v1.0 FiberHMM) | No |
| Disagree, `coord=molecular` appears anywhere | Molecular | `declared` | **Yes** |
| Disagree, no declaration | Majority of chain votes; SEQ on a tie | `majority` | **Yes** |

**FiberHMM differs in the last row only.** A wrong frame changes calls, so
`legacy_tag_frame` returns no frame there: `fiberhmm-recall-tfs` /
`-recall-nucs --input-frame auto` stop at the first read that carries
`ns`/`nl`/`as`/`al` (reads whose only footprints are fibertools `Ma` tags need
no frame) and ask for `--input-frame query` or `molecular`. Pass-through tools
record nothing and `fiberhmm-tag-m5c` / `-call-m5c` keep their historical
fallback (the `coord=molecular` marker, else SEQ).

FiberHMM consensus reads legacy tags only for Hia5 input without `MA`/`Ma`,
under its `legacy_hia5_annotation_frame` option. When that is `disabled`, it
applies molecular frame only when **every** chain's last writer is fibertools
and no `@CO` marker is present, which is stricter than this rule. A FiberHMM
writer's `coord=molecular` covers the reads it tagged. The reads it left
without `MA` are ones it skipped, and they keep whatever tags they had before
(for example SEQ-frame tags from FiberHMM ≤ 2.12). Otherwise consensus stops
and asks for the option.

FiberHMM pass-through tools that carried molecular-frame tags add
`coord=molecular (footprint tags carried over from the input)` to their own
`@PG` `DS`. By this rule that makes them writers of the frame they carried,
so it gives the same answer as looking through them. Consensus's stricter
check does look through them.

FiberBrowser reports this in `diagnose_bam_tags(...)["legacy_tag_frame"]`
(fields `frame`, `source`, `ambiguous`, `votes`, `writers`, `chains`). When the
frame is ambiguous and legacy tags are present, the Diagnose BAM
recommendations include a warning.

## Examples

**Example 1:** `ft0.6.2_addnuc_fire.bam`.

```text
@PG ID:ft.1 PN:fibertools-rs VN:0.6.2 CL:ft add-nucleosomes -t 1 input.bam n062.bam
@PG ID:ft.2 PN:fibertools-rs VN:0.6.2 PP:ft.1 CL:ft fire -t 1 n062.bam f062.bam
```

`ft.2` is the only leaf. It is a fibertools writer, so the chain votes
molecular and the frame is molecular (source `provenance`). The result is the
same if the two lines are swapped, if both lose `PN`, or if a
`PN:samtools ID:samtools-fiberhmm-output PP:ft.2` record is appended.

**Example 2:** a Hia5 amplicon BAM merged from several runs (abridged).

```text
@PG PN:samtools ID:samtools PP:pbmm2 CL:samtools cat ...
@PG PN:fiberhmm-call ID:fiberhmm-call PP:samtools DS:...; coord=molecular ...
@PG PN:fibertools-rs ID:ft.1 PP:lima CL:ft fire - -
@PG PN:samtools ID:samtools.1 PP:fiberhmm-call CL:samtools sort ...
@PG PN:samtools ID:samtools.2 PP:ft.1 CL:samtools sort ...
@PG PN:fiberhmm-call ID:fiberhmm-call.2 PP:samtools.2 DS:...; coord=molecular ...
...
```

Nine leaves give five writer votes, all molecular, so the frame is molecular
and not ambiguous.

**Example 3:** an unmarked `fiberhmm-apply` that ran after `ft predict-m6a`.
Its PP points at `ft.1`, even though its record is listed first in the header.
The chain's last writer is the unmarked FiberHMM run, so the frame is SEQ.

## Known limit

FiberHMM before June 2026 wrote SEQ-frame tags and **no** `@PG`. Run on a
fibertools BAM, it leaves only the fibertools record in the header, so this
rule reads its tags as molecular. No header signal can tell these files apart.
FiberHMM releases since June 2026 always write a `@PG` record, and their footprint writers declare `coord=molecular`.
