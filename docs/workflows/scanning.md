# Scanning features genome-wide: `fiberhmm-scan`

`fiberhmm-scan` runs FiberHMM's pooled feature finders over fixed units of
the genome, a set of windows or a BED file, and writes one versioned **scan
directory**. Three modules ship:

| Module | Finds | Engine |
|---|---|---|
| `fp` | footprint classes and each molecule's membership | the lattice recaller, as `fiberhmm-consensus` runs it |
| `nfr` | NFRs, NFR variants and each molecule's state (EXPERIMENTAL, as `fiberhmm-nfr`) | the accessibility engine, as `fiberhmm-nfr` runs it |
| `peaks` | accessibility peaks with Strong/Medium/Weak/Rare tiers | FiberBrowser's peak scan, ported |

Pooled results (classes, NFRs, peaks, callable territory) are written as
bgzip + tabix tables. Per-molecule results (class memberships, NFR states,
per-molecule scalars) are written into the reads themselves, as extra
groups of the `MA` tag in an annotated copy of each input BAM, so they
follow the molecules through [`fiberhmm-lift`](lifting.md) and FiberBrowser's
live lift. FiberBrowser runs the same engine (`fiberhmm.scan.run_units`) for
its own scans, so a browser scan and a command-line scan of the same units
give the same rows.

Inputs must be called by FiberHMM (`fiberhmm-call` or `fiberhmm-apply`);
the scan does not call footprints.

## Quick start

```bash
# every contig of the header on the fixed grid (or --contigs chr3)
fiberhmm-scan --bam out/sample.calls.bam --genome --contigs chr3 --cores 8 -o out/scan

# the genome-grid units whose zones meet a window (FiberBrowser's view scans print this form)
fiberhmm-scan --bam out/sample.calls.bam --genome --zones chr3:128625000-128725000 -o out/scan_view

# a window or a BED file of units
fiberhmm-scan --bam out/sample.calls.bam --region chr3:128650000-128700000 -o out/scan_window
fiberhmm-scan --bam out/sample.calls.bam --bed promoters.bed --modules fp,peaks -o out/scan_promoters
```

`--modules` chooses the modules (default `fp,nfr,peaks`). Repeat `--bam` for
separate datasets; group files into datasets with `--datasets datasets.json`
(`[{"dataset_id": "A", "paths": ["a1.bam", "a2.bam"]}, ...]`). Datasets are
pooled for discovery, as in `fiberhmm-consensus`, and every result carries
values per dataset and channel. `--bam` datasets are labelled `dataset_1`,
`dataset_2`, ... in a fixed order (by file content), not in the order they
are listed, so listing the BAMs in another order gives the same scan.

## Units, zones and seams

- **Genome and long windows.** Units are 50 kb (`--unit-bp`) at a 25 kb
  stride (`--stride-bp`): each unit computes 50 kb and owns its central
  25 kb, so every owned base has 12.5 kb of context on both sides. 50 kb is
  the engines' analysis budget (`compute.maximum_region_bp`) and is never
  raised silently.
- **Zones of the genome grid** (`--zones CHROM:START-END`, repeatable). The
  genome grid's units whose owned zones meet the window, unchanged: same
  ids, spans and zones as in a `--genome` scan, so a genome scan can reuse
  them and they can reuse a genome scan's (see Resuming). A `--region`
  instead plans its own grid from the window's start, clipped to it. At the
  edges of a `--zones` scan the grid goes on unscanned: a peak that may depend
  on what lies beyond the edge (near it, or tied to it through the tiers'
  masks) or a patch that reaches it is flagged `open_end`, and a footprint class or NFR at the edge is kept by its anchor
  (no neighbour to reconcile with).
- **BED units and short windows.** Each interval is a unit and owns all of
  itself. Overlapping or touching intervals are merged; an interval longer
  than `--unit-bp` is planned as a grid restricted to it.
- **Seams.** A feature near a zone boundary is computed by both neighbouring
  units. The scan keeps one version: the features of both units that touch
  the boundary form connected groups by overlap, and each group is taken
  from the unit whose centre is nearest it. Peaks are joined from per-base
  counts instead, so a peak does not depend on where unit boundaries fall.
- **Empty units** (no read in the unit) are recorded and skipped.

## Per-molecule results: annotated BAMs or tables

| Scan | Default output of per-molecule results |
|---|---|
| `--genome`, or a `--region` longer than one unit | annotated BAMs (`annotated/*.scan.bam`) |
| `--bed`, or a `--region` that fits one unit | tables (`<module>/states.tsv.bgz`, `molecules.tsv.bgz`, ...) |

`--annotate` and `--no-annotate` override the default. A scan whose modules
write no per-molecule results (`--modules peaks`) writes no annotated BAM.
Annotated BAMs are written when every input declares molecular coordinates
(`coord=molecular`, written by FiberHMM 2.13.2 or later), or fibertools is the
last footprint caller on every `@PG` chain. Fibertools inputs need no FiberHMM
re-call. PN, collision-suffixed ID and the command executable identify the
caller; linked sorting passes preserve that provenance. Other inputs make the
whole scan write tables. Records whose calls cannot be put
into molecular frame are counted in the manifest
(`status.records_not_carried`). The scan touches no other annotation of a record. An annotated BAM of
a genome scan holds every record of its input; one of a BED or window scan
holds the records overlapping the units. The scan's `MA` groups:

| Group | Interval | Bytes |
|---|---|---|
| `fpcls` | one per (class, molecule): the molecule's class call if it is a member, else the class span | state (1 member, 2 non-member, 3 abstain), class posterior x255, tier (1 core, 2 edge, 3 loose) |
| `nfrvar` | one per (NFR or NFR variant, callable molecule) | state (1 open, 2 closed), P(variant) x255 (255/0 for an NFR) |
| `nfrmol` | one per record, covering the whole record | openness (x0.004), NFR-sized linkers (count), longest linker (bp, two bytes) |

The `AN` name of a state mark is the hit id in `<module>/hits.tsv.bgz`.
Each group is declared in the header (`MA-TYPES:v1` and a
`FIBERHMM-SCAN-MA:v1` comment with its byte meanings and scale factors). See
the [scan format](../reference/scan-format.md).

Bytes are coarse by design. `--no-annotate` writes the exact values to
`<module>/molecules.tsv.bgz` and `<module>/states.tsv.bgz`.

## Peaks

The `peaks` module gives FiberBrowser's accessibility peaks: per base, the
share of reads whose call there is a gap or shorter than 90 bp, smoothed
over 100 bp, then tiered at 50%, 30%, 15% and 8%. Per-position scores equal
FiberBrowser's. Two rules make grid scans independent of where units fall:

- **Rare tier** (`tier_rule`). Every scan uses `local` by default: a base
  takes part in the rare tier only where at least `min_reads` molecules are
  measured, which does not depend on the view. FiberBrowser uses the same
  rule. Its earlier rule, `max(8%, min_reads / reads in the window)`
  (`window`), changes with the view, so it is accepted for BED and
  short-window scans only.
- **Rows** (`row_rule`). Every scan counts every primary and supplementary
  record by default (`records`). FiberBrowser's earlier rule (`view`: a
  supplementary counts only when its primary is out of view) depends on the
  view and is accepted for BED and short-window scans only.
- `auto` (the default before 3.1) picks `window` and `view` for BED and
  short-window scans and `local` and `records` for grid scans; it is kept
  for one release.

Circular contigs (`@SQ TP:circular`) are not scanned for peaks: FiberBrowser
joins records that run past the contig end with their wrapped tails, which
is not ported yet. Tiers Strong to Weak never depend on the rare-tier rule. The row rule
changes the per-base counts where supplementary alignments lie, so it can
change any tier there. `signal_source` is `accessibility` (default) or
`tf_density`.

## Resuming and reruns

A unit's result is kept as soon as it finishes. After an interruption, run
the same command with `--continue`: finished units are reused, the rest are
computed, and the tables, annotated BAMs and manifest are rebuilt. The
result is byte-identical to an uninterrupted run. `--continue` refuses a
command whose inputs, units, read policy or module settings differ; only
`--cores`, `--window-jobs` and `--json-progress` may change.

A unit's result does not depend on the rest of the plan. The Python API's
`run_units(..., continue_run=True, extend=True)` (FiberBrowser's live scans)
runs another plan in the same directory and reuses every unit result whose
inputs, labels, read policy and module settings match, so extending a view
scan computes only its new units.

`--json-progress` prints progress as JSON lines on stderr (FiberBrowser's
job manager reads the same stream).

## Inspecting a scan

```bash
fiberhmm-scan --inspect out/scan       # summary, plus validation of every table, index and annotated BAM
fiberhmm-scan --schema --modules fp    # every parameter with its default, as JSON
```

```python
from fiberhmm.scan import open_scan
scan = open_scan('out/scan')
scan.hits('fp', 'chr3', 128_600_000, 128_700_000)     # list of dicts, one per class
scan.values('peaks', 'chr3', 128_600_000, 128_700_000)
```

## Parameters

`--parameters params.json` takes one group per module:

```json
{
  "fp": {"recaller": {"stringency": 0.9}},
  "nfr": {"nfr": {"bootstrap": 200}},
  "peaks": {"min_reads": 5, "tier_rule": "local"},
  "read_policy": {"callable_min_molecules": 5}
}
```

`fp` takes `fiberhmm-consensus`'s parameter groups and `nfr` takes
`fiberhmm-nfr`'s (`fiberhmm-consensus --schema`, `fiberhmm-nfr --schema`).
The molecule caps (`--max-discovery-molecules`,
`--max-quantification-molecules`, `--sample-seed`) apply to `fp` and `nfr`
as they do to those commands; an fp or nfr unit then gives the same rows as
`fiberhmm-consensus` or `fiberhmm-nfr` run on that window with the same caps.


Python callers can supply `run_units(..., evidence=records)` or a callback
returning records. Supply each dataset's `header`/`headers` and yield complete
`pysam.AlignedSegment` records; multi-input sources yield
`(dataset_id, input_index, record)` tuples. A sorted snapshot in
`work/_evidence` feeds the ordinary loaders and dataset calibrations. Include
the full calibration population. Optional dataset `content_ids` preserve the
identity of the same immutable source records; never reuse an ID after a lift
or another record change. See [contracts §14](../../CONTRACTS.md) for equality,
continuation and metadata rules. `--inspect` includes `units.zones`.
