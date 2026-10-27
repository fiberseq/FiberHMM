# Scan directory format

The output of [`fiberhmm-scan`](../workflows/scanning.md) (schema
`fiberhmm.scan.v1`). FiberBrowser reads and writes the same format.

## Layout

```
scan_dir/
  manifest.json                 # read this first; no timings
  run.json                      # attempts and timings (not part of the result)
  units.tsv.bgz(.tbi)           # every planned unit
  callable.tsv.bgz(.tbi)        # per dataset: bases with >= callable_min_molecules molecules
  <module>/
    hits.tsv.bgz(.tbi)          # pooled hits after seam reconciliation
    values.tsv.bgz(.tbi)        # per (hit, group) values, long format
    classes|variants.tsv.bgz    # the engine's own table, columns verbatim (fp: classes, nfr: variants)
    states.tsv.bgz(.tbi)        # tables mode only: per (hit, source record) state
    marks.tsv.bgz(.tbi)         # tables mode only: molecule-wide interval marks
    molecules.tsv.bgz(.tbi)     # tables mode only: per source record scalars, exact
    units.tsv.bgz(.tbi)         # tables mode only: engine evidence unit -> member source records
    extra/                      # module records named in the manifest (peaks: calibration.json)
  annotated/                    # annotated mode: one BAM (+ .csi) per input file
  work/                         # per-unit results and resume markers (internal)
```

Two runs with the same inputs, units, read policy and settings write
byte-identical `manifest.json`, tables, indexes and annotated BAMs. Units are
named by the zone they own (`chrom:zone_start-zone_end`).

## Tables

BGZF, one header line starting with `#`, tabix-indexed on columns 1, 4, 5
(`tabix -0 -s1 -b4 -e5`; `.tbi`, or `.csi` for a table with positions at or
beyond 2^29), sorted by contig (header order) and position. Contig names
starting with `#` are refused.
Every table starts with `chrom start end idx_start idx_end`, 0-based
half-open. `idx_start`/`idx_end` are what tabix indexes: equal to
`start`/`end`, except that a zero-width row at `p` indexes `[p, p + 1)` so a
fetch at `p` finds it.

| Table | Columns after the first five |
|---|---|
| `units` | `unit_id zone_start zone_end mode status records` (`status`: done, empty, failed, planned) |
| `callable` | `dataset_id min_molecules` |
| `hits` | `hit_id type anchor strand unit_id score label flags parent_id` + the module's columns |
| `values` | `hit_id group group_kind value_name value lo hi n_molecules n_callable` |
| `states` | `hit_id dataset_id input_index qname record molecule state` + module columns (`state`: 1, 0, .) |
| `marks` | `group dataset_id input_index qname record molecule mol_start mol_len q name` |
| `molecules` | `dataset_id input_index qname record molecule` + the module's scalars |
| module `units` | `evidence_unit dataset_id input_index qname record molecule` |

`input_index` is the record's file within its dataset (its position in the
dataset's `files`), so a record of a multi-file dataset is found without
reading the BAMs, and identical records in two files stay apart.

Module columns:

| Type | Columns |
|---|---|
| `fp.class` | `class_id group start_exact end_exact L0 L1 R0 R1 calls stability status supported_channels` |
| `nfr.region` | `nfr status source callable reads censored overflow_reads profile_peak k closed` |
| `nfr.variant` (parent: its region) | `nfr name L R Lsd Rsd width depth relation prevalence_strict prevalence_em ci_lo ci_hi only with_ref gain_nats stability exp_reads robust` |
| `peaks.peak`, `peaks.broad` | `tier tier_label threshold width summit mean_reads tier_rule row_rule` |

## Identities

| Identity | Form |
|---|---|
| hit | `hit_id` = `<module>:<type>:<chrom>:<anchor>:<8 hex>` |
| source record | `record` = `<SHA-256 of the record's SAM text>:<occurrence>`, with `qname`, `dataset_id` and `input_index` |
| physical molecule | `molecule` = `mol:v1 <16 hex>`: the digest of the molecule key (`pacbio/<movie>/<zmw>` for PacBio names, else `read/<name>`); records of one engine evidence unit (DAF duplicates, duplex sources) share the smallest key |
| dataset file | `content_id` = `cid:v1:` + SHA-256 of the file size, its first MiB and its last MiB |

`record` always names the record in the **input** BAM. An annotated record
has different bytes; the manifest lists every annotated BAM with its own
`content_id`.

## Manifest

`manifest.json` holds the schema, `scan_id`, the FiberHMM version, the
reference contigs and their digest, the frame (`linear` or `lifted`), the
datasets (canonical `ordinal`, chemistry, files with `content_id`), the read
policy (filters, molecule caps, `callable_min_molecules`), the plan (`mode`,
`unit_bp`, `stride_bp`, `grid_offset`, contigs or zones or regions or BED
digest, merged intervals), each module's version, resolved settings and
`settings_hash`, feature types, columns and MA groups (and, for modules with
scan-wide statistics, `scan_statistics`: nuc's Benjamini-Hochberg threshold
per kind), the outputs (tables with row counts and SHA-256, annotated BAMs)
and the status (units done, empty and failed; seam components; open ends).

Datasets are ordered by their files' content ids (`ordinal`), and engines
see them in that order. A dataset given without a label (`--bam`, or the
Python API without `dataset_id`) is labelled `dataset_<ordinal + 1>`, so
neither labels, nor unit keys, nor rows depend on the order of `--bam`.
`--datasets` keeps the labels it gives.

**Reuse.** A unit's result is keyed by the inputs, labels and read policy,
the module (version, settings, calibration digest) and the unit itself (id,
span, zone), not by the plan. A scan extended with more units
(`run_units(..., continue_run=True, extend=True)`), or a `--zones` subset of
the genome grid, reuses every unit it shares with an earlier scan in the
same directory. Hits away from a plan's edges (units whose two grid
neighbours are planned) are the same in both; at an edge, anchor hits fall
back to the anchor-in-zone rule and stitched hits reaching the edge are
flagged `open_end`. nuc's patches are a Benjamini-Hochberg selection over the
windows a scan tests, so a smaller scan can select other patches from the
same window statistics; its molecule-wide rows (marks, molecules, annotated
groups) do not depend on the plan.

## MA groups

Scan groups are extra groups of the `MA`/`AQ`/`AN` tags, in molecular
coordinates. Each written group is declared twice in the header: in
`MA-TYPES:v1` and in one `@CO FIBERHMM-SCAN-MA:v1:<json>` line:

```json
{"group": "nfrmol", "group_version": 1, "kind": "scalar", "module": "nfr", "module_version": "1.0",
 "scan_id": "…", "settings_hash": "…", "coordinates": "molecular",
 "quality_names": ["openness", "n_nfr", "longest_linker_hi", "longest_linker_lo"],
 "codecs": [{"name": "openness", "nbytes": 1, "scale": 0.004, "offset": 0.0, "unit": "fraction"}, …],
 "state_codes": {}, "an": "group", "calls_program": "fiberhmm-call"}
```

- **Names.** A group's name starts with its module's name (`fpcls`,
  `nfrvar`, `nfrmol`; the nucleosome module's `nucsz`, `nucseg`, `nucmol`).
  The name never changes; a change of byte layout or meaning bumps
  `group_version`.
- **State groups** (`fpcls`, `nfrvar`): one interval per (hit, molecule);
  byte 0 is the state (`1` = state 1, `2` = state 0, `3` = abstain); `AN` is
  the hit id.
- **Scalar groups** (`nfrmol`): one interval covering the record's
  molecule. Each scalar has 1 or 2 bytes: `value = offset + scale * q`,
  `q` = the byte (0-254; 255 = missing) or `256 * hi + lo` (0-65534; 65535 =
  missing).
- **Staleness.** Scan groups describe the calls they were computed from. A
  reader treats them as stale when a footprint-writing `@PG` (other than
  `fiberhmm-lift`) follows the one named in `calls_program`.
- Re-annotating a BAM replaces the groups it writes and keeps every other
  group.

## Python API

```python
from fiberhmm.scan import plan, run_units, open_scan
units = plan(reference, genome=True, contigs=['chr3'])      # reference: contig -> length
run_units([{'dataset_id': 'A', 'paths': ['a.bam']}], units, modules=['fp', 'peaks'], out='scan_dir')
scan = open_scan('scan_dir')
scan.hits('fp', 'chr3', 0, 1_000_000)
```

A module implements `fiberhmm.scan.modules.ScanModuleBase`: `resolve`
(settings), `feature_types`, `ma_groups`, `needs`, `calibrate` (once per
scan; `calibration_uses_plan` says whether the record depends on the plan),
`run_unit` (one unit), `molecule_outputs` (one complete source record,
called once per record genome-wide) and, for modules whose hits are joined
across units, `stitch`. A stitch module with statistics over the whole scan
declares `scan_statistics_tracks`; the scan then runs `stitch_statistics` on
every contig first, `combine_statistics` once, and passes the result to
`stitch` as `scan_statistics=`.

`fiberhmm.scan.state_at(read, 'nfr.region', chrom, start, end)` gives the
state of a transferable feature type on one record without a scan (`1`
open, `0` closed, `.` not callable), by the rule and with the nucleosomes
the scan uses (`fiberhmm.scan.states`).
