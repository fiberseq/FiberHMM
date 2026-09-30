# Tag consensus states: `fiberhmm-tag-consensus`

`fiberhmm-tag-consensus` stores a frozen assignment of `tf_sr` calls to
recurrent footprint families inside the BAM. It extends the strand-rescue
shadow layer `tf_sr.QQQ` to `tf_sr.QQQQQ` by appending two bytes to every
`tf_sr` annotation:

- `fi`: a family slot, 1–255, reusable along a contig (0 = unassigned);
- `fq`: `round(255 × assignment_probability)` (at least 1 for an assigned
  call), the producer's declared assignment confidence, not a biological
  occupancy.

Ordinary `tf` calls and every `tf_sr` interval stay unchanged. A family is
identified by its genomic neighbourhood plus `fi`, so the same `fi` can be
reused by families at least 24 bp apart on a contig.

## Input

- A BAM written by `fiberhmm-strand-rescue-annotate` (it must declare `tf_sr`
  in `MA-TYPES` and carry exactly one v4 or v6 strand-rescue contract).
- An assignment TSV. No FiberHMM command writes this table: it is the
  interface for a family assignment produced elsewhere (for example a
  cross-fitted analysis), kept separate from BAM writing so it can be audited.

## Assignment TSV

Tab-separated, with exactly this header (v1), or with a final
`calibration_scope` column (v2):

```text
read_name  alignment_occurrence  tf_sr_ordinal  family_id  assignment_probability  family_key  contig  call_start  call_end  [calibration_scope]
```

| Column | Meaning |
|---|---|
| `read_name` | the record's query name |
| `alignment_occurrence` | 0-based occurrence of `read_name` in file order (0 for a name seen once) |
| `tf_sr_ordinal` | 0-based position of the call in the record's `tf_sr` group |
| `family_id` | 1–255, written as `fi` |
| `assignment_probability` | in (0, 1], written as `fq` |
| `family_key` | a stable family identifier; one `family_id` and contig per key |
| `contig`, `call_start`, `call_end` | the call's reference interval, 0-based half-open; must match the BAM exactly |
| `calibration_scope` | (v2) how the probability was calibrated; v1 files record `legacy_unspecified` |

Every row must match a `tf_sr` call; mismatched intervals, duplicate keys,
inconsistent family keys and a `family_id` reused by families closer than
24 bp are errors.

## Run it

```bash
fiberhmm-tag-consensus -i sr_bams/sample.strand-rescue.bam -a assignments.tsv \
    -o sample.families.bam
```

It prints a JSON summary (records, `tf_sr` annotations, assigned calls,
assignment SHA-256) and writes a new sorted, indexed BAM (`--force` replaces
an existing output). The header gains a `FIBERHMM-TF-FAMILY:v1:` line that
declares the byte meanings and the SHA-256 of the assignment table, and an
`@PG` record.

Every option: [`fiberhmm-tag-consensus`](../reference/cli.md#fiberhmm-tag-consensus).
