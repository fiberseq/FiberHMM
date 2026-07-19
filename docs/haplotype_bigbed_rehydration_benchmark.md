# Internal HP/PS bigBed rehydration benchmark

This note defines a one-off migration path for already-extracted FiberHMM
bigBeds. It is intentionally not a public command: normal production should use
`fiberhmm-extract --haplotype-fields` directly.

## Scope and invariants

Given an existing bigBed and its exact source called BAM, produce a new bigBed
whose only row change is two final signed integer fields, `hp` and `ps`.

- Preserve every existing BED field as text, every row, genomic sort order,
  autoSQL description/sample marker, and block/circular metadata.
- Copy BAM `HP` and `PS` independently. Use `-1` for a missing/non-integer tag.
- Never overwrite the source bigBed. Build and validate a separate output, then
  use an atomic rename only if deployment is explicitly requested.
- Fail on ambiguous read-name joins or inconsistent HP/PS values for the same
  primary read name; do not guess.

## One-off algorithm

1. Read the embedded autoSQL and chromosome dictionary from the source bigBed.
2. Recover every BED row with `bigBedToBed`. Where that binary is unavailable,
   iterate `pyBigWig.chroms()` and `entries()` and reconstruct each row as
   `chrom`, `start`, `end`, plus the returned payload. Do not parse or rewrite
   existing payload fields.
3. Scan the called BAM once, keeping mapped primary records. Build
   `query_name -> (HP, PS)`, mapping absent/non-integer tags to `-1`. If repeated
   primary names disagree, stop and report them.
4. Join on BED column 4 (`name`). Circular rows may use FiberHMM's decorated
   name. Remove a suffix only when it exactly matches the row's known feature
   type and circular columns:
   `|<type>|<circId>|<circPart>/<circParts>`. Do not split arbitrary names at
   the first `|`.
5. Append `hp` and `ps` to each row without serializing any pre-existing field
   again. Report total rows, matched rows, missing names, phased rows, and
   independent HP-only/PS-only counts.
6. Insert the two signed autoSQL declarations immediately before the closing
   parenthesis, leaving all prior schema text unchanged. Set
   `-type=bed12+K`, where `K = observed_column_count - 12` after appending.
7. Rebuild with `bedToBigBed`, using chromosome sizes from the source bigBed.
8. Reopen the result and prove: row count is unchanged; each new row excluding
   its last two fields equals the recovered source row exactly; coordinates and
   names are identical; the two schema fields are last; and all output rows
   have one consistent column count.

## Benchmark plan

Run the full, unsampled migration on each nucleosome/MSP/TF bigBed from one
completed dataset. Record separately:

- BAM index build time and peak RSS;
- source bigBed decode time;
- qname join/append time and unmatched/conflict counts;
- `bedToBigBed` rebuild time;
- validation time;
- input/output BED-equivalent row counts and byte sizes.

The BAM mapping should be built once and reused across all three tracks. For a
fair comparison with re-extraction, benchmark both paths from warm local `/mnt/k`
storage and keep sort/build temporary files on that same volume. The migration
is acceptable only with zero conflicts, zero unexplained unmatched names, exact
equality of every pre-existing field, and successful full-row bigBed round-trip
validation. A small full-output smoke run should precede the first whole-genome
run; it tests correctness but is not a throughput estimate.

### Full-output smoke result (2026-07-19)

The private harness was run on a clean v2.16.3 called-BAM smoke artifact: 532
mapped primary names, of which 318 carried at least one phasing tag. The BAM map
took 1.02 s. Results were:

| Track | Rows | Matched | Unmatched | Rebuild | Source -> output bytes |
|---|---:|---:|---:|---:|---:|
| nucleosome | 522 | 522 | 0 | 0.89 s | 272,145 -> 272,647 |
| MSP | 522 | 522 | 0 | 0.22 s | 193,316 -> 194,081 |
| TF | 517 | 517 | 0 | 0.34 s | 164,900 -> 165,416 |

Peak RSS was 45,044 KiB. All 1,561 rows were validated, not sampled: each
rehydrated row minus its final `hp`/`ps` values exactly equaled the recovered
source row, and the rebuilt bigBeds round-tripped through pyBigWig with the new
fields last in autoSQL. All three migrated bigBeds were also byte-for-byte
identical (`cmp` and SHA-256) to a fresh full `fiberhmm-extract
--haplotype-fields` run on the same BAM. These timings establish feasibility
only; the whole-genome benchmark above is still required for throughput and
peak-space planning.
