# Checking outputs for re-runs

`fiberhmm-check` tells you whether an existing FiberHMM output was made
before a fix or default change that alters it, from the provenance the output
records. It reads BAMs (their header and a bounded sample of records, plus the
QC report `fiberhmm-call` writes beside them), `fiberhmm-qc` reports,
`fiberhmm-posteriors` files and `fiberhmm-consensus` result directories.

```bash
fiberhmm-check calls.bam other.bam qc/sample.qc.json consensus_out/
```

```text
ont_hia5.bam: RE-RUN REQUIRED
  - hia5-nanopore-gt-table (rerun-required; affected, medium confidence)
    Nanopore Hia5 calls made with the context-swapped emission table [calls]
    evidence:
      @PG fiberhmm-call.3 command line runs code from a tree named after commit 1ca7d0a, which
      predates the fix (dc7abba)
    why:
      The Nanopore Hia5 emission table shipped in every 2.x release was indexed in ACGT context
      ...
    fix:
      Re-call from the aligned reads with FiberHMM >= 3.0: fiberhmm-call -i aligned.bam -o calls.bam
      --enzyme hia5 --seq nanopore (or fiberhmm-pipeline). ...
calls_v3.bam: clean
```

| Exit status | Meaning |
|---|---|
| 0 | nothing to re-run: every path is clean or has only `info` advisories |
| 3 | at least one path has a `rerun-required` or `rerun-recommended` advisory (affected or possibly affected) |
| 2 | a path could not be read or is not a FiberHMM output |

`--json` prints one document for scripts and GUIs (see
[JSON](#json-output)); `--list` lists every known advisory;
`--scan-records N` sets how many BAM records are scanned for read-level
evidence (default 5000; `0` reads only the header); `--no-sidecars` skips the
QC report beside a BAM. The full option table is in the
[command-line reference](cli.md#fiberhmm-check).

## Severity and status

Each advisory has a **severity**, fixed per advisory:

| Severity | Meaning |
|---|---|
| `rerun-required` | made by code or tables with a known error (for example the 2.x Nanopore Hia5 table); re-run it |
| `rerun-recommended` | made under rules 3.0 corrects (for example DAF dedup by orientation); re-run where those rules matter |
| `info` | a default changed; the output is not wrong, but a 3.0 run with defaults would differ |

and a **status** with a **confidence** for this output:

- `affected`: the evidence shows the output predates the fix;
- `possibly_affected`: the evidence cannot tell (for example a bare version
  string that builds on both sides of the fix reported).

Outputs the evidence clears are not listed.

## What counts as evidence

Strongest first:

1. **Table digests.** Since 3.0 every run's `FIBERHMM-CHEMISTRY`
   declaration records the sha256 of the emission tables it read
   (`apply_sha256`, `recall_sha256`; see
   [Header declarations](headers.md#fiberhmm-chemistry)). FiberHMM ships the
   digest of every historical Nanopore Hia5 and DddB table, so table
   advisories are decided exactly, whatever the version string.
2. **Commit.** The declaration's `fiberhmm_commit` (a git checkout, or a
   wheel built from one), or a commit named in the `@PG CL` path of the
   program that ran (a source tree such as `frozen_dc7abba/`), checked
   against the commits that contain each fix.
3. **Version.** The `@PG VN` of the program. Development builds between
   2.16.8 and 3.0.0 reported `2.16.8` while some already contained 3.0
   fixes, so `2.16.8` alone gives `possibly_affected` (low confidence);
   earlier versions give `affected`.
4. **Options.** `@PG CL`/`DS` decide whether an advisory applies at all:
   `--enzyme`/`--seq`/`mode=` pick the chemistry, `-m` without `--enzyme`
   marks a custom-table run, an explicit `--prob-threshold` or
   `--primary` silences the matching default-change note.
5. **Records** (BAMs, when scanned): `di`/`ds` duplicate-cluster tags,
   pairing tags and calls without any FiberHMM `@PG`.

Only the calls a file holds count: on each `@PG` history (the `PP` chains),
the last `fiberhmm-call`/`-apply` run and the recalls after it. A BAM
re-called with 3.0 is clean even if its header still lists a 2.x call. A
`samtools merge` keeps every input's chain (clashing `@PG` IDs get a
`-XXXXXXXX` suffix) and joins them with its own records; the calls of every
merged branch are in the file, so if any branch is affected the file is
affected. A FiberHMM run after the merge re-calls every read and supersedes
them all. When the header cannot settle it -- `@PG` records without `PP`
links (a history written without links, or a merge), or chemistry
declarations whose `pg=` names an ID that the merge renamed -- every
plausible reading is checked: an advisory that holds in all of them is
reported as found, one that holds in only some as `possibly_affected`
(low confidence); never clean. PP links that form a cycle or name an ID
several records carry never count as proof that a call was replaced, and
when the pairings of declarations and runs are too many to check one by one,
each run is checked with every declaration it could carry (at most possibly
affected). `samtools cat` of several files (and Picard `GatherBamFiles`)
keeps only one input's header, so the other inputs' calls have no recorded
history: unless a later FiberHMM call re-called every read, such a file is
at least possibly affected, never clean. (fiberhmm-call's own `samtools cat`
of its region files is recognised and does not count.)

## Advisories

`fiberhmm-check --list` prints the current list. For 3.0.0:

| ID | Severity | Output | Applies to |
|---|---|---|---|
| `hia5-nanopore-gt-table` | rerun-required | calls | Nanopore Hia5 calls with the 2.x context-swapped table |
| `dddb-gt-table` | rerun-required | calls | DddB calls with the pre-3.0 context-swapped table |
| `hia5-nanopore-called-as-pacbio` | rerun-required | calls | Hia5 run without `--seq` before 3.0 on reads the header shows are Nanopore (`@RG PL`, `map-ont`, basecaller `@PG`) |
| `custom-table-without-enzyme-defaults` | rerun-recommended | calls | `-m` without `--enzyme` on DddA/DddB input, before 3.0 |
| `daf-dedup-orientation` | rerun-recommended | dedup flags | DAF duplicates grouped by alignment orientation, before 3.0 |
| `pair-paired-duplicates` | rerun-recommended | calls | `fiberhmm-pair` before 3.0 on duplicate-flagged input |
| `tag-m5c-missing-ucg` | rerun-recommended | calls | `fiberhmm-tag-m5c` before 3.0 (no `ddda_ucg`) |
| `posteriors-reverse-frame` | rerun-required | posteriors | `fiberhmm-posteriors` files before 3.0 |
| `qc-nanopore-opportunities` | rerun-recommended | QC | Nanopore `fiberhmm-qc` reports before 3.0 |
| `recaller-tier-double-count` | rerun-recommended | consensus tiers | lattice-recaller results with edge/loose tiers before the fix |
| `hia5-nanopore-ml-threshold` | info | calls | ONT Hia5 called at the 2.x default ML threshold |
| `primary-only-default` | info | calls | secondary/supplementary alignments called (2.x default) |
| `ddda-cpg-mask-default` | info | calls | DddA recall with CpG observations (2.x default) |
| `untracked-calls` | info | calls | calls without any FiberHMM `@PG` (cannot be checked) |

QC reports, posteriors files and `fiberhmm-dedup`/`fiberhmm-merge` BAMs record
their FiberHMM version since 3.0; older ones without it are reported as
`possibly_affected` where the advisory applies to them.

## JSON output

`fiberhmm-check --json` prints `{"schema": "fiberhmm.advisory_check.v1",
"reports": [...]}` with one report per path, the same object
`fiberhmm.advisories.report(path)` returns:

```json
{
  "schema": "fiberhmm.advisory_report.v1",
  "path": "ont_hia5.bam",
  "kind": "bam",
  "status": "rerun-required",
  "needs_rerun": true,
  "confirmed": true,
  "error": null,
  "checked_with": {"fiberhmm_version": "3.0.0", "advisories_revision": 1},
  "advisories": [
    {
      "id": "hia5-nanopore-gt-table",
      "severity": "rerun-required",
      "status": "affected",
      "confidence": "medium",
      "needs_rerun": true,
      "title": "Nanopore Hia5 calls made with the context-swapped emission table",
      "reason": "...",
      "artifact": "calls",
      "fix": "Re-call from the aligned reads with FiberHMM >= 3.0: ...",
      "fixed_in": "3.0.0",
      "evidence": ["@PG fiberhmm-call.3 command line runs code from a tree named after commit 1ca7d0a, which predates the fix (dc7abba)"],
      "program": "fiberhmm-call.3",
      "path": "ont_hia5.bam"
    }
  ]
}
```

- `kind`: `bam`, `qc`, `posteriors`, `consensus`, or `null` on error.
- `status`: `clean`, `info`, `rerun-recommended`, `rerun-required` (the
  worst severity present), or `error` (then `error` holds the message and
  `advisories` is empty).
- `needs_rerun`: any advisory other than `info`; `confirmed`: any such
  advisory has status `affected` (a GUI can show "needs re-run" vs
  "may need re-run").
- `advisories[].artifact`: `calls`, `dedup flags`, `QC`, `posteriors` or
  `consensus tiers`; `program` is the `@PG` ID matched (null for file-level
  checks); `path` is the file concerned (a QC report beside a BAM has its own
  path).

## Python

```python
from fiberhmm.advisories import check_path, check_bam, check_header, report

report("calls.bam")                        # dict above; never raises for bad paths
check_bam("calls.bam", scan_records=0)     # header only -> list[Advisory]
check_header(bam.header)                   # pysam header, header dict or SAM text
check_path("consensus_out/")               # BAM, QC JSON, posteriors, consensus dir
```

`Advisory` is a frozen dataclass with the fields of the JSON object above
(`evidence` is a tuple) and `.to_dict()`. `check_*` raise
`fiberhmm.advisories.AdvisoryInputError` for unreadable or unknown paths;
`report` returns `status: "error"` instead. The advisory list ships in the
package as `fiberhmm/advisories.json` (schema `fiberhmm.advisories.v1`), so a
newer FiberHMM knows about more fixes: check with the newest version you have.
