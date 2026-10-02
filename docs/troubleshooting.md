# Troubleshooting

Messages are quoted as the commands print them.

## Installation

**`requires a different Python` / `SyntaxError` / `int has no attribute bit_count`.**
FiberHMM 3.x needs Python 3.10 or later (3.9 lacks `int.bit_count` and the
Numba parallel kernels fail). Create an environment with a newer Python; see
[Installation](getting-started/installation.md).

**`ModuleNotFoundError: numba` / `sklearn` from `fiberhmm-consensus`.** You
have a pre-3.0 install. In 3.0 these are core dependencies:
`pip install --upgrade fiberhmm`.

**`pysam` fails to build, or Numba import errors.** See
[Installation: troubleshooting](getting-started/installation.md#troubleshooting).

**Stale compiled files after copying or syncing a source tree.** Point Python
and Numba at fresh cache directories
(`PYTHONPYCACHEPREFIX=/tmp/fh-pyc NUMBA_CACHE_DIR=/tmp/fh-numba`); see
[Environment variables](reference/environment.md).

## Chemistry

### The input BAM declares a different chemistry

```text
error: fiberhmm-recall-tfs: the input BAM declares chemistry [assay=daf enzyme=dddb platform=nanopore mode=daf] but this run would declare [assay=fiber-seq enzyme=hia5 platform=nanopore mode=nanopore-fiber]. Pass --enzyme/--seq matching the input (with --model for a custom table), or --replace-chemistry to re-declare the output deliberately.
```

The BAM was called with another chemistry than the one this run requests.
Usually `--enzyme`/`--seq` is wrong: fix it, or leave them out and give only
`-m` for a custom table (it then inherits the declared chemistry). If you
really are re-calling with another chemistry, add `--replace-chemistry`. See
[Choosing the chemistry](getting-started/choosing-chemistry.md#re-calling-a-bam-fiberhmm-already-called).

With stdin input, a custom model cannot inherit the chemistry in time:

```text
... the input declares enzyme=dddb, but its header was not available before this custom-model run chose its enzyme-dependent defaults (stdin input). Pass --enzyme dddb (with --model for the custom table), or --replace-chemistry to declare the run as custom.
```

### The platform cannot be inferred

```text
error: fiberhmm-call: cannot infer the sequencing platform for --enzyme hia5: MM specs are mixed: ... Pass --seq pacbio or --seq nanopore.
```

The reads (or reads and header) disagree about PacBio versus Nanopore. Pass
`--seq`. For stdin, detection is impossible and PacBio is assumed with a
warning; pass `--seq nanopore` for Nanopore data.

### Nanopore Hia5 calls from 2.x

Re-run them with 3.0: the 2.x Nanopore Hia5 table was context-swapped. See
[Upgrading from 2.x](upgrading.md).

## Calling

### More than 90% of records were skipped as unmapped

```text
WARNING: 300 of 300 records (100.0%) were skipped as unmapped, so this run produced essentially no calls. For unaligned (uBAM) or streamed input pass --process-unmapped (enabled automatically for stdin, unindexed and unaligned input); pass --no-process-unmapped to keep unmapped reads as untouched pass-through deliberately.
```

For an indexed, aligned BAM, unmapped reads are passed through by default;
if nearly all records are unmapped the run fails (exit 1) rather than
producing an output with no calls. Pass `--process-unmapped` to call them,
or `--no-process-unmapped` to accept the pass-through (the run then exits 0
with this warning).

### Worker read failures

```text
  worker read failures: <N> of <M> processed reads (passed through unannotated)
  --- worker failure traceback 1 ---
...
error: <N> of <M> reads failed inside workers (limit: 1% of processed reads); the output was not written. See the traceback(s) above.
```

Reads that raise an error inside a worker are written through uncalled. A
few are tolerated (with their tracebacks printed); more than 1% of processed
reads (any, when fewer than 100 reads were processed) fails the run and no
output is published. The tracebacks point at the offending reads; please
report them.

### DAF-seq calling needs deamination calls

```text
error: DAF-seq calling needs deamination calls, and none of the supported
  sources were found in the first 10 mapped reads of out/nomd.bam:
    - R/Y IUPAC codes in the stored query sequence
      (produced by fiberhmm-daf-encode), or
    - MD tags on aligned reads
      (set by 'minimap2 --MD' or 'samtools calmd'), or
    - a reference FASTA via --reference ref.fa
```

Align with `minimap2 --MD`, run `samtools calmd`, or pass `--reference`.

### `--region-parallel needs a coordinate-sorted, indexed BAM`

Index the BAM (`samtools index`), or drop `--region-parallel` to stream.
`--region-parallel` also cannot read stdin or write stdout.

### Hard-clipped records are skipped (`hard_clipped_mm`)

Supplementary alignments hard-clipped by minimap2 keep the full read's
`MM`/`ML`, which cannot be matched to the clipped `SEQ`; FiberHMM skips them.
Primary alignments are unaffected. Align with `minimap2 -Y` to soft-clip
supplementary alignments if you need them (`--no-primary`).

### `fiberhmm-apply` rejects `--chroms`, `--skip-scaffolds`, `--region-size`, `--scores-db`, `-l`

These never had an effect in `fiberhmm-apply`. Use
`fiberhmm-call --region-parallel` for region selection.

### A failed run left no output

Intended: outputs are published only when the run succeeds, and an earlier
output at the same path is kept. A `--region-parallel` run keeps its finished
regions; see below.

### An interrupted `--region-parallel` run

Rerun the same command with `--resume`: finished regions in
`.<output name>.fiberhmm-work/` (or `--work-dir`) are reused. See
[Long runs and resuming](workflows/calling.md#long-runs-and-resuming).

### `work directory … from an earlier interrupted run exists`

A previous `--region-parallel` run to the same output stopped before
publishing. Add `--resume` to continue it, or delete the named directory to
start over. FiberHMM never discards finished regions silently.

### `--resume refused: the input or parameters differ from the interrupted run`

The input BAM (its resolved path, size, content or header digest, or its
index) or an effective parameter differs from the run that made the work
directory; the message names the fields. Rerun with the original input and
options, or delete the work directory to start over. Touching a file does not
count as a change; changing its content, or moving or copying it to another
path, does.

### `--resume needs the region-parallel pipeline`

Only region-parallel runs keep per-region results. Streaming runs (stdin,
stdout, unsorted or unaligned input) restart from the beginning; sort and
index the input to make a long run resumable.

## DAF-seq

**`fiberhmm-tag-m5c`: `input BAM has no Y/R-encoded DAF sequence`.** The
CpG-island caller needs R/Y-encoded reads: run `fiberhmm-daf-encode` before
`fiberhmm-call` ([DAF-seq](workflows/daf-seq.md#ddda-cpg-island-methylation)).

**`fiberhmm-pair --from-paired`: `these pairing options have no effect there`.**
`--from-paired` skips pairing; drop `--reference`, `--sequence-only`,
`--pairs-tsv`, `--model` and the `--min-*` pairing thresholds.

**Recall warns that the SNP mask cannot be re-applied.** Recall re-derives
deaminations from the reads; re-run `fiberhmm-call` with the same options if
you need SNP-masked calls ([Re-calling](workflows/recalling.md#what-is-kept-and-what-is-regenerated)).

## Consensus and transfer

### A knob the engine does not use

```text
fiberhmm-consensus: error: cr.seed is not used by the lattice recaller; reset it to its default (the recaller's own controls are in the recaller group)
fiberhmm-consensus: error: recaller.stringency is not used by staged families; reset it to its default (use the families controls)
fiberhmm-consensus: error: --stop-after applies to --engine staged_native_families only; the lattice recaller discovers and scores classes in one pass (every run saves evidence.json.gz for replay with --evidence or --resume)
```

Each engine rejects settings it would ignore, so a run never records a
setting that had no effect. Remove the setting, or choose the engine it
belongs to. The lattice recaller's knobs are listed in
[Footprint classes](workflows/consensus.md#parameters).

```text
fiberhmm-consensus: error: recaller.abutting was removed in FiberHMM 3.0: ...
```

Use the "+ edge" prevalence tier, or `recaller.linker=either`.

### Missing or unsupported chemistry

```text
fiberhmm-consensus: error: Every BAM needs chemistry metadata or an explicit dataset chemistry: dataset_1; the header declares none, or enzyme=custom (a custom -m without --enzyme). Pass --chemistry (ddda, dddb, hia5-pacbio, hia5-nanopore) or a dataset chemistry
```

The BAM has no chemistry declaration (for example it was called with a
custom `-m` and no `--enzyme`, so it declares `enzyme=custom`). Pass
`--chemistry ddda|dddb|hia5-pacbio|hia5-nanopore` (or a dataset's
`chemistry`) to say which emission model to use.

```text
fiberhmm-consensus: error: BAM header declares ecogii chemistry (ecogii-pacbio); consensus has no ecogii profile (supported: ddda, dddb, hia5-pacbio, hia5-nanopore). Another enzyme's emissions are never substituted: run consensus only on supported chemistries
```

Consensus supports only the four chemistries above; EcoGII and other enzymes
are rejected rather than scored with Hia5 emissions. Both errors are raised
before any results are written (exit 2); `fiberhmm-transfer` reports them the
same way.

### `Output directory must be empty; existing results are never overwritten`

Choose a new `--output` directory. If that directory holds an interrupted
multi-window BAM run, add `--continue` to finish it in place.

### `--continue refused: inputs or parameters differ from the original run`

`--continue` only finishes the run recorded in `consensus_run.json`; the
message names the fields that differ (BAMs or their indexes, windows,
parameters, BAM-export options, DAF run mask). Use the original command
(only `--cores`, `--window-jobs` and `--json-progress` may change), or write
the changed analysis to a new `--output`.

### `--continue` versus `--resume`

`--continue` finishes an interrupted multi-window run in its own `--output`;
`--resume RUN_DIR` starts a new run from one finished window's saved
evidence. See
[Long runs and resuming](workflows/consensus.md#long-runs-and-resuming).

### A window failed in a multi-window run

The error names the window and its `logs/window_NNNNNN.log`. Completed
windows are kept: fix the cause and rerun the same command with `--continue`.

### No classes, or classes marked `unscored`

`unscored` means no molecule spans the class's scoring window or a channel
has fewer than `recaller.minimum_channel_units` (20) molecules, common at
data or amplicon ends; widen the window or add data. When discovery finds
nothing, check `manifest.json` → `recaller.dropped_by_core_rule` and the
per-tile `discovery` diagnostics, and consider a lower `recaller.stringency`
(0.6–0.7) for finer classes.

### Transfer asks for `--dataset-map`

Several source datasets share the target's chemistry, so the source channel
is ambiguous; name it with `--dataset-map TARGET=SOURCE`.

## Strand rescue

**`--nuc-site/--forced-nuc-sites-only need --independent-nuc-edge-refinement`.**
Nucleosome edge normalization is off by default; add
`--independent-nuc-edge-refinement` to use nucleosome sites.

## QC

**`INSUFFICIENT`.** Too few reads or opportunities to grade: QC samples
aligned primary reads at MAPQ ≥ 20, so unaligned call sets always report
`INSUFFICIENT`.

**Rate capped at WARN with a threshold mismatch.** The run's ML threshold
differs from the QC reference's calibration by more than 5; use the default
threshold for the chemistry.

## Known issues in 3.0.0

- Tools that re-read `MM`/`ML` from a BAM (`recall-tfs`, `extract`, `qc`) use
  ML 125 for non-Nanopore chemistries while `call`/`apply`/`posteriors` use
  128. Pass `--prob-threshold` to align them.
- Legacy `.pickle` models execute code when loaded; load only trusted files.
