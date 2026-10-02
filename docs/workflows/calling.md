# Calling footprints

`fiberhmm-call` runs the whole per-read caller in one process: the HMM,
nucleosome recall and TF recall (see [How FiberHMM works](../concepts/how-it-works.md)).
For DAF input it also marks PCR duplicates and screens for SNPs first, and it
finishes with a bounded QC report.

```bash
fiberhmm-call -i aligned.bam -o calls.bam --enzyme hia5 --seq pacbio -c 8 --region-parallel
```

Every option is listed in the [command-line reference](../reference/cli.md#fiberhmm-call).

## Inputs

| Input | Works | Notes |
|---|---|---|
| Aligned, coordinate-sorted and indexed BAM | yes | the fastest path, with `--region-parallel` |
| Aligned BAM, unsorted or unindexed | yes | streaming mode |
| Unaligned BAM (uBAM) | yes | streaming mode; reads are called unaligned |
| stdin (`-i -`) | yes | streaming mode |

Hia5 input needs `MM`/`ML` tags with m6A calls (`A+a`, and `T-a` on PacBio).
DAF input needs its deaminations as R/Y codes, a usable `MD` tag, or
`--reference` (see [DAF-seq](daf-seq.md#where-deaminations-come-from)).

Reads below `--min-mapq` (0) or `--min-read-length` (1000 bp), reads with no
usable observations, and records that are skipped for another reason are
written to the output unchanged (without call tags), so the output always
holds every input record.

## Two execution strategies

**Region-parallel** (`--region-parallel`) splits the genome into
`--region-size` (10 Mb) regions and gives one to each worker. It needs a
coordinate-sorted, indexed input, scales with `--cores` up to about the number
of regions, and writes a sorted, indexed output. Use it for aligned data.

```bash
fiberhmm-call -i sorted.bam -o calls.bam --enzyme hia5 --seq pacbio \
    -c 16 --region-parallel --skip-scaffolds
```

- `--skip-scaffolds` leaves unplaced contigs, `_random`/`_alt`/`chrUn_`
  sequences and GenBank/RefSeq scaffolds uncalled (they are still copied to
  the output). Main chromosomes of human, mouse, fly, yeast (chrI–chrXVI),
  worm, RefSeq `NC_` accessions and `chrEBV` are kept.
- `--chroms chr2L chr3R` calls only those contigs; the others are copied.
- A missing output directory is created.

**Streaming** (the default without `--region-parallel`) reads the input once,
in order, with `--cores` workers each taking `--chunk-size` (500) reads. It
accepts unsorted, unaligned and stdin input and can write to stdout (`-o -`,
unsorted, not indexed). A file output is sorted if needed, then indexed.

```bash
# Pipe into FIRE (fibertools' ft, installed separately) without an intermediate file
fiberhmm-call -i aligned.bam -o - --enzyme hia5 --seq pacbio -c 8 | ft fire - fire.bam
```

`-c 0` uses every CPU. `--io-threads` (8) sets the htslib compression threads
per stage. `--max-reads N` (streaming only) stops after *N* reads, for a quick
look.

## Long runs and resuming

A genome-scale `--region-parallel` run can take many hours. It keeps every
finished region in a work directory, so an interrupted run (Ctrl-C, `kill`,
a closed laptop lid, a crashed node) continues where it stopped:

```bash
fiberhmm-call -i sorted.bam -o calls.bam --enzyme hia5 --seq pacbio \
    -c 16 --region-parallel --skip-scaffolds
# ... interrupted after 212 of 318 regions ...
fiberhmm-call -i sorted.bam -o calls.bam --enzyme hia5 --seq pacbio \
    -c 16 --region-parallel --skip-scaffolds --resume
```

- The work directory is `.<output name>.fiberhmm-work` beside the output
  (`.calls.bam.fiberhmm-work/`), or `--work-dir DIR`. It holds each finished
  region's BAM, a `region_NNNNNN.done.json` marker written after the region
  finished (with the size and SHA-256 of the region's BAM), and
  `manifest.json` with the run identity: the input BAM (path, size, content
  SHA-256 of the BAM and its index, header SHA-256), every effective
  parameter including the resolved chemistry and defaults, the content
  digests of the model, nucleosome profile, SNP mask and reference, the
  region plan and the FiberHMM version.
- Identities compare content, not file dates: a touched input, or a file
  replaced by an identical copy at the same path, is the same input; a
  changed one is refused even if its size and date were preserved. The
  resolved path is part of the identity, so the same data at another path
  (for example staged to scratch) is a different input. A digest is recomputed unless the file's device, inode, size,
  modification time and status-change time (which no tool can set back) are
  all unchanged, so resuming does not rehash unchanged inputs.
- `--resume` reuses every region whose marker and BAM (size and SHA-256)
  validate, reruns missing, partial or altered regions, then merges and
  publishes the output
  atomically, exactly as an uninterrupted run would: the records are
  identical. The header's `@PG` keeps the original command line. Integrated
  dedup, the DAF SNP screen and the NRL estimate are recomputed (they are
  deterministic).
- A change to the input BAM or to any parameter is refused, naming the
  fields that differ; nothing is mixed. `--cores` and `--io-threads` may
  change.
- `--resume` implies `--region-parallel` for an indexed, aligned BAM. With no
  work directory it starts a new run, so a runner can always pass it.
  Streaming runs (stdin, stdout, unsorted or unaligned input) cannot resume.
- Without `--resume`, an existing work directory is never discarded: the
  run stops and asks for `--resume` (or for the directory to be deleted).
- A work directory has one owner: a run holds a lock on it (`.lock`) from
  the moment it inspects it until it has published and cleaned up, and a
  second run on the same work directory is refused while the first is
  alive. The lock disappears with its process, however it ends, so there
  is never a stale lock to remove.
- The work directory is removed after a successful publish
  (`--keep-work-dir` keeps it) and kept when a run fails or is interrupted.
  It needs about as much space as the output BAM.
- `SIGINT`, `SIGTERM` and `SIGHUP` stop the run at once (queued regions are
  cancelled, workers are stopped); a partly written output is never
  published, and the exit status is 128 + the signal number.

### Machine-readable progress

`--progress-json` writes one JSON object per line to stderr, or appends them
to a file with `--progress-json FILE`. Every line has `schema`
(`fiberhmm.progress.v1`), `tool`, `event` and `time` (Unix seconds):

| `event` | Extra fields |
|---|---|
| `start` | `regions_total`, `regions_done`, `regions_reused`, `work_dir`, `output` |
| `region` | `region` (`[chrom, start, end]`), `regions_done`, `regions_total`, `regions_reused`, `reads`, `reads_with_footprints`, `reads_per_s`, `elapsed_s`, `eta_s` |
| `merge` | `regions_total`, `bams` |
| `done` | `output`, `regions_total`, `regions_reused`, `reads`, `reads_with_footprints`, `elapsed_s` |
| `stopped` | `reason`, `regions_done`, `regions_total`, `work_dir` |

```json
{"schema": "fiberhmm.progress.v1", "tool": "fiberhmm-call", "event": "region", "time": 1790733514.5, "region": ["chr2L", 2000000, 4000000], "regions_done": 57, "regions_total": 318, "regions_reused": 40, "reads": 812344, "reads_with_footprints": 790012, "reads_per_s": 1840.2, "elapsed_s": 212.4, "eta_s": 3120.0}
```

`reads_per_s`, `elapsed_s` and `eta_s` cover only regions processed in this
attempt (reused regions are excluded); `eta_s` scales the elapsed time by the
remaining called base pairs and is `null` until the first region finishes.
Region-parallel runs emit every event; streaming runs emit only `start` and
`done`. The text progress line shows the same ETA.

## Which records are called

- **Primary alignments only** (default). Secondary and supplementary records
  are written through uncalled; `--no-primary` calls them too.
- **Hard-clipped records whose MM/ML cannot match SEQ** are always skipped
  (`hard_clipped_mm` in the skip report). minimap2 hard-clips supplementary
  alignments unless run with `-Y`, and their `MM`/`ML` still describe the
  full read.
- **Unmapped reads** are called automatically for stdin, unindexed and
  unaligned (no `@SQ`) input, and passed through for indexed aligned BAMs.
  `--process-unmapped` / `--no-process-unmapped` force either.

The run ends with a count of processed and skipped records by reason. If more
than 90% of records were skipped as unmapped, the run fails (see
[Troubleshooting](../troubleshooting.md#more-than-90-of-records-were-skipped-as-unmapped))
unless you passed `--no-process-unmapped`.

## Model and chemistry

`--enzyme` and `--seq` choose the bundled models and every chemistry default
([Choosing the chemistry](../getting-started/choosing-chemistry.md)).

| Option | Default | Use |
|---|---|---|
| `-m/--model` | bundled | a custom HMM model (JSON) |
| `--recall-model` | the HMM model (DddA: `ddda_TF.json`) | a separate table for TF recall |
| `-k/--context-size` | from the model | validated against the model's table |
| `--replace-chemistry` | off | re-declare the input's chemistry deliberately |
| `--prob-threshold` | 128; 248 for Hia5 Nanopore | minimum ML for an MM/ML call |

A custom `--model` on a BAM that declares its chemistry inherits that enzyme
and platform (and their defaults) when the observation mode matches.

## TF recall options

| Option | Default | Effect |
|---|---|---|
| `--min-llr` | 5.0 (every preset) | per-interval cost λ of the TF decoder |
| `--min-opps` | 3 | minimum informative targets per TF call |
| `--unify-threshold` | 90 | HMM footprints shorter than this are TF candidates |
| `--emission-uplift` | 1.0 | power transform of the emissions (sensitivity experiments only) |
| `--use-m5c` / `--no-use-m5c` | on for DddA | CpG-aware recall ([DAF-seq](daf-seq.md#ddda-cpg-island-methylation)) |
| `--cpg-mask-policy` | `unmethylated-only` | `methylated-only` masks only `ddda_mcg` spans (the pre-3.0 behaviour) |

## Nucleosome recall options

| Option | Default | Effect |
|---|---|---|
| `--recall-nucs` / `--no-recall-nucs` | on | nucleosome recall; off gives raw HMM nucleosomes (`nuc.Q`) |
| `--nuc-recall-policy` | `auto` | `conservative`, `topology`; `auto` = topology for Nanopore Hia5 |
| `--phase-nrl` | `auto` | periodicity prior: `auto`, `off`, or a repeat length in bp |
| `--split-min-llr` | 4.0 | accessible evidence needed to split a footprint |
| `--split-min-opps` | 3 | informative positions needed in a split |
| `--nuc-min-size` | 85 | minimum nucleosome size; also what bounds an MSP |
| `--msp-min-size` | 0 | minimum MSP size |
| `--ddda-derived-tf-max-edge-gap` | 12 | DddA: evidence needed on both sides of TFs exposed by radial recall (`-1` off) |

`--phase-nrl auto` needs a file input; on stdin it falls back to 185 bp.

## DAF options

For `--enzyme dddb` and `--enzyme ddda` on a file input, three stages run
automatically; each has its own section in [DAF-seq](daf-seq.md):

1. [PCR-duplicate marking](daf-seq.md#pcr-duplicates) (`--dedup`/`--no-dedup`,
   `--dedup-collapse` and the `--dedup-*` tuning options);
2. [recurrent-SNP screening](daf-seq.md#snp-screening) after duplicate
   marking, when coverage allows (`--daf-call-snps`/`--no-daf-call-snps`,
   `--daf-snp-mask BED`);
3. calling, with the [strand-swap chimera filter](daf-seq.md#strand-swap-chimeras)
   (`--keep-chimeras`) and [adjacent-target thinning](daf-seq.md#adjacent-target-thinning)
   (`--daf-mask-runs`).

`--reference ref.fa` supplies the reference when reads carry neither R/Y
codes nor a usable `MD` tag. `--ddda-mcg` was retired; it prints the
replacement workflow.

## QC after calling

For a file output, `fiberhmm-call` samples the new BAM and writes a QC report
next to it (`<output dir>/qc/<BAM stem>.qc.*`), selecting the reference data
set from the resolved enzyme and platform. `--no-qc` skips it,
`--qc-output-prefix` moves it, and `--qc-sample-reads`, `--qc-seed`,
`--qc-min-mapq` tune the sample. QC is skipped for stdout output. See
[Quality control](qc.md).

## Output

| Content | Where |
|---|---|
| Calls | `ns`/`nl`, `as`/`al`, `nq`, and `MA`/`AQ` (`nuc`, `msp`, `tf`); see [Annotations](../concepts/annotations.md) |
| Provenance | `@PG` with the resolved settings in `DS` (for example `mode=`, `enzyme=`, `prob_threshold=`, `primary_only=`, `recall_nucs=`, `phase_nrl=`, `dedup=`, `daf_snp_mask=`, `daf_run_mask=`, `cpg_mask=`, `coord=molecular`) |
| Chemistry | [`@CO FIBERHMM-CHEMISTRY:v1:`](../reference/headers.md#fiberhmm-chemistry) |
| Layer list | [`@CO MA-TYPES:v1:nuc,msp,tf`](../reference/headers.md#ma-types) |
| QC and DAF side outputs | `qc/` next to the output BAM |

Tag options:

- `--no-legacy-tags`: write only `MA`/`AQ`, not `ns`/`nl`/`as`/`al`.
- `--downstream-compat`: put TF calls into `ns`/`nl` and write no
  `MA`/`AQ`/`AN`, for tools that read only legacy tags.
- `--scores` (alias `--with-scores`): with `--no-recall-nucs`, write the HMM
  posterior mean of each nucleosome as `nq`.
- `-r/--circular`: circular molecules ([Annotations](../concepts/annotations.md#circular-molecules)).

**Outputs are published atomically.** The BAM is written and indexed under a
temporary name and renamed with its index only when the run succeeds, so a
failed or interrupted run leaves no valid-looking partial BAM, and an earlier
output at the same path is kept.

**Failures inside workers are not silent.** A read that raises an error in a
worker is written through uncalled and counted. If more than 1% of processed
reads fail (any failure when fewer than 100 reads were processed), the run
stops with the first tracebacks and writes no output.

## `fiberhmm-apply`: HMM only

`fiberhmm-apply` runs only the HMM (step 2): nucleosome and MSP calls in
`ns`/`nl`/`as`/`al`, no TF recall, no `MA`. It is useful for inspecting the
raw HMM, for a staged workflow, or with `fiberhmm-recall-tfs`/`-recall-nucs`
afterwards ([Re-calling](recalling.md)).

```bash
fiberhmm-apply -i demo/hia5_pacbio.bam --enzyme hia5 --seq pacbio -o out/apply -c 2
```

```text
Processed 300 reads -> 300 with footprints
BAM: out/apply/hia5_pacbio_footprints.bam
BAM index: out/apply/hia5_pacbio_footprints.bam.bai
```

- `-o` is an output **directory**; the BAM is `<input stem>_footprints.bam`.
  `-o -` writes the BAM to stdout.
- `--streaming` processes unaligned, unindexed or stdin input in a pipeline;
  `-c 0` uses every CPU; `-c 1` (default) runs in one process.
- `--scores` writes posterior-mean `nq`/`aq`; `--no-msps` omits `as`/`al`/`aq`;
  `--stats` samples reads for summary plots (needs `fiberhmm[plots]`);
  `--output-posteriors FILE` also exports HMM posteriors; `-t` excludes the
  read IDs a model was trained on.
- `fiberhmm-apply` records the same provenance as `fiberhmm-call`: an `@PG`
  line with its resolved settings and a `FIBERHMM-CHEMISTRY` declaration, so
  recall, extract and QC read the chemistry from its output.
- `--chroms`, `--skip-scaffolds`, `--region-size`, `--scores-db` and `-l`
  never had an effect in `fiberhmm-apply` and are rejected; use
  `fiberhmm-call --region-parallel` to select regions.

## Performance

- Use `--region-parallel` with `-c` set to the cores you have (`-c 0` = all)
  on sorted, indexed input; add `--skip-scaffolds` on assemblies with many
  small contigs.
- Install `samtools`: FiberHMM falls back to pysam for sorting, indexing and
  concatenation, which is slower.
- Pipe `-o -` into `ft fire` or `samtools` to avoid intermediate files.
- Numba compiles the HMM kernels when each worker starts, which takes a few
  seconds; it is negligible on real data sets.
- For long runs, see [Long runs and resuming](#long-runs-and-resuming): the
  resume bookkeeping costs one extra read of the input BAM when the run
  starts (its SHA-256, at about 2-3 GB/s) and a digest of each region BAM;
  a resume does not rehash an unchanged input.
