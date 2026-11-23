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

`--min-read-length` measures the read as the HMM models it: the full SEQ,
soft clips included, because soft-clipped bases carry m6A calls like any
other. A long molecule with a short alignment (large soft clips) is still
called. For DAF with `--daf-mask-unaligned` (the default), soft-clipped bases
carry no evidence, so the aligned length counts instead. When many reads are
skipped as too short although most of each is soft-clipped, the run summary
says so.

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

`-c/--cores` sets the worker processes (default 4); `-c 0` uses every CPU. `--io-threads` (8) sets the htslib compression threads
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
  (for example staged to scratch) is a different input. A digest is
  recomputed unless the file's device, inode, size, modification time and
  status-change time (which no tool can set back) are all unchanged, so
  resuming does not rehash unchanged inputs.
- `--resume` reuses every region whose marker and BAM (size and SHA-256)
  validate, reruns missing, partial or altered regions, then merges and
  publishes the output
  atomically, exactly as an uninterrupted run would: the records are
  identical. The header's `@PG` keeps the original command line. Integrated
  dedup and the DAF SNP screen are recomputed (they are deterministic).
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

- **Primary and supplementary alignments** (default, `--alignments
  primary-supplementary`). A supplementary record is another part of the
  read (the far side of a structural variant, an insertion's TE copy) and is
  called on its aligned bases only; its soft clips are the primary record's
  sequence. Secondary records (alternative placements of the same bases) are
  written through uncalled. `--primary` calls primary records only,
  `--no-primary` (`--alignments all`) every record.
- **DAF bases without a reference counterpart** (insertions, soft clips) and
  SNP-masked sites are no evidence: a deamination is a read-versus-reference
  mismatch, so these bases can show none. Unaligned stretches of 50 bp or more
  get no calls (no nucleosome, MSP or TF; calls running into them are trimmed
  at their edge). `--no-daf-mask-unaligned` turns this off. Fiber-seq m6A is
  read-intrinsic: insertions and clips are called normally.
- **DAF insertions carried by many reads** (at least
  `--daf-insert-min-carriers`, default 20, within ±30 bp) are re-encoded
  against a deamination-aware consensus of the carriers' inserts, so they get
  real calls (`--daf-insert-consensus off` keeps them masked). Each
  cluster's position, length, carriers per strand, consensus sequence and the
  fraction of confidently called columns are written to
  `qc/<output>.insert_consensus.json`.
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
| `--use-m5c` / `--no-use-m5c` | on for DddA | CpG-aware calling: CpGs masked in the first-pass HMM and recall ([DAF-seq](daf-seq.md#ddda-cpg-island-methylation)) |
| `--cpg-mask-policy` | `unmethylated-only` | `methylated-only` masks only `ddda_mcg` spans (the pre-3.0 behaviour) |

### When to turn the DddA CpG mask off

DddA cannot deaminate a methylated C. In mammals most CpGs are methylated, so
an undeaminated CpG reads as protection whether or not a protein sits there.
The DddA CpG mask therefore makes every CpG observation neutral (no evidence,
no opportunity) except inside the confident unmethylated islands
(`ddda_ucg`) that `fiberhmm-tag-m5c` writes. It is on by default for
`--enzyme ddda`.

In organisms without meaningful CpG methylation, such as *Drosophila*,
undeaminated CpGs carry the same footprint information as any other C, and
masking them throws that evidence away. Turn the mask off there:

```bash
fiberhmm-call -i aligned.bam -o calls.bam --enzyme ddda --no-use-m5c
```

`--no-use-m5c` turns the mask off in the first-pass HMM and in nucleosome and
TF recall. It is the same switch on every DddA tool (`fiberhmm-call` and
`--hmm-only`, `fiberhmm-recall-tfs`/`-recall-nucs`, `fiberhmm-pair`/`-merge`,
`fiberhmm-posteriors`, `fiberhmm-qc`, `fiberhmm-consensus` and
`fiberhmm-transfer`). The call records it in its `@PG` (`cpg_mask=off
hmm_cpg_mask=off`), and later tools that replay the calls inherit it:
recall-tfs/-nucs, pair/merge, posteriors, the qc light call, the nucleosome
caller and consensus/transfer all keep the mask off for that BAM unless you
pass `--use-m5c` (or `--cpg-mask-policy`), which always wins over the record.
The record they read is the newest `fiberhmm-call`, `-recall-tfs`,
`-recall-nucs` or `-merge` `@PG` line (the nucleosome caller reads the last
`fiberhmm-call` line, whose settings it replays); its resolved `cpg_mask=`
value counts, and for older records without it the last `--use-m5c` /
`--no-use-m5c` on its command line. A run whose mask is off by inheritance or
by `--no-use-m5c` also records `cpg_mask_source=inherited|explicit`, so an
inherited `off` survives custom-model recalls. A BAM merged from inputs called
with and without the mask keeps both `@PG` histories: DddA replays
(recall-tfs/-nucs, pair/merge, posteriors, consensus/transfer) refuse it until
you pass `--use-m5c`, `--no-use-m5c` or `--cpg-mask-policy`, and the qc light
call is skipped. A run given the setting records itself on both histories, so
later replays inherit it. The nucleosome caller still replays the last
`fiberhmm-call` line.

## Nucleosome recall options

| Option | Default | Effect |
|---|---|---|
| `--recall-nucs` / `--no-recall-nucs` | on | nucleosome recall; off gives raw HMM nucleosomes (`nuc.Q`) |
| `--split-min-llr` | 4.0 | accessible evidence needed to split a footprint |
| `--split-min-opps` | 3 | informative positions needed in a split |
| `--nuc-min-size` | 85 | minimum nucleosome size; also what bounds an MSP |
| `--msp-min-size` | 0 | minimum MSP size |
| `--ddda-derived-tf-max-edge-gap` | 12 | DddA: evidence needed on both sides of TFs exposed by radial recall (`-1` off) |
| `--nuc-recall-policy` | `auto` (= `conservative`) | `topology` keeps fragment structure in nucleosome recall (below) |

### Nucleosome recall policy

`--nuc-recall-policy` chooses how nucleosome recall treats the HMM's
footprints. `conservative` (the default; `auto` resolves to it) cuts every
qualifying accessible run inside a footprint and trims each nucleosome to
conservative edges from its protected evidence. `topology` keeps fragment
structure: it accepts a cut only when every resulting piece can still hold a
nucleosome (`--nuc-min-size`), and each piece keeps its HMM extent instead of
turning unresolved edges into accessible sequence. It works well on some
ONT Hia5 datasets, where it was tuned, and also runs for Hia5 PacBio and
DddB. Compare it against the default on your data before using it.

```bash
fiberhmm-call -i ont.bam -o ont.topology.bam --enzyme hia5 --seq nanopore \
    --nuc-recall-policy topology
```

DddA refuses `topology` (exit 2), whether `--enzyme ddda` is given or
inherited from the input header: DddA nucleosome recall uses the radial
model, which has no topology policy. The resolved policy is written to the
`@PG` `DS` (`nuc_recall_policy=`), and `fiberhmm-recall-nucs`,
`fiberhmm.nuc` and the consensus Hia5 replay take it from there.
`fiberhmm-pipeline` has no option of its own; pass
`--call-args "--nuc-recall-policy topology"`.

## DAF options

For `--enzyme dddb` and `--enzyme ddda` on a file input, three stages run
automatically; each has its own section in [DAF-seq](daf-seq.md):

1. [PCR-duplicate marking](daf-seq.md#pcr-duplicates) (`--dedup`/`--no-dedup`,
   `--dedup-collapse` and the `--dedup-*` tuning options);
2. [recurrent-SNP screening](daf-seq.md#snp-screening) after duplicate
   marking, when coverage allows (`--daf-call-snps`/`--no-daf-call-snps`,
   `--daf-snp-mask BED`);
3. calling, with the [strand-swap chimera filter](daf-seq.md#strand-swap-chimeras)
   (`--chimeras flag` marks `0x200` + `xc:Z:strand_swap` and retains uncalled;
   `remove` drops all records of the read; `keep` disables detection;
   deprecated `--keep-chimeras` means `keep`; `--chimera-min-seg` 5 and
   `--chimera-purity` 0.8 tune it; the opt-in
   [strict detector](daf-seq.md#strict-detector-opt-in),
   `--chimera-detector strict`, adds base-quality, span, several-switch and
   error-mask filters and a QC grade; `--chimeras split` works with either
   detector) and
   [adjacent-target thinning](daf-seq.md#adjacent-target-thinning)
   (`--daf-mask-runs`, default 2 with `--daf-run-policy keep-one` for DddA,
   off for DddB).

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
| Calls | `MA`/`AQ` (`nuc`, `msp`, `tf`); `ns`/`nl`, `as`/`al`, `nq` too with `--legacy-tags`; see [Annotations](../concepts/annotations.md) |
| Provenance | `@PG` with the resolved settings in `DS` (for example `mode=`, `enzyme=`, `prob_threshold=`, `primary_only=`, `recall_nucs=`, `dedup=`, `daf_snp_mask=`, `daf_run_mask=`, `cpg_mask=`, `hmm_cpg_mask=`), and `@CO fiberhmm:coord=molecular` (calls are in the molecular frame; see [Coordinates](../concepts/coordinates.md)) |
| Chemistry | [`@CO FIBERHMM-CHEMISTRY:v1:`](../reference/headers.md#fiberhmm-chemistry) |
| Layer list | [`@CO MA-TYPES:v1:nuc,msp,tf`](../reference/headers.md#ma-types) |
| QC and DAF side outputs | `qc/` next to the output BAM |

Tag options:

- `--legacy-tags`: also write the legacy `ns`/`nl`/`nq` and `as`/`al`, for
  FiberBrowser ≤ 3.1.0 (blank read rows without them), fibertools ≤ 0.12
  (`ft fire` 0.12 and older needs `as`/`al`) and old scripts. Since 3.3.0
  the default is `MA`/`AQ` only, which FiberHMM, fibertools 0.13+ and
  FiberBrowser 3.1.1+ read; `--no-legacy-tags` is a
  deprecated no-op.
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

## `--hmm-only`: HMM calls only

`fiberhmm-call --hmm-only` runs only the HMM (step 2): no nucleosome recall,
no TF recall. It writes every HMM call to `MA`/`AQ`, classed by size alone:

| HMM call | MA group | AQ bytes |
|---|---|---|
| footprint of `--nuc-min-size` bp or more (default 85) | `nuc.Q` | `0`; with `--scores` the HMM posterior mean of the footprint (×255) |
| footprint shorter than `--nuc-min-size`, wherever it lies (inside MSPs too) | `tf+QQQ` | `0,0,0`: no LLR and no edge evidence (recall writes those) |
| accessible run between nucleosome-sized footprints (the HMM MSP) | `msp.` | none |

The threshold is the one the HMM's own MSP step uses (only footprints of
`--nuc-min-size` bp or more bound an MSP), so `nuc` and `msp` tile the read
and every `tf` lies inside an `msp`. It is not `--unify-threshold` (90 bp),
which only recall uses. Nothing is dropped: the `nuc` and `tf` groups
together are every HMM footprint, the list `fiberhmm-apply --legacy-tags`
wrote to `ns`/`nl` before 3.3.

```bash
fiberhmm-call -i demo/hia5_pacbio.bam -o out/hmm.bam --enzyme hia5 --seq pacbio --hmm-only -c 2
```

```text
  fiberhmm-call --hmm-only — HMM calls, no recall (streaming)
  ...
  nuc >= 85 bp, tf < 85 bp, msp: HMM
```

- The output is input for `fiberhmm-recall-nucs` (nucleosome + TF recall)
  and `fiberhmm-recall-tfs` (TF recall only); they read the `nuc` + `tf`
  groups of an hmm-only call as the HMM footprints. `--hmm-only` then
  `fiberhmm-recall-nucs` gives the same calls as `fiberhmm-call` with
  matching options, and `--hmm-only` then `fiberhmm-recall-tfs` the same as
  `fiberhmm-call --no-recall-nucs` ([Re-calling](recalling.md)).
- The `@PG` record is `fiberhmm-call`'s, with `hmm_only=on`,
  `recall_nucs=False`, `tf_recall=off`, `nuc_min_size=N` and
  `msps=hmm|kept` in its `DS` (and `--hmm-only` in `CL`). Readers use the
  `DS` token: a file whose latest FiberHMM footprint writer is an hmm-only
  call has HMM footprints, not recalled TFs, in its `tf` group.
- Everything else is `fiberhmm-call`'s: streaming or `--region-parallel`
  (with `--resume`), the DAF pre-passes (dedup, chimera flagging, SNP screen,
  insert consensus), the first-pass CpG mask, QC. No recall table is read,
  so a custom `-m` of any context size works; `--recall-nucs` and
  `--recall-model` are refused with `--hmm-only`.
- `--legacy-tags` adds `ns`/`nl` = the `nuc` group and `as`/`al` = the `msp`
  group (the legacy view `fiberhmm-call` writes).
- `--keep-msps` (with `--hmm-only` only) keeps each record's own MSPs instead
  of writing the HMM's: its `MA`/`Ma` `msp` group, legacy `as`/`al` with any
  `aq` (fibertools ≤ 0.12 FIRE), the fibertools ≥ 0.13 `fire` group and every
  other non-footprint group. The HMM footprints are still classed by size.
  Use it to add HMM footprints to fibertools MSP/FIRE calls.
- `--train-reads TSV` (a `rid` column) leaves the reads a custom model was
  trained on uncalled.
- HMM posteriors are exported by `fiberhmm-posteriors`
  ([Posteriors](posteriors.md)).

### `fiberhmm-apply` (deprecated)

`fiberhmm-apply` is a deprecated alias for one release and is removed in the
release after 3.3. It prints a deprecation notice, maps its options onto
`fiberhmm-call --hmm-only` and runs that; the `@PG` records the translated
`fiberhmm-call` command.

| `fiberhmm-apply` | runs as |
|---|---|
| `-o DIR` | `-o DIR/<input stem>_footprints.bam` (`-o -` is stdout) |
| `--no-msps` | `--keep-msps` |
| `-t/--train-reads` | `--train-reads` |
| `--scores` | `--scores` (nucleosome `nq`; no MSP score) |
| `--stats` (`--stats-sample`, `--stats-seed`) | `fiberhmm-call` QC (`--qc-sample-reads`, `--qc-seed`); without `--stats`, `--no-qc` |
| `--legacy-tags` / `--harmonize-legacy-tags` | `--legacy-tags` (`ns`/`nl` = `MA` `nuc`: the harmonized view); `--harmonize-legacy-tags` warns |
| `--output-posteriors` | ignored with a warning; use `fiberhmm-posteriors` |
| `--debug-timing`, `--streaming` | ignored with a warning |
| `--chroms`, `--skip-scaffolds`, `--region-size`, `--scores-db`, `-l` | rejected, as before; use `fiberhmm-call --region-parallel` |

Every other option is passed through unchanged; apply's defaults (`-c 1`,
`--io-threads 4`) are kept. As `fiberhmm-call` runs, DAF input also gets
call's pre-passes (dedup marking, chimera flagging, SNP screen), which
`fiberhmm-apply` never ran.

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
