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
- The output directory must exist.

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
- `fiberhmm-apply` writes no `FIBERHMM-CHEMISTRY` declaration (only the
  `@CO fiberhmm:coord=molecular` frame marker), so recall on its output needs
  `--enzyme`.
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
