# DAF-seq

DAF-seq marks accessible DNA by deamination: a deaminated C reads as T on
C→T ("CT") reads and as A on G→A ("GA") reads. FiberHMM supports DddB
(`--enzyme dddb`) and DddA (`--enzyme ddda`). This page covers what is
specific to DAF data: where the deaminations come from, duplicates, SNPs,
chimeras, adjacent-target masking and DddA CpG-island methylation. Calling
itself is the same `fiberhmm-call` command as for Fiber-seq
([Calling footprints](calling.md)).

```bash
fiberhmm-call -i aligned.bam -o calls.bam --enzyme dddb -c 8 --region-parallel
fiberhmm-call -i aligned.bam -o calls.bam --enzyme ddda -c 8 --region-parallel
```

## Where deaminations come from

FiberHMM has to tell a DAF conversion from a base that differs from the
reference for another reason. It uses, in this order:

1. **R/Y codes in `SEQ`**, written by `fiberhmm-daf-encode` (Y = deaminated C
   on a CT read, R = deaminated G on a GA read), with an `st:Z:CT|GA` tag;
2. **a usable `MD` tag** (from `minimap2 --MD` or `samtools calmd`), combined
   with the CIGAR and the read sequence to find reference-C→T and
   reference-G→A substitutions;
3. **`--reference ref.fa`**, an indexed FASTA of the assembly the reads were
   aligned to, when `MD` is missing or its reference span disagrees with the
   CIGAR;
4. **`MM`/`ML` dU calls** (modification code `u`), read at ML ≥ 128 by default.

`--reference` never overrides R/Y or a usable `MD`, and never realigns or
rewrites reads. A stale `MD` with the right span is not detected; refresh it
with `samtools calmd -b aligned.bam ref.fa > aligned.calmd.bam`.

`fiberhmm-call` checks the first mapped reads of a file input and stops if
none of these sources is present, instead of skipping every read.

Each read's flavour (CT or GA) is decided from its dominant conversion, not
from its alignment orientation.

## R/Y encoding: `fiberhmm-daf-encode`

Optional for calling (`fiberhmm-call` reads `MD` directly), but needed for:

- `fiberhmm-tag-m5c`, which requires R/Y in `SEQ`;
- a BAM that depends on `--reference`, if you want the automatic duplicate
  marking (which does not use the FASTA);
- downstream tools that read R/Y or `st`.

```bash
fiberhmm-daf-encode -i demo/ddda.bam -o out/ddda.encoded.bam
samtools view -b demo/dddb.bam | fiberhmm-daf-encode -i - -o out/dddb.encoded.bam
```

```text
fiberhmm-daf-encode summary
  Total reads:                377
  Encoded:                    377
    CT (+ strand):            247
    GA (- strand):            130
  Skipped:                      0
```

Records that are unmapped, secondary, supplementary, below `-q/--min-mapq`
(20) or `--min-read-length` (1000), or without detectable conversions are
written through unchanged and counted as skipped. `--strand CT|GA` forces the
flavour; `auto` (default) decides per read.

## PCR duplicates

DAF libraries, especially amplicons, can be heavily PCR-duplicated, and
coordinate-based duplicate marking does not work: every read of an amplicon
has the same primer-defined ends and there are no UMIs. FiberHMM uses each
read's **deamination fingerprint** (the set of reference positions it
deaminated). PCR copies share it, up to a few sequencing errors.

Two reads are duplicates when:

- both aligned reference ends agree within `--max-end-diff` (50 bp);
- they have the same deamination flavour (CT or GA; `--ignore-strand`
  clusters across flavours);
- their deamination sets have Jaccard similarity ≥ `--min-jaccard` (0.95),
  found with MinHash and locality-sensitive hashing.

Reads with fewer than `--min-deam` (10) deaminations are not fingerprinted.
Within a cluster, the representative is the read with the highest MAPQ, then
the most complete alignment.

**Integrated (automatic).** `fiberhmm-call --enzyme ddda|dddb` on a file
input marks duplicates before SNP screening and calling. It is
**nondestructive**: every read is kept; non-representatives get flag `0x400`;
every member of a duplicate cluster gets `di` (cluster id) and `ds` (cluster
size). Pooled SNP, periodicity and QC statistics ignore flagged copies. A
`<stem>.dedup.json` report goes to the QC directory. Options:
`--no-dedup` (off), `--dedup` (force), `--dedup-collapse` (remove the
non-representatives instead), and `--dedup-min-jaccard`,
`--dedup-max-end-diff`, `--dedup-min-deam`, `--dedup-prob-threshold`,
`--dedup-ignore-strand`, `--dedup-stats-tsv`. Integrated dedup needs a file
input (not stdin) and does not use `--reference`.

**Standalone.** `fiberhmm-dedup` **collapses** by default, keeping one read
per cluster; `--flag-only` marks instead:

```bash
fiberhmm-dedup -i demo/ddda.bam -o out/ddda.dedup.bam
fiberhmm-dedup -i demo/ddda.bam -o out/ddda.markdup.bam --flag-only --stats-tsv out/ddda.clusters.tsv
```

```text
Clustering (Jaccard >= 0.95; ends ±50 bp): 377 reads -> 325 molecules | 52 duplicates (13.8%) | mean 1.16 copies/molecule [1s]
Pass 2: wrote 325 reads (52 duplicates collapsed) -> out/ddda.dedup.bam [1s]
```

Reads flagged `0x400` are skipped as pairing candidates by `fiberhmm-pair`,
and `fiberhmm-extract` marks their rows `isDuplicate` (hidden by default in
FiberBrowser tracks).

## SNP screening

A genomic C/T or G/A variant looks like a deamination on every read. The SNP
caller separates the two by using the reads of the *other* flavour: a
recurrent C→T mismatch is called a SNP only if it also appears across
G→A-dominant reads (and G→A only across C→T-dominant reads), where it cannot
be a deamination.

The production policy `bidirectional_five_fiber_v1` requires, independently
in each conversion-direction class, a mismatch fraction ≥ 0.2, depth ≥ 5 and
at least 5 mismatch-carrying reads. Duplicate-flagged reads are excluded.
Amplicons need at least 20 reads to be summarized.
Amplicons need at least 20 reads to be summarized.

Reference bases come from each read's `MD` tag; the reference FASTA
(`--reference`) is used only for reads without a usable `MD`. A read whose
`MD` does not match its CIGAR is read against the FASTA, or skipped without
one. If reads' `MD` tags disagree about a site's base (C in some, G in
others), the base reported by more C→T- or G→A-dominant reads is kept (a tie
keeps C), independent of the background site sample and of the thresholds,
and the JSON counts such sites under
`accounting.reference_base_conflict_sites`. The same input gives
byte-identical outputs on every run.

**Integrated.** For DddA/DddB file input, `fiberhmm-call` screens
automatically after duplicate marking when a bounded preflight finds enough
local depth, and masks the called sites from the DAF observations. Masking
is observational: read sequences and `MD` are unchanged. Outputs go to
`qc/<stem>.daf_snps.{bed,vcf,json,amplicons.tsv}`. Options:
`--no-daf-call-snps`, `--daf-call-snps` (force), `--daf-snp-mask BED` (apply
a reviewed mask instead), and `--daf-snp-min-*` thresholds.

**Standalone.**

```bash
fiberhmm-daf-snps -i demo/ddda.bam -o out/ddda.daf_snps
```

```text
FiberHMM DAF SNP mask: 0 sites
  BED mask: out/ddda.daf_snps.bed
  VCF:      out/ddda.daf_snps.vcf
  amplicons: out/ddda.daf_snps.amplicons.tsv
  report:   out/ddda.daf_snps.json
```

Overriding a threshold is recorded as a `custom` policy in the JSON.

## Strand-swap chimeras

A read deaminated C→T in one segment and G→A in another is a chimera of two
molecules. For DAF input, `fiberhmm-call` drops such reads from calling
(they are written through uncalled) and counts them. A segment needs at least
`--chimera-min-seg` (5) same-flavour deaminations at a purity of at least
`--chimera-purity` (0.8). `--keep-chimeras` turns the filter off.

## Adjacent-target thinning

Adjacent targets on the deaminated strand (CC on CT reads, GG on GA reads)
convert together rather than independently, which the per-site model does not
describe. `--daf-mask-runs N` thins runs of *N* or more targets, measured on
the original sequence: `--daf-run-policy keep-one` keeps the 5'-most target,
`drop` removes the run.

| Chemistry | Default |
|---|---|
| DddA | `N = 2`, keep-one |
| DddB | off |

`--daf-mask-runs 0` disables it. The setting applies to the HMM, both
recallers, the consensus lattices and duplex recall, and is recorded in
`@PG` (`daf_run_mask=>=2/keep-one`). Consensus refuses a masked lattice when
native calls are not replayed. For the validation behind the DddA default see
[How FiberHMM works](../concepts/how-it-works.md#1-context-specific-emissions).

## DddA CpG-island methylation

DAF amplification removes native 5mC, but methylated CpGs keep a strong DddA
signature: DddA deaminates methylated CpGs much less. FiberHMM reads this out
per molecule at the scale of complete CpG islands, using the molecule's own
non-CpG deamination rate inside its initial MSP as the accessibility control.
The caller is calibrated for **genome-wide DddA DAF-seq only**; do not use it
on DddB or on targeted/amplicon DddA.

The workflow needs R/Y-encoded reads:

```bash
fiberhmm-daf-encode -i aligned.bam -o encoded.bam
fiberhmm-call -i encoded.bam -o calls.initial.bam --enzyme ddda -c 8 --region-parallel
fiberhmm-tag-m5c -i calls.initial.bam -o calls.m5c.bam -r reference.fa --enzyme ddda \
    --write-cpg-islands islands.used.bed --calls-tsv island_calls.tsv
fiberhmm-call -i calls.m5c.bam -o calls.bam --enzyme ddda -c 8
```

The second `fiberhmm-call` re-calls nucleosomes and TFs with CpG-aware
recall, which keeps CpG observations inside the confident unmethylated
(`ddda_ucg`) islands, and keeps the island calls. Use `fiberhmm-call` here,
not `fiberhmm-recall-tfs`: a recall of DddA call output does not reproduce the
call ([Re-calling](recalling.md#ddda-call-output)).

On the demo data:

```text
[tag_m5c] defined_islands=2, reads=377, eligible_reads=74, overlapped_islands=86, methylated_islands=0, unmethylated_islands=50, uninformative_islands=36, tagged_reads=50
```

Given an `MD`-only BAM, `fiberhmm-tag-m5c` stops:

```text
input BAM has no Y/R-encoded DAF sequence in the first 377 primary reads; run fiberhmm-daf-encode before apply/tagging
```

How it works:

- **Islands.** By default the tagger infers islands from the reference:
  200 bp windows every 10 bp with GC fraction ≥ 0.50 and CpG
  observed/expected ≥ 0.60, merged. `--cpg-islands` supplies your own
  merged BED3 catalog; `--write-cpg-islands` records the one used.
- **Calls.** One state per molecule per complete island, from observations
  inside the molecule's initial MSP. A call needs ≥ 15 CpG observations
  (`--min-island-cpg`), ≥ 10 non-CpG observations (`--min-other`), and a
  posterior ≥ 0.99 for methylated or ≤ 0.01 for unmethylated (`--posterior`).
  Other overlaps are recorded as `uninformative` in `--calls-tsv`.
- **Output.** Confident methylated islands become `MA` intervals
  `ddda_mcg.`, confident unmethylated ones `ddda_ucg.`, each spanning the
  whole island. There is no boundary inside an island.
- **Recall.** CpG-aware recall (on for DddA in `fiberhmm-call`,
  `fiberhmm-recall-tfs`/`-recall-nucs` and `fiberhmm-pair`) excludes every CpG
  observation except those inside `ddda_ucg` islands the read carries. A
  first `fiberhmm-call` on fresh data therefore neutralizes all CpGs;
  re-calling after `fiberhmm-tag-m5c` restores the CpGs of confidently
  unmethylated islands. `--cpg-mask-policy methylated-only` reproduces the
  older policy of masking only `ddda_mcg` spans; `--no-use-m5c` turns
  CpG-aware recall off.
- The 5' context correction uses calibrated DddA factors;
  `--estimate-factors` re-estimates them from the BAM.

The per-CpG `fiberhmm-call --ddda-mcg` mode of development builds after
2.16.8 was retired and prints this workflow instead.

### Aggregate domains: `fiberhmm-call-m5c`

For validation, `fiberhmm-call-m5c` calls methylated and unmethylated domains
across all molecules in a region (1 kb windows, a two-state HMM over the
region) and writes BED6:

```bash
fiberhmm-call-m5c -i calls.initial.bam -r reference.fa --region chr1 -o domains.bed --enzyme ddda
```

```text
chrDemo	1000	28000	m5c_unmethylated	999	.
```

`--region` uses 1-based display syntax (`chr1`, `chr1:1-5000000`); a whole
contig is fine, observations are collected in `--chunk-bp` (5 Mb) chunks and
the HMM still runs once over the region. `--tag-bam` / `--tag-output` also
write island annotations into a BAM (`--tag-mode island`, or `locus` to copy
the aggregate domains).
