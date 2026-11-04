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
Clusters connect all qualifying pairs found in shared hash buckets. A read
can bridge two families whose representatives do not directly match. The
similarity and endpoint limits apply to each connecting pair, not every pair
in the final family. Hash-based candidate retrieval is approximate.
Within a cluster, the representative is the read with the highest MAPQ, then
the most complete alignment. Supplementary and secondary records of a
duplicate read are flagged with it.

Marks from an earlier run are recomputed, not accumulated: before marking,
both the integrated and the standalone dedup clear `0x400`, `di` and `ds` on
every input record (including marks set by another tool) and report how many
records they cleared. Re-deduplicating a called or deduplicated BAM, or
re-calling it, therefore gives the marks of the current settings.

**Integrated (automatic).** `fiberhmm-call --enzyme ddda|dddb` on a file
input marks duplicates before SNP screening and calling. It is
**nondestructive**: every read is kept; non-representatives get flag `0x400`;
every member of a duplicate cluster gets `di` (cluster id) and `ds` (cluster
size). Pooled SNP, periodicity and QC statistics ignore flagged copies. A
`<stem>.dedup.json` report goes to the QC directory. Options:
`--no-dedup` (off), `--dedup` (force), `--dedup-collapse` (remove the
non-representatives instead), and `--dedup-min-jaccard`,
`--dedup-max-end-diff`, `--dedup-min-deam`, `--dedup-prob-threshold`,
`--dedup-ignore-strand`, `--dedup-stats-tsv`. `--dedup-prob-threshold`
defaults to the call's `--prob-threshold` (128). Integrated dedup needs a file
input (not stdin) and does not use `--reference`.

**Standalone.** `fiberhmm-dedup` **collapses** by default, keeping one read
per cluster; `--flag-only` marks instead. Its options drop the `dedup-`
prefix (`--min-jaccard`, `--max-end-diff`, `--min-deam`, `--ignore-strand`,
`--stats-tsv`, `-p/--prob-threshold`, default 128). The output is published
atomically:

```bash
fiberhmm-dedup -i demo/ddda.bam -o out/ddda.dedup.bam
fiberhmm-dedup -i demo/ddda.bam -o out/ddda.markdup.bam --flag-only --stats-tsv out/ddda.clusters.tsv
```

**Optional 90% matching.** The default remains 0.95. For a library with
residual near-identical fingerprints, compare the default with a lower
threshold, retaining the same endpoint and deamination-flavour checks:

```bash
fiberhmm-call -i aligned.bam -o calls.j90.bam --enzyme dddb --dedup-min-jaccard 0.90
fiberhmm-dedup -i aligned.bam -o marked.j90.bam --flag-only --min-jaccard 0.90
```

Use the original aligned BAM, since reads removed by an earlier collapse
cannot come back (earlier marks are cleared and recomputed). Lower thresholds merge more
families and may merge distinct molecules; choose based on fingerprint
separation and controls, rather than the desired accessibility distribution.

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

A read deaminated C→T in one segment and G→A in another is a strand-swap
chimera. This detection is specific to DAF (DddA/DddB); m6A modes do not
perform strand-swap detection. The detector orders a read's deaminations
along the query and looks for contiguous segments of alternating strand.
Each segment needs:

- at least `--chimera-min-seg` (default 10) same-strand deaminations,
- at `--chimera-purity` (default 0.9) or greater purity,
- a span of at least `--chimera-min-span` (default 200) query bp from its
  first to its last same-strand deamination (0 disables), so a dense
  cluster of mismatches from one misaligned stretch is not a segment.

Only deaminations with base quality of at least `--chimera-min-baseq`
(default 15) count when the record has QUAL; records without QUAL are not
filtered, and 0 disables the filter. Low-quality basecalls are the main
source of opposite-strand noise on nanopore reads. The detector uses the
same reference, SNP and unaligned-base masks as calling, plus the
chimera-only error mask described below. The base-quality filter and the
error mask apply to the chimera test only; calling uses every deamination
as before.

A read with one switch (CT then GA) has two segments. With
`--chimera-multi-switch` (default on) a read with several switches, such
as an opposite-strand island inside a read (CT → GA → CT), is also a
chimera when every segment passes the same rule; `--no-chimera-multi-switch`
tests one switch only. These defaults are strict, so that a flagged read is
unambiguous. FiberHMM 3.0.3 used min_seg 5, purity 0.8, no base-quality or
span limit and one switch;
`--chimera-min-seg 5 --chimera-purity 0.8 --chimera-min-baseq 0
--chimera-min-span 0 --no-chimera-multi-switch --no-chimera-error-mask`
restores that behaviour (output records match 3.0.3; the `@PG` DS line
differs). One decision drives flag, remove and split, the count and the
grade.

### Chimera error mask

Some sites give recurrent opposite-strand mismatches on one strand only, for
example nanopore miscalls in a sequence motif. The SNP screen does not mask
them, because it requires support on both strands. When the
[SNP screen](#snp-screening) runs, its two passes also build a chimera-only
error mask, with no extra pass over the BAM. Only clearly single-strand
fibers vote: at least 10 deaminations of one strand at purity 0.92 or more,
counting deaminations with base quality of at least 15 (fixed, independent
of `--chimera-min-baseq`; records without QUAL are not filtered), and not
chimeric under the run's detector. Chimeric fibers do not vote, so
real accessible sites on chimeric reads are not masked. A site is masked
for the opposite-strand event when, among voters of the other strand, it has

- depth of at least 20 fibers,
- at least 4 fibers with the opposite-strand mismatch,
- a mismatch fraction of at least 0.05, and
- a binomial survival P(X ≥ k | depth, p = 0.01) below 1e-4.

The mask goes to `qc/<stem>.daf_snps.chimera_error_mask.bed` and `.json`,
never into `<stem>.daf_snps.bed`, and is never applied to calling. Sites
already called as SNPs are not repeated. Without the SNP screen (the
low-depth preflight skipped it, stdin/stdout, `--no-daf-call-snps`, or a
user-supplied `--daf-snp-mask`) there is no error mask; the chimeras sidecar
records that and the reason, and the base-quality filter still applies.
`--no-chimera-error-mask` turns the mask off.

`fiberhmm-call --chimeras flag` is the default: retain the read uncalled,
set SAM QC-fail `0x200`, and write the local tag `xc:Z:strand_swap` naming
the reason. All supplementary and secondary records of that read receive
the same mark. `--chimeras remove` drops all records of detected reads;
`--chimeras keep` disables detection and attempts calls. `--keep-chimeras`
is a deprecated alias for `--chimeras keep`. `--chimeras split` is
described below.

Re-running flag/remove clears earlier `xc:Z:strand_swap` marks and recomputes
decisions using the run's alignment selection. Only the current run's decision
controls NRL estimation, insertion evidence and calling. Keep mode (including
`--keep-chimeras`) clears earlier FiberHMM marks and calls those reads as ordinary
reads, so downstream extract, consensus and posterior export can use them.
The companion `xq:i` tag
records whether QC-fail was already set (1) or FiberHMM set it (0). Cleanup
clears 0x200 only with `xq:i:0`; pre-existing QC failures, older marks without
ownership metadata, and foreign `xc` values preserve the QC-fail flag.
The QNAME-aware pre-call pass reads regular files without copying them;
only stdin/non-seekable input is spooled. The caller applies the decisions
to every record, including a supplementary record preceding its primary.
m6A and keep mode bypass this pass entirely. Keep mode clears marks while
reading records for NRL estimation and calling; m6A leaves them alone.
`fiberhmm-pipeline` also accepts `--chimeras`; its legacy `--chimera-filter`
and `--no-chimera-filter` map to flag and keep, respectively. Pipeline settings
carry the exact effective mode in `call_args` so FiberBrowser re-calls preserve
remove mode as well as flag and keep.

The run summary lists `chimera` with other skip reasons. QC JSON has a
`chimeras` section (mode, checked read names, chimeric read names, fraction,
thresholds and sampled evidence). The authoritative full-run counts,
measured before removal, are saved in `qc/<stem>.chimeras.json`, even with
`--no-qc`. Duplicate-flagged (`0x400`) reads are tested and marked like any
read but are not counted: they are left out of `n_reads_checked`,
`n_chimeric`, the fraction, the regions and the grade, and
`n_duplicates_excluded` reports how many were left out. The sidecar and the
`@PG` DS line record the detector settings (`min_seg`, `purity`,
`min_baseq`, `min_span`, `multi_switch`), the error mask (applied or not,
number of sites, reason) and the duplicates left out. Keep this sidecar beside the
BAM for standalone QC to reproduce remove-mode counts. QC TSV and PDF/PNG
report chimeras alongside PCR duplication.
Without the sidecar, standalone QC with the default auto assay mode reports
the DAF policy from the BAM header and the fraction of marked sampled reads.

### How many reads are chimeric

Each run counts the reads the detector calls (`n_chimeric`, `fraction`).
The fraction is graded: PASS below 10%, WARN from 10%, and WARN with
severity `high` from 20%. At 10% or more `fiberhmm-call` and
`fiberhmm-pipeline` print a `WARNING:` line on stderr (in the pipeline log
too). The count, grade and a per-region table (reads grouped by the 10-kb
start bin of their primary record; adjacent occupied bins form a region)
are in `qc/<stem>.chimeras.json` and the QC report. A count made with other
detector settings, such as a FiberHMM 3.0.3 sidecar or BAM (5 and 0.8), is
graded as recorded, and the grade names its basis (`lenient`, with a note).

Opposite-strand chimeras are the only kind this test can see. A chimera of
two molecules deaminated on the same strand looks like an ordinary read.
With balanced strands about half of all chimeras are same-strand, so the
unflagged reads still contain roughly as many undetected chimeras as were
flagged: at a flagged fraction of 10%, about one unflagged read in nine is
still a chimera. Read counts, occupancy and long-range results
(co-accessibility, phasing) from a library with many chimeras should be
taken with caution. Every chimera report carries this note.

### Split mode

`--chimeras split` is an option for recovering data from chimeric reads; it
is not the default. For each alignment record that is itself chimeric, the
detector's best single switch gives a breakpoint gap: from the last
event of the first segment's strand to the first event of the second
segment's strand. Where the switch happened inside the gap is unknown, so
the gap is dropped and the record is written as single-strand records. Each
piece is tested again with its own deaminations and cut again while it is
still chimeric; a read with several switches (found with
`--chimera-multi-switch`) is cut at every switch. All valid pieces are
written:

- `<name>/c0`, `<name>/c1`, ... from left to right on the reference. The
  first piece ends on its last own-strand event, the last piece starts on
  its first own-strand event, and an interior piece runs from its first to
  its last own-strand event. No piece ends in an event-free stretch made by
  the cut, so the cut does not create footprints at the inner edges.
  Breakpoints are anchored on deaminations that pass
  `--chimera-min-baseq`, so a low-quality basecall inside the gap does not
  extend a piece past the switch.
- CIGAR, SEQ, QUAL and MD are cut together, `NM` is recomputed and the
  reference start is adjusted. Outer soft and hard clips stay with their
  piece.
- Tags: `xc:Z:split`, `xi:i` (piece index, 0 to n-1 from left to right),
  `xn:Z` (original read name) and `xb:B:i` (every breakpoint gap of the
  original record as `[start, end)` pairs, 0-based on its reference, left to
  right; piece `xi` lies between gap `xi`-1 and gap `xi`; a one-switch split
  has one pair).
  `RG`, `ds`, `di`, `qs`, `du`, `ch`, `rn` and `fn` are copied.
  `MM`/`ML` (and the older `Mm`/`Ml`) are kept, re-indexed to the piece: the
  calls inside the piece stay, skip counts are recomputed over the piece in
  the read's original orientation (reverse-complemented for reverse-strand
  records, for `+` and `-` entries alike), ML values follow their calls, and
  each entry keeps its codes and `.`/`?` flag. `MM` without `ML` (ML is
  optional) is re-indexed on its own. `MN` is set to the piece's
  length. A piece of a hard-clipped record, or of one whose `MM`/`ML` do not
  parse or do not match SEQ, gets no `MM`/`ML` (counted as
  `n_pieces_modifications_dropped`). Every other position-dependent tag
  (`SA`, the DAF strand tag `st`, earlier FiberHMM calls) is dropped.
- Pieces keep the record's flags and are not QC-fail. Pieces shorter than
  `--min-read-length` (aligned bases) are dropped and counted; a record with
  no piece left is kept whole and flagged.
- Other records of a chimeric read (supplementary or secondary records that
  are not chimeric themselves, or are not selected by `--alignments`) are
  flagged as in flag mode. Unnamed records are not split. Unmapped records
  (`--process-unmapped`, or streamed input) are not split either: they have
  no CIGAR or reference position to cut, so a chimeric unmapped record is
  kept whole and flagged (counted as `n_records_unmapped_kept_whole`). A
  piece that still shows a swap after cutting is flagged.
- Each piece is called with its own strand. Split mode runs before the
  insert-consensus pre-pass, so pieces carry their own insertion evidence.

The pieces are written to a private temporary BAM (sorted and indexed when
the input is coordinate-sorted). The chimeras sidecar and QC report list
records split, records with several switches, pieces written, pieces
dropped as too short, the fraction of chimeric bases kept, and the
distributions of the reference gap and piece lengths. Caveats: the switch point is only bracketed, not found; two
pieces of one read are two molecules of unknown relation (they may be PCR
copies or unrelated fragments); and same-strand chimeras are not split.

Extract, consensus/recall inputs and posterior export exclude `0x200` +
`xc:Z:strand_swap` records by default. `fiberhmm-footprint-model` takes
`--include-chimeras`, next to its duplicate option. Other QC-fail records
keep each tool's existing policy.

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
calling, which keeps CpG observations inside the confident unmethylated
(`ddda_ucg`) islands, and keeps the island calls. Use `fiberhmm-call` here,
not `fiberhmm-recall-tfs`: a recall of DddA call output does not reproduce the
call ([Re-calling](recalling.md#ddda-call-output)).

On the demo data:

```text
[tag_m5c] defined_islands=2, reads=377, eligible_reads=74, overlapped_islands=86, methylated_islands=0, unmethylated_islands=50, uninformative_islands=36, tagged_reads=50
```

The second `fiberhmm-call` is a complete call: it clears and re-marks the
duplicates (`cleared earlier duplicate marks (0x400/di/ds) on 78 input
records before marking` on the demo data), re-runs the SNP screen and masks
the sites it finds, and writes its own QC.

`fiberhmm-tag-m5c` writes the BAM, `--calls-tsv` and `--write-cpg-islands`
under hidden names and publishes them together only when the run succeeds,
so a failed run keeps the previous files. The output is not indexed (the
index of an earlier output at that path is removed). No output may be an
input, an input's index, another output, or the index an output's
publication would replace; such a run stops before writing anything:

```text
fiberhmm-tag-m5c: error: ... --output out/m/calls.initial.bam is the same file as --input out/m/calls.initial.bam; writing it would destroy the input. Choose a different output path
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
- **Calling.** CpG-aware calling (on for DddA) excludes every CpG
  observation except those inside `ddda_ucg` islands the read carries. The
  mask applies to the first-pass HMM and to nucleosome and TF recall in
  `fiberhmm-call` (and so in `fiberhmm-pipeline`), to the HMM of
  `fiberhmm-apply` and `fiberhmm-posteriors`, to the joint HMM and recall of
  `fiberhmm-pair`/`-merge`, to `fiberhmm-recall-tfs`/`-recall-nucs`, and to
  the light call of `fiberhmm-qc`, so every tool segments a read the same
  way. Each of these except `fiberhmm-qc` takes `--use-m5c`/`--no-use-m5c`
  and `--cpg-mask-policy`; `fiberhmm-call` records `cpg_mask=` and
  `hmm_cpg_mask=` in its `@PG`. A first `fiberhmm-call` on fresh data therefore neutralizes all
  CpGs; re-calling after `fiberhmm-tag-m5c` restores the CpGs of confidently
  unmethylated islands. `--cpg-mask-policy methylated-only` reproduces the
  older policy of masking only `ddda_mcg` spans; `--no-use-m5c` turns
  CpG-aware calling off.
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
chrDemo	0	29000	m5c_unmethylated	999	.
```

`--region` uses 1-based display syntax (`chr1`, `chr1:1-5000000`); a whole
contig is fine, observations are collected in `--chunk-bp` (5 Mb) chunks and
the HMM still runs once over the region. `--tag-bam` / `--tag-output` also
write island annotations into a BAM (`--tag-mode island`, or `locus` to copy
the aggregate domains). The BED, the tagged BAM and the island catalog are
published together only when the run succeeds, and an output path that
matches an input BAM, another output or the reference stops the run before
anything is written.
