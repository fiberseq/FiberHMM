# Quality control

`fiberhmm-qc` grades a called BAM against a reference data set of the same
chemistry and draws a one-page report. It runs automatically at the end of
`fiberhmm-call` for file outputs, and can be run on any FiberHMM-compatible
BAM or on several at once.

```bash
fiberhmm-qc -i out/ddda.calls.bam                                   # one BAM
fiberhmm-qc -i out/pacbio.calls.bam out/dddb.calls.bam -o out/qc_compare   # several
```

## It is bounded

QC never reads a whole BAM. An indexed BAM is sampled through seeded random
genomic windows (168 windows of 20 kb, filled from a capped reservoir when
the index is sparse) to a target of `--sample-reads` (2,000) primary reads at
MAPQ ≥ `--min-mapq` (20); an unindexed BAM uses a reservoir over at most 10×
the target records. The JSON records the strategy, seed (`--seed`, default
20260824), records examined and `whole_bam_scanned: false`. QC never
deduplicates.

Because the sample uses aligned reads, QC of an unaligned call set reports
`INSUFFICIENT`; align the reads first.

## Outputs

One input writes `<BAM directory>/qc/<BAM stem>.qc.json` and `.qc.tsv`, plus
`.qc.png` and a vector `.qc.pdf` (TrueType text, editable in Illustrator)
when matplotlib is installed (`fiberhmm[plots]`). `-o/--output-dir` changes
the directory. After `fiberhmm-call` the prefix is
`<output BAM directory>/qc/<BAM stem>` unless `--qc-output-prefix` is set.

Several inputs write each sample's files plus
`combined.qc.{json,tsv,png,pdf,html}`; `combined.qc.html` is a
self-contained index page. Inputs from different directories need `-o`.

```text
out/qc_compare/combined.qc.html
out/qc_compare/combined.qc.json   combined.qc.pdf   combined.qc.png   combined.qc.tsv
out/qc_compare/pacbio.calls.qc.json   ... .pdf ... .png ... .tsv
out/qc_compare/dddb.calls.qc.json     ... .pdf ... .png ... .tsv
```

## The scorecard

The DddB demo call (`out/dddb.calls.bam`):

```text
  FiberHMM QC: WARN  score=64/100
  sample: 300 reads via indexed random windows (168 x 20,000 bp) + bounded fill
  deamination: 8.24% median  [PASS 92/100]
  nucleosome periodicity: NRL=193 bp, AC=0.680  [WARN 37/100]
  reference phasogram agreement: r=0.609, matched amplitude=1.91x
  footprint tags: nuc n=6,254, median=147 bp; TF n=1,597, median=51 bp
  nuc-tagged long spans: 0.9% >300 bp; 0.0% >1 kb
  PCR deduplication: flagged, duplicates=0.0% (full run)
  DAF SNP mask: 0 sites applied; MD preserved
```

- **Signal rate** (m6A labelling or deamination): the median per-read rate
  (reads with at least `--min-opportunities`, 200, targets). Inside the
  reference's interquartile range is PASS, elsewhere within its 5th–95th
  percentiles WARN, outside FAIL.
- **Nucleosome periodicity**: the autocorrelation of mark spacing, its
  nucleosome repeat length, and agreement with the reference phasogram. The
  grade is conjunctive: peak strength, repeat length, correlation with the
  reference curve and the amplitude of the reference-shaped component must
  all support it, so a flat or noisy curve cannot pass by chance.
- **Overall**: PASS, WARN or FAIL; `INSUFFICIENT` when there is too little
  evidence, rather than a failure.
- **Footprint sizes** (from `MA`, or legacy `nl` for nucleosomes), PCR
  duplication (exact from the call's `.dedup.json` sidecar when integrated
  dedup ran, otherwise estimated from `0x400`/`di`/`ds` in the sample), and
  the DAF SNP summary when available.

If the run's ML threshold differs by more than 5 from the reference's
calibration threshold, the rate score is capped at WARN and the mismatch is
reported. The defaults match the references (248 for Hia5 Nanopore, 125
otherwise).

The figure shows the sample against the reference: rate ECDF and median,
phasogram, nucleosome and TF size distributions, duplication, a mismatch
landscape, an amplicon SNP map and example molecules. The combined figure
compares signal median, periodicity score, repeat length, autocorrelation
strength and nucleosome and TF size across samples.

Reference intervals are screening diagnostics, not biological exclusion
criteria: a sample outside them deserves a look, not automatic rejection.

## Reference data sets

| Profile | Source |
|---|---|
| `dddb` | Drosophila embryo DddB DAF-seq |
| `ddda` | human DddA DAF-seq (NAPA and UBA1 amplicons) |
| `hia5_pacbio` | Drosophila embryo PacBio Fiber-seq |
| `hia5_nanopore` | Drosophila embryo Nanopore Fiber-seq (ML ≥ 248) |

Only aggregate curves and a few anonymized example molecules are packaged,
never reads or coordinates; provenance and construction are in
[`fiberhmm/qc/README.md`](https://github.com/fiberseq/FiberHMM/blob/main/fiberhmm/qc/README.md).

After `fiberhmm-call`, the profile is chosen from the resolved enzyme and
platform. Standalone `fiberhmm-qc` infers mode and enzyme from the BAM's
header (`--mode`, `--enzyme` override) and picks the matching profile;
`--reference-profile` chooses one explicitly or `none`. Incompatible
combinations are rejected.

## Options

| Option | Default | Use |
|---|---|---|
| `--prob-threshold` | 248 for Hia5 Nanopore, 125 otherwise | ML threshold for MM/ML calls |
| `--reference` | — | indexed FASTA for raw DAF BAMs without `MD` or R/Y |
| `--snp-mask BED`, `--snp-report JSON` | — | single input: apply a SNP mask to the rate and periodicity, and plot a `fiberhmm-daf-snps` report |
| `--fail-on-qc` | off | exit 2 when the overall status is FAIL (WARN and INSUFFICIENT exit 0) |

Every option: [`fiberhmm-qc`](../reference/cli.md#fiberhmm-qc).
