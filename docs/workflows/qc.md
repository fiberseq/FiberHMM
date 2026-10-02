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
when matplotlib is installed (`fiberhmm[plots]`), and `.qc.curves.json`, the
data behind the plot (see [below](#the-curves-file)). `-o/--output-dir` changes
the directory. After `fiberhmm-call` the prefix is
`<output BAM directory>/qc/<BAM stem>` unless `--qc-output-prefix` is set.

Several inputs write each sample's files plus
`combined.qc.{json,tsv,html}` (and `combined.qc.{png,pdf}` with
`fiberhmm[plots]`); `combined.qc.html` is a self-contained index page.
Inputs from different directories need `-o`. With `fiberhmm[plots]`
installed:

```text
out/qc_compare/combined.qc.html
out/qc_compare/combined.qc.json   combined.qc.pdf   combined.qc.png   combined.qc.tsv
out/qc_compare/pacbio.calls.qc.json   ... .pdf ... .png ... .tsv
out/qc_compare/dddb.calls.qc.json     ... .pdf ... .png ... .tsv
```

## The curves file

`<prefix>.qc.curves.json` (schema `fiberhmm.qc.curves.v1`) holds what the
plot draws, so another program (FiberBrowser, a notebook) can redraw it:

| Key | Contents |
|---|---|
| `verdicts` | `overall`, `signal`, `periodicity`, `efficiency`, `background` status (PASS/WARN/FAIL/INSUFFICIENT) and scores, and `verdict_basis` |
| `state_rates` | `source`, `min_msp_bp`, the sorted per-read `msp_per_read_rates` and `outside_msp_per_read_rates`, and the reference quantiles (`null` when unavailable) |
| `signal_rate` | `per_read_rates` (sorted fractions, one per sampled read), their `ecdf`, the bundled `reference_ecdf` (`rates`, `probabilities`), `warn_interval`, `reference_iqr` |
| `phasogram` | `lags_bp`, the sample's `detrended_pair_frequency`, raw `pair_distance_counts`, `nrl_bp` and the bundled `reference` curve |
| `footprint_sizes` | `nucleosome` and `tf`: `bin_edges_bp`, `sample_count_per_bin`, `sample_fraction_per_bin`, `sample_n`, `sample_overflow` and `reference_fraction_per_bin` |
| `duplicates` | `duplicate_fraction` and the molecule cluster-size histogram |

Rates are fractions, not percent. Reference parts are `null` when the assay
has no bundled reference.

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
- **In-MSP rate (efficiency) and outside-MSP rate (background)**: see
  [State-aware rates](#state-aware-rates).
- **Overall**: PASS, WARN or FAIL; `INSUFFICIENT` when there is too little
  evidence, rather than a failure. Where the reference has a state-aware
  calibration, the overall status combines efficiency, background and
  periodicity (`overall.verdict_basis: state-aware`); otherwise it combines
  the signal rate and periodicity (`overall-rate`). Any FAIL fails; any WARN
  or INSUFFICIENT component gives WARN.
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

## State-aware rates

The overall modification rate mixes two things: how efficiently the enzyme
labels accessible DNA, and how much of each molecule is accessible. A
targeted amplicon at an open locus can have 20% of its length in MSPs where
genome-wide data have 10%, so its overall rate is higher (or, under a
different opportunity count, lower) at the *same* enzyme efficiency. QC
therefore splits every sampled molecule by FiberHMM state:

| Metric | Meaning |
|---|---|
| in-MSP rate (`efficiency`) | modified / opportunities inside MSPs of ≥ `--min-msp-bp` (85) bp: how well the enzyme labels accessible DNA |
| outside-MSP rate (`background`) | the same everywhere else: nucleosomes, linkers and accessible gaps shorter than 85 bp. Internal and linker marks are pooled on purpose |
| `msp_to_outside_ratio` | pooled in-MSP rate / pooled outside-MSP rate: signal over background |
| `msp_length_fraction` | share of the sampled read length inside MSPs: how open the sample is |
| `all_states` | the overall rate under the same opportunity count, for comparison |

On the example data, a DddB amplicon set and the genome-wide DddB reference
differ 1.3-fold in overall rate and 1.8-fold in MSP share, while their
in-MSP rates agree within a few percent (42% and 45%).

**Where the states come from.** If at least half of the sampled reads carry
FiberHMM calls (`MA`, fibertools `Ma`, or legacy `ns/nl/as/al` arrays, in
the frame the header declares), those calls are used (`state_rates.source:
tags`). Otherwise QC runs a *light call*: the bundled apply HMM of the
declared chemistry, through the same extraction, encoding and Viterbi code
as `fiberhmm-call`, without nucleosome or TF recall, on at most
`--light-call-reads` (400) sampled reads, taken in a seeded random order, and
a `--light-call-seconds` (60 s) budget checked between reads (a soft limit:
one read in progress finishes; a time-stopped subset depends on machine
speed) (`source: light_call`, with model, SHA-256, reads called and what
stopped it). fibertools' own calls (`Ma` without `MA`) are reported as
`fibertools_tags` and not graded, since the references are FiberHMM calls.
`--state-source tags|light-call|none` forces a source or skips.

**Opportunities** are read off the observation encoding calling uses, so they
are exactly the sites the HMM sees: DAF targets on the deaminated strand only
(C on CT reads, G on GA reads) with the adjacent-run mask the call recorded
in its `@PG` (else the chemistry default: keep-one on runs ≥ 2 for DddA);
sites under the DAF SNP mask are removed as both events and opportunities;
Hia5 PacBio A and T; Hia5 Nanopore the basecalled-strand A;
MM `?`-unlisted bases excluded; 10 bases at each read end excluded as in
calling. These differ from the signal rate's opportunities (all aligned C/G
for DAF), so `all_states` and `signal` are not the same number.

**Read ends.** A terminal state segment is truncated by the read end: a
nucleosome cut below 85 bp no longer bounds an MSP and joins it, and an MSP
cut below 85 bp counts as outside. Each terminal segment is excluded up to
85 bp from its read end (`definition.terminal_segments: cap`), a heuristic
that removes most of these errors (a partial nucleosome followed by a linker
can carry a few linker bases past it).

**Grading.** Each compartment reports pooled `n_events`, `n_opportunities`,
`aggregate_rate` with a nominal binomial `aggregate_rate_wilson95` (it
ignores molecule-to-molecule variation and state-calling uncertainty, so it
is narrower than a molecule bootstrap), and per-read quantiles over reads with
at least 50 opportunities in the compartment.
Efficiency grades the median per-read in-MSP rate against the reference's
per-read in-MSP rates, one-sided: PASS at or above the reference 25th
percentile, WARN down to the 5th, FAIL below. Background grades the median
per-read outside-MSP rate: PASS at or below the reference 75th percentile,
WARN up to the 95th, FAIL above. The reference is the one computed from the
same state source (calls made by this release's `fiberhmm-call` defaults, or
the light call), under the same definition; a different `--min-msp-bp`
reports the rates without grading them. The quantiles bounding PASS/WARN are
`state_rates.grading` in `fiberhmm/qc/references.json`. As for the signal
rate, an ML threshold more than 5 away from the reference's caps the grade at
WARN. These are screening thresholds on the reference's molecules, not
false-failure probabilities for a sample median. When one compartment lacks
evidence (fewer than 20 reads with enough opportunities) the other still
decides with periodicity; a graded FAIL is never hidden.

Both rates are conditioned on states inferred from the same marks, so they
are empirical diagnostics rather than direct biochemical measurements: a
change in state calling, read length or sequence composition can move them
without any change in the enzyme.

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
| `--state-source` | `auto` | in-MSP/outside-MSP states from the BAM's calls (`tags`), a bounded `light-call`, or `none` |
| `--min-msp-bp` | 85 | shortest MSP counted as accessible; references are calibrated at 85 |
| `--light-call-reads`, `--light-call-seconds` | 400, 60 | bounds of the light call on uncalled BAMs |
| `--fail-on-qc` | off | exit 2 when the overall status is FAIL (WARN and INSUFFICIENT exit 0) |

Every option: [`fiberhmm-qc`](../reference/cli.md#fiberhmm-qc).
