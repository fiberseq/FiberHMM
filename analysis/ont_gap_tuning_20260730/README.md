# ONT Drosophila Fiber-seq gap tuning (2026-07-30)

## Conclusion

The ONT excess is a nucleosome-recaller failure caused by sparse,
single-strand evidence being interpreted as positive evidence of
accessibility. It is not a simple Dorado m6A burden or a threshold problem.

Two historical recaller decisions compound:

1. every accessible Kadane run splits an HMM footprint, even when the cut
   shatters one nucleosome into sub-85-bp fragments; and
2. neutral/ambiguous sequence outside the conservative protected core is
   relabeled accessible.

The implemented `topology` policy fixes both. It chooses the maximum-evidence
cut chain that leaves a nucleosome-sized fragment on every outer and
intervening side, then retains the post-cut HMM fragments as occupancy
intervals. Unresolved boundaries receive zero edge-sharpness bytes rather than
becoming NFR sequence.

End-to-end calls on matched train/holdout subsets are:

| Configuration | Nucleosome coverage | MSP/NFR coverage, calls >=60 bp | Nucs/kb | Median nuc | Median NFR |
|---|---:|---:|---:|---:|---:|
| Existing PacBio 2-4 hr (`as/al`, legacy semantics) | 82.9% | 6.0% | 3.25 | 164 bp | 82 bp |
| PacBio, current caller, no nuc recaller | 83.0% | 11.1% | 3.25 | 163 bp | 89 bp |
| ONT 1.5-3 hr, supplied calls | 71.0% | 24.4% | — | 159 bp | 115 bp |
| ONT 3-4.5 hr, supplied calls | 71.4% | 23.5% | — | 153 bp | 108 bp |
| **ONT 1.5-3 hr, topology recall** | **82.1%** | **13.2%** | **2.80** | **177 bp** | **105 bp** |
| **ONT 3-4.5 hr, topology recall** | **83.0%** | **11.6%** | **3.02** | **171 bp** | **97 bp** |

The older PacBio `as/al` track is not a direct NFR target. Re-deriving gaps from
its trusted `ns/nl` calls gives 13.7% NFR coverage at `eve`, while its stored
legacy `as/al` reports only 7.4%. Current FiberHMM intentionally reports
inter-nucleosome gaps, including space containing short protected footprints.

## Recommended ONT call

Keep the empirically necessary hard-call threshold (`ML >= 248`), use the new
topology recaller, and retain the 60 bp NFR/MSP floor. The policy is selected by
`auto` for Nanopore, but it is written explicitly here for reproducible
provenance. `phase-nrl off` is the conservative choice for these data; enabling
the phase prior changed nucleosome subdivision but had almost no effect on
reported NFRs >=60 bp.

```bash
fiberhmm-call \
  --input INPUT.bam \
  --output OUTPUT.bam \
  --enzyme hia5 \
  --seq nanopore \
  --prob-threshold 248 \
  --recall-nucs \
  --nuc-recall-policy topology \
  --phase-nrl off \
  --msp-min-size 60 \
  --region-parallel \
  --skip-scaffolds \
  --cores 12 \
  --io-threads 8
```

[`run_recommended_ont.sh`](run_recommended_ont.sh) wraps this command.

Use the pre-FiberHMM aligned BAM as input when available. A previously called
BAM is also accepted, but using the raw aligned input avoids carrying stale tags
on reads skipped by length or alignment filters.

## What was tested

The comparison used one BAM per cohort and eight 50 kb windows spread across
chr2L, chr2R, chr3L, and chr3R. Four windows were designated tuning loci and
four were held out. Primary alignments were hash-subsampled to similar molecule
counts:

- ONT 1.5-3 hr: 4,679 local reads, 2,985 analyzed reads >=1 kb.
- ONT 3-4.5 hr: 5,114 local reads, 3,623 analyzed reads >=1 kb.
- PacBio 2-4 hr: 6,221 local reads, 6,216 tagged reads >=1 kb.

Train and holdout results were effectively identical:

| Tuned ONT cohort | Split | Nucleosome coverage | MSP >=60 bp |
|---|---|---:|---:|
| 1.5-3 hr | train | 82.0% | 13.2% |
| 1.5-3 hr | holdout | 82.1% | 13.2% |
| 3-4.5 hr | train | 82.8% | 11.6% |
| 3-4.5 hr | holdout | 83.1% | 11.6% |

### Why the old recaller went haywire

Holding `ML >= 248` fixed:

| ONT setting | 1.5-3 hr MSP >=60 bp | 3-4.5 hr MSP >=60 bp |
|---|---:|---:|
| Supplied/default recaller | 24.4% | 23.5% |
| Phase off, default split (LLR 4/opps 3) | 24.4% | 23.4% |
| Phase off, conservative split (LLR 8/opps 5) | 22.4% | 21.2% |
| Phase off, effectively no splitting (LLR 100/opps 10) | 20.1% | 18.9% |
| **Nucleosome recaller off** | **12.1%** | **10.4%** |

The auto phase prior contributes little. Higher split thresholds help, but even
preventing splits leaves an 8-9 percentage-point excess relative to turning the
recaller off. Stage-level instrumentation on 2,000 reads per cohort showed:

- default split cuts consume 5.7-5.8% of raw ONT nucleosome bp versus 0.9% in
  PacBio;
- conservative edge trimming removes 5.7-6.1% of eligible ONT fragment bp
  versus 1.6% in PacBio; and
- ONT has roughly tenfold more fragment bp demoted because the conservative
  protected core falls below 85 bp (about 2.1-2.2% of raw nuc bp versus 0.2%).

The topology constraint retains only cuts that can actually separate
nucleosomes. On the audit cohort it reduced ONT NFR coverage from 23-25% to
11-13% while raising nucleosome density by splitting genuine long over-merges.

### eve NFR population benchmark

The acceptance target was molecule-level geometry, not a scalar accessible
fraction. Calls were projected to dm6 in a 52 kb window centered on
`eve` (`chr2R:9,979,318-9,980,795`) and compared by NFR length, per-read count,
and positional occupancy. PacBio NFRs were re-derived as >=60 bp gaps between
its existing nucleosome calls to avoid the old `as/al` semantic difference.

| Population | NFR fraction | Median | p75 | p90 | Median NFRs / 10 kb / molecule |
|---|---:|---:|---:|---:|---:|
| PacBio 2-4 hr, existing nuc complement | 13.7% | 95 bp | 153 bp | 263 bp | 9.02 |
| ONT early, supplied | 27.4% | 127 bp | 247 bp | 424 bp | 14.77 |
| **ONT early, topology output** | **15.0%** | **112 bp** | **190 bp** | **305 bp** | **9.75** |
| ONT late, supplied | 25.6% | 117 bp | 229 bp | 377 bp | 15.05 |
| **ONT late, topology output** | **12.8%** | **102 bp** | **167 bp** | **267 bp** | **9.24** |

Against the PacBio population, end-to-end NFR length quantile error fell from
61 to 20 bp for early ONT and from 43 to 8 bp for late ONT.
Positional-occupancy RMSE across the `eve` window fell from 0.191 to 0.116 and
from 0.180 to 0.108, respectively. The fix therefore restores the size/count
population while
retaining locus-specific open-chromatin structure.

### Raw m6A evidence at threshold 248

Overall high-confidence m6A calls per target opportunity were 7.68% (early ONT),
7.53% (late ONT), and 9.22% (PacBio). The ONT BAMs therefore do not simply have
a larger global high-confidence m6A burden. ONT events are more isolated than
PacBio events, as expected from single-strand sampling and/or residual caller
noise. The topology recaller therefore treats the HMM footprint as the
occupancy prior and requires a cut to establish valid multi-nucleosome
topology.

## Artifacts

- [`gap_tuning_summary.png`](results/gap_tuning_summary.png) and
  [`gap_tuning_summary.pdf`](results/gap_tuning_summary.pdf): main comparison.
- [`summary_msp60.tsv`](results/summary_msp60.tsv): aggregate, train/holdout,
  and per-region call geometry.
- [`topology_output_summary.tsv`](results/topology_output_summary.tsv):
  end-to-end topology output on global train/holdout windows.
- [`per_read_msp60.tsv`](results/per_read_msp60.tsv): per-read metrics.
- [`raw_m6a_thr248.tsv`](results/raw_m6a_thr248.tsv): raw m6A contrast.
- [`eve_nfr_population.png`](results/eve/eve_nfr_population.png): NFR size,
  per-molecule count, and positional occupancy around `eve`.
- [`eve_summary.tsv`](results/eve/eve_summary.tsv) and
  [`eve_comparison_to_pacbio.tsv`](results/eve_end_to_end/eve_comparison_to_pacbio.tsv):
  locus population metrics and end-to-end validation.
- [`instrument_recaller.py`](instrument_recaller.py) and
  [`benchmark_eve_nfrs.py`](benchmark_eve_nfrs.py): stage audit and locus
  benchmark.
- [`summarize_calls.py`](summarize_calls.py),
  [`summarize_raw_m6a.py`](summarize_raw_m6a.py), and
  [`plot_summary.py`](plot_summary.py): reproducible analysis.
- [`regions.bed`](regions.bed): full 20-window candidate panel; eight listed
  windows were used for the compact local BAMs.

The compact source subsets and tuned BAMs are in
`/tmp/fiberhmm_ont_gap_tuning_20260730/` (about 0.54 GB for the three inputs and
1.4 GB total including eve subsets and retained tuned outputs). Trial-only BAMs
were removed.
No full 88-180 GB BAM was copied.

## Provenance note

The two supplied filenames begin with `WT_`, while their embedded original
FiberHMM command lines use `siGAF_yw_*`. This is a resolved naming issue: the
whole batch was initially mislabeled `siGAF`, and these files are the renamed
yellow-white (`yw`)/WT controls. They are distinct from the separate `siGAF_*`
BAMs in the same directory.
