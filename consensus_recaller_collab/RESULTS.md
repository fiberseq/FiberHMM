# Provisional prototype results — 2026-07-11

> **Superseded for PacBio composite deconvolution.** The PacBio EM model below
> cannot identify a continuous nucleosome versus a TF tiling that protects the
> same bases. Use [`PRODUCTION_VALIDATION.md`](./PRODUCTION_VALIDATION.md) and
> [`revised_prototype.py`](./revised_prototype.py) for the current two-pass
> implementation. The stranded DAF/Nanopore exploration remains useful history.

These are **candidate call-site events**, not validated calls or unique
molecules. No BAM was rewritten. The original exploratory reports were made in
disposable storage and are not the reproducible handoff. Use the retained
`consensus_visualization_outputs/paired_v7/` reports and
[`PAIRED_VALIDATION.md`](./PAIRED_VALIDATION.md) for the current panel. The
historical run used:

- posterior threshold 0.95;
- a positive target-molecule Bayes factor versus the current call;
- at least one informative opportunity and positive LLR in every proposed TF;
- an empirical prior learned from the other strand for stranded assays, or
  from pooled duplex molecules for PacBio;
- at most eight strongly supported, locally enriched focal sites per region;
- hard modification calls only (`ML >= 248` for Nanopore Hia5); and
- DddA radial-N likelihood marginalized across a +/-20 bp dyad prior.

The table reports the conservative `100x` existing-nucleosome prior scenario.

| Chemistry | Region / callset | Reads | Sites | MSP→TF | nuc→TF/gap tiling |
|---|---|---:|---:|---:|---:|
| DddB DAF | sna, chr2L:15,472,750-15,490,350 | 17,182 | 8 | 137 | 0 |
| DddB DAF | ind, chr3L:15,036,750-15,053,050 | 14,600 | 8 | 319 | 0 |
| DddA DAF | napa, radial-recalled subset | 3,202 | 8 | 184 | 0 |
| DddA DAF | uba1, radial-recalled subset | 6,340 | 8 | 837 | 0 |
| DddA DAF | napa, legacy callset (first 10,000) | 10,000 | 8 | 919 | 545 |
| Hia5 Nanopore | pooled near-timepoint sna BAMs | 3,665 | 8 | 16 | 0 |

## Findings

1. **The Nanopore model no longer assumes long reads.**  In the 17.6 kb sna
   region, only 13 of 3,665 mapped reads span the whole interval.  Median aligned
   span is 1,340 bp.  The eight focal sites nevertheless have 192--243 reads per
   strand because every site's prior is fit independently.  The pooled run makes
   16 conservative MSP rescues and no nuc breakups at this arbitrary locus.
2. **DddB primarily supports missing-TF rescue.**  At `ind`, nine nuc-break
   candidates appear with a neutral nuc prior, four survive `10x`, and none
   survive `100x`.  Sparse DddB gap evidence therefore leaves the strong
   `>90 bp` nucleosome state intact while still rescuing focal TFs in MSPs.
3. **The DddA radial likelihood is discriminating, not merely permissive.**
   In the radial-recalled napa and uba1 subsets, 2,316 and 9,819 eligible
   nuc/site events, respectively, produce zero nuc breakups.  The older napa
   callset yields 545 at `100x`; stored examples remain strongly positive after
   marginalizing dyad uncertainty (17--40 nats).  This is promising, but the
   legacy and radial-recalled files are not a paired random sample, so it is not
   yet a formal concordance estimate.
4. **Arbitrary TF subsets work.**  The state space includes accessible gaps and
   every non-overlapping subset of focal sites.  The development reports include
   two-TF-plus-gap rescues; nominated loci are needed to exercise three-TF and
   mixed nuc/TF cases deliberately.

## Calibration requirements

- Run investigator-nominated positive and negative Fiber-seq loci.
- Replace heuristic focal-site discovery with an opportunity-matched
  site-existence Bayes factor, learned on the source population only.
- Add duplicate-family or effective-sample-size control for amplified DAF; raw
  amplicon depth must not become arbitrarily strong prior odds.
- Calibrate posterior and nuc-prior thresholds with shifted-site decoys,
  held-out strands or BAMs, hard-call thinning, and source-read bootstrap
  intervals.
- Learn pairwise occupancy only for site pairs with spanning reads; retain the
  factorized prior for short-read Nanopore data.
- Model long DddA calls with explicit `N+N`, `N+TF`, and multi-N alternatives
  rather than extending a single radial profile beyond 220 bp.

## Boundary-focused PacBio validation

Five indexed, post-TF/post-nuc 2--4 hr PacBio Fiber-seq BAMs in
`/mnt/g/v3seg_mp` were pooled virtually with repeated `-i`; no BAM was merged or
rewritten.  The element intervals use dm6 coordinates.  Homie is based on the
[experimentally cloned element](https://pmc.ncbi.nlm.nih.gov/articles/PMC6119122/),
Nhomie on the [endogenous deletion interval](https://pmc.ncbi.nlm.nih.gov/articles/PMC13240896/),
and SF1/SF2 on the [dm6 knockout intervals](https://pmc.ncbi.nlm.nih.gov/articles/PMC9106302/).
This avoids accidentally applying the older ~2.68 Mb dm3 SF1/SF2 coordinates
to dm6 BAMs.

The first PacBio implementation incorrectly split reads by forward/reverse
alignment flag. Those proposal counts are invalid and were discarded. PacBio
HiFi Fiber-seq has both biochemical strands on every molecule: the BAMs contain
`A+a` and `T-a` MM blocks on the same read. The corrected model combines both
channels per read, pools every read into one `BOTH` population, and fits the
joint local configuration distribution without using alignment orientation as
evidence.

The corrected pass used the exact dm6 element intervals and required at least
50 high-TQ calls per pooled focal template. `MSP→TF` and strict `nuc→TF` use a
0.95 posterior threshold and a positive molecule-level Bayes factor. The review
tier instead requires a TF-tiling-versus-N log BF within +/-2 nats and empirical
TF-tiling:N prior odds of at least 0.5.

| Boundary | Duplex reads | Templates | MSP→TF | Strict nuc→TF (1× / 10× / 100× N prior) | Nuc review |
|---|---:|---:|---:|---:|---:|
| Nhomie | 953 | 5 | 82 | 0 / 0 / 0 | 2 |
| Homie | 1,061 | 3 | 303 | 0 / 0 / 0 | 8 |
| SF1 | 1,304 | 5 | 192 | 0 / 0 / 0 | 0 |
| SF2 | 1,341 | 6 | 233 | 3 / 1 / 0 | 3 |

The Homie two-site core is the clearest fused-call example. Among 1,029
spanning duplex molecules, the fitted joint weights are 0.110 accessible,
0.070 TF1, 0.192 TF2, 0.321 TF1+TF2, and 0.307 nucleosome. Thus the population
prior gives the paired-TF and N explanations nearly equal mass. Eight current
90--103 bp nuc calls are likelihood-ambiguous. The top three have neutral-prior
paired-TF posteriors of 0.57--0.78. Their full calls contain 62--63 combined A/T
opportunities, while only 7--9 lie outside the proposed TF pair; those few
outside bases are the only molecule-level evidence distinguishing two adjacent
TFs from one continuous nucleosome.

Nhomie supplies two analogous reviews, including one two-TF tiling with
posterior 0.70 and only eight outside-TF opportunities. SF2 supplies three
three-site/two-site reviews and three strict two-TF resegmentations under the
neutral prior. Only one strict SF2 call survives a 10× N multiplier and none
survive 100×, which is the expected sensitivity to the strong `>90 bp` N prior.
SF1 has abundant focal TF consensus but no current nuc whose hard calls and
population prior jointly make a TF tiling competitive.

A corrected leave-one-BAM-out check used four BAMs for site discovery/prior
learning and the fifth only for scoring, across all five rotations:

| Boundary | Held-out reads | Held-out MSP→TF | MSP-positive folds | Strict nuc→TF (1× / 10× / 100×) | Nuc reviews (positive folds) |
|---|---:|---:|---:|---:|---:|
| Nhomie | 953 | 112 | 5/5 | 0 / 0 / 0 | 3 (2/5) |
| Homie | 1,061 | 295 | 5/5 | 0 / 0 / 0 | 11 (4/5) |
| SF1 | 1,304 | 186 | 5/5 | 0 / 0 / 0 | 0 (0/5) |
| SF2 | 1,341 | 249 | 5/5 | 3 / 1 / 0 | 3 (3/5) |

The central templates reproduce across folds, although a few lower-support
satellite templates enter or leave the automatic site set. These remain a
deconvolution proof of principle rather than calibrated production calls. A
final validation should hold out genomic loci, add shifted-site decoys, and
estimate an effective population prior rather than treating all deeply sampled
molecules as independent.
