# Associating features

`fiberhmm.assoc` tests whether two kinds of features go together on the same molecules or at the same loci. It
works on what `fiberhmm-scan` writes, or on plain arrays from the Python API. It never reads BAM records itself.

There are four modes.

| Mode | Question | Unit | Test |
|---|---|---|---|
| M1 `cooccurrence` | On one molecule, does B's state follow A's state? | a pair of hits (within 5 kb) × the molecules spanning both | per pair: the exact conditional test; pooled: a pair-stratified score adjusted for mark density, openness and channel, with variance clustered by molecule |
| M2 `condition` | Given A's state at an anchor, does a molecule metric differ? | an anchor × the molecules spanning it | Δ_anchor by permuting states within locus × density stratum; Δ_site (anchors against matched backgrounds) by permuting which locus of each matched set plays the anchor |
| M3 `rate_ratio` | Is B denser inside A than outside, on the same molecule? | a molecule (inside versus outside A) | conditional binomial rate ratio; circular shifts of A on the molecule's admissible positions |
| M4 `overlap_enrichment` | Are B hits nearer A hits than chance? | a hit | relocation of A's clusters within coverage × site-density strata; a max-statistic across distances |

Every mode reports an effect with an interval and a p-value, and counts what it left out.

## From a scan directory

M1–M3 read the per-molecule tables (`states`, `marks`, `molecules`). A scan writes them in table mode
(`--no-annotate`; the default for BED-scoped scans). M4 needs only hits and the callable territory.

```bash
fiberhmm-scan --bam a.bam --bed units.bed --modules fp,nfr,nuc,peaks --no-annotate -o scan/

# M1: NFR state against footprint-class binding on the same molecules
fiberhmm-scan assoc scan/ --mode cooccur --a nfr.region --b fp.class -o assoc_m1/

# M2: does an NFR's state shift the molecule's repeat length? Anchors in an enhancer BED,
# other NFRs as matched backgrounds
fiberhmm-scan assoc scan/ --mode condition --a nfr.region --response nuc:nrl --bed enhancers.bed -o assoc_m2/

# M3: are regular arrays depleted of TF footprints?
fiberhmm-scan assoc scan/ --mode rate --a nuc.seg.regular --b tf -o assoc_m3/

# M4: do peaks sit near NFRs? (both directions are reported)
fiberhmm-scan assoc scan/ --mode overlap --a nfr.region --b peaks.peak -o assoc_m4/
```

Each run writes `result.json`. It also writes `pairs.tsv` (M1) or `anchors.tsv` (M2) when the mode has per-pair
or per-anchor rows.

Groups: `--group dataset`, or `--group haplotype`, which takes the haplotype from the input BAMs' `HP` tag. A
group contrast is an interaction test: the difference of the groups' effects, with a joint bootstrap. Unphased
molecules are left out of haplotype contrasts and reported as their own group.

## From Python

```python
from fiberhmm.assoc import cooccurrence, condition, rate_ratio, overlap_enrichment
from fiberhmm.nuc import condition_stats, association_stats   # nucleosome analyses

# phasing at anchors, open versus closed molecules, against matched random backgrounds
res = condition_stats(molecules, anchors, cal=cal, frames=('anchor', 'molecule'))
```

`condition_stats` conditions nucleosome phasing on the state at each anchor. The state is open (an MSP of at
least 80 bp covers the anchor ± 30 bp) or closed (a nucleosome covers the anchor). It reports the phasing
resultant in whole-period bins, in the anchor frame and in the molecule's own barrier frame. Anchors found from the
same molecules (`pooled_msp_anchors`) should go through `held_out`, which discovers anchors on one half of the
molecules, tests them on the other half, and cross-fits the two directions.

`association_stats` measures regular arrays (or another segment kind) against TF footprints on each molecule's
informative sites.

## What the tests control

- **Shared molecules.** One long molecule spans many pairs and anchors. Variances are clustered by physical
  molecule. Intervals come from a bootstrap over locus groups (loci that share molecules). When long molecules
  chain most loci into one group, the variance becomes the two-way (locus × molecule) clustered variance.
- **Mark density.** Sparse molecules look closed and unbound. M1 adjusts for density and openness. M2 balances
  the two states within 12 density strata at each locus. M3 compares within molecules.
- **Self-seeding.** Conditioning on what sits at an anchor phases its own flanks, at real and random sites alike.
  Only Δ_site, against matched backgrounds with the same state rule, says whether an anchor is special.
- **Selection.** Anchors discovered from the tested molecules use held-out discovery.

Each mode's main test was checked on simulated nulls with these confounders built in, with 2,000 replicates per
test: the false-positive rate at 0.05 stayed within 4–6%. M2 group contrasts use
the joint two-way block-bootstrap covariance, whole matched-set × molecule
intersections, and group-specific small-sample inference; their dataset and
haplotype null FPRs are 4.93% and 5.06% in a 1,200-replicate rerun each. See the
[calibration report](assoc_m2_calibration/README.md) for support counts, KS tests,
power, and main-test identity checks. M4 group contrasts remain conservative.


Association analyses accept `cancel=lambda: should_stop` and optional
`progress(stage, completed, total)` callbacks. Cancellation is checked at
resampling batch boundaries and raises `fiberhmm.assoc.AssociationCancelled`;
callback state is local to the current thread/task and nested calls inherit it.
Callbacks do not change random streams or statistics.

For dataset comparisons with different nucleosome calibration, call
`condition_stats(..., cal_by_dataset=calibrations, dataset_of=selector,
group='dataset')`. Each molecule uses its own dataset's period, prior, size
classes and low-information floor. Corresponding whole-period bins are
compared; `calibration_by_dataset` lists their physical spans and digests.
Oscillation metrics require equal periods. M1 pooled summaries also contain
`a`, `b`, `c`, `d` counts. M3's `scanio.rate_data_from_marks` accepts
`b_scan=other_scan` for states in another scan of the same reference/frame.
For scanio windows, `None` means all and `[]` means none. Annotated-scan CLI
input and FASTA-derived density strata remain open.
