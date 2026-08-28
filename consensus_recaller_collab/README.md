# Focal consensus recaller

> **Archived research workspace (2026-07-14).** This directory is no longer a
> FiberHMM runtime package. CR moved to the annotation-only regional
> [FiberBrowser specification](../docs/FIBERBROWSER_CONSENSUS_RECONSTRUCTION.md),
> while chemistry-aware SR moved to `fiberhmm-strand-rescue`. See
> [ARCHIVE_NOTICE.md](./ARCHIVE_NOTICE.md). Command names and implementation
> guidance below are historical unless explicitly restated in those documents.

> **Read [`CHANGES_2026_07_14.md`](./CHANGES_2026_07_14.md) first**
> (evidence in [`CALIBRATION_AUDIT.md`](./CALIBRATION_AUDIT.md)). A 2026-07-14
> audit found that both passes systematically over-called and that the control
> panel could not detect it. All fixes are in; the proposal counts in
> [`PRODUCTION_VALIDATION.md`](./PRODUCTION_VALIDATION.md),
> [`RESULTS.md`](./RESULTS.md), [`REVISED_RESULTS.md`](./REVISED_RESULTS.md) and
> [`PAIRED_VALIDATION.md`](./PAIRED_VALIDATION.md) predate them and must not be
> cited. Two headline changes:
>
> - Molecule LLRs are now scaled by each molecule's own deamination efficiency
>   (`--global-deamination` restores the old behaviour). Only 15% of the
>   previously reported strong DddB strand rescues survive.
> - `N` and `TF` now use geometry priors fitted the same way from the same
>   source population, so a block with no evidence returns the prior instead of
>   a ~19x free Bayes factor toward splitting.

The former experimental inference implementation was `fiberhmm-consensus-recall`
([`revised_prototype.py`](./revised_prototype.py)). It deliberately exposes two
independent report-only passes:

- `strand_rescue`, for transferring site-level evidence between physical
  strands in DAF and Nanopore Hia5; and
- `composite_deconvolution`, for testing a nuc-like protected block against
  source-observed one-, two-, three-, or higher-order TF configurations.

Use `--skip-strand-rescue` or `--skip-composite-deconvolution` to run either
pass alone. The earlier `prototype.py` is retained as a development record, but
its PacBio A/TF/N EM mixture is non-identifiable when a continuous nucleosome
and a tiled TF configuration protect the same bases; its fitted PacBio N
fractions must not be interpreted as occupancy estimates. See
[`REVISED_RESULTS.md`](./REVISED_RESULTS.md) for the current model and tests.
The manifest-driven cross-assay hierarchy, authoritative full datasets, and
new A/TF/N site-existence validation are documented in
[`validation/README.md`](./validation/README.md) and
[`validation/INITIAL_RESULTS.md`](./validation/INITIAL_RESULTS.md).
The frozen operating point and full multi-control results are in
[`PRODUCTION_VALIDATION.md`](./PRODUCTION_VALIDATION.md).
The aggressive connected-layer implementation and eight-panel focal results
are in [`PAIRED_VALIDATION.md`](./PAIRED_VALIDATION.md).
The optional, non-destructive FiberBrowser handoff is
[`fiberhmm-consensus-annotate`](./VISUALIZATION.md); it writes regional
derivative BAMs with `nuc_cr`/`tf_cr` and `nuc_sr`/`tf_sr` MA posterior layers.
Its `--paired` mode emits the complete aggressive decision surface with linked,
complementary N/TF scores for one browser threshold; the exact implementation
contract is in [`PAIRED_RECALL_PLAN.md`](./PAIRED_RECALL_PLAN.md).
The retained PacBio/DddB/Nanopore/DddA panel can be regenerated without system
temporary storage using
[`run_paired_validation.sh`](./run_paired_validation.sh).

This directory retains the archived inference, validation, and visualization
implementation for a focal, population-informed recaller. The historical caller
does **not** rewrite BAM tags; JSON and optional TSV proposals remain the
authoritative inference output. The separate annotator never edits an input
BAM and only materializes those proposals in indexed regional derivatives.

The caller:

1. discovers focal TF templates from high-confidence `tf` MA annotations;
2. groups nearby, non-overlapping templates into small local windows;
3. represents a current call as accessible, one protected nucleosome, or any
   non-overlapping subset of its TF templates (one TF plus a gap, two TFs,
   three TFs, and so on);
4. fits `A`/`TF`/`N` occupancy from hard modification calls; and
5. uses the resulting focal population distribution as an empirical prior when
   revisiting weakly resolved molecules.

The observation/prior structure is chemistry-specific:

- DddA/DddB DAF and Nanopore Hia5 are stranded assays. A per-site model is fit
  on one biochemical/read strand and used when scoring the other.
- PacBio Fiber-seq is duplex at the molecule level. Every HiFi read contains
  both the `A+a` and `T-a` MM channels, so both strands contribute jointly on
  that read. Forward/reverse alignment flags are only reference orientation
  and are never used as separate evidence strands. All molecules are pooled
  into one `BOTH` group, and a joint prior is fit across complete local TF
  configurations plus the nucleosome state.

Reports include proposal counts by local TF configuration and the spanning-read
configuration distribution before and after rescue.  Each existing TF is
matched to one best non-overlapping template, so a broad template and its
overlapping split alternatives cannot be counted on the same molecule.

Existing calls are proposals only.  Their LLRs are not added to the raw
hard-call likelihood, which avoids double-counting.  A proposed TF must have at
least one informative opportunity and positive molecule-level evidence.
Population depth can promote weak positive evidence, but it cannot overturn a
target molecule whose hard calls favor its existing state.

Nucleosome resegmentation also has a separate review tier. It reports current
nucs for which the molecule-level TF-tiling-versus-N log Bayes factor lies in a
configurable ambiguity band and the empirical TF-tiling prior is within a
configurable odds ratio of N. Each record reports hard opportunities inside
the call, inside the proposed TFs, and outside the TFs. For PacBio these are
the combined A/T opportunities on that same duplex read; the outside-TF bases
are what actually distinguish a TF tiling from one continuous nucleosome.

Only standard BAM sequence, alignment, `MM`/`ML`, and FiberHMM `MA` annotations
are used. The caller does not use raw IPD, pulse features, or sub-threshold
`ML` values.  The Nanopore Hia5 preset uses `ML >= 248` by default.

## Stranded short-read handling

No source read is required to span a locus, an amplicon, a complete candidate
configuration, or even the requested flanking window.  A source prior is fit
separately for each focal TF site from reads spanning that site.  The bases of a
short read that lie in the local flanks contribute normally; missing flanks add
no likelihood term.  Priors for multi-TF configurations are then composed from
these per-site marginals.  This factorization is intentionally conservative for
short Nanopore Hia5 reads; a later model can add pairwise occupancy terms only
where actual reads span both sites.

This factorization is not used for PacBio. PacBio reads normally span each
small focal configuration, so the caller learns the full joint
configuration distribution directly from the pooled duplex molecules.

## Nucleosome likelihoods

- N marginalizes over an **empirical geometry prior** drawn from source
  molecules carrying an explicit nuc call over the same focal sites, using the
  same `-log(support)` normalization, the same Gaussian call-edge kernel, and
  the same leave-one-molecule-out holdout that every TF configuration uses.
  Both hypotheses therefore carry comparably sharp, data-fitted geometry
  priors, so a block with no informative bases returns the prior. Giving N a
  vague uniform prior over every possible dyad position while TF got a fitted
  one was worth ~2.9 nats (~19x) toward splitting on a block carrying no
  evidence at all, and displaced the whole N:TF axis; see
  [`CALIBRATION_AUDIT.md`](./CALIBRATION_AUDIT.md). The uniform enumeration is
  retained only as a fallback below `--min-nuc-geometries` (default 20); the
  path taken is reported per candidate in `integrated_nuc.geometry_prior`.
- Every molecule's LLRs are scaled by its own deamination efficiency, estimated
  from its own accessible (`msp`) calls and shrunk toward the model
  expectation for its context composition. A protected call is evidenced by the
  *absence* of deamination, so a single global accessible hit rate credits a
  poorly deaminated molecule with protection it never demonstrated. Use
  `--global-deamination` to restore the old, over-crediting behaviour.
- DddB and Hia5 use the ordinary context-aware protected-versus-accessible LLR
  over the exact current nuc interval.
- DddA uses the empirical radial deamination profile in
  `fiberhmm/models/ddda_nuc_profile.json`.  It marginalizes over nearby dyad
  positions, rather than assuming the current edges place the dyad exactly.
- In all chemistries, composite deconvolution has a hard 220 bp ceiling (and a
  90 bp lower bound). A lower ceiling may be requested, but a higher one is
  rejected. Focal blocks above the selected ceiling are counted in the report
  rather than silently treated as nucleosomes. Blocks above 220 bp must first
  be resolved by the upstream nuc recaller; this pass does not attempt to
  invent a weakly identified multi-nucleosome/TF tiling.
- The existing `>90 bp` nuc state receives a configurable prior multiplier
  (`1,10,100` in the development panel). At a focal site, the global TF prior
  can be raised only to the same-cohort Wilson lower bound on the fraction of
  spanning molecules with an explicit high-TQ configuration. Uncalled
  molecules remain in the denominator, the candidate molecule is held out of
  its own occupancy and configuration geometry, and short reads that do not
  span the template hull cannot vote. Input-BAM identity never partitions or
  supplies this prior.
- Composite recall is hierarchical. Stage 1 marginalizes over every supported
  TF layout and tests canonical N versus the aggregate TF-complex class. Stage
  2 emits an exact split only when the best layout has at least 0.8 posterior
  conditional on that class; otherwise it reports one unresolved TF-complex
  envelope. No short-length trigger is used within the 90–220 bp test range.
- MA calls with less than 80% of their molecular interval mapped to reference
  are excluded. This prevents a long call extending through a soft clip from
  masquerading as a short focal nucleosome after projection.

## Example inputs

The repository/workspace currently has examples of all intended chemistries:

- DddB targeted DAF: `nuc_recaller_collab/data/dddb_dev_ind_sna_gt.bam`
- DddA amplicon: `ddda_nuc_output/napa_recaller_TF.bam`
- Nanopore Hia5: `Drosophila_phase2/.../2-point_timecourse/bam/*.bam`
- PacBio Hia5: `nuc_recaller_collab/data/fiberseq_dev_full.bam`

The input must already contain post-nucleosome/post-TF `MA` annotations.  The
caller only writes JSON and optional TSV reports.

Repeat `-i` only to explicitly declare compatible BAM shards/timepoints as one
self-contained inference cohort, without creating a physical merged BAM. All
of those molecules contribute symmetrically; a BAM path is retained only for
output routing and as the stratum within which amplified-DAF PCR families are
collapsed. It is never a prior partition or a leave-one-library-out source.

`--target-bam` is limited to derivative BAMs containing a subset of the same
cohort molecule names, which is verified at runtime. Independent libraries or
assays belong in `fiberhmm-consensus-validate`; they never contribute focal
occupancy, TF configurations, boundary distributions, or candidate calls.

Run `fiberhmm-consensus-recall --help` for options.

Example:

```bash
fiberhmm-consensus-recall \
  -i library_shard_1.bam -i library_shard_2.bam \
  --preset hia5-pacbio \
  --region chr2L:5000000-5100000 \
  --min-support 5 \
  --max-auto-sites 8 \
  -o report.json
```

The automatic site finder is intentionally only a proposal mechanism.  The
current caller uses high-TQ support, boundary MAD, and local enrichment; a
candidate can also be nominated explicitly with repeatable `--site START-END`
arguments. `--forced-sites-only` prevents automatic sites from entering that
test. External geometry does not manufacture support: explicit source calls
and raw likelihoods are recomputed from the selected BAMs.

The validation pipeline implements a nested A/TF/N site-existence test,
full-read DAF duplicate-family collapse, five opportunity-matched local shifts,
deduplicated pseudo-site nulls, leave-one-locus calibration, and independent
library/assay replication. Those comparisons evaluate frozen calls; they do
not feed the inference cohort. Use `fiberhmm-consensus-validate --help`; see
[`PRODUCTION_VALIDATION.md`](./PRODUCTION_VALIDATION.md) for the frozen scope.
