# Strand-rescue v4 validation record

> Historical snapshot: this document validates the v4 production artifacts and
> BAM round-trip contract. The current v6 development model uses an
> exact-family action-set `q0` and posterior-predictive edge evidence; see
> `STRAND_RESCUE_V6_DEVELOPMENT.md`. Values below must not be presented as v6
> calibration results.

Status: final implementation, packaging, and real-data validation, 2026-07-15

Report schema: `fiberhmm.strand_rescue.v4`

Audit schema: `fiberhmm.strand_rescue.audit.v4`

Artifact root: `consensus_validation_outputs/strand_rescue_v4/`

The persistent absolute artifact root is:

```text
/mnt/g/Dropbox/Fiber-NET-seq/FiberHMM v1.0/Release v2.0.0/consensus_validation_outputs/strand_rescue_v4/
```

It is on the Windows-mounted Dropbox drive, not in WSL `/tmp`.

This record validates strand rescue (SR) as a focal, chemistry-aware secondary
pass with three outputs:

1. weak-but-positive MSP-to-TF recovery from opposite-strand population
   support;
2. one-for-one shared-edge normalization of already accepted TF calls; and
3. one-for-one shared-edge normalization of already accepted nucleosome calls.

SR never promotes a TF from a nucleosome and never creates, removes, splits,
merges, promotes, demotes, or reclassifies a nucleosome. Ordinary `nuc`, `msp`,
and `tf` annotations remain unchanged. The optional BAM output adds complete
`nuc_sr.QQQ` and `tf_sr.QQQ` shadow callsets. No nucleosome-length ceiling is
imposed.

The v4 change is deliberately narrow: it replaces the overloaded v3
`QQQQQ` rows with one connected alternative-hypothesis probability and two
boundary confidences. It does not broaden SR into consensus reconstruction or
fused-nucleosome deconvolution.

The reported probabilities are normalized model probabilities, not held-out
calibrated probabilities of biological truth. The selected cases test software
behavior, exact materialization, and the aggressive result surface. They do not
define an optimal browser threshold or establish genome-wide feasibility.

## Inference guardrails exercised

- Every repeated `-i/--bam` is part of one explicitly pooled, same-assay
  inference cohort. BAM identity is retained for exact output routing and
  per-input amplified-DAF collapse, not as a separate prior.
- No other library or assay supplies an inference prior. Cross-assay and
  PacBio Fiber-seq data are evaluation-only.
- PacBio Hia5 is not an SR preset. A HiFi read already exposes both A and T
  channels, and alignment orientation is not a biochemical strand.
- SR reads only standard aligned sequence, hard DAF mismatches/native calls,
  hard Dorado `MM`/`ML`, and post-TF/post-nuc `MA` annotations. It never reads
  IPD, pulse, raw-current, or sub-threshold modification evidence.
- Ordinary TF calls are source observations without a second TQ gate.
- The source direction is normalized by accepted-call fraction among molecules
  fully spanning the complete family, not by raw counts or center-only depth.
  Center-only coverage is retained only as a diagnostic.
- Short reads contribute wherever they fully map a footprint. They need not
  span the requested locus or a complete multi-footprint configuration.
- Amplified DddA/DddB population support is PCR-family collapsed independently
  within each input BAM. Every raw alignment remains a target and remains in
  the regional output.
- Each rescued TF component must be fully contained by an ordinary MSP, be
  fully mapped, contain at least one informative opportunity, and have a
  strictly positive protected-versus-accessible LLR.
- Any ordinary TF or nuc annotation with less than 80% mapped molecular span is
  excluded from geometry learning and harmonization but retained as a
  topology-only obstacle. It can block a rescue or edge expansion and cannot
  disappear merely because it is mostly soft-clipped.
- Existing TF/nuc state identity is fixed during edge scoring. Only bases added
  to or removed from an edge contribute to its target-molecule chemistry Bayes
  factor.
- An explicit geometry null, one-call/one-family assignment, endpoint mapping,
  and joint TF/nuc topology checks prevent forced or newly overlapping edge
  updates. Existing baseline overlaps are grandfathered. An edge expansion
  yields to an otherwise valid rescued TF.

## The v4 connected-layer quality contract

The annotator emits complete MA-spec-valid custom layers:

```text
nuc_sr.QQQ
tf_sr.QQQ
```

Every byte is a linearly quantized unit-interval model score. `q0` is the
alternative-hypothesis probability, while `q1` and `q2` are boundary
confidence scores:

```text
Q(p) = clamp(round(255 * p), 0, 255)
```

They are not Phred scores, ordinary TF `tq`, or nucleosome `nq`.

| byte | meaning for a named `Rn` rescue | meaning for a named `H` edge alternative |
|---|---|---|
| `q0` | probability of the exact selected TF configuration versus the ordinary MSP-accessible hypothesis | probability of the canonical SR interval versus the source ordinary TF/nuc interval |
| `q1` | confidence in the selected component's molecular-left boundary | marginal probability of the canonical molecular-left boundary versus the ordinary boundary |
| `q2` | confidence in the selected component's molecular-right boundary | marginal probability of the canonical molecular-right boundary versus the ordinary boundary |

The BAM stores edges in the molecular coordinate frame. Consequently, the
annotator swaps reference-left and reference-right confidence values on a
reverse alignment so `q1` always describes molecular left and `q2` molecular
right.

Unpaired unchanged shadow calls receive the fixed sentinel row `(255,0,0)`.
That row means only “this complete shadow entry is the ordinary baseline”; it
is not evidence favoring an alternative. A threshold must therefore be applied
to named `H` and `Rn` groups only.

### `Rn`: exact rescued-configuration probability

For an ordinary MSP, the model enumerates the accessible state `A` and every
nonoverlapping TF subset supported by the local target molecule. It combines
the opposite-strand state-model prior with the target molecule's hard-call
likelihood. If `C*` is the highest-scoring supported TF configuration, then:

```text
log_odds_R = log prior(C*) + log L(target | C*)
             - log prior(A) - log L(target | A)
P_R = logistic(log_odds_R)
q0 = Q(P_R)
```

Thus `q0` compares the exact selected configuration with the ordinary MSP
hypothesis. It is not the posterior mass summed over all possible TF
configurations. The latter remains in report JSON as
`supported_tf_probability_vs_accessible` for diagnosis.

All components of a multi-TF `R0`, `R1`, ... decision share the same `q0` and
must switch atomically. Each component retains its own `q1` and `q2`. For a
boundary `e`, the current canonical-geometry confidence is:

```text
support_rel = min(1, log(1 + total_family_support) / log(101))
strand_rel  = 1.0 when both strands have robust geometry, otherwise 0.75
edge_rel(e) = support_rel
              * exp(-MAD_e / 12)
              * strand_rel
              * exp(-strand_disagreement_e / 24)
```

The default boundary-reliability scale is 12 bp. `q1` and `q2` are
`Q(edge_rel)` for that rescued component's left and right canonical edges.

### `H`: canonical-edge probability against the ordinary source call

An accepted ordinary TF or nuc is assigned one-for-one to a same-class geometry
family. For a target-strand call with ordinary edge vector `mu_B`, canonical
edge vector `mu_SR`, and molecule-collapsed opposite-strand edge samples `x_i`,
the population term uses robust marginal scales and a regularized bivariate
covariance:

```text
scale_e = max(2, 1.4826 * MAD_e)
corr    = clamp(Pearson(start, end), -0.9, 0.9)
Sigma   = [[scale_left^2, corr * scale_left * scale_right],
           [corr * scale_left * scale_right, scale_right^2]]

log BF_population = 1/2 * sum_i [
    (x_i - mu_B)'  Sigma^-1 (x_i - mu_B)
  - (x_i - mu_SR)' Sigma^-1 (x_i - mu_SR)
]
```

The target chemistry term is the hard-call log Bayes factor from only the bases
whose membership changes when moving the ordinary edge to the canonical edge.
Under equal prior odds:

```text
log BF_H = log BF_population + log BF_changed_target_chemistry
P_H      = logistic(log BF_H)
q0       = Q(P_H)
```

`q1` and `q2` use the analogous univariate population and per-edge chemistry
Bayes factors. If an edge is unchanged between the two hypotheses, it
contributes no Bayes factor and its byte is fixed at 255 because the hypotheses
agree on that boundary. With fewer than three opposite-strand samples, the
correlation is fixed at zero. With no samples, the population sum is exactly
zero and the finite score reduces to target-molecule chemistry.

`H` acceptance itself remains intentionally aggressive: assignment must beat
the explicit null, the canonical interval must project, and topology must be
safe, but a favorable `q0` is not an emission gate. This preserves the complete
alternative surface for a browser slider.

### FiberBrowser switching semantics

For a threshold `T`, display the named SR alternative exactly when `q0 >= T`:

- a hidden `Rn` group falls back to the ordinary MSP already present in the
  baseline layers;
- a hidden `H` group falls back to the exact source ordinary TF or nuc encoded
  by its source annotation ordinal; and
- an unpaired `(255,0,0)` shadow row is never interpreted as a named
  alternative.

Ordinary annotations preserve the baseline coordinates, while the named
`tf_sr`/`nuc_sr` annotation preserves the alternative coordinates. This keeps
both edge hypotheses without attempting to encode coordinates in AQ bytes.
`tf_sr` and `nuc_sr` are complete same-class shadow layers, not complementary
TF-versus-nucleosome hypotheses, so a slider must not transfer population
between them.

## Real-data cohorts

Four cases cover all three supported presets. `raw reads` are evidence records
passing SR input filters; `molecules` are population representatives after the
configured collapse. Regional output record counts are reported separately
below because the overlay preserves every fetched BAM record, including records
that did not enter inference.

| case | assay | focal region | BAMs | raw reads | molecules | collapsed duplicates | TF families | nuc families |
|---|---|---|---:|---:|---:|---:|---:|---:|
| GM12878 UBA1 | DddA DAF | `chrX:47194800-47195020` | 1 | 6,242 | 4,999 | 1,243 | 1 | 5 |
| GM12878 NAPA | DddA DAF | `chr19:47515400-47515620` | 1 | 25,623 | 16,276 | 9,347 | 1 | 2 |
| fly ind | DddB DAF | `chr3L:15039880-15040260` | 6 | 22,057 | 21,663 | 394 | 2 | 12 |
| fly ind | Nanopore Hia5 | `chr3L:15042180-15042440` | 2 | 1,158 | 1,158 | 0 | 2 | 6 |
| **total** |  |  | **10** | **55,080** | **44,096** | **10,984** | **6** | **25** |

The DddB run pools six compatible time-course BAMs. The Nanopore run pools two
compatible time points. DddA and DddB collapse is performed separately within
each BAM; it never crosses an independently amplified input. The Nanopore
preset uses the requested hard `ML >= 248` threshold and does not collapse
reads.

## MSP-to-TF recovery surface

Every locally supported decision is retained, including `retain_current`.
Tiers summarize the unrounded v4 `q0`; they are not report-generation gates.
Counts such as `inside nuc` and `TF blocked` are site/read opportunities and are
not unique read counts.

| case | MSP groups | locally unsupported | source sites inside nuc | existing-TF overlaps blocked | not contained | not fully mapped | R decisions/components | strong / review / retain |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DddA UBA1 | 760 | 749 | 1,355 | 483 | 25 | 5 | 11 / 11 | 11 / 0 / 0 |
| DddA NAPA | 3,178 | 3,069 | 6,565 | 940 | 0 | 67 | 109 / 109 | 22 / 85 / 2 |
| DddB | 754 | 559 | 12,008 | 863 | 9 | 50 | 195 / 195 | 22 / 173 / 0 |
| Nanopore Hia5 | 29 | 28 | 367 | 73 | 1 | 2 | 1 / 1 | 0 / 0 / 1 |
| **total** | **4,721** | **4,405** | **20,295** | **2,359** | **35** | **124** | **316 / 316** | **55 / 258 / 3** |

All 316 validation decisions contain one TF component, so decision and
component counts are equal in this panel. Atomic multi-TF configurations remain
part of the implementation and focused tests; no selected real-data decision
needed more than one component.

## Existing-call edge normalization

The edge tables distinguish the full loaded call inventory from calls matched
to a focal geometry family. `eligible / topology-only` is counted across the
loaded raw target records. `H` is an accepted one-for-one edge alternative.
`unassigned` includes explicit-null or joint-assignment retention. `direct
topology` is a conflict with an ordinary call on the same molecule. `joint`
rejects a newly overlapping or order-inverting component across proposed TF/nuc
updates. `rescue` means the edge retained baseline coordinates so an `R` call
could remain available.

| case | type | families | eligible / topology-only | matched | H | unassigned | not spanned | direct topology | joint / rescue |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DddA UBA1 | TF | 1 | 80,891 / 29 | 937 | 492 | 430 | 2 | 13 | 0 / 0 |
| DddA UBA1 | nuc | 5 | 110,480 / 68 | 2,619 | 2,231 | 25 | 19 | 344 | 0 / 0 |
| DddA NAPA | TF | 1 | 155,907 / 77 | 3,544 | 3,167 | 293 | 82 | 2 | 0 / 0 |
| DddA NAPA | nuc | 2 | 258,405 / 636 | 7,256 | 4,269 | 2,369 | 8 | 609 | 0 / 0 |
| DddB | TF | 2 | 142,265 / 893 | 1,305 | 1,169 | 3 | 54 | 79 | 0 / 0 |
| DddB | nuc | 12 | 366,104 / 3,028 | 14,587 | 8,577 | 3,114 | 522 | 2,333 | 32 / 9 |
| Nanopore Hia5 | TF | 2 | 3,849 / 42 | 36 | 17 | 17 | 1 | 1 | 0 / 0 |
| Nanopore Hia5 | nuc | 6 | 20,437 / 463 | 335 | 277 | 37 | 15 | 6 | 0 / 0 |
| **total** | **TF** | **6** | **382,912 / 1,041** | **5,822** | **4,845** | **743** | **139** | **95** | **0 / 0** |
| **total** | **nuc** | **25** | **755,426 / 4,195** | **24,797** | **15,354** | **5,545** | **564** | **3,292** | **32 / 9** |

One NAPA nucleosome was already exactly canonical. Every other matched call is
accounted for by the displayed outcomes. No case had an
`insufficient_shared_geometry` or `insufficient_opposite_geometry_samples`
match.

## Observed v4 Q distributions

The following bins are counts of named alternatives in the emitted BAM AQ
rows, after molecular-orientation edge swapping. They are byte bins, not bins
of unrounded probabilities.

### `q0` by case and role

| case | role | total | 0-63 | 64-127 | 128-191 | 192-254 | 255 |
|---|---|---:|---:|---:|---:|---:|---:|
| DddA UBA1 | H | 2,723 | 313 | 11 | 5 | 22 | 2,372 |
| DddA UBA1 | R | 11 | 0 | 0 | 0 | 0 | 11 |
| DddA NAPA | H | 7,436 | 486 | 4 | 1 | 10 | 6,935 |
| DddA NAPA | R | 109 | 0 | 2 | 20 | 87 | 0 |
| DddB | H | 9,746 | 1,061 | 5 | 23 | 321 | 8,336 |
| DddB | R | 195 | 0 | 0 | 19 | 176 | 0 |
| Nanopore Hia5 | H | 294 | 18 | 4 | 2 | 36 | 234 |
| Nanopore Hia5 | R | 1 | 1 | 0 | 0 | 0 | 0 |
| **total** | **H** | **20,199** | **1,878** | **24** | **31** | **389** | **17,877** |
| **total** | **R** | **316** | **1** | **2** | **39** | **263** | **11** |

The exact aggregate threshold states recorded by the audits are:

| threshold `T` | H: SR edges | H: ordinary edges | R: TF | R: ordinary MSP |
|---:|---:|---:|---:|---:|
| 0 | 20,199 | 0 | 316 | 0 |
| 64 | 18,321 | 1,878 | 315 | 1 |
| 128 | 18,297 | 1,902 | 313 | 3 |
| 192 | 18,266 | 1,933 | 274 | 42 |
| 255 | 17,877 | 2,322 | 11 | 305 |

The large number of saturated H scores is expected from deep,
molecule-collapsed opposite-strand populations: independent supporting edge
samples accumulate in the population log Bayes factor. A byte of 255 means the
probability rounded to 255, not that biological truth has been proven. This is
one reason calibration and threshold selection must use held-out loci.

### Aggregate boundary bytes

| role / byte | total | 0-63 | 64-127 | 128-191 | 192-254 | 255 |
|---|---:|---:|---:|---:|---:|---:|
| H `q1` | 20,199 | 2,895 | 115 | 396 | 988 | 15,805 |
| H `q2` | 20,199 | 3,674 | 165 | 141 | 511 | 15,708 |
| R `q1` | 316 | 77 | 119 | 113 | 7 | 0 |
| R `q2` | 316 | 78 | 118 | 7 | 113 | 0 |

For H calls, a 255 boundary byte often means that boundary is unchanged and
shared by both hypotheses. It must not be interpreted as independent evidence
that the other boundary is correct.

## Exact materialization and BAM audits

For a v4 report, every selected R decision and every accepted H decision must
map back to the exact source record and annotation. Identity includes canonical
input path, query name, reference start, flag, CIGAR, SHA-256 of the original
SAM record, occurrence index for byte-identical duplicate records, molecular
interval, and source annotation ordinal.

The annotator stages every BAM and BAI for the selected cohort. It publishes
the cohort only after every report decision materializes exactly and every
index is built. If any BAM fails, no staged cohort is published. If publication
fails after existing outputs were moved aside, all previous BAMs/BAIs are
restored. Thus a multi-BAM report cannot leave a partly updated cohort.

All four report cohorts materialized with zero unmatched R or H decisions:

| case | BAMs | R expected/matched | TF H expected/matched | nuc H expected/matched | `nuc_sr` | `tf_sr` | audit |
|---|---:|---:|---:|---:|---:|---:|---|
| DddA UBA1 | 1 | 11 / 11 | 492 / 492 | 2,231 / 2,231 | 111,115 | 81,057 | valid |
| DddA NAPA | 1 | 109 / 109 | 3,167 / 3,167 | 4,269 / 4,269 | 259,268 | 156,254 | valid |
| DddB | 6 | 195 / 195 | 1,169 / 1,169 | 8,577 / 8,577 | 375,282 | 146,634 | valid |
| Nanopore Hia5 | 2 | 1 / 1 | 17 / 17 | 277 / 277 | 21,974 | 4,188 | valid |
| **total** | **10** | **316 / 316** | **4,845 / 4,845** | **15,354 / 15,354** | **767,639** | **388,133** | **valid** |

The 767,639 `nuc_sr` annotations exactly equal the 767,639 ordinary nuc
annotations. The 388,133 `tf_sr` annotations equal 387,817 ordinary TFs plus
316 rescued components. Across 62,319 regional records, audits found 20,199 H
groups, 316 R groups, 1,135,257 fixed baseline shadow rows, zero errors, and
valid threshold states for every named group.

Each output passed BAM quickcheck, index access, the v4 `@CO` and `MA-TYPES`
declarations, exact MA/AQ/AN positional alignment, three-byte quality arity,
one-for-one ordinary-call representation including duplicate intervals, valid H
source ordinals, fixed nuc state, atomic R q0/roles, per-component R boundary
bytes, molecular edge orientation, shadow-order preservation, and overlap
checks.

## Source preservation

The persistent validator compares each source/output record pair over the
report's complete loaded region. It requires:

- byte-identical first 11 SAM fields;
- identical non-MA/AQ/AN tags, including hard modification evidence;
- identical ordinary MA intervals and strand/quality specifications;
- identical complete ordinary AQ rows; and
- identical ordinary AN names.

Only newly rebuilt `nuc_sr`/`tf_sr` MA/AQ/AN content may differ.

| case | files | records | ordinary nuc | ordinary MSP | ordinary TF | all ordinary annotations | result |
|---|---:|---:|---:|---:|---:|---:|---|
| DddA UBA1 | 1 | 6,283 | 111,115 | 90,725 | 81,046 | 282,886 | valid |
| DddA NAPA | 1 | 25,624 | 259,268 | 115,462 | 156,145 | 530,875 | valid |
| DddB | 6 | 29,048 | 375,282 | 397,952 | 146,439 | 919,673 | valid |
| Nanopore Hia5 | 2 | 1,364 | 21,974 | 22,893 | 4,187 | 49,054 | valid |
| **total** | **10** | **62,319** | **767,639** | **627,032** | **387,817** | **1,782,488** | **valid** |

The validator remains at:

```text
consensus_validation_outputs/strand_rescue_v3/validate_source_preservation.py
```

Its location and output schema retain the v3 name because the invariant it
checks is independent of SR quality-row version. Its SHA-256 is
`c21c4e9daea9ba607d8ea59e9995ba194675c5e1bc1de81fef9fba909039897b`.

## Model and report provenance

DddA uses separate chemistry models for TF rescue/TF edges and nucleosome
edges. The nucleosome model never adds a nucleosome occupancy alternative to an
MSP rescue. DddB and Nanopore Hia5 use one assay model for both edge types.

| preset/case | TF model | TF SHA-256 | nuc-edge model | nuc SHA-256 | hard threshold |
|---|---|---|---|---|---|
| DddA UBA1/NAPA | `models/ddda_TF.json` | `5e1f29ba6abbf7c1909f2566efbd4c3f82cf4ecf4aa5de2c96f0a51b48e85bfb` | `models/ddda_nuc.json` | `c9da3116b4148ba67a85fa5cf86edd31ea7e434fcea52794e65c953be0b86c4b` | DAF call; no ML threshold |
| DddB | `models/dddb_nanopore.json` | `c936f971e671f2350e4469323fd9cbe04214fc0a77184d1bf01a7e19d8d0d164` | same | same | DAF call; no ML threshold |
| Nanopore Hia5 | `models/hia5_nanopore.json` | `73bc6ddfe9dfecddb81a30d0a26a6a22e00119196cdc0a999663235144737286` | same | same | `ML >= 248` |

This table records the exact models used for the historical v5 validation
outputs below; it is not a current-model registry. The current DddA preset
resolves the packaged `fiberhmm/models/ddda_TF.json`. Its independently
promoted physical-duplex calibration and frozen nucleosome likelihood table
are documented by the paper-analysis promotion receipt and require a fresh SR
validation receipt before replacing these historical report hashes.

The final report hashes, also embedded in output `@CO` provenance, are:

| report | SHA-256 |
|---|---|
| `ind-ddda-uba1.json` | `8384c94f6d154afb972c51d53575e0d58476b13f07a35f41adad1572ee4986bb` |
| `ind-ddda-napa.json` | `55785ed33eb3bd3ea6e7822dc9ac655b02a594fcd7ac6549d28635f2a1860347` |
| `ind-dddb.json` | `98a6352ee1cc987ac01ad911faa055cad6d82bbd3e2986b429675737e345141b` |
| `ind-nanopore.json` | `01f32638d9ebe2b19efa704b29de37f5bef1778568be95fdaf06650160fcb5b4` |

The materialized BAM hashes are:

| output BAM | SHA-256 |
|---|---|
| `ind-ddda-uba1-bams/uba1_ddda_recall.strand-rescue.bam` | `622a78ca01c07f11aa1e48870f9936a4385c0fb489bf39d74a1367edf2856b26` |
| `ind-ddda-napa-bams/napa_recaller_TF.strand-rescue.bam` | `fc5884f9e095831764912e34b5db9a6284af56de368d946f6ec2127a08e37cd4` |
| `ind-dddb-bams/1.5-2_recalled.strand-rescue.bam` | `459b5383ec35ff423a3e045e31bd89c6e909a4b10588238344eeb96ddf4e7313` |
| `ind-dddb-bams/2-2.5_recalled.strand-rescue.bam` | `7cb64f98cdc5ada8724bd9c53490c8d3838283a52f3b74f2ddbc0c4490e9a2d3` |
| `ind-dddb-bams/2.5-3_recalled.strand-rescue.bam` | `bca543d4287b3b624453a47c359515ba2b7c4ff0299285079e1f30bd8deb291e` |
| `ind-dddb-bams/3-3.5_recalled.strand-rescue.bam` | `b2610ea2735309c6c55920fb2d5ec07888cc84c688c50c26af8c6be200b0df6e` |
| `ind-dddb-bams/3.5-4_recalled.strand-rescue.bam` | `a0d34cb46ab714bc49f26c494d7150edb3b649340907e64700c2d6eb4c9accb1` |
| `ind-dddb-bams/yw_2-4_recalled.strand-rescue.bam` | `11abc9ccc7316f43a1ae18dbbed08432f42fc925dd2f62e1b96539bb3253cba1` |
| `ind-nanopore-bams/siGAF_1.5-3hr.fiberhmm.thr248.strand-rescue.bam` | `b4f567fcba41a2421395a6f7bb71aaa1a4e59db0c101b90c73f11bb0f7a52a18` |
| `ind-nanopore-bams/siGAF_3-4.5hr.fiberhmm.thr248.strand-rescue.bam` | `655e23090bbd40aea8cbd8f0bb75547f1e063f8c7fc5488727f023c9c9ab80cf` |

The validation-record hashes are:

| case | `audit.json` SHA-256 | `source-preservation.json` SHA-256 |
|---|---|---|
| DddA UBA1 | `327d9a4ce886b8edb3f799c31890a42782dd63a3e80cca906728c1b4d9b44250` | `de4ae58008d535eac377bc510541fced11189e4c1da91240342cda1ac3049b3f` |
| DddA NAPA | `75be6709fa4b825670e27d6881a968969b9308334d74b176df243ef893663d1e` | `8ffc44c4aea25753ad647feba8f72cc5729fc03c9bedbf62133208aaa51fcb26` |
| DddB | `527115e2fa94e06ca0212492c16cc86143699bca3978776a712dc9988069f7e4` | `9a2489e7e1330cf55b5f83d5e0567e590a8e201390f077a0f0ca12fb4ca002cc` |
| Nanopore Hia5 | `993d7bb623ff79b51ec9835651cfc88f20c6158ac4d12530d32fa334ac243b63` | `72f44e24f19deb36c85b402c8984afcca093f16a29725df5c59b8efa461a0e88` |

Every report also retains canonical input paths, file size/mtime fingerprints,
all effective parameters, model hashes, per-input collapse diagnostics,
per-molecule efficiency calibration, geometry families, source state models,
and every molecule-level decision. The four `audit.json` and four
`source-preservation.json` files retain per-file detail and zero-error status.

## Reproduction commands

Run all commands from the FiberHMM release root. Repeated `-i` arguments in one
report command are one explicitly pooled same-assay population.

### 1. Generate the four reports

```bash
python -m fiberhmm.cli.strand_rescue \
  -i ddda_profile/uba1_ddda_recall.bam \
  --preset ddda \
  --region chrX:47194800-47195020 \
  --site 47194899-47194929 \
  --forced-sites-only \
  --min-support 100 \
  --strand-min-source-support 100 \
  --proposal-tsv consensus_validation_outputs/strand_rescue_v4/ind-ddda-uba1.tsv \
  -o consensus_validation_outputs/strand_rescue_v4/ind-ddda-uba1.json

python -m fiberhmm.cli.strand_rescue \
  -i ddda_nuc_output/napa_recaller_TF.bam \
  --preset ddda \
  --region chr19:47515400-47515620 \
  --site 47515501-47515514 \
  --forced-sites-only \
  --min-support 100 \
  --strand-min-source-support 100 \
  --proposal-tsv consensus_validation_outputs/strand_rescue_v4/ind-ddda-napa.tsv \
  -o consensus_validation_outputs/strand_rescue_v4/ind-ddda-napa.json

python -m fiberhmm.cli.strand_rescue \
  -i /mnt/g/Dropbox/Fiber-NET-seq/Drosophila_phase2/Datasets/DAF-seq/spacetime_updated/fp_update/1.5-2_recalled.bam \
  -i /mnt/g/Dropbox/Fiber-NET-seq/Drosophila_phase2/Datasets/DAF-seq/spacetime_updated/fp_update/2-2.5_recalled.bam \
  -i /mnt/g/Dropbox/Fiber-NET-seq/Drosophila_phase2/Datasets/DAF-seq/spacetime_updated/fp_update/2.5-3_recalled.bam \
  -i /mnt/g/Dropbox/Fiber-NET-seq/Drosophila_phase2/Datasets/DAF-seq/spacetime_updated/fp_update/3-3.5_recalled.bam \
  -i /mnt/g/Dropbox/Fiber-NET-seq/Drosophila_phase2/Datasets/DAF-seq/spacetime_updated/fp_update/3.5-4_recalled.bam \
  -i /mnt/g/Dropbox/Fiber-NET-seq/Drosophila_phase2/Datasets/DAF-seq/spacetime_updated/fp_update/yw_2-4_recalled.bam \
  --preset dddb \
  --region chr3L:15039880-15040260 \
  --site 15039948-15040022 \
  --site 15040154-15040235 \
  --forced-sites-only \
  --min-support 50 \
  --strand-min-source-support 50 \
  --proposal-tsv consensus_validation_outputs/strand_rescue_v4/ind-dddb.tsv \
  -o consensus_validation_outputs/strand_rescue_v4/ind-dddb.json

python -m fiberhmm.cli.strand_rescue \
  -i /mnt/g/Dropbox/Fiber-NET-seq/Drosophila_phase2/Datasets/Fiber-seq/2-point_timecourse/bam/siGAF_1.5-3hr.fiberhmm.thr248.bam \
  -i /mnt/g/Dropbox/Fiber-NET-seq/Drosophila_phase2/Datasets/Fiber-seq/2-point_timecourse/bam/siGAF_3-4.5hr.fiberhmm.thr248.bam \
  --preset hia5-nanopore \
  --region chr3L:15042180-15042440 \
  --site 15042227-15042243 \
  --site 15042377-15042396 \
  --forced-sites-only \
  --min-support 3 \
  --strand-min-source-support 3 \
  --proposal-tsv consensus_validation_outputs/strand_rescue_v4/ind-nanopore.tsv \
  -o consensus_validation_outputs/strand_rescue_v4/ind-nanopore.json
```

### 2. Materialize complete shadow layers

The default `--minimum-posterior 0` is intentional: it preserves strong,
review, and retain-current R decisions. Every input BAM recorded by each report
is selected when `-i` is omitted.

```bash
python -m fiberhmm.cli.strand_rescue_annotate \
  --report consensus_validation_outputs/strand_rescue_v4/ind-ddda-uba1.json \
  --output-dir consensus_validation_outputs/strand_rescue_v4/ind-ddda-uba1-bams \
  --io-threads 4

python -m fiberhmm.cli.strand_rescue_annotate \
  --report consensus_validation_outputs/strand_rescue_v4/ind-ddda-napa.json \
  --output-dir consensus_validation_outputs/strand_rescue_v4/ind-ddda-napa-bams \
  --io-threads 4

python -m fiberhmm.cli.strand_rescue_annotate \
  --report consensus_validation_outputs/strand_rescue_v4/ind-dddb.json \
  --output-dir consensus_validation_outputs/strand_rescue_v4/ind-dddb-bams \
  --io-threads 4

python -m fiberhmm.cli.strand_rescue_annotate \
  --report consensus_validation_outputs/strand_rescue_v4/ind-nanopore.json \
  --output-dir consensus_validation_outputs/strand_rescue_v4/ind-nanopore-bams \
  --io-threads 4
```

### 3. Audit the four materialized cohorts

```bash
python -m fiberhmm.cli.strand_rescue_audit \
  -i consensus_validation_outputs/strand_rescue_v4/ind-ddda-uba1-bams/uba1_ddda_recall.strand-rescue.bam \
  -o consensus_validation_outputs/strand_rescue_v4/ind-ddda-uba1-bams/audit.json

python -m fiberhmm.cli.strand_rescue_audit \
  -i consensus_validation_outputs/strand_rescue_v4/ind-ddda-napa-bams/napa_recaller_TF.strand-rescue.bam \
  -o consensus_validation_outputs/strand_rescue_v4/ind-ddda-napa-bams/audit.json

python -m fiberhmm.cli.strand_rescue_audit \
  -i consensus_validation_outputs/strand_rescue_v4/ind-dddb-bams/1.5-2_recalled.strand-rescue.bam \
  -i consensus_validation_outputs/strand_rescue_v4/ind-dddb-bams/2-2.5_recalled.strand-rescue.bam \
  -i consensus_validation_outputs/strand_rescue_v4/ind-dddb-bams/2.5-3_recalled.strand-rescue.bam \
  -i consensus_validation_outputs/strand_rescue_v4/ind-dddb-bams/3-3.5_recalled.strand-rescue.bam \
  -i consensus_validation_outputs/strand_rescue_v4/ind-dddb-bams/3.5-4_recalled.strand-rescue.bam \
  -i consensus_validation_outputs/strand_rescue_v4/ind-dddb-bams/yw_2-4_recalled.strand-rescue.bam \
  -o consensus_validation_outputs/strand_rescue_v4/ind-dddb-bams/audit.json

python -m fiberhmm.cli.strand_rescue_audit \
  -i consensus_validation_outputs/strand_rescue_v4/ind-nanopore-bams/siGAF_1.5-3hr.fiberhmm.thr248.strand-rescue.bam \
  -i consensus_validation_outputs/strand_rescue_v4/ind-nanopore-bams/siGAF_3-4.5hr.fiberhmm.thr248.strand-rescue.bam \
  -o consensus_validation_outputs/strand_rescue_v4/ind-nanopore-bams/audit.json
```

### 4. Verify source preservation

```bash
python consensus_validation_outputs/strand_rescue_v3/validate_source_preservation.py \
  --report consensus_validation_outputs/strand_rescue_v4/ind-ddda-uba1.json \
  --output-dir consensus_validation_outputs/strand_rescue_v4/ind-ddda-uba1-bams \
  -o consensus_validation_outputs/strand_rescue_v4/ind-ddda-uba1-bams/source-preservation.json

python consensus_validation_outputs/strand_rescue_v3/validate_source_preservation.py \
  --report consensus_validation_outputs/strand_rescue_v4/ind-ddda-napa.json \
  --output-dir consensus_validation_outputs/strand_rescue_v4/ind-ddda-napa-bams \
  -o consensus_validation_outputs/strand_rescue_v4/ind-ddda-napa-bams/source-preservation.json

python consensus_validation_outputs/strand_rescue_v3/validate_source_preservation.py \
  --report consensus_validation_outputs/strand_rescue_v4/ind-dddb.json \
  --output-dir consensus_validation_outputs/strand_rescue_v4/ind-dddb-bams \
  -o consensus_validation_outputs/strand_rescue_v4/ind-dddb-bams/source-preservation.json

python consensus_validation_outputs/strand_rescue_v3/validate_source_preservation.py \
  --report consensus_validation_outputs/strand_rescue_v4/ind-nanopore.json \
  --output-dir consensus_validation_outputs/strand_rescue_v4/ind-nanopore-bams \
  -o consensus_validation_outputs/strand_rescue_v4/ind-nanopore-bams/source-preservation.json
```

## Software and compatibility validation

The focused implementation/package suite was rerun against v4:

```bash
python -m pytest \
  tests/test_strand_rescue.py \
  tests/test_strand_rescue_overlay.py \
  tests/test_package_consistency.py
```

Result: **79 passed**.

The final full-suite result is: **619 passed, 3 skipped, 26 benchmark
deselected**.

The focused tests cover hard Nanopore threshold 248, absence of a PacBio
orientation preset, strand-balanced geometry, full-site rather than center-only
source normalization, short-read contribution, per-molecule efficiency,
MSP-only weak-positive rescue, exact selected-configuration q0, multi-TF
atomicity with per-component boundary bytes, nucleosome immutability, DddA model
separation, changed-edge Bayes factors, regularized bivariate opposite-strand
edge evidence, marginal edge scores, explicit-null assignment, no
nucleosome-length ceiling, soft-clipped topology-only obstacles, reverse
molecular projection and edge-byte swapping, duplicate qnames and
byte-identical record occurrences, duplicate ordinary intervals, source-ordinal
H roles, complete shadow cardinality, overlap/order checks, strict exact
materialization, cohort-wide staged publication/rollback, BAM indexing, v4
audit, v3 report conversion, and legacy-v2 header replacement.

The current v4 auditor was also run directly against retained real v2 and v3
UBA1 outputs:

| input contract | records | SR annotations | named groups | errors | result |
|---:|---:|---:|---:|---:|---|
| v2 | 6,283 | 81,054 | 732 | 0 | valid |
| v3 | 6,283 | 192,172 | 2,734 | 0 | valid |

This establishes read/audit compatibility for older BAM contracts; it does not
reinterpret their legacy five-byte values as v4 QQQ values. New materialization
always writes one v4 contract comment and v4 QQQ layers.

The final persistent wheel target is:

```text
consensus_validation_outputs/strand_rescue_v4/wheel-smoke/fiberhmm-2.16.3-py3-none-any.whl
```

Final wheel SHA-256:
`362c84ad29c625408ccdaf9ceabd9349b9b61147d7354cbf8a90547c5e8e6497`.

The isolated wheel smoke check loaded `strand_rescue.py` from the installed
venv `site-packages` rather than the source tree. It also ran `--help` through
all three installed console entry points with exit status zero:
`fiberhmm-strand-rescue`, `fiberhmm-strand-rescue-annotate`, and
`fiberhmm-strand-rescue-audit`.

## Scientific validation boundary

The real-data runs establish that SR v4 works from standard DAF output and hard
Dorado MM/ML output, that every reported decision round-trips into portable
MA-valid regional shadow layers, and that ordinary calls and hard evidence are
preserved exactly. They also demonstrate the intended aggressive secondary
surface: 316 MSP-to-TF alternatives and 20,199 same-class edge alternatives
remain available for thresholded visualization.

They do not establish that every aggressive H edge is biologically exact, that
the current probability bytes are empirically calibrated, that a single q0
threshold transfers across assays or depths, or that focal parameters are
suitable genome-wide. Deep populations can saturate a model Bayes factor while
remaining vulnerable to correlated biological or caller error.

Future scientific validation should freeze parameters and loci, select browser
thresholds on training loci, and evaluate held-out focal populations against
independent PacBio Fiber-seq, DAF, and Nanopore Hia5 evidence. Those other
assays must remain evaluation data rather than inference priors. Consensus
alternative decompositions of fused nucleosomes remain a separate
FiberBrowser-level hypothesis problem, not an SR occupancy operation.
