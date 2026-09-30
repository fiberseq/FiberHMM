# Strand rescue

In DddA and DddB DAF-seq and in Nanopore Hia5 Fiber-seq, each read observes
one biochemical strand, so a footprint can be clear on reads of one strand and
missed on reads of the other. `fiberhmm-strand-rescue` is an optional,
focal (one region at a time) secondary caller that uses the population of the
opposite strand to:

1. recover a TF call missed on one strand, when the read still shows an MSP
   there, has weak but positive evidence of protection, and the footprint is
   recurrent on the opposite strand;
2. move accepted TF calls onto a strand-balanced shared geometry;
3. optionally (experimental, `--independent-nuc-edge-refinement`) move
   nucleosome edges onto a strand-balanced shared geometry.

It never rewrites the ordinary calls. It produces a JSON report, from which
`fiberhmm-strand-rescue-annotate` writes new BAMs with two complete **shadow
layers** (`nuc_sr`, `tf_sr`), so a viewer can switch between the ordinary
and rescued interpretation with a quality threshold.

Nucleosomes are never promoted, demoted, split, merged, created or removed;
every ordinary nucleosome has exactly one call in `nuc_sr`, and no
nucleosome-length ceiling applies. A TF is never promoted from a nucleosome.

## Run it

```bash
fiberhmm-strand-rescue -i out/dddb.calls.bam --preset dddb --region chrDemo:9500-10700 \
    -o out/sr.json --proposal-tsv out/sr.tsv
fiberhmm-strand-rescue-annotate --report out/sr.json --output-dir out/sr_bams
fiberhmm-strand-rescue-audit -i out/sr_bams/dddb.calls.strand-rescue.bam -o out/sr_bams/audit.json
```

The rescue command prints structured progress (one JSON line per stage) and
a JSON summary; the audit ends with `"valid": true` when the BAM satisfies
the contract below.

Inputs must be coordinate-sorted, indexed BAMs with called `MA` annotations.
Repeating `-i` pools shards or compatible timepoints of **one** assay into one
population; another assay or library must not be used as a prior (cross-assay
overlap is validation, not inference).

| `--preset` | Strand groups | Observations | TF model | Nucleosome-edge model | ML threshold |
|---|---|---|---|---|---|
| `ddda` | CT, GA | DAF deaminations | `ddda_TF.json` | `ddda_nuc.json` | — |
| `dddb` | CT, GA | DAF deaminations | `dddb_nanopore.json` | same | — |
| `hia5-nanopore` | FWD, REV | Dorado `MM`/`ML` m6A | `hia5_nanopore.json` | same | 248 |

`--model` and `--nuc-model` override the tables. PacBio Hia5 is not
supported: a HiFi read already observes both strands. Strand rescue reads
only sequence, hard `MM`/`ML` (or DAF) calls and `MA`; never kinetics or
sub-threshold probabilities.

## How it works

**Sites.** Every geometry-eligible ordinary `tf` call can support a recurrent
footprint, whatever its `tq`. Calls are projected to the reference, their
centres clustered separately per strand (`--center-radius`,
`--peak-distance`), matching clusters merged into one site, and a canonical
start/end taken as the median of the per-strand medians, so a deeply
sequenced strand cannot drag the geometry. A strand votes once it has
`--minimum-geometry-support` (3) calls. Calls whose alignment does not extend
past both edges by `--source-boundary-margin` (10 bp) do not teach geometry,
and calls with less than 80% mapped span are kept only as obstacles.
`--site START-END` seeds a site (its support and edges are still learned from
the calls); `--forced-sites-only` analyses only seeded sites.

**Source-strand model.** For each site and strand, a local mixture of
accessible / TF / nucleosome states is fitted to the molecules spanning it.
For a target strand, a site can supply a prior from the opposite strand only
when it has at least `--strand-min-source-support` calls, a focal enrichment
of at least `--strand-min-source-enrichment` (1.5) over the local background,
and is at least as well represented there (as a fraction of reads that fully
map it).

**Rescue.** Only ordinary MSPs are examined. The site must lie completely
inside the MSP and be fully mapped, must not overlap an ordinary TF, and each
selected component needs at least one informative target and a strictly
positive protection LLR ("borderline, not unsupported"). The target read's
weak evidence is combined with the opposite-strand prior. Nearby sites are
composed into every compatible non-overlapping configuration (up to
`--maximum-sites-per-decision`, 8). Absence of marks is judged against each
molecule's own accessible rate, estimated from its MSPs
(`--per-molecule-efficiency`, default; `--global-efficiency` uses the
model-wide rate). For amplified DAF, `--molecule-collapse auto` counts each
PCR family once when learning support, while every read remains a target.

**Edge normalization.** An existing TF call is moved to the shared edges only
when both strands support the same family and its assignment beats an
explicit unmatched null. Updates that would create a new overlap or invert
call order are rejected, and the ordinary edge is kept. Nucleosome edges are
normalized only with `--independent-nuc-edge-refinement` (experimental; not
the TF-conditioned nucleosome reconciliation of consensus), optionally seeded
with `--nuc-site`/`--forced-nuc-sites-only`:

```bash
fiberhmm-strand-rescue -i calls.bam --preset dddb --region chr3L:15039880-15040260 \
    --independent-nuc-edge-refinement --nuc-site 15039920-15040110 -o nuc_sr.json
```

## The report

`-o` writes a JSON report (schema `fiberhmm.strand_rescue.v6`) with every
parameter, input and model provenance, molecule calibration and collapse
diagnostics, the TF (and nucleosome) geometry families, the source models,
every MSP decision and every edge decision. `--proposal-tsv` writes the
proposals as a table. Large regions spill action details into BGZF sidecars
(`--report-layout auto`; `stream` forces it). The report also holds an
experimental `pooled_iterative_geometry_model` block per modelled locus; it is
not a production result.

For a rescued configuration `H*` in a locally supported MSP, with `A` the
ordinary MSP and `S` all supported TF configurations,

```text
P(H* | A or S) = exp(score(H*)) / ( exp(score(A)) + Σ_{H in S} exp(score(H)) )
```

where each score combines the opposite-strand prior with the target read's
chemistry likelihood. Summary tiers: `strong` (≥ 0.95), `review`
(0.5–0.95), `retain_current` (< 0.5). These are normalized model
probabilities, not held-out calibrated probabilities of biological truth,
and the tiers are summaries, not gates: low-probability alternatives are part
of the output.

## Shadow layers

`fiberhmm-strand-rescue-annotate` writes new, sorted, indexed regional BAMs
(one per input with `--output-dir`, or `-o` for one input) with:

```text
nuc_sr.QQQ    one interval per ordinary nuc
tf_sr.QQQ     one interval per ordinary tf, plus every rescued TF component
```

| Byte | Rescued TF (`R0`, `R1`, …) | Edge-normalized call (`H`) | Unchanged call |
|---|---|---|---|
| `q0` | P(H* \| A or S) ×255, shared by all components of one rescue (0 if the action set was truncated) | posterior of the shared edges against this exact ordinary interval ×255 | 255 (sentinel; ignore) |
| `q1` | molecular-left edge reliability | molecular-left edge posterior (255 if unchanged; 0 if the changed edge has no target opportunity) | 0 |
| `q2` | molecular-right edge reliability | molecular-right edge posterior (as `q1`) | 0 |

Rescued components share one `AN` name prefix with roles `R0`, `R1`, … and
switch together; `R` roles occur only in `tf_sr`. An edge-normalized call is
a named singleton `H` that records the ordinal of the ordinary call it
replaces. Unchanged calls are unnamed (`AN` token `.`) with bytes
`(255, 0, 0)`.

A viewer applies one threshold *T* only to named `R` and `H` groups: at
`q0 ≥ T` it shows the rescued interval set, otherwise the baseline (the
containing MSP for `R`, the ordinary call for `H`). `nuc_sr` and `tf_sr` are
complete shadows, not complementary nucleosome/TF states. All three bytes are
linear 0–255 quantities, not Phred scores and not comparable with `tq` or
`nq`.

Ordinary `nuc`, `msp` and `tf` stay in the output; the source BAM is never
edited; old `nuc_sr`/`tf_sr` groups are rebuilt. `--minimum-posterior` (0)
optionally drops low-`q0` rescues from the output. The header carries a
`FIBERHMM-STRAND-RESCUE:v6:` contract and `MA-TYPES` for both layers. The
annotator also accepts v2–v5 reports.

## Audit

`fiberhmm-strand-rescue-audit` checks the v6 header and `MA-TYPES`; exact
`MA`/`AQ`/`AN` alignment with three bytes per shadow interval; `nuc_sr`
cardinality equal to ordinary `nuc`; `tf_sr` cardinality equal to ordinary
TFs plus rescued components; fixed `(255, 0, 0)` rows for unchanged calls;
singleton `H` roles with a source ordinal; atomic contiguous `R` groups
sharing one `q0` and confined to `tf_sr`; 255 on unchanged edges of `H`
groups; no newly introduced overlaps; and BAM integrity and indexing. It
also validates v2–v5 BAMs.

## Scope of the claims

The validation benchmark (five-fold molecule-disjoint mask-and-recover on two
loci, distributed with the paper) measures recovery of ordinary FiberHMM
calls, a silver label, not biological sensitivity, FDR or TF identity. Treat
rescued calls as alternatives to inspect, not as ground truth.

Every option: [`fiberhmm-strand-rescue`](../reference/cli.md#fiberhmm-strand-rescue),
[`-annotate`](../reference/cli.md#fiberhmm-strand-rescue-annotate),
[`-audit`](../reference/cli.md#fiberhmm-strand-rescue-audit).
Compact consensus-state slots can be added to `tf_sr` with
[`fiberhmm-tag-consensus`](tag-consensus.md).
