# Consensus-recaller MA visualization layers

`fiberhmm-consensus-annotate` converts a completed consensus-recaller JSON
report into indexed regional BAMs that FiberBrowser can display. It is a
visualization handoff, not an in-place replacement: input BAMs and the ordinary
`nuc`/`tf` annotations are never changed.

There are two output semantics. `--paired` is the high-sensitivity focal mode
and is the recommended interface for the consensus recaller. It writes every
locally supported CR/SR alternative, including non-MAP alternatives, so one
FiberBrowser threshold can move calls between N and TF. The older default,
`--strong-only`, and `--all-tested` modes write one already-selected shadow
callset and remain available while paired rendering is integrated and
validated. The implementation contract is recorded in
[`PAIRED_RECALL_PLAN.md`](./PAIRED_RECALL_PLAN.md).
The detailed FiberBrowser parser, UI, failure-mode, and acceptance-test handoff
is [`FIBERBROWSER_HANDOFF.md`](./FIBERBROWSER_HANDOFF.md).

## Paired aggressive mode

Run with `--paired`. The derivative BAM retains every original tag and adds up
to four custom annotation types:

| MA group | Paired meaning |
|---|---|
| `nuc_cr.QQQQQ` | Baseline nucs plus the N side of every composite-recall decision |
| `tf_cr.QQQQQ` | Baseline TFs plus the linked TF-complex side of every composite-recall decision |
| `nuc_sr.QQQQQ` | Baseline nucs plus the N side of every strand-rescue decision on a current nuc |
| `tf_sr.QQQQQ` | Baseline TFs plus linked strand-rescue alternatives from current nucs or MSPs |

For an N→TF decision, both annotations intentionally coexist in raw MA. They
are not two calls to draw simultaneously. `AN` labels link them, for example:

```text
fhcr_0123456789abcdef_N
fhcr_0123456789abcdef_T0
fhcr_0123456789abcdef_T1
```

The shared prefix identifies one decision. All `T0`, `T1`, ... components are
one atomic multi-TF alternative. An MSP→TF strand rescue is one-sided because
the unchanged ordinary `msp` annotation already represents the current
accessible state; its components use `AT0`, `AT1`, ... suffixes so FiberBrowser
can distinguish it from a malformed N→TF group that lost its N member.

The five linear quality bytes are:

| byte | CR | SR |
|---:|---|---|
| `q0` | represented N or aggregate TF-complex posterior | represented current or aggregate TF posterior |
| `q1` | exact configuration probability conditional on TF complex | configuration probability conditional on TF |
| `q2` | equal-odds probability from the molecule likelihood ratio | equal-odds probability from the molecule likelihood ratio |
| `q3` | same-cohort local TF-complex prior | opposite-strand source-population TF prior |
| `q4` | true-boundary preference over the best shifted control | explicit source-support fraction |

For each N/TF pair, the writer first rounds `q0_tf` and stores
`q0_nuc = 255 - q0_tf`; the bytes therefore sum to exactly 255. Auxiliary
bytes are identical on every member of the decision. A carried baseline call
has `(255,0,0,0,0)`: it is fixed and was not evaluated by consensus.

For slider byte threshold `T`, FiberBrowser should group annotations by their
`AN` prefix and render:

```text
TF components when q0_tf >= T
N component     when q0_nuc >= 256 - T
```

Equivalently, render the TF components when `q0_tf >= T`, otherwise the N
component. `T=128` is the MAP split, a lower value is more TF-aggressive, and
a higher value is more nucleosome-conservative. For a one-sided MSP→TF rescue,
draw the TF only when it passes `T`; otherwise the ordinary MSP remains.

The BAM header advertises the types with `MA-TYPES:v1` and declares these
semantics in `FIBERHMM-CONSENSUS:v3`. Empty positional `AN` fields are used for
unnamed annotations. The JSON report remains authoritative for uncompressed
probabilities, log likelihoods, source depth, hard-call opportunity counts,
and controls.

## Selected-state legacy mode

Without `--paired`, the derivative BAM may add four logical feature
groups to the existing `MA`/`AQ` containers:

| MA group | Meaning |
|---|---|
| `nuc_cr.Q` | Complete nucleosome callset after composite recall: original nucs minus selected demotions and any final TF overlaps |
| `tf_cr.Q` | Complete TF/TF-complex callset after composite recall: original TFs plus selected replacements |
| `nuc_sr.Q` | Complete nucleosome callset after strand rescue, with final TF overlaps removed |
| `tf_sr.Q` | Complete TF callset after strand rescue: original TFs plus rescued calls |

These are complete shadow callsets on every read written to the regional BAM.
They are initialized from the ordinary calls and then the selected state change
is applied. Thus a CR-demoted block is absent from `nuc_cr` and its replacement
is present in `tf_cr`; `nuc_cr` and `tf_cr` are not simultaneous hypotheses.
As a final invariant, carried or new TF calls overwrite every overlapping
shadow nuc. This also cleans legacy inputs that already contain an ordinary TF
under an older HMM nuc.
The unchanged ordinary `nuc` layer is deliberately preserved for comparison,
so `nuc` can overlap a recalled `tf_cr`. Turn off ordinary `nuc` when viewing
only the post-CR callset.

Stage 1 tests canonical nucleosome versus the aggregate TF-complex class. Stage
2 chooses an exact one-/multi-TF layout only when its posterior conditional on
the complex class is at least 0.8. Otherwise the original envelope becomes one
broad `tf_cr` interval representing an unresolved TF agglomeration. This
records “not a canonical nucleosome” without manufacturing a precise split.

All four names are advertised through `@CO MA-TYPES:v1:` so FiberBrowser can
initialize their layers without first sampling a rare annotated record. A
second `@CO FIBERHMM-CONSENSUS:v2:` declaration records the callset semantics, Q transform, tier
mode, and exact report SHA-256 inside each derivative BAM.

FiberBrowser exposes these as custom layers and leaves custom layers off by
default. Explicitly enable `tf_cr` and/or `nuc_cr` under **Data Layers**. Its
quality-filter UI calls the one posterior dimension `q0`.

### Legacy Q definition

Each challenged or newly added annotation has one quality byte:

```text
Q = round(255 * P(selected state | target hard calls, source population))
```

This is a linear posterior byte, not Phred and not the ordinary TF `tq`
(`10 * LLR`). Useful landmarks are:

| Posterior | Q |
|---:|---:|
| 0.50 | 128 |
| 0.90 | 230 |
| 0.95 | 242 |
| 0.99 | 252 |
| 1.00 | 255 |

For a retained tested nuc this is `P(N)`. For a resolved replacement it is the
joint posterior of that exact configuration. For a broad unresolved replacement
it is the aggregate `P(TF complex)`. An unchanged baseline call that was never
challenged receives `Q=255` (retained by construction); its original evidence
quality remains available on ordinary `nuc`/`tf`.

The JSON `proposal_tier` remains authoritative. Byte rounding means a review
proposal immediately below 0.95 can also round to 242. Use `--strong-only` to
make an exactly tier-filtered BAM. No `QQQ` edge bytes are invented: the
population configuration evidence is not a molecule-specific left/right edge
measurement, and low-resolution DddB/Nanopore support does not gain boundary
authority through depth.

## Usage

Aggressive paired alternatives for one focal report:

```bash
fiberhmm-consensus-annotate \
  --report homie.consensus.json \
  --paired \
  --output-dir homie_paired
```

All report JSON, BAM, BAI, manifest, and audit outputs should be kept in a
persistent project directory. The writer stages each BAM beside its final
output for atomic replacement; it does not place scientific outputs in the
system temporary directory.

Audit one or more completed paired BAMs and retain the machine-readable result:

```bash
fiberhmm-consensus-audit-pairs \
  -i homie_paired/2-4hr_11.consensus-overlay.bam \
  -i homie_paired/2-4hr_4.consensus-overlay.bam \
  -o homie_paired/paired-integrity.audit.json
```

One selected input BAM:

```bash
fiberhmm-consensus-annotate \
  --report homie.consensus.json \
  -i /path/to/2-4hr_11.bam \
  -o homie.2-4hr_11.overlay.bam
```

Every BAM represented in a pooled report, without physically merging them:

```bash
fiberhmm-consensus-annotate \
  --report homie.consensus.json \
  --output-dir homie_overlays
```

Strong proposals only:

```bash
fiberhmm-consensus-annotate \
  --report ind.dddb.consensus.json \
  --strong-only \
  -o ind.dddb.strong.overlay.bam
```

Every scored CR decision, including retain-N calls:

```bash
fiberhmm-consensus-annotate \
  --report homie.consensus.json \
  --all-tested \
  --output-dir homie_all_tested
```

By default the writer uses the report's loaded focal region and applies strong
and review proposals. `--all-tested` applies the stage-1 MAP state for every
scored CR decision, including decisions that failed the boundary-control tier;
a MAP N stays in `nuc_cr`, while a MAP TF complex replaces it. It never emits
simultaneous N/TF alternatives. This is the permissive mode intended for
quality-filtered visual review. Reports produced before the complete state
table was added are rejected with a rerun instruction. Each
output is coordinate sorted, indexed, and
contains a `@PG` record with the report SHA-256, Q definition, and tier mode.
Input size and modification time are checked against report provenance. A
truncated proposal list is rejected unless explicitly acknowledged.

For amplified DAF, proposals apply to the representative molecule retained by
the report's deamination-fingerprint collapse. They are not propagated back to
every PCR-family member, because that would visually restore the depth bias the
collapse was designed to remove.

## Previously validated selected-state examples

The recommended combined BAMs are:

- `consensus_visualization_outputs/homie/self_contained_shadow_all/homie.pooled-library.self-contained-shadow-all.bam`
- `consensus_visualization_outputs/nhomie/self_contained_shadow_all/nhomie.pooled-library.self-contained-shadow-all.bam`
- `consensus_visualization_outputs/sf1/self_contained_shadow_all/sf1.pooled-library.self-contained-shadow-all.bam`
- `consensus_visualization_outputs/sf2/self_contained_shadow_all/sf2.pooled-library.self-contained-shadow-all.bam`

Across Homie/Nhomie/SF1/SF2, all 3,978 tested calls matched their source MA
annotation. The permissive MAP callsets remove 59/58/95/157 nucs and add
115/110/197/275 TF or TF-complex intervals. A full interval audit
found zero exact or partial `nuc_cr`/`tf_cr` overlaps. All per-input and
combined BAMs are indexed and pass `samtools quickcheck`.

Existing strand-rescue examples remain available:

- Focal fly DddB: all 1,003 strong+review SR proposals matched, alongside 113
  CR MAP demotions in the permissive callset.
- UBA1 DddA: 134 `tf_sr` overlays (111 strong, 23 review).
- NAPA DddA: 89 review-only `tf_sr` overlays.
- Short threshold-248 Nanopore: all 69 current proposals matched across two regional
  derivative BAMs; multiple calls on one short molecule remain distinct.

All 36 self-contained example BAMs passed quick checks and indexing. A complete audit
found zero N/TF overlaps within either shadow pass. Forward and reverse
alignments are projected into the molecular coordinate frame required by the
MA specification.

Those zero-overlap audits apply specifically to selected-state v2 shadow
callsets. Paired v3 files intentionally contain linked N/TF overlaps; their
integrity criterion is instead one shared `AN` prefix, complementary `q0`,
identical auxiliary bytes, and atomic TF components.
