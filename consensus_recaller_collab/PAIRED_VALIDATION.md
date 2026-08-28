# Aggressive paired consensus validation — 2026-07-14

This document records the first complete focal validation of the aggressive
paired consensus-recaller output. It supplements, rather than replaces, the
conservative selected-state validation in
[`PRODUCTION_VALIDATION.md`](./PRODUCTION_VALIDATION.md).

The baseline FiberHMM `nuc`, `tf`, and `msp` calls remain unchanged. Every
result below is a secondary alternative in a new regional derivative BAM.
Independent assays were not used as priors: repeated BAMs are only explicitly
pooled shards/timepoints from the same inference cohort.

## Frozen implementation

- CR retains every supported 90--220 bp current nuc, including non-MAP
  alternatives, and compares canonical N with an aggregate one-/multi-TF
  complex. Exact decomposition remains a separate conditional score.
- SR retains every source-supported current MSP or 90--220 bp current nuc with
  positive local target evidence. Nuc targets receive linked N/TF records;
  MSP targets receive explicit one-sided `ATn` TF records because ordinary
  `msp` already represents the current accessible state.
- Protected blocks longer than 220 bp are not scored. They must first be
  resolved by the baseline nucleosome recaller.
- A nuc already overlapping an ordinary TF is not treated as a current N
  alternative; the existing TF already overwrites that contradictory shadow
  nuc.
- Sites separated by up to 220 bp are modeled in one SR window by default, so
  one eligible nuc cannot silently become two independent decisions.
- The paired collector rejects any report that still assigns two decisions to
  the same current nuc.
- Paired MA groups use `QQQQQ`; N and TF `q0` sum to exactly 255, auxiliary
  qualities are shared, and `AN` links atomic components. `Tn` requires one N;
  `ATn` explicitly marks an accessible-origin one-sided rescue.
- The full FiberBrowser consumer contract is
  [`FIBERBROWSER_HANDOFF.md`](./FIBERBROWSER_HANDOFF.md).

Only standard aligned sequence, hard `MM`/`ML`, and existing `MA` calls were
used. Nanopore Hia5 used the explicit hard threshold `ML >= 248`. No raw IPD,
pulse features, sub-threshold kinetic signal, PWM, external assay, or
cross-library occupancy prior entered inference.

## Persistent panel

All outputs are under the Dropbox-backed directory:

```text
consensus_visualization_outputs/paired_v7/
```

Exact commands are frozen in
[`run_paired_validation.sh`](./run_paired_validation.sh). The script writes
reports, TSVs, BAMs, BAIs, combined visualization BAMs, and JSON integrity
audits directly under that directory. It does not use system temporary
storage for scientific artifacts. Individual BAM writes are staged atomically
beside their final output.

The panel contains:

- five pooled fly 2--4 hr PacBio shards at Homie, Nhomie, SF1, and SF2;
- six explicitly pooled fly targeted DddB time-course/cohort BAMs;
- two explicitly pooled short fly Nanopore Hia5 BAMs at hard threshold 248;
- GM12878 UBA1 DddA; and
- GM12878 NAPA DddA.

## Decision surface

`TF @ T` is the number of decisions rendered as TF at the indicated connected
slider threshold. For a paired decision the alternative current state is N;
for `ATn` the alternative is ordinary MSP/accessibility. `T=128` is the
rounded MAP split. `T=64` is more aggressive and `T=192` more conservative.
Strong/review are the pre-existing conservative proposal tiers; paired output
is not gated by them.

| Panel | total decisions | paired N→TF | one-sided A→TF | TF @ 64 | TF @ 128 | TF @ 192 | strong | review |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Homie PacBio CR | 574 | 574 | 0 | 76 | 59 | 52 | 31 | 28 |
| Nhomie PacBio CR | 198 | 198 | 0 | 74 | 58 | 41 | 15 | 42 |
| SF1 PacBio CR | 1,590 | 1,590 | 0 | 110 | 95 | 81 | 41 | 52 |
| SF2 PacBio CR | 1,616 | 1,616 | 0 | 183 | 157 | 112 | 52 | 101 |
| fly DddB SR | 4,510 | 3,507 | 1,003 | 2,024 | 1,574 | 1,410 | 998 | 362 |
| fly Nanopore Hia5 SR | 206 | 142 | 64 | 64 | 64 | 42 | 0 | 64 |
| UBA1 DddA SR | 597 | 463 | 134 | 328 | 326 | 303 | 265 | 61 |
| NAPA DddA SR | 326 | 236 | 90 | 306 | 304 | 301 | 0 | 300 |
| **Total** | **9,617** | **8,326** | **1,291** | — | — | — | — | — |

The 3,978 PacBio CR rows are the complete tested surface. Their `T=128` counts
exactly reproduce the previous 59/58/95/157 MAP demotions, while lower
thresholds expose additional borderline alternatives without rerunning the
model. CR alternatives contain one, two, or three TF components and are linked
atomically.

The stranded pass adds the major requested capability: 4,348 current
nucleosomes now have an explicit opposite-strand-supported TF alternative in
addition to the 1,291 accessible-origin rescues. Chemistry still matters:

- DddB depth yields 3,507 testable nuc alternatives and 1,003 MSP rescues; at
  `T=128`, 571 nuc alternatives and all 1,003 MSP alternatives render as TF.
- Short threshold-248 Nanopore yields 142 valid nuc alternatives, but none
  reaches `T=64`; its 64 accessible-origin calls supply the visible signal.
  This is appropriately weak behavior for the available short, perturbed
  panel, and no read is required to span both focal sites.
- UBA1 yields 463 nuc alternatives; 192 render as TF at `T=128`, alongside all
  134 MSP-origin alternatives.
- NAPA yields 236 nuc alternatives; 215 render as TF at `T=128`, alongside 89
  of 90 MSP-origin alternatives. These are review-tier/hypothesis-generating,
  not newly authoritative calls. In particular, a high conditional N-vs-TF
  byte does not grant DddA boundary authority or bypass the full-model tier.

DddA nucleosome fingerprints are not used as an extra subnucleosomal prior in
this implementation. That deliberately de-prioritized extension can be tested
later without changing the connected-layer contract.

## Ceiling and contradiction guardrails

Focal source-supported nucs skipped because they exceeded 220 bp were:

| Panel | skipped >220 bp | skipped because ordinary TF already overlaps N |
|---|---:|---:|
| Homie CR | 155 | 0 |
| Nhomie CR | 222 | 0 |
| SF1 CR | 613 | 0 |
| SF2 CR | 896 | 0 |
| DddB SR | 5,855 | 0 |
| Nanopore SR | 68 | 0 |
| UBA1 SR | 5 | 23 |
| NAPA SR | 2,874 | 0 |

The large DddB/NAPA counts confirm why the ceiling is necessary: weak stranded
inputs retain many very broad protected blocks, but those blocks do not offer
an identifiable focal N-versus-TF test. They are reported and excluded rather
than converted into thousands of spurious rescues.

## BAM integrity

The reusable `fiberhmm-consensus-audit-pairs` command checks:

- v3 header semantics and `QQQQQ` declaration;
- MA/AQ/AN positional alignment, including empty AN fields;
- exactly one N for `Tn` groups and no N for explicit `ATn` groups;
- one shared quality row across every multi-TF component;
- identical auxiliary bytes and exact `q0_nuc + q0_tf = 255`;
- expected pass/layer membership, unique roles, indexing, and quickcheck; and
- exact state counts at thresholds 0, 64, 128, 192, and 255.

All eight recommended combined/single BAMs are valid. Across 67,800 regional
alignment records, the audit found:

- 9,617 decisions;
- 8,326 paired N→TF decisions;
- 1,291 explicit accessible-origin decisions;
- 1,861,999 fixed baseline consensus annotations; and
- zero malformed, orphaned, non-complementary, or inconsistent decisions.

The consolidated machine-readable result is
`consensus_visualization_outputs/paired_v7/paired-v7.all-panels.audit.json`.

The recommended BAMs are:

```text
paired_v7/homie/homie.pooled.paired-v7.bam
paired_v7/nhomie/nhomie.pooled.paired-v7.bam
paired_v7/sf1/sf1.pooled.paired-v7.bam
paired_v7/sf2/sf2.pooled.paired-v7.bam
paired_v7/ind_dddb/ind-dddb.pooled.paired-v7.bam
paired_v7/ind_nanopore/ind-nanopore.pooled.paired-v7.bam
paired_v7/uba1_ddda/uba1_ddda_recall.consensus-overlay.bam
paired_v7/napa_ddda/napa_recaller_TF.consensus-overlay.bam
```

Every BAM has a neighboring index. Each panel directory contains the exact
JSON audit, including BAM SHA-256, q0 histogram, TF-component histogram, and
threshold-state table. The pooled visualization files were made with
`samtools merge`; no FiberHMM merge implementation was touched.

## Regression status

- Focused consensus tests: 81 passed.
- Full repository regression: 640 passed, 3 skipped, 26 benchmark tests
  deselected.
- A clean wheel contains the four consensus console scripts, validation policy
  JSON, manifest, and FiberHMM models; all four installed entry points pass
  out-of-tree help/import smoke tests.
- All report JSON/TSV writes are atomic and all candidate tables used by
  paired output are untruncated.
- All scientific validation artifacts are retained in the Dropbox project
  tree; `/tmp` is not part of the reproducible workflow.

## Next boundary

The FiberHMM side is ready for focal visual review. FiberBrowser should
implement the connected parser/slider exactly as specified in
[`FIBERBROWSER_HANDOFF.md`](./FIBERBROWSER_HANDOFF.md), including fixed
baseline behavior, `ATn` handling, decision-level auxiliary filters, and v2
fallback. Genome-wide execution remains intentionally deferred until visual
review and threshold calibration across these focal panels are satisfactory.
