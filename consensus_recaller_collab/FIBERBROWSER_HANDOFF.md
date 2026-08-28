# FiberBrowser handoff: connected consensus layers

> **Superseded for CR (2026-07-14).** The current CR ownership, call-space
> model, JSON API, no-220-bp rule, and optional `nuc_cr`/`tf_cr` export contract
> are in
> [`../docs/FIBERBROWSER_CONSENSUS_RECONSTRUCTION.md`](../docs/FIBERBROWSER_CONSENSUS_RECONSTRUCTION.md).
> This document remains historical evidence for the connected-layer prototype.
> Production physical-strand rescue now uses the narrower
> `fiberhmm-strand-rescue*` contract and only `nuc_sr`/`tf_sr`.

This document is the implementation contract for rendering aggressive
FiberHMM consensus-recaller output in FiberBrowser. It is intentionally
complete enough to hand directly to a FiberBrowser coding agent. The ordinary
FiberHMM caller and ordinary `nuc`, `tf`, and `msp` layers are unchanged.

## User-visible result

FiberHMM writes two optional connected layer families:

| pass | current-state layer | TF-alternative layer |
|---|---|---|
| composite recall (CR) | `nuc_cr` | `tf_cr` |
| physical-strand rescue (SR) | `nuc_sr` | `tf_sr` |

Each family should have one integer threshold slider `T` in `[0,255]`.
For a linked N-versus-TF decision, moving the slider must display exactly one
state when both layers are enabled:

```text
TF when q0_tf >= T
N  when q0_nuc >= 256 - T
```

The stored bytes are complementary, so this is an exact partition with no gap
and no double display. `T=128` is the rounded MAP split. Lower `T` is more
TF-aggressive; higher `T` is more nucleosome-conservative.

The slider is a secondary-call exploration control. It never changes the
ordinary `nuc`, `tf`, or `msp` records and never writes to the BAM.

## Detecting the format

Require the following BAM header comment before enabling connected behavior:

```text
@CO FIBERHMM-CONSENSUS:v3:...;semantics=paired_hypotheses;quality_spec=QQQQQ;...
```

In a `pysam`-style header dictionary, `CO` contains the text after `@CO`.
The full comment currently declares:

```text
FIBERHMM-CONSENSUS:v3:
groups=nuc_cr,tf_cr,nuc_sr,tf_sr;
semantics=paired_hypotheses;
quality_spec=QQQQQ;
q_scale=linear_probability;
q0=represented_state_posterior;
pair_sum=255;
pairing=AN_shared_prefix;
tf_if=q0_tf>=T;
nuc_if=q0_nuc>=256-T;
accessible_current=ordinary_msp;
accessible_tf_role=ATn;
aux=q1_configuration,q2_molecule,q3_population,q4_specificity;
baseline_q0=255;
baseline_aux=0;
tiers=paired-aggressive;
report_sha256=<sha256>
```

Whitespace/newlines above are only for readability; the real `@CO` record is
one line. Parse semicolon-delimited key/value fields after the version prefix.
Unknown fields must be ignored for forward compatibility.

`@CO MA-TYPES:v1:nuc_cr,tf_cr,nuc_sr,tf_sr` advertises layer names for normal
custom-layer discovery. It does not by itself authorize connected rendering.

Files with `FIBERHMM-CONSENSUS:v2:` use selected shadow-callset semantics.
They must continue to use the existing independent-layer quality filtering;
an N/TF overlap in v2 is not a connected pair. If no recognized consensus
header is present, do not infer connected behavior merely from a layer name.

## MA, AQ, and AN layout

Paired files follow the standard Molecular Annotation specification. Their
custom sections look like:

```text
nuc_cr.QQQQQ:<start>-<length>,...
tf_cr.QQQQQ:<start>-<length>,...
nuc_sr.QQQQQ:<start>-<length>,...
tf_sr.QQQQQ:<start>-<length>,...
```

The five `AQ` bytes for every annotation are positional in normal MA section
and interval order. Do not sort annotations before associating their AQ row
or AN name. First parse the complete positional records, then sort for drawing
if needed.

`AN` contains exactly one comma-delimited field per MA interval across all MA
sections. Empty fields are valid and significant positional placeholders.
Preserve leading, internal, and trailing empty fields when splitting. For
compatibility with older FiberHMM records, a literal `.` can be normalized to
an empty name, but paired v3 output itself uses empty fields.

The paired-name grammar is:

```regex
^(fh(?:cr|sr)_[0-9a-f]{16})_(N|T[0-9]+|AT[0-9]+)$
```

Examples:

```text
fhcr_0123456789abcdef_N
fhcr_0123456789abcdef_T0
fhcr_0123456789abcdef_T1
fhsr_fedcba9876543210_N
fhsr_fedcba9876543210_T0
fhsr_0123abcd4567ef89_AT0
```

Treat the prefix as an opaque decision ID. Do not derive biology, coordinates,
or read identity from the hash. The suffix is the member role:

- `_N`: current nucleosome state;
- `_T0`, `_T1`, ...: all components of one TF-complex alternative.
- `_AT0`, `_AT1`, ...: TF components rescued from an accessible/MSP
  current state; these are intentionally one-sided.

Every T component with the same prefix switches atomically. A one-component
TF alternative still uses `_T0`.

An annotation with an empty/nonmatching AN name is an unpaired baseline call.
It remains fixed and should not be switched by the connected threshold.

## Quality-byte meanings

All qualities are linear probability bytes: approximately `q / 255`, not
Phred scores and not ordinary FiberHMM `tq = 10 * LLR`.

| byte | CR meaning | SR meaning |
|---:|---|---|
| `q0` | posterior of the represented N or aggregate TF-complex state | posterior of the represented current or aggregate TF state, conditional on current-versus-TF |
| `q1` | best exact configuration probability conditional on TF complex | best configuration probability conditional on TF |
| `q2` | equal-odds probability from the molecule-only log likelihood ratio | equal-odds probability from the molecule-only log likelihood ratio versus the current state |
| `q3` | selected same-cohort local TF-complex prior probability | source-population TF prior probability versus the current state |
| `q4` | equal-odds probability that true boundaries beat the best shifted-boundary control | minimum explicit source-support fraction across selected sites |

For a paired decision:

- all T components have the same five-byte row;
- `q0_nuc + q0_tf == 255` exactly;
- `q1..q4` are identical on the N and every T component;
- `q1..q4` describe evidence for/resolution of the TF alternative. They do not
  reverse meaning on the N record.

An unavailable neutral auxiliary value is encoded as 128. An untested carried
baseline record is `(255,0,0,0,0)`. The zero auxiliary bytes on a baseline
record mean “not evaluated by consensus,” not biological evidence against it.

The authoritative JSON report retains unrounded probabilities, likelihood
ratios, opportunity/hit counts, source depth, and shifted controls. Browser
tooltips should call the bytes compact diagnostics rather than raw evidence.

## Decision types

### CR: current nucleosome versus TF complex

CR decisions always have one `_N` member and at least one `_Tn` member. A
resolved complex may have one, two, three, or more T components. An unresolved
TF agglomeration has one broad `_T0` interval, often identical to the N
envelope; this is intentional. Its low `q1` distinguishes uncertain exact
decomposition from uncertain N-versus-TF class.

### SR from a current nucleosome

These also have `_N` and `_Tn` members and use the same connected threshold.
The `fhsr_` prefix distinguishes their evidence semantics from CR.

### SR from a current accessible MSP

These are deliberately one-sided and contain `_Tn` members but no `_N`.
The unchanged ordinary `msp` annotation is the accessible/current state.
Render the T components only when their decision passes the TF threshold;
otherwise render no `tf_sr` component and leave normal MSP rendering alone.

## Rendering algorithm

Build decisions per read after MA/AQ/AN positional parsing. Do not group across
reads, even if a malformed file reuses a prefix.

Recommended TypeScript-like data structures:

```ts
type ConsensusPass = "cr" | "sr";

type ConnectedMember = {
  layer: "nuc_cr" | "tf_cr" | "nuc_sr" | "tf_sr";
  start: number;
  length: number;
  qualities: [number, number, number, number, number];
  name: string;
  role: "N" | `T${number}` | `AT${number}`;
};

type ConnectedDecision = {
  id: string;                  // shared AN prefix
  pass: ConsensusPass;
  n?: ConnectedMember;
  tf: ConnectedMember[];       // sorted by numeric T suffix for identity
  qN?: number;
  qTF: number;
  auxiliary: [number, number, number, number];
  current: "N" | "A";
};
```

Grouping pseudocode:

```ts
const pairedName = /^(fh(?:cr|sr)_[0-9a-f]{16})_(N|T([0-9]+)|AT([0-9]+))$/;
const decisions = new Map<string, ConnectedDecision>();
const fixed: ParsedAnnotation[] = [];

for (const annotation of parsedAnnotationsInOriginalOrder) {
  if (!connectedLayerNames.has(annotation.type)) continue;
  const match = pairedName.exec(annotation.anName ?? "");
  if (!match) {
    fixed.push(annotation);
    continue;
  }
  if (annotation.qualities.length !== 5) markMalformed(annotation);

  const id = match[1];
  const role = match[2];
  const pass = id.startsWith("fhcr_") ? "cr" : "sr";
  const decision = getOrCreateDecisionForThisRead(id, pass);
  addMemberWithoutReorderingQualityAssociation(decision, role, annotation);
}

for (const decision of decisions.values()) validateDecision(decision);
```

`validateDecision` sets `current="A"` only when every TF role is `ATn` and no
N member exists. Ordinary `Tn` roles require exactly one N. Reject mixed
`Tn`/`ATn` roles. This explicit distinction prevents a truncated or corrupt
N-versus-TF group from being misread as an intentional MSP-origin rescue.

Threshold and drawing pseudocode:

```ts
function selectedState(d: ConnectedDecision, T: number): "TF" | "N" | "A" {
  if (!Number.isInteger(T) || T < 0 || T > 255) throw new RangeError();
  if (d.current === "A") return d.qTF >= T ? "TF" : "A";

  // These are equivalent for a valid pair. Checking both in development
  // catches corrupt or incorrectly parsed AQ rows.
  const tfSelected = d.qTF >= T;
  const nSelected = d.qN! >= 256 - T;
  if (tfSelected === nSelected) throw new Error("non-complementary pair");
  return tfSelected ? "TF" : "N";
}

function visibleMembers(
  d: ConnectedDecision,
  T: number,
  enabledLayers: Set<string>,
): ConnectedMember[] {
  const state = selectedState(d, T);
  if (state === "TF") {
    return d.tf.filter(x => enabledLayers.has(x.layer));
  }
  if (state === "N" && d.n && enabledLayers.has(d.n.layer)) return [d.n];
  return []; // state A is represented by the ordinary MSP layer
}
```

Always render fixed/unpaired baseline members when their layer is enabled.
In particular, `T=0` must not hide unchallenged nucleosomes. The connected
threshold applies to linked decisions, not indiscriminately to every q0 in the
two layers.

If only one member layer is enabled, still compute the decision state first,
then apply layer visibility. For example, if only `tf_cr` is enabled and the
decision selects N, display neither member. Do not fall back to drawing TF just
because the N layer is disabled.

## Auxiliary filters and tooltips

Recommended initial UI:

- one linked threshold slider for CR and one for SR, both defaulting to 128;
- presets such as “aggressive” (64), “MAP” (128), and “conservative” (192);
- a tooltip showing `q0..q4`, approximate probabilities, decision ID, and the
  number of TF components;
- optional minimum filters for `q1..q4` only after connected rendering works.

If an auxiliary filter rejects a decision, fall back to its current state:

- paired N→TF: render N;
- one-sided A→TF: suppress the TF alternative and leave ordinary MSP alone.

Apply auxiliary filters once per decision, never independently to individual
T components. Do not apply them to unpaired baseline annotations; their zeroes
mean untested. A useful tooltip label set is:

```text
q0 state posterior
q1 configuration certainty
q2 molecule evidence
q3 population support
q4 specificity/source support
```

CR and SR `q4` have different exact meanings, so include the pass-specific
description in an expanded tooltip.

## Required validation

A valid paired N→TF decision has:

1. exactly one `_N` member;
2. one or more uniquely numbered `_Tn` members;
3. N in `nuc_cr` and T in `tf_cr`, or N in `nuc_sr` and T in `tf_sr`;
4. five AQ bytes per member;
5. one common T quality row across every component;
6. identical `q1..q4` on N and T;
7. `q0_nuc + q0_tf == 255`;
8. no duplicate role within the decision.

A valid one-sided decision has no N, at least one `ATn` member, and must use an
`fhsr_` prefix in `tf_sr`. `ATn` suffixes must be unique. A one-sided group
using ordinary `Tn` is malformed because it has lost its expected N member.
Contiguous numbering is expected but is not necessary for drawing if all
other checks pass.

Recommended failure behavior:

- log one deduplicated warning with read name and decision prefix;
- do not connected-render a malformed decision;
- keep ordinary `nuc`/`tf`/`msp` layers available;
- never guess a missing state or silently combine qualities;
- expose a malformed-pair count in diagnostics.

Validation must be scoped per read. A decision hash is stable and includes
the source library in its construction, but consumers should not rely on
global uniqueness as a substitute for read scoping.

## Exact threshold test vectors

For a valid pair with `q0_tf=178` and `q0_nuc=77`:

| T | expected state |
|---:|---|
| 0 | TF |
| 64 | TF |
| 128 | TF |
| 178 | TF |
| 179 | N |
| 255 | N |

For `q0_tf=127`, `q0_nuc=128`, `T=128` must select N. For
`q0_tf=128`, `q0_nuc=127`, `T=128` must select TF. At every threshold exactly
one condition is true for a complementary pair.

For a one-sided SR decision with `q0_tf=127`, draw its AT components at
`T<=127` and suppress them at `T>=128`.

## Minimal synthetic record

This example contains one CR decision with two TF components. Coordinates are
illustrative. AQ rows are shown separated for readability, although BAM stores
one flat byte array.

```text
MA:Z:200;nuc_cr.QQQQQ:51-100;tf_cr.QQQQQ:51-30,121-30
AQ:B:C:
  77,204,153,191,230,
  178,204,153,191,230,
  178,204,153,191,230
AN:Z:fhcr_0123456789abcdef_N,fhcr_0123456789abcdef_T0,fhcr_0123456789abcdef_T1
```

At `T=178`, draw both TF intervals and not the N interval. At `T=179`, draw
the N interval and neither TF interval. `q1..q4` must be reported once for the
whole decision or identically for each component.

## FiberBrowser acceptance tests

The FiberBrowser implementation is complete when automated tests cover:

1. v3 header detection and v2/no-header fallback;
2. positional MA/AQ/AN parsing with leading and trailing empty AN fields;
3. the exact threshold vectors above, including 0 and 255;
4. atomic two- and three-TF switching;
5. one-sided MSP→TF behavior;
6. fixed baseline calls at every slider value;
7. layer-enable behavior after, not before, state selection;
8. decision-level auxiliary filtering with current-state fallback;
9. malformed complement, inconsistent auxiliary, duplicate role, wrong-layer,
   and missing-N diagnostics;
10. unchanged rendering of existing v2 selected shadow callsets;
11. reverse-aligned reads (coordinates are already in the MA molecular frame;
    the connected logic must not reverse them a second time);
12. a real paired BAM audit in which every valid prefix changes atomically and
    no ordinary annotation is modified.

The FiberHMM-side unit tests and persistent real-data pair audit are the source
of truth for the writer. FiberBrowser should report the paired header version
and malformed-pair count in its debug/session metadata so cross-repo failures
can be diagnosed without inspecting screenshots.

## Persistent implementation fixtures

The current Dropbox-backed fixture root is:

```text
/mnt/g/Dropbox/Fiber-NET-seq/FiberHMM v1.0/Release v2.0.0/consensus_visualization_outputs/paired_v7
```

Use these three BAMs for consumer development:

- `homie/homie.pooled.paired-v7.bam`: CR, including atomic one- and two-TF
  alternatives;
- `ind_nanopore/ind-nanopore.pooled.paired-v7.bam`: the smallest fixture,
  including one-sided `ATn` SR and weak paired-N alternatives;
- `uba1_ddda/uba1_ddda_recall.consensus-overlay.bam`: strong/review DddA SR.

Each has a neighboring `.bai`. The authoritative aggregate writer audit is
`paired-v7.all-panels.audit.json` at the fixture root; it covers all eight
recommended BAMs. These paths are test fixtures, not format detection: the
consumer must still recognize semantics from the BAM header.
