# Annotations and scores

FiberHMM writes its calls into the BAM in two forms: the fibertools-style
legacy tags (`ns`/`nl`, `as`/`al`) that every Fiber-seq tool reads, and the
[Molecular-annotation](https://github.com/fiberseq/Molecular-annotation-spec)
tags (`MA`, `AQ`, `AN`) that carry every call type with its quality bytes.
This page explains what they mean; [BAM tags](../reference/bam-tags.md) lists
every tag FiberHMM writes.

## Legacy tags

| Tag | Content |
|---|---|
| `ns` / `nl` | nucleosome starts and lengths (0-based, [molecular frame](coordinates.md)) |
| `as` / `al` | MSP starts and lengths |
| `nq` | one byte per nucleosome (below) |
| `aq` | one byte per MSP: HMM posterior mean ×255, only from `fiberhmm-apply --scores` |

TF calls are **not** in `ns`/`nl` by default, only in `MA`/`AQ`. For a tool
that reads only the legacy tags, `--downstream-compat` writes TF calls into
`ns`/`nl` next to the nucleosomes (sorted by start; entries shorter than
`--unify-threshold`, 90 bp, are TFs) and writes no `MA`/`AQ`/`AN`. This loses
the per-TF quality bytes.

## MA and AQ

```text
MA:Z:<read_length>;nuc.QQQ:s1-l1,s2-l2,...;msp.:s1-l1,...;tf.QQQ:s1-l1,...
AQ:B:C,<bytes of every annotation in MA order>
```

Each group is `name`, a strand field (always `.` here) and a quality
specification: one `Q` per quality byte each annotation carries. Intervals
are `start-length` with a 1-based start. `AQ` concatenates the quality bytes
of every annotation in `MA` order; groups without `Q`s contribute none.

The groups `fiberhmm-call` writes:

| Group | Bytes | Meaning |
|---|---|---|
| `nuc.QQQ` | `nq`, `el`, `er` | nucleosomes after nucleosome recall (the default) |
| `nuc.Q` | `nq` | nucleosomes without nucleosome recall (`--no-recall-nucs`, and `fiberhmm-recall-tfs`) |
| `msp.` | none | methylase-sensitive patches |
| `tf.QQQ` | `tq`, `el`, `er` | TF / Pol II and other sub-nucleosomal footprints |

Later commands add their own groups (`ddda_mcg`, `ddda_ucg`, `deam+`/`deam-`,
`nuc_sr`, `tf_sr`, `tf_consensus`, `tf_recaller`); a re-call keeps groups it
does not regenerate. The header's
[`MA-TYPES`](../reference/headers.md#ma-types) line lists the group names a
BAM may contain.

## Quality bytes

### `tq`: TF evidence

```text
tq = min(255, round(10 × LLR))        LLR in nats
```

`tq` is the log-likelihood ratio of the footprint (protected against
accessible) under the emission table. Every 23 `tq` points are a factor of
ten in likelihood ratio.

| `tq` | LLR | Likelihood ratio |
|---|---|---|
| 50 | 5 nats | about 150 : 1 (the per-interval cost every preset calls at) |
| 100 | 10 nats | about 22,000 : 1 |
| 255 | ≥ 25.5 nats | saturated |

It is continuous evidence, not a posterior probability or an FDR, and its
scale is specific to the model version. Keep the calls and filter on `tq`
downstream (for example `fiberhmm-extract --min-tq 100`).

### `el`, `er`: edge sharpness

The recaller places each edge conservatively, just past the last unmarked
target. The true edge could extend up to the next mark. The edge bytes encode
that ambiguity:

```text
el = round(255 × max(0, 1 − left_ambiguity_bp / 30))
er = round(255 × max(0, 1 − right_ambiguity_bp / 30))
```

255 means a mark sits right at the edge (the size is exact); 0 means the
bracketing mark is 30 bp or more away (the size is a lower bound). For
example, `el, er ≥ 230` means at most 3 bp of ambiguity at either edge. For
DddA radial nucleosomes, the edge is the posterior median and `el`/`er`
encode the width of its central 90% interval on the same 30 bp scale.

### `nq`: nucleosome evidence

With nucleosome recall (the default), `nq` is the LLR ×10 of the retained
configuration, like `tq`; 0 means unresolved (for example a DddA nucleosome
with no radial dyad, or an unresolved boundary). Without nucleosome recall
it is the HMM posterior mean ×255 when `--scores` was given, the input's `nq`
when re-calling, or 0.

For DddA radial recall, `nq` scores the topology-changing linker
configuration (a split or an inward edge), not the easier question of
protected DNA against open linker, so it does not saturate on every
nucleosome.

## Reading the output in Python

```python
import pysam
from fiberhmm.io.ma_tags import flip_intervals_to_seq, parse_aq_array, parse_ma_tag

bam = pysam.AlignmentFile("calls.bam", "rb", check_sq=False)
for read in bam:
    if not read.has_tag("MA"):
        continue
    parsed = parse_ma_tag(read.get_tag("MA"))
    qual_specs = [group[2] for group in parsed["raw_types"]]
    n_per_group = [len(group[3]) for group in parsed["raw_types"]]
    per_annotation = parse_aq_array(read.get_tag("AQ"), qual_specs, n_per_group)
    # parsed["nuc"], parsed["msp"], parsed["tf"]: [(start, length), ...],
    #   0-based, molecular frame.
    # per_annotation: one list of quality bytes per annotation, in MA order:
    #   [nq, el, er] per nuc.QQQ call, [] per msp, [tq, el, er] per tf call.
    starts, lengths = zip(*parsed["tf"]) if parsed["tf"] else ((), ())
    tf_query = flip_intervals_to_seq(starts, lengths, read)   # query frame
```

The legacy tags hold the same nucleosome and MSP calls:

```python
ns, nl = list(read.get_tag("ns")), list(read.get_tag("nl"))
as_, al = list(read.get_tag("as")), list(read.get_tag("al"))
```

## Circular molecules

With `fiberhmm-call -r/--circular` (plasmids, mitochondrial DNA), each read
is tiled three times internally so features can cross the arbitrary read
origin, and calls are projected back onto the molecule. A feature that wraps
is written as two valid intervals, one at each end of the read, sharing an
`AN` name:

```text
MA:Z:1000;tf.QQQ:1-45,971-30
AQ:B:C,180,20,30,180,20,30
AN:Z:fhw_tf_0,fhw_tf_0
```

Tools that ignore `AN` still see two valid linear intervals. FiberBrowser and
`fiberhmm-extract --circular-groups` use `AN` to rebuild the wrapped feature.
Legacy tags are split the same way but carry no shared identity.
`fiberhmm-recall-nucs` handles linear reads only; for circular molecules use
`fiberhmm-call -r`.

## Stale tags

Records a run skips (filtered, secondary or supplementary, no observations)
are written without the previous run's call tags, so a re-call never leaves
an old call next to a new one. DddA island groups (`ddda_mcg`, `ddda_ucg`) are
kept.
