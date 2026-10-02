# BAM tags

Every per-read tag FiberHMM writes. Intervals are 0-based in the
[molecular frame](../concepts/coordinates.md) (in `MA` strings the start is
written 1-based). The meaning of the scores is explained in
[Annotations and scores](../concepts/annotations.md).

## Footprint calls

Written by `fiberhmm-call`, `fiberhmm-apply` (legacy tags only),
`fiberhmm-recall-tfs`/`-recall-nucs`, and the joint recall of
`fiberhmm-pair`/`fiberhmm-merge --recall`.

| Tag | Type | Content |
|---|---|---|
| `ns` / `nl` | `B:I` | nucleosome starts / lengths; with `--downstream-compat` also TF calls |
| `as` / `al` | `B:I` | MSP starts / lengths (`fiberhmm-apply --no-msps` omits them) |
| `nq` | `B:C` | one byte per `ns` entry: with nucleosome recall the LLR ×10 (0 = unresolved); otherwise the HMM posterior mean ×255 (`--scores`), the input's `nq`, or 0 |
| `aq` | `B:C` | one byte per MSP: mean HMM posterior P(accessible) over the MSP ×255 (high = confidently accessible); only `fiberhmm-apply --scores` |
| `MA` | `Z` | molecular annotations (groups below) |
| `AQ` | `B:C` | quality bytes of every `MA` annotation, in `MA` order |
| `AN` | `Z` | one name per `MA` annotation (`.` for unnamed); written when a call wraps a circular origin (`fhw_*`), for strand-rescue roles, consensus class tokens, or when preserved groups carry names |

FiberHMM also reads fibertools-rs 0.13+ `Ma`/`Aq`/`An` tags (same spec,
always molecular frame) as input; it never writes them. See
[Coordinate frames](../concepts/coordinates.md#fibertools-ma-tags).

`--no-legacy-tags` writes only `MA`/`AQ` (existing legacy tags are left as
they were); `--downstream-compat` writes TF calls into `ns`/`nl` and no
`MA`/`AQ`/`AN`. Records a run skips are written without this run's call tags;
DddA island groups are kept.

## `MA` groups

| Group | `AQ` bytes | Written by | Meaning |
|---|---|---|---|
| `nuc.QQQ` | `nq`, `el`, `er` | nucleosome recall (`fiberhmm-call` default, `recall-nucs`, duplex recall) | recalled nucleosomes |
| `nuc.Q` | `nq` | `recall-tfs`, `call --no-recall-nucs` | nucleosomes without recall |
| `msp.` | — | all callers | methylase-sensitive patches |
| `tf.QQQ` | `tq`, `el`, `er` | all callers | TF / Pol II and other sub-nucleosomal footprints |
| `ddda_mcg.` | — | `fiberhmm-tag-m5c`, `fiberhmm-call-m5c --tag-bam` | whole CpG island, confidently methylated on this molecule |
| `ddda_ucg.` | — | same | whole CpG island, confidently unmethylated (the CpG-aware recall whitelist) |
| `deam+` / `deam-` | — | `fiberhmm-pair` merge | CT- and GA-source aligned coverage of a joint duplex molecule |
| `nuc_sr.QQQ`, `tf_sr.QQQ` | `q0`, `q1`, `q2` | `fiberhmm-strand-rescue-annotate` | strand-rescue shadow layers ([Strand rescue](../workflows/strand-rescue.md#shadow-layers)) |
| `tf_sr.QQQQQ` | as above plus `fi`, `fq` | `fiberhmm-tag-consensus` | family slot (0 = unassigned) and assignment confidence ×255 |
| `tf_consensus.QQQQQQ` | `tq`, `fi`, `fq`, `op`, `sq`, `q0` | `fiberhmm-consensus`, `fiberhmm-transfer` | native TF calls labelled with their class ([Consensus](../workflows/consensus.md#family-tagged-bams)) |
| `tf_recaller.QQQQQQ` | `tq`, `fi`, `tier`, `q0`, `lr`, `rr` | `fiberhmm-consensus --bam-recaller-layer` | the lattice recaller's own class calls |
| `tf_cross_consensus.QQQQQQ` | as `tf_consensus` | `fiberhmm-consensus --engine staged_native_families` (XCR) | deprecated staged engine |

The strand field of every FiberHMM group is `.`, except the duplex
coverage groups of `fiberhmm-pair`, whose strand is `+` (`deam+`, CT source)
or `-` (`deam-`, GA source). `ddda_mcg+`/`-` and
`ddda_mcg_hemi+`/`-` from older experimental per-CpG callers are still read
but no longer written.

## Quality bytes

| Byte | Scale |
|---|---|
| `tq` | `min(255, round(10 × LLR))`; 23 points = one order of magnitude in likelihood ratio |
| `nq` | as `tq` for recalled nucleosomes (0 = unresolved) |
| `el`, `er` | `round(255 × max(0, 1 − ambiguity_bp / 30))`, molecular left / right edge |
| `q0`, `q1`, `q2` (strand rescue) | linear 0–255 probabilities; see [Strand rescue](../workflows/strand-rescue.md#shadow-layers) |
| `fi` | family / class slot, 1–255 (0 = none) |
| `fq` | assignment confidence ×255 (0 = unavailable in consensus exports) |
| `op` | informative opportunities in the call |
| `sq` | DAF core protection ceiling of the molecule (1 + LLR × 10) |
| `q0` (consensus) | lattice recaller: the molecule's EM class posterior ×255; staged engine: the class's share of the call's evidence ×255 |
| `tier` | 1 core, 2 edge, 3 loose |
| `lr`, `rr` | left / right edge-range width in bp (0 = exact edge), saturated at 255 |

## DAF tags

| Tag | Type | Writer | Content |
|---|---|---|---|
| `st` | `Z` | `fiberhmm-daf-encode` | conversion flavour: `CT` or `GA`; `SEQ` then carries Y/R codes. `fiberhmm-call`'s one-pass DAF path derives the flavour internally and does not write it |
| `di` | `i` | `fiberhmm-dedup`, integrated dedup | duplicate-cluster id, on every member of a cluster of two or more |
| `ds` | `i` | same | cluster size (copies of the molecule in the input) |

Duplicates are also flagged `0x400` (standalone `--flag-only`, and the
integrated default); a collapse keeps only the representative, which keeps
`di`/`ds`.

## Duplex pairing tags

| Tag | Type | On | Content |
|---|---|---|---|
| `mt` | `A` | source reads | `P` pair member, `U` had candidates but failed, `.` no opposite-flavour candidate |
| `mp` | `Z` | pair members | the mate's query name |
| `pm` | `A` | pair members, joint molecule | route: `S` sequence, `D` sequence-free model (`F`: pre-3.0 footprint pairer) |
| `pa` | `A` | sequence pairs | `R` reciprocal, `C` constrained 2×2 |
| `sb` | `i` | sequence pairs | shared reference-A/T bases |
| `sd` | `i` | sequence pairs | differences at those bases |
| `sr` | `i` | sequence pairs | difference rate ×1,000,000 |
| `sg` | `i` | sequence pairs | assignment margin ×1,000,000 |
| `mc` | `i` | sequence pairs (optional) | nucleosome dyad correlation ×1000 |
| `dm` | `i` | model pairs | decision score ×1000 |
| `mg` | `i` | model pairs | reciprocal decision margin ×1000 |
| `mv` | `Z` | model pairs | model identifier |
| `cs` | `Z` | joint molecule | source names, `<ct_name>;<ga_name>` |
| `dc` | `i` | joint molecule | deamination count (C→T and G→A) |
| `bc` | `i` | joint molecule | source-base conflicts written as `N` |

The joint molecule (`<ct_name>.cs`) has no `mt`/`mp` and carries its route's
evidence tags.

## Tags FiberHMM reads

| Tag | Used for |
|---|---|
| `MM`/`ML` (or `Mm`/`Ml`) | m6A (`A+a`, `T-a`), 5mC and dU calls; `?`-mode unlisted bases are unknown, not unmodified |
| `MD` | DAF deaminations when `SEQ` has no R/Y |
| `MN` | detects hard-clipped records whose `MM`/`ML` cannot match `SEQ` |
| `st` | DAF flavour of R/Y-encoded reads |
| `HP`, `PS` | copied by `fiberhmm-extract --haplotype-fields` |
| `ns`/`nl`/`as`/`al`, `MA`/`AQ`/`AN` | input footprints for recall, extraction, QC and every downstream command |
