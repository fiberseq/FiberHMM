# Paired-duplex scDAF integration

`fiberhmm-crossstrand` is an optional DddA scDAF workflow for two sequenced
copies inferred to come from the same physical duplex. In this document,
"consensus" means only that within-duplex evidence union. It never means a
population TF-class consensus across independent molecules.

## Pipeline

```text
ordinary called reads
        |
        v
sequence-first CT/GA pairing ----> footprint fallback for sequence ambiguity
        |
        v
one reference-frame joint molecule with deam+ and deam- source coverage
        |
        v
ordinary HMM + nucleosome recaller + TF recaller on both channels
```

The one-command form is:

```bash
fiberhmm-crossstrand \
  -i calls.bam \
  -o duplex-recalled.bam \
  --pairs-tsv duplex-pairs.tsv \
  --pairs-only
```

The input must already contain ordinary FiberHMM calls and be coordinate
sorted and indexed. A reference FASTA supplied with `--reference` provides
deamination-safe A/T sequence for pairing. Without a FASTA, the same evidence
is reconstructed from MD+CIGAR when possible. Reads that remain
sequence-ambiguous can be assigned by reciprocal nucleosome-pattern matching.

## Pair evidence

Paired source reads retain explicit tags:

| tag | meaning |
|---|---|
| `mt:A:P` | accepted pair member |
| `mp:Z` | reciprocal mate query name |
| `pm:A:S/F` | sequence assignment or footprint fallback |
| `pa:A:R/C` | strict reciprocal sequence edge or constrained complete 2x2 assignment |
| `sb`, `sd`, `sr`, `sg` | safe sequence bases, differences, rate, and assignment margin |
| `mc`, `mg` | footprint correlation and footprint-selection margin (`mg` only for fallback pairs) |

Query names must be unique among primary featurizable reads. The pairer and
merger fail closed on duplicate names, missing mates, same-flavor pairs, or
non-reciprocal tags.

Sequence assignment is independent of footprint similarity. Footprint
fallback is useful for the remaining ambiguous reads, but any later analysis
of footprint agreement must stratify by `pm`: the fallback stratum was selected
partly for footprint similarity and is therefore not an independent
concordance benchmark.

The default sequence preference margin is 0.002 in difference-rate units. It
is intentionally conservative: a weak one-base preference is left for the
footprint fallback rather than promoted as orthogonal pairing evidence.

## Joint evidence and coordinates

The joint record is forward/reference-frame and spans the union of its source
alignments. CT deaminations are encoded as `Y`, GA deaminations as `R`, and the
exact aligned-reference coverage of each source is recorded as `deam+` and
`deam-` MA intervals. Source deletions split those intervals and therefore
remain missing evidence; a base observed only on the mate cannot become a miss
on the deleted source strand.

The current joint record uses an all-`M` CIGAR. Insertions are omitted and
deletions occupy reference-frame `N` positions. This makes reference
coordinates exact for substitutions and footprints, but it is not an
indel-preserving sequence consensus. Indel-sensitive applications should use
the two source alignments.

The caller constructs separate CT and GA DAF observations and applies their
coverage masks before merging them. In their overlap, reference C and G
positions both contribute. In a single-source flank, bases for the absent
source are non-target/missing rather than protected. Before 7-mer contexts are
encoded, deaminations from both channels are reverted to their canonical bases;
an event on one strand therefore cannot corrupt the context code of a nearby
target on the other strand.

Canonical source-base disagreements are encoded as `N`. Joint records do not
invent base qualities and do not carry an `NM` edit-distance tag. The local
`dc:i` and `bc:i` tags instead report the deamination count and the number of
source-base conflicts masked as `N`. Footprint confidence remains in `AQ`.

## Output quality

Joint `tf.QQQ` retains the ordinary FiberHMM quality contract:

- `tq = round(10 * LLR_nats)`, saturated to a byte;
- `el` and `er` encode boundary ambiguity, with
  `round(255 * (1 - ambiguity_bp / 30))` and zero at 30 bp or greater.

These are model-evidence and edge-resolution quantities, not truth
probabilities. For example, `tq >= 100` means LLR >= 10 nats and
`el,er >= 230` means no more than 3 bp of ambiguity at either edge.

## Scientific boundary

Appropriate claims are technical and molecule-local: the joint caller used
both observed channels; a call was visible on CT, GA, both, or only after
combining evidence; and the result changed predictably with local opportunity
and quality thresholds.

The joint callset is a higher-information silver reference, not biological
ground truth. A joint-only call is a combined-evidence nomination, not a
proven rescue. A local alternative-mate control—re-calling an overlapping,
non-inferred CT/GA combination under identical parameters—is required to quantify the generic
effect of doubling opportunity density. Without physical labels, the alternative
mate is a null assignment, not a known wrong duplex. Low-coverage scDAF does not authorize a locus-population
TF-class prior. Population class learning belongs to separately evaluated,
deeply covered focal cohorts such as high-coverage Fiber-seq or targeted DAF.
