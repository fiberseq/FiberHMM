# Paired-duplex scDAF integration

`fiberhmm-pair` is the DddA scDAF workflow for two sequenced
copies inferred to come from the same physical duplex. In this document,
"consensus" means only that within-duplex evidence union. It never means a
population TF-class consensus across independent molecules.

## Why pair the two strands

DddA deaminates cytosine on both strands of a duplex. After denaturation the
two strands are sequenced as separate reads of opposite flavor: a CT read
(C->T; informative only at reference C) and a GA read (G->A; informative only
at reference G). A footprint is read out as a run of target bases that were
not deaminated. Because each strand reports only at its own base, a single
read is blind wherever that base is sparse; merging the two reads of one
molecule makes both reference C and reference G informative in their overlap.
The two reads cannot be matched by deamination pattern, because they mark
disjoint bases.

## Pipeline

```text
ordinary called reads
        |
        v
sequence-supported assignment + high-confidence sequence-free model
        |
        v
one reference-frame joint molecule with deam+ and deam- source coverage
        |
        v
ordinary HMM + nucleosome recaller + TF recaller on both channels
```

One command runs all three stages (pair -> merge -> recall) by default:

```bash
fiberhmm-pair \
  -i calls.bam \
  -o duplex-recalled.bam \
  -r hg38.fa \
  --pairs-tsv duplex-pairs.tsv \
  --pairs-only
```

`--stop-after pair` writes every input record with pair tags on the pair
members (add `--pairs-only` to write only the pair members), and
`--stop-after merge` merges without re-calling. `--from-paired` starts from an
already paired BAM; it rejects pairing-stage options and cannot be combined
with `--stop-after pair`. `fiberhmm-pair` replaces the former
`fiberhmm-crossstrand` command, which is no longer installed.
`fiberhmm-merge` is still installed but deprecated; use
`fiberhmm-pair --from-paired` for an already paired BAM. Reads flagged as PCR
duplicates (`0x400`) are never pairing candidates and pass through without pair
tags. `-p/--prob-threshold` (default 128) applies to MM/ML-native dU calls
only.

The input must already contain ordinary FiberHMM calls and be coordinate
sorted and indexed. The default workflow requires a reference FASTA (every BAM
contig, with matching lengths) because the sequence-free score enumerates
non-CpG DddA opportunities. The FASTA also supplies deamination-safe A/T
evidence for direct sequence-supported assignment. `--sequence-only` disables
the model route and can reconstruct A/T evidence from MD+CIGAR when the FASTA
is omitted.

## Sequence-supported assignment (`pm:S`)

At reference A/T positions DddA changes nothing, so the query base there is the
molecule's genotype. For each read, FiberHMM collects (reference position,
query base) pairs at reference A/T sites only; reference C/G sites are dropped,
and query Y/R are canonicalized to C/G so a genuine alternate allele still
counts. With `--reference` the reference base comes from the FASTA; without it
(`--sequence-only` only) it is reconstructed from MD+CIGAR, and a read with no
MD contributes no sequence evidence. Two opposite-flavor reads are compared
over their shared A/T sites to give shared bases (`sb`), differences (`sd`) and
a difference rate (`sr`, x1,000,000).

An edge is sequence-comparable when the reads share at least 500 A/T sites
(`--min-sequence-bases`); no genomic-overlap minimum applies to this route.
Two assignment kinds are accepted:

- **Reciprocal (`pa:R`)**: each read's lowest-rate admissible partner is the
  other, the rate is at most `--max-sequence-pair-rate` (0.01), and both reads
  have a competing admissible edge whose rate is higher by at least
  `--min-sequence-margin` (0.002). Edges above 0.01 are neither selected nor
  counted as competitors.
- **Constrained 2x2 (`pa:C`)**: in a component of exactly two CT and two GA
  reads with all four edges comparable, the diagonal with the lower summed rate
  is taken only if the diagonals differ by at least 0.002, both chosen edges are
  at most 0.01, and a rejected edge exceeds `--min-component-discordance-rate`
  (0.02), i.e. a clear allele conflict rules the other diagonal out.

Larger overlap chains are never solved jointly, so a weak local choice cannot
force assignments along a chain. This route is precise but limited in recall:
it resolves a pair only where a distinguishing allele falls in the overlap and
there is local competition.

## Sequence-free assignment (`pm:D`)

The remaining pairs come from the frozen sequence-free model (see
[duplex.md](duplex.md)). One of its inputs is the lag-tolerant nucleosome dyad
correlation: each read's MA `nuc` dyads are placed on a 10-bp reference grid as
Gaussian bumps (sigma 30 bp), and the two tracks are compared by normalized
cross-correlation over their overlap, maximized over shifts of up to +/-60 bp.
The route requires >= 1,500 bp overlap and >= 4 dyads per read in the overlap.
The model has no A/T, haplotype, TF-LLR or sequence-veto input.

Both routes run independently within each complete local overlap component.
Sequence pairs take precedence: a model pair that would reuse a read from a
sequence pair is dropped, and the other read of that model edge stays
unresolved (`mt:U`). A `pm:D` pair is therefore not checked against A/T
genotype and may carry an allele conflict; analyses that need
genotype-consistent pairs should use `--sequence-only` or keep `pm:S`.
`--min-margin` and `--null-floor` are in model decision-score units (defaults
1.0 and 0.0), not correlation units.

## Pair evidence

Paired source reads retain explicit tags:

| tag | meaning |
|---|---|
| `mt:A:P` | accepted pair member |
| `mp:Z` | reciprocal mate query name |
| `pm:A:S/D` | direct sequence assignment or frozen sequence-free model |
| `pa:A:R/C` | strict reciprocal sequence edge or constrained complete 2x2 assignment |
| `sb`, `sd`, `sr`, `sg` | safe sequence bases, differences, rate, and assignment margin |
| `mc` | optional nucleosome correlation for a sequence-supported pair |
| `dm`, `mg`, `mv` | sequence-free decision score, reciprocal margin (x1000), and model ID |
| `mt:A:U` / `mt:A:.` | candidate that failed the gate / no opposite-flavor candidate |

`mg` is the model margin on `pm:D` pairs; BAMs from the pre-3.0 footprint
pairer can instead carry `pm:F` with a correlation margin, which the merger
still accepts.

Query names must be unique among primary featurizable reads. The pairer and
merger fail closed on duplicate names, missing mates, same-flavor pairs, or
non-reciprocal tags.

Sequence assignment is independent of footprint similarity. The sequence-free
route uses nucleosome and DddA protection concordance, so later analyses of
those features must stratify by `pm` and must not present `pm:D` as an
independent concordance benchmark.

The default sequence preference margin is 0.002 in difference-rate units. It
is intentionally conservative: a weak one-base preference is left unresolved
by the sequence route rather than promoted as orthogonal pairing evidence.

## Joint evidence and coordinates

The joint record is forward/reference-frame, named `<ct_read>.cs`, and spans
the union of its source alignments. Its MAPQ is the lower source MAPQ. It
carries `cs:Z` (`<ct_name>;<ga_name>`), `dc:i`, `bc:i`, the `deam+`/`deam-`
MA coverage track and the pairing evidence tags (`pm`, `pa`, `sb`, `sd`, `sr`,
`sg`, `mc` for sequence pairs; `pm`, `dm`, `mg`, `mv` for model pairs), but not
`mt` or `mp`. By default the two source reads are replaced by this record and
all other reads pass through unchanged, keeping their `mt` status tags.
`deam+` AND `deam-` is the both-strand region. CT deaminations are encoded as `Y`, GA deaminations as `R`, and the
exact aligned-reference coverage of each source is recorded as `deam+` and
`deam-` MA intervals. Source deletions split those intervals and therefore
remain missing evidence; a base observed only on the mate cannot become a miss
on the deleted source strand.

The current joint record uses an all-`M` CIGAR. Insertions are omitted; a
position deleted in one source takes the mate's base when the mate covers it,
and is `N` only when neither source covers it or their canonical bases
conflict. This makes reference
coordinates exact for substitutions and footprints, but it is not an
indel-preserving sequence consensus. Indel-sensitive applications should use
the two source alignments.

The caller constructs separate CT and GA DAF observations and applies their
coverage masks before merging them. In their overlap, reference C and G
positions both contribute. In a single-source flank, bases for the absent
source are non-target/missing rather than protected. Before each channel's
7-mer contexts are encoded, the other channel's deaminations are reverted to
their canonical bases; an event on one strand therefore cannot corrupt the
context code of a nearby target on the other strand.

## Joint recall

The joint recall runs the same stack as `fiberhmm-call` for DddA: HMM apply,
the nucleosome recaller (DddA nucleosome-refinement table and profile,
`--phase-nrl 196`, `--nuc-recall-policy conservative`) and the TF recaller
(`--ddda-derived-tf-max-edge-gap 12`). The observation merges a C-target pass
masked to `deam+` with a G-target pass (reverse-complemented into the same
context-code space) masked to `deam-`. Reference C and G never share a
position, so no recall math is special-cased. The DddA CC/GG keep-one run mask
applies (`--daf-mask-runs`). CpG-aware recall is on by default (`--use-m5c`;
`--no-use-m5c` for an ablation): CpG observations are excluded except inside
`ddda_ucg` islands. An island called on either source read (by
`fiberhmm-tag-m5c`) is projected onto the joint record, so the joint molecule
uses the union of the two strands' island calls.

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

## Strand asymmetry (archived measurement)

Because a CT read reports only at reference C and a GA read only at reference
G, a footprint in a base-skewed window is resolvable mostly on one strand. On
the archived validation set (800 joint records recalled both-strand, CT-only
and GA-only, under the earlier pairing and recall stack), the fraction of
both-strand TF calls also resolved by one strand was:

| window composition | CT-only resolves | GA-only resolves |
|---|---|---|
| strong G-rich | 24% | 55% |
| mild G | 35% | 44% |
| balanced | 40% | 39% |
| mild C | 45% | 35% |
| strong C-rich | 53% | 25% |

On that set, both-strand recall gave about twice the TF-call yield of either
single strand. The archived check of the both-strand encoding on real reads
found 0 strand-swap disagreements, 0 C/G overlap and all LLRs finite. These
figures predate the current pairing route and CpG-aware recall.

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
