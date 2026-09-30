# Duplex: pairing the two strands of a molecule

DddA deaminates cytosines on both strands of a DNA duplex. After
denaturation the two strands can be sequenced as two separate reads of
opposite flavour: a CT read, informative only at reference C, and a GA read,
informative only at reference G. Each read alone is blind wherever its base is
sparse. `fiberhmm-pair` finds the two reads of one molecule, merges them into
one both-strand molecule, and re-calls footprints on the joint evidence, where
both reference C and reference G are informative.

"Consensus" on this page means only this within-molecule union of the two
strands, never a population consensus across molecules (that is
[Footprint classes](consensus.md)).

## Run it

The input is a DddA BAM already called by `fiberhmm-call`, coordinate-sorted
and indexed, with a matching indexed FASTA:

```bash
fiberhmm-pair -i out/ddda.calls.bam -r demo/ref.fa -o out/ddda.duplex.bam \
    --pairs-tsv out/ddda.pairs.tsv --receipt-json out/ddda.pairs.json
```

```text
fiberhmm-pair: stages pair -> merge -> recall (use --stop-after pair to write tagged pairs only)
merge-recall: loaded ddda apply+recall models (k=3); re-calling footprints (HMM + nuc + TF recallers) on both-strand consensus reads; CpG-aware recall unmethylated-only
resolved pairs seen: 28  | consensus reads built: 28  | build failures (MD/coverage): 0
...
fiberhmm-pair: 28 pairs [sequence 0; sequence-free 28] (52 marked PCR duplicates excluded) in 4.2s
```

The output holds one joint molecule per pair (replacing its two source
reads) and every other read unchanged, sorted and indexed.

| Option | Effect |
|---|---|
| `--stop-after pair` | only tag the pairs on the source reads (every input record is written) |
| `--stop-after merge` | build joint molecules without re-calling them |
| `--from-paired` | start from a BAM tagged by `--stop-after pair`; pairing options are rejected |
| `--pairs-only` | write only paired records (joint molecules, or pair members with `--stop-after pair`) |
| `--sequence-only` | accept only sequence-supported pairs (below); the FASTA becomes optional if reads have `MD` |
| `--pairs-tsv`, `--receipt-json` | evidence for every selected pair, and a machine-readable run receipt |

```bash
fiberhmm-pair -i out/ddda.calls.bam -r demo/ref.fa -o out/ddda.paired.bam --stop-after pair
fiberhmm-pair -i out/ddda.paired.bam -o out/ddda.duplex2.bam --from-paired
```

Reads flagged as PCR duplicates (`0x400`) are never pairing candidates. The
MM/ML dU threshold `-p/--prob-threshold` is 128. `fiberhmm-merge` is the
older standalone merge step, still installed but deprecated: use
`fiberhmm-pair --from-paired`. `fiberhmm-crossstrand` no longer exists.

## How pairs are chosen

Two independent routes run within each connected group of overlapping reads.

### Sequence-supported pairs (`pm:S`)

DddA never changes reference A or T, so at those positions the read shows the
molecule's genotype. For each pair of opposite-flavour reads, FiberHMM
compares their bases at shared reference A/T positions (reference bases from
the FASTA, or reconstructed from `MD`+CIGAR with `--sequence-only`; R/Y are
read as C/G), giving shared bases (`sb`), differences (`sd`) and a
difference rate (`sr`). A pair needs at least 500 shared A/T bases
(`--min-sequence-bases`) and is accepted as:

- **reciprocal** (`pa:R`): each read's lowest-rate partner is the other, the
  rate is ≤ `--max-sequence-pair-rate` (0.01), and both reads have a competing
  partner at least `--min-sequence-margin` (0.002) worse; or
- **constrained 2×2** (`pa:C`): in a group of exactly two CT and two GA reads,
  the diagonal with the lower summed rate, if the diagonals differ by at least
  0.002, both chosen edges are ≤ 0.01, and a rejected edge exceeds
  `--min-component-discordance-rate` (0.02), a clear allele conflict.

This route is precise but needs a distinguishing allele in the overlap and
competition nearby. Longer chains are never solved jointly.

### Sequence-free pairs (`pm:D`)

The remaining pairs come from a frozen linear model that uses no sequence
identity. A candidate must overlap by at least 1,500 bp (`--min-overlap`),
have at least four nucleosome dyads on each read (`--min-nucs`) and enough
comparable 20 bp bins. The model combines:

- lag-tolerant nucleosome-dyad correlation (dyads from the reads' `MA` `nuc`
  calls as Gaussian bumps on a 10 bp grid, compared over shifts up to
  ±60 bp);
- dyad-anchor F1 at three tolerance/shift settings;
- overlap length, dyad count and dyad-count balance;
- aligned-span Jaccard overlap;
- the 20 bp-binned non-CpG DddA protection correlation, raw and after
  removing the pattern shared by every read of the local group.

The FASTA is used only to enumerate non-CpG DddA opportunities. A pair must
be the best choice for both reads by a decision-score margin of at least
`--min-margin` (1.0) over the next candidate or a virtual null
(`--null-floor`, 0.0).

Sequence pairs take precedence: a model pair that would reuse a read from a
sequence pair is dropped. A `pm:D` pair is not checked against the A/T
genotype and may carry an allele conflict; analyses that need
genotype-consistent pairs should use `--sequence-only` or keep `pm:S` pairs.
Because the model uses nucleosome and protection concordance, later analyses
of those features must stratify by `pm` and must not treat `pm:D` pairs as an
independent concordance benchmark.

**Calibration layers.** `--call-layer auto` (default) uses the coefficients
fit on FiberHMM 3.0 rotational DddA nucleosome calls when the BAM declares
`nuc_model=ddda_phase_posterior_v1`, and the archived-`MA` coefficients
otherwise; `rotational-recall` and `input-ma` force one. Both were trained on
A/T-selected mates in GRCh38 chr1:[0, 40.1 Mb) and evaluated once on the
disjoint chr1:[40.5, 80.5 Mb) in the same 11 scDAF libraries:

| Call layer | True edge ranked first | Assignments at margin 1 |
|---|---:|---|
| archived input MA | 72/80 (90.0%) | 55: 51 correct, 4 wrong (92.7%) |
| 3.0 rotational recall | 73/80 (91.3%) | 53: 48 correct, 5 wrong (90.6%) |

These are observed validation rates, not guarantees. The model files
(`ddda_duplex_v1.json`, `ddda_duplex_rotational_v1.json`) carry the status
`externally_replicated_experimental`; `--model` supplies another.

## The joint molecule

Each pair becomes one record named `<ct_read>.cs`:

- forward, reference-frame, spanning the union of the two alignments, with an
  all-`M` CIGAR and MAPQ equal to the lower of the two;
- CT deaminations written as `Y`, GA deaminations as `R`; a base the two
  sources disagree on is `N`; insertions are omitted, and a base deleted in
  one source takes the mate's base where the mate covers it (so coordinates
  are exact for substitutions and footprints, but this is not an
  indel-preserving consensus sequence);
- `MA` groups `deam+` and `deam-` record the exact aligned coverage of the CT
  and GA sources; their intersection is the both-strand region, and a base
  seen only on one source is missing, not protected, on the other;
- tags `cs:Z` (`<ct_name>;<ga_name>`), `dc:i` (deamination count), `bc:i`
  (source-base conflicts written as `N`) and the pair-evidence tags of its
  route; no `mt`/`mp`, no base qualities, no `NM`.

## Joint recall

The joint recall runs the same stack as `fiberhmm-call --enzyme ddda`: HMM,
DddA radial nucleosome recall (`--phase-nrl 196`,
`--nuc-recall-policy conservative`) and TF recall
(`--ddda-derived-tf-max-edge-gap 12`). The observation merges a C-target pass
masked to `deam+` with a G-target pass masked to `deam-`; before each
channel's contexts are encoded, the other channel's deaminations are reverted
so one strand's events cannot corrupt the other's context codes. The DddA
keep-one run mask applies (`--daf-mask-runs`), and CpG-aware recall is on
(`--no-use-m5c` to turn off): an island called on either source read is
projected onto the joint molecule. The output declares `MA-TYPES` `deam`
next to `nuc,msp,tf`.

Joint `tf.QQQ` keeps the ordinary quality contract (`tq = 10 × LLR`,
`el`/`er` edge sharpness). These are evidence and edge-resolution
quantities, not probabilities of truth.

## Pair tags

On source reads (`--stop-after pair`, and unpaired reads in the default
output):

| Tag | Type | Meaning |
|---|---|---|
| `mt` | A | `P` accepted pair member, `U` had candidates but failed the gate, `.` no opposite-flavour candidate |
| `mp` | Z | the mate's query name |
| `pm` | A | route: `S` sequence, `D` sequence-free model (`F` = pre-3.0 footprint pairer, still accepted when merging) |
| `pa` | A | sequence pairs: `R` reciprocal, `C` constrained 2×2 |
| `sb`, `sd` | i | sequence pairs: shared A/T bases, differences |
| `sr`, `sg` | i | sequence pairs: difference rate and assignment margin, ×1,000,000 |
| `mc` | i | sequence pairs (optional): nucleosome dyad correlation ×1000 |
| `dm`, `mg` | i | model pairs: decision score and reciprocal margin, ×1000 |
| `mv` | Z | model pairs: model identifier |

The joint molecule carries `cs`, `dc`, `bc` and its route's evidence tags
(`pm`, `pa`, `sb`, `sd`, `sr`, `sg`, `mc` or `pm`, `dm`, `mg`, `mv`).

Query names must be unique among primary reads; the pairer and merger stop
on duplicate names, missing mates, same-flavour pairs or non-reciprocal tags.

## Viewing the both-strand region

```bash
fiberhmm-extract -i out/ddda.duplex.bam -o out/duplex_tracks --both-strand --bed-only
```

writes `ddda.duplex_bothstrand.bed`, one BED12 row per joint molecule over
the `deam+` ∩ `deam-` region.

## Interpreting joint calls

On an archived validation set (800 joint molecules, earlier pairing and
recall stack), both-strand recall gave about twice the TF-call yield of
either strand alone, and the share of joint TF calls also resolved by one
strand followed base composition (for example 53% by CT alone and 25% by GA
alone in strongly C-rich windows). The joint call set is a
higher-information silver reference, not ground truth: a joint-only call is a
combined-evidence nomination. Quantifying the generic gain from doubling
opportunity density needs an alternative-mate control (re-calling an
overlapping, non-inferred CT/GA combination with identical parameters).
Low-coverage scDAF does not justify a population TF-class prior; population
class learning belongs to deeply covered focal cohorts
([Footprint classes](consensus.md)).

Every option: [`fiberhmm-pair`](../reference/cli.md#fiberhmm-pair),
[`fiberhmm-merge`](../reference/cli.md#fiberhmm-merge).
