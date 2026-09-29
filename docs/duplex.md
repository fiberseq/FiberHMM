# Sequence-free scDAF duplex pairing

The sequence-free route inside `fiberhmm-pair` predicts which CT- and GA-flavor
reads came from the same physical duplex without using sequence identity. It is
used alongside direct sequence-supported assignments by default. Its route
label is `pm:D`; sequence-supported assignments carry `pm:S`.

```bash
fiberhmm-pair \
  -i sample.fiberhmm.bam \
  -r hg38.fa \
  -o sample.duplex.bam \
  --pairs-tsv sample.duplex.tsv \
  --receipt-json sample.duplex.json
```

The input must be coordinate sorted, indexed, DddA called, and contain an `MA`
`nuc` track. The reference FASTA is used only to enumerate non-CpG C/G enzyme
opportunities. The scorer never derives or retains A/T allele agreement.

## Evidence and selection

Each candidate must overlap by at least 1,500 bp and contain at least four
nucleosome dyads on both reads. The frozen linear model combines:

- lag-tolerant smoothed dyad correlation;
- dyad-anchor F1 at three tolerance/shift settings;
- overlap length, dyad count and dyad-count balance;
- aligned-span Jaccard overlap;
- the 20-bp-binned non-CpG DddA protection correlation; and
- the protection correlation after subtracting the opportunity-weighted pattern
  shared by all reads in the complete local overlap component.

The model has no A/T mismatch, haplotype, TF LLR, or sequence-veto input.
In the default combined workflow, direct sequence-supported assignments are
selected independently and take precedence if the two routes consume the same
read. `--sequence-only` disables this model route completely.
Within each complete overlap component, an edge must be best for both reads.
The default two-sided decision-score margin is 1.0 against the next candidate
or a virtual null score of zero. Reads that fail this gate remain unresolved.

The BAM retains all input records by default. Pair members carry `mt:A:P`,
their reciprocal partner in `mp:Z`, `pm:A:D`, the decision score in `dm:i`
(×1,000), margin in `mg:i` (×1,000), and model ID in `mv:Z`. Candidate reads
that abstain carry `mt:A:U`. `--paired-only` writes only selected primary source
records and remains compatible with `fiberhmm-pair --from-paired`.

## Nucleosome call layers

`--call-layer auto` is the default. BAMs declaring
`nuc_model=ddda_phase_posterior_v1` use the calibration fit after FiberHMM
FiberHMM 3.0 phase-marginal rotational DddA recall. Older BAMs use the archived-MA
calibration. `--call-layer rotational-recall` or `--call-layer input-ma` is
available when provenance was stripped during BAM processing.

The two calibrations were trained on independent A/T-selected mates in GRCh38
chr1:[0,40.1 Mb) and evaluated once on the disjoint interval
chr1:[40.5,80.5 Mb) in the same 11 scDAF libraries:

| MA call layer | Competitive true edge first | Margin-1 outcomes |
|---|---:|---:|
| archived input MA | 72/80 (90.0%) | 51 correct, 4 wrong, 38 abstained |
| FiberHMM 3.0 rotational recall | 73/80 (91.3%) | 48 correct, 5 wrong, 40 abstained |

The rotational layer rescued one competitive rank error and introduced no rank
regressions on the 80 common external cases. Its margin-1 set is slightly less
precise, which is why the two layers retain separate coefficients rather than
sharing one model.

These are observed validation rates, not a universal error guarantee. The raw
DddA profile is part of the pairing score and also underlies footprint calls;
tests of downstream footprint transfer should therefore use disjoint genomic
windows or disjoint molecules.
