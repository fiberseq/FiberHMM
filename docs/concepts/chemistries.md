# Chemistries and platforms

## Support matrix

**Supported** means a bundled, validated model selected by `--enzyme`, with
defaults tuned for it and coverage in the test suite.

| Chemistry | Platform | Status | Preset | Notes |
|---|---|---|---|---|
| Hia5 (m6A) Fiber-seq | PacBio | supported | `--enzyme hia5 --seq pacbio` | m6A on both strands (`A+a`, `T-a`) |
| Hia5 (m6A) Fiber-seq | Nanopore | supported | `--enzyme hia5 --seq nanopore` | m6A on the sequenced strand (`A+a`); strict ML ≥ 248 |
| DddB DAF-seq | any | supported | `--enzyme dddb` | double-stranded deaminase; one strand per read |
| DddA DAF-seq | any | supported | `--enzyme ddda` | separate nucleosome and TF models, radial nucleosome recall, CpG-aware recall |
| EcoGII (m6A) | PacBio, Nanopore | development | custom `-m` only | a model file is shipped (`ecogii_pacbio.json`); not accepted by `--enzyme`, not validated; rejected by `fiberhmm-consensus` |
| CpG methyltransferase (SssI) | Nanopore | development | custom `-m` only | `cpg_nanopore.json`, observation mode `cpg` |
| GpC methyltransferase | — | development | custom `-m` only | observation mode `gpc`; no bundled model |
| Other enzymes | — | unsupported | train your own | see [Training a model](../workflows/training.md) |

The public preset list is `fiberhmm/models/SUPPORTED_MODES.json`. Model
files for development chemistries are shipped for method development and
carry no supported-workflow claim.

Some commands support a subset:

| Command | Chemistries |
|---|---|
| `fiberhmm-consensus`, `fiberhmm-transfer` | `ddda`, `dddb`, `hia5-pacbio`, `hia5-nanopore` |
| `fiberhmm-strand-rescue` | `ddda`, `dddb`, `hia5-nanopore` (PacBio Hia5 already observes both strands) |
| `fiberhmm-pair`, `fiberhmm-merge` | DddA |
| `fiberhmm-tag-m5c`, `fiberhmm-call-m5c` | genome-wide DddA only |
| `fiberhmm-dedup`, `fiberhmm-daf-snps`, `fiberhmm-daf-encode` | DAF (DddA, DddB) |

## Default settings per chemistry

The resolved chemistry (`--enzyme`, the given or
[detected](../getting-started/choosing-chemistry.md#automatic-detection-of-seq)
`--seq`, or the input's `FIBERHMM-CHEMISTRY` declaration for tools without
`--enzyme`) sets these defaults. An explicit option always wins.

| Setting | Hia5 PacBio | Hia5 Nanopore | DddB | DddA |
|---|---|---|---|---|
| Observation mode | `pacbio-fiber` | `nanopore-fiber` | `daf` | `daf` |
| HMM model | `hia5_pacbio.json` | `hia5_nanopore.json` | `dddb_nanopore.json` | `ddda_nuc.json` |
| TF-recall model | same | same | same | `ddda_TF.json` |
| ML threshold: `call`, `apply` | 128 | **248** | 128 | 128 |
| ML threshold: `recall-tfs`/`-nucs`, `extract`, `qc` | 125 | **248** | 125 | 125 |
| ML threshold: `dedup`, `pair`, `merge` (MM/ML dU) | — | — | 128 | 128 |
| TF interval cost `--min-llr` | 5.0 | 5.0 | 5.0 | 5.0 |
| Nucleosome recall (`--nuc-recall-policy auto`) | conservative | conservative | conservative | phase-aware radial |
| CpG-aware recall (`--use-m5c`) | off | off | off | **on** (call, recall, pair/merge) |
| Adjacent-target thinning (`--daf-mask-runs`) | — | — | off | 2, keep-one |
| Strand-swap chimera filter | — | — | on | on |
| Duplicate marking and SNP screen (file input) | — | — | on | on |
| Declared platform when `--seq` is omitted | detected | detected | `nanopore` | `pacbio` |

Notes:

- **ML thresholds** apply to calls read from `MM`/`ML`. Deaminations
  encoded as R/Y in the sequence or read from `MD` mismatches are binary and
  ignore the threshold. Hia5 Nanopore uses 248 because Dorado's m6A ML values
  are reliable only near the top of the scale; the bundled Hia5 Nanopore QC
  reference and the `hia5-nanopore` strand-rescue preset are calibrated at 248.
- Tools that re-read `MM`/`ML` from an existing BAM (`recall-tfs`, `extract`,
  `qc`) use 125 for non-Nanopore chemistries while `call`/`apply` use 128.
  This is a known inconsistency kept for compatibility; pass
  `--prob-threshold` to make them agree.
- `fiberhmm-posteriors` resolves its threshold like `fiberhmm-call` (248 for
  Hia5 Nanopore, 128 otherwise).
- Other tools keep their own defaults: `fiberhmm-probs` 128, `fiberhmm-train` 125,
  `fiberhmm-utils transfer` 128, `fiberhmm-consensus` 125 for Hia5 PacBio and
  248 for Hia5 Nanopore.
- `fiberhmm-call` and `fiberhmm-apply` call **primary alignments only**
  (`--no-primary` to also call secondary and supplementary records).

## Hia5 on PacBio and on Nanopore

A PacBio HiFi read reports m6A on both strands of the molecule: A's on the
read strand (`A+a`) and A's on the opposite strand, which appear as T's in
the read (`T-a`). The `pacbio-fiber` mode uses both, merging reverse-
complement contexts. A Nanopore read reports m6A only on the strand that went
through the pore (`A+a`), so `nanopore-fiber` sees only half the targets.

Consequences for Nanopore Hia5:

- sparser evidence per read: nucleosome recall splits fewer long protected
  blocks, and trimmed edge pieces shorter than `--nuc-min-size` are reported as
  60–89-bp footprint-scale calls more often than on PacBio;
- the stricter ML threshold (248);
- strand rescue can recover footprints missed on one strand by using the
  other orientation ([Strand rescue](../workflows/strand-rescue.md));
- in consensus, both alignment orientations of Hia5 are pooled into one
  channel.

!!! warning "2.x Nanopore Hia5 table"
    The Nanopore Hia5 emission table shipped in every 2.x release was indexed
    in the wrong context order. Re-run ONT Hia5 calls made with 2.x; see
    [Upgrading from 2.x](../upgrading.md).

## DAF-seq: DddA and DddB

DAF-seq deaminates accessible cytosines. After amplification, a deamination
on the strand that matches the reference reads as C→T ("CT" reads) and one on
the opposite strand as G→A ("GA" reads). Each read carries one flavour and is
informative only at reference C (CT) or reference G (GA). Its flavour, not
its alignment orientation, decides which bases are targets.

- **DddB**: one model (`dddb_nanopore.json`) is used for the HMM and TF
  recall.
- **DddA** also deaminates inside nucleosomes, in a helically phased pattern.
  FiberHMM therefore uses separate models for
  the nucleosome HMM (`ddda_nuc.json`), TF recall (`ddda_TF.json`) and radial
  nucleosome refinement, and enables CpG-aware recall and keep-one masking by
  default. DddA is also the chemistry of scDAF duplex pairing and of the
  CpG-island methylation caller.

Where the deaminations come from, duplicate marking, SNP screening and the
chimera filter are described in [DAF-seq](../workflows/daf-seq.md).
