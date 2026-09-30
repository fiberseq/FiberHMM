# Choosing the chemistry

FiberHMM needs to know two things about your data: the **enzyme** that marked
accessible DNA and, for Hia5, the **sequencing platform**. Together they pick
the bundled model, the observation mode and every chemistry-dependent default.

## `--enzyme`

| Your assay | `--enzyme` | What FiberHMM reads |
|---|---|---|
| Fiber-seq with the m6A methyltransferase Hia5 | `hia5` | m6A calls in `MM`/`ML` |
| DAF-seq with the double-stranded deaminase DddB | `dddb` | C→T / G→A conversions |
| DAF-seq with the single-strand-preferring deaminase DddA | `ddda` | C→T / G→A conversions |

`fiberhmm-call`, `fiberhmm-apply`, `fiberhmm-recall-tfs`/`-recall-nucs` and
`fiberhmm-posteriors` take `--enzyme`. Only these three presets are accepted.
For anything else, supply a model file with `-m/--model` (see
[Training a model](../workflows/training.md)).

## `--seq`

`--seq pacbio|nanopore` matters for Hia5 only:

- **PacBio** reports m6A on both strands of the molecule (A calls on the read
  strand as `A+a`, and A calls on the opposite strand as `T-a`).
- **Nanopore** reports m6A only on the sequenced strand (`A+a`).

The two platforms therefore have different models (`hia5_pacbio.json`,
`hia5_nanopore.json`), different observation modes (`pacbio-fiber`,
`nanopore-fiber`) and different defaults (see below).

For DddA and DddB the deamination read-out does not depend on the platform:
both use the `daf` mode on every platform, and `--seq` only sets the platform
recorded in the output header (default: `pacbio` for DddA, `nanopore` for
DddB).

## Automatic detection of `--seq`

When `--seq` is omitted for Hia5, `fiberhmm-call`, `fiberhmm-apply` and
`fiberhmm-recall-tfs`/`-recall-nucs` look at the input, strongest evidence
first:

1. a `FIBERHMM-CHEMISTRY` declaration in the header (a BAM FiberHMM already
   called);
2. the `MM` specifications of the first 200 records: `T-a` present means
   PacBio, `A+a` alone means Nanopore;
3. `@RG PL` values and `@PG` program names (for example `pbmm2`, `ccs`,
   `jasmine` for PacBio; `dorado`, `guppy`, `minimap2 -x map-ont` for
   Nanopore).

The choice is printed:

```text
NOTE: --seq not given; using --seq nanopore (detected from MM specs of 200 read(s) (A+a only, no T-a)).
```

If the sources disagree, or more than 10% of the inspected reads show the
other `MM` pattern, the run stops and asks for `--seq`:

```text
error: fiberhmm-call: cannot infer the sequencing platform for --enzyme hia5: MM specs are mixed: ... Pass --seq pacbio or --seq nanopore.
```

Input that cannot be inspected (stdin) falls back to PacBio with a warning,
so pass `--seq` when you stream:

```text
WARNING: --seq not given for --enzyme hia5 and the platform could not be detected (stdin (not inspected)); assuming PacBio. Pass --seq nanopore for Nanopore data.
```

An explicit `--seq` always wins; if the input looks like the other platform
you get a warning, not an error. `fiberhmm-posteriors` does not detect the
platform: without `--seq` it uses PacBio.

## What the chemistry changes

| Setting | Hia5 PacBio | Hia5 Nanopore | DddB | DddA |
|---|---|---|---|---|
| Model | `hia5_pacbio.json` | `hia5_nanopore.json` | `dddb_nanopore.json` | `ddda_nuc.json` (HMM), `ddda_TF.json` (TF recall) |
| Observation mode | `pacbio-fiber` | `nanopore-fiber` | `daf` | `daf` |
| ML threshold in `fiberhmm-call` | 128 | **248** | 128 (MM/ML dU only) | 128 (MM/ML dU only) |
| Nucleosome recall | conservative | topology | conservative | phase-aware radial |
| CpG-aware recall | off | off | off | on |
| Adjacent-target thinning | — | — | off | runs ≥ 2, keep-one |
| Duplicate marking, SNP screen | — | — | on (file input) | on (file input) |

The full table, including the thresholds of the other tools, is in
[Chemistries and platforms](../concepts/chemistries.md#default-settings-per-chemistry).

## Re-calling a BAM FiberHMM already called

`fiberhmm-call` and `fiberhmm-recall-tfs`/`-recall-nucs` write a
[`FIBERHMM-CHEMISTRY`](../reference/headers.md#fiberhmm-chemistry) line into
the output header. Tools that read such a BAM take the chemistry from it:

- `fiberhmm-recall-tfs`/`-recall-nucs` fill a missing `--seq` from it;
- `fiberhmm-extract` and `fiberhmm-qc` use it for their ML threshold and QC
  reference;
- `fiberhmm-consensus` and `fiberhmm-transfer` use it to pick the emission
  model.

A run whose chemistry contradicts the declaration stops with a one-line fix:

```text
error: fiberhmm-recall-tfs: the input BAM declares chemistry [assay=daf enzyme=dddb platform=nanopore mode=daf] but this run would declare [assay=fiber-seq enzyme=hia5 platform=nanopore mode=nanopore-fiber]. Pass --enzyme/--seq matching the input (with --model for a custom table), or --replace-chemistry to re-declare the output deliberately.
```

A custom `--model` given without `--enzyme` inherits the declared enzyme and
platform when its observation mode matches, and then uses that enzyme's
defaults:

```text
NOTE: fiberhmm-recall-tfs: custom model on an input declaring [assay=daf enzyme=dddb platform=nanopore mode=daf]; using the defaults of --enzyme dddb --seq nanopore with the given model file(s).
```

`--replace-chemistry` drops the input's declaration and writes this run's
(for a custom model, `enzyme=custom;platform=unknown`). Use it only when you
are deliberately re-calling with a different chemistry.

`fiberhmm-apply` does not write a chemistry declaration, so recalling an
apply output needs `--enzyme` (and `--seq` for Hia5, unless it can be
detected from the reads).
