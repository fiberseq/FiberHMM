# Shared defaults

`fiberhmm-call` defines the call defaults. Every tool that calls footprints,
re-reads modifications from a BAM or reads calls back uses the same values
unless the table of documented differences below lists it. The values live in
one module, `fiberhmm/defaults.py`, and the test
`tests/test_defaults_parity.py` checks every command's options, the run-time
resolvers, the engine function defaults and the parameter classes against it.

Defaults that depend on the chemistry (ML threshold, models, CpG masking, run
thinning) are listed per chemistry in
[Chemistries and platforms](../concepts/chemistries.md#default-settings-per-chemistry).

## Canonical values

| Setting | Value | Options |
|---|---|---|
| Minimum MAPQ | 0 | `--min-mapq` |
| Minimum read length | 1000 bp (full SEQ; DAF: aligned length) | `--min-read-length` |
| Alignment records called | primary and supplementary | `--alignments` |
| ML threshold (MM/ML) | 128; Hia5 Nanopore 248 | `--prob-threshold` |
| Edge trim | 10 bp | `--edge-trim` |
| DddA CpG mask | unmethylated-only (CpGs neutral except in `ddda_ucg` islands); off for other enzymes | `--use-m5c`, `--cpg-mask-policy` |
| DAF run thinning | DddA: runs ≥ 2, keep-one; off otherwise | `--daf-mask-runs`, `--daf-run-policy` |
| Nucleosome minimum size | 85 bp | `--nuc-min-size` |
| MSP minimum size | 0 bp | `--msp-min-size` |
| TF interval cost, emission uplift | 5.0, 1.0 (all enzymes) | `--min-llr`, `--emission-uplift` |
| TF minimum opportunities | 3 | `--min-opps` |
| Nucleosome demotion threshold | 90 bp | `--unify-threshold` |
| Nucleosome split | LLR 4.0, 3 positions | `--split-min-llr`, `--split-min-opps` |
| DAF chimeras | flag (0x200 + `xc:Z:strand_swap`); min segment 5, purity 0.8 | `--chimeras` |
| DAF duplicates | Jaccard 0.95, ≥ 10 deaminations, ends within 50 bp | `--dedup-*` |
| DAF insert consensus | ≥ 20 carriers | `--daf-insert-min-carriers` |
| Legacy `ns/nl/nq`, `as/al` beside `MA` (every writer: call, call `--hmm-only`, recall-tfs/-nucs, merge `--recall`, pipeline) | off (`MA`/`AQ` only; legacy tags on rewritten records are removed) | `--legacy-tags` |
| Nucleosome/MSP call source (every reader) | `MA` `nuc`/`msp` groups (else fibertools `Ma`); legacy `ns/nl`, `as/al` only when the record has no such group | none (`fiberhmm.io.footprint_calls`) |

Every tool that reads nucleosome or MSP calls back (recall-tfs, recall-nucs,
consensus, scan, extract, QC, `fiberhmm.nuc`, call-m5c, call `--hmm-only --keep-msps`,
`fiberhmm-probs`, stats) takes them from the record's `MA` `nuc` and `msp`
groups, which is what `fiberhmm-call`, `-recall-tfs` and `-recall-nucs`
write, else from fibertools' `Ma`. The legacy `ns/nl` and `as/al` pairs are
read only when the record has no `nuc` or `msp` group in either tag, under
the array-pair rule (two integer arrays of equal length). An `MA` holding only
other groups (m5C islands, FIRE, a lone `tf` layer) does not hide them.
`tests/test_legacy_pair_readers_guard.py` fails when a module reads the
legacy pairs without this rule.

## Documented differences

| Tool | Setting | Value | Why |
|---|---|---|---|
| `fiberhmm-recall-tfs`/`-nucs`, `fiberhmm-extract`, `fiberhmm-qc`, `fiberhmm-consensus` (Hia5 PacBio, DAF MM/ML) | ML threshold | 125 (Hia5 Nanopore 248) | historical value for re-reading MM/ML, kept for compatibility; pass `--prob-threshold` to match the call |
| `fiberhmm-train`, `fiberhmm-footprint --ml` | ML threshold | 125 | fibertools' value (training data, ft footprint analogue) |
| `fiberhmm-probs`, `fiberhmm-utils transfer` | ML threshold | 128 | no chemistry option |
| `fiberhmm-train` | context size | 3 | the bundled models' context size |
| `fiberhmm-pipeline` | MAPQ, read length, chimeras | unset | not passed: `fiberhmm-call`'s default applies (MAPQ: or the value an earlier run recorded in the output folder) |
| `fiberhmm-posteriors` | records | primary only | documented on the [Posteriors](../workflows/posteriors.md) page |
| `fiberhmm-scan` states, peaks, NFR | nucleosome size | 90 bp | the FiberBrowser accessibility rule |
| `fiberhmm.nuc` decoding (`NucParams`, `fiberhmm-scan` nuc) | edge trim | unset | the trim the call recorded (`--edge-trim`), else 10 bp; any value you pass, 10 included, is used |

Python entry points (`process_bam_for_footprints` and the pipeline functions)
take the resolved threshold, read length and alignment set from their caller;
their own parameter defaults are listed in `ALLOWED_OVERRIDES`.
