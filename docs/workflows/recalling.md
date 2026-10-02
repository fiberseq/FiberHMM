# Re-calling

The recallers re-run the likelihood-ratio passes on a BAM that already has
HMM calls, without running the HMM again. Use them to:

- add TF calls to the output of `fiberhmm-apply`;
- re-call with a refit or custom emission table;
- change recall settings (`--min-llr`, nucleosome policy, CpG policy) on an
  existing call set.

For DddA output of `fiberhmm-call`, use `fiberhmm-call` again instead (see
[DddA call output](#ddda-call-output)); that includes re-calling after
`fiberhmm-tag-m5c` adds CpG-island states
([DAF-seq](daf-seq.md#ddda-cpg-island-methylation)).

| Command | Does |
|---|---|
| `fiberhmm-recall-tfs` | TF recall over the input's MSPs and short footprints; nucleosomes pass through (`nuc.Q`) |
| `fiberhmm-recall-nucs` | nucleosome recall, MSP re-derivation, then TF recall and promotion (same as `fiberhmm-recall-tfs --recall-nucs`) |

Both rebuild each read's observations from its own `MM`/`ML` (or DAF
evidence) and sequence, and take the HMM footprints from the input's
`ns`/`nl`/`as`/`al`, or from its `MA` when the legacy tags are absent
(`--no-legacy-tags` output, fibertools `Ma`). On `fiberhmm-apply` output,
`fiberhmm-recall-nucs` gives the same footprint tags as `fiberhmm-call` for
matching `--phase-nrl` and `--nuc-recall-policy`.

## After `fiberhmm-apply`

```bash
fiberhmm-apply -i demo/hia5_pacbio.bam --enzyme hia5 --seq pacbio -o out/apply -c 2
fiberhmm-recall-tfs  -i out/apply/hia5_pacbio_footprints.bam -o out/apply.tfs.bam \
    --enzyme hia5 --seq pacbio -c 2
fiberhmm-recall-nucs -i out/apply/hia5_pacbio_footprints.bam -o out/apply.recalled.bam \
    --enzyme hia5 --seq pacbio -c 2
```

```text
[recall_tfs] enzyme=hia5 mode=pacbio-fiber k=3 min_llr=5.00 uplift=1.00 tf_decoder=multi_interval_v1 cpg_mask=off unify_threshold=90 cores=2 numba=on
[recall_tfs] ML threshold for MM/ML calls: 125
[recall_tfs] processed 300 reads; 300 carried v2 tags; ... TF calls emitted; ... v2 short nucs demoted to tf+
```

`fiberhmm-apply` output declares its chemistry like `fiberhmm-call` output,
so a refit table given with `-m` alone inherits it (see below). The recall
output declares its chemistry and `MA-TYPES`, keeps the input order, and is
not indexed; run `samtools index` if the input was sorted.

## With a refit or custom table

On a BAM that `fiberhmm-call` or `fiberhmm-apply` produced, give only the model:

```bash
fiberhmm-recall-tfs -i out/dddb.calls.bam -o out/dddb.refit.bam -m dddb_refit.json -c 2
```

```text
NOTE: fiberhmm-recall-tfs: custom model on an input declaring [assay=daf enzyme=dddb platform=nanopore mode=daf]; using the defaults of --enzyme dddb --seq nanopore with the given model file(s).
```

The custom model inherits the declared enzyme and platform because its
observation mode (`daf`) matches, and the run uses that enzyme's defaults
(ML threshold, CpG masking, keep-one, nucleosome policy, TF presets) with
your table. The output keeps the input's chemistry declaration.

If the table's mode does not match, or you pass a conflicting
`--enzyme`/`--seq`, the run stops:

```text
error: fiberhmm-recall-tfs: the input BAM declares chemistry [assay=daf enzyme=dddb platform=nanopore mode=daf] but this run would declare [assay=fiber-seq enzyme=hia5 platform=nanopore mode=nanopore-fiber]. Pass --enzyme/--seq matching the input (with --model for a custom table), or --replace-chemistry to re-declare the output deliberately.
```

`--replace-chemistry` drops the input's declaration and declares this run's
own chemistry instead; for a custom table that is
`enzyme=custom;platform=unknown`, and no enzyme defaults are inherited. Use it
only for a deliberate change of chemistry.

A custom DddA table given without an inherited DddA declaration does not
turn on CpG-aware recall automatically; add `--use-m5c`.

## What is kept and what is regenerated

- Regenerated: `nuc` (from `recall-nucs`; `recall-tfs` passes nucleosomes
  through), `msp`, `tf`, and the legacy tags (unless `--no-legacy-tags`).
- Kept unchanged, with their quality bytes: any other `MA` group, for example
  `ddda_mcg`/`ddda_ucg` and the `deam+`/`deam-` coverage of a merged duplex.
- Short HMM footprints (`nl` < `--unify-threshold`) that a recaller call
  overlaps move to `tf`; unmatched ones stay in `nuc` with `nq` = 0.
- Records the recaller skips lose this run's call tags rather than keeping
  stale ones.

The recallers re-derive deaminations from each read. They do not re-apply a
SNP mask or a `--reference` used at calling time, and say so:

```text
WARNING: the input was called by fiberhmm-call with SNP mask (daf_snp_mask=on/0sites). Recall re-derives deaminations from each read and does not re-apply the SNP mask or reference, so masked sites count as hits again. Re-run fiberhmm-call with the same options for SNP-masked calls.
```

## DddA call output

Recalling a DddA BAM from `fiberhmm-call` does not reproduce the call. With
DddA, `fiberhmm-call` chooses the TF scan space from the HMM's own footprints
and drops TFs that only the radial nucleosome refinement exposed. Its output
keeps only the refined footprints, so a recall scans different space: on the
demo data, `recall-tfs` with identical settings changes the TF calls on 370 of
377 reads. The recallers print a warning on such input. Re-run
`fiberhmm-call` on the BAM instead: it re-runs the HMM, gives the same calls
on its own output, and keeps the `ddda_ucg`/`ddda_mcg` island calls, so it
also re-calls DddA after `fiberhmm-tag-m5c`. Hia5 and DddB recalls of
`fiberhmm-call` output reproduce the call.

## Options that differ from `fiberhmm-call`

| Option | Default | Notes |
|---|---|---|
| `--prob-threshold` | 125; 248 for Hia5 Nanopore | `fiberhmm-call` uses 128 for non-Nanopore data |
| `--input-frame` | `auto` | frame of the input's legacy tags. `auto` applies the [shared frame rule](../reference/footprint-tag-frame.md): the last footprint writer on each `@PG` PP chain decides (fibertools: molecular; FiberHMM call/apply/recall: molecular when it declares `coord=molecular`, else query); with no writer, a `coord=molecular` declaration means molecular, otherwise query (FiberHMM 2.12 output). Merged histories that disagree with nothing declared stop the run at the first read with these tags. fibertools `Ma` tags are always molecular. |
| `-c/--cores` | 1 | `0` = all CPUs |
| `--chunk-size` | 1024 | reads per worker chunk |
| `--phase-nrl` | `auto` | estimated from the input's existing nucleosome tags |
| `--recall-nucs` | off (`recall-tfs`), on (`recall-nucs`) | |

Input and output may be `-` (stdin/stdout) for piping. `fiberhmm-recall-nucs`
handles linear reads only; circular molecules need `fiberhmm-call -r`.

## Output modes

- **Spec mode** (default): `MA`/`AQ` carry `nuc`, `msp` and `tf` with their
  scores; the legacy `ns`/`nl` are refreshed and hold nucleosomes only.
- **Downstream-compat mode** (`--downstream-compat`): TF calls go into
  `ns`/`nl` next to the nucleosomes and no `MA`/`AQ` is written (an existing
  one is removed). Per-TF `tq`/`el`/`er` are lost. Use it only for tools that
  read legacy tags alone.

Every option: [`fiberhmm-recall-tfs`](../reference/cli.md#fiberhmm-recall-tfs),
[`fiberhmm-recall-nucs`](../reference/cli.md#fiberhmm-recall-nucs).
