# Training a model

The bundled models cover Hia5 (PacBio, Nanopore), DddB and DddA. Train your
own when you use another enzyme or platform, or have matched controls for a
condition the bundled tables do not describe. A model has two parts:

- an **emission table**: for each sequence context, the probability of a mark
  in the accessible and in the protected state, estimated from control data
  by `fiberhmm-probs`;
- **transition and start probabilities**, fitted to a sample by
  `fiberhmm-train` (Baum-Welch with the emissions held fixed).

## 1. Emission tables: `fiberhmm-probs`

You need an **accessible** control (naked or dechromatinized DNA treated with
the enzyme) and an **inaccessible** control (DNA whose marks reflect only
background: untreated, or native chromatin, depending on the design). The
demo data include a naked-DNA Hia5 control:

```bash
fiberhmm-probs -a demo/hia5_pacbio.naked.bam -u demo/hia5_pacbio.bam \
    -o out/probs/demo --mode pacbio-fiber -k 3
```

```text
    k=3 (7-mer): 3344 accessible, 3378 inaccessible contexts
Creating combined probability files for fiberhmm-train:
  out/probs/demo/tables/demo_A_k3_probs.tsv (3378 contexts)
```

`-o` is an output directory whose name becomes the table prefix:

```text
out/probs/demo/
  demo_accessible_A.probs.pkl         raw counts (can be re-read by fiberhmm-train)
  demo_inaccessible_A.probs.pkl
  tables/demo_accessible_A_k3.tsv     encode, context, hit, nohit, ratio
  tables/demo_inaccessible_A_k3.tsv
  tables/demo_A_k3_probs.tsv          encode, context, accessible_prob, inaccessible_prob
  plots/                              with --stats (needs fiberhmm[plots])
```

The letter is the target base: `A` for Fiber-seq modes, `C` for `daf` (G
targets on GA reads are reverse-complemented into C-centred contexts).

- `--mode` must match the data: `pacbio-fiber`, `nanopore-fiber` or `daf`
  (`gpc`, `cpg` are development modes).
- `-k/--context-sizes` (default `3 4 5 6`): bases on each side; 3 gives
  7-mers (4^6 = 4,096 contexts). Larger contexts need much more data.
- `-n/--max-reads` (100,000 per sample; 0 = all), `-q/--min-mapq` (20),
  `-p/--prob-threshold` (128), `--min-read-length` (1000), `-e/--edge-trim`
  (10). DAF inputs are read like `fiberhmm-call` reads them (R/Y, `MD`, or
  MM/ML dU).
- The command exits non-zero when no read passes the filters.

**How contexts are numbered.** Emission tables are indexed by the code the
inference encoder gives each context, which numbers bases **A = 0, C = 1,
T = 2, G = 3**, not alphabetically. The `encode` column of every table is
that code. Tables built by FiberHMM before 3.0 through the alphabetical path
had every G and T swapped; this is why the 2.x Nanopore Hia5 and DddB tables
were wrong. The builder can no longer produce an alphabetical table.

## 2. Transitions: `fiberhmm-train`

```bash
fiberhmm-train -i demo/hia5_pacbio.bam \
    -p out/probs/demo/tables/demo_accessible_A_k3.tsv out/probs/demo/tables/demo_inaccessible_A_k3.tsv \
    -o out/model -k 3 -c 3 -r 100
```

```text
Best model selected
Start probabilities: [0.84872966 0.15127034]
Transition matrix:
[[0.99190042 0.00809958]
 [0.02415595 0.97584405]]
Saving to out/model
  Saved: best-model.json
```

- `-p` takes the accessible and the inaccessible table, in that order (TSV,
  or the `.probs.pkl` counts). The combined `*_probs.tsv` is not accepted
  here.
- `-k` must match the context size of the tables.
- `-r/--read-count` (500) reads are sampled from `-i`; `-c/--iterations` (10)
  random initializations are run and the best log-likelihood kept. `--seed`
  (42) makes it reproducible.
- `--base-model model.json` skips training and combines that model's
  transitions with the new emissions (no `-i` needed).
- `-a/--prob-adjust` scales the accessible emission probabilities;
  `--stats` writes example plots (needs `fiberhmm[plots]`).

Outputs in `-o`:

| File | Content |
|---|---|
| `best-model.json` | the model to use |
| `all_models.json` | every initialization |
| `model_config.json` | context size, mode, edge trim, probability adjustment |
| `training-reads.tsv` | the read IDs used (pass to `fiberhmm-apply -t` to exclude them) |

## 3. Use the model

```bash
fiberhmm-call -i demo/hia5_pacbio.bam -o out/custom.calls.bam \
    -m out/model/best-model.json -c 2 --region-parallel
```

A custom model is used for both the HMM and TF recall unless you give
`--recall-model`. Its observation mode comes from the model file; a custom
model without valid mode metadata is refused. The output declares
`enzyme=custom` in its chemistry line, so no enzyme preset defaults apply;
set `--min-llr`, `--prob-threshold`, `--use-m5c` and friends explicitly. When
re-calling a BAM that already declares a supported enzyme with the same
mode, the custom model inherits that enzyme's defaults instead
([Re-calling](recalling.md#with-a-refit-or-custom-table)).

## Validating a model

- `fiberhmm-utils inspect model.json` prints mode, context size, start and
  transition probabilities and emission summaries (`--full` prints the whole
  table). Accessible marking rates should be high and inaccessible rates low.
- Call a data set with known structure and compare against the bundled
  model: nucleosome size (about 147 bp), repeat length, and
  [QC](qc.md) periodicity are quick checks.
- For a chemistry with a bundled table, the table's per-context rates should
  correlate with rates observed on your data under the encoder numbering
  above; a strong correlation only under a base relabelling points to an
  indexing error.

## Model utilities

```bash
fiberhmm-utils inspect out/model/best-model.json
fiberhmm-utils adjust out/model/best-model.json --state accessible --scale 1.1 -o out/model/adjusted.json
fiberhmm-utils convert old_model.pickle new_model.json
```

- `adjust` multiplies the emission probabilities of one or both states.
- `convert` turns a legacy `.pickle`/`.pkl` or `.npz` model into JSON. Loading
  a pickle executes code; convert only files you trust.
- `transfer` estimates emissions for a target chemistry (for example DAF) by
  regressing its per-context rates on accessibility learned from footprint
  calls in a reference BAM (`--reference-bam`) or a priors table
  (`--accessibility-priors`). **Known issue in 3.0.0:** `fiberhmm-utils
  transfer` stops with `KeyError: 'total'` while estimating emissions; use
  `fiberhmm-probs` with matched controls instead.

Models are JSON (preferred); `.npz` and `.pickle` still load (`-m` accepts
them). Every option: [`fiberhmm-probs`](../reference/cli.md#fiberhmm-probs),
[`fiberhmm-train`](../reference/cli.md#fiberhmm-train),
[`fiberhmm-utils`](../reference/cli.md#fiberhmm-utils).
