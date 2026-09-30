# Bundled models

Models ship inside the package under `fiberhmm/models/`. `--enzyme` (and for
Hia5 `--seq`) selects them; `-m/--model` is only for custom models. For a
bundled model, the enzyme/platform registry decides the observation mode even
if the file's own metadata disagrees; a custom model uses its embedded
`mode`.

## Presets

| File | `--enzyme` | `--seq` | Mode | Role |
|---|---|---|---|---|
| `hia5_pacbio.json` | `hia5` | `pacbio` | `pacbio-fiber` | HMM and TF recall |
| `hia5_nanopore.json` | `hia5` | `nanopore` | `nanopore-fiber` | HMM and TF recall (reindexed to encoder order in 3.0) |
| `dddb_nanopore.json` | `dddb` | any | `daf` | HMM and TF recall (in-vivo accessible-state rates on the reindexed naked table, 3.0) |
| `ddda_nuc.json` | `ddda` | any | `daf` | first-pass nucleosome HMM |
| `ddda_TF.json` | `ddda` | any | `daf` | TF recall ([model card](https://github.com/fiberseq/FiberHMM/blob/main/fiberhmm/models/ddda_TF.MODEL_CARD.md)) |
| `ddda_nuc_refine.json` | `ddda` | any | `daf` | internal radial-nucleosome likelihoods (context-independent: one hit probability per state) |
| `ddda_nuc_profile.json` | `ddda` | any | — | radial nucleosome profile `ddda_phase_posterior_v1` (dyad and helical-phase template) |

The public preset list is
[`fiberhmm/models/SUPPORTED_MODES.json`](https://github.com/fiberseq/FiberHMM/blob/main/fiberhmm/models/SUPPORTED_MODES.json)
(schema `fiberhmm.supported_modes.v1`); every preset uses `tf_min_llr` 5.0.

For DddA, `fiberhmm-call --enzyme ddda` runs all three tables in one pass;
keeping them separate means that updating the TF table does not retune the
HMM or the radial nucleosome caller. The nucleosome profile's identity and
digest are recorded in the output (`nuc_model`, `nuc_sha256`).

## Other bundled files

| File | Used by | Content |
|---|---|---|
| `ddda_duplex_v1.json` | `fiberhmm-pair` (`--call-layer input-ma`) | sequence-free duplex ranker for archived-`MA` calls |
| `ddda_duplex_rotational_v1.json` | `fiberhmm-pair` (`--call-layer rotational-recall`, the default for 3.0 calls) | the same ranker calibrated on 3.0 rotational DddA calls |
| `ecogii_pacbio.json` | custom `-m` only | EcoGII m6A (development) |
| `cpg_nanopore.json` | custom `-m` only | CpG methyltransferase, `cpg` mode (development) |

The duplex models carry status `externally_replicated_experimental` and their
training and validation intervals.

## Legacy tables

Kept under `fiberhmm/models/legacy/` only to reproduce older results; never
selected by a preset.

| File | What it is |
|---|---|
| `hia5_nanopore_gt_swapped_legacy.json` | the Nanopore Hia5 table as shipped through 2.x, with contexts in alphabetical order (every context with a G or T read another context's emission) |
| `dddb_nanopore_gt_swapped_legacy.json` | the DddB table before its reindexing, with the same error |
| `dddb_nanopore_naked_2f10003c.json` | the reindexed naked DddB table the in-vivo table is built on (3.0 development builds only) |
| `ddda_nuc_refine_context_v2.6.json` | the earlier per-context radial-nucleosome likelihoods (3.0 development builds only) |
| `ddda_pacbio.json` | the older one-pass DddA model, superseded by `ddda_nuc.json` + `ddda_TF.json` |
| `hia5_pacbio_fp0.1x.json` | an experimental low-false-positive Hia5 PacBio calibration that did not generalize |

To reproduce a 2.x Nanopore Hia5 call, pass the legacy table explicitly:

```bash
LEGACY=$(python -c "import os, fiberhmm.models as m; print(os.path.join(m.__path__[0], 'legacy', 'hia5_nanopore_gt_swapped_legacy.json'))")
fiberhmm-call -i ont.bam -o ont.2x-table.bam -m "$LEGACY" --enzyme hia5 --seq nanopore
```

Other 3.0 defaults also changed (see [Upgrading from 2.x](../upgrading.md));
add the matching options, such as `--prob-threshold 128`, to reproduce 2.x
numbers closely.

A source checkout also has a top-level `models/` directory: a compatibility
mirror for scripts that used paths such as `models/hia5_pacbio.json`. The
packaged `fiberhmm/models/` is authoritative.

## Model file format

Models are JSON:

```json
{
  "model_type": "FiberHMM",
  "version": "2.0",
  "n_states": 2,
  "startprob": [...],
  "transmat": [[...], [...]],
  "emissionprob": [[...], [...]],
  "context_size": 3,
  "mode": "pacbio-fiber"
}
```

Each emission row has, for *k* = 3, 8,194 entries: 4,096 context codes for
"marked", two sentinel positions, and 4,096 for "unmarked". Context codes use
the encoder's base order (A = 0, C = 1, T = 2, G = 3; see
[Training](../workflows/training.md#1-emission-tables-fiberhmm-probs)). The
loader orders the states protected/accessible from the emission rows.
`fiberhmm-utils inspect model.json` prints a summary.

`.npz` and legacy `.pickle` models still load. **Loading a pickle executes
code**: load or `fiberhmm-utils convert` only pickles you trust, and prefer
JSON.
