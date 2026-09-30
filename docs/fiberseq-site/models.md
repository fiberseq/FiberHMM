# Pre-trained models

Models ship inside the package; `--enzyme` (and `--seq` for Hia5) selects
them, and `-m` is only for custom models.

| Chemistry | `--enzyme` | `--seq` | Model | Mode |
|-----------|-----------|---------|-------|------|
| Hia5 Fiber-seq, PacBio | `hia5` | `pacbio` | `hia5_pacbio.json` | `pacbio-fiber` |
| Hia5 Fiber-seq, Nanopore | `hia5` | `nanopore` | `hia5_nanopore.json` | `nanopore-fiber` |
| DddB DAF-seq | `dddb` | any | `dddb_nanopore.json` | `daf` |
| DddA DAF-seq | `ddda` | any | `ddda_nuc.json` (nucleosome HMM), `ddda_TF.json` (TF recall), `ddda_nuc_refine.json` (internal radial-nucleosome likelihoods) | `daf` |

`fiberhmm-call --enzyme ddda` uses all three DddA tables in one pass. Each
preset has its own defaults: for example Nanopore Hia5 reads m6A at ML ≥ 248,
and DddA uses phase-aware radial nucleosome recall and CpG-aware TF recall.
See [Chemistries and platforms](https://fiberseq.github.io/FiberHMM/concepts/chemistries/).

Older tables, including the context-swapped 2.x Nanopore Hia5 table
(`hia5_nanopore_gt_swapped_legacy.json`), are kept in
`fiberhmm/models/legacy/` to reproduce earlier results.

```bash
fiberhmm-utils inspect model.json      # mode, context size, transitions, emission summary
```

Train your own model for another enzyme or with matched controls for your
condition: see [Training a model](https://fiberseq.github.io/FiberHMM/workflows/training/)
and [Bundled models](https://fiberseq.github.io/FiberHMM/reference/models/).
