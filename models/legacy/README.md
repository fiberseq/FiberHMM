# Legacy models

These models are kept for reproducibility of older results but are
**superseded** by the active models in the parent `models/` directory.
Do not use these for new analyses unless you have a specific reason
(e.g. comparing to a previously-published call set).

| File | Status | Replaced by |
|---|---|---|
| `ddda_pacbio.json` | one-pass DddA model -- limited TF resolution | `ddda_nuc.json` (apply step) + `ddda_TF.json` (recall step). See main README "DddA workflow -- two passes, two models" section. |
| `hia5_pacbio_fp0.1x.json` | Hia5 PacBio variant with 0.1× FP rate scaling | `hia5_pacbio.json` (default). The 0.1× variant was an experimental low-FP-rate calibration that did not generalize. |
| `ddda_v3/` | DddA emission/transition sweep results from the ddda_nuc.json development cycle (varying false-negative rate, breathing, FP scale) | `ddda_nuc.json` (final pick from the sweep). Kept for reference; `ddda_sweep.py` at the package root reads from this directory. |
| `hia5_nanopore_gt_swapped_legacy.json` | Hia5 Nanopore table as shipped through 2.16.8: contexts numbered alphabetically (ACGT) while the encoder uses A=0, C=1, T=2, G=3, so every context with a G or T read the wrong emission | `hia5_nanopore.json` (reindexed 2026-09-29; same values, encoder order). Use only to reproduce calls made before the fix. |
| `ddda_nuc_refine_context_v2.6.json` | DddA radial-nucleosome likelihoods through 2026-09-29: the v2.6.0 `ddda_TF.json` frozen on 2026-09-01; its per-context pattern did not track SsDddA context rates | `fiberhmm/models/ddda_nuc_refine.json` (context-independent, same state means and transitions). Use only to reproduce earlier radial-nucleosome calls. |
