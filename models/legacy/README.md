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
