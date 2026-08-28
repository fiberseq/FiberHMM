# Footprint population model Dropbox handoff

This directory is the portable, Dropbox-synchronized example bundle for the
footprint population model and its first OCT-family-caller integration.

Start with:

```text
YWMJBX_4_siGAF_4.5-5.5hr_kr.footprint-model.fiberlayers.json
```

The manifest names every other artifact in this directory. The authoritative
OCT input is:

```text
YWMJBX_4_siGAF_4.5-5.5hr_kr.footprint-model.tsv
```

The per-read propagation table is:

```text
YWMJBX_4_siGAF_4.5-5.5hr_kr.footprint-model.assignments.tsv
```

The complete implementation and OCT contract is:

```text
../../docs/FOOTPRINT_POPULATION_MODEL_HANDOFF.md
```

## Source BAM on another Dropbox host

The absolute source path recorded in the manifest belongs to the generating
host. From this directory, the synchronized BAM is expected at:

```text
../../../../Drosophila_phase2/Datasets/DAF-seq/spacetime_updated/kr_compare/fp/YWMJBX_4_siGAF_4.5-5.5hr_kr_recalled.bam
```

Resolve that path from the local Dropbox checkout instead of requiring the
manifest's original `/Users/...` prefix.

## Supported-population rule

For the first OCT integration:

```text
geometry_ready == 1 AND population_ready == 1
```

Rank passing hypotheses by `n_tf` descending. The population BED/BigBed score
also equals `min(n_tf, 1000)`; occupancy is reported separately and must not be
used as support.

## Bundle results

- 915 denominator molecules
- 4,170 projectable TF calls
- 601 population loci
- 1,829 TF-binding hypotheses
- 432 hypotheses passing both default readiness thresholds
- 4,130 assigned and 40 auditable unassigned calls

The FiberLayers BigBed contains the assigned per-read overlay. Its
`blockSiteIds` values join to population `site_id`. FiberBrowser can load the
generic layer and preserve those IDs; model-aware filtering and
cross-highlighting remain downstream work.

Use `SHA256SUMS` to confirm Dropbox finished synchronizing every artifact.

