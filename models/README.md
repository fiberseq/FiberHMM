# Source-tree model mirror

The authoritative runtime and packaged models are under `fiberhmm/models/`.
Files in this top-level directory are a source-tree compatibility mirror for
older scripts that supplied paths such as `models/hia5_pacbio.json`; they are
not a second model registry. Current mirrored filenames must remain
byte-identical to the corresponding packaged files. The explicitly named
`models/legacy/` directory contains only the 2024-method reproducibility
artifacts and is never selected by a current enzyme preset.

The machine-readable public workflow surface is
`fiberhmm/models/SUPPORTED_MODES.json`. Other model-development artifacts do
not constitute supported enzyme presets.
