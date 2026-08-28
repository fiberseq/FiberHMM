"""I/O helpers for spec-compliant tags and portable analysis artifacts."""

from fiberhmm.io.bam_header import (
    CHEMISTRY_PREFIX,
    append_chemistry,
    declared_chemistries,
    infer_legacy_chemistry,
)

from fiberhmm.io.footprint_bam import (
    BamFootprintDiagnostics,
    BamFootprintInput,
    BamFootprintInputError,
    ReferenceRegion,
    load_footprint_molecules_from_bam,
    parse_reference_region,
)
from fiberhmm.io.tf_models import (
    FootprintModelBigBedPaths,
    FootprintModelBundlePaths,
    TFModelBigBedPaths,
    TFModelBundlePaths,
    convert_footprint_model_bundle_to_bigbed,
    convert_tf_model_bundle_to_bigbed,
    footprint_model_bundle_paths,
    write_footprint_model_bundle,
    write_tf_model_bundle,
)

__all__ = [
    "BamFootprintDiagnostics",
    "BamFootprintInput",
    "BamFootprintInputError",
    "CHEMISTRY_PREFIX",
    "FootprintModelBigBedPaths",
    "FootprintModelBundlePaths",
    "ReferenceRegion",
    "TFModelBigBedPaths",
    "TFModelBundlePaths",
    "append_chemistry",
    "convert_footprint_model_bundle_to_bigbed",
    "convert_tf_model_bundle_to_bigbed",
    "declared_chemistries",
    "footprint_model_bundle_paths",
    "infer_legacy_chemistry",
    "load_footprint_molecules_from_bam",
    "parse_reference_region",
    "write_footprint_model_bundle",
    "write_tf_model_bundle",
]
