"""HMM inference engine, parallel processing, and statistics."""

from fiberhmm.inference.bam_output import (
    convert_to_bigbed,
    extract_bed_from_tagged_bam,
    write_bed12_records_direct,
)
from fiberhmm.inference.engine import (
    detect_mode_from_bam,
    predict_footprints,
    predict_footprints_and_msps,
)
from fiberhmm.inference.parallel import (
    _get_genome_regions,
    process_bam_for_footprints,
)
from fiberhmm.inference.stats import FootprintStats
from fiberhmm.inference.tf_family_ids import (
    DEFAULT_FAMILY_SEPARATION_BP,
    TFFamilyInterval,
    allocate_repeating_family_ids,
)
from fiberhmm.inference.tf_sites import (
    BaselineMolecule,
    BindingHypothesis,
    BindingHypothesisAssignment,
    FootprintModelLocus,
    FootprintObservation,
    FootprintPopulationModel,
    SiteDiscoveryConfig,
    TFModelAssignment,
    TFModelCatalog,
    TFModelFamily,
    TFModelLocus,
    TFObservation,
    build_footprint_population_model,
    build_tf_model_catalog,
)

__all__ = [
    'predict_footprints',
    'predict_footprints_and_msps',
    'detect_mode_from_bam',
    'process_bam_for_footprints',
    '_get_genome_regions',
    'FootprintStats',
    'DEFAULT_FAMILY_SEPARATION_BP',
    'TFFamilyInterval',
    'allocate_repeating_family_ids',
    'write_bed12_records_direct',
    'convert_to_bigbed',
    'extract_bed_from_tagged_bam',
    'BaselineMolecule',
    'BindingHypothesis',
    'BindingHypothesisAssignment',
    'FootprintModelLocus',
    'FootprintObservation',
    'FootprintPopulationModel',
    'SiteDiscoveryConfig',
    'TFModelAssignment',
    'TFModelCatalog',
    'TFModelFamily',
    'TFModelLocus',
    'TFObservation',
    'build_footprint_population_model',
    'build_tf_model_catalog',
]
