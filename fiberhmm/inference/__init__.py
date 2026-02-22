"""HMM inference engine, parallel processing, and statistics."""

from fiberhmm.inference.engine import (
    predict_footprints,
    predict_footprints_and_msps,
    detect_mode_from_bam,
)
from fiberhmm.inference.parallel import (
    process_bam_for_footprints,
    _get_genome_regions,
)
from fiberhmm.inference.stats import FootprintStats
from fiberhmm.inference.bam_output import (
    write_bed12_records_direct,
    convert_to_bigbed,
    extract_bed_from_tagged_bam,
)

__all__ = [
    'predict_footprints',
    'predict_footprints_and_msps',
    'detect_mode_from_bam',
    'process_bam_for_footprints',
    '_get_genome_regions',
    'FootprintStats',
    'write_bed12_records_direct',
    'convert_to_bigbed',
    'extract_bed_from_tagged_bam',
]
