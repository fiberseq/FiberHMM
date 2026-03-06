"""
Unified posteriors writer interface.

Selects between HDF5 and TSV backends based on format parameter.

Usage:
    from fiberhmm.posteriors.writer import create_writer

    with create_writer('posteriors.h5', format='hdf5', ...) as writer:
        writer.write_fiber(...)

    # Or TSV format:
    with create_writer('posteriors.tsv.gz', format='tsv', ...) as writer:
        writer.write_fiber(...)
"""

import numpy as np
from typing import Optional


def create_writer(output_path: str, format: str = 'auto',
                  mode: str = 'pacbio-fiber', context_size: int = 3,
                  edge_trim: int = 10, source_bam: str = '',
                  batch_size: int = 1000, compress: bool = True):
    """
    Create a posteriors writer with the appropriate backend.

    Args:
        output_path: Output file path
        format: 'hdf5', 'tsv', or 'auto' (detect from extension)
        mode: HMM analysis mode
        context_size: Context size k
        edge_trim: Edge trim setting
        source_bam: Source BAM filename for metadata
        batch_size: Buffer size for HDF5 backend
        compress: Whether to gzip TSV output

    Returns:
        Writer object (PosteriorWriter or PosteriorsTSVWriter)
    """
    if format == 'auto':
        if output_path.endswith('.h5') or output_path.endswith('.hdf5'):
            format = 'hdf5'
        else:
            format = 'tsv'

    if format == 'hdf5':
        from fiberhmm.posteriors.hdf5_backend import PosteriorWriter
        return PosteriorWriter(
            output_path, mode, context_size, edge_trim,
            source_bam, batch_size
        )
    elif format == 'tsv':
        from fiberhmm.posteriors.tsv_backend import PosteriorsTSVWriter
        return PosteriorsTSVWriter(
            output_path, mode, context_size, edge_trim,
            source_bam, compress
        )
    else:
        raise ValueError(f"Unknown posteriors format: {format!r}. Use 'hdf5' or 'tsv'.")
