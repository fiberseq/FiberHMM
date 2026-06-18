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

import os
from dataclasses import dataclass

AUTO_WRITER_FORMAT = 'auto'
HDF5_WRITER_FORMAT = 'hdf5'
TSV_WRITER_FORMAT = 'tsv'
HDF5_OUTPUT_SUFFIXES = ('.h5', '.hdf5')
POSTERIOR_WRITER_FORMATS = (HDF5_WRITER_FORMAT, TSV_WRITER_FORMAT)


@dataclass(frozen=True)
class _PosteriorWriterRequest:
    output_path: str
    format: str = AUTO_WRITER_FORMAT
    mode: str = 'pacbio-fiber'
    context_size: int = 3
    edge_trim: int = 10
    source_bam: str = ''
    batch_size: int = 1000
    compress: bool = True


def _path_endswith(output_path: str, suffixes: tuple[str, ...]) -> bool:
    return os.fspath(output_path).lower().endswith(suffixes)


def _is_hdf5_output_path(output_path: str) -> bool:
    return _path_endswith(output_path, HDF5_OUTPUT_SUFFIXES)


def _resolve_writer_format(output_path: str, format: str) -> str:
    format = AUTO_WRITER_FORMAT if format is None else str(format).strip().lower()
    if format != AUTO_WRITER_FORMAT:
        return format
    if _is_hdf5_output_path(output_path):
        return HDF5_WRITER_FORMAT
    return TSV_WRITER_FORMAT


def _unknown_writer_format_message(format: str) -> str:
    valid_formats = " or ".join(repr(item) for item in POSTERIOR_WRITER_FORMATS)
    return f"Unknown posteriors format: {format!r}. Use {valid_formats}."


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
    return create_writer_from_request(
        _PosteriorWriterRequest(
            output_path=output_path,
            format=format,
            mode=mode,
            context_size=context_size,
            edge_trim=edge_trim,
            source_bam=source_bam,
            batch_size=batch_size,
            compress=compress,
        ),
    )


def create_writer_from_request(request: _PosteriorWriterRequest):
    format = _resolve_writer_format(request.output_path, request.format)

    if format == HDF5_WRITER_FORMAT:
        from fiberhmm.posteriors.hdf5_backend import PosteriorWriter
        return PosteriorWriter(
            request.output_path,
            request.mode,
            request.context_size,
            request.edge_trim,
            request.source_bam,
            request.batch_size,
        )
    elif format == TSV_WRITER_FORMAT:
        from fiberhmm.posteriors.tsv_backend import PosteriorsTSVWriter
        return PosteriorsTSVWriter(
            request.output_path,
            request.mode,
            request.context_size,
            request.edge_trim,
            request.source_bam,
            request.compress,
        )
    else:
        raise ValueError(_unknown_writer_format_message(format))
