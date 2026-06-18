"""
Streaming HDF5 writer for HMM posteriors.

Import this in apply_model.py and use during BAM processing to
write posteriors inline without a separate pass.

Usage:
    from fiberhmm.posteriors.hdf5_backend import PosteriorWriter

    with PosteriorWriter('posteriors.h5', mode, context_size, ...) as writer:
        for read in bam:
            # ... process read, get posteriors ...
            writer.add_fiber(chrom, fiber_data)

    # Automatically finalized on context exit
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import h5py
import numpy as np

from fiberhmm.core.bam_reader import get_reference_positions_array
from fiberhmm.io.path_status import path_size_mb


def _int32_array(values) -> np.ndarray:
    if values is None or len(values) == 0:
        return np.array([], dtype=np.int32)
    return np.asarray(values, dtype=np.int32)


def _create_gzip_dataset(
    group,
    name: str,
    data,
    compression_opts: Optional[int] = None,
) -> None:
    kwargs = {'compression': 'gzip'}
    if compression_opts is not None:
        kwargs['compression_opts'] = compression_opts
    group.create_dataset(name, data=data, **kwargs)


@dataclass(frozen=True)
class _Hdf5FileMetadataRequest:
    h5_file: object
    mode: str
    context_size: int
    edge_trim: int
    source_bam: str
    model_path: Optional[str] = None


@dataclass(frozen=True)
class _FiberMetadataDatasetWriteRequest:
    group: object
    fiber_ids: object
    starts: object
    ends: object
    strands: object
    n_fibers: Optional[int] = None


@dataclass(frozen=True)
class _PosteriorChromGroupCreateRequest:
    h5_file: object
    chrom: str
    include_ref_positions: bool = True


def write_fiber_metadata_datasets(
    group,
    fiber_ids,
    starts,
    ends,
    strands,
    n_fibers: Optional[int] = None,
) -> None:
    write_fiber_metadata_datasets_from_request(
        _FiberMetadataDatasetWriteRequest(
            group=group,
            fiber_ids=fiber_ids,
            starts=starts,
            ends=ends,
            strands=strands,
            n_fibers=n_fibers,
        )
    )


def write_fiber_metadata_datasets_from_request(
    request: _FiberMetadataDatasetWriteRequest,
) -> None:
    n = (
        len(request.fiber_ids)
        if request.n_fibers is None
        else int(request.n_fibers)
    )
    if n == 0:
        request.group.attrs['n_fibers'] = 0
        return

    dt = h5py.special_dtype(vlen=str)
    request.group.create_dataset('fiber_ids', data=request.fiber_ids, dtype=dt)
    _create_gzip_dataset(request.group, 'fiber_starts', _int32_array(request.starts))
    _create_gzip_dataset(request.group, 'fiber_ends', _int32_array(request.ends))
    request.group.create_dataset('strands', data=request.strands, dtype=dt)
    request.group.attrs['n_fibers'] = n


def _hdf5_file_metadata_attrs(
    *,
    mode: str,
    context_size: int,
    edge_trim: int,
    source_bam: str,
    model_path: str = None,
) -> dict:
    attrs = {
        'mode': mode,
        'context_size': context_size,
        'edge_trim': edge_trim,
        'source_bam': os.path.basename(os.fspath(source_bam)),
        'format_version': 2,
    }
    if model_path is not None:
        attrs['model_path'] = os.path.basename(os.fspath(model_path))
    return attrs


def write_hdf5_file_metadata(
    h5_file,
    *,
    mode: str,
    context_size: int,
    edge_trim: int,
    source_bam: str,
    model_path: str = None,
) -> None:
    write_hdf5_file_metadata_from_request(
        _Hdf5FileMetadataRequest(
            h5_file=h5_file,
            mode=mode,
            context_size=context_size,
            edge_trim=edge_trim,
            source_bam=source_bam,
            model_path=model_path,
        )
    )


def write_hdf5_file_metadata_from_request(
    request: _Hdf5FileMetadataRequest,
) -> None:
    for key, value in _hdf5_file_metadata_attrs(
        mode=request.mode,
        context_size=request.context_size,
        edge_trim=request.edge_trim,
        source_bam=request.source_bam,
        model_path=request.model_path,
    ).items():
        request.h5_file.attrs[key] = value


def _posterior_chrom_subgroups(include_ref_positions: bool = True) -> List[str]:
    groups = ['posteriors']
    if include_ref_positions:
        groups.append('ref_positions')
    groups.extend(['footprint_starts', 'footprint_sizes'])
    return groups


def create_posterior_chrom_group(
    h5_file,
    chrom: str,
    *,
    include_ref_positions: bool = True,
):
    return create_posterior_chrom_group_from_request(
        _PosteriorChromGroupCreateRequest(
            h5_file=h5_file,
            chrom=chrom,
            include_ref_positions=include_ref_positions,
        )
    )


def create_posterior_chrom_group_from_request(
    request: _PosteriorChromGroupCreateRequest,
):
    grp = request.h5_file.create_group(request.chrom)
    for name in _posterior_chrom_subgroups(request.include_ref_positions):
        grp.create_group(name)
    return grp


@dataclass
class _ChromMetadata:
    ids: List[str] = field(default_factory=list)
    starts: List[int] = field(default_factory=list)
    ends: List[int] = field(default_factory=list)
    strands: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class _FiberArrayDatasetSpec:
    group_name: str
    data: np.ndarray
    compression_opts: Optional[int]


@dataclass(frozen=True)
class _FiberArrayDatasetWriteRequest:
    group: object
    index: int
    fiber: Dict


@dataclass(frozen=True)
class _PosteriorWriterCloseResult:
    total_written: int
    file_size_mb: float

    def as_tuple(self) -> tuple[int, float]:
        return self.total_written, self.file_size_mb


def _new_chrom_metadata() -> _ChromMetadata:
    return _ChromMetadata()


def _append_fiber_metadata(meta: _ChromMetadata, fiber: Dict) -> None:
    meta.ids.append(fiber['read_name'])
    meta.starts.append(fiber['ref_start'])
    meta.ends.append(fiber['ref_end'])
    meta.strands.append(fiber.get('strand', '.'))


def _file_size_mb(path: str) -> float:
    return path_size_mb(path)


def _fiber_array_dataset_specs(fiber: Dict) -> List[_FiberArrayDatasetSpec]:
    return [
        _FiberArrayDatasetSpec(
            group_name='posteriors',
            data=fiber['posteriors'].astype(np.float16),
            compression_opts=4,
        ),
        _FiberArrayDatasetSpec(
            group_name='ref_positions',
            data=_int32_array(fiber.get('ref_positions')),
            compression_opts=4,
        ),
        _FiberArrayDatasetSpec(
            group_name='footprint_starts',
            data=_int32_array(fiber.get('footprint_starts')),
            compression_opts=None,
        ),
        _FiberArrayDatasetSpec(
            group_name='footprint_sizes',
            data=_int32_array(fiber.get('footprint_sizes')),
            compression_opts=None,
        ),
    ]


def _write_fiber_array_datasets(group, index: int, fiber: Dict) -> None:
    _write_fiber_array_datasets_from_request(
        _FiberArrayDatasetWriteRequest(
            group=group,
            index=index,
            fiber=fiber,
        )
    )


def _write_fiber_array_datasets_from_request(
    request: _FiberArrayDatasetWriteRequest,
) -> None:
    idx = str(request.index)
    for spec in _fiber_array_dataset_specs(request.fiber):
        _create_gzip_dataset(
            request.group[spec.group_name],
            idx,
            spec.data,
            compression_opts=spec.compression_opts,
        )


def _write_chrom_fiber_metadata(
    group,
    meta: _ChromMetadata,
    n_fibers: int,
) -> None:
    write_fiber_metadata_datasets(
        group,
        meta.ids,
        meta.starts,
        meta.ends,
        meta.strands,
        n_fibers=n_fibers,
    )


class PosteriorWriter:
    """
    Streaming writer for HMM posteriors to HDF5.

    Buffers posteriors by chromosome and writes in batches to avoid
    memory issues with large BAM files. Must be used from main process only
    (HDF5 is not process-safe for writes).

    Memory usage is O(batch_size * avg_read_length) regardless of BAM size.
    """

    def __init__(
        self,
        output_path: str,
        mode: str,
        context_size: int,
        edge_trim: int,
        source_bam: str,
        batch_size: int = 1000
    ):
        """
        Initialize the posterior writer.

        Args:
            output_path: Path to output HDF5 file
            mode: HMM mode (pacbio-fiber, nanopore-fiber, daf)
            context_size: HMM context size
            edge_trim: Bases trimmed from read edges
            source_bam: Source BAM filename (for metadata)
            batch_size: Number of fibers to buffer before writing
        """
        output_path = os.fspath(output_path)
        self.output_path = output_path
        self.batch_size = batch_size

        # Open H5 file
        self.h5 = h5py.File(output_path, 'w')

        write_hdf5_file_metadata(
            self.h5,
            mode=mode,
            context_size=context_size,
            edge_trim=edge_trim,
            source_bam=source_bam,
        )

        # Per-chromosome state
        self._chrom_groups: Dict[str, h5py.Group] = {}
        self._chrom_counts: Dict[str, int] = {}
        self._chrom_buffers: Dict[str, List[Dict]] = {}
        self._chrom_metadata: Dict[str, _ChromMetadata] = {}

        self.total_written = 0
        self._finalized = False
        self._closed = False

    def _raise_if_closed(self):
        if self._closed:
            raise RuntimeError("PosteriorWriter is closed")

    def _raise_if_finalized(self):
        if self._finalized:
            raise RuntimeError("PosteriorWriter is finalized")

    def _ensure_chrom(self, chrom: str):
        """Create chromosome group if it doesn't exist."""
        if chrom not in self._chrom_groups:
            grp = create_posterior_chrom_group(self.h5, chrom)

            self._chrom_groups[chrom] = grp
            self._chrom_counts[chrom] = 0
            self._chrom_buffers[chrom] = []
            self._chrom_metadata[chrom] = _new_chrom_metadata()

    def add_fiber(self, chrom: str, fiber_data: Dict):
        """
        Add a fiber to the write buffer.

        Args:
            chrom: Chromosome name
            fiber_data: Dict with keys:
                - read_name: str
                - ref_start: int (reference start position)
                - ref_end: int (reference end position)
                - strand: str ('+', '-', or '.')
                - posteriors: np.ndarray float16, P(footprint) per position
                - ref_positions: np.ndarray int32, reference coord per query pos
                - footprint_starts: np.ndarray int32
                - footprint_sizes: np.ndarray int32
        """
        self._raise_if_closed()
        self._raise_if_finalized()

        self._ensure_chrom(chrom)
        self._chrom_buffers[chrom].append(fiber_data)

        # Flush if buffer is full
        if len(self._chrom_buffers[chrom]) >= self.batch_size:
            self._flush_chrom(chrom)

    def _flush_chrom(self, chrom: str):
        """Write buffered fibers for a chromosome to H5."""
        buffer = self._chrom_buffers[chrom]
        if not buffer:
            return

        grp = self._chrom_groups[chrom]
        start_idx = self._chrom_counts[chrom]
        meta = self._chrom_metadata[chrom]

        for i, fiber in enumerate(buffer):
            idx = start_idx + i

            # Write variable-length arrays with compression
            _write_fiber_array_datasets(grp, idx, fiber)

            # Accumulate metadata for final arrays
            _append_fiber_metadata(meta, fiber)

        self._chrom_counts[chrom] += len(buffer)
        self.total_written += len(buffer)
        self._chrom_buffers[chrom] = []

    def flush(self):
        """Flush all chromosome buffers."""
        self._raise_if_closed()
        for chrom in list(self._chrom_buffers.keys()):
            self._flush_chrom(chrom)

    def finalize(self):
        """Flush all buffers and write metadata arrays."""
        self._raise_if_closed()
        if self._finalized:
            return

        self.flush()

        for chrom, grp in self._chrom_groups.items():
            meta = self._chrom_metadata[chrom]
            n_fibers = self._chrom_counts[chrom]
            _write_chrom_fiber_metadata(grp, meta, n_fibers)
        self._finalized = True

    def close(self):
        """Finalize and close the H5 file."""
        if self._closed:
            return _PosteriorWriterCloseResult(
                total_written=self.total_written,
                file_size_mb=0,
            ).as_tuple()

        try:
            self.finalize()
        finally:
            try:
                self.h5.close()
            finally:
                self._closed = True

        # Return stats
        file_size = _file_size_mb(self.output_path)
        return _PosteriorWriterCloseResult(
            total_written=self.total_written,
            file_size_mb=file_size,
        ).as_tuple()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


def get_ref_positions_from_read(read) -> np.ndarray:
    """
    Extract query-to-reference position mapping from a pysam read.

    Returns array where ref_positions[query_idx] = reference_position
    or -1 for insertions (no reference position).
    """
    if read.is_unmapped:
        return np.array([], dtype=np.int32)

    return get_reference_positions_array(read)
