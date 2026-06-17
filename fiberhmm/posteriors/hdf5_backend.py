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
from typing import Dict, List, Optional

import h5py
import numpy as np

from fiberhmm.core.bam_reader import get_reference_positions_array


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


def write_fiber_metadata_datasets(
    group,
    fiber_ids,
    starts,
    ends,
    strands,
    n_fibers: Optional[int] = None,
) -> None:
    n = len(fiber_ids) if n_fibers is None else int(n_fibers)
    if n == 0:
        group.attrs['n_fibers'] = 0
        return

    dt = h5py.special_dtype(vlen=str)
    group.create_dataset('fiber_ids', data=fiber_ids, dtype=dt)
    _create_gzip_dataset(group, 'fiber_starts', _int32_array(starts))
    _create_gzip_dataset(group, 'fiber_ends', _int32_array(ends))
    group.create_dataset('strands', data=strands, dtype=dt)
    group.attrs['n_fibers'] = n


def write_hdf5_file_metadata(
    h5_file,
    *,
    mode: str,
    context_size: int,
    edge_trim: int,
    source_bam: str,
    model_path: str = None,
) -> None:
    h5_file.attrs['mode'] = mode
    h5_file.attrs['context_size'] = context_size
    if model_path is not None:
        h5_file.attrs['model_path'] = os.path.basename(model_path)
    h5_file.attrs['edge_trim'] = edge_trim
    h5_file.attrs['source_bam'] = os.path.basename(source_bam)
    h5_file.attrs['format_version'] = 2


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
        self._chrom_metadata: Dict[str, Dict[str, List]] = {}

        self.total_written = 0
        self._closed = False

    def _ensure_chrom(self, chrom: str):
        """Create chromosome group if it doesn't exist."""
        if chrom not in self._chrom_groups:
            grp = self.h5.create_group(chrom)
            grp.create_group('posteriors')
            grp.create_group('ref_positions')
            grp.create_group('footprint_starts')
            grp.create_group('footprint_sizes')

            self._chrom_groups[chrom] = grp
            self._chrom_counts[chrom] = 0
            self._chrom_buffers[chrom] = []
            self._chrom_metadata[chrom] = {
                'ids': [], 'starts': [], 'ends': [], 'strands': []
            }

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
        if self._closed:
            raise RuntimeError("PosteriorWriter is closed")

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
            _create_gzip_dataset(
                grp['posteriors'],
                str(idx),
                fiber['posteriors'].astype(np.float16),
                compression_opts=4,
            )

            ref_pos = _int32_array(fiber.get('ref_positions'))
            _create_gzip_dataset(
                grp['ref_positions'],
                str(idx),
                ref_pos,
                compression_opts=4,
            )

            fp_starts = _int32_array(fiber.get('footprint_starts'))
            fp_sizes = _int32_array(fiber.get('footprint_sizes'))

            _create_gzip_dataset(grp['footprint_starts'], str(idx), fp_starts)
            _create_gzip_dataset(grp['footprint_sizes'], str(idx), fp_sizes)

            # Accumulate metadata for final arrays
            meta['ids'].append(fiber['read_name'])
            meta['starts'].append(fiber['ref_start'])
            meta['ends'].append(fiber['ref_end'])
            meta['strands'].append(fiber.get('strand', '.'))

        self._chrom_counts[chrom] += len(buffer)
        self.total_written += len(buffer)
        self._chrom_buffers[chrom] = []

    def flush(self):
        """Flush all chromosome buffers."""
        for chrom in list(self._chrom_buffers.keys()):
            self._flush_chrom(chrom)

    def finalize(self):
        """Flush all buffers and write metadata arrays."""
        self.flush()

        for chrom, grp in self._chrom_groups.items():
            meta = self._chrom_metadata[chrom]
            n_fibers = self._chrom_counts[chrom]

            write_fiber_metadata_datasets(
                grp,
                meta['ids'],
                meta['starts'],
                meta['ends'],
                meta['strands'],
                n_fibers=n_fibers,
            )

    def close(self):
        """Finalize and close the H5 file."""
        if self._closed:
            return self.total_written, 0

        try:
            self.finalize()
        finally:
            try:
                self.h5.close()
            finally:
                self._closed = True

        # Return stats
        file_size = os.path.getsize(self.output_path) / (1024 * 1024)
        return self.total_written, file_size

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
