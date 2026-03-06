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
import numpy as np
import h5py
from typing import Dict, List, Optional


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

        # Write file-level metadata
        self.h5.attrs['mode'] = mode
        self.h5.attrs['context_size'] = context_size
        self.h5.attrs['edge_trim'] = edge_trim
        self.h5.attrs['source_bam'] = os.path.basename(source_bam)
        self.h5.attrs['format_version'] = 2

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
            grp['posteriors'].create_dataset(
                str(idx),
                data=fiber['posteriors'].astype(np.float16),
                compression='gzip',
                compression_opts=4
            )

            ref_pos = fiber.get('ref_positions')
            if ref_pos is None or len(ref_pos) == 0:
                ref_pos = np.array([], dtype=np.int32)
            grp['ref_positions'].create_dataset(
                str(idx),
                data=ref_pos.astype(np.int32),
                compression='gzip',
                compression_opts=4
            )

            fp_starts = fiber.get('footprint_starts', np.array([], dtype=np.int32))
            fp_sizes = fiber.get('footprint_sizes', np.array([], dtype=np.int32))

            grp['footprint_starts'].create_dataset(
                str(idx),
                data=fp_starts.astype(np.int32),
                compression='gzip'
            )
            grp['footprint_sizes'].create_dataset(
                str(idx),
                data=fp_sizes.astype(np.int32),
                compression='gzip'
            )

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

        # Write metadata arrays for each chromosome
        dt = h5py.special_dtype(vlen=str)

        for chrom, grp in self._chrom_groups.items():
            meta = self._chrom_metadata[chrom]
            n_fibers = self._chrom_counts[chrom]

            if n_fibers == 0:
                grp.attrs['n_fibers'] = 0
                continue

            grp.create_dataset('fiber_ids', data=meta['ids'], dtype=dt)
            grp.create_dataset(
                'fiber_starts',
                data=np.array(meta['starts'], dtype=np.int32),
                compression='gzip'
            )
            grp.create_dataset(
                'fiber_ends',
                data=np.array(meta['ends'], dtype=np.int32),
                compression='gzip'
            )
            grp.create_dataset('strands', data=meta['strands'], dtype=dt)
            grp.attrs['n_fibers'] = n_fibers

    def close(self):
        """Finalize and close the H5 file."""
        if self._closed:
            return self.total_written, 0

        self.finalize()
        self.h5.close()
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

    # Get aligned pairs: (query_pos, ref_pos)
    # ref_pos is None for insertions
    pairs = read.get_aligned_pairs()

    query_len = read.query_length
    ref_positions = np.full(query_len, -1, dtype=np.int32)

    for query_pos, ref_pos in pairs:
        if query_pos is not None and ref_pos is not None:
            ref_positions[query_pos] = ref_pos

    return ref_positions
