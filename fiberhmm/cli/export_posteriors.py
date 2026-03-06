#!/usr/bin/env python3
"""
Export HMM posterior probabilities for downstream analysis.

Supports two output formats:
  - TSV (gzipped): No extra dependencies, base64-encoded uint8 posteriors
  - HDF5: Requires h5py, streaming batched writes for large BAMs

Usage:
    # TSV output (no extra deps)
    python export_posteriors.py -i tagged.bam -m model.json -o posteriors.tsv.gz -c 4

    # HDF5 output (requires h5py)
    python export_posteriors.py -i tagged.bam -m model.json -o posteriors.h5 -c 4
"""

import argparse
import os
import sys
import time
import numpy as np
from typing import Dict, List, Optional, Tuple, Set
from concurrent.futures import ProcessPoolExecutor, as_completed
import pysam
from tqdm import tqdm

# Package imports
from fiberhmm.core.bam_reader import (
    encode_from_query_sequence, detect_daf_strand,
    get_reference_positions, ContextEncoder
)
from fiberhmm.core.model_io import load_model_with_metadata
from fiberhmm.core.hmm import FiberHMM
from fiberhmm.inference.parallel import _get_genome_regions
from fiberhmm.cli.common import (
    add_mode_args, add_parallel_args, add_edge_trim_args,
    add_verbose_args, add_version_args,
)


def _detect_format(output_path: str, format_arg: str) -> str:
    """Detect output format from extension or explicit --format."""
    if format_arg != 'auto':
        return format_arg
    if output_path.endswith('.h5') or output_path.endswith('.hdf5'):
        return 'hdf5'
    return 'tsv'


def extract_posteriors_from_read(read, model: FiberHMM, mode: str,
                                  context_size: int, edge_trim: int) -> Optional[Dict]:
    """
    Extract posterior probabilities from a single read.
    Returns dict with fiber data or None if filtered.
    """
    if read.is_unmapped or read.is_secondary or read.is_supplementary:
        return None

    sequence = read.query_sequence
    if sequence is None or len(sequence) < 100:
        return None

    # Get modification positions from MM/ML tags
    try:
        mod_dict = dict(read.modified_bases_forward)
    except (TypeError, ValueError):
        mod_dict = {}

    mod_positions = set()
    for key, positions in mod_dict.items():
        for pos, qual in positions:
            if qual >= 128:
                mod_positions.add(pos)

    if len(mod_positions) < 10:
        return None

    # Determine strand
    if mode == 'daf':
        strand = detect_daf_strand(sequence, mod_positions)
    else:
        strand = '-' if read.is_reverse else '+'

    # Encode read
    encoded = encode_from_query_sequence(
        sequence, mod_positions, edge_trim,
        mode=mode, strand=strand, context_size=context_size
    )

    if len(encoded) == 0:
        return None

    # Get posteriors using forward-backward
    posteriors = model.predict_proba(encoded)
    p_footprint = posteriors[:, 0].astype(np.float16)

    # Get reference position mapping
    q2r = get_reference_positions(read)
    ref_positions = np.array([p if p is not None else -1 for p in q2r], dtype=np.int32)

    # Viterbi path for footprint intervals
    states = model.predict(encoded)

    # Extract footprint intervals
    states_padded = np.concatenate([[1], states, [1]])
    diff = np.diff(states_padded)
    fp_start_idx = np.where(diff == -1)[0]
    fp_end_idx = np.where(diff == 1)[0]

    fp_starts_ref = []
    fp_sizes_ref = []

    if len(fp_start_idx) > 0 and len(ref_positions) > 0:
        for s, e in zip(fp_start_idx, fp_end_idx):
            s_clamped = min(s, len(ref_positions) - 1)
            e_clamped = min(e, len(ref_positions)) - 1

            ref_s = ref_positions[s_clamped]
            ref_e = ref_positions[e_clamped] if e_clamped >= 0 else ref_s

            if ref_s >= 0 and ref_e >= 0:
                fp_starts_ref.append(ref_s)
                fp_sizes_ref.append(max(1, ref_e - ref_s))

    return {
        'read_name': read.query_name,
        'ref_start': read.reference_start,
        'ref_end': read.reference_end,
        'strand': strand,
        'posteriors': p_footprint,
        'ref_positions': ref_positions,
        'footprint_starts': np.array(fp_starts_ref, dtype=np.int32),
        'footprint_sizes': np.array(fp_sizes_ref, dtype=np.int32),
    }


# Global worker state
_worker_model = None
_worker_params = None


def _init_worker(model_path: str, params: dict):
    """Initialize worker with model and warmup numba JIT."""
    global _worker_model, _worker_params

    # Disable numba caching to avoid file lock contention
    import os
    os.environ['NUMBA_CACHE_DIR'] = ''

    _worker_model, _, _ = load_model_with_metadata(model_path, normalize=True)
    _worker_params = params

    # Warmup numba JIT with a dummy sequence
    dummy = np.zeros(100, dtype=np.int32)
    try:
        _worker_model.predict(dummy)
        _worker_model.predict_proba(dummy)
    except:
        pass  # OK if warmup fails


def _process_region_worker(args) -> Tuple[str, int, int, List[Dict]]:
    """Worker to process a region and extract posteriors."""
    global _worker_model, _worker_params

    chrom, start, end, input_bam = args
    mode = _worker_params['mode']
    context_size = _worker_params['context_size']
    edge_trim = _worker_params['edge_trim']

    results = []

    with pysam.AlignmentFile(input_bam, "rb") as bam:
        for read in bam.fetch(chrom, start, end):
            result = extract_posteriors_from_read(
                read, _worker_model, mode, context_size, edge_trim
            )
            if result is not None:
                results.append(result)

    return (chrom, start, end, results)


def _write_batch_to_h5(grp, fibers: List[Dict], start_idx: int):
    """
    Write a batch of fibers to HDF5 efficiently.

    Uses batched dataset creation which is faster than individual creates.
    """
    n = len(fibers)
    if n == 0:
        return

    # Pre-allocate metadata arrays
    fiber_ids = []
    starts = np.zeros(n, dtype=np.int32)
    ends = np.zeros(n, dtype=np.int32)
    strands = []

    for i, fiber in enumerate(fibers):
        fiber_ids.append(fiber['read_name'])
        starts[i] = fiber['ref_start']
        ends[i] = fiber['ref_end']
        strands.append(fiber['strand'])

        idx = start_idx + i

        # Write variable-length arrays
        grp['posteriors'].create_dataset(
            str(idx), data=fiber['posteriors'],
            compression='gzip', compression_opts=4
        )
        grp['ref_positions'].create_dataset(
            str(idx), data=fiber['ref_positions'],
            compression='gzip', compression_opts=4
        )
        grp['footprint_starts'].create_dataset(
            str(idx), data=fiber['footprint_starts'], compression='gzip'
        )
        grp['footprint_sizes'].create_dataset(
            str(idx), data=fiber['footprint_sizes'], compression='gzip'
        )

    return fiber_ids, starts, ends, strands


def _process_regions(regions, input_bam, model_path, params,
                     n_cores, verbose, result_callback):
    """Process all regions and call result_callback(chrom, results) for each."""
    if n_cores > 1:
        with ProcessPoolExecutor(
            max_workers=n_cores,
            initializer=_init_worker,
            initargs=(model_path, params)
        ) as executor:
            pending = {}
            region_iter = iter(regions)
            max_pending = n_cores * 2

            for _ in range(min(max_pending, len(regions))):
                try:
                    chrom, start, end = next(region_iter)
                    args = (chrom, start, end, input_bam)
                    future = executor.submit(_process_region_worker, args)
                    pending[future] = (chrom, start, end)
                except StopIteration:
                    break

            pbar = tqdm(total=len(regions), desc="Processing regions", disable=not verbose)

            while pending:
                done = [fut for fut in pending if fut.done()]

                if not done:
                    time.sleep(0.02)
                    continue

                for future in done:
                    region_info = pending.pop(future)

                    try:
                        chrom, start, end, results = future.result()
                        result_callback(chrom, results)
                        del results
                    except Exception as e:
                        print(f"Error processing {region_info}: {e}")

                    pbar.update(1)

                    try:
                        chrom, start, end = next(region_iter)
                        args = (chrom, start, end, input_bam)
                        future = executor.submit(_process_region_worker, args)
                        pending[future] = (chrom, start, end)
                    except StopIteration:
                        pass

            pbar.close()
    else:
        _init_worker(model_path, params)
        for chrom, start, end in tqdm(regions, desc="Processing regions", disable=not verbose):
            args = (chrom, start, end, input_bam)
            _, _, _, results = _process_region_worker(args)
            result_callback(chrom, results)
            del results


def export_posteriors_tsv(
    input_bam: str,
    model_path: str,
    output_path: str,
    chroms: Optional[Set[str]] = None,
    edge_trim: int = 100,
    n_cores: int = 4,
    region_size: int = 5_000_000,
    verbose: bool = True,
    mode_override: str = None,
    context_size_override: int = None,
) -> int:
    """Export posterior probabilities to gzipped TSV."""
    from fiberhmm.posteriors.tsv_backend import PosteriorsTSVWriter

    model, model_context_size, model_mode = load_model_with_metadata(model_path, normalize=True)
    mode = mode_override if mode_override else model_mode
    context_size = context_size_override if context_size_override else model_context_size

    if verbose:
        print(f"Loaded model: mode={model_mode}, context_size={model_context_size}")
        if mode_override:
            print(f"  Mode override: {mode}")

    regions = _get_genome_regions(input_bam, region_size, chroms=chroms)
    if verbose:
        print(f"Processing {len(regions)} regions from {input_bam}")
        print(f"Using {n_cores} cores, output format: TSV")

    params = {
        'mode': mode,
        'context_size': context_size,
        'edge_trim': edge_trim
    }

    compress = output_path.endswith('.gz')
    writer = PosteriorsTSVWriter(
        output_path, mode=mode, context_size=context_size,
        edge_trim=edge_trim, source_bam=input_bam, compress=compress
    )

    def on_results(chrom, results):
        for fiber in results:
            writer.write_fiber(
                read_id=fiber['read_name'],
                chrom=chrom,
                start=fiber['ref_start'],
                end=fiber['ref_end'],
                strand=fiber['strand'],
                posteriors=fiber['posteriors'].astype(np.float32),
                fp_starts=fiber['footprint_starts'],
                fp_sizes=fiber['footprint_sizes'],
            )

    _process_regions(regions, input_bam, model_path, params, n_cores, verbose, on_results)

    total = writer.close()

    if verbose:
        out_file = writer.output_path
        file_size = os.path.getsize(out_file) / (1024 * 1024)
        print(f"Wrote {out_file} ({file_size:.1f} MB, {total:,} fibers)")

    return total


def export_posteriors_hdf5(
    input_bam: str,
    model_path: str,
    output_h5: str,
    chroms: Optional[Set[str]] = None,
    edge_trim: int = 100,
    n_cores: int = 4,
    region_size: int = 5_000_000,
    write_batch_size: int = 1000,
    verbose: bool = True,
    mode_override: str = None,
    context_size_override: int = None,
) -> int:
    """
    Export posterior probabilities to HDF5.

    STREAMING + BATCHED: Results written in batches as regions complete.
    Memory usage stays bounded regardless of BAM size.
    """
    import h5py

    model, model_context_size, model_mode = load_model_with_metadata(model_path, normalize=True)
    mode = mode_override if mode_override else model_mode
    context_size = context_size_override if context_size_override else model_context_size

    if verbose:
        print(f"Loaded model: mode={model_mode}, context_size={model_context_size}")
        if mode_override:
            print(f"  Mode override: {mode}")

    regions = _get_genome_regions(input_bam, region_size, chroms=chroms)
    if verbose:
        print(f"Processing {len(regions)} regions from {input_bam}")
        print(f"Using {n_cores} cores with streaming/batched writes, output format: HDF5")

    params = {
        'mode': mode,
        'context_size': context_size,
        'edge_trim': edge_trim
    }

    # Group regions by chromosome
    regions_by_chrom = {}
    for chrom, start, end in regions:
        if chrom not in regions_by_chrom:
            regions_by_chrom[chrom] = []
        regions_by_chrom[chrom].append((start, end))

    # Track per-chromosome data
    chrom_fiber_counts = {chrom: 0 for chrom in regions_by_chrom}
    chrom_metadata = {chrom: {'ids': [], 'starts': [], 'ends': [], 'strands': []}
                      for chrom in regions_by_chrom}

    # Pending writes buffer per chromosome
    write_buffers = {chrom: [] for chrom in regions_by_chrom}

    with h5py.File(output_h5, 'w') as f:
        # Store file metadata
        f.attrs['mode'] = mode
        f.attrs['context_size'] = context_size
        f.attrs['model_path'] = os.path.basename(model_path)
        f.attrs['edge_trim'] = edge_trim
        f.attrs['source_bam'] = os.path.basename(input_bam)
        f.attrs['format_version'] = 2

        # Pre-create chromosome groups
        for chrom in regions_by_chrom:
            grp = f.create_group(chrom)
            grp.create_group('posteriors')
            grp.create_group('ref_positions')
            grp.create_group('footprint_starts')
            grp.create_group('footprint_sizes')

        def flush_buffer(chrom):
            """Write buffered fibers to HDF5."""
            buffer = write_buffers[chrom]
            if not buffer:
                return

            grp = f[chrom]
            start_idx = chrom_fiber_counts[chrom]

            ids, starts, ends, strands = _write_batch_to_h5(grp, buffer, start_idx)

            # Accumulate metadata
            chrom_metadata[chrom]['ids'].extend(ids)
            chrom_metadata[chrom]['starts'].append(starts)
            chrom_metadata[chrom]['ends'].append(ends)
            chrom_metadata[chrom]['strands'].extend(strands)

            chrom_fiber_counts[chrom] += len(buffer)
            write_buffers[chrom] = []

        def on_results(chrom, results):
            write_buffers[chrom].extend(results)
            if len(write_buffers[chrom]) >= write_batch_size:
                flush_buffer(chrom)

        _process_regions(regions, input_bam, model_path, params, n_cores, verbose, on_results)

        # Flush remaining buffers
        for chrom in regions_by_chrom:
            flush_buffer(chrom)

        # Finalize metadata (fast - just concatenating pre-built arrays)
        if verbose:
            print("Finalizing metadata...")

        dt = h5py.special_dtype(vlen=str)
        for chrom in regions_by_chrom:
            grp = f[chrom]
            meta = chrom_metadata[chrom]
            n_fibers = chrom_fiber_counts[chrom]

            if n_fibers == 0:
                grp.attrs['n_fibers'] = 0
                continue

            # Concatenate pre-built arrays (fast)
            grp.create_dataset('fiber_ids', data=meta['ids'], dtype=dt)
            grp.create_dataset('fiber_starts',
                              data=np.concatenate(meta['starts']) if meta['starts'] else np.array([], dtype=np.int32),
                              compression='gzip')
            grp.create_dataset('fiber_ends',
                              data=np.concatenate(meta['ends']) if meta['ends'] else np.array([], dtype=np.int32),
                              compression='gzip')
            grp.create_dataset('strands', data=meta['strands'], dtype=dt)
            grp.attrs['n_fibers'] = n_fibers

    total_fibers = sum(chrom_fiber_counts.values())

    if verbose:
        file_size = os.path.getsize(output_h5) / (1024 * 1024)
        print(f"Wrote {output_h5} ({file_size:.1f} MB, {total_fibers:,} fibers)")

    return total_fibers


def export_posteriors(
    input_bam: str,
    model_path: str,
    output_path: str,
    format: str = 'auto',
    chroms: Optional[Set[str]] = None,
    edge_trim: int = 100,
    n_cores: int = 4,
    region_size: int = 5_000_000,
    write_batch_size: int = 1000,
    verbose: bool = True,
    mode_override: str = None,
    context_size_override: int = None,
) -> int:
    """
    Export posterior probabilities to TSV or HDF5.

    Dispatches to format-specific implementation.
    """
    fmt = _detect_format(output_path, format)

    if fmt == 'hdf5':
        return export_posteriors_hdf5(
            input_bam=input_bam,
            model_path=model_path,
            output_h5=output_path,
            chroms=chroms,
            edge_trim=edge_trim,
            n_cores=n_cores,
            region_size=region_size,
            write_batch_size=write_batch_size,
            verbose=verbose,
            mode_override=mode_override,
            context_size_override=context_size_override,
        )
    else:
        return export_posteriors_tsv(
            input_bam=input_bam,
            model_path=model_path,
            output_path=output_path,
            chroms=chroms,
            edge_trim=edge_trim,
            n_cores=n_cores,
            region_size=region_size,
            verbose=verbose,
            mode_override=mode_override,
            context_size_override=context_size_override,
        )


# === Reader classes ===

class PosteriorReader:
    """Read HMM posteriors from HDF5 for downstream analysis."""

    def __init__(self, h5_path: str):
        import h5py
        self.h5_path = h5_path
        self.h5 = h5py.File(h5_path, 'r')

        self.mode = self.h5.attrs.get('mode', 'pacbio-fiber')
        self.context_size = self.h5.attrs.get('context_size', 3)
        self.format_version = self.h5.attrs.get('format_version', 1)

        self._chrom_info = {}
        for chrom in self.h5.keys():
            import h5py as _h5py
            if isinstance(self.h5[chrom], _h5py.Group):
                grp = self.h5[chrom]
                n_fibers = grp.attrs.get('n_fibers', 0)
                if n_fibers > 0:
                    self._chrom_info[chrom] = {
                        'n_fibers': n_fibers,
                        'starts': grp['fiber_starts'][:],
                        'ends': grp['fiber_ends'][:],
                    }

    @property
    def chromosomes(self) -> List[str]:
        return list(self._chrom_info.keys())

    def get_n_fibers(self, chrom: str) -> int:
        if chrom not in self._chrom_info:
            return 0
        return self._chrom_info[chrom]['n_fibers']

    def get_fibers_overlapping(self, chrom: str, start: int, end: int,
                                min_overlap: int = 1) -> List['FiberPosterior']:
        if chrom not in self._chrom_info:
            return []

        info = self._chrom_info[chrom]
        overlaps = (info['ends'] > start + min_overlap) & (info['starts'] < end - min_overlap)
        indices = np.where(overlaps)[0]

        return self._load_fibers(chrom, indices)

    def get_fibers_spanning(self, chrom: str, start: int, end: int) -> List['FiberPosterior']:
        if chrom not in self._chrom_info:
            return []

        info = self._chrom_info[chrom]
        spans = (info['starts'] <= start) & (info['ends'] >= end)
        indices = np.where(spans)[0]

        return self._load_fibers(chrom, indices)

    def _load_fibers(self, chrom: str, indices: np.ndarray) -> List['FiberPosterior']:
        import h5py
        grp = self.h5[chrom]
        ids = grp['fiber_ids']
        starts = grp['fiber_starts']
        ends = grp['fiber_ends']
        strands = grp.get('strands', None)

        post_grp = grp['posteriors']
        ref_pos_grp = grp.get('ref_positions', None)
        fp_starts_grp = grp.get('footprint_starts', None)
        fp_sizes_grp = grp.get('footprint_sizes', None)

        fibers = []
        for idx in indices:
            idx = int(idx)
            posteriors = post_grp[str(idx)][:]

            ref_positions = ref_pos_grp[str(idx)][:] if ref_pos_grp else None
            fp_starts = fp_starts_grp[str(idx)][:] if fp_starts_grp else np.array([], dtype=np.int32)
            fp_sizes = fp_sizes_grp[str(idx)][:] if fp_sizes_grp else np.array([], dtype=np.int32)

            strand = strands[idx] if strands is not None else '.'
            if isinstance(strand, bytes):
                strand = strand.decode()

            fibers.append(FiberPosterior(
                fiber_id=ids[idx] if isinstance(ids[idx], str) else ids[idx].decode(),
                start=int(starts[idx]),
                end=int(ends[idx]),
                strand=strand,
                posteriors=posteriors.astype(np.float32),
                ref_positions=ref_positions,
                footprint_starts=fp_starts,
                footprint_sizes=fp_sizes,
            ))

        return fibers

    def close(self):
        self.h5.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class FiberPosterior:
    """Represents a single fiber with HMM posterior probabilities."""

    def __init__(self, fiber_id: str, start: int, end: int, strand: str,
                 posteriors: np.ndarray, ref_positions: Optional[np.ndarray],
                 footprint_starts: np.ndarray, footprint_sizes: np.ndarray):
        self.fiber_id = fiber_id
        self.start = start
        self.end = end
        self.strand = strand
        self.posteriors = posteriors
        self.ref_positions = ref_positions
        self.footprint_starts = footprint_starts
        self.footprint_sizes = footprint_sizes

    def __repr__(self):
        return f"FiberPosterior({self.fiber_id}, {self.start}-{self.end}, {len(self.posteriors)} positions)"

    def project_to_reference(self, region_start: int, region_end: int,
                              method: str = 'max') -> np.ndarray:
        if self.ref_positions is None:
            raise ValueError("No reference position mapping")

        region_len = region_end - region_start
        result = np.zeros(region_len, dtype=np.float32)
        counts = np.zeros(region_len, dtype=np.int32)

        for q_idx, ref_pos in enumerate(self.ref_positions):
            if ref_pos < 0 or ref_pos < region_start or ref_pos >= region_end:
                continue

            rel_pos = ref_pos - region_start
            post_val = self.posteriors[q_idx]

            if method == 'max':
                result[rel_pos] = max(result[rel_pos], post_val)
            elif method == 'mean':
                result[rel_pos] += post_val
                counts[rel_pos] += 1
            else:
                if counts[rel_pos] == 0:
                    result[rel_pos] = post_val
                    counts[rel_pos] = 1

        if method == 'mean':
            valid = counts > 0
            result[valid] /= counts[valid]

        return result

    def get_footprint_coverage(self, region_start: int, region_end: int) -> np.ndarray:
        region_len = region_end - region_start
        coverage = np.zeros(region_len, dtype=np.float32)

        for fp_start, fp_size in zip(self.footprint_starts, self.footprint_sizes):
            s = max(0, fp_start - region_start)
            e = min(region_len, fp_start + fp_size - region_start)
            if s < e:
                coverage[s:e] = 1.0

        return coverage

    def spans_region(self, start: int, end: int) -> bool:
        return self.start <= start and self.end >= end


def main():
    parser = argparse.ArgumentParser(
        description="Export HMM posterior probabilities for downstream analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('-i', '--input', required=True, help='Input BAM file')
    parser.add_argument('-m', '--model', required=True, help='FiberHMM model file')
    parser.add_argument('-o', '--output', required=True,
                       help='Output file (.tsv.gz for TSV, .h5/.hdf5 for HDF5)')
    parser.add_argument('--format', choices=['auto', 'hdf5', 'tsv'], default='auto',
                       help="Output format (default: auto-detect from extension)")

    add_mode_args(parser, required=False)
    add_edge_trim_args(parser, default=100)
    add_parallel_args(parser, default_cores=4, default_region_size=5_000_000)

    parser.add_argument('--batch-size', type=int, default=1000,
                       help='Fibers per HDF5 write batch')

    add_verbose_args(parser)
    add_version_args(parser)

    args = parser.parse_args()

    chroms = set(args.chroms) if args.chroms else None

    export_posteriors(
        input_bam=args.input,
        model_path=args.model,
        output_path=args.output,
        format=args.format,
        chroms=chroms,
        edge_trim=args.edge_trim,
        n_cores=args.cores,
        region_size=args.region_size,
        write_batch_size=args.batch_size,
        verbose=args.verbose or True,
        mode_override=args.mode,
    )


# Alias for apply_model.py integration
def export_posteriors_from_bam(
    input_bam: str,
    model_path: str,
    output_path: str,
    format: str = 'auto',
    mode: str = None,
    context_size: int = None,
    edge_trim: int = 100,
    min_mapq: int = 20,
    prob_threshold: int = 125,
    n_workers: int = 4,
    chroms: Optional[Set[str]] = None,
    verbose: bool = True,
) -> int:
    return export_posteriors(
        input_bam=input_bam,
        model_path=model_path,
        output_path=output_path,
        format=format,
        chroms=chroms,
        edge_trim=edge_trim,
        n_cores=n_workers,
        verbose=verbose,
        mode_override=mode,
        context_size_override=context_size,
    )


if __name__ == '__main__':
    main()
