"""Legacy chunked BAM apply pipeline."""

from __future__ import annotations

import sys
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Optional, Set, Tuple

import numpy as np
import pysam

from fiberhmm.inference.bam_output import _sort_and_index_bam
from fiberhmm.inference.engine import (
    _extract_fiber_read_from_pysam,
    _process_single_read,
)
from fiberhmm.inference.mp_context import _MP_CONTEXT
from fiberhmm.inference.streaming_workers import (
    _init_bam_worker,
    _process_chunk_worker,
)
from fiberhmm.inference.tagging import set_legacy_apply_tags
from fiberhmm.inference.worker_results import coerce_worker_chunk_result

# Optional: inline posteriors export
try:
    from fiberhmm.posteriors.hdf5_backend import PosteriorWriter, get_ref_positions_from_read
    HAS_POSTERIOR_WRITER = True
except ImportError:
    HAS_POSTERIOR_WRITER = False


def _process_and_write_chunk(chunk_reads: list, chunk_read_objs: list,
                              outbam, model, executor,
                              edge_trim: int, circular: bool,
                              mode: str, context_size: int,
                              msp_min_size: int, nuc_min_size: int = 85,
                              with_scores: bool = False,
                              return_posteriors: bool = False,
                              write_msps: bool = True) -> Tuple[int, int, int, Optional[list]]:
    """
    Process a chunk of reads and write to BAM.

    Returns:
        (reads_with_footprints, no_footprints, worker_failures,
         results_with_posteriors or None)

        If return_posteriors=True, returns results list for caller to write posteriors.
    """

    if executor is not None:
        # Parallel: submit chunk to worker
        future = executor.submit(
            _process_chunk_worker,
            chunk_reads, edge_trim, circular, mode, context_size, msp_min_size,
            nuc_min_size, with_scores, return_posteriors
        )
        results, worker_failures = coerce_worker_chunk_result(future.result())
    else:
        # Single-threaded: process directly
        results = []
        worker_failures = 0
        for fiber_read in chunk_reads:
            result = _process_single_read(
                fiber_read, model, edge_trim, circular,
                mode, context_size, msp_min_size, nuc_min_size=nuc_min_size,
                with_scores=with_scores,
                return_posteriors=return_posteriors
            )
            results.append(result)

    # Write annotated reads
    reads_with_footprints = 0
    no_footprints = 0
    for read_obj, result in zip(chunk_read_objs, results):
        if result is not None:
            set_legacy_apply_tags(read_obj, result, with_scores, write_msps)
            reads_with_footprints += 1
        else:
            no_footprints += 1

        outbam.write(read_obj)

    # Return results for posteriors if requested
    if return_posteriors:
        return (
            reads_with_footprints,
            no_footprints,
            worker_failures,
            list(zip(chunk_read_objs, chunk_reads, results)),
        )
    return reads_with_footprints, no_footprints, worker_failures, None


def _write_chunk_posteriors(posterior_writer, chunk_results):
    if not posterior_writer or not chunk_results:
        return

    for read_obj, fiber_read_data, result in chunk_results:
        if result and result.get('posteriors') is not None:
            chrom = read_obj.reference_name
            if chrom:
                ref_positions = (
                    get_ref_positions_from_read(read_obj)
                    if HAS_POSTERIOR_WRITER
                    else np.array([], dtype=np.int32)
                )
                posterior_writer.add_fiber(chrom, {
                    'read_name': read_obj.query_name,
                    'ref_start': read_obj.reference_start,
                    'ref_end': read_obj.reference_end,
                    'strand': result.get('strand', '.'),
                    'posteriors': result['posteriors'],
                    'ref_positions': ref_positions,
                    'footprint_starts': result['ns'],
                    'footprint_sizes': result['nl'],
                })


def _process_bam_legacy_pipeline(
    input_bam: str,
    output_bam: str,
    model,
    model_path: Optional[str],
    train_rids: Set[str],
    edge_trim: int,
    circular: bool,
    mode: str,
    context_size: int,
    msp_min_size: int,
    nuc_min_size: int = 85,
    min_mapq: int = 0,
    prob_threshold: int = 0,
    min_read_length: int = 0,
    with_scores: bool = False,
    n_cores: int = 1,
    max_reads: Optional[int] = None,
    debug_timing: bool = False,
    primary_only: bool = False,
    output_posteriors: Optional[str] = None,
    write_msps: bool = True,
    io_threads: int = 4,
) -> Tuple[int, int]:
    """Process a BAM through the legacy chunked apply path."""
    total_reads = 0
    reads_with_footprints = 0
    written = 0
    skipped = 0
    worker_failures = 0

    # Track skip reasons
    skip_reasons = {
        'unmapped': 0,
        'secondary_supplementary': 0,
        'low_mapq': 0,
        'too_short': 0,
        'training_excluded': 0,
        'no_modifications': 0,
        'extraction_failed': 0,
        'no_footprints': 0,
    }

    chunk_size = 2000  # Reads per chunk
    start_time = time.time()

    # Initialized after BAM handles open so failure cleanup is centralized in
    # the processing finally block.
    posterior_writer = None
    posterior_stats = None
    return_posteriors = False

    print("Processing and writing BAM (streaming)...")
    sys.stdout.flush()

    # Open input and output BAMs
    with pysam.AlignmentFile(input_bam, "rb", threads=io_threads, check_sq=False) as inbam:
        with pysam.AlignmentFile(output_bam, "wb", header=inbam.header, threads=io_threads) as outbam:

            if output_posteriors:
                if HAS_POSTERIOR_WRITER:
                    posterior_writer = PosteriorWriter(
                        output_posteriors, mode, context_size,
                        edge_trim, input_bam, batch_size=1000
                    )
                    return_posteriors = True
                    print(f"Posteriors will be written to: {output_posteriors}")
                else:
                    print("WARNING: posterior_writer.py not found, skipping posteriors export")

            chunk_reads = []  # Buffer for current chunk
            chunk_read_objs = []  # Corresponding pysam read objects

            if n_cores > 1 and model_path:
                # Parallel processing with ProcessPoolExecutor
                executor = ProcessPoolExecutor(
                    max_workers=n_cores,
                    mp_context=_MP_CONTEXT,
                    initializer=_init_bam_worker,
                    initargs=(model_path, debug_timing)
                )
            else:
                executor = None

            try:
                for read in inbam:
                    # Pass through unmapped reads (no sequence to process)
                    if read.is_unmapped:
                        outbam.write(read)
                        written += 1
                        skipped += 1
                        skip_reasons['unmapped'] += 1
                        continue

                    # Skip secondary/supplementary if primary_only mode
                    if primary_only and (read.is_secondary or read.is_supplementary):
                        outbam.write(read)
                        written += 1
                        skipped += 1
                        skip_reasons['secondary_supplementary'] += 1
                        continue

                    # Check filters (apply to all alignments)
                    if read.mapping_quality < min_mapq:
                        outbam.write(read)
                        written += 1
                        skipped += 1
                        skip_reasons['low_mapq'] += 1
                        continue

                    if read.query_alignment_length < min_read_length:
                        outbam.write(read)
                        written += 1
                        skipped += 1
                        skip_reasons['too_short'] += 1
                        continue

                    read_id = read.query_name
                    if train_rids and read_id in train_rids:
                        outbam.write(read)
                        written += 1
                        skipped += 1
                        skip_reasons['training_excluded'] += 1
                        continue

                    # Extract data needed for processing
                    try:
                        fiber_read = _extract_fiber_read_from_pysam(read, mode, prob_threshold)
                        if fiber_read is None:
                            outbam.write(read)
                            written += 1
                            skipped += 1
                            skip_reasons['no_modifications'] += 1
                            continue
                    except Exception:
                        outbam.write(read)
                        written += 1
                        skipped += 1
                        skip_reasons['extraction_failed'] += 1
                        continue

                    chunk_reads.append(fiber_read)
                    chunk_read_objs.append(read)
                    total_reads += 1

                    # Check max reads limit
                    if max_reads and total_reads >= max_reads:
                        # Process final chunk and exit
                        break

                    # Process chunk when full
                    if len(chunk_reads) >= chunk_size:
                        n_fp, n_nofp, n_failed, chunk_results = _process_and_write_chunk(
                            chunk_reads, chunk_read_objs, outbam,
                            model, executor, edge_trim, circular,
                            mode, context_size, msp_min_size, nuc_min_size=nuc_min_size,
                            with_scores=with_scores,
                            return_posteriors=return_posteriors,
                            write_msps=write_msps
                        )
                        reads_with_footprints += n_fp
                        skip_reasons['no_footprints'] += n_nofp
                        worker_failures += n_failed
                        written += len(chunk_read_objs)

                        _write_chunk_posteriors(posterior_writer, chunk_results)

                        # Progress update
                        elapsed = time.time() - start_time
                        rate = total_reads / elapsed if elapsed > 0 else 0
                        print(f"\r  Processed: {total_reads:,} | Skipped: {skipped:,} | {rate:.1f} reads/s", end='')
                        sys.stdout.flush()

                        chunk_reads = []
                        chunk_read_objs = []

                # Process final partial chunk
                if chunk_reads:
                    n_fp, n_nofp, n_failed, chunk_results = _process_and_write_chunk(
                        chunk_reads, chunk_read_objs, outbam,
                        model, executor, edge_trim, circular,
                        mode, context_size, msp_min_size, nuc_min_size=nuc_min_size,
                        with_scores=with_scores,
                        return_posteriors=return_posteriors,
                        write_msps=write_msps,
                    )
                    reads_with_footprints += n_fp
                    skip_reasons['no_footprints'] += n_nofp
                    worker_failures += n_failed
                    written += len(chunk_read_objs)

                    _write_chunk_posteriors(posterior_writer, chunk_results)

            finally:
                try:
                    if executor:
                        executor.shutdown(wait=True)
                finally:
                    if posterior_writer:
                        posterior_stats = posterior_writer.close()
                        posterior_writer = None

    elapsed = time.time() - start_time
    rate = total_reads / elapsed if elapsed > 0 else 0
    print(f"\r  Processed: {total_reads:,} | Skipped: {skipped:,} | With footprints: {reads_with_footprints:,} | {rate:.1f} reads/s")
    if worker_failures:
        print(f"  Worker read failures: {worker_failures:,} (passed through unchanged)")

    # Print skip reasons summary
    if skipped > 0:
        print("  Skip reasons:")
        for reason, count in sorted(skip_reasons.items(), key=lambda x: -x[1]):
            if count > 0:
                pct = 100 * count / (total_reads + skipped)
                print(f"    {reason}: {count:,} ({pct:.1f}%)")

    # Index the output BAM (sort first if needed)
    _sort_and_index_bam(output_bam, threads=n_cores)

    # Report posteriors after the writer has been closed by the processing finally block.
    if posterior_stats:
        n_fibers, file_size = posterior_stats
        print(f"Posteriors: {n_fibers:,} fibers -> {output_posteriors} ({file_size:.1f} MB)")

    return total_reads, reads_with_footprints
