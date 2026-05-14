"""FiberHMM BAM processing compatibility orchestration."""

import sys
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Optional, Set, Tuple

import numpy as np
import pysam

from fiberhmm.core.model_io import freeze_model_for_inference, load_model
from fiberhmm.inference.bam_output import (
    _sort_and_index_bam,
)
from fiberhmm.inference.engine import (
    _extract_fiber_read_from_pysam,
    _process_single_read,
)
from fiberhmm.inference.mp_context import _MP_CONTEXT, _select_mp_context  # noqa: F401
from fiberhmm.inference.read_filters import (  # noqa: F401
    ReadFilterConfig,
    streaming_skip_reason,
)
from fiberhmm.inference.region_pipeline import (
    _process_bam_region_parallel,
    _process_bam_region_parallel_fused,  # noqa: F401
    _process_bed_region_parallel,  # noqa: F401
)
from fiberhmm.inference.region_planning import (
    _get_genome_regions,  # noqa: F401
)
from fiberhmm.inference.region_planning import (
    _is_main_chromosome as _region_is_main_chromosome,
)
from fiberhmm.inference.region_workers import (
    _init_fused_region_worker,  # noqa: F401
    _init_region_worker,  # noqa: F401
    _process_region_to_bam,  # noqa: F401
    _process_region_to_bam_fused,  # noqa: F401
    _process_region_to_bed,  # noqa: F401
)
from fiberhmm.inference.streaming_drain import (
    _drain_oldest_chunk,  # noqa: F401
    _drain_oldest_fused_chunk,  # noqa: F401
)
from fiberhmm.inference.streaming_pipeline import (
    _process_bam_streaming_pipeline,
    _process_bam_streaming_pipeline_fused,  # noqa: F401
)
from fiberhmm.inference.streaming_workers import (
    _init_bam_worker,
    _init_fused_worker,  # noqa: F401
    _process_chunk_worker,
    _process_fused_payload_chunk_worker,  # noqa: F401
    _process_payload_chunk_worker,  # noqa: F401
)
from fiberhmm.inference.tagging import (
    set_legacy_apply_tags,
)
from fiberhmm.inference.worker_results import coerce_worker_chunk_result

_is_main_chromosome = _region_is_main_chromosome

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


def process_bam_for_footprints(input_bam: str, output_bam: str,
                                model_or_path, train_rids: Set[str],
                                edge_trim: int, circular: bool,
                                mode: str, context_size: int,
                                msp_min_size: int,
                                nuc_min_size: int = 85,
                                min_mapq: int = 0,
                                prob_threshold: int = 0,
                                min_read_length: int = 0,
                                with_scores: bool = False,
                                n_cores: int = 1,
                                max_reads: Optional[int] = None,
                                debug_timing: bool = False,
                                region_parallel: bool = False,
                                region_size: int = 10_000_000,
                                skip_scaffolds: bool = False,
                                chroms: Optional[Set[str]] = None,
                                primary_only: bool = False,
                                output_posteriors: Optional[str] = None,
                                write_msps: bool = True,
                                io_threads: int = 4,
                                streaming_pipeline: bool = False,
                                chunk_size: int = 500,
                                process_unmapped: bool = False) -> Tuple[int, int]:
    """
    Process BAM file and add footprint tags - SINGLE PASS STREAMING.

    Reads chunks from BAM, processes in parallel, writes immediately.
    Memory usage is bounded to one chunk at a time.

    Args:
        region_parallel: Use region-based parallelism (each worker reads BAM)
        primary_only: Only process primary alignments (skip secondary/supplementary)
        skip_scaffolds: If True, skip scaffold/contig chromosomes
        chroms: If provided, only process these chromosomes
        output_posteriors: If provided, write HMM posteriors to this H5 file (inline)

    Returns:
        (total_reads_processed, reads_with_footprints)
    """
    # Suppress htslib warnings
    pysam.set_verbosity(0)

    # Get model path for workers
    if isinstance(model_or_path, str):
        model_path = model_or_path
        model = freeze_model_for_inference(load_model(model_path))
    else:
        model = model_or_path
        model_path = None

    # Dispatch to region-parallel if requested (requires model_path for workers)
    if region_parallel:
        if model_path is None:
            print("Warning: region_parallel requires model path, falling back to standard parallel")
        else:
            return _process_bam_region_parallel(
                input_bam=input_bam,
                output_bam=output_bam,
                model_path=model_path,
                train_rids=train_rids,
                edge_trim=edge_trim,
                circular=circular,
                mode=mode,
                context_size=context_size,
                msp_min_size=msp_min_size,
                nuc_min_size=nuc_min_size,
                min_mapq=min_mapq,
                prob_threshold=prob_threshold,
                min_read_length=min_read_length,
                with_scores=with_scores,
                n_cores=n_cores,
                region_size=region_size,
                skip_scaffolds=skip_scaffolds,
                chroms=chroms,
                primary_only=primary_only,
                output_posteriors=output_posteriors,
                write_msps=write_msps,
                io_threads=io_threads
            )

    # Dispatch to streaming pipeline if requested (requires model_path for workers)
    if streaming_pipeline:
        if model_path is None:
            print("Warning: streaming_pipeline requires model path, falling back to standard parallel")
        else:
            return _process_bam_streaming_pipeline(
                input_bam=input_bam,
                output_bam=output_bam,
                model_path=model_path,
                train_rids=train_rids,
                edge_trim=edge_trim,
                circular=circular,
                mode=mode,
                context_size=context_size,
                msp_min_size=msp_min_size,
                nuc_min_size=nuc_min_size,
                min_mapq=min_mapq,
                prob_threshold=prob_threshold,
                min_read_length=min_read_length,
                with_scores=with_scores,
                n_cores=n_cores,
                chunk_size=chunk_size,
                io_threads=io_threads,
                primary_only=primary_only,
                output_posteriors=output_posteriors,
                write_msps=write_msps,
                max_reads=max_reads,
                debug_timing=debug_timing,
                process_unmapped=process_unmapped,
            )

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

                        # Write posteriors if requested
                        if posterior_writer and chunk_results:
                            for read_obj, fiber_read_data, result in chunk_results:
                                if result and result.get('posteriors') is not None:
                                    chrom = read_obj.reference_name
                                    if chrom:
                                        # Get ref_positions from the read object
                                        ref_positions = get_ref_positions_from_read(read_obj) if HAS_POSTERIOR_WRITER else np.array([], dtype=np.int32)
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

                    # Write posteriors if requested
                    if posterior_writer and chunk_results:
                        for read_obj, fiber_read_data, result in chunk_results:
                            if result and result.get('posteriors') is not None:
                                chrom = read_obj.reference_name
                                if chrom:
                                    ref_positions = get_ref_positions_from_read(read_obj) if HAS_POSTERIOR_WRITER else np.array([], dtype=np.int32)
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
