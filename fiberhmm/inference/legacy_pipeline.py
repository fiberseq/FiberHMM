"""Legacy chunked BAM apply pipeline."""

from __future__ import annotations

import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Optional, Set, Tuple

import numpy as np
import pysam

from fiberhmm.inference.bam_output import _sort_and_index_bam
from fiberhmm.inference.engine import (
    _extract_fiber_read_from_pysam,
    _process_single_read,
)
from fiberhmm.inference.mp_context import _MP_CONTEXT
from fiberhmm.inference.posterior_records import posterior_fiber_data
from fiberhmm.inference.read_filters import ReadFilterConfig, streaming_skip_reason
from fiberhmm.inference.skip_reasons import (
    NO_FOOTPRINTS_SKIP_REASON,
    iter_nonzero_skip_reasons,
    new_skip_reasons,
    record_skip_reason,
)
from fiberhmm.inference.streaming_workers import (
    _init_bam_worker,
    _process_chunk_worker,
)
from fiberhmm.inference.tagging import set_legacy_apply_tags
from fiberhmm.inference.worker_results import coerce_worker_chunk_result
from fiberhmm.io.bam_header import append_coord_marker

# Optional: inline posteriors export
try:
    from fiberhmm.posteriors.hdf5_backend import PosteriorWriter, get_ref_positions_from_read
    HAS_POSTERIOR_WRITER = True
except ImportError:
    HAS_POSTERIOR_WRITER = False


_LEGACY_DEFAULT_CHUNK_SIZE = 2000


@dataclass(frozen=True)
class _LegacyChunkResult:
    results: list
    worker_failures: int


@dataclass(frozen=True)
class _LegacyPosteriorStats:
    n_fibers: int
    file_size_mb: float


@dataclass(frozen=True)
class _LegacyChunkRecordResult:
    reads_with_footprints: int
    worker_failures: int


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

    chunk_result = _process_legacy_chunk_results(
        chunk_reads,
        model,
        executor,
        edge_trim,
        circular,
        mode,
        context_size,
        msp_min_size,
        nuc_min_size,
        with_scores,
        return_posteriors,
    )

    # Write annotated reads
    reads_with_footprints, no_footprints = _write_processed_legacy_reads(
        chunk_read_objs,
        chunk_result.results,
        outbam,
        with_scores,
        write_msps,
    )

    # Return results for posteriors if requested
    if return_posteriors:
        return (
            reads_with_footprints,
            no_footprints,
            chunk_result.worker_failures,
            list(zip(chunk_read_objs, chunk_reads, chunk_result.results)),
        )
    return reads_with_footprints, no_footprints, chunk_result.worker_failures, None


def _process_legacy_chunk_results(
    chunk_reads: list,
    model,
    executor,
    edge_trim: int,
    circular: bool,
    mode: str,
    context_size: int,
    msp_min_size: int,
    nuc_min_size: int,
    with_scores: bool,
    return_posteriors: bool,
) -> _LegacyChunkResult:
    if executor is not None:
        # Parallel: submit chunk to worker
        future = executor.submit(
            _process_chunk_worker,
            chunk_reads, edge_trim, circular, mode, context_size, msp_min_size,
            nuc_min_size, with_scores, return_posteriors
        )
        worker_result = coerce_worker_chunk_result(future.result())
        results = worker_result.results
        worker_failures = worker_result.read_failures
    else:
        # Single-threaded: process directly
        results = _process_direct_legacy_chunk_results(
            chunk_reads,
            model,
            edge_trim,
            circular,
            mode,
            context_size,
            msp_min_size,
            nuc_min_size,
            with_scores,
            return_posteriors,
        )
        worker_failures = 0

    return _LegacyChunkResult(results, worker_failures)


def _process_direct_legacy_chunk_results(
    chunk_reads: list,
    model,
    edge_trim: int,
    circular: bool,
    mode: str,
    context_size: int,
    msp_min_size: int,
    nuc_min_size: int,
    with_scores: bool,
    return_posteriors: bool,
) -> list:
    results = []
    for fiber_read in chunk_reads:
        result = _process_single_read(
            fiber_read, model, edge_trim, circular,
            mode, context_size, msp_min_size, nuc_min_size=nuc_min_size,
            with_scores=with_scores,
            return_posteriors=return_posteriors,
        )
        results.append(result)
    return results


def _write_processed_legacy_reads(
    chunk_read_objs: list,
    results: list,
    outbam,
    with_scores: bool,
    write_msps: bool,
) -> Tuple[int, int]:
    reads_with_footprints = 0
    no_footprints = 0
    for read_obj, result in zip(chunk_read_objs, results):
        if result is not None:
            set_legacy_apply_tags(read_obj, result, with_scores, write_msps)
            reads_with_footprints += 1
        else:
            no_footprints += 1

        outbam.write(read_obj)

    return reads_with_footprints, no_footprints


def _legacy_posterior_ref_positions(read_obj):
    if HAS_POSTERIOR_WRITER:
        return get_ref_positions_from_read(read_obj)
    return np.array([], dtype=np.int32)


def _write_chunk_posteriors(posterior_writer, chunk_results):
    if posterior_writer is None or not chunk_results:
        return

    for read_obj, fiber_read_data, result in chunk_results:
        if result is not None and result.get('posteriors') is not None:
            chrom = read_obj.reference_name
            if chrom:
                posterior_writer.add_fiber(
                    chrom,
                    posterior_fiber_data(
                        read_obj,
                        result,
                        _legacy_posterior_ref_positions(read_obj),
                    ),
                )


def _process_legacy_chunk_and_record(
    chunk_reads: list,
    chunk_read_objs: list,
    outbam,
    model,
    executor,
    edge_trim: int,
    circular: bool,
    mode: str,
    context_size: int,
    msp_min_size: int,
    skip_reasons: dict,
    posterior_writer,
    nuc_min_size: int = 85,
    with_scores: bool = False,
    return_posteriors: bool = False,
    write_msps: bool = True,
) -> _LegacyChunkRecordResult:
    n_fp, n_nofp, n_failed, chunk_results = _process_and_write_chunk(
        chunk_reads,
        chunk_read_objs,
        outbam,
        model,
        executor,
        edge_trim,
        circular,
        mode,
        context_size,
        msp_min_size,
        nuc_min_size=nuc_min_size,
        with_scores=with_scores,
        return_posteriors=return_posteriors,
        write_msps=write_msps,
    )
    skip_reasons[NO_FOOTPRINTS_SKIP_REASON] += n_nofp
    _write_chunk_posteriors(posterior_writer, chunk_results)
    return _LegacyChunkRecordResult(
        reads_with_footprints=n_fp,
        worker_failures=n_failed,
    )


def _open_legacy_posterior_writer(
    output_posteriors: Optional[str],
    mode: str,
    context_size: int,
    edge_trim: int,
    input_bam: str,
) -> tuple[object | None, bool]:
    if not output_posteriors:
        return None, False

    if not HAS_POSTERIOR_WRITER:
        print("WARNING: posterior_writer.py not found, skipping posteriors export")
        return None, False

    writer = PosteriorWriter(
        output_posteriors, mode, context_size,
        edge_trim, input_bam, batch_size=1000
    )
    print(f"Posteriors will be written to: {output_posteriors}")
    return writer, True


def _new_legacy_executor(model_path: str, n_cores: int, debug_timing: bool):
    return ProcessPoolExecutor(
        max_workers=n_cores,
        mp_context=_MP_CONTEXT,
        initializer=_init_bam_worker,
        initargs=(model_path, debug_timing)
    )


def _legacy_executor_for_config(
    model_path: Optional[str],
    n_cores: int,
    debug_timing: bool,
):
    if n_cores > 1 and model_path:
        return _new_legacy_executor(model_path, n_cores, debug_timing)
    return None


def _shutdown_legacy_resources(executor, posterior_writer):
    posterior_stats = None
    try:
        if executor is not None:
            executor.shutdown(wait=True)
    finally:
        if posterior_writer is not None:
            closed_stats = posterior_writer.close()
            if closed_stats is not None:
                n_fibers, file_size_mb = closed_stats
                posterior_stats = _LegacyPosteriorStats(
                    n_fibers=n_fibers,
                    file_size_mb=file_size_mb,
                )
    return posterior_stats


def _legacy_processing_rate(total_reads: int, elapsed: float) -> float:
    return total_reads / elapsed if elapsed > 0 else 0


def _legacy_progress_message(total_reads: int, skipped: int, rate: float) -> str:
    return (
        f"\r  Processed: {total_reads:,} | Skipped: {skipped:,} | "
        f"{rate:.1f} reads/s"
    )


def _print_legacy_progress(total_reads: int, skipped: int, start_time: float) -> None:
    elapsed = time.time() - start_time
    print(
        _legacy_progress_message(
            total_reads,
            skipped,
            _legacy_processing_rate(total_reads, elapsed),
        ),
        end='',
    )
    sys.stdout.flush()


def _legacy_completion_message(
    total_reads: int,
    skipped: int,
    reads_with_footprints: int,
    rate: float,
) -> str:
    return (
        f"\r  Processed: {total_reads:,} | Skipped: {skipped:,} | "
        f"With footprints: {reads_with_footprints:,} | {rate:.1f} reads/s"
    )


def _print_legacy_skip_summary(
    skip_reasons: dict,
    total_reads: int,
    skipped: int,
) -> None:
    if skipped <= 0:
        return

    print("  Skip reasons:")
    for reason, count in iter_nonzero_skip_reasons(skip_reasons):
        pct = 100 * count / (total_reads + skipped)
        print(f"    {reason}: {count:,} ({pct:.1f}%)")


def _print_legacy_completion_summary(
    total_reads: int,
    skipped: int,
    reads_with_footprints: int,
    worker_failures: int,
    skip_reasons: dict,
    elapsed: float,
) -> None:
    print(
        _legacy_completion_message(
            total_reads,
            skipped,
            reads_with_footprints,
            _legacy_processing_rate(total_reads, elapsed),
        )
    )
    if worker_failures:
        print(f"  Worker read failures: {worker_failures:,} (passed through unchanged)")
    _print_legacy_skip_summary(skip_reasons, total_reads, skipped)


def _print_legacy_posterior_summary(
    posterior_stats,
    output_posteriors: Optional[str],
) -> None:
    if not posterior_stats:
        return
    print(
        f"Posteriors: {posterior_stats.n_fibers:,} fibers -> "
        f"{output_posteriors} ({posterior_stats.file_size_mb:.1f} MB)"
    )


def _process_legacy_chunk_buffer(
    chunk_reads: list,
    chunk_read_objs: list,
    outbam,
    model,
    executor,
    edge_trim: int,
    circular: bool,
    mode: str,
    context_size: int,
    msp_min_size: int,
    skip_reasons: dict,
    posterior_writer,
    nuc_min_size: int,
    with_scores: bool,
    return_posteriors: bool,
    write_msps: bool,
) -> _LegacyChunkRecordResult:
    if not chunk_reads:
        return _LegacyChunkRecordResult(0, 0)
    return _process_legacy_chunk_and_record(
        chunk_reads,
        chunk_read_objs,
        outbam,
        model,
        executor,
        edge_trim,
        circular,
        mode,
        context_size,
        msp_min_size,
        skip_reasons,
        posterior_writer,
        nuc_min_size=nuc_min_size,
        with_scores=with_scores,
        return_posteriors=return_posteriors,
        write_msps=write_msps,
    )


def _legacy_chunk_buffer_kwargs(
    outbam,
    model,
    executor,
    edge_trim: int,
    circular: bool,
    mode: str,
    context_size: int,
    msp_min_size: int,
    skip_reasons: dict,
    posterior_writer,
    nuc_min_size: int,
    with_scores: bool,
    return_posteriors: bool,
    write_msps: bool,
) -> dict:
    return {
        'outbam': outbam,
        'model': model,
        'executor': executor,
        'edge_trim': edge_trim,
        'circular': circular,
        'mode': mode,
        'context_size': context_size,
        'msp_min_size': msp_min_size,
        'skip_reasons': skip_reasons,
        'posterior_writer': posterior_writer,
        'nuc_min_size': nuc_min_size,
        'with_scores': with_scores,
        'return_posteriors': return_posteriors,
        'write_msps': write_msps,
    }


@dataclass(frozen=True)
class _LegacyReadProcessingResult:
    total_reads: int
    reads_with_footprints: int
    skipped: int
    worker_failures: int


def _process_legacy_reads(
    reads,
    outbam,
    model,
    executor,
    filter_config: ReadFilterConfig,
    mode: str,
    prob_threshold: int,
    edge_trim: int,
    circular: bool,
    context_size: int,
    msp_min_size: int,
    skip_reasons: dict,
    posterior_writer,
    start_time: float,
    max_reads: Optional[int],
    chunk_size: int,
    nuc_min_size: int = 85,
    with_scores: bool = False,
    return_posteriors: bool = False,
    write_msps: bool = True,
) -> _LegacyReadProcessingResult:
    total_reads = 0
    reads_with_footprints = 0
    skipped = 0
    worker_failures = 0
    chunk_reads = []
    chunk_read_objs = []
    chunk_kwargs = _legacy_chunk_buffer_kwargs(
        outbam, model, executor, edge_trim, circular, mode, context_size,
        msp_min_size, skip_reasons, posterior_writer, nuc_min_size,
        with_scores, return_posteriors, write_msps,
    )

    for read in reads:
        fiber_read, skip_reason = _legacy_fiber_read_or_skip(
            read, filter_config, mode, prob_threshold,
        )
        if skip_reason:
            skipped += _write_skipped_legacy_read(
                outbam, read, skip_reasons, skip_reason
            )
            continue

        chunk_reads.append(fiber_read)
        chunk_read_objs.append(read)
        total_reads += 1

        if max_reads and total_reads >= max_reads:
            break

        if len(chunk_reads) >= chunk_size:
            chunk_result = _process_legacy_chunk_buffer(
                chunk_reads,
                chunk_read_objs,
                **chunk_kwargs,
            )
            reads_with_footprints += chunk_result.reads_with_footprints
            worker_failures += chunk_result.worker_failures

            _print_legacy_progress(total_reads, skipped, start_time)

            chunk_reads = []
            chunk_read_objs = []

    chunk_result = _process_legacy_chunk_buffer(
        chunk_reads,
        chunk_read_objs,
        **chunk_kwargs,
    )
    reads_with_footprints += chunk_result.reads_with_footprints
    worker_failures += chunk_result.worker_failures

    return _LegacyReadProcessingResult(
        total_reads=total_reads,
        reads_with_footprints=reads_with_footprints,
        skipped=skipped,
        worker_failures=worker_failures,
    )


@dataclass(frozen=True)
class _LegacyPipelineResult:
    total_reads: int
    reads_with_footprints: int
    skipped: int
    worker_failures: int
    posterior_stats: object | None


def _new_legacy_skip_reasons() -> dict:
    return new_skip_reasons(NO_FOOTPRINTS_SKIP_REASON)


def _legacy_filter_config(
    *,
    min_mapq: int,
    min_read_length: int,
    primary_only: bool,
    train_rids: Set[str],
) -> ReadFilterConfig:
    return ReadFilterConfig(
        min_mapq=min_mapq,
        min_read_length=min_read_length,
        primary_only=primary_only,
        process_unmapped=False,
        train_rids=train_rids,
    )


def _write_skipped_legacy_read(outbam, read, skip_reasons: dict, reason: str) -> int:
    outbam.write(read)
    record_skip_reason(skip_reasons, reason)
    return 1


def _legacy_fiber_read_or_skip(read, filter_config: ReadFilterConfig,
                               mode: str, prob_threshold: int):
    skip_reason = streaming_skip_reason(read, filter_config)
    if skip_reason:
        return None, skip_reason

    try:
        fiber_read = _extract_fiber_read_from_pysam(read, mode, prob_threshold)
    except Exception:
        return None, 'extraction_failed'

    if fiber_read is None:
        return None, 'no_modifications'
    return fiber_read, None


def _run_legacy_bam_processing(
    input_bam: str,
    output_bam: str,
    model,
    model_path: Optional[str],
    filter_config: ReadFilterConfig,
    skip_reasons: dict,
    edge_trim: int,
    circular: bool,
    mode: str,
    context_size: int,
    msp_min_size: int,
    nuc_min_size: int,
    prob_threshold: int,
    with_scores: bool,
    n_cores: int,
    max_reads: Optional[int],
    debug_timing: bool,
    output_posteriors: Optional[str],
    write_msps: bool,
    io_threads: int,
    start_time: float,
    chunk_size: int,
) -> _LegacyPipelineResult:
    executor = None
    posterior_writer = None
    posterior_stats = None

    with pysam.AlignmentFile(input_bam, "rb", threads=io_threads, check_sq=False) as inbam:
        with pysam.AlignmentFile(output_bam, "wb",
                                 header=append_coord_marker(inbam.header),
                                 threads=io_threads) as outbam:
            try:
                posterior_writer, return_posteriors = _open_legacy_posterior_writer(
                    output_posteriors, mode, context_size, edge_trim, input_bam,
                )
                executor = _legacy_executor_for_config(
                    model_path, n_cores, debug_timing,
                )

                read_result = _process_legacy_reads(
                    inbam,
                    outbam,
                    model,
                    executor,
                    filter_config,
                    mode,
                    prob_threshold,
                    edge_trim,
                    circular,
                    context_size,
                    msp_min_size,
                    skip_reasons,
                    posterior_writer,
                    start_time,
                    max_reads,
                    chunk_size,
                    nuc_min_size=nuc_min_size,
                    with_scores=with_scores,
                    return_posteriors=return_posteriors,
                    write_msps=write_msps,
                )
            finally:
                posterior_stats = _shutdown_legacy_resources(
                    executor, posterior_writer,
                )

    return _LegacyPipelineResult(
        total_reads=read_result.total_reads,
        reads_with_footprints=read_result.reads_with_footprints,
        skipped=read_result.skipped,
        worker_failures=read_result.worker_failures,
        posterior_stats=posterior_stats,
    )


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
    skip_reasons = _new_legacy_skip_reasons()
    filter_config = _legacy_filter_config(
        min_mapq=min_mapq,
        min_read_length=min_read_length,
        primary_only=primary_only,
        train_rids=train_rids,
    )

    chunk_size = _LEGACY_DEFAULT_CHUNK_SIZE
    start_time = time.time()

    print("Processing and writing BAM (streaming)...")
    sys.stdout.flush()

    result = _run_legacy_bam_processing(
        input_bam=input_bam,
        output_bam=output_bam,
        model=model,
        model_path=model_path,
        filter_config=filter_config,
        skip_reasons=skip_reasons,
        edge_trim=edge_trim,
        circular=circular,
        mode=mode,
        context_size=context_size,
        msp_min_size=msp_min_size,
        nuc_min_size=nuc_min_size,
        prob_threshold=prob_threshold,
        with_scores=with_scores,
        n_cores=n_cores,
        max_reads=max_reads,
        debug_timing=debug_timing,
        output_posteriors=output_posteriors,
        write_msps=write_msps,
        io_threads=io_threads,
        start_time=start_time,
        chunk_size=chunk_size,
    )

    _print_legacy_completion_summary(
        result.total_reads,
        result.skipped,
        result.reads_with_footprints,
        result.worker_failures,
        skip_reasons,
        time.time() - start_time,
    )

    # Index the output BAM (sort first if needed)
    _sort_and_index_bam(output_bam, threads=n_cores)

    _print_legacy_posterior_summary(result.posterior_stats, output_posteriors)

    return result.total_reads, result.reads_with_footprints
