"""Streaming BAM pipeline coordinators for apply and fused apply+recall."""

from __future__ import annotations

import os
import sys
import time
from collections import deque
from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import dataclass
from typing import Callable, Optional, Set, Tuple

import pysam

from fiberhmm.inference.bam_output import _sort_and_index_bam
from fiberhmm.inference.engine import make_apply_payload
from fiberhmm.io.bam_header import append_coord_marker, maybe_append_pg
from fiberhmm.inference.mp_context import _MP_CONTEXT
from fiberhmm.inference.read_filters import ReadFilterConfig, streaming_skip_reason
from fiberhmm.inference.skip_reasons import (
    NO_FOOTPRINTS_SKIP_REASON,
    new_skip_reasons,
)
from fiberhmm.inference.streaming_drain import (
    _drain_oldest_chunk,
    _drain_oldest_fused_chunk,
)
from fiberhmm.inference.streaming_workers import (
    _init_bam_worker,
    _init_fused_worker,
    _process_fused_payload_chunk_worker,
    _process_payload_chunk_worker,
)

try:
    from fiberhmm.posteriors.hdf5_backend import PosteriorWriter
    HAS_POSTERIOR_WRITER = True
except ImportError:
    HAS_POSTERIOR_WRITER = False


def _buffer_skipped_read(chunk_read_objs, chunk_skip_flags, skip_reasons, read, reason) -> int:
    chunk_read_objs.append(read)
    chunk_skip_flags.append(True)
    skip_reasons[reason] += 1
    return 1


def _buffer_processable_read(chunk_items, chunk_read_objs, chunk_skip_flags, payload, read) -> int:
    chunk_items.append(payload)
    chunk_read_objs.append(read)
    chunk_skip_flags.append(False)
    return 1


def _streaming_payload_or_skip(read, filter_config: ReadFilterConfig,
                               mode: str, ref_fasta=None):
    skip_reason = streaming_skip_reason(read, filter_config)
    if skip_reason:
        return None, skip_reason

    payload = make_apply_payload(read, mode=mode, ref_fasta=ref_fasta)
    if payload is None:
        return None, 'no_modifications'

    return payload, None


def _buffer_streaming_read(
    read,
    filter_config: ReadFilterConfig,
    mode: str,
    ref_fasta,
    chunk_items,
    chunk_read_objs,
    chunk_skip_flags,
    skip_reasons,
) -> tuple[int, int]:
    payload, skip_reason = _streaming_payload_or_skip(
        read, filter_config, mode, ref_fasta,
    )
    if skip_reason:
        return 0, _buffer_skipped_read(
            chunk_read_objs, chunk_skip_flags, skip_reasons, read, skip_reason,
        )

    return _buffer_processable_read(
        chunk_items, chunk_read_objs, chunk_skip_flags, payload, read,
    ), 0


def _completed_empty_future() -> Future:
    future = Future()
    future.set_result([])
    return future


def _new_streaming_chunk_buffers() -> tuple[list, list, list]:
    return [], [], []


def _submit_streaming_chunk(inflight, executor, worker_fn, chunk_items,
                            chunk_read_objs, chunk_skip_flags, worker_args) -> None:
    if chunk_items:
        future = executor.submit(worker_fn, chunk_items, *worker_args)
    else:
        future = _completed_empty_future()
    inflight.append((future, chunk_read_objs, chunk_items, chunk_skip_flags))


def _drain_if_inflight_full(
    inflight,
    max_inflight: int,
    drain_chunk: Callable[[], None],
) -> None:
    if len(inflight) >= max_inflight:
        drain_chunk()


def _flush_streaming_chunk(
    inflight,
    executor,
    worker_fn,
    chunk_items,
    chunk_read_objs,
    chunk_skip_flags,
    worker_args,
    max_inflight: int,
    drain_chunk: Callable[[], None],
) -> tuple[list, list, list]:
    _drain_if_inflight_full(inflight, max_inflight, drain_chunk)
    _submit_streaming_chunk(
        inflight,
        executor,
        worker_fn,
        chunk_items,
        chunk_read_objs,
        chunk_skip_flags,
        worker_args,
    )
    return _new_streaming_chunk_buffers()


def _drain_all_streaming_chunks(inflight, drain_chunk: Callable[[], None]) -> None:
    while inflight:
        drain_chunk()


@dataclass(frozen=True)
class _StreamingFlushContext:
    inflight: object
    executor: object
    worker_fn: object
    worker_args: tuple
    max_inflight: int
    drain_chunk: Callable[[], None]
    log: object
    progress_label: str
    start_time: float
    rate_unit: str


def _flush_streaming_chunk_and_report_progress(
    context: _StreamingFlushContext,
    chunk_items,
    chunk_read_objs,
    chunk_skip_flags,
    total_reads: int,
    skipped: int,
    last_progress_reads: int,
    last_progress_time: float,
) -> tuple[list, list, list, int, float]:
    chunk_items, chunk_read_objs, chunk_skip_flags = _flush_streaming_chunk(
        context.inflight,
        context.executor,
        context.worker_fn,
        chunk_items,
        chunk_read_objs,
        chunk_skip_flags,
        context.worker_args,
        context.max_inflight,
        context.drain_chunk,
    )
    now = time.time()
    last_progress_reads, last_progress_time = _print_streaming_progress(
        context.log,
        context.progress_label,
        total_reads,
        skipped,
        len(context.inflight),
        last_progress_reads,
        context.start_time,
        last_progress_time,
        now,
        context.rate_unit,
    )
    return (
        chunk_items,
        chunk_read_objs,
        chunk_skip_flags,
        last_progress_reads,
        last_progress_time,
    )


def _stream_reads_to_workers(
    reads,
    filter_config: ReadFilterConfig,
    mode: str,
    ref_fasta,
    inflight,
    executor,
    worker_fn,
    worker_args,
    max_reads: Optional[int],
    chunk_size: int,
    max_inflight: int,
    drain_chunk: Callable[[], None],
    skip_reasons: dict,
    log,
    progress_label: str,
    rate_unit: str,
    start_time: float,
) -> tuple[int, int]:
    total_reads = 0
    skipped = 0
    chunk_items, chunk_read_objs, chunk_skip_flags = _new_streaming_chunk_buffers()
    last_progress_reads = 0
    last_progress_time = time.time()
    flush_context = _StreamingFlushContext(
        inflight=inflight,
        executor=executor,
        worker_fn=worker_fn,
        worker_args=worker_args,
        max_inflight=max_inflight,
        drain_chunk=drain_chunk,
        log=log,
        progress_label=progress_label,
        start_time=start_time,
        rate_unit=rate_unit,
    )

    for read in reads:
        processed_delta, skipped_delta = _buffer_streaming_read(
            read,
            filter_config,
            mode,
            ref_fasta,
            chunk_items,
            chunk_read_objs,
            chunk_skip_flags,
            skip_reasons,
        )
        skipped += skipped_delta
        if skipped_delta:
            continue

        total_reads += processed_delta

        if max_reads and total_reads >= max_reads:
            break

        if len(chunk_read_objs) >= chunk_size:
            (
                chunk_items,
                chunk_read_objs,
                chunk_skip_flags,
                last_progress_reads,
                last_progress_time,
            ) = _flush_streaming_chunk_and_report_progress(
                flush_context,
                chunk_items,
                chunk_read_objs,
                chunk_skip_flags,
                total_reads,
                skipped,
                last_progress_reads,
                last_progress_time,
            )

    if chunk_read_objs:
        _flush_streaming_chunk(
            inflight,
            executor,
            worker_fn,
            chunk_items,
            chunk_read_objs,
            chunk_skip_flags,
            worker_args,
            max_inflight,
            drain_chunk,
        )

    _drain_all_streaming_chunks(inflight, drain_chunk)
    return total_reads, skipped


def _run_streaming_worker_loop(
    reads,
    filter_config: ReadFilterConfig,
    mode: str,
    ref_fasta,
    executor,
    worker_fn,
    worker_args,
    max_reads: Optional[int],
    chunk_size: int,
    max_inflight: int,
    drain_chunk_factory,
    skip_reasons: dict,
    log,
    progress_label: str,
    rate_unit: str,
    start_time: float,
) -> tuple[int, int]:
    inflight = deque()
    drain_chunk = drain_chunk_factory(inflight)
    try:
        return _stream_reads_to_workers(
            reads,
            filter_config,
            mode,
            ref_fasta,
            inflight,
            executor,
            worker_fn,
            worker_args,
            max_reads,
            chunk_size,
            max_inflight,
            drain_chunk,
            skip_reasons,
            log,
            progress_label,
            rate_unit,
            start_time,
        )
    finally:
        executor.shutdown(wait=True)


def _worker_common_args(
    edge_trim: int,
    circular: bool,
    mode: str,
    context_size: int,
    msp_min_size: int,
    nuc_min_size: int,
    with_scores: bool,
) -> tuple:
    return (
        edge_trim,
        circular,
        mode,
        context_size,
        msp_min_size,
        nuc_min_size,
        with_scores,
    )


def _apply_worker_args(
    edge_trim: int,
    circular: bool,
    mode: str,
    context_size: int,
    msp_min_size: int,
    nuc_min_size: int,
    with_scores: bool,
    return_posteriors: bool,
    prob_threshold: int,
) -> tuple:
    return (
        *_worker_common_args(
            edge_trim, circular, mode, context_size,
            msp_min_size, nuc_min_size, with_scores,
        ),
        return_posteriors,
        prob_threshold,
    )


def _fused_worker_args(
    edge_trim: int,
    circular: bool,
    mode: str,
    context_size: int,
    msp_min_size: int,
    nuc_min_size: int,
    with_scores: bool,
    prob_threshold: int,
    min_llr: float,
    min_opps: int,
    unify_threshold: int,
) -> tuple:
    return (
        *_worker_common_args(
            edge_trim, circular, mode, context_size,
            msp_min_size, nuc_min_size, with_scores,
        ),
        prob_threshold,
        mode,
        context_size,
        min_llr,
        min_opps,
        unify_threshold,
    )


def _new_streaming_counters() -> dict:
    return {
        'reads_with_footprints': 0,
        'no_footprints': 0,
        'worker_failures': 0,
        'written': 0,
        'chimera': 0,
    }


def _new_streaming_skip_reasons(*, include_no_footprints: bool = False) -> dict:
    extras = (NO_FOOTPRINTS_SKIP_REASON,) if include_no_footprints else ()
    return new_skip_reasons(*extras)


def _streaming_filter_config(
    min_mapq: int,
    min_read_length: int,
    primary_only: bool,
    process_unmapped: bool,
    train_rids,
) -> ReadFilterConfig:
    return ReadFilterConfig(
        min_mapq=min_mapq,
        min_read_length=min_read_length,
        primary_only=primary_only,
        process_unmapped=process_unmapped,
        train_rids=train_rids,
    )


def _print_worker_failure_summary(counters: dict, log) -> None:
    if counters['worker_failures']:
        print(
            f"  Worker read failures: {counters['worker_failures']:,} "
            f"(passed through unchanged)",
            file=log,
        )


def _streaming_log_for_output(output_bam: str):
    return sys.stderr if output_bam == '-' else sys.stdout


def _streaming_output_target(output_bam: str):
    if output_bam == '-':
        return os.fdopen(1, 'wb', closefd=False)
    return output_bam


def _open_streaming_posterior_writer(
    output_posteriors: Optional[str],
    mode: str,
    context_size: int,
    edge_trim: int,
    input_bam: str,
    log,
) -> tuple[object | None, bool]:
    if not output_posteriors:
        return None, False

    if not HAS_POSTERIOR_WRITER:
        print(
            "WARNING: posterior_writer.py not found, skipping posteriors export",
            file=log,
        )
        return None, False

    writer = PosteriorWriter(
        output_posteriors, mode, context_size,
        edge_trim, input_bam, batch_size=1000,
    )
    print(f"Posteriors will be written to: {output_posteriors}", file=log)
    return writer, True


def _new_apply_streaming_executor(
    model_path: str,
    n_cores: int,
    debug_timing: bool,
):
    return ProcessPoolExecutor(
        max_workers=n_cores,
        mp_context=_MP_CONTEXT,
        initializer=_init_bam_worker,
        initargs=(model_path, debug_timing),
    )


def _new_fused_streaming_executor(
    model_path: str,
    recall_model_path: str,
    emission_uplift: float,
    recall_nucs: bool,
    split_min_llr: float,
    split_min_opps: int,
    filter_chimeras: bool,
    chimera_min_seg: int,
    chimera_purity: float,
    phase_nrl: int,
    n_cores: int,
):
    return ProcessPoolExecutor(
        max_workers=n_cores,
        mp_context=_MP_CONTEXT,
        initializer=_init_fused_worker,
        initargs=(
            model_path,
            recall_model_path,
            emission_uplift,
            False,
            recall_nucs,
            split_min_llr,
            split_min_opps,
            filter_chimeras,
            chimera_min_seg,
            chimera_purity,
            phase_nrl,
        ),
    )


def _print_streaming_skip_summary(skip_reasons: dict, total_reads: int,
                                  skipped: int, log) -> None:
    if skipped <= 0:
        return
    print("  Skip reasons:", file=log)
    for reason, count in sorted(skip_reasons.items(), key=lambda x: -x[1]):
        if count > 0:
            pct = 100 * count / (total_reads + skipped)
            print(f"    {reason}: {count:,} ({pct:.1f}%)", file=log)


def _streaming_progress_rates(
    total_reads: int,
    last_progress_reads: int,
    start_time: float,
    last_progress_time: float,
    now: float,
) -> tuple[float, float]:
    elapsed = now - start_time
    avg_rate = total_reads / elapsed if elapsed > 0 else 0
    dt = now - last_progress_time
    inst_rate = (total_reads - last_progress_reads) / dt if dt > 0 else 0
    return inst_rate, avg_rate


def _streaming_progress_message(
    label: str,
    total_reads: int,
    skipped: int,
    inflight_count: int,
    inst_rate: float,
    avg_rate: float,
    rate_unit: str,
) -> str:
    return (
        f"\r  {label}: {total_reads:,} | Skipped: {skipped:,} | "
        f"Inflight: {inflight_count} | "
        f"{inst_rate:.0f} {rate_unit} (avg {avg_rate:.0f})"
    )


def _print_streaming_progress(
    log,
    label: str,
    total_reads: int,
    skipped: int,
    inflight_count: int,
    last_progress_reads: int,
    start_time: float,
    last_progress_time: float,
    now: float,
    rate_unit: str,
) -> tuple[int, float]:
    inst_rate, avg_rate = _streaming_progress_rates(
        total_reads, last_progress_reads, start_time, last_progress_time, now,
    )
    print(
        _streaming_progress_message(
            label,
            total_reads,
            skipped,
            inflight_count,
            inst_rate,
            avg_rate,
            rate_unit,
        ),
        end='',
        file=log,
    )
    log.flush()
    return total_reads, now


def _streaming_rate(total_reads: int, elapsed: float) -> float:
    return total_reads / elapsed if elapsed > 0 else 0


def _streaming_completion_message(
    label: str,
    total_reads: int,
    skipped: int,
    reads_with_footprints: int,
    rate: float,
    rate_unit: str,
) -> str:
    return (
        f"\r  {label}: {total_reads:,} | Skipped: {skipped:,} | "
        f"With footprints: {reads_with_footprints:,} | "
        f"{rate:.1f} {rate_unit}"
    )


def _print_streaming_completion_summary(
    label: str,
    total_reads: int,
    skipped: int,
    counters: dict,
    elapsed: float,
    rate_unit: str,
    skip_reasons: dict,
    log,
) -> int:
    reads_with_footprints = counters['reads_with_footprints']
    print(
        _streaming_completion_message(
            label,
            total_reads,
            skipped,
            reads_with_footprints,
            _streaming_rate(total_reads, elapsed),
            rate_unit,
        ),
        file=log,
    )
    _print_worker_failure_summary(counters, log)
    _print_streaming_skip_summary(skip_reasons, total_reads, skipped, log)
    return reads_with_footprints


def _print_streaming_posterior_summary(
    posterior_stats,
    output_posteriors: Optional[str],
    log,
) -> None:
    if not posterior_stats:
        return
    n_fibers, file_size = posterior_stats
    print(
        f"Posteriors: {n_fibers:,} fibers -> "
        f"{output_posteriors} ({file_size:.1f} MB)",
        file=log,
    )


def _finalize_apply_streaming_pipeline(
    *,
    total_reads: int,
    skipped: int,
    counters: dict,
    start_time: float,
    skip_reasons: dict,
    output_bam: str,
    process_unmapped: bool,
    n_cores: int,
    posterior_stats,
    output_posteriors: Optional[str],
    log,
) -> int:
    reads_with_footprints = _print_streaming_completion_summary(
        "Processed",
        total_reads,
        skipped,
        counters,
        time.time() - start_time,
        "reads/s",
        skip_reasons,
        log,
    )

    if output_bam != '-' and not process_unmapped:
        _sort_and_index_bam(output_bam, threads=n_cores)

    _print_streaming_posterior_summary(posterior_stats, output_posteriors, log)
    return reads_with_footprints


def _print_fused_chimera_summary(counters: dict, log) -> None:
    if counters.get('chimera'):
        print(
            f"  DAF strand-swap chimeras filtered: {counters['chimera']:,}",
            file=log,
        )


def _finalize_fused_streaming_pipeline(
    *,
    total_reads: int,
    skipped: int,
    counters: dict,
    start_time: float,
    skip_reasons: dict,
    log,
) -> int:
    reads_with_fp = _print_streaming_completion_summary(
        "Fused",
        total_reads,
        skipped,
        counters,
        time.time() - start_time,
        "r/s",
        skip_reasons,
        log,
    )
    _print_fused_chimera_summary(counters, log)
    return reads_with_fp


def _process_bam_streaming_pipeline_fused(
    input_bam: str, output_bam: str,
    model_path: str, recall_model_path: str,
    train_rids,
    edge_trim: int, circular: bool,
    mode: str, context_size: int,
    msp_min_size: int, nuc_min_size: int,
    min_mapq: int, prob_threshold: int, min_read_length: int,
    with_scores: bool,
    min_llr: float, min_opps: int, unify_threshold: int,
    emission_uplift: float,
    also_write_legacy: bool, downstream_compat: bool,
    max_reads: int, n_cores: int, chunk_size: int,
    io_threads: int,
    process_unmapped: bool = False,
    primary_only: bool = False,
    ref_fasta_path: Optional[str] = None,
    recall_nucs: bool = False,
    split_min_llr: float = 4.0,
    split_min_opps: int = 3,
    filter_chimeras: bool = True,
    chimera_min_seg: int = 5,
    chimera_purity: float = 0.8,
    phase_nrl: int = 0,
    pg_record: dict = None,
):
    """Fused apply+recall streaming pipeline."""
    ref_fasta = None
    pysam.set_verbosity(0)
    max_inflight = n_cores + 2
    start_time = time.time()
    counters = _new_streaming_counters()

    total_reads = 0
    skipped = 0
    skip_reasons = _new_streaming_skip_reasons()
    filter_config = _streaming_filter_config(
        min_mapq=min_mapq,
        min_read_length=min_read_length,
        primary_only=primary_only,
        process_unmapped=process_unmapped,
        train_rids=train_rids,
    )

    _log = _streaming_log_for_output(output_bam)

    _output_target = _streaming_output_target(output_bam)

    with pysam.AlignmentFile(input_bam, "rb", threads=io_threads, check_sq=False) as inbam:
        with pysam.AlignmentFile(_output_target, "wb",
                                 header=maybe_append_pg(inbam.header, pg_record),
                                 threads=io_threads) as outbam:
            if ref_fasta_path:
                ref_fasta = pysam.FastaFile(ref_fasta_path)

            executor = _new_fused_streaming_executor(
                model_path,
                recall_model_path,
                emission_uplift,
                recall_nucs,
                split_min_llr,
                split_min_opps,
                filter_chimeras,
                chimera_min_seg,
                chimera_purity,
                phase_nrl,
                n_cores,
            )

            worker_args = _fused_worker_args(
                edge_trim, circular, mode, context_size,
                msp_min_size, nuc_min_size, with_scores,
                prob_threshold, min_llr, min_opps, unify_threshold,
            )

            try:
                total_reads, skipped = _run_streaming_worker_loop(
                    inbam.fetch(until_eof=True),
                    filter_config,
                    mode,
                    ref_fasta,
                    executor,
                    _process_fused_payload_chunk_worker,
                    worker_args,
                    max_reads,
                    chunk_size,
                    max_inflight,
                    lambda inflight: lambda: _drain_oldest_fused_chunk(
                        inflight, outbam, with_scores,
                        also_write_legacy, downstream_compat, counters,
                    ),
                    skip_reasons,
                    _log,
                    "Fused",
                    "r/s",
                    start_time,
                )
            finally:
                if ref_fasta is not None:
                    ref_fasta.close()
                    ref_fasta = None

    reads_with_fp = _finalize_fused_streaming_pipeline(
        total_reads=total_reads,
        skipped=skipped,
        counters=counters,
        start_time=start_time,
        skip_reasons=skip_reasons,
        log=_log,
    )

    return total_reads, reads_with_fp


def _process_bam_streaming_pipeline(
    input_bam: str, output_bam: str,
    model_path: str, train_rids: Set[str],
    edge_trim: int, circular: bool,
    mode: str, context_size: int,
    msp_min_size: int,
    nuc_min_size: int = 85,
    min_mapq: int = 0,
    prob_threshold: int = 0,
    min_read_length: int = 0,
    with_scores: bool = False,
    n_cores: int = 4,
    chunk_size: int = 500,
    max_inflight: Optional[int] = None,
    io_threads: int = 4,
    primary_only: bool = False,
    output_posteriors: Optional[str] = None,
    write_msps: bool = True,
    max_reads: Optional[int] = None,
    debug_timing: bool = False,
    process_unmapped: bool = False,
) -> Tuple[int, int]:
    """Streaming producer-consumer pipeline for BAM processing."""
    if max_inflight is None:
        max_inflight = 2 * n_cores

    pysam.set_verbosity(0)

    ref_fasta = None

    total_reads = 0
    skip_reasons = _new_streaming_skip_reasons(include_no_footprints=True)
    filter_config = _streaming_filter_config(
        min_mapq=min_mapq,
        min_read_length=min_read_length,
        primary_only=primary_only,
        process_unmapped=process_unmapped,
        train_rids=train_rids,
    )

    counters = _new_streaming_counters()

    posterior_writer = None
    posterior_stats = None
    return_posteriors = False

    _log = _streaming_log_for_output(output_bam)

    print(f"Processing BAM (streaming pipeline, {n_cores} workers, "
          f"chunk_size={chunk_size}, max_inflight={max_inflight})...", file=_log)
    _log.flush()

    start_time = time.time()

    _output_target = _streaming_output_target(output_bam)

    with pysam.AlignmentFile(input_bam, "rb", threads=io_threads, check_sq=False) as inbam:
        with pysam.AlignmentFile(_output_target, "wb",
                                 header=append_coord_marker(inbam.header),
                                 threads=io_threads) as outbam:

            posterior_writer, return_posteriors = _open_streaming_posterior_writer(
                output_posteriors, mode, context_size, edge_trim, input_bam, _log,
            )

            executor = _new_apply_streaming_executor(
                model_path, n_cores, debug_timing,
            )

            worker_args = _apply_worker_args(
                edge_trim, circular, mode, context_size,
                msp_min_size, nuc_min_size, with_scores,
                return_posteriors, prob_threshold,
            )
            skipped = 0

            try:
                total_reads, skipped = _run_streaming_worker_loop(
                    inbam.fetch(until_eof=True),
                    filter_config,
                    mode,
                    ref_fasta,
                    executor,
                    _process_payload_chunk_worker,
                    worker_args,
                    max_reads,
                    chunk_size,
                    max_inflight,
                    lambda inflight: lambda: _drain_oldest_chunk(
                        inflight, outbam, with_scores, write_msps,
                        posterior_writer, counters,
                    ),
                    skip_reasons,
                    _log,
                    "Processed",
                    "reads/s",
                    start_time,
                )

            finally:
                if posterior_writer:
                    posterior_stats = posterior_writer.close()
                    posterior_writer = None

    reads_with_footprints = _finalize_apply_streaming_pipeline(
        total_reads=total_reads,
        skipped=skipped,
        counters=counters,
        start_time=start_time,
        skip_reasons=skip_reasons,
        output_bam=output_bam,
        process_unmapped=process_unmapped,
        n_cores=n_cores,
        posterior_stats=posterior_stats,
        output_posteriors=output_posteriors,
        log=_log,
    )

    return total_reads, reads_with_footprints
