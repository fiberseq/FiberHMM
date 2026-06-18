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
from fiberhmm.inference.mp_context import _MP_CONTEXT
from fiberhmm.inference.read_filters import ReadFilterConfig, streaming_skip_reason
from fiberhmm.inference.skip_reasons import (
    NO_FOOTPRINTS_SKIP_REASON,
    iter_nonzero_skip_reasons,
    new_skip_reasons,
    record_skip_reason,
)
from fiberhmm.inference.streaming_drain import (
    _drain_oldest_chunk,
    _drain_oldest_fused_chunk,
    _SubmittedChunk,
)
from fiberhmm.inference.streaming_workers import (
    _init_bam_worker,
    _init_fused_worker,
    _process_fused_payload_chunk_worker,
    _process_payload_chunk_worker,
)
from fiberhmm.io.bam_header import append_coord_marker, maybe_append_pg

try:
    from fiberhmm.posteriors.hdf5_backend import PosteriorWriter
    HAS_POSTERIOR_WRITER = True
except ImportError:
    HAS_POSTERIOR_WRITER = False


@dataclass(frozen=True)
class _StreamingPayloadResult:
    payload: object | None
    skip_reason: Optional[str]


@dataclass(frozen=True)
class _StreamingProgressRates:
    instant: float
    average: float


@dataclass(frozen=True)
class _StreamingProgressCheckpoint:
    reads: int
    time: float


@dataclass(frozen=True)
class _StreamingChunkBuffers:
    items: list
    read_objs: list
    skip_flags: list


@dataclass(frozen=True)
class _ProcessableReadBufferRequest:
    chunk_items: list
    chunk_read_objs: list
    chunk_skip_flags: list
    payload: object
    read: object


@dataclass(frozen=True)
class _SkippedReadBufferRequest:
    chunk_read_objs: list
    chunk_skip_flags: list
    skip_reasons: dict
    read: object
    reason: str


@dataclass(frozen=True)
class _StreamingChunkSubmitRequest:
    inflight: object
    executor: object
    worker_fn: object
    buffers: _StreamingChunkBuffers
    worker_args: tuple


@dataclass(frozen=True)
class _StreamingChunkFlushRequest:
    inflight: object
    executor: object
    worker_fn: object
    buffers: _StreamingChunkBuffers
    worker_args: tuple
    max_inflight: int
    drain_chunk: Callable[[], None]


@dataclass(frozen=True)
class _StreamingFlushProgress:
    buffers: _StreamingChunkBuffers
    last_progress_reads: int
    last_progress_time: float


@dataclass(frozen=True)
class _StreamingReadCounts:
    processed: int
    skipped: int


@dataclass(frozen=True)
class _StreamingReadDelta:
    processed: int
    skipped: int


@dataclass(frozen=True)
class _StreamingWorkerCommonArgs:
    edge_trim: int
    circular: bool
    mode: str
    context_size: int
    msp_min_size: int
    nuc_min_size: int
    with_scores: bool

    def as_tuple(self) -> tuple:
        return (
            self.edge_trim,
            self.circular,
            self.mode,
            self.context_size,
            self.msp_min_size,
            self.nuc_min_size,
            self.with_scores,
        )


@dataclass(frozen=True)
class _StreamingApplyWorkerArgs:
    common: _StreamingWorkerCommonArgs
    return_posteriors: bool
    prob_threshold: int

    def as_tuple(self) -> tuple:
        return (
            *self.common.as_tuple(),
            self.return_posteriors,
            self.prob_threshold,
        )


@dataclass(frozen=True)
class _StreamingFusedWorkerArgs:
    common: _StreamingWorkerCommonArgs
    prob_threshold: int
    min_llr: float
    min_opps: int
    unify_threshold: int

    def as_tuple(self) -> tuple:
        return (
            *self.common.as_tuple(),
            self.prob_threshold,
            self.common.mode,
            self.common.context_size,
            self.min_llr,
            self.min_opps,
            self.unify_threshold,
        )


def _buffer_skipped_read_from_request(
    request: _SkippedReadBufferRequest,
) -> int:
    request.chunk_read_objs.append(request.read)
    request.chunk_skip_flags.append(True)
    record_skip_reason(request.skip_reasons, request.reason)
    return 1


def _buffer_skipped_read(chunk_read_objs, chunk_skip_flags, skip_reasons, read, reason) -> int:
    return _buffer_skipped_read_from_request(
        _SkippedReadBufferRequest(
            chunk_read_objs=chunk_read_objs,
            chunk_skip_flags=chunk_skip_flags,
            skip_reasons=skip_reasons,
            read=read,
            reason=reason,
        )
    )


def _buffer_processable_read_from_request(
    request: _ProcessableReadBufferRequest,
) -> int:
    request.chunk_items.append(request.payload)
    request.chunk_read_objs.append(request.read)
    request.chunk_skip_flags.append(False)
    return 1


def _buffer_processable_read(chunk_items, chunk_read_objs, chunk_skip_flags, payload, read) -> int:
    return _buffer_processable_read_from_request(
        _ProcessableReadBufferRequest(
            chunk_items=chunk_items,
            chunk_read_objs=chunk_read_objs,
            chunk_skip_flags=chunk_skip_flags,
            payload=payload,
            read=read,
        )
    )


def _streaming_payload_or_skip(read, filter_config: ReadFilterConfig,
                               mode: str, ref_fasta=None) -> _StreamingPayloadResult:
    skip_reason = streaming_skip_reason(read, filter_config)
    if skip_reason:
        return _StreamingPayloadResult(payload=None, skip_reason=skip_reason)

    payload = make_apply_payload(read, mode=mode, ref_fasta=ref_fasta)
    if payload is None:
        return _StreamingPayloadResult(
            payload=None,
            skip_reason='no_modifications',
        )

    return _StreamingPayloadResult(payload=payload, skip_reason=None)


def _buffer_streaming_read(
    read,
    filter_config: ReadFilterConfig,
    mode: str,
    ref_fasta,
    chunk_items,
    chunk_read_objs,
    chunk_skip_flags,
    skip_reasons,
) -> _StreamingReadDelta:
    payload_result = _streaming_payload_or_skip(
        read, filter_config, mode, ref_fasta,
    )
    if payload_result.skip_reason:
        return _StreamingReadDelta(
            processed=0,
            skipped=_buffer_skipped_read(
                chunk_read_objs,
                chunk_skip_flags,
                skip_reasons,
                read,
                payload_result.skip_reason,
            ),
        )

    return _StreamingReadDelta(
        processed=_buffer_processable_read(
            chunk_items,
            chunk_read_objs,
            chunk_skip_flags,
            payload_result.payload,
            read,
        ),
        skipped=0,
    )


def _completed_empty_future() -> Future:
    future = Future()
    future.set_result([])
    return future


def _new_streaming_chunk_buffers() -> _StreamingChunkBuffers:
    return _StreamingChunkBuffers(items=[], read_objs=[], skip_flags=[])


def _submit_streaming_chunk_from_request(
    request: _StreamingChunkSubmitRequest,
) -> None:
    if request.buffers.items:
        future = request.executor.submit(
            request.worker_fn,
            request.buffers.items,
            *request.worker_args,
        )
    else:
        future = _completed_empty_future()
    request.inflight.append(
        _SubmittedChunk(
            future=future,
            read_objs=request.buffers.read_objs,
            items=request.buffers.items,
            skip_flags=request.buffers.skip_flags,
        )
    )


def _submit_streaming_chunk(inflight, executor, worker_fn, chunk_items,
                            chunk_read_objs, chunk_skip_flags, worker_args) -> None:
    _submit_streaming_chunk_from_request(
        _StreamingChunkSubmitRequest(
            inflight=inflight,
            executor=executor,
            worker_fn=worker_fn,
            buffers=_StreamingChunkBuffers(
                items=chunk_items,
                read_objs=chunk_read_objs,
                skip_flags=chunk_skip_flags,
            ),
            worker_args=worker_args,
        )
    )


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
) -> _StreamingChunkBuffers:
    return _flush_streaming_chunk_from_request(
        _StreamingChunkFlushRequest(
            inflight=inflight,
            executor=executor,
            worker_fn=worker_fn,
            buffers=_StreamingChunkBuffers(
                items=chunk_items,
                read_objs=chunk_read_objs,
                skip_flags=chunk_skip_flags,
            ),
            worker_args=worker_args,
            max_inflight=max_inflight,
            drain_chunk=drain_chunk,
        )
    )


def _flush_streaming_chunk_from_request(
    request: _StreamingChunkFlushRequest,
) -> _StreamingChunkBuffers:
    _drain_if_inflight_full(
        request.inflight,
        request.max_inflight,
        request.drain_chunk,
    )
    _submit_streaming_chunk_from_request(
        _StreamingChunkSubmitRequest(
            inflight=request.inflight,
            executor=request.executor,
            worker_fn=request.worker_fn,
            buffers=request.buffers,
            worker_args=request.worker_args,
        )
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


@dataclass(frozen=True)
class _StreamingFlushAndProgressRequest:
    context: _StreamingFlushContext
    buffers: _StreamingChunkBuffers
    total_reads: int
    skipped: int
    last_progress_reads: int
    last_progress_time: float


@dataclass(frozen=True)
class _StreamReadsToWorkersRequest:
    reads: object
    filter_config: ReadFilterConfig
    mode: str
    ref_fasta: object
    inflight: object
    executor: object
    worker_fn: object
    worker_args: tuple
    max_reads: Optional[int]
    chunk_size: int
    max_inflight: int
    drain_chunk: Callable[[], None]
    skip_reasons: dict
    log: object
    progress_label: str
    rate_unit: str
    start_time: float


@dataclass(frozen=True)
class _StreamingWorkerLoopRequest:
    reads: object
    filter_config: ReadFilterConfig
    mode: str
    ref_fasta: object
    executor: object
    worker_fn: object
    worker_args: tuple
    max_reads: Optional[int]
    chunk_size: int
    max_inflight: int
    drain_chunk_factory: object
    skip_reasons: dict
    log: object
    progress_label: str
    rate_unit: str
    start_time: float


@dataclass(frozen=True)
class _StreamingPosteriorWriter:
    writer: object | None
    enabled: bool


@dataclass(frozen=True)
class _StreamingPosteriorWriterOpenRequest:
    output_posteriors: Optional[str]
    mode: str
    context_size: int
    edge_trim: int
    input_bam: str
    log: object


@dataclass(frozen=True)
class _StreamingPosteriorStats:
    n_fibers: int
    file_size_mb: float

    def as_tuple(self) -> tuple[int, float]:
        return self.n_fibers, self.file_size_mb


@dataclass(frozen=True)
class _ApplyStreamingFinalizeRequest:
    total_reads: int
    skipped: int
    counters: dict
    start_time: float
    skip_reasons: dict
    output_bam: str
    process_unmapped: bool
    n_cores: int
    posterior_stats: object
    output_posteriors: Optional[str]
    log: object


@dataclass(frozen=True)
class _FusedStreamingFinalizeRequest:
    total_reads: int
    skipped: int
    counters: dict
    start_time: float
    skip_reasons: dict
    log: object


@dataclass(frozen=True)
class _FusedStreamingPipelineRequest:
    input_bam: str
    output_bam: str
    model_path: str
    recall_model_path: Optional[str]
    train_rids: Set[str]
    edge_trim: int
    circular: bool
    mode: str
    context_size: int
    msp_min_size: int
    nuc_min_size: int
    min_mapq: int
    prob_threshold: int
    min_read_length: int
    with_scores: bool
    min_llr: float
    min_opps: int
    unify_threshold: int
    emission_uplift: float
    also_write_legacy: bool
    downstream_compat: bool
    max_reads: Optional[int]
    n_cores: int
    chunk_size: int
    io_threads: int
    process_unmapped: bool
    primary_only: bool
    ref_fasta_path: Optional[str]
    recall_nucs: bool
    split_min_llr: float
    split_min_opps: int
    filter_chimeras: bool
    chimera_min_seg: int
    chimera_purity: float
    phase_nrl: int
    pg_record: Optional[dict]


@dataclass(frozen=True)
class _ApplyStreamingPipelineRequest:
    input_bam: str
    output_bam: str
    model_path: str
    train_rids: Set[str]
    edge_trim: int
    circular: bool
    mode: str
    context_size: int
    msp_min_size: int
    nuc_min_size: int
    min_mapq: int
    prob_threshold: int
    min_read_length: int
    with_scores: bool
    n_cores: int
    chunk_size: int
    max_inflight: Optional[int]
    io_threads: int
    primary_only: bool
    output_posteriors: Optional[str]
    write_msps: bool
    max_reads: Optional[int]
    debug_timing: bool
    process_unmapped: bool


@dataclass(frozen=True)
class _ApplyStreamingReadLoopRequest:
    pipeline_request: _ApplyStreamingPipelineRequest
    inbam: object
    outbam: object
    filter_config: ReadFilterConfig
    max_inflight: int
    counters: dict
    skip_reasons: dict
    log: object
    ref_fasta: object
    start_time: float


@dataclass(frozen=True)
class _ApplyStreamingReadLoopResult:
    read_counts: _StreamingReadCounts
    posterior_stats: object


def _flush_streaming_chunk_and_report_progress(
    context: _StreamingFlushContext,
    chunk_items,
    chunk_read_objs,
    chunk_skip_flags,
    total_reads: int,
    skipped: int,
    last_progress_reads: int,
    last_progress_time: float,
) -> _StreamingFlushProgress:
    return _flush_streaming_chunk_and_report_progress_from_request(
        _StreamingFlushAndProgressRequest(
            context=context,
            buffers=_StreamingChunkBuffers(
                items=chunk_items,
                read_objs=chunk_read_objs,
                skip_flags=chunk_skip_flags,
            ),
            total_reads=total_reads,
            skipped=skipped,
            last_progress_reads=last_progress_reads,
            last_progress_time=last_progress_time,
        )
    )


def _flush_streaming_chunk_and_report_progress_from_request(
    request: _StreamingFlushAndProgressRequest,
) -> _StreamingFlushProgress:
    context = request.context
    buffers = _flush_streaming_chunk(
        context.inflight,
        context.executor,
        context.worker_fn,
        request.buffers.items,
        request.buffers.read_objs,
        request.buffers.skip_flags,
        context.worker_args,
        context.max_inflight,
        context.drain_chunk,
    )
    now = time.time()
    checkpoint = _print_streaming_progress(
        context.log,
        context.progress_label,
        request.total_reads,
        request.skipped,
        len(context.inflight),
        request.last_progress_reads,
        context.start_time,
        request.last_progress_time,
        now,
        context.rate_unit,
    )
    return _StreamingFlushProgress(
        buffers=buffers,
        last_progress_reads=checkpoint.reads,
        last_progress_time=checkpoint.time,
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
) -> _StreamingReadCounts:
    return _stream_reads_to_workers_from_request(
        _StreamReadsToWorkersRequest(
            reads=reads,
            filter_config=filter_config,
            mode=mode,
            ref_fasta=ref_fasta,
            inflight=inflight,
            executor=executor,
            worker_fn=worker_fn,
            worker_args=worker_args,
            max_reads=max_reads,
            chunk_size=chunk_size,
            max_inflight=max_inflight,
            drain_chunk=drain_chunk,
            skip_reasons=skip_reasons,
            log=log,
            progress_label=progress_label,
            rate_unit=rate_unit,
            start_time=start_time,
        )
    )


def _stream_reads_to_workers_from_request(
    request: _StreamReadsToWorkersRequest,
) -> _StreamingReadCounts:
    total_reads = 0
    skipped = 0
    chunk_buffers = _new_streaming_chunk_buffers()
    last_progress_reads = 0
    last_progress_time = time.time()
    flush_context = _StreamingFlushContext(
        inflight=request.inflight,
        executor=request.executor,
        worker_fn=request.worker_fn,
        worker_args=request.worker_args,
        max_inflight=request.max_inflight,
        drain_chunk=request.drain_chunk,
        log=request.log,
        progress_label=request.progress_label,
        start_time=request.start_time,
        rate_unit=request.rate_unit,
    )

    for read in request.reads:
        read_delta = _buffer_streaming_read(
            read,
            request.filter_config,
            request.mode,
            request.ref_fasta,
            chunk_buffers.items,
            chunk_buffers.read_objs,
            chunk_buffers.skip_flags,
            request.skip_reasons,
        )
        skipped += read_delta.skipped
        if read_delta.skipped:
            continue

        total_reads += read_delta.processed

        if request.max_reads and total_reads >= request.max_reads:
            break

        if len(chunk_buffers.read_objs) >= request.chunk_size:
            flush_progress = _flush_streaming_chunk_and_report_progress(
                flush_context,
                chunk_buffers.items,
                chunk_buffers.read_objs,
                chunk_buffers.skip_flags,
                total_reads,
                skipped,
                last_progress_reads,
                last_progress_time,
            )
            chunk_buffers = flush_progress.buffers
            last_progress_reads = flush_progress.last_progress_reads
            last_progress_time = flush_progress.last_progress_time

    if chunk_buffers.read_objs:
        _flush_streaming_chunk(
            request.inflight,
            request.executor,
            request.worker_fn,
            chunk_buffers.items,
            chunk_buffers.read_objs,
            chunk_buffers.skip_flags,
            request.worker_args,
            request.max_inflight,
            request.drain_chunk,
        )

    _drain_all_streaming_chunks(request.inflight, request.drain_chunk)
    return _StreamingReadCounts(processed=total_reads, skipped=skipped)


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
) -> _StreamingReadCounts:
    return _run_streaming_worker_loop_from_request(
        _StreamingWorkerLoopRequest(
            reads=reads,
            filter_config=filter_config,
            mode=mode,
            ref_fasta=ref_fasta,
            executor=executor,
            worker_fn=worker_fn,
            worker_args=worker_args,
            max_reads=max_reads,
            chunk_size=chunk_size,
            max_inflight=max_inflight,
            drain_chunk_factory=drain_chunk_factory,
            skip_reasons=skip_reasons,
            log=log,
            progress_label=progress_label,
            rate_unit=rate_unit,
            start_time=start_time,
        )
    )


def _run_streaming_worker_loop_from_request(
    request: _StreamingWorkerLoopRequest,
) -> _StreamingReadCounts:
    inflight = deque()
    drain_chunk = request.drain_chunk_factory(inflight)
    try:
        return _stream_reads_to_workers(
            request.reads,
            request.filter_config,
            request.mode,
            request.ref_fasta,
            inflight,
            request.executor,
            request.worker_fn,
            request.worker_args,
            request.max_reads,
            request.chunk_size,
            request.max_inflight,
            drain_chunk,
            request.skip_reasons,
            request.log,
            request.progress_label,
            request.rate_unit,
            request.start_time,
        )
    finally:
        request.executor.shutdown(wait=True)


def _worker_common_args_from_request(
    request: _StreamingWorkerCommonArgs,
) -> tuple:
    return request.as_tuple()


def _new_streaming_worker_common_args(
    edge_trim: int,
    circular: bool,
    mode: str,
    context_size: int,
    msp_min_size: int,
    nuc_min_size: int,
    with_scores: bool,
) -> _StreamingWorkerCommonArgs:
    return _StreamingWorkerCommonArgs(
        edge_trim=edge_trim,
        circular=circular,
        mode=mode,
        context_size=context_size,
        msp_min_size=msp_min_size,
        nuc_min_size=nuc_min_size,
        with_scores=with_scores,
    )


def _worker_common_args(
    edge_trim: int,
    circular: bool,
    mode: str,
    context_size: int,
    msp_min_size: int,
    nuc_min_size: int,
    with_scores: bool,
) -> tuple:
    return _worker_common_args_from_request(
        _new_streaming_worker_common_args(
            edge_trim,
            circular,
            mode,
            context_size,
            msp_min_size,
            nuc_min_size,
            with_scores,
        )
    )


def _apply_worker_args_from_request(
    request: _StreamingApplyWorkerArgs,
) -> tuple:
    return request.as_tuple()


def _fused_worker_args_from_request(
    request: _StreamingFusedWorkerArgs,
) -> tuple:
    return request.as_tuple()


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
    return _apply_worker_args_from_request(
        _StreamingApplyWorkerArgs(
            common=_new_streaming_worker_common_args(
                edge_trim,
                circular,
                mode,
                context_size,
                msp_min_size,
                nuc_min_size,
                with_scores,
            ),
            return_posteriors=return_posteriors,
            prob_threshold=prob_threshold,
        ),
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
    return _fused_worker_args_from_request(
        _StreamingFusedWorkerArgs(
            common=_new_streaming_worker_common_args(
                edge_trim,
                circular,
                mode,
                context_size,
                msp_min_size,
                nuc_min_size,
                with_scores,
            ),
            prob_threshold=prob_threshold,
            min_llr=min_llr,
            min_opps=min_opps,
            unify_threshold=unify_threshold,
        ),
    )


def _apply_worker_args_from_pipeline_request(
    request: _ApplyStreamingPipelineRequest,
    return_posteriors: bool,
) -> tuple:
    return _apply_worker_args(
        request.edge_trim,
        request.circular,
        request.mode,
        request.context_size,
        request.msp_min_size,
        request.nuc_min_size,
        request.with_scores,
        return_posteriors,
        request.prob_threshold,
    )


def _fused_worker_args_from_pipeline_request(
    request: _FusedStreamingPipelineRequest,
) -> tuple:
    return _fused_worker_args(
        request.edge_trim,
        request.circular,
        request.mode,
        request.context_size,
        request.msp_min_size,
        request.nuc_min_size,
        request.with_scores,
        request.prob_threshold,
        request.min_llr,
        request.min_opps,
        request.unify_threshold,
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


def _streaming_filter_config_from_request(request) -> ReadFilterConfig:
    return _streaming_filter_config(
        min_mapq=request.min_mapq,
        min_read_length=request.min_read_length,
        primary_only=request.primary_only,
        process_unmapped=request.process_unmapped,
        train_rids=request.train_rids,
    )


def _apply_streaming_max_inflight(
    request: _ApplyStreamingPipelineRequest,
) -> int:
    if request.max_inflight is not None:
        return request.max_inflight
    return 2 * request.n_cores


def _fused_streaming_max_inflight(
    request: _FusedStreamingPipelineRequest,
) -> int:
    return request.n_cores + 2


def _run_apply_streaming_reads_from_request(
    request: _ApplyStreamingReadLoopRequest,
) -> _ApplyStreamingReadLoopResult:
    pipeline_request = request.pipeline_request
    posterior_writer = None
    posterior_stats = None
    try:
        posterior_output = _open_streaming_posterior_writer(
            pipeline_request.output_posteriors,
            pipeline_request.mode,
            pipeline_request.context_size,
            pipeline_request.edge_trim,
            pipeline_request.input_bam,
            request.log,
        )
        posterior_writer = posterior_output.writer

        executor = _new_apply_streaming_executor(
            pipeline_request.model_path,
            pipeline_request.n_cores,
            pipeline_request.debug_timing,
        )

        worker_args = _apply_worker_args_from_pipeline_request(
            pipeline_request,
            posterior_output.enabled,
        )

        read_counts = _run_streaming_worker_loop(
            request.inbam.fetch(until_eof=True),
            request.filter_config,
            pipeline_request.mode,
            request.ref_fasta,
            executor,
            _process_payload_chunk_worker,
            worker_args,
            pipeline_request.max_reads,
            pipeline_request.chunk_size,
            request.max_inflight,
            _apply_drain_chunk_factory(
                request.outbam,
                pipeline_request.with_scores,
                pipeline_request.write_msps,
                posterior_writer,
                request.counters,
            ),
            request.skip_reasons,
            request.log,
            "Processed",
            "reads/s",
            request.start_time,
        )
    finally:
        posterior_stats = _close_streaming_posterior_writer(posterior_writer)

    return _ApplyStreamingReadLoopResult(
        read_counts=read_counts,
        posterior_stats=posterior_stats,
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
) -> _StreamingPosteriorWriter:
    return _open_streaming_posterior_writer_from_request(
        _StreamingPosteriorWriterOpenRequest(
            output_posteriors=output_posteriors,
            mode=mode,
            context_size=context_size,
            edge_trim=edge_trim,
            input_bam=input_bam,
            log=log,
        )
    )


def _open_streaming_posterior_writer_from_request(
    request: _StreamingPosteriorWriterOpenRequest,
) -> _StreamingPosteriorWriter:
    if not request.output_posteriors:
        return _StreamingPosteriorWriter(writer=None, enabled=False)

    if not HAS_POSTERIOR_WRITER:
        print(
            "WARNING: posterior_writer.py not found, skipping posteriors export",
            file=request.log,
        )
        return _StreamingPosteriorWriter(writer=None, enabled=False)

    writer = PosteriorWriter(
        request.output_posteriors,
        request.mode,
        request.context_size,
        request.edge_trim,
        request.input_bam,
        batch_size=1000,
    )
    print(
        f"Posteriors will be written to: {request.output_posteriors}",
        file=request.log,
    )
    return _StreamingPosteriorWriter(writer=writer, enabled=True)


def _close_streaming_posterior_writer(posterior_writer):
    if posterior_writer is None:
        return None
    closed_stats = posterior_writer.close()
    if closed_stats is None:
        return None
    n_fibers, file_size_mb = closed_stats
    return _StreamingPosteriorStats(
        n_fibers=n_fibers,
        file_size_mb=file_size_mb,
    )


def _close_ref_fasta(ref_fasta) -> None:
    if ref_fasta is not None:
        ref_fasta.close()


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


def _apply_drain_chunk_factory(
    outbam,
    with_scores: bool,
    write_msps: bool,
    posterior_writer,
    counters: dict,
):
    def drain_chunk_factory(inflight):
        def drain_chunk():
            _drain_oldest_chunk(
                inflight,
                outbam,
                with_scores,
                write_msps,
                posterior_writer,
                counters,
            )

        return drain_chunk

    return drain_chunk_factory


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


def _fused_drain_chunk_factory(
    outbam,
    with_scores: bool,
    also_write_legacy: bool,
    downstream_compat: bool,
    counters: dict,
):
    def drain_chunk_factory(inflight):
        def drain_chunk():
            _drain_oldest_fused_chunk(
                inflight,
                outbam,
                with_scores,
                also_write_legacy,
                downstream_compat,
                counters,
            )

        return drain_chunk

    return drain_chunk_factory


def _print_streaming_skip_summary(skip_reasons: dict, total_reads: int,
                                  skipped: int, log) -> None:
    if skipped <= 0:
        return
    print("  Skip reasons:", file=log)
    for reason, count in iter_nonzero_skip_reasons(skip_reasons):
        pct = 100 * count / (total_reads + skipped)
        print(f"    {reason}: {count:,} ({pct:.1f}%)", file=log)


def _streaming_progress_rates(
    total_reads: int,
    last_progress_reads: int,
    start_time: float,
    last_progress_time: float,
    now: float,
) -> _StreamingProgressRates:
    elapsed = now - start_time
    avg_rate = total_reads / elapsed if elapsed > 0 else 0
    dt = now - last_progress_time
    inst_rate = (total_reads - last_progress_reads) / dt if dt > 0 else 0
    return _StreamingProgressRates(instant=inst_rate, average=avg_rate)


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
) -> _StreamingProgressCheckpoint:
    rates = _streaming_progress_rates(
        total_reads, last_progress_reads, start_time, last_progress_time, now,
    )
    print(
        _streaming_progress_message(
            label,
            total_reads,
            skipped,
            inflight_count,
            rates.instant,
            rates.average,
            rate_unit,
        ),
        end='',
        file=log,
    )
    log.flush()
    return _StreamingProgressCheckpoint(reads=total_reads, time=now)


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
    print(
        f"Posteriors: {posterior_stats.n_fibers:,} fibers -> "
        f"{output_posteriors} ({posterior_stats.file_size_mb:.1f} MB)",
        file=log,
    )


def _should_sort_streaming_output(output_bam: str, process_unmapped: bool) -> bool:
    return output_bam != '-' and not process_unmapped


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
    return _finalize_apply_streaming_pipeline_from_request(
        _ApplyStreamingFinalizeRequest(
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
            log=log,
        )
    )


def _finalize_apply_streaming_pipeline_from_request(
    request: _ApplyStreamingFinalizeRequest,
) -> int:
    reads_with_footprints = _print_streaming_completion_summary(
        "Processed",
        request.total_reads,
        request.skipped,
        request.counters,
        time.time() - request.start_time,
        "reads/s",
        request.skip_reasons,
        request.log,
    )

    if _should_sort_streaming_output(request.output_bam, request.process_unmapped):
        _sort_and_index_bam(request.output_bam, threads=request.n_cores)

    _print_streaming_posterior_summary(
        request.posterior_stats,
        request.output_posteriors,
        request.log,
    )
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
    return _finalize_fused_streaming_pipeline_from_request(
        _FusedStreamingFinalizeRequest(
            total_reads=total_reads,
            skipped=skipped,
            counters=counters,
            start_time=start_time,
            skip_reasons=skip_reasons,
            log=log,
        )
    )


def _finalize_fused_streaming_pipeline_from_request(
    request: _FusedStreamingFinalizeRequest,
) -> int:
    reads_with_fp = _print_streaming_completion_summary(
        "Fused",
        request.total_reads,
        request.skipped,
        request.counters,
        time.time() - request.start_time,
        "r/s",
        request.skip_reasons,
        request.log,
    )
    _print_fused_chimera_summary(request.counters, request.log)
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
    return _process_bam_streaming_pipeline_fused_from_request(
        _FusedStreamingPipelineRequest(
            input_bam=input_bam,
            output_bam=output_bam,
            model_path=model_path,
            recall_model_path=recall_model_path,
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
            min_llr=min_llr,
            min_opps=min_opps,
            unify_threshold=unify_threshold,
            emission_uplift=emission_uplift,
            also_write_legacy=also_write_legacy,
            downstream_compat=downstream_compat,
            max_reads=max_reads,
            n_cores=n_cores,
            chunk_size=chunk_size,
            io_threads=io_threads,
            process_unmapped=process_unmapped,
            primary_only=primary_only,
            ref_fasta_path=ref_fasta_path,
            recall_nucs=recall_nucs,
            split_min_llr=split_min_llr,
            split_min_opps=split_min_opps,
            filter_chimeras=filter_chimeras,
            chimera_min_seg=chimera_min_seg,
            chimera_purity=chimera_purity,
            phase_nrl=phase_nrl,
            pg_record=pg_record,
        )
    )


def _process_bam_streaming_pipeline_fused_from_request(
    request: _FusedStreamingPipelineRequest,
) -> Tuple[int, int]:
    ref_fasta = None
    pysam.set_verbosity(0)
    max_inflight = _fused_streaming_max_inflight(request)
    start_time = time.time()
    counters = _new_streaming_counters()

    total_reads = 0
    skipped = 0
    skip_reasons = _new_streaming_skip_reasons()
    filter_config = _streaming_filter_config_from_request(request)

    _log = _streaming_log_for_output(request.output_bam)

    _output_target = _streaming_output_target(request.output_bam)

    with pysam.AlignmentFile(
        request.input_bam,
        "rb",
        threads=request.io_threads,
        check_sq=False,
    ) as inbam:
        with pysam.AlignmentFile(_output_target, "wb",
                                 header=maybe_append_pg(inbam.header, request.pg_record),
                                 threads=request.io_threads) as outbam:
            try:
                if request.ref_fasta_path:
                    ref_fasta = pysam.FastaFile(request.ref_fasta_path)

                executor = _new_fused_streaming_executor(
                    request.model_path,
                    request.recall_model_path,
                    request.emission_uplift,
                    request.recall_nucs,
                    request.split_min_llr,
                    request.split_min_opps,
                    request.filter_chimeras,
                    request.chimera_min_seg,
                    request.chimera_purity,
                    request.phase_nrl,
                    request.n_cores,
                )

                worker_args = _fused_worker_args_from_pipeline_request(request)

                read_counts = _run_streaming_worker_loop(
                    inbam.fetch(until_eof=True),
                    filter_config,
                    request.mode,
                    ref_fasta,
                    executor,
                    _process_fused_payload_chunk_worker,
                    worker_args,
                    request.max_reads,
                    request.chunk_size,
                    max_inflight,
                    _fused_drain_chunk_factory(
                        outbam,
                        request.with_scores,
                        request.also_write_legacy,
                        request.downstream_compat,
                        counters,
                    ),
                    skip_reasons,
                    _log,
                    "Fused",
                    "r/s",
                    start_time,
                )
                total_reads = read_counts.processed
                skipped = read_counts.skipped
            finally:
                _close_ref_fasta(ref_fasta)
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
    return _process_bam_streaming_pipeline_from_request(
        _ApplyStreamingPipelineRequest(
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
            max_inflight=max_inflight,
            io_threads=io_threads,
            primary_only=primary_only,
            output_posteriors=output_posteriors,
            write_msps=write_msps,
            max_reads=max_reads,
            debug_timing=debug_timing,
            process_unmapped=process_unmapped,
        )
    )


def _process_bam_streaming_pipeline_from_request(
    request: _ApplyStreamingPipelineRequest,
) -> Tuple[int, int]:
    max_inflight = _apply_streaming_max_inflight(request)

    pysam.set_verbosity(0)

    ref_fasta = None

    total_reads = 0
    skip_reasons = _new_streaming_skip_reasons(include_no_footprints=True)
    filter_config = _streaming_filter_config_from_request(request)

    counters = _new_streaming_counters()

    posterior_stats = None

    _log = _streaming_log_for_output(request.output_bam)

    print(f"Processing BAM (streaming pipeline, {request.n_cores} workers, "
          f"chunk_size={request.chunk_size}, max_inflight={max_inflight})...",
          file=_log)
    _log.flush()

    start_time = time.time()

    _output_target = _streaming_output_target(request.output_bam)

    with pysam.AlignmentFile(
        request.input_bam,
        "rb",
        threads=request.io_threads,
        check_sq=False,
    ) as inbam:
        with pysam.AlignmentFile(_output_target, "wb",
                                 header=append_coord_marker(inbam.header),
                                 threads=request.io_threads) as outbam:
            skipped = 0

            read_loop_result = _run_apply_streaming_reads_from_request(
                _ApplyStreamingReadLoopRequest(
                    pipeline_request=request,
                    inbam=inbam,
                    outbam=outbam,
                    filter_config=filter_config,
                    max_inflight=max_inflight,
                    counters=counters,
                    skip_reasons=skip_reasons,
                    log=_log,
                    ref_fasta=ref_fasta,
                    start_time=start_time,
                )
            )
            total_reads = read_loop_result.read_counts.processed
            skipped = read_loop_result.read_counts.skipped
            posterior_stats = read_loop_result.posterior_stats

    reads_with_footprints = _finalize_apply_streaming_pipeline(
        total_reads=total_reads,
        skipped=skipped,
        counters=counters,
        start_time=start_time,
        skip_reasons=skip_reasons,
        output_bam=request.output_bam,
        process_unmapped=request.process_unmapped,
        n_cores=request.n_cores,
        posterior_stats=posterior_stats,
        output_posteriors=request.output_posteriors,
        log=_log,
    )

    return total_reads, reads_with_footprints
