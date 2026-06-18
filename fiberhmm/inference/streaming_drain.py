"""Order-preserving drain helpers for streaming inference pipelines."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fiberhmm.inference.posterior_records import posterior_fiber_data
from fiberhmm.inference.streaming_workers import CHIMERA_RESULT
from fiberhmm.inference.tagging import set_legacy_apply_tags, write_fused_recall_tags
from fiberhmm.inference.worker_results import coerce_worker_chunk_result

try:
    from fiberhmm.posteriors.hdf5_backend import get_ref_positions_from_read
    HAS_POSTERIOR_WRITER = True
except ImportError:
    HAS_POSTERIOR_WRITER = False


@dataclass(frozen=True)
class _SubmittedChunk:
    future: object
    read_objs: object
    items: object
    skip_flags: object


@dataclass(frozen=True)
class _InflightChunk:
    results: object
    worker_failures: int
    read_objs: object
    items: object
    skip_flags: object


@dataclass(frozen=True)
class _PosteriorFiberAddRequest:
    posterior_writer: object
    read_obj: object
    result: dict


@dataclass(frozen=True)
class _DrainChunkInOrderRequest:
    chunk_read_objs: object
    chunk_items: object
    chunk_skip_flags: object
    results: object
    outbam: object
    counters: object
    record_result: object


@dataclass(frozen=True)
class _DrainOldestWithRecorderRequest:
    inflight: object
    outbam: object
    counters: object
    record_result: object


def _increment_counter(counters, key: str, amount: int = 1) -> None:
    counters[key] = counters.get(key, 0) + amount


def _record_worker_failures(counters, worker_failures: int) -> None:
    if worker_failures:
        _increment_counter(counters, 'worker_failures', worker_failures)


def _record_reads_with_footprints(counters) -> None:
    _increment_counter(counters, 'reads_with_footprints')


def _record_no_footprints(counters) -> None:
    _increment_counter(counters, 'no_footprints')


def _record_chimera(counters) -> None:
    _increment_counter(counters, 'chimera')


def _write_passthrough(outbam, read_obj, counters) -> None:
    outbam.write(read_obj)
    _increment_counter(counters, 'written')


def _empty_ref_positions():
    return np.array([], dtype=np.int32)


def _posterior_ref_positions(read_obj):
    if not HAS_POSTERIOR_WRITER:
        return _empty_ref_positions()
    try:
        return get_ref_positions_from_read(read_obj)
    except Exception:
        return _empty_ref_positions()


def _fused_read_length(read_obj) -> int:
    return len(read_obj.query_sequence) if read_obj.query_sequence else 0


def _result_has_posteriors(result: dict) -> bool:
    return result.get('posteriors') is not None


def _posterior_chrom(read_obj):
    return read_obj.reference_name


def _add_posterior_fiber_from_request(
    request: _PosteriorFiberAddRequest,
) -> bool:
    if request.posterior_writer is None or not _result_has_posteriors(request.result):
        return False
    chrom = _posterior_chrom(request.read_obj)
    if not chrom:
        return False
    request.posterior_writer.add_fiber(
        chrom,
        posterior_fiber_data(
            request.read_obj,
            request.result,
            _posterior_ref_positions(request.read_obj),
        ),
    )
    return True


def _add_posterior_fiber_if_available(posterior_writer, read_obj, result: dict) -> bool:
    return _add_posterior_fiber_from_request(
        _PosteriorFiberAddRequest(
            posterior_writer=posterior_writer,
            read_obj=read_obj,
            result=result,
        )
    )


def _pop_inflight_chunk(inflight):
    submitted = inflight.popleft()
    worker_result = coerce_worker_chunk_result(submitted.future.result())
    return _InflightChunk(
        results=worker_result.results,
        worker_failures=worker_result.read_failures,
        read_objs=submitted.read_objs,
        items=submitted.items,
        skip_flags=submitted.skip_flags,
    )


def _drain_chunk_in_order(
    chunk_read_objs,
    chunk_items,
    chunk_skip_flags,
    results,
    outbam,
    counters,
    record_result,
) -> None:
    _drain_chunk_in_order_from_request(
        _DrainChunkInOrderRequest(
            chunk_read_objs=chunk_read_objs,
            chunk_items=chunk_items,
            chunk_skip_flags=chunk_skip_flags,
            results=results,
            outbam=outbam,
            counters=counters,
            record_result=record_result,
        )
    )


def _drain_chunk_in_order_from_request(
    request: _DrainChunkInOrderRequest,
) -> None:
    result_iter = iter(request.results)
    item_iter = iter(request.chunk_items)

    for read_obj, is_skipped in zip(
        request.chunk_read_objs,
        request.chunk_skip_flags,
    ):
        if is_skipped:
            _write_passthrough(request.outbam, read_obj, request.counters)
            continue

        next(item_iter)
        request.record_result(read_obj, next(result_iter))
        _write_passthrough(request.outbam, read_obj, request.counters)


def _drain_oldest_with_recorder_from_request(
    request: _DrainOldestWithRecorderRequest,
) -> None:
    chunk = _pop_inflight_chunk(request.inflight)
    _record_worker_failures(request.counters, chunk.worker_failures)
    _drain_chunk_in_order_from_request(
        _DrainChunkInOrderRequest(
            chunk_read_objs=chunk.read_objs,
            chunk_items=chunk.items,
            chunk_skip_flags=chunk.skip_flags,
            results=chunk.results,
            outbam=request.outbam,
            counters=request.counters,
            record_result=request.record_result,
        )
    )


def _drain_oldest_with_recorder(inflight, outbam, counters, record_result) -> None:
    _drain_oldest_with_recorder_from_request(
        _DrainOldestWithRecorderRequest(
            inflight=inflight,
            outbam=outbam,
            counters=counters,
            record_result=record_result,
        )
    )


def _record_apply_result(
    read_obj,
    result,
    with_scores,
    write_msps,
    posterior_writer,
    counters,
) -> None:
    if result is not None:
        set_legacy_apply_tags(read_obj, result, with_scores, write_msps)
        _record_reads_with_footprints(counters)
        _add_posterior_fiber_if_available(
            posterior_writer,
            read_obj,
            result,
        )
    else:
        _record_no_footprints(counters)


def _record_fused_result(
    read_obj,
    result,
    also_write_legacy,
    downstream_compat,
    counters,
) -> None:
    if result == CHIMERA_RESULT:
        _record_chimera(counters)
    elif result is not None:
        write_fused_recall_tags(
            read_obj,
            read_length=_fused_read_length(read_obj),
            result=result,
            also_write_legacy=also_write_legacy,
            downstream_compat=downstream_compat,
        )
        _record_reads_with_footprints(counters)
    else:
        _record_no_footprints(counters)


def _drain_oldest_chunk(
    inflight,
    outbam,
    with_scores,
    write_msps,
    posterior_writer,
    counters,
):
    """
    Block on the oldest in-flight apply chunk, apply tags, and write to BAM.

    Walks chunk_read_objs in original stream order so skipped reads are
    interleaved with processed reads at their correct positions. This preserves
    coordinate sort order when the input is coordinate sorted.
    """
    def record_result(read_obj, result):
        _record_apply_result(
            read_obj, result, with_scores, write_msps, posterior_writer,
            counters,
        )

    _drain_oldest_with_recorder(inflight, outbam, counters, record_result)


def _drain_oldest_fused_chunk(
    inflight,
    outbam,
    with_scores,
    also_write_legacy,
    downstream_compat,
    counters,
):
    """Drain one fused apply+recall chunk and write reads in input order."""
    def record_result(read_obj, result):
        _record_fused_result(
            read_obj, result, also_write_legacy, downstream_compat, counters,
        )

    _drain_oldest_with_recorder(inflight, outbam, counters, record_result)
