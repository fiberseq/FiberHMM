"""Order-preserving drain helpers for streaming inference pipelines."""

from __future__ import annotations

import numpy as np

from fiberhmm.inference.streaming_workers import CHIMERA_RESULT
from fiberhmm.inference.tagging import set_legacy_apply_tags, write_fused_recall_tags
from fiberhmm.inference.worker_results import coerce_worker_chunk_result

try:
    from fiberhmm.posteriors.hdf5_backend import get_ref_positions_from_read
    HAS_POSTERIOR_WRITER = True
except ImportError:
    HAS_POSTERIOR_WRITER = False


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
    future, chunk_read_objs, chunk_reads, chunk_skip_flags = inflight.popleft()
    results, worker_failures = coerce_worker_chunk_result(future.result())
    if worker_failures:
        counters['worker_failures'] = counters.get('worker_failures', 0) + worker_failures
    result_iter = iter(results)
    fiber_iter = iter(chunk_reads)

    for read_obj, is_skipped in zip(chunk_read_objs, chunk_skip_flags):
        if is_skipped:
            outbam.write(read_obj)
            counters['written'] += 1
            continue

        next(fiber_iter)
        result = next(result_iter)
        if result is not None:
            set_legacy_apply_tags(read_obj, result, with_scores, write_msps)
            counters['reads_with_footprints'] += 1

            if posterior_writer and result.get('posteriors') is not None:
                chrom = read_obj.reference_name
                if chrom:
                    try:
                        ref_positions = (
                            get_ref_positions_from_read(read_obj)
                            if HAS_POSTERIOR_WRITER
                            else np.array([], dtype=np.int32)
                        )
                    except Exception:
                        ref_positions = np.array([], dtype=np.int32)
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
        else:
            counters['no_footprints'] += 1

        outbam.write(read_obj)
        counters['written'] += 1


def _drain_oldest_fused_chunk(
    inflight,
    outbam,
    with_scores,
    also_write_legacy,
    downstream_compat,
    counters,
):
    """Drain one fused apply+recall chunk and write reads in input order."""
    future, chunk_read_objs, chunk_payloads, chunk_skip_flags = inflight.popleft()
    results, worker_failures = coerce_worker_chunk_result(future.result())
    if worker_failures:
        counters['worker_failures'] = counters.get('worker_failures', 0) + worker_failures
    result_iter = iter(results)
    payload_iter = iter(chunk_payloads)

    for read_obj, is_skipped in zip(chunk_read_objs, chunk_skip_flags):
        if is_skipped:
            outbam.write(read_obj)
            counters['written'] += 1
            continue

        next(payload_iter)
        result = next(result_iter)
        if result == CHIMERA_RESULT:
            counters['chimera'] = counters.get('chimera', 0) + 1
        elif result is not None:
            write_fused_recall_tags(
                read_obj,
                read_length=len(read_obj.query_sequence) if read_obj.query_sequence else 0,
                result=result,
                also_write_legacy=also_write_legacy,
                downstream_compat=downstream_compat,
            )
            spans = result.get('ddda_mcg_spans')
            if spans:
                counters['ddda_mcg_reads'] = counters.get('ddda_mcg_reads', 0) + 1
                counters['ddda_mcg_spans'] = counters.get('ddda_mcg_spans', 0) + len(spans)
            if result.get('ddda_mcg_failed'):
                counters['ddda_mcg_failures'] = counters.get('ddda_mcg_failures', 0) + 1
            counters['reads_with_footprints'] += 1
        else:
            counters['no_footprints'] += 1

        outbam.write(read_obj)
        counters['written'] += 1
