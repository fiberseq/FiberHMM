"""Streaming BAM pipeline coordinators for apply and fused apply+recall."""

from __future__ import annotations

import os
import sys
import time
from collections import deque
from concurrent.futures import Future, ProcessPoolExecutor
from typing import Optional, Set, Tuple

import pysam

from fiberhmm.inference.bam_output import _sort_and_index_bam
from fiberhmm.inference.engine import make_apply_payload
from fiberhmm.io.bam_header import append_coord_marker, maybe_append_pg
from fiberhmm.inference.mp_context import _MP_CONTEXT
from fiberhmm.inference.read_filters import ReadFilterConfig, streaming_skip_reason
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
    counters = {
        'reads_with_footprints': 0,
        'no_footprints': 0,
        'worker_failures': 0,
        'written': 0,
        'chimera': 0,
    }

    total_reads = 0
    skipped = 0
    skip_reasons = {
        'unmapped': 0, 'secondary_supplementary': 0, 'low_mapq': 0,
        'too_short': 0, 'training_excluded': 0, 'no_modifications': 0,
        'extraction_failed': 0,
    }
    filter_config = ReadFilterConfig(
        min_mapq=min_mapq,
        min_read_length=min_read_length,
        primary_only=primary_only,
        process_unmapped=process_unmapped,
        train_rids=train_rids,
    )

    _log = sys.stderr if output_bam == '-' else sys.stdout

    _output_target = output_bam
    if output_bam == '-':
        _output_target = os.fdopen(1, 'wb', closefd=False)

    with pysam.AlignmentFile(input_bam, "rb", threads=io_threads, check_sq=False) as inbam:
        with pysam.AlignmentFile(_output_target, "wb",
                                 header=maybe_append_pg(inbam.header, pg_record),
                                 threads=io_threads) as outbam:
            if ref_fasta_path:
                ref_fasta = pysam.FastaFile(ref_fasta_path)

            executor = ProcessPoolExecutor(
                max_workers=n_cores,
                mp_context=_MP_CONTEXT,
                initializer=_init_fused_worker,
                initargs=(model_path, recall_model_path, emission_uplift, False,
                          recall_nucs, split_min_llr, split_min_opps,
                          filter_chimeras, chimera_min_seg, chimera_purity,
                          phase_nrl),
            )

            inflight = deque()
            chunk_payloads = []
            chunk_read_objs = []
            chunk_skip_flags = []
            last_progress_reads = 0
            last_progress_time = time.time()

            def _buffer_skip(rd, reason):
                nonlocal skipped
                chunk_read_objs.append(rd)
                chunk_skip_flags.append(True)
                skipped += 1
                skip_reasons[reason] += 1

            try:
                for read in inbam.fetch(until_eof=True):
                    skip_reason = streaming_skip_reason(read, filter_config)
                    if skip_reason:
                        _buffer_skip(read, skip_reason)
                        continue

                    payload = make_apply_payload(read, mode=mode, ref_fasta=ref_fasta)
                    if payload is None:
                        _buffer_skip(read, 'no_modifications')
                        continue

                    chunk_payloads.append(payload)
                    chunk_read_objs.append(read)
                    chunk_skip_flags.append(False)
                    total_reads += 1

                    if max_reads and total_reads >= max_reads:
                        break

                    if len(chunk_read_objs) >= chunk_size:
                        if len(inflight) >= max_inflight:
                            _drain_oldest_fused_chunk(
                                inflight, outbam, with_scores,
                                also_write_legacy, downstream_compat, counters,
                            )
                        future = executor.submit(
                            _process_fused_payload_chunk_worker,
                            chunk_payloads, edge_trim, circular, mode,
                            context_size, msp_min_size, nuc_min_size,
                            with_scores, prob_threshold,
                            mode, context_size,
                            min_llr, min_opps, unify_threshold,
                        )
                        inflight.append((future, chunk_read_objs, chunk_payloads, chunk_skip_flags))
                        chunk_payloads = []
                        chunk_read_objs = []
                        chunk_skip_flags = []

                        now = time.time()
                        elapsed = now - start_time
                        avg = total_reads / elapsed if elapsed > 0 else 0
                        dt = now - last_progress_time
                        inst = (total_reads - last_progress_reads) / dt if dt > 0 else 0
                        last_progress_reads = total_reads
                        last_progress_time = now
                        print(f"\r  Fused: {total_reads:,} | Skipped: {skipped:,} | "
                              f"Inflight: {len(inflight)} | {inst:.0f} r/s (avg {avg:.0f})",
                              end='', file=_log)
                        _log.flush()

                if chunk_read_objs:
                    if len(inflight) >= max_inflight:
                        _drain_oldest_fused_chunk(
                            inflight, outbam, with_scores,
                            also_write_legacy, downstream_compat, counters,
                        )
                    if chunk_payloads:
                        future = executor.submit(
                            _process_fused_payload_chunk_worker,
                            chunk_payloads, edge_trim, circular, mode,
                            context_size, msp_min_size, nuc_min_size,
                            with_scores, prob_threshold,
                            mode, context_size,
                            min_llr, min_opps, unify_threshold,
                        )
                    else:
                        future = Future()
                        future.set_result([])
                    inflight.append((future, chunk_read_objs, chunk_payloads, chunk_skip_flags))

                while inflight:
                    _drain_oldest_fused_chunk(
                        inflight, outbam, with_scores,
                        also_write_legacy, downstream_compat, counters,
                    )
            finally:
                try:
                    executor.shutdown(wait=True)
                finally:
                    if ref_fasta is not None:
                        ref_fasta.close()
                        ref_fasta = None

    elapsed = time.time() - start_time
    rate = total_reads / elapsed if elapsed > 0 else 0
    reads_with_fp = counters['reads_with_footprints']
    print(f"\r  Fused: {total_reads:,} | Skipped: {skipped:,} | "
          f"With footprints: {reads_with_fp:,} | {rate:.1f} r/s", file=_log)
    if counters['worker_failures']:
        print(
            f"  Worker read failures: {counters['worker_failures']:,} "
            f"(passed through unchanged)",
            file=_log,
        )
    if skipped > 0:
        print("  Skip reasons:", file=_log)
        for reason, count in sorted(skip_reasons.items(), key=lambda x: -x[1]):
            if count > 0:
                pct = 100 * count / (total_reads + skipped)
                print(f"    {reason}: {count:,} ({pct:.1f}%)", file=_log)
    if counters.get('chimera'):
        print(f"  DAF strand-swap chimeras filtered: {counters['chimera']:,}",
              file=_log)

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
    filter_config = ReadFilterConfig(
        min_mapq=min_mapq,
        min_read_length=min_read_length,
        primary_only=primary_only,
        process_unmapped=process_unmapped,
        train_rids=train_rids,
    )

    counters = {
        'reads_with_footprints': 0,
        'no_footprints': 0,
        'worker_failures': 0,
        'written': 0,
        'chimera': 0,
    }

    posterior_writer = None
    posterior_stats = None
    return_posteriors = False

    _log = sys.stderr if output_bam == '-' else sys.stdout

    print(f"Processing BAM (streaming pipeline, {n_cores} workers, "
          f"chunk_size={chunk_size}, max_inflight={max_inflight})...", file=_log)
    _log.flush()

    start_time = time.time()

    _output_target = output_bam
    if output_bam == '-':
        _output_target = os.fdopen(1, 'wb', closefd=False)

    with pysam.AlignmentFile(input_bam, "rb", threads=io_threads, check_sq=False) as inbam:
        with pysam.AlignmentFile(_output_target, "wb",
                                 header=append_coord_marker(inbam.header),
                                 threads=io_threads) as outbam:

            if output_posteriors:
                if HAS_POSTERIOR_WRITER:
                    posterior_writer = PosteriorWriter(
                        output_posteriors, mode, context_size,
                        edge_trim, input_bam, batch_size=1000
                    )
                    return_posteriors = True
                    print(f"Posteriors will be written to: {output_posteriors}", file=_log)
                else:
                    print(
                        "WARNING: posterior_writer.py not found, skipping posteriors export",
                        file=_log,
                    )

            executor = ProcessPoolExecutor(
                max_workers=n_cores,
                mp_context=_MP_CONTEXT,
                initializer=_init_bam_worker,
                initargs=(model_path, debug_timing)
            )

            inflight = deque()
            chunk_reads = []
            chunk_read_objs = []
            chunk_skip_flags = []
            skipped = 0
            last_progress_reads = 0
            last_progress_time = time.time()

            def _buffer_skip(rd, reason):
                nonlocal skipped
                chunk_read_objs.append(rd)
                chunk_skip_flags.append(True)
                skipped += 1
                skip_reasons[reason] += 1

            try:
                for read in inbam.fetch(until_eof=True):
                    skip_reason = streaming_skip_reason(read, filter_config)
                    if skip_reason:
                        _buffer_skip(read, skip_reason)
                        continue

                    payload = make_apply_payload(read, mode=mode, ref_fasta=ref_fasta)
                    if payload is None:
                        _buffer_skip(read, 'no_modifications')
                        continue

                    chunk_reads.append(payload)
                    chunk_read_objs.append(read)
                    chunk_skip_flags.append(False)
                    total_reads += 1

                    if max_reads and total_reads >= max_reads:
                        break

                    if len(chunk_read_objs) >= chunk_size:
                        if len(inflight) >= max_inflight:
                            _drain_oldest_chunk(
                                inflight, outbam, with_scores, write_msps,
                                posterior_writer, counters
                            )

                        future = executor.submit(
                            _process_payload_chunk_worker,
                            chunk_reads, edge_trim, circular, mode, context_size,
                            msp_min_size, nuc_min_size, with_scores, return_posteriors,
                            prob_threshold,
                        )
                        inflight.append((future, chunk_read_objs, chunk_reads, chunk_skip_flags))
                        chunk_reads = []
                        chunk_read_objs = []
                        chunk_skip_flags = []

                        now = time.time()
                        elapsed = now - start_time
                        avg_rate = total_reads / elapsed if elapsed > 0 else 0
                        dt = now - last_progress_time
                        inst_rate = (total_reads - last_progress_reads) / dt if dt > 0 else 0
                        last_progress_reads = total_reads
                        last_progress_time = now
                        print(f"\r  Processed: {total_reads:,} | "
                              f"Skipped: {skipped:,} | "
                              f"Inflight: {len(inflight)} | "
                              f"{inst_rate:.0f} reads/s (avg {avg_rate:.0f})", end='', file=_log)
                        _log.flush()

                if chunk_read_objs:
                    if len(inflight) >= max_inflight:
                        _drain_oldest_chunk(
                            inflight, outbam, with_scores, write_msps,
                            posterior_writer, counters
                        )

                    if chunk_reads:
                        future = executor.submit(
                            _process_payload_chunk_worker,
                            chunk_reads, edge_trim, circular, mode, context_size,
                            msp_min_size, nuc_min_size, with_scores, return_posteriors,
                            prob_threshold,
                        )
                    else:
                        future = Future()
                        future.set_result([])
                    inflight.append((future, chunk_read_objs, chunk_reads, chunk_skip_flags))

                while inflight:
                    _drain_oldest_chunk(
                        inflight, outbam, with_scores, write_msps,
                        posterior_writer, counters
                    )

            finally:
                try:
                    executor.shutdown(wait=True)
                finally:
                    if posterior_writer:
                        posterior_stats = posterior_writer.close()
                        posterior_writer = None

    elapsed = time.time() - start_time
    rate = total_reads / elapsed if elapsed > 0 else 0
    reads_with_footprints = counters['reads_with_footprints']
    print(f"\r  Processed: {total_reads:,} | Skipped: {skipped:,} | "
          f"With footprints: {reads_with_footprints:,} | {rate:.1f} reads/s", file=_log)
    if counters['worker_failures']:
        print(
            f"  Worker read failures: {counters['worker_failures']:,} "
            f"(passed through unchanged)",
            file=_log,
        )

    if skipped > 0:
        print("  Skip reasons:", file=_log)
        for reason, count in sorted(skip_reasons.items(), key=lambda x: -x[1]):
            if count > 0:
                pct = 100 * count / (total_reads + skipped)
                print(f"    {reason}: {count:,} ({pct:.1f}%)", file=_log)

    if output_bam != '-' and not process_unmapped:
        _sort_and_index_bam(output_bam, threads=n_cores)

    if posterior_stats:
        n_fibers, file_size = posterior_stats
        print(f"Posteriors: {n_fibers:,} fibers -> {output_posteriors} ({file_size:.1f} MB)", file=_log)

    return total_reads, reads_with_footprints
