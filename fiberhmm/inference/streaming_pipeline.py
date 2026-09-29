"""Streaming BAM pipeline coordinators for apply and fused apply+recall.

File outputs are written to a hidden temporary sibling of the requested path
and published with an atomic rename only when the run succeeds. A crash, a
per-read worker failure rate above the shared policy, or (when requested) a
run that skipped nearly every record as unmapped therefore never leaves a
valid-looking BAM at the output path.
"""

from __future__ import annotations

import os
import sys
import time
from collections import deque
from concurrent.futures import Future, ProcessPoolExecutor
from typing import Optional, Set, Tuple

import pysam

from fiberhmm.inference.bam_output import _sort_and_index_bam, atomic_output
from fiberhmm.inference.engine import configure_daf_snp_mask, make_apply_payload
from fiberhmm.inference.mp_context import _MP_CONTEXT
from fiberhmm.inference.read_filters import (
    MostlyUnmappedError,
    ReadFilterConfig,
    mostly_unmapped_message,
    new_skip_counts,
    streaming_skip_reason,
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
from fiberhmm.inference.worker_results import enforce_worker_failure_policy
from fiberhmm.io.bam_header import (
    append_coord_marker,
    append_ma_types,
)

try:
    from fiberhmm.posteriors.hdf5_backend import PosteriorWriter
    HAS_POSTERIOR_WRITER = True
except ImportError:
    HAS_POSTERIOR_WRITER = False


def _output_target(output_path):
    """pysam target for a path, or a non-closing binary stdout for ``'-'``."""
    if output_path == '-':
        return os.fdopen(1, 'wb', closefd=False)
    return output_path


def _print_skip_reasons(skip_reasons, total_reads, skipped, log) -> None:
    if skipped <= 0:
        return
    print("  Skip reasons:", file=log)
    for reason, count in sorted(skip_reasons.items(), key=lambda x: -x[1]):
        if count > 0:
            pct = 100 * count / (total_reads + skipped)
            print(f"    {reason}: {count:,} ({pct:.1f}%)", file=log)


def _check_run_outcome(counters, total_reads, skip_reasons, skipped,
                       fail_on_mostly_unmapped, log) -> None:
    """Apply the worker-failure and mostly-unmapped policies (may raise)."""
    enforce_worker_failure_policy(
        counters.get('worker_failures', 0),
        total_reads,
        counters.get('failure_messages', ()),
        log=log,
    )
    message = mostly_unmapped_message(skip_reasons, total_reads + skipped)
    if message:
        print(f"\n  WARNING: {message}\n", file=log)
        if fail_on_mostly_unmapped:
            raise MostlyUnmappedError(message)


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
    nuc_recall_policy: str = "conservative",
    filter_chimeras: bool = True,
    chimera_min_seg: int = 5,
    chimera_purity: float = 0.8,
    phase_nrl: int = 0,
    nuc_profile_path: str = None,
    nuc_model_path: str = None,
    derived_tf_max_edge_ambiguity: int = None,
    pg_record: dict = None,
    ddda_mcg: bool = False,
    daf_snp_mask_path: str = None,
    fail_on_mostly_unmapped: bool = False,
):
    """Fused apply+recall streaming pipeline."""
    from fiberhmm.cli.provenance import output_header_with_provenance

    configure_daf_snp_mask(daf_snp_mask_path)
    ref_fasta = None
    pysam.set_verbosity(0)
    max_inflight = n_cores + 2
    start_time = time.time()
    counters = {
        'reads_with_footprints': 0,
        'no_footprints': 0,
        'worker_failures': 0,
        'failure_messages': [],
        'written': 0,
        'chimera': 0,
        'ddda_mcg_reads': 0,
        'ddda_mcg_spans': 0,
        'ddda_mcg_failures': 0,
    }

    total_reads = 0
    skipped = 0
    skip_reasons = new_skip_counts()
    filter_config = ReadFilterConfig(
        min_mapq=min_mapq,
        min_read_length=min_read_length,
        primary_only=primary_only,
        process_unmapped=process_unmapped,
        train_rids=train_rids,
        mode=mode,
        has_reference=bool(ref_fasta_path),
    )

    _log = sys.stderr if output_bam == '-' else sys.stdout

    with atomic_output(output_bam) as output_path:
        with pysam.AlignmentFile(input_bam, "rb", threads=io_threads,
                                 check_sq=False) as inbam:
            ma_types = [] if downstream_compat else ["nuc", "msp", "tf"]
            if ddda_mcg:
                ma_types.append("ddda_mcg")
            output_header = append_ma_types(
                output_header_with_provenance(inbam.header, pg_record),
                ma_types,
            )
            with pysam.AlignmentFile(_output_target(output_path), "wb",
                                     header=output_header,
                                     threads=io_threads) as outbam:
                if ref_fasta_path:
                    ref_fasta = pysam.FastaFile(ref_fasta_path)

                executor = ProcessPoolExecutor(
                    max_workers=n_cores,
                    mp_context=_MP_CONTEXT,
                    initializer=_init_fused_worker,
                    initargs=(model_path, recall_model_path, emission_uplift,
                              False, recall_nucs, split_min_llr,
                              split_min_opps, nuc_recall_policy,
                              filter_chimeras, chimera_min_seg, chimera_purity,
                              phase_nrl, nuc_profile_path,
                              derived_tf_max_edge_ambiguity, ddda_mcg,
                              nuc_model_path),
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

                def _drain():
                    _drain_oldest_fused_chunk(
                        inflight, outbam, with_scores,
                        also_write_legacy, downstream_compat, counters,
                    )

                def _submit():
                    return executor.submit(
                        _process_fused_payload_chunk_worker,
                        chunk_payloads, edge_trim, circular, mode,
                        context_size, msp_min_size, nuc_min_size,
                        with_scores, prob_threshold,
                        mode, context_size,
                        min_llr, min_opps, unify_threshold,
                    )

                try:
                    for read in inbam.fetch(until_eof=True):
                        skip_reason = streaming_skip_reason(read, filter_config)
                        if skip_reason:
                            _buffer_skip(read, skip_reason)
                            continue

                        payload = make_apply_payload(
                            read, mode=mode, ref_fasta=ref_fasta,
                            include_ddda_mcg=ddda_mcg,
                        )
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
                                _drain()
                            inflight.append((_submit(), chunk_read_objs,
                                             chunk_payloads, chunk_skip_flags))
                            chunk_payloads = []
                            chunk_read_objs = []
                            chunk_skip_flags = []

                            now = time.time()
                            elapsed = now - start_time
                            avg = total_reads / elapsed if elapsed > 0 else 0
                            dt = now - last_progress_time
                            inst = ((total_reads - last_progress_reads) / dt
                                    if dt > 0 else 0)
                            last_progress_reads = total_reads
                            last_progress_time = now
                            print(f"\r  Fused: {total_reads:,} | Skipped: {skipped:,} | "
                                  f"Inflight: {len(inflight)} | {inst:.0f} r/s "
                                  f"(avg {avg:.0f})", end='', file=_log)
                            _log.flush()

                    if chunk_read_objs:
                        if len(inflight) >= max_inflight:
                            _drain()
                        if chunk_payloads:
                            future = _submit()
                        else:
                            future = Future()
                            future.set_result([])
                        inflight.append((future, chunk_read_objs, chunk_payloads,
                                         chunk_skip_flags))

                    while inflight:
                        _drain()
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
        _print_skip_reasons(skip_reasons, total_reads, skipped, _log)
        if counters.get('chimera'):
            print(f"  DAF strand-swap chimeras filtered: {counters['chimera']:,}",
                  file=_log)
        if ddda_mcg:
            print(
                f"  DddA mCG: {counters['ddda_mcg_spans']:,} spans on "
                f"{counters['ddda_mcg_reads']:,} reads; "
                f"per-read failures={counters['ddda_mcg_failures']:,}",
                file=_log,
            )
        # Inside the atomic context: raising here discards the temporary BAM.
        _check_run_outcome(counters, total_reads, skip_reasons, skipped,
                           fail_on_mostly_unmapped, _log)

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
    fail_on_mostly_unmapped: bool = False,
) -> Tuple[int, int]:
    """Streaming producer-consumer pipeline for BAM processing."""
    if max_inflight is None:
        max_inflight = 2 * n_cores

    pysam.set_verbosity(0)

    total_reads = 0
    skip_reasons = new_skip_counts()
    filter_config = ReadFilterConfig(
        min_mapq=min_mapq,
        min_read_length=min_read_length,
        primary_only=primary_only,
        process_unmapped=process_unmapped,
        train_rids=train_rids,
        mode=mode,
    )

    counters = {
        'reads_with_footprints': 0,
        'no_footprints': 0,
        'worker_failures': 0,
        'failure_messages': [],
        'written': 0,
        'chimera': 0,
    }

    posterior_writer = None
    posterior_stats = None
    return_posteriors = False
    skipped = 0
    has_references = False

    _log = sys.stderr if output_bam == '-' else sys.stdout

    print(f"Processing BAM (streaming pipeline, {n_cores} workers, "
          f"chunk_size={chunk_size}, max_inflight={max_inflight})...", file=_log)
    _log.flush()

    start_time = time.time()

    with atomic_output(output_bam) as output_path:
        with pysam.AlignmentFile(input_bam, "rb", threads=io_threads,
                                 check_sq=False) as inbam:
            has_references = bool(inbam.references)
            with pysam.AlignmentFile(_output_target(output_path), "wb",
                                     header=append_coord_marker(inbam.header),
                                     threads=io_threads) as outbam:

                if output_posteriors:
                    if HAS_POSTERIOR_WRITER:
                        posterior_writer = PosteriorWriter(
                            output_posteriors, mode, context_size,
                            edge_trim, input_bam, batch_size=1000
                        )
                        return_posteriors = True
                        print(f"Posteriors will be written to: {output_posteriors}",
                              file=_log)
                    else:
                        print(
                            "WARNING: posterior_writer.py not found, skipping "
                            "posteriors export",
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
                last_progress_reads = 0
                last_progress_time = time.time()

                def _buffer_skip(rd, reason):
                    nonlocal skipped
                    chunk_read_objs.append(rd)
                    chunk_skip_flags.append(True)
                    skipped += 1
                    skip_reasons[reason] += 1

                def _drain():
                    _drain_oldest_chunk(
                        inflight, outbam, with_scores, write_msps,
                        posterior_writer, counters
                    )

                def _submit():
                    return executor.submit(
                        _process_payload_chunk_worker,
                        chunk_reads, edge_trim, circular, mode, context_size,
                        msp_min_size, nuc_min_size, with_scores,
                        return_posteriors, prob_threshold,
                    )

                try:
                    for read in inbam.fetch(until_eof=True):
                        skip_reason = streaming_skip_reason(read, filter_config)
                        if skip_reason:
                            _buffer_skip(read, skip_reason)
                            continue

                        payload = make_apply_payload(read, mode=mode, ref_fasta=None)
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
                                _drain()
                            inflight.append((_submit(), chunk_read_objs,
                                             chunk_reads, chunk_skip_flags))
                            chunk_reads = []
                            chunk_read_objs = []
                            chunk_skip_flags = []

                            now = time.time()
                            elapsed = now - start_time
                            avg_rate = total_reads / elapsed if elapsed > 0 else 0
                            dt = now - last_progress_time
                            inst_rate = ((total_reads - last_progress_reads) / dt
                                         if dt > 0 else 0)
                            last_progress_reads = total_reads
                            last_progress_time = now
                            print(f"\r  Processed: {total_reads:,} | "
                                  f"Skipped: {skipped:,} | "
                                  f"Inflight: {len(inflight)} | "
                                  f"{inst_rate:.0f} reads/s (avg {avg_rate:.0f})",
                                  end='', file=_log)
                            _log.flush()

                    if chunk_read_objs:
                        if len(inflight) >= max_inflight:
                            _drain()
                        if chunk_reads:
                            future = _submit()
                        else:
                            future = Future()
                            future.set_result([])
                        inflight.append((future, chunk_read_objs, chunk_reads,
                                         chunk_skip_flags))

                    while inflight:
                        _drain()

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
              f"With footprints: {reads_with_footprints:,} | {rate:.1f} reads/s",
              file=_log)
        _print_skip_reasons(skip_reasons, total_reads, skipped, _log)
        # Inside the atomic context: raising here discards the temporary BAM.
        _check_run_outcome(counters, total_reads, skip_reasons, skipped,
                           fail_on_mostly_unmapped, _log)

    # Unaligned / unmapped-processing output has no coordinates to index.
    if output_bam != '-' and not process_unmapped and has_references:
        _sort_and_index_bam(output_bam, threads=n_cores)

    if posterior_stats:
        n_fibers, file_size = posterior_stats
        print(f"Posteriors: {n_fibers:,} fibers -> {output_posteriors} "
              f"({file_size:.1f} MB)", file=_log)

    return total_reads, reads_with_footprints
