"""
Correctness tests for the streaming producer-consumer pipeline.
"""

import io
import sys
from collections import deque
from dataclasses import replace

import pysam
import pytest

from fiberhmm.inference import streaming_pipeline
from fiberhmm.inference.parallel import process_bam_for_footprints
from fiberhmm.inference.streaming_pipeline import (
    _apply_drain_chunk_factory,
    _apply_worker_args,
    _apply_worker_args_from_request,
    _ApplyStreamingFinalizeRequest,
    _buffer_processable_read,
    _buffer_processable_read_from_request,
    _buffer_skipped_read,
    _buffer_skipped_read_from_request,
    _buffer_streaming_read,
    _drain_all_streaming_chunks,
    _drain_if_inflight_full,
    _finalize_apply_streaming_pipeline,
    _flush_streaming_chunk,
    _fused_drain_chunk_factory,
    _fused_worker_args,
    _fused_worker_args_from_request,
    _FusedStreamingFinalizeRequest,
    _new_apply_streaming_executor,
    _new_fused_streaming_executor,
    _new_streaming_chunk_buffers,
    _open_streaming_posterior_writer,
    _open_streaming_posterior_writer_from_request,
    _print_fused_chimera_summary,
    _print_streaming_completion_summary,
    _print_streaming_posterior_summary,
    _print_streaming_progress,
    _ProcessableReadBufferRequest,
    _should_sort_streaming_output,
    _SkippedReadBufferRequest,
    _stream_reads_to_workers,
    _streaming_completion_message,
    _streaming_filter_config,
    _streaming_log_for_output,
    _streaming_output_target,
    _streaming_progress_message,
    _streaming_progress_rates,
    _streaming_rate,
    _StreamingApplyWorkerArgs,
    _StreamingChunkBuffers,
    _StreamingChunkFlushRequest,
    _StreamingChunkSubmitRequest,
    _StreamingFlushAndProgressRequest,
    _StreamingFlushProgress,
    _StreamingFusedWorkerArgs,
    _StreamingPayloadResult,
    _StreamingPosteriorStats,
    _StreamingPosteriorWriter,
    _StreamingPosteriorWriterOpenRequest,
    _StreamingProgressCheckpoint,
    _StreamingProgressRates,
    _StreamingReadCounts,
    _StreamingReadDelta,
    _StreamingWorkerCommonArgs,
    _StreamingWorkerLoopRequest,
    _StreamReadsToWorkersRequest,
    _submit_streaming_chunk,
    _submit_streaming_chunk_from_request,
    _worker_common_args,
    _worker_common_args_from_request,
)


class _FakeAlignmentFile:
    def __init__(self, *args, **kwargs):
        self.header = {"HD": {"SO": "coordinate"}}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def fetch(self, **kwargs):
        return []


def test_apply_worker_args_match_payload_worker_contract():
    common_args = _StreamingWorkerCommonArgs(
        edge_trim=11,
        circular=True,
        mode="deam",
        context_size=5,
        msp_min_size=61,
        nuc_min_size=86,
        with_scores=True,
    )
    assert _worker_common_args_from_request(common_args) == (
        11, True, "deam", 5, 61, 86, True,
    )
    assert _worker_common_args(
        11, True, "deam", 5, 61, 86, True,
    ) == (
        11, True, "deam", 5, 61, 86, True,
    )
    assert _apply_worker_args_from_request(
        _StreamingApplyWorkerArgs(
            common=common_args,
            return_posteriors=False,
            prob_threshold=127,
        )
    ) == (
        11, True, "deam", 5, 61, 86, True, False, 127,
    )
    assert _apply_worker_args(
        11, True, "deam", 5, 61, 86, True, False, 127
    ) == (
        11, True, "deam", 5, 61, 86, True, False, 127
    )


def test_fused_worker_args_include_recall_contract():
    common_args = _StreamingWorkerCommonArgs(
        edge_trim=12,
        circular=False,
        mode="pacbio-fiber",
        context_size=7,
        msp_min_size=62,
        nuc_min_size=87,
        with_scores=False,
    )
    assert _fused_worker_args_from_request(
        _StreamingFusedWorkerArgs(
            common=common_args,
            prob_threshold=126,
            min_llr=4.5,
            min_opps=6,
            unify_threshold=91,
        )
    ) == (
        12, False, "pacbio-fiber", 7, 62, 87, False, 126,
        "pacbio-fiber", 7, 4.5, 6, 91,
    )
    assert _fused_worker_args(
        12, False, "pacbio-fiber", 7, 62, 87, False, 126, 4.5, 6, 91
    ) == (
        12, False, "pacbio-fiber", 7, 62, 87, False, 126,
        "pacbio-fiber", 7, 4.5, 6, 91,
    )


def test_streaming_progress_rates_handle_elapsed_and_zero_dt():
    assert _streaming_progress_rates(
        total_reads=120,
        last_progress_reads=20,
        start_time=10.0,
        last_progress_time=20.0,
        now=30.0,
    ) == _StreamingProgressRates(instant=10.0, average=6.0)

    assert _streaming_progress_rates(
        total_reads=120,
        last_progress_reads=20,
        start_time=30.0,
        last_progress_time=30.0,
        now=30.0,
    ) == _StreamingProgressRates(instant=0, average=0)


def test_streaming_progress_message_formats_counts_and_rates():
    assert _streaming_progress_message(
        label="Processed",
        total_reads=1234,
        skipped=56,
        inflight_count=3,
        inst_rate=78.9,
        avg_rate=12.3,
        rate_unit="reads/s",
    ) == "\r  Processed: 1,234 | Skipped: 56 | Inflight: 3 | 79 reads/s (avg 12)"


def test_print_streaming_progress_writes_message_and_returns_checkpoint():
    log = io.StringIO()

    checkpoint = _print_streaming_progress(
        log,
        label="Processed",
        total_reads=120,
        skipped=5,
        inflight_count=2,
        last_progress_reads=20,
        start_time=10.0,
        last_progress_time=20.0,
        now=30.0,
        rate_unit="reads/s",
    )

    assert checkpoint == _StreamingProgressCheckpoint(reads=120, time=30.0)
    assert log.getvalue() == (
        "\r  Processed: 120 | Skipped: 5 | Inflight: 2 | "
        "10 reads/s (avg 6)"
    )


def test_streaming_rate_handles_zero_elapsed():
    assert _streaming_rate(120, 20.0) == 6.0
    assert _streaming_rate(120, 0.0) == 0


def test_streaming_completion_message_formats_final_counts():
    assert _streaming_completion_message(
        label="Fused",
        total_reads=1234,
        skipped=56,
        reads_with_footprints=78,
        rate=9.87,
        rate_unit="r/s",
    ) == "\r  Fused: 1,234 | Skipped: 56 | With footprints: 78 | 9.9 r/s"


def test_print_streaming_completion_summary_reports_failures_and_skips():
    log = io.StringIO()
    counters = {
        "reads_with_footprints": 7,
        "worker_failures": 2,
    }

    reads_with_footprints = _print_streaming_completion_summary(
        label="Processed",
        total_reads=10,
        skipped=3,
        counters=counters,
        elapsed=2.0,
        rate_unit="reads/s",
        skip_reasons={"low_mapq": 3},
        log=log,
    )

    assert reads_with_footprints == 7
    assert log.getvalue() == (
        "\r  Processed: 10 | Skipped: 3 | With footprints: 7 | 5.0 reads/s\n"
        "  Worker read failures: 2 (passed through unchanged)\n"
        "  Skip reasons:\n"
        "    low_mapq: 3 (23.1%)\n"
    )


def test_print_streaming_posterior_summary_handles_empty_and_values():
    log = io.StringIO()

    _print_streaming_posterior_summary(None, "out.h5", log)
    assert log.getvalue() == ""

    _print_streaming_posterior_summary(
        _StreamingPosteriorStats(1234, 5.678),
        "out.h5",
        log,
    )
    assert log.getvalue() == (
        "Posteriors: 1,234 fibers -> out.h5 (5.7 MB)\n"
    )


def test_should_sort_streaming_output_skips_stdout_and_unmapped_passthrough():
    assert _should_sort_streaming_output("out.bam", process_unmapped=False) is True
    assert _should_sort_streaming_output("-", process_unmapped=False) is False
    assert _should_sort_streaming_output("out.bam", process_unmapped=True) is False


def test_finalize_apply_streaming_pipeline_summarizes_sorts_and_reports_posteriors(
    monkeypatch,
):
    calls = []
    log = io.StringIO()

    monkeypatch.setattr(streaming_pipeline.time, "time", lambda: 15.0)
    monkeypatch.setattr(
        streaming_pipeline,
        "_print_streaming_completion_summary",
        lambda *args: calls.append(("summary", args)) or 7,
    )
    monkeypatch.setattr(
        streaming_pipeline,
        "_sort_and_index_bam",
        lambda output_bam, threads: calls.append(("sort", output_bam, threads)),
    )
    monkeypatch.setattr(
        streaming_pipeline,
        "_print_streaming_posterior_summary",
        lambda stats, path, got_log: calls.append(("posteriors", stats, path, got_log)),
    )

    request = _ApplyStreamingFinalizeRequest(
        total_reads=10,
        skipped=2,
        counters={"reads_with_footprints": 7},
        start_time=5.0,
        skip_reasons={"low_mapq": 2},
        output_bam="out.bam",
        process_unmapped=False,
        n_cores=3,
        posterior_stats=(4, 1.5),
        output_posteriors="post.h5",
        log=log,
    )

    assert streaming_pipeline._finalize_apply_streaming_pipeline_from_request(
        request,
    ) == 7

    assert calls == [
        (
            "summary",
            (
                "Processed",
                10,
                2,
                {"reads_with_footprints": 7},
                10.0,
                "reads/s",
                {"low_mapq": 2},
                log,
            ),
        ),
        ("sort", "out.bam", 3),
        ("posteriors", (4, 1.5), "post.h5", log),
    ]


def test_finalize_apply_streaming_pipeline_skips_sort_for_stdout_or_unmapped(
    monkeypatch,
):
    calls = []

    monkeypatch.setattr(
        streaming_pipeline,
        "_print_streaming_completion_summary",
        lambda *args: 0,
    )
    monkeypatch.setattr(
        streaming_pipeline,
        "_sort_and_index_bam",
        lambda *args: calls.append(("sort", args)),
    )
    monkeypatch.setattr(
        streaming_pipeline,
        "_print_streaming_posterior_summary",
        lambda *args: calls.append(("posteriors", args)),
    )

    for output_bam, process_unmapped in [("-", False), ("out.bam", True)]:
        _finalize_apply_streaming_pipeline(
            total_reads=0,
            skipped=0,
            counters={"reads_with_footprints": 0},
            start_time=0.0,
            skip_reasons={},
            output_bam=output_bam,
            process_unmapped=process_unmapped,
            n_cores=1,
            posterior_stats=None,
            output_posteriors=None,
            log=io.StringIO(),
        )

    assert [call[0] for call in calls] == ["posteriors", "posteriors"]


def test_print_fused_chimera_summary_only_reports_nonzero_counts():
    log = io.StringIO()

    _print_fused_chimera_summary({}, log)
    _print_fused_chimera_summary({"chimera": 0}, log)
    assert log.getvalue() == ""

    _print_fused_chimera_summary({"chimera": 1234}, log)
    assert log.getvalue() == (
        "  DAF strand-swap chimeras filtered: 1,234\n"
    )


def test_finalize_fused_streaming_pipeline_summarizes_and_reports_chimeras(
    monkeypatch,
):
    calls = []
    log = io.StringIO()

    monkeypatch.setattr(streaming_pipeline.time, "time", lambda: 15.0)
    monkeypatch.setattr(
        streaming_pipeline,
        "_print_streaming_completion_summary",
        lambda *args: calls.append(("summary", args)) or 5,
    )
    monkeypatch.setattr(
        streaming_pipeline,
        "_print_fused_chimera_summary",
        lambda counters, got_log: calls.append(("chimera", counters, got_log)),
    )

    request = _FusedStreamingFinalizeRequest(
        total_reads=10,
        skipped=2,
        counters={"reads_with_footprints": 5, "chimera": 3},
        start_time=5.0,
        skip_reasons={"low_mapq": 2},
        log=log,
    )

    assert streaming_pipeline._finalize_fused_streaming_pipeline_from_request(
        request,
    ) == 5

    assert calls == [
        (
            "summary",
            (
                "Fused",
                10,
                2,
                {"reads_with_footprints": 5, "chimera": 3},
                10.0,
                "r/s",
                {"low_mapq": 2},
                log,
            ),
        ),
        ("chimera", {"reads_with_footprints": 5, "chimera": 3}, log),
    ]


def test_streaming_filter_config_captures_read_filter_settings():
    config = _streaming_filter_config(
        min_mapq=20,
        min_read_length=50,
        primary_only=True,
        process_unmapped=False,
        train_rids={"read1"},
    )

    assert config.min_mapq == 20
    assert config.min_read_length == 50
    assert config.primary_only is True
    assert config.process_unmapped is False
    assert config.train_rids == {"read1"}


def test_apply_streaming_request_config_helpers_use_request_values():
    request = streaming_pipeline._ApplyStreamingPipelineRequest(
        input_bam="in.bam",
        output_bam="out.bam",
        model_path="model.json",
        train_rids={"read1"},
        edge_trim=0,
        circular=False,
        mode="daf",
        context_size=5,
        msp_min_size=10,
        nuc_min_size=20,
        min_mapq=30,
        prob_threshold=40,
        min_read_length=50,
        with_scores=True,
        n_cores=3,
        chunk_size=100,
        max_inflight=None,
        io_threads=2,
        primary_only=True,
        output_posteriors=None,
        write_msps=False,
        max_reads=99,
        debug_timing=True,
        process_unmapped=False,
    )

    assert streaming_pipeline._apply_streaming_max_inflight(request) == 6

    explicit = replace(request, max_inflight=7)
    assert streaming_pipeline._apply_streaming_max_inflight(explicit) == 7

    config = streaming_pipeline._streaming_filter_config_from_request(request)
    assert config.min_mapq == 30
    assert config.min_read_length == 50
    assert config.primary_only is True
    assert config.process_unmapped is False
    assert config.train_rids == {"read1"}


def test_fused_streaming_request_config_helpers_use_request_values():
    request = streaming_pipeline._FusedStreamingPipelineRequest(
        input_bam="in.bam",
        output_bam="out.bam",
        model_path="model.json",
        recall_model_path=None,
        train_rids={"read2"},
        edge_trim=0,
        circular=False,
        mode="pacbio-fiber",
        context_size=5,
        msp_min_size=10,
        nuc_min_size=20,
        min_mapq=31,
        prob_threshold=40,
        min_read_length=51,
        with_scores=True,
        min_llr=1.5,
        min_opps=2,
        unify_threshold=3,
        emission_uplift=1.0,
        also_write_legacy=True,
        downstream_compat=False,
        max_reads=99,
        n_cores=4,
        chunk_size=100,
        io_threads=2,
        process_unmapped=True,
        primary_only=False,
        ref_fasta_path="ref.fa",
        recall_nucs=True,
        split_min_llr=4.0,
        split_min_opps=3,
        filter_chimeras=True,
        chimera_min_seg=5,
        chimera_purity=0.8,
        phase_nrl=147,
        pg_record={"ID": "fiberhmm"},
    )

    assert streaming_pipeline._fused_streaming_max_inflight(request) == 6

    config = streaming_pipeline._streaming_filter_config_from_request(request)
    assert config.min_mapq == 31
    assert config.min_read_length == 51
    assert config.primary_only is False
    assert config.process_unmapped is True
    assert config.train_rids == {"read2"}


def test_buffer_skipped_read_keeps_chunk_lists_and_reasons_aligned():
    reads = []
    skip_flags = []
    skip_reasons = {"low_mapq": 0}
    read = object()
    request = _SkippedReadBufferRequest(
        chunk_read_objs=reads,
        chunk_skip_flags=skip_flags,
        skip_reasons=skip_reasons,
        read=read,
        reason="low_mapq",
    )

    added = _buffer_skipped_read_from_request(request)

    assert added == 1
    assert reads == [read]
    assert skip_flags == [True]
    assert skip_reasons == {"low_mapq": 1}

    wrapper_reads = []
    wrapper_skip_flags = []
    wrapper_skip_reasons = {"low_mapq": 0}
    assert _buffer_skipped_read(
        wrapper_reads,
        wrapper_skip_flags,
        wrapper_skip_reasons,
        read,
        "low_mapq",
    ) == 1
    assert wrapper_reads == [read]
    assert wrapper_skip_flags == [True]
    assert wrapper_skip_reasons == {"low_mapq": 1}


def test_buffer_processable_read_keeps_chunk_lists_aligned():
    payloads = []
    reads = []
    skip_flags = []
    read = object()
    payload = {"read": "read1"}
    request = _ProcessableReadBufferRequest(
        chunk_items=payloads,
        chunk_read_objs=reads,
        chunk_skip_flags=skip_flags,
        payload=payload,
        read=read,
    )

    added = _buffer_processable_read_from_request(request)

    assert added == 1
    assert payloads == [payload]
    assert reads == [read]
    assert skip_flags == [False]

    wrapper_payloads = []
    wrapper_reads = []
    wrapper_skip_flags = []
    assert _buffer_processable_read(
        wrapper_payloads,
        wrapper_reads,
        wrapper_skip_flags,
        payload,
        read,
    ) == 1
    assert wrapper_payloads == [payload]
    assert wrapper_reads == [read]
    assert wrapper_skip_flags == [False]


def test_buffer_streaming_read_adds_processable_payload(monkeypatch):
    read = object()
    filter_config = object()
    ref_fasta = object()
    payload = {"read": "read1"}
    calls = []

    def fake_payload_or_skip(read_arg, filter_arg, mode_arg, ref_arg):
        calls.append((read_arg, filter_arg, mode_arg, ref_arg))
        return _StreamingPayloadResult(payload=payload, skip_reason=None)

    monkeypatch.setattr(
        streaming_pipeline,
        "_streaming_payload_or_skip",
        fake_payload_or_skip,
    )

    chunk_items = []
    chunk_reads = []
    skip_flags = []
    delta = _buffer_streaming_read(
        read,
        filter_config,
        "deam",
        ref_fasta,
        chunk_items,
        chunk_reads,
        skip_flags,
        {},
    )

    assert delta == _StreamingReadDelta(processed=1, skipped=0)
    assert calls == [(read, filter_config, "deam", ref_fasta)]
    assert chunk_items == [payload]
    assert chunk_reads == [read]
    assert skip_flags == [False]


def test_buffer_streaming_read_tracks_skipped_read(monkeypatch):
    read = object()

    monkeypatch.setattr(
        streaming_pipeline,
        "_streaming_payload_or_skip",
        lambda read, filter_config, mode, ref_fasta: _StreamingPayloadResult(
            payload=None,
            skip_reason="low_mapq",
        ),
    )

    chunk_items = []
    chunk_reads = []
    skip_flags = []
    skip_reasons = {"low_mapq": 0}
    delta = _buffer_streaming_read(
        read,
        object(),
        "deam",
        None,
        chunk_items,
        chunk_reads,
        skip_flags,
        skip_reasons,
    )

    assert delta == _StreamingReadDelta(processed=0, skipped=1)
    assert chunk_items == []
    assert chunk_reads == [read]
    assert skip_flags == [True]
    assert skip_reasons == {"low_mapq": 1}


def test_new_streaming_chunk_buffers_returns_independent_lists():
    buffers = _new_streaming_chunk_buffers()
    assert buffers == _StreamingChunkBuffers(items=[], read_objs=[], skip_flags=[])

    buffers.items.append("payload")
    buffers.read_objs.append("read")
    buffers.skip_flags.append(False)

    assert _new_streaming_chunk_buffers() == _StreamingChunkBuffers(
        items=[], read_objs=[], skip_flags=[],
    )


def test_drain_if_inflight_full_only_drains_at_capacity():
    inflight = deque(["old"])
    drained = []

    _drain_if_inflight_full(inflight, max_inflight=2, drain_chunk=drained.append)
    assert drained == []

    _drain_if_inflight_full(
        inflight,
        max_inflight=1,
        drain_chunk=lambda: drained.append(inflight.popleft()),
    )
    assert drained == ["old"]
    assert list(inflight) == []


def test_submit_streaming_chunk_records_worker_or_empty_future():
    class Executor:
        def __init__(self):
            self.submitted = []

        def submit(self, worker_fn, *args):
            self.submitted.append((worker_fn, args))
            return "future"

    executor = Executor()
    inflight = deque()
    worker_fn = object()
    buffers = _StreamingChunkBuffers(
        items=["payload"],
        read_objs=["read"],
        skip_flags=[False],
    )

    _submit_streaming_chunk_from_request(
        _StreamingChunkSubmitRequest(
            inflight=inflight,
            executor=executor,
            worker_fn=worker_fn,
            buffers=buffers,
            worker_args=("arg",),
        )
    )

    assert executor.submitted == [(worker_fn, (buffers.items, "arg"))]
    assert list(inflight) == [
        streaming_pipeline._SubmittedChunk(
            future="future",
            read_objs=buffers.read_objs,
            items=buffers.items,
            skip_flags=buffers.skip_flags,
        )
    ]

    empty_inflight = deque()
    _submit_streaming_chunk(
        empty_inflight,
        executor,
        worker_fn,
        [],
        ["skipped"],
        [True],
        ("arg",),
    )

    assert executor.submitted == [(worker_fn, (buffers.items, "arg"))]
    assert len(empty_inflight) == 1
    submitted = empty_inflight[0]
    assert submitted.future.result() == []
    assert submitted.read_objs == ["skipped"]
    assert submitted.items == []
    assert submitted.skip_flags == [True]


def test_flush_streaming_chunk_drains_submits_and_returns_new_buffers():
    class Executor:
        def __init__(self):
            self.submitted = []

        def submit(self, worker_fn, *args):
            self.submitted.append((worker_fn, args))
            return "future"

    inflight = deque(["old"])
    drained = []
    executor = Executor()
    worker_fn = object()
    chunk_items = ["payload"]
    chunk_reads = ["read"]
    chunk_skip_flags = [False]

    request = _StreamingChunkFlushRequest(
        inflight=inflight,
        executor=executor,
        worker_fn=worker_fn,
        buffers=_StreamingChunkBuffers(
            items=chunk_items,
            read_objs=chunk_reads,
            skip_flags=chunk_skip_flags,
        ),
        worker_args=("arg",),
        max_inflight=1,
        drain_chunk=lambda: drained.append(inflight.popleft()),
    )

    new_buffers = streaming_pipeline._flush_streaming_chunk_from_request(request)

    assert drained == ["old"]
    assert executor.submitted == [(worker_fn, (chunk_items, "arg"))]
    assert list(inflight) == [
        streaming_pipeline._SubmittedChunk(
            future="future",
            read_objs=chunk_reads,
            items=chunk_items,
            skip_flags=chunk_skip_flags,
        )
    ]
    assert new_buffers == _StreamingChunkBuffers(items=[], read_objs=[], skip_flags=[])
    assert new_buffers.items is not chunk_items
    assert new_buffers.read_objs is not chunk_reads
    assert new_buffers.skip_flags is not chunk_skip_flags

    drained.clear()
    inflight = deque(["old"])
    assert _flush_streaming_chunk(
        inflight,
        executor,
        worker_fn,
        chunk_items,
        chunk_reads,
        chunk_skip_flags,
        ("arg",),
        max_inflight=1,
        drain_chunk=lambda: drained.append(inflight.popleft()),
    ) == _StreamingChunkBuffers(items=[], read_objs=[], skip_flags=[])
    assert drained == ["old"]


def test_drain_all_streaming_chunks_repeats_until_empty():
    inflight = deque(["a", "b", "c"])
    drained = []

    _drain_all_streaming_chunks(
        inflight, lambda: drained.append(inflight.popleft()),
    )

    assert drained == ["a", "b", "c"]
    assert list(inflight) == []


def test_flush_streaming_chunk_and_report_progress(monkeypatch):
    calls = []
    inflight = deque(["old"])
    executor = object()
    worker_fn = object()

    def fake_flush(*args):
        calls.append(("flush", args))
        inflight.append("new")
        return _StreamingChunkBuffers(items=[], read_objs=[], skip_flags=[])

    def fake_progress(*args):
        calls.append(("progress", args))
        return _StreamingProgressCheckpoint(reads=20, time=30.0)

    monkeypatch.setattr(
        streaming_pipeline, "_flush_streaming_chunk", fake_flush,
    )
    monkeypatch.setattr(
        streaming_pipeline, "_print_streaming_progress", fake_progress,
    )
    monkeypatch.setattr(streaming_pipeline.time, "time", lambda: 25.0)

    context = streaming_pipeline._StreamingFlushContext(
        inflight=inflight,
        executor=executor,
        worker_fn=worker_fn,
        worker_args=("arg",),
        max_inflight=4,
        drain_chunk=lambda: None,
        log=io.StringIO(),
        progress_label="Processed",
        start_time=1.0,
        rate_unit="reads/s",
    )

    request = _StreamingFlushAndProgressRequest(
        context=context,
        buffers=_StreamingChunkBuffers(
            items=["payload"],
            read_objs=["read"],
            skip_flags=[False],
        ),
        total_reads=20,
        skipped=3,
        last_progress_reads=10,
        last_progress_time=2.0,
    )

    assert streaming_pipeline._flush_streaming_chunk_and_report_progress_from_request(
        request,
    ) == _StreamingFlushProgress(
        buffers=_StreamingChunkBuffers(items=[], read_objs=[], skip_flags=[]),
        last_progress_reads=20,
        last_progress_time=30.0,
    )

    assert calls[0][0] == "flush"
    assert calls[0][1][:8] == (
        inflight,
        executor,
        worker_fn,
        ["payload"],
        ["read"],
        [False],
        ("arg",),
        4,
    )
    assert calls[1][0] == "progress"
    assert calls[1][1][1:10] == (
        "Processed",
        20,
        3,
        2,
        10,
        1.0,
        2.0,
        25.0,
        "reads/s",
    )


def test_stream_reads_to_workers_buffers_submits_and_reports_progress(monkeypatch):
    class Executor:
        def __init__(self):
            self.submitted = []

        def submit(self, worker_fn, *args):
            future = f"future-{len(self.submitted)}"
            self.submitted.append((worker_fn, args, future))
            return future

    def fake_buffer_read(
        read,
        filter_config,
        mode,
        ref_fasta,
        chunk_items,
        chunk_read_objs,
        chunk_skip_flags,
        skip_reasons,
    ):
        chunk_read_objs.append(read)
        if read == "skip":
            chunk_skip_flags.append(True)
            skip_reasons["low_mapq"] = skip_reasons.get("low_mapq", 0) + 1
            return _StreamingReadDelta(processed=0, skipped=1)
        chunk_items.append(f"payload-{read}")
        chunk_skip_flags.append(False)
        return _StreamingReadDelta(processed=1, skipped=0)

    progress_calls = []

    def fake_print_progress(*args):
        progress_calls.append(args)
        total_reads = args[2]
        now = args[8]
        return _StreamingProgressCheckpoint(reads=total_reads, time=now)

    monkeypatch.setattr(
        streaming_pipeline, "_buffer_streaming_read", fake_buffer_read,
    )
    monkeypatch.setattr(
        streaming_pipeline, "_print_streaming_progress", fake_print_progress,
    )
    inflight = deque()
    drained = []
    executor = Executor()
    worker_fn = object()
    skip_reasons = {}

    request = _StreamReadsToWorkersRequest(
        reads=["p1", "skip", "p2"],
        filter_config=object(),
        mode="daf",
        ref_fasta=object(),
        inflight=inflight,
        executor=executor,
        worker_fn=worker_fn,
        worker_args=("arg",),
        max_reads=None,
        chunk_size=2,
        max_inflight=10,
        drain_chunk=lambda: drained.append(inflight.popleft()),
        skip_reasons=skip_reasons,
        log=io.StringIO(),
        progress_label="Processed",
        rate_unit="reads/s",
        start_time=10.0,
    )

    counts = streaming_pipeline._stream_reads_to_workers_from_request(request)

    assert counts == _StreamingReadCounts(processed=2, skipped=1)
    assert skip_reasons == {"low_mapq": 1}
    assert executor.submitted == [
        (worker_fn, (["payload-p1", "payload-p2"], "arg"), "future-0")
    ]
    assert drained == [
        streaming_pipeline._SubmittedChunk(
            future="future-0",
            read_objs=["p1", "skip", "p2"],
            items=["payload-p1", "payload-p2"],
            skip_flags=[False, True, False],
        )
    ]
    assert list(inflight) == []
    assert progress_calls[0][1:5] == ("Processed", 2, 1, 1)
    assert progress_calls[0][9] == "reads/s"

    assert _stream_reads_to_workers(
        [],
        object(),
        "daf",
        object(),
        deque(),
        executor,
        worker_fn,
        ("arg",),
        max_reads=None,
        chunk_size=2,
        max_inflight=10,
        drain_chunk=lambda: None,
        skip_reasons={},
        log=io.StringIO(),
        progress_label="Processed",
        rate_unit="reads/s",
        start_time=10.0,
    ) == _StreamingReadCounts(processed=0, skipped=0)


def test_run_streaming_worker_loop_builds_inflight_and_shuts_down(monkeypatch):
    class Executor:
        def __init__(self):
            self.shutdown_calls = []

        def shutdown(self, **kwargs):
            self.shutdown_calls.append(kwargs)

    executor = Executor()
    drain_calls = []
    seen = {}

    def drain_chunk_factory(inflight):
        seen["inflight"] = inflight

        def drain_chunk():
            drain_calls.append(list(inflight))

        return drain_chunk

    def fake_stream_reads(
        reads,
        filter_config,
        mode,
        ref_fasta,
        inflight,
        got_executor,
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
    ):
        inflight.append("future")
        drain_chunk()
        return _StreamingReadCounts(processed=5, skipped=1)

    monkeypatch.setattr(
        streaming_pipeline, "_stream_reads_to_workers", fake_stream_reads,
    )

    request = _StreamingWorkerLoopRequest(
        reads=["r1"],
        filter_config=object(),
        mode="daf",
        ref_fasta=None,
        executor=executor,
        worker_fn=object(),
        worker_args=("arg",),
        max_reads=None,
        chunk_size=2,
        max_inflight=3,
        drain_chunk_factory=drain_chunk_factory,
        skip_reasons={},
        log=io.StringIO(),
        progress_label="Processed",
        rate_unit="reads/s",
        start_time=10.0,
    )

    assert streaming_pipeline._run_streaming_worker_loop_from_request(
        request,
    ) == _StreamingReadCounts(processed=5, skipped=1)

    assert isinstance(seen["inflight"], deque)
    assert drain_calls == [["future"]]
    assert executor.shutdown_calls == [{"wait": True}]


def test_streaming_log_for_output_uses_stderr_for_stdout_bam():
    assert _streaming_log_for_output("-") is sys.stderr
    assert _streaming_log_for_output("out.bam") is sys.stdout


def test_streaming_output_target_resolves_stdout_and_file(monkeypatch):
    sentinel = object()
    calls = []

    def fake_fdopen(fd, mode, closefd):
        calls.append((fd, mode, closefd))
        return sentinel

    monkeypatch.setattr(streaming_pipeline.os, "fdopen", fake_fdopen)

    assert _streaming_output_target("out.bam") == "out.bam"
    assert _streaming_output_target("-") is sentinel
    assert calls == [(1, "wb", False)]


def test_open_streaming_posterior_writer_enables_worker_posteriors(monkeypatch):
    created = []

    class FakePosteriorWriter:
        def __init__(self, *args, **kwargs):
            created.append((args, kwargs))

    monkeypatch.setattr(streaming_pipeline, "HAS_POSTERIOR_WRITER", True)
    monkeypatch.setattr(
        streaming_pipeline,
        "PosteriorWriter",
        FakePosteriorWriter,
        raising=False,
    )
    log = io.StringIO()

    posterior_output = _open_streaming_posterior_writer_from_request(
        _StreamingPosteriorWriterOpenRequest(
            output_posteriors="out.h5",
            mode="deam",
            context_size=4,
            edge_trim=10,
            input_bam="in.bam",
            log=log,
        )
    )

    assert isinstance(posterior_output.writer, FakePosteriorWriter)
    assert posterior_output.enabled is True
    assert created == [
        (("out.h5", "deam", 4, 10, "in.bam"), {"batch_size": 1000})
    ]
    assert log.getvalue() == "Posteriors will be written to: out.h5\n"


def test_open_streaming_posterior_writer_warns_when_backend_missing(monkeypatch):
    monkeypatch.setattr(streaming_pipeline, "HAS_POSTERIOR_WRITER", False)
    log = io.StringIO()

    posterior_output = _open_streaming_posterior_writer(
        "out.h5", "deam", 4, 10, "in.bam", log,
    )

    assert posterior_output == _StreamingPosteriorWriter(
        writer=None,
        enabled=False,
    )
    assert log.getvalue() == (
        "WARNING: posterior_writer.py not found, skipping posteriors export\n"
    )


def test_streaming_close_helpers_handle_optional_resources():
    class Resource:
        def __init__(self, close_result=None):
            self.close_result = close_result
            self.closed = 0

        def __bool__(self):
            return False

        def close(self):
            self.closed += 1
            return self.close_result

    writer = Resource(close_result=(12, 3.5))
    ref_fasta = Resource()

    assert streaming_pipeline._close_streaming_posterior_writer(None) is None
    assert streaming_pipeline._close_streaming_posterior_writer(
        writer,
    ) == _StreamingPosteriorStats(12, 3.5)
    assert _StreamingPosteriorStats(12, 3.5).as_tuple() == (12, 3.5)
    assert writer.closed == 1

    assert streaming_pipeline._close_ref_fasta(None) is None
    assert streaming_pipeline._close_ref_fasta(ref_fasta) is None
    assert ref_fasta.closed == 1


def test_new_apply_streaming_executor_configures_pool(monkeypatch):
    class FakeExecutor:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr(streaming_pipeline, "ProcessPoolExecutor", FakeExecutor)

    executor = _new_apply_streaming_executor(
        model_path="model.pkl",
        n_cores=3,
        debug_timing=True,
    )

    assert executor.kwargs == {
        "max_workers": 3,
        "mp_context": streaming_pipeline._MP_CONTEXT,
        "initializer": streaming_pipeline._init_bam_worker,
        "initargs": ("model.pkl", True),
    }


def test_apply_streaming_closes_posterior_writer_on_executor_setup_failure(
    monkeypatch,
):
    class Writer:
        def __init__(self):
            self.closed = 0

        def close(self):
            self.closed += 1
            return (0, 0.0)

    writer = Writer()
    monkeypatch.setattr(streaming_pipeline.pysam, "set_verbosity", lambda value: None)
    monkeypatch.setattr(streaming_pipeline.pysam, "AlignmentFile", _FakeAlignmentFile)
    monkeypatch.setattr(streaming_pipeline, "append_coord_marker", lambda header: header)
    monkeypatch.setattr(
        streaming_pipeline,
        "_open_streaming_posterior_writer",
        lambda *args: _StreamingPosteriorWriter(writer=writer, enabled=True),
    )
    monkeypatch.setattr(
        streaming_pipeline,
        "_new_apply_streaming_executor",
        lambda *args: (_ for _ in ()).throw(RuntimeError("executor failed")),
    )

    with pytest.raises(RuntimeError, match="executor failed"):
        streaming_pipeline._process_bam_streaming_pipeline(
            input_bam="in.bam",
            output_bam="out.bam",
            model_path="model.json",
            train_rids=set(),
            edge_trim=0,
            circular=False,
            mode="daf",
            context_size=5,
            msp_min_size=0,
            n_cores=1,
            output_posteriors="post.h5",
        )

    assert writer.closed == 1


def test_fused_streaming_closes_reference_on_executor_setup_failure(
    monkeypatch,
):
    class FakeFastaFile:
        def __init__(self, path):
            self.path = path
            self.closed = 0

        def close(self):
            self.closed += 1

    fasta = FakeFastaFile("ref.fa")
    monkeypatch.setattr(streaming_pipeline.pysam, "set_verbosity", lambda value: None)
    monkeypatch.setattr(streaming_pipeline.pysam, "AlignmentFile", _FakeAlignmentFile)
    monkeypatch.setattr(streaming_pipeline.pysam, "FastaFile", lambda path: fasta)
    monkeypatch.setattr(streaming_pipeline, "maybe_append_pg", lambda header, pg: header)
    monkeypatch.setattr(
        streaming_pipeline,
        "_new_fused_streaming_executor",
        lambda *args: (_ for _ in ()).throw(RuntimeError("executor failed")),
    )

    with pytest.raises(RuntimeError, match="executor failed"):
        streaming_pipeline._process_bam_streaming_pipeline_fused(
            input_bam="in.bam",
            output_bam="out.bam",
            model_path="model.json",
            recall_model_path=None,
            train_rids=set(),
            edge_trim=0,
            circular=False,
            mode="daf",
            context_size=5,
            msp_min_size=0,
            nuc_min_size=0,
            min_mapq=0,
            prob_threshold=0,
            min_read_length=0,
            with_scores=False,
            min_llr=0.0,
            min_opps=0,
            unify_threshold=0,
            emission_uplift=1.0,
            also_write_legacy=True,
            downstream_compat=True,
            max_reads=None,
            n_cores=1,
            chunk_size=1,
            io_threads=1,
            ref_fasta_path="ref.fa",
        )

    assert fasta.closed == 1


def test_apply_drain_chunk_factory_calls_apply_drain(monkeypatch):
    calls = []

    def fake_drain(
        inflight,
        outbam,
        with_scores,
        write_msps,
        posterior_writer,
        counters,
    ):
        calls.append(
            (inflight, outbam, with_scores, write_msps, posterior_writer, counters)
        )

    monkeypatch.setattr(streaming_pipeline, "_drain_oldest_chunk", fake_drain)

    inflight = deque(["chunk"])
    outbam = object()
    posterior_writer = object()
    counters = {}

    drain_chunk = _apply_drain_chunk_factory(
        outbam,
        with_scores=True,
        write_msps=False,
        posterior_writer=posterior_writer,
        counters=counters,
    )(inflight)
    drain_chunk()

    assert len(calls) == 1
    assert calls[0][0] is inflight
    assert calls[0][1:] == (
        outbam,
        True,
        False,
        posterior_writer,
        counters,
    )


def test_new_fused_streaming_executor_configures_pool(monkeypatch):
    class FakeExecutor:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr(streaming_pipeline, "ProcessPoolExecutor", FakeExecutor)

    executor = _new_fused_streaming_executor(
        model_path="apply.pkl",
        recall_model_path="recall.pkl",
        emission_uplift=1.5,
        recall_nucs=True,
        split_min_llr=4.0,
        split_min_opps=3,
        filter_chimeras=False,
        chimera_min_seg=5,
        chimera_purity=0.8,
        phase_nrl=147,
        n_cores=4,
    )

    assert executor.kwargs == {
        "max_workers": 4,
        "mp_context": streaming_pipeline._MP_CONTEXT,
        "initializer": streaming_pipeline._init_fused_worker,
        "initargs": (
            "apply.pkl",
            "recall.pkl",
            1.5,
            False,
            True,
            4.0,
            3,
            False,
            5,
            0.8,
            147,
        ),
    }


def test_fused_drain_chunk_factory_calls_fused_drain(monkeypatch):
    calls = []

    def fake_drain(
        inflight,
        outbam,
        with_scores,
        also_write_legacy,
        downstream_compat,
        counters,
    ):
        calls.append(
            (
                inflight,
                outbam,
                with_scores,
                also_write_legacy,
                downstream_compat,
                counters,
            )
        )

    monkeypatch.setattr(streaming_pipeline, "_drain_oldest_fused_chunk", fake_drain)

    inflight = deque(["chunk"])
    outbam = object()
    counters = {}

    drain_chunk = _fused_drain_chunk_factory(
        outbam,
        with_scores=True,
        also_write_legacy=False,
        downstream_compat=True,
        counters=counters,
    )(inflight)
    drain_chunk()

    assert len(calls) == 1
    assert calls[0][0] is inflight
    assert calls[0][1:] == (
        outbam,
        True,
        False,
        True,
        counters,
    )


def _count_bam_reads(bam_path):
    """Count total reads in a BAM file."""
    with pysam.AlignmentFile(bam_path, "rb") as bam:
        return sum(1 for _ in bam.fetch(until_eof=True))


def _read_tags_by_name(bam_path):
    """Read BAM and return dict of {read_name: {tag: value, ...}}."""
    result = {}
    with pysam.AlignmentFile(bam_path, "rb") as bam:
        for read in bam.fetch(until_eof=True):
            tags = {}
            for tag in ['ns', 'nl', 'as', 'al', 'nq', 'aq']:
                if read.has_tag(tag):
                    tags[tag] = list(read.get_tag(tag))
            result[read.query_name] = tags
    return result


def _read_names_in_order(bam_path):
    """Return list of read names in BAM order."""
    with pysam.AlignmentFile(bam_path, "rb") as bam:
        return [read.query_name for read in bam.fetch(until_eof=True)]


class TestStreamingBasic:
    """Basic correctness tests for the streaming pipeline."""

    def test_all_reads_preserved(self, synthetic_bam_small, benchmark_model_path, tmp_path):
        """Every read in input appears in output."""
        output = str(tmp_path / "out.bam")
        process_bam_for_footprints(
            input_bam=synthetic_bam_small, output_bam=output,
            model_or_path=benchmark_model_path,
            train_rids=set(), edge_trim=10, circular=False,
            mode='pacbio-fiber', context_size=3, msp_min_size=60,
            n_cores=2, streaming_pipeline=True, chunk_size=20,
            min_mapq=0, min_read_length=0, prob_threshold=0,
        )
        assert _count_bam_reads(output) == _count_bam_reads(synthetic_bam_small)

    def test_footprint_tags_present(self, synthetic_bam_small, benchmark_model_path, tmp_path):
        """Processed reads get ns/nl tags."""
        output = str(tmp_path / "out.bam")
        total, with_fp = process_bam_for_footprints(
            input_bam=synthetic_bam_small, output_bam=output,
            model_or_path=benchmark_model_path,
            train_rids=set(), edge_trim=10, circular=False,
            mode='pacbio-fiber', context_size=3, msp_min_size=60,
            n_cores=2, streaming_pipeline=True, chunk_size=20,
            min_mapq=0, min_read_length=0, prob_threshold=0,
        )
        tags = _read_tags_by_name(output)
        tagged = sum(1 for t in tags.values() if 'ns' in t)
        assert tagged == with_fp
        assert with_fp > 0

    def test_msp_tags_present(self, synthetic_bam_small, benchmark_model_path, tmp_path):
        """as/al tags present by default."""
        output = str(tmp_path / "out.bam")
        process_bam_for_footprints(
            input_bam=synthetic_bam_small, output_bam=output,
            model_or_path=benchmark_model_path,
            train_rids=set(), edge_trim=10, circular=False,
            mode='pacbio-fiber', context_size=3, msp_min_size=60,
            n_cores=2, streaming_pipeline=True, chunk_size=20,
            min_mapq=0, min_read_length=0, prob_threshold=0,
        )
        tags = _read_tags_by_name(output)
        has_msp = sum(1 for t in tags.values() if 'as' in t)
        assert has_msp > 0

    def test_no_msp_tags_when_disabled(self, synthetic_bam_small, benchmark_model_path, tmp_path):
        """as/al tags absent when write_msps=False."""
        output = str(tmp_path / "out.bam")
        process_bam_for_footprints(
            input_bam=synthetic_bam_small, output_bam=output,
            model_or_path=benchmark_model_path,
            train_rids=set(), edge_trim=10, circular=False,
            mode='pacbio-fiber', context_size=3, msp_min_size=60,
            n_cores=2, streaming_pipeline=True, chunk_size=20,
            min_mapq=0, min_read_length=0, prob_threshold=0,
            write_msps=False,
        )
        tags = _read_tags_by_name(output)
        has_msp = sum(1 for t in tags.values() if 'as' in t)
        assert has_msp == 0

    def test_tag_values_reasonable(self, synthetic_bam_small, benchmark_model_path, tmp_path):
        """ns values within read bounds, nl > 0."""
        output = str(tmp_path / "out.bam")
        process_bam_for_footprints(
            input_bam=synthetic_bam_small, output_bam=output,
            model_or_path=benchmark_model_path,
            train_rids=set(), edge_trim=10, circular=False,
            mode='pacbio-fiber', context_size=3, msp_min_size=60,
            n_cores=2, streaming_pipeline=True, chunk_size=20,
            min_mapq=0, min_read_length=0, prob_threshold=0,
        )
        with pysam.AlignmentFile(output, "rb") as bam:
            for read in bam:
                if read.has_tag('ns'):
                    starts = list(read.get_tag('ns'))
                    lengths = list(read.get_tag('nl'))
                    assert len(starts) == len(lengths)
                    for s, length in zip(starts, lengths):
                        assert s >= 0
                        assert length > 0
                        assert s + length <= read.query_length

    def test_scores_when_requested(self, synthetic_bam_small, benchmark_model_path, tmp_path):
        """nq tags present when with_scores=True."""
        output = str(tmp_path / "out.bam")
        process_bam_for_footprints(
            input_bam=synthetic_bam_small, output_bam=output,
            model_or_path=benchmark_model_path,
            train_rids=set(), edge_trim=10, circular=False,
            mode='pacbio-fiber', context_size=3, msp_min_size=60,
            n_cores=2, streaming_pipeline=True, chunk_size=20,
            min_mapq=0, min_read_length=0, prob_threshold=0,
            with_scores=True,
        )
        tags = _read_tags_by_name(output)
        has_scores = sum(1 for t in tags.values() if 'nq' in t)
        assert has_scores > 0


class TestStreamingOrderAndDeterminism:
    """Order preservation and deterministic output."""

    def test_order_preserved(self, synthetic_bam_small, benchmark_model_path, tmp_path):
        """Output read order matches input read order."""
        output = str(tmp_path / "out.bam")
        process_bam_for_footprints(
            input_bam=synthetic_bam_small, output_bam=output,
            model_or_path=benchmark_model_path,
            train_rids=set(), edge_trim=10, circular=False,
            mode='pacbio-fiber', context_size=3, msp_min_size=60,
            n_cores=2, streaming_pipeline=True, chunk_size=20,
            min_mapq=0, min_read_length=0, prob_threshold=0,
        )
        input_names = _read_names_in_order(synthetic_bam_small)
        output_names = _read_names_in_order(output)
        assert input_names == output_names

    def test_deterministic(self, synthetic_bam_small, benchmark_model_path, tmp_path):
        """Two runs produce identical tag values."""
        out1 = str(tmp_path / "out1.bam")
        out2 = str(tmp_path / "out2.bam")

        for out in [out1, out2]:
            process_bam_for_footprints(
                input_bam=synthetic_bam_small, output_bam=out,
                model_or_path=benchmark_model_path,
                train_rids=set(), edge_trim=10, circular=False,
                mode='pacbio-fiber', context_size=3, msp_min_size=60,
                n_cores=2, streaming_pipeline=True, chunk_size=20,
                min_mapq=0, min_read_length=0, prob_threshold=0,
            )

        tags1 = _read_tags_by_name(out1)
        tags2 = _read_tags_by_name(out2)
        assert tags1 == tags2

    def test_single_core_matches_multi_core(
        self, synthetic_bam_small, benchmark_model_path, tmp_path,
    ):
        """n_cores=1 and n_cores=2 produce same tags."""
        out1 = str(tmp_path / "out_1core.bam")
        out2 = str(tmp_path / "out_2core.bam")

        for out, cores in [(out1, 1), (out2, 2)]:
            process_bam_for_footprints(
                input_bam=synthetic_bam_small, output_bam=out,
                model_or_path=benchmark_model_path,
                train_rids=set(), edge_trim=10, circular=False,
                mode='pacbio-fiber', context_size=3, msp_min_size=60,
                n_cores=cores, streaming_pipeline=True, chunk_size=20,
                min_mapq=0, min_read_length=0, prob_threshold=0,
            )

        tags1 = _read_tags_by_name(out1)
        tags2 = _read_tags_by_name(out2)
        assert tags1 == tags2


class TestStreamingEdgeCases:
    """Edge cases and special inputs."""

    def test_empty_bam(self, empty_bam, benchmark_model_path, tmp_path):
        """Empty BAM produces empty output, returns (0, 0)."""
        output = str(tmp_path / "out.bam")
        total, fp = process_bam_for_footprints(
            input_bam=empty_bam, output_bam=output,
            model_or_path=benchmark_model_path,
            train_rids=set(), edge_trim=10, circular=False,
            mode='pacbio-fiber', context_size=3, msp_min_size=60,
            n_cores=2, streaming_pipeline=True, chunk_size=20,
            min_mapq=0, min_read_length=0, prob_threshold=0,
        )
        assert total == 0
        assert fp == 0
        assert _count_bam_reads(output) == 0

    def test_works_without_index(self, synthetic_bam_small, benchmark_model_path, tmp_path):
        """Streaming pipeline works with unindexed BAM."""
        # Copy BAM without its index
        import shutil
        unindexed = str(tmp_path / "noindex.bam")
        shutil.copy2(synthetic_bam_small, unindexed)
        # Don't copy .bai

        output = str(tmp_path / "out.bam")
        total, fp = process_bam_for_footprints(
            input_bam=unindexed, output_bam=output,
            model_or_path=benchmark_model_path,
            train_rids=set(), edge_trim=10, circular=False,
            mode='pacbio-fiber', context_size=3, msp_min_size=60,
            n_cores=2, streaming_pipeline=True, chunk_size=20,
            min_mapq=0, min_read_length=0, prob_threshold=0,
        )
        assert total > 0
        assert _count_bam_reads(output) == _count_bam_reads(synthetic_bam_small)

    def test_unmapped_passthrough(self, unaligned_bam, benchmark_model_path, tmp_path):
        """Unmapped reads written unchanged when process_unmapped=False."""
        output = str(tmp_path / "out.bam")
        total, fp = process_bam_for_footprints(
            input_bam=unaligned_bam, output_bam=output,
            model_or_path=benchmark_model_path,
            train_rids=set(), edge_trim=10, circular=False,
            mode='pacbio-fiber', context_size=3, msp_min_size=60,
            n_cores=2, streaming_pipeline=True, chunk_size=20,
            min_mapq=0, min_read_length=0, prob_threshold=0,
            process_unmapped=False,
        )
        assert total == 0  # All skipped as unmapped
        assert fp == 0
        # All reads should be written (passthrough)
        assert _count_bam_reads(output) == _count_bam_reads(unaligned_bam)

    def test_unmapped_processing(self, unaligned_bam, benchmark_model_path, tmp_path):
        """Unmapped reads with sequences are processed when process_unmapped=True."""
        output = str(tmp_path / "out.bam")
        total, fp = process_bam_for_footprints(
            input_bam=unaligned_bam, output_bam=output,
            model_or_path=benchmark_model_path,
            train_rids=set(), edge_trim=10, circular=False,
            mode='pacbio-fiber', context_size=3, msp_min_size=60,
            n_cores=2, streaming_pipeline=True, chunk_size=20,
            min_mapq=0, min_read_length=0, prob_threshold=0,
            process_unmapped=True,
        )
        assert total > 0  # Some reads processed
        tags = _read_tags_by_name(output)
        tagged = sum(1 for t in tags.values() if 'ns' in t)
        assert tagged > 0

    def test_max_reads_limit(self, synthetic_bam_small, benchmark_model_path, tmp_path):
        """max_reads parameter limits processing."""
        output = str(tmp_path / "out.bam")
        total, fp = process_bam_for_footprints(
            input_bam=synthetic_bam_small, output_bam=output,
            model_or_path=benchmark_model_path,
            train_rids=set(), edge_trim=10, circular=False,
            mode='pacbio-fiber', context_size=3, msp_min_size=60,
            n_cores=2, streaming_pipeline=True, chunk_size=5,
            min_mapq=0, min_read_length=0, prob_threshold=0,
            max_reads=10,
        )
        assert total == 10

    def test_chunk_size_1(self, synthetic_bam_small, benchmark_model_path, tmp_path):
        """chunk_size=1 still works correctly (degenerate case)."""
        output = str(tmp_path / "out.bam")
        total, fp = process_bam_for_footprints(
            input_bam=synthetic_bam_small, output_bam=output,
            model_or_path=benchmark_model_path,
            train_rids=set(), edge_trim=10, circular=False,
            mode='pacbio-fiber', context_size=3, msp_min_size=60,
            n_cores=2, streaming_pipeline=True, chunk_size=1,
            min_mapq=0, min_read_length=0, prob_threshold=0,
        )
        assert total == _count_bam_reads(synthetic_bam_small)
        assert _count_bam_reads(output) == _count_bam_reads(synthetic_bam_small)

    def test_large_chunk_size(self, synthetic_bam_small, benchmark_model_path, tmp_path):
        """chunk_size larger than total reads still works."""
        output = str(tmp_path / "out.bam")
        total, fp = process_bam_for_footprints(
            input_bam=synthetic_bam_small, output_bam=output,
            model_or_path=benchmark_model_path,
            train_rids=set(), edge_trim=10, circular=False,
            mode='pacbio-fiber', context_size=3, msp_min_size=60,
            n_cores=2, streaming_pipeline=True, chunk_size=10000,
            min_mapq=0, min_read_length=0, prob_threshold=0,
        )
        assert total == _count_bam_reads(synthetic_bam_small)
