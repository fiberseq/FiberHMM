"""
Tests for fiberhmm.inference.parallel module — region splitting and chromosome filtering.
"""
import sys
from collections import deque
from concurrent.futures import Future
from types import SimpleNamespace

import numpy as np
import pytest

import fiberhmm.inference.legacy_pipeline as legacy_pipeline
import fiberhmm.inference.mp_context as mp_context
import fiberhmm.inference.parallel as parallel
import fiberhmm.inference.region_pipeline as region_pipeline
import fiberhmm.inference.region_workers as region_workers
import fiberhmm.inference.streaming_drain as streaming_drain
import fiberhmm.inference.streaming_pipeline as streaming_pipeline
import fiberhmm.inference.streaming_workers as streaming_workers
from fiberhmm.inference.engine import CHIMERA_SKIP
from fiberhmm.inference.parallel import (
    _is_main_chromosome,
    _model_and_path_for_processing,
)
from fiberhmm.inference.read_filters import ReadFilterConfig
from fiberhmm.inference.worker_results import WorkerChunkResult, coerce_worker_chunk_result


class _OutBam:
    def __init__(self):
        self.written = []

    def write(self, read):
        self.written.append(read)


def _done_future(value):
    future = Future()
    future.set_result(value)
    return future


class _Executor:
    def __init__(self):
        self.submitted = []

    def submit(self, fn, *args):
        self.submitted.append((fn, args))
        return _done_future(("submitted", fn, args))


def _pipeline_options(**overrides):
    attrs = {
        "input_bam": "in.bam",
        "output_bam": "out.bam",
        "train_rids": set(),
        "edge_trim": 0,
        "circular": False,
        "mode": "daf",
        "context_size": 5,
        "msp_min_size": 60,
        "nuc_min_size": 85,
        "min_mapq": 0,
        "prob_threshold": 0,
        "min_read_length": 0,
        "with_scores": False,
        "n_cores": 1,
        "primary_only": False,
        "output_posteriors": None,
        "write_msps": True,
        "io_threads": 1,
    }
    attrs.update(overrides)
    return parallel._FootprintPipelineOptions(**attrs)


def test_worker_chunk_result_coerces_legacy_lists():
    assert coerce_worker_chunk_result([None]) == WorkerChunkResult([None], 0)
    assert coerce_worker_chunk_result(
        WorkerChunkResult([None], 2)
    ) == WorkerChunkResult([None], 2)


def test_parallel_reexports_multiprocessing_context():
    assert parallel._select_mp_context is mp_context._select_mp_context
    assert parallel._MP_CONTEXT is mp_context._MP_CONTEXT


def test_mp_context_helpers_normalize_override_and_default_policy():
    assert mp_context._normalize_mp_context_override(" Spawn ") == "spawn"
    assert mp_context._normalize_mp_context_override("forkserver") == "forkserver"
    assert mp_context._normalize_mp_context_override("invalid") is None
    assert mp_context._normalize_mp_context_override("") is None

    assert mp_context._default_mp_context_name((3, 13)) == "fork"
    assert mp_context._default_mp_context_name((3, 14)) == "spawn"


def test_model_and_path_for_processing_splits_path_and_loaded_model():
    source = _model_and_path_for_processing("model.json")
    assert source.model is None
    assert source.path == "model.json"

    model = object()
    source = _model_and_path_for_processing(model)
    assert source.model is model
    assert source.path is None


def test_parallel_reexports_legacy_pipeline_entry_points():
    assert parallel._process_and_write_chunk is legacy_pipeline._process_and_write_chunk
    assert parallel._process_bam_legacy_pipeline is legacy_pipeline._process_bam_legacy_pipeline


def test_legacy_skip_helpers_write_and_count_reason():
    read = object()
    outbam = _OutBam()
    skip_reasons = legacy_pipeline._new_legacy_skip_reasons()

    assert legacy_pipeline._write_skipped_legacy_read(
        outbam,
        read,
        skip_reasons,
        "no_modifications",
    ) == 1

    assert outbam.written == [read]
    assert skip_reasons["no_modifications"] == 1
    assert skip_reasons["low_mapq"] == 0


def test_legacy_posterior_ref_positions_uses_backend_when_available(monkeypatch):
    read = object()
    monkeypatch.setattr(legacy_pipeline, "HAS_POSTERIOR_WRITER", False)
    empty = legacy_pipeline._legacy_posterior_ref_positions(read)

    assert empty.dtype == np.int32
    assert empty.size == 0

    positions = np.asarray([10, 20], dtype=np.int32)
    monkeypatch.setattr(legacy_pipeline, "HAS_POSTERIOR_WRITER", True)
    monkeypatch.setattr(
        legacy_pipeline,
        "get_ref_positions_from_read",
        lambda got_read: positions,
    )

    assert legacy_pipeline._legacy_posterior_ref_positions(read) is positions


def test_open_legacy_posterior_writer_enables_posteriors(monkeypatch, capsys):
    created = []

    class FakePosteriorWriter:
        def __init__(self, *args, **kwargs):
            created.append((args, kwargs))

    monkeypatch.setattr(legacy_pipeline, "HAS_POSTERIOR_WRITER", True)
    monkeypatch.setattr(
        legacy_pipeline,
        "PosteriorWriter",
        FakePosteriorWriter,
        raising=False,
    )

    posterior_output = legacy_pipeline._open_legacy_posterior_writer_from_request(
        legacy_pipeline._LegacyPosteriorWriterOpenRequest(
            output_posteriors="out.h5",
            mode="daf",
            context_size=5,
            edge_trim=11,
            input_bam="input.bam",
        )
    )

    assert isinstance(posterior_output.writer, FakePosteriorWriter)
    assert posterior_output.enabled is True
    assert created == [
        (("out.h5", "daf", 5, 11, "input.bam"), {"batch_size": 1000})
    ]
    assert capsys.readouterr().out == "Posteriors will be written to: out.h5\n"


def test_open_legacy_posterior_writer_handles_disabled_and_missing_backend(
    monkeypatch, capsys,
):
    assert legacy_pipeline._open_legacy_posterior_writer(
        None, "daf", 5, 11, "input.bam",
    ) == legacy_pipeline._LegacyPosteriorWriter(writer=None, enabled=False)
    assert capsys.readouterr().out == ""

    monkeypatch.setattr(legacy_pipeline, "HAS_POSTERIOR_WRITER", False)
    assert legacy_pipeline._open_legacy_posterior_writer(
        "out.h5", "daf", 5, 11, "input.bam",
    ) == legacy_pipeline._LegacyPosteriorWriter(writer=None, enabled=False)
    assert capsys.readouterr().out == (
        "WARNING: posterior_writer.py not found, skipping posteriors export\n"
    )


def test_legacy_executor_helpers_configure_parallel_pool(monkeypatch):
    class FakeExecutor:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr(legacy_pipeline, "ProcessPoolExecutor", FakeExecutor)

    executor = legacy_pipeline._new_legacy_executor(
        model_path="model.pkl", n_cores=3, debug_timing=True,
    )

    assert executor.kwargs == {
        "max_workers": 3,
        "mp_context": legacy_pipeline._MP_CONTEXT,
        "initializer": legacy_pipeline._init_bam_worker,
        "initargs": ("model.pkl", True),
    }
    assert legacy_pipeline._legacy_executor_for_config(None, 3, True) is None
    assert legacy_pipeline._legacy_executor_for_config("model.pkl", 1, True) is None
    assert isinstance(
        legacy_pipeline._legacy_executor_for_config("model.pkl", 2, False),
        FakeExecutor,
    )


def test_legacy_filter_config_preserves_apply_filters():
    train_rids = {"read-a"}

    config = legacy_pipeline._legacy_filter_config(
        min_mapq=20,
        min_read_length=100,
        primary_only=True,
        train_rids=train_rids,
    )

    assert config == ReadFilterConfig(
        min_mapq=20,
        min_read_length=100,
        primary_only=True,
        process_unmapped=False,
        train_rids=train_rids,
    )


def test_shutdown_legacy_resources_closes_executor_and_posteriors():
    calls = []

    class Executor:
        def __bool__(self):
            return False

        def shutdown(self, wait):
            calls.append(("shutdown", wait))

    class Writer:
        def __bool__(self):
            return False

        def close(self):
            calls.append(("close",))
            return (123, 1.5)

    assert legacy_pipeline._legacy_posterior_stats_from_close_result(None) is None
    assert legacy_pipeline._legacy_posterior_stats_from_close_result(
        (123, 1.5),
    ) == legacy_pipeline._LegacyPosteriorStats(123, 1.5)
    assert legacy_pipeline._shutdown_legacy_resources(
        Executor(),
        Writer(),
    ) == legacy_pipeline._LegacyPosteriorStats(123, 1.5)
    assert calls == [("shutdown", True), ("close",)]


def test_shutdown_legacy_resources_closes_posteriors_after_executor_error():
    calls = []

    class Executor:
        def __bool__(self):
            return False

        def shutdown(self, wait):
            calls.append(("shutdown", wait))
            raise RuntimeError("shutdown failed")

    class Writer:
        def __bool__(self):
            return False

        def close(self):
            calls.append(("close",))

    with pytest.raises(RuntimeError, match="shutdown failed"):
        legacy_pipeline._shutdown_legacy_resources(Executor(), Writer())

    assert calls == [("shutdown", True), ("close",)]


def test_write_chunk_posteriors_accepts_falsey_writer(monkeypatch):
    read = SimpleNamespace(reference_name="chr1")
    seen = []

    class Result(dict):
        def __bool__(self):
            return False

    result = Result(posteriors=[0.1, 0.9])

    class Writer:
        def __bool__(self):
            return False

        def add_fiber(self, chrom, data):
            seen.append((chrom, data))

    monkeypatch.setattr(
        legacy_pipeline,
        "_legacy_posterior_ref_positions",
        lambda got_read: ["ref-pos"],
    )
    monkeypatch.setattr(
        legacy_pipeline,
        "posterior_fiber_data",
        lambda got_read, got_result, ref_positions: {
            "read": got_read,
            "result": got_result,
            "ref_positions": ref_positions,
        },
    )

    legacy_pipeline._write_chunk_posteriors(
        Writer(),
        [(read, {}, result)],
    )

    assert seen == [(
        "chr1",
        {
            "read": read,
            "result": result,
            "ref_positions": ["ref-pos"],
        },
    )]


def test_run_legacy_bam_processing_closes_posterior_on_executor_setup_failure(
    monkeypatch,
):
    class FakeAlignmentFile:
        def __init__(self, *args, **kwargs):
            self.header = {"HD": {"SO": "coordinate"}}

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class Writer:
        def __init__(self):
            self.closed = 0

        def close(self):
            self.closed += 1
            return (0, 0.0)

    writer = Writer()
    monkeypatch.setattr(legacy_pipeline.pysam, "AlignmentFile", FakeAlignmentFile)
    monkeypatch.setattr(legacy_pipeline, "append_coord_marker", lambda header: header)
    monkeypatch.setattr(
        legacy_pipeline,
        "_open_legacy_posterior_writer",
        lambda *args: legacy_pipeline._LegacyPosteriorWriter(
            writer=writer,
            enabled=True,
        ),
    )
    monkeypatch.setattr(
        legacy_pipeline,
        "_legacy_executor_for_config",
        lambda *args: (_ for _ in ()).throw(RuntimeError("executor failed")),
    )

    with pytest.raises(RuntimeError, match="executor failed"):
        legacy_pipeline._run_legacy_bam_processing_from_request(
            legacy_pipeline._LegacyBamProcessingRequest(
                input_bam="in.bam",
                output_bam="out.bam",
                model=object(),
                model_path="model.json",
                filter_config=ReadFilterConfig(),
                skip_reasons=legacy_pipeline._new_legacy_skip_reasons(),
                edge_trim=0,
                circular=False,
                mode="daf",
                context_size=5,
                msp_min_size=0,
                nuc_min_size=0,
                prob_threshold=0,
                with_scores=False,
                n_cores=2,
                max_reads=None,
                debug_timing=False,
                output_posteriors="post.h5",
                write_msps=True,
                io_threads=1,
                start_time=10.0,
                chunk_size=1,
            )
        )

    assert writer.closed == 1


def test_run_legacy_bam_processing_builds_request(monkeypatch):
    calls = []
    model = object()
    filter_config = ReadFilterConfig()
    skip_reasons = legacy_pipeline._new_legacy_skip_reasons()
    pipeline_result = legacy_pipeline._LegacyPipelineResult(
        total_reads=1,
        reads_with_footprints=2,
        skipped=3,
        worker_failures=4,
        posterior_stats=None,
    )

    monkeypatch.setattr(
        legacy_pipeline,
        "_run_legacy_bam_processing_from_request",
        lambda request: calls.append(request) or pipeline_result,
    )

    assert legacy_pipeline._run_legacy_bam_processing(
        input_bam="in.bam",
        output_bam="out.bam",
        model=model,
        model_path="model.json",
        filter_config=filter_config,
        skip_reasons=skip_reasons,
        edge_trim=11,
        circular=True,
        mode="daf",
        context_size=5,
        msp_min_size=61,
        nuc_min_size=87,
        prob_threshold=128,
        with_scores=True,
        n_cores=2,
        max_reads=10,
        debug_timing=True,
        output_posteriors="post.h5",
        write_msps=False,
        io_threads=1,
        start_time=10.0,
        chunk_size=100,
    ) is pipeline_result
    assert calls == [
        legacy_pipeline._LegacyBamProcessingRequest(
            input_bam="in.bam",
            output_bam="out.bam",
            model=model,
            model_path="model.json",
            filter_config=filter_config,
            skip_reasons=skip_reasons,
            edge_trim=11,
            circular=True,
            mode="daf",
            context_size=5,
            msp_min_size=61,
            nuc_min_size=87,
            prob_threshold=128,
            with_scores=True,
            n_cores=2,
            max_reads=10,
            debug_timing=True,
            output_posteriors="post.h5",
            write_msps=False,
            io_threads=1,
            start_time=10.0,
            chunk_size=100,
        )
    ]


def test_process_bam_legacy_pipeline_uses_default_chunk_size(monkeypatch):
    calls = []

    monkeypatch.setattr(
        legacy_pipeline,
        "_run_legacy_bam_processing",
        lambda **kwargs: calls.append(kwargs) or legacy_pipeline._LegacyPipelineResult(
            total_reads=10,
            reads_with_footprints=4,
            skipped=1,
            worker_failures=0,
            posterior_stats=None,
        ),
    )
    monkeypatch.setattr(
        legacy_pipeline,
        "_print_legacy_completion_summary",
        lambda *args: None,
    )
    monkeypatch.setattr(
        legacy_pipeline,
        "_sort_and_index_bam",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        legacy_pipeline,
        "_print_legacy_posterior_summary",
        lambda *args: None,
    )

    assert legacy_pipeline._process_bam_legacy_pipeline(
        input_bam="in.bam",
        output_bam="out.bam",
        model=object(),
        model_path="model.json",
        train_rids=set(),
        edge_trim=0,
        circular=False,
        mode="daf",
        context_size=5,
        msp_min_size=0,
    ) == (10, 4)

    assert calls[0]["chunk_size"] == legacy_pipeline._LEGACY_DEFAULT_CHUNK_SIZE


def test_legacy_progress_and_completion_messages_format_counts():
    assert legacy_pipeline._legacy_processing_rate(10, 2.0) == 5.0
    assert legacy_pipeline._legacy_processing_rate(10, 0.0) == 0
    assert legacy_pipeline._legacy_progress_message(
        total_reads=1234,
        skipped=56,
        rate=7.89,
    ) == "\r  Processed: 1,234 | Skipped: 56 | 7.9 reads/s"
    assert legacy_pipeline._legacy_completion_message(
        total_reads=1234,
        skipped=56,
        reads_with_footprints=789,
        rate=7.89,
    ) == (
        "\r  Processed: 1,234 | Skipped: 56 | "
        "With footprints: 789 | 7.9 reads/s"
    )


def test_print_legacy_completion_summary_reports_failures_and_skips(capsys):
    legacy_pipeline._print_legacy_completion_summary(
        total_reads=8,
        skipped=2,
        reads_with_footprints=5,
        worker_failures=1,
        skip_reasons={"low_mapq": 2, "no_modifications": 0},
        elapsed=4.0,
    )

    assert capsys.readouterr().out == (
        "\r  Processed: 8 | Skipped: 2 | With footprints: 5 | 2.0 reads/s\n"
        "  Worker read failures: 1 (passed through unchanged)\n"
        "  Skip reasons:\n"
        "    low_mapq: 2 (20.0%)\n"
    )


def test_print_legacy_posterior_summary_handles_empty_and_values(capsys):
    legacy_pipeline._print_legacy_posterior_summary(None, "out.h5")
    assert capsys.readouterr().out == ""

    legacy_pipeline._print_legacy_posterior_summary(
        legacy_pipeline._LegacyPosteriorStats(1234, 5.678), "out.h5",
    )
    assert capsys.readouterr().out == (
        "Posteriors: 1,234 fibers -> out.h5 (5.7 MB)\n"
    )


def test_process_legacy_chunk_buffer_skips_empty_and_delegates(monkeypatch):
    calls = []
    outbam = object()
    model = object()
    executor = object()
    skip_reasons = {}
    posterior_writer = object()

    monkeypatch.setattr(
        legacy_pipeline,
        "_process_legacy_chunk_and_record",
        lambda *args, **kwargs: calls.append((args, kwargs))
        or legacy_pipeline._LegacyChunkRecordResult(3, 1),
    )

    empty_request = legacy_pipeline._LegacyChunkBufferRequest(
        chunk_reads=[],
        chunk_read_objs=[],
        outbam=outbam,
        model=model,
        executor=executor,
        edge_trim=1,
        circular=False,
        mode="daf",
        context_size=5,
        msp_min_size=30,
        skip_reasons=skip_reasons,
        posterior_writer=posterior_writer,
        nuc_min_size=90,
        with_scores=True,
        return_posteriors=True,
        write_msps=False,
    )

    assert legacy_pipeline._process_legacy_chunk_buffer_from_request(
        empty_request,
    ) == legacy_pipeline._LegacyChunkRecordResult(0, 0)
    assert calls == []

    chunk_reads = ["fiber"]
    chunk_read_objs = ["read"]
    request = legacy_pipeline._LegacyChunkBufferRequest(
        chunk_reads=chunk_reads,
        chunk_read_objs=chunk_read_objs,
        outbam=outbam,
        model=model,
        executor=executor,
        edge_trim=1,
        circular=False,
        mode="daf",
        context_size=5,
        msp_min_size=30,
        skip_reasons=skip_reasons,
        posterior_writer=posterior_writer,
        nuc_min_size=90,
        with_scores=True,
        return_posteriors=True,
        write_msps=False,
    )

    assert legacy_pipeline._process_legacy_chunk_buffer_from_request(
        request,
    ) == legacy_pipeline._LegacyChunkRecordResult(3, 1)

    args, kwargs = calls[0]
    assert args == (
        chunk_reads,
        chunk_read_objs,
        outbam,
        model,
        executor,
        1,
        False,
        "daf",
        5,
        30,
        skip_reasons,
        posterior_writer,
    )
    assert kwargs == {
        "nuc_min_size": 90,
        "with_scores": True,
        "return_posteriors": True,
        "write_msps": False,
    }

    calls.clear()
    assert legacy_pipeline._process_legacy_chunk_buffer(
        chunk_reads,
        chunk_read_objs,
        outbam,
        model,
        executor,
        1,
        False,
        "daf",
        5,
        30,
        skip_reasons,
        posterior_writer,
        nuc_min_size=90,
        with_scores=True,
        return_posteriors=True,
        write_msps=False,
    ) == legacy_pipeline._LegacyChunkRecordResult(3, 1)
    assert calls


def test_legacy_chunk_buffer_request_combines_buffers_and_config():
    outbam = object()
    model = object()
    executor = object()
    skip_reasons = {}
    posterior_writer = object()
    buffers = legacy_pipeline._LegacyChunkBuffers(
        fiber_reads=["fiber"],
        read_objs=["read"],
    )
    config = legacy_pipeline._legacy_chunk_buffer_config(
        outbam,
        model,
        executor,
        edge_trim=11,
        circular=True,
        mode="daf",
        context_size=5,
        msp_min_size=61,
        skip_reasons=skip_reasons,
        posterior_writer=posterior_writer,
        nuc_min_size=87,
        with_scores=True,
        return_posteriors=True,
        write_msps=False,
    )

    assert legacy_pipeline._legacy_chunk_buffer_request(
        buffers,
        config,
    ) == legacy_pipeline._LegacyChunkBufferRequest(
        chunk_reads=["fiber"],
        chunk_read_objs=["read"],
        outbam=outbam,
        model=model,
        executor=executor,
        edge_trim=11,
        circular=True,
        mode="daf",
        context_size=5,
        msp_min_size=61,
        skip_reasons=skip_reasons,
        posterior_writer=posterior_writer,
        nuc_min_size=87,
        with_scores=True,
        return_posteriors=True,
        write_msps=False,
    )


def test_new_legacy_chunk_buffers_returns_independent_lists():
    buffers = legacy_pipeline._new_legacy_chunk_buffers()
    assert buffers == legacy_pipeline._LegacyChunkBuffers(fiber_reads=[], read_objs=[])

    buffers.fiber_reads.append("fiber")
    buffers.read_objs.append("read")

    assert legacy_pipeline._new_legacy_chunk_buffers() == (
        legacy_pipeline._LegacyChunkBuffers(fiber_reads=[], read_objs=[])
    )


def test_process_legacy_reads_buffers_skips_chunks_and_progress(monkeypatch):
    outbam = _OutBam()
    filter_config = object()
    model = object()
    executor = object()
    posterior_writer = object()
    chunk_calls = []
    progress_calls = []

    def fake_fiber_read_or_skip(read, got_filter_config, mode, prob_threshold):
        assert got_filter_config is filter_config
        assert mode == "daf"
        assert prob_threshold == 128
        if read == "skip":
            return legacy_pipeline._LegacyFiberReadResult(
                fiber_read=None,
                skip_reason="low_mapq",
            )
        return legacy_pipeline._LegacyFiberReadResult(
            fiber_read=f"fiber-{read}",
            skip_reason=None,
        )

    def fake_process_chunk(
        chunk_reads,
        chunk_read_objs,
        got_outbam,
        got_model,
        got_executor,
        edge_trim,
        circular,
        mode,
        context_size,
        msp_min_size,
        skip_reasons,
        got_posterior_writer,
        **kwargs,
    ):
        chunk_calls.append(
            (
                list(chunk_reads),
                list(chunk_read_objs),
                got_outbam,
                got_model,
                got_executor,
                edge_trim,
                circular,
                mode,
                context_size,
                msp_min_size,
                got_posterior_writer,
                kwargs,
            )
        )
        skip_reasons["no_footprints"] += 1
        return legacy_pipeline._LegacyChunkRecordResult(
            reads_with_footprints=len(chunk_reads),
            worker_failures=1,
        )

    monkeypatch.setattr(
        legacy_pipeline, "_legacy_fiber_read_or_skip", fake_fiber_read_or_skip,
    )
    monkeypatch.setattr(
        legacy_pipeline,
        "_process_legacy_chunk_and_record",
        fake_process_chunk,
    )
    monkeypatch.setattr(
        legacy_pipeline,
        "_print_legacy_progress",
        lambda *args: progress_calls.append(args),
    )
    skip_reasons = legacy_pipeline._new_legacy_skip_reasons()
    request = legacy_pipeline._LegacyReadsRequest(
        reads=["r1", "skip", "r2", "r3"],
        outbam=outbam,
        model=model,
        executor=executor,
        filter_config=filter_config,
        mode="daf",
        prob_threshold=128,
        edge_trim=11,
        circular=True,
        context_size=5,
        msp_min_size=61,
        skip_reasons=skip_reasons,
        posterior_writer=posterior_writer,
        start_time=10.0,
        max_reads=None,
        chunk_size=2,
        nuc_min_size=87,
        with_scores=True,
        return_posteriors=True,
        write_msps=False,
    )

    result = legacy_pipeline._process_legacy_reads_from_request(request)

    assert result == legacy_pipeline._LegacyReadProcessingResult(
        total_reads=3,
        reads_with_footprints=3,
        skipped=1,
        worker_failures=2,
    )
    assert outbam.written == ["skip"]
    assert skip_reasons["low_mapq"] == 1
    assert skip_reasons["no_footprints"] == 2
    assert progress_calls == [(2, 1, 10.0)]
    assert chunk_calls == [
        (
            ["fiber-r1", "fiber-r2"],
            ["r1", "r2"],
            outbam,
            model,
            executor,
            11,
            True,
            "daf",
            5,
            61,
            posterior_writer,
            {
                "nuc_min_size": 87,
                "with_scores": True,
                "return_posteriors": True,
                "write_msps": False,
            },
        ),
        (
            ["fiber-r3"],
            ["r3"],
            outbam,
            model,
            executor,
            11,
            True,
            "daf",
            5,
            61,
            posterior_writer,
            {
                "nuc_min_size": 87,
                "with_scores": True,
                "return_posteriors": True,
                "write_msps": False,
            },
        ),
    ]

    assert legacy_pipeline._process_legacy_reads(
        [],
        outbam,
        model,
        executor,
        filter_config,
        mode="daf",
        prob_threshold=128,
        edge_trim=11,
        circular=True,
        context_size=5,
        msp_min_size=61,
        skip_reasons=legacy_pipeline._new_legacy_skip_reasons(),
        posterior_writer=posterior_writer,
        start_time=10.0,
        max_reads=None,
        chunk_size=2,
        nuc_min_size=87,
        with_scores=True,
        return_posteriors=True,
        write_msps=False,
    ) == legacy_pipeline._LegacyReadProcessingResult(
        total_reads=0,
        reads_with_footprints=0,
        skipped=0,
        worker_failures=0,
    )


def test_parallel_reexports_streaming_worker_entry_points():
    assert parallel._init_bam_worker is streaming_workers._init_bam_worker
    assert parallel._init_fused_worker is streaming_workers._init_fused_worker
    assert parallel._process_chunk_worker is streaming_workers._process_chunk_worker
    assert (
        parallel._process_payload_chunk_worker
        is streaming_workers._process_payload_chunk_worker
    )
    assert (
        parallel._process_fused_payload_chunk_worker
        is streaming_workers._process_fused_payload_chunk_worker
    )


def test_parallel_reexports_streaming_pipeline_entry_points():
    assert (
        parallel._process_bam_streaming_pipeline
        is streaming_pipeline._process_bam_streaming_pipeline
    )
    assert (
        parallel._process_bam_streaming_pipeline_fused
        is streaming_pipeline._process_bam_streaming_pipeline_fused
    )


def test_parallel_reexports_region_worker_entry_points():
    assert parallel._init_region_worker is region_workers._init_region_worker
    assert parallel._init_fused_region_worker is region_workers._init_fused_region_worker
    assert parallel._process_region_to_bam is region_workers._process_region_to_bam
    assert parallel._process_region_to_bed is region_workers._process_region_to_bed
    assert (
        parallel._process_region_to_bam_fused
        is region_workers._process_region_to_bam_fused
    )


def test_parallel_reexports_region_pipeline_entry_points():
    assert parallel._process_bam_region_parallel is region_pipeline._process_bam_region_parallel
    assert (
        parallel._process_bam_region_parallel_fused
        is region_pipeline._process_bam_region_parallel_fused
    )
    assert parallel._process_bed_region_parallel is region_pipeline._process_bed_region_parallel


def test_streaming_summary_helpers_format_nonzero_counts(capsys):
    streaming_pipeline._print_worker_failure_summary(
        {"worker_failures": 3},
        sys.stdout,
    )
    streaming_pipeline._print_streaming_skip_summary(
        {"low_mapq": 2, "empty": 0},
        total_reads=10,
        skipped=2,
        log=sys.stdout,
    )

    out = capsys.readouterr().out
    assert "Worker read failures: 3 (passed through unchanged)" in out
    assert "low_mapq: 2 (16.7%)" in out
    assert "empty" not in out


def _streaming_read(**overrides):
    attrs = {
        "is_unmapped": False,
        "is_secondary": False,
        "is_supplementary": False,
        "mapping_quality": 60,
        "query_alignment_length": 100,
        "query_length": 100,
        "query_name": "read1",
        "query_sequence": "A" * 100,
    }
    attrs.update(overrides)
    return SimpleNamespace(**attrs)


def test_legacy_fiber_read_or_skip_filters_and_extracts(monkeypatch):
    config = ReadFilterConfig(min_mapq=20, min_read_length=50)
    extracted = []

    def fake_extract(read, mode, prob_threshold):
        extracted.append((read.query_name, mode, prob_threshold))
        return {"read_id": read.query_name}

    monkeypatch.setattr(legacy_pipeline, "_extract_fiber_read_from_pysam", fake_extract)

    fiber_result = legacy_pipeline._legacy_fiber_read_or_skip(
        _streaming_read(), config, "daf", 128,
    )
    assert fiber_result.fiber_read == {"read_id": "read1"}
    assert fiber_result.skip_reason is None
    assert extracted == [("read1", "daf", 128)]

    fiber_result = legacy_pipeline._legacy_fiber_read_or_skip(
        _streaming_read(mapping_quality=0), config, "daf", 128,
    )
    assert fiber_result.fiber_read is None
    assert fiber_result.skip_reason == "low_mapq"
    assert extracted == [("read1", "daf", 128)]

    monkeypatch.setattr(
        legacy_pipeline,
        "_extract_fiber_read_from_pysam",
        lambda read, mode, prob_threshold: None,
    )
    fiber_result = legacy_pipeline._legacy_fiber_read_or_skip(
        _streaming_read(), config, "daf", 128,
    )
    assert fiber_result.fiber_read is None
    assert fiber_result.skip_reason == "no_modifications"

    def fail_extract(read, mode, prob_threshold):
        raise ValueError("bad read")

    monkeypatch.setattr(legacy_pipeline, "_extract_fiber_read_from_pysam", fail_extract)
    fiber_result = legacy_pipeline._legacy_fiber_read_or_skip(
        _streaming_read(), config, "daf", 128,
    )
    assert fiber_result.fiber_read is None
    assert fiber_result.skip_reason == "extraction_failed"


def test_write_processed_legacy_reads_tags_results_and_counts(monkeypatch):
    tagged = []
    reads = [SimpleNamespace(query_name="read1"), SimpleNamespace(query_name="read2")]
    outbam = _OutBam()

    def fake_set_tags(read, result, with_scores, write_msps):
        tagged.append((read.query_name, result, with_scores, write_msps))

    monkeypatch.setattr(legacy_pipeline, "set_legacy_apply_tags", fake_set_tags)
    request = legacy_pipeline._ProcessedLegacyReadsWriteRequest(
        chunk_read_objs=reads,
        results=[{"ns": [1]}, None],
        outbam=outbam,
        with_scores=True,
        write_msps=False,
    )

    assert legacy_pipeline._write_processed_legacy_reads_from_request(
        request,
    ) == legacy_pipeline._LegacyWriteCounts(
        reads_with_footprints=1,
        no_footprints=1,
    )
    assert tagged == [("read1", {"ns": [1]}, True, False)]
    assert outbam.written == reads

    tagged.clear()
    outbam = _OutBam()
    assert legacy_pipeline._write_processed_legacy_reads(
        reads,
        [{"ns": [1]}, None],
        outbam,
        with_scores=True,
        write_msps=False,
    ) == legacy_pipeline._LegacyWriteCounts(
        reads_with_footprints=1,
        no_footprints=1,
    )
    assert tagged == [("read1", {"ns": [1]}, True, False)]
    assert outbam.written == reads


def test_process_legacy_chunk_results_uses_executor_when_available():
    class Executor:
        def __init__(self):
            self.submitted = []

        def submit(self, fn, *args):
            self.submitted.append((fn, args))
            return _done_future(WorkerChunkResult([{"ns": [1]}], read_failures=2))

    executor = Executor()
    model = object()
    request = legacy_pipeline._LegacyChunkResultsRequest(
        chunk_reads=["fiber"],
        model=model,
        executor=executor,
        edge_trim=11,
        circular=True,
        mode="daf",
        context_size=5,
        msp_min_size=61,
        nuc_min_size=87,
        with_scores=True,
        return_posteriors=False,
    )

    chunk_result = legacy_pipeline._process_legacy_chunk_results_from_request(
        request,
    )

    assert chunk_result == legacy_pipeline._LegacyChunkResult(
        results=[{"ns": [1]}],
        worker_failures=2,
    )
    assert executor.submitted == [(
        legacy_pipeline._process_chunk_worker,
        (["fiber"], 11, True, "daf", 5, 61, 87, True, False),
    )]

    executor = Executor()
    assert legacy_pipeline._process_legacy_chunk_results(
        ["fiber"],
        model=model,
        executor=executor,
        edge_trim=11,
        circular=True,
        mode="daf",
        context_size=5,
        msp_min_size=61,
        nuc_min_size=87,
        with_scores=True,
        return_posteriors=False,
    ) == legacy_pipeline._LegacyChunkResult(
        results=[{"ns": [1]}],
        worker_failures=2,
    )


def test_process_legacy_chunk_results_runs_direct_without_executor(monkeypatch):
    calls = []

    def fake_direct(request):
        calls.append(request)
        return [{"ns": [1]}, {"ns": [2]}]

    monkeypatch.setattr(
        legacy_pipeline,
        "_process_direct_legacy_chunk_results_from_request",
        fake_direct,
    )
    model = object()
    request = legacy_pipeline._LegacyChunkResultsRequest(
        chunk_reads=["fiber-a", "fiber-b"],
        model=model,
        executor=None,
        edge_trim=11,
        circular=True,
        mode="daf",
        context_size=5,
        msp_min_size=61,
        nuc_min_size=87,
        with_scores=True,
        return_posteriors=True,
    )

    chunk_result = legacy_pipeline._process_legacy_chunk_results_from_request(
        request,
    )

    assert chunk_result == legacy_pipeline._LegacyChunkResult(
        results=[{"ns": [1]}, {"ns": [2]}],
        worker_failures=0,
    )
    assert calls == [(
        legacy_pipeline._LegacyDirectChunkRequest(
            chunk_reads=["fiber-a", "fiber-b"],
            model=model,
            edge_trim=11,
            circular=True,
            mode="daf",
            context_size=5,
            msp_min_size=61,
            nuc_min_size=87,
            with_scores=True,
            return_posteriors=True,
        )
    )]

    calls.clear()
    assert legacy_pipeline._process_legacy_chunk_results(
        ["fiber-a", "fiber-b"],
        model=model,
        executor=None,
        edge_trim=11,
        circular=True,
        mode="daf",
        context_size=5,
        msp_min_size=61,
        nuc_min_size=87,
        with_scores=True,
        return_posteriors=True,
    ) == legacy_pipeline._LegacyChunkResult(
        results=[{"ns": [1]}, {"ns": [2]}],
        worker_failures=0,
    )
    assert calls == [(
        legacy_pipeline._LegacyDirectChunkRequest(
            chunk_reads=["fiber-a", "fiber-b"],
            model=model,
            edge_trim=11,
            circular=True,
            mode="daf",
            context_size=5,
            msp_min_size=61,
            nuc_min_size=87,
            with_scores=True,
            return_posteriors=True,
        )
    )]


def test_process_direct_legacy_chunk_results_runs_single_read(monkeypatch):
    seen = []

    def fake_process(*args, **kwargs):
        seen.append((args, kwargs))
        return {"ns": [len(seen)]}

    monkeypatch.setattr(legacy_pipeline, "_process_single_read", fake_process)
    model = object()

    request = legacy_pipeline._LegacyDirectChunkRequest(
        chunk_reads=["fiber-a", "fiber-b"],
        model=model,
        edge_trim=11,
        circular=True,
        mode="daf",
        context_size=5,
        msp_min_size=61,
        nuc_min_size=87,
        with_scores=True,
        return_posteriors=True,
    )

    results = legacy_pipeline._process_direct_legacy_chunk_results_from_request(
        request,
    )

    assert results == [{"ns": [1]}, {"ns": [2]}]
    assert seen == [
        (
            ("fiber-a", model, 11, True, "daf", 5, 61),
            {
                "nuc_min_size": 87,
                "with_scores": True,
                "return_posteriors": True,
            },
        ),
        (
            ("fiber-b", model, 11, True, "daf", 5, 61),
            {
                "nuc_min_size": 87,
                "with_scores": True,
                "return_posteriors": True,
            },
        ),
    ]

    seen.clear()
    assert legacy_pipeline._process_direct_legacy_chunk_results(
        ["fiber-a", "fiber-b"],
        model=model,
        edge_trim=11,
        circular=True,
        mode="daf",
        context_size=5,
        msp_min_size=61,
        nuc_min_size=87,
        with_scores=True,
        return_posteriors=True,
    ) == [{"ns": [1]}, {"ns": [2]}]
    assert seen == [
        (
            ("fiber-a", model, 11, True, "daf", 5, 61),
            {
                "nuc_min_size": 87,
                "with_scores": True,
                "return_posteriors": True,
            },
        ),
        (
            ("fiber-b", model, 11, True, "daf", 5, 61),
            {
                "nuc_min_size": 87,
                "with_scores": True,
                "return_posteriors": True,
            },
        ),
    ]


def test_process_and_write_chunk_uses_requests_and_records_posteriors(monkeypatch):
    chunk_reads = ["fiber"]
    chunk_read_objs = ["read"]
    outbam = object()
    model = object()
    executor = object()
    result_calls = []
    write_calls = []

    def fake_process_results(request):
        result_calls.append(request)
        return legacy_pipeline._LegacyChunkResult(
            results=[{"ns": [1]}],
            worker_failures=2,
        )

    def fake_write_reads(request):
        write_calls.append(request)
        return legacy_pipeline._LegacyWriteCounts(
            reads_with_footprints=1,
            no_footprints=0,
        )

    monkeypatch.setattr(
        legacy_pipeline,
        "_process_legacy_chunk_results_from_request",
        fake_process_results,
    )
    monkeypatch.setattr(
        legacy_pipeline,
        "_write_processed_legacy_reads_from_request",
        fake_write_reads,
    )
    request = legacy_pipeline._ProcessAndWriteLegacyChunkRequest(
        chunk_reads=chunk_reads,
        chunk_read_objs=chunk_read_objs,
        outbam=outbam,
        model=model,
        executor=executor,
        edge_trim=11,
        circular=True,
        mode="daf",
        context_size=5,
        msp_min_size=61,
        nuc_min_size=87,
        with_scores=True,
        return_posteriors=True,
        write_msps=False,
    )

    assert legacy_pipeline._process_and_write_chunk_from_request(
        request,
    ) == legacy_pipeline._ProcessedLegacyChunk(
        reads_with_footprints=1,
        no_footprints=0,
        worker_failures=2,
        posterior_records=[("read", "fiber", {"ns": [1]})],
    )
    assert result_calls == [
        legacy_pipeline._LegacyChunkResultsRequest(
            chunk_reads=chunk_reads,
            model=model,
            executor=executor,
            edge_trim=11,
            circular=True,
            mode="daf",
            context_size=5,
            msp_min_size=61,
            nuc_min_size=87,
            with_scores=True,
            return_posteriors=True,
        )
    ]
    assert write_calls == [
        legacy_pipeline._ProcessedLegacyReadsWriteRequest(
            chunk_read_objs=chunk_read_objs,
            results=[{"ns": [1]}],
            outbam=outbam,
            with_scores=True,
            write_msps=False,
        )
    ]

    assert legacy_pipeline._process_and_write_chunk(
        chunk_reads,
        chunk_read_objs,
        outbam,
        model,
        executor,
        edge_trim=11,
        circular=True,
        mode="daf",
        context_size=5,
        msp_min_size=61,
        nuc_min_size=87,
        with_scores=True,
        return_posteriors=False,
        write_msps=False,
    ) == legacy_pipeline._ProcessedLegacyChunk(
        reads_with_footprints=1,
        no_footprints=0,
        worker_failures=2,
        posterior_records=None,
    )


def test_process_legacy_chunk_and_record_updates_counts_and_posteriors(monkeypatch):
    chunk_reads = ["fiber"]
    chunk_read_objs = ["read"]
    outbam = object()
    model = object()
    executor = object()
    posterior_writer = object()
    chunk_results = ["chunk-result"]
    process_calls = []
    posterior_calls = []

    def fake_process_and_write_chunk(*args, **kwargs):
        process_calls.append((args, kwargs))
        return legacy_pipeline._ProcessedLegacyChunk(
            reads_with_footprints=2,
            no_footprints=3,
            worker_failures=1,
            posterior_records=chunk_results,
        )

    monkeypatch.setattr(
        legacy_pipeline, "_process_and_write_chunk", fake_process_and_write_chunk,
    )
    monkeypatch.setattr(
        legacy_pipeline,
        "_write_chunk_posteriors",
        lambda writer, results: posterior_calls.append((writer, results)),
    )
    skip_reasons = legacy_pipeline._new_legacy_skip_reasons()
    request = legacy_pipeline._LegacyChunkRecordRequest(
        chunk_reads=chunk_reads,
        chunk_read_objs=chunk_read_objs,
        outbam=outbam,
        model=model,
        executor=executor,
        edge_trim=11,
        circular=True,
        mode="daf",
        context_size=5,
        msp_min_size=61,
        skip_reasons=skip_reasons,
        posterior_writer=posterior_writer,
        nuc_min_size=87,
        with_scores=True,
        return_posteriors=True,
        write_msps=False,
    )

    chunk_result = legacy_pipeline._process_legacy_chunk_and_record_from_request(
        request,
    )

    assert chunk_result == legacy_pipeline._LegacyChunkRecordResult(2, 1)
    assert skip_reasons["no_footprints"] == 3
    assert posterior_calls == [(posterior_writer, chunk_results)]
    assert process_calls == [
        (
            (
                chunk_reads,
                chunk_read_objs,
                outbam,
                model,
                executor,
                11,
                True,
                "daf",
                5,
                61,
            ),
            {
                "nuc_min_size": 87,
                "with_scores": True,
                "return_posteriors": True,
                "write_msps": False,
            },
        )
    ]

    skip_reasons = legacy_pipeline._new_legacy_skip_reasons()
    process_calls.clear()
    posterior_calls.clear()
    assert legacy_pipeline._process_legacy_chunk_and_record(
        chunk_reads,
        chunk_read_objs,
        outbam,
        model,
        executor,
        edge_trim=11,
        circular=True,
        mode="daf",
        context_size=5,
        msp_min_size=61,
        skip_reasons=skip_reasons,
        posterior_writer=posterior_writer,
        nuc_min_size=87,
        with_scores=True,
        return_posteriors=True,
        write_msps=False,
    ) == legacy_pipeline._LegacyChunkRecordResult(2, 1)
    assert skip_reasons["no_footprints"] == 3
    assert posterior_calls == [(posterior_writer, chunk_results)]
    assert process_calls == [
        (
            (
                chunk_reads,
                chunk_read_objs,
                outbam,
                model,
                executor,
                11,
                True,
                "daf",
                5,
                61,
            ),
            {
                "nuc_min_size": 87,
                "with_scores": True,
                "return_posteriors": True,
                "write_msps": False,
            },
        )
    ]


def test_streaming_payload_or_skip_filters_and_builds_payload(monkeypatch):
    config = ReadFilterConfig(min_mapq=20, min_read_length=50)
    built = []

    def fake_make_payload(read, mode, ref_fasta):
        built.append((read.query_name, mode, ref_fasta))
        return {"read_id": read.query_name}

    monkeypatch.setattr(streaming_pipeline, "make_apply_payload", fake_make_payload)

    payload_result = streaming_pipeline._streaming_payload_or_skip(
        _streaming_read(), config, "daf", ref_fasta="ref.fa",
    )
    assert payload_result.payload == {"read_id": "read1"}
    assert payload_result.skip_reason is None
    assert built == [("read1", "daf", "ref.fa")]

    payload_result = streaming_pipeline._streaming_payload_or_skip(
        _streaming_read(mapping_quality=0), config, "daf",
    )
    assert payload_result.payload is None
    assert payload_result.skip_reason == "low_mapq"
    assert built == [("read1", "daf", "ref.fa")]

    monkeypatch.setattr(
        streaming_pipeline, "make_apply_payload",
        lambda read, mode, ref_fasta: None,
    )
    payload_result = streaming_pipeline._streaming_payload_or_skip(
        _streaming_read(), config, "daf",
    )
    assert payload_result.payload is None
    assert payload_result.skip_reason == "no_modifications"


def test_streaming_chunk_submission_uses_completed_future_for_empty_items():
    inflight = deque()
    executor = _Executor()

    streaming_pipeline._submit_streaming_chunk(
        inflight,
        executor,
        object(),
        [],
        ["read"],
        [True],
        ("arg",),
    )

    submitted = inflight.pop()
    assert executor.submitted == []
    assert submitted.future.result() == []
    assert submitted.read_objs == ["read"]
    assert submitted.items == []
    assert submitted.skip_flags == [True]


def test_streaming_chunk_submission_submits_nonempty_items():
    inflight = deque()
    executor = _Executor()
    worker_fn = object()

    streaming_pipeline._submit_streaming_chunk(
        inflight,
        executor,
        worker_fn,
        ["payload"],
        ["read"],
        [False],
        ("arg",),
    )

    submitted = inflight.pop()
    assert executor.submitted == [(worker_fn, (["payload"], "arg"))]
    assert submitted.future.result() == ("submitted", worker_fn, (["payload"], "arg"))
    assert submitted.read_objs == ["read"]
    assert submitted.items == ["payload"]
    assert submitted.skip_flags == [False]


def test_streaming_dispatch_does_not_load_model_in_parent(monkeypatch):
    monkeypatch.setattr(
        parallel,
        "load_model",
        lambda path: (_ for _ in ()).throw(AssertionError("parent load")),
    )
    monkeypatch.setattr(
        parallel,
        "_process_bam_streaming_pipeline",
        lambda **kwargs: ("streaming", kwargs["model_path"]),
    )

    assert parallel.process_bam_for_footprints(
        input_bam="in.bam",
        output_bam="out.bam",
        model_or_path="model.json",
        train_rids=set(),
        edge_trim=0,
        circular=False,
        mode="pacbio-fiber",
        context_size=3,
        msp_min_size=60,
        streaming_pipeline=True,
    ) == ("streaming", "model.json")


def test_footprint_pipeline_kwargs_preserve_common_options():
    train_rids = {"read-a"}
    options = parallel._FootprintPipelineOptions(
        input_bam="in.bam",
        output_bam="out.bam",
        train_rids=train_rids,
        edge_trim=2,
        circular=True,
        mode="daf",
        context_size=5,
        msp_min_size=30,
        nuc_min_size=90,
        min_mapq=20,
        prob_threshold=128,
        min_read_length=50,
        with_scores=True,
        n_cores=4,
        primary_only=True,
        output_posteriors="post.h5",
        write_msps=False,
        io_threads=3,
    )

    assert parallel._footprint_pipeline_options(
        input_bam="in.bam",
        output_bam="out.bam",
        train_rids=train_rids,
        edge_trim=2,
        circular=True,
        mode="daf",
        context_size=5,
        msp_min_size=30,
        nuc_min_size=90,
        min_mapq=20,
        prob_threshold=128,
        min_read_length=50,
        with_scores=True,
        n_cores=4,
        primary_only=True,
        output_posteriors="post.h5",
        write_msps=False,
        io_threads=3,
    ) == options
    assert parallel._footprint_pipeline_kwargs(
        input_bam="in.bam",
        output_bam="out.bam",
        train_rids=train_rids,
        edge_trim=2,
        circular=True,
        mode="daf",
        context_size=5,
        msp_min_size=30,
        nuc_min_size=90,
        min_mapq=20,
        prob_threshold=128,
        min_read_length=50,
        with_scores=True,
        n_cores=4,
        primary_only=True,
        output_posteriors="post.h5",
        write_msps=False,
        io_threads=3,
    ) == {
        "input_bam": "in.bam",
        "output_bam": "out.bam",
        "train_rids": train_rids,
        "edge_trim": 2,
        "circular": True,
        "mode": "daf",
        "context_size": 5,
        "msp_min_size": 30,
        "nuc_min_size": 90,
        "min_mapq": 20,
        "prob_threshold": 128,
        "min_read_length": 50,
        "with_scores": True,
        "n_cores": 4,
        "primary_only": True,
        "output_posteriors": "post.h5",
        "write_msps": False,
        "io_threads": 3,
    }


def test_requested_parallel_pipeline_prefers_region_dispatch(monkeypatch):
    calls = []

    monkeypatch.setattr(
        parallel,
        "_dispatch_region_parallel_if_requested",
        lambda **kwargs: calls.append(("region", kwargs)) or ("region", 1),
    )
    monkeypatch.setattr(
        parallel,
        "_dispatch_streaming_pipeline_if_requested",
        lambda **kwargs: pytest.fail("streaming should not run after region match"),
    )

    assert parallel._dispatch_requested_parallel_pipeline(
        model_path="model.json",
        region_parallel=True,
        region_size=100,
        skip_scaffolds=True,
        chroms={"chr1"},
        streaming_pipeline=True,
        chunk_size=10,
        max_reads=25,
        debug_timing=True,
        process_unmapped=True,
        pipeline_options=_pipeline_options(input_bam="in.bam"),
    ) == ("region", 1)

    assert calls[0][0] == "region"
    assert calls[0][1]["region_size"] == 100
    assert calls[0][1]["pipeline_options"].input_bam == "in.bam"


def test_requested_parallel_pipeline_falls_through_to_streaming(monkeypatch):
    calls = []

    monkeypatch.setattr(
        parallel,
        "_dispatch_region_parallel_if_requested",
        lambda **kwargs: calls.append(("region", kwargs)) or None,
    )
    monkeypatch.setattr(
        parallel,
        "_dispatch_streaming_pipeline_if_requested",
        lambda **kwargs: calls.append(("streaming", kwargs)) or ("streaming", 2),
    )

    assert parallel._dispatch_requested_parallel_pipeline(
        model_path="model.json",
        region_parallel=False,
        region_size=100,
        skip_scaffolds=False,
        chroms=None,
        streaming_pipeline=True,
        chunk_size=10,
        max_reads=25,
        debug_timing=True,
        process_unmapped=True,
        pipeline_options=_pipeline_options(input_bam="in.bam"),
    ) == ("streaming", 2)

    assert [call[0] for call in calls] == ["region", "streaming"]
    assert calls[1][1]["chunk_size"] == 10
    assert calls[1][1]["pipeline_options"].input_bam == "in.bam"


def test_region_dispatch_does_not_load_model_in_parent(monkeypatch):
    monkeypatch.setattr(
        parallel,
        "load_model",
        lambda path: (_ for _ in ()).throw(AssertionError("parent load")),
    )
    monkeypatch.setattr(
        parallel,
        "_process_bam_region_parallel",
        lambda **kwargs: ("region", kwargs["model_path"]),
    )

    assert parallel.process_bam_for_footprints(
        input_bam="in.bam",
        output_bam="out.bam",
        model_or_path="model.json",
        train_rids=set(),
        edge_trim=0,
        circular=False,
        mode="pacbio-fiber",
        context_size=3,
        msp_min_size=60,
        region_parallel=True,
    ) == ("region", "model.json")


def test_legacy_single_core_dispatch_loads_model_for_direct_processing(monkeypatch):
    loaded_model = object()
    monkeypatch.setattr(parallel, "load_model", lambda path: loaded_model)
    monkeypatch.setattr(parallel, "freeze_model_for_inference", lambda model: model)
    monkeypatch.setattr(
        parallel,
        "_process_bam_legacy_pipeline",
        lambda **kwargs: ("legacy", kwargs["model"], kwargs["model_path"]),
    )

    assert parallel.process_bam_for_footprints(
        input_bam="in.bam",
        output_bam="out.bam",
        model_or_path="model.json",
        train_rids=set(),
        edge_trim=0,
        circular=False,
        mode="pacbio-fiber",
        context_size=3,
        msp_min_size=60,
        n_cores=1,
    ) == ("legacy", loaded_model, "model.json")


def test_legacy_multicore_dispatch_defers_model_to_worker_initializer(monkeypatch):
    monkeypatch.setattr(
        parallel,
        "load_model",
        lambda path: (_ for _ in ()).throw(AssertionError("parent load")),
    )
    monkeypatch.setattr(
        parallel,
        "_process_bam_legacy_pipeline",
        lambda **kwargs: ("legacy", kwargs["model"], kwargs["model_path"]),
    )

    assert parallel.process_bam_for_footprints(
        input_bam="in.bam",
        output_bam="out.bam",
        model_or_path="model.json",
        train_rids=set(),
        edge_trim=0,
        circular=False,
        mode="pacbio-fiber",
        context_size=3,
        msp_min_size=60,
        n_cores=2,
    ) == ("legacy", None, "model.json")


def test_payload_worker_counts_per_read_failures(monkeypatch):
    def fake_extract(payload, mode, prob_threshold):
        if payload == "extract-bad":
            raise RuntimeError("bad payload")
        if payload == "empty":
            return None
        return {"payload": payload}

    def fake_process(fiber_read, *args, **kwargs):
        if fiber_read["payload"] == "process-bad":
            raise ValueError("bad read")
        return {"payload": fiber_read["payload"]}

    monkeypatch.setattr(streaming_workers, "extract_fiber_read_from_payload", fake_extract)
    monkeypatch.setattr(streaming_workers, "_process_single_read", fake_process)
    monkeypatch.setattr(streaming_workers, "_worker_model", object())

    chunk_result = streaming_workers._process_payload_chunk_worker(
        ["ok", "empty", "process-bad", "extract-bad"],
        edge_trim=0,
        circular=False,
        mode="pacbio-fiber",
        context_size=3,
        msp_min_size=60,
    )

    assert isinstance(chunk_result, WorkerChunkResult)
    assert chunk_result.read_failures == 2
    assert chunk_result.results == [{"payload": "ok"}, None, None, None]


def test_streaming_payload_fiber_read_result_maps_chimera(monkeypatch):
    payload = {"read_id": "read1"}
    fiber_read = {"query_sequence": "ACGT"}

    def fake_extract(got_payload, mode, prob_threshold):
        assert got_payload is payload
        assert mode == "pacbio-fiber"
        assert prob_threshold == 128
        return fiber_read

    monkeypatch.setattr(streaming_workers, "extract_fiber_read_from_payload", fake_extract)
    assert streaming_workers._payload_fiber_read_result(
        payload, "pacbio-fiber", 128,
    ) == fiber_read

    monkeypatch.setattr(
        streaming_workers, "extract_fiber_read_from_payload", lambda *_: None
    )
    assert streaming_workers._payload_fiber_read_result(
        payload, "pacbio-fiber", 128,
    ) is None

    monkeypatch.setattr(
        streaming_workers, "extract_fiber_read_from_payload", lambda *_: CHIMERA_SKIP
    )
    assert streaming_workers._payload_fiber_read_result(
        payload, "pacbio-fiber", 128,
    ) is None
    assert streaming_workers._payload_fiber_read_result(
        payload,
        "pacbio-fiber",
        128,
        chimera_result=streaming_workers.CHIMERA_RESULT,
    ) == streaming_workers.CHIMERA_RESULT

    def fail_extract(*_):
        raise ValueError("bad payload")

    monkeypatch.setattr(streaming_workers, "extract_fiber_read_from_payload", fail_extract)
    with pytest.raises(ValueError):
        streaming_workers._payload_fiber_read_result(payload, "pacbio-fiber", 128)


def test_fused_payload_fiber_read_result_preserves_picklable_chimera(monkeypatch):
    monkeypatch.setattr(
        streaming_workers,
        "extract_fiber_read_from_payload",
        lambda *_: CHIMERA_SKIP,
    )

    assert streaming_workers._fused_payload_fiber_read_result(
        {"read_id": "read1"},
        "pacbio-fiber",
        128,
    ) == streaming_workers.CHIMERA_RESULT


def test_run_worker_single_read_forwards_apply_arguments(monkeypatch):
    model = object()
    fiber_read = {"query_sequence": "ACGT"}
    seen = {}

    def fake_process(*args, **kwargs):
        seen["args"] = args
        seen["kwargs"] = kwargs
        return {"ok": True}

    monkeypatch.setattr(streaming_workers, "_worker_model", model)
    monkeypatch.setattr(streaming_workers, "_process_single_read", fake_process)

    assert streaming_workers._run_worker_single_read(
        fiber_read,
        edge_trim=1,
        circular=True,
        mode="pacbio-fiber",
        context_size=7,
        msp_min_size=60,
        nuc_min_size=85,
        with_scores=True,
        return_posteriors=True,
    ) == {"ok": True}
    assert seen["args"] == (
        fiber_read,
        model,
        1,
        True,
        "pacbio-fiber",
        7,
        60,
    )
    assert seen["kwargs"] == {
        "nuc_min_size": 85,
        "with_scores": True,
        "return_posteriors": True,
    }


def test_payload_worker_config_preserves_chunk_arguments():
    assert streaming_workers._payload_worker_config(
        edge_trim=1,
        circular=True,
        mode="pacbio-fiber",
        context_size=7,
        msp_min_size=60,
        nuc_min_size=85,
        with_scores=True,
        return_posteriors=False,
        prob_threshold=128,
    ) == streaming_workers._PayloadWorkerConfig(
        edge_trim=1,
        circular=True,
        mode="pacbio-fiber",
        context_size=7,
        msp_min_size=60,
        nuc_min_size=85,
        with_scores=True,
        return_posteriors=False,
        prob_threshold=128,
    )


def test_run_payload_configured_read_forwards_config(monkeypatch):
    fiber_read = {"query_sequence": "ACGT"}
    seen = {}

    def fake_run(*args):
        seen["args"] = args
        return {"apply": True}

    monkeypatch.setattr(streaming_workers, "_run_worker_single_read", fake_run)
    config = streaming_workers._PayloadWorkerConfig(
        edge_trim=1,
        circular=True,
        mode="pacbio-fiber",
        context_size=7,
        msp_min_size=60,
        nuc_min_size=85,
        with_scores=True,
        return_posteriors=False,
        prob_threshold=128,
    )

    assert streaming_workers._run_payload_configured_read(
        fiber_read, config,
    ) == {"apply": True}
    assert seen["args"] == (
        fiber_read,
        1,
        True,
        "pacbio-fiber",
        7,
        60,
        85,
        True,
        False,
    )


def test_process_payload_item_runs_parse_and_apply(monkeypatch):
    payload = {"read_id": "read1"}
    fiber_read = {"query_sequence": "ACGT"}
    seen = {}
    config = streaming_workers._PayloadWorkerConfig(
        edge_trim=1,
        circular=True,
        mode="pacbio-fiber",
        context_size=7,
        msp_min_size=60,
        nuc_min_size=85,
        with_scores=True,
        return_posteriors=True,
        prob_threshold=128,
    )

    def fake_extract(*args):
        seen["extract"] = args
        return fiber_read

    def fake_process(*args):
        seen["process"] = args
        return {"apply": True}

    monkeypatch.setattr(
        streaming_workers, "_payload_fiber_read_result", fake_extract,
    )
    monkeypatch.setattr(streaming_workers, "_run_worker_single_read", fake_process)

    assert streaming_workers._process_payload_item(
        payload,
        config,
    ) == {"apply": True}
    assert seen["extract"] == (payload, "pacbio-fiber", 128)
    assert seen["process"] == (
        fiber_read,
        1,
        True,
        "pacbio-fiber",
        7,
        60,
        85,
        True,
        True,
    )


def test_fused_recall_state_preserves_tables_and_thresholds():
    hit = object()
    miss = object()

    assert streaming_workers._fused_recall_state(
        hit,
        miss,
        recall_nucs=True,
        split_min_llr=4.5,
        split_min_opps=5,
        phase_nrl=185,
    ) == {
        "llr_hit": hit,
        "llr_miss": miss,
        "recall_nucs": True,
        "split_min_llr": 4.5,
        "split_min_opps": 5,
        "phase_nrl": 185,
    }


def test_worker_recall_options_uses_state_with_defaults(monkeypatch):
    monkeypatch.setattr(streaming_workers, "_worker_recall_state", {})
    assert streaming_workers._worker_recall_options(
        nuc_min_size=85,
        msp_min_size=20,
    ) == {
        "recall_nucs": False,
        "split_min_llr": 4.0,
        "split_min_opps": 3,
        "nuc_min_size": 85,
        "msp_min_size": 20,
        "phase_nrl": 0,
    }

    monkeypatch.setattr(streaming_workers, "_worker_recall_state", {
        "recall_nucs": True,
        "split_min_llr": 5.5,
        "split_min_opps": 7,
        "phase_nrl": 185,
    })
    assert streaming_workers._worker_recall_options(
        nuc_min_size=90,
        msp_min_size=25,
    ) == {
        "recall_nucs": True,
        "split_min_llr": 5.5,
        "split_min_opps": 7,
        "nuc_min_size": 90,
        "msp_min_size": 25,
        "phase_nrl": 185,
    }


def test_worker_recall_tables_reads_current_state(monkeypatch):
    hit = object()
    miss = object()
    monkeypatch.setattr(streaming_workers, "_worker_recall_state", {
        "llr_hit": hit,
        "llr_miss": miss,
    })

    assert streaming_workers._worker_recall_tables() == streaming_workers._WorkerRecallTables(
        llr_hit=hit,
        llr_miss=miss,
    )


def test_run_worker_fused_apply_stage_forwards_arguments(monkeypatch):
    model = object()
    fiber_read = {"query_sequence": "ACGT"}
    seen = {}

    def fake_apply(*args):
        seen["args"] = args
        return {"apply": True}

    monkeypatch.setattr(streaming_workers, "_worker_model", model)
    monkeypatch.setattr(streaming_workers, "run_hmm_apply_stage", fake_apply)

    assert streaming_workers._run_worker_fused_apply_stage_from_request(
        streaming_workers._WorkerFusedApplyStageRequest(
            fiber_read=fiber_read,
            edge_trim=1,
            circular=True,
            mode="pacbio-fiber",
            context_size=7,
            msp_min_size=60,
            nuc_min_size=85,
            with_scores=True,
        )
    ) == {"apply": True}
    assert seen["args"] == (
        fiber_read,
        model,
        1,
        True,
        "pacbio-fiber",
        7,
        60,
        85,
        True,
    )
    seen.clear()

    assert streaming_workers._run_worker_fused_apply_stage(
        fiber_read,
        edge_trim=1,
        circular=True,
        mode="pacbio-fiber",
        context_size=7,
        msp_min_size=60,
        nuc_min_size=85,
        with_scores=True,
    ) == {"apply": True}
    assert seen["args"][0] is fiber_read
    assert seen["args"][1] is model


def test_fused_payload_worker_config_preserves_chunk_arguments():
    assert streaming_workers._fused_payload_worker_config(
        edge_trim=1,
        circular=True,
        mode="pacbio-fiber",
        context_size=7,
        msp_min_size=60,
        nuc_min_size=85,
        with_scores=True,
        prob_threshold=128,
        min_llr=4.5,
        min_opps=5,
        unify_threshold=90,
    ) == streaming_workers._FusedPayloadWorkerConfig(
        edge_trim=1,
        circular=True,
        mode="pacbio-fiber",
        context_size=7,
        msp_min_size=60,
        nuc_min_size=85,
        with_scores=True,
        prob_threshold=128,
        min_llr=4.5,
        min_opps=5,
        unify_threshold=90,
    )


def test_run_fused_configured_apply_stage_forwards_config(monkeypatch):
    fiber_read = {"query_sequence": "ACGT"}
    seen = {}

    def fake_apply(request):
        seen["request"] = request
        return {"apply": True}

    monkeypatch.setattr(
        streaming_workers,
        "_run_worker_fused_apply_stage_from_request",
        fake_apply,
    )
    config = streaming_workers._FusedPayloadWorkerConfig(
        edge_trim=1,
        circular=True,
        mode="pacbio-fiber",
        context_size=7,
        msp_min_size=60,
        nuc_min_size=85,
        with_scores=True,
        prob_threshold=128,
        min_llr=4.5,
        min_opps=5,
        unify_threshold=90,
    )

    assert streaming_workers._run_fused_configured_apply_stage_from_request(
        streaming_workers._FusedConfiguredApplyStageRequest(
            fiber_read=fiber_read,
            config=config,
        )
    ) == {"apply": True}
    assert seen["request"] == streaming_workers._WorkerFusedApplyStageRequest(
        fiber_read=fiber_read,
        edge_trim=1,
        circular=True,
        mode="pacbio-fiber",
        context_size=7,
        msp_min_size=60,
        nuc_min_size=85,
        with_scores=True,
    )
    assert seen["request"].fiber_read is fiber_read
    seen.clear()

    assert streaming_workers._run_fused_configured_apply_stage(
        fiber_read,
        config,
    ) == {"apply": True}
    assert seen["request"].fiber_read is fiber_read


def test_build_worker_fused_recall_result_forwards_options(monkeypatch):
    fiber_read = {"query_sequence": "ACGT"}
    apply_result = {"ns": [1]}
    hit = object()
    miss = object()
    seen = {}

    def fake_build(*args, **kwargs):
        seen["args"] = args
        seen["kwargs"] = kwargs
        return {"recall": True}

    monkeypatch.setattr(streaming_workers, "build_fused_recall_result", fake_build)
    monkeypatch.setattr(streaming_workers, "_worker_recall_state", {
        "recall_nucs": True,
        "split_min_llr": 5.5,
        "split_min_opps": 7,
        "phase_nrl": 185,
    })

    assert streaming_workers._build_worker_fused_recall_result_from_request(
        streaming_workers._WorkerFusedRecallResultRequest(
            fiber_read=fiber_read,
            apply_result=apply_result,
            llr_hit=hit,
            llr_miss=miss,
            min_llr=4.0,
            min_opps=3,
            unify_threshold=90,
            with_scores=True,
            nuc_min_size=85,
            msp_min_size=20,
        )
    ) == {"recall": True}
    assert seen["args"] == (
        fiber_read,
        apply_result,
        hit,
        miss,
        4.0,
        3,
        90,
        True,
    )
    assert seen["kwargs"] == {
        "recall_nucs": True,
        "split_min_llr": 5.5,
        "split_min_opps": 7,
        "nuc_min_size": 85,
        "msp_min_size": 20,
        "phase_nrl": 185,
    }
    seen.clear()

    assert streaming_workers._build_worker_fused_recall_result(
        fiber_read,
        apply_result,
        hit,
        miss,
        min_llr=4.0,
        min_opps=3,
        unify_threshold=90,
        with_scores=True,
        nuc_min_size=85,
        msp_min_size=20,
    ) == {"recall": True}
    assert seen["args"][0] is fiber_read
    assert seen["args"][1] is apply_result


def test_build_fused_configured_recall_result_forwards_config(monkeypatch):
    fiber_read = {"query_sequence": "ACGT"}
    apply_result = {"ns": [1]}
    hit = object()
    miss = object()
    seen = {}

    def fake_build(request):
        seen["request"] = request
        return {"recall": True}

    monkeypatch.setattr(
        streaming_workers, "_build_worker_fused_recall_result_from_request", fake_build,
    )
    config = streaming_workers._FusedPayloadWorkerConfig(
        edge_trim=1,
        circular=True,
        mode="pacbio-fiber",
        context_size=7,
        msp_min_size=60,
        nuc_min_size=85,
        with_scores=True,
        prob_threshold=128,
        min_llr=4.5,
        min_opps=5,
        unify_threshold=90,
    )

    assert streaming_workers._build_fused_configured_recall_result_from_request(
        streaming_workers._FusedConfiguredRecallResultRequest(
            fiber_read=fiber_read,
            apply_result=apply_result,
            llr_hit=hit,
            llr_miss=miss,
            config=config,
        )
    ) == {"recall": True}
    assert seen["request"] == streaming_workers._WorkerFusedRecallResultRequest(
        fiber_read=fiber_read,
        apply_result=apply_result,
        llr_hit=hit,
        llr_miss=miss,
        min_llr=4.5,
        min_opps=5,
        unify_threshold=90,
        with_scores=True,
        nuc_min_size=85,
        msp_min_size=60,
    )
    assert seen["request"].fiber_read is fiber_read
    assert seen["request"].apply_result is apply_result
    seen.clear()

    assert streaming_workers._build_fused_configured_recall_result(
        fiber_read,
        apply_result,
        hit,
        miss,
        config,
    ) == {"recall": True}
    assert seen["request"].fiber_read is fiber_read


def test_process_fused_fiber_read_applies_and_recalls_or_passes_through(monkeypatch):
    fiber_read = {"query_sequence": "ACGT"}
    apply_result = {"ns": [1]}
    hit = object()
    miss = object()
    seen = {}
    config = streaming_workers._FusedPayloadWorkerConfig(
        edge_trim=1,
        circular=True,
        mode="pacbio-fiber",
        context_size=7,
        msp_min_size=60,
        nuc_min_size=85,
        with_scores=True,
        prob_threshold=128,
        min_llr=4.5,
        min_opps=5,
        unify_threshold=90,
    )

    def fake_apply(request):
        seen["apply"] = request
        return apply_result

    def fake_build(request):
        seen["build"] = request
        return {"recall": True}

    monkeypatch.setattr(
        streaming_workers, "_run_fused_configured_apply_stage_from_request", fake_apply,
    )
    monkeypatch.setattr(
        streaming_workers,
        "_build_fused_configured_recall_result_from_request",
        fake_build,
    )
    monkeypatch.setattr(
        streaming_workers, "apply_result_has_footprints", lambda result: True,
    )

    assert streaming_workers._process_fused_fiber_read_from_request(
        streaming_workers._FusedFiberReadProcessRequest(
            fiber_read=fiber_read,
            config=config,
            llr_hit=hit,
            llr_miss=miss,
        )
    ) == {"recall": True}
    assert seen["apply"] == streaming_workers._FusedConfiguredApplyStageRequest(
        fiber_read=fiber_read,
        config=config,
    )
    assert seen["build"] == streaming_workers._FusedConfiguredRecallResultRequest(
        fiber_read=fiber_read,
        apply_result=apply_result,
        llr_hit=hit,
        llr_miss=miss,
        config=config,
    )
    assert seen["apply"].fiber_read is fiber_read
    assert seen["build"].apply_result is apply_result

    monkeypatch.setattr(
        streaming_workers, "apply_result_has_footprints", lambda result: False,
    )
    assert streaming_workers._process_fused_fiber_read(
        fiber_read, config, hit, miss,
    ) is None


def test_process_fused_payload_item_runs_parse_apply_and_recall(monkeypatch):
    payload = {"read_id": "read1"}
    fiber_read = {"query_sequence": "ACGT"}
    apply_result = {"ns": [1]}
    hit = object()
    miss = object()
    seen = {}

    config = streaming_workers._FusedPayloadWorkerConfig(
        edge_trim=1,
        circular=True,
        mode="pacbio-fiber",
        context_size=7,
        msp_min_size=60,
        nuc_min_size=85,
        with_scores=True,
        prob_threshold=128,
        min_llr=4.0,
        min_opps=3,
        unify_threshold=90,
    )

    def fake_extract(*args):
        seen["extract"] = args
        return fiber_read

    def fake_apply(request):
        seen["apply"] = request
        return apply_result

    def fake_build(request):
        seen["build"] = request
        return {"recall": True}

    monkeypatch.setattr(
        streaming_workers, "_fused_payload_fiber_read_result", fake_extract,
    )
    monkeypatch.setattr(
        streaming_workers, "_run_worker_fused_apply_stage_from_request", fake_apply,
    )
    monkeypatch.setattr(
        streaming_workers, "apply_result_has_footprints", lambda result: True,
    )
    monkeypatch.setattr(
        streaming_workers, "_build_worker_fused_recall_result_from_request", fake_build,
    )

    assert streaming_workers._process_fused_payload_item_from_request(
        streaming_workers._FusedPayloadItemProcessRequest(
            payload=payload,
            config=config,
            llr_hit=hit,
            llr_miss=miss,
        )
    ) == {"recall": True}
    assert seen["extract"] == (payload, "pacbio-fiber", 128)
    assert seen["apply"] == streaming_workers._WorkerFusedApplyStageRequest(
        fiber_read=fiber_read,
        edge_trim=1,
        circular=True,
        mode="pacbio-fiber",
        context_size=7,
        msp_min_size=60,
        nuc_min_size=85,
        with_scores=True,
    )
    assert seen["apply"].fiber_read is fiber_read
    assert seen["build"] == streaming_workers._WorkerFusedRecallResultRequest(
        fiber_read=fiber_read,
        apply_result=apply_result,
        llr_hit=hit,
        llr_miss=miss,
        min_llr=4.0,
        min_opps=3,
        unify_threshold=90,
        with_scores=True,
        nuc_min_size=85,
        msp_min_size=60,
    )
    assert seen["build"].fiber_read is fiber_read
    assert seen["build"].apply_result is apply_result
    seen.clear()

    assert streaming_workers._process_fused_payload_item(
        payload,
        config,
        hit,
        miss,
    ) == {"recall": True}
    assert seen["extract"] == (payload, "pacbio-fiber", 128)


def test_fused_payload_worker_counts_per_read_failures(monkeypatch):
    def fake_extract(payload, mode, prob_threshold):
        if payload == "extract-bad":
            raise RuntimeError("bad payload")
        return {"payload": payload, "query_sequence": "ACGT"}

    def fake_apply(fiber_read, *args, **kwargs):
        if fiber_read["payload"] == "apply-bad":
            raise ValueError("bad apply")
        return {"payload": fiber_read["payload"]}

    def fake_has_footprints(apply_result):
        return apply_result["payload"] != "empty"

    def fake_build(fiber_read, *args, **kwargs):
        if fiber_read["payload"] == "build-bad":
            raise RuntimeError("bad recall")
        return {"payload": fiber_read["payload"], "tf_calls": []}

    monkeypatch.setattr(streaming_workers, "extract_fiber_read_from_payload", fake_extract)
    monkeypatch.setattr(streaming_workers, "run_hmm_apply_stage", fake_apply)
    monkeypatch.setattr(streaming_workers, "apply_result_has_footprints", fake_has_footprints)
    monkeypatch.setattr(streaming_workers, "build_fused_recall_result", fake_build)
    monkeypatch.setattr(streaming_workers, "_worker_model", object())
    monkeypatch.setattr(streaming_workers, "_worker_recall_state", {
        "llr_hit": object(),
        "llr_miss": object(),
    })

    chunk_result = streaming_workers._process_fused_payload_chunk_worker(
        ["ok", "empty", "apply-bad", "build-bad", "extract-bad"],
        edge_trim=0,
        circular=False,
        mode="pacbio-fiber",
        context_size=3,
        msp_min_size=60,
    )

    assert isinstance(chunk_result, WorkerChunkResult)
    assert chunk_result.read_failures == 3
    assert chunk_result.results == [
        {"payload": "ok", "tf_calls": []},
        None,
        None,
        None,
        None,
    ]


def test_streaming_drain_counter_helpers_write_passthrough():
    read = object()
    outbam = _OutBam()
    counters = {}

    streaming_drain._increment_counter(counters, "written", 2)
    streaming_drain._record_worker_failures(counters, 3)
    streaming_drain._record_reads_with_footprints(counters)
    streaming_drain._record_no_footprints(counters)
    streaming_drain._record_chimera(counters)
    streaming_drain._write_passthrough(outbam, read, counters)

    assert counters == {
        "written": 3,
        "worker_failures": 3,
        "reads_with_footprints": 1,
        "no_footprints": 1,
        "chimera": 1,
    }
    assert outbam.written == [read]


def test_empty_ref_positions_returns_int32_empty_array():
    empty = streaming_drain._empty_ref_positions()

    assert empty.dtype == np.int32
    assert empty.size == 0


def test_fused_read_length_uses_query_sequence_when_present():
    assert (
        streaming_drain._fused_read_length(SimpleNamespace(query_sequence="ACGT"))
        == 4
    )
    assert (
        streaming_drain._fused_read_length(SimpleNamespace(query_sequence=None))
        == 0
    )


def test_posterior_ref_positions_handles_backend_fallbacks(monkeypatch):
    read = object()
    monkeypatch.setattr(streaming_drain, "HAS_POSTERIOR_WRITER", False)
    empty = streaming_drain._posterior_ref_positions(read)
    assert empty.dtype == np.int32
    assert empty.size == 0

    positions = np.asarray([1, 2, 3], dtype=np.int32)
    monkeypatch.setattr(streaming_drain, "HAS_POSTERIOR_WRITER", True)
    monkeypatch.setattr(
        streaming_drain, "get_ref_positions_from_read", lambda got_read: positions
    )
    assert streaming_drain._posterior_ref_positions(read) is positions

    def fail_positions(_):
        raise RuntimeError("bad read")

    monkeypatch.setattr(streaming_drain, "get_ref_positions_from_read", fail_positions)
    empty = streaming_drain._posterior_ref_positions(read)
    assert empty.dtype == np.int32
    assert empty.size == 0


def test_result_has_posteriors_requires_non_none_posteriors():
    assert streaming_drain._result_has_posteriors({"posteriors": [0.1]})
    assert not streaming_drain._result_has_posteriors({"posteriors": None})
    assert not streaming_drain._result_has_posteriors({})


def test_posterior_chrom_reads_reference_name():
    assert streaming_drain._posterior_chrom(
        SimpleNamespace(reference_name="chr1"),
    ) == "chr1"
    assert streaming_drain._posterior_chrom(
        SimpleNamespace(reference_name=None),
    ) is None


def test_add_posterior_fiber_if_available_guards_and_writes(monkeypatch):
    read = SimpleNamespace(reference_name="chr1")
    result = {"posteriors": [0.1]}
    seen = []

    class Writer:
        def __bool__(self):
            return False

        def add_fiber(self, chrom, data):
            seen.append((chrom, data))

    monkeypatch.setattr(
        streaming_drain,
        "_posterior_ref_positions",
        lambda got_read: ["ref-pos"],
    )
    monkeypatch.setattr(
        streaming_drain,
        "posterior_fiber_data",
        lambda got_read, got_result, ref_positions: {
            "read": got_read,
            "result": got_result,
            "ref_positions": ref_positions,
        },
    )

    assert streaming_drain._add_posterior_fiber_from_request(
        streaming_drain._PosteriorFiberAddRequest(
            posterior_writer=Writer(),
            read_obj=read,
            result=result,
        )
    )
    assert seen == [(
        "chr1",
        {
            "read": read,
            "result": result,
            "ref_positions": ["ref-pos"],
        },
    )]
    assert not streaming_drain._add_posterior_fiber_if_available(None, read, result)
    assert not streaming_drain._add_posterior_fiber_if_available(
        Writer(), read, {"posteriors": None},
    )
    assert not streaming_drain._add_posterior_fiber_if_available(
        Writer(), SimpleNamespace(reference_name=None), result,
    )


def test_pop_inflight_chunk_coerces_worker_result_and_removes_entry():
    read = object()
    chunk_items = [{"payload": "ok"}]
    skip_flags = [False]
    inflight = deque([
        streaming_drain._SubmittedChunk(
            future=_done_future(WorkerChunkResult([{"ok": True}], read_failures=2)),
            read_objs=[read],
            items=chunk_items,
            skip_flags=skip_flags,
        )
    ])

    assert streaming_drain._pop_inflight_chunk(
        inflight
    ) == streaming_drain._InflightChunk(
        results=[{"ok": True}],
        worker_failures=2,
        read_objs=[read],
        items=chunk_items,
        skip_flags=skip_flags,
    )
    assert not inflight


def test_drain_chunk_in_order_interleaves_skips_and_results():
    skipped_read = object()
    processed_read = object()
    result = {"ns": [1]}
    outbam = _OutBam()
    counters = {}
    recorded = []

    streaming_drain._drain_chunk_in_order_from_request(
        streaming_drain._DrainChunkInOrderRequest(
            chunk_read_objs=[skipped_read, processed_read],
            chunk_items=[{"payload": "processed"}],
            chunk_skip_flags=[True, False],
            results=[result],
            outbam=outbam,
            counters=counters,
            record_result=lambda read, got_result: recorded.append(
                (read, got_result)
            ),
        )
    )

    assert outbam.written == [skipped_read, processed_read]
    assert counters == {"written": 2}
    assert recorded == [(processed_read, result)]


def test_drain_oldest_with_recorder_records_failures_and_order():
    skipped_read = object()
    processed_read = object()
    result = {"ns": [1]}
    outbam = _OutBam()
    counters = {}
    recorded = []
    inflight = deque([
        streaming_drain._SubmittedChunk(
            future=_done_future(WorkerChunkResult([result], read_failures=2)),
            read_objs=[skipped_read, processed_read],
            items=[{"payload": "processed"}],
            skip_flags=[True, False],
        )
    ])

    streaming_drain._drain_oldest_with_recorder_from_request(
        streaming_drain._DrainOldestWithRecorderRequest(
            inflight=inflight,
            outbam=outbam,
            counters=counters,
            record_result=lambda read, got_result: recorded.append(
                (read, got_result)
            ),
        )
    )

    assert not inflight
    assert outbam.written == [skipped_read, processed_read]
    assert counters == {"worker_failures": 2, "written": 2}
    assert recorded == [(processed_read, result)]


def test_record_apply_result_tags_posteriors_or_counts_no_footprints(monkeypatch):
    read = object()
    result = {"ns": [1]}
    counters = {}
    tagged = []
    posteriors = []

    def fake_set_tags(got_read, got_result, with_scores, write_msps):
        tagged.append((got_read, got_result, with_scores, write_msps))

    def fake_add_posterior(writer, got_read, got_result):
        posteriors.append((writer, got_read, got_result))
        return True

    monkeypatch.setattr(streaming_drain, "set_legacy_apply_tags", fake_set_tags)
    monkeypatch.setattr(
        streaming_drain,
        "_add_posterior_fiber_if_available",
        fake_add_posterior,
    )

    streaming_drain._record_apply_result_from_request(
        streaming_drain._ApplyResultRecordRequest(
            read_obj=read,
            result=result,
            with_scores=True,
            write_msps=False,
            posterior_writer="writer",
            counters=counters,
        )
    )
    streaming_drain._record_apply_result(
        read, None, with_scores=True, write_msps=False,
        posterior_writer="writer", counters=counters,
    )

    assert tagged == [(read, result, True, False)]
    assert posteriors == [("writer", read, result)]
    assert counters == {"reads_with_footprints": 1, "no_footprints": 1}


def test_record_fused_result_tags_chimeras_or_counts_no_footprints(monkeypatch):
    read = SimpleNamespace(query_sequence="ACGT")
    result = {"tf_calls": []}
    counters = {}
    tagged = []

    def fake_write_tags(
        got_read,
        read_length,
        result,
        also_write_legacy,
        downstream_compat,
    ):
        tagged.append((
            got_read,
            read_length,
            result,
            also_write_legacy,
            downstream_compat,
        ))

    monkeypatch.setattr(streaming_drain, "write_fused_recall_tags", fake_write_tags)

    streaming_drain._record_fused_result_from_request(
        streaming_drain._FusedResultRecordRequest(
            read_obj=read,
            result=result,
            also_write_legacy=True,
            downstream_compat=False,
            counters=counters,
        )
    )
    streaming_drain._record_fused_result(
        read, streaming_workers.CHIMERA_RESULT,
        also_write_legacy=True, downstream_compat=False, counters=counters,
    )
    streaming_drain._record_fused_result(
        read, None, also_write_legacy=True, downstream_compat=False,
        counters=counters,
    )

    assert tagged == [(read, 4, result, True, False)]
    assert counters == {
        "reads_with_footprints": 1,
        "chimera": 1,
        "no_footprints": 1,
    }


def test_streaming_drain_counts_worker_failures_and_passes_read_through():
    read = object()
    outbam = _OutBam()
    counters = {"reads_with_footprints": 0, "no_footprints": 0, "written": 0}
    inflight = deque([
        streaming_drain._SubmittedChunk(
            future=_done_future(WorkerChunkResult([None], read_failures=1)),
            read_objs=[read],
            items=[{"payload": "bad"}],
            skip_flags=[False],
        )
    ])

    streaming_drain._drain_oldest_chunk_from_request(
        streaming_drain._DrainOldestApplyChunkRequest(
            inflight=inflight,
            outbam=outbam,
            with_scores=False,
            write_msps=True,
            posterior_writer=None,
            counters=counters,
        )
    )

    assert counters == {
        "reads_with_footprints": 0,
        "no_footprints": 1,
        "worker_failures": 1,
        "written": 1,
    }
    assert outbam.written == [read]


def test_fused_drain_counts_worker_failures_and_passes_read_through():
    read = object()
    outbam = _OutBam()
    counters = {"reads_with_footprints": 0, "no_footprints": 0, "written": 0}
    inflight = deque([
        streaming_drain._SubmittedChunk(
            future=_done_future(WorkerChunkResult([None], read_failures=1)),
            read_objs=[read],
            items=[{"payload": "bad"}],
            skip_flags=[False],
        )
    ])

    streaming_drain._drain_oldest_fused_chunk_from_request(
        streaming_drain._DrainOldestFusedChunkRequest(
            inflight=inflight,
            outbam=outbam,
            with_scores=False,
            also_write_legacy=True,
            downstream_compat=True,
            counters=counters,
        )
    )

    assert counters == {
        "reads_with_footprints": 0,
        "no_footprints": 1,
        "worker_failures": 1,
        "written": 1,
    }
    assert outbam.written == [read]


class TestIsMainChromosome:
    """Parametrized tests for chromosome filtering."""

    @pytest.mark.parametrize("chrom", [
        "chr1", "chr2", "chr10", "chr22",
        "chrX", "chrY", "chrM",
        "1", "2", "10", "22",
        "X", "Y", "M", "MT",
    ])
    def test_human_main_chromosomes(self, chrom):
        assert _is_main_chromosome(chrom) is True

    @pytest.mark.parametrize("chrom", [
        "2L", "2R", "3L", "3R", "4",
        "chr2L", "chr2R", "chr3L", "chr3R", "chr4",
    ])
    def test_drosophila_chromosomes(self, chrom):
        assert _is_main_chromosome(chrom) is True

    @pytest.mark.parametrize("chrom", [
        "I", "II", "III", "IV", "V", "VI",
    ])
    def test_c_elegans_chromosomes(self, chrom):
        assert _is_main_chromosome(chrom) is True

    @pytest.mark.parametrize("chrom", [
        "chrUn_gl000220", "chr1_random",
        "scaffold_1", "contig_100",
        "chr1_gl000191_random",
        "chrUn_KI270442v1",
    ])
    def test_scaffolds_and_contigs_rejected(self, chrom):
        assert _is_main_chromosome(chrom) is False

    @pytest.mark.parametrize("chrom", [
        "chr1_KI270706v1_random",
        "chr6_GL000256v2_alt",
        "chrUn_GL000220v1",
    ])
    def test_alt_and_fix_rejected(self, chrom):
        assert _is_main_chromosome(chrom) is False

    def test_case_insensitive(self):
        """Chromosome name matching should be case-insensitive."""
        assert _is_main_chromosome("CHR1") is True
        assert _is_main_chromosome("Chr1") is True
        assert _is_main_chromosome("chr1") is True

    def test_empty_string(self):
        """Empty string should not be a main chromosome."""
        assert _is_main_chromosome("") is False
