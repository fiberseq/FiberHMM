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
    _drain_oldest_chunk,
    _drain_oldest_fused_chunk,
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


def test_worker_chunk_result_coerces_legacy_lists():
    assert coerce_worker_chunk_result([None]) == ([None], 0)
    assert coerce_worker_chunk_result(WorkerChunkResult([None], 2)) == ([None], 2)


def test_parallel_reexports_multiprocessing_context():
    assert parallel._select_mp_context is mp_context._select_mp_context
    assert parallel._MP_CONTEXT is mp_context._MP_CONTEXT


def test_model_and_path_for_processing_splits_path_and_loaded_model():
    assert _model_and_path_for_processing("model.json") == (None, "model.json")

    model = object()
    assert _model_and_path_for_processing(model) == (model, None)


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

    fiber_read, reason = legacy_pipeline._legacy_fiber_read_or_skip(
        _streaming_read(), config, "daf", 128,
    )
    assert fiber_read == {"read_id": "read1"}
    assert reason is None
    assert extracted == [("read1", "daf", 128)]

    fiber_read, reason = legacy_pipeline._legacy_fiber_read_or_skip(
        _streaming_read(mapping_quality=0), config, "daf", 128,
    )
    assert fiber_read is None
    assert reason == "low_mapq"
    assert extracted == [("read1", "daf", 128)]

    monkeypatch.setattr(
        legacy_pipeline,
        "_extract_fiber_read_from_pysam",
        lambda read, mode, prob_threshold: None,
    )
    fiber_read, reason = legacy_pipeline._legacy_fiber_read_or_skip(
        _streaming_read(), config, "daf", 128,
    )
    assert fiber_read is None
    assert reason == "no_modifications"

    def fail_extract(read, mode, prob_threshold):
        raise ValueError("bad read")

    monkeypatch.setattr(legacy_pipeline, "_extract_fiber_read_from_pysam", fail_extract)
    fiber_read, reason = legacy_pipeline._legacy_fiber_read_or_skip(
        _streaming_read(), config, "daf", 128,
    )
    assert fiber_read is None
    assert reason == "extraction_failed"


def test_streaming_payload_or_skip_filters_and_builds_payload(monkeypatch):
    config = ReadFilterConfig(min_mapq=20, min_read_length=50)
    built = []

    def fake_make_payload(read, mode, ref_fasta):
        built.append((read.query_name, mode, ref_fasta))
        return {"read_id": read.query_name}

    monkeypatch.setattr(streaming_pipeline, "make_apply_payload", fake_make_payload)

    payload, reason = streaming_pipeline._streaming_payload_or_skip(
        _streaming_read(), config, "daf", ref_fasta="ref.fa",
    )
    assert payload == {"read_id": "read1"}
    assert reason is None
    assert built == [("read1", "daf", "ref.fa")]

    payload, reason = streaming_pipeline._streaming_payload_or_skip(
        _streaming_read(mapping_quality=0), config, "daf",
    )
    assert payload is None
    assert reason == "low_mapq"
    assert built == [("read1", "daf", "ref.fa")]

    monkeypatch.setattr(
        streaming_pipeline, "make_apply_payload",
        lambda read, mode, ref_fasta: None,
    )
    payload, reason = streaming_pipeline._streaming_payload_or_skip(
        _streaming_read(), config, "daf",
    )
    assert payload is None
    assert reason == "no_modifications"


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

    future, read_objs, chunk_items, skip_flags = inflight.pop()
    assert executor.submitted == []
    assert future.result() == []
    assert read_objs == ["read"]
    assert chunk_items == []
    assert skip_flags == [True]


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

    future, read_objs, chunk_items, skip_flags = inflight.pop()
    assert executor.submitted == [(worker_fn, (["payload"], "arg"))]
    assert future.result() == ("submitted", worker_fn, (["payload"], "arg"))
    assert read_objs == ["read"]
    assert chunk_items == ["payload"]
    assert skip_flags == [False]


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

    assert streaming_workers._worker_recall_tables() == (hit, miss)


def test_run_worker_fused_apply_stage_forwards_arguments(monkeypatch):
    model = object()
    fiber_read = {"query_sequence": "ACGT"}
    seen = {}

    def fake_apply(*args):
        seen["args"] = args
        return {"apply": True}

    monkeypatch.setattr(streaming_workers, "_worker_model", model)
    monkeypatch.setattr(streaming_workers, "run_hmm_apply_stage", fake_apply)

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

    assert streaming_drain._add_posterior_fiber_if_available(
        Writer(), read, result,
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


def test_streaming_drain_counts_worker_failures_and_passes_read_through():
    read = object()
    outbam = _OutBam()
    counters = {"reads_with_footprints": 0, "no_footprints": 0, "written": 0}
    inflight = deque([(
        _done_future(WorkerChunkResult([None], read_failures=1)),
        [read],
        [{"payload": "bad"}],
        [False],
    )])

    _drain_oldest_chunk(inflight, outbam, False, True, None, counters)

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
    inflight = deque([(
        _done_future(WorkerChunkResult([None], read_failures=1)),
        [read],
        [{"payload": "bad"}],
        [False],
    )])

    _drain_oldest_fused_chunk(inflight, outbam, False, True, True, counters)

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
