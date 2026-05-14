"""
Tests for fiberhmm.inference.parallel module — region splitting and chromosome filtering.
"""
from collections import deque
from concurrent.futures import Future

import pytest

import fiberhmm.inference.mp_context as mp_context
import fiberhmm.inference.parallel as parallel
import fiberhmm.inference.region_pipeline as region_pipeline
import fiberhmm.inference.region_workers as region_workers
import fiberhmm.inference.streaming_pipeline as streaming_pipeline
import fiberhmm.inference.streaming_workers as streaming_workers
from fiberhmm.inference.parallel import (
    _drain_oldest_chunk,
    _drain_oldest_fused_chunk,
    _is_main_chromosome,
)
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


def test_worker_chunk_result_coerces_legacy_lists():
    assert coerce_worker_chunk_result([None]) == ([None], 0)
    assert coerce_worker_chunk_result(WorkerChunkResult([None], 2)) == ([None], 2)


def test_parallel_reexports_multiprocessing_context():
    assert parallel._select_mp_context is mp_context._select_mp_context
    assert parallel._MP_CONTEXT is mp_context._MP_CONTEXT


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
