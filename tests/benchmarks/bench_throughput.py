"""
Throughput benchmarks: reads/second for each processing mode.
"""
import time
import pytest

from fiberhmm.inference.parallel import process_bam_for_footprints

pytestmark = pytest.mark.benchmark


def _run_and_time(bam_path, model_path, output_path, **kwargs):
    """Run processing and return (total_reads, elapsed_seconds)."""
    start = time.perf_counter()
    total, with_fp = process_bam_for_footprints(
        input_bam=bam_path, output_bam=output_path,
        model_or_path=model_path,
        train_rids=set(), edge_trim=10, circular=False,
        mode='pacbio-fiber', context_size=3, msp_min_size=60,
        min_mapq=0, min_read_length=0, prob_threshold=0,
        **kwargs,
    )
    elapsed = time.perf_counter() - start
    return total, with_fp, elapsed


class TestThroughput:
    """Throughput comparison across processing modes."""

    def test_streaming_1_core(self, bench_bam_medium, bench_model_path, tmp_path):
        """Streaming pipeline, 1 core."""
        total, fp, elapsed = _run_and_time(
            bench_bam_medium, bench_model_path, str(tmp_path / "out.bam"),
            n_cores=1, streaming_pipeline=True, chunk_size=500,
        )
        rate = total / elapsed
        print(f"\n  Streaming 1-core: {total} reads in {elapsed:.1f}s = {rate:.0f} reads/s")
        assert total > 0

    def test_streaming_2_cores(self, bench_bam_medium, bench_model_path, tmp_path):
        """Streaming pipeline, 2 cores."""
        total, fp, elapsed = _run_and_time(
            bench_bam_medium, bench_model_path, str(tmp_path / "out.bam"),
            n_cores=2, streaming_pipeline=True, chunk_size=500,
        )
        rate = total / elapsed
        print(f"\n  Streaming 2-core: {total} reads in {elapsed:.1f}s = {rate:.0f} reads/s")
        assert total > 0

    def test_streaming_4_cores(self, bench_bam_medium, bench_model_path, tmp_path):
        """Streaming pipeline, 4 cores."""
        total, fp, elapsed = _run_and_time(
            bench_bam_medium, bench_model_path, str(tmp_path / "out.bam"),
            n_cores=4, streaming_pipeline=True, chunk_size=500,
        )
        rate = total / elapsed
        print(f"\n  Streaming 4-core: {total} reads in {elapsed:.1f}s = {rate:.0f} reads/s")
        assert total > 0

    def test_region_parallel_2_cores(self, bench_bam_medium, bench_model_path, tmp_path):
        """Region-parallel, 2 cores."""
        total, fp, elapsed = _run_and_time(
            bench_bam_medium, bench_model_path, str(tmp_path / "out.bam"),
            n_cores=2, region_parallel=True,
        )
        rate = total / elapsed
        print(f"\n  Region-parallel 2-core: {total} reads in {elapsed:.1f}s = {rate:.0f} reads/s")
        assert total > 0

    def test_region_parallel_4_cores(self, bench_bam_medium, bench_model_path, tmp_path):
        """Region-parallel, 4 cores."""
        total, fp, elapsed = _run_and_time(
            bench_bam_medium, bench_model_path, str(tmp_path / "out.bam"),
            n_cores=4, region_parallel=True,
        )
        rate = total / elapsed
        print(f"\n  Region-parallel 4-core: {total} reads in {elapsed:.1f}s = {rate:.0f} reads/s")
        assert total > 0

    def test_legacy_chunk_1_core(self, bench_bam_medium, bench_model_path, tmp_path):
        """Legacy chunk mode, 1 core (baseline)."""
        total, fp, elapsed = _run_and_time(
            bench_bam_medium, bench_model_path, str(tmp_path / "out.bam"),
            n_cores=1, region_parallel=False, streaming_pipeline=False,
        )
        rate = total / elapsed
        print(f"\n  Legacy 1-core: {total} reads in {elapsed:.1f}s = {rate:.0f} reads/s")
        assert total > 0
