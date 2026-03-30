"""
I/O vs compute breakdown: identify bottlenecks.
"""
import time
import pytest
import pysam
import numpy as np

pytestmark = pytest.mark.benchmark


class TestIOvsCompute:
    """Measure I/O time vs HMM compute time to identify bottleneck."""

    def test_component_breakdown(self, bench_bam_medium, bench_model_path, tmp_path):
        """
        Time each pipeline component separately:
        1. BAM read + extract (I/O bound)
        2. HMM compute only (CPU bound)
        3. Full pipeline
        """
        from fiberhmm.inference.engine import _extract_fiber_read_from_pysam, _process_single_read
        from fiberhmm.core.model_io import load_model

        # Step 1: Read-only timing (iterate + extract fiber reads)
        start = time.perf_counter()
        fiber_reads = []
        pysam.set_verbosity(0)
        with pysam.AlignmentFile(bench_bam_medium, "rb") as bam:
            for read in bam:
                if not read.is_unmapped:
                    fr = _extract_fiber_read_from_pysam(read, 'pacbio-fiber', 0)
                    if fr:
                        fiber_reads.append(fr)
        read_time = time.perf_counter() - start

        # Step 2: Compute-only timing (HMM inference on cached reads)
        model = load_model(bench_model_path)
        start = time.perf_counter()
        for fr in fiber_reads:
            _process_single_read(
                fr, model, edge_trim=10, circular=False,
                mode='pacbio-fiber', context_size=3, msp_min_size=60,
                with_scores=False,
            )
        compute_time = time.perf_counter() - start

        # Print breakdown
        total = read_time + compute_time
        n = len(fiber_reads)
        print(f"\n  I/O vs Compute Breakdown ({n} reads):")
        print(f"    BAM read + extract: {read_time:.2f}s ({100*read_time/total:.1f}%) "
              f"= {n/read_time:.0f} reads/s")
        print(f"    HMM compute:        {compute_time:.2f}s ({100*compute_time/total:.1f}%) "
              f"= {n/compute_time:.0f} reads/s")
        print(f"    Sum:                {total:.2f}s = {n/total:.0f} reads/s")

        assert n > 0


class TestSlowIOSimulation:
    """Simulate slow I/O to demonstrate streaming pipeline advantage."""

    def test_slow_io_theory(self, bench_bam_medium, bench_model_path, tmp_path):
        """
        With slow I/O, streaming pipeline should approach:
            total_time ~ max(io_time, compute_time)
        instead of:
            total_time ~ io_time + compute_time

        This test measures the components and reports the theoretical advantage.
        """
        from fiberhmm.inference.engine import _extract_fiber_read_from_pysam, _process_single_read
        from fiberhmm.core.model_io import load_model

        # Measure read speed
        start = time.perf_counter()
        n_reads = 0
        pysam.set_verbosity(0)
        with pysam.AlignmentFile(bench_bam_medium, "rb") as bam:
            for read in bam:
                if not read.is_unmapped:
                    _extract_fiber_read_from_pysam(read, 'pacbio-fiber', 0)
                    n_reads += 1
        io_time = time.perf_counter() - start

        # Measure compute speed (sample)
        model = load_model(bench_model_path)
        sample_size = min(100, n_reads)
        fiber_reads = []
        with pysam.AlignmentFile(bench_bam_medium, "rb") as bam:
            for i, read in enumerate(bam):
                if i >= sample_size:
                    break
                if not read.is_unmapped:
                    fr = _extract_fiber_read_from_pysam(read, 'pacbio-fiber', 0)
                    if fr:
                        fiber_reads.append(fr)

        start = time.perf_counter()
        for fr in fiber_reads:
            _process_single_read(
                fr, model, edge_trim=10, circular=False,
                mode='pacbio-fiber', context_size=3, msp_min_size=60,
                with_scores=False,
            )
        sample_compute = time.perf_counter() - start
        compute_time_est = sample_compute / len(fiber_reads) * n_reads

        serial_total = io_time + compute_time_est
        pipeline_total = max(io_time, compute_time_est)
        speedup = serial_total / pipeline_total if pipeline_total > 0 else 1.0

        print(f"\n  Theoretical Pipeline Advantage ({n_reads} reads):")
        print(f"    I/O time:         {io_time:.2f}s")
        print(f"    Compute time:     {compute_time_est:.2f}s")
        print(f"    Serial total:     {serial_total:.2f}s")
        print(f"    Pipeline total:   {pipeline_total:.2f}s (overlapping I/O + compute)")
        print(f"    Speedup:          {speedup:.2f}x")
        print(f"    With slow I/O (10x): {speedup:.2f}x -> "
              f"{(io_time*10 + compute_time_est) / max(io_time*10, compute_time_est):.2f}x")

        assert n_reads > 0
