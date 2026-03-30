"""
Core-count scaling benchmarks.
"""
import time
import multiprocessing
import pytest

from fiberhmm.inference.parallel import process_bam_for_footprints

pytestmark = pytest.mark.benchmark


class TestScaling:
    """Measure parallel scaling efficiency."""

    def test_streaming_scaling_curve(self, bench_bam_large, bench_model_path, tmp_path):
        """Streaming pipeline scaling: 1, 2, 4 cores (8 if available)."""
        available = multiprocessing.cpu_count()
        core_counts = [1, 2, 4]
        if available >= 8:
            core_counts.append(8)

        results = {}
        for n in core_counts:
            output = str(tmp_path / f"scale_{n}.bam")
            start = time.perf_counter()
            total, _ = process_bam_for_footprints(
                input_bam=bench_bam_large, output_bam=output,
                model_or_path=bench_model_path,
                train_rids=set(), edge_trim=10, circular=False,
                mode='pacbio-fiber', context_size=3, msp_min_size=60,
                n_cores=n, streaming_pipeline=True, chunk_size=500,
                min_mapq=0, min_read_length=0, prob_threshold=0,
            )
            elapsed = time.perf_counter() - start
            results[n] = {'reads': total, 'time': elapsed, 'rate': total / elapsed}

        # Print scaling table
        base_rate = results[core_counts[0]]['rate']
        print(f"\n  {'Cores':>5} | {'Reads/s':>10} | {'Speedup':>8} | {'Efficiency':>10}")
        print(f"  {'-' * 45}")
        for n in core_counts:
            speedup = results[n]['rate'] / base_rate
            efficiency = speedup / n
            print(f"  {n:5d} | {results[n]['rate']:10.1f} | {speedup:8.2f}x | {efficiency:10.1%}")

        # Multi-core should be faster than single-core
        if len(core_counts) > 1:
            assert results[core_counts[-1]]['rate'] > results[1]['rate'], \
                "Multi-core should be faster than single-core"

    def test_region_parallel_scaling_curve(self, bench_bam_large, bench_model_path, tmp_path):
        """Region-parallel scaling: 1, 2, 4 cores."""
        core_counts = [2, 4]

        results = {}
        for n in core_counts:
            output = str(tmp_path / f"region_{n}.bam")
            start = time.perf_counter()
            total, _ = process_bam_for_footprints(
                input_bam=bench_bam_large, output_bam=output,
                model_or_path=bench_model_path,
                train_rids=set(), edge_trim=10, circular=False,
                mode='pacbio-fiber', context_size=3, msp_min_size=60,
                n_cores=n, region_parallel=True,
                min_mapq=0, min_read_length=0, prob_threshold=0,
            )
            elapsed = time.perf_counter() - start
            results[n] = {'reads': total, 'time': elapsed, 'rate': total / elapsed}

        print(f"\n  Region-parallel scaling:")
        for n in core_counts:
            print(f"    {n} cores: {results[n]['rate']:.1f} reads/s ({results[n]['time']:.1f}s)")

        assert all(r['reads'] > 0 for r in results.values())
