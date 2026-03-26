"""
Memory usage benchmarks: verify bounded memory for streaming mode.
"""
import time
import tracemalloc
import pytest

from fiberhmm.inference.parallel import process_bam_for_footprints

pytestmark = pytest.mark.benchmark


class TestMemoryUsage:
    """Track peak RSS during processing."""

    def test_peak_memory_streaming(self, bench_bam_medium, bench_model_path, tmp_path):
        """
        Peak memory during streaming pipeline.
        Key invariant: memory should NOT scale linearly with read count.
        """
        tracemalloc.start()

        output = str(tmp_path / "mem_test.bam")
        total, _ = process_bam_for_footprints(
            input_bam=bench_bam_medium, output_bam=output,
            model_or_path=bench_model_path,
            train_rids=set(), edge_trim=10, circular=False,
            mode='pacbio-fiber', context_size=3, msp_min_size=60,
            n_cores=2, streaming_pipeline=True, chunk_size=500,
            min_mapq=0, min_read_length=0, prob_threshold=0,
        )

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        peak_mb = peak / 1024 / 1024
        print(f"\n  Peak memory: {peak_mb:.1f} MB")
        print(f"  Current memory: {current / 1024 / 1024:.1f} MB")
        print(f"  Reads processed: {total}")
        if total > 0:
            print(f"  MB per read (peak): {peak_mb / total:.3f}")

        # Streaming should keep peak memory well under 1GB for 5k reads
        assert peak < 1024 * 1024 * 1024, f"Peak memory {peak_mb:.0f}MB exceeds 1GB"

    def test_memory_bounded_vs_read_count(self, bench_bam_medium, bench_model_path, tmp_path):
        """
        Memory should plateau, not grow linearly.
        Process 1000 reads vs 5000 reads and check ratio.
        """
        peaks = {}
        for max_reads, label in [(1000, "1k"), (None, "5k")]:
            tracemalloc.start()
            output = str(tmp_path / f"mem_{label}.bam")
            total, _ = process_bam_for_footprints(
                input_bam=bench_bam_medium, output_bam=output,
                model_or_path=bench_model_path,
                train_rids=set(), edge_trim=10, circular=False,
                mode='pacbio-fiber', context_size=3, msp_min_size=60,
                n_cores=2, streaming_pipeline=True, chunk_size=500,
                max_reads=max_reads,
                min_mapq=0, min_read_length=0, prob_threshold=0,
            )
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            peaks[label] = peak

        ratio = peaks["5k"] / peaks["1k"]
        print(f"\n  Memory scaling:")
        print(f"    1k reads: {peaks['1k'] / 1024 / 1024:.1f} MB")
        print(f"    5k reads: {peaks['5k'] / 1024 / 1024:.1f} MB")
        print(f"    Ratio (5k/1k): {ratio:.2f}x (linear would be 5.0x)")

        # Memory should be sublinear — ratio well under 5x
        assert ratio < 3.0, f"Memory ratio {ratio:.2f}x suggests non-bounded growth"
