"""
pysam threading benchmarks: impact of htslib decompression threads.
"""
import time
import pytest
import pysam

pytestmark = pytest.mark.benchmark


class TestPysamThreads:
    """Impact of pysam threads= parameter on BAM read speed."""

    @pytest.mark.parametrize("pysam_threads", [0, 1, 2, 4])
    def test_read_speed_with_threads(self, pysam_threads, bench_bam_medium):
        """Measure BAM iteration speed with different pysam thread counts."""
        pysam.set_verbosity(0)

        start = time.perf_counter()
        count = 0
        with pysam.AlignmentFile(bench_bam_medium, "rb", threads=pysam_threads) as bam:
            for read in bam:
                count += 1
        elapsed = time.perf_counter() - start

        print(f"\n  pysam threads={pysam_threads}: {count} reads in {elapsed:.3f}s "
              f"({count / elapsed:.0f} reads/s)")
        assert count > 0

    @pytest.mark.parametrize("pysam_threads", [0, 1, 2, 4])
    def test_read_extract_speed_with_threads(self, pysam_threads, bench_bam_medium):
        """Measure BAM read + extract speed with different thread counts."""
        from fiberhmm.inference.engine import _extract_fiber_read_from_pysam
        pysam.set_verbosity(0)

        start = time.perf_counter()
        count = 0
        with pysam.AlignmentFile(bench_bam_medium, "rb", threads=pysam_threads) as bam:
            for read in bam:
                if not read.is_unmapped:
                    _extract_fiber_read_from_pysam(read, 'pacbio-fiber', 0)
                    count += 1
        elapsed = time.perf_counter() - start

        print(f"\n  pysam threads={pysam_threads} (read+extract): {count} reads in {elapsed:.3f}s "
              f"({count / elapsed:.0f} reads/s)")
        assert count > 0
