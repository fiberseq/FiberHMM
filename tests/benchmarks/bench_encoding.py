"""Encoding benchmarks for mode-specific context encoders."""

import time

import numpy as np
import pytest

import fiberhmm.core.bam_reader as bam_reader

pytestmark = pytest.mark.benchmark


def test_daf_encoder_fast_path_vs_vectorized_fallback():
    """Compare DAF single-pass encoding to the vectorized fallback oracle."""
    sequence = "ACGTCCGTAAGGTTCCGGAANCGTCCGTAAGGTTCCGGAA" * 120
    mod_positions = {
        i
        for i, base in enumerate(sequence)
        if base == "T" and 6 <= i < len(sequence) - 6 and i % 3 == 0
    }
    kwargs = dict(edge_trim=5, mode="daf", strand="+", context_size=3)
    original_has_numba = bam_reader._HAS_NUMBA

    try:
        bam_reader._HAS_NUMBA = True
        fast_result = bam_reader.encode_from_query_sequence(
            sequence, mod_positions, **kwargs
        )

        bam_reader._HAS_NUMBA = False
        fallback_start = time.perf_counter()
        for _ in range(200):
            fallback_result = bam_reader.encode_from_query_sequence(
                sequence, mod_positions, **kwargs
            )
        fallback_time = time.perf_counter() - fallback_start

        bam_reader._HAS_NUMBA = True
        fast_start = time.perf_counter()
        for _ in range(200):
            fast_result = bam_reader.encode_from_query_sequence(
                sequence, mod_positions, **kwargs
            )
        fast_time = time.perf_counter() - fast_start
    finally:
        bam_reader._HAS_NUMBA = original_has_numba

    np.testing.assert_array_equal(fast_result, fallback_result)
    print(
        "\n  DAF encoder:"
        f" vectorized={fallback_time:.4f}s"
        f" single-pass={fast_time:.4f}s"
        f" speedup={fallback_time / fast_time:.2f}x"
    )
    if original_has_numba:
        assert fast_time < fallback_time
