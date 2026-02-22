"""
Package consistency regression tests.

Verify that package imports (fiberhmm.*) produce identical results
to legacy flat imports.
"""
import pytest
import numpy as np
import os
import sys

# Package imports
from fiberhmm.core.hmm import FiberHMM
from fiberhmm.core.model_io import load_model, save_model, load_model_with_metadata
from fiberhmm.core.bam_reader import (
    ContextEncoder,
    encode_from_query_sequence,
)
from fiberhmm.inference.engine import predict_footprints, predict_footprints_and_msps


class TestPackageImports:
    """Verify all expected symbols are importable from package."""

    def test_core_hmm_imports(self):
        from fiberhmm.core.hmm import FiberHMM, train_model, HAS_NUMBA
        assert FiberHMM is not None
        assert callable(train_model)

    def test_core_bam_reader_imports(self):
        from fiberhmm.core.bam_reader import (
            ContextEncoder,
            encode_from_query_sequence,
            parse_mm_tag_query_positions,
        )
        assert ContextEncoder is not None
        assert callable(encode_from_query_sequence)
        assert callable(parse_mm_tag_query_positions)

    def test_core_model_io_imports(self):
        from fiberhmm.core.model_io import load_model, save_model, load_model_with_metadata
        assert callable(load_model)
        assert callable(save_model)
        assert callable(load_model_with_metadata)

    def test_inference_engine_imports(self):
        from fiberhmm.inference.engine import (
            predict_footprints,
            predict_footprints_and_msps,
            detect_mode_from_bam,
        )
        assert callable(predict_footprints)
        assert callable(predict_footprints_and_msps)
        assert callable(detect_mode_from_bam)

    def test_inference_parallel_imports(self):
        from fiberhmm.inference.parallel import _get_genome_regions
        assert callable(_get_genome_regions)

    def test_probabilities_imports(self):
        from fiberhmm.probabilities.context_counter import ContextCounter
        from fiberhmm.probabilities.stats import generate_probability_stats
        assert ContextCounter is not None
        assert callable(generate_probability_stats)

    def test_cli_apply_import(self):
        from fiberhmm.cli.apply import main
        assert callable(main)

    def test_cli_train_import(self):
        from fiberhmm.cli.train import main
        assert callable(main)

    def test_cli_generate_probs_import(self):
        from fiberhmm.cli.generate_probs import main
        assert callable(main)

    def test_cli_extract_tags_import(self):
        from fiberhmm.cli.extract_tags import main
        assert callable(main)

    def test_cli_utils_import(self):
        from fiberhmm.cli.utils import main, cmd_convert, cmd_inspect, cmd_transfer, cmd_adjust
        assert callable(main)
        assert callable(cmd_convert)
        assert callable(cmd_inspect)
        assert callable(cmd_transfer)
        assert callable(cmd_adjust)


class TestEncodingConsistency:
    """Verify encoding produces consistent results via package path."""

    def test_context_encoder_codes(self):
        """ContextEncoder gives consistent lookup codes."""
        lookup = ContextEncoder.get_lookup('A', 3, include_rc=False)
        assert len(lookup) == 4096
        # Verify specific contexts
        assert 'AAAAAAA' in lookup
        code = lookup['AAAAAAA']
        assert isinstance(code, int)
        assert 0 <= code < 4096

    def test_encode_deterministic(self):
        """Encoding same input gives same output."""
        sequence = "ACGTACGTACGTACGTACGTACGT"
        mod_positions = {0, 4, 8, 12, 16, 20}

        enc1 = encode_from_query_sequence(
            sequence, mod_positions, edge_trim=0,
            mode='pacbio-fiber', context_size=3
        )
        enc2 = encode_from_query_sequence(
            sequence, mod_positions, edge_trim=0,
            mode='pacbio-fiber', context_size=3
        )
        np.testing.assert_array_equal(enc1, enc2)


class TestModelConsistency:
    """Verify model operations produce consistent results."""

    def test_save_load_round_trip(self, tmp_path):
        """Model survives save/load round trip."""
        model = FiberHMM(n_states=2)
        np.random.seed(42)
        model.startprob_ = np.array([0.3, 0.7])
        model.transmat_ = np.array([[0.95, 0.05], [0.05, 0.95]])
        model.emissionprob_ = np.random.dirichlet(np.ones(128), size=2)

        filepath = str(tmp_path / "test_model.json")
        save_model(model, filepath, context_size=3, mode='pacbio-fiber')
        loaded = load_model(filepath, normalize=False)

        np.testing.assert_allclose(loaded.startprob_, model.startprob_, rtol=1e-5)
        np.testing.assert_allclose(loaded.transmat_, model.transmat_, rtol=1e-5)
        np.testing.assert_allclose(loaded.emissionprob_, model.emissionprob_, rtol=1e-5)

    def test_viterbi_consistency(self):
        """Viterbi decoding produces consistent results."""
        model = FiberHMM(n_states=2)
        model.startprob_ = np.array([0.5, 0.5])
        model.transmat_ = np.array([[0.9, 0.1], [0.1, 0.9]])
        model.emissionprob_ = np.array([
            [0.1, 0.2, 0.3, 0.4],
            [0.4, 0.3, 0.2, 0.1],
        ])

        obs = np.array([0, 0, 0, 3, 3, 3, 3, 0, 0, 0], dtype=np.int32)
        path1 = model.predict(obs)
        path2 = model.predict(obs)
        np.testing.assert_array_equal(path1, path2)
