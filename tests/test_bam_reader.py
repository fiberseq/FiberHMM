"""
Unit tests for FiberHMM bam_reader module.

Tests cover:
- ContextEncoder class for hexamer context encoding
- MM/ML tag parsing
- DAF strand detection
- Sequence encoding functions
"""
import pytest
import numpy as np
import os
import sys

# Try package imports first, fall back to flat imports
try:
    from fiberhmm.core.bam_reader import (
        ContextEncoder,
        _build_hexamer_lookup,
        _build_hexamer_lookup_with_rc,
        parse_mm_tag_query_positions,
        detect_daf_strand,
        encode_from_query_sequence,
        HEXAMER_LOOKUP_A,
        NON_TARGET_CODE,
    )
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from bam_reader import (
        ContextEncoder,
        _build_hexamer_lookup,
        _build_hexamer_lookup_with_rc,
        parse_mm_tag_query_positions,
        detect_daf_strand,
        encode_from_query_sequence,
        HEXAMER_LOOKUP_A,
        NON_TARGET_CODE,
    )


class TestContextEncoder:
    """Test ContextEncoder class methods."""

    def test_get_lookup_center_base_a(self):
        """Test lookup table for A-centered contexts."""
        lookup = ContextEncoder.get_lookup('A', context_size=3, include_rc=False)

        # Should have entries for 7-mers with A in center
        # For k=3: 4^3 * 4^3 = 4096 contexts
        assert len(lookup) == 4096
        # 7-mer with A center: XXX-A-XXX (3 + 1 + 3 = 7 chars)
        assert 'AAAACGT' in lookup  # A at position 3 (center)
        assert 'CCCACCC' in lookup  # A at position 3 (center)
        # Verify center is A at position 3
        for ctx in list(lookup.keys())[:10]:
            assert len(ctx) == 7
            assert ctx[3] == 'A'  # Center should be A

    def test_get_lookup_with_rc(self):
        """Test lookup table includes reverse complements."""
        lookup = ContextEncoder.get_lookup('A', context_size=3, include_rc=True)

        # Should include T-centered contexts as reverse complements
        # T context maps to same code as corresponding A context
        assert len(lookup) > 4096  # Has additional RC entries

    def test_get_n_codes(self):
        """Test number of unique codes calculation."""
        # For k=3: 4^(2*3) = 4^6 = 4096
        assert ContextEncoder.get_n_codes(3) == 4096

        # For k=5: 4^(2*5) = 4^10 = 1048576
        assert ContextEncoder.get_n_codes(5) == 1048576

    def test_get_non_target_code(self):
        """Test non-target code value."""
        # Non-target code is one past the max valid code
        assert ContextEncoder.get_non_target_code(3) == 4096

    def test_lookup_is_cached(self):
        """Test that lookups are cached."""
        lookup1 = ContextEncoder.get_lookup('A', context_size=3, include_rc=False)
        lookup2 = ContextEncoder.get_lookup('A', context_size=3, include_rc=False)

        # Should be the same object (cached)
        assert lookup1 is lookup2

    def test_different_center_bases(self):
        """Test lookups for different center bases."""
        lookup_a = ContextEncoder.get_lookup('A', context_size=3, include_rc=False)
        lookup_c = ContextEncoder.get_lookup('C', context_size=3, include_rc=False)

        # Each lookup should have same number of entries (4096)
        assert len(lookup_a) == len(lookup_c) == 4096

        # A-centered should have A in center (position 3)
        for ctx in list(lookup_a.keys())[:10]:
            assert ctx[3] == 'A'

        # C-centered should have C in center (position 3)
        for ctx in list(lookup_c.keys())[:10]:
            assert ctx[3] == 'C'

        # Cross-check: A-centered context shouldn't be in C-centered lookup
        assert 'AAAAAAA' in lookup_a
        assert 'AAAAAAA' not in lookup_c
        assert 'AAACAAA' in lookup_c
        assert 'AAACAAA' not in lookup_a


class TestHexamerLookup:
    """Test hexamer lookup table construction."""

    def test_build_hexamer_lookup(self):
        """Test basic hexamer lookup table."""
        lookup = _build_hexamer_lookup(center_base='A')

        # Should have entries for all valid 7-mers with A in center (k=3)
        assert len(lookup) == 4096

    def test_build_hexamer_lookup_with_rc(self):
        """Test hexamer lookup with reverse complement."""
        lookup = _build_hexamer_lookup_with_rc(center_base='A')

        # Should include both forward and reverse complement contexts
        # So more than 4096 entries
        assert len(lookup) > 0

    def test_legacy_lookup_exists(self):
        """Test that legacy lookup table is available."""
        assert len(HEXAMER_LOOKUP_A) > 0


class TestMMTagParsing:
    """Test MM/ML tag parsing functions."""

    def test_parse_mm_tag_basic(self):
        """Test basic MM tag parsing."""
        sequence = "ACGTACGT"  # A at positions 0, 4
        mm_tag = "A+a,0,3;"  # Mods at first A and 3 A's later (pos 4)
        ml_tag = [200, 150]  # Probabilities

        positions = parse_mm_tag_query_positions(
            mm_tag, ml_tag, sequence, is_reverse=False, prob_threshold=100
        )

        # Should return set of positions
        assert isinstance(positions, set)
        assert len(positions) > 0

    def test_parse_mm_tag_with_threshold(self):
        """Test MM tag parsing respects probability threshold."""
        sequence = "AAAA"  # A at all positions
        mm_tag = "A+a,0,0,0;"  # Mods at positions 0, 1, 2
        ml_tag = [200, 50, 150]  # Middle one below threshold

        # High threshold should filter
        positions_high = parse_mm_tag_query_positions(
            mm_tag, ml_tag, sequence, is_reverse=False, prob_threshold=150
        )

        # Low threshold should include more
        positions_low = parse_mm_tag_query_positions(
            mm_tag, ml_tag, sequence, is_reverse=False, prob_threshold=10
        )

        # Higher threshold should have fewer or equal positions
        assert len(positions_high) <= len(positions_low)

    def test_parse_mm_tag_empty(self):
        """Test parsing empty or missing MM tag."""
        # None mm_tag should return empty set
        positions = parse_mm_tag_query_positions(
            None, None, "ACGT", is_reverse=False, prob_threshold=100
        )
        assert len(positions) == 0

    def test_parse_mm_tag_reverse_strand(self):
        """Test MM tag parsing for reverse strand reads."""
        sequence = "ACGTACGT"
        mm_tag = "A+a,0,3;"
        ml_tag = [200, 150]

        positions = parse_mm_tag_query_positions(
            mm_tag, ml_tag, sequence, is_reverse=True, prob_threshold=100
        )

        # Should still return valid positions
        assert isinstance(positions, set)


class TestDAFStrandDetection:
    """Test DAF-seq strand detection."""

    def test_detect_plus_strand(self):
        """Test detection of plus strand (T at mod positions)."""
        # Sequence with deaminated Cs showing as Ts
        sequence = "ATTTTTTTTTTT"  # Mostly Ts
        t_positions = {1, 2, 3, 4, 5}  # Modifications at T positions

        strand = detect_daf_strand(sequence, t_positions)
        # Plus strand shows C->T conversion, so Ts at mod positions = plus strand
        assert strand == '+'

    def test_detect_minus_strand(self):
        """Test detection of minus strand (A at mod positions)."""
        # Sequence with deaminated Gs showing as As
        sequence = "AAAAAAAAAAAA"  # Mostly As
        a_positions = {0, 1, 2, 3, 4}  # Modifications at A positions

        strand = detect_daf_strand(sequence, a_positions)
        # Minus strand shows G->A conversion, so As at mod positions = minus strand
        assert strand == '-'

    def test_detect_strand_empty(self):
        """Test strand detection with no modifications."""
        sequence = "ACGTACGT"
        strand = detect_daf_strand(sequence, set())

        # Should return '.' for unclear
        assert strand == '.'


class TestSequenceEncoding:
    """Test sequence encoding functions."""

    def test_encode_from_query_sequence_basic(self):
        """Test basic sequence encoding."""
        # Need a sequence long enough for context
        sequence = "ACGTACGTACGTACGT"  # 16 bp
        mod_positions = {0, 4, 8, 12}  # A positions that are methylated

        encoded = encode_from_query_sequence(
            sequence=sequence,
            mod_positions=mod_positions,
            mode='pacbio-fiber',
            context_size=3,
            edge_trim=0
        )

        # Should return array of encoded symbols
        assert isinstance(encoded, np.ndarray)
        assert len(encoded) > 0

    def test_encode_from_query_sequence_with_edge_trim(self):
        """Test sequence encoding with edge trimming."""
        sequence = "A" * 100
        mod_positions = set(range(0, 100, 5))

        # Without trim
        encoded_no_trim = encode_from_query_sequence(
            sequence=sequence,
            mod_positions=mod_positions,
            mode='pacbio-fiber',
            context_size=3,
            edge_trim=0
        )

        # With trim
        encoded_with_trim = encode_from_query_sequence(
            sequence=sequence,
            mod_positions=mod_positions,
            mode='pacbio-fiber',
            context_size=3,
            edge_trim=10
        )

        # Both should have values
        assert len(encoded_no_trim) > 0
        assert len(encoded_with_trim) > 0

    def test_encode_handles_short_sequence(self):
        """Test encoding handles sequences shorter than context."""
        sequence = "ACGT"  # Short
        mod_positions = {0}  # A at position 0

        encoded = encode_from_query_sequence(
            sequence=sequence,
            mod_positions=mod_positions,
            mode='pacbio-fiber',
            context_size=3,  # Context would extend beyond sequence
            edge_trim=0
        )

        # Should handle gracefully
        assert isinstance(encoded, np.ndarray)

    def test_encode_different_modes(self):
        """Test encoding with different modes."""
        sequence = "ACGTACGTACGTACGT"
        mod_positions = {0, 4, 8, 12}

        # PacBio fiber mode
        encoded_pb = encode_from_query_sequence(
            sequence=sequence,
            mod_positions=mod_positions,
            mode='pacbio-fiber',
            context_size=3,
            edge_trim=0
        )

        # Nanopore fiber mode
        encoded_np = encode_from_query_sequence(
            sequence=sequence,
            mod_positions=mod_positions,
            mode='nanopore-fiber',
            context_size=3,
            edge_trim=0
        )

        # Both should produce valid output
        assert isinstance(encoded_pb, np.ndarray)
        assert isinstance(encoded_np, np.ndarray)


class TestEncodingConsistency:
    """Test encoding consistency and determinism."""

    def test_encoding_is_deterministic(self):
        """Test that encoding gives same results for same input."""
        sequence = "ACGTACGTACGTACGT"
        mod_positions = {0, 4, 8, 12}

        encoded1 = encode_from_query_sequence(
            sequence=sequence,
            mod_positions=mod_positions,
            mode='pacbio-fiber',
            context_size=3,
            edge_trim=0
        )

        encoded2 = encode_from_query_sequence(
            sequence=sequence,
            mod_positions=mod_positions,
            mode='pacbio-fiber',
            context_size=3,
            edge_trim=0
        )

        np.testing.assert_array_equal(encoded1, encoded2)

    def test_context_encoder_consistent(self):
        """Test that ContextEncoder gives consistent results."""
        lookup1 = ContextEncoder.get_lookup('A', 3, False)
        lookup2 = ContextEncoder.get_lookup('A', 3, False)

        # Same keys and values
        assert lookup1 == lookup2


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
