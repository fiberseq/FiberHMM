"""
Unit tests for FiberHMM bam_reader module.

Tests cover:
- ContextEncoder class for hexamer context encoding
- MM/ML tag parsing
- DAF strand detection
- Sequence encoding functions
"""
import os
import sys

import numpy as np
import pytest

# Try package imports first, fall back to flat imports
try:
    import fiberhmm.core.bam_reader as bam_reader
    from fiberhmm.core.bam_reader import (
        HEXAMER_LOOKUP_A,
        ContextEncoder,
        _build_hexamer_lookup,
        _build_hexamer_lookup_with_rc,
        detect_daf_strand,
        encode_from_query_sequence,
        parse_mm_tag_query_positions,
    )
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    import bam_reader as bam_reader
    from bam_reader import (
        HEXAMER_LOOKUP_A,
        ContextEncoder,
        _build_hexamer_lookup,
        _build_hexamer_lookup_with_rc,
        detect_daf_strand,
        encode_from_query_sequence,
        parse_mm_tag_query_positions,
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

    @pytest.mark.parametrize(
        ("mode", "is_reverse"),
        [
            ("pacbio-fiber", False),
            ("nanopore-fiber", False),
            ("nanopore-fiber", True),
        ],
    )
    def test_m6a_numba_fast_path_matches_vectorized(self, monkeypatch, mode, is_reverse):
        """The m6A single-pass encoder must match the vectorized fallback."""
        sequence = "ACGTCCGTAAGGTTCCGGAANACGTCCGTAAGGTTCCGGAA" * 3
        target_base = "T" if mode == "nanopore-fiber" and is_reverse else "A"
        mod_positions = {
            i
            for i, base in enumerate(sequence)
            if base == target_base and 6 <= i < len(sequence) - 6 and i % 4 == 0
        }

        monkeypatch.setattr(bam_reader, "_HAS_NUMBA", False)
        expected = bam_reader.encode_from_query_sequence(
            sequence,
            mod_positions,
            edge_trim=5,
            mode=mode,
            strand=".",
            context_size=3,
            is_reverse=is_reverse,
        )

        monkeypatch.setattr(bam_reader, "_HAS_NUMBA", True)
        actual = bam_reader.encode_from_query_sequence(
            sequence,
            mod_positions,
            edge_trim=5,
            mode=mode,
            strand=".",
            context_size=3,
            is_reverse=is_reverse,
        )

        np.testing.assert_array_equal(actual, expected)

    @pytest.mark.parametrize(("strand", "deam_base"), [("+", "T"), ("-", "A")])
    def test_daf_numba_fast_path_matches_vectorized(self, monkeypatch, strand, deam_base):
        """The DAF single-pass encoder must match the vectorized fallback."""
        sequence = "ACGTCCGTAAGGTTCCGGAANCGTCCGTAAGGTTCCGGAA" * 3
        mod_positions = {
            i
            for i, base in enumerate(sequence)
            if base == deam_base and 6 <= i < len(sequence) - 6 and i % 3 == 0
        }

        monkeypatch.setattr(bam_reader, "_HAS_NUMBA", False)
        expected = bam_reader.encode_from_query_sequence(
            sequence,
            mod_positions,
            edge_trim=5,
            mode="daf",
            strand=strand,
            context_size=3,
        )

        monkeypatch.setattr(bam_reader, "_HAS_NUMBA", True)
        actual = bam_reader.encode_from_query_sequence(
            sequence,
            mod_positions,
            edge_trim=5,
            mode="daf",
            strand=strand,
            context_size=3,
        )

        np.testing.assert_array_equal(actual, expected)


class TestNanoporeReverseStrand:
    """Regression tests for the nanopore reverse-strand bug.

    Bug: nanopore-fiber mode silently dropped all m6A on reverse-aligned
    reads because the encoder only looked at A positions in SAM SEQ.
    For reverse-aligned reads, basecalled A's are stored as T's (RC),
    so all m6A positions fell on non-target bases and were ignored.

    The fix: for is_reverse=True in nanopore mode, encode T positions
    with RC context to recover basecalled-forward emission space.
    """

    def _make_complement(self, seq):
        return seq.translate(str.maketrans('ACGT', 'TGCA'))

    def test_nanopore_forward_has_targets(self):
        """Forward nanopore read: A positions should be encoded as targets."""
        seq = "ACGTACGTACGTACGTACGTACGTACGT"  # 28 bp, A at 0,4,8,...
        mods = {0, 8, 16, 24}  # m6A at some A positions
        enc = encode_from_query_sequence(
            seq, mods, mode='nanopore-fiber', context_size=3,
            edge_trim=0, is_reverse=False)
        n_codes = 4096
        non_target = n_codes
        # A positions should be target (not non_target)
        a_positions = [i for i, b in enumerate(seq) if b == 'A']
        for p in a_positions:
            if 3 <= p < len(seq) - 3:  # skip edge-trimmed
                assert enc[p] != non_target and enc[p] != non_target + non_target + 1, \
                    f"Forward A at pos {p} should be a valid target"

    def test_nanopore_reverse_has_targets(self):
        """Reverse nanopore read: T positions (basecalled A→SEQ T) should
        be encoded as targets. This was the core bug — without the fix,
        ALL T positions get non_target_code and the read encodes as
        entirely non-informative."""
        seq = "ACGTACGTACGTACGTACGTACGTACGT"
        # For a reverse read, m6A positions point at T's in SEQ
        t_positions = [i for i, b in enumerate(seq) if b == 'T']
        mods = set(t_positions[:4])  # m6A at first 4 T positions
        enc = encode_from_query_sequence(
            seq, mods, mode='nanopore-fiber', context_size=3,
            edge_trim=0, is_reverse=True)
        n_codes = 4096
        non_target = n_codes
        # T positions should now be valid targets (not skipped)
        for p in t_positions:
            if 3 <= p < len(seq) - 3:
                assert enc[p] != non_target and enc[p] != non_target + non_target + 1, \
                    f"Reverse T at pos {p} should be a valid target (bug: was non_target)"

    def test_nanopore_reverse_without_fix_would_fail(self):
        """Demonstrate that is_reverse=False on a reverse read drops m6A."""
        seq = "ACGTACGTACGTACGTACGTACGTACGT"
        t_positions = [i for i, b in enumerate(seq) if b == 'T']
        mods = set(t_positions[:4])
        # Simulate the BUG: is_reverse=False on a reverse read
        enc_buggy = encode_from_query_sequence(
            seq, mods, mode='nanopore-fiber', context_size=3,
            edge_trim=0, is_reverse=False)
        n_codes = 4096
        non_target = n_codes
        unmeth_offset = n_codes + 1
        # With the bug, T positions are non_target → mods at T's are invisible
        for p in t_positions:
            if 3 <= p < len(seq) - 3:
                assert enc_buggy[p] == non_target + unmeth_offset, \
                    f"Bug: T at pos {p} should be non_target when is_reverse=False"

    def test_nanopore_strand_symmetry(self):
        """A forward read and its RC (as a reverse read) should produce
        equivalent encoding patterns at corresponding positions."""
        fwd_seq = "AACGATCGAACGATCGAACGATCG"  # 24 bp
        fwd_mods = {0, 1, 8, 9, 16, 17}  # m6A at some A positions

        # Simulate reverse read: RC the sequence, flip mod positions
        rev_seq = self._make_complement(fwd_seq)[::-1]
        q_len = len(fwd_seq)
        rev_mods = {q_len - 1 - p for p in fwd_mods}

        enc_fwd = encode_from_query_sequence(
            fwd_seq, fwd_mods, mode='nanopore-fiber', context_size=3,
            edge_trim=0, is_reverse=False)
        enc_rev = encode_from_query_sequence(
            rev_seq, rev_mods, mode='nanopore-fiber', context_size=3,
            edge_trim=0, is_reverse=True)

        n_codes = 4096
        non_target = n_codes
        unmeth_offset = n_codes + 1

        # Count how many positions are valid targets in each
        fwd_valid = np.sum((enc_fwd != non_target + unmeth_offset) &
                            (enc_fwd != non_target))
        rev_valid = np.sum((enc_rev != non_target + unmeth_offset) &
                            (enc_rev != non_target))
        # Should have similar number of valid target positions
        assert rev_valid > 0, "Reverse read must have valid targets"
        assert abs(fwd_valid - rev_valid) <= 2, \
            f"Strand asymmetry: fwd={fwd_valid} rev={rev_valid} valid targets"

        # Count methylated targets
        fwd_meth = np.sum(enc_fwd < n_codes)
        rev_meth = np.sum(enc_rev < n_codes)
        assert fwd_meth == rev_meth, \
            f"Methylated count mismatch: fwd={fwd_meth} rev={rev_meth}"

    def test_pacbio_mode_unaffected(self):
        """PacBio mode should not change behavior with is_reverse flag."""
        seq = "ACGTACGTACGTACGTACGTACGTACGT"
        mods = {0, 8, 16, 24}
        enc_fwd = encode_from_query_sequence(
            seq, mods, mode='pacbio-fiber', context_size=3,
            edge_trim=0, is_reverse=False)
        enc_rev = encode_from_query_sequence(
            seq, mods, mode='pacbio-fiber', context_size=3,
            edge_trim=0, is_reverse=True)
        # PacBio already handles both A and T → is_reverse shouldn't matter
        np.testing.assert_array_equal(enc_fwd, enc_rev)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
