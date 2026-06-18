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
from types import SimpleNamespace

import numpy as np
import pytest

# Try package imports first, fall back to flat imports
try:
    import fiberhmm.core.bam_reader as bam_reader
    from fiberhmm.core.bam_reader import (
        HEXAMER_LOOKUP_A,
        ContextEncoder,
        _append_mm_mod_result,
        _assign_context_codes_for_target_base,
        _build_hexamer_lookup,
        _build_hexamer_lookup_with_rc,
        _cigar_op_len_arrays,
        _context_codes_for_target_positions,
        _context_flanks,
        _daf_deamination_base_counts,
        _daf_strand_params,
        _daf_target_masks,
        _has_mm_ml_inputs,
        _hmm_symbol_offsets,
        _iter_mm_mod_specs,
        _ml_tag_to_uint8_array,
        _mm_base_and_mod_code,
        _mm_base_indices,
        _mm_ml_slice_for_spec,
        _mm_mod_spec_parts,
        _mm_positions_from_spec,
        _mm_qualities_for_valid_positions,
        _mm_target_base,
        _mm_walk_context,
        _modified_base_quality_passes,
        _print_mm_parse_debug,
        _reverse_complement_context,
        detect_daf_strand,
        encode_from_query_sequence,
        get_reference_positions_array,
        parse_mm_ml_per_mod_type,
        parse_mm_tag_query_positions,
    )
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    import bam_reader as bam_reader
    from bam_reader import (
        HEXAMER_LOOKUP_A,
        ContextEncoder,
        _append_mm_mod_result,
        _assign_context_codes_for_target_base,
        _build_hexamer_lookup,
        _build_hexamer_lookup_with_rc,
        _cigar_op_len_arrays,
        _context_codes_for_target_positions,
        _context_flanks,
        _daf_deamination_base_counts,
        _daf_strand_params,
        _daf_target_masks,
        _has_mm_ml_inputs,
        _hmm_symbol_offsets,
        _iter_mm_mod_specs,
        _ml_tag_to_uint8_array,
        _mm_base_and_mod_code,
        _mm_base_indices,
        _mm_ml_slice_for_spec,
        _mm_mod_spec_parts,
        _mm_positions_from_spec,
        _mm_qualities_for_valid_positions,
        _mm_target_base,
        _mm_walk_context,
        _modified_base_quality_passes,
        _print_mm_parse_debug,
        _reverse_complement_context,
        detect_daf_strand,
        encode_from_query_sequence,
        get_reference_positions_array,
        parse_mm_ml_per_mod_type,
        parse_mm_tag_query_positions,
    )


def test_modified_base_quality_passes_unknown_or_threshold_values():
    assert _modified_base_quality_passes(-1, 125)
    assert _modified_base_quality_passes(125, 125)
    assert not _modified_base_quality_passes(124, 125)


def test_context_lookup_helpers_build_flanks_and_reverse_complements():
    assert _context_flanks(0) == [""]
    assert _context_flanks(1) == ["A", "C", "G", "T"]
    assert _context_flanks(2)[:5] == ["AA", "AC", "AG", "AT", "CA"]
    assert _reverse_complement_context("ACGTNRY") == "NNNACGT"


def test_context_codes_for_target_positions_filters_and_reverse_complements():
    seq_int = bam_reader._sequence_base_int_array("AACGT")
    positions = np.asarray([1, 2], dtype=np.int64)
    left_offsets = np.asarray([-1], dtype=np.int64)
    right_offsets = np.asarray([1], dtype=np.int64)
    powers = np.asarray([1], dtype=np.int64)

    valid_positions, codes = _context_codes_for_target_positions(
        positions, seq_int, left_offsets, right_offsets, powers, k=1,
        use_rc=False,
    )

    np.testing.assert_array_equal(valid_positions, [1, 2])
    np.testing.assert_array_equal(codes, [1, 3])

    valid_positions, codes = _context_codes_for_target_positions(
        np.asarray([1], dtype=np.int64),
        seq_int,
        left_offsets,
        right_offsets,
        powers,
        k=1,
        use_rc=True,
    )

    np.testing.assert_array_equal(valid_positions, [1])
    np.testing.assert_array_equal(codes, [14])

    invalid_seq = bam_reader._sequence_base_int_array("AANGT")
    valid_positions, codes = _context_codes_for_target_positions(
        np.asarray([1, 3], dtype=np.int64),
        invalid_seq,
        left_offsets,
        right_offsets,
        powers,
        k=1,
        use_rc=False,
    )

    np.testing.assert_array_equal(valid_positions, [])
    np.testing.assert_array_equal(codes, [])


def test_assign_context_codes_for_target_base_updates_only_matches():
    seq_int = bam_reader._sequence_base_int_array("AACGT")
    encoded = np.full(5, 99, dtype=np.int32)
    positions = np.asarray([1, 2, 3], dtype=np.int64)
    left_offsets = np.asarray([-1], dtype=np.int64)
    right_offsets = np.asarray([1], dtype=np.int64)
    powers = np.asarray([1], dtype=np.int64)

    _assign_context_codes_for_target_base(
        encoded,
        seq_int,
        positions,
        bam_reader._TARGET_BASE_INT["A"],
        left_offsets,
        right_offsets,
        powers,
        k=1,
        use_rc=False,
    )

    np.testing.assert_array_equal(encoded, [99, 1, 99, 99, 99])


def test_daf_target_masks_reconstruct_deaminated_bases():
    seq_int = bam_reader._sequence_base_int_array("TCAT")
    mod_mask = np.asarray([True, False, True, False], dtype=bool)

    is_deaminated, is_non_deaminated, recon_int = _daf_target_masks(
        seq_int,
        mod_mask,
        bam_reader._TARGET_BASE_INT["T"],
        bam_reader._TARGET_BASE_INT["C"],
    )

    np.testing.assert_array_equal(is_deaminated, [True, False, False, False])
    np.testing.assert_array_equal(is_non_deaminated, [False, True, False, False])
    np.testing.assert_array_equal(
        recon_int,
        bam_reader._sequence_base_int_array("CCAT"),
    )


def test_has_mm_ml_inputs_handles_empty_and_numpy_ml_tags():
    assert not _has_mm_ml_inputs("", [128])
    assert not _has_mm_ml_inputs("A+a,0;", None)
    assert not _has_mm_ml_inputs("A+a,0;", [])
    assert not _has_mm_ml_inputs("A+a,0;", np.asarray([], dtype=np.uint8))
    assert _has_mm_ml_inputs("A+a,0;", np.asarray([128, 255], dtype=np.uint8))
    assert _has_mm_ml_inputs("A+a,0;", 128)


def test_ml_tag_to_uint8_array_promotes_scalar_values():
    np.testing.assert_array_equal(_ml_tag_to_uint8_array(128), [128])
    np.testing.assert_array_equal(
        _ml_tag_to_uint8_array(np.asarray(128, dtype=np.uint8)),
        [128],
    )

    assert parse_mm_tag_query_positions(
        "A+a,0;", 128, "A", False, 1, mode="pacbio-fiber",
    ) == {0}


def test_mm_walk_context_builds_forward_and_reverse_search_sequences():
    context = _mm_walk_context(
        "AaGC", is_reverse=False,
    )

    assert context.seq_upper == "AAGC"
    assert context.q_len == 4
    assert context.search_seq == "AAGC"
    np.testing.assert_array_equal(
        context.search_bytes,
        np.frombuffer(b"AAGC", dtype=np.uint8),
    )

    context = _mm_walk_context(
        "AAGC", is_reverse=True,
    )

    assert context.seq_upper == "AAGC"
    assert context.q_len == 4
    assert context.search_seq == "GCTT"
    np.testing.assert_array_equal(
        context.search_bytes,
        np.frombuffer(b"GCTT", dtype=np.uint8),
    )


def test_print_mm_parse_debug_reports_sequence_and_ml_context(capsys):
    _print_mm_parse_debug(
        "AACGT",
        "AACGT",
        "A+a,0,1;",
        np.asarray([200, 150], dtype=np.uint8),
        is_reverse=True,
    )

    out = capsys.readouterr().out
    assert "Seq len=5, bases: A=2 C=1 G=1 T=1" in out
    assert "MM tag: A+a,0,1;..." in out
    assert "ML tag len: 2, first 10 values: [200, 150]" in out
    assert "is_reverse: True, walking on RC(SEQ)" in out


def test_mm_ml_slice_for_spec_returns_view_and_next_index():
    ml = np.asarray([10, 20, 30, 40], dtype=np.uint8)

    ml_slice = _mm_ml_slice_for_spec(ml, 1, 2)

    np.testing.assert_array_equal(ml_slice.values, np.asarray([20, 30], dtype=np.uint8))
    assert np.shares_memory(ml_slice.values, ml)
    assert ml_slice.next_idx == 3


def test_parse_mm_ml_per_mod_type_groups_positions_and_qualities():
    parsed = parse_mm_ml_per_mod_type(
        "A+a,0,1;C+m,2;",
        np.asarray([200, 150, 99], dtype=np.uint8),
        "AAACCC",
        is_reverse=False,
    )

    a_pos, a_qual = parsed[("A", "a")]
    c_pos, c_qual = parsed[("C", "m")]

    np.testing.assert_array_equal(a_pos, [0, 2])
    np.testing.assert_array_equal(a_qual, [200, 150])
    np.testing.assert_array_equal(c_pos, [5])
    np.testing.assert_array_equal(c_qual, [99])


def test_read_mod_query_positions_accepts_numpy_ml_tag(monkeypatch):
    ml_tag = np.asarray([200], dtype=np.uint8)

    class Read:
        query_sequence = "AAAA"
        is_reverse = False

        def has_tag(self, tag):
            return tag in {"MM", "ML"}

        def get_tag(self, tag):
            if tag == "MM":
                return "A+a,0;"
            if tag == "ML":
                return ml_tag
            raise KeyError(tag)

    captured = {}

    def fake_parse(mm_tag, got_ml_tag, sequence, is_reverse, prob_threshold, mode):
        captured.update({
            "mm_tag": mm_tag,
            "ml_tag": got_ml_tag,
            "sequence": sequence,
            "is_reverse": is_reverse,
            "prob_threshold": prob_threshold,
            "mode": mode,
        })
        return {0}

    monkeypatch.setattr(
        bam_reader,
        "get_modified_positions_pysam",
        lambda *args, **kwargs: set(),
    )
    monkeypatch.setattr(bam_reader, "parse_mm_tag_query_positions", fake_parse)

    assert bam_reader._read_mod_query_positions(
        Read(),
        prob_threshold=125,
        mode="pacbio-fiber",
    ) == {0}
    assert captured.pop("ml_tag") is ml_tag
    assert captured == {
        "mm_tag": "A+a,0;",
        "sequence": "AAAA",
        "is_reverse": False,
        "prob_threshold": 125,
        "mode": "pacbio-fiber",
    }


def test_read_bam_keeps_raw_ml_container_for_manual_parser(monkeypatch):
    import array as pyarray

    raw_ml = pyarray.array('B', [255])

    class FakeRead:
        is_unmapped = False
        is_secondary = False
        is_supplementary = False
        mapping_quality = 60
        query_sequence = 'A' * 20
        reference_start = 0
        reference_end = 20
        query_name = 'read1'
        reference_name = 'chr1'
        is_reverse = False

        def has_tag(self, tag):
            return tag in {'MM', 'ML'}

        def get_tag(self, tag):
            if tag == 'MM':
                return 'A+a,0;'
            if tag == 'ML':
                return raw_ml
            raise KeyError(tag)

    class FakeBam:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def fetch(self, region=None):
            return iter([FakeRead()])

    captured = {}

    def fake_parse(mm_tag, ml_tag, sequence, is_reverse, prob_threshold, mode):
        captured['ml_tag'] = ml_tag
        return {0}

    monkeypatch.setattr(bam_reader.pysam, 'AlignmentFile', lambda *args, **kwargs: FakeBam())
    monkeypatch.setattr(bam_reader, 'get_modified_positions_pysam', lambda *args, **kwargs: set())
    monkeypatch.setattr(bam_reader, 'get_reference_positions', lambda read: [])
    monkeypatch.setattr(bam_reader, 'parse_mm_tag_query_positions', fake_parse)

    reads = list(bam_reader.read_bam('fake.bam', min_mapq=0, min_read_length=0))

    assert len(reads) == 1
    assert reads[0].m6a_query_positions == {0}
    assert captured['ml_tag'] is raw_ml


def test_get_reference_positions_array_uses_minus_one_for_insertions():
    class FakeRead:
        reference_start = 10
        cigartuples = [(0, 2), (1, 1), (0, 1)]

    np.testing.assert_array_equal(
        get_reference_positions_array(FakeRead()),
        np.array([10, 11, -1, 12], dtype=np.int32),
    )


def test_cigar_op_len_arrays_split_ops_and_lengths():
    cigar_ops, cigar_lens = _cigar_op_len_arrays([(0, 10), (1, 2), (4, 3)])

    assert cigar_ops.dtype == np.int64
    assert cigar_lens.dtype == np.int64
    np.testing.assert_array_equal(cigar_ops, [0, 1, 4])
    np.testing.assert_array_equal(cigar_lens, [10, 2, 3])


def test_read_bam_filter_helper_applies_basic_read_filters():
    def read(**overrides):
        attrs = {
            "is_unmapped": False,
            "is_secondary": False,
            "is_supplementary": False,
            "mapping_quality": 20,
            "query_sequence": "ACGT",
            "reference_start": 10,
            "reference_end": 14,
        }
        attrs.update(overrides)
        return SimpleNamespace(**attrs)

    assert bam_reader._passes_read_bam_filters(read(), 20, 4) is True
    assert bam_reader._passes_read_bam_filters(
        read(mapping_quality=19), 20, 4,
    ) is False
    assert bam_reader._passes_read_bam_filters(
        read(query_sequence=None), 20, 4,
    ) is False
    assert bam_reader._passes_read_bam_filters(
        read(reference_end=13), 20, 4,
    ) is False
    assert bam_reader._passes_read_bam_filters(
        read(is_secondary=True), 20, 4,
    ) is False
    assert bam_reader._passes_read_bam_filters(
        read(is_supplementary=True), 20, 4,
    ) is False
    assert bam_reader._passes_read_bam_filters(
        read(is_unmapped=True), 20, 4,
    ) is False


def test_fiber_read_from_segment_preserves_read_metadata(monkeypatch):
    read = SimpleNamespace(
        query_name="read1",
        reference_name="chr1",
        reference_start=10,
        reference_end=14,
        is_reverse=True,
        query_sequence="ACGT",
    )
    monkeypatch.setattr(
        bam_reader,
        "get_reference_positions",
        lambda segment: [10, 11, 12, 13],
    )

    fiber = bam_reader._fiber_read_from_segment(read, {1, 3})

    assert fiber.read_id == "read1"
    assert fiber.chrom == "chr1"
    assert fiber.ref_start == 10
    assert fiber.ref_end == 14
    assert fiber.strand == "-"
    assert fiber.query_sequence == "ACGT"
    assert fiber.m6a_query_positions == {1, 3}
    assert fiber.query_to_ref == [10, 11, 12, 13]
    assert fiber.is_reverse is True


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

    def test_hmm_symbol_offsets_follow_context_code_count(self):
        assert _hmm_symbol_offsets(1) == (16, 17)
        assert _hmm_symbol_offsets(3) == (4096, 4097)

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

    def test_mm_mod_spec_helpers_parse_base_mod_and_skips(self):
        spec = _mm_mod_spec_parts("A+a.,0,5,3")

        assert spec.base_mod == "A+a."
        np.testing.assert_array_equal(spec.skip_counts, [0, 5, 3])
        assert spec.n_mods == 3
        padded_spec = _mm_mod_spec_parts(" A+a. , 0, 5, ")
        assert padded_spec.base_mod == "A+a."
        np.testing.assert_array_equal(padded_spec.skip_counts, [0, 5])
        assert padded_spec.n_mods == 2
        assert _mm_target_base(spec.base_mod) == "A"
        assert _mm_target_base("") is None
        base_mod = _mm_base_and_mod_code(spec.base_mod)
        assert base_mod.base == "A"
        assert base_mod.mod_code == "a"
        assert base_mod.as_tuple() == ("A", "a")
        assert _mm_base_and_mod_code("A") is None
        assert _mm_mod_spec_parts("A+a.") is None
        assert _mm_mod_spec_parts(" ,0") is None

    def test_iter_mm_mod_specs_skips_empty_and_malformed_specs(self):
        specs = list(_iter_mm_mod_specs(" A+a,0,5 ; ;bad; C+m?,2 ;"))

        assert [spec.base_mod for spec in specs] == ["A+a", "C+m?"]
        assert [spec.n_mods for spec in specs] == [2, 1]
        np.testing.assert_array_equal(specs[0].skip_counts, [0, 5])
        np.testing.assert_array_equal(specs[1].skip_counts, [2])

    def test_mm_base_indices_convert_skip_counts_to_base_offsets(self):
        np.testing.assert_array_equal(
            _mm_base_indices(np.array([0, 0, 2], dtype=np.int64)),
            [0, 1, 4],
        )
        np.testing.assert_array_equal(
            _mm_base_indices(np.array([], dtype=np.int64)),
            [],
        )

    def test_mm_qualities_for_valid_positions_handles_truncated_ml(self):
        valid = np.array([True, False, True], dtype=bool)

        np.testing.assert_array_equal(
            _mm_qualities_for_valid_positions(
                np.array([200, 100, 180], dtype=np.uint8),
                valid,
                n_mods=3,
            ),
            [200, 180],
        )
        np.testing.assert_array_equal(
            _mm_qualities_for_valid_positions(
                np.array([200, 100], dtype=np.uint8),
                valid,
                n_mods=3,
            ),
            [200],
        )

    def test_append_mm_mod_result_concatenates_duplicate_keys(self):
        result = {}

        _append_mm_mod_result(
            result,
            ("A", "a"),
            np.array([1], dtype=np.int64),
            np.array([200], dtype=np.uint8),
        )
        _append_mm_mod_result(
            result,
            ("A", "a"),
            np.array([4], dtype=np.int64),
            np.array([180], dtype=np.uint8),
        )

        positions, qualities = result[("A", "a")]
        np.testing.assert_array_equal(positions, [1, 4])
        np.testing.assert_array_equal(qualities, [200, 180])

    def test_mm_positions_from_spec_filters_quality_bounds_and_reverse(self):
        skip_arr = np.array([0, 1, 1], dtype=np.int64)
        base_positions = np.array([1, 4, 8], dtype=np.int64)
        ml_slice = np.array([200, 100], dtype=np.uint8)

        forward = _mm_positions_from_spec(
            skip_arr, base_positions, ml_slice,
            q_len=10, is_reverse=False, prob_threshold=125,
        )
        reverse = _mm_positions_from_spec(
            skip_arr, base_positions, ml_slice,
            q_len=10, is_reverse=True, prob_threshold=125,
        )

        np.testing.assert_array_equal(forward, [1])
        np.testing.assert_array_equal(reverse, [8])

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

    def test_deamination_base_counts_ignore_out_of_range_positions(self):
        counts = _daf_deamination_base_counts("taCG", {-1, 0, 1, 2, 99})

        assert counts.t == 1
        assert counts.a == 1

    def test_daf_strand_params_names_encoding_values(self):
        plus_params = _daf_strand_params("TTTT", {0, 1}, "+")
        minus_params = _daf_strand_params("AAAA", {0, 1}, "-")
        detected_params = _daf_strand_params("TTTT", {0, 1}, ".")

        assert plus_params.deaminated_base_int == 2
        assert plus_params.original_base_int == 1
        assert not plus_params.use_reverse_complement
        assert minus_params.deaminated_base_int == 0
        assert minus_params.original_base_int == 3
        assert minus_params.use_reverse_complement
        assert detected_params == plus_params


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
