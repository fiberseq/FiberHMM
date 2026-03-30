"""
Tests for IUPAC R/Y encoded DAF-seq support.

DAF-seq collaborators may encode deamination events as IUPAC ambiguity codes
(R/Y) in the BAM sequence instead of MM/ML tags.  Y marks deaminated C
(+ strand) and R marks deaminated G (- strand).
"""

import numpy as np
import pytest

from fiberhmm.core.bam_reader import (
    has_iupac_encoding,
    extract_daf_iupac_positions,
    encode_from_query_sequence,
    detect_daf_strand,
)


# =========================================================================
# TestHasIupacEncoding
# =========================================================================

class TestHasIupacEncoding:
    """Test has_iupac_encoding() detection."""

    def test_standard_sequence(self):
        assert not has_iupac_encoding("ACGTACGTACGT")

    def test_sequence_with_y(self):
        assert has_iupac_encoding("ACYTACGT")

    def test_sequence_with_r(self):
        assert has_iupac_encoding("ACRTAGGT")

    def test_sequence_with_both(self):
        assert has_iupac_encoding("ACYTRAGT")

    def test_lowercase(self):
        assert has_iupac_encoding("acytacgt")
        assert has_iupac_encoding("acrtaggt")

    def test_empty_string(self):
        assert not has_iupac_encoding("")

    def test_n_bases_only(self):
        assert not has_iupac_encoding("ACNGTACNGT")


# =========================================================================
# TestExtractDafIupacPositions
# =========================================================================

class TestExtractDafIupacPositions:
    """Test extract_daf_iupac_positions() conversion and strand detection."""

    def test_ct_strand_y_positions(self):
        """+ strand (CT): Y marks deaminated C → converted to T."""
        seq = "ACYTCYCAGT"
        #       0123456789
        # Y at positions 2 and 5
        mod_pos, strand, conv = extract_daf_iupac_positions(seq, st_tag="CT")
        assert strand == "+"
        assert mod_pos == {2, 5}
        assert conv == "ACTTCTCAGT"  # Y→T at 2 and 5
        # Verify no R or Y remain
        assert "R" not in conv
        assert "Y" not in conv

    def test_ga_strand_r_positions(self):
        """- strand (GA): R marks deaminated G → converted to A."""
        seq = "AGRTGRGATC"
        # A G R T G R G A T C
        # 0 1 2 3 4 5 6 7 8 9
        # R at 2 and 5
        mod_pos, strand, conv = extract_daf_iupac_positions(seq, st_tag="GA")
        assert strand == "-"
        assert mod_pos == {2, 5}
        assert conv == "AGATGAGATC"  # R→A at 2 and 5
        assert "R" not in conv
        assert "Y" not in conv

    def test_st_tag_inference_y_dominant(self):
        """Without st tag, more Y than R → infer + strand."""
        seq = "ACYYCCAGT"
        mod_pos, strand, conv = extract_daf_iupac_positions(seq, st_tag=None)
        assert strand == "+"
        assert len(mod_pos) == 2  # two Y positions

    def test_st_tag_inference_r_dominant(self):
        """Without st tag, more R than Y → infer - strand."""
        seq = "AGRRGGATC"
        mod_pos, strand, conv = extract_daf_iupac_positions(seq, st_tag=None)
        assert strand == "-"
        assert len(mod_pos) == 2  # two R positions

    def test_pure_acgt_no_modifications(self):
        """Pure ACGT sequence produces no modification positions."""
        seq = "ACGTACGTACGT"
        mod_pos, strand, conv = extract_daf_iupac_positions(seq, st_tag="CT")
        assert mod_pos == set()
        assert conv == seq
        assert strand == "+"

    def test_output_is_pure_acgt(self):
        """Converted sequence contains only A, C, G, T."""
        seq = "AYRCCGYRTAG"
        _, _, conv = extract_daf_iupac_positions(seq, st_tag="CT")
        assert set(conv).issubset({"A", "C", "G", "T"})

    def test_unknown_st_tag(self):
        """Unknown st tag value produces '.' strand."""
        seq = "ACYTACGT"
        _, strand, _ = extract_daf_iupac_positions(seq, st_tag="XY")
        assert strand == "."

    def test_case_insensitive(self):
        """Lowercase input is handled correctly."""
        seq = "acytcycagt"
        mod_pos, strand, conv = extract_daf_iupac_positions(seq, st_tag="CT")
        assert strand == "+"
        assert mod_pos == {2, 5}
        # Output is uppercased
        assert conv == conv.upper()


# =========================================================================
# TestIupacEncoderIntegration
# =========================================================================

class TestIupacEncoderIntegration:
    """Test that IUPAC-derived data passes through encode_from_query_sequence()."""

    @pytest.fixture
    def plus_strand_data(self):
        """+ strand IUPAC read → converted to T-based representation."""
        # 200-bp sequence with some Y positions (deaminated C)
        rng = np.random.RandomState(42)
        bases = list("ACGT")
        seq_list = [bases[rng.randint(0, 4)] for _ in range(200)]
        # Mark some C positions as Y
        c_positions = [i for i, b in enumerate(seq_list) if b == "C"]
        deam_positions = sorted(c_positions[:10])  # first 10 C's are deaminated
        for p in deam_positions:
            seq_list[p] = "Y"
        raw_seq = "".join(seq_list)
        mod_pos, strand, conv_seq = extract_daf_iupac_positions(raw_seq, st_tag="CT")
        return conv_seq, mod_pos, strand

    @pytest.fixture
    def minus_strand_data(self):
        """- strand IUPAC read → converted to A-based representation."""
        rng = np.random.RandomState(43)
        bases = list("ACGT")
        seq_list = [bases[rng.randint(0, 4)] for _ in range(200)]
        g_positions = [i for i, b in enumerate(seq_list) if b == "G"]
        deam_positions = sorted(g_positions[:10])
        for p in deam_positions:
            seq_list[p] = "R"
        raw_seq = "".join(seq_list)
        mod_pos, strand, conv_seq = extract_daf_iupac_positions(raw_seq, st_tag="GA")
        return conv_seq, mod_pos, strand

    def test_plus_strand_encodes(self, plus_strand_data):
        conv_seq, mod_pos, strand = plus_strand_data
        encoded = encode_from_query_sequence(
            conv_seq, mod_pos, edge_trim=10, mode="daf", strand=strand
        )
        assert len(encoded) == len(conv_seq)
        assert encoded.dtype == np.int32
        # At least some positions should have valid context codes (not all non-target)
        non_target = 4096  # k=3 default
        assert np.sum(encoded < non_target) > 0, "Expected some methylated context codes"

    def test_minus_strand_encodes(self, minus_strand_data):
        conv_seq, mod_pos, strand = minus_strand_data
        encoded = encode_from_query_sequence(
            conv_seq, mod_pos, edge_trim=10, mode="daf", strand=strand
        )
        assert len(encoded) == len(conv_seq)
        assert np.sum(encoded < 4096) > 0, "Expected some methylated context codes"

    def test_no_iupac_codes_in_encoder_input(self, plus_strand_data):
        """Encoder receives only ACGT — never R or Y."""
        conv_seq, _, _ = plus_strand_data
        assert "R" not in conv_seq
        assert "Y" not in conv_seq


# =========================================================================
# TestIupacMmmlEquivalence
# =========================================================================

class TestIupacMmmlEquivalence:
    """
    Critical: IUPAC and MM/ML paths must produce identical encoder output.

    Construct identical reads via both paths and verify
    encode_from_query_sequence produces the same arrays.
    """

    def _make_equivalent_reads(self, strand, seed=42):
        """
        Build a read pair where IUPAC and MM/ML paths produce the same
        (converted_sequence, mod_positions, strand).

        For + strand (CT):
          Original genome had C; deamination turned some to T.
          MM/ML path: sequence has T at deaminated positions, mod_positions = those T positions.
          IUPAC path:  sequence has Y at those positions → Y→T, mod_positions = same.

        For - strand (GA):
          Original genome had G; deamination turned some to A.
          MM/ML path: sequence has A at deaminated positions, mod_positions = those A positions.
          IUPAC path:  sequence has R at those positions → R→A, mod_positions = same.
        """
        rng = np.random.RandomState(seed)
        length = 300
        bases = list("ACGT")
        seq_list = [bases[rng.randint(0, 4)] for _ in range(length)]

        if strand == "+":
            # + strand: deaminated C → T in read
            target = "C"
            deam_base = "T"
            iupac_code = "Y"
        else:
            # - strand: deaminated G → A in read
            target = "G"
            deam_base = "A"
            iupac_code = "R"

        target_positions = [i for i, b in enumerate(seq_list) if b == target]
        # Deaminate ~20% of target positions
        n_deam = max(1, len(target_positions) // 5)
        deam_positions = set(sorted(target_positions[:n_deam]))

        # Build MM/ML-style sequence (what the BAM would contain after deamination)
        mmml_seq_list = list(seq_list)
        for p in deam_positions:
            mmml_seq_list[p] = deam_base
        mmml_seq = "".join(mmml_seq_list)
        mmml_mod_positions = deam_positions

        # Build IUPAC-style sequence
        iupac_seq_list = list(seq_list)
        for p in deam_positions:
            iupac_seq_list[p] = iupac_code
        iupac_seq = "".join(iupac_seq_list)
        st_tag = "CT" if strand == "+" else "GA"
        iupac_mod_pos, iupac_strand, iupac_conv = extract_daf_iupac_positions(
            iupac_seq, st_tag=st_tag
        )

        return {
            "mmml_seq": mmml_seq,
            "mmml_mod_positions": mmml_mod_positions,
            "mmml_strand": strand,
            "iupac_conv_seq": iupac_conv,
            "iupac_mod_positions": iupac_mod_pos,
            "iupac_strand": iupac_strand,
        }

    def test_plus_strand_equivalence(self):
        data = self._make_equivalent_reads("+")
        assert data["iupac_strand"] == data["mmml_strand"]
        assert data["iupac_conv_seq"] == data["mmml_seq"]
        assert data["iupac_mod_positions"] == data["mmml_mod_positions"]

        # Encode both and compare
        enc_mmml = encode_from_query_sequence(
            data["mmml_seq"], data["mmml_mod_positions"],
            edge_trim=10, mode="daf", strand=data["mmml_strand"],
        )
        enc_iupac = encode_from_query_sequence(
            data["iupac_conv_seq"], data["iupac_mod_positions"],
            edge_trim=10, mode="daf", strand=data["iupac_strand"],
        )
        np.testing.assert_array_equal(enc_mmml, enc_iupac)

    def test_minus_strand_equivalence(self):
        data = self._make_equivalent_reads("-")
        assert data["iupac_strand"] == data["mmml_strand"]
        assert data["iupac_conv_seq"] == data["mmml_seq"]
        assert data["iupac_mod_positions"] == data["mmml_mod_positions"]

        enc_mmml = encode_from_query_sequence(
            data["mmml_seq"], data["mmml_mod_positions"],
            edge_trim=10, mode="daf", strand=data["mmml_strand"],
        )
        enc_iupac = encode_from_query_sequence(
            data["iupac_conv_seq"], data["iupac_mod_positions"],
            edge_trim=10, mode="daf", strand=data["iupac_strand"],
        )
        np.testing.assert_array_equal(enc_mmml, enc_iupac)

    def test_equivalence_multiple_seeds(self):
        """Test equivalence across many random reads."""
        for seed in range(10):
            for strand in ["+", "-"]:
                data = self._make_equivalent_reads(strand, seed=seed)
                enc_mmml = encode_from_query_sequence(
                    data["mmml_seq"], data["mmml_mod_positions"],
                    edge_trim=10, mode="daf", strand=data["mmml_strand"],
                )
                enc_iupac = encode_from_query_sequence(
                    data["iupac_conv_seq"], data["iupac_mod_positions"],
                    edge_trim=10, mode="daf", strand=data["iupac_strand"],
                )
                np.testing.assert_array_equal(
                    enc_mmml, enc_iupac,
                    err_msg=f"Mismatch at seed={seed}, strand={strand}",
                )
