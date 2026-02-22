"""
Tests for fiberhmm.inference.parallel module — region splitting and chromosome filtering.
"""
import pytest

from fiberhmm.inference.parallel import _is_main_chromosome


class TestIsMainChromosome:
    """Parametrized tests for chromosome filtering."""

    @pytest.mark.parametrize("chrom", [
        "chr1", "chr2", "chr10", "chr22",
        "chrX", "chrY", "chrM",
        "1", "2", "10", "22",
        "X", "Y", "M", "MT",
    ])
    def test_human_main_chromosomes(self, chrom):
        assert _is_main_chromosome(chrom) is True

    @pytest.mark.parametrize("chrom", [
        "2L", "2R", "3L", "3R", "4",
        "chr2L", "chr2R", "chr3L", "chr3R", "chr4",
    ])
    def test_drosophila_chromosomes(self, chrom):
        assert _is_main_chromosome(chrom) is True

    @pytest.mark.parametrize("chrom", [
        "I", "II", "III", "IV", "V", "VI",
    ])
    def test_c_elegans_chromosomes(self, chrom):
        assert _is_main_chromosome(chrom) is True

    @pytest.mark.parametrize("chrom", [
        "chrUn_gl000220", "chr1_random",
        "scaffold_1", "contig_100",
        "chr1_gl000191_random",
        "chrUn_KI270442v1",
    ])
    def test_scaffolds_and_contigs_rejected(self, chrom):
        assert _is_main_chromosome(chrom) is False

    @pytest.mark.parametrize("chrom", [
        "chr1_KI270706v1_random",
        "chr6_GL000256v2_alt",
        "chrUn_GL000220v1",
    ])
    def test_alt_and_fix_rejected(self, chrom):
        assert _is_main_chromosome(chrom) is False

    def test_case_insensitive(self):
        """Chromosome name matching should be case-insensitive."""
        assert _is_main_chromosome("CHR1") is True
        assert _is_main_chromosome("Chr1") is True
        assert _is_main_chromosome("chr1") is True

    def test_empty_string(self):
        """Empty string should not be a main chromosome."""
        assert _is_main_chromosome("") is False
