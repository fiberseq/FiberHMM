"""Tests for probability-generation utility helpers."""

from fiberhmm.probabilities.utils import (
    detect_strand_and_base,
    get_base_name,
    reverse_complement,
    setup_output_dirs,
)


def test_reverse_complement_preserves_existing_base_mapping():
    assert reverse_complement("ACGT") == "ACGT"
    assert reverse_complement("TCA") == "TGA"


def test_detect_strand_and_base_for_fiber_modes():
    assert detect_strand_and_base("ACGT", {1, 2}, "pacbio-fiber") == (".", "A")
    assert detect_strand_and_base("ACGT", {1, 2}, "nanopore-fiber") == (".", "A")
    assert detect_strand_and_base("ACGT", {1, 2}, "unknown") == (".", "A")


def test_detect_strand_and_base_for_daf_modes():
    assert detect_strand_and_base("TTA", {0, 1}, "daf") == ("+", "C")
    assert detect_strand_and_base("TAA", {1, 2}, "daf") == ("-", "G")
    assert detect_strand_and_base("TAA", {0, 1}, "daf") == (".", "C")


def test_output_directory_helpers(tmp_path):
    output_dir, tables_dir, plots_dir = setup_output_dirs(str(tmp_path / "run"))

    assert output_dir == tmp_path / "run"
    assert tables_dir == output_dir / "tables"
    assert plots_dir == output_dir / "plots"
    assert output_dir.is_dir()
    assert tables_dir.is_dir()
    assert plots_dir.is_dir()


def test_get_base_name_handles_trailing_slash_and_empty_path():
    assert get_base_name("/tmp/example/") == "example"
    assert get_base_name("", default="fallback") == "fallback"
