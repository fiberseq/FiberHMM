"""Tests for BAM index helpers."""

from fiberhmm.io.bam_index import bam_index_exists, bam_index_paths


def test_bam_index_exists_checks_adjacent_and_replaced_suffixes(tmp_path):
    bam_path = tmp_path / "sample.bam"
    assert bam_index_paths(bam_path) == (
        str(tmp_path / "sample.bam.bai"),
        str(tmp_path / "sample.bai"),
    )
    assert not bam_index_exists(bam_path)

    adjacent = tmp_path / "sample.bam.bai"
    adjacent.write_text("")
    assert bam_index_exists(bam_path)

    adjacent.unlink()
    replaced = tmp_path / "sample.bai"
    replaced.write_text("")
    assert bam_index_exists(bam_path)


def test_bam_index_paths_replace_only_terminal_bam_suffix(tmp_path):
    bam_dir = tmp_path / "run.bam.dir"
    bam_dir.mkdir()
    bam_path = bam_dir / "sample.BAM"

    assert bam_index_paths(str(bam_path)) == (
        str(bam_dir / "sample.BAM.bai"),
        str(bam_dir / "sample.bai"),
    )
