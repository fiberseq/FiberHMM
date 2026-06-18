"""Tests for shared filesystem status helpers."""

from fiberhmm.io.path_status import path_is_nonempty_file


def test_path_is_nonempty_file_requires_regular_nonempty_file(tmp_path):
    missing = tmp_path / "missing.txt"
    empty = tmp_path / "empty.txt"
    directory = tmp_path / "directory.txt"
    nonempty = tmp_path / "nonempty.txt"

    empty.write_text("")
    directory.mkdir()
    nonempty.write_text("x")

    assert not path_is_nonempty_file(missing)
    assert not path_is_nonempty_file(empty)
    assert not path_is_nonempty_file(directory)
    assert path_is_nonempty_file(nonempty)
