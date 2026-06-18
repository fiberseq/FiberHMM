"""Tests for shared filesystem status helpers."""

import pytest

from fiberhmm.io.path_status import (
    path_is_nonempty_file,
    path_is_regular_file,
    path_size_gb,
    path_size_mb,
)


def test_path_status_helpers_require_regular_files(tmp_path):
    missing = tmp_path / "missing.txt"
    empty = tmp_path / "empty.txt"
    directory = tmp_path / "directory.txt"
    nonempty = tmp_path / "nonempty.txt"

    empty.write_text("")
    directory.mkdir()
    nonempty.write_text("x")

    assert not path_is_regular_file(missing)
    assert path_is_regular_file(empty)
    assert not path_is_regular_file(directory)
    assert path_is_regular_file(nonempty)

    assert not path_is_nonempty_file(missing)
    assert not path_is_nonempty_file(empty)
    assert not path_is_nonempty_file(directory)
    assert path_is_nonempty_file(nonempty)


def test_path_size_helpers_use_binary_units(tmp_path):
    payload = tmp_path / "payload.bin"
    payload.write_bytes(b"x" * 1024)

    assert path_size_mb(payload) == pytest.approx(1024 / (1024 * 1024))
    assert path_size_gb(payload) == pytest.approx(1024 / (1024**3))
