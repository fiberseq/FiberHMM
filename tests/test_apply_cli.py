"""CLI characterization tests for `fiberhmm-apply` helpers."""

from types import SimpleNamespace

import pytest

from fiberhmm.cli.apply import (
    _resolve_model_path,
    _resolve_process_unmapped,
    _use_streaming_pipeline,
)


def test_apply_model_resolution_uses_custom_path():
    args = SimpleNamespace(model="/tmp/custom.json", enzyme=None, seq=None)

    assert _resolve_model_path(args) == "/tmp/custom.json"


def test_apply_model_resolution_requires_model_or_enzyme(capsys):
    args = SimpleNamespace(model=None, enzyme=None, seq=None)

    with pytest.raises(SystemExit) as exc:
        _resolve_model_path(args)

    assert exc.value.code == 1
    assert "one of --model or --enzyme must be provided" in capsys.readouterr().err


def test_apply_streaming_pipeline_selection():
    assert _use_streaming_pipeline("-", 1)
    assert _use_streaming_pipeline("input.bam", 2)
    assert not _use_streaming_pipeline("input.bam", 1)


def test_apply_process_unmapped_auto_enables_without_index(tmp_path, capsys):
    args = SimpleNamespace(
        input=str(tmp_path / "input.bam"),
        process_unmapped=False,
    )

    assert _resolve_process_unmapped(args, use_streaming=True)
    assert "no BAM index" in capsys.readouterr().out


def test_apply_process_unmapped_respects_explicit_true(tmp_path):
    args = SimpleNamespace(
        input=str(tmp_path / "input.bam"),
        process_unmapped=True,
    )

    assert _resolve_process_unmapped(args, use_streaming=False)
