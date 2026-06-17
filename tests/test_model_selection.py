"""Shared CLI model-selection helper tests."""

import io
from types import SimpleNamespace

import fiberhmm.models as models
from fiberhmm.cli.model_selection import (
    _model_resolution_error_message,
    _print_bundled_model_message,
    _print_model_required_error,
    _resolve_bundled_model_path,
)


def test_print_model_required_error_describes_model_choices():
    out = io.StringIO()

    _print_model_required_error(out)

    assert "one of --model or --enzyme" in out.getvalue()
    assert "Use --enzyme hia5/dddb/ddda" in out.getvalue()
    assert "use --model /path/to/model.json" in out.getvalue()


def test_model_resolution_error_message_prefixes_error():
    message = _model_resolution_error_message(FileNotFoundError("missing model"))

    assert message == "error: missing model"


def test_resolve_bundled_model_path_passes_cli_args(monkeypatch):
    calls = []

    def fake_get_model_path(enzyme, *, tool, seq):
        calls.append((enzyme, tool, seq))
        return "/models/hia5.json"

    monkeypatch.setattr(models, "get_model_path", fake_get_model_path)

    args = SimpleNamespace(enzyme="hia5", seq="nanopore")

    assert _resolve_bundled_model_path(args, "recall") == "/models/hia5.json"
    assert calls == [("hia5", "recall", "nanopore")]


def test_print_bundled_model_message_formats_configured_message():
    out = io.StringIO()

    _print_bundled_model_message(
        "/models/hia5.json",
        "Using bundled model: {model_path}",
        out,
    )

    assert out.getvalue() == "Using bundled model: /models/hia5.json\n"


def test_print_bundled_model_message_skips_unconfigured_message():
    out = io.StringIO()

    _print_bundled_model_message("/models/hia5.json", None, out)

    assert out.getvalue() == ""
