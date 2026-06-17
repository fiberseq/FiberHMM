"""Shared CLI model-selection helper tests."""

import io

from fiberhmm.cli.model_selection import (
    _print_bundled_model_message,
    _print_model_required_error,
)


def test_print_model_required_error_describes_model_choices():
    out = io.StringIO()

    _print_model_required_error(out)

    assert "one of --model or --enzyme" in out.getvalue()
    assert "Use --enzyme hia5/dddb/ddda" in out.getvalue()
    assert "use --model /path/to/model.json" in out.getvalue()


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
