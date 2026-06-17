"""Shared CLI model-selection helper tests."""

import io

from fiberhmm.cli.model_selection import _print_bundled_model_message


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
