"""Shared CLI model path resolution helpers."""

from __future__ import annotations

import sys

_MODEL_REQUIRED_MESSAGE = (
    "error: one of --model or --enzyme must be provided.\n"
    "  Use --enzyme hia5/dddb/ddda to pick a bundled model, or\n"
    "  use --model /path/to/model.json for a custom model."
)


def _print_model_required_error(error_file=None) -> None:
    if error_file is None:
        error_file = sys.stderr
    print(_MODEL_REQUIRED_MESSAGE, file=error_file)


def _model_resolution_error_message(error: Exception) -> str:
    return f"error: {error}"


def _print_bundled_model_message(
    model_path: str,
    bundled_message: str | None,
    bundled_message_file=None,
) -> None:
    if bundled_message is not None:
        print(
            bundled_message.format(model_path=model_path),
            file=bundled_message_file,
        )


def _resolve_bundled_model_path(args, tool: str) -> str:
    from fiberhmm.models import get_model_path
    return get_model_path(args.enzyme, tool=tool, seq=args.seq)


def resolve_model_path(
    args,
    *,
    tool: str,
    bundled_message: str = None,
    bundled_message_file=None,
):
    """Resolve a CLI model path from explicit --model or bundled --enzyme."""
    if args.model is not None:
        return args.model

    if args.enzyme is None:
        _print_model_required_error()
        sys.exit(1)

    try:
        model_path = _resolve_bundled_model_path(args, tool)
    except (KeyError, FileNotFoundError) as e:
        print(_model_resolution_error_message(e), file=sys.stderr)
        sys.exit(1)

    _print_bundled_model_message(
        model_path,
        bundled_message,
        bundled_message_file,
    )
    return model_path
