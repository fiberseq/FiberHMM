"""Shared CLI model path resolution helpers."""

from __future__ import annotations

import sys


_MODEL_REQUIRED_MESSAGE = (
    "error: one of --model or --enzyme must be provided.\n"
    "  Use --enzyme hia5/dddb/ddda to pick a bundled model, or\n"
    "  use --model /path/to/model.json for a custom model."
)


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
        print(_MODEL_REQUIRED_MESSAGE, file=sys.stderr)
        sys.exit(1)

    from fiberhmm.models import get_model_path
    try:
        model_path = get_model_path(args.enzyme, tool=tool, seq=args.seq)
    except (KeyError, FileNotFoundError) as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)

    if bundled_message is not None:
        print(
            bundled_message.format(model_path=model_path),
            file=bundled_message_file,
        )
    return model_path
