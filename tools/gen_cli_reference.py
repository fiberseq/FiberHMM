#!/usr/bin/env python3
"""Render every FiberHMM console script's argparse options into docs/reference/cli.md.

The flag tables in ``docs/reference/cli.md`` (between the ``BEGIN``/``END
GENERATED CLI REFERENCE`` markers) are generated from the parsers themselves,
so they cannot drift from ``--help``. The page is part of the documentation
site (``mkdocs.yml``):

    python tools/gen_cli_reference.py            # rewrite the generated section
    python tools/gen_cli_reference.py --check    # exit 1 if it is out of date

Each command's parser is captured in-process: ``argparse.ArgumentParser.
parse_known_args`` is patched to raise with the parser as soon as the command's
``main()`` asks it to parse, so no command body runs and no console script has
to be installed. ``tests/test_cli_reference.py`` runs the ``--check`` logic for
the main commands.
"""
from __future__ import annotations

import argparse
import contextlib
import importlib
import io
import os
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
REFERENCE = REPO_ROOT / "docs" / "reference" / "cli.md"
BEGIN = "<!-- BEGIN GENERATED CLI REFERENCE (tools/gen_cli_reference.py) -->"
END = "<!-- END GENERATED CLI REFERENCE -->"

# Console script -> (module, function). Order is the order of the document.
# Matches [project.scripts] in pyproject.toml (checked by the test).
COMMANDS: list[tuple[str, str, str]] = [
    ("fiberhmm-pipeline", "fiberhmm.cli.pipeline", "main"),
    ("fiberhmm-call", "fiberhmm.cli.call", "main"),
    ("fiberhmm-apply", "fiberhmm.cli.apply", "main"),
    ("fiberhmm-recall-tfs", "fiberhmm.cli.recall_tfs", "main"),
    ("fiberhmm-recall-nucs", "fiberhmm.cli.recall_tfs", "main_recall_nucs"),
    ("fiberhmm-qc", "fiberhmm.cli.qc", "main"),
    ("fiberhmm-extract", "fiberhmm.cli.extract_tags", "main"),
    ("fiberhmm-dedup", "fiberhmm.cli.dedup", "main"),
    ("fiberhmm-daf-encode", "fiberhmm.cli.daf_encode", "main"),
    ("fiberhmm-daf-snps", "fiberhmm.cli.daf_snps", "main"),
    ("fiberhmm-pair", "fiberhmm.cli.pair", "main"),
    ("fiberhmm-merge", "fiberhmm.cli.merge", "main"),
    ("fiberhmm-tag-m5c", "fiberhmm.cli.tag_m5c", "main"),
    ("fiberhmm-call-m5c", "fiberhmm.cli.call_m5c", "main"),
    ("fiberhmm-consensus", "fiberhmm.inference.consensus.cli", "main"),
    ("fiberhmm-transfer", "fiberhmm.inference.consensus.transfer_cli", "main"),
    ("fiberhmm-footprint-model", "fiberhmm.cli.footprint_model", "main"),
    ("fiberhmm-strand-rescue", "fiberhmm.cli.strand_rescue", "main"),
    ("fiberhmm-strand-rescue-annotate", "fiberhmm.cli.strand_rescue_annotate", "main"),
    ("fiberhmm-strand-rescue-audit", "fiberhmm.cli.strand_rescue_audit", "main"),
    ("fiberhmm-tag-consensus", "fiberhmm.cli.tag_families", "main"),
    ("fiberhmm-posteriors", "fiberhmm.cli.export_posteriors", "main"),
    ("fiberhmm-probs", "fiberhmm.cli.generate_probs", "main"),
    ("fiberhmm-train", "fiberhmm.cli.train", "main"),
    ("fiberhmm-utils", "fiberhmm.cli.utils", "main"),
]


class _Captured(Exception):
    def __init__(self, parser):
        super().__init__("parser captured")
        self.parser = parser


def capture_parser(module_name: str, function: str) -> argparse.ArgumentParser:
    """Return the ArgumentParser the command's ``main`` builds, without running it."""
    module = importlib.import_module(module_name)
    entry = getattr(module, function)
    original = argparse.ArgumentParser.parse_known_args

    def intercept(self, args=None, namespace=None):
        raise _Captured(self)

    argparse.ArgumentParser.parse_known_args = intercept
    saved_argv = sys.argv
    sys.argv = [module_name, "--help"]
    try:
        with contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()):
            entry()
    except _Captured as captured:
        return captured.parser
    finally:
        argparse.ArgumentParser.parse_known_args = original
        sys.argv = saved_argv
    raise RuntimeError(f"{module_name}.{function} never parsed its arguments")


def _flags(action: argparse.Action) -> str:
    if not action.option_strings:
        return f"`{action.metavar or action.dest}`"
    return " / ".join(f"`{flag}`" for flag in action.option_strings)


def _default(parser: argparse.ArgumentParser, action: argparse.Action) -> str:
    if action.required:
        return "required"
    default = action.default
    if isinstance(action, argparse.BooleanOptionalAction):
        if default is None:
            return "auto"
        return "on" if default else "off"
    if isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction)):
        # A --x/--no-x pair sharing one dest whose store_true default is None
        # is resolved at run time. (parser.get_default() cannot tell: it skips
        # None defaults and returns the store_false sibling's True.)
        siblings = [other for other in parser._actions if other.dest == action.dest]
        if any(isinstance(other, argparse._StoreTrueAction) and other.default is None
               for other in siblings):
            return "auto"
        if isinstance(action, argparse._StoreTrueAction):
            return "on" if default else "off"
        return "off" if default else "on"
    if isinstance(action, (argparse._CountAction,)):
        return str(default or 0)
    if default is None or default is argparse.SUPPRESS:
        return "—"
    if isinstance(default, (list, tuple)):
        return "`" + " ".join(str(item) for item in default) + "`" if default else "—"
    return f"`{default}`"


def _help(parser: argparse.ArgumentParser, action: argparse.Action) -> str:
    formatter = parser._get_formatter()
    text = formatter._expand_help(action) if action.help else ""
    if action.choices and not isinstance(action, argparse._SubParsersAction):
        choices = ", ".join(f"`{choice}`" for choice in action.choices)
        text = f"{text} Choices: {choices}." if text else f"Choices: {choices}."
    text = re.sub(r"\s+", " ", text).strip()
    # BooleanOptionalAction appends "(default: True)" on some Python versions
    # only; the Default column already carries it.
    text = re.sub(r"\s*\(default: (True|False|None)\)$", "", text)
    # Placeholders such as <BAM stem> would be read as HTML tags by Markdown.
    text = text.replace("<", "&lt;").replace(">", "&gt;")
    # ... and an underscore next to such a placeholder would open emphasis.
    text = re.sub(r"(?<=&gt;)_|_(?=&lt;)", r"\\_", text)
    return text.replace("|", "\\|")


def _table(parser: argparse.ArgumentParser) -> list[str]:
    rows = ["| Flag | Default | Description |", "|------|---------|-------------|"]
    for action in parser._actions:
        if isinstance(action, (argparse._HelpAction, argparse._SubParsersAction,
                               argparse._VersionAction)):
            continue
        if action.help is argparse.SUPPRESS:
            continue
        rows.append(f"| {_flags(action)} | {_default(parser, action)} | "
                    f"{_help(parser, action)} |")
    return rows


def _subparsers(parser: argparse.ArgumentParser):
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            yield from action.choices.items()


def render_command(name: str, module: str, function: str) -> list[str]:
    parser = capture_parser(module, function)
    lines = [f"## {name}", ""]
    rows = _table(parser)
    if len(rows) > 2:
        lines.extend(rows)
        lines.append("")
    for sub_name, sub_parser in _subparsers(parser):
        lines.extend([f"### {name} {sub_name}", ""])
        sub_rows = _table(sub_parser)
        if len(sub_rows) > 2:
            lines.extend(sub_rows)
        else:
            lines.append("No options.")
        lines.append("")
    return lines


def render(commands=COMMANDS) -> str:
    os.environ.setdefault("FIBERHMM_NO_UPDATE_CHECK", "1")
    lines = [
        BEGIN,
        "",
        "Generated from each command's argparse definition by "
        "`python tools/gen_cli_reference.py`; do not edit by hand. Hidden "
        "compatibility options are omitted. `auto` means the value is resolved "
        "at run time as the description says.",
        "",
    ]
    for name, module, function in commands:
        lines.extend(render_command(name, module, function))
    lines.append(END)
    return "\n".join(lines) + "\n"


def generated_section(text: str) -> str:
    start = text.find(BEGIN)
    end = text.find(END)
    if start < 0 or end < 0:
        raise ValueError(f"{REFERENCE} lacks the generated-section markers")
    return text[start:end + len(END)] + "\n"


def command_section(section: str, name: str) -> str:
    """The generated text for one command (its ## block up to the next ##)."""
    match = re.search(rf"^## {re.escape(name)}\n.*?(?=^## |\Z|^{re.escape(END)})",
                      section, flags=re.S | re.M)
    return match.group(0) if match else ""


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true",
                        help="Exit 1 if docs/reference/cli.md is out of date.")
    args = parser.parse_args(argv)
    sys.path.insert(0, str(REPO_ROOT))
    text = REFERENCE.read_text()
    current = generated_section(text)
    fresh = render()
    if args.check:
        if current != fresh:
            print("docs/reference/cli.md CLI reference is out of date; run "
                  "python tools/gen_cli_reference.py", file=sys.stderr)
            return 1
        return 0
    start = text.find(BEGIN)
    end = text.find(END) + len(END) + 1
    REFERENCE.write_text(text[:start] + fresh + text[end:])
    return 0


if __name__ == "__main__":
    sys.exit(main())
