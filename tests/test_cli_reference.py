"""docs/reference.md's generated flag tables must match the argparse parsers.

Regenerate with ``python tools/gen_cli_reference.py`` after changing any
command's options or help text.
"""
from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

MAIN_COMMANDS = (
    "fiberhmm-call",
    "fiberhmm-apply",
    "fiberhmm-recall-tfs",
    "fiberhmm-recall-nucs",
    "fiberhmm-qc",
    "fiberhmm-extract",
    "fiberhmm-dedup",
    "fiberhmm-pair",
    "fiberhmm-consensus",
    "fiberhmm-transfer",
)


def _generator():
    spec = importlib.util.spec_from_file_location(
        "gen_cli_reference", REPO_ROOT / "tools" / "gen_cli_reference.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_generator_covers_every_console_script():
    generator = _generator()
    # tomllib is Python >= 3.11; read the [project.scripts] table directly.
    text = (REPO_ROOT / "pyproject.toml").read_text()
    table = text.split("[project.scripts]", 1)[1].split("\n[", 1)[0]
    scripts = set(re.findall(r"^(fiberhmm-[\w-]+)\s*=", table, flags=re.M))
    assert scripts
    assert {name for name, _, _ in generator.COMMANDS} == scripts


@pytest.mark.parametrize("name", MAIN_COMMANDS)
def test_reference_flags_match_argparse(name):
    generator = _generator()
    command = next(entry for entry in generator.COMMANDS if entry[0] == name)
    documented = generator.command_section(
        generator.generated_section(generator.REFERENCE.read_text()), name)
    fresh = "\n".join(generator.render_command(*command)) + "\n"
    assert documented.strip() == fresh.strip(), (
        f"{name} options drifted from docs/reference.md; run "
        "python tools/gen_cli_reference.py")
