"""Release-packaging invariants read straight from pyproject.toml.

The parser is a small regex reader rather than ``tomllib`` so the test also
runs on Python 3.9.
"""

import re
from pathlib import Path

import fiberhmm

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"


def _text():
    return PYPROJECT.read_text()


def _section(name):
    """Return the body of ``[name]`` up to the next table header."""
    match = re.search(
        rf"^\[{re.escape(name)}\]\s*$(.*?)(?=^\[|\Z)", _text(), re.M | re.S
    )
    assert match, f"[{name}] missing from pyproject.toml"
    return match.group(1)


def test_package_version_matches_pyproject():
    declared = re.search(r'^version\s*=\s*"([^"]+)"', _text(), re.M).group(1)
    assert fiberhmm.__version__ == declared
    assert declared.split(".")[0] == "3"


def _console_scripts():
    scripts = {}
    for line in _section("project.scripts").splitlines():
        match = re.match(r'^\s*([\w-]+)\s*=\s*"([\w.]+):(\w+)"', line)
        if match:
            scripts[match.group(1)] = (match.group(2), match.group(3))
    assert scripts, "no console scripts parsed"
    return scripts


def _dependency_names(block):
    return {
        re.split(r"[<>=!~\[; ]", item, maxsplit=1)[0].lower()
        for item in re.findall(r'"([^"]+)"', block)
    }


def test_wheel_ships_no_top_level_modules_or_removed_commands():
    text = _text()
    assert "py-modules" not in text
    scripts = _console_scripts()
    assert "fiberhmm-run" not in scripts
    root = PYPROJECT.parent
    for legacy in ("apply_model", "train_model", "generate_probs", "extract_tags",
                   "fiberhmm_utils", "export_posteriors"):
        assert not (root / f"{legacy}.py").exists(), legacy
    assert not (root / "fiberhmm" / "cli" / "run.py").exists()


def test_consensus_stack_is_a_core_dependency():
    core = _dependency_names(
        re.search(r"^dependencies\s*=\s*\[(.*?)^\]", _text(), re.M | re.S).group(1)
    )
    assert {"numba", "scikit-learn", "joblib", "threadpoolctl"} <= core
    # FiberBrowser pins fiberhmm[consensus]; the alias extra must keep resolving.
    assert re.search(r"^consensus\s*=", _section("project.optional-dependencies"), re.M)


_HELP_PROBE = r"""
import importlib, importlib.abc, json, sys

OPTIONAL = {'torch', 'h5py', 'matplotlib', 'pyBigWig'}

class _Block(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split('.')[0] in OPTIONAL:
            raise ModuleNotFoundError(f"No module named {name!r} (optional extra)")
        return None

sys.meta_path.insert(0, _Block())
results = {}
for script, (module, attr) in json.loads(sys.argv[1]).items():
    sys.argv = [script, '--help']
    try:
        getattr(importlib.import_module(module), attr)()
        code = 0
    except SystemExit as exc:
        code = exc.code or 0
    except BaseException as exc:  # noqa: BLE001 - reported to the parent
        code = f'{type(exc).__name__}: {exc}'
    results[script] = code
print('RESULTS=' + json.dumps(results))
"""


def test_every_console_script_prints_help_without_optional_extras():
    """A core-only install (no [all]/[cuda]) must give working --help everywhere."""
    import json
    import os
    import subprocess
    import sys

    env = dict(os.environ, FIBERHMM_NO_UPDATE_CHECK="1")
    env["PYTHONPATH"] = os.pathsep.join(
        [str(PYPROJECT.parent)] + [p for p in [env.get("PYTHONPATH")] if p]
    )
    proc = subprocess.run(
        [sys.executable, "-c", _HELP_PROBE, json.dumps(_console_scripts())],
        capture_output=True, text=True, env=env, timeout=600,
    )
    line = [ln for ln in proc.stdout.splitlines() if ln.startswith("RESULTS=")]
    assert line, proc.stderr[-4000:]
    results = json.loads(line[-1][len("RESULTS="):])
    failed = {name: code for name, code in results.items() if code != 0}
    assert not failed


def test_retired_modules_and_unregistered_wrappers_are_gone():
    import fiberhmm.cli._entry as entry

    root = PYPROJECT.parent
    # Orphaned by the site-consensus retirement; nothing imports it.
    assert not (root / "fiberhmm" / "inference" / "cuda_likelihood.py").exists()
    # _entry wrappers must all be registered console scripts (no dead shims).
    registered = {attr for module, attr in _console_scripts().values()
                  if module == "fiberhmm.cli._entry"}
    wrappers = {name for name in dir(entry)
                if name.endswith("_main") and callable(getattr(entry, name))}
    assert wrappers <= registered, sorted(wrappers - registered)


def test_python_floor_matches_classifiers():
    floor = re.search(r'^requires-python\s*=\s*">=\s*3\.(\d+)"', _text(), re.M)
    assert floor, "requires-python must declare a >=3.x floor"
    minors = [int(m) for m in re.findall(
        r'"Programming Language :: Python :: 3\.(\d+)"', _section("project"))]
    assert minors and min(minors) == int(floor.group(1))
    # int.bit_count() and numba parfor kernels in the consensus engine need 3.10+.
    assert int(floor.group(1)) >= 10
