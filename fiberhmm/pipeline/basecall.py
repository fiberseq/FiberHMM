"""Optional first step of ``fiberhmm-pipeline``: basecall raw Nanopore data.

POD5 files (or folders of them) are basecalled with Oxford Nanopore's
``dorado basecaller`` into one unaligned BAM, which the pipeline then aligns
and calls like any other unaligned BAM (dorado's own ``@PG``/``@RG`` lines,
with the exact models, are carried into the outputs).

Dorado is not bundled with FiberHMM (it needs a GPU for practical speed, is
large, and is distributed by Oxford Nanopore under the Oxford Nanopore
Technologies PLC. Public License); :func:`find_dorado` looks for an
installed copy: ``--dorado PATH``, ``$FIBERHMM_DORADO``, ``PATH``, then the
usual install locations.

Defaults follow the chemistry: Hia5 (m6A Fiber-seq) is basecalled with the
``sup`` model and the 6mA modification model (``--modified-bases 6mA``); DAF-seq
(DddA/DddB) is plain basecalling (deaminations are read from the sequence).

The step resumes: dorado writes to ``<sample>.basecalled.partial.bam``; when a
run stops, the next one salvages that file's complete records and passes them
to ``dorado --resume-from`` so finished reads are not basecalled again, and a
finished basecalling (its marker and BAM intact) is never repeated.
"""
from __future__ import annotations

import glob
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from typing import Optional

RAW_EXTENSIONS = (".pod5", ".fast5")
DEFAULT_MODEL = "sup"
DEFAULT_MODIFIED_BASES = {"hia5": "6mA", "ddda": None, "dddb": None}
# Model downloads go here instead of dorado's default (the current directory).
MODELS_DIR_ENV = "FIBERHMM_DORADO_MODELS_DIR"
DORADO_ENV = "FIBERHMM_DORADO"

INSTALL_HELP = """\
The input contains raw Nanopore signal (POD5), which must be basecalled with
Oxford Nanopore's dorado first, and dorado was not found. FiberHMM does not
bundle dorado (it is distributed by Oxford Nanopore under its own licence and
needs a GPU for practical speed). Install it:

  1. Download the archive for your system from
     https://github.com/nanoporetech/dorado (Releases / "Installation"),
     e.g. dorado-<version>-osx-arm64.zip, -linux-x64.tar.gz or -win64.zip.
  2. Unpack it anywhere, e.g. ~/dorado, and either add its bin/ folder to
     PATH or pass the program with --dorado ~/dorado/bin/dorado
     (or set FIBERHMM_DORADO).

Then run the same command again. To basecall yourself instead, run
  dorado basecaller sup <pod5 folder> --modified-bases 6mA > calls.bam
(no --modified-bases for DAF-seq) and give calls.bam to fiberhmm-pipeline."""


class DoradoNotFound(RuntimeError):
    pass


@dataclass
class Dorado:
    path: str
    version: str

    def describe(self) -> str:
        return f"dorado {self.version} ({self.path})"


def is_raw_path(path: str) -> bool:
    """A POD5/FAST5 file, or a folder holding them (searched recursively)."""
    if os.path.isdir(path):
        return bool(raw_files_in(path, limit=1))
    return str(path).lower().endswith(RAW_EXTENSIONS)


def raw_files_in(folder: str, limit: Optional[int] = None) -> list[str]:
    found: list[str] = []
    for root, dirs, files in os.walk(folder):
        dirs[:] = sorted(d for d in dirs if not d.startswith("."))
        for name in sorted(files):
            if name.lower().endswith(RAW_EXTENSIONS) and not name.startswith("."):
                found.append(os.path.join(root, name))
                if limit and len(found) >= limit:
                    return found
    return found


def _version_key(path: str):
    return [int(part) if part.isdigit() else part
            for part in re.split(r"(\d+)", os.path.basename(os.path.dirname(
                os.path.dirname(path))))]


def candidate_paths() -> list[str]:
    """Usual install locations (newest version first within each pattern)."""
    home = os.path.expanduser("~")
    patterns = [
        os.path.join(home, ".local", "bin", "dorado"),
        os.path.join(home, "local", "dorado*", "bin", "dorado"),
        os.path.join(home, "local", "dorado", "dorado*", "bin", "dorado"),
        os.path.join(home, "dorado*", "bin", "dorado"),
        os.path.join(home, "Applications", "dorado*", "bin", "dorado"),
        os.path.join(home, "opt", "dorado*", "bin", "dorado"),
        "/opt/dorado*/bin/dorado",
        "/opt/ont/dorado*/bin/dorado",
        "/usr/local/dorado*/bin/dorado",
        "/Applications/dorado*/bin/dorado",
    ]
    found: list[str] = []
    for pattern in patterns:
        for path in sorted(glob.glob(pattern), key=_version_key, reverse=True):
            if path not in found:
                found.append(path)
    return found


def dorado_version(path: str) -> str:
    """``dorado --version`` (dorado prints it on stderr), e.g. ``2.0.0+20e87c8``."""
    try:
        result = subprocess.run([path, "--version"], capture_output=True, text=True,
                                timeout=60)
    except (OSError, subprocess.SubprocessError) as exc:
        raise DoradoNotFound(f"{path} could not be run ({exc}).\n\n{INSTALL_HELP}")
    text = (result.stdout + "\n" + result.stderr).strip()
    match = re.search(r"\b(\d+\.\d+\.\d+\S*)", text)
    if result.returncode != 0 or not match:
        raise DoradoNotFound(f"{path} did not report a dorado version "
                             f"(exit {result.returncode}: {text[-300:]!r}).\n\n{INSTALL_HELP}")
    return match.group(1)


def find_dorado(explicit: Optional[str] = None) -> Dorado:
    """Locate dorado: ``explicit`` (a program, or a folder holding bin/dorado),
    ``$FIBERHMM_DORADO``, ``PATH``, then :func:`candidate_paths`."""
    requested = explicit or os.environ.get(DORADO_ENV)
    if requested:
        path = os.path.expanduser(requested)
        if os.path.isdir(path):
            for sub in (os.path.join(path, "bin", "dorado"), os.path.join(path, "dorado")):
                if os.path.isfile(sub):
                    path = sub
                    break
        if not (os.path.isfile(path) and os.access(path, os.X_OK)):
            raise DoradoNotFound(f"dorado was not found at {requested}.\n\n{INSTALL_HELP}")
        return Dorado(os.path.abspath(path), dorado_version(path))
    for path in [shutil.which("dorado")] + candidate_paths():
        if path and os.path.isfile(path) and os.access(path, os.X_OK):
            try:
                return Dorado(os.path.abspath(path), dorado_version(path))
            except DoradoNotFound:
                continue
    raise DoradoNotFound(INSTALL_HELP)


def accepts_fast5(version: str) -> bool:
    """dorado read FAST5 until its 1.0 release; later versions read POD5 only."""
    match = re.match(r"(\d+)\.(\d+)", version or "")
    return bool(match) and (int(match.group(1)), int(match.group(2))) < (1, 0)


def models_dir(explicit: Optional[str] = None) -> str:
    path = explicit or os.environ.get(MODELS_DIR_ENV) or os.path.join(
        os.path.expanduser("~"), ".fiberhmm", "dorado_models")
    return os.path.abspath(os.path.expanduser(path))


@dataclass
class BasecallSettings:
    model: str = DEFAULT_MODEL
    modified_bases: Optional[list] = None   # codes, e.g. ["6mA"]
    modbase_models: Optional[list] = None   # names or paths
    device: str = "auto"
    batchsize: Optional[int] = None
    models_directory: Optional[str] = None
    extra_args: tuple = ()

    def fingerprint(self) -> dict:
        """What changes the basecalls (device, batch size and the model
        folder do not). Models given as paths, and files named in the extra
        arguments (e.g. ``--read-ids``), are identified by path, size and
        modification time of their files."""
        return {"model": self.model, "modified_bases": list(self.modified_bases or []),
                "modbase_models": list(self.modbase_models or []),
                "extra_args": list(self.extra_args),
                "files": path_identity([self.model, *(self.modbase_models or []),
                                        *self.extra_args])}


def path_identity(arguments) -> list[list]:
    """``[[path, size, mtime_ns], ...]`` for every argument that is an existing
    file, or a folder (its files, e.g. a dorado model folder)."""
    out = []
    for argument in arguments:
        text = str(argument)
        if not text or text.startswith("-") or not os.path.exists(text):
            continue
        if os.path.isdir(text):
            for root, dirs, files in os.walk(text):
                dirs.sort()
                for name in sorted(files):
                    path = os.path.join(root, name)
                    st = os.stat(path)
                    out.append([os.path.abspath(path), st.st_size, st.st_mtime_ns])
        else:
            st = os.stat(text)
            out.append([os.path.abspath(text), st.st_size, st.st_mtime_ns])
    return out


def resolve_settings(enzyme: str, model: Optional[str] = None,
                     modified_bases: Optional[str] = None,
                     modbase_models: Optional[str] = None, device: str = "auto",
                     batchsize: Optional[int] = None,
                     models_directory: Optional[str] = None,
                     extra_args=()) -> BasecallSettings:
    """The basecalling setup for ``enzyme`` with the user's choices.

    ``modified_bases``: space/comma separated codes, or ``none``;
    ``modbase_models``: comma separated names/paths (replaces the codes).
    """
    if modified_bases and modbase_models:
        raise ValueError("give --dorado-modified-bases or --dorado-modbase-models, "
                         "not both")
    codes = None
    models = None
    if modbase_models:
        models = [m.strip() for m in str(modbase_models).split(",") if m.strip()]
    elif modified_bases:
        if str(modified_bases).strip().lower() != "none":
            codes = [c for c in re.split(r"[\s,]+", str(modified_bases).strip()) if c]
    elif "," not in str(model or ""):
        default = DEFAULT_MODIFIED_BASES.get(enzyme)
        codes = [default] if default else None
    return BasecallSettings(model=model or DEFAULT_MODEL, modified_bases=codes,
                            modbase_models=models, device=device or "auto",
                            batchsize=batchsize, models_directory=models_dir(models_directory),
                            extra_args=tuple(extra_args or ()))


def basecall_command(dorado: Dorado, settings: BasecallSettings, data: str,
                     recursive: bool, resume_from: Optional[str] = None) -> list[str]:
    """``dorado basecaller <model> <data> ...``; the BAM goes to stdout."""
    cmd = [dorado.path, "basecaller", settings.model, data,
           "--models-directory", settings.models_directory or models_dir()]
    if recursive:
        cmd.append("--recursive")
    if settings.device and settings.device != "auto":
        cmd += ["--device", settings.device]
    if settings.batchsize:
        cmd += ["--batchsize", str(int(settings.batchsize))]
    if settings.modbase_models:
        cmd += ["--modified-bases-models", ",".join(settings.modbase_models)]
    elif settings.modified_bases:
        cmd += ["--modified-bases", *settings.modified_bases]
    if resume_from:
        cmd += ["--resume-from", resume_from]
    cmd += list(settings.extra_args)
    return cmd


def raw_file_identity(paths: list[str]) -> list[list]:
    """``[[path, size, mtime_ns], ...]`` of every raw file under ``paths``.

    POD5 runs reach hundreds of gigabytes, so files are identified by path,
    size and modification time rather than a content digest.
    """
    out = []
    for path in paths:
        files = raw_files_in(path) if os.path.isdir(path) else [path]
        for item in files:
            st = os.stat(item)
            out.append([os.path.abspath(item), st.st_size, st.st_mtime_ns])
    return out


def salvage_bam(source: str, target: str) -> int:
    """Copy the complete records of a (possibly truncated) BAM into ``target``.

    Returns the number of records copied (``target`` is removed when none):
    a dorado run that was killed leaves a BAM without its end, which
    ``dorado --resume-from`` needs readable.
    """
    import contextlib
    import pysam
    count = 0
    tmp = target + ".tmp"
    src = None
    try:
        # A killed writer leaves no BGZF EOF block (and maybe a cut block):
        # read what is complete; closing such a file can raise too.
        src = pysam.AlignmentFile(source, "rb", check_sq=False, ignore_truncation=True)
        with pysam.AlignmentFile(tmp, "wb", header=src.header) as out:
            try:
                for read in src.fetch(until_eof=True):
                    out.write(read)
                    count += 1
            except (OSError, ValueError):
                pass  # truncated: keep what was complete
    except (OSError, ValueError):
        count = 0
    finally:
        if src is not None:
            with contextlib.suppress(Exception):
                src.close()
    if count:
        os.replace(tmp, target)
    else:
        for leftover in (tmp, target):
            if os.path.exists(leftover):
                os.remove(leftover)
    return count


def count_records(path: str) -> int:
    import pysam
    with pysam.AlignmentFile(path, "rb", check_sq=False) as bam:
        return sum(1 for _ in bam.fetch(until_eof=True))


def stage_inputs(paths: list[str], stage_dir: str) -> tuple[str, bool]:
    """``(data argument, recursive)`` for dorado, which takes one file or folder.

    One folder is passed as it is (``--recursive``); one file too; several
    inputs are linked into ``stage_dir`` (one flat folder of symlinks).
    """
    if len(paths) == 1:
        path = os.path.abspath(paths[0])
        return path, os.path.isdir(path)
    if os.path.isdir(stage_dir):
        shutil.rmtree(stage_dir)
    os.makedirs(stage_dir)
    files: list[str] = []
    for path in paths:
        files += raw_files_in(path) if os.path.isdir(path) else [path]
    for i, item in enumerate(files):
        os.symlink(os.path.abspath(item),
                   os.path.join(stage_dir, f"{i:05d}_{os.path.basename(item)}"))
    return stage_dir, False
