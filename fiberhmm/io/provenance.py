"""Basecaller provenance: which basecaller and models made a BAM's reads.

:func:`basecaller_provenance` reads it from a BAM header (and, optionally,
the first reads) and returns a plain JSON-able dict::

    {
      "available": True,             # anything known at all
      "platform": "nanopore",        # "nanopore" | "pacbio" | None
      "program": "dorado",           # basecaller program, or None
      "version": "2.0.1",            # its version, or None
      "basecall_model": "dna_r10.4.1_e8.2_400bps_sup@v5.2.0",   # or None
      "modbase_models": ["dna_r10.4.1_e8.2_400bps_sup@v5.2.0_6mA@v1"],
      "modbase_caller": "dorado",    # program that wrote MM/ML (PacBio: jasmine,
                                     # primrose, ft predict-m6a), or None
      "modbase_caller_version": "2.0.1",
      "instrument_basecaller_version": None,  # PacBio @RG DS BASECALLERVERSION
      "command": "dorado basecaller ...",     # the basecaller's @PG CL, or None
      "sources": {"program": "pg", "basecall_model": "rg_ds", ...},
      "mixed": [],                   # fields with several values (joined by ",")
      "notes": [],                   # human-readable remarks
    }

``modbase_models`` distinguishes **unknown** (``None``) from **known none**
(``[]``: dorado ran without a modification model, e.g. DAF-seq basecalling).
Dorado writes ``modbase_models=`` into ``@RG DS`` only when it used one, so an
``@RG DS`` that names a ``basecall_model`` and no ``modbase_models`` is
known-none; so is a dorado ``@PG CL`` without ``--modified-bases``/
``--modified-bases-models`` or a ``model,mods`` complex.

Precedence, per field (the source code recorded in ``sources``):

1. ``override``  -- an explicit ``--basecaller-info``/``--modbase-model``;
2. ``recorded-override`` -- an override an earlier FiberHMM run recorded in
   its ``@PG DS`` (``basecaller_sources=...:override``);
3. ``rg_ds``     -- ``@RG DS`` (``basecall_model=``, ``modbase_models=``;
   PacBio ``BASECALLERVERSION=``);
4. ``pg`` / ``pg_cl`` -- the basecaller's ``@PG`` (``PN``/``VN``), and its
   ``CL`` (dorado's model positional, ``--modified-bases-models``,
   ``--modified-bases``, a ``sup,6mA`` complex);
5. ``read_rg``   -- the per-read ``RG:Z:<runid>_<model>[_<modbase>]`` suffix;
6. ``recorded``  -- other values an earlier FiberHMM run recorded (the last
   fallback: a header merged since may legitimately say more).

:func:`ds_tokens` renders the result as the space-separated ``key=value``
tokens FiberHMM writes into its ``@PG DS`` (``fiberhmm-pipeline``,
``fiberhmm-call``); :func:`parse_ds_tokens` reads them back.
"""
from __future__ import annotations

import os
import re
import shlex
from typing import Iterable, Optional

ONT_BASECALLERS = ("dorado", "guppy", "guppy_basecaller", "bonito", "minknow")
PACBIO_BASECALLERS = ("ccs", "pbccs")
PACBIO_MODBASE_CALLERS = ("jasmine", "primrose", "ft", "fibertools", "fibertools-rs")

FIELDS = ("program", "version", "basecall_model", "modbase_models")
SOURCE_RANK = {"override": 0, "recorded-override": 1, "rg_ds": 2, "pg": 3,
               "pg_cl": 3, "read_rg": 4, "recorded": 5}

UNKNOWN = "unknown"
NONE = "none"

# FiberHMM @PG DS tokens (see ds_tokens).
DS_KEYS = {"program": "basecaller", "version": "basecaller_version",
           "basecall_model": "basecall_model", "modbase_models": "modbase_models"}
DS_SOURCES_KEY = "basecaller_sources"

# dorado per-read read groups: <runid>_<basecall model>[_<modbase suffix>][_<barcode>]
_MODEL_RE = re.compile(
    r"(?P<model>(?:dna|rna\d*)_[A-Za-z0-9.]+(?:_[A-Za-z0-9.]+)*?_(?:fast|hac|sup)"
    r"@v\d+(?:\.\d+)*)(?:_(?P<rest>.+))?$")
_MODBASE_SUFFIX_RE = re.compile(r"^(?P<mods>.*?@v\d+(?:\.\d+)*)")
# dorado options that take a value (so the positionals can be found in a CL).
_DORADO_VALUE_OPTIONS = {
    "-x", "--device", "--models-directory", "-l", "--read-ids", "-n", "--max-reads",
    "--resume-from", "--min-qscore", "-o", "--output-dir", "--reference", "--bed-file",
    "--mm2-opts", "--modified-bases-models", "--modified-bases-threshold",
    "--modified-bases-batchsize", "--kit-name", "--sample-sheet",
    "--barcode-arrangement", "--barcode-sequences", "--primer-sequences", "--trim",
    "--poly-a-config", "-b", "--batchsize", "-c", "--chunksize", "-k", "--overlap",
    "--recover-from", "--stereo-model", "--pair",
}
_OVERRIDE_KEYS = {
    "program": "program", "basecaller": "program",
    "version": "version", "basecaller_version": "version",
    "basecall_model": "basecall_model", "model": "basecall_model",
    "modbase_models": "modbase_models", "modbase_model": "modbase_models",
}


class OverrideError(ValueError):
    """A malformed ``--basecaller-info`` / ``--modbase-model`` value."""


# ---------------------------------------------------------------------------
# Overrides
# ---------------------------------------------------------------------------

def _split_models(value) -> list[str]:
    if value is None:
        return []
    items = value if isinstance(value, (list, tuple)) else [value]
    out: list[str] = []
    for item in items:
        for part in str(item).split(","):
            part = part.strip()
            if part and part not in out:
                out.append(part)
    return out


def parse_override(basecaller_info: Optional[str] = None,
                   modbase_model=None) -> Optional[dict]:
    """The override dict from ``--basecaller-info`` and ``--modbase-model``.

    ``basecaller_info`` is ``key=value`` pairs separated by spaces or ``;``
    (keys: ``program``, ``version``, ``basecall_model``, ``modbase_models``;
    aliases ``basecaller``, ``basecaller_version``, ``model``,
    ``modbase_model``). ``modbase_model`` is one name, a comma list or a list
    (repeatable flag); ``none`` states that no modification model was used.
    Returns None when neither is given.
    """
    out: dict = {}
    if basecaller_info:
        for token in re.split(r"[\s;]+", str(basecaller_info).strip()):
            if not token:
                continue
            key, sep, value = token.partition("=")
            field = _OVERRIDE_KEYS.get(key.strip().lower())
            if not sep or field is None or not value.strip():
                raise OverrideError(
                    f"--basecaller-info: cannot read {token!r}; give key=value pairs, "
                    "e.g. \"program=dorado version=0.9.6 "
                    "basecall_model=dna_r10.4.1_e8.2_400bps_sup@v5.0.0 "
                    "modbase_models=dna_r10.4.1_e8.2_400bps_sup@v5.0.0_6mA@v2\"")
            if field == "modbase_models":
                out[field] = _modbase_value(value)
            else:
                out[field] = value.strip()
    models = _split_models(modbase_model)
    if models:
        out["modbase_models"] = _modbase_value(",".join(models))
    return out or None


def _modbase_value(text: str) -> list[str]:
    models = _split_models(text)
    if [m.lower() for m in models] == [NONE]:
        return []
    if any(m.lower() in (NONE, UNKNOWN) for m in models):
        raise OverrideError(f"modbase models: {text!r} mixes model names with "
                            "'none'/'unknown'")
    return models


# ---------------------------------------------------------------------------
# Header sources
# ---------------------------------------------------------------------------

def _header_dict(header) -> dict:
    if header is None:
        return {}
    if hasattr(header, "to_dict"):
        return header.to_dict()
    return dict(header)


def _ds_pairs(text: str) -> dict:
    pairs = {}
    for token in re.split(r"[\s;]+", str(text or "")):
        key, sep, value = token.partition("=")
        if sep and key:
            pairs.setdefault(key, value)
    return pairs


def _model_name(text: str) -> str:
    """A model given as a path is recorded by its directory name."""
    text = str(text).rstrip("/\\")
    return os.path.basename(text) or text


def parse_dorado_command(command: str) -> dict:
    """``{"basecall_model", "modbase_models"}`` from a dorado ``@PG CL``.

    ``modbase_models`` is ``[]`` when the command used none, and holds
    short codes (``6mA``) for ``--modified-bases`` or a ``sup,6mA`` complex.
    Unknown keys are absent.
    """
    try:
        argv = shlex.split(str(command or ""))
    except ValueError:
        argv = str(command or "").split()
    out: dict = {}
    sub = next((i for i, a in enumerate(argv) if a in ("basecaller", "duplex")), None)
    if sub is None:
        return out
    positionals: list[str] = []
    mods: Optional[list[str]] = None
    i = sub + 1
    while i < len(argv):
        arg = argv[i]
        name, eq, inline = arg.partition("=")
        if arg == "--modified-bases":
            i += 1
            codes = []
            while i < len(argv) and not argv[i].startswith("-"):
                codes.append(argv[i])
                i += 1
            mods = (mods or []) + codes
            continue
        if name == "--modified-bases" and eq:
            mods = (mods or []) + inline.split()
        elif name == "--modified-bases-models":
            value = inline if eq else (argv[i + 1] if i + 1 < len(argv) else "")
            if not eq:
                i += 1
            mods = (mods or []) + [_model_name(m) for m in _split_models(value)]
        elif arg.startswith("-"):
            if not eq and name in _DORADO_VALUE_OPTIONS:
                i += 1
        else:
            positionals.append(arg)
        i += 1
    if positionals:
        model = positionals[0]
        if "," in model and os.path.sep not in model:
            # model complex: "sup@v5.2.0,6mA" / "hac,5mCG_5hmCG"
            head, *complex_mods = [p.strip() for p in model.split(",")]
            out["basecall_model"] = head
            mods = (mods or []) + [m for m in complex_mods if m]
        else:
            out["basecall_model"] = _model_name(model)
    out["modbase_models"] = list(dict.fromkeys(mods or []))
    return out


def _dorado_subcommand(command: str) -> Optional[str]:
    """The dorado subcommand of a ``@PG CL`` (``basecaller``, ``aligner``...)."""
    for token in str(command or "").split()[1:]:
        if not token.startswith("-"):
            return token if re.fullmatch(r"[a-z][a-z_-]*", token) else None
    return None


def parse_read_group_id(rg: str) -> dict:
    """``{"basecall_model", "modbase_models"?}`` from a dorado per-read read
    group (``<runid>_<model>[_<modbase>]``); ``{}`` when it is not one.

    Per-read read groups of recent dorado versions omit the modification
    model, so ``modbase_models`` is absent (unknown) unless the suffix names
    one.
    """
    text = str(rg or "")
    _, sep, tail = text.partition("_")
    if not sep:
        return {}
    match = _MODEL_RE.search(tail)
    if not match:
        return {}
    out = {"basecall_model": match.group("model")}
    rest = match.group("rest") or ""
    mods = _MODBASE_SUFFIX_RE.match(rest)
    if mods:
        out["modbase_models"] = [f"{match.group('model')}_{mods.group('mods')}"]
    return out


def parse_ds_tokens(ds: str) -> dict:
    """The basecaller fields FiberHMM recorded in a ``@PG DS`` (``{}`` if none).

    Returns ``{"values": {field: value}, "sources": {field: source}}``;
    ``unknown`` values are left out, ``modbase_models=none`` is ``[]``.
    """
    pairs = _ds_pairs(ds)
    if DS_KEYS["program"] not in pairs:
        return {}
    values: dict = {}
    for field, key in DS_KEYS.items():
        value = pairs.get(key)
        if value is None or value == UNKNOWN:
            continue
        if field == "modbase_models":
            values[field] = [] if value == NONE else _split_models(value)
        else:
            values[field] = value
    sources = {}
    for item in str(pairs.get(DS_SOURCES_KEY, "")).split(","):
        field, sep, source = item.partition(":")
        if sep and field in FIELDS:
            sources[field] = source
    return {"values": values, "sources": sources}


def _program_name(pg: dict) -> str:
    return str(pg.get("PN") or pg.get("ID") or "").strip().lower()


def _is_fiberhmm(pg: dict) -> bool:
    return _program_name(pg).startswith("fiberhmm")


def _is_ft_m6a(pg: dict) -> bool:
    name = _program_name(pg)
    command = str(pg.get("CL", ""))
    return name in ("ft", "fibertools", "fibertools-rs") and (
        "m6a" in command.lower() or "predict" in command.lower() or not command)


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

class _Field:
    """Candidate values for one field: the best-ranked source wins."""

    def __init__(self):
        self.rank = None
        self.source = None
        self.values: list = []

    def offer(self, value, source: str) -> None:
        if value is None:
            return
        rank = SOURCE_RANK[source]
        if self.rank is None or rank < self.rank:
            self.rank, self.source, self.values = rank, source, []
        if rank == self.rank and value not in self.values:
            self.values.append(value)


def _read_group_ids(reads) -> list[str]:
    ids: list[str] = []
    for read in reads or ():
        if isinstance(read, str):
            value = read
        else:
            try:
                value = read.get_tag("RG") if read.has_tag("RG") else None
            except (AttributeError, KeyError):
                value = None
        if value and value not in ids:
            ids.append(str(value))
    return ids


def basecaller_provenance(header, reads: Optional[Iterable] = None,
                          override: Optional[dict] = None,
                          max_reads: int = 200) -> dict:
    """Basecaller provenance of a BAM (see the module docstring).

    ``header``: a pysam header or header dict (``None``/``{}`` for FASTQ).
    ``reads``: optional reads (pysam records, or ``RG`` strings) whose
    per-read read groups are the last header-independent source; at most
    ``max_reads`` are read. ``override``: a dict from :func:`parse_override`.
    """
    data = _header_dict(header)
    fields = {name: _Field() for name in FIELDS}
    extra = {"modbase_caller": None, "modbase_caller_version": None,
             "instrument_basecaller_version": None, "command": None}
    notes: list[str] = []
    platform_votes: set[str] = set()

    for name, value in (override or {}).items():
        if name in fields:
            fields[name].offer(_freeze(value), "override")

    programs = [pg for pg in data.get("PG", []) or [] if isinstance(pg, dict)]
    # FiberHMM's own records, newest last: the newest one with tokens counts.
    recorded = None
    for pg in programs:
        if _is_fiberhmm(pg):
            parsed = parse_ds_tokens(pg.get("DS", ""))
            if parsed:
                recorded = parsed
    if recorded:
        for name, value in recorded["values"].items():
            # An override stays one through any number of later FiberHMM runs.
            source = ("recorded-override"
                      if recorded["sources"].get(name) in ("override", "recorded-override")
                      else "recorded")
            fields[name].offer(_freeze(value), source)

    for group in data.get("RG", []) or []:
        platform = str(group.get("PL", "")).upper()
        if platform in ("ONT", "NANOPORE", "OXFORD_NANOPORE"):
            platform_votes.add("nanopore")
        elif platform in ("PACBIO", "PACBIO_SMRT"):
            platform_votes.add("pacbio")
        pairs = _ds_pairs(group.get("DS", ""))
        if pairs.get("basecall_model"):
            fields["basecall_model"].offer(pairs["basecall_model"], "rg_ds")
            fields["modbase_models"].offer(
                _freeze(_split_models(pairs.get("modbase_models"))), "rg_ds")
        elif pairs.get("modbase_models"):
            fields["modbase_models"].offer(_freeze(_split_models(pairs["modbase_models"])),
                                           "rg_ds")
        if pairs.get("BASECALLERVERSION"):
            extra["instrument_basecaller_version"] = pairs["BASECALLERVERSION"]
            platform_votes.add("pacbio")

    for pg in programs:
        if _is_fiberhmm(pg):
            continue
        name = _program_name(pg)
        base = name.split(".", 1)[0]
        if base in ONT_BASECALLERS:
            platform_votes.add("nanopore")
            command = str(pg.get("CL", "") or "")
            parsed = parse_dorado_command(command) if base == "dorado" else {}
            if base == "dorado" and command and not parsed and _dorado_subcommand(command):
                # dorado aligner / demux / trim: the same program, not the basecaller
                continue
            fields["program"].offer(base, "pg")
            fields["version"].offer(pg.get("VN") or None, "pg")
            if command:
                extra["command"] = extra["command"] or command
            if base == "dorado":
                if parsed.get("basecall_model"):
                    fields["basecall_model"].offer(parsed["basecall_model"], "pg_cl")
                if parsed:
                    fields["modbase_models"].offer(_freeze(parsed.get("modbase_models", [])),
                                                   "pg_cl")
                extra["modbase_caller"] = extra["modbase_caller"] or "dorado"
                extra["modbase_caller_version"] = (extra["modbase_caller_version"]
                                                   or pg.get("VN"))
        elif base in PACBIO_BASECALLERS:
            platform_votes.add("pacbio")
            fields["program"].offer("ccs", "pg")
            fields["version"].offer(pg.get("VN") or None, "pg")
            extra["command"] = extra["command"] or (pg.get("CL") or None)
        elif base in ("jasmine", "primrose") or _is_ft_m6a(pg):
            platform_votes.add("pacbio")
            caller = base if base in ("jasmine", "primrose") else "ft predict-m6a"
            extra["modbase_caller"] = caller
            extra["modbase_caller_version"] = pg.get("VN") or None

    rg_ids = _read_group_ids(_take(reads, max_reads))
    for rg in rg_ids:
        parsed = parse_read_group_id(rg)
        if parsed.get("basecall_model"):
            fields["basecall_model"].offer(parsed["basecall_model"], "read_rg")
        if parsed.get("modbase_models"):
            fields["modbase_models"].offer(_freeze(parsed["modbase_models"]), "read_rg")

    result: dict = {"available": False, "platform": None}
    sources: dict = {}
    mixed: list[str] = []
    for name in FIELDS:
        field = fields[name]
        if field.source is None:
            result[name] = None
            continue
        sources[name] = field.source
        if name == "modbase_models":
            sets = [list(v) for v in field.values]
            union: list[str] = []
            for models in sets:
                union += [m for m in models if m not in union]
            result[name] = union
            if len({tuple(s) for s in sets}) > 1:
                mixed.append(name)
        else:
            result[name] = ",".join(str(v) for v in field.values)
            if len(field.values) > 1:
                mixed.append(name)
    result.update(extra)
    if result["modbase_caller"] is None and result["program"] == "dorado":
        result["modbase_caller"] = "dorado"
    model = result.get("basecall_model") or ""
    if re.match(r"(?:dna|rna\d*)_r\d", model):
        platform_votes.add("nanopore")
    program = str(result.get("program") or "").lower()
    if program in ONT_BASECALLERS:
        platform_votes.add("nanopore")
    elif program in PACBIO_BASECALLERS:
        platform_votes.add("pacbio")
    if len(platform_votes) == 1:
        result["platform"] = next(iter(platform_votes))
    elif len(platform_votes) > 1:
        notes.append("the header names both Nanopore and PacBio programs")
    result["available"] = any(result[name] is not None for name in FIELDS) or any(
        extra[key] for key in ("modbase_caller", "instrument_basecaller_version"))
    if mixed:
        notes.append("the reads come from several basecalling setups ("
                     + ", ".join(f"{n}: {_display(result[n])}" for n in mixed) + ")")
    if not result["available"]:
        notes.append("no basecaller provenance (no basecaller @PG, @RG DS or "
                     "per-read read group)")
    elif result["platform"] == "nanopore" and result["modbase_models"] is None:
        notes.append("the modification (modbase) model is not recorded")
    result["sources"] = sources
    result["mixed"] = mixed
    result["notes"] = notes
    return result


def _take(items, limit):
    if items is None:
        return
    for i, item in enumerate(items):
        if i >= limit:
            break
        yield item


def _freeze(value):
    return tuple(value) if isinstance(value, list) else value


def _display(value) -> str:
    if value is None:
        return UNKNOWN
    if isinstance(value, (list, tuple)):
        return ",".join(value) if value else NONE
    return str(value)


def _token(value) -> str:
    return re.sub(r"\s+", "_", _display(value)) or UNKNOWN


def ds_tokens(prov: Optional[dict]) -> str:
    """``basecaller=... basecaller_version=... basecall_model=...
    modbase_models=... basecaller_sources=field:source,...`` for a ``@PG DS``.

    Unknown values are ``unknown``; ``modbase_models=none`` means known-none.
    """
    prov = prov or {}
    parts = [f"{DS_KEYS[name]}={_token(prov.get(name))}" for name in FIELDS]
    sources = prov.get("sources") or {}
    parts.append(f"{DS_SOURCES_KEY}="
                 + (",".join(f"{k}:{sources[k]}" for k in FIELDS if k in sources) or NONE))
    return " ".join(parts)


def describe(prov: Optional[dict]) -> str:
    """One line for logs: ``dorado 2.0.1, model ..., modbase ...``."""
    prov = prov or {}
    if not prov.get("available"):
        return "unknown"
    head = " ".join(str(prov[k]) for k in ("program", "version") if prov.get(k)) or "basecaller"
    parts = [head]
    if prov.get("basecall_model"):
        parts.append(f"model {prov['basecall_model']}")
    if prov.get("modbase_models") is not None:
        parts.append(f"modbase {_display(prov['modbase_models'])}")
    elif prov.get("modbase_caller") and prov.get("modbase_caller") != prov.get("program"):
        parts.append("modbase caller " + " ".join(
            str(prov[k]) for k in ("modbase_caller", "modbase_caller_version") if prov.get(k)))
    sources = sorted(set((prov.get("sources") or {}).values()))
    return ", ".join(parts) + (f" (from {', '.join(sources)})" if sources else "")


def missing_note(prov: Optional[dict], *, flag_hint: str = "--basecaller-info / "
                 "--modbase-model") -> Optional[str]:
    """A note to print when provenance (or the modbase model) is missing."""
    prov = prov or {}
    if not prov.get("available"):
        return ("basecaller provenance is not recorded in the input (FASTQ, or a BAM "
                "without the basecaller's @PG/@RG lines); pass " + flag_hint
                + " to record it")
    if prov.get("platform") == "nanopore" and prov.get("modbase_models") is None:
        return ("the input does not record which modification (modbase) model "
                "called its MM/ML tags; pass --modbase-model to record it")
    return None


# ---------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------

def fastq_read_groups(path: str, records: int = 200) -> list[str]:
    """``RG:Z:`` values in the header comments of a FASTQ's first records
    (dorado ``--emit-fastq`` writes them)."""
    import gzip
    opener = gzip.open if str(path).lower().endswith(".gz") else open
    found: list[str] = []
    with opener(path, "rb") as handle:
        for i, line in enumerate(handle):
            if i // 4 >= records:
                break
            if i % 4:
                continue
            for token in line.decode("utf-8", "replace").split()[1:]:
                if token.startswith("RG:Z:") and token[5:] not in found:
                    found.append(token[5:])
    return found


def bam_provenance(path: str, override: Optional[dict] = None,
                   max_reads: int = 200) -> dict:
    """:func:`basecaller_provenance` of a BAM file (header + first reads)."""
    import pysam
    with pysam.AlignmentFile(path, check_sq=False) as bam:
        header = bam.header.to_dict()
        reads = []
        for read in bam.fetch(until_eof=True):
            reads.append(read.get_tag("RG") if read.has_tag("RG") else None)
            if len(reads) >= max_reads:
                break
    return basecaller_provenance(header, [r for r in reads if r], override, max_reads)
