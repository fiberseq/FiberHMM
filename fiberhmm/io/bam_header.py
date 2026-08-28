"""Helpers for recording FiberHMM provenance in the output BAM header."""
from __future__ import annotations

import re
from typing import Optional

import pysam


# Optional FiberBrowser discovery convention for logical MA annotation names.
# A pysam header dict stores only the text following ``@CO<TAB>``.
MA_TYPES_PREFIX = "MA-TYPES:v1:"
_MA_NAME_RE = re.compile(r"^[A-Za-z0-9_]+$")
_MA_SECTION_HEAD_RE = re.compile(r"^([A-Za-z0-9_]+)[+.-][PQ]*$")

# Stable assay/enzyme/platform contract consumed by FiberBrowser and other
# downstream tools. This is deliberately separate from free-text @PG DS/CL
# provenance so scientific model selection never depends on filenames or
# command-line parsing.
CHEMISTRY_PREFIX = "FIBERHMM-CHEMISTRY:v1:"
_CHEMISTRY_FIELD_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_CHEMISTRY_VALUE_RE = re.compile(r"^[A-Za-z0-9_.+-]+$")
_CHEMISTRY_REQUIRED_FIELDS = ("assay", "enzyme", "platform", "mode")
_LEGACY_MODE_RE = re.compile(r"(?:^|[\s;(])mode=([A-Za-z0-9_.+-]+)", re.IGNORECASE)
_LEGACY_ENZYME_RE = re.compile(r"(?:^|[\s;(])enzyme=([A-Za-z0-9_.+-]+)", re.IGNORECASE)
_LEGACY_SEQ_RE = re.compile(r"(?:^|\s)--seq(?:=|\s+)(pacbio|nanopore)(?:\s|$)", re.IGNORECASE)


def _header_to_dict(header) -> dict:
    """Header -> dict, accepting a pysam AlignmentHeader or a plain dict."""
    if hasattr(header, 'to_dict'):
        return header.to_dict()
    return dict(header)


def append_pg_record(header, record: dict):
    """Return a copy of ``header`` with a ``@PG`` program-group line appended.

    ``record`` supplies ``PN``/``VN``/``CL``/``DS``; the ``ID`` is auto-assigned
    (suffixed on re-runs so it stays unique) and ``PP`` is chained to the last
    existing ``@PG`` so the program history is well-formed.
    """
    d = _header_to_dict(header)
    pgs = list(d.get('PG', []))
    existing_ids = {p.get('ID') for p in pgs}

    base = record.get('PN') or 'fiberhmm'
    pid = base
    i = 1
    while pid in existing_ids:
        i += 1
        pid = f"{base}.{i}"

    pg = {'ID': pid}
    for key in ('PN', 'VN', 'CL', 'DS'):
        val = record.get(key)
        if val:
            pg[key] = str(val)
    if pgs and pgs[-1].get('ID'):
        pg['PP'] = pgs[-1]['ID']

    d['PG'] = pgs + [pg]
    return pysam.AlignmentHeader.from_dict(d)


def maybe_append_pg(header, record: Optional[dict]):
    """``append_pg_record`` when ``record`` is provided, else ``header`` unchanged."""
    if not record:
        return header
    output = append_pg_record(header, record)
    chemistry = record.get("chemistry")
    return append_chemistry(output, chemistry) if chemistry else output


def _parse_chemistry_comment(comment: str) -> Optional[dict[str, str]]:
    if not str(comment).startswith(CHEMISTRY_PREFIX):
        return None
    payload = str(comment)[len(CHEMISTRY_PREFIX):]
    fields: dict[str, str] = {}
    for item in payload.split(";"):
        key, separator, value = item.partition("=")
        if (
            not separator
            or key in fields
            or not _CHEMISTRY_FIELD_RE.fullmatch(key)
            or not _CHEMISTRY_VALUE_RE.fullmatch(value)
        ):
            return None
        fields[key] = value
    if any(not fields.get(key) for key in _CHEMISTRY_REQUIRED_FIELDS):
        return None
    return fields


def declared_chemistries(header) -> list[dict[str, str]]:
    """Return valid v1 chemistry declarations in first-seen order."""
    declarations: list[dict[str, str]] = []
    seen = set()
    for raw_comment in _header_to_dict(header).get("CO", []):
        parsed = _parse_chemistry_comment(str(raw_comment))
        if parsed is None:
            continue
        identity = tuple(sorted(parsed.items()))
        if identity not in seen:
            seen.add(identity)
            declarations.append(parsed)
    return declarations


def infer_legacy_chemistry(header) -> Optional[dict[str, str]]:
    """Recover chemistry from a pre-v1 ``fiberhmm-call`` program record.

    This compatibility path is intentionally narrower than general filename or
    read-content guessing. Its result is inferred provenance, never equivalent
    to an explicit :data:`CHEMISTRY_PREFIX` declaration.
    """
    programs = _header_to_dict(header).get("PG", [])
    for program in reversed(programs):
        program_name = str(program.get("PN") or program.get("ID") or "").lower()
        if "fiberhmm-call" not in program_name:
            continue
        description = str(program.get("DS", ""))
        command = str(program.get("CL", ""))
        joined = f"{description} {command}"
        mode_match = _LEGACY_MODE_RE.search(joined)
        enzyme_match = _LEGACY_ENZYME_RE.search(joined)
        seq_match = _LEGACY_SEQ_RE.search(command)
        mode = mode_match.group(1).lower() if mode_match else ""
        enzyme = enzyme_match.group(1).lower() if enzyme_match else ""
        platform = seq_match.group(1).lower() if seq_match else ""
        if not platform and mode == "pacbio-fiber":
            platform = "pacbio"
        elif not platform and mode == "nanopore-fiber":
            platform = "nanopore"
        if mode == "daf":
            assay = "daf"
        elif mode in {"pacbio-fiber", "nanopore-fiber"}:
            assay = "fiber-seq"
        else:
            assay = "custom"
        if not mode and not enzyme:
            continue
        return {
            "assay": assay,
            "enzyme": enzyme or "custom",
            "platform": platform or "unknown",
            "mode": mode or "custom",
        }
    return None


def append_chemistry(header, chemistry):
    """Append an authoritative ``FIBERHMM-CHEMISTRY:v1`` declaration.

    Existing incompatible declarations are rejected: silently relabelling a
    BAM would be worse than requiring an explicit reprocessing decision.
    Additional safe fields such as ``model`` are permitted for provenance.
    """
    if not isinstance(chemistry, dict):
        raise ValueError("chemistry declaration must be a mapping")
    normalized: dict[str, str] = {}
    for raw_key, raw_value in chemistry.items():
        key = str(raw_key).strip().lower()
        value = str(raw_value).strip()
        if not _CHEMISTRY_FIELD_RE.fullmatch(key):
            raise ValueError(f"invalid chemistry field name: {raw_key!r}")
        if key in normalized:
            raise ValueError(f"duplicate chemistry field after normalization: {raw_key!r}")
        if not _CHEMISTRY_VALUE_RE.fullmatch(value):
            raise ValueError(f"invalid chemistry field value for {key}: {raw_value!r}")
        normalized[key] = value
    missing = [key for key in _CHEMISTRY_REQUIRED_FIELDS if not normalized.get(key)]
    if missing:
        raise ValueError("chemistry declaration missing required fields: " + ",".join(missing))

    existing = declared_chemistries(header)
    if normalized in existing:
        return header
    core = {key: normalized[key].lower() for key in _CHEMISTRY_REQUIRED_FIELDS}
    for declaration in existing:
        prior_core = {
            key: declaration[key].lower() for key in _CHEMISTRY_REQUIRED_FIELDS
        }
        if prior_core != core:
            raise ValueError(
                "incompatible FIBERHMM-CHEMISTRY declarations: "
                f"existing={prior_core}, requested={core}"
            )

    ordered_keys = [*_CHEMISTRY_REQUIRED_FIELDS]
    ordered_keys.extend(sorted(key for key in normalized if key not in ordered_keys))
    comment = CHEMISTRY_PREFIX + ";".join(
        f"{key}={normalized[key]}" for key in ordered_keys
    )
    data = _header_to_dict(header)
    data["CO"] = [*list(data.get("CO", [])), comment]
    return pysam.AlignmentHeader.from_dict(data)


def is_valid_ma_name(name: str) -> bool:
    """Return whether ``name`` satisfies the MA logical-name grammar."""
    return bool(_MA_NAME_RE.fullmatch(str(name)))


def ma_types_from_tag(ma_value) -> list[str]:
    """Discover valid, non-empty logical annotation names in one MA value.

    Invalid sections are skipped independently so one malformed extension does
    not hide otherwise discoverable names. Strand and quality suffixes are
    deliberately discarded.
    """
    names = []
    seen = set()
    for section in str(ma_value).split(";")[1:]:
        head, separator, body = section.partition(":")
        match = _MA_SECTION_HEAD_RE.fullmatch(head)
        if not separator or match is None or not any(body.split(",")):
            continue
        name = match.group(1)
        if name not in seen:
            seen.add(name)
            names.append(name)
    return names


def declared_ma_types(header) -> list[str]:
    """Return valid declared MA names in union/first-seen order.

    The declaration is advisory. Malformed comments, unknown convention
    versions, invalid names, and duplicate names are ignored gracefully.
    Names are case-sensitive and contain neither strand nor quality suffixes.
    """
    d = _header_to_dict(header)
    names: list[str] = []
    seen = set()
    for raw_comment in d.get("CO", []):
        comment = str(raw_comment)
        if not comment.startswith(MA_TYPES_PREFIX):
            continue
        fields = comment[len(MA_TYPES_PREFIX):].split(",")
        # One bad field invalidates this declaration, but not other @CO lines.
        if not fields or any(not is_valid_ma_name(name) for name in fields):
            continue
        for name in fields:
            if name not in seen:
                seen.add(name)
                names.append(name)
    return names


def append_ma_types(header, annotation_names):
    """Append one ``MA-TYPES:v1`` @CO declaration for newly advertised names.

    Existing declarations and unrelated comments are left untouched. Only
    names absent from the valid ordered union are appended, making repeated
    calls idempotent while allowing successive tools to extend the header.
    Invalid producer-supplied names raise ``ValueError`` because they indicate
    a programming error; invalid declarations already present in an input
    header remain untouched and are ignored by :func:`declared_ma_types`.
    """
    if isinstance(annotation_names, str):
        annotation_names = (annotation_names,)

    requested = []
    requested_seen = set()
    for raw_name in annotation_names:
        name = str(raw_name)
        if not is_valid_ma_name(name):
            raise ValueError(f"invalid MA annotation name: {name!r}")
        if name not in requested_seen:
            requested_seen.add(name)
            requested.append(name)

    declared = set(declared_ma_types(header))
    missing = [name for name in requested if name not in declared]
    if not missing:
        return header

    d = _header_to_dict(header)
    comments = list(d.get("CO", []))
    comments.append(MA_TYPES_PREFIX + ",".join(missing))
    d["CO"] = comments
    return pysam.AlignmentHeader.from_dict(d)


# Stable, version-independent token marking that ns/nl/as/al (and MA) are written
# in molecular (original-fiber) coordinates. Downstream consumers (FiberBrowser)
# key off the exact token `coord=molecular`. fiberhmm-call carries it in its @PG
# DS; paths without a full @PG (e.g. fiberhmm-apply) emit it as a @CO comment.
COORD_MOLECULAR_MARKER = "fiberhmm:coord=molecular"


def append_coord_marker(header):
    """Append the molecular-frame @CO marker to ``header`` (idempotent)."""
    d = _header_to_dict(header)
    comments = list(d.get('CO', []))
    if COORD_MOLECULAR_MARKER not in comments:
        comments.append(COORD_MOLECULAR_MARKER)
    d['CO'] = comments
    return pysam.AlignmentHeader.from_dict(d)


def header_has_coord_marker(header) -> bool:
    """Return True if the header records molecular-frame coordinates.

    FiberHMM records this as either ``@CO fiberhmm:coord=molecular`` or a full
    ``@PG`` record whose ``DS`` contains ``coord=molecular``. Its absence means
    the legacy/v1.0 convention: ns/nl are stored in SEQ (query) frame, so a
    second-pass recaller must not flip them again on reverse reads.
    """
    d = _header_to_dict(header)
    if COORD_MOLECULAR_MARKER in list(d.get('CO', [])):
        return True
    return any(
        "coord=molecular" in str(program.get("DS", ""))
        for program in d.get("PG", [])
    )
