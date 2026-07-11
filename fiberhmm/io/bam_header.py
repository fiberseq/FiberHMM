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
    return append_pg_record(header, record) if record else header


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
