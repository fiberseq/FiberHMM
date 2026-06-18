"""Helpers for recording FiberHMM provenance in the output BAM header."""
from __future__ import annotations

from typing import Optional

import pysam


def _header_to_dict(header) -> dict:
    """Header -> dict, accepting a pysam AlignmentHeader or a plain dict."""
    if hasattr(header, 'to_dict'):
        return header.to_dict()
    return dict(header)


def _next_pg_id(pgs, base: str) -> str:
    existing_ids = {pg.get('ID') for pg in pgs}
    pid = base
    i = 1
    while pid in existing_ids:
        i += 1
        pid = f"{base}.{i}"
    return pid


def _record_text(record: dict, key: str) -> Optional[str]:
    value = record.get(key)
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _pg_fields_from_record(record: dict) -> dict:
    fields = {}
    for key in ('PN', 'VN', 'CL', 'DS'):
        value = _record_text(record, key)
        if value is not None:
            fields[key] = value
    return fields


def _last_pg_id(pgs) -> Optional[str]:
    if pgs and pgs[-1].get('ID'):
        return pgs[-1]['ID']
    return None


def _new_pg_record(pgs, record: dict) -> dict:
    base = _record_text(record, 'PN') or 'fiberhmm'
    pg = {'ID': _next_pg_id(pgs, base)}
    pg.update(_pg_fields_from_record(record))

    previous_id = _last_pg_id(pgs)
    if previous_id:
        pg['PP'] = previous_id
    return pg


def append_pg_record(header, record: dict):
    """Return a copy of ``header`` with a ``@PG`` program-group line appended.

    ``record`` supplies ``PN``/``VN``/``CL``/``DS``; the ``ID`` is auto-assigned
    (suffixed on re-runs so it stays unique) and ``PP`` is chained to the last
    existing ``@PG`` so the program history is well-formed.
    """
    d = _header_to_dict(header)
    pgs = list(d.get('PG', []))
    d['PG'] = pgs + [_new_pg_record(pgs, record)]
    return pysam.AlignmentHeader.from_dict(d)


def maybe_append_pg(header, record: Optional[dict]):
    """``append_pg_record`` when ``record`` is provided, else ``header`` unchanged."""
    return append_pg_record(header, record) if record else header


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
