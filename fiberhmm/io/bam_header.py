"""Helpers for recording FiberHMM provenance in the output BAM header."""
from __future__ import annotations

from typing import Optional

import pysam


def append_pg_record(header, record: dict):
    """Return a copy of ``header`` with a ``@PG`` program-group line appended.

    ``record`` supplies ``PN``/``VN``/``CL``/``DS``; the ``ID`` is auto-assigned
    (suffixed on re-runs so it stays unique) and ``PP`` is chained to the last
    existing ``@PG`` so the program history is well-formed.
    """
    d = header.to_dict()
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
