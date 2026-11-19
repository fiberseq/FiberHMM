"""Identities of a scan: content ids, canonical dataset order, molecules, records and hits.

See the scan format reference (``docs/reference/scan-format.md``) for the
definitions these functions implement. Everything here is deterministic and
independent of paths, labels and input order.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

CONTENT_ID_PREFIX = 'cid:v1:'
MOLECULE_PREFIX = 'mol:v1 '
_MIB = 1 << 20


def _json_default(value):
    try:
        import numpy as np
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
    except ImportError:  # pragma: no cover
        pass
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(f'not JSON-serialisable: {type(value).__name__}')


def canonical_json(value: Any) -> str:
    """Sorted keys, compact separators, floats as Python repr (JSON default)."""
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False, default=_json_default)


def digest(value: Any, n: int = 64) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()[:n]


def settings_hash(settings: Mapping[str, Any]) -> str:
    return digest(settings, 16)


def _index_path(path: Path):
    for candidate in (Path(str(path) + '.csi'), Path(str(path) + '.bai'), path.with_suffix('.bai')):
        if candidate.is_file():
            return candidate
    return None


def content_id(path) -> str:
    """``cid:v1:<sha256>`` of a BAM: its size and its first and last MiB.

    The BAM header lies in the first MiB and the BGZF end-of-file block in the
    last. No timestamp enters it, so copies of one file share an id and a
    rerun that writes the same bytes gets the same id. FiberBrowser computes
    the same value when it registers a dataset, which is how a scan binds to a
    loaded file."""
    p = Path(path)
    size = p.stat().st_size
    h = hashlib.sha256()
    h.update(b'cid:v1\n')
    h.update(str(size).encode() + b'\n')
    with open(p, 'rb') as handle:
        h.update(handle.read(min(size, _MIB)))
        handle.seek(max(0, size - _MIB))
        h.update(handle.read(min(size, _MIB)))
    return CONTENT_ID_PREFIX + h.hexdigest()


def canonical_order(content_ids_per_dataset: Sequence[Sequence[str]]) -> list:
    """Ordinal of each dataset: its rank when datasets are sorted by their files' content ids.

    Ties (the same files given twice) keep input order."""
    order = sorted(range(len(content_ids_per_dataset)),
                   key=lambda i: (tuple(content_ids_per_dataset[i]), i))
    ordinal = [0] * len(order)
    for rank, i in enumerate(order):
        ordinal[i] = rank
    return ordinal


def molecule_key(name: str) -> str:
    """The physical-molecule key of one read name: ``pacbio/<movie>/<zmw>`` or ``read/<name>``.

    The same normalisation FiberHMM uses to reject a PacBio molecule supplied
    twice (``consensus.harmonized_families.workflow.prepare_sources``)."""
    name = str(name or '')
    pieces = name.split('/')
    if len(pieces) >= 3 and pieces[1].isdigit():
        return f'pacbio/{pieces[0]}/{pieces[1]}'
    return f'read/{name}'


def molecule_id(names: Iterable[str]) -> str:
    """``mol:v1 <16 hex>`` of the class holding ``names``: the digest of the minimum key."""
    keys = sorted({molecule_key(n) for n in names if n})
    if not keys:
        return ''
    return MOLECULE_PREFIX + hashlib.sha256(('mol:v1\n' + keys[0]).encode()).hexdigest()[:16]


def hit_id(module: str, type_id: str, chrom: str, anchor: int, start: int, end: int, key: str,
           unit_id: str, module_version: str, settings_digest: str) -> str:
    """``<module>:<type suffix>:<chrom>:<anchor>:<8 hex>``."""
    suffix = type_id.split('.', 1)[1] if type_id.startswith(module + '.') else type_id
    tail = digest([type_id, int(start), int(end), str(key), unit_id, module_version, settings_digest], 8)
    return f'{module}:{suffix}:{chrom}:{int(anchor)}:{tail}'


def contigs_sha256(contigs: Sequence[Sequence[Any]]) -> str:
    return digest([[str(name), int(length)] for name, length in contigs])


def record_sha256(read) -> str:
    """SHA-256 of a pysam record's SAM text: FiberHMM's and FiberBrowser's record identity."""
    return hashlib.sha256(read.to_string().encode('utf-8')).hexdigest()


def unit_dir_name(unit_id: str) -> str:
    """A filesystem-safe, injective directory name for a unit id (readable part + digest of the full id)."""
    return f'{safe_name(unit_id)}-{hashlib.sha256(unit_id.encode()).hexdigest()[:12]}'


def safe_name(text: str) -> str:
    """A filesystem-safe rendering of a key (not injective: use :func:`unit_dir_name` for unit ids)."""
    out = []
    for ch in str(text):
        out.append(ch if ch.isalnum() or ch in '-_.' else '_')
    return ''.join(out)


def file_size(path) -> int:
    return os.stat(path).st_size
