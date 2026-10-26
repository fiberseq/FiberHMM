"""bgzip + tabix TSV tables of a scan directory: deterministic writer and region reader.

Every table starts with ``chrom start end idx_start idx_end``. ``idx_start`` /
``idx_end`` are the columns tabix indexes (``tabix -0 -s1 -b4 -e5``); they equal
``start``/``end`` except for zero-width rows, which index ``[p, p + 1)`` so a
fetch at ``p`` finds them. Rows are sorted by contig order (the reference
header's), ``idx_start``, ``idx_end`` and then the remaining cells as text, so
the same rows always give the same bytes.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence

COMMON = ('chrom', 'start', 'end', 'idx_start', 'idx_end')
TBI_MAX = (1 << 29) - 1


def index_path(path) -> Path:
    """The table's index: ``.tbi``, or ``.csi`` for tables with positions beyond 2^29."""
    p = Path(path)
    csi = Path(str(p) + '.csi')
    return csi if csi.is_file() else Path(str(p) + '.tbi')


def cell(value: Any) -> str:
    if value is None:
        return ''
    if isinstance(value, bool):
        return 'True' if value else 'False'
    if isinstance(value, float):
        return '' if (math.isnan(value) or math.isinf(value)) else repr(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, (list, tuple)):
        return ','.join(cell(v) for v in value)
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True, separators=(',', ':'))
    try:
        import numpy as np
        if isinstance(value, np.generic):
            return cell(value.item())
    except ImportError:  # pragma: no cover
        pass
    text = str(value)
    if '\t' in text or '\n' in text:
        raise ValueError(f'table cell contains a tab or newline: {text[:40]!r}')
    return text


def index_span(start: int, end: int):
    start, end = int(start), int(end)
    return (start, end) if end > start else (start, start + 1)


class TableWriter:
    """Collects rows of one table and publishes it sorted, BGZF-compressed and tabix-indexed."""

    def __init__(self, path, columns: Sequence[str], contig_order: Mapping[str, int]):
        self.path = Path(path)
        self.columns = list(columns)
        if tuple(self.columns[:5]) != COMMON:
            raise ValueError(f'{self.path.name}: a scan table must start with {COMMON}')
        self.order = dict(contig_order)
        self._rows: List[tuple] = []

    def add(self, chrom: str, start: int, end: int, values: Sequence[Any]) -> None:
        if len(values) != len(self.columns) - 5:
            raise ValueError(f'{self.path.name}: {len(values)} cells for {len(self.columns) - 5} columns')
        if chrom not in self.order:
            raise ValueError(f'{self.path.name}: contig {chrom!r} is not in the reference')
        a, b = index_span(start, end)
        self._rows.append((self.order[chrom], a, b, chrom, int(start), int(end),
                           tuple(cell(v) for v in values)))

    def __len__(self) -> int:
        return len(self._rows)

    def close(self) -> dict:
        """Write ``path`` and ``path.tbi``; returns ``{"rows", "sha256"}`` of the compressed table."""
        import pysam
        self._rows.sort(key=lambda r: (r[0], r[1], r[2], r[6]))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, raw = tempfile.mkstemp(prefix='.' + self.path.name + '.', suffix='.tsv', dir=self.path.parent)
        try:
            with os.fdopen(fd, 'w', newline='') as handle:
                handle.write('#' + '\t'.join(self.columns) + '\n')
                for _o, a, b, chrom, start, end, cells in self._rows:
                    handle.write('\t'.join((chrom, str(start), str(end), str(a), str(b), *cells)) + '\n')
            tmp_gz = raw + '.bgz'
            pysam.tabix_compress(raw, tmp_gz, force=True)
            # TBI addresses positions below 2^29; a table reaching further gets a CSI index instead.
            csi = any(r[2] > TBI_MAX for r in self._rows)
            pysam.tabix_index(tmp_gz, seq_col=0, start_col=3, end_col=4, zerobased=True,
                              meta_char='#', force=True, csi=csi)
            suffix = '.csi' if csi else '.tbi'
            for other in ('.tbi', '.csi'):
                if other != suffix and os.path.exists(str(self.path) + other):
                    os.unlink(str(self.path) + other)
            os.replace(tmp_gz, self.path)
            os.replace(tmp_gz + suffix, str(self.path) + suffix)
        finally:
            for leftover in (raw, raw + '.bgz', raw + '.bgz.tbi', raw + '.bgz.csi'):
                if os.path.exists(leftover):
                    os.unlink(leftover)
        n = len(self._rows)
        self._rows = []
        return dict(rows=n, sha256=sha256_file(self.path))


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


class Table:
    """Region reader over one scan table."""

    def __init__(self, path):
        import pysam
        self.path = Path(path)
        self._tabix = pysam.TabixFile(str(self.path), index=str(index_path(self.path)))
        header = list(self._tabix.header)
        if not header or not header[-1].startswith('#'):
            raise ValueError(f'{self.path}: missing header line')
        self.columns = header[-1][1:].split('\t')

    @property
    def contigs(self) -> List[str]:
        return list(self._tabix.contigs)

    def fetch(self, chrom: str, start: Optional[int] = None, end: Optional[int] = None) -> Iterator[Dict[str, str]]:
        if chrom not in self._tabix.contigs:
            return iter(())
        it = self._tabix.fetch(chrom, start, end) if start is not None else self._tabix.fetch(chrom)
        return (dict(zip(self.columns, line.split('\t'))) for line in it)

    def rows(self) -> Iterator[Dict[str, str]]:
        for chrom in self._tabix.contigs:
            yield from self.fetch(chrom)

    def close(self) -> None:
        self._tabix.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def read_all(path) -> List[Dict[str, str]]:
    """Every row of a table, in file order (header via the gzip stream, no index needed)."""
    import gzip
    with gzip.open(path, 'rt') as handle:
        lines = handle.read().splitlines()
    if not lines or not lines[0].startswith('#'):
        raise ValueError(f'{path}: missing header line')
    cols = lines[0][1:].split('\t')
    return [dict(zip(cols, line.split('\t'))) for line in lines[1:]]
